#!/usr/bin/env python3
"""
train_tcn.py - Step 4: train a causal TCN on simulated IMU windows.

Input : 200 ms window of corrupted IMU + EMG, tensor [B, W, 13]
        (W = 40 at 200 Hz; channels from imu_augment.CHANNELS; units g, rad/s, 0..1)
Output: [B, 10] in physical units
        theta_shoulder, theta_elbow (rad)  - joint angles at the LAST sample (causal)
        omega_shoulder, omega_elbow (rad/s)
        gyro bias upper xyz, fore xyz (rad/s)

Loss  : L = MSE(theta) + lambda_bias * MSE(bias) + lambda_omega * MSE(omega)
        Every target is standardised first, so the lambdas weigh the terms fairly
        (bias is ~0.05 rad/s, angles ~1 rad - raw MSE would ignore the bias).

Data  : data/sim_imu.npz from sim_imu_dataset.py. It is cut into 60 s chunks and
        split 80/10/10 into train/val/test BY CHUNK, so no window of the test set
        overlaps training data. Training noise is re-drawn every epoch.

Saves : models/model.pth   (weights + normalisation + config, for PyTorch)
        models/model.onnx  (same network incl. normalisation, raw units in/out)
        models/report.json (test errors overall and per motion type)
        models/test_plot.png

Usage:
    python train_tcn.py                         defaults (25 epochs)
    python train_tcn.py --epochs 5              quick run
    python train_tcn.py --emg-mode zero         train without the EMG proxy channel

Needs: pip install torch numpy matplotlib   (+ pip install onnx for a checked ONNX export)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imu_augment import AugmentConfig, CHANNELS, corrupt, torque_scale   # noqa: E402

MOTIONS = ("reach", "circle", "flick", "wave", "hold", "multisine")
OUTPUTS = ["theta_shoulder", "theta_elbow", "omega_shoulder", "omega_elbow",
           "bias_gx_u", "bias_gy_u", "bias_gz_u", "bias_gx_f", "bias_gy_f", "bias_gz_f"]


# =============================================================================
# Model
# =============================================================================
class CausalBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, C, W]
        y = self.drop(F.relu(self.conv1(F.pad(x, (self.pad, 0)))))
        y = self.drop(F.relu(self.conv2(F.pad(y, (self.pad, 0)))))
        return F.relu(y + self.res(x))


class IMUTCN(nn.Module):
    """Causal TCN. Normalisation is stored inside, so it takes and returns raw units."""

    def __init__(self, n_in: int = 13, n_out: int = 10, hidden: int = 64,
                 dilations=(1, 2, 4, 8), kernel: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers, c = [], n_in
        for d in dilations:
            layers.append(CausalBlock(c, hidden, kernel, d, dropout))
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.register_buffer("x_mean", torch.zeros(n_in))
        self.register_buffer("x_std", torch.ones(n_in))
        self.register_buffer("y_mean", torch.zeros(n_out))
        self.register_buffer("y_std", torch.ones(n_out))
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)

    def forward_normalised(self, x: torch.Tensor) -> torch.Tensor:   # x raw [B, W, C] -> y normalised
        x = (x - self.x_mean) / self.x_std
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # raw in, raw out
        return self.forward_normalised(x) * self.y_std + self.y_mean


# =============================================================================
# Data
# =============================================================================
class WindowData:
    """Chunks of one split. corrupt() is applied per chunk; windows end at every sample."""

    def __init__(self, d: dict, chunks: list, rate: float, window: int, cfg: AugmentConfig,
                 tau_sc: np.ndarray) -> None:
        self.d, self.chunks, self.rate, self.W = d, chunks, rate, window
        self.cfg, self.tau_sc = cfg, tau_sc
        self.X = self.Y = self.M = self.ends = None

    def draw(self, rng: np.random.Generator) -> None:
        Xs, Ys, Ms, ends, off = [], [], [], [], 0
        for a, b in self.chunks:
            X, bias = corrupt(self.d["acc"][a:b], self.d["gyr"][a:b], self.d["tau"][a:b],
                              self.rate, rng, self.cfg, self.tau_sc)
            Y = np.concatenate([self.d["q"][a:b], self.d["qd"][a:b], bias], axis=1).astype(np.float32)
            Xs.append(X); Ys.append(Y); Ms.append(self.d["motion"][a:b])
            local = np.arange(self.W - 1, b - a)
            if "contact" in self.d:
                # skip windows that contain a contact, or the 0.25 s settling time after one
                bad = self.d["contact"][a:b].astype(bool)
                guard = int(0.25 * self.rate)
                if guard and bad.any():
                    spread = np.convolve(bad.astype(int), np.ones(guard + 1, int))[:len(bad)]
                    bad = spread > 0                        # contact at i also marks i+1..i+guard
                cs = np.concatenate([[0], np.cumsum(bad)])
                local = local[cs[local + 1] - cs[local + 1 - self.W] == 0]
            ends.append(off + local)
            off += b - a
        self.X, self.Y = np.concatenate(Xs), np.concatenate(Ys)
        self.M, self.ends = np.concatenate(Ms), np.concatenate(ends)
        self.total_windows = sum(b - a - self.W + 1 for a, b in self.chunks)

    def batch(self, end_idx: np.ndarray):
        idx = end_idx[:, None] - (self.W - 1) + np.arange(self.W)[None, :]
        return self.X[idx], self.Y[end_idx], self.M[end_idx]


def make_chunks(d: dict, rate: float, chunk_s: float, window: int):
    """Contiguous chunks that never cross an episode reset."""
    n = len(d["t"])
    L = int(chunk_s * rate)
    chunks, start = [], 0
    ep = d["episode"]
    while start < n:
        stop = min(start + L, n)
        brk = np.nonzero(ep[start:stop] != ep[start])[0]
        if len(brk):
            stop = start + int(brk[0])
        if stop - start >= window * 4:
            chunks.append((start, stop))
        start = stop
    return chunks


def split_chunks(chunks: list, rng: np.random.Generator):
    order = rng.permutation(len(chunks))
    n_val = max(1, round(0.1 * len(chunks)))
    n_test = max(1, round(0.1 * len(chunks)))
    test = [chunks[i] for i in sorted(order[:n_test])]
    val = [chunks[i] for i in sorted(order[n_test:n_test + n_val])]
    train = [chunks[i] for i in sorted(order[n_test + n_val:])]
    if not train:
        raise SystemExit("Not enough data for a train/val/test split - generate more minutes.")
    return train, val, test


# =============================================================================
# Train / evaluate
# =============================================================================
def loss_fn(pred_n: torch.Tensor, y_n: torch.Tensor, lam_b: float, lam_w: float):
    l_theta = F.mse_loss(pred_n[:, 0:2], y_n[:, 0:2])
    l_omega = F.mse_loss(pred_n[:, 2:4], y_n[:, 2:4])
    l_bias = F.mse_loss(pred_n[:, 4:10], y_n[:, 4:10])
    return l_theta + lam_b * l_bias + lam_w * l_omega, (l_theta.item(), l_omega.item(), l_bias.item())


@torch.no_grad()
def predict(model: IMUTCN, data: WindowData, device, stride: int = 1, bs: int = 4096):
    model.eval()
    ends = data.ends[::stride]
    P, Y, M = [], [], []
    for i in range(0, len(ends), bs):
        x, y, m = data.batch(ends[i:i + bs])
        P.append(model(torch.from_numpy(x).to(device)).cpu().numpy())
        Y.append(y); M.append(m)
    return np.concatenate(P), np.concatenate(Y), np.concatenate(M)


def metrics(P: np.ndarray, Y: np.ndarray) -> dict:
    deg = 180.0 / math.pi
    err = P - Y
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    return {
        "theta_shoulder_rmse_deg": rmse(err[:, 0]) * deg,
        "theta_elbow_rmse_deg": rmse(err[:, 1]) * deg,
        "omega_shoulder_rmse_degps": rmse(err[:, 2]) * deg,
        "omega_elbow_rmse_degps": rmse(err[:, 3]) * deg,
        "gyro_bias_rmse_degps": rmse(err[:, 4:10]) * deg,
        "gyro_bias_rmse_if_predict_zero_degps": rmse(Y[:, 4:10]) * deg,
        "n_windows": int(len(Y)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Train the IMU TCN on simulated data.")
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--data", default=os.path.join(here, "data", "sim_imu.npz"))
    p.add_argument("--out", default=os.path.join(here, "models"))
    p.add_argument("--window-ms", type=float, default=200.0)
    p.add_argument("--chunk-s", type=float, default=60.0)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--windows-per-epoch", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lambda-bias", type=float, default=1.0)
    p.add_argument("--lambda-omega", type=float, default=0.2)
    p.add_argument("--emg-mode", choices=["torque", "zero"], default="torque")
    p.add_argument("--mount-jitter", type=float, default=10.0,
                   help="max random IMU mounting error in degrees. 10 = uncalibrated straps; "
                        "~3 = what is left after the hang/home calibration in imu_teleop.py")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is available")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw = np.load(args.data)
    d = {k: raw[k] for k in raw.files}
    rate = float(1.0 / np.median(np.diff(d["t"])))
    W = max(4, int(round(args.window_ms * 1e-3 * rate)))
    cfg = AugmentConfig(emg_mode=args.emg_mode, mount_jitter_deg=args.mount_jitter)
    tau_sc = torque_scale(d["tau"])
    chunks = make_chunks(d, rate, args.chunk_s, W)
    tr_c, va_c, te_c = split_chunks(chunks, rng)
    print(f"data: {len(d['t'])} samples at {rate:.0f} Hz | window {W} samples ({args.window_ms:.0f} ms) "
          f"| chunks train/val/test = {len(tr_c)}/{len(va_c)}/{len(te_c)} | device {device}")

    train = WindowData(d, tr_c, rate, W, cfg, tau_sc)
    val = WindowData(d, va_c, rate, W, cfg, tau_sc)
    test = WindowData(d, te_c, rate, W, cfg, tau_sc)
    val.draw(np.random.default_rng(1001))            # fixed corruption for val/test
    test.draw(np.random.default_rng(2002))
    train.draw(rng)
    if "contact" in d:
        print(f"contact filter: kept {len(train.ends)}/{train.total_windows} training windows "
              f"({100 * (1 - len(train.ends) / max(train.total_windows, 1)):.1f}% skipped: arm touching "
              f"something, or < 0.25 s after)")
    else:
        print("note: this dataset has no contact flags - regenerate it with the latest "
              "sim_imu_dataset.py to skip collision moments")

    # grow the dilation stack until the TCN sees the whole window (RF = 1 + 2*(k-1)*sum(d), k = 3)
    dilations = [1, 2, 4, 8]
    while 1 + 4 * sum(dilations) < W:
        dilations.append(dilations[-1] * 2)
    model = IMUTCN(n_in=len(CHANNELS), n_out=len(OUTPUTS), hidden=args.hidden,
                   dilations=tuple(dilations)).to(device)
    model.x_mean.copy_(torch.from_numpy(train.X.mean(0)))
    model.x_std.copy_(torch.from_numpy(train.X.std(0) + 1e-6))
    model.y_mean.copy_(torch.from_numpy(train.Y.mean(0)))
    model.y_std.copy_(torch.from_numpy(train.Y.std(0) + 1e-6))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: causal TCN, {n_params/1e3:.1f}k parameters, receptive field {model.receptive_field} samples")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, args.windows_per_epoch // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    os.makedirs(args.out, exist_ok=True)
    best, best_state = float("inf"), None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if epoch > 1:
            train.draw(rng)                          # fresh noise, bias drift and EMG every epoch
        model.train()
        parts = np.zeros(3)
        for _ in range(steps_per_epoch):
            ends = train.ends[rng.integers(0, len(train.ends), args.batch)]
            x, y, _ = train.batch(ends)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            y_n = (y - model.y_mean) / model.y_std
            loss, part = loss_fn(model.forward_normalised(x), y_n, args.lambda_bias, args.lambda_omega)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            parts += part
        parts /= steps_per_epoch
        Pv, Yv, _ = predict(model, val, device, stride=4)
        mv = metrics(Pv, Yv)
        score = mv["theta_shoulder_rmse_deg"] + mv["theta_elbow_rmse_deg"] + mv["gyro_bias_rmse_degps"]
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}/{args.epochs} | train mse theta {parts[0]:.4f} omega {parts[1]:.4f} "
              f"bias {parts[2]:.4f} | val: shoulder {mv['theta_shoulder_rmse_deg']:5.2f} deg, "
              f"elbow {mv['theta_elbow_rmse_deg']:5.2f} deg, bias {mv['gyro_bias_rmse_degps']:5.2f} deg/s "
              f"| {time.time() - t0:4.1f} s")

    model.load_state_dict(best_state)
    model.to("cpu")
    Pt, Yt, Mt = predict(model, test, "cpu")
    report = {"overall": metrics(Pt, Yt), "per_motion": {}}
    for i, name in enumerate(MOTIONS):
        sel = Mt == i
        if sel.sum() > 50:
            report["per_motion"][name] = metrics(Pt[sel], Yt[sel])
    report["config"] = dict(vars(args), rate=rate, window=W, channels=CHANNELS, outputs=OUTPUTS,
                            params=n_params, receptive_field=model.receptive_field,
                            dilations=list(dilations))

    o = report["overall"]
    print("\nTEST (unseen chunks, unseen noise)")
    print(f"  shoulder angle RMSE {o['theta_shoulder_rmse_deg']:.2f} deg | elbow {o['theta_elbow_rmse_deg']:.2f} deg")
    print(f"  shoulder speed RMSE {o['omega_shoulder_rmse_degps']:.1f} deg/s | elbow {o['omega_elbow_rmse_degps']:.1f} deg/s")
    print(f"  gyro bias RMSE {o['gyro_bias_rmse_degps']:.2f} deg/s "
          f"(predicting zero would give {o['gyro_bias_rmse_if_predict_zero_degps']:.2f})")
    for name, m in report["per_motion"].items():
        print(f"    {name:9s} shoulder {m['theta_shoulder_rmse_deg']:5.2f}  elbow {m['theta_elbow_rmse_deg']:5.2f}  "
              f"bias {m['gyro_bias_rmse_degps']:5.2f}")

    torch.save({"state_dict": model.state_dict(), "config": report["config"],
                "channels": CHANNELS, "outputs": OUTPUTS}, os.path.join(args.out, "model.pth"))
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    onnx_path = os.path.join(args.out, "model.onnx")
    dummy = torch.zeros(1, W, len(CHANNELS))
    kw = dict(input_names=["imu_window"], output_names=["pred"],
              dynamic_axes={"imu_window": {0: "batch"}, "pred": {0: "batch"}}, opset_version=17)
    try:
        try:
            torch.onnx.export(model, dummy, onnx_path, dynamo=False, **kw)
        except TypeError:                            # older torch without the dynamo argument
            torch.onnx.export(model, dummy, onnx_path, **kw)
        print(f"saved {onnx_path}")
    except Exception as exc:
        print(f"ONNX export skipped ({exc}). model.pth is saved; for ONNX run: pip install onnx")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(len(Pt), int(30 * rate))
        tt = np.arange(n) / rate
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for j, nm in enumerate(("shoulder", "elbow")):
            ax[j].plot(tt, np.degrees(Yt[:n, j]), label="ground truth")
            ax[j].plot(tt, np.degrees(Pt[:n, j]), lw=0.8, label="TCN")
            ax[j].set_ylabel(f"{nm} (deg)"); ax[j].legend(loc="upper right")
        ax[2].plot(tt, np.degrees(Yt[:n, 9]), label="true bias")
        ax[2].plot(tt, np.degrees(Pt[:n, 9]), lw=0.8, label="TCN bias")
        ax[2].set_ylabel("forearm gyro z bias (deg/s)"); ax[2].set_xlabel("time (s)"); ax[2].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "test_plot.png"), dpi=110)
        print(f"saved {os.path.join(args.out, 'test_plot.png')}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")
    print(f"saved {os.path.join(args.out, 'model.pth')} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
