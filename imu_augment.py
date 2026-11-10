#!/usr/bin/env python3
"""
imu_augment.py - Step 3: domain randomization for simulated IMUs + EMG proxy.

Turns the clean MuJoCo IMU signals into realistic MPU-6050-like readings. A new
random corruption is drawn every time corrupt() is called, so the training script
re-draws it every epoch and the network never sees the same noise twice.

Per IMU, per chunk of data:
    mounting error     small random rotation of the sensor axes (default up to 10 deg)
    scale error        per-axis gain error (default +/-3 %)
    gyro bias          initial offset + random walk + warm-up (temperature) drift
    accel bias         constant offset + slow random walk
    white noise        Gaussian, different level per chunk
    quantization       MPU-6050 LSB (8192 LSB/g, 65.5 LSB/(deg/s)) and saturation (4 g, 500 deg/s)
    sample repeats     occasional duplicated sample (USB hiccup)
EMG proxy (MyoWare-like envelope, 0..1):
    driven by |actuator torque| of shoulder + elbow, LEADING the motion by a random
    50-150 ms electromechanical delay, low-pass ~5 Hz, baseline + gain + noise.

Output channel order (13): acc_upper xyz, gyr_upper xyz, acc_fore xyz, gyr_fore xyz, emg
Units: g, rad/s, emg 0..1.   True gyro bias (6): upper xyz, fore xyz in rad/s.

Preview what it does:
    python imu_augment.py data/sim_imu.npz            saves data/augment_preview.png
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import numpy as np

ACC_LSB_PER_G = 8192.0
GYRO_LSB_PER_RAD = 65.5 * 180.0 / math.pi
ACC_MAX_G = 4.0
GYRO_MAX_RAD = 500.0 * math.pi / 180.0
CHANNELS = ["ax_u", "ay_u", "az_u", "gx_u", "gy_u", "gz_u",
            "ax_f", "ay_f", "az_f", "gx_f", "gy_f", "gz_f", "emg"]


@dataclass
class AugmentConfig:
    mount_jitter_deg: float = 10.0
    scale_error: float = 0.03
    acc_noise_g: tuple = (0.002, 0.012)            # range of white-noise std, drawn per chunk
    gyr_noise_rad: tuple = (0.0005, 0.004)
    gyr_bias0_rad: float = 0.05                    # std of the initial gyro bias (~3 deg/s)
    gyr_bias_rw: tuple = (1e-4, 1e-3)              # random-walk strength, rad/s per sqrt(s)
    gyr_warmup_rad: float = 0.03                   # std of the warm-up drift amplitude
    gyr_warmup_tau_s: tuple = (60.0, 600.0)        # warm-up time constant range
    acc_bias_g: float = 0.02
    acc_bias_rw: float = 2e-4                      # g per sqrt(s)
    repeat_prob: float = 0.002
    quantize: bool = True
    emg_mode: str = "torque"                       # "torque" or "zero"
    emg_lead_ms: tuple = (50.0, 150.0)
    emg_cutoff_hz: float = 5.0
    emg_gain_counts: tuple = (800.0, 2500.0)
    emg_baseline_counts: tuple = (30.0, 120.0)
    emg_noise_counts: float = 20.0


def random_rotation(rng: np.random.Generator, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = math.radians(rng.uniform(0.0, max_deg))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def lowpass(x: np.ndarray, rate: float, fc: float) -> np.ndarray:
    a = 1.0 - math.exp(-2 * math.pi * fc / rate)
    y = np.empty_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def torque_scale(tau: np.ndarray) -> np.ndarray:
    """Robust per-joint torque scale (95th percentile of |tau|) for normalising the EMG drive."""
    return np.maximum(np.percentile(np.abs(tau), 95, axis=0), 1e-6)


def corrupt(acc: np.ndarray, gyr: np.ndarray, tau: np.ndarray, rate: float,
            rng: np.random.Generator, cfg: AugmentConfig = AugmentConfig(),
            tau_scale: np.ndarray | None = None):
    """acc, gyr: (N, 2, 3) clean, in g and rad/s. tau: (N, 2). Returns X (N, 13), bias (N, 6)."""
    N = acc.shape[0]
    dt = 1.0 / rate
    t = np.arange(N) * dt
    acc_out = np.empty_like(acc)
    gyr_out = np.empty_like(gyr)
    bias_out = np.empty((N, 2, 3))

    for i in range(2):
        R = random_rotation(rng, cfg.mount_jitter_deg)           # sensor axes in the link-site frame
        a = acc[:, i] @ R
        w = gyr[:, i] @ R
        a = a * (1 + rng.uniform(-cfg.scale_error, cfg.scale_error, 3))
        w = w * (1 + rng.uniform(-cfg.scale_error, cfg.scale_error, 3))

        # gyro bias = initial + random walk + warm-up drift (chunk may start mid-warm-up)
        b0 = rng.normal(0, cfg.gyr_bias0_rad, 3)
        rw = rng.uniform(*cfg.gyr_bias_rw)
        walk = np.cumsum(rng.normal(0, rw * math.sqrt(dt), (N, 3)), axis=0)
        tau_w = rng.uniform(*cfg.gyr_warmup_tau_s)
        t_on = rng.uniform(0, 600.0)
        warm = rng.normal(0, cfg.gyr_warmup_rad, 3) * (1 - np.exp(-(t + t_on) / tau_w))[:, None]
        bg = b0 + walk + warm

        ba = rng.normal(0, cfg.acc_bias_g, 3) + np.cumsum(
            rng.normal(0, cfg.acc_bias_rw * math.sqrt(dt), (N, 3)), axis=0)

        a = a + ba + rng.normal(0, rng.uniform(*cfg.acc_noise_g), (N, 3))
        w = w + bg + rng.normal(0, rng.uniform(*cfg.gyr_noise_rad), (N, 3))

        if cfg.quantize:
            a = np.round(np.clip(a, -ACC_MAX_G, ACC_MAX_G) * ACC_LSB_PER_G) / ACC_LSB_PER_G
            w = np.round(np.clip(w, -GYRO_MAX_RAD, GYRO_MAX_RAD) * GYRO_LSB_PER_RAD) / GYRO_LSB_PER_RAD
        acc_out[:, i], gyr_out[:, i], bias_out[:, i] = a, w, bg

    # EMG proxy
    if cfg.emg_mode == "zero":
        emg = np.zeros(N)
    else:
        sc = torque_scale(tau) if tau_scale is None else tau_scale
        drive = (np.abs(tau) / sc).sum(axis=1) / 2.0
        lead = int(round(rng.uniform(*cfg.emg_lead_ms) * 1e-3 * rate))
        if lead > 0:
            drive = np.concatenate([drive[lead:], np.repeat(drive[-1:], lead)])   # EMG precedes torque
        env = lowpass(np.clip(drive, 0, 3), rate, cfg.emg_cutoff_hz)
        counts = (rng.uniform(*cfg.emg_baseline_counts)
                  + rng.uniform(*cfg.emg_gain_counts) * env * (1 + 0.1 * rng.normal(size=N))
                  + rng.normal(0, cfg.emg_noise_counts, N))
        emg = np.clip(counts, 0, 4095) / 4095.0

    X = np.concatenate([acc_out[:, 0], gyr_out[:, 0], acc_out[:, 1], gyr_out[:, 1], emg[:, None]], axis=1)
    bias = bias_out.reshape(N, 6)

    # occasional repeated sample (serial hiccup): repeat the input, keep the truth
    rep = np.nonzero(rng.random(N) < cfg.repeat_prob)[0]
    rep = rep[rep > 0]
    X[rep] = X[rep - 1]
    return X.astype(np.float32), bias.astype(np.float32)


def _preview(path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(path)
    rate = 1.0 / np.median(np.diff(d["t"]))
    n = int(min(len(d["t"]), 15 * rate))
    X, b = corrupt(d["acc"][:n], d["gyr"][:n], d["tau"][:n], rate, np.random.default_rng(0),
                   tau_scale=torque_scale(d["tau"]))
    t = d["t"][:n] - d["t"][0]
    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(t, d["acc"][:n, 1, 0], label="clean"); ax[0].plot(t, X[:, 6], lw=0.6, label="corrupted")
    ax[0].set_ylabel("forearm acc x (g)"); ax[0].legend()
    ax[1].plot(t, d["gyr"][:n, 1, 2], label="clean"); ax[1].plot(t, X[:, 11], lw=0.6, label="corrupted")
    ax[1].plot(t, b[:, 5], "k--", label="true bias"); ax[1].set_ylabel("forearm gyro z (rad/s)"); ax[1].legend()
    ax[2].plot(t, np.degrees(d["q"][:n])); ax[2].set_ylabel("shoulder, elbow (deg)")
    ax[3].plot(t, X[:, 12]); ax[3].set_ylabel("EMG proxy (0..1)"); ax[3].set_xlabel("time (s)")
    out = path.replace(".npz", "_augment_preview.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    _preview(sys.argv[1] if len(sys.argv) > 1 else "data/sim_imu.npz")
