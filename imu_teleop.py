#!/usr/bin/env python3
"""
imu_teleop.py - Drive the cycloidal-arm MuJoCo twin from the wearable
(ESP32 + 2 IMUs + MyoWare EMG).

    ESP32 (esp32_imu_emg_streamer.ino) --USB serial--> sensor fusion + calibration
        --> joint targets --> sim_comms.ArmSimulator --> MuJoCo viewer

Put this file in the same folder as sim_comms.py, scene.xml, cycloidal_arm.xml.

What drives what
----------------
    IMU A on the UPPER ARM  -> base yaw  (sweeping the arm left/right)
                            -> shoulder  (raising the upper arm forward)
    IMU B on the FOREARM    -> elbow     (forearm angle relative to the upper arm)
                            -> wrist roll (twisting the forearm, palm up/down)
    EMG on forearm flexors  -> gripper   (make a fist = close, relax = open)
    wrist pitch             -> held at home (add a 3rd IMU on the hand later)

Mounting does not need to be exact: the calibration finds each IMU's limb axis
by itself. Shoulder/elbow are measured in the vertical plane of the arm, the
same plane the robot's shoulder and elbow move in.

Usage
-----
    python imu_teleop.py                    auto-detect the ESP32, calibrate, run the sim
    python imu_teleop.py --port COM5        pick the serial port yourself
    python imu_teleop.py --check            calibrate and print live angles, no MuJoCo
    python imu_teleop.py --fake             synthetic wearable data, no hardware needed
    python imu_teleop.py --no-base          keep the base still (avoids slow yaw drift)
    python imu_teleop.py --list-ports       show serial ports and exit

While running, type a letter in THIS terminal and press Enter:
    c = re-zero the home pose     e = recalibrate EMG     r = full recalibration
    b = toggle base on/off        q = quit

Every session is logged to logs/session_*.csv (raw IMU + EMG + computed
angles) so the data can be used later for training the learned estimator.

Needs: pip install mujoco pyserial numpy pygame   (pygame because sim_comms imports it)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

log = logging.getLogger("imu_teleop")

# MPU-6050 scale factors (must match the ESP32 sketch)
ACC_LSB_PER_G = 8192.0          # +/-4 g
GYRO_LSB_PER_DPS = 65.5         # +/-500 deg/s
DEG = math.pi / 180.0


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class TeleopConfig:
    """Mapping from human angles to robot joint targets. Edit here.

    Each robot target = home + sign * gain * (human angle change since the home pose).
    If a joint moves the wrong way in the viewer, flip its sign.
    Human sign conventions (see --check): base + = arm swings to your LEFT,
    shoulder + = upper arm raised forward, elbow + = forearm raised relative to the
    upper arm, roll + = forearm twisted (check which way on your mount).
    """

    base_sign: float = +1.0
    shoulder_sign: float = -1.0     # sim_comms: negative shoulder ctrl swings the arm forward
    elbow_sign: float = -1.0        # sim_comms: negative elbow ctrl swings the arm forward
    roll_sign: float = +1.0
    base_gain: float = 1.0
    shoulder_gain: float = 1.0
    elbow_gain: float = 1.0
    roll_gain: float = 1.0
    use_base: bool = True

    smoothing_tau: float = 0.08     # s, low-pass on joint targets (removes tremor)
    max_joint_rate: float = 2.5     # rad/s, safety rate limit on every joint target
    stale_after: float = 0.25       # s without sensor data -> hold the arm still

    # Gripper (EMG)
    grip_on_frac: float = 0.35      # close when EMG > rest + 35 % of (max - rest)
    grip_off_frac: float = 0.15     # open when EMG < rest + 15 % of (max - rest)
    grip_on_samples: int = 3        # 30 ms above threshold to close
    grip_off_samples: int = 8       # 80 ms below threshold to open (resists accidental drops)
    gripper_close_is_low: bool = True   # sim_comms: the "open" button increases gripper ctrl
    gripper_speed: float = 0.06     # gripper ctrl units (m) per second

    # Madgwick filter gain (higher = trusts the accelerometer more)
    madgwick_beta: float = 0.05


# =============================================================================
# Quaternion helpers  (q = [w, x, y, z], R(q) maps sensor frame -> world, z up)
# =============================================================================
def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def q_from_accel(acc: np.ndarray) -> np.ndarray:
    """Orientation with zero yaw whose gravity direction matches the accelerometer."""
    a = acc / (np.linalg.norm(acc) + 1e-12)
    roll = math.atan2(a[1], a[2])
    pitch = math.atan2(-a[0], math.hypot(a[1], a[2]))
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    return np.array([cr * cp, sr * cp, cr * sp, -sr * sp])   # qy(pitch) * qx(roll)


def twist_about_z(q: np.ndarray) -> float:
    """Rotation angle about the world vertical contained in q (swing-twist split)."""
    return wrap(2.0 * math.atan2(q[3], q[0]))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot_z(angle: float, v: np.ndarray) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class Madgwick:
    """6-axis Madgwick AHRS (gyro + accelerometer). Yaw is gyro-only and drifts slowly."""

    def __init__(self, beta: float) -> None:
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, acc: np.ndarray) -> None:
        self.q = q_from_accel(acc)

    def update(self, gyr: np.ndarray, acc: np.ndarray, dt: float) -> np.ndarray:
        q0, q1, q2, q3 = self.q
        gx, gy, gz = gyr
        qdot = 0.5 * np.array([
            -q1 * gx - q2 * gy - q3 * gz,
            q0 * gx + q2 * gz - q3 * gy,
            q0 * gy - q1 * gz + q3 * gx,
            q0 * gz + q1 * gy - q2 * gx,
        ])
        n = np.linalg.norm(acc)
        if n > 1e-6:
            ax, ay, az = acc / n
            f = np.array([
                2 * (q1 * q3 - q0 * q2) - ax,
                2 * (q0 * q1 + q2 * q3) - ay,
                2 * (0.5 - q1 * q1 - q2 * q2) - az,
            ])
            J = np.array([
                [-2 * q2, 2 * q3, -2 * q0, 2 * q1],
                [2 * q1, 2 * q0, 2 * q3, 2 * q2],
                [0.0, -4 * q1, -4 * q2, 0.0],
            ])
            step = J.T @ f
            sn = np.linalg.norm(step)
            if sn > 1e-12:
                qdot -= self.beta * step / sn
        q = self.q + qdot * dt
        self.q = q / np.linalg.norm(q)
        return self.q


# =============================================================================
# Wearable data sources
# =============================================================================
@dataclass
class Sample:
    t: float                 # seconds (sensor clock)
    acc: np.ndarray          # (2, 3) in g      row 0 = upper arm, row 1 = forearm
    gyr: np.ndarray          # (2, 3) in rad/s
    emg: float               # 0..4095
    flags: int = 0           # bit0 upper-arm IMU failed, bit1 forearm IMU failed


def list_ports() -> list:
    from serial.tools import list_ports as lp
    return list(lp.comports())


def find_esp32_port() -> Optional[str]:
    ports = list_ports()
    keys = ("cp210", "ch340", "ch910", "ch343", "silicon labs", "usb-serial", "usb serial",
            "esp32", "uart", "jtag")
    hits = [p.device for p in ports
            if any(k in f"{p.description} {p.manufacturer or ''}".lower() for k in keys)]
    if len(hits) == 1:
        return hits[0]
    if len(ports) == 1:
        return ports[0].device
    return None


class SerialWearable:
    """Reads the ESP32 CSV stream."""

    def __init__(self, port: str, baud: int) -> None:
        import serial
        self.port, self.baud = port, baud
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)                      # opening the port resets most ESP32 boards
        self.ser.reset_input_buffer()
        self._t_prev_us: Optional[int] = None
        self._t = 0.0

    def samples(self, stop: threading.Event) -> Iterator[Sample]:
        bad = 0
        while not stop.is_set():
            raw = self.ser.readline()
            if not raw:
                continue
            line = raw.decode("ascii", "ignore").strip()
            if not line:
                continue
            if line.startswith("#"):
                log.info("ESP32: %s", line[1:].strip())
                continue
            parts = line.split(",")
            if parts[0] != "D" or len(parts) != 16:
                bad += 1
                if bad % 50 == 1:
                    log.warning("Unexpected line from ESP32: %r", line[:80])
                continue
            try:
                t_us = int(parts[1])
                vals = np.array([float(v) for v in parts[2:14]]).reshape(2, 6)
                emg = float(parts[14])
                flags = int(parts[15])
            except ValueError:
                continue
            if self._t_prev_us is not None:
                self._t += ((t_us - self._t_prev_us) & 0xFFFFFFFF) * 1e-6
            self._t_prev_us = t_us
            yield Sample(self._t, vals[:, :3] / ACC_LSB_PER_G,
                         vals[:, 3:] / GYRO_LSB_PER_DPS * DEG, emg, flags)

    def close(self) -> None:
        self.ser.close()


class FakeWearable:
    """Synthetic arm + IMUs + EMG, for testing without hardware.

    Follows the calibration prompts (hang / squeeze / home) and then moves the arm.
    IMUs are mounted at random rotations with gyro bias and noise, to exercise the
    automatic limb-axis calibration.
    """

    HOME = dict(base=0.0, shoulder=0.35, elbow=1.2, roll=0.0)   # rad, human home pose
    HANG = dict(base=0.0, shoulder=0.0, elbow=0.0, roll=0.0)

    def __init__(self, phase: Callable[[], str], realtime: bool = True, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.phase = phase
        self.realtime = realtime
        self.mount = [self._random_rotation(rng) for _ in range(2)]
        self.bias = rng.normal(0, 0.02, (2, 3))        # rad/s
        self.rng = rng
        self.dt = 0.01
        self.truth: dict = dict(self.HANG)
        self._pose = dict(self.HANG)
        self._t_free: Optional[float] = None

    @staticmethod
    def _random_rotation(rng) -> np.ndarray:
        q = rng.normal(size=4)
        return q_to_R(q / np.linalg.norm(q))

    @staticmethod
    def segment_rotations(p: dict) -> tuple[np.ndarray, np.ndarray]:
        """World rotations of the upper-arm and forearm segments. Limb axis = -z when hanging;
        x world = forward, y world = left, z = up."""
        def Rz(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        def Ry(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        Ru = Rz(p["base"]) @ Ry(-p["shoulder"])
        Rf = Rz(p["base"]) @ Ry(-(p["shoulder"] + p["elbow"])) @ Rz(-p["roll"])
        return Ru, Rf

    def _target(self, t: float) -> tuple[dict, float]:
        ph = self.phase()
        if ph in ("still", "relax"):
            return self.HANG, 60.0
        if ph == "squeeze":
            return self.HANG, 2600.0
        if ph == "home":
            return self.HOME, 60.0
        if self._t_free is None:
            self._t_free = t
        s = t - self._t_free
        pose = dict(
            base=0.5 * math.sin(0.4 * s),
            shoulder=self.HOME["shoulder"] + 0.6 * math.sin(0.55 * s),
            elbow=self.HOME["elbow"] + 0.5 * math.sin(0.8 * s + 1.0),
            roll=0.7 * math.sin(0.65 * s + 0.3),
        )
        emg = 2400.0 if math.sin(0.5 * s) > 0.3 else 70.0
        return pose, emg

    def samples(self, stop: threading.Event) -> Iterator[Sample]:
        t = 0.0
        emg_level = 60.0
        Ru, Rf = self.segment_rotations(self._pose)
        prev = [Ru @ self.mount[0], Rf @ self.mount[1]]
        start = time.monotonic()
        while not stop.is_set():
            t += self.dt
            target, emg_target = self._target(t)
            # move smoothly towards the target pose (~0.3 s time constant)
            k = 1 - math.exp(-self.dt / 0.3)
            for key in self._pose:
                self._pose[key] += k * (target[key] - self._pose[key])
            emg_level += (1 - math.exp(-self.dt / 0.05)) * (emg_target - emg_level)
            self.truth = dict(self._pose, grip=emg_target > 1000)
            Ru, Rf = self.segment_rotations(self._pose)
            cur = [Ru @ self.mount[0], Rf @ self.mount[1]]
            acc = np.zeros((2, 3))
            gyr = np.zeros((2, 3))
            for i in range(2):
                acc[i] = cur[i].T @ np.array([0.0, 0.0, 1.0]) + self.rng.normal(0, 0.004, 3)
                dR = prev[i].T @ cur[i]
                ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
                if ang < 1e-9:
                    w = np.zeros(3)
                else:
                    w = ang / (2 * math.sin(ang)) * np.array(
                        [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
                gyr[i] = w / self.dt + self.bias[i] + self.rng.normal(0, 0.003, 3)
            prev = cur
            emg = emg_level + self.rng.normal(0, 15)
            yield Sample(t, acc, gyr, float(max(0.0, emg)), 0)
            if self.realtime:
                lag = start + t - time.monotonic()
                if lag > 0:
                    time.sleep(lag)

    def close(self) -> None:
        pass


# =============================================================================
# Tracker: fusion + calibration + human joint angles (runs at sensor rate)
# =============================================================================
@dataclass
class Angles:
    base: float
    shoulder: float
    elbow: float
    roll: float


class Tracker:
    def __init__(self, cfg: TeleopConfig) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.filters = [Madgwick(cfg.madgwick_beta), Madgwick(cfg.madgwick_beta)]
        self.bias = np.zeros((2, 3))
        self.limb_axis = [np.array([1.0, 0, 0]), np.array([1.0, 0, 0])]   # sensor frame
        self.ref_axis = [np.array([0, 1.0, 0]), np.array([0, 1.0, 0])]    # sensor frame, _|_ limb
        self.have_axes = False
        self.home_q: Optional[list] = None
        self.home_h: list = [np.array([1.0, 0, 0]), np.array([1.0, 0, 0])]
        self.home_angles: Optional[tuple] = None
        self.emg_rest: Optional[float] = None
        self.emg_max: Optional[float] = None
        self.emg_env = 0.0
        self.grip = False
        self._grip_count = 0
        self._last_roll = 0.0
        self._t_prev: Optional[float] = None
        self._collect: Optional[str] = None
        self._buf: list = []
        self.angles: Optional[Angles] = None
        self.last_sample_wall = 0.0
        self.last_flags = 0
        self.samples_seen = 0
        self.warnings: list = []

    # ---- calibration collection ---------------------------------------------
    def begin(self, kind: str) -> None:
        with self.lock:
            self._collect = kind
            self._buf = []

    def end(self, kind: str) -> bool:
        with self.lock:
            buf, self._collect = self._buf, None
            if len(buf) < 20:
                self.warnings.append(f"{kind}: only {len(buf)} samples - is the ESP32 streaming?")
                return False
            acc = np.array([s.acc for s in buf])     # (N, 2, 3)
            gyr = np.array([s.gyr for s in buf])
            emg = np.array([s.emg for s in buf])
            if kind == "still":
                motion = np.abs(gyr - gyr.mean(axis=0)).max()
                if motion > 0.35:
                    self.warnings.append("arm moved during the hanging step; recalibrate with 'r'")
                self.bias = gyr.mean(axis=0)
                for i in range(2):
                    a = acc[:, i].mean(axis=0)
                    limb = -a / np.linalg.norm(a)          # hanging limb points opposite to 'up'
                    self.limb_axis[i] = limb
                    e = np.eye(3)[int(np.argmin(np.abs(limb)))]
                    r = e - (e @ limb) * limb
                    self.ref_axis[i] = r / np.linalg.norm(r)
                    self.filters[i].reset(a)
                self.have_axes = True
                self.emg_rest = float(np.median(emg))
                self.home_q = None
                self.angles = None
            elif kind == "relax":
                self.emg_rest = float(np.median(emg))
            elif kind == "squeeze":
                self.emg_max = float(np.percentile(emg, 90))
                if self.emg_rest is not None and self.emg_max - self.emg_rest < 50:
                    self.warnings.append(
                        f"EMG barely changed (rest {self.emg_rest:.0f}, fist {self.emg_max:.0f}); "
                        "check electrode placement - gripper disabled until 'e' succeeds")
                    self.emg_max = None
            elif kind == "home":
                if not self.have_axes:
                    self.warnings.append("home pose needs the hanging step first ('r')")
                    return False
                self.home_q = [f.q.copy() for f in self.filters]
                for i in range(2):
                    v = q_to_R(self.home_q[i]) @ self.limb_axis[i]
                    hz = np.array([v[0], v[1], 0.0])
                    if np.linalg.norm(hz) < 0.17:
                        self.warnings.append(
                            ("upper arm" if i == 0 else "forearm") +
                            " is almost vertical in the home pose; tilt it ~20 deg forward and press 'c'")
                        hz = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(hz) < 1e-6 else hz
                    self.home_h[i] = hz / np.linalg.norm(hz)
                self.home_angles = None          # filled on the next sample
            return True

    def pop_warnings(self) -> list:
        with self.lock:
            w, self.warnings = self.warnings, []
        return w

    # ---- per-sample processing ----------------------------------------------
    def process(self, s: Sample) -> None:
        with self.lock:
            self.samples_seen += 1
            self.last_sample_wall = time.monotonic()
            self.last_flags = s.flags
            dt = 0.01 if self._t_prev is None else min(max(s.t - self._t_prev, 1e-3), 0.1)
            self._t_prev = s.t
            if self._collect is not None:
                self._buf.append(s)
            if s.flags & 3:
                return
            for i in range(2):
                self.filters[i].update(s.gyr[i] - self.bias[i], s.acc[i], dt)
            self._update_emg(s.emg)
            if self.have_axes and self.home_q is not None:
                raw = self._human_angles()
                if self.home_angles is None:
                    self.home_angles = raw
                b0, sh0, el0, ro0 = self.home_angles
                self.angles = Angles(base=raw[0] - b0, shoulder=raw[1] - sh0,
                                     elbow=wrap(raw[2] - el0), roll=wrap(raw[3] - ro0))

    def _human_angles(self) -> tuple:
        z = np.array([0.0, 0.0, 1.0])
        phi = []
        yaw = []
        for i in range(2):
            q = self.filters[i].q
            v = q_to_R(q) @ self.limb_axis[i]
            y = twist_about_z(q_mul(q, q_conj(self.home_q[i])))
            h = rot_z(y, self.home_h[i])
            phi.append(math.atan2(float(v @ h), float(-v[2])))   # 0 = hanging, + = forward
            yaw.append(y)
        # forearm twist about its own axis, measured against world vertical
        Rf = q_to_R(self.filters[1].q)
        f = Rf @ self.limb_axis[1]
        r = Rf @ self.ref_axis[1]
        p = r - (r @ f) * f
        up = z - (z @ f) * f
        if np.linalg.norm(up) > 0.2 and np.linalg.norm(p) > 1e-6:
            roll = math.atan2(float(f @ np.cross(up, p)), float(up @ p))
            self._last_roll = roll
        else:
            roll = self._last_roll                    # forearm vertical: twist undefined, hold
        return (yaw[0], phi[0], wrap(phi[1] - phi[0]), roll)

    def _update_emg(self, emg: float) -> None:
        self.emg_env += 0.3 * (emg - self.emg_env)
        if self.emg_rest is None or self.emg_max is None:
            self.grip = False
            return
        span = max(self.emg_max - self.emg_rest, 1.0)
        level = (self.emg_env - self.emg_rest) / span
        c = self.cfg
        if not self.grip:
            self._grip_count = self._grip_count + 1 if level > c.grip_on_frac else 0
            if self._grip_count >= c.grip_on_samples:
                self.grip, self._grip_count = True, 0
        else:
            self._grip_count = self._grip_count + 1 if level < c.grip_off_frac else 0
            if self._grip_count >= c.grip_off_samples:
                self.grip, self._grip_count = False, 0

    def snapshot(self) -> dict:
        with self.lock:
            span = None
            if self.emg_rest is not None and self.emg_max is not None:
                span = max(self.emg_max - self.emg_rest, 1.0)
            return dict(
                angles=self.angles, grip=self.grip,
                emg_level=None if span is None else (self.emg_env - self.emg_rest) / span,
                emg_env=self.emg_env, age=time.monotonic() - self.last_sample_wall,
                flags=self.last_flags, samples=self.samples_seen,
            )


# =============================================================================
# Calibration sequence (non-blocking, so the viewer keeps rendering)
# =============================================================================
STEPS = {
    "still": ("HANG: let your arm hang straight down by your side, elbow straight, "
              "palm facing your thigh. Relax your hand. Stay still.", 4.0, 2.0),
    "relax": ("RELAX: open, relaxed hand. Stay still.", 2.5, 1.5),
    "squeeze": ("SQUEEZE: keep the arm where it is and make a HARD fist. Hold it.", 3.0, 2.0),
    "home": ("HOME: upper arm tilted ~20-30 deg forward, elbow bent ~90 deg, forearm level "
             "pointing forward, thumb up. Hold still - this pose = the robot's home pose.", 4.0, 2.0),
}
FULL = ["still", "squeeze", "home"]


class Calibrator:
    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker
        self.queue: list = []
        self.step: Optional[str] = None
        self.collecting = False
        self.t_mark = 0.0
        self._last_count = -1

    @property
    def busy(self) -> bool:
        return self.step is not None or bool(self.queue)

    @property
    def phase(self) -> str:
        return self.step or "free"

    def start(self, steps: list) -> None:
        self.queue = list(steps)
        self.step = None

    def tick(self, now: float) -> None:
        if self.step is None:
            if not self.queue:
                return
            self.step = self.queue.pop(0)
            self.collecting = False
            self.t_mark = now
            self._last_count = -1
            print(f"\n>>> {STEPS[self.step][0]}", flush=True)
        text, prep, hold = STEPS[self.step]
        if not self.collecting:
            left = prep - (now - self.t_mark)
            if int(left) != self._last_count and left > 0:
                self._last_count = int(left)
                print(f"    starting in {int(left) + 1} ...", flush=True)
            if left <= 0:
                self.collecting = True
                self.t_mark = now
                self.tracker.begin(self.step)
                print("    recording - hold still", flush=True)
        elif now - self.t_mark >= hold:
            ok = self.tracker.end(self.step)
            print(f"    {self.step}: {'done' if ok else 'FAILED'}", flush=True)
            for w in self.tracker.pop_warnings():
                print(f"    WARNING: {w}", flush=True)
            self.step = None
            if not self.queue:
                print("\nCalibration finished. Commands: c=re-zero home  e=EMG  r=full  "
                      "b=base on/off  q=quit\n", flush=True)


# =============================================================================
# Logging + reader thread + terminal commands
# =============================================================================
class SessionLogger:
    HEADER = (["host_time", "t", "ax1", "ay1", "az1", "gx1", "gy1", "gz1",
               "ax2", "ay2", "az2", "gx2", "gy2", "gz2", "emg", "flags",
               "base", "shoulder", "elbow", "roll", "grip", "calib_phase"])

    def __init__(self, folder: str) -> None:
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, time.strftime("session_%Y%m%d_%H%M%S.csv"))
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(self.HEADER)
        self._last_flush = time.monotonic()

    def write(self, s: Sample, snap_angles: Optional[Angles], grip: bool, phase: str) -> None:
        a = snap_angles
        self.w.writerow([f"{time.time():.4f}", f"{s.t:.4f}",
                         *[f"{v:.5f}" for v in s.acc[0]], *[f"{v:.5f}" for v in s.gyr[0]],
                         *[f"{v:.5f}" for v in s.acc[1]], *[f"{v:.5f}" for v in s.gyr[1]],
                         f"{s.emg:.0f}", s.flags,
                         *([f"{a.base:.4f}", f"{a.shoulder:.4f}", f"{a.elbow:.4f}", f"{a.roll:.4f}"]
                           if a else ["", "", "", ""]),
                         int(grip), phase])
        if time.monotonic() - self._last_flush > 1.0:
            self.f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        self.f.close()


def start_reader(source, tracker: Tracker, logger: Optional[SessionLogger],
                 phase: Callable[[], str], stop: threading.Event) -> threading.Thread:
    def run() -> None:
        try:
            for s in source.samples(stop):
                tracker.process(s)
                if logger:
                    logger.write(s, tracker.angles, tracker.grip, phase())
        except Exception as exc:                       # serial unplugged etc.
            log.error("Sensor stream stopped: %s", exc)
            stop.set()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th


def start_keyboard() -> "queue.Queue[str]":
    q: "queue.Queue[str]" = queue.Queue()

    def run() -> None:
        for line in sys.stdin:
            q.put(line.strip().lower())
    threading.Thread(target=run, daemon=True).start()
    return q


# =============================================================================
# Robot mapping (main thread, frame rate)
# =============================================================================
class RobotMapper:
    def __init__(self, sim, cfg: TeleopConfig) -> None:
        m, d = sim.model, sim.data
        self.sim, self.cfg = sim, cfg
        self.home = np.zeros(m.nu)
        for a in range(m.nu):
            jid = m.actuator_trnid[a, 0]
            self.home[a] = d.qpos[m.jnt_qposadr[jid]] if jid >= 0 else d.ctrl[a]
        self.target = self.home.copy()
        self.out = self.home.copy()
        g = sim.ids["gripper"]
        self.grip_closed = sim.ctrl_lo[g] if cfg.gripper_close_is_low else sim.ctrl_hi[g]
        self.grip_open = sim.ctrl_hi[g] if cfg.gripper_close_is_low else sim.ctrl_lo[g]
        self.out[g] = self.grip_open

    def update(self, snap: dict, dt: float) -> None:
        c, ids, sim = self.cfg, self.sim.ids, self.sim
        a: Optional[Angles] = snap["angles"]
        live = a is not None and snap["age"] < c.stale_after and not (snap["flags"] & 3)
        if live:
            self.target[ids["base"]] = self.home[ids["base"]] + (
                c.base_sign * c.base_gain * a.base if c.use_base else 0.0)
            self.target[ids["shoulder"]] = self.home[ids["shoulder"]] + c.shoulder_sign * c.shoulder_gain * a.shoulder
            self.target[ids["elbow"]] = self.home[ids["elbow"]] + c.elbow_sign * c.elbow_gain * a.elbow
            self.target[ids["wrist_roll"]] = self.home[ids["wrist_roll"]] + c.roll_sign * c.roll_gain * a.roll
            self.target[ids["wrist"]] = self.home[ids["wrist"]]
        else:
            self.target[:] = self.out                  # hold where we are
        g = ids["gripper"]
        self.target[g] = self.grip_closed if (live and snap["grip"]) else (
            self.out[g] if not live else self.grip_open)
        np.clip(self.target, sim.ctrl_lo, sim.ctrl_hi, out=self.target)

        alpha = 1 - math.exp(-dt / c.smoothing_tau)
        for j in range(len(self.out)):
            if j == g:
                step = c.gripper_speed * dt
                self.out[j] += max(-step, min(step, self.target[j] - self.out[j]))
                continue
            desired = self.out[j] + alpha * (self.target[j] - self.out[j])
            step = c.max_joint_rate * dt
            self.out[j] += max(-step, min(step, desired - self.out[j]))
        sim.data.ctrl[:] = self.out


# =============================================================================
# Run modes
# =============================================================================
def status_line(snap: dict, cal: Calibrator) -> str:
    a = snap["angles"]
    if snap["samples"] == 0:
        return "waiting for data from the ESP32 ..."
    if snap["flags"] & 3:
        bad = [n for b, n in ((1, "upper-arm"), (2, "forearm")) if snap["flags"] & b]
        return f"IMU read error: {', '.join(bad)} - check wiring"
    emg = "--" if snap["emg_level"] is None else f"{100 * snap['emg_level']:4.0f}%"
    if a is None:
        return f"calibrating ({cal.phase}) | EMG {emg} raw {snap['emg_env']:.0f}"
    return (f"base {a.base / DEG:+6.1f}  shoulder {a.shoulder / DEG:+6.1f}  "
            f"elbow {a.elbow / DEG:+6.1f}  roll {a.roll / DEG:+6.1f} deg | EMG {emg} "
            f"| grip {'CLOSED' if snap['grip'] else 'open  '}")


def handle_key(key: str, cal: Calibrator, cfg: TeleopConfig) -> bool:
    if key == "q":
        return False
    if key == "c":
        cal.start(["home"])
    elif key == "e":
        cal.start(["relax", "squeeze"])
    elif key == "r":
        cal.start(FULL)
    elif key == "b":
        cfg.use_base = not cfg.use_base
        print(f"\nbase {'ON' if cfg.use_base else 'OFF (held at home)'}", flush=True)
    elif key:
        print("\ncommands: c e r b q", flush=True)
    return True


def open_source(args, cal: Calibrator):
    if args.fake:
        return FakeWearable(phase=lambda: cal.phase)
    port = args.port or find_esp32_port()
    if port is None:
        ports = list_ports()
        print("Could not pick the ESP32 port automatically. Ports found:")
        for p in ports:
            print(f"  {p.device:10s} {p.description}")
        print("Run again with --port <name>, e.g.  python imu_teleop.py --port COM5")
        sys.exit(1)
    log.info("Opening %s at %d baud", port, args.baud)
    return SerialWearable(port, args.baud)


def run(args) -> int:
    cfg = TeleopConfig(use_base=not args.no_base)
    tracker = Tracker(cfg)
    cal = Calibrator(tracker)
    source = open_source(args, cal)
    stop = threading.Event()
    logger = None if args.no_log else SessionLogger(os.path.join(args.dir, "logs"))
    if logger:
        log.info("Logging to %s", logger.path)
    start_reader(source, tracker, logger, lambda: cal.phase, stop)
    keys = start_keyboard()
    cal.start(FULL)

    try:
        if args.check:
            _loop_check(tracker, cal, keys, cfg, stop)
        else:
            _loop_sim(args, tracker, cal, keys, cfg, stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        source.close()
        if logger:
            logger.close()
            log.info("Saved %s", logger.path)
    return 0


def _loop_check(tracker, cal, keys, cfg, stop) -> None:
    last = 0.0
    while not stop.is_set():
        now = time.monotonic()
        cal.tick(now)
        while not keys.empty():
            if not handle_key(keys.get(), cal, cfg):
                return
        if not cal.collecting and now - last > 0.1:
            last = now
            print("\r" + status_line(tracker.snapshot(), cal) + "   ", end="", flush=True)
        time.sleep(0.01)


def _loop_sim(args, tracker, cal, keys, cfg, stop) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    os.chdir(os.path.expanduser(args.dir))
    try:
        from sim_comms import ArmSimulator, SimConfig
    except ImportError as exc:
        raise SystemExit(f"Could not import sim_comms.py ({exc}). Put imu_teleop.py next to "
                         "sim_comms.py and run: pip install mujoco pygame numpy")
    import mujoco.viewer

    sim = ArmSimulator(SimConfig(fps=args.fps))
    mapper = RobotMapper(sim, cfg)
    log.info("Launching MuJoCo viewer. Follow the prompts in this terminal.")
    last_print = 0.0
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.lookat = [0.15, 0.0, 0.55]
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 140.0
        t_prev = time.monotonic()
        while viewer.is_running() and not stop.is_set():
            t0 = time.monotonic()
            dt, t_prev = t0 - t_prev, t0
            cal.tick(t0)
            while not keys.empty():
                if not handle_key(keys.get(), cal, cfg):
                    return
            snap = tracker.snapshot()
            if cal.busy:
                snap = dict(snap, angles=None)         # hold the robot at home while calibrating
                if cal.step == "home":
                    mapper.out[:] = mapper.home
            mapper.update(snap, max(dt, 1e-3))
            sim.step()
            viewer.sync()
            if not cal.collecting and t0 - last_print > 0.2:
                last_print = t0
                print("\r" + status_line(snap, cal) + "   ", end="", flush=True)
            remaining = 1.0 / args.fps - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wearable IMU + EMG teleoperation of the MuJoCo arm.")
    p.add_argument("--port", help="serial port of the ESP32 (e.g. COM5 or /dev/ttyUSB0)")
    p.add_argument("--baud", type=int, default=921600, help="must match BAUD in the sketch")
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="folder containing scene.xml and sim_comms.py")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--check", action="store_true", help="no MuJoCo: print live angles only")
    p.add_argument("--fake", action="store_true", help="use synthetic sensor data")
    p.add_argument("--no-base", action="store_true", help="hold the base at home")
    p.add_argument("--no-log", action="store_true", help="do not write logs/session_*.csv")
    p.add_argument("--list-ports", action="store_true", help="list serial ports and exit")
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    if args.list_ports:
        for p in list_ports():
            print(f"{p.device:12s} {p.description}")
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
