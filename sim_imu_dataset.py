#!/usr/bin/env python3
"""
sim_imu_dataset.py - Steps 1 + 2: scripted motions on the cycloidal arm in MuJoCo,
logging ground truth (Y) and clean simulated IMUs (X).

What it does
------------
1. Loads scene.xml and adds two virtual IMUs (accelerometer + gyro) without
   touching your XML files:
      IMU A ("upper")  on the link moved by the SHOULDER joint
      IMU B ("fore")   on the link moved by the ELBOW joint
   Each IMU sits at the link's centre of mass by default (override with
   --upper-pos / --fore-pos, in the link's own frame, metres).
2. Drives the arm through a random mix of motion segments:
      reach    slow minimum-jerk reaches between random poses
      circle   slow circles (shoulder + elbow sines 90 deg apart)
      flick    rapid minimum-jerk flicks (0.15-0.4 s)
      wave     fast elbow waving + wrist roll
      hold     static holds of 2-6 s (zero-velocity periods for bias learning)
      multisine  random sums of sines on every joint (0.1-2 Hz)
   Base and wrist joints move too, so the IMUs see realistic coupled motion.
3. Logs at --rate Hz (default 200): joint angles + angular velocities of the
   shoulder and elbow (ground truth), clean accel (g) and gyro (rad/s) of both
   IMUs, actuator torques (used later for the EMG proxy), and the motion label.

Output: data/sim_imu.npz  (+ data/sim_imu_meta.json, + scene_with_imus.xml)

Usage (from the folder that has scene.xml):
    python sim_imu_dataset.py                     20 simulated minutes, headless
    python sim_imu_dataset.py --minutes 60        more data
    python sim_imu_dataset.py --view              watch it in the MuJoCo viewer (real time)
    python sim_imu_dataset.py --view --minutes 2  quick visual check

Needs: pip install "mujoco>=3.2" numpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

# Actuator names used by sim_comms.py: logical name -> (XML name, fallback index)
ACTUATORS = (
    ("base", ("base_act", 0)),
    ("shoulder", ("shoulder_act", 1)),
    ("elbow", ("Elbow_act", 2)),
    ("wrist", ("wrist_act", 3)),
    ("wrist_roll", ("wrist_roll_act", 4)),
    ("gripper", ("gripper_act", 5)),
)
MOVING = ("base", "shoulder", "elbow", "wrist", "wrist_roll")
MOTIONS = ("reach", "circle", "flick", "wave", "hold", "multisine")
MOTION_WEIGHTS = (0.22, 0.14, 0.14, 0.12, 0.18, 0.20)


# =============================================================================
# Step 1: trajectory generator (pure numpy, no MuJoCo needed)
# =============================================================================
def min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class TrajectoryGenerator:
    """Produces joint position targets for the moving joints, segment by segment.

    lo, hi: per-joint target limits (same order as `names`). A 10 % margin is kept
    inside the limits. Segments blend into each other over 0.4 s, so targets never jump.
    max_speed: per-joint target speed cap in rad/s (human-like: ~290 deg/s for shoulder/elbow).
    """

    # base, shoulder, elbow, wrist pitch, wrist roll  (rad/s)
    MAX_SPEED = (3.0, 5.0, 5.0, 6.0, 8.0)

    def __init__(self, names, lo, hi, rng: np.random.Generator, start=None, max_speed=None) -> None:
        self.names = list(names)
        self.vmax = np.asarray(max_speed if max_speed is not None else self.MAX_SPEED, float)
        self._t_last = None
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        pad = 0.1 * (hi - lo)
        self.lo, self.hi = lo + pad, hi - pad
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        self.rng = rng
        self.cur = self.mid.copy() if start is None else np.clip(np.asarray(start, float), self.lo, self.hi)
        self.kind = "hold"
        self.t0 = 0.0
        self.dur = 0.0
        self.f = lambda t: self.cur
        self.blend_from = self.cur.copy()
        self.blend_T = 0.4

    def idx(self, name: str) -> int:
        return self.names.index(name)

    def rand_pose(self, scale: float = 1.0) -> np.ndarray:
        return self.mid + scale * self.half * self.rng.uniform(-1, 1, len(self.names))

    def _new_segment(self, t: float) -> None:
        r = self.rng
        self.kind = str(r.choice(MOTIONS, p=MOTION_WEIGHTS))
        self.t0 = t
        self.blend_from = self.cur.copy()
        start = self.cur.copy()
        n = len(self.names)

        if self.kind == "hold":
            self.dur = r.uniform(2.0, 6.0)
            self.f = lambda tt, p=start: p

        elif self.kind == "reach":
            self.dur = r.uniform(1.5, 4.0)
            goal = self.rand_pose(0.9)
            self.f = lambda tt, a=start, b=goal, T=self.dur: a + (b - a) * min_jerk(tt / T)

        elif self.kind == "flick":
            hops = int(r.integers(3, 7))
            durs = r.uniform(0.2, 0.45, hops)
            pts, p = [start], start
            for _ in range(hops):                     # each flick moves 0.15-0.5 rad per joint
                p = np.clip(p + r.choice([-1.0, 1.0], n) * r.uniform(0.15, 0.5, n), self.lo, self.hi)
                pts.append(p)
            ends = np.cumsum(durs)
            self.dur = float(ends[-1]) + r.uniform(0.2, 0.6)

            def f(tt, pts=pts, durs=durs, ends=ends):
                k = int(np.searchsorted(ends, tt))
                if k >= len(durs):
                    return pts[-1]
                t_start = ends[k] - durs[k]
                return pts[k] + (pts[k + 1] - pts[k]) * min_jerk((tt - t_start) / durs[k])
            self.f = f

        elif self.kind == "circle":
            self.dur = r.uniform(4.0, 10.0)
            fr = r.uniform(0.15, 0.5)
            amp = self.half * r.uniform(0.2, 0.5, n)
            center = np.clip(start, self.lo + amp, self.hi - amp)
            ph = np.zeros(n)
            ph[self.idx("elbow")] = math.pi / 2
            mask = np.zeros(n)
            mask[[self.idx("shoulder"), self.idx("elbow")]] = 1.0
            mask[self.idx("base")] = r.choice([0.0, 0.3])
            self.f = lambda tt, c=center, A=amp * mask, w=2 * math.pi * fr, p=ph: \
                c + A * np.sin(w * tt + p)

        elif self.kind == "wave":
            self.dur = r.uniform(3.0, 6.0)
            amp = np.zeros(n)
            freq = np.zeros(n)
            e, wr = self.idx("elbow"), self.idx("wrist_roll")
            freq[e] = r.uniform(1.0, 2.5)
            amp[e] = min(r.uniform(0.1, 0.4), self.half[e], 0.8 * self.vmax[e] / (2 * math.pi * freq[e]))
            freq[wr] = r.uniform(1.0, 2.5)
            amp[wr] = min(r.uniform(0.2, 0.6), self.half[wr], 0.8 * self.vmax[wr] / (2 * math.pi * freq[wr]))
            center = np.clip(start, self.lo + amp, self.hi - amp)
            self.f = lambda tt, c=center, A=amp, w=2 * math.pi * freq: c + A * np.sin(w * tt)

        else:  # multisine
            self.dur = r.uniform(6.0, 12.0)
            k = 3
            freqs = r.uniform(0.1, 2.0, (k, n))
            phases = r.uniform(0, 2 * math.pi, (k, n))
            amps = r.uniform(0.2, 1.0, (k, n))
            amps *= (0.6 * self.half / amps.sum(axis=0))       # stay inside the range
            # keep the summed peak speed under ~60 % of the joint's speed cap
            speed = (amps * 2 * math.pi * freqs).sum(axis=0)
            amps *= np.minimum(1.0, 0.6 * self.vmax / np.maximum(speed, 1e-9))
            center = self.mid + 0.2 * self.half * r.uniform(-1, 1, n)
            self.f = lambda tt, c=center, A=amps, w=2 * math.pi * freqs, p=phases: \
                c + (A * np.sin(w * tt + p)).sum(axis=0)

    def target(self, t: float) -> np.ndarray:
        if t - self.t0 >= self.dur:
            self._new_segment(t)
        tt = t - self.t0
        x = np.asarray(self.f(tt), float)
        if tt < self.blend_T and self.kind not in ("reach", "flick", "hold"):
            w = min_jerk(tt / self.blend_T)
            x = (1 - w) * self.blend_from + w * x
        if self._t_last is not None:                  # human-like speed cap on every joint
            step = self.vmax * max(t - self._t_last, 0.0)
            x = self.cur + np.clip(x - self.cur, -step, step)
        self._t_last = t
        self.cur = np.clip(x, self.lo, self.hi)
        return self.cur

    @property
    def label(self) -> int:
        return MOTIONS.index(self.kind)


# =============================================================================
# Step 2: MuJoCo model with virtual IMUs + logging
# =============================================================================
# The MjSpec API changed names between MuJoCo releases (find_body -> body, etc.).
# These helpers try every known spelling so the script works on any MuJoCo >= 3.2.
def spec_find_body(spec, name: str):
    for attr in ("body", "find_body"):
        fn = getattr(spec, attr, None)
        if callable(fn):
            try:
                b = fn(name)
            except Exception:
                b = None
            if b is not None:
                return b
    for attr in ("bodies",):                          # last resort: scan every body
        bodies = getattr(spec, attr, None)
        if bodies is not None:
            for b in bodies:
                if getattr(b, "name", None) == name:
                    return b
    worldbody = getattr(spec, "worldbody", None)      # older releases: walk the tree
    stack = [worldbody] if worldbody is not None else []
    while stack:
        b = stack.pop()
        if getattr(b, "name", None) == name:
            return b
        for attr in ("bodies", "first_body"):
            kids = getattr(b, attr, None)
            if kids is None:
                continue
            if callable(kids):
                break
            stack.extend(list(kids))
            break
    return None


def _set_fields(obj, fields: dict) -> None:
    for k, v in fields.items():
        cur = getattr(obj, k, None)
        if isinstance(cur, np.ndarray) and not isinstance(v, str):
            cur[:len(v)] = v                          # array fields: write in place
        else:
            setattr(obj, k, v)


def spec_add_site(body, **fields):
    try:
        return body.add_site(**fields)
    except TypeError:                                 # older API: create, then set fields
        site = body.add_site()
        _set_fields(site, fields)
        return site


def spec_add_sensor(spec, **fields):
    try:
        return spec.add_sensor(**fields)
    except TypeError:
        sensor = spec.add_sensor()
        _set_fields(sensor, fields)
        return sensor


def build_model(scene: str, upper_pos, fore_pos, save_xml: str | None):
    import mujoco
    if not hasattr(mujoco, "MjSpec"):
        raise SystemExit("This needs mujoco >= 3.2 (for MjSpec). Run: pip install -U mujoco")

    base_model = mujoco.MjModel.from_xml_path(scene)
    ids = {}
    for key, (name, fb) in ACTUATORS:
        a = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a == -1:
            print(f"  actuator '{name}' not found, using index {fb}")
            a = fb
        ids[key] = a

    def link_of(key: str):
        jid = base_model.actuator_trnid[ids[key], 0]
        bid = base_model.jnt_bodyid[jid]
        jname = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        bname = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not bname:
            raise SystemExit(f"The link moved by joint '{jname}' has no name in the XML. "
                             f"Give that <body> a name= attribute and run again.")
        return jid, jname, bname, base_model.body_ipos[bid].copy()

    spec = mujoco.MjSpec.from_file(scene)
    imus = {}
    for tag, key, override in (("upper", "shoulder", upper_pos), ("fore", "elbow", fore_pos)):
        jid, jname, bname, com = link_of(key)
        pos = np.asarray(override, float) if override is not None else com
        body = spec_find_body(spec, bname)
        if body is None:
            raise SystemExit(f"Could not find body '{bname}' in the spec.")
        spec_add_site(body, name=f"imu_{tag}", pos=pos.tolist(), size=[0.012, 0.012, 0.012],
                      rgba=[1.0, 0.4, 0.0, 1.0])
        spec_add_sensor(spec, name=f"imu_{tag}_acc", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"imu_{tag}")
        spec_add_sensor(spec, name=f"imu_{tag}_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                        objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"imu_{tag}")
        imus[tag] = dict(joint=jname, body=bname, pos=pos.tolist())
        print(f"  IMU '{tag}': on link '{bname}' (moved by joint '{jname}') at {np.round(pos, 4).tolist()} m")

    model = spec.compile()
    if save_xml:
        try:
            with open(save_xml, "w") as fh:
                fh.write(spec.to_xml())
            print(f"  wrote {save_xml} (open it in the viewer to see the orange IMU sites)")
        except Exception as exc:                     # to_xml can fail on some asset setups
            print(f"  (could not write {save_xml}: {exc})")
    return model, ids, imus


def contact_geoms(data):
    """(geom_a, geom_b) of every active contact, for both old and new MuJoCo field names."""
    n = data.ncon
    con = data.contact
    try:
        g = np.asarray(con.geom).reshape(-1, 2)[:n]           # MuJoCo 3.x: geom[2]
        return [(int(a), int(b)) for a, b in g]
    except Exception:
        pass
    try:
        g1 = np.asarray(con.geom1).ravel()[:n]
        g2 = np.asarray(con.geom2).ravel()[:n]
        return [(int(a), int(b)) for a, b in zip(g1, g2)]
    except Exception:
        return []


def actuator_limits(model, ids):
    lo = model.actuator_ctrlrange[:, 0].copy()
    hi = model.actuator_ctrlrange[:, 1].copy()
    for a in range(model.nu):
        jid = model.actuator_trnid[a, 0]
        if not model.actuator_ctrllimited[a]:
            lo[a], hi[a] = (-math.pi, math.pi)
        if model.jnt_limited[jid]:                     # never aim past the joint's real limits
            jlo, jhi = model.jnt_range[jid]
            lo[a], hi[a] = max(lo[a], jlo), min(hi[a], jhi)
    return lo, hi


def generate(args) -> None:
    import mujoco
    os.chdir(os.path.expanduser(args.dir))
    print(f"Loading {args.scene} ...")
    model, ids, imus = build_model(args.scene, args.upper_pos, args.fore_pos,
                                   None if args.no_xml else "scene_with_imus.xml")
    data = mujoco.MjData(model)

    def sensor_slice(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = model.sensor_adr[sid]
        return slice(adr, adr + 3)
    S = {k: sensor_slice(k) for k in ("imu_upper_acc", "imu_upper_gyro", "imu_fore_acc", "imu_fore_gyro")}
    g = float(np.linalg.norm(model.opt.gravity)) or 9.81

    qadr = {k: model.jnt_qposadr[model.actuator_trnid[ids[k], 0]] for k in MOVING}
    vadr = {k: model.jnt_dofadr[model.actuator_trnid[ids[k], 0]] for k in MOVING}

    lo, hi = actuator_limits(model, ids)
    for spec_str in args.limit or []:                 # manual overrides: exact range in degrees
        try:
            name, a, b = spec_str.split(":")
            a, b = math.radians(float(a)), math.radians(float(b))
            if b <= a:
                raise ValueError
            # the generator keeps a 10 % margin inside lo..hi; expand so the arm moves exactly in a..b
            lo[ids[name]], hi[ids[name]] = a - 0.125 * (b - a), b + 0.125 * (b - a)
        except (ValueError, KeyError):
            raise SystemExit(f"Bad --limit '{spec_str}'. Use NAME:MIN:MAX in degrees, NAME one of "
                             f"{', '.join(MOVING)}  e.g.  --limit shoulder:-36:110")
    print("  range the arm will move in (deg):")
    for k in MOVING:
        jid = model.actuator_trnid[ids[k], 0]
        a, b = lo[ids[k]] + 0.1 * (hi[ids[k]] - lo[ids[k]]), hi[ids[k]] - 0.1 * (hi[ids[k]] - lo[ids[k]])
        jr = (f"joint limit {math.degrees(model.jnt_range[jid][0]):7.1f} .. "
              f"{math.degrees(model.jnt_range[jid][1]):6.1f}" if model.jnt_limited[jid] else "joint unlimited")
        print(f"    {k:10s} {math.degrees(a):7.1f} .. {math.degrees(b):6.1f}   ({jr})")
    mov_ids = [ids[k] for k in MOVING]
    rng = np.random.default_rng(args.seed)
    contact_pairs: dict = {}
    contact_samples = 0
    body_name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"

    def reset():
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        start = np.array([data.qpos[qadr[k]] for k in MOVING])
        return TrajectoryGenerator(MOVING, lo[mov_ids], hi[mov_ids], rng, start=start)

    gen = reset()
    dt = model.opt.timestep
    every = max(1, round((1.0 / args.rate) / dt))
    real_rate = 1.0 / (every * dt)
    if abs(real_rate - args.rate) > 0.5:
        print(f"  note: timestep {dt} s gives a logging rate of {real_rate:.1f} Hz")
    n_steps = int(args.minutes * 60 / dt)
    n_log = n_steps // every + 1

    out = {
        "t": np.zeros(n_log), "episode": np.zeros(n_log, np.int32), "motion": np.zeros(n_log, np.int8),
        "q": np.zeros((n_log, 2)), "qd": np.zeros((n_log, 2)),          # shoulder, elbow
        "q_all": np.zeros((n_log, len(MOVING))),                         # base..wrist_roll
        "acc": np.zeros((n_log, 2, 3)), "gyr": np.zeros((n_log, 2, 3)),  # [upper, fore] in g, rad/s
        "tau": np.zeros((n_log, 2)),                                     # shoulder, elbow actuator force
        "target": np.zeros((n_log, 2)),
        "contact": np.zeros(n_log, bool),                                # arm touching anything

    }

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 2.2, -12.0, 140.0
        viewer.cam.lookat[:] = [0.15, 0.0, 0.55]

    episode, k, sim_t = 0, 0, 0.0
    grip_lo, grip_hi = lo[ids["gripper"]], hi[ids["gripper"]]
    next_grip = 0.0
    wall0 = time.monotonic()
    print(f"Simulating {args.minutes:.1f} min at dt={dt*1000:.2f} ms, logging {real_rate:.0f} Hz "
          f"({n_log} samples) ...")
    for step in range(n_steps):
        target = gen.target(sim_t)
        data.ctrl[mov_ids] = target
        if sim_t >= next_grip:                       # gripper opens/closes now and then
            data.ctrl[ids["gripper"]] = rng.choice([grip_lo, grip_hi])
            next_grip = sim_t + rng.uniform(1.0, 5.0)
        mujoco.mj_step(model, data)
        sim_t += dt

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            print(f"  unstable at t={sim_t:.2f}s - resetting (new episode)")
            gen = reset()
            episode += 1
            continue

        if step % every == 0 and k < n_log:
            mujoco.mj_forward(model, data)           # sensors consistent with the logged state
            sd = data.sensordata
            out["t"][k] = sim_t
            out["episode"][k] = episode
            out["motion"][k] = gen.label
            out["q"][k] = [data.qpos[qadr["shoulder"]], data.qpos[qadr["elbow"]]]
            out["qd"][k] = [data.qvel[vadr["shoulder"]], data.qvel[vadr["elbow"]]]
            out["q_all"][k] = [data.qpos[qadr[j]] for j in MOVING]
            out["acc"][k, 0] = sd[S["imu_upper_acc"]] / g
            out["acc"][k, 1] = sd[S["imu_fore_acc"]] / g
            out["gyr"][k, 0] = sd[S["imu_upper_gyro"]]
            out["gyr"][k, 1] = sd[S["imu_fore_gyro"]]
            out["tau"][k] = [data.actuator_force[ids["shoulder"]], data.actuator_force[ids["elbow"]]]
            out["target"][k] = [target[MOVING.index("shoulder")], target[MOVING.index("elbow")]]
            if data.ncon:
                contact_samples += 1
                out["contact"][k] = True
                seen = set()
                for ga, gb in contact_geoms(data):
                    if ga < 0 or gb < 0:
                        continue
                    pair = tuple(sorted((body_name(model.geom_bodyid[ga]),
                                         body_name(model.geom_bodyid[gb]))))
                    if pair not in seen:
                        seen.add(pair)
                        contact_pairs[pair] = contact_pairs.get(pair, 0) + 1
            k += 1

        if viewer is not None:
            if not viewer.is_running():
                print("  viewer closed - continuing headless")
                viewer = None
            elif step % max(1, round(1 / (60 * dt))) == 0:
                viewer.sync()
                lag = wall0 + sim_t - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
        if step % int(60 / dt) == 0 and step:
            print(f"  {sim_t/60:5.1f} / {args.minutes:.1f} min simulated")

    if viewer is not None:
        viewer.close()
    for key in out:
        out[key] = out[key][:k]

    # Coverage report: did the actuators actually produce fast and slow motion?
    qd_deg = np.abs(out["qd"]) * 180 / math.pi
    track = np.abs(out["q"] - out["target"]) * 180 / math.pi
    print("\nCoverage")
    for j, name in enumerate(("shoulder", "elbow")):
        print(f"  {name:8s} range {np.degrees(out['q'][:, j].min()):7.1f} .. "
              f"{np.degrees(out['q'][:, j].max()):6.1f} deg | speed 95th pct "
              f"{np.percentile(qd_deg[:, j], 95):6.1f} deg/s, max {qd_deg[:, j].max():6.1f} deg/s | "
              f"median tracking error {np.median(track[:, j]):5.2f} deg")
    counts = np.bincount(out["motion"], minlength=len(MOTIONS)) / max(k, 1)
    print("  motion mix: " + ", ".join(f"{m} {100*c:.0f}%" for m, c in zip(MOTIONS, counts)))
    acc_n = np.linalg.norm(out["acc"], axis=2)
    print(f"  accel magnitude: median {np.median(acc_n):.2f} g, max {acc_n.max():.2f} g, "
          f"above 4 g in {100 * (acc_n.max(axis=1) > 4).mean():.2f}% of samples "
          f"(MPU-6050 range used later: 4 g)")
    print(f"  contacts: in {100 * contact_samples / max(k, 1):.1f}% of samples "
          f"(train_tcn.py skips these moments + 0.25 s after each)")
    for (b1, b2), n in sorted(contact_pairs.items(), key=lambda kv: -kv[1])[:6]:
        print(f"    {b1} <-> {b2}: {100 * n / max(k, 1):.1f}% of samples")
    if np.percentile(qd_deg, 95) < 30:
        print("  WARNING: the arm moves slowly - the position actuators may be too weak to follow fast "
              "targets. Data is still valid, but has few fast motions.")

    os.makedirs("data", exist_ok=True)
    np.savez_compressed(args.out, **out)
    meta = dict(rate=real_rate, timestep=dt, gravity=g, minutes=args.minutes, seed=args.seed,
                imus=imus, motions=MOTIONS, joints_q=["shoulder", "elbow"], q_all=list(MOVING),
                acc_units="g", gyr_units="rad/s", tau_units="actuator force (N m)")
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nSaved {args.out} ({k} samples, {k / real_rate / 60:.1f} min) in "
          f"{time.monotonic() - wall0:.0f} s wall time")


def parse_vec(s):
    if s is None:
        return None
    v = [float(x) for x in s.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("expected x,y,z")
    return v


def main() -> int:
    p = argparse.ArgumentParser(description="Generate simulated IMU training data with MuJoCo.")
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="folder containing scene.xml")
    p.add_argument("--scene", default="scene.xml")
    p.add_argument("--minutes", type=float, default=20.0, help="simulated minutes")
    p.add_argument("--rate", type=float, default=200.0, help="logging rate in Hz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=os.path.join("data", "sim_imu.npz"))
    p.add_argument("--view", action="store_true", help="show the MuJoCo viewer (real-time speed)")
    p.add_argument("--upper-pos", type=parse_vec, help="IMU A position on its link, x,y,z (m)")
    p.add_argument("--fore-pos", type=parse_vec, help="IMU B position on its link, x,y,z (m)")
    p.add_argument("--no-xml", action="store_true", help="do not write scene_with_imus.xml")
    p.add_argument("--limit", action="append", metavar="NAME:MIN:MAX",
                   help="exact range a joint moves in, degrees, e.g. --limit shoulder:-36:110 "
                        "(repeat for several joints)")
    args = p.parse_args()
    generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
