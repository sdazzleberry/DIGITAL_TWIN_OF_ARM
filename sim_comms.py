#!/usr/bin/env python3
"""
sim_comms.py - Joystick comms for the cycloidal arm MuJoCo simulation.

Simulation twin of the real arm's `joy_chassis_arm.py`. It reads the same
joystick with the same mapping, builds the same `msg_arm` command dictionary
that the hardware receives on /arm_client, and applies it to the MuJoCo model
instead of publishing it over ROS 2.

    Joystick -> ArmCommand (== msg_arm) -> position-target integration -> MuJoCo

Input map (Logitech Extreme 3D Pro)
-----------------------------------
    Axis 1   Stick Y          Command  'w' / 's' / 'c'  -> drives SHOULDER + ELBOW together
                              (the two cycloidal joints; whole arm reaches forward/back).
                              Stick forward = arm forward. Joint list: SimConfig.command_joints
    Axis 2   Stick twist      Base yaw direction
                              base_input = axis2 + BASE_TWIST_OFFSET, then a
                              +/-BASE_TWIST_THRESHOLD deadband decides direction
    Axis 3   Throttle slider  Base yaw speed, 0..224
                              speed = (1 - axis3) * 112   (slider back = fast)
    Btn 2 / 4                 Shoulder  +  / -
    Btn 5 / 3                 Elbow     +  / -
    Hat 0 up/down             Wrist pitch  + / -
    Hat 0 left/right          Wrist roll   (msg roll = -hat_x, as on hardware)
    Btn 1 / 0                 Gripper open / close   (swapped vs. the real arm)
    Btn 11                    Reset arm to the "home" keyframe (sim only)

Usage
-----
    python3 sim_comms.py                     run the simulation
    python3 sim_comms.py --dir <folder>      folder containing scene.xml
    python3 sim_comms.py --diagnose          print raw axes/buttons/hats (no sim)
    python3 sim_comms.py --verbose           print every command message
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

# Keep pygame headless so it does not open a second window beside the viewer.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

log = logging.getLogger("sim_comms")


# =============================================================================
# Configuration
# =============================================================================
@dataclass(frozen=True)
class JoystickMap:
    """Every joystick input used by the arm, by index. Edit here only."""

    device_name: str = "logitech extreme 3d pro"   # preferred device (case-insensitive)

    # ---- Axes ---------------------------------------------------------------
    axis_command: int = 1          # Stick Y        -> 'w' / 's' / 'c'
    axis_base_twist: int = 2       # Stick twist    -> base yaw DIRECTION
    axis_throttle: int = 3         # Throttle slider-> base yaw SPEED

    # ---- Buttons ------------------------------------------------------------
    btn_shoulder_fwd: int = 2
    btn_shoulder_rev: int = 4
    btn_elbow_fwd: int = 5
    btn_elbow_rev: int = 3
    btn_gripper_open: int = 1      # swapped: button 1 now OPENS
    btn_gripper_close: int = 0     # swapped: button 0 now CLOSES
    btn_reset: int = 11            # simulation only

    # ---- Hat (D-pad) --------------------------------------------------------
    hat_wrist: int = 0             # x -> roll, y -> pitch

    # ---- Axis conditioning (identical to the real comms file) ---------------
    base_twist_offset: float = 0.2     # added to axis 2 (compensates twist drift)
    base_twist_threshold: float = 0.5  # deadband on (axis2 + offset)
    command_threshold: float = 0.5     # deadband on axis 1
    throttle_gain: float = 112.0       # speed = (1 - axis3) * gain  -> 0..224
    throttle_fallback: float = 112.0   # speed used if the stick has no axis 3


@dataclass(frozen=True)
class CommandJoint:
    """One joint driven by the command axis ('w' / 's' / 'c')."""

    actuator: str          # key into ArmSimulator.ids ("shoulder", "elbow", ...)
    sign: int = +1         # +1: 'w' raises the target, -1: 'w' lowers it
    rate: float = 1.2      # rad/s while the stick is held


@dataclass(frozen=True)
class SimConfig:
    """Simulation tuning: speeds, directions and limits."""

    scene_file: str = "scene.xml"
    keyframe: int = 0
    fps: float = 60.0

    # Rates while a command is held (rad/s; gripper in m/s).
    # 1.2 rad/s == the 0.02 rad/frame @ 60 Hz used by launch_urdf.py.
    shoulder_rate: float = 1.2
    elbow_rate: float = 1.2
    pitch_rate: float = 1.2
    roll_rate: float = 1.2
    gripper_rate: float = 0.03
    base_max_rate: float = 1.5         # rad/s at full throttle (speed 224)
    base_speed_max: float = 224.0

    # Command axis (stick Y): drives the shoulder and elbow together, like the real arm.
    #   'w' = axis > +threshold (stick pulled back on a standard driver)  -> arm back
    #   's' = axis < -threshold (stick pushed forward)                    -> arm forward
    # A NEGATIVE shoulder or elbow ctrl swings the arm forward (checked against the model), so
    # both joints use the same sign. To add/remove a joint or change its speed/direction, edit
    # this table only - flip a joint's `sign` if it moves the wrong way.
    command_joints: tuple = (
        CommandJoint("shoulder", sign=+1, rate=1.2),
        CommandJoint("elbow", sign=+1, rate=1.2),
    )

    # Direction correction per joint (flip to -1 if a joint moves backwards).
    shoulder_sign: int = +1
    elbow_sign: int = -1               # elbow BUTTONS (5/3) only - reversed; command axis is separate
    pitch_sign: int = +1
    roll_sign: int = +1
    gripper_sign: int = +1
    base_sign: int = -1                # matches launch_urdf.py (ctrl -= axis)


# =============================================================================
# Command message
# =============================================================================
@dataclass
class ArmCommand:
    """Identical fields to the `msg_arm` published by the real arm comms file."""

    command: str = "c"
    shoulder: int = 0
    elbow: int = 0
    pitch: int = 0
    roll: int = 0
    gripper: int = 0
    base: int = 0

    def is_idle(self) -> bool:
        return (self.command == "c" and not any(
            (self.shoulder, self.elbow, self.pitch, self.roll, self.gripper, self.base)))

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# =============================================================================
# Joystick
# =============================================================================
def _ternary(positive: int, negative: int) -> int:
    """Two buttons -> +1 / -1 / 0 (positive wins, as on hardware)."""
    if positive == 1:
        return 1
    if negative == 1:
        return -1
    return 0


def _deadband_sign(value: float, threshold: float) -> int:
    """Analog value -> +1 / -1 / 0 with a deadband."""
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


class ArmJoystick:
    """Owns the pygame joystick: discovery, hot-plug and safe raw access."""

    def __init__(self, mapping: JoystickMap) -> None:
        self.map = mapping
        self.device: Optional[pygame.joystick.JoystickType] = None
        self._last_scan = 0.0

    # ---- connection ---------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self.device is not None

    def scan(self) -> None:
        """(Re)select the arm joystick. Prefers the configured device name,
        otherwise falls back to the first joystick so any external stick works."""
        # NOTE: do NOT quit()/init() the joystick subsystem here. Re-initialising
        # makes pygame emit a fresh JOYDEVICEADDED for every stick, which would
        # trigger another scan and loop forever.
        sticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
        if not sticks:
            self.device = None
            return
        for s in sticks:
            s.init()
            log.info("Joystick found: %s (axes=%d buttons=%d hats=%d)",
                     s.get_name(), s.get_numaxes(), s.get_numbuttons(), s.get_numhats())

        wanted = self.map.device_name.strip().lower()
        self.device = next((s for s in sticks if s.get_name().strip().lower() == wanted), None)
        if self.device is None:
            self.device = sticks[0]
            log.warning("'%s' not found; using '%s' instead.",
                        self.map.device_name, self.device.get_name())
        log.info("Arm joystick: %s", self.device.get_name())

    def poll_connection(self) -> bool:
        """Pump events, handle hot-plug, retry discovery once a second."""
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEADDED:
                # pygame also emits this at startup for sticks we already have;
                # only scan when we actually have no joystick.
                if self.device is None:
                    log.info("Joystick added.")
                    self.scan()
            elif event.type == pygame.JOYDEVICEREMOVED:
                if self.device is not None and event.instance_id == self.device.get_instance_id():
                    log.warning("Joystick disconnected.")
                    self.device = None
        if self.device is None and time.monotonic() - self._last_scan > 1.0:
            self._last_scan = time.monotonic()
            self.scan()
        return self.device is not None

    # ---- safe raw access (out-of-range index -> neutral value, never a crash)
    def axis(self, index: int, default: float = 0.0) -> float:
        d = self.device
        if d is None or not 0 <= index < d.get_numaxes():
            return default
        return d.get_axis(index)

    def button(self, index: int) -> int:
        d = self.device
        if d is None or not 0 <= index < d.get_numbuttons():
            return 0
        return d.get_button(index)

    def hat(self, index: int) -> tuple[int, int]:
        d = self.device
        if d is None or not 0 <= index < d.get_numhats():
            return (0, 0)
        return d.get_hat(index)

    def has_axis(self, index: int) -> bool:
        return self.device is not None and 0 <= index < self.device.get_numaxes()

    # ---- build the command message -----------------------------------------
    def read_command(self) -> ArmCommand:
        if not self.connected:
            return ArmCommand()
        m = self.map

        # Axis 1: command
        cmd_dir = _deadband_sign(self.axis(m.axis_command), m.command_threshold)
        command = {1: "w", -1: "s", 0: "c"}[cmd_dir]

        # Axis 2: base twist -> direction (offset first, then deadband)
        twist = self.axis(m.axis_base_twist) + m.base_twist_offset
        base_dir = _deadband_sign(twist, m.base_twist_threshold)

        # Axis 3: throttle -> speed 0..224 (slider pulled back = faster)
        if self.has_axis(m.axis_throttle):
            base_speed = (-self.axis(m.axis_throttle) + 1.0) * m.throttle_gain
        else:
            base_speed = m.throttle_fallback

        hat_x, hat_y = self.hat(m.hat_wrist)

        return ArmCommand(
            command=command,
            shoulder=_ternary(self.button(m.btn_shoulder_fwd), self.button(m.btn_shoulder_rev)),
            elbow=_ternary(self.button(m.btn_elbow_fwd), self.button(m.btn_elbow_rev)),
            pitch=hat_y,
            roll=-hat_x,
            gripper=_ternary(self.button(m.btn_gripper_open), self.button(m.btn_gripper_close)),
            base=int(base_dir * base_speed),
        )


# =============================================================================
# Simulation
# =============================================================================
class ArmSimulator:
    """Wraps the MuJoCo model and turns ArmCommands into actuator targets."""

    _ACTUATORS = (            # logical name -> (actuator name in XML, fallback index)
        ("base", ("base_act", 0)),
        ("shoulder", ("shoulder_act", 1)),
        ("elbow", ("Elbow_act", 2)),
        ("wrist", ("wrist_act", 3)),
        ("wrist_roll", ("wrist_roll_act", 4)),
        ("gripper", ("gripper_act", 5)),
    )

    def __init__(self, cfg: SimConfig) -> None:
        import mujoco  # imported lazily so --diagnose works without MuJoCo
        self.mj = mujoco
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.scene_file)
        self.data = mujoco.MjData(self.model)
        self.reset()

        self.ids = {}
        for key, (name, fallback) in self._ACTUATORS:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx == -1:
                log.warning("Actuator '%s' not found; falling back to index %d.", name, fallback)
                idx = fallback
            self.ids[key] = idx

        self.steps_per_frame = max(1, round((1.0 / cfg.fps) / self.model.opt.timestep))
        self.frame_dt = self.steps_per_frame * self.model.opt.timestep

        # Per-actuator target limits: ctrlrange if set, else the joint range.
        lim = self.model.actuator_ctrlrange.copy()
        limited = self.model.actuator_ctrllimited.astype(bool)
        for a in range(self.model.nu):
            if not limited[a]:
                jid = self.model.actuator_trnid[a, 0]
                lim[a] = self.model.jnt_range[jid]
        self.ctrl_lo, self.ctrl_hi = lim[:, 0], lim[:, 1]

    def reset(self) -> None:
        self.mj.mj_resetDataKeyframe(self.model, self.data, self.cfg.keyframe)

    def apply(self, cmd: ArmCommand) -> None:
        c, dt, ctrl, ids = self.cfg, self.frame_dt, self.data.ctrl, self.ids
        ctrl[ids["shoulder"]] += c.shoulder_sign * cmd.shoulder * c.shoulder_rate * dt
        ctrl[ids["elbow"]] += c.elbow_sign * cmd.elbow * c.elbow_rate * dt
        ctrl[ids["wrist"]] += c.pitch_sign * cmd.pitch * c.pitch_rate * dt
        ctrl[ids["wrist_roll"]] += c.roll_sign * cmd.roll * c.roll_rate * dt
        ctrl[ids["gripper"]] += c.gripper_sign * cmd.gripper * c.gripper_rate * dt
        # Command axis: 'w' -> +1, 's' -> -1, 'c' -> 0, applied to every joint in the table.
        cdir = {"w": 1, "s": -1, "c": 0}.get(cmd.command, 0)
        if cdir:
            for j in c.command_joints:
                ctrl[ids[j.actuator]] += j.sign * cdir * j.rate * dt
        # cmd.base is a signed speed in -224..+224 -> rad/s
        ctrl[ids["base"]] += (c.base_sign * (cmd.base / c.base_speed_max)
                              * c.base_max_rate * dt)
        np.clip(ctrl, self.ctrl_lo, self.ctrl_hi, out=ctrl)

    def step(self) -> None:
        for _ in range(self.steps_per_frame):
            self.mj.mj_step(self.model, self.data)


# =============================================================================
# Run modes
# =============================================================================
def run_simulation(mapping: JoystickMap, cfg: SimConfig, verbose: bool) -> None:
    import mujoco.viewer

    joy = ArmJoystick(mapping)
    joy.scan()
    if not joy.connected:
        log.warning("No joystick detected yet - plug it in, I'll keep looking.")

    sim = ArmSimulator(cfg)
    reset_was_down = False
    last_status = 0.0

    log.info("Launching MuJoCo viewer. Close the window or press Ctrl+C to quit.")
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        viewer.cam.distance = 2.2
        viewer.cam.lookat = [0.15, 0.0, 0.55]
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 140.0

        while viewer.is_running():
            t0 = time.monotonic()

            joy.poll_connection()
            cmd = joy.read_command()

            reset_down = joy.button(mapping.btn_reset) == 1
            if reset_down and not reset_was_down:
                sim.reset()
                log.info("Arm reset to home.")
            reset_was_down = reset_down

            if verbose:
                print(cmd.to_json(), flush=True)
            elif not cmd.is_idle() and t0 - last_status > 0.25:
                last_status = t0
                log.info("%s", cmd)

            sim.apply(cmd)
            sim.step()
            viewer.sync()

            remaining = 1.0 / cfg.fps - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


def run_diagnose(mapping: JoystickMap) -> None:
    """Live raw dump of every axis/button/hat, flagging the ones this file uses.
    Use it to confirm axes 2 (twist) and 3 (throttle) on your actual stick."""
    joy = ArmJoystick(mapping)
    joy.scan()
    print("\nDiagnose mode - move the stick. Ctrl+C to quit.\n"
          f"  Axis {mapping.axis_command}: command   "
          f"Axis {mapping.axis_base_twist}: base twist   "
          f"Axis {mapping.axis_throttle}: throttle\n")
    labels = {mapping.axis_command: "CMD", mapping.axis_base_twist: "TWIST",
              mapping.axis_throttle: "THROTTLE"}
    try:
        while True:
            if not joy.poll_connection():
                print("\rNo joystick...", end="", flush=True)
                time.sleep(0.2)
                continue
            d = joy.device
            axes = "  ".join(
                f"A{i}{'(' + labels[i] + ')' if i in labels else ''}={d.get_axis(i):+.2f}"
                for i in range(d.get_numaxes()))
            pressed = [str(i) for i in range(d.get_numbuttons()) if d.get_button(i)]
            hats = [d.get_hat(i) for i in range(d.get_numhats())]
            cmd = joy.read_command()
            print(f"\r{axes} | btn={','.join(pressed) or '-'} | hat={hats} "
                  f"| base={cmd.base:+4d} cmd={cmd.command}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()


# =============================================================================
# Entry point
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Joystick comms for the cycloidal arm simulation.")
    p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="folder containing scene.xml (default: this script's folder)")
    p.add_argument("--diagnose", action="store_true",
                   help="print raw joystick axes/buttons/hats and exit with Ctrl+C")
    p.add_argument("--verbose", action="store_true",
                   help="print every command message as JSON")
    p.add_argument("--fps", type=float, default=SimConfig.fps, help="viewer/control rate")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    pygame.display.init()
    pygame.joystick.init()
    mapping = JoystickMap()

    try:
        if args.diagnose:
            run_diagnose(mapping)
        else:
            os.chdir(os.path.expanduser(args.dir))
            run_simulation(mapping, SimConfig(fps=args.fps), args.verbose)
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as exc:
        log.error("Could not load the model: %s (use --dir to point at scene.xml's folder)", exc)
        return 1
    finally:
        pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())