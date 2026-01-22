#!/usr/bin/env python3
# snake_bullet/run_sim.py
"""Unified simulator runner.

This is the single entrypoint to run the PyBullet simulation.

It is configured by snake_bullet/param/sim_params.yaml:
  - choose controller type and its settings
  - optionally enable joystick teleop (pygame)
  - configure terminal printing

Run (repo root):
  PYTHONPATH=snake_bullet/src:snake_control/src python snake_bullet/run_sim.py
  PYTHONPATH=snake_bullet/src:snake_control/src python snake_bullet/run_sim.py --cfg snake_bullet/param/sim_params.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from snake_bullet.sim_env import SimEnv, JointCommand

from snake_bullet.logging import CSVLogger

from snake_control.controllers import (
    GaitPositionController,
    create_controller,
)
from snake_control.teleop import RosLikeJoystickTeleop


def _repo_root() -> Path:
    # snake_pipe/snake_bullet/run_sim.py -> parents[1] is snake_pipe/
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _resolve_path(repo_root: Path, p: str) -> Path:
    pp = Path(str(p))
    return pp if pp.is_absolute() else (repo_root / pp)


def _pretty_state(t: float, dt: float, extra: str = "") -> str:
    if extra:
        return f"[run_sim] t={t:8.3f}  dt={dt:7.4f}  {extra}"
    return f"[run_sim] t={t:8.3f}  dt={dt:7.4f}"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cfg",
        type=str,
        default="snake_bullet/param/sim_params.yaml",
        help="Path to sim_params.yaml (repo-relative by default)",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = _repo_root()

    sim_cfg_path = _resolve_path(repo_root, args.cfg)
    sim_cfg = _load_yaml(sim_cfg_path)

    # ---- runner config ----
    runner_cfg = sim_cfg.get("runner", {}) or {}
    ctrl_cfg = runner_cfg.get("controller", {}) or {}
    teleop_cfg = runner_cfg.get("teleop", {}) or {}
    print_cfg = runner_cfg.get("print", {}) or {}

    # ---- paths ----
    snake_params_path = _resolve_path(
        repo_root,
        runner_cfg.get("snake_params_yaml", "snake_control/param/snake_params.yaml"),
    )
    snake_params = _load_yaml(snake_params_path)

    snake_type = str(sim_cfg.get("robot", {}).get("snake_type", "SEA")).upper()

    # ---- sim env ----
    env = SimEnv(sim_cfg)
    n = len(env.joint_idx)
    if n == 0:
        raise RuntimeError("No controllable joints found (env.joint_idx is empty).")

    # ---- CSV logging (optional) ----
    # Config lives at top-level: logging: { enable, out_dir, run_name, signals, joints, ... }
    log_cfg = sim_cfg.get("logging", {}) or {}
    logger = CSVLogger(log_cfg, repo_root=repo_root, joint_names=list(env.joint_names))

    # ---- controller ----
    # Provide defaults from sim config, but allow controller section to override.
    controller = create_controller(
        ctrl_cfg,
        defaults={
            "snake_type": snake_type,
            "params_yaml": str(snake_params_path),
        },
    )

    # Some controllers accept reset(state)
    try:
        controller.reset(env.get_state())
    except Exception:
        pass

    # ---- joystick teleop (optional) ----
    teleop: Optional[RosLikeJoystickTeleop] = None
    if bool(teleop_cfg.get("enable", False)):
        joy_to_gait = _resolve_path(
            repo_root,
            teleop_cfg.get("joy_to_gait_yaml", "snake_control/param/joy_to_gait.yaml"),
        )
        joy_mapping = _resolve_path(
            repo_root,
            teleop_cfg.get(
                "joy_mapping_yaml",
                "snake_control/param/logi_cordless_rumblepad2_mappings.yaml",
            ),
        )

        teleop = RosLikeJoystickTeleop(
            snake_type=snake_type,
            snake_params_yaml=snake_params_path,
            joy_to_gait_yaml=joy_to_gait,
            joy_mapping_yaml=joy_mapping,
            joy_index=int(teleop_cfg.get("joy_index", 0)),
            deadzone=float(teleop_cfg.get("deadzone", 0.18)),
            verbose=bool(teleop_cfg.get("verbose", True)),
        )
        print(f"[run_sim] teleop enabled, joy_map={joy_to_gait}, joy_mapping={joy_mapping}")

    # ---- printing ----
    print_state = bool(print_cfg.get("state", True))
    print_state_every = float(print_cfg.get("state_every_s", 1.0))
    print_params = bool(print_cfg.get("params", False))
    print_params_every = float(print_cfg.get("params_every_s", 1.0))
    print_joy = bool(print_cfg.get("joy_active", False))

    # ---- duration ----
    duration_s = float(runner_cfg.get("duration_s", 0.0))  # 0 => run until quit
    t0_wall = time.time()
    last_state_wall = 0.0
    last_params_wall = 0.0

    # These are kept here (not inside controller) so joystick behavior matches the current manner
    # used in snake_bullet/run_gait.py.
    base_gait_overrides: Dict[str, Any] = dict(ctrl_cfg.get("gait_params", {}) or {})
    base_pole_overrides: Dict[str, Any] = dict(ctrl_cfg.get("pole_params", {}) or {})

    # Home pose
    home_pose = snake_params.get(snake_type, {}).get("home", None)

    print(f"[run_sim] snake_type={snake_type}  joints={n}")
    print(f"[run_sim] controller={type(controller).__name__}")
    print("[run_sim] Press Ctrl+C to stop.")

    # ------------------------------------------------------------------
    # HOLD behavior (teleop idle)
    #
    # The ROS-style teleop publishes command_name="hold_position" whenever the
    # operator is not actively issuing a gait command.
    #
    # Previously we held the *current measured* joint angles (q_meas). For pole
    # / pipe climbs this can allow the robot to slowly relax/unwind (reducing
    # normal force) and slide.
    #
    # Instead, we latch the *last commanded* joint angles (q_cmd) and keep
    # commanding that posture until a new gait command arrives.
    # ------------------------------------------------------------------
    last_q_cmd: Optional[list] = None  # last commanded joint positions (len=n)
    q_hold: Optional[list] = None      # latched hold posture while idle
    hold_active: bool = False

    # ---- CSV logging (optional) ----
    log_cfg = sim_cfg.get("logging", {}) or {}
    logger = CSVLogger(log_cfg, repo_root=repo_root, joint_names=list(env.joint_names) if env.joint_names else [f"j{i}" for i in range(n)])

    try:
        while True:
            st = env.get_state()

            # --- teleop update ---
            teleop_cmd_name: Optional[str] = None
            teleop_overrides: Dict[str, Any] = {}
            if teleop is not None:
                cmd = teleop.step(print_joy=print_joy)
                if cmd.quit:
                    break
                teleop_cmd_name = str(cmd.command_name)
                teleop_overrides = cmd.as_param_dict()

            # --- special commands (match existing behavior) ---
            if teleop_cmd_name == "hold_position":
                # Teleop idle: latch the *last commanded* posture and keep sending it.
                # This prevents the snake from relaxing to q_meas and sliding.
                if not hold_active:
                    q_hold = list(last_q_cmd) if last_q_cmd is not None else list(st.q)
                    hold_active = True
                joint_cmd = JointCommand(mode="position", position=list(q_hold) if q_hold is not None else list(st.q), effort=None)

            elif teleop_cmd_name == "home":
                hold_active = False
                q_hold = None
                if isinstance(home_pose, (list, tuple)) and len(home_pose) == n:
                    joint_cmd = JointCommand(mode="position", position=[float(x) for x in home_pose], effort=None)
                else:
                    joint_cmd = JointCommand(mode="position", position=list(st.q), effort=None)

            else:
                hold_active = False
                q_hold = None
                # For gait controller, keep joystick semantics identical to run_gait.py.
                if isinstance(controller, GaitPositionController):
                    # Treat command_name as gait if it exists in YAML defaults.
                    if teleop_cmd_name and teleop_cmd_name in controller.runner.gaitlib.default_gait_params:
                        controller.set_gait(teleop_cmd_name, st)

                    # Priority: YAML defaults (inside gaitlib) < base overrides (sim_params) < teleop overrides
                    controller.cfg.gait_params = dict(base_gait_overrides)
                    controller.cfg.gait_params.update(teleop_overrides)

                    # Pole params group (YAML pole_climb) is merged inside controller.
                    controller.cfg.pole_params = dict(base_pole_overrides)

                # Generic hook
                try:
                    controller.on_teleop(teleop_cmd_name, st)  # type: ignore[arg-type]
                except Exception:
                    pass

                joint_cmd = controller.step(st)

            # Track the last commanded posture for hold behavior.
            if (joint_cmd is not None) and ((joint_cmd.mode or "").lower() == "position") and (joint_cmd.position is not None):
                last_q_cmd = list(joint_cmd.position)

            # --- logging ---
            try:
                dbg = controller.debug()  # type: ignore[assignment]
            except Exception:
                dbg = {}

            # Always add pipe-only contact summary metrics so we can correlate
            # diameter transitions with contact loading (outside-pipe climbs too).
            if not isinstance(dbg, dict):
                dbg = {}
            try:
                dbg.update(env.get_pipe_contact_metrics())
            except Exception:
                pass

            try:
                logger.record(st, joint_cmd, dbg=dbg, teleop_cmd=teleop_cmd_name)
            except Exception:
                pass

            env.apply_command(joint_cmd)
            env.step()

            # --- printing ---
            now = time.time() - t0_wall

            if print_state and (now - last_state_wall) >= print_state_every:
                last_state_wall = now
                extra = ""
                if isinstance(controller, GaitPositionController):
                    extra = f"gait={controller.gait_name}  snake_time={controller.snake_time:7.3f}"
                elif teleop_cmd_name:
                    extra = f"cmd={teleop_cmd_name}"
                print(_pretty_state(float(st.t), float(st.dt), extra=extra))

            if print_params and (now - last_params_wall) >= print_params_every:
                last_params_wall = now
                if isinstance(controller, GaitPositionController):
                    controller.print_param_summary()
                else:
                    dbg = {}
                    try:
                        dbg = controller.debug()  # type: ignore[assignment]
                    except Exception:
                        dbg = {}
                    if dbg:
                        print("[run_sim] controller debug:", dbg)

            # --- duration exit ---
            if duration_s > 0.0 and (now >= duration_s) and (teleop is None):
                break

    except KeyboardInterrupt:
        pass
    finally:
        try:
            logger.close()
        except Exception:
            pass
        env.close()
        print("\n[run_sim] done.")


if __name__ == "__main__":
    main()
