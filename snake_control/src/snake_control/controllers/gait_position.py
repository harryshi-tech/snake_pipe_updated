# snake_control/src/snake_control/controllers/gait_position.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import json

from snake_bullet.sim_env import RobotState, JointCommand
from snake_control.gaitlib.runner import GaitRunner
from snake_control.controllers.base import BaseController
from snake_control.sensors.sim_imu import pole_direction_sign_from_imu


def _sign(x: float) -> float:
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


@dataclass
class GaitPositionCfg:
    # Which gait to run
    gait: str = "rolling_helix"

    # YAML + gaitlib configuration
    snake_type: str = "SEA"  # "SEA" | "REU" | "RSNAKE"
    params_yaml: str = "snake_control/param/snake_params.yaml"

    # Per-run param overrides for this gait (merged over YAML defaults)
    gait_params: Dict[str, Any] = field(default_factory=dict)

    # Pole-climb param-group overrides (merged over YAML pole_climb group)
    pole_params: Dict[str, Any] = field(default_factory=dict)

    # Lab-style semantics
    use_snake_time: bool = True
    transition_time: float = 1.5
    freeze_time_for_headlook: bool = True

    # Sim IMU assist: flip pole_direction when the robot is inverted (top-view left/right).
    imu_correct_pole_direction: bool = True
    imu_pole_deadband: float = 0.2

    # Only required if you want to run head_look_ik
    robot_model: Optional[Any] = None


class GaitPositionController(BaseController):
    """
    Position controller that executes gaits with lab-style semantics:

      - internal snake_time:
          snake_dt = headlook_multiplier * sign(pole_direction) * speed_multiplier * dt
          snake_time += snake_dt
      - smooth transition blending on gait change:
          q_cmd = q_start + (q_des - q_start) * alpha, alpha ramps 0->1 over transition_time
      - special input injection:
          rolling_in_shape: current_angles = q_start (frozen at gait switch)
          head_look:        current_angles = measured q (live)
          head_look_ik:     current_angles = measured q + robot=model
      - provides pole_params (merged from YAML pole_climb + overrides) if gait accepts it

    Preserves *all* gaits; no whitelist.
    """

    def __init__(self, cfg: GaitPositionCfg):
        self.cfg = cfg
        self.runner = GaitRunner(
            snake_type=str(cfg.snake_type).upper(),
            params_yaml=str(cfg.params_yaml),
            shape_model=None,
        )

        self._snake_time: float = 0.0
        self._current_gait: str = str(cfg.gait)
        self._transition_progress: float = 1.0
        self._start_joint_angles: Optional[List[float]] = None

        self._last_meta: Dict[str, Any] = {}

        # Latched correction from simulated IMU (set on teleop 'pole_direction')
        self._imu_pole_sign: float = 1.0

        # Lightweight debug payload for logging
        self._dbg: Dict[str, Any] = {}

    def reset(self, state: RobotState) -> None:
        self._snake_time = 0.0
        self._transition_progress = 1.0
        self._start_joint_angles = list(state.q)

    def on_teleop(self, teleop_cmd: Any, state: Optional[RobotState] = None) -> None:
        """Latch a simulated-IMU correction for pole_direction.

        In pole-climb mode, the operator sets wrapping direction via the 'pole_direction' teleop command.
        On the real robot, IMU orientation helps keep this consistent with a global top-view notion of left/right.
        In simulation, we emulate that by reading the base quaternion from RobotState (filled by SimEnv).
        """
        if not bool(self.cfg.imu_correct_pole_direction):
            return
        if str(teleop_cmd) == "pole_direction" and state is not None:
            self._imu_pole_sign = float(
                pole_direction_sign_from_imu(state, deadband=float(self.cfg.imu_pole_deadband))
            )

    def set_gait(self, gait_name: str, state: RobotState, transition_override: bool = False) -> None:
        gait_name = str(gait_name).strip()
        if gait_name != self._current_gait:
            self._current_gait = gait_name
            self._start_joint_angles = list(state.q)
            self._transition_progress = 0.0
        if transition_override:
            self._transition_progress = 1.0

    @property
    def gait_name(self) -> str:
        return self._current_gait

    @property
    def snake_time(self) -> float:
        return float(self._snake_time)

    @property
    def last_meta(self) -> Dict[str, Any]:
        return dict(self._last_meta)

    def debug(self) -> Dict[str, Any]:
        """Small debug dict (numeric + vectors) for logging/printing."""
        return dict(self._dbg)

    def _merged_pole_params(self) -> Dict[str, Any]:
        # Merge YAML pole_climb group + cfg.pole_params overrides
        defaults = self.runner.gaitlib.default_gait_params.get("pole_climb", {}) or {}
        pole = dict(defaults)
        pole.update(self.cfg.pole_params or {})
        return pole

    @staticmethod
    def _pretty(obj: Any) -> str:
        return json.dumps(obj, indent=2, sort_keys=True, default=str)

    def print_param_summary(self) -> None:
        if not self._last_meta:
            print("[gait_position] No meta yet (call step() once).")
            return
        print(f"\n[gait_position] --- param summary (gait={self._current_gait}, snake_time={self._snake_time:.3f}) ---")
        print("[gait_position] defaults:\n", self._pretty(self._last_meta.get("defaults", {})))
        print("[gait_position] params_merged:\n", self._pretty(self._last_meta.get("params_merged", {})))
        print("[gait_position] params_used:\n", self._pretty(self._last_meta.get("params_used", {})))
        if self._last_meta.get("autofilled", {}):
            print("[gait_position] autofilled:\n", self._pretty(self._last_meta.get("autofilled", {})))
        print("[gait_position] --------------------------------------------------------------\n")

    def step(self, state: RobotState) -> JointCommand:
        if self._start_joint_angles is None:
            self.reset(state)

        dt = float(state.dt)

        # Extra injections (runner only passes to gait if gait signature accepts them)
        extra: Dict[str, Any] = {"pole_params": self._merged_pole_params()}

        if self._current_gait == "rolling_in_shape":
            extra["current_angles"] = list(self._start_joint_angles)
        elif self._current_gait == "head_look":
            extra["current_angles"] = list(state.q)
        elif self._current_gait == "head_look_ik":
            extra["current_angles"] = list(state.q)
            if self.cfg.robot_model is None:
                raise RuntimeError("head_look_ik requested but cfg.robot_model is None.")
            extra["robot"] = self.cfg.robot_model

        # Evaluate gait at snake_time (lab) or sim time
        t_eval = float(self._snake_time) if self.cfg.use_snake_time else float(state.t)

        # Apply simulated-IMU correction for pole_direction (latched via on_teleop)
        overrides: Dict[str, Any] = {} if self.cfg.gait_params is None else dict(self.cfg.gait_params)
        if bool(self.cfg.imu_correct_pole_direction) and "pole_direction" in overrides:
            try:
                pd = float(overrides.get("pole_direction", 0.0))
            except Exception:
                pd = 0.0
            if pd != 0.0:
                overrides["pole_direction"] = float(pd) * float(self._imu_pole_sign)


        out = self.runner.step(
            self._current_gait,
            t=t_eval,
            overrides=overrides,
            extra=extra,
            include_shape=False,
        )
        q_des = [float(x) for x in out.q.tolist()]
        self._last_meta = dict(out.meta)

        # Update snake_time using lab rule
        if self.cfg.use_snake_time:
            merged = out.meta.get("params_merged", {}) or {}
            wave_direction = float(merged.get("speed_multiplier", 1.0))
            pole_direction = _sign(float(merged.get("pole_direction", 1.0)))

            headlook_multiplier = 1.0
            if self.cfg.freeze_time_for_headlook and self._current_gait == "head_look":
                headlook_multiplier = 0.0

            self._snake_time += headlook_multiplier * pole_direction * wave_direction * dt

        # Transition blending (lab rule: head_look bypasses blending)
        if self._current_gait == "head_look":
            q_cmd = q_des
        else:
            a = float(self._transition_progress)
            q_cmd = [
                float(q0) + (float(qd) - float(q0)) * a
                for q0, qd in zip(self._start_joint_angles, q_des)
            ]

        # Advance transition alpha
        T = max(1e-6, float(self.cfg.transition_time))
        self._transition_progress = min(1.0, self._transition_progress + abs(dt) / T)

        # Debug payload (used by CSV logger)
        merged = (out.meta or {}).get("params_merged", {}) or {}
        self._dbg = {
            "gait_name": str(self._current_gait),
            "snake_time": float(self._snake_time),
            "q_nom": list(q_des),
            "q_cmd": list(q_cmd),
            "speed_multiplier": float(merged.get("speed_multiplier", 0.0)),
            "tightness": float(merged.get("tightness", 0.0)),
            "pole_direction": float(merged.get("pole_direction", 0.0)),
            "wt_direction": float(merged.get("wt_direction", merged.get("wt_dir", 0.0))),
        }

        return JointCommand(mode="position", position=q_cmd, effort=None)
