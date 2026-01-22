# snake_control/src/snake_control/controllers/closed_loop_sine.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import math

# We import types from snake_bullet, but keep it lightweight/ROS-agnostic.
from snake_bullet.sim_env import RobotState, JointCommand
from snake_control.controllers.base import BaseController


@dataclass
class ClosedLoopSineCfg:
    amp: float = 0.35          # rad
    freq_hz: float = 0.5       # Hz
    phase_step: float = 0.35   # rad between joints (traveling wave)
    kp_track: float = 0.6      # closed-loop correction gain (unitless)


class ClosedLoopSineController(BaseController):
    """
    Minimal closed-loop "see it move" controller.

    It generates a traveling sinusoid target q_des(j, t),
    then applies a small closed-loop correction:
        q_cmd = q_des + kp_track * (q_des - q_meas)

    This is still position control (through SimEnv.apply_command),
    but it *uses feedback* so you know the loop is closed.
    """

    def __init__(self, cfg: Optional[ClosedLoopSineCfg] = None):
        self.cfg = cfg if cfg is not None else ClosedLoopSineCfg()
        self._t0: Optional[float] = None
        self._n: Optional[int] = None

    def reset(self, n_joints: int, t0: float = 0.0) -> None:
        self._n = int(n_joints)
        self._t0 = float(t0)

    def step(self, state: RobotState) -> JointCommand:
        if self._t0 is None or self._n is None:
            self.reset(n_joints=len(state.q), t0=state.t)

        t = float(state.t - self._t0)

        amp = float(self.cfg.amp)
        w = 2.0 * math.pi * float(self.cfg.freq_hz)
        dphi = float(self.cfg.phase_step)
        kp = float(self.cfg.kp_track)

        q_des: List[float] = []
        q_cmd: List[float] = []

        for j in range(self._n):
            des = amp * math.sin(w * t + dphi * j)
            q_des.append(des)

            # closed-loop correction using measured q
            err = des - float(state.q[j])
            cmd = des + kp * err
            q_cmd.append(cmd)

        return JointCommand(mode="position", position=q_cmd, effort=None)
