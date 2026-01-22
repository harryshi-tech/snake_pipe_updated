# snake_control/src/snake_control/controllers/position_open_loop.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from snake_bullet.sim_env import RobotState, JointCommand
from snake_control.controllers.base import BaseController


@dataclass
class OpenLoopPositionCfg:
    """Config for :class:`OpenLoopPositionController`.

    If ``target`` is None, the controller will default to holding the robot's
    current joint angles on first reset.
    """

    target: Optional[List[float]] = None


class OpenLoopPositionController(BaseController):
    """Pure open-loop position controller.

    It *does not* compute gaits or use feedback. It simply forwards the desired
    joint angles to the simulator every step.
    """

    def __init__(self, cfg: Optional[OpenLoopPositionCfg] = None):
        self.cfg = cfg if cfg is not None else OpenLoopPositionCfg()
        self._q_target: Optional[List[float]] = None

    def reset(self, state: RobotState) -> None:
        if self.cfg.target is None:
            self._q_target = [float(x) for x in state.q]
        else:
            self._q_target = [float(x) for x in self.cfg.target]

    def set_target(self, q_target: List[float]) -> None:
        self._q_target = [float(x) for x in q_target]

    @property
    def target(self) -> Optional[List[float]]:
        return None if self._q_target is None else list(self._q_target)

    def step(self, state: RobotState) -> JointCommand:
        if self._q_target is None:
            self.reset(state)
        return JointCommand(mode="position", position=list(self._q_target), effort=None)
