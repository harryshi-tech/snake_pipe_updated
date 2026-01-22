# snake_control/src/snake_control/controllers/base.py
# Defines the minimal controller interface that the simulation runner expects

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from snake_bullet.sim_env import RobotState, JointCommand


class BaseController(ABC):
    """Minimal controller interface used by snake_bullet/run_sim.py."""

    def reset(self, state: RobotState) -> None:  # pragma: no cover
        """Optional reset called once at startup."""

    @abstractmethod
    def step(self, state: RobotState) -> JointCommand:
        """Compute a command given the current robot state."""

    def on_teleop(self, teleop_cmd: Any, state: Optional[RobotState] = None) -> None:  # pragma: no cover
        """Optional hook to consume teleop commands."""

    def debug(self) -> Dict[str, Any]:
        """Optional lightweight debug dict for printing."""
        return {}
