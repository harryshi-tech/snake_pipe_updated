"""Controller implementations for snake_pipe.

This package is intentionally lightweight and ROS-agnostic.

Controllers produce :class:`snake_bullet.sim_env.JointCommand` given the current
:class:`snake_bullet.sim_env.RobotState`.
"""

from .gait_position import GaitPositionController, GaitPositionCfg
from .sbc_position import SBCPositionController, SBCPositionCfg
from .closed_loop_sine import ClosedLoopSineController, ClosedLoopSineCfg
from .position_open_loop import OpenLoopPositionController, OpenLoopPositionCfg
from .factory import create_controller

__all__ = [
    "GaitPositionController",
    "GaitPositionCfg",
    "SBCPositionController",
    "SBCPositionCfg",
    "ClosedLoopSineController",
    "ClosedLoopSineCfg",
    "OpenLoopPositionController",
    "OpenLoopPositionCfg",
    "create_controller",
]
