# snake_control/src/snake_control/controllers/factory.py
# The controller selector/sconstructor based on YAML config.

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Optional

from snake_control.controllers.base import BaseController
from snake_control.controllers.closed_loop_sine import ClosedLoopSineController, ClosedLoopSineCfg
from snake_control.controllers.gait_position import GaitPositionController, GaitPositionCfg
from snake_control.controllers.position_open_loop import OpenLoopPositionController, OpenLoopPositionCfg
from snake_control.controllers.sbc_position import SBCPositionController, SBCPositionCfg


def _dataclass_from_dict(cls, d: Optional[Dict[str, Any]]):
    if d is None:
        return cls()  # type: ignore
    allowed = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in dict(d).items() if k in allowed}
    return cls(**kwargs)  # type: ignore


def create_controller(ctrl_cfg: Dict[str, Any], defaults: Optional[Dict[str, Any]] = None) -> BaseController:
    """Create a controller from config.

    Expected schema:
      controller:
        type: gait_position | closed_loop_sine | position_open_loop
        ... (type-specific fields)
    """
    defaults = {} if defaults is None else dict(defaults)
    cfg = dict(defaults)
    cfg.update(ctrl_cfg or {})

    ctrl_type = str(cfg.get("type", "gait_position")).strip().lower()

    if ctrl_type in ("gait_position", "gait", "position_gait"):
        dc = _dataclass_from_dict(GaitPositionCfg, cfg)
        return GaitPositionController(dc)  # type: ignore[arg-type]

    if ctrl_type in ("sbc_position", "sbc", "sbc_gait", "gait_sbc"):
        dc = _dataclass_from_dict(SBCPositionCfg, cfg)
        return SBCPositionController(dc)  # type: ignore[arg-type]

    if ctrl_type in ("closed_loop_sine", "sine"):
        dc = _dataclass_from_dict(ClosedLoopSineCfg, cfg)
        return ClosedLoopSineController(dc)

    if ctrl_type in ("position_open_loop", "open_loop_position", "position"):
        dc = _dataclass_from_dict(OpenLoopPositionCfg, cfg)
        return OpenLoopPositionController(dc)

    raise ValueError(f"Unknown controller type '{ctrl_type}'.")
