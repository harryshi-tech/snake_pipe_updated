# snake_control/src/snake_control/gaitlib/runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import inspect
import numpy as np

from snake_control.gaitlib.reu_gaits import ReuGaits
from snake_control.gaitlib.sea_gaits import SeaGaits
from snake_control.gaitlib.rsnake_gaits import RsnakeGaits


@dataclass
class GaitOutput:
    q: np.ndarray                 # (N,)
    meta: Dict[str, Any]          # params_used, defaults, overrides, etc.


class GaitRunner:
    """
    Unified wrapper: YAML defaults + gait methods -> desired joint angles q_des.

    SBC-ready features:
      - optional shape_model to compute sigma0 and J from q_nom
      - meta contains params_used (true used params if gait supports compute=False)
      - meta contains 'autofilled' if we injected missing keys for robustness
    """

    _TYPE_TO_CLASS = {
        "REU": ReuGaits,
        "SEA": SeaGaits,
        "RSNAKE": RsnakeGaits,
    }

    # YAML keys that are *parameter groups* (not callable gaits)
    _PARAM_GROUP_KEYS = {"pole_climb"}

    def __init__(
        self,
        snake_type: str = "REU",
        params_yaml: Optional[str] = None,
        shape_model: Any = None,   # e.g., SerpenoidLinearShapeModel
    ):
        st = str(snake_type).upper()
        if st not in self._TYPE_TO_CLASS:
            raise ValueError(f"Unknown snake_type '{snake_type}'. Expected one of {list(self._TYPE_TO_CLASS)}")

        self.snake_type = st
        self.gaitlib = self._TYPE_TO_CLASS[st](snake_type=st, params_yaml=params_yaml)
        self.shape_model = shape_model

        # For stateful gaits (e.g., rolling_in_shape), we keep the last q we produced.
        self._last_q: Dict[str, np.ndarray] = {}

    @property
    def n_joints(self) -> int:
        return int(self.gaitlib.num_modules)

    def list_yaml_gaits(self, include_param_groups: bool = False) -> Tuple[str, ...]:
        """List gait names defined in snake_params.yaml.

        By default we exclude non-gait parameter groups (e.g., 'pole_climb').
        """
        keys = list((self.gaitlib.default_gait_params or {}).keys())
        if not include_param_groups:
            keys = [k for k in keys if k not in self._PARAM_GROUP_KEYS]
        return tuple(sorted(keys))

    def list_param_groups(self) -> Tuple[str, ...]:
        """List known parameter-group keys (non-callable YAML sections)."""
        keys = [k for k in (self.gaitlib.default_gait_params or {}).keys() if k in self._PARAM_GROUP_KEYS]
        return tuple(sorted(keys))

    def list_implemented_gaits(self) -> Tuple[str, ...]:
        ignore = {
            "create_gait", "update_params", "parse_params_yaml",
            "flip_axes", "gait_params_filepath",
        }
        names = []
        for name in dir(self.gaitlib):
            if name.startswith("_") or name in ignore:
                continue
            attr = getattr(self.gaitlib, name)
            if callable(attr):
                names.append(name)
        return tuple(sorted(names))

    def _ensure_defaults_exist(self, gait_name: str) -> Dict[str, Any]:
        if self.gaitlib.default_gait_params is None:
            self.gaitlib.default_gait_params = {}
        if gait_name not in self.gaitlib.default_gait_params or self.gaitlib.default_gait_params[gait_name] is None:
            self.gaitlib.default_gait_params[gait_name] = {}
        defaults = self.gaitlib.default_gait_params[gait_name]
        if not isinstance(defaults, dict):
            raise TypeError(f"default gait params for '{gait_name}' must be dict, got {type(defaults)}")
        return defaults

    def _merged_params(self, gait_name: str, overrides: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Return (defaults, merged, autofilled).

        merged = defaults + overrides + autofills (autofills only fill missing keys)
        """
        defaults = self._ensure_defaults_exist(gait_name)
        merged = dict(defaults)
        if overrides:
            merged.update(overrides)

        autofilled: Dict[str, Any] = {}

        # rolling_helix expects these keys in gait_params
        if gait_name == "rolling_helix":
            if "pole_direction" not in merged:
                autofilled["pole_direction"] = 1.0
            if "tightness" not in merged:
                A_even = merged.get("A_even", 0.0)
                autofilled["tightness"] = float(abs(A_even))

        # t_junction expects these in self.current_gait_params AND indexes into params[...] later
        if gait_name == "t_junction":
            if "wt_direction" not in merged:
                autofilled["wt_direction"] = 1.0
            if "pole_direction" not in merged:
                autofilled["pole_direction"] = 1.0
            if "tightness" not in merged:
                A_even = merged.get("A_even", 0.0)
                autofilled["tightness"] = float(abs(A_even))

            # keys referenced as params['...'] inside t_junction.py
            for k, v in {
                "A_1_multiplier": 1.0,
                "A_2_multiplier": 1.0,
                "mu": (self.n_joints - 1) / 2.0,
                "phi_0": 0.0,
                "s_0": 0.4,
            }.items():
                if k not in merged:
                    autofilled[k] = float(v)

        # T-junction component gaits: mirror the same injected params so they can run standalone
        if gait_name in ("windowed_rolling_helix", "spiraling"):
            if "wt_direction" not in merged:
                autofilled["wt_direction"] = 1.0
            if "pole_direction" not in merged:
                autofilled["pole_direction"] = 1.0
            if "tightness" not in merged:
                A_even = merged.get("A_even", 0.0)
                autofilled["tightness"] = float(abs(A_even))

            for k, v in {
                "A_1_multiplier": 1.0,
                "A_2_multiplier": 1.0,
                "mu": (self.n_joints - 1) / 2.0,
                "phi_0": 0.0,
                "s_0": 0.4,
            }.items():
                if k not in merged:
                    autofilled[k] = float(v)

        # Apply autofills (do not overwrite explicit overrides)
        for k, v in autofilled.items():
            merged.setdefault(k, v)

        return defaults, merged, autofilled

    def _auto_extra(self, gait_name: str, merged_params: Dict[str, Any], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Auto-inject common extra args based on gait signature.

        - pole_params: pulled from YAML 'pole_climb' group
        - current_angles: pulled from last output (or zeros)
        """
        extra2 = {} if extra is None else dict(extra)

        gait_fn = getattr(self.gaitlib, gait_name)
        sig = inspect.signature(gait_fn)

        # pole_params: treat YAML 'pole_climb' as a parameter group
        if "pole_params" in sig.parameters and "pole_params" not in extra2:
            pole_defaults = (self.gaitlib.default_gait_params or {}).get("pole_climb", {})
            extra2["pole_params"] = {} if pole_defaults is None else dict(pole_defaults)

        # current_angles: stateful gaits (rolling_in_shape, head_look, etc.)
        if "current_angles" in sig.parameters and "current_angles" not in extra2:
            last = self._last_q.get(gait_name, None)
            if last is None:
                extra2["current_angles"] = np.zeros(self.n_joints, dtype=float)
            else:
                extra2["current_angles"] = np.asarray(last, dtype=float).copy()

        return extra2

    def _call_gait(self, gait_name: str, t: float, params_to_pass: Dict[str, Any], extra: Dict[str, Any]):
        if not hasattr(self.gaitlib, gait_name):
            raise AttributeError(f"Gait '{gait_name}' is not implemented as a method on {type(self.gaitlib).__name__}")

        gait_fn = getattr(self.gaitlib, gait_name)
        sig = inspect.signature(gait_fn)

        kwargs = {}
        if "t" in sig.parameters:
            kwargs["t"] = t

        if "params" in sig.parameters:
            kwargs["params"] = dict(params_to_pass)

        # pass only what the gait accepts (pole_params, current_angles, compute, etc.)
        for k, v in (extra or {}).items():
            if k in sig.parameters:
                kwargs[k] = v

        return gait_fn(**kwargs)

    def _get_params_used(self, gait_name: str, t: float, merged: Dict[str, Any], extra: Dict[str, Any]):
        """If gait supports compute=False and returns gait_params, use it. Else return merged."""
        extra2 = dict(extra or {})
        extra2["compute"] = False
        try:
            params_used = self._call_gait(gait_name, t=t, params_to_pass=merged, extra=extra2)
            if isinstance(params_used, dict):
                return params_used
        except Exception:
            pass
        return merged

    def step(
        self,
        gait_name: str,
        t: float,
        overrides: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        apply_flip_axes: bool = False,
        include_shape: bool = True,
    ) -> GaitOutput:
        gait_name = str(gait_name)
        defaults, merged, autofilled = self._merged_params(gait_name, overrides)

        extra2 = self._auto_extra(gait_name=gait_name, merged_params=merged, extra=extra)

        # 1) call gait
        try:
            q = self._call_gait(gait_name, t=t, params_to_pass=merged, extra=extra2)
        except KeyError as e:
            missing = str(e)
            raise KeyError(
                f"Gait '{gait_name}' raised KeyError {missing}. "
                f"Likely missing key in snake_params.yaml defaults for '{self.snake_type}->{gait_name}'."
            ) from e

        q = np.asarray(q, dtype=float).reshape(-1)
        if q.size != self.n_joints:
            raise ValueError(f"Gait '{gait_name}' returned length {q.size}, expected {self.n_joints}")
        if not np.all(np.isfinite(q)):
            raise ValueError(f"Gait '{gait_name}' produced NaN/Inf at t={t}")

        # 2) optional legacy sign flipping
        if apply_flip_axes:
            q2 = np.asarray(q, dtype=float)[None, :]
            q2 = self.gaitlib.flip_axes(q2)
            q = q2[0, :]

        # 3) meta
        params_used = self._get_params_used(gait_name, t=t, merged=merged, extra=extra2)

        meta: Dict[str, Any] = {
            "snake_type": self.snake_type,
            "gait_name": gait_name,
            "t": float(t),
            "defaults": dict(defaults),
            "overrides": {} if overrides is None else dict(overrides),
            "params_merged": dict(merged),
            "params_used": dict(params_used) if isinstance(params_used, dict) else params_used,
            "autofilled": dict(autofilled),
        }

        # 4) SBC-ready shape projection
        if include_shape and (self.shape_model is not None):
            try:
                meta["sigma0"] = self.shape_model.project(q)
                meta["J_shape"] = self.shape_model.jacobian()
            except Exception as e:
                meta["shape_error"] = repr(e)

        # keep last output for stateful gaits
        self._last_q[gait_name] = q.copy()

        return GaitOutput(q=q, meta=meta)
