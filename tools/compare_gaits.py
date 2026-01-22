#!/usr/bin/env python3
"""Compare gait waveforms between the reference and snake_pipe_updated implementations.

This tool is intentionally ROS-agnostic and does *not* import reference code at runtime
outside of this script. Reference imports are done by temporarily modifying sys.path.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import types

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
REF_ROOT = REPO_ROOT / "references" / "snakes_on_pipes-main" / "snakelib_control" / "src"
TARGET_ROOT = REPO_ROOT / "references" / "snake_pipe_updated" / "snake_control" / "src"


@dataclass
class GaitSpec:
    name: str
    param_sets: List[Dict[str, float]]
    pole_params: Dict[str, float]


class GaitShim:
    def __init__(self, default_gait_params: Dict[str, Dict[str, float]], snake_type: str = "REU", num_modules: int = 16):
        self.default_gait_params = default_gait_params
        self.snake_type = snake_type
        self.num_modules = num_modules
        self.current_gait_params = {}
        self.current_gait = None

    @staticmethod
    def update_params(params_dict: Dict[str, float], params_to_update: Dict[str, float]) -> Dict[str, float]:
        merged = dict(params_dict or {})
        merged.update(params_to_update or {})
        return merged


def _bind_method(obj: GaitShim, func: Callable) -> None:
    setattr(obj, func.__name__, types.MethodType(func, obj))


def _with_sys_path(path: Path):
    class _PathCtx:
        def __enter__(self):
            sys.path.insert(0, str(path))
            return self

        def __exit__(self, exc_type, exc, tb):
            if str(path) in sys.path:
                sys.path.remove(str(path))

    return _PathCtx()


def build_reference_shim() -> GaitShim:
    with _with_sys_path(REF_ROOT):
        from snakelib_control.gaitlib.reu_gaits import compound_serpenoid as ref_compound
        from snakelib_control.gaitlib.reu_gaits import rolling_helix as ref_rolling_helix
        from snakelib_control.gaitlib.reu_gaits import t_junction as ref_t_junction

    shim = GaitShim(default_gait_params={})
    _bind_method(shim, ref_compound.compound_serpenoid)
    _bind_method(shim, ref_rolling_helix.rolling_helix)
    _bind_method(shim, ref_t_junction.t_junction)
    _bind_method(shim, ref_t_junction.gaussian_window)
    _bind_method(shim, ref_t_junction.sinus_window)
    _bind_method(shim, ref_t_junction.amplitude_reduced)
    _bind_method(shim, ref_t_junction.amplitude_reduced_sinus)
    _bind_method(shim, ref_t_junction.parameter_windowed)
    _bind_method(shim, ref_t_junction.exp_window)
    return shim


def build_target_shim() -> GaitShim:
    with _with_sys_path(TARGET_ROOT):
        from snake_control.gaitlib.reu_gaits import compound_serpenoid as tgt_compound
        from snake_control.gaitlib.reu_gaits import rolling_helix as tgt_rolling_helix
        from snake_control.gaitlib.reu_gaits import t_junction as tgt_t_junction
        from snake_control.gaitlib.reu_gaits import spiraling as tgt_spiraling
        from snake_control.gaitlib.reu_gaits import windowed_rolling_helix as tgt_windowed

    shim = GaitShim(default_gait_params={})
    _bind_method(shim, tgt_compound.compound_serpenoid)
    _bind_method(shim, tgt_rolling_helix.rolling_helix)
    _bind_method(shim, tgt_t_junction.t_junction)
    _bind_method(shim, tgt_spiraling.spiraling)
    _bind_method(shim, tgt_windowed.windowed_rolling_helix)
    _bind_method(shim, tgt_t_junction.gaussian_window)
    _bind_method(shim, tgt_t_junction.sinus_window)
    _bind_method(shim, tgt_t_junction.amplitude_reduced)
    _bind_method(shim, tgt_t_junction.amplitude_reduced_sinus)
    _bind_method(shim, tgt_t_junction.parameter_windowed)
    _bind_method(shim, tgt_t_junction.exp_window)
    return shim


def _reference_windowed_rolling_helix(shim: GaitShim, t: float, params: Dict[str, float], pole_params: Dict[str, float]) -> np.ndarray:
    """Extract the pre-blend (windowed rolling helix) portion from the legacy t_junction math."""
    gait_params = shim.update_params(params, {})
    A_transition = pole_params.get("A_transition", 0.35)
    A_max = pole_params.get("A_max", 1.25)
    dWs_dAodd = pole_params.get("dWs_dAodd", 2.5 / 0.75)

    A_even = gait_params["A_even"]
    wS_even = gait_params["wS_even"]
    wT_even = gait_params["wT_even"]
    tightness = gait_params["tightness"]
    pole_direction = gait_params["pole_direction"]

    wS_max = wS_even
    A_min = A_even
    if tightness < A_transition:
        wS_odd = 0.0
    else:
        wS_odd = min(wS_max, (tightness - A_transition) * dWs_dAodd)

    wS_odd *= -pole_direction

    if tightness < A_min:
        A_odd = A_min
    else:
        A_odd = min(tightness, A_max)

    A_odd *= -pole_direction

    wS_even = wS_odd
    A_even = A_odd

    A_1_multiplier = params["A_1_multiplier"]
    A_2_multiplier = params["A_2_multiplier"]
    mu = params["mu"]
    phi_0 = params["phi_0"]
    m = params["m"]
    sig = params["sig"]

    A_1 = A_even * A_1_multiplier
    A_2 = A_even * A_2_multiplier
    A_set = [A_1, A_2]
    wS_set = [wS_even, wS_even]

    target_angles = np.zeros(shim.num_modules)
    for i in range(shim.num_modules):
        offset = np.pi if (i % 2 == 0) else (-np.pi / 2)
        offset_hook = np.sin(phi_0 + wS_even * i + wT_even * t + offset)
        target_angles[i] = shim.amplitude_reduced(i, A_set, m, mu, sig) * np.sin(
            shim.parameter_windowed(i, wS_set, mu, m) * i + wT_even * t + offset
        ) + offset_hook * A_even * shim.gaussian_window(i / 15, mu / 15, sig)
        target_angles[i] = min(max(target_angles[i], -np.pi / 2), np.pi / 2)

    target_angles[2::4] *= -1
    target_angles[3::4] *= -1
    return target_angles


def _evaluate_gait_series(
    eval_fn: Callable[[float], np.ndarray],
    t_grid: np.ndarray,
) -> np.ndarray:
    out = np.zeros((len(t_grid), len(eval_fn(t_grid[0]))), dtype=float)
    for i, t in enumerate(t_grid):
        out[i, :] = eval_fn(float(t))
    return out


def _evaluate_compound_series(shim: GaitShim, params: Dict[str, float], t_grid: np.ndarray) -> np.ndarray:
    out = np.zeros((len(t_grid), shim.num_modules), dtype=float)
    for i, t in enumerate(t_grid):
        for n in range(shim.num_modules):
            out[i, n] = shim.compound_serpenoid(float(t), n, params)
    return out


def _save_csv(out_dir: Path, name: str, t_grid: np.ndarray, data: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["t"] + [f"q_{i}" for i in range(data.shape[1])]
        writer.writerow(header)
        for i, t in enumerate(t_grid):
            writer.writerow([f"{t:.6f}"] + [f"{v:.9f}" for v in data[i]])


def _compare(ref: np.ndarray, tgt: np.ndarray) -> Tuple[float, float]:
    diff = ref - tgt
    max_err = float(np.max(np.abs(diff)))
    rms_err = float(np.sqrt(np.mean(diff ** 2)))
    return max_err, rms_err


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare gaits against the legacy reference implementation.")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "references" / "snake_pipe_updated" / "tools" / "out"))
    parser.add_argument("--no-csv", action="store_true", help="Disable CSV output.")
    args = parser.parse_args()

    t_grid = np.linspace(0.0, 2.0, 201)
    pole_params = {"A_transition": 0.35, "A_max": 1.25, "dWs_dAodd": 2.5 / 0.75}

    specs = [
        GaitSpec(
            name="compound_serpenoid",
            param_sets=[
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.3,
                    "A_odd": 0.3,
                    "wS_even": 0.0,
                    "wS_odd": 0.0,
                    "wT_even": -4.0,
                    "wT_odd": -4.0,
                    "delta": 1.57079632679,
                },
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.6,
                    "A_odd": 0.6,
                    "wS_even": 0.5,
                    "wS_odd": 0.5,
                    "wT_even": 3.5,
                    "wT_odd": 3.5,
                    "delta": 0.78539816339,
                },
            ],
            pole_params=pole_params,
        ),
        GaitSpec(
            name="rolling_helix",
            param_sets=[
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.2,
                    "A_odd": 0.2,
                    "wS_even": 1.2,
                    "wS_odd": 1.2,
                    "wT_even": 1.75,
                    "wT_odd": 1.75,
                    "delta": -1.57079632679,
                    "speed_multiplier": 1.0,
                    "tightness": 0.6,
                    "pole_direction": 1.0,
                },
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.2,
                    "A_odd": 0.2,
                    "wS_even": 1.2,
                    "wS_odd": 1.2,
                    "wT_even": 1.75,
                    "wT_odd": 1.75,
                    "delta": -1.57079632679,
                    "speed_multiplier": 1.0,
                    "tightness": 1.0,
                    "pole_direction": -1.0,
                },
            ],
            pole_params=pole_params,
        ),
        GaitSpec(
            name="t_junction",
            param_sets=[
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.2,
                    "A_odd": 0.2,
                    "wS_even": 14.406,
                    "wS_odd": 14.406,
                    "wT_even": 2.0,
                    "wT_odd": 2.0,
                    "delta": -1.57079632679,
                    "speed_multiplier": 1.0,
                    "tightness": 0.6,
                    "pole_direction": 1.0,
                    "wt_direction": 1.0,
                    "A_1_multiplier": 1.0,
                    "A_2_multiplier": 1.0,
                    "mu": 7.5,
                    "phi_0": 0.0,
                    "s_0": 0.4,
                    "m": 50.0,
                    "sig": 0.05,
                    "T": 0.25,
                },
            ],
            pole_params=pole_params,
        ),
        GaitSpec(
            name="spiraling",
            param_sets=[
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.2,
                    "A_odd": 0.2,
                    "wS_even": 14.406,
                    "wS_odd": 14.406,
                    "wT_even": 2.0,
                    "wT_odd": 2.0,
                    "delta": -1.57079632679,
                    "speed_multiplier": 1.0,
                    "tightness": 0.6,
                    "pole_direction": 1.0,
                    "wt_direction": 1.0,
                    "A_1_multiplier": 1.0,
                    "A_2_multiplier": 1.0,
                    "mu": 7.5,
                    "phi_0": 0.0,
                    "s_0": 0.4,
                    "m": 50.0,
                    "sig": 0.05,
                    "T": 0.25,
                },
            ],
            pole_params=pole_params,
        ),
        GaitSpec(
            name="windowed_rolling_helix",
            param_sets=[
                {
                    "beta_even": 0.0,
                    "beta_odd": 0.0,
                    "A_even": 0.2,
                    "A_odd": 0.2,
                    "wS_even": 14.406,
                    "wS_odd": 14.406,
                    "wT_even": 2.0,
                    "wT_odd": 2.0,
                    "delta": -1.57079632679,
                    "speed_multiplier": 1.0,
                    "tightness": 0.6,
                    "pole_direction": 1.0,
                    "wt_direction": 1.0,
                    "A_1_multiplier": 1.0,
                    "A_2_multiplier": 1.0,
                    "mu": 7.5,
                    "phi_0": 0.0,
                    "s_0": 0.4,
                    "m": 50.0,
                    "sig": 0.05,
                    "T": 0.25,
                },
            ],
            pole_params=pole_params,
        ),
    ]

    ref = build_reference_shim()
    tgt = build_target_shim()
    out_dir = Path(args.out_dir)

    for spec in specs:
        print(f"\n=== {spec.name} ===")
        for i, params in enumerate(spec.param_sets, start=1):
            label = f"{spec.name}_set{i}"
            if spec.name == "compound_serpenoid":
                ref_series = _evaluate_compound_series(ref, params, t_grid)
                tgt_series = _evaluate_compound_series(tgt, params, t_grid)
            elif spec.name == "windowed_rolling_helix":
                ref_series = _evaluate_gait_series(
                    lambda t: _reference_windowed_rolling_helix(ref, t, params, spec.pole_params),
                    t_grid,
                )
                tgt_series = _evaluate_gait_series(
                    lambda t: tgt.windowed_rolling_helix(t=t, params=params, pole_params=spec.pole_params, compute=True),
                    t_grid,
                )
            elif spec.name == "spiraling":
                with redirect_stdout(io.StringIO()):
                    ref_series = _evaluate_gait_series(
                        lambda t: ref.t_junction(t=t, params=params, pole_params=spec.pole_params),
                        t_grid,
                    )
                tgt_series = _evaluate_gait_series(
                    lambda t: tgt.spiraling(t=t, params=params, pole_params=spec.pole_params, compute=True),
                    t_grid,
                )
            else:
                if spec.name == "t_junction":
                    with redirect_stdout(io.StringIO()):
                        ref_series = _evaluate_gait_series(
                            lambda t: getattr(ref, spec.name)(t=t, params=params, pole_params=spec.pole_params),
                            t_grid,
                        )
                else:
                    ref_series = _evaluate_gait_series(
                        lambda t: getattr(ref, spec.name)(t=t, params=params, pole_params=spec.pole_params, compute=True),
                        t_grid,
                    )
                tgt_series = _evaluate_gait_series(
                    lambda t: getattr(tgt, spec.name)(t=t, params=params, pole_params=spec.pole_params, compute=True),
                    t_grid,
                )

            max_err, rms_err = _compare(ref_series, tgt_series)
            print(f"{label}: max_err={max_err:.6e} rms_err={rms_err:.6e}")

            if not args.no_csv:
                _save_csv(out_dir, f"{label}_ref", t_grid, ref_series)
                _save_csv(out_dir, f"{label}_target", t_grid, tgt_series)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
