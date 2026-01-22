"""Windowed rolling-helix gait used for T-junction navigation.

This is the *pre-blend* portion of the original `snakes_on_pipes` T-junction gait
(i.e., `target_angles[...]` before mixing in the spiraling component).

Exposed as a standalone gait so you can run:
  --gait windowed_rolling_helix
"""

from copy import deepcopy
import numpy as np

from .utils import flip_axes


def windowed_rolling_helix(self, t=0, params=None, pole_params=None, compute=True):
    params = {} if params is None else params
    pole_params = {} if pole_params is None else pole_params

    self.current_gait = "windowed_rolling_helix"

    # Robust defaults: if YAML doesn't include this gait, fall back to t_junction defaults.
    defaults = (
        self.default_gait_params.get(self.current_gait)
        or self.default_gait_params.get("t_junction")
        or {}
    )
    self.current_gait_params = self.update_params(defaults, params)
    gait_params = deepcopy(self.current_gait_params)

    # Pole-climb shaping constants (same as t_junction)
    A_transition = pole_params.get("A_transition", 0.35)
    A_max = pole_params.get("A_max", 1.25)
    dWs_dAodd = pole_params.get("dWs_dAodd", 2.5 / 0.75)

    # Extract / default runtime knobs (runner injects these for junction gaits)
    wt_direction = float(gait_params.get("wt_direction", 1.0))
    tightness = float(gait_params.get("tightness", abs(gait_params.get("A_even", 0.0))))
    pole_direction = float(gait_params.get("pole_direction", 1.0))

    # Pull base parameters from the gait dict
    A_even = float(gait_params.get("A_even", 0.0))
    wS_even = float(gait_params.get("wS_even", 0.0))
    wT_even = float(gait_params.get("wT_even", 0.0))

    # Update spatial frequency using commanded tightness.
    wS_max = wS_even
    A_min = A_even
    if tightness < A_transition:
        wS_odd = 0.0
    else:
        wS_odd = min(wS_max, (tightness - A_transition) * dWs_dAodd)
    wS_odd *= -pole_direction

    # Update amplitude using commanded tightness.
    if tightness < A_min:
        A_odd = A_min
    else:
        A_odd = min(tightness, A_max)
    A_odd *= -pole_direction

    # Use even=odd for rolling-helix style behavior
    wS_even = wS_odd
    A_even = A_odd

    if not compute:
        gait_params["A_even"] = A_even
        gait_params["A_odd"] = A_odd
        gait_params["wS_even"] = wS_even
        gait_params["wS_odd"] = wS_odd
        return gait_params

    # Parameters that define the T-junction windowing behavior
    A_1_multiplier = float(params.get("A_1_multiplier", 1.0))
    A_2_multiplier = float(params.get("A_2_multiplier", 1.0))
    mu = float(params.get("mu", (self.num_modules - 1) / 2.0))
    phi_0 = float(params.get("phi_0", 0.0))

    # Window parameters (same as t_junction)
    m = float(params.get("m", 50.0))
    sig = float(params.get("sig", 0.05))

    # Apply multipliers
    A_1 = A_even * A_1_multiplier
    A_2 = A_even * A_2_multiplier
    A_set = [A_1, A_2]
    wS_set = [wS_even, wS_even]

    target_angles = np.zeros(self.num_modules, dtype=float)

    for i in range(self.num_modules):
        offset = np.pi if (i % 2 == 0) else (-np.pi / 2.0)
        offset_hook = np.sin(phi_0 + wS_even * i + wT_even * t + offset)

        target_angles[i] = (
            self.amplitude_reduced(i, A_set, m, mu, sig)
            * np.sin(self.parameter_windowed(i, wS_set, mu, m) * i + wT_even * t + offset)
            + offset_hook * A_even * self.gaussian_window(i / 15.0, mu / 15.0, sig)
        )

        target_angles[i] = min(max(target_angles[i], -np.pi / 2.0), np.pi / 2.0)

    return flip_axes(target_angles)