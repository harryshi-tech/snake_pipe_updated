"""snake_control.gaitlib.reu_gaits.spiraling

Standalone spiraling gait for T-junction navigation.

This implementation matches the "spiraling" block used in the original
snakes_on_pipes T-junction gait:
- compute a baseline rolling-helix (with amplitude/spatial-frequency determined by
  commanded tightness)
- compute a "bump" helix (different A and wS) intended to locally change helix radius
- smoothly blend baseline and bump using a sinusoidal window along the body

Why we don't use the per-gait YAML defaults here:
- historically, many repos had a placeholder "spiraling" YAML entry (wT=0, wS=1, ...)
  which causes a dramatic shape jump and no motion.
- for consistent behavior, we take defaults from t_junction (or rolling_helix fallback)
  and then apply any provided overrides.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np


def spiraling(self, t=0, params=None, pole_params=None, compute: bool = True):
    params = {} if params is None else params
    pole_params = {} if pole_params is None else pole_params

    self.current_gait = "spiraling"

    # Use t_junction defaults (contains the correct rolling-helix baseline parameters).
    defaults = self.default_gait_params.get("t_junction") or {}
    if not isinstance(defaults, dict) or len(defaults) == 0:
        defaults = self.default_gait_params.get("rolling_helix") or {}

    self.current_gait_params = self.update_params(defaults, params)

    # Pole-climb relationship between amplitude and spatial frequency.
    A_transition = pole_params.get("A_transition", 0.35)
    A_max = pole_params.get("A_max", 1.25)
    dWs_dAodd = pole_params.get("dWs_dAodd", 2.5 / 0.75)

    # Extract individual gait parameters from dictionary
    beta_even = self.current_gait_params.get("beta_even", 0.0)
    beta_odd = self.current_gait_params.get("beta_odd", 0.0)
    A_even = float(self.current_gait_params.get("A_even", 0.2))
    A_odd = float(self.current_gait_params.get("A_odd", 0.2))
    wS_even = float(self.current_gait_params.get("wS_even", 14.406))
    wS_odd = float(self.current_gait_params.get("wS_odd", 14.406))
    wT_even = float(self.current_gait_params.get("wT_even", 2.0))
    wT_odd = float(self.current_gait_params.get("wT_odd", 2.0))
    delta = float(self.current_gait_params.get("delta", -1.57079632679))

    # These are injected by teleop/controller overrides.
    tightness = float(self.current_gait_params.get("tightness", abs(A_even)))
    pole_direction = float(self.current_gait_params.get("pole_direction", 1.0))

    # --- Update (A, wS) based on commanded tightness (same logic as t_junction) ---
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

    # Parameters that shape the windowing/blending
    wS_1 = wS_even
    wS_2 = wS_even
    A_1_multiplier = float(params.get("A_1_multiplier", 1.0))
    A_2_multiplier = float(params.get("A_2_multiplier", 1.0))
    mu = float(params.get("mu", (self.num_modules - 1) / 2.0))
    phi_0 = float(params.get("phi_0", 0.0))
    s_0 = float(params.get("s_0", 0.4))
    speed_multiplier = float(params.get("speed_multiplier", 1.0))
    debug_print = bool(params.get("debug_print", False))

    # Tunable window parameters (allow teleop/YAML overrides)
    m = float(params.get("m", 50.0))
    sig = float(params.get("sig", 0.05))
    T = float(params.get("T", 0.25))

    A_1 = A_even * A_1_multiplier
    A_2 = A_even * A_2_multiplier

    # --- Bump helix computation (from snakes_on_pipes t_junction) ---
    if self.snake_type == "SEA":
        module_length = 0.064
    elif self.snake_type == "REU":
        module_length = 0.050
    else:
        module_length = float(pole_params.get("module_length", 0.050))

    if abs(A_even) > (A_transition + 0.2):
        # Original helix pitch/radius (using baseline A_even, wS_even)
        p = module_length / ((np.abs(A_even) / 2 / np.sin(np.abs(wS_even))) ** 2 + 1) / np.abs(wS_even)
        r = np.abs(A_even) / 2 / np.sin(np.abs(wS_even)) * p

        # Heuristic offsets for the bump helix
        offset_p = -p * 0.25
        offset_r = r * 10

        wS_bump = p / ((r + offset_r) ** 2 + (p + offset_p) ** 2) * module_length * np.sign(wS_even)
        A_bump = 2 * (r + offset_r) / (p + offset_p) * np.sin(np.abs(wS_bump)) * np.sign(A_even)
    else:
        A_bump = A_even
        wS_bump = wS_even

    A_set = [A_1, A_2]
    wS_set = [wS_1, wS_2]

    A_set_spiraling = [A_bump * A_1_multiplier, A_bump * A_2_multiplier]
    wS_set_spiraling = [wS_bump, wS_bump]

    den = float(max(1, self.num_modules - 1))
    target_angles = np.zeros(self.num_modules, dtype=float)

    for i in range(self.num_modules):
        # Phase offsets: match the snakes_on_pipes convention (even/odd joints)
        if i % 2 == 0:
            offset = np.pi
        else:
            offset = -np.pi / 2

        # Small "hook" term used in the original implementation
        offset_hook = np.sin(phi_0 + wS_even * i + wT_even * t + offset)

        base_i = self.amplitude_reduced(i, A_set, m, mu, sig) * np.sin(
            self.parameter_windowed(i, wS_set, mu, m) * i + wT_even * t + offset
        ) + offset_hook * A_even * self.gaussian_window(i / den, mu / den, sig)

        # Continuity offset for the bump helix
        cont_offset = (wS_even - wS_bump) * ((mu / den) - (T / 2.0)) * self.num_modules

        bump_i = self.amplitude_reduced(i, A_set_spiraling, m, mu, sig) * np.sin(
            self.parameter_windowed(i, wS_set_spiraling, mu, m) * i + cont_offset + wT_even * t + offset
        ) + offset_hook * A_even * self.gaussian_window(i / den, mu / den, sig)

        s = self.sinus_window(i / den, s_0, T)
        target_angles[i] = base_i * (1 - s) + bump_i * s

        # joint limits
        target_angles[i] = min(max(target_angles[i], -np.pi / 2), np.pi / 2)

    # Flip axes to match hardware conventions (same as snakes_on_pipes)
    target_angles[2::4] *= -1
    target_angles[3::4] *= -1

    if debug_print:
        param_names = [
            "A1_multiplier",
            "A2_multiplier",
            "A_even",
            "wS_even",
            "wT_even",
            "mu",
            "phi_0",
            "s_0",
            "tightness",
            "speed_multiplier",
            "m",
            "sig",
            "T",
        ]
        param_values = [
            A_1_multiplier,
            A_2_multiplier,
            A_even,
            wS_even,
            wT_even,
            mu,
            phi_0,
            s_0,
            tightness,
            speed_multiplier,
            m,
            sig,
            T,
        ]
        print(dict(zip(param_names, param_values)))

    if not compute:
        out = deepcopy(self.current_gait_params)
        out.update(
            {
                "A_1_multiplier": A_1_multiplier,
                "A_2_multiplier": A_2_multiplier,
                "mu": mu,
                "phi_0": phi_0,
                "s_0": s_0,
                "speed_multiplier": speed_multiplier,
                "m": m,
                "sig": sig,
                "T": T,
            }
        )
        return out

    return target_angles