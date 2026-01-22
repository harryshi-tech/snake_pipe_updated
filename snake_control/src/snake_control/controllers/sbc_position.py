# snake_control/src/snake_control/controllers/sbc_position.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from snake_bullet.sim_env import RobotState, JointCommand

from snake_control.controllers.gait_position import GaitPositionController, GaitPositionCfg


@dataclass
class SBCPositionCfg(GaitPositionCfg):
    """Shape-Based Compliance (SBC) position controller config.

    This controller executes the same gait library as :class:`GaitPositionController`,
    but adds a *shape-based compliance* mechanism inspired by the provided Matlab
    rolling-helix SBC script:

      - Maintain an internal amplitude state per joint (A_d) with 2nd-order dynamics
      - Drive A_d down when |tau| is large (C_A = -|tau|)
      - Reconstruct the joint command by scaling only the oscillatory component of
        the nominal gait (keep beta offsets)

    Notes:
      - For gaits that don't expose (A_even/A_odd, beta_even/beta_odd) in params,
        compliance falls back to "no-op" (pure gait position).
      - The gains are intentionally exposed for tuning.
    """

    # --- SBC enable ---
    enable_sbc: bool = True

    # --- decentralized coupling (windowed SBC) ---
    # 0 => per-joint (maximally decentralized, backwards compatible)
    # 1 => one global amplitude state per parity (even/odd joints)
    # W>1 => W coupling windows along the body (per parity)
    sbc_windows: int = 3

    # Gaussian window width in *joint-index* units. If None, auto-chosen from (#joints / sbc_windows).
    sbc_window_sigma: Optional[float] = None

    # --- torque filtering ---
    torque_lpf_tau: float = 1 # [s] low-pass filter time constant
    torque_deadband: float = 0.0  # [Nm] ignore |tau| below this

    # --- amplitude dynamics (direction-dependent, like the Matlab example) ---
    M_sig_pos: float = 0.8
    K_sig_pos: float = 65
    B_sig_pos: Optional[float] = None  # if None -> critical damping 2*sqrt(K*M)

    M_sig_neg: float = 0.8
    K_sig_neg: float = 65
    B_sig_neg: Optional[float] = None

    # Scale on the shape force term: C_A = -C_gain*|tau|
    C_gain: float = 60

    # Clamp on compliant amplitude (magnitude)
    #
    # Important nuance:
    #   Some gaits (notably rolling_helix) may run with a *small* nominal amplitude A0
    #   (A0 ~= |tightness| after internal mapping). If A_min is larger than A0, then
    #   the clamp window would collapse and SBC would effectively become a no-op.
    #
    #   To keep rolling_helix compliant out-of-the-box, the implementation below
    #   uses A_min for "normal" amplitudes, but when A0 < A_min it falls back to
    #   a fraction of A0 (see A_min_small_ratio).
    A_min: float = 0.5

    # When the nominal amplitude A0 is smaller than A_min, use this fraction of A0
    # as the lower clamp instead. (e.g., 0.2 means A >= 0.2*A0)
    A_min_small_ratio: float = 0.5
    A_max_factor: float = 1.0  # A_d <= A_max_factor * A0


class SBCPositionController(GaitPositionController):
    """Gait position controller with an SBC amplitude-compliance layer."""

    def __init__(self, cfg: SBCPositionCfg):
        super().__init__(cfg)
        self.cfg: SBCPositionCfg = cfg

        self._tau_f: Optional[List[float]] = None
        self._A0: Optional[List[float]] = None  # nominal amplitude magnitudes
        self._A: Optional[List[float]] = None   # compliant amplitude magnitudes
        # Per-joint dynamics state (used when sbc_windows<=0 or sbc_windows>=#joints)
        self._Adot: Optional[List[float]] = None

        # Windowed decentralized coupling (used when sbc_windows is 1..(#joints-1))
        self._sbc_mode: str = "joint"  # 'joint' | 'window'
        self._win_count: int = 0
        self._win_w: Optional[List[List[float]]] = None  # [W][N], normalized per joint (sum_j w[j][i] = 1)
        self._w_sum_even: Optional[List[float]] = None   # [W], sum of window weights over even joints
        self._w_sum_odd: Optional[List[float]] = None    # [W], sum of window weights over odd joints

        self._win_centers: Optional[List[float]] = None  # [W], joint-index centers
        self._win_sigma: Optional[float] = None          # Gaussian sigma in joint-index units

        self._A0_even_win: Optional[List[float]] = None
        self._A0_odd_win: Optional[List[float]] = None
        self._A_even_win: Optional[List[float]] = None
        self._A_odd_win: Optional[List[float]] = None
        self._Adot_even_win: Optional[List[float]] = None
        self._Adot_odd_win: Optional[List[float]] = None

    def reset(self, state: RobotState) -> None:
        super().reset(state)
        n = len(state.q)
        self._tau_f = [0.0] * n
        self._A0 = None
        self._A = None
        self._Adot = [0.0] * n

        self._sbc_mode = "joint"
        self._win_count = 0
        self._win_w = None
        self._w_sum_even = None
        self._w_sum_odd = None
        self._win_centers = None
        self._win_sigma = None
        self._A0_even_win = None
        self._A0_odd_win = None
        self._A_even_win = None
        self._A_odd_win = None
        self._Adot_even_win = None
        self._Adot_odd_win = None

        # Debug payload (merged into parent debug())
        self._dbg = {}

    def set_gait(self, gait_name: str, state: RobotState, transition_override: bool = False) -> None:
        """Same semantics as :meth:`GaitPositionController.set_gait`, but also resets SBC state."""
        prev = getattr(self, "_current_gait", None)
        super().set_gait(gait_name, state, transition_override=transition_override)
        if prev != getattr(self, "_current_gait", None):
            # Gait changed: reinitialize amplitude states on next step().
            self._A0 = None
            self._A = None
            if self._Adot is not None:
                self._Adot = [0.0] * len(self._Adot)

            # Windowed state also gets rebuilt on next step().
            self._sbc_mode = "joint"
            self._win_count = 0
            self._win_w = None
            self._w_sum_even = None
            self._w_sum_odd = None
            self._win_centers = None
            self._win_sigma = None
            self._A0_even_win = None
            self._A0_odd_win = None
            self._A_even_win = None
            self._A_odd_win = None
            self._Adot_even_win = None
            self._Adot_odd_win = None

    # -----------------
    # helpers
    # -----------------
    @staticmethod
    def _sign_flip_from_index(n0: int) -> float:
        # Matches gaitlib compound_serpenoid: alpha *= (-1)**floor(n/2)
        return -1.0 if ((n0 // 2) % 2 == 1) else 1.0

    def _lpf_tau(self, tau: List[float], dt: float) -> List[float]:
        if self._tau_f is None:
            self._tau_f = [0.0] * len(tau)
        tau_c = max(0.0, float(self.cfg.torque_lpf_tau))
        if tau_c <= 0.0:
            self._tau_f = [float(x) for x in tau]
            return list(self._tau_f)
        a = float(dt) / (tau_c + float(dt)) if dt > 0 else 1.0
        for i in range(len(tau)):
            self._tau_f[i] = (1.0 - a) * float(self._tau_f[i]) + a * float(tau[i])
        return list(self._tau_f)


    def _build_windows(self, n_joints: int) -> None:
        """Build fixed Gaussian coupling windows along joint indices.

        Windows are *fixed* in joint-index space (do not move with the wave).
        The weight matrix is normalized per joint so that for every joint i:
            sum_j w[j][i] = 1

        SBC dynamics are then integrated per-window *and* per-parity (even/odd),
        and reconstructed back to per-joint compliant amplitudes.
        """
        W_cfg = int(getattr(self.cfg, "sbc_windows", 0) or 0)

        # Modes:
        #   sbc_windows <= 0 : per-joint (backwards compatible)
        #   sbc_windows == 1 : one global window (per parity)
        #   1 < sbc_windows < N : windowed coupling
        #   sbc_windows >= N : treat as per-joint (avoids weird/degenerate windows)
        if W_cfg <= 0 or W_cfg >= int(n_joints):
            self._sbc_mode = "joint"
            self._win_count = 0
            self._win_w = None
            self._w_sum_even = None
            self._w_sum_odd = None
            self._win_centers = None
            self._win_sigma = None
            self._A0_even_win = None
            self._A0_odd_win = None
            self._A_even_win = None
            self._A_odd_win = None
            self._Adot_even_win = None
            self._Adot_odd_win = None
            return

        W = max(1, W_cfg)
        self._sbc_mode = "window"
        self._win_count = W

        # Window centers in joint-index space.
        if W == 1:
            centers = [0.5 * float(n_joints - 1)]
        else:
            centers = [float(n_joints - 1) * float(j) / float(W - 1) for j in range(W)]

        sigma = getattr(self.cfg, "sbc_window_sigma", None)
        if sigma is None:
            # Heuristic: about half a window length.
            sigma = 0.5 * (float(n_joints) / float(W))
        sigma = max(1e-6, float(sigma))

        # Raw Gaussian weights
        win_w: List[List[float]] = []
        for c in centers:
            row = []
            for i in range(int(n_joints)):
                d = (float(i) - float(c)) / sigma
                row.append(math.exp(-0.5 * d * d))
            win_w.append(row)

        # Normalize per joint: sum_j w[j][i] = 1
        for i in range(int(n_joints)):
            s = 0.0
            for j in range(W):
                s += float(win_w[j][i])
            if s <= 1e-12:
                for j in range(W):
                    win_w[j][i] = 1.0 / float(W)
            else:
                inv = 1.0 / s
                for j in range(W):
                    win_w[j][i] = float(win_w[j][i]) * inv

        # Parity weight sums (used for window-averaging)
        w_sum_even = [0.0] * W
        w_sum_odd = [0.0] * W
        for j in range(W):
            se = 0.0
            so = 0.0
            for i in range(int(n_joints)):
                if (i % 2) == 0:
                    se += float(win_w[j][i])
                else:
                    so += float(win_w[j][i])
            w_sum_even[j] = se
            w_sum_odd[j] = so

        # Window nominal amplitudes (weighted averages over each parity)
        A0 = self._A0 if self._A0 is not None else [0.0] * int(n_joints)

        eps = 1e-9
        A0_even_win: List[float] = []
        A0_odd_win: List[float] = []
        for j in range(W):
            se = float(w_sum_even[j])
            so = float(w_sum_odd[j])

            if se > eps:
                num = 0.0
                for i in range(int(n_joints)):
                    if (i % 2) == 0:
                        num += float(win_w[j][i]) * float(A0[i])
                A0_even_win.append(num / se)
            else:
                A0_even_win.append(0.0)

            if so > eps:
                num = 0.0
                for i in range(int(n_joints)):
                    if (i % 2) == 1:
                        num += float(win_w[j][i]) * float(A0[i])
                A0_odd_win.append(num / so)
            else:
                A0_odd_win.append(0.0)

        # Initialize window amplitude states at nominal.
        self._win_w = win_w
        self._w_sum_even = w_sum_even
        self._w_sum_odd = w_sum_odd
        self._win_centers = centers
        self._win_sigma = sigma

        self._A0_even_win = A0_even_win
        self._A0_odd_win = A0_odd_win
        self._A_even_win = list(A0_even_win)
        self._A_odd_win = list(A0_odd_win)
        self._Adot_even_win = [0.0] * W
        self._Adot_odd_win = [0.0] * W
    
    def _ensure_amp_state(self, params_used: Dict[str, Any], n_joints: int) -> None:
        """Initialize nominal amplitude (A0) and compliance states.

        Always initializes per-joint A0/A (for logging + reconstruction), then
        optionally enables *windowed decentralized coupling* depending on
        cfg.sbc_windows.
        """
        if self._A0 is not None and self._A is not None:
            return

        A_even = float(params_used.get("A_even", 0.0))
        A_odd = float(params_used.get("A_odd", 0.0))

        A0 = [0.0] * int(n_joints)
        for i in range(int(n_joints)):
            A0[i] = abs(A_even) if ((i % 2) == 0) else abs(A_odd)

        self._A0 = A0
        self._A = list(A0)

        if self._Adot is None or len(self._Adot) != int(n_joints):
            self._Adot = [0.0] * int(n_joints)
        else:
            # Keep existing length-consistent state, but re-zero on new init
            self._Adot = [0.0] * int(n_joints)

        # Configure decentralized coupling mode
        self._build_windows(int(n_joints))


    
    def _step_amp_dynamics(self, tau_f: List[float], dt: float) -> None:
        if self._A0 is None or self._A is None:
            return
        if dt <= 0.0:
            return

        dead = float(self.cfg.torque_deadband)
        C_gain = float(self.cfg.C_gain)

        # Direction-dependent gains (like the Matlab script)
        def gains_for_tau(t: float) -> tuple[float, float, float]:
            if t >= 0.0:
                M = float(self.cfg.M_sig_pos)
                K = float(self.cfg.K_sig_pos)
                B = float(self.cfg.B_sig_pos) if self.cfg.B_sig_pos is not None else 2.0 * (K * M) ** 0.5
                return M, B, K
            else:
                M = float(self.cfg.M_sig_neg)
                K = float(self.cfg.K_sig_neg)
                B = float(self.cfg.B_sig_neg) if self.cfg.B_sig_neg is not None else 2.0 * (K * M) ** 0.5
                return M, B, K

        def clamp_amp(A_new: float, A0: float) -> float:
            if A0 <= 1e-9:
                return 0.0

            upper = float(self.cfg.A_max_factor) * A0

            a_min = float(self.cfg.A_min)
            if A0 < a_min:
                ratio = max(0.0, float(getattr(self.cfg, "A_min_small_ratio", 0.0)))
                lower = ratio * A0
            else:
                lower = a_min

            lower = min(lower, upper)
            return max(lower, min(A_new, upper))

        # -----------------
        # windowed decentralized coupling
        # -----------------
        if self._sbc_mode == "window":
            # Defensive: rebuild windows if missing (shouldn't happen unless config changes at runtime)
            if self._win_w is None or self._A_even_win is None or self._A_odd_win is None:
                self._build_windows(len(self._A))
            if self._win_w is not None and self._A_even_win is not None and self._A_odd_win is not None:
                W = int(self._win_count)
                n = int(len(self._A))

                # helpers for weighted averages over parity sets
                def parity_weighted_stats(j: int, parity: int) -> tuple[float, float, float]:
                    # returns (avg_signed_tau, avg_abs_tau, w_sum)
                    wj = self._win_w[j]
                    s_w = 0.0
                    s_tau = 0.0
                    s_abs = 0.0
                    for i in range(n):
                        if (i % 2) != parity:
                            continue
                        wi = float(wj[i])
                        s_w += wi
                        ti = float(tau_f[i])
                        s_tau += wi * ti
                        s_abs += wi * abs(ti)

                    if s_w <= 1e-12:
                        return 0.0, 0.0, 0.0

                    avg_signed = s_tau / s_w
                    avg_abs = s_abs / s_w
                    if avg_abs < dead:
                        avg_signed = 0.0
                        avg_abs = 0.0
                    return avg_signed, avg_abs, s_w

                # integrate each window's amplitude dynamics (even + odd)
                for j in range(W):
                    # even parity
                    avg_signed, avg_abs, _ = parity_weighted_stats(j, parity=0)
                    A0e = float(self._A0_even_win[j]) if self._A0_even_win is not None else 0.0
                    Ae = float(self._A_even_win[j])
                    Aedot = float(self._Adot_even_win[j]) if self._Adot_even_win is not None else 0.0

                    if A0e <= 1e-9:
                        self._A_even_win[j] = 0.0
                        if self._Adot_even_win is not None:
                            self._Adot_even_win[j] = 0.0
                    else:
                        M_sig, B_sig, K_sig = gains_for_tau(avg_signed)
                        M_A = max(1e-9, M_sig)
                        B_A = B_sig + 2.0 * M_sig
                        K_A = K_sig + B_sig + M_sig
                        C_A = -C_gain * avg_abs

                        Addot = (C_A - B_A * Aedot - K_A * (Ae - A0e)) / M_A
                        Aedot_new = Aedot + Addot * dt
                        Ae_new = Ae + Aedot_new * dt

                        self._A_even_win[j] = clamp_amp(Ae_new, A0e)
                        if self._Adot_even_win is not None:
                            self._Adot_even_win[j] = Aedot_new

                    # odd parity
                    avg_signed, avg_abs, _ = parity_weighted_stats(j, parity=1)
                    A0o = float(self._A0_odd_win[j]) if self._A0_odd_win is not None else 0.0
                    Ao = float(self._A_odd_win[j])
                    Aodot = float(self._Adot_odd_win[j]) if self._Adot_odd_win is not None else 0.0

                    if A0o <= 1e-9:
                        self._A_odd_win[j] = 0.0
                        if self._Adot_odd_win is not None:
                            self._Adot_odd_win[j] = 0.0
                    else:
                        M_sig, B_sig, K_sig = gains_for_tau(avg_signed)
                        M_A = max(1e-9, M_sig)
                        B_A = B_sig + 2.0 * M_sig
                        K_A = K_sig + B_sig + M_sig
                        C_A = -C_gain * avg_abs

                        Addot = (C_A - B_A * Aodot - K_A * (Ao - A0o)) / M_A
                        Aodot_new = Aodot + Addot * dt
                        Ao_new = Ao + Aodot_new * dt

                        self._A_odd_win[j] = clamp_amp(Ao_new, A0o)
                        if self._Adot_odd_win is not None:
                            self._Adot_odd_win[j] = Aodot_new

                # reconstruct per-joint compliant amplitudes by blending windows
                for i in range(n):
                    if (i % 2) == 0:
                        Ai = 0.0
                        Adoti = 0.0
                        for j in range(W):
                            wi = float(self._win_w[j][i])
                            Ai += wi * float(self._A_even_win[j])
                            if self._Adot_even_win is not None:
                                Adoti += wi * float(self._Adot_even_win[j])
                        self._A[i] = Ai
                        if self._Adot is not None:
                            self._Adot[i] = Adoti
                    else:
                        Ai = 0.0
                        Adoti = 0.0
                        for j in range(W):
                            wi = float(self._win_w[j][i])
                            Ai += wi * float(self._A_odd_win[j])
                            if self._Adot_odd_win is not None:
                                Adoti += wi * float(self._Adot_odd_win[j])
                        self._A[i] = Ai
                        if self._Adot is not None:
                            self._Adot[i] = Adoti

                return

            # If we couldn't do windowed mode, fall back.
            self._sbc_mode = "joint"

        # -----------------
        # per-joint (maximally decentralized) mode
        # -----------------
        if self._Adot is None or len(self._Adot) != len(self._A):
            self._Adot = [0.0] * len(self._A)

        for i in range(len(self._A)):
            A0 = float(self._A0[i])
            if A0 <= 1e-9:
                self._A[i] = 0.0
                self._Adot[i] = 0.0
                continue

            tau_i = float(tau_f[i])
            if abs(tau_i) < dead:
                tau_i = 0.0

            M_sig, B_sig, K_sig = gains_for_tau(tau_i)

            # Matlab uses projected gains with J_A. Since J_A'J_A = 1, these reduce to:
            #   M_A = M_sig
            #   B_A = B_sig + 2*M_sig
            #   K_A = K_sig + B_sig + M_sig
            M_A = max(1e-9, M_sig)
            B_A = B_sig + 2.0 * M_sig
            K_A = K_sig + B_sig + M_sig

            C_A = -C_gain * abs(tau_i)

            A = float(self._A[i])
            Adot = float(self._Adot[i])
            Addot = (C_A - B_A * Adot - K_A * (A - A0)) / M_A

            # Semi-implicit Euler
            Adot_new = Adot + Addot * dt
            A_new = A + Adot_new * dt

            self._A[i] = clamp_amp(A_new, A0)
            self._Adot[i] = Adot_new


    def debug(self) -> Dict[str, Any]:
        # Start with the most recent per-step debug payload (if any)
        out: Dict[str, Any] = dict(getattr(self, "_dbg", {}) or {})
        if self._A0 is not None and self._A is not None:
            scales = []
            for a, a0 in zip(self._A, self._A0):
                if a0 > 1e-9:
                    scales.append(float(a) / float(a0))
            if scales:
                out["sbc_scale_mean"] = float(sum(scales) / len(scales))
                out["sbc_scale_min"] = float(min(scales))
                out["sbc_scale_max"] = float(max(scales))
        return out

    def print_param_summary(self) -> None:
        """Print the normal gait param summary + a small SBC state summary."""
        super().print_param_summary()
        dbg = self.debug()
        if dbg:
            print("[sbc_position] sbc_state:", dbg)

    # -----------------
    # main control step
    # -----------------
    def step(self, state: RobotState) -> JointCommand:
        # Copy of GaitPositionController.step with an SBC post-process on q_des.
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

        out = self.runner.step(
            self._current_gait,
            t=t_eval,
            overrides=self.cfg.gait_params,
            extra=extra,
            include_shape=False,
        )
        q_nom = [float(x) for x in out.q.tolist()]
        q_des = list(q_nom)
        self._last_meta = dict(out.meta)

        # --- SBC compliance layer ---
        q_sbc: Optional[List[float]] = None
        tau_f: Optional[List[float]] = None
        sbc_active: bool = False
        sbc_reason: str = "disabled"

        if bool(self.cfg.enable_sbc):
            sbc_reason = "inactive"
            params_used = (out.meta or {}).get("params_used", {}) or {}
            # Only activate if this gait looks like compound-serpenoid (A_even/A_odd present)
            if ("A_even" in params_used) and ("A_odd" in params_used):
                sbc_active = True
                sbc_reason = "active"
                n_j = len(q_des)
                self._ensure_amp_state(params_used, n_j)
                tau_f = self._lpf_tau(list(state.tau), dt)
                self._step_amp_dynamics(tau_f, dt)

                # Reconstruct joint command by scaling oscillatory component and preserving beta offset.
                beta_even = float(params_used.get("beta_even", 0.0))
                beta_odd = float(params_used.get("beta_odd", 0.0))

                assert self._A0 is not None and self._A is not None
                q_sbc = [0.0] * n_j
                for i in range(n_j):
                    # nominal offset (beta) in *joint space* must include sign flip
                    flip = self._sign_flip_from_index(i)
                    beta_i = (beta_even if (i % 2) == 0 else beta_odd) * flip

                    A0 = float(self._A0[i])
                    if A0 <= 1e-9:
                        q_sbc[i] = q_des[i]
                        continue

                    scale = float(self._A[i]) / A0
                    q_sbc[i] = beta_i + scale * (float(q_des[i]) - beta_i)

                q_des = q_sbc
            else:
                sbc_reason = "missing_A_even_A_odd"

        # Snapshot the SBC-processed vector (before transition blending)
        if q_sbc is None and bool(self.cfg.enable_sbc):
            q_sbc = list(q_des)

        # Update snake_time using lab rule (same as gait_position)
        if self.cfg.use_snake_time:
            merged = out.meta.get("params_merged", {}) or {}
            wave_direction = float(merged.get("speed_multiplier", 1.0))
            pole_direction = 1.0
            try:
                pole_direction = 1.0 if float(merged.get("pole_direction", 1.0)) > 0 else -1.0
            except Exception:
                pole_direction = 1.0

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
        A_scale = None
        if self._A0 is not None and self._A is not None:
            s = []
            for a, a0 in zip(self._A, self._A0):
                s.append(float(a) / float(a0) if a0 > 1e-9 else 1.0)
            A_scale = s

        self._dbg = {
            "gait_name": str(self._current_gait),
            "sbc_active": bool(sbc_active),
            "sbc_reason": str(sbc_reason),
            "sbc_mode": str(getattr(self, "_sbc_mode", "joint")),
            "sbc_windows": int(getattr(self.cfg, "sbc_windows", 0) or 0),
            "sbc_window_sigma": float(self._win_sigma) if getattr(self, "_win_sigma", None) is not None else None,
            "sbc_mode": str(getattr(self, "_sbc_mode", "joint")),
            "sbc_windows": int(getattr(self.cfg, "sbc_windows", 0) or 0),
            "sbc_window_sigma": float(self._win_sigma) if getattr(self, "_win_sigma", None) is not None else None,
            "snake_time": float(self._snake_time),
            "q_nom": list(q_nom),
            "q_sbc": list(q_sbc) if q_sbc is not None else list(q_des),
            "q_cmd": list(q_cmd),
            "tau_f": list(tau_f) if tau_f is not None else None,
            "A0": list(self._A0) if self._A0 is not None else None,
            "A": list(self._A) if self._A is not None else None,
            "A_scale": list(A_scale) if A_scale is not None else None,
            "speed_multiplier": float(merged.get("speed_multiplier", 0.0)),
            "tightness": float(merged.get("tightness", 0.0)),
            "pole_direction": float(merged.get("pole_direction", 0.0)),
            "wt_direction": float(merged.get("wt_direction", merged.get("wt_dir", 0.0))),
        }

        return JointCommand(mode="position", position=q_cmd, effort=None)
