# snake_control/teleop/joystick_teleop.py
"""ROS-like joystick teleop for the PyBullet simulation.

Goal
----
Match the behavior of the lab's ROS joystick teleop (snakelib_control/src/snakelib_control/joystick_teleop.py)
as closely as possible, but without ROS:

* Same joy_to_gait.yaml schema (mode__*, switch-flag 0/1/2 mapping)
* Same command processing logic (modes, gait selection, speed, tightness, pole direction, etc.)
* Produces a lightweight SnakeCommand-like object that the simulation can consume.

T-Junction Navigation (Karumanchi et al. 2025)
----------------------------------------------
Implements Algorithm 2's n_t sweep for T-junction traversal:
- Progress variable n_t increments per Eq. 10: ṅt = lM / (π × dM × ωt)
- Spiraling phase: n_t sweeps from N-1 to 0 (pulse travels head to tail)
- Turn phase: n_t sweeps from 0 to N (modules transfer to target pipe)
- Completion: maneuver finishes when n_t reaches boundary, not after fixed time

Notes
-----
* We read raw joystick values via pygame, then assemble a ROS Joy-like (axes, buttons) layout.
* For the Logitech Cordless RumblePad 2 in "Direct" mode, the canonical layout we build is:
  axes  : [dpad_x, dpad_y, r_stick_x, r_stick_y, l_stick_x, l_stick_y]
  buttons: [X, A, B, Y, l_bumper, r_bumper, l_trigger, r_trigger, back, start, l_stick, r_stick]

If your controller reports a different ordering in pygame, use --print_joy in run_gait.py
and adjust the adapter mapping in PygameJoyAdapter.DEFAULT_* constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import time
import numpy as np
import yaml


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    return yaml.safe_load(path.read_text())


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


@dataclass
class TeleopCommand:
    """A minimal stand-in for snakelib_msgs/SnakeCommand."""

    command_name: str
    param_name: List[str]
    param_value: List[float]
    quit: bool = False

    def as_param_dict(self) -> Dict[str, float]:
        return {str(k): float(v) for k, v in zip(self.param_name, self.param_value)}


@dataclass
class TJunctionState:
    """State for T-junction navigation macro (Algorithm 2).
    
    Uses n_t sweep instead of fixed time windows.
    """
    active: bool = False
    phase: str = "idle"          # "idle" | "pre_hold" | "spiral" | "turn" | "post_hold" | "complete"
    kind: str = ""               # "straight" | "turn"
    direction: int = 0           # -1 left, +1 right, 0 straight
    
    # Progress variables
    n_t: float = 0.0             # Transition location [0..N]
    s_0: float = 0.9             # Pulse position for spiraling [0..1]
    
    # User-selected parameters
    mu: float = 7.5              # Bend module location
    phi_0: float = 0.0           # Phase offset for helix orientation
    
    # Timing (for pre/post hold phases only)
    t0: float = 0.0
    phase_start: float = 0.0
    
    # Snapshot of baseline helix params at macro start
    baseline_tightness: float = 0.6
    baseline_wT: float = 2.0
    baseline_pole_direction: float = 1.0
    baseline_wt_direction: float = 1.0


@dataclass
class TeleopLatchState:
    """Latched auto-run state for spiral-forward and T-junction transitions."""
    mode: str = "idle"  # idle | spiral_fwd | transition_left | transition_right
    start_time: float = 0.0
    duration_s: float = 0.0
    timeout_s: float = 0.0
    direction: int = 0  # -1 left, +1 right, 0 straight


class PygameJoyAdapter:
    """Read a pygame joystick and produce a ROS Joy-like (axes, buttons) in a canonical order."""

    # Canonical ROS-like ordering we produce
    AXES_ORDER = ["dpad_x", "dpad_y", "r_stick_x", "r_stick_y", "l_stick_x", "l_stick_y"]
    BUTTONS_ORDER = [
        "X",
        "A",
        "B",
        "Y",
        "l_bumper",
        "r_bumper",
        "l_trigger",
        "r_trigger",
        "back",
        "start",
        "l_stick",
        "r_stick",
    ]

    # Heuristic mapping from pygame indices -> canonical names (Logitech Cordless RumblePad 2, Direct mode)
    # Axes: pygame typically reports sticks as axes[0..3] and D-pad as hat[0].
    DEFAULT_PYGAME_AXIS = {
        "l_stick_x": 0,
        "l_stick_y": 1,
        "r_stick_x": 2,
        "r_stick_y": 3,
    }

    # Buttons: this is the most controller/driver-dependent part.
    # The mapping below matches the common layout for RumblePad2 DirectInput on Linux.
    DEFAULT_PYGAME_BUTTON = {
        "X": 0,
        "A": 1,
        "B": 2,
        "Y": 3,
        "l_bumper": 4,
        "r_bumper": 5,
        "l_trigger": 6,
        "r_trigger": 7,
        "back": 8,
        "start": 9,
        "l_stick": 10,
        "r_stick": 11,
    }

    def __init__(
        self,
        joy_index: int = 0,
        deadzone: float = 0.18,
        verbose: bool = False,
    ) -> None:
        import pygame

        self.pygame = pygame
        self.deadzone = float(deadzone)
        self.verbose = bool(verbose)

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= int(joy_index):
            raise RuntimeError(
                f"No joystick found at joy_index={joy_index}. count={pygame.joystick.get_count()}"
            )

        self.js = pygame.joystick.Joystick(int(joy_index))
        self.js.init()

        self.axis_map = dict(self.DEFAULT_PYGAME_AXIS)
        self.button_map = dict(self.DEFAULT_PYGAME_BUTTON)

        if self.verbose:
            print(f"[teleop] joystick: {self.js.get_name()}")
            print(
                f"[teleop] pygame axes={self.js.get_numaxes()} buttons={self.js.get_numbuttons()} hats={self.js.get_numhats()}"
            )

    def _dz(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else float(v)

    def read(self) -> Tuple[List[float], List[int], bool, List[Tuple[str, str, float]]]:
        """Returns (axes[6], buttons[12], quit_flag, active_named_inputs)."""

        pg = self.pygame
        quit_flag = False

        # Drain events (so hats/buttons update)
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                quit_flag = True

        # --- dpad from hat(0) ---
        dpad_x = 0.0
        dpad_y = 0.0
        if self.js.get_numhats() > 0:
            hx, hy = self.js.get_hat(0)
            dpad_x = float(hx)
            dpad_y = float(hy)

        # --- sticks from axes ---
        def axis(name: str) -> float:
            idx = self.axis_map.get(name, None)
            if idx is None:
                return 0.0
            if idx >= self.js.get_numaxes():
                return 0.0
            return self._dz(self.js.get_axis(int(idx)))

        l_stick_x = axis("l_stick_x")
        l_stick_y = axis("l_stick_y")
        r_stick_x = axis("r_stick_x")
        r_stick_y = axis("r_stick_y")

        # Canonical axes order expected by the ROS teleop logic
        axes = [dpad_x, dpad_y, r_stick_x, r_stick_y, l_stick_x, l_stick_y]

        # --- buttons ---
        def btn(name: str) -> int:
            idx = self.button_map.get(name, None)
            if idx is None:
                return 0
            if idx >= self.js.get_numbuttons():
                return 0
            return int(self.js.get_button(int(idx)) != 0)

        buttons = [btn(n) for n in self.BUTTONS_ORDER]

        active: List[Tuple[str, str, float]] = []
        for i, n in enumerate(self.AXES_ORDER):
            v = float(axes[i])
            if abs(v) > 1e-6:
                active.append(("axes", n, v))
        for i, n in enumerate(self.BUTTONS_ORDER):
            if buttons[i] == 1:
                active.append(("buttons", n, 1.0))

        return axes, buttons, quit_flag, active


class RosLikeJoystickTeleop:
    """Port of the lab's ROS joystick teleop logic with Algorithm 2 n_t sweep."""

    def __init__(
        self,
        snake_type: str,
        snake_params_yaml: Path,
        joy_to_gait_yaml: Path,
        joy_mapping_yaml: Path,
        joy_index: int = 0,
        deadzone: float = 0.18,
        verbose: bool = False,
        joy_adapter: Optional[Any] = None,
        time_fn: Optional[Callable[[], float]] = None,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.snake_type = str(snake_type)
        self.verbose = bool(verbose)

        # params
        self._snake_cfg = _load_yaml(snake_params_yaml)
        self._snake_params = self._snake_cfg.get(self.snake_type, {})
        self._gait_params = self._snake_params.get("gait_params", {})

        self._joy_to_gait = _load_yaml(joy_to_gait_yaml)
        self._joystick_mappings = _load_yaml(joy_mapping_yaml)
        # YAML int keys might come in as str depending on dumper
        self._joystick_mappings = {int(k): str(v) for k, v in self._joystick_mappings.items()}

        # Steps / limits
        self._speed_step = float(self._joy_to_gait.get("speed_step", 0.01))
        self._speed_min = float(self._joy_to_gait.get("speed_min", 0.01))
        self._speed_max = float(self._joy_to_gait.get("speed_max", 2.0))

        self._tightness_step = float(self._joy_to_gait.get("tightness_step", 0.005))
        self._tightness_min = float(self._joy_to_gait.get("tightness_min", 0.0))
        self._tightness_max = float(
            self._gait_params.get("pole_climb", {}).get("A_max", 1.5)
        )

        # Rolling helix teleop params
        self._mu_step = float(self._joy_to_gait.get("mu_step", 0.075))
        self._mu_min = float(self._joy_to_gait.get("mu_min", 0.0))
        self._mu_max = float(self._joy_to_gait.get("mu_max", 15.0))

        self._s_0_step = float(self._joy_to_gait.get("s_0_step", -0.01))
        self._s_0_min = float(self._joy_to_gait.get("s_0_min", -0.2))
        self._s_0_max = float(self._joy_to_gait.get("s_0_max", 1.2))

        self._phi_0_step = float(self._joy_to_gait.get("phi_0_step", 0.015))
        self._phi_0_min = float(self._joy_to_gait.get("phi_0_min", -3.14159))
        self._phi_0_max = float(self._joy_to_gait.get("phi_0_max", 3.14159))

        self._A_1_multiplier_step = float(self._joy_to_gait.get("A_1_multiplier_step", 0.005))
        self._A_1_multiplier_min = float(self._joy_to_gait.get("A_1_multiplier_min", -1.0))
        self._A_1_multiplier_max = float(self._joy_to_gait.get("A_1_multiplier_max", 1.0))

        self._A_2_multiplier_step = float(self._joy_to_gait.get("A_2_multiplier_step", 0.005))
        self._A_2_multiplier_min = float(self._joy_to_gait.get("A_2_multiplier_min", -1.0))
        self._A_2_multiplier_max = float(self._joy_to_gait.get("A_2_multiplier_max", 1.0))

        # Conical sidewinding slope
        self._slope_max = float(self._gait_params.get("conical_sidewinding", {}).get("max_slope", 0.1))
        self._slope_default = float(self._joy_to_gait.get("slope_default", 0.0))
        self._slope_step = float(self._joy_to_gait.get("slope_step", 0.005))

        # State (mirrors ROS teleop)
        self._num_axes = 6
        self.current_mode = "mode__normal"
        self._prev_mode = ""
        self._prev_index = np.array([])
        self._last_index: Optional[int] = None

        self._mapped_command = ""
        self._switch_flag = 0
        self._switch_flag_hold = False
        self._has_pole_direction = False

        self._direction = 0
        self._pole_direction = 0
        self._wt_dir = 1

        self._tightness = self._tightness_min
        self._slope_sidewind = self._slope_default

        # Rolling-helix / t-junction teleop
        self._mu = 0.0
        self._mu_last = 7.5
        self._phi_0 = 0.0
        self._phi_0_last = 0.0
        self._A_1_multiplier = 0
        self._A_1_multiplier_last = 1.0
        self._A_2_multiplier = 0
        self._A_2_multiplier_last = 1.0
        self._s_0 = 0
        self._s_0_last = 0.0

        self._x_state = self._y_state = self._z_state = 0.0
        self._roll = self._pitch = self._yaw = 0.0

        self._last_sent = "hold_position"
        self._speed_factor = 1.0

        # T-junction macro configuration
        teleop_cfg = (self._snake_params.get("teleop", {}) or {})
        self._tj_cfg = dict((teleop_cfg.get("tjunction", {}) or {}))
        self._spiral_auto_s = float(teleop_cfg.get("spiral_forward_auto_duration_s", 2.0))
        self._transition_timeout_s = float(teleop_cfg.get("transition_auto_timeout_s", 6.0))

        # Module geometry for n_t calculation (Eq. 10)
        module_geo = self._snake_params.get("module_geometry", {}) or {}
        tj_params = self._gait_params.get("t_junction", {}) or {}
        
        if self.snake_type == "SEA":
            self._l_M = float(module_geo.get("l_M", tj_params.get("l_M", 0.064)))
            self._d_M = float(module_geo.get("d_M", tj_params.get("d_M", 0.051)))
        elif self.snake_type == "REU":
            self._l_M = float(module_geo.get("l_M", tj_params.get("l_M", 0.050)))
            self._d_M = float(module_geo.get("d_M", tj_params.get("d_M", 0.050)))
        else:
            self._l_M = float(module_geo.get("l_M", tj_params.get("l_M", 0.050)))
            self._d_M = float(module_geo.get("d_M", tj_params.get("d_M", 0.050)))

        # Determine module count
        n_mod = 0
        try:
            n_mod = len(self._snake_params.get("module_names", []) or [])
        except Exception:
            n_mod = 0
        if n_mod <= 0:
            try:
                n_mod = len(self._snake_params.get("home", []) or [])
            except Exception:
                n_mod = 0
        self._n_modules = int(n_mod) if int(n_mod) > 0 else 16

        # T-junction state (Algorithm 2)
        self._tj = TJunctionState()
        self._latch = TeleopLatchState()
        self._transition_complete = False
        self._transition_complete_fn: Optional[Callable[[], bool]] = None
        
        # Control loop timing
        self._time_fn = time_fn or time.monotonic
        self._input_fn = input_fn or input
        self._last_step_time = self._time_fn()
        self._dt = 0.01  # Will be updated each step

        # Input adapter
        self._joy = joy_adapter or PygameJoyAdapter(
            joy_index=joy_index,
            deadzone=deadzone,
            verbose=verbose,
        )

    def set_transition_complete(self, value: bool) -> None:
        """Set transition completion flag from controller/sim layer."""
        self._transition_complete = bool(value)

    def set_transition_complete_fn(self, fn: Optional[Callable[[], bool]]) -> None:
        """Register a completion callback used for latching transitions."""
        self._transition_complete_fn = fn

    def _compute_nt_dot(self, wT: float) -> float:
        """Compute n_t update rate from Eq. 10: ṅt = lM / (π × dM × ωt)
        
        With default module geometry, Eq. 10 gives ~0.16 modules/s which is too slow
        for practical use. The nt_sweep_rate_mult parameter scales this up.
        """
        if abs(wT) < 1e-6:
            return 0.0
        base_rate = self._l_M / (np.pi * self._d_M * abs(wT))
        # Apply multiplier for practical sweep speed
        mult = float(self._tj_cfg.get("nt_sweep_rate_mult", 50.0))
        return base_rate * mult

    # --------------------
    # Input mapping helpers
    # --------------------
    def _joy_cb(self, axes: List[float], buttons: List[int]) -> None:
        # Concatenate axes+buttons (like ROS code)
        joy_state = list(axes) + list(buttons)

        # Rolling helix teleop parameters (same indices/meaning as ROS version)
        # mu  <- -axes[2]
        # phi <-  axes[5]
        self._mu = -float(axes[2])
        self._phi_0 = float(axes[5])
        self._A_1_multiplier = int(buttons[2])  # B
        self._A_2_multiplier = int(buttons[1])  # A
        self._s_0 = int(buttons[3])  # Y

        # Extract non-zero indices
        index_array = np.nonzero(np.array(joy_state))[0]

        # switch-flag logic (mirrors ROS code)
        for i in range(len(index_array)):
            name = self._joystick_mappings.get(int(index_array[i]), "")

            if name == "back":
                self._switch_flag = 1
                index_array = np.delete(index_array, i)
                break

            # Enable behavior (switch_flag=2 for pole-climb helpers, switch_flag=3 for T-junction macro)
            cmd = (
                self._joy_to_gait.get(self.current_mode, {})
                .get(name, {})
                .get(self._switch_flag, "")
            )
            if str(cmd).split("__")[0] == "enable":
                parts = str(cmd).split("__")

                # enable__tjunction__  -> hold modifier for T-junction direction selection
                if len(parts) >= 2 and parts[1] == "tjunction":
                    self._switch_flag = 3
                    self._switch_flag_hold = True
                else:
                    # enable__pole_climb__in/out  -> used to set wt_direction and enable pole_direction selection
                    if len(parts) >= 3 and parts[2] == "in":
                        self._wt_dir = -1
                    else:
                        self._wt_dir = 1
                    self._switch_flag = 2
                    self._switch_flag_hold = True

                index_array = np.delete(index_array, i)
                break

            if not self._switch_flag_hold:
                self._switch_flag = 0

        if index_array.size > 0:
            self._peek(index_array, joy_state)
        else:
            self._mapped_command = ""
            self._prev_index = np.array([])
            self._last_index = None
            if not self._switch_flag_hold:
                self._switch_flag = 0

    def _peek(self, index_array: np.ndarray, joy_state: List[float]) -> None:
        self._prev_index = np.intersect1d(index_array, self._prev_index)

        i = 0
        while i < len(index_array):
            name = self._joystick_mappings.get(int(index_array[i]), "")
            command_name = (
                self._joy_to_gait.get(self.current_mode, {})
                .get(name, {})
                .get(self._switch_flag, "")
            )

            if command_name == "":
                index_array = np.delete(index_array, i)
                continue

            if int(index_array[i]) not in self._prev_index:
                self._mapped_command = str(command_name)
                if self._switch_flag_hold:
                    self._switch_flag_hold = False
                    self._switch_flag = 0

                # Direction for axis-mapped gaits
                if int(index_array[i]) < self._num_axes:
                    self._direction = 1 if float(joy_state[int(index_array[i])]) > 0 else -1

                self._prev_index = np.append(self._prev_index, int(index_array[i]))
                self._last_index = int(index_array[i])
                break

            i += 1

        if len(index_array) == 0:
            self._mapped_command = ""
            self._prev_index = np.array([])
            self._last_index = None
            return

        if self._last_index not in self._prev_index:
            name0 = self._joystick_mappings.get(int(index_array[0]), "")
            self._mapped_command = (
                self._joy_to_gait.get(self.current_mode, {})
                .get(name0, {})
                .get(self._switch_flag, "")
            )

            if int(index_array[0]) < self._num_axes:
                self._direction = 1 if float(joy_state[int(index_array[0])]) > 0 else -1

            self._prev_index = np.array([int(index_array[0])])
            self._last_index = int(index_array[0])

    # --------------------
    # One-step command generation
    # --------------------
    def step(self, print_joy: bool = False) -> TeleopCommand:
        now = self._time_fn()
        self._dt = max(0.001, now - self._last_step_time)
        self._last_step_time = now
        
        axes, buttons, quit_flag, active = self._joy.read()
        if print_joy and active:
            print("[teleop] active:", active)

        self._joy_cb(axes, buttons)

        # =====================================================================
        # Latched auto-run logic (spiral-forward / T-junction transitions)
        # =====================================================================
        if self._mapped_command and self._latch.mode != "idle":
            # Manual override cancels any active latch
            self._cancel_latch()

        # Start latch if requested
        if self._mapped_command:
            cl0 = str(self._mapped_command).split("__")
            if cl0 and cl0[0] == "tjnav":
                kind = cl0[1] if len(cl0) >= 2 else ""

                # "True D-pad" (mapped as l_stick_y here): on most Logitech-style controllers
                # pushing **forward** reports a NEGATIVE value.
                if kind == "straight":
                    if self._direction < 0:
                        self._start_latch_spiral(now)
                elif kind == "turn":
                    turn_dir = int(self._direction)
                    if turn_dir != 0:
                        self._start_latch_transition(now, turn_dir)

                # Avoid re-triggering every frame
                self._mapped_command = ""

        if self._latch.mode != "idle":
            return self._process_latch(now, quit_flag)

        # =====================================================================
        # Normal (non-macro) mapping
        # =====================================================================
        return self._process_normal_command(axes, buttons, quit_flag)

    def _cancel_latch(self) -> None:
        self._latch = TeleopLatchState()

    def _start_latch_spiral(self, now: float) -> None:
        self._init_tj_baseline(kind="straight", direction=0)
        self._latch = TeleopLatchState(
            mode="spiral_fwd",
            start_time=float(now),
            duration_s=float(self._spiral_auto_s),
            timeout_s=0.0,
            direction=0,
        )
        if self.verbose:
            print("[teleop] latch: spiral forward started")

    def _start_latch_transition(self, now: float, direction: int) -> None:
        self._init_tj_baseline(kind="turn", direction=direction)
        self._latch = TeleopLatchState(
            mode="transition_left" if direction < 0 else "transition_right",
            start_time=float(now),
            duration_s=0.0,
            timeout_s=float(self._transition_timeout_s),
            direction=int(direction),
        )
        if self.verbose:
            print(f"[teleop] latch: transition started dir={direction}")

    def _transition_done(self) -> bool:
        done = bool(self._transition_complete)
        if self._transition_complete_fn is not None:
            try:
                done = bool(self._transition_complete_fn())
            except Exception:
                done = bool(done)
        return done

    def _process_latch(self, now: float, quit_flag: bool) -> TeleopCommand:
        elapsed = float(now - self._latch.start_time)

        if self._latch.mode == "spiral_fwd":
            if elapsed >= self._latch.duration_s:
                self._cancel_latch()
                return TeleopCommand(
                    command_name="hold_position",
                    param_name=["speed_multiplier"],
                    param_value=[0.0],
                    quit=bool(quit_flag),
                )
            return self._make_spiral_command(quit_flag)

        if self._latch.mode in ("transition_left", "transition_right"):
            if self._transition_done():
                self._cancel_latch()
                self._transition_complete = False
                return TeleopCommand(
                    command_name="hold_position",
                    param_name=["speed_multiplier"],
                    param_value=[0.0],
                    quit=bool(quit_flag),
                )
            if self._latch.timeout_s > 0 and elapsed >= self._latch.timeout_s:
                self._cancel_latch()
                return TeleopCommand(
                    command_name="hold_position",
                    param_name=["speed_multiplier"],
                    param_value=[0.0],
                    quit=bool(quit_flag),
                )
            return self._make_tjunction_command(quit_flag)

        return TeleopCommand(
            command_name="hold_position",
            param_name=["speed_multiplier"],
            param_value=[0.0],
            quit=bool(quit_flag),
        )

    def _init_tj_baseline(self, kind: str, direction: int) -> None:
        pole_dir = float(self._pole_direction) if self._has_pole_direction and float(self._pole_direction) != 0.0 else 1.0
        if kind == "turn":
            mu_sel, phi0_sel = self._select_turn_params(direction)
        else:
            mu_sel = float((self._n_modules - 1) / 2.0)
            phi0_sel = float(self._tj_cfg.get("phi_0_straight", 0.0))

        self._tj = TJunctionState(
            active=False,
            phase="idle",
            kind=str(kind),
            direction=int(direction),
            n_t=float(self._n_modules - 1),
            s_0=float(self._tj_cfg.get("s_0_start", 0.4)),
            mu=mu_sel,
            phi_0=phi0_sel,
            t0=self._time_fn(),
            phase_start=self._time_fn(),
            baseline_tightness=float(self._tightness) if self._tightness > 0.3 else float(self._tj_cfg.get("tightness_target", 0.8)),
            baseline_wT=float(self._gait_params.get("t_junction", {}).get("wT_even", 2.0)),
            baseline_pole_direction=pole_dir,
            baseline_wt_direction=float(self._wt_dir),
        )

    def _select_turn_params(self, turn_dir: int) -> Tuple[float, float]:
        try:
            prompt = (
                f"[teleop] T-junction: {'LEFT' if turn_dir < 0 else 'RIGHT'} selected.\n"
                f"Enter module number [1..{self._n_modules}] (head=1): "
            )
            s = self._input_fn(prompt)
            m = int(str(s).strip())
            m = max(1, min(self._n_modules, m))
            mu_sel = float(m - 1)
        except Exception:
            mu_sel = float((self._n_modules - 1) / 2.0)

        if turn_dir < 0:
            phi0_sel = float(self._tj_cfg.get("phi_0_left", np.pi / 2))
        else:
            phi0_sel = float(self._tj_cfg.get("phi_0_right", -np.pi / 2))

        return mu_sel, phi0_sel

    def _make_spiral_command(self, quit_flag: bool) -> TeleopCommand:
        params = self._tj_params_base(include_window=True)
        return TeleopCommand(
            command_name="spiraling",
            param_name=list(params.keys()),
            param_value=list(params.values()),
            quit=bool(quit_flag),
        )

    def _make_tjunction_command(self, quit_flag: bool) -> TeleopCommand:
        params = self._tj_params_base(include_window=False)
        return TeleopCommand(
            command_name="t_junction",
            param_name=list(params.keys()),
            param_value=list(params.values()),
            quit=bool(quit_flag),
        )

    def _tj_params_base(self, include_window: bool) -> Dict[str, float]:
        tight_target = self._tj.baseline_tightness
        A1 = float(self._tj_cfg.get("A_1_multiplier", 1.0))
        A2 = float(self._tj_cfg.get("A_2_multiplier", 1.0))
        params = {
            "speed_multiplier": 1.0,
            "tightness": tight_target,
            "pole_direction": self._tj.baseline_pole_direction,
            "wt_direction": self._tj.baseline_wt_direction,
            "A_1_multiplier": A1,
            "A_2_multiplier": A2,
            "mu": float(self._tj.mu),
            "phi_0": float(self._tj.phi_0),
            "s_0": float(self._tj.s_0),
        }
        if include_window:
            params.update(
                {
                    "m": float(self._tj_cfg.get("m", 50.0)),
                    "sig": float(self._tj_cfg.get("sig", 0.05)),
                    "T": float(self._tj_cfg.get("T", 0.25)),
                }
            )
        return params

    def _start_tj_straight(self) -> None:
        """Start T-junction STRAIGHT mode (spiraling only)."""
        # Capture pole_direction like original: must be set AND non-zero
        pole_dir = float(self._pole_direction) if self._has_pole_direction and float(self._pole_direction) != 0.0 else 1.0
        
        print(f"[teleop] TJ_START: _has_pole_direction={self._has_pole_direction}, _pole_direction={self._pole_direction}, captured pole_dir={pole_dir}")
        print(f"[teleop] TJ_START: _tightness={self._tightness}, _wt_dir={self._wt_dir}")
        
        self._tj = TJunctionState(
            active=True,
            phase="pre_hold",
            kind="straight",
            direction=0,
            n_t=float(self._n_modules - 1),  # Start at tail
            s_0=float(self._tj_cfg.get("s_0_start", 0.4)),
            mu=float((self._n_modules - 1) / 2.0),
            phi_0=float(self._tj_cfg.get("phi_0_straight", 0.0)),
            t0=self._time_fn(),
            phase_start=self._time_fn(),
            baseline_tightness=float(self._tightness) if self._tightness > 0.3 else float(self._tj_cfg.get("tightness_target", 0.8)),
            baseline_wT=float(self._gait_params.get("t_junction", {}).get("wT_even", 2.0)),
            baseline_pole_direction=pole_dir,
            baseline_wt_direction=float(self._wt_dir),
        )

    def _start_tj_turn(self, turn_dir: int) -> None:
        """Start T-junction TURN mode with module selection."""
        # Prompt for module index
        try:
            prompt = f"[teleop] T-junction: {'LEFT' if turn_dir < 0 else 'RIGHT'} selected.\nEnter module number [1..{self._n_modules}] (head=1): "
            s = self._input_fn(prompt)
            m = int(str(s).strip())
            m = max(1, min(self._n_modules, m))
            mu_sel = float(m - 1)
        except Exception:
            mu_sel = float((self._n_modules - 1) / 2.0)

        if turn_dir < 0:
            phi0_sel = float(self._tj_cfg.get("phi_0_left", np.pi / 2))
        else:
            phi0_sel = float(self._tj_cfg.get("phi_0_right", -np.pi / 2))

        # Capture pole_direction like original: must be set AND non-zero
        pole_dir = float(self._pole_direction) if self._has_pole_direction and float(self._pole_direction) != 0.0 else 1.0

        # NOTE: input() blocks. Set t0 *after* the prompt so timing starts correctly.
        self._tj = TJunctionState(
            active=True,
            phase="pre_hold",
            kind="turn",
            direction=turn_dir,
            n_t=float(self._n_modules - 1),  # Start spiraling from tail
            s_0=float(self._tj_cfg.get("s_0_start", 0.4)),
            mu=mu_sel,
            phi_0=phi0_sel,
            t0=self._time_fn(),
            phase_start=self._time_fn(),
            baseline_tightness=float(self._tightness) if self._tightness > 0.3 else float(self._tj_cfg.get("tightness_target", 0.8)),
            baseline_wT=float(self._gait_params.get("t_junction", {}).get("wT_even", 2.0)),
            baseline_pole_direction=pole_dir,
            baseline_wt_direction=float(self._wt_dir),
        )
        print(f"[teleop] T-junction: TURN selected, mu={mu_sel:.1f} -> spiraling, then turn...")

    def _process_tj_macro(self, axes: List[float], quit_flag: bool) -> TeleopCommand:
        """Process T-junction macro using n_t sweep (Algorithm 2)."""
        now = self._time_fn()
        
        # Configuration
        use_nt_sweep = bool(self._tj_cfg.get("use_nt_sweep", True))
        pre_hold = float(self._tj_cfg.get("pre_hold_s", 0.25))
        post_hold = float(self._tj_cfg.get("post_hold_s", 0.25))
        debug = bool(self._tj_cfg.get("debug", False))
        
        # Gait parameters
        tight_target = self._tj.baseline_tightness
        A1 = float(self._tj_cfg.get("A_1_multiplier", 1.0))
        A2 = float(self._tj_cfg.get("A_2_multiplier", 1.0))
        m = float(self._tj_cfg.get("m", 50.0))
        sig = float(self._tj_cfg.get("sig", 0.05))
        T = float(self._tj_cfg.get("T", 0.25))

        pole_dir = self._tj.baseline_pole_direction
        wt_dir = self._tj.baseline_wt_direction

        # Active φ₀ control (optional)
        if self._tj_cfg.get("enable_phi0_axis", False):
            phi0_rate_gain = float(self._tj_cfg.get("phi0_rate_gain", 0.5))
            phi0_deadzone = float(self._tj_cfg.get("phi0_deadzone", 0.1))
            r_stick_x = float(axes[2])
            if abs(r_stick_x) > phi0_deadzone:
                self._tj.phi_0 += r_stick_x * phi0_rate_gain * self._dt
                self._tj.phi_0 = _clip(self._tj.phi_0, -np.pi, np.pi)

        # Build base parameters
        # speed_multiplier = 1.0 always (matches Karthik's original)
        base_params = {
            "speed_multiplier": 1.0,
            "tightness": tight_target,
            "pole_direction": pole_dir,
            "wt_direction": wt_dir,
            "A_1_multiplier": A1,
            "A_2_multiplier": A2,
            "mu": self._tj.mu,
            "phi_0": self._tj.phi_0,
            "s_0": self._tj.s_0,
            "n_t": self._tj.n_t,
            "m": m,
            "sig": sig,
            "T": T,
            "l_M": self._l_M,
            "d_M": self._d_M,
            "debug_print": debug,
        }

        def make_cmd(cmd_name: str, mode: str = "spiral") -> TeleopCommand:
            d = dict(base_params)
            d["mode"] = mode
            # Separate numeric and string parameters
            param_name = []
            param_value = []
            for k, v in d.items():
                param_name.append(str(k))
                if isinstance(v, str):
                    # Store string params with a special encoding (mode: 0=spiral, 1=turn, 2=idle)
                    if k == "mode":
                        mode_map = {"spiral": 0.0, "turn": 1.0, "idle": 2.0}
                        param_value.append(mode_map.get(v, 0.0))
                    else:
                        param_value.append(0.0)  # fallback for unknown strings
                elif isinstance(v, bool):
                    param_value.append(1.0 if v else 0.0)
                else:
                    param_value.append(float(v))
            return TeleopCommand(
                command_name=str(cmd_name),
                param_name=param_name,
                param_value=param_value,
                quit=bool(quit_flag),
            )

        elapsed_in_phase = now - self._tj.phase_start

        # =====================================================================
        # State machine for T-junction macro
        # =====================================================================
        
        if self._tj.phase == "pre_hold":
            # Brief hold before starting
            if elapsed_in_phase < pre_hold:
                self._last_sent = "hold_position"
                return TeleopCommand(
                    command_name="hold_position",
                    param_name=["speed_multiplier"],
                    param_value=[0.0],
                    quit=bool(quit_flag),
                )
            # Transition to spiral phase
            self._tj.phase = "spiral"
            self._tj.phase_start = now
            self._tj.n_t = 0.0  # n_t only used in turn phase
            self._tj.s_0 = float(self._tj_cfg.get("s_0_start", 0.4))  # Fixed pulse position

        elif self._tj.phase == "spiral":
            # Spiraling phase (Algorithm 1): fixed time with pulse at s_0
            # The helix rolls via wT, while the pulse at s_0 creates a bump
            # s_0 can be adjusted via joystick if desired
            
            spiral_s = float(self._tj_cfg.get("spiral_s", 10.0))  # Default 10 seconds
            if elapsed_in_phase >= spiral_s:
                if self._tj.kind == "turn":
                    # Transition to turn phase (Algorithm 2)
                    self._tj.phase = "turn"
                    self._tj.phase_start = now
                    self._tj.n_t = 0.0  # Start turn from head
                else:
                    # Straight mode done - just spiraling, no turn
                    self._tj.phase = "post_hold"
                    self._tj.phase_start = now

            self._last_sent = "spiraling"
            # Call spiraling gait directly (like original), not t_junction with mode
            return TeleopCommand(
                command_name="spiraling",
                param_name=["speed_multiplier", "tightness", "pole_direction", "wt_direction",
                           "A_1_multiplier", "A_2_multiplier", "mu", "phi_0", "s_0",
                           "m", "sig", "T"],
                param_value=[1.0, tight_target, pole_dir, wt_dir,
                            A1, A2, self._tj.mu, self._tj.phi_0, self._tj.s_0,
                            m, sig, T],
                quit=bool(quit_flag),
            )

        elif self._tj.phase == "turn":
            # Turn phase: run t_junction gait for fixed time (like original)
            turn_s = float(self._tj_cfg.get("turn_s", 15.0))
            if elapsed_in_phase >= turn_s:
                self._tj.phase = "post_hold"
                self._tj.phase_start = now

            self._last_sent = "t_junction"
            # Call t_junction gait directly (like original), without mode
            return TeleopCommand(
                command_name="t_junction",
                param_name=["speed_multiplier", "tightness", "pole_direction", "wt_direction",
                           "A_1_multiplier", "A_2_multiplier", "mu", "phi_0", "s_0"],
                param_value=[1.0, tight_target, pole_dir, wt_dir,
                            A1, A2, self._tj.mu, self._tj.phi_0, self._tj.s_0],
                quit=bool(quit_flag),
            )

        elif self._tj.phase == "post_hold":
            # Brief hold after completion
            if elapsed_in_phase < post_hold:
                self._last_sent = "hold_position"
                return TeleopCommand(
                    command_name="hold_position",
                    param_name=["speed_multiplier"],
                    param_value=[0.0],
                    quit=bool(quit_flag),
                )
            # Macro complete
            self._tj.phase = "complete"
            self._tj.active = False
            print("[teleop] T-junction macro complete.")
            print(f"[teleop] TJ_END: _has_pole_direction={self._has_pole_direction}, _pole_direction={self._pole_direction}")
            print(f"[teleop] TJ_END: _tightness={self._tightness}, _wt_dir={self._wt_dir}")

        # Macro finished - return to normal
        self._last_sent = "hold_position"
        return TeleopCommand(
            command_name="hold_position",
            param_name=["speed_multiplier"],
            param_value=[0.0],
            quit=bool(quit_flag),
        )

    def _process_normal_command(self, axes: List[float], buttons: List[int], quit_flag: bool) -> TeleopCommand:
        """Process normal (non-TJ-macro) joystick commands."""
        speed_multiplier = 1.0
        command = ""

        if self._mapped_command == "":
            command = "hold_position"
            speed_multiplier = 0.0
            self._slope_sidewind = self._slope_default
        else:
            cl = str(self._mapped_command).split("__")

            if cl[0] == "mode":
                # cache previous mode for head_look exit
                if len(cl) >= 2 and cl[1] == "head_look":
                    self._prev_mode = self.current_mode
                    self.current_mode = "mode__head_look"
                elif len(cl) >= 2 and cl[1] == "head_look_ik":
                    self._prev_mode = self.current_mode
                    self.current_mode = "mode__head_look_ik"
                else:
                    self._prev_mode = self.current_mode
                    self.current_mode = f"mode__{cl[1]}" if len(cl) >= 2 else "mode__normal"

                command = "hold_position"
                speed_multiplier = 0.0
                self._mapped_command = ""  # avoid repeated mode flip

            elif cl[0] == "head_look_exit":
                self.current_mode = self._prev_mode or "mode__normal"
                command = "head_look_exit"
                speed_multiplier = 0.0

            elif cl[0] == "home":
                command = "home"
                speed_multiplier = 0.0
                self._speed_factor = 1.0
                self._has_pole_direction = False
                self._switch_flag = 0
                self._tightness = self._tightness_min
                self._pole_direction = 0
                self._slope_sidewind = self._slope_default

                self._mu_last = 7.5
                self._phi_0_last = 0.0
                self._A_1_multiplier_last = 1.0
                self._A_2_multiplier_last = 1.0
                self._s_0_last = 0.0
                self.current_mode = "mode__normal"

            elif cl[0] == "t_junction":
                command = "t_junction"
                speed_multiplier = 0.0
                self._speed_factor = 1.0

                # t_junction__reset (used in some mappings) resets the rolling-helix knobs
                if len(cl) >= 2 and cl[1] == "reset":
                    self._mu_last = 7.5
                    self._phi_0_last = 0.0
                    self._A_1_multiplier_last = 1.0
                    self._A_2_multiplier_last = 1.0
                    self._s_0_last = 0.0

            elif cl[0] == "g":
                command = cl[1] if len(cl) >= 2 else "hold_position"
                speed_multiplier = 1.0

                if command == "head_look":
                    # ROS code uses joy_state[2:4] for x/y; here axes[2],axes[3]
                    self._x_state, self._y_state = float(axes[2]), float(axes[3])

                if command == "head_look_ik":
                    # axes[:6]
                    self._x_state, self._y_state, self._z_state = float(axes[0]), float(axes[1]), float(axes[2])
                    self._roll, self._pitch, self._yaw = float(axes[3]), float(axes[4]), float(axes[5])

                if command != "conical_sidewinding":
                    self._slope_sidewind = self._slope_default

                if len(cl) >= 3 and cl[2] == "plus":
                    self._direction = 1
                elif len(cl) >= 3 and cl[2] == "minus":
                    self._direction = -1

            elif cl[0] == "g_pole" and self._has_pole_direction:
                command = cl[1] if len(cl) >= 2 else "hold_position"
                speed_multiplier = 1.0

            elif cl[0] == "speed" and command not in ["hold_position", "home"]:
                if len(cl) >= 2 and cl[1] == "plus":
                    self._speed_factor += self._speed_step
                else:
                    self._speed_factor -= self._speed_step
                self._speed_factor = _clip(self._speed_factor, self._speed_min, self._speed_max)

            elif cl[0] == "tightness" and self._has_pole_direction:
                # Increase/decrease tightness (pole climb).
                self._tightness += self._tightness_step if (len(cl) >= 2 and cl[1] == "plus") else -self._tightness_step
                self._tightness = _clip(self._tightness, self._tightness_min, self._tightness_max)

                command = "rolling_helix"

                # If we weren't already rolling, don't force a direction yet.
                if self._last_sent != "rolling_helix":
                    self._direction = 0

                speed_multiplier = 0.0

            elif cl[0] == "slope" and command == "conical_sidewinding":
                if len(cl) >= 2 and cl[1] == "plus":
                    self._slope_sidewind += self._slope_step
                else:
                    self._slope_sidewind -= self._slope_step
                self._slope_sidewind = _clip(self._slope_sidewind, -self._slope_max, self._slope_max)

            elif cl[0] == "pole_direction":
                command = "pole_direction"
                self._pole_direction = self._direction
                speed_multiplier = 0.0
                self._speed_factor = 1.0
                self._has_pole_direction = True

            elif cl[0] == "light_toggle":
                # no-op in simulation
                command = "hold_position"
                speed_multiplier = 0.0

            else:
                command = "hold_position"
                speed_multiplier = 0.0

        # publish-style output
        speed_multiplier *= float(self._direction) * float(self._speed_factor)

        param_name = ["speed_multiplier"]
        param_value = [float(speed_multiplier)]
        
        # Add pole parameters only when pole climbing direction is set
        if self._has_pole_direction:
            param_name.extend(["tightness", "pole_direction", "wt_direction"])
            param_value.extend([float(self._tightness), float(self._pole_direction), float(self._wt_dir)])

        if command == "head_look":
            param_name.extend(["x_state", "y_state"])
            param_value.extend([float(self._x_state), float(self._y_state)])

        if command == "head_look_ik":
            param_name.extend(["x_state", "y_state", "z_state", "roll", "pitch", "yaw"])
            param_value.extend(
                [
                    float(self._x_state),
                    float(self._y_state),
                    float(self._z_state),
                    float(self._roll),
                    float(self._pitch),
                    float(self._yaw),
                ]
            )

        if command == "conical_sidewinding":
            param_name.append("slope")
            param_value.append(float(self._slope_sidewind))

        self._last_sent = str(command)

        return TeleopCommand(
            command_name=str(command),
            param_name=param_name,
            param_value=param_value,
            quit=bool(quit_flag),
        )
