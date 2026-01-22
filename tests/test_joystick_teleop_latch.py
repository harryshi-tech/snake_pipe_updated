from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from pathlib import Path

from snake_control.teleop.joystick_teleop import RosLikeJoystickTeleop


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += float(dt)


class FakeJoyAdapter:
    def __init__(self, frames: List[Tuple[List[float], List[int]]]):
        self.frames = list(frames)
        self.last = ([0.0] * 6, [0] * 12)

    def read(self):
        if self.frames:
            axes, buttons = self.frames.pop(0)
            self.last = (axes, buttons)
        else:
            axes, buttons = self.last
        return axes, buttons, False, []


def _make_teleop(clock: FakeClock, frames: List[Tuple[List[float], List[int]]]) -> RosLikeJoystickTeleop:
    base_dir = Path(__file__).resolve().parents[1]
    joy = FakeJoyAdapter(frames)
    teleop = RosLikeJoystickTeleop(
        snake_type="REU",
        snake_params_yaml=base_dir / "snake_control" / "param" / "snake_params.yaml",
        joy_to_gait_yaml=base_dir / "snake_control" / "param" / "joy_to_gait.yaml",
        joy_mapping_yaml=base_dir
        / "snake_control"
        / "param"
        / "logi_cordless_rumblepad2_mappings.yaml",
        joy_adapter=joy,
        time_fn=clock.now,
        input_fn=lambda _: "1",
    )
    teleop.current_mode = "mode__pole_climb"
    teleop._spiral_auto_s = 0.2
    teleop._transition_timeout_s = 0.5
    return teleop


def _buttons_with(**kwargs) -> List[int]:
    buttons = [0] * 12
    mapping = {
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
    for name, val in kwargs.items():
        buttons[mapping[name]] = int(val)
    return buttons


def test_spiral_forward_latch_expires() -> None:
    clock = FakeClock()
    frames = [
        ([0.0] * 6, _buttons_with(B=1)),  # enable tjunction (switch_flag=3)
        ([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], _buttons_with()),  # l_stick_y forward
        ([0.0] * 6, _buttons_with()),
    ]
    teleop = _make_teleop(clock, frames)

    cmd = teleop.step()
    clock.advance(0.05)
    cmd = teleop.step()
    assert cmd.command_name == "spiraling"

    clock.advance(0.1)
    cmd = teleop.step()
    assert cmd.command_name == "spiraling"

    clock.advance(0.2)
    cmd = teleop.step()
    assert cmd.command_name == "hold_position"


def test_transition_latch_completes_on_flag() -> None:
    clock = FakeClock()
    frames = [
        ([0.0] * 6, _buttons_with(B=1)),
        ([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], _buttons_with()),  # l_stick_x left
    ]
    teleop = _make_teleop(clock, frames)

    teleop.step()
    clock.advance(0.05)
    cmd = teleop.step()
    assert cmd.command_name == "t_junction"

    teleop.set_transition_complete(True)
    clock.advance(0.05)
    cmd = teleop.step()
    assert cmd.command_name == "hold_position"


def test_manual_override_cancels_latch() -> None:
    clock = FakeClock()
    frames = [
        ([0.0] * 6, _buttons_with(B=1)),
        ([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], _buttons_with()),
        ([0.0] * 6, _buttons_with()),
    ]
    teleop = _make_teleop(clock, frames)

    teleop.step()
    clock.advance(0.05)
    teleop.step()  # latch running

    teleop.current_mode = "mode__normal"
    teleop._joy.frames.insert(0, ([0.0] * 6, _buttons_with(l_bumper=1)))
    clock.advance(0.05)
    cmd = teleop.step()
    assert cmd.command_name == "rolling"


def test_new_latch_wins_over_old() -> None:
    clock = FakeClock()
    frames = [
        ([0.0] * 6, _buttons_with(B=1)),
        ([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], _buttons_with()),  # spiral latch
        ([0.0] * 6, _buttons_with(B=1)),
        ([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], _buttons_with()),  # transition latch
    ]
    teleop = _make_teleop(clock, frames)

    teleop.step()
    clock.advance(0.05)
    teleop.step()

    clock.advance(0.05)
    teleop.step()
    clock.advance(0.05)
    cmd = teleop.step()
    assert cmd.command_name == "t_junction"
