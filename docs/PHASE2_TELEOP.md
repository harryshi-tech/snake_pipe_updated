# Phase 2 Teleop Latching

This note documents the new latch behaviors in `snake_control/teleop/joystick_teleop.py`.

## YAML additions

Added under each `teleop:` section in `snake_control/param/snake_params.yaml`:

```yaml
teleop:
  spiral_forward_auto_duration_s: 2.0
  transition_auto_timeout_s: 6.0
```

- **`spiral_forward_auto_duration_s`**: duration for the spiral-forward latch (seconds).
- **`transition_auto_timeout_s`**: safety timeout for left/right transition latch (seconds).

## Behavior summary

- **Spiral-forward latch**: Triggered by the existing `tjnav__straight__` mapping (no new mappings). Press once to start spiraling for the configured duration, then return to idle.
- **Transition latch**: Triggered by the existing `tjnav__turn__` mapping (left/right). Runs until the transition completion signal is received or timeout expires.
- **Manual override**: Any new explicit gait/mode trigger cancels the latch immediately.

## Tests

Run unit tests (no pygame hardware required):

```bash
python3 -m pytest -q references/snake_pipe_updated/tests/test_joystick_teleop_latch.py
```
