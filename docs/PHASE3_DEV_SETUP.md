# Phase 3: Dev Setup (fresh venv)

This repo is ROS-agnostic. Use the commands below in a clean Python 3.8+ virtualenv
to install dependencies and run the Phase 2 tests locally. These are the exact
local commands that were validated.

## Reproducible local workflow (verbatim)

```bash
cd ~/snake_pipe
python3 -m venv --copies .venv
source .venv/bin/activate
pip install pybullet numpy pyyaml pygame matplotlib
python -m pip install -U pip setuptools wheel pytest
pip install -e references/snake_pipe_updated/snake_control
pip install -e references/snake_pipe_updated/snake_bullet
python -m pytest references/snake_pipe_updated/tests/test_gait_equivalence.py -q
python -m pytest references/snake_pipe_updated/tests/test_joystick_teleop_latch.py -q
```

## Common pitfalls

- If YAML path errors happen, run tests from inside `references/snake_pipe_updated/`
  or ensure the package is installed editable.
- Ensure `pyyaml` is installed (required by `snake_control`).
- ROS is **not** required for these tests.

## Related docs

- Phase 2 teleop notes: `docs/PHASE2_TELEOP.md`
- Gait equivalence harness: `docs/GAIT_EQUIVALENCE.md`
