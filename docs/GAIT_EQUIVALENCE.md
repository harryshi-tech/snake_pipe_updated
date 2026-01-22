# Gait Equivalence Harness (Phase 1)

This document explains how to run the gait waveform comparison tool and the unit tests
that compare `snake_pipe_updated` against the authoritative `snakes_on_pipes-main`
reference math.

## 1) Waveform comparison tool

Runs a time-grid comparison and prints max/RMS errors for each gait/parameter set.
CSV outputs are written to `references/snake_pipe_updated/tools/out/` by default.

```bash
python references/snake_pipe_updated/tools/compare_gaits.py
```

To disable CSV outputs:

```bash
python references/snake_pipe_updated/tools/compare_gaits.py --no-csv
```

## 2) Automated tests

The tests compare the reference and target gait waveforms on a fixed time grid.

```bash
pytest references/snake_pipe_updated/tests/test_gait_equivalence.py
```

## Notes

- These tests are ROS-agnostic and do not require PyBullet.
- The reference implementation is loaded only inside the tool/test using a temporary
  `sys.path` override (never in production modules).
