# Reference Mapping (Phase 0)

This document maps the gait equations and control/state-machine logic in **Karthik’s legacy reference** (`references/snakes_on_pipes-main`) to the **current working codebase** (`references/snake_pipe_updated`). It also notes key deltas in `snakes_on_pipes-main_new` that are relevant for parameter/schema updates (e.g., `delta_even`) but should *not* override the older gait math unless explicitly called out.

## 1) Gait Equation Mapping (authoritative math = `snakes_on_pipes-main`)

| Behavior | Primary reference (snakes_on_pipes-main) | Secondary reference (snakes_on_pipes-main_new) | Current implementation target (snake_pipe_updated) |
| --- | --- | --- | --- |
| **Compound serpenoid core** (even/odd joint equation) | `snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py` (base serpenoid equation).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L1-L31】 | Adds `delta_even` in the even-term phase (schema update).【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L1-L31】 | `snake_control/gaitlib/reu_gaits/compound_serpenoid.py` (current baseline).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/compound_serpenoid.py†L1-L31】 |
| **Rolling helix** (tightness→amplitude/spatial frequency rule + serpenoid eval) | `snakelib_control/gaitlib/reu_gaits/rolling_helix.py` (tightness update, then compound serpenoid).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L1-L76】 | Same shape; adds minor logging in new branch (not a math change). | `snake_control/gaitlib/reu_gaits/rolling_helix.py` (current implementation).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/rolling_helix.py†L1-L77】 |
| **T-junction gait (local equations)** (windowed rolling-helix + spiraling blend) | `snakelib_control/gaitlib/reu_gaits/t_junction.py` (full local math including spiraling window blend).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L1-L188】 | Refactored to manager handoff (no local equations).【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L1-L45】 | `snake_control/gaitlib/reu_gaits/t_junction.py` (local math version; same structure as old reference).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/t_junction.py†L1-L233】 |
| **Windowed rolling-helix** (pre-blend portion of T-junction) | Embedded in old T-junction equation block (see reference above).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L124-L156】 | Not present (logic moved to manager). | `snake_control/gaitlib/reu_gaits/windowed_rolling_helix.py` (explicit pre-blend extractor).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/windowed_rolling_helix.py†L1-L103】 |
| **Spiraling (bump helix + sinusoidal blend)** | Embedded inside old T-junction equation block (spiraling portion).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L98-L188】 | A standalone `spiralling.py` exists, but it only performs ROS handoff logic (no local math).【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/spiralling.py†L1-L42】 | `snake_control/gaitlib/reu_gaits/spiraling.py` (local math version derived from the old T-junction block).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/spiraling.py†L1-L213】 |
| **Other gaits** (lateral undulation, slithering, turn-in-place, rolling, etc.) | Defined in `snakelib_control/gaitlib/reu_gaits/*.py` (one file per gait).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling.py†L1-L36】 | No major changes beyond parameter additions in YAML. | `snake_control/gaitlib/reu_gaits/*.py` (mirrors original layout).【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/rolling.py†L1-L36】 |

## 2) Control / State-Machine Mapping

| Behavior | Primary reference (snakes_on_pipes-main) | Current implementation target (snake_pipe_updated) |
| --- | --- | --- |
| **Command dispatch / controller switching** | `snakelib_control/command_manager.py` (ROS message-based command manager).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/command_manager.py†L10-L240】 | `snake_control/controllers/gait_position.py` (ROS-agnostic controller with similar lab-style gait semantics).【F:references/snake_pipe_updated/snake_control/src/snake_control/controllers/gait_position.py†L47-L235】 |
| **Gaitlib controller semantics** (transition blending, parameter updates, extra inputs) | `snakelib_control/gaitlib_controller.py` (transition blending + per-gait extra inputs).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib_controller.py†L11-L189】 | `snake_control/controllers/gait_position.py` (transition blending, extra gait inputs, snake_time update).【F:references/snake_pipe_updated/snake_control/src/snake_control/controllers/gait_position.py†L47-L233】 |
| **T-junction control sequencing** | Embedded in original T-junction gait math (no external manager).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L98-L188】 | Teleop/controller scaffolding exists in `snake_control/teleop/joystick_teleop.py` + individual gaits (`t_junction`, `spiraling`, `windowed_rolling_helix`).【F:references/snake_pipe_updated/snake_control/src/snake_control/teleop/joystick_teleop.py†L703-L749】【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/t_junction.py†L1-L184】 |

## 2.1) Gait Parameter Flow (where parameters enter the system)

This section clarifies how parameters flow from YAML/teleop into gait evaluation so later ports can be surgical without changing joystick mappings.

1. **Defaults**: Loaded from `snake_control/param/snake_params.yaml`, per-snake under `gait_params`.【F:references/snake_pipe_updated/snake_control/param/snake_params.yaml†L24-L193】
2. **Runtime overrides**: Injected by controllers (e.g., teleop changes), merged into `GaitPositionController` overrides and passed into the gait runner.【F:references/snake_pipe_updated/snake_control/src/snake_control/controllers/gait_position.py†L172-L193】
3. **Pole-climb group**: `pole_climb` parameters are merged separately and passed as `pole_params` to gaits that accept them (rolling_helix/t_junction).【F:references/snake_pipe_updated/snake_control/src/snake_control/controllers/gait_position.py†L130-L191】

## 3) Parameter Schema & Naming Notes

### Authoritative base parameters (snakes_on_pipes-main)
The canonical parameter names (e.g., `beta_even`, `beta_odd`, `A_even`, `A_odd`, `wS_even`, `wS_odd`, `wT_even`, `wT_odd`, `delta`, `speed_multiplier`) are defined in `snakelib_control/param/snake_params.yaml` for both REU and SEA snakes.【F:references/snakes_on_pipes-main/snakelib_control/param/snake_params.yaml†L33-L133】

### Schema deltas in snakes_on_pipes-main_new
The newer reference adds `delta_even` to most gaits and updates some parameter defaults; this is a **schema update** to allow independent even/odd phase offsets, but it is *not* part of the original gait math baseline unless explicitly desired in snake_pipe_updated.【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/param/snake_params.yaml†L33-L153】

### Current snake_pipe_updated parameter layout
`snake_pipe_updated` already tracks a superset of the legacy parameters (including `t_junction`, `spiraling`, and `windowed_rolling_helix`), with the older naming (`delta` only) in the default YAML. This file is the current binding for controller defaults and teleop overrides.【F:references/snake_pipe_updated/snake_control/param/snake_params.yaml†L24-L193】

### Known parameter gaps vs. schema updates
* `delta_even` is not present in the current `snake_pipe_updated` YAML defaults or compound serpenoid equation, which matches the older reference but diverges from the newer schema. This is a **schema decision point** for later phases.【F:references/snake_pipe_updated/snake_control/param/snake_params.yaml†L24-L193】【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/compound_serpenoid.py†L1-L31】

## 4) Parameter Meaning Cheat-Sheet (legacy semantics)

The following meanings are based on the original `snakes_on_pipes-main` behavior, and should be preserved when porting math and state logic:

- **`beta_even`, `beta_odd`**: baseline offsets for even/odd joints in the compound serpenoid equation.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L10-L31】
- **`A_even`, `A_odd`**: even/odd amplitude terms in the compound serpenoid equation; updated by tightness logic in rolling-helix/t-junction gaits.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L43-L63】
- **`wS_even`, `wS_odd`**: spatial frequencies, used with module index `n` and tightened based on `A_transition` in rolling-helix/t-junction logic.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L43-L63】
- **`wT_even`, `wT_odd`**: temporal frequencies, used with time `t` to advance gait phase.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L10-L31】
- **`delta`**: phase offset applied to odd joints (and even joints in the legacy model).【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L10-L31】
- **`delta_even`** (new branch only): even-joint phase offset added in `snakes_on_pipes-main_new`.【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/compound_serpenoid.py†L10-L31】
- **`tightness`, `pole_direction`**: rolling-helix/t-junction controls that modulate amplitude and spatial frequency based on pole-wrapping direction and “tightness”.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L43-L63】
- **`A_transition`, `A_max`, `dWs_dAodd`** (pole-climb group): constants controlling the tightness→(A, wS) mapping in rolling-helix/t-junction gaits.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L15-L60】
- **T-junction blend controls** (`A_1_multiplier`, `A_2_multiplier`, `mu`, `phi_0`, `s_0`, `m`, `sig`, `T`): windowed amplitudes and sinusoidal blending used to merge the baseline helix with the “bump” helix for spiraling/turning behavior.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L98-L188】

---

## 5) Summary of Phase-0 Findings

- The **authoritative gait math** for rolling-helix, the T-junction blending, and the spiraling bump helix lives in `snakes_on_pipes-main`’s gaitlib files, especially `t_junction.py` and `rolling_helix.py`.【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L1-L188】【F:references/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/rolling_helix.py†L1-L76】
- `snakes_on_pipes-main_new` should be used **only for schema/parameter updates** (like `delta_even`) and ROS integration patterns; it offloads T-junction logic to a manager node and does not preserve the local equations.【F:references/snakes_on_pipes-main_new/snakes_on_pipes-main/snakelib_control/src/snakelib_control/gaitlib/reu_gaits/t_junction.py†L1-L45】
- `snake_pipe_updated` already has a **local T-junction/spiraling implementation** that mirrors the old gait math; this will be the main surface for Phase 1+ ports and equivalence testing.【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/t_junction.py†L1-L233】【F:references/snake_pipe_updated/snake_control/src/snake_control/gaitlib/reu_gaits/spiraling.py†L1-L213】
