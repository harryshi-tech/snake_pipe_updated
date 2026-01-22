# snake_pipe (ROS-agnostic)

This repo contains:

* **snake_bullet**: PyBullet simulation backend
* **snake_control**: controllers, gait library, and joystick teleop
* **snakelib_description**: URDF + meshes

## Run the simulator

From the repo root:

```bash
PYTHONPATH=snake_bullet/src:snake_control/src python snake_bullet/run_sim.py
```

Configuration is in:

* `snake_bullet/param/sim_params.yaml`

In that YAML you can:

* choose the controller (`runner.controller.type`)
* enable/disable joystick teleop (`runner.teleop.enable`)
* configure terminal printing (`runner.print.*`)

## T-junction demo (no teleop)

This update adds standalone T-junction navigation gaits:

* `t_junction` (blends windowed rolling-helix → spiraling)
* `windowed_rolling_helix`
* `spiraling`

Run the built-in demo config (junction world, teleop disabled):

```bash
PYTHONPATH=snake_bullet/src:snake_control/src python snake_bullet/run_sim.py --cfg snake_bullet/param/sim_params_tjunction.yaml
```

To try components, change `runner.controller.gait` in the YAML to:
`windowed_rolling_helix` or `spiraling`.
