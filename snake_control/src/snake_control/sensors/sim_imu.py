# snake_control/src/snake_control/sensors/sim_imu.py
"""Simple 'virtual IMU' helpers for PyBullet simulation.

In the real robot, an IMU gives you orientation relative to gravity (and often yaw via magnetometer).
In PyBullet we can read the base link quaternion and derive similar signals.

This module focuses on one practical use-case for SEA snake teleop:
- Correcting pole-climb wrapping-direction inversion when the robot is effectively 'rolled' w.r.t. the
  global frame (so left/right from a top view can flip even if joint-space signs are consistent).

We intentionally keep this lightweight (no external deps).
"""

from __future__ import annotations

from typing import Optional, Tuple, List

from snake_bullet.sim_env import RobotState


def _rotmat_from_quat(quat_xyzw: Tuple[float, float, float, float]) -> List[List[float]]:
    """Return 3x3 rotation matrix (world_R_body) from quaternion (x,y,z,w)."""
    # Avoid importing pybullet here; implement the standard formula.
    x, y, z, w = [float(v) for v in quat_xyzw]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ]


def _matvec(R: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def pole_direction_sign_from_imu(state: RobotState, deadband: float = 0.2) -> float:
    """Return a sign (+1 or -1) used to correct pole_direction using a simulated IMU.

    Heuristic:
      - Use the base link 'up' axis (body +Z) in world frame.
      - If it points mostly *down* (dot(world_up, body_up) < -deadband), we consider the robot inverted
        for left/right purposes, and return -1.
      - If it points mostly *up* (dot > +deadband), return +1.
      - If it is near perpendicular (|dot| <= deadband), return +1 (do not flip to avoid chatter).

    This matches the common real-robot practice: when the body rolls ~180 degrees, left/right commands
    should be swapped in a global (top-view) interpretation.

    Returns:
      +1.0 or -1.0
    """
    if state.base_quat is None:
        return 1.0

    R = _rotmat_from_quat(state.base_quat)
    body_up_world = _matvec(R, (0.0, 0.0, 1.0))  # world_R_body * z_body
    dot = float(body_up_world[2])  # dot(body_up, world_up) since world_up=(0,0,1)

    if dot < -abs(float(deadband)):
        return -1.0
    return 1.0
