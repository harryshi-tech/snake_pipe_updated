# snake_bullet/src/snake_bullet/sim_env.py
#
# PyBullet simulation backend (ROS-agnostic, ROS-ready).
# Supports worlds (sim_params.yaml):
#   - flat
#   - pipe
#   - junction (vertical + horizontal branch, arbitrary branch_angle [rad])
#   - varying_pipe_3section (lower -> transition -> upper; transition approximated by stacked cylinders)
#
# Notes:
# - This builds SOLID cylinders (placeholders). It’s enough to validate geometry, placement,
#   and controller integration. Later we can switch to hollow pipe collision (inside-the-pipe).
# - World placement uses the same robust idea as the lab PipeWorld code:
#   stack segments along LOCAL Z, and rotate offsets into WORLD coordinates.

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import math

import pybullet as p
import pybullet_data

import numpy as np
import yaml


# -----------------------------
# ROS-ready, ROS-agnostic types
# -----------------------------
@dataclass
class RobotState:
    t: float
    dt: float
    q: List[float]
    dq: List[float]
    tau: List[float]
    joint_names: Optional[List[str]] = None
    # Optional pose/velocity (simulated IMU-friendly fields).
    # Filled by snake_bullet.SimEnv.get_state(); None if unavailable.
    base_pos: Optional[Tuple[float, float, float]] = None
    base_quat: Optional[Tuple[float, float, float, float]] = None  # xyzw
    base_lin_vel: Optional[Tuple[float, float, float]] = None
    base_ang_vel: Optional[Tuple[float, float, float]] = None


@dataclass
class JointCommand:
    mode: str = "position"  # "position" | "velocity" | "torque"
    position: Optional[List[float]] = None
    velocity: Optional[List[float]] = None
    effort: Optional[List[float]] = None


def _quat_from_euler(euler_xyz: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler([float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2])])


def _rotmat_from_quat(quat_xyzw: Tuple[float, float, float, float]) -> List[List[float]]:
    m = p.getMatrixFromQuaternion(quat_xyzw)  # 9 vals row-major
    return [
        [float(m[0]), float(m[1]), float(m[2])],
        [float(m[3]), float(m[4]), float(m[5])],
        [float(m[6]), float(m[7]), float(m[8])],
    ]


def _matvec(R: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def _qmul(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    # quaternion composition using bullet: q = q1 * q2
    _, q = p.multiplyTransforms([0.0, 0.0, 0.0], q1, [0.0, 0.0, 0.0], q2)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


class SimEnv:
    """
    Bullet simulation backend.

    Public API:
      - get_state() -> RobotState
      - apply_command(cmd: JointCommand) -> None
      - step() -> None
      - close() -> None
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

        # ---- sim ----
        sim_cfg = cfg.get("sim", {})
        self.gui = bool(sim_cfg.get("gui", True))
        self.realtime = bool(sim_cfg.get("realtime", True))
        self.dt = float(sim_cfg.get("timestep", 1.0 / 240.0))
        self.substeps = int(sim_cfg.get("substeps", 1))
        self.gravity = sim_cfg.get("gravity", [0.0, 0.0, -9.81])
        self.enable_file_caching = bool(sim_cfg.get("enable_file_caching", False))

        # ---- control ----
        ctrl_cfg = cfg.get("control", {})
        self.control_mode = str(ctrl_cfg.get("mode", "position")).lower()
        self.max_torque = float(ctrl_cfg.get("max_torque", 7.0))

        # Optional sign map to reconcile per-joint sign conventions.
        # This is useful when the joint axis definition in the URDF causes
        # some joints to report torques with the opposite sign relative to
        # (q_cmd - q_meas) when using Bullet's reaction/contact torque signals.
        #
        # Recommended usage for minimal disruption:
        #   - set tau_sign_map (path) and enable apply_tau_sign_map
        #   - keep apply_state_sign_map/apply_cmd_sign_map false
        #
        # YAML example:
        #   control:
        #     tau_sign_map: "snake_bullet/param/joint_sign_map_sea.yaml"
        #     apply_tau_sign_map: true
        self.tau_sign_map_path = str(
            ctrl_cfg.get("tau_sign_map", ctrl_cfg.get("joint_sign_map", ""))
        ).strip()
        self.apply_tau_sign_map = bool(ctrl_cfg.get("apply_tau_sign_map", False))
        self.apply_state_sign_map = bool(ctrl_cfg.get("apply_state_sign_map", False))
        self.apply_cmd_sign_map = bool(ctrl_cfg.get("apply_cmd_sign_map", False))
        self._joint_sign: Optional[List[float]] = None

        # Which torque signal should populate RobotState.tau?
        # - "applied": motor effort applied by Bullet (getJointState()[3])
        # - "contact": generalized joint torques induced by external contacts
        #              (mapped from contact forces via J^T F)
        self.torque_reading = str(
            ctrl_cfg.get("torque_reading", ctrl_cfg.get("tau_reading", ctrl_cfg.get("tau_source", "applied")))
        ).lower()

        # Optional filtering for the torque signal returned in RobotState.tau.
        # This is useful because contact/reaction-based torques are impulsive/noisy.
        #
        # YAML examples:
        #   control:
        #     torque_reading: "reaction"   # or "contact"
        #     tau_filter:
        #       type: "median_lpf"         # "none" | "lpf" | "median" | "median_lpf"
        #       window: 5                  # for median filters
        #       tau: 0.05                  # [s] for lpf
        #     tau_deadband: 0.1            # [Nm]
        tf_cfg = ctrl_cfg.get("tau_filter", ctrl_cfg.get("torque_filter", {}))
        if isinstance(tf_cfg, str):
            tf_cfg = {"type": tf_cfg}
        self.tau_filter_type = str(tf_cfg.get("type", tf_cfg.get("mode", "none"))).lower()
        self.tau_filter_tau = float(tf_cfg.get("tau", tf_cfg.get("lpf_tau", 0.0)) or 0.0)
        self.tau_filter_window = int(tf_cfg.get("window", tf_cfg.get("median_window", 5)) or 5)
        self.tau_deadband = float(ctrl_cfg.get("tau_deadband", ctrl_cfg.get("torque_deadband", 0.0)) or 0.0)

        # Filter runtime state (initialized once joint_idx is known).
        self._tau_filter_inited = False
        self._tau_lpf_state: Optional[List[float]] = None
        self._tau_med_bufs: Optional[List[deque]] = None

        # Cache: list of Bullet joint indices that have 1-DoF (revolute/prismatic).
        # PyBullet's calculateJacobian expects joint arrays sized to the number of DoFs,
        # NOT p.getNumJoints().
        self._dof_joints: Optional[List[int]] = None

        # Optional Bullet PD gains for POSITION_CONTROL / VELOCITY_CONTROL.
        # If left unset, PyBullet's internal defaults are used.
        # NOTE: Lowering these gains is often the easiest way to reduce
        # torque saturation while keeping reasonable tracking.
        pg = ctrl_cfg.get("position_gain", ctrl_cfg.get("pos_gain", None))
        vg = ctrl_cfg.get("velocity_gain", ctrl_cfg.get("vel_gain", None))
        self.position_gain = None if pg is None else float(pg)
        self.velocity_gain = None if vg is None else float(vg)

        # ---- camera ----
        cam_cfg = cfg.get("camera", {})
        self.tracking_cam = bool(cam_cfg.get("tracking", False))
        self.cam_distance = float(cam_cfg.get("distance", 2.0))
        self.cam_yaw = float(cam_cfg.get("yaw", 45.0))
        self.cam_pitch = float(cam_cfg.get("pitch", -45.0))
        self.cam_target = str(cam_cfg.get("target", "mid_joint")).lower()

        # ---- dynamics defaults ----
        dyn_cfg = cfg.get("dynamics", {})
        self.dyn_default = {
            "lateral_friction": float(dyn_cfg.get("lateral_friction", 1.0)),
            "spinning_friction": float(dyn_cfg.get("spinning_friction", 0.0)),
            "rolling_friction": float(dyn_cfg.get("rolling_friction", 0.0)),
            "restitution": float(dyn_cfg.get("restitution", 0.0)),
            "friction_anchor": int(dyn_cfg.get("friction_anchor", 1)),
        }

        # ---- robot ----
        robot_cfg = cfg.get("robot", {})
        self.urdf_path_cfg = str(robot_cfg.get("urdf_path", "")).strip()
        self.base_pos = robot_cfg.get("base_pos", [0.0, 0.0, 0.25])
        self.base_rpy = robot_cfg.get("base_rpy", [0.0, 0.0, 0.0])
        self.fixed_base = bool(robot_cfg.get("fixed_base", False))
        self.self_collision = bool(robot_cfg.get("self_collision", True))

        self.motor_idx_rule = str(robot_cfg.get("motor_idx_rule", "auto_revolute")).lower()
        self.explicit_motor_indices = robot_cfg.get("motor_indices", [])
        # Joint order convention:
        # Prefer "reverse_joint_order". Also accept legacy typo "reverser_joint_order".
        self.reverse_joint_order = bool(
            robot_cfg.get("reverse_joint_order", robot_cfg.get("reverser_joint_order", False))
        )

        # runtime
        self.client_id: Optional[int] = None
        self.robot_id: Optional[int] = None
        self.joint_idx: List[int] = []
        self.joint_names: List[str] = []

        self._t_sim = 0.0
        self._t_wall_last = time.time()

        # keep created world body ids (for cleanup)
        self.world_bodies: List[int] = []

        # Subset of world bodies that correspond to pipe geometry (excluding ground plane).
        # Used for logging pipe-only contact metrics (normal force, contact count, etc.).
        self.pipe_bodies: List[int] = []

        self._connect()
        self._configure_physics()

        # Worlds (expects list)
        worlds_cfg = cfg.get("worlds", [])
        if not isinstance(worlds_cfg, list):
            raise ValueError("cfg['worlds'] must be a list of world entries.")
        self._build_worlds(worlds_cfg)

        # Robot
        self._load_robot()
        self._apply_robot_dynamics(self.dyn_default)
        self._enable_joint_torque_sensors(True)

        # Optional per-joint sign map (used to sign-correct returned tau/q/dq and/or commands).
        self._init_joint_sign_map()

        # Sign-map init (needs joint_names)
        self._init_joint_sign_map()

    # -----------------------------
    # setup / teardown
    # -----------------------------
    def _connect(self) -> None:
        self.client_id = p.connect(p.GUI if self.gui else p.DIRECT)
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Keep Bullet data path so plane.urdf works
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setPhysicsEngineParameter(enableFileCaching=int(self.enable_file_caching))


    def _configure_physics(self) -> None:
        p.setTimeStep(self.dt)
        p.setGravity(*self.gravity)

    def close(self) -> None:
        if self.client_id is not None:
            try:
                for bid in self.world_bodies:
                    try:
                        p.removeBody(bid)
                    except Exception:
                        pass
                self.world_bodies = []
                if self.robot_id is not None:
                    try:
                        p.removeBody(self.robot_id)
                    except Exception:
                        pass
                self.robot_id = None
                p.disconnect(self.client_id)
            finally:
                self.client_id = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -----------------------------
    # joint sign map utilities
    # -----------------------------
    @staticmethod
    def _repo_root_guess() -> Path:
        """Best-effort guess of the repository root.

        sim_env.py lives at: <repo>/snake_bullet/src/snake_bullet/sim_env.py
        so parents[3] should be <repo>.
        """
        try:
            return Path(__file__).resolve().parents[3]
        except Exception:
            return Path.cwd()

    @staticmethod
    def _resolve_repo_path(pth: str) -> Path:
        pth = str(pth).strip()
        if not pth:
            return Path("")
        p = Path(pth)
        return p if p.is_absolute() else (SimEnv._repo_root_guess() / p)

    @staticmethod
    def _load_joint_sign_map_yaml(pth: Path) -> Dict[str, float]:
        """Load a joint->sign mapping from YAML.

        Expected format:
          joint_sign:
            SA001__MoJo: 1
            SA002__MoJo: -1
        """
        if pth is None or str(pth) == "":
            return {}
        if not pth.exists():
            return {}
        try:
            data = yaml.safe_load(pth.read_text()) or {}
            m = data.get("joint_sign", data.get("joint_sign_map", data.get("sign_map", {})))
            if not isinstance(m, dict):
                return {}
            out: Dict[str, float] = {}
            for k, v in m.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
            return out
        except Exception:
            return {}

    def _init_joint_sign_map(self) -> None:
        """Initialize per-controlled-joint sign array if configured."""
        if not self.tau_sign_map_path:
            self._joint_sign = None
            return

        pth = self._resolve_repo_path(self.tau_sign_map_path)
        m = self._load_joint_sign_map_yaml(pth)
        if not m:
            self._joint_sign = None
            return

        signs: List[float] = []
        for jn in self.joint_names:
            s = float(m.get(jn, 1.0))
            if s == 0.0:
                s = 1.0
            signs.append(1.0 if s >= 0.0 else -1.0)
        self._joint_sign = signs

        # Lightweight visibility (doesn't spam): only prints once at init.
        try:
            n_neg = sum(1 for s in signs if s < 0)
            print(f"[SimEnv] Loaded tau_sign_map: {pth}  (n_joints={len(signs)}, neg={n_neg})")
        except Exception:
            pass

    @staticmethod
    def _apply_sign(vec: List[float], signs: Optional[List[float]]) -> List[float]:
        if not signs:
            return vec
        n = min(len(vec), len(signs))
        out = list(vec)
        for i in range(n):
            out[i] = float(signs[i]) * float(out[i])
        return out

    # -----------------------------
    # worlds
    # -----------------------------
    def _build_worlds(self, worlds_cfg: List[Dict[str, Any]]) -> None:
        for w in worlds_cfg:
            wtype = str(w.get("type", "")).lower().strip()

            if wtype == "flat":
                self.world_bodies.append(self._add_flat(w))

            elif wtype == "pipe":
                self.world_bodies.append(self._add_straight_pipe_from_cfg(w))

            elif wtype == "junction":
                self.world_bodies.extend(self._add_junction_from_cfg(w))

            elif wtype == "varying_pipe_3section":
                self.world_bodies.extend(self._add_varying_pipe_3section_from_cfg(w))

            else:
                raise ValueError(f"Unknown world type: '{wtype}'")

    def _apply_body_dynamics(self, body_id: int, dyn_override: Optional[Dict[str, Any]] = None) -> None:
        dyn = dict(self.dyn_default)
        if isinstance(dyn_override, dict):
            dyn.update(dyn_override)

        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=float(dyn["lateral_friction"]),
            spinningFriction=float(dyn["spinning_friction"]),
            rollingFriction=float(dyn["rolling_friction"]),
            restitution=float(dyn["restitution"]),
            frictionAnchor=int(dyn["friction_anchor"]),
        )

    def _add_flat(self, w: Dict[str, Any]) -> int:
        # plane.urdf lies on z=0
        return p.loadURDF("plane.urdf")

    # -----------------------------
    # geometry builders (correct placement)
    # -----------------------------
    def _add_straight_pipe(
        self,
        radius: float,
        length: float,
        pos: Tuple[float, float, float],
        euler: Tuple[float, float, float],
        rgba: Tuple[float, float, float, float],
        dyn_override: Optional[Dict[str, Any]] = None,
        fixed: bool = True,
    ) -> int:
        """
        Create a single cylinder pipe. Convention (as used here):
          - pipe axis is LOCAL +Z of the body frame
          - euler=(0,0,0) => vertical pipe (axis along world Z)
        """
        quat = _quat_from_euler(euler)
        mass = 0.0 if fixed else 1.0

        col_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=float(radius),
            height=float(length),
        )
        vis_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=float(radius),
            length=float(length),
            rgbaColor=[float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])],
        )

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[float(pos[0]), float(pos[1]), float(pos[2])],
            baseOrientation=quat,
        )

        self._apply_body_dynamics(body_id, dyn_override)

        # Track pipe geometry separately so pipe-only contact metrics are not polluted
        # by ground plane or other world objects.
        try:
            self.pipe_bodies.append(int(body_id))
        except Exception:
            pass

        return body_id

    def _add_tapered_pipe(
        self,
        radius_start: float,
        radius_end: float,
        length: float,
        pos: Tuple[float, float, float],
        euler: Tuple[float, float, float],
        n_segments: int,
        rgba: Tuple[float, float, float, float],
        dyn_override: Optional[Dict[str, Any]] = None,
        fixed: bool = True,
    ) -> List[int]:
        """
        Taper approximated by stacking short cylinders along LOCAL Z.
        Placement uses: seg_pos = base_pos + R @ offset_local
        """
        n_segments = max(2, int(n_segments))
        seg_len = float(length) / float(n_segments)

        quat = _quat_from_euler(euler)
        R = _rotmat_from_quat(quat)
        base_pos = (float(pos[0]), float(pos[1]), float(pos[2]))

        seg_ids: List[int] = []
        for i in range(n_segments):
            s = float(i) / float(n_segments - 1)
            r_i = float(radius_start) + (float(radius_end) - float(radius_start)) * s

            # segment center along LOCAL Z, centered around assembly pos
            z_local = -0.5 * float(length) + (i + 0.5) * seg_len
            off = _matvec(R, (0.0, 0.0, z_local))
            seg_pos = (base_pos[0] + off[0], base_pos[1] + off[1], base_pos[2] + off[2])

            seg_id = self._add_straight_pipe(
                radius=r_i,
                length=seg_len,
                pos=seg_pos,
                euler=euler,
                rgba=rgba,
                dyn_override=dyn_override,
                fixed=fixed,
            )
            seg_ids.append(seg_id)

        return seg_ids

    def _add_varying_pipe_3section(
        self,
        lower_radius: float,
        upper_radius: float,
        L_lower: float,
        L_transition: float,
        L_upper: float,
        pos: Tuple[float, float, float],
        euler: Tuple[float, float, float],
        n_transition_segments: int,
        rgba_lower: Tuple[float, float, float, float],
        rgba_transition: Tuple[float, float, float, float],
        rgba_upper: Tuple[float, float, float, float],
        dyn_override: Optional[Dict[str, Any]] = None,
        fixed: bool = True,
    ) -> List[int]:
        """
        3-section pipe: lower -> transition(taper) -> upper.
        The entire assembly is centered at 'pos' and oriented by 'euler' (LOCAL Z axis).
        """
        quat = _quat_from_euler(euler)
        R = _rotmat_from_quat(quat)
        base_pos = (float(pos[0]), float(pos[1]), float(pos[2]))

        L_total = float(L_lower) + float(L_transition) + float(L_upper)

        # local Z centers relative to assembly center
        z_center_lower = -0.5 * L_total + 0.5 * float(L_lower)
        z_center_trans = -0.5 * L_total + float(L_lower) + 0.5 * float(L_transition)
        z_center_upper = -0.5 * L_total + float(L_lower) + float(L_transition) + 0.5 * float(L_upper)

        def world_center(z_local: float) -> Tuple[float, float, float]:
            off = _matvec(R, (0.0, 0.0, z_local))
            return (base_pos[0] + off[0], base_pos[1] + off[1], base_pos[2] + off[2])

        pos_lower = world_center(z_center_lower)
        pos_trans = world_center(z_center_trans)
        pos_upper = world_center(z_center_upper)

        ids: List[int] = []
        ids.append(
            self._add_straight_pipe(
                radius=float(lower_radius),
                length=float(L_lower),
                pos=pos_lower,
                euler=euler,
                rgba=rgba_lower,
                dyn_override=dyn_override,
                fixed=fixed,
            )
        )

        ids.extend(
            self._add_tapered_pipe(
                radius_start=float(lower_radius),
                radius_end=float(upper_radius),
                length=float(L_transition),
                pos=pos_trans,
                euler=euler,
                n_segments=int(n_transition_segments),
                rgba=rgba_transition,
                dyn_override=dyn_override,
                fixed=fixed,
            )
        )

        ids.append(
            self._add_straight_pipe(
                radius=float(upper_radius),
                length=float(L_upper),
                pos=pos_upper,
                euler=euler,
                rgba=rgba_upper,
                dyn_override=dyn_override,
                fixed=fixed,
            )
        )

        return ids

    def _add_straight_pipe_from_cfg(self, w: Dict[str, Any]) -> int:
        pos = w.get("pos", [0.0, 0.0, 0.6])
        rpy = w.get("rpy", [0.0, 0.0, 0.0])

        radius = float(w["radius"])
        length = float(w["length"])
        rgba = w.get("rgba", [0.7, 0.7, 0.7, 1.0])

        dyn = w.get("pipe_dynamics", None)

        return self._add_straight_pipe(
            radius=radius,
            length=length,
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            euler=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            rgba=(float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])),
            dyn_override=dyn,
            fixed=True,
        )

    def _add_junction_from_cfg(self, w: Dict[str, Any]) -> List[int]:
        """
        Build a junction:
          - main pipe: oriented by w.rpy, centered at w.pos, axis along main LOCAL Z
          - branch pipe: horizontal, with yaw (about main LOCAL Z) = branch_angle [rad]
          - intersection point: at the TOP of main pipe (in main LOCAL Z)
        """
        pos = w.get("pos", [0.0, 0.0, 0.6])
        rpy = w.get("rpy", [0.0, 0.0, 0.0])
        main = w["main_pipe"]
        branch = w["branch_pipe"]

        r_main = float(main["radius"])
        L_main = float(main["length"])

        r_branch = float(branch["radius"])
        L_branch = float(branch["length"])
        ang = float(branch.get("branch_angle", math.pi / 2.0))

        rgba_main = w.get("rgba_main", [0.6, 0.6, 0.6, 1.0])
        rgba_branch = w.get("rgba_branch", [0.8, 0.5, 0.3, 1.0])

        dyn = w.get("pipe_dynamics", None)

        main_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        main_euler = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
        q_main = _quat_from_euler(main_euler)
        R_main = _rotmat_from_quat(q_main)

        # main cylinder
        main_id = self._add_straight_pipe(
            radius=r_main,
            length=L_main,
            pos=main_pos,
            euler=main_euler,
            rgba=(float(rgba_main[0]), float(rgba_main[1]), float(rgba_main[2]), float(rgba_main[3])),
            dyn_override=dyn,
            fixed=True,
        )

        # branch intersects at top of main: local offset +0.5*L_main along main LOCAL Z
        off_top = _matvec(R_main, (0.0, 0.0, 0.5 * L_main))
        junction_pos = (main_pos[0] + off_top[0], main_pos[1] + off_top[1], main_pos[2] + off_top[2])

        # Build branch orientation:
        # Start from local Z, first rotate about local Y by +90deg (Z -> X),
        # then yaw about local Z by 'ang' to sweep in the XY plane.
        q_y90 = _quat_from_euler((0.0, math.pi / 2.0, 0.0))
        q_yaw = _quat_from_euler((0.0, 0.0, ang))
        q_local = _qmul(q_yaw, q_y90)     # q_local = yaw * y90
        q_branch = _qmul(q_main, q_local) # q_branch = main * local

        # Convert q_branch back to euler is unnecessary; we can set orientation directly
        # by creating the body with quaternion. To reuse _add_straight_pipe(), we’ll create here.
        col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(r_branch), height=float(L_branch))
        vis_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=float(r_branch),
            length=float(L_branch),
            rgbaColor=[float(rgba_branch[0]), float(rgba_branch[1]), float(rgba_branch[2]), float(rgba_branch[3])],
        )
        branch_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[float(junction_pos[0]), float(junction_pos[1]), float(junction_pos[2])],
            baseOrientation=q_branch,
        )
        self._apply_body_dynamics(branch_id, dyn)

        # Track branch as pipe geometry for contact metrics.
        try:
            self.pipe_bodies.append(int(branch_id))
        except Exception:
            pass

        return [main_id, branch_id]

    def _add_varying_pipe_3section_from_cfg(self, w: Dict[str, Any]) -> List[int]:
        pos = w.get("pos", [0.0, 0.0, 0.6])
        rpy = w.get("rpy", [0.0, 0.0, 0.0])

        dyn = w.get("pipe_dynamics", None)

        return self._add_varying_pipe_3section(
            lower_radius=float(w["lower_radius"]),
            upper_radius=float(w["upper_radius"]),
            L_lower=float(w["L_lower"]),
            L_transition=float(w["L_transition"]),
            L_upper=float(w["L_upper"]),
            pos=(float(pos[0]), float(pos[1]), float(pos[2])),
            euler=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            n_transition_segments=int(w.get("n_transition_segments", 24)),
            rgba_lower=tuple(float(x) for x in w.get("rgba_lower", [0.2, 0.6, 0.9, 1.0])),          # type: ignore
            rgba_transition=tuple(float(x) for x in w.get("rgba_transition", [0.9, 0.7, 0.2, 1.0])), # type: ignore
            rgba_upper=tuple(float(x) for x in w.get("rgba_upper", [0.2, 0.9, 0.4, 1.0])),           # type: ignore
            dyn_override=dyn,
            fixed=True,
        )

    # -----------------------------
    # robot
    # -----------------------------
    def _repo_root(self) -> Path:
        # snake_pipe/snake_bullet/src/snake_bullet/sim_env.py -> parents[3] is snake_pipe/
        return Path(__file__).resolve().parents[3]

    def _resolve_urdf_path(self) -> str:
        if not self.urdf_path_cfg:
            raise ValueError("robot.urdf_path is empty.")
        pth = Path(self.urdf_path_cfg)
        if pth.is_absolute():
            return str(pth)
        return str((self._repo_root() / pth).resolve())

    def _load_robot(self) -> None:
        urdf_path = self._resolve_urdf_path()
        quat = p.getQuaternionFromEuler([float(self.base_rpy[0]), float(self.base_rpy[1]), float(self.base_rpy[2])])

        flags = 0
        if self.self_collision:
            flags |= p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT

        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=self.base_pos,
            baseOrientation=quat,
            useFixedBase=int(self.fixed_base),
            flags=flags,
        )
        self._compute_joint_indices()

    def _compute_joint_indices(self) -> None:
        assert self.robot_id is not None

        revolute = []
        revolute_names = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            if joint_type == p.JOINT_REVOLUTE:
                revolute.append(j)
                revolute_names.append(joint_name)

        if self.motor_idx_rule == "explicit":
            self.joint_idx = [int(i) for i in self.explicit_motor_indices]
            self.joint_names = [p.getJointInfo(self.robot_id, j)[1].decode("utf-8") for j in self.joint_idx]
        elif self.motor_idx_rule == "auto_revolute":
            self.joint_idx = revolute
            self.joint_names = revolute_names
        else:
            raise ValueError(f"Unknown motor_idx_rule: {self.motor_idx_rule}")
        
        if self.reverse_joint_order:
            self.joint_idx = list(reversed(self.joint_idx))
            self.joint_names = list(reversed(self.joint_names))

    def _apply_robot_dynamics(self, dyn: Dict[str, Any]) -> None:
        assert self.robot_id is not None

        p.changeDynamics(
            self.robot_id,
            -1,
            lateralFriction=float(dyn["lateral_friction"]),
            spinningFriction=float(dyn["spinning_friction"]),
            rollingFriction=float(dyn["rolling_friction"]),
            restitution=float(dyn["restitution"]),
            frictionAnchor=int(dyn["friction_anchor"]),
        )

        for j in range(p.getNumJoints(self.robot_id)):
            p.changeDynamics(
                self.robot_id,
                j,
                lateralFriction=float(dyn["lateral_friction"]),
                spinningFriction=float(dyn["spinning_friction"]),
                rollingFriction=float(dyn["rolling_friction"]),
                restitution=float(dyn["restitution"]),
                frictionAnchor=int(dyn["friction_anchor"]),
            )

    def _enable_joint_torque_sensors(self, enable: bool) -> None:
        if self.robot_id is None:
            return
        for j in range(p.getNumJoints(self.robot_id)):
            try:
                p.enableJointForceTorqueSensor(self.robot_id, j, enableSensor=int(enable))
            except Exception:
                pass

    # -----------------------------
    # contact -> generalized torque
    # -----------------------------
    def _get_dof_joint_indices(self) -> List[int]:
        """Return Bullet joint indices that correspond to 1-DoF joints.

        PyBullet's calculateJacobian() expects q/dq/ddq arrays sized to the
        number of *DoFs* (revolute/prismatic), not p.getNumJoints().
        We cache the result after the robot is loaded.
        """
        assert self.robot_id is not None
        if self._dof_joints is not None:
            return self._dof_joints

        dof = []
        for j in range(p.getNumJoints(self.robot_id)):
            try:
                jtype = int(p.getJointInfo(self.robot_id, j)[2])
            except Exception:
                continue
            if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                dof.append(j)
        self._dof_joints = dof
        return dof

    def _compute_contact_joint_torques(self) -> List[float]:
        """Compute generalized joint torques induced by external contact forces.

        We map contact point forces (normal + friction) into generalized torques via:
            tau_contact = J_lin(p)^T * F_contact

        Returned torques are in Bullet joint-index order (0..numJoints-1).
        """
        assert self.robot_id is not None

        num_joints = p.getNumJoints(self.robot_id)
        tau_full = [0.0] * num_joints
        if num_joints == 0:
            return tau_full

        dof_joints = self._get_dof_joint_indices()
        dof_n = len(dof_joints)
        if dof_n == 0:
            return tau_full

        # Build q/dq in DoF-order for calculateJacobian.
        jstates = p.getJointStates(self.robot_id, dof_joints)
        q_dof = [float(s[0]) for s in jstates]
        dq_dof = [float(s[1]) for s in jstates]
        ddq_dof = [0.0] * dof_n

        # Gather contact points involving the robot (either as bodyA or bodyB).
        contacts = list(p.getContactPoints(bodyA=self.robot_id)) + list(p.getContactPoints(bodyB=self.robot_id))
        if not contacts:
            return tau_full

        # Deduplicate contacts (Bullet can report duplicates when queried both ways).
        uniq = []
        seen = set()
        for c in contacts:
            try:
                key = (
                    int(c[1]), int(c[2]), int(c[3]), int(c[4]),
                    round(float(c[5][0]), 6), round(float(c[5][1]), 6), round(float(c[5][2]), 6),
                    round(float(c[6][0]), 6), round(float(c[6][1]), 6), round(float(c[6][2]), 6),
                )
            except Exception:
                key = id(c)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)

        for c in uniq:
            body_a = int(c[1])
            body_b = int(c[2])

            # Exclude self-collisions.
            if body_a == self.robot_id and body_b == self.robot_id:
                continue

            # Choose the link on the robot and the corresponding contact point.
            if body_a == self.robot_id:
                link_idx = int(c[3])
                p_world = c[5]
                normal_on_b = c[7]
                normal_f = float(c[9])

                lat_f1 = float(c[10]) if len(c) > 10 else 0.0
                lat_dir1 = c[11] if len(c) > 11 else (0.0, 0.0, 0.0)
                lat_f2 = float(c[12]) if len(c) > 12 else 0.0
                lat_dir2 = c[13] if len(c) > 13 else (0.0, 0.0, 0.0)

                # contactNormalOnB points from B->A; force on A is opposite.
                fx = -(normal_f * float(normal_on_b[0]) + lat_f1 * float(lat_dir1[0]) + lat_f2 * float(lat_dir2[0]))
                fy = -(normal_f * float(normal_on_b[1]) + lat_f1 * float(lat_dir1[1]) + lat_f2 * float(lat_dir2[1]))
                fz = -(normal_f * float(normal_on_b[2]) + lat_f1 * float(lat_dir1[2]) + lat_f2 * float(lat_dir2[2]))
                F = (fx, fy, fz)

            elif body_b == self.robot_id:
                link_idx = int(c[4])
                p_world = c[6]
                normal_on_b = c[7]
                normal_f = float(c[9])

                lat_f1 = float(c[10]) if len(c) > 10 else 0.0
                lat_dir1 = c[11] if len(c) > 11 else (0.0, 0.0, 0.0)
                lat_f2 = float(c[12]) if len(c) > 12 else 0.0
                lat_dir2 = c[13] if len(c) > 13 else (0.0, 0.0, 0.0)

                # contactNormalOnB points from B->A; force on B is along the normal.
                fx = (normal_f * float(normal_on_b[0]) + lat_f1 * float(lat_dir1[0]) + lat_f2 * float(lat_dir2[0]))
                fy = (normal_f * float(normal_on_b[1]) + lat_f1 * float(lat_dir1[1]) + lat_f2 * float(lat_dir2[1]))
                fz = (normal_f * float(normal_on_b[2]) + lat_f1 * float(lat_dir1[2]) + lat_f2 * float(lat_dir2[2]))
                F = (fx, fy, fz)
            else:
                continue

            # NOTE: Bullet may report base-link contacts with link_idx == -1 (e.g., when fixed sublinks are merged).
            # We try to process them too; if Jacobian isn't available for the base, the Jacobian call will fail
            # and we will skip that contact.
            # Convert world contact point into link-COM local frame for calculateJacobian.
            try:
                # Get COM pose (world) for the contacted link.
                if link_idx == -1:
                    com_pos_w, com_orn_w = p.getBasePositionAndOrientation(self.robot_id)
                else:
                    ls = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=1)
                    # Indices 0,1 are world COM position/orientation.
                    com_pos_w = ls[0]
                    com_orn_w = ls[1]

                R = p.getMatrixFromQuaternion(com_orn_w)  # row-major 3x3
                px = float(p_world[0]) - float(com_pos_w[0])
                py = float(p_world[1]) - float(com_pos_w[1])
                pz = float(p_world[2]) - float(com_pos_w[2])
                # local = R^T (p_world - com_pos)
                local_pos = [
                    R[0] * px + R[3] * py + R[6] * pz,
                    R[1] * px + R[4] * py + R[7] * pz,
                    R[2] * px + R[5] * py + R[8] * pz,
                ]

                j_lin = None

                # First attempt: DoF-sized arrays (preferred for MultiBody).
                try:
                    j_lin, _j_ang = p.calculateJacobian(
                        self.robot_id,
                        link_idx,
                        local_pos,
                        q_dof,
                        dq_dof,
                        ddq_dof,
                    )
                except Exception:
                    # Fallback: some Bullet builds expect full joint arrays.
                    js_all = p.getJointStates(self.robot_id, list(range(num_joints)))
                    q_all = [float(s[0]) for s in js_all]
                    dq_all = [float(s[1]) for s in js_all]
                    ddq_all = [0.0] * num_joints
                    j_lin, _j_ang = p.calculateJacobian(
                        self.robot_id,
                        link_idx,
                        local_pos,
                        q_all,
                        dq_all,
                        ddq_all,
                    )

                if j_lin is None:
                    continue

                ncol = len(j_lin[0]) if (len(j_lin) > 0) else 0
                if ncol == dof_n:
                    # Accumulate tau[dof] += J_lin^T * F
                    for k in range(dof_n):
                        jidx = dof_joints[k]
                        tau_full[jidx] += (
                            float(j_lin[0][k]) * float(F[0])
                            + float(j_lin[1][k]) * float(F[1])
                            + float(j_lin[2][k]) * float(F[2])
                        )
                elif ncol == num_joints:
                    # Direct joint-indexed Jacobian.
                    for j in range(num_joints):
                        tau_full[j] += (
                            float(j_lin[0][j]) * float(F[0])
                            + float(j_lin[1][j]) * float(F[1])
                            + float(j_lin[2][j]) * float(F[2])
                        )
                else:
                    # Unexpected Jacobian shape; skip.
                    continue
            except Exception:
                # Jacobian can fail for some link types; skip.
                continue

        return tau_full


    def _compute_reaction_joint_torques(self) -> List[float]:
        """Compute per-joint torque about each joint axis from Bullet joint reaction wrenches.

        Bullet's getJointState/getJointStates returns a 6D reaction wrench for each joint:
            (Fx, Fy, Fz, Mx, My, Mz)

        For a 1-DoF revolute joint, the physically meaningful scalar load torque is the
        component of the reaction moment along the joint axis.

        Returned torques are in Bullet joint-index order (0..numJoints-1). Non-DoF joints return 0.
        """
        assert self.robot_id is not None
        num_joints = p.getNumJoints(self.robot_id)
        tau_full = [0.0] * num_joints
        if num_joints == 0:
            return tau_full

        # Cache axes in joint frame for speed
        if not hasattr(self, "_joint_axis_cache"):
            axis_cache = {}
            for j in range(num_joints):
                try:
                    info = p.getJointInfo(self.robot_id, j)
                    jtype = int(info[2])
                    if jtype != p.JOINT_REVOLUTE and jtype != p.JOINT_PRISMATIC:
                        continue
                    axis = info[13]
                    ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
                    n = (ax*ax + ay*ay + az*az) ** 0.5
                    if n > 1e-9:
                        ax, ay, az = ax/n, ay/n, az/n
                    axis_cache[j] = (ax, ay, az)
                except Exception:
                    continue
            self._joint_axis_cache = axis_cache

        # Read states for all joints
        jstates = p.getJointStates(self.robot_id, list(range(num_joints)))
        for j, s in enumerate(jstates):
            if j not in self._joint_axis_cache:
                continue
            rxn = s[2]
            if rxn is None or len(rxn) < 6:
                continue
            mx, my, mz = float(rxn[3]), float(rxn[4]), float(rxn[5])
            ax, ay, az = self._joint_axis_cache[j]
            tau_full[j] = mx*ax + my*ay + mz*az

        return tau_full

    def _ensure_tau_filter_state(self, n: int) -> None:
        """Initialize per-joint filter state once we know joint vector length."""
        if self._tau_filter_inited and self._tau_lpf_state is not None and self._tau_med_bufs is not None:
            if len(self._tau_lpf_state) == n and len(self._tau_med_bufs) == n:
                # If window size changed in YAML between runs, refresh buffers.
                if self._tau_med_bufs[0].maxlen == max(1, int(self.tau_filter_window)):
                    return

        win = max(1, int(self.tau_filter_window))
        self._tau_lpf_state = [0.0] * n
        self._tau_med_bufs = [deque(maxlen=win) for _ in range(n)]
        self._tau_filter_inited = True

    def _apply_tau_filter(self, tau: List[float]) -> List[float]:
        """Filter RobotState.tau (in controlled joint order)."""
        ttype = (self.tau_filter_type or "none").lower()
        if ttype in ["none", "off", "false", "0", "disable", "disabled", ""]:
            # Still apply deadband if requested.
            if self.tau_deadband > 0.0:
                db = float(self.tau_deadband)
                return [0.0 if abs(x) < db else x for x in tau]
            return tau

        n = len(tau)
        if n == 0:
            return tau
        self._ensure_tau_filter_state(n)
        assert self._tau_lpf_state is not None
        assert self._tau_med_bufs is not None

        # LPF coefficient
        alpha = 0.0
        if self.tau_filter_tau and self.tau_filter_tau > 1e-9:
            try:
                alpha = math.exp(-float(self.dt) / float(self.tau_filter_tau))
            except Exception:
                alpha = 0.0

        out = [0.0] * n
        for i, x in enumerate(tau):
            xi = float(x)

            # Median stage (optional)
            if ttype in ["median", "median_lpf", "med", "med_lpf"]:
                buf = self._tau_med_bufs[i]
                buf.append(xi)
                if len(buf) == 0:
                    xm = xi
                else:
                    # nan-safe median
                    xm = float(np.nanmedian(np.array(list(buf), dtype=float)))
            else:
                xm = xi

            # LPF stage (optional)
            if ttype in ["lpf", "median_lpf", "lowpass", "low_pass", "med_lpf"] and alpha > 0.0:
                prev = float(self._tau_lpf_state[i])
                yf = alpha * prev + (1.0 - alpha) * xm
                self._tau_lpf_state[i] = yf
            elif ttype in ["lpf", "median_lpf", "lowpass", "low_pass", "med_lpf"]:
                # tau_filter_tau invalid -> behave like passthrough
                yf = xm
                self._tau_lpf_state[i] = yf
            else:
                yf = xm

            # Deadband (after filtering)
            if self.tau_deadband and self.tau_deadband > 0.0 and abs(yf) < float(self.tau_deadband):
                yf = 0.0

            out[i] = float(yf)

        return out

    # state / command / step
    # -----------------------------
    def get_state(self) -> RobotState:
        assert self.robot_id is not None

        # Read all joint states once (Bullet joint index order).
        num_joints = p.getNumJoints(self.robot_id)
        if num_joints > 0:
            jstates = p.getJointStates(self.robot_id, list(range(num_joints)))
            q_full = [float(s[0]) for s in jstates]
            dq_full = [float(s[1]) for s in jstates]
            tau_applied_full = [float(s[3]) for s in jstates]
        else:
            q_full, dq_full, tau_applied_full = [], [], []

        # Choose torque signal.
        if self.torque_reading in ["reaction", "rxn", "joint_reaction"]:
            tau_full = self._compute_reaction_joint_torques()
        elif self.torque_reading in ["contact", "contacts", "contact_jacobian"]:
            tau_full = self._compute_contact_joint_torques()
            # If Jacobian mapping yields (near) zero while we are clearly in contact, fall back to
            # reaction-based torques so the signal reflects environment interaction.
            try:
                contacts = list(p.getContactPoints(bodyA=self.robot_id)) + list(p.getContactPoints(bodyB=self.robot_id))
                if contacts and max(abs(x) for x in tau_full) < 1e-10:
                    tau_full = self._compute_reaction_joint_torques()
            except Exception:
                pass
        else:
            tau_full = tau_applied_full

        # Return in our controlled joint order.
        q = [q_full[j] for j in self.joint_idx] if q_full else []
        dq = [dq_full[j] for j in self.joint_idx] if dq_full else []
        tau = [tau_full[j] for j in self.joint_idx] if tau_full else []

        # Optional smoothing of the returned torque signal.
        tau = self._apply_tau_filter(tau)

        # Optional sign correction.
        if self._joint_sign is not None:
            if self.apply_tau_sign_map:
                tau = self._apply_sign(tau, self._joint_sign)
            if self.apply_state_sign_map:
                q = self._apply_sign(q, self._joint_sign)
                dq = self._apply_sign(dq, self._joint_sign)

        # Simulated IMU-friendly pose/velocity from Bullet
        base_pos = None
        base_quat = None
        base_lin_vel = None
        base_ang_vel = None
        try:
            bp, bq = p.getBasePositionAndOrientation(self.robot_id)
            blv, bav = p.getBaseVelocity(self.robot_id)
            base_pos = (float(bp[0]), float(bp[1]), float(bp[2]))
            base_quat = (float(bq[0]), float(bq[1]), float(bq[2]), float(bq[3]))
            base_lin_vel = (float(blv[0]), float(blv[1]), float(blv[2]))
            base_ang_vel = (float(bav[0]), float(bav[1]), float(bav[2]))
        except Exception:
            pass

        return RobotState(t=self._t_sim, dt=self.dt, q=q, dq=dq, tau=tau, joint_names=self.joint_names, base_pos=base_pos, base_quat=base_quat, base_lin_vel=base_lin_vel, base_ang_vel=base_ang_vel)

    def get_pipe_contact_metrics(self) -> Dict[str, float]:
        """Return pipe-only contact summary metrics.

        Pipe-only means: uses self.pipe_bodies so ground-plane contacts don't pollute logs.

        Returns dict (merge into dbg):
          - pipe_num_contacts : count of contact points
          - pipe_sum_Fn       : sum of normal forces over contact points [N]
          - pipe_sum_Ft       : sum of lateral friction magnitudes [N]
          - pipe_max_Fn       : max normal force among contact points [N]
        """
        if self.robot_id is None or not getattr(self, 'pipe_bodies', None):
            return {
                'pipe_num_contacts': 0.0,
                'pipe_sum_Fn': 0.0,
                'pipe_sum_Ft': 0.0,
                'pipe_max_Fn': 0.0,
            }

        n_c = 0
        sum_fn = 0.0
        sum_ft = 0.0
        max_fn = 0.0

        # Bullet contact tuple indices (14-tuple):
        # normalForce idx=9
        # lateralFriction1 idx=10
        # lateralFriction2 idx=12
        for pid in list(self.pipe_bodies):
            try:
                cps = p.getContactPoints(bodyA=self.robot_id, bodyB=int(pid))
            except Exception:
                cps = []
            for cp in cps:
                n_c += 1
                try:
                    fn = float(cp[9])
                except Exception:
                    fn = 0.0
                if fn > max_fn:
                    max_fn = fn
                sum_fn += fn

                try:
                    f1 = float(cp[10])
                except Exception:
                    f1 = 0.0
                try:
                    f2 = float(cp[12])
                except Exception:
                    f2 = 0.0
                sum_ft += abs(f1) + abs(f2)

        return {
            'pipe_num_contacts': float(n_c),
            'pipe_sum_Fn': float(sum_fn),
            'pipe_sum_Ft': float(sum_ft),
            'pipe_max_Fn': float(max_fn),
        }

    def apply_command(self, cmd: JointCommand) -> None:
        assert self.robot_id is not None
        mode = (cmd.mode or self.control_mode).lower()

        if mode == "position":
            if cmd.position is None:
                return
            forces = cmd.effort if cmd.effort is not None else [self.max_torque] * len(self.joint_idx)

            target_pos = list(cmd.position)
            if self._joint_sign is not None and self.apply_cmd_sign_map:
                target_pos = self._apply_sign(target_pos, self._joint_sign)

            kwargs = {
                "targetPositions": target_pos,
                "forces": list(forces),
            }
            if self.position_gain is not None:
                kwargs["positionGains"] = [self.position_gain] * len(self.joint_idx)
            if self.velocity_gain is not None:
                kwargs["velocityGains"] = [self.velocity_gain] * len(self.joint_idx)

            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_idx,
                p.POSITION_CONTROL,
                **kwargs,
            )

        elif mode == "velocity":
            if cmd.velocity is None:
                return
            forces = cmd.effort if cmd.effort is not None else [self.max_torque] * len(self.joint_idx)

            target_vel = list(cmd.velocity)
            if self._joint_sign is not None and self.apply_cmd_sign_map:
                target_vel = self._apply_sign(target_vel, self._joint_sign)

            kwargs = {
                "targetVelocities": target_vel,
                "forces": list(forces),
            }
            # VELOCITY_CONTROL also uses velocityGains.
            if self.velocity_gain is not None:
                kwargs["velocityGains"] = [self.velocity_gain] * len(self.joint_idx)

            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_idx,
                p.VELOCITY_CONTROL,
                **kwargs,
            )

        elif mode == "torque":
            if cmd.effort is None:
                return
            eff = list(cmd.effort)
            if self._joint_sign is not None and self.apply_cmd_sign_map:
                eff = self._apply_sign(eff, self._joint_sign)

            p.setJointMotorControlArray(
                self.robot_id,
                self.joint_idx,
                p.TORQUE_CONTROL,
                forces=eff,
            )
        else:
            raise ValueError(f"Unknown command mode: '{mode}'")

    def _update_camera(self) -> None:
        if not self.tracking_cam or self.robot_id is None:
            return

        if self.cam_target == "base":
            pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        else:
            if len(self.joint_idx) == 0:
                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            else:
                mid = max(0, int(len(self.joint_idx) / 2) - 1)
                link_idx = self.joint_idx[mid]
                link_state = p.getLinkState(self.robot_id, link_idx)
                pos = link_state[0]

        p.resetDebugVisualizerCamera(self.cam_distance, self.cam_yaw, self.cam_pitch, pos)

    def step(self) -> None:
        for _ in range(max(1, self.substeps)):
            p.stepSimulation()
            self._t_sim += self.dt

        self._update_camera()

        if self.realtime:
            now = time.time()
            elapsed = now - self._t_wall_last
            target = self.dt * max(1, self.substeps)
            if elapsed < target:
                time.sleep(target - elapsed)
            self._t_wall_last = time.time()
