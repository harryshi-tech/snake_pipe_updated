from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_stamp() -> str:
    # YYYYMMDD_HHMMSS
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _to_1d(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _select_joint_indices(cfg: Dict[str, Any], n: int) -> List[int]:
    jcfg = cfg.get("joints", {}) if isinstance(cfg.get("joints", {}), dict) else {}
    mode = str(jcfg.get("mode", "all")).lower()

    if mode == "all":
        return list(range(n))

    if mode == "range":
        start = int(jcfg.get("start", 0))
        stop = int(jcfg.get("stop", n))
        start = max(0, min(n, start))
        stop = max(0, min(n, stop))
        return list(range(start, stop))

    if mode == "list":
        out: List[int] = []
        for v in jcfg.get("list", []):
            try:
                i = int(v)
                if 0 <= i < n:
                    out.append(i)
            except Exception:
                pass
        return out if out else list(range(n))

    return list(range(n))


@dataclass
class CSVLoggerFiles:
    # One CSV per joint-vector signal (e.g., q_meas.csv)
    joint_files: Dict[str, Any]
    # One CSV per scalar signal (e.g., speed_multiplier.csv)
    scalar_files: Dict[str, Any]
    # One CSV per base/IMU-like signal (e.g., base_pos.csv)
    base_files: Dict[str, Any]
    # Optional event log (events.csv)
    event_file: Optional[Any]


class CSVLogger:
    """Simple CSV logger that records selected signals during simulation.

    Designed to be enabled/disabled from sim_params.yaml.
    """

    DEFAULT_JOINT_SIGNALS = [
        "q_meas",
        "q_nom",
        "q_cmd",
        "tau",
        "tau_f",
        "A",
        "A_scale",
    ]

    DEFAULT_SCALAR_SIGNALS = [
        "snake_time",
        "speed_multiplier",
        "tightness",
        "pole_direction",
        "wt_direction",
        "sbc_scale_mean",
        "sbc_scale_min",
        "sbc_scale_max",
    ]

    DEFAULT_BASE_SIGNALS = [
        "base_pos",
        "base_quat",
        "base_lin_vel",
        "base_ang_vel",
    ]

    def __init__(
        self,
        cfg: Dict[str, Any],
        repo_root: Path,
        joint_names: Sequence[str],
    ) -> None:
        self.cfg = cfg or {}
        self.enable = bool(self.cfg.get("enable", False))
        self.repo_root = Path(repo_root)

        self.n_joints = int(len(joint_names))
        self.joint_names = list(joint_names)
        self.joint_sel = _select_joint_indices(self.cfg, self.n_joints)

        self.decimation = max(1, int(self.cfg.get("decimation", 1)))
        self._k = 0

        # output directory
        out_dir = str(self.cfg.get("out_dir", "log/data"))
        self.out_dir = (self.repo_root / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir)
        _ensure_dir(self.out_dir)

        run_name = str(self.cfg.get("run_name", "latest"))
        add_ts = bool(self.cfg.get("add_timestamp", True))
        self.run_id = f"{run_name}_{_now_stamp()}" if add_ts else run_name


        # Each run writes into its own folder inside out_dir
        self.trial_dir = self.out_dir / self.run_id
        _ensure_dir(self.trial_dir)
        # which signals
        sig_cfg = self.cfg.get("signals", {}) if isinstance(self.cfg.get("signals", {}), dict) else {}
        self.joint_signals = [str(s) for s in sig_cfg.get("joint", self.DEFAULT_JOINT_SIGNALS)]
        self.scalar_signals = [str(s) for s in sig_cfg.get("scalar", self.DEFAULT_SCALAR_SIGNALS)]

        # Ensure pipe contact metrics are logged even if the user specified
        # an explicit scalar list in YAML.
        # Disable by setting logging.signals.auto_add_contact_metrics: false
        auto_contact = bool(sig_cfg.get("auto_add_contact_metrics", True))
        if auto_contact:
            for k in ["pipe_num_contacts", "pipe_sum_Fn", "pipe_sum_Ft", "pipe_max_Fn"]:
                if k not in self.scalar_signals:
                    self.scalar_signals.append(k)
        self.base_signals = [str(s) for s in sig_cfg.get("base", self.DEFAULT_BASE_SIGNALS)]
        self.enable_events = bool(sig_cfg.get("events", True))

        self.flush_every = max(1, int(self.cfg.get("flush_every", 240)))

        self.files = self._open_files()

        # Write a small meta file for convenience
        self._write_meta()

    # -----------------
    # file management
    # -----------------
    def _open_files(self) -> CSVLoggerFiles:
        # Each selected signal gets its own CSV inside trial_dir.
        joint_files: Dict[str, Any] = {}
        for key in self.joint_signals:
            fp = self.trial_dir / f"{key}.csv"
            f = fp.open("w", newline="")
            w = csv.writer(f)
            cols = ["t"] + [self.joint_names[i] for i in self.joint_sel]
            w.writerow(cols)
            joint_files[key] = (f, w)

        scalar_files: Dict[str, Any] = {}
        for key in self.scalar_signals:
            fp = self.trial_dir / f"{key}.csv"
            f = fp.open("w", newline="")
            w = csv.writer(f)
            w.writerow(["t", key])
            scalar_files[key] = (f, w)

        base_files: Dict[str, Any] = {}
        for s in self.base_signals:
            fp = self.trial_dir / f"{s}.csv"
            f = fp.open("w", newline="")
            w = csv.writer(f)

            if s == "base_pos":
                w.writerow(["t", "base_x", "base_y", "base_z"])
            elif s == "base_quat":
                w.writerow(["t", "quat_x", "quat_y", "quat_z", "quat_w"])
            elif s == "base_lin_vel":
                w.writerow(["t", "v_x", "v_y", "v_z"])
            elif s == "base_ang_vel":
                w.writerow(["t", "w_x", "w_y", "w_z"])
            else:
                w.writerow(["t", s])

            base_files[s] = (f, w)

        event_file = None
        if self.enable_events:
            fp = self.trial_dir / "events.csv"
            f = fp.open("w", newline="")
            w = csv.writer(f)
            w.writerow(["t", "gait", "teleop_cmd"])  # strings are fine in CSV
            event_file = (f, w)

        return CSVLoggerFiles(
            joint_files=joint_files,
            scalar_files=scalar_files,
            base_files=base_files,
            event_file=event_file,
        )

    def _write_meta(self) -> None:
        try:
            meta_path = self.trial_dir / "meta.txt"
            lines = []
            lines.append(f"run_id: {self.run_id}")
            lines.append(f"out_dir: {self.out_dir}")
            lines.append(f"decimation: {self.decimation}")
            lines.append(f"flush_every: {self.flush_every}")
            lines.append(f"joint_sel: {self.joint_sel}")
            lines.append(f"joint_names: {self.joint_names}")
            lines.append(f"joint_signals: {self.joint_signals}")
            lines.append(f"scalar_signals: {self.scalar_signals}")
            lines.append(f"base_signals: {self.base_signals}")
            lines.append(f"events: {self.enable_events}")
            meta_path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if not self.enable:
            return

        def _close_pair(pair: Any) -> None:
            if pair is None:
                return
            f, _ = pair
            try:
                f.flush()
                f.close()
            except Exception:
                pass

        # joint/scalar/base files
        for _, pair in self.files.joint_files.items():
            _close_pair(pair)
        for _, pair in self.files.scalar_files.items():
            _close_pair(pair)
        for _, pair in self.files.base_files.items():
            _close_pair(pair)

        # event file
        _close_pair(self.files.event_file)

    # -----------------
    # recording
    # -----------------
    def record(self, state: Any, cmd: Any, dbg: Optional[Dict[str, Any]] = None, teleop_cmd: Optional[str] = None) -> None:
        if not self.enable:
            return

        self._k += 1
        if (self._k % self.decimation) != 0:
            return

        dbg = dbg or {}
        t = float(getattr(state, "t", 0.0))

        # helpers to fetch signals
        def get_joint_vec(key: str) -> Optional[np.ndarray]:
            if key == "q_meas":
                v = getattr(state, "q", None)
            elif key == "dq":
                v = getattr(state, "dq", None)
            elif key == "tau":
                v = getattr(state, "tau", None)
            elif key == "q_cmd":
                v = dbg.get("q_cmd", None)
                if v is None:
                    v = getattr(cmd, "position", None)
            elif key == "q_nom":
                v = dbg.get("q_nom", None)
            elif key == "q_sbc":
                v = dbg.get("q_sbc", None)
            else:
                v = dbg.get(key, None)

            if v is None:
                return None
            arr = _to_1d(v)
            if arr.size < self.n_joints:
                # pad
                arr = np.pad(arr, (0, self.n_joints - arr.size), constant_values=np.nan)
            return arr

        # joint vectors -> one file per signal
        for key, (f, w) in self.files.joint_files.items():
            arr = get_joint_vec(key)
            if arr is None:
                continue
            row = [t] + [float(arr[i]) for i in self.joint_sel]
            w.writerow(row)

        # scalars -> one file per signal
        for key, (f, w) in self.files.scalar_files.items():
            v = dbg.get(key, None)
            if v is None:
                # allow some fallbacks
                if key == "dt":
                    v = getattr(state, "dt", None)
                elif key == "snake_time":
                    v = dbg.get("snake_time", None)
            try:
                vv = float(v) if v is not None else np.nan
            except Exception:
                vv = np.nan
            w.writerow([t, vv])

        # base/IMU-like -> one file per signal
        for key, (f, w) in self.files.base_files.items():
            if key == "base_pos":
                v = getattr(state, "base_pos", None)
                if v is None:
                    w.writerow([t, np.nan, np.nan, np.nan])
                else:
                    vv = list(v)
                    w.writerow([t, float(vv[0]), float(vv[1]), float(vv[2])])
            elif key == "base_quat":
                v = getattr(state, "base_quat", None)
                if v is None:
                    w.writerow([t, np.nan, np.nan, np.nan, np.nan])
                else:
                    vv = list(v)
                    w.writerow([t, float(vv[0]), float(vv[1]), float(vv[2]), float(vv[3])])
            elif key == "base_lin_vel":
                v = getattr(state, "base_lin_vel", None)
                if v is None:
                    w.writerow([t, np.nan, np.nan, np.nan])
                else:
                    vv = list(v)
                    w.writerow([t, float(vv[0]), float(vv[1]), float(vv[2])])
            elif key == "base_ang_vel":
                v = getattr(state, "base_ang_vel", None)
                if v is None:
                    w.writerow([t, np.nan, np.nan, np.nan])
                else:
                    vv = list(v)
                    w.writerow([t, float(vv[0]), float(vv[1]), float(vv[2])])
            else:
                # unknown: store NaN
                w.writerow([t, np.nan])

        # events

        if self.files.event_file is not None:
            f, w = self.files.event_file
            gait = str(dbg.get("gait_name", dbg.get("gait", "")))
            w.writerow([t, gait, "" if teleop_cmd is None else str(teleop_cmd)])

        # flushing
        if (self._k % self.flush_every) == 0:
            try:
                for _, (ff, _) in self.files.joint_files.items():
                    ff.flush()
                for _, (ff, _) in self.files.scalar_files.items():
                    ff.flush()
                for _, (ff, _) in self.files.base_files.items():
                    ff.flush()
                if self.files.event_file is not None:
                    self.files.event_file[0].flush()
            except Exception:
                pass
