#!/usr/bin/env python3
"""
plot_csv_logs.py

Standalone plotting utility for snake_pipe CSV logs.

Features
--------
- LaTeX-style labels (mathtext) in titles + legends.
- Dual-axis plot: left = torque, right = deviation (theta0 - thetad).
- Easy global styling (linewidth, etc.).
- Multi-joint plots for ONE variable (e.g., tau or q_nom), where **each joint has a different color**.
- (Optional) Highlight gait-execution regions using events.csv (shaded background spans) on EVERY plot.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# User convenience defaults
# -----------------------------
RUN_DIR = ""  # e.g. "log/data/test2_20260110_153012"

# Optional: manually restrict the time interval to plot.
# Set either/both to None to disable.
PLOT_T_MIN = None  # seconds
PLOT_T_MAX = None  # seconds


# -----------------------------
# Styling (easy to edit)
# -----------------------------
# Base style per VARIABLE (we will NOT force the color in multi-joint plots,
# because colors should differ by joint).
LINE_STYLE: Dict[str, Dict] = {
    "tau":       dict(linestyle="-",  linewidth=2.0),
    "deviation": dict(linestyle="--", linewidth=2.0),

    "q_nom":     dict(linestyle="-",  linewidth=2.0),
    "q_cmd":     dict(linestyle="-",  linewidth=2.0),
    "q_meas":    dict(linestyle="-",  linewidth=2.0),

    "A_scale":   dict(linestyle="-",  linewidth=1.8),
}

# If you want fixed colors for joint indices, edit this list.
# If None, matplotlib's default color cycle is used automatically.
JOINT_COLORS: Optional[List[str]] = None
# Example:
# JOINT_COLORS = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]


# -----------------------------
# LaTeX labels (applies everywhere: title + legend)
# -----------------------------
# In this logging scheme:
#   q_cmd  -> theta_d
#   q_nom  -> theta_0
#   q_meas -> theta_meas
LATEX_LABEL: Dict[str, str] = {
    "q_cmd":  r"$\theta_d$",
    "q_nom":  r"$\theta_0$",
    "q_meas": r"$\theta_{\mathrm{meas}}$",

    "tau": r"$\tau$",
    "deviation": r"$\theta_0-\theta_d$",

    "A_scale": r"$A/A_0$",
}


def _legend_name(name: str) -> str:
    return LATEX_LABEL.get(name, name)


def _style_for(name: str, override: Optional[Dict] = None) -> Dict:
    base = dict(LINE_STYLE.get(name, {}))
    if override:
        base.update(override)
    return base


# -----------------------------
# Events highlight (gait regions)
# -----------------------------
HIGHLIGHT_GAIT_REGIONS = True
GAIT_SPAN_ALPHA = 0.10
GAIT_TEXT_ALPHA = 0.65
GAIT_TEXT_Y = 0.98  # axes fraction (near top)


def _extract_gait_name_from_row(row_cells: List[str], header: Optional[List[str]]) -> Optional[str]:
    """
    Best-effort parsing to detect gait selection events from events.csv rows.

    Supports common patterns:
      - A cell like "gait=rolling_helix" or "gait: rolling_helix"
      - Columns like event/name/type == "gait" with another column containing the gait string
      - Free-form messages containing "gait" and a token-like name after '=' or ':'

    Returns:
      gait_name (str) or None
    """
    cells = [str(c).strip() for c in row_cells if str(c).strip() != ""]
    if len(cells) == 0:
        return None

    # If header exists, look for obvious "gait" columns
    if header is not None:
        h = [str(x).strip().lower() for x in header]
        # Try columns explicitly named "gait"
        if "gait" in h:
            gi = h.index("gait")
            if gi < len(row_cells):
                g = str(row_cells[gi]).strip()
                return g if g else None

        # Try columns like event/name/type and value/message
        event_keys = {"event", "name", "type", "key"}
        value_keys = {"value", "val", "data", "msg", "message", "detail", "info"}

        event_idx = None
        value_idx = None
        for k in event_keys:
            if k in h:
                event_idx = h.index(k)
                break
        for k in value_keys:
            if k in h:
                value_idx = h.index(k)
                break

        if event_idx is not None:
            ev = str(row_cells[event_idx]).strip().lower()
            if "gait" in ev:
                # Prefer value column if present
                if value_idx is not None and value_idx < len(row_cells):
                    g = str(row_cells[value_idx]).strip()
                    return g if g else None
                # Otherwise search other cells
                for c in cells:
                    if c.lower() not in ev and "gait" not in c.lower():
                        # heuristic: first other token-like cell
                        return c

    # Join free-form text
    s = " | ".join(cells)
    sl = s.lower()

    # common patterns gait=NAME or gait: NAME
    for token in ["gait=", "gait:", "gait ", "set_gait=", "set_gait:", "set gait", "setgait"]:
        if token in sl:
            # find the first occurrence in the ORIGINAL string
            idx = sl.find(token)
            tail = s[idx + len(token):].strip()

            # remove leading separators
            while len(tail) > 0 and tail[0] in [" ", "=", ":", ",", ";"]:
                tail = tail[1:].strip()

            # take first "word-like" token (stop at separator)
            for sep in ["|", ",", ";", " "]:
                if sep in tail:
                    tail = tail.split(sep, 1)[0].strip()
                    break

            # sanity: must be non-empty and not just "gait"
            if tail and tail.lower() not in ["gait", "set_gait", "setgait"]:
                return tail

    return None


def _load_events_gaits(run_dir: Path) -> List[Tuple[float, str]]:
    """
    Read events.csv and return a list of (time, gait_name) sorted by time.

    If events.csv doesn't exist or no gait events found, returns [].
    """
    path = run_dir / "events.csv"
    if not path.exists():
        return []

    try:
        with path.open("r", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return []

    if len(rows) == 0:
        return []

    # Detect header
    first = rows[0]
    header: Optional[List[str]] = None
    has_header = any(not _is_float(str(x).strip()) for x in first)
    data_rows = rows

    if has_header:
        header = [str(x).strip() for x in first]
        data_rows = rows[1:]

    # Find time column if header exists
    time_idx = 0
    if header is not None:
        hl = [h.strip().lower() for h in header]
        for k in ["t", "time", "sec", "seconds", "snake_time"]:
            if k in hl:
                time_idx = hl.index(k)
                break

    out: List[Tuple[float, str]] = []
    for r in data_rows:
        if len(r) == 0:
            continue
        if time_idx >= len(r):
            continue

        t_str = str(r[time_idx]).strip()
        if not _is_float(t_str):
            continue
        t = float(t_str)

        # Build row cells excluding the time column for parsing
        row_cells = [r[j] for j in range(len(r)) if j != time_idx]
        gait = _extract_gait_name_from_row(row_cells, header=(None if header is None else [h for i, h in enumerate(header) if i != time_idx]))
        if gait is not None:
            out.append((t, gait))

    # Sort and de-duplicate adjacent identical gait names
    out.sort(key=lambda x: x[0])
    dedup: List[Tuple[float, str]] = []
    for t, g in out:
        if len(dedup) == 0 or dedup[-1][1] != g:
            dedup.append((t, g))
    return dedup


def _add_gait_spans(ax: plt.Axes, gait_events: List[Tuple[float, str]], tmin: float, tmax: float) -> None:
    """
    Shade background regions corresponding to each gait interval.

    gait_events: list of (time, gait_name)
    Regions: [t_i, t_{i+1}) with gait_i; last extends to tmax.
    """
    if not HIGHLIGHT_GAIT_REGIONS:
        return
    if len(gait_events) == 0:
        return

    # Clip events to [tmin, tmax] with one event before tmin if available
    # so we can know what gait is active at tmin.
    events = gait_events[:]
    # Find last event before tmin
    prev = None
    for t, g in events:
        if t <= tmin:
            prev = (t, g)
        else:
            break

    clipped: List[Tuple[float, str]] = []
    if prev is not None:
        clipped.append(prev)
    clipped.extend([(t, g) for (t, g) in events if tmin < t < tmax])

    if len(clipped) == 0:
        return

    # Build segments
    segs: List[Tuple[float, float, str]] = []
    for i in range(len(clipped)):
        t0, g0 = clipped[i]
        t1 = clipped[i + 1][0] if i + 1 < len(clipped) else tmax
        a = max(t0, tmin)
        b = min(t1, tmax)
        if b > a:
            segs.append((a, b, g0))

    if len(segs) == 0:
        return

    # Assign colors per gait name using matplotlib cycle (light alpha)
    # Use a stable mapping based on first appearance order.
    gait_to_color: Dict[str, str] = {}
    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6"])
    ci = 0
    for _, _, g in segs:
        if g not in gait_to_color:
            gait_to_color[g] = cycle_colors[ci % len(cycle_colors)]
            ci += 1

    # Draw spans + text label (at top of axes)
    for a, b, g in segs:
        ax.axvspan(a, b, color=gait_to_color[g], alpha=GAIT_SPAN_ALPHA, linewidth=0)

        # Put label roughly centered, but only if wide enough
        if (b - a) > 0.5:  # seconds threshold to avoid clutter
            xm = 0.5 * (a + b)
            ax.text(
                xm, GAIT_TEXT_Y, g,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=9,
                alpha=GAIT_TEXT_ALPHA,
            )


# -----------------------------
# CSV loading
# -----------------------------
@dataclass
class LoadedCSV:
    t: np.ndarray
    y: np.ndarray
    header: List[str]
    path: Path


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_numeric_csv(path: Path) -> LoadedCSV:
    if not path.exists():
        raise FileNotFoundError(str(path))

    with path.open("r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"Empty CSV: {path}")

    first = rows[0]
    has_header = any((not _is_float(str(x).strip())) for x in first)

    if has_header:
        header = [str(x).strip() for x in first]
        data_rows = rows[1:]
    else:
        ncols = len(first)
        header = ["t"] + [f"c{i}" for i in range(1, ncols)]
        data_rows = rows

    if len(header) < 2:
        raise ValueError(f"CSV needs at least 2 columns (t + data): {path}")

    t_list: List[float] = []
    y_list: List[List[float]] = []

    for r in data_rows:
        if len(r) == 0:
            continue
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        if len(r) > len(header):
            r = r[: len(header)]

        if all((str(x).strip() == "" for x in r)):
            continue

        t_str = str(r[0]).strip()
        if t_str == "":
            continue
        try:
            t_val = float(t_str)
        except Exception:
            continue

        vals: List[float] = []
        for j in range(1, len(header)):
            s = str(r[j]).strip()
            if s == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(s))
                except Exception:
                    vals.append(np.nan)

        t_list.append(t_val)
        y_list.append(vals)

    if len(t_list) == 0:
        raise ValueError(f"No numeric rows parsed from: {path}")

    t = np.array(t_list, dtype=float)
    y = np.array(y_list, dtype=float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Crop time window
    if PLOT_T_MIN is not None or PLOT_T_MAX is not None:
        tmin = -np.inf if PLOT_T_MIN is None else float(PLOT_T_MIN)
        tmax = np.inf if PLOT_T_MAX is None else float(PLOT_T_MAX)
        mask = (t >= tmin) & (t <= tmax)
        if np.any(mask):
            t = t[mask]
            y = y[mask, :]

    return LoadedCSV(t=t, y=y, header=header, path=path)


def list_csv_variables(run_dir: Path) -> List[str]:
    files = sorted(run_dir.glob("*.csv"))
    return [f.stem for f in files]


def _try_load(run_dir: Path, var: str) -> Optional[LoadedCSV]:
    path = run_dir / f"{var}.csv"
    if not path.exists():
        return None
    try:
        return load_numeric_csv(path)
    except Exception as e:
        print(f"[plot_csv_logs] WARNING: failed to read {path.name}: {e}")
        return None


def _interp_1d(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    t_src = np.asarray(t_src).astype(float)
    y_src = np.asarray(y_src).astype(float)
    t_dst = np.asarray(t_dst).astype(float)

    m = np.isfinite(t_src) & np.isfinite(y_src)
    if np.sum(m) < 2:
        return np.full_like(t_dst, np.nan, dtype=float)

    ts = t_src[m]
    ys = y_src[m]
    order = np.argsort(ts)
    ts = ts[order]
    ys = ys[order]
    return np.interp(t_dst, ts, ys, left=ys[0], right=ys[-1])


# -----------------------------
# Plot configuration
# -----------------------------
PLOTS = [
    dict(
        type="overlay_vars_one_joint",
        title=r"Module 3 - Nominal, Desired, and Measured Joint Angle",
        vars=["q_nom", "q_cmd", "q_meas"],
        col=3,
        ylabel=r"angle [rad]",
    ),

        dict(
        type="overlay_vars_one_joint",
        title=r"Module 4 - Nominal, Desired, and Measured Joint Angle",
        vars=["q_nom", "q_cmd", "q_meas"],
        col=4,
        ylabel=r"angle [rad]",
    ),

        dict(
        type="overlay_vars_one_joint",
        title=r"Module 5 - Nominal, Desired, and Measured Joint Angle",
        vars=["q_nom", "q_cmd", "q_meas"],
        col=5,
        ylabel=r"angle [rad]",
    ),

        dict(
        type="overlay_vars_one_joint",
        title=r"Module 6 - Nominal, Desired, and Measured Joint Angle",
        vars=["q_nom", "q_cmd", "q_meas"],
        col=6,
        ylabel=r"angle [rad]",
    ),

    dict(
        type="tau_and_deviation_one_joint",
        title=r"Module 9 - Torque (left) and Deviation $(\theta_0-\theta_d)$ (right)",
        col=0,
        tau_var="tau",
        nom_var="q_nom",
        des_var="q_cmd",
        ylabel_left=r"torque $\tau$ [N$\cdot$m]",
        ylabel_right=r"deviation $(\theta_0-\theta_d)$ [rad]",
    ),

    # Multi-joint, ONE variable, different colors per joint
    dict(
        type="multi_joint",
        title=r"Torque $\tau$ for selected joints (each joint different color)",
        var="tau",
        cols=[10,11,12,13],  # 0-based joint columns
        ylabel=r"torque $\tau$ [N$\cdot$m]",
        legend_mode="joint",    # "joint" or "header" or "col"
        # Optional per-plot style overrides applied to all lines:
        # style={"linewidth": 2.5, "linestyle": "-"},
        # Optional per-column overrides:
        # style_by_col={0: {"alpha": 0.9}, 15: {"alpha": 0.6}},
    ),

    # Example: same multi-joint plot but for q_nom
    # dict(
    #     type="multi_joint",
    #     title=r"Nominal angle $\theta_0$ for selected joints (each joint different color)",
    #     var="q_nom",
    #     cols=[0, 1, 3, 8, 15],
    #     ylabel=r"angle [rad]",
    #     legend_mode="joint",
    # ),
]


# -----------------------------
# Plotting functions
# -----------------------------
def _maybe_add_spans(ax: plt.Axes, run_dir: Path, tmin: float, tmax: float) -> None:
    if not HIGHLIGHT_GAIT_REGIONS:
        return
    gait_events = _load_events_gaits(run_dir)
    _add_gait_spans(ax, gait_events, tmin=tmin, tmax=tmax)


def _plot_overlay_one_joint(run_dir: Path, spec: Dict) -> None:
    vars_ = [str(v) for v in spec.get("vars", [])]
    col = int(spec.get("col", 0))
    title = str(spec.get("title", f"overlay col {col}"))
    ylabel = str(spec.get("ylabel", "value"))

    style_overrides: Dict[str, Dict] = spec.get("style", {}) or {}

    fig, ax = plt.subplots()
    any_plotted = False
    tmin_plot = None
    tmax_plot = None

    for v in vars_:
        dat = _try_load(run_dir, v)
        if dat is None:
            continue
        if col < 0 or col >= dat.y.shape[1]:
            continue

        ax.plot(
            dat.t,
            dat.y[:, col],
            label=_legend_name(v),
            **_style_for(v, override=style_overrides.get(v)),
        )
        any_plotted = True

        tmin_plot = float(np.min(dat.t)) if tmin_plot is None else min(tmin_plot, float(np.min(dat.t)))
        tmax_plot = float(np.max(dat.t)) if tmax_plot is None else max(tmax_plot, float(np.max(dat.t)))

    if not any_plotted:
        print(f"[plot_csv_logs] skip: overlay_vars_one_joint: nothing plotted for col={col}")
        plt.close(fig)
        return

    # Gait spans
    if tmin_plot is not None and tmax_plot is not None:
        _maybe_add_spans(ax, run_dir, tmin_plot, tmax_plot)

    ax.set_title(title)
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def _plot_tau_and_deviation_one_joint(run_dir: Path, spec: Dict) -> None:
    col = int(spec.get("col", 0))
    title = str(spec.get("title", "Torque + Deviation"))

    tau_var = str(spec.get("tau_var", "tau"))
    nom_var = str(spec.get("nom_var", "q_nom"))
    des_var = str(spec.get("des_var", "q_cmd"))

    ylabel_left = str(spec.get("ylabel_left", r"torque $\tau$"))
    ylabel_right = str(spec.get("ylabel_right", r"deviation"))

    style_overrides: Dict[str, Dict] = spec.get("style", {}) or {}

    dat_tau = _try_load(run_dir, tau_var)
    dat_nom = _try_load(run_dir, nom_var)
    dat_des = _try_load(run_dir, des_var)

    if dat_tau is None or dat_nom is None or dat_des is None:
        print("[plot_csv_logs] skip: tau_and_deviation_one_joint: missing required CSV(s)")
        return

    if (
        col < 0
        or col >= dat_tau.y.shape[1]
        or col >= dat_nom.y.shape[1]
        or col >= dat_des.y.shape[1]
    ):
        print(f"[plot_csv_logs] skip: tau_and_deviation_one_joint: col={col} out of range")
        return

    # Use tau timebase; interpolate nom/des if needed
    t = dat_tau.t
    tau = dat_tau.y[:, col]
    q_nom = _interp_1d(dat_nom.t, dat_nom.y[:, col], t)
    q_des = _interp_1d(dat_des.t, dat_des.y[:, col], t)
    deviation = q_nom - q_des

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(
        t,
        tau,
        label=_legend_name(tau_var),
        **_style_for(tau_var, override=style_overrides.get(tau_var)),
    )
    ax1.set_ylabel(ylabel_left)

    ax2.plot(
        t,
        deviation,
        label=_legend_name("deviation"),
        **_style_for("deviation", override=style_overrides.get("deviation")),
    )
    ax2.set_ylabel(ylabel_right)

    # Gait spans (draw on ax1)
    _maybe_add_spans(ax1, run_dir, float(np.min(t)), float(np.max(t)))

    ax1.set_title(title)
    ax1.set_xlabel(r"time $t$ [s]")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")


def _multi_joint_label(var: str, col: int, mode: str) -> str:
    m = str(mode).lower().strip()
    if m == "col":
        return f"col{col}"
    if m == "header":
        # we don’t have header here; handled in plot function if desired
        return f"col{col}"
    # default: joint index (LaTeX)
    return rf"$j_{{{col}}}$"


def _plot_multi_joint(run_dir: Path, spec: Dict) -> None:
    """
    Plot ONE variable across multiple joint columns on the same axes.

    - Colors differ by joint (NOT by variable).
    - Style (linewidth/linestyle) is shared unless overridden per column.

    Spec keys:
      var: str (e.g. "tau" or "q_nom")
      cols: list[int]
      legend_mode: "joint" | "col" | "header"
      style: dict applied to all lines (e.g. {"linewidth": 2, "linestyle": "-"})
      style_by_col: dict[int]->dict applied to specific joint columns
    """
    var = str(spec.get("var", "tau"))
    cols = [int(c) for c in spec.get("cols", [])]
    title = str(spec.get("title", var))
    ylabel = str(spec.get("ylabel", var))
    legend_mode = str(spec.get("legend_mode", "joint"))

    dat = _try_load(run_dir, var)
    if dat is None:
        print(f"[plot_csv_logs] skip: {var}.csv not found or unreadable")
        return

    style_all: Dict = spec.get("style", {}) or {}
    style_by_col: Dict[int, Dict] = spec.get("style_by_col", {}) or {}

    fig, ax = plt.subplots()

    # Choose colors per plotted joint
    if JOINT_COLORS is None:
        # Let matplotlib cycle colors automatically by NOT specifying "color" unless user forces it.
        colors = [None for _ in cols]
    else:
        colors = [JOINT_COLORS[i % len(JOINT_COLORS)] for i in range(len(cols))]

    any_plotted = False
    for i, c in enumerate(cols):
        if c < 0 or c >= dat.y.shape[1]:
            continue

        # label
        if legend_mode.lower().strip() == "header":
            lab = str(dat.header[c + 1]) if (c + 1) < len(dat.header) else f"col{c}"
        elif legend_mode.lower().strip() == "col":
            lab = f"col{c}"
        else:
            lab = rf"$j_{{{c}}}$"

        # build style: base(var) + style_all + style_by_col
        override = {}
        override.update(style_all)
        override.update(style_by_col.get(c, {}))

        # enforce different colors by joint
        if colors[i] is not None:
            override["color"] = colors[i]
        else:
            # ensure we do NOT accidentally fix color via LINE_STYLE (we don't set color there)
            override.pop("color", None)

        ax.plot(dat.t, dat.y[:, c], label=lab, **_style_for(var, override=override))
        any_plotted = True

    if not any_plotted:
        print(f"[plot_csv_logs] skip: multi_joint: nothing plotted for var={var}")
        plt.close(fig)
        return

    # Gait spans
    _maybe_add_spans(ax, run_dir, float(np.min(dat.t)), float(np.max(dat.t)))

    ax.set_title(title)
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def plot_run(run_dir: Path, plot_specs: Sequence[Dict], list_only: bool = False) -> None:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    csv_files = list(run_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise RuntimeError(f"No CSV files found in: {run_dir}")

    if list_only:
        print(f"[plot_csv_logs] CSV variables in {run_dir}:")
        for v in list_csv_variables(run_dir):
            print("  -", v)
        return

    for spec in plot_specs:
        t = str(spec.get("type", "")).lower().strip()
        if t == "overlay_vars_one_joint":
            _plot_overlay_one_joint(run_dir, spec)
        elif t == "tau_and_deviation_one_joint":
            _plot_tau_and_deviation_one_joint(run_dir, spec)
        elif t == "multi_joint":
            _plot_multi_joint(run_dir, spec)
        else:
            print(f"[plot_csv_logs] WARNING: unknown plot type: {t}")

    plt.show()


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        type=str,
        default=RUN_DIR,
        help="Path to a single run folder that contains *.csv (e.g., log/data/test2_YYYYMMDD_HHMMSS)",
    )
    ap.add_argument("--list", action="store_true", help="List available CSV variables and exit.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir is None or str(run_dir).strip() == "":
        raise RuntimeError(
            "Please provide --run_dir or set RUN_DIR at top of this file.\n"
            "Example: python log/plot_csv_logs.py --run_dir log/data/test2_20260110_153012"
        )

    plot_run(run_dir, PLOTS, list_only=bool(args.list))


if __name__ == "__main__":
    main()