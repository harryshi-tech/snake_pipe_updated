#!/usr/bin/env python3
"""
check_tau_resultant.py

Step-0 diagnostic: detect "small resultant torque" caused by sign cancellation.

Given a run folder (e.g., log/data/<run_id>/), this script loads the tau CSV and computes:
  T_sum = sum_j tau_j                    (signed sum; can cancel to ~0)
  T_abs = sum_j |tau_j|                  (no cancellation)
  T_rms = sqrt(mean_j tau_j^2)           (no cancellation)

It then prints summary stats and the cancellation ratio:
  ratio = |T_sum| / T_abs   (near 0 => strong cancellation)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas", file=sys.stderr)
    raise


def pick_latest_run(log_root: Path) -> Path:
    if not log_root.exists():
        raise FileNotFoundError(f"log_root does not exist: {log_root}")
    subdirs = [p for p in log_root.iterdir() if p.is_dir()]
    # latest modified directory
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for d in subdirs:
        if find_tau_csv(d) is not None:
            return d
    raise FileNotFoundError(f"No run dirs under {log_root} contain a tau CSV.")


def find_tau_csv(run_dir: Path) -> Path | None:
    # common patterns in your project
    candidates = []
    candidates += list(run_dir.glob("*__tau*.csv"))
    candidates += list(run_dir.glob("*tau*.csv"))
    # prefer the most "tau-like" file
    if not candidates:
        return None

    def score(p: Path) -> tuple:
        name = p.name.lower()
        return (
            0 if "__tau" in name else 1,
            0 if "contact" in name else 1,
            len(name),
        )

    candidates.sort(key=score)
    return candidates[0]


def choose_time_col(df) -> str | None:
    lc = {c.lower(): c for c in df.columns}
    for key in ["t", "time", "sim_time", "sec", "timestamp"]:
        if key in lc:
            return lc[key]
    return None


def choose_tau_cols(df, prefix: str | None) -> list[str]:
    # numeric columns only
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    tcol = choose_time_col(df)
    if tcol in num_cols:
        num_cols.remove(tcol)

    if not num_cols:
        return []

    # If user gave a prefix, use it.
    if prefix:
        cols = [c for c in num_cols if c.lower().startswith(prefix.lower()) or prefix.lower() in c.lower()]
        return cols

    # Auto-pick: prefer contact torque if present
    contact_cols = [c for c in num_cols if "contact" in c.lower()]
    if len(contact_cols) >= 2:
        return contact_cols

    # Otherwise use any columns that look like joint torques
    tau_like = [c for c in num_cols if "tau" in c.lower() or "torque" in c.lower()]
    if len(tau_like) >= 2:
        return tau_like

    # Fallback: assume all numeric (excluding time) are "torque-ish"
    return num_cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="Path to a single run dir under log/data/")
    ap.add_argument("--log_root", type=str, default="log/data", help="Root folder containing run dirs")
    ap.add_argument("--prefix", type=str, default="", help="Optional: filter tau columns by name/prefix (e.g. 'tau_contact')")
    ap.add_argument("--head", type=int, default=8, help="How many columns to preview")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    run_dir = Path(args.run_dir) if args.run_dir else pick_latest_run(log_root)

    tau_path = find_tau_csv(run_dir)
    if tau_path is None:
        print(f"ERROR: No tau CSV found in {run_dir}", file=sys.stderr)
        return 2

    df = pd.read_csv(tau_path)
    if df.empty:
        print(f"ERROR: tau CSV is empty: {tau_path}", file=sys.stderr)
        return 2

    time_col = choose_time_col(df)
    tau_cols = choose_tau_cols(df, args.prefix if args.prefix else None)

    if len(tau_cols) < 2:
        print("ERROR: Could not find >=2 torque columns to aggregate.", file=sys.stderr)
        print("  Tip: pass --prefix tau_contact or --prefix tau", file=sys.stderr)
        print(f"  Columns found: {list(df.columns)[:50]}", file=sys.stderr)
        return 2

    # time vector
    if time_col is None:
        t = np.arange(len(df), dtype=float)
        time_col = "(index)"
    else:
        t = df[time_col].to_numpy(dtype=float)

    tau = df[tau_cols].to_numpy(dtype=float)  # shape (T, J)

    # Step-0 metrics
    T_sum = np.sum(tau, axis=1)
    T_abs = np.sum(np.abs(tau), axis=1)
    T_rms = np.sqrt(np.mean(tau * tau, axis=1))

    # cancellation ratio (avoid div by 0)
    eps = 1e-12
    ratio = np.abs(T_sum) / (T_abs + eps)

    # summarize
    print("\n=== Step-0: Resultant Torque Cancellation Check ===")
    print(f"Run dir : {run_dir}")
    print(f"Tau CSV : {tau_path.name}")
    print(f"Samples : {len(df)}")
    print(f"Time col: {time_col}")
    print(f"Joints  : {tau.shape[1]}")
    print(f"Tau cols (first {min(args.head, len(tau_cols))}/{len(tau_cols)}): {tau_cols[:args.head]}")
    print("\n--- Key stats (median / max) ---")
    print(f"|T_sum| : {np.median(np.abs(T_sum)):.6g} / {np.max(np.abs(T_sum)):.6g}")
    print(f"T_abs  : {np.median(T_abs):.6g} / {np.max(T_abs):.6g}")
    print(f"T_rms  : {np.median(T_rms):.6g} / {np.max(T_rms):.6g}")
    print("\n--- Cancellation ratio r = |T_sum|/T_abs ---")
    print(f"r (median/max): {np.median(ratio):.6g} / {np.max(ratio):.6g}")
    print("Interpretation: r near 0 => strong sign cancellation (signed sum looks 'small' even if joints are loaded).")

    # find times with big RMS but tiny sum (classic cancellation)
    big = T_rms > np.percentile(T_rms, 90)
    tiny = ratio < 0.05
    idx = np.where(big & tiny)[0]
    if idx.size > 0:
        show = idx[:10]
        print("\n--- Examples where RMS is high but signed sum cancels (first up to 10) ---")
        for k in show:
            print(f"t={t[k]:.4f}:  T_rms={T_rms[k]:.6g},  |T_sum|={abs(T_sum[k]):.6g},  T_abs={T_abs[k]:.6g},  r={ratio[k]:.6g}")
    else:
        print("\nNo strong-cancellation events found with the default thresholds (90th percentile RMS and r<0.05).")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
