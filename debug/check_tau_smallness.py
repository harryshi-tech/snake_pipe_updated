#!/usr/bin/env python3
"""
check_tau_smallness.py

Diagnostics for "why is per-joint torque small?"

It will:
- Load tau CSV in a run dir
- Print per-joint torque stats (median |tau|, p95 |tau|, max |tau|)
- If q_cmd and q_meas exist: compute tracking error and correlate |tau| with |error|
- Optionally check saturation if you pass --tau_limit

Run:
  python debug/check_tau_smallness.py --run_dir log/data/position
  python debug/check_tau_smallness.py --run_dir log/data/position --tau_limit 1.5
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


def find_csv(run_dir: Path, keywords: list[str]) -> Path | None:
    cands = []
    for p in run_dir.glob("*.csv"):
        name = p.name.lower()
        if all(k in name for k in keywords):
            cands.append(p)
    if not cands:
        return None
    # prefer shorter / more direct names
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


def choose_time_col(df: pd.DataFrame) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for key in ["t", "time", "sim_time", "sec", "timestamp"]:
        if key in cols:
            return cols[key]
    return None


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--tau_limit", type=float, default=None, help="Optional torque limit (N*m) to check saturation")
    ap.add_argument("--top", type=int, default=8, help="How many joints to list in 'largest' summaries")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    tau_path = find_csv(run_dir, ["tau"]) or find_csv(run_dir, ["torque"])
    if tau_path is None:
        print(f"ERROR: could not find a tau/torque CSV in {run_dir}", file=sys.stderr)
        return 2

    df_tau = pd.read_csv(tau_path)
    if df_tau.empty:
        print(f"ERROR: tau CSV empty: {tau_path}", file=sys.stderr)
        return 2

    tcol_tau = choose_time_col(df_tau)
    tau_num = numeric_cols(df_tau)
    if tcol_tau in tau_num:
        tau_num.remove(tcol_tau)

    if len(tau_num) < 1:
        print(f"ERROR: no numeric torque columns found in {tau_path.name}", file=sys.stderr)
        return 2

    t_tau = df_tau[tcol_tau].to_numpy(dtype=float) if tcol_tau else np.arange(len(df_tau), dtype=float)
    tau = df_tau[tau_num].to_numpy(dtype=float)  # (T, J)
    abs_tau = np.abs(tau)

    # --- Per-joint torque stats ---
    med = np.median(abs_tau, axis=0)
    p95 = np.percentile(abs_tau, 95, axis=0)
    mx = np.max(abs_tau, axis=0)

    order = np.argsort(mx)[::-1]

    print("\n=== Per-joint torque magnitude stats ===")
    print(f"Run dir : {run_dir}")
    print(f"Tau CSV : {tau_path.name}")
    print(f"Samples : {tau.shape[0]}")
    print(f"Joints  : {tau.shape[1]}")
    if tcol_tau:
        print(f"Time col: {tcol_tau}")
    else:
        print("Time col: (index)")

    print("\nTop joints by max |tau|:")
    for k in order[: min(args.top, len(order))]:
        print(f"  {tau_num[k]:>14s}: median={med[k]:.6g}, p95={p95[k]:.6g}, max={mx[k]:.6g}")

    print("\nOverall (across all joints & time):")
    print(f"  median |tau|: {np.median(abs_tau):.6g}")
    print(f"  p95   |tau|: {np.percentile(abs_tau,95):.6g}")
    print(f"  max   |tau|: {np.max(abs_tau):.6g}")

    # --- Optional: saturation check ---
    if args.tau_limit is not None:
        lim = float(args.tau_limit)
        sat_mask = abs_tau >= 0.95 * lim
        sat_frac = np.mean(sat_mask)
        print("\n=== Saturation check ===")
        print(f"tau_limit = {lim:.6g} N*m (threshold = 0.95*limit)")
        print(f"fraction of samples saturated (all joints pooled): {sat_frac*100:.3f}%")
        sat_per_joint = np.mean(sat_mask, axis=0)
        top_sat = np.argsort(sat_per_joint)[::-1]
        print("Top joints by saturation fraction:")
        for k in top_sat[: min(args.top, len(top_sat))]:
            if sat_per_joint[k] <= 0:
                break
            print(f"  {tau_num[k]:>14s}: {sat_per_joint[k]*100:.3f}%")

    # --- Load q_cmd / q_meas if available ---
    q_cmd_path = find_csv(run_dir, ["q_cmd"])
    q_meas_path = find_csv(run_dir, ["q_meas"])
    if q_cmd_path is None or q_meas_path is None:
        print("\n(No q_cmd/q_meas found in this run dir, skipping error correlation checks.)\n")
        return 0

    df_qc = pd.read_csv(q_cmd_path)
    df_qm = pd.read_csv(q_meas_path)
    if df_qc.empty or df_qm.empty:
        print("\n(q_cmd/q_meas empty, skipping.)\n")
        return 0

    tcol_qc = choose_time_col(df_qc)
    tcol_qm = choose_time_col(df_qm)
    qc_num = numeric_cols(df_qc)
    qm_num = numeric_cols(df_qm)
    if tcol_qc in qc_num:
        qc_num.remove(tcol_qc)
    if tcol_qm in qm_num:
        qm_num.remove(tcol_qm)

    # Align columns by intersection of names
    common = [c for c in qc_num if c in qm_num]
    if not common:
        print("\nFound q_cmd/q_meas but no matching joint columns by name. Skipping correlation.\n")
        return 0

    # Align in time by interpolating q onto tau timebase (simple and robust)
    tq = t_tau
    tc = df_qc[tcol_qc].to_numpy(dtype=float) if tcol_qc else np.arange(len(df_qc), dtype=float)
    tm = df_qm[tcol_qm].to_numpy(dtype=float) if tcol_qm else np.arange(len(df_qm), dtype=float)

    qc = df_qc[common].to_numpy(dtype=float)
    qm = df_qm[common].to_numpy(dtype=float)

    # interpolate each joint
    qc_i = np.vstack([np.interp(tq, tc, qc[:, j]) for j in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(tq, tm, qm[:, j]) for j in range(qm.shape[1])]).T

    err = qc_i - qm_i
    abs_err = np.abs(err)

    # correlate |tau| with |error| for matching joints (by name)
    # Need to map tau columns to common q columns; if names differ, we can’t do 1:1.
    # We'll do a best-effort: match by substring overlap.
    tau_map = {}
    for i, tname in enumerate(tau_num):
        for cname in common:
            if cname in tname or tname in cname:
                tau_map[cname] = i
                break

    mapped = [c for c in common if c in tau_map]
    if len(mapped) < 3:
        print("\nq_cmd/q_meas found, but tau columns don’t match joint names well, so only error stats printed.\n")
        print("Error magnitude overall:")
        print(f"  median |e|: {np.median(abs_err):.6g}")
        print(f"  p95   |e|: {np.percentile(abs_err,95):.6g}")
        print(f"  max   |e|: {np.max(abs_err):.6g}\n")
        return 0

    print("\n=== Torque vs tracking error (using |tau| and |e|) ===")
    print(f"Matched joints: {len(mapped)} / {len(common)}")

    # per joint: median error and correlation
    rows = []
    for j, cname in enumerate(common):
        if cname not in tau_map:
            continue
        ti = tau_map[cname]
        a = abs_tau[:, ti]
        b = abs_err[:, j]
        # corrcoef can NaN if constant
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        rows.append((cname, float(np.median(b)), float(np.percentile(b, 95)), float(np.max(b)), corr, float(np.max(a))))

    # sort by max error
    rows.sort(key=lambda r: r[3], reverse=True)

    print("Top joints by max |error|:")
    for r in rows[: min(args.top, len(rows))]:
        name, emed, e95, emax, corr, taumax = r
        corr_s = "nan" if np.isnan(corr) else f"{corr:.3f}"
        print(f"  {name:>14s}: median|e|={emed:.4g}, p95|e|={e95:.4g}, max|e|={emax:.4g}, corr(|tau|,|e|)={corr_s}, max|tau|={taumax:.4g}")

    print("\nInterpretation tips:")
    print("- If |e| is tiny most of the time, motor torque can legitimately be small (no load / good tracking).")
    print("- If |e| is large but |tau| stays small, your motor may not be applying torque as you think (wrong field, disabled motor, low gains, etc.).")
    if args.tau_limit is None:
        print("- If you expect saturation, rerun with --tau_limit <your_joint_limit> to see if clipping occurs.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
