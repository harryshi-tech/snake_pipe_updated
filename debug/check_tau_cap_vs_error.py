#!/usr/bin/env python3
"""
check_tau_cap_vs_error.py

Per-joint diagnostic:
- Estimate effective torque cap from log (near-max percentile)
- Measure how often torque is near cap (saturation fraction)
- Measure how often error is "large"
- Check if, during large error, torque is near cap (evidence of maxForce limiting)

Run:
  python debug/check_tau_cap_vs_error.py --run_dir log/data/position
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def find_csv(run_dir: Path, key: str) -> Path:
    cands = list(run_dir.glob(f"*{key}*.csv"))
    if not cands:
        raise FileNotFoundError(f"Can't find *{key}*.csv in {run_dir}")
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]


def time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ["t", "time", "sim_time", "sec", "timestamp"]:
            return c
    return None


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--err_thr", default=0.8, type=float, help="Large error threshold (rad), per joint")
    ap.add_argument("--sat_frac", default=0.95, type=float, help="Near-cap threshold as fraction of cap")
    ap.add_argument("--cap_q", default=99.9, type=float, help="Percentile to estimate torque cap from logs")
    ap.add_argument("--top", default=10, type=int)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tau_path = find_csv(run_dir, "tau")
    qc_path = find_csv(run_dir, "q_cmd")
    qm_path = find_csv(run_dir, "q_meas")

    df_tau = pd.read_csv(tau_path)
    df_qc  = pd.read_csv(qc_path)
    df_qm  = pd.read_csv(qm_path)

    t_tau = time_col(df_tau)
    t_qc  = time_col(df_qc)
    t_qm  = time_col(df_qm)

    tau_cols = [c for c in df_tau.columns if c != t_tau and np.issubdtype(df_tau[c].dtype, np.number)]
    qc_cols  = [c for c in df_qc.columns  if c != t_qc  and np.issubdtype(df_qc[c].dtype, np.number)]
    qm_cols  = [c for c in df_qm.columns  if c != t_qm  and np.issubdtype(df_qm[c].dtype, np.number)]

    common = [c for c in qc_cols if c in qm_cols]
    if not common:
        raise RuntimeError("No matching joint columns between q_cmd and q_meas.")

    # Map tau columns to joint names by substring match (works for SAxxx__MoJo style)
    tau_map = {}
    for i, tc in enumerate(tau_cols):
        for jc in common:
            if jc in tc or tc in jc:
                tau_map[jc] = i
                break

    mapped = [j for j in common if j in tau_map]
    if len(mapped) != len(common):
        missing = [j for j in common if j not in tau_map][:10]
        print(f"Warning: couldn't map {len(common)-len(mapped)} joints to tau cols. Example missing: {missing}")

    # Time base: tau
    t = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float) if t_qc else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float) if t_qm else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)
    abs_tau = np.abs(tau)

    qc = df_qc[common].to_numpy(float)
    qm = df_qm[common].to_numpy(float)

    # Interpolate q onto tau time
    qc_i = np.vstack([np.interp(t, tc, qc[:, j]) for j in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, j]) for j in range(qm.shape[1])]).T

    e = wrap_to_pi(qc_i - qm_i)
    abs_e = np.abs(e)

    rows = []
    for j_idx, jname in enumerate(common):
        if jname not in tau_map:
            continue
        i_tau = tau_map[jname]

        tau_j = abs_tau[:, i_tau]
        e_j   = abs_e[:, j_idx]

        # Estimate cap from near-max percentile (more robust than max)
        cap = np.percentile(tau_j, args.cap_q)
        sat_thr = args.sat_frac * cap

        sat = tau_j >= sat_thr
        hiE = e_j >= args.err_thr

        # How often saturated overall / during high error
        sat_all = float(np.mean(sat))
        hiE_all = float(np.mean(hiE))
        sat_given_hiE = float(np.mean(sat[hiE])) if np.any(hiE) else np.nan

        rows.append((jname, cap, sat_all, hiE_all, sat_given_hiE,
                     float(np.max(tau_j)), float(np.max(e_j))))

    # Sort by max error first (the suspicious ones)
    rows.sort(key=lambda r: r[6], reverse=True)

    print("\n=== Torque-cap vs error check (per joint) ===")
    print(f"Run dir: {run_dir}")
    print(f"cap estimate percentile: {args.cap_q}")
    print(f"large error threshold: {args.err_thr} rad")
    print(f"near-cap threshold: {args.sat_frac} * cap\n")

    header = f"{'joint':>14s}  {'cap~':>7s}  {'sat%':>7s}  {'hiE%':>7s}  {'sat|hiE%':>9s}  {'max|tau|':>8s}  {'max|e|':>7s}"
    print(header)
    print("-" * len(header))

    for r in rows[: min(args.top, len(rows))]:
        jname, cap, sat_all, hiE_all, sat_hiE, taumax, emax = r
        sat_hiE_s = "nan" if np.isnan(sat_hiE) else f"{sat_hiE*100:8.2f}"
        print(f"{jname:>14s}  {cap:7.3f}  {sat_all*100:6.2f}%  {hiE_all*100:6.2f}%  {sat_hiE_s}%  {taumax:8.3f}  {emax:7.3f}")

    print("\nHow to read this:")
    print("- If cap~ is ~1.0 for most joints and sat|hiE% is high => you are torque-limited (maxForce too low).")
    print("- If hiE% is tiny => those big max errors are rare spikes; torque can look 'small' most of the time.")
    print("- If hiE% is sizable but sat|hiE% is low => not torque-limited; then we check gains / what tau.csv actually logs.\n")


if __name__ == "__main__":
    main()
