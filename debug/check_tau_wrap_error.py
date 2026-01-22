#!/usr/bin/env python3
"""
check_tau_wrap_error.py

Checks if "big tracking error but small torque" is caused by angle wrapping / mismatch.

- Loads tau.csv, q_cmd.csv, q_meas.csv from a run_dir
- Aligns by time (interpolation)
- Computes:
    e_raw = q_cmd - q_meas
    e_wrap = wrap_to_pi(e_raw)
- Prints per-joint stats + compares tau magnitude when error is large.
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


def find_csv(run_dir: Path, key: str) -> Path | None:
    cands = list(run_dir.glob(f"*{key}*.csv"))
    if not cands:
        return None
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


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--err_thresh", type=float, default=1.0, help="Threshold on |e_wrap| (rad) for 'large error' slices")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tau_path = find_csv(run_dir, "tau")
    qc_path = find_csv(run_dir, "q_cmd")
    qm_path = find_csv(run_dir, "q_meas")

    if tau_path is None or qc_path is None or qm_path is None:
        print("ERROR: need tau.csv, q_cmd.csv, q_meas.csv in run_dir", file=sys.stderr)
        print(f"Found: tau={tau_path}, q_cmd={qc_path}, q_meas={qm_path}", file=sys.stderr)
        return 2

    df_tau = pd.read_csv(tau_path)
    df_qc  = pd.read_csv(qc_path)
    df_qm  = pd.read_csv(qm_path)

    tcol_tau = choose_time_col(df_tau)
    tcol_qc  = choose_time_col(df_qc)
    tcol_qm  = choose_time_col(df_qm)

    tau_cols = numeric_cols(df_tau)
    qc_cols  = numeric_cols(df_qc)
    qm_cols  = numeric_cols(df_qm)
    if tcol_tau in tau_cols: tau_cols.remove(tcol_tau)
    if tcol_qc  in qc_cols:  qc_cols.remove(tcol_qc)
    if tcol_qm  in qm_cols:  qm_cols.remove(tcol_qm)

    common = [c for c in qc_cols if c in qm_cols]
    if len(common) < 1:
        print("ERROR: no matching joint columns between q_cmd and q_meas", file=sys.stderr)
        return 2

    # Timebase: tau time
    t = df_tau[tcol_tau].to_numpy(float) if tcol_tau else np.arange(len(df_tau), dtype=float)

    tc = df_qc[tcol_qc].to_numpy(float) if tcol_qc else np.arange(len(df_qc), dtype=float)
    tm = df_qm[tcol_qm].to_numpy(float) if tcol_qm else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)
    qc  = df_qc[common].to_numpy(float)
    qm  = df_qm[common].to_numpy(float)

    # Interpolate q onto tau time
    qc_i = np.vstack([np.interp(t, tc, qc[:, j]) for j in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, j]) for j in range(qm.shape[1])]).T

    e_raw  = qc_i - qm_i
    e_wrap = wrap_to_pi(e_raw)

    abs_tau = np.abs(tau)
    abs_e_raw  = np.abs(e_raw)
    abs_e_wrap = np.abs(e_wrap)

    # Map tau columns to q columns (best-effort by substring)
    tau_map = {}
    for i, tname in enumerate(tau_cols):
        for cname in common:
            if cname in tname or tname in cname:
                tau_map[cname] = i
                break

    mapped = [c for c in common if c in tau_map]
    if len(mapped) < 1:
        print("ERROR: couldn't match tau columns to q columns by name", file=sys.stderr)
        print("Tau cols example:", tau_cols[:10], file=sys.stderr)
        print("Q cols example  :", common[:10], file=sys.stderr)
        return 2

    # Per-joint wrap diagnosis
    rows = []
    for j, cname in enumerate(common):
        if cname not in tau_map:
            continue
        i_tau = tau_map[cname]
        rows.append((
            cname,
            float(np.max(abs_e_raw[:, j])),
            float(np.max(abs_e_wrap[:, j])),
            float(np.max(abs_tau[:, i_tau])),
        ))

    # Sort by largest raw error
    rows.sort(key=lambda r: r[1], reverse=True)

    print("\n=== Wrap check: does huge error come from angle wrapping? ===")
    print(f"Run dir: {run_dir}")
    print(f"err_thresh (wrap): {args.err_thresh} rad")
    print("\nTop joints by max |e_raw| (showing max |e_raw| vs max |e_wrap|):")
    for r in rows[: min(args.top, len(rows))]:
        name, emax_raw, emax_wrap, taumax = r
        print(f"  {name:>14s}: max|e_raw|={emax_raw:.3f}, max|e_wrap|={emax_wrap:.3f}, max|tau|={taumax:.3f}")

    # Global comparison
    print("\nOverall error stats (all joints pooled):")
    print(f"  max |e_raw| : {np.max(abs_e_raw):.3f} rad")
    print(f"  max |e_wrap|: {np.max(abs_e_wrap):.3f} rad")
    print(f"  p95 |e_raw| : {np.percentile(abs_e_raw,95):.3f} rad")
    print(f"  p95 |e_wrap|: {np.percentile(abs_e_wrap,95):.3f} rad")

    # Torque when wrapped error is large
    # Build a single representative scalar: mean abs tau across matched joints
    idxs = [tau_map[c] for c in mapped]
    tau_mean = np.mean(abs_tau[:, idxs], axis=1)
    ewrap_mean = np.mean(abs_e_wrap[:, [common.index(c) for c in mapped]], axis=1)

    mask_hi = ewrap_mean > args.err_thresh
    if np.any(mask_hi):
        print("\nMean-|tau| conditioned on wrapped error:")
        print(f"  mean|tau| when mean|e_wrap| <= {args.err_thresh}: {np.mean(tau_mean[~mask_hi]):.4f}")
        print(f"  mean|tau| when mean|e_wrap| >  {args.err_thresh}: {np.mean(tau_mean[mask_hi]):.4f}")
        print(f"  fraction of time in high-error: {np.mean(mask_hi)*100:.2f}%")
    else:
        print("\nNo timesteps where mean|e_wrap| exceeds threshold; try a smaller --err_thresh.")

    print("\nInterpretation:")
    print("- If max|e_wrap| drops a lot vs max|e_raw|, your previous 'large error' was mostly wrap/mismatch.")
    print("- If e_wrap is still large but |tau| stays small, then we next check maxForce/torque limits and what tau.csv actually logs.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
