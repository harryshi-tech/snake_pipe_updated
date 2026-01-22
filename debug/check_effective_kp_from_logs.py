#!/usr/bin/env python3
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
    return (x + np.pi) % (2*np.pi) - np.pi

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--err_thr", default=0.5, type=float, help="Use samples with |e| > err_thr (rad)")
    ap.add_argument("--vel_thr", default=2.0, type=float, help="Use samples with |dq| < vel_thr (rad/s)")
    ap.add_argument("--top", default=10, type=int)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    tau_path = find_csv(run_dir, "tau")
    qc_path  = find_csv(run_dir, "q_cmd")
    qm_path  = find_csv(run_dir, "q_meas")

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

    # map tau columns to joints by substring
    tau_map = {}
    for i, tc in enumerate(tau_cols):
        for jc in common:
            if jc in tc or tc in jc:
                tau_map[jc] = i
                break
    mapped = [j for j in common if j in tau_map]
    if len(mapped) < len(common):
        print(f"Warning: could only match {len(mapped)}/{len(common)} joints to tau columns")

    # Time base = tau time
    t = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float) if t_qc else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float) if t_qm else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)

    qc = df_qc[common].to_numpy(float)
    qm = df_qm[common].to_numpy(float)

    # interpolate q onto tau time
    qc_i = np.vstack([np.interp(t, tc, qc[:, j]) for j in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, j]) for j in range(qm.shape[1])]).T

    e = wrap_to_pi(qc_i - qm_i)  # (T,J)
    # finite-diff velocity from measured q
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    dq = np.gradient(qm_i, axis=0) / dt[:, None]

    rows = []
    for j_idx, jname in enumerate(common):
        if jname not in tau_map:
            continue
        i_tau = tau_map[jname]
        tau_j = tau[:, i_tau]
        e_j   = e[:, j_idx]
        dq_j  = dq[:, j_idx]

        mask = (np.abs(e_j) > args.err_thr) & (np.abs(dq_j) < args.vel_thr)
        if np.any(mask):
            kp_eff = np.median(np.abs(tau_j[mask]) / (np.abs(e_j[mask]) + 1e-12))
        else:
            kp_eff = np.nan

        # signed correlation (should be positive-ish if it's PD-like)
        if np.std(tau_j) < 1e-12 or np.std(e_j) < 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(tau_j, e_j)[0,1])

        rows.append((
            jname,
            float(np.percentile(np.abs(e_j), 95)),
            float(np.max(np.abs(e_j))),
            float(np.percentile(np.abs(tau_j), 95)),
            float(np.max(np.abs(tau_j))),
            kp_eff,
            corr
        ))

    rows.sort(key=lambda r: r[2], reverse=True)

    print("\n=== Effective Kp check from logs ===")
    print(f"Run dir: {run_dir}")
    print(f"Using samples with |e|>{args.err_thr} rad and |dq|<{args.vel_thr} rad/s")
    print(f"{'joint':>14s}  {'p95|e|':>7s}  {'max|e|':>7s}  {'p95|tau|':>9s}  {'max|tau|':>9s}  {'med|tau|/|e|':>14s}  {'corr(tau,e)':>11s}")
    print("-"*86)
    for r in rows[: min(args.top, len(rows))]:
        j, e95, emax, t95, tmax, kp, corr = r
        kp_s = "nan" if np.isnan(kp) else f"{kp:14.4f}"
        corr_s = "nan" if np.isnan(corr) else f"{corr:11.3f}"
        print(f"{j:>14s}  {e95:7.3f}  {emax:7.3f}  {t95:9.3f}  {tmax:9.3f}  {kp_s}  {corr_s}")

    print("\nHow to read this:")
    print("- If med|tau|/|e| is very small (e.g., 0.1–1.0 N*m/rad) and corr(tau,e) ~ 0 => your position gains are likely too low OR tau.csv isn't actuator torque.")
    print("- If corr(tau,e) is clearly positive and med|tau|/|e| looks like a reasonable stiffness => the torque is behaving like PD.\n")

if __name__ == "__main__":
    main()
