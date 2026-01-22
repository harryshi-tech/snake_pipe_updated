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

def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--err_thr", default=0.3, type=float, help="only analyze samples with |e|>err_thr (rad)")
    ap.add_argument("--top", default=16, type=int)
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

    # map tau cols to joints by substring match
    tau_map = {}
    for i, tc in enumerate(tau_cols):
        for jc in common:
            if jc in tc or tc in jc:
                tau_map[jc] = i
                break

    # time base = tau time
    t = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float) if t_qc else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float) if t_qm else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)

    qc = df_qc[common].to_numpy(float)
    qm = df_qm[common].to_numpy(float)

    qc_i = np.vstack([np.interp(t, tc, qc[:, j]) for j in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, j]) for j in range(qm.shape[1])]).T

    e = wrap_to_pi(qc_i - qm_i)

    rows = []
    for j_idx, jname in enumerate(common):
        if jname not in tau_map:
            continue
        i_tau = tau_map[jname]
        tau_j = tau[:, i_tau]
        e_j   = e[:, j_idx]

        mask = np.abs(e_j) > args.err_thr
        n = int(np.sum(mask))
        if n < 50:
            rows.append((jname, n, float("nan"), float("nan"), "too_few"))
            continue

        c_pos = corr(tau_j[mask], e_j[mask])
        c_neg = corr(tau_j[mask], -e_j[mask])

        # which sign matches better?
        if np.isnan(c_pos) or np.isnan(c_neg):
            verdict = "nan"
        elif abs(c_neg) > abs(c_pos) and c_neg > 0:
            verdict = "tau ≈ K*(-e)  (sign flipped)"
        elif c_pos > 0:
            verdict = "tau ≈ K*(e)   (consistent)"
        else:
            verdict = "unclear (non-PD / different signal)"

        rows.append((jname, n, c_pos, c_neg, verdict))

    # sort by most confidently "sign flipped"
    def key(r):
        _, n, cpos, cneg, verdict = r
        score = 0.0
        if isinstance(cneg, float) and not np.isnan(cneg):
            score = cneg
        return (-score, -n)
    rows.sort(key=key)

    print("\n=== Torque vs error sign consistency ===")
    print(f"Run dir: {run_dir}")
    print(f"Using samples with |e| > {args.err_thr} rad")
    print(f"{'joint':>14s}  {'N':>6s}  {'corr(tau,e)':>12s}  {'corr(tau,-e)':>13s}  verdict")
    print("-"*75)
    for r in rows[: min(args.top, len(rows))]:
        j, n, cpos, cneg, verdict = r
        cpos_s = "nan" if np.isnan(cpos) else f"{cpos:12.3f}"
        cneg_s = "nan" if np.isnan(cneg) else f"{cneg:13.3f}"
        print(f"{j:>14s}  {n:6d}  {cpos_s}  {cneg_s}  {verdict}")

    print("\nIf many joints say 'sign flipped', your tau sign and q error sign conventions disagree (axis direction / mapping).")

if __name__ == "__main__":
    main()
