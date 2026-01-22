#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def find_csv(run_dir: Path, key: str) -> Path:
    cands = list(run_dir.glob(f"*{key}*.csv"))
    if not cands:
        raise FileNotFoundError(f"Can't find *{key}*.csv in {run_dir}")
    cands.sort(key=lambda p: (len(p.name), p.name))
    return cands[0]

def time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() in ["t","time","sim_time","sec","timestamp"]:
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
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sign_map", required=True)
    ap.add_argument("--err_thr", type=float, default=0.3)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(args.sign_map, "r") as f:
        joint_sign = yaml.safe_load(f)["joint_sign"]  # dict: name -> +/-1

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
    joints = [c for c in qc_cols if c in qm_cols]

    # map tau cols to joints by substring
    tau_map = {}
    for i, tc in enumerate(tau_cols):
        for j in joints:
            if j in tc or tc in j:
                tau_map[j] = i
                break

    t  = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float)   if t_qc  else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float)   if t_qm  else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)
    qc  = df_qc[joints].to_numpy(float)
    qm  = df_qm[joints].to_numpy(float)

    qc_i = np.vstack([np.interp(t, tc, qc[:, k]) for k in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, k]) for k in range(qm.shape[1])]).T

    e = wrap_to_pi(qc_i - qm_i)

    print("\n=== corr(tau, e_corrected) after applying sign map ===")
    print(f"Run dir: {run_dir}")
    print(f"Sign map: {args.sign_map}")
    print(f"Using |e| > {args.err_thr} rad")
    print(f"{'joint':>14s}  {'sign':>5s}  {'corr(tau, e_corr)':>18s}")
    print("-"*43)

    for k, jname in enumerate(joints):
        if jname not in tau_map:
            continue
        s = int(joint_sign.get(jname, 1))
        tj = tau[:, tau_map[jname]]
        ej = e[:, k]
        ej_corr = s * ej

        mask = np.abs(ej_corr) > args.err_thr
        c = corr(tj[mask], ej_corr[mask]) if np.any(mask) else float("nan")
        print(f"{jname:>14s}  {s:5d}  {c:18.3f}")

if __name__ == "__main__":
    main()
