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
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--err_thr", type=float, default=0.3, help="only use samples with |e|>err_thr")
    ap.add_argument("--out", default="debug/joint_sign_map.yaml")
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
    joints = [c for c in qc_cols if c in qm_cols]
    if not joints:
        raise RuntimeError("No matching joint columns between q_cmd and q_meas.")

    # Map tau columns to joints by substring match
    tau_map = {}
    for i, tc in enumerate(tau_cols):
        for j in joints:
            if j in tc or tc in j:
                tau_map[j] = i
                break

    # Timebase = tau time
    t  = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float)   if t_qc  else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float)   if t_qm  else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)
    qc  = df_qc[joints].to_numpy(float)
    qm  = df_qm[joints].to_numpy(float)

    qc_i = np.vstack([np.interp(t, tc, qc[:, k]) for k in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, k]) for k in range(qm.shape[1])]).T

    e = wrap_to_pi(qc_i - qm_i)

    sign_map = {}
    report = []

    for k, jname in enumerate(joints):
        if jname not in tau_map:
            continue
        tj = tau[:, tau_map[jname]]
        ej = e[:, k]
        mask = np.abs(ej) > args.err_thr
        if np.sum(mask) < 50:
            continue
        c = corr(tj[mask], ej[mask])
        s = -1 if (not np.isnan(c) and c < 0) else 1
        sign_map[jname] = int(s)
        report.append((jname, float(c), int(s)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump({"joint_sign": sign_map}, f, sort_keys=True)

    print(f"\nWrote sign map to: {out_path}")
    print("Format: joint_sign[joint_name] = +1 or -1")
    print("\nSample (first 16):")
    for j, c, s in report[:16]:
        print(f"  {j:>14s}: corr(tau,e)={c: .3f}  => sign {s:+d}")

if __name__ == "__main__":
    main()
