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

def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0,1])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sign_map", required=True)
    ap.add_argument("--err_thr", type=float, default=0.5)
    ap.add_argument("--vel_thr", type=float, default=2.0)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(args.sign_map, "r") as f:
        joint_sign = yaml.safe_load(f)["joint_sign"]  # dict: joint -> +/-1

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

    # time base = tau time
    t  = df_tau[t_tau].to_numpy(float) if t_tau else np.arange(len(df_tau), dtype=float)
    tc = df_qc[t_qc].to_numpy(float)   if t_qc  else np.arange(len(df_qc), dtype=float)
    tm = df_qm[t_qm].to_numpy(float)   if t_qm  else np.arange(len(df_qm), dtype=float)

    tau = df_tau[tau_cols].to_numpy(float)
    qc  = df_qc[joints].to_numpy(float)
    qm  = df_qm[joints].to_numpy(float)

    # interpolate q onto tau time
    qc_i = np.vstack([np.interp(t, tc, qc[:, k]) for k in range(qc.shape[1])]).T
    qm_i = np.vstack([np.interp(t, tm, qm[:, k]) for k in range(qm.shape[1])]).T

    # measured velocity (bullet coords)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    dq = np.gradient(qm_i, axis=0) / dt[:, None]

    rows = []
    for k, jname in enumerate(joints):
        if jname not in tau_map:
            continue
        s = int(joint_sign.get(jname, 1))

        # IMPORTANT: convert q_cmd(log) -> q_cmd_bullet via sign, then form bullet error
        e_bullet = (s * qc_i[:, k]) - qm_i[:, k]

        tj = tau[:, tau_map[jname]]
        dqj = dq[:, k]

        mask = (np.abs(e_bullet) > args.err_thr) & (np.abs(dqj) < args.vel_thr)
        kp_eff = np.median(np.abs(tj[mask]) / (np.abs(e_bullet[mask]) + 1e-12)) if np.any(mask) else float("nan")
        c = corr(tj[mask], e_bullet[mask]) if np.any(mask) else float("nan")

        rows.append((jname, s,
                     float(np.percentile(np.abs(e_bullet),95)), float(np.max(np.abs(e_bullet))),
                     float(np.percentile(np.abs(tj),95)), float(np.max(np.abs(tj))),
                     kp_eff, c))

    rows.sort(key=lambda r: r[3], reverse=True)

    print("\n=== Effective Kp check (Bullet error) ===")
    print(f"Run dir: {run_dir}")
    print(f"Sign map: {args.sign_map}")
    print(f"Using samples with |e_bullet|>{args.err_thr} rad and |dq|<{args.vel_thr} rad/s")
    print(f"{'joint':>14s} {'s':>3s} {'p95|e|':>7s} {'max|e|':>7s} {'p95|tau|':>9s} {'max|tau|':>9s} {'med|tau|/|e|':>14s} {'corr(tau,e)':>11s}")
    print("-"*92)
    for r in rows[: min(args.top, len(rows))]:
        j, s, e95, emax, t95, tmax, kp, c = r
        kp_s = "nan" if np.isnan(kp) else f"{kp:14.4f}"
        c_s  = "nan" if np.isnan(c)  else f"{c:11.3f}"
        print(f"{j:>14s} {s:3d} {e95:7.3f} {emax:7.3f} {t95:9.3f} {tmax:9.3f} {kp_s} {c_s}")

if __name__ == "__main__":
    main()
