"""Solve shoplite for every case (train and test) and store the solutions as feature tables.

    python shoplite_batch.py lp      # LP relaxation, ~2s/case  -> data/processed/shoplite_lp_*.parquet
    python shoplite_batch.py mip     # binary commitment, ~4s/case

Outputs (one file per mode):
  shoplite_<mode>_unit.parquet   one row per (run_no, hour, unit): u, p (MW), q (m3/s), lwv (EUR/MWh)
  shoplite_<mode>_case.parquet   one row per (run_no, hour): reservoir volume v_*, water shadow
                                 price mu_* (EUR/Mm3, LP only), river flows f_*

Solutions are checkpointed in chunks under data/processed/shoplite_<mode>_parts/, so an
interrupted run resumes where it stopped.
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import shoplite as S

OUT = S.ROOT / "data/processed"
CHUNK = 100


def all_cases() -> pd.DataFrame:
    uc = pd.read_csv(S.ROOT / "data/kernel/Unit_commitment_decisions.csv",
                     usecols=["Run No", "starttime"])
    pm = pd.read_csv(S.ROOT / "prediction_mapping.csv")
    c = pd.concat([uc, pm], ignore_index=True).rename(columns={"Run No": "run_no"})
    c["starttime"] = pd.to_datetime(c["starttime"])
    return c


def _solve(args):
    run_no, start, mip = args
    try:
        o = S.solve_case(start, mip=mip, time_limit=60)
    except Exception as e:
        return run_no, None, repr(e)
    if f"u_{next(iter(S.load_topology().units))}" not in o:
        return run_no, None, o.get("status")
    return run_no, o, o["status"]


def _tables(run_no: int, o: dict):
    units = list(S.load_topology().units)
    T = S.T
    unit = pd.DataFrame({
        "run_no": np.int32(run_no),
        "hour": np.tile(np.arange(T, dtype=np.int16), len(units)),
        "unit": np.repeat([S.sanitize(u) for u in units], T),
        **{k: np.concatenate([o[f"{k}_{u}"] for u in units]).astype(np.float32)
           for k in ("u", "p", "q", "lwv") if f"{k}_{units[0]}" in o},
    })
    case = pd.DataFrame({"run_no": np.int32(run_no), "hour": np.arange(T, dtype=np.int16),
                         **{k: np.asarray(v, np.float32) for k, v in o.items()
                            if k.startswith(("v_", "mu_", "f_")) and np.ndim(v) == 1}})
    case["status_ok"] = o["status"] == "Optimal"
    return unit, case


def _warm(_):
    S.load_raw()
    S.load_topology()


def run(mode: str, procs: int = 4):
    mip = mode == "mip"
    parts = OUT / f"shoplite_{mode}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    cases = all_cases()
    todo = [k for k in range(0, len(cases), CHUNK)
            if not (parts / f"chunk_{k // CHUNK:03d}.parquet").exists()]
    t0 = time.time()
    # one pool for the whole run: each worker parses the raw CSVs once, not once per chunk
    with Pool(procs, initializer=_warm, initargs=(None,)) as pool:
        for k in todo:
            sub = cases.iloc[k:k + CHUNK]
            res = pool.map(_solve, [(r, s, mip) for r, s in zip(sub.run_no, sub.starttime)],
                           chunksize=1)
            units, cs, bad = [], [], []
            for run_no, o, status in res:
                if o is None:
                    bad.append((run_no, status))
                    continue
                a, b = _tables(run_no, o)
                units.append(a)
                cs.append(b)
            pd.concat(units).to_parquet(parts / f"chunk_{k // CHUNK:03d}.parquet")
            pd.concat(cs).to_parquet(parts / f"case_{k // CHUNK:03d}.parquet")
            print(f"[{mode}] {k + len(sub)}/{len(cases)} cases | {time.time() - t0:.0f}s"
                  + (f" | failed {bad}" if bad else ""), flush=True)
    unit = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("chunk_*.parquet"))])
    case = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("case_*.parquet"))])
    unit.to_parquet(OUT / f"shoplite_{mode}_unit.parquet", index=False)
    case.to_parquet(OUT / f"shoplite_{mode}_case.parquet", index=False)
    print(f"[{mode}] wrote {len(unit):,} unit rows, {len(case):,} case rows | "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "lp")
