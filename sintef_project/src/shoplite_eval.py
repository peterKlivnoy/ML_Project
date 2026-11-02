"""Score shoplite's raw optimiser output against the SHOP labels, with no learning at all."""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import shoplite as S

UNITS = list(S.load_topology().units)


def _one(args):
    start, mip, kw = args
    opt = S.Options(**kw)
    try:
        o = S.solve_case(start, mip=mip, opt=opt, time_limit=60)
    except Exception as e:  # keep the sweep alive; a failed case is reported, not hidden
        return start, None, repr(e)
    return start, o, None


def run(starts, mip=False, procs=4, **kw):
    t = time.time()
    with Pool(procs) as pool:
        res = pool.map(_one, [(s, mip, kw) for s in starts], chunksize=1)
    bad = [(s, e) for s, o, e in res if o is None or f"u_{UNITS[0]}" not in o]
    print(f"solved {len(res) - len(bad)}/{len(res)} cases in {time.time() - t:.0f}s"
          + (f" | failed: {bad[:3]}" if bad else ""), flush=True)
    return {s: o for s, o, e in res if o is not None and f"u_{UNITS[0]}" in o}


def score(out: dict, labels: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Micro and per-unit AUC of several raw scores. labels: indexed by starttime."""
    rows = {k: [] for k in ("y", "unit", "u", "p", "margin")}
    for s, o in out.items():
        lab = labels.loc[pd.Timestamp(s)]
        pr = price.reindex(pd.date_range(s, periods=S.T, freq="h")).to_numpy()
        for u in UNITS:
            rows["y"].append(lab[[f"result_committed_{u}_t{h}" for h in range(S.T)]].to_numpy(float))
            rows["unit"].append(np.full(S.T, u))
            rows["u"].append(o[f"u_{u}"])
            rows["p"].append(o[f"p_{u}"] / S.load_topology().units[u].pmax)
            rows["margin"].append(pr - o[f"lwv_{u}"] if f"lwv_{u}" in o else np.full(S.T, np.nan))
    d = {k: np.concatenate(v) for k, v in rows.items()}
    res = []
    for key in ("u", "p", "margin"):
        if np.isnan(d[key]).all():
            continue
        r = {"score": key, "micro": roc_auc_score(d["y"], d[key])}
        for u in UNITS:
            m = d["unit"] == u
            yy = d["y"][m]
            r[u] = roc_auc_score(yy, d[key][m]) if 0 < yy.mean() < 1 else np.nan
        res.append(r)
    return pd.DataFrame(res).set_index("score")


def labels_frame():
    uc = pd.read_csv(S.ROOT / "data/kernel/Unit_commitment_decisions.csv", parse_dates=["starttime"])
    return uc.set_index("starttime")


if __name__ == "__main__":
    lab = labels_frame()
    starts = list(lab.index[::int(sys.argv[1]) if len(sys.argv) > 1 else 50])
    price = S.load_raw()[0]
    out = run(starts, mip=False)
    pd.set_option("display.width", 250)
    print(score(out, lab, price).round(3).T)
