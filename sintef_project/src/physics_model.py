"""Physics-first model: shoplite optimiser features + a gradient-boosted correction.

    python physics_model.py walkforward [stride]   # 6 x 6-month folds, 2020-H1 .. 2022-H2
    python physics_model.py submit                 # fit on all 2,922 cases, write submission

Walk-forward is the project's reference protocol (log 024/028): each fold trains on every case
that starts before the fold and scores the six months that follow. The submitted XGBoost model
scored 0.9638 +/- 0.0115 on it (log 036).
"""
from __future__ import annotations

import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import physics_features as PF
import shoplite as S

PROC = S.ROOT / "data/processed"
FOLDS = [("2020-01-01", "2020-07-01"), ("2020-07-01", "2021-01-01"),
         ("2021-01-01", "2021-07-01"), ("2021-07-01", "2022-01-01"),
         ("2022-01-01", "2022-07-01"), ("2022-07-01", "2023-01-01")]
# Half-years hardest to tell apart from 2023-24 on case-level inputs (adversarial.py):
# 2021-H2 AUC 0.785, 2022-H2 0.879; every other half-year is >= 0.97.
TEST_LIKE = ["2021-07", "2022-07"]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=4, verbose=-1, seed=0)
ROUNDS = 600
CATEGORICAL = ["s_unit", "s_plant"]


def load(modes=("lp",)):
    """Feature matrix for every case (train and test) from the stored optimiser solutions."""
    import shoplite_batch as SB
    cases = SB.all_cases()
    labels = pd.read_csv(S.ROOT / "data/kernel/Unit_commitment_decisions.csv",
                         parse_dates=["starttime"])
    metas, Xs, names = None, [], []
    for m in modes:
        unit = pd.read_parquet(PROC / f"shoplite_{m}_unit.parquet")
        case = pd.read_parquet(PROC / f"shoplite_{m}_case.parquet")
        meta, X, nm = PF.build(cases, unit, case, labels=labels if metas is None else None,
                               prefix=m)
        if metas is None:
            metas, Xs, names = meta, [X], nm
        else:   # second optimiser: keep only its own lp-style block
            keep = [i for i, n in enumerate(nm) if n.startswith(m + "_")]
            Xs.append(X[:, keep])
            names += [nm[i] for i in keep]
    X = Xs[0] if len(Xs) == 1 else np.hstack(Xs)
    metas["starttime"] = metas["run_no"].map(cases.set_index("run_no")["starttime"])
    return metas, X, names


def fit(X, y, names, rounds=ROUNDS, params=None):
    import lightgbm as lgb
    ds = lgb.Dataset(X, label=y, feature_name=names,
                     categorical_feature=[c for c in CATEGORICAL if c in names],
                     free_raw_data=True)
    return lgb.train(params or PARAMS, ds, num_boost_round=rounds)


def per_unit_auc(y, p, unit):
    units = [S.sanitize(u) for u in S.load_topology().units]
    out = {}
    for j, nm in enumerate(units):
        m = unit == j
        out[nm] = roc_auc_score(y[m], p[m]) if 0 < y[m].mean() < 1 else np.nan
    return out


def walkforward(stride=1, rounds=ROUNDS, folds=FOLDS, drop_prefix=(), modes=("lp",)):
    meta, X, names = load(modes)
    keep = [i for i, n in enumerate(names) if not n.startswith(tuple(drop_prefix))]
    if len(keep) < len(names):
        X, names = X[:, keep], [names[i] for i in keep]
    print(f"features: {len(names)} | rows {len(meta):,}", flush=True)
    st = meta["starttime"].to_numpy()
    y = meta["y"].to_numpy()
    run = meta["run_no"].to_numpy()
    rows = []
    for a, b in folds:
        t0 = time.time()
        a, b = np.datetime64(a), np.datetime64(b)
        tr = (st < a) & ~np.isnan(y)
        if stride > 1:
            tr &= (run % stride == 0)
        va = (st >= a) & (st < b) & ~np.isnan(y)
        bst = fit(X[tr], y[tr], names, rounds)
        p = bst.predict(X[va])
        lp = X[va, names.index("lp_p_rel")]
        r = dict(fold=f"{str(a)[:7]}", n_train_cases=len(np.unique(run[tr])),
                 micro=roc_auc_score(y[va], p), lp_raw=roc_auc_score(y[va], lp))
        r.update(per_unit_auc(y[va], p, meta["unit"].to_numpy()[va]))
        rows.append(r)
        print(f"  {r['fold']}  train {r['n_train_cases']} cases | micro {r['micro']:.4f} | "
              f"raw optimiser {r['lp_raw']:.4f} | Lio {r['Lio_G1']:.4f} | "
              f"{time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame(rows).set_index("fold")
    print(f"MEAN micro {d.micro.mean():.4f} (SD {d.micro.std():.4f}) | raw optimiser "
          f"{d.lp_raw.mean():.4f}", flush=True)
    like = [f for f in TEST_LIKE if f in d.index]
    if like:
        print(f"TEST-LIKE folds {like}: micro {d.loc[like, 'micro'].mean():.4f}", flush=True)
    return d, bst, names


def submit(out_path, rounds=ROUNDS, modes=("lp",)):
    meta, X, names = load(modes)
    y = meta["y"].to_numpy()
    tr = ~np.isnan(y)
    bst = fit(X[tr], y[tr], names, rounds)
    te = ~tr
    p = bst.predict(X[te])
    m = meta[te].copy()
    m["p"] = p
    units = list(S.load_topology().units)
    sample = pd.read_csv(S.ROOT / "sample_submission.csv")
    wide = m.pivot_table(index="run_no", columns=["unit", "hour"], values="p")
    cols = {}
    for c in sample.columns[1:]:
        g = re.fullmatch(r"result_committed_(.+)_t(\d+)", c)
        cols[c] = wide[(units.index(g.group(1)), int(g.group(2)))].to_numpy()
    out = pd.DataFrame(cols, index=wide.index).rename_axis("Run No").reset_index()
    assert list(out.columns) == list(sample.columns)
    assert set(out["Run No"]) == set(sample["Run No"]) and len(out) == len(sample)
    assert out.drop(columns="Run No").isna().sum().sum() == 0
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} {out.shape}", flush=True)
    return out, bst, names


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "walkforward"
    if cmd == "walkforward":
        d, bst, names = walkforward(stride=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
        pd.set_option("display.width", 250)
        print(d.round(4).T)
    elif cmd == "submit":
        submit(sys.argv[2] if len(sys.argv) > 2 else str(S.ROOT / "submission_physics.csv"))
