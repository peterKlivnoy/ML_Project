"""Walk-forward score of the submitted XGBoost model (submit.py), fold by fold.

Log 036 quotes 0.9638 +/- 0.0115 over the six folds but only one per-fold value survives in the
report, and the folds were run in notebooks. This re-runs the same architecture with
`submit.py`'s own building blocks, so the physics model can be compared on identical folds:

    python baseline_walkforward.py [stride]

Per fold the count model that RANKS the count-table features is fitted on that fold's training
unit-weeks only (submit.py fits it on all labelled weeks, which is fine at submission time but
would leak validation labels into feature selection here).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import count_model as CM
import features as F
import hourly_structure as HS
import longform as LF
import submit as SU
from physics_model import FOLDS, TEST_LIKE


def prepare():
    """Everything that does not depend on the fold: stage features and the count table."""
    tr, te, units = SU.build_features(verbose=False)
    both = pd.concat([tr, te], ignore_index=True)
    ct = SU.build_count_table(both, units)
    ct["bin"] = CM.bin_count(ct["count"].fillna(0))
    hs = HS.build(tr)
    tr = pd.concat([tr, hs], axis=1)
    tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    pruned = pd.read_csv(SU._ROOT / "data/processed/pruned_features.csv")["feature"].tolist()
    tw = [c for c in tr.columns if c.startswith("f07_term_spread_")
          or c in ("f07_terminal_slope", "f07_terminal_wv_lag8_minus_d7")]
    shared = [c for c in pruned + tw + list(hs.columns) if c in tr.columns]
    gens = F.load_topology()["model"]["generator"]
    merit = {"Tokke_G1": 1, "Tokke_G3": 2, "Tokke_G4": 3, "Tokke_G2": 4,
             "Vinje_G1": 1, "Vinje_G2": 2, "Vinje_G3": 3}
    stat = pd.DataFrame([{
        "unit": u,
        "s_pmax": gens[[k for k in gens if F.sanitize(k) == u][0]]["p_max"],
        "s_pmin": gens[[k for k in gens if F.sanitize(k) == u][0]]["p_min"],
        "s_merit": merit.get(u, 1),
        "s_nsib": sum(1 for v in units if v.rsplit("_G", 1)[0] == u.rsplit("_G", 1)[0]),
    } for u in units])
    return tr, ct, units, shared, stat


def run(stride: int = 3, folds=FOLDS):
    from xgboost import QuantileDMatrix, XGBClassifier, train as xgb_train
    t0 = time.time()
    tr, ct, units, shared, stat = prepare()
    print(f"prepared in {time.time() - t0:.0f}s", flush=True)
    start = tr.groupby("run_no")["starttime"].first()
    feats = [c for c in ct.columns if c not in ("run_no", "unit", "count", "cls", "bin")]
    rows = []
    for a, b in folds:
        t1 = time.time()
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        fit_runs = set(start.index[start < a])
        val_runs = set(start.index[(start >= a) & (start < b)])
        ct_fit = ct[ct.run_no.isin(fit_runs) & ct["count"].notna()]
        seen = sorted(ct_fit["bin"].unique())
        cm = XGBClassifier(objective="multi:softprob", num_class=len(seen),
                           eval_metric="mlogloss", **SU.COUNT_PARAMS)
        cm.fit(ct_fit[feats], ct_fit["bin"].map({x: i for i, x in enumerate(seen)}))
        imp = pd.Series(cm.feature_importances_, index=feats).sort_values(ascending=False)
        keep = list(imp.head(SU.N_CORE).index)
        keep += [c for c in imp.index[SU.N_CORE:]
                 if c.startswith(("own_", "upstream_"))][:SU.N_UNITREL]
        ctx = ct.set_index(["run_no", "unit"])[keep]

        def long(runs, cs):
            L = LF.to_long(tr[tr.run_no.isin(runs)], units, shared, static=stat, case_stride=cs)
            j = ctx.reindex(pd.MultiIndex.from_arrays([L["run_no"], L["unit"].astype(str)]))
            blk = pd.DataFrame(j.to_numpy(dtype=np.float32),
                               columns=[f"cf_{c}" for c in keep], index=L.index)
            return pd.concat([L, blk], axis=1)

        L, V = long(fit_runs, stride), long(val_runs, 1)
        cols = [c for c in L.columns if c not in ("run_no", "hour", "y")]
        dtr = QuantileDMatrix(L[cols], label=L["y"], enable_categorical=True,
                              max_bin=SU.POINT_PARAMS["max_bin"])
        bst = xgb_train({**SU.POINT_PARAMS, "nthread": 4}, dtr, num_boost_round=SU.POINT_ROUNDS)
        dva = QuantileDMatrix(V[cols], enable_categorical=True, ref=dtr,
                              max_bin=SU.POINT_PARAMS["max_bin"])
        p = bst.predict(dva)
        y = V["y"].to_numpy()
        r = dict(fold=str(a)[:7], n_train_cases=len(fit_runs), micro=roc_auc_score(y, p))
        for u in units:
            m = (V["unit"].astype(str) == u).to_numpy()
            r[u] = roc_auc_score(y[m], p[m]) if 0 < y[m].mean() < 1 else np.nan
        rows.append(r)
        print(f"  {r['fold']}  micro {r['micro']:.4f} | Lio {r['Lio_G1']:.4f} | "
              f"{time.time() - t1:.0f}s", flush=True)
        del L, V, dtr, dva, bst
    d = pd.DataFrame(rows).set_index("fold")
    like = [f for f in TEST_LIKE if f in d.index]
    print(f"MEAN micro {d.micro.mean():.4f} (SD {d.micro.std():.4f}) | test-like "
          f"{d.loc[like, 'micro'].mean():.4f}", flush=True)
    return d


if __name__ == "__main__":
    d = run(stride=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
    pd.set_option("display.width", 250)
    print(d.round(4).T)
    if len(sys.argv) > 2:
        d.to_csv(sys.argv[2])
