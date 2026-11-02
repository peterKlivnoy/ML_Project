"""Sensitivity of the submitted LightGBM settings, one knob at a time.

    python lgbm_sensitivity.py [stride]

The settings in Short notebook 1 were carried over from Track A's tuned XGBoost (log 034), not
tuned for this model. This varies each knob around them on the two test-like walk-forward folds
(2021-H2, 2022-H2; adversarial.py) and writes report/runs/lgbm_sensitivity_s<stride>.csv.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import physics_model as PM

FOLDS = [f for f in PM.FOLDS if f[0][:7] in PM.TEST_LIKE]
VARIANTS = [
    ("submitted", {}, PM.ROUNDS),
    ("num_leaves 31", {"num_leaves": 31}, PM.ROUNDS),
    ("num_leaves 255", {"num_leaves": 255}, PM.ROUNDS),
    ("min_data_in_leaf 100", {"min_data_in_leaf": 100}, PM.ROUNDS),
    ("min_data_in_leaf 2000", {"min_data_in_leaf": 2000}, PM.ROUNDS),
    ("learning rate 0.1, 300 rounds", {"learning_rate": 0.1}, 300),
    ("learning rate 0.025, 1200 rounds", {"learning_rate": 0.025}, 1200),
    ("feature_fraction 0.5", {"feature_fraction": 0.5}, PM.ROUNDS),
    ("feature_fraction 1.0", {"feature_fraction": 1.0}, PM.ROUNDS),
]


def main(stride=3):
    meta, X, names = PM.load()
    st = meta["starttime"].to_numpy()
    y = meta["y"].to_numpy()
    run = meta["run_no"].to_numpy()
    rows = []
    for a, b in FOLDS:
        a64, b64 = np.datetime64(a), np.datetime64(b)
        tr = (st < a64) & ~np.isnan(y) & (run % stride == 0)
        va = (st >= a64) & (st < b64) & ~np.isnan(y)
        for label, change, rounds in VARIANTS:
            t0 = time.time()
            bst = PM.fit(X[tr], y[tr], names, rounds, params={**PM.PARAMS, **change})
            auc = roc_auc_score(y[va], bst.predict(X[va]))
            rows.append(dict(fold=a[:7], variant=label, micro=auc, fit_s=round(time.time() - t0)))
            print(f"  {a[:7]}  {label:<34} micro {auc:.4f}  ({rows[-1]['fit_s']}s)", flush=True)
        pd.DataFrame(rows).to_csv(PM.S.ROOT / f"report/runs/lgbm_sensitivity_s{stride}.csv",
                                  index=False)
    d = pd.DataFrame(rows).pivot(index="variant", columns="fold", values="micro")
    d["mean"] = d.mean(axis=1)
    print(d.loc[[v[0] for v in VARIANTS]].round(4).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
