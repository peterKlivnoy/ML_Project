"""LP + LightGBM on Track C's validation protocol, so the two selected models can be compared.

    python lp_yearly_cv.py

Expanding window by year: train on every labelled case starting more than 7 days before 1 January of
year Y (all cases, no stride, as Track C does), score the cases starting in Y. Writes
report/runs/lp_lightgbm_yearly_cv.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import physics_model as PM


def main(years=(2019, 2020, 2021, 2022)):
    meta, X, names = PM.load()
    st = meta["starttime"].to_numpy()
    y = meta["y"].to_numpy()
    rows = []
    for Y in years:
        t0 = time.time()
        cut = np.datetime64(f"{Y}-01-01") - np.timedelta64(7, "D")
        tr = (st < cut) & ~np.isnan(y)
        va = (st >= np.datetime64(f"{Y}-01-01")) & (st < np.datetime64(f"{Y + 1}-01-01")) & ~np.isnan(y)
        p = PM.fit(X[tr], y[tr], names).predict(X[va])
        rows.append(dict(year=Y, micro=roc_auc_score(y[va], p), fit_s=round(time.time() - t0)))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(PM.S.ROOT / "report/runs/lp_lightgbm_yearly_cv.csv", index=False)
    d = pd.DataFrame(rows)
    print(f"mean {d.micro.mean():.4f}")


if __name__ == "__main__":
    main()
