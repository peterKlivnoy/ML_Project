"""Different predictor types on identical features and folds.

    python predictor_comparison.py [stride]

Every model in the project so far was a gradient-boosted tree ensemble (XGBoost in Track A,
LightGBM in Track B). This runs four predictor types on the same 61 LP + input features
(physics_features.py) and the same six walk-forward folds as physics_model.walkforward:

    logistic regression   linear model: median-imputed, standardised, one-hot unit and plant
    random forest         bagged deep trees (no boosting)
    XGBoost               boosted trees, Track A's tuned settings (log 034)
    LightGBM              boosted trees, the submitted model's settings

Writes report/runs/predictor_comparison_s<stride>.csv (micro ROC-AUC per fold and predictor,
plus fit time).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import physics_model as PM

XGB_PARAMS = dict(max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=500, tree_method="hist", max_bin=256,
                  objective="binary:logistic", nthread=4, seed=0)
XGB_ROUNDS = 600


def logistic(names):
    cat = [names.index(c) for c in PM.CATEGORICAL if c in names]
    num = [i for i in range(len(names)) if i not in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, LogisticRegression(C=1.0, max_iter=500))


def random_forest(names):
    # sklearn's forests accept NaN; min_samples_leaf mirrors the boosted models' 500-row leaves
    # at a quarter of the rows per tree
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=100, max_features="sqrt",
                                  max_samples=0.25, n_jobs=4, random_state=0)


def fit_predict(kind, Xtr, ytr, Xva, names):
    if kind == "logistic regression":
        m = logistic(names).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    if kind == "random forest":
        m = random_forest(names).fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]
    if kind == "XGBoost":
        import xgboost as xgb
        ft = ["c" if n in PM.CATEGORICAL else "q" for n in names]
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=names, feature_types=ft,
                                  enable_categorical=True, max_bin=XGB_PARAMS["max_bin"])
        bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=XGB_ROUNDS)
        dva = xgb.DMatrix(Xva, feature_names=names, feature_types=ft, enable_categorical=True)
        return bst.predict(dva)
    if kind == "LightGBM":
        return PM.fit(Xtr, ytr, names).predict(Xva)
    raise ValueError(kind)


KINDS = ["logistic regression", "random forest", "XGBoost", "LightGBM"]


def main(stride=3):
    meta, X, names = PM.load()
    st = meta["starttime"].to_numpy()
    y = meta["y"].to_numpy()
    run = meta["run_no"].to_numpy()
    rows = []
    for a, b in PM.FOLDS:
        a64, b64 = np.datetime64(a), np.datetime64(b)
        tr = (st < a64) & ~np.isnan(y) & (run % stride == 0)
        va = (st >= a64) & (st < b64) & ~np.isnan(y)
        for kind in KINDS:
            t0 = time.time()
            p = fit_predict(kind, X[tr], y[tr], X[va], names)
            r = dict(fold=a[:7], predictor=kind, micro=roc_auc_score(y[va], p),
                     fit_s=round(time.time() - t0), n_train_rows=int(tr.sum()))
            rows.append(r)
            print(f"  {r['fold']}  {kind:<20} micro {r['micro']:.4f}  ({r['fit_s']}s)", flush=True)
        pd.DataFrame(rows).to_csv(PM.S.ROOT / f"report/runs/predictor_comparison_s{stride}.csv",
                                  index=False)
    d = pd.DataFrame(rows).pivot(index="fold", columns="predictor", values="micro")[KINDS]
    print(d.round(4).to_string())
    print("mean", d.mean().round(4).to_dict())
    print("test-like", d.loc[[f for f in PM.TEST_LIKE if f in d.index]].mean().round(4).to_dict())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
