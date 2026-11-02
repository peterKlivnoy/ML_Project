"""
Fast screening harness for the count -> pointwise architecture.

`report_rules.md` has required a screening mode since log 010: rank candidates cheaply, confirm
finalists at full cost, and never quote a screening number as a result. That convention lapsed
when the architecture changed to count-then-pointwise, and iteration cost drifted back up to
~80s per arm -- ~10 min for an eight-arm ablation, which is too slow to think in.

Where the time goes, measured on the log 030 ablation:

    count model fit (40,908 x 437)                ~20s
    pointwise fit (490k hourly rows, 300 trees)   ~60s   <- the bottleneck

The pointwise fit is the wasteful part. Each case contributes 168 near-duplicate hourly rows,
and consecutive cases overlap 144 wall-clock hours at 88% label agreement (log 017). Fitting all
of them buys almost nothing over fitting a strided subset.

`case_stride` therefore keeps every k-th case rather than every k-th ROW: dropping rows within a
case would leave the remaining hours of that same case in the training set, which is the
near-duplicate leak log 017 documented, and would flatter the screening score. Dropping whole
cases removes the duplicate group intact.

Validation rows are never subsampled -- evaluation is nearly free and a strided validation set
would add variance exactly where we are trying to read small differences.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

FULL = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", multi_strategy="multi_output_tree",
            objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=0)
SCREEN = {**FULL, "n_estimators": 150}


def pointwise_eval(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], targets: list[str],
                   case_stride: int = 1, screen: bool = False) -> tuple[float, pd.Series]:
    """Fit the pointwise model and return (micro-AUC, per-unit AUC).

    case_stride=3 with screen=True is the fast path: ~6x cheaper than the full fit.
    """
    tr = train
    if case_stride > 1:
        keep = np.sort(tr["run_no"].unique())[::case_stride]
        tr = tr[tr["run_no"].isin(keep)]
    m = XGBClassifier(**(SCREEN if screen else FULL))
    m.fit(tr[feats], tr[targets].astype(int))
    Y = val[targets].to_numpy()
    P = m.predict_proba(val[feats])
    micro = roc_auc_score(Y, P, average="micro")
    per = pd.Series({t[3:]: roc_auc_score(Y[:, i], P[:, i]) for i, t in enumerate(targets)})
    return float(micro), per
