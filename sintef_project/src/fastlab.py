"""Fast experiment harness: sub-40s arms instead of ~110s.

Measured breakdown of a 110s arm in the long-format pipeline:

    ~35s  to_long rebuilt from scratch for every arm
    ~25s  attaching 120 `cf_` columns one at a time to a DataFrame (O(n^2) reallocation)
    ~35s  600 trees, depth 8, 1.7M rows
    ~15s  predict, AUC, teardown

plus ~66s of setup per PROCESS refitting the count model and rebuilding the feature table.

None of that is inherent. The shared part of the long frame does not change between arms, the
`cf_` block is a dense numeric matrix that has no business being inserted column-by-column into
pandas, and a screening fit does not need 600 trees to rank candidates correctly.

This module holds the invariant part in memory as one preallocated float32 matrix and swaps only
the columns an arm actually changes. Feature blocks are assigned into slices, never concatenated.

USE THIS FOR RANKING, NOT FOR SCORING. Screening parameters bias the level downward (the
convention since log 010); confirm anything that looks real at full settings before quoting it.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import QuantileDMatrix, train as xgb_train

warnings.filterwarnings("ignore")

SCREEN = dict(max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=500, tree_method="hist", max_bin=128,
              objective="binary:logistic", eval_metric="logloss", nthread=-1, seed=0)
FULL = dict(SCREEN, max_bin=256)
SCREEN_ROUNDS, FULL_ROUNDS = 300, 600


class Lab:
    """Holds the long-format matrix once; arms differ only by which columns are live.

    `base` is the invariant block (shared case/hour features, static unit attributes, unit id).
    `pool` is every candidate per-unit feature, materialised once; an arm names a subset and the
    rest are masked to NaN rather than rebuilt, so switching feature sets costs a memcpy instead
    of a table build.
    """

    def __init__(self, base: pd.DataFrame, pool: pd.DataFrame, y: np.ndarray,
                 unit: np.ndarray, is_val: np.ndarray):
        self.bcols = list(base.columns)
        self.pcols = list(pool.columns)
        n = len(base)
        self.X = np.empty((n, len(self.bcols) + len(self.pcols)), dtype=np.float32)
        self.X[:, :len(self.bcols)] = base.to_numpy(dtype=np.float32)
        self.X[:, len(self.bcols):] = pool.to_numpy(dtype=np.float32)
        self.pool_start = len(self.bcols)
        self.y = y.astype(np.float32)
        self.unit = unit
        self.tr = ~is_val
        self.va = is_val
        self.units = sorted(set(unit.tolist()))
        self._pool_backup = self.X[:, self.pool_start:].copy()

    def _live(self, keep: list[str], masks: dict[str, list[str]] | None):
        """Restore the pool, then blank whatever this arm does not use."""
        self.X[:, self.pool_start:] = self._pool_backup
        keep_set = set(keep)
        for j, c in enumerate(self.pcols):
            col = self.pool_start + j
            if c not in keep_set:
                self.X[:, col] = np.nan
            elif masks and c in masks:
                off = ~np.isin(self.unit, masks[c])
                self.X[off, col] = np.nan

    def run(self, keep: list[str], label: str, masks=None, screen=True, quiet=False):
        t = time.time()
        self._live(keep, masks)
        cols = self.bcols + self.pcols
        params = SCREEN if screen else FULL
        rounds = SCREEN_ROUNDS if screen else FULL_ROUNDS
        # max_bin must match between the QuantileDMatrix and the booster; the default is 256,
        # so screening at 128 has to say so explicitly or XGBoost refuses to train.
        mb = params["max_bin"]
        dtr = QuantileDMatrix(self.X[self.tr], label=self.y[self.tr],
                              feature_names=cols, max_bin=mb)
        dva = QuantileDMatrix(self.X[self.va], label=self.y[self.va],
                              feature_names=cols, ref=dtr, max_bin=mb)
        bst = xgb_train(params, dtr, num_boost_round=rounds)
        p = bst.predict(dva)
        yv = self.y[self.va]
        uv = self.unit[self.va]
        micro = roc_auc_score(yv, p)
        per = pd.Series({u: roc_auc_score(yv[uv == u], p[uv == u]) for u in self.units})
        if not quiet:
            print(f"  {label:<34} {len(keep) + len(self.bcols):3d}f | micro {micro:.4f} | "
                  f"Lio {per.get('Lio_G1', np.nan):.4f} Hauk {per.get('Haukeli_G1', np.nan):.4f}"
                  f" | {time.time() - t:4.0f}s", flush=True)
        return micro, per, bst
