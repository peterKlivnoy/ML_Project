"""
Two-stage (stacked) model.

Stage 1: the joint multi-output tree, which is good at cross-unit structure.
Stage 2: one small model per unit, which gets

  * the unit's OWN stage-1 prediction as `base_margin` (an exact logit offset, so stage 2 learns
    only the residual correction rather than re-deriving the joint prediction through splits),
  * the OTHER 13 units' stage-1 predictions as ordinary features (cross-unit conditioning),
  * that unit's own neighbourhood features - own/upstream/downstream reservoir state and the
    environmental constraints actually wired to it in the topology.

This gives each unit highly specific inputs without bloating the shared feature space, which is
what global gain-ranked pruning was destroying: pruning to a pooled top-15 cost Lio_G1 0.109 AUC
and Hogga_G1 0.098, while the micro-average hid it.

Stage 2 MUST train on out-of-fold stage-1 predictions. Fitted in-sample, stage-1 predictions are
near-perfect, stage 2 learns to trust them totally, and the whole thing collapses at inference.
`oof_stage1` cross-fits by case to avoid that.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
import features as F  # noqa: E402
from kaggle_metric import score as kaggle_score  # noqa: E402

STAGE1_PARAMS = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     multi_strategy="multi_output_tree", objective="binary:logistic",
                     eval_metric="logloss", n_jobs=-1, random_state=0)
STAGE2_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     objective="binary:logistic", eval_metric="logloss",
                     n_jobs=-1, random_state=0)


# ---------------------------------------------------------------- unit neighbourhoods

def _walk(conns, node, key_from, key_to, depth=4):
    seen, frontier = set(), {node}
    for _ in range(depth):
        nxt = set()
        for c in conns:
            if c[key_to] in frontier and c[key_from] not in seen:
                seen.add(c[key_from]); nxt.add(c[key_from])
        frontier = nxt
    return seen


def unit_neighbourhoods() -> dict[str, dict]:
    """For each plant: its own reservoir, upstream/downstream sets, and wired constraints."""
    topo = F.load_topology()
    conns = topo["connections"]
    p2r = F.plant_to_reservoir()
    mv_res = [c for c in pd.read_csv(_ROOT / "data/extended/Constraint_min_volume.csv", nrows=1)
              .columns if c != "date"]
    mf_riv = [c for c in pd.read_csv(_ROOT / "data/extended/Constraint_min_flow.csv", nrows=1)
              .columns if c != "date"]
    out = {}
    for plant in topo["model"]["plant"]:
        up = _walk(conns, plant, "from", "to")
        down = _walk(conns, plant, "to", "from")
        near = up | down
        out[plant] = {"own_reservoir": p2r.get(plant),
                      "upstream": sorted(r for r in up if r in set(mv_res) | set(p2r.values())),
                      "min_vol": [r for r in mv_res if r in near],
                      "min_flow": [r for r in mf_riv if r in near]}
    return out


def add_constraint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Join the two constraint files - two whole data files we had never used."""
    out = df.copy()
    for name, path in [("minvol", "data/extended/Constraint_min_volume.csv"),
                       ("minflow", "data/extended/Constraint_min_flow.csv")]:
        c = pd.read_csv(_ROOT / path)
        c["timestamp"] = pd.to_datetime(c["date"]).dt.tz_localize(None)
        c = c.drop(columns="date").rename(
            columns={k: f"{name}_{F.sanitize(k)}" for k in c.columns if k != "timestamp"})
        out = out.merge(c, on="timestamp", how="left")
    return out


def unit_feature_cols(df: pd.DataFrame, unit: str, nb: dict) -> list[str]:
    """Columns specific to one unit's hydraulic neighbourhood."""
    plant = unit.rsplit("_G", 1)[0]
    info = nb.get(plant) or nb.get(plant.replace("_", " ")) or {}
    pl = F.sanitize(plant)
    cols = []
    r = info.get("own_reservoir")
    if r:
        rs = F.sanitize(r)
        cols += [c for c in (f"storage_{rs}", f"water_value_{rs}", f"inflow_{rs}") if c in df.columns]
    # per-plant engineered columns already in the frame
    cols += [c for c in df.columns if c.endswith("_" + pl)]
    # the constraints actually wired to this plant
    for r_ in info.get("min_vol", []):
        cols += [c for c in df.columns if c == f"minvol_{F.sanitize(r_)}"]
    for r_ in info.get("min_flow", []):
        cols += [c for c in df.columns if c == f"minflow_{F.sanitize(r_)}"]
    seen, uniq = set(), []
    for c in cols:
        if c not in seen and df[c].nunique(dropna=False) > 1:
            seen.add(c); uniq.append(c)
    return uniq


# ---------------------------------------------------------------- stage 1 / stage 2

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def oof_stage1(df: pd.DataFrame, fit_idx, shared_cols, targets, n_folds=3, stride=2, verbose=True):
    """Out-of-fold stage-1 predictions for the training rows, cross-fit by case."""
    fit = df.loc[fit_idx]
    cases = np.sort(fit["run_no"].unique())
    folds = np.array_split(cases, n_folds)
    oof = pd.DataFrame(index=fit.index, columns=[f"p1_{t[3:]}" for t in targets], dtype=float)
    for k, hold in enumerate(folds):
        tr = fit[~fit["run_no"].isin(hold)]
        tr = tr[tr["hour"] % stride == 0]
        te = fit[fit["run_no"].isin(hold)]
        m = XGBClassifier(**STAGE1_PARAMS)
        m.fit(tr[shared_cols], tr[targets].astype(int))
        oof.loc[te.index, :] = m.predict_proba(te[shared_cols])
        if verbose:
            print(f"    oof fold {k+1}/{n_folds} done", flush=True)
    return oof


def run_stacked(df, fit_years, val_year, shared_cols, n_folds=3, stride=2, verbose=True):
    df = df.reset_index(drop=True)
    targets = [c for c in df.columns if c.startswith("on_")]
    units = [t[3:] for t in targets]
    nb = unit_neighbourhoods()

    fit_idx = df.index[df["year"].isin(fit_years)]
    val_idx = df.index[df["year"] == val_year]

    if verbose:
        print("  stage 1: out-of-fold predictions on training rows", flush=True)
    oof = oof_stage1(df, fit_idx, shared_cols, targets, n_folds, stride, verbose)

    if verbose:
        print("  stage 1: final fit -> validation predictions", flush=True)
    tr = df.loc[fit_idx]
    tr_s = tr[tr["hour"] % stride == 0]
    m1 = XGBClassifier(**STAGE1_PARAMS)
    m1.fit(tr_s[shared_cols], tr_s[targets].astype(int))
    val_p1 = pd.DataFrame(m1.predict_proba(df.loc[val_idx, shared_cols]),
                          index=val_idx, columns=[f"p1_{u}" for u in units])

    baseline = {u: roc_auc_score(df.loc[val_idx, f"on_{u}"], val_p1[f"p1_{u}"]) for u in units}

    if verbose:
        print("  stage 2: per-unit residual models", flush=True)
    final = {}
    for u in units:
        own, others = f"p1_{u}", [f"p1_{v}" for v in units if v != u]
        ufc = unit_feature_cols(df, u, nb)
        Xtr = pd.concat([oof[others], df.loc[fit_idx, ufc]], axis=1)
        Xva = pd.concat([val_p1[others], df.loc[val_idx, ufc]], axis=1)
        m2 = XGBClassifier(**STAGE2_PARAMS)
        m2.fit(Xtr, df.loc[fit_idx, f"on_{u}"].astype(int),
               base_margin=_logit(oof[own].to_numpy(dtype=float)))
        m2.set_params(base_margin=None)
        p = m2.predict_proba(Xva, base_margin=_logit(val_p1[own].to_numpy(dtype=float)))[:, 1]
        final[u] = p
        if verbose:
            print(f"    {u:<16} {len(ufc):>3} unit feats | "
                  f"stage1 {baseline[u]:.4f} -> stacked {roc_auc_score(df.loc[val_idx, f'on_{u}'], p):.4f}",
                  flush=True)

    pred = pd.DataFrame(final, index=val_idx)
    return baseline, pred, df.loc[val_idx]


def micro_auc(df_val, pred, units):
    y = df_val[[f"on_{u}" for u in units]].to_numpy()
    p = pred[units].to_numpy()
    return roc_auc_score(y, p, average="micro")
