"""Build a Kaggle submission from the best validated configuration.

Architecture (log 032 onward, validated at 0.9638 +/- 0.0115 over six walk-forward folds):

    one model, one row per (case, hour, unit)
    no count model in the forward path -- it is fitted only to RANK features
    169 features: the count model's top-120 minus those measuring <=0, plus 30 own_/upstream_

Differences from every experiment run so far, each of which matters:

  TRAINS ON ALL 2,922 CASES. Every validation fold trained on 1,826-2,557. Training volume has
  been the most reliable driver in the project, so the submission model sees more data than any
  measurement we have taken of it.

  PREDICTS 24 MONTHS AHEAD. Folds predicted at most six. The test period is 2023-01-01 to
  2024-12-25, strictly after all training data, across the regime break that moved Haukeli's base
  rate from 0.87 to 0.14 and broke Lio mid-2021. Walk-forward cannot measure that, so 0.9638 is
  an upper bound rather than an estimate.

  FEATURES ARE BUILT ON TRAIN AND TEST TOGETHER, then split. The builders read the raw files
  directly and those cover the test window (volume and water value to 2025-01-01, price to
  2025-12-31), so post-decision traces exist for every test case. The one boundary case is the
  final week: 2024-12-25 needs a terminal water value at +7 days = 2025-01-01, which is the last
  row of the file. The off-by-one fix in log 033 is what makes that land inside the data; at the
  old +8 days it would have fallen off the end.

Output is wide: one row per case, `result_committed_<Plant>_<Unit>_t<hour>`, with the plant name
spelled as the topology spells it (`Vesle Kjela`, with a space) rather than sanitised.
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import count_features as CF
import count_model as CM
import features as F
import hourly_structure as HS
import longform as LF
import production_features as PF
import unit_relative as UR
from modeling.pipeline import STAGES

COUNT_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.6, tree_method="hist", n_jobs=-1, random_state=0)
POINT_PARAMS = dict(max_depth=8, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=500, tree_method="hist", max_bin=256,
                    objective="binary:logistic", eval_metric="logloss", nthread=-1, seed=0)
POINT_ROUNDS = 600
N_CORE, N_UNITREL = 120, 30


def submission_columns(units: list[str]) -> dict[str, str]:
    """Map sanitised unit -> the generator name the submission file uses.

    `sanitize` turns "Vesle Kjela_G1" into "Vesle_Kjela_G1", and the submission wants the
    original. Inverting by replacing underscores would also corrupt the `_G1` suffix, so the
    mapping is taken from the topology rather than reconstructed.
    """
    gens = F.load_topology()["model"]["generator"]
    return {F.sanitize(g): g for g in gens if F.sanitize(g) in set(units)}


def build_features(verbose=True) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Feature stages over train and test together, then split back.

    Built jointly because several stages compare a case against the whole record (seasonal
    percentiles, trailing windows); computing them on train alone and test alone would put the
    two on different scales.
    """
    tr = pd.read_parquet(_ROOT / "data/processed/train_hourly.parquet")
    te = pd.read_parquet(_ROOT / "data/processed/test_hourly.parquet")
    targets = [c for c in tr.columns if c.startswith("on_")]
    units = [t[3:] for t in targets]
    te = te.assign(**{c: np.nan for c in targets})
    both = pd.concat([tr, te], ignore_index=True)
    both["year"] = both["starttime"].dt.year
    both["_is_test"] = np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)]
    for name, fn in STAGES:
        t0 = time.time()
        out = fn(both)
        new = [c for c in out.columns if c not in both.columns]
        both = pd.concat([both, out[new]], axis=1)
        if verbose:
            print(f"  {name:<22} +{len(new):>3} cols  ({time.time()-t0:.1f}s)", flush=True)
    m = both["_is_test"].to_numpy()
    return both[~m].reset_index(drop=True), both[m].reset_index(drop=True), units


def build_count_table(df: pd.DataFrame, units: list[str]) -> pd.DataFrame:
    """The (case, unit) table the count model ranks features on."""
    dates = df.groupby("run_no")["starttime"].first()
    sys_ = CF.build_count_features(dates); sys_.index = dates.index
    prod = PF.build(dates); prod.index = dates.index
    cons = UR.add_constraint_series(dates); cons.index = dates.index
    wvt = PF.water_value_trace(dates); wvt.index = dates.index

    rows = []
    g = df.groupby("run_no")
    for u in units:
        pl = u.rsplit("_G", 1)[0]
        col = f"f06_spread_{pl}" if f"f06_spread_{pl}" in df.columns else "f06_spread_sys"
        gs = g[col]
        r = pd.DataFrame({"run_no": g["year"].first().index, "unit": u,
                          "spread_mean": gs.mean().values, "spread_std": gs.std().values,
                          "spread_max": gs.max().values, "spread_min": gs.min().values})
        cnt = g[f"on_{u}"].sum()
        r["count"] = cnt.values
        rows.append(r)
    cu = pd.concat(rows, ignore_index=True)
    return (UR.build_unit_relative(cu.merge(sys_, left_on="run_no", right_index=True, how="left"),
                                   prod, cons, units)
            .merge(prod.add_prefix("abs_"), left_on="run_no", right_index=True, how="left")
            .merge(cons.add_prefix("c_"), left_on="run_no", right_index=True, how="left")
            .merge(wvt, left_on="run_no", right_index=True, how="left")
            .replace([np.inf, -np.inf], np.nan))


def run(stride: int = 1, out_path: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Fit on all training cases, predict the 725 test cases, write the wide submission."""
    from xgboost import QuantileDMatrix, XGBClassifier, train as xgb_train

    t0 = time.time()
    tr, te, units = build_features(verbose=verbose)
    targets = [f"on_{u}" for u in units]
    print(f"features built: train {tr.shape}, test {te.shape} | {time.time()-t0:.0f}s", flush=True)

    both = pd.concat([tr, te], ignore_index=True)
    ct = build_count_table(both, units)
    ct["bin"] = CM.bin_count(ct["count"].fillna(0))
    feats = [c for c in ct.columns if c not in ("run_no", "unit", "count", "cls", "bin")]
    train_runs = set(tr["run_no"].unique())
    ct_tr = ct[ct.run_no.isin(train_runs) & ct["count"].notna()]
    print(f"count table {ct.shape} | ranking on {len(ct_tr):,} train unit-weeks "
          f"| {time.time()-t0:.0f}s", flush=True)

    # the count model is fitted ONLY to rank features; its predictions are never forwarded
    seen = sorted(ct_tr["bin"].unique())
    cm = XGBClassifier(objective="multi:softprob", num_class=len(seen),
                       eval_metric="mlogloss", **COUNT_PARAMS)
    cm.fit(ct_tr[feats], ct_tr["bin"].map({b: i for i, b in enumerate(seen)}))
    imp = pd.Series(cm.feature_importances_, index=feats).sort_values(ascending=False)
    keep = list(imp.head(N_CORE).index)
    keep += [c for c in imp.index[N_CORE:] if c.startswith(("own_", "upstream_"))][:N_UNITREL]
    print(f"selected {len(keep)} count features | {time.time()-t0:.0f}s", flush=True)

    hs_tr = HS.build(tr); tr = pd.concat([tr, hs_tr], axis=1)
    hs_te = HS.build(te); te = pd.concat([te, hs_te], axis=1)
    shared = [c for c in hs_tr.columns]
    pruned = pd.read_csv(_ROOT / "data/processed/pruned_features.csv")["feature"].tolist()
    tw = [c for c in tr.columns if c.startswith("f07_term_spread_")
          or c in ("f07_terminal_slope", "f07_terminal_wv_lag8_minus_d7")]
    shared = [c for c in pruned + tw + shared if c in tr.columns and c in te.columns]

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

    for d in (tr, te):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
    ctx = ct.set_index(["run_no", "unit"])[keep]

    def to_long(d, is_train, cs):
        L = LF.to_long(d, units, shared, static=stat, case_stride=cs)
        j = ctx.reindex(pd.MultiIndex.from_arrays([L["run_no"], L["unit"].astype(str)]))
        blk = pd.DataFrame(j.to_numpy(dtype=np.float32),
                           columns=[f"cf_{c}" for c in keep], index=L.index)
        return pd.concat([L, blk], axis=1)

    L = to_long(tr, True, stride)
    V = to_long(te, False, 1)
    cols = [c for c in L.columns if c not in ("run_no", "hour", "y")]
    print(f"long: train {len(L):,} rows, test {len(V):,} rows, {len(cols)} features "
          f"| {time.time()-t0:.0f}s", flush=True)

    dtr = QuantileDMatrix(L[cols], label=L["y"], enable_categorical=True,
                          max_bin=POINT_PARAMS["max_bin"])
    bst = xgb_train(POINT_PARAMS, dtr, num_boost_round=POINT_ROUNDS)
    print(f"fitted {POINT_ROUNDS} rounds | {time.time()-t0:.0f}s", flush=True)
    dva = QuantileDMatrix(V[cols], enable_categorical=True, ref=dtr,
                          max_bin=POINT_PARAMS["max_bin"])
    V["p"] = bst.predict(dva)

    # wide: one row per case, `result_committed_<Plant>_<Unit>_t<hour>`
    name = submission_columns(units)
    wide = V.pivot_table(index="run_no", columns=["unit", "hour"], values="p", observed=True)
    # Emit columns in the SAMPLE's order, not ours. The sample orders units as the topology
    # does (Hogga first, Vesle Kjela last); `units` comes from the parquet, which is
    # alphabetical. Building from the sample's header makes the order correct by construction
    # instead of by coincidence.
    sample = pd.read_csv(_ROOT / "sample_submission.csv")
    inv = {v: k for k, v in name.items()}
    cols = {}
    for c in sample.columns:
        if c == "Run No":
            continue
        m = re.fullmatch(r"result_committed_(.+)_t(\d+)", c)
        gen, h = m.group(1), int(m.group(2))
        cols[c] = wide[(inv[gen], h)].to_numpy()
    out = pd.DataFrame(cols, index=wide.index)
    out.index.name = "Run No"
    out = out.reset_index()

    assert list(out.columns) == list(sample.columns), "column mismatch against sample_submission"
    assert len(out) == len(sample) and set(out["Run No"]) == set(sample["Run No"])
    assert out.drop(columns="Run No").isna().sum().sum() == 0, "NaNs in predictions"
    if out_path:
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path} | {out.shape} | {time.time()-t0:.0f}s", flush=True)
    return out
