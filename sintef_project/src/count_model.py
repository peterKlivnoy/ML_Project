"""
Count model, v3 -- the (case, unit) model that predicts hours-on per unit-week.

This module exists because the architecture that produced 0.9402 lived entirely in throwaway
scripts. It also fixes four structural problems diagnosed in log 029:

  A. WIDTH.  The v2 assembly merged every plant's columns back in under `abs_`/`wvt_`/`c_`
     prefixes after unit-relative resolution had already happened. Result: 437 features of
     which only 60 varied with the unit, and 72.6% of model importance sat on the 377 that
     were IDENTICAL across all 14 units of a case. A shared feature cannot discriminate
     between units within a case, which is most of what micro-AUC rewards.
     Fix: resolve the per-plant blocks into own/upstream/downstream, and convert the
     remaining cross-plant information into the unit's RANK and Z-SCORE among plants --
     which is unit-specific and scale-free instead of shared and drifting.

  B. HEAD.  "mixed" weeks are bimodal (for most units only 17-24% of mixed weeks fall in the
     middle third; they cluster near-off and near-on). Fitting one regressor to that returns
     the mean, which lands in the sparse middle. Fix: classify over count BINS and carry the
     whole predicted distribution forward instead of collapsing to a point estimate.

  C. STATIONARITY.  14.1% of importance sat on features drifting >0.5 SD between 2015-18 and
     2019-22 -- all of them absolute price / water-value levels (the 2021-22 energy crisis).
     Test is 2023-24, on the far side of that. Fix: split level features into a stationary
     shape (ratio to trailing reference) plus an explicitly quarantined level.

  D. GROUPS.  Unit count distributions sort by merit order within a plant. Structural group
     attributes let similar units share statistical strength.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import production_features as PF
import unit_relative as UR

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
san = PF.san

# Count bins for the distributional head (B). Edges are inclusive-low, exclusive-high,
# except the two degenerate endpoints which are exact: 0 hours and 168 hours.
BIN_EDGES = [0, 1, 29, 85, 141, 168]
BIN_NAMES = ["off", "vlow", "low", "high", "vhigh", "on"]


def bin_count(count: pd.Series) -> pd.Series:
    """Map hours-on (0..168) to one of six ordered bins."""
    c = count.to_numpy()
    b = np.full(len(c), 2, dtype=int)
    b[c == 0] = 0
    b[(c >= 1) & (c < 29)] = 1
    b[(c >= 29) & (c < 85)] = 2
    b[(c >= 85) & (c < 141)] = 3
    b[(c >= 141) & (c < 168)] = 4
    b[c == 168] = 5
    return pd.Series(b, index=count.index)


def distribution_outputs(Q: np.ndarray, bin_means: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Forward the WHOLE predicted count distribution, not a four-number summary.

    The pipeline has been collapsing six bin probabilities into `p_off`, `p_on`,
    `p_low = b1+b2`, `p_high = b3+b4` and `exp_count`. That summary was designed for the WIDE
    model, where every forwarded number cost 14 columns and compression was worth real accuracy.
    In long format a forwarded number costs ONE column, on the row it belongs to, so the
    compression buys nothing and discards the shape it was invented to preserve -- which is the
    thing that made the binned head work in the first place (log 030).

    Emits all six bin probabilities plus the distribution's own summary statistics: expected
    count, its standard deviation (how sure the model is, distinct from where it points),
    entropy (flatness), the modal bin, and two cumulative masses at the decision-relevant
    thresholds of a quarter and a half of the week.
    """
    eps = 1e-9
    m = np.nan_to_num(bin_means)
    exp = Q @ m
    var = np.clip((Q @ (m ** 2)) - exp ** 2, 0, None)
    ent = -(Q * np.log(np.clip(Q, eps, 1.0))).sum(axis=1)
    cols = [Q[:, i] for i in range(6)] + [
        exp, np.sqrt(var), ent, Q.argmax(axis=1).astype(float),
        Q[:, :2].sum(axis=1),           # P(count < 29h)  -- barely on or off
        Q[:, 3:].sum(axis=1),           # P(count >= 85h) -- more than half the week
    ]
    # bins 0 and 5 are the exact all-off and all-on endpoints, so they keep the names the rest
    # of the pipeline already uses (`cross_unit_context` and `neighbour_context` look for
    # `p_off`/`p_on`); only the interior bins are new.
    names = ["p_off", "p_bin1", "p_bin2", "p_bin3", "p_bin4", "p_on"] + [
        "exp_count", "count_sd", "entropy", "mode_bin", "p_lt_quarter", "p_ge_half"]
    return np.column_stack(cols), names


def plant_list(prod_cols: list[str], units: list[str]) -> list[str]:
    """Sanitised plant names, longest first so `Vesle_Kjela` wins over `Kjela`."""
    return sorted({u.rsplit("_G", 1)[0] for u in units}, key=len, reverse=True)


def per_plant_metrics(prod: pd.DataFrame, plants: list[str]) -> list[str]:
    """Metrics present for EVERY plant -- only these can be ranked across plants."""
    by_metric: dict[str, set[str]] = {}
    for c in prod.columns:
        for p in plants:
            if c.endswith("_" + p):
                by_metric.setdefault(c[: -(len(p) + 1)], set()).add(p)
                break
    return sorted(m for m, ps in by_metric.items() if len(ps) == len(plants))


def cross_plant_position(prod: pd.DataFrame, plants: list[str], metrics: list[str],
                         plant_of_unit: str) -> pd.DataFrame:
    """(A + C) Replace 149 shared per-plant columns with the unit's POSITION among plants.

    For each metric this yields, per case, where this unit's plant sits in the cross-sectional
    distribution: its rank (0-1), its z-score, and its ratio to the system total. All three are
    unit-specific (they differ across the 14 rows of a case) and scale-free (they do not drift
    with the price level), which is exactly what the shared absolute columns were not.
    """
    out = {}
    for m in metrics:
        cols = [f"{m}_{p}" for p in plants]
        M = prod[cols].to_numpy(dtype=float)
        own = prod[f"{m}_{plant_of_unit}"].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            rank = (M < own[:, None]).sum(axis=1) / max(len(plants) - 1, 1)
            mu = np.nanmean(M, axis=1)
            sd = np.nanstd(M, axis=1)
            tot = np.nansum(np.abs(M), axis=1)
            out[f"pos_rank_{m}"] = rank
            out[f"pos_z_{m}"] = np.where(sd > 0, (own - mu) / sd, 0.0)
            out[f"pos_share_{m}"] = np.where(tot > 0, own / tot, 0.0)
    return pd.DataFrame(out, index=prod.index)


def resolve_wvt(wvt: pd.DataFrame, plants: list[str], plant_of_unit: str,
                upstream: list[str], downstream: str | None) -> pd.DataFrame:
    """(A) Water-value-trace block resolved to own / upstream / downstream, not all 9 plants."""
    metrics = per_plant_metrics(wvt, plants)
    out = {}
    for m in metrics:
        own_c = f"{m}_{plant_of_unit}"
        if own_c in wvt.columns:
            out[f"own_{m}"] = wvt[own_c].to_numpy(dtype=float)
        ups = [f"{m}_{q}" for q in upstream if f"{m}_{q}" in wvt.columns]
        out[f"upstream_{m}"] = (wvt[ups].mean(axis=1).to_numpy(dtype=float) if ups
                                else np.zeros(len(wvt)))
        if downstream and f"{m}_{downstream}" in wvt.columns:
            out[f"downstream_{m}"] = wvt[f"{m}_{downstream}"].to_numpy(dtype=float)
        else:
            out[f"downstream_{m}"] = np.zeros(len(wvt))
    for c in wvt.columns:
        if c.startswith("wvt_sys_"):
            out[c] = wvt[c].to_numpy(dtype=float)
    return pd.DataFrame(out, index=wvt.index)


def downstream_plant_map(units: list[str]) -> dict[str, str | None]:
    """The plant each plant discharges into -- missing from v2 entirely."""
    rel = UR.relevance_map()
    src, dst, _ = PF.plant_graph()
    out = {}
    for u in units:
        pl_raw = next((p for p in rel if san(p) == u.rsplit("_G", 1)[0]), None)
        tgt = dst.get(pl_raw) if pl_raw else None
        down = next((san(q) for q, s in src.items() if s == tgt and q != pl_raw), None)
        out[u] = down
    return out


# --------------------------------------------------------------------------------------- C
# Level features whose drift between 2015-18 and 2019-22 exceeded 0.5 SD (log 029 screen).
# Each is split into a stationary shape and a quarantined level.
_LEVEL_PREFIXES = ("price_", "wv_", "own_wv", "term_spread", "spread_")


def stationarise(df: pd.DataFrame, cols: list[str], ref: str = "price_back_1y",
                 wv_ref: str = "wv_back_1y") -> tuple[pd.DataFrame, list[str]]:
    """(C) Split drifting LEVEL features into scale-free shape + explicit level.

    A price in 2016 and a price in 2022 are not the same quantity -- the 2021-22 crisis moved
    the level by most of a standard deviation. But `price_week / price_back_1y` means the same
    thing in both regimes. We emit the ratio (stationary, safe to lean on) and keep the level
    under a `lvl_` prefix so an ablation can drop the whole block at once.
    """
    out, made = {}, []
    for c in cols:
        if c in (ref, wv_ref):
            continue
        base = wv_ref if c.startswith(("wv_", "own_wv")) else ref
        if base not in df.columns:
            continue
        denom = df[base].to_numpy(dtype=float)
        v = df[c].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            r = np.where(np.abs(denom) > 1e-9, v / denom, np.nan)
        out[f"st_{c}"] = r
        made.append(f"st_{c}")
    return pd.DataFrame(out, index=df.index), made


def drifting_levels(df: pd.DataFrame, feats: list[str], year_col: str = "year",
                    split: int = 2019, thresh: float = 0.5) -> list[str]:
    """Features whose mean shifts by more than `thresh` SD across the period split."""
    early = df[df[year_col] < split]
    late = df[df[year_col] >= split]
    out = []
    for c in feats:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        sd = float(df[c].std())
        if not np.isfinite(sd) or sd == 0:
            continue
        if abs(float(late[c].mean()) - float(early[c].mean())) / sd > thresh:
            out.append(c)
    return out


# --------------------------------------------------------------------------------------- D
def group_attributes(units: list[str]) -> pd.DataFrame:
    """(D) Structural (NOT target-derived) group keys.

    Deliberately excludes anything computed from historical on/off labels: those stop at
    2022-12-31 while the test runs to 2024-12-25, so a label-derived per-unit attribute is
    stale by 12-24 months and near-constant across the test set -- the mechanism that made the
    historical marginals backfire on Haukeli (log 029).
    """
    rel = UR.relevance_map()
    src, dst, up = PF.plant_graph()
    rows = []
    for u in units:
        pl = u.rsplit("_G", 1)[0]
        pl_raw = next((p for p in rel if san(p) == pl), None)
        info = rel.get(pl_raw, {}) if pl_raw else {}
        n_sib = sum(1 for v in units if v.rsplit("_G", 1)[0] == pl)
        rows.append({
            "unit": u,
            "grp_n_siblings": n_sib,
            "grp_is_solo": int(n_sib == 1),
            "grp_n_upstream": len(info.get("upstream_plants", [])),
            "grp_has_downstream": int(bool(dst.get(pl_raw))) if pl_raw else 0,
            "grp_n_constraints": len(info.get("min_vol", [])) + len(info.get("min_flow", [])),
        })
    return pd.DataFrame(rows)


def redundancy_prune(df: pd.DataFrame, feats: list[str], importance: pd.Series,
                     thresh: float = 0.95, sample: int = 8000,
                     random_state: int = 0) -> list[str]:
    """Drop features that duplicate one already kept, regardless of which block they came from.

    The v3 experiment cut 437 -> 230 by removing a whole FAMILY (the absolute per-plant block)
    and lost 0.0285 for it. That was the wrong axis: the absolute block carries real case-level
    information, it is just expressed several times over. `abs_release_Tokke`,
    `abs_energy_mwh_Tokke` and `abs_implied_hours_Tokke` are the same quantity through three
    unit conversions, and the redundancy runs ACROSS blocks as much as within them.

    Greedy by importance: walk features best-first, keep one, and drop anything correlating above
    `thresh` with something already kept. The survivor of each cluster is its most useful member,
    so information is preserved where block removal destroyed it.
    """
    cols = [c for c in feats if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    sub = df[cols]
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=random_state)
    corr = sub.corr().abs().fillna(0.0)
    order = importance.reindex(cols).fillna(0.0).sort_values(ascending=False).index
    kept: list[str] = []
    for c in order:
        if all(corr.loc[c, k] <= thresh for k in kept):
            kept.append(c)
    return kept


def build_v3(CU: pd.DataFrame, PROD: pd.DataFrame, CONS: pd.DataFrame, WVT: pd.DataFrame,
             units: list[str], with_position: bool = True,
             with_downstream: bool = True) -> pd.DataFrame:
    """(A) Unit-relative assembly with NO shared per-plant blocks merged back in."""
    base = UR.build_unit_relative(CU, PROD, CONS, units)
    plants = plant_list(list(PROD.columns), units)
    metrics = per_plant_metrics(PROD, plants)
    rel = UR.relevance_map()
    down_map = downstream_plant_map(units)

    parts = []
    for u in units:
        pl = u.rsplit("_G", 1)[0]
        pl_raw = next((p for p in rel if san(p) == pl), None)
        info = rel.get(pl_raw, {}) if pl_raw else {}
        ups = [san(q) for q in info.get("upstream_plants", [])]
        down = down_map.get(u)

        sub = base[base.unit == u].reset_index(drop=True)
        prod = PROD.reindex(sub["run_no"]).reset_index(drop=True)
        wvt = WVT.reindex(sub["run_no"]).reset_index(drop=True)

        blocks = [sub]
        if with_position:
            blocks.append(cross_plant_position(prod, plants, metrics, pl))
        if with_downstream and down:
            dcols = [f"{m}_{down}" for m in metrics if f"{m}_{down}" in prod.columns]
            d = prod[dcols].copy()
            d.columns = [f"downstream_{c[: -(len(down) + 1)]}" for c in dcols]
            blocks.append(d)
        elif with_downstream:
            blocks.append(pd.DataFrame(
                {f"downstream_{m}": np.zeros(len(sub)) for m in metrics}, index=sub.index))
        blocks.append(resolve_wvt(wvt, plant_list(list(WVT.columns), units), pl, ups, down))
        parts.append(pd.concat(blocks, axis=1))

    out = pd.concat(parts, ignore_index=True)
    return out.merge(group_attributes(units), on="unit", how="left")
