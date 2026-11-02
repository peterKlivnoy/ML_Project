"""
Feature builders for the unit-commitment task.

Each builder takes the modelling table (one row per case-hour) and returns a DataFrame of new
columns, so stages compose cumulatively in the iteration notebook. The same functions build
test-set features for submission, which is why they live here rather than in a notebook.

Two invariants every builder must respect:

1. **Causality is a choice here, not a requirement.** The competition withholds only test-period
   *targets*, so feature-side statistics may be computed over the whole dataset. A global
   statistic keeps CV valid provided it is computed identically for train and test rows - the
   reference distribution is then symmetric. Builders come in both flavours where it matters
   (`stage_seasonal` expanding vs. `stage_seasonal_global`) so the folds can decide. What is
   *never* usable is anything derived from past targets: the test period has no labels at all.

2. **Row-layout compatibility.** Rows are (case, hour) with 14 target columns, so a feature must
   be a property of the row. A static per-unit constant (e.g. `startcost_const`) becomes a
   constant column here and carries zero information; per-unit attributes are only useful in
   long format, or combined with time-varying quantities into per-plant columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = _ROOT / "data/extended/Tokke_Vinje_topology.yaml"

# ---------------------------------------------------------------- topology helpers

_topo_cache = None


def load_topology() -> dict:
    global _topo_cache
    if _topo_cache is None:
        with open(TOPOLOGY) as f:
            _topo_cache = yaml.safe_load(f)
    return _topo_cache


def plant_to_reservoir() -> dict[str, str]:
    """Direct reservoir -> plant edges. Songa has none (it is fed via two tunnels)."""
    topo = load_topology()
    return {c["to"]: c["from"] for c in topo["connections"]
            if c["to_type"] == "plant" and c["from_type"] == "reservoir"}


def unit_to_plant(units: list[str]) -> dict[str, str]:
    return {u: u.rsplit("_G", 1)[0] for u in units}


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


# ---------------------------------------------------------------- point-in-time primitives

def expanding_seasonal_percentile(series: pd.Series, at_dates: pd.Series,
                                   window_days: int = 7) -> np.ndarray:
    """Percentile of each value against the same time of year in *strictly earlier* years.

    For each target date d, compares series[d] against all historical observations whose
    day-of-year is within +/- `window_days` of d AND whose date is before d. Returns NaN where
    fewer than 5 prior observations exist (the first year or so).

    The expanding restriction is what makes this safe: a 2019 case is never ranked against 2020+.
    """
    s = series.sort_index().dropna()
    idx_doy = s.index.dayofyear.to_numpy()
    idx_time = s.index.to_numpy()
    vals = s.to_numpy()

    out = np.full(len(at_dates), np.nan)
    for i, d in enumerate(pd.to_datetime(pd.Series(at_dates)).to_numpy()):
        d_ts = pd.Timestamp(d)
        doy = d_ts.dayofyear
        # circular day-of-year distance
        delta = np.abs(idx_doy - doy)
        delta = np.minimum(delta, 365 - delta)
        mask = (delta <= window_days) & (idx_time < d)
        if mask.sum() < 5:
            continue
        prior = vals[mask]
        cur = vals[idx_time == d]
        if len(cur) == 0:
            # value not in the series itself; fall back to the nearest earlier observation
            earlier = vals[idx_time < d]
            if len(earlier) == 0:
                continue
            cur = earlier[-1:]
        out[i] = (prior < cur[0]).mean()
    return out


def expanding_trailing_mean(series: pd.Series, at_dates: pd.Series, days: int) -> np.ndarray:
    """Mean of the series over the `days` before each target date (strictly backward-looking)."""
    s = series.sort_index().dropna()
    roll = s.rolling(f"{days}D", min_periods=2).mean()
    return roll.reindex(pd.to_datetime(pd.Series(at_dates)), method="ffill").to_numpy()


# ---------------------------------------------------------------- stage builders

def stage_seasonal(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 1: seasonal normalisation of the slow series (strongest EDA evidence, +0.537)."""
    kernel = _ROOT / "data/kernel"
    vol = pd.read_csv(kernel / "Historical_volume_2015_2024.csv")
    wv = pd.read_csv(kernel / "Synthetic_water_value_2015_2024.csv")
    inf = pd.read_csv(kernel / "Historical_inflow_1958_2025.csv")
    for d in (vol, wv, inf):
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    vol, wv = vol.set_index("date"), wv.set_index("date")
    inf_d = inf.set_index("date").resample("D").mean()

    res = list(vol.columns)
    sys_storage = vol[res].sum(axis=1)
    sys_wv = wv[res].mean(axis=1)
    sys_inflow = inf_d[[c for c in inf_d.columns if c in res]].sum(axis=1)

    case_day = df.groupby("run_no")["starttime"].first()
    feats = pd.DataFrame(index=case_day.index)
    feats["f01_storage_pct_doy"] = expanding_seasonal_percentile(sys_storage, case_day)
    feats["f01_wv_pct_doy"] = expanding_seasonal_percentile(sys_wv, case_day)
    feats["f01_inflow_pct_doy"] = expanding_seasonal_percentile(sys_inflow, case_day)
    # absolute anomaly as well as rank - trees may use them differently
    feats["f01_storage_anom"] = (
        sys_storage.reindex(case_day.values).to_numpy()
        - expanding_trailing_mean(sys_storage, case_day, 365)
    )
    return df[["run_no"]].merge(feats, left_on="run_no", right_index=True, how="left") \
                          .drop(columns="run_no").set_index(df.index)


def stage_inflow_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 9: wet-year / dry-year regime (+0.331 for the 365d trailing mean)."""
    inf = pd.read_csv(_ROOT / "data/kernel/Historical_inflow_1958_2025.csv")
    inf["date"] = pd.to_datetime(inf["date"]).dt.tz_localize(None)
    inf_d = inf.set_index("date").resample("D").mean()
    vol_cols = pd.read_csv(_ROOT / "data/kernel/Historical_volume_2015_2024.csv", nrows=1).columns
    res = [c for c in inf_d.columns if c in set(vol_cols)]
    sys_inflow = inf_d[res].sum(axis=1)

    case_day = df.groupby("run_no")["starttime"].first()
    feats = pd.DataFrame(index=case_day.index)
    for w in (30, 90, 365):
        feats[f"f09_inflow_trail_{w}d"] = expanding_trailing_mean(sys_inflow, case_day, w)
    feats["f09_inflow_30_over_365"] = (
        feats["f09_inflow_trail_30d"] / feats["f09_inflow_trail_365d"].replace(0, np.nan)
    )
    return df[["run_no"]].merge(feats, left_on="run_no", right_index=True, how="left") \
                          .drop(columns="run_no").set_index(df.index)


def stage_price_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 2: case-relative, scale-free price (+0.263 z-score, +0.252 percentile)."""
    g = df.groupby("run_no")["price_eur_mwh"]
    mean, std = g.transform("mean"), g.transform("std")
    p = df["price_eur_mwh"]
    out = pd.DataFrame(index=df.index)
    out["f02_price_z_case"] = (p - mean) / std.replace(0, np.nan)
    out["f02_price_pct_case"] = g.rank(pct=True)
    out["f02_price_log_ratio"] = np.log(p.clip(lower=0.1)) - np.log(mean.clip(lower=0.1))
    out["f02_price_case_std"] = std
    day = df["timestamp"].dt.floor("D")
    gd = df.groupby([df["run_no"], day])["price_eur_mwh"]
    out["f02_price_z_day"] = (p - gd.transform("mean")) / gd.transform("std").replace(0, np.nan)
    out["f02_price_rank_in_day"] = gd.rank()
    out["f02_price_minus_case_min"] = p - g.transform("min")
    out["f02_case_max_minus_price"] = g.transform("max") - p
    return out


def stage_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 6: price - water value spread. Raw, NOT normalised (+0.241 vs +0.114 z-scored)."""
    wv_cols = [c for c in df.columns if c.startswith("water_value_")]
    sys_wv = df[wv_cols].mean(axis=1)
    p2r = plant_to_reservoir()
    out = pd.DataFrame(index=df.index)
    out["f06_spread_sys"] = df["price_eur_mwh"] - sys_wv
    for plant, r in p2r.items():
        col = f"water_value_{sanitize(r)}"
        if col in df.columns:
            out[f"f06_spread_{sanitize(plant)}"] = df["price_eur_mwh"] - df[col]
    out["f06_spread_sys_positive"] = (out["f06_spread_sys"] > 0).astype(int)
    out["f06_spread_hours_positive_case"] = (
        out.groupby(df["run_no"])["f06_spread_sys_positive"].transform("sum")
    )
    return out


def stage_terminal_wv(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 7: terminal water value -- the boundary condition of SHOP's own objective.

    `Dataset_definitions_and_explanation.pdf`, under "Reservoir water value", specifies the
    scheduling problem exactly: for a case starting 2015-01-01 00:00 and ending 2015-01-07 23:00,
    the initial volume is the value at 2015-01-01 00:00 and "its final water value should be the
    value at 2015-01-08 00:00" -- i.e. **start + 7 days**.

    This previously read `days=8`, so every case priced its terminal water one day late. The
    quantity matters: SHOP maximises revenue plus the salvage value of water left in the
    reservoir, so the terminal water value IS the marginal cost of generating. A unit runs when
    price exceeds it. That makes `price - terminal_wv` the economically correct spread, and the
    contemporaneous daily water value -- which `f06_spread_*` uses -- the wrong quantity: the
    daily series is an input to the problem, not the price its decisions are taken against.

    Per-reservoir terminal values are emitted alongside the system mean, so each plant can be
    priced against its own reservoir rather than a basin-wide average.

    Verified unavailable for exactly one test case (2024-12-25); falls back to the last
    observation, which is recorded rather than silently NaN.
    """
    wv = pd.read_csv(_ROOT / "data/kernel/Synthetic_water_value_2015_2024.csv")
    wv["date"] = pd.to_datetime(wv["date"]).dt.tz_localize(None)
    wv = wv.set_index("date").sort_index()
    res = list(wv.columns)
    sys_wv = wv[res].mean(axis=1)

    case_day = df.groupby("run_no")["starttime"].first()
    horizon = case_day.values + pd.Timedelta(days=7)
    terminal = sys_wv.reindex(horizon)
    initial = sys_wv.reindex(case_day.values)
    term = terminal.to_numpy()
    n_missing = int(np.isnan(term).sum())
    if n_missing:
        term = pd.Series(term).ffill().to_numpy()  # documented fallback for the final case

    feats = pd.DataFrame(index=case_day.index)
    feats["f07_terminal_wv"] = term
    feats["f07_terminal_minus_initial"] = term - initial.to_numpy()
    feats["f07_terminal_over_initial"] = term / np.where(initial.to_numpy() == 0, np.nan,
                                                          initial.to_numpy())

    # The documented horizon is start+7, but every result up to log 032 was built on start+8,
    # and the water value series is smooth enough that the two are highly correlated. Rather
    # than assume the doc settles it empirically, emit the neighbouring lags as well and let
    # the model choose; the difference between them is itself the local slope of the water
    # value at the horizon, which is what "is water getting scarcer" means.
    for lag in (6, 8, 9):
        t_l = sys_wv.reindex(case_day.values + pd.Timedelta(days=lag)).to_numpy()
        if np.isnan(t_l).any():
            t_l = pd.Series(t_l).ffill().to_numpy()
        feats[f"f07_terminal_wv_lag{lag}"] = t_l
        feats[f"f07_terminal_wv_lag{lag}_minus_d7"] = t_l - term
    feats["f07_terminal_slope"] = feats["f07_terminal_wv_lag8"] - feats["f07_terminal_wv_lag6"]
    # per-reservoir terminal values: the marginal cost of water for the plant drawing on it
    p2r = plant_to_reservoir()
    for plant, r in p2r.items():
        col = sanitize(r)
        if r not in wv.columns:
            continue
        t_r = wv[r].reindex(horizon).to_numpy()
        i_r = wv[r].reindex(case_day.values).to_numpy()
        if np.isnan(t_r).any():
            t_r = pd.Series(t_r).ffill().to_numpy()
        feats[f"f07_terminal_wv_{col}"] = t_r
        feats[f"f07_terminal_minus_initial_{col}"] = t_r - i_r
    feats.attrs["n_terminal_missing"] = n_missing
    res_df = df[["run_no"]].merge(feats, left_on="run_no", right_index=True, how="left") \
                            .drop(columns="run_no").set_index(df.index)
    res_df["f07_price_minus_terminal_wv"] = df["price_eur_mwh"] - res_df["f07_terminal_wv"]
    # the economically correct per-plant spread: price against THIS plant's terminal water value
    for plant, r in p2r.items():
        col = sanitize(r)
        if f"f07_terminal_wv_{col}" in res_df.columns:
            res_df[f"f07_term_spread_{sanitize(plant)}"] = (
                df["price_eur_mwh"] - res_df[f"f07_terminal_wv_{col}"])
    return res_df


def stage_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Themes 10-12: head, energy conversion, start-cost economics, penstock structure.

    Static per-unit constants are useless in this row layout, so everything here is either a
    per-plant time-varying quantity or a combination of a static attribute with one.
    """
    topo = load_topology()
    gens, plants, reservoirs = topo["model"]["generator"], topo["model"]["plant"], topo["model"]["reservoir"]
    p2r = plant_to_reservoir()
    out = pd.DataFrame(index=df.index)

    for plant, r in p2r.items():
        scol = f"storage_{sanitize(r)}"
        if scol not in df.columns:
            continue
        spec = reservoirs[r]
        vh = spec["vol_head"]
        # storage -> reservoir level -> gross head above the plant's outlet
        level = np.interp(df[scol].to_numpy(), vh["x"], vh["y"])
        head = level - plants[plant]["outlet_line"]
        pl = sanitize(plant)
        out[f"f11_head_{pl}"] = head
        lo, hi = min(vh["y"]) - plants[plant]["outlet_line"], max(vh["y"]) - plants[plant]["outlet_line"]
        out[f"f11_head_frac_{pl}"] = (head - lo) / (hi - lo) if hi > lo else 0.0

        pgens = [g for g in gens if sanitize(g).startswith(pl + "_G")]
        if not pgens:
            continue
        g0 = gens[pgens[0]]
        # turbine efficiency interpolated across the head-referenced curves
        refs = np.array([c["ref"] for c in g0.get("turb_eff_curves", [])], dtype=float)
        peaks = np.array([max(c["y"]) for c in g0.get("turb_eff_curves", [])], dtype=float)
        if len(refs) >= 2:
            order = np.argsort(refs)
            out[f"f11_turb_eff_{pl}"] = np.interp(head, refs[order], peaks[order])
        else:
            out[f"f11_turb_eff_{pl}"] = peaks[0] if len(peaks) else np.nan
        # MWh per m3: rho*g*H*eta / 3.6e9, with eta as a fraction
        out[f"f11_mwh_per_m3_{pl}"] = (
            1000 * 9.81 * head * (out[f"f11_turb_eff_{pl}"] / 100.0) / 3.6e9
        )

        # start-cost economics: break-even run length, using the plant's own spread
        spread_col = f"f06_spread_{pl}"
        spread = df[spread_col] if spread_col in df.columns else None
        if spread is None:
            wvcol = f"water_value_{sanitize(r)}"
            spread = df["price_eur_mwh"] - df[wvcol] if wvcol in df.columns else None
        if spread is not None:
            p_max_total = sum(gens[g]["p_max"] for g in pgens)
            startcost = g0.get("startcost_const", np.nan)
            hourly_margin = spread * gens[pgens[0]]["p_max"]
            out[f"f10_breakeven_hours_{pl}"] = startcost / hourly_margin.where(hourly_margin > 0)
            out[f"f10_startcost_mwh_equiv_{pl}"] = startcost / df["price_eur_mwh"].clip(lower=0.1)
            out[f"f11_plant_energy_capacity_{pl}"] = p_max_total * out[f"f11_head_frac_{pl}"]

        # penstock structure: how many distinct penstocks the plant's units sit on
        pens = {gens[g].get("penstock", 1) for g in pgens}
        out[f"f12_n_penstocks_{pl}"] = len(pens)
        out[f"f12_units_per_penstock_{pl}"] = len(pgens) / max(len(pens), 1)

    return out


def drop_dead_columns(df: pd.DataFrame) -> list[str]:
    """Theme 8.7: columns with a single unique value carry zero information."""
    feature_like = [c for c in df.columns
                    if c.startswith(("storage_", "water_value_", "inflow_", "price_", "f0", "f1"))]
    return [c for c in feature_like if df[c].nunique(dropna=False) <= 1]


# ---------------------------------------------------------------- global (non-causal) variants

def global_seasonal_percentile(series: pd.Series, at_dates: pd.Series,
                                window_days: int = 7) -> np.ndarray:
    """Seasonal percentile against the WHOLE history, including later years.

    The competition withholds only test-period targets, so feature-side statistics computed over
    the full dataset are permitted. This stays valid for CV because the reference distribution is
    identical for train and test rows. It is strictly more informative than the expanding version
    in early years, where little prior history exists.
    """
    s = series.sort_index().dropna()
    idx_doy = s.index.dayofyear.to_numpy()
    vals = s.to_numpy()
    idx_time = s.index.to_numpy()

    out = np.full(len(at_dates), np.nan)
    for i, d in enumerate(pd.to_datetime(pd.Series(at_dates)).to_numpy()):
        doy = pd.Timestamp(d).dayofyear
        delta = np.abs(idx_doy - doy)
        delta = np.minimum(delta, 365 - delta)
        prior = vals[delta <= window_days]
        if len(prior) < 5:
            continue
        cur = vals[idx_time == d]
        if len(cur) == 0:
            earlier = vals[idx_time < d]
            if len(earlier) == 0:
                continue
            cur = earlier[-1:]
        out[i] = (prior < cur[0]).mean()
    return out


def stage_seasonal_global(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 1, global variant. A/B against stage_seasonal (expanding)."""
    kernel = _ROOT / "data/kernel"
    vol = pd.read_csv(kernel / "Historical_volume_2015_2024.csv")
    wv = pd.read_csv(kernel / "Synthetic_water_value_2015_2024.csv")
    inf = pd.read_csv(kernel / "Historical_inflow_1958_2025.csv")
    for d in (vol, wv, inf):
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    vol, wv = vol.set_index("date"), wv.set_index("date")
    inf_d = inf.set_index("date").resample("D").mean()
    res = list(vol.columns)

    case_day = df.groupby("run_no")["starttime"].first()
    feats = pd.DataFrame(index=case_day.index)
    feats["f01g_storage_pct_doy"] = global_seasonal_percentile(vol[res].sum(axis=1), case_day)
    feats["f01g_wv_pct_doy"] = global_seasonal_percentile(wv[res].mean(axis=1), case_day)
    feats["f01g_inflow_pct_doy"] = global_seasonal_percentile(
        inf_d[[c for c in inf_d.columns if c in res]].sum(axis=1), case_day)
    return df[["run_no"]].merge(feats, left_on="run_no", right_index=True, how="left") \
                          .drop(columns="run_no").set_index(df.index)


# ---------------------------------------------------------------- what the optimizer computes

def stage_water_budget(df: pd.DataFrame) -> pd.DataFrame:
    """The quantity SHOP actually resolves: available water vs. profitable hours.

    SHOP maximises revenue - start/stop costs + terminal water value. The binding tension is that
    a week contains more profitable hours than the water supports, so it must ration. We give the
    model that trade-off directly:

      energy available  = (usable storage + in-horizon inflow) x MWh/m3   [head-derived]
      hours affordable  = energy available / plant capacity
      merit dispatch    = price_rank_in_case vs. hours affordable

    `merit_margin` is effectively the greedy solution to the optimizer's own problem: run the
    N most expensive hours the water can pay for. Its sign is a direct run/don't-run signal.

    Requires stage_physics to have run (uses the f11_mwh_per_m3_* conversion factors).
    """
    topo = load_topology()
    gens, reservoirs = topo["model"]["generator"], topo["model"]["reservoir"]
    p2r = plant_to_reservoir()
    out = pd.DataFrame(index=df.index)

    # in-horizon inflow: total arriving over each case's own 168h window, per reservoir
    inflow_cols = {c[len("inflow_"):]: c for c in df.columns if c.startswith("inflow_")}
    case_inflow = df.groupby("run_no")[list(inflow_cols.values())].sum()  # m3/s summed over 168h

    # price rank within the case, 1 = most expensive hour of the week
    price_rank = df.groupby("run_no")["price_eur_mwh"].rank(ascending=False, method="first")
    out["f20_price_rank_desc"] = price_rank

    for plant, r in p2r.items():
        pl = sanitize(plant)
        scol, conv_col = f"storage_{sanitize(r)}", f"f11_mwh_per_m3_{pl}"
        if scol not in df.columns or conv_col not in df.columns:
            continue
        spec = reservoirs[r]
        vh = spec["vol_head"]
        lrl_vol = float(np.interp(spec["lrl"], vh["y"], vh["x"]))  # dead storage below lrl

        usable_Mm3 = (df[scol] - lrl_vol).clip(lower=0)
        icol = inflow_cols.get(sanitize(r))
        inflow_Mm3 = (df["run_no"].map(case_inflow[icol]) * 3600 / 1e6) if icol else 0.0

        energy_mwh = (usable_Mm3 + inflow_Mm3) * 1e6 * df[conv_col]
        pgens = [g for g in gens if sanitize(g).startswith(pl + "_G")]
        capacity_mw = sum(gens[g]["p_max"] for g in pgens) or np.nan

        hours_affordable = energy_mwh / capacity_mw
        out[f"f20_energy_avail_{pl}"] = energy_mwh
        out[f"f20_hours_affordable_{pl}"] = hours_affordable
        # the greedy dispatch signal: is this hour inside the affordable top-N of the week?
        out[f"f20_merit_margin_{pl}"] = hours_affordable.clip(upper=168) - price_rank
        out[f"f20_in_merit_{pl}"] = (out[f"f20_merit_margin_{pl}"] > 0).astype(int)
        # how tight the week is: water supply relative to the whole horizon
        out[f"f20_scarcity_{pl}"] = (hours_affordable / 168.0).clip(upper=5)

    return out


# ---------------------------------------------------------------- round 2 builders
# Prioritised by log 013: hourly-varying features hold 96.4% of model gain, and encoded
# decision rules beat statistical transforms. Case-level scalars are deprioritised.

def stage_spread_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 6 extensions. The spread family supplied 60.8% of model gain, so push on it."""
    p2r = plant_to_reservoir()
    gens = load_topology()["model"]["generator"]
    wv_cols = [c for c in df.columns if c.startswith("water_value_")]
    sys_wv = df[wv_cols].mean(axis=1)
    g = df.groupby("run_no")
    out = pd.DataFrame(index=df.index)

    base = df["price_eur_mwh"] - sys_wv
    out["f06x_spread_rank_case"] = base.groupby(df["run_no"]).rank(pct=True)
    out["f06x_spread_cummax_case"] = base.groupby(df["run_no"]).cummax()
    out["f06x_spread_vs_case_max"] = base.groupby(df["run_no"]).transform("max") - base
    out["f06x_spread_vs_case_mean"] = base - base.groupby(df["run_no"]).transform("mean")

    # forward-looking spread within the horizon: is a profitable block coming?
    piv = df.pivot(index="run_no", columns="hour", values="price_eur_mwh").sort_index()
    wv_piv = sys_wv.groupby([df["run_no"], df["hour"]]).first().unstack().sort_index()
    S = (piv - wv_piv).to_numpy()
    csum = np.cumsum(np.concatenate([np.zeros((S.shape[0], 1)), S], axis=1), axis=1)
    pos = np.cumsum(np.concatenate([np.zeros((S.shape[0], 1)), (S > 0).astype(float)], axis=1), axis=1)
    idx = pd.MultiIndex.from_product([piv.index, piv.columns], names=["run_no", "hour"])
    for w in (12, 24, 48):
        hi = np.minimum(np.arange(168) + w, 168)
        lo = np.arange(168)
        fwd = (csum[:, hi] - csum[:, lo]) / np.maximum(hi - lo, 1)
        fwd_pos = (pos[:, hi] - pos[:, lo])
        out[f"f06x_spread_fwd_{w}h"] = pd.Series(fwd.ravel(), index=idx).reindex(
            pd.MultiIndex.from_arrays([df["run_no"], df["hour"]])).to_numpy()
        out[f"f06x_profitable_hours_next_{w}h"] = pd.Series(fwd_pos.ravel(), index=idx).reindex(
            pd.MultiIndex.from_arrays([df["run_no"], df["hour"]])).to_numpy()

    # per-plant economic margin, scaled by that plant's capacity
    for plant, r in p2r.items():
        pl, col = sanitize(plant), f"water_value_{sanitize(r)}"
        if col not in df.columns:
            continue
        pgens = [g_ for g_ in gens if sanitize(g_).startswith(pl + "_G")]
        cap = sum(gens[g_]["p_max"] for g_ in pgens) or np.nan
        sp = df["price_eur_mwh"] - df[col]
        out[f"f06x_margin_{pl}"] = sp * cap
        out[f"f06x_spread_pos_{pl}"] = (sp > 0).astype(int)
    return out


def stage_price_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 4: where this hour sits in the week's price structure. All hourly-varying."""
    g = df.groupby("run_no")["price_eur_mwh"]
    rank_desc = g.rank(ascending=False, method="first")
    out = pd.DataFrame(index=df.index)
    for k in (12, 24, 48, 72):
        out[f"f04_in_top_{k}h"] = (rank_desc <= k).astype(int)
    out["f04_price_cummean_case"] = g.cummean() if hasattr(g, "cummean") else \
        g.transform(lambda s: s.expanding().mean())
    piv = df.pivot(index="run_no", columns="hour", values="price_eur_mwh").sort_index()
    P = piv.to_numpy()
    idx = pd.MultiIndex.from_product([piv.index, piv.columns], names=["run_no", "hour"])
    target = pd.MultiIndex.from_arrays([df["run_no"], df["hour"]])
    # signed distance to the week's cheapest / dearest hour
    argmax, argmin = P.argmax(axis=1), P.argmin(axis=1)
    hours = np.arange(168)[None, :]
    out["f04_hours_to_case_peak"] = pd.Series((argmax[:, None] - hours).ravel(), index=idx).reindex(target).to_numpy()
    out["f04_hours_since_case_trough"] = pd.Series((hours - argmin[:, None]).ravel(), index=idx).reindex(target).to_numpy()
    # share of the week's total price mass consumed so far
    csum = np.cumsum(P, axis=1)
    out["f04_cum_price_share"] = pd.Series((csum / csum[:, -1:]).ravel(), index=idx).reindex(target).to_numpy()
    return out


def stage_price_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 3: forward/backward price windows, clipped to the case horizon."""
    piv = df.pivot(index="run_no", columns="hour", values="price_eur_mwh").sort_index()
    P = piv.to_numpy()
    csum = np.cumsum(np.concatenate([np.zeros((P.shape[0], 1)), P], axis=1), axis=1)
    idx = pd.MultiIndex.from_product([piv.index, piv.columns], names=["run_no", "hour"])
    target = pd.MultiIndex.from_arrays([df["run_no"], df["hour"]])
    out = pd.DataFrame(index=df.index)
    for w in (6, 12, 24, 48, 72):
        hi = np.minimum(np.arange(168) + w, 168); lo = np.arange(168)
        fwd = (csum[:, hi] - csum[:, lo]) / np.maximum(hi - lo, 1)
        lo2 = np.maximum(np.arange(168) - w + 1, 0); hi2 = np.arange(168) + 1
        bwd = (csum[:, hi2] - csum[:, lo2]) / np.maximum(hi2 - lo2, 1)
        out[f"f03_price_fwd_{w}h"] = pd.Series(fwd.ravel(), index=idx).reindex(target).to_numpy()
        out[f"f03_price_minus_fwd_{w}h"] = pd.Series((P - fwd).ravel(), index=idx).reindex(target).to_numpy()
        out[f"f03_price_minus_bwd_{w}h"] = pd.Series((P - bwd).ravel(), index=idx).reindex(target).to_numpy()
    return out


def stage_wv_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 5: water value as one common factor plus per-reservoir deviations (PC1 = 91%)."""
    wv_cols = [c for c in df.columns if c.startswith("water_value_")]
    X = df[wv_cols]
    sys_wv = X.mean(axis=1)
    out = pd.DataFrame(index=df.index)
    out["f05_sys_wv"] = sys_wv
    out["f05_wv_dispersion"] = X.std(axis=1)
    for plant, r in plant_to_reservoir().items():
        col = f"water_value_{sanitize(r)}"
        if col in df.columns:
            out[f"f05_wv_dev_{sanitize(plant)}"] = df[col] - sys_wv
    out["f05_wv_rank_spread"] = X.rank(axis=1).std(axis=1)
    return out


def stage_merit_order(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 15: plant capacity structure crossed with the hourly economic signal.

    Merit rank itself is static per unit and therefore invisible in this row layout, so it only
    enters combined with hourly quantities.
    """
    topo = load_topology()
    gens = topo["model"]["generator"]
    p2r = plant_to_reservoir()
    wv_cols = [c for c in df.columns if c.startswith("water_value_")]
    sys_wv = df[wv_cols].mean(axis=1)
    out = pd.DataFrame(index=df.index)
    for plant, r in p2r.items():
        pl = sanitize(plant)
        pgens = sorted(g for g in gens if sanitize(g).startswith(pl + "_G"))
        if not pgens:
            continue
        col = f"water_value_{sanitize(r)}"
        sp = df["price_eur_mwh"] - (df[col] if col in df.columns else sys_wv)
        cap_each = gens[pgens[0]]["p_max"]
        p_min = gens[pgens[0]]["p_min"]
        # how many units the hourly margin can justify running, given each unit's minimum output
        out[f"f15_units_justified_{pl}"] = (sp.clip(lower=0) * cap_each / max(p_min, 1e-6)).clip(upper=len(pgens))
        out[f"f15_cap_share_needed_{pl}"] = sp.clip(lower=0) / max(sp.abs().max(), 1e-6)
        out[f"f15_n_units_{pl}"] = len(pgens)
    return out


def stage_graph_cascade(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 14: upstream cascade aggregates via multi-hop traversal of the topology graph."""
    topo = load_topology()
    conns = topo["connections"]
    upstream: dict[str, set] = {}

    def walk(node, seen=None):
        seen = seen or set()
        acc = set()
        for c in conns:
            if c["to"] == node and c["from"] not in seen:
                seen.add(c["from"])
                if c["from_type"] == "reservoir":
                    acc.add(c["from"])
                acc |= walk(c["from"], seen)
        return acc

    plants = list(topo["model"]["plant"])
    for p in plants:
        upstream[p] = walk(p)

    out = pd.DataFrame(index=df.index)
    for plant, ups in upstream.items():
        pl = sanitize(plant)
        scols = [f"storage_{sanitize(r)}" for r in ups if f"storage_{sanitize(r)}" in df.columns]
        wcols = [f"water_value_{sanitize(r)}" for r in ups if f"water_value_{sanitize(r)}" in df.columns]
        icols = [f"inflow_{sanitize(r)}" for r in ups if f"inflow_{sanitize(r)}" in df.columns]
        if scols:
            out[f"f14_upstream_storage_{pl}"] = df[scols].sum(axis=1)
        if wcols:
            out[f"f14_upstream_wv_{pl}"] = df[wcols].mean(axis=1)
            out[f"f14_upstream_spread_{pl}"] = df["price_eur_mwh"] - df[wcols].mean(axis=1)
        if icols:
            out[f"f14_upstream_inflow_{pl}"] = df[icols].sum(axis=1)
        out[f"f14_n_upstream_{pl}"] = len(ups)
    return out


def stage_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Theme 17/18: calendar position. Hour-of-day is hourly-varying; the rest are case-level."""
    ts = df["timestamp"]
    out = pd.DataFrame(index=df.index)
    out["f17_hour_of_day"] = ts.dt.hour
    out["f17_day_of_week"] = ts.dt.dayofweek
    out["f17_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["f17_day_of_year"] = ts.dt.dayofyear
    out["f17_month"] = ts.dt.month
    out["f17_doy_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    out["f17_doy_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    out["f18_day_in_case"] = df["hour"] // 24
    out["f18_hours_remaining"] = 167 - df["hour"]
    return out
