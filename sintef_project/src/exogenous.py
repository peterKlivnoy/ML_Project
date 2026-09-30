"""
Two families of genuinely NEW information, identified in log 029 as present in the data and
unused. Everything else we have tested since log 025 rearranged features we already had; these
add inputs the model has never seen.

1. INFLOW CLIMATOLOGY FROM THE FULL 67-YEAR RECORD.
   `Historical_inflow_1958_2025.csv` covers 1958-2025 (594k hourly rows). Every seasonal feature
   in the project builds its reference distribution from the CASE dates -- 2015 onward, roughly
   eight years of a sixty-seven year record. An eight-year reference is a poor estimate of "what
   is normal for week 23", and worse, it is contaminated by the same 2021-22 regime we are trying
   to normalise against. Rebuilding the reference from all 67 years gives a stable denominator,
   and it is pure exogenous hydrology: no targets, and it covers the 2023-24 test period.

2. MINIMUM-FLOW OBLIGATION RESOLVED TO THE PLANT THAT MUST SATISFY IT.
   Both constraint files are currently joined as raw weekly-mean series and, in
   `unit_relative`, summarised over a loose "near" neighbourhood (upstream union downstream to
   depth 3). That throws away the actual physics: a minimum flow at reservoir R is an obligation
   on RELEASE FROM R, and the plant drawing from R is what can satisfy it by generating.

   Mapping source reservoirs to constraints gives a clean ownership for four plants:

       b_Byrtevatn  -> Lio          (0.00-0.20 m3/s)
       b_Kjelavatn  -> Vesle Kjela  (0.30-0.75)
       b_Vinjevatn  -> Tokke        (1.00-2.00)
       b_Bandak     -> Hogga        (4.00-5.00)

   Converting the obligation into a FORCED CAPACITY FACTOR is the point. A plant running at
   p_min passes q_min cubic metres per second; if the obligation is f, then f/q_min is the
   fraction of the week the plant must run purely to discharge water it has no choice about.
   That is a direct lower bound on hours-on from physics rather than from economics, and it is
   exactly the shape of an explanation for a chronically always-on unit.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import production_features as PF

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
_KERNEL = _ROOT / "data/kernel"
_EXT = _ROOT / "data/extended"
san = PF.san

_SMOOTH = 7          # +/- days pooled into each day-of-year reference
_WEEK = 168


def _daily_inflow() -> pd.DataFrame:
    """Daily mean inflow per reservoir over the FULL 1958-2025 record."""
    d = pd.read_csv(_KERNEL / "Historical_inflow_1958_2025.csv")
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=True).dt.tz_localize(None)
    d = d.set_index("date").sort_index().resample("D").mean()
    d.columns = [san(c) for c in d.columns]
    return d


def _doy_reference(s: pd.Series) -> pd.DataFrame:
    """Day-of-year reference distribution built from every year in the record.

    Each day-of-year pools a +/-7 day window across all 67 years, so the reference for week 23 is
    estimated from ~1000 observations rather than the ~8 a 2015-onward window would give.

    Computed as a per-day aggregate followed by a CIRCULAR rolling mean over day-of-year rather
    than by masking the full record 366 times: each day-of-year carries the same ~67 observations,
    so pooling then smoothing matches masking for the mean and is a close approximation for the
    dispersion terms, at a fraction of the cost.
    """
    f = pd.DataFrame({"v": s.to_numpy(), "doy": s.index.dayofyear})
    g = f.groupby("doy")["v"]
    agg = pd.DataFrame({"mean": g.mean(), "std": g.std(),
                        "p10": g.quantile(.10), "p50": g.quantile(.50),
                        "p90": g.quantile(.90)}).reindex(range(1, 367)).interpolate()
    w = 2 * _SMOOTH + 1
    tripled = pd.concat([agg, agg, agg], ignore_index=True)
    sm = tripled.rolling(w, center=True, min_periods=1).mean()
    return sm.iloc[366:732].set_axis(range(1, 367))


def inflow_climatology(dates: pd.Series, reservoirs: list[str] | None = None) -> pd.DataFrame:
    """Family 1: this week's inflow against a 67-year day-of-year reference.

    Returns one row per case. All features are scale-free comparisons against the long reference
    (z-score, percentile, ratio), plus year-to-date accumulation anomaly and snowmelt timing --
    the slow-moving hydrological state that decides whether a reservoir is in a drawdown or a
    filling regime, which is what actually sets how many hours a unit runs.
    """
    inf = _daily_inflow()
    cols = reservoirs or list(inf.columns)
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    out = pd.DataFrame(index=range(len(d)))

    series = {c: inf[c] for c in cols if c in inf.columns}
    series["sys"] = inf[[c for c in cols if c in inf.columns]].sum(axis=1)

    for name, s in series.items():
        ref = _doy_reference(s)
        # this case's forward week
        wk = s.rolling(7, min_periods=1).mean().shift(-6)
        v = wk.reindex(d).to_numpy(dtype=float)
        doy = d.dt.dayofyear.to_numpy()
        m = ref["mean"].reindex(doy).to_numpy()
        sd = ref["std"].reindex(doy).to_numpy()
        p50 = ref["p50"].reindex(doy).to_numpy()
        p10 = ref["p10"].reindex(doy).to_numpy()
        p90 = ref["p90"].reindex(doy).to_numpy()
        with np.errstate(all="ignore"):
            out[f"clim_{name}_z"] = np.where(sd > 0, (v - m) / sd, 0.0)
            out[f"clim_{name}_ratio"] = np.where(np.abs(m) > 1e-9, v / m, np.nan)
            out[f"clim_{name}_vs_p50"] = v - p50
            # where in the 67-year spread this week sits, 0 at p10 and 1 at p90
            out[f"clim_{name}_spread_pos"] = np.where(p90 > p10, (v - p10) / (p90 - p10), 0.5)

        # year-to-date accumulation vs the climatological year-to-date -- drought severity
        ytd = s.groupby(s.index.year).cumsum()
        ref_cum = ref["mean"].cumsum()
        yv = ytd.reindex(d).to_numpy(dtype=float)
        rc = ref_cum.reindex(doy).to_numpy()
        with np.errstate(all="ignore"):
            out[f"clim_{name}_ytd_ratio"] = np.where(np.abs(rc) > 1e-9, yv / rc, np.nan)
            out[f"clim_{name}_ytd_anom"] = yv - rc

        # snowmelt timing: how many days ahead of / behind normal this year's melt is,
        # measured by when year-to-date accumulation crosses half the annual normal
        annual = float(ref["mean"].sum())
        half = annual / 2.0
        norm_doy = int(np.argmax(ref_cum.to_numpy() >= half)) + 1 if annual > 0 else 183
        frac = np.where(annual > 0, yv / annual, np.nan)
        out[f"clim_{name}_year_frac"] = frac
        out[f"clim_{name}_melt_offset"] = doy - norm_doy

        # trailing year against the long-run mean -- multi-year wet/dry regime
        tr = s.rolling(365, min_periods=30).mean().reindex(d).to_numpy(dtype=float)
        lm = float(s.mean())
        with np.errstate(all="ignore"):
            out[f"clim_{name}_trail1y_ratio"] = tr / lm if lm else np.nan
    return out


def minflow_obligation(dates: pd.Series, storage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Family 2: minimum-flow obligations resolved to the plant that must satisfy them.

    `storage` is the per-case storage frame (columns `storage_<Reservoir>`), used to evaluate the
    head-dependent energy conversion at the actual operating point rather than a nominal one.
    Returns one row per case with a `_<Plant>` suffixed block, to be resolved per unit by the
    caller the same way the other per-plant blocks are.
    """
    t = PF.topo()
    src, dst, _ = PF.plant_graph()
    gens = t["model"]["generator"]
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    mf = pd.read_csv(_EXT / "Constraint_min_flow.csv")
    mf["date"] = pd.to_datetime(mf["date"], format="mixed", utc=True).dt.tz_localize(None)
    mf = mf.set_index("date").sort_index().resample("D").mean()
    mfw = mf.rolling(7, min_periods=1).mean().shift(-6)          # forward-week mean

    mv = pd.read_csv(_EXT / "Constraint_min_volume.csv")
    mv["date"] = pd.to_datetime(mv["date"], format="mixed", utc=True).dt.tz_localize(None)
    mv = mv.set_index("date").sort_index().resample("D").mean()
    mvw = mv.rolling(7, min_periods=1).mean().shift(-6)

    # reservoir -> constraint column, matching on the sanitised reservoir name
    def find(cols, res):
        for c in cols:
            if san(c).lstrip("b_").lstrip("r_") == san(res) or san(c) == "b_" + san(res):
                return c
        return None

    out = pd.DataFrame(index=range(len(d)))
    for plant_raw in t["model"]["plant"]:
        p = san(plant_raw)
        res = src.get(plant_raw)
        tgt = dst.get(plant_raw)
        gs = [g for g in gens if san(g).startswith(p + "_G")]
        if not gs:
            continue
        p_min = float(gens[gs[0]].get("p_min", np.nan))
        n_units = len(gs)

        # head-dependent m3 -> MWh at this case's actual storage
        if storage is not None and res is not None and f"storage_{san(res)}" in storage.columns:
            sv = storage[f"storage_{san(res)}"].to_numpy(dtype=float)
            conv = np.array([PF.mwh_per_m3(plant_raw, x) for x in sv])
        else:
            conv = np.full(len(d), np.nan)

        # cubic metres per second passed by ONE unit held at minimum load
        with np.errstate(all="ignore"):
            q_min = np.where(conv > 0, p_min / conv / 3600.0, np.nan)

        own_c = find(mf.columns, res) if res else None
        if own_c is not None:
            f = mfw[own_c].reindex(d).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                cf = np.where(q_min > 0, f / q_min, 0.0)
                out[f"mfo_required_flow_{p}"] = f
                out[f"mfo_qmin_{p}"] = q_min
                out[f"mfo_forced_cf_{p}"] = cf
                # hours this unit must run purely to pass water it has no discretion over
                out[f"mfo_forced_hours_{p}"] = np.clip(cf, 0, 1) * _WEEK
                out[f"mfo_must_run_all_week_{p}"] = (cf >= 1.0).astype(float)
                out[f"mfo_forced_vol_week_{p}"] = f * 3600 * _WEEK
        else:
            for k in ("required_flow", "qmin", "forced_cf", "forced_hours",
                      "must_run_all_week", "forced_vol_week"):
                out[f"mfo_{k}_{p}"] = 0.0 if k != "qmin" else q_min

        # obligation at the reservoir this plant discharges INTO: releasing here helps satisfy it
        tgt_c = find(mf.columns, tgt) if tgt else None
        out[f"mfo_downstream_req_{p}"] = (mfw[tgt_c].reindex(d).to_numpy(dtype=float)
                                          if tgt_c is not None else 0.0)
        # terminal plants discharge out of the modelled system: no downstream pull at all
        out[f"mfo_is_terminal_{p}"] = float(tgt is None)
        out[f"mfo_n_units_{p}"] = float(n_units)

        # minimum-volume headroom: storage the plant is forbidden to draw below
        mv_c = next((c for c in mv.columns if san(c) == san(res)), None) if res else None
        if (mv_c is not None and storage is not None
                and f"storage_{san(res)}" in storage.columns):
            floor = mvw[mv_c].reindex(d).to_numpy(dtype=float)
            sv = storage[f"storage_{san(res)}"].to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                out[f"mfo_vol_headroom_{p}"] = sv - floor
                out[f"mfo_vol_headroom_frac_{p}"] = np.where(sv > 0, (sv - floor) / sv, np.nan)
        else:
            out[f"mfo_vol_headroom_{p}"] = np.nan
            out[f"mfo_vol_headroom_frac_{p}"] = np.nan
    return out
