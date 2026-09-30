"""Post-decision traces at DAILY resolution -- the biggest piece of unused information left.

`Historical_volume` and `Synthetic_water_value` are both daily, so there are seven observations
inside every 168-hour case. `production_features._daily()` resamples to daily and then
immediately takes a 7-day rolling mean, collapsing all seven into one number per case. Every
storage and water-value feature in the project is built on that weekly aggregate.

That matters because the storage delta is how we read post-decision generation, and it was the
single largest gain in the project (+0.074, log 022) using ONE number per week. At daily
resolution the same trace says WHICH DAY the water left the reservoir -- and which-day is
allocation information, the half of the problem we have never had real data for. The count stage
has always been the strong half precisely because weekly resolution could only inform a weekly
quantity.

The features here are (case, day) level, which sits between the case level everything else lives
at and the hourly level the model predicts at. In long format a row is (case, hour, unit) and
`day = hour // 24`, so each row gets the trace for ITS OWN day rather than a case-wide average.

Legality is the same as the existing post-decision features (log 022): the volume file runs to
2025-01-01 and the test period ends 2024-12-25, so the trace exists for every test case.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import production_features as PF

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
san = PF.san
DAYS = 7


def _daily_frame(path: str) -> pd.DataFrame:
    d = pd.read_csv(_ROOT / path)
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=True).dt.tz_localize(None)
    d = d.set_index("date").sort_index()
    if d.index.to_series().diff().median() < pd.Timedelta(days=1):
        d = d.resample("D").mean()
    d.columns = [san(c) for c in d.columns]
    return d


def build(case_start: pd.Series, units: list[str]) -> pd.DataFrame:
    """One row per (case, day) per plant, indexed by (run_no, day).

    Emits `dt_<metric>_<Plant>` so the caller can resolve it per unit the same way as the other
    per-plant blocks. Everything is expressed relative to the case's own week -- share of the
    week's drawdown, rank among the seven days -- so nothing carries the absolute level that
    log 029 showed drifts across the regime break.
    """
    vol = _daily_frame("data/kernel/Historical_volume_2015_2024.csv")
    wv = _daily_frame("data/kernel/Synthetic_water_value_2015_2024.csv")
    inf = _daily_frame("data/kernel/Historical_inflow_1958_2025.csv") * 3600 * 24 / 1e6
    src, dst, _ = PF.plant_graph()
    cs = pd.to_datetime(case_start)

    idx = pd.MultiIndex.from_product([cs.index, range(DAYS)], names=["run_no", "day"])
    out = pd.DataFrame(index=idx)
    # dates for every (case, day) and the day after, in one shot
    d0 = np.repeat(cs.to_numpy(), DAYS) + np.tile(
        [np.timedelta64(k, "D") for k in range(DAYS)], len(cs))
    d1 = d0 + np.timedelta64(1, "D")

    for plant in src:
        res = src.get(plant)
        if res is None:
            continue
        r, p = san(res), san(plant)
        if r not in vol.columns:
            continue
        s0 = vol[r].reindex(d0, method="ffill").to_numpy(dtype=float)
        s1 = vol[r].reindex(d1, method="ffill").to_numpy(dtype=float)
        q = inf[r].reindex(d0, method="ffill").to_numpy(dtype=float) if r in inf.columns else 0.0
        draw = s0 - s1                                  # Mm3 removed that day
        release = draw + q                              # plus what arrived and left again

        # upstream plants' discharge is this reservoir's inflow, same day
        up = 0.0
        for qq, tgt in dst.items():
            if tgt == res and qq != plant and san(qq) in vol.columns:
                u = san(qq)
                us0 = vol[u].reindex(d0, method="ffill").to_numpy(dtype=float)
                us1 = vol[u].reindex(d1, method="ffill").to_numpy(dtype=float)
                ui = inf[u].reindex(d0, method="ffill").to_numpy(dtype=float) if u in inf.columns else 0.0
                up = up + np.clip((us0 - us1) + ui, 0, None)
        release = release + up

        e = np.array([PF.mwh_per_m3(plant, x) for x in s0])
        out[f"dt_draw_{p}"] = draw
        out[f"dt_release_{p}"] = release
        out[f"dt_energy_{p}"] = release * 1e6 * e
        out[f"dt_upstream_{p}"] = up

        # shape within the week: which day did the water actually leave?
        M = release.reshape(len(cs), DAYS)
        tot = np.nansum(np.clip(M, 0, None), axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            share = np.where(tot > 0, np.clip(M, 0, None) / tot, 1.0 / DAYS)
        rankd = np.argsort(np.argsort(-M, axis=1), axis=1) / (DAYS - 1)
        out[f"dt_share_{p}"] = share.reshape(-1)
        out[f"dt_rank_{p}"] = rankd.reshape(-1)
        out[f"dt_vs_weekmean_{p}"] = (M - M.mean(axis=1, keepdims=True)).reshape(-1)
        # is the week front- or back-loaded? (same number on every day of a case, by design)
        w = np.arange(DAYS)[None, :]
        with np.errstate(all="ignore"):
            centroid = np.where(tot.squeeze() > 0,
                                (share * w).sum(axis=1), (DAYS - 1) / 2.0)
        out[f"dt_centroid_{p}"] = np.repeat(centroid, DAYS)

        # the optimiser's own shadow price, moving day by day
        if r in wv.columns:
            w0 = wv[r].reindex(d0, method="ffill").to_numpy(dtype=float)
            w1 = wv[r].reindex(d1, method="ffill").to_numpy(dtype=float)
            out[f"dt_wv_{p}"] = w0
            out[f"dt_wvchg_{p}"] = w1 - w0
            W = (w1 - w0).reshape(len(cs), DAYS)
            out[f"dt_wvchg_vs_week_{p}"] = (W - W.mean(axis=1, keepdims=True)).reshape(-1)

        # neighbouring days, so a row knows whether the drawdown is starting or ending
        R3 = M.copy()
        prev = np.concatenate([R3[:, :1], R3[:, :-1]], axis=1)
        nxt = np.concatenate([R3[:, 1:], R3[:, -1:]], axis=1)
        out[f"dt_release_prev_{p}"] = prev.reshape(-1)
        out[f"dt_release_next_{p}"] = nxt.reshape(-1)
    return out.replace([np.inf, -np.inf], np.nan)


def attach(long: pd.DataFrame, daily: pd.DataFrame, units: list[str],
           run_col: str = "run_no", hour_col: str = "hour") -> pd.DataFrame:
    """Resolve the per-plant daily block onto (case, hour, unit) rows as `own_dt_*`.

    Each row takes the trace for its own day (`hour // 24`) and its own unit's plant, so the
    model sees "how much water left MY reservoir on THIS day" rather than a weekly average.
    """
    day = (long[hour_col].to_numpy() // 24).astype(int)
    key = pd.MultiIndex.from_arrays([long[run_col].to_numpy(), day])
    metrics = sorted({c.rsplit("_", 1)[0] for c in daily.columns})
    plant_of = {u: u.rsplit("_G", 1)[0] for u in units}
    out = {}
    uarr = long["unit"].astype(str).to_numpy()
    for m in metrics:
        vals = np.full(len(long), np.nan, dtype=np.float32)
        for u in units:
            col = f"{m}_{plant_of[u]}"
            if col not in daily.columns:
                continue
            sel = uarr == u
            vals[sel] = daily[col].reindex(key[sel]).to_numpy(dtype=np.float32)
        out[f"own_{m}"] = vals
    return pd.DataFrame(out, index=long.index)
