"""Rates of change of the economic state, per plant and system-wide.

The terminal-water-value work added lags around the decision horizon plus a slope, and that block
produced one of the largest single gains in the project (~+0.006). The reason generalises: SHOP
does not decide on levels, it decides on whether water is getting scarcer or cheaper relative to
price over the horizon. A reservoir at 40% that is filling and a reservoir at 40% that is
draining call for opposite commitments, and a level feature cannot tell them apart.

Derivatives are also, for free, the fix for the drift problem of log 029. That entry found 14% of
model importance sitting on absolute price and water-value levels that shift by most of a standard
deviation across the 2021-22 crisis, while every stationary feature was a difference or a ratio.
A slope IS a difference, so this family is stationary by construction and should transfer across
the regime break into the 2023-24 test period.

Everything here is computed on daily series and evaluated at case-relative offsets, so a feature
means the same thing regardless of where the case sits in the record.
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
san = PF.san

# offsets in days relative to case start; 0 = start, 7 = the documented decision horizon
_POINTS = (-28, -14, -7, 0, 7, 14)


def _daily(name: str) -> pd.DataFrame:
    d = pd.read_csv(_KERNEL / name)
    d["date"] = pd.to_datetime(d["date"], format="mixed", utc=True).dt.tz_localize(None)
    d = d.set_index("date").sort_index()
    if not d.index.inferred_freq:
        d = d.resample("D").mean()
    d.columns = [san(c) for c in d.columns]
    return d


def _sample(s: pd.Series, starts: pd.Series, offset: int) -> np.ndarray:
    return s.reindex(starts + pd.Timedelta(days=offset), method="ffill").to_numpy(dtype=float)


def build(dates: pd.Series, per_plant: bool = True) -> pd.DataFrame:
    """Slopes, accelerations and horizon-relative changes for water value, storage and price.

    One row per case. Returns per-plant blocks suffixed `_<Plant>` alongside system aggregates,
    matching the layout the rest of the pipeline resolves per unit.
    """
    starts = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    wv = _daily("Synthetic_water_value_2015_2024.csv")
    vol = _daily("Historical_volume_2015_2024.csv")
    inf = _daily("Historical_inflow_1958_2025.csv")
    pr = _daily("Historical_day_ahead_price_2015_2025.csv")
    price = pr[pr.columns[0]]

    src, _, _ = PF.plant_graph()
    out = pd.DataFrame(index=range(len(starts)))

    def block(s: pd.Series, tag: str):
        vals = {o: _sample(s, starts, o) for o in _POINTS}
        # slope across the decision horizon: what the optimiser is pricing against
        out[f"sl_{tag}_d0_d7"] = vals[7] - vals[0]
        out[f"sl_{tag}_d7_d14"] = vals[14] - vals[7]
        # trailing slopes: the regime the case starts in
        out[f"sl_{tag}_m7_d0"] = vals[0] - vals[-7]
        out[f"sl_{tag}_m28_d0"] = (vals[0] - vals[-28]) / 4.0
        # acceleration: is the trend itself turning
        out[f"sl_{tag}_accel"] = (vals[7] - vals[0]) - (vals[0] - vals[-7])
        # forward vs trailing: does the horizon continue or reverse the recent past
        with np.errstate(all="ignore"):
            tr = vals[0] - vals[-7]
            fw = vals[7] - vals[0]
            out[f"sl_{tag}_reversal"] = np.where(np.abs(tr) > 1e-9, fw / tr, 0.0)
        # normalised by level so it is comparable across regimes and reservoirs
        with np.errstate(all="ignore"):
            out[f"sl_{tag}_d0_d7_pct"] = np.where(np.abs(vals[0]) > 1e-9,
                                                  (vals[7] - vals[0]) / np.abs(vals[0]), 0.0)

    sys_wv = wv.mean(axis=1)
    sys_vol = vol.sum(axis=1)
    sys_inf = inf[[c for c in inf.columns if c in vol.columns]].sum(axis=1)
    block(sys_wv, "wv_sys")
    block(sys_vol, "stor_sys")
    block(price, "price_sys")
    block(sys_inf, "inflow_sys")

    # the spread is the quantity the decision actually turns on, so its slope matters most
    block(price - sys_wv, "spread_sys")

    if per_plant:
        for plant, res in src.items():
            if res is None:
                continue
            r = san(res); p = san(plant)
            if r in wv.columns:
                block(wv[r], f"wv_{p}")
                block(price - wv[r], f"spread_{p}")
            if r in vol.columns:
                block(vol[r], f"stor_{p}")
            if r in inf.columns:
                block(inf[r], f"inflow_{p}")
    return out.replace([np.inf, -np.inf], np.nan)


def themes(cols: list[str]) -> dict[str, list[str]]:
    """Split the family into small single-idea blocks, ~7 features each.

    Adding all 287 at once was a mistake twice over. It made the result unattributable -- the
    first sweep lost 0.0020 overall while gaining 0.0154 on Haukeli, and with one undifferentiated
    block there was no way to keep the part that worked. And it silently broke the tuning:
    `colsample_bytree=0.8` draws ~156 of 195 columns per tree but ~386 of 482, so the same
    nominal value is a completely different amount of regularisation. The 287-feature arm was not
    the tuned model plus features, it was an untuned model, and comparing it against the tuned
    baseline was not a fair test.

    Each theme here is one quantity's dynamics, small enough to add without re-tuning.
    """
    groups: dict[str, list[str]] = {}
    for c in cols:
        if not c.startswith("sl_"):
            continue
        body = c[3:]
        for q in ("wv", "stor", "price", "inflow", "spread"):
            if body.startswith(q + "_"):
                rest = body[len(q) + 1:]
                scope = "sys" if rest.startswith("sys_") else "plant"
                groups.setdefault(f"{q}_{scope}", []).append(c)
                break
    return groups
