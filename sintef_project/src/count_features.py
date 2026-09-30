"""
Multi-horizon time-series features for the count model (hours-on per unit-week).

The count is the binding constraint (log 018/019/020): the oracle reaches 0.9765 given true
counts, and our count MAE is 48.5h against an 87h mean. These features target it specifically.

Design principles, from what the project has established:
  * Windows BOTH directions. Water is storable, so SHOP trades volume across weeks: high prices
    next week mean saving water now, lowering this week's count. Forward windows are permitted -
    only test-period targets are withheld (log 011).
  * Multi-horizon normalisation. A week's level means nothing absolutely; it means something
    relative to the month, the year, and the whole history. The spread changed sign after 2020,
    so absolute levels are actively misleading across regimes (log 020).
  * Latent quantities inferred from observables. SHOP sees demand; we do not. Price level
    relative to its seasonal norm, price volatility, and intraday peakiness are all proxies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
K = _ROOT / "data/kernel"

WINDOWS_BACK = {"1w": 7, "1m": 30, "3m": 90, "1y": 365}
WINDOWS_FWD = {"f1w": 7, "f1m": 30}


def _daily(path, how="mean") -> pd.DataFrame:
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d = d.set_index("date").sort_index()
    return d.resample("D").mean() if how == "mean" else d.resample("D").sum()


def base_series() -> dict[str, pd.Series]:
    """System-level daily series, plus the spread that every surviving feature is built from."""
    price = _daily(K / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    wv = _daily(K / "Synthetic_water_value_2015_2024.csv")
    vol = _daily(K / "Historical_volume_2015_2024.csv")
    inf = _daily(K / "Historical_inflow_1958_2025.csv")
    res = list(vol.columns)
    sys_wv = wv[res].mean(axis=1)
    idx = price.index
    out = {
        "price": price,
        "wv": sys_wv.reindex(idx).ffill(),
        "spread": price - sys_wv.reindex(idx).ffill(),
        "storage": vol[res].sum(axis=1).reindex(idx).ffill(),
        "inflow": inf[[c for c in inf.columns if c in res]].sum(axis=1).reindex(idx).ffill(),
    }
    # inferred demand proxies: SHOP sees demand, we infer it from price behaviour
    out["price_vol"] = price.rolling(7, min_periods=2).std()
    return out


def horizon_frame(series: pd.Series, at_dates: pd.Series, name: str) -> pd.DataFrame:
    """Level in the case week, in windows before and after it, and every relative comparison."""
    s = series.sort_index()
    d = pd.to_datetime(pd.Series(at_dates)).reset_index(drop=True)
    out = pd.DataFrame(index=range(len(d)))

    # the case's own week
    wk = s.rolling(7, min_periods=1).mean().shift(-6)     # mean of [d, d+6]
    out[f"{name}_week"] = wk.reindex(d).to_numpy()

    # trailing windows (strictly before the case)
    for tag, n in WINDOWS_BACK.items():
        b = s.rolling(n, min_periods=2).mean().shift(1)
        out[f"{name}_back_{tag}"] = b.reindex(d).to_numpy()

    # forward windows (after the case week) - legal, and the storability argument makes them matter
    for tag, n in WINDOWS_FWD.items():
        f = s.rolling(n, min_periods=2).mean().shift(-(n + 6))
        out[f"{name}_{tag}"] = f.reindex(d).to_numpy()

    # previous and next week explicitly
    out[f"{name}_prev_week"] = s.rolling(7, min_periods=1).mean().shift(1).reindex(d).to_numpy()
    out[f"{name}_next_week"] = s.rolling(7, min_periods=1).mean().shift(-13).reindex(d).to_numpy()

    # relative comparisons - what actually carries signal across regimes
    w = out[f"{name}_week"]
    for tag in list(WINDOWS_BACK) + list(WINDOWS_FWD):
        col = f"{name}_back_{tag}" if tag in WINDOWS_BACK else f"{name}_{tag}"
        out[f"{name}_vs_{tag}"] = w - out[col]
        denom = out[col].replace(0, np.nan)
        out[f"{name}_ratio_{tag}"] = w / denom
    out[f"{name}_vs_prev_week"] = w - out[f"{name}_prev_week"]
    out[f"{name}_vs_next_week"] = w - out[f"{name}_next_week"]
    out[f"{name}_next_minus_prev"] = out[f"{name}_next_week"] - out[f"{name}_prev_week"]

    # seasonal position: this week against the same calendar week across all history
    doy = pd.Series(s.index.dayofyear, index=s.index)
    seas = s.groupby(doy).transform("mean")
    out[f"{name}_vs_seasonal"] = w - seas.reindex(d).to_numpy()
    return out


def build_count_features(at_dates: pd.Series) -> pd.DataFrame:
    """All horizons for all system series, indexed to match `at_dates` (one row per case)."""
    ser = base_series()
    frames = [horizon_frame(s, at_dates, n) for n, s in ser.items()]
    return pd.concat(frames, axis=1)


def own_reservoir_features(reservoir: str, at_dates: pd.Series) -> pd.DataFrame:
    """The same horizon treatment for one unit's own reservoir."""
    price = _daily(K / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    wv = _daily(K / "Synthetic_water_value_2015_2024.csv")
    vol = _daily(K / "Historical_volume_2015_2024.csv")
    if reservoir not in wv.columns:
        return pd.DataFrame(index=range(len(at_dates)))
    own_wv = wv[reservoir].reindex(price.index).ffill()
    frames = [
        horizon_frame(own_wv, at_dates, "ownwv"),
        horizon_frame(price - own_wv, at_dates, "ownspread"),
    ]
    if reservoir in vol.columns:
        frames.append(horizon_frame(vol[reservoir].reindex(price.index).ffill(), at_dates, "ownstor"))
    return pd.concat(frames, axis=1)
