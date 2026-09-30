"""Seasonal position and long-horizon arbitrage -- the regime where the model actually loses.

Per-case error analysis puts the damage in a clear tail: the hardest 15% of cases are September
weeks (month 9.1 vs 4.1, a 1.44 SD gap) with high prices (113 vs 55), negative spread (-32.7 vs
-1.1), profitable hours in the minority (28% vs 59%), and nearly three times the switching (57.6
vs 20.0 per week). Median case AUC is 0.975; the worst decile averages 0.879.

Two gaps explain why the model has nothing to say there.

NO SEASONAL POSITION. The shared feature block carries hour-of-day and day-of-week and nothing
longer. Month is the single largest separator between hard and easy cases, and the model cannot
see it.

NO SEASONAL HORIZON. Forward features stop at one month (`price_f1w`, `wv_f1m`). In September
winter is two to four months out. The autumn decision is intertemporal -- hold water now because
it is worth more in January -- and a one-month horizon cannot express it. A negative spread means
water is worth more than today's price, so the unit should NOT run on current economics; what
decides it is how much more it will be worth later, and by when.

Everything here is exogenous (prices, inflow climatology, calendar). The price file runs to
2025-12-31 against a test period ending 2024-12-25, so the long horizons exist for every test
case, under the same convention as the other forward features (log 022).
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
# horizons that span the hoarding decision, unlike the 1w/1m the model already has
HORIZONS = (30, 60, 90, 180)


def build(case_start: pd.Series) -> pd.DataFrame:
    """One row per case: seasonal position, long-horizon arbitrage, and their interaction."""
    d = pd.to_datetime(pd.Series(case_start)).reset_index(drop=True)
    out = pd.DataFrame(index=range(len(d)))

    pr = pd.read_csv(_ROOT / "data/kernel/Historical_day_ahead_price_2015_2025.csv")
    pr["date"] = pd.to_datetime(pr["date"], format="mixed", utc=True).dt.tz_localize(None)
    price = pr.set_index("date")[pr.columns[1]].sort_index().resample("D").mean()
    wv = pd.read_csv(_ROOT / "data/kernel/Synthetic_water_value_2015_2024.csv")
    wv["date"] = pd.to_datetime(wv["date"], format="mixed", utc=True).dt.tz_localize(None)
    wv = wv.set_index("date").sort_index()
    wv.columns = [san(c) for c in wv.columns]
    sys_wv = wv.mean(axis=1)

    # ---- where in the year are we -------------------------------------------------
    doy = d.dt.dayofyear.to_numpy()
    out["sr_doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["sr_doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["sr_month"] = d.dt.month.to_numpy().astype(float)
    # distance to the two turning points of the hydrological year: midwinter (peak prices,
    # deepest drawdown) and midsummer (melt, refill). Signed, so before and after differ.
    out["sr_days_to_midwinter"] = ((d - pd.to_datetime(
        d.dt.year.astype(str) + "-01-01")).dt.days.to_numpy() + 182.6) % 365.25 - 182.6
    out["sr_days_to_midsummer"] = ((d - pd.to_datetime(
        d.dt.year.astype(str) + "-07-01")).dt.days.to_numpy() + 182.6) % 365.25 - 182.6
    out["sr_is_autumn"] = d.dt.month.isin([9, 10, 11]).to_numpy().astype(float)
    out["sr_is_drawdown"] = d.dt.month.isin([11, 12, 1, 2, 3]).to_numpy().astype(float)

    # ---- long-horizon arbitrage: is water worth more later, and how much later? -----
    p0 = price.reindex(d, method="ffill").to_numpy(dtype=float)
    w0 = sys_wv.reindex(d, method="ffill").to_numpy(dtype=float)
    out["sr_price_now"] = p0
    for h in HORIZONS:
        fwd = price.rolling(h, min_periods=1).mean().shift(-h + 1)
        pf = fwd.reindex(d, method="ffill").to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            out[f"sr_price_fwd{h}_ratio"] = np.where(np.abs(p0) > 1e-9, pf / p0, np.nan)
            out[f"sr_price_fwd{h}_minus_now"] = pf - p0
            # the hoarding signal: how far the FORWARD price sits above today's water value.
            # positive means holding water is worth more than selling it now.
            out[f"sr_hold_premium{h}"] = pf - w0
    # how fast the premium decays -- near vs far horizon tells you WHEN to spend the water
    out["sr_premium_slope"] = out["sr_hold_premium90"] - out["sr_hold_premium30"]

    # ---- the regime itself, as the model would need to condition on it --------------
    spread_now = p0 - w0
    out["sr_spread_now"] = spread_now
    out["sr_spread_negative"] = (spread_now < 0).astype(float)
    # negative spread in autumn is the specific hard regime; give it directly rather than
    # hoping the tree finds the interaction from month and spread separately
    out["sr_autumn_x_negspread"] = out["sr_is_autumn"] * out["sr_spread_negative"]
    out["sr_negspread_x_premium"] = out["sr_spread_negative"] * out["sr_hold_premium90"]

    # ---- where in the fill cycle, against the 67-year record ------------------------
    inf = pd.read_csv(_ROOT / "data/kernel/Historical_inflow_1958_2025.csv")
    inf["date"] = pd.to_datetime(inf["date"], format="mixed", utc=True).dt.tz_localize(None)
    inf = inf.set_index("date").sort_index().resample("D").mean()
    s = inf.sum(axis=1)
    clim = s.groupby(s.index.dayofyear).mean()
    ytd = s.groupby(s.index.year).cumsum()
    ann = float(clim.sum())
    with np.errstate(all="ignore"):
        out["sr_year_inflow_frac"] = ytd.reindex(d, method="ffill").to_numpy(dtype=float) / ann
        out["sr_clim_inflow_now"] = clim.reindex(doy).to_numpy(dtype=float)
    return out.replace([np.inf, -np.inf], np.nan)
