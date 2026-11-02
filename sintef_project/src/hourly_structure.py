"""
Hour-level temporal structure for the pointwise model.

The pointwise model has 15 engineered features and all 15 are price-versus-water-value economics
(`f05`, `f06`, `f07`, `f10`, `f14`, `f15`). There is not one temporal feature among them: no
hour-of-day, no lag, no persistence, no window aggregate. Every hour is scored as if it were an
independent draw.

That is contradicted by the data. Measured over 2015-2022:

    P(on at t | on at t-1) = 0.995        P(on at t | off at t-1) = 0.007
    switches per unit-week  = 0.5 - 3.6   vs 9.3 - 27.2 if the same hours were scattered
    contiguous ON blocks    = 1.2 - 2.7 per unit-week

Commitments are blocks, roughly an order of magnitude more contiguous than chance. Start costs
in the topology are the obvious reason: the optimiser will not pay to cycle a unit hourly.

The naive consequence -- allocate a unit's `c` hours as one contiguous window -- was tested and is
WORSE than ranking hours by spread (0.9057 vs 0.9485 with oracle counts), because 1.2-2.7 blocks
is not 1 block, and a single window is too rigid a shape to impose.

So the structure belongs in features, not in a hard rule. What matters at hour t is not the
spread at t but the spread over a STRETCH around t, because a unit running c hours runs during a
high-total-spread stretch. These features give the model that at several scales and let it learn
which scale matters, rather than committing to one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WINDOWS = (3, 6, 12, 24, 48, 96)
H = 168


def _centered_window_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Mean over a window of length w centred on each hour, edges shrinking rather than padding.

    Edge hours matter here -- hour 0 and hour 167 of a case are real decisions, not padding --
    so the window truncates at the boundary instead of wrapping or filling with zeros, which
    would invent price structure that is not there.
    """
    cs = np.r_[0.0, np.cumsum(a)]
    n = len(a)
    idx = np.arange(n)
    lo = np.maximum(idx - w // 2, 0)
    hi = np.minimum(idx + (w - w // 2), n)
    return (cs[hi] - cs[lo]) / (hi - lo)


def build(df: pd.DataFrame, signal: str = "f06_spread_sys",
          run_col: str = "run_no", hour_col: str = "hour") -> pd.DataFrame:
    """Per-hour block-structure features, one row per (case, hour) aligned to `df`.

    All features are computed WITHIN a case and are either scale-free (ranks, z-scores) or
    differences against the case's own level, so none of them carry the absolute price level
    that log 029 showed drifts across the 2021-22 crisis.
    """
    d = df[[run_col, hour_col, signal]].copy()
    d = d.sort_values([run_col, hour_col])
    out = {k: np.empty(len(d)) for k in _feature_names()}
    pos = 0
    for _, g in d.groupby(run_col, sort=False):
        s = g[signal].to_numpy(dtype=float)
        n = len(s)
        sl = slice(pos, pos + n)
        med = np.nanmedian(s)
        sd = np.nanstd(s)
        order = np.argsort(np.argsort(-s))          # 0 = highest spread hour in the case

        out["hs_rank"][sl] = order / max(n - 1, 1)
        out["hs_z"][sl] = (s - np.nanmean(s)) / sd if sd > 0 else 0.0
        out["hs_dist_to_peak"][sl] = np.abs(np.arange(n) - int(np.argmax(s))) / n
        out["hs_dist_to_trough"][sl] = np.abs(np.arange(n) - int(np.argmin(s))) / n

        for w in WINDOWS:
            m = _centered_window_mean(s, w)
            out[f"hs_win{w}"][sl] = m - med
            r = np.argsort(np.argsort(-m)) / max(n - 1, 1)
            out[f"hs_win{w}_rank"][sl] = r
            # is this hour inside the best length-w stretch of the week?
            if n >= w:
                cs = np.r_[0.0, np.cumsum(s)]
                tot = cs[w:] - cs[:-w]
                j = int(np.argmax(tot))
                inb = np.zeros(n); inb[j:j + w] = 1.0
            else:
                inb = np.ones(n)
            out[f"hs_in_best{w}"][sl] = inb

        # run structure: how deep into a contiguous above-median stretch this hour sits
        above = (s > med).astype(int)
        run_id = np.cumsum(np.r_[1, np.diff(above) != 0])
        depth = np.zeros(n); length = np.zeros(n)
        for rid in np.unique(run_id):
            m_ = run_id == rid
            k = int(m_.sum())
            depth[m_] = np.arange(k) / max(k - 1, 1)
            length[m_] = k / n
        out["hs_run_pos"][sl] = depth
        out["hs_run_len"][sl] = length
        out["hs_above_med"][sl] = above

        # lags and leads within the case, as differences so no level leaks in
        for lag in (1, 3, 24):
            out[f"hs_lag{lag}"][sl] = s - np.r_[np.full(min(lag, n), s[0]), s[:-lag]][:n]
            out[f"hs_lead{lag}"][sl] = s - np.r_[s[lag:], np.full(min(lag, n), s[-1])][:n]

        # cumulative share of the week's positive spread consumed before this hour
        pos_s = np.clip(s, 0, None)
        tot = pos_s.sum()
        out["hs_cum_share"][sl] = (np.cumsum(pos_s) / tot) if tot > 0 else np.linspace(0, 1, n)

        # calendar position, cyclically encoded so hour 23 and hour 0 are adjacent
        hod = g[hour_col].to_numpy() % 24
        out["hs_hod_sin"][sl] = np.sin(2 * np.pi * hod / 24)
        out["hs_hod_cos"][sl] = np.cos(2 * np.pi * hod / 24)
        dow = (g[hour_col].to_numpy() // 24) % 7
        out["hs_dow_sin"][sl] = np.sin(2 * np.pi * dow / 7)
        out["hs_dow_cos"][sl] = np.cos(2 * np.pi * dow / 7)
        pos += n

    res = pd.DataFrame(out, index=d.index).reindex(df.index)
    return res


def count_coupled(df: pd.DataFrame, exp_counts: dict[str, np.ndarray], units: list[str],
                  signal: str = "f06_spread_sys", run_col: str = "run_no",
                  hour_col: str = "hour") -> pd.DataFrame:
    """Couple allocation to the count model: the stretch whose LENGTH is what this unit will run.

    If the count model says a unit runs c hours, the hours it actually runs are a high-total-spread
    stretch of about that length. `hs_cwin_<unit>` is the spread averaged over a window of exactly
    c hours centred on t, and `hs_in_cbest_<unit>` marks the single best such window. This is the
    only place the two stages interact on the hour axis rather than through a scalar.
    """
    d = df[[run_col, hour_col, signal]].copy().sort_values([run_col, hour_col])
    runs = d[run_col].to_numpy()
    out = {}
    for u in units:
        c_by_run = pd.Series(exp_counts[u])
        cwin = np.empty(len(d)); inb = np.empty(len(d))
        pos = 0
        for rn, g in d.groupby(run_col, sort=False):
            s = g[signal].to_numpy(dtype=float); n = len(s)
            c = int(np.clip(round(float(c_by_run.get(rn, np.nan) if not np.isnan(
                c_by_run.get(rn, np.nan)) else n / 2)), 1, n))
            m = _centered_window_mean(s, c)
            cwin[pos:pos + n] = m - np.nanmedian(s)
            cs = np.r_[0.0, np.cumsum(s)]
            if n > c:
                tot = cs[c:] - cs[:-c]; j = int(np.argmax(tot))
                b = np.zeros(n); b[j:j + c] = 1.0
            else:
                b = np.ones(n)
            inb[pos:pos + n] = b
            pos += n
        out[f"hs_cwin_{u}"] = cwin
        out[f"hs_in_cbest_{u}"] = inb
    return pd.DataFrame(out, index=d.index).reindex(df.index)


def _feature_names() -> list[str]:
    n = ["hs_rank", "hs_z", "hs_dist_to_peak", "hs_dist_to_trough",
         "hs_run_pos", "hs_run_len", "hs_above_med", "hs_cum_share",
         "hs_hod_sin", "hs_hod_cos", "hs_dow_sin", "hs_dow_cos"]
    for w in WINDOWS:
        n += [f"hs_win{w}", f"hs_win{w}_rank", f"hs_in_best{w}"]
    for lag in (1, 3, 24):
        n += [f"hs_lag{lag}", f"hs_lead{lag}"]
    return n
