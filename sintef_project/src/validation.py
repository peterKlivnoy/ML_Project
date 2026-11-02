"""Rotating two-month block validation.

Every protocol we have used so far holds out a whole YEAR (or a six-month walk-forward fold).
That wastes data -- 2021 alone is 365 of 2,922 cases removed from training -- and confounds
"which year" with "which regime".

Holding out the last two months of each year instead gives eight blocks, one per year, all
regimes represented, and ~86% of cases still available for training. But Nov-Dec is a single
season: winter, peak demand, reservoir drawdown. The test set is a full two years including
spring melt, so a Nov-Dec-only estimate would measure one season and report it as general.

Rotating the held-out block across years fixes that at no cost: 2015 holds out Jan-Feb, 2016
Mar-Apr, 2017 May-Jun, and so on. Same number of blocks, same data volume, but every season and
every year appears exactly once.

BUFFER. Consecutive cases overlap 144 of their 168 hours at 88% label agreement (log 017), so a
case starting a week before a held-out block covers hours inside it. Without a gap the block is
not really held out. `buffer_days` drops cases whose window reaches into the block; they are
excluded from BOTH sides rather than silently assigned to training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 7


def rotating_blocks(case_start: pd.Series, n_months: int = 2,
                    buffer_days: int = WEEK) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Yield (train_idx, val_idx, label) over rotating two-month blocks, one per year.

    `case_start` maps case id -> start timestamp. Indices returned are positions into it.
    """
    s = pd.to_datetime(case_start)
    years = sorted(s.dt.year.unique())
    out = []
    for k, y in enumerate(years):
        first = (k * n_months) % 12 + 1
        months = [((first - 1 + i) % 12) + 1 for i in range(n_months)]
        # a rotation may wrap past December into the following calendar year
        lo = pd.Timestamp(year=y, month=months[0], day=1)
        hi = lo + pd.DateOffset(months=n_months)
        val = (s >= lo) & (s < hi)
        # exclude cases whose 168h window reaches into the block from either side
        near = (s >= lo - pd.Timedelta(days=buffer_days)) & (s < hi + pd.Timedelta(days=buffer_days))
        train = ~near
        if val.sum() == 0:
            continue
        out.append((np.flatnonzero(train.to_numpy()), np.flatnonzero(val.to_numpy()),
                    f"{y}-{months[0]:02d}..{months[-1]:02d}"))
    return out


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Per-fold table plus mean and SD. Judge on the sign pattern, not the mean alone."""
    d = pd.DataFrame(rows)
    agg = pd.DataFrame([{**{c: d[c].mean() for c in d.columns if d[c].dtype != object},
                         "fold": "MEAN"},
                        {**{c: d[c].std() for c in d.columns if d[c].dtype != object},
                         "fold": "SD"}])
    return pd.concat([d, agg], ignore_index=True)
