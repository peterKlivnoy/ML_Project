"""Which training period looks most like the 2023-24 test period? (adversarial validation)

Inputs are known for the test weeks, so the question can be answered from data: for each training
year, fit a classifier to separate that year's weeks from the test weeks using only case-level
inputs. Cross-validated AUC near 0.5 means indistinguishable (test-like); near 1.0 means a
different regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

import shoplite as S
import shoplite_batch as SB


def case_table() -> pd.DataFrame:
    price, inflow, vol, wv, _, _ = S.load_raw()
    cases = SB.all_cases()
    st = cases["starttime"].to_numpy()
    hours = st[:, None] + np.arange(S.T)[None, :] * np.timedelta64(1, "h")
    P = price.reindex(hours.ravel()).ffill().to_numpy().reshape(len(st), S.T)
    wv_end = wv.reindex(st + np.timedelta64(7, "D")).ffill()
    sys_wv = wv_end.mean(1).to_numpy()
    topo = S.load_topology()
    fill = (vol.reindex(st).ffill() / pd.Series({r: topo.res[r]["max_vol"] for r in topo.res}))
    infl = inflow.reindex(hours.ravel()).ffill().sum(1).to_numpy().reshape(len(st), S.T).mean(1)
    d = pd.DataFrame({
        "price_mean": P.mean(1), "price_std": P.std(1), "price_min": P.min(1),
        "price_neg_share": (P < 0).mean(1), "price_range_rel": np.ptp(P, 1) / (np.abs(P.mean(1)) + 1),
        "wv_sys": sys_wv, "spread_mean": P.mean(1) - sys_wv,
        "hours_above_wv": (P > sys_wv[:, None]).mean(1),
        "fill_sys": fill.mean(1).to_numpy(), "inflow": infl,
        "month": pd.DatetimeIndex(st).month,
    })
    d["period"] = np.where(pd.DatetimeIndex(st).year >= 2023, "test",
                           pd.DatetimeIndex(st).year.astype(str))
    return d


if __name__ == "__main__":
    import lightgbm as lgb
    d = case_table()
    feats = [c for c in d.columns if c not in ("period", "month")]
    test = d[d.period == "test"]
    rows = []
    for y in sorted(p for p in d.period.unique() if p != "test"):
        a = pd.concat([d[d.period == y], test])
        lab = (a.period == "test").astype(int)
        clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05,
                                 verbose=-1, n_jobs=1)
        auc = cross_val_score(clf, a[feats], lab, cv=5, scoring="roc_auc").mean()
        g = d[d.period == y]
        rows.append(dict(year=y, adv_auc=auc, price_mean=g.price_mean.mean(),
                         price_std=g.price_std.mean(), neg_share=g.price_neg_share.mean(),
                         spread=g.spread_mean.mean(), wv=g.wv_sys.mean()))
    t = test
    rows.append(dict(year="TEST 23-24", adv_auc=np.nan, price_mean=t.price_mean.mean(),
                     price_std=t.price_std.mean(), neg_share=t.price_neg_share.mean(),
                     spread=t.spread_mean.mean(), wv=t.wv_sys.mean()))
    print(pd.DataFrame(rows).set_index("year").round(3).to_string())
