"""Long-format (case, hour, unit) features built around the shoplite optimiser.

The optimiser already answers most of the question (raw micro-AUC ~0.967 with no learning), so
the learner's job is narrower than before: correct the places where our replica of SHOP is
systematically wrong (frozen heads, unknown initial state, Lio's min-flow behaviour, degenerate
weeks where timing is arbitrary) and turn hard 0/1 solutions into calibrated probabilities.

Feature groups (prefixes):
  lp_   optimiser solution and its temporal context (commitment, power, local water value)
  ec_   economics straight from the inputs (price shape, spread to terminal water value)
  cn_   constraint series (min flow, min volume), shared by all units
  st_   observed storage trajectory over the case window (post-decision, available for test)
  s_    static unit attributes

Everything is computed on dense (case, unit, hour) float32 tensors (broadcasts stay views) and
written once into a single float32 matrix in (case, unit, hour) order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import shoplite as S

T = S.T


def _roll(x: np.ndarray, w: int) -> np.ndarray:
    """Centred rolling mean along the last axis, edges use the available window."""
    c = np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)]), axis=-1, dtype=np.float64)
    n = x.shape[-1]
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    return ((c[..., hi] - c[..., lo]) / (hi - lo)).astype(np.float32)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if k > 0:
        out[..., k:] = x[..., :-k]
    else:
        out[..., :k] = x[..., -k:]
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    """Rank within the last axis, scaled to [0, 1]."""
    return (np.argsort(np.argsort(x, axis=-1), axis=-1) / (x.shape[-1] - 1)).astype(np.float32)


def _run_features(u: np.ndarray):
    """Length of the current run of the (rounded) state and position inside it."""
    b = (u > 0.5)
    n = b.shape[-1]
    start = np.zeros(b.shape, np.int32)
    for t in range(1, n):
        same = b[..., t] == b[..., t - 1]
        start[..., t] = np.where(same, start[..., t - 1], t)
    end = np.zeros(b.shape, np.int32)
    end[..., -1] = n - 1
    for t in range(n - 2, -1, -1):
        same = b[..., t] == b[..., t + 1]
        end[..., t] = np.where(same, end[..., t + 1], t)
    t_idx = np.arange(n)
    length = (end - start + 1).astype(np.float32)
    since = (t_idx - start).astype(np.float32)
    until = (end - t_idx).astype(np.float32)
    return length, since, until


def build(cases: pd.DataFrame, lp_unit: pd.DataFrame, lp_case: pd.DataFrame,
          labels: pd.DataFrame | None = None, prefix: str = "lp"):
    """cases: run_no, starttime. Returns (meta frame, float32 matrix, feature names).

    Rows are in (case, unit, hour) order; `meta` holds run_no, unit index, hour and y.
    """
    topo = S.load_topology()
    units = list(topo.units)
    su = [S.sanitize(u) for u in units]
    N, U = len(cases), len(units)
    runs = cases["run_no"].to_numpy()
    starts = pd.to_datetime(cases["starttime"]).to_numpy()
    price_s, inflow, vol, wv, minvol, minflow = S.load_raw()

    # ---- optimiser tensors (case, unit, hour)
    lu = lp_unit.set_index(["run_no", "unit", "hour"]).sort_index()
    idx = pd.MultiIndex.from_product([runs, su, range(T)], names=["run_no", "unit", "hour"])
    lu = lu.reindex(idx)

    def ten(col):
        return lu[col].to_numpy(np.float32).reshape(N, U, T)
    u, p, q = ten("u"), ten("p"), ten("q")
    lwv = ten("lwv") if "lwv" in lu.columns else None
    lc = lp_case.set_index(["run_no", "hour"]).sort_index()
    lc = lc.reindex(pd.MultiIndex.from_product([runs, range(T)]))

    # ---- inputs per case
    hours = (starts[:, None] + np.arange(T)[None, :] * np.timedelta64(1, "h"))
    price = price_s.reindex(hours.ravel()).ffill().to_numpy(np.float32).reshape(N, T)
    end = starts + np.timedelta64(7, "D")
    wv_end = wv.reindex(end).ffill()
    wv_0 = wv.reindex(starts).ffill()
    v_0 = vol.reindex(starts).ffill()

    pmax = np.array([topo.units[x].pmax for x in units], np.float32)
    pmin = np.array([topo.units[x].pmin for x in units], np.float32)
    qmax = np.array([S.q_range(topo.units[x])[1] for x in units], np.float32)
    updn = [S.unit_up_down(topo, x) for x in units]
    plant_of = [topo.units[x].plant for x in units]
    plants = sorted(set(plant_of))
    merit = {}
    for chain in S.SIBLING_ORDER:
        for k, nm in enumerate(chain):
            merit[nm] = k
    nsib = np.array([sum(1 for y in units if topo.units[y].plant == topo.units[x].plant)
                     for x in units], np.float32)

    F = {}
    bc = lambda a: np.broadcast_to(np.asarray(a, np.float32), (N, U, T))  # noqa: E731
    pre = f"{prefix}_"

    # ---- lp_: optimiser solution + temporal context
    F[pre + "u"] = u
    F[pre + "p_rel"] = p / pmax[None, :, None]
    F[pre + "q_rel"] = q / qmax[None, :, None]
    for w in (3, 7, 25, 73):
        F[pre + f"u_roll{w}"] = _roll(u, w)
    F[pre + "u_lag1"] = _shift(u, 1)
    F[pre + "u_lead1"] = _shift(u, -1)
    F[pre + "u_week"] = bc(u.mean(-1, keepdims=True))
    F[pre + "p_week"] = bc(F[pre + "p_rel"].mean(-1, keepdims=True))
    day = np.arange(T) // 24
    dmean = np.stack([u[..., day == d].mean(-1) for d in range(7)], -1)
    F[pre + "u_day"] = dmean[..., day]
    length, since, until = _run_features(u)
    F[pre + "run_len"], F[pre + "run_since"], F[pre + "run_until"] = length, since, until
    pl_n = np.zeros_like(u)
    for pn in plants:
        m = np.array([pl == pn for pl in plant_of])
        pl_n[:, m, :] = u[:, m, :].sum(1, keepdims=True)
    F[pre + "plant_n_on"] = pl_n
    if lwv is not None:
        marg = price[:, None, :] - lwv
        F[pre + "lwv"] = lwv
        F[pre + "margin"] = marg
        F[pre + "margin_rel"] = marg / np.maximum(np.abs(lwv), 1.0)
        F[pre + "margin_rank"] = _rank(marg)
        F[pre + "margin_roll7"] = _roll(marg, 7)
        F[pre + "margin_roll25"] = _roll(marg, 25)
        F[pre + "lwv_week"] = bc(lwv.mean(-1, keepdims=True))

    # reservoir state from the optimiser, for the unit's own up/down reservoirs
    vu = np.zeros((N, U, T), np.float32)
    vd = np.zeros((N, U, T), np.float32)
    for j, (r_up, r_dn) in enumerate(updn):
        vu[:, j] = lc[f"v_{r_up}"].to_numpy(np.float32).reshape(N, T) / topo.res[r_up]["max_vol"]
        if r_dn:
            vd[:, j] = lc[f"v_{r_dn}"].to_numpy(np.float32).reshape(N, T) / topo.res[r_dn]["max_vol"]
    F[pre + "vup"] = vu
    F[pre + "vup_end"] = bc(vu[..., -1:])
    F[pre + "vup_min"] = bc(vu.min(-1, keepdims=True))
    F[pre + "vdn"] = vd
    F[pre + "vdn_max"] = bc(vd.max(-1, keepdims=True))

    # ---- ec_: economics from the inputs
    F["ec_price"] = bc(price[:, None, :])
    F["ec_price_rank"] = bc(_rank(price)[:, None, :])
    pz = (price - price.mean(-1, keepdims=True)) / (price.std(-1, keepdims=True) + 1e-3)
    F["ec_price_z"] = bc(pz[:, None, :])
    wvu_end = np.stack([wv_end[r].to_numpy(np.float32) for r, _ in updn], 1)
    wvu_0 = np.stack([wv_0[r].to_numpy(np.float32) for r, _ in updn], 1)
    wvd_end = np.stack([wv_end[r].to_numpy(np.float32) if r else np.zeros(N, np.float32)
                        for _, r in updn], 1)
    F["ec_wv_up_end"] = bc(wvu_end[..., None])
    F["ec_wv_up_change"] = bc((wvu_end - wvu_0)[..., None])
    F["ec_wv_dn_end"] = bc(wvd_end[..., None])
    F["ec_spread"] = price[:, None, :] - wvu_end[..., None]
    F["ec_spread_rel"] = F["ec_spread"] / np.maximum(wvu_end[..., None], 1.0)
    F["ec_hours_above"] = bc((price[:, None, :] > wvu_end[..., None]).mean(-1, keepdims=True))
    hod = (pd.DatetimeIndex(hours.ravel()).hour.to_numpy().reshape(N, T)).astype(np.float32)
    F["ec_hour_of_day"] = bc(hod[:, None, :])
    F["ec_t"] = bc(np.arange(T, dtype=np.float32)[None, None, :])
    doy = pd.DatetimeIndex(starts).dayofyear.to_numpy().astype(np.float32)
    F["ec_doy_sin"] = bc(np.sin(2 * np.pi * doy / 365.25)[:, None, None])
    F["ec_doy_cos"] = bc(np.cos(2 * np.pi * doy / 365.25)[:, None, None])
    v0u = np.stack([v_0[r].to_numpy(np.float32) / topo.res[r]["max_vol"] for r, _ in updn], 1)
    F["ec_v0_up"] = bc(v0u[..., None])
    inf_w = inflow.reindex(hours.ravel()).ffill()
    infu = np.stack([inf_w[r].to_numpy(np.float32).reshape(N, T).mean(-1) for r, _ in updn], 1)
    F["ec_inflow_up_rel"] = bc((infu / qmax[None, :])[..., None])

    # ---- cn_: constraint series shared across units
    mf = minflow.reindex(hours.ravel()).ffill().fillna(0.0)
    for c in mf.columns:
        F[f"cn_minflow_{c}"] = bc(mf[c].to_numpy(np.float32).reshape(N, 1, T))
    mv = minvol.reindex(hours.ravel()).ffill().fillna(0.0)
    for c in mv.columns:
        F[f"cn_minvol_{c}"] = bc((mv[c].to_numpy(np.float32).reshape(N, 1, T)
                                  / topo.res[c]["max_vol"]))

    # ---- st_: observed storage trajectory (post-decision proxy for the week's release)
    days = starts[:, None] + np.arange(8)[None, :] * np.timedelta64(1, "D")
    vd_obs = vol.reindex(days.ravel()).ffill()
    inf_d = inflow.resample("D").mean()
    inf_days = inf_d.reindex(days[:, :7].ravel()).ffill()
    rel_day = np.zeros((N, U, T), np.float32)
    dv_up = np.zeros((N, U), np.float32)
    dv_dn = np.zeros((N, U), np.float32)
    for j, (r_up, r_dn) in enumerate(updn):
        vv = vd_obs[r_up].to_numpy(np.float32).reshape(N, 8)
        ii = inf_days[r_up].to_numpy(np.float32).reshape(N, 7) if r_up in inf_days else 0.0
        # net outflow of the upstream reservoir that day, m3/s, relative to plant capacity
        out_d = (vv[:, :-1] - vv[:, 1:]) / (S.K * 24) + ii
        nsib_j = nsib[j]
        rel_day[:, j] = (out_d / (qmax[j] * nsib_j))[:, day]
        dv_up[:, j] = (vv[:, -1] - vv[:, 0]) / topo.res[r_up]["max_vol"]
        if r_dn:
            vdn = vd_obs[r_dn].to_numpy(np.float32).reshape(N, 8)
            dv_dn[:, j] = (vdn[:, -1] - vdn[:, 0]) / topo.res[r_dn]["max_vol"]
    F["st_release_day"] = rel_day
    F["st_release_week"] = bc(rel_day.mean(-1, keepdims=True))
    F["st_dv_up"] = bc(dv_up[..., None])
    F["st_dv_dn"] = bc(dv_dn[..., None])

    # ---- s_: static
    F["s_unit"] = bc(np.arange(U, dtype=np.float32)[None, :, None])
    F["s_plant"] = bc(np.array([plants.index(x) for x in plant_of], np.float32)[None, :, None])
    F["s_pmax"] = bc(pmax[None, :, None])
    F["s_pmin_frac"] = bc((pmin / pmax)[None, :, None])
    F["s_merit"] = bc(np.array([merit.get(x, 0) for x in units], np.float32)[None, :, None])

    names = list(F)
    meta = pd.DataFrame({
        "run_no": np.repeat(runs, U * T).astype(np.int32),
        "unit": np.tile(np.repeat(np.arange(U, dtype=np.int16), T), N),
        "hour": np.tile(np.arange(T, dtype=np.int16), N * U),
    })
    X = np.empty((N * U * T, len(names)), np.float32)
    for k, nm in enumerate(names):
        X[:, k] = np.asarray(F[nm], np.float32).reshape(-1)
        F[nm] = None
    if labels is not None:
        lab = labels.set_index("starttime") if "starttime" in labels.columns else labels
        y = np.full((N, U, T), np.nan, np.float32)
        have = pd.DatetimeIndex(starts).isin(lab.index)
        sub = lab.loc[pd.DatetimeIndex(starts)[have]]
        for j, nm in enumerate(units):
            y[have, j] = sub[[f"result_committed_{nm}_t{h}" for h in range(T)]].to_numpy(np.float32)
        meta["y"] = y.reshape(-1)
    return meta, X, names
