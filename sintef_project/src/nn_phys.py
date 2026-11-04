"""Track C: a sequence neural network (UCNet) on 83 base + 16 closed-form physics features.

Rebuilt from Anders Bærheim's specification (report/track_c_specification.md). Selected run: Kaggle
public 0.99024. No optimisation solver, no LightGBM, no blending.

    python nn_phys.py features        # build X4 (n_cases, 14, 168, 99), print checksums
    python nn_phys.py cv [year ...]   # expanding-window CV, one seed
    python nn_phys.py submit [out]    # 3 seeds on all labelled cases, mean of probabilities
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
H = 168
Q2V = 0.0036
UNITS = ["Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4", "Lio_G1", "Byrte_G1",
         "Vinje_G1", "Vinje_G2", "Vinje_G3", "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1"]
U = len(UNITS)
PLANT_OF = [u.rsplit("_", 1)[0] for u in UNITS]

PLANTS = {
    "Hogga":       dict(up=["Bandak"],                   down=None,         storage=["Bandak"]),
    "Tokke":       dict(up=["Vinjevatn"],                down="Bandak",     storage=["Vinjevatn", "Totak", "Vaamarvatn"]),
    "Lio":         dict(up=["Byrtevatn"],                down="Bandak",     storage=["Byrtevatn", "Botnedalsvatn"]),
    "Byrte":       dict(up=["Botnedalsvatn"],            down="Byrtevatn",  storage=["Botnedalsvatn"]),
    "Vinje":       dict(up=["Vaamarvatn"],               down="Vinjevatn",  storage=["Vaamarvatn", "Totak", "Venemo"]),
    "Haukeli":     dict(up=["Vatjern"],                  down="Vinjevatn",  storage=["Langeidvatn"]),
    "Songa":       dict(up=["Songavatn", "Bitdalsvatn"], down="Totak",      storage=["Songavatn", "Bitdalsvatn"]),
    "Kjela":       dict(up=["Foersvatn"],                down="Hyljelihyl", storage=["Foersvatn", "Bordalsvatn", "Langesae"]),
    "Vesle Kjela": dict(up=["Kjelavatn"],                down="Foersvatn",  storage=["Kjelavatn", "Staavatn"]),
}
RES_NEXT_PLANT = {"Bandak": "Hogga", "Vinjevatn": "Tokke", "Byrtevatn": "Lio", "Botnedalsvatn": "Byrte",
                  "Vaamarvatn": "Vinje", "Totak": "Vinje", "Venemo": "Vinje", "Hyljelihyl": "Vinje",
                  "Vatjern": "Haukeli", "Langeidvatn": "Haukeli", "Songavatn": "Songa",
                  "Bitdalsvatn": "Songa", "Foersvatn": "Kjela", "Bordalsvatn": "Kjela",
                  "Langesae": "Kjela", "Kjelavatn": "Vesle Kjela", "Staavatn": "Vesle Kjela"}
PLANT_TO_RES = {"Hogga": None, "Tokke": "Bandak", "Lio": "Bandak", "Byrte": "Byrtevatn",
                "Vinje": "Vinjevatn", "Haukeli": "Vinjevatn", "Songa": "Totak", "Kjela": "Venemo",
                "Vesle Kjela": "Foersvatn"}

# generator table: turbine curve at the plant head, minus losses, times generator efficiency,
# reduced to its concave envelope (derivation: specification, Appendix A)
GEN = {
    "Hogga":       dict(q0=33.72, p0=2.80783564998715, dq=[109.59, 25.29], slope=[0.10149342429890193, 0.09165437534151188], conv=0.09720535914384072),
    "Tokke":       dict(q0=12.8, p0=36.49549394378961, dq=[11.2, 8.0], slope=[4.0026121528933984, 3.313129037897717], conv=3.3885312523414863),
    "Lio":         dict(q0=5.6, p0=14.141421253321798, dq=[4.2, 4.2], slope=[3.5045463937930403, 2.6763506387522713], conv=2.9449506231890377),
    "Byrte":       dict(q0=3.2, p0=6.5446661254834035, dq=[2.4, 2.4], slope=[2.835237067711068, 2.1612044496302003], conv=2.3837919799982084),
    "Vinje":       dict(q0=22.0, p0=35.26560630253504, dq=[19.25, 13.75], slope=[2.2322633833435157, 1.8085301543845225], conv=1.8966467013793387),
    "Haukeli":     dict(q0=0.32, p0=0.2802197642385298, dq=[1.15, 0.63], slope=[2.5403253807377744, 1.9733216496339865], conv=2.1779550694469183),
    "Songa":       dict(q0=20.8, p0=41.55567700838549, dq=[18.2, 13.0], slope=[2.7921037972127514, 2.2843096196145445], conv=2.3685119517348094),
    "Kjela":       dict(q0=16.0, p0=18.1412658580685, dq=[14.0, 10.0], slope=[1.5546837627368437, 1.2368336508207853], conv=1.3302279512128103),
    "Vesle Kjela": dict(q0=8.0, p0=3.366750917186088, dq=[6.0, 6.0], slope=[0.5867579930451935, 0.44310961864109927], conv=0.4919499196755178),
}
FEED = {"Hogga": ["Bandak"], "Tokke": ["Vinjevatn"], "Lio": ["Byrtevatn"], "Byrte": ["Botnedalsvatn"],
        "Vesle Kjela": ["Kjelavatn", "Staavatn"], "Vinje": ["Venemo", "Hyljelihyl", "Totak", "Vaamarvatn"],
        "Kjela": ["Langesae", "Bordalsvatn", "Foersvatn"], "Songa": ["Songavatn", "Bitdalsvatn"],
        "Haukeli": ["Langeidvatn", "Vatjern"]}
EASTER = ["2015-04-05", "2016-03-27", "2017-04-16", "2018-04-01", "2019-04-21", "2020-04-12",
          "2021-04-04", "2022-04-17", "2023-04-09", "2024-03-31", "2025-04-20"]


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def _read(path):
    d = pd.read_csv(path)
    d.index = pd.to_datetime(d.pop("date"), utc=True)
    return d[~d.index.duplicated(keep="first")]


def load(root=ROOT):
    k, e = root / "data/kernel", root / "data/extended"
    D = dict(
        price=_read(k / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0].resample("1h").mean(),
        inflow=_read(k / "Historical_inflow_1958_2025.csv"),
        vol=_read(k / "Historical_volume_2015_2024.csv"),
        wv=_read(k / "Synthetic_water_value_2015_2024.csv"),
        minvol=_read(e / "Constraint_min_volume.csv"),
        minflow=_read(e / "Constraint_min_flow.csv"),
        topo=yaml.safe_load(open(e / "Tokke_Vinje_topology.yaml"))["model"],
    )
    lab = pd.read_csv(k / "Unit_commitment_decisions.csv")
    test = pd.read_csv(root / "prediction_mapping.csv")
    starts = pd.to_datetime(pd.concat([lab["starttime"], test["starttime"]]).to_numpy()).tz_localize("UTC")
    cols = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(H)]
    Y = lab[cols].to_numpy(np.uint8).reshape(len(lab), U, H)
    D.update(starts=starts, Y=Y, n_train=len(lab), test_runs=test["Run No"].to_numpy())
    return D


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _roll(a, w, fn):
    """Centred rolling window along the last (hour) axis."""
    shape = a.shape
    df = pd.DataFrame(a.reshape(-1, shape[-1]).T)
    r = getattr(df.rolling(w, center=True, min_periods=1), fn)()
    return r.to_numpy().T.reshape(shape)


def _rank(a, axis):
    return a.argsort(axis).argsort(axis) / (a.shape[axis] - 1)


def _b(a):
    """Broadcast (n,), (n,U), (n,H) or (n,U,H) to (n,U,H)."""
    a = np.asarray(a, dtype=np.float64)
    n = a.shape[0]
    if a.ndim == 1:
        return np.broadcast_to(a[:, None, None], (n, U, H))
    if a.ndim == 2 and a.shape[1] == U and a.shape[1] != H:
        return np.broadcast_to(a[:, :, None], (n, U, H))
    if a.ndim == 2:
        return np.broadcast_to(a[:, None, :], (n, U, H))
    return np.broadcast_to(a, (n, U, H))


def _hours(starts):
    return starts.values[:, None] + (np.arange(H) * np.timedelta64(1, "h"))[None, :]


def _hourly(series_or_frame, starts, fill=None):
    """(n, H) or (n, H, C) values at each case hour."""
    idx = pd.DatetimeIndex(_hours(starts).ravel(), tz="UTC")
    x = series_or_frame.reindex(idx)
    if fill == "ffill":
        x = x.ffill()
    if fill == "ffill0":
        x = x.ffill().fillna(0)
    if fill == "zero":
        x = x.fillna(0)
    v = x.to_numpy(np.float64)
    return v.reshape(len(starts), H, *v.shape[1:])


def _daily(frame, starts):
    """(n, 8, C) values at start + k days, k = 0..7, exactly (no interpolation)."""
    idx = pd.DatetimeIndex((starts.values[:, None] + (np.arange(8) * np.timedelta64(1, "D"))[None, :]).ravel(),
                           tz="UTC")
    return frame.reindex(idx).to_numpy(np.float64).reshape(len(starts), 8, frame.shape[1])


# ---------------------------------------------------------------------------------------------
# base features (83)
# ---------------------------------------------------------------------------------------------
def _head_and_glob(topo):
    res, plants = topo["reservoir"], topo["plant"]
    head = {p: (res[PLANTS[p]["up"][0]]["lrl"] + res[PLANTS[p]["up"][0]]["hrl"]) / 2
            - plants[p]["outlet_line"] for p in PLANTS}

    def E(r):
        if r is None:
            return 0.0
        p = RES_NEXT_PLANT[r]
        return head[p] + E(PLANT_TO_RES[p])
    glob = {r: E(r) for r in RES_NEXT_PLANT}
    return head, glob


def base_features(D, starts):
    topo = D["topo"]
    n = len(starts)
    R = list(D["wv"].columns)
    ri = {r: i for i, r in enumerate(R)}
    max_vol = np.array([topo["reservoir"][r]["max_vol"] for r in R], dtype=np.float64)
    head, glob = _head_and_glob(topo)

    P = _hourly(D["price"], starts, "ffill")                       # (n, H)
    WV = _daily(D["wv"], starts)                                   # (n, 8, R)
    VOL = _daily(D["vol"][R], starts)                              # (n, 8, R)
    infl_cols = list(D["inflow"].columns)
    INF = _hourly(D["inflow"], starts)                             # (n, H, 18)

    def up_mean(arr, p):           # arr (..., R)
        return np.mean([arr[..., ri[r]] for r in PLANTS[p]["up"]], 0)

    def stor_sum(arr, p):
        return np.sum([arr[..., ri[r]] for r in PLANTS[p]["storage"]], 0)

    def per_unit(fn):              # fn(plant) -> (n, ...) ; stack over units on axis 1
        return np.stack([fn(p) for p in PLANT_OF], 1)

    F = {}
    wv_end_up = per_unit(lambda p: up_mean(WV[:, 7], p))           # (n, U)
    wv_start_up = per_unit(lambda p: up_mean(WV[:, 0], p))
    margin = P[:, None, :] - wv_end_up[:, :, None]                 # (n, U, H)

    # ---- economic core
    F["margin_end"] = margin
    F["margin_start"] = P[:, None, :] - wv_start_up[:, :, None]
    F["margin_frac_pos"] = (margin > 0).mean(-1)
    F["margin_week_mean"] = margin.mean(-1)
    F["margin_rel"] = P[:, None, :] / (np.abs(wv_end_up)[:, :, None] + 1)
    F["margin_day_mean"] = np.repeat(margin.reshape(n, U, 7, 24).mean(-1), 24, -1)
    F["margin_roll12"] = _roll(margin, 12, "mean")
    F["margin_roll24_min"] = _roll(margin, 24, "min")
    F["margin_roll24_max"] = _roll(margin, 24, "max")

    # ---- price
    p_mean = P.mean(1, keepdims=True)
    p_std = P.std(1, keepdims=True) + 1e-3
    day = P.reshape(n, 7, 24)
    dmin = np.repeat(day.min(-1), 24, -1)
    dmax = np.repeat(day.max(-1), 24, -1)
    F["price"] = P
    F["price_z"] = (P - p_mean) / p_std
    F["price_rank"] = _rank(P, 1)
    F["price_day_rank"] = _rank(day, 2).reshape(n, H)
    F["price_day_pos"] = (P - dmin) / (dmax - dmin + 1e-3)
    F["price_day_spread"] = dmax - dmin
    F["price_roll6"] = _roll(P, 6, "mean")
    F["price_roll24"] = _roll(P, 24, "mean")
    F["price_roll48_rank"] = (P - _roll(P, 48, "mean")) / (_roll(P, 48, "std") + 1e-3)
    for lag in (1, 3, 6):
        F[f"price_lag{lag}"] = np.concatenate([np.repeat(P[:, :1], lag, 1), P[:, :-lag]], 1) - P
        F[f"price_lead{lag}"] = np.concatenate([P[:, lag:], np.repeat(P[:, -1:], lag, 1)], 1) - P
    rem = np.cumsum(P[:, ::-1], 1)[:, ::-1] / np.arange(H, 0, -1)[None, :]
    F["price_minus_rem_mean"] = P - rem
    F["p_week_mean"] = p_mean[:, 0]
    F["p_week_std"] = p_std[:, 0]

    # ---- water value
    def down_col(arr, p):          # arr (n, R) -> (n,), 0 if no downstream reservoir
        d = PLANTS[p]["down"]
        return arr[:, ri[d]] if d else np.zeros(n)
    F["wv_end_up"] = wv_end_up
    F["wv_start_up"] = wv_start_up
    F["wv_change_up"] = wv_end_up - wv_start_up
    F["wv_end_over_pmean"] = wv_end_up / (P.mean(1)[:, None] + 1)
    F["wv_end_down"] = per_unit(lambda p: down_col(WV[:, 7], p))
    be = per_unit(lambda p: (up_mean(WV[:, 7], p) * glob[PLANTS[p]["up"][0]]
                             - (down_col(WV[:, 7], p) * glob[PLANTS[p]["down"]] if PLANTS[p]["down"] else 0))
                  / head[p])
    F["margin_global"] = P[:, None, :] - be[:, :, None]
    wv_path = per_unit(lambda p: up_mean(WV, p))                   # (n, U, 8)
    F["wv_path_daily_slope"] = np.diff(wv_path, axis=-1).std(-1)
    F["margin_daily_wv"] = P[:, None, :] - np.repeat(wv_path[:, :, 1:8], 24, -1)
    F["wv_sys_mean_end"] = WV[:, 7].mean(1)

    # ---- state
    fill = VOL[:, 0] / max_vol                                      # (n, R)
    F["fill_up"] = per_unit(lambda p: up_mean(fill, p))
    F["fill_storage"] = per_unit(lambda p: stor_sum(VOL[:, 0], p) / stor_sum(max_vol, p))
    F["fill_down"] = per_unit(lambda p: down_col(fill, p))
    F["fill_sys"] = VOL[:, 0].sum(1) / max_vol.sum()
    for r in ("Totak", "Songavatn", "Bandak", "Vaamarvatn", "Vinjevatn"):
        F[f"fill_{r}"] = fill[:, ri[r]]

    # ---- inflow
    wk = INF.sum(1) * 3600 / 1e6                                    # (n, 18)
    ii = {c: i for i, c in enumerate(infl_cols)}

    def stor_infl(p):
        return np.sum([wk[:, ii[r]] for r in PLANTS[p]["storage"] if r in ii], 0)
    F["inflow_storage_wk"] = per_unit(stor_infl)
    F["inflow_up_wk_rel"] = per_unit(lambda p: stor_infl(p) / (stor_sum(max_vol, p) + 1e-3))
    F["inflow_sys_wk"] = wk.sum(axis=1)
    hist = D["inflow"][D["inflow"].index < pd.Timestamp("2015-01-01", tz="UTC")].sum(axis=1)
    daily = hist.resample("1D").sum() * 3600 / 1e6
    fwd = daily.rolling(7).sum().shift(-6)
    clim = fwd.groupby(fwd.index.dayofyear).mean().reindex(range(1, 366)).ffill().to_numpy()
    doy = np.clip(starts.dayofyear.to_numpy(), 1, 365)
    F["inflow_sys_anom"] = wk.sum(1) / clim[doy - 1]
    F["inflow_sys_h"] = INF.sum(2)

    # ---- constraints
    MF = _hourly(D["minflow"], starts, "zero")
    for i, c in enumerate(["b_Kjelavatn", "b_Hyljelihyl", "b_Vinjevatn", "b_Byrtevatn", "b_Bandak",
                           "r_Vest_Vassdraget"]):
        F[f"minflow_{c}"] = MF[:, :, list(D["minflow"].columns).index(c)]
    MV = _hourly(D["minvol"], starts, "zero")
    for r in ("Totak", "Byrtevatn", "Staavatn"):
        F[f"minvol_gap_{r}"] = VOL[:, 0, ri[r]][:, None] - MV[:, :, list(D["minvol"].columns).index(r)]

    # ---- future volume path (observed storage, available for test cases too)
    dV = np.diff(VOL, axis=1)                                       # (n, 7, R)
    inf_day = INF.reshape(n, 7, 24, -1).sum(2) * 3600 / 1e6         # (n, 7, 18)
    INF_day = np.zeros((n, 7, len(R)))
    for r in R:
        if r in ii:
            INF_day[:, :, ri[r]] = inf_day[:, :, ii[r]]
    rel = INF_day - dV

    def per_plant(arr, which):     # (n, 7, R) -> (n, U, 7)
        return per_unit(lambda p: np.sum([arr[:, :, ri[r]] for r in
                                          (PLANTS[p][which] if isinstance(PLANTS[p][which], list)
                                           else [PLANTS[p][which]])], 0))
    dVs, relu, rels = per_plant(dV, "storage"), per_plant(rel, "up"), per_plant(rel, "storage")
    F["fv_dV_storage_wk"] = dVs.sum(-1)
    F["fv_dV_storage_day"] = np.repeat(dVs, 24, -1)
    F["fv_release_up_day"] = np.repeat(relu, 24, -1)
    F["fv_release_up_wk"] = relu.sum(-1)
    F["fv_release_storage_day"] = np.repeat(rels, 24, -1)
    F["fv_release_storage_wk"] = rels.sum(-1)
    F["fv_dV_down_wk"] = per_unit(lambda p: dV[:, :, ri[PLANTS[p]["down"]]].sum(axis=1)
                                  if PLANTS[p]["down"] else np.zeros(n))
    F["fv_dV_sys_wk"] = dV.sum((1, 2))

    # ---- unit (static)
    gens = topo["generator"]
    F["unit_id"] = np.broadcast_to(np.arange(U, dtype=np.float64), (n, U))
    for k, key in (("p_min", "p_min"), ("p_max", "p_max"), ("startcost", "startcost_const")):
        F[k] = np.broadcast_to(np.array([gens[u].get(key, 0) for u in UNITS], np.float64), (n, U))
    F["unit_idx_in_plant"] = np.broadcast_to(np.array([float(u.rsplit("_G", 1)[1]) for u in UNITS]), (n, U))
    F["n_units_plant"] = np.broadcast_to(np.array([PLANT_OF.count(p) for p in PLANT_OF], np.float64), (n, U))
    F["head"] = np.broadcast_to(np.array([head[p] for p in PLANT_OF]), (n, U))
    F["t"] = np.broadcast_to(np.arange(H, dtype=np.float64), (n, H))

    # ---- calendar (Europe/Oslo)
    loc = pd.DatetimeIndex(_hours(starts).ravel(), tz="UTC").tz_convert("Europe/Oslo")
    F["hod"] = loc.hour.to_numpy().reshape(n, H).astype(np.float64)
    F["dow"] = loc.dayofweek.to_numpy().reshape(n, H).astype(np.float64)
    sdoy = starts.dayofyear.to_numpy()
    F["doy_sin"] = np.sin(2 * np.pi * sdoy / 365.25)
    F["doy_cos"] = np.cos(2 * np.pi * sdoy / 365.25)
    hol = set()
    for e in EASTER:
        e = pd.Timestamp(e)
        hol |= {(e + pd.Timedelta(days=d)).date() for d in (-3, -2, 0, 1, 39, 49, 50)}
    for y in range(2015, 2026):
        hol |= {pd.Timestamp(f"{y}-{md}").date() for md in
                ("01-01", "05-01", "05-17", "12-24", "12-25", "12-26", "12-31")}
    F["holiday"] = np.isin(np.array(loc.date), list(hol)).reshape(n, H).astype(np.float64)

    # ---- plant context
    sysm = margin.mean(1, keepdims=True)
    F["margin_sys_mean"] = sysm[:, 0, :]
    F["margin_minus_sys"] = margin - sysm
    F["margin_rank_across_units"] = _rank(margin, 1)
    assert len(F) == 83, len(F)
    return F


# ---------------------------------------------------------------------------------------------
# physics features (16): closed form from the generator curves, no solver
# ---------------------------------------------------------------------------------------------
def _runs(mask, x):
    """For each position of a boolean (m, T) mask: length and sum of x over its run of equal values."""
    m, T = mask.shape
    fl, fs, bl, bs = (np.zeros((m, T)) for _ in range(4))
    for t in range(T):
        same = mask[:, t] == mask[:, t - 1] if t else np.zeros(m, bool)
        fl[:, t] = np.where(same, fl[:, t - 1] + 1, 1) if t else 1
        fs[:, t] = np.where(same, fs[:, t - 1] + x[:, t], x[:, t]) if t else x[:, t]
    for t in range(T - 1, -1, -1):
        same = mask[:, t] == mask[:, t + 1] if t < T - 1 else np.zeros(m, bool)
        bl[:, t] = np.where(same, bl[:, t + 1] + 1, 1) if t < T - 1 else 1
        bs[:, t] = np.where(same, bs[:, t + 1] + x[:, t], x[:, t]) if t < T - 1 else x[:, t]
    return fl + bl - 1, fs + bs - x


def physics_features(D, starts):
    topo = D["topo"]
    n = len(starts)
    P = _hourly(D["price"], starts, "ffill")[:, None, :]            # (n, 1, H)
    WV = _daily(D["wv"], starts)
    VOL = _daily(D["vol"][list(D["wv"].columns)], starts)
    ri = {r: i for i, r in enumerate(D["wv"].columns)}
    WV7 = {r: WV[:, 7, i] for r, i in ri.items()}
    V0 = {r: VOL[:, 0, i] for r, i in ri.items()}
    INF = _hourly(D["inflow"], starts, "ffill0")
    inflow = {c: INF[:, :, i] for i, c in enumerate(D["inflow"].columns)}
    MV = _hourly(D["minvol"], starts)
    min_vol = {c: MV[:, :, i] for i, c in enumerate(D["minvol"].columns)}
    max_vol = {r: topo["reservoir"][r]["max_vol"] for r in ri}
    gens = [dict(GEN[p], plant=p, p_min=topo["generator"][u]["p_min"], p_max=topo["generator"][u]["p_max"],
                 start=topo["generator"][u].get("startcost_const", 0)) for u, p in zip(UNITS, PLANT_OF)]
    plants = PLANT_OF
    intake_res = {p: PLANTS[p]["up"] for p in PLANTS}
    F = {}

    def bc(a):
        return np.broadcast_to(a, (n, U, H))

    conv_best = {p: max(g["conv"] for g in gens if g["plant"] == p) for p in plants}
    wv = np.stack([np.mean([WV7[r] for r in intake_res[p]], 0) for p in plants], 1)
    q_full = np.array([g["q0"] + sum(g["dq"]) for g in gens])
    p_full = np.array([g["p0"] + np.dot(g["slope"], g["dq"]) for g in gens])

    def rel(mw_per_q):
        return np.array([conv_best[p] for p in plants]) / mw_per_q
    p0q0 = np.array([g["p0"] / g["q0"] for g in gens])
    be_min = wv * rel(p0q0)
    be_full = wv * rel(p_full / q_full)
    be_last = wv * rel(np.array([g["slope"][-1] for g in gens]))
    F["phys_be_rel_min"] = bc(rel(p0q0)[None, :, None])
    F["phys_m_min"] = P - be_min[..., None]
    F["phys_m_full"] = P - be_full[..., None]
    F["phys_m_last"] = P - be_last[..., None]
    pmax, pmin, start = (np.array([g[k] for g in gens], np.float64)[None, :, None]
                         for k in ("p_max", "p_min", "start"))
    F["phys_start_per_mw"] = bc(start / pmax)
    m_full = F["phys_m_full"]

    Qp = {p: sum(q_full[k] for k in range(U) if plants[k] == p) * Q2V for p in plants}

    def infl(rs):
        return sum(inflow[r].sum(1) for r in rs if r in inflow) * Q2V

    def minv(rs):
        return sum(min_vol[r][:, 0] for r in rs if r in min_vol)
    afford, forced = np.zeros((n, U)), np.zeros((n, U))
    for p in dict.fromkeys(plants):
        ks = [k for k in range(U) if plants[k] == p]
        rs, ins = FEED[p], intake_res[p]
        afford[:, ks] = (np.maximum(sum(V0[r] for r in rs) - minv(rs), 0) + infl(rs))[:, None] / Qp[p]
        forced[:, ks] = np.maximum(sum(V0[r] for r in ins) + infl(ins)
                                   - sum(max_vol[r] for r in ins), 0)[:, None] / Qp[p]
    afford, forced = np.clip(afford, 0, 10 * H), np.clip(forced, 0, H)
    npos = (m_full > 0).sum(-1)
    F["phys_afford_h"] = bc(np.minimum(afford, H)[..., None] / H)
    F["phys_afford_log"] = bc(np.log1p(afford)[..., None])
    F["phys_forced_h"] = bc(forced[..., None] / H)
    F["phys_npos_h"] = bc(npos[..., None] / H)

    k_on = np.clip(npos, forced, np.minimum(afford, H))
    rank = (-m_full).argsort(-1).argsort(-1)
    F["phys_greedy_k"] = bc(k_on[..., None] / H)
    F["phys_greedy_on"] = (rank < k_on[..., None]).astype(np.float64)
    F["phys_greedy_dist"] = (k_on[..., None] - rank) / H
    F["phys_forced_on"] = (rank < forced[..., None]).astype(np.float64)

    pos = (m_full > 0).reshape(n * U, H)
    L, S = _runs(pos, (m_full * pmax).reshape(n * U, H))
    Lg, Sg = _runs(pos, (np.minimum(F["phys_m_min"], 0) * pmin).reshape(n * U, H))
    st = np.broadcast_to(start, (n, U, H)).reshape(n * U, H)
    F["phys_block_len"] = np.where(pos, L, -Lg).reshape(n, U, H) / 24
    F["phys_block_profit"] = np.where(pos, S - st, 0).reshape(n, U, H)
    F["phys_stay_on_gain"] = np.where(pos, 0, st + Sg).reshape(n, U, H)
    assert len(F) == 16
    return F


def build_X4(D, chunk=256, out=None):
    """(n_cases, 14, 168, 99) float32, built in chunks of cases."""
    starts = D["starts"]
    n = len(starts)
    X = out if out is not None else np.empty((n, U, H, 99), np.float32)
    names = None
    for a in range(0, n, chunk):
        s = starts[a:a + chunk]
        F = {**base_features(D, s), **physics_features(D, s)}
        names = list(F)
        for j, k in enumerate(names):
            X[a:a + len(s), :, :, j] = _b(F[k])
    assert np.isfinite(X).all(), "non-finite features: a reindex is misaligned"
    return X, names


# ---------------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------------
class Scaler:
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        idx = np.random.default_rng(0).choice(len(flat), min(len(flat), 400_000), replace=False)
        s = np.asarray(flat[np.sort(idx)], dtype=np.float64)
        self.med = np.median(s, 0)
        q = np.percentile(s, [10, 90], axis=0)
        self.scale = np.where(q[1] - q[0] > 1e-6, q[1] - q[0], s.std(0) + 1e-6)
        return self

    def __call__(self, X, chunk=64):
        out = np.empty(X.shape, np.float32)
        for a in range(0, len(X), chunk):
            out[a:a + chunk] = np.clip((X[a:a + chunk] - self.med) / self.scale, -8, 8)
        return out


def make_model(n_feat=99, d=96):
    import torch.nn as nn
    import torch

    class UCNet(nn.Module):
        def __init__(self, n_feat, n_units, d=96):
            super().__init__()
            self.unit_emb = nn.Embedding(n_units, 16)
            self.inp = nn.Sequential(nn.Linear(n_feat + 16, d), nn.GELU(), nn.Linear(d, d))
            self.conv = nn.Sequential(
                nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU())
            self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
            self.unit_attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
            self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

        def forward(self, x):
            B, Un, T, _ = x.shape
            e = self.unit_emb(torch.arange(Un, device=x.device))[None, :, None, :].expand(B, Un, T, -1)
            h = self.inp(torch.cat([x, e], -1)).reshape(B * Un, T, -1)
            h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
            h = self.norm1(h + self.gru(h)[0])
            hu = h.reshape(B, Un, T, -1).permute(0, 2, 1, 3).reshape(B * T, Un, -1)
            hu = self.norm2(hu + self.unit_attn(hu, hu, hu, need_weights=False)[0])
            h = hu.reshape(B, T, Un, -1).permute(0, 2, 1, 3)
            return self.head(h).squeeze(-1)

    return UCNet(n_feat, U, d)


def device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_predict(Xtr, Ytr, Xte, seed, epochs=20, bs=32, lr=2e-3, verbose=True, Yva=None, dev=None):
    import torch
    from torch.nn import BCEWithLogitsLoss
    from torch.nn.utils import clip_grad_norm_
    dev = dev or device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr.astype(np.float32))
    model = make_model(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * math.ceil(len(Xtr_t) / bs),
                                                pct_start=0.15)
    lossf = BCEWithLogitsLoss()
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), bs):
            b = perm[i:i + bs]
            loss = lossf(model(Xtr_t[b].to(dev)), Ytr_t[b].to(dev))
            opt.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        msg = f"    seed {seed} epoch {ep + 1:>2}/{epochs}  loss {tot / len(Xtr_t):.4f}  {time.time() - t0:.0f}s"
        if Yva is not None and (ep + 1) % 4 == 0:
            from sklearn.metrics import roc_auc_score
            msg += f"  val AUC {roc_auc_score(Yva.ravel(), predict(model, Xte, dev).ravel()):.4f}"
        if verbose:
            print(msg, flush=True)
    return predict(model, Xte, dev)


def predict(model, X, dev, bs=64):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[a:a + bs]).to(dev))).cpu().numpy())
    return np.concatenate(out)


def write_submission(P, test_runs, out_path, root=ROOT):
    """P (725, 14, 168) -> wide CSV in the sample submission's column order."""
    sample = pd.read_csv(root / "sample_submission.csv", nrows=1)
    cols = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(H)]
    out = pd.DataFrame(P.reshape(len(P), U * H), columns=cols)
    out.insert(0, "Run No", test_runs)
    assert list(out.columns) == list(sample.columns), "column mismatch against sample_submission"
    assert out.shape == (725, 2353) and np.isfinite(out.iloc[:, 1:].to_numpy()).all()
    assert out.iloc[:, 1:].to_numpy().min() >= 0 and out.iloc[:, 1:].to_numpy().max() <= 1
    out.to_csv(out_path, index=False)
    return out


# ---------------------------------------------------------------------------------------------
def checksums(X, names, n_train):
    """Mean and SD on every 7th training case, mean on every 3rd test case (specification, App. B)."""
    rows = []
    for j, f in enumerate(names):
        tr = X[:n_train:7, :, :, j].astype(np.float64)
        te = X[n_train::3, :, :, j].astype(np.float64)
        rows.append(dict(feature=f, train_mean=tr.mean(), train_std=tr.std(), test_mean=te.mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "features"
    t0 = time.time()
    D = load()
    X, names = build_X4(D)
    n = D["n_train"]
    print(f"X4 {X.shape} in {time.time() - t0:.0f}s", flush=True)
    if cmd == "features":
        c = checksums(X, names, n)
        c.to_csv(ROOT / "report/runs/nn_phys_feature_checksums.csv", index=False)
        print(c.to_string())
        k = names.index
        print("spot", {f: round(float(X[0, 1, 10, k(f)]), 6) for f in
                       ("price", "wv_end_up", "margin_end", "fill_up", "inflow_sys_anom", "hod",
                        "phys_m_full", "phys_afford_h", "phys_greedy_k", "phys_block_profit")})
    elif cmd == "cv":
        from sklearn.metrics import roc_auc_score
        years = [int(y) for y in sys.argv[2:]] or [2019, 2020, 2021, 2022]
        st = D["starts"][:n]
        rows = []
        for y in years:
            tr = np.where(st < pd.Timestamp(f"{y}-01-01", tz="UTC") - pd.Timedelta(days=7))[0]
            va = np.where(st.year == y)[0]
            sc = Scaler().fit(X[tr[::3]])
            p = train_predict(sc(X[tr]), D["Y"][tr], sc(X[va]), seed=0, Yva=D["Y"][va])
            rows.append(dict(year=y, auc=roc_auc_score(D["Y"][va].ravel(), p.ravel())))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(ROOT / "report/runs/nn_phys_cv.csv", index=False)
    elif cmd == "submit":
        out = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "submission_nn_phys_rerun.csv")
        sc = Scaler().fit(X[:n:3])
        Xtr, Xte = sc(X[:n]), sc(X[n:])
        del X
        P = np.mean([train_predict(Xtr, D["Y"], Xte, seed=s) for s in (0, 1, 2)], 0)
        write_submission(P, D["test_runs"], out)
        print(f"wrote {out}, mean probability {P.mean():.4f}", flush=True)
