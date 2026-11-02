"""
Production / water-balance features for the count model.

Rationale. The count is hours-on per unit-week, and hours-on x capacity = energy produced =
water released x conversion. So anything that estimates released water estimates the count
*directly*, without learning it. The crude system-level proxy `storage_next_minus_prev` is
already the single strongest count feature (log 021); this module builds the attributed version.

The organising principle is that SHOP resolves

    count  ~  min( hours the available water can pay for ,
                   hours the price makes profitable )

so we measure both limits, their ratio, and which one binds.

Everything here is derived from data available for the test period (volume, inflow, price, water
value, topology). Nothing uses past targets.
"""
from __future__ import annotations

from pathlib import Path

import warnings
import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
K = _ROOT / "data/kernel"
_topo = None


def topo():
    global _topo
    if _topo is None:
        _topo = yaml.safe_load(open(_ROOT / "data/extended/Tokke_Vinje_topology.yaml"))
    return _topo


def san(s):
    return s.replace(" ", "_")


def _daily(path, how="mean"):
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    d = d.set_index("date").sort_index()
    return d.resample("D").mean() if how == "mean" else d.resample("D").sum()


def _through_tunnels(c, node, reservoirs, depth=6):
    """Reservoirs feeding `node`, walking back through any chain of tunnel hops.

    The topology connects some reservoirs to plants DIRECTLY and others through a named tunnel:

        Vinjevatn -------------------------------> Tokke     (direct)
        Songavatn --> Songavatn_Songa -----------> Songa      (via tunnel)

    Matching only direct reservoir->plant edges therefore misses every tunnel-fed plant. That is
    not hypothetical: it left `src["Songa"] = None`, so every `own_*` feature for Songa was NaN
    and `mwh_per_m3("Songa", .)` returned NaN -- while Songavatn carries the largest storage
    variance of any reservoir in the system (sd 150) and Bitdalsvatn adds another (sd 25). Log
    027 diagnosed Songa as "tunnel-fed" and attacked it with water-balance corrections; the
    actual defect was here, in the graph.
    """
    out, seen, frontier = set(), set(), {node}
    for _ in range(depth):
        nxt = set()
        for x in c:
            if x["to"] in frontier and x["from"] not in seen:
                seen.add(x["from"])
                if x["from"] in reservoirs:
                    out.add(x["from"])
                elif x["from_type"] == "tunnel":
                    nxt.add(x["from"])
        frontier = nxt
        if not frontier:
            break
    return out


def plant_sources():
    """Every source reservoir of every plant, tunnels resolved. Songa has two."""
    t = topo(); c = t["connections"]; res = set(t["model"]["reservoir"])
    return {p: sorted(_through_tunnels(c, p, res)) for p in t["model"]["plant"]}


def reservoir_tunnels():
    """Reservoir-to-reservoir tunnel links, invisible to a direct-edge-only traversal.

    Totak->Vaamarvatn, Bordalsvatn->Foersvatn and Venemo->Totak are real transfers carrying water
    into a plant's source reservoir. Missing them understates upstream availability for Vinje
    (the entire upper cascade, including Songa's discharge) and Kjela (Bordalsvatn, sd 43).
    """
    t = topo(); c = t["connections"]; res = set(t["model"]["reservoir"])
    out = []
    for x in c:
        if x["from_type"] == "reservoir" and x["to_type"] == "tunnel":
            out += [(x["from"], y["to"]) for y in c if y["from"] == x["to"] and y["to"] in res]
    return out


def effective_upstream_plants(plant):
    """Plants whose discharge reaches `plant`'s source reservoir, including via tunnel.

    `unit_relative` identifies upstream plants by exact match -- a plant q is upstream of p if
    `dst[q] == src[p]`. With the tunnel links restored that test is too strict. Songa discharges
    into Totak, and Totak tunnels into Vaamarvatn, which is Vinje's source: Songa IS upstream of
    Vinje, and its production is Vinje's water, but an exact match never sees it.
    """
    src, dst, _ = plant_graph()
    tun = reservoir_tunnels()
    target = src.get(plant)
    if target is None:
        return []
    # reservoirs that drain into this plant's source, directly or through a tunnel chain
    feeds = {target}
    for _ in range(4):
        grown = feeds | {a for a, b in tun if b in feeds}
        if grown == feeds:
            break
        feeds = grown
    return [q for q, d in dst.items() if d in feeds and q != plant]


def plant_graph():
    """source reservoir, discharge reservoir, and upstream set for each plant.

    `src` keeps its single-reservoir contract for existing callers, but is now resolved through
    tunnels; where a plant has several sources the largest-capacity one is primary and
    `plant_sources()` exposes the rest.
    """
    t = topo(); c = t["connections"]; res = set(t["model"]["reservoir"])
    direct = {x["to"]: x["from"] for x in c
              if x["to_type"] == "plant" and x["from_type"] == "reservoir"}
    allsrc = plant_sources()

    def cap(r):
        vh = t["model"]["reservoir"][r].get("vol_head", {})
        return max(vh.get("x", [0.0])) if vh else 0.0

    src = dict(direct)
    for p in t["model"]["plant"]:
        if p not in src and allsrc.get(p):
            src[p] = max(allsrc[p], key=cap)
    dst = {x["from"]: x["to"] for x in c if x["from_type"] == "plant" and x["to_type"] == "reservoir"}
    up = {}
    for p in t["model"]["plant"]:
        seen, frontier = set(), {p}
        for _ in range(4):
            nxt = set()
            for x in c:
                if x["to"] in frontier and x["from"] not in seen:
                    seen.add(x["from"]); nxt.add(x["from"])
            frontier = nxt
        up[p] = {r for r in seen if r in t["model"]["reservoir"]}
    return src, dst, up


def mwh_per_m3(plant, storage_value):
    """Head-derived conversion: storage -> level -> head above outlet -> energy per m3."""
    t = topo()
    src, _, _ = plant_graph()
    r = src.get(plant)
    if r is None:
        return np.nan
    spec = t["model"]["reservoir"][r]; vh = spec["vol_head"]
    level = np.interp(storage_value, vh["x"], vh["y"])
    head = level - t["model"]["plant"][plant]["outlet_line"]
    gens = [g for g in t["model"]["generator"] if san(g).startswith(san(plant) + "_G")]
    eff = 85.0
    if gens:
        curves = t["model"]["generator"][gens[0]].get("turb_eff_curves", [])
        if curves:
            refs = np.array([c["ref"] for c in curves], float)
            peaks = np.array([max(c["y"]) for c in curves], float)
            o = np.argsort(refs); eff = np.interp(head, refs[o], peaks[o])
    return 1000 * 9.81 * head * (eff / 100.0) / 3.6e9


def build(at_dates: pd.Series) -> pd.DataFrame:
    """One row per case; columns are per-plant water-balance and binding-constraint features."""
    price = _daily(K / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    wv = _daily(K / "Synthetic_water_value_2015_2024.csv")
    vol = _daily(K / "Historical_volume_2015_2024.csv")
    inf_d = _daily(K / "Historical_inflow_1958_2025.csv") * 3600 * 24 / 1e6   # m3/s -> Mm3/day
    src, dst, up = plant_graph()
    t = topo(); gens = t["model"]["generator"]; res_spec = t["model"]["reservoir"]

    d = pd.to_datetime(pd.Series(at_dates)).reset_index(drop=True)
    d_end = d + pd.Timedelta(days=7)
    out = pd.DataFrame(index=range(len(d)))

    def at(s, when):
        return s.reindex(when).to_numpy()

    def win_sum(s, a, b):
        c = s.cumsum()
        return c.reindex(b).to_numpy() - c.reindex(a).to_numpy()

    # ---------- system-level balance (the crude proxy that already wins) ----------
    sys_stor = vol.sum(axis=1)
    sys_inf = inf_d[[c for c in inf_d.columns if c in vol.columns]].sum(axis=1)
    out["sys_drawdown"] = at(sys_stor, d) - at(sys_stor, d_end)
    out["sys_inflow_wk"] = win_sum(sys_inf, d, d_end)
    out["sys_release"] = out["sys_drawdown"] + out["sys_inflow_wk"]
    out["sys_release_over_inflow"] = out["sys_release"] / out["sys_inflow_wk"].replace(0, np.nan)
    out["sys_drawdown_frac"] = out["sys_drawdown"] / at(sys_stor, d)

    for plant, r in src.items():
        pl = san(plant)
        if r not in vol.columns:
            continue
        stor = vol[r]; inflow = inf_d[r] if r in inf_d.columns else pd.Series(0.0, index=vol.index)
        s0, s1 = at(stor, d), at(stor, d_end)
        inflow_wk = win_sum(inflow, d, d_end)

        # upstream plants discharge INTO this reservoir -- the term the naive attempt omitted
        up_disch = np.zeros(len(d))
        for p2, dr in dst.items():
            if dr == r and p2 != plant and src.get(p2) in vol.columns:
                s2 = vol[src[p2]]
                i2 = inf_d[src[p2]] if src[p2] in inf_d.columns else pd.Series(0.0, index=vol.index)
                up_disch += (at(s2, d) - at(s2, d_end)) + win_sum(i2, d, d_end)

        release = (s0 - s1) + inflow_wk + up_disch
        conv = mwh_per_m3(plant, s0)
        cap = sum(gens[g]["p_max"] for g in gens if san(g).startswith(pl + "_G")) or np.nan
        n_units = sum(1 for g in gens if san(g).startswith(pl + "_G"))
        energy = release * 1e6 * conv                       # MWh released this week

        out[f"drawdown_{pl}"] = s0 - s1
        out[f"release_{pl}"] = release
        out[f"upstream_discharge_{pl}"] = up_disch
        out[f"energy_mwh_{pl}"] = energy
        # implied hours at full output -- a DIRECT physical estimate of the count
        out[f"implied_hours_{pl}"] = energy / cap
        out[f"capacity_factor_{pl}"] = energy / (cap * 168.0)
        out[f"implied_unit_hours_{pl}"] = energy / (cap / max(n_units, 1))

        # normalisations
        mv = res_spec[r]["max_vol"]
        out[f"stor_frac_{pl}"] = s0 / mv
        out[f"stor_frac_end_{pl}"] = s1 / mv
        out[f"release_over_maxvol_{pl}"] = release / mv
        out[f"release_over_inflow_{pl}"] = release / np.where(inflow_wk == 0, np.nan, inflow_wk)
        out[f"inflow_over_maxvol_{pl}"] = inflow_wk / mv

        # ---------- the two limits, and which one binds ----------
        own_wv = wv[r].reindex(price.index).ffill() if r in wv.columns else wv.mean(axis=1)
        spread = price - own_wv
        prof_hours = (spread > 0).astype(float)
        hours_profitable = win_sum(prof_hours, d, d_end) * 24     # daily series -> hours
        water_hours = out[f"implied_hours_{pl}"]
        out[f"hours_profitable_{pl}"] = hours_profitable
        out[f"hours_water_{pl}"] = water_hours
        out[f"binding_min_{pl}"] = np.minimum(hours_profitable, water_hours)
        out[f"water_over_profitable_{pl}"] = water_hours / np.where(hours_profitable == 0, np.nan,
                                                                     hours_profitable)
        out[f"water_is_binding_{pl}"] = (water_hours < hours_profitable).astype(float)
        out[f"mean_spread_{pl}"] = win_sum(spread.fillna(0), d, d_end) / 7.0
    return out


# ---------------------------------------------------------------- water-value trace
# Water values are SHOP *outputs* - shadow prices on stored water. Generating depletes the
# reservoir and raises the marginal value of what remains, so week-over-week water-value change
# is a post-decision trace of the schedule, through a different channel than physical storage.
# Measured: corr(wv change, storage drawdown) = +0.068, i.e. the two are near-orthogonal, and
# water value is the stronger system-level signal (-0.215 vs -0.095). It also covers Vatjern and
# Hyljelihyl, whose storage series are constant and therefore give no trace at all (log 008).

def water_value_trace(at_dates: pd.Series) -> pd.DataFrame:
    wv = _daily(K / "Synthetic_water_value_2015_2024.csv")
    price = _daily(K / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    src, dst, up = plant_graph()
    d = pd.to_datetime(pd.Series(at_dates)).reset_index(drop=True)
    d1 = d + pd.Timedelta(days=7)
    d2 = d + pd.Timedelta(days=14)
    dm1 = d - pd.Timedelta(days=7)
    out = {}

    sysw = wv.mean(axis=1)
    out["wvt_sys_change"] = sysw.reindex(d1).to_numpy() - sysw.reindex(d).to_numpy()
    out["wvt_sys_change_prev"] = sysw.reindex(d).to_numpy() - sysw.reindex(dm1).to_numpy()
    out["wvt_sys_change_next"] = sysw.reindex(d2).to_numpy() - sysw.reindex(d1).to_numpy()
    out["wvt_sys_accel"] = out["wvt_sys_change"] - out["wvt_sys_change_prev"]
    out["wvt_sys_dispersion_change"] = (wv.std(axis=1).reindex(d1).to_numpy()
                                        - wv.std(axis=1).reindex(d).to_numpy())

    for plant, r in src.items():
        if r not in wv.columns:
            continue
        pl = san(plant)
        w0, w1 = wv[r].reindex(d).to_numpy(), wv[r].reindex(d1).to_numpy()
        out[f"wvt_change_{pl}"] = w1 - w0
        out[f"wvt_change_prev_{pl}"] = w0 - wv[r].reindex(dm1).to_numpy()
        out[f"wvt_change_next_{pl}"] = wv[r].reindex(d2).to_numpy() - w1
        # relative to the system move: did THIS reservoir deplete faster than the system?
        out[f"wvt_change_vs_sys_{pl}"] = (w1 - w0) - out["wvt_sys_change"]
        # scale-free
        out[f"wvt_change_pct_{pl}"] = (w1 - w0) / np.where(np.abs(w0) < 1e-6, np.nan, np.abs(w0))
        # against the price it was competing with
        p_wk = price.rolling(7, min_periods=1).mean().shift(-6).reindex(d).to_numpy()
        out[f"wvt_change_over_price_{pl}"] = (w1 - w0) / np.where(p_wk == 0, np.nan, p_wk)
        # downstream reservoir's water value also moves when this plant discharges into it
        dr = dst.get(plant)
        if dr in wv.columns:
            out[f"wvt_downstream_change_{pl}"] = (wv[dr].reindex(d1).to_numpy()
                                                   - wv[dr].reindex(d).to_numpy())
    return pd.DataFrame(out)


# ---------------------------------------------------------------- v2: fix the four worst units
# Diagnosed per-unit failures of build():
#   Songa   - src=None (tunnel-fed from Songavatn + Bitdalsvatn), balance built on nothing
#   Haukeli - source Vatjern has CONSTANT storage, so no trace exists on the source side
#   Lio     - Byrte's noisy release estimate swamps Lio's own drawdown (corr release/drawdown -0.25)
#   Hogga   - r_Vest_Vassdraget flows into Bandak and was never counted (the one river with inflow)

def effective_sources() -> dict[str, list[str]]:
    """Source reservoirs per plant, resolving tunnel-fed plants to their originating reservoirs."""
    t = topo(); c = t["connections"]
    direct = {x["to"]: [x["from"]] for x in c
              if x["to_type"] == "plant" and x["from_type"] == "reservoir"}
    for p in t["model"]["plant"]:
        if p in direct:
            continue
        tunnels = [x["from"] for x in c if x["to"] == p and x["from_type"] == "tunnel"]
        res = [x["from"] for x in c
               if x["to"] in tunnels and x["from_type"] == "reservoir"]
        if res:
            direct[p] = res
    return direct


def river_inflows_to() -> dict[str, list[str]]:
    """Rivers feeding each reservoir (only r_Vest_Vassdraget carries inflow data)."""
    t = topo()
    out = {}
    for x in t["connections"]:
        if x["to_type"] == "reservoir" and x["from_type"] == "river":
            out.setdefault(x["to"], []).append(x["from"])
    return out


def build_v2(at_dates: pd.Series) -> pd.DataFrame:
    """Corrected balance: multi-source plants, river inflows, and downstream-side estimates."""
    vol = _daily(K / "Historical_volume_2015_2024.csv")
    inf_d = _daily(K / "Historical_inflow_1958_2025.csv") * 3600 * 24 / 1e6
    src_multi = effective_sources()
    _, dst, _ = plant_graph()
    rivers = river_inflows_to()
    gens = topo()["model"]["generator"]

    d = pd.to_datetime(pd.Series(at_dates)).reset_index(drop=True)
    d1 = d + pd.Timedelta(days=7)
    out = {}

    def at(s, w):
        return s.reindex(w).to_numpy()

    def wsum(s, a, b):
        c_ = s.cumsum()
        return c_.reindex(b).to_numpy() - c_.reindex(a).to_numpy()

    def res_inflow(r):
        """Natural inflow plus any river inflow feeding this reservoir."""
        tot = inf_d[r] if r in inf_d.columns else pd.Series(0.0, index=vol.index)
        for rv in rivers.get(r, []):
            if rv in inf_d.columns:          # r_Vest_Vassdraget is the one with data
                tot = tot.add(inf_d[rv], fill_value=0.0)
        return tot

    for plant, sources in src_multi.items():
        pl = san(plant)
        usable = [r for r in sources if r in vol.columns and vol[r].nunique() > 1]
        cap = sum(gens[g]["p_max"] for g in gens if san(g).startswith(pl + "_G")) or np.nan

        # ---- source side, summed over ALL source reservoirs (fixes Songa)
        if usable:
            s0 = sum(at(vol[r], d) for r in usable)
            s1 = sum(at(vol[r], d1) for r in usable)
            inflow = sum(wsum(res_inflow(r), d, d1) for r in usable)   # fixes Hogga
            up = np.zeros(len(d))
            for p2, dr in dst.items():
                if dr in usable and p2 != plant:
                    for r2 in src_multi.get(p2, []):
                        if r2 in vol.columns and vol[r2].nunique() > 1:
                            up = up + (at(vol[r2], d) - at(vol[r2], d1)) + wsum(res_inflow(r2), d, d1)
            out[f"v2_drawdown_{pl}"] = s0 - s1
            out[f"v2_inflow_{pl}"] = inflow
            out[f"v2_release_raw_{pl}"] = (s0 - s1) + inflow          # WITHOUT upstream correction
            out[f"v2_release_up_{pl}"] = (s0 - s1) + inflow + up      # WITH it (fixes Lio via choice)
            out[f"v2_upstream_{pl}"] = up
            out[f"v2_n_sources_{pl}"] = len(usable)
        # ---- downstream side: the reservoir this plant discharges INTO rises when it runs
        dr = dst.get(plant)
        if dr in vol.columns and vol[dr].nunique() > 1:
            g0, g1 = at(vol[dr], d), at(vol[dr], d1)
            dr_inflow = wsum(res_inflow(dr), d, d1)
            # other plants also discharging into dr
            others = [p2 for p2, x in dst.items() if x == dr and p2 != plant]
            out[f"v2_down_rise_{pl}"] = (g1 - g0) - dr_inflow          # fixes Haukeli
            out[f"v2_down_n_other_{pl}"] = len(others)
            if not usable:                                             # no source trace at all
                out[f"v2_release_proxy_{pl}"] = (g1 - g0) - dr_inflow
        if f"v2_release_up_{pl}" in out and not np.isnan(cap):
            out[f"v2_hours_{pl}"] = out[f"v2_release_up_{pl}"] / cap
            out[f"v2_hours_raw_{pl}"] = out[f"v2_release_raw_{pl}"] / cap
    return pd.DataFrame(out)
