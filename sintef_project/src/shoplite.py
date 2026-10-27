"""
shoplite: a first-principles replica of the weekly SHOP scheduling problem.

The labels are not observations of a noisy process. They are the output of one deterministic
optimiser (SHOP v17.17) run on inputs we have in full: the topology YAML is SHOP-ready, and price,
inflow, initial storage, terminal water value and the min-flow/min-volume constraints are all in
the kernel/extended data. So instead of learning the mapping inputs -> commitment from scratch,
this module rebuilds the optimisation SHOP solved and solves it ourselves with HiGHS:

    maximise   sum_t price_t * (production_t - friction losses_t)
             - start/stop costs - river flow costs - spill/bypass costs - constraint penalties
             + sum_r terminal_water_value_r * V_r,T

    subject to hourly water balance over 17 reservoirs, the Vest-Vassdraget river junction and
    the Songa tunnel junction; reservoir bounds; min-volume and min-flow (soft); per-unit on/off
    with p_min..p_max and a concave PQ curve; start/stop logic.

Simplifications versus SHOP (each is a place where our solution can disagree with the labels):
  * heads are frozen at the case's INITIAL reservoir levels (SHOP iterates on head);
  * turbine efficiency is linearised once per case (concave envelope of the PQ curve);
  * friction losses (main tunnel, shared penstocks) are a convex PWL cubic on plant flow;
  * tunnels between reservoirs are lossless free-flow links;
  * the initial on/off state of each unit is unknown, so hour 0 carries no start cost.

Terminal value. The water value file is in EUR/MWh. How SHOP turned it into EUR/Mm3 is not
documented, so it was measured from the labels instead (see `terminal_values`): each plant's
break-even price is the water value of the reservoir feeding it, independent of downstream values.

Outputs, per case: commitment u and power p per (hour, unit), and the hourly shadow price of
water in every reservoir (LP only). See `solve_case`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import highspy
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
T = 168
K = 0.0036          # Mm3 per (m3/s * hour)
G = 9.81e-3         # MW per (m3/s * m): P = G * H * Q * eta
BIG_PEN = 5_000.0
SIBLING_ORDER = (("Tokke_G1", "Tokke_G3", "Tokke_G4", "Tokke_G2"),
                 ("Vinje_G1", "Vinje_G2", "Vinje_G3"))   # EUR per (m3/s*h) of min-flow deficit; per Mm3 handled via /K


def sanitize(name: str) -> str:
    return name.replace(" ", "_")


# --------------------------------------------------------------------------------------------
# topology
# --------------------------------------------------------------------------------------------
@dataclass
class Unit:
    name: str
    plant: str
    penstock: int
    pmin: float
    pmax: float
    gen_x: np.ndarray
    gen_y: np.ndarray
    curves: list            # [(head, q[], eta[])] sorted by head
    start: float
    stop: float


@dataclass
class Plant:
    name: str
    outlet: float
    main_loss: float
    pen_loss: list
    units: list = field(default_factory=list)
    up: list = field(default_factory=list)      # [(type, name)] reservoir or tunnel
    down: tuple | None = None                   # (type, name) or None = sea


@dataclass
class Topology:
    res: dict
    rivers: dict
    plants: dict
    units: dict
    tunnels: dict
    river_up: dict          # river -> (type, name)
    river_down: dict        # river -> (type, name) or None
    tunnel_up: dict
    tunnel_down: dict


def _f(x):
    return float(x[0]) if isinstance(x, list) else float(x)


@lru_cache(maxsize=1)
def load_topology() -> Topology:
    d = yaml.safe_load(open(ROOT / "data/extended/Tokke_Vinje_topology.yaml"))
    m = d["model"]
    res = {k: dict(max_vol=float(v["max_vol"]), lrl=float(v["lrl"]), hrl=float(v["hrl"]),
                   vx=np.array(v["vol_head"]["x"], float), vy=np.array(v["vol_head"]["y"], float))
           for k, v in m["reservoir"].items()}
    plants = {k: Plant(k, float(v["outlet_line"]), _f(v["main_loss"]),
                       [float(x) for x in v["penstock_loss"]]) for k, v in m["plant"].items()}
    units = {}
    for k, v in m["generator"].items():
        curves = sorted(((float(c["ref"]), np.array(c["x"], float), np.array(c["y"], float) / 100)
                         for c in v["turb_eff_curves"]), key=lambda c: c[0])
        units[k] = Unit(k, "", int(v["penstock"]), float(v["p_min"]), float(v["p_max"]),
                        np.array(v["gen_eff_curve"]["x"], float),
                        np.array(v["gen_eff_curve"]["y"], float) / 100, curves,
                        float(v["startcost_const"]), float(v["stopcost_const"]))
    tunnels = {k: dict(loss=float(v["loss_factor"])) for k, v in m["tunnel"].items()}
    rivers = m["river"]
    river_up, river_down, tunnel_up, tunnel_down = {}, {}, {}, {}
    for c in d["connections"]:
        ft, f, tt, to = c["from_type"], c["from"], c["to_type"], c["to"]
        if ft == "generator":
            units[f].plant = to
            plants[to].units.append(f)
        elif tt == "plant":
            plants[to].up.append((ft, f))
            if ft == "tunnel":
                tunnel_down[f] = (tt, to)
        elif ft == "plant":
            plants[f].down = (tt, to)
        elif tt == "river":
            if ft == "reservoir":
                river_up[to] = (ft, f)
            else:                       # river -> river (into r_Vest_Vassdraget)
                river_down[f] = (tt, to)
        elif ft == "river":
            river_down[f] = (tt, to)
        elif tt == "tunnel":
            tunnel_up[to] = (ft, f)
        elif ft == "tunnel":
            tunnel_down[f] = (tt, to)
    for r in rivers:
        river_down.setdefault(r, None)
    return Topology(res, rivers, plants, units, tunnels, river_up, river_down,
                    tunnel_up, tunnel_down)


def level(topo: Topology, r: str, vol: float) -> float:
    R = topo.res[r]
    return float(np.interp(vol, R["vx"], R["vy"]))


# --------------------------------------------------------------------------------------------
# efficiency, PQ curves, conversion factors
# --------------------------------------------------------------------------------------------
def turbine_eta(u: Unit, q: np.ndarray, head: float) -> np.ndarray:
    """Turbine efficiency at flow q, linearly interpolated between the reference-head curves."""
    hs = [c[0] for c in u.curves]
    if head <= hs[0]:
        c = u.curves[0]
        return np.interp(q, c[1], c[2])
    if head >= hs[-1]:
        c = u.curves[-1]
        return np.interp(q, c[1], c[2])
    j = int(np.searchsorted(hs, head)) - 1
    (h0, x0, y0), (h1, x1, y1) = u.curves[j], u.curves[j + 1]
    w = (head - h0) / (h1 - h0)
    return (1 - w) * np.interp(q, x0, y0) + w * np.interp(q, x1, y1)


def q_range(u: Unit) -> tuple[float, float]:
    lo = max(c[1][0] for c in u.curves)
    hi = min(c[1][-1] for c in u.curves)
    return lo, hi


def unit_pq(u: Unit, h_gross: float, plant: Plant, n: int = 40):
    """Single-unit PQ points at gross head h_gross (losses are added separately at plant level).

    Efficiency is evaluated at the head net of this unit's own friction losses, which is the
    head the turbine actually sees when it runs alone.
    """
    qlo, qhi = q_range(u)
    q = np.linspace(qlo, qhi, n)
    kpen = plant.pen_loss[u.penstock - 1] if len(plant.pen_loss) >= u.penstock else 0.0
    h_eval = h_gross - (plant.main_loss + kpen) * q ** 2
    eta_t = turbine_eta(u, q, h_eval.mean())
    p = G * h_gross * q * eta_t
    p = p * np.interp(p, u.gen_x, u.gen_y)
    # operating range is capped at p_max; keep at least two points
    keep = p <= u.pmax + 1e-9
    keep[:2] = True
    return q[keep], p[keep]


def concave_envelope(q: np.ndarray, p: np.ndarray):
    """Upper concave hull of (q, p) starting at the first point. Returns breakpoints."""
    pts = []
    for x, y in zip(q, p):
        while len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            # remove middle point if it lies below the chord
            if (y2 - y1) * (x - x1) <= (y - y1) * (x2 - x1):
                pts.pop()
            else:
                break
        pts.append((x, y))
    a = np.array(pts)
    return a[:, 0], a[:, 1]


def head_for_plant(topo: Topology, plant: Plant, V0: dict) -> float:
    """Gross head from the initial upstream level to max(outlet line, downstream level)."""
    ups = []
    for t, n in plant.up:
        if t == "reservoir":
            ups.append(level(topo, n, V0[n]))
        elif t == "tunnel":
            r = topo.tunnel_up[n][1]
            ups.append(level(topo, r, V0[r]))
    up = float(np.mean(ups))
    tail = plant.outlet
    if plant.down and plant.down[0] == "reservoir":
        tail = max(tail, level(topo, plant.down[1], V0[plant.down[1]]))
    return max(up - tail, 1.0)


def plant_conversion(topo: Topology, plant: Plant, head: float) -> float:
    """MWh per Mm3 at the best operating point of the plant's first unit."""
    u = topo.units[plant.units[0]]
    q, p = unit_pq(u, head, plant)
    return float(np.max(p / q)) / K


def main_path(topo: Topology, r: str) -> list[tuple[str, str]]:
    """(plant, feeding reservoir) for every plant on the main waterway from r to the sea.

    The feeding reservoir is the last reservoir the water passed through before the plant.
    """
    out, seen, node, last = [], set(), ("reservoir", r), r
    while node and node not in seen:
        seen.add(node)
        t, n = node
        if t == "reservoir":
            last = n
        if t == "plant":
            out.append((n, last))
            node = topo.plants[n].down
            continue
        if t == "river":
            node = topo.river_down.get(n)
            continue
        # reservoir: plant > tunnel > main river
        nxt = None
        for p in topo.plants.values():
            if ("reservoir", n) in p.up:
                nxt = ("plant", p.name)
        if nxt is None:
            for tn, (ut, un) in topo.tunnel_up.items():
                if un == n:
                    dt, dn = topo.tunnel_down[tn]
                    nxt = (dt, dn)
        if nxt is None:
            for rv, (ut, un) in topo.river_up.items():
                if un == n and topo.rivers[rv].get("main_river"):
                    nxt = topo.river_down[rv]
        node = nxt
    return out


def terminal_values(topo: Topology, wv: dict, conv: dict, opt) -> tuple[dict, dict]:
    """EUR/Mm3 of water left in each reservoir at the end of the week.

    "local" (default, measured): one Mm3 in r is worth, at every plant k on its way to the sea,
    conv_k MWh priced at the water value of the reservoir feeding k. Then the break-even price
    of plant k is exactly WV(feeding reservoir) whatever the downstream values are, which is
    what the labels show: regressing each plant's implied price threshold on terminal water
    values gives ~1.0 x upstream WV and ~0 x downstream WV (Byrte R2 0.99, Vesle Kjela 0.97).

    "ecf": EUR/MWh x the reservoir's full-cascade conversion factor (SHOP's documented
    energy_value_input conversion). Puts a large negative weight on downstream WV; refuted.
    """
    ecf = {r: sum(conv[p] for p, _ in main_path(topo, r)) for r in topo.res}
    if opt.wv_mode == "ecf":
        wvm = {r: wv[r] * ecf[r] for r in topo.res}
    elif opt.wv_mode == "local":
        wvm = {r: sum(wv[f] * conv[p] for p, f in main_path(topo, r)) for r in topo.res}
    else:
        wvm = {r: wv[r] * opt.flat_ecf for r in topo.res}
    return wvm, ecf


# --------------------------------------------------------------------------------------------
# raw data
# --------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_raw():
    k = ROOT / "data/kernel"
    e = ROOT / "data/extended"

    def rd(p):
        df = pd.read_csv(p)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df.set_index("date")
    price = rd(k / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0]
    inflow = rd(k / "Historical_inflow_1958_2025.csv")
    inflow = inflow[inflow.index >= "2014-12-01"]
    vol = rd(k / "Historical_volume_2015_2024.csv")
    wv = rd(k / "Synthetic_water_value_2015_2024.csv")
    minvol = rd(e / "Constraint_min_volume.csv")
    minflow = rd(e / "Constraint_min_flow.csv")
    return price, inflow, vol, wv, minvol, minflow


def case_inputs(start: pd.Timestamp) -> dict:
    price, inflow, vol, wv, minvol, minflow = load_raw()
    idx = pd.date_range(start, periods=T, freq="h")
    end = start + pd.Timedelta(days=7)
    return dict(
        price=price.reindex(idx).ffill().bfill().to_numpy(float),
        inflow=inflow.reindex(idx).ffill().bfill(),
        V0=vol.loc[start].to_dict(),
        wv=wv.loc[end].to_dict() if end in wv.index else wv.iloc[-1].to_dict(),
        minvol=minvol.reindex(idx).ffill().fillna(0.0),
        minflow=minflow.reindex(idx).ffill().fillna(0.0),
    )


# --------------------------------------------------------------------------------------------
# model builder
# --------------------------------------------------------------------------------------------
class _Builder:
    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.rows_lo, self.rows_hi = [], []
        self.ri, self.ci, self.v = [], [], []
        self.n = 0
        self.m = 0

    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0, integer=False):
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel())
        self.c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel())
        self.integ.append(np.full(size, integer))
        return idx

    def rows(self, terms, lo, hi):
        """terms: list of (col_idx[T], coef scalar or [T]); one row per element."""
        nrow = len(terms[0][0])
        r = np.arange(self.m, self.m + nrow)
        self.m += nrow
        for idx, coef in terms:
            idx = np.asarray(idx)
            self.ri.append(r)
            self.ci.append(idx)
            self.v.append(np.broadcast_to(np.asarray(coef, float), idx.shape).astype(float))
        self.rows_lo.append(np.broadcast_to(np.asarray(lo, float), (nrow,)))
        self.rows_hi.append(np.broadcast_to(np.asarray(hi, float), (nrow,)))
        return r

    def solve(self, mip: bool, time_limit: float, gap: float, threads: int = 1):
        import scipy.sparse as sp
        lb, ub, c = (np.concatenate(x) for x in (self.lb, self.ub, self.c))
        A = sp.csc_matrix((np.concatenate(self.v), (np.concatenate(self.ri),
                                                    np.concatenate(self.ci))),
                          shape=(self.m, self.n))
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.n, self.m
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lb, ub
        lp.row_lower_ = np.concatenate(self.rows_lo)
        lp.row_upper_ = np.concatenate(self.rows_hi)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = A.indptr
        lp.a_matrix_.index_ = A.indices
        lp.a_matrix_.value_ = A.data
        lp.sense_ = highspy.ObjSense.kMaximize
        integ = np.concatenate(self.integ)
        if mip:
            lp.integrality_ = [highspy.HighsVarType.kInteger if b else
                               highspy.HighsVarType.kContinuous for b in integ]
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", threads)
        if mip:
            h.setOptionValue("time_limit", float(time_limit))
            h.setOptionValue("mip_rel_gap", float(gap))
        h.passModel(lp)
        h.run()
        sol = h.getSolution()
        info = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus())
        x = np.array(sol.col_value)
        y = np.array(sol.row_dual) if (not mip and sol.dual_valid) else None
        return x, y, status, info.objective_function_value


@dataclass
class Options:
    wv_mode: str = "local"          # see `terminal_values`
    river_cost: str = "total"       # r_ flow_cost_curve: "total" EUR/h, or "marginal"
    losses: bool = True             # plant main-tunnel / shared-penstock friction
    start_costs: bool = True
    sibling_order: bool = True
    flat_ecf: float = 1000.0


def solve_case(start, mip: bool = False, opt: Options | None = None,
               time_limit: float = 30.0, gap: float = 1e-3) -> dict:
    """Build and solve one weekly case. Returns arrays keyed by unit / reservoir."""
    opt = opt or Options()
    topo = load_topology()
    inp = case_inputs(pd.Timestamp(start))
    price = inp["price"]
    V0 = {r: float(inp["V0"][r]) for r in topo.res}
    B = _Builder()

    # --- heads, conversion factors, terminal values ---
    head = {p: head_for_plant(topo, P, V0) for p, P in topo.plants.items()}
    conv = {p: plant_conversion(topo, P, head[p]) for p, P in topo.plants.items()}
    wvm, ecf = terminal_values(topo, inp["wv"], conv, opt)

    # --- units ---
    U = {}
    for name, u in topo.units.items():
        P = topo.plants[u.plant]
        q, p = unit_pq(u, head[u.plant], P)
        qe, pe = concave_envelope(q, p)
        w = np.diff(qe)
        slope = np.diff(pe) / w
        # u carries the minimum block (qmin -> pmin); segments carry the rest of the PQ curve
        uu = B.var((T,), 0, 1, price * pe[0], integer=True)
        st = B.var((T,), 0, 1, -(u.start if opt.start_costs else 0.0))
        sd = B.var((T,), 0, 1, -(u.stop if opt.start_costs else 0.0))
        seg = B.var((len(w), T), 0, np.repeat(w[:, None], T, 1), slope[:, None] * price[None, :])
        U[name] = dict(u=uu, st=st, sd=sd, seg=seg, qmin=qe[0], pmin=pe[0], w=w, slope=slope)
        for k in range(len(w)):
            B.rows([(seg[k], 1.0), (uu, -w[k])], -np.inf, 0.0)
        B.rows([(st[1:], 1.0), (uu[1:], -1.0), (uu[:-1], 1.0)], 0.0, np.inf)
        B.rows([(sd[1:], 1.0), (uu[1:], 1.0), (uu[:-1], -1.0)], 0.0, np.inf)

    # SHOP's tie-break between identical siblings, as measured on the labels: one Tokke unit
    # on is G1 99% of the time, two are G1+G3 94%, three G1+G3+G4 66%; Vinje is G1, G1+G2.
    # Encoding the order turns "which unit" into "how many units" and breaks MIP symmetry.
    if opt.sibling_order:
        for chain in SIBLING_ORDER:
            for a, b in zip(chain[:-1], chain[1:]):
                B.rows([(U[a]["u"], 1.0), (U[b]["u"], -1.0)], 0.0, np.inf)

    def flow_terms(unit_names, sign=1.0):
        """Terms whose sum is the total discharge of the listed units."""
        out = []
        for nm in unit_names:
            d = U[nm]
            out.append((d["u"], sign * d["qmin"]))
            for k in range(len(d["w"])):
                out.append((d["seg"][k], sign))
        return out

    # --- friction losses: L >= tangent cuts of c*Q^3, priced at max(price, 0) ---
    if opt.losses:
        ppos = np.maximum(price, 0.0)
        for pn, P in topo.plants.items():
            groups = []
            if P.main_loss > 0:
                groups.append((P.main_loss, P.units))
            for j, kp in enumerate(P.pen_loss):
                on_pen = [nm for nm in P.units if topo.units[nm].penstock == j + 1]
                if kp > 0 and on_pen:
                    groups.append((kp, on_pen))
            eta = 0.9
            for kloss, members in groups:
                qmax = sum(q_range(topo.units[nm])[1] for nm in members)
                c3 = G * eta * kloss          # MW lost = c3 * Q^3
                L = B.var((T,), 0, np.inf, -ppos)
                for qj in np.linspace(qmax / 8, qmax, 8):
                    a, b = 3 * c3 * qj ** 2, -2 * c3 * qj ** 3
                    B.rows([(L, 1.0)] + flow_terms(members, -a), b, np.inf)

    # --- rivers ---
    F = {}
    mf = inp["minflow"]
    for rv, R in topo.rivers.items():
        if rv == "r_Vest_Vassdraget":
            F[rv] = B.var((T,), 0, np.inf, 0)
            continue
        if "flow_cost_curve" in R:
            x = np.array(R["flow_cost_curve"]["x"], float)
            y = np.array(R["flow_cost_curve"]["y"], float)
            # "total": y is the cost per hour at flow x, so the marginal cost is the slope,
            # made non-decreasing so the LP fills segments in order. "marginal": y is itself
            # the cost per (m3/s*h) -- refuted on Kjela, whose outflow crosses r_Hyljelihyl.
            if opt.river_cost == "marginal":
                mc = 0.5 * (y[:-1] + y[1:])
            else:
                mc = np.maximum.accumulate(np.diff(y) / np.diff(x))
            F[rv] = [B.var((T,), 0, x[j + 1] - x[j], -mc[j]) for j in range(len(x) - 1)]
        else:
            cost = float(R.get("flow_cost_const", 0.0))
            ub = float(R.get("max_flow_const", np.inf)) if rv.startswith("b_") else np.inf
            F[rv] = B.var((T,), 0, ub, -cost)
        if rv in mf.columns:
            req = mf[rv].to_numpy(float)
            if req.max() > 0:
                sl = B.var((T,), 0, np.inf, -BIG_PEN)
                terms = [(s, 1.0) for s in F[rv]] if isinstance(F[rv], list) else [(F[rv], 1.0)]
                B.rows(terms + [(sl, 1.0)], req, np.inf)

    def river_terms(rv, sign):
        f = F[rv]
        return [(s, sign) for s in f] if isinstance(f, list) else [(f, sign)]

    # --- tunnels ---
    TN = {}
    for tn in topo.tunnels:
        TN[tn] = B.var((T,), -1000.0, 1000.0, 0.0)

    # --- reservoirs ---
    V = {}
    mv = inp["minvol"]
    for r, R in topo.res.items():
        ub = max(R["max_vol"], 0.0)
        cost = np.zeros(T)
        cost[-1] = wvm[r]
        V[r] = B.var((T,), 0.0, ub, cost)
        if r in mv.columns:
            req = mv[r].to_numpy(float)
            if req.max() > 0:
                sl = B.var((T,), 0, np.inf, -1e5)
                B.rows([(V[r], 1.0), (sl, 1.0)], req, np.inf)

    # water balance: V[t] - V[t-1] - K*(in - out) = K*inflow  (V[-1] = V0)
    infl = inp["inflow"]
    bal_rows = {}
    for r in topo.res:
        ins, outs = [], []
        for pn, P in topo.plants.items():
            if P.down == ("reservoir", r):
                ins += flow_terms(P.units, 1.0)
            if ("reservoir", r) in P.up:
                outs += flow_terms(P.units, 1.0)
        for rv in topo.rivers:
            if topo.river_down.get(rv) == ("reservoir", r):
                ins += river_terms(rv, 1.0)
            if topo.river_up.get(rv) == ("reservoir", r):
                outs += river_terms(rv, 1.0)
        for tn in topo.tunnels:
            if topo.tunnel_down.get(tn) == ("reservoir", r):
                ins.append((TN[tn], 1.0))
            if topo.tunnel_up.get(tn) == ("reservoir", r):
                outs.append((TN[tn], 1.0))
        rhs = K * infl[r].to_numpy(float) if r in infl.columns else np.zeros(T)
        rhs = rhs.copy()
        rhs[0] += V0[r]
        # build the per-t rows: V[t] - V[t-1] - K*sum(ins) + K*sum(outs)
        rows_terms = [(V[r], 1.0)]
        rows_terms += [(i, -K * np.asarray(c)) for i, c in ins]
        rows_terms += [(i, K * np.asarray(c)) for i, c in outs]
        rid = B.rows(rows_terms, rhs, rhs)
        # add -V[t-1] for t >= 1 directly into the COO
        B.ri.append(rid[1:])
        B.ci.append(V[r][:-1])
        B.v.append(-np.ones(T - 1))
        bal_rows[r] = rid

    # river junction r_Vest_Vassdraget: F = own inflow + inputs; F flows into Bandak (above)
    ins = []
    for rv in topo.rivers:
        if topo.river_down.get(rv) == ("river", "r_Vest_Vassdraget"):
            ins += river_terms(rv, -1.0)
    for pn, P in topo.plants.items():
        if P.down == ("river", "r_Vest_Vassdraget"):
            ins += flow_terms(P.units, -1.0)
    rhs = infl["r_Vest_Vassdraget"].to_numpy(float)
    B.rows([(F["r_Vest_Vassdraget"], 1.0)] + ins, rhs, rhs)
    if "r_Vest_Vassdraget" in mf.columns:
        req = mf["r_Vest_Vassdraget"].to_numpy(float)
        sl = B.var((T,), 0, np.inf, -BIG_PEN)
        B.rows([(F["r_Vest_Vassdraget"], 1.0), (sl, 1.0)], req, np.inf)

    # plants fed by tunnels (Songa): plant flow = sum of tunnel flows
    for pn, P in topo.plants.items():
        tins = [n for t, n in P.up if t == "tunnel"]
        if tins:
            B.rows(flow_terms(P.units, 1.0) + [(TN[t], -1.0) for t in tins], 0.0, 0.0)

    x, y, status, obj = B.solve(mip, time_limit, gap)

    out = dict(status=status, obj=obj, ecf=ecf, head=head, conv=conv)
    if x.size == 0 or not np.isfinite(x).all():
        return out
    for nm, d in U.items():
        u = x[d["u"]]
        q = d["qmin"] * u + sum(x[d["seg"][k]] for k in range(len(d["w"])))
        p = d["pmin"] * u + sum(d["slope"][k] * x[d["seg"][k]] for k in range(len(d["w"])))
        out[f"u_{nm}"] = u
        out[f"q_{nm}"] = q
        out[f"p_{nm}"] = p
    for rv, f in F.items():
        out[f"f_{rv}"] = sum(x[s] for s in f) if isinstance(f, list) else x[f]
    for r in topo.res:
        out[f"v_{r}"] = x[V[r]]
        if y is not None:
            # dual of the balance row = EUR per Mm3 of extra water in r at hour t
            out[f"mu_{r}"] = np.abs(y[bal_rows[r]])
    if y is not None:
        # local water value of each unit, EUR/MWh: the water it uses is worth mu_up and lands
        # downstream worth mu_down; divide by MWh per Mm3 at the unit's best point
        for nm, d in U.items():
            up, dn = unit_up_down(topo, nm)
            mu_dn = out[f"mu_{dn}"] if dn else 0.0
            best = float(np.max((d["pmin"] + np.r_[0, np.cumsum(d["slope"] * d["w"])])
                                / (d["qmin"] + np.r_[0, np.cumsum(d["w"])])))
            out[f"lwv_{nm}"] = (out[f"mu_{up}"] - mu_dn) * K / best
    return out


def unit_up_down(topo: Topology, unit: str) -> tuple[str, str | None]:
    """Reservoir a unit draws from, and the reservoir its discharge reaches first (or None)."""
    P = topo.plants[topo.units[unit].plant]
    t, n = P.up[0]
    up = n if t == "reservoir" else topo.tunnel_up[n][1]
    node = P.down
    while node and node[0] == "river":
        node = topo.river_down.get(node[1])
    return up, (node[1] if node else None)
