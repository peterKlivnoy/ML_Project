"""
Unit-relative feature assembly for the count model.

The flaw this fixes: the count model trains at (case, unit) but was given every plant's columns
under absolute names, so a Tokke_G1 row could see `binding_min_Haukeli`. The model latched onto
whichever plant happened to correlate across training, which is spurious and will not generalise.

Here every row sees only features relevant to ITS OWN unit, under generic names:

    own_*         that unit's plant
    upstream_*    plants discharging into this plant's source reservoir
    downstream_*  the reservoir this plant discharges into
    ownmv_/ownmf_ the environmental constraints wired to this plant in the topology
    sys_*         genuine system-wide context, shared by all units

The upstream terms matter because an upstream plant's production IS downstream water
availability: Tokke's inflow is Vinje plus Haukeli's output; Hogga's is Tokke's.
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


def relevance_map() -> dict[str, dict]:
    """Per plant: source, discharge target, upstream plants, and wired constraints."""
    t = PF.topo(); c = t["connections"]
    src, dst, up = PF.plant_graph()
    mv = [x for x in pd.read_csv(_ROOT / "data/extended/Constraint_min_volume.csv", nrows=1).columns
          if x != "date"]
    mf = [x for x in pd.read_csv(_ROOT / "data/extended/Constraint_min_flow.csv", nrows=1).columns
          if x != "date"]

    def down(node, depth=3):
        seen, fr = set(), {node}
        for _ in range(depth):
            nx = set()
            for x in c:
                if x["from"] in fr and x["to"] not in seen:
                    seen.add(x["to"]); nx.add(x["to"])
            fr = nx
        return seen

    out = {}
    for p in t["model"]["plant"]:
        near = up[p] | down(p)
        # plants whose discharge lands in this plant's source reservoir
        upstream_plants = [q for q, d in dst.items() if d == src.get(p) and q != p]
        out[p] = {
            "source": src.get(p),
            "discharges_to": dst.get(p),
            "upstream_plants": upstream_plants,
            "min_vol": [r for r in mv if r in near],
            "min_flow": [r for r in mf if r in near],
        }
    return out


def add_constraint_series(dates: pd.Series) -> pd.DataFrame:
    """Weekly mean of each constraint series, one row per case."""
    out = pd.DataFrame(index=range(len(dates)))
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    for tag, path in [("mv", "data/extended/Constraint_min_volume.csv"),
                      ("mf", "data/extended/Constraint_min_flow.csv")]:
        c = pd.read_csv(_ROOT / path)
        c["date"] = pd.to_datetime(c["date"]).dt.tz_localize(None)
        c = c.set_index("date").sort_index().resample("D").mean()
        wk = c.rolling(7, min_periods=1).mean().shift(-6)
        for col in c.columns:
            out[f"{tag}_{san(col)}"] = wk[col].reindex(d).to_numpy()
    return out


def build_unit_relative(CU: pd.DataFrame, PROD: pd.DataFrame, CONS: pd.DataFrame,
                        units: list[str]) -> pd.DataFrame:
    """Reassemble per-plant columns into unit-relative ones. CU must carry run_no and unit."""
    rel = relevance_map()
    plant_of = {u: u.rsplit("_G", 1)[0] for u in units}
    parts = []
    for u in units:
        pl_raw = next((p for p in rel if san(p) == plant_of[u]), plant_of[u])
        info = rel.get(pl_raw, {})
        pl = san(pl_raw)
        sub = CU[CU.unit == u].copy()
        prod = PROD.reindex(sub["run_no"]).reset_index(drop=True)
        cons = CONS.reindex(sub["run_no"]).reset_index(drop=True)
        sub = sub.reset_index(drop=True)

        # own plant, generic names
        own = prod[[c for c in prod.columns if c.endswith("_" + pl)]].copy()
        own.columns = ["own_" + c[: -(len(pl) + 1)] for c in own.columns]

        # upstream plants -> their production is this plant's inflow
        ups = [san(q) for q in info.get("upstream_plants", [])]
        up_cols = {}
        for base in ("release", "energy_mwh", "implied_hours", "capacity_factor", "drawdown"):
            cs = [f"{base}_{q}" for q in ups if f"{base}_{q}" in prod.columns]
            # no upstream plant means upstream discharge is genuinely ZERO, not missing
            up_cols[f"upstream_{base}"] = (prod[cs].sum(axis=1) if cs
                                           else pd.Series(0.0, index=prod.index))
        upf = pd.DataFrame(up_cols)

        # system context, shared legitimately
        sysf = prod[[c for c in prod.columns if c.startswith("sys_")]].copy()

        # constraints actually wired to this plant
        cc = {}
        for r in info.get("min_vol", []):
            if f"mv_{san(r)}" in cons.columns:
                cc[f"ownmv_{san(r)}"] = cons[f"mv_{san(r)}"]
        for r in info.get("min_flow", []):
            if f"mf_{san(r)}" in cons.columns:
                cc[f"ownmf_{san(r)}"] = cons[f"mf_{san(r)}"]
        consf = pd.DataFrame(cc) if cc else pd.DataFrame(index=sub.index)
        consf = consf.reindex(sub.index)
        # summarise so the column set is identical across units
        consf = pd.DataFrame({
            "own_minvol_level": consf[[c for c in consf.columns if c.startswith("ownmv_")]].mean(axis=1)
            if any(c.startswith("ownmv_") for c in consf.columns) else np.nan,
            "own_minflow_level": consf[[c for c in consf.columns if c.startswith("ownmf_")]].mean(axis=1)
            if any(c.startswith("ownmf_") for c in consf.columns) else np.nan,
            "own_n_minvol": len(info.get("min_vol", [])),
            "own_n_minflow": len(info.get("min_flow", [])),
            "own_n_upstream_plants": len(ups),
        }, index=sub.index)

        parts.append(pd.concat([sub, own, upf, sysf, consf], axis=1))
    return pd.concat(parts, ignore_index=True)
