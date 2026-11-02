"""Economic bounds on generation, from the topology attributes we never read.

Fifteen of the twenty-two attributes in `Tokke_Vinje_topology.yaml` were unused. Most are
uninteresting (`p_nom` equals `p_max` for all fourteen units; `stopcost_const` equals
`startcost_const` exactly, so a cycle simply costs twice a start). Three are not, and they
combine into quantities that bound the thing the count model is trying to predict.

RESERVOIR CAPACITY AS AN ARBITRAGE WINDOW. `max_vol` with `lrl`/`hrl` gives how much energy a
reservoir can hold, and dividing by plant capacity gives how many HOURS the plant could run on
water it already owns. That number spans 21 to 3,659 hours across the nine plants -- a 174-fold
range describing two completely different economic problems. Songa can defer generation for
twenty-two weeks and wait for a seasonal price peak. Tokke, on a full Vinjevatn, has 21 hours
against a 168-hour horizon: it cannot arbitrage at all, and must generate as water arrives. We
have been giving both the same features.

SPILL AS A FORCED-GENERATION SIGNAL. If storage plus the coming week's inflow exceeds `max_vol`,
the surplus is spilled, and spilled water is revenue thrown away. The optimiser generates to
avoid it whether or not the price justifies it. Measured on 2015-2022, Tokke faces spill pressure
in 73% of weeks and Haukeli in 29%.

Together these BRACKET the count: `hours_from_storage` is an upper bound from owned water,
`spill_hours` a lower bound from avoided waste.

HEAD LOSS. `main_loss` and `penstock_loss` are the quadratic head-loss coefficients SHOP uses,
and `mwh_per_m3` ignores both, so every energy conversion in the project is biased high. The loss
is flow-dependent, which matters economically at multi-unit plants: running another unit raises
flow, raises the loss, and lowers everyone's efficiency. That is a real marginal cost of the
fourth Tokke unit that no current feature expresses.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import production_features as PF

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parents[1]
san = PF.san
WEEK_H = 168


def _topo():
    return yaml.safe_load(open(_ROOT / "data/extended/Tokke_Vinje_topology.yaml"))


# One economic idea per theme, so each can be added and judged on its own rather than as a bloc.
THEME_STEMS = {
    "capacity":  ("hours_storage", "weeks_storage", "storage_caps_week", "hours_storage_capped"),
    "spill":     ("spill_hours", "spill_risk"),
    "inflow":    ("hours_inflow", "inflow_over_storage"),
    "available": ("hours_available",),
    "band":      ("band_pos", "band_m", "headroom_hours"),
    "losses":    ("head_loss_frac", "marginal_loss"),
    "archetype": ("arbitrage_hours", "is_run_of_river"),
}


def themes(cols: list[str]) -> dict[str, list[str]]:
    """Group the emitted columns into single-idea blocks.

    Columns are `ec_<stem>_<Plant>`, and plant names contain underscores (`Vesle_Kjela`), so the
    split has to be made on a known stem list rather than by splitting on the last underscore.
    """
    out: dict[str, list[str]] = {}
    for theme, stems in THEME_STEMS.items():
        for c in cols:
            if any(c.startswith(f"ec_{s}_") for s in stems):
                out.setdefault(theme, []).append(c)
    return out


def plant_capacity() -> dict[str, float]:
    m = _topo()["model"]
    out = {}
    for p in m["plant"]:
        gens = [g for g in m["generator"] if san(g).startswith(san(p) + "_G")]
        out[p] = sum(m["generator"][g]["p_max"] for g in gens)
    return out


def head_loss_fraction(plant: str, flow_m3s: np.ndarray, storage: np.ndarray) -> np.ndarray:
    """Quadratic head loss as a fraction of gross head, at the given flow.

    SHOP models conduit loss as `loss = k * Q^2` in metres. `main_loss` varies by three orders of
    magnitude across plants (Haukeli 2.93, Tokke 0.0012), but so does flow, and the resulting
    loss lands at a few percent of head everywhere -- which is why ignoring it biased every
    conversion in the same direction rather than scrambling them.
    """
    t = _topo(); m = t["model"]
    spec = m["plant"][plant]
    k = float(np.sum([float(x) for x in spec.get("main_loss", [0])]))
    k += float(np.sum([float(x) for x in spec.get("penstock_loss", [0])]))
    src, _, _ = PF.plant_graph()
    res = src.get(plant)
    if res is None:
        return np.zeros_like(flow_m3s, dtype=float)
    vh = m["reservoir"][res]["vol_head"]
    level = np.interp(storage, vh["x"], vh["y"])
    gross = np.clip(level - spec["outlet_line"], 1e-6, None)
    with np.errstate(all="ignore"):
        return np.clip(k * flow_m3s ** 2 / gross, 0, 0.5)


def build(dates: pd.Series, storage: pd.DataFrame) -> pd.DataFrame:
    """Per-case economic bounds, emitted as `_<Plant>` blocks for per-unit resolution.

    `storage` carries `storage_<Reservoir>` columns at case start -- causal, known when the
    decision is made. Weekly inflow is exogenous and available across the test period.
    """
    t = _topo(); m = t["model"]
    src, dst, _ = PF.plant_graph()
    cap = plant_capacity()
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    inf = pd.read_csv(_ROOT / "data/kernel/Historical_inflow_1958_2025.csv")
    inf["date"] = pd.to_datetime(inf["date"], format="mixed", utc=True).dt.tz_localize(None)
    inf = inf.set_index("date").sort_index().resample("D").mean()
    inf.columns = [san(c) for c in inf.columns]
    # Mm3 arriving over the coming week
    wk = (inf.rolling(7, min_periods=1).mean().shift(-6) * 3600 * 24 * 7 / 1e6)

    out = pd.DataFrame(index=range(len(d)))
    for plant in m["plant"]:
        res = src.get(plant)
        p = san(plant)
        pmax = cap.get(plant, 0.0)
        if res is None or pmax <= 0:
            continue
        rn = san(res)
        scol = f"storage_{rn}"
        if scol not in storage.columns:
            continue
        R = m["reservoir"][res]
        s = storage[scol].to_numpy(dtype=float)
        e = np.array([PF.mwh_per_m3(plant, x) for x in s])          # MWh per m3 at this state
        q = wk[rn].reindex(d).to_numpy(dtype=float) if rn in wk.columns else np.zeros(len(d))

        # --- upper bound: hours the unit could run on water it already owns
        stored_mwh = np.clip(s, 0, None) * 1e6 * e
        hrs_storage = stored_mwh / pmax
        out[f"ec_hours_storage_{p}"] = hrs_storage
        out[f"ec_weeks_storage_{p}"] = hrs_storage / WEEK_H
        out[f"ec_storage_caps_week_{p}"] = (hrs_storage < WEEK_H).astype(float)
        out[f"ec_hours_storage_capped_{p}"] = np.clip(hrs_storage, 0, WEEK_H)

        # --- inflow: hours the coming week's arrivals alone would support
        inflow_mwh = np.clip(q, 0, None) * 1e6 * e
        out[f"ec_hours_inflow_{p}"] = inflow_mwh / pmax
        out[f"ec_inflow_over_storage_{p}"] = np.where(stored_mwh > 0, inflow_mwh / stored_mwh, 0.0)

        # --- lower bound: hours forced to run to avoid spilling revenue away
        overflow = np.clip(s + q - R["max_vol"], 0, None) * 1e6 * e
        out[f"ec_spill_hours_{p}"] = overflow / pmax
        out[f"ec_spill_risk_{p}"] = (overflow > 0).astype(float)

        # --- the two bounds together: total water available, in hours of running
        out[f"ec_hours_available_{p}"] = np.clip((stored_mwh + inflow_mwh) / pmax, 0, 3 * WEEK_H)

        # --- where the reservoir sits in its REGULATED band (comparable across reservoirs,
        #     unlike an absolute volume, and it is the band that constrains operation)
        vh = R["vol_head"]
        level = np.interp(s, vh["x"], vh["y"])
        band = max(R["hrl"] - R["lrl"], 1e-6)
        out[f"ec_band_pos_{p}"] = (level - R["lrl"]) / band
        out[f"ec_band_m_{p}"] = band
        out[f"ec_headroom_hours_{p}"] = np.clip(R["max_vol"] - s, 0, None) * 1e6 * e / pmax

        # --- head loss at full load: the efficiency penalty of running flat out
        q_full = pmax / np.clip(e, 1e-12, None) / 3600.0               # m3/s at p_max
        out[f"ec_head_loss_frac_{p}"] = head_loss_fraction(plant, q_full, s)
        n_units = len([g for g in m["generator"] if san(g).startswith(p + "_G")])
        if n_units > 1:
            # marginal cost of the next unit: loss rises with the square of total flow
            q_one = q_full / n_units
            l1 = head_loss_fraction(plant, q_one * (n_units - 1), s)
            out[f"ec_marginal_loss_{p}"] = out[f"ec_head_loss_frac_{p}"] - l1
        else:
            out[f"ec_marginal_loss_{p}"] = 0.0

        # --- static archetype: arbitrage window, the economic character of the plant
        full_mwh = R["max_vol"] * 1e6 * PF.mwh_per_m3(plant, R["max_vol"] * 0.5)
        out[f"ec_arbitrage_hours_{p}"] = full_mwh / pmax
        out[f"ec_is_run_of_river_{p}"] = float(full_mwh / pmax < WEEK_H)
    return out.replace([np.inf, -np.inf], np.nan)
