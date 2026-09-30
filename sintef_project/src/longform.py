"""
Long-format pointwise model: one row per (case, hour, UNIT).

Every pointwise model in this project has been wide -- one row per (case, hour) with 14 outputs
under `multi_strategy="multi_output_tree"`. That shape forces three compromises at once, and they
are the three things we have been failing to fix separately:

1. SHARED SPLITS. A vector-leaf tree picks one split per node and applies it to all 14 outputs.
   Lio and Vesle_Kjela have almost nothing in common, but any split that helps one is paid for by
   all. The model cannot specialise, and we have never measured what that costs.

2. THE COUNT BLOCK IS 14x WIDER THAN IT NEEDS TO BE. Forwarding 5-6 numbers per unit means 70-84
   columns, laid out as `cnt_p_on_Lio_G1`, `cnt_p_on_Songa_G1`, ... For Lio's output to use Lio's
   own count prediction, the tree must first split on the column belonging to Lio -- and that
   split then applies to the other 13 heads, where it is noise. Log 029 measured the symptom
   (72.6% of count-model importance on unit-invariant columns) and log 030 showed the obvious fix
   was the wrong one. In long format the same information is 6 columns named `own_*`, used
   directly, with no routing.

3. NO PER-UNIT WEIGHTING. `sample_weight` applies to a ROW, and a wide row carries all 14 units,
   so a unit cannot be weighted at all. Lio is worth +0.0146 on its own (oracle test) and is the
   single largest drag on the pooled score; in long format it is finally addressable.

The cost is row count: 2,922 cases x 168 hours x 14 units = 6.9M rows. `case_stride` trades that
against fidelity, and log 017's near-duplicate result says the trade is cheap -- consecutive
cases overlap 144 wall-clock hours at 88% label agreement, so a strided subset of cases loses far
less than its row count suggests.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_OWN = ("p_off", "p_on", "p_low", "p_high", "exp_count", "margin",
        # full-distribution outputs (count_model.distribution_outputs); p_off/p_on above are
        # the exact all-off and all-on endpoint bins, so only the interior bins are listed here
        "p_bin1", "p_bin2", "p_bin3", "p_bin4",
        "count_sd", "entropy", "mode_bin", "p_lt_quarter", "p_ge_half")


def to_long(wide: pd.DataFrame, units: list[str], shared: list[str],
            count_prefix: str = "cnt_", coupled_prefix: tuple[str, ...] = ("hs_", "nb_"),
            static: pd.DataFrame | None = None,
            case_stride: int = 1, run_col: str = "run_no",
            dtype=np.float32) -> pd.DataFrame:
    """Melt a wide (case, hour) frame into (case, hour, unit) rows.

    `shared` columns are copied to every unit's rows unchanged -- they are genuine case/hour
    context (system spread, hour-of-day, window aggregates). Per-unit columns named
    `<prefix><field>_<unit>` collapse into generic `own_<field>` columns, which is the whole
    point: the model sees "my own predicted count", not "Lio's predicted count".
    """
    w = wide
    if case_stride > 1:
        keep = np.sort(w[run_col].unique())[::case_stride]
        w = w[w[run_col].isin(keep)]

    shared = [c for c in shared if c in w.columns]
    base = w[[run_col, "hour"] + shared].reset_index(drop=True)
    base[shared] = base[shared].astype(dtype)

    parts = []
    for u in units:
        own = {}
        for f in _OWN:
            c = f"{count_prefix}{f}_{u}"
            if c in w.columns:
                own[f"own_{f}"] = w[c].to_numpy(dtype=dtype)
        prefixes = (coupled_prefix,) if isinstance(coupled_prefix, str) else coupled_prefix
        for c in w.columns:
            for pre in prefixes:
                if c.startswith(pre) and c.endswith("_" + u):
                    own["own_" + c[len(pre):-(len(u) + 1)]] = w[c].to_numpy(dtype=dtype)
                    break
        p = pd.concat([base, pd.DataFrame(own)], axis=1)
        p["unit"] = u
        p["y"] = w[f"on_{u}"].to_numpy(dtype=np.int8)
        parts.append(p)

    out = pd.concat(parts, ignore_index=True)
    if static is not None:
        out = out.merge(static, on="unit", how="left")
    out["unit"] = out["unit"].astype("category")
    return out


def cross_unit_context(wide: pd.DataFrame, units: list[str],
                       count_prefix: str = "cnt_") -> pd.DataFrame:
    """Give long format back the cross-unit interaction that the wide shape had for free.

    This is the one real advantage of the wide layout: a wide row carries all 14 units at once,
    so the model can see that Tokke_G1 is expected to run a lot WHILE Tokke_G2 is not, and
    allocate accordingly. Units compete for the same water and the same high-price hours, so that
    competition is genuine signal, not an artefact.

    A long row knows only its own unit, so the interaction has to be handed back explicitly --
    as RELATIVE quantities (rank among units, share of the system total, gap to same-plant
    siblings) rather than as the other units' raw columns. Relative form is also what makes it
    transferable: `own_count_rank` means the same thing in a wet year and a dry one, where
    "Songa's expected count is 140" does not.

    Returns wide columns named `<field>_<unit>`, which `to_long` collapses to `own_<field>`.
    """
    ec = np.column_stack([wide[f"{count_prefix}exp_count_{u}"].to_numpy(dtype=float)
                          for u in units])
    pon = np.column_stack([wide[f"{count_prefix}p_on_{u}"].to_numpy(dtype=float)
                           for u in units])
    tot = np.nansum(ec, axis=1)
    med = np.nanmedian(ec, axis=1)
    rank = np.argsort(np.argsort(-ec, axis=1), axis=1) / max(len(units) - 1, 1)

    plant = {u: u.rsplit("_G", 1)[0] for u in units}
    out = {}
    for i, u in enumerate(units):
        out[f"xctx_count_rank_{u}"] = rank[:, i]
        with np.errstate(all="ignore"):
            out[f"xctx_count_share_{u}"] = np.where(tot > 0, ec[:, i] / tot, 0.0)
        out[f"xctx_count_vs_med_{u}"] = ec[:, i] - med
        sib = [j for j, v in enumerate(units) if plant[v] == plant[u] and v != u]
        if sib:
            out[f"xctx_sibling_count_{u}"] = np.nanmean(ec[:, sib], axis=1)
            out[f"xctx_vs_sibling_{u}"] = ec[:, i] - np.nanmean(ec[:, sib], axis=1)
        else:
            out[f"xctx_sibling_count_{u}"] = 0.0
            out[f"xctx_vs_sibling_{u}"] = 0.0
    # genuine system context, identical for every unit of a case
    shared = pd.DataFrame({"xsys_total_exp_count": tot,
                           "xsys_n_expected_on": np.nansum(pon, axis=1),
                           "xsys_count_dispersion": np.nanstd(ec, axis=1)},
                          index=wide.index)
    return pd.concat([pd.DataFrame(out, index=wide.index), shared], axis=1)


def neighbour_context(wide: pd.DataFrame, units: list[str], fields=("exp_count", "p_on", "p_off"),
                      count_prefix: str = "cnt_") -> pd.DataFrame:
    """Each unit sees its HYDRAULIC NEIGHBOURS' predicted counts, not just aggregate context.

    `cross_unit_context` gives a unit its position among all 14 -- rank, share, dispersion. That
    is the right shape for competition over price, but it is blind to topology: it cannot say
    that the plant immediately upstream is about to release a week's worth of water into my
    reservoir, which is the strongest single determinant of whether I run.

    Three neighbourhoods, from the tunnel-resolved graph:

        sibling     other units at my own plant, sharing one reservoir and one penstock
        upstream    units whose discharge becomes my inflow  (Songa -> Vinje, Vinje -> Tokke)
        downstream  units drawing from the reservoir I discharge into

    Sibling terms matter for a different reason than the other two: units at a plant are
    dispatched in merit order, so a sibling's predicted count is close to a direct statement
    about where I sit in that order.
    """
    import production_features as PF

    src, dst, _ = PF.plant_graph()
    plant_of = {u: u.rsplit("_G", 1)[0] for u in units}
    raw = {PF.san(p): p for p in src}

    def units_at(plant_sanitised):
        return [v for v in units if plant_of[v] == plant_sanitised]

    out = {}
    for u in units:
        pl = plant_of[u]
        pl_raw = raw.get(pl)
        sib = [v for v in units_at(pl) if v != u]
        ups = [PF.san(q) for q in (PF.effective_upstream_plants(pl_raw) if pl_raw else [])]
        tgt = dst.get(pl_raw) if pl_raw else None
        downs = [PF.san(q) for q, s in src.items() if s == tgt and PF.san(q) != pl] if tgt else []

        for tag, group in (("sib", sib),
                           ("up", [v for p in ups for v in units_at(p)]),
                           ("down", [v for p in downs for v in units_at(p)])):
            for f in fields:
                cols = [f"{count_prefix}{f}_{v}" for v in group
                        if f"{count_prefix}{f}_{v}" in wide.columns]
                vals = (wide[cols].mean(axis=1).to_numpy(dtype=float) if cols
                        else np.zeros(len(wide)))
                out[f"nb_{tag}_{f}_{u}"] = vals
            # a neighbour that does not exist is a fact about the topology, not missing data
            out[f"nb_{tag}_n_{u}"] = float(len(group))
        # merit-order position expressed against siblings specifically
        own = f"{count_prefix}exp_count_{u}"
        sib_cols = [f"{count_prefix}exp_count_{v}" for v in sib
                    if f"{count_prefix}exp_count_{v}" in wide.columns]
        if own in wide.columns and sib_cols:
            out[f"nb_sib_gap_{u}"] = (wide[own].to_numpy(dtype=float)
                                      - wide[sib_cols].mean(axis=1).to_numpy(dtype=float))
        else:
            out[f"nb_sib_gap_{u}"] = np.zeros(len(wide))
    return pd.DataFrame(out, index=wide.index)


def mask_to_units(long: pd.DataFrame, cols: list[str], keep: list[str],
                  unit_col: str = "unit") -> pd.DataFrame:
    """Give a feature block to some units only, by blanking it elsewhere.

    Different units need different inputs. The slope family is the worked example: it moves
    Haukeli +0.0154 and makes eleven of the other thirteen units worse, so as a global block it
    is a net loss that contains a real win.

    Long format lets us have both without splitting the model. A feature set to NaN carries no
    information for that row -- XGBoost sends it down the learned default direction -- so the
    block is live for the units that need it and inert for the rest. One model, one shared core,
    a small per-unit tail. Masking beats relying on the tree to learn `unit == X AND slope > y`,
    which needs two specific splits to land in the same branch and mostly does not.

    WHICH units get a block must be decided structurally, not from per-unit validation scores --
    picking the units a feature happened to help is fitting the validation set. The slope block
    goes to units whose source reservoir has degenerate storage (Haukeli's Vatjern is a
    documented placeholder constant; Hogga's Bandak has 136 distinct values in ten years), since
    those are precisely the units with no storage dynamics to read.
    """
    out = long
    present = [c for c in cols if c in out.columns]
    if not present:
        return out
    off = ~out[unit_col].astype(str).isin(keep).to_numpy()
    out.loc[off, present] = np.nan
    return out


def degenerate_storage_units(units: list[str], threshold: float = 0.10) -> list[str]:
    """Units whose source reservoir barely moves, so post-decision storage tells them nothing.

    `threshold` is on the coefficient of variation of daily storage. Vatjern and Hyljelihyl have
    no historical record at all and are documented as held at 50% full, so they score 0.
    """
    import production_features as PF

    root = Path(__file__).resolve().parents[1]
    vol = pd.read_csv(root / "data/kernel/Historical_volume_2015_2024.csv")
    vol = vol.drop(columns=["date"])
    vol.columns = [PF.san(c) for c in vol.columns]
    src, _, _ = PF.plant_graph()
    out = []
    for u in units:
        pl = u.rsplit("_G", 1)[0]
        raw = next((p for p in src if PF.san(p) == pl), None)
        res = PF.san(src[raw]) if raw and src.get(raw) else None
        if res is None or res not in vol.columns:
            continue
        s = vol[res]
        cv = float(s.std() / s.mean()) if s.mean() else 0.0
        if cv < threshold:
            out.append(u)
    return out


def unit_weights(units_col: pd.Series, boost: dict[str, float] | None = None) -> np.ndarray:
    """Per-row weights -- only possible in long format (see reason 3 above)."""
    w = np.ones(len(units_col), dtype=np.float32)
    for u, k in (boost or {}).items():
        w[units_col.to_numpy() == u] = k
    return w
