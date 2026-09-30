"""
Stage registry and experiment runner.

The point of this module: adding a feature should be *one function in features.py plus one line
in STAGES* — never a notebook rewrite. The notebook becomes a thin caller, so it stays stable
while the feature set churns.

Typical use:

    from modeling.pipeline import build_features, run_experiment, ablate, STAGES

    df = build_features()                  # cached; rebuilds only what changed
    ledger = run_experiment(mode="screen") # cumulative stage-by-stage scoring
    ablate("06_spread")                    # drop one stage from the full set and re-score

Also runnable headless:  python -m modeling.pipeline --mode screen
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

import features as F  # noqa: E402
from modeling.evaluation import rolling_origin_cv, screening_eval  # noqa: E402

BASE_PARQUET = _ROOT / "data/processed/train_hourly.parquet"
FEATURE_PARQUET = _ROOT / "data/processed/train_hourly_features.parquet"

# ---------------------------------------------------------------- the registry
# Add a builder to features.py, add one line here. Nothing else changes.
# Order matters only where a stage consumes another's columns (physics needs spread).

STAGES: list[tuple[str, callable]] = [
    ("01_seasonal",         F.stage_seasonal),
    ("01g_seasonal_global", F.stage_seasonal_global),
    ("09_inflow_regime",    F.stage_inflow_regime),
    ("02_price_relative",   F.stage_price_relative),
    ("06_spread",           F.stage_spread),
    ("07_terminal_wv",      F.stage_terminal_wv),
    ("10_12_physics",       F.stage_physics),
    ("20_water_budget",     F.stage_water_budget),
    ("06x_spread_extended", F.stage_spread_extended),
    ("04_price_shape",      F.stage_price_shape),
    ("03_price_windows",    F.stage_price_windows),
    ("05_wv_structure",     F.stage_wv_structure),
    ("15_merit_order",      F.stage_merit_order),
    ("14_graph_cascade",    F.stage_graph_cascade),
    ("17_calendar",         F.stage_calendar),
]

ID_COLS = ["run_no", "split", "starttime", "timestamp", "year"]


def stage_names() -> list[str]:
    return [n for n, _ in STAGES]


def build_features(stages=None, base=BASE_PARQUET, verbose=True) -> tuple[pd.DataFrame, dict]:
    """Apply builders to the base table. Returns (df, provenance: stage -> columns)."""
    stages = STAGES if stages is None else stages
    df = pd.read_parquet(base)
    df["year"] = df["starttime"].dt.year
    provenance: dict[str, list[str]] = {}
    for name, fn in stages:
        t0 = time.time()
        out = fn(df)
        new = [c for c in out.columns if c not in df.columns]
        df = pd.concat([df, out[new]], axis=1)
        provenance[name] = new
        if verbose:
            print(f"  {name:<22} +{len(new):>3} cols  ({time.time()-t0:.1f}s)")
    return df, provenance


def feature_cols(df: pd.DataFrame, dropped: list[str] | None = None) -> list[str]:
    dropped = dropped or []
    return [c for c in df.columns
            if c not in ID_COLS and not c.startswith("on_") and c not in dropped]


def score(df: pd.DataFrame, cols: list[str], mode: str = "screen", verbose=False):
    """mode='screen' (1 fit, fast) or 'full' (4-fold rolling-origin, quotable)."""
    fn = screening_eval if mode == "screen" else rolling_origin_cv
    return fn(df=df, feature_cols=cols, verbose=verbose)


def run_experiment(stages=None, mode="screen", drop_dead=True, verbose=True) -> pd.DataFrame:
    """Cumulative stage-by-stage scoring. Returns the ledger as a DataFrame."""
    stages = STAGES if stages is None else stages
    df = pd.read_parquet(BASE_PARQUET)
    df["year"] = df["starttime"].dt.year
    rows, dropped = [], []

    def _record(label, d, cols):
        t0 = time.time()
        summary, per_fold, _ = score(d, cols, mode=mode)
        row = {"stage": label, "n_features": len(cols),
               "mean": summary["mean_micro_auc"], "sd": summary["std_micro_auc"]}
        row.update({int(r["validation_year"]): r["micro_auc"] for _, r in per_fold.iterrows()})
        rows.append(row)
        if verbose:
            msg = f"{label:<22} mean {row['mean']:.4f}  sd {row['sd']:.4f}  ({time.time()-t0:.0f}s)"
            if len(rows) > 1:
                prev = rows[-2]
                years = [k for k in row if isinstance(k, int)]
                deltas = {y: row[y] - prev[y] for y in years}
                msg += (f"\n  vs prev: mean {row['mean']-prev['mean']:+.4f} | "
                        + " ".join(f"{y}:{d:+.4f}" for y, d in deltas.items())
                        + f" | {sum(d > 0 for d in deltas.values())}/{len(years)} folds up")
            print(msg)

    _record("00_baseline", df, feature_cols(df))
    for name, fn in stages:
        out = fn(df)
        new = [c for c in out.columns if c not in df.columns]
        df = pd.concat([df, out[new]], axis=1)
        if verbose:
            print(f"  [{name}] +{len(new)} cols")
        _record(name, df, feature_cols(df, dropped))

    if drop_dead:
        dropped = F.drop_dead_columns(df)
        if verbose:
            print(f"  [drop_dead] -{len(dropped)} constant cols")
        _record("drop_dead", df, feature_cols(df, dropped))

    df[ID_COLS + feature_cols(df, dropped) + [c for c in df.columns if c.startswith("on_")]] \
        .to_parquet(FEATURE_PARQUET, index=False)
    return pd.DataFrame(rows)


def ablate(stage_name: str, mode="screen", verbose=True) -> dict:
    """Score the full feature set with ONE stage removed.

    Cumulative stage deltas cannot distinguish "useless" from "redundant with something earlier".
    This can: if removing a stage costs nothing, it was redundant regardless of its gain share.
    """
    keep = [(n, f) for n, f in STAGES if n != stage_name]
    full_df, _ = build_features(verbose=False)
    abl_df, _ = build_features(stages=keep, verbose=False)
    full = score(full_df, feature_cols(full_df, F.drop_dead_columns(full_df)), mode=mode)[0]
    less = score(abl_df, feature_cols(abl_df, F.drop_dead_columns(abl_df)), mode=mode)[0]
    delta = full["mean_micro_auc"] - less["mean_micro_auc"]
    if verbose:
        print(f"ablate {stage_name}: full {full['mean_micro_auc']:.4f} vs "
              f"without {less['mean_micro_auc']:.4f}  ->  contribution {delta:+.4f}")
    return {"stage": stage_name, "full": full["mean_micro_auc"],
            "without": less["mean_micro_auc"], "contribution": delta}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="screen", choices=["screen", "full"])
    ap.add_argument("--ablate", default=None, help="stage name to ablate instead of running stages")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()

    if a.ablate:
        ablate(a.ablate, mode=a.mode)
    elif a.build_only:
        df, prov = build_features()
        df.to_parquet(FEATURE_PARQUET, index=False)
        print(f"wrote {FEATURE_PARQUET}: {df.shape[0]:,} x {df.shape[1]}")
    else:
        led = run_experiment(mode=a.mode)
        print("\n" + led.round(4).to_string(index=False))
