"""
Rolling-origin cross-validation for the unit-commitment task.

Why this exists: a single held-out year cannot resolve the size of effect most individual
features produce. The n=200 / 300 / 700 tree comparison spanned ~0.006 AUC on one year and we
judged it noise. Feature experiments need an error bar, so every scored experiment from here on
uses this harness.

Protocol: for each validation year Y, train on every case starting before Y and validate on the
cases starting in Y. Scored with the competition's own kaggle_metric.py (micro-average ROC-AUC).
Reports per-fold scores plus mean and spread, so a feature's effect can be compared against
fold-to-fold variation rather than assumed real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from kaggle_metric import score as kaggle_score  # noqa: E402

TRAIN_PARQUET = _PROJECT_ROOT / "data/processed/train_hourly.parquet"
LABELS_CSV = _PROJECT_ROOT / "data/kernel/Unit_commitment_decisions.csv"

VALIDATION_YEARS = (2019, 2020, 2021, 2022)
ID_COLS = ("run_no", "split", "starttime", "timestamp", "year")

DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", multi_strategy="multi_output_tree",
    objective="binary:logistic", eval_metric="logloss",
    n_jobs=-1, random_state=0,
)


def load_training_table(path=TRAIN_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["year"] = df["starttime"].dt.year
    return df


def default_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS and not c.startswith("on_")]


def _score_fold(model, X_val, val_df, units, solution_all) -> tuple[float, dict[str, float]]:
    """Official micro-AUC for one fold, plus per-unit AUCs."""
    proba = model.predict_proba(X_val)
    preds = pd.DataFrame({"run_no": val_df["run_no"].values, "hour": val_df["hour"].values})
    for i, u in enumerate(units):
        preds[f"pred_{u}"] = proba[:, i]

    wide = []
    for u in units:
        piv = preds.pivot(index="run_no", columns="hour", values=f"pred_{u}")
        piv.columns = [f"result_committed_{u}_t{h}" for h in piv.columns]
        wide.append(piv)
    submission = pd.concat(wide, axis=1).reset_index().rename(columns={"run_no": "Run No"})
    solution = solution_all[solution_all["Run No"].isin(val_df["run_no"].unique())]

    overall = kaggle_score(solution, submission)
    per_unit = {u: roc_auc_score(val_df[f"on_{u}"], preds[f"pred_{u}"]) for u in units}
    return overall, per_unit


def rolling_origin_cv(
    df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
    validation_years=VALIDATION_YEARS,
    verbose: bool = True,
):
    """Run the rolling-origin CV and return (summary_row, per_fold_df, per_unit_df).

    `feature_cols` defaults to every non-id, non-target column, so passing an augmented
    dataframe with new engineered columns is all an experiment needs to do.
    """
    df = load_training_table() if df is None else df
    feature_cols = default_feature_cols(df) if feature_cols is None else feature_cols
    params = {**DEFAULT_PARAMS, **(params or {})}

    target_cols = [c for c in df.columns if c.startswith("on_")]
    units = [c[len("on_"):] for c in target_cols]
    solution_all = pd.read_csv(LABELS_CSV)

    fold_rows, per_unit_rows = [], {}
    for year in validation_years:
        fit_df = df[df["year"] < year]
        val_df = df[df["year"] == year]
        if fit_df.empty or val_df.empty:
            continue

        model = XGBClassifier(**params)
        model.fit(fit_df[feature_cols], fit_df[target_cols].astype(int))
        overall, per_unit = _score_fold(model, val_df[feature_cols], val_df, units, solution_all)

        fold_rows.append({"validation_year": year, "micro_auc": overall,
                          "n_fit_cases": fit_df["run_no"].nunique(),
                          "n_val_cases": val_df["run_no"].nunique()})
        per_unit_rows[year] = per_unit
        if verbose:
            print(f"  fold {year}: micro-AUC {overall:.4f}  "
                  f"(fit {fit_df['run_no'].nunique()} cases, val {val_df['run_no'].nunique()})")

    per_fold = pd.DataFrame(fold_rows)
    per_unit_df = pd.DataFrame(per_unit_rows)
    summary = {
        "mean_micro_auc": per_fold["micro_auc"].mean(),
        "std_micro_auc": per_fold["micro_auc"].std(),
        "min_micro_auc": per_fold["micro_auc"].min(),
        "max_micro_auc": per_fold["micro_auc"].max(),
        "n_features": len(feature_cols),
    }
    if verbose:
        print(f"  MEAN {summary['mean_micro_auc']:.4f}  "
              f"SD {summary['std_micro_auc']:.4f}  "
              f"range [{summary['min_micro_auc']:.4f}, {summary['max_micro_auc']:.4f}]")
    return summary, per_fold, per_unit_df


# ---------------------------------------------------------------- fast screening mode

SCREENING_PARAMS = {**DEFAULT_PARAMS, "n_estimators": 150}
SCREEN_FIT_YEARS = (2015, 2016, 2017, 2018, 2019)
SCREEN_VAL_YEARS = (2020, 2021, 2022)


def screening_eval(
    df: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
    params: dict | None = None,
    fit_years=SCREEN_FIT_YEARS,
    val_years=SCREEN_VAL_YEARS,
    hour_stride: int = 2,
    verbose: bool = True,
):
    """Train ONCE, score several non-overlapping validation years. ~16x cheaper than full CV.

    Why this is a fair screen, and where it is not:

    - Training once and scoring multiple years makes the with/without-feature comparison
      *cleaner* than rolling-origin, because the training set is identical between variants;
      only the feature changes.
    - The train->validation gaps (1-3 years for 2020/2021/2022 off a 2015-2019 fit) bracket the
      real setting, which trains through 2022 and predicts 2023-24.
    - Absolute scores are biased low versus the full protocol, because each model sees less
      training data than its rolling-origin counterpart. Use it to *rank* candidates, then
      confirm winners with `rolling_origin_cv`.

    `hour_stride` subsamples hours within each case (2 = every other hour). Case structure and
    all case-relative features are unaffected because features are built before subsampling.
    """
    df = load_training_table() if df is None else df
    feature_cols = default_feature_cols(df) if feature_cols is None else feature_cols
    params = {**SCREENING_PARAMS, **(params or {})}

    target_cols = [c for c in df.columns if c.startswith("on_")]
    units = [c[len("on_"):] for c in target_cols]
    solution_all = pd.read_csv(LABELS_CSV)

    fit_df = df[df["year"].isin(fit_years)]
    if hour_stride > 1:
        fit_df = fit_df[fit_df["hour"] % hour_stride == 0]

    model = XGBClassifier(**params)
    model.fit(fit_df[feature_cols], fit_df[target_cols].astype(int))

    rows, per_unit_rows = [], {}
    for year in val_years:
        val_df = df[df["year"] == year]
        if val_df.empty:
            continue
        overall, per_unit = _score_fold(model, val_df[feature_cols], val_df, units, solution_all)
        rows.append({"validation_year": year, "micro_auc": overall,
                     "n_val_cases": val_df["run_no"].nunique()})
        per_unit_rows[year] = per_unit
        if verbose:
            print(f"  val {year}: micro-AUC {overall:.4f}")

    per_fold = pd.DataFrame(rows)
    summary = {
        "mean_micro_auc": per_fold["micro_auc"].mean(),
        "std_micro_auc": per_fold["micro_auc"].std(),
        "min_micro_auc": per_fold["micro_auc"].min(),
        "max_micro_auc": per_fold["micro_auc"].max(),
        "n_features": len(feature_cols),
        "n_fit_rows": len(fit_df),
        "mode": "screening",
    }
    if verbose:
        print(f"  MEAN {summary['mean_micro_auc']:.4f}  SD {summary['std_micro_auc']:.4f}  "
              f"(1 fit on {len(fit_df):,} rows, {len(feature_cols)} features)")
    return summary, per_fold, pd.DataFrame(per_unit_rows)


if __name__ == "__main__":
    print("Rolling-origin CV, current raw-feature baseline (n_estimators=300):")
    summary, per_fold, per_unit = rolling_origin_cv()
    print("\nSummary:", {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()})
