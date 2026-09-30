# TDT4173 × SINTEF: Hydropower Unit-Commitment Prediction

This project predicts, for each hour of a 168-hour weekly case, the probability that each of 14
generating units in the Tokke–Vinje hydropower system is **ON**. The targets are the commitment
decisions from SINTEF's SHOP optimiser. Scoring uses micro-averaged ROC-AUC over every
(case, hour, unit) cell.

Everything lives in [`sintef_project/`](sintef_project/). The task statement and data
documentation are in [`ML_Task_TDT4173.pdf`](sintef_project/ML_Task_TDT4173.pdf) and
[`Dataset_definitions_and_explanation.pdf`](sintef_project/Dataset_definitions_and_explanation.pdf).

---

## Current best model at a glance

| | |
|---|---|
| **Kaggle public score** | **0.98058** (first and only submission so far) |
| Honest offline estimate | 0.9638 ± 0.0115 (walk-forward, 6 × 6-month folds, 2020–2022) |
| Code | [`sintef_project/src/submit.py`](sintef_project/src/submit.py), function `run()` |
| Submitted file | [`sintef_project/submission.csv`](sintef_project/submission.csv) |
| Write-up | [`report/log/036_submission.tex`](sintef_project/report/log/036_submission.tex) |

**Architecture ("lean long format"):**

- **One row per (case, hour, unit).** That gives 2,922 train cases × 168 hours × 14 units =
  6.87M rows. A single binary XGBoost model is shared across all units, and each unit is
  identified by its features, not by a separate output head.
- **207 features:**
  - 15 pruned economic features (price and water-value spreads, margins, break-evens).
  - 35 terminal water-value features (read at start + 7 days, per the dataset docs).
  - 36 hour-level temporal-structure features (block/run structure, rolling windows, lags).
  - 141 per-unit features from a (case, unit) "count table" (`own_*`, `upstream_*`, absolute).
  - 5 static unit attributes (p_max, p_min, merit order, number of siblings).
- **The count model is used only for feature selection.** A 6-class XGBoost model over
  hours-ON bins is fitted to rank the count-table features. Its top 120 are kept, plus up to 30
  unit-relative features. The count model's predictions are **not** fed forward, and dropping
  them was worth +0.008.
- **Hyperparameters:** 600 rounds, depth 8, eta 0.05, `min_child_weight=500` (the most
  important parameter), subsample and colsample 0.8.
- **Training data:** all 2,922 labelled cases (2015–2022). Features are built over train and
  test together so that seasonal and relative features share one scale.

**Why the Kaggle score is higher than the offline estimate:** the submission trains on more
data than any validation fold, and the 2023–24 test period falls after the 2021–22 energy
crisis that makes the validation folds hard. See log 036 for details.

**Known weaknesses:**

- **Lio_G1** is the weakest unit (about 0.86 AUC across folds, driven by a regime change in
  mid-2021).
- Predicted schedules switch about 7.6 times a week, against 3.0 in reality.
- The hardest cases are autumn weeks with negative spreads and frequent unit switching.

---

## Reproducing the submission

Requires Python 3.12+ with `pandas`, `numpy`, `pyarrow`, `xgboost` (3.x), `scikit-learn` and
`pyyaml`. Development used Python 3.13, xgboost 3.4, pandas 2.3, numpy 2.2 and scikit-learn 1.7.

```bash
cd sintef_project

# 1. Build the base hourly tables from the raw data. Writes data/processed/{train,test}_hourly.parquet.
#    These files are already committed, so this step is only needed if the raw data changes.
python src/build_modeling_table.py

# 2. Train on all cases and write the submission.
python -c "import sys; sys.path.insert(0, 'src'); import submit; submit.run(out_path='submission.csv')"
```

⚠️ Training at full resolution uses about 6.9M rows × 207 features, which needs a lot of RAM
(the development machine had 26 GB). For a quick smoke test, `submit.run(stride=3)` trains on
every third case.

`data/processed/pruned_features.csv` (the 15 pruned economic features) is an input to
`submit.py`. It is committed and was produced in
[`src/modeling/feature_iteration.ipynb`](sintef_project/src/modeling/feature_iteration.ipynb)
(log 015).

---

## Repository map

```
sintef_project/
├── ML_Task_TDT4173.pdf, Dataset_definitions_and_explanation.pdf   task + data docs
├── kaggle_metric.py            official scoring function
├── prediction_mapping.csv      test cases to predict (725 weeks, 2023-01-01 → 2024-12-25)
├── sample_submission.csv       required submission format
├── submission.csv              ← our submitted predictions (public LB 0.98058)
├── schedule_visualization.ipynb
├── data/
│   ├── kernel/                 core inputs: price, inflow, storage, water value, UC labels
│   ├── extended/               topology (YAML), min-flow / min-volume constraints
│   └── processed/              derived tables (the 114 MB feature cache is git-ignored)
├── src/
│   ├── submit.py               ★ best model end to end
│   ├── build_modeling_table.py raw files → one row per (case, hour)
│   ├── features.py             feature stages (spreads, terminal water value, seasonal, …)
│   ├── hourly_structure.py     hour-level temporal/block features
│   ├── longform.py             wide → long (case, hour, unit) reshaping
│   ├── count_features.py, count_model.py, production_features.py, unit_relative.py
│   │                           (case, unit) count-table features and the ranking model
│   ├── validation.py, fastlab.py, screening.py   validation / experiment helpers
│   ├── modeling/               pipeline stage registry, evaluation, experiment notebooks
│   ├── EDA/                    exploratory notebooks (rounds 1–3b)
│   └── economics.py, slopes.py, seasonal_regime.py, daily_trace.py, exogenous.py
│                               feature families that were tried and REJECTED (kept for the record)
└── report/
    ├── leaderboard.md          ★ every model tried, with its score and protocol (best place to start)
    ├── log/001–036_*.tex       chronological lab notebook: what we did, why, what we found
    ├── main.tex                assembles the log into one PDF (build with latexmk; PDF not committed)
    ├── figures/ + figures.md   every figure, with how it was generated
    └── feature_backlog.md      feature ideas and their status
```

## How we got here (short version)

| Step | Score | Protocol |
|---|---|---|
| 14 independent XGBoost models on raw features | 0.647 | single year |
| Engineered economic features, pruned to top 15 | 0.816 | 4-fold rolling CV |
| Count-model features + budget blend | 0.935 | walk-forward |
| Long format, one row per (case, hour, unit) | 0.950 | 2021 hold-out |
| Terminal water-value off-by-one fix + tuning | 0.963 | 2021 hold-out |
| **Submission (all data)** | **0.981** | **Kaggle public** |

**Validation note:** random row or case splits leak heavily and inflate AUC by more than 0.10,
because consecutive weekly cases overlap by 144 hours. All reported numbers use temporal splits,
and walk-forward is the reference protocol. The full list of models, including about 18 rejected
feature families, is in [`report/leaderboard.md`](sintef_project/report/leaderboard.md).
