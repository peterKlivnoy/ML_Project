# TDT4173 × SINTEF: Hydropower Unit-Commitment Prediction

This project predicts, for each hour of a 168-hour weekly case, the probability that each of 14
generating units in the Tokke–Vinje hydropower system is **ON**. The targets are the commitment
decisions from SINTEF's SHOP optimiser. Scoring uses micro-averaged ROC-AUC over every
(case, hour, unit) cell.

Everything lives in [`sintef_project/`](sintef_project/). The task statement and data
documentation are in [`ML_Task_TDT4173.pdf`](sintef_project/ML_Task_TDT4173.pdf) and
[`Dataset_definitions_and_explanation.pdf`](sintef_project/Dataset_definitions_and_explanation.pdf).

---

## Course deliverables (`sintef_project/deliverables/`)

| file | what it is | checked |
|---|---|---|
| `Short notebook 1.ipynb` | LP + LightGBM, end to end from the raw data (all code inline) | reproduces `submission_physics.csv` to 2e-13; LP code reproduces the stored solutions bit for bit |
| `Short notebook 2.ipynb` | long-format XGBoost, public 0.98058 (all code inline) | design matrix verified identical to `submit.py`'s; full run compared with `submission.csv` |
| `Report.pdf` | the report, built from `report/Report.tex` (`pdflatex Report.tex` twice) | |

The notebooks are generated from the source modules by `src/deliverables_build/make_short_notebook{1,2}.py`,
so a change to `src/` is carried into them by re-running the generator. Place a notebook inside the
extracted `sintef_project/` folder (or next to it) and run it top to bottom.

---

## New: physics-first model (walk-forward 0.9912, not yet submitted)

The labels are the output of one deterministic optimiser (SHOP) whose inputs we have in full, so
instead of learning inputs -> commitment from scratch, [`src/shoplite.py`](sintef_project/src/shoplite.py)
rebuilds the weekly problem SHOP solved from the SHOP-ready topology YAML and solves it per case
with HiGHS (LP, ~2 s/case). A LightGBM model then learns only where our replica disagrees with SHOP.

| walk-forward, 6 half-year folds, stride 3 | mean micro-AUC | SD | test-like folds (2021-H2, 2022-H2) |
|---|---|---|---|
| submitted XGBoost model (re-run on the same folds) | 0.9619 | 0.0140 | 0.9644 |
| raw LP solution, **no learning at all** | 0.9710 | 0.0079 | 0.9748 |
| **LP + LightGBM** | **0.9912** | **0.0040** | **0.9934** |

All six folds and all 14 units improve (Lio 0.864 -> 0.979, Haukeli 0.920 -> 0.973). The
ablation says the optimiser is the result: the same LightGBM without the LP features scores 0.9483,
and removing the post-decision storage trace changes nothing (0.9910), so the model uses only
information available before the week starts. Three
conventions were measured from the labels rather than assumed (terminal water value, river flow
cost, sibling tie-break). Full write-up: [`report/log/037_first_principles_optimiser.tex`](sintef_project/report/log/037_first_principles_optimiser.tex).

```bash
cd sintef_project/src
python shoplite_batch.py lp                 # solve all 3,647 cases (~1.6 h on 4 cores; results committed)
python physics_model.py walkforward 3       # walk-forward table above
python physics_model.py submit ../submission_physics.csv
# or everything unattended, committing results after each step:
nohup bash run_physics_pipeline.sh > pipeline.log 2>&1 &
```

Needs `lightgbm` and `highspy` in addition to `requirements.txt`. The LP solutions for every case
are committed in `data/processed/shoplite_lp_{unit,case}.parquet`, so the walk-forward and the
submission run without re-solving.

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

## Getting started after cloning

Requires Python 3.12+. Development used Python 3.13.

```bash
git clone https://github.com/peterKlivnoy/ML_Project.git
cd ML_Project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core packages in [`requirements.txt`](requirements.txt) are pinned to the versions that
produced the submission. The notebook packages at the bottom of that file are only needed to
open the EDA and experiment notebooks.

A fresh clone already contains everything `submit.py` needs: the raw data in `data/kernel/` and
`data/extended/`, and the base tables and pruned feature list in `data/processed/`. The one
derived file that isn't committed is `data/processed/train_hourly_features.parquet` (114 MB,
over GitHub's file size limit). It's only a cache for the experiment notebooks. To regenerate it:

```bash
cd sintef_project/src && python -m modeling.pipeline --build-only
```

## Reproducing the submission

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
│   └── processed/              derived tables (the 114 MB feature cache is git-ignored, see above)
├── src/
│   ├── submit.py               ★ best model end to end
│   ├── shoplite.py             ★ first-principles replica of SHOP's weekly optimisation (HiGHS)
│   ├── shoplite_batch.py, shoplite_eval.py   solve every case / score the raw solution
│   ├── physics_features.py, physics_model.py ★ LP + LightGBM (walk-forward 0.9912, submission)
│   ├── baseline_walkforward.py walk-forward re-run of submit.py's model, fold by fold
│   ├── adversarial.py          which training period resembles 2023-24 (2021-H2)
│   ├── run_physics_pipeline.sh unattended end-to-end run
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
