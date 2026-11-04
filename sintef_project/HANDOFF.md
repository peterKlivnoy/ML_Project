# Handoff: finishing the TDT4173 SINTEF project

Status file for a new Claude Code session (run locally in this repository). Read this first, then
`sintef_project/report/report_rules.md` (conventions) and the grading PDF summary below.

**Deadline: Sunday 8 November 2026, 22:00 Oslo time** (Canvas upload + Kaggle selection). Late
submission costs 30 points.

Working branch: `claude/elegant-keller-0xnflm`. Its earlier work was merged into `main` through
PR #2 (peterKlivnoy/ML_Project#2). Everything after that is on the branch only and must be merged
again at the end.

---

## 1. Who and what

- **Group 300**, Kaggle team name **`[300]`**.
  - Anders Bærheim, student ID 114407.
  - Peter Klivnoy, student ID 114439.
  - Names, IDs and team name are already at the top of both notebooks and in the report title block.
  - Both notebooks get them from `src/deliverables_build/team.py`; the report has them in its preamble macros.
- **Selected predictions** (select both on Kaggle):

  | | model | file | public score |
  |---|---|---|---|
  | Short notebook 1 | LP + LightGBM (Track B) | `sintef_project/submission_physics.csv` | **0.99522** (best) |
  | Short notebook 2 | UCNet neural network (Track C, Anders) | `sintef_project/submission_nn.csv`, the saved run of the notebook | 0.99024 |

- **Three tracks**, all in the report:
  - **A:** XGBoost feature engineering, public 0.98058, *not selected*. Its notebook is in `sintef_project/notebooks/`.
  - **B:** our LP replica of SHOP used as a feature generator, plus LightGBM.
  - **C:** Anders's sequence network. It is rebuilt in `src/nn_phys.py` from his spec, saved as `sintef_project/report/track_c_specification.md`.

## 2. Grading rules that matter

From the course's "Project Submission and Grading" PDF:

- **Submit:** 2 selected Kaggle predictions; `Short notebook 1.ipynb` and `Short notebook 2.ipynb`; `Report.pdf`. All code must be in the notebooks.
- **Graders re-run the notebooks:**
  - offline, from the raw data;
  - after deleting everything except the 2 notebooks and **one config file** (`requirements.txt`);
  - on a "normal PC": at most 4 CPU cores, 32 GB RAM, 12 hours per notebook.
  - If you use a GPU, state its model and the software needed.
  - At least one notebook must repeat the best score. Small differences are accepted.
- **Hand-set hyperparameters** in the notebooks must be explained in the report.
- **Header:** names, student IDs and Kaggle team name at the top of every notebook and of the report. *Done.*
- **Possible deductions** (−3 each). The report needs clear section titles for each:
  - **EDA:** at least 4 of the 6 listed activities (domain knowledge, intuitive data, how the data was generated, individual features, pairs/groups, clean-up). *Done: Section 2 has one subsection per activity.*
  - **At least two types of predictor.** *Done:* logistic regression, random forest, XGBoost, LightGBM and the neural network. The comparison table still has to be filled in; see 3.1.
  - **Feature engineering, selection and/or extraction.** *Done: Section 4.*
  - **Model interpretation:** importance, SHAP, PDP or LIME. *Done: Section 7 has gain importance, TreeSHAP, ablation, LP calibration and an example week.*

## 3. What is left

**Status 2026-11-02, local session: 3.1–3.3 are done.**
- All placeholders are filled. LP + LightGBM on Track C's folds: 0.9921. Sensitivity: no variant moves the mean by more than 0.0004.
- Short notebook 2 has been executed on an Apple M4: 32 min of training. Its output `submission_nn.csv` scored 0.99024 and is the selected file; it correlates 0.9993 with Anders's earlier upload `submission_nn_phys.csv`.
- The lab log is out of the report (the files stay in `report/log/`). Track C's CV figures stay from Anders's original run and are labelled so in the report.
- The report is rebuilt.
- Left: merge into `main` (3.4, step 1), then the Kaggle selection and Canvas upload.

### 3.1 Fill the four placeholders in `sintef_project/report/Report.tex`

Search for these markers (three remain). Each needs one small table or paragraph built from a result file in `sintef_project/report/runs/`. If a result file is missing or incomplete, re-run its command; all of them run from `sintef_project/src/`.

| marker (section) | content | result file | command to (re)generate | time on 4 cores |
|---|---|---|---|---|
| ~~`% PREDICTOR TABLE` (5.1)~~ | **done** in the cloud session: table and text are in the report (LR 0.9833, RF 0.9895, XGBoost 0.9912, LightGBM 0.9912 mean) | `predictor_comparison_s3.csv` | — | — |
| `% SENSITIVITY` (6.2) | Table of the 9 LightGBM variants on the two test-like folds (submitted, num_leaves 31/255, min_data_in_leaf 100/2000, lr 0.1×300 / 0.025×1200, feature_fraction 0.5/1.0). Then one sentence on how much the AUC moves, honestly: if a variant beats the submitted one, say so; do not change the submitted model. | `lgbm_sensitivity_s3.csv` | `python lgbm_sensitivity.py 3` | ~30 min |
| `% NN-VS-LP` (5.4) | LP + LightGBM on Track C's yearly folds (2019–2022, 7-day gap, all cases) next to the network's 0.9837 / 0.9685 / 0.9658 / 0.9779 (mean 0.9740): a like-for-like comparison of the two selected models. | `lp_lightgbm_yearly_cv.csv` | `python lp_yearly_cv.py` | ~20 min |
| `% NN-CPU-TIME` (Section 8 table) and `% NN-CPU-TIME-APPENDIX` (Appendix A) | Runtime of Short notebook 2. **Measured on idle cores: 1.25 s per batch, so 1.9 min per epoch and ~1.9 h for 3 seeds × 20 epochs on 4 CPU cores. No GPU is needed** (an earlier 33 s/batch figure was contention from other jobs). The cloud session queued a full CPU run; if `deliverables/Short notebook 2.ipynb` has outputs, use its measured times. | `nn_cpu_timing.txt`, executed notebook | `python ../src/deliverables_build/execute_notebook.py "Short notebook 2.ipynb" "Short notebook 2.ipynb" . python3` from `deliverables/` | ~2 h |

The cloud session started all four runs. Check whether their result files are complete before re-running anything.

### 3.2 Execute Short notebook 2 (CPU ~2 h, or GPU ~40 min) — if the cloud session did not finish it

Check first whether `deliverables/Short notebook 2.ipynb` already has outputs (the cloud session queued a full run). If not, run it once: it proves the notebook works end to end and gives outputs and runtime for the report.

1. In a fresh virtual environment: `pip install -r sintef_project/deliverables/requirements.txt jupyter nbclient`. On a Mac, PyTorch uses MPS automatically.
2. Put Anders's `submission_nn_phys.csv` in `sintef_project/`, next to `data/`. The last cell then compares the re-run with it.
3. Run it from `sintef_project/deliverables/`:
   ```
   python ../src/deliverables_build/execute_notebook.py "Short notebook 2.ipynb" "Short notebook 2.ipynb" . python3
   ```
   Expect ~2 min for the features (a spot check asserts them against the original run), then ~40 min of training.
4. Commit the executed notebook. Put the measured runtime and the correlation with the uploaded file into the report (Section 8 table, Appendix A).

Feature status: the 99 features in `src/nn_phys.py` match Anders's checksums (`report/runs/nn_phys_feature_checksums.csv` against Appendix B of the spec). The largest difference is 1e-4 SD, which is rounding; the spot check is exact. The cloud machine has no GPU, but CPU training is fast enough (~2 h).

### 3.3 Rebuild and package

```
cd sintef_project/report && pdflatex Report.tex && pdflatex Report.tex && pdflatex Report.tex
cp Report.pdf ../deliverables/Report.pdf && rm -f Report.aux Report.log Report.out Report.toc Report.pdf
cd ../.. && python sintef_project/src/deliverables_build/flatten_report.py   # single-file Report.tex + zip
```

- **Before calling it done:** check the log for `^! ` errors and render a few pages to look at them.
- **Last build:** 97 pages, 1 negligible overfull box.

**Regenerating notebooks (careful):**
- `make_short_notebook1.py`, `make_short_notebook2.py` and `make_xgboost_notebook.py` write *unexecuted* notebooks over the executed ones.
- If you change only markdown, copy the executed notebook somewhere first, regenerate, then restore its outputs with `python src/deliverables_build/transplant_outputs.py <executed copy> <regenerated>`. That script asserts the code cells are identical.
- Short notebook 1's executed run took 2 h. Don't redo it unless its code changes.

### 3.4 Merge and submit

1. Commit everything, push the branch, open a PR into `main` and merge it, as with PR #2.
2. **Kaggle:** select `submission_physics.csv` (0.99522) and `submission_nn.csv` (0.99024). Check the team name is exactly `[300]`.
3. **Canvas** (before Nov 8, 22:00), all from `sintef_project/deliverables/`: `Short notebook 1.ipynb`, `Short notebook 2.ipynb`, `Report.pdf`, `requirements.txt`. Register for the exam.

---

## 4. Facts already established — don't redo

- **Short notebook 1 (LP + LightGBM):**
  - executed end to end from the raw data: 3,647 LPs in 2.0 h on 4 cores, features and training 6 min, peak ~5 GB;
  - output matches `submission_physics.csv` to 2.2e-13;
  - its LP solutions are byte-identical to `data/processed/shoplite_lp_*.parquet`;
  - also identical (0 difference on 41 cases) under the older pinned numpy/pandas.
- **Track A XGBoost notebook:**
  - executed, 41 min, peak 24.4 GB including swap;
  - selects 143 count features instead of the original 141 on x86 Linux, with or without threads (4 and 8 identical), so its output correlates 0.995 with `submission.csv` rather than matching it;
  - not selected any more, so this no longer matters for grading.
- **Walk-forward** (6 half-year folds, stride 3): LP + LightGBM 0.9912, Track A XGBoost 0.9619, the LP feature alone 0.9710.
- **Ablation:** without the LP features 0.9483; without the storage trace 0.9910.
- **Adversarial validation:** 2021-H2 (0.785) and 2022-H2 (0.879) are the test-like half-years.
- **Track C** (from the spec):
  - yearly CV: 2019 0.9837, 2020 0.9685, 2021 0.9658, 2022 0.9779, mean 0.9740;
  - without the physics features 0.9697; LightGBM on the same 99 features 0.9722.
- **Track C hyperparameters:** Anders set them by hand; no record of a search exists. The report says exactly that and gives the reasoning each value follows. Don't invent a tuning procedure.
- **Report structure:** a lookup table of graded items, then:
  1. Task & data
  2. EDA (six subsections named after the course list, plus train/test shift)
  3. Validation
  4. Feature engineering (extraction I/II/III, selection, inputs per model)
  5. Predictors (types compared, every model, Track B, Track C)
  6. Hyperparameters
  7. Interpretation
  8. Selected predictions
  9. Discussion
  - Appendix A Reproducibility; Appendix B full lab log.

## 5. Where things are

| path | what |
|---|---|
| `sintef_project/deliverables/` | the two short notebooks, `requirements.txt`, `Report.pdf`, single-file `Report.tex` + `Report_source.zip` |
| `sintef_project/report/Report.tex` | report source; `preamble.tex`, `log/*.tex` (lab log), `figures/` |
| `sintef_project/report/runs/` | every experiment's results |
| `sintef_project/report/leaderboard.md`, `figures.md` | experiment and figure registers (`report_rules.md` says to keep them updated) |
| `sintef_project/src/shoplite.py`, `physics_features.py`, `physics_model.py` | Track B |
| `sintef_project/src/nn_phys.py` | Track C (`features`, `cv`, `submit` commands) |
| `sintef_project/src/predictor_comparison.py`, `lgbm_sensitivity.py`, `lp_yearly_cv.py` | the three experiments behind the placeholders |
| `sintef_project/src/report_figures.py` | all `physics_*` figures |
| `sintef_project/src/deliverables_build/` | notebook generators, `team.py`, `flatten_report.py`, `execute_notebook.py`, `transplant_outputs.py` |
