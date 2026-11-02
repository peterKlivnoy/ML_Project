# Report conventions

This folder is the running record of the project: what we did, why, what we found, and how
to reproduce every figure. It exists so the final report is mostly assembly, not a
from-scratch write-up at the end. These are the rules for keeping it that way.

## Folder layout

```
report/
  main.tex           <- master document; \input's every log entry, in order
  preamble.tex        <- shared packages / macros / page setup
  report_rules.md      <- this file
  figures.md           <- one table row per figure: what it is, how it was made
  figures/              <- the actual image files referenced by the log and figures.md
  leaderboard.md         <- one table row per model/experiment: config + validation score
  log/
    001_topic_name.tex  <- one numbered entry per work session / topic
    002_topic_name.tex
    ...
```

## The log (`log/*.tex`)

- **One file per discrete work session or topic**, not per calendar day — multiple entries
  can share a date if several things happened in one sitting.
- **Numbered sequentially** (`001_`, `002_`, ...) in the order they were written. The number
  is the only thing that has to scale; nothing here requires renumbering old entries when new
  ones are added, so this stays cheap indefinitely across a multi-week project.
- Each entry starts with a dated header (`\subsection*{2026-09-21 --- Topic}`) so the
  chronology survives even though filenames aren't date-based.
- Each entry documents, in roughly this order: **what we did**, **why** (the reasoning /
  decision, not just the action), **what we found**, and **what it implies for next steps**.
  Write it as it happens — this is a log, not retrospective prose.
- Reference figures with `\includegraphics` pointing at `figures/<file>`, and cross-reference
  the matching row in `figures.md` in the caption or text so the two stay linked.
- When a new file is added, add one `\input{log/0NN_name}` line to `main.tex`, in order.
  That's the only edit `main.tex` itself should ever need for routine logging.

## Figures (`figures/` + `figures.md`)

- Every figure that appears in the log (or eventually the final report) is a real file in
  `figures/`, not an inline plot only visible inside a notebook.
- `figures.md` holds exactly one markdown table, one row per figure, columns:

  | Column | Contents |
  |---|---|
  | `id` | filename without extension, e.g. `eda_01_unit_on_rate` |
  | `file` | path relative to `report/`, e.g. `figures/eda_01_unit_on_rate.png` |
  | `source` | the notebook/script that generated it, e.g. `src/EDA/eda.ipynb` |
  | `generated_by` | the specific cell/function/command that produced it |
  | `description` | what's plotted, what data it covers, and the one-line takeaway |
  | `date_added` | when it was generated |

- Filenames are `<source-prefix>_<NN>_<short-slug>.png`, numbered in the order they appear
  within their source notebook (`eda_01_...`, `eda_02_...`, `baseline_01_...`). This keeps
  filenames stable and sortable as more notebooks are added (`modeling_01_...`, etc.).
- If a figure is regenerated (updated data, restyled plot), overwrite the same file and update
  its `figures.md` row rather than creating a new numbered entry, unless the content is
  genuinely a different figure.

## Validation protocol

Model comparisons are only meaningful if every model is scored the same way. **Superseded once
already** — the original single-year protocol (train 2015–2021, validate 2022) could not resolve
feature-sized effects and was replaced in log 010.

**Standing protocol — rolling-origin CV.** For each validation year in 2019–2022: train on every
case starting before it, validate on cases starting in it. Scored with `kaggle_metric.py`
(micro-average ROC-AUC). Implemented in `src/modeling/evaluation.py:rolling_origin_cv`.

- **Judge on paired per-fold deltas and the sign pattern, not the mean.** Fold-to-fold SD on the
  raw-feature baseline was **0.061**, roughly ten times a typical feature effect. A change that
  improves all four folds is credible; one that moves the mean via a single fold is not.
- The raw-feature baseline under this protocol is **0.7074**. Scores from before log 010 used the
  single-year protocol and are **not comparable**.

**Screening mode** (`evaluation.py:screening_eval`) trains once on 2015–2019 and scores
2020/2021/2022, with 150 trees and every second training hour — about 16x cheaper, since the
four-fold cost was four *trainings* and validation is nearly free. Use it to **rank** candidates
during search; confirm anything that looks real with the full protocol before quoting a number.
Screening scores are biased low and must be labelled as such in `leaderboard.md`.

**Any entry that deviates must say so explicitly**, in the text and in its `leaderboard.md` row.
An unlabelled deviation silently invalidates every comparison against it.

## Feature backlog (`feature_backlog.md`)

- Holds the candidate feature space, grouped into themes. Each row is one buildable feature.
- **Every row carries a status**: `todo` / `building` / `tested` / `kept` / `rejected`. A row that
  has been through the harness records its CV delta and per-fold sign pattern inline.
- **Never delete a rejected row.** A feature that failed is a result, and an expensive one to
  rediscover. The seasonal-percentile family is the worked example: highest univariate
  correlation in the project, and it made the model worse.
- **Correlation is a screening signal, not evidence.** Priority ordering derived from univariate
  correlation proved actively misleading (log 013). Order by *tested* effect once tests exist,
  and treat untested correlation-based ordering as a guess.

## Leaderboard (`leaderboard.md`)

- One markdown table, one row per model/experiment that was actually run and scored, columns:

  | Column | Contents |
  |---|---|
  | `model` | short name, e.g. `xgb_14_independent` |
  | `log_entry` | which `log/0NN_*.tex` entry describes it |
  | `validation_protocol` | `standard` (see above) or a short description of the deviation |
  | `overall_micro_auc` | the headline number |
  | `notes` | one line: what's notable (e.g. worst/best unit, what changed vs. the prior row) |
  | `date` | when it was run |

- Add a row the same time you write the log entry for that model — the leaderboard is a
  by-product of logging, not a separate task to remember.
- Never delete or overwrite a row to "clean up" a worse-performing experiment. A negative result
  (a change that didn't help) is exactly as valuable to keep as a positive one.

## Revising earlier decisions and findings

Multi-week projects change their minds, and that's fine — but the log should show that
honestly rather than quietly rewriting history.

- **Don't edit an old entry's substance** to match a later, better understanding. If entry 002
  claimed something that entry 007 later overturns, entry 002 stays as written.
- Instead, add a one-line marker to the old entry using `\supersededby{<label>}{<short reason>}`
  (defined in `preamble.tex`), pointing forward to the entry that revises it. The new entry
  explains the revision in full and links back with `Section~\ref{<old label>}` as usual.
- This applies to findings, design decisions, and abandoned plans alike (e.g. "we said we'd test
  X before Y, then didn't" is worth a marker, not a silent deviation) — see log entry 005 for an
  example of narrating a resequencing decision in the new entry.

## Building the document

From `report/`: `latexmk -pdf main.tex` (or `pdflatex main.tex` run twice for cross-references).
The document should always compile cleanly from a fresh checkout — if it doesn't, fix that
before adding new content on top of it.

## What doesn't belong here

- Code belongs in `src/`, not in the log as inline listings of any real length — reference the
  file/notebook by path instead.
- Data files belong in `data/`. Nothing under `report/` should be a required input to any
  pipeline — this folder is documentation *of* the project, not part of it.
