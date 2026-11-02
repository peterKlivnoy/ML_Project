"""Generate `Short notebook 2.ipynb`: the sequence neural network (UCNet) submission, self-contained.

The model was built by Anders Bærheim (Kaggle public 0.99013); src/nn_phys.py is a rebuild from
his specification whose 99 features match his reference checksums. The notebook is that module's
definitions, lifted by AST, plus the cells that run them.

    python src/deliverables_build/make_short_notebook2.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from team import DATA, HEADER


def defs(path: Path, names: list[str]) -> dict[str, str]:
    """Source of the named top-level definitions."""
    import ast
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    out = {}
    for node in ast.parse(text).body:
        if getattr(node, "name", None) in names:
            out[node.name] = "".join(lines[node.lineno - 1:node.end_lineno]).rstrip() + "\n"
    assert not set(names) - set(out), set(names) - set(out)
    return out

SRC = Path(__file__).resolve().parents[1]
OUT = SRC.parent / "deliverables" / "Short notebook 2.ipynb"
MODULE = SRC / "nn_phys.py"

text = MODULE.read_text()
constants = text[text.index("H = 168\n"):text.index("# ---------------------------------------------------------------------------------------------\n# data")]
constants = constants.replace("ROOT = Path(__file__).resolve().parents[1]\n", "")
N = defs(MODULE, ["_read", "load", "_roll", "_rank", "_b", "_hours", "_hourly", "_daily",
                  "_head_and_glob", "base_features", "_runs", "physics_features", "build_X4",
                  "Scaler", "device", "make_model", "train_predict", "predict", "write_submission"])


def join(*names):
    return "\n\n".join(N[n].rstrip() for n in names) + "\n"


md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Short notebook 2 — sequence neural network on closed-form physics features (Kaggle public 0.99013)

""" + HEADER + """

This notebook reproduces our second selected prediction from the raw competition data. The model was
built by Anders Bærheim (Track C in the report):

1. **83 base features** per (case, unit, hour) from prices, water values, volumes, inflow,
   constraints and calendar;
2. **16 closed-form physics features** from the generator efficiency curves in the topology file —
   break-even prices, a weekly water budget, a greedy schedule over the best-margin hours, and block
   profit against start cost. Arithmetic, sorting and run-length scans only: no optimisation solver;
3. **one sequence network (UCNet)** per case — temporal convolution and a bidirectional GRU along the
   168 hours of each unit, self-attention across the 14 units at each hour, a learned unit
   embedding — trained on all 2,922 labelled cases with three seeds; the submission is the mean of
   the three seeds' probabilities, with no post-processing.

**Hardware.** No GPU is needed: on 4 CPU cores training takes about 2 hours (1.9 minutes per epoch,
3 seeds × 20 epochs). The original run used an Apple M-series GPU (PyTorch, MPS backend), about 40
minutes; the notebook uses CUDA if available, then MPS, then the CPU. Training is not
bit-reproducible across devices, so a re-run lands near 0.990 rather than exactly on 0.99013. The
features are deterministic: Section 3 checks them against values recorded from the original run.

**Memory:** the feature tensor is 3,647 × 14 × 168 × 99 float32 ≈ 3.4 GB; peak about 8 GB.

""" + DATA),
    code('''from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Folder holding data/kernel, data/extended, prediction_mapping.csv, sample_submission.csv
CANDIDATES = [Path("."), Path(".."), Path("sintef_project"), Path("../sintef_project")]
ROOT = next(p.resolve() for p in CANDIDATES if (p / "data/kernel").is_dir())
SEEDS = (0, 1, 2)
EPOCHS = 20
OUT_PATH = Path("submission_nn.csv")
print("data root:", ROOT, "| torch", torch.__version__)

''' + constants.rstrip() + "\n"),
    md("## 1. Data"),
    code(join("_read", "load") + "\n\nt0 = time.time()\nD = load(ROOT)\n"
         "print(f\"{D['n_train']} labelled cases, {len(D['starts']) - D['n_train']} test cases\")"),
    md("""## 2. Features

**Base features (83):** economic core (price minus the intake reservoirs' terminal water value, and
its daily and rolling views), price shape within the week and day, water values up- and downstream,
reservoir fill, inflow and its anomaly against the pre-2015 climatology, minimum-flow and
minimum-volume constraints, the observed volume path over the week, static unit attributes,
Norwegian local calendar and holidays, and cross-unit context.

**Physics features (16):** from each generator's efficiency envelope at the plant head,
break-even prices at minimum, full and marginal load; hours of water the plant can afford over the
week and hours it is forced to run to avoid spilling; a greedy schedule that runs the plant in its
best-margin hours; and, for every run of positive-margin hours, its profit against the start cost."""),
    code(join("_roll", "_rank", "_b", "_hours", "_hourly", "_daily", "_head_and_glob", "base_features")),
    code(join("_runs", "physics_features", "build_X4")),
    md("""## 3. Build the feature tensor and check it

Reference values recorded from the original run (training case 0, Tokke G1, hour 10)."""),
    code('''X, names = build_X4(D)
n = D["n_train"]
print(f"X4 {X.shape} in {time.time() - t0:.0f}s")
ref = {"price": 27.13, "wv_end_up": 19.65, "margin_end": 7.48, "fill_up": 0.71282,
       "inflow_sys_anom": 1.79670, "hod": 11, "phys_m_full": 7.37007, "phys_afford_h": 0.141246,
       "phys_greedy_k": 0.141246, "phys_block_profit": 166309.14}
got = {k: float(X[0, 1, 10, names.index(k)]) for k in ref}
assert all(abs(got[k] - v) <= 1e-4 * max(1.0, abs(v)) for k, v in ref.items()), got
print("spot check against the original run: OK")'''),
    md("""## 4. Model

Robust per-feature scaling (median, 10–90% range, clipped at ±8) fitted on every third training
case; UCNet with hidden width 96; AdamW with a one-cycle learning-rate schedule; binary
cross-entropy over all unit-hours. Hyperparameters and how they were chosen: report, Section 6."""),
    code(join("Scaler", "device", "make_model", "train_predict", "predict", "write_submission")),
    md("## 5. Train three seeds on all labelled cases, average, write the submission"),
    code('''dev = device()
print("device:", dev)
sc = Scaler().fit(X[:n:3])
Xtr, Xte = sc(X[:n]), sc(X[n:])
del X
P = []
for s in SEEDS:
    t1 = time.time()
    P.append(train_predict(Xtr, D["Y"], Xte, seed=s, epochs=EPOCHS, dev=dev))
    print(f"seed {s} done in {(time.time() - t1) / 60:.1f} min")
P = np.mean(P, 0)
submission = write_submission(P, D["test_runs"], OUT_PATH, ROOT)
print(f"wrote {OUT_PATH} {submission.shape} | mean probability {P.mean():.4f} | {(time.time() - t0) / 60:.0f} min")'''),
    code('''# Optional: compare with the file uploaded to Kaggle, if it is next to the data.
ref_path = ROOT / "submission_nn_phys.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path).set_index("Run No").loc[submission["Run No"]]
    a, b = submission.drop(columns="Run No").to_numpy().ravel(), ref.to_numpy().ravel()
    print(f"correlation {np.corrcoef(a, b)[0, 1]:.4f}; max |difference| {np.abs(a - b).max():.3f}; "
          f"same side of 0.5 in {((a > 0.5) == (b > 0.5)).mean():.1%} of cells")'''),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({sum(c.source.count(chr(10)) for c in cells if c.cell_type == 'code')} code lines)")
