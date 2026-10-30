"""Check that Short notebook 2's `long_matrix` reproduces submit.py's long format exactly.

Builds the same inputs submit.run builds (via baseline_walkforward.prepare), then compares
  * the original: longform.to_long(...) + the cf_ block, as a pandas frame (unit categorical);
  * the notebook: long_matrix(...) as a float32 matrix with feature_types;
column names, every value (NaN-aware) and labels, and finally a 30-round XGBoost fitted on each
representation: identical matrices must give identical predictions.

    python src/deliverables_build/test_long_matrix.py [stride]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import baseline_walkforward as B  # noqa: E402
import longform as LF  # noqa: E402
import submit as SU  # noqa: E402

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nb = nbformat.read(SRC.parent / "deliverables/Short notebook 2.ipynb", as_version=4)
pipeline_cell = [c.source for c in nb.cells if c.cell_type == "code" and "def long_matrix" in c.source][0]
ns = {"np": np, "pd": pd, "_OWN": LF._OWN}
exec(compile(pipeline_cell[:pipeline_cell.index("\n\n\n", pipeline_cell.index("def long_matrix"))],
             "<long_matrix>", "exec"), ns)
long_matrix = ns["long_matrix"]

tr, ct, units, shared, stat = B.prepare()
feats = [c for c in ct.columns if c not in ("run_no", "unit", "count", "cls", "bin")]
keep = feats[:120]
ctx = ct.set_index(["run_no", "unit"])[keep]

# original path (submit.run's inner to_long)
L = LF.to_long(tr, units, shared, static=stat, case_stride=stride)
j = ctx.reindex(pd.MultiIndex.from_arrays([L["run_no"], L["unit"].astype(str)]))
L = pd.concat([L, pd.DataFrame(j.to_numpy(dtype=np.float32), columns=[f"cf_{c}" for c in keep],
                               index=L.index)], axis=1)
cols_o = [c for c in L.columns if c not in ("run_no", "hour", "y")]

run, hour, ucode, X, y, cols, ftypes, cats = long_matrix(tr, units, shared, stat, ctx, keep, stride)

print("rows", len(L), X.shape[0], "| columns identical:", cols_o == cols)
Xo = L[cols_o].copy()
Xo["unit"] = Xo["unit"].cat.codes
Xo = Xo.to_numpy(dtype=np.float32)
print("values identical:", np.array_equal(Xo, X, equal_nan=True),
      "| labels identical:", np.array_equal(L["y"].to_numpy(np.float32), y),
      "| run/hour identical:", np.array_equal(L["run_no"].to_numpy(), run)
      and np.array_equal(L["hour"].to_numpy(), hour),
      "| categories:", list(L["unit"].cat.categories) == cats)

from xgboost import QuantileDMatrix, train  # noqa: E402
p = dict(SU.POINT_PARAMS, nthread=4)
d1 = QuantileDMatrix(L[cols_o], label=L["y"], enable_categorical=True, max_bin=p["max_bin"])
d2 = QuantileDMatrix(X, label=y, feature_names=cols, feature_types=ftypes,
                     enable_categorical=True, max_bin=p["max_bin"])
b1, b2 = train(p, d1, 30), train(p, d2, 30)
p1 = b1.predict(QuantileDMatrix(L[cols_o], enable_categorical=True, ref=d1, max_bin=p["max_bin"]))
p2 = b2.predict(QuantileDMatrix(X, feature_names=cols, feature_types=ftypes,
                                enable_categorical=True, ref=d2, max_bin=p["max_bin"]))
print(f"30-round XGBoost predictions: max |diff| {np.abs(p1 - p2).max():.3e}")
