"""Generate `Short notebook 2.ipynb`: the XGBoost submission (public 0.98058), self-contained.

Inlines every definition `submit.run` reaches across eight modules (src/deliverables_build/
extract.py walks the call graph), plus the raw -> hourly table step from build_modeling_table.py,
so the notebook needs only the competition data. Changes against the modules are limited to:
  * module prefixes removed (one namespace); `hourly_structure.build` -> `hs_build`,
    `production_features.build` -> `pf_build`; duplicate `_ROOT`/`K`/`_daily`/`san` kept once;
  * the two intermediate files the pipeline read are produced in memory instead: the hourly base
    tables (built from raw CSVs) and the 15-feature pruned list (a constant, from log 015).

    python src/deliverables_build/make_short_notebook2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract as E  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
OUT = SRC.parent / "deliverables" / "Short notebook 2.ipynb"

MODS = {"submit": "submit.py", "features": "features.py", "count_features": "count_features.py",
        "count_model": "count_model.py", "hourly_structure": "hourly_structure.py",
        "longform": "longform.py", "production_features": "production_features.py",
        "unit_relative": "unit_relative.py", "pipeline": "modeling/pipeline.py"}
ROOTS = [("submit", "run"), ("submit", "COUNT_PARAMS"), ("submit", "POINT_PARAMS"),
         ("submit", "POINT_ROUNDS"), ("submit", "N_CORE"), ("submit", "N_UNITREL")]
RENAME = {("hourly_structure", "build"): "hs_build", ("production_features", "build"): "pf_build"}
# defined identically (or as an alias) in more than one module: keep the first, drop the rest
DROP = {("submit", "_ROOT"), ("features", "_ROOT"), ("count_features", "_ROOT"),
        ("production_features", "_ROOT"), ("unit_relative", "_ROOT"),
        ("production_features", "K"), ("production_features", "_daily"), ("unit_relative", "san")}


def sub(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    assert n == count, f"expected {count} match(es), found {n}: {old[:70]!r}"
    return text.replace(old, new)


mods = E.load_modules(SRC, MODS)
order = [k for k in E.reach(mods, ROOTS) if k not in DROP]
names = [n for _, n in order]
dups = {n for n in names if names.count(n) > 1} - {v for v in RENAME.values()}
assert not {n for n in dups if not any((m, n) in RENAME for m, _ in order)}, f"unresolved: {dups}"


def code_for(module_names: list[str]) -> str:
    parts, emitted = [], set()
    for key in order:
        node = mods[key[0]].defs[key[1]]
        if key[0] in module_names and id(node) not in emitted:   # tuple targets share a node
            emitted.add(id(node))
            src = E.rewrite(mods, key, RENAME)
            if key in RENAME:                      # the definition itself
                src = src.replace(f"def {key[1]}(", f"def {RENAME[key]}(", 1)
            parts.append(src)
    return "\n\n".join(parts)


# ---- the raw -> hourly table step (build_modeling_table.main), returning frames ----------
bmt = (SRC / "build_modeling_table.py").read_text()
main = bmt[bmt.index("def main():"):bmt.index('if __name__ == "__main__":')].rstrip() + "\n"
main = sub(main, "def main():", 'def build_base_tables():\n    """Raw CSVs -> one row per (case, hour): joined inputs, targets for train cases."""')
main = main[:main.index("    train_df.to_parquet(")] + (
    "    # the pipeline was developed on these tables read back from parquet (nanosecond\n"
    "    # timestamps); pandas >= 3 builds microseconds, so restore the original resolution\n"
    "    for d in (train_df, test_df):\n"
    "        for c in (\"starttime\", \"timestamp\"):\n"
    "            d[c] = d[c].astype(\"datetime64[ns]\")\n"
    "    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)\n")
main = sub(main, 'f"{DATA}/', 'ROOT / f"data/', 5)
main = sub(main, 'pd.read_csv("prediction_mapping.csv")', 'pd.read_csv(ROOT / "prediction_mapping.csv")')
main = main.replace(' f"data/kernel/', ' "data/kernel/').replace(' f"data/extended/', ' "data/extended/')

# ---- submit: read the in-memory tables and the pruned list --------------------------------
submit_src = code_for(["submit", "pipeline"])
submit_src = sub(submit_src,
                 '    tr = pd.read_parquet(_ROOT / "data/processed/train_hourly.parquet")\n'
                 '    te = pd.read_parquet(_ROOT / "data/processed/test_hourly.parquet")\n',
                 "    tr, te = build_base_tables()\n")
submit_src = sub(submit_src,
                 '    pruned = pd.read_csv(_ROOT / "data/processed/pruned_features.csv")["feature"].tolist()\n',
                 "    pruned = list(PRUNED_FEATURES)\n")
assert "read_parquet" not in submit_src and "processed/" not in submit_src
# ---- memory: the original long-format assembly peaks above 14 GB at full resolution --------
# (pandas concat of 14 per-unit frames, a merge, a second concat, then L[cols] copies). The
# matrix below has the same columns, row order and float32 values, built in place; the
# equivalence is checked in tests (src/deliverables_build/test_long_matrix.py).
submit_src = sub(submit_src, """    both = pd.concat([tr, te], ignore_index=True)
    ct = build_count_table(both, units)
""", """    both = pd.concat([tr, te], ignore_index=True)
    ct = build_count_table(both, units)
    del both
    gc.collect()
""")
old_block = submit_src[submit_src.index("    def to_long(d, is_train, cs):\n"):
                       submit_src.index('    V["p"] = bst.predict(dva)\n') + len('    V["p"] = bst.predict(dva)\n')]
submit_src = sub(submit_src, old_block, """    L_run, L_hour, L_unit, Xtr, ytr, cols, ftypes, cats = long_matrix(
        tr, units, shared, stat, ctx, keep, stride)
    print(f"long: train {Xtr.shape[0]:,} rows, {len(cols)} features | {time.time()-t0:.0f}s",
          flush=True)
    del tr, hs_tr                                 # not used again; frees ~1.7 GB before the peak
    gc.collect()
    dtr = QuantileDMatrix(Xtr, label=ytr, feature_names=cols, feature_types=ftypes,
                          enable_categorical=True, max_bin=POINT_PARAMS["max_bin"])
    del Xtr, L_run, L_hour, L_unit
    gc.collect()
    bst = xgb_train(POINT_PARAMS, dtr, num_boost_round=POINT_ROUNDS)
    print(f"fitted {POINT_ROUNDS} rounds | {time.time()-t0:.0f}s", flush=True)
    V_run, V_hour, V_unit, Xte, _, _, _, _ = long_matrix(te, units, shared, stat, ctx, keep, 1)
    dva = QuantileDMatrix(Xte, feature_names=cols, feature_types=ftypes, enable_categorical=True,
                          ref=dtr, max_bin=POINT_PARAMS["max_bin"])
    V = pd.DataFrame({"run_no": V_run, "hour": V_hour,
                      "unit": pd.Categorical.from_codes(V_unit, categories=cats)})
    V["p"] = bst.predict(dva)
""")

LONG_MATRIX = r"""def long_matrix(w, units, shared, static, ctx, keep, case_stride=1):
    '''(case, hour, unit) design matrix, as `to_long` + the cf_ block, without pandas copies.

    Columns: shared, own_*, unit (categorical code), static attributes, cf_* count features --
    the order `to_long(...)` then `concat([L, blk])` produced. Rows: unit-major, cases in order.
    '''
    if case_stride > 1:
        sel = np.sort(w["run_no"].unique())[::case_stride]
        w = w[w["run_no"].isin(sel)]
    shared = [c for c in shared if c in w.columns]
    own_src, own_names = {}, []
    for u in units:
        src = {}
        for f in _OWN:
            if f"cnt_{f}_{u}" in w.columns:
                src[f"own_{f}"] = f"cnt_{f}_{u}"
        for c in w.columns:
            for pre in ("hs_", "nb_"):
                if c.startswith(pre) and c.endswith("_" + u):
                    src["own_" + c[len(pre):-(len(u) + 1)]] = c
                    break
        own_src[u] = src
        own_names += [k for k in src if k not in own_names]
    stat_cols = [c for c in static.columns if c != "unit"]
    cols = shared + own_names + ["unit"] + stat_cols + [f"cf_{c}" for c in keep]
    ftypes = ["c" if c == "unit" else "q" for c in cols]
    cats = sorted(units)                          # pandas orders string categories this way
    n, U = len(w), len(units)
    X = np.full((n * U, len(cols)), np.nan, np.float32)
    y = np.empty(n * U, np.float32)
    run = np.tile(w["run_no"].to_numpy(), U)
    hour = np.tile(w["hour"].to_numpy(), U)
    ucode = np.repeat([cats.index(u) for u in units], n).astype(np.int16)
    ns, no, ju = len(shared), len(own_names), len(shared) + len(own_names)
    shared_block = w[shared].to_numpy(dtype=np.float32)
    st = static.set_index("unit")[stat_cols]
    for j, u in enumerate(units):
        r = slice(j * n, (j + 1) * n)
        X[r, :ns] = shared_block
        for k, c in own_src[u].items():
            X[r, ns + own_names.index(k)] = w[c].to_numpy(dtype=np.float32)
        X[r, ju] = cats.index(u)
        X[r, ju + 1:ju + 1 + len(stat_cols)] = st.loc[u].to_numpy(dtype=np.float32)
        idx = pd.MultiIndex.from_arrays([w["run_no"].to_numpy(), np.full(n, u, dtype=object)])
        X[r, ju + 1 + len(stat_cols):] = ctx.reindex(idx).to_numpy(dtype=np.float32)
        y[r] = w[f"on_{u}"].to_numpy(dtype=np.float32) if f"on_{u}" in w.columns else np.nan
    return run, hour, ucode, X, y, cols, ftypes, cats
"""
submit_src = LONG_MATRIX + "\n\n" + submit_src

pruned = [l.strip() for l in (SRC.parent / "data/processed/pruned_features.csv").read_text()
          .splitlines()[1:] if l.strip()]

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Short notebook 2 — long-format XGBoost (Kaggle public 0.98058)

This notebook reproduces our first Kaggle submission (public leaderboard **0.98058**) from the raw
competition data. It is the end point of the feature-engineering track described in the report
(logs 001–036): one row per (case, hour, unit), a single binary XGBoost model shared by all units,
207 features.

1. **Base table** — join price, inflow, storage and water value onto one row per (case, hour).
2. **Feature stages** — economic spreads and margins, terminal water value (read at start + 7
   days), seasonal percentiles, water budget, merit order, cascade graph, calendar.
3. **Count table** — one row per (case, unit) with multi-horizon storage/inflow/production
   features; a 6-class XGBoost model over hours-on bins is fitted **only to rank** these
   features, and the top 120 (+ up to 30 unit-relative) are attached per unit.
4. **Hourly structure** — block/run features within the week.
5. **Long format + XGBoost** — 600 rounds, depth 8, `min_child_weight=500`, on all 2,922
   labelled cases; predict the 725 test cases.

**Resources.** Full resolution trains on 6.87M rows × ~207 features. On a 4-core machine it
takes about 40 minutes and peaks at **~24 GB of memory** (RAM plus swap), once while XGBoost
quantises the training matrix and again at prediction. A 16 GB machine needs swap; macOS adds it
automatically. `STRIDE = 3` trains on every
third case for a quicker, lighter run.

**Requirements:** the versions the uploaded file was produced with: `numpy==2.2.6 pandas==2.3.2
pyarrow==23.0.1 xgboost==3.4.1 scikit-learn==1.7.2 PyYAML==6.0.3`.

**Reproducibility.** The uploaded file was produced on an Apple Silicon Mac. On a 4-core x86 Linux
machine with the same package versions this notebook selects 143 count-table features instead of
141, so the final model differs slightly: its predictions correlate 0.995 with the uploaded file
and fall on the same side of 0.5 in 98% of cells. The difference is not the thread count (4 and 8
threads select identical features); we attribute it to platform-level floating-point differences in
XGBoost's histogram sums, which reorder near-tied features in the count model's importance ranking.
The last cell compares the result with the uploaded file when `submission.csv` is present."""),
    code('''from __future__ import annotations

import gc
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Folder holding data/kernel, data/extended, prediction_mapping.csv, sample_submission.csv
CANDIDATES = [Path("."), Path(".."), Path("sintef_project"), Path("../sintef_project")]
ROOT = next(p.resolve() for p in CANDIDATES if (p / "data/kernel").is_dir())
_ROOT = ROOT
STRIDE = 1                                    # 1 = full resolution (the submitted model)
OUT_PATH = Path("submission_xgboost.csv")
print("data root:", ROOT)'''),
    md("## 1. Base table: one row per (case, hour)"),
    code("N_HOURS = 168\n\n\n" + main),
    md("""## 2. Feature stages over the (case, hour) table

Built on train and test together so that seasonal and relative features share one scale. The
15 pruned economic features were selected by gain from 299 candidates in log 015."""),
    code("PRUNED_FEATURES = (\n" + "".join(f'    "{p}",\n' for p in pruned) + ")\n\n\n"
         + code_for(["features"])),
    md("""## 3. Count-table features: one row per (case, unit)

Multi-horizon trailing and forward windows of price, inflow and storage; cascade water balance
and implied production per plant; constraint series; unit-relative (`own_`, `upstream_`) views.
The forward windows read the observed storage trajectory, which the volume file provides for the
test period too."""),
    code(code_for(["count_features", "production_features", "unit_relative", "count_model"])),
    md("## 4. Hourly block structure and the long (case, hour, unit) format"),
    code(code_for(["hourly_structure", "longform"])),
    md("""## 5. Pipeline: features → count-model ranking → long format → XGBoost → submission

The count model is fitted only to rank the count-table features; its predictions are not
forwarded (forwarding them cost 0.008, log 035)."""),
    code(submit_src),
    code('''t0 = time.time()
submission = run(stride=STRIDE, out_path=str(OUT_PATH))
print(f"done in {time.time() - t0:.0f}s")'''),
    code('''# Optional: compare with the file we uploaded to Kaggle, if it is next to this notebook.
ref_path = ROOT / "submission.csv"
if ref_path.exists() and STRIDE == 1:
    ref = pd.read_csv(ref_path).set_index("Run No").loc[submission["Run No"]]
    a = submission.drop(columns="Run No").to_numpy().ravel()
    b = ref.to_numpy().ravel()
    print(f"max |difference| to the uploaded submission: {np.abs(a - b).max():.2e}")
    print(f"correlation {np.corrcoef(a, b)[0, 1]:.4f}; "
          f"same side of 0.5 in {((a > 0.5) == (b > 0.5)).mean():.1%} of cells")'''),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({sum(c.source.count(chr(10)) for c in cells if c.cell_type == 'code')} code lines)")
