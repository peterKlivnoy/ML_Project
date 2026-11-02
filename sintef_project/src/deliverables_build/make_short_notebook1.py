"""Generate `Short notebook 1.ipynb`: the LP + LightGBM submission, self-contained.

The course requires all code to live in the notebook, so the definitions the submission actually
uses are lifted from src/shoplite.py, src/shoplite_batch.py, src/physics_features.py and
src/physics_model.py by AST, and rewritten minimally:
  * module prefixes (S., SB., PF.) removed -- everything shares one namespace;
  * options that were only used in experiments (MILP, alternative water-value and river-cost
    conventions) removed, keeping exactly the code path that produced the submission;
  * the solver takes its case inputs as an argument, so worker processes need no data files.
Every replacement is asserted to match exactly once.

    python src/deliverables_build/make_short_notebook1.py
"""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf

SRC = Path(__file__).resolve().parents[1]
OUT = SRC.parent / "deliverables" / "Short notebook 1.ipynb"


def defs(path: Path, names: list[str]) -> dict[str, str]:
    """Source of the named top-level definitions, decorators included."""
    text = path.read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    out = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out[name] = "".join(lines[start - 1:node.end_lineno]).rstrip() + "\n"
    missing = set(names) - set(out)
    assert not missing, f"{path.name}: missing {missing}"
    return out


def sub(text: str, old: str, new: str, count: int = 1) -> str:
    n = text.count(old)
    assert n == count, f"expected {count} match(es), found {n}: {old[:70]!r}"
    return text.replace(old, new)


S = defs(SRC / "shoplite.py", [
    "sanitize", "Unit", "Plant", "Topology", "_f", "load_topology", "level", "turbine_eta",
    "q_range", "unit_pq", "concave_envelope", "head_for_plant", "plant_conversion",
    "main_path", "unit_up_down", "load_raw", "case_inputs", "_Builder", "solve_case"])
B = defs(SRC / "shoplite_batch.py", ["all_cases", "_tables"])
P = defs(SRC / "physics_features.py", ["_roll", "_shift", "_rank", "_run_features", "build"])

# ---- shoplite: keep only the path that produced the submission ---------------------------
builder = S["_Builder"]
builder = sub(builder, "        self.lb, self.ub, self.c, self.integ = [], [], [], []\n",
              "        self.lb, self.ub, self.c = [], [], []\n")
builder = sub(builder, "    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0, integer=False):\n",
              "    def var(self, shape, lb=0.0, ub=np.inf, cost=0.0):\n")
builder = sub(builder, "        self.integ.append(np.full(size, integer))\n", "")
builder = sub(builder, "    def solve(self, mip: bool, time_limit: float, gap: float, threads: int = 1):\n",
              "    def solve(self, threads: int = 1):\n        \"\"\"Solve the LP with HiGHS (dual simplex); returns primal, duals, status.\"\"\"\n")
builder = sub(builder, """        integ = np.concatenate(self.integ)
        if mip:
            lp.integrality_ = [highspy.HighsVarType.kInteger if b else
                               highspy.HighsVarType.kContinuous for b in integ]
""", "")
builder = sub(builder, """        if mip:
            h.setOptionValue("time_limit", float(time_limit))
            h.setOptionValue("mip_rel_gap", float(gap))
""", "")
builder = sub(builder, "        info = h.getInfo()\n", "")
builder = sub(builder, "        y = np.array(sol.row_dual) if (not mip and sol.dual_valid) else None\n"
                       "        return x, y, status, info.objective_function_value\n",
              "        y = np.array(sol.row_dual) if sol.dual_valid else None\n"
              "        return x, y, status\n")

solve = S["solve_case"]
solve = sub(solve, '''def solve_case(start, mip: bool = False, opt: Options | None = None,
               time_limit: float = 30.0, gap: float = 1e-3) -> dict:
    """Build and solve one weekly case. Returns arrays keyed by unit / reservoir."""
    opt = opt or Options()
    topo = load_topology()
    inp = case_inputs(pd.Timestamp(start))
''', '''def solve_case(topo: Topology, inp: dict) -> dict:
    """Build and solve one weekly case. Returns arrays keyed by unit / reservoir."""
''')
solve = sub(solve, "    wvm, ecf = terminal_values(topo, inp[\"wv\"], conv, opt)\n",
            "    # terminal value, EUR/Mm3: at every plant k on the way to the sea, conv_k MWh priced at\n"
            "    # the water value of the reservoir feeding k (measured from the labels, see the report)\n"
            "    wvm = {r: sum(inp[\"wv\"][f] * conv[pl] for pl, f in main_path(topo, r)) for r in topo.res}\n")
solve = sub(solve, "        uu = B.var((T,), 0, 1, price * pe[0], integer=True)\n",
            "        uu = B.var((T,), 0, 1, price * pe[0])\n")
solve = sub(solve, "        st = B.var((T,), 0, 1, -(u.start if opt.start_costs else 0.0))\n"
                   "        sd = B.var((T,), 0, 1, -(u.stop if opt.start_costs else 0.0))\n",
            "        st = B.var((T,), 0, 1, -u.start)\n"
            "        sd = B.var((T,), 0, 1, -u.stop)\n")
solve = sub(solve, """    if opt.sibling_order:
        for chain in SIBLING_ORDER:
            for a, b in zip(chain[:-1], chain[1:]):
                B.rows([(U[a]["u"], 1.0), (U[b]["u"], -1.0)], 0.0, np.inf)
""", """    for chain in SIBLING_ORDER:
        for a, b in zip(chain[:-1], chain[1:]):
            B.rows([(U[a]["u"], 1.0), (U[b]["u"], -1.0)], 0.0, np.inf)
""")
old_losses = solve[solve.index("    if opt.losses:\n"):solve.index("    # --- rivers ---\n")]
body = old_losses.split("\n", 1)[1]
new_losses = "".join(l[4:] if l.startswith("    ") else l for l in body.splitlines(keepends=True))
solve = sub(solve, old_losses, new_losses)
solve = sub(solve, """            # "total": y is the cost per hour at flow x, so the marginal cost is the slope,
            # made non-decreasing so the LP fills segments in order. "marginal": y is itself
            # the cost per (m3/s*h) -- refuted on Kjela, whose outflow crosses r_Hyljelihyl.
            if opt.river_cost == "marginal":
                mc = 0.5 * (y[:-1] + y[1:])
            else:
                mc = np.maximum.accumulate(np.diff(y) / np.diff(x))
""", """            # y is the cost per hour at flow x, so the marginal cost is the slope, made
            # non-decreasing so the LP fills segments in order (measured on Kjela, see report)
            mc = np.maximum.accumulate(np.diff(y) / np.diff(x))
""")
solve = sub(solve, "    x, y, status, obj = B.solve(mip, time_limit, gap)\n\n"
                   "    out = dict(status=status, obj=obj, ecf=ecf, head=head, conv=conv)\n",
            "    x, y, status = B.solve()\n\n    out = dict(status=status)\n")

load_topology = sub(S["load_topology"], "@lru_cache(maxsize=1)\n", "")
load_raw = sub(S["load_raw"], "@lru_cache(maxsize=1)\n", "")

# ---- features ----------------------------------------------------------------------------
build = P["build"]
build = sub(build, "def build(", "def build_features(")
build = sub(build, "    topo = S.load_topology()\n", "    topo = TOPO\n")
build = sub(build, "    price_s, inflow, vol, wv, minvol, minflow = S.load_raw()\n",
            "    price_s, inflow, vol, wv, minvol, minflow = RAW\n")
for a, b_ in [("S.sanitize", "sanitize"), ("S.q_range", "q_range"),
              ("S.unit_up_down", "unit_up_down"), ("S.SIBLING_ORDER", "SIBLING_ORDER")]:
    build = build.replace(a, b_)
build = sub(build, "S.K", "K")
assert "S." not in build.replace("S.T", ""), "unresolved shoplite reference in build"

all_cases = B["all_cases"]
all_cases = sub(all_cases, 'S.ROOT / "data/kernel/Unit_commitment_decisions.csv"',
                'ROOT / "data/kernel/Unit_commitment_decisions.csv"')
all_cases = sub(all_cases, 'S.ROOT / "prediction_mapping.csv"', 'ROOT / "prediction_mapping.csv"')
tables = sub(B["_tables"], "    units = list(S.load_topology().units)\n    T = S.T\n",
             "    units = list(TOPO.units)\n")
tables = sub(tables, "S.sanitize(u)", "sanitize(u)")

# ---- notebook ----------------------------------------------------------------------------
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Short notebook 1 — LP + LightGBM (our best Kaggle submission)

This notebook reproduces our best Kaggle prediction end to end from the raw competition data.

**Idea.** The target is the on/off decision of SINTEF's SHOP optimiser, and every input SHOP used
is in the data, including the SHOP-ready topology file. So instead of learning the input→commitment
map from scratch we

1. **rebuild the weekly optimisation problem SHOP solves** from `Tokke_Vinje_topology.yaml` —
   water balance over 17 reservoirs, rivers and tunnels, turbine/generator efficiency, friction
   losses, start/stop costs, min-flow and min-volume constraints, terminal water value — and solve
   its LP relaxation for every case with HiGHS (~8 s per case on one core on average);
2. turn the solution (commitment, power, flow, local water value from the duals, reservoir
   trajectories) plus a few input features into one row per (case, hour, unit);
3. fit **one LightGBM model** on all 2,922 labelled cases and predict the 725 test cases.

Walk-forward validation (six half-year folds, 2020–2022): **0.9912** micro-AUC, against 0.9619 for
our previous XGBoost model on the same folds. Details and interpretation are in the report.

**Runtime** on a 4-core machine: ~2 h for the 3,647 LPs (cached to `lp_cache/` after the first
run), ~6 min for features and training. Peak memory ~5 GB.

**Requirements:** `numpy pandas pyarrow pyyaml scipy highspy lightgbm joblib`
(developed with highspy 1.15.1, lightgbm 4.7.0, pandas 3.0, Python 3.13)."""),
    code('''from __future__ import annotations

import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import highspy
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Folder holding data/kernel, data/extended, prediction_mapping.csv, sample_submission.csv
# (the extracted `sintef_project` folder). Found automatically next to or above this notebook.
CANDIDATES = [Path("."), Path(".."), Path("sintef_project"), Path("../sintef_project")]
ROOT = next(p.resolve() for p in CANDIDATES if (p / "data/kernel").is_dir())
N_JOBS = -1                       # LP solves run in parallel, one HiGHS thread per process
LP_CACHE = Path("lp_cache")       # solved LPs are stored here and reused on a re-run
OUT_PATH = Path("submission_lp_lightgbm.csv")
print("data root:", ROOT)'''),
    md("""## 1. Topology: the SHOP model of Tokke–Vinje

Parsed from the SHOP-ready YAML: reservoirs (volume→level curves), plants (outlet level, friction
losses), generators (p_min/p_max, turbine efficiency curves at reference heads, generator
efficiency, start/stop costs), rivers (bypass gates, spillways, costed rivers), tunnels, and the
connections between them."""),
    code('''T = 168             # hours per case
K = 0.0036          # Mm3 per (m3/s * hour)
G = 9.81e-3         # MW per (m3/s * m): P = G * H * Q * eta
BIG_PEN = 5_000.0   # EUR per (m3/s*h) of min-flow deficit
# SHOP's tie-break between identical sibling units, measured on the labels (report section 5):
# one Tokke unit on is G1 99% of the time, two are G1+G3 94%; Vinje is always G1, then G1+G2.
SIBLING_ORDER = (("Tokke_G1", "Tokke_G3", "Tokke_G4", "Tokke_G2"),
                 ("Vinje_G1", "Vinje_G2", "Vinje_G3"))


''' + "\n\n".join([S["sanitize"], "@dataclass\n" + S["Unit"].split("@dataclass\n")[-1],
                   "@dataclass\n" + S["Plant"].split("@dataclass\n")[-1],
                   "@dataclass\n" + S["Topology"].split("@dataclass\n")[-1], S["_f"],
                   load_topology, S["level"], S["turbine_eta"], S["q_range"], S["unit_pq"],
                   S["concave_envelope"], S["head_for_plant"], S["plant_conversion"],
                   S["main_path"], S["unit_up_down"]]) + '''

TOPO = load_topology()
print(f"{len(TOPO.res)} reservoirs, {len(TOPO.plants)} plants, {len(TOPO.units)} units, "
      f"{len(TOPO.rivers)} rivers, {len(TOPO.tunnels)} tunnels")'''),
    md("""## 2. Inputs per case

A case starts at 00:00 UTC on its start date and covers 168 hours. Its inputs are the hourly
price and inflows, the reservoir volumes at the start, the water value at start + 7 days (the
terminal water value, as the dataset documentation specifies), and the min-flow / min-volume
series."""),
    code(load_raw + "\n\n" + S["case_inputs"].replace("    price, inflow, vol, wv, minvol, minflow = load_raw()\n",
                                                      "    price, inflow, vol, wv, minvol, minflow = RAW\n")
         + "\n\n" + all_cases + '''

RAW = load_raw()
CASES = all_cases()
print(f"{len(CASES)} cases: {(CASES.starttime.dt.year < 2023).sum()} labelled, "
      f"{(CASES.starttime.dt.year >= 2023).sum()} test")'''),
    md("""## 3. The weekly optimisation problem

One linear programme per case (the LP relaxation of SHOP's unit-commitment problem):

* **objective** — maximise market revenue minus friction losses, start/stop costs, river,
  bypass and spill costs and constraint penalties, plus the value of the water left at the end;
* **units** — on/off variable $u \\in [0,1]$ carrying the minimum block ($q_{\\min}$, $p_{\\min}$),
  plus segments of the concave envelope of the PQ curve at the case's initial head;
* **water** — hourly balance in every reservoir, the Vest-Vassdraget river junction and the
  Songa tunnel junction; reservoir bounds; min volume and min flow as heavily penalised soft
  constraints;
* **terminal value** — every plant's break-even price equals the water value of the reservoir
  feeding it (measured from the labels, report section 5).

The LP's duals give the hourly shadow price of water in every reservoir, from which each unit's
*local water value* (EUR/MWh) and margin (price − local water value) follow."""),
    code(builder + "\n\n" + solve),
    md("""## 4. Solve every case

Cases are independent, so they are solved in parallel. Results are stored in float32 tables:
per (case, hour, unit) the commitment `u`, power `p`, flow `q` and local water value `lwv`; per
(case, hour) the reservoir volumes, water shadow prices and river flows. Delete `lp_cache/` to
force a re-solve."""),
    code(tables + '''

def solve_one(run_no, inp):
    o = solve_case(TOPO, inp)
    assert o["status"] == "Optimal", (run_no, o["status"])
    return _tables(run_no, o)


t0 = time.time()
if (LP_CACHE / "lp_unit.parquet").exists():
    LP_UNIT = pd.read_parquet(LP_CACHE / "lp_unit.parquet")
    LP_CASE = pd.read_parquet(LP_CACHE / "lp_case.parquet")
    print(f"loaded cached LP solutions for {LP_UNIT.run_no.nunique()} cases")
else:
    res = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(solve_one)(r, case_inputs(s)) for r, s in zip(CASES.run_no, CASES.starttime))
    LP_UNIT = pd.concat([a for a, _ in res], ignore_index=True)
    LP_CASE = pd.concat([b for _, b in res], ignore_index=True)
    LP_CACHE.mkdir(exist_ok=True)
    LP_UNIT.to_parquet(LP_CACHE / "lp_unit.parquet", index=False)
    LP_CASE.to_parquet(LP_CACHE / "lp_case.parquet", index=False)
    print(f"solved {len(res)} LPs in {time.time() - t0:.0f}s")'''),
    md("""## 5. Features: one row per (case, hour, unit)

61 features in five groups:

* `lp_` — the optimiser's solution and its temporal context: commitment, power, flow, local water
  value and margin, rolling windows, run lengths, plant-level count of units on, reservoir
  trajectories;
* `ec_` — economics from the inputs: price level/rank/z-score within the week, spread to the
  terminal water value, season, inflow;
* `cn_` — min-flow and min-volume constraint series;
* `st_` — the observed storage trajectory over the week (available for test cases too);
* `s_` — static unit attributes and the unit/plant identity."""),
    code("\n\n".join([P["_roll"], P["_shift"], P["_rank"], P["_run_features"], build]) + '''

labels = pd.read_csv(ROOT / "data/kernel/Unit_commitment_decisions.csv", parse_dates=["starttime"])
t0 = time.time()
META, X, NAMES = build_features(CASES, LP_UNIT, LP_CASE, labels=labels)
print(f"feature matrix {X.shape} ({X.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s")'''),
    md("""## 6. LightGBM on all labelled cases, then predict the test cases

Hyperparameters as validated in walk-forward: 600 rounds, learning rate 0.05, 127 leaves,
`min_data_in_leaf=500` (consecutive hours of one unit-week are nearly identical, so the effective
sample size is far below the row count). `num_threads=4` and the seed are fixed so the run is
reproducible."""),
    code('''PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=4, verbose=-1, seed=0)
ROUNDS = 600
CATEGORICAL = ["s_unit", "s_plant"]

y = META["y"].to_numpy()
train = ~np.isnan(y)
t0 = time.time()
ds = lgb.Dataset(X[train], label=y[train], feature_name=NAMES,
                 categorical_feature=CATEGORICAL, free_raw_data=True)
booster = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
print(f"trained on {train.sum():,} rows in {time.time() - t0:.0f}s")

pred = META[~train].copy()
pred["p"] = booster.predict(X[~train])'''),
    code('''# wide submission: one row per test case, columns in the sample submission's order
units = list(TOPO.units)
sample = pd.read_csv(ROOT / "sample_submission.csv")
wide = pred.pivot_table(index="run_no", columns=["unit", "hour"], values="p")
cols = {}
for c in sample.columns[1:]:
    g = re.fullmatch(r"result_committed_(.+)_t(\\d+)", c)
    cols[c] = wide[(units.index(g.group(1)), int(g.group(2)))].to_numpy()
submission = pd.DataFrame(cols, index=wide.index).rename_axis("Run No").reset_index()

assert list(submission.columns) == list(sample.columns)
assert set(submission["Run No"]) == set(sample["Run No"]) and len(submission) == len(sample)
assert submission.drop(columns="Run No").isna().sum().sum() == 0
submission.to_csv(OUT_PATH, index=False)
print(f"wrote {OUT_PATH} {submission.shape}")'''),
    code('''# Optional: compare with the file we uploaded to Kaggle, if it is next to this notebook.
ref_path = ROOT / "submission_physics.csv"
if ref_path.exists():
    ref = pd.read_csv(ref_path).set_index("Run No").loc[submission["Run No"]]
    diff = np.abs(submission.drop(columns="Run No").to_numpy() - ref.to_numpy())
    print(f"max |difference| to the uploaded submission: {diff.max():.2e}")'''),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT}")
