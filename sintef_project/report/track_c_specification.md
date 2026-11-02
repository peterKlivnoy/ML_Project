# Rebuilding `submission_nn_phys.csv`: a pure neural network with no solver

This is the full specification of the submission `submission_nn_phys.csv`. It scored **0.99013 ROC-AUC on the Kaggle public leaderboard**, with a cross-validation (CV) score of 0.9740. It is written so that a coding agent (Claude Code or similar) can rebuild the pipeline from scratch, given only the competition data.

The pipeline has three parts:

1. **83 base features** per (case, unit, hour), built from prices, water values, volumes, inflow, constraints and calendar.
2. **16 closed-form "physics" features** per (case, unit, hour), built from the generator efficiency curves in the topology YAML. They use only arithmetic, sorting and run-length scans. No optimisation solver is involved.
3. **One sequence neural network (UCNet)**. It is trained on all labelled cases (2015–2022) with 3 seeds, and its sigmoid outputs are averaged. These averaged probabilities are the submission, with no post-processing.

No LightGBM, no linear program, no blending and no smoothing are used.

> **On exactness.** The original run trained on Apple MPS (GPU). GPU training is not bit-reproducible across machines, so expect a score near 0.990, not exactly 0.99013. The features, by contrast, are deterministic, and Appendix B gives statistics to check them against.

---

## 0. Environment and data

**Original environment:** Python 3.13.1, torch 2.14.1, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1, PyYAML 6.0.3. Any recent versions should work.

**Hardware:** The original run used a 16 GB Mac with MPS and took about 40 minutes for 3 seeds × 20 epochs. Use CUDA if available. The original code only checks for `mps`, otherwise `cpu`, so add `cuda`. CPU works but is slow, likely several hours.

**Memory:** The full feature tensor is 3647 × 14 × 168 × 99 float32 ≈ 3.4 GB. The scaled training copy adds another ≈ 2.7 GB. On a 16 GB machine, build the tensor into a memory-mapped `.npy` file (`numpy.lib.format.open_memmap`) and scale it in chunks of 64 cases.

**Files needed.** All come with the competition.

| file | used for |
|---|---|
| `data/kernel/Historical_day_ahead_price_2015_2025.csv` | hourly price (column 0); resample to hourly mean, since it turns 15-minute in late 2025 |
| `data/kernel/Historical_inflow_1958_2025.csv` | hourly inflow per reservoir (18 columns, including `r_Vest_Vassdraget`) |
| `data/kernel/Historical_volume_2015_2024.csv` | daily reservoir volume (Mm³), 17 reservoirs |
| `data/kernel/Synthetic_water_value_2015_2024.csv` | daily water value per reservoir (EUR/MWh), 17 reservoirs |
| `data/kernel/Unit_commitment_decisions.csv` | labels |
| `data/extended/Constraint_min_volume.csv` | hourly min volume (Totak, Byrtevatn, Staavatn) |
| `data/extended/Constraint_min_flow.csv` | hourly min flow (6 columns) |
| `data/extended/Tokke_Vinje_topology.yaml` | reservoirs (`max_vol`, `lrl`, `hrl`), plants (`outlet_line`, `main_loss`, `penstock_loss`), generators (curves, `p_min`, `p_max`, `startcost_const`), `connections` |
| `prediction_mapping.csv` | test cases (`Run No`, `starttime`) |
| `sample_submission.csv` | column order of the output |

**Loading rules:**

- Every hourly or daily CSV has a `date` column. Parse it as a datetime index and drop duplicated index entries, keeping the first.
- The timestamps are UTC (tz-aware).
- A case's start time is naive. Localise it to UTC.

## 1. Problem layout

- **Labels:** `Unit_commitment_decisions.csv` has 2922 rows (cases). Each case starts daily at 00:00 UTC, from 2015-01-01 to 2022-12-31. The columns are `result_committed_{unit}_t{0..167}`.
- **Units:** There are U = 14 units, in this order, which is the order in the label columns:
  `Hogga_G1, Tokke_G1, Tokke_G2, Tokke_G3, Tokke_G4, Lio_G1, Byrte_G1, Vinje_G1, Vinje_G2, Vinje_G3, Haukeli_G1, Songa_G1, Kjela_G1, Vesle Kjela_G1`.
  The plant of a unit is `unit.rsplit("_", 1)[0]`.
- **Label array:** Y has shape (2922, 14, 168), uint8, with a mean of 0.516.
- **Test set:** 725 cases, daily from 2023-01-01 to 2024-12-25 (`Run No` 2923…).
- **Horizon:** H = 168 hours. Hour t of a case is `start + t hours` (UTC).
- **Day k of a case** (k = 0..7) is `start + k days`. It is used for daily series: volume and water value. Day 7 is the end of the horizon.
- **Feature arrays:** Every feature is an array of shape (n_cases, U, H), or something that broadcasts to it.

---

## 2. Base features (83)

The list below is in the order the original used. The NN does not depend on feature order, apart from negligible scaler-sampling effects.

### 2.1 Notation and helpers

**Inputs:**

- `P` (n, H): hourly price. Take `price.reindex(hours).ffill()`.
- `WV` (n, 8, R): water value at day 0..7, taken as `wv.reindex(start + k days)`. R = the 17 water-value columns, in file order.
- `VOL` (n, 8, R): volume at day 0..7, with the same columns as WV.
- `INF` (n, H, 18): hourly inflow, taken as `inflow.reindex(hours)`, with no fill.
- `max_vol[r]`: from the YAML (`model.reservoir.<r>.max_vol`).

**Helper functions:**

- `_roll(a, w, fn)`: a pandas rolling window along the hour axis, with `center=True, min_periods=1`. `fn` is one of mean, min, max or std.
- `_rank(a, axis)`: `a.argsort(axis).argsort(axis) / (len - 1)`. This is the rank scaled to [0, 1].

**Broadcasting conventions:**

- An (n, H) feature is the same for all units. Insert the unit axis.
- An (n, U) feature is the same for all hours.
- An (n,) feature is constant per case.

**Plant mapping, hand-written** (`up` = intake reservoirs, `down` = outlet reservoir, `storage` = the large reservoirs feeding the intake):

```python
PLANTS = {
    "Hogga":       dict(up=["Bandak"],                   down=None,         storage=["Bandak"]),
    "Tokke":       dict(up=["Vinjevatn"],                down="Bandak",     storage=["Vinjevatn", "Totak", "Vaamarvatn"]),
    "Lio":         dict(up=["Byrtevatn"],                down="Bandak",     storage=["Byrtevatn", "Botnedalsvatn"]),
    "Byrte":       dict(up=["Botnedalsvatn"],            down="Byrtevatn",  storage=["Botnedalsvatn"]),
    "Vinje":       dict(up=["Vaamarvatn"],               down="Vinjevatn",  storage=["Vaamarvatn", "Totak", "Venemo"]),
    "Haukeli":     dict(up=["Vatjern"],                  down="Vinjevatn",  storage=["Langeidvatn"]),
    "Songa":       dict(up=["Songavatn", "Bitdalsvatn"], down="Totak",      storage=["Songavatn", "Bitdalsvatn"]),
    "Kjela":       dict(up=["Foersvatn"],                down="Hyljelihyl", storage=["Foersvatn", "Bordalsvatn", "Langesae"]),
    "Vesle Kjela": dict(up=["Kjelavatn"],                down="Foersvatn",  storage=["Kjelavatn", "Staavatn"]),
}
RES_NEXT_PLANT = {"Bandak": "Hogga", "Vinjevatn": "Tokke", "Byrtevatn": "Lio", "Botnedalsvatn": "Byrte",
    "Vaamarvatn": "Vinje", "Totak": "Vinje", "Venemo": "Vinje", "Hyljelihyl": "Vinje", "Vatjern": "Haukeli",
    "Langeidvatn": "Haukeli", "Songavatn": "Songa", "Bitdalsvatn": "Songa", "Foersvatn": "Kjela",
    "Bordalsvatn": "Kjela", "Langesae": "Kjela", "Kjelavatn": "Vesle Kjela", "Staavatn": "Vesle Kjela"}
PLANT_TO_RES = {"Hogga": None, "Tokke": "Bandak", "Lio": "Bandak", "Byrte": "Byrtevatn", "Vinje": "Vinjevatn",
    "Haukeli": "Vinjevatn", "Songa": "Totak", "Kjela": "Venemo", "Vesle Kjela": "Foersvatn"}
```

**Head and global energy:**

- `head[p] = (lrl + hrl)/2 of reservoir up[0] − plant.outlet_line`.
- `glob[r] = E(r)`, where `E(None) = 0` and `E(r) = head[RES_NEXT_PLANT[r]] + E(PLANT_TO_RES[RES_NEXT_PLANT[r]])`.
- Expected values: head = {Hogga 11.23, Tokke 391.75, Lio 342.65, Byrte 281.2, Vinje 220.15, Haukeli 266.5, Songa 274.2, Kjela 156.45, Vesle Kjela 56.7}.

**Aggregations over a plant's reservoirs:**

- `up_mean(arr, p)`: the mean over `up` reservoirs.
- `stor_sum(arr, p)`: the sum over `storage` reservoirs.

### 2.2 The feature list

**Economic core (9 features):**

- `wv_end_up = up_mean(WV[:,7])` and `wv_start_up = up_mean(WV[:,0])`, each per unit's plant, shape (n, U).
- `margin = P − wv_end_up`, shape (n, U, H).

| # | name | definition |
|---|---|---|
| 0 | margin_end | `margin` |
| 1 | margin_start | `P − wv_start_up` |
| 2 | margin_frac_pos | mean over hours of `margin > 0` |
| 3 | margin_week_mean | mean over hours of `margin` |
| 4 | margin_rel | `P / (abs(wv_end_up) + 1)` |
| 5 | margin_day_mean | daily mean of `margin` (reshape to 7×24), repeated 24× |
| 6 | margin_roll12 | `_roll(margin, 12, mean)` |
| 7 | margin_roll24_min | `_roll(margin, 24, min)` |
| 8 | margin_roll24_max | `_roll(margin, 24, max)` |

**price (18 features).** All are (n, H). `p_mean` and `p_std` are taken over the week, and `p_std` gets `+1e-3`.

| # | name | definition |
|---|---|---|
| 9 | price | P |
| 10 | price_z | (P − p_mean)/p_std |
| 11 | price_rank | `_rank(P)` over the week |
| 12 | price_day_rank | `_rank` within each day (7×24) |
| 13 | price_day_pos | (P − daymin)/(daymax − daymin + 1e-3) |
| 14 | price_day_spread | daymax − daymin |
| 15–16 | price_roll6, price_roll24 | `_roll(P, w)` mean |
| 17 | price_roll48_rank | (P − roll48 mean)/(roll48 std + 1e-3) |
| 18–23 | price_lag{1,3,6}, price_lead{1,3,6} | lag: `[P[0] repeated lag times, P[:-lag]] − P`; lead: `[P[lag:], P[-1] repeated lag times] − P` (order: lag1, lead1, lag3, lead3, lag6, lead6) |
| 24 | price_minus_rem_mean | P − mean of P from this hour to hour 167 |
| 25 | p_week_mean | p_mean |
| 26 | p_week_std | p_std |

**wv (9 features):**

| # | name | definition |
|---|---|---|
| 27 | wv_end_up | (n, U) |
| 28 | wv_start_up | (n, U) |
| 29 | wv_change_up | end − start |
| 30 | wv_end_over_pmean | wv_end_up / (mean_week(P) + 1) |
| 31 | wv_end_down | WV[:,7, down] (0 if none) |
| 32 | margin_global | P − be, with be = (up_mean(WV[:,7]) · glob[up[0]] − WV[:,7,down] · glob[down] (0 if none)) / head[p] |
| 33 | wv_path_daily_slope | std over the 7 day-to-day diffs of up_mean(WV[:, 0..7]) |
| 34 | margin_daily_wv | P − wv_path[day d+1] repeated per hour of day d (d = 0..6, i.e. uses WV days 1..7) |
| 35 | wv_sys_mean_end | mean over all 17 reservoirs of WV[:,7], per case |

**state (9 features).** `fill = VOL/max_vol`, taken at day 0.

| # | name | definition |
|---|---|---|
| 36 | fill_up | up_mean(fill) |
| 37 | fill_storage | stor_sum(VOL)/stor_sum(max_vol) |
| 38 | fill_down | fill of down reservoir (0 if none) |
| 39 | fill_sys | sum(VOL)/sum(max_vol) |
| 40–44 | fill_Totak, fill_Songavatn, fill_Bandak, fill_Vaamarvatn, fill_Vinjevatn | per case |

**inflow (5 features).** `wk = INF.sum(hours) · 3600/1e6` is the weekly inflow in Mm³ per inflow column. A plant's inflow is the sum of `wk` over its `storage` reservoirs that have an inflow column.

| # | name | definition |
|---|---|---|
| 45 | inflow_storage_wk | plant storage inflow |
| 46 | inflow_up_wk_rel | that / (stor_sum(max_vol) + 1e-3) |
| 47 | inflow_sys_wk | sum of wk over all 18 columns |
| 48 | inflow_sys_anom | inflow_sys_wk / clim[doy − 1], where doy = case start day-of-year clipped to [1, 365] |
| 49 | inflow_sys_h | hourly sum over all 18 columns, (n, H) |

The climatology `clim` uses inflow before 2015-01-01 only:

1. Sum the 18 columns per hour.
2. Take daily sums × 3600/1e6.
3. Compute the 7-day forward sum `rolling(7).sum().shift(-6)`.
4. Take the mean by day-of-year, reindex to 1..365 and forward-fill.

**constraints (9 features):**

| # | name | definition |
|---|---|---|
| 50–55 | minflow_{b_Kjelavatn, b_Hyljelihyl, b_Vinjevatn, b_Byrtevatn, b_Bandak, r_Vest_Vassdraget} | hourly min flow, `reindex(hours).fillna(0)` |
| 56–58 | minvol_gap_{Totak, Byrtevatn, Staavatn} | VOL[day 0, r] − hourly min volume (fillna 0) |

**future_vol (8 features).** These use the realised volume path, which is available for test cases too.

Intermediate quantities:

- `dV = diff(VOL over days)`, shape (n, 7, R).
- `INF_day`: daily inflow in Mm³, mapped onto the R reservoirs (0 where a reservoir has no inflow column).
- `rel = INF_day − dV`, an outflow proxy.
- `per_plant(arr, which)`: the sum over the plant's `which` reservoirs. Daily values are repeated 24×.

| # | name | definition |
|---|---|---|
| 59 | fv_dV_storage_wk | Σ_days per_plant(dV, storage) |
| 60 | fv_dV_storage_day | per_plant(dV, storage), daily |
| 61 | fv_release_up_day | per_plant(rel, up), daily |
| 62 | fv_release_up_wk | weekly sum of above |
| 63 | fv_release_storage_day | per_plant(rel, storage), daily |
| 64 | fv_release_storage_wk | weekly sum |
| 65 | fv_dV_down_wk | Σ_days dV[down] (0 if none) |
| 66 | fv_dV_sys_wk | Σ over days and all reservoirs of dV |

**unit (8 features).** All are static.

| # | name | definition |
|---|---|---|
| 67 | unit_id | 0..13 (float) |
| 68 | p_min | generator `p_min` |
| 69 | p_max | generator `p_max` |
| 70 | startcost | generator `startcost_const` (0 if absent) |
| 71 | unit_idx_in_plant | the integer after "G" in the unit name |
| 72 | n_units_plant | number of units in the plant |
| 73 | head | head[plant] |
| 74 | t | hour index 0..167 |

**calendar (5 features).** Hour-based features use Europe/Oslo local time.

| # | name | definition |
|---|---|---|
| 75 | hod | local hour of day |
| 76 | dow | local day of week (Mon = 0) |
| 77 | doy_sin | sin(2π · doy/365.25), where doy is the day-of-year of the case start |
| 78 | doy_cos | cos(2π · doy/365.25) |
| 79 | holiday | 1 if the local date is a Norwegian holiday |

The holiday list covers 2015–2025:

- **Easter-based days:** Easter −3, −2, 0, +1, +39, +49, +50 days.
- **Fixed dates:** 01-01, 05-01, 05-17, 12-24, 12-25, 12-26, 12-31.
- **Easter Sundays:** 2015-04-05, 2016-03-27, 2017-04-16, 2018-04-01, 2019-04-21, 2020-04-12, 2021-04-04, 2022-04-17, 2023-04-09, 2024-03-31, 2025-04-20.

**plant_ctx (3 features).** These are cross-unit features built from `margin_end`.

| # | name | definition |
|---|---|---|
| 80 | margin_sys_mean | mean over the 14 units, per hour |
| 81 | margin_minus_sys | margin − that |
| 82 | margin_rank_across_units | `_rank(margin, axis=unit)` |

---

## 3. Physics features (16)

Pre-compute these once for all 3647 train and test case starts. They take seconds. Everything below is per case.

**Inputs:**

- `P` (n, H): price, as before.
- `WV7[r]`: water value at day 7.
- `V0[r]`: volume at day 0.
- `inflow[r]` (n, H): hourly inflow, `reindex(hours).ffill().fillna(0)`.
- `min_vol[r][:, 0]`: min volume at the first hour of the case. Only Totak, Byrtevatn and Staavatn have one.
- `Q2V = 0.0036`: Mm³ per (m³/s · h).

### 3.1 Static generator table

These numbers come from the YAML: the turbine efficiency curve at the plant head, minus losses, times generator efficiency, reduced to its concave envelope. Hard-code them. The derivation is in Appendix A, in case it is needed.

| unit | plant | q0 (m³/s) | p0 (MW) | dq (2 segments) | slope (MW per m³/s) | p_min | p_max | start | conv (best MW per m³/s) |
|---|---|---|---|---|---|---|---|---|---|
| Hogga_G1 | Hogga | 33.72 | 2.80783564998715 | [109.59, 25.29] | [0.10149342429890193, 0.09165437534151188] | 3 | 17 | 94 | 0.09720535914384072 |
| Tokke_G1..G4 | Tokke | 12.8 | 36.49549394378961 | [11.2, 8.0] | [4.0026121528933984, 3.313129037897717] | 36.9 | 107.5 | 214 | 3.3885312523414863 |
| Lio_G1 | Lio | 5.6 | 14.141421253321798 | [4.2, 4.2] | [3.5045463937930403, 2.6763506387522713] | 14.8 | 43 | 163 | 2.9449506231890377 |
| Byrte_G1 | Byrte | 3.2 | 6.5446661254834035 | [2.4, 2.4] | [2.835237067711068, 2.1612044496302003] | 6.9 | 20 | 135 | 2.3837919799982084 |
| Vinje_G1..G3 | Vinje | 22.0 | 35.26560630253504 | [19.25, 13.75] | [2.2322633833435157, 1.8085301543845225] | 34.4 | 100 | 198 | 1.8966467013793387 |
| Haukeli_G1 | Haukeli | 0.32 | 0.2802197642385298 | [1.15, 0.63] | [2.5403253807377744, 1.9733216496339865] | 0.3 | 4.9 | 143 | 2.1779550694469183 |
| Songa_G1 | Songa | 20.8 | 41.55567700838549 | [18.2, 13.0] | [2.7921037972127514, 2.2843096196145445] | 41.2 | 120 | 222 | 2.3685119517348094 |
| Kjela_G1 | Kjela | 16.0 | 18.1412658580685 | [14.0, 10.0] | [1.5546837627368437, 1.2368336508207853] | 20.6 | 60 | 171 | 1.3302279512128103 |
| Vesle Kjela_G1 | Vesle Kjela | 8.0 | 3.366750917186088 | [6.0, 6.0] | [0.5867579930451935, 0.44310961864109927] | 2.9 | 8.5 | 92 | 0.4919499196755178 |

**Reservoir maps.** Note that `feed` differs from `PLANTS[...]["storage"]` in section 2.

- `intake_res[p]` (the reservoirs the plant draws from): the same as `PLANTS[p]["up"]`.
- `feed[p]` (every reservoir whose water is turbined by p first):
  - Hogga: [Bandak]
  - Tokke: [Vinjevatn]
  - Lio: [Byrtevatn]
  - Byrte: [Botnedalsvatn]
  - Vesle Kjela: [Kjelavatn, Staavatn]
  - Vinje: [Venemo, Hyljelihyl, Totak, Vaamarvatn]
  - Kjela: [Langesae, Bordalsvatn, Foersvatn]
  - Songa: [Songavatn, Bitdalsvatn]
  - Haukeli: [Langeidvatn, Vatjern]

**`max_vol` (Mm³):** Bandak 174.2, Vinjevatn 9.36, Byrtevatn 76.71, Botnedalsvatn 54.17, Kjelavatn 84.5, Staavatn 48, Venemo 23, Langesae 78, Hyljelihyl 0.26, Bordalsvatn 184.2, Foersvatn 122.3, Songavatn 640, Bitdalsvatn 110, Totak 257.8, Vaamarvatn 26.2, Langeidvatn 32.1, Vatjern 0.3.

### 3.2 Computation (reference code)

`units` is the list of 14 units, `gens[k]` is the table row of unit k, and `plants[k]` is its plant.

```python
n, U = len(starts), 14
P = P[:, None, :]                                                   # (n, 1, H)

# ---- unit: break-even prices from the generator curve ----
conv_best = {p: max(g["conv"] for g in gens if g["plant"] == p) for p in plants}
wv = np.stack([np.mean([WV7[r] for r in intake_res[p]], 0) for p in plants], 1)   # (n, U)
q_full = np.array([g["q0"] + sum(g["dq"]) for g in gens])
p_full = np.array([g["p0"] + np.dot(g["slope"], g["dq"]) for g in gens])
rel = lambda mw_per_q: np.array([conv_best[p] for p in plants]) / mw_per_q   # >= 1
be_min  = wv * rel(np.array([g["p0"] / g["q0"] for g in gens]))
be_full = wv * rel(p_full / q_full)
be_last = wv * rel(np.array([g["slope"][-1] for g in gens]))
F["phys_be_rel_min"] = broadcast(rel(p0/q0)[None, :, None])          # (n,U,H)
F["phys_m_min"]  = P - be_min[..., None]
F["phys_m_full"] = P - be_full[..., None]
F["phys_m_last"] = P - be_last[..., None]
pmax, pmin, start = (np.array([g[k] for g in gens])[None, :, None] for k in ("p_max", "p_min", "start"))
F["phys_start_per_mw"] = broadcast(start / pmax)
m_full = F["phys_m_full"]

# ---- budget: water over the week per plant, in hours of full plant flow ----
Qp = {p: sum(q_full[k] for k in range(U) if plants[k] == p) * Q2V for p in plants}   # Mm3/h at full plant flow
infl = lambda rs: sum(inflow[r].sum(1) for r in rs if r in inflow) * Q2V           # Mm3 over the week
minv = lambda rs: sum(min_vol[r][:, 0] for r in rs if r in min_vol)                # 0 if none
afford, forced = np.zeros((n, U)), np.zeros((n, U))
for p in unique(plants):
    ks = [k for k in range(U) if plants[k] == p]
    rs, ins = feed[p], intake_res[p]
    afford[:, ks] = (np.maximum(sum(V0[r] for r in rs) - minv(rs), 0) + infl(rs))[:, None] / Qp[p]
    forced[:, ks] = np.maximum(sum(V0[r] for r in ins) + infl(ins) - sum(max_vol[r] for r in ins), 0)[:, None] / Qp[p]
afford, forced = np.clip(afford, 0, 10 * H), np.clip(forced, 0, H)
npos = (m_full > 0).sum(-1)                                                  # (n, U)
F["phys_afford_h"]   = broadcast(np.minimum(afford, H)[..., None] / H)
F["phys_afford_log"] = broadcast(np.log1p(afford)[..., None])
F["phys_forced_h"]   = broadcast(forced[..., None] / H)
F["phys_npos_h"]     = broadcast(npos[..., None] / H)

# ---- greedy: the plant runs in its k best-margin hours ----
k_on = np.clip(npos, forced, np.minimum(afford, H))      # NB np.clip with array bounds: upper bound wins if lo > hi
rank = (-m_full).argsort(-1).argsort(-1)                 # 0 = best hour
F["phys_greedy_k"]    = broadcast(k_on[..., None] / H)
F["phys_greedy_on"]   = (rank < k_on[..., None]).astype(np.float32)
F["phys_greedy_dist"] = (k_on[..., None] - rank) / H
F["phys_forced_on"]   = (rank < forced[..., None]).astype(np.float32)

# ---- start: block profit vs start cost ----
def _runs(mask, x):
    """For each position of a boolean (m, T) mask: length and sum of x over the run of equal mask values it lies in."""
    m, T = mask.shape
    fl, fs, bl, bs = (np.zeros((m, T)) for _ in range(4))
    for t in range(T):
        same = mask[:, t] == mask[:, t - 1] if t else np.zeros(m, bool)
        fl[:, t] = np.where(same, fl[:, t - 1] + 1, 1) if t else 1
        fs[:, t] = np.where(same, fs[:, t - 1] + x[:, t], x[:, t]) if t else x[:, t]
    for t in range(T - 1, -1, -1):
        same = mask[:, t] == mask[:, t + 1] if t < T - 1 else np.zeros(m, bool)
        bl[:, t] = np.where(same, bl[:, t + 1] + 1, 1) if t < T - 1 else 1
        bs[:, t] = np.where(same, bs[:, t + 1] + x[:, t], x[:, t]) if t < T - 1 else x[:, t]
    return fl + bl - 1, fs + bs - x

pos = (m_full > 0).reshape(n * U, H)
L, S   = _runs(pos, (m_full * pmax).reshape(n * U, H))                          # EUR over the run at full load
Lg, Sg = _runs(pos, (np.minimum(F["phys_m_min"], 0) * pmin).reshape(n * U, H))   # loss of idling at p_min
st = np.broadcast_to(start, (n, U, H)).reshape(n * U, H)
F["phys_block_len"]    = np.where(pos, L, -Lg).reshape(n, U, H) / 24
F["phys_block_profit"] = np.where(pos, S - st, 0).reshape(n, U, H)
F["phys_stay_on_gain"] = np.where(pos, 0, st + Sg).reshape(n, U, H)            # >0: cheaper to idle than restart
# cast all to float32; assert all finite
```

**Order of the physics features** (appended after the 83 base features):

```
phys_be_rel_min, phys_m_min, phys_m_full, phys_m_last, phys_start_per_mw,
phys_afford_h, phys_afford_log, phys_forced_h, phys_npos_h,
phys_greedy_k, phys_greedy_on, phys_greedy_dist, phys_forced_on,
phys_block_len, phys_block_profit, phys_stay_on_gain
```

The final input tensor is `X4` with shape (n_cases, 14, 168, 99). Build it for train (2922) and test (725) with identical code.

---

## 4. The model: UCNet

Each case is one sample, with input (U = 14, T = 168, F = 99) and target (14, 168). The network models three things:

- a temporal convolution and a bidirectional GRU across the 168 hours of each unit (persistence of on/off states);
- self-attention across the 14 units at each hour (units share system information);
- a learned unit embedding.

```python
class UCNet(nn.Module):
    def __init__(self, n_feat, n_units, d=96):
        super().__init__()
        self.unit_emb = nn.Embedding(n_units, 16)
        self.inp = nn.Sequential(nn.Linear(n_feat + 16, d), nn.GELU(), nn.Linear(d, d))
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, 5, padding=2), nn.GELU(), nn.Conv1d(d, d, 5, padding=4, dilation=2), nn.GELU())
        self.gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.unit_attn = nn.MultiheadAttention(d, 4, batch_first=True, dropout=0.1)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1), nn.Linear(d, 1))

    def forward(self, x):  # x (B, U, T, F)
        B, U, T, _ = x.shape
        e = self.unit_emb(torch.arange(U, device=x.device))[None, :, None, :].expand(B, U, T, -1)
        h = self.inp(torch.cat([x, e], -1)).reshape(B * U, T, -1)          # (BU, T, d)
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)                 # residual temporal conv
        h = self.norm1(h + self.gru(h)[0])                                   # residual biGRU
        hu = h.reshape(B, U, T, -1).permute(0, 2, 1, 3).reshape(B * T, U, -1)   # attention across units, per hour
        hu = self.norm2(hu + self.unit_attn(hu, hu, hu, need_weights=False)[0])
        h = hu.reshape(B, T, U, -1).permute(0, 2, 1, 3)
        return self.head(h).squeeze(-1)                                     # logits (B, U, T)
```

The hidden width is d = 96. The unit embedding has dimension 16 and attention uses 4 heads. The model outputs logits.

## 5. Input scaling

This is a robust per-feature scaler. It is fitted on training cases only.

```python
class Scaler:
    def fit(self, X):                     # X = X4[train_idx[::3]]  (every 3rd training case)
        flat = X.reshape(-1, X.shape[-1])
        idx = np.random.default_rng(0).choice(len(flat), min(len(flat), 400_000), replace=False)
        s = np.asarray(flat[np.sort(idx)], dtype=np.float64)
        self.med = np.median(s, 0)
        q = np.percentile(s, [10, 90], axis=0)
        self.scale = np.where(q[1] - q[0] > 1e-6, q[1] - q[0], s.std(0) + 1e-6)
        return self
    def __call__(self, X):                # apply in chunks of 64 cases to save memory
        return np.clip((X - self.med) / self.scale, -8, 8).astype(np.float32)
```

## 6. Training (final submission)

**Training set:** all 2922 labelled cases (2015-01-01 … 2022-12-31). There is no validation set, no early stopping and no gap. The **last-epoch** weights are used.

| hyperparameter | value |
|---|---|
| loss | `BCEWithLogitsLoss` (mean over all U·T outputs, unweighted) |
| optimizer | AdamW, lr 2e-3, weight_decay 1e-2 (other args default) |
| schedule | OneCycleLR, max_lr 2e-3, `pct_start=0.15`, total_steps = epochs × ceil(2922/32), stepped every batch |
| epochs | 20 |
| batch size | 32 cases (= 32 × 14 × 168 outputs) |
| shuffling | `torch.randperm` each epoch |
| gradient clipping | `clip_grad_norm_(params, 1.0)` |
| dropout | 0.1 (GRU between layers, attention, head) |
| EMA / SWA | none |
| seeds | 0, 1, 2. For each: `torch.manual_seed(s); np.random.seed(s)`, fresh model and optimizer, scaler identical (rng 0) |
| prediction | `model.eval()`, `sigmoid(logits)`, test cases in batches of 64 |
| ensembling | **arithmetic mean of the 3 seeds' probabilities** |

**Training loop**, per seed:

```python
torch.manual_seed(seed); np.random.seed(seed)
sc = Scaler().fit(X4[tr[::3]])
Xtr, Ytr, Xte = torch.from_numpy(sc(X4[tr])), torch.from_numpy(Y.astype(np.float32)), torch.from_numpy(sc(X4[te]))
model = UCNet(99, 14, d=96).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=20 * ceil(len(Xtr) / 32), pct_start=0.15)
for ep in range(20):
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 32):
        b = perm[i:i + 32]
        loss = BCEWithLogitsLoss()(model(Xtr[b].to(dev)), Ytr[b].to(dev))
        opt.zero_grad(); loss.backward(); clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
```

## 7. Writing the submission

1. Take P = the mean over seeds, shape (725, 14, 168). Do not apply clipping, calibration or smoothing.
2. Reshape it to (725, 2352). The column order is unit-major, then hour, so that `result_committed_{unit}_t{h}` follows the unit order of section 1 and h = 0..167.
3. Insert `Run No` from `prediction_mapping.csv` as the first column.
4. Assert that the columns equal those of `sample_submission.csv`.
5. Assert that the shape is (725, 2353) and that all values are finite and in [0, 1].
6. Write the CSV with `index=False`.

For reference, the original file's mean predicted probability is **0.4705**.

## 8. Validation protocol (optional, but use it to check the build)

Use expanding-window CV by year:

- For validation year Y, train on cases whose start is before (Y-01-01 − 7 days) and validate on cases starting in year Y. The 7-day gap exists because each case's horizon overlaps the next week.
- Use the same training recipe with one seed. Fit the scaler on that fold's training cases (every 3rd).
- The metric is micro ROC-AUC over all (case, unit, hour) of the fold, computed with `roc_auc_score(Y[va].ravel(), p.ravel())`.

Reference results for 1 seed and 20 epochs on MPS:

| fold | AUC | val AUC by epoch (4 / 8 / 12 / 16 / 20) |
|---|---|---|
| 2019 | 0.9837 | 0.9783 / 0.9790 / 0.9817 / 0.9833 / 0.9837 |
| 2020 | 0.9685 | 0.9643 / 0.9649 / 0.9641 / 0.9696 / 0.9685 |
| 2021 | 0.9658 | 0.9637 / 0.9648 / 0.9626 / 0.9670 / 0.9658 |
| 2022 | 0.9779 | 0.9748 / 0.9710 / 0.9772 / 0.9781 / 0.9779 |
| **mean** | **0.9740** | |

The training loss at epoch 20 is about 0.10. An epoch took 25–40 s on MPS for a fold. If the CV mean lands near 0.973–0.975, the build matches. The public leaderboard turned out much kinder than CV for this model (0.990).

**For comparison:**

- The same NN without the 16 physics features had a CV of 0.9697.
- LightGBM on the same 99 features had a CV of 0.9722.

So the physics features are worth about +0.004 CV for the NN, mostly on the 2021 fold (0.9514 → 0.9658).

## 9. Pitfalls

- **Time zones.** Prices, inflows and constraints are reindexed on UTC hourly timestamps. Only the calendar features use Europe/Oslo.
- **Daily series.** Water value and volume are reindexed exactly at `start + k days` (00:00 UTC). Day 7 is the terminal value. Do not interpolate.
- **Inflow fill.** The base features use inflow with no fill. The physics features use ffill then fillna(0). There are no gaps in the case range, so this does not matter in practice.
- **NaNs.** The original feature matrices had no NaNs, for both train and test. If you get NaNs, a reindex is misaligned. Fix the alignment rather than imputing.
- **Unit order** must be identical everywhere: in the labels, the embedding index, the `unit_id` feature and the submission columns.
- **Test labels.** The test cases need no labels (the original padded with zeros). The scaler never sees test data.
- **Speed.** MPS and CUDA runs are not bit-identical to CPU runs. Use `--device cpu` for repeatable debugging and the GPU for the real run.

---

## Appendix A: deriving the generator table (only if not hard-coding)

Per generator g in plant p:

1. **Head.** `intake_res(p)` = the reservoirs connected upstream of the plant in `connections`. When the upstream node is a tunnel, take the tunnel's source. `head = mean over intake reservoirs of (lrl + hrl)/2 − plant.outlet_line`.
2. **Efficiency curve.** From `turb_eff_curves`, take the curve whose `ref` is closest to `head`. That gives flow `q` and efficiency `eta_t = y/100`.
3. **Loss.** `loss = float(plant.main_loss[0]) + plant.penstock_loss[penstock − 1]`, where `penstock` defaults to 1. The effective head is `h_eff = max(head − loss·q², 1)`.
4. **Power.** `p_mw = 9.81e-3 · h_eff · q · eta_t`. Then multiply by the generator efficiency `np.interp(p_mw, gen_eff_curve.x, gen_eff_curve.y/100)`.
5. **Envelope.** Take the upper concave hull of the points (q, p_mw). Keep only 3 points: the first, the best-efficiency point (argmax p/q on the hull) and the last.
   - `q0, p0` = the first point.
   - `dq` = the diffs in q.
   - `slope` = the diffs in p / the diffs in q.
   - `conv = max(p_mw/q)` over the raw (pre-hull) points.
   - `start = startcost_const`.

## Appendix B: feature checksums

The table gives the mean and standard deviation of each of the 99 features on every 7th training case (cases 0, 7, 14, … of 2922, all units and hours), and the mean on every 3rd test case. Values are the raw, unscaled features. Small differences in rolling or quantile features are fine. Differences in the order of a whole feature mean a bug.

Spot check for training case 0 (2015-01-01), unit Tokke_G1, hour 10: price 27.13, wv_end_up 19.65, margin_end 7.48, fill_up 0.71282, inflow_sys_anom 1.79670, hod 11, phys_m_full 7.37007, phys_afford_h 0.141246, phys_greedy_k 0.141246, phys_block_profit 166309.14.

| # | feature | train mean | train std | test mean |
|---|---|---|---|---|
| 0 | margin_end | -7.5677 | 71.361 | -7.6186 |
| 1 | margin_start | -7.4112 | 68.075 | -8.1323 |
| 2 | margin_frac_pos | 0.54917 | 0.45579 | 0.45962 |
| 3 | margin_week_mean | -7.5677 | 67.71 | -7.6186 |
| 4 | margin_rel | 5.6035 | 19.427 | 6.7675 |
| 5 | margin_day_mean | -7.5677 | 69.492 | -7.6186 |
| 6 | margin_roll12 | -7.5171 | 69.972 | -7.558 |
| 7 | margin_roll24_min | -18.508 | 82.418 | -30.815 |
| 8 | margin_roll24_max | 4.6104 | 66.199 | 18.336 |
| 9 | price | 56.568 | 77.846 | 64.8 |
| 10 | price_z | 2.1505e-11 | 0.9994 | -7.6186e-11 |
| 11 | price_rank | 0.5 | 0.2904 | 0.5 |
| 12 | price_day_rank | 0.5 | 0.30096 | 0.5 |
| 13 | price_day_pos | 0.54234 | 0.32709 | 0.51453 |
| 14 | price_day_spread | 23.13 | 49.156 | 48.687 |
| 15 | price_roll6 | 56.566 | 77.146 | 64.794 |
| 16 | price_roll24 | 56.629 | 76.153 | 64.784 |
| 17 | price_roll48_rank | -0.00071936 | 0.92527 | -0.00087299 |
| 18 | price_lag1 | -0.0094899 | 9.1718 | -0.01126 |
| 19 | price_lead1 | 0.0094899 | 9.1718 | 0.01126 |
| 20 | price_lag3 | -0.051895 | 18.877 | -0.081496 |
| 21 | price_lead3 | 0.037883 | 18.887 | 0.054689 |
| 22 | price_lag6 | -0.20829 | 25.421 | -0.34947 |
| 23 | price_lead6 | -0.020764 | 25.498 | 0.026202 |
| 24 | price_minus_rem_mean | -1.1634 | 22.315 | -0.32917 |
| 25 | p_week_mean | 56.568 | 74.513 | 64.8 |
| 26 | p_week_std | 10.511 | 19.934 | 22.011 |
| 27 | wv_end_up | 64.136 | 122.77 | 72.419 |
| 28 | wv_start_up | 63.979 | 122.74 | 72.933 |
| 29 | wv_change_up | 0.15649 | 54.031 | -0.51372 |
| 30 | wv_end_over_pmean | 0.91121 | 0.66108 | 1.0814 |
| 31 | wv_end_down | 57.51 | 107.19 | 64.544 |
| 32 | margin_global | -17.668 | 293.72 | -5.1378 |
| 33 | wv_path_daily_slope | 9.0746 | 19.537 | 11.739 |
| 34 | margin_daily_wv | -8.3408 | 67.97 | -7.491 |
| 35 | wv_sys_mean_end | 66.504 | 117.74 | 74.223 |
| 36 | fill_up | 0.63247 | 0.24962 | 0.60627 |
| 37 | fill_storage | 0.66282 | 0.256 | 0.60709 |
| 38 | fill_down | 0.65294 | 0.24044 | 0.63025 |
| 39 | fill_sys | 0.66522 | 0.19626 | 0.67005 |
| 40 | fill_Totak | 0.6766 | 0.26169 | 0.56108 |
| 41 | fill_Songavatn | 0.64724 | 0.22474 | 0.72614 |
| 42 | fill_Bandak | 0.76042 | 0.051363 | 0.76423 |
| 43 | fill_Vaamarvatn | 0.55874 | 0.32305 | 0.46349 |
| 44 | fill_Vinjevatn | 0.71114 | 0.12182 | 0.67957 |
| 45 | inflow_storage_wk | 10.343 | 14.983 | 12.255 |
| 46 | inflow_up_wk_rel | 0.045514 | 0.066248 | 0.052683 |
| 47 | inflow_sys_wk | 75.16 | 80.46 | 84.903 |
| 48 | inflow_sys_anom | 1.2927 | 0.92433 | 1.3404 |
| 49 | inflow_sys_h | 124.27 | 149.13 | 140.38 |
| 50 | minflow_b_Kjelavatn | 0.46913 | 0.21153 | 0.47022 |
| 51 | minflow_b_Hyljelihyl | 1.0639 | 0.70509 | 1.0676 |
| 52 | minflow_b_Vinjevatn | 1.4183 | 0.49328 | 1.4215 |
| 53 | minflow_b_Byrtevatn | 0.067806 | 0.094676 | 0.068241 |
| 54 | minflow_b_Bandak | 4.4183 | 0.49328 | 4.4215 |
| 55 | minflow_r_Vest_Vassdraget | 7.272 | 3.2875 | 7.2916 |
| 56 | minvol_gap_Totak | 106.36 | 81.348 | 76.144 |
| 57 | minvol_gap_Byrtevatn | 46.883 | 24.217 | 44.305 |
| 58 | minvol_gap_Staavatn | 15.684 | 18.831 | 13.865 |
| 59 | fv_dV_storage_wk | -0.034372 | 14.788 | 0.35894 |
| 60 | fv_dV_storage_day | -0.0049103 | 2.5173 | 0.051277 |
| 61 | fv_release_up_day | 0.7918 | 1.692 | 0.91963 |
| 62 | fv_release_up_wk | 5.5426 | 9.8234 | 6.4374 |
| 63 | fv_release_storage_day | 1.4825 | 2.4008 | 1.6994 |
| 64 | fv_release_storage_wk | 10.377 | 13.136 | 11.896 |
| 65 | fv_dV_down_wk | -0.011146 | 7.217 | 0.040248 |
| 66 | fv_dV_sys_wk | -0.25822 | 65.952 | 1.5392 |
| 67 | unit_id | 6.5 | 4.0311 | 6.5 |
| 68 | p_min | 24.321 | 14.931 | 24.321 |
| 69 | p_max | 71.671 | 42.21 | 71.671 |
| 70 | startcost | 176.43 | 43.165 | 176.43 |
| 71 | unit_idx_in_plant | 1.6429 | 0.97153 | 1.6429 |
| 72 | n_units_plant | 2.2857 | 1.3325 | 2.2857 |
| 73 | head | 258.31 | 118.21 | 258.31 |
| 74 | t | 83.5 | 48.497 | 83.5 |
| 75 | hod | 11.5 | 6.9222 | 11.5 |
| 76 | dow | 3 | 2 | 2.9999 |
| 77 | doy_sin | 5.8759e-06 | 0.70662 | 0.00026201 |
| 78 | doy_cos | 0.0013671 | 0.70759 | -0.0061918 |
| 79 | holiday | 0.038591 | 0.19262 | 0.036993 |
| 80 | margin_sys_mean | -7.5677 | 57.745 | -7.6186 |
| 81 | margin_minus_sys | 2.1257e-08 | 41.928 | -2.8857e-08 |
| 82 | margin_rank_across_units | 0.5 | 0.31009 | 0.5 |
| 83 | phys_be_rel_min | 1.2727 | 0.33695 | 1.2727 |
| 84 | phys_m_min | -25.809 | 120.09 | -28.995 |
| 85 | phys_m_full | -8.5617 | 73.017 | -8.7158 |
| 86 | phys_m_last | -11.471 | 77.855 | -11.935 |
| 87 | phys_start_per_mw | 5.3343 | 7.0717 | 5.3343 |
| 88 | phys_afford_h | 0.73794 | 0.36417 | 0.72142 |
| 89 | phys_afford_log | 5.5157 | 1.6084 | 5.4393 |
| 90 | phys_forced_h | 0.047642 | 0.14573 | 0.063691 |
| 91 | phys_npos_h | 0.53982 | 0.45708 | 0.45201 |
| 92 | phys_greedy_k | 0.40774 | 0.41667 | 0.36506 |
| 93 | phys_greedy_on | 0.40859 | 0.49157 | 0.36589 |
| 94 | phys_greedy_dist | -0.089284 | 0.50689 | -0.13196 |
| 95 | phys_forced_on | 0.048436 | 0.21469 | 0.064492 |
| 96 | phys_block_len | 0.46779 | 5.9092 | -0.79821 |
| 97 | phys_block_profit | 1.4206e+05 | 3.8e+05 | 1.3438e+05 |
| 98 | phys_stay_on_gain | -1.0324e+05 | 3.7652e+05 | -1.0979e+05 |

## Appendix C: suggested notebook structure for the rebuild

1. Imports, constants (U, H, Q2V, PLANTS, …), device selection (`cuda` > `mps` > `cpu`).
2. Data loading.
3. Base features (section 2), then check them against Appendix B.
4. Physics features (section 3), then check them against Appendix B.
5. Build `X4` for train and test, either as a memory-mapped file or in RAM.
6. Scaler and UCNet (sections 4–5).
7. Optional: 1-fold CV sanity check (2019 fold, about 0.983 after 20 epochs).
8. Final training (3 seeds) and prediction, then write `submission_nn_phys.csv` (section 7).
