# Feature backlog

Candidate features for the iteration phase, organised into the 20 agreed themes. Each entry is
meant to be buildable in one pass and scored with `src/modeling/evaluation.py`.

**Status**: `todo` / `building` / `tested` / `kept` / `rejected`. Record CV mean, SD and the
per-fold sign pattern for anything tested. **Keep rejected rows** — a negative result is a result
(`report_rules.md`).

**Grounding tags**: `[R1]` `[R2]` `[R3]` `[R3b]` = the EDA round that motivates it, with the
measured correlation where one exists. Untagged entries are untested hypotheses.

**Two global rules**
1. *(from [R3])* Trees are invariant to monotonic transforms of a **single** column. Rescaling or
   logging one feature does nothing. Only column **combinations**, **time-window** aggregates,
   **graph/space** aggregates and **cross-sectional ranks** add information.
2. *(from [R3b])* Normalise **fast** series against the **case**; normalise **slow** series
   against the **season**.

**Testing protocol**: rolling-origin CV, 4 folds (2019–2022). Fold-to-fold SD is **0.061**, about
ten times a typical feature effect, so **judge on paired per-fold deltas and the sign pattern, not
the mean** (`log 010`).

---

## Tested so far (log 013) — read this before picking what to build next

Full rolling-origin CV, cumulative stages. Baseline 0.7074 → **0.8080** overall.

| Theme | Status | Δ mean | Folds up | Verdict |
|---|---|---|---|---|
| 6 — price − water-value spread | **kept** | **+0.075** | **4/4** | 60.8% of all model gain; `f06_spread_sys_positive` alone has ~14x the gain of the next feature |
| 7 — terminal water value | **kept** | **+0.034** | **4/4** | The documented unused boundary condition; paid off |
| 9 — inflow regime | tested | +0.006 | 2/4 | Ambiguous, within noise |
| 2 — case-relative price | tested | +0.006 | 2/4 | Ambiguous, but cut fold SD 0.063 → 0.036 |
| 10–12 — physics | tested | +0.001 | 2/4 | 13.5% of gain but ~no incremental value; start-cost break-even features are the useful part |
| 8.7 — drop dead columns | **kept** | −0.000 | 2/4 | Correctly neutral; cut fold SD 0.033 → 0.025 |
| 1 — seasonal percentile | **rejected** | **−0.020** | **0/4** | Highest univariate correlation in the project (+0.537) and it *hurt*. Kept in this table as the worked example |

### Three rules this produced

1. **Univariate correlation does not predict feature value.** It ignores what the other features
   already supply. Order by tested effect; treat correlation-based ordering as a guess.
2. **Hourly-varying beats case-level.** Micro-AUC pools every (case, hour, unit) prediction, so
   most of the problem is *which hours within a week*. Measured on the fitted model: hourly
   features hold **96.4%** of gain, case-level constants **3.6%**. Theme 1, 9 and the case-level
   parts of 18 are structurally handicapped for this metric.
3. **Encoded decision rules beat statistical transforms.** The winner is the optimizer's own
   dispatch condition. Prioritise themes that mirror what SHOP computes over themes that
   re-normalise inputs.

### Re-prioritised order (replaces the correlation-based ordering below)

1. **Water budget / merit dispatch** (new theme 21) — hourly *and* an encoded decision rule, i.e.
   both winning properties
2. **Theme 6 extensions** — more spread variants, since spread dominates everything
3. **Theme 4** price shape, **theme 3** forward windows — hourly-varying
4. **Theme 15** merit order, **theme 14** graph aggregates
5. **Theme 20** case-curve representation
6. Deprioritised: themes 1, 9 and other case-level scalars, on the evidence above

---

## 1. Seasonal normalisation of the slow series `[R3b corr +0.537 — but REJECTED in testing: −0.020, 0/4 folds]`

Absolute levels confound season; "unusual *for this time of year*" isolates the scarcity signal.
All strictly backward-looking, so point-in-time safe.

| # | Feature | Notes |
|---|---|---|
| 1.1 | System storage percentile vs. same day-of-year historically | `rejected` — built, tested, hurt the model |
| 1.2 | Own-reservoir storage percentile vs. same day-of-year | Per-unit version of 1.1 |
| 1.3 | Water-value percentile vs. same day-of-year | −0.417 `[R3b]` |
| 1.4 | Inflow percentile vs. same day-of-year | +0.266 `[R3b]` |
| 1.5 | Storage level − seasonal median (absolute anomaly, not rank) | Rank vs. anomaly may behave differently in trees |
| 1.6 | Seasonal percentile over a ±7d day-of-year window | Smoother; exact-day has few samples |
| 1.7 | Seasonal percentile of the price − water-value spread | Combines themes 1 and 6 |
| 1.8 | Z-score (not percentile) vs. same day-of-year distribution | Keeps magnitude, not just rank |
| 1.9 | Years-of-history-weighted percentile (recent years weighted higher) | Guards against the drift in `[R2]` finding 9 |

## 2. Case-relative price normalisation `[R3: raw −0.185 → z-scored +0.263]`

| # | Feature | Notes |
|---|---|---|
| 2.1 | `price_z_case` = (price − case mean) / case std | +0.263; centring flips the sign, scaling adds the rest |
| 2.2 | `price_pct_case` = percentile rank within the 168h case | +0.252, distribution-free |
| 2.3 | `price_log_ratio_case` = log(price) − log(case mean) | +0.231; safer when case std is tiny |
| 2.4 | `price_z_day` = z-score within the row's own calendar day | Separates intraday shape from weekly level |
| 2.5 | `price_rank_in_day` (1–24) | "Is this a peak hour today" |
| 2.6 | `price − case_min`, `case_max − price` | Distance to the week's cheapest/dearest hour |
| 2.7 | `price_z_case` computed on log price | Tames 2022 without losing ordering |
| 2.8 | Case `price_std`, `price_iqr`, `max/min` broadcast to every row | How much spread there is to exploit |

## 3. Price temporal windows, forward and backward `[R3b: |corr| min at k=0, ≈−0.245 at ±96h]`

Clip every window to the case horizon — [R3b] showed the uncorrected version reached 82h past it.

| # | Feature | Notes |
|---|---|---|
| 3.1 | Forward mean of price over next 6/12/24/48/72h | Safe: price file verified to cover the test window |
| 3.2 | Backward mean over previous 6/12/24/48/72h | Past matters nearly as much as future |
| 3.3 | `price − forward_mean_24h`, `price − backward_mean_24h` | Best rolling variants reached only +0.125 `[R3b]` |
| 3.4 | Centred window means (±12/24/48h), clipped to horizon | |
| 3.5 | Signed hours until the case's max-price hour | "Wait for the peak" behaviour |
| 3.6 | Signed hours since the case's min-price hour | |
| 3.7 | Forward/backward price slope (linear fit over 12h) | Ramp direction |
| 3.8 | Forward rolling std of price (next 24h) | Volatility ahead |
| 3.9 | Beyond-horizon forward mean (deliberately crosses the boundary) | Permitted, unmotivated — worth one test to settle it `[R3b]` |

## 4. Price curve shape and structure

| # | Feature | Notes |
|---|---|---|
| 4.1 | Count of local price peaks in the case | How many commitment cycles the week needs |
| 4.2 | Case price skew and kurtosis | One spike vs. broadly elevated |
| 4.3 | Ratio of case top-24h mean to bottom-24h mean | Is the spread worth cycling for |
| 4.4 | Binary: is this hour in the case's top-k by price (k=12/24/48) | Interacts naturally with merit order |
| 4.5 | Price autocorrelation at lag 24 within the case | How repeatable the daily shape is |
| 4.6 | Cumulative share of case price mass up to this hour | Position along the week's value curve |
| 4.7 | Number of contiguous runs above the case median | Block structure of profitable periods |

## 5. Water value: common factor vs. deviation `[R3: PC1 = 91% of variance]`

| # | Feature | Notes |
|---|---|---|
| 5.1 | `sys_wv` = mean water value across all 17 reservoirs | The single dominant factor |
| 5.2 | `own_wv − sys_wv` | **Strong for Vesle_Kjela/Byrte/Kjela/Lio; ~0.01 for Tokke** `[R3]` |
| 5.3 | `own_wv / sys_wv` | Scale-free variant |
| 5.4 | PC1 score of the water-value matrix as one column | Replaces 17 near-duplicate columns |
| 5.5 | PC2/PC3 scores | Residual structure after the common factor |
| 5.6 | Rank of own reservoir's water value among the 17 | Cross-sectional rank |
| 5.7 | Drop the 15 redundant per-reservoir water-value columns | Test whether removal helps (less split dilution) |
| 5.8 | Dispersion of water value across reservoirs (std) | System-wide disagreement about scarcity |

## 6. Price − water-value spread `[R2: beats both components for every unit; R3: do not normalise]`

| # | Feature | Notes |
|---|---|---|
| 6.1 | `price − own_reservoir_wv` | **Raw +0.241 vs z-scored +0.114 — use raw** `[R3]` |
| 6.2 | `price − sys_wv` | System-level version |
| 6.3 | `price / own_wv` | Ratio form |
| 6.4 | Sign of the spread (binary) | The actual dispatch rule |
| 6.5 | Forward rolling mean of the spread (next 24/48h) | Combines themes 3 and 6 |
| 6.6 | Count of hours in the case where spread > 0 | "How many profitable hours this week" |
| 6.7 | Spread × unit `p_max` | Approximate marginal profit of running |
| 6.8 | Spread − start-cost-per-MWh threshold | Ties to theme 10; the true break-even |

## 7. Terminal / horizon boundary conditions `[R3: terminal wv −0.41 vs initial −0.26]`

| # | Feature | Notes |
|---|---|---|
| 7.1 | Terminal (day+8) water value per reservoir | **Documented in the dataset PDF, entirely unused** |
| 7.2 | `terminal_wv − initial_wv` (drift over the horizon) | Is water getting more or less valuable |
| 7.3 | Terminal system-mean water value | |
| 7.4 | Water value interpolated to the row's own day | Richer than holding day-1 constant |
| 7.5 | `price − terminal_wv` spread | End-of-horizon version of theme 6 |
| 7.6 | Ratio terminal/initial | |
| 7.7 | Fallback for the final test case (2024-12-25) | **Required**: day+8 exceeds the file `[R3b verified]` |

## 8. Storage state and trajectory

| # | Feature | Notes |
|---|---|---|
| 8.1 | System percent full (Σ storage / Σ `max_vol`) | Genuine cross-reservoir combination |
| 8.2 | Own reservoir percent full − system percent full | Relative fullness |
| 8.3 | Storage change over previous 7/30 days | Filling vs. drawing down; weak alone `[R3b]` |
| 8.4 | Headroom to `hrl` and to `lrl` | Physical limits from topology, unused |
| 8.5 | Fraction of usable range remaining ((vol−lrl_vol)/(hrl_vol−lrl_vol)) | Normalised against the *regulated* range, not max_vol |
| 8.6 | Days of full-output generation the storage supports | storage ÷ (p_max × conversion); ties to theme 11 |
| 8.7 | Drop `storage_Vatjern`, `storage_Hyljelihyl` | **Constant columns, zero information** `[R2]` |
| 8.8 | Upstream cascade total storage | Needs the multi-hop map (theme 14) |

## 9. Inflow and hydrology regime `[R3b: trailing 365d mean +0.331 vs level +0.128]`

| # | Feature | Notes |
|---|---|---|
| 9.1 | Inflow trailing 365d mean | **+0.331 — the wet-year/dry-year regime** |
| 9.2 | Total system inflow | PC1 = 70%, aggregate captures most `[R3]` |
| 9.3 | Own-reservoir inflow ÷ its `max_vol` | Fill rate, not absolute flow |
| 9.4 | Forward inflow over next 24/72/168h | Known at solve time |
| 9.5 | Cumulative inflow so far in the case | |
| 9.6 | Inflow trailing 30d ÷ trailing 365d | Short-run vs. long-run regime ratio |
| 9.7 | Upstream cascade inflow (all reservoirs draining to this unit) | Needs the graph walk |
| 9.8 | Data-driven snowmelt onset (inflow surge date) vs. calendar | Season without assuming fixed dates |

## 10. Commitment economics: start/stop costs 🆕 `[corr(startcost/MW, toggling) = −0.673]`

`startcost_const` and `stopcost_const` exist per generator and were never opened. They are the
central economic driver of unit commitment and mechanically explain the run-length differences
measured in `[R2]`.

| # | Feature | Notes |
|---|---|---|
| 10.1 | `startcost_const` per unit (static) | Range 92 (Vesle Kjela) to 222 (Songa) |
| 10.2 | `startcost / p_max` — start cost per MW | **−0.673 with observed toggling**; Haukeli 29.2, Vinje 1.98 |
| 10.3 | Break-even run hours = startcost ÷ (spread × p_max) | How long a run must be to pay back the start |
| 10.4 | Binary: does the forward price window exceed the break-even? | Turns 10.3 into the actual decision rule |
| 10.5 | Contiguous hours in the case where spread > 0, vs. break-even hours | Is any profitable block long enough to justify starting |
| 10.6 | `stopcost_const` and start+stop round-trip cost | Symmetric here (both 214 for Tokke) but encode anyway |
| 10.7 | Start cost expressed in MWh-equivalent at the current price | Puts cost and revenue in the same units |

## 11. Head and energy conversion 🆕 `[verified: level − outlet_line matches efficiency-curve head refs for all 8 plants]`

The chain **storage → `vol_head` → reservoir level → minus plant `outlet_line` → gross head →
efficiency curve → MW per m³/s** is fully computable and entirely absent from the feature set.
Head swings widely for some plants (Kjela 125–191 m, Vesle Kjela 44–71 m), so the power obtained
per unit of water depends strongly on reservoir fullness.

| # | Feature | Notes |
|---|---|---|
| 11.1 | Current gross head = `vol_head`(storage) − plant `outlet_line` | **The core physical feature** |
| 11.2 | Head as a fraction of the plant's own head range | Scale-free across plants |
| 11.3 | Net head after `main_loss` and `penstock_loss` | Losses are flow-dependent (quadratic) |
| 11.4 | Turbine efficiency at current head, interpolated across the 3 `turb_eff_curves` | Efficiency is head-dependent |
| 11.5 | Best-efficiency flow at the current head | Where the unit "wants" to run |
| 11.6 | Energy conversion factor (MWh per m³) at current head | Converts water to a common currency |
| 11.7 | Stored energy upstream = Σ (volume × conversion factor) | MWh, not m³ — comparable across reservoirs |
| 11.8 | Head-adjusted marginal value = spread × conversion factor | Ties themes 6, 10 and 11 together |
| 11.9 | `gen_eff_curve` value at the unit's nominal output | Generator (not turbine) efficiency |

## 12. Penstock and hydraulic coupling 🆕 `[explains Tokke's G1→G3 merit order]`

Tokke G1+G2 share penstock 1; G3+G4 share penstock 2. Vinje's three units have separate
penstocks. The observed merit order G1 → G1+**G3** → … is one unit per penstock before doubling
up, consistent with limiting quadratic penstock head loss.

| # | Feature | Notes |
|---|---|---|
| 12.1 | Penstock id per unit (static categorical) | |
| 12.2 | Number of units sharing this unit's penstock | 2 for Tokke, 1 for Vinje |
| 12.3 | Is this unit the first or second on its penstock in merit order | Directly encodes the observed pattern |
| 12.4 | Count of *other* units on the same penstock currently committed | Requires cross-unit prediction (theme 20) |
| 12.5 | Penstock loss coefficient from the plant record | `penstock_loss` list, per penstock |
| 12.6 | Expected head loss at full plant flow through this penstock | Quadratic in flow |
| 12.7 | Plant capacity reachable without doubling up on a penstock | The "cheap" commitment level |

## 13. Flow limits and hydraulic losses (river / tunnel) 🆕

37 river and 5 tunnel records, none used. Rivers carry `up_head_flow_curve` (a head → maximum
flow relationship) and `flow_cost_const`; tunnels carry `length`, `loss_factor` and endpoint
heights.

| # | Feature | Notes |
|---|---|---|
| 13.1 | Max discharge capacity of the unit's outflow river at current head | A hard physical throughput ceiling |
| 13.2 | Headroom = capacity − current required flow | How constrained the path is |
| 13.3 | Tunnel `loss_factor` × `length` on the unit's supply path | Cumulative hydraulic resistance |
| 13.4 | Tunnel elevation drop (`start_height` − `end_height`) | Gravity feed vs. pumped |
| 13.5 | `flow_cost_const` on the unit's spill path | Penalty the optimizer pays for spilling |
| 13.6 | Binary: is the unit fed via tunnel rather than a direct reservoir | Songa and Haukeli differ structurally |
| 13.7 | Sum of min_flow obligations on downstream rivers | Forced-release pressure; ties to constraints |

## 14. Topology graph aggregates, multi-hop `[R2: Songa needs 2-hop; R3: deviations help selectively]`

| # | Feature | Notes |
|---|---|---|
| 14.1 | Correct multi-hop unit → reservoir map | **Prerequisite for most of this theme.** Songa draws via two tunnels |
| 14.2 | 1-hop upstream mean of water value / storage / inflow | |
| 14.3 | Full-cascade upstream aggregate | The "GNN-lite" test |
| 14.4 | Downstream reservoir state (where the unit discharges) | Downstream headroom constrains release |
| 14.5 | Graph depth / number of upstream reservoirs per unit | Static structural attribute |
| 14.6 | Upstream cascade percent full | Cascade-level scarcity |
| 14.7 | Water-value gradient along the cascade (own − upstream mean) | Where water is worth most |
| 14.8 | Shared-reservoir indicator (units competing for the same water) | Tokke and Haukeli both touch Vinjevatn |

## 15. Plant aggregates and merit order `[R2: count → exact pattern 97.8% / 99.9%]`

| # | Feature | Notes |
|---|---|---|
| 15.1 | Static merit-order rank within the plant | Tokke G1=1, G3=2, then G4/G2; Vinje G1→G2→G3 |
| 15.2 | Plant total `p_max`, and this unit's share | |
| 15.3 | Cumulative capacity at or below this unit's rank | "How much load before this unit is needed" |
| 15.4 | Number of units at the plant | Separates single- from multi-unit plants |
| 15.5 | Binary: is this the marginal (last-in-order) unit | The genuinely ambiguous slot |
| 15.6 | Implied plant load = f(spread) ÷ plant capacity | Proxy for the count |
| 15.7 | Binary: is this the ambiguous Tokke third slot (G2 vs G4) | The one 34/66 cell `[R2]` |

## 16. Unit static attributes and identity

| # | Feature | Notes |
|---|---|---|
| 16.1 | `p_min`, `p_max`, `p_nom` | Currently unused entirely |
| 16.2 | `p_min / p_max` — how "lumpy" the unit is | Governs minimum viable commitment |
| 16.3 | Historical ON-rate prior per unit (train period only) | Static base rate; recompute per fold to avoid leakage |
| 16.4 | Historical mean toggles per week per unit | Baseload vs. cycling character `[R2]` |
| 16.5 | Historical flat-week fraction per unit | Haukeli 81%, Vinje_G3 46% `[R2]` |
| 16.6 | Unit id as a categorical | Lets the joint model separate units explicitly |
| 16.7 | Peak turbine efficiency and its flow | From `turb_eff_curves` |

## 17. Calendar and seasonality `[R1: intraday strong, day-of-week weak]`

| # | Feature | Notes |
|---|---|---|
| 17.1 | Hour of day (0–23) | Much stronger than day-of-week `[R1]` |
| 17.2 | sin/cos of hour of day | No-op for trees; test only alongside non-tree models |
| 17.3 | Day of year, and sin/cos encoding | Snowmelt cycle |
| 17.4 | Month | |
| 17.5 | Is-weekend | Weak `[R1]` but free |
| 17.6 | Season defined by inflow regime rather than calendar | Data-driven; ties to 9.8 |

## 18. Case structure, horizon position and case-level summaries

| # | Feature | Notes |
|---|---|---|
| 18.1 | `hour` (0–167) — already present | |
| 18.2 | Day within the case (0–6) | |
| 18.3 | Hours remaining to horizon end | |
| 18.4 | Case mean/std/min/max of price, spread and inflow | Gives every row its week's context |
| 18.5 | Case-level system storage and water value at start | |
| 18.6 | Case start day-of-year / season | |
| 18.7 | ~~Sample weighting by horizon position~~ | **Not justified — disagreement is flat** `[R3]` |

## 19. Case-sequence dynamics (input-only) 🆕

Consecutive cases overlap on 144 wall-clock hours. Features comparing a case to its predecessor
use only *inputs*, so they stay legal — unlike anything derived from previous labels, which is
unusable on the test set (`[R3b]` verified: no test labels exist).

| # | Feature | Notes |
|---|---|---|
| 19.1 | Change in price forecast vs. the previous case, same wall-clock hour | "The outlook got worse since yesterday" |
| 19.2 | Change in initial storage vs. the previous case | Did the system draw down faster than expected |
| 19.3 | Change in water value vs. the previous case | Scarcity re-pricing |
| 19.4 | Change in the terminal water value vs. the previous case | Boundary condition movement |
| 19.5 | Rolling 7-case mean of the above deltas | Trend in revisions |
| 19.6 | ⚠️ Previous case's *labels* at the overlapping hour | **Unusable beyond the first test case** — documented so nobody builds it |

## 20. Case-curve representation 🆕

The 168h price path is the single richest object in a case, currently compressed only into
per-hour scalars. Encode its *shape* directly.

| # | Feature | Notes |
|---|---|---|
| 20.1 | Top-k PCA components of the case's 168h price curve | Case shape in a handful of numbers |
| 20.2 | Fourier coefficients at the daily (24h) and weekly harmonics | Strength of the daily cycle this week |
| 20.3 | Cluster id from k-means over normalised case price curves | Categorical "week type" |
| 20.4 | Distance to each cluster centroid | Soft version of 20.3 |
| 20.5 | PCA components of the case's storage/water-value trajectory | Same idea on the slow series |
| 20.6 | Reconstruction error against the k-component basis | "Is this an unusual week" |

---

## Cross-cutting practices

Applied within themes rather than being themes of their own.

| Practice | Rationale |
|---|---|
| **Recency weighting / shorter training window** | `[R2]` Haukeli drifts 0.87 → 0.14 across the period; the drift is larger than the 2022 price shock |
| **Interactions and ratios** | e.g. `price_z_case` × merit rank, spread × percent-full, spread ÷ `p_min` |
| **Historical analogue matching** | Nearest historical case by price-curve shape (theme 20 gives the distance metric); use with care re: leakage |
| **Feature removal** | The 2 constant columns (8.7), the 15 redundant water-value columns (5.7); test whether removal *helps* |
| **Target reformulation** | `[R2]` predict per-plant counts and expand by merit order; or split "does it move this week" from "when" |

## Iteration order

1. **1.1 + 1.3** seasonal percentiles of storage and water value — strongest evidence (+0.537 / −0.417)
2. **9.1** inflow trailing 365d mean (+0.331)
3. **2.1 + 2.2** price z-score and percentile within case (+0.263 / +0.252)
4. **6.1** raw price − own-reservoir water-value spread
5. **7.1 + 7.2** terminal water value and drift (mind the 7.7 fallback)
6. **8.7** drop the two constant columns (free)
7. **10.1 + 10.2** start costs — new source, strong mechanism, trivially cheap (static per unit)
8. **11.1 + 11.6** gross head and the energy conversion factor — new source, real physics
9. **12.1–12.3** penstock structure — explains an ordering we otherwise can't
10. **5.1 + 5.2** water-value factor and own-minus-system deviation
11. **14.1** the multi-hop map (unblocks theme 14 and parts of 8, 9, 13)
12. **Cross-cutting: recency weighting**
13. **20.1** case-curve PCA
14. **Target reformulation (counts + merit order)** — biggest structural change, last

**Deprioritised by evidence**: rolling-window and EMA relative-price features (3.1–3.4) are about
half as strong as the case z-score; diffs and momentum are weak everywhere tested `[R3b]`.
