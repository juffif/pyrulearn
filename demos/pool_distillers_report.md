# Rule pools x distillers

## Setup

**Full run**. Quick sanity check instead: `python demos/pool_distillers.py` with no arguments.

Three rule-pool generators (`CARMiner`, `SKLRandomForest`,
`pylord_candidates`) crossed with seven rule distillers (`RuleFit`,
`CBA`, `CMAR`, `IDS`, `IDS-greedy`, `Max` and `Vote` -- the last two
with no distillation at all, just the whole pool predicted from
directly), each pool built once per fold and shared across every
distiller. `IDS`/`IDS-greedy` are the same native
`pyrulearn.learners.ids.IDS`, differing only in `optimizer=`: `"sls"`
(Smooth Local Search, the paper-faithful default -- stochastic, no
convergence guarantee for a non-monotone objective) vs. `"greedy"` (a
cheap, deterministic fallback); both are seeded (`seed=0`) -- the
objective's weights are chosen on a random validation split, and SLS samples
at random -- so every run learns the same models. `Max`/`Vote` are the same `FlatRuleSet`
wrapper, differing only in `combiner=`.

Two studies, since the pool generators don't scale alike (see the
module docstring):

- **Study 1** -- vote, SPECT, hepatitis, tic-tac-toe, heart-statlog (5-fold) -- uses all three
  pool generators (`CARMiner`: `min_support=0.05`,
  `max_len=3`; `SKLRandomForest`: `max_depth=3`;
  `pylord_candidates`: defaults) and all seven distillers.
- **Study 2** -- credit-g, kr-vs-kp, sick, spambase, mushroom, bank-marketing, adult (3-fold) -- uses
  `SKLRandomForest` (flat pool size/build time regardless of scale) and
  `pylord_candidates` (one local search per training row, growing
  super-linearly -- feasible on the smaller datasets here, expected to
  run long or time out on `bank-marketing`/`adult`), and only six
  distillers: plain `IDS` (`SmoothLocalSearch`) is dropped here -- on
  Study 1's own small datasets it was already the slowest distiller by
  a wide margin (its cost doesn't track data size in any simple way --
  see "Why IDS is slow and unstable" below), so repeating it on bigger
  data wasn't worth the time. `IDS-greedy` stays.

Protocol: stratified cross-validation (`pyrulearn.experiments.runner.run_cv`) in two
passes over the same folds -- first every pool is built on every training fold, then
every distiller is fit on the pool built on its fold (the time to load the stored pool is
not counted in the distiller's fit time). Data in the library's default representation
(N-lists). Discretization: `max_intervals=3`. Distiller fits are
capped at 300s; pool building gets a longer cap of
1800s, since `pylord_candidates`' cost grows
super-linearly with row count and can need much longer than any
distiller fit on the biggest Study 2 datasets. A time-out or an error
counts as a failure, not as the end of the run. Every learner runs single-threaded; fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: pool build time and rule count per generator; per
(pool, distiller) mean accuracy, rule count and fit time; mean rank of
the distillers, computed separately per pool; per-dataset detail
(mean across folds) for every cell.

How failures are treated, throughout: a failed cell is dropped from
every mean it would otherwise enter (accuracy, fit time, build time,
rule count), never counted as a 0 or averaged in as a NaN, and the
number of failures is reported alongside every such mean (a "failures"
or "pool-build failures" column) rather than silently disappearing. A
pool that failed to build on some dataset removes every distiller's
row for that (dataset, pool) from the per-dataset table too -- no
distiller was ever attempted there. One consequence: if a pool fails on
a dataset the others don't, that pool's own mean rank ends up averaged
over fewer datasets than the pools it's compared against in the same
table -- flagged inline wherever it actually happens, with a second,
common-datasets-only table alongside for a fair head-to-head.

## Study 1: small datasets, all three pool generators

### Pool stats (mean across datasets and folds)

| pool | build time (s) | n_rules | mean accuracy (all distillers) | pool-build failures |
|---|--:|--:|--:|--:|
| CAR | 0.73 | 11035.5 | 82.98% | 0 |
| LORD | 0.52 | 92.6 | 85.95% | 0 |
| RF | 0.34 | 740.0 | 82.92% | 0 |

### Pool x distiller grid

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| CAR | RuleFit | 87.48% | 52.2 | 0.55 | 0 |
| CAR | CBA | 88.38% | 27.3 | 0.56 | 0 |
| CAR | CMAR | 82.17% | 23.8 | 0.47 | 0 |
| CAR | IDS | 79.09% | 22.2 | 44.26 | 0 |
| CAR | IDS-greedy | 81.01% | 8.0 | 7.07 | 0 |
| CAR | Max | 83.78% | 11035.2 | 0.00 | 0 |
| CAR | Vote | 78.96% | 11035.2 | 0.00 | 0 |
| LORD | RuleFit | 87.18% | 30.5 | 0.02 | 0 |
| LORD | CBA | 86.79% | 39.4 | 0.02 | 0 |
| LORD | CMAR | 85.83% | 34.9 | 0.01 | 0 |
| LORD | IDS | 81.70% | 9.0 | 74.20 | 0 |
| LORD | IDS-greedy | 85.13% | 10.0 | 10.33 | 0 |
| LORD | Max | 87.46% | 92.1 | 0.00 | 0 |
| LORD | Vote | 87.57% | 92.1 | 0.00 | 0 |
| RF | RuleFit | 87.78% | 43.7 | 0.05 | 0 |
| RF | CBA | 84.34% | 34.8 | 0.05 | 0 |
| RF | CMAR | 82.02% | 28.7 | 0.03 | 0 |
| RF | IDS | 76.37% | 21.3 | 55.70 | 0 |
| RF | IDS-greedy | 81.06% | 8.5 | 7.21 | 0 |
| RF | Max | 84.71% | 740.0 | 0.00 | 0 |
| RF | Vote | 84.16% | 740.0 | 0.00 | 0 |

### Mean rank of the distillers, by pool

| distiller | CAR | LORD | RF | overall |
|---|--:|--:|--:|--:|
| RuleFit | 2.80 | 2.80 | 1.90 | 2.50 |
| CBA | 1.90 | 3.50 | 3.90 | 3.10 |
| Vote | 5.60 | 2.20 | 2.20 | 3.33 |
| Max | 3.90 | 2.80 | 4.20 | 3.63 |
| IDS-greedy | 4.30 | 5.30 | 4.30 | 4.63 |
| CMAR | 4.00 | 4.70 | 5.50 | 4.73 |
| IDS | 5.50 | 6.70 | 6.00 | 6.07 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| SPECT | CAR | CBA | 80.92% | 33.6 | 0.712 |
| SPECT | CAR | CMAR | 79.03% | 47.2 | 0.217 |
| SPECT | CAR | IDS | 62.93% | 21.2 | 35.205 |
| SPECT | CAR | IDS-greedy | 70.42% | 8.2 | 5.913 |
| SPECT | CAR | Max | 79.40% | 9483.4 | 0.003 |
| SPECT | CAR | RuleFit | 82.02% | 48.0 | 0.467 |
| SPECT | CAR | Vote | 79.40% | 9483.4 | 0.002 |
| SPECT | LORD | CBA | 82.42% | 43.2 | 0.014 |
| SPECT | LORD | CMAR | 79.02% | 25.0 | 0.004 |
| SPECT | LORD | IDS | 79.40% | 6.6 | 93.406 |
| SPECT | LORD | IDS-greedy | 79.40% | 5.8 | 3.258 |
| SPECT | LORD | Max | 82.37% | 70.4 | 0.000 |
| SPECT | LORD | RuleFit | 83.88% | 31.4 | 0.012 |
| SPECT | LORD | Vote | 83.14% | 70.4 | 0.000 |
| SPECT | RF | CBA | 81.66% | 32.6 | 0.046 |
| SPECT | RF | CMAR | 79.03% | 40.2 | 0.014 |
| SPECT | RF | IDS | 56.17% | 27.8 | 40.273 |
| SPECT | RF | IDS-greedy | 66.32% | 8.6 | 6.810 |
| SPECT | RF | Max | 79.40% | 692.2 | 0.000 |
| SPECT | RF | RuleFit | 83.13% | 32.6 | 0.038 |
| SPECT | RF | Vote | 80.52% | 692.2 | 0.000 |
| heart-statlog | CAR | CBA | 83.70% | 40.8 | 0.165 |
| heart-statlog | CAR | CMAR | 81.85% | 23.0 | 1.097 |
| heart-statlog | CAR | IDS | 78.52% | 19.2 | 36.837 |
| heart-statlog | CAR | IDS-greedy | 78.52% | 8.4 | 6.791 |
| heart-statlog | CAR | Max | 81.48% | 17368.4 | 0.004 |
| heart-statlog | CAR | RuleFit | 78.89% | 87.4 | 0.786 |
| heart-statlog | CAR | Vote | 81.48% | 17368.4 | 0.004 |
| heart-statlog | LORD | CBA | 80.00% | 48.2 | 0.016 |
| heart-statlog | LORD | CMAR | 80.00% | 42.2 | 0.007 |
| heart-statlog | LORD | IDS | 75.19% | 8.2 | 76.038 |
| heart-statlog | LORD | IDS-greedy | 78.15% | 13.0 | 20.325 |
| heart-statlog | LORD | Max | 81.11% | 85.0 | 0.000 |
| heart-statlog | LORD | RuleFit | 78.89% | 35.2 | 0.013 |
| heart-statlog | LORD | Vote | 81.11% | 85.0 | 0.000 |
| heart-statlog | RF | CBA | 79.63% | 44.0 | 0.032 |
| heart-statlog | RF | CMAR | 80.37% | 23.4 | 0.030 |
| heart-statlog | RF | IDS | 75.56% | 17.2 | 51.762 |
| heart-statlog | RF | IDS-greedy | 80.37% | 7.6 | 6.667 |
| heart-statlog | RF | Max | 80.00% | 784.6 | 0.000 |
| heart-statlog | RF | RuleFit | 80.37% | 49.2 | 0.041 |
| heart-statlog | RF | Vote | 81.48% | 784.6 | 0.000 |
| hepatitis | CAR | CBA | 82.58% | 19.8 | 0.628 |
| hepatitis | CAR | CMAR | 83.87% | 15.2 | 0.329 |
| hepatitis | CAR | IDS | 81.29% | 17.0 | 73.661 |
| hepatitis | CAR | IDS-greedy | 81.29% | 6.6 | 3.727 |
| hepatitis | CAR | Max | 80.65% | 8826.8 | 0.002 |
| hepatitis | CAR | RuleFit | 80.65% | 45.0 | 0.375 |
| hepatitis | CAR | Vote | 79.35% | 8826.8 | 0.002 |
| hepatitis | LORD | CBA | 81.29% | 24.2 | 0.007 |
| hepatitis | LORD | CMAR | 80.00% | 18.2 | 0.003 |
| hepatitis | LORD | IDS | 78.71% | 10.0 | 5.982 |
| hepatitis | LORD | IDS-greedy | 81.94% | 10.8 | 3.678 |
| hepatitis | LORD | Max | 83.87% | 37.8 | 0.000 |
| hepatitis | LORD | RuleFit | 80.65% | 18.0 | 0.007 |
| hepatitis | LORD | Vote | 83.87% | 37.8 | 0.000 |
| hepatitis | RF | CBA | 77.42% | 22.4 | 0.032 |
| hepatitis | RF | CMAR | 80.65% | 17.0 | 0.020 |
| hepatitis | RF | IDS | 78.71% | 14.0 | 97.715 |
| hepatitis | RF | IDS-greedy | 81.94% | 12.4 | 12.174 |
| hepatitis | RF | Max | 81.94% | 682.2 | 0.000 |
| hepatitis | RF | RuleFit | 81.94% | 27.4 | 0.029 |
| hepatitis | RF | Vote | 85.81% | 682.2 | 0.000 |
| tic-tac-toe | CAR | CBA | 100.00% | 10.0 | 1.038 |
| tic-tac-toe | CAR | CMAR | 73.69% | 23.4 | 0.461 |
| tic-tac-toe | CAR | IDS | 78.71% | 25.8 | 58.574 |
| tic-tac-toe | CAR | IDS-greedy | 79.64% | 12.4 | 16.821 |
| tic-tac-toe | CAR | Max | 82.88% | 16517.2 | 0.004 |
| tic-tac-toe | CAR | RuleFit | 100.00% | 47.6 | 0.942 |
| tic-tac-toe | CAR | Vote | 65.34% | 16517.2 | 0.004 |
| tic-tac-toe | LORD | CBA | 96.66% | 47.6 | 0.040 |
| tic-tac-toe | LORD | CMAR | 97.49% | 60.6 | 0.021 |
| tic-tac-toe | LORD | IDS | 87.16% | 11.8 | 186.702 |
| tic-tac-toe | LORD | IDS-greedy | 94.67% | 13.6 | 22.713 |
| tic-tac-toe | LORD | Max | 95.72% | 226.2 | 0.000 |
| tic-tac-toe | LORD | RuleFit | 98.02% | 43.8 | 0.032 |
| tic-tac-toe | LORD | Vote | 94.78% | 226.2 | 0.000 |
| tic-tac-toe | RF | CBA | 87.37% | 45.8 | 0.065 |
| tic-tac-toe | RF | CMAR | 75.37% | 42.0 | 0.045 |
| tic-tac-toe | RF | IDS | 75.99% | 20.6 | 26.211 |
| tic-tac-toe | RF | IDS-greedy | 81.52% | 8.4 | 7.637 |
| tic-tac-toe | RF | Max | 87.26% | 800.0 | 0.000 |
| tic-tac-toe | RF | RuleFit | 97.81% | 81.4 | 0.079 |
| tic-tac-toe | RF | Vote | 77.35% | 800.0 | 0.000 |
| vote | CAR | CBA | 94.71% | 32.4 | 0.237 |
| vote | CAR | CMAR | 92.41% | 10.4 | 0.228 |
| vote | CAR | IDS | 94.02% | 27.8 | 17.009 |
| vote | CAR | IDS-greedy | 95.17% | 4.4 | 2.098 |
| vote | CAR | Max | 94.48% | 2980.0 | 0.001 |
| vote | CAR | RuleFit | 95.86% | 32.8 | 0.179 |
| vote | CAR | Vote | 89.20% | 2980.0 | 0.001 |
| vote | LORD | CBA | 93.56% | 34.0 | 0.014 |
| vote | LORD | CMAR | 92.64% | 28.6 | 0.005 |
| vote | LORD | IDS | 88.05% | 8.2 | 8.864 |
| vote | LORD | IDS-greedy | 91.49% | 6.6 | 1.668 |
| vote | LORD | Max | 94.25% | 41.0 | 0.000 |
| vote | LORD | RuleFit | 94.48% | 24.2 | 0.034 |
| vote | LORD | Vote | 94.94% | 41.0 | 0.000 |
| vote | RF | CBA | 95.63% | 29.4 | 0.056 |
| vote | RF | CMAR | 94.71% | 20.8 | 0.034 |
| vote | RF | IDS | 95.40% | 26.8 | 62.529 |
| vote | RF | IDS-greedy | 95.17% | 5.6 | 2.772 |
| vote | RF | Max | 94.94% | 740.8 | 0.000 |
| vote | RF | RuleFit | 95.63% | 27.8 | 0.059 |
| vote | RF | Vote | 95.63% | 740.8 | 0.000 |

## Why IDS is slow and unstable

`IDS` (`optimizer="sls"`, Smooth Local Search, the paper-faithful default) and `IDS-greedy` (`optimizer="greedy"`, a cheap deterministic fallback) both compress the *same* pool, so any difference here is the optimizer, not the data. On Study 1's small datasets:

| pool | optimizer | mean accuracy | mean fit time (s) |
|---|---|--:|--:|
| CAR | IDS | 79.09% | 44.26 |
| CAR | IDS-greedy | 81.01% | 7.07 |
| LORD | IDS | 81.70% | 74.20 |
| LORD | IDS-greedy | 85.13% | 10.33 |
| RF | IDS | 76.37% | 55.70 |
| RF | IDS-greedy | 81.06% | 7.21 |

Plain `IDS`'s fit time isn't a steady cost that happens to be high -- it varies enormously by (dataset, pool), not just by pool: slowest measured here is `tic-tac-toe`/`LORD` at 186.7s mean, fastest is `hepatitis`/`LORD` at 5.98s -- a 31x spread between datasets this demo treats as comparably small. This is consistent with Smooth Local Search's randomized-restart search having no convergence guarantee on a non-monotone objective, unlike `IDS-greedy`'s single deterministic pass -- not a cost that scales predictably with the data, but per-instance variance. A second, independent symptom of the same instability: on `kr-vs-kp` (Study 2's pool generator, `SKLRandomForest`, tested there before plain `IDS` was dropped from that study), one fold's accuracy came in at 52.30% (near chance) against 84.00-89.70% on the other two folds of the same dataset and pool. Both the speed and the accuracy can apparently collapse independently of data size -- why plain `IDS` stays out of Study 2 (see Setup) while `IDS-greedy`, deterministic, stays in.

## Study 2: larger datasets, RandomForest and LORD pools

### Pool stats (mean across datasets and folds)

| pool | build time (s) | n_rules | mean accuracy (all distillers) | pool-build failures |
|---|--:|--:|--:|--:|
| RF | 4.41 | 751.3 | 88.82% | 0 |
| LORD | 230.97 | 2773.4 | 89.68% | 0 |

### Distiller results

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| RF | RuleFit | 91.10% | 146.5 | 2.13 | 0 |
| RF | CBA | 90.45% | 61.3 | 0.50 | 0 |
| RF | CMAR | 88.76% | 29.4 | 0.19 | 0 |
| RF | IDS-greedy | 86.27% | 5.1 | 5.66 | 0 |
| RF | Max | 89.02% | 751.3 | 0.00 | 0 |
| RF | Vote | 87.33% | 751.3 | 0.00 | 0 |
| LORD | RuleFit | 90.83% | 702.9 | 10.23 | 0 |
| LORD | CBA | 90.54% | 1375.0 | 20.21 | 0 |
| LORD | CMAR | 88.07% | 662.9 | 1.16 | 0 |
| LORD | IDS-greedy | 87.14% | 11.1 | 23.85 | 0 |
| LORD | Max | 90.74% | 2773.4 | 0.00 | 0 |
| LORD | Vote | 90.73% | 2773.4 | 0.00 | 0 |

### Mean rank of the distillers

| distiller | RF | LORD | overall |
|---|--:|--:|--:|
| RuleFit | 1.50 | 2.14 | 1.82 |
| CBA | 2.21 | 3.14 | 2.68 |
| Max | 4.14 | 2.29 | 3.21 |
| Vote | 4.86 | 2.71 | 3.79 |
| CMAR | 3.57 | 5.00 | 4.29 |
| IDS-greedy | 4.71 | 5.71 | 5.21 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| adult | LORD | CBA | 83.01% | 4510.7 | 75.276 |
| adult | LORD | CMAR | 82.47% | 1172.3 | 4.428 |
| adult | LORD | IDS-greedy | 78.37% | 5.3 | 17.187 |
| adult | LORD | Max | 83.37% | 9338.3 | 0.010 |
| adult | LORD | RuleFit | 85.08% | 2224.7 | 36.884 |
| adult | LORD | Vote | 84.20% | 9338.3 | 0.010 |
| adult | RF | CBA | 84.27% | 115.0 | 2.044 |
| adult | RF | CMAR | 83.74% | 40.3 | 0.514 |
| adult | RF | IDS-greedy | 79.71% | 4.0 | 8.333 |
| adult | RF | Max | 82.84% | 798.3 | 0.009 |
| adult | RF | RuleFit | 86.03% | 276.3 | 6.928 |
| adult | RF | Vote | 80.67% | 798.3 | 0.009 |
| bank-marketing | LORD | CBA | 88.42% | 4466.3 | 65.288 |
| bank-marketing | LORD | CMAR | 87.22% | 2984.0 | 3.517 |
| bank-marketing | LORD | IDS-greedy | 88.33% | 9.0 | 37.960 |
| bank-marketing | LORD | Max | 89.12% | 8823.0 | 0.009 |
| bank-marketing | LORD | RuleFit | 89.06% | 2283.0 | 34.003 |
| bank-marketing | LORD | Vote | 88.77% | 8823.0 | 0.009 |
| bank-marketing | RF | CBA | 88.85% | 59.0 | 0.939 |
| bank-marketing | RF | CMAR | 89.36% | 48.7 | 0.450 |
| bank-marketing | RF | IDS-greedy | 89.34% | 4.0 | 7.260 |
| bank-marketing | RF | Max | 88.37% | 799.3 | 0.008 |
| bank-marketing | RF | RuleFit | 89.87% | 302.0 | 6.868 |
| bank-marketing | RF | Vote | 88.57% | 799.3 | 0.008 |
| credit-g | LORD | CBA | 72.30% | 182.7 | 0.102 |
| credit-g | LORD | CMAR | 70.90% | 140.3 | 0.023 |
| credit-g | LORD | IDS-greedy | 70.60% | 13.3 | 36.233 |
| credit-g | LORD | Max | 73.60% | 341.3 | 0.000 |
| credit-g | LORD | RuleFit | 71.10% | 123.7 | 0.105 |
| credit-g | LORD | Vote | 74.00% | 341.3 | 0.000 |
| credit-g | RF | CBA | 74.80% | 84.3 | 0.071 |
| credit-g | RF | CMAR | 73.80% | 32.0 | 0.025 |
| credit-g | RF | IDS-greedy | 70.40% | 5.7 | 4.764 |
| credit-g | RF | Max | 72.90% | 781.3 | 0.000 |
| credit-g | RF | RuleFit | 72.20% | 148.7 | 0.172 |
| credit-g | RF | Vote | 70.80% | 781.3 | 0.000 |
| kr-vs-kp | LORD | CBA | 98.87% | 43.0 | 0.061 |
| kr-vs-kp | LORD | CMAR | 98.91% | 44.3 | 0.016 |
| kr-vs-kp | LORD | IDS-greedy | 97.47% | 14.3 | 25.520 |
| kr-vs-kp | LORD | Max | 98.97% | 121.3 | 0.001 |
| kr-vs-kp | LORD | RuleFit | 99.16% | 35.3 | 0.074 |
| kr-vs-kp | LORD | Vote | 98.25% | 121.3 | 0.000 |
| kr-vs-kp | RF | CBA | 95.56% | 20.7 | 0.057 |
| kr-vs-kp | RF | CMAR | 91.93% | 12.3 | 0.073 |
| kr-vs-kp | RF | IDS-greedy | 82.95% | 4.0 | 2.310 |
| kr-vs-kp | RF | Max | 94.56% | 688.3 | 0.001 |
| kr-vs-kp | RF | RuleFit | 97.93% | 55.3 | 0.140 |
| kr-vs-kp | RF | Vote | 90.68% | 688.3 | 0.001 |
| mushroom | LORD | CBA | 100.00% | 16.7 | 0.052 |
| mushroom | LORD | CMAR | 99.95% | 19.7 | 0.013 |
| mushroom | LORD | IDS-greedy | 99.58% | 12.0 | 5.918 |
| mushroom | LORD | Max | 99.96% | 34.3 | 0.002 |
| mushroom | LORD | RuleFit | 99.96% | 16.7 | 0.075 |
| mushroom | LORD | Vote | 99.96% | 34.3 | 0.001 |
| mushroom | RF | CBA | 100.00% | 14.0 | 0.100 |
| mushroom | RF | CMAR | 95.49% | 8.0 | 0.125 |
| mushroom | RF | IDS-greedy | 96.45% | 5.7 | 5.308 |
| mushroom | RF | Max | 99.48% | 644.7 | 0.001 |
| mushroom | RF | RuleFit | 100.00% | 25.7 | 0.184 |
| mushroom | RF | Vote | 94.14% | 644.7 | 0.001 |
| sick | LORD | CBA | 97.38% | 91.3 | 0.135 |
| sick | LORD | CMAR | 85.68% | 55.3 | 0.017 |
| sick | LORD | IDS-greedy | 95.55% | 12.3 | 20.978 |
| sick | LORD | Max | 96.21% | 144.3 | 0.001 |
| sick | LORD | RuleFit | 97.69% | 57.0 | 0.146 |
| sick | LORD | Vote | 95.86% | 144.3 | 0.001 |
| sick | RF | CBA | 97.69% | 49.7 | 0.115 |
| sick | RF | CMAR | 96.63% | 24.3 | 0.043 |
| sick | RF | IDS-greedy | 97.77% | 8.7 | 8.665 |
| sick | RF | Max | 94.06% | 760.0 | 0.001 |
| sick | RF | RuleFit | 97.99% | 65.0 | 0.160 |
| sick | RF | Vote | 95.17% | 760.0 | 0.001 |
| spambase | LORD | CBA | 93.78% | 314.0 | 0.538 |
| spambase | LORD | CMAR | 91.37% | 224.3 | 0.084 |
| spambase | LORD | IDS-greedy | 80.11% | 11.3 | 23.121 |
| spambase | LORD | Max | 93.96% | 611.0 | 0.001 |
| spambase | LORD | RuleFit | 93.78% | 180.0 | 0.321 |
| spambase | LORD | Vote | 94.04% | 611.0 | 0.001 |
| spambase | RF | CBA | 91.98% | 86.7 | 0.167 |
| spambase | RF | CMAR | 90.35% | 40.0 | 0.088 |
| spambase | RF | IDS-greedy | 87.29% | 4.0 | 2.964 |
| spambase | RF | Max | 90.96% | 787.0 | 0.001 |
| spambase | RF | RuleFit | 93.70% | 152.3 | 0.486 |
| spambase | RF | Vote | 91.26% | 787.0 | 0.001 |

