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
(vertical bitmaps (`VerticalBitmapRepresentation`)). Discretization: `max_intervals=3`. Distiller fits are
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
| CAR | 1.22 | 12160.2 | 83.16% | 0 |
| LORD | 0.51 | 92.6 | 86.03% | 0 |
| RF | 0.51 | 739.8 | 82.64% | 0 |

### Pool x distiller grid

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| CAR | RuleFit | 87.78% | 57.8 | 0.32 | 0 |
| CAR | CBA | 88.46% | 27.2 | 0.45 | 0 |
| CAR | CMAR | 82.10% | 23.9 | 0.59 | 0 |
| CAR | IDS | 78.73% | 23.1 | 59.17 | 5 |
| CAR | IDS-greedy | 81.01% | 8.0 | 10.01 | 0 |
| CAR | Max | 83.85% | 12159.9 | 0.00 | 0 |
| CAR | Vote | 79.33% | 12159.9 | 0.00 | 0 |
| LORD | RuleFit | 87.48% | 30.6 | 0.04 | 0 |
| LORD | CBA | 86.86% | 39.5 | 0.02 | 0 |
| LORD | CMAR | 85.83% | 35.0 | 0.01 | 0 |
| LORD | IDS | 80.43% | 8.6 | 106.98 | 5 |
| LORD | IDS-greedy | 85.28% | 10.0 | 15.48 | 0 |
| LORD | Max | 87.46% | 92.1 | 0.00 | 0 |
| LORD | Vote | 87.72% | 92.1 | 0.00 | 0 |
| RF | RuleFit | 87.85% | 44.4 | 0.05 | 0 |
| RF | CBA | 84.19% | 34.8 | 0.04 | 0 |
| RF | CMAR | 82.02% | 29.0 | 0.03 | 0 |
| RF | IDS | 69.24% | 21.7 | 84.00 | 10 |
| RF | IDS-greedy | 80.25% | 8.5 | 10.47 | 0 |
| RF | Max | 85.08% | 739.8 | 0.00 | 0 |
| RF | Vote | 84.46% | 739.8 | 0.00 | 0 |

### Mean rank of the distillers, by pool

| distiller | CAR | LORD | RF | overall |
|---|--:|--:|--:|--:|
| RuleFit | 2.60 | 2.60 | 2.10 | 2.43 |
| CBA | 1.90 | 3.40 | 3.50 | 2.93 |
| Vote | 5.10 | 2.10 | 2.20 | 3.13 |
| Max | 3.60 | 2.90 | 3.40 | 3.30 |
| IDS-greedy | 4.40 | 5.30 | 4.60 | 4.77 |
| CMAR | 4.40 | 5.00 | 5.40 | 4.93 |
| IDS | 6.00 | 6.70 | 6.80 | 6.50 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| SPECT | CAR | CBA | 80.92% | 33.6 | 0.507 |
| SPECT | CAR | CMAR | 79.03% | 47.2 | 0.264 |
| SPECT | CAR | IDS | 62.93% | 21.2 | 51.000 |
| SPECT | CAR | IDS-greedy | 70.42% | 8.2 | 7.965 |
| SPECT | CAR | Max | 79.40% | 9483.4 | 0.003 |
| SPECT | CAR | RuleFit | 82.02% | 48.0 | 0.186 |
| SPECT | CAR | Vote | 79.40% | 9483.4 | 0.003 |
| SPECT | LORD | CBA | 82.42% | 43.2 | 0.016 |
| SPECT | LORD | CMAR | 79.02% | 25.0 | 0.005 |
| SPECT | LORD | IDS | 79.40% | 6.6 | 132.914 |
| SPECT | LORD | IDS-greedy | 79.40% | 5.8 | 4.362 |
| SPECT | LORD | Max | 82.37% | 70.4 | 0.000 |
| SPECT | LORD | RuleFit | 83.88% | 31.4 | 0.029 |
| SPECT | LORD | Vote | 83.14% | 70.4 | 0.000 |
| SPECT | RF | CBA | 81.66% | 32.6 | 0.033 |
| SPECT | RF | CMAR | 79.03% | 40.2 | 0.012 |
| SPECT | RF | IDS | 56.17% | 27.8 | 56.691 |
| SPECT | RF | IDS-greedy | 66.32% | 8.6 | 8.669 |
| SPECT | RF | Max | 79.40% | 692.2 | 0.000 |
| SPECT | RF | RuleFit | 83.13% | 32.6 | 0.046 |
| SPECT | RF | Vote | 80.52% | 692.2 | 0.000 |
| heart-statlog | CAR | CBA | 84.07% | 40.2 | 0.179 |
| heart-statlog | CAR | CMAR | 81.48% | 23.4 | 1.330 |
| heart-statlog | CAR | IDS | 79.26% | 17.8 | 47.498 |
| heart-statlog | CAR | IDS-greedy | 78.52% | 8.2 | 8.738 |
| heart-statlog | CAR | Max | 81.85% | 22992.2 | 0.007 |
| heart-statlog | CAR | RuleFit | 80.37% | 115.4 | 0.524 |
| heart-statlog | CAR | Vote | 83.33% | 22992.2 | 0.006 |
| heart-statlog | LORD | CBA | 80.37% | 48.4 | 0.018 |
| heart-statlog | LORD | CMAR | 80.00% | 42.6 | 0.006 |
| heart-statlog | LORD | IDS | 75.56% | 9.4 | 130.758 |
| heart-statlog | LORD | IDS-greedy | 78.89% | 13.4 | 31.324 |
| heart-statlog | LORD | Max | 81.11% | 85.0 | 0.000 |
| heart-statlog | LORD | RuleFit | 80.37% | 35.6 | 0.010 |
| heart-statlog | LORD | Vote | 81.85% | 85.0 | 0.000 |
| heart-statlog | RF | CBA | 78.89% | 44.0 | 0.029 |
| heart-statlog | RF | CMAR | 80.37% | 25.2 | 0.028 |
| heart-statlog | RF | IDS | 75.56% | 16.8 | 116.188 |
| heart-statlog | RF | IDS-greedy | 76.30% | 7.6 | 10.895 |
| heart-statlog | RF | Max | 81.85% | 783.8 | 0.000 |
| heart-statlog | RF | RuleFit | 80.74% | 52.8 | 0.048 |
| heart-statlog | RF | Vote | 82.96% | 783.8 | 0.000 |
| hepatitis | CAR | CBA | 82.58% | 19.8 | 0.470 |
| hepatitis | CAR | CMAR | 83.87% | 15.2 | 0.425 |
| hepatitis | CAR | IDS | n/a | n/a | 90.613 |
| hepatitis | CAR | IDS-greedy | 81.29% | 6.6 | 5.940 |
| hepatitis | CAR | Max | 80.65% | 8826.8 | 0.003 |
| hepatitis | CAR | RuleFit | 80.65% | 45.0 | 0.272 |
| hepatitis | CAR | Vote | 79.35% | 8826.8 | 0.004 |
| hepatitis | LORD | CBA | 81.29% | 24.2 | 0.009 |
| hepatitis | LORD | CMAR | 80.00% | 18.2 | 0.004 |
| hepatitis | LORD | IDS | 78.71% | 10.0 | 9.922 |
| hepatitis | LORD | IDS-greedy | 81.94% | 10.8 | 5.637 |
| hepatitis | LORD | Max | 83.87% | 37.8 | 0.000 |
| hepatitis | LORD | RuleFit | 80.65% | 18.0 | 0.007 |
| hepatitis | LORD | Vote | 83.87% | 37.8 | 0.000 |
| hepatitis | RF | CBA | 77.42% | 22.4 | 0.026 |
| hepatitis | RF | CMAR | 80.65% | 17.0 | 0.019 |
| hepatitis | RF | IDS | n/a | n/a | 136.914 |
| hepatitis | RF | IDS-greedy | 81.94% | 12.4 | 17.940 |
| hepatitis | RF | Max | 81.94% | 682.2 | 0.000 |
| hepatitis | RF | RuleFit | 81.94% | 27.4 | 0.018 |
| hepatitis | RF | Vote | 85.81% | 682.2 | 0.000 |
| tic-tac-toe | CAR | CBA | 100.00% | 10.0 | 0.843 |
| tic-tac-toe | CAR | CMAR | 73.69% | 23.4 | 0.704 |
| tic-tac-toe | CAR | IDS | 78.71% | 25.8 | 81.564 |
| tic-tac-toe | CAR | IDS-greedy | 79.64% | 12.4 | 23.803 |
| tic-tac-toe | CAR | Max | 82.88% | 16517.2 | 0.005 |
| tic-tac-toe | CAR | RuleFit | 100.00% | 47.6 | 0.524 |
| tic-tac-toe | CAR | Vote | 65.34% | 16517.2 | 0.006 |
| tic-tac-toe | LORD | CBA | 96.66% | 47.6 | 0.058 |
| tic-tac-toe | LORD | CMAR | 97.49% | 60.6 | 0.025 |
| tic-tac-toe | LORD | IDS | n/a | n/a | 248.126 |
| tic-tac-toe | LORD | IDS-greedy | 94.67% | 13.6 | 33.535 |
| tic-tac-toe | LORD | Max | 95.72% | 226.2 | 0.001 |
| tic-tac-toe | LORD | RuleFit | 98.02% | 43.8 | 0.043 |
| tic-tac-toe | LORD | Vote | 94.78% | 226.2 | 0.000 |
| tic-tac-toe | RF | CBA | 87.37% | 45.8 | 0.067 |
| tic-tac-toe | RF | CMAR | 75.37% | 42.0 | 0.046 |
| tic-tac-toe | RF | IDS | 75.99% | 20.6 | 38.010 |
| tic-tac-toe | RF | IDS-greedy | 81.52% | 8.4 | 10.338 |
| tic-tac-toe | RF | Max | 87.26% | 800.0 | 0.000 |
| tic-tac-toe | RF | RuleFit | 97.81% | 81.4 | 0.076 |
| tic-tac-toe | RF | Vote | 77.35% | 800.0 | 0.000 |
| vote | CAR | CBA | 94.71% | 32.4 | 0.227 |
| vote | CAR | CMAR | 92.41% | 10.4 | 0.241 |
| vote | CAR | IDS | 94.02% | 27.8 | 25.183 |
| vote | CAR | IDS-greedy | 95.17% | 4.4 | 3.628 |
| vote | CAR | Max | 94.48% | 2980.0 | 0.001 |
| vote | CAR | RuleFit | 95.86% | 32.8 | 0.111 |
| vote | CAR | Vote | 89.20% | 2980.0 | 0.002 |
| vote | LORD | CBA | 93.56% | 34.0 | 0.018 |
| vote | LORD | CMAR | 92.64% | 28.6 | 0.006 |
| vote | LORD | IDS | 88.05% | 8.2 | 13.200 |
| vote | LORD | IDS-greedy | 91.49% | 6.6 | 2.538 |
| vote | LORD | Max | 94.25% | 41.0 | 0.000 |
| vote | LORD | RuleFit | 94.48% | 24.2 | 0.102 |
| vote | LORD | Vote | 94.94% | 41.0 | 0.000 |
| vote | RF | CBA | 95.63% | 29.4 | 0.033 |
| vote | RF | CMAR | 94.71% | 20.8 | 0.028 |
| vote | RF | IDS | n/a | n/a | 72.181 |
| vote | RF | IDS-greedy | 95.17% | 5.6 | 4.532 |
| vote | RF | Max | 94.94% | 740.8 | 0.001 |
| vote | RF | RuleFit | 95.63% | 27.8 | 0.049 |
| vote | RF | Vote | 95.63% | 740.8 | 0.000 |

## Why IDS is slow and unstable

`IDS` (`optimizer="sls"`, Smooth Local Search, the paper-faithful default) and `IDS-greedy` (`optimizer="greedy"`, a cheap deterministic fallback) both compress the *same* pool, so any difference here is the optimizer, not the data. On Study 1's small datasets:

| pool | optimizer | mean accuracy | mean fit time (s) |
|---|---|--:|--:|
| CAR | IDS | 78.73% | 59.17 |
| CAR | IDS-greedy | 81.01% | 10.01 |
| LORD | IDS | 80.43% | 106.98 |
| LORD | IDS-greedy | 85.28% | 15.48 |
| RF | IDS | 69.24% | 84.00 |
| RF | IDS-greedy | 80.25% | 10.47 |

Plain `IDS`'s fit time isn't a steady cost that happens to be high -- it varies enormously by (dataset, pool), not just by pool: slowest measured here is `tic-tac-toe`/`LORD` at 248.1s mean, fastest is `hepatitis`/`LORD` at 9.92s -- a 25x spread between datasets this demo treats as comparably small. This is consistent with Smooth Local Search's randomized-restart search having no convergence guarantee on a non-monotone objective, unlike `IDS-greedy`'s single deterministic pass -- not a cost that scales predictably with the data, but per-instance variance. A second, independent symptom of the same instability: on `kr-vs-kp` (Study 2's pool generator, `SKLRandomForest`, tested there before plain `IDS` was dropped from that study), one fold's accuracy came in at 52.30% (near chance) against 84.00-89.70% on the other two folds of the same dataset and pool. Both the speed and the accuracy can apparently collapse independently of data size -- why plain `IDS` stays out of Study 2 (see Setup) while `IDS-greedy`, deterministic, stays in.

## Study 2: larger datasets, RandomForest and LORD pools

### Pool stats (mean across datasets and folds)

| pool | build time (s) | n_rules | mean accuracy (all distillers) | pool-build failures |
|---|--:|--:|--:|--:|
| RF | 6.12 | 751.2 | 88.77% | 0 |
| LORD | 157.25 | 2773.3 | 89.66% | 0 |

### Distiller results

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| RF | RuleFit | 91.14% | 146.6 | 2.64 | 0 |
| RF | CBA | 90.04% | 61.8 | 0.58 | 0 |
| RF | CMAR | 88.76% | 29.2 | 0.16 | 0 |
| RF | IDS-greedy | 86.49% | 5.2 | 7.30 | 0 |
| RF | Max | 89.00% | 751.2 | 0.00 | 0 |
| RF | Vote | 87.20% | 751.2 | 0.00 | 0 |
| LORD | RuleFit | 90.80% | 703.5 | 12.60 | 0 |
| LORD | CBA | 90.45% | 1375.0 | 23.83 | 0 |
| LORD | CMAR | 88.14% | 663.2 | 1.19 | 0 |
| LORD | IDS-greedy | 87.07% | 11.1 | 30.42 | 0 |
| LORD | Max | 90.71% | 2773.3 | 0.00 | 0 |
| LORD | Vote | 90.77% | 2773.3 | 0.00 | 0 |

### Mean rank of the distillers

| distiller | RF | LORD | overall |
|---|--:|--:|--:|
| RuleFit | 1.36 | 2.29 | 1.82 |
| CBA | 2.64 | 3.14 | 2.89 |
| Max | 4.00 | 2.29 | 3.14 |
| Vote | 5.00 | 2.71 | 3.86 |
| CMAR | 3.43 | 4.86 | 4.14 |
| IDS-greedy | 4.57 | 5.71 | 5.14 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| adult | LORD | CBA | 83.01% | 4510.7 | 85.018 |
| adult | LORD | CMAR | 82.47% | 1172.3 | 4.111 |
| adult | LORD | IDS-greedy | 78.37% | 5.3 | 20.993 |
| adult | LORD | Max | 83.37% | 9338.3 | 0.013 |
| adult | LORD | RuleFit | 85.08% | 2224.7 | 47.132 |
| adult | LORD | Vote | 84.20% | 9338.3 | 0.017 |
| adult | RF | CBA | 84.27% | 115.0 | 2.626 |
| adult | RF | CMAR | 83.74% | 40.3 | 0.436 |
| adult | RF | IDS-greedy | 79.71% | 4.0 | 9.512 |
| adult | RF | Max | 82.84% | 798.3 | 0.015 |
| adult | RF | RuleFit | 86.03% | 276.3 | 7.762 |
| adult | RF | Vote | 80.67% | 798.3 | 0.012 |
| bank-marketing | LORD | CBA | 88.42% | 4466.3 | 80.519 |
| bank-marketing | LORD | CMAR | 87.22% | 2984.0 | 4.044 |
| bank-marketing | LORD | IDS-greedy | 88.33% | 9.0 | 42.875 |
| bank-marketing | LORD | Max | 89.12% | 8823.0 | 0.013 |
| bank-marketing | LORD | RuleFit | 89.06% | 2283.0 | 40.072 |
| bank-marketing | LORD | Vote | 88.77% | 8823.0 | 0.011 |
| bank-marketing | RF | CBA | 88.85% | 59.0 | 0.868 |
| bank-marketing | RF | CMAR | 89.36% | 48.7 | 0.389 |
| bank-marketing | RF | IDS-greedy | 89.34% | 4.0 | 8.405 |
| bank-marketing | RF | Max | 88.37% | 799.3 | 0.010 |
| bank-marketing | RF | RuleFit | 89.87% | 302.0 | 9.282 |
| bank-marketing | RF | Vote | 88.57% | 799.3 | 0.011 |
| credit-g | LORD | CBA | 71.70% | 182.7 | 0.150 |
| credit-g | LORD | CMAR | 71.40% | 142.3 | 0.025 |
| credit-g | LORD | IDS-greedy | 70.10% | 13.7 | 48.751 |
| credit-g | LORD | Max | 73.40% | 340.7 | 0.000 |
| credit-g | LORD | RuleFit | 70.90% | 128.0 | 0.146 |
| credit-g | LORD | Vote | 74.30% | 340.7 | 0.000 |
| credit-g | RF | CBA | 71.90% | 87.3 | 0.074 |
| credit-g | RF | CMAR | 73.80% | 31.0 | 0.030 |
| credit-g | RF | IDS-greedy | 71.90% | 6.3 | 7.526 |
| credit-g | RF | Max | 72.70% | 780.7 | 0.000 |
| credit-g | RF | RuleFit | 72.50% | 149.7 | 0.251 |
| credit-g | RF | Vote | 69.90% | 780.7 | 0.001 |
| kr-vs-kp | LORD | CBA | 98.87% | 43.0 | 0.094 |
| kr-vs-kp | LORD | CMAR | 98.91% | 44.3 | 0.018 |
| kr-vs-kp | LORD | IDS-greedy | 97.47% | 14.3 | 31.486 |
| kr-vs-kp | LORD | Max | 98.97% | 121.3 | 0.001 |
| kr-vs-kp | LORD | RuleFit | 99.16% | 35.3 | 0.115 |
| kr-vs-kp | LORD | Vote | 98.25% | 121.3 | 0.001 |
| kr-vs-kp | RF | CBA | 95.56% | 20.7 | 0.064 |
| kr-vs-kp | RF | CMAR | 91.93% | 12.3 | 0.062 |
| kr-vs-kp | RF | IDS-greedy | 82.95% | 4.0 | 3.882 |
| kr-vs-kp | RF | Max | 94.56% | 688.3 | 0.001 |
| kr-vs-kp | RF | RuleFit | 97.93% | 55.3 | 0.151 |
| kr-vs-kp | RF | Vote | 90.68% | 688.3 | 0.001 |
| mushroom | LORD | CBA | 100.00% | 16.7 | 0.093 |
| mushroom | LORD | CMAR | 99.95% | 19.7 | 0.025 |
| mushroom | LORD | IDS-greedy | 99.58% | 12.0 | 8.598 |
| mushroom | LORD | Max | 99.96% | 34.3 | 0.002 |
| mushroom | LORD | RuleFit | 99.96% | 16.7 | 0.119 |
| mushroom | LORD | Vote | 99.96% | 34.3 | 0.003 |
| mushroom | RF | CBA | 100.00% | 14.0 | 0.106 |
| mushroom | RF | CMAR | 95.49% | 8.0 | 0.117 |
| mushroom | RF | IDS-greedy | 96.45% | 5.7 | 7.277 |
| mushroom | RF | Max | 99.48% | 644.7 | 0.003 |
| mushroom | RF | RuleFit | 100.00% | 25.7 | 0.221 |
| mushroom | RF | Vote | 94.14% | 644.7 | 0.003 |
| sick | LORD | CBA | 97.38% | 91.3 | 0.165 |
| sick | LORD | CMAR | 85.68% | 55.3 | 0.013 |
| sick | LORD | IDS-greedy | 95.55% | 12.3 | 28.565 |
| sick | LORD | Max | 96.21% | 144.3 | 0.001 |
| sick | LORD | RuleFit | 97.69% | 57.0 | 0.133 |
| sick | LORD | Vote | 95.86% | 144.3 | 0.001 |
| sick | RF | CBA | 97.69% | 49.7 | 0.114 |
| sick | RF | CMAR | 96.63% | 24.3 | 0.050 |
| sick | RF | IDS-greedy | 97.77% | 8.7 | 11.100 |
| sick | RF | Max | 94.06% | 760.0 | 0.001 |
| sick | RF | RuleFit | 97.99% | 65.0 | 0.214 |
| sick | RF | Vote | 95.17% | 760.0 | 0.001 |
| spambase | LORD | CBA | 93.78% | 314.0 | 0.742 |
| spambase | LORD | CMAR | 91.37% | 224.3 | 0.081 |
| spambase | LORD | IDS-greedy | 80.11% | 11.3 | 31.643 |
| spambase | LORD | Max | 93.96% | 611.0 | 0.001 |
| spambase | LORD | RuleFit | 93.78% | 180.0 | 0.453 |
| spambase | LORD | Vote | 94.04% | 611.0 | 0.001 |
| spambase | RF | CBA | 91.98% | 86.7 | 0.192 |
| spambase | RF | CMAR | 90.35% | 40.0 | 0.057 |
| spambase | RF | IDS-greedy | 87.29% | 4.0 | 3.377 |
| spambase | RF | Max | 90.96% | 787.0 | 0.001 |
| spambase | RF | RuleFit | 93.70% | 152.3 | 0.578 |
| spambase | RF | Vote | 91.26% | 787.0 | 0.001 |

