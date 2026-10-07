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
cheap, deterministic fallback). `Max`/`Vote` are the same `FlatRuleSet`
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
  distillers: plain `IDS` (`optimizer="sls"`) is dropped here -- on
  Study 1's own small datasets it was already the slowest distiller by
  a wide margin (its cost doesn't track data size in any simple way --
  see "Why IDS is slow and unstable" below), so repeating it on bigger
  data wasn't worth the time. `IDS-greedy` stays.

Discretization: `max_intervals=3`. Distiller fits are
capped at 300s; pool building gets a longer cap of
1800s, since `pylord_candidates`' cost grows
super-linearly with row count and can need much longer than any
distiller fit on the biggest Study 2 datasets. A time-out or an error
counts as a failure, not as the end of the run.

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
| CAR | 0.31 | 11035.5 | 0.830 | 0 |
| LORD | 0.83 | 92.6 | 0.857 | 0 |
| RF | 0.23 | 740.0 | 0.827 | 0 |

### Pool x distiller grid

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| CAR | RuleFit | 0.876 | 54.2 | 0.89 | 0 |
| CAR | CBA | 0.884 | 27.3 | 0.33 | 0 |
| CAR | CMAR | 0.819 | 24.0 | 0.31 | 0 |
| CAR | IDS | 0.794 | 22.2 | 28.36 | 0 |
| CAR | IDS-greedy | 0.809 | 8.1 | 4.82 | 0 |
| CAR | Max | 0.838 | 11035.5 | 0.48 | 0 |
| CAR | Vote | 0.790 | 11035.5 | 0.54 | 0 |
| LORD | RuleFit | 0.872 | 30.4 | 0.02 | 0 |
| LORD | CBA | 0.864 | 39.3 | 0.02 | 0 |
| LORD | CMAR | 0.858 | 34.4 | 0.01 | 0 |
| LORD | IDS | 0.814 | 8.7 | 44.91 | 0 |
| LORD | IDS-greedy | 0.854 | 10.7 | 7.13 | 0 |
| LORD | Max | 0.869 | 92.6 | 0.01 | 0 |
| LORD | Vote | 0.868 | 92.6 | 0.01 | 0 |
| RF | RuleFit | 0.877 | 44.1 | 0.07 | 0 |
| RF | CBA | 0.839 | 34.8 | 0.08 | 0 |
| RF | CMAR | 0.817 | 29.4 | 0.07 | 0 |
| RF | IDS | 0.767 | 20.0 | 47.43 | 0 |
| RF | IDS-greedy | 0.809 | 8.6 | 4.92 | 0 |
| RF | Max | 0.842 | 740.0 | 0.05 | 0 |
| RF | Vote | 0.839 | 740.0 | 0.03 | 0 |

### Mean rank of the distillers, by pool

| distiller | CAR | LORD | RF | overall |
|---|--:|--:|--:|--:|
| RuleFit | 2.80 | 3.00 | 2.30 | 2.70 |
| CBA | 1.90 | 3.80 | 3.90 | 3.20 |
| Vote | 5.60 | 2.40 | 2.40 | 3.47 |
| Max | 3.90 | 3.00 | 3.80 | 3.57 |
| IDS-greedy | 4.30 | 4.70 | 4.20 | 4.40 |
| CMAR | 4.00 | 4.90 | 5.00 | 4.63 |
| IDS | 5.50 | 6.20 | 6.40 | 6.03 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| SPECT | CAR | CBA | 0.809 | 33.600 | 0.360 |
| SPECT | CAR | CMAR | 0.790 | 47.200 | 0.150 |
| SPECT | CAR | IDS | 0.678 | 19.600 | 23.456 |
| SPECT | CAR | IDS-greedy | 0.704 | 8.200 | 3.899 |
| SPECT | CAR | Max | 0.794 | 9483.400 | 0.422 |
| SPECT | CAR | RuleFit | 0.820 | 47.800 | 0.652 |
| SPECT | CAR | Vote | 0.794 | 9483.400 | 0.416 |
| SPECT | LORD | CBA | 0.824 | 43.200 | 0.012 |
| SPECT | LORD | CMAR | 0.790 | 25.000 | 0.005 |
| SPECT | LORD | IDS | 0.794 | 6.600 | 59.632 |
| SPECT | LORD | IDS-greedy | 0.794 | 5.800 | 2.221 |
| SPECT | LORD | Max | 0.824 | 70.400 | 0.006 |
| SPECT | LORD | RuleFit | 0.843 | 31.600 | 0.013 |
| SPECT | LORD | Vote | 0.831 | 70.400 | 0.001 |
| SPECT | RF | CBA | 0.817 | 32.600 | 0.067 |
| SPECT | RF | CMAR | 0.790 | 40.200 | 0.061 |
| SPECT | RF | IDS | 0.569 | 24.800 | 41.078 |
| SPECT | RF | IDS-greedy | 0.663 | 7.800 | 3.676 |
| SPECT | RF | Max | 0.794 | 692.200 | 0.032 |
| SPECT | RF | RuleFit | 0.831 | 32.600 | 0.083 |
| SPECT | RF | Vote | 0.805 | 692.200 | 0.027 |
| heart-statlog | CAR | CBA | 0.837 | 40.800 | 0.088 |
| heart-statlog | CAR | CMAR | 0.819 | 23.000 | 0.670 |
| heart-statlog | CAR | IDS | 0.778 | 18.200 | 22.495 |
| heart-statlog | CAR | IDS-greedy | 0.785 | 8.400 | 4.221 |
| heart-statlog | CAR | Max | 0.815 | 17368.400 | 0.751 |
| heart-statlog | CAR | RuleFit | 0.796 | 95.400 | 1.310 |
| heart-statlog | CAR | Vote | 0.815 | 17368.400 | 0.934 |
| heart-statlog | LORD | CBA | 0.800 | 48.200 | 0.016 |
| heart-statlog | LORD | CMAR | 0.800 | 42.200 | 0.011 |
| heart-statlog | LORD | IDS | 0.730 | 8.200 | 59.571 |
| heart-statlog | LORD | IDS-greedy | 0.781 | 13.600 | 13.028 |
| heart-statlog | LORD | Max | 0.811 | 85.000 | 0.004 |
| heart-statlog | LORD | RuleFit | 0.789 | 35.200 | 0.002 |
| heart-statlog | LORD | Vote | 0.811 | 85.000 | 0.016 |
| heart-statlog | RF | CBA | 0.796 | 44.000 | 0.046 |
| heart-statlog | RF | CMAR | 0.804 | 23.400 | 0.044 |
| heart-statlog | RF | IDS | 0.741 | 13.000 | 91.327 |
| heart-statlog | RF | IDS-greedy | 0.800 | 9.400 | 5.535 |
| heart-statlog | RF | Max | 0.800 | 784.600 | 0.089 |
| heart-statlog | RF | RuleFit | 0.800 | 49.000 | 0.058 |
| heart-statlog | RF | Vote | 0.815 | 784.600 | 0.029 |
| hepatitis | CAR | CBA | 0.826 | 19.800 | 0.318 |
| hepatitis | CAR | CMAR | 0.839 | 15.200 | 0.197 |
| hepatitis | CAR | IDS | 0.813 | 22.200 | 11.742 |
| hepatitis | CAR | IDS-greedy | 0.813 | 6.200 | 2.299 |
| hepatitis | CAR | Max | 0.806 | 8826.800 | 0.316 |
| hepatitis | CAR | RuleFit | 0.806 | 44.800 | 0.589 |
| hepatitis | CAR | Vote | 0.794 | 8826.800 | 0.394 |
| hepatitis | LORD | CBA | 0.813 | 24.200 | 0.007 |
| hepatitis | LORD | CMAR | 0.800 | 18.200 | 0.006 |
| hepatitis | LORD | IDS | 0.813 | 10.400 | 4.838 |
| hepatitis | LORD | IDS-greedy | 0.819 | 11.800 | 2.854 |
| hepatitis | LORD | Max | 0.839 | 37.800 | 0.004 |
| hepatitis | LORD | RuleFit | 0.806 | 18.000 | 0.007 |
| hepatitis | LORD | Vote | 0.839 | 37.800 | 0.001 |
| hepatitis | RF | CBA | 0.774 | 22.400 | 0.113 |
| hepatitis | RF | CMAR | 0.806 | 17.000 | 0.031 |
| hepatitis | RF | IDS | 0.794 | 9.600 | 51.955 |
| hepatitis | RF | IDS-greedy | 0.819 | 11.000 | 6.796 |
| hepatitis | RF | Max | 0.819 | 682.200 | 0.023 |
| hepatitis | RF | RuleFit | 0.819 | 27.400 | 0.050 |
| hepatitis | RF | Vote | 0.858 | 682.200 | 0.028 |
| tic-tac-toe | CAR | CBA | 1.000 | 10.000 | 0.699 |
| tic-tac-toe | CAR | CMAR | 0.723 | 24.400 | 0.318 |
| tic-tac-toe | CAR | IDS | 0.762 | 28.800 | 65.474 |
| tic-tac-toe | CAR | IDS-greedy | 0.792 | 13.000 | 11.696 |
| tic-tac-toe | CAR | Max | 0.829 | 16518.800 | 0.779 |
| tic-tac-toe | CAR | RuleFit | 1.000 | 50.800 | 1.647 |
| tic-tac-toe | CAR | Vote | 0.653 | 16518.800 | 0.836 |
| tic-tac-toe | LORD | CBA | 0.946 | 46.800 | 0.030 |
| tic-tac-toe | LORD | CMAR | 0.973 | 57.800 | 0.020 |
| tic-tac-toe | LORD | IDS | 0.855 | 11.800 | 95.084 |
| tic-tac-toe | LORD | IDS-greedy | 0.959 | 14.400 | 15.936 |
| tic-tac-toe | LORD | Max | 0.928 | 228.600 | 0.009 |
| tic-tac-toe | LORD | RuleFit | 0.978 | 43.200 | 0.031 |
| tic-tac-toe | LORD | Vote | 0.911 | 228.600 | 0.007 |
| tic-tac-toe | RF | CBA | 0.853 | 45.600 | 0.094 |
| tic-tac-toe | RF | CMAR | 0.737 | 45.400 | 0.143 |
| tic-tac-toe | RF | IDS | 0.794 | 26.400 | 15.828 |
| tic-tac-toe | RF | IDS-greedy | 0.811 | 9.400 | 6.286 |
| tic-tac-toe | RF | Max | 0.847 | 800.000 | 0.039 |
| tic-tac-toe | RF | RuleFit | 0.980 | 83.400 | 0.097 |
| tic-tac-toe | RF | Vote | 0.759 | 800.000 | 0.035 |
| vote | CAR | CBA | 0.947 | 32.400 | 0.169 |
| vote | CAR | CMAR | 0.924 | 10.400 | 0.203 |
| vote | CAR | IDS | 0.940 | 22.400 | 18.625 |
| vote | CAR | IDS-greedy | 0.952 | 4.800 | 1.964 |
| vote | CAR | Max | 0.945 | 2980.000 | 0.108 |
| vote | CAR | RuleFit | 0.959 | 32.000 | 0.268 |
| vote | CAR | Vote | 0.892 | 2980.000 | 0.125 |
| vote | LORD | CBA | 0.936 | 34.000 | 0.012 |
| vote | LORD | CMAR | 0.926 | 28.600 | 0.006 |
| vote | LORD | IDS | 0.880 | 6.600 | 5.414 |
| vote | LORD | IDS-greedy | 0.917 | 7.800 | 1.625 |
| vote | LORD | Max | 0.943 | 41.000 | 0.003 |
| vote | LORD | RuleFit | 0.945 | 24.200 | 0.037 |
| vote | LORD | Vote | 0.949 | 41.000 | 0.002 |
| vote | RF | CBA | 0.956 | 29.400 | 0.104 |
| vote | RF | CMAR | 0.947 | 20.800 | 0.066 |
| vote | RF | IDS | 0.936 | 26.000 | 36.950 |
| vote | RF | IDS-greedy | 0.952 | 5.400 | 2.327 |
| vote | RF | Max | 0.949 | 740.800 | 0.043 |
| vote | RF | RuleFit | 0.956 | 28.000 | 0.081 |
| vote | RF | Vote | 0.956 | 740.800 | 0.048 |

## Why IDS is slow and unstable

`IDS` (`optimizer="sls"`, Smooth Local Search, the paper-faithful default) and `IDS-greedy` (`optimizer="greedy"`, a cheap deterministic fallback) both compress the *same* pool, so any difference here is the optimizer, not the data. On Study 1's small datasets:

| pool | optimizer | mean accuracy | mean fit time (s) |
|---|---|--:|--:|
| CAR | IDS | 0.794 | 28.36 |
| CAR | IDS-greedy | 0.809 | 4.82 |
| LORD | IDS | 0.814 | 44.91 |
| LORD | IDS-greedy | 0.854 | 7.13 |
| RF | IDS | 0.767 | 47.43 |
| RF | IDS-greedy | 0.809 | 4.92 |

Plain `IDS`'s fit time isn't a steady cost that happens to be high -- it varies enormously by (dataset, pool), not just by pool: slowest measured here is `tic-tac-toe`/`LORD` at 95.1s mean, fastest is `hepatitis`/`LORD` at 4.84s -- a 20x spread between datasets this demo treats as comparably small. This is consistent with Smooth Local Search's randomized-restart search having no convergence guarantee on a non-monotone objective, unlike `IDS-greedy`'s single deterministic pass -- not a cost that scales predictably with the data, but per-instance variance. A second, independent symptom of the same instability: on `kr-vs-kp` (Study 2's pool generator, `SKLRandomForest`, tested there before plain `IDS` was dropped from that study), one fold's accuracy came in at 0.523 (near chance) against 0.840-0.897 on the other two folds of the same dataset and pool. Both the speed and the accuracy can apparently collapse independently of data size -- why plain `IDS` stays out of Study 2 (see Setup) while `IDS-greedy`, deterministic, stays in.

## Study 2: larger datasets, RandomForest and LORD pools

### Pool stats (mean across datasets and folds)

| pool | build time (s) | n_rules | mean accuracy (own datasets) | mean accuracy (5 common datasets) | pool-build failures |
|---|--:|--:|--:|--:|--:|
| RF | 3.59 | 751.3 | 0.888 | 0.900 | 0 |
| LORD | 34.72 | 250.5 | 0.913 | 0.913 | 6 |

*LORD has no successful pool on ['adult', 'bank-marketing'] (see "pool-build failures"). "own datasets" is each pool averaged over whatever it actually succeeded on; "5 common datasets" restricts every pool to the same datasets, for a fair comparison.*


### Distiller results

| pool | distiller | accuracy | n_rules | fit_time (s) | failures |
|---|---|--:|--:|--:|--:|
| RF | RuleFit | 0.911 | 144.0 | 1.94 | 0 |
| RF | CBA | 0.905 | 61.3 | 0.47 | 0 |
| RF | CMAR | 0.888 | 29.4 | 0.32 | 0 |
| RF | IDS-greedy | 0.863 | 5.1 | 5.61 | 0 |
| RF | Max | 0.890 | 751.3 | 0.04 | 0 |
| RF | Vote | 0.873 | 751.3 | 0.04 | 0 |
| LORD | RuleFit | 0.923 | 82.8 | 0.19 | 0 |
| LORD | CBA | 0.925 | 129.5 | 0.35 | 0 |
| LORD | CMAR | 0.894 | 96.8 | 0.04 | 0 |
| LORD | IDS-greedy | 0.888 | 13.3 | 20.29 | 0 |
| LORD | Max | 0.925 | 250.5 | 0.01 | 0 |
| LORD | Vote | 0.924 | 250.5 | 0.01 | 0 |

### Mean rank of the distillers

| distiller | RF | LORD | overall |
|---|--:|--:|--:|
| RuleFit | 1.50 | 2.40 | 1.95 |
| CBA | 2.21 | 2.80 | 2.51 |
| Max | 4.14 | 2.40 | 3.27 |
| Vote | 4.86 | 3.00 | 3.93 |
| CMAR | 3.57 | 4.80 | 4.19 |
| IDS-greedy | 4.71 | 5.60 | 5.16 |

*Columns above don't all cover the same datasets -- LORD has no successful pool on ['adult', 'bank-marketing'] (see "pool-build failures" in the pool stats table). Restricted to the 5 datasets common to every pool here, for a fair head-to-head:*

| distiller | RF | LORD | overall |
|---|--:|--:|--:|
| RuleFit | 1.70 | 2.40 | 2.05 |
| CBA | 1.90 | 2.80 | 2.35 |
| Max | 3.80 | 2.40 | 3.10 |
| Vote | 4.80 | 3.00 | 3.90 |
| CMAR | 4.00 | 4.80 | 4.40 |
| IDS-greedy | 4.80 | 5.60 | 5.20 |

### Per-dataset results (mean across folds)

| dataset | pool | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|---|
| adult | RF | CBA | 0.843 | 115.000 | 1.838 |
| adult | RF | CMAR | 0.837 | 40.333 | 0.957 |
| adult | RF | IDS-greedy | 0.797 | 4.000 | 10.689 |
| adult | RF | Max | 0.828 | 798.333 | 0.038 |
| adult | RF | RuleFit | 0.860 | 270.333 | 6.780 |
| adult | RF | Vote | 0.807 | 798.333 | 0.048 |
| bank-marketing | RF | CBA | 0.889 | 59.000 | 0.825 |
| bank-marketing | RF | CMAR | 0.894 | 48.667 | 0.703 |
| bank-marketing | RF | IDS-greedy | 0.893 | 4.000 | 9.015 |
| bank-marketing | RF | Max | 0.884 | 799.333 | 0.044 |
| bank-marketing | RF | RuleFit | 0.899 | 292.667 | 5.531 |
| bank-marketing | RF | Vote | 0.886 | 799.333 | 0.041 |
| credit-g | LORD | CBA | 0.723 | 182.667 | 0.082 |
| credit-g | LORD | CMAR | 0.709 | 140.333 | 0.025 |
| credit-g | LORD | IDS-greedy | 0.706 | 13.333 | 26.707 |
| credit-g | LORD | Max | 0.736 | 341.333 | 0.016 |
| credit-g | LORD | RuleFit | 0.711 | 123.667 | 0.076 |
| credit-g | LORD | Vote | 0.740 | 341.333 | 0.029 |
| credit-g | RF | CBA | 0.748 | 84.333 | 0.083 |
| credit-g | RF | CMAR | 0.738 | 32.000 | 0.048 |
| credit-g | RF | IDS-greedy | 0.704 | 5.667 | 3.121 |
| credit-g | RF | Max | 0.729 | 781.333 | 0.042 |
| credit-g | RF | RuleFit | 0.722 | 149.333 | 0.136 |
| credit-g | RF | Vote | 0.708 | 781.333 | 0.030 |
| kr-vs-kp | LORD | CBA | 0.989 | 43.000 | 1.070 |
| kr-vs-kp | LORD | CMAR | 0.989 | 44.333 | 0.016 |
| kr-vs-kp | LORD | IDS-greedy | 0.975 | 14.333 | 20.403 |
| kr-vs-kp | LORD | Max | 0.990 | 121.333 | 0.003 |
| kr-vs-kp | LORD | RuleFit | 0.992 | 35.667 | 0.419 |
| kr-vs-kp | LORD | Vote | 0.982 | 121.333 | 0.008 |
| kr-vs-kp | RF | CBA | 0.956 | 20.667 | 0.092 |
| kr-vs-kp | RF | CMAR | 0.919 | 12.333 | 0.119 |
| kr-vs-kp | RF | IDS-greedy | 0.829 | 4.000 | 2.379 |
| kr-vs-kp | RF | Max | 0.946 | 688.333 | 0.049 |
| kr-vs-kp | RF | RuleFit | 0.978 | 53.000 | 0.179 |
| kr-vs-kp | RF | Vote | 0.907 | 688.333 | 0.019 |
| mushroom | LORD | CBA | 1.000 | 16.667 | 0.034 |
| mushroom | LORD | CMAR | 1.000 | 19.667 | 0.019 |
| mushroom | LORD | IDS-greedy | 0.999 | 13.000 | 5.658 |
| mushroom | LORD | Max | 1.000 | 34.333 | 0.005 |
| mushroom | LORD | RuleFit | 1.000 | 16.667 | 0.045 |
| mushroom | LORD | Vote | 1.000 | 34.333 | 0.003 |
| mushroom | RF | CBA | 1.000 | 14.000 | 0.134 |
| mushroom | RF | CMAR | 0.955 | 8.000 | 0.172 |
| mushroom | RF | IDS-greedy | 0.965 | 5.667 | 4.645 |
| mushroom | RF | Max | 0.995 | 644.667 | 0.048 |
| mushroom | RF | RuleFit | 1.000 | 26.333 | 0.246 |
| mushroom | RF | Vote | 0.941 | 644.667 | 0.048 |
| sick | LORD | CBA | 0.974 | 91.333 | 0.093 |
| sick | LORD | CMAR | 0.857 | 55.333 | 0.014 |
| sick | LORD | IDS-greedy | 0.961 | 14.333 | 29.821 |
| sick | LORD | Max | 0.962 | 144.333 | 0.008 |
| sick | LORD | RuleFit | 0.977 | 57.000 | 0.074 |
| sick | LORD | Vote | 0.959 | 144.333 | 0.005 |
| sick | RF | CBA | 0.977 | 49.667 | 0.126 |
| sick | RF | CMAR | 0.966 | 24.333 | 0.073 |
| sick | RF | IDS-greedy | 0.978 | 8.667 | 6.954 |
| sick | RF | Max | 0.941 | 760.000 | 0.042 |
| sick | RF | RuleFit | 0.980 | 65.333 | 0.184 |
| sick | RF | Vote | 0.952 | 760.000 | 0.034 |
| spambase | LORD | CBA | 0.938 | 314.000 | 0.482 |
| spambase | LORD | CMAR | 0.914 | 224.333 | 0.119 |
| spambase | LORD | IDS-greedy | 0.801 | 11.333 | 18.844 |
| spambase | LORD | Max | 0.940 | 611.000 | 0.030 |
| spambase | LORD | RuleFit | 0.938 | 181.000 | 0.325 |
| spambase | LORD | Vote | 0.940 | 611.000 | 0.019 |
| spambase | RF | CBA | 0.920 | 86.667 | 0.207 |
| spambase | RF | CMAR | 0.903 | 40.000 | 0.165 |
| spambase | RF | IDS-greedy | 0.873 | 4.000 | 2.472 |
| spambase | RF | Max | 0.910 | 787.000 | 0.044 |
| spambase | RF | RuleFit | 0.937 | 151.333 | 0.509 |
| spambase | RF | Vote | 0.913 | 787.000 | 0.030 |

