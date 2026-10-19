# Covering strategies and boosting

## Setup

**Full run**. Quick sanity check instead: `python demos/covering_boosting.py` with no arguments.

Covering strategies (plain `SeCo`, beam width 5) x heuristics
(`Laplace`, `WRAcc`, m-estimate m=16), plus `CPAR`/`LRI` as complete algorithms, on
molecular-biology_promoters, hepatitis, SPECT, heart-statlog, breast-cancer, heart-h, heart-c, colic, vote, dresses-sales, cylinder-bands, credit-approval, tic-tac-toe, credit-g, kr-vs-kp, sick, mushroom, adult (10-fold; 5-fold for `adult`, the
catalog's one "large" dataset here -- weighted covering and boosting
are no faster on it). Boosting (`Slipper`, `ENDER` x 3 configs,
`Boomer`) on the same datasets/folds, at rule-count budgets
[10, 20, 50, 100].

Protocol: stratified cross-validation (`pyrulearn.experiments.runner.run_cv`), one
`DataSpec` per training fold (`build_dataspec(max_intervals=8)`), the test
fold encoded against that same `DataSpec`, in the library's default representation
(vertical bitmaps (`VerticalBitmapRepresentation`)). Each fit is capped at 300s; a time-out or an error counts
as a failure, not as the end of the run, and is excluded from every
mean it would otherwise enter (a "failures" column reports the count
instead). Every learner runs single-threaded; fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: accuracy, rule count, conditions per rule, fit time, and
overlap (mean rules covering a test row, pooled across every concept --
see the module docstring for why this isn't restricted to one
"positive" class).

## Section 1: covering strategies

| name | accuracy (own cells) | accuracy (170 common cells) | n_rules | conditions/rule | fit_time (s) | overlap | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| Removal+Laplace | 83.33% | 83.33% | 81.6 | 2.9 | 5.50 | 2.4 | 5 |
| Removal+WRAcc | 81.16% | 81.18% | 9.2 | 6.5 | 0.35 | 2.3 | 0 |
| Removal+MEstimate16 | 83.06% | 82.99% | 72.5 | 8.0 | 4.48 | 2.2 | 0 |
| Weighted-Mult+Laplace | 83.80% | 83.84% | 64.6 | 2.7 | 0.89 | 3.2 | 0 |
| Weighted-Mult+WRAcc | 81.60% | 81.64% | 36.6 | 7.0 | 5.58 | 10.6 | 0 |
| Weighted-Mult+MEstimate16 | 84.53% | 84.48% | 90.9 | 5.7 | 4.55 | 7.7 | 0 |
| Weighted-Add+Laplace | 83.38% | 83.43% | 49.9 | 2.7 | 0.90 | 2.8 | 0 |
| Weighted-Add+WRAcc | 82.11% | 82.11% | 39.8 | 8.6 | 9.78 | 14.0 | 5 |
| Weighted-Add+MEstimate16 | 84.70% | 84.75% | 66.2 | 5.6 | 5.84 | 8.1 | 0 |
| Weighted-LRI+Laplace | 81.76% | 81.80% | 24.2 | 3.0 | 0.46 | 2.2 | 0 |
| Weighted-LRI+WRAcc | 82.67% | 82.69% | 74.7 | 7.2 | 8.00 | 17.4 | 0 |
| Weighted-LRI+MEstimate16 | 81.41% | 81.44% | 20.2 | 2.5 | 0.38 | 1.6 | 0 |
| CPAR | 84.07% | 84.12% | 240.4 | 6.6 | 2.24 | 9.0 | 0 |
| LRI | 84.27% | 84.20% | 397.2 | 3.9 | 1.37 | 60.0 | 0 |

*Removal+Laplace missing 5 of 175 cells; Weighted-Add+WRAcc missing 5 of 175 cells (see the "failures" column). "own cells" is each name averaged over whatever (dataset, fold) cells it succeeded on; "170 common cells" restricts every name to the same cells, for a fair comparison.*


### Mean rank

| name | mean rank |
|---|--:|
| Weighted-Mult+MEstimate16 | 4.17 |
| Weighted-Add+MEstimate16 | 4.25 |
| LRI | 5.42 |
| CPAR | 5.64 |
| Weighted-Mult+Laplace | 6.67 |
| Weighted-LRI+WRAcc | 7.14 |
| Removal+MEstimate16 | 7.17 |
| Removal+Laplace | 7.53 |
| Weighted-Add+Laplace | 8.06 |
| Weighted-LRI+MEstimate16 | 9.06 |
| Weighted-Add+WRAcc | 9.28 |
| Weighted-LRI+Laplace | 9.67 |
| Removal+WRAcc | 10.33 |
| Weighted-Mult+WRAcc | 10.64 |

### Per-dataset results (mean across folds)

| dataset | learner | accuracy | n_rules | fit_time | overlap |
|---|---|---|---|---|---|
| SPECT | CPAR | 82.78% | 107.8 | 0.138 | 10.5 |
| SPECT | LRI | 80.81% | 399.8 | 0.464 | 49.6 |
| SPECT | Removal+Laplace | 82.36% | 49.3 | 0.328 | 3.9 |
| SPECT | Removal+MEstimate16 | 81.27% | 19.8 | 0.185 | 2.5 |
| SPECT | Removal+WRAcc | 79.43% | 9.6 | 0.076 | 2.7 |
| SPECT | Weighted-Add+Laplace | 85.00% | 35.2 | 0.507 | 3.8 |
| SPECT | Weighted-Add+MEstimate16 | 84.62% | 50.6 | 1.365 | 8.8 |
| SPECT | Weighted-Add+WRAcc | 84.64% | 29.4 | 0.554 | 8.2 |
| SPECT | Weighted-LRI+Laplace | 79.34% | 33.2 | 0.419 | 2.6 |
| SPECT | Weighted-LRI+MEstimate16 | 77.92% | 28.0 | 0.298 | 2.2 |
| SPECT | Weighted-LRI+WRAcc | 83.18% | 66.7 | 0.735 | 12.5 |
| SPECT | Weighted-Mult+Laplace | 84.63% | 48.1 | 0.685 | 4.8 |
| SPECT | Weighted-Mult+MEstimate16 | 83.86% | 71.7 | 1.029 | 9.5 |
| SPECT | Weighted-Mult+WRAcc | 79.05% | 49.0 | 0.515 | 10.9 |
| adult | CPAR | 82.18% | 2097.0 | 66.641 | 14.1 |
| adult | LRI | 86.93% | 400.0 | 20.423 | 59.7 |
| adult | Removal+Laplace | n/a | n/a | 300.125 | n/a |
| adult | Removal+MEstimate16 | 85.75% | 1525.0 | 139.277 | 4.3 |
| adult | Removal+WRAcc | 80.54% | 16.6 | 3.766 | 5.0 |
| adult | Weighted-Add+Laplace | 81.60% | 87.4 | 13.054 | 1.6 |
| adult | Weighted-Add+MEstimate16 | 83.18% | 132.8 | 68.388 | 5.6 |
| adult | Weighted-Add+WRAcc | n/a | n/a | 300.110 | n/a |
| adult | Weighted-LRI+Laplace | 80.29% | 27.0 | 5.855 | 0.7 |
| adult | Weighted-LRI+MEstimate16 | 80.30% | 19.6 | 4.428 | 0.3 |
| adult | Weighted-LRI+WRAcc | 81.70% | 90.4 | 153.812 | 26.7 |
| adult | Weighted-Mult+Laplace | 82.43% | 172.4 | 13.526 | 1.8 |
| adult | Weighted-Mult+MEstimate16 | 86.06% | 198.6 | 51.090 | 3.3 |
| adult | Weighted-Mult+WRAcc | 80.32% | 73.6 | 139.058 | 26.0 |
| breast-cancer | CPAR | 73.42% | 216.1 | 0.353 | 10.8 |
| breast-cancer | LRI | 70.64% | 395.0 | 0.419 | 60.8 |
| breast-cancer | Removal+Laplace | 67.14% | 85.3 | 0.264 | 1.7 |
| breast-cancer | Removal+MEstimate16 | 68.26% | 32.0 | 0.372 | 1.9 |
| breast-cancer | Removal+WRAcc | 71.66% | 8.9 | 0.139 | 2.4 |
| breast-cancer | Weighted-Add+Laplace | 71.67% | 58.5 | 0.344 | 2.2 |
| breast-cancer | Weighted-Add+MEstimate16 | 74.50% | 100.5 | 3.065 | 10.4 |
| breast-cancer | Weighted-Add+WRAcc | 73.44% | 55.0 | 2.445 | 19.6 |
| breast-cancer | Weighted-LRI+Laplace | 69.22% | 26.3 | 0.256 | 1.3 |
| breast-cancer | Weighted-LRI+MEstimate16 | 68.50% | 21.3 | 0.202 | 0.7 |
| breast-cancer | Weighted-LRI+WRAcc | 72.75% | 96.3 | 1.852 | 22.6 |
| breast-cancer | Weighted-Mult+Laplace | 72.01% | 70.5 | 0.363 | 2.3 |
| breast-cancer | Weighted-Mult+MEstimate16 | 73.45% | 115.1 | 2.357 | 9.2 |
| breast-cancer | Weighted-Mult+WRAcc | 72.04% | 40.2 | 0.800 | 11.7 |
| colic | CPAR | 83.98% | 157.0 | 0.229 | 9.9 |
| colic | LRI | 85.07% | 400.0 | 0.707 | 65.4 |
| colic | Removal+Laplace | 82.91% | 79.2 | 0.190 | 3.2 |
| colic | Removal+MEstimate16 | 84.54% | 26.0 | 0.245 | 2.9 |
| colic | Removal+WRAcc | 86.43% | 10.4 | 0.123 | 2.2 |
| colic | Weighted-Add+Laplace | 81.82% | 70.6 | 0.423 | 4.3 |
| colic | Weighted-Add+MEstimate16 | 86.69% | 72.0 | 3.305 | 11.6 |
| colic | Weighted-Add+WRAcc | 83.69% | 30.9 | 3.005 | 9.2 |
| colic | Weighted-LRI+Laplace | 80.72% | 32.9 | 0.273 | 2.6 |
| colic | Weighted-LRI+MEstimate16 | 80.45% | 21.9 | 0.220 | 1.4 |
| colic | Weighted-LRI+WRAcc | 85.08% | 80.4 | 1.302 | 15.2 |
| colic | Weighted-Mult+Laplace | 81.53% | 70.8 | 0.350 | 3.6 |
| colic | Weighted-Mult+MEstimate16 | 83.71% | 91.3 | 1.867 | 8.2 |
| colic | Weighted-Mult+WRAcc | 83.70% | 42.9 | 0.980 | 9.0 |
| credit-approval | CPAR | 85.22% | 367.5 | 0.486 | 15.6 |
| credit-approval | LRI | 87.25% | 398.7 | 0.690 | 59.3 |
| credit-approval | Removal+Laplace | 82.46% | 120.0 | 0.404 | 2.4 |
| credit-approval | Removal+MEstimate16 | 85.36% | 34.5 | 0.709 | 2.6 |
| credit-approval | Removal+WRAcc | 83.62% | 9.0 | 0.217 | 2.4 |
| credit-approval | Weighted-Add+Laplace | 84.78% | 69.2 | 0.592 | 5.1 |
| credit-approval | Weighted-Add+MEstimate16 | 86.52% | 79.4 | 7.190 | 15.2 |
| credit-approval | Weighted-Add+WRAcc | 85.80% | 44.2 | 7.014 | 18.9 |
| credit-approval | Weighted-LRI+Laplace | 75.94% | 31.2 | 0.365 | 3.4 |
| credit-approval | Weighted-LRI+MEstimate16 | 79.28% | 17.8 | 0.278 | 1.4 |
| credit-approval | Weighted-LRI+WRAcc | 86.38% | 90.7 | 3.163 | 20.6 |
| credit-approval | Weighted-Mult+Laplace | 83.77% | 81.4 | 0.529 | 4.2 |
| credit-approval | Weighted-Mult+MEstimate16 | 86.09% | 134.9 | 5.112 | 12.8 |
| credit-approval | Weighted-Mult+WRAcc | 85.07% | 44.4 | 1.815 | 13.5 |
| credit-g | CPAR | 74.40% | 523.3 | 1.071 | 10.6 |
| credit-g | LRI | 75.30% | 400.0 | 0.807 | 58.0 |
| credit-g | Removal+Laplace | 74.40% | 232.1 | 0.768 | 2.3 |
| credit-g | Removal+MEstimate16 | 73.00% | 76.4 | 1.252 | 1.8 |
| credit-g | Removal+WRAcc | 71.70% | 13.0 | 0.357 | 3.0 |
| credit-g | Weighted-Add+Laplace | 73.70% | 102.3 | 0.589 | 2.2 |
| credit-g | Weighted-Add+MEstimate16 | 75.00% | 158.5 | 4.153 | 7.5 |
| credit-g | Weighted-Add+WRAcc | 71.20% | 117.4 | 6.974 | 37.7 |
| credit-g | Weighted-LRI+Laplace | 72.70% | 25.3 | 0.282 | 0.7 |
| credit-g | Weighted-LRI+MEstimate16 | 74.00% | 21.1 | 0.307 | 0.5 |
| credit-g | Weighted-LRI+WRAcc | 70.30% | 100.0 | 3.567 | 21.5 |
| credit-g | Weighted-Mult+Laplace | 75.20% | 133.5 | 0.639 | 2.5 |
| credit-g | Weighted-Mult+MEstimate16 | 75.10% | 190.2 | 4.290 | 6.3 |
| credit-g | Weighted-Mult+WRAcc | 70.80% | 52.2 | 2.037 | 14.0 |
| cylinder-bands | CPAR | 77.04% | 284.8 | 0.494 | 10.9 |
| cylinder-bands | LRI | 82.22% | 399.4 | 0.875 | 56.2 |
| cylinder-bands | Removal+Laplace | 74.81% | 155.5 | 0.446 | 1.7 |
| cylinder-bands | Removal+MEstimate16 | 74.44% | 49.9 | 1.958 | 1.6 |
| cylinder-bands | Removal+WRAcc | 73.70% | 9.0 | 1.526 | 2.2 |
| cylinder-bands | Weighted-Add+Laplace | 71.67% | 75.5 | 0.361 | 1.4 |
| cylinder-bands | Weighted-Add+MEstimate16 | 75.56% | 102.0 | 13.519 | 6.4 |
| cylinder-bands | Weighted-Add+WRAcc | 72.59% | 69.0 | 19.954 | 27.4 |
| cylinder-bands | Weighted-LRI+Laplace | 65.93% | 21.6 | 0.230 | 0.7 |
| cylinder-bands | Weighted-LRI+MEstimate16 | 65.93% | 17.4 | 0.189 | 0.4 |
| cylinder-bands | Weighted-LRI+WRAcc | 73.33% | 99.0 | 22.458 | 21.4 |
| cylinder-bands | Weighted-Mult+Laplace | 72.22% | 88.9 | 0.373 | 1.4 |
| cylinder-bands | Weighted-Mult+MEstimate16 | 75.56% | 142.8 | 13.605 | 7.4 |
| cylinder-bands | Weighted-Mult+WRAcc | 71.67% | 32.6 | 8.051 | 11.4 |
| dresses-sales | CPAR | 57.00% | 659.0 | 0.989 | 13.0 |
| dresses-sales | LRI | 58.80% | 394.6 | 0.647 | 58.5 |
| dresses-sales | Removal+Laplace | 59.60% | 226.3 | 0.668 | 1.6 |
| dresses-sales | Removal+MEstimate16 | 57.60% | 71.0 | 1.942 | 1.7 |
| dresses-sales | Removal+WRAcc | 56.20% | 12.2 | 0.767 | 2.9 |
| dresses-sales | Weighted-Add+Laplace | 58.60% | 80.3 | 0.274 | 0.7 |
| dresses-sales | Weighted-Add+MEstimate16 | 60.20% | 139.8 | 7.370 | 4.9 |
| dresses-sales | Weighted-Add+WRAcc | 58.40% | 113.6 | 13.213 | 33.6 |
| dresses-sales | Weighted-LRI+Laplace | 62.20% | 28.2 | 0.230 | 0.6 |
| dresses-sales | Weighted-LRI+MEstimate16 | 58.00% | 27.0 | 0.208 | 0.3 |
| dresses-sales | Weighted-LRI+WRAcc | 59.40% | 99.5 | 8.633 | 22.2 |
| dresses-sales | Weighted-Mult+Laplace | 61.40% | 87.0 | 0.268 | 0.7 |
| dresses-sales | Weighted-Mult+MEstimate16 | 59.20% | 165.8 | 7.476 | 4.9 |
| dresses-sales | Weighted-Mult+WRAcc | 56.60% | 49.4 | 4.183 | 12.9 |
| heart-c | CPAR | 80.55% | 141.9 | 0.188 | 10.4 |
| heart-c | LRI | 78.27% | 399.6 | 0.515 | 59.0 |
| heart-c | Removal+Laplace | 81.23% | 58.6 | 0.143 | 2.6 |
| heart-c | Removal+MEstimate16 | 78.24% | 20.7 | 0.216 | 2.1 |
| heart-c | Removal+WRAcc | 76.60% | 8.3 | 0.100 | 2.1 |
| heart-c | Weighted-Add+Laplace | 79.57% | 51.2 | 0.368 | 3.8 |
| heart-c | Weighted-Add+MEstimate16 | 81.28% | 69.7 | 3.113 | 14.5 |
| heart-c | Weighted-Add+WRAcc | 76.92% | 27.6 | 2.228 | 10.9 |
| heart-c | Weighted-LRI+Laplace | 77.90% | 25.2 | 0.240 | 2.6 |
| heart-c | Weighted-LRI+MEstimate16 | 79.51% | 20.1 | 0.241 | 1.7 |
| heart-c | Weighted-LRI+WRAcc | 79.92% | 88.4 | 1.416 | 20.7 |
| heart-c | Weighted-Mult+Laplace | 80.24% | 66.3 | 0.369 | 4.0 |
| heart-c | Weighted-Mult+MEstimate16 | 79.96% | 82.6 | 1.577 | 10.7 |
| heart-c | Weighted-Mult+WRAcc | 78.92% | 33.5 | 0.605 | 12.6 |
| heart-h | CPAR | 80.98% | 166.8 | 0.221 | 10.7 |
| heart-h | LRI | 81.64% | 396.6 | 0.466 | 56.5 |
| heart-h | Removal+Laplace | 78.87% | 58.4 | 0.200 | 2.2 |
| heart-h | Removal+MEstimate16 | 82.64% | 21.1 | 0.276 | 2.0 |
| heart-h | Removal+WRAcc | 78.52% | 7.9 | 0.125 | 1.9 |
| heart-h | Weighted-Add+Laplace | 78.89% | 34.7 | 0.293 | 2.5 |
| heart-h | Weighted-Add+MEstimate16 | 79.60% | 56.5 | 3.733 | 11.4 |
| heart-h | Weighted-Add+WRAcc | 77.53% | 29.9 | 4.036 | 13.8 |
| heart-h | Weighted-LRI+Laplace | 79.91% | 25.1 | 0.276 | 2.6 |
| heart-h | Weighted-LRI+MEstimate16 | 80.25% | 17.6 | 0.152 | 1.4 |
| heart-h | Weighted-LRI+WRAcc | 81.30% | 83.3 | 1.806 | 19.1 |
| heart-h | Weighted-Mult+Laplace | 78.89% | 36.1 | 0.223 | 2.3 |
| heart-h | Weighted-Mult+MEstimate16 | 80.29% | 78.2 | 2.065 | 10.1 |
| heart-h | Weighted-Mult+WRAcc | 76.16% | 36.3 | 0.848 | 12.8 |
| heart-statlog | CPAR | 82.22% | 126.4 | 0.180 | 9.4 |
| heart-statlog | LRI | 76.30% | 399.6 | 0.526 | 59.3 |
| heart-statlog | Removal+Laplace | 78.52% | 50.4 | 0.136 | 2.5 |
| heart-statlog | Removal+MEstimate16 | 78.15% | 18.5 | 0.228 | 2.1 |
| heart-statlog | Removal+WRAcc | 78.15% | 8.4 | 0.103 | 2.2 |
| heart-statlog | Weighted-Add+Laplace | 80.37% | 52.4 | 0.365 | 3.9 |
| heart-statlog | Weighted-Add+MEstimate16 | 80.00% | 61.7 | 3.466 | 12.8 |
| heart-statlog | Weighted-Add+WRAcc | 78.15% | 33.4 | 2.339 | 13.1 |
| heart-statlog | Weighted-LRI+Laplace | 78.15% | 26.9 | 0.257 | 2.8 |
| heart-statlog | Weighted-LRI+MEstimate16 | 79.26% | 22.6 | 0.244 | 1.9 |
| heart-statlog | Weighted-LRI+WRAcc | 80.37% | 88.6 | 1.584 | 21.0 |
| heart-statlog | Weighted-Mult+Laplace | 80.37% | 62.5 | 0.364 | 3.9 |
| heart-statlog | Weighted-Mult+MEstimate16 | 80.37% | 72.9 | 1.576 | 10.5 |
| heart-statlog | Weighted-Mult+WRAcc | 74.44% | 33.4 | 0.654 | 12.4 |
| hepatitis | CPAR | 80.71% | 69.7 | 0.079 | 8.4 |
| hepatitis | LRI | 82.12% | 399.4 | 0.416 | 64.7 |
| hepatitis | Removal+Laplace | 81.38% | 24.5 | 0.049 | 2.6 |
| hepatitis | Removal+MEstimate16 | 76.25% | 14.3 | 0.077 | 2.6 |
| hepatitis | Removal+WRAcc | 81.33% | 7.8 | 0.055 | 2.6 |
| hepatitis | Weighted-Add+Laplace | 82.62% | 23.6 | 0.260 | 3.8 |
| hepatitis | Weighted-Add+MEstimate16 | 83.25% | 38.4 | 1.487 | 10.1 |
| hepatitis | Weighted-Add+WRAcc | 81.33% | 25.0 | 0.868 | 11.8 |
| hepatitis | Weighted-LRI+Laplace | 85.17% | 20.8 | 0.174 | 3.9 |
| hepatitis | Weighted-LRI+MEstimate16 | 83.33% | 17.1 | 0.175 | 2.4 |
| hepatitis | Weighted-LRI+WRAcc | 83.87% | 63.1 | 0.649 | 15.9 |
| hepatitis | Weighted-Mult+Laplace | 84.58% | 29.3 | 0.227 | 4.2 |
| hepatitis | Weighted-Mult+MEstimate16 | 81.92% | 45.8 | 0.517 | 8.9 |
| hepatitis | Weighted-Mult+WRAcc | 80.62% | 30.7 | 0.329 | 10.4 |
| kr-vs-kp | CPAR | 99.28% | 50.6 | 0.278 | 3.8 |
| kr-vs-kp | LRI | 99.12% | 399.7 | 1.137 | 62.6 |
| kr-vs-kp | Removal+Laplace | 99.34% | 68.6 | 0.327 | 2.2 |
| kr-vs-kp | Removal+MEstimate16 | 99.22% | 32.3 | 0.321 | 1.7 |
| kr-vs-kp | Removal+WRAcc | 94.68% | 7.5 | 0.095 | 1.6 |
| kr-vs-kp | Weighted-Add+Laplace | 97.03% | 26.7 | 0.582 | 1.6 |
| kr-vs-kp | Weighted-Add+MEstimate16 | 98.50% | 27.4 | 2.239 | 2.8 |
| kr-vs-kp | Weighted-Add+WRAcc | 92.49% | 11.1 | 2.991 | 4.4 |
| kr-vs-kp | Weighted-LRI+Laplace | 92.34% | 11.9 | 0.269 | 1.2 |
| kr-vs-kp | Weighted-LRI+MEstimate16 | 86.08% | 16.6 | 0.250 | 1.1 |
| kr-vs-kp | Weighted-LRI+WRAcc | 95.90% | 35.0 | 1.615 | 9.2 |
| kr-vs-kp | Weighted-Mult+Laplace | 98.44% | 44.3 | 0.519 | 1.9 |
| kr-vs-kp | Weighted-Mult+MEstimate16 | 99.41% | 56.5 | 2.900 | 4.2 |
| kr-vs-kp | Weighted-Mult+WRAcc | 92.18% | 18.0 | 0.817 | 8.1 |
| molecular-biology_promoters | CPAR | 80.36% | 31.5 | 0.048 | 5.3 |
| molecular-biology_promoters | LRI | 82.91% | 390.5 | 0.494 | 62.1 |
| molecular-biology_promoters | Removal+Laplace | 82.09% | 11.9 | 0.040 | 1.6 |
| molecular-biology_promoters | Removal+MEstimate16 | 79.36% | 6.9 | 0.039 | 1.5 |
| molecular-biology_promoters | Removal+WRAcc | 74.91% | 5.1 | 0.049 | 1.2 |
| molecular-biology_promoters | Weighted-Add+Laplace | 85.00% | 21.8 | 0.543 | 3.3 |
| molecular-biology_promoters | Weighted-Add+MEstimate16 | 84.09% | 19.8 | 0.519 | 5.2 |
| molecular-biology_promoters | Weighted-Add+WRAcc | 85.91% | 16.3 | 0.432 | 5.4 |
| molecular-biology_promoters | Weighted-LRI+Laplace | 86.82% | 25.2 | 0.313 | 4.0 |
| molecular-biology_promoters | Weighted-LRI+MEstimate16 | 85.09% | 27.2 | 0.334 | 4.9 |
| molecular-biology_promoters | Weighted-LRI+WRAcc | 78.45% | 74.2 | 1.143 | 21.0 |
| molecular-biology_promoters | Weighted-Mult+Laplace | 82.18% | 42.5 | 0.499 | 5.3 |
| molecular-biology_promoters | Weighted-Mult+MEstimate16 | 84.09% | 25.9 | 0.238 | 6.3 |
| molecular-biology_promoters | Weighted-Mult+WRAcc | 85.00% | 21.7 | 0.278 | 6.4 |
| mushroom | CPAR | 99.96% | 21.3 | 0.257 | 5.1 |
| mushroom | LRI | 100.00% | 386.9 | 2.814 | 67.7 |
| mushroom | Removal+Laplace | 100.00% | 23.0 | 0.130 | 1.5 |
| mushroom | Removal+MEstimate16 | 100.00% | 18.6 | 0.103 | 1.6 |
| mushroom | Removal+WRAcc | 99.98% | 4.5 | 0.087 | 1.1 |
| mushroom | Weighted-Add+Laplace | 98.98% | 15.1 | 0.741 | 0.9 |
| mushroom | Weighted-Add+MEstimate16 | 98.98% | 19.4 | 1.280 | 2.8 |
| mushroom | Weighted-Add+WRAcc | 99.79% | 6.4 | 3.767 | 2.7 |
| mushroom | Weighted-LRI+Laplace | 95.72% | 9.0 | 0.368 | 0.8 |
| mushroom | Weighted-LRI+MEstimate16 | 95.72% | 9.0 | 0.369 | 0.8 |
| mushroom | Weighted-LRI+WRAcc | 99.94% | 31.6 | 8.251 | 11.5 |
| mushroom | Weighted-Mult+Laplace | 99.75% | 25.1 | 0.775 | 1.7 |
| mushroom | Weighted-Mult+MEstimate16 | 100.00% | 28.2 | 1.422 | 4.0 |
| mushroom | Weighted-Mult+WRAcc | 99.79% | 8.3 | 2.387 | 3.3 |
| sick | CPAR | 98.44% | 143.6 | 0.594 | 7.3 |
| sick | LRI | 98.49% | 392.0 | 1.527 | 69.4 |
| sick | Removal+Laplace | 97.93% | 86.9 | 0.784 | 2.7 |
| sick | Removal+MEstimate16 | 98.04% | 30.1 | 0.756 | 2.9 |
| sick | Removal+WRAcc | 97.19% | 10.6 | 0.273 | 2.2 |
| sick | Weighted-Add+Laplace | 97.93% | 39.5 | 2.094 | 3.1 |
| sick | Weighted-Add+MEstimate16 | 98.12% | 45.1 | 10.656 | 6.8 |
| sick | Weighted-Add+WRAcc | 97.56% | 22.8 | 8.287 | 10.1 |
| sick | Weighted-LRI+Laplace | 97.40% | 23.1 | 0.721 | 2.2 |
| sick | Weighted-LRI+MEstimate16 | 98.04% | 17.3 | 0.511 | 0.7 |
| sick | Weighted-LRI+WRAcc | 96.82% | 60.9 | 3.945 | 10.4 |
| sick | Weighted-Mult+Laplace | 98.04% | 66.9 | 1.735 | 4.7 |
| sick | Weighted-Mult+MEstimate16 | 97.64% | 97.6 | 6.905 | 9.4 |
| sick | Weighted-Mult+WRAcc | 95.68% | 41.5 | 3.214 | 9.8 |
| tic-tac-toe | CPAR | 98.33% | 57.2 | 0.163 | 2.9 |
| tic-tac-toe | LRI | 96.97% | 400.0 | 0.705 | 50.7 |
| tic-tac-toe | Removal+Laplace | 98.33% | 28.6 | 0.082 | 1.3 |
| tic-tac-toe | Removal+MEstimate16 | 98.96% | 20.9 | 0.082 | 1.3 |
| tic-tac-toe | Removal+WRAcc | 81.20% | 12.3 | 0.061 | 2.4 |
| tic-tac-toe | Weighted-Add+Laplace | 96.87% | 52.6 | 0.565 | 2.7 |
| tic-tac-toe | Weighted-Add+MEstimate16 | 97.91% | 35.4 | 0.707 | 2.5 |
| tic-tac-toe | Weighted-Add+WRAcc | 83.92% | 34.4 | 0.575 | 7.3 |
| tic-tac-toe | Weighted-LRI+Laplace | 97.39% | 26.2 | 0.276 | 1.6 |
| tic-tac-toe | Weighted-LRI+MEstimate16 | 97.50% | 24.5 | 0.292 | 1.6 |
| tic-tac-toe | Weighted-LRI+WRAcc | 83.61% | 60.0 | 0.584 | 15.5 |
| tic-tac-toe | Weighted-Mult+Laplace | 97.39% | 57.1 | 0.532 | 2.8 |
| tic-tac-toe | Weighted-Mult+MEstimate16 | 99.69% | 51.0 | 0.604 | 2.9 |
| tic-tac-toe | Weighted-Mult+WRAcc | 90.91% | 38.8 | 0.304 | 6.6 |
| vote | CPAR | 95.41% | 34.6 | 0.065 | 6.5 |
| vote | LRI | 95.41% | 400.0 | 0.471 | 60.9 |
| vote | Removal+Laplace | 95.18% | 28.1 | 0.076 | 4.1 |
| vote | Removal+MEstimate16 | 95.42% | 13.6 | 0.043 | 3.1 |
| vote | Removal+WRAcc | 94.74% | 8.3 | 0.019 | 2.3 |
| vote | Weighted-Add+Laplace | 93.81% | 19.8 | 0.384 | 3.7 |
| vote | Weighted-Add+MEstimate16 | 95.87% | 16.6 | 0.912 | 4.9 |
| vote | Weighted-Add+WRAcc | 92.42% | 9.6 | 0.429 | 4.2 |
| vote | Weighted-LRI+Laplace | 93.81% | 18.1 | 0.236 | 4.0 |
| vote | Weighted-LRI+MEstimate16 | 95.65% | 17.9 | 0.236 | 4.2 |
| vote | Weighted-LRI+WRAcc | 95.20% | 43.5 | 0.365 | 11.7 |
| vote | Weighted-Mult+Laplace | 94.73% | 34.8 | 0.350 | 5.2 |
| vote | Weighted-Mult+MEstimate16 | 95.88% | 41.4 | 0.481 | 6.9 |
| vote | Weighted-Mult+WRAcc | 95.19% | 31.2 | 0.299 | 6.5 |

## Section 2: boosting

| family | budget | accuracy (own cells) | accuracy (170 common cells) | n_rules (actual) | fit_time (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| Slipper | 10 | 83.24% | 83.18% | 7.6 | 0.06 | 0 |
| Slipper | 20 | 83.82% | 83.75% | 13.7 | 0.12 | 0 |
| Slipper | 50 | 83.98% | 83.90% | 28.5 | 0.27 | 0 |
| Slipper | 100 | 83.72% | 83.63% | 47.1 | 0.51 | 0 |
| ENDER-CSLog | 10 | 79.10% | 79.19% | 9.6 | 0.04 | 0 |
| ENDER-CSLog | 20 | 81.43% | 81.41% | 17.5 | 0.08 | 0 |
| ENDER-CSLog | 50 | 83.36% | 83.32% | 42.0 | 0.19 | 0 |
| ENDER-CSLog | 100 | 84.23% | 84.19% | 83.0 | 0.37 | 0 |
| ENDER-CSExp | 10 | 79.21% | 79.30% | 9.9 | 0.03 | 0 |
| ENDER-CSExp | 20 | 81.28% | 81.29% | 18.3 | 0.06 | 0 |
| ENDER-CSExp | 50 | 83.81% | 83.78% | 43.6 | 0.15 | 0 |
| ENDER-CSExp | 100 | 84.66% | 84.63% | 84.6 | 0.29 | 0 |
| ENDER-Newton | 10 | 80.19% | 80.25% | 9.5 | 0.04 | 0 |
| ENDER-Newton | 20 | 82.06% | 82.12% | 17.1 | 0.08 | 0 |
| ENDER-Newton | 50 | 83.59% | 83.54% | 40.1 | 0.18 | 0 |
| ENDER-Newton | 100 | 85.26% | 85.24% | 76.1 | 0.34 | 0 |
| Boomer | 10 | 82.61% | 82.59% | 9.1 | 0.05 | 0 |
| Boomer | 20 | 84.28% | 84.24% | 16.6 | 0.09 | 0 |
| Boomer | 50 | 84.99% | 84.95% | 35.3 | 0.21 | 0 |
| Boomer | 100 | 84.91% | 84.85% | 60.4 | 0.40 | 0 |

*"own cells" is each name averaged over whatever (dataset, fold) cells it succeeded on; "170 common cells" is the same cell subset section 1's summary table uses, so the two tables' accuracy columns are directly comparable (see the "failures" column).*


### Per-dataset results (mean across folds)

| dataset | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|
| SPECT | Boomer@10 | 82.81% | 9.2 | 0.010 |
| SPECT | Boomer@100 | 83.49% | 46.6 | 0.078 |
| SPECT | Boomer@20 | 83.90% | 14.6 | 0.017 |
| SPECT | Boomer@50 | 83.50% | 29.0 | 0.040 |
| SPECT | ENDER-CSExp@10 | 79.42% | 9.7 | 0.008 |
| SPECT | ENDER-CSExp@100 | 85.00% | 53.9 | 0.059 |
| SPECT | ENDER-CSExp@20 | 79.42% | 16.2 | 0.015 |
| SPECT | ENDER-CSExp@50 | 83.18% | 32.2 | 0.029 |
| SPECT | ENDER-CSLog@10 | 79.42% | 9.6 | 0.010 |
| SPECT | ENDER-CSLog@100 | 84.26% | 62.3 | 0.072 |
| SPECT | ENDER-CSLog@20 | 79.42% | 15.9 | 0.017 |
| SPECT | ENDER-CSLog@50 | 84.27% | 33.1 | 0.038 |
| SPECT | ENDER-Newton@10 | 79.42% | 10.3 | 0.010 |
| SPECT | ENDER-Newton@100 | 85.01% | 68.3 | 0.088 |
| SPECT | ENDER-Newton@20 | 79.42% | 17.6 | 0.020 |
| SPECT | ENDER-Newton@50 | 83.90% | 38.1 | 0.041 |
| SPECT | Slipper@10 | 84.25% | 5.9 | 0.029 |
| SPECT | Slipper@100 | 81.20% | 44.4 | 0.260 |
| SPECT | Slipper@20 | 83.13% | 11.3 | 0.053 |
| SPECT | Slipper@50 | 81.97% | 26.8 | 0.136 |
| adult | Boomer@10 | 83.26% | 11.0 | 0.960 |
| adult | Boomer@100 | 86.70% | 99.2 | 7.761 |
| adult | Boomer@20 | 85.66% | 20.6 | 1.722 |
| adult | Boomer@50 | 86.31% | 49.6 | 4.110 |
| adult | ENDER-CSExp@10 | 76.07% | 10.8 | 0.549 |
| adult | ENDER-CSExp@100 | 85.82% | 99.4 | 4.962 |
| adult | ENDER-CSExp@20 | 81.10% | 20.8 | 1.035 |
| adult | ENDER-CSExp@50 | 84.82% | 50.2 | 2.573 |
| adult | ENDER-CSLog@10 | 76.07% | 10.6 | 0.677 |
| adult | ENDER-CSLog@100 | 85.71% | 96.2 | 6.576 |
| adult | ENDER-CSLog@20 | 82.01% | 20.2 | 1.296 |
| adult | ENDER-CSLog@50 | 84.60% | 47.4 | 3.275 |
| adult | ENDER-Newton@10 | 78.32% | 10.4 | 0.707 |
| adult | ENDER-Newton@100 | 85.94% | 96.4 | 6.394 |
| adult | ENDER-Newton@20 | 80.17% | 19.8 | 1.366 |
| adult | ENDER-Newton@50 | 85.18% | 48.2 | 3.279 |
| adult | Slipper@10 | 85.45% | 7.2 | 0.976 |
| adult | Slipper@100 | 86.63% | 69.8 | 7.998 |
| adult | Slipper@20 | 86.08% | 14.2 | 1.792 |
| adult | Slipper@50 | 86.68% | 36.2 | 4.389 |
| breast-cancer | Boomer@10 | 72.38% | 9.8 | 0.012 |
| breast-cancer | Boomer@100 | 68.17% | 58.1 | 0.086 |
| breast-cancer | Boomer@20 | 71.35% | 16.5 | 0.025 |
| breast-cancer | Boomer@50 | 69.56% | 34.9 | 0.050 |
| breast-cancer | ENDER-CSExp@10 | 70.30% | 11.0 | 0.009 |
| breast-cancer | ENDER-CSExp@100 | 72.75% | 97.4 | 0.085 |
| breast-cancer | ENDER-CSExp@20 | 70.65% | 20.9 | 0.016 |
| breast-cancer | ENDER-CSExp@50 | 74.48% | 50.0 | 0.041 |
| breast-cancer | ENDER-CSLog@10 | 70.30% | 11.0 | 0.011 |
| breast-cancer | ENDER-CSLog@100 | 72.76% | 97.5 | 0.090 |
| breast-cancer | ENDER-CSLog@20 | 71.00% | 20.7 | 0.020 |
| breast-cancer | ENDER-CSLog@50 | 73.08% | 49.8 | 0.050 |
| breast-cancer | ENDER-Newton@10 | 71.33% | 10.7 | 0.011 |
| breast-cancer | ENDER-Newton@100 | 72.39% | 87.5 | 0.085 |
| breast-cancer | ENDER-Newton@20 | 72.03% | 19.8 | 0.021 |
| breast-cancer | ENDER-Newton@50 | 72.72% | 46.2 | 0.046 |
| breast-cancer | Slipper@10 | 72.72% | 7.3 | 0.021 |
| breast-cancer | Slipper@100 | 70.27% | 47.6 | 0.180 |
| breast-cancer | Slipper@20 | 72.03% | 13.9 | 0.035 |
| breast-cancer | Slipper@50 | 69.57% | 29.4 | 0.093 |
| colic | Boomer@10 | 85.61% | 8.3 | 0.013 |
| colic | Boomer@100 | 84.53% | 72.9 | 0.115 |
| colic | Boomer@20 | 85.61% | 17.0 | 0.024 |
| colic | Boomer@50 | 85.08% | 39.6 | 0.056 |
| colic | ENDER-CSExp@10 | 80.16% | 9.3 | 0.010 |
| colic | ENDER-CSExp@100 | 84.52% | 89.8 | 0.089 |
| colic | ENDER-CSExp@20 | 84.52% | 17.3 | 0.019 |
| colic | ENDER-CSExp@50 | 84.25% | 44.7 | 0.044 |
| colic | ENDER-CSLog@10 | 81.00% | 8.6 | 0.010 |
| colic | ENDER-CSLog@100 | 84.79% | 87.8 | 0.101 |
| colic | ENDER-CSLog@20 | 84.25% | 15.8 | 0.019 |
| colic | ENDER-CSLog@50 | 84.52% | 42.5 | 0.051 |
| colic | ENDER-Newton@10 | 83.98% | 8.5 | 0.013 |
| colic | ENDER-Newton@100 | 85.07% | 85.2 | 0.095 |
| colic | ENDER-Newton@20 | 85.33% | 17.0 | 0.021 |
| colic | ENDER-Newton@50 | 85.07% | 43.3 | 0.048 |
| colic | Slipper@10 | 83.18% | 7.7 | 0.022 |
| colic | Slipper@100 | 80.17% | 47.2 | 0.181 |
| colic | Slipper@20 | 82.88% | 13.9 | 0.040 |
| colic | Slipper@50 | 82.08% | 28.4 | 0.091 |
| credit-approval | Boomer@10 | 86.09% | 9.9 | 0.021 |
| credit-approval | Boomer@100 | 86.38% | 75.6 | 0.158 |
| credit-approval | Boomer@20 | 86.81% | 18.7 | 0.036 |
| credit-approval | Boomer@50 | 86.67% | 43.3 | 0.080 |
| credit-approval | ENDER-CSExp@10 | 84.78% | 10.9 | 0.015 |
| credit-approval | ENDER-CSExp@100 | 86.38% | 98.2 | 0.135 |
| credit-approval | ENDER-CSExp@20 | 85.65% | 20.7 | 0.027 |
| credit-approval | ENDER-CSExp@50 | 86.67% | 49.6 | 0.066 |
| credit-approval | ENDER-CSLog@10 | 84.49% | 10.4 | 0.017 |
| credit-approval | ENDER-CSLog@100 | 86.38% | 95.1 | 0.144 |
| credit-approval | ENDER-CSLog@20 | 85.65% | 19.9 | 0.032 |
| credit-approval | ENDER-CSLog@50 | 86.52% | 47.1 | 0.072 |
| credit-approval | ENDER-Newton@10 | 85.22% | 10.7 | 0.018 |
| credit-approval | ENDER-Newton@100 | 86.52% | 92.8 | 0.164 |
| credit-approval | ENDER-Newton@20 | 85.94% | 20.3 | 0.032 |
| credit-approval | ENDER-Newton@50 | 86.38% | 47.8 | 0.070 |
| credit-approval | Slipper@10 | 84.64% | 7.4 | 0.031 |
| credit-approval | Slipper@100 | 86.09% | 48.1 | 0.232 |
| credit-approval | Slipper@20 | 84.93% | 14.2 | 0.057 |
| credit-approval | Slipper@50 | 86.09% | 29.7 | 0.128 |
| credit-g | Boomer@10 | 73.20% | 11.0 | 0.033 |
| credit-g | Boomer@100 | 75.70% | 91.2 | 0.259 |
| credit-g | Boomer@20 | 74.00% | 20.8 | 0.068 |
| credit-g | Boomer@50 | 74.40% | 49.2 | 0.128 |
| credit-g | ENDER-CSExp@10 | 70.00% | 11.0 | 0.027 |
| credit-g | ENDER-CSExp@100 | 74.70% | 100.9 | 0.223 |
| credit-g | ENDER-CSExp@20 | 70.00% | 21.0 | 0.044 |
| credit-g | ENDER-CSExp@50 | 72.80% | 51.0 | 0.123 |
| credit-g | ENDER-CSLog@10 | 70.00% | 11.0 | 0.026 |
| credit-g | ENDER-CSLog@100 | 74.50% | 100.8 | 0.264 |
| credit-g | ENDER-CSLog@20 | 70.10% | 21.0 | 0.059 |
| credit-g | ENDER-CSLog@50 | 73.20% | 51.0 | 0.123 |
| credit-g | ENDER-Newton@10 | 70.10% | 11.0 | 0.037 |
| credit-g | ENDER-Newton@100 | 75.40% | 100.1 | 0.250 |
| credit-g | ENDER-Newton@20 | 72.10% | 21.0 | 0.058 |
| credit-g | ENDER-Newton@50 | 74.50% | 50.9 | 0.132 |
| credit-g | Slipper@10 | 73.20% | 7.6 | 0.050 |
| credit-g | Slipper@100 | 73.40% | 60.1 | 0.404 |
| credit-g | Slipper@20 | 73.50% | 14.4 | 0.090 |
| credit-g | Slipper@50 | 74.40% | 33.7 | 0.183 |
| cylinder-bands | Boomer@10 | 69.63% | 10.2 | 0.024 |
| cylinder-bands | Boomer@100 | 79.81% | 80.6 | 0.258 |
| cylinder-bands | Boomer@20 | 75.19% | 19.4 | 0.045 |
| cylinder-bands | Boomer@50 | 78.15% | 43.1 | 0.120 |
| cylinder-bands | ENDER-CSExp@10 | 58.52% | 11.0 | 0.026 |
| cylinder-bands | ENDER-CSExp@100 | 75.56% | 100.8 | 0.218 |
| cylinder-bands | ENDER-CSExp@20 | 65.74% | 21.0 | 0.047 |
| cylinder-bands | ENDER-CSExp@50 | 71.67% | 50.9 | 0.116 |
| cylinder-bands | ENDER-CSLog@10 | 58.89% | 11.0 | 0.030 |
| cylinder-bands | ENDER-CSLog@100 | 75.93% | 101.0 | 0.254 |
| cylinder-bands | ENDER-CSLog@20 | 68.52% | 21.0 | 0.053 |
| cylinder-bands | ENDER-CSLog@50 | 75.56% | 51.0 | 0.139 |
| cylinder-bands | ENDER-Newton@10 | 62.41% | 10.0 | 0.021 |
| cylinder-bands | ENDER-Newton@100 | 77.41% | 91.1 | 0.196 |
| cylinder-bands | ENDER-Newton@20 | 62.96% | 18.7 | 0.037 |
| cylinder-bands | ENDER-Newton@50 | 71.67% | 46.5 | 0.094 |
| cylinder-bands | Slipper@10 | 69.44% | 9.5 | 0.027 |
| cylinder-bands | Slipper@100 | 75.93% | 61.2 | 0.235 |
| cylinder-bands | Slipper@20 | 70.56% | 17.4 | 0.052 |
| cylinder-bands | Slipper@50 | 73.89% | 37.6 | 0.123 |
| dresses-sales | Boomer@10 | 62.00% | 10.6 | 0.023 |
| dresses-sales | Boomer@100 | 59.40% | 69.3 | 0.152 |
| dresses-sales | Boomer@20 | 63.40% | 19.8 | 0.040 |
| dresses-sales | Boomer@50 | 60.80% | 41.8 | 0.088 |
| dresses-sales | ENDER-CSExp@10 | 58.20% | 10.9 | 0.016 |
| dresses-sales | ENDER-CSExp@100 | 62.80% | 99.3 | 0.147 |
| dresses-sales | ENDER-CSExp@20 | 59.00% | 20.7 | 0.032 |
| dresses-sales | ENDER-CSExp@50 | 62.80% | 50.2 | 0.074 |
| dresses-sales | ENDER-CSLog@10 | 58.00% | 10.9 | 0.019 |
| dresses-sales | ENDER-CSLog@100 | 62.40% | 100.0 | 0.188 |
| dresses-sales | ENDER-CSLog@20 | 59.40% | 20.8 | 0.037 |
| dresses-sales | ENDER-CSLog@50 | 61.20% | 50.5 | 0.087 |
| dresses-sales | ENDER-Newton@10 | 59.80% | 11.0 | 0.017 |
| dresses-sales | ENDER-Newton@100 | 61.40% | 90.8 | 0.144 |
| dresses-sales | ENDER-Newton@20 | 62.00% | 20.7 | 0.033 |
| dresses-sales | ENDER-Newton@50 | 61.60% | 48.8 | 0.077 |
| dresses-sales | Slipper@10 | 59.20% | 8.6 | 0.022 |
| dresses-sales | Slipper@100 | 58.40% | 52.8 | 0.193 |
| dresses-sales | Slipper@20 | 58.80% | 16.2 | 0.044 |
| dresses-sales | Slipper@50 | 59.40% | 35.0 | 0.096 |
| heart-c | Boomer@10 | 81.54% | 10.8 | 0.013 |
| heart-c | Boomer@100 | 80.55% | 71.7 | 0.092 |
| heart-c | Boomer@20 | 80.90% | 20.0 | 0.024 |
| heart-c | Boomer@50 | 80.90% | 42.4 | 0.054 |
| heart-c | ENDER-CSExp@10 | 79.22% | 11.0 | 0.009 |
| heart-c | ENDER-CSExp@100 | 83.54% | 98.4 | 0.080 |
| heart-c | ENDER-CSExp@20 | 80.56% | 20.6 | 0.018 |
| heart-c | ENDER-CSExp@50 | 82.23% | 49.8 | 0.041 |
| heart-c | ENDER-CSLog@10 | 77.90% | 10.9 | 0.010 |
| heart-c | ENDER-CSLog@100 | 82.86% | 98.6 | 0.101 |
| heart-c | ENDER-CSLog@20 | 81.54% | 20.5 | 0.019 |
| heart-c | ENDER-CSLog@50 | 80.89% | 49.6 | 0.044 |
| heart-c | ENDER-Newton@10 | 79.88% | 10.8 | 0.011 |
| heart-c | ENDER-Newton@100 | 83.20% | 94.0 | 0.091 |
| heart-c | ENDER-Newton@20 | 81.54% | 20.0 | 0.021 |
| heart-c | ENDER-Newton@50 | 81.89% | 48.9 | 0.045 |
| heart-c | Slipper@10 | 78.60% | 7.5 | 0.020 |
| heart-c | Slipper@100 | 75.28% | 48.4 | 0.189 |
| heart-c | Slipper@20 | 79.25% | 13.5 | 0.038 |
| heart-c | Slipper@50 | 79.87% | 28.4 | 0.096 |
| heart-h | Boomer@10 | 79.56% | 9.0 | 0.012 |
| heart-h | Boomer@100 | 79.91% | 50.9 | 0.081 |
| heart-h | Boomer@20 | 80.59% | 17.0 | 0.023 |
| heart-h | Boomer@50 | 80.59% | 31.4 | 0.046 |
| heart-h | ENDER-CSExp@10 | 78.21% | 10.9 | 0.010 |
| heart-h | ENDER-CSExp@100 | 83.00% | 94.1 | 0.072 |
| heart-h | ENDER-CSExp@20 | 82.33% | 20.0 | 0.015 |
| heart-h | ENDER-CSExp@50 | 83.00% | 48.0 | 0.038 |
| heart-h | ENDER-CSLog@10 | 76.83% | 10.8 | 0.009 |
| heart-h | ENDER-CSLog@100 | 82.67% | 94.0 | 0.081 |
| heart-h | ENDER-CSLog@20 | 81.66% | 19.7 | 0.018 |
| heart-h | ENDER-CSLog@50 | 82.66% | 47.4 | 0.045 |
| heart-h | ENDER-Newton@10 | 78.90% | 10.6 | 0.013 |
| heart-h | ENDER-Newton@100 | 81.63% | 86.3 | 0.081 |
| heart-h | ENDER-Newton@20 | 79.92% | 19.7 | 0.017 |
| heart-h | ENDER-Newton@50 | 81.64% | 46.7 | 0.048 |
| heart-h | Slipper@10 | 78.55% | 7.4 | 0.021 |
| heart-h | Slipper@100 | 79.25% | 41.2 | 0.194 |
| heart-h | Slipper@20 | 78.90% | 14.3 | 0.037 |
| heart-h | Slipper@50 | 78.21% | 25.8 | 0.090 |
| heart-statlog | Boomer@10 | 80.37% | 10.4 | 0.012 |
| heart-statlog | Boomer@100 | 77.78% | 67.7 | 0.091 |
| heart-statlog | Boomer@20 | 79.63% | 19.8 | 0.021 |
| heart-statlog | Boomer@50 | 80.74% | 42.0 | 0.050 |
| heart-statlog | ENDER-CSExp@10 | 80.00% | 11.0 | 0.008 |
| heart-statlog | ENDER-CSExp@100 | 83.33% | 98.0 | 0.082 |
| heart-statlog | ENDER-CSExp@20 | 84.07% | 20.9 | 0.016 |
| heart-statlog | ENDER-CSExp@50 | 84.07% | 50.3 | 0.039 |
| heart-statlog | ENDER-CSLog@10 | 80.74% | 10.9 | 0.010 |
| heart-statlog | ENDER-CSLog@100 | 84.07% | 98.3 | 0.088 |
| heart-statlog | ENDER-CSLog@20 | 85.19% | 20.7 | 0.019 |
| heart-statlog | ENDER-CSLog@50 | 83.70% | 50.3 | 0.046 |
| heart-statlog | ENDER-Newton@10 | 79.63% | 10.9 | 0.010 |
| heart-statlog | ENDER-Newton@100 | 84.44% | 93.7 | 0.091 |
| heart-statlog | ENDER-Newton@20 | 83.33% | 20.6 | 0.019 |
| heart-statlog | ENDER-Newton@50 | 82.96% | 49.1 | 0.043 |
| heart-statlog | Slipper@10 | 79.26% | 7.5 | 0.020 |
| heart-statlog | Slipper@100 | 78.89% | 45.4 | 0.174 |
| heart-statlog | Slipper@20 | 81.85% | 13.5 | 0.039 |
| heart-statlog | Slipper@50 | 79.26% | 28.2 | 0.091 |
| hepatitis | Boomer@10 | 81.42% | 8.5 | 0.010 |
| hepatitis | Boomer@100 | 81.46% | 45.1 | 0.078 |
| hepatitis | Boomer@20 | 84.00% | 15.1 | 0.015 |
| hepatitis | Boomer@50 | 83.33% | 30.3 | 0.037 |
| hepatitis | ENDER-CSExp@10 | 79.38% | 10.2 | 0.007 |
| hepatitis | ENDER-CSExp@100 | 82.71% | 81.4 | 0.063 |
| hepatitis | ENDER-CSExp@20 | 79.38% | 18.7 | 0.015 |
| hepatitis | ENDER-CSExp@50 | 82.00% | 43.2 | 0.033 |
| hepatitis | ENDER-CSLog@10 | 79.38% | 10.5 | 0.009 |
| hepatitis | ENDER-CSLog@100 | 82.75% | 85.3 | 0.074 |
| hepatitis | ENDER-CSLog@20 | 80.00% | 19.2 | 0.015 |
| hepatitis | ENDER-CSLog@50 | 80.08% | 45.6 | 0.037 |
| hepatitis | ENDER-Newton@10 | 81.29% | 10.4 | 0.009 |
| hepatitis | ENDER-Newton@100 | 82.04% | 73.6 | 0.063 |
| hepatitis | ENDER-Newton@20 | 82.62% | 19.1 | 0.015 |
| hepatitis | ENDER-Newton@50 | 81.42% | 41.4 | 0.033 |
| hepatitis | Slipper@10 | 81.96% | 7.6 | 0.018 |
| hepatitis | Slipper@100 | 82.71% | 37.8 | 0.162 |
| hepatitis | Slipper@20 | 82.67% | 12.8 | 0.037 |
| hepatitis | Slipper@50 | 82.08% | 24.5 | 0.090 |
| kr-vs-kp | Boomer@10 | 93.59% | 6.0 | 0.038 |
| kr-vs-kp | Boomer@100 | 99.37% | 33.8 | 0.273 |
| kr-vs-kp | Boomer@20 | 94.09% | 11.1 | 0.069 |
| kr-vs-kp | Boomer@50 | 96.72% | 20.2 | 0.150 |
| kr-vs-kp | ENDER-CSExp@10 | 90.43% | 6.2 | 0.030 |
| kr-vs-kp | ENDER-CSExp@100 | 96.81% | 63.3 | 0.281 |
| kr-vs-kp | ENDER-CSExp@20 | 92.24% | 11.7 | 0.061 |
| kr-vs-kp | ENDER-CSExp@50 | 95.78% | 30.6 | 0.122 |
| kr-vs-kp | ENDER-CSLog@10 | 90.43% | 5.8 | 0.042 |
| kr-vs-kp | ENDER-CSLog@100 | 95.68% | 45.3 | 0.319 |
| kr-vs-kp | ENDER-CSLog@20 | 90.43% | 7.8 | 0.074 |
| kr-vs-kp | ENDER-CSLog@50 | 94.24% | 19.3 | 0.168 |
| kr-vs-kp | ENDER-Newton@10 | 90.43% | 4.2 | 0.030 |
| kr-vs-kp | ENDER-Newton@100 | 94.34% | 26.0 | 0.227 |
| kr-vs-kp | ENDER-Newton@20 | 90.43% | 6.5 | 0.054 |
| kr-vs-kp | ENDER-Newton@50 | 94.09% | 16.5 | 0.132 |
| kr-vs-kp | Slipper@10 | 96.90% | 7.6 | 0.065 |
| kr-vs-kp | Slipper@100 | 99.59% | 51.6 | 0.510 |
| kr-vs-kp | Slipper@20 | 97.69% | 14.6 | 0.117 |
| kr-vs-kp | Slipper@50 | 99.25% | 31.5 | 0.294 |
| molecular-biology_promoters | Boomer@10 | 84.09% | 9.2 | 0.012 |
| molecular-biology_promoters | Boomer@100 | 93.45% | 51.5 | 0.100 |
| molecular-biology_promoters | Boomer@20 | 87.91% | 16.8 | 0.021 |
| molecular-biology_promoters | Boomer@50 | 91.64% | 33.2 | 0.045 |
| molecular-biology_promoters | ENDER-CSExp@10 | 85.91% | 10.0 | 0.008 |
| molecular-biology_promoters | ENDER-CSExp@100 | 91.73% | 86.4 | 0.074 |
| molecular-biology_promoters | ENDER-CSExp@20 | 87.09% | 18.9 | 0.015 |
| molecular-biology_promoters | ENDER-CSExp@50 | 88.82% | 45.6 | 0.040 |
| molecular-biology_promoters | ENDER-CSLog@10 | 85.91% | 10.1 | 0.010 |
| molecular-biology_promoters | ENDER-CSLog@100 | 91.64% | 87.8 | 0.080 |
| molecular-biology_promoters | ENDER-CSLog@20 | 88.00% | 19.1 | 0.019 |
| molecular-biology_promoters | ENDER-CSLog@50 | 89.73% | 46.2 | 0.043 |
| molecular-biology_promoters | ENDER-Newton@10 | 81.09% | 10.4 | 0.010 |
| molecular-biology_promoters | ENDER-Newton@100 | 90.73% | 82.8 | 0.078 |
| molecular-biology_promoters | ENDER-Newton@20 | 87.73% | 19.5 | 0.019 |
| molecular-biology_promoters | ENDER-Newton@50 | 87.82% | 44.2 | 0.045 |
| molecular-biology_promoters | Slipper@10 | 81.18% | 7.3 | 0.033 |
| molecular-biology_promoters | Slipper@100 | 89.73% | 48.0 | 0.183 |
| molecular-biology_promoters | Slipper@20 | 86.64% | 12.5 | 0.047 |
| molecular-biology_promoters | Slipper@50 | 89.73% | 27.4 | 0.096 |
| mushroom | Boomer@10 | 98.92% | 6.7 | 0.108 |
| mushroom | Boomer@100 | 99.95% | 25.0 | 0.721 |
| mushroom | Boomer@20 | 98.93% | 10.4 | 0.196 |
| mushroom | Boomer@50 | 99.95% | 16.3 | 0.412 |
| mushroom | ENDER-CSExp@10 | 98.65% | 9.4 | 0.076 |
| mushroom | ENDER-CSExp@100 | 100.00% | 51.9 | 0.584 |
| mushroom | ENDER-CSExp@20 | 99.35% | 15.1 | 0.143 |
| mushroom | ENDER-CSExp@50 | 99.93% | 30.7 | 0.304 |
| mushroom | ENDER-CSLog@10 | 98.52% | 7.2 | 0.096 |
| mushroom | ENDER-CSLog@100 | 99.91% | 40.9 | 0.776 |
| mushroom | ENDER-CSLog@20 | 98.52% | 11.3 | 0.171 |
| mushroom | ENDER-CSLog@50 | 99.51% | 25.5 | 0.408 |
| mushroom | ENDER-Newton@10 | 96.16% | 6.1 | 0.087 |
| mushroom | ENDER-Newton@100 | 99.91% | 25.7 | 0.621 |
| mushroom | ENDER-Newton@20 | 98.88% | 10.0 | 0.160 |
| mushroom | ENDER-Newton@50 | 98.98% | 18.4 | 0.353 |
| mushroom | Slipper@10 | 100.00% | 7.6 | 0.110 |
| mushroom | Slipper@100 | 100.00% | 18.9 | 0.829 |
| mushroom | Slipper@20 | 100.00% | 10.5 | 0.245 |
| mushroom | Slipper@50 | 100.00% | 14.7 | 0.442 |
| sick | Boomer@10 | 98.12% | 9.4 | 0.062 |
| sick | Boomer@100 | 98.28% | 57.3 | 0.364 |
| sick | Boomer@20 | 98.04% | 16.5 | 0.096 |
| sick | Boomer@50 | 98.17% | 34.6 | 0.206 |
| sick | ENDER-CSExp@10 | 93.88% | 9.8 | 0.034 |
| sick | ENDER-CSExp@100 | 97.96% | 83.9 | 0.320 |
| sick | ENDER-CSExp@20 | 94.99% | 18.3 | 0.070 |
| sick | ENDER-CSExp@50 | 97.99% | 43.6 | 0.165 |
| sick | ENDER-CSLog@10 | 93.88% | 10.0 | 0.051 |
| sick | ENDER-CSLog@100 | 98.14% | 87.8 | 0.427 |
| sick | ENDER-CSLog@20 | 97.53% | 17.9 | 0.093 |
| sick | ENDER-CSLog@50 | 98.06% | 42.4 | 0.213 |
| sick | ENDER-Newton@10 | 97.88% | 10.1 | 0.045 |
| sick | ENDER-Newton@100 | 98.28% | 76.8 | 0.370 |
| sick | ENDER-Newton@20 | 97.96% | 17.8 | 0.098 |
| sick | ENDER-Newton@50 | 98.17% | 41.0 | 0.220 |
| sick | Slipper@10 | 97.77% | 7.0 | 0.071 |
| sick | Slipper@100 | 98.33% | 54.4 | 0.627 |
| sick | Slipper@20 | 97.96% | 13.5 | 0.140 |
| sick | Slipper@50 | 98.12% | 30.2 | 0.297 |
| tic-tac-toe | Boomer@10 | 79.12% | 9.0 | 0.014 |
| tic-tac-toe | Boomer@100 | 98.44% | 62.7 | 0.123 |
| tic-tac-toe | Boomer@20 | 92.38% | 15.6 | 0.026 |
| tic-tac-toe | Boomer@50 | 98.33% | 34.0 | 0.063 |
| tic-tac-toe | ENDER-CSExp@10 | 65.87% | 9.2 | 0.011 |
| tic-tac-toe | ENDER-CSExp@100 | 82.04% | 89.7 | 0.108 |
| tic-tac-toe | ENDER-CSExp@20 | 71.50% | 17.9 | 0.021 |
| tic-tac-toe | ENDER-CSExp@50 | 78.91% | 45.1 | 0.053 |
| tic-tac-toe | ENDER-CSLog@10 | 65.34% | 8.9 | 0.013 |
| tic-tac-toe | ENDER-CSLog@100 | 77.04% | 83.3 | 0.127 |
| tic-tac-toe | ENDER-CSLog@20 | 67.33% | 16.4 | 0.029 |
| tic-tac-toe | ENDER-CSLog@50 | 73.80% | 42.0 | 0.067 |
| tic-tac-toe | ENDER-Newton@10 | 72.87% | 8.8 | 0.015 |
| tic-tac-toe | ENDER-Newton@100 | 95.20% | 60.4 | 0.116 |
| tic-tac-toe | ENDER-Newton@20 | 78.60% | 13.7 | 0.026 |
| tic-tac-toe | ENDER-Newton@50 | 81.94% | 29.5 | 0.059 |
| tic-tac-toe | Slipper@10 | 97.71% | 9.0 | 0.029 |
| tic-tac-toe | Slipper@100 | 98.54% | 59.1 | 0.262 |
| tic-tac-toe | Slipper@20 | 98.33% | 15.0 | 0.060 |
| tic-tac-toe | Slipper@50 | 97.71% | 31.1 | 0.131 |
| vote | Boomer@10 | 95.65% | 6.1 | 0.007 |
| vote | Boomer@100 | 95.87% | 46.6 | 0.070 |
| vote | Boomer@20 | 95.42% | 11.2 | 0.016 |
| vote | Boomer@50 | 95.64% | 27.4 | 0.037 |
| vote | ENDER-CSExp@10 | 95.18% | 5.6 | 0.006 |
| vote | ENDER-CSExp@100 | 95.87% | 42.8 | 0.053 |
| vote | ENDER-CSExp@20 | 95.42% | 10.6 | 0.011 |
| vote | ENDER-CSExp@50 | 95.64% | 23.2 | 0.030 |
| vote | ENDER-CSLog@10 | 95.18% | 4.7 | 0.008 |
| vote | ENDER-CSLog@100 | 95.42% | 37.9 | 0.062 |
| vote | ENDER-CSLog@20 | 95.42% | 7.9 | 0.015 |
| vote | ENDER-CSLog@50 | 95.42% | 17.6 | 0.030 |
| vote | ENDER-Newton@10 | 93.80% | 6.1 | 0.007 |
| vote | ENDER-Newton@100 | 96.10% | 48.6 | 0.062 |
| vote | ENDER-Newton@20 | 95.18% | 8.2 | 0.013 |
| vote | ENDER-Newton@50 | 95.42% | 20.0 | 0.029 |
| vote | Slipper@10 | 95.43% | 6.2 | 0.020 |
| vote | Slipper@100 | 94.03% | 23.9 | 0.162 |
| vote | Slipper@20 | 94.73% | 10.9 | 0.040 |
| vote | Slipper@50 | 94.73% | 18.0 | 0.086 |

## Comparative discussion

Section 1's best covering-strategy point is `Weighted-Add+MEstimate16` (mean accuracy 84.75% over the cells every section-1 config completed). Section 2's best boosting point is `ENDER-Newton@100` (85.26%, same restriction). Boosting and weighted covering aren't as separate as they might look. `AdaBoostReweighting` (Slipper's mechanism, and `WeightedCovering`'s rule-weight-assigning option, which plain `SeCo` itself refuses to build into an unweighted set) and `ENDER`'s `ExponentialLoss` configuration are the *same* thing by a classical result (Friedman, Hastie & Tibshirani 2000): AdaBoost **is** gradient boosting on the exponential loss. `ExponentialLoss` here literally defines its per-example quantity as ``w = exp(-y f)`` -- AdaBoost's reweighting formula -- and fits each rule's weight with the identical closed form `AdaBoostReweighting.confidence` uses. At budget 100, `Slipper@100` scores 83.72% and `ENDER-CSExp@100` scores 84.66% -- close, as expected from the theory below. What actually distinguishes the two families: section 1's *other* reweighting schemes (`MultiplicativeReweighting`'s ``gamma**k``, `AdditiveReweighting`'s ``1/(k+1)``, `LRIReweighting`'s ``1 + e**3``) are hand-designed update rules with no loss function behind them, so they can't generalize the way `ENDER`'s pluggable-loss framework does (logistic, exponential, sigmoid, five weight-fitting methods). And weighted covering uses its weights only to steer training -- it still predicts with a heuristic score over an *unweighted* rule set -- while boosting bakes the fitted weight into the model itself (`LinearRuleModel`, summed at predict time): there the weight is part of the answer, not just how training got there.

## Accuracy vs. fit time

Every point here is on the same accuracy basis as the tables above (the 170 common cells), against mean fit time on a log scale, since it spans sub-second boosting fits to multi-minute (and timed-out) covering fits. Color is the algorithm family -- the four covering strategies, `CPAR`/`LRI`, and the five boosting families; for section 1, marker shape is the heuristic, so a covering strategy's three heuristic variants show up as same-color points of different shapes, joined by a dotted line in the order Laplace, m-estimate, WRAcc. A boosting family's four rule-count budgets (10/20/50/100) are connected by a line, tracing out that family's own accuracy-vs-cost curve rather than four unrelated points. `Boomer` and `ENDER-Newton`'s lines nearly overlap at the top right -- both reach similarly high accuracy at a similar cost, a real finding (the two are closely related mechanisms; see the comparative discussion above), not a plotting artifact.

![accuracy vs. fit time](covering_boosting_plots/covering_boosting_accuracy_vs_time.png)

