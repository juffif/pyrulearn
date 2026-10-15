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
(N-lists). Each fit is capped at 300s; a time-out or an error counts
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
| Removal+Laplace | 83.33% | 83.33% | 81.6 | 2.9 | 7.26 | 2.4 | 5 |
| Removal+WRAcc | 81.16% | 81.18% | 9.2 | 6.5 | 0.51 | 2.3 | 0 |
| Removal+MEstimate16 | 83.06% | 82.99% | 72.5 | 8.0 | 6.04 | 2.2 | 0 |
| Weighted-Mult+Laplace | 83.80% | 83.84% | 64.6 | 2.7 | 0.81 | 3.2 | 0 |
| Weighted-Mult+WRAcc | 81.60% | 81.64% | 36.6 | 7.0 | 4.01 | 10.6 | 0 |
| Weighted-Mult+MEstimate16 | 84.53% | 84.48% | 90.9 | 5.7 | 4.30 | 7.7 | 0 |
| Weighted-Add+Laplace | 83.36% | 83.41% | 49.9 | 2.7 | 0.77 | 2.8 | 0 |
| Weighted-Add+WRAcc | 82.11% | 82.11% | 39.7 | 8.6 | 10.04 | 14.0 | 5 |
| Weighted-Add+MEstimate16 | 84.70% | 84.74% | 66.2 | 5.6 | 5.29 | 8.1 | 0 |
| Weighted-LRI+Laplace | 81.76% | 81.80% | 24.2 | 3.0 | 0.42 | 2.2 | 0 |
| Weighted-LRI+WRAcc | 82.67% | 82.69% | 74.7 | 7.2 | 6.62 | 17.4 | 0 |
| Weighted-LRI+MEstimate16 | 81.41% | 81.44% | 20.2 | 2.5 | 0.35 | 1.6 | 0 |
| CPAR | 84.07% | 84.12% | 240.4 | 6.6 | 1.95 | 9.0 | 0 |
| LRI | 84.27% | 84.20% | 397.2 | 3.9 | 1.16 | 60.0 | 0 |

*Removal+Laplace missing 5 of 175 cells; Weighted-Add+WRAcc missing 5 of 175 cells (see the "failures" column). "own cells" is each name averaged over whatever (dataset, fold) cells it succeeded on; "170 common cells" restricts every name to the same cells, for a fair comparison.*


### Mean rank

| name | mean rank |
|---|--:|
| Weighted-Add+MEstimate16 | 4.17 |
| Weighted-Mult+MEstimate16 | 4.19 |
| LRI | 5.42 |
| CPAR | 5.64 |
| Weighted-Mult+Laplace | 6.64 |
| Weighted-LRI+WRAcc | 7.11 |
| Removal+MEstimate16 | 7.17 |
| Removal+Laplace | 7.53 |
| Weighted-Add+Laplace | 8.17 |
| Weighted-LRI+MEstimate16 | 9.06 |
| Weighted-Add+WRAcc | 9.28 |
| Weighted-LRI+Laplace | 9.67 |
| Removal+WRAcc | 10.33 |
| Weighted-Mult+WRAcc | 10.64 |

### Per-dataset results (mean across folds)

| dataset | learner | accuracy | n_rules | fit_time | overlap |
|---|---|---|---|---|---|
| SPECT | CPAR | 82.78% | 107.8 | 0.169 | 10.5 |
| SPECT | LRI | 80.81% | 399.8 | 0.526 | 49.6 |
| SPECT | Removal+Laplace | 82.36% | 49.3 | 0.447 | 3.9 |
| SPECT | Removal+MEstimate16 | 81.27% | 19.8 | 0.238 | 2.5 |
| SPECT | Removal+WRAcc | 79.43% | 9.6 | 0.095 | 2.7 |
| SPECT | Weighted-Add+Laplace | 85.00% | 35.2 | 0.630 | 3.8 |
| SPECT | Weighted-Add+MEstimate16 | 84.62% | 50.8 | 1.633 | 8.9 |
| SPECT | Weighted-Add+WRAcc | 84.64% | 29.4 | 0.708 | 8.2 |
| SPECT | Weighted-LRI+Laplace | 79.34% | 33.2 | 0.511 | 2.6 |
| SPECT | Weighted-LRI+MEstimate16 | 77.92% | 28.0 | 0.354 | 2.2 |
| SPECT | Weighted-LRI+WRAcc | 83.18% | 66.7 | 0.893 | 12.5 |
| SPECT | Weighted-Mult+Laplace | 84.63% | 48.1 | 0.828 | 4.8 |
| SPECT | Weighted-Mult+MEstimate16 | 83.86% | 71.7 | 1.205 | 9.5 |
| SPECT | Weighted-Mult+WRAcc | 79.05% | 49.0 | 0.607 | 10.9 |
| adult | CPAR | 82.18% | 2097.0 | 55.568 | 14.1 |
| adult | LRI | 86.93% | 400.0 | 13.285 | 59.7 |
| adult | Removal+Laplace | n/a | n/a | 300.136 | n/a |
| adult | Removal+MEstimate16 | 85.75% | 1525.0 | 190.843 | 4.3 |
| adult | Removal+WRAcc | 80.54% | 16.6 | 7.874 | 5.0 |
| adult | Weighted-Add+Laplace | 81.60% | 87.4 | 7.301 | 1.6 |
| adult | Weighted-Add+MEstimate16 | 83.18% | 132.8 | 40.840 | 5.6 |
| adult | Weighted-Add+WRAcc | n/a | n/a | 300.119 | n/a |
| adult | Weighted-LRI+Laplace | 80.29% | 27.0 | 3.344 | 0.7 |
| adult | Weighted-LRI+MEstimate16 | 80.30% | 19.6 | 2.547 | 0.3 |
| adult | Weighted-LRI+WRAcc | 81.70% | 90.4 | 100.003 | 26.7 |
| adult | Weighted-Mult+Laplace | 82.43% | 172.4 | 9.478 | 1.8 |
| adult | Weighted-Mult+MEstimate16 | 86.06% | 198.6 | 37.311 | 3.3 |
| adult | Weighted-Mult+WRAcc | 80.32% | 73.6 | 81.909 | 26.0 |
| breast-cancer | CPAR | 73.42% | 216.1 | 0.377 | 10.8 |
| breast-cancer | LRI | 70.64% | 395.0 | 0.468 | 60.8 |
| breast-cancer | Removal+Laplace | 67.14% | 85.3 | 0.336 | 1.7 |
| breast-cancer | Removal+MEstimate16 | 68.26% | 32.0 | 0.464 | 1.9 |
| breast-cancer | Removal+WRAcc | 71.66% | 8.9 | 0.175 | 2.4 |
| breast-cancer | Weighted-Add+Laplace | 71.67% | 58.1 | 0.412 | 2.2 |
| breast-cancer | Weighted-Add+MEstimate16 | 74.15% | 100.6 | 3.503 | 10.5 |
| breast-cancer | Weighted-Add+WRAcc | 73.44% | 55.0 | 2.904 | 19.6 |
| breast-cancer | Weighted-LRI+Laplace | 69.22% | 26.3 | 0.304 | 1.3 |
| breast-cancer | Weighted-LRI+MEstimate16 | 68.50% | 21.3 | 0.229 | 0.7 |
| breast-cancer | Weighted-LRI+WRAcc | 72.75% | 96.3 | 2.192 | 22.6 |
| breast-cancer | Weighted-Mult+Laplace | 72.01% | 70.5 | 0.425 | 2.3 |
| breast-cancer | Weighted-Mult+MEstimate16 | 73.45% | 115.1 | 2.636 | 9.2 |
| breast-cancer | Weighted-Mult+WRAcc | 72.04% | 40.2 | 0.925 | 11.7 |
| colic | CPAR | 83.98% | 157.0 | 0.261 | 9.9 |
| colic | LRI | 85.07% | 400.0 | 0.737 | 65.4 |
| colic | Removal+Laplace | 82.91% | 79.2 | 0.240 | 3.2 |
| colic | Removal+MEstimate16 | 84.54% | 26.0 | 0.290 | 2.9 |
| colic | Removal+WRAcc | 86.43% | 10.4 | 0.145 | 2.2 |
| colic | Weighted-Add+Laplace | 81.82% | 70.6 | 0.469 | 4.3 |
| colic | Weighted-Add+MEstimate16 | 86.69% | 71.8 | 3.620 | 11.6 |
| colic | Weighted-Add+WRAcc | 83.69% | 30.7 | 3.275 | 9.1 |
| colic | Weighted-LRI+Laplace | 80.72% | 32.9 | 0.300 | 2.6 |
| colic | Weighted-LRI+MEstimate16 | 80.45% | 21.9 | 0.234 | 1.4 |
| colic | Weighted-LRI+WRAcc | 85.08% | 80.4 | 1.433 | 15.2 |
| colic | Weighted-Mult+Laplace | 81.53% | 70.8 | 0.386 | 3.6 |
| colic | Weighted-Mult+MEstimate16 | 83.71% | 91.3 | 2.014 | 8.2 |
| colic | Weighted-Mult+WRAcc | 83.70% | 42.9 | 1.020 | 9.0 |
| credit-approval | CPAR | 85.22% | 367.5 | 0.557 | 15.6 |
| credit-approval | LRI | 87.25% | 398.7 | 0.730 | 59.3 |
| credit-approval | Removal+Laplace | 82.46% | 120.0 | 0.503 | 2.4 |
| credit-approval | Removal+MEstimate16 | 85.36% | 34.5 | 0.836 | 2.6 |
| credit-approval | Removal+WRAcc | 83.62% | 9.0 | 0.280 | 2.4 |
| credit-approval | Weighted-Add+Laplace | 84.78% | 69.2 | 0.664 | 5.1 |
| credit-approval | Weighted-Add+MEstimate16 | 86.52% | 79.3 | 8.105 | 15.1 |
| credit-approval | Weighted-Add+WRAcc | 85.80% | 44.2 | 8.005 | 18.8 |
| credit-approval | Weighted-LRI+Laplace | 75.94% | 31.2 | 0.423 | 3.4 |
| credit-approval | Weighted-LRI+MEstimate16 | 79.28% | 17.8 | 0.313 | 1.4 |
| credit-approval | Weighted-LRI+WRAcc | 86.38% | 90.7 | 3.508 | 20.6 |
| credit-approval | Weighted-Mult+Laplace | 83.77% | 81.4 | 0.573 | 4.2 |
| credit-approval | Weighted-Mult+MEstimate16 | 86.09% | 134.9 | 5.586 | 12.8 |
| credit-approval | Weighted-Mult+WRAcc | 85.07% | 44.4 | 1.980 | 13.5 |
| credit-g | CPAR | 74.40% | 523.3 | 1.202 | 10.6 |
| credit-g | LRI | 75.30% | 400.0 | 0.854 | 58.0 |
| credit-g | Removal+Laplace | 74.40% | 232.1 | 0.979 | 2.3 |
| credit-g | Removal+MEstimate16 | 73.00% | 76.4 | 1.393 | 1.8 |
| credit-g | Removal+WRAcc | 71.70% | 13.0 | 0.447 | 3.0 |
| credit-g | Weighted-Add+Laplace | 73.80% | 102.4 | 0.642 | 2.2 |
| credit-g | Weighted-Add+MEstimate16 | 75.10% | 158.5 | 4.363 | 7.5 |
| credit-g | Weighted-Add+WRAcc | 71.20% | 117.4 | 7.713 | 37.7 |
| credit-g | Weighted-LRI+Laplace | 72.70% | 25.3 | 0.305 | 0.7 |
| credit-g | Weighted-LRI+MEstimate16 | 74.00% | 21.1 | 0.321 | 0.5 |
| credit-g | Weighted-LRI+WRAcc | 70.30% | 100.0 | 3.970 | 21.5 |
| credit-g | Weighted-Mult+Laplace | 75.20% | 133.5 | 0.675 | 2.5 |
| credit-g | Weighted-Mult+MEstimate16 | 75.10% | 190.2 | 4.567 | 6.3 |
| credit-g | Weighted-Mult+WRAcc | 70.80% | 52.2 | 2.218 | 14.0 |
| cylinder-bands | CPAR | 77.04% | 284.8 | 0.523 | 10.9 |
| cylinder-bands | LRI | 82.22% | 399.4 | 0.853 | 56.2 |
| cylinder-bands | Removal+Laplace | 74.81% | 155.5 | 0.503 | 1.7 |
| cylinder-bands | Removal+MEstimate16 | 74.44% | 49.9 | 2.098 | 1.6 |
| cylinder-bands | Removal+WRAcc | 73.70% | 9.0 | 1.735 | 2.2 |
| cylinder-bands | Weighted-Add+Laplace | 71.67% | 75.5 | 0.379 | 1.4 |
| cylinder-bands | Weighted-Add+MEstimate16 | 75.74% | 102.8 | 13.706 | 6.5 |
| cylinder-bands | Weighted-Add+WRAcc | 72.59% | 69.0 | 20.283 | 27.4 |
| cylinder-bands | Weighted-LRI+Laplace | 65.93% | 21.6 | 0.235 | 0.7 |
| cylinder-bands | Weighted-LRI+MEstimate16 | 65.93% | 17.4 | 0.185 | 0.4 |
| cylinder-bands | Weighted-LRI+WRAcc | 73.33% | 99.0 | 22.896 | 21.4 |
| cylinder-bands | Weighted-Mult+Laplace | 72.22% | 88.9 | 0.387 | 1.4 |
| cylinder-bands | Weighted-Mult+MEstimate16 | 75.56% | 142.8 | 13.511 | 7.4 |
| cylinder-bands | Weighted-Mult+WRAcc | 71.67% | 32.6 | 8.140 | 11.4 |
| dresses-sales | CPAR | 57.00% | 659.0 | 1.081 | 13.0 |
| dresses-sales | LRI | 58.80% | 394.6 | 0.662 | 58.5 |
| dresses-sales | Removal+Laplace | 59.60% | 226.3 | 0.791 | 1.6 |
| dresses-sales | Removal+MEstimate16 | 57.60% | 71.0 | 2.218 | 1.7 |
| dresses-sales | Removal+WRAcc | 56.20% | 12.2 | 0.885 | 2.9 |
| dresses-sales | Weighted-Add+Laplace | 58.60% | 80.3 | 0.273 | 0.7 |
| dresses-sales | Weighted-Add+MEstimate16 | 60.20% | 138.9 | 7.712 | 4.9 |
| dresses-sales | Weighted-Add+WRAcc | 58.40% | 112.7 | 13.986 | 33.3 |
| dresses-sales | Weighted-LRI+Laplace | 62.20% | 28.2 | 0.237 | 0.6 |
| dresses-sales | Weighted-LRI+MEstimate16 | 58.00% | 27.0 | 0.213 | 0.3 |
| dresses-sales | Weighted-LRI+WRAcc | 59.40% | 99.5 | 9.258 | 22.2 |
| dresses-sales | Weighted-Mult+Laplace | 61.40% | 87.0 | 0.282 | 0.7 |
| dresses-sales | Weighted-Mult+MEstimate16 | 59.20% | 165.8 | 7.813 | 4.9 |
| dresses-sales | Weighted-Mult+WRAcc | 56.60% | 49.4 | 4.449 | 12.9 |
| heart-c | CPAR | 80.55% | 141.9 | 0.223 | 10.4 |
| heart-c | LRI | 78.27% | 399.6 | 0.567 | 59.0 |
| heart-c | Removal+Laplace | 81.23% | 58.6 | 0.184 | 2.6 |
| heart-c | Removal+MEstimate16 | 78.24% | 20.7 | 0.264 | 2.1 |
| heart-c | Removal+WRAcc | 76.60% | 8.3 | 0.129 | 2.1 |
| heart-c | Weighted-Add+Laplace | 79.57% | 51.4 | 0.434 | 3.8 |
| heart-c | Weighted-Add+MEstimate16 | 81.28% | 69.7 | 3.633 | 14.6 |
| heart-c | Weighted-Add+WRAcc | 76.92% | 27.7 | 2.601 | 10.9 |
| heart-c | Weighted-LRI+Laplace | 77.90% | 25.2 | 0.270 | 2.6 |
| heart-c | Weighted-LRI+MEstimate16 | 79.51% | 20.1 | 0.262 | 1.7 |
| heart-c | Weighted-LRI+WRAcc | 79.92% | 88.4 | 1.660 | 20.7 |
| heart-c | Weighted-Mult+Laplace | 80.24% | 66.3 | 0.417 | 4.0 |
| heart-c | Weighted-Mult+MEstimate16 | 79.96% | 82.6 | 1.799 | 10.7 |
| heart-c | Weighted-Mult+WRAcc | 78.92% | 33.5 | 0.710 | 12.6 |
| heart-h | CPAR | 80.98% | 166.8 | 0.244 | 10.7 |
| heart-h | LRI | 81.64% | 396.6 | 0.501 | 56.5 |
| heart-h | Removal+Laplace | 78.87% | 58.4 | 0.247 | 2.2 |
| heart-h | Removal+MEstimate16 | 82.64% | 21.1 | 0.345 | 2.0 |
| heart-h | Removal+WRAcc | 78.52% | 7.9 | 0.158 | 1.9 |
| heart-h | Weighted-Add+Laplace | 78.89% | 34.7 | 0.360 | 2.5 |
| heart-h | Weighted-Add+MEstimate16 | 79.60% | 56.5 | 4.398 | 11.4 |
| heart-h | Weighted-Add+WRAcc | 77.53% | 30.1 | 4.644 | 13.9 |
| heart-h | Weighted-LRI+Laplace | 79.91% | 25.1 | 0.331 | 2.6 |
| heart-h | Weighted-LRI+MEstimate16 | 80.25% | 17.6 | 0.177 | 1.4 |
| heart-h | Weighted-LRI+WRAcc | 81.30% | 83.3 | 2.145 | 19.1 |
| heart-h | Weighted-Mult+Laplace | 78.89% | 36.1 | 0.249 | 2.3 |
| heart-h | Weighted-Mult+MEstimate16 | 80.29% | 78.2 | 2.362 | 10.1 |
| heart-h | Weighted-Mult+WRAcc | 76.16% | 36.3 | 0.994 | 12.8 |
| heart-statlog | CPAR | 82.22% | 126.4 | 0.206 | 9.4 |
| heart-statlog | LRI | 76.30% | 399.6 | 0.547 | 59.3 |
| heart-statlog | Removal+Laplace | 78.52% | 50.4 | 0.151 | 2.5 |
| heart-statlog | Removal+MEstimate16 | 78.15% | 18.5 | 0.261 | 2.1 |
| heart-statlog | Removal+WRAcc | 78.15% | 8.4 | 0.125 | 2.2 |
| heart-statlog | Weighted-Add+Laplace | 80.00% | 52.7 | 0.408 | 4.0 |
| heart-statlog | Weighted-Add+MEstimate16 | 80.00% | 61.9 | 3.751 | 12.9 |
| heart-statlog | Weighted-Add+WRAcc | 78.15% | 33.4 | 2.596 | 13.1 |
| heart-statlog | Weighted-LRI+Laplace | 78.15% | 26.9 | 0.278 | 2.8 |
| heart-statlog | Weighted-LRI+MEstimate16 | 79.26% | 22.6 | 0.244 | 1.9 |
| heart-statlog | Weighted-LRI+WRAcc | 80.37% | 88.6 | 1.716 | 21.0 |
| heart-statlog | Weighted-Mult+Laplace | 80.37% | 62.5 | 0.376 | 3.9 |
| heart-statlog | Weighted-Mult+MEstimate16 | 80.37% | 72.9 | 1.674 | 10.5 |
| heart-statlog | Weighted-Mult+WRAcc | 74.44% | 33.4 | 0.729 | 12.4 |
| hepatitis | CPAR | 80.71% | 69.7 | 0.091 | 8.4 |
| hepatitis | LRI | 82.12% | 399.4 | 0.471 | 64.7 |
| hepatitis | Removal+Laplace | 81.38% | 24.5 | 0.063 | 2.6 |
| hepatitis | Removal+MEstimate16 | 76.25% | 14.3 | 0.094 | 2.6 |
| hepatitis | Removal+WRAcc | 81.33% | 7.8 | 0.071 | 2.6 |
| hepatitis | Weighted-Add+Laplace | 82.62% | 23.6 | 0.304 | 3.8 |
| hepatitis | Weighted-Add+MEstimate16 | 83.25% | 38.3 | 1.691 | 10.1 |
| hepatitis | Weighted-Add+WRAcc | 81.33% | 25.0 | 1.003 | 11.8 |
| hepatitis | Weighted-LRI+Laplace | 85.17% | 20.8 | 0.209 | 3.9 |
| hepatitis | Weighted-LRI+MEstimate16 | 83.33% | 17.1 | 0.193 | 2.4 |
| hepatitis | Weighted-LRI+WRAcc | 83.87% | 63.1 | 0.753 | 15.9 |
| hepatitis | Weighted-Mult+Laplace | 84.58% | 29.3 | 0.274 | 4.2 |
| hepatitis | Weighted-Mult+MEstimate16 | 81.92% | 45.8 | 0.567 | 8.9 |
| hepatitis | Weighted-Mult+WRAcc | 80.62% | 30.7 | 0.366 | 10.4 |
| kr-vs-kp | CPAR | 99.28% | 50.6 | 0.319 | 3.8 |
| kr-vs-kp | LRI | 99.12% | 399.7 | 1.186 | 62.6 |
| kr-vs-kp | Removal+Laplace | 99.34% | 68.6 | 0.446 | 2.2 |
| kr-vs-kp | Removal+MEstimate16 | 99.22% | 32.3 | 0.428 | 1.7 |
| kr-vs-kp | Removal+WRAcc | 94.68% | 7.5 | 0.150 | 1.6 |
| kr-vs-kp | Weighted-Add+Laplace | 97.03% | 26.5 | 0.695 | 1.6 |
| kr-vs-kp | Weighted-Add+MEstimate16 | 98.50% | 27.1 | 2.569 | 2.7 |
| kr-vs-kp | Weighted-Add+WRAcc | 92.49% | 11.1 | 3.553 | 4.4 |
| kr-vs-kp | Weighted-LRI+Laplace | 92.34% | 11.9 | 0.316 | 1.2 |
| kr-vs-kp | Weighted-LRI+MEstimate16 | 86.08% | 16.6 | 0.285 | 1.1 |
| kr-vs-kp | Weighted-LRI+WRAcc | 95.90% | 35.0 | 1.909 | 9.2 |
| kr-vs-kp | Weighted-Mult+Laplace | 98.44% | 44.3 | 0.606 | 1.9 |
| kr-vs-kp | Weighted-Mult+MEstimate16 | 99.41% | 56.5 | 3.347 | 4.2 |
| kr-vs-kp | Weighted-Mult+WRAcc | 92.18% | 18.0 | 0.955 | 8.1 |
| molecular-biology_promoters | CPAR | 80.36% | 31.5 | 0.049 | 5.3 |
| molecular-biology_promoters | LRI | 82.91% | 390.5 | 0.496 | 62.1 |
| molecular-biology_promoters | Removal+Laplace | 82.09% | 11.9 | 0.036 | 1.6 |
| molecular-biology_promoters | Removal+MEstimate16 | 79.36% | 6.9 | 0.039 | 1.5 |
| molecular-biology_promoters | Removal+WRAcc | 74.91% | 5.1 | 0.047 | 1.2 |
| molecular-biology_promoters | Weighted-Add+Laplace | 85.00% | 21.8 | 0.529 | 3.3 |
| molecular-biology_promoters | Weighted-Add+MEstimate16 | 84.09% | 19.6 | 0.534 | 5.1 |
| molecular-biology_promoters | Weighted-Add+WRAcc | 85.91% | 16.4 | 0.410 | 5.5 |
| molecular-biology_promoters | Weighted-LRI+Laplace | 86.82% | 25.2 | 0.311 | 4.0 |
| molecular-biology_promoters | Weighted-LRI+MEstimate16 | 85.09% | 27.2 | 0.336 | 4.9 |
| molecular-biology_promoters | Weighted-LRI+WRAcc | 78.45% | 74.2 | 1.077 | 21.0 |
| molecular-biology_promoters | Weighted-Mult+Laplace | 82.18% | 42.5 | 0.493 | 5.3 |
| molecular-biology_promoters | Weighted-Mult+MEstimate16 | 84.09% | 25.9 | 0.209 | 6.3 |
| molecular-biology_promoters | Weighted-Mult+WRAcc | 85.00% | 21.7 | 0.266 | 6.4 |
| mushroom | CPAR | 99.96% | 21.3 | 0.277 | 5.1 |
| mushroom | LRI | 100.00% | 386.9 | 2.439 | 67.7 |
| mushroom | Removal+Laplace | 100.00% | 23.0 | 0.171 | 1.5 |
| mushroom | Removal+MEstimate16 | 100.00% | 18.6 | 0.147 | 1.6 |
| mushroom | Removal+WRAcc | 99.98% | 4.5 | 0.157 | 1.1 |
| mushroom | Weighted-Add+Laplace | 98.98% | 15.1 | 0.697 | 0.9 |
| mushroom | Weighted-Add+MEstimate16 | 98.98% | 19.9 | 1.186 | 3.0 |
| mushroom | Weighted-Add+WRAcc | 99.79% | 6.5 | 3.607 | 2.8 |
| mushroom | Weighted-LRI+Laplace | 95.72% | 9.0 | 0.375 | 0.8 |
| mushroom | Weighted-LRI+MEstimate16 | 95.72% | 9.0 | 0.387 | 0.8 |
| mushroom | Weighted-LRI+WRAcc | 99.94% | 31.6 | 7.610 | 11.5 |
| mushroom | Weighted-Mult+Laplace | 99.75% | 25.1 | 0.747 | 1.7 |
| mushroom | Weighted-Mult+MEstimate16 | 100.00% | 28.2 | 1.385 | 4.0 |
| mushroom | Weighted-Mult+WRAcc | 99.79% | 8.3 | 2.173 | 3.3 |
| sick | CPAR | 98.44% | 143.6 | 0.562 | 7.3 |
| sick | LRI | 98.49% | 392.0 | 1.306 | 69.4 |
| sick | Removal+Laplace | 97.93% | 86.9 | 0.997 | 2.7 |
| sick | Removal+MEstimate16 | 98.04% | 30.1 | 0.931 | 2.9 |
| sick | Removal+WRAcc | 97.19% | 10.6 | 0.356 | 2.2 |
| sick | Weighted-Add+Laplace | 97.93% | 39.5 | 1.885 | 3.1 |
| sick | Weighted-Add+MEstimate16 | 98.12% | 45.0 | 9.938 | 6.8 |
| sick | Weighted-Add+WRAcc | 97.56% | 22.8 | 7.264 | 10.1 |
| sick | Weighted-LRI+Laplace | 97.40% | 23.1 | 0.690 | 2.2 |
| sick | Weighted-LRI+MEstimate16 | 98.04% | 17.3 | 0.480 | 0.7 |
| sick | Weighted-LRI+WRAcc | 96.82% | 60.9 | 3.675 | 10.4 |
| sick | Weighted-Mult+Laplace | 98.04% | 66.9 | 1.672 | 4.7 |
| sick | Weighted-Mult+MEstimate16 | 97.64% | 97.6 | 6.668 | 9.4 |
| sick | Weighted-Mult+WRAcc | 95.68% | 41.5 | 2.938 | 9.8 |
| tic-tac-toe | CPAR | 98.33% | 57.2 | 0.187 | 2.9 |
| tic-tac-toe | LRI | 96.97% | 400.0 | 0.752 | 50.7 |
| tic-tac-toe | Removal+Laplace | 98.33% | 28.6 | 0.113 | 1.3 |
| tic-tac-toe | Removal+MEstimate16 | 98.96% | 20.9 | 0.111 | 1.3 |
| tic-tac-toe | Removal+WRAcc | 81.20% | 12.3 | 0.085 | 2.4 |
| tic-tac-toe | Weighted-Add+Laplace | 96.87% | 52.6 | 0.656 | 2.7 |
| tic-tac-toe | Weighted-Add+MEstimate16 | 97.91% | 35.3 | 0.790 | 2.5 |
| tic-tac-toe | Weighted-Add+WRAcc | 83.92% | 33.5 | 0.681 | 7.0 |
| tic-tac-toe | Weighted-LRI+Laplace | 97.39% | 26.2 | 0.328 | 1.6 |
| tic-tac-toe | Weighted-LRI+MEstimate16 | 97.50% | 24.5 | 0.354 | 1.6 |
| tic-tac-toe | Weighted-LRI+WRAcc | 83.61% | 60.0 | 0.684 | 15.5 |
| tic-tac-toe | Weighted-Mult+Laplace | 97.39% | 57.1 | 0.660 | 2.8 |
| tic-tac-toe | Weighted-Mult+MEstimate16 | 99.69% | 51.0 | 0.711 | 2.9 |
| tic-tac-toe | Weighted-Mult+WRAcc | 90.91% | 38.8 | 0.363 | 6.6 |
| vote | CPAR | 95.41% | 34.6 | 0.075 | 6.5 |
| vote | LRI | 95.41% | 400.0 | 0.547 | 60.9 |
| vote | Removal+Laplace | 95.18% | 28.1 | 0.098 | 4.1 |
| vote | Removal+MEstimate16 | 95.42% | 13.6 | 0.055 | 3.1 |
| vote | Removal+WRAcc | 94.74% | 8.3 | 0.024 | 2.3 |
| vote | Weighted-Add+Laplace | 93.81% | 19.8 | 0.459 | 3.7 |
| vote | Weighted-Add+MEstimate16 | 95.87% | 16.6 | 1.081 | 4.9 |
| vote | Weighted-Add+WRAcc | 92.42% | 9.6 | 0.488 | 4.2 |
| vote | Weighted-LRI+Laplace | 93.81% | 18.1 | 0.267 | 4.0 |
| vote | Weighted-LRI+MEstimate16 | 95.65% | 17.9 | 0.247 | 4.2 |
| vote | Weighted-LRI+WRAcc | 95.20% | 43.5 | 0.433 | 11.7 |
| vote | Weighted-Mult+Laplace | 94.73% | 34.8 | 0.439 | 5.2 |
| vote | Weighted-Mult+MEstimate16 | 95.88% | 41.4 | 0.567 | 6.9 |
| vote | Weighted-Mult+WRAcc | 95.19% | 31.2 | 0.341 | 6.5 |

## Section 2: boosting

| family | budget | accuracy (own cells) | accuracy (170 common cells) | n_rules (actual) | fit_time (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| Slipper | 10 | 83.24% | 83.18% | 7.6 | 0.05 | 0 |
| Slipper | 20 | 83.82% | 83.75% | 13.7 | 0.10 | 0 |
| Slipper | 50 | 84.00% | 83.92% | 28.5 | 0.23 | 0 |
| Slipper | 100 | 83.75% | 83.67% | 47.1 | 0.45 | 0 |
| ENDER-CSLog | 10 | 79.14% | 79.23% | 9.6 | 0.04 | 0 |
| ENDER-CSLog | 20 | 81.40% | 81.39% | 17.5 | 0.07 | 0 |
| ENDER-CSLog | 50 | 83.39% | 83.35% | 42.0 | 0.17 | 0 |
| ENDER-CSLog | 100 | 84.23% | 84.19% | 83.0 | 0.33 | 0 |
| ENDER-CSExp | 10 | 79.19% | 79.28% | 9.9 | 0.03 | 0 |
| ENDER-CSExp | 20 | 81.25% | 81.25% | 18.3 | 0.06 | 0 |
| ENDER-CSExp | 50 | 83.82% | 83.79% | 43.6 | 0.14 | 0 |
| ENDER-CSExp | 100 | 84.68% | 84.65% | 84.6 | 0.27 | 0 |
| ENDER-Newton | 10 | 80.21% | 80.27% | 9.5 | 0.04 | 0 |
| ENDER-Newton | 20 | 82.08% | 82.14% | 17.1 | 0.07 | 0 |
| ENDER-Newton | 50 | 83.57% | 83.53% | 40.1 | 0.15 | 0 |
| ENDER-Newton | 100 | 85.25% | 85.23% | 76.1 | 0.30 | 0 |
| Boomer | 10 | 82.56% | 82.54% | 9.1 | 0.05 | 0 |
| Boomer | 20 | 84.24% | 84.20% | 16.6 | 0.08 | 0 |
| Boomer | 50 | 85.00% | 84.96% | 35.2 | 0.18 | 0 |
| Boomer | 100 | 84.95% | 84.90% | 60.3 | 0.34 | 0 |

*"own cells" is each name averaged over whatever (dataset, fold) cells it succeeded on; "170 common cells" is the same cell subset section 1's summary table uses, so the two tables' accuracy columns are directly comparable (see the "failures" column).*


### Per-dataset results (mean across folds)

| dataset | learner | accuracy | n_rules | fit_time |
|---|---|---|---|---|
| SPECT | Boomer@10 | 82.81% | 9.2 | 0.012 |
| SPECT | Boomer@100 | 83.49% | 46.6 | 0.084 |
| SPECT | Boomer@20 | 83.90% | 14.6 | 0.019 |
| SPECT | Boomer@50 | 83.50% | 29.0 | 0.043 |
| SPECT | ENDER-CSExp@10 | 79.42% | 9.7 | 0.009 |
| SPECT | ENDER-CSExp@100 | 85.00% | 53.9 | 0.066 |
| SPECT | ENDER-CSExp@20 | 79.42% | 16.2 | 0.016 |
| SPECT | ENDER-CSExp@50 | 83.18% | 32.2 | 0.035 |
| SPECT | ENDER-CSLog@10 | 79.42% | 9.6 | 0.011 |
| SPECT | ENDER-CSLog@100 | 84.26% | 62.3 | 0.076 |
| SPECT | ENDER-CSLog@20 | 79.42% | 15.9 | 0.018 |
| SPECT | ENDER-CSLog@50 | 84.27% | 33.1 | 0.040 |
| SPECT | ENDER-Newton@10 | 79.42% | 10.3 | 0.011 |
| SPECT | ENDER-Newton@100 | 85.01% | 68.0 | 0.083 |
| SPECT | ENDER-Newton@20 | 79.42% | 17.6 | 0.019 |
| SPECT | ENDER-Newton@50 | 83.90% | 38.2 | 0.043 |
| SPECT | Slipper@10 | 84.25% | 5.9 | 0.031 |
| SPECT | Slipper@100 | 81.57% | 44.1 | 0.271 |
| SPECT | Slipper@20 | 83.13% | 11.2 | 0.058 |
| SPECT | Slipper@50 | 81.60% | 26.6 | 0.135 |
| adult | Boomer@10 | 83.26% | 11.0 | 0.732 |
| adult | Boomer@100 | 86.70% | 99.2 | 5.458 |
| adult | Boomer@20 | 85.66% | 20.6 | 1.236 |
| adult | Boomer@50 | 86.31% | 49.6 | 2.935 |
| adult | ENDER-CSExp@10 | 76.07% | 10.8 | 0.496 |
| adult | ENDER-CSExp@100 | 85.82% | 99.4 | 3.839 |
| adult | ENDER-CSExp@20 | 81.10% | 20.8 | 0.892 |
| adult | ENDER-CSExp@50 | 84.82% | 50.2 | 2.028 |
| adult | ENDER-CSLog@10 | 76.07% | 10.6 | 0.635 |
| adult | ENDER-CSLog@100 | 85.71% | 96.2 | 5.085 |
| adult | ENDER-CSLog@20 | 82.01% | 20.2 | 1.132 |
| adult | ENDER-CSLog@50 | 84.60% | 47.4 | 2.576 |
| adult | ENDER-Newton@10 | 78.32% | 10.4 | 0.609 |
| adult | ENDER-Newton@100 | 85.94% | 96.4 | 4.782 |
| adult | ENDER-Newton@20 | 80.17% | 19.8 | 1.084 |
| adult | ENDER-Newton@50 | 85.18% | 48.2 | 2.414 |
| adult | Slipper@10 | 85.45% | 7.2 | 0.755 |
| adult | Slipper@100 | 86.63% | 69.8 | 6.098 |
| adult | Slipper@20 | 86.08% | 14.2 | 1.398 |
| adult | Slipper@50 | 86.68% | 36.2 | 3.182 |
| breast-cancer | Boomer@10 | 72.38% | 9.8 | 0.015 |
| breast-cancer | Boomer@100 | 68.17% | 57.7 | 0.102 |
| breast-cancer | Boomer@20 | 71.35% | 16.5 | 0.027 |
| breast-cancer | Boomer@50 | 69.91% | 34.6 | 0.056 |
| breast-cancer | ENDER-CSExp@10 | 70.30% | 11.0 | 0.011 |
| breast-cancer | ENDER-CSExp@100 | 72.75% | 97.4 | 0.099 |
| breast-cancer | ENDER-CSExp@20 | 70.65% | 20.9 | 0.020 |
| breast-cancer | ENDER-CSExp@50 | 74.48% | 50.0 | 0.051 |
| breast-cancer | ENDER-CSLog@10 | 70.30% | 11.0 | 0.013 |
| breast-cancer | ENDER-CSLog@100 | 72.76% | 97.5 | 0.114 |
| breast-cancer | ENDER-CSLog@20 | 71.00% | 20.7 | 0.023 |
| breast-cancer | ENDER-CSLog@50 | 73.08% | 49.8 | 0.056 |
| breast-cancer | ENDER-Newton@10 | 71.69% | 10.7 | 0.013 |
| breast-cancer | ENDER-Newton@100 | 72.04% | 87.4 | 0.107 |
| breast-cancer | ENDER-Newton@20 | 72.03% | 19.8 | 0.025 |
| breast-cancer | ENDER-Newton@50 | 72.38% | 46.1 | 0.056 |
| breast-cancer | Slipper@10 | 72.72% | 7.3 | 0.021 |
| breast-cancer | Slipper@100 | 70.27% | 47.6 | 0.191 |
| breast-cancer | Slipper@20 | 72.03% | 13.9 | 0.040 |
| breast-cancer | Slipper@50 | 69.57% | 29.4 | 0.094 |
| colic | Boomer@10 | 85.61% | 8.3 | 0.015 |
| colic | Boomer@100 | 84.53% | 73.0 | 0.128 |
| colic | Boomer@20 | 85.61% | 17.2 | 0.028 |
| colic | Boomer@50 | 85.08% | 39.4 | 0.063 |
| colic | ENDER-CSExp@10 | 80.16% | 9.3 | 0.011 |
| colic | ENDER-CSExp@100 | 84.52% | 89.8 | 0.102 |
| colic | ENDER-CSExp@20 | 84.52% | 17.3 | 0.020 |
| colic | ENDER-CSExp@50 | 84.25% | 44.7 | 0.050 |
| colic | ENDER-CSLog@10 | 81.00% | 8.6 | 0.012 |
| colic | ENDER-CSLog@100 | 84.79% | 87.8 | 0.108 |
| colic | ENDER-CSLog@20 | 84.25% | 15.8 | 0.022 |
| colic | ENDER-CSLog@50 | 84.52% | 42.5 | 0.055 |
| colic | ENDER-Newton@10 | 83.98% | 8.5 | 0.013 |
| colic | ENDER-Newton@100 | 85.07% | 85.1 | 0.106 |
| colic | ENDER-Newton@20 | 85.33% | 17.0 | 0.025 |
| colic | ENDER-Newton@50 | 85.07% | 43.3 | 0.059 |
| colic | Slipper@10 | 83.18% | 7.7 | 0.023 |
| colic | Slipper@100 | 80.17% | 47.2 | 0.192 |
| colic | Slipper@20 | 82.88% | 13.9 | 0.041 |
| colic | Slipper@50 | 82.08% | 28.4 | 0.098 |
| credit-approval | Boomer@10 | 86.09% | 9.9 | 0.021 |
| credit-approval | Boomer@100 | 86.38% | 75.6 | 0.169 |
| credit-approval | Boomer@20 | 86.81% | 18.7 | 0.039 |
| credit-approval | Boomer@50 | 86.67% | 43.3 | 0.093 |
| credit-approval | ENDER-CSExp@10 | 84.78% | 10.9 | 0.016 |
| credit-approval | ENDER-CSExp@100 | 86.67% | 98.1 | 0.146 |
| credit-approval | ENDER-CSExp@20 | 85.80% | 20.7 | 0.029 |
| credit-approval | ENDER-CSExp@50 | 86.81% | 49.5 | 0.071 |
| credit-approval | ENDER-CSLog@10 | 84.49% | 10.4 | 0.018 |
| credit-approval | ENDER-CSLog@100 | 86.38% | 95.1 | 0.161 |
| credit-approval | ENDER-CSLog@20 | 85.65% | 19.9 | 0.032 |
| credit-approval | ENDER-CSLog@50 | 86.52% | 47.1 | 0.079 |
| credit-approval | ENDER-Newton@10 | 85.22% | 10.7 | 0.019 |
| credit-approval | ENDER-Newton@100 | 86.52% | 92.8 | 0.160 |
| credit-approval | ENDER-Newton@20 | 85.94% | 20.3 | 0.035 |
| credit-approval | ENDER-Newton@50 | 86.38% | 47.8 | 0.081 |
| credit-approval | Slipper@10 | 84.64% | 7.4 | 0.030 |
| credit-approval | Slipper@100 | 86.09% | 48.1 | 0.239 |
| credit-approval | Slipper@20 | 84.93% | 14.2 | 0.052 |
| credit-approval | Slipper@50 | 86.09% | 29.7 | 0.125 |
| credit-g | Boomer@10 | 73.20% | 11.0 | 0.032 |
| credit-g | Boomer@100 | 75.70% | 91.2 | 0.235 |
| credit-g | Boomer@20 | 74.00% | 20.8 | 0.058 |
| credit-g | Boomer@50 | 74.40% | 49.2 | 0.130 |
| credit-g | ENDER-CSExp@10 | 70.00% | 11.0 | 0.023 |
| credit-g | ENDER-CSExp@100 | 74.70% | 100.9 | 0.199 |
| credit-g | ENDER-CSExp@20 | 70.00% | 21.0 | 0.044 |
| credit-g | ENDER-CSExp@50 | 72.80% | 51.0 | 0.101 |
| credit-g | ENDER-CSLog@10 | 70.00% | 11.0 | 0.027 |
| credit-g | ENDER-CSLog@100 | 74.50% | 100.8 | 0.235 |
| credit-g | ENDER-CSLog@20 | 70.10% | 21.0 | 0.050 |
| credit-g | ENDER-CSLog@50 | 73.20% | 51.0 | 0.121 |
| credit-g | ENDER-Newton@10 | 70.10% | 11.0 | 0.029 |
| credit-g | ENDER-Newton@100 | 75.30% | 100.0 | 0.222 |
| credit-g | ENDER-Newton@20 | 72.30% | 21.0 | 0.053 |
| credit-g | ENDER-Newton@50 | 74.40% | 50.9 | 0.118 |
| credit-g | Slipper@10 | 73.20% | 7.6 | 0.037 |
| credit-g | Slipper@100 | 73.40% | 60.1 | 0.332 |
| credit-g | Slipper@20 | 73.50% | 14.4 | 0.071 |
| credit-g | Slipper@50 | 74.40% | 33.7 | 0.173 |
| cylinder-bands | Boomer@10 | 69.63% | 10.2 | 0.028 |
| cylinder-bands | Boomer@100 | 80.19% | 81.0 | 0.273 |
| cylinder-bands | Boomer@20 | 75.00% | 19.2 | 0.050 |
| cylinder-bands | Boomer@50 | 78.33% | 42.8 | 0.133 |
| cylinder-bands | ENDER-CSExp@10 | 58.52% | 11.0 | 0.028 |
| cylinder-bands | ENDER-CSExp@100 | 75.74% | 100.8 | 0.235 |
| cylinder-bands | ENDER-CSExp@20 | 65.37% | 21.0 | 0.051 |
| cylinder-bands | ENDER-CSExp@50 | 71.67% | 50.9 | 0.121 |
| cylinder-bands | ENDER-CSLog@10 | 58.89% | 11.0 | 0.031 |
| cylinder-bands | ENDER-CSLog@100 | 75.93% | 101.0 | 0.276 |
| cylinder-bands | ENDER-CSLog@20 | 68.52% | 21.0 | 0.058 |
| cylinder-bands | ENDER-CSLog@50 | 75.56% | 51.0 | 0.139 |
| cylinder-bands | ENDER-Newton@10 | 62.41% | 10.0 | 0.022 |
| cylinder-bands | ENDER-Newton@100 | 77.41% | 91.1 | 0.195 |
| cylinder-bands | ENDER-Newton@20 | 62.96% | 18.7 | 0.040 |
| cylinder-bands | ENDER-Newton@50 | 71.48% | 46.5 | 0.099 |
| cylinder-bands | Slipper@10 | 69.44% | 9.5 | 0.027 |
| cylinder-bands | Slipper@100 | 76.11% | 61.6 | 0.238 |
| cylinder-bands | Slipper@20 | 70.56% | 17.4 | 0.050 |
| cylinder-bands | Slipper@50 | 74.63% | 37.6 | 0.119 |
| dresses-sales | Boomer@10 | 62.00% | 10.6 | 0.028 |
| dresses-sales | Boomer@100 | 59.80% | 69.2 | 0.170 |
| dresses-sales | Boomer@20 | 63.80% | 19.8 | 0.046 |
| dresses-sales | Boomer@50 | 60.40% | 41.7 | 0.099 |
| dresses-sales | ENDER-CSExp@10 | 58.20% | 10.9 | 0.020 |
| dresses-sales | ENDER-CSExp@100 | 63.00% | 99.3 | 0.167 |
| dresses-sales | ENDER-CSExp@20 | 59.00% | 20.7 | 0.038 |
| dresses-sales | ENDER-CSExp@50 | 62.80% | 50.2 | 0.084 |
| dresses-sales | ENDER-CSLog@10 | 58.00% | 10.9 | 0.023 |
| dresses-sales | ENDER-CSLog@100 | 62.40% | 99.9 | 0.199 |
| dresses-sales | ENDER-CSLog@20 | 59.40% | 20.8 | 0.042 |
| dresses-sales | ENDER-CSLog@50 | 61.40% | 50.5 | 0.100 |
| dresses-sales | ENDER-Newton@10 | 59.80% | 11.0 | 0.020 |
| dresses-sales | ENDER-Newton@100 | 61.60% | 90.7 | 0.153 |
| dresses-sales | ENDER-Newton@20 | 62.20% | 20.7 | 0.037 |
| dresses-sales | ENDER-Newton@50 | 62.00% | 48.8 | 0.085 |
| dresses-sales | Slipper@10 | 59.20% | 8.6 | 0.025 |
| dresses-sales | Slipper@100 | 58.40% | 52.8 | 0.193 |
| dresses-sales | Slipper@20 | 58.80% | 16.2 | 0.044 |
| dresses-sales | Slipper@50 | 59.40% | 35.0 | 0.102 |
| heart-c | Boomer@10 | 81.54% | 10.8 | 0.014 |
| heart-c | Boomer@100 | 80.55% | 71.7 | 0.116 |
| heart-c | Boomer@20 | 80.90% | 20.0 | 0.027 |
| heart-c | Boomer@50 | 80.90% | 42.4 | 0.062 |
| heart-c | ENDER-CSExp@10 | 79.22% | 11.0 | 0.011 |
| heart-c | ENDER-CSExp@100 | 83.20% | 98.4 | 0.096 |
| heart-c | ENDER-CSExp@20 | 80.56% | 20.6 | 0.020 |
| heart-c | ENDER-CSExp@50 | 82.23% | 49.8 | 0.048 |
| heart-c | ENDER-CSLog@10 | 77.90% | 10.9 | 0.012 |
| heart-c | ENDER-CSLog@100 | 82.86% | 98.6 | 0.104 |
| heart-c | ENDER-CSLog@20 | 81.54% | 20.5 | 0.022 |
| heart-c | ENDER-CSLog@50 | 80.89% | 49.6 | 0.052 |
| heart-c | ENDER-Newton@10 | 79.88% | 10.8 | 0.013 |
| heart-c | ENDER-Newton@100 | 83.20% | 94.0 | 0.108 |
| heart-c | ENDER-Newton@20 | 81.54% | 20.0 | 0.024 |
| heart-c | ENDER-Newton@50 | 81.89% | 48.9 | 0.056 |
| heart-c | Slipper@10 | 78.60% | 7.5 | 0.021 |
| heart-c | Slipper@100 | 75.28% | 48.4 | 0.202 |
| heart-c | Slipper@20 | 79.25% | 13.5 | 0.040 |
| heart-c | Slipper@50 | 79.87% | 28.4 | 0.102 |
| heart-h | Boomer@10 | 79.56% | 9.0 | 0.014 |
| heart-h | Boomer@100 | 79.91% | 50.9 | 0.098 |
| heart-h | Boomer@20 | 80.59% | 17.0 | 0.026 |
| heart-h | Boomer@50 | 80.59% | 31.4 | 0.053 |
| heart-h | ENDER-CSExp@10 | 78.21% | 10.9 | 0.010 |
| heart-h | ENDER-CSExp@100 | 83.00% | 94.1 | 0.091 |
| heart-h | ENDER-CSExp@20 | 82.33% | 20.0 | 0.019 |
| heart-h | ENDER-CSExp@50 | 83.00% | 48.0 | 0.046 |
| heart-h | ENDER-CSLog@10 | 76.83% | 10.8 | 0.011 |
| heart-h | ENDER-CSLog@100 | 82.67% | 94.0 | 0.103 |
| heart-h | ENDER-CSLog@20 | 81.66% | 19.7 | 0.020 |
| heart-h | ENDER-CSLog@50 | 82.66% | 47.4 | 0.050 |
| heart-h | ENDER-Newton@10 | 78.90% | 10.6 | 0.012 |
| heart-h | ENDER-Newton@100 | 81.63% | 86.3 | 0.098 |
| heart-h | ENDER-Newton@20 | 79.92% | 19.7 | 0.023 |
| heart-h | ENDER-Newton@50 | 81.64% | 46.7 | 0.054 |
| heart-h | Slipper@10 | 78.55% | 7.4 | 0.022 |
| heart-h | Slipper@100 | 79.25% | 41.2 | 0.183 |
| heart-h | Slipper@20 | 78.90% | 14.3 | 0.038 |
| heart-h | Slipper@50 | 78.21% | 25.8 | 0.093 |
| heart-statlog | Boomer@10 | 80.37% | 10.4 | 0.015 |
| heart-statlog | Boomer@100 | 77.78% | 67.7 | 0.113 |
| heart-statlog | Boomer@20 | 79.63% | 19.8 | 0.027 |
| heart-statlog | Boomer@50 | 80.74% | 42.0 | 0.061 |
| heart-statlog | ENDER-CSExp@10 | 79.63% | 11.0 | 0.011 |
| heart-statlog | ENDER-CSExp@100 | 83.33% | 98.2 | 0.094 |
| heart-statlog | ENDER-CSExp@20 | 83.70% | 20.9 | 0.021 |
| heart-statlog | ENDER-CSExp@50 | 84.07% | 50.4 | 0.047 |
| heart-statlog | ENDER-CSLog@10 | 81.48% | 10.9 | 0.013 |
| heart-statlog | ENDER-CSLog@100 | 84.07% | 98.4 | 0.104 |
| heart-statlog | ENDER-CSLog@20 | 84.81% | 20.7 | 0.022 |
| heart-statlog | ENDER-CSLog@50 | 84.07% | 50.3 | 0.053 |
| heart-statlog | ENDER-Newton@10 | 79.63% | 10.9 | 0.012 |
| heart-statlog | ENDER-Newton@100 | 84.44% | 93.7 | 0.103 |
| heart-statlog | ENDER-Newton@20 | 83.33% | 20.6 | 0.023 |
| heart-statlog | ENDER-Newton@50 | 82.96% | 49.1 | 0.054 |
| heart-statlog | Slipper@10 | 79.26% | 7.5 | 0.021 |
| heart-statlog | Slipper@100 | 78.89% | 45.4 | 0.192 |
| heart-statlog | Slipper@20 | 81.85% | 13.5 | 0.040 |
| heart-statlog | Slipper@50 | 79.26% | 28.2 | 0.095 |
| hepatitis | Boomer@10 | 81.42% | 8.5 | 0.012 |
| hepatitis | Boomer@100 | 81.46% | 45.1 | 0.087 |
| hepatitis | Boomer@20 | 84.00% | 15.1 | 0.019 |
| hepatitis | Boomer@50 | 83.33% | 30.3 | 0.048 |
| hepatitis | ENDER-CSExp@10 | 79.38% | 10.2 | 0.009 |
| hepatitis | ENDER-CSExp@100 | 82.71% | 81.4 | 0.070 |
| hepatitis | ENDER-CSExp@20 | 79.38% | 18.7 | 0.017 |
| hepatitis | ENDER-CSExp@50 | 82.00% | 43.2 | 0.036 |
| hepatitis | ENDER-CSLog@10 | 79.38% | 10.5 | 0.009 |
| hepatitis | ENDER-CSLog@100 | 82.75% | 85.3 | 0.076 |
| hepatitis | ENDER-CSLog@20 | 80.00% | 19.2 | 0.017 |
| hepatitis | ENDER-CSLog@50 | 80.08% | 45.6 | 0.041 |
| hepatitis | ENDER-Newton@10 | 81.29% | 10.4 | 0.009 |
| hepatitis | ENDER-Newton@100 | 82.04% | 73.7 | 0.073 |
| hepatitis | ENDER-Newton@20 | 82.62% | 19.1 | 0.018 |
| hepatitis | ENDER-Newton@50 | 81.42% | 41.4 | 0.040 |
| hepatitis | Slipper@10 | 81.96% | 7.6 | 0.020 |
| hepatitis | Slipper@100 | 82.71% | 37.8 | 0.171 |
| hepatitis | Slipper@20 | 82.67% | 12.8 | 0.035 |
| hepatitis | Slipper@50 | 82.08% | 24.5 | 0.088 |
| kr-vs-kp | Boomer@10 | 93.59% | 6.0 | 0.031 |
| kr-vs-kp | Boomer@100 | 99.37% | 33.8 | 0.251 |
| kr-vs-kp | Boomer@20 | 94.09% | 11.1 | 0.058 |
| kr-vs-kp | Boomer@50 | 96.72% | 20.2 | 0.128 |
| kr-vs-kp | ENDER-CSExp@10 | 90.43% | 6.2 | 0.024 |
| kr-vs-kp | ENDER-CSExp@100 | 96.81% | 63.3 | 0.247 |
| kr-vs-kp | ENDER-CSExp@20 | 92.24% | 11.7 | 0.046 |
| kr-vs-kp | ENDER-CSExp@50 | 95.78% | 30.6 | 0.117 |
| kr-vs-kp | ENDER-CSLog@10 | 90.43% | 5.8 | 0.033 |
| kr-vs-kp | ENDER-CSLog@100 | 95.68% | 45.3 | 0.295 |
| kr-vs-kp | ENDER-CSLog@20 | 90.43% | 7.8 | 0.056 |
| kr-vs-kp | ENDER-CSLog@50 | 94.24% | 19.3 | 0.139 |
| kr-vs-kp | ENDER-Newton@10 | 90.43% | 4.2 | 0.024 |
| kr-vs-kp | ENDER-Newton@100 | 94.34% | 26.1 | 0.207 |
| kr-vs-kp | ENDER-Newton@20 | 90.43% | 6.5 | 0.044 |
| kr-vs-kp | ENDER-Newton@50 | 94.09% | 16.5 | 0.110 |
| kr-vs-kp | Slipper@10 | 96.90% | 7.6 | 0.048 |
| kr-vs-kp | Slipper@100 | 99.59% | 51.6 | 0.419 |
| kr-vs-kp | Slipper@20 | 97.69% | 14.6 | 0.093 |
| kr-vs-kp | Slipper@50 | 99.25% | 31.5 | 0.217 |
| molecular-biology_promoters | Boomer@10 | 83.18% | 9.2 | 0.012 |
| molecular-biology_promoters | Boomer@100 | 93.45% | 51.1 | 0.109 |
| molecular-biology_promoters | Boomer@20 | 87.00% | 16.8 | 0.022 |
| molecular-biology_promoters | Boomer@50 | 91.64% | 32.9 | 0.053 |
| molecular-biology_promoters | ENDER-CSExp@10 | 85.91% | 10.0 | 0.009 |
| molecular-biology_promoters | ENDER-CSExp@100 | 91.73% | 86.4 | 0.079 |
| molecular-biology_promoters | ENDER-CSExp@20 | 87.09% | 18.9 | 0.019 |
| molecular-biology_promoters | ENDER-CSExp@50 | 88.82% | 45.6 | 0.040 |
| molecular-biology_promoters | ENDER-CSLog@10 | 85.91% | 10.1 | 0.010 |
| molecular-biology_promoters | ENDER-CSLog@100 | 91.64% | 88.0 | 0.083 |
| molecular-biology_promoters | ENDER-CSLog@20 | 88.00% | 19.1 | 0.018 |
| molecular-biology_promoters | ENDER-CSLog@50 | 89.73% | 46.2 | 0.042 |
| molecular-biology_promoters | ENDER-Newton@10 | 81.09% | 10.4 | 0.011 |
| molecular-biology_promoters | ENDER-Newton@100 | 90.73% | 82.8 | 0.087 |
| molecular-biology_promoters | ENDER-Newton@20 | 87.73% | 19.5 | 0.020 |
| molecular-biology_promoters | ENDER-Newton@50 | 87.82% | 44.2 | 0.045 |
| molecular-biology_promoters | Slipper@10 | 81.18% | 7.3 | 0.031 |
| molecular-biology_promoters | Slipper@100 | 89.73% | 48.0 | 0.185 |
| molecular-biology_promoters | Slipper@20 | 86.64% | 12.5 | 0.045 |
| molecular-biology_promoters | Slipper@50 | 89.73% | 27.4 | 0.097 |
| mushroom | Boomer@10 | 98.92% | 6.7 | 0.114 |
| mushroom | Boomer@100 | 99.95% | 24.8 | 0.645 |
| mushroom | Boomer@20 | 98.93% | 10.4 | 0.186 |
| mushroom | Boomer@50 | 99.95% | 16.1 | 0.365 |
| mushroom | ENDER-CSExp@10 | 98.65% | 9.4 | 0.085 |
| mushroom | ENDER-CSExp@100 | 100.00% | 51.9 | 0.532 |
| mushroom | ENDER-CSExp@20 | 99.35% | 15.1 | 0.142 |
| mushroom | ENDER-CSExp@50 | 99.93% | 30.7 | 0.293 |
| mushroom | ENDER-CSLog@10 | 98.52% | 7.2 | 0.104 |
| mushroom | ENDER-CSLog@100 | 99.91% | 40.9 | 0.719 |
| mushroom | ENDER-CSLog@20 | 98.52% | 11.3 | 0.173 |
| mushroom | ENDER-CSLog@50 | 99.51% | 25.5 | 0.393 |
| mushroom | ENDER-Newton@10 | 96.16% | 6.1 | 0.091 |
| mushroom | ENDER-Newton@100 | 99.91% | 25.8 | 0.552 |
| mushroom | ENDER-Newton@20 | 98.88% | 10.0 | 0.154 |
| mushroom | ENDER-Newton@50 | 98.98% | 18.5 | 0.318 |
| mushroom | Slipper@10 | 100.00% | 7.6 | 0.111 |
| mushroom | Slipper@100 | 100.00% | 18.9 | 0.716 |
| mushroom | Slipper@20 | 100.00% | 10.5 | 0.188 |
| mushroom | Slipper@50 | 100.00% | 14.7 | 0.389 |
| sick | Boomer@10 | 98.12% | 9.4 | 0.051 |
| sick | Boomer@100 | 98.28% | 57.3 | 0.333 |
| sick | Boomer@20 | 98.04% | 16.5 | 0.087 |
| sick | Boomer@50 | 98.17% | 34.6 | 0.185 |
| sick | ENDER-CSExp@10 | 93.88% | 9.8 | 0.034 |
| sick | ENDER-CSExp@100 | 97.96% | 83.9 | 0.315 |
| sick | ENDER-CSExp@20 | 94.99% | 18.3 | 0.062 |
| sick | ENDER-CSExp@50 | 97.99% | 43.6 | 0.153 |
| sick | ENDER-CSLog@10 | 93.88% | 10.0 | 0.048 |
| sick | ENDER-CSLog@100 | 98.14% | 87.8 | 0.421 |
| sick | ENDER-CSLog@20 | 97.53% | 17.9 | 0.085 |
| sick | ENDER-CSLog@50 | 98.06% | 42.4 | 0.202 |
| sick | ENDER-Newton@10 | 97.88% | 10.1 | 0.045 |
| sick | ENDER-Newton@100 | 98.28% | 76.8 | 0.348 |
| sick | ENDER-Newton@20 | 97.96% | 17.8 | 0.079 |
| sick | ENDER-Newton@50 | 98.17% | 41.0 | 0.182 |
| sick | Slipper@10 | 97.77% | 7.0 | 0.062 |
| sick | Slipper@100 | 98.30% | 53.9 | 0.605 |
| sick | Slipper@20 | 97.96% | 13.6 | 0.123 |
| sick | Slipper@50 | 98.12% | 30.2 | 0.294 |
| tic-tac-toe | Boomer@10 | 79.12% | 9.0 | 0.017 |
| tic-tac-toe | Boomer@100 | 98.44% | 62.7 | 0.144 |
| tic-tac-toe | Boomer@20 | 92.38% | 15.6 | 0.032 |
| tic-tac-toe | Boomer@50 | 98.33% | 34.0 | 0.075 |
| tic-tac-toe | ENDER-CSExp@10 | 65.87% | 9.2 | 0.014 |
| tic-tac-toe | ENDER-CSExp@100 | 82.04% | 89.7 | 0.122 |
| tic-tac-toe | ENDER-CSExp@20 | 71.50% | 17.9 | 0.025 |
| tic-tac-toe | ENDER-CSExp@50 | 78.91% | 45.1 | 0.060 |
| tic-tac-toe | ENDER-CSLog@10 | 65.34% | 8.9 | 0.015 |
| tic-tac-toe | ENDER-CSLog@100 | 77.04% | 83.3 | 0.140 |
| tic-tac-toe | ENDER-CSLog@20 | 67.33% | 16.4 | 0.028 |
| tic-tac-toe | ENDER-CSLog@50 | 73.80% | 42.0 | 0.070 |
| tic-tac-toe | ENDER-Newton@10 | 72.87% | 8.8 | 0.017 |
| tic-tac-toe | ENDER-Newton@100 | 95.20% | 60.4 | 0.137 |
| tic-tac-toe | ENDER-Newton@20 | 78.60% | 13.7 | 0.029 |
| tic-tac-toe | ENDER-Newton@50 | 81.94% | 29.5 | 0.068 |
| tic-tac-toe | Slipper@10 | 97.71% | 9.0 | 0.028 |
| tic-tac-toe | Slipper@100 | 98.54% | 59.1 | 0.287 |
| tic-tac-toe | Slipper@20 | 98.33% | 15.0 | 0.057 |
| tic-tac-toe | Slipper@50 | 97.71% | 31.1 | 0.143 |
| vote | Boomer@10 | 95.65% | 6.1 | 0.009 |
| vote | Boomer@100 | 95.87% | 46.6 | 0.086 |
| vote | Boomer@20 | 95.42% | 11.2 | 0.017 |
| vote | Boomer@50 | 95.64% | 27.4 | 0.042 |
| vote | ENDER-CSExp@10 | 95.18% | 5.6 | 0.007 |
| vote | ENDER-CSExp@100 | 95.87% | 42.8 | 0.059 |
| vote | ENDER-CSExp@20 | 95.42% | 10.6 | 0.013 |
| vote | ENDER-CSExp@50 | 95.64% | 23.2 | 0.030 |
| vote | ENDER-CSLog@10 | 95.18% | 4.7 | 0.008 |
| vote | ENDER-CSLog@100 | 95.42% | 37.9 | 0.071 |
| vote | ENDER-CSLog@20 | 95.42% | 7.9 | 0.014 |
| vote | ENDER-CSLog@50 | 95.42% | 17.6 | 0.034 |
| vote | ENDER-Newton@10 | 93.80% | 6.1 | 0.008 |
| vote | ENDER-Newton@100 | 96.10% | 48.6 | 0.075 |
| vote | ENDER-Newton@20 | 95.18% | 8.2 | 0.014 |
| vote | ENDER-Newton@50 | 95.42% | 20.0 | 0.034 |
| vote | Slipper@10 | 95.43% | 6.2 | 0.022 |
| vote | Slipper@100 | 94.03% | 23.9 | 0.170 |
| vote | Slipper@20 | 94.73% | 10.9 | 0.041 |
| vote | Slipper@50 | 94.73% | 18.0 | 0.091 |

## Comparative discussion

Section 1's best covering-strategy point is `Weighted-Add+MEstimate16` (mean accuracy 84.74% over the cells every section-1 config completed). Section 2's best boosting point is `ENDER-Newton@100` (85.25%, same restriction). Boosting and weighted covering aren't as separate as they might look. `AdaBoostReweighting` (Slipper's mechanism, and `WeightedCovering`'s rule-weight-assigning option, which plain `SeCo` itself refuses to build into an unweighted set) and `ENDER`'s `ExponentialLoss` configuration are the *same* thing by a classical result (Friedman, Hastie & Tibshirani 2000): AdaBoost **is** gradient boosting on the exponential loss. `ExponentialLoss` here literally defines its per-example quantity as ``w = exp(-y f)`` -- AdaBoost's reweighting formula -- and fits each rule's weight with the identical closed form `AdaBoostReweighting.confidence` uses. At budget 100, `Slipper@100` scores 83.75% and `ENDER-CSExp@100` scores 84.68% -- close, as expected from the theory below. What actually distinguishes the two families: section 1's *other* reweighting schemes (`MultiplicativeReweighting`'s ``gamma**k``, `AdditiveReweighting`'s ``1/(k+1)``, `LRIReweighting`'s ``1 + e**3``) are hand-designed update rules with no loss function behind them, so they can't generalize the way `ENDER`'s pluggable-loss framework does (logistic, exponential, sigmoid, five weight-fitting methods). And weighted covering uses its weights only to steer training -- it still predicts with a heuristic score over an *unweighted* rule set -- while boosting bakes the fitted weight into the model itself (`LinearRuleModel`, summed at predict time): there the weight is part of the answer, not just how training got there.

## Accuracy vs. fit time

Every point here is on the same accuracy basis as the tables above (the 170 common cells), against mean fit time on a log scale, since it spans sub-second boosting fits to multi-minute (and timed-out) covering fits. Color is the algorithm family -- the four covering strategies, `CPAR`/`LRI`, and the five boosting families; for section 1, marker shape is the heuristic, so a covering strategy's three heuristic variants show up as same-color points of different shapes, joined by a dotted line in the order Laplace, m-estimate, WRAcc. A boosting family's four rule-count budgets (10/20/50/100) are connected by a line, tracing out that family's own accuracy-vs-cost curve rather than four unrelated points. `Boomer` and `ENDER-Newton`'s lines nearly overlap at the top right -- both reach similarly high accuracy at a similar cost, a real finding (the two are closely related mechanisms; see the comparative discussion above), not a plotting artifact.

![accuracy vs. fit time](covering_boosting_plots/covering_boosting_accuracy_vs_time.png)

