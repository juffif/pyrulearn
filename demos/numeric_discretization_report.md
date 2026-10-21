# Numeric attributes: intrinsic handling vs. discretization (JRip / J48)

## Setup

**Full run** (27 small and medium-sized datasets, all attributes numeric, 10-fold). For a quick sanity check instead (5 small datasets, 3-fold): `python demos/numeric_discretization.py` with no arguments -- its output isn't checked in (see `numeric_discretization_quick_report.md` after running it).

Weka's JRip and J48 compared across 9 data preparations per dataset: `raw`
(Weka's own intrinsic numeric thresholding), our own decision-tree-based
discretization (`build_dataspec`/`tree_thresholds`) and `sklearn.preprocessing.
KBinsDiscretizer` (`strategy='quantile'`, unsupervised), each of the
latter two at 3, 5, 8, 12 intervals. Both discretizers
are fit on the training fold only, per fold; the test fold is binarized
against that same fold's thresholds.

Measures: test accuracy, rule count, condition count, and (for the discretized
preps) the number of binary features that discretization produced. Fit time
is recorded but secondary here -- every fit is a single bounded Weka
subprocess call regardless of preparation. Each fit is capped at
120s; a timeout or an exception is recorded as a failure, not
fatal to the run.

Missing values: in the discretized preps a missing value makes both a
threshold feature and its negation false (e.g. neither `plas>=154.5` nor
`plas<154.5`), so a rule never covers it through that attribute. Weka's
own `raw` handling differs, which can hurt the discretized preps on
datasets with many missing values -- here mainly `diabetes` (376 of 768
rows with a missing value; `breast-w` has 16 of 699).

![accuracy vs. discretization level](numeric_discretization_plots/numeric_discretization_accuracy.png)

## Binary features produced per discretization level

| prep | mean binary features |
|---|--:|
| tree@3 | 78.7 |
| tree@5 | 155.1 |
| tree@8 | 266.3 |
| tree@12 | 407.7 |
| kbins@3 | 69.7 |
| kbins@5 | 136.3 |
| kbins@8 | 231.6 |
| kbins@12 | 350.0 |

## Summary by variant (mean across every dataset and fold)

| variant | accuracy | n_rules | conds/rule | fit_time (s) | wins | mean rank |
|---|--:|--:|--:|--:|--:|--:|
| JRip:raw | 84.36% | 7.5 | 2.6 | 0.59 | 0.0 | 7.17 |
| JRip:tree@12 | 84.37% | 7.6 | 2.8 | 0.93 | 0.0 | 7.59 |
| J48:kbins@12 | 83.62% | 47.3 | 7.5 | 0.59 | 3.0 | 7.72 |
| J48:raw | 84.32% | 40.7 | 6.3 | 0.48 | 4.0 | 7.74 |
| JRip:tree@8 | 84.21% | 7.3 | 2.8 | 0.74 | 1.0 | 7.80 |
| J48:tree@12 | 83.48% | 41.7 | 7.8 | 0.61 | 1.0 | 8.81 |
| J48:tree@8 | 83.38% | 37.8 | 7.2 | 0.54 | 1.0 | 8.83 |
| JRip:kbins@12 | 84.14% | 8.0 | 3.0 | 0.88 | 1.5 | 8.94 |
| J48:kbins@8 | 83.33% | 43.0 | 7.0 | 0.53 | 5.0 | 9.02 |
| JRip:kbins@8 | 83.75% | 7.7 | 3.1 | 0.71 | 0.5 | 9.20 |
| JRip:tree@5 | 83.52% | 6.8 | 2.8 | 0.58 | 2.0 | 9.89 |
| J48:tree@5 | 83.15% | 30.3 | 6.3 | 0.47 | 1.0 | 9.94 |
| JRip:tree@3 | 82.82% | 5.9 | 2.7 | 0.47 | 3.0 | 10.00 |
| J48:tree@3 | 82.57% | 22.1 | 5.2 | 0.41 | 1.0 | 10.46 |
| JRip:kbins@5 | 83.03% | 7.0 | 3.0 | 0.57 | 0.0 | 10.65 |
| J48:kbins@5 | 82.77% | 36.6 | 6.0 | 0.45 | 0.0 | 11.11 |
| J48:kbins@3 | 82.48% | 27.5 | 4.8 | 0.41 | 2.0 | 12.31 |
| JRip:kbins@3 | 81.48% | 6.1 | 3.2 | 0.46 | 1.0 | 13.80 |

`wins` -- datasets where a variant's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last.

## Per-dataset results, per variant (mean across folds)

| dataset | algorithm | prep | accuracy | n_rules | n_conditions | conds_per_rule | fit_time |
|---|---|---|---|---|---|---|---|
| analcatdata_authorship | J48 | kbins@12 | 95.48% | 21.3 | 123.3 | 5.7 | 0.860 |
| analcatdata_authorship | J48 | kbins@3 | 92.39% | 26.2 | 145.9 | 5.5 | 0.512 |
| analcatdata_authorship | J48 | kbins@5 | 90.61% | 25.5 | 140.2 | 5.5 | 0.584 |
| analcatdata_authorship | J48 | kbins@8 | 93.58% | 25.3 | 148.7 | 5.9 | 0.782 |
| analcatdata_authorship | J48 | raw | 95.12% | 20.2 | 97.5 | 4.8 | 0.596 |
| analcatdata_authorship | J48 | tree@12 | 95.01% | 18.6 | 98.2 | 5.3 | 0.859 |
| analcatdata_authorship | J48 | tree@3 | 93.58% | 20.7 | 107.1 | 5.1 | 0.528 |
| analcatdata_authorship | J48 | tree@5 | 93.58% | 19.9 | 101.4 | 5.1 | 0.631 |
| analcatdata_authorship | J48 | tree@8 | 94.53% | 18.3 | 91.4 | 5.0 | 0.734 |
| analcatdata_authorship | JRip | kbins@12 | 92.51% | 9.4 | 19.9 | 2.1 | 1.351 |
| analcatdata_authorship | JRip | kbins@3 | 92.74% | 11.2 | 31.5 | 2.8 | 0.642 |
| analcatdata_authorship | JRip | kbins@5 | 93.46% | 10.0 | 25.4 | 2.6 | 0.881 |
| analcatdata_authorship | JRip | kbins@8 | 93.34% | 8.7 | 21.1 | 2.4 | 1.082 |
| analcatdata_authorship | JRip | raw | 94.65% | 9.8 | 18.7 | 1.9 | 0.752 |
| analcatdata_authorship | JRip | tree@12 | 94.88% | 9.5 | 18.6 | 2.0 | 1.600 |
| analcatdata_authorship | JRip | tree@3 | 95.48% | 10.5 | 20.4 | 1.9 | 0.656 |
| analcatdata_authorship | JRip | tree@5 | 94.29% | 10.2 | 20.5 | 2.0 | 0.826 |
| analcatdata_authorship | JRip | tree@8 | 94.65% | 9.5 | 18.2 | 1.9 | 1.141 |
| balance-scale | J48 | kbins@12 | 78.86% | 34.5 | 223.9 | 6.4 | 0.350 |
| balance-scale | J48 | kbins@3 | 78.23% | 24.4 | 128.1 | 5.2 | 0.344 |
| balance-scale | J48 | kbins@5 | 79.34% | 24.0 | 134.7 | 5.5 | 0.347 |
| balance-scale | J48 | kbins@8 | 78.86% | 34.5 | 223.9 | 6.4 | 0.344 |
| balance-scale | J48 | raw | 78.23% | 42.9 | 278.0 | 6.5 | 0.393 |
| balance-scale | J48 | tree@12 | 78.70% | 42.5 | 313.5 | 7.4 | 0.376 |
| balance-scale | J48 | tree@3 | 75.50% | 17.8 | 82.4 | 4.6 | 0.346 |
| balance-scale | J48 | tree@5 | 78.70% | 42.5 | 313.5 | 7.4 | 0.365 |
| balance-scale | J48 | tree@8 | 78.70% | 42.5 | 313.5 | 7.4 | 0.373 |
| balance-scale | JRip | kbins@12 | 82.71% | 9.5 | 30.6 | 3.2 | 0.385 |
| balance-scale | JRip | kbins@3 | 78.69% | 8.1 | 25.6 | 3.1 | 0.374 |
| balance-scale | JRip | kbins@5 | 79.98% | 7.4 | 21.9 | 2.9 | 0.382 |
| balance-scale | JRip | kbins@8 | 82.71% | 9.5 | 30.6 | 3.2 | 0.391 |
| balance-scale | JRip | raw | 79.35% | 10.0 | 29.9 | 3.0 | 0.402 |
| balance-scale | JRip | tree@12 | 80.31% | 9.0 | 27.8 | 3.1 | 0.398 |
| balance-scale | JRip | tree@3 | 78.38% | 6.3 | 17.9 | 2.9 | 0.360 |
| balance-scale | JRip | tree@5 | 80.31% | 9.0 | 27.8 | 3.1 | 0.404 |
| balance-scale | JRip | tree@8 | 80.31% | 9.0 | 27.8 | 3.1 | 0.386 |
| banknote-authentication | J48 | kbins@12 | 98.32% | 17.8 | 88.7 | 4.9 | 0.449 |
| banknote-authentication | J48 | kbins@3 | 94.39% | 15.1 | 74.3 | 4.9 | 0.355 |
| banknote-authentication | J48 | kbins@5 | 98.47% | 20.1 | 99.4 | 4.9 | 0.380 |
| banknote-authentication | J48 | kbins@8 | 99.27% | 17.7 | 88.8 | 5.0 | 0.410 |
| banknote-authentication | J48 | raw | 98.69% | 15.6 | 71.0 | 4.5 | 0.401 |
| banknote-authentication | J48 | tree@12 | 98.69% | 13.1 | 56.1 | 4.2 | 0.400 |
| banknote-authentication | J48 | tree@3 | 91.26% | 4.9 | 12.4 | 2.5 | 0.336 |
| banknote-authentication | J48 | tree@5 | 97.01% | 9.4 | 33.5 | 3.5 | 0.346 |
| banknote-authentication | J48 | tree@8 | 98.69% | 11.3 | 44.3 | 3.9 | 0.365 |
| banknote-authentication | JRip | kbins@12 | 99.13% | 7.6 | 20.2 | 2.7 | 0.492 |
| banknote-authentication | JRip | kbins@3 | 93.66% | 5.4 | 16.8 | 3.1 | 0.396 |
| banknote-authentication | JRip | kbins@5 | 98.32% | 7.7 | 23.8 | 3.1 | 0.432 |
| banknote-authentication | JRip | kbins@8 | 99.12% | 8.3 | 24.4 | 2.9 | 0.471 |
| banknote-authentication | JRip | raw | 97.96% | 7.1 | 16.0 | 2.3 | 0.465 |
| banknote-authentication | JRip | tree@12 | 98.76% | 6.2 | 15.1 | 2.4 | 0.475 |
| banknote-authentication | JRip | tree@3 | 91.04% | 2.4 | 5.2 | 2.1 | 0.366 |
| banknote-authentication | JRip | tree@5 | 97.23% | 4.7 | 10.4 | 2.2 | 0.405 |
| banknote-authentication | JRip | tree@8 | 98.91% | 5.3 | 12.0 | 2.3 | 0.437 |
| blood-transfusion-service-center | J48 | kbins@12 | 79.40% | 8.7 | 38.1 | 4.1 | 0.398 |
| blood-transfusion-service-center | J48 | kbins@3 | 77.81% | 4.0 | 9.0 | 2.2 | 0.327 |
| blood-transfusion-service-center | J48 | kbins@5 | 77.00% | 6.1 | 20.9 | 3.4 | 0.341 |
| blood-transfusion-service-center | J48 | kbins@8 | 78.33% | 6.5 | 24.3 | 3.5 | 0.355 |
| blood-transfusion-service-center | J48 | raw | 78.99% | 6.8 | 26.9 | 3.6 | 0.351 |
| blood-transfusion-service-center | J48 | tree@12 | 78.74% | 8.2 | 32.5 | 3.8 | 0.385 |
| blood-transfusion-service-center | J48 | tree@3 | 79.27% | 5.0 | 14.0 | 2.8 | 0.339 |
| blood-transfusion-service-center | J48 | tree@5 | 79.40% | 5.4 | 16.2 | 3.0 | 0.327 |
| blood-transfusion-service-center | J48 | tree@8 | 78.74% | 6.6 | 22.3 | 3.3 | 0.359 |
| blood-transfusion-service-center | JRip | kbins@12 | 78.87% | 2.5 | 7.1 | 2.8 | 0.389 |
| blood-transfusion-service-center | JRip | kbins@3 | 76.60% | 1.0 | 3.1 | 3.1 | 0.336 |
| blood-transfusion-service-center | JRip | kbins@5 | 77.40% | 1.2 | 3.7 | 3.0 | 0.361 |
| blood-transfusion-service-center | JRip | kbins@8 | 78.47% | 1.2 | 4.1 | 3.4 | 0.368 |
| blood-transfusion-service-center | JRip | raw | 78.34% | 1.9 | 5.2 | 2.8 | 0.385 |
| blood-transfusion-service-center | JRip | tree@12 | 78.47% | 2.2 | 5.9 | 2.6 | 0.405 |
| blood-transfusion-service-center | JRip | tree@3 | 79.27% | 2.0 | 5.1 | 2.5 | 0.336 |
| blood-transfusion-service-center | JRip | tree@5 | 79.00% | 2.1 | 5.4 | 2.5 | 0.356 |
| blood-transfusion-service-center | JRip | tree@8 | 79.14% | 2.0 | 5.1 | 2.5 | 0.369 |
| breast-w | J48 | kbins@12 | 93.85% | 10.8 | 43.4 | 3.9 | 0.358 |
| breast-w | J48 | kbins@3 | 93.42% | 11.5 | 45.1 | 3.9 | 0.327 |
| breast-w | J48 | kbins@5 | 94.42% | 8.6 | 29.4 | 3.4 | 0.333 |
| breast-w | J48 | kbins@8 | 94.14% | 10.3 | 39.8 | 3.8 | 0.360 |
| breast-w | J48 | raw | 92.99% | 12.2 | 52.1 | 4.2 | 0.358 |
| breast-w | J48 | tree@12 | 92.85% | 10.2 | 43.1 | 3.9 | 0.392 |
| breast-w | J48 | tree@3 | 94.00% | 8.7 | 32.2 | 3.6 | 0.340 |
| breast-w | J48 | tree@5 | 93.56% | 8.4 | 30.6 | 3.6 | 0.359 |
| breast-w | J48 | tree@8 | 93.13% | 9.2 | 36.5 | 3.7 | 0.383 |
| breast-w | JRip | kbins@12 | 96.42% | 4.8 | 10.3 | 2.1 | 0.401 |
| breast-w | JRip | kbins@3 | 96.14% | 3.6 | 8.7 | 2.4 | 0.350 |
| breast-w | JRip | kbins@5 | 94.85% | 4.2 | 8.6 | 2.0 | 0.369 |
| breast-w | JRip | kbins@8 | 95.13% | 4.4 | 9.2 | 2.1 | 0.389 |
| breast-w | JRip | raw | 95.42% | 5.0 | 10.1 | 2.0 | 0.404 |
| breast-w | JRip | tree@12 | 95.86% | 5.1 | 11.3 | 2.2 | 0.427 |
| breast-w | JRip | tree@3 | 95.42% | 4.8 | 10.5 | 2.1 | 0.370 |
| breast-w | JRip | tree@5 | 94.70% | 4.5 | 9.6 | 2.1 | 0.389 |
| breast-w | JRip | tree@8 | 95.71% | 4.8 | 9.9 | 2.0 | 0.412 |
| climate-model-simulation-crashes | J48 | kbins@12 | 90.93% | 16.7 | 92.1 | 5.4 | 0.458 |
| climate-model-simulation-crashes | J48 | kbins@3 | 89.81% | 14.8 | 82.9 | 5.6 | 0.365 |
| climate-model-simulation-crashes | J48 | kbins@5 | 91.85% | 15.2 | 89.3 | 5.8 | 0.386 |
| climate-model-simulation-crashes | J48 | kbins@8 | 90.19% | 15.7 | 101.1 | 6.3 | 0.418 |
| climate-model-simulation-crashes | J48 | raw | 91.85% | 12.1 | 59.6 | 4.9 | 0.417 |
| climate-model-simulation-crashes | J48 | tree@12 | 91.67% | 9.7 | 42.6 | 4.2 | 0.486 |
| climate-model-simulation-crashes | J48 | tree@3 | 93.70% | 8.8 | 31.8 | 3.5 | 0.354 |
| climate-model-simulation-crashes | J48 | tree@5 | 93.15% | 10.0 | 42.2 | 4.1 | 0.399 |
| climate-model-simulation-crashes | J48 | tree@8 | 93.33% | 10.6 | 45.8 | 4.2 | 0.425 |
| climate-model-simulation-crashes | JRip | kbins@12 | 93.33% | 2.4 | 8.2 | 3.4 | 0.477 |
| climate-model-simulation-crashes | JRip | kbins@3 | 91.85% | 2.6 | 10.6 | 4.1 | 0.372 |
| climate-model-simulation-crashes | JRip | kbins@5 | 94.07% | 2.6 | 9.3 | 3.5 | 0.400 |
| climate-model-simulation-crashes | JRip | kbins@8 | 90.93% | 2.9 | 11.4 | 4.0 | 0.435 |
| climate-model-simulation-crashes | JRip | raw | 93.15% | 2.0 | 6.6 | 3.3 | 0.445 |
| climate-model-simulation-crashes | JRip | tree@12 | 93.33% | 3.0 | 9.7 | 3.2 | 0.500 |
| climate-model-simulation-crashes | JRip | tree@3 | 91.48% | 2.5 | 7.5 | 3.0 | 0.387 |
| climate-model-simulation-crashes | JRip | tree@5 | 93.89% | 2.7 | 8.4 | 3.1 | 0.406 |
| climate-model-simulation-crashes | JRip | tree@8 | 94.81% | 2.7 | 8.1 | 3.0 | 0.460 |
| diabetes | J48 | kbins@12 | 54.80% | 58.6 | 634.8 | 10.7 | 0.471 |
| diabetes | J48 | kbins@3 | 64.45% | 22.7 | 138.8 | 6.0 | 0.381 |
| diabetes | J48 | kbins@5 | 55.34% | 57.3 | 512.2 | 8.9 | 0.409 |
| diabetes | J48 | kbins@8 | 53.88% | 52.5 | 564.4 | 10.5 | 0.448 |
| diabetes | J48 | raw | 69.91% | 16.1 | 83.0 | 5.1 | 0.383 |
| diabetes | J48 | tree@12 | 55.73% | 50.4 | 617.8 | 12.0 | 0.495 |
| diabetes | J48 | tree@3 | 58.06% | 27.8 | 190.1 | 6.7 | 0.351 |
| diabetes | J48 | tree@5 | 59.38% | 36.2 | 310.6 | 8.5 | 0.375 |
| diabetes | J48 | tree@8 | 56.63% | 43.0 | 438.7 | 10.1 | 0.442 |
| diabetes | JRip | kbins@12 | 75.39% | 3.1 | 9.8 | 3.1 | 0.494 |
| diabetes | JRip | kbins@3 | 76.30% | 1.9 | 5.9 | 2.9 | 0.374 |
| diabetes | JRip | kbins@5 | 75.26% | 3.1 | 9.3 | 2.9 | 0.417 |
| diabetes | JRip | kbins@8 | 75.52% | 3.5 | 11.2 | 3.1 | 0.455 |
| diabetes | JRip | raw | 75.65% | 2.6 | 6.4 | 2.4 | 0.418 |
| diabetes | JRip | tree@12 | 74.09% | 3.1 | 9.5 | 3.0 | 0.513 |
| diabetes | JRip | tree@3 | 73.95% | 3.1 | 8.9 | 2.9 | 0.395 |
| diabetes | JRip | tree@5 | 71.88% | 4.0 | 13.5 | 3.3 | 0.421 |
| diabetes | JRip | tree@8 | 72.53% | 3.9 | 12.4 | 3.1 | 0.459 |
| ecoli | J48 | kbins@12 | 81.60% | 15.9 | 78.8 | 4.8 | 0.360 |
| ecoli | J48 | kbins@3 | 80.94% | 12.3 | 50.6 | 4.1 | 0.331 |
| ecoli | J48 | kbins@5 | 80.65% | 15.6 | 69.7 | 4.4 | 0.328 |
| ecoli | J48 | kbins@8 | 82.76% | 13.9 | 59.3 | 4.1 | 0.343 |
| ecoli | J48 | raw | 80.70% | 18.8 | 95.2 | 5.0 | 0.356 |
| ecoli | J48 | tree@12 | 80.40% | 16.3 | 80.3 | 4.9 | 0.363 |
| ecoli | J48 | tree@3 | 80.10% | 11.7 | 46.6 | 3.9 | 0.336 |
| ecoli | J48 | tree@5 | 81.58% | 14.0 | 63.9 | 4.4 | 0.338 |
| ecoli | J48 | tree@8 | 80.12% | 15.6 | 75.3 | 4.8 | 0.349 |
| ecoli | JRip | kbins@12 | 81.00% | 7.7 | 20.1 | 2.6 | 0.417 |
| ecoli | JRip | kbins@3 | 79.47% | 5.5 | 14.1 | 2.6 | 0.348 |
| ecoli | JRip | kbins@5 | 80.93% | 7.7 | 20.1 | 2.6 | 0.375 |
| ecoli | JRip | kbins@8 | 82.48% | 6.0 | 14.4 | 2.4 | 0.391 |
| ecoli | JRip | raw | 81.56% | 8.1 | 17.0 | 2.1 | 0.383 |
| ecoli | JRip | tree@12 | 80.98% | 7.4 | 16.3 | 2.2 | 0.414 |
| ecoli | JRip | tree@3 | 81.60% | 6.5 | 13.6 | 2.1 | 0.356 |
| ecoli | JRip | tree@5 | 80.73% | 7.3 | 15.2 | 2.1 | 0.370 |
| ecoli | JRip | tree@8 | 81.59% | 7.1 | 15.7 | 2.2 | 0.398 |
| glass | J48 | kbins@12 | 69.61% | 28.1 | 199.7 | 7.1 | 0.369 |
| glass | J48 | kbins@3 | 69.13% | 21.9 | 109.4 | 5.0 | 0.339 |
| glass | J48 | kbins@5 | 67.32% | 28.0 | 176.1 | 6.2 | 0.331 |
| glass | J48 | kbins@8 | 75.76% | 26.1 | 169.3 | 6.4 | 0.342 |
| glass | J48 | raw | 66.47% | 23.6 | 140.9 | 5.9 | 0.347 |
| glass | J48 | tree@12 | 70.15% | 23.2 | 149.0 | 6.4 | 0.384 |
| glass | J48 | tree@3 | 67.77% | 14.4 | 71.0 | 4.9 | 0.336 |
| glass | J48 | tree@5 | 66.41% | 20.0 | 116.5 | 5.8 | 0.338 |
| glass | J48 | tree@8 | 67.81% | 22.7 | 141.2 | 6.2 | 0.355 |
| glass | JRip | kbins@12 | 69.20% | 8.2 | 22.1 | 2.7 | 0.391 |
| glass | JRip | kbins@3 | 61.23% | 5.5 | 18.0 | 3.3 | 0.356 |
| glass | JRip | kbins@5 | 69.68% | 7.4 | 21.9 | 2.9 | 0.364 |
| glass | JRip | kbins@8 | 63.03% | 8.3 | 22.1 | 2.7 | 0.366 |
| glass | JRip | raw | 66.36% | 6.7 | 15.1 | 2.2 | 0.362 |
| glass | JRip | tree@12 | 67.40% | 7.1 | 16.2 | 2.3 | 0.404 |
| glass | JRip | tree@3 | 65.91% | 6.6 | 17.4 | 2.7 | 0.329 |
| glass | JRip | tree@5 | 65.97% | 6.4 | 15.4 | 2.4 | 0.350 |
| glass | JRip | tree@8 | 66.84% | 6.6 | 16.1 | 2.4 | 0.368 |
| ionosphere | J48 | kbins@12 | 90.30% | 12.2 | 70.7 | 5.7 | 0.472 |
| ionosphere | J48 | kbins@3 | 87.44% | 19.2 | 94.3 | 4.9 | 0.362 |
| ionosphere | J48 | kbins@5 | 88.31% | 16.6 | 93.2 | 5.6 | 0.389 |
| ionosphere | J48 | kbins@8 | 88.30% | 16.4 | 86.4 | 5.2 | 0.424 |
| ionosphere | J48 | raw | 89.17% | 13.2 | 70.9 | 5.3 | 0.407 |
| ionosphere | J48 | tree@12 | 93.44% | 8.1 | 36.3 | 4.4 | 0.490 |
| ionosphere | J48 | tree@3 | 92.29% | 10.1 | 52.4 | 5.0 | 0.362 |
| ionosphere | J48 | tree@5 | 92.01% | 9.6 | 49.5 | 5.0 | 0.392 |
| ionosphere | J48 | tree@8 | 93.72% | 8.2 | 37.2 | 4.5 | 0.440 |
| ionosphere | JRip | kbins@12 | 88.87% | 5.8 | 9.7 | 1.7 | 0.507 |
| ionosphere | JRip | kbins@3 | 87.74% | 6.2 | 14.1 | 2.2 | 0.380 |
| ionosphere | JRip | kbins@5 | 88.33% | 6.2 | 12.0 | 1.9 | 0.452 |
| ionosphere | JRip | kbins@8 | 85.75% | 5.8 | 11.4 | 1.9 | 0.477 |
| ionosphere | JRip | raw | 88.60% | 4.0 | 5.7 | 1.4 | 0.422 |
| ionosphere | JRip | tree@12 | 88.31% | 3.9 | 6.0 | 1.4 | 0.486 |
| ionosphere | JRip | tree@3 | 88.87% | 5.1 | 7.9 | 1.6 | 0.363 |
| ionosphere | JRip | tree@5 | 87.47% | 4.5 | 6.7 | 1.4 | 0.393 |
| ionosphere | JRip | tree@8 | 88.88% | 4.3 | 6.4 | 1.4 | 0.441 |
| iris | J48 | kbins@12 | 93.33% | 3.8 | 8.2 | 2.1 | 0.312 |
| iris | J48 | kbins@3 | 93.33% | 4.6 | 12.2 | 2.6 | 0.306 |
| iris | J48 | kbins@5 | 90.67% | 8.0 | 28.4 | 3.5 | 0.318 |
| iris | J48 | kbins@8 | 96.00% | 4.1 | 9.3 | 2.2 | 0.312 |
| iris | J48 | raw | 92.67% | 4.5 | 11.6 | 2.5 | 0.305 |
| iris | J48 | tree@12 | 94.00% | 4.9 | 14.0 | 2.7 | 0.314 |
| iris | J48 | tree@3 | 92.00% | 3.0 | 5.0 | 1.7 | 0.298 |
| iris | J48 | tree@5 | 93.33% | 3.8 | 8.2 | 2.1 | 0.305 |
| iris | J48 | tree@8 | 94.00% | 4.7 | 12.9 | 2.6 | 0.309 |
| iris | JRip | kbins@12 | 92.00% | 2.5 | 3.6 | 1.4 | 0.316 |
| iris | JRip | kbins@3 | 92.00% | 2.6 | 4.2 | 1.6 | 0.316 |
| iris | JRip | kbins@5 | 92.00% | 4.0 | 8.3 | 2.1 | 0.308 |
| iris | JRip | kbins@8 | 95.33% | 2.7 | 3.9 | 1.4 | 0.309 |
| iris | JRip | raw | 92.67% | 2.6 | 4.3 | 1.7 | 0.310 |
| iris | JRip | tree@12 | 93.33% | 2.5 | 4.3 | 1.7 | 0.338 |
| iris | JRip | tree@3 | 92.00% | 2.1 | 3.1 | 1.4 | 0.303 |
| iris | JRip | tree@5 | 91.33% | 2.7 | 3.2 | 1.1 | 0.303 |
| iris | JRip | tree@8 | 95.33% | 2.5 | 3.9 | 1.6 | 0.325 |
| kc1 | J48 | kbins@12 | 84.40% | 51.2 | 537.8 | 10.5 | 0.817 |
| kc1 | J48 | kbins@3 | 84.54% | 1.0 | 0.0 | 0.0 | 0.433 |
| kc1 | J48 | kbins@5 | 83.97% | 11.7 | 59.9 | 4.5 | 0.513 |
| kc1 | J48 | kbins@8 | 85.11% | 17.7 | 125.9 | 6.7 | 0.630 |
| kc1 | J48 | raw | 84.50% | 55.4 | 520.0 | 9.1 | 0.594 |
| kc1 | J48 | tree@12 | 84.21% | 52.3 | 580.0 | 11.0 | 0.976 |
| kc1 | J48 | tree@3 | 85.44% | 19.4 | 135.4 | 6.7 | 0.482 |
| kc1 | J48 | tree@5 | 83.97% | 33.0 | 267.4 | 7.6 | 0.627 |
| kc1 | J48 | tree@8 | 84.97% | 50.6 | 543.7 | 10.7 | 0.765 |
| kc1 | JRip | kbins@12 | 84.12% | 2.3 | 8.3 | 3.5 | 0.896 |
| kc1 | JRip | kbins@3 | 84.54% | 0.0 | 0.0 | n/a | 0.413 |
| kc1 | JRip | kbins@5 | 84.16% | 1.2 | 4.1 | 3.3 | 0.544 |
| kc1 | JRip | kbins@8 | 84.92% | 1.9 | 6.8 | 3.5 | 0.746 |
| kc1 | JRip | raw | 84.73% | 2.3 | 6.9 | 2.8 | 0.708 |
| kc1 | JRip | tree@12 | 84.87% | 3.6 | 13.4 | 3.7 | 1.384 |
| kc1 | JRip | tree@3 | 85.02% | 2.6 | 8.6 | 3.2 | 0.567 |
| kc1 | JRip | tree@5 | 84.97% | 2.9 | 10.3 | 3.4 | 0.736 |
| kc1 | JRip | tree@8 | 85.21% | 3.0 | 11.2 | 3.6 | 1.003 |
| kc2 | J48 | kbins@12 | 83.31% | 20.1 | 148.3 | 6.9 | 0.478 |
| kc2 | J48 | kbins@3 | 81.21% | 9.0 | 40.2 | 4.1 | 0.355 |
| kc2 | J48 | kbins@5 | 82.17% | 7.2 | 28.2 | 3.2 | 0.382 |
| kc2 | J48 | kbins@8 | 80.82% | 14.6 | 95.7 | 5.6 | 0.422 |
| kc2 | J48 | raw | 82.36% | 18.9 | 106.9 | 5.2 | 0.416 |
| kc2 | J48 | tree@12 | 81.97% | 26.7 | 235.9 | 8.6 | 0.510 |
| kc2 | J48 | tree@3 | 84.27% | 11.0 | 53.9 | 4.6 | 0.370 |
| kc2 | J48 | tree@5 | 82.94% | 14.1 | 79.3 | 5.3 | 0.415 |
| kc2 | J48 | tree@8 | 83.88% | 19.8 | 148.1 | 7.0 | 0.463 |
| kc2 | JRip | kbins@12 | 82.16% | 1.8 | 4.6 | 2.4 | 0.458 |
| kc2 | JRip | kbins@3 | 81.98% | 1.1 | 4.1 | 3.8 | 0.346 |
| kc2 | JRip | kbins@5 | 82.55% | 1.3 | 2.4 | 1.8 | 0.394 |
| kc2 | JRip | kbins@8 | 82.55% | 1.6 | 4.0 | 2.3 | 0.427 |
| kc2 | JRip | raw | 82.35% | 1.6 | 3.1 | 1.8 | 0.429 |
| kc2 | JRip | tree@12 | 82.92% | 2.2 | 5.6 | 2.5 | 0.519 |
| kc2 | JRip | tree@3 | 83.89% | 1.4 | 2.8 | 1.9 | 0.377 |
| kc2 | JRip | tree@5 | 84.84% | 2.3 | 6.2 | 2.4 | 0.438 |
| kc2 | JRip | tree@8 | 82.92% | 2.4 | 5.8 | 2.1 | 0.465 |
| mfeat-morphological | J48 | kbins@12 | 69.40% | 47.9 | 375.9 | 7.8 | 0.536 |
| mfeat-morphological | J48 | kbins@3 | 66.00% | 13.0 | 55.7 | 4.3 | 0.405 |
| mfeat-morphological | J48 | kbins@5 | 65.65% | 24.1 | 147.6 | 6.1 | 0.441 |
| mfeat-morphological | J48 | kbins@8 | 69.80% | 32.0 | 219.5 | 6.8 | 0.470 |
| mfeat-morphological | J48 | raw | 72.35% | 80.1 | 704.3 | 8.8 | 0.537 |
| mfeat-morphological | J48 | tree@12 | 71.30% | 34.8 | 262.7 | 7.5 | 0.532 |
| mfeat-morphological | J48 | tree@3 | 67.35% | 16.0 | 76.6 | 4.7 | 0.405 |
| mfeat-morphological | J48 | tree@5 | 66.90% | 23.2 | 130.8 | 5.6 | 0.457 |
| mfeat-morphological | J48 | tree@8 | 69.55% | 27.9 | 175.1 | 6.2 | 0.487 |
| mfeat-morphological | JRip | kbins@12 | 68.35% | 18.4 | 52.7 | 2.8 | 0.753 |
| mfeat-morphological | JRip | kbins@3 | 56.50% | 9.1 | 22.8 | 2.5 | 0.485 |
| mfeat-morphological | JRip | kbins@5 | 56.70% | 10.6 | 28.9 | 2.7 | 0.555 |
| mfeat-morphological | JRip | kbins@8 | 68.15% | 16.4 | 44.4 | 2.7 | 0.676 |
| mfeat-morphological | JRip | raw | 69.55% | 20.4 | 48.7 | 2.4 | 0.727 |
| mfeat-morphological | JRip | tree@12 | 70.55% | 15.8 | 42.6 | 2.7 | 0.780 |
| mfeat-morphological | JRip | tree@3 | 59.75% | 9.8 | 24.4 | 2.5 | 0.505 |
| mfeat-morphological | JRip | tree@5 | 62.40% | 11.7 | 31.6 | 2.7 | 0.594 |
| mfeat-morphological | JRip | tree@8 | 65.90% | 14.0 | 35.3 | 2.5 | 0.677 |
| pc1 | J48 | kbins@12 | 92.61% | 11.1 | 60.2 | 4.5 | 0.581 |
| pc1 | J48 | kbins@3 | 93.06% | 1.0 | 0.0 | 0.0 | 0.386 |
| pc1 | J48 | kbins@5 | 93.06% | 1.0 | 0.0 | 0.0 | 0.435 |
| pc1 | J48 | kbins@8 | 93.42% | 3.5 | 7.7 | 1.9 | 0.482 |
| pc1 | J48 | raw | 93.60% | 11.6 | 49.4 | 3.9 | 0.474 |
| pc1 | J48 | tree@12 | 93.06% | 8.5 | 33.3 | 3.8 | 0.604 |
| pc1 | J48 | tree@3 | 92.79% | 4.8 | 12.8 | 2.5 | 0.400 |
| pc1 | J48 | tree@5 | 92.79% | 6.4 | 20.7 | 3.1 | 0.449 |
| pc1 | J48 | tree@8 | 93.06% | 7.3 | 26.1 | 3.3 | 0.527 |
| pc1 | JRip | kbins@12 | 92.70% | 0.5 | 1.9 | 3.9 | 0.629 |
| pc1 | JRip | kbins@3 | 93.06% | 0.0 | 0.0 | n/a | 0.381 |
| pc1 | JRip | kbins@5 | 93.06% | 0.0 | 0.0 | n/a | 0.492 |
| pc1 | JRip | kbins@8 | 92.79% | 0.4 | 2.0 | 5.0 | 0.525 |
| pc1 | JRip | raw | 93.60% | 2.2 | 5.9 | 2.7 | 0.536 |
| pc1 | JRip | tree@12 | 93.06% | 2.3 | 6.0 | 2.5 | 0.733 |
| pc1 | JRip | tree@3 | 92.97% | 1.8 | 5.0 | 2.7 | 0.413 |
| pc1 | JRip | tree@5 | 93.15% | 2.4 | 6.8 | 2.9 | 0.500 |
| pc1 | JRip | tree@8 | 92.97% | 2.6 | 7.4 | 2.7 | 0.618 |
| pc3 | J48 | kbins@12 | 87.46% | 45.6 | 575.6 | 12.3 | 0.868 |
| pc3 | J48 | kbins@3 | 89.44% | 7.8 | 62.1 | 2.5 | 0.475 |
| pc3 | J48 | kbins@5 | 88.80% | 35.9 | 380.8 | 10.3 | 0.568 |
| pc3 | J48 | kbins@8 | 87.97% | 40.4 | 444.6 | 10.7 | 0.727 |
| pc3 | J48 | raw | 86.88% | 37.5 | 343.0 | 8.2 | 0.621 |
| pc3 | J48 | tree@12 | 88.93% | 46.6 | 609.5 | 12.3 | 0.990 |
| pc3 | J48 | tree@3 | 88.36% | 28.7 | 226.3 | 7.7 | 0.505 |
| pc3 | J48 | tree@5 | 88.99% | 37.4 | 397.4 | 9.8 | 0.706 |
| pc3 | J48 | tree@8 | 88.99% | 52.0 | 662.1 | 12.7 | 0.811 |
| pc3 | JRip | kbins@12 | 88.61% | 2.6 | 11.0 | 4.3 | 1.080 |
| pc3 | JRip | kbins@3 | 89.12% | 0.5 | 2.6 | 5.2 | 0.493 |
| pc3 | JRip | kbins@5 | 88.61% | 1.8 | 8.0 | 4.4 | 0.625 |
| pc3 | JRip | kbins@8 | 89.12% | 2.5 | 11.5 | 4.5 | 0.927 |
| pc3 | JRip | raw | 89.44% | 1.9 | 7.6 | 4.1 | 0.686 |
| pc3 | JRip | tree@12 | 88.55% | 2.4 | 9.8 | 4.2 | 1.299 |
| pc3 | JRip | tree@3 | 88.80% | 2.2 | 9.3 | 4.4 | 0.540 |
| pc3 | JRip | tree@5 | 87.65% | 2.8 | 12.6 | 4.4 | 0.773 |
| pc3 | JRip | tree@8 | 88.36% | 1.9 | 8.3 | 4.3 | 0.980 |
| pc4 | J48 | kbins@12 | 89.23% | 61.3 | 925.5 | 15.0 | 0.834 |
| pc4 | J48 | kbins@3 | 88.13% | 42.6 | 420.3 | 9.8 | 0.480 |
| pc4 | J48 | kbins@5 | 89.16% | 31.2 | 312.7 | 9.9 | 0.543 |
| pc4 | J48 | kbins@8 | 88.27% | 59.5 | 888.2 | 14.9 | 0.726 |
| pc4 | J48 | raw | 89.30% | 39.1 | 378.9 | 9.2 | 0.584 |
| pc4 | J48 | tree@12 | 88.68% | 57.5 | 1225.5 | 21.2 | 0.978 |
| pc4 | J48 | tree@3 | 89.44% | 30.9 | 318.6 | 10.0 | 0.497 |
| pc4 | J48 | tree@5 | 89.16% | 48.4 | 667.3 | 13.3 | 0.622 |
| pc4 | J48 | tree@8 | 88.13% | 55.4 | 950.9 | 17.1 | 0.776 |
| pc4 | JRip | kbins@12 | 88.34% | 5.1 | 25.1 | 4.9 | 1.158 |
| pc4 | JRip | kbins@3 | 87.11% | 3.0 | 13.8 | 4.5 | 0.534 |
| pc4 | JRip | kbins@5 | 88.41% | 2.6 | 12.1 | 4.6 | 0.693 |
| pc4 | JRip | kbins@8 | 88.21% | 4.9 | 25.6 | 5.2 | 1.002 |
| pc4 | JRip | raw | 89.16% | 3.4 | 14.1 | 4.1 | 0.707 |
| pc4 | JRip | tree@12 | 87.65% | 3.9 | 17.0 | 4.3 | 1.413 |
| pc4 | JRip | tree@3 | 89.50% | 3.6 | 16.6 | 4.5 | 0.557 |
| pc4 | JRip | tree@5 | 87.92% | 4.6 | 22.3 | 4.7 | 0.806 |
| pc4 | JRip | tree@8 | 87.99% | 4.6 | 21.6 | 4.6 | 1.085 |
| phoneme | J48 | kbins@12 | 87.31% | 175.8 | 1801.3 | 10.2 | 1.153 |
| phoneme | J48 | kbins@3 | 81.38% | 25.1 | 129.1 | 5.1 | 0.517 |
| phoneme | J48 | kbins@5 | 84.20% | 88.5 | 741.5 | 8.3 | 0.740 |
| phoneme | J48 | kbins@8 | 86.07% | 155.8 | 1579.1 | 10.1 | 1.004 |
| phoneme | J48 | raw | 86.86% | 116.0 | 1046.1 | 9.0 | 0.728 |
| phoneme | J48 | tree@12 | 85.86% | 152.6 | 1578.8 | 10.3 | 1.092 |
| phoneme | J48 | tree@3 | 80.51% | 28.2 | 154.1 | 5.5 | 0.510 |
| phoneme | J48 | tree@5 | 82.49% | 60.3 | 438.2 | 7.2 | 0.641 |
| phoneme | J48 | tree@8 | 85.38% | 116.2 | 1057.5 | 9.0 | 0.909 |
| phoneme | JRip | kbins@12 | 86.07% | 20.3 | 110.4 | 5.4 | 2.865 |
| phoneme | JRip | kbins@3 | 81.46% | 7.9 | 33.2 | 4.2 | 0.730 |
| phoneme | JRip | kbins@5 | 83.33% | 14.2 | 73.9 | 5.2 | 1.278 |
| phoneme | JRip | kbins@8 | 84.71% | 20.5 | 107.9 | 5.2 | 2.146 |
| phoneme | JRip | raw | 85.27% | 17.7 | 79.3 | 4.4 | 1.346 |
| phoneme | JRip | tree@12 | 84.47% | 21.5 | 113.7 | 5.3 | 2.831 |
| phoneme | JRip | tree@3 | 80.20% | 7.8 | 33.2 | 4.2 | 0.694 |
| phoneme | JRip | tree@5 | 82.20% | 12.1 | 59.4 | 4.9 | 1.097 |
| phoneme | JRip | tree@8 | 83.81% | 16.7 | 86.5 | 5.2 | 1.876 |
| qsar-biodeg | J48 | kbins@12 | 82.18% | 53.1 | 537.6 | 10.1 | 0.639 |
| qsar-biodeg | J48 | kbins@3 | 81.42% | 51.5 | 382.6 | 7.4 | 0.435 |
| qsar-biodeg | J48 | kbins@5 | 83.33% | 54.7 | 449.5 | 8.2 | 0.491 |
| qsar-biodeg | J48 | kbins@8 | 82.37% | 54.0 | 557.2 | 10.3 | 0.572 |
| qsar-biodeg | J48 | raw | 82.85% | 56.7 | 490.9 | 8.7 | 0.570 |
| qsar-biodeg | J48 | tree@12 | 83.80% | 47.3 | 491.6 | 10.3 | 0.695 |
| qsar-biodeg | J48 | tree@3 | 82.38% | 42.1 | 386.5 | 9.0 | 0.486 |
| qsar-biodeg | J48 | tree@5 | 82.29% | 51.6 | 539.3 | 10.4 | 0.593 |
| qsar-biodeg | J48 | tree@8 | 83.61% | 42.2 | 389.8 | 9.3 | 0.632 |
| qsar-biodeg | JRip | kbins@12 | 82.18% | 7.6 | 31.1 | 4.1 | 0.982 |
| qsar-biodeg | JRip | kbins@3 | 81.23% | 6.7 | 27.3 | 4.1 | 0.493 |
| qsar-biodeg | JRip | kbins@5 | 83.41% | 6.4 | 25.7 | 4.0 | 0.568 |
| qsar-biodeg | JRip | kbins@8 | 82.75% | 7.8 | 32.9 | 4.2 | 0.767 |
| qsar-biodeg | JRip | raw | 83.32% | 5.8 | 20.0 | 3.4 | 0.685 |
| qsar-biodeg | JRip | tree@12 | 82.09% | 7.3 | 27.7 | 3.8 | 1.166 |
| qsar-biodeg | JRip | tree@3 | 82.74% | 5.8 | 24.4 | 4.2 | 0.564 |
| qsar-biodeg | JRip | tree@5 | 82.19% | 6.5 | 24.9 | 3.8 | 0.714 |
| qsar-biodeg | JRip | tree@8 | 81.90% | 6.7 | 26.8 | 4.0 | 0.970 |
| segment | J48 | kbins@12 | 95.58% | 51.9 | 440.1 | 8.5 | 0.775 |
| segment | J48 | kbins@3 | 91.90% | 54.5 | 372.3 | 6.8 | 0.521 |
| segment | J48 | kbins@5 | 93.55% | 60.7 | 494.3 | 8.1 | 0.583 |
| segment | J48 | kbins@8 | 94.55% | 53.8 | 512.5 | 9.5 | 0.691 |
| segment | J48 | raw | 96.58% | 40.9 | 319.0 | 7.8 | 0.607 |
| segment | J48 | tree@12 | 95.67% | 44.6 | 432.9 | 9.7 | 0.760 |
| segment | J48 | tree@3 | 90.52% | 36.8 | 310.9 | 8.4 | 0.511 |
| segment | J48 | tree@5 | 94.89% | 34.7 | 284.0 | 8.1 | 0.562 |
| segment | J48 | tree@8 | 95.28% | 42.0 | 399.0 | 9.5 | 0.678 |
| segment | JRip | kbins@12 | 94.50% | 22.9 | 67.3 | 2.9 | 1.725 |
| segment | JRip | kbins@3 | 90.82% | 22.8 | 80.7 | 3.5 | 0.770 |
| segment | JRip | kbins@5 | 91.95% | 24.0 | 80.2 | 3.3 | 1.045 |
| segment | JRip | kbins@8 | 94.42% | 22.6 | 65.1 | 2.9 | 1.268 |
| segment | JRip | raw | 95.06% | 18.1 | 52.1 | 2.9 | 0.908 |
| segment | JRip | tree@12 | 94.98% | 19.7 | 60.8 | 3.1 | 1.770 |
| segment | JRip | tree@3 | 89.87% | 14.7 | 42.9 | 2.9 | 0.683 |
| segment | JRip | tree@5 | 93.77% | 19.0 | 54.3 | 2.8 | 0.947 |
| segment | JRip | tree@8 | 95.15% | 20.0 | 58.2 | 2.9 | 1.231 |
| sonar | J48 | kbins@12 | 74.60% | 18.9 | 115.6 | 6.1 | 0.486 |
| sonar | J48 | kbins@3 | 79.33% | 18.1 | 89.1 | 4.9 | 0.355 |
| sonar | J48 | kbins@5 | 75.57% | 18.9 | 107.0 | 5.6 | 0.393 |
| sonar | J48 | kbins@8 | 72.14% | 18.0 | 94.6 | 5.2 | 0.449 |
| sonar | J48 | raw | 76.88% | 14.7 | 68.6 | 4.7 | 0.394 |
| sonar | J48 | tree@12 | 67.86% | 18.2 | 123.6 | 6.7 | 0.491 |
| sonar | J48 | tree@3 | 70.67% | 16.7 | 93.6 | 5.5 | 0.357 |
| sonar | J48 | tree@5 | 69.76% | 19.0 | 132.0 | 6.8 | 0.396 |
| sonar | J48 | tree@8 | 70.14% | 18.1 | 119.8 | 6.5 | 0.458 |
| sonar | JRip | kbins@12 | 75.57% | 4.4 | 9.8 | 2.2 | 0.512 |
| sonar | JRip | kbins@3 | 70.19% | 4.1 | 9.9 | 2.4 | 0.372 |
| sonar | JRip | kbins@5 | 75.57% | 4.3 | 9.3 | 2.1 | 0.409 |
| sonar | JRip | kbins@8 | 72.52% | 4.2 | 9.2 | 2.1 | 0.468 |
| sonar | JRip | raw | 73.62% | 3.2 | 6.7 | 2.1 | 0.431 |
| sonar | JRip | tree@12 | 72.60% | 4.9 | 9.7 | 2.0 | 0.539 |
| sonar | JRip | tree@3 | 73.64% | 4.8 | 11.7 | 2.5 | 0.384 |
| sonar | JRip | tree@5 | 73.07% | 4.7 | 10.5 | 2.2 | 0.426 |
| sonar | JRip | tree@8 | 72.60% | 4.7 | 10.1 | 2.1 | 0.476 |
| steel-plates-fault | J48 | kbins@12 | 71.41% | 204.6 | 2788.7 | 13.6 | 1.115 |
| steel-plates-fault | J48 | kbins@3 | 66.36% | 206.4 | 1990.2 | 9.6 | 0.724 |
| steel-plates-fault | J48 | kbins@5 | 70.79% | 198.2 | 2219.1 | 11.2 | 0.781 |
| steel-plates-fault | J48 | kbins@8 | 72.13% | 193.4 | 2332.1 | 12.0 | 0.916 |
| steel-plates-fault | J48 | raw | 75.02% | 162.0 | 1923.9 | 11.9 | 0.731 |
| steel-plates-fault | J48 | tree@12 | 72.59% | 189.7 | 2983.8 | 15.7 | 1.154 |
| steel-plates-fault | J48 | tree@3 | 71.00% | 127.7 | 1351.6 | 10.6 | 0.644 |
| steel-plates-fault | J48 | tree@5 | 73.16% | 155.9 | 1973.7 | 12.6 | 0.786 |
| steel-plates-fault | J48 | tree@8 | 70.58% | 181.1 | 2616.2 | 14.4 | 0.976 |
| steel-plates-fault | JRip | kbins@12 | 71.41% | 25.8 | 104.7 | 4.1 | 3.196 |
| steel-plates-fault | JRip | kbins@3 | 67.23% | 26.2 | 126.4 | 4.8 | 1.045 |
| steel-plates-fault | JRip | kbins@5 | 69.04% | 23.3 | 101.7 | 4.4 | 1.510 |
| steel-plates-fault | JRip | kbins@8 | 71.46% | 24.1 | 97.1 | 4.0 | 2.139 |
| steel-plates-fault | JRip | raw | 72.02% | 21.3 | 70.4 | 3.3 | 1.213 |
| steel-plates-fault | JRip | tree@12 | 73.42% | 25.3 | 99.1 | 3.9 | 3.287 |
| steel-plates-fault | JRip | tree@3 | 70.79% | 22.4 | 90.4 | 4.0 | 1.054 |
| steel-plates-fault | JRip | tree@5 | 75.17% | 21.7 | 83.7 | 3.8 | 1.451 |
| steel-plates-fault | JRip | tree@8 | 74.60% | 24.6 | 95.7 | 3.9 | 2.297 |
| vehicle | J48 | kbins@12 | 70.21% | 94.1 | 1017.7 | 10.8 | 0.604 |
| vehicle | J48 | kbins@3 | 67.39% | 61.6 | 489.9 | 7.9 | 0.419 |
| vehicle | J48 | kbins@5 | 68.46% | 87.6 | 861.8 | 9.8 | 0.492 |
| vehicle | J48 | kbins@8 | 70.10% | 91.7 | 964.3 | 10.5 | 0.547 |
| vehicle | J48 | raw | 73.07% | 68.4 | 549.2 | 8.0 | 0.449 |
| vehicle | J48 | tree@12 | 71.41% | 90.0 | 1066.5 | 11.8 | 0.627 |
| vehicle | J48 | tree@3 | 69.04% | 48.7 | 344.6 | 7.0 | 0.428 |
| vehicle | J48 | tree@5 | 67.26% | 66.4 | 612.2 | 9.1 | 0.480 |
| vehicle | J48 | tree@8 | 68.44% | 83.6 | 933.4 | 11.1 | 0.558 |
| vehicle | JRip | kbins@12 | 66.44% | 13.9 | 39.6 | 2.8 | 0.881 |
| vehicle | JRip | kbins@3 | 64.67% | 9.5 | 25.8 | 2.7 | 0.473 |
| vehicle | JRip | kbins@5 | 64.54% | 12.6 | 34.7 | 2.7 | 0.588 |
| vehicle | JRip | kbins@8 | 68.09% | 14.5 | 40.5 | 2.8 | 0.733 |
| vehicle | JRip | raw | 69.99% | 15.0 | 38.4 | 2.6 | 0.566 |
| vehicle | JRip | tree@12 | 69.99% | 13.4 | 33.6 | 2.5 | 0.885 |
| vehicle | JRip | tree@3 | 62.77% | 10.3 | 30.8 | 3.0 | 0.490 |
| vehicle | JRip | tree@5 | 68.57% | 12.6 | 34.0 | 2.7 | 0.571 |
| vehicle | JRip | tree@8 | 68.45% | 13.9 | 38.4 | 2.8 | 0.726 |
| wdbc | J48 | kbins@12 | 95.78% | 11.4 | 45.6 | 3.9 | 0.523 |
| wdbc | J48 | kbins@3 | 95.43% | 9.2 | 34.1 | 3.7 | 0.363 |
| wdbc | J48 | kbins@5 | 93.85% | 13.3 | 57.3 | 4.3 | 0.406 |
| wdbc | J48 | kbins@8 | 94.02% | 13.5 | 58.6 | 4.3 | 0.482 |
| wdbc | J48 | raw | 93.86% | 11.3 | 45.9 | 4.0 | 0.432 |
| wdbc | J48 | tree@12 | 95.08% | 11.4 | 49.3 | 4.3 | 0.522 |
| wdbc | J48 | tree@3 | 95.60% | 11.8 | 50.2 | 4.2 | 0.371 |
| wdbc | J48 | tree@5 | 95.44% | 11.6 | 52.0 | 4.4 | 0.414 |
| wdbc | J48 | tree@8 | 95.44% | 10.9 | 45.4 | 4.1 | 0.470 |
| wdbc | JRip | kbins@12 | 95.25% | 4.1 | 8.7 | 2.1 | 0.591 |
| wdbc | JRip | kbins@3 | 94.02% | 4.4 | 10.3 | 2.3 | 0.390 |
| wdbc | JRip | kbins@5 | 93.33% | 5.3 | 13.0 | 2.4 | 0.445 |
| wdbc | JRip | kbins@8 | 95.26% | 5.1 | 11.9 | 2.4 | 0.509 |
| wdbc | JRip | raw | 94.91% | 3.6 | 7.3 | 2.0 | 0.464 |
| wdbc | JRip | tree@12 | 95.60% | 5.1 | 10.9 | 2.1 | 0.595 |
| wdbc | JRip | tree@3 | 95.25% | 4.6 | 10.5 | 2.3 | 0.393 |
| wdbc | JRip | tree@5 | 93.85% | 4.8 | 10.8 | 2.3 | 0.437 |
| wdbc | JRip | tree@8 | 94.55% | 4.8 | 10.7 | 2.2 | 0.509 |
| wilt | J48 | kbins@12 | 98.28% | 27.6 | 185.7 | 6.7 | 0.694 |
| wilt | J48 | kbins@3 | 96.09% | 10.9 | 58.0 | 5.2 | 0.447 |
| wilt | J48 | kbins@5 | 97.66% | 12.7 | 58.9 | 4.5 | 0.485 |
| wilt | J48 | kbins@8 | 98.31% | 23.1 | 146.0 | 6.3 | 0.577 |
| wilt | J48 | raw | 98.10% | 27.1 | 162.5 | 5.9 | 0.601 |
| wilt | J48 | tree@12 | 97.40% | 7.4 | 29.6 | 3.9 | 0.602 |
| wilt | J48 | tree@3 | 96.94% | 3.4 | 6.6 | 1.9 | 0.409 |
| wilt | J48 | tree@5 | 97.00% | 7.1 | 28.5 | 3.8 | 0.481 |
| wilt | J48 | tree@8 | 97.40% | 7.4 | 28.9 | 3.7 | 0.528 |
| wilt | JRip | kbins@12 | 98.00% | 6.1 | 20.3 | 3.3 | 0.960 |
| wilt | JRip | kbins@3 | 95.89% | 3.3 | 12.6 | 3.7 | 0.516 |
| wilt | JRip | kbins@5 | 97.54% | 3.8 | 12.0 | 3.1 | 0.607 |
| wilt | JRip | kbins@8 | 98.02% | 5.1 | 16.1 | 3.1 | 0.739 |
| wilt | JRip | raw | 98.22% | 7.0 | 17.7 | 2.5 | 0.793 |
| wilt | JRip | tree@12 | 97.35% | 2.1 | 5.9 | 2.8 | 0.756 |
| wilt | JRip | tree@3 | 97.00% | 1.1 | 2.3 | 2.0 | 0.450 |
| wilt | JRip | tree@5 | 97.11% | 1.9 | 6.0 | 3.0 | 0.537 |
| wilt | JRip | tree@8 | 97.35% | 2.2 | 6.1 | 2.8 | 0.620 |
| wine | J48 | kbins@12 | 96.11% | 5.5 | 14.5 | 2.6 | 0.373 |
| wine | J48 | kbins@3 | 94.41% | 7.1 | 22.5 | 3.1 | 0.335 |
| wine | J48 | kbins@5 | 93.27% | 6.0 | 16.6 | 2.7 | 0.331 |
| wine | J48 | kbins@8 | 90.42% | 7.0 | 23.5 | 3.2 | 0.346 |
| wine | J48 | raw | 93.14% | 5.1 | 12.5 | 2.4 | 0.339 |
| wine | J48 | tree@12 | 91.01% | 6.4 | 19.4 | 2.9 | 0.360 |
| wine | J48 | tree@3 | 89.25% | 5.8 | 15.7 | 2.6 | 0.322 |
| wine | J48 | tree@5 | 92.12% | 6.3 | 19.1 | 2.9 | 0.329 |
| wine | J48 | tree@8 | 91.57% | 6.5 | 19.9 | 3.0 | 0.344 |
| wine | JRip | kbins@12 | 92.06% | 2.8 | 5.3 | 1.9 | 0.379 |
| wine | JRip | kbins@3 | 90.95% | 3.1 | 6.8 | 2.2 | 0.334 |
| wine | JRip | kbins@5 | 89.25% | 3.4 | 6.1 | 1.8 | 0.345 |
| wine | JRip | kbins@8 | 90.42% | 3.0 | 5.7 | 1.9 | 0.359 |
| wine | JRip | raw | 95.49% | 2.9 | 5.4 | 1.9 | 0.342 |
| wine | JRip | tree@12 | 94.44% | 2.7 | 5.2 | 1.9 | 0.372 |
| wine | JRip | tree@3 | 90.46% | 2.9 | 5.7 | 1.9 | 0.342 |
| wine | JRip | tree@5 | 92.22% | 2.8 | 5.3 | 1.9 | 0.336 |
| wine | JRip | tree@8 | 94.41% | 2.7 | 5.1 | 1.9 | 0.362 |
| yeast | J48 | kbins@12 | 53.43% | 168.8 | 2045.8 | 12.0 | 0.662 |
| yeast | J48 | kbins@3 | 49.40% | 48.2 | 314.7 | 6.5 | 0.449 |
| yeast | J48 | kbins@5 | 57.27% | 110.7 | 1014.4 | 9.1 | 0.544 |
| yeast | J48 | kbins@8 | 53.30% | 159.2 | 1785.8 | 11.2 | 0.611 |
| yeast | J48 | raw | 56.46% | 167.5 | 1734.7 | 10.3 | 0.536 |
| yeast | J48 | tree@12 | 55.86% | 125.4 | 1578.3 | 12.6 | 0.658 |
| yeast | J48 | tree@3 | 58.36% | 31.2 | 192.3 | 6.1 | 0.436 |
| yeast | J48 | tree@5 | 57.75% | 63.7 | 556.6 | 8.7 | 0.505 |
| yeast | J48 | tree@8 | 55.52% | 106.5 | 1232.1 | 11.5 | 0.602 |
| yeast | JRip | kbins@12 | 56.60% | 14.0 | 46.7 | 3.3 | 0.949 |
| yeast | JRip | kbins@3 | 44.81% | 9.3 | 39.0 | 4.2 | 0.518 |
| yeast | JRip | kbins@5 | 56.06% | 13.4 | 47.7 | 3.6 | 0.623 |
| yeast | JRip | kbins@8 | 55.93% | 13.3 | 44.9 | 3.4 | 0.729 |
| yeast | JRip | raw | 57.28% | 15.2 | 38.4 | 2.5 | 0.673 |
| yeast | JRip | tree@12 | 59.70% | 14.8 | 39.7 | 2.7 | 0.897 |
| yeast | JRip | tree@3 | 59.97% | 11.8 | 28.5 | 2.4 | 0.518 |
| yeast | JRip | tree@5 | 59.16% | 14.0 | 38.5 | 2.7 | 0.631 |
| yeast | JRip | tree@8 | 58.89% | 15.7 | 42.5 | 2.7 | 0.766 |

