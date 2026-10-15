# Numeric attributes: intrinsic handling vs. discretization (JRip / J48)

## Setup

**Full run** (27 small and medium-sized datasets, all attributes numeric, 10-fold). For a quick sanity check instead (5 small datasets, 3-fold): `python demos/numeric_discretization.py` with no arguments -- its output isn't checked in (see `numeric_discretization_quick_report.md` after running it).

Weka's JRip and J48 compared across seven data preparations per dataset: `raw`
(Weka's own intrinsic numeric thresholding), our own decision-tree-based
discretization (`build_dataspec`/`tree_thresholds`) and `sklearn.preprocessing.
KBinsDiscretizer` (`strategy='quantile'`, unsupervised), each of the
latter two at 3, 5, 8 intervals. Both discretizers
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
| tree@3 | 91.5 |
| tree@5 | 164.0 |
| tree@8 | 271.8 |
| kbins@3 | 69.7 |
| kbins@5 | 136.3 |
| kbins@8 | 231.6 |

## Summary by variant (mean across every dataset and fold)

| variant | accuracy | n_rules | conds/rule | fit_time (s) | wins | mean rank |
|---|--:|--:|--:|--:|--:|--:|
| JRip:raw | 84.36% | 7.5 | 2.6 | 0.57 | 1.0 | 5.00 |
| J48:raw | 84.32% | 40.7 | 6.3 | 0.46 | 6.0 | 5.76 |
| JRip:tree@8 | 83.98% | 7.4 | 2.8 | 1.30 | 3.0 | 6.15 |
| J48:tree@8 | 83.28% | 38.0 | 7.3 | 1.10 | 3.0 | 6.91 |
| J48:kbins@8 | 83.33% | 43.0 | 7.0 | 1.00 | 4.0 | 6.93 |
| JRip:kbins@8 | 83.75% | 7.7 | 3.1 | 1.17 | 1.0 | 7.17 |
| J48:tree@5 | 83.17% | 30.5 | 6.5 | 0.81 | 2.0 | 7.35 |
| JRip:tree@5 | 83.18% | 6.9 | 2.8 | 0.93 | 0.0 | 7.56 |
| JRip:kbins@5 | 83.03% | 7.0 | 3.0 | 0.85 | 0.0 | 7.93 |
| JRip:tree@3 | 82.74% | 6.1 | 2.7 | 0.68 | 1.0 | 8.19 |
| J48:tree@3 | 82.57% | 23.3 | 5.4 | 0.62 | 2.0 | 8.22 |
| J48:kbins@5 | 82.77% | 36.6 | 6.0 | 0.74 | 0.0 | 8.24 |
| J48:kbins@3 | 82.48% | 27.5 | 4.8 | 0.56 | 2.0 | 9.13 |
| JRip:kbins@3 | 81.48% | 6.1 | 3.2 | 0.62 | 2.0 | 10.48 |

`wins` -- datasets where a variant's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last.

## Per-dataset results, per variant (mean across folds)

| dataset | algorithm | prep | accuracy | n_rules | n_conditions | conds_per_rule | fit_time |
|---|---|---|---|---|---|---|---|
| analcatdata_authorship | J48 | kbins@3 | 92.39% | 26.2 | 145.9 | 5.5 | 1.030 |
| analcatdata_authorship | J48 | kbins@5 | 90.61% | 25.5 | 140.2 | 5.5 | 1.630 |
| analcatdata_authorship | J48 | kbins@8 | 93.58% | 25.3 | 148.7 | 5.9 | 2.380 |
| analcatdata_authorship | J48 | raw | 95.12% | 20.2 | 97.5 | 4.8 | 0.594 |
| analcatdata_authorship | J48 | tree@3 | 93.46% | 20.6 | 107.0 | 5.2 | 1.178 |
| analcatdata_authorship | J48 | tree@5 | 93.46% | 19.7 | 100.6 | 5.1 | 1.723 |
| analcatdata_authorship | J48 | tree@8 | 94.53% | 18.3 | 91.4 | 5.0 | 2.618 |
| analcatdata_authorship | JRip | kbins@3 | 92.74% | 11.2 | 31.5 | 2.8 | 1.206 |
| analcatdata_authorship | JRip | kbins@5 | 93.46% | 10.0 | 25.4 | 2.6 | 1.841 |
| analcatdata_authorship | JRip | kbins@8 | 93.34% | 8.7 | 21.1 | 2.4 | 2.710 |
| analcatdata_authorship | JRip | raw | 94.65% | 9.8 | 18.7 | 1.9 | 0.729 |
| analcatdata_authorship | JRip | tree@3 | 95.13% | 10.8 | 21.6 | 2.0 | 1.282 |
| analcatdata_authorship | JRip | tree@5 | 93.57% | 10.1 | 19.8 | 2.0 | 2.006 |
| analcatdata_authorship | JRip | tree@8 | 93.70% | 10.0 | 19.7 | 2.0 | 3.042 |
| balance-scale | J48 | kbins@3 | 78.23% | 24.4 | 128.1 | 5.2 | 0.388 |
| balance-scale | J48 | kbins@5 | 79.34% | 24.0 | 134.7 | 5.5 | 0.405 |
| balance-scale | J48 | kbins@8 | 78.86% | 34.5 | 223.9 | 6.4 | 0.452 |
| balance-scale | J48 | raw | 78.23% | 42.9 | 278.0 | 6.5 | 0.367 |
| balance-scale | J48 | tree@3 | 77.58% | 40.2 | 309.3 | 7.7 | 0.452 |
| balance-scale | J48 | tree@5 | 77.58% | 40.2 | 309.3 | 7.7 | 0.429 |
| balance-scale | J48 | tree@8 | 77.58% | 40.2 | 309.3 | 7.7 | 0.464 |
| balance-scale | JRip | kbins@3 | 78.69% | 8.1 | 25.6 | 3.1 | 0.422 |
| balance-scale | JRip | kbins@5 | 79.98% | 7.4 | 21.9 | 2.9 | 0.443 |
| balance-scale | JRip | kbins@8 | 82.71% | 9.5 | 30.6 | 3.2 | 0.431 |
| balance-scale | JRip | raw | 79.35% | 10.0 | 29.9 | 3.0 | 0.427 |
| balance-scale | JRip | tree@3 | 76.31% | 7.4 | 23.8 | 3.2 | 0.476 |
| balance-scale | JRip | tree@5 | 76.31% | 7.4 | 23.8 | 3.2 | 0.488 |
| balance-scale | JRip | tree@8 | 76.31% | 7.4 | 23.8 | 3.2 | 0.493 |
| banknote-authentication | J48 | kbins@3 | 94.39% | 15.1 | 74.3 | 4.9 | 0.437 |
| banknote-authentication | J48 | kbins@5 | 98.47% | 20.1 | 99.4 | 4.9 | 0.554 |
| banknote-authentication | J48 | kbins@8 | 99.27% | 17.7 | 88.8 | 5.0 | 0.616 |
| banknote-authentication | J48 | raw | 98.69% | 15.6 | 71.0 | 4.5 | 0.432 |
| banknote-authentication | J48 | tree@3 | 91.26% | 4.9 | 12.4 | 2.5 | 0.476 |
| banknote-authentication | J48 | tree@5 | 97.01% | 9.4 | 33.5 | 3.5 | 0.500 |
| banknote-authentication | J48 | tree@8 | 98.69% | 11.3 | 44.3 | 3.9 | 0.571 |
| banknote-authentication | JRip | kbins@3 | 93.66% | 5.4 | 16.8 | 3.1 | 0.494 |
| banknote-authentication | JRip | kbins@5 | 98.32% | 7.7 | 23.8 | 3.1 | 0.552 |
| banknote-authentication | JRip | kbins@8 | 99.12% | 8.3 | 24.4 | 2.9 | 0.684 |
| banknote-authentication | JRip | raw | 97.96% | 7.1 | 16.0 | 2.3 | 0.471 |
| banknote-authentication | JRip | tree@3 | 91.04% | 2.4 | 5.2 | 2.1 | 0.433 |
| banknote-authentication | JRip | tree@5 | 97.23% | 4.7 | 10.4 | 2.2 | 0.541 |
| banknote-authentication | JRip | tree@8 | 98.91% | 5.3 | 12.0 | 2.3 | 0.643 |
| blood-transfusion-service-center | J48 | kbins@3 | 77.81% | 4.0 | 9.0 | 2.2 | 0.376 |
| blood-transfusion-service-center | J48 | kbins@5 | 77.00% | 6.1 | 20.9 | 3.4 | 0.438 |
| blood-transfusion-service-center | J48 | kbins@8 | 78.33% | 6.5 | 24.3 | 3.5 | 0.485 |
| blood-transfusion-service-center | J48 | raw | 78.99% | 6.8 | 26.9 | 3.6 | 0.353 |
| blood-transfusion-service-center | J48 | tree@3 | 79.27% | 5.0 | 14.0 | 2.8 | 0.364 |
| blood-transfusion-service-center | J48 | tree@5 | 79.40% | 5.4 | 16.2 | 3.0 | 0.437 |
| blood-transfusion-service-center | J48 | tree@8 | 78.74% | 6.6 | 22.3 | 3.3 | 0.471 |
| blood-transfusion-service-center | JRip | kbins@3 | 76.60% | 1.0 | 3.1 | 3.1 | 0.419 |
| blood-transfusion-service-center | JRip | kbins@5 | 77.40% | 1.2 | 3.7 | 3.0 | 0.431 |
| blood-transfusion-service-center | JRip | kbins@8 | 78.47% | 1.2 | 4.1 | 3.4 | 0.477 |
| blood-transfusion-service-center | JRip | raw | 78.34% | 1.9 | 5.2 | 2.8 | 0.407 |
| blood-transfusion-service-center | JRip | tree@3 | 79.27% | 2.0 | 5.1 | 2.5 | 0.429 |
| blood-transfusion-service-center | JRip | tree@5 | 79.00% | 2.1 | 5.4 | 2.5 | 0.478 |
| blood-transfusion-service-center | JRip | tree@8 | 79.14% | 2.0 | 5.1 | 2.5 | 0.527 |
| breast-w | J48 | kbins@3 | 93.42% | 11.5 | 45.1 | 3.9 | 0.389 |
| breast-w | J48 | kbins@5 | 94.42% | 8.6 | 29.4 | 3.4 | 0.423 |
| breast-w | J48 | kbins@8 | 94.14% | 10.3 | 39.8 | 3.8 | 0.493 |
| breast-w | J48 | raw | 92.99% | 12.2 | 52.1 | 4.2 | 0.369 |
| breast-w | J48 | tree@3 | 92.56% | 12.4 | 61.9 | 4.8 | 0.707 |
| breast-w | J48 | tree@5 | 92.56% | 12.4 | 61.9 | 4.8 | 0.728 |
| breast-w | J48 | tree@8 | 92.56% | 12.4 | 61.9 | 4.8 | 0.729 |
| breast-w | JRip | kbins@3 | 96.14% | 3.6 | 8.7 | 2.4 | 0.412 |
| breast-w | JRip | kbins@5 | 94.85% | 4.2 | 8.6 | 2.0 | 0.448 |
| breast-w | JRip | kbins@8 | 95.13% | 4.4 | 9.2 | 2.1 | 0.484 |
| breast-w | JRip | raw | 95.42% | 5.0 | 10.1 | 2.0 | 0.414 |
| breast-w | JRip | tree@3 | 94.84% | 5.1 | 12.9 | 2.5 | 0.789 |
| breast-w | JRip | tree@5 | 94.84% | 5.1 | 12.9 | 2.5 | 0.782 |
| breast-w | JRip | tree@8 | 94.84% | 5.1 | 12.9 | 2.5 | 0.777 |
| climate-model-simulation-crashes | J48 | kbins@3 | 89.81% | 14.8 | 82.9 | 5.6 | 0.491 |
| climate-model-simulation-crashes | J48 | kbins@5 | 91.85% | 15.2 | 89.3 | 5.8 | 0.601 |
| climate-model-simulation-crashes | J48 | kbins@8 | 90.19% | 15.7 | 101.1 | 6.3 | 0.766 |
| climate-model-simulation-crashes | J48 | raw | 91.85% | 12.1 | 59.6 | 4.9 | 0.401 |
| climate-model-simulation-crashes | J48 | tree@3 | 93.70% | 8.8 | 31.8 | 3.5 | 0.453 |
| climate-model-simulation-crashes | J48 | tree@5 | 93.15% | 10.0 | 42.2 | 4.1 | 0.587 |
| climate-model-simulation-crashes | J48 | tree@8 | 93.33% | 10.6 | 45.8 | 4.2 | 0.805 |
| climate-model-simulation-crashes | JRip | kbins@3 | 91.85% | 2.6 | 10.6 | 4.1 | 0.531 |
| climate-model-simulation-crashes | JRip | kbins@5 | 94.07% | 2.6 | 9.3 | 3.5 | 0.620 |
| climate-model-simulation-crashes | JRip | kbins@8 | 90.93% | 2.9 | 11.4 | 4.0 | 0.828 |
| climate-model-simulation-crashes | JRip | raw | 93.15% | 2.0 | 6.6 | 3.3 | 0.453 |
| climate-model-simulation-crashes | JRip | tree@3 | 91.48% | 2.5 | 7.5 | 3.0 | 0.537 |
| climate-model-simulation-crashes | JRip | tree@5 | 93.89% | 2.7 | 8.4 | 3.1 | 0.630 |
| climate-model-simulation-crashes | JRip | tree@8 | 94.81% | 2.7 | 8.1 | 3.0 | 0.831 |
| diabetes | J48 | kbins@3 | 64.45% | 22.7 | 138.8 | 6.0 | 0.459 |
| diabetes | J48 | kbins@5 | 55.34% | 57.3 | 512.2 | 8.9 | 0.570 |
| diabetes | J48 | kbins@8 | 53.88% | 52.5 | 564.4 | 10.5 | 0.628 |
| diabetes | J48 | raw | 69.91% | 16.1 | 83.0 | 5.1 | 0.412 |
| diabetes | J48 | tree@3 | 58.06% | 27.8 | 190.1 | 6.7 | 0.449 |
| diabetes | J48 | tree@5 | 59.38% | 36.2 | 310.6 | 8.5 | 0.531 |
| diabetes | J48 | tree@8 | 56.63% | 43.0 | 438.7 | 10.1 | 0.646 |
| diabetes | JRip | kbins@3 | 76.30% | 1.9 | 5.9 | 2.9 | 0.470 |
| diabetes | JRip | kbins@5 | 75.26% | 3.1 | 9.3 | 2.9 | 0.555 |
| diabetes | JRip | kbins@8 | 75.52% | 3.5 | 11.2 | 3.1 | 0.684 |
| diabetes | JRip | raw | 75.65% | 2.6 | 6.4 | 2.4 | 0.409 |
| diabetes | JRip | tree@3 | 73.95% | 3.1 | 8.9 | 2.9 | 0.461 |
| diabetes | JRip | tree@5 | 71.88% | 4.0 | 13.5 | 3.3 | 0.561 |
| diabetes | JRip | tree@8 | 72.53% | 3.9 | 12.4 | 3.1 | 0.655 |
| ecoli | J48 | kbins@3 | 80.94% | 12.3 | 50.6 | 4.1 | 0.373 |
| ecoli | J48 | kbins@5 | 80.65% | 15.6 | 69.7 | 4.4 | 0.401 |
| ecoli | J48 | kbins@8 | 82.76% | 13.9 | 59.3 | 4.1 | 0.458 |
| ecoli | J48 | raw | 80.70% | 18.8 | 95.2 | 5.0 | 0.358 |
| ecoli | J48 | tree@3 | 80.10% | 11.7 | 46.6 | 3.9 | 0.352 |
| ecoli | J48 | tree@5 | 81.58% | 14.0 | 63.9 | 4.4 | 0.386 |
| ecoli | J48 | tree@8 | 80.12% | 15.6 | 75.3 | 4.8 | 0.447 |
| ecoli | JRip | kbins@3 | 79.47% | 5.5 | 14.1 | 2.6 | 0.393 |
| ecoli | JRip | kbins@5 | 80.93% | 7.7 | 20.1 | 2.6 | 0.433 |
| ecoli | JRip | kbins@8 | 82.48% | 6.0 | 14.4 | 2.4 | 0.467 |
| ecoli | JRip | raw | 81.56% | 8.1 | 17.0 | 2.1 | 0.400 |
| ecoli | JRip | tree@3 | 82.78% | 6.7 | 14.5 | 2.2 | 0.406 |
| ecoli | JRip | tree@5 | 81.60% | 7.5 | 16.4 | 2.2 | 0.427 |
| ecoli | JRip | tree@8 | 83.39% | 7.2 | 16.2 | 2.3 | 0.485 |
| glass | J48 | kbins@3 | 69.13% | 21.9 | 109.4 | 5.0 | 0.353 |
| glass | J48 | kbins@5 | 67.32% | 28.0 | 176.1 | 6.2 | 0.391 |
| glass | J48 | kbins@8 | 75.76% | 26.1 | 169.3 | 6.4 | 0.423 |
| glass | J48 | raw | 66.47% | 23.6 | 140.9 | 5.9 | 0.334 |
| glass | J48 | tree@3 | 67.77% | 14.4 | 71.0 | 4.9 | 0.349 |
| glass | J48 | tree@5 | 66.41% | 20.0 | 116.5 | 5.8 | 0.414 |
| glass | J48 | tree@8 | 67.81% | 22.7 | 141.2 | 6.2 | 0.455 |
| glass | JRip | kbins@3 | 61.23% | 5.5 | 18.0 | 3.3 | 0.375 |
| glass | JRip | kbins@5 | 69.68% | 7.4 | 21.9 | 2.9 | 0.396 |
| glass | JRip | kbins@8 | 63.03% | 8.3 | 22.1 | 2.7 | 0.435 |
| glass | JRip | raw | 66.36% | 6.7 | 15.1 | 2.2 | 0.358 |
| glass | JRip | tree@3 | 65.91% | 6.6 | 17.4 | 2.7 | 0.377 |
| glass | JRip | tree@5 | 65.97% | 6.4 | 15.4 | 2.4 | 0.429 |
| glass | JRip | tree@8 | 66.84% | 6.6 | 16.1 | 2.4 | 0.475 |
| ionosphere | J48 | kbins@3 | 87.44% | 19.2 | 94.3 | 4.9 | 0.460 |
| ionosphere | J48 | kbins@5 | 88.31% | 16.6 | 93.2 | 5.6 | 0.597 |
| ionosphere | J48 | kbins@8 | 88.30% | 16.4 | 86.4 | 5.2 | 0.796 |
| ionosphere | J48 | raw | 89.17% | 13.2 | 70.9 | 5.3 | 0.377 |
| ionosphere | J48 | tree@3 | 92.29% | 10.1 | 52.4 | 5.0 | 0.472 |
| ionosphere | J48 | tree@5 | 92.01% | 9.6 | 49.5 | 5.0 | 0.612 |
| ionosphere | J48 | tree@8 | 93.72% | 8.2 | 37.2 | 4.5 | 0.836 |
| ionosphere | JRip | kbins@3 | 87.74% | 6.2 | 14.1 | 2.2 | 0.492 |
| ionosphere | JRip | kbins@5 | 88.33% | 6.2 | 12.0 | 1.9 | 0.636 |
| ionosphere | JRip | kbins@8 | 85.75% | 5.8 | 11.4 | 1.9 | 0.824 |
| ionosphere | JRip | raw | 88.60% | 4.0 | 5.7 | 1.4 | 0.400 |
| ionosphere | JRip | tree@3 | 89.72% | 4.7 | 6.5 | 1.3 | 0.484 |
| ionosphere | JRip | tree@5 | 86.33% | 4.7 | 7.5 | 1.5 | 0.634 |
| ionosphere | JRip | tree@8 | 86.88% | 5.8 | 10.0 | 1.6 | 0.857 |
| iris | J48 | kbins@3 | 93.33% | 4.6 | 12.2 | 2.6 | 0.304 |
| iris | J48 | kbins@5 | 90.67% | 8.0 | 28.4 | 3.5 | 0.318 |
| iris | J48 | kbins@8 | 96.00% | 4.1 | 9.3 | 2.2 | 0.333 |
| iris | J48 | raw | 92.67% | 4.5 | 11.6 | 2.5 | 0.296 |
| iris | J48 | tree@3 | 92.00% | 3.0 | 5.0 | 1.7 | 0.299 |
| iris | J48 | tree@5 | 93.33% | 3.8 | 8.2 | 2.1 | 0.316 |
| iris | J48 | tree@8 | 94.00% | 4.7 | 12.9 | 2.6 | 0.339 |
| iris | JRip | kbins@3 | 92.00% | 2.6 | 4.2 | 1.6 | 0.303 |
| iris | JRip | kbins@5 | 92.00% | 4.0 | 8.3 | 2.1 | 0.320 |
| iris | JRip | kbins@8 | 95.33% | 2.7 | 3.9 | 1.4 | 0.339 |
| iris | JRip | raw | 92.67% | 2.6 | 4.3 | 1.7 | 0.294 |
| iris | JRip | tree@3 | 92.00% | 2.1 | 3.1 | 1.4 | 0.304 |
| iris | JRip | tree@5 | 91.33% | 2.7 | 3.2 | 1.1 | 0.320 |
| iris | JRip | tree@8 | 95.33% | 2.5 | 3.9 | 1.6 | 0.341 |
| kc1 | J48 | kbins@3 | 84.54% | 1.0 | 0.0 | 0.0 | 0.686 |
| kc1 | J48 | kbins@5 | 83.97% | 11.7 | 59.9 | 4.5 | 0.989 |
| kc1 | J48 | kbins@8 | 85.11% | 17.7 | 125.9 | 6.7 | 1.343 |
| kc1 | J48 | raw | 84.50% | 55.4 | 520.0 | 9.1 | 0.567 |
| kc1 | J48 | tree@3 | 85.44% | 18.7 | 129.7 | 6.7 | 0.815 |
| kc1 | J48 | tree@5 | 83.93% | 32.8 | 268.9 | 7.6 | 1.231 |
| kc1 | J48 | tree@8 | 84.88% | 51.1 | 553.6 | 10.7 | 1.848 |
| kc1 | JRip | kbins@3 | 84.54% | 0.0 | 0.0 | n/a | 0.670 |
| kc1 | JRip | kbins@5 | 84.16% | 1.2 | 4.1 | 3.3 | 1.018 |
| kc1 | JRip | kbins@8 | 84.92% | 1.9 | 6.8 | 3.5 | 1.474 |
| kc1 | JRip | raw | 84.73% | 2.3 | 6.9 | 2.8 | 0.655 |
| kc1 | JRip | tree@3 | 85.06% | 2.4 | 7.8 | 3.1 | 0.883 |
| kc1 | JRip | tree@5 | 84.83% | 2.9 | 10.5 | 3.4 | 1.361 |
| kc1 | JRip | tree@8 | 85.16% | 3.0 | 11.2 | 3.6 | 2.074 |
| kc2 | J48 | kbins@3 | 81.21% | 9.0 | 40.2 | 4.1 | 0.419 |
| kc2 | J48 | kbins@5 | 82.17% | 7.2 | 28.2 | 3.2 | 0.524 |
| kc2 | J48 | kbins@8 | 80.82% | 14.6 | 95.7 | 5.6 | 0.669 |
| kc2 | J48 | raw | 82.36% | 18.9 | 106.9 | 5.2 | 0.374 |
| kc2 | J48 | tree@3 | 83.69% | 11.3 | 57.0 | 4.6 | 0.473 |
| kc2 | J48 | tree@5 | 84.07% | 15.3 | 102.4 | 5.9 | 0.598 |
| kc2 | J48 | tree@8 | 84.08% | 20.3 | 155.9 | 7.2 | 0.783 |
| kc2 | JRip | kbins@3 | 81.98% | 1.1 | 4.1 | 3.8 | 0.422 |
| kc2 | JRip | kbins@5 | 82.55% | 1.3 | 2.4 | 1.8 | 0.527 |
| kc2 | JRip | kbins@8 | 82.55% | 1.6 | 4.0 | 2.3 | 0.672 |
| kc2 | JRip | raw | 82.35% | 1.6 | 3.1 | 1.8 | 0.392 |
| kc2 | JRip | tree@3 | 83.12% | 2.1 | 5.9 | 2.5 | 0.482 |
| kc2 | JRip | tree@5 | 82.16% | 1.9 | 4.0 | 1.8 | 0.618 |
| kc2 | JRip | tree@8 | 82.93% | 2.4 | 6.0 | 2.3 | 0.802 |
| mfeat-morphological | J48 | kbins@3 | 66.00% | 13.0 | 55.7 | 4.3 | 0.502 |
| mfeat-morphological | J48 | kbins@5 | 65.65% | 24.1 | 147.6 | 6.1 | 0.581 |
| mfeat-morphological | J48 | kbins@8 | 69.80% | 32.0 | 219.5 | 6.8 | 0.683 |
| mfeat-morphological | J48 | raw | 72.35% | 80.1 | 704.3 | 8.8 | 0.498 |
| mfeat-morphological | J48 | tree@3 | 67.35% | 16.8 | 81.7 | 4.8 | 0.611 |
| mfeat-morphological | J48 | tree@5 | 66.80% | 22.1 | 123.8 | 5.6 | 0.681 |
| mfeat-morphological | J48 | tree@8 | 69.50% | 28.0 | 177.6 | 6.3 | 0.772 |
| mfeat-morphological | JRip | kbins@3 | 56.50% | 9.1 | 22.8 | 2.5 | 0.567 |
| mfeat-morphological | JRip | kbins@5 | 56.70% | 10.6 | 28.9 | 2.7 | 0.687 |
| mfeat-morphological | JRip | kbins@8 | 68.15% | 16.4 | 44.4 | 2.7 | 0.852 |
| mfeat-morphological | JRip | raw | 69.55% | 20.4 | 48.7 | 2.4 | 0.689 |
| mfeat-morphological | JRip | tree@3 | 61.85% | 10.9 | 24.6 | 2.3 | 0.718 |
| mfeat-morphological | JRip | tree@5 | 57.90% | 12.0 | 31.7 | 2.6 | 0.831 |
| mfeat-morphological | JRip | tree@8 | 65.75% | 14.8 | 39.8 | 2.7 | 0.967 |
| pc1 | J48 | kbins@3 | 93.06% | 1.0 | 0.0 | 0.0 | 0.541 |
| pc1 | J48 | kbins@5 | 93.06% | 1.0 | 0.0 | 0.0 | 0.736 |
| pc1 | J48 | kbins@8 | 93.42% | 3.5 | 7.7 | 1.9 | 0.989 |
| pc1 | J48 | raw | 93.60% | 11.6 | 49.4 | 3.9 | 0.437 |
| pc1 | J48 | tree@3 | 92.79% | 4.8 | 12.8 | 2.5 | 0.581 |
| pc1 | J48 | tree@5 | 92.79% | 6.4 | 20.7 | 3.1 | 0.801 |
| pc1 | J48 | tree@8 | 93.06% | 7.3 | 26.1 | 3.3 | 1.131 |
| pc1 | JRip | kbins@3 | 93.06% | 0.0 | 0.0 | n/a | 0.542 |
| pc1 | JRip | kbins@5 | 93.06% | 0.0 | 0.0 | n/a | 0.758 |
| pc1 | JRip | kbins@8 | 92.79% | 0.4 | 2.0 | 5.0 | 1.024 |
| pc1 | JRip | raw | 93.60% | 2.2 | 5.9 | 2.7 | 0.492 |
| pc1 | JRip | tree@3 | 92.97% | 1.8 | 5.0 | 2.7 | 0.603 |
| pc1 | JRip | tree@5 | 93.15% | 2.4 | 6.8 | 2.9 | 0.846 |
| pc1 | JRip | tree@8 | 92.97% | 2.6 | 7.4 | 2.7 | 1.220 |
| pc3 | J48 | kbins@3 | 89.44% | 7.8 | 62.1 | 2.5 | 0.832 |
| pc3 | J48 | kbins@5 | 88.80% | 35.9 | 380.8 | 10.3 | 1.237 |
| pc3 | J48 | kbins@8 | 87.97% | 40.4 | 444.6 | 10.7 | 1.866 |
| pc3 | J48 | raw | 86.88% | 37.5 | 343.0 | 8.2 | 0.569 |
| pc3 | J48 | tree@3 | 88.55% | 24.8 | 193.0 | 7.3 | 0.967 |
| pc3 | J48 | tree@5 | 89.06% | 39.6 | 426.0 | 10.4 | 1.532 |
| pc3 | J48 | tree@8 | 88.03% | 53.5 | 698.0 | 13.1 | 2.263 |
| pc3 | JRip | kbins@3 | 89.12% | 0.5 | 2.6 | 5.2 | 0.848 |
| pc3 | JRip | kbins@5 | 88.61% | 1.8 | 8.0 | 4.4 | 1.290 |
| pc3 | JRip | kbins@8 | 89.12% | 2.5 | 11.5 | 4.5 | 2.042 |
| pc3 | JRip | raw | 89.44% | 1.9 | 7.6 | 4.1 | 0.634 |
| pc3 | JRip | tree@3 | 88.10% | 2.9 | 12.8 | 4.5 | 1.002 |
| pc3 | JRip | tree@5 | 88.29% | 3.3 | 13.7 | 4.1 | 1.592 |
| pc3 | JRip | tree@8 | 88.16% | 2.7 | 11.6 | 4.2 | 2.464 |
| pc4 | J48 | kbins@3 | 88.13% | 42.6 | 420.3 | 9.8 | 0.796 |
| pc4 | J48 | kbins@5 | 89.16% | 31.2 | 312.7 | 9.9 | 1.138 |
| pc4 | J48 | kbins@8 | 88.27% | 59.5 | 888.2 | 14.9 | 1.723 |
| pc4 | J48 | raw | 89.30% | 39.1 | 378.9 | 9.2 | 0.535 |
| pc4 | J48 | tree@3 | 88.68% | 34.0 | 348.4 | 10.0 | 0.947 |
| pc4 | J48 | tree@5 | 89.71% | 49.9 | 731.2 | 14.4 | 1.408 |
| pc4 | J48 | tree@8 | 87.31% | 57.0 | 1008.7 | 17.4 | 2.114 |
| pc4 | JRip | kbins@3 | 87.11% | 3.0 | 13.8 | 4.5 | 0.858 |
| pc4 | JRip | kbins@5 | 88.41% | 2.6 | 12.1 | 4.6 | 1.292 |
| pc4 | JRip | kbins@8 | 88.21% | 4.9 | 25.6 | 5.2 | 2.007 |
| pc4 | JRip | raw | 89.16% | 3.4 | 14.1 | 4.1 | 0.661 |
| pc4 | JRip | tree@3 | 89.09% | 3.5 | 14.8 | 4.1 | 1.011 |
| pc4 | JRip | tree@5 | 89.03% | 4.5 | 22.3 | 4.9 | 1.580 |
| pc4 | JRip | tree@8 | 88.48% | 4.2 | 19.7 | 4.6 | 2.471 |
| phoneme | J48 | kbins@3 | 81.38% | 25.1 | 129.1 | 5.1 | 0.774 |
| phoneme | J48 | kbins@5 | 84.20% | 88.5 | 741.5 | 8.3 | 1.160 |
| phoneme | J48 | kbins@8 | 86.07% | 155.8 | 1579.1 | 10.1 | 1.688 |
| phoneme | J48 | raw | 86.86% | 116.0 | 1046.1 | 9.0 | 0.694 |
| phoneme | J48 | tree@3 | 80.51% | 28.2 | 154.1 | 5.5 | 0.779 |
| phoneme | J48 | tree@5 | 82.49% | 60.3 | 438.2 | 7.2 | 1.087 |
| phoneme | J48 | tree@8 | 85.38% | 116.2 | 1057.5 | 9.0 | 1.594 |
| phoneme | JRip | kbins@3 | 81.46% | 7.9 | 33.2 | 4.2 | 0.972 |
| phoneme | JRip | kbins@5 | 83.33% | 14.2 | 73.9 | 5.2 | 1.685 |
| phoneme | JRip | kbins@8 | 84.71% | 20.5 | 107.9 | 5.2 | 2.720 |
| phoneme | JRip | raw | 85.27% | 17.7 | 79.3 | 4.4 | 1.282 |
| phoneme | JRip | tree@3 | 80.20% | 7.8 | 33.2 | 4.2 | 0.951 |
| phoneme | JRip | tree@5 | 82.20% | 12.1 | 59.4 | 4.9 | 1.512 |
| phoneme | JRip | tree@8 | 83.81% | 16.7 | 86.5 | 5.2 | 2.495 |
| qsar-biodeg | J48 | kbins@3 | 81.42% | 51.5 | 382.6 | 7.4 | 0.603 |
| qsar-biodeg | J48 | kbins@5 | 83.33% | 54.7 | 449.5 | 8.2 | 0.825 |
| qsar-biodeg | J48 | kbins@8 | 82.37% | 54.0 | 557.2 | 10.3 | 1.144 |
| qsar-biodeg | J48 | raw | 82.85% | 56.7 | 490.9 | 8.7 | 0.517 |
| qsar-biodeg | J48 | tree@3 | 82.56% | 48.1 | 474.5 | 9.8 | 0.966 |
| qsar-biodeg | J48 | tree@5 | 83.33% | 52.1 | 545.0 | 10.5 | 1.253 |
| qsar-biodeg | J48 | tree@8 | 84.18% | 45.6 | 452.6 | 9.8 | 1.597 |
| qsar-biodeg | JRip | kbins@3 | 81.23% | 6.7 | 27.3 | 4.1 | 0.666 |
| qsar-biodeg | JRip | kbins@5 | 83.41% | 6.4 | 25.7 | 4.0 | 0.896 |
| qsar-biodeg | JRip | kbins@8 | 82.75% | 7.8 | 32.9 | 4.2 | 1.343 |
| qsar-biodeg | JRip | raw | 83.32% | 5.8 | 20.0 | 3.4 | 0.653 |
| qsar-biodeg | JRip | tree@3 | 82.85% | 6.1 | 24.5 | 4.0 | 1.075 |
| qsar-biodeg | JRip | tree@5 | 83.98% | 6.8 | 26.7 | 3.9 | 1.456 |
| qsar-biodeg | JRip | tree@8 | 81.62% | 7.0 | 27.3 | 3.9 | 1.938 |
| segment | J48 | kbins@3 | 91.90% | 54.5 | 372.3 | 6.8 | 0.788 |
| segment | J48 | kbins@5 | 93.55% | 60.7 | 494.3 | 8.1 | 1.083 |
| segment | J48 | kbins@8 | 94.55% | 53.8 | 512.5 | 9.5 | 1.550 |
| segment | J48 | raw | 96.58% | 40.9 | 319.0 | 7.8 | 0.571 |
| segment | J48 | tree@3 | 90.48% | 36.5 | 310.0 | 8.4 | 0.835 |
| segment | J48 | tree@5 | 94.89% | 34.7 | 284.0 | 8.1 | 1.124 |
| segment | J48 | tree@8 | 95.37% | 41.3 | 389.4 | 9.4 | 1.577 |
| segment | JRip | kbins@3 | 90.82% | 22.8 | 80.7 | 3.5 | 1.006 |
| segment | JRip | kbins@5 | 91.95% | 24.0 | 80.2 | 3.3 | 1.517 |
| segment | JRip | kbins@8 | 94.42% | 22.6 | 65.1 | 2.9 | 2.072 |
| segment | JRip | raw | 95.06% | 18.1 | 52.1 | 2.9 | 0.874 |
| segment | JRip | tree@3 | 89.65% | 15.6 | 43.0 | 2.7 | 1.029 |
| segment | JRip | tree@5 | 94.29% | 19.3 | 55.8 | 2.9 | 1.479 |
| segment | JRip | tree@8 | 94.98% | 20.2 | 58.8 | 2.9 | 2.110 |
| sonar | J48 | kbins@3 | 79.33% | 18.1 | 89.1 | 4.9 | 0.502 |
| sonar | J48 | kbins@5 | 75.57% | 18.9 | 107.0 | 5.6 | 0.705 |
| sonar | J48 | kbins@8 | 72.14% | 18.0 | 94.6 | 5.2 | 1.005 |
| sonar | J48 | raw | 76.88% | 14.7 | 68.6 | 4.7 | 0.375 |
| sonar | J48 | tree@3 | 70.67% | 16.7 | 93.6 | 5.5 | 0.516 |
| sonar | J48 | tree@5 | 69.76% | 19.0 | 132.0 | 6.8 | 0.729 |
| sonar | J48 | tree@8 | 70.14% | 18.1 | 119.8 | 6.5 | 1.023 |
| sonar | JRip | kbins@3 | 70.19% | 4.1 | 9.9 | 2.4 | 0.527 |
| sonar | JRip | kbins@5 | 75.57% | 4.3 | 9.3 | 2.1 | 0.734 |
| sonar | JRip | kbins@8 | 72.52% | 4.2 | 9.2 | 2.1 | 1.034 |
| sonar | JRip | raw | 73.62% | 3.2 | 6.7 | 2.1 | 0.409 |
| sonar | JRip | tree@3 | 73.64% | 4.8 | 11.7 | 2.5 | 0.527 |
| sonar | JRip | tree@5 | 73.07% | 4.7 | 10.5 | 2.2 | 0.731 |
| sonar | JRip | tree@8 | 72.60% | 4.7 | 10.1 | 2.1 | 1.034 |
| steel-plates-fault | J48 | kbins@3 | 66.36% | 206.4 | 1990.2 | 9.6 | 1.027 |
| steel-plates-fault | J48 | kbins@5 | 70.79% | 198.2 | 2219.1 | 11.2 | 1.430 |
| steel-plates-fault | J48 | kbins@8 | 72.13% | 193.4 | 2332.1 | 12.0 | 2.016 |
| steel-plates-fault | J48 | raw | 75.02% | 162.0 | 1923.9 | 11.9 | 0.675 |
| steel-plates-fault | J48 | tree@3 | 71.15% | 127.8 | 1360.4 | 10.6 | 1.008 |
| steel-plates-fault | J48 | tree@5 | 73.31% | 156.4 | 1979.8 | 12.6 | 1.452 |
| steel-plates-fault | J48 | tree@8 | 70.58% | 178.9 | 2551.8 | 14.2 | 2.139 |
| steel-plates-fault | JRip | kbins@3 | 67.23% | 26.2 | 126.4 | 4.8 | 1.330 |
| steel-plates-fault | JRip | kbins@5 | 69.04% | 23.3 | 101.7 | 4.4 | 2.105 |
| steel-plates-fault | JRip | kbins@8 | 71.46% | 24.1 | 97.1 | 4.0 | 3.135 |
| steel-plates-fault | JRip | raw | 72.02% | 21.3 | 70.4 | 3.3 | 1.118 |
| steel-plates-fault | JRip | tree@3 | 70.17% | 22.4 | 94.5 | 4.2 | 1.431 |
| steel-plates-fault | JRip | tree@5 | 73.93% | 23.4 | 91.9 | 3.9 | 2.164 |
| steel-plates-fault | JRip | tree@8 | 73.31% | 24.5 | 95.5 | 3.9 | 3.431 |
| vehicle | J48 | kbins@3 | 67.39% | 61.6 | 489.9 | 7.9 | 0.547 |
| vehicle | J48 | kbins@5 | 68.46% | 87.6 | 861.8 | 9.8 | 0.721 |
| vehicle | J48 | kbins@8 | 70.10% | 91.7 | 964.3 | 10.5 | 0.954 |
| vehicle | J48 | raw | 73.07% | 68.4 | 549.2 | 8.0 | 0.426 |
| vehicle | J48 | tree@3 | 69.04% | 48.7 | 344.6 | 7.0 | 0.537 |
| vehicle | J48 | tree@5 | 67.26% | 66.4 | 612.2 | 9.1 | 0.715 |
| vehicle | J48 | tree@8 | 68.44% | 83.6 | 933.4 | 11.1 | 0.962 |
| vehicle | JRip | kbins@3 | 64.67% | 9.5 | 25.8 | 2.7 | 0.599 |
| vehicle | JRip | kbins@5 | 64.54% | 12.6 | 34.7 | 2.7 | 0.822 |
| vehicle | JRip | kbins@8 | 68.09% | 14.5 | 40.5 | 2.8 | 1.129 |
| vehicle | JRip | raw | 69.99% | 15.0 | 38.4 | 2.6 | 0.540 |
| vehicle | JRip | tree@3 | 62.77% | 10.3 | 30.8 | 3.0 | 0.610 |
| vehicle | JRip | tree@5 | 68.57% | 12.6 | 34.0 | 2.7 | 0.802 |
| vehicle | JRip | tree@8 | 68.45% | 13.9 | 38.4 | 2.8 | 1.120 |
| wdbc | J48 | kbins@3 | 95.43% | 9.2 | 34.1 | 3.7 | 0.511 |
| wdbc | J48 | kbins@5 | 93.85% | 13.3 | 57.3 | 4.3 | 0.704 |
| wdbc | J48 | kbins@8 | 94.02% | 13.5 | 58.6 | 4.3 | 0.982 |
| wdbc | J48 | raw | 93.86% | 11.3 | 45.9 | 4.0 | 0.398 |
| wdbc | J48 | tree@3 | 95.60% | 11.8 | 50.2 | 4.2 | 0.518 |
| wdbc | J48 | tree@5 | 95.44% | 11.6 | 52.0 | 4.4 | 0.701 |
| wdbc | J48 | tree@8 | 95.44% | 10.9 | 45.4 | 4.1 | 0.979 |
| wdbc | JRip | kbins@3 | 94.02% | 4.4 | 10.3 | 2.3 | 0.542 |
| wdbc | JRip | kbins@5 | 93.33% | 5.3 | 13.0 | 2.4 | 0.743 |
| wdbc | JRip | kbins@8 | 95.26% | 5.1 | 11.9 | 2.4 | 1.032 |
| wdbc | JRip | raw | 94.91% | 3.6 | 7.3 | 2.0 | 0.444 |
| wdbc | JRip | tree@3 | 95.25% | 4.6 | 10.5 | 2.3 | 0.541 |
| wdbc | JRip | tree@5 | 93.85% | 4.8 | 10.8 | 2.3 | 0.726 |
| wdbc | JRip | tree@8 | 94.55% | 4.8 | 10.7 | 2.2 | 1.025 |
| wilt | J48 | kbins@3 | 96.09% | 10.9 | 58.0 | 5.2 | 0.685 |
| wilt | J48 | kbins@5 | 97.66% | 12.7 | 58.9 | 4.5 | 0.892 |
| wilt | J48 | kbins@8 | 98.31% | 23.1 | 146.0 | 6.3 | 1.214 |
| wilt | J48 | raw | 98.10% | 27.1 | 162.5 | 5.9 | 0.593 |
| wilt | J48 | tree@3 | 96.94% | 3.4 | 6.6 | 1.9 | 0.664 |
| wilt | J48 | tree@5 | 97.00% | 7.1 | 28.5 | 3.8 | 0.873 |
| wilt | J48 | tree@8 | 97.40% | 7.4 | 28.9 | 3.7 | 1.163 |
| wilt | JRip | kbins@3 | 95.89% | 3.3 | 12.6 | 3.7 | 0.766 |
| wilt | JRip | kbins@5 | 97.54% | 3.8 | 12.0 | 3.1 | 1.001 |
| wilt | JRip | kbins@8 | 98.02% | 5.1 | 16.1 | 3.1 | 1.374 |
| wilt | JRip | raw | 98.22% | 7.0 | 17.7 | 2.5 | 0.742 |
| wilt | JRip | tree@3 | 97.00% | 1.1 | 2.3 | 2.0 | 0.690 |
| wilt | JRip | tree@5 | 97.11% | 1.9 | 6.0 | 3.0 | 0.938 |
| wilt | JRip | tree@8 | 97.35% | 2.2 | 6.1 | 2.8 | 1.258 |
| wine | J48 | kbins@3 | 94.41% | 7.1 | 22.5 | 3.1 | 0.338 |
| wine | J48 | kbins@5 | 93.27% | 6.0 | 16.6 | 2.7 | 0.377 |
| wine | J48 | kbins@8 | 90.42% | 7.0 | 23.5 | 3.2 | 0.442 |
| wine | J48 | raw | 93.14% | 5.1 | 12.5 | 2.4 | 0.306 |
| wine | J48 | tree@3 | 89.25% | 5.8 | 15.7 | 2.6 | 0.339 |
| wine | J48 | tree@5 | 92.12% | 6.3 | 19.1 | 2.9 | 0.383 |
| wine | J48 | tree@8 | 91.57% | 6.5 | 19.9 | 3.0 | 0.443 |
| wine | JRip | kbins@3 | 90.95% | 3.1 | 6.8 | 2.2 | 0.345 |
| wine | JRip | kbins@5 | 89.25% | 3.4 | 6.1 | 1.8 | 0.386 |
| wine | JRip | kbins@8 | 90.42% | 3.0 | 5.7 | 1.9 | 0.452 |
| wine | JRip | raw | 95.49% | 2.9 | 5.4 | 1.9 | 0.319 |
| wine | JRip | tree@3 | 90.46% | 2.9 | 5.7 | 1.9 | 0.340 |
| wine | JRip | tree@5 | 92.22% | 2.8 | 5.3 | 1.9 | 0.382 |
| wine | JRip | tree@8 | 94.41% | 2.7 | 5.1 | 1.9 | 0.452 |
| yeast | J48 | kbins@3 | 49.40% | 48.2 | 314.7 | 6.5 | 0.501 |
| yeast | J48 | kbins@5 | 57.27% | 110.7 | 1014.4 | 9.1 | 0.664 |
| yeast | J48 | kbins@8 | 53.30% | 159.2 | 1785.8 | 11.2 | 0.826 |
| yeast | J48 | raw | 56.46% | 167.5 | 1734.7 | 10.3 | 0.497 |
| yeast | J48 | tree@3 | 58.49% | 31.6 | 195.9 | 6.1 | 0.530 |
| yeast | J48 | tree@5 | 57.82% | 63.8 | 559.2 | 8.7 | 0.662 |
| yeast | J48 | tree@8 | 55.52% | 106.6 | 1233.8 | 11.5 | 0.838 |
| yeast | JRip | kbins@3 | 44.81% | 9.3 | 39.0 | 4.2 | 0.594 |
| yeast | JRip | kbins@5 | 56.06% | 13.4 | 47.7 | 3.6 | 0.740 |
| yeast | JRip | kbins@8 | 55.93% | 13.3 | 44.9 | 3.4 | 0.927 |
| yeast | JRip | raw | 57.28% | 15.2 | 38.4 | 2.5 | 0.607 |
| yeast | JRip | tree@3 | 59.43% | 12.1 | 29.3 | 2.4 | 0.617 |
| yeast | JRip | tree@5 | 59.44% | 14.6 | 38.0 | 2.6 | 0.779 |
| yeast | JRip | tree@8 | 60.31% | 14.0 | 36.9 | 2.6 | 1.003 |

