# RIPPER-family comparison: Weka:JRip / Witt:RIPPER / Pypper / Slipper

## Setup

**Full run** (44 binary datasets). For a quick sanity check instead (10 small datasets, 3-fold, well under a minute): `python demos/ripper_comparison.py` with no arguments -- its output isn't checked in (see `ripper_comparison_quick_report.md` after running it).

Four RIPPER-family rule learners -- Weka:JRip, Witt:RIPPER,
Pypper, Slipper -- compared on 44 binary datasets from
`pyrulearn.experiments.catalog`.

A fifth, IMod:Slipper (`imodels`' SlipperClassifier), is checked
separately first against the native Slipper on a handful of small,
low-feature-count datasets, then left out of the main comparison below --
see "Why IMod:Slipper isn't in the main comparison". Across 4 datasets (3-fold), IMod:Slipper was no more accurate than Slipper (mean accuracy 68.95% vs. 80.38%) while taking 102x as long per fit (7.47s vs. 0.07s, mean) -- not worth its cost at the scale of the main comparison below.

Protocol: `10`-fold stratified cross-validation
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals=6)`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical
binary features per fold (as N-lists, the library's default representation;
the Weka learners convert them to the matrix they need), so differences
reflect the algorithms, not the data preparation. Each fit is capped at 180s; a
timeout or an exception is recorded as a failure, not fatal to the run.

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, rule count, total condition count, fit time
(seconds, JVM startup included for Weka:JRip). Witt:RIPPER auto-dispatches
one-vs-rest for multi-class (see the module docstring); every other
learner handles multi-class natively.

Every learner runs single-threaded; fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

![accuracy vs. rule-set complexity](ripper_comparison_plots/ripper_comparison_accuracy.png)

![fit time per dataset](ripper_comparison_plots/ripper_comparison_fit_time.png)

![critical-difference diagram (accuracy)](ripper_comparison_plots/ripper_comparison_cd.png)

## Why IMod:Slipper isn't in the main comparison

Across 4 datasets (3-fold), IMod:Slipper was no more accurate than Slipper (mean accuracy 68.95% vs. 80.38%) while taking 102x as long per fit (7.47s vs. 0.07s, mean) -- not worth its cost at the scale of the main comparison below.

| learner | accuracy | fit_time |
|---|---|---|
| IMod:Slipper | 68.95% | 7.473 |
| Slipper | 80.38% | 0.073 |

## Summary by learner (mean across every dataset and fold)

| learner | accuracy | n_rules | conds/rule | fit_time (s) | wins | mean rank |
|---|--:|--:|--:|--:|--:|--:|
| Slipper | 85.56% | 13.6 | 2.1 | 0.11 | 17.2 | 2.09 |
| Weka:JRip | 84.92% | 5.2 | 3.0 | 1.86 | 14.2 | 2.15 |
| Pypper | 84.98% | 3.9 | 3.0 | 0.24 | 8.2 | 2.42 |
| Witt:RIPPER | 79.60% | 6.4 | 3.1 | 1.71 | 4.2 | 3.34 |

`wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly (`pyrulearn.experiments.stats.win_counts`); `mean rank` -- average accuracy rank across datasets, failures tied for last (`stats.mean_rank`).

## Per-dataset results, per learner (mean across folds)

| dataset | learner | accuracy | n_rules | n_conditions | conds_per_rule | fit_time |
|---|---|---|---|---|---|---|
| SPECT | Pypper | 79.00% | 0.9 | 5.0 | 5.6 | 0.029 |
| SPECT | Slipper | 83.13% | 11.2 | 31.4 | 2.8 | 0.084 |
| SPECT | Weka:JRip | 80.87% | 1.0 | 5.5 | 5.5 | 0.436 |
| SPECT | Witt:RIPPER | 78.68% | 1.6 | 12.0 | 7.5 | 0.128 |
| Titanic | Pypper | 79.22% | 2.2 | 5.1 | 2.3 | 0.089 |
| Titanic | Slipper | 79.60% | 13.1 | 23.8 | 1.8 | 0.070 |
| Titanic | Weka:JRip | 78.92% | 3.2 | 7.1 | 2.2 | 0.659 |
| Titanic | Witt:RIPPER | 75.25% | 6.0 | 22.8 | 3.3 | 0.487 |
| banknote-authentication | Pypper | 98.40% | 4.6 | 10.9 | 2.4 | 0.085 |
| banknote-authentication | Slipper | 98.47% | 12.9 | 26.1 | 2.0 | 0.074 |
| banknote-authentication | Weka:JRip | 98.69% | 5.2 | 12.3 | 2.4 | 0.551 |
| banknote-authentication | Witt:RIPPER | 98.76% | 5.8 | 14.4 | 2.4 | 0.169 |
| blood-transfusion-service-center | Pypper | 79.54% | 1.9 | 5.1 | 2.7 | 0.047 |
| blood-transfusion-service-center | Slipper | 78.60% | 11.6 | 20.1 | 1.7 | 0.070 |
| blood-transfusion-service-center | Weka:JRip | 79.00% | 1.9 | 4.8 | 2.5 | 0.452 |
| blood-transfusion-service-center | Witt:RIPPER | 64.57% | 3.9 | 9.2 | 2.4 | 0.154 |
| breast-cancer | Pypper | 74.84% | 1.0 | 2.0 | 2.0 | 0.031 |
| breast-cancer | Slipper | 72.03% | 13.9 | 22.3 | 1.6 | 0.069 |
| breast-cancer | Weka:JRip | 71.37% | 1.3 | 2.5 | 1.9 | 0.673 |
| breast-cancer | Witt:RIPPER | 53.72% | 1.5 | 3.7 | 2.7 | 0.193 |
| breast-w | Pypper | 93.99% | 4.2 | 12.0 | 2.9 | 0.101 |
| breast-w | Slipper | 94.28% | 14.7 | 20.0 | 1.4 | 0.069 |
| breast-w | Weka:JRip | 94.84% | 5.1 | 12.9 | 2.5 | 1.036 |
| breast-w | Witt:RIPPER | 94.28% | 4.6 | 8.6 | 1.9 | 0.416 |
| churn | Pypper | 91.78% | 6.9 | 20.4 | 3.0 | 0.299 |
| churn | Slipper | 91.88% | 15.9 | 36.2 | 2.3 | 0.198 |
| churn | Weka:JRip | 92.40% | 10.1 | 30.1 | 3.0 | 3.979 |
| churn | Witt:RIPPER | 89.50% | 12.7 | 70.8 | 5.6 | 6.514 |
| climate-model-simulation-crashes | Pypper | 94.44% | 1.8 | 5.9 | 3.3 | 0.039 |
| climate-model-simulation-crashes | Slipper | 93.33% | 13.9 | 33.3 | 2.4 | 0.065 |
| climate-model-simulation-crashes | Weka:JRip | 94.81% | 2.7 | 8.1 | 3.0 | 0.714 |
| climate-model-simulation-crashes | Witt:RIPPER | 93.70% | 3.5 | 11.2 | 3.2 | 0.361 |
| colic | Pypper | 85.62% | 1.8 | 4.2 | 2.3 | 0.037 |
| colic | Slipper | 81.81% | 13.9 | 21.0 | 1.5 | 0.050 |
| colic | Weka:JRip | 84.52% | 3.9 | 9.2 | 2.3 | 0.608 |
| colic | Witt:RIPPER | 80.98% | 3.9 | 11.7 | 3.0 | 0.331 |
| compas-two-years | Pypper | 67.58% | 3.9 | 12.1 | 3.1 | 0.817 |
| compas-two-years | Slipper | 67.90% | 13.5 | 28.1 | 2.1 | 0.193 |
| compas-two-years | Weka:JRip | 67.68% | 4.9 | 13.0 | 2.6 | 2.345 |
| compas-two-years | Witt:RIPPER | 64.82% | 4.9 | 19.5 | 3.5 | 2.773 |
| credit-approval | Pypper | 83.62% | 3.2 | 8.0 | 2.2 | 0.098 |
| credit-approval | Slipper | 84.78% | 13.3 | 24.2 | 1.8 | 0.072 |
| credit-approval | Weka:JRip | 85.22% | 4.0 | 8.7 | 2.1 | 0.793 |
| credit-approval | Witt:RIPPER | 84.78% | 6.3 | 17.9 | 2.8 | 0.408 |
| credit-g | Pypper | 71.70% | 2.8 | 11.3 | 3.9 | 0.113 |
| credit-g | Slipper | 73.60% | 14.8 | 29.8 | 2.0 | 0.101 |
| credit-g | Weka:JRip | 73.10% | 4.0 | 15.3 | 3.7 | 0.973 |
| credit-g | Witt:RIPPER | 71.60% | 2.7 | 12.4 | 4.6 | 0.660 |
| cylinder-bands | Pypper | 67.59% | 4.2 | 13.3 | 3.2 | 0.163 |
| cylinder-bands | Slipper | 70.56% | 17.4 | 20.1 | 1.2 | 0.072 |
| cylinder-bands | Weka:JRip | 63.89% | 6.6 | 17.6 | 2.6 | 1.452 |
| cylinder-bands | Witt:RIPPER | 63.52% | 5.1 | 12.4 | 2.4 | 1.111 |
| diabetes | Pypper | 74.49% | 3.7 | 12.5 | 3.3 | 0.141 |
| diabetes | Slipper | 75.78% | 13.3 | 27.7 | 2.1 | 0.081 |
| diabetes | Weka:JRip | 73.17% | 3.8 | 12.5 | 3.2 | 0.732 |
| diabetes | Witt:RIPPER | 66.66% | 5.8 | 19.8 | 3.4 | 0.442 |
| dresses-sales | Pypper | 61.20% | 1.1 | 1.2 | 1.1 | 0.043 |
| dresses-sales | Slipper | 58.80% | 16.2 | 19.2 | 1.2 | 0.052 |
| dresses-sales | Weka:JRip | 59.20% | 2.1 | 3.7 | 1.7 | 0.932 |
| dresses-sales | Witt:RIPPER | 45.80% | 1.3 | 3.8 | 3.2 | 0.586 |
| heart-c | Pypper | 80.56% | 2.6 | 6.2 | 2.4 | 0.062 |
| heart-c | Slipper | 79.23% | 13.2 | 20.9 | 1.6 | 0.075 |
| heart-c | Weka:JRip | 77.60% | 3.6 | 9.1 | 2.4 | 0.629 |
| heart-c | Witt:RIPPER | 81.16% | 4.2 | 11.7 | 2.8 | 0.233 |
| heart-h | Pypper | 78.93% | 2.8 | 7.0 | 2.4 | 0.065 |
| heart-h | Slipper | 79.91% | 14.2 | 21.2 | 1.5 | 0.054 |
| heart-h | Weka:JRip | 77.86% | 2.4 | 5.5 | 2.1 | 0.544 |
| heart-h | Witt:RIPPER | 50.01% | 3.4 | 8.4 | 2.5 | 0.202 |
| heart-statlog | Pypper | 79.63% | 2.4 | 5.9 | 2.4 | 0.046 |
| heart-statlog | Slipper | 81.48% | 13.4 | 19.3 | 1.4 | 0.057 |
| heart-statlog | Weka:JRip | 82.22% | 3.4 | 7.5 | 2.2 | 0.548 |
| heart-statlog | Witt:RIPPER | 80.74% | 3.6 | 9.6 | 2.6 | 0.189 |
| heloc | Pypper | 70.44% | 7.2 | 25.7 | 3.5 | 3.451 |
| heloc | Slipper | 70.69% | 14.5 | 38.1 | 2.6 | 0.463 |
| heloc | Weka:JRip | 70.91% | 8.1 | 30.9 | 3.7 | 11.649 |
| heloc | Witt:RIPPER | 67.62% | 15.4 | 100.0 | 6.5 | 18.455 |
| hepatitis | Pypper | 78.12% | 1.7 | 2.7 | 1.5 | 0.020 |
| hepatitis | Slipper | 84.00% | 13.1 | 20.6 | 1.6 | 0.043 |
| hepatitis | Weka:JRip | 76.29% | 2.1 | 4.6 | 2.1 | 0.462 |
| hepatitis | Witt:RIPPER | 78.92% | 2.6 | 5.8 | 2.1 | 0.133 |
| ilpd | Pypper | 66.91% | 1.7 | 5.9 | 3.4 | 0.058 |
| ilpd | Slipper | 71.69% | 12.5 | 24.5 | 2.0 | 0.067 |
| ilpd | Weka:JRip | 70.50% | 1.6 | 5.2 | 3.3 | 0.573 |
| ilpd | Witt:RIPPER | 55.57% | 2.7 | 7.5 | 2.8 | 0.233 |
| ionosphere | Pypper | 89.75% | 3.8 | 5.8 | 1.5 | 0.063 |
| ionosphere | Slipper | 91.15% | 14.4 | 16.7 | 1.2 | 0.055 |
| ionosphere | Weka:JRip | 86.89% | 5.8 | 10.0 | 1.6 | 0.949 |
| ionosphere | Witt:RIPPER | 90.89% | 5.7 | 9.6 | 1.7 | 0.575 |
| kc1 | Pypper | 84.64% | 2.1 | 7.6 | 3.5 | 0.100 |
| kc1 | Slipper | 84.83% | 13.3 | 25.9 | 1.9 | 0.109 |
| kc1 | Weka:JRip | 84.83% | 3.0 | 11.5 | 3.5 | 1.896 |
| kc1 | Witt:RIPPER | 55.48% | 3.0 | 10.9 | 3.5 | 1.236 |
| kc2 | Pypper | 84.27% | 1.6 | 4.0 | 2.3 | 0.038 |
| kc2 | Slipper | 83.51% | 13.3 | 19.2 | 1.4 | 0.063 |
| kc2 | Weka:JRip | 83.89% | 2.3 | 5.8 | 2.4 | 0.873 |
| kc2 | Witt:RIPPER | 76.02% | 2.7 | 6.1 | 2.3 | 0.409 |
| kr-vs-kp | Pypper | 99.09% | 13.6 | 44.0 | 3.2 | 0.430 |
| kr-vs-kp | Slipper | 97.69% | 14.6 | 39.7 | 2.7 | 0.112 |
| kr-vs-kp | Weka:JRip | 99.31% | 14.6 | 46.2 | 3.2 | 1.444 |
| kr-vs-kp | Witt:RIPPER | 99.28% | 17.5 | 57.8 | 3.3 | 0.956 |
| mofn-3-7-10 | Pypper | 94.71% | 14.0 | 67.0 | 4.7 | 0.467 |
| mofn-3-7-10 | Slipper | 97.89% | 14.1 | 41.3 | 2.9 | 0.095 |
| mofn-3-7-10 | Weka:JRip | 96.97% | 15.6 | 75.2 | 4.8 | 0.711 |
| mofn-3-7-10 | Witt:RIPPER | 98.64% | 19.3 | 95.6 | 4.9 | 0.557 |
| molecular-biology_promoters | Pypper | 83.18% | 2.1 | 3.8 | 1.8 | 0.028 |
| molecular-biology_promoters | Slipper | 86.64% | 12.5 | 16.3 | 1.3 | 0.044 |
| molecular-biology_promoters | Weka:JRip | 77.45% | 2.5 | 5.4 | 2.1 | 0.697 |
| molecular-biology_promoters | Witt:RIPPER | 75.45% | 2.3 | 4.6 | 2.0 | 0.505 |
| monks-problems-1 | Pypper | 95.30% | 3.6 | 10.2 | 2.7 | 0.073 |
| monks-problems-1 | Slipper | 100.00% | 12.8 | 31.6 | 2.4 | 0.078 |
| monks-problems-1 | Weka:JRip | 93.04% | 3.7 | 10.4 | 2.5 | 0.534 |
| monks-problems-1 | Witt:RIPPER | 94.57% | 4.6 | 16.2 | 3.4 | 0.128 |
| monks-problems-2 | Pypper | 70.39% | 2.3 | 13.3 | 5.9 | 0.063 |
| monks-problems-2 | Slipper | 71.73% | 12.9 | 39.8 | 3.1 | 0.080 |
| monks-problems-2 | Weka:JRip | 80.57% | 7.6 | 44.6 | 5.8 | 0.523 |
| monks-problems-2 | Witt:RIPPER | 82.85% | 14.2 | 50.7 | 3.5 | 0.453 |
| monks-problems-3 | Pypper | 98.92% | 3.0 | 5.0 | 1.7 | 0.036 |
| monks-problems-3 | Slipper | 98.92% | 10.4 | 21.5 | 2.1 | 0.057 |
| monks-problems-3 | Weka:JRip | 98.92% | 3.0 | 5.0 | 1.7 | 0.488 |
| monks-problems-3 | Witt:RIPPER | 98.92% | 3.0 | 5.0 | 1.7 | 0.084 |
| mushroom | Pypper | 99.96% | 5.2 | 10.6 | 2.1 | 0.225 |
| mushroom | Slipper | 100.00% | 10.5 | 20.6 | 2.0 | 0.229 |
| mushroom | Weka:JRip | 99.98% | 6.0 | 10.1 | 1.7 | 5.455 |
| mushroom | Witt:RIPPER | 98.71% | 6.9 | 13.9 | 2.0 | 1.993 |
| ozone-level-8hr | Pypper | 92.90% | 2.9 | 12.1 | 4.3 | 0.201 |
| ozone-level-8hr | Slipper | 93.73% | 13.6 | 39.2 | 2.9 | 0.294 |
| ozone-level-8hr | Weka:JRip | 92.58% | 5.8 | 25.9 | 4.4 | 7.131 |
| ozone-level-8hr | Witt:RIPPER | 81.69% | 5.9 | 16.0 | 2.7 | 4.648 |
| pc1 | Pypper | 92.61% | 1.5 | 4.1 | 2.6 | 0.038 |
| pc1 | Slipper | 92.61% | 13.6 | 26.7 | 2.0 | 0.080 |
| pc1 | Weka:JRip | 93.15% | 1.9 | 5.4 | 2.8 | 1.278 |
| pc1 | Witt:RIPPER | 71.78% | 1.8 | 4.8 | 2.7 | 0.478 |
| pc3 | Pypper | 88.99% | 1.7 | 6.9 | 4.1 | 0.071 |
| pc3 | Slipper | 88.87% | 12.8 | 28.6 | 2.2 | 0.135 |
| pc3 | Weka:JRip | 87.91% | 3.0 | 12.7 | 4.2 | 2.399 |
| pc3 | Witt:RIPPER | 66.93% | 3.9 | 12.8 | 3.2 | 1.523 |
| pc4 | Pypper | 89.23% | 2.9 | 12.9 | 4.3 | 0.111 |
| pc4 | Slipper | 89.58% | 12.6 | 38.5 | 3.1 | 0.152 |
| pc4 | Weka:JRip | 88.27% | 5.2 | 26.0 | 4.9 | 2.338 |
| pc4 | Witt:RIPPER | 82.92% | 5.2 | 12.0 | 2.3 | 1.234 |
| phoneme | Pypper | 81.72% | 10.0 | 53.5 | 5.2 | 0.911 |
| phoneme | Slipper | 81.61% | 14.3 | 49.0 | 3.4 | 0.203 |
| phoneme | Weka:JRip | 82.62% | 12.9 | 68.7 | 5.3 | 2.226 |
| phoneme | Witt:RIPPER | 74.46% | 15.8 | 81.0 | 5.2 | 3.943 |
| qsar-biodeg | Pypper | 84.08% | 5.1 | 19.4 | 3.9 | 0.199 |
| qsar-biodeg | Slipper | 83.43% | 14.1 | 37.1 | 2.6 | 0.112 |
| qsar-biodeg | Weka:JRip | 82.94% | 7.5 | 30.2 | 4.0 | 1.935 |
| qsar-biodeg | Witt:RIPPER | 80.48% | 10.3 | 30.9 | 3.0 | 1.380 |
| sick | Pypper | 98.09% | 3.4 | 11.1 | 3.2 | 0.121 |
| sick | Slipper | 97.85% | 13.8 | 40.5 | 2.9 | 0.163 |
| sick | Weka:JRip | 98.28% | 5.3 | 18.4 | 3.4 | 1.904 |
| sick | Witt:RIPPER | 82.03% | 7.0 | 18.2 | 2.6 | 0.901 |
| sonar | Pypper | 77.88% | 3.5 | 7.0 | 2.0 | 0.058 |
| sonar | Slipper | 74.93% | 15.2 | 18.8 | 1.2 | 0.052 |
| sonar | Weka:JRip | 76.98% | 4.5 | 9.4 | 2.1 | 0.979 |
| sonar | Witt:RIPPER | 77.52% | 3.2 | 7.2 | 2.3 | 0.773 |
| spambase | Pypper | 92.70% | 12.0 | 55.9 | 4.6 | 1.274 |
| spambase | Slipper | 92.50% | 15.1 | 50.8 | 3.4 | 0.285 |
| spambase | Weka:JRip | 93.22% | 17.4 | 78.6 | 4.5 | 13.668 |
| spambase | Witt:RIPPER | 92.59% | 24.1 | 115.9 | 4.8 | 16.744 |
| tic-tac-toe | Pypper | 96.25% | 7.5 | 23.1 | 3.1 | 0.138 |
| tic-tac-toe | Slipper | 98.33% | 15.0 | 43.9 | 2.9 | 0.079 |
| tic-tac-toe | Weka:JRip | 98.12% | 8.4 | 26.6 | 3.1 | 0.725 |
| tic-tac-toe | Witt:RIPPER | 98.02% | 8.4 | 25.7 | 3.1 | 0.262 |
| vote | Pypper | 95.18% | 1.7 | 3.3 | 1.6 | 0.041 |
| vote | Slipper | 94.73% | 10.9 | 22.6 | 2.1 | 0.083 |
| vote | Weka:JRip | 95.42% | 3.1 | 7.6 | 2.2 | 0.544 |
| vote | Witt:RIPPER | 94.26% | 3.0 | 6.1 | 2.0 | 0.110 |
| wdbc | Pypper | 94.37% | 3.8 | 8.6 | 2.3 | 0.065 |
| wdbc | Slipper | 95.25% | 13.8 | 24.1 | 1.7 | 0.077 |
| wdbc | Weka:JRip | 94.73% | 4.8 | 10.7 | 2.2 | 1.086 |
| wdbc | Witt:RIPPER | 93.84% | 4.9 | 11.8 | 2.4 | 0.569 |
| wilt | Pypper | 97.27% | 1.9 | 5.6 | 2.9 | 0.071 |
| wilt | Slipper | 97.17% | 12.2 | 34.9 | 2.9 | 0.140 |
| wilt | Weka:JRip | 97.33% | 2.1 | 5.6 | 2.7 | 1.237 |
| wilt | Witt:RIPPER | 94.33% | 8.6 | 20.3 | 2.0 | 1.814 |

