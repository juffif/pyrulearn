# Dense and modular rule learners compared

## Setup

**Full run** (25 small and 19 medium binary datasets). Quick sanity check instead: `python demos/dense_vs_modular.py` with no arguments.

The same rule-learning algorithms written twice: specialized for speed on a dense
matrix (`Dense...`), and built from the library's exchangeable components -- a
`RuleSearch`, an `Objective`, a covering or boosting loop -- so that they run on every
data representation and take any search. Each dense learner is paired with its
modular twin on the same data -- the dense one on the Boolean matrix it is written for, the
modular one on N-lists, the library's default representation; the modular learners are then
run again on Boolean data; and the modular `ORB` and `LRI` are run
with a beam search of width 5 and 10, which the dense versions don't have.

- **CPAR** -- Classification based on Predictive Association Rules (Yin & Han 2003), built from `GainAscentHillClimbing` with branching and a `WeightedCovering`; `DenseCPAR` is the same learner with a hand-written dense search.
  - **CPAR PRM** -- `gain_similarity=1`: only exact ties branch, which is close to PRM, the variant of the paper without the copies of a rule (`DenseCPAR PRM` on the dense search).
- **ENDER** -- Gradient boosting of rules, constant-step criterion and logistic loss (CS-Log), 500 rules, grown through the representation's own counting; `DenseENDER` is the same learner on a dense float matrix, and each variant has a dense twin too.
  - **ENDER CS-Exp** -- the exponential loss -- the best-ranked setting of the paper.
  - **ENDER MLRules** -- the Newton criterion and half the examples per round -- MLRules (Dembczynski et al. 2008).
- **Boomer** -- BOOMER's single-label case (Newton criterion, L2-regularized, 1000 rules) as an ENDER setting; `DenseBoomer` is the same on a dense float matrix.
- **ORB** -- Optimal rule boosting (10 rules), each rule found by a `RuleSearch` on the XGBoost-style gain. `ORB(greedy)` uses a greedy `HillClimbing`, and `DenseORB(greedy)` is the same learner with a hand-written dense greedy search. The dense version's exhaustive search is checked separately (see below).
  - **ORB(beam 5)** -- a `BeamSearch` of width 5.
  - **ORB(beam 10)** -- a `BeamSearch` of width 10 (`ORB`'s default).
- **LRI** -- Lightweight Rule Induction (50 DNF rules per class): terms by a greedy `HillClimbing` without a local-optimum stop on the weighted error `FP + k*FN`; `DenseLRI` is the same learner with a hand-written dense term search.
  - **LRI(beam 5)** -- terms grown by a `BeamSearch` of width 5 (at most 5 conditions).
  - **LRI(beam 10)** -- terms grown by a `BeamSearch` of width 10.

All learners run with their defaults (listed above) on 44 binary datasets from
`pyrulearn.experiments.catalog` -- every one of the small and medium size classes in the full run, numeric and symbolic attributes alike; `ORB` boosts a binary loss, so no multi-class
datasets. Exhaustive `ORB` was checked separately and left out -- see "Why exhaustive ORB
isn't in the main comparison". On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 96.67% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 1.7s per fit against 0.35s and 0.02s -- 5x as long as the beam search; 6 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

Protocol: 10-fold stratified cross-validation (5-fold for the medium datasets)
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals=8)`), the test fold encoded against that same
`DataSpec` -- every learner sees the identical binary features per fold, the dense learners as a
Boolean matrix and the modular ones as N-lists; the `[Boolean]` rows are the modular learners on
the same data as Boolean matrices. Each fit is capped at
300s; a time-out or an error counts as a failure (and as last in the ranking), not as the end
of the run. Fit times are wall-clock seconds of one fit in a worker process (see below for
how many ran at once).

Measures: test accuracy, number of rules and conditions, fit time, and whether a dense
learner and its modular twin learned the same model (the same rules, whatever their order).

Every learner runs single-threaded. Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

![fit time, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_fit_time.png)

![accuracy, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_accuracy.png)

![critical-difference diagram (accuracy)](dense_vs_modular_plots/dense_vs_modular_cd.png)

## Why exhaustive ORB isn't in the main comparison

On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 96.67% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 1.7s per fit against 0.35s and 0.02s -- 5x as long as the beam search; 6 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

Accuracy (mean over the folds that finished):

| dataset | ORB(greedy) | ORB(beam 5) | ORB(beam 10) | DenseORB(greedy) | DenseORB(exhaustive) | ORB(exhaustive) |
|---|--:|--:|--:|--:|--:|--:|
| breast-cancer | 69.58% | 69.94% | 67.49% | 69.58% |  |  |
| mofn-3-7-10 | 93.73% | 95.09% | 95.32% | 93.73% | 95.47% | 95.47% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.92% | 98.92% | 98.92% |
| tic-tac-toe | 83.09% | 87.89% | 85.91% | 83.09% | 83.92% |  |
| vote | 95.86% | 95.86% | 95.63% | 95.86% | 95.63% | 95.63% |

Fit time in seconds (mean over the folds that finished; a blank cell: every fold hit the cap):

| dataset | ORB(greedy) | ORB(beam 5) | ORB(beam 10) | DenseORB(greedy) | DenseORB(exhaustive) | ORB(exhaustive) |
|---|--:|--:|--:|--:|--:|--:|
| breast-cancer | 0.03 | 0.43 | 0.74 | 0.01 |  |  |
| mofn-3-7-10 | 0.02 | 0.10 | 0.17 | 0.01 | 0.28 | 0.77 |
| monks-problems-3 | 0.02 | 0.10 | 0.20 | 0.00 | 0.40 | 1.15 |
| tic-tac-toe | 0.02 | 0.18 | 0.34 | 0.01 | 62.80 |  |
| vote | 0.02 | 0.16 | 0.29 | 0.01 | 1.61 | 3.29 |

## Dense vs. modular

For each pair, over the folds where both fits finished: how often the two learned the same model (the same rules in any order), their mean accuracy, and the ratio of the modular to the dense fit time -- the geometric mean over datasets, so a value below 1 means the modular learner is faster. The dense learner on the Boolean matrix, the modular one on N-lists (its default); the modular learners on Boolean data are compared with N-lists below.

| pair | identical models | accuracy dense | accuracy modular | difference (points) | time modular / dense, small | medium | all |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPAR | 345/345 | 85.63% | 85.63% | +0.00 | 2.45 | 1.66 | 2.08 |
| CPAR PRM | 267/345 | 85.57% | 85.43% | -0.14 | 2.57 | 1.78 | 2.19 |
| ENDER | 333/345 | 85.83% | 85.83% | -0.00 | 2.02 | 0.95 | 1.46 |
| ENDER CS-Exp | 328/345 | 86.04% | 86.04% | +0.00 | 2.14 | 0.99 | 1.54 |
| ENDER MLRules | 315/345 | 86.19% | 86.15% | -0.04 | 1.82 | 0.81 | 1.28 |
| Boomer | 326/345 | 86.43% | 86.42% | -0.01 | 2.10 | 0.85 | 1.42 |
| ORB | 327/345 | 83.50% | 83.48% | -0.02 | 3.49 | 1.76 | 2.60 |
| LRI | 48/345 | 85.03% | 85.04% | +0.01 | 2.55 | 1.24 | 1.87 |

## Summary by learner (mean across the 44 datasets and their folds)

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DenseENDER CS-Exp | 86.04% | 396.4 | 1750.1 | 4.2 | 1.29 | 3.7 | 7.77 | 0 |
| ENDER CS-Exp | 86.04% | 396.4 | 1750.0 | 4.2 | 1.26 | 3.7 | 7.84 | 0 |
| DenseENDER | 85.83% | 406.6 | 1906.8 | 4.4 | 1.56 | 2.3 | 7.85 | 0 |
| ENDER | 85.83% | 406.7 | 1907.0 | 4.4 | 1.49 | 2.3 | 7.94 | 0 |
| ENDER MLRules | 86.15% | 346.0 | 1433.1 | 4.1 | 1.31 | 1.2 | 8.09 | 0 |
| DenseENDER MLRules | 86.19% | 346.0 | 1432.3 | 4.0 | 1.69 | 1.2 | 8.12 | 0 |
| LRI(beam 5) | 86.13% | 392.5 | 1365.8 | 3.5 | 5.38 | 4.2 | 8.23 | 0 |
| LRI | 85.04% | 396.3 | 1579.0 | 4.0 | 1.18 | 1.1 | 8.90 | 0 |
| DenseLRI | 85.03% | 396.4 | 1561.4 | 3.9 | 0.88 | 4.0 | 9.01 | 0 |
| LRI(beam 10) | 86.18% | 392.1 | 1405.0 | 3.6 | 9.95 | 3.7 | 9.02 | 0 |
| Boomer | 86.42% | 273.1 | 1793.4 | 5.4 | 2.95 | 4.1 | 10.34 | 0 |
| DenseBoomer | 86.43% | 272.9 | 1792.1 | 5.4 | 3.48 | 5.1 | 10.43 | 0 |
| CPAR | 85.63% | 373.5 | 2431.9 | 4.6 | 1.45 | 1.2 | 11.77 | 0 |
| DenseCPAR | 85.63% | 373.5 | 2431.9 | 4.6 | 1.07 | 1.2 | 11.77 | 0 |
| DenseCPAR PRM | 85.57% | 309.1 | 1941.8 | 4.6 | 1.04 | 2.2 | 12.77 | 0 |
| CPAR PRM | 85.43% | 307.3 | 1937.1 | 4.6 | 1.63 | 0.2 | 13.45 | 0 |
| ORB(beam 5) | 84.06% | 9.8 | 48.5 | 4.9 | 2.09 | 0.2 | 13.82 | 0 |
| ORB(beam 10) | 84.34% | 9.8 | 51.1 | 5.2 | 4.32 | 1.2 | 13.92 | 0 |
| DenseORB(greedy) | 83.50% | 9.7 | 39.6 | 4.0 | 0.03 | 0.1 | 14.36 | 0 |
| ORB(greedy) | 83.48% | 9.7 | 39.6 | 4.1 | 0.05 | 1.1 | 14.57 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised. A dense learner and its twin that learn the same models tie for rank, so the ranking separates the algorithms and the searches, not dense from modular.

### Small datasets only (25, fewer than 1,000 examples)

`SPECT`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `climate-model-simulation-crashes`, `colic`, `credit-approval`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `ilpd`, `ionosphere`, `kc2`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `sonar`, `tic-tac-toe`, `vote`, `wdbc`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ENDER CS-Exp | 84.75% | 396.9 | 1548.8 | 3.7 | 0.89 | 2.1 | 7.28 | 0 |
| DenseENDER CS-Exp | 84.75% | 396.9 | 1549.1 | 3.7 | 0.44 | 2.1 | 7.36 | 0 |
| ENDER | 84.45% | 406.1 | 1660.8 | 3.9 | 0.99 | 1.2 | 8.16 | 0 |
| DenseENDER | 84.45% | 406.0 | 1660.5 | 3.9 | 0.52 | 1.2 | 8.16 | 0 |
| DenseENDER MLRules | 84.95% | 343.1 | 1278.6 | 3.7 | 0.53 | 0.6 | 8.24 | 0 |
| ENDER MLRules | 84.89% | 343.1 | 1279.0 | 3.7 | 0.89 | 0.6 | 8.66 | 0 |
| DenseLRI | 83.66% | 396.3 | 1530.7 | 3.9 | 0.36 | 2.9 | 9.70 | 0 |
| LRI(beam 5) | 84.86% | 391.6 | 1328.7 | 3.4 | 4.68 | 1.1 | 9.70 | 0 |
| LRI | 83.68% | 396.3 | 1551.3 | 3.9 | 0.89 | 1.1 | 9.88 | 0 |
| LRI(beam 10) | 84.94% | 391.6 | 1370.5 | 3.5 | 8.78 | 1.6 | 10.42 | 0 |
| DenseBoomer | 85.42% | 229.2 | 1483.0 | 5.3 | 1.22 | 2.9 | 10.82 | 0 |
| Boomer | 85.40% | 229.3 | 1484.6 | 5.3 | 2.15 | 1.9 | 10.84 | 0 |
| CPAR | 84.42% | 180.7 | 856.5 | 4.2 | 0.44 | 0.7 | 11.22 | 0 |
| DenseCPAR | 84.42% | 180.7 | 856.5 | 4.2 | 0.19 | 0.7 | 11.22 | 0 |
| DenseCPAR PRM | 84.38% | 165.0 | 778.2 | 4.1 | 0.19 | 2.2 | 12.14 | 0 |
| CPAR PRM | 84.22% | 163.7 | 771.8 | 4.1 | 0.48 | 0.2 | 12.62 | 0 |
| ORB(beam 10) | 82.97% | 9.8 | 48.7 | 4.9 | 2.66 | 1.2 | 13.10 | 0 |
| DenseORB(greedy) | 82.05% | 9.7 | 36.5 | 3.7 | 0.01 | 0.1 | 13.22 | 0 |
| ORB(greedy) | 82.03% | 9.7 | 36.6 | 3.7 | 0.03 | 1.1 | 13.46 | 0 |
| ORB(beam 5) | 82.55% | 9.8 | 46.3 | 4.7 | 1.32 | 0.2 | 13.80 | 0 |

### Medium datasets only (19, 1,000 to 10,000 examples)

`Titanic`, `banknote-authentication`, `churn`, `compas-two-years`, `credit-g`, `heloc`, `kc1`, `kr-vs-kp`, `mofn-3-7-10`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `spambase`, `wilt`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LRI(beam 5) | 89.50% | 394.8 | 1463.4 | 3.7 | 7.20 | 3.2 | 6.29 | 0 |
| LRI(beam 10) | 89.44% | 393.4 | 1495.8 | 3.8 | 13.04 | 2.2 | 7.18 | 0 |
| ENDER MLRules | 89.46% | 353.6 | 1838.7 | 4.9 | 2.44 | 0.7 | 7.34 | 0 |
| DenseENDER | 89.47% | 408.1 | 2554.9 | 5.8 | 4.30 | 1.2 | 7.45 | 0 |
| LRI | 88.61% | 396.6 | 1652.0 | 4.2 | 1.93 | 0.1 | 7.61 | 0 |
| ENDER | 89.47% | 408.1 | 2555.0 | 5.8 | 2.82 | 1.2 | 7.66 | 0 |
| DenseENDER MLRules | 89.44% | 353.5 | 1836.6 | 4.9 | 4.74 | 0.7 | 7.97 | 0 |
| DenseLRI | 88.61% | 396.6 | 1642.3 | 4.1 | 2.23 | 1.1 | 8.11 | 0 |
| DenseENDER CS-Exp | 89.43% | 395.3 | 2279.2 | 5.4 | 3.52 | 1.7 | 8.32 | 0 |
| ENDER CS-Exp | 89.42% | 395.3 | 2279.5 | 5.4 | 2.23 | 1.7 | 8.58 | 0 |
| Boomer | 89.09% | 388.3 | 2606.1 | 5.8 | 5.05 | 2.2 | 9.68 | 0 |
| DenseBoomer | 89.08% | 388.2 | 2605.8 | 5.8 | 9.42 | 2.2 | 9.92 | 0 |
| CPAR | 88.80% | 880.9 | 6577.5 | 5.9 | 4.11 | 0.5 | 12.50 | 0 |
| DenseCPAR | 88.80% | 880.9 | 6577.5 | 5.9 | 3.41 | 0.5 | 12.50 | 0 |
| DenseCPAR PRM | 88.69% | 688.3 | 5003.7 | 5.8 | 3.25 | 0.0 | 13.61 | 0 |
| ORB(beam 5) | 88.02% | 9.7 | 54.3 | 5.5 | 4.12 | 0.0 | 13.84 | 0 |
| CPAR PRM | 88.60% | 685.5 | 5003.6 | 5.8 | 4.67 | 0.0 | 14.55 | 0 |
| ORB(beam 10) | 87.92% | 9.7 | 57.4 | 5.9 | 8.71 | 0.0 | 15.00 | 0 |
| DenseORB(greedy) | 87.32% | 9.7 | 47.5 | 4.9 | 0.07 | 0.0 | 15.87 | 0 |
| ORB(greedy) | 87.30% | 9.7 | 47.7 | 4.9 | 0.10 | 0.0 | 16.03 | 0 |

## Beam search for ORB and LRI

The dense `ORB` and `LRI` come with their own greedy (and, for `ORB`, exhaustive) search; the modular ones take any `RuleSearch`. Mean over all datasets:

| learner | accuracy | n_conditions | fit_time (s) |
|---|--:|--:|--:|
| ORB(greedy) | 83.48% | 39.6 | 0.05 |
| ORB(beam 5) | 84.06% | 48.5 | 2.09 |
| ORB(beam 10) | 84.34% | 51.1 | 4.32 |
| LRI | 85.04% | 1579.0 | 1.18 |
| LRI(beam 5) | 86.13% | 1365.8 | 5.38 |
| LRI(beam 10) | 86.18% | 1405.0 | 9.95 |

## The modular learners on Boolean data

The modular learners above run on `NListRepresentation`, the library's default. Here they are run again on `BooleanDataRepresentation`, the matrix the dense learners work on, over the folds where both finished; a time ratio above 1 means the N-list is faster:

| learner | accuracy, N-list | accuracy, Boolean | time Boolean / N-list (geometric mean over datasets) |
|---|--:|--:|--:|
| CPAR | 85.63% | 85.63% | 0.89 |
| CPAR PRM | 85.43% | 85.54% | 0.95 |
| ENDER | 85.83% | 85.84% | 0.76 |
| ENDER CS-Exp | 86.04% | 86.03% | 0.73 |
| ENDER MLRules | 86.15% | 86.19% | 0.77 |
| Boomer | 86.42% | 86.41% | 0.87 |
| ORB(greedy) | 83.48% | 83.52% | 0.95 |
| LRI | 85.04% | 85.04% | 0.98 |
| ORB(beam 5) | 84.06% | 84.04% | 0.72 |
| ORB(beam 10) | 84.34% | 84.22% | 0.71 |
| LRI(beam 5) | 86.13% | 86.13% | 0.86 |
| LRI(beam 10) | 86.18% | 86.18% | 0.84 |

## Per-dataset results

Accuracy (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 82.78% | 82.04% | 84.63% | 84.62% | 83.49% | 82.34% | 83.90% | 80.81% | 82.79% | 82.44% | 83.82% | 83.83% | 82.78% | 82.42% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% |
| Titanic | 78.84% | 78.61% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.07% | 79.68% | 79.45% | 79.68% | 79.14% | 78.84% | 78.76% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.68% |
| banknote-authentication | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% | 98.18% | 97.67% | 99.05% | 99.05% | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% |
| blood-transfusion-service-center | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 77.54% | 76.20% | 76.86% | 77.53% | 77.40% | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 78.20% |
| breast-cancer | 73.42% | 72.72% | 72.39% | 73.09% | 71.69% | 69.66% | 70.65% | 70.64% | 69.58% | 73.07% | 71.69% | 72.77% | 73.42% | 73.76% | 72.39% | 73.09% | 72.03% | 69.66% | 70.65% | 71.33% |
| breast-w | 93.42% | 93.84% | 96.28% | 96.57% | 95.57% | 95.99% | 95.71% | 96.28% | 94.85% | 95.13% | 96.00% | 95.57% | 93.42% | 93.56% | 96.28% | 96.57% | 95.57% | 95.99% | 95.71% | 96.00% |
| churn | 93.02% | 93.32% | 92.58% | 92.02% | 93.38% | 93.44% | 89.56% | 93.24% | 91.08% | 90.92% | 93.28% | 93.64% | 93.02% | 93.42% | 92.58% | 92.02% | 93.36% | 93.38% | 89.56% | 93.20% |
| climate-model-simulation-crashes | 93.52% | 93.70% | 92.96% | 93.15% | 94.07% | 93.52% | 92.41% | 93.89% | 92.78% | 92.78% | 93.89% | 93.70% | 93.52% | 93.89% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 94.26% |
| colic | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.07% | 84.26% | 82.36% | 84.81% | 85.35% | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.35% |
| compas-two-years | 65.88% | 65.95% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.75% | 67.22% | 67.30% | 67.18% | 67.47% | 65.88% | 66.62% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.89% |
| credit-approval | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 87.25% | 86.81% | 86.81% | 86.52% | 87.39% | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 86.96% |
| credit-g | 74.00% | 73.30% | 76.50% | 76.80% | 76.50% | 74.70% | 70.20% | 75.30% | 71.40% | 71.50% | 76.00% | 76.30% | 74.00% | 73.20% | 76.50% | 76.80% | 76.30% | 74.70% | 70.30% | 75.50% |
| cylinder-bands | 77.04% | 79.44% | 80.93% | 81.30% | 81.11% | 87.22% | 71.48% | 82.22% | 74.26% | 74.07% | 82.96% | 81.85% | 77.04% | 79.44% | 80.93% | 81.11% | 81.11% | 87.22% | 71.48% | 82.22% |
| diabetes | 76.03% | 75.90% | 74.74% | 74.48% | 74.61% | 74.09% | 74.87% | 75.39% | 74.35% | 74.48% | 76.04% | 75.78% | 76.03% | 76.42% | 74.74% | 74.48% | 74.61% | 74.09% | 74.87% | 75.53% |
| dresses-sales | 57.00% | 55.60% | 62.20% | 63.20% | 59.00% | 58.20% | 57.40% | 58.80% | 54.80% | 54.80% | 58.20% | 59.40% | 57.00% | 56.80% | 62.20% | 63.20% | 59.40% | 58.40% | 58.20% | 58.80% |
| heart-c | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 78.27% | 80.89% | 77.87% | 78.90% | 80.24% | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 80.25% |
| heart-h | 80.98% | 80.60% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.64% | 81.26% | 80.59% | 80.61% | 79.95% | 80.98% | 80.60% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.31% |
| heart-statlog | 82.22% | 80.00% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 76.30% | 80.00% | 80.37% | 76.67% | 78.15% | 82.22% | 81.11% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 75.56% |
| heloc | 70.23% | 70.00% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.88% | 70.25% | 70.14% | 71.72% | 71.72% | 70.23% | 70.42% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.55% |
| hepatitis | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 84.50% | 82.12% | 80.88% | 78.88% | 82.12% | 79.50% | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 83.87% | 80.88% |
| ilpd | 71.17% | 70.32% | 69.98% | 71.02% | 70.15% | 69.46% | 70.15% | 71.68% | 70.49% | 71.86% | 70.13% | 70.13% | 71.17% | 69.81% | 69.98% | 71.02% | 70.49% | 69.29% | 70.15% | 70.99% |
| ionosphere | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% | 91.72% | 91.73% | 93.16% | 92.60% | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% |
| kc1 | 84.21% | 83.97% | 85.63% | 85.49% | 84.87% | 84.12% | 85.02% | 85.44% | 84.97% | 85.16% | 85.63% | 85.40% | 84.21% | 83.88% | 85.63% | 85.59% | 84.87% | 83.97% | 85.02% | 85.30% |
| kc2 | 82.94% | 83.89% | 84.26% | 84.64% | 82.93% | 81.79% | 83.12% | 82.36% | 82.16% | 81.60% | 82.36% | 81.40% | 82.94% | 83.89% | 84.26% | 84.64% | 82.93% | 81.79% | 83.12% | 81.79% |
| kr-vs-kp | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% | 97.47% | 97.22% | 99.00% | 99.09% | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% |
| mofn-3-7-10 | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% | 94.64% | 95.54% | 100.00% | 100.00% | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% |
| molecular-biology_promoters | 80.36% | 85.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 82.91% | 81.27% | 86.00% | 80.27% | 82.27% | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 77.55% |
| monks-problems-1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% |
| monks-problems-2 | 96.17% | 97.17% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 67.21% | 75.70% | 77.53% | 93.35% | 95.35% | 96.17% | 97.50% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 71.39% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 98.20% | 98.92% | 98.92% | 97.84% | 97.84% | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 97.84% |
| mushroom | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% | 99.48% | 99.84% | 100.00% | 100.00% | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% |
| ozone-level-8hr | 93.73% | 93.73% | 93.92% | 93.92% | 94.28% | 94.28% | 93.01% | 94.24% | 92.98% | 92.94% | 94.28% | 93.96% | 93.73% | 93.80% | 94.00% | 93.96% | 94.20% | 94.28% | 93.09% | 94.20% |
| pc1 | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.06% | 92.16% | 93.06% | 92.97% | 92.16% | 92.88% | 93.06% | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.15% | 92.16% | 93.15% |
| pc3 | 89.63% | 88.74% | 88.74% | 89.38% | 89.12% | 88.61% | 89.38% | 89.38% | 89.19% | 89.12% | 89.31% | 88.74% | 89.63% | 88.87% | 88.74% | 89.38% | 88.99% | 88.48% | 89.38% | 89.06% |
| pc4 | 89.99% | 89.57% | 89.78% | 89.64% | 89.78% | 89.64% | 88.41% | 90.12% | 89.30% | 88.89% | 90.19% | 90.12% | 89.99% | 89.57% | 89.78% | 89.71% | 89.71% | 89.64% | 88.61% | 89.92% |
| phoneme | 85.90% | 85.44% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 84.97% | 81.72% | 81.42% | 85.33% | 85.27% | 85.90% | 85.51% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 85.25% |
| qsar-biodeg | 84.45% | 83.70% | 85.88% | 86.16% | 85.31% | 85.21% | 80.95% | 85.97% | 83.51% | 82.94% | 85.88% | 85.40% | 84.45% | 83.89% | 85.88% | 86.16% | 85.31% | 85.21% | 80.95% | 85.88% |
| sick | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% | 98.52% | 98.36% | 98.67% | 98.52% | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% |
| sonar | 76.02% | 75.00% | 84.14% | 82.69% | 85.10% | 85.62% | 79.33% | 81.29% | 74.98% | 79.86% | 82.71% | 81.74% | 76.02% | 76.43% | 84.14% | 82.69% | 85.10% | 86.12% | 78.83% | 81.29% |
| spambase | 93.76% | 94.18% | 95.28% | 95.09% | 95.00% | 95.52% | 92.20% | 95.39% | 92.70% | 92.89% | 95.26% | 95.35% | 93.76% | 94.18% | 95.28% | 95.09% | 95.00% | 95.52% | 92.20% | 95.39% |
| tic-tac-toe | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 96.97% | 84.45% | 85.07% | 99.06% | 99.06% | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 97.60% |
| vote | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% | 94.95% | 95.64% | 96.09% | 95.86% | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% |
| wdbc | 95.78% | 95.43% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.31% | 95.42% | 95.60% | 96.66% | 96.48% | 95.78% | 95.08% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.83% |
| wilt | 96.86% | 96.67% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.11% | 97.11% | 97.09% | 97.13% | 97.19% | 96.86% | 96.67% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.09% |

Fit time in seconds (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 0.26 | 0.28 | 0.69 | 0.62 | 0.72 | 1.37 | 0.02 | 0.91 | 0.26 | 0.50 | 3.48 | 6.53 | 0.08 | 0.08 | 0.24 | 0.19 | 0.24 | 0.40 | 0.00 | 0.24 |
| Titanic | 1.07 | 1.12 | 1.22 | 1.08 | 1.10 | 1.95 | 0.03 | 0.88 | 0.43 | 0.87 | 3.75 | 7.12 | 0.46 | 0.45 | 0.79 | 0.65 | 0.82 | 1.18 | 0.02 | 0.53 |
| banknote-authentication | 0.17 | 0.17 | 0.90 | 0.75 | 0.82 | 1.48 | 0.02 | 0.92 | 0.37 | 0.72 | 4.22 | 7.53 | 0.07 | 0.06 | 0.56 | 0.40 | 0.44 | 0.83 | 0.01 | 0.45 |
| blood-transfusion-service-center | 0.47 | 0.42 | 0.78 | 0.62 | 0.72 | 1.26 | 0.03 | 0.79 | 0.24 | 0.48 | 3.65 | 6.44 | 0.19 | 0.17 | 0.40 | 0.30 | 0.38 | 0.58 | 0.01 | 0.32 |
| breast-cancer | 0.60 | 0.63 | 1.00 | 0.87 | 0.82 | 1.68 | 0.03 | 0.73 | 0.39 | 0.73 | 3.87 | 7.17 | 0.21 | 0.24 | 0.39 | 0.34 | 0.36 | 0.58 | 0.01 | 0.25 |
| breast-w | 0.18 | 0.20 | 1.35 | 1.21 | 0.88 | 1.96 | 0.03 | 0.76 | 1.21 | 2.26 | 4.32 | 7.89 | 0.07 | 0.09 | 0.78 | 0.63 | 0.58 | 1.25 | 0.01 | 0.37 |
| churn | 9.51 | 9.59 | 4.32 | 3.57 | 3.59 | 7.29 | 0.17 | 2.57 | 3.04 | 6.03 | 7.83 | 14.17 | 8.32 | 7.08 | 5.55 | 4.53 | 6.24 | 13.05 | 0.11 | 2.50 |
| climate-model-simulation-crashes | 0.22 | 0.19 | 1.06 | 0.91 | 0.95 | 2.45 | 0.04 | 0.93 | 1.47 | 2.89 | 5.04 | 9.60 | 0.11 | 0.10 | 0.64 | 0.52 | 0.71 | 1.55 | 0.01 | 0.46 |
| colic | 0.36 | 0.41 | 0.86 | 0.80 | 0.81 | 2.02 | 0.03 | 1.08 | 0.67 | 1.36 | 4.55 | 8.37 | 0.20 | 0.19 | 0.48 | 0.44 | 0.64 | 1.09 | 0.01 | 0.55 |
| compas-two-years | 7.94 | 8.47 | 2.86 | 2.07 | 2.41 | 3.45 | 0.07 | 1.80 | 0.87 | 1.62 | 5.16 | 9.19 | 6.78 | 6.87 | 2.94 | 2.00 | 3.13 | 4.43 | 0.07 | 1.67 |
| credit-approval | 0.72 | 0.81 | 1.34 | 1.20 | 1.13 | 2.43 | 0.04 | 0.96 | 0.92 | 1.79 | 4.48 | 8.17 | 0.38 | 0.40 | 0.82 | 0.69 | 0.76 | 1.54 | 0.01 | 0.45 |
| credit-g | 1.43 | 1.64 | 1.69 | 1.47 | 1.44 | 3.04 | 0.06 | 1.11 | 0.91 | 1.74 | 4.63 | 8.61 | 0.64 | 0.70 | 1.05 | 0.83 | 1.04 | 2.19 | 0.02 | 0.65 |
| cylinder-bands | 0.73 | 0.95 | 2.13 | 1.90 | 1.38 | 8.16 | 0.08 | 1.11 | 5.09 | 10.43 | 7.12 | 13.53 | 0.34 | 0.41 | 1.41 | 1.19 | 1.31 | 6.90 | 0.03 | 0.55 |
| diabetes | 1.54 | 1.73 | 1.14 | 0.99 | 1.13 | 2.63 | 0.04 | 1.19 | 0.58 | 1.13 | 4.32 | 8.12 | 0.62 | 0.67 | 0.65 | 0.49 | 0.64 | 1.37 | 0.01 | 0.48 |
| dresses-sales | 1.42 | 1.67 | 1.49 | 1.26 | 1.00 | 2.12 | 0.05 | 0.88 | 1.85 | 3.67 | 4.95 | 9.52 | 0.70 | 0.75 | 0.86 | 0.75 | 0.75 | 1.52 | 0.02 | 0.41 |
| heart-c | 0.32 | 0.36 | 0.91 | 0.82 | 0.80 | 1.96 | 0.03 | 0.80 | 0.50 | 1.10 | 4.22 | 7.73 | 0.11 | 0.11 | 0.34 | 0.31 | 0.35 | 0.70 | 0.01 | 0.26 |
| heart-h | 0.38 | 0.38 | 0.84 | 0.79 | 0.79 | 1.45 | 0.03 | 0.74 | 0.52 | 1.02 | 3.80 | 7.18 | 0.13 | 0.13 | 0.33 | 0.30 | 0.33 | 0.54 | 0.01 | 0.25 |
| heart-statlog | 0.31 | 0.31 | 0.83 | 0.80 | 0.79 | 1.80 | 0.03 | 0.78 | 0.52 | 1.05 | 4.05 | 7.41 | 0.10 | 0.11 | 0.34 | 0.35 | 0.34 | 0.63 | 0.01 | 0.29 |
| heloc | 24.13 | 37.55 | 7.28 | 5.28 | 7.51 | 15.57 | 0.30 | 4.44 | 5.02 | 10.05 | 12.44 | 21.23 | 20.94 | 26.19 | 14.84 | 11.85 | 21.73 | 40.06 | 0.26 | 7.71 |
| hepatitis | 0.13 | 0.14 | 0.60 | 0.59 | 0.58 | 1.46 | 0.02 | 0.73 | 0.40 | 0.76 | 3.92 | 7.32 | 0.04 | 0.05 | 0.20 | 0.19 | 0.20 | 0.41 | 0.00 | 0.18 |
| ilpd | 1.11 | 1.22 | 1.15 | 1.07 | 1.03 | 1.99 | 0.04 | 0.93 | 0.72 | 1.45 | 4.05 | 7.74 | 0.46 | 0.45 | 0.58 | 0.47 | 0.54 | 1.04 | 0.01 | 0.41 |
| ionosphere | 0.17 | 0.20 | 0.96 | 0.90 | 0.93 | 2.31 | 0.04 | 1.07 | 3.18 | 6.53 | 6.71 | 13.01 | 0.07 | 0.08 | 0.54 | 0.49 | 0.67 | 1.36 | 0.01 | 0.40 |
| kc1 | 3.49 | 3.13 | 1.83 | 1.42 | 1.62 | 3.09 | 0.05 | 1.37 | 1.97 | 4.06 | 5.59 | 10.59 | 2.26 | 2.03 | 2.25 | 1.53 | 2.56 | 4.62 | 0.04 | 1.26 |
| kc2 | 0.53 | 0.55 | 0.97 | 0.81 | 0.89 | 1.99 | 0.03 | 0.98 | 1.86 | 3.70 | 5.18 | 9.54 | 0.25 | 0.24 | 0.61 | 0.49 | 0.70 | 1.45 | 0.01 | 0.47 |
| kr-vs-kp | 0.45 | 0.45 | 2.22 | 1.80 | 1.49 | 2.88 | 0.06 | 1.65 | 0.72 | 1.33 | 5.23 | 9.46 | 0.19 | 0.19 | 1.58 | 1.19 | 1.22 | 2.09 | 0.02 | 0.85 |
| mofn-3-7-10 | 0.57 | 0.70 | 0.67 | 0.50 | 0.78 | 1.38 | 0.02 | 1.15 | 0.12 | 0.19 | 3.55 | 6.33 | 0.18 | 0.21 | 0.41 | 0.23 | 0.33 | 0.55 | 0.01 | 0.43 |
| molecular-biology_promoters | 0.08 | 0.08 | 0.63 | 0.62 | 0.63 | 1.67 | 0.03 | 0.70 | 1.11 | 2.20 | 5.56 | 10.17 | 0.04 | 0.04 | 0.30 | 0.28 | 0.36 | 0.74 | 0.01 | 0.26 |
| monks-problems-1 | 0.14 | 0.14 | 0.73 | 0.64 | 0.76 | 0.99 | 0.02 | 0.56 | 0.12 | 0.22 | 3.47 | 6.34 | 0.05 | 0.05 | 0.31 | 0.25 | 0.28 | 0.36 | 0.00 | 0.20 |
| monks-problems-2 | 0.52 | 0.58 | 0.74 | 0.63 | 0.90 | 1.51 | 0.03 | 1.05 | 0.12 | 0.25 | 3.46 | 6.57 | 0.17 | 0.17 | 0.35 | 0.28 | 0.41 | 0.55 | 0.01 | 0.31 |
| monks-problems-3 | 0.10 | 0.10 | 0.66 | 0.61 | 0.72 | 1.35 | 0.02 | 0.74 | 0.11 | 0.22 | 3.18 | 5.97 | 0.03 | 0.04 | 0.31 | 0.24 | 0.31 | 0.47 | 0.01 | 0.31 |
| mushroom | 0.34 | 0.32 | 3.51 | 2.77 | 2.77 | 5.95 | 0.12 | 2.84 | 4.54 | 9.98 | 8.44 | 13.67 | 0.17 | 0.17 | 4.71 | 4.11 | 4.51 | 7.37 | 0.07 | 2.37 |
| ozone-level-8hr | 1.87 | 2.05 | 4.49 | 3.52 | 3.89 | 12.82 | 0.26 | 3.09 | 22.16 | 47.85 | 17.26 | 31.60 | 1.37 | 1.35 | 11.19 | 9.63 | 11.93 | 32.61 | 0.24 | 5.99 |
| pc1 | 0.99 | 0.88 | 1.48 | 1.16 | 1.21 | 2.29 | 0.04 | 1.13 | 2.13 | 4.33 | 5.51 | 9.97 | 0.51 | 0.44 | 0.99 | 0.81 | 1.19 | 2.31 | 0.02 | 0.67 |
| pc3 | 1.80 | 1.85 | 2.14 | 1.92 | 1.89 | 4.22 | 0.10 | 1.53 | 5.86 | 12.08 | 7.71 | 14.79 | 1.17 | 1.12 | 2.56 | 2.17 | 3.04 | 6.78 | 0.04 | 1.25 |
| pc4 | 1.25 | 1.26 | 2.01 | 1.72 | 2.06 | 4.67 | 0.08 | 1.63 | 6.95 | 14.34 | 8.26 | 15.12 | 0.75 | 0.77 | 2.26 | 1.92 | 2.76 | 6.37 | 0.04 | 1.12 |
| phoneme | 8.04 | 8.06 | 2.81 | 2.10 | 2.93 | 4.42 | 0.07 | 1.87 | 0.58 | 1.11 | 4.97 | 8.55 | 5.92 | 5.76 | 2.89 | 1.89 | 2.86 | 5.05 | 0.04 | 1.57 |
| qsar-biodeg | 1.03 | 1.29 | 1.96 | 1.78 | 1.52 | 4.04 | 0.07 | 1.51 | 4.52 | 9.13 | 7.52 | 14.48 | 0.54 | 0.65 | 2.01 | 1.56 | 1.93 | 4.51 | 0.03 | 0.87 |
| sick | 0.65 | 0.68 | 2.89 | 2.30 | 2.17 | 3.54 | 0.07 | 1.54 | 1.30 | 2.39 | 4.96 | 8.79 | 0.40 | 0.40 | 2.56 | 2.10 | 2.49 | 4.34 | 0.04 | 1.21 |
| sonar | 0.17 | 0.17 | 1.11 | 1.01 | 1.04 | 3.08 | 0.04 | 1.04 | 7.23 | 15.00 | 9.72 | 19.05 | 0.08 | 0.07 | 0.52 | 0.50 | 0.65 | 1.89 | 0.01 | 0.39 |
| spambase | 10.12 | 7.96 | 6.75 | 5.54 | 5.10 | 11.13 | 0.30 | 3.70 | 16.23 | 36.42 | 14.89 | 28.13 | 11.50 | 6.35 | 20.54 | 18.22 | 19.81 | 37.67 | 0.28 | 10.19 |
| tic-tac-toe | 0.27 | 0.27 | 0.99 | 0.93 | 1.00 | 1.90 | 0.03 | 1.07 | 0.20 | 0.40 | 3.88 | 7.28 | 0.11 | 0.10 | 0.59 | 0.46 | 0.60 | 0.93 | 0.01 | 0.49 |
| vote | 0.12 | 0.12 | 0.56 | 0.50 | 0.60 | 1.31 | 0.02 | 0.80 | 0.18 | 0.33 | 3.31 | 6.00 | 0.04 | 0.04 | 0.23 | 0.18 | 0.24 | 0.40 | 0.00 | 0.27 |
| wdbc | 0.12 | 0.14 | 1.21 | 1.08 | 1.23 | 2.99 | 0.05 | 1.03 | 3.49 | 6.98 | 6.82 | 12.86 | 0.06 | 0.06 | 0.81 | 0.72 | 0.97 | 2.29 | 0.02 | 0.53 |
| wilt | 3.31 | 1.47 | 2.45 | 1.55 | 1.97 | 2.73 | 0.05 | 1.86 | 0.61 | 1.25 | 4.81 | 8.35 | 2.60 | 1.01 | 2.07 | 1.27 | 1.98 | 2.89 | 0.03 | 1.16 |

