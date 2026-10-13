# Dense and modular rule learners compared

## Setup

**Full run** (25 small and 19 medium binary datasets). Quick sanity check instead: `python demos/dense_vs_modular.py` with no arguments.

The same rule-learning algorithms written twice: specialized for speed on a dense
matrix (`Dense...`), and built from the library's exchangeable components -- a
`RuleSearch`, an `Objective`, a covering or boosting loop -- so that they run on every
data representation and take any search. Each dense learner is paired with its
modular twin on the same Boolean data; the modular learners are then run again on
N-lists, the library's default representation; and the modular `ORB` and `LRI` are run
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
isn't in the main comparison". On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 93.49% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 15.2s per fit against 0.11s and 0.01s -- 134x as long as the beam search; 3 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

Protocol: 10-fold stratified cross-validation (5-fold for the medium datasets)
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals=8)`), the test fold encoded against that same
`DataSpec` -- every learner sees the identical Boolean feature matrix per fold; the
`[N-list]` rows are the modular learners on the same data as N-lists. Each fit is capped at
300s; a time-out or an error counts as a failure (and as last in the ranking), not as the end
of the run. Fit times are wall-clock seconds of one fit in a worker process, with nothing else
running.

Measures: test accuracy, number of rules and conditions, fit time, and whether a dense
learner and its modular twin learned the same model (the same rules, whatever their order).

![fit time, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_fit_time.png)

![accuracy, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_accuracy.png)

![critical-difference diagram (accuracy)](dense_vs_modular_plots/dense_vs_modular_cd.png)

## Why exhaustive ORB isn't in the main comparison

On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 93.49% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 15.2s per fit against 0.11s and 0.01s -- 134x as long as the beam search; 3 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

Accuracy (mean over the folds that finished):

| dataset | ORB(greedy) | ORB(beam 5) | ORB(beam 10) | DenseORB(greedy) | DenseORB(exhaustive) | ORB(exhaustive) |
|---|--:|--:|--:|--:|--:|--:|
| breast-cancer | 69.58% | 69.94% | 67.49% | 69.58% |  |  |
| mofn-3-7-10 | 93.73% | 95.09% | 95.32% | 93.73% | 95.47% | 95.47% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.92% | 98.92% | 98.92% |
| tic-tac-toe | 83.09% | 87.89% | 85.91% | 83.09% | 83.92% | 83.92% |
| vote | 95.86% | 95.86% | 95.63% | 95.86% | 95.63% | 95.63% |

Fit time in seconds (mean over the folds that finished; a blank cell: every fold hit the cap):

| dataset | ORB(greedy) | ORB(beam 5) | ORB(beam 10) | DenseORB(greedy) | DenseORB(exhaustive) | ORB(exhaustive) |
|---|--:|--:|--:|--:|--:|--:|
| breast-cancer | 0.02 | 0.14 | 0.24 | 0.01 |  |  |
| mofn-3-7-10 | 0.01 | 0.03 | 0.04 | 0.00 | 0.49 | 0.19 |
| monks-problems-3 | 0.01 | 0.04 | 0.05 | 0.00 | 0.60 | 0.33 |
| tic-tac-toe | 0.01 | 0.07 | 0.15 | 0.01 | 42.55 | 59.15 |
| vote | 0.01 | 0.04 | 0.08 | 0.00 | 1.18 | 1.00 |

## Dense vs. modular

For each pair, over the folds where both fits finished: how often the two learned the same model (the same rules in any order), their mean accuracy, and the ratio of the modular to the dense fit time -- the geometric mean over datasets, so a value below 1 means the modular learner is faster. Both on the same Boolean data.

| pair | identical models | accuracy dense | accuracy modular | difference (points) | time modular / dense, small | medium | all |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPAR | 345/345 | 85.63% | 85.63% | +0.00 | 2.36 | 2.66 | 2.48 |
| CPAR PRM | 229/345 | 85.57% | 85.54% | -0.03 | 2.47 | 3.25 | 2.78 |
| ENDER | 336/345 | 85.83% | 85.85% | +0.02 | 1.27 | 1.09 | 1.19 |
| ENDER CS-Exp | 319/345 | 86.03% | 86.03% | -0.00 | 1.35 | 1.10 | 1.24 |
| ENDER MLRules | 311/345 | 86.18% | 86.19% | +0.00 | 1.17 | 1.21 | 1.19 |
| Boomer | 307/345 | 86.40% | 86.40% | +0.00 | 1.51 | 1.99 | 1.70 |
| ORB | 324/345 | 83.50% | 83.52% | +0.02 | 2.06 | 2.14 | 2.09 |
| LRI | 48/345 | 85.03% | 85.04% | +0.01 | 1.70 | 0.70 | 1.16 |

## Summary by learner (mean across the 44 datasets and their folds)

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ENDER MLRules | 86.19% | 345.9 | 1432.1 | 4.05 | 1.45 | 3.1 | 7.57 | 0 |
| ENDER | 85.85% | 406.6 | 1906.7 | 4.44 | 1.25 | 2.3 | 7.80 | 0 |
| DenseENDER CS-Exp | 86.03% | 396.4 | 1749.9 | 4.19 | 0.96 | 4.1 | 7.88 | 0 |
| ENDER CS-Exp | 86.03% | 396.4 | 1749.9 | 4.19 | 1.07 | 3.1 | 7.88 | 0 |
| DenseENDER | 85.83% | 406.6 | 1906.8 | 4.44 | 1.15 | 2.3 | 8.02 | 0 |
| DenseENDER MLRules | 86.18% | 346.0 | 1432.4 | 4.05 | 1.22 | 0.7 | 8.03 | 0 |
| LRI(beam 5) | 86.13% | 392.5 | 1365.8 | 3.47 | 5.30 | 4.2 | 8.33 | 0 |
| LRI | 85.04% | 396.3 | 1579.0 | 3.98 | 1.60 | 1.1 | 8.89 | 0 |
| DenseLRI | 85.03% | 396.4 | 1561.4 | 3.93 | 2.40 | 4.0 | 8.99 | 0 |
| LRI(beam 10) | 86.18% | 392.1 | 1405.0 | 3.58 | 9.29 | 2.7 | 9.14 | 0 |
| Boomer | 86.40% | 273.0 | 1793.0 | 5.42 | 4.53 | 3.1 | 10.50 | 0 |
| DenseBoomer | 86.40% | 273.0 | 1790.7 | 5.42 | 2.41 | 4.1 | 10.60 | 0 |
| CPAR | 85.63% | 373.5 | 2431.9 | 4.64 | 1.84 | 1.2 | 11.84 | 0 |
| DenseCPAR | 85.63% | 373.5 | 2431.9 | 4.64 | 0.79 | 1.2 | 11.84 | 0 |
| DenseCPAR PRM | 85.57% | 309.2 | 1943.6 | 4.58 | 0.81 | 2.2 | 12.69 | 0 |
| CPAR PRM | 85.54% | 307.4 | 1933.4 | 4.58 | 2.98 | 1.2 | 13.08 | 0 |
| ORB(beam 10) | 84.23% | 9.8 | 51.1 | 5.18 | 5.97 | 2.2 | 13.89 | 0 |
| ORB(beam 5) | 84.03% | 9.8 | 48.5 | 4.92 | 3.20 | 0.2 | 14.02 | 0 |
| DenseORB(greedy) | 83.50% | 9.7 | 39.6 | 4.04 | 0.05 | 0.1 | 14.50 | 0 |
| ORB(greedy) | 83.52% | 9.7 | 39.6 | 4.05 | 0.10 | 1.1 | 14.52 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised. A dense learner and its twin that learn the same models tie for rank, so the ranking separates the algorithms and the searches, not dense from modular.

### Small datasets only (25, fewer than 1,000 examples)

`SPECT`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `climate-model-simulation-crashes`, `colic`, `credit-approval`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `ilpd`, `ionosphere`, `kc2`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `sonar`, `tic-tac-toe`, `vote`, `wdbc`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DenseENDER CS-Exp | 84.74% | 396.8 | 1548.6 | 3.75 | 0.31 | 2.6 | 7.40 | 0 |
| ENDER CS-Exp | 84.73% | 396.9 | 1548.7 | 3.75 | 0.39 | 1.6 | 7.48 | 0 |
| ENDER | 84.47% | 406.1 | 1660.4 | 3.91 | 0.45 | 1.2 | 8.00 | 0 |
| ENDER MLRules | 84.93% | 342.9 | 1277.6 | 3.72 | 0.43 | 1.1 | 8.10 | 0 |
| DenseENDER MLRules | 84.94% | 343.1 | 1278.1 | 3.72 | 0.38 | 0.1 | 8.14 | 0 |
| DenseENDER | 84.44% | 406.1 | 1660.4 | 3.91 | 0.36 | 1.2 | 8.36 | 0 |
| DenseLRI | 83.66% | 396.3 | 1530.7 | 3.86 | 0.48 | 2.9 | 9.58 | 0 |
| LRI(beam 5) | 84.86% | 391.6 | 1328.7 | 3.39 | 2.61 | 1.1 | 9.84 | 0 |
| LRI | 83.68% | 396.3 | 1551.3 | 3.91 | 0.58 | 1.1 | 9.86 | 0 |
| LRI(beam 10) | 84.94% | 391.6 | 1370.5 | 3.49 | 4.73 | 1.6 | 10.42 | 0 |
| Boomer | 85.38% | 229.1 | 1483.7 | 5.28 | 1.38 | 2.4 | 10.84 | 0 |
| DenseBoomer | 85.39% | 229.3 | 1482.5 | 5.27 | 0.90 | 2.4 | 10.88 | 0 |
| CPAR | 84.42% | 180.7 | 856.5 | 4.16 | 0.29 | 0.7 | 11.26 | 0 |
| DenseCPAR | 84.42% | 180.7 | 856.5 | 4.16 | 0.12 | 0.7 | 11.26 | 0 |
| DenseCPAR PRM | 84.38% | 165.0 | 778.2 | 4.13 | 0.13 | 2.2 | 12.18 | 0 |
| CPAR PRM | 84.34% | 164.3 | 771.7 | 4.13 | 0.34 | 0.2 | 12.24 | 0 |
| ORB(greedy) | 82.07% | 9.7 | 36.5 | 3.73 | 0.02 | 1.1 | 13.28 | 0 |
| DenseORB(greedy) | 82.05% | 9.7 | 36.5 | 3.73 | 0.01 | 0.1 | 13.32 | 0 |
| ORB(beam 10) | 82.82% | 9.8 | 48.7 | 4.92 | 1.51 | 1.2 | 13.44 | 0 |
| ORB(beam 5) | 82.52% | 9.8 | 46.4 | 4.69 | 0.73 | 0.2 | 14.12 | 0 |

### Medium datasets only (19, 1,000 to 10,000 examples)

`Titanic`, `banknote-authentication`, `churn`, `compas-two-years`, `credit-g`, `heloc`, `kc1`, `kr-vs-kp`, `mofn-3-7-10`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `spambase`, `wilt`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LRI(beam 5) | 89.50% | 394.8 | 1463.4 | 3.70 | 12.37 | 3.2 | 6.34 | 0 |
| ENDER MLRules | 89.50% | 353.8 | 1838.7 | 4.92 | 4.12 | 2.0 | 6.87 | 0 |
| LRI(beam 10) | 89.44% | 393.4 | 1495.8 | 3.79 | 21.28 | 1.2 | 7.45 | 0 |
| ENDER | 89.48% | 408.1 | 2554.8 | 5.83 | 3.34 | 1.2 | 7.53 | 0 |
| DenseENDER | 89.48% | 408.1 | 2555.2 | 5.83 | 3.21 | 1.2 | 7.58 | 0 |
| LRI | 88.61% | 396.6 | 1652.0 | 4.16 | 4.28 | 0.1 | 7.61 | 0 |
| DenseENDER MLRules | 89.45% | 353.6 | 1838.5 | 4.92 | 3.44 | 0.7 | 7.89 | 0 |
| DenseLRI | 88.61% | 396.6 | 1642.3 | 4.14 | 7.46 | 1.1 | 8.21 | 0 |
| ENDER CS-Exp | 89.44% | 395.3 | 2279.3 | 5.37 | 2.87 | 1.5 | 8.39 | 0 |
| DenseENDER CS-Exp | 89.43% | 395.3 | 2279.4 | 5.37 | 2.66 | 1.5 | 8.50 | 0 |
| Boomer | 89.07% | 388.5 | 2607.1 | 5.81 | 12.85 | 0.7 | 10.05 | 0 |
| DenseBoomer | 89.04% | 388.2 | 2601.8 | 5.81 | 6.39 | 1.7 | 10.24 | 0 |
| CPAR | 88.80% | 880.9 | 6577.5 | 5.92 | 5.92 | 0.5 | 12.61 | 0 |
| DenseCPAR | 88.80% | 880.9 | 6577.5 | 5.92 | 2.56 | 0.5 | 12.61 | 0 |
| DenseCPAR PRM | 88.70% | 688.7 | 5010.3 | 5.76 | 2.61 | 0.0 | 13.37 | 0 |
| ORB(beam 5) | 88.02% | 9.7 | 54.2 | 5.53 | 9.72 | 0.0 | 13.89 | 0 |
| CPAR PRM | 88.70% | 684.2 | 4990.7 | 5.77 | 9.94 | 1.0 | 14.18 | 0 |
| ORB(beam 10) | 87.93% | 9.7 | 57.4 | 5.87 | 17.69 | 1.0 | 14.47 | 0 |
| DenseORB(greedy) | 87.32% | 9.7 | 47.5 | 4.86 | 0.14 | 0.0 | 16.05 | 0 |
| ORB(greedy) | 87.31% | 9.7 | 47.7 | 4.88 | 0.30 | 0.0 | 16.16 | 0 |

## Beam search for ORB and LRI

The dense `ORB` and `LRI` come with their own greedy (and, for `ORB`, exhaustive) search; the modular ones take any `RuleSearch`. Mean over all datasets:

| learner | accuracy | n_conditions | fit_time (s) |
|---|--:|--:|--:|
| ORB(greedy) | 83.52% | 39.6 | 0.10 |
| ORB(beam 5) | 84.03% | 48.5 | 3.20 |
| ORB(beam 10) | 84.23% | 51.1 | 5.97 |
| LRI | 85.04% | 1579.0 | 1.60 |
| LRI(beam 5) | 86.13% | 1365.8 | 5.30 |
| LRI(beam 10) | 86.18% | 1405.0 | 9.29 |

## The modular learners on N-lists

The same modular learners on `NListRepresentation` (the library's default, which the dense learners don't use: they work on the Boolean matrix) against Boolean data, over the folds where both finished:

| learner | accuracy, Boolean | accuracy, N-list | time N-list / Boolean (geometric mean over datasets) |
|---|--:|--:|--:|
| CPAR | 85.63% | 85.63% | 1.04 |
| CPAR PRM | 85.54% | 85.48% | 0.94 |
| ENDER | 85.85% | 85.85% | 1.32 |
| ENDER CS-Exp | 86.03% | 86.03% | 1.33 |
| ENDER MLRules | 86.19% | 86.16% | 1.17 |
| Boomer | 86.40% | 86.39% | 0.93 |
| ORB(greedy) | 83.52% | 83.51% | 1.26 |
| LRI | 85.04% | 85.04% | 0.87 |
| ORB(beam 5) | 84.03% | 84.04% | 1.28 |
| ORB(beam 10) | 84.23% | 84.22% | 1.32 |
| LRI(beam 5) | 86.13% | 86.13% | 1.03 |
| LRI(beam 10) | 86.18% | 86.18% | 1.05 |

## Per-dataset results

Accuracy (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 82.78% | 82.42% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% | 82.79% | 82.44% | 83.82% | 83.83% | 82.78% | 82.42% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% |
| Titanic | 78.84% | 78.61% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.07% | 79.68% | 79.45% | 79.68% | 79.14% | 78.84% | 78.76% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.68% |
| banknote-authentication | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% | 98.18% | 97.67% | 99.05% | 99.05% | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% |
| blood-transfusion-service-center | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 77.54% | 76.20% | 76.86% | 77.53% | 77.40% | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 78.20% |
| breast-cancer | 73.42% | 72.36% | 72.39% | 73.09% | 71.69% | 68.94% | 70.65% | 70.64% | 69.58% | 73.07% | 71.69% | 72.77% | 73.42% | 73.76% | 72.39% | 73.09% | 71.69% | 69.66% | 70.65% | 71.33% |
| breast-w | 93.42% | 93.70% | 96.57% | 96.42% | 95.57% | 95.99% | 95.71% | 96.28% | 94.85% | 95.13% | 96.00% | 95.57% | 93.42% | 93.56% | 96.28% | 96.57% | 95.57% | 95.99% | 95.71% | 96.00% |
| churn | 93.02% | 93.30% | 92.58% | 92.02% | 93.38% | 93.44% | 89.56% | 93.24% | 91.08% | 90.92% | 93.28% | 93.64% | 93.02% | 93.40% | 92.58% | 92.02% | 93.38% | 93.40% | 89.56% | 93.20% |
| climate-model-simulation-crashes | 93.52% | 93.89% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 93.89% | 92.78% | 92.78% | 93.89% | 93.70% | 93.52% | 93.89% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 94.26% |
| colic | 83.98% | 83.18% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.07% | 84.26% | 82.36% | 84.81% | 85.35% | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.35% |
| compas-two-years | 65.88% | 66.41% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.75% | 67.22% | 67.30% | 67.18% | 67.47% | 65.88% | 66.62% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.89% |
| credit-approval | 85.22% | 85.65% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 87.25% | 86.81% | 86.81% | 86.52% | 87.39% | 85.22% | 85.51% | 86.52% | 86.67% | 86.67% | 85.22% | 86.23% | 86.96% |
| credit-g | 74.00% | 73.70% | 76.50% | 76.80% | 76.80% | 74.70% | 70.20% | 75.30% | 71.40% | 71.50% | 76.00% | 76.30% | 74.00% | 73.20% | 76.50% | 76.80% | 76.50% | 74.10% | 70.30% | 75.50% |
| cylinder-bands | 77.04% | 79.44% | 80.93% | 80.93% | 81.11% | 87.04% | 71.48% | 82.22% | 74.81% | 74.44% | 82.96% | 81.85% | 77.04% | 79.44% | 81.11% | 80.93% | 81.85% | 87.22% | 71.48% | 82.22% |
| diabetes | 76.03% | 76.29% | 74.74% | 74.48% | 74.87% | 74.09% | 74.87% | 75.39% | 74.35% | 74.48% | 76.04% | 75.78% | 76.03% | 76.42% | 74.74% | 74.48% | 74.87% | 74.09% | 74.87% | 75.53% |
| dresses-sales | 57.00% | 57.20% | 62.60% | 63.20% | 58.80% | 58.40% | 58.20% | 58.80% | 54.80% | 54.80% | 58.20% | 59.40% | 57.00% | 56.80% | 62.20% | 63.20% | 58.80% | 58.40% | 58.20% | 58.80% |
| heart-c | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 78.27% | 80.89% | 77.87% | 78.90% | 80.24% | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 80.25% |
| heart-h | 80.98% | 80.94% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.64% | 81.26% | 80.59% | 80.61% | 79.95% | 80.98% | 80.60% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.31% |
| heart-statlog | 82.22% | 80.37% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 76.30% | 79.26% | 80.37% | 76.67% | 78.15% | 82.22% | 81.11% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 75.56% |
| heloc | 70.23% | 70.12% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.88% | 70.25% | 70.14% | 71.72% | 71.72% | 70.23% | 70.50% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.55% |
| hepatitis | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 84.50% | 82.12% | 80.21% | 78.88% | 82.12% | 79.50% | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 83.87% | 80.88% |
| ilpd | 71.17% | 70.32% | 69.98% | 71.02% | 70.49% | 69.12% | 70.15% | 71.68% | 70.67% | 71.86% | 70.13% | 70.13% | 71.17% | 69.81% | 69.98% | 71.02% | 70.49% | 69.12% | 70.15% | 70.99% |
| ionosphere | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% | 92.01% | 91.73% | 93.16% | 92.60% | 90.88% | 89.17% | 94.29% | 93.73% | 94.02% | 94.01% | 90.02% | 93.44% |
| kc1 | 84.21% | 83.97% | 85.63% | 85.59% | 84.87% | 83.97% | 85.02% | 85.44% | 84.97% | 85.16% | 85.63% | 85.40% | 84.21% | 83.97% | 85.63% | 85.49% | 84.87% | 83.97% | 85.02% | 85.30% |
| kc2 | 82.94% | 83.70% | 84.26% | 84.64% | 82.93% | 81.79% | 83.12% | 82.36% | 81.78% | 81.21% | 82.36% | 81.40% | 82.94% | 83.89% | 84.26% | 84.64% | 82.93% | 81.60% | 83.12% | 81.79% |
| kr-vs-kp | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% | 97.47% | 97.22% | 99.00% | 99.09% | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% |
| mofn-3-7-10 | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% | 94.64% | 95.54% | 100.00% | 100.00% | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% |
| molecular-biology_promoters | 80.36% | 85.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 82.91% | 81.27% | 82.27% | 80.27% | 82.27% | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 77.55% |
| monks-problems-1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% |
| monks-problems-2 | 96.17% | 97.50% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 67.21% | 75.70% | 77.53% | 93.35% | 95.35% | 96.17% | 97.50% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 71.39% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 98.20% | 98.92% | 98.92% | 97.84% | 97.84% | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 97.84% |
| mushroom | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% | 99.48% | 99.84% | 100.00% | 100.00% | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% |
| ozone-level-8hr | 93.73% | 93.73% | 94.00% | 93.96% | 94.32% | 94.28% | 93.01% | 94.24% | 92.98% | 92.94% | 94.28% | 93.96% | 93.73% | 93.76% | 94.00% | 93.96% | 94.00% | 94.16% | 93.09% | 94.20% |
| pc1 | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.06% | 92.16% | 93.06% | 92.97% | 92.16% | 92.88% | 93.06% | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.15% | 92.16% | 93.15% |
| pc3 | 89.63% | 88.87% | 88.80% | 89.44% | 89.12% | 88.42% | 89.38% | 89.38% | 89.19% | 89.12% | 89.31% | 88.74% | 89.63% | 88.87% | 88.74% | 89.44% | 89.12% | 88.48% | 89.38% | 89.06% |
| pc4 | 89.99% | 89.57% | 89.78% | 89.71% | 90.12% | 89.64% | 88.61% | 90.12% | 89.30% | 88.89% | 90.19% | 90.12% | 89.99% | 89.71% | 89.78% | 89.64% | 89.78% | 89.64% | 88.61% | 89.92% |
| phoneme | 85.90% | 86.10% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 84.97% | 81.72% | 81.42% | 85.33% | 85.27% | 85.90% | 85.51% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 85.25% |
| qsar-biodeg | 84.45% | 83.89% | 85.88% | 86.16% | 85.40% | 85.21% | 80.95% | 85.97% | 83.51% | 82.94% | 85.88% | 85.40% | 84.45% | 83.89% | 85.88% | 86.16% | 85.31% | 85.21% | 80.95% | 85.88% |
| sick | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% | 98.52% | 98.36% | 98.67% | 98.52% | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% |
| sonar | 76.02% | 75.48% | 84.14% | 82.69% | 85.10% | 86.12% | 78.83% | 81.29% | 74.98% | 79.86% | 82.71% | 81.74% | 76.02% | 76.43% | 84.14% | 82.69% | 85.10% | 85.62% | 78.83% | 81.29% |
| spambase | 93.76% | 94.09% | 95.28% | 95.09% | 95.00% | 95.52% | 92.20% | 95.39% | 92.70% | 92.89% | 95.26% | 95.35% | 93.76% | 94.09% | 95.31% | 95.11% | 94.96% | 95.54% | 92.20% | 95.39% |
| tic-tac-toe | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 96.97% | 84.45% | 85.07% | 99.06% | 99.06% | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 97.60% |
| vote | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% | 94.95% | 95.64% | 96.09% | 95.86% | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% |
| wdbc | 95.78% | 95.08% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.31% | 95.42% | 95.60% | 96.66% | 96.48% | 95.78% | 95.08% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.83% |
| wilt | 96.86% | 96.65% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.11% | 97.11% | 97.23% | 97.13% | 97.19% | 96.86% | 96.67% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.09% |

Fit time in seconds (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 0.10 | 0.11 | 0.22 | 0.18 | 0.24 | 0.42 | 0.01 | 0.37 | 0.07 | 0.13 | 1.36 | 2.44 | 0.05 | 0.05 | 0.18 | 0.13 | 0.18 | 0.28 | 0.01 | 0.18 |
| Titanic | 0.61 | 0.65 | 0.61 | 0.47 | 0.59 | 1.15 | 0.02 | 0.58 | 0.18 | 0.31 | 1.90 | 3.31 | 0.32 | 0.29 | 0.51 | 0.46 | 0.58 | 0.94 | 0.01 | 0.31 |
| banknote-authentication | 0.31 | 0.10 | 0.42 | 0.31 | 0.34 | 0.85 | 0.01 | 0.53 | 0.15 | 0.25 | 1.85 | 3.39 | 0.04 | 0.04 | 0.36 | 0.26 | 0.34 | 0.58 | 0.01 | 0.27 |
| blood-transfusion-service-center | 0.33 | 0.21 | 0.32 | 0.24 | 0.30 | 0.54 | 0.01 | 0.45 | 0.08 | 0.15 | 1.56 | 2.74 | 0.11 | 0.11 | 0.26 | 0.19 | 0.25 | 0.40 | 0.01 | 0.22 |
| breast-cancer | 0.22 | 0.25 | 0.29 | 0.27 | 0.25 | 0.53 | 0.01 | 0.30 | 0.11 | 0.21 | 1.45 | 2.59 | 0.10 | 0.12 | 0.21 | 0.18 | 0.21 | 0.34 | 0.01 | 0.13 |
| breast-w | 0.11 | 0.13 | 0.65 | 0.59 | 0.48 | 1.26 | 0.02 | 0.54 | 0.57 | 1.19 | 2.39 | 4.24 | 0.05 | 0.05 | 0.53 | 0.46 | 0.44 | 0.88 | 0.01 | 0.25 |
| churn | 10.41 | 13.35 | 4.59 | 3.48 | 9.17 | 24.75 | 0.37 | 5.39 | 4.99 | 9.21 | 14.18 | 24.41 | 5.85 | 6.16 | 5.78 | 5.58 | 9.79 | 8.81 | 0.15 | 10.01 |
| climate-model-simulation-crashes | 0.13 | 0.13 | 0.53 | 0.45 | 0.52 | 1.55 | 0.03 | 0.61 | 0.78 | 1.58 | 2.85 | 5.37 | 0.07 | 0.07 | 0.45 | 0.37 | 0.50 | 1.11 | 0.01 | 0.26 |
| colic | 0.22 | 0.25 | 0.38 | 0.32 | 0.36 | 1.05 | 0.02 | 0.61 | 0.28 | 0.55 | 2.29 | 4.16 | 0.10 | 0.11 | 0.31 | 0.25 | 0.33 | 0.65 | 0.01 | 0.25 |
| compas-two-years | 6.06 | 7.35 | 1.75 | 1.16 | 2.08 | 5.35 | 0.12 | 2.36 | 0.55 | 1.04 | 6.05 | 10.22 | 4.23 | 4.34 | 1.87 | 1.35 | 1.77 | 2.56 | 0.05 | 3.88 |
| credit-approval | 0.44 | 0.51 | 0.60 | 0.53 | 0.56 | 1.32 | 0.02 | 0.62 | 0.41 | 0.82 | 2.48 | 4.34 | 0.22 | 0.23 | 0.48 | 0.46 | 0.54 | 1.12 | 0.01 | 0.31 |
| credit-g | 1.00 | 1.23 | 0.68 | 0.53 | 0.60 | 1.88 | 0.03 | 0.72 | 0.34 | 0.68 | 2.44 | 4.49 | 0.35 | 0.38 | 0.61 | 0.52 | 0.70 | 1.45 | 0.02 | 1.74 |
| cylinder-bands | 0.74 | 1.04 | 1.12 | 0.97 | 0.90 | 8.09 | 0.06 | 1.31 | 3.14 | 6.24 | 5.49 | 10.02 | 0.22 | 0.25 | 1.03 | 0.88 | 1.00 | 5.32 | 0.04 | 1.82 |
| diabetes | 1.18 | 1.41 | 0.75 | 0.58 | 0.67 | 1.65 | 0.02 | 0.83 | 0.26 | 0.54 | 2.92 | 5.11 | 0.55 | 0.60 | 0.62 | 0.50 | 0.63 | 1.37 | 0.01 | 0.41 |
| dresses-sales | 1.33 | 1.64 | 0.78 | 0.66 | 0.59 | 1.86 | 0.04 | 0.87 | 0.90 | 1.86 | 3.68 | 6.57 | 0.50 | 0.52 | 0.66 | 0.58 | 0.61 | 1.25 | 0.02 | 1.20 |
| heart-c | 0.14 | 0.17 | 0.36 | 0.29 | 0.31 | 0.71 | 0.01 | 0.41 | 0.19 | 0.37 | 1.88 | 3.34 | 0.07 | 0.07 | 0.23 | 0.21 | 0.25 | 0.46 | 0.01 | 0.17 |
| heart-h | 0.17 | 0.19 | 0.33 | 0.28 | 0.28 | 0.56 | 0.00 | 0.36 | 0.19 | 0.37 | 1.75 | 3.11 | 0.08 | 0.08 | 0.22 | 0.18 | 0.22 | 0.36 | 0.01 | 0.16 |
| heart-statlog | 0.13 | 0.13 | 0.34 | 0.28 | 0.28 | 0.61 | 0.01 | 0.40 | 0.19 | 0.36 | 1.79 | 3.28 | 0.06 | 0.06 | 0.22 | 0.19 | 0.22 | 0.42 | 0.00 | 0.17 |
| heloc | 48.69 | 104.74 | 10.77 | 7.97 | 14.00 | 34.03 | 0.90 | 13.58 | 9.61 | 17.32 | 35.65 | 59.51 | 16.56 | 21.82 | 11.73 | 9.16 | 15.00 | 26.31 | 0.61 | 28.33 |
| hepatitis | 0.05 | 0.05 | 0.18 | 0.22 | 0.19 | 0.45 | 0.01 | 0.30 | 0.12 | 0.28 | 1.63 | 2.91 | 0.03 | 0.02 | 0.13 | 0.13 | 0.14 | 0.29 | 0.00 | 0.12 |
| ilpd | 0.56 | 0.68 | 0.47 | 0.37 | 0.42 | 0.90 | 0.02 | 0.52 | 0.24 | 0.50 | 2.06 | 3.64 | 0.27 | 0.30 | 0.38 | 0.31 | 0.39 | 0.74 | 0.01 | 0.23 |
| ionosphere | 0.16 | 0.18 | 0.48 | 0.44 | 0.56 | 1.78 | 0.03 | 0.78 | 1.73 | 3.71 | 4.24 | 7.88 | 0.04 | 0.05 | 0.37 | 0.33 | 0.45 | 0.96 | 0.01 | 1.01 |
| kc1 | 7.03 | 8.49 | 2.46 | 1.98 | 3.34 | 17.75 | 0.32 | 4.08 | 2.95 | 5.75 | 12.13 | 20.86 | 2.62 | 2.60 | 2.08 | 1.66 | 2.75 | 4.12 | 0.09 | 8.25 |
| kc2 | 0.42 | 0.48 | 0.50 | 0.42 | 0.53 | 1.65 | 0.03 | 0.76 | 0.85 | 1.77 | 3.29 | 5.96 | 0.17 | 0.14 | 0.41 | 0.34 | 0.50 | 1.04 | 0.01 | 1.15 |
| kr-vs-kp | 0.41 | 0.53 | 1.14 | 0.91 | 0.87 | 2.82 | 0.04 | 1.29 | 0.33 | 0.64 | 3.36 | 5.62 | 0.12 | 0.12 | 1.19 | 0.94 | 1.02 | 1.83 | 0.02 | 2.09 |
| mofn-3-7-10 | 0.23 | 0.28 | 0.29 | 0.18 | 0.27 | 0.54 | 0.01 | 0.51 | 0.04 | 0.06 | 1.34 | 2.27 | 0.10 | 0.13 | 0.24 | 0.14 | 0.25 | 0.36 | 0.01 | 0.28 |
| molecular-biology_promoters | 0.04 | 0.04 | 0.23 | 0.26 | 0.29 | 0.79 | 0.01 | 0.38 | 0.62 | 1.22 | 2.97 | 5.38 | 0.02 | 0.02 | 0.17 | 0.14 | 0.18 | 0.38 | 0.01 | 0.14 |
| monks-problems-1 | 0.05 | 0.06 | 0.23 | 0.19 | 0.22 | 0.34 | 0.01 | 0.25 | 0.03 | 0.05 | 1.26 | 2.30 | 0.02 | 0.02 | 0.19 | 0.12 | 0.19 | 0.22 | 0.00 | 0.14 |
| monks-problems-2 | 0.22 | 0.21 | 0.28 | 0.23 | 0.33 | 0.52 | 0.01 | 0.41 | 0.03 | 0.07 | 1.36 | 2.41 | 0.09 | 0.11 | 0.22 | 0.17 | 0.27 | 0.35 | 0.00 | 0.19 |
| monks-problems-3 | 0.03 | 0.04 | 0.25 | 0.21 | 0.24 | 0.40 | 0.01 | 0.41 | 0.04 | 0.08 | 1.31 | 2.28 | 0.02 | 0.02 | 0.20 | 0.14 | 0.19 | 0.29 | 0.00 | 0.18 |
| mushroom | 0.86 | 1.02 | 4.35 | 4.47 | 7.26 | 20.74 | 0.32 | 7.05 | 9.60 | 22.55 | 20.63 | 35.30 | 0.13 | 0.12 | 4.25 | 3.96 | 4.38 | 7.27 | 0.18 | 8.33 |
| ozone-level-8hr | 4.06 | 5.55 | 9.10 | 8.99 | 9.59 | 39.97 | 1.20 | 10.49 | 51.53 | 97.08 | 33.41 | 62.11 | 1.43 | 1.19 | 8.07 | 6.74 | 7.66 | 22.00 | 0.48 | 16.79 |
| pc1 | 0.81 | 0.92 | 0.81 | 0.70 | 0.99 | 3.05 | 0.05 | 1.36 | 1.57 | 3.26 | 4.57 | 8.26 | 0.30 | 0.28 | 0.73 | 0.60 | 0.93 | 1.72 | 0.03 | 2.28 |
| pc3 | 2.74 | 3.60 | 2.28 | 1.92 | 3.15 | 18.22 | 0.42 | 5.11 | 14.93 | 29.73 | 16.93 | 29.00 | 1.01 | 0.94 | 1.51 | 1.25 | 1.57 | 7.10 | 0.11 | 8.68 |
| pc4 | 1.71 | 2.20 | 1.88 | 1.61 | 2.85 | 10.48 | 0.18 | 3.44 | 10.07 | 19.01 | 11.27 | 19.75 | 0.62 | 0.56 | 1.33 | 1.12 | 1.57 | 3.18 | 0.08 | 6.33 |
| phoneme | 7.41 | 9.28 | 1.86 | 1.30 | 2.12 | 5.37 | 0.09 | 2.33 | 0.40 | 0.74 | 5.09 | 8.32 | 3.89 | 3.85 | 1.88 | 1.28 | 1.87 | 3.34 | 0.04 | 3.56 |
| qsar-biodeg | 1.19 | 1.78 | 1.36 | 1.18 | 1.51 | 6.13 | 0.12 | 2.15 | 3.03 | 6.09 | 7.50 | 13.10 | 0.40 | 0.47 | 1.13 | 0.92 | 1.01 | 2.29 | 0.05 | 4.04 |
| sick | 0.83 | 1.07 | 1.81 | 1.46 | 2.14 | 6.27 | 0.13 | 2.40 | 1.82 | 3.56 | 6.17 | 10.47 | 0.30 | 0.30 | 1.81 | 1.34 | 1.54 | 2.65 | 0.06 | 3.64 |
| sonar | 0.14 | 0.16 | 0.68 | 0.63 | 0.67 | 2.84 | 0.05 | 0.95 | 4.87 | 10.59 | 7.05 | 13.60 | 0.06 | 0.06 | 0.47 | 0.43 | 0.52 | 1.53 | 0.02 | 1.05 |
| spambase | 15.78 | 25.33 | 15.63 | 14.79 | 15.71 | 40.64 | 1.22 | 15.70 | 72.01 | 117.90 | 45.57 | 75.66 | 8.60 | 5.26 | 14.34 | 12.41 | 11.29 | 22.75 | 0.70 | 29.86 |
| tic-tac-toe | 0.15 | 0.14 | 0.49 | 0.39 | 0.45 | 0.90 | 0.02 | 0.63 | 0.08 | 0.14 | 1.87 | 3.33 | 0.06 | 0.06 | 0.40 | 0.32 | 0.41 | 0.67 | 0.01 | 0.31 |
| vote | 0.04 | 0.05 | 0.19 | 0.17 | 0.21 | 0.44 | 0.01 | 0.35 | 0.06 | 0.09 | 1.33 | 2.28 | 0.02 | 0.02 | 0.17 | 0.13 | 0.17 | 0.29 | 0.00 | 0.18 |
| wdbc | 0.11 | 0.17 | 0.73 | 0.65 | 0.89 | 3.22 | 0.05 | 1.13 | 2.39 | 4.91 | 5.05 | 8.92 | 0.03 | 0.04 | 0.60 | 0.53 | 0.74 | 1.68 | 0.02 | 1.74 |
| wilt | 2.43 | 1.34 | 1.57 | 1.01 | 1.76 | 4.11 | 0.08 | 2.30 | 0.53 | 0.99 | 5.04 | 8.33 | 1.88 | 0.75 | 1.56 | 0.93 | 1.35 | 2.05 | 0.04 | 3.00 |

