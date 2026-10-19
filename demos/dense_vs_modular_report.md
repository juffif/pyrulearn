# Dense and modular rule learners compared

## Setup

**Full run** (25 small and 19 medium binary datasets). Quick sanity check instead: `python demos/dense_vs_modular.py` with no arguments.

The same rule-learning algorithms written twice: specialized for speed on a dense
matrix (`Dense...`), and built from the library's exchangeable components -- a
`RuleSearch`, an `Objective`, a covering or boosting loop -- so that they run on every
data representation and take any search. Each dense learner is paired with its
modular twin on the same data -- the dense one on the Boolean matrix it is written for, the
modular one on vertical bitmaps, the library's default representation; the modular learners are then
run again on the horizontal bitmap; and the modular `ORB` and `LRI` are run
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
isn't in the main comparison". On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 93.49% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 26.2s per fit against 0.18s and 0.01s -- 147x as long as the beam search; 3 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

Protocol: 10-fold stratified cross-validation (5-fold for the medium datasets)
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals=8)`), the test fold encoded against that same
`DataSpec` -- every learner sees the identical binary features per fold, the dense learners as a
Boolean matrix and the modular ones as vertical bitmaps; the `[HBitmap]` rows are the modular learners on
the same data as horizontal bitmaps. Each fit is capped at
300s; a time-out or an error counts as a failure (and as last in the ranking), not as the end
of the run. Fit times are wall-clock seconds of one fit in a worker process (see below for
how many ran at once).

A fold that times out fails the whole cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are not run, and all its folds count as failures, so no mean rests on just the folds that finished.

Measures: test accuracy, number of rules and conditions, fit time, and whether a dense
learner and its modular twin learned the same model (the same rules, whatever their order).

Every learner runs single-threaded. Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

![fit time, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_fit_time.png)

![accuracy, modular vs. dense](dense_vs_modular_plots/dense_vs_modular_accuracy.png)

![critical-difference diagram (accuracy)](dense_vs_modular_plots/dense_vs_modular_cd.png)

## Why exhaustive ORB isn't in the main comparison

On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 93.49% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 26.2s per fit against 0.18s and 0.01s -- 147x as long as the beam search; 3 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

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
| breast-cancer | 0.01 | 0.19 | 0.33 | 0.01 |  |  |
| mofn-3-7-10 | 0.01 | 0.07 | 0.12 | 0.01 | 0.27 | 0.58 |
| monks-problems-3 | 0.01 | 0.07 | 0.10 | 0.00 | 0.40 | 0.78 |
| tic-tac-toe | 0.01 | 0.11 | 0.21 | 0.01 | 60.44 | 101.67 |
| vote | 0.01 | 0.10 | 0.13 | 0.00 | 1.67 | 1.87 |

## Dense vs. modular

For each pair, over the folds where both fits finished: how often the two learned the same model (the same rules in any order), their mean accuracy, and the ratio of the modular to the dense fit time -- the geometric mean over datasets, so a value below 1 means the modular learner is faster. The dense learner on the Boolean matrix, the modular one on vertical bitmaps (the default); the modular learners on the horizontal bitmap are compared with them below.

| pair | identical models | accuracy dense | accuracy modular | difference (points) | time modular / dense, small | medium | all |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPAR | 345/345 | 85.63% | 85.63% | +0.00 | 1.56 | 1.36 | 1.47 |
| CPAR PRM | 266/345 | 85.57% | 85.42% | -0.15 | 1.61 | 1.51 | 1.57 |
| ENDER | 341/345 | 85.83% | 85.84% | +0.01 | 1.23 | 0.80 | 1.02 |
| ENDER CS-Exp | 335/345 | 86.04% | 86.03% | -0.01 | 1.25 | 0.80 | 1.03 |
| ENDER MLRules | 322/345 | 86.19% | 86.19% | -0.00 | 1.09 | 0.63 | 0.86 |
| Boomer | 328/345 | 86.43% | 86.41% | -0.02 | 1.18 | 0.65 | 0.91 |
| ORB | 325/345 | 83.50% | 83.52% | +0.02 | 1.97 | 1.33 | 1.66 |
| LRI | 48/345 | 85.03% | 85.04% | +0.01 | 1.77 | 1.07 | 1.42 |

## Summary by learner (mean across the 44 datasets and their folds)

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ENDER MLRules | 86.19% | 345.9 | 1432.1 | 4.0 | 1.02 | 2.6 | 7.56 | 0 |
| DenseENDER CS-Exp | 86.04% | 396.4 | 1750.1 | 4.2 | 1.34 | 4.1 | 7.81 | 0 |
| ENDER | 85.84% | 406.6 | 1906.8 | 4.4 | 1.22 | 2.3 | 7.81 | 0 |
| ENDER CS-Exp | 86.03% | 396.4 | 1749.8 | 4.2 | 0.96 | 3.1 | 7.88 | 0 |
| DenseENDER | 85.83% | 406.6 | 1906.8 | 4.4 | 1.71 | 2.3 | 7.92 | 0 |
| DenseENDER MLRules | 86.19% | 346.0 | 1432.3 | 4.0 | 1.85 | 1.2 | 8.15 | 0 |
| LRI(beam 5) | 86.13% | 392.5 | 1365.8 | 3.5 | 4.40 | 4.2 | 8.30 | 0 |
| LRI | 85.04% | 396.3 | 1579.0 | 4.0 | 1.02 | 1.1 | 8.90 | 0 |
| DenseLRI | 85.03% | 396.4 | 1561.4 | 3.9 | 0.99 | 4.0 | 9.01 | 0 |
| LRI(beam 10) | 86.18% | 392.1 | 1405.0 | 3.6 | 8.06 | 3.7 | 9.07 | 0 |
| DenseBoomer | 86.43% | 272.9 | 1792.1 | 5.4 | 3.65 | 4.6 | 10.38 | 0 |
| Boomer | 86.41% | 272.9 | 1792.0 | 5.4 | 2.18 | 3.6 | 10.39 | 0 |
| CPAR | 85.63% | 373.5 | 2431.9 | 4.6 | 1.28 | 1.2 | 11.84 | 0 |
| DenseCPAR | 85.63% | 373.5 | 2431.9 | 4.6 | 1.16 | 1.2 | 11.84 | 0 |
| DenseCPAR PRM | 85.57% | 309.1 | 1941.8 | 4.6 | 1.13 | 1.7 | 12.73 | 0 |
| CPAR PRM | 85.42% | 308.0 | 1941.2 | 4.6 | 1.47 | 0.7 | 13.38 | 0 |
| ORB(beam 5) | 84.04% | 9.8 | 48.5 | 4.9 | 1.63 | 0.2 | 14.01 | 0 |
| ORB(beam 10) | 84.24% | 9.8 | 51.1 | 5.2 | 3.40 | 1.2 | 14.24 | 0 |
| DenseORB(greedy) | 83.50% | 9.7 | 39.6 | 4.0 | 0.03 | 0.1 | 14.40 | 0 |
| ORB(greedy) | 83.52% | 9.7 | 39.6 | 4.0 | 0.04 | 1.1 | 14.42 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised. A dense learner and its twin that learn the same models tie for rank, so the ranking separates the algorithms and the searches, not dense from modular.

### Small datasets only (25, fewer than 1,000 examples)

`SPECT`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `climate-model-simulation-crashes`, `colic`, `credit-approval`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `ilpd`, `ionosphere`, `kc2`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `sonar`, `tic-tac-toe`, `vote`, `wdbc`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DenseENDER CS-Exp | 84.75% | 396.9 | 1549.1 | 3.7 | 0.45 | 2.6 | 7.28 | 0 |
| ENDER CS-Exp | 84.73% | 396.8 | 1548.6 | 3.7 | 0.54 | 1.6 | 7.52 | 0 |
| ENDER | 84.46% | 406.1 | 1660.6 | 3.9 | 0.64 | 1.2 | 8.06 | 0 |
| ENDER MLRules | 84.93% | 342.9 | 1277.6 | 3.7 | 0.56 | 0.6 | 8.14 | 0 |
| DenseENDER MLRules | 84.95% | 343.1 | 1278.6 | 3.7 | 0.54 | 0.6 | 8.18 | 0 |
| DenseENDER | 84.45% | 406.0 | 1660.5 | 3.9 | 0.54 | 1.2 | 8.18 | 0 |
| DenseLRI | 83.66% | 396.3 | 1530.7 | 3.9 | 0.37 | 2.9 | 9.66 | 0 |
| LRI(beam 5) | 84.86% | 391.6 | 1328.7 | 3.4 | 3.45 | 1.1 | 9.78 | 0 |
| LRI | 83.68% | 396.3 | 1551.3 | 3.9 | 0.65 | 1.1 | 9.88 | 0 |
| LRI(beam 10) | 84.94% | 391.6 | 1370.5 | 3.5 | 6.38 | 1.6 | 10.36 | 0 |
| DenseBoomer | 85.42% | 229.2 | 1483.0 | 5.3 | 1.23 | 2.9 | 10.74 | 0 |
| Boomer | 85.39% | 229.2 | 1484.0 | 5.3 | 1.28 | 1.9 | 10.84 | 0 |
| CPAR | 84.42% | 180.7 | 856.5 | 4.2 | 0.31 | 0.7 | 11.26 | 0 |
| DenseCPAR | 84.42% | 180.7 | 856.5 | 4.2 | 0.20 | 0.7 | 11.26 | 0 |
| DenseCPAR PRM | 84.38% | 165.0 | 778.2 | 4.1 | 0.21 | 1.7 | 12.04 | 0 |
| CPAR PRM | 84.19% | 164.3 | 773.5 | 4.1 | 0.33 | 0.7 | 12.78 | 0 |
| ORB(greedy) | 82.07% | 9.7 | 36.5 | 3.7 | 0.02 | 1.1 | 13.18 | 0 |
| DenseORB(greedy) | 82.05% | 9.7 | 36.5 | 3.7 | 0.01 | 0.1 | 13.22 | 0 |
| ORB(beam 10) | 82.84% | 9.8 | 48.7 | 4.9 | 1.86 | 1.2 | 13.54 | 0 |
| ORB(beam 5) | 82.53% | 9.8 | 46.4 | 4.7 | 0.93 | 0.2 | 14.10 | 0 |

### Medium datasets only (19, 1,000 to 10,000 examples)

`Titanic`, `banknote-authentication`, `churn`, `compas-two-years`, `credit-g`, `heloc`, `kc1`, `kr-vs-kp`, `mofn-3-7-10`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `spambase`, `wilt`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LRI(beam 5) | 89.50% | 394.8 | 1463.4 | 3.7 | 6.92 | 3.2 | 6.34 | 0 |
| ENDER MLRules | 89.50% | 353.8 | 1838.6 | 4.9 | 2.22 | 2.0 | 6.79 | 0 |
| LRI(beam 10) | 89.44% | 393.4 | 1495.8 | 3.8 | 12.48 | 2.2 | 7.37 | 0 |
| ENDER | 89.48% | 408.1 | 2554.8 | 5.8 | 2.74 | 1.2 | 7.47 | 0 |
| DenseENDER | 89.47% | 408.1 | 2554.9 | 5.8 | 4.80 | 1.2 | 7.58 | 0 |
| LRI | 88.61% | 396.6 | 1652.0 | 4.2 | 2.01 | 0.1 | 7.61 | 0 |
| DenseENDER MLRules | 89.44% | 353.5 | 1836.6 | 4.9 | 5.31 | 0.7 | 8.11 | 0 |
| DenseLRI | 88.61% | 396.6 | 1642.3 | 4.1 | 2.60 | 1.1 | 8.16 | 0 |
| ENDER CS-Exp | 89.44% | 395.3 | 2279.3 | 5.4 | 2.06 | 1.5 | 8.34 | 0 |
| DenseENDER CS-Exp | 89.43% | 395.3 | 2279.2 | 5.4 | 3.70 | 1.5 | 8.50 | 0 |
| Boomer | 89.08% | 388.0 | 2602.7 | 5.8 | 4.56 | 1.7 | 9.79 | 0 |
| DenseBoomer | 89.08% | 388.2 | 2605.8 | 5.8 | 10.02 | 1.7 | 9.89 | 0 |
| CPAR | 88.80% | 880.9 | 6577.5 | 5.9 | 3.83 | 0.5 | 12.61 | 0 |
| DenseCPAR | 88.80% | 880.9 | 6577.5 | 5.9 | 3.71 | 0.5 | 12.61 | 0 |
| DenseCPAR PRM | 88.69% | 688.3 | 5003.7 | 5.8 | 3.58 | 0.0 | 13.63 | 0 |
| ORB(beam 5) | 88.02% | 9.7 | 54.3 | 5.5 | 3.45 | 0.0 | 13.89 | 0 |
| CPAR PRM | 88.63% | 686.2 | 5013.9 | 5.8 | 4.47 | 0.0 | 14.16 | 0 |
| ORB(beam 10) | 87.92% | 9.7 | 57.4 | 5.9 | 7.43 | 0.0 | 15.16 | 0 |
| DenseORB(greedy) | 87.32% | 9.7 | 47.5 | 4.9 | 0.08 | 0.0 | 15.95 | 0 |
| ORB(greedy) | 87.31% | 9.7 | 47.7 | 4.9 | 0.09 | 0.0 | 16.05 | 0 |

## Beam search for ORB and LRI

The dense `ORB` and `LRI` come with their own greedy (and, for `ORB`, exhaustive) search; the modular ones take any `RuleSearch`. Mean over all datasets:

| learner | accuracy | n_conditions | fit_time (s) |
|---|--:|--:|--:|
| ORB(greedy) | 83.52% | 39.6 | 0.04 |
| ORB(beam 5) | 84.04% | 48.5 | 1.63 |
| ORB(beam 10) | 84.24% | 51.1 | 3.40 |
| LRI | 85.04% | 1579.0 | 1.02 |
| LRI(beam 5) | 86.13% | 1365.8 | 4.40 |
| LRI(beam 10) | 86.18% | 1405.0 | 8.06 |

## The modular learners on the horizontal bitmap

The modular learners above run on `VerticalBitmapRepresentation`, the library's default. Here they are run again on `HorizontalBitmapRepresentation`, the matrix the dense learners work on, over the folds where both finished; a time ratio above 1 means the vertical bitmap is faster:

| learner | accuracy, VBitmap | accuracy, HBitmap | time HBitmap / VBitmap (geometric mean over datasets) |
|---|--:|--:|--:|
| CPAR | 85.63% | 85.63% | 1.25 |
| CPAR PRM | 85.42% | 85.54% | 1.32 |
| ENDER | 85.84% | 85.84% | 1.11 |
| ENDER CS-Exp | 86.03% | 86.03% | 1.13 |
| ENDER MLRules | 86.19% | 86.19% | 1.18 |
| Boomer | 86.41% | 86.41% | 1.37 |
| ORB(greedy) | 83.52% | 83.52% | 1.42 |
| LRI | 85.04% | 85.04% | 1.29 |
| ORB(beam 5) | 84.04% | 84.04% | 1.17 |
| ORB(beam 10) | 84.24% | 84.22% | 1.18 |
| LRI(beam 5) | 86.13% | 86.13% | 1.18 |
| LRI(beam 10) | 86.18% | 86.18% | 1.15 |

## Per-dataset results

Accuracy (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 82.78% | 81.31% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% | 82.79% | 82.44% | 83.82% | 83.83% | 82.78% | 82.42% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% |
| Titanic | 78.84% | 78.76% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.07% | 79.68% | 79.45% | 79.68% | 79.14% | 78.84% | 78.76% | 80.98% | 80.75% | 79.45% | 77.77% | 79.45% | 79.68% |
| banknote-authentication | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% | 98.18% | 97.67% | 99.05% | 99.05% | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% |
| blood-transfusion-service-center | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 77.54% | 76.20% | 76.86% | 77.53% | 77.40% | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 78.20% |
| breast-cancer | 73.42% | 72.72% | 72.39% | 73.09% | 71.69% | 68.94% | 70.65% | 70.64% | 69.58% | 73.07% | 71.69% | 72.77% | 73.42% | 73.76% | 72.39% | 73.09% | 72.03% | 69.66% | 70.65% | 71.33% |
| breast-w | 93.42% | 93.84% | 96.57% | 96.42% | 95.57% | 95.99% | 95.71% | 96.28% | 94.85% | 95.13% | 96.00% | 95.57% | 93.42% | 93.56% | 96.28% | 96.57% | 95.57% | 95.99% | 95.71% | 96.00% |
| churn | 93.02% | 93.38% | 92.58% | 92.02% | 93.38% | 93.46% | 89.56% | 93.24% | 91.08% | 90.92% | 93.28% | 93.64% | 93.02% | 93.42% | 92.58% | 92.02% | 93.36% | 93.38% | 89.56% | 93.20% |
| climate-model-simulation-crashes | 93.52% | 93.70% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 93.89% | 92.78% | 92.78% | 93.89% | 93.70% | 93.52% | 93.89% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 94.26% |
| colic | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.07% | 84.26% | 82.36% | 84.81% | 85.35% | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.35% |
| compas-two-years | 65.88% | 65.95% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.75% | 67.22% | 67.30% | 67.18% | 67.47% | 65.88% | 66.62% | 68.04% | 68.25% | 67.94% | 67.05% | 66.88% | 67.89% |
| credit-approval | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 87.25% | 87.25% | 86.67% | 86.52% | 87.39% | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 86.96% |
| credit-g | 74.00% | 73.20% | 76.50% | 76.80% | 76.80% | 74.70% | 70.20% | 75.30% | 71.40% | 71.50% | 76.00% | 76.30% | 74.00% | 73.20% | 76.50% | 76.80% | 76.30% | 74.70% | 70.30% | 75.50% |
| cylinder-bands | 77.04% | 78.89% | 80.93% | 80.93% | 81.11% | 87.04% | 71.48% | 82.22% | 74.81% | 74.44% | 82.96% | 81.85% | 77.04% | 79.44% | 80.93% | 81.11% | 81.11% | 87.22% | 71.48% | 82.22% |
| diabetes | 76.03% | 76.42% | 74.74% | 74.48% | 74.87% | 74.09% | 74.87% | 75.39% | 74.35% | 74.48% | 76.04% | 75.78% | 76.03% | 76.42% | 74.74% | 74.48% | 74.61% | 74.09% | 74.87% | 75.53% |
| dresses-sales | 57.00% | 57.40% | 62.20% | 63.20% | 58.80% | 58.40% | 58.20% | 58.80% | 54.80% | 54.80% | 58.20% | 59.40% | 57.00% | 56.80% | 62.20% | 63.20% | 59.40% | 58.40% | 58.20% | 58.80% |
| heart-c | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 78.27% | 80.89% | 77.87% | 78.90% | 80.24% | 80.55% | 77.59% | 82.88% | 81.89% | 81.24% | 76.63% | 81.23% | 80.25% |
| heart-h | 80.98% | 80.60% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.64% | 81.26% | 80.59% | 80.61% | 79.95% | 80.98% | 80.60% | 81.63% | 81.62% | 81.31% | 80.99% | 81.24% | 81.31% |
| heart-statlog | 82.22% | 80.00% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 76.30% | 79.26% | 80.37% | 76.67% | 78.15% | 82.22% | 81.11% | 81.48% | 81.11% | 79.63% | 77.78% | 79.26% | 75.56% |
| heloc | 70.23% | 70.29% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.88% | 70.25% | 70.03% | 71.72% | 71.72% | 70.23% | 70.42% | 72.19% | 71.79% | 72.06% | 70.05% | 70.35% | 71.55% |
| hepatitis | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 84.50% | 82.12% | 80.21% | 78.88% | 82.12% | 79.50% | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 83.87% | 80.88% |
| ilpd | 71.17% | 69.46% | 69.98% | 71.02% | 70.49% | 69.12% | 70.15% | 71.68% | 70.67% | 71.86% | 70.13% | 70.13% | 71.17% | 69.81% | 69.98% | 71.02% | 70.49% | 69.29% | 70.15% | 70.99% |
| ionosphere | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% | 91.72% | 91.73% | 93.16% | 92.60% | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% |
| kc1 | 84.21% | 83.93% | 85.63% | 85.59% | 84.87% | 84.12% | 85.02% | 85.44% | 84.97% | 85.16% | 85.63% | 85.40% | 84.21% | 83.88% | 85.63% | 85.59% | 84.87% | 83.97% | 85.02% | 85.30% |
| kc2 | 82.94% | 83.70% | 84.26% | 84.64% | 82.93% | 81.98% | 83.12% | 82.36% | 81.78% | 81.21% | 82.36% | 81.40% | 82.94% | 83.89% | 84.26% | 84.64% | 82.93% | 81.79% | 83.12% | 81.79% |
| kr-vs-kp | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% | 97.47% | 97.22% | 99.00% | 99.09% | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% |
| mofn-3-7-10 | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% | 94.64% | 95.54% | 100.00% | 100.00% | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% |
| molecular-biology_promoters | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 82.91% | 81.27% | 82.27% | 80.27% | 82.27% | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 77.55% |
| monks-problems-1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% |
| monks-problems-2 | 96.17% | 97.34% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 67.21% | 75.70% | 77.53% | 93.35% | 95.35% | 96.17% | 97.50% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 71.39% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 98.20% | 98.92% | 98.92% | 97.84% | 97.84% | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 97.84% |
| mushroom | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% | 99.48% | 99.84% | 100.00% | 100.00% | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% |
| ozone-level-8hr | 93.73% | 93.73% | 94.00% | 93.96% | 94.32% | 94.28% | 93.01% | 94.24% | 92.98% | 92.94% | 94.28% | 93.96% | 93.73% | 93.80% | 94.00% | 93.96% | 94.20% | 94.28% | 93.09% | 94.20% |
| pc1 | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.06% | 92.16% | 93.06% | 92.97% | 92.16% | 92.88% | 93.06% | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.15% | 92.16% | 93.15% |
| pc3 | 89.63% | 88.87% | 88.80% | 89.44% | 89.12% | 88.42% | 89.38% | 89.38% | 89.19% | 89.12% | 89.31% | 88.74% | 89.63% | 88.87% | 88.74% | 89.38% | 88.99% | 88.48% | 89.38% | 89.06% |
| pc4 | 89.99% | 89.57% | 89.78% | 89.71% | 90.12% | 89.64% | 88.61% | 90.12% | 89.30% | 88.89% | 90.19% | 90.12% | 89.99% | 89.57% | 89.78% | 89.71% | 89.71% | 89.64% | 88.61% | 89.92% |
| phoneme | 85.90% | 85.38% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 84.97% | 81.72% | 81.42% | 85.33% | 85.27% | 85.90% | 85.51% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 85.25% |
| qsar-biodeg | 84.45% | 83.89% | 85.88% | 86.16% | 85.40% | 85.21% | 80.95% | 85.97% | 83.51% | 82.94% | 85.88% | 85.40% | 84.45% | 83.89% | 85.88% | 86.16% | 85.31% | 85.21% | 80.95% | 85.88% |
| sick | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% | 98.52% | 98.36% | 98.67% | 98.52% | 98.17% | 98.17% | 98.33% | 98.06% | 98.52% | 98.52% | 98.12% | 98.59% |
| sonar | 76.02% | 75.48% | 84.14% | 82.69% | 85.10% | 86.12% | 78.83% | 81.29% | 74.98% | 80.33% | 82.71% | 81.74% | 76.02% | 76.43% | 84.14% | 82.69% | 85.10% | 86.12% | 78.83% | 81.29% |
| spambase | 93.76% | 94.13% | 95.28% | 95.09% | 95.00% | 95.52% | 92.20% | 95.39% | 92.70% | 92.87% | 95.26% | 95.35% | 93.76% | 94.18% | 95.28% | 95.09% | 95.00% | 95.52% | 92.20% | 95.39% |
| tic-tac-toe | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 96.97% | 84.45% | 85.07% | 99.06% | 99.06% | 98.33% | 98.33% | 90.29% | 95.29% | 99.38% | 100.00% | 83.82% | 97.60% |
| vote | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% | 94.95% | 95.64% | 96.09% | 95.86% | 95.41% | 95.41% | 95.87% | 95.64% | 95.63% | 95.63% | 94.95% | 95.41% |
| wdbc | 95.78% | 95.08% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.31% | 95.42% | 95.60% | 96.66% | 96.48% | 95.78% | 95.08% | 96.66% | 96.66% | 96.66% | 96.83% | 94.73% | 96.83% |
| wilt | 96.86% | 96.67% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.11% | 97.11% | 97.09% | 97.13% | 97.19% | 96.86% | 96.67% | 97.15% | 96.96% | 97.15% | 97.00% | 97.11% | 97.09% |

Fit time in seconds (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 0.16 | 0.17 | 0.35 | 0.31 | 0.34 | 0.63 | 0.01 | 0.54 | 0.12 | 0.24 | 2.22 | 3.80 | 0.10 | 0.11 | 0.26 | 0.21 | 0.26 | 0.41 | 0.01 | 0.29 |
| Titanic | 0.88 | 0.82 | 1.03 | 0.69 | 0.79 | 1.35 | 0.03 | 0.66 | 0.28 | 0.52 | 2.81 | 5.10 | 0.54 | 0.49 | 0.78 | 0.65 | 0.79 | 1.33 | 0.01 | 0.51 |
| banknote-authentication | 0.14 | 0.15 | 0.69 | 0.55 | 0.47 | 0.98 | 0.02 | 0.82 | 0.21 | 0.39 | 2.65 | 4.84 | 0.07 | 0.07 | 0.60 | 0.38 | 0.48 | 0.83 | 0.01 | 0.49 |
| blood-transfusion-service-center | 0.33 | 0.34 | 0.51 | 0.43 | 0.49 | 0.88 | 0.02 | 0.59 | 0.19 | 0.29 | 2.42 | 4.47 | 0.20 | 0.20 | 0.44 | 0.32 | 0.39 | 0.63 | 0.01 | 0.35 |
| breast-cancer | 0.34 | 0.38 | 0.49 | 0.41 | 0.42 | 0.81 | 0.02 | 0.46 | 0.20 | 0.43 | 2.47 | 4.36 | 0.20 | 0.22 | 0.36 | 0.34 | 0.30 | 0.52 | 0.01 | 0.23 |
| breast-w | 0.14 | 0.16 | 0.84 | 0.73 | 0.56 | 1.24 | 0.02 | 0.60 | 0.79 | 1.50 | 3.16 | 5.85 | 0.08 | 0.09 | 0.78 | 0.63 | 0.62 | 1.24 | 0.01 | 0.37 |
| churn | 8.47 | 8.69 | 3.71 | 2.86 | 3.06 | 5.62 | 0.12 | 2.51 | 2.15 | 4.71 | 7.47 | 13.53 | 8.10 | 7.03 | 5.58 | 4.95 | 6.36 | 13.20 | 0.10 | 2.87 |
| climate-model-simulation-crashes | 0.18 | 0.17 | 0.75 | 0.64 | 0.70 | 1.54 | 0.03 | 0.72 | 0.93 | 1.97 | 3.96 | 7.44 | 0.13 | 0.11 | 0.63 | 0.56 | 0.68 | 1.56 | 0.01 | 0.38 |
| colic | 0.26 | 0.25 | 0.54 | 0.48 | 0.49 | 1.08 | 0.02 | 0.76 | 0.39 | 0.80 | 3.22 | 5.93 | 0.16 | 0.16 | 0.44 | 0.39 | 0.48 | 0.96 | 0.01 | 0.39 |
| compas-two-years | 7.83 | 7.86 | 2.72 | 1.87 | 2.29 | 3.46 | 0.07 | 2.01 | 0.62 | 1.25 | 4.76 | 7.96 | 7.20 | 6.78 | 2.98 | 2.26 | 3.23 | 4.68 | 0.05 | 1.62 |
| credit-approval | 0.58 | 0.59 | 0.79 | 0.68 | 0.73 | 1.46 | 0.03 | 0.74 | 0.61 | 1.05 | 3.14 | 5.91 | 0.35 | 0.38 | 0.71 | 0.59 | 0.74 | 1.48 | 0.01 | 0.43 |
| credit-g | 1.18 | 1.14 | 1.17 | 0.96 | 0.91 | 1.97 | 0.04 | 0.93 | 0.64 | 1.00 | 4.01 | 6.57 | 0.67 | 0.77 | 1.03 | 0.81 | 1.04 | 2.23 | 0.03 | 0.64 |
| cylinder-bands | 0.53 | 0.72 | 1.43 | 1.24 | 0.99 | 5.23 | 0.05 | 0.95 | 3.83 | 7.58 | 5.88 | 11.25 | 0.33 | 0.38 | 1.32 | 1.15 | 1.27 | 6.64 | 0.03 | 0.59 |
| diabetes | 1.03 | 1.14 | 0.78 | 0.67 | 0.72 | 1.49 | 0.02 | 0.77 | 0.36 | 0.62 | 2.98 | 5.34 | 0.68 | 0.75 | 0.70 | 0.55 | 0.71 | 1.50 | 0.01 | 0.52 |
| dresses-sales | 1.11 | 1.22 | 0.90 | 0.83 | 0.69 | 1.44 | 0.04 | 0.67 | 1.26 | 2.48 | 3.94 | 7.49 | 0.72 | 0.78 | 0.89 | 0.72 | 0.76 | 1.60 | 0.02 | 0.41 |
| heart-c | 0.21 | 0.23 | 0.52 | 0.44 | 0.46 | 0.99 | 0.02 | 0.58 | 0.30 | 0.58 | 2.70 | 5.15 | 0.14 | 0.14 | 0.41 | 0.32 | 0.40 | 0.74 | 0.01 | 0.28 |
| heart-h | 0.23 | 0.24 | 0.50 | 0.45 | 0.43 | 0.79 | 0.02 | 0.48 | 0.30 | 0.55 | 2.81 | 4.83 | 0.13 | 0.14 | 0.36 | 0.30 | 0.32 | 0.59 | 0.01 | 0.25 |
| heart-statlog | 0.17 | 0.20 | 0.45 | 0.40 | 0.43 | 1.01 | 0.02 | 0.57 | 0.27 | 0.57 | 2.64 | 5.13 | 0.12 | 0.13 | 0.35 | 0.33 | 0.39 | 0.69 | 0.01 | 0.28 |
| heloc | 22.71 | 36.35 | 8.38 | 5.31 | 7.68 | 14.72 | 0.24 | 5.17 | 3.92 | 8.30 | 13.79 | 25.14 | 23.41 | 31.18 | 17.07 | 13.66 | 27.11 | 40.80 | 0.29 | 10.44 |
| hepatitis | 0.08 | 0.09 | 0.34 | 0.29 | 0.30 | 0.71 | 0.01 | 0.44 | 0.19 | 0.37 | 2.71 | 5.02 | 0.06 | 0.06 | 0.26 | 0.20 | 0.23 | 0.48 | 0.01 | 0.23 |
| ilpd | 0.75 | 0.87 | 0.78 | 0.61 | 0.61 | 1.24 | 0.02 | 0.68 | 0.40 | 0.85 | 3.00 | 5.39 | 0.53 | 0.49 | 0.61 | 0.50 | 0.58 | 1.11 | 0.01 | 0.41 |
| ionosphere | 0.12 | 0.14 | 0.63 | 0.57 | 0.61 | 1.34 | 0.02 | 0.78 | 2.22 | 4.58 | 5.27 | 10.04 | 0.11 | 0.13 | 0.69 | 0.61 | 0.84 | 1.66 | 0.02 | 0.54 |
| kc1 | 3.04 | 3.03 | 1.80 | 1.24 | 1.44 | 2.68 | 0.05 | 1.52 | 1.72 | 3.09 | 5.08 | 9.89 | 2.84 | 2.22 | 2.36 | 1.53 | 2.75 | 5.24 | 0.05 | 1.25 |
| kc2 | 0.39 | 0.37 | 0.68 | 0.51 | 0.61 | 1.34 | 0.02 | 0.78 | 1.30 | 2.49 | 4.15 | 7.56 | 0.26 | 0.26 | 0.63 | 0.48 | 0.70 | 1.50 | 0.01 | 0.49 |
| kr-vs-kp | 0.27 | 0.34 | 1.67 | 1.22 | 1.05 | 2.02 | 0.03 | 1.08 | 0.38 | 0.78 | 3.38 | 5.88 | 0.19 | 0.17 | 1.76 | 1.40 | 1.25 | 2.18 | 0.02 | 1.03 |
| mofn-3-7-10 | 0.38 | 0.45 | 0.51 | 0.33 | 0.49 | 0.88 | 0.02 | 0.78 | 0.06 | 0.10 | 2.21 | 4.14 | 0.19 | 0.23 | 0.38 | 0.25 | 0.40 | 0.63 | 0.01 | 0.49 |
| molecular-biology_promoters | 0.05 | 0.06 | 0.43 | 0.39 | 0.40 | 0.96 | 0.02 | 0.51 | 0.84 | 1.71 | 4.10 | 7.91 | 0.03 | 0.03 | 0.27 | 0.25 | 0.30 | 0.59 | 0.01 | 0.21 |
| monks-problems-1 | 0.09 | 0.08 | 0.46 | 0.35 | 0.42 | 0.54 | 0.01 | 0.42 | 0.06 | 0.10 | 2.25 | 4.10 | 0.04 | 0.05 | 0.34 | 0.23 | 0.30 | 0.38 | 0.01 | 0.20 |
| monks-problems-2 | 0.32 | 0.34 | 0.48 | 0.39 | 0.56 | 0.89 | 0.02 | 0.67 | 0.07 | 0.13 | 2.25 | 4.00 | 0.22 | 0.22 | 0.38 | 0.34 | 0.44 | 0.59 | 0.01 | 0.41 |
| monks-problems-3 | 0.06 | 0.06 | 0.42 | 0.34 | 0.41 | 0.66 | 0.01 | 0.51 | 0.06 | 0.11 | 2.08 | 3.62 | 0.04 | 0.03 | 0.29 | 0.23 | 0.32 | 0.51 | 0.01 | 0.29 |
| mushroom | 0.31 | 0.32 | 3.41 | 2.69 | 2.49 | 4.83 | 0.10 | 2.69 | 2.97 | 6.70 | 7.90 | 12.47 | 0.17 | 0.17 | 5.01 | 4.23 | 4.85 | 8.49 | 0.07 | 3.36 |
| ozone-level-8hr | 1.80 | 2.05 | 4.58 | 4.09 | 3.83 | 12.78 | 0.26 | 4.14 | 19.63 | 45.55 | 17.90 | 33.55 | 1.77 | 1.48 | 13.21 | 9.67 | 14.65 | 36.01 | 0.23 | 8.22 |
| pc1 | 0.75 | 0.73 | 0.96 | 0.82 | 0.98 | 1.83 | 0.04 | 0.92 | 1.48 | 3.09 | 4.18 | 7.98 | 0.46 | 0.46 | 1.04 | 0.81 | 1.26 | 2.45 | 0.02 | 0.65 |
| pc3 | 1.65 | 1.60 | 2.00 | 1.66 | 1.79 | 3.64 | 0.07 | 1.65 | 4.91 | 9.19 | 7.66 | 14.04 | 1.35 | 1.20 | 2.86 | 2.26 | 3.33 | 7.23 | 0.05 | 1.52 |
| pc4 | 1.26 | 1.16 | 1.62 | 1.44 | 1.74 | 3.48 | 0.06 | 1.54 | 5.33 | 10.47 | 7.37 | 13.92 | 0.78 | 0.82 | 2.27 | 1.96 | 2.90 | 6.88 | 0.04 | 1.30 |
| phoneme | 6.85 | 7.13 | 2.73 | 1.75 | 2.30 | 3.96 | 0.06 | 1.82 | 0.43 | 0.79 | 4.29 | 7.29 | 5.66 | 5.91 | 2.61 | 1.78 | 3.00 | 5.44 | 0.05 | 1.64 |
| qsar-biodeg | 0.79 | 1.03 | 1.68 | 1.37 | 1.25 | 2.73 | 0.05 | 1.27 | 3.19 | 6.97 | 6.52 | 12.85 | 0.56 | 0.62 | 2.15 | 1.55 | 1.93 | 4.51 | 0.03 | 0.87 |
| sick | 0.66 | 0.68 | 2.34 | 1.98 | 1.82 | 3.33 | 0.07 | 1.45 | 1.08 | 1.97 | 4.88 | 8.01 | 0.37 | 0.38 | 2.59 | 2.09 | 2.50 | 4.17 | 0.05 | 1.34 |
| sonar | 0.12 | 0.13 | 0.77 | 0.72 | 0.68 | 2.07 | 0.03 | 0.81 | 5.70 | 12.15 | 8.28 | 16.41 | 0.09 | 0.09 | 0.55 | 0.50 | 0.68 | 1.87 | 0.02 | 0.39 |
| spambase | 10.49 | 9.95 | 8.94 | 7.10 | 6.00 | 13.92 | 0.30 | 5.53 | 16.08 | 35.47 | 20.32 | 37.12 | 12.96 | 6.93 | 24.94 | 18.69 | 21.03 | 41.27 | 0.29 | 9.92 |
| tic-tac-toe | 0.21 | 0.20 | 0.80 | 0.67 | 0.69 | 1.16 | 0.02 | 0.72 | 0.12 | 0.20 | 2.72 | 4.95 | 0.10 | 0.10 | 0.59 | 0.49 | 0.55 | 0.93 | 0.01 | 0.44 |
| vote | 0.07 | 0.07 | 0.33 | 0.23 | 0.33 | 0.60 | 0.01 | 0.48 | 0.09 | 0.16 | 2.06 | 3.74 | 0.05 | 0.04 | 0.27 | 0.20 | 0.29 | 0.45 | 0.01 | 0.34 |
| wdbc | 0.09 | 0.11 | 0.92 | 0.70 | 0.81 | 1.95 | 0.03 | 0.94 | 2.70 | 5.06 | 5.73 | 9.84 | 0.07 | 0.07 | 0.85 | 0.72 | 0.99 | 2.07 | 0.02 | 0.58 |
| wilt | 3.29 | 1.42 | 2.14 | 1.30 | 1.85 | 2.36 | 0.04 | 1.79 | 0.46 | 0.84 | 4.21 | 6.91 | 3.19 | 1.02 | 1.92 | 1.29 | 1.94 | 2.78 | 0.03 | 1.31 |

