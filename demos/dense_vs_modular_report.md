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
isn't in the main comparison". On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 96.67% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 1.2s per fit against 0.18s and 0.01s -- 7x as long as the beam search; 6 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

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

On 5 very small datasets (monks-problems-3, mofn-3-7-10, vote, tic-tac-toe, breast-cancer), exhaustive `ORB` reached a mean accuracy of 96.67% against 88.65% for a beam search of width 10 and 88.24% for greedy search (over the fits that finished), in 1.2s per fit against 0.18s and 0.01s -- 7x as long as the beam search; 6 of its 15 fits hit the 120s cap. Branch and bound is exponential in the number of conditions it may combine, so it doesn't scale to the feature spaces of most datasets (dozens to hundreds of binary features after discretizing numeric attributes) -- it is left out of the main comparison.

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
| breast-cancer | 0.02 | 0.21 | 0.38 | 0.01 |  |  |
| mofn-3-7-10 | 0.02 | 0.07 | 0.11 | 0.01 | 0.49 | 0.56 |
| monks-problems-3 | 0.01 | 0.07 | 0.11 | 0.01 | 0.76 | 0.95 |
| tic-tac-toe | 0.02 | 0.14 | 0.19 | 0.01 | 86.02 |  |
| vote | 0.01 | 0.09 | 0.13 | 0.01 | 2.54 | 2.18 |

## Dense vs. modular

For each pair, over the folds where both fits finished: how often the two learned the same model (the same rules in any order), their mean accuracy, and the ratio of the modular to the dense fit time -- the geometric mean over datasets, so a value below 1 means the modular learner is faster. The dense learner on the Boolean matrix, the modular one on vertical bitmaps (the default); the modular learners on the horizontal bitmap are compared with them below.

| pair | identical models | accuracy dense | accuracy modular | difference (points) | time modular / dense, small | medium | all |
|---|--:|--:|--:|--:|--:|--:|--:|
| CPAR | 345/345 | 85.64% | 85.64% | +0.00 | 1.56 | 1.16 | 1.38 |
| CPAR PRM | 259/345 | 85.70% | 85.58% | -0.12 | 1.61 | 1.37 | 1.50 |
| ENDER | 341/345 | 85.83% | 85.83% | +0.00 | 1.18 | 0.75 | 0.97 |
| ENDER CS-Exp | 337/345 | 86.01% | 86.01% | -0.01 | 1.27 | 0.75 | 1.02 |
| ENDER MLRules | 322/345 | 86.24% | 86.22% | -0.02 | 1.11 | 0.59 | 0.84 |
| Boomer | 327/345 | 86.37% | 86.36% | -0.01 | 1.19 | 0.63 | 0.91 |
| ORB | 328/345 | 83.54% | 83.55% | +0.02 | 2.03 | 1.18 | 1.60 |
| LRI | 44/345 | 85.01% | 85.08% | +0.07 | 1.76 | 1.01 | 1.38 |

## Summary by learner (mean across the 44 datasets and their folds)

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ENDER MLRules | 86.22% | 348.1 | 1432.6 | 4.0 | 1.13 | 2.7 | 7.58 | 0 |
| DenseENDER MLRules | 86.24% | 348.2 | 1432.9 | 4.0 | 2.05 | 1.7 | 7.78 | 0 |
| DenseENDER | 85.83% | 406.1 | 1876.8 | 4.4 | 1.90 | 3.1 | 7.85 | 0 |
| ENDER | 85.83% | 406.1 | 1876.9 | 4.4 | 1.33 | 3.1 | 7.88 | 0 |
| DenseENDER CS-Exp | 86.01% | 395.6 | 1722.8 | 4.1 | 1.55 | 3.0 | 8.26 | 0 |
| ENDER CS-Exp | 86.01% | 395.6 | 1722.6 | 4.1 | 1.09 | 3.0 | 8.28 | 0 |
| LRI(beam 10) | 86.32% | 392.3 | 1404.1 | 3.6 | 9.35 | 3.7 | 8.30 | 0 |
| LRI(beam 5) | 86.09% | 392.7 | 1365.8 | 3.5 | 5.12 | 3.2 | 8.49 | 0 |
| DenseLRI | 85.01% | 396.5 | 1563.4 | 3.9 | 1.14 | 6.0 | 9.11 | 0 |
| LRI | 85.08% | 396.7 | 1584.7 | 4.0 | 1.15 | 0.1 | 9.20 | 0 |
| Boomer | 86.36% | 275.3 | 1803.2 | 5.4 | 2.47 | 4.1 | 10.39 | 0 |
| DenseBoomer | 86.37% | 275.3 | 1803.3 | 5.4 | 4.13 | 4.1 | 10.59 | 0 |
| DenseCPAR PRM | 85.70% | 320.0 | 2025.6 | 4.6 | 1.36 | 1.7 | 11.98 | 0 |
| CPAR | 85.64% | 386.9 | 2526.4 | 4.6 | 1.35 | 0.7 | 12.06 | 0 |
| DenseCPAR | 85.64% | 386.9 | 2526.4 | 4.6 | 1.33 | 0.7 | 12.06 | 0 |
| CPAR PRM | 85.58% | 320.9 | 2045.2 | 4.6 | 1.59 | 0.7 | 12.57 | 0 |
| ORB(beam 10) | 84.28% | 9.8 | 50.4 | 5.1 | 3.77 | 1.2 | 14.20 | 0 |
| ORB(beam 5) | 84.06% | 9.8 | 48.0 | 4.9 | 1.83 | 0.2 | 14.22 | 0 |
| DenseORB(greedy) | 83.54% | 9.7 | 39.6 | 4.1 | 0.04 | 0.1 | 14.59 | 0 |
| ORB(greedy) | 83.55% | 9.7 | 39.6 | 4.1 | 0.04 | 1.1 | 14.61 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised. A dense learner and its twin that learn the same models tie for rank, so the ranking separates the algorithms and the searches, not dense from modular.

### Small datasets only (25, fewer than 1,000 examples)

`SPECT`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `climate-model-simulation-crashes`, `colic`, `credit-approval`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `ilpd`, `ionosphere`, `kc2`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `sonar`, `tic-tac-toe`, `vote`, `wdbc`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DenseENDER CS-Exp | 84.72% | 395.8 | 1519.5 | 3.7 | 0.54 | 2.1 | 7.72 | 0 |
| ENDER CS-Exp | 84.72% | 395.8 | 1519.1 | 3.7 | 0.66 | 2.1 | 7.76 | 0 |
| ENDER MLRules | 84.97% | 346.0 | 1283.5 | 3.7 | 0.68 | 1.1 | 7.82 | 0 |
| DenseENDER MLRules | 84.99% | 346.2 | 1284.5 | 3.7 | 0.65 | 1.1 | 7.86 | 0 |
| ENDER | 84.42% | 405.4 | 1629.0 | 3.8 | 0.75 | 1.2 | 8.32 | 0 |
| DenseENDER | 84.42% | 405.4 | 1629.0 | 3.8 | 0.65 | 1.2 | 8.32 | 0 |
| LRI(beam 10) | 85.15% | 391.9 | 1371.8 | 3.5 | 7.80 | 1.6 | 9.00 | 0 |
| DenseLRI | 83.64% | 396.5 | 1533.8 | 3.9 | 0.45 | 2.9 | 9.86 | 0 |
| LRI(beam 5) | 84.81% | 391.9 | 1329.3 | 3.4 | 4.22 | 1.1 | 9.88 | 0 |
| LRI | 83.73% | 396.5 | 1554.9 | 3.9 | 0.78 | 0.1 | 10.36 | 0 |
| DenseBoomer | 85.35% | 231.5 | 1499.0 | 5.3 | 1.50 | 2.4 | 11.10 | 0 |
| Boomer | 85.32% | 231.6 | 1500.3 | 5.3 | 1.59 | 2.4 | 11.16 | 0 |
| CPAR | 84.49% | 183.1 | 862.3 | 4.1 | 0.36 | 0.7 | 11.34 | 0 |
| DenseCPAR | 84.49% | 183.1 | 862.3 | 4.1 | 0.23 | 0.7 | 11.34 | 0 |
| DenseCPAR PRM | 84.59% | 166.8 | 785.0 | 4.1 | 0.25 | 1.7 | 11.34 | 0 |
| CPAR PRM | 84.45% | 166.1 | 779.8 | 4.1 | 0.39 | 0.7 | 11.92 | 0 |
| ORB(beam 10) | 82.90% | 9.8 | 48.0 | 4.9 | 2.27 | 1.2 | 13.20 | 0 |
| ORB(greedy) | 82.09% | 9.7 | 36.6 | 3.7 | 0.03 | 1.1 | 13.56 | 0 |
| DenseORB(greedy) | 82.06% | 9.7 | 36.6 | 3.7 | 0.01 | 0.1 | 13.60 | 0 |
| ORB(beam 5) | 82.55% | 9.8 | 45.7 | 4.6 | 1.14 | 0.2 | 14.54 | 0 |

### Medium datasets only (19, 1,000 to 10,000 examples)

`Titanic`, `banknote-authentication`, `churn`, `compas-two-years`, `credit-g`, `heloc`, `kc1`, `kr-vs-kp`, `mofn-3-7-10`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `spambase`, `wilt`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LRI(beam 5) | 89.46% | 394.9 | 1462.0 | 3.7 | 7.49 | 2.2 | 6.66 | 0 |
| DenseENDER | 89.51% | 408.0 | 2529.1 | 5.8 | 5.19 | 1.9 | 7.24 | 0 |
| ENDER MLRules | 89.52% | 353.7 | 1824.9 | 4.9 | 2.32 | 1.7 | 7.26 | 0 |
| ENDER | 89.51% | 408.0 | 2529.2 | 5.8 | 2.86 | 1.9 | 7.29 | 0 |
| LRI(beam 10) | 89.39% | 393.3 | 1489.2 | 3.8 | 13.46 | 2.2 | 7.37 | 0 |
| DenseENDER MLRules | 89.52% | 353.7 | 1823.5 | 4.9 | 5.72 | 0.7 | 7.68 | 0 |
| LRI | 88.64% | 397.2 | 1663.2 | 4.2 | 2.12 | 0.1 | 7.68 | 0 |
| DenseLRI | 88.62% | 396.6 | 1641.4 | 4.1 | 2.94 | 3.1 | 8.13 | 0 |
| ENDER CS-Exp | 89.41% | 395.1 | 2258.0 | 5.3 | 2.21 | 0.9 | 8.97 | 0 |
| DenseENDER CS-Exp | 89.41% | 395.0 | 2258.0 | 5.3 | 4.23 | 0.9 | 8.97 | 0 |
| Boomer | 89.08% | 390.3 | 2600.0 | 5.8 | 4.78 | 1.7 | 9.37 | 0 |
| DenseBoomer | 89.05% | 390.6 | 2604.0 | 5.8 | 11.04 | 1.7 | 9.92 | 0 |
| DenseCPAR PRM | 88.62% | 723.1 | 5290.5 | 5.8 | 4.26 | 0.0 | 12.82 | 0 |
| CPAR | 88.69% | 923.3 | 6905.6 | 6.0 | 3.97 | 0.0 | 13.00 | 0 |
| DenseCPAR | 88.69% | 923.3 | 6905.6 | 6.0 | 4.20 | 0.0 | 13.00 | 0 |
| CPAR PRM | 88.54% | 728.4 | 5375.1 | 5.8 | 4.74 | 0.0 | 13.42 | 0 |
| ORB(beam 5) | 88.05% | 9.7 | 54.0 | 5.5 | 3.65 | 0.0 | 13.79 | 0 |
| ORB(beam 10) | 87.93% | 9.7 | 56.8 | 5.8 | 7.73 | 0.0 | 15.53 | 0 |
| DenseORB(greedy) | 87.42% | 9.7 | 47.5 | 4.9 | 0.10 | 0.0 | 15.89 | 0 |
| ORB(greedy) | 87.41% | 9.7 | 47.7 | 4.9 | 0.09 | 0.0 | 16.00 | 0 |

## Beam search for ORB and LRI

The dense `ORB` and `LRI` come with their own greedy (and, for `ORB`, exhaustive) search; the modular ones take any `RuleSearch`. Mean over all datasets:

| learner | accuracy | n_conditions | fit_time (s) |
|---|--:|--:|--:|
| ORB(greedy) | 83.55% | 39.6 | 0.04 |
| ORB(beam 5) | 84.06% | 48.0 | 1.83 |
| ORB(beam 10) | 84.28% | 50.4 | 3.77 |
| LRI | 85.08% | 1584.7 | 1.15 |
| LRI(beam 5) | 86.09% | 1365.8 | 5.12 |
| LRI(beam 10) | 86.32% | 1404.1 | 9.35 |

## The modular learners on the horizontal bitmap

The modular learners above run on `VerticalBitmapRepresentation`, the library's default. Here they are run again on `HorizontalBitmapRepresentation`, the matrix the dense learners work on, over the folds where both finished; a time ratio above 1 means the vertical bitmap is faster:

| learner | accuracy, VBitmap | accuracy, HBitmap | time HBitmap / VBitmap (geometric mean over datasets) |
|---|--:|--:|--:|
| CPAR | 85.64% | 85.64% | 1.13 |
| CPAR PRM | 85.58% | 85.68% | 1.15 |
| ENDER | 85.83% | 85.83% | 0.99 |
| ENDER CS-Exp | 86.01% | 86.01% | 0.98 |
| ENDER MLRules | 86.22% | 86.22% | 1.05 |
| Boomer | 86.36% | 86.36% | 1.20 |
| ORB(greedy) | 83.55% | 83.55% | 1.32 |
| LRI | 85.08% | 85.08% | 1.14 |
| ORB(beam 5) | 84.06% | 84.06% | 1.04 |
| ORB(beam 10) | 84.28% | 84.27% | 1.02 |
| LRI(beam 5) | 86.09% | 86.09% | 1.01 |
| LRI(beam 10) | 86.32% | 86.32% | 1.00 |

## Per-dataset results

Accuracy (mean over the folds that finished):

| dataset | CPAR | CPAR PRM | ENDER | ENDER CS-Exp | ENDER MLRules | Boomer | ORB(greedy) | LRI | ORB(beam 5) | ORB(beam 10) | LRI(beam 5) | LRI(beam 10) | DenseCPAR | DenseCPAR PRM | DenseENDER | DenseENDER CS-Exp | DenseENDER MLRules | DenseBoomer | DenseORB(greedy) | DenseLRI |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SPECT | 82.78% | 81.31% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% | 82.79% | 82.44% | 83.82% | 83.83% | 82.78% | 82.42% | 84.63% | 84.62% | 83.86% | 82.34% | 83.90% | 80.81% |
| Titanic | 78.84% | 77.62% | 80.67% | 80.44% | 79.99% | 78.30% | 79.83% | 79.37% | 79.91% | 79.75% | 79.45% | 79.07% | 78.84% | 78.00% | 80.67% | 80.44% | 79.99% | 78.30% | 79.83% | 78.69% |
| banknote-authentication | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% | 98.18% | 97.67% | 99.05% | 99.05% | 98.54% | 98.61% | 98.98% | 98.91% | 99.05% | 99.20% | 97.30% | 99.13% |
| blood-transfusion-service-center | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 77.54% | 76.20% | 76.86% | 77.53% | 77.40% | 77.67% | 77.67% | 77.53% | 77.80% | 77.93% | 77.80% | 76.60% | 78.20% |
| breast-cancer | 73.42% | 72.72% | 72.39% | 73.09% | 71.69% | 68.94% | 70.65% | 70.64% | 69.58% | 73.07% | 71.69% | 72.77% | 73.42% | 73.76% | 72.39% | 73.09% | 72.03% | 69.66% | 70.65% | 71.33% |
| breast-w | 95.28% | 95.85% | 96.85% | 97.00% | 96.85% | 96.42% | 95.71% | 96.14% | 95.85% | 95.85% | 96.57% | 96.71% | 95.28% | 95.85% | 96.85% | 97.00% | 96.85% | 96.42% | 95.71% | 96.28% |
| churn | 93.02% | 93.38% | 92.58% | 92.02% | 93.38% | 93.46% | 89.56% | 93.24% | 91.08% | 90.92% | 93.28% | 93.64% | 93.02% | 93.42% | 92.58% | 92.02% | 93.36% | 93.38% | 89.56% | 93.20% |
| climate-model-simulation-crashes | 93.52% | 93.70% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 93.89% | 92.78% | 92.78% | 93.89% | 93.70% | 93.52% | 93.89% | 92.96% | 93.15% | 94.07% | 93.52% | 93.15% | 94.26% |
| colic | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.07% | 84.26% | 82.36% | 84.81% | 85.35% | 83.98% | 83.45% | 84.80% | 84.80% | 84.00% | 83.73% | 82.89% | 85.35% |
| compas-two-years | 63.72% | 64.49% | 68.09% | 68.11% | 67.98% | 66.65% | 66.86% | 67.64% | 67.26% | 67.26% | 67.58% | 67.71% | 63.72% | 64.68% | 68.09% | 68.11% | 67.98% | 66.65% | 66.86% | 67.07% |
| credit-approval | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 87.25% | 87.25% | 86.67% | 86.52% | 87.39% | 85.22% | 85.51% | 86.81% | 86.67% | 86.67% | 85.22% | 86.23% | 86.96% |
| credit-g | 73.50% | 73.50% | 76.90% | 76.90% | 76.20% | 74.70% | 70.80% | 75.70% | 72.20% | 70.30% | 75.10% | 74.90% | 73.50% | 74.40% | 76.90% | 76.90% | 76.30% | 74.40% | 70.90% | 76.00% |
| cylinder-bands | 77.04% | 78.15% | 80.19% | 80.74% | 81.11% | 86.85% | 71.48% | 82.22% | 76.67% | 73.89% | 82.41% | 83.15% | 77.04% | 78.52% | 80.19% | 80.93% | 81.11% | 86.85% | 71.48% | 82.22% |
| diabetes | 76.03% | 76.42% | 74.74% | 74.48% | 74.87% | 74.09% | 74.87% | 75.39% | 74.35% | 74.48% | 76.04% | 75.78% | 76.03% | 76.42% | 74.74% | 74.48% | 74.61% | 74.09% | 74.87% | 75.53% |
| dresses-sales | 57.00% | 57.40% | 62.20% | 63.20% | 58.80% | 58.40% | 58.20% | 58.80% | 54.80% | 54.80% | 58.20% | 59.40% | 57.00% | 56.80% | 62.20% | 63.20% | 59.40% | 58.40% | 58.20% | 58.80% |
| heart-c | 81.22% | 81.89% | 82.87% | 81.90% | 80.27% | 76.66% | 81.87% | 79.25% | 80.23% | 78.88% | 77.62% | 80.58% | 81.22% | 81.89% | 82.87% | 81.90% | 80.27% | 76.66% | 81.87% | 78.92% |
| heart-h | 79.92% | 80.62% | 81.28% | 80.60% | 81.62% | 80.31% | 80.93% | 81.30% | 80.23% | 80.92% | 80.63% | 81.32% | 79.92% | 80.62% | 81.28% | 80.60% | 81.62% | 80.31% | 80.93% | 81.31% |
| heart-statlog | 81.85% | 80.74% | 81.48% | 81.11% | 79.63% | 77.04% | 79.26% | 77.78% | 79.63% | 80.37% | 77.04% | 78.15% | 81.85% | 80.74% | 81.48% | 81.11% | 79.63% | 77.04% | 79.26% | 75.93% |
| heloc | 70.14% | 70.27% | 72.19% | 71.78% | 72.07% | 69.64% | 70.34% | 72.00% | 70.24% | 70.11% | 71.93% | 71.35% | 70.14% | 69.98% | 72.19% | 71.78% | 72.07% | 69.64% | 70.34% | 71.88% |
| hepatitis | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 84.50% | 82.12% | 80.21% | 78.88% | 82.12% | 79.50% | 80.71% | 78.75% | 79.54% | 80.88% | 80.25% | 81.50% | 83.87% | 80.88% |
| ilpd | 71.17% | 69.46% | 69.98% | 71.02% | 70.49% | 69.12% | 70.15% | 71.68% | 70.67% | 71.86% | 70.13% | 70.13% | 71.17% | 69.81% | 69.98% | 71.02% | 70.49% | 69.29% | 70.15% | 70.99% |
| ionosphere | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% | 91.72% | 91.73% | 93.16% | 92.60% | 90.88% | 89.17% | 94.29% | 93.73% | 94.30% | 94.01% | 90.02% | 93.44% |
| kc1 | 84.26% | 84.02% | 85.63% | 85.59% | 85.02% | 83.78% | 85.06% | 85.49% | 84.97% | 85.16% | 85.44% | 85.44% | 84.26% | 83.88% | 85.63% | 85.59% | 85.02% | 83.78% | 85.06% | 85.35% |
| kc2 | 83.50% | 83.88% | 84.26% | 84.84% | 83.31% | 81.40% | 83.12% | 81.59% | 80.82% | 81.21% | 81.98% | 82.36% | 83.50% | 83.69% | 84.26% | 84.84% | 83.31% | 81.21% | 83.12% | 81.98% |
| kr-vs-kp | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% | 97.47% | 97.22% | 99.00% | 99.09% | 99.19% | 99.28% | 99.03% | 99.09% | 99.53% | 99.50% | 95.18% | 98.91% |
| mofn-3-7-10 | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% | 94.64% | 95.54% | 100.00% | 100.00% | 98.26% | 98.26% | 100.00% | 100.00% | 100.00% | 100.00% | 93.88% | 83.99% |
| molecular-biology_promoters | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 82.91% | 81.27% | 82.27% | 80.27% | 82.27% | 80.36% | 84.09% | 92.55% | 93.55% | 93.45% | 90.55% | 85.09% | 77.55% |
| monks-problems-1 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 82.39% | 100.00% |
| monks-problems-2 | 96.17% | 97.34% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 67.21% | 75.70% | 77.53% | 93.35% | 95.35% | 96.17% | 97.50% | 65.40% | 65.73% | 76.54% | 98.84% | 69.88% | 71.39% |
| monks-problems-3 | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 98.20% | 98.92% | 98.92% | 97.84% | 97.84% | 98.92% | 98.92% | 98.92% | 98.74% | 97.66% | 97.66% | 98.92% | 97.84% |
| mushroom | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% | 99.48% | 99.84% | 100.00% | 100.00% | 99.95% | 99.95% | 100.00% | 100.00% | 100.00% | 100.00% | 99.54% | 100.00% |
| ozone-level-8hr | 93.73% | 93.73% | 94.00% | 93.96% | 94.32% | 94.28% | 93.01% | 94.24% | 92.98% | 92.94% | 94.28% | 93.96% | 93.73% | 93.80% | 94.00% | 93.96% | 94.20% | 94.28% | 93.09% | 94.20% |
| pc1 | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.06% | 92.16% | 93.06% | 92.97% | 92.16% | 92.88% | 93.06% | 92.61% | 91.98% | 92.70% | 92.79% | 92.61% | 92.15% | 92.16% | 93.15% |
| pc3 | 88.99% | 89.38% | 88.93% | 89.25% | 89.06% | 89.12% | 89.12% | 89.25% | 89.31% | 88.74% | 89.38% | 89.12% | 88.99% | 89.31% | 88.80% | 89.25% | 88.99% | 88.87% | 89.12% | 89.57% |
| pc4 | 90.12% | 89.71% | 89.92% | 89.51% | 89.85% | 89.92% | 88.75% | 89.98% | 88.89% | 89.23% | 90.26% | 90.19% | 90.12% | 89.99% | 89.99% | 89.51% | 89.92% | 89.92% | 88.75% | 89.92% |
| phoneme | 85.90% | 85.38% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 84.97% | 81.72% | 81.42% | 85.33% | 85.27% | 85.90% | 85.51% | 84.23% | 83.94% | 85.25% | 86.08% | 80.00% | 85.25% |
| qsar-biodeg | 85.59% | 83.79% | 86.16% | 86.35% | 85.97% | 84.83% | 81.99% | 86.07% | 83.32% | 84.08% | 85.78% | 85.50% | 85.59% | 83.79% | 86.16% | 86.35% | 85.97% | 84.83% | 81.99% | 86.45% |
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
| SPECT | 0.19 | 0.20 | 0.40 | 0.39 | 0.44 | 0.74 | 0.02 | 0.61 | 0.17 | 0.33 | 2.98 | 4.80 | 0.13 | 0.11 | 0.36 | 0.26 | 0.35 | 0.53 | 0.01 | 0.36 |
| Titanic | 1.39 | 1.31 | 0.96 | 0.87 | 0.75 | 1.54 | 0.02 | 0.82 | 0.27 | 0.51 | 3.45 | 5.73 | 1.04 | 0.88 | 0.86 | 0.63 | 0.92 | 1.49 | 0.01 | 0.58 |
| banknote-authentication | 0.12 | 0.15 | 0.78 | 0.57 | 0.54 | 1.14 | 0.02 | 0.81 | 0.24 | 0.47 | 3.29 | 5.58 | 0.08 | 0.09 | 0.73 | 0.51 | 0.55 | 0.97 | 0.01 | 0.53 |
| blood-transfusion-service-center | 0.41 | 0.36 | 0.61 | 0.54 | 0.57 | 1.05 | 0.02 | 0.78 | 0.16 | 0.37 | 3.29 | 5.48 | 0.32 | 0.29 | 0.58 | 0.41 | 0.52 | 0.79 | 0.01 | 0.48 |
| breast-cancer | 0.41 | 0.46 | 0.59 | 0.53 | 0.57 | 1.12 | 0.02 | 0.56 | 0.24 | 0.46 | 3.32 | 6.16 | 0.25 | 0.26 | 0.46 | 0.35 | 0.37 | 0.64 | 0.01 | 0.28 |
| breast-w | 0.13 | 0.15 | 1.17 | 1.07 | 1.01 | 2.56 | 0.04 | 1.03 | 1.18 | 2.66 | 5.85 | 10.20 | 0.08 | 0.10 | 1.11 | 1.01 | 1.14 | 2.72 | 0.02 | 0.64 |
| churn | 8.13 | 9.37 | 3.91 | 3.06 | 2.95 | 5.91 | 0.12 | 2.38 | 2.35 | 4.59 | 7.89 | 13.69 | 9.89 | 9.69 | 7.17 | 6.04 | 7.78 | 15.82 | 0.11 | 3.22 |
| climate-model-simulation-crashes | 0.20 | 0.19 | 0.85 | 0.74 | 0.74 | 1.61 | 0.03 | 0.80 | 1.16 | 2.24 | 4.51 | 8.69 | 0.13 | 0.12 | 0.80 | 0.66 | 0.89 | 2.04 | 0.02 | 0.52 |
| colic | 0.28 | 0.31 | 0.58 | 0.54 | 0.64 | 1.30 | 0.02 | 0.95 | 0.46 | 1.03 | 3.81 | 7.19 | 0.19 | 0.21 | 0.50 | 0.45 | 0.53 | 1.13 | 0.01 | 0.51 |
| compas-two-years | 7.48 | 7.81 | 2.92 | 1.87 | 2.44 | 3.78 | 0.07 | 2.29 | 0.58 | 1.17 | 5.52 | 9.85 | 8.94 | 9.23 | 4.37 | 3.11 | 4.52 | 6.76 | 0.07 | 2.27 |
| credit-approval | 0.69 | 0.64 | 0.97 | 0.90 | 0.87 | 1.79 | 0.03 | 0.93 | 0.71 | 1.39 | 4.04 | 7.39 | 0.42 | 0.44 | 0.88 | 0.70 | 0.88 | 1.80 | 0.02 | 0.59 |
| credit-g | 0.90 | 1.08 | 1.15 | 0.89 | 0.91 | 2.07 | 0.04 | 0.81 | 0.60 | 1.17 | 3.83 | 7.36 | 0.78 | 0.83 | 1.30 | 1.01 | 1.33 | 2.97 | 0.02 | 0.69 |
| cylinder-bands | 0.64 | 0.94 | 1.93 | 1.59 | 1.31 | 7.14 | 0.06 | 1.18 | 5.09 | 9.98 | 7.43 | 14.60 | 0.48 | 0.51 | 1.69 | 1.46 | 1.56 | 8.39 | 0.03 | 0.72 |
| diabetes | 1.16 | 1.30 | 0.88 | 0.74 | 0.80 | 1.69 | 0.03 | 0.91 | 0.37 | 0.81 | 3.45 | 6.07 | 0.83 | 1.02 | 0.91 | 0.70 | 0.89 | 1.83 | 0.02 | 0.68 |
| dresses-sales | 1.32 | 1.49 | 1.12 | 0.97 | 0.74 | 1.57 | 0.04 | 0.85 | 1.46 | 2.91 | 4.75 | 8.93 | 0.89 | 0.95 | 1.02 | 0.93 | 0.91 | 1.68 | 0.02 | 0.57 |
| heart-c | 0.24 | 0.26 | 0.55 | 0.54 | 0.53 | 1.17 | 0.02 | 0.69 | 0.38 | 0.68 | 3.44 | 5.78 | 0.14 | 0.16 | 0.41 | 0.36 | 0.42 | 0.74 | 0.01 | 0.30 |
| heart-h | 0.27 | 0.28 | 0.55 | 0.49 | 0.53 | 1.03 | 0.02 | 0.62 | 0.33 | 0.68 | 3.56 | 6.74 | 0.17 | 0.17 | 0.44 | 0.38 | 0.42 | 0.73 | 0.01 | 0.32 |
| heart-statlog | 0.24 | 0.28 | 0.69 | 0.55 | 0.62 | 1.26 | 0.02 | 0.71 | 0.38 | 0.75 | 3.64 | 6.13 | 0.15 | 0.16 | 0.48 | 0.40 | 0.46 | 0.84 | 0.01 | 0.36 |
| heloc | 23.98 | 38.23 | 8.68 | 6.06 | 8.31 | 15.08 | 0.27 | 5.40 | 4.32 | 8.90 | 15.26 | 27.84 | 26.67 | 36.79 | 18.32 | 16.19 | 31.98 | 50.21 | 0.53 | 13.06 |
| hepatitis | 0.10 | 0.12 | 0.42 | 0.42 | 0.40 | 0.90 | 0.02 | 0.57 | 0.28 | 0.56 | 3.02 | 5.48 | 0.06 | 0.07 | 0.31 | 0.27 | 0.25 | 0.58 | 0.01 | 0.26 |
| ilpd | 0.95 | 0.93 | 0.87 | 0.69 | 0.76 | 1.37 | 0.03 | 0.76 | 0.54 | 0.91 | 3.42 | 6.41 | 0.56 | 0.53 | 0.63 | 0.52 | 0.65 | 1.22 | 0.01 | 0.42 |
| ionosphere | 0.14 | 0.17 | 0.80 | 0.75 | 0.71 | 1.57 | 0.03 | 0.87 | 2.76 | 5.37 | 6.43 | 12.13 | 0.09 | 0.11 | 0.68 | 0.61 | 0.80 | 1.55 | 0.01 | 0.50 |
| kc1 | 2.88 | 2.93 | 1.71 | 1.20 | 1.52 | 2.89 | 0.05 | 1.67 | 1.49 | 3.24 | 5.29 | 10.54 | 2.40 | 2.48 | 2.50 | 1.71 | 2.91 | 4.97 | 0.05 | 1.51 |
| kc2 | 0.47 | 0.49 | 0.85 | 0.65 | 0.74 | 1.54 | 0.03 | 0.89 | 1.47 | 2.99 | 4.88 | 9.40 | 0.28 | 0.32 | 0.71 | 0.57 | 0.85 | 1.75 | 0.01 | 0.55 |
| kr-vs-kp | 0.37 | 0.45 | 1.74 | 1.77 | 1.40 | 2.32 | 0.04 | 1.42 | 0.52 | 0.89 | 4.38 | 7.17 | 0.21 | 0.20 | 1.73 | 1.33 | 1.31 | 2.32 | 0.03 | 1.15 |
| mofn-3-7-10 | 0.36 | 0.44 | 0.48 | 0.36 | 0.47 | 0.76 | 0.02 | 0.73 | 0.06 | 0.09 | 2.23 | 4.21 | 0.20 | 0.23 | 0.44 | 0.27 | 0.44 | 0.68 | 0.01 | 0.56 |
| molecular-biology_promoters | 0.06 | 0.07 | 0.53 | 0.52 | 0.53 | 1.13 | 0.02 | 0.62 | 1.16 | 2.32 | 5.46 | 9.69 | 0.04 | 0.04 | 0.34 | 0.30 | 0.36 | 0.72 | 0.01 | 0.28 |
| monks-problems-1 | 0.09 | 0.09 | 0.44 | 0.38 | 0.43 | 0.61 | 0.01 | 0.44 | 0.07 | 0.11 | 2.24 | 4.25 | 0.05 | 0.06 | 0.38 | 0.28 | 0.34 | 0.44 | 0.01 | 0.27 |
| monks-problems-2 | 0.40 | 0.39 | 0.50 | 0.45 | 0.65 | 0.92 | 0.02 | 0.70 | 0.08 | 0.13 | 2.46 | 4.59 | 0.22 | 0.25 | 0.49 | 0.39 | 0.54 | 0.75 | 0.01 | 0.45 |
| monks-problems-3 | 0.06 | 0.08 | 0.48 | 0.40 | 0.44 | 0.75 | 0.01 | 0.60 | 0.07 | 0.12 | 2.38 | 3.87 | 0.05 | 0.05 | 0.38 | 0.33 | 0.40 | 0.62 | 0.01 | 0.38 |
| mushroom | 0.31 | 0.34 | 3.46 | 2.73 | 2.66 | 5.53 | 0.10 | 2.77 | 3.46 | 7.03 | 8.72 | 12.70 | 0.18 | 0.19 | 5.33 | 4.55 | 5.11 | 8.94 | 0.10 | 3.32 |
| ozone-level-8hr | 1.90 | 2.06 | 4.77 | 4.11 | 3.84 | 12.18 | 0.22 | 4.03 | 21.77 | 45.80 | 18.45 | 35.50 | 1.88 | 1.73 | 13.71 | 10.91 | 13.82 | 37.32 | 0.25 | 7.62 |
| pc1 | 0.60 | 0.62 | 1.01 | 0.89 | 0.98 | 1.96 | 0.04 | 1.13 | 1.60 | 2.96 | 4.55 | 8.57 | 0.54 | 0.49 | 1.14 | 0.98 | 1.40 | 2.49 | 0.03 | 0.88 |
| pc3 | 1.53 | 2.08 | 2.12 | 1.60 | 1.62 | 3.52 | 0.06 | 1.92 | 5.20 | 9.54 | 8.47 | 15.09 | 1.50 | 1.58 | 2.66 | 2.30 | 3.00 | 7.07 | 0.05 | 1.30 |
| pc4 | 1.06 | 1.05 | 1.90 | 1.56 | 1.58 | 3.87 | 0.07 | 1.56 | 5.61 | 11.23 | 7.76 | 14.93 | 1.08 | 0.86 | 2.33 | 2.05 | 3.06 | 6.51 | 0.04 | 1.42 |
| phoneme | 7.55 | 8.18 | 2.67 | 2.08 | 2.31 | 4.66 | 0.08 | 2.12 | 0.45 | 0.83 | 5.12 | 8.32 | 6.89 | 6.40 | 3.02 | 2.55 | 3.36 | 5.78 | 0.06 | 1.96 |
| qsar-biodeg | 1.11 | 1.13 | 1.81 | 1.49 | 1.31 | 3.30 | 0.05 | 1.47 | 3.50 | 6.88 | 7.38 | 13.76 | 0.87 | 0.64 | 2.12 | 1.88 | 2.22 | 5.30 | 0.04 | 1.09 |
| sick | 0.58 | 0.69 | 2.58 | 1.92 | 1.98 | 3.19 | 0.05 | 1.52 | 0.99 | 1.96 | 4.49 | 7.78 | 0.48 | 0.44 | 2.88 | 2.28 | 2.61 | 4.41 | 0.05 | 1.35 |
| sonar | 0.11 | 0.13 | 0.85 | 0.79 | 0.78 | 2.46 | 0.04 | 1.01 | 6.59 | 13.52 | 9.30 | 19.15 | 0.09 | 0.10 | 0.61 | 0.59 | 0.77 | 2.15 | 0.02 | 0.47 |
| spambase | 11.82 | 10.77 | 9.50 | 7.44 | 6.67 | 14.23 | 0.29 | 5.70 | 15.79 | 38.60 | 21.82 | 39.76 | 13.10 | 7.13 | 25.66 | 20.64 | 20.41 | 42.68 | 0.43 | 11.94 |
| tic-tac-toe | 0.20 | 0.22 | 0.75 | 0.64 | 0.74 | 1.26 | 0.02 | 0.86 | 0.12 | 0.22 | 2.85 | 5.46 | 0.12 | 0.12 | 0.75 | 0.52 | 0.61 | 1.02 | 0.01 | 0.52 |
| vote | 0.08 | 0.09 | 0.37 | 0.33 | 0.40 | 0.73 | 0.01 | 0.58 | 0.10 | 0.18 | 2.36 | 4.39 | 0.04 | 0.05 | 0.30 | 0.24 | 0.28 | 0.51 | 0.01 | 0.30 |
| wdbc | 0.11 | 0.12 | 0.99 | 0.87 | 0.96 | 2.44 | 0.03 | 1.04 | 3.09 | 6.00 | 6.67 | 11.94 | 0.07 | 0.07 | 0.96 | 0.80 | 1.11 | 2.27 | 0.02 | 0.61 |
| wilt | 3.41 | 1.41 | 2.13 | 1.54 | 1.92 | 2.84 | 0.04 | 1.83 | 0.48 | 0.91 | 4.42 | 7.28 | 3.05 | 1.14 | 2.37 | 1.36 | 2.03 | 3.16 | 0.03 | 1.41 |

