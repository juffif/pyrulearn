# Multi-class decomposition: one-vs-rest vs. pairwise

## Setup

**Full run.** Quick sanity check instead: `python demos/multiclass_decomposition.py`.

A rule learner that learns rules for one class against the rest has to
decompose a problem with several classes into two-class problems. This
demo compares the ways of doing that in pyrulearn, with Pypper
(pyrulearn's RIPPER), PFossil and PFoil as base learners -- in essence the
experiments of Fürnkranz, *Round Robin Classification* (JMLR 2002), with
Pypper standing in for Ripper. The paper's argument for pairwise
classification is its cost: a pairwise decomposition trains on about
(c - 1) times the examples in all, one-vs-rest on c times as many, but in many
small problems -- a large gain for a learner whose cost grows faster than
linearly with the examples. Of the three, PFossil grows at most linearly with
the examples, PFoil clearly faster (on `adult`, from 2,000 to 16,000 examples,
about n^1.6, with a rule count growing almost in proportion), Pypper in
between, depending on the data (n^1.0-1.8). CN2 would be another superlinear
learner, but its Laplace heuristic takes the number of classes of the whole
problem into every sub-problem (26 in a pair of `letter`'s classes), which
changes its searches and its results; it is left out until that is settled
(see ROADMAP.md):

- **ovr** -- one-vs-rest: one model per class, "this class vs. all
  others", every model trained on all examples; conflicts resolved by the
  best rule.
- **ordered** -- ordered one-vs-rest: classes from least to most frequent,
  each learned against the classes not yet handled; the most frequent
  class is the default. This is Ripper's (and Pypper's) own strategy.
- **pw_smaller / pw_larger** -- pairwise (round robin): one model per pair
  of classes, trained on those two classes' examples only, with the
  pair's smaller (larger) class as the target.
- **pw_both** -- the double round robin: for every pair, a model for each
  of its two classes.

A pairwise model is scored with three vote schemes on the same fitted
model: **vote** (each pair model votes for one class), **weighted_vote**
(the vote is split by the reliability of the rule that decided it) and
**accuracy_vote** (the vote is split by the pair model's accuracy on its
own two classes). Only prediction changes, so the three share their
training.

**Why training time is interesting.** Pairwise learns many more models --
c(c-1)/2 for c classes -- but each on few examples: every example is used
in c - 1 pair models, so pairwise learns from (c - 1) * n examples in
total, against c * n for one-vs-rest. For a learner whose time grows
linearly with the examples, one-vs-rest therefore takes about c / (c - 1)
times as long; for one that grows faster, pairwise gains more with every
class. Prediction goes the other way: every pair model has to be asked.

Datasets, with their numbers of classes: `letter` (26), `audiology` (24), `primary-tumor` (21), `soybean` (19), `kropt` (18), `vowel` (11), `texture` (11), `yeast` (10), `led24` (10), `optdigits` (10), `zoo` (7), `segment` (7), `solar-flare` (6). Protocol: 10-fold stratified cross-validation (5-fold for the large kropt, letter), one
`DataSpec` per training fold (`build_dataspec(max_intervals=8)`).
Each fit is capped at 300s; a time-out counts as a failure.
Training time is measured inside the worker process, so it excludes
shipping data and models between processes. Every learner runs single-threaded.
Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

## Pypper

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.11% | 2.27 | 431.0 | 1143.6 | 11.34 | 0.412 | 0 |
| pw_both:vote | 77.90% | 2.96 | 431.0 | 1143.6 | 11.34 | 0.318 | 0 |
| pw_smaller:accuracy_vote | 77.95% | 4.27 | 209.2 | 547.5 | 5.37 | 0.204 | 0 |
| pw_larger:accuracy_vote | 78.09% | 4.81 | 221.1 | 597.7 | 6.52 | 0.209 | 0 |
| pw_smaller:vote | 78.10% | 5.19 | 209.2 | 547.5 | 5.37 | 0.158 | 0 |
| pw_both:weighted_vote | 70.92% | 6.12 | 431.0 | 1143.6 | 11.34 | 2.989 | 0 |
| ordered | 76.14% | 6.35 | 39.1 | 172.4 | 2.63 | 0.002 | 0 |
| pw_larger:vote | 69.05% | 6.92 | 221.1 | 597.7 | 6.52 | 0.163 | 0 |
| ovr | 72.88% | 7.81 | 40.6 | 215.5 | 2.96 | 0.008 | 0 |
| pw_smaller:weighted_vote | 51.36% | 8.69 | 209.2 | 547.5 | 5.37 | 1.384 | 0 |
| pw_larger:weighted_vote | 40.11% | 10.62 | 221.1 | 597.7 | 6.52 | 1.645 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFossil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.15% | 2.77 | 569.5 | 1376.2 | 1.60 | 0.409 | 0 |
| pw_both:vote | 78.26% | 3.62 | 569.5 | 1376.2 | 1.60 | 0.314 | 0 |
| pw_smaller:accuracy_vote | 77.47% | 4.96 | 283.7 | 716.2 | 1.05 | 0.204 | 0 |
| pw_larger:accuracy_vote | 77.44% | 5.19 | 283.2 | 659.0 | 1.03 | 0.205 | 0 |
| pw_both:weighted_vote | 71.55% | 5.27 | 569.5 | 1376.2 | 1.60 | 2.898 | 0 |
| pw_smaller:vote | 77.12% | 5.65 | 283.7 | 716.2 | 1.05 | 0.156 | 0 |
| pw_larger:vote | 76.21% | 6.35 | 283.2 | 659.0 | 1.03 | 0.159 | 0 |
| ovr | 76.12% | 6.46 | 52.4 | 274.4 | 0.40 | 0.007 | 0 |
| ordered | 76.00% | 6.92 | 46.6 | 214.8 | 0.40 | 0.002 | 0 |
| pw_smaller:weighted_vote | 50.76% | 8.04 | 283.7 | 716.2 | 1.05 | 1.380 | 0 |
| pw_larger:weighted_vote | 38.56% | 10.77 | 283.2 | 659.0 | 1.03 | 1.555 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFoil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.06% | 2.65 | 1107.7 | 3894.9 | 3.67 | 0.605 | 0 |
| pw_both:vote | 78.06% | 3.12 | 1107.7 | 3894.9 | 3.67 | 0.510 | 0 |
| ovr | 78.83% | 3.77 | 326.6 | 2262.8 | 4.42 | 0.114 | 0 |
| pw_both:weighted_vote | 73.43% | 4.81 | 1107.7 | 3894.9 | 3.67 | 3.691 | 0 |
| pw_larger:accuracy_vote | 76.16% | 5.08 | 564.1 | 1998.5 | 2.06 | 0.306 | 0 |
| pw_larger:vote | 74.22% | 5.85 | 564.1 | 1998.5 | 2.06 | 0.258 | 0 |
| pw_smaller:accuracy_vote | 74.11% | 6.19 | 541.5 | 1895.4 | 2.05 | 0.298 | 0 |
| pw_smaller:vote | 73.77% | 6.50 | 541.5 | 1895.4 | 2.05 | 0.251 | 0 |
| pw_smaller:weighted_vote | 51.59% | 8.35 | 541.5 | 1895.4 | 2.05 | 1.673 | 0 |
| ordered | 64.22% | 9.08 | 217.6 | 1214.0 | 2.09 | 0.009 | 0 |
| pw_larger:weighted_vote | 39.20% | 10.62 | 564.1 | 1998.5 | 2.06 | 2.039 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## Training and prediction time over the number of classes

One point per dataset and base learner: the mean one-vs-rest time divided by the mean pairwise time. Above 1, pairwise is faster. The dashed line is what a learner whose time grows linearly with the number of examples would give.

![training time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_train_ratio.png)

![training time ratio, one-vs-rest / double round robin](multiclass_decomposition_plots/multiclass_decomposition_train_ratio_double.png)

![prediction time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_predict_ratio.png)

## Growth of the base learners and the pairwise speed-up

The argument for pairwise classification is about cost: if a learner's training time grows like n^b with the number of examples n, one-vs-rest costs c n^b for c classes, pairwise the sum over the pairs of (n_i + n_j)^b -- about (c - 1) n examples in all, but in many small problems, so the larger b, the more pairwise gains. Here b is measured for each base learner on each dataset with at least 1,000 training rows: one-vs-rest is fitted on nested, stratified subsamples of the first training fold (25%, 50%, 100% of it, outside the cross-validation, each with its own discretization), and b is the least-squares slope of log time over log rows. The predicted ratio uses the fold's real class counts; the measured one is the cross-validation's one-vs-rest / pairwise (`pw_smaller`) training time. Pairwise also has costs per pair the model leaves out -- building each pair's data, evaluating its model for the vote weights -- which weigh most with many classes and few examples.

Run times of rule learners are noisy in a way their results are not: one rule more or less can change a fit's time a lot and its quality hardly at all, so b from three subsamples, and the ratios, scatter. The points follow the line only roughly; a controlled experiment, varying the training-set size on a single dataset, would separate the growth from the noise better.

![pairwise speed-up, measured vs. predicted](multiclass_decomposition_plots/multiclass_decomposition_growth.png)

| dataset | classes | training rows | base | b | ovr / pairwise, measured | predicted (b) | predicted (b = 1) |
|---|--:|--:|---|--:|--:|--:|--:|
| yeast | 10 | 1,335 | PFoil | 1.30 | 1.06 | 1.63 | 1.11 |
| yeast | 10 | 1,335 | PFossil | 0.36 | 0.37 | 0.43 | 1.11 |
| yeast | 10 | 1,335 | Pypper | 1.04 | 0.69 | 1.17 | 1.11 |
| segment | 7 | 2,079 | PFoil | 0.82 | 1.43 | 0.93 | 1.17 |
| segment | 7 | 2,079 | PFossil | 0.49 | 0.77 | 0.61 | 1.17 |
| segment | 7 | 2,079 | Pypper | 0.74 | 1.28 | 0.85 | 1.17 |
| led24 | 10 | 2,880 | PFoil | 1.30 | 1.45 | 1.80 | 1.11 |
| led24 | 10 | 2,880 | PFossil | 0.04 | 0.42 | 0.24 | 1.11 |
| led24 | 10 | 2,880 | Pypper | 0.57 | 0.56 | 0.55 | 1.11 |
| texture | 11 | 4,950 | PFoil | 1.22 | 0.99 | 1.61 | 1.10 |
| texture | 11 | 4,950 | PFossil | 0.75 | 0.56 | 0.72 | 1.10 |
| texture | 11 | 4,950 | Pypper | 1.32 | 1.29 | 1.91 | 1.10 |
| optdigits | 10 | 5,058 | PFoil | 1.23 | 0.65 | 1.62 | 1.11 |
| optdigits | 10 | 5,058 | PFossil | 0.86 | 0.40 | 0.89 | 1.11 |
| optdigits | 10 | 5,058 | Pypper | 1.32 | 0.84 | 1.85 | 1.11 |
| letter | 26 | 16,000 | PFoil | 1.41 | 1.60 | 3.00 | 1.04 |
| letter | 26 | 16,000 | PFossil | 0.51 | 0.33 | 0.30 | 1.04 |
| letter | 26 | 16,000 | Pypper | 1.52 | 0.89 | 3.96 | 1.04 |
| kropt | 18 | 22,444 | PFoil | 1.51 | 3.17 | 2.78 | 1.06 |
| kropt | 18 | 22,444 | PFossil | 0.79 | 0.28 | 0.70 | 1.06 |
| kropt | 18 | 22,444 | Pypper | 1.75 | 0.34 | 4.36 | 1.06 |

Median b: PFoil 1.30, PFossil 0.51, Pypper 1.32.

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_conditions | train_time | predict_time |
|---|---|---|---|---|---|
| audiology | PFoil:ordered | 62.91% | 57.1 | 0.056 | 0.001 |
| audiology | PFoil:ovr | 77.04% | 94.3 | 0.056 | 0.002 |
| audiology | PFoil:pw_both:accuracy_vote | 67.27% | 284.8 | 0.465 | 0.028 |
| audiology | PFoil:pw_both:vote | 61.52% | 284.8 | 0.465 | 0.023 |
| audiology | PFoil:pw_both:weighted_vote | 46.46% | 284.8 | 0.465 | 0.128 |
| audiology | PFoil:pw_larger:accuracy_vote | 70.42% | 139.0 | 0.296 | 0.015 |
| audiology | PFoil:pw_larger:vote | 58.46% | 139.0 | 0.296 | 0.011 |
| audiology | PFoil:pw_larger:weighted_vote | 46.46% | 139.0 | 0.296 | 0.078 |
| audiology | PFoil:pw_smaller:accuracy_vote | 65.08% | 145.8 | 0.290 | 0.014 |
| audiology | PFoil:pw_smaller:vote | 71.23% | 145.8 | 0.290 | 0.011 |
| audiology | PFoil:pw_smaller:weighted_vote | 54.80% | 145.8 | 0.290 | 0.079 |
| audiology | PFossil:ordered | 84.05% | 72.5 | 0.059 | 0.001 |
| audiology | PFossil:ovr | 77.89% | 115.2 | 0.060 | 0.001 |
| audiology | PFossil:pw_both:accuracy_vote | 81.40% | 586.0 | 0.543 | 0.041 |
| audiology | PFossil:pw_both:vote | 81.40% | 586.0 | 0.543 | 0.035 |
| audiology | PFossil:pw_both:weighted_vote | 46.46% | 586.0 | 0.543 | 0.217 |
| audiology | PFossil:pw_larger:accuracy_vote | 80.53% | 292.0 | 0.345 | 0.020 |
| audiology | PFossil:pw_larger:vote | 81.40% | 292.0 | 0.345 | 0.018 |
| audiology | PFossil:pw_larger:weighted_vote | 46.46% | 292.0 | 0.345 | 0.094 |
| audiology | PFossil:pw_smaller:accuracy_vote | 80.95% | 294.0 | 0.327 | 0.019 |
| audiology | PFossil:pw_smaller:vote | 81.38% | 294.0 | 0.327 | 0.017 |
| audiology | PFossil:pw_smaller:weighted_vote | 54.86% | 294.0 | 0.327 | 0.085 |
| audiology | Pypper:ordered | 75.28% | 16.0 | 0.122 | 0.000 |
| audiology | Pypper:ovr | 64.64% | 26.9 | 0.135 | 0.001 |
| audiology | Pypper:pw_both:accuracy_vote | 68.18% | 142.9 | 1.330 | 0.025 |
| audiology | Pypper:pw_both:vote | 57.04% | 142.9 | 1.330 | 0.019 |
| audiology | Pypper:pw_both:weighted_vote | 46.46% | 142.9 | 1.330 | 0.125 |
| audiology | Pypper:pw_larger:accuracy_vote | 69.51% | 87.7 | 0.833 | 0.013 |
| audiology | Pypper:pw_larger:vote | 7.92% | 87.7 | 0.833 | 0.010 |
| audiology | Pypper:pw_larger:weighted_vote | 46.46% | 87.7 | 0.833 | 0.067 |
| audiology | Pypper:pw_smaller:accuracy_vote | 65.51% | 55.4 | 0.637 | 0.011 |
| audiology | Pypper:pw_smaller:vote | 70.83% | 55.4 | 0.637 | 0.008 |
| audiology | Pypper:pw_smaller:weighted_vote | 48.64% | 55.4 | 0.637 | 0.068 |
| kropt | PFoil:ordered | 69.80% | 15119.2 | 32.393 | 0.124 |
| kropt | PFoil:ovr | 75.41% | 31233.6 | 76.444 | 2.106 |
| kropt | PFoil:pw_both:accuracy_vote | 82.10% | 47380.2 | 46.542 | 7.164 |
| kropt | PFoil:pw_both:vote | 81.88% | 47380.2 | 46.542 | 6.364 |
| kropt | PFoil:pw_both:weighted_vote | 64.03% | 47380.2 | 46.542 | 36.748 |
| kropt | PFoil:pw_larger:accuracy_vote | 75.46% | 24408.4 | 23.755 | 3.708 |
| kropt | PFoil:pw_larger:vote | 75.29% | 24408.4 | 23.755 | 3.276 |
| kropt | PFoil:pw_larger:weighted_vote | 32.70% | 24408.4 | 23.755 | 23.532 |
| kropt | PFoil:pw_smaller:accuracy_vote | 75.78% | 22971.8 | 24.151 | 3.500 |
| kropt | PFoil:pw_smaller:vote | 78.50% | 22971.8 | 24.151 | 3.104 |
| kropt | PFoil:pw_smaller:weighted_vote | 72.94% | 22971.8 | 24.151 | 13.846 |
| kropt | PFossil:ordered | 35.48% | 420.6 | 1.745 | 0.004 |
| kropt | PFossil:ovr | 24.99% | 342.6 | 1.204 | 0.032 |
| kropt | PFossil:pw_both:accuracy_vote | 40.70% | 5670.4 | 7.013 | 3.354 |
| kropt | PFossil:pw_both:vote | 39.41% | 5670.4 | 7.013 | 2.549 |
| kropt | PFossil:pw_both:weighted_vote | 32.21% | 5670.4 | 7.013 | 23.392 |
| kropt | PFossil:pw_larger:accuracy_vote | 36.74% | 2446.2 | 3.915 | 1.704 |
| kropt | PFossil:pw_larger:vote | 34.82% | 2446.2 | 3.915 | 1.317 |
| kropt | PFossil:pw_larger:weighted_vote | 22.80% | 2446.2 | 3.915 | 14.335 |
| kropt | PFossil:pw_smaller:accuracy_vote | 39.58% | 3224.2 | 4.246 | 1.656 |
| kropt | PFossil:pw_smaller:vote | 36.35% | 3224.2 | 4.246 | 1.240 |
| kropt | PFossil:pw_smaller:weighted_vote | 39.95% | 3224.2 | 4.246 | 9.440 |
| kropt | Pypper:ordered | 54.59% | 1299.2 | 33.521 | 0.014 |
| kropt | Pypper:ovr | 33.63% | 1130.0 | 24.484 | 0.063 |
| kropt | Pypper:pw_both:accuracy_vote | 66.17% | 11866.4 | 167.498 | 3.964 |
| kropt | Pypper:pw_both:vote | 67.00% | 11866.4 | 167.498 | 3.132 |
| kropt | Pypper:pw_both:weighted_vote | 50.76% | 11866.4 | 167.498 | 27.591 |
| kropt | Pypper:pw_larger:accuracy_vote | 62.95% | 6464.0 | 97.613 | 2.060 |
| kropt | Pypper:pw_larger:vote | 63.52% | 6464.0 | 97.613 | 1.652 |
| kropt | Pypper:pw_larger:weighted_vote | 30.84% | 6464.0 | 97.613 | 17.569 |
| kropt | Pypper:pw_smaller:accuracy_vote | 62.28% | 5402.4 | 72.725 | 1.896 |
| kropt | Pypper:pw_smaller:vote | 62.10% | 5402.4 | 72.725 | 1.487 |
| kropt | Pypper:pw_smaller:weighted_vote | 60.03% | 5402.4 | 72.725 | 10.338 |
| led24 | PFoil:ordered | 47.06% | 2042.6 | 0.992 | 0.009 |
| led24 | PFoil:ovr | 61.81% | 3508.9 | 1.819 | 0.025 |
| led24 | PFoil:pw_both:accuracy_vote | 64.12% | 5456.2 | 2.433 | 0.124 |
| led24 | PFoil:pw_both:vote | 63.56% | 5456.2 | 2.433 | 0.120 |
| led24 | PFoil:pw_both:weighted_vote | 69.28% | 5456.2 | 2.433 | 0.722 |
| led24 | PFoil:pw_larger:accuracy_vote | 58.09% | 2730.4 | 1.249 | 0.064 |
| led24 | PFoil:pw_larger:vote | 55.44% | 2730.4 | 1.249 | 0.059 |
| led24 | PFoil:pw_larger:weighted_vote | 29.87% | 2730.4 | 1.249 | 0.368 |
| led24 | PFoil:pw_smaller:accuracy_vote | 54.34% | 2723.0 | 1.254 | 0.063 |
| led24 | PFoil:pw_smaller:vote | 53.88% | 2723.0 | 1.254 | 0.058 |
| led24 | PFoil:pw_smaller:weighted_vote | 30.91% | 2723.0 | 1.254 | 0.374 |
| led24 | PFossil:ordered | 70.03% | 250.2 | 0.175 | 0.002 |
| led24 | PFossil:ovr | 70.00% | 272.8 | 0.208 | 0.004 |
| led24 | PFossil:pw_both:accuracy_vote | 70.88% | 1698.1 | 0.941 | 0.075 |
| led24 | PFossil:pw_both:vote | 71.47% | 1698.1 | 0.941 | 0.059 |
| led24 | PFossil:pw_both:weighted_vote | 71.41% | 1698.1 | 0.941 | 0.478 |
| led24 | PFossil:pw_larger:accuracy_vote | 66.44% | 891.4 | 0.515 | 0.038 |
| led24 | PFossil:pw_larger:vote | 66.84% | 891.4 | 0.515 | 0.031 |
| led24 | PFossil:pw_larger:weighted_vote | 28.56% | 891.4 | 0.515 | 0.253 |
| led24 | PFossil:pw_smaller:accuracy_vote | 67.31% | 812.8 | 0.492 | 0.037 |
| led24 | PFossil:pw_smaller:vote | 67.34% | 812.8 | 0.492 | 0.030 |
| led24 | PFossil:pw_smaller:weighted_vote | 30.75% | 812.8 | 0.492 | 0.235 |
| led24 | Pypper:ordered | 71.50% | 88.7 | 0.587 | 0.001 |
| led24 | Pypper:ovr | 68.00% | 124.0 | 0.822 | 0.002 |
| led24 | Pypper:pw_both:accuracy_vote | 72.31% | 500.9 | 2.824 | 0.057 |
| led24 | Pypper:pw_both:vote | 72.19% | 500.9 | 2.824 | 0.043 |
| led24 | Pypper:pw_both:weighted_vote | 72.28% | 500.9 | 2.824 | 0.388 |
| led24 | Pypper:pw_larger:accuracy_vote | 71.47% | 250.7 | 1.423 | 0.030 |
| led24 | Pypper:pw_larger:vote | 71.22% | 250.7 | 1.423 | 0.023 |
| led24 | Pypper:pw_larger:weighted_vote | 31.87% | 250.7 | 1.423 | 0.205 |
| led24 | Pypper:pw_smaller:accuracy_vote | 72.09% | 251.5 | 1.466 | 0.032 |
| led24 | Pypper:pw_smaller:vote | 71.97% | 251.5 | 1.466 | 0.030 |
| led24 | Pypper:pw_smaller:weighted_vote | 34.38% | 251.5 | 1.466 | 0.195 |
| letter | PFoil:ordered | 59.28% | 4562.0 | 9.001 | 0.041 |
| letter | PFoil:ovr | 88.14% | 7100.6 | 16.873 | 0.452 |
| letter | PFoil:pw_both:accuracy_vote | 91.88% | 18456.8 | 18.349 | 6.103 |
| letter | PFoil:pw_both:vote | 92.01% | 18456.8 | 18.349 | 4.782 |
| letter | PFoil:pw_both:weighted_vote | 92.34% | 18456.8 | 18.349 | 44.204 |
| letter | PFoil:pw_larger:accuracy_vote | 89.43% | 9317.6 | 10.457 | 2.979 |
| letter | PFoil:pw_larger:vote | 89.75% | 9317.6 | 10.457 | 2.365 |
| letter | PFoil:pw_larger:weighted_vote | 26.68% | 9317.6 | 10.457 | 21.545 |
| letter | PFoil:pw_smaller:accuracy_vote | 89.16% | 9141.8 | 10.568 | 3.021 |
| letter | PFoil:pw_smaller:vote | 89.12% | 9141.8 | 10.568 | 2.379 |
| letter | PFoil:pw_smaller:weighted_vote | 19.82% | 9141.8 | 10.568 | 22.514 |
| letter | PFossil:ordered | 77.29% | 1170.6 | 2.681 | 0.011 |
| letter | PFossil:ovr | 75.67% | 1249.6 | 2.850 | 0.055 |
| letter | PFossil:pw_both:accuracy_vote | 86.04% | 11414.6 | 13.456 | 5.342 |
| letter | PFossil:pw_both:vote | 85.99% | 11414.6 | 13.456 | 4.085 |
| letter | PFossil:pw_both:weighted_vote | 86.30% | 11414.6 | 13.456 | 39.383 |
| letter | PFossil:pw_larger:accuracy_vote | 84.05% | 5755.2 | 8.709 | 2.620 |
| letter | PFossil:pw_larger:vote | 83.32% | 5755.2 | 8.709 | 2.031 |
| letter | PFossil:pw_larger:weighted_vote | 26.63% | 5755.2 | 8.709 | 19.498 |
| letter | PFossil:pw_smaller:accuracy_vote | 83.11% | 5675.0 | 8.760 | 2.658 |
| letter | PFossil:pw_smaller:vote | 82.84% | 5675.0 | 8.760 | 2.047 |
| letter | PFossil:pw_smaller:weighted_vote | 20.05% | 5675.0 | 8.760 | 20.423 |
| letter | Pypper:ordered | 81.39% | 1272.4 | 15.436 | 0.014 |
| letter | Pypper:ovr | 77.00% | 1662.0 | 25.851 | 0.089 |
| letter | Pypper:pw_both:accuracy_vote | 90.03% | 8894.4 | 55.329 | 4.991 |
| letter | Pypper:pw_both:vote | 89.98% | 8894.4 | 55.329 | 3.776 |
| letter | Pypper:pw_both:weighted_vote | 89.71% | 8894.4 | 55.329 | 38.202 |
| letter | Pypper:pw_larger:accuracy_vote | 88.29% | 4482.0 | 29.823 | 2.478 |
| letter | Pypper:pw_larger:vote | 88.28% | 4482.0 | 29.823 | 1.891 |
| letter | Pypper:pw_larger:weighted_vote | 27.89% | 4482.0 | 29.823 | 18.830 |
| letter | Pypper:pw_smaller:accuracy_vote | 87.93% | 4411.2 | 29.111 | 2.514 |
| letter | Pypper:pw_smaller:vote | 87.93% | 4411.2 | 29.111 | 1.914 |
| letter | Pypper:pw_smaller:weighted_vote | 22.22% | 4411.2 | 29.111 | 19.862 |
| optdigits | PFoil:ordered | 88.74% | 423.3 | 1.293 | 0.004 |
| optdigits | PFoil:ovr | 93.35% | 692.6 | 1.547 | 0.014 |
| optdigits | PFoil:pw_both:accuracy_vote | 96.62% | 1076.3 | 3.039 | 0.117 |
| optdigits | PFoil:pw_both:vote | 96.55% | 1076.3 | 3.039 | 0.094 |
| optdigits | PFoil:pw_both:weighted_vote | 96.78% | 1076.3 | 3.039 | 0.844 |
| optdigits | PFoil:pw_larger:accuracy_vote | 94.57% | 537.7 | 2.434 | 0.061 |
| optdigits | PFoil:pw_larger:vote | 94.59% | 537.7 | 2.434 | 0.048 |
| optdigits | PFoil:pw_larger:weighted_vote | 32.03% | 537.7 | 2.434 | 0.404 |
| optdigits | PFoil:pw_smaller:accuracy_vote | 94.18% | 539.5 | 2.373 | 0.060 |
| optdigits | PFoil:pw_smaller:vote | 94.40% | 539.5 | 2.373 | 0.049 |
| optdigits | PFoil:pw_smaller:weighted_vote | 43.43% | 539.5 | 2.373 | 0.436 |
| optdigits | PFossil:ordered | 90.73% | 300.1 | 1.043 | 0.003 |
| optdigits | PFossil:ovr | 90.28% | 460.0 | 0.989 | 0.007 |
| optdigits | PFossil:pw_both:accuracy_vote | 95.66% | 1097.1 | 3.157 | 0.117 |
| optdigits | PFossil:pw_both:vote | 95.50% | 1097.1 | 3.157 | 0.092 |
| optdigits | PFossil:pw_both:weighted_vote | 95.85% | 1097.1 | 3.157 | 0.806 |
| optdigits | PFossil:pw_larger:accuracy_vote | 93.91% | 540.3 | 2.437 | 0.059 |
| optdigits | PFossil:pw_larger:vote | 93.91% | 540.3 | 2.437 | 0.047 |
| optdigits | PFossil:pw_larger:weighted_vote | 33.01% | 540.3 | 2.437 | 0.387 |
| optdigits | PFossil:pw_smaller:accuracy_vote | 93.91% | 558.9 | 2.460 | 0.062 |
| optdigits | PFossil:pw_smaller:vote | 93.99% | 558.9 | 2.460 | 0.048 |
| optdigits | PFossil:pw_smaller:weighted_vote | 40.18% | 558.9 | 2.460 | 0.422 |
| optdigits | Pypper:ordered | 90.91% | 199.5 | 2.534 | 0.002 |
| optdigits | Pypper:ovr | 90.18% | 301.1 | 3.539 | 0.006 |
| optdigits | Pypper:pw_both:accuracy_vote | 95.89% | 625.8 | 6.654 | 0.104 |
| optdigits | Pypper:pw_both:vote | 96.00% | 625.8 | 6.654 | 0.078 |
| optdigits | Pypper:pw_both:weighted_vote | 95.78% | 625.8 | 6.654 | 0.748 |
| optdigits | Pypper:pw_larger:accuracy_vote | 94.40% | 310.0 | 4.222 | 0.052 |
| optdigits | Pypper:pw_larger:vote | 94.07% | 310.0 | 4.222 | 0.041 |
| optdigits | Pypper:pw_larger:weighted_vote | 33.02% | 310.0 | 4.222 | 0.376 |
| optdigits | Pypper:pw_smaller:accuracy_vote | 94.64% | 315.9 | 4.203 | 0.053 |
| optdigits | Pypper:pw_smaller:vote | 94.54% | 315.9 | 4.203 | 0.041 |
| optdigits | Pypper:pw_smaller:weighted_vote | 44.31% | 315.9 | 4.203 | 0.378 |
| primary-tumor | PFoil:ordered | 9.45% | 223.8 | 0.113 | 0.002 |
| primary-tumor | PFoil:ovr | 43.35% | 329.0 | 0.148 | 0.004 |
| primary-tumor | PFoil:pw_both:accuracy_vote | 42.79% | 1259.7 | 0.800 | 0.050 |
| primary-tumor | PFoil:pw_both:vote | 44.25% | 1259.7 | 0.800 | 0.044 |
| primary-tumor | PFoil:pw_both:weighted_vote | 33.62% | 1259.7 | 0.800 | 0.241 |
| primary-tumor | PFoil:pw_larger:accuracy_vote | 41.32% | 637.6 | 0.467 | 0.027 |
| primary-tumor | PFoil:pw_larger:vote | 35.12% | 637.6 | 0.467 | 0.024 |
| primary-tumor | PFoil:pw_larger:weighted_vote | 24.78% | 637.6 | 0.467 | 0.129 |
| primary-tumor | PFoil:pw_smaller:accuracy_vote | 42.17% | 620.8 | 0.423 | 0.024 |
| primary-tumor | PFoil:pw_smaller:vote | 36.28% | 620.8 | 0.423 | 0.020 |
| primary-tumor | PFoil:pw_smaller:weighted_vote | 41.86% | 620.8 | 0.423 | 0.115 |
| primary-tumor | PFossil:ordered | 32.44% | 330.7 | 0.138 | 0.002 |
| primary-tumor | PFossil:ovr | 43.38% | 341.1 | 0.137 | 0.002 |
| primary-tumor | PFossil:pw_both:accuracy_vote | 46.02% | 1464.9 | 0.803 | 0.053 |
| primary-tumor | PFossil:pw_both:vote | 37.16% | 1464.9 | 0.803 | 0.047 |
| primary-tumor | PFossil:pw_both:weighted_vote | 35.08% | 1464.9 | 0.803 | 0.257 |
| primary-tumor | PFossil:pw_larger:accuracy_vote | 39.84% | 652.7 | 0.438 | 0.028 |
| primary-tumor | PFossil:pw_larger:vote | 28.61% | 652.7 | 0.438 | 0.025 |
| primary-tumor | PFossil:pw_larger:weighted_vote | 24.78% | 652.7 | 0.438 | 0.130 |
| primary-tumor | PFossil:pw_smaller:accuracy_vote | 39.22% | 813.1 | 0.452 | 0.024 |
| primary-tumor | PFossil:pw_smaller:vote | 38.62% | 813.1 | 0.452 | 0.022 |
| primary-tumor | PFossil:pw_smaller:weighted_vote | 41.59% | 813.1 | 0.452 | 0.103 |
| primary-tumor | Pypper:ordered | 39.25% | 20.5 | 0.130 | 0.000 |
| primary-tumor | Pypper:ovr | 38.65% | 21.9 | 0.124 | 0.000 |
| primary-tumor | Pypper:pw_both:accuracy_vote | 47.20% | 260.0 | 2.153 | 0.030 |
| primary-tumor | Pypper:pw_both:vote | 44.55% | 260.0 | 2.153 | 0.023 |
| primary-tumor | Pypper:pw_both:weighted_vote | 27.15% | 260.0 | 2.153 | 0.150 |
| primary-tumor | Pypper:pw_larger:accuracy_vote | 43.97% | 139.7 | 1.353 | 0.016 |
| primary-tumor | Pypper:pw_larger:vote | 0.00% | 139.7 | 1.353 | 0.013 |
| primary-tumor | Pypper:pw_larger:weighted_vote | 24.78% | 139.7 | 1.353 | 0.078 |
| primary-tumor | Pypper:pw_smaller:accuracy_vote | 43.97% | 120.9 | 0.887 | 0.014 |
| primary-tumor | Pypper:pw_smaller:vote | 43.07% | 120.9 | 0.887 | 0.011 |
| primary-tumor | Pypper:pw_smaller:weighted_vote | 33.07% | 120.9 | 0.887 | 0.088 |
| segment | PFoil:ordered | 84.29% | 134.7 | 0.136 | 0.001 |
| segment | PFoil:ovr | 95.24% | 292.8 | 0.240 | 0.004 |
| segment | PFoil:pw_both:accuracy_vote | 94.29% | 296.6 | 0.288 | 0.023 |
| segment | PFoil:pw_both:vote | 93.29% | 296.6 | 0.288 | 0.018 |
| segment | PFoil:pw_both:weighted_vote | 96.32% | 296.6 | 0.288 | 0.145 |
| segment | PFoil:pw_larger:accuracy_vote | 90.35% | 135.0 | 0.168 | 0.013 |
| segment | PFoil:pw_larger:vote | 88.74% | 135.0 | 0.168 | 0.010 |
| segment | PFoil:pw_larger:weighted_vote | 44.16% | 135.0 | 0.168 | 0.067 |
| segment | PFoil:pw_smaller:accuracy_vote | 90.35% | 135.0 | 0.168 | 0.012 |
| segment | PFoil:pw_smaller:vote | 88.74% | 135.0 | 0.168 | 0.011 |
| segment | PFoil:pw_smaller:weighted_vote | 44.16% | 135.0 | 0.168 | 0.065 |
| segment | PFossil:ordered | 94.85% | 75.5 | 0.095 | 0.001 |
| segment | PFossil:ovr | 93.33% | 121.8 | 0.116 | 0.002 |
| segment | PFossil:pw_both:accuracy_vote | 95.15% | 180.8 | 0.225 | 0.022 |
| segment | PFossil:pw_both:vote | 95.32% | 180.8 | 0.225 | 0.017 |
| segment | PFossil:pw_both:weighted_vote | 95.54% | 180.8 | 0.225 | 0.131 |
| segment | PFossil:pw_larger:accuracy_vote | 94.81% | 103.4 | 0.149 | 0.012 |
| segment | PFossil:pw_larger:vote | 94.46% | 103.4 | 0.149 | 0.009 |
| segment | PFossil:pw_larger:weighted_vote | 47.06% | 103.4 | 0.149 | 0.064 |
| segment | PFossil:pw_smaller:accuracy_vote | 94.81% | 103.4 | 0.151 | 0.011 |
| segment | PFossil:pw_smaller:vote | 94.46% | 103.4 | 0.151 | 0.009 |
| segment | PFossil:pw_smaller:weighted_vote | 47.06% | 103.4 | 0.151 | 0.060 |
| segment | Pypper:ordered | 94.72% | 45.6 | 0.280 | 0.001 |
| segment | Pypper:ovr | 93.46% | 77.9 | 0.499 | 0.001 |
| segment | Pypper:pw_both:accuracy_vote | 95.41% | 127.3 | 0.787 | 0.020 |
| segment | Pypper:pw_both:vote | 95.54% | 127.3 | 0.787 | 0.015 |
| segment | Pypper:pw_both:weighted_vote | 95.37% | 127.3 | 0.787 | 0.121 |
| segment | Pypper:pw_larger:accuracy_vote | 94.76% | 62.8 | 0.389 | 0.012 |
| segment | Pypper:pw_larger:vote | 94.68% | 62.8 | 0.389 | 0.010 |
| segment | Pypper:pw_larger:weighted_vote | 52.73% | 62.8 | 0.389 | 0.060 |
| segment | Pypper:pw_smaller:accuracy_vote | 94.76% | 62.8 | 0.391 | 0.011 |
| segment | Pypper:pw_smaller:vote | 94.68% | 62.8 | 0.391 | 0.009 |
| segment | Pypper:pw_smaller:weighted_vote | 52.73% | 62.8 | 0.391 | 0.063 |
| solar-flare | PFoil:ordered | 58.72% | 241.1 | 0.122 | 0.002 |
| solar-flare | PFoil:ovr | 73.73% | 403.4 | 0.195 | 0.005 |
| solar-flare | PFoil:pw_both:accuracy_vote | 72.51% | 694.9 | 0.332 | 0.014 |
| solar-flare | PFoil:pw_both:vote | 68.01% | 694.9 | 0.332 | 0.013 |
| solar-flare | PFoil:pw_both:weighted_vote | 73.45% | 694.9 | 0.332 | 0.069 |
| solar-flare | PFoil:pw_larger:accuracy_vote | 68.11% | 411.2 | 0.200 | 0.008 |
| solar-flare | PFoil:pw_larger:vote | 67.64% | 411.2 | 0.200 | 0.007 |
| solar-flare | PFoil:pw_larger:weighted_vote | 56.66% | 411.2 | 0.200 | 0.045 |
| solar-flare | PFoil:pw_smaller:accuracy_vote | 59.19% | 283.7 | 0.147 | 0.006 |
| solar-flare | PFoil:pw_smaller:vote | 60.60% | 283.7 | 0.147 | 0.006 |
| solar-flare | PFoil:pw_smaller:weighted_vote | 62.11% | 283.7 | 0.147 | 0.029 |
| solar-flare | PFossil:ordered | 70.64% | 50.0 | 0.036 | 0.000 |
| solar-flare | PFossil:ovr | 74.02% | 62.5 | 0.042 | 0.001 |
| solar-flare | PFossil:pw_both:accuracy_vote | 73.83% | 161.9 | 0.110 | 0.008 |
| solar-flare | PFossil:pw_both:vote | 74.30% | 161.9 | 0.110 | 0.006 |
| solar-flare | PFossil:pw_both:weighted_vote | 74.77% | 161.9 | 0.110 | 0.043 |
| solar-flare | PFossil:pw_larger:accuracy_vote | 74.77% | 73.0 | 0.055 | 0.004 |
| solar-flare | PFossil:pw_larger:vote | 73.93% | 73.0 | 0.055 | 0.003 |
| solar-flare | PFossil:pw_larger:weighted_vote | 58.82% | 73.0 | 0.055 | 0.022 |
| solar-flare | PFossil:pw_smaller:accuracy_vote | 72.23% | 88.9 | 0.065 | 0.004 |
| solar-flare | PFossil:pw_smaller:vote | 72.33% | 88.9 | 0.065 | 0.003 |
| solar-flare | PFossil:pw_smaller:weighted_vote | 71.39% | 88.9 | 0.065 | 0.022 |
| solar-flare | Pypper:ordered | 74.77% | 8.3 | 0.096 | 0.000 |
| solar-flare | Pypper:ovr | 69.43% | 21.4 | 0.160 | 0.000 |
| solar-flare | Pypper:pw_both:accuracy_vote | 73.92% | 54.3 | 0.499 | 0.007 |
| solar-flare | Pypper:pw_both:vote | 74.40% | 54.3 | 0.499 | 0.005 |
| solar-flare | Pypper:pw_both:weighted_vote | 74.40% | 54.3 | 0.499 | 0.041 |
| solar-flare | Pypper:pw_larger:accuracy_vote | 72.89% | 27.5 | 0.291 | 0.004 |
| solar-flare | Pypper:pw_larger:vote | 72.70% | 27.5 | 0.291 | 0.003 |
| solar-flare | Pypper:pw_larger:weighted_vote | 55.90% | 27.5 | 0.291 | 0.021 |
| solar-flare | Pypper:pw_smaller:accuracy_vote | 74.68% | 26.8 | 0.218 | 0.004 |
| solar-flare | Pypper:pw_smaller:vote | 74.77% | 26.8 | 0.218 | 0.003 |
| solar-flare | Pypper:pw_smaller:weighted_vote | 71.11% | 26.8 | 0.218 | 0.020 |
| soybean | PFoil:ordered | 85.65% | 83.5 | 0.083 | 0.001 |
| soybean | PFoil:ovr | 90.92% | 153.8 | 0.096 | 0.002 |
| soybean | PFoil:pw_both:accuracy_vote | 92.97% | 446.2 | 0.466 | 0.056 |
| soybean | PFoil:pw_both:vote | 93.41% | 446.2 | 0.466 | 0.050 |
| soybean | PFoil:pw_both:weighted_vote | 62.22% | 446.2 | 0.466 | 0.282 |
| soybean | PFoil:pw_larger:accuracy_vote | 92.68% | 223.6 | 0.297 | 0.027 |
| soybean | PFoil:pw_larger:vote | 92.83% | 223.6 | 0.297 | 0.023 |
| soybean | PFoil:pw_larger:weighted_vote | 40.12% | 223.6 | 0.297 | 0.135 |
| soybean | PFoil:pw_smaller:accuracy_vote | 91.80% | 230.8 | 0.300 | 0.026 |
| soybean | PFoil:pw_smaller:vote | 90.92% | 230.8 | 0.300 | 0.021 |
| soybean | PFoil:pw_smaller:weighted_vote | 74.81% | 230.8 | 0.300 | 0.137 |
| soybean | PFossil:ordered | 92.98% | 70.2 | 0.078 | 0.001 |
| soybean | PFossil:ovr | 92.24% | 135.4 | 0.083 | 0.001 |
| soybean | PFossil:pw_both:accuracy_vote | 92.39% | 418.9 | 0.449 | 0.052 |
| soybean | PFossil:pw_both:vote | 93.12% | 418.9 | 0.449 | 0.042 |
| soybean | PFossil:pw_both:weighted_vote | 61.65% | 418.9 | 0.449 | 0.257 |
| soybean | PFossil:pw_larger:accuracy_vote | 93.26% | 207.0 | 0.293 | 0.026 |
| soybean | PFossil:pw_larger:vote | 93.70% | 207.0 | 0.293 | 0.021 |
| soybean | PFossil:pw_larger:weighted_vote | 37.02% | 207.0 | 0.293 | 0.152 |
| soybean | PFossil:pw_smaller:accuracy_vote | 93.41% | 201.5 | 0.284 | 0.025 |
| soybean | PFossil:pw_smaller:vote | 93.27% | 201.5 | 0.284 | 0.020 |
| soybean | PFossil:pw_smaller:weighted_vote | 75.25% | 201.5 | 0.284 | 0.135 |
| soybean | Pypper:ordered | 90.93% | 45.9 | 0.224 | 0.001 |
| soybean | Pypper:ovr | 89.91% | 58.8 | 0.261 | 0.001 |
| soybean | Pypper:pw_both:accuracy_vote | 93.12% | 373.4 | 1.882 | 0.052 |
| soybean | Pypper:pw_both:vote | 92.83% | 373.4 | 1.882 | 0.041 |
| soybean | Pypper:pw_both:weighted_vote | 61.79% | 373.4 | 1.882 | 0.281 |
| soybean | Pypper:pw_larger:accuracy_vote | 93.42% | 185.4 | 1.015 | 0.026 |
| soybean | Pypper:pw_larger:vote | 90.94% | 185.4 | 1.015 | 0.021 |
| soybean | Pypper:pw_larger:weighted_vote | 40.56% | 185.4 | 1.015 | 0.147 |
| soybean | Pypper:pw_smaller:accuracy_vote | 91.95% | 188.0 | 1.013 | 0.025 |
| soybean | Pypper:pw_smaller:vote | 91.80% | 188.0 | 1.013 | 0.021 |
| soybean | Pypper:pw_smaller:weighted_vote | 73.79% | 188.0 | 1.013 | 0.132 |
| texture | PFoil:ordered | 84.27% | 469.9 | 1.026 | 0.004 |
| texture | PFoil:ovr | 94.51% | 777.3 | 1.384 | 0.016 |
| texture | PFoil:pw_both:accuracy_vote | 94.95% | 903.5 | 1.889 | 0.127 |
| texture | PFoil:pw_both:vote | 94.87% | 903.5 | 1.889 | 0.102 |
| texture | PFoil:pw_both:weighted_vote | 96.04% | 903.5 | 1.889 | 0.871 |
| texture | PFoil:pw_larger:accuracy_vote | 92.85% | 460.8 | 1.409 | 0.066 |
| texture | PFoil:pw_larger:vote | 91.95% | 460.8 | 1.409 | 0.050 |
| texture | PFoil:pw_larger:weighted_vote | 31.51% | 460.8 | 1.409 | 0.423 |
| texture | PFoil:pw_smaller:accuracy_vote | 92.85% | 460.8 | 1.403 | 0.066 |
| texture | PFoil:pw_smaller:vote | 91.95% | 460.8 | 1.403 | 0.052 |
| texture | PFoil:pw_smaller:weighted_vote | 31.51% | 460.8 | 1.403 | 0.437 |
| texture | PFossil:ordered | 90.45% | 219.7 | 0.670 | 0.002 |
| texture | PFossil:ovr | 92.22% | 385.6 | 0.731 | 0.011 |
| texture | PFossil:pw_both:accuracy_vote | 94.05% | 674.9 | 1.737 | 0.119 |
| texture | PFossil:pw_both:vote | 93.58% | 674.9 | 1.737 | 0.091 |
| texture | PFossil:pw_both:weighted_vote | 94.58% | 674.9 | 1.737 | 0.810 |
| texture | PFossil:pw_larger:accuracy_vote | 92.35% | 315.6 | 1.307 | 0.069 |
| texture | PFossil:pw_larger:vote | 91.71% | 315.6 | 1.307 | 0.046 |
| texture | PFossil:pw_larger:weighted_vote | 30.64% | 315.6 | 1.307 | 0.437 |
| texture | PFossil:pw_smaller:accuracy_vote | 92.35% | 315.6 | 1.310 | 0.069 |
| texture | PFossil:pw_smaller:vote | 91.71% | 315.6 | 1.310 | 0.046 |
| texture | PFossil:pw_smaller:weighted_vote | 30.64% | 315.6 | 1.310 | 0.384 |
| texture | Pypper:ordered | 91.56% | 188.5 | 2.202 | 0.002 |
| texture | Pypper:ovr | 91.78% | 309.9 | 3.606 | 0.006 |
| texture | Pypper:pw_both:accuracy_vote | 95.33% | 486.8 | 4.444 | 0.116 |
| texture | Pypper:pw_both:vote | 94.91% | 486.8 | 4.444 | 0.090 |
| texture | Pypper:pw_both:weighted_vote | 95.15% | 486.8 | 4.444 | 0.769 |
| texture | Pypper:pw_larger:accuracy_vote | 94.53% | 252.7 | 2.789 | 0.058 |
| texture | Pypper:pw_larger:vote | 94.38% | 252.7 | 2.789 | 0.044 |
| texture | Pypper:pw_larger:weighted_vote | 32.73% | 252.7 | 2.789 | 0.411 |
| texture | Pypper:pw_smaller:accuracy_vote | 94.53% | 252.7 | 2.794 | 0.063 |
| texture | Pypper:pw_smaller:vote | 94.38% | 252.7 | 2.794 | 0.048 |
| texture | Pypper:pw_smaller:weighted_vote | 32.73% | 252.7 | 2.794 | 0.392 |
| vowel | PFoil:ordered | 68.99% | 278.7 | 0.164 | 0.002 |
| vowel | PFoil:ovr | 81.82% | 465.9 | 0.259 | 0.004 |
| vowel | PFoil:pw_both:accuracy_vote | 87.27% | 790.4 | 0.515 | 0.033 |
| vowel | PFoil:pw_both:vote | 86.57% | 790.4 | 0.515 | 0.030 |
| vowel | PFoil:pw_both:weighted_vote | 87.47% | 790.4 | 0.515 | 0.183 |
| vowel | PFoil:pw_larger:accuracy_vote | 81.01% | 390.2 | 0.282 | 0.017 |
| vowel | PFoil:pw_larger:vote | 81.41% | 390.2 | 0.282 | 0.015 |
| vowel | PFoil:pw_larger:weighted_vote | 38.89% | 390.2 | 0.282 | 0.091 |
| vowel | PFoil:pw_smaller:accuracy_vote | 81.01% | 390.2 | 0.282 | 0.017 |
| vowel | PFoil:pw_smaller:vote | 81.41% | 390.2 | 0.282 | 0.015 |
| vowel | PFoil:pw_smaller:weighted_vote | 38.89% | 390.2 | 0.282 | 0.089 |
| vowel | PFossil:ordered | 80.40% | 247.7 | 0.148 | 0.002 |
| vowel | PFossil:ovr | 77.27% | 401.3 | 0.222 | 0.003 |
| vowel | PFossil:pw_both:accuracy_vote | 83.54% | 744.6 | 0.492 | 0.034 |
| vowel | PFossil:pw_both:vote | 81.82% | 744.6 | 0.492 | 0.030 |
| vowel | PFossil:pw_both:weighted_vote | 82.22% | 744.6 | 0.492 | 0.178 |
| vowel | PFossil:pw_larger:accuracy_vote | 82.73% | 372.1 | 0.275 | 0.018 |
| vowel | PFossil:pw_larger:vote | 81.31% | 372.1 | 0.275 | 0.014 |
| vowel | PFossil:pw_larger:weighted_vote | 35.96% | 372.1 | 0.275 | 0.087 |
| vowel | PFossil:pw_smaller:accuracy_vote | 82.73% | 372.1 | 0.276 | 0.017 |
| vowel | PFossil:pw_smaller:vote | 81.31% | 372.1 | 0.276 | 0.014 |
| vowel | PFossil:pw_smaller:weighted_vote | 35.96% | 372.1 | 0.276 | 0.088 |
| vowel | Pypper:ordered | 69.70% | 123.4 | 0.477 | 0.001 |
| vowel | Pypper:ovr | 63.84% | 171.5 | 0.657 | 0.002 |
| vowel | Pypper:pw_both:accuracy_vote | 79.70% | 509.1 | 2.063 | 0.030 |
| vowel | Pypper:pw_both:vote | 78.99% | 509.1 | 2.063 | 0.027 |
| vowel | Pypper:pw_both:weighted_vote | 78.48% | 509.1 | 2.063 | 0.167 |
| vowel | Pypper:pw_larger:accuracy_vote | 77.88% | 254.8 | 1.026 | 0.015 |
| vowel | Pypper:pw_larger:vote | 78.59% | 254.8 | 1.026 | 0.013 |
| vowel | Pypper:pw_larger:weighted_vote | 37.07% | 254.8 | 1.026 | 0.083 |
| vowel | Pypper:pw_smaller:accuracy_vote | 77.88% | 254.8 | 1.033 | 0.015 |
| vowel | Pypper:pw_smaller:vote | 78.59% | 254.8 | 1.033 | 0.013 |
| vowel | Pypper:pw_smaller:weighted_vote | 37.07% | 254.8 | 1.033 | 0.083 |
| yeast | PFoil:ordered | 23.99% | 756.7 | 0.404 | 0.005 |
| yeast | PFoil:ovr | 56.47% | 1247.8 | 0.644 | 0.013 |
| yeast | PFoil:pw_both:accuracy_vote | 54.99% | 2567.3 | 1.303 | 0.054 |
| yeast | PFoil:pw_both:vote | 52.70% | 2567.3 | 1.303 | 0.049 |
| yeast | PFoil:pw_both:weighted_vote | 52.30% | 2567.3 | 1.303 | 0.320 |
| yeast | PFoil:pw_larger:accuracy_vote | 48.05% | 1431.1 | 0.748 | 0.030 |
| yeast | PFoil:pw_larger:vote | 46.90% | 1431.1 | 0.748 | 0.028 |
| yeast | PFoil:pw_larger:weighted_vote | 35.92% | 1431.1 | 0.748 | 0.187 |
| yeast | PFoil:pw_smaller:accuracy_vote | 42.79% | 1136.2 | 0.606 | 0.024 |
| yeast | PFoil:pw_smaller:vote | 38.89% | 1136.2 | 0.606 | 0.022 |
| yeast | PFoil:pw_smaller:weighted_vote | 57.14% | 1136.2 | 0.606 | 0.129 |
| yeast | PFossil:ordered | 56.06% | 148.8 | 0.110 | 0.001 |
| yeast | PFossil:ovr | 57.48% | 180.5 | 0.117 | 0.002 |
| yeast | PFossil:pw_both:accuracy_vote | 59.57% | 898.5 | 0.510 | 0.034 |
| yeast | PFossil:pw_both:vote | 59.77% | 898.5 | 0.510 | 0.029 |
| yeast | PFossil:pw_both:weighted_vote | 51.76% | 898.5 | 0.510 | 0.203 |
| yeast | PFossil:pw_larger:accuracy_vote | 57.28% | 337.3 | 0.238 | 0.018 |
| yeast | PFossil:pw_larger:vote | 56.54% | 337.3 | 0.238 | 0.014 |
| yeast | PFossil:pw_larger:weighted_vote | 35.37% | 337.3 | 0.238 | 0.115 |
| yeast | PFossil:pw_smaller:accuracy_vote | 57.34% | 561.2 | 0.314 | 0.017 |
| yeast | PFossil:pw_smaller:vote | 57.48% | 561.2 | 0.314 | 0.014 |
| yeast | PFossil:pw_smaller:weighted_vote | 58.49% | 561.2 | 0.314 | 0.086 |
| yeast | Pypper:ordered | 58.96% | 38.7 | 0.358 | 0.001 |
| yeast | Pypper:ovr | 59.30% | 66.8 | 0.556 | 0.001 |
| yeast | Pypper:pw_both:accuracy_vote | 59.16% | 222.1 | 1.843 | 0.028 |
| yeast | Pypper:pw_both:vote | 58.89% | 222.1 | 1.843 | 0.022 |
| yeast | Pypper:pw_both:weighted_vote | 49.73% | 222.1 | 1.843 | 0.170 |
| yeast | Pypper:pw_larger:accuracy_vote | 57.68% | 108.3 | 1.079 | 0.015 |
| yeast | Pypper:pw_larger:vote | 58.15% | 108.3 | 1.079 | 0.011 |
| yeast | Pypper:pw_larger:weighted_vote | 36.45% | 108.3 | 1.079 | 0.095 |
| yeast | Pypper:pw_smaller:accuracy_vote | 59.23% | 113.8 | 0.802 | 0.014 |
| yeast | Pypper:pw_smaller:vote | 58.42% | 113.8 | 0.802 | 0.010 |
| yeast | Pypper:pw_smaller:weighted_vote | 59.30% | 113.8 | 0.802 | 0.084 |
| zoo | PFoil:ordered | 92.00% | 15.8 | 0.010 | 0.000 |
| zoo | PFoil:ovr | 96.00% | 20.3 | 0.010 | 0.000 |
| zoo | PFoil:pw_both:accuracy_vote | 94.00% | 43.9 | 0.037 | 0.002 |
| zoo | PFoil:pw_both:vote | 95.00% | 43.9 | 0.037 | 0.002 |
| zoo | PFoil:pw_both:weighted_vote | 89.00% | 43.9 | 0.037 | 0.007 |
| zoo | PFoil:pw_larger:accuracy_vote | 94.00% | 21.9 | 0.023 | 0.001 |
| zoo | PFoil:pw_larger:vote | 95.00% | 21.9 | 0.023 | 0.001 |
| zoo | PFoil:pw_larger:weighted_vote | 60.36% | 21.9 | 0.023 | 0.004 |
| zoo | PFoil:pw_smaller:accuracy_vote | 93.09% | 22.0 | 0.022 | 0.001 |
| zoo | PFoil:pw_smaller:vote | 93.09% | 22.0 | 0.022 | 0.001 |
| zoo | PFoil:pw_smaller:weighted_vote | 93.09% | 22.0 | 0.022 | 0.004 |
| zoo | PFossil:ordered | 93.00% | 16.7 | 0.010 | 0.000 |
| zoo | PFossil:ovr | 95.00% | 21.0 | 0.010 | 0.000 |
| zoo | PFossil:pw_both:accuracy_vote | 94.00% | 45.6 | 0.038 | 0.002 |
| zoo | PFossil:pw_both:vote | 93.00% | 45.6 | 0.038 | 0.002 |
| zoo | PFossil:pw_both:weighted_vote | 90.00% | 45.6 | 0.038 | 0.007 |
| zoo | PFossil:pw_larger:accuracy_vote | 93.00% | 22.7 | 0.024 | 0.001 |
| zoo | PFossil:pw_larger:vote | 93.00% | 22.7 | 0.024 | 0.001 |
| zoo | PFossil:pw_larger:weighted_vote | 60.36% | 22.7 | 0.024 | 0.004 |
| zoo | PFossil:pw_smaller:accuracy_vote | 94.00% | 22.9 | 0.023 | 0.001 |
| zoo | PFossil:pw_smaller:vote | 94.00% | 22.9 | 0.023 | 0.001 |
| zoo | PFossil:pw_smaller:weighted_vote | 93.00% | 22.9 | 0.023 | 0.004 |
| zoo | Pypper:ordered | 88.09% | 8.0 | 0.033 | 0.000 |
| zoo | Pypper:ovr | 90.09% | 10.3 | 0.039 | 0.000 |
| zoo | Pypper:pw_both:accuracy_vote | 91.00% | 39.8 | 0.191 | 0.002 |
| zoo | Pypper:pw_both:vote | 91.00% | 39.8 | 0.191 | 0.002 |
| zoo | Pypper:pw_both:weighted_vote | 84.18% | 39.8 | 0.191 | 0.008 |
| zoo | Pypper:pw_larger:accuracy_vote | 91.00% | 19.8 | 0.102 | 0.001 |
| zoo | Pypper:pw_larger:vote | 90.09% | 19.8 | 0.102 | 0.001 |
| zoo | Pypper:pw_larger:weighted_vote | 60.36% | 19.8 | 0.102 | 0.004 |
| zoo | Pypper:pw_smaller:accuracy_vote | 91.00% | 20.0 | 0.098 | 0.001 |
| zoo | Pypper:pw_smaller:vote | 89.09% | 20.0 | 0.098 | 0.001 |
| zoo | Pypper:pw_smaller:weighted_vote | 88.09% | 20.0 | 0.098 | 0.004 |

## Why the weighted vote does poorly

The weighted vote is clearly worse than the plain vote here, and much slower to predict. Both have the same cause in how it works: a pair model's vote is weighted by the Laplace estimate of the rule that decided it, and split -- weight `p` to the class it predicts, `1 - p` to the other class of its pair. Prediction is slower because it has to find, for every example and every pair model, the deciding rule among all covering rules, instead of just taking each model's prediction.

Most of a pair model's decisions are made by its default rule: of the c(c-1)/2 pair models, only c - 1 have the example's own class as one of theirs, so most of them see an example of neither of their classes, and their rules rarely fire. pyrulearn annotates every rule with the training examples its body covers, independently of the other rules -- so a rule means the same in every model it is part of, and the counts don't change when a rule set is reordered or turned into a decision list. The default rule's body is empty ("true"), so it is counted over all of its pair's examples, and its Laplace estimate is simply its class's share of the pair. For a balanced pair that is about 0.5: the default votes are almost neutral, while a rule that fires -- rightly or wrongly -- votes with a weight near 1. With the larger class as the pair's target, the default is the smaller class and its share is below 0.5, so the split gives most of the default's vote to the *other* class, against the model's own prediction.

To check that this is the cause, the default rules' counts are swapped, for prediction only, for counts over the examples each default actually decides (those no rule covers), on 4 small datasets (`primary-tumor`, `soybean`, `vowel`, `segment`; the first two with very unbalanced, the last two with balanced classes), 3-fold. Mean accuracy:

| base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|--:|--:|--:|--:|
| PFoil | larger | 74.07% | 75.50% | 37.95% | 76.69% |
| PFoil | smaller | 74.47% | 74.68% | 51.64% | 77.28% |
| PFossil | larger | 73.94% | 76.25% | 36.04% | 76.02% |
| PFossil | smaller | 76.66% | 77.54% | 50.08% | 77.32% |
| Pypper | larger | 63.75% | 75.36% | 36.22% | 74.39% |
| Pypper | smaller | 74.84% | 74.74% | 47.51% | 72.74% |

Counted over the examples it decides, the default rule gets a weight near its real reliability, and the weighted vote comes up to the level of the other two -- but not above it. The two ways of counting the default are two different meanings of a rule's counts (a statement in its own right vs. its role in this particular model); pyrulearn keeps the first, so the weighted vote stays as it is. The result: when most decisions come from default rules, rule-based vote weights add little, and the plain or the accuracy-weighted vote is the better choice.

Per dataset:

| dataset | base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|---|--:|--:|--:|--:|
| primary-tumor | PFoil | larger | 36.28% | 40.71% | 24.78% | 41.59% |
| primary-tumor | PFoil | smaller | 36.87% | 35.99% | 44.25% | 46.90% |
| primary-tumor | PFossil | larger | 30.09% | 37.46% | 24.78% | 38.35% |
| primary-tumor | PFossil | smaller | 40.41% | 42.48% | 40.71% | 44.25% |
| primary-tumor | Pypper | larger | 0.29% | 44.25% | 24.78% | 40.71% |
| primary-tumor | Pypper | smaller | 41.89% | 42.48% | 32.15% | 38.05% |
| segment | PFoil | larger | 89.61% | 90.30% | 48.27% | 93.46% |
| segment | PFoil | smaller | 89.61% | 90.30% | 48.27% | 93.46% |
| segment | PFossil | larger | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | PFossil | smaller | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | Pypper | larger | 94.63% | 94.50% | 46.41% | 94.37% |
| segment | Pypper | smaller | 94.63% | 94.50% | 46.41% | 94.37% |
| soybean | PFoil | larger | 90.49% | 91.07% | 39.97% | 90.78% |
| soybean | PFoil | smaller | 91.51% | 92.53% | 75.25% | 87.85% |
| soybean | PFossil | larger | 91.95% | 92.39% | 37.19% | 91.22% |
| soybean | PFossil | smaller | 92.53% | 92.54% | 77.45% | 90.48% |
| soybean | Pypper | larger | 88.44% | 92.10% | 38.95% | 91.36% |
| soybean | Pypper | smaller | 91.22% | 91.37% | 76.72% | 87.41% |
| vowel | PFoil | larger | 79.90% | 79.90% | 38.79% | 80.91% |
| vowel | PFoil | smaller | 79.90% | 79.90% | 38.79% | 80.91% |
| vowel | PFossil | larger | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | PFossil | smaller | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | Pypper | larger | 71.62% | 70.61% | 34.75% | 71.11% |
| vowel | Pypper | smaller | 71.62% | 70.61% | 34.75% | 71.11% |

