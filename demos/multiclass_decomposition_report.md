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
Each fit is capped at 300s; a time-out counts as a failure, and it fails the whole
cross-validation of that learner on that dataset (`run_cv`'s `stop_on_timeout`): its remaining folds are
not run, and all its folds count as failures.
Training time is measured inside the worker process, so it excludes
shipping data and models between processes. Every learner runs single-threaded.
Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

## Pypper

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.12% | 2.12 | 430.9 | 1143.3 | 5.12 | 0.466 | 0 |
| pw_both:vote | 77.92% | 2.88 | 430.9 | 1143.3 | 5.12 | 0.352 | 0 |
| pw_smaller:accuracy_vote | 77.95% | 4.35 | 209.2 | 547.4 | 2.51 | 0.235 | 0 |
| pw_larger:accuracy_vote | 78.10% | 4.81 | 221.1 | 597.5 | 2.79 | 0.254 | 0 |
| pw_smaller:vote | 78.10% | 5.35 | 209.2 | 547.4 | 2.51 | 0.179 | 0 |
| pw_both:weighted_vote | 70.93% | 6.12 | 430.9 | 1143.3 | 5.12 | 3.807 | 0 |
| ordered | 76.22% | 6.31 | 39.1 | 172.4 | 1.19 | 0.001 | 0 |
| pw_larger:vote | 69.08% | 6.85 | 221.1 | 597.5 | 2.79 | 0.204 | 0 |
| ovr | 72.88% | 7.88 | 40.6 | 215.5 | 1.51 | 0.009 | 0 |
| pw_smaller:weighted_vote | 51.36% | 8.73 | 209.2 | 547.4 | 2.51 | 1.685 | 0 |
| pw_larger:weighted_vote | 40.14% | 10.62 | 221.1 | 597.5 | 2.79 | 2.132 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFossil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.15% | 2.77 | 569.5 | 1376.1 | 1.21 | 0.454 | 0 |
| pw_both:vote | 78.26% | 3.62 | 569.5 | 1376.1 | 1.21 | 0.355 | 0 |
| pw_smaller:accuracy_vote | 77.47% | 4.96 | 283.7 | 716.1 | 0.65 | 0.240 | 0 |
| pw_larger:accuracy_vote | 77.45% | 5.12 | 283.2 | 659.0 | 0.63 | 0.238 | 0 |
| pw_both:weighted_vote | 71.55% | 5.27 | 569.5 | 1376.1 | 1.21 | 3.572 | 0 |
| pw_smaller:vote | 77.13% | 5.65 | 283.7 | 716.1 | 0.65 | 0.205 | 0 |
| pw_larger:vote | 76.20% | 6.42 | 283.2 | 659.0 | 0.63 | 0.187 | 0 |
| ovr | 76.12% | 6.46 | 52.3 | 274.2 | 0.36 | 0.007 | 0 |
| ordered | 76.00% | 6.92 | 46.6 | 214.8 | 0.29 | 0.001 | 0 |
| pw_smaller:weighted_vote | 50.76% | 8.04 | 283.7 | 716.1 | 0.65 | 1.755 | 0 |
| pw_larger:weighted_vote | 38.57% | 10.77 | 283.2 | 659.0 | 0.63 | 1.868 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFoil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 78.94% | 2.65 | 1107.4 | 3893.0 | 3.15 | 0.639 | 0 |
| pw_both:vote | 77.88% | 3.27 | 1107.4 | 3893.0 | 3.15 | 0.535 | 0 |
| ovr | 78.91% | 3.69 | 326.5 | 2262.4 | 3.73 | 0.116 | 0 |
| pw_both:weighted_vote | 73.31% | 4.81 | 1107.4 | 3893.0 | 3.15 | 4.290 | 0 |
| pw_larger:accuracy_vote | 76.16% | 4.96 | 563.9 | 1997.5 | 1.58 | 0.312 | 0 |
| pw_larger:vote | 74.31% | 5.81 | 563.9 | 1997.5 | 1.58 | 0.299 | 0 |
| pw_smaller:accuracy_vote | 74.00% | 6.27 | 541.4 | 1894.5 | 1.55 | 0.320 | 0 |
| pw_smaller:vote | 73.58% | 6.58 | 541.4 | 1894.5 | 1.55 | 0.267 | 0 |
| pw_smaller:weighted_vote | 51.40% | 8.50 | 541.4 | 1894.5 | 1.55 | 1.927 | 0 |
| ordered | 64.12% | 8.85 | 217.4 | 1212.7 | 1.81 | 0.003 | 0 |
| pw_larger:weighted_vote | 39.23% | 10.62 | 563.9 | 1997.5 | 1.58 | 2.449 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## Training and prediction time over the number of classes

One point per dataset and base learner: the mean one-vs-rest time divided by the mean pairwise time. Above 1, pairwise is faster. The dashed line is what a learner whose time grows linearly with the number of examples would give.

![training time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_train_ratio.png)

![training time ratio, one-vs-rest / double round robin](multiclass_decomposition_plots/multiclass_decomposition_train_ratio_double.png)

![prediction time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_predict_ratio.png)

## Growth of the base learners and the pairwise speed-up

The argument for pairwise classification is about cost: if a learner's training time grows like n^b with the number of examples n, one-vs-rest costs c n^b for c classes, pairwise the sum over the pairs of (n_i + n_j)^b -- about (c - 1) n examples in all, but in many small problems, so the larger b, the more pairwise gains. Here b is measured for each base learner on each dataset with at least 1,000 training rows: one-vs-rest is fitted on nested, stratified subsamples of the first training fold (25%, 50%, 100% of it, outside the cross-validation, each with its own discretization; a learner that times out on one is not tried on the larger ones), and b is the least-squares slope of log time over log rows. The predicted ratio uses the fold's real class counts; the measured one is the cross-validation's one-vs-rest / pairwise (`pw_smaller`) training time. Pairwise also has costs per pair the model leaves out -- building each pair's data, evaluating its model for the vote weights -- which weigh most with many classes and few examples.

Run times of rule learners are noisy in a way their results are not: one rule more or less can change a fit's time a lot and its quality hardly at all, so b from three subsamples, and the ratios, scatter. The points follow the line only roughly; a controlled experiment, varying the training-set size on a single dataset, would separate the growth from the noise better.

![pairwise speed-up, measured vs. predicted](multiclass_decomposition_plots/multiclass_decomposition_growth.png)

| dataset | classes | training rows | base | b | ovr / pairwise, measured | predicted (b) | predicted (b = 1) |
|---|--:|--:|---|--:|--:|--:|--:|
| yeast | 10 | 1,335 | PFoil | 1.24 | 1.07 | 1.51 | 1.11 |
| yeast | 10 | 1,335 | PFossil | 0.36 | 0.43 | 0.43 | 1.11 |
| yeast | 10 | 1,335 | Pypper | 0.75 | 0.66 | 0.79 | 1.11 |
| segment | 7 | 2,079 | PFoil | 0.82 | 1.85 | 0.93 | 1.17 |
| segment | 7 | 2,079 | PFossil | 0.46 | 1.15 | 0.60 | 1.17 |
| segment | 7 | 2,079 | Pypper | 0.46 | 1.11 | 0.59 | 1.17 |
| led24 | 10 | 2,880 | PFoil | 1.37 | 1.54 | 2.00 | 1.11 |
| led24 | 10 | 2,880 | PFossil | 0.04 | 0.56 | 0.24 | 1.11 |
| led24 | 10 | 2,880 | Pypper | 0.51 | 0.66 | 0.50 | 1.11 |
| texture | 11 | 4,950 | PFoil | 1.22 | 1.74 | 1.60 | 1.10 |
| texture | 11 | 4,950 | PFossil | 0.74 | 1.11 | 0.70 | 1.10 |
| texture | 11 | 4,950 | Pypper | 0.88 | 1.31 | 0.90 | 1.10 |
| optdigits | 10 | 5,058 | PFoil | 0.96 | 1.39 | 1.04 | 1.11 |
| optdigits | 10 | 5,058 | PFossil | 0.64 | 0.85 | 0.63 | 1.11 |
| optdigits | 10 | 5,058 | Pypper | 0.96 | 0.91 | 1.05 | 1.11 |
| letter | 26 | 16,000 | PFoil | 1.36 | 1.85 | 2.62 | 1.04 |
| letter | 26 | 16,000 | PFossil | 0.37 | 0.47 | 0.21 | 1.04 |
| letter | 26 | 16,000 | Pypper | 1.28 | 0.83 | 2.12 | 1.04 |
| kropt | 18 | 22,444 | PFoil | 1.45 | 3.09 | 2.49 | 1.06 |
| kropt | 18 | 22,444 | PFossil | 0.57 | 0.36 | 0.44 | 1.06 |
| kropt | 18 | 22,444 | Pypper | 1.43 | 0.40 | 2.41 | 1.06 |

Median b: PFoil 1.24, PFossil 0.46, Pypper 0.88.

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_conditions | train_time | predict_time |
|---|---|---|---|---|---|
| audiology | PFoil:ordered | 62.91% | 57.1 | 0.056 | 0.000 |
| audiology | PFoil:ovr | 77.04% | 94.3 | 0.066 | 0.001 |
| audiology | PFoil:pw_both:accuracy_vote | 67.27% | 284.8 | 0.442 | 0.039 |
| audiology | PFoil:pw_both:vote | 61.52% | 284.8 | 0.442 | 0.023 |
| audiology | PFoil:pw_both:weighted_vote | 46.46% | 284.8 | 0.442 | 0.213 |
| audiology | PFoil:pw_larger:accuracy_vote | 70.42% | 139.0 | 0.233 | 0.018 |
| audiology | PFoil:pw_larger:vote | 58.46% | 139.0 | 0.233 | 0.014 |
| audiology | PFoil:pw_larger:weighted_vote | 46.46% | 139.0 | 0.233 | 0.108 |
| audiology | PFoil:pw_smaller:accuracy_vote | 65.08% | 145.8 | 0.244 | 0.019 |
| audiology | PFoil:pw_smaller:vote | 71.23% | 145.8 | 0.244 | 0.011 |
| audiology | PFoil:pw_smaller:weighted_vote | 54.80% | 145.8 | 0.244 | 0.117 |
| audiology | PFossil:ordered | 84.05% | 72.5 | 0.055 | 0.000 |
| audiology | PFossil:ovr | 77.89% | 115.2 | 0.070 | 0.001 |
| audiology | PFossil:pw_both:accuracy_vote | 81.40% | 586.0 | 0.482 | 0.038 |
| audiology | PFossil:pw_both:vote | 81.40% | 586.0 | 0.482 | 0.029 |
| audiology | PFossil:pw_both:weighted_vote | 46.46% | 586.0 | 0.482 | 0.251 |
| audiology | PFossil:pw_larger:accuracy_vote | 80.53% | 292.0 | 0.262 | 0.016 |
| audiology | PFossil:pw_larger:vote | 81.40% | 292.0 | 0.262 | 0.015 |
| audiology | PFossil:pw_larger:weighted_vote | 46.46% | 292.0 | 0.262 | 0.119 |
| audiology | PFossil:pw_smaller:accuracy_vote | 80.95% | 294.0 | 0.271 | 0.018 |
| audiology | PFossil:pw_smaller:vote | 81.38% | 294.0 | 0.271 | 0.016 |
| audiology | PFossil:pw_smaller:weighted_vote | 54.86% | 294.0 | 0.271 | 0.121 |
| audiology | Pypper:ordered | 75.28% | 16.0 | 0.132 | 0.000 |
| audiology | Pypper:ovr | 64.64% | 26.9 | 0.151 | 0.000 |
| audiology | Pypper:pw_both:accuracy_vote | 68.18% | 142.9 | 1.522 | 0.027 |
| audiology | Pypper:pw_both:vote | 57.04% | 142.9 | 1.522 | 0.019 |
| audiology | Pypper:pw_both:weighted_vote | 46.46% | 142.9 | 1.522 | 0.149 |
| audiology | Pypper:pw_larger:accuracy_vote | 69.51% | 87.7 | 0.848 | 0.019 |
| audiology | Pypper:pw_larger:vote | 7.92% | 87.7 | 0.848 | 0.013 |
| audiology | Pypper:pw_larger:weighted_vote | 46.46% | 87.7 | 0.848 | 0.116 |
| audiology | Pypper:pw_smaller:accuracy_vote | 65.51% | 55.4 | 0.712 | 0.017 |
| audiology | Pypper:pw_smaller:vote | 70.83% | 55.4 | 0.712 | 0.012 |
| audiology | Pypper:pw_smaller:weighted_vote | 48.64% | 55.4 | 0.712 | 0.104 |
| kropt | PFoil:ordered | 69.80% | 15119.2 | 29.805 | 0.039 |
| kropt | PFoil:ovr | 75.41% | 31233.6 | 65.018 | 2.166 |
| kropt | PFoil:pw_both:accuracy_vote | 82.10% | 47380.2 | 44.979 | 7.593 |
| kropt | PFoil:pw_both:vote | 81.88% | 47380.2 | 44.979 | 6.940 |
| kropt | PFoil:pw_both:weighted_vote | 64.03% | 47380.2 | 44.979 | 43.199 |
| kropt | PFoil:pw_larger:accuracy_vote | 75.46% | 24408.4 | 21.342 | 3.814 |
| kropt | PFoil:pw_larger:vote | 75.29% | 24408.4 | 21.342 | 4.032 |
| kropt | PFoil:pw_larger:weighted_vote | 32.70% | 24408.4 | 21.342 | 29.203 |
| kropt | PFoil:pw_smaller:accuracy_vote | 75.78% | 22971.8 | 21.045 | 3.960 |
| kropt | PFoil:pw_smaller:vote | 78.50% | 22971.8 | 21.045 | 3.359 |
| kropt | PFoil:pw_smaller:weighted_vote | 72.94% | 22971.8 | 21.045 | 15.344 |
| kropt | PFossil:ordered | 35.48% | 420.6 | 1.596 | 0.001 |
| kropt | PFossil:ovr | 24.99% | 342.6 | 1.263 | 0.040 |
| kropt | PFossil:pw_both:accuracy_vote | 40.70% | 5670.4 | 6.253 | 3.966 |
| kropt | PFossil:pw_both:vote | 39.41% | 5670.4 | 6.253 | 3.220 |
| kropt | PFossil:pw_both:weighted_vote | 32.21% | 5670.4 | 6.253 | 31.225 |
| kropt | PFossil:pw_larger:accuracy_vote | 36.74% | 2446.2 | 2.879 | 2.307 |
| kropt | PFossil:pw_larger:vote | 34.82% | 2446.2 | 2.879 | 1.737 |
| kropt | PFossil:pw_larger:weighted_vote | 22.80% | 2446.2 | 2.879 | 17.971 |
| kropt | PFossil:pw_smaller:accuracy_vote | 39.58% | 3224.2 | 3.543 | 2.330 |
| kropt | PFossil:pw_smaller:vote | 36.35% | 3224.2 | 3.543 | 2.093 |
| kropt | PFossil:pw_smaller:weighted_vote | 39.95% | 3224.2 | 3.543 | 13.278 |
| kropt | Pypper:ordered | 54.59% | 1299.2 | 12.921 | 0.005 |
| kropt | Pypper:ovr | 33.63% | 1130.0 | 10.901 | 0.086 |
| kropt | Pypper:pw_both:accuracy_vote | 66.17% | 11866.4 | 55.933 | 4.613 |
| kropt | Pypper:pw_both:vote | 67.00% | 11866.4 | 55.933 | 3.514 |
| kropt | Pypper:pw_both:weighted_vote | 50.76% | 11866.4 | 55.933 | 38.664 |
| kropt | Pypper:pw_larger:accuracy_vote | 62.95% | 6464.0 | 32.131 | 2.478 |
| kropt | Pypper:pw_larger:vote | 63.52% | 6464.0 | 32.131 | 2.012 |
| kropt | Pypper:pw_larger:weighted_vote | 30.84% | 6464.0 | 32.131 | 23.903 |
| kropt | Pypper:pw_smaller:accuracy_vote | 62.28% | 5402.4 | 27.269 | 2.474 |
| kropt | Pypper:pw_smaller:vote | 62.10% | 5402.4 | 27.269 | 1.867 |
| kropt | Pypper:pw_smaller:weighted_vote | 60.03% | 5402.4 | 27.269 | 14.406 |
| led24 | PFoil:ordered | 47.06% | 2042.6 | 1.045 | 0.004 |
| led24 | PFoil:ovr | 61.81% | 3508.9 | 1.895 | 0.031 |
| led24 | PFoil:pw_both:accuracy_vote | 64.12% | 5456.2 | 2.427 | 0.119 |
| led24 | PFoil:pw_both:vote | 63.56% | 5456.2 | 2.427 | 0.107 |
| led24 | PFoil:pw_both:weighted_vote | 69.28% | 5456.2 | 2.427 | 0.918 |
| led24 | PFoil:pw_larger:accuracy_vote | 58.09% | 2730.4 | 1.273 | 0.065 |
| led24 | PFoil:pw_larger:vote | 55.44% | 2730.4 | 1.273 | 0.059 |
| led24 | PFoil:pw_larger:weighted_vote | 29.87% | 2730.4 | 1.273 | 0.513 |
| led24 | PFoil:pw_smaller:accuracy_vote | 54.34% | 2723.0 | 1.230 | 0.072 |
| led24 | PFoil:pw_smaller:vote | 53.88% | 2723.0 | 1.230 | 0.062 |
| led24 | PFoil:pw_smaller:weighted_vote | 30.91% | 2723.0 | 1.230 | 0.522 |
| led24 | PFossil:ordered | 70.03% | 250.2 | 0.207 | 0.001 |
| led24 | PFossil:ovr | 70.00% | 272.8 | 0.249 | 0.004 |
| led24 | PFossil:pw_both:accuracy_vote | 70.88% | 1698.1 | 0.928 | 0.073 |
| led24 | PFossil:pw_both:vote | 71.47% | 1698.1 | 0.928 | 0.071 |
| led24 | PFossil:pw_both:weighted_vote | 71.41% | 1698.1 | 0.928 | 0.784 |
| led24 | PFossil:pw_larger:accuracy_vote | 66.44% | 891.4 | 0.511 | 0.046 |
| led24 | PFossil:pw_larger:vote | 66.84% | 891.4 | 0.511 | 0.036 |
| led24 | PFossil:pw_larger:weighted_vote | 28.56% | 891.4 | 0.511 | 0.336 |
| led24 | PFossil:pw_smaller:accuracy_vote | 67.31% | 812.8 | 0.441 | 0.044 |
| led24 | PFossil:pw_smaller:vote | 67.34% | 812.8 | 0.441 | 0.035 |
| led24 | PFossil:pw_smaller:weighted_vote | 30.75% | 812.8 | 0.441 | 0.331 |
| led24 | Pypper:ordered | 71.50% | 88.7 | 0.559 | 0.000 |
| led24 | Pypper:ovr | 68.00% | 124.0 | 0.802 | 0.002 |
| led24 | Pypper:pw_both:accuracy_vote | 72.31% | 500.9 | 2.239 | 0.073 |
| led24 | Pypper:pw_both:vote | 72.19% | 500.9 | 2.239 | 0.054 |
| led24 | Pypper:pw_both:weighted_vote | 72.28% | 500.9 | 2.239 | 0.567 |
| led24 | Pypper:pw_larger:accuracy_vote | 71.47% | 250.7 | 1.111 | 0.041 |
| led24 | Pypper:pw_larger:vote | 71.22% | 250.7 | 1.111 | 0.030 |
| led24 | Pypper:pw_larger:weighted_vote | 31.87% | 250.7 | 1.111 | 0.328 |
| led24 | Pypper:pw_smaller:accuracy_vote | 72.09% | 251.5 | 1.207 | 0.038 |
| led24 | Pypper:pw_smaller:vote | 71.97% | 251.5 | 1.207 | 0.026 |
| led24 | Pypper:pw_smaller:weighted_vote | 34.38% | 251.5 | 1.207 | 0.264 |
| letter | PFoil:ordered | 59.28% | 4562.0 | 6.509 | 0.009 |
| letter | PFoil:ovr | 88.14% | 7100.6 | 12.070 | 0.426 |
| letter | PFoil:pw_both:accuracy_vote | 91.88% | 18456.8 | 12.221 | 6.384 |
| letter | PFoil:pw_both:vote | 92.01% | 18456.8 | 12.221 | 4.919 |
| letter | PFoil:pw_both:weighted_vote | 92.34% | 18456.8 | 12.221 | 49.992 |
| letter | PFoil:pw_larger:accuracy_vote | 89.43% | 9317.6 | 6.382 | 3.003 |
| letter | PFoil:pw_larger:vote | 89.75% | 9317.6 | 6.382 | 2.568 |
| letter | PFoil:pw_larger:weighted_vote | 26.68% | 9317.6 | 6.382 | 24.423 |
| letter | PFoil:pw_smaller:accuracy_vote | 89.16% | 9141.8 | 6.528 | 3.024 |
| letter | PFoil:pw_smaller:vote | 89.12% | 9141.8 | 6.528 | 2.442 |
| letter | PFoil:pw_smaller:weighted_vote | 19.82% | 9141.8 | 6.528 | 25.616 |
| letter | PFossil:ordered | 77.29% | 1170.6 | 1.791 | 0.003 |
| letter | PFossil:ovr | 75.67% | 1249.6 | 2.308 | 0.052 |
| letter | PFossil:pw_both:accuracy_vote | 86.04% | 11414.6 | 9.365 | 5.686 |
| letter | PFossil:pw_both:vote | 85.99% | 11414.6 | 9.365 | 4.276 |
| letter | PFossil:pw_both:weighted_vote | 86.30% | 11414.6 | 9.365 | 44.554 |
| letter | PFossil:pw_larger:accuracy_vote | 84.05% | 5755.2 | 5.016 | 2.725 |
| letter | PFossil:pw_larger:vote | 83.32% | 5755.2 | 5.016 | 2.243 |
| letter | PFossil:pw_larger:weighted_vote | 26.63% | 5755.2 | 5.016 | 21.874 |
| letter | PFossil:pw_smaller:accuracy_vote | 83.11% | 5675.0 | 4.921 | 2.765 |
| letter | PFossil:pw_smaller:vote | 82.84% | 5675.0 | 4.921 | 2.291 |
| letter | PFossil:pw_smaller:weighted_vote | 20.05% | 5675.0 | 4.921 | 23.911 |
| letter | Pypper:ordered | 81.39% | 1272.4 | 6.757 | 0.003 |
| letter | Pypper:ovr | 77.00% | 1662.0 | 11.617 | 0.094 |
| letter | Pypper:pw_both:accuracy_vote | 90.03% | 8894.4 | 28.357 | 5.471 |
| letter | Pypper:pw_both:vote | 89.98% | 8894.4 | 28.357 | 4.091 |
| letter | Pypper:pw_both:weighted_vote | 89.71% | 8894.4 | 28.357 | 44.490 |
| letter | Pypper:pw_larger:accuracy_vote | 88.29% | 4482.0 | 14.539 | 2.949 |
| letter | Pypper:pw_larger:vote | 88.28% | 4482.0 | 14.539 | 2.412 |
| letter | Pypper:pw_larger:weighted_vote | 27.89% | 4482.0 | 14.539 | 22.600 |
| letter | Pypper:pw_smaller:accuracy_vote | 87.93% | 4411.2 | 13.995 | 2.533 |
| letter | Pypper:pw_smaller:vote | 87.93% | 4411.2 | 13.995 | 1.935 |
| letter | Pypper:pw_smaller:weighted_vote | 22.22% | 4411.2 | 13.995 | 21.655 |
| optdigits | PFoil:ordered | 88.26% | 420.3 | 0.749 | 0.001 |
| optdigits | PFoil:ovr | 93.35% | 692.5 | 1.399 | 0.015 |
| optdigits | PFoil:pw_both:accuracy_vote | 96.62% | 1076.2 | 1.702 | 0.162 |
| optdigits | PFoil:pw_both:vote | 96.55% | 1076.2 | 1.702 | 0.100 |
| optdigits | PFoil:pw_both:weighted_vote | 96.80% | 1076.2 | 1.702 | 1.119 |
| optdigits | PFoil:pw_larger:accuracy_vote | 94.57% | 537.7 | 0.989 | 0.063 |
| optdigits | PFoil:pw_larger:vote | 94.59% | 537.7 | 0.989 | 0.049 |
| optdigits | PFoil:pw_larger:weighted_vote | 32.03% | 537.7 | 0.989 | 0.497 |
| optdigits | PFoil:pw_smaller:accuracy_vote | 94.15% | 539.4 | 1.010 | 0.066 |
| optdigits | PFoil:pw_smaller:vote | 94.38% | 539.4 | 1.010 | 0.067 |
| optdigits | PFoil:pw_smaller:weighted_vote | 43.43% | 539.4 | 1.010 | 0.656 |
| optdigits | PFossil:ordered | 90.71% | 299.6 | 0.567 | 0.002 |
| optdigits | PFossil:ovr | 90.30% | 457.4 | 0.896 | 0.008 |
| optdigits | PFossil:pw_both:accuracy_vote | 95.64% | 1096.9 | 1.871 | 0.121 |
| optdigits | PFossil:pw_both:vote | 95.48% | 1096.9 | 1.871 | 0.115 |
| optdigits | PFossil:pw_both:weighted_vote | 95.82% | 1096.9 | 1.871 | 1.141 |
| optdigits | PFossil:pw_larger:accuracy_vote | 93.95% | 540.3 | 0.935 | 0.079 |
| optdigits | PFossil:pw_larger:vote | 93.84% | 540.3 | 0.935 | 0.055 |
| optdigits | PFossil:pw_larger:weighted_vote | 33.04% | 540.3 | 0.935 | 0.607 |
| optdigits | PFossil:pw_smaller:accuracy_vote | 93.90% | 558.7 | 1.050 | 0.077 |
| optdigits | PFossil:pw_smaller:vote | 94.04% | 558.7 | 1.050 | 0.060 |
| optdigits | PFossil:pw_smaller:weighted_vote | 40.18% | 558.7 | 1.050 | 0.618 |
| optdigits | Pypper:ordered | 90.91% | 199.6 | 1.412 | 0.001 |
| optdigits | Pypper:ovr | 90.18% | 301.1 | 2.072 | 0.007 |
| optdigits | Pypper:pw_both:accuracy_vote | 95.84% | 625.3 | 4.381 | 0.109 |
| optdigits | Pypper:pw_both:vote | 95.98% | 625.3 | 4.381 | 0.090 |
| optdigits | Pypper:pw_both:weighted_vote | 95.77% | 625.3 | 4.381 | 0.924 |
| optdigits | Pypper:pw_larger:accuracy_vote | 94.34% | 309.3 | 2.214 | 0.077 |
| optdigits | Pypper:pw_larger:vote | 94.00% | 309.3 | 2.214 | 0.058 |
| optdigits | Pypper:pw_larger:weighted_vote | 33.04% | 309.3 | 2.214 | 0.570 |
| optdigits | Pypper:pw_smaller:accuracy_vote | 94.64% | 315.8 | 2.277 | 0.077 |
| optdigits | Pypper:pw_smaller:vote | 94.54% | 315.8 | 2.277 | 0.061 |
| optdigits | Pypper:pw_smaller:weighted_vote | 44.32% | 315.8 | 2.277 | 0.623 |
| primary-tumor | PFoil:ordered | 9.45% | 223.8 | 0.120 | 0.001 |
| primary-tumor | PFoil:ovr | 43.35% | 329.0 | 0.171 | 0.004 |
| primary-tumor | PFoil:pw_both:accuracy_vote | 42.79% | 1259.7 | 0.685 | 0.044 |
| primary-tumor | PFoil:pw_both:vote | 44.25% | 1259.7 | 0.685 | 0.029 |
| primary-tumor | PFoil:pw_both:weighted_vote | 33.62% | 1259.7 | 0.685 | 0.285 |
| primary-tumor | PFoil:pw_larger:accuracy_vote | 41.32% | 637.6 | 0.408 | 0.023 |
| primary-tumor | PFoil:pw_larger:vote | 35.12% | 637.6 | 0.408 | 0.018 |
| primary-tumor | PFoil:pw_larger:weighted_vote | 24.78% | 637.6 | 0.408 | 0.165 |
| primary-tumor | PFoil:pw_smaller:accuracy_vote | 42.17% | 620.8 | 0.342 | 0.025 |
| primary-tumor | PFoil:pw_smaller:vote | 36.28% | 620.8 | 0.342 | 0.022 |
| primary-tumor | PFoil:pw_smaller:weighted_vote | 41.86% | 620.8 | 0.342 | 0.152 |
| primary-tumor | PFossil:ordered | 32.44% | 330.7 | 0.156 | 0.001 |
| primary-tumor | PFossil:ovr | 43.38% | 341.1 | 0.149 | 0.002 |
| primary-tumor | PFossil:pw_both:accuracy_vote | 46.02% | 1464.9 | 0.742 | 0.051 |
| primary-tumor | PFossil:pw_both:vote | 37.16% | 1464.9 | 0.742 | 0.044 |
| primary-tumor | PFossil:pw_both:weighted_vote | 35.08% | 1464.9 | 0.742 | 0.340 |
| primary-tumor | PFossil:pw_larger:accuracy_vote | 39.84% | 652.7 | 0.422 | 0.022 |
| primary-tumor | PFossil:pw_larger:vote | 28.61% | 652.7 | 0.422 | 0.020 |
| primary-tumor | PFossil:pw_larger:weighted_vote | 24.78% | 652.7 | 0.422 | 0.176 |
| primary-tumor | PFossil:pw_smaller:accuracy_vote | 39.22% | 813.1 | 0.422 | 0.025 |
| primary-tumor | PFossil:pw_smaller:vote | 38.62% | 813.1 | 0.422 | 0.018 |
| primary-tumor | PFossil:pw_smaller:weighted_vote | 41.59% | 813.1 | 0.422 | 0.159 |
| primary-tumor | Pypper:ordered | 39.25% | 20.5 | 0.122 | 0.000 |
| primary-tumor | Pypper:ovr | 38.65% | 21.9 | 0.129 | 0.000 |
| primary-tumor | Pypper:pw_both:accuracy_vote | 47.20% | 260.0 | 2.212 | 0.034 |
| primary-tumor | Pypper:pw_both:vote | 44.55% | 260.0 | 2.212 | 0.029 |
| primary-tumor | Pypper:pw_both:weighted_vote | 27.15% | 260.0 | 2.212 | 0.244 |
| primary-tumor | Pypper:pw_larger:accuracy_vote | 43.97% | 139.7 | 1.382 | 0.019 |
| primary-tumor | Pypper:pw_larger:vote | 0.00% | 139.7 | 1.382 | 0.013 |
| primary-tumor | Pypper:pw_larger:weighted_vote | 24.78% | 139.7 | 1.382 | 0.112 |
| primary-tumor | Pypper:pw_smaller:accuracy_vote | 43.97% | 120.9 | 0.861 | 0.018 |
| primary-tumor | Pypper:pw_smaller:vote | 43.07% | 120.9 | 0.861 | 0.013 |
| primary-tumor | Pypper:pw_smaller:weighted_vote | 33.07% | 120.9 | 0.861 | 0.105 |
| segment | PFoil:ordered | 83.81% | 133.7 | 0.117 | 0.000 |
| segment | PFoil:ovr | 95.19% | 291.3 | 0.254 | 0.003 |
| segment | PFoil:pw_both:accuracy_vote | 93.90% | 293.9 | 0.248 | 0.024 |
| segment | PFoil:pw_both:vote | 93.20% | 293.9 | 0.248 | 0.018 |
| segment | PFoil:pw_both:weighted_vote | 96.32% | 293.9 | 0.248 | 0.175 |
| segment | PFoil:pw_larger:accuracy_vote | 90.39% | 133.6 | 0.131 | 0.012 |
| segment | PFoil:pw_larger:vote | 89.26% | 133.6 | 0.131 | 0.009 |
| segment | PFoil:pw_larger:weighted_vote | 44.16% | 133.6 | 0.131 | 0.079 |
| segment | PFoil:pw_smaller:accuracy_vote | 90.39% | 133.6 | 0.137 | 0.011 |
| segment | PFoil:pw_smaller:vote | 89.26% | 133.6 | 0.137 | 0.009 |
| segment | PFoil:pw_smaller:weighted_vote | 44.16% | 133.6 | 0.137 | 0.078 |
| segment | PFossil:ordered | 94.85% | 75.5 | 0.075 | 0.000 |
| segment | PFossil:ovr | 93.33% | 121.8 | 0.111 | 0.002 |
| segment | PFossil:pw_both:accuracy_vote | 95.15% | 180.8 | 0.188 | 0.028 |
| segment | PFossil:pw_both:vote | 95.32% | 180.8 | 0.188 | 0.024 |
| segment | PFossil:pw_both:weighted_vote | 95.54% | 180.8 | 0.188 | 0.200 |
| segment | PFossil:pw_larger:accuracy_vote | 94.81% | 103.4 | 0.102 | 0.016 |
| segment | PFossil:pw_larger:vote | 94.46% | 103.4 | 0.102 | 0.012 |
| segment | PFossil:pw_larger:weighted_vote | 47.06% | 103.4 | 0.102 | 0.107 |
| segment | PFossil:pw_smaller:accuracy_vote | 94.81% | 103.4 | 0.097 | 0.016 |
| segment | PFossil:pw_smaller:vote | 94.46% | 103.4 | 0.097 | 0.012 |
| segment | PFossil:pw_smaller:weighted_vote | 47.06% | 103.4 | 0.097 | 0.101 |
| segment | Pypper:ordered | 94.72% | 45.6 | 0.175 | 0.000 |
| segment | Pypper:ovr | 93.46% | 77.9 | 0.333 | 0.001 |
| segment | Pypper:pw_both:accuracy_vote | 95.41% | 126.9 | 0.602 | 0.019 |
| segment | Pypper:pw_both:vote | 95.54% | 126.9 | 0.602 | 0.015 |
| segment | Pypper:pw_both:weighted_vote | 95.37% | 126.9 | 0.602 | 0.129 |
| segment | Pypper:pw_larger:accuracy_vote | 94.76% | 62.8 | 0.285 | 0.016 |
| segment | Pypper:pw_larger:vote | 94.68% | 62.8 | 0.285 | 0.009 |
| segment | Pypper:pw_larger:weighted_vote | 52.73% | 62.8 | 0.285 | 0.089 |
| segment | Pypper:pw_smaller:accuracy_vote | 94.76% | 62.8 | 0.299 | 0.012 |
| segment | Pypper:pw_smaller:vote | 94.68% | 62.8 | 0.299 | 0.008 |
| segment | Pypper:pw_smaller:weighted_vote | 52.73% | 62.8 | 0.299 | 0.072 |
| solar-flare | PFoil:ordered | 58.72% | 241.1 | 0.129 | 0.001 |
| solar-flare | PFoil:ovr | 73.73% | 403.4 | 0.193 | 0.006 |
| solar-flare | PFoil:pw_both:accuracy_vote | 72.51% | 694.9 | 0.310 | 0.013 |
| solar-flare | PFoil:pw_both:vote | 68.01% | 694.9 | 0.310 | 0.012 |
| solar-flare | PFoil:pw_both:weighted_vote | 73.45% | 694.9 | 0.310 | 0.103 |
| solar-flare | PFoil:pw_larger:accuracy_vote | 68.11% | 411.2 | 0.193 | 0.007 |
| solar-flare | PFoil:pw_larger:vote | 67.64% | 411.2 | 0.193 | 0.006 |
| solar-flare | PFoil:pw_larger:weighted_vote | 56.66% | 411.2 | 0.193 | 0.045 |
| solar-flare | PFoil:pw_smaller:accuracy_vote | 59.19% | 283.7 | 0.142 | 0.006 |
| solar-flare | PFoil:pw_smaller:vote | 60.60% | 283.7 | 0.142 | 0.005 |
| solar-flare | PFoil:pw_smaller:weighted_vote | 62.11% | 283.7 | 0.142 | 0.037 |
| solar-flare | PFossil:ordered | 70.64% | 50.0 | 0.040 | 0.000 |
| solar-flare | PFossil:ovr | 74.02% | 62.5 | 0.047 | 0.002 |
| solar-flare | PFossil:pw_both:accuracy_vote | 73.83% | 161.9 | 0.108 | 0.009 |
| solar-flare | PFossil:pw_both:vote | 74.30% | 161.9 | 0.108 | 0.006 |
| solar-flare | PFossil:pw_both:weighted_vote | 74.77% | 161.9 | 0.108 | 0.055 |
| solar-flare | PFossil:pw_larger:accuracy_vote | 74.77% | 73.0 | 0.059 | 0.005 |
| solar-flare | PFossil:pw_larger:vote | 73.93% | 73.0 | 0.059 | 0.004 |
| solar-flare | PFossil:pw_larger:weighted_vote | 58.82% | 73.0 | 0.059 | 0.030 |
| solar-flare | PFossil:pw_smaller:accuracy_vote | 72.23% | 88.9 | 0.064 | 0.005 |
| solar-flare | PFossil:pw_smaller:vote | 72.33% | 88.9 | 0.064 | 0.004 |
| solar-flare | PFossil:pw_smaller:weighted_vote | 71.39% | 88.9 | 0.064 | 0.031 |
| solar-flare | Pypper:ordered | 74.77% | 8.3 | 0.097 | 0.000 |
| solar-flare | Pypper:ovr | 69.43% | 21.4 | 0.144 | 0.000 |
| solar-flare | Pypper:pw_both:accuracy_vote | 73.92% | 54.3 | 0.391 | 0.008 |
| solar-flare | Pypper:pw_both:vote | 74.40% | 54.3 | 0.391 | 0.007 |
| solar-flare | Pypper:pw_both:weighted_vote | 74.40% | 54.3 | 0.391 | 0.058 |
| solar-flare | Pypper:pw_larger:accuracy_vote | 72.89% | 27.5 | 0.252 | 0.004 |
| solar-flare | Pypper:pw_larger:vote | 72.70% | 27.5 | 0.252 | 0.004 |
| solar-flare | Pypper:pw_larger:weighted_vote | 55.90% | 27.5 | 0.252 | 0.030 |
| solar-flare | Pypper:pw_smaller:accuracy_vote | 74.68% | 26.8 | 0.198 | 0.005 |
| solar-flare | Pypper:pw_smaller:vote | 74.77% | 26.8 | 0.198 | 0.004 |
| solar-flare | Pypper:pw_smaller:weighted_vote | 71.11% | 26.8 | 0.198 | 0.025 |
| soybean | PFoil:ordered | 85.65% | 83.5 | 0.065 | 0.000 |
| soybean | PFoil:ovr | 90.92% | 153.8 | 0.097 | 0.001 |
| soybean | PFoil:pw_both:accuracy_vote | 92.97% | 446.2 | 0.378 | 0.056 |
| soybean | PFoil:pw_both:vote | 93.41% | 446.2 | 0.378 | 0.041 |
| soybean | PFoil:pw_both:weighted_vote | 62.22% | 446.2 | 0.378 | 0.348 |
| soybean | PFoil:pw_larger:accuracy_vote | 92.68% | 223.6 | 0.183 | 0.028 |
| soybean | PFoil:pw_larger:vote | 92.83% | 223.6 | 0.183 | 0.027 |
| soybean | PFoil:pw_larger:weighted_vote | 40.12% | 223.6 | 0.183 | 0.204 |
| soybean | PFoil:pw_smaller:accuracy_vote | 91.80% | 230.8 | 0.198 | 0.028 |
| soybean | PFoil:pw_smaller:vote | 90.92% | 230.8 | 0.198 | 0.019 |
| soybean | PFoil:pw_smaller:weighted_vote | 74.81% | 230.8 | 0.198 | 0.172 |
| soybean | PFossil:ordered | 92.98% | 70.2 | 0.064 | 0.000 |
| soybean | PFossil:ovr | 92.24% | 135.4 | 0.091 | 0.001 |
| soybean | PFossil:pw_both:accuracy_vote | 92.39% | 418.9 | 0.342 | 0.050 |
| soybean | PFossil:pw_both:vote | 93.12% | 418.9 | 0.342 | 0.040 |
| soybean | PFossil:pw_both:weighted_vote | 61.65% | 418.9 | 0.342 | 0.375 |
| soybean | PFossil:pw_larger:accuracy_vote | 93.26% | 207.0 | 0.199 | 0.032 |
| soybean | PFossil:pw_larger:vote | 93.70% | 207.0 | 0.199 | 0.022 |
| soybean | PFossil:pw_larger:weighted_vote | 37.02% | 207.0 | 0.199 | 0.189 |
| soybean | PFossil:pw_smaller:accuracy_vote | 93.41% | 201.5 | 0.180 | 0.033 |
| soybean | PFossil:pw_smaller:vote | 93.27% | 201.5 | 0.180 | 0.022 |
| soybean | PFossil:pw_smaller:weighted_vote | 75.25% | 201.5 | 0.180 | 0.208 |
| soybean | Pypper:ordered | 90.93% | 45.9 | 0.223 | 0.000 |
| soybean | Pypper:ovr | 89.91% | 58.8 | 0.275 | 0.001 |
| soybean | Pypper:pw_both:accuracy_vote | 93.12% | 373.4 | 1.760 | 0.059 |
| soybean | Pypper:pw_both:vote | 92.83% | 373.4 | 1.760 | 0.049 |
| soybean | Pypper:pw_both:weighted_vote | 61.79% | 373.4 | 1.760 | 0.408 |
| soybean | Pypper:pw_larger:accuracy_vote | 93.42% | 185.4 | 0.920 | 0.028 |
| soybean | Pypper:pw_larger:vote | 90.94% | 185.4 | 0.920 | 0.022 |
| soybean | Pypper:pw_larger:weighted_vote | 40.56% | 185.4 | 0.920 | 0.207 |
| soybean | Pypper:pw_smaller:accuracy_vote | 91.95% | 188.0 | 0.915 | 0.028 |
| soybean | Pypper:pw_smaller:vote | 91.80% | 188.0 | 0.915 | 0.021 |
| soybean | Pypper:pw_smaller:weighted_vote | 73.79% | 188.0 | 0.915 | 0.200 |
| texture | PFoil:ordered | 84.27% | 469.9 | 0.686 | 0.002 |
| texture | PFoil:ovr | 94.51% | 777.3 | 1.213 | 0.018 |
| texture | PFoil:pw_both:accuracy_vote | 94.95% | 903.5 | 1.147 | 0.143 |
| texture | PFoil:pw_both:vote | 94.87% | 903.5 | 1.147 | 0.093 |
| texture | PFoil:pw_both:weighted_vote | 96.04% | 903.5 | 1.147 | 1.045 |
| texture | PFoil:pw_larger:accuracy_vote | 92.85% | 460.8 | 0.653 | 0.071 |
| texture | PFoil:pw_larger:vote | 91.95% | 460.8 | 0.653 | 0.065 |
| texture | PFoil:pw_larger:weighted_vote | 31.51% | 460.8 | 0.653 | 0.584 |
| texture | PFoil:pw_smaller:accuracy_vote | 92.85% | 460.8 | 0.699 | 0.073 |
| texture | PFoil:pw_smaller:vote | 91.95% | 460.8 | 0.699 | 0.068 |
| texture | PFoil:pw_smaller:weighted_vote | 31.51% | 460.8 | 0.699 | 0.613 |
| texture | PFossil:ordered | 90.45% | 219.7 | 0.363 | 0.001 |
| texture | PFossil:ovr | 92.22% | 385.6 | 0.578 | 0.009 |
| texture | PFossil:pw_both:accuracy_vote | 94.05% | 674.9 | 0.993 | 0.161 |
| texture | PFossil:pw_both:vote | 93.58% | 674.9 | 0.993 | 0.119 |
| texture | PFossil:pw_both:weighted_vote | 94.58% | 674.9 | 0.993 | 1.294 |
| texture | PFossil:pw_larger:accuracy_vote | 92.35% | 315.6 | 0.579 | 0.078 |
| texture | PFossil:pw_larger:vote | 91.71% | 315.6 | 0.579 | 0.059 |
| texture | PFossil:pw_larger:weighted_vote | 30.64% | 315.6 | 0.579 | 0.598 |
| texture | PFossil:pw_smaller:accuracy_vote | 92.35% | 315.6 | 0.522 | 0.080 |
| texture | PFossil:pw_smaller:vote | 91.71% | 315.6 | 0.522 | 0.068 |
| texture | PFossil:pw_smaller:weighted_vote | 30.64% | 315.6 | 0.522 | 0.640 |
| texture | Pypper:ordered | 91.56% | 188.5 | 1.014 | 0.001 |
| texture | Pypper:ovr | 91.78% | 309.9 | 1.873 | 0.006 |
| texture | Pypper:pw_both:accuracy_vote | 95.33% | 486.8 | 2.683 | 0.150 |
| texture | Pypper:pw_both:vote | 94.91% | 486.8 | 2.683 | 0.111 |
| texture | Pypper:pw_both:weighted_vote | 95.15% | 486.8 | 2.683 | 1.110 |
| texture | Pypper:pw_larger:accuracy_vote | 94.53% | 252.7 | 1.324 | 0.087 |
| texture | Pypper:pw_larger:vote | 94.38% | 252.7 | 1.324 | 0.062 |
| texture | Pypper:pw_larger:weighted_vote | 32.73% | 252.7 | 1.324 | 0.591 |
| texture | Pypper:pw_smaller:accuracy_vote | 94.53% | 252.7 | 1.427 | 0.084 |
| texture | Pypper:pw_smaller:vote | 94.38% | 252.7 | 1.427 | 0.070 |
| texture | Pypper:pw_smaller:weighted_vote | 32.73% | 252.7 | 1.427 | 0.536 |
| vowel | PFoil:ordered | 68.99% | 278.7 | 0.182 | 0.001 |
| vowel | PFoil:ovr | 81.82% | 465.9 | 0.306 | 0.005 |
| vowel | PFoil:pw_both:accuracy_vote | 87.27% | 790.4 | 0.529 | 0.033 |
| vowel | PFoil:pw_both:vote | 86.57% | 790.4 | 0.529 | 0.029 |
| vowel | PFoil:pw_both:weighted_vote | 87.47% | 790.4 | 0.529 | 0.235 |
| vowel | PFoil:pw_larger:accuracy_vote | 81.01% | 390.2 | 0.270 | 0.017 |
| vowel | PFoil:pw_larger:vote | 81.41% | 390.2 | 0.270 | 0.015 |
| vowel | PFoil:pw_larger:weighted_vote | 38.89% | 390.2 | 0.270 | 0.126 |
| vowel | PFoil:pw_smaller:accuracy_vote | 81.01% | 390.2 | 0.247 | 0.019 |
| vowel | PFoil:pw_smaller:vote | 81.41% | 390.2 | 0.247 | 0.016 |
| vowel | PFoil:pw_smaller:weighted_vote | 38.89% | 390.2 | 0.247 | 0.134 |
| vowel | PFossil:ordered | 80.40% | 247.7 | 0.148 | 0.001 |
| vowel | PFossil:ovr | 77.27% | 401.3 | 0.249 | 0.002 |
| vowel | PFossil:pw_both:accuracy_vote | 83.54% | 744.6 | 0.487 | 0.054 |
| vowel | PFossil:pw_both:vote | 81.82% | 744.6 | 0.487 | 0.032 |
| vowel | PFossil:pw_both:weighted_vote | 82.22% | 744.6 | 0.487 | 0.262 |
| vowel | PFossil:pw_larger:accuracy_vote | 82.73% | 372.1 | 0.253 | 0.019 |
| vowel | PFossil:pw_larger:vote | 81.31% | 372.1 | 0.253 | 0.013 |
| vowel | PFossil:pw_larger:weighted_vote | 35.96% | 372.1 | 0.253 | 0.118 |
| vowel | PFossil:pw_smaller:accuracy_vote | 82.73% | 372.1 | 0.248 | 0.017 |
| vowel | PFossil:pw_smaller:vote | 81.31% | 372.1 | 0.248 | 0.013 |
| vowel | PFossil:pw_smaller:weighted_vote | 35.96% | 372.1 | 0.248 | 0.109 |
| vowel | Pypper:ordered | 69.70% | 123.4 | 0.384 | 0.000 |
| vowel | Pypper:ovr | 63.84% | 171.5 | 0.589 | 0.001 |
| vowel | Pypper:pw_both:accuracy_vote | 79.70% | 509.1 | 1.804 | 0.034 |
| vowel | Pypper:pw_both:vote | 78.99% | 509.1 | 1.804 | 0.024 |
| vowel | Pypper:pw_both:weighted_vote | 78.48% | 509.1 | 1.804 | 0.260 |
| vowel | Pypper:pw_larger:accuracy_vote | 77.88% | 254.8 | 0.878 | 0.019 |
| vowel | Pypper:pw_larger:vote | 78.59% | 254.8 | 0.878 | 0.013 |
| vowel | Pypper:pw_larger:weighted_vote | 37.07% | 254.8 | 0.878 | 0.140 |
| vowel | Pypper:pw_smaller:accuracy_vote | 77.88% | 254.8 | 0.859 | 0.015 |
| vowel | Pypper:pw_smaller:vote | 78.59% | 254.8 | 0.859 | 0.012 |
| vowel | Pypper:pw_smaller:weighted_vote | 37.07% | 254.8 | 0.859 | 0.115 |
| yeast | PFoil:ordered | 23.79% | 744.7 | 0.419 | 0.002 |
| yeast | PFoil:ovr | 57.41% | 1245.0 | 0.620 | 0.012 |
| yeast | PFoil:pw_both:accuracy_vote | 54.85% | 2548.4 | 1.346 | 0.051 |
| yeast | PFoil:pw_both:vote | 52.63% | 2548.4 | 1.346 | 0.039 |
| yeast | PFoil:pw_both:weighted_vote | 51.82% | 2548.4 | 1.346 | 0.427 |
| yeast | PFoil:pw_larger:accuracy_vote | 48.05% | 1421.4 | 0.697 | 0.028 |
| yeast | PFoil:pw_larger:vote | 47.44% | 1421.4 | 0.697 | 0.022 |
| yeast | PFoil:pw_larger:weighted_vote | 36.25% | 1421.4 | 0.697 | 0.247 |
| yeast | PFoil:pw_smaller:accuracy_vote | 43.40% | 1127.0 | 0.579 | 0.025 |
| yeast | PFoil:pw_smaller:vote | 38.14% | 1127.0 | 0.579 | 0.021 |
| yeast | PFoil:pw_smaller:weighted_vote | 56.81% | 1127.0 | 0.579 | 0.156 |
| yeast | PFossil:ordered | 56.06% | 148.8 | 0.109 | 0.001 |
| yeast | PFossil:ovr | 57.48% | 180.5 | 0.133 | 0.002 |
| yeast | PFossil:pw_both:accuracy_vote | 59.57% | 898.5 | 0.503 | 0.039 |
| yeast | PFossil:pw_both:vote | 59.77% | 898.5 | 0.503 | 0.029 |
| yeast | PFossil:pw_both:weighted_vote | 51.76% | 898.5 | 0.503 | 0.266 |
| yeast | PFossil:pw_larger:accuracy_vote | 57.28% | 337.3 | 0.222 | 0.021 |
| yeast | PFossil:pw_larger:vote | 56.54% | 337.3 | 0.222 | 0.017 |
| yeast | PFossil:pw_larger:weighted_vote | 35.37% | 337.3 | 0.222 | 0.210 |
| yeast | PFossil:pw_smaller:accuracy_vote | 57.34% | 561.2 | 0.311 | 0.021 |
| yeast | PFossil:pw_smaller:vote | 57.48% | 561.2 | 0.311 | 0.017 |
| yeast | PFossil:pw_smaller:weighted_vote | 58.49% | 561.2 | 0.311 | 0.140 |
| yeast | Pypper:ordered | 58.96% | 38.7 | 0.289 | 0.000 |
| yeast | Pypper:ovr | 59.30% | 66.8 | 0.430 | 0.001 |
| yeast | Pypper:pw_both:accuracy_vote | 59.30% | 220.1 | 1.497 | 0.032 |
| yeast | Pypper:pw_both:vote | 59.10% | 220.1 | 1.497 | 0.028 |
| yeast | Pypper:pw_both:weighted_vote | 49.87% | 220.1 | 1.497 | 0.245 |
| yeast | Pypper:pw_larger:accuracy_vote | 57.82% | 106.8 | 0.832 | 0.018 |
| yeast | Pypper:pw_larger:vote | 58.56% | 106.8 | 0.832 | 0.014 |
| yeast | Pypper:pw_larger:weighted_vote | 36.86% | 106.8 | 0.832 | 0.144 |
| yeast | Pypper:pw_smaller:accuracy_vote | 59.23% | 113.3 | 0.653 | 0.021 |
| yeast | Pypper:pw_smaller:vote | 58.42% | 113.3 | 0.653 | 0.014 |
| yeast | Pypper:pw_smaller:weighted_vote | 59.30% | 113.3 | 0.653 | 0.146 |
| zoo | PFoil:ordered | 92.00% | 15.8 | 0.009 | 0.000 |
| zoo | PFoil:ovr | 96.00% | 20.3 | 0.011 | 0.000 |
| zoo | PFoil:pw_both:accuracy_vote | 93.00% | 43.7 | 0.037 | 0.002 |
| zoo | PFoil:pw_both:vote | 93.00% | 43.7 | 0.037 | 0.001 |
| zoo | PFoil:pw_both:weighted_vote | 88.00% | 43.7 | 0.037 | 0.010 |
| zoo | PFoil:pw_larger:accuracy_vote | 94.00% | 21.9 | 0.019 | 0.001 |
| zoo | PFoil:pw_larger:vote | 95.00% | 21.9 | 0.019 | 0.001 |
| zoo | PFoil:pw_larger:weighted_vote | 60.36% | 21.9 | 0.019 | 0.005 |
| zoo | PFoil:pw_smaller:accuracy_vote | 91.09% | 21.8 | 0.021 | 0.001 |
| zoo | PFoil:pw_smaller:vote | 91.09% | 21.8 | 0.021 | 0.001 |
| zoo | PFoil:pw_smaller:weighted_vote | 91.09% | 21.8 | 0.021 | 0.006 |
| zoo | PFossil:ordered | 93.00% | 16.7 | 0.011 | 0.000 |
| zoo | PFossil:ovr | 95.00% | 21.0 | 0.013 | 0.000 |
| zoo | PFossil:pw_both:accuracy_vote | 94.00% | 45.6 | 0.037 | 0.002 |
| zoo | PFossil:pw_both:vote | 93.00% | 45.6 | 0.037 | 0.002 |
| zoo | PFossil:pw_both:weighted_vote | 90.00% | 45.6 | 0.037 | 0.011 |
| zoo | PFossil:pw_larger:accuracy_vote | 93.00% | 22.7 | 0.022 | 0.001 |
| zoo | PFossil:pw_larger:vote | 93.00% | 22.7 | 0.022 | 0.001 |
| zoo | PFossil:pw_larger:weighted_vote | 60.36% | 22.7 | 0.022 | 0.005 |
| zoo | PFossil:pw_smaller:accuracy_vote | 94.00% | 22.9 | 0.019 | 0.001 |
| zoo | PFossil:pw_smaller:vote | 94.00% | 22.9 | 0.019 | 0.001 |
| zoo | PFossil:pw_smaller:weighted_vote | 93.00% | 22.9 | 0.019 | 0.004 |
| zoo | Pypper:ordered | 89.09% | 7.8 | 0.041 | 0.000 |
| zoo | Pypper:ovr | 90.09% | 10.3 | 0.044 | 0.000 |
| zoo | Pypper:pw_both:accuracy_vote | 91.00% | 40.0 | 0.211 | 0.002 |
| zoo | Pypper:pw_both:vote | 91.00% | 40.0 | 0.211 | 0.002 |
| zoo | Pypper:pw_both:weighted_vote | 84.18% | 40.0 | 0.211 | 0.010 |
| zoo | Pypper:pw_larger:accuracy_vote | 91.00% | 20.0 | 0.117 | 0.001 |
| zoo | Pypper:pw_larger:vote | 90.09% | 20.0 | 0.117 | 0.001 |
| zoo | Pypper:pw_larger:weighted_vote | 60.36% | 20.0 | 0.117 | 0.005 |
| zoo | Pypper:pw_smaller:accuracy_vote | 91.00% | 20.0 | 0.115 | 0.001 |
| zoo | Pypper:pw_smaller:vote | 89.09% | 20.0 | 0.115 | 0.001 |
| zoo | Pypper:pw_smaller:weighted_vote | 88.09% | 20.0 | 0.115 | 0.005 |

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

