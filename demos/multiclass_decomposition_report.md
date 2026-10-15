# Multi-class decomposition: one-vs-rest vs. pairwise

## Setup

**Full run.** Quick sanity check instead: `python demos/multiclass_decomposition.py`.

A rule learner that learns rules for one class against the rest has to
decompose a problem with several classes into two-class problems. This
demo compares the ways of doing that in pyrulearn, with Pypper
(pyrulearn's RIPPER) and PFossil as base learners -- in essence the
experiments of Fürnkranz, *Round Robin Classification* (JMLR 2002), with
Pypper standing in for Ripper:

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
| pw_both:accuracy_vote | 79.11% | 2.27 | 431.0 | 1143.6 | 15.13 | 0.450 | 0 |
| pw_both:vote | 77.90% | 2.96 | 431.0 | 1143.6 | 15.13 | 0.346 | 0 |
| pw_smaller:accuracy_vote | 77.95% | 4.27 | 209.2 | 547.5 | 8.31 | 0.220 | 0 |
| pw_larger:accuracy_vote | 78.09% | 4.81 | 221.1 | 597.7 | 9.57 | 0.223 | 0 |
| pw_smaller:vote | 78.10% | 5.19 | 209.2 | 547.5 | 8.31 | 0.176 | 0 |
| pw_both:weighted_vote | 70.92% | 6.12 | 431.0 | 1143.6 | 15.13 | 3.328 | 0 |
| ordered | 76.14% | 6.35 | 39.1 | 172.4 | 3.91 | 0.002 | 0 |
| pw_larger:vote | 69.05% | 6.92 | 221.1 | 597.7 | 9.57 | 0.172 | 0 |
| ovr | 72.88% | 7.81 | 40.6 | 215.5 | 3.56 | 0.009 | 0 |
| pw_smaller:weighted_vote | 51.36% | 8.69 | 209.2 | 547.5 | 8.31 | 1.482 | 0 |
| pw_larger:weighted_vote | 40.11% | 10.62 | 221.1 | 597.7 | 9.57 | 1.825 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## PFossil

| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |
|---|--:|--:|--:|--:|--:|--:|--:|
| pw_both:accuracy_vote | 79.15% | 2.77 | 569.5 | 1376.2 | 4.02 | 0.447 | 0 |
| pw_both:vote | 78.26% | 3.62 | 569.5 | 1376.2 | 4.02 | 0.334 | 0 |
| pw_smaller:accuracy_vote | 77.47% | 4.96 | 283.7 | 716.2 | 3.32 | 0.230 | 0 |
| pw_larger:accuracy_vote | 77.44% | 5.19 | 283.2 | 659.0 | 3.31 | 0.225 | 0 |
| pw_both:weighted_vote | 71.55% | 5.27 | 569.5 | 1376.2 | 4.02 | 3.229 | 0 |
| pw_smaller:vote | 77.12% | 5.65 | 283.7 | 716.2 | 3.32 | 0.172 | 0 |
| pw_larger:vote | 76.21% | 6.35 | 283.2 | 659.0 | 3.31 | 0.174 | 0 |
| ovr | 76.12% | 6.46 | 52.4 | 274.4 | 0.54 | 0.007 | 0 |
| ordered | 76.00% | 6.92 | 46.6 | 214.8 | 1.37 | 0.002 | 0 |
| pw_smaller:weighted_vote | 50.76% | 8.04 | 283.7 | 716.2 | 3.32 | 1.518 | 0 |
| pw_larger:weighted_vote | 38.56% | 10.77 | 283.2 | 659.0 | 3.31 | 1.693 | 0 |

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a decomposition's models together. The three vote schemes of a pairwise decomposition share its models and training time.

## Training and prediction time over the number of classes

One point per dataset and base learner: the mean one-vs-rest time divided by the mean pairwise time. Above 1, pairwise is faster. The dashed line is what a learner whose time grows linearly with the number of examples would give.

![training time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_train_ratio.png)

![training time ratio, one-vs-rest / double round robin](multiclass_decomposition_plots/multiclass_decomposition_train_ratio_double.png)

![prediction time ratio, one-vs-rest / pairwise](multiclass_decomposition_plots/multiclass_decomposition_predict_ratio.png)

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_conditions | train_time | predict_time |
|---|---|---|---|---|---|
| audiology | PFossil:ordered | 84.05% | 72.5 | 0.175 | 0.001 |
| audiology | PFossil:ovr | 77.89% | 115.2 | 0.085 | 0.002 |
| audiology | PFossil:pw_both:accuracy_vote | 81.40% | 586.0 | 1.285 | 0.052 |
| audiology | PFossil:pw_both:vote | 81.40% | 586.0 | 1.285 | 0.043 |
| audiology | PFossil:pw_both:weighted_vote | 46.46% | 586.0 | 1.285 | 0.191 |
| audiology | PFossil:pw_larger:accuracy_vote | 80.53% | 292.0 | 1.049 | 0.025 |
| audiology | PFossil:pw_larger:vote | 81.40% | 292.0 | 1.049 | 0.020 |
| audiology | PFossil:pw_larger:weighted_vote | 46.46% | 292.0 | 1.049 | 0.109 |
| audiology | PFossil:pw_smaller:accuracy_vote | 80.95% | 294.0 | 1.027 | 0.023 |
| audiology | PFossil:pw_smaller:vote | 81.38% | 294.0 | 1.027 | 0.018 |
| audiology | PFossil:pw_smaller:weighted_vote | 54.86% | 294.0 | 1.027 | 0.103 |
| audiology | Pypper:ordered | 75.28% | 16.0 | 0.233 | 0.000 |
| audiology | Pypper:ovr | 64.64% | 26.9 | 0.170 | 0.000 |
| audiology | Pypper:pw_both:accuracy_vote | 68.18% | 142.9 | 2.207 | 0.026 |
| audiology | Pypper:pw_both:vote | 57.04% | 142.9 | 2.207 | 0.026 |
| audiology | Pypper:pw_both:weighted_vote | 46.46% | 142.9 | 2.207 | 0.156 |
| audiology | Pypper:pw_larger:accuracy_vote | 69.51% | 87.7 | 1.658 | 0.013 |
| audiology | Pypper:pw_larger:vote | 7.92% | 87.7 | 1.658 | 0.010 |
| audiology | Pypper:pw_larger:weighted_vote | 46.46% | 87.7 | 1.658 | 0.065 |
| audiology | Pypper:pw_smaller:accuracy_vote | 65.51% | 55.4 | 1.443 | 0.013 |
| audiology | Pypper:pw_smaller:vote | 70.83% | 55.4 | 1.443 | 0.009 |
| audiology | Pypper:pw_smaller:weighted_vote | 48.64% | 55.4 | 1.443 | 0.087 |
| kropt | PFossil:ordered | 35.48% | 420.6 | 4.099 | 0.004 |
| kropt | PFossil:ovr | 24.99% | 342.6 | 1.495 | 0.033 |
| kropt | PFossil:pw_both:accuracy_vote | 40.70% | 5670.4 | 13.492 | 3.646 |
| kropt | PFossil:pw_both:vote | 39.41% | 5670.4 | 13.492 | 2.704 |
| kropt | PFossil:pw_both:weighted_vote | 32.21% | 5670.4 | 13.492 | 26.260 |
| kropt | PFossil:pw_larger:accuracy_vote | 36.74% | 2446.2 | 9.718 | 1.775 |
| kropt | PFossil:pw_larger:vote | 34.82% | 2446.2 | 9.718 | 1.390 |
| kropt | PFossil:pw_larger:weighted_vote | 22.80% | 2446.2 | 9.718 | 15.158 |
| kropt | PFossil:pw_smaller:accuracy_vote | 39.58% | 3224.2 | 9.983 | 1.939 |
| kropt | PFossil:pw_smaller:vote | 36.35% | 3224.2 | 9.983 | 1.446 |
| kropt | PFossil:pw_smaller:weighted_vote | 39.95% | 3224.2 | 9.983 | 10.308 |
| kropt | Pypper:ordered | 54.59% | 1299.2 | 39.543 | 0.016 |
| kropt | Pypper:ovr | 33.63% | 1130.0 | 28.023 | 0.065 |
| kropt | Pypper:pw_both:accuracy_vote | 66.17% | 11866.4 | 194.754 | 4.142 |
| kropt | Pypper:pw_both:vote | 67.00% | 11866.4 | 194.754 | 3.392 |
| kropt | Pypper:pw_both:weighted_vote | 50.76% | 11866.4 | 194.754 | 30.517 |
| kropt | Pypper:pw_larger:accuracy_vote | 62.95% | 6464.0 | 114.372 | 2.159 |
| kropt | Pypper:pw_larger:vote | 63.52% | 6464.0 | 114.372 | 1.737 |
| kropt | Pypper:pw_larger:weighted_vote | 30.84% | 6464.0 | 114.372 | 19.328 |
| kropt | Pypper:pw_smaller:accuracy_vote | 62.28% | 5402.4 | 88.242 | 2.123 |
| kropt | Pypper:pw_smaller:vote | 62.10% | 5402.4 | 88.242 | 1.538 |
| kropt | Pypper:pw_smaller:weighted_vote | 60.03% | 5402.4 | 88.242 | 10.623 |
| led24 | PFossil:ordered | 70.03% | 250.2 | 0.300 | 0.002 |
| led24 | PFossil:ovr | 70.00% | 272.8 | 0.287 | 0.004 |
| led24 | PFossil:pw_both:accuracy_vote | 70.88% | 1698.1 | 1.431 | 0.080 |
| led24 | PFossil:pw_both:vote | 71.47% | 1698.1 | 1.431 | 0.068 |
| led24 | PFossil:pw_both:weighted_vote | 71.41% | 1698.1 | 1.431 | 0.545 |
| led24 | PFossil:pw_larger:accuracy_vote | 66.44% | 891.4 | 0.902 | 0.040 |
| led24 | PFossil:pw_larger:vote | 66.84% | 891.4 | 0.902 | 0.038 |
| led24 | PFossil:pw_larger:weighted_vote | 28.56% | 891.4 | 0.902 | 0.280 |
| led24 | PFossil:pw_smaller:accuracy_vote | 67.31% | 812.8 | 0.834 | 0.044 |
| led24 | PFossil:pw_smaller:vote | 67.34% | 812.8 | 0.834 | 0.034 |
| led24 | PFossil:pw_smaller:weighted_vote | 30.75% | 812.8 | 0.834 | 0.298 |
| led24 | Pypper:ordered | 71.50% | 88.7 | 0.811 | 0.001 |
| led24 | Pypper:ovr | 68.00% | 124.0 | 1.032 | 0.002 |
| led24 | Pypper:pw_both:accuracy_vote | 72.31% | 500.9 | 3.624 | 0.066 |
| led24 | Pypper:pw_both:vote | 72.19% | 500.9 | 3.624 | 0.051 |
| led24 | Pypper:pw_both:weighted_vote | 72.28% | 500.9 | 3.624 | 0.455 |
| led24 | Pypper:pw_larger:accuracy_vote | 71.47% | 250.7 | 1.923 | 0.035 |
| led24 | Pypper:pw_larger:vote | 71.22% | 250.7 | 1.923 | 0.028 |
| led24 | Pypper:pw_larger:weighted_vote | 31.87% | 250.7 | 1.923 | 0.256 |
| led24 | Pypper:pw_smaller:accuracy_vote | 72.09% | 251.5 | 2.061 | 0.038 |
| led24 | Pypper:pw_smaller:vote | 71.97% | 251.5 | 2.061 | 0.036 |
| led24 | Pypper:pw_smaller:weighted_vote | 34.38% | 251.5 | 2.061 | 0.259 |
| letter | PFossil:ordered | 77.29% | 1170.6 | 8.592 | 0.012 |
| letter | PFossil:ovr | 75.67% | 1249.6 | 3.827 | 0.061 |
| letter | PFossil:pw_both:accuracy_vote | 86.04% | 11414.6 | 32.128 | 5.770 |
| letter | PFossil:pw_both:vote | 85.99% | 11414.6 | 32.128 | 4.259 |
| letter | PFossil:pw_both:weighted_vote | 86.30% | 11414.6 | 32.128 | 43.369 |
| letter | PFossil:pw_larger:accuracy_vote | 84.05% | 5755.2 | 26.607 | 2.936 |
| letter | PFossil:pw_larger:vote | 83.32% | 5755.2 | 26.607 | 2.244 |
| letter | PFossil:pw_larger:weighted_vote | 26.63% | 5755.2 | 26.607 | 21.446 |
| letter | PFossil:pw_smaller:accuracy_vote | 83.11% | 5675.0 | 26.390 | 2.905 |
| letter | PFossil:pw_smaller:vote | 82.84% | 5675.0 | 26.390 | 2.148 |
| letter | PFossil:pw_smaller:weighted_vote | 20.05% | 5675.0 | 26.390 | 22.109 |
| letter | Pypper:ordered | 81.39% | 1272.4 | 22.796 | 0.015 |
| letter | Pypper:ovr | 77.00% | 1662.0 | 30.573 | 0.094 |
| letter | Pypper:pw_both:accuracy_vote | 90.03% | 8894.4 | 81.682 | 5.549 |
| letter | Pypper:pw_both:vote | 89.98% | 8894.4 | 81.682 | 3.999 |
| letter | Pypper:pw_both:weighted_vote | 89.71% | 8894.4 | 81.682 | 41.941 |
| letter | Pypper:pw_larger:accuracy_vote | 88.29% | 4482.0 | 51.107 | 2.640 |
| letter | Pypper:pw_larger:vote | 88.28% | 4482.0 | 51.107 | 1.972 |
| letter | Pypper:pw_larger:weighted_vote | 27.89% | 4482.0 | 51.107 | 20.891 |
| letter | Pypper:pw_smaller:accuracy_vote | 87.93% | 4411.2 | 50.321 | 2.590 |
| letter | Pypper:pw_smaller:vote | 87.93% | 4411.2 | 50.321 | 2.244 |
| letter | Pypper:pw_smaller:weighted_vote | 22.22% | 4411.2 | 50.321 | 21.279 |
| optdigits | PFossil:ordered | 90.73% | 300.1 | 5.067 | 0.003 |
| optdigits | PFossil:ovr | 90.28% | 460.0 | 1.353 | 0.008 |
| optdigits | PFossil:pw_both:accuracy_vote | 95.66% | 1097.1 | 11.309 | 0.138 |
| optdigits | PFossil:pw_both:vote | 95.50% | 1097.1 | 11.309 | 0.115 |
| optdigits | PFossil:pw_both:weighted_vote | 95.85% | 1097.1 | 11.309 | 0.982 |
| optdigits | PFossil:pw_larger:accuracy_vote | 93.91% | 540.3 | 10.167 | 0.079 |
| optdigits | PFossil:pw_larger:vote | 93.91% | 540.3 | 10.167 | 0.050 |
| optdigits | PFossil:pw_larger:weighted_vote | 33.01% | 540.3 | 10.167 | 0.470 |
| optdigits | PFossil:pw_smaller:accuracy_vote | 93.91% | 558.9 | 10.415 | 0.073 |
| optdigits | PFossil:pw_smaller:vote | 93.99% | 558.9 | 10.415 | 0.054 |
| optdigits | PFossil:pw_smaller:weighted_vote | 40.18% | 558.9 | 10.415 | 0.529 |
| optdigits | Pypper:ordered | 90.91% | 199.5 | 7.181 | 0.003 |
| optdigits | Pypper:ovr | 90.18% | 301.1 | 4.855 | 0.007 |
| optdigits | Pypper:pw_both:accuracy_vote | 95.89% | 625.8 | 15.198 | 0.139 |
| optdigits | Pypper:pw_both:vote | 96.00% | 625.8 | 15.198 | 0.111 |
| optdigits | Pypper:pw_both:weighted_vote | 95.78% | 625.8 | 15.198 | 1.128 |
| optdigits | Pypper:pw_larger:accuracy_vote | 94.40% | 310.0 | 12.696 | 0.068 |
| optdigits | Pypper:pw_larger:vote | 94.07% | 310.0 | 12.696 | 0.045 |
| optdigits | Pypper:pw_larger:weighted_vote | 33.02% | 310.0 | 12.696 | 0.443 |
| optdigits | Pypper:pw_smaller:accuracy_vote | 94.64% | 315.9 | 12.273 | 0.075 |
| optdigits | Pypper:pw_smaller:vote | 94.54% | 315.9 | 12.273 | 0.048 |
| optdigits | Pypper:pw_smaller:weighted_vote | 44.31% | 315.9 | 12.273 | 0.533 |
| primary-tumor | PFossil:ordered | 32.44% | 330.7 | 0.250 | 0.002 |
| primary-tumor | PFossil:ovr | 43.38% | 341.1 | 0.228 | 0.002 |
| primary-tumor | PFossil:pw_both:accuracy_vote | 46.02% | 1464.9 | 1.310 | 0.093 |
| primary-tumor | PFossil:pw_both:vote | 37.16% | 1464.9 | 1.310 | 0.052 |
| primary-tumor | PFossil:pw_both:weighted_vote | 35.08% | 1464.9 | 1.310 | 0.296 |
| primary-tumor | PFossil:pw_larger:accuracy_vote | 39.84% | 652.7 | 0.798 | 0.035 |
| primary-tumor | PFossil:pw_larger:vote | 28.61% | 652.7 | 0.798 | 0.030 |
| primary-tumor | PFossil:pw_larger:weighted_vote | 24.78% | 652.7 | 0.798 | 0.199 |
| primary-tumor | PFossil:pw_smaller:accuracy_vote | 39.22% | 813.1 | 0.910 | 0.030 |
| primary-tumor | PFossil:pw_smaller:vote | 38.62% | 813.1 | 0.910 | 0.026 |
| primary-tumor | PFossil:pw_smaller:weighted_vote | 41.59% | 813.1 | 0.910 | 0.150 |
| primary-tumor | Pypper:ordered | 39.25% | 20.5 | 0.222 | 0.000 |
| primary-tumor | Pypper:ovr | 38.65% | 21.9 | 0.180 | 0.000 |
| primary-tumor | Pypper:pw_both:accuracy_vote | 47.20% | 260.0 | 2.885 | 0.038 |
| primary-tumor | Pypper:pw_both:vote | 44.55% | 260.0 | 2.885 | 0.026 |
| primary-tumor | Pypper:pw_both:weighted_vote | 27.15% | 260.0 | 2.885 | 0.179 |
| primary-tumor | Pypper:pw_larger:accuracy_vote | 43.97% | 139.7 | 1.986 | 0.021 |
| primary-tumor | Pypper:pw_larger:vote | 0.00% | 139.7 | 1.986 | 0.016 |
| primary-tumor | Pypper:pw_larger:weighted_vote | 24.78% | 139.7 | 1.986 | 0.107 |
| primary-tumor | Pypper:pw_smaller:accuracy_vote | 43.97% | 120.9 | 1.473 | 0.017 |
| primary-tumor | Pypper:pw_smaller:vote | 43.07% | 120.9 | 1.473 | 0.012 |
| primary-tumor | Pypper:pw_smaller:weighted_vote | 33.07% | 120.9 | 1.473 | 0.090 |
| segment | PFossil:ordered | 94.85% | 75.5 | 0.332 | 0.001 |
| segment | PFossil:ovr | 93.33% | 121.8 | 0.184 | 0.002 |
| segment | PFossil:pw_both:accuracy_vote | 95.15% | 180.8 | 0.675 | 0.024 |
| segment | PFossil:pw_both:vote | 95.32% | 180.8 | 0.675 | 0.021 |
| segment | PFossil:pw_both:weighted_vote | 95.54% | 180.8 | 0.675 | 0.155 |
| segment | PFossil:pw_larger:accuracy_vote | 94.81% | 103.4 | 0.566 | 0.013 |
| segment | PFossil:pw_larger:vote | 94.46% | 103.4 | 0.566 | 0.011 |
| segment | PFossil:pw_larger:weighted_vote | 47.06% | 103.4 | 0.566 | 0.083 |
| segment | PFossil:pw_smaller:accuracy_vote | 94.81% | 103.4 | 0.587 | 0.014 |
| segment | PFossil:pw_smaller:vote | 94.46% | 103.4 | 0.587 | 0.013 |
| segment | PFossil:pw_smaller:weighted_vote | 47.06% | 103.4 | 0.587 | 0.079 |
| segment | Pypper:ordered | 94.72% | 45.6 | 0.520 | 0.001 |
| segment | Pypper:ovr | 93.46% | 77.9 | 0.687 | 0.002 |
| segment | Pypper:pw_both:accuracy_vote | 95.41% | 127.3 | 1.367 | 0.023 |
| segment | Pypper:pw_both:vote | 95.54% | 127.3 | 1.367 | 0.018 |
| segment | Pypper:pw_both:weighted_vote | 95.37% | 127.3 | 1.367 | 0.136 |
| segment | Pypper:pw_larger:accuracy_vote | 94.76% | 62.8 | 0.894 | 0.012 |
| segment | Pypper:pw_larger:vote | 94.68% | 62.8 | 0.894 | 0.010 |
| segment | Pypper:pw_larger:weighted_vote | 52.73% | 62.8 | 0.894 | 0.067 |
| segment | Pypper:pw_smaller:accuracy_vote | 94.76% | 62.8 | 0.910 | 0.012 |
| segment | Pypper:pw_smaller:vote | 94.68% | 62.8 | 0.910 | 0.009 |
| segment | Pypper:pw_smaller:weighted_vote | 52.73% | 62.8 | 0.910 | 0.079 |
| solar-flare | PFossil:ordered | 70.64% | 50.0 | 0.070 | 0.000 |
| solar-flare | PFossil:ovr | 74.02% | 62.5 | 0.061 | 0.002 |
| solar-flare | PFossil:pw_both:accuracy_vote | 73.83% | 161.9 | 0.194 | 0.009 |
| solar-flare | PFossil:pw_both:vote | 74.30% | 161.9 | 0.194 | 0.008 |
| solar-flare | PFossil:pw_both:weighted_vote | 74.77% | 161.9 | 0.194 | 0.051 |
| solar-flare | PFossil:pw_larger:accuracy_vote | 74.77% | 73.0 | 0.119 | 0.004 |
| solar-flare | PFossil:pw_larger:vote | 73.93% | 73.0 | 0.119 | 0.003 |
| solar-flare | PFossil:pw_larger:weighted_vote | 58.82% | 73.0 | 0.119 | 0.024 |
| solar-flare | PFossil:pw_smaller:accuracy_vote | 72.23% | 88.9 | 0.133 | 0.004 |
| solar-flare | PFossil:pw_smaller:vote | 72.33% | 88.9 | 0.133 | 0.004 |
| solar-flare | PFossil:pw_smaller:weighted_vote | 71.39% | 88.9 | 0.133 | 0.023 |
| solar-flare | Pypper:ordered | 74.77% | 8.3 | 0.147 | 0.000 |
| solar-flare | Pypper:ovr | 69.43% | 21.4 | 0.219 | 0.000 |
| solar-flare | Pypper:pw_both:accuracy_vote | 73.92% | 54.3 | 0.641 | 0.007 |
| solar-flare | Pypper:pw_both:vote | 74.40% | 54.3 | 0.641 | 0.006 |
| solar-flare | Pypper:pw_both:weighted_vote | 74.40% | 54.3 | 0.641 | 0.043 |
| solar-flare | Pypper:pw_larger:accuracy_vote | 72.89% | 27.5 | 0.431 | 0.005 |
| solar-flare | Pypper:pw_larger:vote | 72.70% | 27.5 | 0.431 | 0.004 |
| solar-flare | Pypper:pw_larger:weighted_vote | 55.90% | 27.5 | 0.431 | 0.026 |
| solar-flare | Pypper:pw_smaller:accuracy_vote | 74.68% | 26.8 | 0.311 | 0.004 |
| solar-flare | Pypper:pw_smaller:vote | 74.77% | 26.8 | 0.311 | 0.003 |
| solar-flare | Pypper:pw_smaller:weighted_vote | 71.11% | 26.8 | 0.311 | 0.025 |
| soybean | PFossil:ordered | 92.98% | 70.2 | 0.260 | 0.001 |
| soybean | PFossil:ovr | 92.24% | 135.4 | 0.113 | 0.002 |
| soybean | PFossil:pw_both:accuracy_vote | 92.39% | 418.9 | 1.086 | 0.065 |
| soybean | PFossil:pw_both:vote | 93.12% | 418.9 | 1.086 | 0.050 |
| soybean | PFossil:pw_both:weighted_vote | 61.65% | 418.9 | 1.086 | 0.370 |
| soybean | PFossil:pw_larger:accuracy_vote | 93.26% | 207.0 | 0.899 | 0.030 |
| soybean | PFossil:pw_larger:vote | 93.70% | 207.0 | 0.899 | 0.024 |
| soybean | PFossil:pw_larger:weighted_vote | 37.02% | 207.0 | 0.899 | 0.160 |
| soybean | PFossil:pw_smaller:accuracy_vote | 93.41% | 201.5 | 0.923 | 0.027 |
| soybean | PFossil:pw_smaller:vote | 93.27% | 201.5 | 0.923 | 0.027 |
| soybean | PFossil:pw_smaller:weighted_vote | 75.25% | 201.5 | 0.923 | 0.147 |
| soybean | Pypper:ordered | 90.93% | 45.9 | 0.416 | 0.001 |
| soybean | Pypper:ovr | 89.91% | 58.8 | 0.322 | 0.001 |
| soybean | Pypper:pw_both:accuracy_vote | 93.12% | 373.4 | 2.719 | 0.061 |
| soybean | Pypper:pw_both:vote | 92.83% | 373.4 | 2.719 | 0.049 |
| soybean | Pypper:pw_both:weighted_vote | 61.79% | 373.4 | 2.719 | 0.323 |
| soybean | Pypper:pw_larger:accuracy_vote | 93.42% | 185.4 | 1.751 | 0.030 |
| soybean | Pypper:pw_larger:vote | 90.94% | 185.4 | 1.751 | 0.023 |
| soybean | Pypper:pw_larger:weighted_vote | 40.56% | 185.4 | 1.751 | 0.175 |
| soybean | Pypper:pw_smaller:accuracy_vote | 91.95% | 188.0 | 1.688 | 0.029 |
| soybean | Pypper:pw_smaller:vote | 91.80% | 188.0 | 1.688 | 0.022 |
| soybean | Pypper:pw_smaller:weighted_vote | 73.79% | 188.0 | 1.688 | 0.147 |
| texture | PFossil:ordered | 90.45% | 219.7 | 3.155 | 0.002 |
| texture | PFossil:ovr | 92.22% | 385.6 | 1.029 | 0.011 |
| texture | PFossil:pw_both:accuracy_vote | 94.05% | 674.9 | 6.252 | 0.127 |
| texture | PFossil:pw_both:vote | 93.58% | 674.9 | 6.252 | 0.103 |
| texture | PFossil:pw_both:weighted_vote | 94.58% | 674.9 | 6.252 | 0.923 |
| texture | PFossil:pw_larger:accuracy_vote | 92.35% | 315.6 | 5.867 | 0.069 |
| texture | PFossil:pw_larger:vote | 91.71% | 315.6 | 5.867 | 0.056 |
| texture | PFossil:pw_larger:weighted_vote | 30.64% | 315.6 | 5.867 | 0.442 |
| texture | PFossil:pw_smaller:accuracy_vote | 92.35% | 315.6 | 5.553 | 0.078 |
| texture | PFossil:pw_smaller:vote | 91.71% | 315.6 | 5.553 | 0.057 |
| texture | PFossil:pw_smaller:weighted_vote | 30.64% | 315.6 | 5.553 | 0.450 |
| texture | Pypper:ordered | 91.56% | 188.5 | 4.990 | 0.002 |
| texture | Pypper:ovr | 91.78% | 309.9 | 4.394 | 0.008 |
| texture | Pypper:pw_both:accuracy_vote | 95.33% | 486.8 | 9.328 | 0.127 |
| texture | Pypper:pw_both:vote | 94.91% | 486.8 | 9.328 | 0.114 |
| texture | Pypper:pw_both:weighted_vote | 95.15% | 486.8 | 9.328 | 0.893 |
| texture | Pypper:pw_larger:accuracy_vote | 94.53% | 252.7 | 7.535 | 0.064 |
| texture | Pypper:pw_larger:vote | 94.38% | 252.7 | 7.535 | 0.049 |
| texture | Pypper:pw_larger:weighted_vote | 32.73% | 252.7 | 7.535 | 0.423 |
| texture | Pypper:pw_smaller:accuracy_vote | 94.53% | 252.7 | 7.450 | 0.063 |
| texture | Pypper:pw_smaller:vote | 94.38% | 252.7 | 7.450 | 0.051 |
| texture | Pypper:pw_smaller:weighted_vote | 32.73% | 252.7 | 7.450 | 0.428 |
| vowel | PFossil:ordered | 80.40% | 247.7 | 0.278 | 0.002 |
| vowel | PFossil:ovr | 77.27% | 401.3 | 0.287 | 0.004 |
| vowel | PFossil:pw_both:accuracy_vote | 83.54% | 744.6 | 0.894 | 0.036 |
| vowel | PFossil:pw_both:vote | 81.82% | 744.6 | 0.894 | 0.033 |
| vowel | PFossil:pw_both:weighted_vote | 82.22% | 744.6 | 0.894 | 0.196 |
| vowel | PFossil:pw_larger:accuracy_vote | 82.73% | 372.1 | 0.615 | 0.022 |
| vowel | PFossil:pw_larger:vote | 81.31% | 372.1 | 0.615 | 0.016 |
| vowel | PFossil:pw_larger:weighted_vote | 35.96% | 372.1 | 0.615 | 0.107 |
| vowel | PFossil:pw_smaller:accuracy_vote | 82.73% | 372.1 | 0.631 | 0.020 |
| vowel | PFossil:pw_smaller:vote | 81.31% | 372.1 | 0.631 | 0.016 |
| vowel | PFossil:pw_smaller:weighted_vote | 35.96% | 372.1 | 0.631 | 0.113 |
| vowel | Pypper:ordered | 69.70% | 123.4 | 0.681 | 0.001 |
| vowel | Pypper:ovr | 63.84% | 171.5 | 0.784 | 0.002 |
| vowel | Pypper:pw_both:accuracy_vote | 79.70% | 509.1 | 2.739 | 0.035 |
| vowel | Pypper:pw_both:vote | 78.99% | 509.1 | 2.739 | 0.031 |
| vowel | Pypper:pw_both:weighted_vote | 78.48% | 509.1 | 2.739 | 0.191 |
| vowel | Pypper:pw_larger:accuracy_vote | 77.88% | 254.8 | 1.626 | 0.017 |
| vowel | Pypper:pw_larger:vote | 78.59% | 254.8 | 1.626 | 0.014 |
| vowel | Pypper:pw_larger:weighted_vote | 37.07% | 254.8 | 1.626 | 0.111 |
| vowel | Pypper:pw_smaller:accuracy_vote | 77.88% | 254.8 | 1.528 | 0.016 |
| vowel | Pypper:pw_smaller:vote | 78.59% | 254.8 | 1.528 | 0.014 |
| vowel | Pypper:pw_smaller:weighted_vote | 37.07% | 254.8 | 1.528 | 0.096 |
| yeast | PFossil:ordered | 56.06% | 148.8 | 0.218 | 0.001 |
| yeast | PFossil:ovr | 57.48% | 180.5 | 0.180 | 0.002 |
| yeast | PFossil:pw_both:accuracy_vote | 59.57% | 898.5 | 0.888 | 0.037 |
| yeast | PFossil:pw_both:vote | 59.77% | 898.5 | 0.888 | 0.031 |
| yeast | PFossil:pw_both:weighted_vote | 51.76% | 898.5 | 0.888 | 0.212 |
| yeast | PFossil:pw_larger:accuracy_vote | 57.28% | 337.3 | 0.484 | 0.022 |
| yeast | PFossil:pw_larger:vote | 56.54% | 337.3 | 0.484 | 0.016 |
| yeast | PFossil:pw_larger:weighted_vote | 35.37% | 337.3 | 0.484 | 0.142 |
| yeast | PFossil:pw_smaller:accuracy_vote | 57.34% | 561.2 | 0.573 | 0.019 |
| yeast | PFossil:pw_smaller:vote | 57.48% | 561.2 | 0.573 | 0.017 |
| yeast | PFossil:pw_smaller:weighted_vote | 58.49% | 561.2 | 0.573 | 0.107 |
| yeast | Pypper:ordered | 58.96% | 38.7 | 0.533 | 0.001 |
| yeast | Pypper:ovr | 59.30% | 66.8 | 0.695 | 0.001 |
| yeast | Pypper:pw_both:accuracy_vote | 59.16% | 222.1 | 2.400 | 0.029 |
| yeast | Pypper:pw_both:vote | 58.89% | 222.1 | 2.400 | 0.023 |
| yeast | Pypper:pw_both:weighted_vote | 49.73% | 222.1 | 2.400 | 0.200 |
| yeast | Pypper:pw_larger:accuracy_vote | 57.68% | 108.3 | 1.478 | 0.017 |
| yeast | Pypper:pw_larger:vote | 58.15% | 108.3 | 1.478 | 0.013 |
| yeast | Pypper:pw_larger:weighted_vote | 36.45% | 108.3 | 1.478 | 0.108 |
| yeast | Pypper:pw_smaller:accuracy_vote | 59.23% | 113.8 | 1.169 | 0.015 |
| yeast | Pypper:pw_smaller:vote | 58.42% | 113.8 | 1.169 | 0.011 |
| yeast | Pypper:pw_smaller:weighted_vote | 59.30% | 113.8 | 1.169 | 0.081 |
| zoo | PFossil:ordered | 93.00% | 16.7 | 0.018 | 0.000 |
| zoo | PFossil:ovr | 95.00% | 21.0 | 0.015 | 0.000 |
| zoo | PFossil:pw_both:accuracy_vote | 94.00% | 45.6 | 0.068 | 0.003 |
| zoo | PFossil:pw_both:vote | 93.00% | 45.6 | 0.068 | 0.002 |
| zoo | PFossil:pw_both:weighted_vote | 90.00% | 45.6 | 0.068 | 0.008 |
| zoo | PFossil:pw_larger:accuracy_vote | 93.00% | 22.7 | 0.047 | 0.001 |
| zoo | PFossil:pw_larger:vote | 93.00% | 22.7 | 0.047 | 0.001 |
| zoo | PFossil:pw_larger:weighted_vote | 60.36% | 22.7 | 0.047 | 0.004 |
| zoo | PFossil:pw_smaller:accuracy_vote | 94.00% | 22.9 | 0.045 | 0.001 |
| zoo | PFossil:pw_smaller:vote | 94.00% | 22.9 | 0.045 | 0.001 |
| zoo | PFossil:pw_smaller:weighted_vote | 93.00% | 22.9 | 0.045 | 0.004 |
| zoo | Pypper:ordered | 88.09% | 8.0 | 0.047 | 0.000 |
| zoo | Pypper:ovr | 90.09% | 10.3 | 0.050 | 0.000 |
| zoo | Pypper:pw_both:accuracy_vote | 91.00% | 39.8 | 0.236 | 0.002 |
| zoo | Pypper:pw_both:vote | 91.00% | 39.8 | 0.236 | 0.002 |
| zoo | Pypper:pw_both:weighted_vote | 84.18% | 39.8 | 0.236 | 0.008 |
| zoo | Pypper:pw_larger:accuracy_vote | 91.00% | 19.8 | 0.153 | 0.001 |
| zoo | Pypper:pw_larger:vote | 90.09% | 19.8 | 0.153 | 0.001 |
| zoo | Pypper:pw_larger:weighted_vote | 60.36% | 19.8 | 0.153 | 0.004 |
| zoo | Pypper:pw_smaller:accuracy_vote | 91.00% | 20.0 | 0.134 | 0.001 |
| zoo | Pypper:pw_smaller:vote | 89.09% | 20.0 | 0.134 | 0.001 |
| zoo | Pypper:pw_smaller:weighted_vote | 88.09% | 20.0 | 0.134 | 0.004 |

## Why the weighted vote does poorly

The weighted vote is clearly worse than the plain vote here, and much slower to predict. Both have the same cause in how it works: a pair model's vote is weighted by the Laplace estimate of the rule that decided it, and split -- weight `p` to the class it predicts, `1 - p` to the other class of its pair. Prediction is slower because it has to find, for every example and every pair model, the deciding rule among all covering rules, instead of just taking each model's prediction.

Most of a pair model's decisions are made by its default rule: of the c(c-1)/2 pair models, only c - 1 have the example's own class as one of theirs, so most of them see an example of neither of their classes, and their rules rarely fire. pyrulearn annotates every rule with the training examples its body covers, independently of the other rules -- so a rule means the same in every model it is part of, and the counts don't change when a rule set is reordered or turned into a decision list. The default rule's body is empty ("true"), so it is counted over all of its pair's examples, and its Laplace estimate is simply its class's share of the pair. For a balanced pair that is about 0.5: the default votes are almost neutral, while a rule that fires -- rightly or wrongly -- votes with a weight near 1. With the larger class as the pair's target, the default is the smaller class and its share is below 0.5, so the split gives most of the default's vote to the *other* class, against the model's own prediction.

To check that this is the cause, the default rules' counts are swapped, for prediction only, for counts over the examples each default actually decides (those no rule covers), on 4 small datasets (`primary-tumor`, `soybean`, `vowel`, `segment`; the first two with very unbalanced, the last two with balanced classes), 3-fold. Mean accuracy:

| base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|--:|--:|--:|--:|
| PFossil | larger | 73.94% | 76.25% | 36.04% | 76.02% |
| PFossil | smaller | 76.66% | 77.54% | 50.08% | 77.32% |
| Pypper | larger | 63.75% | 75.36% | 36.22% | 74.39% |
| Pypper | smaller | 74.84% | 74.74% | 47.51% | 72.74% |

Counted over the examples it decides, the default rule gets a weight near its real reliability, and the weighted vote comes up to the level of the other two -- but not above it. The two ways of counting the default are two different meanings of a rule's counts (a statement in its own right vs. its role in this particular model); pyrulearn keeps the first, so the weighted vote stays as it is. The result: when most decisions come from default rules, rule-based vote weights add little, and the plain or the accuracy-weighted vote is the better choice.

Per dataset:

| dataset | base learner | pair target | vote | accuracy-weighted | weighted (default over all rows) | weighted (default over the rows it decides) |
|---|---|---|--:|--:|--:|--:|
| primary-tumor | PFossil | larger | 30.09% | 37.46% | 24.78% | 38.35% |
| primary-tumor | PFossil | smaller | 40.41% | 42.48% | 40.71% | 44.25% |
| primary-tumor | Pypper | larger | 0.29% | 44.25% | 24.78% | 40.71% |
| primary-tumor | Pypper | smaller | 41.89% | 42.48% | 32.15% | 38.05% |
| segment | PFossil | larger | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | PFossil | smaller | 92.60% | 92.81% | 44.50% | 93.42% |
| segment | Pypper | larger | 94.63% | 94.50% | 46.41% | 94.37% |
| segment | Pypper | smaller | 94.63% | 94.50% | 46.41% | 94.37% |
| soybean | PFossil | larger | 91.95% | 92.39% | 37.19% | 91.22% |
| soybean | PFossil | smaller | 92.53% | 92.54% | 77.45% | 90.48% |
| soybean | Pypper | larger | 88.44% | 92.10% | 38.95% | 91.36% |
| soybean | Pypper | smaller | 91.22% | 91.37% | 76.72% | 87.41% |
| vowel | PFossil | larger | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | PFossil | smaller | 81.11% | 82.32% | 37.68% | 81.11% |
| vowel | Pypper | larger | 71.62% | 70.61% | 34.75% | 71.11% |
| vowel | Pypper | smaller | 71.62% | 70.61% | 34.75% | 71.11% |

