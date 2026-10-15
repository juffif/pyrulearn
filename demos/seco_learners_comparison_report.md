# Separate-and-conquer rule learners compared

## Setup

**Full run** (29 datasets). Quick sanity check instead: `python demos/seco_learners_comparison.py` with no arguments.

pyrulearn's separate-and-conquer rule learners compared with each other
and with two external baselines, on 18 binary and
11 multi-class datasets from `pyrulearn.experiments.catalog`.
All of them learn one rule at a time and remove the examples it covers;
they differ in how a rule is searched for, when it stops, and whether and
how the rule set is pruned. Every learner runs with its default settings,
including its default way of handling several classes.

The datasets have mostly symbolic attributes (at least half nominal), the
kind these learners were designed for -- numeric attributes and their
discretization are the topic of the numeric-discretization demo. Artificial
concepts (the monks problems, m-of-n, Hayes-Roth, LED) and datasets with
many classes (left for the multi-class demo) are not included.

- **CN2** -- Beam search (width 5) with the Laplace estimate; CN2's likelihood-ratio significance test stops rules that aren't significant. Learns rules for every class (one-vs-rest rule set).
- **PFoil** -- FOIL: greedy search that adds the condition with the highest FOIL gain; Quinlan's encoding-length restriction stops rules that cost more bits than the examples they explain. Rules for every class.
- **PFossil** -- FOSSIL: hill climbing on the correlation between rule and class; rules below a correlation of 0.3 are dropped. Rules for every class.
- **Pypper** -- pyrulearn's RIPPER: grow on two thirds of the data, prune on the rest (IREP*), then optimize the rule set (replace/revise). Rules for the classes from least to most frequent, the most frequent one as default.
- **Opus(WRAcc)** -- Opus: in every covering step the rule with the highest weighted relative accuracy among all conjunctions of up to 3 conditions, found by exhaustive branch-and-bound search (OPUS). Rules for every class.
- **PyLORD** -- A simplified LORD: for every training example, the best rule covering it (greedy m-estimate search, then pruning); a test example is classified by the best rule that covers it. Many overlapping rules.
- **Weka:JRip** -- Weka's RIPPER, run as a subprocess on the same binarized data. Fit time includes the JVM start.
- **JavaLord** -- The reference LORD implementation (Java), run as a subprocess on the same binarized data, and, in this demo only, on a single thread: by default it uses half the machine's cores, which no other learner here does.

AQR, the AQ baseline CN2 was designed to improve on, and Opus with the
Laplace estimate were checked separately and left out -- see "Why AQR and
Opus(Laplace) aren't in the main comparison".
Across 4 small datasets (3-fold), AQR was less accurate than CN2 (mean accuracy 70.27% vs. 75.08%), learned rules 7.4x as long (17.6 vs. 2.4 conditions per rule; 494.3 vs. 22.6 conditions in total), and took 120x as long per fit (5.1s vs. 0.04s, mean) -- requiring every rule to be consistent overfits noisy data, and on larger datasets with numeric attributes the fit times would dominate the whole comparison. Opus with the Laplace estimate (at most 3 conditions per rule), which finds the rule CN2's beam search only approximates, was not less accurate than CN2 (mean accuracy 76.60%), learned 4.5x as many rules (41.5 vs. 9.2, mean) and took 2158x as long per fit (91.4s, mean). Webb (1993) found classifiers that optimize the Laplace estimate exactly often less accurate than those of a heuristic search that fails to -- the result behind Quinlan & Cameron-Jones's 'oversearching'. The main comparison runs Opus with WRAcc instead.

Protocol: 10-fold stratified cross-validation (5-fold for the large adult, connect-4)
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals=8)`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical binary
features per fold (as N-lists, the library's default representation; the learners
that need the dense matrix convert them). Each fit is capped at 300s; a
time-out or an error counts as a failure (and as last in the ranking),
not as the end of the run. The cap is deliberately kept for the large
datasets too: which learners scale to tens of thousands of examples is
part of the result (a single CN2 fit on adult takes about 20 minutes).

Every learner runs single-threaded; the reference Java LORD, which by default uses half the machine's cores, is limited to one thread here (`thread_count=1`, for this comparison only). Fits run 10 at a time (`run_cv`'s `n_jobs`), each in its own process and timed there, with the numerical libraries limited to one thread (`OMP_NUM_THREADS=1` and its relatives). A check of the `dense_vs_modular` learners on 11 datasets found the fit times of 5 concurrent fits as close to a sequential run as a second sequential run (per-learner ratios and rankings alike); time ratios between learners are reproducible to about +-10% either way.

Measures: test accuracy, number of rules and conditions, fit time.

![accuracy vs. rule-set complexity](seco_learners_comparison_plots/seco_learners_comparison_accuracy.png)

![fit time per dataset](seco_learners_comparison_plots/seco_learners_comparison_fit_time.png)

![critical-difference diagram (accuracy)](seco_learners_comparison_plots/seco_learners_comparison_cd.png)

## Why AQR and Opus(Laplace) aren't in the main comparison

Across 4 small datasets (3-fold), AQR was less accurate than CN2 (mean accuracy 70.27% vs. 75.08%), learned rules 7.4x as long (17.6 vs. 2.4 conditions per rule; 494.3 vs. 22.6 conditions in total), and took 120x as long per fit (5.1s vs. 0.04s, mean) -- requiring every rule to be consistent overfits noisy data, and on larger datasets with numeric attributes the fit times would dominate the whole comparison. Opus with the Laplace estimate (at most 3 conditions per rule), which finds the rule CN2's beam search only approximates, was not less accurate than CN2 (mean accuracy 76.60%), learned 4.5x as many rules (41.5 vs. 9.2, mean) and took 2158x as long per fit (91.4s, mean). Webb (1993) found classifiers that optimize the Laplace estimate exactly often less accurate than those of a heuristic search that fails to -- the result behind Quinlan & Cameron-Jones's 'oversearching'. The main comparison runs Opus with WRAcc instead.

| dataset | learner | accuracy | n_rules | n_conditions | conds_per_rule | fit_time |
|---|---|---|---|---|---|---|
| breast-cancer | AQR | 57.70% | 27.3 | 489.0 | 17.7 | 3.623 |
| breast-cancer | CN2 | 70.28% | 3.3 | 8.3 | 2.5 | 0.019 |
| breast-cancer | Opus(Laplace) | 66.44% | 58.7 | 171.7 | 2.9 | 57.206 |
| credit-approval | AQR | 75.94% | 37.0 | 1155.0 | 31.3 | 13.541 |
| credit-approval | CN2 | 78.99% | 16.3 | 46.0 | 2.8 | 0.074 |
| credit-approval | Opus(Laplace) | 81.96% | 75.0 | 217.5 | 2.9 | 277.231 |
| heart-statlog | AQR | 67.41% | 17.0 | 277.0 | 16.1 | 2.477 |
| heart-statlog | CN2 | 70.37% | 8.0 | 19.3 | 2.4 | 0.034 |
| heart-statlog | Opus(Laplace) | 81.11% | 29.0 | 85.0 | 2.9 | 28.547 |
| hepatitis | AQR | 80.03% | 10.3 | 56.3 | 5.4 | 0.680 |
| hepatitis | CN2 | 80.67% | 9.0 | 16.7 | 1.9 | 0.043 |
| hepatitis | Opus(Laplace) | 78.70% | 14.3 | 35.3 | 2.5 | 2.651 |

## Summary by learner (mean across the 27 datasets and their folds)

Without the two large datasets (adult, connect-4), where most learners hit the time cap -- see the section on them below. On these 27 datasets every learner has a result for (almost) every fold, so the means compare like with like.

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| PFoil | 82.52% | 56.9 | 243.9 | 3.6 | 0.21 | 4.2 | 3.57 | 0 |
| PyLORD | 82.70% | 133.7 | 695.0 | 4.0 | 4.69 | 2.2 | 3.87 | 0 |
| JavaLord | 81.84% | 129.2 | 618.0 | 3.7 | 8.58 | 4.2 | 4.00 | 0 |
| PFossil | 81.80% | 13.1 | 47.1 | 4.1 | 0.06 | 5.0 | 4.09 | 0 |
| Pypper | 81.27% | 4.9 | 16.2 | 2.8 | 0.20 | 6.0 | 4.37 | 0 |
| Weka:JRip | 81.02% | 5.6 | 18.7 | 2.8 | 1.40 | 2.5 | 4.43 | 0 |
| Opus(WRAcc) | 78.64% | 9.8 | 25.5 | 2.6 | 39.31 | 2.5 | 5.54 | 19 |
| CN2 | 78.21% | 22.8 | 78.2 | 2.8 | 0.27 | 0.2 | 6.13 | 0 |

Sorted by mean rank. `wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across datasets, failures tied for last. Means over successful fits only; `failures` counts the fits that timed out or raised.

### Small datasets only (17, fewer than 1,000 examples)

`SPECT`, `analcatdata_dmft`, `anneal`, `breast-cancer`, `colic`, `credit-approval`, `cylinder-bands`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `hepatitis`, `lymph`, `molecular-biology_promoters`, `tic-tac-toe`, `vote`, `zoo`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| PFoil | 79.46% | 44.2 | 153.4 | 3.1 | 0.11 | 3.0 | 3.59 | 0 |
| PyLORD | 79.68% | 92.5 | 400.0 | 3.5 | 0.90 | 1.0 | 3.88 | 0 |
| JavaLord | 78.26% | 87.3 | 356.9 | 3.2 | 0.58 | 3.0 | 4.32 | 0 |
| PFossil | 78.24% | 11.9 | 39.4 | 3.9 | 0.03 | 3.0 | 4.38 | 0 |
| Weka:JRip | 77.33% | 3.7 | 8.7 | 2.4 | 0.65 | 1.5 | 4.50 | 0 |
| Pypper | 77.44% | 3.1 | 7.5 | 2.4 | 0.07 | 3.0 | 4.71 | 0 |
| Opus(WRAcc) | 77.26% | 9.6 | 25.0 | 2.6 | 28.45 | 2.5 | 4.74 | 9 |
| CN2 | 76.13% | 11.9 | 28.5 | 2.4 | 0.06 | 0.0 | 5.88 | 0 |

### Medium datasets only (10, 1,000 to 10,000 examples)

`car`, `cmc`, `credit-g`, `dna`, `hypothyroid`, `kr-vs-kp`, `mushroom`, `sick`, `solar-flare`, `splice`

| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | wins | mean rank | failures |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| JavaLord | 87.91% | 200.6 | 1061.8 | 4.5 | 22.18 | 1.2 | 3.45 | 0 |
| PFoil | 87.73% | 78.5 | 397.8 | 4.5 | 0.38 | 1.2 | 3.55 | 0 |
| PFossil | 87.84% | 15.3 | 60.1 | 4.2 | 0.10 | 2.0 | 3.60 | 0 |
| Pypper | 87.78% | 7.9 | 30.9 | 3.4 | 0.41 | 3.0 | 3.80 | 0 |
| PyLORD | 87.84% | 203.8 | 1196.6 | 4.9 | 11.14 | 1.2 | 3.85 | 0 |
| Weka:JRip | 87.28% | 8.8 | 35.8 | 3.6 | 2.67 | 1.0 | 4.30 | 0 |
| CN2 | 81.75% | 41.3 | 162.6 | 3.5 | 0.63 | 0.2 | 6.55 | 0 |
| Opus(WRAcc) | 81.10% | 10.1 | 26.6 | 2.6 | 57.78 | 0.0 | 6.90 | 10 |

## Mean accuracy by target type

| learner | binary | multi-class |
|---|--:|--:|
| PFoil | 83.97% | 80.06% |
| PyLORD | 84.25% | 80.06% |
| JavaLord | 83.29% | 79.37% |
| PFossil | 82.71% | 80.24% |
| Pypper | 83.00% | 78.33% |
| Weka:JRip | 82.56% | 78.39% |
| Opus(WRAcc) | 81.69% | 73.18% |
| CN2 | 80.70% | 73.99% |

## The large datasets: adult, connect-4

5-fold cross-validation, with the same 300s cap per fit as everywhere else -- which learners get through tens of thousands of examples at all is the result here. `folds` counts the folds that finished; accuracy, size and fit time are means over those folds only.

| dataset | learner | folds | accuracy | n_rules | n_conditions | fit_time (s) |
|---|---|--:|--:|--:|--:|--:|
| adult | CN2 | 5/5 | 79.10% | 398.2 | 1563.4 | 22.22 |
| adult | JavaLord | 0/5 | -- | -- | -- | -- |
| adult | Opus(WRAcc) | 0/5 | -- | -- | -- | -- |
| adult | PFoil | 5/5 | 85.88% | 1695.8 | 16910.0 | 86.14 |
| adult | PFossil | 5/5 | 82.22% | 9.2 | 56.6 | 0.72 |
| adult | PyLORD | 0/5 | -- | -- | -- | -- |
| adult | Pypper | 5/5 | 84.22% | 13.2 | 79.2 | 8.51 |
| adult | Weka:JRip | 5/5 | 84.87% | 22.8 | 138.0 | 115.55 |
| connect-4 | CN2 | 5/5 | 67.97% | 270.2 | 1251.0 | 25.10 |
| connect-4 | JavaLord | 0/5 | -- | -- | -- | -- |
| connect-4 | Opus(WRAcc) | 0/5 | -- | -- | -- | -- |
| connect-4 | PFoil | 0/5 | -- | -- | -- | -- |
| connect-4 | PFossil | 5/5 | 65.83% | 0.0 | 0.0 | 0.99 |
| connect-4 | PyLORD | 0/5 | -- | -- | -- | -- |
| connect-4 | Pypper | 4/5 | 76.59% | 67.8 | 579.8 | 171.34 |
| connect-4 | Weka:JRip | 0/5 | -- | -- | -- | -- |

## Per-dataset results, per learner (mean across folds)

| dataset | learner | accuracy | n_rules | n_conditions | fit_time |
|---|---|---|---|---|---|
| SPECT | CN2 | 80.90% | 8.3 | 19.0 | 0.040 |
| SPECT | JavaLord | 84.30% | 60.9 | 253.9 | 0.278 |
| SPECT | Opus(WRAcc) | 78.65% | 9.8 | 20.6 | 0.833 |
| SPECT | PFoil | 82.05% | 25.8 | 82.1 | 0.054 |
| SPECT | PFossil | 78.68% | 9.7 | 41.1 | 0.030 |
| SPECT | PyLORD | 83.93% | 61.7 | 276.2 | 0.867 |
| SPECT | Pypper | 79.00% | 0.9 | 5.0 | 0.033 |
| SPECT | Weka:JRip | 80.87% | 1.0 | 5.5 | 0.459 |
| analcatdata_dmft | CN2 | 19.07% | 7.2 | 25.0 | 0.098 |
| analcatdata_dmft | JavaLord | 19.08% | 470.8 | 2781.4 | 0.942 |
| analcatdata_dmft | Opus(WRAcc) | 21.69% | 24.6 | 71.1 | 3.981 |
| analcatdata_dmft | PFoil | 19.07% | 253.7 | 973.8 | 0.633 |
| analcatdata_dmft | PFossil | 19.95% | 0.3 | 3.3 | 0.028 |
| analcatdata_dmft | PyLORD | 18.58% | 480.7 | 3038.0 | 3.515 |
| analcatdata_dmft | Pypper | 18.94% | 1.0 | 2.8 | 0.105 |
| analcatdata_dmft | Weka:JRip | 19.45% | 1.0 | 3.5 | 0.590 |
| anneal | CN2 | 98.78% | 27.9 | 61.2 | 0.171 |
| anneal | JavaLord | 90.87% | 30.7 | 47.1 | 0.873 |
| anneal | Opus(WRAcc) | 97.33% | 11.9 | 25.6 | 1.020 |
| anneal | PFoil | 97.55% | 26.9 | 74.5 | 0.066 |
| anneal | PFossil | 99.78% | 19.6 | 34.5 | 0.036 |
| anneal | PyLORD | 98.55% | 47.9 | 145.2 | 1.107 |
| anneal | Pypper | 94.10% | 9.9 | 23.5 | 0.163 |
| anneal | Weka:JRip | 89.53% | 7.8 | 13.6 | 0.871 |
| breast-cancer | CN2 | 70.30% | 1.6 | 3.6 | 0.008 |
| breast-cancer | JavaLord | 71.37% | 92.7 | 417.9 | 0.319 |
| breast-cancer | Opus(WRAcc) | 71.34% | 7.9 | 22.9 | 3.727 |
| breast-cancer | PFoil | 72.02% | 43.2 | 156.8 | 0.091 |
| breast-cancer | PFossil | 69.26% | 7.4 | 30.0 | 0.024 |
| breast-cancer | PyLORD | 72.04% | 93.9 | 447.1 | 0.672 |
| breast-cancer | Pypper | 74.84% | 1.0 | 2.0 | 0.026 |
| breast-cancer | Weka:JRip | 71.37% | 1.3 | 2.5 | 0.441 |
| car | CN2 | 93.98% | 64.8 | 343.8 | 0.642 |
| car | JavaLord | 95.83% | 104.1 | 637.1 | 1.524 |
| car | Opus(WRAcc) | 81.94% | 8.0 | 19.0 | 0.264 |
| car | PFoil | 96.70% | 75.4 | 449.1 | 0.297 |
| car | PFossil | 93.87% | 16.2 | 59.1 | 0.057 |
| car | PyLORD | 95.37% | 106.2 | 697.2 | 3.227 |
| car | Pypper | 93.63% | 17.1 | 86.4 | 0.525 |
| car | Weka:JRip | 94.33% | 18.2 | 90.1 | 0.963 |
| cmc | CN2 | 45.35% | 22.0 | 81.7 | 0.226 |
| cmc | JavaLord | 52.76% | 651.9 | 4472.5 | 2.892 |
| cmc | Opus(WRAcc) | 48.13% | 16.5 | 47.9 | 8.872 |
| cmc | PFoil | 51.40% | 211.0 | 1266.9 | 0.817 |
| cmc | PFossil | 52.61% | 2.1 | 9.2 | 0.024 |
| cmc | PyLORD | 52.08% | 662.7 | 5357.8 | 8.298 |
| cmc | Pypper | 54.79% | 4.1 | 12.6 | 0.268 |
| cmc | Weka:JRip | 51.94% | 3.6 | 14.5 | 0.831 |
| colic | CN2 | 75.26% | 16.2 | 41.0 | 0.069 |
| colic | JavaLord | 78.26% | 74.5 | 205.9 | 0.614 |
| colic | Opus(WRAcc) | 86.70% | 9.5 | 22.8 | 5.073 |
| colic | PFoil | 84.52% | 38.5 | 110.0 | 0.079 |
| colic | PFossil | 82.89% | 19.8 | 50.7 | 0.039 |
| colic | PyLORD | 83.45% | 85.0 | 250.7 | 0.613 |
| colic | Pypper | 85.08% | 1.8 | 4.0 | 0.039 |
| colic | Weka:JRip | 84.50% | 3.4 | 8.4 | 0.699 |
| credit-approval | CN2 | 78.70% | 19.8 | 59.8 | 0.116 |
| credit-approval | JavaLord | 83.62% | 126.5 | 499.1 | 0.995 |
| credit-approval | Opus(WRAcc) | 84.78% | 8.6 | 24.8 | 10.214 |
| credit-approval | PFoil | 84.49% | 57.5 | 220.2 | 0.169 |
| credit-approval | PFossil | 82.17% | 8.8 | 38.9 | 0.038 |
| credit-approval | PyLORD | 84.20% | 127.8 | 525.7 | 1.468 |
| credit-approval | Pypper | 83.91% | 3.8 | 11.0 | 0.128 |
| credit-approval | Weka:JRip | 84.78% | 4.2 | 10.1 | 0.767 |
| credit-g | CN2 | 71.60% | 22.8 | 63.4 | 0.116 |
| credit-g | JavaLord | 75.20% | 258.0 | 985.1 | 1.946 |
| credit-g | Opus(WRAcc) | 72.50% | 10.7 | 31.7 | 65.352 |
| credit-g | PFoil | 73.80% | 98.4 | 464.6 | 0.341 |
| credit-g | PFossil | 73.50% | 3.6 | 31.1 | 0.037 |
| credit-g | PyLORD | 75.00% | 260.2 | 1003.9 | 2.413 |
| credit-g | Pypper | 72.00% | 2.5 | 8.9 | 0.117 |
| credit-g | Weka:JRip | 71.80% | 3.9 | 15.9 | 1.018 |
| cylinder-bands | CN2 | 69.44% | 25.1 | 35.2 | 0.088 |
| cylinder-bands | JavaLord | 72.96% | 129.5 | 303.1 | 1.375 |
| cylinder-bands | Opus(WRAcc) | 75.93% | 8.0 | 22.0 | 298.135 |
| cylinder-bands | PFoil | 78.70% | 49.6 | 183.0 | 0.172 |
| cylinder-bands | PFossil | 70.37% | 9.4 | 64.4 | 0.063 |
| cylinder-bands | PyLORD | 78.33% | 128.2 | 306.1 | 1.015 |
| cylinder-bands | Pypper | 65.00% | 4.3 | 13.6 | 0.134 |
| cylinder-bands | Weka:JRip | 64.26% | 6.9 | 18.1 | 1.468 |
| dna | CN2 | 84.78% | 105.9 | 506.9 | 2.312 |
| dna | JavaLord | 91.78% | 323.0 | 1471.8 | 65.327 |
| dna | Opus(WRAcc) | 67.73% | 11.2 | 28.6 | 183.065 |
| dna | PFoil | 92.50% | 89.0 | 469.4 | 0.660 |
| dna | PFossil | 93.69% | 27.8 | 137.4 | 0.227 |
| dna | PyLORD | 91.78% | 320.8 | 1463.3 | 16.671 |
| dna | Pypper | 92.22% | 13.2 | 68.2 | 0.905 |
| dna | Weka:JRip | 91.96% | 17.3 | 97.4 | 5.631 |
| dresses-sales | CN2 | 58.20% | 3.2 | 5.7 | 0.018 |
| dresses-sales | JavaLord | 58.80% | 171.3 | 514.3 | 1.010 |
| dresses-sales | Opus(WRAcc) | 59.20% | 10.4 | 30.0 | 141.929 |
| dresses-sales | PFoil | 58.20% | 67.1 | 243.2 | 0.195 |
| dresses-sales | PFossil | 59.40% | 2.2 | 20.4 | 0.033 |
| dresses-sales | PyLORD | 60.60% | 190.3 | 607.5 | 1.146 |
| dresses-sales | Pypper | 61.20% | 1.1 | 1.2 | 0.049 |
| dresses-sales | Weka:JRip | 61.00% | 1.9 | 3.1 | 0.952 |
| heart-c | CN2 | 71.61% | 9.1 | 22.9 | 0.042 |
| heart-c | JavaLord | 81.24% | 62.3 | 219.1 | 0.374 |
| heart-c | Opus(WRAcc) | 77.23% | 8.3 | 23.2 | 3.092 |
| heart-c | PFoil | 79.59% | 30.2 | 102.3 | 0.069 |
| heart-c | PFossil | 77.26% | 17.4 | 65.8 | 0.047 |
| heart-c | PyLORD | 80.90% | 63.4 | 225.8 | 0.511 |
| heart-c | Pypper | 81.55% | 3.0 | 7.6 | 0.052 |
| heart-c | Weka:JRip | 79.24% | 3.6 | 8.6 | 0.523 |
| heart-h | CN2 | 77.18% | 14.2 | 33.8 | 0.057 |
| heart-h | JavaLord | 74.14% | 60.8 | 218.8 | 0.350 |
| heart-h | Opus(WRAcc) | 78.53% | 8.1 | 23.1 | 1.982 |
| heart-h | PFoil | 79.25% | 34.6 | 108.7 | 0.069 |
| heart-h | PFossil | 80.26% | 17.0 | 53.2 | 0.035 |
| heart-h | PyLORD | 78.23% | 62.6 | 241.3 | 0.538 |
| heart-h | Pypper | 78.60% | 1.9 | 3.6 | 0.036 |
| heart-h | Weka:JRip | 79.56% | 2.0 | 3.3 | 0.506 |
| heart-statlog | CN2 | 73.33% | 9.4 | 24.2 | 0.070 |
| heart-statlog | JavaLord | 79.26% | 54.1 | 185.4 | 0.466 |
| heart-statlog | Opus(WRAcc) | 79.26% | 8.2 | 23.5 | 3.303 |
| heart-statlog | PFoil | 80.00% | 26.5 | 86.7 | 0.084 |
| heart-statlog | PFossil | 80.74% | 17.8 | 60.9 | 0.060 |
| heart-statlog | PyLORD | 79.26% | 54.0 | 188.2 | 0.578 |
| heart-statlog | Pypper | 78.52% | 2.4 | 6.1 | 0.075 |
| heart-statlog | Weka:JRip | 78.89% | 4.3 | 10.4 | 0.713 |
| hepatitis | CN2 | 83.25% | 8.5 | 16.8 | 0.028 |
| hepatitis | JavaLord | 81.38% | 28.2 | 66.6 | 0.212 |
| hepatitis | Opus(WRAcc) | 80.75% | 7.4 | 20.3 | 0.733 |
| hepatitis | PFoil | 77.58% | 17.1 | 37.0 | 0.027 |
| hepatitis | PFossil | 78.75% | 15.0 | 38.6 | 0.027 |
| hepatitis | PyLORD | 83.38% | 29.8 | 73.0 | 0.180 |
| hepatitis | Pypper | 79.42% | 1.7 | 3.4 | 0.030 |
| hepatitis | Weka:JRip | 78.75% | 2.1 | 4.0 | 0.474 |
| hypothyroid | CN2 | 98.83% | 29.3 | 93.3 | 0.412 |
| hypothyroid | JavaLord | 98.33% | 38.0 | 137.4 | 5.167 |
| hypothyroid | Opus(WRAcc) | 95.63% | 10.1 | 24.6 | 4.213 |
| hypothyroid | PFoil | 99.05% | 31.5 | 100.2 | 0.153 |
| hypothyroid | PFossil | 98.89% | 21.5 | 55.8 | 0.102 |
| hypothyroid | PyLORD | 98.81% | 47.7 | 162.8 | 8.002 |
| hypothyroid | Pypper | 99.28% | 3.3 | 8.9 | 0.252 |
| hypothyroid | Weka:JRip | 99.20% | 3.4 | 8.9 | 1.948 |
| kr-vs-kp | CN2 | 90.80% | 30.8 | 93.8 | 0.252 |
| kr-vs-kp | JavaLord | 99.41% | 47.5 | 224.6 | 5.265 |
| kr-vs-kp | Opus(WRAcc) | 93.96% | 8.0 | 20.0 | 2.905 |
| kr-vs-kp | PFoil | 99.34% | 40.8 | 178.8 | 0.206 |
| kr-vs-kp | PFossil | 99.37% | 14.5 | 56.7 | 0.081 |
| kr-vs-kp | PyLORD | 99.44% | 49.7 | 259.4 | 7.780 |
| kr-vs-kp | Pypper | 99.09% | 13.6 | 44.0 | 0.488 |
| kr-vs-kp | Weka:JRip | 99.31% | 14.6 | 46.2 | 1.445 |
| lymph | CN2 | 78.52% | 11.9 | 33.2 | 0.052 |
| lymph | JavaLord | 83.10% | 28.9 | 79.4 | 0.198 |
| lymph | Opus(WRAcc) | 76.38% | 8.2 | 20.3 | 1.415 |
| lymph | PFoil | 81.86% | 16.0 | 39.6 | 0.027 |
| lymph | PFossil | 80.38% | 16.3 | 45.2 | 0.030 |
| lymph | PyLORD | 82.43% | 28.7 | 79.6 | 0.193 |
| lymph | Pypper | 73.62% | 2.7 | 5.8 | 0.039 |
| lymph | Weka:JRip | 82.48% | 4.1 | 8.2 | 0.458 |
| molecular-biology_promoters | CN2 | 80.27% | 9.3 | 17.5 | 0.045 |
| molecular-biology_promoters | JavaLord | 86.82% | 16.9 | 33.0 | 0.322 |
| molecular-biology_promoters | Opus(WRAcc) | 77.64% | 4.7 | 12.2 | 5.822 |
| molecular-biology_promoters | PFoil | 86.09% | 9.0 | 18.6 | 0.018 |
| molecular-biology_promoters | PFossil | 82.27% | 7.4 | 19.7 | 0.016 |
| molecular-biology_promoters | PyLORD | 86.00% | 16.4 | 31.9 | 0.155 |
| molecular-biology_promoters | Pypper | 83.18% | 2.1 | 3.8 | 0.053 |
| molecular-biology_promoters | Weka:JRip | 77.45% | 2.5 | 5.4 | 0.712 |
| mushroom | CN2 | 100.00% | 23.0 | 30.0 | 0.219 |
| mushroom | JavaLord | 100.00% | 23.0 | 38.7 | 43.520 |
| mushroom | Opus(WRAcc) | 98.58% | 4.3 | 12.6 | 3.050 |
| mushroom | PFoil | 100.00% | 11.5 | 24.7 | 0.136 |
| mushroom | PFossil | 99.95% | 12.3 | 23.0 | 0.152 |
| mushroom | PyLORD | 100.00% | 23.0 | 38.7 | 28.460 |
| mushroom | Pypper | 99.96% | 5.2 | 10.6 | 0.277 |
| mushroom | Weka:JRip | 99.98% | 6.0 | 10.1 | 4.979 |
| sick | CN2 | 97.72% | 53.4 | 165.1 | 0.789 |
| sick | JavaLord | 97.85% | 96.2 | 402.4 | 7.522 |
| sick | Opus(WRAcc) | 96.66% | 10.5 | 27.4 | 8.395 |
| sick | PFoil | 97.99% | 63.5 | 241.2 | 0.299 |
| sick | PFossil | 98.30% | 16.4 | 59.4 | 0.091 |
| sick | PyLORD | 97.83% | 103.9 | 531.4 | 10.145 |
| sick | Pypper | 98.22% | 3.5 | 11.9 | 0.117 |
| sick | Weka:JRip | 98.22% | 5.3 | 18.1 | 1.952 |
| solar-flare | CN2 | 69.13% | 39.7 | 168.1 | 0.902 |
| solar-flare | JavaLord | 73.26% | 209.1 | 1171.2 | 0.885 |
| solar-flare | Opus(WRAcc) | 74.77% | 11.2 | 27.4 | 1.535 |
| solar-flare | PFoil | 73.73% | 85.2 | 403.4 | 0.282 |
| solar-flare | PFossil | 74.02% | 14.5 | 62.5 | 0.066 |
| solar-flare | PyLORD | 73.45% | 209.2 | 1377.7 | 4.381 |
| solar-flare | Pypper | 74.77% | 4.9 | 8.3 | 0.148 |
| solar-flare | Weka:JRip | 71.21% | 6.5 | 16.9 | 0.798 |
| splice | CN2 | 65.27% | 21.2 | 80.3 | 0.395 |
| splice | JavaLord | 94.70% | 255.3 | 1077.6 | 87.788 |
| splice | Opus(WRAcc) | n/a | n/a | n/a | 300.172 |
| splice | PFoil | 92.76% | 78.2 | 379.4 | 0.634 |
| splice | PFossil | 94.23% | 23.7 | 106.8 | 0.212 |
| splice | PyLORD | 94.61% | 254.4 | 1073.4 | 22.027 |
| splice | Pypper | 93.82% | 11.4 | 49.6 | 1.021 |
| splice | Weka:JRip | 94.80% | 9.7 | 39.4 | 7.160 |
| tic-tac-toe | CN2 | 98.33% | 11.6 | 35.0 | 0.075 |
| tic-tac-toe | JavaLord | 98.44% | 32.6 | 119.6 | 1.190 |
| tic-tac-toe | Opus(WRAcc) | 77.04% | 10.9 | 28.9 | 2.200 |
| tic-tac-toe | PFoil | 99.06% | 23.2 | 83.9 | 0.062 |
| tic-tac-toe | PFossil | 97.71% | 14.7 | 52.5 | 0.041 |
| tic-tac-toe | PyLORD | 95.93% | 53.9 | 222.2 | 2.061 |
| tic-tac-toe | Pypper | 96.25% | 7.5 | 23.1 | 0.129 |
| tic-tac-toe | Weka:JRip | 98.12% | 8.4 | 26.6 | 0.668 |
| vote | CN2 | 94.96% | 14.0 | 41.0 | 0.065 |
| vote | JavaLord | 92.87% | 33.7 | 102.0 | 0.227 |
| vote | Opus(WRAcc) | 94.74% | 8.3 | 16.2 | 0.083 |
| vote | PFoil | 94.71% | 23.3 | 67.8 | 0.043 |
| vote | PFossil | 95.18% | 10.2 | 29.2 | 0.021 |
| vote | PyLORD | 93.79% | 38.7 | 121.4 | 0.576 |
| vote | Pypper | 95.18% | 1.7 | 3.3 | 0.032 |
| vote | Weka:JRip | 95.42% | 3.1 | 7.6 | 0.385 |
| zoo | CN2 | 86.18% | 5.8 | 10.2 | 0.022 |
| zoo | JavaLord | 94.00% | 9.1 | 20.3 | 0.138 |
| zoo | Opus(WRAcc) | 95.00% | 7.0 | 14.0 | 0.058 |
| zoo | PFoil | 96.00% | 9.5 | 20.3 | 0.016 |
| zoo | PFossil | 95.00% | 8.9 | 21.0 | 0.016 |
| zoo | PyLORD | 95.00% | 9.1 | 20.3 | 0.086 |
| zoo | Pypper | 88.09% | 5.4 | 8.0 | 0.053 |
| zoo | Weka:JRip | 89.00% | 5.8 | 9.2 | 0.423 |

