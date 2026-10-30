# Base against composite models: bagged rule sets, combined at two levels or one

## Setup

**Full run.** Quick sanity check instead: `python demos/base_vs_composite.py`.

Bagging with rule sets as members: 10 concept sets -- one concept
description per class, learned one-vs-rest by Pypper, PFossil, PFoil -- each from a
bootstrap sample of the training fold. The same members are combined in two
ways, each with one of four conflict resolutions (**best rule**, the
covering rule with the best Laplace estimate; **voting**, one vote per
covering rule; **micro** / **macro distribution voting**, the covering
rules' class counts / class proportions summed):

- **two-level** -- every member resolves its own covering rules, and the
  members' predictions are combined by a plain vote;
- **one-level** -- the union of all members' rules as one rule set, its
  conflicts resolved directly.

A member whose rules don't cover an example abstains at both levels (the
members' defaults are ignored); examples no member covers go to the
majority class. **single** is the baseline: one concept set learned from the
whole training fold, with the same conflict resolution.

Datasets, with their numbers of classes: `balance-scale` (3), `cmc` (3), `dna` (3), `hayes-roth` (3), `iris` (3), `splice` (3), `wine` (3), `car` (4), `lymph` (4), `vehicle` (4), `anneal` (5), `glass` (6), `solar-flare` (6), `segment` (7), `zoo` (7), `ecoli` (8), `led24` (10), `yeast` (10), `optdigits` (10), `texture` (11), `vowel` (11), `soybean` (19), `primary-tumor` (21), `audiology` (24), `letter` (26). Two-class datasets: `SPECT`, `Titanic`, `banknote-authentication`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `churn`, `climate-model-simulation-crashes`, `colic`, `compas-two-years`, `credit-approval`, `credit-g`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `heloc`, `hepatitis`, `ilpd`, `ionosphere`, `kc1`, `kc2`, `kr-vs-kp`, `mofn-3-7-10`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `sonar`, `spambase`, `tic-tac-toe`, `vote`, `wdbc`, `wilt`. Protocol:
10-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals=8)`). Each fit -- the members and the
single concept set together -- is capped at 1800s (3600s for
`letter`).

## Pypper

Rules per ensemble: 138.0, conditions: 532.5; rules of the single concept set: 16.0.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 82.76% (8.07) | 84.79% (4.88) | 84.01% (6.64) |
| voting | 82.33% (8.99) | 84.34% (5.82) | 84.75% (4.77) |
| micro distribution voting | 82.39% (9.05) | 84.20% (6.33) | 83.66% (7.54) |
| macro distribution voting | 82.76% (7.84) | 84.86% (4.30) | 85.09% (3.76) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## PFossil

Rules per ensemble: 201.4, conditions: 898.4; rules of the single concept set: 19.2.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 82.55% (7.56) | 84.32% (4.49) | 82.63% (7.64) |
| voting | 81.65% (9.20) | 83.42% (6.38) | 84.10% (4.81) |
| micro distribution voting | 81.48% (9.38) | 83.13% (6.22) | 82.99% (6.94) |
| macro distribution voting | 82.39% (7.83) | 84.34% (4.31) | 84.80% (3.25) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## PFoil

Rules per ensemble: 858.1, conditions: 4732.2; rules of the single concept set: 99.0.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 84.38% (7.44) | 86.01% (4.36) | 83.70% (7.85) |
| voting | 82.17% (10.40) | 85.15% (6.00) | 85.84% (4.55) |
| micro distribution voting | 82.62% (9.41) | 85.39% (6.05) | 84.34% (7.32) |
| macro distribution voting | 84.31% (7.38) | 85.94% (4.01) | 86.26% (3.22) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## Two-level against one-level

Per (dataset, base learner) pair, the accuracy of the two-level combination minus that of the one-level one, in percentage points, and the number of pairs on which either is ahead.

| conflict resolution | mean difference | two-level ahead | one-level ahead | ties |
|---|--:|--:|--:|--:|
| best rule | +1.60 | 150 | 50 | 7 |
| voting | -0.59 | 60 | 132 | 15 |
| micro distribution voting | +0.58 | 130 | 63 | 14 |
| macro distribution voting | -0.34 | 67 | 118 | 22 |

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_rules |
|---|---|---|---|
| SPECT | PFoil:one-level (best rule) | 84.66% | 229.5 |
| SPECT | PFoil:one-level (macro distribution voting) | 84.25% | 229.5 |
| SPECT | PFoil:one-level (micro distribution voting) | 79.42% | 229.5 |
| SPECT | PFoil:one-level (voting) | 83.46% | 229.5 |
| SPECT | PFoil:single (best rule) | 82.05% | 229.5 |
| SPECT | PFoil:single (macro distribution voting) | 80.94% | 229.5 |
| SPECT | PFoil:single (micro distribution voting) | 76.81% | 229.5 |
| SPECT | PFoil:single (voting) | 79.77% | 229.5 |
| SPECT | PFoil:two-level (best rule) | 85.00% | 229.5 |
| SPECT | PFoil:two-level (macro distribution voting) | 84.25% | 229.5 |
| SPECT | PFoil:two-level (micro distribution voting) | 79.40% | 229.5 |
| SPECT | PFoil:two-level (voting) | 84.22% | 229.5 |
| SPECT | PFossil:one-level (best rule) | 84.66% | 111.6 |
| SPECT | PFossil:one-level (macro distribution voting) | 80.88% | 111.6 |
| SPECT | PFossil:one-level (micro distribution voting) | 82.74% | 111.6 |
| SPECT | PFossil:one-level (voting) | 79.03% | 111.6 |
| SPECT | PFossil:single (best rule) | 78.68% | 111.6 |
| SPECT | PFossil:single (macro distribution voting) | 77.55% | 111.6 |
| SPECT | PFossil:single (micro distribution voting) | 78.68% | 111.6 |
| SPECT | PFossil:single (voting) | 77.55% | 111.6 |
| SPECT | PFossil:two-level (best rule) | 79.79% | 111.6 |
| SPECT | PFossil:two-level (macro distribution voting) | 79.03% | 111.6 |
| SPECT | PFossil:two-level (micro distribution voting) | 79.42% | 111.6 |
| SPECT | PFossil:two-level (voting) | 79.42% | 111.6 |
| SPECT | Pypper:one-level (best rule) | 83.89% | 50.8 |
| SPECT | Pypper:one-level (macro distribution voting) | 83.13% | 50.8 |
| SPECT | Pypper:one-level (micro distribution voting) | 82.76% | 50.8 |
| SPECT | Pypper:one-level (voting) | 79.76% | 50.8 |
| SPECT | Pypper:single (best rule) | 79.02% | 50.8 |
| SPECT | Pypper:single (macro distribution voting) | 79.02% | 50.8 |
| SPECT | Pypper:single (micro distribution voting) | 79.02% | 50.8 |
| SPECT | Pypper:single (voting) | 79.02% | 50.8 |
| SPECT | Pypper:two-level (best rule) | 80.53% | 50.8 |
| SPECT | Pypper:two-level (macro distribution voting) | 80.53% | 50.8 |
| SPECT | Pypper:two-level (micro distribution voting) | 80.53% | 50.8 |
| SPECT | Pypper:two-level (voting) | 80.53% | 50.8 |
| Titanic | PFoil:one-level (best rule) | 78.23% | 1000.1 |
| Titanic | PFoil:one-level (macro distribution voting) | 79.14% | 1000.1 |
| Titanic | PFoil:one-level (micro distribution voting) | 75.94% | 1000.1 |
| Titanic | PFoil:one-level (voting) | 78.53% | 1000.1 |
| Titanic | PFoil:single (best rule) | 78.76% | 1000.1 |
| Titanic | PFoil:single (macro distribution voting) | 77.92% | 1000.1 |
| Titanic | PFoil:single (micro distribution voting) | 75.55% | 1000.1 |
| Titanic | PFoil:single (voting) | 76.78% | 1000.1 |
| Titanic | PFoil:two-level (best rule) | 78.91% | 1000.1 |
| Titanic | PFoil:two-level (macro distribution voting) | 78.68% | 1000.1 |
| Titanic | PFoil:two-level (micro distribution voting) | 77.24% | 1000.1 |
| Titanic | PFoil:two-level (voting) | 79.22% | 1000.1 |
| Titanic | PFossil:one-level (best rule) | 78.53% | 64.9 |
| Titanic | PFossil:one-level (macro distribution voting) | 78.84% | 64.9 |
| Titanic | PFossil:one-level (micro distribution voting) | 78.53% | 64.9 |
| Titanic | PFossil:one-level (voting) | 78.68% | 64.9 |
| Titanic | PFossil:single (best rule) | 77.77% | 64.9 |
| Titanic | PFossil:single (macro distribution voting) | 77.85% | 64.9 |
| Titanic | PFossil:single (micro distribution voting) | 76.78% | 64.9 |
| Titanic | PFossil:single (voting) | 78.15% | 64.9 |
| Titanic | PFossil:two-level (best rule) | 78.91% | 64.9 |
| Titanic | PFossil:two-level (macro distribution voting) | 78.61% | 64.9 |
| Titanic | PFossil:two-level (micro distribution voting) | 77.84% | 64.9 |
| Titanic | PFossil:two-level (voting) | 77.08% | 64.9 |
| Titanic | Pypper:one-level (best rule) | 78.91% | 69.1 |
| Titanic | Pypper:one-level (macro distribution voting) | 78.53% | 69.1 |
| Titanic | Pypper:one-level (micro distribution voting) | 78.38% | 69.1 |
| Titanic | Pypper:one-level (voting) | 78.30% | 69.1 |
| Titanic | Pypper:single (best rule) | 79.22% | 69.1 |
| Titanic | Pypper:single (macro distribution voting) | 79.30% | 69.1 |
| Titanic | Pypper:single (micro distribution voting) | 78.84% | 69.1 |
| Titanic | Pypper:single (voting) | 78.84% | 69.1 |
| Titanic | Pypper:two-level (best rule) | 78.53% | 69.1 |
| Titanic | Pypper:two-level (macro distribution voting) | 78.84% | 69.1 |
| Titanic | Pypper:two-level (micro distribution voting) | 78.84% | 69.1 |
| Titanic | Pypper:two-level (voting) | 78.76% | 69.1 |
| anneal | PFoil:one-level (best rule) | 96.99% | 240.8 |
| anneal | PFoil:one-level (macro distribution voting) | 98.33% | 240.8 |
| anneal | PFoil:one-level (micro distribution voting) | 93.21% | 240.8 |
| anneal | PFoil:one-level (voting) | 98.44% | 240.8 |
| anneal | PFoil:single (best rule) | 97.21% | 240.8 |
| anneal | PFoil:single (macro distribution voting) | 97.66% | 240.8 |
| anneal | PFoil:single (micro distribution voting) | 93.43% | 240.8 |
| anneal | PFoil:single (voting) | 94.88% | 240.8 |
| anneal | PFoil:two-level (best rule) | 98.88% | 240.8 |
| anneal | PFoil:two-level (macro distribution voting) | 99.00% | 240.8 |
| anneal | PFoil:two-level (micro distribution voting) | 98.11% | 240.8 |
| anneal | PFoil:two-level (voting) | 98.55% | 240.8 |
| anneal | PFossil:one-level (best rule) | 98.22% | 181.0 |
| anneal | PFossil:one-level (macro distribution voting) | 99.44% | 181.0 |
| anneal | PFossil:one-level (micro distribution voting) | 98.33% | 181.0 |
| anneal | PFossil:one-level (voting) | 98.33% | 181.0 |
| anneal | PFossil:single (best rule) | 99.22% | 181.0 |
| anneal | PFossil:single (macro distribution voting) | 99.66% | 181.0 |
| anneal | PFossil:single (micro distribution voting) | 98.88% | 181.0 |
| anneal | PFossil:single (voting) | 99.66% | 181.0 |
| anneal | PFossil:two-level (best rule) | 99.33% | 181.0 |
| anneal | PFossil:two-level (macro distribution voting) | 99.55% | 181.0 |
| anneal | PFossil:two-level (micro distribution voting) | 99.22% | 181.0 |
| anneal | PFossil:two-level (voting) | 99.55% | 181.0 |
| anneal | Pypper:one-level (best rule) | 97.21% | 118.6 |
| anneal | Pypper:one-level (macro distribution voting) | 98.33% | 118.6 |
| anneal | Pypper:one-level (micro distribution voting) | 97.10% | 118.6 |
| anneal | Pypper:one-level (voting) | 98.22% | 118.6 |
| anneal | Pypper:single (best rule) | 96.10% | 118.6 |
| anneal | Pypper:single (macro distribution voting) | 96.33% | 118.6 |
| anneal | Pypper:single (micro distribution voting) | 95.77% | 118.6 |
| anneal | Pypper:single (voting) | 96.10% | 118.6 |
| anneal | Pypper:two-level (best rule) | 97.99% | 118.6 |
| anneal | Pypper:two-level (macro distribution voting) | 98.22% | 118.6 |
| anneal | Pypper:two-level (micro distribution voting) | 97.99% | 118.6 |
| anneal | Pypper:two-level (voting) | 98.22% | 118.6 |
| audiology | PFoil:one-level (best rule) | 70.40% | 374.1 |
| audiology | PFoil:one-level (macro distribution voting) | 82.37% | 374.1 |
| audiology | PFoil:one-level (micro distribution voting) | 71.32% | 374.1 |
| audiology | PFoil:one-level (voting) | 77.94% | 374.1 |
| audiology | PFoil:single (best rule) | 77.04% | 374.1 |
| audiology | PFoil:single (macro distribution voting) | 78.81% | 374.1 |
| audiology | PFoil:single (micro distribution voting) | 61.09% | 374.1 |
| audiology | PFoil:single (voting) | 62.41% | 374.1 |
| audiology | PFoil:two-level (best rule) | 79.70% | 374.1 |
| audiology | PFoil:two-level (macro distribution voting) | 79.72% | 374.1 |
| audiology | PFoil:two-level (micro distribution voting) | 77.51% | 374.1 |
| audiology | PFoil:two-level (voting) | 73.54% | 374.1 |
| audiology | PFossil:one-level (best rule) | 71.19% | 355.4 |
| audiology | PFossil:one-level (macro distribution voting) | 80.57% | 355.4 |
| audiology | PFossil:one-level (micro distribution voting) | 73.87% | 355.4 |
| audiology | PFossil:one-level (voting) | 81.48% | 355.4 |
| audiology | PFossil:single (best rule) | 77.89% | 355.4 |
| audiology | PFossil:single (macro distribution voting) | 77.00% | 355.4 |
| audiology | PFossil:single (micro distribution voting) | 74.78% | 355.4 |
| audiology | PFossil:single (voting) | 75.26% | 355.4 |
| audiology | PFossil:two-level (best rule) | 80.55% | 355.4 |
| audiology | PFossil:two-level (macro distribution voting) | 79.68% | 355.4 |
| audiology | PFossil:two-level (micro distribution voting) | 78.34% | 355.4 |
| audiology | PFossil:two-level (voting) | 77.47% | 355.4 |
| audiology | Pypper:one-level (best rule) | 73.56% | 120.0 |
| audiology | Pypper:one-level (macro distribution voting) | 75.28% | 120.0 |
| audiology | Pypper:one-level (micro distribution voting) | 70.83% | 120.0 |
| audiology | Pypper:one-level (voting) | 74.82% | 120.0 |
| audiology | Pypper:single (best rule) | 64.64% | 120.0 |
| audiology | Pypper:single (macro distribution voting) | 65.10% | 120.0 |
| audiology | Pypper:single (micro distribution voting) | 63.75% | 120.0 |
| audiology | Pypper:single (voting) | 62.02% | 120.0 |
| audiology | Pypper:two-level (best rule) | 74.80% | 120.0 |
| audiology | Pypper:two-level (macro distribution voting) | 75.69% | 120.0 |
| audiology | Pypper:two-level (micro distribution voting) | 74.82% | 120.0 |
| audiology | Pypper:two-level (voting) | 75.75% | 120.0 |
| balance-scale | PFoil:one-level (best rule) | 85.60% | 600.8 |
| balance-scale | PFoil:one-level (macro distribution voting) | 88.16% | 600.8 |
| balance-scale | PFoil:one-level (micro distribution voting) | 88.48% | 600.8 |
| balance-scale | PFoil:one-level (voting) | 84.47% | 600.8 |
| balance-scale | PFoil:single (best rule) | 82.87% | 600.8 |
| balance-scale | PFoil:single (macro distribution voting) | 82.54% | 600.8 |
| balance-scale | PFoil:single (micro distribution voting) | 82.39% | 600.8 |
| balance-scale | PFoil:single (voting) | 81.11% | 600.8 |
| balance-scale | PFoil:two-level (best rule) | 86.72% | 600.8 |
| balance-scale | PFoil:two-level (macro distribution voting) | 85.91% | 600.8 |
| balance-scale | PFoil:two-level (micro distribution voting) | 86.88% | 600.8 |
| balance-scale | PFoil:two-level (voting) | 84.80% | 600.8 |
| balance-scale | PFossil:one-level (best rule) | 83.34% | 240.3 |
| balance-scale | PFossil:one-level (macro distribution voting) | 87.03% | 240.3 |
| balance-scale | PFossil:one-level (micro distribution voting) | 86.23% | 240.3 |
| balance-scale | PFossil:one-level (voting) | 87.68% | 240.3 |
| balance-scale | PFossil:single (best rule) | 81.60% | 240.3 |
| balance-scale | PFossil:single (macro distribution voting) | 82.08% | 240.3 |
| balance-scale | PFossil:single (micro distribution voting) | 80.47% | 240.3 |
| balance-scale | PFossil:single (voting) | 79.67% | 240.3 |
| balance-scale | PFossil:two-level (best rule) | 86.23% | 240.3 |
| balance-scale | PFossil:two-level (macro distribution voting) | 86.07% | 240.3 |
| balance-scale | PFossil:two-level (micro distribution voting) | 86.23% | 240.3 |
| balance-scale | PFossil:two-level (voting) | 86.07% | 240.3 |
| balance-scale | Pypper:one-level (best rule) | 84.95% | 115.0 |
| balance-scale | Pypper:one-level (macro distribution voting) | 86.40% | 115.0 |
| balance-scale | Pypper:one-level (micro distribution voting) | 85.92% | 115.0 |
| balance-scale | Pypper:one-level (voting) | 86.09% | 115.0 |
| balance-scale | Pypper:single (best rule) | 80.95% | 115.0 |
| balance-scale | Pypper:single (macro distribution voting) | 80.95% | 115.0 |
| balance-scale | Pypper:single (micro distribution voting) | 80.62% | 115.0 |
| balance-scale | Pypper:single (voting) | 80.47% | 115.0 |
| balance-scale | Pypper:two-level (best rule) | 86.08% | 115.0 |
| balance-scale | Pypper:two-level (macro distribution voting) | 85.92% | 115.0 |
| balance-scale | Pypper:two-level (micro distribution voting) | 85.92% | 115.0 |
| balance-scale | Pypper:two-level (voting) | 86.08% | 115.0 |
| banknote-authentication | PFoil:one-level (best rule) | 98.25% | 207.6 |
| banknote-authentication | PFoil:one-level (macro distribution voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:one-level (micro distribution voting) | 98.62% | 207.6 |
| banknote-authentication | PFoil:one-level (voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:single (best rule) | 98.40% | 207.6 |
| banknote-authentication | PFoil:single (macro distribution voting) | 98.76% | 207.6 |
| banknote-authentication | PFoil:single (micro distribution voting) | 98.18% | 207.6 |
| banknote-authentication | PFoil:single (voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:two-level (best rule) | 98.91% | 207.6 |
| banknote-authentication | PFoil:two-level (macro distribution voting) | 99.05% | 207.6 |
| banknote-authentication | PFoil:two-level (micro distribution voting) | 98.91% | 207.6 |
| banknote-authentication | PFoil:two-level (voting) | 99.05% | 207.6 |
| banknote-authentication | PFossil:one-level (best rule) | 98.62% | 116.1 |
| banknote-authentication | PFossil:one-level (macro distribution voting) | 98.69% | 116.1 |
| banknote-authentication | PFossil:one-level (micro distribution voting) | 94.46% | 116.1 |
| banknote-authentication | PFossil:one-level (voting) | 98.18% | 116.1 |
| banknote-authentication | PFossil:single (best rule) | 99.20% | 116.1 |
| banknote-authentication | PFossil:single (macro distribution voting) | 99.05% | 116.1 |
| banknote-authentication | PFossil:single (micro distribution voting) | 94.32% | 116.1 |
| banknote-authentication | PFossil:single (voting) | 97.89% | 116.1 |
| banknote-authentication | PFossil:two-level (best rule) | 99.13% | 116.1 |
| banknote-authentication | PFossil:two-level (macro distribution voting) | 99.05% | 116.1 |
| banknote-authentication | PFossil:two-level (micro distribution voting) | 94.32% | 116.1 |
| banknote-authentication | PFossil:two-level (voting) | 97.74% | 116.1 |
| banknote-authentication | Pypper:one-level (best rule) | 98.25% | 90.4 |
| banknote-authentication | Pypper:one-level (macro distribution voting) | 98.69% | 90.4 |
| banknote-authentication | Pypper:one-level (micro distribution voting) | 98.32% | 90.4 |
| banknote-authentication | Pypper:one-level (voting) | 98.76% | 90.4 |
| banknote-authentication | Pypper:single (best rule) | 98.54% | 90.4 |
| banknote-authentication | Pypper:single (macro distribution voting) | 98.61% | 90.4 |
| banknote-authentication | Pypper:single (micro distribution voting) | 98.54% | 90.4 |
| banknote-authentication | Pypper:single (voting) | 98.61% | 90.4 |
| banknote-authentication | Pypper:two-level (best rule) | 98.69% | 90.4 |
| banknote-authentication | Pypper:two-level (macro distribution voting) | 98.91% | 90.4 |
| banknote-authentication | Pypper:two-level (micro distribution voting) | 98.61% | 90.4 |
| banknote-authentication | Pypper:two-level (voting) | 98.83% | 90.4 |
| blood-transfusion-service-center | PFoil:one-level (best rule) | 76.47% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level (macro distribution voting) | 76.61% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level (micro distribution voting) | 76.07% | 536.1 |
| blood-transfusion-service-center | PFoil:one-level (voting) | 73.40% | 536.1 |
| blood-transfusion-service-center | PFoil:single (best rule) | 76.07% | 536.1 |
| blood-transfusion-service-center | PFoil:single (macro distribution voting) | 75.94% | 536.1 |
| blood-transfusion-service-center | PFoil:single (micro distribution voting) | 76.07% | 536.1 |
| blood-transfusion-service-center | PFoil:single (voting) | 68.05% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level (best rule) | 77.14% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level (macro distribution voting) | 76.47% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level (micro distribution voting) | 76.34% | 536.1 |
| blood-transfusion-service-center | PFoil:two-level (voting) | 77.00% | 536.1 |
| blood-transfusion-service-center | PFossil:one-level (best rule) | 73.12% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level (macro distribution voting) | 73.52% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level (micro distribution voting) | 73.26% | 18.9 |
| blood-transfusion-service-center | PFossil:one-level (voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:single (best rule) | 76.86% | 18.9 |
| blood-transfusion-service-center | PFossil:single (macro distribution voting) | 76.86% | 18.9 |
| blood-transfusion-service-center | PFossil:single (micro distribution voting) | 76.86% | 18.9 |
| blood-transfusion-service-center | PFossil:single (voting) | 76.86% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level (best rule) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level (macro distribution voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level (micro distribution voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | PFossil:two-level (voting) | 68.46% | 18.9 |
| blood-transfusion-service-center | Pypper:one-level (best rule) | 78.47% | 41.9 |
| blood-transfusion-service-center | Pypper:one-level (macro distribution voting) | 78.60% | 41.9 |
| blood-transfusion-service-center | Pypper:one-level (micro distribution voting) | 78.34% | 41.9 |
| blood-transfusion-service-center | Pypper:one-level (voting) | 78.73% | 41.9 |
| blood-transfusion-service-center | Pypper:single (best rule) | 78.60% | 41.9 |
| blood-transfusion-service-center | Pypper:single (macro distribution voting) | 78.60% | 41.9 |
| blood-transfusion-service-center | Pypper:single (micro distribution voting) | 78.47% | 41.9 |
| blood-transfusion-service-center | Pypper:single (voting) | 78.47% | 41.9 |
| blood-transfusion-service-center | Pypper:two-level (best rule) | 78.86% | 41.9 |
| blood-transfusion-service-center | Pypper:two-level (macro distribution voting) | 78.86% | 41.9 |
| blood-transfusion-service-center | Pypper:two-level (micro distribution voting) | 78.73% | 41.9 |
| blood-transfusion-service-center | Pypper:two-level (voting) | 78.73% | 41.9 |
| breast-cancer | PFoil:one-level (best rule) | 75.20% | 357.7 |
| breast-cancer | PFoil:one-level (macro distribution voting) | 74.11% | 357.7 |
| breast-cancer | PFoil:one-level (micro distribution voting) | 73.45% | 357.7 |
| breast-cancer | PFoil:one-level (voting) | 74.82% | 357.7 |
| breast-cancer | PFoil:single (best rule) | 72.02% | 357.7 |
| breast-cancer | PFoil:single (macro distribution voting) | 72.73% | 357.7 |
| breast-cancer | PFoil:single (micro distribution voting) | 72.71% | 357.7 |
| breast-cancer | PFoil:single (voting) | 70.94% | 357.7 |
| breast-cancer | PFoil:two-level (best rule) | 74.47% | 357.7 |
| breast-cancer | PFoil:two-level (macro distribution voting) | 74.13% | 357.7 |
| breast-cancer | PFoil:two-level (micro distribution voting) | 72.38% | 357.7 |
| breast-cancer | PFoil:two-level (voting) | 72.72% | 357.7 |
| breast-cancer | PFossil:one-level (best rule) | 74.50% | 114.8 |
| breast-cancer | PFossil:one-level (macro distribution voting) | 75.16% | 114.8 |
| breast-cancer | PFossil:one-level (micro distribution voting) | 74.50% | 114.8 |
| breast-cancer | PFossil:one-level (voting) | 73.41% | 114.8 |
| breast-cancer | PFossil:single (best rule) | 69.26% | 114.8 |
| breast-cancer | PFossil:single (macro distribution voting) | 69.98% | 114.8 |
| breast-cancer | PFossil:single (micro distribution voting) | 68.56% | 114.8 |
| breast-cancer | PFossil:single (voting) | 68.92% | 114.8 |
| breast-cancer | PFossil:two-level (best rule) | 75.16% | 114.8 |
| breast-cancer | PFossil:two-level (macro distribution voting) | 74.47% | 114.8 |
| breast-cancer | PFossil:two-level (micro distribution voting) | 74.82% | 114.8 |
| breast-cancer | PFossil:two-level (voting) | 74.14% | 114.8 |
| breast-cancer | Pypper:one-level (best rule) | 73.05% | 36.9 |
| breast-cancer | Pypper:one-level (macro distribution voting) | 74.47% | 36.9 |
| breast-cancer | Pypper:one-level (micro distribution voting) | 73.44% | 36.9 |
| breast-cancer | Pypper:one-level (voting) | 73.41% | 36.9 |
| breast-cancer | Pypper:single (best rule) | 74.84% | 36.9 |
| breast-cancer | Pypper:single (macro distribution voting) | 74.84% | 36.9 |
| breast-cancer | Pypper:single (micro distribution voting) | 75.20% | 36.9 |
| breast-cancer | Pypper:single (voting) | 75.20% | 36.9 |
| breast-cancer | Pypper:two-level (best rule) | 73.76% | 36.9 |
| breast-cancer | Pypper:two-level (macro distribution voting) | 73.76% | 36.9 |
| breast-cancer | Pypper:two-level (micro distribution voting) | 73.42% | 36.9 |
| breast-cancer | Pypper:two-level (voting) | 73.42% | 36.9 |
| breast-w | PFoil:one-level (best rule) | 91.56% | 144.6 |
| breast-w | PFoil:one-level (macro distribution voting) | 96.28% | 144.6 |
| breast-w | PFoil:one-level (micro distribution voting) | 94.71% | 144.6 |
| breast-w | PFoil:one-level (voting) | 96.14% | 144.6 |
| breast-w | PFoil:single (best rule) | 95.85% | 144.6 |
| breast-w | PFoil:single (macro distribution voting) | 95.99% | 144.6 |
| breast-w | PFoil:single (micro distribution voting) | 95.42% | 144.6 |
| breast-w | PFoil:single (voting) | 94.70% | 144.6 |
| breast-w | PFoil:two-level (best rule) | 96.14% | 144.6 |
| breast-w | PFoil:two-level (macro distribution voting) | 96.14% | 144.6 |
| breast-w | PFoil:two-level (micro distribution voting) | 95.85% | 144.6 |
| breast-w | PFoil:two-level (voting) | 95.71% | 144.6 |
| breast-w | PFossil:one-level (best rule) | 93.42% | 120.0 |
| breast-w | PFossil:one-level (macro distribution voting) | 96.14% | 120.0 |
| breast-w | PFossil:one-level (micro distribution voting) | 95.56% | 120.0 |
| breast-w | PFossil:one-level (voting) | 96.14% | 120.0 |
| breast-w | PFossil:single (best rule) | 94.13% | 120.0 |
| breast-w | PFossil:single (macro distribution voting) | 94.99% | 120.0 |
| breast-w | PFossil:single (micro distribution voting) | 94.27% | 120.0 |
| breast-w | PFossil:single (voting) | 94.99% | 120.0 |
| breast-w | PFossil:two-level (best rule) | 95.56% | 120.0 |
| breast-w | PFossil:two-level (macro distribution voting) | 95.42% | 120.0 |
| breast-w | PFossil:two-level (micro distribution voting) | 95.71% | 120.0 |
| breast-w | PFossil:two-level (voting) | 95.56% | 120.0 |
| breast-w | Pypper:one-level (best rule) | 94.13% | 57.0 |
| breast-w | Pypper:one-level (macro distribution voting) | 96.85% | 57.0 |
| breast-w | Pypper:one-level (micro distribution voting) | 96.14% | 57.0 |
| breast-w | Pypper:one-level (voting) | 96.85% | 57.0 |
| breast-w | Pypper:single (best rule) | 94.57% | 57.0 |
| breast-w | Pypper:single (macro distribution voting) | 95.00% | 57.0 |
| breast-w | Pypper:single (micro distribution voting) | 94.57% | 57.0 |
| breast-w | Pypper:single (voting) | 95.00% | 57.0 |
| breast-w | Pypper:two-level (best rule) | 96.57% | 57.0 |
| breast-w | Pypper:two-level (macro distribution voting) | 96.71% | 57.0 |
| breast-w | Pypper:two-level (micro distribution voting) | 96.42% | 57.0 |
| breast-w | Pypper:two-level (voting) | 96.71% | 57.0 |
| car | PFoil:one-level (best rule) | 95.49% | 682.0 |
| car | PFoil:one-level (macro distribution voting) | 96.87% | 682.0 |
| car | PFoil:one-level (micro distribution voting) | 94.85% | 682.0 |
| car | PFoil:one-level (voting) | 96.53% | 682.0 |
| car | PFoil:single (best rule) | 96.70% | 682.0 |
| car | PFoil:single (macro distribution voting) | 96.35% | 682.0 |
| car | PFoil:single (micro distribution voting) | 95.60% | 682.0 |
| car | PFoil:single (voting) | 94.73% | 682.0 |
| car | PFoil:two-level (best rule) | 97.22% | 682.0 |
| car | PFoil:two-level (macro distribution voting) | 97.28% | 682.0 |
| car | PFoil:two-level (micro distribution voting) | 96.41% | 682.0 |
| car | PFoil:two-level (voting) | 96.12% | 682.0 |
| car | PFossil:one-level (best rule) | 91.49% | 182.2 |
| car | PFossil:one-level (macro distribution voting) | 93.46% | 182.2 |
| car | PFossil:one-level (micro distribution voting) | 86.05% | 182.2 |
| car | PFossil:one-level (voting) | 91.72% | 182.2 |
| car | PFossil:single (best rule) | 93.87% | 182.2 |
| car | PFossil:single (macro distribution voting) | 93.87% | 182.2 |
| car | PFossil:single (micro distribution voting) | 84.90% | 182.2 |
| car | PFossil:single (voting) | 90.51% | 182.2 |
| car | PFossil:two-level (best rule) | 94.39% | 182.2 |
| car | PFossil:two-level (macro distribution voting) | 93.75% | 182.2 |
| car | PFossil:two-level (micro distribution voting) | 85.47% | 182.2 |
| car | PFossil:two-level (voting) | 90.68% | 182.2 |
| car | Pypper:one-level (best rule) | 93.98% | 191.6 |
| car | Pypper:one-level (macro distribution voting) | 94.21% | 191.6 |
| car | Pypper:one-level (micro distribution voting) | 91.49% | 191.6 |
| car | Pypper:one-level (voting) | 93.06% | 191.6 |
| car | Pypper:single (best rule) | 91.32% | 191.6 |
| car | Pypper:single (macro distribution voting) | 91.15% | 191.6 |
| car | Pypper:single (micro distribution voting) | 89.99% | 191.6 |
| car | Pypper:single (voting) | 90.39% | 191.6 |
| car | Pypper:two-level (best rule) | 94.39% | 191.6 |
| car | Pypper:two-level (macro distribution voting) | 93.98% | 191.6 |
| car | Pypper:two-level (micro distribution voting) | 92.59% | 191.6 |
| car | Pypper:two-level (voting) | 93.23% | 191.6 |
| churn | PFoil:one-level (best rule) | 90.88% | 1440.8 |
| churn | PFoil:one-level (macro distribution voting) | 92.62% | 1440.8 |
| churn | PFoil:one-level (micro distribution voting) | 86.20% | 1440.8 |
| churn | PFoil:one-level (voting) | 93.50% | 1440.8 |
| churn | PFoil:single (best rule) | 92.36% | 1440.8 |
| churn | PFoil:single (macro distribution voting) | 91.58% | 1440.8 |
| churn | PFoil:single (micro distribution voting) | 89.94% | 1440.8 |
| churn | PFoil:single (voting) | 89.86% | 1440.8 |
| churn | PFoil:two-level (best rule) | 93.28% | 1440.8 |
| churn | PFoil:two-level (macro distribution voting) | 92.98% | 1440.8 |
| churn | PFoil:two-level (micro distribution voting) | 91.32% | 1440.8 |
| churn | PFoil:two-level (voting) | 93.34% | 1440.8 |
| churn | PFossil:one-level (best rule) | 91.00% | 104.3 |
| churn | PFossil:one-level (macro distribution voting) | 91.22% | 104.3 |
| churn | PFossil:one-level (micro distribution voting) | 88.52% | 104.3 |
| churn | PFossil:one-level (voting) | 91.10% | 104.3 |
| churn | PFossil:single (best rule) | 90.00% | 104.3 |
| churn | PFossil:single (macro distribution voting) | 89.98% | 104.3 |
| churn | PFossil:single (micro distribution voting) | 89.18% | 104.3 |
| churn | PFossil:single (voting) | 89.00% | 104.3 |
| churn | PFossil:two-level (best rule) | 91.12% | 104.3 |
| churn | PFossil:two-level (macro distribution voting) | 91.16% | 104.3 |
| churn | PFossil:two-level (micro distribution voting) | 90.28% | 104.3 |
| churn | PFossil:two-level (voting) | 90.20% | 104.3 |
| churn | Pypper:one-level (best rule) | 93.02% | 154.4 |
| churn | Pypper:one-level (macro distribution voting) | 93.72% | 154.4 |
| churn | Pypper:one-level (micro distribution voting) | 92.20% | 154.4 |
| churn | Pypper:one-level (voting) | 93.76% | 154.4 |
| churn | Pypper:single (best rule) | 93.42% | 154.4 |
| churn | Pypper:single (macro distribution voting) | 93.42% | 154.4 |
| churn | Pypper:single (micro distribution voting) | 93.36% | 154.4 |
| churn | Pypper:single (voting) | 93.28% | 154.4 |
| churn | Pypper:two-level (best rule) | 93.74% | 154.4 |
| churn | Pypper:two-level (macro distribution voting) | 93.78% | 154.4 |
| churn | Pypper:two-level (micro distribution voting) | 93.64% | 154.4 |
| churn | Pypper:two-level (voting) | 93.68% | 154.4 |
| climate-model-simulation-crashes | PFoil:one-level (best rule) | 91.85% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level (macro distribution voting) | 93.52% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level (micro distribution voting) | 91.85% | 153.7 |
| climate-model-simulation-crashes | PFoil:one-level (voting) | 94.07% | 153.7 |
| climate-model-simulation-crashes | PFoil:single (best rule) | 93.33% | 153.7 |
| climate-model-simulation-crashes | PFoil:single (macro distribution voting) | 93.89% | 153.7 |
| climate-model-simulation-crashes | PFoil:single (micro distribution voting) | 93.33% | 153.7 |
| climate-model-simulation-crashes | PFoil:single (voting) | 93.52% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level (best rule) | 93.15% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level (macro distribution voting) | 93.70% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level (micro distribution voting) | 92.96% | 153.7 |
| climate-model-simulation-crashes | PFoil:two-level (voting) | 93.33% | 153.7 |
| climate-model-simulation-crashes | PFossil:one-level (best rule) | 91.67% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level (macro distribution voting) | 93.52% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level (micro distribution voting) | 91.48% | 110.2 |
| climate-model-simulation-crashes | PFossil:one-level (voting) | 93.15% | 110.2 |
| climate-model-simulation-crashes | PFossil:single (best rule) | 92.78% | 110.2 |
| climate-model-simulation-crashes | PFossil:single (macro distribution voting) | 92.96% | 110.2 |
| climate-model-simulation-crashes | PFossil:single (micro distribution voting) | 92.78% | 110.2 |
| climate-model-simulation-crashes | PFossil:single (voting) | 92.78% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level (best rule) | 92.41% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level (macro distribution voting) | 93.52% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level (micro distribution voting) | 92.41% | 110.2 |
| climate-model-simulation-crashes | PFossil:two-level (voting) | 92.41% | 110.2 |
| climate-model-simulation-crashes | Pypper:one-level (best rule) | 92.96% | 52.4 |
| climate-model-simulation-crashes | Pypper:one-level (macro distribution voting) | 94.81% | 52.4 |
| climate-model-simulation-crashes | Pypper:one-level (micro distribution voting) | 93.15% | 52.4 |
| climate-model-simulation-crashes | Pypper:one-level (voting) | 94.63% | 52.4 |
| climate-model-simulation-crashes | Pypper:single (best rule) | 94.81% | 52.4 |
| climate-model-simulation-crashes | Pypper:single (macro distribution voting) | 94.44% | 52.4 |
| climate-model-simulation-crashes | Pypper:single (micro distribution voting) | 94.81% | 52.4 |
| climate-model-simulation-crashes | Pypper:single (voting) | 94.81% | 52.4 |
| climate-model-simulation-crashes | Pypper:two-level (best rule) | 94.26% | 52.4 |
| climate-model-simulation-crashes | Pypper:two-level (macro distribution voting) | 94.26% | 52.4 |
| climate-model-simulation-crashes | Pypper:two-level (micro distribution voting) | 94.26% | 52.4 |
| climate-model-simulation-crashes | Pypper:two-level (voting) | 94.26% | 52.4 |
| cmc | PFoil:one-level (best rule) | 53.64% | 2078.5 |
| cmc | PFoil:one-level (macro distribution voting) | 53.78% | 2078.5 |
| cmc | PFoil:one-level (micro distribution voting) | 53.10% | 2078.5 |
| cmc | PFoil:one-level (voting) | 51.94% | 2078.5 |
| cmc | PFoil:single (best rule) | 51.40% | 2078.5 |
| cmc | PFoil:single (macro distribution voting) | 51.40% | 2078.5 |
| cmc | PFoil:single (micro distribution voting) | 52.96% | 2078.5 |
| cmc | PFoil:single (voting) | 50.31% | 2078.5 |
| cmc | PFoil:two-level (best rule) | 52.42% | 2078.5 |
| cmc | PFoil:two-level (macro distribution voting) | 52.49% | 2078.5 |
| cmc | PFoil:two-level (micro distribution voting) | 52.35% | 2078.5 |
| cmc | PFoil:two-level (voting) | 51.40% | 2078.5 |
| cmc | PFossil:one-level (best rule) | 53.02% | 40.9 |
| cmc | PFossil:one-level (macro distribution voting) | 52.82% | 40.9 |
| cmc | PFossil:one-level (micro distribution voting) | 51.74% | 40.9 |
| cmc | PFossil:one-level (voting) | 51.60% | 40.9 |
| cmc | PFossil:single (best rule) | 52.61% | 40.9 |
| cmc | PFossil:single (macro distribution voting) | 52.27% | 40.9 |
| cmc | PFossil:single (micro distribution voting) | 51.73% | 40.9 |
| cmc | PFossil:single (voting) | 51.73% | 40.9 |
| cmc | PFossil:two-level (best rule) | 52.68% | 40.9 |
| cmc | PFossil:two-level (macro distribution voting) | 52.48% | 40.9 |
| cmc | PFossil:two-level (micro distribution voting) | 51.67% | 40.9 |
| cmc | PFossil:two-level (voting) | 51.94% | 40.9 |
| cmc | Pypper:one-level (best rule) | 54.18% | 82.1 |
| cmc | Pypper:one-level (macro distribution voting) | 54.18% | 82.1 |
| cmc | Pypper:one-level (micro distribution voting) | 54.39% | 82.1 |
| cmc | Pypper:one-level (voting) | 54.45% | 82.1 |
| cmc | Pypper:single (best rule) | 52.41% | 82.1 |
| cmc | Pypper:single (macro distribution voting) | 52.48% | 82.1 |
| cmc | Pypper:single (micro distribution voting) | 52.48% | 82.1 |
| cmc | Pypper:single (voting) | 52.48% | 82.1 |
| cmc | Pypper:two-level (best rule) | 54.45% | 82.1 |
| cmc | Pypper:two-level (macro distribution voting) | 54.38% | 82.1 |
| cmc | Pypper:two-level (micro distribution voting) | 54.52% | 82.1 |
| cmc | Pypper:two-level (voting) | 54.38% | 82.1 |
| colic | PFoil:one-level (best rule) | 85.35% | 313.4 |
| colic | PFoil:one-level (macro distribution voting) | 84.80% | 313.4 |
| colic | PFoil:one-level (micro distribution voting) | 85.34% | 313.4 |
| colic | PFoil:one-level (voting) | 85.34% | 313.4 |
| colic | PFoil:single (best rule) | 84.52% | 313.4 |
| colic | PFoil:single (macro distribution voting) | 83.98% | 313.4 |
| colic | PFoil:single (micro distribution voting) | 84.52% | 313.4 |
| colic | PFoil:single (voting) | 83.15% | 313.4 |
| colic | PFoil:two-level (best rule) | 85.62% | 313.4 |
| colic | PFoil:two-level (macro distribution voting) | 85.35% | 313.4 |
| colic | PFoil:two-level (micro distribution voting) | 85.62% | 313.4 |
| colic | PFoil:two-level (voting) | 85.35% | 313.4 |
| colic | PFossil:one-level (best rule) | 81.01% | 168.4 |
| colic | PFossil:one-level (macro distribution voting) | 85.89% | 168.4 |
| colic | PFossil:one-level (micro distribution voting) | 86.43% | 168.4 |
| colic | PFossil:one-level (voting) | 86.70% | 168.4 |
| colic | PFossil:single (best rule) | 82.89% | 168.4 |
| colic | PFossil:single (macro distribution voting) | 81.79% | 168.4 |
| colic | PFossil:single (micro distribution voting) | 86.15% | 168.4 |
| colic | PFossil:single (voting) | 82.88% | 168.4 |
| colic | PFossil:two-level (best rule) | 85.89% | 168.4 |
| colic | PFossil:two-level (macro distribution voting) | 85.89% | 168.4 |
| colic | PFossil:two-level (micro distribution voting) | 86.70% | 168.4 |
| colic | PFossil:two-level (voting) | 85.61% | 168.4 |
| colic | Pypper:one-level (best rule) | 83.45% | 51.1 |
| colic | Pypper:one-level (macro distribution voting) | 86.15% | 51.1 |
| colic | Pypper:one-level (micro distribution voting) | 85.88% | 51.1 |
| colic | Pypper:one-level (voting) | 85.88% | 51.1 |
| colic | Pypper:single (best rule) | 85.88% | 51.1 |
| colic | Pypper:single (macro distribution voting) | 85.88% | 51.1 |
| colic | Pypper:single (micro distribution voting) | 85.88% | 51.1 |
| colic | Pypper:single (voting) | 85.88% | 51.1 |
| colic | Pypper:two-level (best rule) | 85.61% | 51.1 |
| colic | Pypper:two-level (macro distribution voting) | 85.88% | 51.1 |
| colic | Pypper:two-level (micro distribution voting) | 85.88% | 51.1 |
| colic | Pypper:two-level (voting) | 85.06% | 51.1 |
| compas-two-years | PFoil:one-level (best rule) | 67.09% | 3583.1 |
| compas-two-years | PFoil:one-level (macro distribution voting) | 67.18% | 3583.1 |
| compas-two-years | PFoil:one-level (micro distribution voting) | 63.57% | 3583.1 |
| compas-two-years | PFoil:one-level (voting) | 65.92% | 3583.1 |
| compas-two-years | PFoil:single (best rule) | 66.65% | 3583.1 |
| compas-two-years | PFoil:single (macro distribution voting) | 66.24% | 3583.1 |
| compas-two-years | PFoil:single (micro distribution voting) | 62.26% | 3583.1 |
| compas-two-years | PFoil:single (voting) | 63.94% | 3583.1 |
| compas-two-years | PFoil:two-level (best rule) | 66.79% | 3583.1 |
| compas-two-years | PFoil:two-level (macro distribution voting) | 66.41% | 3583.1 |
| compas-two-years | PFoil:two-level (micro distribution voting) | 64.15% | 3583.1 |
| compas-two-years | PFoil:two-level (voting) | 64.63% | 3583.1 |
| compas-two-years | PFossil:one-level (best rule) | 64.65% | 12.1 |
| compas-two-years | PFossil:one-level (macro distribution voting) | 64.85% | 12.1 |
| compas-two-years | PFossil:one-level (micro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:one-level (voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:single (best rule) | 52.96% | 12.1 |
| compas-two-years | PFossil:single (macro distribution voting) | 52.96% | 12.1 |
| compas-two-years | PFossil:single (micro distribution voting) | 52.96% | 12.1 |
| compas-two-years | PFossil:single (voting) | 52.96% | 12.1 |
| compas-two-years | PFossil:two-level (best rule) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level (macro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level (micro distribution voting) | 64.95% | 12.1 |
| compas-two-years | PFossil:two-level (voting) | 64.95% | 12.1 |
| compas-two-years | Pypper:one-level (best rule) | 68.13% | 90.5 |
| compas-two-years | Pypper:one-level (macro distribution voting) | 67.81% | 90.5 |
| compas-two-years | Pypper:one-level (micro distribution voting) | 67.26% | 90.5 |
| compas-two-years | Pypper:one-level (voting) | 67.89% | 90.5 |
| compas-two-years | Pypper:single (best rule) | 67.05% | 90.5 |
| compas-two-years | Pypper:single (macro distribution voting) | 67.05% | 90.5 |
| compas-two-years | Pypper:single (micro distribution voting) | 67.13% | 90.5 |
| compas-two-years | Pypper:single (voting) | 67.15% | 90.5 |
| compas-two-years | Pypper:two-level (best rule) | 67.75% | 90.5 |
| compas-two-years | Pypper:two-level (macro distribution voting) | 67.73% | 90.5 |
| compas-two-years | Pypper:two-level (micro distribution voting) | 67.51% | 90.5 |
| compas-two-years | Pypper:two-level (voting) | 67.53% | 90.5 |
| credit-approval | PFoil:one-level (best rule) | 86.38% | 460.0 |
| credit-approval | PFoil:one-level (macro distribution voting) | 85.94% | 460.0 |
| credit-approval | PFoil:one-level (micro distribution voting) | 87.39% | 460.0 |
| credit-approval | PFoil:one-level (voting) | 85.80% | 460.0 |
| credit-approval | PFoil:single (best rule) | 84.49% | 460.0 |
| credit-approval | PFoil:single (macro distribution voting) | 85.07% | 460.0 |
| credit-approval | PFoil:single (micro distribution voting) | 83.77% | 460.0 |
| credit-approval | PFoil:single (voting) | 83.33% | 460.0 |
| credit-approval | PFoil:two-level (best rule) | 85.51% | 460.0 |
| credit-approval | PFoil:two-level (macro distribution voting) | 85.65% | 460.0 |
| credit-approval | PFoil:two-level (micro distribution voting) | 85.22% | 460.0 |
| credit-approval | PFoil:two-level (voting) | 84.78% | 460.0 |
| credit-approval | PFossil:one-level (best rule) | 83.77% | 149.4 |
| credit-approval | PFossil:one-level (macro distribution voting) | 84.78% | 149.4 |
| credit-approval | PFossil:one-level (micro distribution voting) | 85.65% | 149.4 |
| credit-approval | PFossil:one-level (voting) | 85.07% | 149.4 |
| credit-approval | PFossil:single (best rule) | 82.17% | 149.4 |
| credit-approval | PFossil:single (macro distribution voting) | 82.03% | 149.4 |
| credit-approval | PFossil:single (micro distribution voting) | 85.07% | 149.4 |
| credit-approval | PFossil:single (voting) | 82.46% | 149.4 |
| credit-approval | PFossil:two-level (best rule) | 85.07% | 149.4 |
| credit-approval | PFossil:two-level (macro distribution voting) | 85.22% | 149.4 |
| credit-approval | PFossil:two-level (micro distribution voting) | 85.36% | 149.4 |
| credit-approval | PFossil:two-level (voting) | 84.78% | 149.4 |
| credit-approval | Pypper:one-level (best rule) | 86.23% | 58.6 |
| credit-approval | Pypper:one-level (macro distribution voting) | 85.80% | 58.6 |
| credit-approval | Pypper:one-level (micro distribution voting) | 85.51% | 58.6 |
| credit-approval | Pypper:one-level (voting) | 85.65% | 58.6 |
| credit-approval | Pypper:single (best rule) | 84.20% | 58.6 |
| credit-approval | Pypper:single (macro distribution voting) | 84.20% | 58.6 |
| credit-approval | Pypper:single (micro distribution voting) | 83.77% | 58.6 |
| credit-approval | Pypper:single (voting) | 83.04% | 58.6 |
| credit-approval | Pypper:two-level (best rule) | 86.09% | 58.6 |
| credit-approval | Pypper:two-level (macro distribution voting) | 86.09% | 58.6 |
| credit-approval | Pypper:two-level (micro distribution voting) | 85.22% | 58.6 |
| credit-approval | Pypper:two-level (voting) | 85.94% | 58.6 |
| credit-g | PFoil:one-level (best rule) | 72.70% | 748.9 |
| credit-g | PFoil:one-level (macro distribution voting) | 76.40% | 748.9 |
| credit-g | PFoil:one-level (micro distribution voting) | 72.90% | 748.9 |
| credit-g | PFoil:one-level (voting) | 77.00% | 748.9 |
| credit-g | PFoil:single (best rule) | 73.10% | 748.9 |
| credit-g | PFoil:single (macro distribution voting) | 73.50% | 748.9 |
| credit-g | PFoil:single (micro distribution voting) | 72.30% | 748.9 |
| credit-g | PFoil:single (voting) | 72.30% | 748.9 |
| credit-g | PFoil:two-level (best rule) | 76.00% | 748.9 |
| credit-g | PFoil:two-level (macro distribution voting) | 76.00% | 748.9 |
| credit-g | PFoil:two-level (micro distribution voting) | 74.40% | 748.9 |
| credit-g | PFoil:two-level (voting) | 76.10% | 748.9 |
| credit-g | PFossil:one-level (best rule) | 73.80% | 81.7 |
| credit-g | PFossil:one-level (macro distribution voting) | 74.60% | 81.7 |
| credit-g | PFossil:one-level (micro distribution voting) | 75.80% | 81.7 |
| credit-g | PFossil:one-level (voting) | 72.90% | 81.7 |
| credit-g | PFossil:single (best rule) | 73.20% | 81.7 |
| credit-g | PFossil:single (macro distribution voting) | 73.20% | 81.7 |
| credit-g | PFossil:single (micro distribution voting) | 73.30% | 81.7 |
| credit-g | PFossil:single (voting) | 73.20% | 81.7 |
| credit-g | PFossil:two-level (best rule) | 73.50% | 81.7 |
| credit-g | PFossil:two-level (macro distribution voting) | 73.50% | 81.7 |
| credit-g | PFossil:two-level (micro distribution voting) | 73.40% | 81.7 |
| credit-g | PFossil:two-level (voting) | 73.20% | 81.7 |
| credit-g | Pypper:one-level (best rule) | 74.30% | 79.1 |
| credit-g | Pypper:one-level (macro distribution voting) | 74.30% | 79.1 |
| credit-g | Pypper:one-level (micro distribution voting) | 72.70% | 79.1 |
| credit-g | Pypper:one-level (voting) | 74.20% | 79.1 |
| credit-g | Pypper:single (best rule) | 71.40% | 79.1 |
| credit-g | Pypper:single (macro distribution voting) | 71.50% | 79.1 |
| credit-g | Pypper:single (micro distribution voting) | 71.50% | 79.1 |
| credit-g | Pypper:single (voting) | 71.50% | 79.1 |
| credit-g | Pypper:two-level (best rule) | 74.10% | 79.1 |
| credit-g | Pypper:two-level (macro distribution voting) | 73.70% | 79.1 |
| credit-g | Pypper:two-level (micro distribution voting) | 74.20% | 79.1 |
| credit-g | Pypper:two-level (voting) | 74.00% | 79.1 |
| cylinder-bands | PFoil:one-level (best rule) | 73.52% | 407.2 |
| cylinder-bands | PFoil:one-level (macro distribution voting) | 81.48% | 407.2 |
| cylinder-bands | PFoil:one-level (micro distribution voting) | 77.59% | 407.2 |
| cylinder-bands | PFoil:one-level (voting) | 81.48% | 407.2 |
| cylinder-bands | PFoil:single (best rule) | 75.00% | 407.2 |
| cylinder-bands | PFoil:single (macro distribution voting) | 76.30% | 407.2 |
| cylinder-bands | PFoil:single (micro distribution voting) | 73.70% | 407.2 |
| cylinder-bands | PFoil:single (voting) | 74.07% | 407.2 |
| cylinder-bands | PFoil:two-level (best rule) | 78.89% | 407.2 |
| cylinder-bands | PFoil:two-level (macro distribution voting) | 79.44% | 407.2 |
| cylinder-bands | PFoil:two-level (micro distribution voting) | 77.96% | 407.2 |
| cylinder-bands | PFoil:two-level (voting) | 79.07% | 407.2 |
| cylinder-bands | PFossil:one-level (best rule) | 75.37% | 103.5 |
| cylinder-bands | PFossil:one-level (macro distribution voting) | 74.81% | 103.5 |
| cylinder-bands | PFossil:one-level (micro distribution voting) | 72.41% | 103.5 |
| cylinder-bands | PFossil:one-level (voting) | 74.44% | 103.5 |
| cylinder-bands | PFossil:single (best rule) | 70.19% | 103.5 |
| cylinder-bands | PFossil:single (macro distribution voting) | 70.56% | 103.5 |
| cylinder-bands | PFossil:single (micro distribution voting) | 68.15% | 103.5 |
| cylinder-bands | PFossil:single (voting) | 67.96% | 103.5 |
| cylinder-bands | PFossil:two-level (best rule) | 74.44% | 103.5 |
| cylinder-bands | PFossil:two-level (macro distribution voting) | 74.26% | 103.5 |
| cylinder-bands | PFossil:two-level (micro distribution voting) | 72.96% | 103.5 |
| cylinder-bands | PFossil:two-level (voting) | 72.41% | 103.5 |
| cylinder-bands | Pypper:one-level (best rule) | 75.93% | 81.3 |
| cylinder-bands | Pypper:one-level (macro distribution voting) | 75.37% | 81.3 |
| cylinder-bands | Pypper:one-level (micro distribution voting) | 70.19% | 81.3 |
| cylinder-bands | Pypper:one-level (voting) | 74.26% | 81.3 |
| cylinder-bands | Pypper:single (best rule) | 68.33% | 81.3 |
| cylinder-bands | Pypper:single (macro distribution voting) | 67.96% | 81.3 |
| cylinder-bands | Pypper:single (micro distribution voting) | 67.41% | 81.3 |
| cylinder-bands | Pypper:single (voting) | 68.33% | 81.3 |
| cylinder-bands | Pypper:two-level (best rule) | 77.59% | 81.3 |
| cylinder-bands | Pypper:two-level (macro distribution voting) | 76.48% | 81.3 |
| cylinder-bands | Pypper:two-level (micro distribution voting) | 72.41% | 81.3 |
| cylinder-bands | Pypper:two-level (voting) | 72.96% | 81.3 |
| diabetes | PFoil:one-level (best rule) | 73.82% | 709.9 |
| diabetes | PFoil:one-level (macro distribution voting) | 75.39% | 709.9 |
| diabetes | PFoil:one-level (micro distribution voting) | 74.47% | 709.9 |
| diabetes | PFoil:one-level (voting) | 77.60% | 709.9 |
| diabetes | PFoil:single (best rule) | 75.77% | 709.9 |
| diabetes | PFoil:single (macro distribution voting) | 75.12% | 709.9 |
| diabetes | PFoil:single (micro distribution voting) | 73.44% | 709.9 |
| diabetes | PFoil:single (voting) | 75.40% | 709.9 |
| diabetes | PFoil:two-level (best rule) | 75.65% | 709.9 |
| diabetes | PFoil:two-level (macro distribution voting) | 75.65% | 709.9 |
| diabetes | PFoil:two-level (micro distribution voting) | 75.78% | 709.9 |
| diabetes | PFoil:two-level (voting) | 74.60% | 709.9 |
| diabetes | PFossil:one-level (best rule) | 72.79% | 127.4 |
| diabetes | PFossil:one-level (macro distribution voting) | 74.34% | 127.4 |
| diabetes | PFossil:one-level (micro distribution voting) | 75.00% | 127.4 |
| diabetes | PFossil:one-level (voting) | 76.04% | 127.4 |
| diabetes | PFossil:single (best rule) | 74.61% | 127.4 |
| diabetes | PFossil:single (macro distribution voting) | 73.83% | 127.4 |
| diabetes | PFossil:single (micro distribution voting) | 75.13% | 127.4 |
| diabetes | PFossil:single (voting) | 74.09% | 127.4 |
| diabetes | PFossil:two-level (best rule) | 75.39% | 127.4 |
| diabetes | PFossil:two-level (macro distribution voting) | 74.74% | 127.4 |
| diabetes | PFossil:two-level (micro distribution voting) | 75.26% | 127.4 |
| diabetes | PFossil:two-level (voting) | 74.99% | 127.4 |
| diabetes | Pypper:one-level (best rule) | 74.73% | 75.5 |
| diabetes | Pypper:one-level (macro distribution voting) | 74.08% | 75.5 |
| diabetes | Pypper:one-level (micro distribution voting) | 75.25% | 75.5 |
| diabetes | Pypper:one-level (voting) | 74.34% | 75.5 |
| diabetes | Pypper:single (best rule) | 76.04% | 75.5 |
| diabetes | Pypper:single (macro distribution voting) | 76.04% | 75.5 |
| diabetes | Pypper:single (micro distribution voting) | 75.78% | 75.5 |
| diabetes | Pypper:single (voting) | 75.91% | 75.5 |
| diabetes | Pypper:two-level (best rule) | 74.60% | 75.5 |
| diabetes | Pypper:two-level (macro distribution voting) | 74.21% | 75.5 |
| diabetes | Pypper:two-level (micro distribution voting) | 74.08% | 75.5 |
| diabetes | Pypper:two-level (voting) | 73.95% | 75.5 |
| dna | PFoil:one-level (best rule) | 92.91% | 699.4 |
| dna | PFoil:one-level (macro distribution voting) | 95.13% | 699.4 |
| dna | PFoil:one-level (micro distribution voting) | 94.76% | 699.4 |
| dna | PFoil:one-level (voting) | 95.01% | 699.4 |
| dna | PFoil:single (best rule) | 92.50% | 699.4 |
| dna | PFoil:single (macro distribution voting) | 91.97% | 699.4 |
| dna | PFoil:single (micro distribution voting) | 92.00% | 699.4 |
| dna | PFoil:single (voting) | 91.05% | 699.4 |
| dna | PFoil:two-level (best rule) | 94.73% | 699.4 |
| dna | PFoil:two-level (macro distribution voting) | 94.95% | 699.4 |
| dna | PFoil:two-level (micro distribution voting) | 94.69% | 699.4 |
| dna | PFoil:two-level (voting) | 95.04% | 699.4 |
| dna | PFossil:one-level (best rule) | 90.58% | 327.8 |
| dna | PFossil:one-level (macro distribution voting) | 95.01% | 327.8 |
| dna | PFossil:one-level (micro distribution voting) | 89.96% | 327.8 |
| dna | PFossil:one-level (voting) | 92.78% | 327.8 |
| dna | PFossil:single (best rule) | 93.69% | 327.8 |
| dna | PFossil:single (macro distribution voting) | 93.53% | 327.8 |
| dna | PFossil:single (micro distribution voting) | 89.89% | 327.8 |
| dna | PFossil:single (voting) | 90.65% | 327.8 |
| dna | PFossil:two-level (best rule) | 94.91% | 327.8 |
| dna | PFossil:two-level (macro distribution voting) | 94.85% | 327.8 |
| dna | PFossil:two-level (micro distribution voting) | 90.08% | 327.8 |
| dna | PFossil:two-level (voting) | 90.43% | 327.8 |
| dna | Pypper:one-level (best rule) | 92.37% | 183.4 |
| dna | Pypper:one-level (macro distribution voting) | 93.19% | 183.4 |
| dna | Pypper:one-level (micro distribution voting) | 91.59% | 183.4 |
| dna | Pypper:one-level (voting) | 93.19% | 183.4 |
| dna | Pypper:single (best rule) | 93.75% | 183.4 |
| dna | Pypper:single (macro distribution voting) | 93.60% | 183.4 |
| dna | Pypper:single (micro distribution voting) | 93.31% | 183.4 |
| dna | Pypper:single (voting) | 93.60% | 183.4 |
| dna | Pypper:two-level (best rule) | 93.44% | 183.4 |
| dna | Pypper:two-level (macro distribution voting) | 93.47% | 183.4 |
| dna | Pypper:two-level (micro distribution voting) | 92.40% | 183.4 |
| dna | Pypper:two-level (voting) | 93.31% | 183.4 |
| dresses-sales | PFoil:one-level (best rule) | 56.60% | 598.7 |
| dresses-sales | PFoil:one-level (macro distribution voting) | 58.60% | 598.7 |
| dresses-sales | PFoil:one-level (micro distribution voting) | 61.00% | 598.7 |
| dresses-sales | PFoil:one-level (voting) | 59.80% | 598.7 |
| dresses-sales | PFoil:single (best rule) | 58.20% | 598.7 |
| dresses-sales | PFoil:single (macro distribution voting) | 59.00% | 598.7 |
| dresses-sales | PFoil:single (micro distribution voting) | 60.00% | 598.7 |
| dresses-sales | PFoil:single (voting) | 56.80% | 598.7 |
| dresses-sales | PFoil:two-level (best rule) | 58.60% | 598.7 |
| dresses-sales | PFoil:two-level (macro distribution voting) | 60.80% | 598.7 |
| dresses-sales | PFoil:two-level (micro distribution voting) | 60.80% | 598.7 |
| dresses-sales | PFoil:two-level (voting) | 58.00% | 598.7 |
| dresses-sales | PFossil:one-level (best rule) | 59.60% | 46.4 |
| dresses-sales | PFossil:one-level (macro distribution voting) | 61.40% | 46.4 |
| dresses-sales | PFossil:one-level (micro distribution voting) | 59.00% | 46.4 |
| dresses-sales | PFossil:one-level (voting) | 60.20% | 46.4 |
| dresses-sales | PFossil:single (best rule) | 59.40% | 46.4 |
| dresses-sales | PFossil:single (macro distribution voting) | 59.40% | 46.4 |
| dresses-sales | PFossil:single (micro distribution voting) | 59.20% | 46.4 |
| dresses-sales | PFossil:single (voting) | 59.20% | 46.4 |
| dresses-sales | PFossil:two-level (best rule) | 59.40% | 46.4 |
| dresses-sales | PFossil:two-level (macro distribution voting) | 59.60% | 46.4 |
| dresses-sales | PFossil:two-level (micro distribution voting) | 60.20% | 46.4 |
| dresses-sales | PFossil:two-level (voting) | 60.60% | 46.4 |
| dresses-sales | Pypper:one-level (best rule) | 60.40% | 42.8 |
| dresses-sales | Pypper:one-level (macro distribution voting) | 62.40% | 42.8 |
| dresses-sales | Pypper:one-level (micro distribution voting) | 60.20% | 42.8 |
| dresses-sales | Pypper:one-level (voting) | 62.00% | 42.8 |
| dresses-sales | Pypper:single (best rule) | 60.40% | 42.8 |
| dresses-sales | Pypper:single (macro distribution voting) | 60.40% | 42.8 |
| dresses-sales | Pypper:single (micro distribution voting) | 60.60% | 42.8 |
| dresses-sales | Pypper:single (voting) | 60.60% | 42.8 |
| dresses-sales | Pypper:two-level (best rule) | 62.60% | 42.8 |
| dresses-sales | Pypper:two-level (macro distribution voting) | 62.60% | 42.8 |
| dresses-sales | Pypper:two-level (micro distribution voting) | 62.80% | 42.8 |
| dresses-sales | Pypper:two-level (voting) | 62.20% | 42.8 |
| ecoli | PFoil:one-level (best rule) | 82.47% | 342.2 |
| ecoli | PFoil:one-level (macro distribution voting) | 86.65% | 342.2 |
| ecoli | PFoil:one-level (micro distribution voting) | 85.16% | 342.2 |
| ecoli | PFoil:one-level (voting) | 84.57% | 342.2 |
| ecoli | PFoil:single (best rule) | 82.51% | 342.2 |
| ecoli | PFoil:single (macro distribution voting) | 83.10% | 342.2 |
| ecoli | PFoil:single (micro distribution voting) | 79.22% | 342.2 |
| ecoli | PFoil:single (voting) | 79.80% | 342.2 |
| ecoli | PFoil:two-level (best rule) | 84.57% | 342.2 |
| ecoli | PFoil:two-level (macro distribution voting) | 85.16% | 342.2 |
| ecoli | PFoil:two-level (micro distribution voting) | 84.55% | 342.2 |
| ecoli | PFoil:two-level (voting) | 85.47% | 342.2 |
| ecoli | PFossil:one-level (best rule) | 80.99% | 240.8 |
| ecoli | PFossil:one-level (macro distribution voting) | 86.38% | 240.8 |
| ecoli | PFossil:one-level (micro distribution voting) | 85.18% | 240.8 |
| ecoli | PFossil:one-level (voting) | 86.97% | 240.8 |
| ecoli | PFossil:single (best rule) | 81.92% | 240.8 |
| ecoli | PFossil:single (macro distribution voting) | 82.22% | 240.8 |
| ecoli | PFossil:single (micro distribution voting) | 84.29% | 240.8 |
| ecoli | PFossil:single (voting) | 83.98% | 240.8 |
| ecoli | PFossil:two-level (best rule) | 85.19% | 240.8 |
| ecoli | PFossil:two-level (macro distribution voting) | 86.08% | 240.8 |
| ecoli | PFossil:two-level (micro distribution voting) | 86.08% | 240.8 |
| ecoli | PFossil:two-level (voting) | 86.07% | 240.8 |
| ecoli | Pypper:one-level (best rule) | 83.68% | 87.5 |
| ecoli | Pypper:one-level (macro distribution voting) | 85.78% | 87.5 |
| ecoli | Pypper:one-level (micro distribution voting) | 82.17% | 87.5 |
| ecoli | Pypper:one-level (voting) | 85.78% | 87.5 |
| ecoli | Pypper:single (best rule) | 81.91% | 87.5 |
| ecoli | Pypper:single (macro distribution voting) | 81.30% | 87.5 |
| ecoli | Pypper:single (micro distribution voting) | 81.29% | 87.5 |
| ecoli | Pypper:single (voting) | 81.29% | 87.5 |
| ecoli | Pypper:two-level (best rule) | 85.78% | 87.5 |
| ecoli | Pypper:two-level (macro distribution voting) | 86.08% | 87.5 |
| ecoli | Pypper:two-level (micro distribution voting) | 83.96% | 87.5 |
| ecoli | Pypper:two-level (voting) | 84.26% | 87.5 |
| glass | PFoil:one-level (best rule) | 70.67% | 279.3 |
| glass | PFoil:one-level (macro distribution voting) | 74.81% | 279.3 |
| glass | PFoil:one-level (micro distribution voting) | 69.26% | 279.3 |
| glass | PFoil:one-level (voting) | 74.78% | 279.3 |
| glass | PFoil:single (best rule) | 72.94% | 279.3 |
| glass | PFoil:single (macro distribution voting) | 72.47% | 279.3 |
| glass | PFoil:single (micro distribution voting) | 66.00% | 279.3 |
| glass | PFoil:single (voting) | 67.38% | 279.3 |
| glass | PFoil:two-level (best rule) | 75.24% | 279.3 |
| glass | PFoil:two-level (macro distribution voting) | 73.90% | 279.3 |
| glass | PFoil:two-level (micro distribution voting) | 72.53% | 279.3 |
| glass | PFoil:two-level (voting) | 73.01% | 279.3 |
| glass | PFossil:one-level (best rule) | 66.36% | 238.9 |
| glass | PFossil:one-level (macro distribution voting) | 73.38% | 238.9 |
| glass | PFossil:one-level (micro distribution voting) | 72.03% | 238.9 |
| glass | PFossil:one-level (voting) | 73.38% | 238.9 |
| glass | PFossil:single (best rule) | 71.97% | 238.9 |
| glass | PFossil:single (macro distribution voting) | 70.58% | 238.9 |
| glass | PFossil:single (micro distribution voting) | 70.11% | 238.9 |
| glass | PFossil:single (voting) | 69.68% | 238.9 |
| glass | PFossil:two-level (best rule) | 72.92% | 238.9 |
| glass | PFossil:two-level (macro distribution voting) | 73.85% | 238.9 |
| glass | PFossil:two-level (micro distribution voting) | 74.81% | 238.9 |
| glass | PFossil:two-level (voting) | 72.90% | 238.9 |
| glass | Pypper:one-level (best rule) | 72.47% | 80.8 |
| glass | Pypper:one-level (macro distribution voting) | 72.49% | 80.8 |
| glass | Pypper:one-level (micro distribution voting) | 68.77% | 80.8 |
| glass | Pypper:one-level (voting) | 70.63% | 80.8 |
| glass | Pypper:single (best rule) | 64.55% | 80.8 |
| glass | Pypper:single (macro distribution voting) | 62.68% | 80.8 |
| glass | Pypper:single (micro distribution voting) | 62.75% | 80.8 |
| glass | Pypper:single (voting) | 64.52% | 80.8 |
| glass | Pypper:two-level (best rule) | 72.99% | 80.8 |
| glass | Pypper:two-level (macro distribution voting) | 73.01% | 80.8 |
| glass | Pypper:two-level (micro distribution voting) | 69.76% | 80.8 |
| glass | Pypper:two-level (voting) | 69.72% | 80.8 |
| hayes-roth | PFoil:one-level (best rule) | 77.50% | 229.1 |
| hayes-roth | PFoil:one-level (macro distribution voting) | 78.75% | 229.1 |
| hayes-roth | PFoil:one-level (micro distribution voting) | 79.38% | 229.1 |
| hayes-roth | PFoil:one-level (voting) | 80.00% | 229.1 |
| hayes-roth | PFoil:single (best rule) | 77.50% | 229.1 |
| hayes-roth | PFoil:single (macro distribution voting) | 72.50% | 229.1 |
| hayes-roth | PFoil:single (micro distribution voting) | 76.88% | 229.1 |
| hayes-roth | PFoil:single (voting) | 70.62% | 229.1 |
| hayes-roth | PFoil:two-level (best rule) | 81.88% | 229.1 |
| hayes-roth | PFoil:two-level (macro distribution voting) | 76.88% | 229.1 |
| hayes-roth | PFoil:two-level (micro distribution voting) | 81.88% | 229.1 |
| hayes-roth | PFoil:two-level (voting) | 78.12% | 229.1 |
| hayes-roth | PFossil:one-level (best rule) | 78.75% | 139.2 |
| hayes-roth | PFossil:one-level (macro distribution voting) | 85.00% | 139.2 |
| hayes-roth | PFossil:one-level (micro distribution voting) | 80.00% | 139.2 |
| hayes-roth | PFossil:one-level (voting) | 84.38% | 139.2 |
| hayes-roth | PFossil:single (best rule) | 82.50% | 139.2 |
| hayes-roth | PFossil:single (macro distribution voting) | 81.88% | 139.2 |
| hayes-roth | PFossil:single (micro distribution voting) | 81.88% | 139.2 |
| hayes-roth | PFossil:single (voting) | 81.25% | 139.2 |
| hayes-roth | PFossil:two-level (best rule) | 85.62% | 139.2 |
| hayes-roth | PFossil:two-level (macro distribution voting) | 81.88% | 139.2 |
| hayes-roth | PFossil:two-level (micro distribution voting) | 80.00% | 139.2 |
| hayes-roth | PFossil:two-level (voting) | 78.12% | 139.2 |
| hayes-roth | Pypper:one-level (best rule) | 80.62% | 86.1 |
| hayes-roth | Pypper:one-level (macro distribution voting) | 83.12% | 86.1 |
| hayes-roth | Pypper:one-level (micro distribution voting) | 77.50% | 86.1 |
| hayes-roth | Pypper:one-level (voting) | 83.12% | 86.1 |
| hayes-roth | Pypper:single (best rule) | 79.38% | 86.1 |
| hayes-roth | Pypper:single (macro distribution voting) | 79.38% | 86.1 |
| hayes-roth | Pypper:single (micro distribution voting) | 75.00% | 86.1 |
| hayes-roth | Pypper:single (voting) | 75.00% | 86.1 |
| hayes-roth | Pypper:two-level (best rule) | 83.12% | 86.1 |
| hayes-roth | Pypper:two-level (macro distribution voting) | 81.88% | 86.1 |
| hayes-roth | Pypper:two-level (micro distribution voting) | 80.00% | 86.1 |
| hayes-roth | Pypper:two-level (voting) | 78.12% | 86.1 |
| heart-c | PFoil:one-level (best rule) | 79.58% | 231.0 |
| heart-c | PFoil:one-level (macro distribution voting) | 83.52% | 231.0 |
| heart-c | PFoil:one-level (micro distribution voting) | 84.19% | 231.0 |
| heart-c | PFoil:one-level (voting) | 82.54% | 231.0 |
| heart-c | PFoil:single (best rule) | 78.92% | 231.0 |
| heart-c | PFoil:single (macro distribution voting) | 78.59% | 231.0 |
| heart-c | PFoil:single (micro distribution voting) | 78.62% | 231.0 |
| heart-c | PFoil:single (voting) | 77.61% | 231.0 |
| heart-c | PFoil:two-level (best rule) | 83.23% | 231.0 |
| heart-c | PFoil:two-level (macro distribution voting) | 81.89% | 231.0 |
| heart-c | PFoil:two-level (micro distribution voting) | 83.84% | 231.0 |
| heart-c | PFoil:two-level (voting) | 80.58% | 231.0 |
| heart-c | PFossil:one-level (best rule) | 76.92% | 153.4 |
| heart-c | PFossil:one-level (macro distribution voting) | 83.53% | 153.4 |
| heart-c | PFossil:one-level (micro distribution voting) | 80.89% | 153.4 |
| heart-c | PFossil:one-level (voting) | 82.86% | 153.4 |
| heart-c | PFossil:single (best rule) | 79.54% | 153.4 |
| heart-c | PFossil:single (macro distribution voting) | 79.88% | 153.4 |
| heart-c | PFossil:single (micro distribution voting) | 77.24% | 153.4 |
| heart-c | PFossil:single (voting) | 78.24% | 153.4 |
| heart-c | PFossil:two-level (best rule) | 81.54% | 153.4 |
| heart-c | PFossil:two-level (macro distribution voting) | 81.87% | 153.4 |
| heart-c | PFossil:two-level (micro distribution voting) | 79.89% | 153.4 |
| heart-c | PFossil:two-level (voting) | 82.22% | 153.4 |
| heart-c | Pypper:one-level (best rule) | 79.90% | 58.5 |
| heart-c | Pypper:one-level (macro distribution voting) | 83.22% | 58.5 |
| heart-c | Pypper:one-level (micro distribution voting) | 83.54% | 58.5 |
| heart-c | Pypper:one-level (voting) | 82.55% | 58.5 |
| heart-c | Pypper:single (best rule) | 82.55% | 58.5 |
| heart-c | Pypper:single (macro distribution voting) | 82.55% | 58.5 |
| heart-c | Pypper:single (micro distribution voting) | 82.56% | 58.5 |
| heart-c | Pypper:single (voting) | 82.56% | 58.5 |
| heart-c | Pypper:two-level (best rule) | 82.24% | 58.5 |
| heart-c | Pypper:two-level (macro distribution voting) | 82.89% | 58.5 |
| heart-c | Pypper:two-level (micro distribution voting) | 83.56% | 58.5 |
| heart-c | Pypper:two-level (voting) | 83.56% | 58.5 |
| heart-h | PFoil:one-level (best rule) | 78.90% | 264.1 |
| heart-h | PFoil:one-level (macro distribution voting) | 80.28% | 264.1 |
| heart-h | PFoil:one-level (micro distribution voting) | 79.57% | 264.1 |
| heart-h | PFoil:one-level (voting) | 80.62% | 264.1 |
| heart-h | PFoil:single (best rule) | 78.93% | 264.1 |
| heart-h | PFoil:single (macro distribution voting) | 79.25% | 264.1 |
| heart-h | PFoil:single (micro distribution voting) | 79.29% | 264.1 |
| heart-h | PFoil:single (voting) | 78.56% | 264.1 |
| heart-h | PFoil:two-level (best rule) | 79.61% | 264.1 |
| heart-h | PFoil:two-level (macro distribution voting) | 79.61% | 264.1 |
| heart-h | PFoil:two-level (micro distribution voting) | 79.92% | 264.1 |
| heart-h | PFoil:two-level (voting) | 80.29% | 264.1 |
| heart-h | PFossil:one-level (best rule) | 80.92% | 144.5 |
| heart-h | PFossil:one-level (macro distribution voting) | 80.94% | 144.5 |
| heart-h | PFossil:one-level (micro distribution voting) | 80.60% | 144.5 |
| heart-h | PFossil:one-level (voting) | 81.29% | 144.5 |
| heart-h | PFossil:single (best rule) | 81.28% | 144.5 |
| heart-h | PFossil:single (macro distribution voting) | 79.92% | 144.5 |
| heart-h | PFossil:single (micro distribution voting) | 78.93% | 144.5 |
| heart-h | PFossil:single (voting) | 80.94% | 144.5 |
| heart-h | PFossil:two-level (best rule) | 80.60% | 144.5 |
| heart-h | PFossil:two-level (macro distribution voting) | 80.94% | 144.5 |
| heart-h | PFossil:two-level (micro distribution voting) | 81.99% | 144.5 |
| heart-h | PFossil:two-level (voting) | 79.94% | 144.5 |
| heart-h | Pypper:one-level (best rule) | 80.25% | 49.2 |
| heart-h | Pypper:one-level (macro distribution voting) | 80.61% | 49.2 |
| heart-h | Pypper:one-level (micro distribution voting) | 80.95% | 49.2 |
| heart-h | Pypper:one-level (voting) | 80.61% | 49.2 |
| heart-h | Pypper:single (best rule) | 78.20% | 49.2 |
| heart-h | Pypper:single (macro distribution voting) | 78.20% | 49.2 |
| heart-h | Pypper:single (micro distribution voting) | 77.85% | 49.2 |
| heart-h | Pypper:single (voting) | 77.85% | 49.2 |
| heart-h | Pypper:two-level (best rule) | 80.60% | 49.2 |
| heart-h | Pypper:two-level (macro distribution voting) | 80.94% | 49.2 |
| heart-h | Pypper:two-level (micro distribution voting) | 80.26% | 49.2 |
| heart-h | Pypper:two-level (voting) | 80.60% | 49.2 |
| heart-statlog | PFoil:one-level (best rule) | 83.33% | 214.4 |
| heart-statlog | PFoil:one-level (macro distribution voting) | 83.70% | 214.4 |
| heart-statlog | PFoil:one-level (micro distribution voting) | 82.96% | 214.4 |
| heart-statlog | PFoil:one-level (voting) | 80.74% | 214.4 |
| heart-statlog | PFoil:single (best rule) | 77.04% | 214.4 |
| heart-statlog | PFoil:single (macro distribution voting) | 76.30% | 214.4 |
| heart-statlog | PFoil:single (micro distribution voting) | 77.04% | 214.4 |
| heart-statlog | PFoil:single (voting) | 75.19% | 214.4 |
| heart-statlog | PFoil:two-level (best rule) | 82.59% | 214.4 |
| heart-statlog | PFoil:two-level (macro distribution voting) | 81.48% | 214.4 |
| heart-statlog | PFoil:two-level (micro distribution voting) | 81.48% | 214.4 |
| heart-statlog | PFoil:two-level (voting) | 78.52% | 214.4 |
| heart-statlog | PFossil:one-level (best rule) | 80.37% | 151.2 |
| heart-statlog | PFossil:one-level (macro distribution voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:one-level (micro distribution voting) | 81.85% | 151.2 |
| heart-statlog | PFossil:one-level (voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:single (best rule) | 80.74% | 151.2 |
| heart-statlog | PFossil:single (macro distribution voting) | 79.26% | 151.2 |
| heart-statlog | PFossil:single (micro distribution voting) | 80.00% | 151.2 |
| heart-statlog | PFossil:single (voting) | 79.26% | 151.2 |
| heart-statlog | PFossil:two-level (best rule) | 82.22% | 151.2 |
| heart-statlog | PFossil:two-level (macro distribution voting) | 79.63% | 151.2 |
| heart-statlog | PFossil:two-level (micro distribution voting) | 80.74% | 151.2 |
| heart-statlog | PFossil:two-level (voting) | 81.11% | 151.2 |
| heart-statlog | Pypper:one-level (best rule) | 79.63% | 56.1 |
| heart-statlog | Pypper:one-level (macro distribution voting) | 82.59% | 56.1 |
| heart-statlog | Pypper:one-level (micro distribution voting) | 81.48% | 56.1 |
| heart-statlog | Pypper:one-level (voting) | 83.33% | 56.1 |
| heart-statlog | Pypper:single (best rule) | 79.26% | 56.1 |
| heart-statlog | Pypper:single (macro distribution voting) | 79.63% | 56.1 |
| heart-statlog | Pypper:single (micro distribution voting) | 80.74% | 56.1 |
| heart-statlog | Pypper:single (voting) | 80.74% | 56.1 |
| heart-statlog | Pypper:two-level (best rule) | 82.22% | 56.1 |
| heart-statlog | Pypper:two-level (macro distribution voting) | 82.59% | 56.1 |
| heart-statlog | Pypper:two-level (micro distribution voting) | 81.11% | 56.1 |
| heart-statlog | Pypper:two-level (voting) | 82.22% | 56.1 |
| heloc | PFoil:one-level (best rule) | 71.16% | 5792.0 |
| heloc | PFoil:one-level (macro distribution voting) | 71.24% | 5792.0 |
| heloc | PFoil:one-level (micro distribution voting) | 70.13% | 5792.0 |
| heloc | PFoil:one-level (voting) | 70.13% | 5792.0 |
| heloc | PFoil:single (best rule) | 67.84% | 5792.0 |
| heloc | PFoil:single (macro distribution voting) | 68.04% | 5792.0 |
| heloc | PFoil:single (micro distribution voting) | 66.05% | 5792.0 |
| heloc | PFoil:single (voting) | 62.21% | 5792.0 |
| heloc | PFoil:two-level (best rule) | 71.06% | 5792.0 |
| heloc | PFoil:two-level (macro distribution voting) | 71.24% | 5792.0 |
| heloc | PFoil:two-level (micro distribution voting) | 70.81% | 5792.0 |
| heloc | PFoil:two-level (voting) | 69.86% | 5792.0 |
| heloc | PFossil:one-level (best rule) | 69.77% | 20.3 |
| heloc | PFossil:one-level (macro distribution voting) | 69.40% | 20.3 |
| heloc | PFossil:one-level (micro distribution voting) | 69.23% | 20.3 |
| heloc | PFossil:one-level (voting) | 69.26% | 20.3 |
| heloc | PFossil:single (best rule) | 68.83% | 20.3 |
| heloc | PFossil:single (macro distribution voting) | 68.83% | 20.3 |
| heloc | PFossil:single (micro distribution voting) | 68.83% | 20.3 |
| heloc | PFossil:single (voting) | 68.83% | 20.3 |
| heloc | PFossil:two-level (best rule) | 69.26% | 20.3 |
| heloc | PFossil:two-level (macro distribution voting) | 69.26% | 20.3 |
| heloc | PFossil:two-level (micro distribution voting) | 69.26% | 20.3 |
| heloc | PFossil:two-level (voting) | 69.26% | 20.3 |
| heloc | Pypper:one-level (best rule) | 71.42% | 167.9 |
| heloc | Pypper:one-level (macro distribution voting) | 71.36% | 167.9 |
| heloc | Pypper:one-level (micro distribution voting) | 70.90% | 167.9 |
| heloc | Pypper:one-level (voting) | 71.37% | 167.9 |
| heloc | Pypper:single (best rule) | 70.45% | 167.9 |
| heloc | Pypper:single (macro distribution voting) | 70.47% | 167.9 |
| heloc | Pypper:single (micro distribution voting) | 70.23% | 167.9 |
| heloc | Pypper:single (voting) | 70.26% | 167.9 |
| heloc | Pypper:two-level (best rule) | 71.44% | 167.9 |
| heloc | Pypper:two-level (macro distribution voting) | 71.47% | 167.9 |
| heloc | Pypper:two-level (micro distribution voting) | 71.10% | 167.9 |
| heloc | Pypper:two-level (voting) | 71.33% | 167.9 |
| hepatitis | PFoil:one-level (best rule) | 82.62% | 131.6 |
| hepatitis | PFoil:one-level (macro distribution voting) | 84.04% | 131.6 |
| hepatitis | PFoil:one-level (micro distribution voting) | 82.00% | 131.6 |
| hepatitis | PFoil:one-level (voting) | 84.75% | 131.6 |
| hepatitis | PFoil:single (best rule) | 77.58% | 131.6 |
| hepatitis | PFoil:single (macro distribution voting) | 77.54% | 131.6 |
| hepatitis | PFoil:single (micro distribution voting) | 79.46% | 131.6 |
| hepatitis | PFoil:single (voting) | 76.29% | 131.6 |
| hepatitis | PFoil:two-level (best rule) | 84.04% | 131.6 |
| hepatitis | PFoil:two-level (macro distribution voting) | 82.79% | 131.6 |
| hepatitis | PFoil:two-level (micro distribution voting) | 82.71% | 131.6 |
| hepatitis | PFoil:two-level (voting) | 85.29% | 131.6 |
| hepatitis | PFossil:one-level (best rule) | 80.08% | 112.1 |
| hepatitis | PFossil:one-level (macro distribution voting) | 81.46% | 112.1 |
| hepatitis | PFossil:one-level (micro distribution voting) | 81.96% | 112.1 |
| hepatitis | PFossil:one-level (voting) | 81.42% | 112.1 |
| hepatitis | PFossil:single (best rule) | 78.75% | 112.1 |
| hepatitis | PFossil:single (macro distribution voting) | 77.54% | 112.1 |
| hepatitis | PFossil:single (micro distribution voting) | 80.04% | 112.1 |
| hepatitis | PFossil:single (voting) | 78.75% | 112.1 |
| hepatitis | PFossil:two-level (best rule) | 82.67% | 112.1 |
| hepatitis | PFossil:two-level (macro distribution voting) | 80.83% | 112.1 |
| hepatitis | PFossil:two-level (micro distribution voting) | 82.67% | 112.1 |
| hepatitis | PFossil:two-level (voting) | 81.42% | 112.1 |
| hepatitis | Pypper:one-level (best rule) | 80.79% | 41.4 |
| hepatitis | Pypper:one-level (macro distribution voting) | 80.12% | 41.4 |
| hepatitis | Pypper:one-level (micro distribution voting) | 80.71% | 41.4 |
| hepatitis | Pypper:one-level (voting) | 81.42% | 41.4 |
| hepatitis | Pypper:single (best rule) | 80.04% | 41.4 |
| hepatitis | Pypper:single (macro distribution voting) | 80.67% | 41.4 |
| hepatitis | Pypper:single (micro distribution voting) | 80.04% | 41.4 |
| hepatitis | Pypper:single (voting) | 80.67% | 41.4 |
| hepatitis | Pypper:two-level (best rule) | 78.83% | 41.4 |
| hepatitis | Pypper:two-level (macro distribution voting) | 78.83% | 41.4 |
| hepatitis | Pypper:two-level (micro distribution voting) | 80.08% | 41.4 |
| hepatitis | Pypper:two-level (voting) | 80.71% | 41.4 |
| ilpd | PFoil:one-level (best rule) | 71.19% | 520.5 |
| ilpd | PFoil:one-level (macro distribution voting) | 72.03% | 520.5 |
| ilpd | PFoil:one-level (micro distribution voting) | 71.01% | 520.5 |
| ilpd | PFoil:one-level (voting) | 71.50% | 520.5 |
| ilpd | PFoil:single (best rule) | 68.94% | 520.5 |
| ilpd | PFoil:single (macro distribution voting) | 69.62% | 520.5 |
| ilpd | PFoil:single (micro distribution voting) | 69.12% | 520.5 |
| ilpd | PFoil:single (voting) | 68.42% | 520.5 |
| ilpd | PFoil:two-level (best rule) | 72.36% | 520.5 |
| ilpd | PFoil:two-level (macro distribution voting) | 72.19% | 520.5 |
| ilpd | PFoil:two-level (micro distribution voting) | 71.85% | 520.5 |
| ilpd | PFoil:two-level (voting) | 70.99% | 520.5 |
| ilpd | PFossil:one-level (best rule) | 70.33% | 83.8 |
| ilpd | PFossil:one-level (macro distribution voting) | 68.96% | 83.8 |
| ilpd | PFossil:one-level (micro distribution voting) | 70.32% | 83.8 |
| ilpd | PFossil:one-level (voting) | 66.18% | 83.8 |
| ilpd | PFossil:single (best rule) | 63.28% | 83.8 |
| ilpd | PFossil:single (macro distribution voting) | 63.28% | 83.8 |
| ilpd | PFossil:single (micro distribution voting) | 63.28% | 83.8 |
| ilpd | PFossil:single (voting) | 63.28% | 83.8 |
| ilpd | PFossil:two-level (best rule) | 66.35% | 83.8 |
| ilpd | PFossil:two-level (macro distribution voting) | 66.35% | 83.8 |
| ilpd | PFossil:two-level (micro distribution voting) | 66.69% | 83.8 |
| ilpd | PFossil:two-level (voting) | 66.01% | 83.8 |
| ilpd | Pypper:one-level (best rule) | 70.85% | 48.2 |
| ilpd | Pypper:one-level (macro distribution voting) | 70.85% | 48.2 |
| ilpd | Pypper:one-level (micro distribution voting) | 71.02% | 48.2 |
| ilpd | Pypper:one-level (voting) | 68.62% | 48.2 |
| ilpd | Pypper:single (best rule) | 69.66% | 48.2 |
| ilpd | Pypper:single (macro distribution voting) | 69.66% | 48.2 |
| ilpd | Pypper:single (micro distribution voting) | 69.66% | 48.2 |
| ilpd | Pypper:single (voting) | 69.66% | 48.2 |
| ilpd | Pypper:two-level (best rule) | 68.79% | 48.2 |
| ilpd | Pypper:two-level (macro distribution voting) | 68.96% | 48.2 |
| ilpd | Pypper:two-level (micro distribution voting) | 68.27% | 48.2 |
| ilpd | Pypper:two-level (voting) | 67.93% | 48.2 |
| ionosphere | PFoil:one-level (best rule) | 80.06% | 140.7 |
| ionosphere | PFoil:one-level (macro distribution voting) | 90.03% | 140.7 |
| ionosphere | PFoil:one-level (micro distribution voting) | 90.59% | 140.7 |
| ionosphere | PFoil:one-level (voting) | 90.32% | 140.7 |
| ionosphere | PFoil:single (best rule) | 90.87% | 140.7 |
| ionosphere | PFoil:single (macro distribution voting) | 92.02% | 140.7 |
| ionosphere | PFoil:single (micro distribution voting) | 91.17% | 140.7 |
| ionosphere | PFoil:single (voting) | 92.30% | 140.7 |
| ionosphere | PFoil:two-level (best rule) | 90.31% | 140.7 |
| ionosphere | PFoil:two-level (macro distribution voting) | 90.31% | 140.7 |
| ionosphere | PFoil:two-level (micro distribution voting) | 89.75% | 140.7 |
| ionosphere | PFoil:two-level (voting) | 90.60% | 140.7 |
| ionosphere | PFossil:one-level (best rule) | 89.44% | 115.2 |
| ionosphere | PFossil:one-level (macro distribution voting) | 90.88% | 115.2 |
| ionosphere | PFossil:one-level (micro distribution voting) | 92.30% | 115.2 |
| ionosphere | PFossil:one-level (voting) | 90.60% | 115.2 |
| ionosphere | PFossil:single (best rule) | 89.75% | 115.2 |
| ionosphere | PFossil:single (macro distribution voting) | 90.03% | 115.2 |
| ionosphere | PFossil:single (micro distribution voting) | 89.75% | 115.2 |
| ionosphere | PFossil:single (voting) | 90.03% | 115.2 |
| ionosphere | PFossil:two-level (best rule) | 92.58% | 115.2 |
| ionosphere | PFossil:two-level (macro distribution voting) | 92.58% | 115.2 |
| ionosphere | PFossil:two-level (micro distribution voting) | 91.17% | 115.2 |
| ionosphere | PFossil:two-level (voting) | 92.87% | 115.2 |
| ionosphere | Pypper:one-level (best rule) | 87.75% | 55.3 |
| ionosphere | Pypper:one-level (macro distribution voting) | 89.17% | 55.3 |
| ionosphere | Pypper:one-level (micro distribution voting) | 88.60% | 55.3 |
| ionosphere | Pypper:one-level (voting) | 88.89% | 55.3 |
| ionosphere | Pypper:single (best rule) | 90.61% | 55.3 |
| ionosphere | Pypper:single (macro distribution voting) | 90.61% | 55.3 |
| ionosphere | Pypper:single (micro distribution voting) | 91.17% | 55.3 |
| ionosphere | Pypper:single (voting) | 90.89% | 55.3 |
| ionosphere | Pypper:two-level (best rule) | 89.45% | 55.3 |
| ionosphere | Pypper:two-level (macro distribution voting) | 90.02% | 55.3 |
| ionosphere | Pypper:two-level (micro distribution voting) | 88.88% | 55.3 |
| ionosphere | Pypper:two-level (voting) | 89.45% | 55.3 |
| iris | PFoil:one-level (best rule) | 94.00% | 69.4 |
| iris | PFoil:one-level (macro distribution voting) | 92.00% | 69.4 |
| iris | PFoil:one-level (micro distribution voting) | 93.33% | 69.4 |
| iris | PFoil:one-level (voting) | 92.00% | 69.4 |
| iris | PFoil:single (best rule) | 92.67% | 69.4 |
| iris | PFoil:single (macro distribution voting) | 92.67% | 69.4 |
| iris | PFoil:single (micro distribution voting) | 92.67% | 69.4 |
| iris | PFoil:single (voting) | 90.67% | 69.4 |
| iris | PFoil:two-level (best rule) | 92.67% | 69.4 |
| iris | PFoil:two-level (macro distribution voting) | 92.00% | 69.4 |
| iris | PFoil:two-level (micro distribution voting) | 92.67% | 69.4 |
| iris | PFoil:two-level (voting) | 92.00% | 69.4 |
| iris | PFossil:one-level (best rule) | 92.67% | 54.4 |
| iris | PFossil:one-level (macro distribution voting) | 94.00% | 54.4 |
| iris | PFossil:one-level (micro distribution voting) | 93.33% | 54.4 |
| iris | PFossil:one-level (voting) | 93.33% | 54.4 |
| iris | PFossil:single (best rule) | 92.00% | 54.4 |
| iris | PFossil:single (macro distribution voting) | 92.00% | 54.4 |
| iris | PFossil:single (micro distribution voting) | 92.00% | 54.4 |
| iris | PFossil:single (voting) | 91.33% | 54.4 |
| iris | PFossil:two-level (best rule) | 93.33% | 54.4 |
| iris | PFossil:two-level (macro distribution voting) | 94.00% | 54.4 |
| iris | PFossil:two-level (micro distribution voting) | 93.33% | 54.4 |
| iris | PFossil:two-level (voting) | 93.33% | 54.4 |
| iris | Pypper:one-level (best rule) | 94.67% | 35.4 |
| iris | Pypper:one-level (macro distribution voting) | 94.00% | 35.4 |
| iris | Pypper:one-level (micro distribution voting) | 94.00% | 35.4 |
| iris | Pypper:one-level (voting) | 94.00% | 35.4 |
| iris | Pypper:single (best rule) | 92.00% | 35.4 |
| iris | Pypper:single (macro distribution voting) | 92.00% | 35.4 |
| iris | Pypper:single (micro distribution voting) | 92.00% | 35.4 |
| iris | Pypper:single (voting) | 92.00% | 35.4 |
| iris | Pypper:two-level (best rule) | 94.67% | 35.4 |
| iris | Pypper:two-level (macro distribution voting) | 94.67% | 35.4 |
| iris | Pypper:two-level (micro distribution voting) | 94.67% | 35.4 |
| iris | Pypper:two-level (voting) | 94.67% | 35.4 |
| kc1 | PFoil:one-level (best rule) | 85.63% | 1143.6 |
| kc1 | PFoil:one-level (macro distribution voting) | 85.97% | 1143.6 |
| kc1 | PFoil:one-level (micro distribution voting) | 84.83% | 1143.6 |
| kc1 | PFoil:one-level (voting) | 84.45% | 1143.6 |
| kc1 | PFoil:single (best rule) | 85.87% | 1143.6 |
| kc1 | PFoil:single (macro distribution voting) | 86.06% | 1143.6 |
| kc1 | PFoil:single (micro distribution voting) | 85.87% | 1143.6 |
| kc1 | PFoil:single (voting) | 83.45% | 1143.6 |
| kc1 | PFoil:two-level (best rule) | 85.87% | 1143.6 |
| kc1 | PFoil:two-level (macro distribution voting) | 85.82% | 1143.6 |
| kc1 | PFoil:two-level (micro distribution voting) | 84.97% | 1143.6 |
| kc1 | PFoil:two-level (voting) | 85.25% | 1143.6 |
| kc1 | PFossil:one-level (best rule) | 85.16% | 50.3 |
| kc1 | PFossil:one-level (macro distribution voting) | 84.92% | 50.3 |
| kc1 | PFossil:one-level (micro distribution voting) | 84.92% | 50.3 |
| kc1 | PFossil:one-level (voting) | 82.27% | 50.3 |
| kc1 | PFossil:single (best rule) | 83.31% | 50.3 |
| kc1 | PFossil:single (macro distribution voting) | 83.31% | 50.3 |
| kc1 | PFossil:single (micro distribution voting) | 83.31% | 50.3 |
| kc1 | PFossil:single (voting) | 83.31% | 50.3 |
| kc1 | PFossil:two-level (best rule) | 84.97% | 50.3 |
| kc1 | PFossil:two-level (macro distribution voting) | 84.97% | 50.3 |
| kc1 | PFossil:two-level (micro distribution voting) | 84.97% | 50.3 |
| kc1 | PFossil:two-level (voting) | 84.97% | 50.3 |
| kc1 | Pypper:one-level (best rule) | 85.30% | 56.2 |
| kc1 | Pypper:one-level (macro distribution voting) | 85.35% | 56.2 |
| kc1 | Pypper:one-level (micro distribution voting) | 85.02% | 56.2 |
| kc1 | Pypper:one-level (voting) | 85.02% | 56.2 |
| kc1 | Pypper:single (best rule) | 84.64% | 56.2 |
| kc1 | Pypper:single (macro distribution voting) | 84.64% | 56.2 |
| kc1 | Pypper:single (micro distribution voting) | 84.64% | 56.2 |
| kc1 | Pypper:single (voting) | 84.64% | 56.2 |
| kc1 | Pypper:two-level (best rule) | 84.97% | 56.2 |
| kc1 | Pypper:two-level (macro distribution voting) | 84.97% | 56.2 |
| kc1 | Pypper:two-level (micro distribution voting) | 84.92% | 56.2 |
| kc1 | Pypper:two-level (voting) | 84.92% | 56.2 |
| kc2 | PFoil:one-level (best rule) | 84.26% | 346.2 |
| kc2 | PFoil:one-level (macro distribution voting) | 84.66% | 346.2 |
| kc2 | PFoil:one-level (micro distribution voting) | 83.70% | 346.2 |
| kc2 | PFoil:one-level (voting) | 83.70% | 346.2 |
| kc2 | PFoil:single (best rule) | 82.55% | 346.2 |
| kc2 | PFoil:single (macro distribution voting) | 81.98% | 346.2 |
| kc2 | PFoil:single (micro distribution voting) | 81.98% | 346.2 |
| kc2 | PFoil:single (voting) | 81.41% | 346.2 |
| kc2 | PFoil:two-level (best rule) | 84.46% | 346.2 |
| kc2 | PFoil:two-level (macro distribution voting) | 84.47% | 346.2 |
| kc2 | PFoil:two-level (micro distribution voting) | 84.08% | 346.2 |
| kc2 | PFoil:two-level (voting) | 84.27% | 346.2 |
| kc2 | PFossil:one-level (best rule) | 82.74% | 102.5 |
| kc2 | PFossil:one-level (macro distribution voting) | 83.89% | 102.5 |
| kc2 | PFossil:one-level (micro distribution voting) | 83.89% | 102.5 |
| kc2 | PFossil:one-level (voting) | 83.70% | 102.5 |
| kc2 | PFossil:single (best rule) | 82.74% | 102.5 |
| kc2 | PFossil:single (macro distribution voting) | 82.35% | 102.5 |
| kc2 | PFossil:single (micro distribution voting) | 81.79% | 102.5 |
| kc2 | PFossil:single (voting) | 82.55% | 102.5 |
| kc2 | PFossil:two-level (best rule) | 83.12% | 102.5 |
| kc2 | PFossil:two-level (macro distribution voting) | 82.93% | 102.5 |
| kc2 | PFossil:two-level (micro distribution voting) | 82.94% | 102.5 |
| kc2 | PFossil:two-level (voting) | 83.32% | 102.5 |
| kc2 | Pypper:one-level (best rule) | 85.22% | 41.6 |
| kc2 | Pypper:one-level (macro distribution voting) | 84.07% | 41.6 |
| kc2 | Pypper:one-level (micro distribution voting) | 85.61% | 41.6 |
| kc2 | Pypper:one-level (voting) | 83.50% | 41.6 |
| kc2 | Pypper:single (best rule) | 83.69% | 41.6 |
| kc2 | Pypper:single (macro distribution voting) | 83.69% | 41.6 |
| kc2 | Pypper:single (micro distribution voting) | 83.69% | 41.6 |
| kc2 | Pypper:single (voting) | 83.69% | 41.6 |
| kc2 | Pypper:two-level (best rule) | 82.93% | 41.6 |
| kc2 | Pypper:two-level (macro distribution voting) | 82.93% | 41.6 |
| kc2 | Pypper:two-level (micro distribution voting) | 83.12% | 41.6 |
| kc2 | Pypper:two-level (voting) | 82.93% | 41.6 |
| kr-vs-kp | PFoil:one-level (best rule) | 99.09% | 367.5 |
| kr-vs-kp | PFoil:one-level (macro distribution voting) | 99.53% | 367.5 |
| kr-vs-kp | PFoil:one-level (micro distribution voting) | 98.87% | 367.5 |
| kr-vs-kp | PFoil:one-level (voting) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:single (best rule) | 99.34% | 367.5 |
| kr-vs-kp | PFoil:single (macro distribution voting) | 99.22% | 367.5 |
| kr-vs-kp | PFoil:single (micro distribution voting) | 99.09% | 367.5 |
| kr-vs-kp | PFoil:single (voting) | 98.84% | 367.5 |
| kr-vs-kp | PFoil:two-level (best rule) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:two-level (macro distribution voting) | 99.47% | 367.5 |
| kr-vs-kp | PFoil:two-level (micro distribution voting) | 99.34% | 367.5 |
| kr-vs-kp | PFoil:two-level (voting) | 99.28% | 367.5 |
| kr-vs-kp | PFossil:one-level (best rule) | 99.12% | 141.0 |
| kr-vs-kp | PFossil:one-level (macro distribution voting) | 98.87% | 141.0 |
| kr-vs-kp | PFossil:one-level (micro distribution voting) | 94.31% | 141.0 |
| kr-vs-kp | PFossil:one-level (voting) | 96.21% | 141.0 |
| kr-vs-kp | PFossil:single (best rule) | 99.37% | 141.0 |
| kr-vs-kp | PFossil:single (macro distribution voting) | 99.41% | 141.0 |
| kr-vs-kp | PFossil:single (micro distribution voting) | 94.31% | 141.0 |
| kr-vs-kp | PFossil:single (voting) | 96.43% | 141.0 |
| kr-vs-kp | PFossil:two-level (best rule) | 99.31% | 141.0 |
| kr-vs-kp | PFossil:two-level (macro distribution voting) | 99.41% | 141.0 |
| kr-vs-kp | PFossil:two-level (micro distribution voting) | 94.31% | 141.0 |
| kr-vs-kp | PFossil:two-level (voting) | 96.46% | 141.0 |
| kr-vs-kp | Pypper:one-level (best rule) | 98.65% | 183.4 |
| kr-vs-kp | Pypper:one-level (macro distribution voting) | 99.19% | 183.4 |
| kr-vs-kp | Pypper:one-level (micro distribution voting) | 98.28% | 183.4 |
| kr-vs-kp | Pypper:one-level (voting) | 99.06% | 183.4 |
| kr-vs-kp | Pypper:single (best rule) | 99.16% | 183.4 |
| kr-vs-kp | Pypper:single (macro distribution voting) | 99.12% | 183.4 |
| kr-vs-kp | Pypper:single (micro distribution voting) | 99.03% | 183.4 |
| kr-vs-kp | Pypper:single (voting) | 98.91% | 183.4 |
| kr-vs-kp | Pypper:two-level (best rule) | 99.06% | 183.4 |
| kr-vs-kp | Pypper:two-level (macro distribution voting) | 99.06% | 183.4 |
| kr-vs-kp | Pypper:two-level (micro distribution voting) | 98.84% | 183.4 |
| kr-vs-kp | Pypper:two-level (voting) | 98.87% | 183.4 |
| led24 | PFoil:one-level (best rule) | 70.97% | 3316.9 |
| led24 | PFoil:one-level (macro distribution voting) | 71.16% | 3316.9 |
| led24 | PFoil:one-level (micro distribution voting) | 71.50% | 3316.9 |
| led24 | PFoil:one-level (voting) | 69.91% | 3316.9 |
| led24 | PFoil:single (best rule) | 61.81% | 3316.9 |
| led24 | PFoil:single (macro distribution voting) | 62.34% | 3316.9 |
| led24 | PFoil:single (micro distribution voting) | 61.97% | 3316.9 |
| led24 | PFoil:single (voting) | 56.84% | 3316.9 |
| led24 | PFoil:two-level (best rule) | 70.47% | 3316.9 |
| led24 | PFoil:two-level (macro distribution voting) | 70.28% | 3316.9 |
| led24 | PFoil:two-level (micro distribution voting) | 70.31% | 3316.9 |
| led24 | PFoil:two-level (voting) | 68.66% | 3316.9 |
| led24 | PFossil:one-level (best rule) | 60.53% | 742.2 |
| led24 | PFossil:one-level (macro distribution voting) | 71.78% | 742.2 |
| led24 | PFossil:one-level (micro distribution voting) | 71.09% | 742.2 |
| led24 | PFossil:one-level (voting) | 71.75% | 742.2 |
| led24 | PFossil:single (best rule) | 70.00% | 742.2 |
| led24 | PFossil:single (macro distribution voting) | 70.28% | 742.2 |
| led24 | PFossil:single (micro distribution voting) | 70.56% | 742.2 |
| led24 | PFossil:single (voting) | 70.34% | 742.2 |
| led24 | PFossil:two-level (best rule) | 71.34% | 742.2 |
| led24 | PFossil:two-level (macro distribution voting) | 71.41% | 742.2 |
| led24 | PFossil:two-level (micro distribution voting) | 70.97% | 742.2 |
| led24 | PFossil:two-level (voting) | 70.94% | 742.2 |
| led24 | Pypper:one-level (best rule) | 69.94% | 241.4 |
| led24 | Pypper:one-level (macro distribution voting) | 72.06% | 241.4 |
| led24 | Pypper:one-level (micro distribution voting) | 72.41% | 241.4 |
| led24 | Pypper:one-level (voting) | 72.03% | 241.4 |
| led24 | Pypper:single (best rule) | 68.00% | 241.4 |
| led24 | Pypper:single (macro distribution voting) | 67.81% | 241.4 |
| led24 | Pypper:single (micro distribution voting) | 67.88% | 241.4 |
| led24 | Pypper:single (voting) | 67.47% | 241.4 |
| led24 | Pypper:two-level (best rule) | 71.94% | 241.4 |
| led24 | Pypper:two-level (macro distribution voting) | 72.03% | 241.4 |
| led24 | Pypper:two-level (micro distribution voting) | 72.25% | 241.4 |
| led24 | Pypper:two-level (voting) | 71.78% | 241.4 |
| letter | PFoil:one-level (best rule) | 87.52% | 9440.8 |
| letter | PFoil:one-level (macro distribution voting) | 93.27% | 9440.8 |
| letter | PFoil:one-level (micro distribution voting) | 81.04% | 9440.8 |
| letter | PFoil:one-level (voting) | 92.35% | 9440.8 |
| letter | PFoil:single (best rule) | 88.78% | 9440.8 |
| letter | PFoil:single (macro distribution voting) | 89.12% | 9440.8 |
| letter | PFoil:single (micro distribution voting) | 72.10% | 9440.8 |
| letter | PFoil:single (voting) | 80.75% | 9440.8 |
| letter | PFoil:two-level (best rule) | 92.56% | 9440.8 |
| letter | PFoil:two-level (macro distribution voting) | 92.89% | 9440.8 |
| letter | PFoil:two-level (micro distribution voting) | 89.34% | 9440.8 |
| letter | PFoil:two-level (voting) | 90.38% | 9440.8 |
| letter | PFossil:one-level (best rule) | 85.49% | 1583.4 |
| letter | PFossil:one-level (macro distribution voting) | 86.37% | 1583.4 |
| letter | PFossil:one-level (micro distribution voting) | 81.27% | 1583.4 |
| letter | PFossil:one-level (voting) | 85.15% | 1583.4 |
| letter | PFossil:single (best rule) | 76.22% | 1583.4 |
| letter | PFossil:single (macro distribution voting) | 76.44% | 1583.4 |
| letter | PFossil:single (micro distribution voting) | 74.39% | 1583.4 |
| letter | PFossil:single (voting) | 75.29% | 1583.4 |
| letter | PFossil:two-level (best rule) | 85.63% | 1583.4 |
| letter | PFossil:two-level (macro distribution voting) | 85.99% | 1583.4 |
| letter | PFossil:two-level (micro distribution voting) | 83.31% | 1583.4 |
| letter | PFossil:two-level (voting) | 84.73% | 1583.4 |
| letter | Pypper:one-level (best rule) | 85.62% | 2092.3 |
| letter | Pypper:one-level (macro distribution voting) | 87.48% | 2092.3 |
| letter | Pypper:one-level (micro distribution voting) | 85.15% | 2092.3 |
| letter | Pypper:one-level (voting) | 87.48% | 2092.3 |
| letter | Pypper:single (best rule) | 77.53% | 2092.3 |
| letter | Pypper:single (macro distribution voting) | 77.68% | 2092.3 |
| letter | Pypper:single (micro distribution voting) | 77.17% | 2092.3 |
| letter | Pypper:single (voting) | 77.28% | 2092.3 |
| letter | Pypper:two-level (best rule) | 87.26% | 2092.3 |
| letter | Pypper:two-level (macro distribution voting) | 87.38% | 2092.3 |
| letter | Pypper:two-level (micro distribution voting) | 86.86% | 2092.3 |
| letter | Pypper:two-level (voting) | 87.33% | 2092.3 |
| lymph | PFoil:one-level (best rule) | 78.52% | 140.2 |
| lymph | PFoil:one-level (macro distribution voting) | 82.57% | 140.2 |
| lymph | PFoil:one-level (micro distribution voting) | 80.52% | 140.2 |
| lymph | PFoil:one-level (voting) | 81.24% | 140.2 |
| lymph | PFoil:single (best rule) | 82.57% | 140.2 |
| lymph | PFoil:single (macro distribution voting) | 81.19% | 140.2 |
| lymph | PFoil:single (micro distribution voting) | 77.86% | 140.2 |
| lymph | PFoil:single (voting) | 78.48% | 140.2 |
| lymph | PFoil:two-level (best rule) | 82.52% | 140.2 |
| lymph | PFoil:two-level (macro distribution voting) | 82.52% | 140.2 |
| lymph | PFoil:two-level (micro distribution voting) | 81.86% | 140.2 |
| lymph | PFoil:two-level (voting) | 82.52% | 140.2 |
| lymph | PFossil:one-level (best rule) | 78.52% | 124.5 |
| lymph | PFossil:one-level (macro distribution voting) | 83.81% | 124.5 |
| lymph | PFossil:one-level (micro distribution voting) | 78.52% | 124.5 |
| lymph | PFossil:one-level (voting) | 82.48% | 124.5 |
| lymph | PFossil:single (best rule) | 81.71% | 124.5 |
| lymph | PFossil:single (macro distribution voting) | 79.71% | 124.5 |
| lymph | PFossil:single (micro distribution voting) | 81.05% | 124.5 |
| lymph | PFossil:single (voting) | 81.10% | 124.5 |
| lymph | PFossil:two-level (best rule) | 80.48% | 124.5 |
| lymph | PFossil:two-level (macro distribution voting) | 82.43% | 124.5 |
| lymph | PFossil:two-level (micro distribution voting) | 81.19% | 124.5 |
| lymph | PFossil:two-level (voting) | 81.19% | 124.5 |
| lymph | Pypper:one-level (best rule) | 81.86% | 50.6 |
| lymph | Pypper:one-level (macro distribution voting) | 81.86% | 50.6 |
| lymph | Pypper:one-level (micro distribution voting) | 80.48% | 50.6 |
| lymph | Pypper:one-level (voting) | 80.48% | 50.6 |
| lymph | Pypper:single (best rule) | 80.52% | 50.6 |
| lymph | Pypper:single (macro distribution voting) | 82.52% | 50.6 |
| lymph | Pypper:single (micro distribution voting) | 77.81% | 50.6 |
| lymph | Pypper:single (voting) | 78.52% | 50.6 |
| lymph | Pypper:two-level (best rule) | 79.86% | 50.6 |
| lymph | Pypper:two-level (macro distribution voting) | 80.52% | 50.6 |
| lymph | Pypper:two-level (micro distribution voting) | 79.81% | 50.6 |
| lymph | Pypper:two-level (voting) | 78.48% | 50.6 |
| mofn-3-7-10 | PFoil:one-level (best rule) | 89.80% | 615.6 |
| mofn-3-7-10 | PFoil:one-level (macro distribution voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:one-level (micro distribution voting) | 98.26% | 615.6 |
| mofn-3-7-10 | PFoil:one-level (voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:single (best rule) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:single (macro distribution voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:single (micro distribution voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:single (voting) | 100.00% | 615.6 |
| mofn-3-7-10 | PFoil:two-level (best rule) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level (macro distribution voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level (micro distribution voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFoil:two-level (voting) | 99.77% | 615.6 |
| mofn-3-7-10 | PFossil:one-level (best rule) | 89.50% | 155.9 |
| mofn-3-7-10 | PFossil:one-level (macro distribution voting) | 97.96% | 155.9 |
| mofn-3-7-10 | PFossil:one-level (micro distribution voting) | 83.68% | 155.9 |
| mofn-3-7-10 | PFossil:one-level (voting) | 99.62% | 155.9 |
| mofn-3-7-10 | PFossil:single (best rule) | 83.68% | 155.9 |
| mofn-3-7-10 | PFossil:single (macro distribution voting) | 83.76% | 155.9 |
| mofn-3-7-10 | PFossil:single (micro distribution voting) | 83.31% | 155.9 |
| mofn-3-7-10 | PFossil:single (voting) | 83.61% | 155.9 |
| mofn-3-7-10 | PFossil:two-level (best rule) | 95.16% | 155.9 |
| mofn-3-7-10 | PFossil:two-level (macro distribution voting) | 96.45% | 155.9 |
| mofn-3-7-10 | PFossil:two-level (micro distribution voting) | 89.19% | 155.9 |
| mofn-3-7-10 | PFossil:two-level (voting) | 91.61% | 155.9 |
| mofn-3-7-10 | Pypper:one-level (best rule) | 87.08% | 208.9 |
| mofn-3-7-10 | Pypper:one-level (macro distribution voting) | 98.34% | 208.9 |
| mofn-3-7-10 | Pypper:one-level (micro distribution voting) | 85.72% | 208.9 |
| mofn-3-7-10 | Pypper:one-level (voting) | 98.41% | 208.9 |
| mofn-3-7-10 | Pypper:single (best rule) | 97.21% | 208.9 |
| mofn-3-7-10 | Pypper:single (macro distribution voting) | 97.66% | 208.9 |
| mofn-3-7-10 | Pypper:single (micro distribution voting) | 95.85% | 208.9 |
| mofn-3-7-10 | Pypper:single (voting) | 96.00% | 208.9 |
| mofn-3-7-10 | Pypper:two-level (best rule) | 95.62% | 208.9 |
| mofn-3-7-10 | Pypper:two-level (macro distribution voting) | 97.05% | 208.9 |
| mofn-3-7-10 | Pypper:two-level (micro distribution voting) | 93.20% | 208.9 |
| mofn-3-7-10 | Pypper:two-level (voting) | 93.80% | 208.9 |
| molecular-biology_promoters | PFoil:one-level (best rule) | 82.09% | 69.6 |
| molecular-biology_promoters | PFoil:one-level (macro distribution voting) | 90.64% | 69.6 |
| molecular-biology_promoters | PFoil:one-level (micro distribution voting) | 87.82% | 69.6 |
| molecular-biology_promoters | PFoil:one-level (voting) | 90.64% | 69.6 |
| molecular-biology_promoters | PFoil:single (best rule) | 86.09% | 69.6 |
| molecular-biology_promoters | PFoil:single (macro distribution voting) | 85.18% | 69.6 |
| molecular-biology_promoters | PFoil:single (micro distribution voting) | 86.09% | 69.6 |
| molecular-biology_promoters | PFoil:single (voting) | 83.36% | 69.6 |
| molecular-biology_promoters | PFoil:two-level (best rule) | 86.73% | 69.6 |
| molecular-biology_promoters | PFoil:two-level (macro distribution voting) | 87.82% | 69.6 |
| molecular-biology_promoters | PFoil:two-level (micro distribution voting) | 86.73% | 69.6 |
| molecular-biology_promoters | PFoil:two-level (voting) | 85.91% | 69.6 |
| molecular-biology_promoters | PFossil:one-level (best rule) | 80.09% | 67.6 |
| molecular-biology_promoters | PFossil:one-level (macro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:one-level (micro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:one-level (voting) | 87.82% | 67.6 |
| molecular-biology_promoters | PFossil:single (best rule) | 82.27% | 67.6 |
| molecular-biology_promoters | PFossil:single (macro distribution voting) | 82.18% | 67.6 |
| molecular-biology_promoters | PFossil:single (micro distribution voting) | 81.36% | 67.6 |
| molecular-biology_promoters | PFossil:single (voting) | 81.27% | 67.6 |
| molecular-biology_promoters | PFossil:two-level (best rule) | 87.82% | 67.6 |
| molecular-biology_promoters | PFossil:two-level (macro distribution voting) | 89.64% | 67.6 |
| molecular-biology_promoters | PFossil:two-level (micro distribution voting) | 83.91% | 67.6 |
| molecular-biology_promoters | PFossil:two-level (voting) | 86.00% | 67.6 |
| molecular-biology_promoters | Pypper:one-level (best rule) | 85.00% | 38.6 |
| molecular-biology_promoters | Pypper:one-level (macro distribution voting) | 84.64% | 38.6 |
| molecular-biology_promoters | Pypper:one-level (micro distribution voting) | 84.73% | 38.6 |
| molecular-biology_promoters | Pypper:one-level (voting) | 83.73% | 38.6 |
| molecular-biology_promoters | Pypper:single (best rule) | 81.36% | 38.6 |
| molecular-biology_promoters | Pypper:single (macro distribution voting) | 81.36% | 38.6 |
| molecular-biology_promoters | Pypper:single (micro distribution voting) | 82.18% | 38.6 |
| molecular-biology_promoters | Pypper:single (voting) | 81.27% | 38.6 |
| molecular-biology_promoters | Pypper:two-level (best rule) | 83.82% | 38.6 |
| molecular-biology_promoters | Pypper:two-level (macro distribution voting) | 83.82% | 38.6 |
| molecular-biology_promoters | Pypper:two-level (micro distribution voting) | 82.91% | 38.6 |
| molecular-biology_promoters | Pypper:two-level (voting) | 83.82% | 38.6 |
| monks-problems-1 | PFoil:one-level (best rule) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level (macro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level (micro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:one-level (voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:single (best rule) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:single (macro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:single (micro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:single (voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level (best rule) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level (macro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level (micro distribution voting) | 100.00% | 127.4 |
| monks-problems-1 | PFoil:two-level (voting) | 100.00% | 127.4 |
| monks-problems-1 | PFossil:one-level (best rule) | 86.36% | 32.5 |
| monks-problems-1 | PFossil:one-level (macro distribution voting) | 83.81% | 32.5 |
| monks-problems-1 | PFossil:one-level (micro distribution voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:one-level (voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:single (best rule) | 80.41% | 32.5 |
| monks-problems-1 | PFossil:single (macro distribution voting) | 80.41% | 32.5 |
| monks-problems-1 | PFossil:single (micro distribution voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:single (voting) | 75.91% | 32.5 |
| monks-problems-1 | PFossil:two-level (best rule) | 79.49% | 32.5 |
| monks-problems-1 | PFossil:two-level (macro distribution voting) | 79.85% | 32.5 |
| monks-problems-1 | PFossil:two-level (micro distribution voting) | 74.64% | 32.5 |
| monks-problems-1 | PFossil:two-level (voting) | 80.05% | 32.5 |
| monks-problems-1 | Pypper:one-level (best rule) | 100.00% | 63.4 |
| monks-problems-1 | Pypper:one-level (macro distribution voting) | 100.00% | 63.4 |
| monks-problems-1 | Pypper:one-level (micro distribution voting) | 91.17% | 63.4 |
| monks-problems-1 | Pypper:one-level (voting) | 100.00% | 63.4 |
| monks-problems-1 | Pypper:single (best rule) | 94.76% | 63.4 |
| monks-problems-1 | Pypper:single (macro distribution voting) | 94.76% | 63.4 |
| monks-problems-1 | Pypper:single (micro distribution voting) | 92.06% | 63.4 |
| monks-problems-1 | Pypper:single (voting) | 92.06% | 63.4 |
| monks-problems-1 | Pypper:two-level (best rule) | 100.00% | 63.4 |
| monks-problems-1 | Pypper:two-level (macro distribution voting) | 100.00% | 63.4 |
| monks-problems-1 | Pypper:two-level (micro distribution voting) | 95.86% | 63.4 |
| monks-problems-1 | Pypper:two-level (voting) | 100.00% | 63.4 |
| monks-problems-2 | PFoil:one-level (best rule) | 81.86% | 616.5 |
| monks-problems-2 | PFoil:one-level (macro distribution voting) | 90.51% | 616.5 |
| monks-problems-2 | PFoil:one-level (micro distribution voting) | 80.86% | 616.5 |
| monks-problems-2 | PFoil:one-level (voting) | 92.17% | 616.5 |
| monks-problems-2 | PFoil:single (best rule) | 97.50% | 616.5 |
| monks-problems-2 | PFoil:single (macro distribution voting) | 97.17% | 616.5 |
| monks-problems-2 | PFoil:single (micro distribution voting) | 97.50% | 616.5 |
| monks-problems-2 | PFoil:single (voting) | 97.34% | 616.5 |
| monks-problems-2 | PFoil:two-level (best rule) | 95.51% | 616.5 |
| monks-problems-2 | PFoil:two-level (macro distribution voting) | 94.68% | 616.5 |
| monks-problems-2 | PFoil:two-level (micro distribution voting) | 93.35% | 616.5 |
| monks-problems-2 | PFoil:two-level (voting) | 91.84% | 616.5 |
| monks-problems-2 | PFossil:one-level (best rule) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level (macro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level (micro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:one-level (voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:single (best rule) | 66.72% | 1.7 |
| monks-problems-2 | PFossil:single (macro distribution voting) | 66.72% | 1.7 |
| monks-problems-2 | PFossil:single (micro distribution voting) | 66.72% | 1.7 |
| monks-problems-2 | PFossil:single (voting) | 66.72% | 1.7 |
| monks-problems-2 | PFossil:two-level (best rule) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level (macro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level (micro distribution voting) | 66.06% | 1.7 |
| monks-problems-2 | PFossil:two-level (voting) | 66.06% | 1.7 |
| monks-problems-2 | Pypper:one-level (best rule) | 75.71% | 74.1 |
| monks-problems-2 | Pypper:one-level (macro distribution voting) | 71.88% | 74.1 |
| monks-problems-2 | Pypper:one-level (micro distribution voting) | 67.72% | 74.1 |
| monks-problems-2 | Pypper:one-level (voting) | 68.22% | 74.1 |
| monks-problems-2 | Pypper:single (best rule) | 72.03% | 74.1 |
| monks-problems-2 | Pypper:single (macro distribution voting) | 72.03% | 74.1 |
| monks-problems-2 | Pypper:single (micro distribution voting) | 71.87% | 74.1 |
| monks-problems-2 | Pypper:single (voting) | 71.87% | 74.1 |
| monks-problems-2 | Pypper:two-level (best rule) | 69.89% | 74.1 |
| monks-problems-2 | Pypper:two-level (macro distribution voting) | 69.72% | 74.1 |
| monks-problems-2 | Pypper:two-level (micro distribution voting) | 68.23% | 74.1 |
| monks-problems-2 | Pypper:two-level (voting) | 68.23% | 74.1 |
| monks-problems-3 | PFoil:one-level (best rule) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level (macro distribution voting) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level (micro distribution voting) | 98.92% | 162.4 |
| monks-problems-3 | PFoil:one-level (voting) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:single (best rule) | 97.66% | 162.4 |
| monks-problems-3 | PFoil:single (macro distribution voting) | 97.66% | 162.4 |
| monks-problems-3 | PFoil:single (micro distribution voting) | 97.66% | 162.4 |
| monks-problems-3 | PFoil:single (voting) | 96.76% | 162.4 |
| monks-problems-3 | PFoil:two-level (best rule) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:two-level (macro distribution voting) | 98.02% | 162.4 |
| monks-problems-3 | PFoil:two-level (micro distribution voting) | 98.20% | 162.4 |
| monks-problems-3 | PFoil:two-level (voting) | 97.84% | 162.4 |
| monks-problems-3 | PFossil:one-level (best rule) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:one-level (macro distribution voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:one-level (micro distribution voting) | 96.40% | 64.8 |
| monks-problems-3 | PFossil:one-level (voting) | 97.84% | 64.8 |
| monks-problems-3 | PFossil:single (best rule) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:single (macro distribution voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:single (micro distribution voting) | 96.40% | 64.8 |
| monks-problems-3 | PFossil:single (voting) | 98.38% | 64.8 |
| monks-problems-3 | PFossil:two-level (best rule) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:two-level (macro distribution voting) | 98.92% | 64.8 |
| monks-problems-3 | PFossil:two-level (micro distribution voting) | 96.40% | 64.8 |
| monks-problems-3 | PFossil:two-level (voting) | 98.92% | 64.8 |
| monks-problems-3 | Pypper:one-level (best rule) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:one-level (macro distribution voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:one-level (micro distribution voting) | 98.56% | 49.5 |
| monks-problems-3 | Pypper:one-level (voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:single (best rule) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:single (macro distribution voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:single (micro distribution voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:single (voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:two-level (best rule) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:two-level (macro distribution voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:two-level (micro distribution voting) | 98.92% | 49.5 |
| monks-problems-3 | Pypper:two-level (voting) | 98.92% | 49.5 |
| mushroom | PFoil:one-level (best rule) | 99.96% | 121.6 |
| mushroom | PFoil:one-level (macro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:one-level (micro distribution voting) | 99.96% | 121.6 |
| mushroom | PFoil:one-level (voting) | 100.00% | 121.6 |
| mushroom | PFoil:single (best rule) | 100.00% | 121.6 |
| mushroom | PFoil:single (macro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:single (micro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:single (voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level (best rule) | 100.00% | 121.6 |
| mushroom | PFoil:two-level (macro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level (micro distribution voting) | 100.00% | 121.6 |
| mushroom | PFoil:two-level (voting) | 100.00% | 121.6 |
| mushroom | PFossil:one-level (best rule) | 99.84% | 117.3 |
| mushroom | PFossil:one-level (macro distribution voting) | 99.95% | 117.3 |
| mushroom | PFossil:one-level (micro distribution voting) | 99.83% | 117.3 |
| mushroom | PFossil:one-level (voting) | 99.95% | 117.3 |
| mushroom | PFossil:single (best rule) | 99.95% | 117.3 |
| mushroom | PFossil:single (macro distribution voting) | 100.00% | 117.3 |
| mushroom | PFossil:single (micro distribution voting) | 99.95% | 117.3 |
| mushroom | PFossil:single (voting) | 99.95% | 117.3 |
| mushroom | PFossil:two-level (best rule) | 99.95% | 117.3 |
| mushroom | PFossil:two-level (macro distribution voting) | 99.99% | 117.3 |
| mushroom | PFossil:two-level (micro distribution voting) | 99.95% | 117.3 |
| mushroom | PFossil:two-level (voting) | 99.95% | 117.3 |
| mushroom | Pypper:one-level (best rule) | 99.91% | 107.3 |
| mushroom | Pypper:one-level (macro distribution voting) | 99.99% | 107.3 |
| mushroom | Pypper:one-level (micro distribution voting) | 99.91% | 107.3 |
| mushroom | Pypper:one-level (voting) | 99.95% | 107.3 |
| mushroom | Pypper:single (best rule) | 99.94% | 107.3 |
| mushroom | Pypper:single (macro distribution voting) | 99.94% | 107.3 |
| mushroom | Pypper:single (micro distribution voting) | 99.94% | 107.3 |
| mushroom | Pypper:single (voting) | 99.94% | 107.3 |
| mushroom | Pypper:two-level (best rule) | 100.00% | 107.3 |
| mushroom | Pypper:two-level (macro distribution voting) | 100.00% | 107.3 |
| mushroom | Pypper:two-level (micro distribution voting) | 100.00% | 107.3 |
| mushroom | Pypper:two-level (voting) | 100.00% | 107.3 |
| optdigits | PFoil:one-level (best rule) | 93.04% | 1171.8 |
| optdigits | PFoil:one-level (macro distribution voting) | 97.58% | 1171.8 |
| optdigits | PFoil:one-level (micro distribution voting) | 96.69% | 1171.8 |
| optdigits | PFoil:one-level (voting) | 97.58% | 1171.8 |
| optdigits | PFoil:single (best rule) | 93.35% | 1171.8 |
| optdigits | PFoil:single (macro distribution voting) | 94.02% | 1171.8 |
| optdigits | PFoil:single (micro distribution voting) | 92.44% | 1171.8 |
| optdigits | PFoil:single (voting) | 92.88% | 1171.8 |
| optdigits | PFoil:two-level (best rule) | 97.06% | 1171.8 |
| optdigits | PFoil:two-level (macro distribution voting) | 97.53% | 1171.8 |
| optdigits | PFoil:two-level (micro distribution voting) | 97.03% | 1171.8 |
| optdigits | PFoil:two-level (voting) | 97.28% | 1171.8 |
| optdigits | PFossil:one-level (best rule) | 92.79% | 722.9 |
| optdigits | PFossil:one-level (macro distribution voting) | 96.48% | 722.9 |
| optdigits | PFossil:one-level (micro distribution voting) | 95.39% | 722.9 |
| optdigits | PFossil:one-level (voting) | 96.37% | 722.9 |
| optdigits | PFossil:single (best rule) | 90.30% | 722.9 |
| optdigits | PFossil:single (macro distribution voting) | 90.39% | 722.9 |
| optdigits | PFossil:single (micro distribution voting) | 90.11% | 722.9 |
| optdigits | PFossil:single (voting) | 90.27% | 722.9 |
| optdigits | PFossil:two-level (best rule) | 96.01% | 722.9 |
| optdigits | PFossil:two-level (macro distribution voting) | 96.44% | 722.9 |
| optdigits | PFossil:two-level (micro distribution voting) | 95.77% | 722.9 |
| optdigits | PFossil:two-level (voting) | 96.12% | 722.9 |
| optdigits | Pypper:one-level (best rule) | 93.17% | 494.4 |
| optdigits | Pypper:one-level (macro distribution voting) | 96.62% | 494.4 |
| optdigits | Pypper:one-level (micro distribution voting) | 95.78% | 494.4 |
| optdigits | Pypper:one-level (voting) | 96.55% | 494.4 |
| optdigits | Pypper:single (best rule) | 90.18% | 494.4 |
| optdigits | Pypper:single (macro distribution voting) | 90.37% | 494.4 |
| optdigits | Pypper:single (micro distribution voting) | 90.27% | 494.4 |
| optdigits | Pypper:single (voting) | 90.14% | 494.4 |
| optdigits | Pypper:two-level (best rule) | 96.17% | 494.4 |
| optdigits | Pypper:two-level (macro distribution voting) | 96.42% | 494.4 |
| optdigits | Pypper:two-level (micro distribution voting) | 95.82% | 494.4 |
| optdigits | Pypper:two-level (voting) | 96.35% | 494.4 |
| ozone-level-8hr | PFoil:one-level (best rule) | 93.69% | 433.9 |
| ozone-level-8hr | PFoil:one-level (macro distribution voting) | 94.12% | 433.9 |
| ozone-level-8hr | PFoil:one-level (micro distribution voting) | 93.69% | 433.9 |
| ozone-level-8hr | PFoil:one-level (voting) | 94.28% | 433.9 |
| ozone-level-8hr | PFoil:single (best rule) | 93.29% | 433.9 |
| ozone-level-8hr | PFoil:single (macro distribution voting) | 93.53% | 433.9 |
| ozone-level-8hr | PFoil:single (micro distribution voting) | 93.37% | 433.9 |
| ozone-level-8hr | PFoil:single (voting) | 93.09% | 433.9 |
| ozone-level-8hr | PFoil:two-level (best rule) | 94.04% | 433.9 |
| ozone-level-8hr | PFoil:two-level (macro distribution voting) | 94.24% | 433.9 |
| ozone-level-8hr | PFoil:two-level (micro distribution voting) | 94.04% | 433.9 |
| ozone-level-8hr | PFoil:two-level (voting) | 94.20% | 433.9 |
| ozone-level-8hr | PFossil:one-level (best rule) | 93.84% | 108.4 |
| ozone-level-8hr | PFossil:one-level (macro distribution voting) | 93.84% | 108.4 |
| ozone-level-8hr | PFossil:one-level (micro distribution voting) | 93.65% | 108.4 |
| ozone-level-8hr | PFossil:one-level (voting) | 93.88% | 108.4 |
| ozone-level-8hr | PFossil:single (best rule) | 93.33% | 108.4 |
| ozone-level-8hr | PFossil:single (macro distribution voting) | 93.33% | 108.4 |
| ozone-level-8hr | PFossil:single (micro distribution voting) | 93.33% | 108.4 |
| ozone-level-8hr | PFossil:single (voting) | 93.33% | 108.4 |
| ozone-level-8hr | PFossil:two-level (best rule) | 93.61% | 108.4 |
| ozone-level-8hr | PFossil:two-level (macro distribution voting) | 93.69% | 108.4 |
| ozone-level-8hr | PFossil:two-level (micro distribution voting) | 93.65% | 108.4 |
| ozone-level-8hr | PFossil:two-level (voting) | 93.61% | 108.4 |
| ozone-level-8hr | Pypper:one-level (best rule) | 93.76% | 73.7 |
| ozone-level-8hr | Pypper:one-level (macro distribution voting) | 94.12% | 73.7 |
| ozone-level-8hr | Pypper:one-level (micro distribution voting) | 93.73% | 73.7 |
| ozone-level-8hr | Pypper:one-level (voting) | 94.20% | 73.7 |
| ozone-level-8hr | Pypper:single (best rule) | 93.33% | 73.7 |
| ozone-level-8hr | Pypper:single (macro distribution voting) | 93.33% | 73.7 |
| ozone-level-8hr | Pypper:single (micro distribution voting) | 93.33% | 73.7 |
| ozone-level-8hr | Pypper:single (voting) | 93.37% | 73.7 |
| ozone-level-8hr | Pypper:two-level (best rule) | 94.16% | 73.7 |
| ozone-level-8hr | Pypper:two-level (macro distribution voting) | 94.12% | 73.7 |
| ozone-level-8hr | Pypper:two-level (micro distribution voting) | 94.16% | 73.7 |
| ozone-level-8hr | Pypper:two-level (voting) | 94.16% | 73.7 |
| pc1 | PFoil:one-level (best rule) | 93.15% | 417.6 |
| pc1 | PFoil:one-level (macro distribution voting) | 92.61% | 417.6 |
| pc1 | PFoil:one-level (micro distribution voting) | 93.06% | 417.6 |
| pc1 | PFoil:one-level (voting) | 91.88% | 417.6 |
| pc1 | PFoil:single (best rule) | 93.06% | 417.6 |
| pc1 | PFoil:single (macro distribution voting) | 93.06% | 417.6 |
| pc1 | PFoil:single (micro distribution voting) | 92.97% | 417.6 |
| pc1 | PFoil:single (voting) | 91.52% | 417.6 |
| pc1 | PFoil:two-level (best rule) | 92.97% | 417.6 |
| pc1 | PFoil:two-level (macro distribution voting) | 92.79% | 417.6 |
| pc1 | PFoil:two-level (micro distribution voting) | 92.79% | 417.6 |
| pc1 | PFoil:two-level (voting) | 92.61% | 417.6 |
| pc1 | PFossil:one-level (best rule) | 93.33% | 69.2 |
| pc1 | PFossil:one-level (macro distribution voting) | 92.97% | 69.2 |
| pc1 | PFossil:one-level (micro distribution voting) | 93.15% | 69.2 |
| pc1 | PFossil:one-level (voting) | 92.97% | 69.2 |
| pc1 | PFossil:single (best rule) | 92.88% | 69.2 |
| pc1 | PFossil:single (macro distribution voting) | 92.79% | 69.2 |
| pc1 | PFossil:single (micro distribution voting) | 92.97% | 69.2 |
| pc1 | PFossil:single (voting) | 92.88% | 69.2 |
| pc1 | PFossil:two-level (best rule) | 92.88% | 69.2 |
| pc1 | PFossil:two-level (macro distribution voting) | 92.97% | 69.2 |
| pc1 | PFossil:two-level (micro distribution voting) | 93.06% | 69.2 |
| pc1 | PFossil:two-level (voting) | 93.06% | 69.2 |
| pc1 | Pypper:one-level (best rule) | 93.24% | 40.5 |
| pc1 | Pypper:one-level (macro distribution voting) | 93.06% | 40.5 |
| pc1 | Pypper:one-level (micro distribution voting) | 92.97% | 40.5 |
| pc1 | Pypper:one-level (voting) | 93.24% | 40.5 |
| pc1 | Pypper:single (best rule) | 92.61% | 40.5 |
| pc1 | Pypper:single (macro distribution voting) | 92.61% | 40.5 |
| pc1 | Pypper:single (micro distribution voting) | 92.61% | 40.5 |
| pc1 | Pypper:single (voting) | 92.61% | 40.5 |
| pc1 | Pypper:two-level (best rule) | 93.42% | 40.5 |
| pc1 | Pypper:two-level (macro distribution voting) | 93.42% | 40.5 |
| pc1 | Pypper:two-level (micro distribution voting) | 93.42% | 40.5 |
| pc1 | Pypper:two-level (voting) | 93.42% | 40.5 |
| pc3 | PFoil:one-level (best rule) | 89.76% | 592.6 |
| pc3 | PFoil:one-level (macro distribution voting) | 89.76% | 592.6 |
| pc3 | PFoil:one-level (micro distribution voting) | 89.76% | 592.6 |
| pc3 | PFoil:one-level (voting) | 89.12% | 592.6 |
| pc3 | PFoil:single (best rule) | 87.78% | 592.6 |
| pc3 | PFoil:single (macro distribution voting) | 87.84% | 592.6 |
| pc3 | PFoil:single (micro distribution voting) | 88.16% | 592.6 |
| pc3 | PFoil:single (voting) | 86.88% | 592.6 |
| pc3 | PFoil:two-level (best rule) | 89.70% | 592.6 |
| pc3 | PFoil:two-level (macro distribution voting) | 89.70% | 592.6 |
| pc3 | PFoil:two-level (micro distribution voting) | 89.57% | 592.6 |
| pc3 | PFoil:two-level (voting) | 89.25% | 592.6 |
| pc3 | PFossil:one-level (best rule) | 89.89% | 66.6 |
| pc3 | PFossil:one-level (macro distribution voting) | 89.51% | 66.6 |
| pc3 | PFossil:one-level (micro distribution voting) | 90.02% | 66.6 |
| pc3 | PFossil:one-level (voting) | 86.88% | 66.6 |
| pc3 | PFossil:single (best rule) | 85.09% | 66.6 |
| pc3 | PFossil:single (macro distribution voting) | 85.09% | 66.6 |
| pc3 | PFossil:single (micro distribution voting) | 85.09% | 66.6 |
| pc3 | PFossil:single (voting) | 85.09% | 66.6 |
| pc3 | PFossil:two-level (best rule) | 87.39% | 66.6 |
| pc3 | PFossil:two-level (macro distribution voting) | 87.46% | 66.6 |
| pc3 | PFossil:two-level (micro distribution voting) | 87.33% | 66.6 |
| pc3 | PFossil:two-level (voting) | 87.33% | 66.6 |
| pc3 | Pypper:one-level (best rule) | 89.83% | 46.2 |
| pc3 | Pypper:one-level (macro distribution voting) | 89.70% | 46.2 |
| pc3 | Pypper:one-level (micro distribution voting) | 89.89% | 46.2 |
| pc3 | Pypper:one-level (voting) | 89.00% | 46.2 |
| pc3 | Pypper:single (best rule) | 88.93% | 46.2 |
| pc3 | Pypper:single (macro distribution voting) | 88.93% | 46.2 |
| pc3 | Pypper:single (micro distribution voting) | 88.93% | 46.2 |
| pc3 | Pypper:single (voting) | 88.93% | 46.2 |
| pc3 | Pypper:two-level (best rule) | 89.51% | 46.2 |
| pc3 | Pypper:two-level (macro distribution voting) | 89.44% | 46.2 |
| pc3 | Pypper:two-level (micro distribution voting) | 89.51% | 46.2 |
| pc3 | Pypper:two-level (voting) | 89.44% | 46.2 |
| pc4 | PFoil:one-level (best rule) | 89.37% | 437.3 |
| pc4 | PFoil:one-level (macro distribution voting) | 90.40% | 437.3 |
| pc4 | PFoil:one-level (micro distribution voting) | 88.34% | 437.3 |
| pc4 | PFoil:one-level (voting) | 90.67% | 437.3 |
| pc4 | PFoil:single (best rule) | 89.85% | 437.3 |
| pc4 | PFoil:single (macro distribution voting) | 90.19% | 437.3 |
| pc4 | PFoil:single (micro distribution voting) | 89.23% | 437.3 |
| pc4 | PFoil:single (voting) | 88.96% | 437.3 |
| pc4 | PFoil:two-level (best rule) | 89.85% | 437.3 |
| pc4 | PFoil:two-level (macro distribution voting) | 90.19% | 437.3 |
| pc4 | PFoil:two-level (micro distribution voting) | 89.51% | 437.3 |
| pc4 | PFoil:two-level (voting) | 90.12% | 437.3 |
| pc4 | PFossil:one-level (best rule) | 89.10% | 142.2 |
| pc4 | PFossil:one-level (macro distribution voting) | 90.05% | 142.2 |
| pc4 | PFossil:one-level (micro distribution voting) | 89.30% | 142.2 |
| pc4 | PFossil:one-level (voting) | 90.40% | 142.2 |
| pc4 | PFossil:single (best rule) | 89.51% | 142.2 |
| pc4 | PFossil:single (macro distribution voting) | 89.71% | 142.2 |
| pc4 | PFossil:single (micro distribution voting) | 89.58% | 142.2 |
| pc4 | PFossil:single (voting) | 89.78% | 142.2 |
| pc4 | PFossil:two-level (best rule) | 89.99% | 142.2 |
| pc4 | PFossil:two-level (macro distribution voting) | 90.19% | 142.2 |
| pc4 | PFossil:two-level (micro distribution voting) | 90.05% | 142.2 |
| pc4 | PFossil:two-level (voting) | 90.19% | 142.2 |
| pc4 | Pypper:one-level (best rule) | 88.62% | 67.9 |
| pc4 | Pypper:one-level (macro distribution voting) | 88.61% | 67.9 |
| pc4 | Pypper:one-level (micro distribution voting) | 88.41% | 67.9 |
| pc4 | Pypper:one-level (voting) | 88.55% | 67.9 |
| pc4 | Pypper:single (best rule) | 87.79% | 67.9 |
| pc4 | Pypper:single (macro distribution voting) | 87.79% | 67.9 |
| pc4 | Pypper:single (micro distribution voting) | 87.79% | 67.9 |
| pc4 | Pypper:single (voting) | 87.79% | 67.9 |
| pc4 | Pypper:two-level (best rule) | 88.20% | 67.9 |
| pc4 | Pypper:two-level (macro distribution voting) | 88.27% | 67.9 |
| pc4 | Pypper:two-level (micro distribution voting) | 88.27% | 67.9 |
| pc4 | Pypper:two-level (voting) | 88.34% | 67.9 |
| phoneme | PFoil:one-level (best rule) | 84.94% | 3538.4 |
| phoneme | PFoil:one-level (macro distribution voting) | 85.90% | 3538.4 |
| phoneme | PFoil:one-level (micro distribution voting) | 82.42% | 3538.4 |
| phoneme | PFoil:one-level (voting) | 86.31% | 3538.4 |
| phoneme | PFoil:single (best rule) | 85.33% | 3538.4 |
| phoneme | PFoil:single (macro distribution voting) | 85.44% | 3538.4 |
| phoneme | PFoil:single (micro distribution voting) | 81.36% | 3538.4 |
| phoneme | PFoil:single (voting) | 84.18% | 3538.4 |
| phoneme | PFoil:two-level (best rule) | 85.81% | 3538.4 |
| phoneme | PFoil:two-level (macro distribution voting) | 86.21% | 3538.4 |
| phoneme | PFoil:two-level (micro distribution voting) | 84.44% | 3538.4 |
| phoneme | PFoil:two-level (voting) | 85.01% | 3538.4 |
| phoneme | PFossil:one-level (best rule) | 78.63% | 50.8 |
| phoneme | PFossil:one-level (macro distribution voting) | 78.55% | 50.8 |
| phoneme | PFossil:one-level (micro distribution voting) | 78.18% | 50.8 |
| phoneme | PFossil:one-level (voting) | 78.35% | 50.8 |
| phoneme | PFossil:single (best rule) | 78.35% | 50.8 |
| phoneme | PFossil:single (macro distribution voting) | 78.39% | 50.8 |
| phoneme | PFossil:single (micro distribution voting) | 78.35% | 50.8 |
| phoneme | PFossil:single (voting) | 78.46% | 50.8 |
| phoneme | PFossil:two-level (best rule) | 77.76% | 50.8 |
| phoneme | PFossil:two-level (macro distribution voting) | 77.78% | 50.8 |
| phoneme | PFossil:two-level (micro distribution voting) | 77.76% | 50.8 |
| phoneme | PFossil:two-level (voting) | 77.87% | 50.8 |
| phoneme | Pypper:one-level (best rule) | 84.47% | 246.3 |
| phoneme | Pypper:one-level (macro distribution voting) | 84.35% | 246.3 |
| phoneme | Pypper:one-level (micro distribution voting) | 83.22% | 246.3 |
| phoneme | Pypper:one-level (voting) | 84.05% | 246.3 |
| phoneme | Pypper:single (best rule) | 83.53% | 246.3 |
| phoneme | Pypper:single (macro distribution voting) | 83.40% | 246.3 |
| phoneme | Pypper:single (micro distribution voting) | 83.24% | 246.3 |
| phoneme | Pypper:single (voting) | 83.40% | 246.3 |
| phoneme | Pypper:two-level (best rule) | 84.36% | 246.3 |
| phoneme | Pypper:two-level (macro distribution voting) | 84.36% | 246.3 |
| phoneme | Pypper:two-level (micro distribution voting) | 84.16% | 246.3 |
| phoneme | Pypper:two-level (voting) | 84.27% | 246.3 |
| primary-tumor | PFoil:one-level (best rule) | 43.07% | 864.2 |
| primary-tumor | PFoil:one-level (macro distribution voting) | 45.73% | 864.2 |
| primary-tumor | PFoil:one-level (micro distribution voting) | 44.27% | 864.2 |
| primary-tumor | PFoil:one-level (voting) | 42.49% | 864.2 |
| primary-tumor | PFoil:single (best rule) | 43.35% | 864.2 |
| primary-tumor | PFoil:single (macro distribution voting) | 43.65% | 864.2 |
| primary-tumor | PFoil:single (micro distribution voting) | 37.75% | 864.2 |
| primary-tumor | PFoil:single (voting) | 36.28% | 864.2 |
| primary-tumor | PFoil:two-level (best rule) | 43.06% | 864.2 |
| primary-tumor | PFoil:two-level (macro distribution voting) | 43.65% | 864.2 |
| primary-tumor | PFoil:two-level (micro distribution voting) | 43.07% | 864.2 |
| primary-tumor | PFoil:two-level (voting) | 40.73% | 864.2 |
| primary-tumor | PFossil:one-level (best rule) | 42.49% | 676.3 |
| primary-tumor | PFossil:one-level (macro distribution voting) | 43.07% | 676.3 |
| primary-tumor | PFossil:one-level (micro distribution voting) | 45.44% | 676.3 |
| primary-tumor | PFossil:one-level (voting) | 43.68% | 676.3 |
| primary-tumor | PFossil:single (best rule) | 43.38% | 676.3 |
| primary-tumor | PFossil:single (macro distribution voting) | 41.61% | 676.3 |
| primary-tumor | PFossil:single (micro distribution voting) | 41.01% | 676.3 |
| primary-tumor | PFossil:single (voting) | 41.01% | 676.3 |
| primary-tumor | PFossil:two-level (best rule) | 41.60% | 676.3 |
| primary-tumor | PFossil:two-level (macro distribution voting) | 44.56% | 676.3 |
| primary-tumor | PFossil:two-level (micro distribution voting) | 42.19% | 676.3 |
| primary-tumor | PFossil:two-level (voting) | 43.08% | 676.3 |
| primary-tumor | Pypper:one-level (best rule) | 44.56% | 66.7 |
| primary-tumor | Pypper:one-level (macro distribution voting) | 44.86% | 66.7 |
| primary-tumor | Pypper:one-level (micro distribution voting) | 43.67% | 66.7 |
| primary-tumor | Pypper:one-level (voting) | 44.27% | 66.7 |
| primary-tumor | Pypper:single (best rule) | 38.65% | 66.7 |
| primary-tumor | Pypper:single (macro distribution voting) | 38.07% | 66.7 |
| primary-tumor | Pypper:single (micro distribution voting) | 38.95% | 66.7 |
| primary-tumor | Pypper:single (voting) | 37.77% | 66.7 |
| primary-tumor | Pypper:two-level (best rule) | 43.67% | 66.7 |
| primary-tumor | Pypper:two-level (macro distribution voting) | 43.38% | 66.7 |
| primary-tumor | Pypper:two-level (micro distribution voting) | 43.38% | 66.7 |
| primary-tumor | Pypper:two-level (voting) | 43.38% | 66.7 |
| qsar-biodeg | PFoil:one-level (best rule) | 84.17% | 515.4 |
| qsar-biodeg | PFoil:one-level (macro distribution voting) | 86.45% | 515.4 |
| qsar-biodeg | PFoil:one-level (micro distribution voting) | 83.32% | 515.4 |
| qsar-biodeg | PFoil:one-level (voting) | 86.36% | 515.4 |
| qsar-biodeg | PFoil:single (best rule) | 84.55% | 515.4 |
| qsar-biodeg | PFoil:single (macro distribution voting) | 84.17% | 515.4 |
| qsar-biodeg | PFoil:single (micro distribution voting) | 81.23% | 515.4 |
| qsar-biodeg | PFoil:single (voting) | 83.89% | 515.4 |
| qsar-biodeg | PFoil:two-level (best rule) | 86.35% | 515.4 |
| qsar-biodeg | PFoil:two-level (macro distribution voting) | 86.26% | 515.4 |
| qsar-biodeg | PFoil:two-level (micro distribution voting) | 84.74% | 515.4 |
| qsar-biodeg | PFoil:two-level (voting) | 85.59% | 515.4 |
| qsar-biodeg | PFossil:one-level (best rule) | 82.94% | 135.5 |
| qsar-biodeg | PFossil:one-level (macro distribution voting) | 85.41% | 135.5 |
| qsar-biodeg | PFossil:one-level (micro distribution voting) | 83.89% | 135.5 |
| qsar-biodeg | PFossil:one-level (voting) | 85.31% | 135.5 |
| qsar-biodeg | PFossil:single (best rule) | 84.65% | 135.5 |
| qsar-biodeg | PFossil:single (macro distribution voting) | 84.74% | 135.5 |
| qsar-biodeg | PFossil:single (micro distribution voting) | 84.17% | 135.5 |
| qsar-biodeg | PFossil:single (voting) | 83.60% | 135.5 |
| qsar-biodeg | PFossil:two-level (best rule) | 85.69% | 135.5 |
| qsar-biodeg | PFossil:two-level (macro distribution voting) | 85.60% | 135.5 |
| qsar-biodeg | PFossil:two-level (micro distribution voting) | 84.18% | 135.5 |
| qsar-biodeg | PFossil:two-level (voting) | 84.45% | 135.5 |
| qsar-biodeg | Pypper:one-level (best rule) | 83.22% | 112.3 |
| qsar-biodeg | Pypper:one-level (macro distribution voting) | 84.36% | 112.3 |
| qsar-biodeg | Pypper:one-level (micro distribution voting) | 83.12% | 112.3 |
| qsar-biodeg | Pypper:one-level (voting) | 83.89% | 112.3 |
| qsar-biodeg | Pypper:single (best rule) | 85.03% | 112.3 |
| qsar-biodeg | Pypper:single (macro distribution voting) | 85.03% | 112.3 |
| qsar-biodeg | Pypper:single (micro distribution voting) | 84.36% | 112.3 |
| qsar-biodeg | Pypper:single (voting) | 84.65% | 112.3 |
| qsar-biodeg | Pypper:two-level (best rule) | 84.36% | 112.3 |
| qsar-biodeg | Pypper:two-level (macro distribution voting) | 84.55% | 112.3 |
| qsar-biodeg | Pypper:two-level (micro distribution voting) | 83.60% | 112.3 |
| qsar-biodeg | Pypper:two-level (voting) | 83.88% | 112.3 |
| segment | PFoil:one-level (best rule) | 94.76% | 609.0 |
| segment | PFoil:one-level (macro distribution voting) | 96.36% | 609.0 |
| segment | PFoil:one-level (micro distribution voting) | 95.24% | 609.0 |
| segment | PFoil:one-level (voting) | 96.41% | 609.0 |
| segment | PFoil:single (best rule) | 95.19% | 609.0 |
| segment | PFoil:single (macro distribution voting) | 95.19% | 609.0 |
| segment | PFoil:single (micro distribution voting) | 93.03% | 609.0 |
| segment | PFoil:single (voting) | 90.91% | 609.0 |
| segment | PFoil:two-level (best rule) | 96.23% | 609.0 |
| segment | PFoil:two-level (macro distribution voting) | 96.45% | 609.0 |
| segment | PFoil:two-level (micro distribution voting) | 96.36% | 609.0 |
| segment | PFoil:two-level (voting) | 96.45% | 609.0 |
| segment | PFossil:one-level (best rule) | 95.06% | 319.3 |
| segment | PFossil:one-level (macro distribution voting) | 95.71% | 319.3 |
| segment | PFossil:one-level (micro distribution voting) | 94.81% | 319.3 |
| segment | PFossil:one-level (voting) | 95.58% | 319.3 |
| segment | PFossil:single (best rule) | 93.33% | 319.3 |
| segment | PFossil:single (macro distribution voting) | 93.16% | 319.3 |
| segment | PFossil:single (micro distribution voting) | 93.12% | 319.3 |
| segment | PFossil:single (voting) | 92.73% | 319.3 |
| segment | PFossil:two-level (best rule) | 95.32% | 319.3 |
| segment | PFossil:two-level (macro distribution voting) | 95.45% | 319.3 |
| segment | PFossil:two-level (micro distribution voting) | 95.19% | 319.3 |
| segment | PFossil:two-level (voting) | 94.94% | 319.3 |
| segment | Pypper:one-level (best rule) | 94.63% | 187.2 |
| segment | Pypper:one-level (macro distribution voting) | 95.24% | 187.2 |
| segment | Pypper:one-level (micro distribution voting) | 94.68% | 187.2 |
| segment | Pypper:one-level (voting) | 95.32% | 187.2 |
| segment | Pypper:single (best rule) | 93.46% | 187.2 |
| segment | Pypper:single (macro distribution voting) | 93.51% | 187.2 |
| segment | Pypper:single (micro distribution voting) | 93.20% | 187.2 |
| segment | Pypper:single (voting) | 93.29% | 187.2 |
| segment | Pypper:two-level (best rule) | 95.15% | 187.2 |
| segment | Pypper:two-level (macro distribution voting) | 95.28% | 187.2 |
| segment | Pypper:two-level (micro distribution voting) | 95.11% | 187.2 |
| segment | Pypper:two-level (voting) | 95.28% | 187.2 |
| sick | PFoil:one-level (best rule) | 97.56% | 502.3 |
| sick | PFoil:one-level (macro distribution voting) | 98.33% | 502.3 |
| sick | PFoil:one-level (micro distribution voting) | 96.37% | 502.3 |
| sick | PFoil:one-level (voting) | 98.25% | 502.3 |
| sick | PFoil:single (best rule) | 97.99% | 502.3 |
| sick | PFoil:single (macro distribution voting) | 98.14% | 502.3 |
| sick | PFoil:single (micro distribution voting) | 96.95% | 502.3 |
| sick | PFoil:single (voting) | 97.51% | 502.3 |
| sick | PFoil:two-level (best rule) | 98.12% | 502.3 |
| sick | PFoil:two-level (macro distribution voting) | 98.36% | 502.3 |
| sick | PFoil:two-level (micro distribution voting) | 97.69% | 502.3 |
| sick | PFoil:two-level (voting) | 98.14% | 502.3 |
| sick | PFossil:one-level (best rule) | 97.48% | 166.5 |
| sick | PFossil:one-level (macro distribution voting) | 98.14% | 166.5 |
| sick | PFossil:one-level (micro distribution voting) | 97.83% | 166.5 |
| sick | PFossil:one-level (voting) | 98.25% | 166.5 |
| sick | PFossil:single (best rule) | 98.30% | 166.5 |
| sick | PFossil:single (macro distribution voting) | 98.25% | 166.5 |
| sick | PFossil:single (micro distribution voting) | 98.20% | 166.5 |
| sick | PFossil:single (voting) | 98.20% | 166.5 |
| sick | PFossil:two-level (best rule) | 98.28% | 166.5 |
| sick | PFossil:two-level (macro distribution voting) | 98.22% | 166.5 |
| sick | PFossil:two-level (micro distribution voting) | 98.20% | 166.5 |
| sick | PFossil:two-level (voting) | 98.20% | 166.5 |
| sick | Pypper:one-level (best rule) | 98.06% | 83.3 |
| sick | Pypper:one-level (macro distribution voting) | 98.20% | 83.3 |
| sick | Pypper:one-level (micro distribution voting) | 97.93% | 83.3 |
| sick | Pypper:one-level (voting) | 98.14% | 83.3 |
| sick | Pypper:single (best rule) | 98.20% | 83.3 |
| sick | Pypper:single (macro distribution voting) | 98.20% | 83.3 |
| sick | Pypper:single (micro distribution voting) | 98.22% | 83.3 |
| sick | Pypper:single (voting) | 98.22% | 83.3 |
| sick | Pypper:two-level (best rule) | 98.20% | 83.3 |
| sick | Pypper:two-level (macro distribution voting) | 98.17% | 83.3 |
| sick | Pypper:two-level (micro distribution voting) | 98.17% | 83.3 |
| sick | Pypper:two-level (voting) | 98.20% | 83.3 |
| solar-flare | PFoil:one-level (best rule) | 73.36% | 875.2 |
| solar-flare | PFoil:one-level (macro distribution voting) | 74.21% | 875.2 |
| solar-flare | PFoil:one-level (micro distribution voting) | 73.92% | 875.2 |
| solar-flare | PFoil:one-level (voting) | 73.17% | 875.2 |
| solar-flare | PFoil:single (best rule) | 73.73% | 875.2 |
| solar-flare | PFoil:single (macro distribution voting) | 74.39% | 875.2 |
| solar-flare | PFoil:single (micro distribution voting) | 74.02% | 875.2 |
| solar-flare | PFoil:single (voting) | 71.30% | 875.2 |
| solar-flare | PFoil:two-level (best rule) | 73.36% | 875.2 |
| solar-flare | PFoil:two-level (macro distribution voting) | 74.30% | 875.2 |
| solar-flare | PFoil:two-level (micro distribution voting) | 75.24% | 875.2 |
| solar-flare | PFoil:two-level (voting) | 71.67% | 875.2 |
| solar-flare | PFossil:one-level (best rule) | 72.42% | 183.8 |
| solar-flare | PFossil:one-level (macro distribution voting) | 74.21% | 183.8 |
| solar-flare | PFossil:one-level (micro distribution voting) | 74.96% | 183.8 |
| solar-flare | PFossil:one-level (voting) | 73.64% | 183.8 |
| solar-flare | PFossil:single (best rule) | 74.02% | 183.8 |
| solar-flare | PFossil:single (macro distribution voting) | 74.77% | 183.8 |
| solar-flare | PFossil:single (micro distribution voting) | 74.30% | 183.8 |
| solar-flare | PFossil:single (voting) | 70.07% | 183.8 |
| solar-flare | PFossil:two-level (best rule) | 73.64% | 183.8 |
| solar-flare | PFossil:two-level (macro distribution voting) | 73.83% | 183.8 |
| solar-flare | PFossil:two-level (micro distribution voting) | 74.77% | 183.8 |
| solar-flare | PFossil:two-level (voting) | 69.14% | 183.8 |
| solar-flare | Pypper:one-level (best rule) | 74.59% | 73.8 |
| solar-flare | Pypper:one-level (macro distribution voting) | 75.34% | 73.8 |
| solar-flare | Pypper:one-level (micro distribution voting) | 74.77% | 73.8 |
| solar-flare | Pypper:one-level (voting) | 74.59% | 73.8 |
| solar-flare | Pypper:single (best rule) | 69.43% | 73.8 |
| solar-flare | Pypper:single (macro distribution voting) | 69.52% | 73.8 |
| solar-flare | Pypper:single (micro distribution voting) | 69.24% | 73.8 |
| solar-flare | Pypper:single (voting) | 69.14% | 73.8 |
| solar-flare | Pypper:two-level (best rule) | 74.96% | 73.8 |
| solar-flare | Pypper:two-level (macro distribution voting) | 74.87% | 73.8 |
| solar-flare | Pypper:two-level (micro distribution voting) | 74.50% | 73.8 |
| solar-flare | Pypper:two-level (voting) | 74.87% | 73.8 |
| sonar | PFoil:one-level (best rule) | 71.24% | 119.2 |
| sonar | PFoil:one-level (macro distribution voting) | 82.26% | 119.2 |
| sonar | PFoil:one-level (micro distribution voting) | 79.88% | 119.2 |
| sonar | PFoil:one-level (voting) | 81.79% | 119.2 |
| sonar | PFoil:single (best rule) | 77.90% | 119.2 |
| sonar | PFoil:single (macro distribution voting) | 75.07% | 119.2 |
| sonar | PFoil:single (micro distribution voting) | 76.00% | 119.2 |
| sonar | PFoil:single (voting) | 75.07% | 119.2 |
| sonar | PFoil:two-level (best rule) | 80.33% | 119.2 |
| sonar | PFoil:two-level (macro distribution voting) | 81.31% | 119.2 |
| sonar | PFoil:two-level (micro distribution voting) | 80.81% | 119.2 |
| sonar | PFoil:two-level (voting) | 77.93% | 119.2 |
| sonar | PFossil:one-level (best rule) | 73.60% | 130.0 |
| sonar | PFossil:one-level (macro distribution voting) | 76.95% | 130.0 |
| sonar | PFossil:one-level (micro distribution voting) | 75.48% | 130.0 |
| sonar | PFossil:one-level (voting) | 76.98% | 130.0 |
| sonar | PFossil:single (best rule) | 73.55% | 130.0 |
| sonar | PFossil:single (macro distribution voting) | 75.07% | 130.0 |
| sonar | PFossil:single (micro distribution voting) | 74.02% | 130.0 |
| sonar | PFossil:single (voting) | 75.52% | 130.0 |
| sonar | PFossil:two-level (best rule) | 78.40% | 130.0 |
| sonar | PFossil:two-level (macro distribution voting) | 76.98% | 130.0 |
| sonar | PFossil:two-level (micro distribution voting) | 76.95% | 130.0 |
| sonar | PFossil:two-level (voting) | 77.93% | 130.0 |
| sonar | Pypper:one-level (best rule) | 70.71% | 57.1 |
| sonar | Pypper:one-level (macro distribution voting) | 78.88% | 57.1 |
| sonar | Pypper:one-level (micro distribution voting) | 75.02% | 57.1 |
| sonar | Pypper:one-level (voting) | 77.43% | 57.1 |
| sonar | Pypper:single (best rule) | 76.48% | 57.1 |
| sonar | Pypper:single (macro distribution voting) | 75.52% | 57.1 |
| sonar | Pypper:single (micro distribution voting) | 75.50% | 57.1 |
| sonar | Pypper:single (voting) | 73.14% | 57.1 |
| sonar | Pypper:two-level (best rule) | 77.88% | 57.1 |
| sonar | Pypper:two-level (macro distribution voting) | 77.90% | 57.1 |
| sonar | Pypper:two-level (micro distribution voting) | 74.98% | 57.1 |
| sonar | Pypper:two-level (voting) | 74.50% | 57.1 |
| soybean | PFoil:one-level (best rule) | 88.44% | 476.0 |
| soybean | PFoil:one-level (macro distribution voting) | 93.42% | 476.0 |
| soybean | PFoil:one-level (micro distribution voting) | 90.05% | 476.0 |
| soybean | PFoil:one-level (voting) | 91.81% | 476.0 |
| soybean | PFoil:single (best rule) | 90.92% | 476.0 |
| soybean | PFoil:single (macro distribution voting) | 90.35% | 476.0 |
| soybean | PFoil:single (micro distribution voting) | 86.98% | 476.0 |
| soybean | PFoil:single (voting) | 84.78% | 476.0 |
| soybean | PFoil:two-level (best rule) | 92.98% | 476.0 |
| soybean | PFoil:two-level (macro distribution voting) | 91.52% | 476.0 |
| soybean | PFoil:two-level (micro distribution voting) | 91.66% | 476.0 |
| soybean | PFoil:two-level (voting) | 86.69% | 476.0 |
| soybean | PFossil:one-level (best rule) | 88.30% | 400.9 |
| soybean | PFossil:one-level (macro distribution voting) | 93.42% | 400.9 |
| soybean | PFossil:one-level (micro distribution voting) | 91.52% | 400.9 |
| soybean | PFossil:one-level (voting) | 92.98% | 400.9 |
| soybean | PFossil:single (best rule) | 92.24% | 400.9 |
| soybean | PFossil:single (macro distribution voting) | 91.36% | 400.9 |
| soybean | PFossil:single (micro distribution voting) | 91.66% | 400.9 |
| soybean | PFossil:single (voting) | 90.34% | 400.9 |
| soybean | PFossil:two-level (best rule) | 93.27% | 400.9 |
| soybean | PFossil:two-level (macro distribution voting) | 92.39% | 400.9 |
| soybean | PFossil:two-level (micro distribution voting) | 92.39% | 400.9 |
| soybean | PFossil:two-level (voting) | 91.51% | 400.9 |
| soybean | Pypper:one-level (best rule) | 89.17% | 243.1 |
| soybean | Pypper:one-level (macro distribution voting) | 92.54% | 243.1 |
| soybean | Pypper:one-level (micro distribution voting) | 88.74% | 243.1 |
| soybean | Pypper:one-level (voting) | 91.66% | 243.1 |
| soybean | Pypper:single (best rule) | 89.91% | 243.1 |
| soybean | Pypper:single (macro distribution voting) | 89.91% | 243.1 |
| soybean | Pypper:single (micro distribution voting) | 87.28% | 243.1 |
| soybean | Pypper:single (voting) | 86.84% | 243.1 |
| soybean | Pypper:two-level (best rule) | 91.81% | 243.1 |
| soybean | Pypper:two-level (macro distribution voting) | 91.81% | 243.1 |
| soybean | Pypper:two-level (micro distribution voting) | 89.61% | 243.1 |
| soybean | Pypper:two-level (voting) | 88.88% | 243.1 |
| spambase | PFoil:one-level (best rule) | 94.37% | 1110.8 |
| spambase | PFoil:one-level (macro distribution voting) | 94.83% | 1110.8 |
| spambase | PFoil:one-level (micro distribution voting) | 93.24% | 1110.8 |
| spambase | PFoil:one-level (voting) | 94.96% | 1110.8 |
| spambase | PFoil:single (best rule) | 93.87% | 1110.8 |
| spambase | PFoil:single (macro distribution voting) | 94.02% | 1110.8 |
| spambase | PFoil:single (micro distribution voting) | 90.50% | 1110.8 |
| spambase | PFoil:single (voting) | 93.59% | 1110.8 |
| spambase | PFoil:two-level (best rule) | 94.98% | 1110.8 |
| spambase | PFoil:two-level (macro distribution voting) | 94.81% | 1110.8 |
| spambase | PFoil:two-level (micro distribution voting) | 94.04% | 1110.8 |
| spambase | PFoil:two-level (voting) | 94.91% | 1110.8 |
| spambase | PFossil:one-level (best rule) | 93.13% | 114.0 |
| spambase | PFossil:one-level (macro distribution voting) | 93.52% | 114.0 |
| spambase | PFossil:one-level (micro distribution voting) | 92.13% | 114.0 |
| spambase | PFossil:one-level (voting) | 93.46% | 114.0 |
| spambase | PFossil:single (best rule) | 92.81% | 114.0 |
| spambase | PFossil:single (macro distribution voting) | 92.76% | 114.0 |
| spambase | PFossil:single (micro distribution voting) | 91.81% | 114.0 |
| spambase | PFossil:single (voting) | 92.41% | 114.0 |
| spambase | PFossil:two-level (best rule) | 93.44% | 114.0 |
| spambase | PFossil:two-level (macro distribution voting) | 93.50% | 114.0 |
| spambase | PFossil:two-level (micro distribution voting) | 92.76% | 114.0 |
| spambase | PFossil:two-level (voting) | 93.05% | 114.0 |
| spambase | Pypper:one-level (best rule) | 93.81% | 217.1 |
| spambase | Pypper:one-level (macro distribution voting) | 93.65% | 217.1 |
| spambase | Pypper:one-level (micro distribution voting) | 92.87% | 217.1 |
| spambase | Pypper:one-level (voting) | 93.70% | 217.1 |
| spambase | Pypper:single (best rule) | 92.96% | 217.1 |
| spambase | Pypper:single (macro distribution voting) | 92.87% | 217.1 |
| spambase | Pypper:single (micro distribution voting) | 92.76% | 217.1 |
| spambase | Pypper:single (voting) | 92.68% | 217.1 |
| spambase | Pypper:two-level (best rule) | 93.74% | 217.1 |
| spambase | Pypper:two-level (macro distribution voting) | 93.76% | 217.1 |
| spambase | Pypper:two-level (micro distribution voting) | 93.44% | 217.1 |
| spambase | Pypper:two-level (voting) | 93.81% | 217.1 |
| splice | PFoil:one-level (best rule) | 94.45% | 598.6 |
| splice | PFoil:one-level (macro distribution voting) | 95.99% | 598.6 |
| splice | PFoil:one-level (micro distribution voting) | 95.89% | 598.6 |
| splice | PFoil:one-level (voting) | 95.89% | 598.6 |
| splice | PFoil:single (best rule) | 92.76% | 598.6 |
| splice | PFoil:single (macro distribution voting) | 92.82% | 598.6 |
| splice | PFoil:single (micro distribution voting) | 92.79% | 598.6 |
| splice | PFoil:single (voting) | 92.41% | 598.6 |
| splice | PFoil:two-level (best rule) | 95.89% | 598.6 |
| splice | PFoil:two-level (macro distribution voting) | 96.14% | 598.6 |
| splice | PFoil:two-level (micro distribution voting) | 95.86% | 598.6 |
| splice | PFoil:two-level (voting) | 95.92% | 598.6 |
| splice | PFossil:one-level (best rule) | 90.72% | 288.6 |
| splice | PFossil:one-level (macro distribution voting) | 95.33% | 288.6 |
| splice | PFossil:one-level (micro distribution voting) | 90.78% | 288.6 |
| splice | PFossil:one-level (voting) | 94.08% | 288.6 |
| splice | PFossil:single (best rule) | 94.23% | 288.6 |
| splice | PFossil:single (macro distribution voting) | 94.29% | 288.6 |
| splice | PFossil:single (micro distribution voting) | 90.38% | 288.6 |
| splice | PFossil:single (voting) | 89.09% | 288.6 |
| splice | PFossil:two-level (best rule) | 95.30% | 288.6 |
| splice | PFossil:two-level (macro distribution voting) | 95.49% | 288.6 |
| splice | PFossil:two-level (micro distribution voting) | 90.91% | 288.6 |
| splice | PFossil:two-level (voting) | 90.60% | 288.6 |
| splice | Pypper:one-level (best rule) | 94.73% | 168.6 |
| splice | Pypper:one-level (macro distribution voting) | 95.36% | 168.6 |
| splice | Pypper:one-level (micro distribution voting) | 95.08% | 168.6 |
| splice | Pypper:one-level (voting) | 95.39% | 168.6 |
| splice | Pypper:single (best rule) | 93.45% | 168.6 |
| splice | Pypper:single (macro distribution voting) | 93.70% | 168.6 |
| splice | Pypper:single (micro distribution voting) | 93.42% | 168.6 |
| splice | Pypper:single (voting) | 93.51% | 168.6 |
| splice | Pypper:two-level (best rule) | 95.39% | 168.6 |
| splice | Pypper:two-level (macro distribution voting) | 95.61% | 168.6 |
| splice | Pypper:two-level (micro distribution voting) | 95.20% | 168.6 |
| splice | Pypper:two-level (voting) | 95.61% | 168.6 |
| texture | PFoil:one-level (best rule) | 93.07% | 1317.8 |
| texture | PFoil:one-level (macro distribution voting) | 97.35% | 1317.8 |
| texture | PFoil:one-level (micro distribution voting) | 95.62% | 1317.8 |
| texture | PFoil:one-level (voting) | 97.20% | 1317.8 |
| texture | PFoil:single (best rule) | 94.51% | 1317.8 |
| texture | PFoil:single (macro distribution voting) | 95.02% | 1317.8 |
| texture | PFoil:single (micro distribution voting) | 91.93% | 1317.8 |
| texture | PFoil:single (voting) | 93.25% | 1317.8 |
| texture | PFoil:two-level (best rule) | 96.96% | 1317.8 |
| texture | PFoil:two-level (macro distribution voting) | 97.38% | 1317.8 |
| texture | PFoil:two-level (micro distribution voting) | 96.51% | 1317.8 |
| texture | PFoil:two-level (voting) | 97.02% | 1317.8 |
| texture | PFossil:one-level (best rule) | 93.07% | 835.4 |
| texture | PFossil:one-level (macro distribution voting) | 95.96% | 835.4 |
| texture | PFossil:one-level (micro distribution voting) | 92.45% | 835.4 |
| texture | PFossil:one-level (voting) | 95.71% | 835.4 |
| texture | PFossil:single (best rule) | 92.22% | 835.4 |
| texture | PFossil:single (macro distribution voting) | 92.45% | 835.4 |
| texture | PFossil:single (micro distribution voting) | 90.29% | 835.4 |
| texture | PFossil:single (voting) | 91.65% | 835.4 |
| texture | PFossil:two-level (best rule) | 95.73% | 835.4 |
| texture | PFossil:two-level (macro distribution voting) | 95.84% | 835.4 |
| texture | PFossil:two-level (micro distribution voting) | 93.24% | 835.4 |
| texture | PFossil:two-level (voting) | 95.20% | 835.4 |
| texture | Pypper:one-level (best rule) | 93.38% | 526.0 |
| texture | Pypper:one-level (macro distribution voting) | 95.95% | 526.0 |
| texture | Pypper:one-level (micro distribution voting) | 94.67% | 526.0 |
| texture | Pypper:one-level (voting) | 95.96% | 526.0 |
| texture | Pypper:single (best rule) | 91.78% | 526.0 |
| texture | Pypper:single (macro distribution voting) | 92.04% | 526.0 |
| texture | Pypper:single (micro distribution voting) | 91.49% | 526.0 |
| texture | Pypper:single (voting) | 91.71% | 526.0 |
| texture | Pypper:two-level (best rule) | 95.85% | 526.0 |
| texture | Pypper:two-level (macro distribution voting) | 95.84% | 526.0 |
| texture | Pypper:two-level (micro distribution voting) | 95.49% | 526.0 |
| texture | Pypper:two-level (voting) | 95.82% | 526.0 |
| tic-tac-toe | PFoil:one-level (best rule) | 97.18% | 238.4 |
| tic-tac-toe | PFoil:one-level (macro distribution voting) | 99.27% | 238.4 |
| tic-tac-toe | PFoil:one-level (micro distribution voting) | 98.85% | 238.4 |
| tic-tac-toe | PFoil:one-level (voting) | 99.58% | 238.4 |
| tic-tac-toe | PFoil:single (best rule) | 99.06% | 238.4 |
| tic-tac-toe | PFoil:single (macro distribution voting) | 98.85% | 238.4 |
| tic-tac-toe | PFoil:single (micro distribution voting) | 99.06% | 238.4 |
| tic-tac-toe | PFoil:single (voting) | 98.43% | 238.4 |
| tic-tac-toe | PFoil:two-level (best rule) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level (macro distribution voting) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level (micro distribution voting) | 99.48% | 238.4 |
| tic-tac-toe | PFoil:two-level (voting) | 99.48% | 238.4 |
| tic-tac-toe | PFossil:one-level (best rule) | 97.49% | 180.1 |
| tic-tac-toe | PFossil:one-level (macro distribution voting) | 97.80% | 180.1 |
| tic-tac-toe | PFossil:one-level (micro distribution voting) | 84.66% | 180.1 |
| tic-tac-toe | PFossil:one-level (voting) | 93.74% | 180.1 |
| tic-tac-toe | PFossil:single (best rule) | 97.71% | 180.1 |
| tic-tac-toe | PFossil:single (macro distribution voting) | 95.10% | 180.1 |
| tic-tac-toe | PFossil:single (micro distribution voting) | 83.72% | 180.1 |
| tic-tac-toe | PFossil:single (voting) | 85.29% | 180.1 |
| tic-tac-toe | PFossil:two-level (best rule) | 98.85% | 180.1 |
| tic-tac-toe | PFossil:two-level (macro distribution voting) | 98.23% | 180.1 |
| tic-tac-toe | PFossil:two-level (micro distribution voting) | 87.79% | 180.1 |
| tic-tac-toe | PFossil:two-level (voting) | 89.77% | 180.1 |
| tic-tac-toe | Pypper:one-level (best rule) | 97.08% | 158.2 |
| tic-tac-toe | Pypper:one-level (macro distribution voting) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:one-level (micro distribution voting) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:one-level (voting) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:single (best rule) | 96.25% | 158.2 |
| tic-tac-toe | Pypper:single (macro distribution voting) | 96.25% | 158.2 |
| tic-tac-toe | Pypper:single (micro distribution voting) | 96.25% | 158.2 |
| tic-tac-toe | Pypper:single (voting) | 96.25% | 158.2 |
| tic-tac-toe | Pypper:two-level (best rule) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:two-level (macro distribution voting) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:two-level (micro distribution voting) | 98.33% | 158.2 |
| tic-tac-toe | Pypper:two-level (voting) | 98.33% | 158.2 |
| vehicle | PFoil:one-level (best rule) | 69.99% | 740.8 |
| vehicle | PFoil:one-level (macro distribution voting) | 74.48% | 740.8 |
| vehicle | PFoil:one-level (micro distribution voting) | 72.12% | 740.8 |
| vehicle | PFoil:one-level (voting) | 74.60% | 740.8 |
| vehicle | PFoil:single (best rule) | 71.87% | 740.8 |
| vehicle | PFoil:single (macro distribution voting) | 71.41% | 740.8 |
| vehicle | PFoil:single (micro distribution voting) | 71.28% | 740.8 |
| vehicle | PFoil:single (voting) | 68.92% | 740.8 |
| vehicle | PFoil:two-level (best rule) | 74.12% | 740.8 |
| vehicle | PFoil:two-level (macro distribution voting) | 73.77% | 740.8 |
| vehicle | PFoil:two-level (micro distribution voting) | 73.77% | 740.8 |
| vehicle | PFoil:two-level (voting) | 73.41% | 740.8 |
| vehicle | PFossil:one-level (best rule) | 71.04% | 236.3 |
| vehicle | PFossil:one-level (macro distribution voting) | 74.71% | 236.3 |
| vehicle | PFossil:one-level (micro distribution voting) | 72.94% | 236.3 |
| vehicle | PFossil:one-level (voting) | 74.00% | 236.3 |
| vehicle | PFossil:single (best rule) | 71.64% | 236.3 |
| vehicle | PFossil:single (macro distribution voting) | 72.82% | 236.3 |
| vehicle | PFossil:single (micro distribution voting) | 70.93% | 236.3 |
| vehicle | PFossil:single (voting) | 71.53% | 236.3 |
| vehicle | PFossil:two-level (best rule) | 73.77% | 236.3 |
| vehicle | PFossil:two-level (macro distribution voting) | 74.36% | 236.3 |
| vehicle | PFossil:two-level (micro distribution voting) | 73.29% | 236.3 |
| vehicle | PFossil:two-level (voting) | 74.47% | 236.3 |
| vehicle | Pypper:one-level (best rule) | 71.05% | 127.9 |
| vehicle | Pypper:one-level (macro distribution voting) | 73.30% | 127.9 |
| vehicle | Pypper:one-level (micro distribution voting) | 71.77% | 127.9 |
| vehicle | Pypper:one-level (voting) | 73.54% | 127.9 |
| vehicle | Pypper:single (best rule) | 63.84% | 127.9 |
| vehicle | Pypper:single (macro distribution voting) | 64.19% | 127.9 |
| vehicle | Pypper:single (micro distribution voting) | 62.30% | 127.9 |
| vehicle | Pypper:single (voting) | 63.72% | 127.9 |
| vehicle | Pypper:two-level (best rule) | 73.66% | 127.9 |
| vehicle | Pypper:two-level (macro distribution voting) | 73.77% | 127.9 |
| vehicle | Pypper:two-level (micro distribution voting) | 73.30% | 127.9 |
| vehicle | Pypper:two-level (voting) | 73.89% | 127.9 |
| vote | PFoil:one-level (best rule) | 94.49% | 176.1 |
| vote | PFoil:one-level (macro distribution voting) | 95.87% | 176.1 |
| vote | PFoil:one-level (micro distribution voting) | 95.18% | 176.1 |
| vote | PFoil:one-level (voting) | 95.64% | 176.1 |
| vote | PFoil:single (best rule) | 94.71% | 176.1 |
| vote | PFoil:single (macro distribution voting) | 94.71% | 176.1 |
| vote | PFoil:single (micro distribution voting) | 94.03% | 176.1 |
| vote | PFoil:single (voting) | 95.17% | 176.1 |
| vote | PFoil:two-level (best rule) | 95.41% | 176.1 |
| vote | PFoil:two-level (macro distribution voting) | 96.33% | 176.1 |
| vote | PFoil:two-level (micro distribution voting) | 95.19% | 176.1 |
| vote | PFoil:two-level (voting) | 95.86% | 176.1 |
| vote | PFossil:one-level (best rule) | 93.55% | 74.0 |
| vote | PFossil:one-level (macro distribution voting) | 95.65% | 74.0 |
| vote | PFossil:one-level (micro distribution voting) | 95.19% | 74.0 |
| vote | PFossil:one-level (voting) | 95.42% | 74.0 |
| vote | PFossil:single (best rule) | 95.18% | 74.0 |
| vote | PFossil:single (macro distribution voting) | 93.57% | 74.0 |
| vote | PFossil:single (micro distribution voting) | 95.42% | 74.0 |
| vote | PFossil:single (voting) | 94.03% | 74.0 |
| vote | PFossil:two-level (best rule) | 95.63% | 74.0 |
| vote | PFossil:two-level (macro distribution voting) | 95.17% | 74.0 |
| vote | PFossil:two-level (micro distribution voting) | 95.19% | 74.0 |
| vote | PFossil:two-level (voting) | 94.02% | 74.0 |
| vote | Pypper:one-level (best rule) | 94.48% | 32.7 |
| vote | Pypper:one-level (macro distribution voting) | 95.42% | 32.7 |
| vote | Pypper:one-level (micro distribution voting) | 95.65% | 32.7 |
| vote | Pypper:one-level (voting) | 95.65% | 32.7 |
| vote | Pypper:single (best rule) | 95.18% | 32.7 |
| vote | Pypper:single (macro distribution voting) | 94.95% | 32.7 |
| vote | Pypper:single (micro distribution voting) | 95.18% | 32.7 |
| vote | Pypper:single (voting) | 94.95% | 32.7 |
| vote | Pypper:two-level (best rule) | 95.42% | 32.7 |
| vote | Pypper:two-level (macro distribution voting) | 95.42% | 32.7 |
| vote | Pypper:two-level (micro distribution voting) | 95.65% | 32.7 |
| vote | Pypper:two-level (voting) | 95.18% | 32.7 |
| vowel | PFoil:one-level (best rule) | 73.64% | 1017.7 |
| vowel | PFoil:one-level (macro distribution voting) | 88.69% | 1017.7 |
| vowel | PFoil:one-level (micro distribution voting) | 82.02% | 1017.7 |
| vowel | PFoil:one-level (voting) | 88.79% | 1017.7 |
| vowel | PFoil:single (best rule) | 81.82% | 1017.7 |
| vowel | PFoil:single (macro distribution voting) | 82.73% | 1017.7 |
| vowel | PFoil:single (micro distribution voting) | 73.54% | 1017.7 |
| vowel | PFoil:single (voting) | 72.73% | 1017.7 |
| vowel | PFoil:two-level (best rule) | 87.98% | 1017.7 |
| vowel | PFoil:two-level (macro distribution voting) | 87.68% | 1017.7 |
| vowel | PFoil:two-level (micro distribution voting) | 87.17% | 1017.7 |
| vowel | PFoil:two-level (voting) | 87.37% | 1017.7 |
| vowel | PFossil:one-level (best rule) | 76.87% | 871.1 |
| vowel | PFossil:one-level (macro distribution voting) | 87.37% | 871.1 |
| vowel | PFossil:one-level (micro distribution voting) | 78.28% | 871.1 |
| vowel | PFossil:one-level (voting) | 87.17% | 871.1 |
| vowel | PFossil:single (best rule) | 77.27% | 871.1 |
| vowel | PFossil:single (macro distribution voting) | 79.80% | 871.1 |
| vowel | PFossil:single (micro distribution voting) | 74.24% | 871.1 |
| vowel | PFossil:single (voting) | 78.89% | 871.1 |
| vowel | PFossil:two-level (best rule) | 85.56% | 871.1 |
| vowel | PFossil:two-level (macro distribution voting) | 87.37% | 871.1 |
| vowel | PFossil:two-level (micro distribution voting) | 83.43% | 871.1 |
| vowel | PFossil:two-level (voting) | 86.57% | 871.1 |
| vowel | Pypper:one-level (best rule) | 71.62% | 295.1 |
| vowel | Pypper:one-level (macro distribution voting) | 75.66% | 295.1 |
| vowel | Pypper:one-level (micro distribution voting) | 69.09% | 295.1 |
| vowel | Pypper:one-level (voting) | 74.24% | 295.1 |
| vowel | Pypper:single (best rule) | 63.84% | 295.1 |
| vowel | Pypper:single (macro distribution voting) | 64.04% | 295.1 |
| vowel | Pypper:single (micro distribution voting) | 62.53% | 295.1 |
| vowel | Pypper:single (voting) | 62.53% | 295.1 |
| vowel | Pypper:two-level (best rule) | 74.85% | 295.1 |
| vowel | Pypper:two-level (macro distribution voting) | 75.15% | 295.1 |
| vowel | Pypper:two-level (micro distribution voting) | 73.13% | 295.1 |
| vowel | Pypper:two-level (voting) | 74.44% | 295.1 |
| wdbc | PFoil:one-level (best rule) | 93.33% | 124.4 |
| wdbc | PFoil:one-level (macro distribution voting) | 96.66% | 124.4 |
| wdbc | PFoil:one-level (micro distribution voting) | 96.48% | 124.4 |
| wdbc | PFoil:one-level (voting) | 97.01% | 124.4 |
| wdbc | PFoil:single (best rule) | 94.38% | 124.4 |
| wdbc | PFoil:single (macro distribution voting) | 94.38% | 124.4 |
| wdbc | PFoil:single (micro distribution voting) | 93.67% | 124.4 |
| wdbc | PFoil:single (voting) | 93.85% | 124.4 |
| wdbc | PFoil:two-level (best rule) | 95.96% | 124.4 |
| wdbc | PFoil:two-level (macro distribution voting) | 96.48% | 124.4 |
| wdbc | PFoil:two-level (micro distribution voting) | 95.60% | 124.4 |
| wdbc | PFoil:two-level (voting) | 96.31% | 124.4 |
| wdbc | PFossil:one-level (best rule) | 95.44% | 106.1 |
| wdbc | PFossil:one-level (macro distribution voting) | 95.78% | 106.1 |
| wdbc | PFossil:one-level (micro distribution voting) | 94.55% | 106.1 |
| wdbc | PFossil:one-level (voting) | 95.60% | 106.1 |
| wdbc | PFossil:single (best rule) | 94.56% | 106.1 |
| wdbc | PFossil:single (macro distribution voting) | 94.91% | 106.1 |
| wdbc | PFossil:single (micro distribution voting) | 93.85% | 106.1 |
| wdbc | PFossil:single (voting) | 94.56% | 106.1 |
| wdbc | PFossil:two-level (best rule) | 95.43% | 106.1 |
| wdbc | PFossil:two-level (macro distribution voting) | 95.43% | 106.1 |
| wdbc | PFossil:two-level (micro distribution voting) | 94.37% | 106.1 |
| wdbc | PFossil:two-level (voting) | 95.08% | 106.1 |
| wdbc | Pypper:one-level (best rule) | 95.08% | 55.7 |
| wdbc | Pypper:one-level (macro distribution voting) | 95.96% | 55.7 |
| wdbc | Pypper:one-level (micro distribution voting) | 94.56% | 55.7 |
| wdbc | Pypper:one-level (voting) | 95.96% | 55.7 |
| wdbc | Pypper:single (best rule) | 94.91% | 55.7 |
| wdbc | Pypper:single (macro distribution voting) | 95.08% | 55.7 |
| wdbc | Pypper:single (micro distribution voting) | 94.90% | 55.7 |
| wdbc | Pypper:single (voting) | 94.72% | 55.7 |
| wdbc | Pypper:two-level (best rule) | 95.08% | 55.7 |
| wdbc | Pypper:two-level (macro distribution voting) | 95.25% | 55.7 |
| wdbc | Pypper:two-level (micro distribution voting) | 95.08% | 55.7 |
| wdbc | Pypper:two-level (voting) | 95.08% | 55.7 |
| wilt | PFoil:one-level (best rule) | 96.73% | 889.6 |
| wilt | PFoil:one-level (macro distribution voting) | 96.94% | 889.6 |
| wilt | PFoil:one-level (micro distribution voting) | 95.31% | 889.6 |
| wilt | PFoil:one-level (voting) | 94.88% | 889.6 |
| wilt | PFoil:single (best rule) | 96.55% | 889.6 |
| wilt | PFoil:single (macro distribution voting) | 96.98% | 889.6 |
| wilt | PFoil:single (micro distribution voting) | 95.85% | 889.6 |
| wilt | PFoil:single (voting) | 93.82% | 889.6 |
| wilt | PFoil:two-level (best rule) | 96.88% | 889.6 |
| wilt | PFoil:two-level (macro distribution voting) | 96.92% | 889.6 |
| wilt | PFoil:two-level (micro distribution voting) | 96.32% | 889.6 |
| wilt | PFoil:two-level (voting) | 96.01% | 889.6 |
| wilt | PFossil:one-level (best rule) | 97.02% | 10.3 |
| wilt | PFossil:one-level (macro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:one-level (micro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:one-level (voting) | 97.02% | 10.3 |
| wilt | PFossil:single (best rule) | 97.02% | 10.3 |
| wilt | PFossil:single (macro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:single (micro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:single (voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level (best rule) | 97.02% | 10.3 |
| wilt | PFossil:two-level (macro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level (micro distribution voting) | 97.02% | 10.3 |
| wilt | PFossil:two-level (voting) | 97.02% | 10.3 |
| wilt | Pypper:one-level (best rule) | 97.40% | 56.8 |
| wilt | Pypper:one-level (macro distribution voting) | 97.38% | 56.8 |
| wilt | Pypper:one-level (micro distribution voting) | 97.40% | 56.8 |
| wilt | Pypper:one-level (voting) | 97.40% | 56.8 |
| wilt | Pypper:single (best rule) | 97.42% | 56.8 |
| wilt | Pypper:single (macro distribution voting) | 97.42% | 56.8 |
| wilt | Pypper:single (micro distribution voting) | 97.42% | 56.8 |
| wilt | Pypper:single (voting) | 97.42% | 56.8 |
| wilt | Pypper:two-level (best rule) | 97.38% | 56.8 |
| wilt | Pypper:two-level (macro distribution voting) | 97.38% | 56.8 |
| wilt | Pypper:two-level (micro distribution voting) | 97.38% | 56.8 |
| wilt | Pypper:two-level (voting) | 97.38% | 56.8 |
| wine | PFoil:one-level (best rule) | 92.65% | 74.5 |
| wine | PFoil:one-level (macro distribution voting) | 96.63% | 74.5 |
| wine | PFoil:one-level (micro distribution voting) | 96.05% | 74.5 |
| wine | PFoil:one-level (voting) | 96.63% | 74.5 |
| wine | PFoil:single (best rule) | 90.98% | 74.5 |
| wine | PFoil:single (macro distribution voting) | 91.54% | 74.5 |
| wine | PFoil:single (micro distribution voting) | 90.98% | 74.5 |
| wine | PFoil:single (voting) | 90.42% | 74.5 |
| wine | PFoil:two-level (best rule) | 95.46% | 74.5 |
| wine | PFoil:two-level (macro distribution voting) | 96.60% | 74.5 |
| wine | PFoil:two-level (micro distribution voting) | 96.05% | 74.5 |
| wine | PFoil:two-level (voting) | 96.60% | 74.5 |
| wine | PFossil:one-level (best rule) | 90.98% | 77.0 |
| wine | PFossil:one-level (macro distribution voting) | 96.63% | 77.0 |
| wine | PFossil:one-level (micro distribution voting) | 96.05% | 77.0 |
| wine | PFossil:one-level (voting) | 96.63% | 77.0 |
| wine | PFossil:single (best rule) | 92.75% | 77.0 |
| wine | PFossil:single (macro distribution voting) | 93.82% | 77.0 |
| wine | PFossil:single (micro distribution voting) | 93.24% | 77.0 |
| wine | PFossil:single (voting) | 93.79% | 77.0 |
| wine | PFossil:two-level (best rule) | 96.60% | 77.0 |
| wine | PFossil:two-level (macro distribution voting) | 96.60% | 77.0 |
| wine | PFossil:two-level (micro distribution voting) | 96.60% | 77.0 |
| wine | PFossil:two-level (voting) | 96.05% | 77.0 |
| wine | Pypper:one-level (best rule) | 92.58% | 49.3 |
| wine | Pypper:one-level (macro distribution voting) | 97.75% | 49.3 |
| wine | Pypper:one-level (micro distribution voting) | 96.60% | 49.3 |
| wine | Pypper:one-level (voting) | 97.75% | 49.3 |
| wine | Pypper:single (best rule) | 92.09% | 49.3 |
| wine | Pypper:single (macro distribution voting) | 90.98% | 49.3 |
| wine | Pypper:single (micro distribution voting) | 90.92% | 49.3 |
| wine | Pypper:single (voting) | 88.69% | 49.3 |
| wine | Pypper:two-level (best rule) | 96.60% | 49.3 |
| wine | Pypper:two-level (macro distribution voting) | 97.16% | 49.3 |
| wine | Pypper:two-level (micro distribution voting) | 95.42% | 49.3 |
| wine | Pypper:two-level (voting) | 94.87% | 49.3 |
| yeast | PFoil:one-level (best rule) | 58.16% | 2029.6 |
| yeast | PFoil:one-level (macro distribution voting) | 59.03% | 2029.6 |
| yeast | PFoil:one-level (micro distribution voting) | 49.47% | 2029.6 |
| yeast | PFoil:one-level (voting) | 58.22% | 2029.6 |
| yeast | PFoil:single (best rule) | 57.41% | 2029.6 |
| yeast | PFoil:single (macro distribution voting) | 56.88% | 2029.6 |
| yeast | PFoil:single (micro distribution voting) | 47.64% | 2029.6 |
| yeast | PFoil:single (voting) | 52.29% | 2029.6 |
| yeast | PFoil:two-level (best rule) | 59.77% | 2029.6 |
| yeast | PFoil:two-level (macro distribution voting) | 59.43% | 2029.6 |
| yeast | PFoil:two-level (micro distribution voting) | 55.80% | 2029.6 |
| yeast | PFoil:two-level (voting) | 55.46% | 2029.6 |
| yeast | PFossil:one-level (best rule) | 55.86% | 435.4 |
| yeast | PFossil:one-level (macro distribution voting) | 60.31% | 435.4 |
| yeast | PFossil:one-level (micro distribution voting) | 57.14% | 435.4 |
| yeast | PFossil:one-level (voting) | 60.17% | 435.4 |
| yeast | PFossil:single (best rule) | 57.48% | 435.4 |
| yeast | PFossil:single (macro distribution voting) | 57.54% | 435.4 |
| yeast | PFossil:single (micro distribution voting) | 55.52% | 435.4 |
| yeast | PFossil:single (voting) | 54.31% | 435.4 |
| yeast | PFossil:two-level (best rule) | 60.31% | 435.4 |
| yeast | PFossil:two-level (macro distribution voting) | 60.24% | 435.4 |
| yeast | PFossil:two-level (micro distribution voting) | 58.22% | 435.4 |
| yeast | PFossil:two-level (voting) | 57.61% | 435.4 |
| yeast | Pypper:one-level (best rule) | 60.11% | 171.7 |
| yeast | Pypper:one-level (macro distribution voting) | 61.39% | 171.7 |
| yeast | Pypper:one-level (micro distribution voting) | 59.10% | 171.7 |
| yeast | Pypper:one-level (voting) | 61.39% | 171.7 |
| yeast | Pypper:single (best rule) | 59.30% | 171.7 |
| yeast | Pypper:single (macro distribution voting) | 59.30% | 171.7 |
| yeast | Pypper:single (micro distribution voting) | 58.62% | 171.7 |
| yeast | Pypper:single (voting) | 58.69% | 171.7 |
| yeast | Pypper:two-level (best rule) | 61.79% | 171.7 |
| yeast | Pypper:two-level (macro distribution voting) | 61.59% | 171.7 |
| yeast | Pypper:two-level (micro distribution voting) | 60.31% | 171.7 |
| yeast | Pypper:two-level (voting) | 60.04% | 171.7 |
| zoo | PFoil:one-level (best rule) | 93.00% | 85.5 |
| zoo | PFoil:one-level (macro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:one-level (micro distribution voting) | 94.00% | 85.5 |
| zoo | PFoil:one-level (voting) | 95.00% | 85.5 |
| zoo | PFoil:single (best rule) | 96.00% | 85.5 |
| zoo | PFoil:single (macro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:single (micro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:single (voting) | 95.00% | 85.5 |
| zoo | PFoil:two-level (best rule) | 95.00% | 85.5 |
| zoo | PFoil:two-level (macro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:two-level (micro distribution voting) | 95.00% | 85.5 |
| zoo | PFoil:two-level (voting) | 94.00% | 85.5 |
| zoo | PFossil:one-level (best rule) | 93.00% | 83.6 |
| zoo | PFossil:one-level (macro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:one-level (micro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:one-level (voting) | 95.00% | 83.6 |
| zoo | PFossil:single (best rule) | 95.00% | 83.6 |
| zoo | PFossil:single (macro distribution voting) | 92.00% | 83.6 |
| zoo | PFossil:single (micro distribution voting) | 94.00% | 83.6 |
| zoo | PFossil:single (voting) | 93.00% | 83.6 |
| zoo | PFossil:two-level (best rule) | 95.00% | 83.6 |
| zoo | PFossil:two-level (macro distribution voting) | 95.00% | 83.6 |
| zoo | PFossil:two-level (micro distribution voting) | 94.00% | 83.6 |
| zoo | PFossil:two-level (voting) | 94.00% | 83.6 |
| zoo | Pypper:one-level (best rule) | 90.09% | 55.6 |
| zoo | Pypper:one-level (macro distribution voting) | 92.09% | 55.6 |
| zoo | Pypper:one-level (micro distribution voting) | 90.09% | 55.6 |
| zoo | Pypper:one-level (voting) | 92.09% | 55.6 |
| zoo | Pypper:single (best rule) | 90.09% | 55.6 |
| zoo | Pypper:single (macro distribution voting) | 90.09% | 55.6 |
| zoo | Pypper:single (micro distribution voting) | 91.09% | 55.6 |
| zoo | Pypper:single (voting) | 90.09% | 55.6 |
| zoo | Pypper:two-level (best rule) | 92.09% | 55.6 |
| zoo | Pypper:two-level (macro distribution voting) | 92.09% | 55.6 |
| zoo | Pypper:two-level (micro distribution voting) | 91.09% | 55.6 |
| zoo | Pypper:two-level (voting) | 91.09% | 55.6 |

