# Base against composite models: bagged rule sets, combined at two levels or one

## Setup

**Full run.** Quick sanity check instead: `python demos/base_vs_composite.py`.

Bagging with rule sets as members: 50 concept sets -- one concept
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

Rules per ensemble: 688.8, conditions: 2655.0; rules of the single concept set: 16.0.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 82.76% (8.41) | 85.19% (4.40) | 83.46% (7.75) |
| voting | 82.33% (9.46) | 84.87% (4.91) | 85.14% (4.29) |
| micro distribution voting | 82.39% (9.53) | 84.78% (5.51) | 83.82% (7.88) |
| macro distribution voting | 82.76% (8.36) | 85.26% (4.08) | 85.40% (3.43) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## PFossil

Rules per ensemble: 1006.3, conditions: 4483.0; rules of the single concept set: 19.2.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 82.55% (7.98) | 84.75% (4.46) | 81.79% (7.93) |
| voting | 81.65% (9.23) | 83.87% (5.76) | 84.60% (4.77) |
| micro distribution voting | 81.48% (9.54) | 83.67% (6.02) | 83.26% (6.97) |
| macro distribution voting | 82.39% (8.03) | 84.76% (4.19) | 85.42% (3.11) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## PFoil

Rules per ensemble: 4287.9, conditions: 23643.7; rules of the single concept set: 99.0.

| conflict resolution | single | two-level | one-level |
|---|--:|--:|--:|
| best rule | 84.38% (8.01) | 86.42% (3.99) | 82.75% (8.48) |
| voting | 82.17% (10.44) | 85.87% (5.13) | 86.17% (4.60) |
| micro distribution voting | 82.62% (9.54) | 85.91% (5.46) | 84.59% (7.38) |
| macro distribution voting | 84.31% (7.99) | 86.44% (3.24) | 86.46% (3.75) |

Accuracy, and in parentheses the mean rank among the twelve variants.

## Two-level against one-level

Per (dataset, base learner) pair, the accuracy of the two-level combination minus that of the one-level one, in percentage points, and the number of pairs on which either is ahead.

| conflict resolution | mean difference | two-level ahead | one-level ahead | ties |
|---|--:|--:|--:|--:|
| best rule | +2.79 | 162 | 39 | 6 |
| voting | -0.44 | 69 | 114 | 24 |
| micro distribution voting | +0.90 | 146 | 42 | 19 |
| macro distribution voting | -0.27 | 85 | 98 | 24 |

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_rules |
|---|---|---|---|
| SPECT | PFoil:one-level (best rule) | 84.29% | 1140.2 |
| SPECT | PFoil:one-level (macro distribution voting) | 83.89% | 1140.2 |
| SPECT | PFoil:one-level (micro distribution voting) | 79.42% | 1140.2 |
| SPECT | PFoil:one-level (voting) | 82.35% | 1140.2 |
| SPECT | PFoil:single (best rule) | 82.05% | 1140.2 |
| SPECT | PFoil:single (macro distribution voting) | 80.94% | 1140.2 |
| SPECT | PFoil:single (micro distribution voting) | 76.81% | 1140.2 |
| SPECT | PFoil:single (voting) | 79.77% | 1140.2 |
| SPECT | PFoil:two-level (best rule) | 85.00% | 1140.2 |
| SPECT | PFoil:two-level (macro distribution voting) | 84.63% | 1140.2 |
| SPECT | PFoil:two-level (micro distribution voting) | 79.40% | 1140.2 |
| SPECT | PFoil:two-level (voting) | 84.23% | 1140.2 |
| SPECT | PFossil:one-level (best rule) | 84.29% | 547.6 |
| SPECT | PFossil:one-level (macro distribution voting) | 82.74% | 547.6 |
| SPECT | PFossil:one-level (micro distribution voting) | 83.87% | 547.6 |
| SPECT | PFossil:one-level (voting) | 80.54% | 547.6 |
| SPECT | PFossil:single (best rule) | 78.68% | 547.6 |
| SPECT | PFossil:single (macro distribution voting) | 77.55% | 547.6 |
| SPECT | PFossil:single (micro distribution voting) | 78.68% | 547.6 |
| SPECT | PFossil:single (voting) | 77.55% | 547.6 |
| SPECT | PFossil:two-level (best rule) | 80.90% | 547.6 |
| SPECT | PFossil:two-level (macro distribution voting) | 80.90% | 547.6 |
| SPECT | PFossil:two-level (micro distribution voting) | 81.27% | 547.6 |
| SPECT | PFossil:two-level (voting) | 81.27% | 547.6 |
| SPECT | Pypper:one-level (best rule) | 84.29% | 253.2 |
| SPECT | Pypper:one-level (macro distribution voting) | 84.25% | 253.2 |
| SPECT | Pypper:one-level (micro distribution voting) | 83.52% | 253.2 |
| SPECT | Pypper:one-level (voting) | 81.28% | 253.2 |
| SPECT | Pypper:single (best rule) | 79.02% | 253.2 |
| SPECT | Pypper:single (macro distribution voting) | 79.02% | 253.2 |
| SPECT | Pypper:single (micro distribution voting) | 79.02% | 253.2 |
| SPECT | Pypper:single (voting) | 79.02% | 253.2 |
| SPECT | Pypper:two-level (best rule) | 80.90% | 253.2 |
| SPECT | Pypper:two-level (macro distribution voting) | 80.90% | 253.2 |
| SPECT | Pypper:two-level (micro distribution voting) | 80.90% | 253.2 |
| SPECT | Pypper:two-level (voting) | 80.90% | 253.2 |
| Titanic | PFoil:one-level (best rule) | 77.92% | 4969.9 |
| Titanic | PFoil:one-level (macro distribution voting) | 79.07% | 4969.9 |
| Titanic | PFoil:one-level (micro distribution voting) | 75.79% | 4969.9 |
| Titanic | PFoil:one-level (voting) | 78.92% | 4969.9 |
| Titanic | PFoil:single (best rule) | 78.76% | 4969.9 |
| Titanic | PFoil:single (macro distribution voting) | 77.92% | 4969.9 |
| Titanic | PFoil:single (micro distribution voting) | 75.55% | 4969.9 |
| Titanic | PFoil:single (voting) | 76.78% | 4969.9 |
| Titanic | PFoil:two-level (best rule) | 79.07% | 4969.9 |
| Titanic | PFoil:two-level (macro distribution voting) | 79.14% | 4969.9 |
| Titanic | PFoil:two-level (micro distribution voting) | 77.69% | 4969.9 |
| Titanic | PFoil:two-level (voting) | 79.91% | 4969.9 |
| Titanic | PFossil:one-level (best rule) | 79.37% | 314.8 |
| Titanic | PFossil:one-level (macro distribution voting) | 78.76% | 314.8 |
| Titanic | PFossil:one-level (micro distribution voting) | 78.23% | 314.8 |
| Titanic | PFossil:one-level (voting) | 78.61% | 314.8 |
| Titanic | PFossil:single (best rule) | 77.77% | 314.8 |
| Titanic | PFossil:single (macro distribution voting) | 77.85% | 314.8 |
| Titanic | PFossil:single (micro distribution voting) | 76.78% | 314.8 |
| Titanic | PFossil:single (voting) | 78.15% | 314.8 |
| Titanic | PFossil:two-level (best rule) | 78.61% | 314.8 |
| Titanic | PFossil:two-level (macro distribution voting) | 78.68% | 314.8 |
| Titanic | PFossil:two-level (micro distribution voting) | 78.38% | 314.8 |
| Titanic | PFossil:two-level (voting) | 78.15% | 314.8 |
| Titanic | Pypper:one-level (best rule) | 79.75% | 336.9 |
| Titanic | Pypper:one-level (macro distribution voting) | 78.91% | 336.9 |
| Titanic | Pypper:one-level (micro distribution voting) | 78.76% | 336.9 |
| Titanic | Pypper:one-level (voting) | 79.14% | 336.9 |
| Titanic | Pypper:single (best rule) | 79.22% | 336.9 |
| Titanic | Pypper:single (macro distribution voting) | 79.30% | 336.9 |
| Titanic | Pypper:single (micro distribution voting) | 78.84% | 336.9 |
| Titanic | Pypper:single (voting) | 78.84% | 336.9 |
| Titanic | Pypper:two-level (best rule) | 79.14% | 336.9 |
| Titanic | Pypper:two-level (macro distribution voting) | 79.14% | 336.9 |
| Titanic | Pypper:two-level (micro distribution voting) | 79.07% | 336.9 |
| Titanic | Pypper:two-level (voting) | 78.99% | 336.9 |
| anneal | PFoil:one-level (best rule) | 96.88% | 1207.6 |
| anneal | PFoil:one-level (macro distribution voting) | 98.10% | 1207.6 |
| anneal | PFoil:one-level (micro distribution voting) | 93.20% | 1207.6 |
| anneal | PFoil:one-level (voting) | 98.88% | 1207.6 |
| anneal | PFoil:single (best rule) | 97.21% | 1207.6 |
| anneal | PFoil:single (macro distribution voting) | 97.66% | 1207.6 |
| anneal | PFoil:single (micro distribution voting) | 93.43% | 1207.6 |
| anneal | PFoil:single (voting) | 94.88% | 1207.6 |
| anneal | PFoil:two-level (best rule) | 99.11% | 1207.6 |
| anneal | PFoil:two-level (macro distribution voting) | 99.44% | 1207.6 |
| anneal | PFoil:two-level (micro distribution voting) | 99.11% | 1207.6 |
| anneal | PFoil:two-level (voting) | 99.11% | 1207.6 |
| anneal | PFossil:one-level (best rule) | 97.99% | 886.7 |
| anneal | PFossil:one-level (macro distribution voting) | 99.33% | 886.7 |
| anneal | PFossil:one-level (micro distribution voting) | 98.44% | 886.7 |
| anneal | PFossil:one-level (voting) | 98.00% | 886.7 |
| anneal | PFossil:single (best rule) | 99.22% | 886.7 |
| anneal | PFossil:single (macro distribution voting) | 99.66% | 886.7 |
| anneal | PFossil:single (micro distribution voting) | 98.88% | 886.7 |
| anneal | PFossil:single (voting) | 99.66% | 886.7 |
| anneal | PFossil:two-level (best rule) | 99.22% | 886.7 |
| anneal | PFossil:two-level (macro distribution voting) | 99.55% | 886.7 |
| anneal | PFossil:two-level (micro distribution voting) | 99.11% | 886.7 |
| anneal | PFossil:two-level (voting) | 99.55% | 886.7 |
| anneal | Pypper:one-level (best rule) | 96.77% | 594.3 |
| anneal | Pypper:one-level (macro distribution voting) | 98.77% | 594.3 |
| anneal | Pypper:one-level (micro distribution voting) | 97.32% | 594.3 |
| anneal | Pypper:one-level (voting) | 98.55% | 594.3 |
| anneal | Pypper:single (best rule) | 96.10% | 594.3 |
| anneal | Pypper:single (macro distribution voting) | 96.33% | 594.3 |
| anneal | Pypper:single (micro distribution voting) | 95.77% | 594.3 |
| anneal | Pypper:single (voting) | 96.10% | 594.3 |
| anneal | Pypper:two-level (best rule) | 98.66% | 594.3 |
| anneal | Pypper:two-level (macro distribution voting) | 98.66% | 594.3 |
| anneal | Pypper:two-level (micro distribution voting) | 98.55% | 594.3 |
| anneal | Pypper:two-level (voting) | 98.55% | 594.3 |
| audiology | PFoil:one-level (best rule) | 67.25% | 1876.5 |
| audiology | PFoil:one-level (macro distribution voting) | 83.70% | 1876.5 |
| audiology | PFoil:one-level (micro distribution voting) | 72.61% | 1876.5 |
| audiology | PFoil:one-level (voting) | 80.10% | 1876.5 |
| audiology | PFoil:single (best rule) | 77.04% | 1876.5 |
| audiology | PFoil:single (macro distribution voting) | 78.81% | 1876.5 |
| audiology | PFoil:single (micro distribution voting) | 61.09% | 1876.5 |
| audiology | PFoil:single (voting) | 62.41% | 1876.5 |
| audiology | PFoil:two-level (best rule) | 78.79% | 1876.5 |
| audiology | PFoil:two-level (macro distribution voting) | 81.44% | 1876.5 |
| audiology | PFoil:two-level (micro distribution voting) | 77.89% | 1876.5 |
| audiology | PFoil:two-level (voting) | 74.41% | 1876.5 |
| audiology | PFossil:one-level (best rule) | 65.47% | 1817.6 |
| audiology | PFossil:one-level (macro distribution voting) | 82.79% | 1817.6 |
| audiology | PFossil:one-level (micro distribution voting) | 72.57% | 1817.6 |
| audiology | PFossil:one-level (voting) | 82.79% | 1817.6 |
| audiology | PFossil:single (best rule) | 77.89% | 1817.6 |
| audiology | PFossil:single (macro distribution voting) | 77.00% | 1817.6 |
| audiology | PFossil:single (micro distribution voting) | 74.78% | 1817.6 |
| audiology | PFossil:single (voting) | 75.26% | 1817.6 |
| audiology | PFossil:two-level (best rule) | 81.90% | 1817.6 |
| audiology | PFossil:two-level (macro distribution voting) | 81.50% | 1817.6 |
| audiology | PFossil:two-level (micro distribution voting) | 80.12% | 1817.6 |
| audiology | PFossil:two-level (voting) | 80.18% | 1817.6 |
| audiology | Pypper:one-level (best rule) | 69.96% | 594.8 |
| audiology | Pypper:one-level (macro distribution voting) | 73.95% | 594.8 |
| audiology | Pypper:one-level (micro distribution voting) | 71.28% | 594.8 |
| audiology | Pypper:one-level (voting) | 73.04% | 594.8 |
| audiology | Pypper:single (best rule) | 64.64% | 594.8 |
| audiology | Pypper:single (macro distribution voting) | 65.10% | 594.8 |
| audiology | Pypper:single (micro distribution voting) | 63.75% | 594.8 |
| audiology | Pypper:single (voting) | 62.02% | 594.8 |
| audiology | Pypper:two-level (best rule) | 73.48% | 594.8 |
| audiology | Pypper:two-level (macro distribution voting) | 73.91% | 594.8 |
| audiology | Pypper:two-level (micro distribution voting) | 73.52% | 594.8 |
| audiology | Pypper:two-level (voting) | 74.39% | 594.8 |
| balance-scale | PFoil:one-level (best rule) | 84.95% | 3040.5 |
| balance-scale | PFoil:one-level (macro distribution voting) | 88.80% | 3040.5 |
| balance-scale | PFoil:one-level (micro distribution voting) | 88.01% | 3040.5 |
| balance-scale | PFoil:one-level (voting) | 84.64% | 3040.5 |
| balance-scale | PFoil:single (best rule) | 82.87% | 3040.5 |
| balance-scale | PFoil:single (macro distribution voting) | 82.54% | 3040.5 |
| balance-scale | PFoil:single (micro distribution voting) | 82.39% | 3040.5 |
| balance-scale | PFoil:single (voting) | 81.11% | 3040.5 |
| balance-scale | PFoil:two-level (best rule) | 86.40% | 3040.5 |
| balance-scale | PFoil:two-level (macro distribution voting) | 85.44% | 3040.5 |
| balance-scale | PFoil:two-level (micro distribution voting) | 86.56% | 3040.5 |
| balance-scale | PFoil:two-level (voting) | 85.76% | 3040.5 |
| balance-scale | PFossil:one-level (best rule) | 85.44% | 1222.8 |
| balance-scale | PFossil:one-level (macro distribution voting) | 87.68% | 1222.8 |
| balance-scale | PFossil:one-level (micro distribution voting) | 87.37% | 1222.8 |
| balance-scale | PFossil:one-level (voting) | 87.68% | 1222.8 |
| balance-scale | PFossil:single (best rule) | 81.60% | 1222.8 |
| balance-scale | PFossil:single (macro distribution voting) | 82.08% | 1222.8 |
| balance-scale | PFossil:single (micro distribution voting) | 80.47% | 1222.8 |
| balance-scale | PFossil:single (voting) | 79.67% | 1222.8 |
| balance-scale | PFossil:two-level (best rule) | 87.69% | 1222.8 |
| balance-scale | PFossil:two-level (macro distribution voting) | 87.20% | 1222.8 |
| balance-scale | PFossil:two-level (micro distribution voting) | 87.36% | 1222.8 |
| balance-scale | PFossil:two-level (voting) | 87.85% | 1222.8 |
| balance-scale | Pypper:one-level (best rule) | 85.12% | 585.6 |
| balance-scale | Pypper:one-level (macro distribution voting) | 88.01% | 585.6 |
| balance-scale | Pypper:one-level (micro distribution voting) | 87.37% | 585.6 |
| balance-scale | Pypper:one-level (voting) | 87.85% | 585.6 |
| balance-scale | Pypper:single (best rule) | 80.95% | 585.6 |
| balance-scale | Pypper:single (macro distribution voting) | 80.95% | 585.6 |
| balance-scale | Pypper:single (micro distribution voting) | 80.62% | 585.6 |
| balance-scale | Pypper:single (voting) | 80.47% | 585.6 |
| balance-scale | Pypper:two-level (best rule) | 87.53% | 585.6 |
| balance-scale | Pypper:two-level (macro distribution voting) | 87.53% | 585.6 |
| balance-scale | Pypper:two-level (micro distribution voting) | 87.21% | 585.6 |
| balance-scale | Pypper:two-level (voting) | 87.53% | 585.6 |
| banknote-authentication | PFoil:one-level (best rule) | 97.81% | 1045.3 |
| banknote-authentication | PFoil:one-level (macro distribution voting) | 98.98% | 1045.3 |
| banknote-authentication | PFoil:one-level (micro distribution voting) | 98.54% | 1045.3 |
| banknote-authentication | PFoil:one-level (voting) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:single (best rule) | 98.40% | 1045.3 |
| banknote-authentication | PFoil:single (macro distribution voting) | 98.76% | 1045.3 |
| banknote-authentication | PFoil:single (micro distribution voting) | 98.18% | 1045.3 |
| banknote-authentication | PFoil:single (voting) | 98.91% | 1045.3 |
| banknote-authentication | PFoil:two-level (best rule) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:two-level (macro distribution voting) | 99.20% | 1045.3 |
| banknote-authentication | PFoil:two-level (micro distribution voting) | 99.05% | 1045.3 |
| banknote-authentication | PFoil:two-level (voting) | 99.20% | 1045.3 |
| banknote-authentication | PFossil:one-level (best rule) | 98.62% | 586.5 |
| banknote-authentication | PFossil:one-level (macro distribution voting) | 98.62% | 586.5 |
| banknote-authentication | PFossil:one-level (micro distribution voting) | 94.46% | 586.5 |
| banknote-authentication | PFossil:one-level (voting) | 98.18% | 586.5 |
| banknote-authentication | PFossil:single (best rule) | 99.20% | 586.5 |
| banknote-authentication | PFossil:single (macro distribution voting) | 99.05% | 586.5 |
| banknote-authentication | PFossil:single (micro distribution voting) | 94.32% | 586.5 |
| banknote-authentication | PFossil:single (voting) | 97.89% | 586.5 |
| banknote-authentication | PFossil:two-level (best rule) | 99.20% | 586.5 |
| banknote-authentication | PFossil:two-level (macro distribution voting) | 99.20% | 586.5 |
| banknote-authentication | PFossil:two-level (micro distribution voting) | 94.32% | 586.5 |
| banknote-authentication | PFossil:two-level (voting) | 97.89% | 586.5 |
| banknote-authentication | Pypper:one-level (best rule) | 97.67% | 445.6 |
| banknote-authentication | Pypper:one-level (macro distribution voting) | 98.61% | 445.6 |
| banknote-authentication | Pypper:one-level (micro distribution voting) | 98.25% | 445.6 |
| banknote-authentication | Pypper:one-level (voting) | 98.61% | 445.6 |
| banknote-authentication | Pypper:single (best rule) | 98.54% | 445.6 |
| banknote-authentication | Pypper:single (macro distribution voting) | 98.61% | 445.6 |
| banknote-authentication | Pypper:single (micro distribution voting) | 98.54% | 445.6 |
| banknote-authentication | Pypper:single (voting) | 98.61% | 445.6 |
| banknote-authentication | Pypper:two-level (best rule) | 98.54% | 445.6 |
| banknote-authentication | Pypper:two-level (macro distribution voting) | 98.69% | 445.6 |
| banknote-authentication | Pypper:two-level (micro distribution voting) | 98.54% | 445.6 |
| banknote-authentication | Pypper:two-level (voting) | 98.61% | 445.6 |
| blood-transfusion-service-center | PFoil:one-level (best rule) | 77.00% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level (macro distribution voting) | 76.47% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level (micro distribution voting) | 76.21% | 2635.5 |
| blood-transfusion-service-center | PFoil:one-level (voting) | 73.26% | 2635.5 |
| blood-transfusion-service-center | PFoil:single (best rule) | 76.07% | 2635.5 |
| blood-transfusion-service-center | PFoil:single (macro distribution voting) | 75.94% | 2635.5 |
| blood-transfusion-service-center | PFoil:single (micro distribution voting) | 76.07% | 2635.5 |
| blood-transfusion-service-center | PFoil:single (voting) | 68.05% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level (best rule) | 76.74% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level (macro distribution voting) | 76.61% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level (micro distribution voting) | 76.61% | 2635.5 |
| blood-transfusion-service-center | PFoil:two-level (voting) | 75.93% | 2635.5 |
| blood-transfusion-service-center | PFossil:one-level (best rule) | 77.94% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level (macro distribution voting) | 79.40% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level (micro distribution voting) | 78.74% | 105.1 |
| blood-transfusion-service-center | PFossil:one-level (voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:single (best rule) | 76.86% | 105.1 |
| blood-transfusion-service-center | PFossil:single (macro distribution voting) | 76.86% | 105.1 |
| blood-transfusion-service-center | PFossil:single (micro distribution voting) | 76.86% | 105.1 |
| blood-transfusion-service-center | PFossil:single (voting) | 76.86% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level (best rule) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level (macro distribution voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level (micro distribution voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | PFossil:two-level (voting) | 68.99% | 105.1 |
| blood-transfusion-service-center | Pypper:one-level (best rule) | 77.27% | 216.2 |
| blood-transfusion-service-center | Pypper:one-level (macro distribution voting) | 78.60% | 216.2 |
| blood-transfusion-service-center | Pypper:one-level (micro distribution voting) | 77.94% | 216.2 |
| blood-transfusion-service-center | Pypper:one-level (voting) | 79.27% | 216.2 |
| blood-transfusion-service-center | Pypper:single (best rule) | 78.60% | 216.2 |
| blood-transfusion-service-center | Pypper:single (macro distribution voting) | 78.60% | 216.2 |
| blood-transfusion-service-center | Pypper:single (micro distribution voting) | 78.47% | 216.2 |
| blood-transfusion-service-center | Pypper:single (voting) | 78.47% | 216.2 |
| blood-transfusion-service-center | Pypper:two-level (best rule) | 79.27% | 216.2 |
| blood-transfusion-service-center | Pypper:two-level (macro distribution voting) | 79.27% | 216.2 |
| blood-transfusion-service-center | Pypper:two-level (micro distribution voting) | 79.27% | 216.2 |
| blood-transfusion-service-center | Pypper:two-level (voting) | 79.27% | 216.2 |
| breast-cancer | PFoil:one-level (best rule) | 72.04% | 1824.4 |
| breast-cancer | PFoil:one-level (macro distribution voting) | 74.11% | 1824.4 |
| breast-cancer | PFoil:one-level (micro distribution voting) | 73.45% | 1824.4 |
| breast-cancer | PFoil:one-level (voting) | 74.80% | 1824.4 |
| breast-cancer | PFoil:single (best rule) | 72.02% | 1824.4 |
| breast-cancer | PFoil:single (macro distribution voting) | 72.73% | 1824.4 |
| breast-cancer | PFoil:single (micro distribution voting) | 72.71% | 1824.4 |
| breast-cancer | PFoil:single (voting) | 70.94% | 1824.4 |
| breast-cancer | PFoil:two-level (best rule) | 72.73% | 1824.4 |
| breast-cancer | PFoil:two-level (macro distribution voting) | 74.11% | 1824.4 |
| breast-cancer | PFoil:two-level (micro distribution voting) | 74.14% | 1824.4 |
| breast-cancer | PFoil:two-level (voting) | 72.03% | 1824.4 |
| breast-cancer | PFossil:one-level (best rule) | 72.35% | 590.1 |
| breast-cancer | PFossil:one-level (macro distribution voting) | 75.52% | 590.1 |
| breast-cancer | PFossil:one-level (micro distribution voting) | 73.78% | 590.1 |
| breast-cancer | PFossil:one-level (voting) | 73.77% | 590.1 |
| breast-cancer | PFossil:single (best rule) | 69.26% | 590.1 |
| breast-cancer | PFossil:single (macro distribution voting) | 69.98% | 590.1 |
| breast-cancer | PFossil:single (micro distribution voting) | 68.56% | 590.1 |
| breast-cancer | PFossil:single (voting) | 68.92% | 590.1 |
| breast-cancer | PFossil:two-level (best rule) | 74.48% | 590.1 |
| breast-cancer | PFossil:two-level (macro distribution voting) | 74.47% | 590.1 |
| breast-cancer | PFossil:two-level (micro distribution voting) | 74.48% | 590.1 |
| breast-cancer | PFossil:two-level (voting) | 74.11% | 590.1 |
| breast-cancer | Pypper:one-level (best rule) | 72.72% | 188.5 |
| breast-cancer | Pypper:one-level (macro distribution voting) | 75.86% | 188.5 |
| breast-cancer | Pypper:one-level (micro distribution voting) | 72.72% | 188.5 |
| breast-cancer | Pypper:one-level (voting) | 74.48% | 188.5 |
| breast-cancer | Pypper:single (best rule) | 74.84% | 188.5 |
| breast-cancer | Pypper:single (macro distribution voting) | 74.84% | 188.5 |
| breast-cancer | Pypper:single (micro distribution voting) | 75.20% | 188.5 |
| breast-cancer | Pypper:single (voting) | 75.20% | 188.5 |
| breast-cancer | Pypper:two-level (best rule) | 74.48% | 188.5 |
| breast-cancer | Pypper:two-level (macro distribution voting) | 74.48% | 188.5 |
| breast-cancer | Pypper:two-level (micro distribution voting) | 74.84% | 188.5 |
| breast-cancer | Pypper:two-level (voting) | 74.84% | 188.5 |
| breast-w | PFoil:one-level (best rule) | 90.42% | 734.6 |
| breast-w | PFoil:one-level (macro distribution voting) | 96.42% | 734.6 |
| breast-w | PFoil:one-level (micro distribution voting) | 95.14% | 734.6 |
| breast-w | PFoil:one-level (voting) | 96.57% | 734.6 |
| breast-w | PFoil:single (best rule) | 95.85% | 734.6 |
| breast-w | PFoil:single (macro distribution voting) | 95.99% | 734.6 |
| breast-w | PFoil:single (micro distribution voting) | 95.42% | 734.6 |
| breast-w | PFoil:single (voting) | 94.70% | 734.6 |
| breast-w | PFoil:two-level (best rule) | 96.14% | 734.6 |
| breast-w | PFoil:two-level (macro distribution voting) | 96.42% | 734.6 |
| breast-w | PFoil:two-level (micro distribution voting) | 95.99% | 734.6 |
| breast-w | PFoil:two-level (voting) | 96.00% | 734.6 |
| breast-w | PFossil:one-level (best rule) | 91.42% | 608.5 |
| breast-w | PFossil:one-level (macro distribution voting) | 96.42% | 608.5 |
| breast-w | PFossil:one-level (micro distribution voting) | 95.71% | 608.5 |
| breast-w | PFossil:one-level (voting) | 96.28% | 608.5 |
| breast-w | PFossil:single (best rule) | 94.13% | 608.5 |
| breast-w | PFossil:single (macro distribution voting) | 94.99% | 608.5 |
| breast-w | PFossil:single (micro distribution voting) | 94.27% | 608.5 |
| breast-w | PFossil:single (voting) | 94.99% | 608.5 |
| breast-w | PFossil:two-level (best rule) | 95.85% | 608.5 |
| breast-w | PFossil:two-level (macro distribution voting) | 95.99% | 608.5 |
| breast-w | PFossil:two-level (micro distribution voting) | 95.85% | 608.5 |
| breast-w | PFossil:two-level (voting) | 95.99% | 608.5 |
| breast-w | Pypper:one-level (best rule) | 91.85% | 280.1 |
| breast-w | Pypper:one-level (macro distribution voting) | 96.71% | 280.1 |
| breast-w | Pypper:one-level (micro distribution voting) | 95.42% | 280.1 |
| breast-w | Pypper:one-level (voting) | 96.71% | 280.1 |
| breast-w | Pypper:single (best rule) | 94.57% | 280.1 |
| breast-w | Pypper:single (macro distribution voting) | 95.00% | 280.1 |
| breast-w | Pypper:single (micro distribution voting) | 94.57% | 280.1 |
| breast-w | Pypper:single (voting) | 95.00% | 280.1 |
| breast-w | Pypper:two-level (best rule) | 95.85% | 280.1 |
| breast-w | Pypper:two-level (macro distribution voting) | 96.42% | 280.1 |
| breast-w | Pypper:two-level (micro distribution voting) | 95.85% | 280.1 |
| breast-w | Pypper:two-level (voting) | 96.28% | 280.1 |
| car | PFoil:one-level (best rule) | 94.50% | 3447.5 |
| car | PFoil:one-level (macro distribution voting) | 96.93% | 3447.5 |
| car | PFoil:one-level (micro distribution voting) | 93.29% | 3447.5 |
| car | PFoil:one-level (voting) | 96.24% | 3447.5 |
| car | PFoil:single (best rule) | 96.70% | 3447.5 |
| car | PFoil:single (macro distribution voting) | 96.35% | 3447.5 |
| car | PFoil:single (micro distribution voting) | 95.60% | 3447.5 |
| car | PFoil:single (voting) | 94.73% | 3447.5 |
| car | PFoil:two-level (best rule) | 97.34% | 3447.5 |
| car | PFoil:two-level (macro distribution voting) | 97.22% | 3447.5 |
| car | PFoil:two-level (micro distribution voting) | 96.93% | 3447.5 |
| car | PFoil:two-level (voting) | 96.76% | 3447.5 |
| car | PFossil:one-level (best rule) | 87.04% | 917.2 |
| car | PFossil:one-level (macro distribution voting) | 92.48% | 917.2 |
| car | PFossil:one-level (micro distribution voting) | 87.09% | 917.2 |
| car | PFossil:one-level (voting) | 91.95% | 917.2 |
| car | PFossil:single (best rule) | 93.87% | 917.2 |
| car | PFossil:single (macro distribution voting) | 93.87% | 917.2 |
| car | PFossil:single (micro distribution voting) | 84.90% | 917.2 |
| car | PFossil:single (voting) | 90.51% | 917.2 |
| car | PFossil:two-level (best rule) | 95.08% | 917.2 |
| car | PFossil:two-level (macro distribution voting) | 94.16% | 917.2 |
| car | PFossil:two-level (micro distribution voting) | 85.59% | 917.2 |
| car | PFossil:two-level (voting) | 90.80% | 917.2 |
| car | Pypper:one-level (best rule) | 94.79% | 953.0 |
| car | Pypper:one-level (macro distribution voting) | 94.44% | 953.0 |
| car | Pypper:one-level (micro distribution voting) | 90.97% | 953.0 |
| car | Pypper:one-level (voting) | 93.40% | 953.0 |
| car | Pypper:single (best rule) | 91.32% | 953.0 |
| car | Pypper:single (macro distribution voting) | 91.15% | 953.0 |
| car | Pypper:single (micro distribution voting) | 89.99% | 953.0 |
| car | Pypper:single (voting) | 90.39% | 953.0 |
| car | Pypper:two-level (best rule) | 95.14% | 953.0 |
| car | Pypper:two-level (macro distribution voting) | 94.85% | 953.0 |
| car | Pypper:two-level (micro distribution voting) | 93.34% | 953.0 |
| car | Pypper:two-level (voting) | 93.98% | 953.0 |
| churn | PFoil:one-level (best rule) | 89.28% | 7208.5 |
| churn | PFoil:one-level (macro distribution voting) | 93.00% | 7208.5 |
| churn | PFoil:one-level (micro distribution voting) | 85.88% | 7208.5 |
| churn | PFoil:one-level (voting) | 94.02% | 7208.5 |
| churn | PFoil:single (best rule) | 92.36% | 7208.5 |
| churn | PFoil:single (macro distribution voting) | 91.58% | 7208.5 |
| churn | PFoil:single (micro distribution voting) | 89.94% | 7208.5 |
| churn | PFoil:single (voting) | 89.86% | 7208.5 |
| churn | PFoil:two-level (best rule) | 93.62% | 7208.5 |
| churn | PFoil:two-level (macro distribution voting) | 93.52% | 7208.5 |
| churn | PFoil:two-level (micro distribution voting) | 92.20% | 7208.5 |
| churn | PFoil:two-level (voting) | 93.88% | 7208.5 |
| churn | PFossil:one-level (best rule) | 92.18% | 557.7 |
| churn | PFossil:one-level (macro distribution voting) | 91.88% | 557.7 |
| churn | PFossil:one-level (micro distribution voting) | 88.54% | 557.7 |
| churn | PFossil:one-level (voting) | 91.74% | 557.7 |
| churn | PFossil:single (best rule) | 90.00% | 557.7 |
| churn | PFossil:single (macro distribution voting) | 89.98% | 557.7 |
| churn | PFossil:single (micro distribution voting) | 89.18% | 557.7 |
| churn | PFossil:single (voting) | 89.00% | 557.7 |
| churn | PFossil:two-level (best rule) | 91.70% | 557.7 |
| churn | PFossil:two-level (macro distribution voting) | 91.74% | 557.7 |
| churn | PFossil:two-level (micro distribution voting) | 90.68% | 557.7 |
| churn | PFossil:two-level (voting) | 90.76% | 557.7 |
| churn | Pypper:one-level (best rule) | 93.00% | 782.2 |
| churn | Pypper:one-level (macro distribution voting) | 93.74% | 782.2 |
| churn | Pypper:one-level (micro distribution voting) | 91.26% | 782.2 |
| churn | Pypper:one-level (voting) | 93.72% | 782.2 |
| churn | Pypper:single (best rule) | 93.42% | 782.2 |
| churn | Pypper:single (macro distribution voting) | 93.42% | 782.2 |
| churn | Pypper:single (micro distribution voting) | 93.36% | 782.2 |
| churn | Pypper:single (voting) | 93.28% | 782.2 |
| churn | Pypper:two-level (best rule) | 93.64% | 782.2 |
| churn | Pypper:two-level (macro distribution voting) | 93.66% | 782.2 |
| churn | Pypper:two-level (micro distribution voting) | 93.56% | 782.2 |
| churn | Pypper:two-level (voting) | 93.54% | 782.2 |
| climate-model-simulation-crashes | PFoil:one-level (best rule) | 91.48% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level (macro distribution voting) | 93.70% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level (micro distribution voting) | 91.67% | 774.0 |
| climate-model-simulation-crashes | PFoil:one-level (voting) | 93.70% | 774.0 |
| climate-model-simulation-crashes | PFoil:single (best rule) | 93.33% | 774.0 |
| climate-model-simulation-crashes | PFoil:single (macro distribution voting) | 93.89% | 774.0 |
| climate-model-simulation-crashes | PFoil:single (micro distribution voting) | 93.33% | 774.0 |
| climate-model-simulation-crashes | PFoil:single (voting) | 93.52% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level (best rule) | 93.15% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level (macro distribution voting) | 94.07% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level (micro distribution voting) | 92.96% | 774.0 |
| climate-model-simulation-crashes | PFoil:two-level (voting) | 93.33% | 774.0 |
| climate-model-simulation-crashes | PFossil:one-level (best rule) | 91.48% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level (macro distribution voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level (micro distribution voting) | 91.48% | 555.6 |
| climate-model-simulation-crashes | PFossil:one-level (voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:single (best rule) | 92.78% | 555.6 |
| climate-model-simulation-crashes | PFossil:single (macro distribution voting) | 92.96% | 555.6 |
| climate-model-simulation-crashes | PFossil:single (micro distribution voting) | 92.78% | 555.6 |
| climate-model-simulation-crashes | PFossil:single (voting) | 92.78% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level (best rule) | 92.78% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level (macro distribution voting) | 93.33% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level (micro distribution voting) | 92.59% | 555.6 |
| climate-model-simulation-crashes | PFossil:two-level (voting) | 92.78% | 555.6 |
| climate-model-simulation-crashes | Pypper:one-level (best rule) | 91.67% | 255.7 |
| climate-model-simulation-crashes | Pypper:one-level (macro distribution voting) | 94.81% | 255.7 |
| climate-model-simulation-crashes | Pypper:one-level (micro distribution voting) | 92.59% | 255.7 |
| climate-model-simulation-crashes | Pypper:one-level (voting) | 94.81% | 255.7 |
| climate-model-simulation-crashes | Pypper:single (best rule) | 94.81% | 255.7 |
| climate-model-simulation-crashes | Pypper:single (macro distribution voting) | 94.44% | 255.7 |
| climate-model-simulation-crashes | Pypper:single (micro distribution voting) | 94.81% | 255.7 |
| climate-model-simulation-crashes | Pypper:single (voting) | 94.81% | 255.7 |
| climate-model-simulation-crashes | Pypper:two-level (best rule) | 94.63% | 255.7 |
| climate-model-simulation-crashes | Pypper:two-level (macro distribution voting) | 94.63% | 255.7 |
| climate-model-simulation-crashes | Pypper:two-level (micro distribution voting) | 94.63% | 255.7 |
| climate-model-simulation-crashes | Pypper:two-level (voting) | 94.63% | 255.7 |
| cmc | PFoil:one-level (best rule) | 53.51% | 10326.3 |
| cmc | PFoil:one-level (macro distribution voting) | 53.57% | 10326.3 |
| cmc | PFoil:one-level (micro distribution voting) | 54.05% | 10326.3 |
| cmc | PFoil:one-level (voting) | 53.03% | 10326.3 |
| cmc | PFoil:single (best rule) | 51.40% | 10326.3 |
| cmc | PFoil:single (macro distribution voting) | 51.40% | 10326.3 |
| cmc | PFoil:single (micro distribution voting) | 52.96% | 10326.3 |
| cmc | PFoil:single (voting) | 50.31% | 10326.3 |
| cmc | PFoil:two-level (best rule) | 53.17% | 10326.3 |
| cmc | PFoil:two-level (macro distribution voting) | 52.83% | 10326.3 |
| cmc | PFoil:two-level (micro distribution voting) | 53.78% | 10326.3 |
| cmc | PFoil:two-level (voting) | 51.81% | 10326.3 |
| cmc | PFossil:one-level (best rule) | 53.23% | 192.0 |
| cmc | PFossil:one-level (macro distribution voting) | 52.95% | 192.0 |
| cmc | PFossil:one-level (micro distribution voting) | 51.46% | 192.0 |
| cmc | PFossil:one-level (voting) | 51.53% | 192.0 |
| cmc | PFossil:single (best rule) | 52.61% | 192.0 |
| cmc | PFossil:single (macro distribution voting) | 52.27% | 192.0 |
| cmc | PFossil:single (micro distribution voting) | 51.73% | 192.0 |
| cmc | PFossil:single (voting) | 51.73% | 192.0 |
| cmc | PFossil:two-level (best rule) | 52.21% | 192.0 |
| cmc | PFossil:two-level (macro distribution voting) | 51.94% | 192.0 |
| cmc | PFossil:two-level (micro distribution voting) | 51.94% | 192.0 |
| cmc | PFossil:two-level (voting) | 51.67% | 192.0 |
| cmc | Pypper:one-level (best rule) | 53.57% | 415.0 |
| cmc | Pypper:one-level (macro distribution voting) | 54.86% | 415.0 |
| cmc | Pypper:one-level (micro distribution voting) | 55.13% | 415.0 |
| cmc | Pypper:one-level (voting) | 55.74% | 415.0 |
| cmc | Pypper:single (best rule) | 52.41% | 415.0 |
| cmc | Pypper:single (macro distribution voting) | 52.48% | 415.0 |
| cmc | Pypper:single (micro distribution voting) | 52.48% | 415.0 |
| cmc | Pypper:single (voting) | 52.48% | 415.0 |
| cmc | Pypper:two-level (best rule) | 55.40% | 415.0 |
| cmc | Pypper:two-level (macro distribution voting) | 55.00% | 415.0 |
| cmc | Pypper:two-level (micro distribution voting) | 55.13% | 415.0 |
| cmc | Pypper:two-level (voting) | 54.93% | 415.0 |
| colic | PFoil:one-level (best rule) | 85.07% | 1555.4 |
| colic | PFoil:one-level (macro distribution voting) | 85.07% | 1555.4 |
| colic | PFoil:one-level (micro distribution voting) | 84.80% | 1555.4 |
| colic | PFoil:one-level (voting) | 85.62% | 1555.4 |
| colic | PFoil:single (best rule) | 84.52% | 1555.4 |
| colic | PFoil:single (macro distribution voting) | 83.98% | 1555.4 |
| colic | PFoil:single (micro distribution voting) | 84.52% | 1555.4 |
| colic | PFoil:single (voting) | 83.15% | 1555.4 |
| colic | PFoil:two-level (best rule) | 86.43% | 1555.4 |
| colic | PFoil:two-level (macro distribution voting) | 86.43% | 1555.4 |
| colic | PFoil:two-level (micro distribution voting) | 86.16% | 1555.4 |
| colic | PFoil:two-level (voting) | 86.16% | 1555.4 |
| colic | PFossil:one-level (best rule) | 82.36% | 859.2 |
| colic | PFossil:one-level (macro distribution voting) | 85.89% | 859.2 |
| colic | PFossil:one-level (micro distribution voting) | 85.88% | 859.2 |
| colic | PFossil:one-level (voting) | 86.97% | 859.2 |
| colic | PFossil:single (best rule) | 82.89% | 859.2 |
| colic | PFossil:single (macro distribution voting) | 81.79% | 859.2 |
| colic | PFossil:single (micro distribution voting) | 86.15% | 859.2 |
| colic | PFossil:single (voting) | 82.88% | 859.2 |
| colic | PFossil:two-level (best rule) | 85.35% | 859.2 |
| colic | PFossil:two-level (macro distribution voting) | 85.62% | 859.2 |
| colic | PFossil:two-level (micro distribution voting) | 86.43% | 859.2 |
| colic | PFossil:two-level (voting) | 85.89% | 859.2 |
| colic | Pypper:one-level (best rule) | 83.43% | 251.7 |
| colic | Pypper:one-level (macro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:one-level (micro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:one-level (voting) | 85.88% | 251.7 |
| colic | Pypper:single (best rule) | 85.88% | 251.7 |
| colic | Pypper:single (macro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:single (micro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:single (voting) | 85.88% | 251.7 |
| colic | Pypper:two-level (best rule) | 85.88% | 251.7 |
| colic | Pypper:two-level (macro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:two-level (micro distribution voting) | 85.88% | 251.7 |
| colic | Pypper:two-level (voting) | 85.88% | 251.7 |
| compas-two-years | PFoil:one-level (best rule) | 67.58% | 17858.5 |
| compas-two-years | PFoil:one-level (macro distribution voting) | 67.24% | 17858.5 |
| compas-two-years | PFoil:one-level (micro distribution voting) | 63.62% | 17858.5 |
| compas-two-years | PFoil:one-level (voting) | 66.46% | 17858.5 |
| compas-two-years | PFoil:single (best rule) | 66.65% | 17858.5 |
| compas-two-years | PFoil:single (macro distribution voting) | 66.24% | 17858.5 |
| compas-two-years | PFoil:single (micro distribution voting) | 62.26% | 17858.5 |
| compas-two-years | PFoil:single (voting) | 63.94% | 17858.5 |
| compas-two-years | PFoil:two-level (best rule) | 67.11% | 17858.5 |
| compas-two-years | PFoil:two-level (macro distribution voting) | 66.80% | 17858.5 |
| compas-two-years | PFoil:two-level (micro distribution voting) | 64.61% | 17858.5 |
| compas-two-years | PFoil:two-level (voting) | 65.82% | 17858.5 |
| compas-two-years | PFossil:one-level (best rule) | 66.22% | 63.6 |
| compas-two-years | PFossil:one-level (macro distribution voting) | 64.80% | 63.6 |
| compas-two-years | PFossil:one-level (micro distribution voting) | 64.70% | 63.6 |
| compas-two-years | PFossil:one-level (voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:single (best rule) | 52.96% | 63.6 |
| compas-two-years | PFossil:single (macro distribution voting) | 52.96% | 63.6 |
| compas-two-years | PFossil:single (micro distribution voting) | 52.96% | 63.6 |
| compas-two-years | PFossil:single (voting) | 52.96% | 63.6 |
| compas-two-years | PFossil:two-level (best rule) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level (macro distribution voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level (micro distribution voting) | 64.68% | 63.6 |
| compas-two-years | PFossil:two-level (voting) | 64.68% | 63.6 |
| compas-two-years | Pypper:one-level (best rule) | 67.81% | 436.7 |
| compas-two-years | Pypper:one-level (macro distribution voting) | 67.77% | 436.7 |
| compas-two-years | Pypper:one-level (micro distribution voting) | 67.39% | 436.7 |
| compas-two-years | Pypper:one-level (voting) | 67.77% | 436.7 |
| compas-two-years | Pypper:single (best rule) | 67.05% | 436.7 |
| compas-two-years | Pypper:single (macro distribution voting) | 67.05% | 436.7 |
| compas-two-years | Pypper:single (micro distribution voting) | 67.13% | 436.7 |
| compas-two-years | Pypper:single (voting) | 67.15% | 436.7 |
| compas-two-years | Pypper:two-level (best rule) | 67.60% | 436.7 |
| compas-two-years | Pypper:two-level (macro distribution voting) | 67.58% | 436.7 |
| compas-two-years | Pypper:two-level (micro distribution voting) | 67.64% | 436.7 |
| compas-two-years | Pypper:two-level (voting) | 67.85% | 436.7 |
| credit-approval | PFoil:one-level (best rule) | 86.52% | 2278.9 |
| credit-approval | PFoil:one-level (macro distribution voting) | 85.80% | 2278.9 |
| credit-approval | PFoil:one-level (micro distribution voting) | 86.67% | 2278.9 |
| credit-approval | PFoil:one-level (voting) | 86.09% | 2278.9 |
| credit-approval | PFoil:single (best rule) | 84.49% | 2278.9 |
| credit-approval | PFoil:single (macro distribution voting) | 85.07% | 2278.9 |
| credit-approval | PFoil:single (micro distribution voting) | 83.77% | 2278.9 |
| credit-approval | PFoil:single (voting) | 83.33% | 2278.9 |
| credit-approval | PFoil:two-level (best rule) | 85.51% | 2278.9 |
| credit-approval | PFoil:two-level (macro distribution voting) | 85.80% | 2278.9 |
| credit-approval | PFoil:two-level (micro distribution voting) | 86.52% | 2278.9 |
| credit-approval | PFoil:two-level (voting) | 85.80% | 2278.9 |
| credit-approval | PFossil:one-level (best rule) | 81.01% | 709.6 |
| credit-approval | PFossil:one-level (macro distribution voting) | 85.22% | 709.6 |
| credit-approval | PFossil:one-level (micro distribution voting) | 85.51% | 709.6 |
| credit-approval | PFossil:one-level (voting) | 85.65% | 709.6 |
| credit-approval | PFossil:single (best rule) | 82.17% | 709.6 |
| credit-approval | PFossil:single (macro distribution voting) | 82.03% | 709.6 |
| credit-approval | PFossil:single (micro distribution voting) | 85.07% | 709.6 |
| credit-approval | PFossil:single (voting) | 82.46% | 709.6 |
| credit-approval | PFossil:two-level (best rule) | 85.07% | 709.6 |
| credit-approval | PFossil:two-level (macro distribution voting) | 85.51% | 709.6 |
| credit-approval | PFossil:two-level (micro distribution voting) | 85.65% | 709.6 |
| credit-approval | PFossil:two-level (voting) | 85.65% | 709.6 |
| credit-approval | Pypper:one-level (best rule) | 85.22% | 275.4 |
| credit-approval | Pypper:one-level (macro distribution voting) | 85.22% | 275.4 |
| credit-approval | Pypper:one-level (micro distribution voting) | 85.65% | 275.4 |
| credit-approval | Pypper:one-level (voting) | 85.07% | 275.4 |
| credit-approval | Pypper:single (best rule) | 84.20% | 275.4 |
| credit-approval | Pypper:single (macro distribution voting) | 84.20% | 275.4 |
| credit-approval | Pypper:single (micro distribution voting) | 83.77% | 275.4 |
| credit-approval | Pypper:single (voting) | 83.04% | 275.4 |
| credit-approval | Pypper:two-level (best rule) | 85.07% | 275.4 |
| credit-approval | Pypper:two-level (macro distribution voting) | 85.07% | 275.4 |
| credit-approval | Pypper:two-level (micro distribution voting) | 85.36% | 275.4 |
| credit-approval | Pypper:two-level (voting) | 85.36% | 275.4 |
| credit-g | PFoil:one-level (best rule) | 71.70% | 3743.0 |
| credit-g | PFoil:one-level (macro distribution voting) | 76.90% | 3743.0 |
| credit-g | PFoil:one-level (micro distribution voting) | 72.40% | 3743.0 |
| credit-g | PFoil:one-level (voting) | 76.30% | 3743.0 |
| credit-g | PFoil:single (best rule) | 73.10% | 3743.0 |
| credit-g | PFoil:single (macro distribution voting) | 73.50% | 3743.0 |
| credit-g | PFoil:single (micro distribution voting) | 72.30% | 3743.0 |
| credit-g | PFoil:single (voting) | 72.30% | 3743.0 |
| credit-g | PFoil:two-level (best rule) | 76.90% | 3743.0 |
| credit-g | PFoil:two-level (macro distribution voting) | 77.10% | 3743.0 |
| credit-g | PFoil:two-level (micro distribution voting) | 75.40% | 3743.0 |
| credit-g | PFoil:two-level (voting) | 76.40% | 3743.0 |
| credit-g | PFossil:one-level (best rule) | 73.30% | 401.7 |
| credit-g | PFossil:one-level (macro distribution voting) | 75.40% | 401.7 |
| credit-g | PFossil:one-level (micro distribution voting) | 75.00% | 401.7 |
| credit-g | PFossil:one-level (voting) | 74.00% | 401.7 |
| credit-g | PFossil:single (best rule) | 73.20% | 401.7 |
| credit-g | PFossil:single (macro distribution voting) | 73.20% | 401.7 |
| credit-g | PFossil:single (micro distribution voting) | 73.30% | 401.7 |
| credit-g | PFossil:single (voting) | 73.20% | 401.7 |
| credit-g | PFossil:two-level (best rule) | 74.50% | 401.7 |
| credit-g | PFossil:two-level (macro distribution voting) | 74.90% | 401.7 |
| credit-g | PFossil:two-level (micro distribution voting) | 74.40% | 401.7 |
| credit-g | PFossil:two-level (voting) | 74.50% | 401.7 |
| credit-g | Pypper:one-level (best rule) | 74.30% | 384.1 |
| credit-g | Pypper:one-level (macro distribution voting) | 74.30% | 384.1 |
| credit-g | Pypper:one-level (micro distribution voting) | 72.90% | 384.1 |
| credit-g | Pypper:one-level (voting) | 72.50% | 384.1 |
| credit-g | Pypper:single (best rule) | 71.40% | 384.1 |
| credit-g | Pypper:single (macro distribution voting) | 71.50% | 384.1 |
| credit-g | Pypper:single (micro distribution voting) | 71.50% | 384.1 |
| credit-g | Pypper:single (voting) | 71.50% | 384.1 |
| credit-g | Pypper:two-level (best rule) | 73.70% | 384.1 |
| credit-g | Pypper:two-level (macro distribution voting) | 73.50% | 384.1 |
| credit-g | Pypper:two-level (micro distribution voting) | 73.70% | 384.1 |
| credit-g | Pypper:two-level (voting) | 73.20% | 384.1 |
| cylinder-bands | PFoil:one-level (best rule) | 70.19% | 2045.2 |
| cylinder-bands | PFoil:one-level (macro distribution voting) | 84.44% | 2045.2 |
| cylinder-bands | PFoil:one-level (micro distribution voting) | 78.89% | 2045.2 |
| cylinder-bands | PFoil:one-level (voting) | 83.70% | 2045.2 |
| cylinder-bands | PFoil:single (best rule) | 75.00% | 2045.2 |
| cylinder-bands | PFoil:single (macro distribution voting) | 76.30% | 2045.2 |
| cylinder-bands | PFoil:single (micro distribution voting) | 73.70% | 2045.2 |
| cylinder-bands | PFoil:single (voting) | 74.07% | 2045.2 |
| cylinder-bands | PFoil:two-level (best rule) | 81.85% | 2045.2 |
| cylinder-bands | PFoil:two-level (macro distribution voting) | 82.22% | 2045.2 |
| cylinder-bands | PFoil:two-level (micro distribution voting) | 81.11% | 2045.2 |
| cylinder-bands | PFoil:two-level (voting) | 82.22% | 2045.2 |
| cylinder-bands | PFossil:one-level (best rule) | 73.33% | 526.1 |
| cylinder-bands | PFossil:one-level (macro distribution voting) | 75.74% | 526.1 |
| cylinder-bands | PFossil:one-level (micro distribution voting) | 71.67% | 526.1 |
| cylinder-bands | PFossil:one-level (voting) | 75.37% | 526.1 |
| cylinder-bands | PFossil:single (best rule) | 70.19% | 526.1 |
| cylinder-bands | PFossil:single (macro distribution voting) | 70.56% | 526.1 |
| cylinder-bands | PFossil:single (micro distribution voting) | 68.15% | 526.1 |
| cylinder-bands | PFossil:single (voting) | 67.96% | 526.1 |
| cylinder-bands | PFossil:two-level (best rule) | 74.63% | 526.1 |
| cylinder-bands | PFossil:two-level (macro distribution voting) | 74.63% | 526.1 |
| cylinder-bands | PFossil:two-level (micro distribution voting) | 72.59% | 526.1 |
| cylinder-bands | PFossil:two-level (voting) | 71.67% | 526.1 |
| cylinder-bands | Pypper:one-level (best rule) | 74.26% | 414.1 |
| cylinder-bands | Pypper:one-level (macro distribution voting) | 77.04% | 414.1 |
| cylinder-bands | Pypper:one-level (micro distribution voting) | 70.93% | 414.1 |
| cylinder-bands | Pypper:one-level (voting) | 77.78% | 414.1 |
| cylinder-bands | Pypper:single (best rule) | 68.33% | 414.1 |
| cylinder-bands | Pypper:single (macro distribution voting) | 67.96% | 414.1 |
| cylinder-bands | Pypper:single (micro distribution voting) | 67.41% | 414.1 |
| cylinder-bands | Pypper:single (voting) | 68.33% | 414.1 |
| cylinder-bands | Pypper:two-level (best rule) | 78.70% | 414.1 |
| cylinder-bands | Pypper:two-level (macro distribution voting) | 78.33% | 414.1 |
| cylinder-bands | Pypper:two-level (micro distribution voting) | 73.89% | 414.1 |
| cylinder-bands | Pypper:two-level (voting) | 74.44% | 414.1 |
| diabetes | PFoil:one-level (best rule) | 74.21% | 3522.2 |
| diabetes | PFoil:one-level (macro distribution voting) | 76.30% | 3522.2 |
| diabetes | PFoil:one-level (micro distribution voting) | 75.51% | 3522.2 |
| diabetes | PFoil:one-level (voting) | 76.82% | 3522.2 |
| diabetes | PFoil:single (best rule) | 75.77% | 3522.2 |
| diabetes | PFoil:single (macro distribution voting) | 75.12% | 3522.2 |
| diabetes | PFoil:single (micro distribution voting) | 73.44% | 3522.2 |
| diabetes | PFoil:single (voting) | 75.40% | 3522.2 |
| diabetes | PFoil:two-level (best rule) | 77.34% | 3522.2 |
| diabetes | PFoil:two-level (macro distribution voting) | 77.08% | 3522.2 |
| diabetes | PFoil:two-level (micro distribution voting) | 75.13% | 3522.2 |
| diabetes | PFoil:two-level (voting) | 76.30% | 3522.2 |
| diabetes | PFossil:one-level (best rule) | 73.69% | 661.5 |
| diabetes | PFossil:one-level (macro distribution voting) | 75.52% | 661.5 |
| diabetes | PFossil:one-level (micro distribution voting) | 75.91% | 661.5 |
| diabetes | PFossil:one-level (voting) | 75.12% | 661.5 |
| diabetes | PFossil:single (best rule) | 74.61% | 661.5 |
| diabetes | PFossil:single (macro distribution voting) | 73.83% | 661.5 |
| diabetes | PFossil:single (micro distribution voting) | 75.13% | 661.5 |
| diabetes | PFossil:single (voting) | 74.09% | 661.5 |
| diabetes | PFossil:two-level (best rule) | 76.17% | 661.5 |
| diabetes | PFossil:two-level (macro distribution voting) | 75.65% | 661.5 |
| diabetes | PFossil:two-level (micro distribution voting) | 76.05% | 661.5 |
| diabetes | PFossil:two-level (voting) | 75.52% | 661.5 |
| diabetes | Pypper:one-level (best rule) | 74.08% | 361.6 |
| diabetes | Pypper:one-level (macro distribution voting) | 74.21% | 361.6 |
| diabetes | Pypper:one-level (micro distribution voting) | 73.82% | 361.6 |
| diabetes | Pypper:one-level (voting) | 74.21% | 361.6 |
| diabetes | Pypper:single (best rule) | 76.04% | 361.6 |
| diabetes | Pypper:single (macro distribution voting) | 76.04% | 361.6 |
| diabetes | Pypper:single (micro distribution voting) | 75.78% | 361.6 |
| diabetes | Pypper:single (voting) | 75.91% | 361.6 |
| diabetes | Pypper:two-level (best rule) | 74.09% | 361.6 |
| diabetes | Pypper:two-level (macro distribution voting) | 73.83% | 361.6 |
| diabetes | Pypper:two-level (micro distribution voting) | 74.47% | 361.6 |
| diabetes | Pypper:two-level (voting) | 73.95% | 361.6 |
| dna | PFoil:one-level (best rule) | 91.97% | 3508.6 |
| dna | PFoil:one-level (macro distribution voting) | 95.32% | 3508.6 |
| dna | PFoil:one-level (micro distribution voting) | 95.10% | 3508.6 |
| dna | PFoil:one-level (voting) | 95.29% | 3508.6 |
| dna | PFoil:single (best rule) | 92.50% | 3508.6 |
| dna | PFoil:single (macro distribution voting) | 91.97% | 3508.6 |
| dna | PFoil:single (micro distribution voting) | 92.00% | 3508.6 |
| dna | PFoil:single (voting) | 91.05% | 3508.6 |
| dna | PFoil:two-level (best rule) | 95.29% | 3508.6 |
| dna | PFoil:two-level (macro distribution voting) | 95.64% | 3508.6 |
| dna | PFoil:two-level (micro distribution voting) | 95.48% | 3508.6 |
| dna | PFoil:two-level (voting) | 95.61% | 3508.6 |
| dna | PFossil:one-level (best rule) | 87.95% | 1657.7 |
| dna | PFossil:one-level (macro distribution voting) | 95.57% | 1657.7 |
| dna | PFossil:one-level (micro distribution voting) | 90.05% | 1657.7 |
| dna | PFossil:one-level (voting) | 94.51% | 1657.7 |
| dna | PFossil:single (best rule) | 93.69% | 1657.7 |
| dna | PFossil:single (macro distribution voting) | 93.53% | 1657.7 |
| dna | PFossil:single (micro distribution voting) | 89.89% | 1657.7 |
| dna | PFossil:single (voting) | 90.65% | 1657.7 |
| dna | PFossil:two-level (best rule) | 95.42% | 1657.7 |
| dna | PFossil:two-level (macro distribution voting) | 95.57% | 1657.7 |
| dna | PFossil:two-level (micro distribution voting) | 90.58% | 1657.7 |
| dna | PFossil:two-level (voting) | 91.18% | 1657.7 |
| dna | Pypper:one-level (best rule) | 92.91% | 928.9 |
| dna | Pypper:one-level (macro distribution voting) | 93.31% | 928.9 |
| dna | Pypper:one-level (micro distribution voting) | 92.03% | 928.9 |
| dna | Pypper:one-level (voting) | 93.25% | 928.9 |
| dna | Pypper:single (best rule) | 93.75% | 928.9 |
| dna | Pypper:single (macro distribution voting) | 93.60% | 928.9 |
| dna | Pypper:single (micro distribution voting) | 93.31% | 928.9 |
| dna | Pypper:single (voting) | 93.60% | 928.9 |
| dna | Pypper:two-level (best rule) | 93.69% | 928.9 |
| dna | Pypper:two-level (macro distribution voting) | 93.63% | 928.9 |
| dna | Pypper:two-level (micro distribution voting) | 92.84% | 928.9 |
| dna | Pypper:two-level (voting) | 93.60% | 928.9 |
| dresses-sales | PFoil:one-level (best rule) | 58.80% | 2981.6 |
| dresses-sales | PFoil:one-level (macro distribution voting) | 60.20% | 2981.6 |
| dresses-sales | PFoil:one-level (micro distribution voting) | 60.00% | 2981.6 |
| dresses-sales | PFoil:one-level (voting) | 60.00% | 2981.6 |
| dresses-sales | PFoil:single (best rule) | 58.20% | 2981.6 |
| dresses-sales | PFoil:single (macro distribution voting) | 59.00% | 2981.6 |
| dresses-sales | PFoil:single (micro distribution voting) | 60.00% | 2981.6 |
| dresses-sales | PFoil:single (voting) | 56.80% | 2981.6 |
| dresses-sales | PFoil:two-level (best rule) | 61.80% | 2981.6 |
| dresses-sales | PFoil:two-level (macro distribution voting) | 60.40% | 2981.6 |
| dresses-sales | PFoil:two-level (micro distribution voting) | 59.80% | 2981.6 |
| dresses-sales | PFoil:two-level (voting) | 59.20% | 2981.6 |
| dresses-sales | PFossil:one-level (best rule) | 58.60% | 250.9 |
| dresses-sales | PFossil:one-level (macro distribution voting) | 63.20% | 250.9 |
| dresses-sales | PFossil:one-level (micro distribution voting) | 60.00% | 250.9 |
| dresses-sales | PFossil:one-level (voting) | 62.20% | 250.9 |
| dresses-sales | PFossil:single (best rule) | 59.40% | 250.9 |
| dresses-sales | PFossil:single (macro distribution voting) | 59.40% | 250.9 |
| dresses-sales | PFossil:single (micro distribution voting) | 59.20% | 250.9 |
| dresses-sales | PFossil:single (voting) | 59.20% | 250.9 |
| dresses-sales | PFossil:two-level (best rule) | 62.00% | 250.9 |
| dresses-sales | PFossil:two-level (macro distribution voting) | 61.40% | 250.9 |
| dresses-sales | PFossil:two-level (micro distribution voting) | 61.40% | 250.9 |
| dresses-sales | PFossil:two-level (voting) | 61.40% | 250.9 |
| dresses-sales | Pypper:one-level (best rule) | 59.20% | 213.2 |
| dresses-sales | Pypper:one-level (macro distribution voting) | 62.20% | 213.2 |
| dresses-sales | Pypper:one-level (micro distribution voting) | 60.20% | 213.2 |
| dresses-sales | Pypper:one-level (voting) | 61.40% | 213.2 |
| dresses-sales | Pypper:single (best rule) | 60.40% | 213.2 |
| dresses-sales | Pypper:single (macro distribution voting) | 60.40% | 213.2 |
| dresses-sales | Pypper:single (micro distribution voting) | 60.60% | 213.2 |
| dresses-sales | Pypper:single (voting) | 60.60% | 213.2 |
| dresses-sales | Pypper:two-level (best rule) | 62.40% | 213.2 |
| dresses-sales | Pypper:two-level (macro distribution voting) | 62.40% | 213.2 |
| dresses-sales | Pypper:two-level (micro distribution voting) | 60.80% | 213.2 |
| dresses-sales | Pypper:two-level (voting) | 61.00% | 213.2 |
| ecoli | PFoil:one-level (best rule) | 80.96% | 1705.1 |
| ecoli | PFoil:one-level (macro distribution voting) | 86.35% | 1705.1 |
| ecoli | PFoil:one-level (micro distribution voting) | 84.86% | 1705.1 |
| ecoli | PFoil:one-level (voting) | 84.88% | 1705.1 |
| ecoli | PFoil:single (best rule) | 82.51% | 1705.1 |
| ecoli | PFoil:single (macro distribution voting) | 83.10% | 1705.1 |
| ecoli | PFoil:single (micro distribution voting) | 79.22% | 1705.1 |
| ecoli | PFoil:single (voting) | 79.80% | 1705.1 |
| ecoli | PFoil:two-level (best rule) | 85.46% | 1705.1 |
| ecoli | PFoil:two-level (macro distribution voting) | 86.05% | 1705.1 |
| ecoli | PFoil:two-level (micro distribution voting) | 84.87% | 1705.1 |
| ecoli | PFoil:two-level (voting) | 85.46% | 1705.1 |
| ecoli | PFossil:one-level (best rule) | 80.39% | 1183.1 |
| ecoli | PFossil:one-level (macro distribution voting) | 86.36% | 1183.1 |
| ecoli | PFossil:one-level (micro distribution voting) | 85.17% | 1183.1 |
| ecoli | PFossil:one-level (voting) | 86.67% | 1183.1 |
| ecoli | PFossil:single (best rule) | 81.92% | 1183.1 |
| ecoli | PFossil:single (macro distribution voting) | 82.22% | 1183.1 |
| ecoli | PFossil:single (micro distribution voting) | 84.29% | 1183.1 |
| ecoli | PFossil:single (voting) | 83.98% | 1183.1 |
| ecoli | PFossil:two-level (best rule) | 85.18% | 1183.1 |
| ecoli | PFossil:two-level (macro distribution voting) | 86.07% | 1183.1 |
| ecoli | PFossil:two-level (micro distribution voting) | 85.18% | 1183.1 |
| ecoli | PFossil:two-level (voting) | 86.35% | 1183.1 |
| ecoli | Pypper:one-level (best rule) | 83.38% | 434.3 |
| ecoli | Pypper:one-level (macro distribution voting) | 85.77% | 434.3 |
| ecoli | Pypper:one-level (micro distribution voting) | 82.77% | 434.3 |
| ecoli | Pypper:one-level (voting) | 85.47% | 434.3 |
| ecoli | Pypper:single (best rule) | 81.91% | 434.3 |
| ecoli | Pypper:single (macro distribution voting) | 81.30% | 434.3 |
| ecoli | Pypper:single (micro distribution voting) | 81.29% | 434.3 |
| ecoli | Pypper:single (voting) | 81.29% | 434.3 |
| ecoli | Pypper:two-level (best rule) | 85.45% | 434.3 |
| ecoli | Pypper:two-level (macro distribution voting) | 86.07% | 434.3 |
| ecoli | Pypper:two-level (micro distribution voting) | 85.46% | 434.3 |
| ecoli | Pypper:two-level (voting) | 85.76% | 434.3 |
| glass | PFoil:one-level (best rule) | 65.09% | 1404.5 |
| glass | PFoil:one-level (macro distribution voting) | 75.28% | 1404.5 |
| glass | PFoil:one-level (micro distribution voting) | 69.26% | 1404.5 |
| glass | PFoil:one-level (voting) | 74.76% | 1404.5 |
| glass | PFoil:single (best rule) | 72.94% | 1404.5 |
| glass | PFoil:single (macro distribution voting) | 72.47% | 1404.5 |
| glass | PFoil:single (micro distribution voting) | 66.00% | 1404.5 |
| glass | PFoil:single (voting) | 67.38% | 1404.5 |
| glass | PFoil:two-level (best rule) | 75.74% | 1404.5 |
| glass | PFoil:two-level (macro distribution voting) | 75.28% | 1404.5 |
| glass | PFoil:two-level (micro distribution voting) | 73.48% | 1404.5 |
| glass | PFoil:two-level (voting) | 72.53% | 1404.5 |
| glass | PFossil:one-level (best rule) | 65.87% | 1211.7 |
| glass | PFossil:one-level (macro distribution voting) | 71.54% | 1211.7 |
| glass | PFossil:one-level (micro distribution voting) | 70.56% | 1211.7 |
| glass | PFossil:one-level (voting) | 73.38% | 1211.7 |
| glass | PFossil:single (best rule) | 71.97% | 1211.7 |
| glass | PFossil:single (macro distribution voting) | 70.58% | 1211.7 |
| glass | PFossil:single (micro distribution voting) | 70.11% | 1211.7 |
| glass | PFossil:single (voting) | 69.68% | 1211.7 |
| glass | PFossil:two-level (best rule) | 72.47% | 1211.7 |
| glass | PFossil:two-level (macro distribution voting) | 72.01% | 1211.7 |
| glass | PFossil:two-level (micro distribution voting) | 73.85% | 1211.7 |
| glass | PFossil:two-level (voting) | 72.45% | 1211.7 |
| glass | Pypper:one-level (best rule) | 68.33% | 410.2 |
| glass | Pypper:one-level (macro distribution voting) | 73.42% | 410.2 |
| glass | Pypper:one-level (micro distribution voting) | 69.24% | 410.2 |
| glass | Pypper:one-level (voting) | 74.83% | 410.2 |
| glass | Pypper:single (best rule) | 64.55% | 410.2 |
| glass | Pypper:single (macro distribution voting) | 62.68% | 410.2 |
| glass | Pypper:single (micro distribution voting) | 62.75% | 410.2 |
| glass | Pypper:single (voting) | 64.52% | 410.2 |
| glass | Pypper:two-level (best rule) | 72.51% | 410.2 |
| glass | Pypper:two-level (macro distribution voting) | 73.90% | 410.2 |
| glass | Pypper:two-level (micro distribution voting) | 71.60% | 410.2 |
| glass | Pypper:two-level (voting) | 71.58% | 410.2 |
| hayes-roth | PFoil:one-level (best rule) | 76.25% | 1131.7 |
| hayes-roth | PFoil:one-level (macro distribution voting) | 76.88% | 1131.7 |
| hayes-roth | PFoil:one-level (micro distribution voting) | 78.12% | 1131.7 |
| hayes-roth | PFoil:one-level (voting) | 75.62% | 1131.7 |
| hayes-roth | PFoil:single (best rule) | 77.50% | 1131.7 |
| hayes-roth | PFoil:single (macro distribution voting) | 72.50% | 1131.7 |
| hayes-roth | PFoil:single (micro distribution voting) | 76.88% | 1131.7 |
| hayes-roth | PFoil:single (voting) | 70.62% | 1131.7 |
| hayes-roth | PFoil:two-level (best rule) | 79.38% | 1131.7 |
| hayes-roth | PFoil:two-level (macro distribution voting) | 75.62% | 1131.7 |
| hayes-roth | PFoil:two-level (micro distribution voting) | 78.12% | 1131.7 |
| hayes-roth | PFoil:two-level (voting) | 74.38% | 1131.7 |
| hayes-roth | PFossil:one-level (best rule) | 79.38% | 675.3 |
| hayes-roth | PFossil:one-level (macro distribution voting) | 84.38% | 675.3 |
| hayes-roth | PFossil:one-level (micro distribution voting) | 81.25% | 675.3 |
| hayes-roth | PFossil:one-level (voting) | 85.00% | 675.3 |
| hayes-roth | PFossil:single (best rule) | 82.50% | 675.3 |
| hayes-roth | PFossil:single (macro distribution voting) | 81.88% | 675.3 |
| hayes-roth | PFossil:single (micro distribution voting) | 81.88% | 675.3 |
| hayes-roth | PFossil:single (voting) | 81.25% | 675.3 |
| hayes-roth | PFossil:two-level (best rule) | 84.38% | 675.3 |
| hayes-roth | PFossil:two-level (macro distribution voting) | 81.25% | 675.3 |
| hayes-roth | PFossil:two-level (micro distribution voting) | 83.12% | 675.3 |
| hayes-roth | PFossil:two-level (voting) | 76.88% | 675.3 |
| hayes-roth | Pypper:one-level (best rule) | 78.75% | 418.6 |
| hayes-roth | Pypper:one-level (macro distribution voting) | 83.12% | 418.6 |
| hayes-roth | Pypper:one-level (micro distribution voting) | 78.75% | 418.6 |
| hayes-roth | Pypper:one-level (voting) | 80.00% | 418.6 |
| hayes-roth | Pypper:single (best rule) | 79.38% | 418.6 |
| hayes-roth | Pypper:single (macro distribution voting) | 79.38% | 418.6 |
| hayes-roth | Pypper:single (micro distribution voting) | 75.00% | 418.6 |
| hayes-roth | Pypper:single (voting) | 75.00% | 418.6 |
| hayes-roth | Pypper:two-level (best rule) | 82.50% | 418.6 |
| hayes-roth | Pypper:two-level (macro distribution voting) | 80.00% | 418.6 |
| hayes-roth | Pypper:two-level (micro distribution voting) | 80.00% | 418.6 |
| hayes-roth | Pypper:two-level (voting) | 76.88% | 418.6 |
| heart-c | PFoil:one-level (best rule) | 81.56% | 1167.5 |
| heart-c | PFoil:one-level (macro distribution voting) | 83.85% | 1167.5 |
| heart-c | PFoil:one-level (micro distribution voting) | 83.54% | 1167.5 |
| heart-c | PFoil:one-level (voting) | 82.86% | 1167.5 |
| heart-c | PFoil:single (best rule) | 78.92% | 1167.5 |
| heart-c | PFoil:single (macro distribution voting) | 78.59% | 1167.5 |
| heart-c | PFoil:single (micro distribution voting) | 78.62% | 1167.5 |
| heart-c | PFoil:single (voting) | 77.61% | 1167.5 |
| heart-c | PFoil:two-level (best rule) | 83.19% | 1167.5 |
| heart-c | PFoil:two-level (macro distribution voting) | 83.53% | 1167.5 |
| heart-c | PFoil:two-level (micro distribution voting) | 83.18% | 1167.5 |
| heart-c | PFoil:two-level (voting) | 81.89% | 1167.5 |
| heart-c | PFossil:one-level (best rule) | 77.91% | 786.6 |
| heart-c | PFossil:one-level (macro distribution voting) | 82.23% | 786.6 |
| heart-c | PFossil:one-level (micro distribution voting) | 81.22% | 786.6 |
| heart-c | PFossil:one-level (voting) | 81.89% | 786.6 |
| heart-c | PFossil:single (best rule) | 79.54% | 786.6 |
| heart-c | PFossil:single (macro distribution voting) | 79.88% | 786.6 |
| heart-c | PFossil:single (micro distribution voting) | 77.24% | 786.6 |
| heart-c | PFossil:single (voting) | 78.24% | 786.6 |
| heart-c | PFossil:two-level (best rule) | 84.19% | 786.6 |
| heart-c | PFossil:two-level (macro distribution voting) | 83.87% | 786.6 |
| heart-c | PFossil:two-level (micro distribution voting) | 81.22% | 786.6 |
| heart-c | PFossil:two-level (voting) | 81.22% | 786.6 |
| heart-c | Pypper:one-level (best rule) | 80.22% | 294.8 |
| heart-c | Pypper:one-level (macro distribution voting) | 83.53% | 294.8 |
| heart-c | Pypper:one-level (micro distribution voting) | 81.88% | 294.8 |
| heart-c | Pypper:one-level (voting) | 82.86% | 294.8 |
| heart-c | Pypper:single (best rule) | 82.55% | 294.8 |
| heart-c | Pypper:single (macro distribution voting) | 82.55% | 294.8 |
| heart-c | Pypper:single (micro distribution voting) | 82.56% | 294.8 |
| heart-c | Pypper:single (voting) | 82.56% | 294.8 |
| heart-c | Pypper:two-level (best rule) | 84.19% | 294.8 |
| heart-c | Pypper:two-level (macro distribution voting) | 83.54% | 294.8 |
| heart-c | Pypper:two-level (micro distribution voting) | 83.20% | 294.8 |
| heart-c | Pypper:two-level (voting) | 84.52% | 294.8 |
| heart-h | PFoil:one-level (best rule) | 78.59% | 1291.2 |
| heart-h | PFoil:one-level (macro distribution voting) | 80.95% | 1291.2 |
| heart-h | PFoil:one-level (micro distribution voting) | 79.59% | 1291.2 |
| heart-h | PFoil:one-level (voting) | 81.63% | 1291.2 |
| heart-h | PFoil:single (best rule) | 78.93% | 1291.2 |
| heart-h | PFoil:single (macro distribution voting) | 79.25% | 1291.2 |
| heart-h | PFoil:single (micro distribution voting) | 79.29% | 1291.2 |
| heart-h | PFoil:single (voting) | 78.56% | 1291.2 |
| heart-h | PFoil:two-level (best rule) | 80.62% | 1291.2 |
| heart-h | PFoil:two-level (macro distribution voting) | 80.28% | 1291.2 |
| heart-h | PFoil:two-level (micro distribution voting) | 80.62% | 1291.2 |
| heart-h | PFoil:two-level (voting) | 80.63% | 1291.2 |
| heart-h | PFossil:one-level (best rule) | 82.28% | 736.2 |
| heart-h | PFossil:one-level (macro distribution voting) | 80.62% | 736.2 |
| heart-h | PFossil:one-level (micro distribution voting) | 80.61% | 736.2 |
| heart-h | PFossil:one-level (voting) | 80.97% | 736.2 |
| heart-h | PFossil:single (best rule) | 81.28% | 736.2 |
| heart-h | PFossil:single (macro distribution voting) | 79.92% | 736.2 |
| heart-h | PFossil:single (micro distribution voting) | 78.93% | 736.2 |
| heart-h | PFossil:single (voting) | 80.94% | 736.2 |
| heart-h | PFossil:two-level (best rule) | 80.26% | 736.2 |
| heart-h | PFossil:two-level (macro distribution voting) | 80.29% | 736.2 |
| heart-h | PFossil:two-level (micro distribution voting) | 81.30% | 736.2 |
| heart-h | PFossil:two-level (voting) | 78.91% | 736.2 |
| heart-h | Pypper:one-level (best rule) | 80.93% | 245.0 |
| heart-h | Pypper:one-level (macro distribution voting) | 81.99% | 245.0 |
| heart-h | Pypper:one-level (micro distribution voting) | 80.30% | 245.0 |
| heart-h | Pypper:one-level (voting) | 81.31% | 245.0 |
| heart-h | Pypper:single (best rule) | 78.20% | 245.0 |
| heart-h | Pypper:single (macro distribution voting) | 78.20% | 245.0 |
| heart-h | Pypper:single (micro distribution voting) | 77.85% | 245.0 |
| heart-h | Pypper:single (voting) | 77.85% | 245.0 |
| heart-h | Pypper:two-level (best rule) | 81.31% | 245.0 |
| heart-h | Pypper:two-level (macro distribution voting) | 80.97% | 245.0 |
| heart-h | Pypper:two-level (micro distribution voting) | 80.99% | 245.0 |
| heart-h | Pypper:two-level (voting) | 81.31% | 245.0 |
| heart-statlog | PFoil:one-level (best rule) | 80.00% | 1074.3 |
| heart-statlog | PFoil:one-level (macro distribution voting) | 82.59% | 1074.3 |
| heart-statlog | PFoil:one-level (micro distribution voting) | 83.70% | 1074.3 |
| heart-statlog | PFoil:one-level (voting) | 81.48% | 1074.3 |
| heart-statlog | PFoil:single (best rule) | 77.04% | 1074.3 |
| heart-statlog | PFoil:single (macro distribution voting) | 76.30% | 1074.3 |
| heart-statlog | PFoil:single (micro distribution voting) | 77.04% | 1074.3 |
| heart-statlog | PFoil:single (voting) | 75.19% | 1074.3 |
| heart-statlog | PFoil:two-level (best rule) | 81.11% | 1074.3 |
| heart-statlog | PFoil:two-level (macro distribution voting) | 80.37% | 1074.3 |
| heart-statlog | PFoil:two-level (micro distribution voting) | 82.59% | 1074.3 |
| heart-statlog | PFoil:two-level (voting) | 81.11% | 1074.3 |
| heart-statlog | PFossil:one-level (best rule) | 79.63% | 750.4 |
| heart-statlog | PFossil:one-level (macro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:one-level (micro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:one-level (voting) | 81.48% | 750.4 |
| heart-statlog | PFossil:single (best rule) | 80.74% | 750.4 |
| heart-statlog | PFossil:single (macro distribution voting) | 79.26% | 750.4 |
| heart-statlog | PFossil:single (micro distribution voting) | 80.00% | 750.4 |
| heart-statlog | PFossil:single (voting) | 79.26% | 750.4 |
| heart-statlog | PFossil:two-level (best rule) | 83.33% | 750.4 |
| heart-statlog | PFossil:two-level (macro distribution voting) | 81.85% | 750.4 |
| heart-statlog | PFossil:two-level (micro distribution voting) | 82.59% | 750.4 |
| heart-statlog | PFossil:two-level (voting) | 81.11% | 750.4 |
| heart-statlog | Pypper:one-level (best rule) | 80.00% | 275.0 |
| heart-statlog | Pypper:one-level (macro distribution voting) | 83.70% | 275.0 |
| heart-statlog | Pypper:one-level (micro distribution voting) | 81.85% | 275.0 |
| heart-statlog | Pypper:one-level (voting) | 83.70% | 275.0 |
| heart-statlog | Pypper:single (best rule) | 79.26% | 275.0 |
| heart-statlog | Pypper:single (macro distribution voting) | 79.63% | 275.0 |
| heart-statlog | Pypper:single (micro distribution voting) | 80.74% | 275.0 |
| heart-statlog | Pypper:single (voting) | 80.74% | 275.0 |
| heart-statlog | Pypper:two-level (best rule) | 84.44% | 275.0 |
| heart-statlog | Pypper:two-level (macro distribution voting) | 84.07% | 275.0 |
| heart-statlog | Pypper:two-level (micro distribution voting) | 82.96% | 275.0 |
| heart-statlog | Pypper:two-level (voting) | 82.96% | 275.0 |
| heloc | PFoil:one-level (best rule) | 71.81% | 28910.0 |
| heloc | PFoil:one-level (macro distribution voting) | 71.64% | 28910.0 |
| heloc | PFoil:one-level (micro distribution voting) | 70.87% | 28910.0 |
| heloc | PFoil:one-level (voting) | 70.99% | 28910.0 |
| heloc | PFoil:single (best rule) | 67.84% | 28910.0 |
| heloc | PFoil:single (macro distribution voting) | 68.04% | 28910.0 |
| heloc | PFoil:single (micro distribution voting) | 66.05% | 28910.0 |
| heloc | PFoil:single (voting) | 62.21% | 28910.0 |
| heloc | PFoil:two-level (best rule) | 71.71% | 28910.0 |
| heloc | PFoil:two-level (macro distribution voting) | 71.78% | 28910.0 |
| heloc | PFoil:two-level (micro distribution voting) | 71.65% | 28910.0 |
| heloc | PFoil:two-level (voting) | 70.65% | 28910.0 |
| heloc | PFossil:one-level (best rule) | 69.99% | 101.4 |
| heloc | PFossil:one-level (macro distribution voting) | 69.38% | 101.4 |
| heloc | PFossil:one-level (micro distribution voting) | 69.34% | 101.4 |
| heloc | PFossil:one-level (voting) | 69.33% | 101.4 |
| heloc | PFossil:single (best rule) | 68.83% | 101.4 |
| heloc | PFossil:single (macro distribution voting) | 68.83% | 101.4 |
| heloc | PFossil:single (micro distribution voting) | 68.83% | 101.4 |
| heloc | PFossil:single (voting) | 68.83% | 101.4 |
| heloc | PFossil:two-level (best rule) | 69.33% | 101.4 |
| heloc | PFossil:two-level (macro distribution voting) | 69.33% | 101.4 |
| heloc | PFossil:two-level (micro distribution voting) | 69.33% | 101.4 |
| heloc | PFossil:two-level (voting) | 69.33% | 101.4 |
| heloc | Pypper:one-level (best rule) | 71.61% | 833.5 |
| heloc | Pypper:one-level (macro distribution voting) | 71.39% | 833.5 |
| heloc | Pypper:one-level (micro distribution voting) | 71.08% | 833.5 |
| heloc | Pypper:one-level (voting) | 71.32% | 833.5 |
| heloc | Pypper:single (best rule) | 70.45% | 833.5 |
| heloc | Pypper:single (macro distribution voting) | 70.47% | 833.5 |
| heloc | Pypper:single (micro distribution voting) | 70.23% | 833.5 |
| heloc | Pypper:single (voting) | 70.26% | 833.5 |
| heloc | Pypper:two-level (best rule) | 71.40% | 833.5 |
| heloc | Pypper:two-level (macro distribution voting) | 71.32% | 833.5 |
| heloc | Pypper:two-level (micro distribution voting) | 71.26% | 833.5 |
| heloc | Pypper:two-level (voting) | 71.48% | 833.5 |
| hepatitis | PFoil:one-level (best rule) | 80.04% | 657.2 |
| hepatitis | PFoil:one-level (macro distribution voting) | 81.42% | 657.2 |
| hepatitis | PFoil:one-level (micro distribution voting) | 81.33% | 657.2 |
| hepatitis | PFoil:one-level (voting) | 83.46% | 657.2 |
| hepatitis | PFoil:single (best rule) | 77.58% | 657.2 |
| hepatitis | PFoil:single (macro distribution voting) | 77.54% | 657.2 |
| hepatitis | PFoil:single (micro distribution voting) | 79.46% | 657.2 |
| hepatitis | PFoil:single (voting) | 76.29% | 657.2 |
| hepatitis | PFoil:two-level (best rule) | 83.96% | 657.2 |
| hepatitis | PFoil:two-level (macro distribution voting) | 82.79% | 657.2 |
| hepatitis | PFoil:two-level (micro distribution voting) | 83.29% | 657.2 |
| hepatitis | PFoil:two-level (voting) | 84.04% | 657.2 |
| hepatitis | PFossil:one-level (best rule) | 80.67% | 565.8 |
| hepatitis | PFossil:one-level (macro distribution voting) | 81.42% | 565.8 |
| hepatitis | PFossil:one-level (micro distribution voting) | 81.33% | 565.8 |
| hepatitis | PFossil:one-level (voting) | 81.42% | 565.8 |
| hepatitis | PFossil:single (best rule) | 78.75% | 565.8 |
| hepatitis | PFossil:single (macro distribution voting) | 77.54% | 565.8 |
| hepatitis | PFossil:single (micro distribution voting) | 80.04% | 565.8 |
| hepatitis | PFossil:single (voting) | 78.75% | 565.8 |
| hepatitis | PFossil:two-level (best rule) | 81.42% | 565.8 |
| hepatitis | PFossil:two-level (macro distribution voting) | 80.17% | 565.8 |
| hepatitis | PFossil:two-level (micro distribution voting) | 82.04% | 565.8 |
| hepatitis | PFossil:two-level (voting) | 82.67% | 565.8 |
| hepatitis | Pypper:one-level (best rule) | 80.00% | 200.1 |
| hepatitis | Pypper:one-level (macro distribution voting) | 81.50% | 200.1 |
| hepatitis | Pypper:one-level (micro distribution voting) | 80.00% | 200.1 |
| hepatitis | Pypper:one-level (voting) | 82.17% | 200.1 |
| hepatitis | Pypper:single (best rule) | 80.04% | 200.1 |
| hepatitis | Pypper:single (macro distribution voting) | 80.67% | 200.1 |
| hepatitis | Pypper:single (micro distribution voting) | 80.04% | 200.1 |
| hepatitis | Pypper:single (voting) | 80.67% | 200.1 |
| hepatitis | Pypper:two-level (best rule) | 80.83% | 200.1 |
| hepatitis | Pypper:two-level (macro distribution voting) | 80.88% | 200.1 |
| hepatitis | Pypper:two-level (micro distribution voting) | 81.46% | 200.1 |
| hepatitis | Pypper:two-level (voting) | 81.46% | 200.1 |
| ilpd | PFoil:one-level (best rule) | 72.55% | 2638.7 |
| ilpd | PFoil:one-level (macro distribution voting) | 70.83% | 2638.7 |
| ilpd | PFoil:one-level (micro distribution voting) | 71.69% | 2638.7 |
| ilpd | PFoil:one-level (voting) | 71.17% | 2638.7 |
| ilpd | PFoil:single (best rule) | 68.94% | 2638.7 |
| ilpd | PFoil:single (macro distribution voting) | 69.62% | 2638.7 |
| ilpd | PFoil:single (micro distribution voting) | 69.12% | 2638.7 |
| ilpd | PFoil:single (voting) | 68.42% | 2638.7 |
| ilpd | PFoil:two-level (best rule) | 70.31% | 2638.7 |
| ilpd | PFoil:two-level (macro distribution voting) | 72.19% | 2638.7 |
| ilpd | PFoil:two-level (micro distribution voting) | 70.84% | 2638.7 |
| ilpd | PFoil:two-level (voting) | 71.69% | 2638.7 |
| ilpd | PFossil:one-level (best rule) | 72.05% | 382.9 |
| ilpd | PFossil:one-level (macro distribution voting) | 71.18% | 382.9 |
| ilpd | PFossil:one-level (micro distribution voting) | 71.01% | 382.9 |
| ilpd | PFossil:one-level (voting) | 66.36% | 382.9 |
| ilpd | PFossil:single (best rule) | 63.28% | 382.9 |
| ilpd | PFossil:single (macro distribution voting) | 63.28% | 382.9 |
| ilpd | PFossil:single (micro distribution voting) | 63.28% | 382.9 |
| ilpd | PFossil:single (voting) | 63.28% | 382.9 |
| ilpd | PFossil:two-level (best rule) | 67.91% | 382.9 |
| ilpd | PFossil:two-level (macro distribution voting) | 67.74% | 382.9 |
| ilpd | PFossil:two-level (micro distribution voting) | 68.42% | 382.9 |
| ilpd | PFossil:two-level (voting) | 67.57% | 382.9 |
| ilpd | Pypper:one-level (best rule) | 69.81% | 240.8 |
| ilpd | Pypper:one-level (macro distribution voting) | 70.50% | 240.8 |
| ilpd | Pypper:one-level (micro distribution voting) | 71.01% | 240.8 |
| ilpd | Pypper:one-level (voting) | 69.82% | 240.8 |
| ilpd | Pypper:single (best rule) | 69.66% | 240.8 |
| ilpd | Pypper:single (macro distribution voting) | 69.66% | 240.8 |
| ilpd | Pypper:single (micro distribution voting) | 69.66% | 240.8 |
| ilpd | Pypper:single (voting) | 69.66% | 240.8 |
| ilpd | Pypper:two-level (best rule) | 69.99% | 240.8 |
| ilpd | Pypper:two-level (macro distribution voting) | 69.99% | 240.8 |
| ilpd | Pypper:two-level (micro distribution voting) | 69.48% | 240.8 |
| ilpd | Pypper:two-level (voting) | 69.48% | 240.8 |
| ionosphere | PFoil:one-level (best rule) | 75.20% | 696.4 |
| ionosphere | PFoil:one-level (macro distribution voting) | 92.01% | 696.4 |
| ionosphere | PFoil:one-level (micro distribution voting) | 91.73% | 696.4 |
| ionosphere | PFoil:one-level (voting) | 92.59% | 696.4 |
| ionosphere | PFoil:single (best rule) | 90.87% | 696.4 |
| ionosphere | PFoil:single (macro distribution voting) | 92.02% | 696.4 |
| ionosphere | PFoil:single (micro distribution voting) | 91.17% | 696.4 |
| ionosphere | PFoil:single (voting) | 92.30% | 696.4 |
| ionosphere | PFoil:two-level (best rule) | 92.30% | 696.4 |
| ionosphere | PFoil:two-level (macro distribution voting) | 92.30% | 696.4 |
| ionosphere | PFoil:two-level (micro distribution voting) | 92.58% | 696.4 |
| ionosphere | PFoil:two-level (voting) | 92.59% | 696.4 |
| ionosphere | PFossil:one-level (best rule) | 85.74% | 556.7 |
| ionosphere | PFossil:one-level (macro distribution voting) | 92.31% | 556.7 |
| ionosphere | PFossil:one-level (micro distribution voting) | 92.87% | 556.7 |
| ionosphere | PFossil:one-level (voting) | 91.74% | 556.7 |
| ionosphere | PFossil:single (best rule) | 89.75% | 556.7 |
| ionosphere | PFossil:single (macro distribution voting) | 90.03% | 556.7 |
| ionosphere | PFossil:single (micro distribution voting) | 89.75% | 556.7 |
| ionosphere | PFossil:single (voting) | 90.03% | 556.7 |
| ionosphere | PFossil:two-level (best rule) | 92.58% | 556.7 |
| ionosphere | PFossil:two-level (macro distribution voting) | 93.16% | 556.7 |
| ionosphere | PFossil:two-level (micro distribution voting) | 91.45% | 556.7 |
| ionosphere | PFossil:two-level (voting) | 92.87% | 556.7 |
| ionosphere | Pypper:one-level (best rule) | 85.75% | 280.1 |
| ionosphere | Pypper:one-level (macro distribution voting) | 90.32% | 280.1 |
| ionosphere | Pypper:one-level (micro distribution voting) | 90.32% | 280.1 |
| ionosphere | Pypper:one-level (voting) | 90.32% | 280.1 |
| ionosphere | Pypper:single (best rule) | 90.61% | 280.1 |
| ionosphere | Pypper:single (macro distribution voting) | 90.61% | 280.1 |
| ionosphere | Pypper:single (micro distribution voting) | 91.17% | 280.1 |
| ionosphere | Pypper:single (voting) | 90.89% | 280.1 |
| ionosphere | Pypper:two-level (best rule) | 90.60% | 280.1 |
| ionosphere | Pypper:two-level (macro distribution voting) | 90.60% | 280.1 |
| ionosphere | Pypper:two-level (micro distribution voting) | 90.32% | 280.1 |
| ionosphere | Pypper:two-level (voting) | 90.89% | 280.1 |
| iris | PFoil:one-level (best rule) | 94.00% | 333.2 |
| iris | PFoil:one-level (macro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:one-level (micro distribution voting) | 93.33% | 333.2 |
| iris | PFoil:one-level (voting) | 92.67% | 333.2 |
| iris | PFoil:single (best rule) | 92.67% | 333.2 |
| iris | PFoil:single (macro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:single (micro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:single (voting) | 90.67% | 333.2 |
| iris | PFoil:two-level (best rule) | 92.67% | 333.2 |
| iris | PFoil:two-level (macro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:two-level (micro distribution voting) | 92.67% | 333.2 |
| iris | PFoil:two-level (voting) | 93.33% | 333.2 |
| iris | PFossil:one-level (best rule) | 94.00% | 263.0 |
| iris | PFossil:one-level (macro distribution voting) | 92.67% | 263.0 |
| iris | PFossil:one-level (micro distribution voting) | 93.33% | 263.0 |
| iris | PFossil:one-level (voting) | 92.67% | 263.0 |
| iris | PFossil:single (best rule) | 92.00% | 263.0 |
| iris | PFossil:single (macro distribution voting) | 92.00% | 263.0 |
| iris | PFossil:single (micro distribution voting) | 92.00% | 263.0 |
| iris | PFossil:single (voting) | 91.33% | 263.0 |
| iris | PFossil:two-level (best rule) | 92.67% | 263.0 |
| iris | PFossil:two-level (macro distribution voting) | 92.00% | 263.0 |
| iris | PFossil:two-level (micro distribution voting) | 93.33% | 263.0 |
| iris | PFossil:two-level (voting) | 92.67% | 263.0 |
| iris | Pypper:one-level (best rule) | 95.33% | 175.4 |
| iris | Pypper:one-level (macro distribution voting) | 94.00% | 175.4 |
| iris | Pypper:one-level (micro distribution voting) | 94.00% | 175.4 |
| iris | Pypper:one-level (voting) | 94.00% | 175.4 |
| iris | Pypper:single (best rule) | 92.00% | 175.4 |
| iris | Pypper:single (macro distribution voting) | 92.00% | 175.4 |
| iris | Pypper:single (micro distribution voting) | 92.00% | 175.4 |
| iris | Pypper:single (voting) | 92.00% | 175.4 |
| iris | Pypper:two-level (best rule) | 94.00% | 175.4 |
| iris | Pypper:two-level (macro distribution voting) | 94.00% | 175.4 |
| iris | Pypper:two-level (micro distribution voting) | 94.00% | 175.4 |
| iris | Pypper:two-level (voting) | 94.00% | 175.4 |
| kc1 | PFoil:one-level (best rule) | 85.59% | 5708.8 |
| kc1 | PFoil:one-level (macro distribution voting) | 85.97% | 5708.8 |
| kc1 | PFoil:one-level (micro distribution voting) | 84.87% | 5708.8 |
| kc1 | PFoil:one-level (voting) | 85.54% | 5708.8 |
| kc1 | PFoil:single (best rule) | 85.87% | 5708.8 |
| kc1 | PFoil:single (macro distribution voting) | 86.06% | 5708.8 |
| kc1 | PFoil:single (micro distribution voting) | 85.87% | 5708.8 |
| kc1 | PFoil:single (voting) | 83.45% | 5708.8 |
| kc1 | PFoil:two-level (best rule) | 86.34% | 5708.8 |
| kc1 | PFoil:two-level (macro distribution voting) | 86.20% | 5708.8 |
| kc1 | PFoil:two-level (micro distribution voting) | 85.40% | 5708.8 |
| kc1 | PFoil:two-level (voting) | 85.78% | 5708.8 |
| kc1 | PFossil:one-level (best rule) | 84.83% | 271.4 |
| kc1 | PFossil:one-level (macro distribution voting) | 84.64% | 271.4 |
| kc1 | PFossil:one-level (micro distribution voting) | 84.64% | 271.4 |
| kc1 | PFossil:one-level (voting) | 82.60% | 271.4 |
| kc1 | PFossil:single (best rule) | 83.31% | 271.4 |
| kc1 | PFossil:single (macro distribution voting) | 83.31% | 271.4 |
| kc1 | PFossil:single (micro distribution voting) | 83.31% | 271.4 |
| kc1 | PFossil:single (voting) | 83.31% | 271.4 |
| kc1 | PFossil:two-level (best rule) | 85.54% | 271.4 |
| kc1 | PFossil:two-level (macro distribution voting) | 85.54% | 271.4 |
| kc1 | PFossil:two-level (micro distribution voting) | 85.54% | 271.4 |
| kc1 | PFossil:two-level (voting) | 85.59% | 271.4 |
| kc1 | Pypper:one-level (best rule) | 85.59% | 268.2 |
| kc1 | Pypper:one-level (macro distribution voting) | 85.54% | 268.2 |
| kc1 | Pypper:one-level (micro distribution voting) | 85.49% | 268.2 |
| kc1 | Pypper:one-level (voting) | 85.02% | 268.2 |
| kc1 | Pypper:single (best rule) | 84.64% | 268.2 |
| kc1 | Pypper:single (macro distribution voting) | 84.64% | 268.2 |
| kc1 | Pypper:single (micro distribution voting) | 84.64% | 268.2 |
| kc1 | Pypper:single (voting) | 84.64% | 268.2 |
| kc1 | Pypper:two-level (best rule) | 84.97% | 268.2 |
| kc1 | Pypper:two-level (macro distribution voting) | 84.92% | 268.2 |
| kc1 | Pypper:two-level (micro distribution voting) | 85.02% | 268.2 |
| kc1 | Pypper:two-level (voting) | 85.02% | 268.2 |
| kc2 | PFoil:one-level (best rule) | 85.03% | 1733.1 |
| kc2 | PFoil:one-level (macro distribution voting) | 84.08% | 1733.1 |
| kc2 | PFoil:one-level (micro distribution voting) | 84.09% | 1733.1 |
| kc2 | PFoil:one-level (voting) | 82.54% | 1733.1 |
| kc2 | PFoil:single (best rule) | 82.55% | 1733.1 |
| kc2 | PFoil:single (macro distribution voting) | 81.98% | 1733.1 |
| kc2 | PFoil:single (micro distribution voting) | 81.98% | 1733.1 |
| kc2 | PFoil:single (voting) | 81.41% | 1733.1 |
| kc2 | PFoil:two-level (best rule) | 84.27% | 1733.1 |
| kc2 | PFoil:two-level (macro distribution voting) | 84.08% | 1733.1 |
| kc2 | PFoil:two-level (micro distribution voting) | 83.88% | 1733.1 |
| kc2 | PFoil:two-level (voting) | 84.46% | 1733.1 |
| kc2 | PFossil:one-level (best rule) | 83.51% | 506.2 |
| kc2 | PFossil:one-level (macro distribution voting) | 83.89% | 506.2 |
| kc2 | PFossil:one-level (micro distribution voting) | 84.27% | 506.2 |
| kc2 | PFossil:one-level (voting) | 83.88% | 506.2 |
| kc2 | PFossil:single (best rule) | 82.74% | 506.2 |
| kc2 | PFossil:single (macro distribution voting) | 82.35% | 506.2 |
| kc2 | PFossil:single (micro distribution voting) | 81.79% | 506.2 |
| kc2 | PFossil:single (voting) | 82.55% | 506.2 |
| kc2 | PFossil:two-level (best rule) | 83.69% | 506.2 |
| kc2 | PFossil:two-level (macro distribution voting) | 83.12% | 506.2 |
| kc2 | PFossil:two-level (micro distribution voting) | 83.11% | 506.2 |
| kc2 | PFossil:two-level (voting) | 83.32% | 506.2 |
| kc2 | Pypper:one-level (best rule) | 83.70% | 193.7 |
| kc2 | Pypper:one-level (macro distribution voting) | 84.45% | 193.7 |
| kc2 | Pypper:one-level (micro distribution voting) | 83.89% | 193.7 |
| kc2 | Pypper:one-level (voting) | 83.89% | 193.7 |
| kc2 | Pypper:single (best rule) | 83.69% | 193.7 |
| kc2 | Pypper:single (macro distribution voting) | 83.69% | 193.7 |
| kc2 | Pypper:single (micro distribution voting) | 83.69% | 193.7 |
| kc2 | Pypper:single (voting) | 83.69% | 193.7 |
| kc2 | Pypper:two-level (best rule) | 83.51% | 193.7 |
| kc2 | Pypper:two-level (macro distribution voting) | 83.51% | 193.7 |
| kc2 | Pypper:two-level (micro distribution voting) | 83.89% | 193.7 |
| kc2 | Pypper:two-level (voting) | 83.31% | 193.7 |
| kr-vs-kp | PFoil:one-level (best rule) | 98.40% | 1835.2 |
| kr-vs-kp | PFoil:one-level (macro distribution voting) | 99.47% | 1835.2 |
| kr-vs-kp | PFoil:one-level (micro distribution voting) | 99.12% | 1835.2 |
| kr-vs-kp | PFoil:one-level (voting) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:single (best rule) | 99.34% | 1835.2 |
| kr-vs-kp | PFoil:single (macro distribution voting) | 99.22% | 1835.2 |
| kr-vs-kp | PFoil:single (micro distribution voting) | 99.09% | 1835.2 |
| kr-vs-kp | PFoil:single (voting) | 98.84% | 1835.2 |
| kr-vs-kp | PFoil:two-level (best rule) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:two-level (macro distribution voting) | 99.44% | 1835.2 |
| kr-vs-kp | PFoil:two-level (micro distribution voting) | 99.34% | 1835.2 |
| kr-vs-kp | PFoil:two-level (voting) | 99.50% | 1835.2 |
| kr-vs-kp | PFossil:one-level (best rule) | 98.56% | 698.2 |
| kr-vs-kp | PFossil:one-level (macro distribution voting) | 99.22% | 698.2 |
| kr-vs-kp | PFossil:one-level (micro distribution voting) | 94.21% | 698.2 |
| kr-vs-kp | PFossil:one-level (voting) | 96.62% | 698.2 |
| kr-vs-kp | PFossil:single (best rule) | 99.37% | 698.2 |
| kr-vs-kp | PFossil:single (macro distribution voting) | 99.41% | 698.2 |
| kr-vs-kp | PFossil:single (micro distribution voting) | 94.31% | 698.2 |
| kr-vs-kp | PFossil:single (voting) | 96.43% | 698.2 |
| kr-vs-kp | PFossil:two-level (best rule) | 99.34% | 698.2 |
| kr-vs-kp | PFossil:two-level (macro distribution voting) | 99.37% | 698.2 |
| kr-vs-kp | PFossil:two-level (micro distribution voting) | 94.31% | 698.2 |
| kr-vs-kp | PFossil:two-level (voting) | 96.40% | 698.2 |
| kr-vs-kp | Pypper:one-level (best rule) | 98.47% | 905.7 |
| kr-vs-kp | Pypper:one-level (macro distribution voting) | 99.19% | 905.7 |
| kr-vs-kp | Pypper:one-level (micro distribution voting) | 98.28% | 905.7 |
| kr-vs-kp | Pypper:one-level (voting) | 99.25% | 905.7 |
| kr-vs-kp | Pypper:single (best rule) | 99.16% | 905.7 |
| kr-vs-kp | Pypper:single (macro distribution voting) | 99.12% | 905.7 |
| kr-vs-kp | Pypper:single (micro distribution voting) | 99.03% | 905.7 |
| kr-vs-kp | Pypper:single (voting) | 98.91% | 905.7 |
| kr-vs-kp | Pypper:two-level (best rule) | 99.12% | 905.7 |
| kr-vs-kp | Pypper:two-level (macro distribution voting) | 99.19% | 905.7 |
| kr-vs-kp | Pypper:two-level (micro distribution voting) | 98.97% | 905.7 |
| kr-vs-kp | Pypper:two-level (voting) | 99.12% | 905.7 |
| led24 | PFoil:one-level (best rule) | 72.00% | 16575.1 |
| led24 | PFoil:one-level (macro distribution voting) | 72.12% | 16575.1 |
| led24 | PFoil:one-level (micro distribution voting) | 72.69% | 16575.1 |
| led24 | PFoil:one-level (voting) | 71.72% | 16575.1 |
| led24 | PFoil:single (best rule) | 61.81% | 16575.1 |
| led24 | PFoil:single (macro distribution voting) | 62.34% | 16575.1 |
| led24 | PFoil:single (micro distribution voting) | 61.97% | 16575.1 |
| led24 | PFoil:single (voting) | 56.84% | 16575.1 |
| led24 | PFoil:two-level (best rule) | 71.78% | 16575.1 |
| led24 | PFoil:two-level (macro distribution voting) | 72.00% | 16575.1 |
| led24 | PFoil:two-level (micro distribution voting) | 72.19% | 16575.1 |
| led24 | PFoil:two-level (voting) | 71.59% | 16575.1 |
| led24 | PFossil:one-level (best rule) | 54.37% | 3696.4 |
| led24 | PFossil:one-level (macro distribution voting) | 71.44% | 3696.4 |
| led24 | PFossil:one-level (micro distribution voting) | 71.41% | 3696.4 |
| led24 | PFossil:one-level (voting) | 71.81% | 3696.4 |
| led24 | PFossil:single (best rule) | 70.00% | 3696.4 |
| led24 | PFossil:single (macro distribution voting) | 70.28% | 3696.4 |
| led24 | PFossil:single (micro distribution voting) | 70.56% | 3696.4 |
| led24 | PFossil:single (voting) | 70.34% | 3696.4 |
| led24 | PFossil:two-level (best rule) | 71.28% | 3696.4 |
| led24 | PFossil:two-level (macro distribution voting) | 71.44% | 3696.4 |
| led24 | PFossil:two-level (micro distribution voting) | 71.50% | 3696.4 |
| led24 | PFossil:two-level (voting) | 71.47% | 3696.4 |
| led24 | Pypper:one-level (best rule) | 70.00% | 1200.0 |
| led24 | Pypper:one-level (macro distribution voting) | 72.53% | 1200.0 |
| led24 | Pypper:one-level (micro distribution voting) | 72.81% | 1200.0 |
| led24 | Pypper:one-level (voting) | 72.41% | 1200.0 |
| led24 | Pypper:single (best rule) | 68.00% | 1200.0 |
| led24 | Pypper:single (macro distribution voting) | 67.81% | 1200.0 |
| led24 | Pypper:single (micro distribution voting) | 67.88% | 1200.0 |
| led24 | Pypper:single (voting) | 67.47% | 1200.0 |
| led24 | Pypper:two-level (best rule) | 72.44% | 1200.0 |
| led24 | Pypper:two-level (macro distribution voting) | 72.38% | 1200.0 |
| led24 | Pypper:two-level (micro distribution voting) | 72.84% | 1200.0 |
| led24 | Pypper:two-level (voting) | 72.44% | 1200.0 |
| letter | PFoil:one-level (best rule) | 84.25% | 47344.3 |
| letter | PFoil:one-level (macro distribution voting) | 93.89% | 47344.3 |
| letter | PFoil:one-level (micro distribution voting) | 82.65% | 47344.3 |
| letter | PFoil:one-level (voting) | 93.38% | 47344.3 |
| letter | PFoil:single (best rule) | 88.78% | 47344.3 |
| letter | PFoil:single (macro distribution voting) | 89.12% | 47344.3 |
| letter | PFoil:single (micro distribution voting) | 72.10% | 47344.3 |
| letter | PFoil:single (voting) | 80.75% | 47344.3 |
| letter | PFoil:two-level (best rule) | 93.55% | 47344.3 |
| letter | PFoil:two-level (macro distribution voting) | 93.93% | 47344.3 |
| letter | PFoil:two-level (micro distribution voting) | 91.48% | 47344.3 |
| letter | PFoil:two-level (voting) | 92.69% | 47344.3 |
| letter | PFossil:one-level (best rule) | 86.00% | 7834.6 |
| letter | PFossil:one-level (macro distribution voting) | 87.33% | 7834.6 |
| letter | PFossil:one-level (micro distribution voting) | 82.23% | 7834.6 |
| letter | PFossil:one-level (voting) | 86.45% | 7834.6 |
| letter | PFossil:single (best rule) | 76.22% | 7834.6 |
| letter | PFossil:single (macro distribution voting) | 76.44% | 7834.6 |
| letter | PFossil:single (micro distribution voting) | 74.39% | 7834.6 |
| letter | PFossil:single (voting) | 75.29% | 7834.6 |
| letter | PFossil:two-level (best rule) | 87.28% | 7834.6 |
| letter | PFossil:two-level (macro distribution voting) | 87.41% | 7834.6 |
| letter | PFossil:two-level (micro distribution voting) | 84.54% | 7834.6 |
| letter | PFossil:two-level (voting) | 86.39% | 7834.6 |
| letter | Pypper:one-level (best rule) | 85.48% | 10391.8 |
| letter | Pypper:one-level (macro distribution voting) | 89.09% | 10391.8 |
| letter | Pypper:one-level (micro distribution voting) | 86.23% | 10391.8 |
| letter | Pypper:one-level (voting) | 88.90% | 10391.8 |
| letter | Pypper:single (best rule) | 77.53% | 10391.8 |
| letter | Pypper:single (macro distribution voting) | 77.68% | 10391.8 |
| letter | Pypper:single (micro distribution voting) | 77.17% | 10391.8 |
| letter | Pypper:single (voting) | 77.28% | 10391.8 |
| letter | Pypper:two-level (best rule) | 88.83% | 10391.8 |
| letter | Pypper:two-level (macro distribution voting) | 89.02% | 10391.8 |
| letter | Pypper:two-level (micro distribution voting) | 88.35% | 10391.8 |
| letter | Pypper:two-level (voting) | 88.88% | 10391.8 |
| lymph | PFoil:one-level (best rule) | 77.76% | 693.6 |
| lymph | PFoil:one-level (macro distribution voting) | 82.57% | 693.6 |
| lymph | PFoil:one-level (micro distribution voting) | 81.86% | 693.6 |
| lymph | PFoil:one-level (voting) | 82.57% | 693.6 |
| lymph | PFoil:single (best rule) | 82.57% | 693.6 |
| lymph | PFoil:single (macro distribution voting) | 81.19% | 693.6 |
| lymph | PFoil:single (micro distribution voting) | 77.86% | 693.6 |
| lymph | PFoil:single (voting) | 78.48% | 693.6 |
| lymph | PFoil:two-level (best rule) | 83.90% | 693.6 |
| lymph | PFoil:two-level (macro distribution voting) | 82.57% | 693.6 |
| lymph | PFoil:two-level (micro distribution voting) | 82.52% | 693.6 |
| lymph | PFoil:two-level (voting) | 83.24% | 693.6 |
| lymph | PFossil:one-level (best rule) | 77.19% | 602.2 |
| lymph | PFossil:one-level (macro distribution voting) | 83.81% | 602.2 |
| lymph | PFossil:one-level (micro distribution voting) | 79.86% | 602.2 |
| lymph | PFossil:one-level (voting) | 82.43% | 602.2 |
| lymph | PFossil:single (best rule) | 81.71% | 602.2 |
| lymph | PFossil:single (macro distribution voting) | 79.71% | 602.2 |
| lymph | PFossil:single (micro distribution voting) | 81.05% | 602.2 |
| lymph | PFossil:single (voting) | 81.10% | 602.2 |
| lymph | PFossil:two-level (best rule) | 82.48% | 602.2 |
| lymph | PFossil:two-level (macro distribution voting) | 83.14% | 602.2 |
| lymph | PFossil:two-level (micro distribution voting) | 80.43% | 602.2 |
| lymph | PFossil:two-level (voting) | 81.76% | 602.2 |
| lymph | Pypper:one-level (best rule) | 79.14% | 261.5 |
| lymph | Pypper:one-level (macro distribution voting) | 79.86% | 261.5 |
| lymph | Pypper:one-level (micro distribution voting) | 77.76% | 261.5 |
| lymph | Pypper:one-level (voting) | 79.86% | 261.5 |
| lymph | Pypper:single (best rule) | 80.52% | 261.5 |
| lymph | Pypper:single (macro distribution voting) | 82.52% | 261.5 |
| lymph | Pypper:single (micro distribution voting) | 77.81% | 261.5 |
| lymph | Pypper:single (voting) | 78.52% | 261.5 |
| lymph | Pypper:two-level (best rule) | 79.14% | 261.5 |
| lymph | Pypper:two-level (macro distribution voting) | 81.19% | 261.5 |
| lymph | Pypper:two-level (micro distribution voting) | 78.48% | 261.5 |
| lymph | Pypper:two-level (voting) | 78.43% | 261.5 |
| mofn-3-7-10 | PFoil:one-level (best rule) | 84.67% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level (macro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level (micro distribution voting) | 98.87% | 3067.1 |
| mofn-3-7-10 | PFoil:one-level (voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:single (best rule) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:single (macro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:single (micro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:single (voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level (best rule) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level (macro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level (micro distribution voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFoil:two-level (voting) | 100.00% | 3067.1 |
| mofn-3-7-10 | PFossil:one-level (best rule) | 85.65% | 767.9 |
| mofn-3-7-10 | PFossil:one-level (macro distribution voting) | 99.85% | 767.9 |
| mofn-3-7-10 | PFossil:one-level (micro distribution voting) | 83.61% | 767.9 |
| mofn-3-7-10 | PFossil:one-level (voting) | 100.00% | 767.9 |
| mofn-3-7-10 | PFossil:single (best rule) | 83.68% | 767.9 |
| mofn-3-7-10 | PFossil:single (macro distribution voting) | 83.76% | 767.9 |
| mofn-3-7-10 | PFossil:single (micro distribution voting) | 83.31% | 767.9 |
| mofn-3-7-10 | PFossil:single (voting) | 83.61% | 767.9 |
| mofn-3-7-10 | PFossil:two-level (best rule) | 97.73% | 767.9 |
| mofn-3-7-10 | PFossil:two-level (macro distribution voting) | 99.47% | 767.9 |
| mofn-3-7-10 | PFossil:two-level (micro distribution voting) | 90.48% | 767.9 |
| mofn-3-7-10 | PFossil:two-level (voting) | 93.65% | 767.9 |
| mofn-3-7-10 | Pypper:one-level (best rule) | 84.59% | 1057.8 |
| mofn-3-7-10 | Pypper:one-level (macro distribution voting) | 100.00% | 1057.8 |
| mofn-3-7-10 | Pypper:one-level (micro distribution voting) | 84.29% | 1057.8 |
| mofn-3-7-10 | Pypper:one-level (voting) | 100.00% | 1057.8 |
| mofn-3-7-10 | Pypper:single (best rule) | 97.21% | 1057.8 |
| mofn-3-7-10 | Pypper:single (macro distribution voting) | 97.66% | 1057.8 |
| mofn-3-7-10 | Pypper:single (micro distribution voting) | 95.85% | 1057.8 |
| mofn-3-7-10 | Pypper:single (voting) | 96.00% | 1057.8 |
| mofn-3-7-10 | Pypper:two-level (best rule) | 97.81% | 1057.8 |
| mofn-3-7-10 | Pypper:two-level (macro distribution voting) | 99.40% | 1057.8 |
| mofn-3-7-10 | Pypper:two-level (micro distribution voting) | 96.22% | 1057.8 |
| mofn-3-7-10 | Pypper:two-level (voting) | 96.90% | 1057.8 |
| molecular-biology_promoters | PFoil:one-level (best rule) | 85.73% | 347.1 |
| molecular-biology_promoters | PFoil:one-level (macro distribution voting) | 90.55% | 347.1 |
| molecular-biology_promoters | PFoil:one-level (micro distribution voting) | 89.64% | 347.1 |
| molecular-biology_promoters | PFoil:one-level (voting) | 93.45% | 347.1 |
| molecular-biology_promoters | PFoil:single (best rule) | 86.09% | 347.1 |
| molecular-biology_promoters | PFoil:single (macro distribution voting) | 85.18% | 347.1 |
| molecular-biology_promoters | PFoil:single (micro distribution voting) | 86.09% | 347.1 |
| molecular-biology_promoters | PFoil:single (voting) | 83.36% | 347.1 |
| molecular-biology_promoters | PFoil:two-level (best rule) | 90.64% | 347.1 |
| molecular-biology_promoters | PFoil:two-level (macro distribution voting) | 91.55% | 347.1 |
| molecular-biology_promoters | PFoil:two-level (micro distribution voting) | 90.64% | 347.1 |
| molecular-biology_promoters | PFoil:two-level (voting) | 92.45% | 347.1 |
| molecular-biology_promoters | PFossil:one-level (best rule) | 75.27% | 330.2 |
| molecular-biology_promoters | PFossil:one-level (macro distribution voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:one-level (micro distribution voting) | 90.73% | 330.2 |
| molecular-biology_promoters | PFossil:one-level (voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:single (best rule) | 82.27% | 330.2 |
| molecular-biology_promoters | PFossil:single (macro distribution voting) | 82.18% | 330.2 |
| molecular-biology_promoters | PFossil:single (micro distribution voting) | 81.36% | 330.2 |
| molecular-biology_promoters | PFossil:single (voting) | 81.27% | 330.2 |
| molecular-biology_promoters | PFossil:two-level (best rule) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:two-level (macro distribution voting) | 93.45% | 330.2 |
| molecular-biology_promoters | PFossil:two-level (micro distribution voting) | 92.55% | 330.2 |
| molecular-biology_promoters | PFossil:two-level (voting) | 92.45% | 330.2 |
| molecular-biology_promoters | Pypper:one-level (best rule) | 81.00% | 192.1 |
| molecular-biology_promoters | Pypper:one-level (macro distribution voting) | 88.73% | 192.1 |
| molecular-biology_promoters | Pypper:one-level (micro distribution voting) | 86.82% | 192.1 |
| molecular-biology_promoters | Pypper:one-level (voting) | 87.73% | 192.1 |
| molecular-biology_promoters | Pypper:single (best rule) | 81.36% | 192.1 |
| molecular-biology_promoters | Pypper:single (macro distribution voting) | 81.36% | 192.1 |
| molecular-biology_promoters | Pypper:single (micro distribution voting) | 82.18% | 192.1 |
| molecular-biology_promoters | Pypper:single (voting) | 81.27% | 192.1 |
| molecular-biology_promoters | Pypper:two-level (best rule) | 86.82% | 192.1 |
| molecular-biology_promoters | Pypper:two-level (macro distribution voting) | 88.64% | 192.1 |
| molecular-biology_promoters | Pypper:two-level (micro distribution voting) | 86.82% | 192.1 |
| molecular-biology_promoters | Pypper:two-level (voting) | 88.64% | 192.1 |
| monks-problems-1 | PFoil:one-level (best rule) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level (macro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level (micro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:one-level (voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:single (best rule) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:single (macro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:single (micro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:single (voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level (best rule) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level (macro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level (micro distribution voting) | 100.00% | 638.9 |
| monks-problems-1 | PFoil:two-level (voting) | 100.00% | 638.9 |
| monks-problems-1 | PFossil:one-level (best rule) | 80.22% | 179.2 |
| monks-problems-1 | PFossil:one-level (macro distribution voting) | 88.31% | 179.2 |
| monks-problems-1 | PFossil:one-level (micro distribution voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:one-level (voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:single (best rule) | 80.41% | 179.2 |
| monks-problems-1 | PFossil:single (macro distribution voting) | 80.41% | 179.2 |
| monks-problems-1 | PFossil:single (micro distribution voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:single (voting) | 75.91% | 179.2 |
| monks-problems-1 | PFossil:two-level (best rule) | 75.36% | 179.2 |
| monks-problems-1 | PFossil:two-level (macro distribution voting) | 75.91% | 179.2 |
| monks-problems-1 | PFossil:two-level (micro distribution voting) | 74.64% | 179.2 |
| monks-problems-1 | PFossil:two-level (voting) | 74.64% | 179.2 |
| monks-problems-1 | Pypper:one-level (best rule) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:one-level (macro distribution voting) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:one-level (micro distribution voting) | 93.34% | 317.1 |
| monks-problems-1 | Pypper:one-level (voting) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:single (best rule) | 94.76% | 317.1 |
| monks-problems-1 | Pypper:single (macro distribution voting) | 94.76% | 317.1 |
| monks-problems-1 | Pypper:single (micro distribution voting) | 92.06% | 317.1 |
| monks-problems-1 | Pypper:single (voting) | 92.06% | 317.1 |
| monks-problems-1 | Pypper:two-level (best rule) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:two-level (macro distribution voting) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:two-level (micro distribution voting) | 100.00% | 317.1 |
| monks-problems-1 | Pypper:two-level (voting) | 100.00% | 317.1 |
| monks-problems-2 | PFoil:one-level (best rule) | 73.71% | 3080.6 |
| monks-problems-2 | PFoil:one-level (macro distribution voting) | 90.68% | 3080.6 |
| monks-problems-2 | PFoil:one-level (micro distribution voting) | 81.19% | 3080.6 |
| monks-problems-2 | PFoil:one-level (voting) | 93.67% | 3080.6 |
| monks-problems-2 | PFoil:single (best rule) | 97.50% | 3080.6 |
| monks-problems-2 | PFoil:single (macro distribution voting) | 97.17% | 3080.6 |
| monks-problems-2 | PFoil:single (micro distribution voting) | 97.50% | 3080.6 |
| monks-problems-2 | PFoil:single (voting) | 97.34% | 3080.6 |
| monks-problems-2 | PFoil:two-level (best rule) | 95.84% | 3080.6 |
| monks-problems-2 | PFoil:two-level (macro distribution voting) | 95.01% | 3080.6 |
| monks-problems-2 | PFoil:two-level (micro distribution voting) | 93.68% | 3080.6 |
| monks-problems-2 | PFoil:two-level (voting) | 93.84% | 3080.6 |
| monks-problems-2 | PFossil:one-level (best rule) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level (macro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level (micro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:one-level (voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:single (best rule) | 66.72% | 9.4 |
| monks-problems-2 | PFossil:single (macro distribution voting) | 66.72% | 9.4 |
| monks-problems-2 | PFossil:single (micro distribution voting) | 66.72% | 9.4 |
| monks-problems-2 | PFossil:single (voting) | 66.72% | 9.4 |
| monks-problems-2 | PFossil:two-level (best rule) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level (macro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level (micro distribution voting) | 69.54% | 9.4 |
| monks-problems-2 | PFossil:two-level (voting) | 69.54% | 9.4 |
| monks-problems-2 | Pypper:one-level (best rule) | 80.20% | 360.3 |
| monks-problems-2 | Pypper:one-level (macro distribution voting) | 69.55% | 360.3 |
| monks-problems-2 | Pypper:one-level (micro distribution voting) | 66.06% | 360.3 |
| monks-problems-2 | Pypper:one-level (voting) | 68.22% | 360.3 |
| monks-problems-2 | Pypper:single (best rule) | 72.03% | 360.3 |
| monks-problems-2 | Pypper:single (macro distribution voting) | 72.03% | 360.3 |
| monks-problems-2 | Pypper:single (micro distribution voting) | 71.87% | 360.3 |
| monks-problems-2 | Pypper:single (voting) | 71.87% | 360.3 |
| monks-problems-2 | Pypper:two-level (best rule) | 71.21% | 360.3 |
| monks-problems-2 | Pypper:two-level (macro distribution voting) | 70.72% | 360.3 |
| monks-problems-2 | Pypper:two-level (micro distribution voting) | 69.89% | 360.3 |
| monks-problems-2 | Pypper:two-level (voting) | 69.89% | 360.3 |
| monks-problems-3 | PFoil:one-level (best rule) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level (macro distribution voting) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level (micro distribution voting) | 98.92% | 796.1 |
| monks-problems-3 | PFoil:one-level (voting) | 98.56% | 796.1 |
| monks-problems-3 | PFoil:single (best rule) | 97.66% | 796.1 |
| monks-problems-3 | PFoil:single (macro distribution voting) | 97.66% | 796.1 |
| monks-problems-3 | PFoil:single (micro distribution voting) | 97.66% | 796.1 |
| monks-problems-3 | PFoil:single (voting) | 96.76% | 796.1 |
| monks-problems-3 | PFoil:two-level (best rule) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level (macro distribution voting) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level (micro distribution voting) | 98.02% | 796.1 |
| monks-problems-3 | PFoil:two-level (voting) | 97.84% | 796.1 |
| monks-problems-3 | PFossil:one-level (best rule) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:one-level (macro distribution voting) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:one-level (micro distribution voting) | 96.40% | 321.7 |
| monks-problems-3 | PFossil:one-level (voting) | 98.02% | 321.7 |
| monks-problems-3 | PFossil:single (best rule) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:single (macro distribution voting) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:single (micro distribution voting) | 96.40% | 321.7 |
| monks-problems-3 | PFossil:single (voting) | 98.38% | 321.7 |
| monks-problems-3 | PFossil:two-level (best rule) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:two-level (macro distribution voting) | 98.92% | 321.7 |
| monks-problems-3 | PFossil:two-level (micro distribution voting) | 96.40% | 321.7 |
| monks-problems-3 | PFossil:two-level (voting) | 98.92% | 321.7 |
| monks-problems-3 | Pypper:one-level (best rule) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:one-level (macro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:one-level (micro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:one-level (voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:single (best rule) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:single (macro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:single (micro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:single (voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:two-level (best rule) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:two-level (macro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:two-level (micro distribution voting) | 98.92% | 246.9 |
| monks-problems-3 | Pypper:two-level (voting) | 98.92% | 246.9 |
| mushroom | PFoil:one-level (best rule) | 99.93% | 609.6 |
| mushroom | PFoil:one-level (macro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:one-level (micro distribution voting) | 99.93% | 609.6 |
| mushroom | PFoil:one-level (voting) | 100.00% | 609.6 |
| mushroom | PFoil:single (best rule) | 100.00% | 609.6 |
| mushroom | PFoil:single (macro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:single (micro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:single (voting) | 100.00% | 609.6 |
| mushroom | PFoil:two-level (best rule) | 100.00% | 609.6 |
| mushroom | PFoil:two-level (macro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:two-level (micro distribution voting) | 100.00% | 609.6 |
| mushroom | PFoil:two-level (voting) | 100.00% | 609.6 |
| mushroom | PFossil:one-level (best rule) | 99.82% | 582.8 |
| mushroom | PFossil:one-level (macro distribution voting) | 99.95% | 582.8 |
| mushroom | PFossil:one-level (micro distribution voting) | 99.80% | 582.8 |
| mushroom | PFossil:one-level (voting) | 99.95% | 582.8 |
| mushroom | PFossil:single (best rule) | 99.95% | 582.8 |
| mushroom | PFossil:single (macro distribution voting) | 100.00% | 582.8 |
| mushroom | PFossil:single (micro distribution voting) | 99.95% | 582.8 |
| mushroom | PFossil:single (voting) | 99.95% | 582.8 |
| mushroom | PFossil:two-level (best rule) | 99.95% | 582.8 |
| mushroom | PFossil:two-level (macro distribution voting) | 100.00% | 582.8 |
| mushroom | PFossil:two-level (micro distribution voting) | 99.95% | 582.8 |
| mushroom | PFossil:two-level (voting) | 99.95% | 582.8 |
| mushroom | Pypper:one-level (best rule) | 99.86% | 534.9 |
| mushroom | Pypper:one-level (macro distribution voting) | 99.96% | 534.9 |
| mushroom | Pypper:one-level (micro distribution voting) | 99.83% | 534.9 |
| mushroom | Pypper:one-level (voting) | 99.96% | 534.9 |
| mushroom | Pypper:single (best rule) | 99.94% | 534.9 |
| mushroom | Pypper:single (macro distribution voting) | 99.94% | 534.9 |
| mushroom | Pypper:single (micro distribution voting) | 99.94% | 534.9 |
| mushroom | Pypper:single (voting) | 99.94% | 534.9 |
| mushroom | Pypper:two-level (best rule) | 99.96% | 534.9 |
| mushroom | Pypper:two-level (macro distribution voting) | 99.96% | 534.9 |
| mushroom | Pypper:two-level (micro distribution voting) | 99.96% | 534.9 |
| mushroom | Pypper:two-level (voting) | 99.96% | 534.9 |
| optdigits | PFoil:one-level (best rule) | 89.45% | 5824.9 |
| optdigits | PFoil:one-level (macro distribution voting) | 97.90% | 5824.9 |
| optdigits | PFoil:one-level (micro distribution voting) | 97.05% | 5824.9 |
| optdigits | PFoil:one-level (voting) | 97.94% | 5824.9 |
| optdigits | PFoil:single (best rule) | 93.35% | 5824.9 |
| optdigits | PFoil:single (macro distribution voting) | 94.02% | 5824.9 |
| optdigits | PFoil:single (micro distribution voting) | 92.44% | 5824.9 |
| optdigits | PFoil:single (voting) | 92.88% | 5824.9 |
| optdigits | PFoil:two-level (best rule) | 97.65% | 5824.9 |
| optdigits | PFoil:two-level (macro distribution voting) | 97.92% | 5824.9 |
| optdigits | PFoil:two-level (micro distribution voting) | 97.56% | 5824.9 |
| optdigits | PFoil:two-level (voting) | 97.86% | 5824.9 |
| optdigits | PFossil:one-level (best rule) | 89.79% | 3682.1 |
| optdigits | PFossil:one-level (macro distribution voting) | 97.01% | 3682.1 |
| optdigits | PFossil:one-level (micro distribution voting) | 95.82% | 3682.1 |
| optdigits | PFossil:one-level (voting) | 96.87% | 3682.1 |
| optdigits | PFossil:single (best rule) | 90.30% | 3682.1 |
| optdigits | PFossil:single (macro distribution voting) | 90.39% | 3682.1 |
| optdigits | PFossil:single (micro distribution voting) | 90.11% | 3682.1 |
| optdigits | PFossil:single (voting) | 90.27% | 3682.1 |
| optdigits | PFossil:two-level (best rule) | 96.74% | 3682.1 |
| optdigits | PFossil:two-level (macro distribution voting) | 97.10% | 3682.1 |
| optdigits | PFossil:two-level (micro distribution voting) | 96.33% | 3682.1 |
| optdigits | PFossil:two-level (voting) | 96.71% | 3682.1 |
| optdigits | Pypper:one-level (best rule) | 91.41% | 2502.6 |
| optdigits | Pypper:one-level (macro distribution voting) | 96.92% | 2502.6 |
| optdigits | Pypper:one-level (micro distribution voting) | 96.05% | 2502.6 |
| optdigits | Pypper:one-level (voting) | 96.89% | 2502.6 |
| optdigits | Pypper:single (best rule) | 90.18% | 2502.6 |
| optdigits | Pypper:single (macro distribution voting) | 90.37% | 2502.6 |
| optdigits | Pypper:single (micro distribution voting) | 90.27% | 2502.6 |
| optdigits | Pypper:single (voting) | 90.14% | 2502.6 |
| optdigits | Pypper:two-level (best rule) | 96.69% | 2502.6 |
| optdigits | Pypper:two-level (macro distribution voting) | 96.85% | 2502.6 |
| optdigits | Pypper:two-level (micro distribution voting) | 96.51% | 2502.6 |
| optdigits | Pypper:two-level (voting) | 96.83% | 2502.6 |
| ozone-level-8hr | PFoil:one-level (best rule) | 93.69% | 2185.7 |
| ozone-level-8hr | PFoil:one-level (macro distribution voting) | 93.92% | 2185.7 |
| ozone-level-8hr | PFoil:one-level (micro distribution voting) | 93.69% | 2185.7 |
| ozone-level-8hr | PFoil:one-level (voting) | 94.20% | 2185.7 |
| ozone-level-8hr | PFoil:single (best rule) | 93.29% | 2185.7 |
| ozone-level-8hr | PFoil:single (macro distribution voting) | 93.53% | 2185.7 |
| ozone-level-8hr | PFoil:single (micro distribution voting) | 93.37% | 2185.7 |
| ozone-level-8hr | PFoil:single (voting) | 93.09% | 2185.7 |
| ozone-level-8hr | PFoil:two-level (best rule) | 93.92% | 2185.7 |
| ozone-level-8hr | PFoil:two-level (macro distribution voting) | 94.16% | 2185.7 |
| ozone-level-8hr | PFoil:two-level (micro distribution voting) | 93.88% | 2185.7 |
| ozone-level-8hr | PFoil:two-level (voting) | 94.00% | 2185.7 |
| ozone-level-8hr | PFossil:one-level (best rule) | 93.69% | 536.2 |
| ozone-level-8hr | PFossil:one-level (macro distribution voting) | 93.80% | 536.2 |
| ozone-level-8hr | PFossil:one-level (micro distribution voting) | 93.69% | 536.2 |
| ozone-level-8hr | PFossil:one-level (voting) | 93.88% | 536.2 |
| ozone-level-8hr | PFossil:single (best rule) | 93.33% | 536.2 |
| ozone-level-8hr | PFossil:single (macro distribution voting) | 93.33% | 536.2 |
| ozone-level-8hr | PFossil:single (micro distribution voting) | 93.33% | 536.2 |
| ozone-level-8hr | PFossil:single (voting) | 93.33% | 536.2 |
| ozone-level-8hr | PFossil:two-level (best rule) | 93.77% | 536.2 |
| ozone-level-8hr | PFossil:two-level (macro distribution voting) | 93.84% | 536.2 |
| ozone-level-8hr | PFossil:two-level (micro distribution voting) | 93.73% | 536.2 |
| ozone-level-8hr | PFossil:two-level (voting) | 93.73% | 536.2 |
| ozone-level-8hr | Pypper:one-level (best rule) | 93.73% | 363.5 |
| ozone-level-8hr | Pypper:one-level (macro distribution voting) | 94.04% | 363.5 |
| ozone-level-8hr | Pypper:one-level (micro distribution voting) | 93.69% | 363.5 |
| ozone-level-8hr | Pypper:one-level (voting) | 94.20% | 363.5 |
| ozone-level-8hr | Pypper:single (best rule) | 93.33% | 363.5 |
| ozone-level-8hr | Pypper:single (macro distribution voting) | 93.33% | 363.5 |
| ozone-level-8hr | Pypper:single (micro distribution voting) | 93.33% | 363.5 |
| ozone-level-8hr | Pypper:single (voting) | 93.37% | 363.5 |
| ozone-level-8hr | Pypper:two-level (best rule) | 94.08% | 363.5 |
| ozone-level-8hr | Pypper:two-level (macro distribution voting) | 94.08% | 363.5 |
| ozone-level-8hr | Pypper:two-level (micro distribution voting) | 94.08% | 363.5 |
| ozone-level-8hr | Pypper:two-level (voting) | 94.08% | 363.5 |
| pc1 | PFoil:one-level (best rule) | 93.06% | 2056.1 |
| pc1 | PFoil:one-level (macro distribution voting) | 93.15% | 2056.1 |
| pc1 | PFoil:one-level (micro distribution voting) | 93.06% | 2056.1 |
| pc1 | PFoil:one-level (voting) | 92.70% | 2056.1 |
| pc1 | PFoil:single (best rule) | 93.06% | 2056.1 |
| pc1 | PFoil:single (macro distribution voting) | 93.06% | 2056.1 |
| pc1 | PFoil:single (micro distribution voting) | 92.97% | 2056.1 |
| pc1 | PFoil:single (voting) | 91.52% | 2056.1 |
| pc1 | PFoil:two-level (best rule) | 93.15% | 2056.1 |
| pc1 | PFoil:two-level (macro distribution voting) | 93.15% | 2056.1 |
| pc1 | PFoil:two-level (micro distribution voting) | 93.24% | 2056.1 |
| pc1 | PFoil:two-level (voting) | 92.97% | 2056.1 |
| pc1 | PFossil:one-level (best rule) | 93.06% | 363.4 |
| pc1 | PFossil:one-level (macro distribution voting) | 92.70% | 363.4 |
| pc1 | PFossil:one-level (micro distribution voting) | 93.15% | 363.4 |
| pc1 | PFossil:one-level (voting) | 92.79% | 363.4 |
| pc1 | PFossil:single (best rule) | 92.88% | 363.4 |
| pc1 | PFossil:single (macro distribution voting) | 92.79% | 363.4 |
| pc1 | PFossil:single (micro distribution voting) | 92.97% | 363.4 |
| pc1 | PFossil:single (voting) | 92.88% | 363.4 |
| pc1 | PFossil:two-level (best rule) | 92.97% | 363.4 |
| pc1 | PFossil:two-level (macro distribution voting) | 92.70% | 363.4 |
| pc1 | PFossil:two-level (micro distribution voting) | 92.97% | 363.4 |
| pc1 | PFossil:two-level (voting) | 92.88% | 363.4 |
| pc1 | Pypper:one-level (best rule) | 93.06% | 199.2 |
| pc1 | Pypper:one-level (macro distribution voting) | 93.24% | 199.2 |
| pc1 | Pypper:one-level (micro distribution voting) | 93.06% | 199.2 |
| pc1 | Pypper:one-level (voting) | 93.51% | 199.2 |
| pc1 | Pypper:single (best rule) | 92.61% | 199.2 |
| pc1 | Pypper:single (macro distribution voting) | 92.61% | 199.2 |
| pc1 | Pypper:single (micro distribution voting) | 92.61% | 199.2 |
| pc1 | Pypper:single (voting) | 92.61% | 199.2 |
| pc1 | Pypper:two-level (best rule) | 93.33% | 199.2 |
| pc1 | Pypper:two-level (macro distribution voting) | 93.33% | 199.2 |
| pc1 | Pypper:two-level (micro distribution voting) | 93.33% | 199.2 |
| pc1 | Pypper:two-level (voting) | 93.42% | 199.2 |
| pc3 | PFoil:one-level (best rule) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level (macro distribution voting) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level (micro distribution voting) | 89.76% | 2927.1 |
| pc3 | PFoil:one-level (voting) | 89.70% | 2927.1 |
| pc3 | PFoil:single (best rule) | 87.78% | 2927.1 |
| pc3 | PFoil:single (macro distribution voting) | 87.84% | 2927.1 |
| pc3 | PFoil:single (micro distribution voting) | 88.16% | 2927.1 |
| pc3 | PFoil:single (voting) | 86.88% | 2927.1 |
| pc3 | PFoil:two-level (best rule) | 90.08% | 2927.1 |
| pc3 | PFoil:two-level (macro distribution voting) | 90.08% | 2927.1 |
| pc3 | PFoil:two-level (micro distribution voting) | 89.70% | 2927.1 |
| pc3 | PFoil:two-level (voting) | 89.76% | 2927.1 |
| pc3 | PFossil:one-level (best rule) | 89.83% | 347.7 |
| pc3 | PFossil:one-level (macro distribution voting) | 89.76% | 347.7 |
| pc3 | PFossil:one-level (micro distribution voting) | 89.76% | 347.7 |
| pc3 | PFossil:one-level (voting) | 86.69% | 347.7 |
| pc3 | PFossil:single (best rule) | 85.09% | 347.7 |
| pc3 | PFossil:single (macro distribution voting) | 85.09% | 347.7 |
| pc3 | PFossil:single (micro distribution voting) | 85.09% | 347.7 |
| pc3 | PFossil:single (voting) | 85.09% | 347.7 |
| pc3 | PFossil:two-level (best rule) | 88.10% | 347.7 |
| pc3 | PFossil:two-level (macro distribution voting) | 88.16% | 347.7 |
| pc3 | PFossil:two-level (micro distribution voting) | 88.10% | 347.7 |
| pc3 | PFossil:two-level (voting) | 87.97% | 347.7 |
| pc3 | Pypper:one-level (best rule) | 89.83% | 230.9 |
| pc3 | Pypper:one-level (macro distribution voting) | 89.70% | 230.9 |
| pc3 | Pypper:one-level (micro distribution voting) | 89.70% | 230.9 |
| pc3 | Pypper:one-level (voting) | 89.19% | 230.9 |
| pc3 | Pypper:single (best rule) | 88.93% | 230.9 |
| pc3 | Pypper:single (macro distribution voting) | 88.93% | 230.9 |
| pc3 | Pypper:single (micro distribution voting) | 88.93% | 230.9 |
| pc3 | Pypper:single (voting) | 88.93% | 230.9 |
| pc3 | Pypper:two-level (best rule) | 89.25% | 230.9 |
| pc3 | Pypper:two-level (macro distribution voting) | 89.19% | 230.9 |
| pc3 | Pypper:two-level (micro distribution voting) | 89.25% | 230.9 |
| pc3 | Pypper:two-level (voting) | 89.12% | 230.9 |
| pc4 | PFoil:one-level (best rule) | 88.27% | 2187.8 |
| pc4 | PFoil:one-level (macro distribution voting) | 90.33% | 2187.8 |
| pc4 | PFoil:one-level (micro distribution voting) | 88.13% | 2187.8 |
| pc4 | PFoil:one-level (voting) | 90.60% | 2187.8 |
| pc4 | PFoil:single (best rule) | 89.85% | 2187.8 |
| pc4 | PFoil:single (macro distribution voting) | 90.19% | 2187.8 |
| pc4 | PFoil:single (micro distribution voting) | 89.23% | 2187.8 |
| pc4 | PFoil:single (voting) | 88.96% | 2187.8 |
| pc4 | PFoil:two-level (best rule) | 89.98% | 2187.8 |
| pc4 | PFoil:two-level (macro distribution voting) | 90.26% | 2187.8 |
| pc4 | PFoil:two-level (micro distribution voting) | 89.85% | 2187.8 |
| pc4 | PFoil:two-level (voting) | 90.05% | 2187.8 |
| pc4 | PFossil:one-level (best rule) | 88.82% | 726.2 |
| pc4 | PFossil:one-level (macro distribution voting) | 89.99% | 726.2 |
| pc4 | PFossil:one-level (micro distribution voting) | 89.58% | 726.2 |
| pc4 | PFossil:one-level (voting) | 89.71% | 726.2 |
| pc4 | PFossil:single (best rule) | 89.51% | 726.2 |
| pc4 | PFossil:single (macro distribution voting) | 89.71% | 726.2 |
| pc4 | PFossil:single (micro distribution voting) | 89.58% | 726.2 |
| pc4 | PFossil:single (voting) | 89.78% | 726.2 |
| pc4 | PFossil:two-level (best rule) | 89.78% | 726.2 |
| pc4 | PFossil:two-level (macro distribution voting) | 90.12% | 726.2 |
| pc4 | PFossil:two-level (micro distribution voting) | 89.64% | 726.2 |
| pc4 | PFossil:two-level (voting) | 89.92% | 726.2 |
| pc4 | Pypper:one-level (best rule) | 89.09% | 346.8 |
| pc4 | Pypper:one-level (macro distribution voting) | 89.03% | 346.8 |
| pc4 | Pypper:one-level (micro distribution voting) | 89.71% | 346.8 |
| pc4 | Pypper:one-level (voting) | 89.16% | 346.8 |
| pc4 | Pypper:single (best rule) | 87.79% | 346.8 |
| pc4 | Pypper:single (macro distribution voting) | 87.79% | 346.8 |
| pc4 | Pypper:single (micro distribution voting) | 87.79% | 346.8 |
| pc4 | Pypper:single (voting) | 87.79% | 346.8 |
| pc4 | Pypper:two-level (best rule) | 89.71% | 346.8 |
| pc4 | Pypper:two-level (macro distribution voting) | 89.78% | 346.8 |
| pc4 | Pypper:two-level (micro distribution voting) | 89.78% | 346.8 |
| pc4 | Pypper:two-level (voting) | 89.85% | 346.8 |
| phoneme | PFoil:one-level (best rule) | 84.46% | 17689.9 |
| phoneme | PFoil:one-level (macro distribution voting) | 85.86% | 17689.9 |
| phoneme | PFoil:one-level (micro distribution voting) | 82.29% | 17689.9 |
| phoneme | PFoil:one-level (voting) | 86.29% | 17689.9 |
| phoneme | PFoil:single (best rule) | 85.33% | 17689.9 |
| phoneme | PFoil:single (macro distribution voting) | 85.44% | 17689.9 |
| phoneme | PFoil:single (micro distribution voting) | 81.36% | 17689.9 |
| phoneme | PFoil:single (voting) | 84.18% | 17689.9 |
| phoneme | PFoil:two-level (best rule) | 86.03% | 17689.9 |
| phoneme | PFoil:two-level (macro distribution voting) | 85.90% | 17689.9 |
| phoneme | PFoil:two-level (micro distribution voting) | 84.86% | 17689.9 |
| phoneme | PFoil:two-level (voting) | 85.60% | 17689.9 |
| phoneme | PFossil:one-level (best rule) | 79.01% | 254.1 |
| phoneme | PFossil:one-level (macro distribution voting) | 78.48% | 254.1 |
| phoneme | PFossil:one-level (micro distribution voting) | 78.65% | 254.1 |
| phoneme | PFossil:one-level (voting) | 77.98% | 254.1 |
| phoneme | PFossil:single (best rule) | 78.35% | 254.1 |
| phoneme | PFossil:single (macro distribution voting) | 78.39% | 254.1 |
| phoneme | PFossil:single (micro distribution voting) | 78.35% | 254.1 |
| phoneme | PFossil:single (voting) | 78.46% | 254.1 |
| phoneme | PFossil:two-level (best rule) | 77.46% | 254.1 |
| phoneme | PFossil:two-level (macro distribution voting) | 77.46% | 254.1 |
| phoneme | PFossil:two-level (micro distribution voting) | 77.46% | 254.1 |
| phoneme | PFossil:two-level (voting) | 77.53% | 254.1 |
| phoneme | Pypper:one-level (best rule) | 83.40% | 1199.6 |
| phoneme | Pypper:one-level (macro distribution voting) | 84.75% | 1199.6 |
| phoneme | Pypper:one-level (micro distribution voting) | 83.40% | 1199.6 |
| phoneme | Pypper:one-level (voting) | 84.27% | 1199.6 |
| phoneme | Pypper:single (best rule) | 83.53% | 1199.6 |
| phoneme | Pypper:single (macro distribution voting) | 83.40% | 1199.6 |
| phoneme | Pypper:single (micro distribution voting) | 83.24% | 1199.6 |
| phoneme | Pypper:single (voting) | 83.40% | 1199.6 |
| phoneme | Pypper:two-level (best rule) | 84.73% | 1199.6 |
| phoneme | Pypper:two-level (macro distribution voting) | 84.70% | 1199.6 |
| phoneme | Pypper:two-level (micro distribution voting) | 84.49% | 1199.6 |
| phoneme | Pypper:two-level (voting) | 84.59% | 1199.6 |
| primary-tumor | PFoil:one-level (best rule) | 43.96% | 4370.6 |
| primary-tumor | PFoil:one-level (macro distribution voting) | 46.90% | 4370.6 |
| primary-tumor | PFoil:one-level (micro distribution voting) | 43.67% | 4370.6 |
| primary-tumor | PFoil:one-level (voting) | 45.13% | 4370.6 |
| primary-tumor | PFoil:single (best rule) | 43.35% | 4370.6 |
| primary-tumor | PFoil:single (macro distribution voting) | 43.65% | 4370.6 |
| primary-tumor | PFoil:single (micro distribution voting) | 37.75% | 4370.6 |
| primary-tumor | PFoil:single (voting) | 36.28% | 4370.6 |
| primary-tumor | PFoil:two-level (best rule) | 46.60% | 4370.6 |
| primary-tumor | PFoil:two-level (macro distribution voting) | 44.83% | 4370.6 |
| primary-tumor | PFoil:two-level (micro distribution voting) | 45.71% | 4370.6 |
| primary-tumor | PFoil:two-level (voting) | 42.80% | 4370.6 |
| primary-tumor | PFossil:one-level (best rule) | 42.17% | 3367.6 |
| primary-tumor | PFossil:one-level (macro distribution voting) | 44.84% | 3367.6 |
| primary-tumor | PFossil:one-level (micro distribution voting) | 46.02% | 3367.6 |
| primary-tumor | PFossil:one-level (voting) | 43.96% | 3367.6 |
| primary-tumor | PFossil:single (best rule) | 43.38% | 3367.6 |
| primary-tumor | PFossil:single (macro distribution voting) | 41.61% | 3367.6 |
| primary-tumor | PFossil:single (micro distribution voting) | 41.01% | 3367.6 |
| primary-tumor | PFossil:single (voting) | 41.01% | 3367.6 |
| primary-tumor | PFossil:two-level (best rule) | 44.55% | 3367.6 |
| primary-tumor | PFossil:two-level (macro distribution voting) | 44.55% | 3367.6 |
| primary-tumor | PFossil:two-level (micro distribution voting) | 43.95% | 3367.6 |
| primary-tumor | PFossil:two-level (voting) | 44.53% | 3367.6 |
| primary-tumor | Pypper:one-level (best rule) | 43.07% | 343.0 |
| primary-tumor | Pypper:one-level (macro distribution voting) | 46.32% | 343.0 |
| primary-tumor | Pypper:one-level (micro distribution voting) | 44.82% | 343.0 |
| primary-tumor | Pypper:one-level (voting) | 45.13% | 343.0 |
| primary-tumor | Pypper:single (best rule) | 38.65% | 343.0 |
| primary-tumor | Pypper:single (macro distribution voting) | 38.07% | 343.0 |
| primary-tumor | Pypper:single (micro distribution voting) | 38.95% | 343.0 |
| primary-tumor | Pypper:single (voting) | 37.77% | 343.0 |
| primary-tumor | Pypper:two-level (best rule) | 45.13% | 343.0 |
| primary-tumor | Pypper:two-level (macro distribution voting) | 45.13% | 343.0 |
| primary-tumor | Pypper:two-level (micro distribution voting) | 45.12% | 343.0 |
| primary-tumor | Pypper:two-level (voting) | 44.83% | 343.0 |
| qsar-biodeg | PFoil:one-level (best rule) | 84.36% | 2573.5 |
| qsar-biodeg | PFoil:one-level (macro distribution voting) | 86.73% | 2573.5 |
| qsar-biodeg | PFoil:one-level (micro distribution voting) | 84.07% | 2573.5 |
| qsar-biodeg | PFoil:one-level (voting) | 86.07% | 2573.5 |
| qsar-biodeg | PFoil:single (best rule) | 84.55% | 2573.5 |
| qsar-biodeg | PFoil:single (macro distribution voting) | 84.17% | 2573.5 |
| qsar-biodeg | PFoil:single (micro distribution voting) | 81.23% | 2573.5 |
| qsar-biodeg | PFoil:single (voting) | 83.89% | 2573.5 |
| qsar-biodeg | PFoil:two-level (best rule) | 86.73% | 2573.5 |
| qsar-biodeg | PFoil:two-level (macro distribution voting) | 86.83% | 2573.5 |
| qsar-biodeg | PFoil:two-level (micro distribution voting) | 85.50% | 2573.5 |
| qsar-biodeg | PFoil:two-level (voting) | 86.73% | 2573.5 |
| qsar-biodeg | PFossil:one-level (best rule) | 83.31% | 674.0 |
| qsar-biodeg | PFossil:one-level (macro distribution voting) | 86.17% | 674.0 |
| qsar-biodeg | PFossil:one-level (micro distribution voting) | 83.60% | 674.0 |
| qsar-biodeg | PFossil:one-level (voting) | 85.79% | 674.0 |
| qsar-biodeg | PFossil:single (best rule) | 84.65% | 674.0 |
| qsar-biodeg | PFossil:single (macro distribution voting) | 84.74% | 674.0 |
| qsar-biodeg | PFossil:single (micro distribution voting) | 84.17% | 674.0 |
| qsar-biodeg | PFossil:single (voting) | 83.60% | 674.0 |
| qsar-biodeg | PFossil:two-level (best rule) | 85.69% | 674.0 |
| qsar-biodeg | PFossil:two-level (macro distribution voting) | 85.98% | 674.0 |
| qsar-biodeg | PFossil:two-level (micro distribution voting) | 84.56% | 674.0 |
| qsar-biodeg | PFossil:two-level (voting) | 84.93% | 674.0 |
| qsar-biodeg | Pypper:one-level (best rule) | 85.59% | 574.5 |
| qsar-biodeg | Pypper:one-level (macro distribution voting) | 84.65% | 574.5 |
| qsar-biodeg | Pypper:one-level (micro distribution voting) | 83.70% | 574.5 |
| qsar-biodeg | Pypper:one-level (voting) | 84.55% | 574.5 |
| qsar-biodeg | Pypper:single (best rule) | 85.03% | 574.5 |
| qsar-biodeg | Pypper:single (macro distribution voting) | 85.03% | 574.5 |
| qsar-biodeg | Pypper:single (micro distribution voting) | 84.36% | 574.5 |
| qsar-biodeg | Pypper:single (voting) | 84.65% | 574.5 |
| qsar-biodeg | Pypper:two-level (best rule) | 84.74% | 574.5 |
| qsar-biodeg | Pypper:two-level (macro distribution voting) | 84.83% | 574.5 |
| qsar-biodeg | Pypper:two-level (micro distribution voting) | 85.12% | 574.5 |
| qsar-biodeg | Pypper:two-level (voting) | 85.02% | 574.5 |
| segment | PFoil:one-level (best rule) | 94.11% | 3060.6 |
| segment | PFoil:one-level (macro distribution voting) | 96.75% | 3060.6 |
| segment | PFoil:one-level (micro distribution voting) | 95.58% | 3060.6 |
| segment | PFoil:one-level (voting) | 96.84% | 3060.6 |
| segment | PFoil:single (best rule) | 95.19% | 3060.6 |
| segment | PFoil:single (macro distribution voting) | 95.19% | 3060.6 |
| segment | PFoil:single (micro distribution voting) | 93.03% | 3060.6 |
| segment | PFoil:single (voting) | 90.91% | 3060.6 |
| segment | PFoil:two-level (best rule) | 96.71% | 3060.6 |
| segment | PFoil:two-level (macro distribution voting) | 96.93% | 3060.6 |
| segment | PFoil:two-level (micro distribution voting) | 96.75% | 3060.6 |
| segment | PFoil:two-level (voting) | 96.84% | 3060.6 |
| segment | PFossil:one-level (best rule) | 95.15% | 1586.1 |
| segment | PFossil:one-level (macro distribution voting) | 95.63% | 1586.1 |
| segment | PFossil:one-level (micro distribution voting) | 94.94% | 1586.1 |
| segment | PFossil:one-level (voting) | 95.58% | 1586.1 |
| segment | PFossil:single (best rule) | 93.33% | 1586.1 |
| segment | PFossil:single (macro distribution voting) | 93.16% | 1586.1 |
| segment | PFossil:single (micro distribution voting) | 93.12% | 1586.1 |
| segment | PFossil:single (voting) | 92.73% | 1586.1 |
| segment | PFossil:two-level (best rule) | 95.41% | 1586.1 |
| segment | PFossil:two-level (macro distribution voting) | 95.89% | 1586.1 |
| segment | PFossil:two-level (micro distribution voting) | 95.24% | 1586.1 |
| segment | PFossil:two-level (voting) | 95.89% | 1586.1 |
| segment | Pypper:one-level (best rule) | 94.07% | 930.9 |
| segment | Pypper:one-level (macro distribution voting) | 95.06% | 930.9 |
| segment | Pypper:one-level (micro distribution voting) | 94.76% | 930.9 |
| segment | Pypper:one-level (voting) | 95.24% | 930.9 |
| segment | Pypper:single (best rule) | 93.46% | 930.9 |
| segment | Pypper:single (macro distribution voting) | 93.51% | 930.9 |
| segment | Pypper:single (micro distribution voting) | 93.20% | 930.9 |
| segment | Pypper:single (voting) | 93.29% | 930.9 |
| segment | Pypper:two-level (best rule) | 95.11% | 930.9 |
| segment | Pypper:two-level (macro distribution voting) | 95.19% | 930.9 |
| segment | Pypper:two-level (micro distribution voting) | 95.02% | 930.9 |
| segment | Pypper:two-level (voting) | 95.24% | 930.9 |
| sick | PFoil:one-level (best rule) | 97.16% | 2521.9 |
| sick | PFoil:one-level (macro distribution voting) | 98.38% | 2521.9 |
| sick | PFoil:one-level (micro distribution voting) | 96.42% | 2521.9 |
| sick | PFoil:one-level (voting) | 98.36% | 2521.9 |
| sick | PFoil:single (best rule) | 97.99% | 2521.9 |
| sick | PFoil:single (macro distribution voting) | 98.14% | 2521.9 |
| sick | PFoil:single (micro distribution voting) | 96.95% | 2521.9 |
| sick | PFoil:single (voting) | 97.51% | 2521.9 |
| sick | PFoil:two-level (best rule) | 98.28% | 2521.9 |
| sick | PFoil:two-level (macro distribution voting) | 98.36% | 2521.9 |
| sick | PFoil:two-level (micro distribution voting) | 97.99% | 2521.9 |
| sick | PFoil:two-level (voting) | 98.20% | 2521.9 |
| sick | PFossil:one-level (best rule) | 96.85% | 806.0 |
| sick | PFossil:one-level (macro distribution voting) | 98.28% | 806.0 |
| sick | PFossil:one-level (micro distribution voting) | 98.01% | 806.0 |
| sick | PFossil:one-level (voting) | 98.36% | 806.0 |
| sick | PFossil:single (best rule) | 98.30% | 806.0 |
| sick | PFossil:single (macro distribution voting) | 98.25% | 806.0 |
| sick | PFossil:single (micro distribution voting) | 98.20% | 806.0 |
| sick | PFossil:single (voting) | 98.20% | 806.0 |
| sick | PFossil:two-level (best rule) | 98.30% | 806.0 |
| sick | PFossil:two-level (macro distribution voting) | 98.38% | 806.0 |
| sick | PFossil:two-level (micro distribution voting) | 98.25% | 806.0 |
| sick | PFossil:two-level (voting) | 98.36% | 806.0 |
| sick | Pypper:one-level (best rule) | 97.67% | 405.1 |
| sick | Pypper:one-level (macro distribution voting) | 98.28% | 405.1 |
| sick | Pypper:one-level (micro distribution voting) | 98.06% | 405.1 |
| sick | Pypper:one-level (voting) | 98.14% | 405.1 |
| sick | Pypper:single (best rule) | 98.20% | 405.1 |
| sick | Pypper:single (macro distribution voting) | 98.20% | 405.1 |
| sick | Pypper:single (micro distribution voting) | 98.22% | 405.1 |
| sick | Pypper:single (voting) | 98.22% | 405.1 |
| sick | Pypper:two-level (best rule) | 98.30% | 405.1 |
| sick | Pypper:two-level (macro distribution voting) | 98.30% | 405.1 |
| sick | Pypper:two-level (micro distribution voting) | 98.30% | 405.1 |
| sick | Pypper:two-level (voting) | 98.38% | 405.1 |
| solar-flare | PFoil:one-level (best rule) | 73.65% | 4341.4 |
| solar-flare | PFoil:one-level (macro distribution voting) | 74.86% | 4341.4 |
| solar-flare | PFoil:one-level (micro distribution voting) | 74.11% | 4341.4 |
| solar-flare | PFoil:one-level (voting) | 72.33% | 4341.4 |
| solar-flare | PFoil:single (best rule) | 73.73% | 4341.4 |
| solar-flare | PFoil:single (macro distribution voting) | 74.39% | 4341.4 |
| solar-flare | PFoil:single (micro distribution voting) | 74.02% | 4341.4 |
| solar-flare | PFoil:single (voting) | 71.30% | 4341.4 |
| solar-flare | PFoil:two-level (best rule) | 73.55% | 4341.4 |
| solar-flare | PFoil:two-level (macro distribution voting) | 74.48% | 4341.4 |
| solar-flare | PFoil:two-level (micro distribution voting) | 75.24% | 4341.4 |
| solar-flare | PFoil:two-level (voting) | 71.58% | 4341.4 |
| solar-flare | PFossil:one-level (best rule) | 72.04% | 911.9 |
| solar-flare | PFossil:one-level (macro distribution voting) | 75.14% | 911.9 |
| solar-flare | PFossil:one-level (micro distribution voting) | 74.96% | 911.9 |
| solar-flare | PFossil:one-level (voting) | 74.20% | 911.9 |
| solar-flare | PFossil:single (best rule) | 74.02% | 911.9 |
| solar-flare | PFossil:single (macro distribution voting) | 74.77% | 911.9 |
| solar-flare | PFossil:single (micro distribution voting) | 74.30% | 911.9 |
| solar-flare | PFossil:single (voting) | 70.07% | 911.9 |
| solar-flare | PFossil:two-level (best rule) | 73.83% | 911.9 |
| solar-flare | PFossil:two-level (macro distribution voting) | 74.20% | 911.9 |
| solar-flare | PFossil:two-level (micro distribution voting) | 75.14% | 911.9 |
| solar-flare | PFossil:two-level (voting) | 69.14% | 911.9 |
| solar-flare | Pypper:one-level (best rule) | 73.37% | 368.6 |
| solar-flare | Pypper:one-level (macro distribution voting) | 74.77% | 368.6 |
| solar-flare | Pypper:one-level (micro distribution voting) | 74.59% | 368.6 |
| solar-flare | Pypper:one-level (voting) | 74.87% | 368.6 |
| solar-flare | Pypper:single (best rule) | 69.43% | 368.6 |
| solar-flare | Pypper:single (macro distribution voting) | 69.52% | 368.6 |
| solar-flare | Pypper:single (micro distribution voting) | 69.24% | 368.6 |
| solar-flare | Pypper:single (voting) | 69.14% | 368.6 |
| solar-flare | Pypper:two-level (best rule) | 74.68% | 368.6 |
| solar-flare | Pypper:two-level (macro distribution voting) | 74.40% | 368.6 |
| solar-flare | Pypper:two-level (micro distribution voting) | 74.77% | 368.6 |
| solar-flare | Pypper:two-level (voting) | 74.40% | 368.6 |
| sonar | PFoil:one-level (best rule) | 71.62% | 601.7 |
| sonar | PFoil:one-level (macro distribution voting) | 80.81% | 601.7 |
| sonar | PFoil:one-level (micro distribution voting) | 79.38% | 601.7 |
| sonar | PFoil:one-level (voting) | 80.33% | 601.7 |
| sonar | PFoil:single (best rule) | 77.90% | 601.7 |
| sonar | PFoil:single (macro distribution voting) | 75.07% | 601.7 |
| sonar | PFoil:single (micro distribution voting) | 76.00% | 601.7 |
| sonar | PFoil:single (voting) | 75.07% | 601.7 |
| sonar | PFoil:two-level (best rule) | 79.86% | 601.7 |
| sonar | PFoil:two-level (macro distribution voting) | 80.33% | 601.7 |
| sonar | PFoil:two-level (micro distribution voting) | 79.86% | 601.7 |
| sonar | PFoil:two-level (voting) | 79.38% | 601.7 |
| sonar | PFossil:one-level (best rule) | 68.29% | 664.2 |
| sonar | PFossil:one-level (macro distribution voting) | 78.43% | 664.2 |
| sonar | PFossil:one-level (micro distribution voting) | 72.64% | 664.2 |
| sonar | PFossil:one-level (voting) | 78.90% | 664.2 |
| sonar | PFossil:single (best rule) | 73.55% | 664.2 |
| sonar | PFossil:single (macro distribution voting) | 75.07% | 664.2 |
| sonar | PFossil:single (micro distribution voting) | 74.02% | 664.2 |
| sonar | PFossil:single (voting) | 75.52% | 664.2 |
| sonar | PFossil:two-level (best rule) | 76.02% | 664.2 |
| sonar | PFossil:two-level (macro distribution voting) | 75.52% | 664.2 |
| sonar | PFossil:two-level (micro distribution voting) | 76.52% | 664.2 |
| sonar | PFossil:two-level (voting) | 75.07% | 664.2 |
| sonar | Pypper:one-level (best rule) | 72.57% | 284.5 |
| sonar | Pypper:one-level (macro distribution voting) | 77.48% | 284.5 |
| sonar | Pypper:one-level (micro distribution voting) | 75.07% | 284.5 |
| sonar | Pypper:one-level (voting) | 77.95% | 284.5 |
| sonar | Pypper:single (best rule) | 76.48% | 284.5 |
| sonar | Pypper:single (macro distribution voting) | 75.52% | 284.5 |
| sonar | Pypper:single (micro distribution voting) | 75.50% | 284.5 |
| sonar | Pypper:single (voting) | 73.14% | 284.5 |
| sonar | Pypper:two-level (best rule) | 77.50% | 284.5 |
| sonar | Pypper:two-level (macro distribution voting) | 77.98% | 284.5 |
| sonar | Pypper:two-level (micro distribution voting) | 76.55% | 284.5 |
| sonar | Pypper:two-level (voting) | 76.55% | 284.5 |
| soybean | PFoil:one-level (best rule) | 83.61% | 2382.2 |
| soybean | PFoil:one-level (macro distribution voting) | 92.83% | 2382.2 |
| soybean | PFoil:one-level (micro distribution voting) | 90.06% | 2382.2 |
| soybean | PFoil:one-level (voting) | 92.54% | 2382.2 |
| soybean | PFoil:single (best rule) | 90.92% | 2382.2 |
| soybean | PFoil:single (macro distribution voting) | 90.35% | 2382.2 |
| soybean | PFoil:single (micro distribution voting) | 86.98% | 2382.2 |
| soybean | PFoil:single (voting) | 84.78% | 2382.2 |
| soybean | PFoil:two-level (best rule) | 92.40% | 2382.2 |
| soybean | PFoil:two-level (macro distribution voting) | 91.37% | 2382.2 |
| soybean | PFoil:two-level (micro distribution voting) | 91.23% | 2382.2 |
| soybean | PFoil:two-level (voting) | 88.58% | 2382.2 |
| soybean | PFossil:one-level (best rule) | 84.64% | 2005.2 |
| soybean | PFossil:one-level (macro distribution voting) | 93.71% | 2005.2 |
| soybean | PFossil:one-level (micro distribution voting) | 91.52% | 2005.2 |
| soybean | PFossil:one-level (voting) | 93.12% | 2005.2 |
| soybean | PFossil:single (best rule) | 92.24% | 2005.2 |
| soybean | PFossil:single (macro distribution voting) | 91.36% | 2005.2 |
| soybean | PFossil:single (micro distribution voting) | 91.66% | 2005.2 |
| soybean | PFossil:single (voting) | 90.34% | 2005.2 |
| soybean | PFossil:two-level (best rule) | 93.12% | 2005.2 |
| soybean | PFossil:two-level (macro distribution voting) | 92.83% | 2005.2 |
| soybean | PFossil:two-level (micro distribution voting) | 92.39% | 2005.2 |
| soybean | PFossil:two-level (voting) | 92.10% | 2005.2 |
| soybean | Pypper:one-level (best rule) | 86.83% | 1210.9 |
| soybean | Pypper:one-level (macro distribution voting) | 92.39% | 1210.9 |
| soybean | Pypper:one-level (micro distribution voting) | 89.91% | 1210.9 |
| soybean | Pypper:one-level (voting) | 91.81% | 1210.9 |
| soybean | Pypper:single (best rule) | 89.91% | 1210.9 |
| soybean | Pypper:single (macro distribution voting) | 89.91% | 1210.9 |
| soybean | Pypper:single (micro distribution voting) | 87.28% | 1210.9 |
| soybean | Pypper:single (voting) | 86.84% | 1210.9 |
| soybean | Pypper:two-level (best rule) | 92.39% | 1210.9 |
| soybean | Pypper:two-level (macro distribution voting) | 92.69% | 1210.9 |
| soybean | Pypper:two-level (micro distribution voting) | 90.93% | 1210.9 |
| soybean | Pypper:two-level (voting) | 90.35% | 1210.9 |
| spambase | PFoil:one-level (best rule) | 93.70% | 5496.1 |
| spambase | PFoil:one-level (macro distribution voting) | 94.76% | 5496.1 |
| spambase | PFoil:one-level (micro distribution voting) | 93.54% | 5496.1 |
| spambase | PFoil:one-level (voting) | 94.83% | 5496.1 |
| spambase | PFoil:single (best rule) | 93.87% | 5496.1 |
| spambase | PFoil:single (macro distribution voting) | 94.02% | 5496.1 |
| spambase | PFoil:single (micro distribution voting) | 90.50% | 5496.1 |
| spambase | PFoil:single (voting) | 93.59% | 5496.1 |
| spambase | PFoil:two-level (best rule) | 95.04% | 5496.1 |
| spambase | PFoil:two-level (macro distribution voting) | 94.94% | 5496.1 |
| spambase | PFoil:two-level (micro distribution voting) | 94.28% | 5496.1 |
| spambase | PFoil:two-level (voting) | 95.02% | 5496.1 |
| spambase | PFossil:one-level (best rule) | 93.57% | 574.3 |
| spambase | PFossil:one-level (macro distribution voting) | 93.50% | 574.3 |
| spambase | PFossil:one-level (micro distribution voting) | 92.37% | 574.3 |
| spambase | PFossil:one-level (voting) | 93.48% | 574.3 |
| spambase | PFossil:single (best rule) | 92.81% | 574.3 |
| spambase | PFossil:single (macro distribution voting) | 92.76% | 574.3 |
| spambase | PFossil:single (micro distribution voting) | 91.81% | 574.3 |
| spambase | PFossil:single (voting) | 92.41% | 574.3 |
| spambase | PFossil:two-level (best rule) | 93.44% | 574.3 |
| spambase | PFossil:two-level (macro distribution voting) | 93.46% | 574.3 |
| spambase | PFossil:two-level (micro distribution voting) | 92.89% | 574.3 |
| spambase | PFossil:two-level (voting) | 93.09% | 574.3 |
| spambase | Pypper:one-level (best rule) | 93.65% | 1097.9 |
| spambase | Pypper:one-level (macro distribution voting) | 93.96% | 1097.9 |
| spambase | Pypper:one-level (micro distribution voting) | 93.24% | 1097.9 |
| spambase | Pypper:one-level (voting) | 93.91% | 1097.9 |
| spambase | Pypper:single (best rule) | 92.96% | 1097.9 |
| spambase | Pypper:single (macro distribution voting) | 92.87% | 1097.9 |
| spambase | Pypper:single (micro distribution voting) | 92.76% | 1097.9 |
| spambase | Pypper:single (voting) | 92.68% | 1097.9 |
| spambase | Pypper:two-level (best rule) | 94.04% | 1097.9 |
| spambase | Pypper:two-level (macro distribution voting) | 94.00% | 1097.9 |
| spambase | Pypper:two-level (micro distribution voting) | 93.74% | 1097.9 |
| spambase | Pypper:two-level (voting) | 93.85% | 1097.9 |
| splice | PFoil:one-level (best rule) | 94.33% | 2978.9 |
| splice | PFoil:one-level (macro distribution voting) | 96.14% | 2978.9 |
| splice | PFoil:one-level (micro distribution voting) | 95.96% | 2978.9 |
| splice | PFoil:one-level (voting) | 96.05% | 2978.9 |
| splice | PFoil:single (best rule) | 92.76% | 2978.9 |
| splice | PFoil:single (macro distribution voting) | 92.82% | 2978.9 |
| splice | PFoil:single (micro distribution voting) | 92.79% | 2978.9 |
| splice | PFoil:single (voting) | 92.41% | 2978.9 |
| splice | PFoil:two-level (best rule) | 96.18% | 2978.9 |
| splice | PFoil:two-level (macro distribution voting) | 96.27% | 2978.9 |
| splice | PFoil:two-level (micro distribution voting) | 96.14% | 2978.9 |
| splice | PFoil:two-level (voting) | 96.27% | 2978.9 |
| splice | PFossil:one-level (best rule) | 89.22% | 1442.8 |
| splice | PFossil:one-level (macro distribution voting) | 95.64% | 1442.8 |
| splice | PFossil:one-level (micro distribution voting) | 90.66% | 1442.8 |
| splice | PFossil:one-level (voting) | 95.20% | 1442.8 |
| splice | PFossil:single (best rule) | 94.23% | 1442.8 |
| splice | PFossil:single (macro distribution voting) | 94.29% | 1442.8 |
| splice | PFossil:single (micro distribution voting) | 90.38% | 1442.8 |
| splice | PFossil:single (voting) | 89.09% | 1442.8 |
| splice | PFossil:two-level (best rule) | 95.30% | 1442.8 |
| splice | PFossil:two-level (macro distribution voting) | 95.49% | 1442.8 |
| splice | PFossil:two-level (micro distribution voting) | 90.78% | 1442.8 |
| splice | PFossil:two-level (voting) | 91.00% | 1442.8 |
| splice | Pypper:one-level (best rule) | 95.24% | 846.2 |
| splice | Pypper:one-level (macro distribution voting) | 95.52% | 846.2 |
| splice | Pypper:one-level (micro distribution voting) | 95.02% | 846.2 |
| splice | Pypper:one-level (voting) | 95.49% | 846.2 |
| splice | Pypper:single (best rule) | 93.45% | 846.2 |
| splice | Pypper:single (macro distribution voting) | 93.70% | 846.2 |
| splice | Pypper:single (micro distribution voting) | 93.42% | 846.2 |
| splice | Pypper:single (voting) | 93.51% | 846.2 |
| splice | Pypper:two-level (best rule) | 95.30% | 846.2 |
| splice | Pypper:two-level (macro distribution voting) | 95.45% | 846.2 |
| splice | Pypper:two-level (micro distribution voting) | 95.17% | 846.2 |
| splice | Pypper:two-level (voting) | 95.39% | 846.2 |
| texture | PFoil:one-level (best rule) | 91.15% | 6610.9 |
| texture | PFoil:one-level (macro distribution voting) | 97.51% | 6610.9 |
| texture | PFoil:one-level (micro distribution voting) | 95.78% | 6610.9 |
| texture | PFoil:one-level (voting) | 97.47% | 6610.9 |
| texture | PFoil:single (best rule) | 94.51% | 6610.9 |
| texture | PFoil:single (macro distribution voting) | 95.02% | 6610.9 |
| texture | PFoil:single (micro distribution voting) | 91.93% | 6610.9 |
| texture | PFoil:single (voting) | 93.25% | 6610.9 |
| texture | PFoil:two-level (best rule) | 97.18% | 6610.9 |
| texture | PFoil:two-level (macro distribution voting) | 97.53% | 6610.9 |
| texture | PFoil:two-level (micro distribution voting) | 96.75% | 6610.9 |
| texture | PFoil:two-level (voting) | 97.51% | 6610.9 |
| texture | PFossil:one-level (best rule) | 92.16% | 4124.7 |
| texture | PFossil:one-level (macro distribution voting) | 96.27% | 4124.7 |
| texture | PFossil:one-level (micro distribution voting) | 92.58% | 4124.7 |
| texture | PFossil:one-level (voting) | 96.02% | 4124.7 |
| texture | PFossil:single (best rule) | 92.22% | 4124.7 |
| texture | PFossil:single (macro distribution voting) | 92.45% | 4124.7 |
| texture | PFossil:single (micro distribution voting) | 90.29% | 4124.7 |
| texture | PFossil:single (voting) | 91.65% | 4124.7 |
| texture | PFossil:two-level (best rule) | 96.24% | 4124.7 |
| texture | PFossil:two-level (macro distribution voting) | 96.45% | 4124.7 |
| texture | PFossil:two-level (micro distribution voting) | 93.64% | 4124.7 |
| texture | PFossil:two-level (voting) | 95.93% | 4124.7 |
| texture | Pypper:one-level (best rule) | 92.31% | 2675.8 |
| texture | Pypper:one-level (macro distribution voting) | 96.33% | 2675.8 |
| texture | Pypper:one-level (micro distribution voting) | 95.07% | 2675.8 |
| texture | Pypper:one-level (voting) | 96.33% | 2675.8 |
| texture | Pypper:single (best rule) | 91.78% | 2675.8 |
| texture | Pypper:single (macro distribution voting) | 92.04% | 2675.8 |
| texture | Pypper:single (micro distribution voting) | 91.49% | 2675.8 |
| texture | Pypper:single (voting) | 91.71% | 2675.8 |
| texture | Pypper:two-level (best rule) | 96.13% | 2675.8 |
| texture | Pypper:two-level (macro distribution voting) | 96.24% | 2675.8 |
| texture | Pypper:two-level (micro distribution voting) | 95.91% | 2675.8 |
| texture | Pypper:two-level (voting) | 96.20% | 2675.8 |
| tic-tac-toe | PFoil:one-level (best rule) | 94.57% | 1187.2 |
| tic-tac-toe | PFoil:one-level (macro distribution voting) | 99.38% | 1187.2 |
| tic-tac-toe | PFoil:one-level (micro distribution voting) | 98.33% | 1187.2 |
| tic-tac-toe | PFoil:one-level (voting) | 99.79% | 1187.2 |
| tic-tac-toe | PFoil:single (best rule) | 99.06% | 1187.2 |
| tic-tac-toe | PFoil:single (macro distribution voting) | 98.85% | 1187.2 |
| tic-tac-toe | PFoil:single (micro distribution voting) | 99.06% | 1187.2 |
| tic-tac-toe | PFoil:single (voting) | 98.43% | 1187.2 |
| tic-tac-toe | PFoil:two-level (best rule) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level (macro distribution voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level (micro distribution voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFoil:two-level (voting) | 99.69% | 1187.2 |
| tic-tac-toe | PFossil:one-level (best rule) | 96.45% | 883.0 |
| tic-tac-toe | PFossil:one-level (macro distribution voting) | 98.85% | 883.0 |
| tic-tac-toe | PFossil:one-level (micro distribution voting) | 85.29% | 883.0 |
| tic-tac-toe | PFossil:one-level (voting) | 96.98% | 883.0 |
| tic-tac-toe | PFossil:single (best rule) | 97.71% | 883.0 |
| tic-tac-toe | PFossil:single (macro distribution voting) | 95.10% | 883.0 |
| tic-tac-toe | PFossil:single (micro distribution voting) | 83.72% | 883.0 |
| tic-tac-toe | PFossil:single (voting) | 85.29% | 883.0 |
| tic-tac-toe | PFossil:two-level (best rule) | 99.17% | 883.0 |
| tic-tac-toe | PFossil:two-level (macro distribution voting) | 99.17% | 883.0 |
| tic-tac-toe | PFossil:two-level (micro distribution voting) | 89.36% | 883.0 |
| tic-tac-toe | PFossil:two-level (voting) | 90.62% | 883.0 |
| tic-tac-toe | Pypper:one-level (best rule) | 96.45% | 788.8 |
| tic-tac-toe | Pypper:one-level (macro distribution voting) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:one-level (micro distribution voting) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:one-level (voting) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:single (best rule) | 96.25% | 788.8 |
| tic-tac-toe | Pypper:single (macro distribution voting) | 96.25% | 788.8 |
| tic-tac-toe | Pypper:single (micro distribution voting) | 96.25% | 788.8 |
| tic-tac-toe | Pypper:single (voting) | 96.25% | 788.8 |
| tic-tac-toe | Pypper:two-level (best rule) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:two-level (macro distribution voting) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:two-level (micro distribution voting) | 98.33% | 788.8 |
| tic-tac-toe | Pypper:two-level (voting) | 98.33% | 788.8 |
| vehicle | PFoil:one-level (best rule) | 67.98% | 3722.8 |
| vehicle | PFoil:one-level (macro distribution voting) | 74.83% | 3722.8 |
| vehicle | PFoil:one-level (micro distribution voting) | 72.12% | 3722.8 |
| vehicle | PFoil:one-level (voting) | 75.19% | 3722.8 |
| vehicle | PFoil:single (best rule) | 71.87% | 3722.8 |
| vehicle | PFoil:single (macro distribution voting) | 71.41% | 3722.8 |
| vehicle | PFoil:single (micro distribution voting) | 71.28% | 3722.8 |
| vehicle | PFoil:single (voting) | 68.92% | 3722.8 |
| vehicle | PFoil:two-level (best rule) | 74.72% | 3722.8 |
| vehicle | PFoil:two-level (macro distribution voting) | 75.66% | 3722.8 |
| vehicle | PFoil:two-level (micro distribution voting) | 74.13% | 3722.8 |
| vehicle | PFoil:two-level (voting) | 74.60% | 3722.8 |
| vehicle | PFossil:one-level (best rule) | 66.91% | 1189.4 |
| vehicle | PFossil:one-level (macro distribution voting) | 74.59% | 1189.4 |
| vehicle | PFossil:one-level (micro distribution voting) | 72.47% | 1189.4 |
| vehicle | PFossil:one-level (voting) | 73.88% | 1189.4 |
| vehicle | PFossil:single (best rule) | 71.64% | 1189.4 |
| vehicle | PFossil:single (macro distribution voting) | 72.82% | 1189.4 |
| vehicle | PFossil:single (micro distribution voting) | 70.93% | 1189.4 |
| vehicle | PFossil:single (voting) | 71.53% | 1189.4 |
| vehicle | PFossil:two-level (best rule) | 72.47% | 1189.4 |
| vehicle | PFossil:two-level (macro distribution voting) | 73.88% | 1189.4 |
| vehicle | PFossil:two-level (micro distribution voting) | 72.35% | 1189.4 |
| vehicle | PFossil:two-level (voting) | 74.82% | 1189.4 |
| vehicle | Pypper:one-level (best rule) | 68.93% | 643.9 |
| vehicle | Pypper:one-level (macro distribution voting) | 72.34% | 643.9 |
| vehicle | Pypper:one-level (micro distribution voting) | 70.93% | 643.9 |
| vehicle | Pypper:one-level (voting) | 72.46% | 643.9 |
| vehicle | Pypper:single (best rule) | 63.84% | 643.9 |
| vehicle | Pypper:single (macro distribution voting) | 64.19% | 643.9 |
| vehicle | Pypper:single (micro distribution voting) | 62.30% | 643.9 |
| vehicle | Pypper:single (voting) | 63.72% | 643.9 |
| vehicle | Pypper:two-level (best rule) | 71.87% | 643.9 |
| vehicle | Pypper:two-level (macro distribution voting) | 71.99% | 643.9 |
| vehicle | Pypper:two-level (micro distribution voting) | 71.87% | 643.9 |
| vehicle | Pypper:two-level (voting) | 72.58% | 643.9 |
| vote | PFoil:one-level (best rule) | 94.73% | 871.4 |
| vote | PFoil:one-level (macro distribution voting) | 96.34% | 871.4 |
| vote | PFoil:one-level (micro distribution voting) | 95.42% | 871.4 |
| vote | PFoil:one-level (voting) | 96.10% | 871.4 |
| vote | PFoil:single (best rule) | 94.71% | 871.4 |
| vote | PFoil:single (macro distribution voting) | 94.71% | 871.4 |
| vote | PFoil:single (micro distribution voting) | 94.03% | 871.4 |
| vote | PFoil:single (voting) | 95.17% | 871.4 |
| vote | PFoil:two-level (best rule) | 95.41% | 871.4 |
| vote | PFoil:two-level (macro distribution voting) | 95.87% | 871.4 |
| vote | PFoil:two-level (micro distribution voting) | 95.18% | 871.4 |
| vote | PFoil:two-level (voting) | 95.87% | 871.4 |
| vote | PFossil:one-level (best rule) | 92.18% | 372.3 |
| vote | PFossil:one-level (macro distribution voting) | 96.10% | 372.3 |
| vote | PFossil:one-level (micro distribution voting) | 95.65% | 372.3 |
| vote | PFossil:one-level (voting) | 96.34% | 372.3 |
| vote | PFossil:single (best rule) | 95.18% | 372.3 |
| vote | PFossil:single (macro distribution voting) | 93.57% | 372.3 |
| vote | PFossil:single (micro distribution voting) | 95.42% | 372.3 |
| vote | PFossil:single (voting) | 94.03% | 372.3 |
| vote | PFossil:two-level (best rule) | 95.87% | 372.3 |
| vote | PFossil:two-level (macro distribution voting) | 95.41% | 372.3 |
| vote | PFossil:two-level (micro distribution voting) | 95.65% | 372.3 |
| vote | PFossil:two-level (voting) | 94.94% | 372.3 |
| vote | Pypper:one-level (best rule) | 95.41% | 175.2 |
| vote | Pypper:one-level (macro distribution voting) | 95.65% | 175.2 |
| vote | Pypper:one-level (micro distribution voting) | 95.65% | 175.2 |
| vote | Pypper:one-level (voting) | 95.65% | 175.2 |
| vote | Pypper:single (best rule) | 95.18% | 175.2 |
| vote | Pypper:single (macro distribution voting) | 94.95% | 175.2 |
| vote | Pypper:single (micro distribution voting) | 95.18% | 175.2 |
| vote | Pypper:single (voting) | 94.95% | 175.2 |
| vote | Pypper:two-level (best rule) | 95.42% | 175.2 |
| vote | Pypper:two-level (macro distribution voting) | 95.42% | 175.2 |
| vote | Pypper:two-level (micro distribution voting) | 95.65% | 175.2 |
| vote | Pypper:two-level (voting) | 95.42% | 175.2 |
| vowel | PFoil:one-level (best rule) | 68.79% | 5132.3 |
| vowel | PFoil:one-level (macro distribution voting) | 89.90% | 5132.3 |
| vowel | PFoil:one-level (micro distribution voting) | 85.96% | 5132.3 |
| vowel | PFoil:one-level (voting) | 90.30% | 5132.3 |
| vowel | PFoil:single (best rule) | 81.82% | 5132.3 |
| vowel | PFoil:single (macro distribution voting) | 82.73% | 5132.3 |
| vowel | PFoil:single (micro distribution voting) | 73.54% | 5132.3 |
| vowel | PFoil:single (voting) | 72.73% | 5132.3 |
| vowel | PFoil:two-level (best rule) | 90.10% | 5132.3 |
| vowel | PFoil:two-level (macro distribution voting) | 90.71% | 5132.3 |
| vowel | PFoil:two-level (micro distribution voting) | 89.70% | 5132.3 |
| vowel | PFoil:two-level (voting) | 90.00% | 5132.3 |
| vowel | PFossil:one-level (best rule) | 72.83% | 4356.5 |
| vowel | PFossil:one-level (macro distribution voting) | 89.60% | 4356.5 |
| vowel | PFossil:one-level (micro distribution voting) | 80.61% | 4356.5 |
| vowel | PFossil:one-level (voting) | 88.99% | 4356.5 |
| vowel | PFossil:single (best rule) | 77.27% | 4356.5 |
| vowel | PFossil:single (macro distribution voting) | 79.80% | 4356.5 |
| vowel | PFossil:single (micro distribution voting) | 74.24% | 4356.5 |
| vowel | PFossil:single (voting) | 78.89% | 4356.5 |
| vowel | PFossil:two-level (best rule) | 87.58% | 4356.5 |
| vowel | PFossil:two-level (macro distribution voting) | 89.70% | 4356.5 |
| vowel | PFossil:two-level (micro distribution voting) | 84.34% | 4356.5 |
| vowel | PFossil:two-level (voting) | 88.18% | 4356.5 |
| vowel | Pypper:one-level (best rule) | 70.40% | 1507.0 |
| vowel | Pypper:one-level (macro distribution voting) | 79.39% | 1507.0 |
| vowel | Pypper:one-level (micro distribution voting) | 72.42% | 1507.0 |
| vowel | Pypper:one-level (voting) | 78.08% | 1507.0 |
| vowel | Pypper:single (best rule) | 63.84% | 1507.0 |
| vowel | Pypper:single (macro distribution voting) | 64.04% | 1507.0 |
| vowel | Pypper:single (micro distribution voting) | 62.53% | 1507.0 |
| vowel | Pypper:single (voting) | 62.53% | 1507.0 |
| vowel | Pypper:two-level (best rule) | 78.89% | 1507.0 |
| vowel | Pypper:two-level (macro distribution voting) | 79.39% | 1507.0 |
| vowel | Pypper:two-level (micro distribution voting) | 76.36% | 1507.0 |
| vowel | Pypper:two-level (voting) | 78.28% | 1507.0 |
| wdbc | PFoil:one-level (best rule) | 91.57% | 605.0 |
| wdbc | PFoil:one-level (macro distribution voting) | 96.48% | 605.0 |
| wdbc | PFoil:one-level (micro distribution voting) | 96.13% | 605.0 |
| wdbc | PFoil:one-level (voting) | 97.18% | 605.0 |
| wdbc | PFoil:single (best rule) | 94.38% | 605.0 |
| wdbc | PFoil:single (macro distribution voting) | 94.38% | 605.0 |
| wdbc | PFoil:single (micro distribution voting) | 93.67% | 605.0 |
| wdbc | PFoil:single (voting) | 93.85% | 605.0 |
| wdbc | PFoil:two-level (best rule) | 96.13% | 605.0 |
| wdbc | PFoil:two-level (macro distribution voting) | 96.31% | 605.0 |
| wdbc | PFoil:two-level (micro distribution voting) | 96.13% | 605.0 |
| wdbc | PFoil:two-level (voting) | 96.13% | 605.0 |
| wdbc | PFossil:one-level (best rule) | 95.43% | 517.1 |
| wdbc | PFossil:one-level (macro distribution voting) | 95.60% | 517.1 |
| wdbc | PFossil:one-level (micro distribution voting) | 94.55% | 517.1 |
| wdbc | PFossil:one-level (voting) | 95.60% | 517.1 |
| wdbc | PFossil:single (best rule) | 94.56% | 517.1 |
| wdbc | PFossil:single (macro distribution voting) | 94.91% | 517.1 |
| wdbc | PFossil:single (micro distribution voting) | 93.85% | 517.1 |
| wdbc | PFossil:single (voting) | 94.56% | 517.1 |
| wdbc | PFossil:two-level (best rule) | 96.13% | 517.1 |
| wdbc | PFossil:two-level (macro distribution voting) | 95.78% | 517.1 |
| wdbc | PFossil:two-level (micro distribution voting) | 94.73% | 517.1 |
| wdbc | PFossil:two-level (voting) | 95.61% | 517.1 |
| wdbc | Pypper:one-level (best rule) | 95.25% | 270.8 |
| wdbc | Pypper:one-level (macro distribution voting) | 95.78% | 270.8 |
| wdbc | Pypper:one-level (micro distribution voting) | 94.73% | 270.8 |
| wdbc | Pypper:one-level (voting) | 95.78% | 270.8 |
| wdbc | Pypper:single (best rule) | 94.91% | 270.8 |
| wdbc | Pypper:single (macro distribution voting) | 95.08% | 270.8 |
| wdbc | Pypper:single (micro distribution voting) | 94.90% | 270.8 |
| wdbc | Pypper:single (voting) | 94.72% | 270.8 |
| wdbc | Pypper:two-level (best rule) | 96.13% | 270.8 |
| wdbc | Pypper:two-level (macro distribution voting) | 95.96% | 270.8 |
| wdbc | Pypper:two-level (micro distribution voting) | 95.08% | 270.8 |
| wdbc | Pypper:two-level (voting) | 95.26% | 270.8 |
| wilt | PFoil:one-level (best rule) | 96.80% | 4390.2 |
| wilt | PFoil:one-level (macro distribution voting) | 97.00% | 4390.2 |
| wilt | PFoil:one-level (micro distribution voting) | 95.16% | 4390.2 |
| wilt | PFoil:one-level (voting) | 95.02% | 4390.2 |
| wilt | PFoil:single (best rule) | 96.55% | 4390.2 |
| wilt | PFoil:single (macro distribution voting) | 96.98% | 4390.2 |
| wilt | PFoil:single (micro distribution voting) | 95.85% | 4390.2 |
| wilt | PFoil:single (voting) | 93.82% | 4390.2 |
| wilt | PFoil:two-level (best rule) | 96.92% | 4390.2 |
| wilt | PFoil:two-level (macro distribution voting) | 97.17% | 4390.2 |
| wilt | PFoil:two-level (micro distribution voting) | 96.28% | 4390.2 |
| wilt | PFoil:two-level (voting) | 95.54% | 4390.2 |
| wilt | PFossil:one-level (best rule) | 96.40% | 51.6 |
| wilt | PFossil:one-level (macro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:one-level (micro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:one-level (voting) | 96.40% | 51.6 |
| wilt | PFossil:single (best rule) | 97.02% | 51.6 |
| wilt | PFossil:single (macro distribution voting) | 97.02% | 51.6 |
| wilt | PFossil:single (micro distribution voting) | 97.02% | 51.6 |
| wilt | PFossil:single (voting) | 97.02% | 51.6 |
| wilt | PFossil:two-level (best rule) | 96.40% | 51.6 |
| wilt | PFossil:two-level (macro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:two-level (micro distribution voting) | 96.40% | 51.6 |
| wilt | PFossil:two-level (voting) | 96.40% | 51.6 |
| wilt | Pypper:one-level (best rule) | 97.17% | 287.0 |
| wilt | Pypper:one-level (macro distribution voting) | 97.44% | 287.0 |
| wilt | Pypper:one-level (micro distribution voting) | 97.33% | 287.0 |
| wilt | Pypper:one-level (voting) | 97.42% | 287.0 |
| wilt | Pypper:single (best rule) | 97.42% | 287.0 |
| wilt | Pypper:single (macro distribution voting) | 97.42% | 287.0 |
| wilt | Pypper:single (micro distribution voting) | 97.42% | 287.0 |
| wilt | Pypper:single (voting) | 97.42% | 287.0 |
| wilt | Pypper:two-level (best rule) | 97.44% | 287.0 |
| wilt | Pypper:two-level (macro distribution voting) | 97.44% | 287.0 |
| wilt | Pypper:two-level (micro distribution voting) | 97.44% | 287.0 |
| wilt | Pypper:two-level (voting) | 97.44% | 287.0 |
| wine | PFoil:one-level (best rule) | 89.22% | 377.7 |
| wine | PFoil:one-level (macro distribution voting) | 97.78% | 377.7 |
| wine | PFoil:one-level (micro distribution voting) | 97.75% | 377.7 |
| wine | PFoil:one-level (voting) | 97.78% | 377.7 |
| wine | PFoil:single (best rule) | 90.98% | 377.7 |
| wine | PFoil:single (macro distribution voting) | 91.54% | 377.7 |
| wine | PFoil:single (micro distribution voting) | 90.98% | 377.7 |
| wine | PFoil:single (voting) | 90.42% | 377.7 |
| wine | PFoil:two-level (best rule) | 96.05% | 377.7 |
| wine | PFoil:two-level (macro distribution voting) | 96.60% | 377.7 |
| wine | PFoil:two-level (micro distribution voting) | 96.08% | 377.7 |
| wine | PFoil:two-level (voting) | 96.60% | 377.7 |
| wine | PFossil:one-level (best rule) | 90.36% | 397.4 |
| wine | PFossil:one-level (macro distribution voting) | 97.19% | 397.4 |
| wine | PFossil:one-level (micro distribution voting) | 97.16% | 397.4 |
| wine | PFossil:one-level (voting) | 97.19% | 397.4 |
| wine | PFossil:single (best rule) | 92.75% | 397.4 |
| wine | PFossil:single (macro distribution voting) | 93.82% | 397.4 |
| wine | PFossil:single (micro distribution voting) | 93.24% | 397.4 |
| wine | PFossil:single (voting) | 93.79% | 397.4 |
| wine | PFossil:two-level (best rule) | 97.19% | 397.4 |
| wine | PFossil:two-level (macro distribution voting) | 97.75% | 397.4 |
| wine | PFossil:two-level (micro distribution voting) | 96.60% | 397.4 |
| wine | PFossil:two-level (voting) | 97.16% | 397.4 |
| wine | Pypper:one-level (best rule) | 90.92% | 245.8 |
| wine | Pypper:one-level (macro distribution voting) | 97.75% | 245.8 |
| wine | Pypper:one-level (micro distribution voting) | 96.60% | 245.8 |
| wine | Pypper:one-level (voting) | 97.75% | 245.8 |
| wine | Pypper:single (best rule) | 92.09% | 245.8 |
| wine | Pypper:single (macro distribution voting) | 90.98% | 245.8 |
| wine | Pypper:single (micro distribution voting) | 90.92% | 245.8 |
| wine | Pypper:single (voting) | 88.69% | 245.8 |
| wine | Pypper:two-level (best rule) | 97.19% | 245.8 |
| wine | Pypper:two-level (macro distribution voting) | 97.19% | 245.8 |
| wine | Pypper:two-level (micro distribution voting) | 96.60% | 245.8 |
| wine | Pypper:two-level (voting) | 95.42% | 245.8 |
| yeast | PFoil:one-level (best rule) | 60.04% | 10179.0 |
| yeast | PFoil:one-level (macro distribution voting) | 59.64% | 10179.0 |
| yeast | PFoil:one-level (micro distribution voting) | 51.22% | 10179.0 |
| yeast | PFoil:one-level (voting) | 58.42% | 10179.0 |
| yeast | PFoil:single (best rule) | 57.41% | 10179.0 |
| yeast | PFoil:single (macro distribution voting) | 56.88% | 10179.0 |
| yeast | PFoil:single (micro distribution voting) | 47.64% | 10179.0 |
| yeast | PFoil:single (voting) | 52.29% | 10179.0 |
| yeast | PFoil:two-level (best rule) | 60.11% | 10179.0 |
| yeast | PFoil:two-level (macro distribution voting) | 59.77% | 10179.0 |
| yeast | PFoil:two-level (micro distribution voting) | 56.74% | 10179.0 |
| yeast | PFoil:two-level (voting) | 56.67% | 10179.0 |
| yeast | PFossil:one-level (best rule) | 52.29% | 2135.2 |
| yeast | PFossil:one-level (macro distribution voting) | 60.91% | 2135.2 |
| yeast | PFossil:one-level (micro distribution voting) | 57.68% | 2135.2 |
| yeast | PFossil:one-level (voting) | 60.51% | 2135.2 |
| yeast | PFossil:single (best rule) | 57.48% | 2135.2 |
| yeast | PFossil:single (macro distribution voting) | 57.54% | 2135.2 |
| yeast | PFossil:single (micro distribution voting) | 55.52% | 2135.2 |
| yeast | PFossil:single (voting) | 54.31% | 2135.2 |
| yeast | PFossil:two-level (best rule) | 60.38% | 2135.2 |
| yeast | PFossil:two-level (macro distribution voting) | 60.37% | 2135.2 |
| yeast | PFossil:two-level (micro distribution voting) | 58.09% | 2135.2 |
| yeast | PFossil:two-level (voting) | 57.68% | 2135.2 |
| yeast | Pypper:one-level (best rule) | 59.50% | 848.9 |
| yeast | Pypper:one-level (macro distribution voting) | 61.86% | 848.9 |
| yeast | Pypper:one-level (micro distribution voting) | 59.23% | 848.9 |
| yeast | Pypper:one-level (voting) | 60.92% | 848.9 |
| yeast | Pypper:single (best rule) | 59.30% | 848.9 |
| yeast | Pypper:single (macro distribution voting) | 59.30% | 848.9 |
| yeast | Pypper:single (micro distribution voting) | 58.62% | 848.9 |
| yeast | Pypper:single (voting) | 58.69% | 848.9 |
| yeast | Pypper:two-level (best rule) | 61.99% | 848.9 |
| yeast | Pypper:two-level (macro distribution voting) | 61.73% | 848.9 |
| yeast | Pypper:two-level (micro distribution voting) | 60.18% | 848.9 |
| yeast | Pypper:two-level (voting) | 60.31% | 848.9 |
| zoo | PFoil:one-level (best rule) | 94.09% | 424.6 |
| zoo | PFoil:one-level (macro distribution voting) | 97.00% | 424.6 |
| zoo | PFoil:one-level (micro distribution voting) | 96.00% | 424.6 |
| zoo | PFoil:one-level (voting) | 96.00% | 424.6 |
| zoo | PFoil:single (best rule) | 96.00% | 424.6 |
| zoo | PFoil:single (macro distribution voting) | 95.00% | 424.6 |
| zoo | PFoil:single (micro distribution voting) | 95.00% | 424.6 |
| zoo | PFoil:single (voting) | 95.00% | 424.6 |
| zoo | PFoil:two-level (best rule) | 98.00% | 424.6 |
| zoo | PFoil:two-level (macro distribution voting) | 98.00% | 424.6 |
| zoo | PFoil:two-level (micro distribution voting) | 98.00% | 424.6 |
| zoo | PFoil:two-level (voting) | 97.00% | 424.6 |
| zoo | PFossil:one-level (best rule) | 94.09% | 420.1 |
| zoo | PFossil:one-level (macro distribution voting) | 98.00% | 420.1 |
| zoo | PFossil:one-level (micro distribution voting) | 96.00% | 420.1 |
| zoo | PFossil:one-level (voting) | 98.00% | 420.1 |
| zoo | PFossil:single (best rule) | 95.00% | 420.1 |
| zoo | PFossil:single (macro distribution voting) | 92.00% | 420.1 |
| zoo | PFossil:single (micro distribution voting) | 94.00% | 420.1 |
| zoo | PFossil:single (voting) | 93.00% | 420.1 |
| zoo | PFossil:two-level (best rule) | 98.00% | 420.1 |
| zoo | PFossil:two-level (macro distribution voting) | 97.00% | 420.1 |
| zoo | PFossil:two-level (micro distribution voting) | 97.00% | 420.1 |
| zoo | PFossil:two-level (voting) | 96.00% | 420.1 |
| zoo | Pypper:one-level (best rule) | 89.09% | 272.5 |
| zoo | Pypper:one-level (macro distribution voting) | 93.09% | 272.5 |
| zoo | Pypper:one-level (micro distribution voting) | 92.09% | 272.5 |
| zoo | Pypper:one-level (voting) | 93.09% | 272.5 |
| zoo | Pypper:single (best rule) | 90.09% | 272.5 |
| zoo | Pypper:single (macro distribution voting) | 90.09% | 272.5 |
| zoo | Pypper:single (micro distribution voting) | 91.09% | 272.5 |
| zoo | Pypper:single (voting) | 90.09% | 272.5 |
| zoo | Pypper:two-level (best rule) | 92.09% | 272.5 |
| zoo | Pypper:two-level (macro distribution voting) | 93.09% | 272.5 |
| zoo | Pypper:two-level (micro distribution voting) | 91.09% | 272.5 |
| zoo | Pypper:two-level (voting) | 91.09% | 272.5 |

