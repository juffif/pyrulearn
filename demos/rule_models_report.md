# Rule models: rule sets, cascades, decision lists

## Setup

**Full run.** Quick sanity check instead: `python demos/rule_models.py`.

A rule learner's covering loop can build different kinds of models from
the same search. This demo compares them, for CN2, PFossil and Pypper as
base learners:

- **rule set** -- one-vs-rest: a rule set per class, each learned against
  all other classes; a conflict between covering rules of different
  classes resolved by **best rule** (the covering rule with the best
  Laplace estimate, i.e. the rules sorted by it, an ordered rule set),
  **votes** (one vote per covering rule) or **distributions** (their class
  counts summed, CN2's own) -- three ways of predicting with the same
  rules.
- **cascade** -- ordered one-vs-rest: the classes one after the other,
  each against those not yet handled, from the **rarest** class (Ripper's
  order) or the **most frequent** one; the last class is the default.
- **decision list** -- one covering loop over all classes, first match
  wins, the default the majority of what no rule covers. Each rule the
  best of **any class** (its head chosen by the heuristic, CN2's ordered
  mode), the best of one search for **each class**, or explaining a random
  uncovered example, the **seed**: generalizing it and predicting its class
  (AQ's seed covering), or with the seed only picking the class (**seed's
  class**).

On two-class data (the second part), a cascade is a concept for one class
with the other as the default; the configurations there are a concept for
the **minority** class and one for the **majority** class, the **rule set**
of both classes and the **decision list** of rules of either class.

Multi-class datasets, with their numbers of classes: `balance-scale` (3), `cmc` (3), `dna` (3), `hayes-roth` (3), `iris` (3), `splice` (3), `wine` (3), `car` (4), `lymph` (4), `vehicle` (4), `anneal` (5), `glass` (6), `solar-flare` (6), `segment` (7), `zoo` (7), `ecoli` (8), `led24` (10), `yeast` (10), `optdigits` (10), `texture` (11), `vowel` (11), `soybean` (19), `primary-tumor` (21), `audiology` (24). Two-class datasets: `SPECT`, `Titanic`, `banknote-authentication`, `blood-transfusion-service-center`, `breast-cancer`, `breast-w`, `churn`, `climate-model-simulation-crashes`, `colic`, `compas-two-years`, `credit-approval`, `credit-g`, `cylinder-bands`, `diabetes`, `dresses-sales`, `heart-c`, `heart-h`, `heart-statlog`, `heloc`, `hepatitis`, `ilpd`, `ionosphere`, `kc1`, `kc2`, `kr-vs-kp`, `mofn-3-7-10`, `molecular-biology_promoters`, `monks-problems-1`, `monks-problems-2`, `monks-problems-3`, `mushroom`, `ozone-level-8hr`, `pc1`, `pc3`, `pc4`, `phoneme`, `qsar-biodeg`, `sick`, `sonar`, `spambase`, `tic-tac-toe`, `vote`, `wdbc`, `wilt`. Protocol:
10-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals=8)`). Each fit is capped at
300s; a time-out fails the whole cross-validation of that
learner on that dataset. Training time is measured inside the worker process.

## Multi-class datasets

Sorted by mean rank (accuracy rank across datasets, failures last). Rules and conditions: all of a model's rules together; the three rule-set variants share their rules and training time.

### CN2

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| decision list (each class) | 80.61% | 2.65 | 34.6 | 136.6 | 3.47 | 0 |
| decision list (seed) | 79.62% | 3.71 | 33.8 | 143.1 | 0.57 | 0 |
| decision list (seed's class) | 79.39% | 4.52 | 38.7 | 144.4 | 0.64 | 0 |
| decision list (any class) | 78.75% | 4.65 | 29.5 | 115.2 | 1.11 | 0 |
| rule set (best rule) | 76.23% | 5.27 | 43.3 | 202.7 | 0.51 | 0 |
| cascade (rarest first) | 75.81% | 5.58 | 29.2 | 122.0 | 0.33 | 0 |
| cascade (most frequent first) | 75.52% | 5.67 | 28.4 | 109.6 | 0.33 | 0 |
| rule set (distributions) | 75.15% | 6.40 | 43.3 | 202.7 | 0.51 | 0 |
| rule set (votes) | 75.15% | 6.56 | 43.3 | 202.7 | 0.51 | 0 |

### PFossil

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 81.37% | 3.62 | 32.6 | 140.5 | 0.09 | 0 |
| decision list (any class) | 80.46% | 4.02 | 21.1 | 58.1 | 0.10 | 0 |
| decision list (each class) | 80.17% | 4.38 | 21.0 | 78.8 | 0.22 | 0 |
| cascade (rarest first) | 79.85% | 4.62 | 24.5 | 94.5 | 0.06 | 0 |
| rule set (distributions) | 79.97% | 4.98 | 32.6 | 140.5 | 0.09 | 0 |
| rule set (votes) | 80.08% | 5.04 | 32.6 | 140.5 | 0.09 | 0 |
| cascade (most frequent first) | 77.87% | 5.60 | 25.0 | 87.6 | 0.06 | 0 |
| decision list (seed's class) | 79.02% | 5.92 | 20.9 | 65.0 | 0.06 | 0 |
| decision list (seed) | 77.38% | 6.81 | 17.4 | 59.5 | 0.05 | 0 |

### Pypper

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| cascade (rarest first) | 79.80% | 3.81 | 13.9 | 46.9 | 0.17 | 0 |
| cascade (most frequent first) | 78.43% | 4.50 | 14.9 | 45.4 | 0.17 | 0 |
| decision list (seed) | 78.80% | 4.60 | 26.4 | 98.4 | 0.13 | 0 |
| rule set (best rule) | 78.40% | 4.65 | 18.8 | 71.3 | 0.26 | 0 |
| decision list (seed's class) | 78.96% | 4.79 | 27.5 | 84.1 | 0.13 | 0 |
| decision list (each class) | 78.18% | 5.06 | 14.0 | 39.4 | 0.31 | 0 |
| decision list (any class) | 77.78% | 5.15 | 15.7 | 49.0 | 0.12 | 0 |
| rule set (votes) | 77.50% | 6.21 | 18.8 | 71.3 | 0.26 | 0 |
| rule set (distributions) | 77.58% | 6.23 | 18.8 | 71.3 | 0.26 | 0 |

### All base learners together

Each kind of model's mean rank, averaged over the 3 base learners (ranked within each base learner on each dataset, as in the tables above), and its accuracy averaged over them.

| model | accuracy | mean rank |
|---|--:|--:|
| decision list (each class) | 79.65% | 4.03 |
| rule set (best rule) | 78.67% | 4.51 |
| decision list (any class) | 79.00% | 4.60 |
| cascade (rarest first) | 78.49% | 4.67 |
| decision list (seed) | 78.60% | 5.04 |
| decision list (seed's class) | 79.12% | 5.08 |
| cascade (most frequent first) | 77.27% | 5.26 |
| rule set (distributions) | 77.57% | 5.87 |
| rule set (votes) | 77.57% | 5.94 |

Friedman test over the 72 (dataset, base learner) pairs: p = 0.00017 (9 kinds of model). Nemenyi critical difference at 5%: 1.42 -- two kinds whose mean ranks differ by less are not significantly different; the diagram joins them by a bar. The pairs of one dataset aren't independent of each other, so the test is on the optimistic side.

![critical-difference diagram, multiclass](rule_models_plots/rule_models_cd_multiclass.png)

## Two-class datasets

A concept for one class, the other the default; the rule set of both classes; the decision list of rules of either class.

### CN2

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 82.56% | 2.77 | 24.7 | 71.6 | 0.15 | 0 |
| rule set (distributions) | 82.22% | 3.08 | 24.7 | 71.6 | 0.15 | 0 |
| rule set (votes) | 82.09% | 3.15 | 24.7 | 71.6 | 0.15 | 0 |
| decision list (any class) | 81.36% | 3.51 | 16.4 | 43.2 | 0.21 | 0 |
| concept (minority class) | 81.30% | 3.77 | 11.9 | 34.4 | 0.06 | 0 |
| concept (majority class) | 74.07% | 4.72 | 12.8 | 37.3 | 0.09 | 0 |

### PFossil

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 83.33% | 2.59 | 9.1 | 34.3 | 0.02 | 0 |
| rule set (votes) | 82.65% | 3.16 | 9.1 | 34.3 | 0.02 | 0 |
| rule set (distributions) | 82.47% | 3.43 | 9.1 | 34.3 | 0.02 | 0 |
| decision list (any class) | 82.00% | 3.77 | 3.9 | 15.9 | 0.02 | 0 |
| concept (minority class) | 81.58% | 3.78 | 4.3 | 17.5 | 0.01 | 0 |
| concept (majority class) | 75.52% | 4.26 | 4.8 | 16.8 | 0.01 | 0 |

### Pypper

| model | accuracy | mean rank | rules | conditions | train (s) | failures |
|---|--:|--:|--:|--:|--:|--:|
| rule set (best rule) | 85.26% | 2.91 | 8.7 | 27.2 | 0.16 | 0 |
| rule set (distributions) | 85.13% | 3.16 | 8.7 | 27.2 | 0.16 | 0 |
| rule set (votes) | 85.08% | 3.18 | 8.7 | 27.2 | 0.16 | 0 |
| decision list (any class) | 85.13% | 3.51 | 6.4 | 25.3 | 0.06 | 0 |
| concept (minority class) | 84.98% | 3.81 | 4.0 | 13.3 | 0.07 | 0 |
| concept (majority class) | 82.48% | 4.43 | 4.7 | 13.9 | 0.09 | 0 |

### All base learners together

Each kind of model's mean rank, averaged over the 3 base learners (ranked within each base learner on each dataset, as in the tables above), and its accuracy averaged over them.

| model | accuracy | mean rank |
|---|--:|--:|
| rule set (best rule) | 83.72% | 2.76 |
| rule set (votes) | 83.27% | 3.16 |
| rule set (distributions) | 83.27% | 3.22 |
| decision list (any class) | 82.83% | 3.60 |
| concept (minority class) | 82.62% | 3.79 |
| concept (majority class) | 77.36% | 4.47 |

Friedman test over the 132 (dataset, base learner) pairs: p = 1.3e-14 (6 kinds of model). Nemenyi critical difference at 5%: 0.66 -- two kinds whose mean ranks differ by less are not significantly different; the diagram joins them by a bar. The pairs of one dataset aren't independent of each other, so the test is on the optimistic side.

![critical-difference diagram, twoclass](rule_models_plots/rule_models_cd_twoclass.png)

## Per dataset (mean across folds)

| dataset | learner | accuracy | n_rules | train_time |
|---|---|---|---|---|
| SPECT | CN2:concept (majority class) | 71.51% | 6.4 | 0.012 |
| SPECT | CN2:concept (minority class) | 71.95% | 1.9 | 0.006 |
| SPECT | CN2:decision list (any class) | 75.67% | 6.3 | 0.024 |
| SPECT | CN2:rule set (best rule) | 80.90% | 8.3 | 0.018 |
| SPECT | CN2:rule set (distributions) | 80.90% | 8.3 | 0.018 |
| SPECT | CN2:rule set (votes) | 78.29% | 8.3 | 0.018 |
| SPECT | PFossil:concept (majority class) | 70.81% | 3.5 | 0.004 |
| SPECT | PFossil:concept (minority class) | 73.76% | 6.2 | 0.010 |
| SPECT | PFossil:decision list (any class) | 74.50% | 4.5 | 0.014 |
| SPECT | PFossil:rule set (best rule) | 78.68% | 9.7 | 0.014 |
| SPECT | PFossil:rule set (distributions) | 78.68% | 9.7 | 0.014 |
| SPECT | PFossil:rule set (votes) | 77.55% | 9.7 | 0.014 |
| SPECT | Pypper:concept (majority class) | 70.81% | 3.1 | 0.040 |
| SPECT | Pypper:concept (minority class) | 79.00% | 0.9 | 0.023 |
| SPECT | Pypper:decision list (any class) | 82.38% | 1.0 | 0.021 |
| SPECT | Pypper:rule set (best rule) | 79.02% | 4.0 | 0.052 |
| SPECT | Pypper:rule set (distributions) | 79.02% | 4.0 | 0.052 |
| SPECT | Pypper:rule set (votes) | 79.02% | 4.0 | 0.052 |
| Titanic | CN2:concept (majority class) | 56.69% | 9.3 | 0.064 |
| Titanic | CN2:concept (minority class) | 73.41% | 17.8 | 0.096 |
| Titanic | CN2:decision list (any class) | 65.01% | 9.1 | 0.127 |
| Titanic | CN2:rule set (best rule) | 75.02% | 27.1 | 0.163 |
| Titanic | CN2:rule set (distributions) | 73.41% | 27.1 | 0.163 |
| Titanic | CN2:rule set (votes) | 74.64% | 27.1 | 0.163 |
| Titanic | PFossil:concept (majority class) | 76.93% | 3.1 | 0.006 |
| Titanic | PFossil:concept (minority class) | 78.46% | 2.0 | 0.005 |
| Titanic | PFossil:decision list (any class) | 78.46% | 2.0 | 0.009 |
| Titanic | PFossil:rule set (best rule) | 77.77% | 5.1 | 0.012 |
| Titanic | PFossil:rule set (distributions) | 76.78% | 5.1 | 0.012 |
| Titanic | PFossil:rule set (votes) | 78.15% | 5.1 | 0.012 |
| Titanic | Pypper:concept (majority class) | 77.54% | 4.6 | 0.071 |
| Titanic | Pypper:concept (minority class) | 79.22% | 2.6 | 0.048 |
| Titanic | Pypper:decision list (any class) | 79.22% | 7.6 | 0.038 |
| Titanic | Pypper:rule set (best rule) | 79.22% | 7.2 | 0.123 |
| Titanic | Pypper:rule set (distributions) | 78.84% | 7.2 | 0.123 |
| Titanic | Pypper:rule set (votes) | 78.84% | 7.2 | 0.123 |
| anneal | CN2:cascade (most frequent first) | 99.33% | 13.3 | 0.022 |
| anneal | CN2:cascade (rarest first) | 85.85% | 16.9 | 0.074 |
| anneal | CN2:decision list (any class) | 98.22% | 12.3 | 0.059 |
| anneal | CN2:decision list (each class) | 98.89% | 14.8 | 0.107 |
| anneal | CN2:decision list (seed's class) | 98.55% | 15.3 | 0.037 |
| anneal | CN2:decision list (seed) | 98.99% | 12.5 | 0.047 |
| anneal | CN2:rule set (best rule) | 98.33% | 27.2 | 0.097 |
| anneal | CN2:rule set (distributions) | 98.33% | 27.2 | 0.097 |
| anneal | CN2:rule set (votes) | 97.44% | 27.2 | 0.097 |
| anneal | PFossil:cascade (most frequent first) | 99.67% | 11.8 | 0.014 |
| anneal | PFossil:cascade (rarest first) | 93.99% | 10.9 | 0.016 |
| anneal | PFossil:decision list (any class) | 99.33% | 10.7 | 0.021 |
| anneal | PFossil:decision list (each class) | 93.65% | 10.0 | 0.033 |
| anneal | PFossil:decision list (seed's class) | 97.33% | 9.0 | 0.012 |
| anneal | PFossil:decision list (seed) | 94.89% | 8.0 | 0.011 |
| anneal | PFossil:rule set (best rule) | 99.22% | 17.8 | 0.023 |
| anneal | PFossil:rule set (distributions) | 98.88% | 17.8 | 0.023 |
| anneal | PFossil:rule set (votes) | 99.66% | 17.8 | 0.023 |
| anneal | Pypper:cascade (most frequent first) | 98.55% | 9.2 | 0.046 |
| anneal | Pypper:cascade (rarest first) | 94.20% | 8.2 | 0.061 |
| anneal | Pypper:decision list (any class) | 98.77% | 8.9 | 0.028 |
| anneal | Pypper:decision list (each class) | 99.00% | 10.1 | 0.066 |
| anneal | Pypper:decision list (seed's class) | 97.33% | 11.4 | 0.028 |
| anneal | Pypper:decision list (seed) | 97.77% | 10.7 | 0.027 |
| anneal | Pypper:rule set (best rule) | 96.10% | 13.7 | 0.094 |
| anneal | Pypper:rule set (distributions) | 95.77% | 13.7 | 0.094 |
| anneal | Pypper:rule set (votes) | 96.10% | 13.7 | 0.094 |
| audiology | CN2:cascade (most frequent first) | 67.73% | 18.4 | 0.236 |
| audiology | CN2:cascade (rarest first) | 83.68% | 24.2 | 0.141 |
| audiology | CN2:decision list (any class) | 73.89% | 10.4 | 0.618 |
| audiology | CN2:decision list (each class) | 78.74% | 21.8 | 1.109 |
| audiology | CN2:decision list (seed's class) | 71.25% | 21.4 | 0.219 |
| audiology | CN2:decision list (seed) | 73.48% | 21.5 | 0.186 |
| audiology | CN2:rule set (best rule) | 75.63% | 28.8 | 0.373 |
| audiology | CN2:rule set (distributions) | 72.53% | 28.8 | 0.373 |
| audiology | CN2:rule set (votes) | 71.64% | 28.8 | 0.373 |
| audiology | PFossil:cascade (most frequent first) | 74.37% | 36.3 | 0.033 |
| audiology | PFossil:cascade (rarest first) | 84.05% | 34.0 | 0.035 |
| audiology | PFossil:decision list (any class) | 79.68% | 28.0 | 0.067 |
| audiology | PFossil:decision list (each class) | 84.09% | 23.6 | 0.248 |
| audiology | PFossil:decision list (seed's class) | 74.35% | 31.2 | 0.026 |
| audiology | PFossil:decision list (seed) | 73.46% | 28.6 | 0.025 |
| audiology | PFossil:rule set (best rule) | 77.89% | 43.3 | 0.048 |
| audiology | PFossil:rule set (distributions) | 74.78% | 43.3 | 0.048 |
| audiology | PFossil:rule set (votes) | 75.26% | 43.3 | 0.048 |
| audiology | Pypper:cascade (most frequent first) | 67.27% | 10.1 | 0.088 |
| audiology | Pypper:cascade (rarest first) | 75.28% | 10.5 | 0.083 |
| audiology | Pypper:decision list (any class) | 65.59% | 7.7 | 0.038 |
| audiology | Pypper:decision list (each class) | 70.40% | 11.9 | 0.272 |
| audiology | Pypper:decision list (seed's class) | 65.47% | 14.8 | 0.044 |
| audiology | Pypper:decision list (seed) | 70.43% | 13.9 | 0.044 |
| audiology | Pypper:rule set (best rule) | 64.64% | 11.2 | 0.096 |
| audiology | Pypper:rule set (distributions) | 63.75% | 11.2 | 0.096 |
| audiology | Pypper:rule set (votes) | 62.02% | 11.2 | 0.096 |
| balance-scale | CN2:cascade (most frequent first) | 43.68% | 19.2 | 0.056 |
| balance-scale | CN2:cascade (rarest first) | 77.59% | 15.5 | 0.044 |
| balance-scale | CN2:decision list (any class) | 73.76% | 17.5 | 0.103 |
| balance-scale | CN2:decision list (each class) | 77.74% | 18.8 | 0.105 |
| balance-scale | CN2:decision list (seed's class) | 77.59% | 28.4 | 0.120 |
| balance-scale | CN2:decision list (seed) | 79.35% | 25.8 | 0.120 |
| balance-scale | CN2:rule set (best rule) | 76.78% | 37.1 | 0.110 |
| balance-scale | CN2:rule set (distributions) | 76.14% | 37.1 | 0.110 |
| balance-scale | CN2:rule set (votes) | 76.62% | 37.1 | 0.110 |
| balance-scale | PFossil:cascade (most frequent first) | 44.16% | 12.4 | 0.017 |
| balance-scale | PFossil:cascade (rarest first) | 78.88% | 13.3 | 0.018 |
| balance-scale | PFossil:decision list (any class) | 78.74% | 13.6 | 0.028 |
| balance-scale | PFossil:decision list (each class) | 79.68% | 16.5 | 0.043 |
| balance-scale | PFossil:decision list (seed's class) | 81.12% | 17.3 | 0.020 |
| balance-scale | PFossil:decision list (seed) | 76.81% | 11.7 | 0.015 |
| balance-scale | PFossil:rule set (best rule) | 81.60% | 26.0 | 0.034 |
| balance-scale | PFossil:rule set (distributions) | 80.47% | 26.0 | 0.034 |
| balance-scale | PFossil:rule set (votes) | 79.67% | 26.0 | 0.034 |
| balance-scale | Pypper:cascade (most frequent first) | 69.77% | 10.3 | 0.085 |
| balance-scale | Pypper:cascade (rarest first) | 79.18% | 7.6 | 0.056 |
| balance-scale | Pypper:decision list (any class) | 80.15% | 15.5 | 0.057 |
| balance-scale | Pypper:decision list (each class) | 81.59% | 14.4 | 0.096 |
| balance-scale | Pypper:decision list (seed's class) | 83.52% | 20.4 | 0.088 |
| balance-scale | Pypper:decision list (seed) | 79.19% | 19.9 | 0.062 |
| balance-scale | Pypper:rule set (best rule) | 80.95% | 14.2 | 0.104 |
| balance-scale | Pypper:rule set (distributions) | 80.62% | 14.2 | 0.104 |
| balance-scale | Pypper:rule set (votes) | 80.47% | 14.2 | 0.104 |
| banknote-authentication | CN2:concept (majority class) | 98.25% | 8.7 | 0.054 |
| banknote-authentication | CN2:concept (minority class) | 97.60% | 8.2 | 0.028 |
| banknote-authentication | CN2:decision list (any class) | 98.76% | 12.8 | 0.087 |
| banknote-authentication | CN2:rule set (best rule) | 97.52% | 16.9 | 0.080 |
| banknote-authentication | CN2:rule set (distributions) | 97.52% | 16.9 | 0.080 |
| banknote-authentication | CN2:rule set (votes) | 97.60% | 16.9 | 0.080 |
| banknote-authentication | PFossil:concept (majority class) | 95.99% | 6.8 | 0.011 |
| banknote-authentication | PFossil:concept (minority class) | 96.57% | 7.2 | 0.011 |
| banknote-authentication | PFossil:decision list (any class) | 95.99% | 6.8 | 0.017 |
| banknote-authentication | PFossil:rule set (best rule) | 99.20% | 14.0 | 0.024 |
| banknote-authentication | PFossil:rule set (distributions) | 94.32% | 14.0 | 0.024 |
| banknote-authentication | PFossil:rule set (votes) | 97.89% | 14.0 | 0.024 |
| banknote-authentication | Pypper:concept (majority class) | 98.91% | 5.0 | 0.038 |
| banknote-authentication | Pypper:concept (minority class) | 98.54% | 4.5 | 0.032 |
| banknote-authentication | Pypper:decision list (any class) | 98.54% | 5.6 | 0.026 |
| banknote-authentication | Pypper:rule set (best rule) | 98.54% | 9.5 | 0.073 |
| banknote-authentication | Pypper:rule set (distributions) | 98.54% | 9.5 | 0.073 |
| banknote-authentication | Pypper:rule set (votes) | 98.61% | 9.5 | 0.073 |
| blood-transfusion-service-center | CN2:concept (majority class) | 67.65% | 12.3 | 0.078 |
| blood-transfusion-service-center | CN2:concept (minority class) | 73.40% | 2.5 | 0.015 |
| blood-transfusion-service-center | CN2:decision list (any class) | 76.74% | 12.4 | 0.157 |
| blood-transfusion-service-center | CN2:rule set (best rule) | 73.81% | 14.8 | 0.092 |
| blood-transfusion-service-center | CN2:rule set (distributions) | 73.81% | 14.8 | 0.092 |
| blood-transfusion-service-center | CN2:rule set (votes) | 73.81% | 14.8 | 0.092 |
| blood-transfusion-service-center | PFossil:concept (majority class) | 58.56% | 0.9 | 0.002 |
| blood-transfusion-service-center | PFossil:concept (minority class) | 76.86% | 1.0 | 0.003 |
| blood-transfusion-service-center | PFossil:decision list (any class) | 77.80% | 1.8 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (best rule) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (distributions) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | PFossil:rule set (votes) | 76.86% | 1.9 | 0.005 |
| blood-transfusion-service-center | Pypper:concept (majority class) | 79.14% | 3.0 | 0.042 |
| blood-transfusion-service-center | Pypper:concept (minority class) | 78.33% | 2.0 | 0.024 |
| blood-transfusion-service-center | Pypper:decision list (any class) | 78.20% | 5.9 | 0.026 |
| blood-transfusion-service-center | Pypper:rule set (best rule) | 78.60% | 5.0 | 0.068 |
| blood-transfusion-service-center | Pypper:rule set (distributions) | 78.47% | 5.0 | 0.068 |
| blood-transfusion-service-center | Pypper:rule set (votes) | 78.47% | 5.0 | 0.068 |
| breast-cancer | CN2:concept (majority class) | 38.10% | 1.1 | 0.003 |
| breast-cancer | CN2:concept (minority class) | 70.30% | 0.5 | 0.002 |
| breast-cancer | CN2:decision list (any class) | 70.30% | 1.1 | 0.005 |
| breast-cancer | CN2:rule set (best rule) | 70.30% | 1.6 | 0.005 |
| breast-cancer | CN2:rule set (distributions) | 70.30% | 1.6 | 0.005 |
| breast-cancer | CN2:rule set (votes) | 70.30% | 1.6 | 0.005 |
| breast-cancer | PFossil:concept (majority class) | 67.13% | 6.2 | 0.009 |
| breast-cancer | PFossil:concept (minority class) | 69.24% | 1.2 | 0.004 |
| breast-cancer | PFossil:decision list (any class) | 69.24% | 1.4 | 0.005 |
| breast-cancer | PFossil:rule set (best rule) | 69.26% | 7.4 | 0.013 |
| breast-cancer | PFossil:rule set (distributions) | 68.56% | 7.4 | 0.013 |
| breast-cancer | PFossil:rule set (votes) | 68.92% | 7.4 | 0.013 |
| breast-cancer | Pypper:concept (majority class) | 70.58% | 1.5 | 0.024 |
| breast-cancer | Pypper:concept (minority class) | 74.84% | 1.0 | 0.011 |
| breast-cancer | Pypper:decision list (any class) | 75.20% | 1.9 | 0.009 |
| breast-cancer | Pypper:rule set (best rule) | 74.84% | 2.5 | 0.035 |
| breast-cancer | Pypper:rule set (distributions) | 75.20% | 2.5 | 0.035 |
| breast-cancer | Pypper:rule set (votes) | 75.20% | 2.5 | 0.035 |
| breast-w | CN2:concept (majority class) | 95.14% | 5.8 | 0.018 |
| breast-w | CN2:concept (minority class) | 94.71% | 18.1 | 0.042 |
| breast-w | CN2:decision list (any class) | 94.99% | 12.5 | 0.037 |
| breast-w | CN2:rule set (best rule) | 94.57% | 23.9 | 0.058 |
| breast-w | CN2:rule set (distributions) | 94.57% | 23.9 | 0.058 |
| breast-w | CN2:rule set (votes) | 94.85% | 23.9 | 0.058 |
| breast-w | PFossil:concept (majority class) | 94.41% | 9.5 | 0.015 |
| breast-w | PFossil:concept (minority class) | 94.56% | 6.1 | 0.008 |
| breast-w | PFossil:decision list (any class) | 94.85% | 5.6 | 0.020 |
| breast-w | PFossil:rule set (best rule) | 94.13% | 15.6 | 0.024 |
| breast-w | PFossil:rule set (distributions) | 94.27% | 15.6 | 0.024 |
| breast-w | PFossil:rule set (votes) | 94.99% | 15.6 | 0.024 |
| breast-w | Pypper:concept (majority class) | 95.13% | 2.6 | 0.023 |
| breast-w | Pypper:concept (minority class) | 95.42% | 3.6 | 0.028 |
| breast-w | Pypper:decision list (any class) | 95.14% | 3.9 | 0.018 |
| breast-w | Pypper:rule set (best rule) | 94.57% | 6.2 | 0.050 |
| breast-w | Pypper:rule set (distributions) | 94.57% | 6.2 | 0.050 |
| breast-w | Pypper:rule set (votes) | 95.00% | 6.2 | 0.050 |
| car | CN2:cascade (most frequent first) | 97.34% | 22.3 | 0.056 |
| car | CN2:cascade (rarest first) | 92.07% | 40.3 | 0.176 |
| car | CN2:decision list (any class) | 84.78% | 9.7 | 0.063 |
| car | CN2:decision list (each class) | 97.11% | 35.4 | 0.348 |
| car | CN2:decision list (seed's class) | 95.95% | 36.9 | 0.132 |
| car | CN2:decision list (seed) | 95.49% | 32.7 | 0.138 |
| car | CN2:rule set (best rule) | 94.10% | 64.8 | 0.272 |
| car | CN2:rule set (distributions) | 93.98% | 64.8 | 0.272 |
| car | CN2:rule set (votes) | 93.81% | 64.8 | 0.272 |
| car | PFossil:cascade (most frequent first) | 95.95% | 19.8 | 0.027 |
| car | PFossil:cascade (rarest first) | 82.87% | 5.4 | 0.015 |
| car | PFossil:decision list (any class) | 94.10% | 14.4 | 0.041 |
| car | PFossil:decision list (each class) | 81.31% | 4.7 | 0.020 |
| car | PFossil:decision list (seed's class) | 79.57% | 2.5 | 0.148 |
| car | PFossil:decision list (seed) | 78.82% | 2.1 | 0.158 |
| car | PFossil:rule set (best rule) | 93.87% | 16.2 | 0.033 |
| car | PFossil:rule set (distributions) | 84.90% | 16.2 | 0.033 |
| car | PFossil:rule set (votes) | 90.51% | 16.2 | 0.033 |
| car | Pypper:cascade (most frequent first) | 97.05% | 21.7 | 0.104 |
| car | Pypper:cascade (rarest first) | 93.63% | 17.1 | 0.181 |
| car | Pypper:decision list (any class) | 94.09% | 16.8 | 0.070 |
| car | Pypper:decision list (each class) | 94.74% | 17.1 | 0.151 |
| car | Pypper:decision list (seed's class) | 94.85% | 28.8 | 0.085 |
| car | Pypper:decision list (seed) | 92.94% | 26.2 | 0.077 |
| car | Pypper:rule set (best rule) | 91.32% | 24.1 | 0.228 |
| car | Pypper:rule set (distributions) | 89.99% | 24.1 | 0.228 |
| car | Pypper:rule set (votes) | 90.39% | 24.1 | 0.228 |
| churn | CN2:concept (majority class) | 47.70% | 22.1 | 0.170 |
| churn | CN2:concept (minority class) | 89.92% | 19.3 | 0.087 |
| churn | CN2:decision list (any class) | 87.28% | 36.0 | 0.541 |
| churn | CN2:rule set (best rule) | 89.72% | 41.4 | 0.260 |
| churn | CN2:rule set (distributions) | 89.44% | 41.4 | 0.260 |
| churn | CN2:rule set (votes) | 89.38% | 41.4 | 0.260 |
| churn | PFossil:concept (majority class) | 91.46% | 6.7 | 0.033 |
| churn | PFossil:concept (minority class) | 90.36% | 2.5 | 0.012 |
| churn | PFossil:decision list (any class) | 92.80% | 8.1 | 0.079 |
| churn | PFossil:rule set (best rule) | 90.00% | 9.2 | 0.052 |
| churn | PFossil:rule set (distributions) | 89.18% | 9.2 | 0.052 |
| churn | PFossil:rule set (votes) | 89.00% | 9.2 | 0.052 |
| churn | Pypper:concept (majority class) | 93.08% | 7.9 | 0.266 |
| churn | Pypper:concept (minority class) | 93.62% | 9.6 | 0.167 |
| churn | Pypper:decision list (any class) | 93.48% | 12.5 | 0.138 |
| churn | Pypper:rule set (best rule) | 93.42% | 17.5 | 0.439 |
| churn | Pypper:rule set (distributions) | 93.36% | 17.5 | 0.439 |
| churn | Pypper:rule set (votes) | 93.28% | 17.5 | 0.439 |
| climate-model-simulation-crashes | CN2:concept (majority class) | 90.37% | 7.2 | 0.018 |
| climate-model-simulation-crashes | CN2:concept (minority class) | 90.56% | 6.0 | 0.011 |
| climate-model-simulation-crashes | CN2:decision list (any class) | 92.22% | 9.3 | 0.038 |
| climate-model-simulation-crashes | CN2:rule set (best rule) | 92.04% | 13.2 | 0.030 |
| climate-model-simulation-crashes | CN2:rule set (distributions) | 92.04% | 13.2 | 0.030 |
| climate-model-simulation-crashes | CN2:rule set (votes) | 92.04% | 13.2 | 0.030 |
| climate-model-simulation-crashes | PFossil:concept (majority class) | 92.22% | 6.4 | 0.009 |
| climate-model-simulation-crashes | PFossil:concept (minority class) | 93.52% | 2.5 | 0.005 |
| climate-model-simulation-crashes | PFossil:decision list (any class) | 93.15% | 2.8 | 0.011 |
| climate-model-simulation-crashes | PFossil:rule set (best rule) | 92.78% | 8.9 | 0.015 |
| climate-model-simulation-crashes | PFossil:rule set (distributions) | 92.78% | 8.9 | 0.015 |
| climate-model-simulation-crashes | PFossil:rule set (votes) | 92.78% | 8.9 | 0.015 |
| climate-model-simulation-crashes | Pypper:concept (majority class) | 93.15% | 3.7 | 0.023 |
| climate-model-simulation-crashes | Pypper:concept (minority class) | 93.89% | 1.9 | 0.018 |
| climate-model-simulation-crashes | Pypper:decision list (any class) | 94.07% | 2.1 | 0.011 |
| climate-model-simulation-crashes | Pypper:rule set (best rule) | 94.81% | 5.6 | 0.041 |
| climate-model-simulation-crashes | Pypper:rule set (distributions) | 94.81% | 5.6 | 0.041 |
| climate-model-simulation-crashes | Pypper:rule set (votes) | 94.81% | 5.6 | 0.041 |
| cmc | CN2:cascade (most frequent first) | 42.43% | 22.6 | 0.119 |
| cmc | CN2:cascade (rarest first) | 41.28% | 5.5 | 0.029 |
| cmc | CN2:decision list (any class) | 45.28% | 7.8 | 0.062 |
| cmc | CN2:decision list (each class) | 48.21% | 19.9 | 0.199 |
| cmc | CN2:decision list (seed's class) | 45.96% | 31.3 | 0.309 |
| cmc | CN2:decision list (seed) | 47.52% | 30.5 | 0.310 |
| cmc | CN2:rule set (best rule) | 46.24% | 22.0 | 0.105 |
| cmc | CN2:rule set (distributions) | 45.35% | 22.0 | 0.105 |
| cmc | CN2:rule set (votes) | 45.08% | 22.0 | 0.105 |
| cmc | PFossil:cascade (most frequent first) | 36.66% | 1.4 | 0.007 |
| cmc | PFossil:cascade (rarest first) | 51.53% | 4.8 | 0.013 |
| cmc | PFossil:decision list (any class) | 45.02% | 0.7 | 0.005 |
| cmc | PFossil:decision list (each class) | 51.46% | 10.9 | 0.037 |
| cmc | PFossil:decision list (seed's class) | 50.58% | 15.2 | 0.082 |
| cmc | PFossil:decision list (seed) | 50.51% | 10.8 | 0.069 |
| cmc | PFossil:rule set (best rule) | 52.61% | 2.1 | 0.011 |
| cmc | PFossil:rule set (distributions) | 51.73% | 2.1 | 0.011 |
| cmc | PFossil:rule set (votes) | 51.73% | 2.1 | 0.011 |
| cmc | Pypper:cascade (most frequent first) | 49.70% | 5.6 | 0.098 |
| cmc | Pypper:cascade (rarest first) | 54.79% | 4.1 | 0.093 |
| cmc | Pypper:decision list (any class) | 50.85% | 8.6 | 0.046 |
| cmc | Pypper:decision list (each class) | 49.29% | 5.0 | 0.054 |
| cmc | Pypper:decision list (seed's class) | 51.60% | 52.7 | 0.332 |
| cmc | Pypper:decision list (seed) | 52.89% | 43.5 | 0.310 |
| cmc | Pypper:rule set (best rule) | 52.41% | 6.5 | 0.111 |
| cmc | Pypper:rule set (distributions) | 52.48% | 6.5 | 0.111 |
| cmc | Pypper:rule set (votes) | 52.48% | 6.5 | 0.111 |
| colic | CN2:concept (majority class) | 68.18% | 11.6 | 0.024 |
| colic | CN2:concept (minority class) | 74.98% | 4.6 | 0.013 |
| colic | CN2:decision list (any class) | 70.65% | 13.4 | 0.046 |
| colic | CN2:rule set (best rule) | 75.26% | 16.2 | 0.036 |
| colic | CN2:rule set (distributions) | 75.26% | 16.2 | 0.036 |
| colic | CN2:rule set (votes) | 75.25% | 16.2 | 0.036 |
| colic | PFossil:concept (majority class) | 81.53% | 9.6 | 0.010 |
| colic | PFossil:concept (minority class) | 82.91% | 10.2 | 0.012 |
| colic | PFossil:decision list (any class) | 83.70% | 6.3 | 0.014 |
| colic | PFossil:rule set (best rule) | 82.89% | 19.8 | 0.023 |
| colic | PFossil:rule set (distributions) | 86.15% | 19.8 | 0.023 |
| colic | PFossil:rule set (votes) | 82.88% | 19.8 | 0.023 |
| colic | Pypper:concept (majority class) | 84.53% | 2.1 | 0.025 |
| colic | Pypper:concept (minority class) | 85.08% | 1.8 | 0.018 |
| colic | Pypper:decision list (any class) | 84.27% | 4.0 | 0.015 |
| colic | Pypper:rule set (best rule) | 85.88% | 3.9 | 0.043 |
| colic | Pypper:rule set (distributions) | 85.88% | 3.9 | 0.043 |
| colic | Pypper:rule set (votes) | 85.88% | 3.9 | 0.043 |
| compas-two-years | CN2:concept (majority class) | 55.87% | 16.2 | 0.202 |
| compas-two-years | CN2:concept (minority class) | 56.50% | 15.8 | 0.129 |
| compas-two-years | CN2:decision list (any class) | 55.08% | 15.1 | 0.273 |
| compas-two-years | CN2:rule set (best rule) | 56.50% | 32.0 | 0.335 |
| compas-two-years | CN2:rule set (distributions) | 56.38% | 32.0 | 0.335 |
| compas-two-years | CN2:rule set (votes) | 56.38% | 32.0 | 0.335 |
| compas-two-years | PFossil:concept (majority class) | 64.78% | 1.0 | 0.007 |
| compas-two-years | PFossil:concept (minority class) | 52.96% | 0.0 | 0.004 |
| compas-two-years | PFossil:decision list (any class) | 52.96% | 0.0 | 0.011 |
| compas-two-years | PFossil:rule set (best rule) | 52.96% | 1.0 | 0.017 |
| compas-two-years | PFossil:rule set (distributions) | 52.96% | 1.0 | 0.017 |
| compas-two-years | PFossil:rule set (votes) | 52.96% | 1.0 | 0.017 |
| compas-two-years | Pypper:concept (majority class) | 67.56% | 4.1 | 0.316 |
| compas-two-years | Pypper:concept (minority class) | 67.01% | 4.0 | 0.205 |
| compas-two-years | Pypper:decision list (any class) | 65.93% | 16.7 | 0.200 |
| compas-two-years | Pypper:rule set (best rule) | 67.05% | 8.1 | 0.523 |
| compas-two-years | Pypper:rule set (distributions) | 67.13% | 8.1 | 0.523 |
| compas-two-years | Pypper:rule set (votes) | 67.15% | 8.1 | 0.523 |
| credit-approval | CN2:concept (majority class) | 79.86% | 11.6 | 0.036 |
| credit-approval | CN2:concept (minority class) | 78.26% | 8.2 | 0.022 |
| credit-approval | CN2:decision list (any class) | 76.23% | 12.5 | 0.063 |
| credit-approval | CN2:rule set (best rule) | 78.70% | 19.8 | 0.057 |
| credit-approval | CN2:rule set (distributions) | 78.70% | 19.8 | 0.057 |
| credit-approval | CN2:rule set (votes) | 78.41% | 19.8 | 0.057 |
| credit-approval | PFossil:concept (majority class) | 83.04% | 6.3 | 0.013 |
| credit-approval | PFossil:concept (minority class) | 84.20% | 2.5 | 0.005 |
| credit-approval | PFossil:decision list (any class) | 84.35% | 3.7 | 0.012 |
| credit-approval | PFossil:rule set (best rule) | 82.17% | 8.8 | 0.018 |
| credit-approval | PFossil:rule set (distributions) | 85.07% | 8.8 | 0.018 |
| credit-approval | PFossil:rule set (votes) | 82.46% | 8.8 | 0.018 |
| credit-approval | Pypper:concept (majority class) | 84.93% | 2.6 | 0.034 |
| credit-approval | Pypper:concept (minority class) | 83.91% | 3.8 | 0.037 |
| credit-approval | Pypper:decision list (any class) | 83.33% | 5.9 | 0.023 |
| credit-approval | Pypper:rule set (best rule) | 84.20% | 6.4 | 0.070 |
| credit-approval | Pypper:rule set (distributions) | 83.77% | 6.4 | 0.070 |
| credit-approval | Pypper:rule set (votes) | 83.04% | 6.4 | 0.070 |
| credit-g | CN2:concept (majority class) | 55.90% | 14.1 | 0.041 |
| credit-g | CN2:concept (minority class) | 69.70% | 8.7 | 0.026 |
| credit-g | CN2:decision list (any class) | 70.50% | 16.4 | 0.084 |
| credit-g | CN2:rule set (best rule) | 71.70% | 22.8 | 0.066 |
| credit-g | CN2:rule set (distributions) | 71.80% | 22.8 | 0.066 |
| credit-g | CN2:rule set (votes) | 71.70% | 22.8 | 0.066 |
| credit-g | PFossil:concept (majority class) | 57.50% | 1.4 | 0.006 |
| credit-g | PFossil:concept (minority class) | 72.70% | 2.0 | 0.009 |
| credit-g | PFossil:decision list (any class) | 73.90% | 2.0 | 0.015 |
| credit-g | PFossil:rule set (best rule) | 73.20% | 3.4 | 0.016 |
| credit-g | PFossil:rule set (distributions) | 73.30% | 3.4 | 0.016 |
| credit-g | PFossil:rule set (votes) | 73.20% | 3.4 | 0.016 |
| credit-g | Pypper:concept (majority class) | 63.30% | 2.3 | 0.056 |
| credit-g | Pypper:concept (minority class) | 71.10% | 2.8 | 0.042 |
| credit-g | Pypper:decision list (any class) | 71.60% | 5.0 | 0.027 |
| credit-g | Pypper:rule set (best rule) | 71.40% | 5.1 | 0.097 |
| credit-g | Pypper:rule set (distributions) | 71.50% | 5.1 | 0.097 |
| credit-g | Pypper:rule set (votes) | 71.50% | 5.1 | 0.097 |
| cylinder-bands | CN2:concept (majority class) | 57.59% | 9.2 | 0.022 |
| cylinder-bands | CN2:concept (minority class) | 70.00% | 15.9 | 0.029 |
| cylinder-bands | CN2:decision list (any class) | 68.33% | 20.2 | 0.064 |
| cylinder-bands | CN2:rule set (best rule) | 69.44% | 25.1 | 0.052 |
| cylinder-bands | CN2:rule set (distributions) | 69.44% | 25.1 | 0.052 |
| cylinder-bands | CN2:rule set (votes) | 69.63% | 25.1 | 0.052 |
| cylinder-bands | PFossil:concept (majority class) | 74.44% | 7.6 | 0.017 |
| cylinder-bands | PFossil:concept (minority class) | 68.70% | 1.8 | 0.013 |
| cylinder-bands | PFossil:decision list (any class) | 74.44% | 6.9 | 0.032 |
| cylinder-bands | PFossil:rule set (best rule) | 70.19% | 9.4 | 0.030 |
| cylinder-bands | PFossil:rule set (distributions) | 68.15% | 9.4 | 0.030 |
| cylinder-bands | PFossil:rule set (votes) | 67.96% | 9.4 | 0.030 |
| cylinder-bands | Pypper:concept (majority class) | 73.70% | 4.9 | 0.059 |
| cylinder-bands | Pypper:concept (minority class) | 65.37% | 4.1 | 0.050 |
| cylinder-bands | Pypper:decision list (any class) | 74.81% | 7.5 | 0.034 |
| cylinder-bands | Pypper:rule set (best rule) | 68.33% | 9.0 | 0.111 |
| cylinder-bands | Pypper:rule set (distributions) | 67.41% | 9.0 | 0.111 |
| cylinder-bands | Pypper:rule set (votes) | 68.33% | 9.0 | 0.111 |
| diabetes | CN2:concept (majority class) | 68.76% | 15.1 | 0.044 |
| diabetes | CN2:concept (minority class) | 72.00% | 8.7 | 0.024 |
| diabetes | CN2:decision list (any class) | 70.97% | 18.9 | 0.094 |
| diabetes | CN2:rule set (best rule) | 71.61% | 23.8 | 0.069 |
| diabetes | CN2:rule set (distributions) | 71.48% | 23.8 | 0.069 |
| diabetes | CN2:rule set (votes) | 71.61% | 23.8 | 0.069 |
| diabetes | PFossil:concept (majority class) | 74.09% | 6.1 | 0.010 |
| diabetes | PFossil:concept (minority class) | 75.26% | 3.0 | 0.007 |
| diabetes | PFossil:decision list (any class) | 73.05% | 3.4 | 0.011 |
| diabetes | PFossil:rule set (best rule) | 74.61% | 9.1 | 0.017 |
| diabetes | PFossil:rule set (distributions) | 75.13% | 9.1 | 0.017 |
| diabetes | PFossil:rule set (votes) | 74.09% | 9.1 | 0.017 |
| diabetes | Pypper:concept (majority class) | 71.35% | 3.6 | 0.050 |
| diabetes | Pypper:concept (minority class) | 75.52% | 3.5 | 0.042 |
| diabetes | Pypper:decision list (any class) | 74.74% | 6.2 | 0.027 |
| diabetes | Pypper:rule set (best rule) | 76.04% | 7.1 | 0.092 |
| diabetes | Pypper:rule set (distributions) | 75.78% | 7.1 | 0.092 |
| diabetes | Pypper:rule set (votes) | 75.91% | 7.1 | 0.092 |
| dna | CN2:cascade (most frequent first) | 88.48% | 39.0 | 0.199 |
| dna | CN2:cascade (rarest first) | 80.76% | 74.8 | 0.745 |
| dna | CN2:decision list (any class) | 88.10% | 63.0 | 0.672 |
| dna | CN2:decision list (each class) | 88.14% | 80.2 | 1.087 |
| dna | CN2:decision list (seed's class) | 87.67% | 82.5 | 0.507 |
| dna | CN2:decision list (seed) | 88.26% | 73.7 | 0.390 |
| dna | CN2:rule set (best rule) | 89.80% | 105.9 | 1.006 |
| dna | CN2:rule set (distributions) | 84.78% | 105.9 | 1.006 |
| dna | CN2:rule set (votes) | 87.48% | 105.9 | 1.006 |
| dna | PFossil:cascade (most frequent first) | 88.26% | 18.1 | 0.051 |
| dna | PFossil:cascade (rarest first) | 89.99% | 15.7 | 0.058 |
| dna | PFossil:decision list (any class) | 89.64% | 12.2 | 0.058 |
| dna | PFossil:decision list (each class) | 93.53% | 12.3 | 0.075 |
| dna | PFossil:decision list (seed's class) | 90.36% | 13.0 | 0.036 |
| dna | PFossil:decision list (seed) | 91.05% | 11.5 | 0.030 |
| dna | PFossil:rule set (best rule) | 93.69% | 27.8 | 0.098 |
| dna | PFossil:rule set (distributions) | 89.89% | 27.8 | 0.098 |
| dna | PFossil:rule set (votes) | 90.65% | 27.8 | 0.098 |
| dna | Pypper:cascade (most frequent first) | 91.18% | 19.3 | 0.202 |
| dna | Pypper:cascade (rarest first) | 92.22% | 13.2 | 0.230 |
| dna | Pypper:decision list (any class) | 90.74% | 20.1 | 0.142 |
| dna | Pypper:decision list (each class) | 92.84% | 17.3 | 0.190 |
| dna | Pypper:decision list (seed's class) | 91.56% | 27.0 | 0.119 |
| dna | Pypper:decision list (seed) | 91.37% | 24.1 | 0.112 |
| dna | Pypper:rule set (best rule) | 93.75% | 22.6 | 0.361 |
| dna | Pypper:rule set (distributions) | 93.31% | 22.6 | 0.361 |
| dna | Pypper:rule set (votes) | 93.60% | 22.6 | 0.361 |
| dresses-sales | CN2:concept (majority class) | 42.60% | 2.4 | 0.005 |
| dresses-sales | CN2:concept (minority class) | 58.00% | 0.8 | 0.003 |
| dresses-sales | CN2:decision list (any class) | 58.00% | 2.8 | 0.010 |
| dresses-sales | CN2:rule set (best rule) | 58.20% | 3.2 | 0.008 |
| dresses-sales | CN2:rule set (distributions) | 58.20% | 3.2 | 0.008 |
| dresses-sales | CN2:rule set (votes) | 58.20% | 3.2 | 0.008 |
| dresses-sales | PFossil:concept (majority class) | 53.20% | 1.5 | 0.007 |
| dresses-sales | PFossil:concept (minority class) | 59.40% | 0.7 | 0.006 |
| dresses-sales | PFossil:decision list (any class) | 57.80% | 1.4 | 0.012 |
| dresses-sales | PFossil:rule set (best rule) | 59.40% | 2.2 | 0.013 |
| dresses-sales | PFossil:rule set (distributions) | 59.20% | 2.2 | 0.013 |
| dresses-sales | PFossil:rule set (votes) | 59.20% | 2.2 | 0.013 |
| dresses-sales | Pypper:concept (majority class) | 51.80% | 1.0 | 0.035 |
| dresses-sales | Pypper:concept (minority class) | 61.20% | 1.1 | 0.021 |
| dresses-sales | Pypper:decision list (any class) | 59.00% | 7.2 | 0.024 |
| dresses-sales | Pypper:rule set (best rule) | 60.40% | 2.1 | 0.056 |
| dresses-sales | Pypper:rule set (distributions) | 60.60% | 2.1 | 0.056 |
| dresses-sales | Pypper:rule set (votes) | 60.60% | 2.1 | 0.056 |
| ecoli | CN2:cascade (most frequent first) | 80.98% | 14.4 | 0.122 |
| ecoli | CN2:cascade (rarest first) | 82.51% | 11.6 | 0.072 |
| ecoli | CN2:decision list (any class) | 80.74% | 10.3 | 0.253 |
| ecoli | CN2:decision list (each class) | 82.81% | 15.1 | 0.559 |
| ecoli | CN2:decision list (seed's class) | 81.32% | 13.8 | 0.111 |
| ecoli | CN2:decision list (seed) | 82.19% | 12.6 | 0.106 |
| ecoli | CN2:rule set (best rule) | 80.42% | 16.9 | 0.145 |
| ecoli | CN2:rule set (distributions) | 81.31% | 16.9 | 0.145 |
| ecoli | CN2:rule set (votes) | 82.79% | 16.9 | 0.145 |
| ecoli | PFossil:cascade (most frequent first) | 82.79% | 20.2 | 0.019 |
| ecoli | PFossil:cascade (rarest first) | 75.64% | 23.0 | 0.027 |
| ecoli | PFossil:decision list (any class) | 85.19% | 19.8 | 0.035 |
| ecoli | PFossil:decision list (each class) | 81.02% | 16.2 | 0.064 |
| ecoli | PFossil:decision list (seed's class) | 84.00% | 14.0 | 0.013 |
| ecoli | PFossil:decision list (seed) | 81.33% | 13.1 | 0.013 |
| ecoli | PFossil:rule set (best rule) | 81.92% | 30.0 | 0.037 |
| ecoli | PFossil:rule set (distributions) | 84.29% | 30.0 | 0.037 |
| ecoli | PFossil:rule set (votes) | 83.98% | 30.0 | 0.037 |
| ecoli | Pypper:cascade (most frequent first) | 81.92% | 8.0 | 0.057 |
| ecoli | Pypper:cascade (rarest first) | 80.70% | 7.3 | 0.048 |
| ecoli | Pypper:decision list (any class) | 81.60% | 11.5 | 0.035 |
| ecoli | Pypper:decision list (each class) | 80.71% | 11.6 | 0.112 |
| ecoli | Pypper:decision list (seed's class) | 80.71% | 11.4 | 0.029 |
| ecoli | Pypper:decision list (seed) | 83.08% | 10.9 | 0.029 |
| ecoli | Pypper:rule set (best rule) | 81.91% | 10.4 | 0.077 |
| ecoli | Pypper:rule set (distributions) | 81.29% | 10.4 | 0.077 |
| ecoli | Pypper:rule set (votes) | 81.29% | 10.4 | 0.077 |
| glass | CN2:cascade (most frequent first) | 68.20% | 12.0 | 0.102 |
| glass | CN2:cascade (rarest first) | 69.20% | 11.4 | 0.047 |
| glass | CN2:decision list (any class) | 67.23% | 10.0 | 0.190 |
| glass | CN2:decision list (each class) | 71.52% | 15.3 | 0.371 |
| glass | CN2:decision list (seed's class) | 68.23% | 15.7 | 0.130 |
| glass | CN2:decision list (seed) | 65.00% | 14.0 | 0.122 |
| glass | CN2:rule set (best rule) | 70.61% | 15.3 | 0.103 |
| glass | CN2:rule set (distributions) | 70.13% | 15.3 | 0.103 |
| glass | CN2:rule set (votes) | 70.11% | 15.3 | 0.103 |
| glass | PFossil:cascade (most frequent first) | 68.29% | 16.8 | 0.018 |
| glass | PFossil:cascade (rarest first) | 63.55% | 15.6 | 0.017 |
| glass | PFossil:decision list (any class) | 68.77% | 17.1 | 0.029 |
| glass | PFossil:decision list (each class) | 63.57% | 14.1 | 0.048 |
| glass | PFossil:decision list (seed's class) | 67.42% | 13.1 | 0.013 |
| glass | PFossil:decision list (seed) | 64.07% | 10.0 | 0.011 |
| glass | PFossil:rule set (best rule) | 71.97% | 31.2 | 0.035 |
| glass | PFossil:rule set (distributions) | 70.11% | 31.2 | 0.035 |
| glass | PFossil:rule set (votes) | 69.68% | 31.2 | 0.035 |
| glass | Pypper:cascade (most frequent first) | 66.26% | 8.5 | 0.052 |
| glass | Pypper:cascade (rarest first) | 68.33% | 6.1 | 0.044 |
| glass | Pypper:decision list (any class) | 66.41% | 9.3 | 0.025 |
| glass | Pypper:decision list (each class) | 68.68% | 8.6 | 0.077 |
| glass | Pypper:decision list (seed's class) | 66.30% | 12.5 | 0.031 |
| glass | Pypper:decision list (seed) | 64.94% | 12.7 | 0.033 |
| glass | Pypper:rule set (best rule) | 64.55% | 9.1 | 0.070 |
| glass | Pypper:rule set (distributions) | 62.75% | 9.1 | 0.070 |
| glass | Pypper:rule set (votes) | 64.52% | 9.1 | 0.070 |
| hayes-roth | CN2:cascade (most frequent first) | 76.88% | 6.3 | 0.016 |
| hayes-roth | CN2:cascade (rarest first) | 85.00% | 5.9 | 0.011 |
| hayes-roth | CN2:decision list (any class) | 85.62% | 8.0 | 0.027 |
| hayes-roth | CN2:decision list (each class) | 83.12% | 8.0 | 0.030 |
| hayes-roth | CN2:decision list (seed's class) | 84.38% | 9.0 | 0.027 |
| hayes-roth | CN2:decision list (seed) | 81.25% | 9.1 | 0.028 |
| hayes-roth | CN2:rule set (best rule) | 83.12% | 12.8 | 0.026 |
| hayes-roth | CN2:rule set (distributions) | 81.88% | 12.8 | 0.026 |
| hayes-roth | CN2:rule set (votes) | 78.75% | 12.8 | 0.026 |
| hayes-roth | PFossil:cascade (most frequent first) | 81.25% | 5.9 | 0.006 |
| hayes-roth | PFossil:cascade (rarest first) | 85.62% | 7.6 | 0.006 |
| hayes-roth | PFossil:decision list (any class) | 84.38% | 8.7 | 0.010 |
| hayes-roth | PFossil:decision list (each class) | 82.50% | 7.3 | 0.013 |
| hayes-roth | PFossil:decision list (seed's class) | 75.00% | 7.9 | 0.008 |
| hayes-roth | PFossil:decision list (seed) | 66.25% | 5.3 | 0.009 |
| hayes-roth | PFossil:rule set (best rule) | 82.50% | 13.5 | 0.012 |
| hayes-roth | PFossil:rule set (distributions) | 81.88% | 13.5 | 0.012 |
| hayes-roth | PFossil:rule set (votes) | 81.25% | 13.5 | 0.012 |
| hayes-roth | Pypper:cascade (most frequent first) | 75.62% | 4.2 | 0.021 |
| hayes-roth | Pypper:cascade (rarest first) | 84.38% | 6.1 | 0.025 |
| hayes-roth | Pypper:decision list (any class) | 78.12% | 7.7 | 0.016 |
| hayes-roth | Pypper:decision list (each class) | 82.50% | 8.0 | 0.028 |
| hayes-roth | Pypper:decision list (seed's class) | 77.50% | 10.5 | 0.022 |
| hayes-roth | Pypper:decision list (seed) | 70.00% | 9.8 | 0.022 |
| hayes-roth | Pypper:rule set (best rule) | 79.38% | 9.4 | 0.040 |
| hayes-roth | Pypper:rule set (distributions) | 75.00% | 9.4 | 0.040 |
| hayes-roth | Pypper:rule set (votes) | 75.00% | 9.4 | 0.040 |
| heart-c | CN2:concept (majority class) | 68.37% | 5.3 | 0.012 |
| heart-c | CN2:concept (minority class) | 71.60% | 3.9 | 0.009 |
| heart-c | CN2:decision list (any class) | 75.59% | 8.6 | 0.034 |
| heart-c | CN2:rule set (best rule) | 70.95% | 9.2 | 0.021 |
| heart-c | CN2:rule set (distributions) | 70.95% | 9.2 | 0.021 |
| heart-c | CN2:rule set (votes) | 70.29% | 9.2 | 0.021 |
| heart-c | PFossil:concept (majority class) | 78.27% | 7.9 | 0.010 |
| heart-c | PFossil:concept (minority class) | 72.95% | 10.3 | 0.015 |
| heart-c | PFossil:decision list (any class) | 75.59% | 6.7 | 0.015 |
| heart-c | PFossil:rule set (best rule) | 79.54% | 18.2 | 0.025 |
| heart-c | PFossil:rule set (distributions) | 77.24% | 18.2 | 0.025 |
| heart-c | PFossil:rule set (votes) | 78.24% | 18.2 | 0.025 |
| heart-c | Pypper:concept (majority class) | 82.27% | 2.7 | 0.024 |
| heart-c | Pypper:concept (minority class) | 81.55% | 3.0 | 0.022 |
| heart-c | Pypper:decision list (any class) | 83.19% | 3.9 | 0.013 |
| heart-c | Pypper:rule set (best rule) | 82.55% | 5.7 | 0.046 |
| heart-c | Pypper:rule set (distributions) | 82.56% | 5.7 | 0.046 |
| heart-c | Pypper:rule set (votes) | 82.56% | 5.7 | 0.046 |
| heart-h | CN2:concept (majority class) | 76.89% | 7.0 | 0.022 |
| heart-h | CN2:concept (minority class) | 73.47% | 6.5 | 0.012 |
| heart-h | CN2:decision list (any class) | 74.17% | 10.5 | 0.037 |
| heart-h | CN2:rule set (best rule) | 74.83% | 13.5 | 0.034 |
| heart-h | CN2:rule set (distributions) | 74.49% | 13.5 | 0.034 |
| heart-h | CN2:rule set (votes) | 75.52% | 13.5 | 0.034 |
| heart-h | PFossil:concept (majority class) | 81.28% | 9.4 | 0.010 |
| heart-h | PFossil:concept (minority class) | 75.15% | 7.8 | 0.010 |
| heart-h | PFossil:decision list (any class) | 75.17% | 7.0 | 0.015 |
| heart-h | PFossil:rule set (best rule) | 81.28% | 17.2 | 0.021 |
| heart-h | PFossil:rule set (distributions) | 78.93% | 17.2 | 0.021 |
| heart-h | PFossil:rule set (votes) | 80.94% | 17.2 | 0.021 |
| heart-h | Pypper:concept (majority class) | 75.45% | 2.5 | 0.026 |
| heart-h | Pypper:concept (minority class) | 78.54% | 1.9 | 0.017 |
| heart-h | Pypper:decision list (any class) | 80.25% | 2.8 | 0.011 |
| heart-h | Pypper:rule set (best rule) | 78.20% | 4.4 | 0.042 |
| heart-h | Pypper:rule set (distributions) | 77.85% | 4.4 | 0.042 |
| heart-h | Pypper:rule set (votes) | 77.85% | 4.4 | 0.042 |
| heart-statlog | CN2:concept (majority class) | 62.96% | 4.8 | 0.011 |
| heart-statlog | CN2:concept (minority class) | 74.44% | 4.1 | 0.010 |
| heart-statlog | CN2:decision list (any class) | 72.59% | 8.3 | 0.032 |
| heart-statlog | CN2:rule set (best rule) | 73.33% | 8.9 | 0.021 |
| heart-statlog | CN2:rule set (distributions) | 73.33% | 8.9 | 0.021 |
| heart-statlog | CN2:rule set (votes) | 72.96% | 8.9 | 0.021 |
| heart-statlog | PFossil:concept (majority class) | 78.52% | 8.1 | 0.010 |
| heart-statlog | PFossil:concept (minority class) | 75.19% | 9.5 | 0.014 |
| heart-statlog | PFossil:decision list (any class) | 77.78% | 7.1 | 0.017 |
| heart-statlog | PFossil:rule set (best rule) | 80.74% | 17.6 | 0.021 |
| heart-statlog | PFossil:rule set (distributions) | 80.00% | 17.6 | 0.021 |
| heart-statlog | PFossil:rule set (votes) | 79.26% | 17.6 | 0.021 |
| heart-statlog | Pypper:concept (majority class) | 83.70% | 2.7 | 0.025 |
| heart-statlog | Pypper:concept (minority class) | 79.26% | 2.3 | 0.020 |
| heart-statlog | Pypper:decision list (any class) | 80.00% | 3.7 | 0.013 |
| heart-statlog | Pypper:rule set (best rule) | 79.26% | 5.0 | 0.042 |
| heart-statlog | Pypper:rule set (distributions) | 80.74% | 5.0 | 0.042 |
| heart-statlog | Pypper:rule set (votes) | 80.74% | 5.0 | 0.042 |
| heloc | CN2:concept (majority class) | 64.33% | 46.5 | 0.518 |
| heloc | CN2:concept (minority class) | 58.79% | 51.8 | 0.475 |
| heloc | CN2:decision list (any class) | 58.91% | 70.1 | 1.628 |
| heloc | CN2:rule set (best rule) | 64.39% | 98.3 | 1.004 |
| heloc | CN2:rule set (distributions) | 64.31% | 98.3 | 1.004 |
| heloc | CN2:rule set (votes) | 64.34% | 98.3 | 1.004 |
| heloc | PFossil:concept (majority class) | 68.83% | 1.0 | 0.020 |
| heloc | PFossil:concept (minority class) | 69.17% | 1.0 | 0.014 |
| heloc | PFossil:decision list (any class) | 68.83% | 1.0 | 0.066 |
| heloc | PFossil:rule set (best rule) | 68.83% | 2.0 | 0.045 |
| heloc | PFossil:rule set (distributions) | 68.83% | 2.0 | 0.045 |
| heloc | PFossil:rule set (votes) | 68.83% | 2.0 | 0.045 |
| heloc | Pypper:concept (majority class) | 70.61% | 7.3 | 0.555 |
| heloc | Pypper:concept (minority class) | 70.12% | 6.8 | 0.589 |
| heloc | Pypper:decision list (any class) | 70.14% | 24.1 | 0.552 |
| heloc | Pypper:rule set (best rule) | 70.45% | 14.1 | 1.149 |
| heloc | Pypper:rule set (distributions) | 70.23% | 14.1 | 1.149 |
| heloc | Pypper:rule set (votes) | 70.26% | 14.1 | 1.149 |
| hepatitis | CN2:concept (majority class) | 72.88% | 6.1 | 0.010 |
| hepatitis | CN2:concept (minority class) | 82.67% | 2.4 | 0.004 |
| hepatitis | CN2:decision list (any class) | 83.38% | 6.9 | 0.018 |
| hepatitis | CN2:rule set (best rule) | 82.62% | 8.5 | 0.014 |
| hepatitis | CN2:rule set (distributions) | 83.25% | 8.5 | 0.014 |
| hepatitis | CN2:rule set (votes) | 82.58% | 8.5 | 0.014 |
| hepatitis | PFossil:concept (majority class) | 76.67% | 8.2 | 0.007 |
| hepatitis | PFossil:concept (minority class) | 73.79% | 6.8 | 0.007 |
| hepatitis | PFossil:decision list (any class) | 75.50% | 6.0 | 0.009 |
| hepatitis | PFossil:rule set (best rule) | 78.75% | 15.0 | 0.014 |
| hepatitis | PFossil:rule set (distributions) | 80.04% | 15.0 | 0.014 |
| hepatitis | PFossil:rule set (votes) | 78.75% | 15.0 | 0.014 |
| hepatitis | Pypper:concept (majority class) | 72.17% | 2.1 | 0.015 |
| hepatitis | Pypper:concept (minority class) | 79.42% | 1.7 | 0.010 |
| hepatitis | Pypper:decision list (any class) | 80.08% | 1.8 | 0.006 |
| hepatitis | Pypper:rule set (best rule) | 80.04% | 3.8 | 0.026 |
| hepatitis | Pypper:rule set (distributions) | 80.04% | 3.8 | 0.026 |
| hepatitis | Pypper:rule set (votes) | 80.67% | 3.8 | 0.026 |
| ilpd | CN2:concept (majority class) | 57.45% | 6.1 | 0.016 |
| ilpd | CN2:concept (minority class) | 71.69% | 5.3 | 0.013 |
| ilpd | CN2:decision list (any class) | 71.36% | 9.5 | 0.041 |
| ilpd | CN2:rule set (best rule) | 71.52% | 11.4 | 0.029 |
| ilpd | CN2:rule set (distributions) | 71.52% | 11.4 | 0.029 |
| ilpd | CN2:rule set (votes) | 71.52% | 11.4 | 0.029 |
| ilpd | PFossil:concept (majority class) | 54.19% | 1.2 | 0.003 |
| ilpd | PFossil:concept (minority class) | 63.79% | 1.6 | 0.005 |
| ilpd | PFossil:decision list (any class) | 71.36% | 1.2 | 0.006 |
| ilpd | PFossil:rule set (best rule) | 63.28% | 2.8 | 0.009 |
| ilpd | PFossil:rule set (distributions) | 63.28% | 2.8 | 0.009 |
| ilpd | PFossil:rule set (votes) | 63.28% | 2.8 | 0.009 |
| ilpd | Pypper:concept (majority class) | 61.06% | 1.8 | 0.043 |
| ilpd | Pypper:concept (minority class) | 68.97% | 1.5 | 0.019 |
| ilpd | Pypper:decision list (any class) | 70.85% | 2.2 | 0.013 |
| ilpd | Pypper:rule set (best rule) | 69.66% | 3.3 | 0.062 |
| ilpd | Pypper:rule set (distributions) | 69.66% | 3.3 | 0.062 |
| ilpd | Pypper:rule set (votes) | 69.66% | 3.3 | 0.062 |
| ionosphere | CN2:concept (majority class) | 80.06% | 7.7 | 0.040 |
| ionosphere | CN2:concept (minority class) | 92.02% | 9.8 | 0.016 |
| ionosphere | CN2:decision list (any class) | 89.17% | 7.9 | 0.025 |
| ionosphere | CN2:rule set (best rule) | 92.01% | 17.5 | 0.056 |
| ionosphere | CN2:rule set (distributions) | 88.60% | 17.5 | 0.056 |
| ionosphere | CN2:rule set (votes) | 89.45% | 17.5 | 0.056 |
| ionosphere | PFossil:concept (majority class) | 88.60% | 3.5 | 0.009 |
| ionosphere | PFossil:concept (minority class) | 87.47% | 10.1 | 0.011 |
| ionosphere | PFossil:decision list (any class) | 88.03% | 3.6 | 0.016 |
| ionosphere | PFossil:rule set (best rule) | 89.75% | 13.6 | 0.019 |
| ionosphere | PFossil:rule set (distributions) | 89.75% | 13.6 | 0.019 |
| ionosphere | PFossil:rule set (votes) | 90.03% | 13.6 | 0.019 |
| ionosphere | Pypper:concept (majority class) | 89.46% | 2.6 | 0.029 |
| ionosphere | Pypper:concept (minority class) | 89.47% | 3.8 | 0.024 |
| ionosphere | Pypper:decision list (any class) | 89.47% | 4.7 | 0.016 |
| ionosphere | Pypper:rule set (best rule) | 90.61% | 6.4 | 0.052 |
| ionosphere | Pypper:rule set (distributions) | 91.17% | 6.4 | 0.052 |
| ionosphere | Pypper:rule set (votes) | 90.89% | 6.4 | 0.052 |
| iris | CN2:cascade (most frequent first) | 94.00% | 3.0 | 0.005 |
| iris | CN2:cascade (rarest first) | 94.00% | 3.0 | 0.005 |
| iris | CN2:decision list (any class) | 95.33% | 3.0 | 0.007 |
| iris | CN2:decision list (each class) | 94.67% | 3.1 | 0.010 |
| iris | CN2:decision list (seed's class) | 89.33% | 3.4 | 0.009 |
| iris | CN2:decision list (seed) | 88.00% | 3.4 | 0.012 |
| iris | CN2:rule set (best rule) | 91.33% | 6.2 | 0.013 |
| iris | CN2:rule set (distributions) | 91.33% | 6.2 | 0.013 |
| iris | CN2:rule set (votes) | 91.33% | 6.2 | 0.013 |
| iris | PFossil:cascade (most frequent first) | 92.67% | 3.7 | 0.004 |
| iris | PFossil:cascade (rarest first) | 92.67% | 3.7 | 0.004 |
| iris | PFossil:decision list (any class) | 92.67% | 4.0 | 0.005 |
| iris | PFossil:decision list (each class) | 92.67% | 3.7 | 0.006 |
| iris | PFossil:decision list (seed's class) | 92.00% | 3.3 | 0.003 |
| iris | PFossil:decision list (seed) | 92.67% | 3.2 | 0.003 |
| iris | PFossil:rule set (best rule) | 92.00% | 5.8 | 0.006 |
| iris | PFossil:rule set (distributions) | 92.00% | 5.8 | 0.006 |
| iris | PFossil:rule set (votes) | 91.33% | 5.8 | 0.006 |
| iris | Pypper:cascade (most frequent first) | 93.33% | 2.0 | 0.010 |
| iris | Pypper:cascade (rarest first) | 93.33% | 2.0 | 0.010 |
| iris | Pypper:decision list (any class) | 92.67% | 2.6 | 0.006 |
| iris | Pypper:decision list (each class) | 92.67% | 2.6 | 0.010 |
| iris | Pypper:decision list (seed's class) | 90.00% | 3.3 | 0.014 |
| iris | Pypper:decision list (seed) | 92.00% | 2.9 | 0.017 |
| iris | Pypper:rule set (best rule) | 92.00% | 3.7 | 0.020 |
| iris | Pypper:rule set (distributions) | 92.00% | 3.7 | 0.020 |
| iris | Pypper:rule set (votes) | 92.00% | 3.7 | 0.020 |
| kc1 | CN2:concept (majority class) | 58.98% | 8.0 | 0.191 |
| kc1 | CN2:concept (minority class) | 84.92% | 10.8 | 0.044 |
| kc1 | CN2:decision list (any class) | 84.78% | 13.6 | 0.488 |
| kc1 | CN2:rule set (best rule) | 85.02% | 18.8 | 0.234 |
| kc1 | CN2:rule set (distributions) | 85.02% | 18.8 | 0.234 |
| kc1 | CN2:rule set (votes) | 85.02% | 18.8 | 0.234 |
| kc1 | PFossil:concept (majority class) | 80.99% | 2.5 | 0.011 |
| kc1 | PFossil:concept (minority class) | 80.51% | 1.0 | 0.006 |
| kc1 | PFossil:decision list (any class) | 80.51% | 1.0 | 0.013 |
| kc1 | PFossil:rule set (best rule) | 83.31% | 3.5 | 0.019 |
| kc1 | PFossil:rule set (distributions) | 83.31% | 3.5 | 0.019 |
| kc1 | PFossil:rule set (votes) | 83.31% | 3.5 | 0.019 |
| kc1 | Pypper:concept (majority class) | 64.39% | 1.0 | 0.099 |
| kc1 | Pypper:concept (minority class) | 84.64% | 2.0 | 0.034 |
| kc1 | Pypper:decision list (any class) | 84.92% | 2.8 | 0.027 |
| kc1 | Pypper:rule set (best rule) | 84.64% | 3.0 | 0.133 |
| kc1 | Pypper:rule set (distributions) | 84.64% | 3.0 | 0.133 |
| kc1 | Pypper:rule set (votes) | 84.64% | 3.0 | 0.133 |
| kc2 | CN2:concept (majority class) | 77.57% | 7.6 | 0.219 |
| kc2 | CN2:concept (minority class) | 83.14% | 4.2 | 0.011 |
| kc2 | CN2:decision list (any class) | 82.76% | 9.9 | 0.377 |
| kc2 | CN2:rule set (best rule) | 83.72% | 11.8 | 0.229 |
| kc2 | CN2:rule set (distributions) | 83.72% | 11.8 | 0.229 |
| kc2 | CN2:rule set (votes) | 83.72% | 11.8 | 0.229 |
| kc2 | PFossil:concept (majority class) | 82.35% | 4.8 | 0.008 |
| kc2 | PFossil:concept (minority class) | 77.38% | 2.6 | 0.007 |
| kc2 | PFossil:decision list (any class) | 77.38% | 2.6 | 0.011 |
| kc2 | PFossil:rule set (best rule) | 82.74% | 7.4 | 0.015 |
| kc2 | PFossil:rule set (distributions) | 81.79% | 7.4 | 0.015 |
| kc2 | PFossil:rule set (votes) | 82.55% | 7.4 | 0.015 |
| kc2 | Pypper:concept (majority class) | 80.63% | 1.3 | 0.033 |
| kc2 | Pypper:concept (minority class) | 83.50% | 2.0 | 0.020 |
| kc2 | Pypper:decision list (any class) | 83.13% | 2.8 | 0.014 |
| kc2 | Pypper:rule set (best rule) | 83.69% | 3.3 | 0.054 |
| kc2 | Pypper:rule set (distributions) | 83.69% | 3.3 | 0.054 |
| kc2 | Pypper:rule set (votes) | 83.69% | 3.3 | 0.054 |
| kr-vs-kp | CN2:concept (majority class) | 81.07% | 9.3 | 0.043 |
| kr-vs-kp | CN2:concept (minority class) | 92.68% | 21.5 | 0.071 |
| kr-vs-kp | CN2:decision list (any class) | 98.69% | 27.1 | 0.152 |
| kr-vs-kp | CN2:rule set (best rule) | 98.34% | 30.8 | 0.116 |
| kr-vs-kp | CN2:rule set (distributions) | 90.80% | 30.8 | 0.116 |
| kr-vs-kp | CN2:rule set (votes) | 89.11% | 30.8 | 0.116 |
| kr-vs-kp | PFossil:concept (majority class) | 95.68% | 4.8 | 0.016 |
| kr-vs-kp | PFossil:concept (minority class) | 98.25% | 9.7 | 0.022 |
| kr-vs-kp | PFossil:decision list (any class) | 94.40% | 4.5 | 0.022 |
| kr-vs-kp | PFossil:rule set (best rule) | 99.37% | 14.5 | 0.041 |
| kr-vs-kp | PFossil:rule set (distributions) | 94.31% | 14.5 | 0.041 |
| kr-vs-kp | PFossil:rule set (votes) | 96.43% | 14.5 | 0.041 |
| kr-vs-kp | Pypper:concept (majority class) | 99.00% | 9.0 | 0.078 |
| kr-vs-kp | Pypper:concept (minority class) | 99.09% | 13.6 | 0.117 |
| kr-vs-kp | Pypper:decision list (any class) | 98.84% | 13.9 | 0.074 |
| kr-vs-kp | Pypper:rule set (best rule) | 99.16% | 22.6 | 0.197 |
| kr-vs-kp | Pypper:rule set (distributions) | 99.03% | 22.6 | 0.197 |
| kr-vs-kp | Pypper:rule set (votes) | 98.91% | 22.6 | 0.197 |
| led24 | CN2:cascade (most frequent first) | 54.31% | 118.7 | 0.749 |
| led24 | CN2:cascade (rarest first) | 45.41% | 123.6 | 0.784 |
| led24 | CN2:decision list (any class) | 65.91% | 95.4 | 1.929 |
| led24 | CN2:decision list (each class) | 65.69% | 118.0 | 5.617 |
| led24 | CN2:decision list (seed's class) | 64.38% | 122.3 | 0.844 |
| led24 | CN2:decision list (seed) | 65.09% | 105.6 | 0.676 |
| led24 | CN2:rule set (best rule) | 63.53% | 201.3 | 1.499 |
| led24 | CN2:rule set (distributions) | 59.75% | 201.3 | 1.499 |
| led24 | CN2:rule set (votes) | 61.12% | 201.3 | 1.499 |
| led24 | PFossil:cascade (most frequent first) | 68.97% | 44.3 | 0.106 |
| led24 | PFossil:cascade (rarest first) | 70.03% | 48.5 | 0.117 |
| led24 | PFossil:decision list (any class) | 71.34% | 33.4 | 0.143 |
| led24 | PFossil:decision list (each class) | 70.59% | 47.7 | 0.359 |
| led24 | PFossil:decision list (seed's class) | 70.19% | 33.1 | 0.073 |
| led24 | PFossil:decision list (seed) | 65.50% | 29.5 | 0.066 |
| led24 | PFossil:rule set (best rule) | 70.00% | 45.4 | 0.153 |
| led24 | PFossil:rule set (distributions) | 70.56% | 45.4 | 0.153 |
| led24 | PFossil:rule set (votes) | 70.34% | 45.4 | 0.153 |
| led24 | Pypper:cascade (most frequent first) | 71.00% | 24.8 | 0.330 |
| led24 | Pypper:cascade (rarest first) | 71.50% | 24.3 | 0.322 |
| led24 | Pypper:decision list (any class) | 65.31% | 27.5 | 0.250 |
| led24 | Pypper:decision list (each class) | 65.06% | 11.6 | 0.435 |
| led24 | Pypper:decision list (seed's class) | 70.34% | 69.8 | 0.405 |
| led24 | Pypper:decision list (seed) | 68.19% | 73.1 | 0.386 |
| led24 | Pypper:rule set (best rule) | 68.00% | 25.4 | 0.474 |
| led24 | Pypper:rule set (distributions) | 67.88% | 25.4 | 0.474 |
| led24 | Pypper:rule set (votes) | 67.47% | 25.4 | 0.474 |
| lymph | CN2:cascade (most frequent first) | 81.86% | 6.0 | 0.017 |
| lymph | CN2:cascade (rarest first) | 75.81% | 7.7 | 0.015 |
| lymph | CN2:decision list (any class) | 82.62% | 5.2 | 0.029 |
| lymph | CN2:decision list (each class) | 84.38% | 7.7 | 0.039 |
| lymph | CN2:decision list (seed's class) | 85.90% | 7.9 | 0.026 |
| lymph | CN2:decision list (seed) | 83.76% | 7.9 | 0.026 |
| lymph | CN2:rule set (best rule) | 79.90% | 11.5 | 0.027 |
| lymph | CN2:rule set (distributions) | 79.90% | 11.5 | 0.027 |
| lymph | CN2:rule set (votes) | 79.90% | 11.5 | 0.027 |
| lymph | PFossil:cascade (most frequent first) | 79.76% | 9.6 | 0.009 |
| lymph | PFossil:cascade (rarest first) | 86.52% | 8.5 | 0.008 |
| lymph | PFossil:decision list (any class) | 80.48% | 9.2 | 0.014 |
| lymph | PFossil:decision list (each class) | 86.48% | 7.0 | 0.017 |
| lymph | PFossil:decision list (seed's class) | 85.10% | 6.9 | 0.007 |
| lymph | PFossil:decision list (seed) | 81.86% | 6.6 | 0.007 |
| lymph | PFossil:rule set (best rule) | 81.71% | 16.2 | 0.016 |
| lymph | PFossil:rule set (distributions) | 81.05% | 16.2 | 0.016 |
| lymph | PFossil:rule set (votes) | 81.10% | 16.2 | 0.016 |
| lymph | Pypper:cascade (most frequent first) | 79.10% | 3.9 | 0.022 |
| lymph | Pypper:cascade (rarest first) | 73.71% | 2.9 | 0.021 |
| lymph | Pypper:decision list (any class) | 75.05% | 4.0 | 0.011 |
| lymph | Pypper:decision list (each class) | 80.43% | 5.1 | 0.027 |
| lymph | Pypper:decision list (seed's class) | 79.86% | 6.2 | 0.015 |
| lymph | Pypper:decision list (seed) | 82.48% | 6.0 | 0.015 |
| lymph | Pypper:rule set (best rule) | 80.52% | 5.6 | 0.036 |
| lymph | Pypper:rule set (distributions) | 77.81% | 5.6 | 0.036 |
| lymph | Pypper:rule set (votes) | 78.52% | 5.6 | 0.036 |
| mofn-3-7-10 | CN2:concept (majority class) | 100.00% | 35.0 | 0.078 |
| mofn-3-7-10 | CN2:concept (minority class) | 99.25% | 20.7 | 0.071 |
| mofn-3-7-10 | CN2:decision list (any class) | 98.72% | 27.8 | 0.120 |
| mofn-3-7-10 | CN2:rule set (best rule) | 99.25% | 55.7 | 0.150 |
| mofn-3-7-10 | CN2:rule set (distributions) | 99.25% | 55.7 | 0.150 |
| mofn-3-7-10 | CN2:rule set (votes) | 99.25% | 55.7 | 0.150 |
| mofn-3-7-10 | PFossil:concept (majority class) | 28.37% | 0.4 | 0.002 |
| mofn-3-7-10 | PFossil:concept (minority class) | 82.78% | 6.0 | 0.010 |
| mofn-3-7-10 | PFossil:decision list (any class) | 82.78% | 6.0 | 0.018 |
| mofn-3-7-10 | PFossil:rule set (best rule) | 83.68% | 6.4 | 0.013 |
| mofn-3-7-10 | PFossil:rule set (distributions) | 83.31% | 6.4 | 0.013 |
| mofn-3-7-10 | PFossil:rule set (votes) | 83.61% | 6.4 | 0.013 |
| mofn-3-7-10 | Pypper:concept (majority class) | 96.76% | 24.3 | 0.179 |
| mofn-3-7-10 | Pypper:concept (minority class) | 94.71% | 14.0 | 0.118 |
| mofn-3-7-10 | Pypper:decision list (any class) | 94.57% | 14.2 | 0.062 |
| mofn-3-7-10 | Pypper:rule set (best rule) | 97.21% | 38.3 | 0.293 |
| mofn-3-7-10 | Pypper:rule set (distributions) | 95.85% | 38.3 | 0.293 |
| mofn-3-7-10 | Pypper:rule set (votes) | 96.00% | 38.3 | 0.293 |
| molecular-biology_promoters | CN2:concept (majority class) | 80.18% | 4.7 | 0.012 |
| molecular-biology_promoters | CN2:concept (minority class) | 82.18% | 4.6 | 0.013 |
| molecular-biology_promoters | CN2:decision list (any class) | 82.18% | 6.9 | 0.019 |
| molecular-biology_promoters | CN2:rule set (best rule) | 80.27% | 9.3 | 0.025 |
| molecular-biology_promoters | CN2:rule set (distributions) | 80.27% | 9.3 | 0.025 |
| molecular-biology_promoters | CN2:rule set (votes) | 81.18% | 9.3 | 0.025 |
| molecular-biology_promoters | PFossil:concept (majority class) | 83.18% | 4.2 | 0.005 |
| molecular-biology_promoters | PFossil:concept (minority class) | 81.27% | 3.2 | 0.004 |
| molecular-biology_promoters | PFossil:decision list (any class) | 80.36% | 3.0 | 0.008 |
| molecular-biology_promoters | PFossil:rule set (best rule) | 82.27% | 7.4 | 0.009 |
| molecular-biology_promoters | PFossil:rule set (distributions) | 81.36% | 7.4 | 0.009 |
| molecular-biology_promoters | PFossil:rule set (votes) | 81.27% | 7.4 | 0.009 |
| molecular-biology_promoters | Pypper:concept (majority class) | 83.18% | 2.3 | 0.014 |
| molecular-biology_promoters | Pypper:concept (minority class) | 83.18% | 2.1 | 0.014 |
| molecular-biology_promoters | Pypper:decision list (any class) | 84.00% | 2.4 | 0.008 |
| molecular-biology_promoters | Pypper:rule set (best rule) | 81.36% | 4.4 | 0.026 |
| molecular-biology_promoters | Pypper:rule set (distributions) | 82.18% | 4.4 | 0.026 |
| molecular-biology_promoters | Pypper:rule set (votes) | 81.27% | 4.4 | 0.026 |
| monks-problems-1 | CN2:concept (majority class) | 80.45% | 1.6 | 0.004 |
| monks-problems-1 | CN2:concept (minority class) | 94.18% | 3.3 | 0.007 |
| monks-problems-1 | CN2:decision list (any class) | 74.64% | 1.0 | 0.006 |
| monks-problems-1 | CN2:rule set (best rule) | 79.18% | 4.9 | 0.011 |
| monks-problems-1 | CN2:rule set (distributions) | 79.18% | 4.9 | 0.011 |
| monks-problems-1 | CN2:rule set (votes) | 79.18% | 4.9 | 0.011 |
| monks-problems-1 | PFossil:concept (majority class) | 81.62% | 2.6 | 0.003 |
| monks-problems-1 | PFossil:concept (minority class) | 77.00% | 1.4 | 0.002 |
| monks-problems-1 | PFossil:decision list (any class) | 82.02% | 3.1 | 0.005 |
| monks-problems-1 | PFossil:rule set (best rule) | 80.41% | 4.0 | 0.005 |
| monks-problems-1 | PFossil:rule set (distributions) | 74.64% | 4.0 | 0.005 |
| monks-problems-1 | PFossil:rule set (votes) | 75.91% | 4.0 | 0.005 |
| monks-problems-1 | Pypper:concept (majority class) | 89.89% | 3.4 | 0.016 |
| monks-problems-1 | Pypper:concept (minority class) | 95.30% | 3.6 | 0.020 |
| monks-problems-1 | Pypper:decision list (any class) | 89.89% | 3.6 | 0.011 |
| monks-problems-1 | Pypper:rule set (best rule) | 94.76% | 7.0 | 0.036 |
| monks-problems-1 | Pypper:rule set (distributions) | 92.06% | 7.0 | 0.036 |
| monks-problems-1 | Pypper:rule set (votes) | 92.06% | 7.0 | 0.036 |
| monks-problems-2 | CN2:concept (majority class) | 40.27% | 1.4 | 0.004 |
| monks-problems-2 | CN2:concept (minority class) | 65.72% | 0.0 | 0.001 |
| monks-problems-2 | CN2:decision list (any class) | 65.89% | 1.4 | 0.008 |
| monks-problems-2 | CN2:rule set (best rule) | 65.72% | 1.4 | 0.006 |
| monks-problems-2 | CN2:rule set (distributions) | 65.72% | 1.4 | 0.006 |
| monks-problems-2 | CN2:rule set (votes) | 65.72% | 1.4 | 0.006 |
| monks-problems-2 | PFossil:concept (majority class) | 34.28% | 0.0 | 0.001 |
| monks-problems-2 | PFossil:concept (minority class) | 66.72% | 0.4 | 0.002 |
| monks-problems-2 | PFossil:decision list (any class) | 65.72% | 0.0 | 0.002 |
| monks-problems-2 | PFossil:rule set (best rule) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | PFossil:rule set (distributions) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | PFossil:rule set (votes) | 66.72% | 0.4 | 0.004 |
| monks-problems-2 | Pypper:concept (majority class) | 76.19% | 10.7 | 0.068 |
| monks-problems-2 | Pypper:concept (minority class) | 70.39% | 2.3 | 0.022 |
| monks-problems-2 | Pypper:decision list (any class) | 67.39% | 4.7 | 0.020 |
| monks-problems-2 | Pypper:rule set (best rule) | 72.03% | 13.0 | 0.090 |
| monks-problems-2 | Pypper:rule set (distributions) | 71.87% | 13.0 | 0.090 |
| monks-problems-2 | Pypper:rule set (votes) | 71.87% | 13.0 | 0.090 |
| monks-problems-3 | CN2:concept (majority class) | 97.84% | 10.5 | 0.029 |
| monks-problems-3 | CN2:concept (minority class) | 98.74% | 6.1 | 0.014 |
| monks-problems-3 | CN2:decision list (any class) | 98.19% | 9.1 | 0.032 |
| monks-problems-3 | CN2:rule set (best rule) | 98.74% | 16.6 | 0.044 |
| monks-problems-3 | CN2:rule set (distributions) | 98.74% | 16.6 | 0.044 |
| monks-problems-3 | CN2:rule set (votes) | 98.74% | 16.6 | 0.044 |
| monks-problems-3 | PFossil:concept (majority class) | 98.56% | 2.9 | 0.004 |
| monks-problems-3 | PFossil:concept (minority class) | 93.33% | 3.9 | 0.005 |
| monks-problems-3 | PFossil:decision list (any class) | 93.69% | 3.7 | 0.008 |
| monks-problems-3 | PFossil:rule set (best rule) | 98.92% | 6.8 | 0.009 |
| monks-problems-3 | PFossil:rule set (distributions) | 96.40% | 6.8 | 0.009 |
| monks-problems-3 | PFossil:rule set (votes) | 98.38% | 6.8 | 0.009 |
| monks-problems-3 | Pypper:concept (majority class) | 98.92% | 2.0 | 0.014 |
| monks-problems-3 | Pypper:concept (minority class) | 98.92% | 3.0 | 0.015 |
| monks-problems-3 | Pypper:decision list (any class) | 98.92% | 3.7 | 0.012 |
| monks-problems-3 | Pypper:rule set (best rule) | 98.92% | 5.0 | 0.029 |
| monks-problems-3 | Pypper:rule set (distributions) | 98.92% | 5.0 | 0.029 |
| monks-problems-3 | Pypper:rule set (votes) | 98.92% | 5.0 | 0.029 |
| mushroom | CN2:concept (majority class) | 100.00% | 14.0 | 0.067 |
| mushroom | CN2:concept (minority class) | 100.00% | 9.0 | 0.048 |
| mushroom | CN2:decision list (any class) | 100.00% | 16.1 | 0.125 |
| mushroom | CN2:rule set (best rule) | 100.00% | 23.0 | 0.120 |
| mushroom | CN2:rule set (distributions) | 100.00% | 23.0 | 0.120 |
| mushroom | CN2:rule set (votes) | 100.00% | 23.0 | 0.120 |
| mushroom | PFossil:concept (majority class) | 99.95% | 5.4 | 0.026 |
| mushroom | PFossil:concept (minority class) | 100.00% | 6.9 | 0.031 |
| mushroom | PFossil:decision list (any class) | 99.95% | 5.4 | 0.061 |
| mushroom | PFossil:rule set (best rule) | 99.95% | 12.3 | 0.067 |
| mushroom | PFossil:rule set (distributions) | 99.95% | 12.3 | 0.067 |
| mushroom | PFossil:rule set (votes) | 99.95% | 12.3 | 0.067 |
| mushroom | Pypper:concept (majority class) | 99.98% | 4.5 | 0.070 |
| mushroom | Pypper:concept (minority class) | 99.96% | 5.2 | 0.076 |
| mushroom | Pypper:decision list (any class) | 99.98% | 5.5 | 0.075 |
| mushroom | Pypper:rule set (best rule) | 99.94% | 9.7 | 0.151 |
| mushroom | Pypper:rule set (distributions) | 99.94% | 9.7 | 0.151 |
| mushroom | Pypper:rule set (votes) | 99.94% | 9.7 | 0.151 |
| optdigits | CN2:cascade (most frequent first) | 89.20% | 83.6 | 1.331 |
| optdigits | CN2:cascade (rarest first) | 85.55% | 69.0 | 0.889 |
| optdigits | CN2:decision list (any class) | 92.74% | 94.3 | 3.047 |
| optdigits | CN2:decision list (each class) | 93.52% | 83.6 | 10.707 |
| optdigits | CN2:decision list (seed's class) | 92.44% | 96.4 | 1.802 |
| optdigits | CN2:decision list (seed) | 92.74% | 83.0 | 1.307 |
| optdigits | CN2:rule set (best rule) | 80.37% | 79.8 | 1.331 |
| optdigits | CN2:rule set (distributions) | 78.90% | 79.8 | 1.331 |
| optdigits | CN2:rule set (votes) | 79.98% | 79.8 | 1.331 |
| optdigits | PFossil:cascade (most frequent first) | 91.25% | 69.8 | 0.363 |
| optdigits | PFossil:cascade (rarest first) | 90.71% | 66.4 | 0.350 |
| optdigits | PFossil:decision list (any class) | 92.44% | 64.0 | 0.786 |
| optdigits | PFossil:decision list (each class) | 92.60% | 50.8 | 1.622 |
| optdigits | PFossil:decision list (seed's class) | 92.40% | 62.6 | 0.349 |
| optdigits | PFossil:decision list (seed) | 92.38% | 53.8 | 0.279 |
| optdigits | PFossil:rule set (best rule) | 90.30% | 81.3 | 0.544 |
| optdigits | PFossil:rule set (distributions) | 90.11% | 81.3 | 0.544 |
| optdigits | PFossil:rule set (votes) | 90.27% | 81.3 | 0.544 |
| optdigits | Pypper:cascade (most frequent first) | 91.74% | 44.7 | 0.862 |
| optdigits | Pypper:cascade (rarest first) | 90.91% | 47.9 | 0.908 |
| optdigits | Pypper:decision list (any class) | 90.41% | 49.0 | 0.721 |
| optdigits | Pypper:decision list (each class) | 91.10% | 47.7 | 2.236 |
| optdigits | Pypper:decision list (seed's class) | 91.51% | 63.3 | 0.473 |
| optdigits | Pypper:decision list (seed) | 91.73% | 62.0 | 0.421 |
| optdigits | Pypper:rule set (best rule) | 90.18% | 62.1 | 1.508 |
| optdigits | Pypper:rule set (distributions) | 90.27% | 62.1 | 1.508 |
| optdigits | Pypper:rule set (votes) | 90.14% | 62.1 | 1.508 |
| ozone-level-8hr | CN2:concept (majority class) | 83.94% | 19.2 | 0.125 |
| ozone-level-8hr | CN2:concept (minority class) | 93.41% | 7.1 | 0.033 |
| ozone-level-8hr | CN2:decision list (any class) | 93.88% | 19.2 | 0.203 |
| ozone-level-8hr | CN2:rule set (best rule) | 93.69% | 26.3 | 0.160 |
| ozone-level-8hr | CN2:rule set (distributions) | 93.69% | 26.3 | 0.160 |
| ozone-level-8hr | CN2:rule set (votes) | 93.73% | 26.3 | 0.160 |
| ozone-level-8hr | PFossil:concept (majority class) | 22.72% | 0.9 | 0.010 |
| ozone-level-8hr | PFossil:concept (minority class) | 93.17% | 2.0 | 0.020 |
| ozone-level-8hr | PFossil:decision list (any class) | 93.25% | 1.5 | 0.047 |
| ozone-level-8hr | PFossil:rule set (best rule) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | PFossil:rule set (distributions) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | PFossil:rule set (votes) | 93.33% | 2.9 | 0.032 |
| ozone-level-8hr | Pypper:concept (majority class) | 82.47% | 3.1 | 0.115 |
| ozone-level-8hr | Pypper:concept (minority class) | 92.94% | 3.0 | 0.060 |
| ozone-level-8hr | Pypper:decision list (any class) | 93.17% | 3.1 | 0.056 |
| ozone-level-8hr | Pypper:rule set (best rule) | 93.33% | 6.1 | 0.173 |
| ozone-level-8hr | Pypper:rule set (distributions) | 93.33% | 6.1 | 0.173 |
| ozone-level-8hr | Pypper:rule set (votes) | 93.37% | 6.1 | 0.173 |
| pc1 | CN2:concept (majority class) | 62.49% | 3.8 | 0.081 |
| pc1 | CN2:concept (minority class) | 92.78% | 3.5 | 0.010 |
| pc1 | CN2:decision list (any class) | 92.97% | 4.6 | 0.166 |
| pc1 | CN2:rule set (best rule) | 92.97% | 7.3 | 0.092 |
| pc1 | CN2:rule set (distributions) | 92.97% | 7.3 | 0.092 |
| pc1 | CN2:rule set (votes) | 92.97% | 7.3 | 0.092 |
| pc1 | PFossil:concept (majority class) | 92.79% | 4.3 | 0.007 |
| pc1 | PFossil:concept (minority class) | 90.80% | 2.9 | 0.008 |
| pc1 | PFossil:decision list (any class) | 90.80% | 2.9 | 0.016 |
| pc1 | PFossil:rule set (best rule) | 92.88% | 7.2 | 0.016 |
| pc1 | PFossil:rule set (distributions) | 92.97% | 7.2 | 0.016 |
| pc1 | PFossil:rule set (votes) | 92.88% | 7.2 | 0.016 |
| pc1 | Pypper:concept (majority class) | 77.99% | 2.5 | 0.049 |
| pc1 | Pypper:concept (minority class) | 92.70% | 1.6 | 0.017 |
| pc1 | Pypper:decision list (any class) | 92.52% | 2.2 | 0.016 |
| pc1 | Pypper:rule set (best rule) | 92.61% | 4.1 | 0.067 |
| pc1 | Pypper:rule set (distributions) | 92.61% | 4.1 | 0.067 |
| pc1 | Pypper:rule set (votes) | 92.61% | 4.1 | 0.067 |
| pc3 | CN2:concept (majority class) | 67.37% | 12.2 | 0.141 |
| pc3 | CN2:concept (minority class) | 87.33% | 7.3 | 0.033 |
| pc3 | CN2:decision list (any class) | 89.76% | 12.1 | 0.240 |
| pc3 | CN2:rule set (best rule) | 87.97% | 19.5 | 0.175 |
| pc3 | CN2:rule set (distributions) | 87.97% | 19.5 | 0.175 |
| pc3 | CN2:rule set (votes) | 87.97% | 19.5 | 0.175 |
| pc3 | PFossil:concept (majority class) | 77.60% | 1.0 | 0.005 |
| pc3 | PFossil:concept (minority class) | 85.09% | 1.1 | 0.008 |
| pc3 | PFossil:decision list (any class) | 85.09% | 1.1 | 0.015 |
| pc3 | PFossil:rule set (best rule) | 85.09% | 2.1 | 0.014 |
| pc3 | PFossil:rule set (distributions) | 85.09% | 2.1 | 0.014 |
| pc3 | PFossil:rule set (votes) | 85.09% | 2.1 | 0.014 |
| pc3 | Pypper:concept (majority class) | 75.55% | 2.3 | 0.092 |
| pc3 | Pypper:concept (minority class) | 88.87% | 1.8 | 0.032 |
| pc3 | Pypper:decision list (any class) | 89.19% | 2.6 | 0.025 |
| pc3 | Pypper:rule set (best rule) | 88.93% | 4.1 | 0.122 |
| pc3 | Pypper:rule set (distributions) | 88.93% | 4.1 | 0.122 |
| pc3 | Pypper:rule set (votes) | 88.93% | 4.1 | 0.122 |
| pc4 | CN2:concept (majority class) | 83.40% | 13.0 | 0.045 |
| pc4 | CN2:concept (minority class) | 88.27% | 7.6 | 0.031 |
| pc4 | CN2:decision list (any class) | 89.37% | 17.6 | 0.094 |
| pc4 | CN2:rule set (best rule) | 89.16% | 20.6 | 0.077 |
| pc4 | CN2:rule set (distributions) | 89.16% | 20.6 | 0.077 |
| pc4 | CN2:rule set (votes) | 89.23% | 20.6 | 0.077 |
| pc4 | PFossil:concept (majority class) | 89.78% | 8.0 | 0.015 |
| pc4 | PFossil:concept (minority class) | 86.01% | 5.8 | 0.020 |
| pc4 | PFossil:decision list (any class) | 85.87% | 3.6 | 0.031 |
| pc4 | PFossil:rule set (best rule) | 89.51% | 13.8 | 0.036 |
| pc4 | PFossil:rule set (distributions) | 89.58% | 13.8 | 0.036 |
| pc4 | PFossil:rule set (votes) | 89.78% | 13.8 | 0.036 |
| pc4 | Pypper:concept (majority class) | 84.29% | 4.0 | 0.058 |
| pc4 | Pypper:concept (minority class) | 87.45% | 3.6 | 0.053 |
| pc4 | Pypper:decision list (any class) | 88.13% | 4.2 | 0.036 |
| pc4 | Pypper:rule set (best rule) | 87.79% | 7.6 | 0.113 |
| pc4 | Pypper:rule set (distributions) | 87.79% | 7.6 | 0.113 |
| pc4 | Pypper:rule set (votes) | 87.79% | 7.6 | 0.113 |
| phoneme | CN2:concept (majority class) | 77.44% | 43.6 | 0.393 |
| phoneme | CN2:concept (minority class) | 71.69% | 58.7 | 0.455 |
| phoneme | CN2:decision list (any class) | 78.29% | 53.2 | 1.093 |
| phoneme | CN2:rule set (best rule) | 81.48% | 102.3 | 0.851 |
| phoneme | CN2:rule set (distributions) | 78.48% | 102.3 | 0.851 |
| phoneme | CN2:rule set (votes) | 79.55% | 102.3 | 0.851 |
| phoneme | PFossil:concept (majority class) | 73.56% | 1.0 | 0.006 |
| phoneme | PFossil:concept (minority class) | 77.11% | 3.4 | 0.015 |
| phoneme | PFossil:decision list (any class) | 77.92% | 2.9 | 0.026 |
| phoneme | PFossil:rule set (best rule) | 78.35% | 4.4 | 0.027 |
| phoneme | PFossil:rule set (distributions) | 78.35% | 4.4 | 0.027 |
| phoneme | PFossil:rule set (votes) | 78.46% | 4.4 | 0.027 |
| phoneme | Pypper:concept (majority class) | 82.66% | 14.6 | 0.480 |
| phoneme | Pypper:concept (minority class) | 82.90% | 12.3 | 0.406 |
| phoneme | Pypper:decision list (any class) | 82.85% | 26.4 | 0.303 |
| phoneme | Pypper:rule set (best rule) | 83.53% | 26.9 | 0.889 |
| phoneme | Pypper:rule set (distributions) | 83.24% | 26.9 | 0.889 |
| phoneme | Pypper:rule set (votes) | 83.40% | 26.9 | 0.889 |
| primary-tumor | CN2:cascade (most frequent first) | 32.46% | 15.2 | 0.110 |
| primary-tumor | CN2:cascade (rarest first) | 41.32% | 13.6 | 0.107 |
| primary-tumor | CN2:decision list (any class) | 35.41% | 8.9 | 0.197 |
| primary-tumor | CN2:decision list (each class) | 38.65% | 17.2 | 0.757 |
| primary-tumor | CN2:decision list (seed's class) | 37.77% | 19.1 | 0.123 |
| primary-tumor | CN2:decision list (seed) | 38.66% | 18.0 | 0.111 |
| primary-tumor | CN2:rule set (best rule) | 42.76% | 18.1 | 0.148 |
| primary-tumor | CN2:rule set (distributions) | 43.35% | 18.1 | 0.148 |
| primary-tumor | CN2:rule set (votes) | 41.89% | 18.1 | 0.148 |
| primary-tumor | PFossil:cascade (most frequent first) | 36.00% | 55.3 | 0.063 |
| primary-tumor | PFossil:cascade (rarest first) | 32.44% | 70.4 | 0.097 |
| primary-tumor | PFossil:decision list (any class) | 37.76% | 49.1 | 0.125 |
| primary-tumor | PFossil:decision list (each class) | 38.34% | 58.3 | 0.589 |
| primary-tumor | PFossil:decision list (seed's class) | 36.58% | 51.4 | 0.051 |
| primary-tumor | PFossil:decision list (seed) | 38.05% | 35.8 | 0.043 |
| primary-tumor | PFossil:rule set (best rule) | 43.38% | 66.3 | 0.105 |
| primary-tumor | PFossil:rule set (distributions) | 41.01% | 66.3 | 0.105 |
| primary-tumor | PFossil:rule set (votes) | 41.01% | 66.3 | 0.105 |
| primary-tumor | Pypper:cascade (most frequent first) | 35.70% | 10.0 | 0.104 |
| primary-tumor | Pypper:cascade (rarest first) | 39.25% | 7.3 | 0.092 |
| primary-tumor | Pypper:decision list (any class) | 35.10% | 6.8 | 0.042 |
| primary-tumor | Pypper:decision list (each class) | 35.42% | 4.1 | 0.156 |
| primary-tumor | Pypper:decision list (seed's class) | 38.94% | 20.1 | 0.160 |
| primary-tumor | Pypper:decision list (seed) | 41.60% | 22.3 | 0.186 |
| primary-tumor | Pypper:rule set (best rule) | 38.65% | 7.3 | 0.090 |
| primary-tumor | Pypper:rule set (distributions) | 38.95% | 7.3 | 0.090 |
| primary-tumor | Pypper:rule set (votes) | 37.77% | 7.3 | 0.090 |
| qsar-biodeg | CN2:concept (majority class) | 75.55% | 16.7 | 0.053 |
| qsar-biodeg | CN2:concept (minority class) | 71.85% | 4.9 | 0.019 |
| qsar-biodeg | CN2:decision list (any class) | 74.89% | 21.6 | 0.109 |
| qsar-biodeg | CN2:rule set (best rule) | 72.14% | 21.6 | 0.072 |
| qsar-biodeg | CN2:rule set (distributions) | 72.14% | 21.6 | 0.072 |
| qsar-biodeg | CN2:rule set (votes) | 72.04% | 21.6 | 0.072 |
| qsar-biodeg | PFossil:concept (majority class) | 80.77% | 8.4 | 0.021 |
| qsar-biodeg | PFossil:concept (minority class) | 85.22% | 4.3 | 0.018 |
| qsar-biodeg | PFossil:decision list (any class) | 82.38% | 8.8 | 0.041 |
| qsar-biodeg | PFossil:rule set (best rule) | 84.65% | 12.7 | 0.041 |
| qsar-biodeg | PFossil:rule set (distributions) | 84.17% | 12.7 | 0.041 |
| qsar-biodeg | PFossil:rule set (votes) | 83.60% | 12.7 | 0.041 |
| qsar-biodeg | Pypper:concept (majority class) | 82.08% | 7.5 | 0.086 |
| qsar-biodeg | Pypper:concept (minority class) | 83.70% | 5.1 | 0.065 |
| qsar-biodeg | Pypper:decision list (any class) | 83.99% | 6.1 | 0.040 |
| qsar-biodeg | Pypper:rule set (best rule) | 85.03% | 12.6 | 0.151 |
| qsar-biodeg | Pypper:rule set (distributions) | 84.36% | 12.6 | 0.151 |
| qsar-biodeg | Pypper:rule set (votes) | 84.65% | 12.6 | 0.151 |
| segment | CN2:cascade (most frequent first) | 88.35% | 19.7 | 0.175 |
| segment | CN2:cascade (rarest first) | 88.35% | 19.7 | 0.176 |
| segment | CN2:decision list (any class) | 94.68% | 27.4 | 1.031 |
| segment | CN2:decision list (each class) | 95.02% | 30.8 | 1.875 |
| segment | CN2:decision list (seed's class) | 94.94% | 33.2 | 0.574 |
| segment | CN2:decision list (seed) | 95.28% | 29.2 | 0.524 |
| segment | CN2:rule set (best rule) | 85.50% | 39.0 | 0.599 |
| segment | CN2:rule set (distributions) | 85.45% | 39.0 | 0.599 |
| segment | CN2:rule set (votes) | 84.55% | 39.0 | 0.599 |
| segment | PFossil:cascade (most frequent first) | 94.85% | 21.4 | 0.053 |
| segment | PFossil:cascade (rarest first) | 94.85% | 21.4 | 0.053 |
| segment | PFossil:decision list (any class) | 95.11% | 16.5 | 0.067 |
| segment | PFossil:decision list (each class) | 94.72% | 23.1 | 0.133 |
| segment | PFossil:decision list (seed's class) | 93.33% | 18.4 | 0.038 |
| segment | PFossil:decision list (seed) | 93.46% | 15.8 | 0.032 |
| segment | PFossil:rule set (best rule) | 93.33% | 30.5 | 0.084 |
| segment | PFossil:rule set (distributions) | 93.12% | 30.5 | 0.084 |
| segment | PFossil:rule set (votes) | 92.73% | 30.5 | 0.084 |
| segment | Pypper:cascade (most frequent first) | 94.72% | 14.5 | 0.143 |
| segment | Pypper:cascade (rarest first) | 94.72% | 14.5 | 0.143 |
| segment | Pypper:decision list (any class) | 94.89% | 18.6 | 0.097 |
| segment | Pypper:decision list (each class) | 94.59% | 16.3 | 0.225 |
| segment | Pypper:decision list (seed's class) | 93.94% | 27.3 | 0.100 |
| segment | Pypper:decision list (seed) | 94.03% | 24.4 | 0.104 |
| segment | Pypper:rule set (best rule) | 93.46% | 21.9 | 0.254 |
| segment | Pypper:rule set (distributions) | 93.20% | 21.9 | 0.254 |
| segment | Pypper:rule set (votes) | 93.29% | 21.9 | 0.254 |
| sick | CN2:concept (majority class) | 97.40% | 18.4 | 0.162 |
| sick | CN2:concept (minority class) | 97.32% | 35.0 | 0.206 |
| sick | CN2:decision list (any class) | 97.03% | 21.8 | 0.351 |
| sick | CN2:rule set (best rule) | 97.72% | 53.4 | 0.370 |
| sick | CN2:rule set (distributions) | 97.72% | 53.4 | 0.370 |
| sick | CN2:rule set (votes) | 97.72% | 53.4 | 0.370 |
| sick | PFossil:concept (majority class) | 98.25% | 10.2 | 0.028 |
| sick | PFossil:concept (minority class) | 97.67% | 6.2 | 0.022 |
| sick | PFossil:decision list (any class) | 97.67% | 6.2 | 0.042 |
| sick | PFossil:rule set (best rule) | 98.30% | 16.4 | 0.054 |
| sick | PFossil:rule set (distributions) | 98.20% | 16.4 | 0.054 |
| sick | PFossil:rule set (votes) | 98.20% | 16.4 | 0.054 |
| sick | Pypper:concept (majority class) | 98.01% | 5.4 | 0.079 |
| sick | Pypper:concept (minority class) | 98.22% | 3.5 | 0.052 |
| sick | Pypper:decision list (any class) | 98.22% | 4.8 | 0.045 |
| sick | Pypper:rule set (best rule) | 98.20% | 8.9 | 0.136 |
| sick | Pypper:rule set (distributions) | 98.22% | 8.9 | 0.136 |
| sick | Pypper:rule set (votes) | 98.22% | 8.9 | 0.136 |
| solar-flare | CN2:cascade (most frequent first) | 68.01% | 27.3 | 0.293 |
| solar-flare | CN2:cascade (rarest first) | 70.54% | 30.5 | 0.304 |
| solar-flare | CN2:decision list (any class) | 72.61% | 8.2 | 0.231 |
| solar-flare | CN2:decision list (each class) | 72.98% | 19.9 | 0.728 |
| solar-flare | CN2:decision list (seed's class) | 68.20% | 28.7 | 0.359 |
| solar-flare | CN2:decision list (seed) | 69.71% | 19.3 | 0.262 |
| solar-flare | CN2:rule set (best rule) | 68.95% | 39.7 | 0.401 |
| solar-flare | CN2:rule set (distributions) | 69.13% | 39.7 | 0.401 |
| solar-flare | CN2:rule set (votes) | 67.82% | 39.7 | 0.401 |
| solar-flare | PFossil:cascade (most frequent first) | 67.92% | 11.5 | 0.018 |
| solar-flare | PFossil:cascade (rarest first) | 70.64% | 12.6 | 0.027 |
| solar-flare | PFossil:decision list (any class) | 72.99% | 10.1 | 0.027 |
| solar-flare | PFossil:decision list (each class) | 72.24% | 15.0 | 0.071 |
| solar-flare | PFossil:decision list (seed's class) | 69.60% | 10.6 | 0.014 |
| solar-flare | PFossil:decision list (seed) | 69.51% | 7.8 | 0.012 |
| solar-flare | PFossil:rule set (best rule) | 74.02% | 14.5 | 0.034 |
| solar-flare | PFossil:rule set (distributions) | 74.30% | 14.5 | 0.034 |
| solar-flare | PFossil:rule set (votes) | 70.07% | 14.5 | 0.034 |
| solar-flare | Pypper:cascade (most frequent first) | 72.24% | 8.2 | 0.085 |
| solar-flare | Pypper:cascade (rarest first) | 74.77% | 4.9 | 0.064 |
| solar-flare | Pypper:decision list (any class) | 73.83% | 6.8 | 0.037 |
| solar-flare | Pypper:decision list (each class) | 70.93% | 3.8 | 0.054 |
| solar-flare | Pypper:decision list (seed's class) | 72.52% | 19.6 | 0.079 |
| solar-flare | Pypper:decision list (seed) | 74.58% | 19.3 | 0.142 |
| solar-flare | Pypper:rule set (best rule) | 69.43% | 7.6 | 0.103 |
| solar-flare | Pypper:rule set (distributions) | 69.24% | 7.6 | 0.103 |
| solar-flare | Pypper:rule set (votes) | 69.14% | 7.6 | 0.103 |
| sonar | CN2:concept (majority class) | 72.62% | 8.7 | 0.021 |
| sonar | CN2:concept (minority class) | 63.88% | 10.1 | 0.027 |
| sonar | CN2:decision list (any class) | 71.62% | 14.1 | 0.050 |
| sonar | CN2:rule set (best rule) | 68.69% | 18.8 | 0.048 |
| sonar | CN2:rule set (distributions) | 69.17% | 18.8 | 0.048 |
| sonar | CN2:rule set (votes) | 69.17% | 18.8 | 0.048 |
| sonar | PFossil:concept (majority class) | 71.12% | 8.4 | 0.014 |
| sonar | PFossil:concept (minority class) | 72.64% | 8.3 | 0.013 |
| sonar | PFossil:decision list (any class) | 73.52% | 6.0 | 0.021 |
| sonar | PFossil:rule set (best rule) | 73.55% | 16.7 | 0.027 |
| sonar | PFossil:rule set (distributions) | 74.02% | 16.7 | 0.027 |
| sonar | PFossil:rule set (votes) | 75.52% | 16.7 | 0.027 |
| sonar | Pypper:concept (majority class) | 70.26% | 3.2 | 0.030 |
| sonar | Pypper:concept (minority class) | 77.43% | 2.8 | 0.028 |
| sonar | Pypper:decision list (any class) | 77.90% | 3.6 | 0.014 |
| sonar | Pypper:rule set (best rule) | 76.48% | 6.0 | 0.057 |
| sonar | Pypper:rule set (distributions) | 75.50% | 6.0 | 0.057 |
| sonar | Pypper:rule set (votes) | 73.14% | 6.0 | 0.057 |
| soybean | CN2:cascade (most frequent first) | 91.36% | 26.7 | 0.170 |
| soybean | CN2:cascade (rarest first) | 89.90% | 27.2 | 0.178 |
| soybean | CN2:decision list (any class) | 88.88% | 23.0 | 0.463 |
| soybean | CN2:decision list (each class) | 91.23% | 25.5 | 2.285 |
| soybean | CN2:decision list (seed's class) | 91.22% | 26.6 | 0.194 |
| soybean | CN2:decision list (seed) | 92.40% | 24.8 | 0.184 |
| soybean | CN2:rule set (best rule) | 92.54% | 33.7 | 0.262 |
| soybean | CN2:rule set (distributions) | 91.81% | 33.7 | 0.262 |
| soybean | CN2:rule set (votes) | 91.52% | 33.7 | 0.262 |
| soybean | PFossil:cascade (most frequent first) | 90.48% | 31.1 | 0.039 |
| soybean | PFossil:cascade (rarest first) | 92.98% | 32.5 | 0.042 |
| soybean | PFossil:decision list (any class) | 93.27% | 35.8 | 0.082 |
| soybean | PFossil:decision list (each class) | 94.29% | 22.8 | 0.164 |
| soybean | PFossil:decision list (seed's class) | 92.69% | 26.7 | 0.029 |
| soybean | PFossil:decision list (seed) | 92.68% | 25.1 | 0.027 |
| soybean | PFossil:rule set (best rule) | 92.24% | 42.1 | 0.063 |
| soybean | PFossil:rule set (distributions) | 91.66% | 42.1 | 0.063 |
| soybean | PFossil:rule set (votes) | 90.34% | 42.1 | 0.063 |
| soybean | Pypper:cascade (most frequent first) | 87.43% | 26.0 | 0.144 |
| soybean | Pypper:cascade (rarest first) | 90.93% | 23.9 | 0.143 |
| soybean | Pypper:decision list (any class) | 89.17% | 25.8 | 0.100 |
| soybean | Pypper:decision list (each class) | 89.61% | 27.2 | 0.479 |
| soybean | Pypper:decision list (seed's class) | 87.58% | 30.1 | 0.069 |
| soybean | Pypper:decision list (seed) | 88.42% | 28.6 | 0.069 |
| soybean | Pypper:rule set (best rule) | 89.91% | 26.0 | 0.183 |
| soybean | Pypper:rule set (distributions) | 87.28% | 26.0 | 0.183 |
| soybean | Pypper:rule set (votes) | 86.84% | 26.0 | 0.183 |
| spambase | CN2:concept (majority class) | 89.05% | 45.6 | 0.454 |
| spambase | CN2:concept (minority class) | 88.63% | 42.5 | 0.295 |
| spambase | CN2:decision list (any class) | 89.76% | 68.1 | 1.074 |
| spambase | CN2:rule set (best rule) | 89.33% | 88.1 | 0.748 |
| spambase | CN2:rule set (distributions) | 89.05% | 88.1 | 0.748 |
| spambase | CN2:rule set (votes) | 89.24% | 88.1 | 0.748 |
| spambase | PFossil:concept (majority class) | 92.05% | 6.0 | 0.037 |
| spambase | PFossil:concept (minority class) | 92.72% | 6.1 | 0.052 |
| spambase | PFossil:decision list (any class) | 92.55% | 5.4 | 0.123 |
| spambase | PFossil:rule set (best rule) | 92.81% | 12.1 | 0.095 |
| spambase | PFossil:rule set (distributions) | 91.81% | 12.1 | 0.095 |
| spambase | PFossil:rule set (votes) | 92.41% | 12.1 | 0.095 |
| spambase | Pypper:concept (majority class) | 92.11% | 15.3 | 0.433 |
| spambase | Pypper:concept (minority class) | 92.70% | 10.4 | 0.279 |
| spambase | Pypper:decision list (any class) | 92.35% | 15.9 | 0.243 |
| spambase | Pypper:rule set (best rule) | 92.96% | 25.7 | 0.711 |
| spambase | Pypper:rule set (distributions) | 92.76% | 25.7 | 0.711 |
| spambase | Pypper:rule set (votes) | 92.68% | 25.7 | 0.711 |
| splice | CN2:cascade (most frequent first) | 89.66% | 25.5 | 0.171 |
| splice | CN2:cascade (rarest first) | 74.80% | 11.6 | 0.144 |
| splice | CN2:decision list (any class) | 91.13% | 67.4 | 0.724 |
| splice | CN2:decision list (each class) | 91.07% | 60.4 | 0.922 |
| splice | CN2:decision list (seed's class) | 91.16% | 64.9 | 0.467 |
| splice | CN2:decision list (seed) | 90.88% | 57.7 | 0.449 |
| splice | CN2:rule set (best rule) | 65.49% | 21.2 | 0.206 |
| splice | CN2:rule set (distributions) | 65.27% | 21.2 | 0.206 |
| splice | CN2:rule set (votes) | 65.42% | 21.2 | 0.206 |
| splice | PFossil:cascade (most frequent first) | 89.50% | 19.0 | 0.064 |
| splice | PFossil:cascade (rarest first) | 94.33% | 8.8 | 0.037 |
| splice | PFossil:decision list (any class) | 89.66% | 13.7 | 0.075 |
| splice | PFossil:decision list (each class) | 93.26% | 12.3 | 0.076 |
| splice | PFossil:decision list (seed's class) | 88.53% | 12.8 | 0.041 |
| splice | PFossil:decision list (seed) | 89.31% | 10.6 | 0.033 |
| splice | PFossil:rule set (best rule) | 94.23% | 23.7 | 0.093 |
| splice | PFossil:rule set (distributions) | 90.38% | 23.7 | 0.093 |
| splice | PFossil:rule set (votes) | 89.09% | 23.7 | 0.093 |
| splice | Pypper:cascade (most frequent first) | 93.01% | 12.6 | 0.154 |
| splice | Pypper:cascade (rarest first) | 93.82% | 11.4 | 0.198 |
| splice | Pypper:decision list (any class) | 93.86% | 15.6 | 0.119 |
| splice | Pypper:decision list (each class) | 93.20% | 14.8 | 0.172 |
| splice | Pypper:decision list (seed's class) | 93.26% | 23.7 | 0.108 |
| splice | Pypper:decision list (seed) | 92.54% | 24.5 | 0.109 |
| splice | Pypper:rule set (best rule) | 93.45% | 21.1 | 0.348 |
| splice | Pypper:rule set (distributions) | 93.42% | 21.1 | 0.348 |
| splice | Pypper:rule set (votes) | 93.51% | 21.1 | 0.348 |
| texture | CN2:cascade (most frequent first) | 79.04% | 63.9 | 2.918 |
| texture | CN2:cascade (rarest first) | 79.04% | 63.9 | 2.907 |
| texture | CN2:decision list (any class) | 94.00% | 79.9 | 13.215 |
| texture | CN2:decision list (each class) | 93.51% | 77.8 | 46.791 |
| texture | CN2:decision list (seed's class) | 94.04% | 84.5 | 7.421 |
| texture | CN2:decision list (seed) | 93.82% | 77.5 | 6.476 |
| texture | CN2:rule set (best rule) | 89.98% | 93.4 | 4.124 |
| texture | CN2:rule set (distributions) | 86.02% | 93.4 | 4.124 |
| texture | CN2:rule set (votes) | 86.38% | 93.4 | 4.124 |
| texture | PFossil:cascade (most frequent first) | 90.45% | 52.1 | 0.237 |
| texture | PFossil:cascade (rarest first) | 90.45% | 52.1 | 0.239 |
| texture | PFossil:decision list (any class) | 91.04% | 48.9 | 0.436 |
| texture | PFossil:decision list (each class) | 90.29% | 41.9 | 0.748 |
| texture | PFossil:decision list (seed's class) | 90.25% | 41.1 | 0.175 |
| texture | PFossil:decision list (seed) | 89.44% | 36.4 | 0.152 |
| texture | PFossil:rule set (best rule) | 92.22% | 77.9 | 0.401 |
| texture | PFossil:rule set (distributions) | 90.29% | 77.9 | 0.401 |
| texture | PFossil:rule set (votes) | 91.65% | 77.9 | 0.401 |
| texture | Pypper:cascade (most frequent first) | 91.56% | 45.6 | 0.749 |
| texture | Pypper:cascade (rarest first) | 91.56% | 45.6 | 0.752 |
| texture | Pypper:decision list (any class) | 91.16% | 51.0 | 0.602 |
| texture | Pypper:decision list (each class) | 91.38% | 43.5 | 1.578 |
| texture | Pypper:decision list (seed's class) | 93.07% | 67.4 | 0.406 |
| texture | Pypper:decision list (seed) | 92.27% | 65.7 | 0.397 |
| texture | Pypper:rule set (best rule) | 91.78% | 66.7 | 1.282 |
| texture | Pypper:rule set (distributions) | 91.49% | 66.7 | 1.282 |
| texture | Pypper:rule set (votes) | 91.71% | 66.7 | 1.282 |
| tic-tac-toe | CN2:concept (majority class) | 70.56% | 3.4 | 0.009 |
| tic-tac-toe | CN2:concept (minority class) | 98.33% | 8.2 | 0.021 |
| tic-tac-toe | CN2:decision list (any class) | 77.03% | 7.9 | 0.042 |
| tic-tac-toe | CN2:rule set (best rule) | 98.33% | 11.6 | 0.031 |
| tic-tac-toe | CN2:rule set (distributions) | 98.33% | 11.6 | 0.031 |
| tic-tac-toe | CN2:rule set (votes) | 98.33% | 11.6 | 0.031 |
| tic-tac-toe | PFossil:concept (majority class) | 83.93% | 7.2 | 0.010 |
| tic-tac-toe | PFossil:concept (minority class) | 79.85% | 7.5 | 0.012 |
| tic-tac-toe | PFossil:decision list (any class) | 76.20% | 3.3 | 0.009 |
| tic-tac-toe | PFossil:rule set (best rule) | 97.71% | 14.7 | 0.022 |
| tic-tac-toe | PFossil:rule set (distributions) | 83.72% | 14.7 | 0.022 |
| tic-tac-toe | PFossil:rule set (votes) | 85.29% | 14.7 | 0.022 |
| tic-tac-toe | Pypper:concept (majority class) | 99.69% | 8.0 | 0.048 |
| tic-tac-toe | Pypper:concept (minority class) | 96.25% | 7.5 | 0.044 |
| tic-tac-toe | Pypper:decision list (any class) | 96.25% | 7.8 | 0.029 |
| tic-tac-toe | Pypper:rule set (best rule) | 96.25% | 15.5 | 0.091 |
| tic-tac-toe | Pypper:rule set (distributions) | 96.25% | 15.5 | 0.091 |
| tic-tac-toe | Pypper:rule set (votes) | 96.25% | 15.5 | 0.091 |
| vehicle | CN2:cascade (most frequent first) | 68.08% | 21.1 | 0.126 |
| vehicle | CN2:cascade (rarest first) | 66.90% | 26.8 | 0.241 |
| vehicle | CN2:decision list (any class) | 65.37% | 36.4 | 0.787 |
| vehicle | CN2:decision list (each class) | 70.34% | 39.7 | 1.066 |
| vehicle | CN2:decision list (seed's class) | 69.75% | 52.6 | 0.596 |
| vehicle | CN2:decision list (seed) | 70.21% | 43.9 | 0.910 |
| vehicle | CN2:rule set (best rule) | 54.73% | 27.0 | 0.259 |
| vehicle | CN2:rule set (distributions) | 54.73% | 27.0 | 0.259 |
| vehicle | CN2:rule set (votes) | 54.85% | 27.0 | 0.259 |
| vehicle | PFossil:cascade (most frequent first) | 70.10% | 20.6 | 0.035 |
| vehicle | PFossil:cascade (rarest first) | 70.22% | 22.4 | 0.037 |
| vehicle | PFossil:decision list (any class) | 66.66% | 8.2 | 0.028 |
| vehicle | PFossil:decision list (each class) | 67.73% | 13.5 | 0.061 |
| vehicle | PFossil:decision list (seed's class) | 69.75% | 16.7 | 0.044 |
| vehicle | PFossil:decision list (seed) | 65.13% | 13.4 | 0.024 |
| vehicle | PFossil:rule set (best rule) | 71.64% | 26.1 | 0.054 |
| vehicle | PFossil:rule set (distributions) | 70.93% | 26.1 | 0.054 |
| vehicle | PFossil:rule set (votes) | 71.53% | 26.1 | 0.054 |
| vehicle | Pypper:cascade (most frequent first) | 70.46% | 9.8 | 0.088 |
| vehicle | Pypper:cascade (rarest first) | 68.21% | 11.8 | 0.098 |
| vehicle | Pypper:decision list (any class) | 68.81% | 14.8 | 0.060 |
| vehicle | Pypper:decision list (each class) | 69.76% | 13.6 | 0.122 |
| vehicle | Pypper:decision list (seed's class) | 70.36% | 25.7 | 0.079 |
| vehicle | Pypper:decision list (seed) | 68.33% | 24.8 | 0.076 |
| vehicle | Pypper:rule set (best rule) | 63.84% | 13.1 | 0.143 |
| vehicle | Pypper:rule set (distributions) | 62.30% | 13.1 | 0.143 |
| vehicle | Pypper:rule set (votes) | 63.72% | 13.1 | 0.143 |
| vote | CN2:concept (majority class) | 94.02% | 6.0 | 0.010 |
| vote | CN2:concept (minority class) | 94.26% | 8.0 | 0.022 |
| vote | CN2:decision list (any class) | 93.11% | 5.7 | 0.024 |
| vote | CN2:rule set (best rule) | 94.50% | 14.0 | 0.031 |
| vote | CN2:rule set (distributions) | 94.96% | 14.0 | 0.031 |
| vote | CN2:rule set (votes) | 94.95% | 14.0 | 0.031 |
| vote | PFossil:concept (majority class) | 93.56% | 6.8 | 0.007 |
| vote | PFossil:concept (minority class) | 93.81% | 3.4 | 0.004 |
| vote | PFossil:decision list (any class) | 95.42% | 4.7 | 0.007 |
| vote | PFossil:rule set (best rule) | 95.18% | 10.2 | 0.012 |
| vote | PFossil:rule set (distributions) | 95.42% | 10.2 | 0.012 |
| vote | PFossil:rule set (votes) | 94.03% | 10.2 | 0.012 |
| vote | Pypper:concept (majority class) | 94.25% | 2.2 | 0.013 |
| vote | Pypper:concept (minority class) | 95.18% | 1.7 | 0.015 |
| vote | Pypper:decision list (any class) | 94.02% | 5.7 | 0.015 |
| vote | Pypper:rule set (best rule) | 95.18% | 3.9 | 0.028 |
| vote | Pypper:rule set (distributions) | 95.18% | 3.9 | 0.028 |
| vote | Pypper:rule set (votes) | 94.95% | 3.9 | 0.028 |
| vowel | CN2:cascade (most frequent first) | 73.33% | 51.4 | 0.280 |
| vowel | CN2:cascade (rarest first) | 73.33% | 51.4 | 0.281 |
| vowel | CN2:decision list (any class) | 78.18% | 57.8 | 1.336 |
| vowel | CN2:decision list (each class) | 80.30% | 53.0 | 3.988 |
| vowel | CN2:decision list (seed's class) | 77.98% | 60.4 | 0.593 |
| vowel | CN2:decision list (seed) | 79.19% | 51.0 | 0.623 |
| vowel | CN2:rule set (best rule) | 65.05% | 68.3 | 0.440 |
| vowel | CN2:rule set (distributions) | 65.05% | 68.3 | 0.440 |
| vowel | CN2:rule set (votes) | 65.76% | 68.3 | 0.440 |
| vowel | PFossil:cascade (most frequent first) | 80.40% | 69.7 | 0.101 |
| vowel | PFossil:cascade (rarest first) | 80.40% | 69.7 | 0.100 |
| vowel | PFossil:decision list (any class) | 75.96% | 57.4 | 0.170 |
| vowel | PFossil:decision list (each class) | 75.25% | 46.2 | 0.443 |
| vowel | PFossil:decision list (seed's class) | 76.06% | 52.7 | 0.081 |
| vowel | PFossil:decision list (seed) | 71.41% | 42.4 | 0.072 |
| vowel | PFossil:rule set (best rule) | 77.27% | 94.9 | 0.164 |
| vowel | PFossil:rule set (distributions) | 74.24% | 94.9 | 0.164 |
| vowel | PFossil:rule set (votes) | 78.89% | 94.9 | 0.164 |
| vowel | Pypper:cascade (most frequent first) | 69.70% | 33.1 | 0.251 |
| vowel | Pypper:cascade (rarest first) | 69.70% | 33.1 | 0.250 |
| vowel | Pypper:decision list (any class) | 55.25% | 23.4 | 0.122 |
| vowel | Pypper:decision list (each class) | 52.02% | 22.7 | 0.531 |
| vowel | Pypper:decision list (seed's class) | 69.60% | 54.9 | 0.154 |
| vowel | Pypper:decision list (seed) | 71.11% | 50.4 | 0.160 |
| vowel | Pypper:rule set (best rule) | 63.84% | 41.5 | 0.354 |
| vowel | Pypper:rule set (distributions) | 62.53% | 41.5 | 0.354 |
| vowel | Pypper:rule set (votes) | 62.53% | 41.5 | 0.354 |
| wdbc | CN2:concept (majority class) | 94.54% | 8.8 | 0.029 |
| wdbc | CN2:concept (minority class) | 93.67% | 4.8 | 0.016 |
| wdbc | CN2:decision list (any class) | 94.55% | 9.9 | 0.040 |
| wdbc | CN2:rule set (best rule) | 94.55% | 13.6 | 0.044 |
| wdbc | CN2:rule set (distributions) | 94.55% | 13.6 | 0.044 |
| wdbc | CN2:rule set (votes) | 94.38% | 13.6 | 0.044 |
| wdbc | PFossil:concept (majority class) | 94.02% | 4.1 | 0.006 |
| wdbc | PFossil:concept (minority class) | 94.02% | 7.6 | 0.012 |
| wdbc | PFossil:decision list (any class) | 94.02% | 7.3 | 0.020 |
| wdbc | PFossil:rule set (best rule) | 94.56% | 11.7 | 0.018 |
| wdbc | PFossil:rule set (distributions) | 93.85% | 11.7 | 0.018 |
| wdbc | PFossil:rule set (votes) | 94.56% | 11.7 | 0.018 |
| wdbc | Pypper:concept (majority class) | 93.13% | 3.0 | 0.024 |
| wdbc | Pypper:concept (minority class) | 94.56% | 3.2 | 0.021 |
| wdbc | Pypper:decision list (any class) | 94.38% | 3.4 | 0.014 |
| wdbc | Pypper:rule set (best rule) | 94.91% | 6.2 | 0.046 |
| wdbc | Pypper:rule set (distributions) | 94.90% | 6.2 | 0.046 |
| wdbc | Pypper:rule set (votes) | 94.72% | 6.2 | 0.046 |
| wilt | CN2:concept (majority class) | 97.07% | 36.1 | 0.372 |
| wilt | CN2:concept (minority class) | 68.86% | 24.0 | 0.240 |
| wilt | CN2:decision list (any class) | 95.95% | 33.4 | 0.808 |
| wilt | CN2:rule set (best rule) | 96.96% | 60.1 | 0.622 |
| wilt | CN2:rule set (distributions) | 96.96% | 60.1 | 0.622 |
| wilt | CN2:rule set (votes) | 92.13% | 60.1 | 0.622 |
| wilt | PFossil:concept (majority class) | 5.39% | 0.0 | 0.003 |
| wilt | PFossil:concept (minority class) | 97.02% | 1.0 | 0.006 |
| wilt | PFossil:decision list (any class) | 97.02% | 1.0 | 0.012 |
| wilt | PFossil:rule set (best rule) | 97.02% | 1.0 | 0.014 |
| wilt | PFossil:rule set (distributions) | 97.02% | 1.0 | 0.014 |
| wilt | PFossil:rule set (votes) | 97.02% | 1.0 | 0.014 |
| wilt | Pypper:concept (majority class) | 97.40% | 4.4 | 0.099 |
| wilt | Pypper:concept (minority class) | 97.21% | 1.7 | 0.038 |
| wilt | Pypper:decision list (any class) | 97.23% | 4.8 | 0.051 |
| wilt | Pypper:rule set (best rule) | 97.42% | 6.1 | 0.140 |
| wilt | Pypper:rule set (distributions) | 97.42% | 6.1 | 0.140 |
| wilt | Pypper:rule set (votes) | 97.42% | 6.1 | 0.140 |
| wine | CN2:cascade (most frequent first) | 91.57% | 5.4 | 0.007 |
| wine | CN2:cascade (rarest first) | 93.27% | 4.0 | 0.008 |
| wine | CN2:decision list (any class) | 93.30% | 5.1 | 0.011 |
| wine | CN2:decision list (each class) | 91.54% | 4.7 | 0.018 |
| wine | CN2:decision list (seed's class) | 89.97% | 5.5 | 0.009 |
| wine | CN2:decision list (seed) | 91.08% | 5.0 | 0.010 |
| wine | CN2:rule set (best rule) | 90.98% | 8.7 | 0.015 |
| wine | CN2:rule set (distributions) | 91.54% | 8.7 | 0.015 |
| wine | CN2:rule set (votes) | 91.54% | 8.7 | 0.015 |
| wine | PFossil:cascade (most frequent first) | 95.49% | 5.4 | 0.004 |
| wine | PFossil:cascade (rarest first) | 94.97% | 4.3 | 0.004 |
| wine | PFossil:decision list (any class) | 92.12% | 4.2 | 0.007 |
| wine | PFossil:decision list (each class) | 91.05% | 4.0 | 0.008 |
| wine | PFossil:decision list (seed's class) | 91.01% | 5.4 | 0.005 |
| wine | PFossil:decision list (seed) | 93.27% | 5.3 | 0.005 |
| wine | PFossil:rule set (best rule) | 92.75% | 9.4 | 0.008 |
| wine | PFossil:rule set (distributions) | 93.24% | 9.4 | 0.008 |
| wine | PFossil:rule set (votes) | 93.79% | 9.4 | 0.008 |
| wine | Pypper:cascade (most frequent first) | 88.20% | 3.4 | 0.016 |
| wine | Pypper:cascade (rarest first) | 92.09% | 2.7 | 0.015 |
| wine | Pypper:decision list (any class) | 90.42% | 2.7 | 0.008 |
| wine | Pypper:decision list (each class) | 87.65% | 3.4 | 0.016 |
| wine | Pypper:decision list (seed's class) | 87.65% | 4.0 | 0.011 |
| wine | Pypper:decision list (seed) | 86.01% | 4.4 | 0.011 |
| wine | Pypper:rule set (best rule) | 92.09% | 5.2 | 0.027 |
| wine | Pypper:rule set (distributions) | 90.92% | 5.2 | 0.027 |
| wine | Pypper:rule set (votes) | 88.69% | 5.2 | 0.027 |
| yeast | CN2:cascade (most frequent first) | 52.16% | 39.5 | 0.519 |
| yeast | CN2:cascade (rarest first) | 53.31% | 38.4 | 0.486 |
| yeast | CN2:decision list (any class) | 54.11% | 41.3 | 1.565 |
| yeast | CN2:decision list (each class) | 57.27% | 55.3 | 4.445 |
| yeast | CN2:decision list (seed's class) | 55.25% | 62.1 | 0.856 |
| yeast | CN2:decision list (seed) | 56.33% | 46.1 | 0.605 |
| yeast | CN2:rule set (best rule) | 56.20% | 53.7 | 0.662 |
| yeast | CN2:rule set (distributions) | 50.81% | 53.7 | 0.662 |
| yeast | CN2:rule set (votes) | 51.55% | 53.7 | 0.662 |
| yeast | PFossil:cascade (most frequent first) | 53.57% | 36.3 | 0.055 |
| yeast | PFossil:cascade (rarest first) | 56.06% | 29.4 | 0.076 |
| yeast | PFossil:decision list (any class) | 58.76% | 20.6 | 0.079 |
| yeast | PFossil:decision list (each class) | 56.80% | 35.3 | 0.381 |
| yeast | PFossil:decision list (seed's class) | 58.15% | 31.0 | 0.053 |
| yeast | PFossil:decision list (seed) | 54.24% | 24.2 | 0.042 |
| yeast | PFossil:rule set (best rule) | 57.48% | 31.2 | 0.089 |
| yeast | PFossil:rule set (distributions) | 55.52% | 31.2 | 0.089 |
| yeast | PFossil:rule set (votes) | 54.31% | 31.2 | 0.089 |
| yeast | Pypper:cascade (most frequent first) | 53.84% | 18.1 | 0.244 |
| yeast | Pypper:cascade (rarest first) | 58.96% | 14.9 | 0.203 |
| yeast | Pypper:decision list (any class) | 52.49% | 16.6 | 0.119 |
| yeast | Pypper:decision list (each class) | 52.76% | 10.0 | 0.281 |
| yeast | Pypper:decision list (seed's class) | 58.42% | 48.6 | 0.258 |
| yeast | Pypper:decision list (seed) | 56.20% | 47.5 | 0.338 |
| yeast | Pypper:rule set (best rule) | 59.30% | 17.5 | 0.299 |
| yeast | Pypper:rule set (distributions) | 58.62% | 17.5 | 0.299 |
| yeast | Pypper:rule set (votes) | 58.69% | 17.5 | 0.299 |
| zoo | CN2:cascade (most frequent first) | 94.09% | 6.0 | 0.008 |
| zoo | CN2:cascade (rarest first) | 90.09% | 4.6 | 0.008 |
| zoo | CN2:decision list (any class) | 88.18% | 5.2 | 0.009 |
| zoo | CN2:decision list (each class) | 88.09% | 5.3 | 0.042 |
| zoo | CN2:decision list (seed's class) | 92.09% | 5.9 | 0.008 |
| zoo | CN2:decision list (seed) | 92.09% | 5.9 | 0.008 |
| zoo | CN2:rule set (best rule) | 87.18% | 5.8 | 0.011 |
| zoo | CN2:rule set (distributions) | 86.18% | 5.8 | 0.011 |
| zoo | CN2:rule set (votes) | 86.18% | 5.8 | 0.011 |
| zoo | PFossil:cascade (most frequent first) | 96.00% | 8.3 | 0.006 |
| zoo | PFossil:cascade (rarest first) | 93.00% | 7.7 | 0.007 |
| zoo | PFossil:decision list (any class) | 96.00% | 6.8 | 0.008 |
| zoo | PFossil:decision list (each class) | 97.00% | 6.0 | 0.015 |
| zoo | PFossil:decision list (seed's class) | 91.09% | 6.2 | 0.005 |
| zoo | PFossil:decision list (seed) | 91.09% | 6.2 | 0.005 |
| zoo | PFossil:rule set (best rule) | 95.00% | 8.9 | 0.008 |
| zoo | PFossil:rule set (distributions) | 94.00% | 8.9 | 0.008 |
| zoo | PFossil:rule set (votes) | 93.00% | 8.9 | 0.008 |
| zoo | Pypper:cascade (most frequent first) | 93.09% | 5.2 | 0.026 |
| zoo | Pypper:cascade (rarest first) | 89.09% | 5.3 | 0.027 |
| zoo | Pypper:decision list (any class) | 92.00% | 6.0 | 0.012 |
| zoo | Pypper:decision list (each class) | 90.00% | 6.1 | 0.038 |
| zoo | Pypper:decision list (seed's class) | 89.09% | 6.2 | 0.015 |
| zoo | Pypper:decision list (seed) | 89.09% | 6.2 | 0.011 |
| zoo | Pypper:rule set (best rule) | 90.09% | 6.1 | 0.031 |
| zoo | Pypper:rule set (distributions) | 91.09% | 6.1 | 0.031 |
| zoo | Pypper:rule set (votes) | 90.09% | 6.1 | 0.031 |

