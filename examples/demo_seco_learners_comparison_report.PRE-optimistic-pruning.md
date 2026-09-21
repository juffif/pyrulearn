# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-08-25 17:37:18. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil` only -- `jrip` needs no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 94.94 +/- 1.72 | 0/5 |
| pfoil_A | 92.41 +/- 2.78 | 0/5 |
| pfoil_B | 93.10 +/- 2.18 | 0/5 |
| cn2beam1_A | 94.48 +/- 1.52 | 0/5 |
| cn2beam1_B | 92.18 +/- 1.69 | 0/5 |
| cn2beam5_A | 94.02 +/- 1.69 | 0/5 |
| cn2beam5_B | 92.87 +/- 1.52 | 0/5 |
| pfossil_A | 90.11 +/- 2.00 | 0/5 |
| pfossil_B | 88.97 +/- 1.56 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.403 |
| pfoil_A | 1.085 |
| pfoil_B | 0.247 |
| cn2beam1_A | 0.129 |
| cn2beam1_B | 0.213 |
| cn2beam5_A | 0.274 |
| cn2beam5_B | 0.564 |
| pfossil_A | 1.487 |
| pfossil_B | 0.990 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.2 | 1.67 |
| pfoil_A | 8.2 | 2.60 |
| pfoil_B | 8.0 | 3.14 |
| cn2beam1_A | 6.2 | 1.88 |
| cn2beam1_B | 8.2 | 2.64 |
| cn2beam5_A | 5.2 | 1.93 |
| cn2beam5_B | 7.4 | 2.69 |
| pfossil_A | 4.8 | 2.00 |
| pfossil_B | 3.4 | 1.62 |

(27.3s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 74.11 +/- 2.17 | 0/5 |
| pfoil_A | 67.48 +/- 3.42 | 0/5 |
| pfoil_B | 48.63 +/- 9.35 | 0/5 |
| cn2beam1_A | 70.27 +/- 2.79 | 0/5 |
| cn2beam1_B | 69.22 +/- 2.50 | 0/5 |
| cn2beam5_A | 69.22 +/- 3.69 | 0/5 |
| cn2beam5_B | 70.97 +/- 1.93 | 0/5 |
| pfossil_A | 73.07 +/- 2.21 | 0/5 |
| pfossil_B | 39.18 +/- 3.42 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.351 |
| pfoil_A | 0.609 |
| pfoil_B | 0.436 |
| cn2beam1_A | 0.164 |
| cn2beam1_B | 0.055 |
| cn2beam5_A | 0.378 |
| cn2beam5_B | 0.121 |
| pfossil_A | 1.489 |
| pfossil_B | 0.421 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.2 | 1.80 |
| pfoil_A | 22.8 | 3.86 |
| pfoil_B | 18.8 | 3.32 |
| cn2beam1_A | 16.2 | 1.39 |
| cn2beam1_B | 5.0 | 1.39 |
| cn2beam5_A | 16.6 | 1.67 |
| cn2beam5_B | 5.4 | 1.77 |
| pfossil_A | 6.0 | 3.09 |
| pfossil_B | 2.0 | 2.70 |

(20.4s total)

---

## colic

n=368, attributes=26, A='1', B='2'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 84.78 +/- 1.98 | 0/5 |
| pfoil_A | 78.82 +/- 2.71 | 0/5 |
| pfoil_B | 74.46 +/- 4.04 | 0/5 |
| cn2beam1_A | 75.55 +/- 4.49 | 0/5 |
| cn2beam1_B | 79.36 +/- 2.70 | 0/5 |
| cn2beam5_A | 76.39 +/- 5.14 | 0/5 |
| cn2beam5_B | 79.90 +/- 3.31 | 0/5 |
| pfossil_A | 80.46 +/- 5.73 | 0/5 |
| pfossil_B | 79.91 +/- 4.32 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.420 |
| pfoil_A | 2.311 |
| pfoil_B | 1.974 |
| cn2beam1_A | 2.438 |
| cn2beam1_B | 1.185 |
| cn2beam5_A | 2.545 |
| cn2beam5_B | 1.674 |
| pfossil_A | 29.747 |
| pfossil_B | 31.576 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.4 | 1.85 |
| pfoil_A | 13.6 | 3.10 |
| pfoil_B | 14.8 | 2.58 |
| cn2beam1_A | 39.4 | 1.27 |
| cn2beam1_B | 18.6 | 1.32 |
| cn2beam5_A | 32.4 | 1.38 |
| cn2beam5_B | 17.0 | 1.32 |
| pfossil_A | 8.0 | 4.20 |
| pfossil_B | 11.4 | 2.75 |

(370.4s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 86.52 +/- 3.03 | 0/5 |
| pfoil_A | 79.42 +/- 1.75 | 0/5 |
| pfoil_B | 79.42 +/- 3.39 | 0/5 |
| cn2beam1_A | 77.25 +/- 2.62 | 0/5 |
| cn2beam1_B | 78.99 +/- 2.10 | 0/5 |
| cn2beam5_A | 77.83 +/- 2.62 | 0/5 |
| cn2beam5_B | 79.57 +/- 2.35 | 0/5 |
| pfossil_A | 63.77 +/- 4.72 | 0/5 |
| pfossil_B | 71.16 +/- 3.59 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.415 |
| pfoil_A | 1.768 |
| pfoil_B | 1.882 |
| cn2beam1_A | 0.648 |
| cn2beam1_B | 0.790 |
| cn2beam5_A | 1.509 |
| cn2beam5_B | 2.191 |
| pfossil_A | 6.329 |
| pfossil_B | 7.512 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 4.4 | 2.40 |
| pfoil_A | 28.0 | 3.39 |
| pfoil_B | 24.2 | 4.33 |
| cn2beam1_A | 15.8 | 2.23 |
| cn2beam1_B | 16.6 | 2.67 |
| cn2beam5_A | 12.4 | 2.42 |
| cn2beam5_B | 15.4 | 2.88 |
| pfossil_A | 8.0 | 1.74 |
| pfossil_B | 7.2 | 3.29 |

(115.9s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 71.80 +/- 1.96 | 0/5 |
| pfoil_A | 63.50 +/- 2.17 | 0/5 |
| pfoil_B | 70.90 +/- 3.77 | 0/5 |
| cn2beam1_A | 71.70 +/- 1.33 | 0/5 |
| cn2beam1_B | 70.60 +/- 3.40 | 0/5 |
| cn2beam5_A | 72.00 +/- 2.12 | 0/5 |
| cn2beam5_B | 66.30 +/- 3.23 | 0/5 |
| pfossil_A | 41.20 +/- 0.93 | 0/5 |
| pfossil_B | 71.10 +/- 0.97 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.525 |
| pfoil_A | 4.675 |
| pfoil_B | 4.982 |
| cn2beam1_A | 0.369 |
| cn2beam1_B | 1.497 |
| cn2beam5_A | 2.573 |
| cn2beam5_B | 5.466 |
| pfossil_A | 3.728 |
| pfossil_B | 3.246 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.4 | 3.43 |
| pfoil_A | 44.2 | 4.95 |
| pfoil_B | 48.8 | 4.54 |
| cn2beam1_A | 8.0 | 2.20 |
| cn2beam1_B | 33.6 | 2.14 |
| cn2beam5_A | 16.4 | 2.50 |
| cn2beam5_B | 38.6 | 2.63 |
| pfossil_A | 8.4 | 2.34 |
| pfossil_B | 6.4 | 2.90 |

(136.3s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 76.05 +/- 2.47 | 0/5 |
| pfoil_A | 67.32 +/- 3.31 | 0/5 |
| pfoil_B | 61.84 +/- 3.47 | 0/5 |
| cn2beam1_A | 69.14 +/- 3.87 | 0/5 |
| cn2beam1_B | 70.70 +/- 1.65 | 0/5 |
| cn2beam5_A | 68.10 +/- 4.29 | 0/5 |
| cn2beam5_B | 71.62 +/- 1.01 | 0/5 |
| pfossil_A | 67.58 +/- 1.95 | 0/5 |
| pfossil_B | 47.65 +/- 3.13 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.422 |
| pfoil_A | 2.146 |
| pfoil_B | 1.978 |
| cn2beam1_A | 0.708 |
| cn2beam1_B | 0.312 |
| cn2beam5_A | 1.994 |
| cn2beam5_B | 0.771 |
| pfossil_A | 2.224 |
| pfossil_B | 1.617 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 4.0 | 2.97 |
| pfoil_A | 42.6 | 4.74 |
| pfoil_B | 36.2 | 4.87 |
| cn2beam1_A | 20.6 | 2.77 |
| cn2beam1_B | 10.6 | 2.63 |
| cn2beam5_A | 19.4 | 2.88 |
| cn2beam5_B | 11.4 | 2.76 |
| pfossil_A | 7.4 | 2.37 |
| pfossil_B | 6.2 | 1.63 |

(61.5s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

*4 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 70.69 +/- 4.53 | 0/5 |
| pfoil_A | 73.58 +/- 7.64 | 0/5 |
| pfoil_B | 69.70 +/- 3.40 | 0/5 |
| cn2beam1_A | 67.28 +/- 4.60 | 0/5 |
| cn2beam1_B | 61.06 +/- 3.49 | 0/5 |
| cn2beam5_A | 75.92 +/- 5.63 | 0/5 |
| cn2beam5_B | 63.94 +/- 2.20 | 0/5 |
| pfossil_A | 74.24 +/- 3.80 | 2/5 |
| pfossil_B | 75.82 +/- 1.74 | 2/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.396 |
| pfoil_A | 3.956 |
| pfoil_B | 4.407 |
| cn2beam1_A | 3.687 |
| cn2beam1_B | 4.187 |
| cn2beam5_A | 4.273 |
| cn2beam5_B | 4.797 |
| pfossil_A | 39.067 |
| pfossil_B | 24.820 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 4.0 | 2.06 |
| pfoil_A | 7.4 | 2.41 |
| pfoil_B | 7.0 | 2.64 |
| cn2beam1_A | 14.0 | 1.35 |
| cn2beam1_B | 16.6 | 1.31 |
| cn2beam5_A | 13.0 | 1.43 |
| cn2beam5_B | 13.6 | 1.46 |
| pfossil_A | 5.0 | 4.34 |
| pfossil_B | 4.7 | 3.87 |

(562.1s total)

---

## ionosphere

n=351, attributes=33, A='b', B='g'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 90.60 +/- 3.20 | 0/5 |
| pfoil_A | 83.48 +/- 4.44 | 0/5 |
| pfoil_B | 82.61 +/- 6.05 | 0/5 |
| cn2beam1_A | 92.60 +/- 1.04 | 0/5 |
| cn2beam1_B | 81.77 +/- 2.74 | 0/5 |
| cn2beam5_A | 93.16 +/- 1.40 | 0/5 |
| cn2beam5_B | 79.75 +/- 4.78 | 0/5 |
| pfossil_A | 78.93 +/- 7.26 | 0/5 |
| pfossil_B | 89.74 +/- 4.82 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.401 |
| pfoil_A | 1.170 |
| pfoil_B | 1.553 |
| cn2beam1_A | 0.643 |
| cn2beam1_B | 1.531 |
| cn2beam5_A | 0.846 |
| cn2beam5_B | 3.663 |
| pfossil_A | 25.410 |
| pfossil_B | 34.444 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.4 | 1.52 |
| pfoil_A | 9.6 | 1.99 |
| pfoil_B | 7.2 | 3.61 |
| cn2beam1_A | 9.0 | 1.15 |
| cn2beam1_B | 8.2 | 3.00 |
| cn2beam5_A | 8.8 | 1.18 |
| cn2beam5_B | 6.6 | 2.83 |
| pfossil_A | 7.6 | 1.73 |
| pfossil_B | 2.8 | 5.50 |

(349.3s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 97.81 +/- 0.90 | 0/5 |
| pfoil_A | 95.51 +/- 2.83 | 0/5 |
| pfoil_B | 99.48 +/- 0.66 | 0/5 |
| cn2beam1_A | 83.71 +/- 1.97 | 0/5 |
| cn2beam1_B | 70.03 +/- 4.75 | 0/5 |
| cn2beam5_A | 83.71 +/- 1.97 | 0/5 |
| cn2beam5_B | 76.94 +/- 11.26 | 0/5 |
| pfossil_A | 56.26 +/- 2.11 | 0/5 |
| pfossil_B | 73.49 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.383 |
| pfoil_A | 0.207 |
| pfoil_B | 0.128 |
| cn2beam1_A | 0.109 |
| cn2beam1_B | 0.090 |
| cn2beam5_A | 0.423 |
| cn2beam5_B | 0.261 |
| pfossil_A | 0.272 |
| pfossil_B | 0.206 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 8.6 | 3.19 |
| pfoil_A | 13.6 | 3.78 |
| pfoil_B | 9.2 | 3.37 |
| cn2beam1_A | 9.4 | 2.79 |
| cn2beam1_B | 8.4 | 2.57 |
| cn2beam5_A | 9.0 | 2.78 |
| cn2beam5_B | 7.4 | 2.40 |
| pfossil_A | 5.6 | 2.46 |
| pfossil_B | 4.6 | 1.96 |

(10.7s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.61 +/- 0.58 | 0/5 |
| pfoil_A | 98.47 +/- 0.36 | 0/5 |
| pfoil_B | 98.03 +/- 0.59 | 0/5 |
| cn2beam1_A | 94.82 +/- 0.85 | 0/5 |
| cn2beam1_B | 92.06 +/- 1.44 | 0/5 |
| cn2beam5_A | 97.89 +/- 1.37 | 0/5 |
| cn2beam5_B | 95.05 +/- 2.32 | 0/5 |
| pfossil_A | 75.73 +/- 5.44 | 0/5 |
| pfossil_B | 76.09 +/- 1.94 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.376 |
| pfoil_A | 0.207 |
| pfoil_B | 0.170 |
| cn2beam1_A | 0.096 |
| cn2beam1_B | 0.076 |
| cn2beam5_A | 0.410 |
| cn2beam5_B | 0.320 |
| pfossil_A | 0.419 |
| pfossil_B | 0.742 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.4 | 2.36 |
| pfoil_A | 13.6 | 2.95 |
| pfoil_B | 11.6 | 3.02 |
| cn2beam1_A | 6.6 | 2.29 |
| cn2beam1_B | 7.2 | 2.17 |
| cn2beam5_A | 6.6 | 2.48 |
| cn2beam5_B | 7.0 | 2.32 |
| pfossil_A | 3.8 | 1.58 |
| pfossil_B | 5.4 | 1.89 |

(14.6s total)

---

## kr-vs-kp

n=3196, attributes=36, A='nowin', B='won'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 99.22 +/- 0.38 | 0/5 |
| pfoil_A | 98.44 +/- 0.56 | 0/5 |
| pfoil_B | 98.56 +/- 0.27 | 0/5 |
| cn2beam1_A | 87.64 +/- 1.78 | 0/5 |
| cn2beam1_B | 87.02 +/- 1.68 | 0/5 |
| cn2beam5_A | 89.71 +/- 0.77 | 0/5 |
| cn2beam5_B | 88.45 +/- 1.49 | 0/5 |
| pfossil_A | 92.49 +/- 1.31 | 0/5 |
| pfossil_B | 77.94 +/- 0.60 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.687 |
| pfoil_A | 2.319 |
| pfoil_B | 2.265 |
| cn2beam1_A | 0.610 |
| cn2beam1_B | 0.351 |
| cn2beam5_A | 0.819 |
| cn2beam5_B | 1.291 |
| pfossil_A | 5.924 |
| pfossil_B | 2.150 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 13.6 | 2.99 |
| pfoil_A | 22.6 | 3.61 |
| pfoil_B | 16.4 | 5.13 |
| cn2beam1_A | 9.2 | 1.94 |
| cn2beam1_B | 4.8 | 2.43 |
| cn2beam5_A | 8.2 | 2.04 |
| cn2beam5_B | 5.0 | 2.75 |
| pfossil_A | 7.2 | 2.64 |
| pfossil_B | 4.0 | 2.00 |

(84.1s total)

---

## mushroom

n=8124, attributes=21, A='e', B='p'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 99.98 +/- 0.05 | 0/5 |
| pfoil_A | 100.00 +/- 0.00 | 0/5 |
| pfoil_B | 100.00 +/- 0.00 | 0/5 |
| cn2beam1_A | 98.86 +/- 0.24 | 0/5 |
| cn2beam1_B | 99.75 +/- 0.10 | 0/5 |
| cn2beam5_A | 99.41 +/- 0.20 | 0/5 |
| cn2beam5_B | 99.75 +/- 0.10 | 0/5 |
| pfossil_A | 99.83 +/- 0.12 | 0/5 |
| pfossil_B | 99.98 +/- 0.05 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 1.058 |
| pfoil_A | 1.414 |
| pfoil_B | 1.166 |
| cn2beam1_A | 1.820 |
| cn2beam1_B | 1.002 |
| cn2beam5_A | 2.241 |
| cn2beam5_B | 1.017 |
| pfossil_A | 254.331 |
| pfossil_B | 742.360 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.4 | 2.09 |
| pfoil_A | 6.2 | 2.42 |
| pfoil_B | 6.2 | 1.84 |
| cn2beam1_A | 13.2 | 1.24 |
| cn2beam1_B | 8.8 | 1.00 |
| cn2beam5_A | 14.0 | 1.36 |
| cn2beam5_B | 8.8 | 1.00 |
| pfossil_A | 1.4 | 7.60 |
| pfossil_B | 3.8 | 2.20 |

(5039.5s total)

---

## Overall summary

### Accuracy

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 94.94 | 92.41 | 93.10 | 94.48 | 92.18 | 94.02 | 92.87 | 90.11 | 88.97 |
| breast-cancer | 74.11 | 67.48 | 48.63 | 70.27 | 69.22 | 69.22 | 70.97 | 73.07 | 39.18 |
| colic | 84.78 | 78.82 | 74.46 | 75.55 | 79.36 | 76.39 | 79.90 | 80.46 | 79.91 |
| credit-approval | 86.52 | 79.42 | 79.42 | 77.25 | 78.99 | 77.83 | 79.57 | 63.77 | 71.16 |
| credit-g | 71.80 | 63.50 | 70.90 | 71.70 | 70.60 | 72.00 | 66.30 | 41.20 | 71.10 |
| diabetes | 76.05 | 67.32 | 61.84 | 69.14 | 70.70 | 68.10 | 71.62 | 67.58 | 47.65 |
| sonar | 70.69 | 73.58 | 69.70 | 67.28 | 61.06 | 75.92 | 63.94 | 74.24 | 75.82 |
| ionosphere | 90.60 | 83.48 | 82.61 | 92.60 | 81.77 | 93.16 | 79.75 | 78.93 | 89.74 |
| tic-tac-toe | 97.81 | 95.51 | 99.48 | 83.71 | 70.03 | 83.71 | 76.94 | 56.26 | 73.49 |
| banknote-authentication | 98.61 | 98.47 | 98.03 | 94.82 | 92.06 | 97.89 | 95.05 | 75.73 | 76.09 |
| kr-vs-kp | 99.22 | 98.44 | 98.56 | 87.64 | 87.02 | 89.71 | 88.45 | 92.49 | 77.94 |
| mushroom | 99.98 | 100.00 | 100.00 | 98.86 | 99.75 | 99.41 | 99.75 | 99.83 | 99.98 |

### Fit time (seconds/fold)

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 0.403 | 1.085 | 0.247 | 0.129 | 0.213 | 0.274 | 0.564 | 1.487 | 0.990 |
| breast-cancer | 0.351 | 0.609 | 0.436 | 0.164 | 0.055 | 0.378 | 0.121 | 1.489 | 0.421 |
| colic | 0.420 | 2.311 | 1.974 | 2.438 | 1.185 | 2.545 | 1.674 | 29.747 | 31.576 |
| credit-approval | 0.415 | 1.768 | 1.882 | 0.648 | 0.790 | 1.509 | 2.191 | 6.329 | 7.512 |
| credit-g | 0.525 | 4.675 | 4.982 | 0.369 | 1.497 | 2.573 | 5.466 | 3.728 | 3.246 |
| diabetes | 0.422 | 2.146 | 1.978 | 0.708 | 0.312 | 1.994 | 0.771 | 2.224 | 1.617 |
| sonar | 0.396 | 3.956 | 4.407 | 3.687 | 4.187 | 4.273 | 4.797 | 39.067 | 24.820 |
| ionosphere | 0.401 | 1.170 | 1.553 | 0.643 | 1.531 | 0.846 | 3.663 | 25.410 | 34.444 |
| tic-tac-toe | 0.383 | 0.207 | 0.128 | 0.109 | 0.090 | 0.423 | 0.261 | 0.272 | 0.206 |
| banknote-authentication | 0.376 | 0.207 | 0.170 | 0.096 | 0.076 | 0.410 | 0.320 | 0.419 | 0.742 |
| kr-vs-kp | 0.687 | 2.319 | 2.265 | 0.610 | 0.351 | 0.819 | 1.291 | 5.924 | 2.150 |
| mushroom | 1.058 | 1.414 | 1.166 | 1.820 | 1.002 | 2.241 | 1.017 | 254.331 | 742.360 |

### Rule count

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 2.2 | 8.2 | 8.0 | 6.2 | 8.2 | 5.2 | 7.4 | 4.8 | 3.4 |
| breast-cancer | 1.2 | 22.8 | 18.8 | 16.2 | 5.0 | 16.6 | 5.4 | 6.0 | 2.0 |
| colic | 3.4 | 13.6 | 14.8 | 39.4 | 18.6 | 32.4 | 17.0 | 8.0 | 11.4 |
| credit-approval | 4.4 | 28.0 | 24.2 | 15.8 | 16.6 | 12.4 | 15.4 | 8.0 | 7.2 |
| credit-g | 2.4 | 44.2 | 48.8 | 8.0 | 33.6 | 16.4 | 38.6 | 8.4 | 6.4 |
| diabetes | 4.0 | 42.6 | 36.2 | 20.6 | 10.6 | 19.4 | 11.4 | 7.4 | 6.2 |
| sonar | 4.0 | 7.4 | 7.0 | 14.0 | 16.6 | 13.0 | 13.6 | 5.0 | 4.7 |
| ionosphere | 5.4 | 9.6 | 7.2 | 9.0 | 8.2 | 8.8 | 6.6 | 7.6 | 2.8 |
| tic-tac-toe | 8.6 | 13.6 | 9.2 | 9.4 | 8.4 | 9.0 | 7.4 | 5.6 | 4.6 |
| banknote-authentication | 5.4 | 13.6 | 11.6 | 6.6 | 7.2 | 6.6 | 7.0 | 3.8 | 5.4 |
| kr-vs-kp | 13.6 | 22.6 | 16.4 | 9.2 | 4.8 | 8.2 | 5.0 | 7.2 | 4.0 |
| mushroom | 5.4 | 6.2 | 6.2 | 13.2 | 8.8 | 14.0 | 8.8 | 1.4 | 3.8 |

### Average conditions per rule

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 1.67 | 2.60 | 3.14 | 1.88 | 2.64 | 1.93 | 2.69 | 2.00 | 1.62 |
| breast-cancer | 1.80 | 3.86 | 3.32 | 1.39 | 1.39 | 1.67 | 1.77 | 3.09 | 2.70 |
| colic | 1.85 | 3.10 | 2.58 | 1.27 | 1.32 | 1.38 | 1.32 | 4.20 | 2.75 |
| credit-approval | 2.40 | 3.39 | 4.33 | 2.23 | 2.67 | 2.42 | 2.88 | 1.74 | 3.29 |
| credit-g | 3.43 | 4.95 | 4.54 | 2.20 | 2.14 | 2.50 | 2.63 | 2.34 | 2.90 |
| diabetes | 2.97 | 4.74 | 4.87 | 2.77 | 2.63 | 2.88 | 2.76 | 2.37 | 1.63 |
| sonar | 2.06 | 2.41 | 2.64 | 1.35 | 1.31 | 1.43 | 1.46 | 4.34 | 3.87 |
| ionosphere | 1.52 | 1.99 | 3.61 | 1.15 | 3.00 | 1.18 | 2.83 | 1.73 | 5.50 |
| tic-tac-toe | 3.19 | 3.78 | 3.37 | 2.79 | 2.57 | 2.78 | 2.40 | 2.46 | 1.96 |
| banknote-authentication | 2.36 | 2.95 | 3.02 | 2.29 | 2.17 | 2.48 | 2.32 | 1.58 | 1.89 |
| kr-vs-kp | 2.99 | 3.61 | 5.13 | 1.94 | 2.43 | 2.04 | 2.75 | 2.64 | 2.00 |
| mushroom | 2.09 | 2.42 | 1.84 | 1.24 | 1.00 | 1.36 | 1.00 | 7.60 | 2.20 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 87.09 | 0.487 | 5.0 | 2.36 | 0 |
| pfoil_A | 83.20 | 1.822 | 19.4 | 3.32 | 0 |
| pfoil_B | 81.39 | 1.766 | 17.4 | 3.53 | 0 |
| cn2beam1_A | 81.94 | 0.952 | 14.0 | 1.87 | 0 |
| cn2beam1_B | 79.39 | 0.941 | 12.2 | 2.11 | 0 |
| cn2beam5_A | 83.11 | 1.524 | 13.5 | 2.00 | 0 |
| cn2beam5_B | 80.43 | 1.845 | 12.0 | 2.23 | 0 |
| pfossil_A | 74.47 | 30.869 | 6.1 | 3.01 | 0 |
| pfossil_B | 74.25 | 70.840 | 5.2 | 2.69 | 0 |

**Average rank per criterion** (1 = best of 9; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 1.88 | 3.08 | 2.21 | 5.25 |
| pfoil_A | 4.62 | 5.83 | 7.75 | 7.67 |
| pfoil_B | 4.79 | 5.25 | 6.71 | 7.92 |
| cn2beam1_A | 5.29 | 2.67 | 6.54 | 2.58 |
| cn2beam1_B | 6.42 | 2.08 | 5.83 | 3.46 |
| cn2beam5_A | 4.25 | 5.50 | 5.96 | 4.08 |
| cn2beam5_B | 5.21 | 5.25 | 5.12 | 4.46 |
| pfossil_A | 6.25 | 8.17 | 3.00 | 4.83 |
| pfossil_B | 6.29 | 7.17 | 1.88 | 4.75 |
