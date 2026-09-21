# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-08-28 22:51:51. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil` only -- `jrip` needs no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

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
| pfossil_A | 93.10 +/- 2.30 | 0/5 |
| pfossil_B | 94.48 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.347 |
| pfoil_A | 0.596 |
| pfoil_B | 0.275 |
| cn2beam1_A | 0.139 |
| cn2beam1_B | 0.265 |
| cn2beam5_A | 0.301 |
| cn2beam5_B | 0.685 |
| pfossil_A | 0.199 |
| pfossil_B | 0.106 |

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
| pfossil_A | 6.0 | 2.76 |
| pfossil_B | 3.8 | 2.25 |

(15.1s total)

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
| cn2beam5_B | 71.67 +/- 2.14 | 0/5 |
| pfossil_A | 73.77 +/- 3.38 | 0/5 |
| pfossil_B | 60.19 +/- 12.61 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.316 |
| pfoil_A | 0.679 |
| pfoil_B | 0.505 |
| cn2beam1_A | 0.187 |
| cn2beam1_B | 0.066 |
| cn2beam5_A | 0.452 |
| cn2beam5_B | 0.161 |
| pfossil_A | 0.248 |
| pfossil_B | 0.221 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.2 | 1.80 |
| pfoil_A | 22.8 | 3.86 |
| pfoil_B | 18.8 | 3.32 |
| cn2beam1_A | 16.2 | 1.39 |
| cn2beam1_B | 5.0 | 1.39 |
| cn2beam5_A | 16.6 | 1.67 |
| cn2beam5_B | 5.6 | 1.82 |
| pfossil_A | 7.8 | 4.13 |
| pfossil_B | 5.4 | 4.36 |

(14.5s total)

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
| cn2beam5_B | 81.25 +/- 2.62 | 0/5 |
| pfossil_A | 84.27 +/- 4.51 | 0/5 |
| pfossil_B | 86.15 +/- 3.20 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.442 |
| pfoil_A | 2.795 |
| pfoil_B | 2.489 |
| cn2beam1_A | 3.051 |
| cn2beam1_B | 1.499 |
| cn2beam5_A | 3.359 |
| cn2beam5_B | 1.605 |
| pfossil_A | 2.000 |
| pfossil_B | 1.083 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.4 | 1.85 |
| pfoil_A | 13.6 | 3.10 |
| pfoil_B | 14.8 | 2.58 |
| cn2beam1_A | 39.4 | 1.27 |
| cn2beam1_B | 18.6 | 1.32 |
| cn2beam5_A | 32.4 | 1.38 |
| cn2beam5_B | 17.0 | 1.33 |
| pfossil_A | 10.0 | 3.24 |
| pfossil_B | 4.2 | 3.89 |

(92.9s total)

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
| cn2beam5_B | 79.13 +/- 2.73 | 0/5 |
| pfossil_A | 84.20 +/- 3.32 | 0/5 |
| pfossil_B | 84.64 +/- 3.85 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.456 |
| pfoil_A | 2.360 |
| pfoil_B | 2.496 |
| cn2beam1_A | 0.877 |
| cn2beam1_B | 1.059 |
| cn2beam5_A | 1.723 |
| cn2beam5_B | 2.819 |
| pfossil_A | 0.309 |
| pfossil_B | 0.797 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 4.4 | 2.40 |
| pfoil_A | 28.0 | 3.39 |
| pfoil_B | 24.2 | 4.33 |
| cn2beam1_A | 15.8 | 2.23 |
| cn2beam1_B | 16.6 | 2.67 |
| cn2beam5_A | 12.4 | 2.42 |
| cn2beam5_B | 15.2 | 2.92 |
| pfossil_A | 3.0 | 3.23 |
| pfossil_B | 7.4 | 4.54 |

(65.5s total)

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
| cn2beam5_A | 71.90 +/- 2.15 | 0/5 |
| cn2beam5_B | 66.30 +/- 3.08 | 0/5 |
| pfossil_A | 71.10 +/- 1.59 | 0/5 |
| pfossil_B | 68.40 +/- 3.20 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.531 |
| pfoil_A | 5.764 |
| pfoil_B | 5.954 |
| cn2beam1_A | 0.486 |
| cn2beam1_B | 1.982 |
| cn2beam5_A | 3.014 |
| cn2beam5_B | 6.702 |
| pfossil_A | 0.682 |
| pfossil_B | 0.543 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.4 | 3.43 |
| pfoil_A | 44.2 | 4.95 |
| pfoil_B | 48.8 | 4.54 |
| cn2beam1_A | 8.0 | 2.20 |
| cn2beam1_B | 33.6 | 2.14 |
| cn2beam5_A | 16.4 | 2.54 |
| cn2beam5_B | 37.8 | 2.65 |
| pfossil_A | 1.8 | 11.13 |
| pfossil_B | 1.6 | 7.40 |

(129.7s total)

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
| cn2beam5_A | 68.49 +/- 4.39 | 0/5 |
| cn2beam5_B | 71.62 +/- 1.01 | 0/5 |
| pfossil_A | 74.22 +/- 1.71 | 0/5 |
| pfossil_B | 70.96 +/- 3.38 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.493 |
| pfoil_A | 2.665 |
| pfoil_B | 2.462 |
| cn2beam1_A | 0.756 |
| cn2beam1_B | 0.391 |
| cn2beam5_A | 2.049 |
| cn2beam5_B | 0.975 |
| pfossil_A | 0.290 |
| pfossil_B | 0.473 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 4.0 | 2.97 |
| pfoil_A | 42.6 | 4.74 |
| pfoil_B | 36.2 | 4.87 |
| cn2beam1_A | 20.6 | 2.77 |
| cn2beam1_B | 10.6 | 2.63 |
| cn2beam5_A | 18.8 | 2.84 |
| cn2beam5_B | 11.4 | 2.76 |
| pfossil_A | 4.4 | 4.06 |
| pfossil_B | 6.6 | 4.38 |

(53.6s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

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
| pfossil_A | 73.60 +/- 7.85 | 0/5 |
| pfossil_B | 73.11 +/- 4.46 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.448 |
| pfoil_A | 6.029 |
| pfoil_B | 5.960 |
| cn2beam1_A | 6.195 |
| cn2beam1_B | 7.004 |
| cn2beam5_A | 7.040 |
| cn2beam5_B | 7.314 |
| pfossil_A | 7.122 |
| pfossil_B | 5.836 |

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
| pfossil_A | 7.6 | 3.08 |
| pfossil_B | 7.8 | 2.45 |

(267.6s total)

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
| pfossil_A | 87.47 +/- 3.53 | 0/5 |
| pfossil_B | 90.32 +/- 2.73 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.481 |
| pfoil_A | 1.809 |
| pfoil_B | 2.436 |
| cn2beam1_A | 1.031 |
| cn2beam1_B | 2.349 |
| cn2beam5_A | 1.072 |
| cn2beam5_B | 5.580 |
| pfossil_A | 1.570 |
| pfossil_B | 1.456 |

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
| pfossil_A | 9.6 | 1.74 |
| pfossil_B | 2.8 | 5.67 |

(90.5s total)

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
| pfossil_A | 80.47 +/- 4.17 | 0/5 |
| pfossil_B | 83.82 +/- 1.57 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.433 |
| pfoil_A | 0.286 |
| pfoil_B | 0.182 |
| cn2beam1_A | 0.165 |
| cn2beam1_B | 0.135 |
| cn2beam5_A | 0.520 |
| cn2beam5_B | 0.368 |
| pfossil_A | 0.184 |
| pfossil_B | 0.161 |

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
| pfossil_A | 7.6 | 3.74 |
| pfossil_B | 7.2 | 3.47 |

(12.7s total)

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
| pfossil_A | 95.77 +/- 0.85 | 0/5 |
| pfossil_B | 96.87 +/- 1.19 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.484 |
| pfoil_A | 0.265 |
| pfoil_B | 0.244 |
| cn2beam1_A | 0.114 |
| cn2beam1_B | 0.112 |
| cn2beam5_A | 0.521 |
| cn2beam5_B | 0.422 |
| pfossil_A | 0.100 |
| pfossil_B | 0.143 |

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
| pfossil_A | 6.2 | 2.19 |
| pfossil_B | 6.8 | 2.44 |

(12.8s total)

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
| pfossil_A | 98.25 +/- 0.80 | 0/5 |
| pfossil_B | 95.62 +/- 0.52 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.789 |
| pfoil_A | 3.202 |
| pfoil_B | 3.172 |
| cn2beam1_A | 0.746 |
| cn2beam1_B | 0.494 |
| cn2beam5_A | 1.087 |
| cn2beam5_B | 1.473 |
| pfossil_A | 1.128 |
| pfossil_B | 1.141 |

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
| pfossil_A | 9.4 | 2.76 |
| pfossil_B | 4.8 | 6.29 |

(69.1s total)

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
| pfossil_A | 99.95 +/- 0.05 | 0/5 |
| pfossil_B | 100.00 +/- 0.00 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 1.044 |
| pfoil_A | 1.918 |
| pfoil_B | 1.457 |
| cn2beam1_A | 2.350 |
| cn2beam1_B | 1.268 |
| cn2beam5_A | 2.698 |
| cn2beam5_B | 1.306 |
| pfossil_A | 1.643 |
| pfossil_B | 1.812 |

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
| pfossil_A | 6.4 | 1.91 |
| pfossil_B | 7.8 | 1.82 |

(88.3s total)

---

## Overall summary

### Accuracy

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 94.94 | 92.41 | 93.10 | 94.48 | 92.18 | 94.02 | 92.87 | 93.10 | 94.48 |
| breast-cancer | 74.11 | 67.48 | 48.63 | 70.27 | 69.22 | 69.22 | 71.67 | 73.77 | 60.19 |
| colic | 84.78 | 78.82 | 74.46 | 75.55 | 79.36 | 76.39 | 81.25 | 84.27 | 86.15 |
| credit-approval | 86.52 | 79.42 | 79.42 | 77.25 | 78.99 | 77.83 | 79.13 | 84.20 | 84.64 |
| credit-g | 71.80 | 63.50 | 70.90 | 71.70 | 70.60 | 71.90 | 66.30 | 71.10 | 68.40 |
| diabetes | 76.05 | 67.32 | 61.84 | 69.14 | 70.70 | 68.49 | 71.62 | 74.22 | 70.96 |
| sonar | 70.69 | 73.58 | 69.70 | 67.28 | 61.06 | 75.92 | 63.94 | 73.60 | 73.11 |
| ionosphere | 90.60 | 83.48 | 82.61 | 92.60 | 81.77 | 93.16 | 79.75 | 87.47 | 90.32 |
| tic-tac-toe | 97.81 | 95.51 | 99.48 | 83.71 | 70.03 | 83.71 | 76.94 | 80.47 | 83.82 |
| banknote-authentication | 98.61 | 98.47 | 98.03 | 94.82 | 92.06 | 97.89 | 95.05 | 95.77 | 96.87 |
| kr-vs-kp | 99.22 | 98.44 | 98.56 | 87.64 | 87.02 | 89.71 | 88.45 | 98.25 | 95.62 |
| mushroom | 99.98 | 100.00 | 100.00 | 98.86 | 99.75 | 99.41 | 99.75 | 99.95 | 100.00 |

### Fit time (seconds/fold)

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 0.347 | 0.596 | 0.275 | 0.139 | 0.265 | 0.301 | 0.685 | 0.199 | 0.106 |
| breast-cancer | 0.316 | 0.679 | 0.505 | 0.187 | 0.066 | 0.452 | 0.161 | 0.248 | 0.221 |
| colic | 0.442 | 2.795 | 2.489 | 3.051 | 1.499 | 3.359 | 1.605 | 2.000 | 1.083 |
| credit-approval | 0.456 | 2.360 | 2.496 | 0.877 | 1.059 | 1.723 | 2.819 | 0.309 | 0.797 |
| credit-g | 0.531 | 5.764 | 5.954 | 0.486 | 1.982 | 3.014 | 6.702 | 0.682 | 0.543 |
| diabetes | 0.493 | 2.665 | 2.462 | 0.756 | 0.391 | 2.049 | 0.975 | 0.290 | 0.473 |
| sonar | 0.448 | 6.029 | 5.960 | 6.195 | 7.004 | 7.040 | 7.314 | 7.122 | 5.836 |
| ionosphere | 0.481 | 1.809 | 2.436 | 1.031 | 2.349 | 1.072 | 5.580 | 1.570 | 1.456 |
| tic-tac-toe | 0.433 | 0.286 | 0.182 | 0.165 | 0.135 | 0.520 | 0.368 | 0.184 | 0.161 |
| banknote-authentication | 0.484 | 0.265 | 0.244 | 0.114 | 0.112 | 0.521 | 0.422 | 0.100 | 0.143 |
| kr-vs-kp | 0.789 | 3.202 | 3.172 | 0.746 | 0.494 | 1.087 | 1.473 | 1.128 | 1.141 |
| mushroom | 1.044 | 1.918 | 1.457 | 2.350 | 1.268 | 2.698 | 1.306 | 1.643 | 1.812 |

### Rule count

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 2.2 | 8.2 | 8.0 | 6.2 | 8.2 | 5.2 | 7.4 | 6.0 | 3.8 |
| breast-cancer | 1.2 | 22.8 | 18.8 | 16.2 | 5.0 | 16.6 | 5.6 | 7.8 | 5.4 |
| colic | 3.4 | 13.6 | 14.8 | 39.4 | 18.6 | 32.4 | 17.0 | 10.0 | 4.2 |
| credit-approval | 4.4 | 28.0 | 24.2 | 15.8 | 16.6 | 12.4 | 15.2 | 3.0 | 7.4 |
| credit-g | 2.4 | 44.2 | 48.8 | 8.0 | 33.6 | 16.4 | 37.8 | 1.8 | 1.6 |
| diabetes | 4.0 | 42.6 | 36.2 | 20.6 | 10.6 | 18.8 | 11.4 | 4.4 | 6.6 |
| sonar | 4.0 | 7.4 | 7.0 | 14.0 | 16.6 | 13.0 | 13.6 | 7.6 | 7.8 |
| ionosphere | 5.4 | 9.6 | 7.2 | 9.0 | 8.2 | 8.8 | 6.6 | 9.6 | 2.8 |
| tic-tac-toe | 8.6 | 13.6 | 9.2 | 9.4 | 8.4 | 9.0 | 7.4 | 7.6 | 7.2 |
| banknote-authentication | 5.4 | 13.6 | 11.6 | 6.6 | 7.2 | 6.6 | 7.0 | 6.2 | 6.8 |
| kr-vs-kp | 13.6 | 22.6 | 16.4 | 9.2 | 4.8 | 8.2 | 5.0 | 9.4 | 4.8 |
| mushroom | 5.4 | 6.2 | 6.2 | 13.2 | 8.8 | 14.0 | 8.8 | 6.4 | 7.8 |

### Average conditions per rule

| dataset | jrip | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B |
|---|---|---|---|---|---|---|---|---|---|
| vote | 1.67 | 2.60 | 3.14 | 1.88 | 2.64 | 1.93 | 2.69 | 2.76 | 2.25 |
| breast-cancer | 1.80 | 3.86 | 3.32 | 1.39 | 1.39 | 1.67 | 1.82 | 4.13 | 4.36 |
| colic | 1.85 | 3.10 | 2.58 | 1.27 | 1.32 | 1.38 | 1.33 | 3.24 | 3.89 |
| credit-approval | 2.40 | 3.39 | 4.33 | 2.23 | 2.67 | 2.42 | 2.92 | 3.23 | 4.54 |
| credit-g | 3.43 | 4.95 | 4.54 | 2.20 | 2.14 | 2.54 | 2.65 | 11.13 | 7.40 |
| diabetes | 2.97 | 4.74 | 4.87 | 2.77 | 2.63 | 2.84 | 2.76 | 4.06 | 4.38 |
| sonar | 2.06 | 2.41 | 2.64 | 1.35 | 1.31 | 1.43 | 1.46 | 3.08 | 2.45 |
| ionosphere | 1.52 | 1.99 | 3.61 | 1.15 | 3.00 | 1.18 | 2.83 | 1.74 | 5.67 |
| tic-tac-toe | 3.19 | 3.78 | 3.37 | 2.79 | 2.57 | 2.78 | 2.40 | 3.74 | 3.47 |
| banknote-authentication | 2.36 | 2.95 | 3.02 | 2.29 | 2.17 | 2.48 | 2.32 | 2.19 | 2.44 |
| kr-vs-kp | 2.99 | 3.61 | 5.13 | 1.94 | 2.43 | 2.04 | 2.75 | 2.76 | 6.29 |
| mushroom | 2.09 | 2.42 | 1.84 | 1.24 | 1.00 | 1.36 | 1.00 | 1.91 | 1.82 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 87.09 | 0.522 | 5.0 | 2.36 | 0 |
| pfoil_A | 83.20 | 2.364 | 19.4 | 3.32 | 0 |
| pfoil_B | 81.39 | 2.303 | 17.4 | 3.53 | 0 |
| cn2beam1_A | 81.94 | 1.341 | 14.0 | 1.87 | 0 |
| cn2beam1_B | 79.39 | 1.385 | 12.2 | 2.11 | 0 |
| cn2beam5_A | 83.14 | 1.986 | 13.4 | 2.00 | 0 |
| cn2beam5_B | 80.56 | 2.451 | 11.9 | 2.24 | 0 |
| pfossil_A | 84.68 | 1.290 | 6.7 | 3.67 | 0 |
| pfossil_B | 83.71 | 1.148 | 5.5 | 4.08 | 0 |

**Average rank per criterion** (1 = best of 9; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 2.00 | 3.67 | 2.17 | 4.50 |
| pfoil_A | 5.08 | 7.08 | 7.38 | 7.08 |
| pfoil_B | 5.33 | 6.25 | 6.38 | 7.42 |
| cn2beam1_A | 6.04 | 3.83 | 6.38 | 2.00 |
| cn2beam1_B | 7.33 | 3.25 | 5.62 | 2.62 |
| cn2beam5_A | 4.83 | 6.83 | 5.62 | 3.42 |
| cn2beam5_B | 6.38 | 6.75 | 5.04 | 3.88 |
| pfossil_A | 4.00 | 4.00 | 3.71 | 6.67 |
| pfossil_B | 4.00 | 3.33 | 2.71 | 7.42 |
