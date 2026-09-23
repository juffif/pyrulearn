# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-09-23 21:32:48. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. sonar, ionosphere, kr-vs-kp, mushroom excluded (pass --include-large). `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil`/`aqr` only -- `jrip`/`lord`/`pylord` need no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 95.40 +/- 1.03 | 0/5 |
| lord | 93.56 +/- 1.87 | 0/5 |
| pylord | 94.02 +/- 2.34 | 0/5 |
| pfoil_A | 61.38 +/- 0.56 | 0/5 |
| pfoil_B | 89.20 +/- 1.56 | 0/5 |
| cn2beam1_A | 61.38 +/- 0.56 | 0/5 |
| cn2beam1_B | 95.63 +/- 1.69 | 0/5 |
| cn2beam5_A | 61.38 +/- 0.56 | 0/5 |
| cn2beam5_B | 94.71 +/- 1.72 | 0/5 |
| pfossil_A | 61.38 +/- 0.56 | 0/5 |
| pfossil_B | 94.25 +/- 1.92 | 0/5 |
| aqr_A | 61.38 +/- 0.56 | 0/5 |
| aqr_B | 91.72 +/- 3.20 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.473 |
| lord | 1.343 |
| pylord | 1.305 |
| pfoil_A | 0.493 |
| pfoil_B | 0.124 |
| cn2beam1_A | 0.039 |
| cn2beam1_B | 0.083 |
| cn2beam5_A | 0.091 |
| cn2beam5_B | 0.239 |
| pfossil_A | 0.056 |
| pfossil_B | 0.037 |
| aqr_A | 0.255 |
| aqr_B | 0.234 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.2 | 1.87 |
| lord | 29.2 | 2.94 |
| pylord | 34.6 | 3.05 |
| pfoil_A | 11.6 | 2.44 |
| pfoil_B | 10.8 | 3.38 |
| cn2beam1_A | 5.2 | 2.11 |
| cn2beam1_B | 8.0 | 3.16 |
| cn2beam5_A | 4.8 | 2.15 |
| cn2beam5_B | 7.0 | 3.26 |
| pfossil_A | 5.8 | 2.49 |
| pfossil_B | 2.8 | 3.10 |
| aqr_A | 8.8 | 3.44 |
| aqr_B | 6.4 | 5.98 |

(24.6s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 73.06 +/- 5.36 | 0/5 |
| lord | 71.32 +/- 1.46 | 0/5 |
| pylord | 72.03 +/- 2.15 | 0/5 |
| pfoil_A | 70.28 +/- 0.21 | 0/5 |
| pfoil_B | 52.45 +/- 7.45 | 0/5 |
| cn2beam1_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam1_B | 69.93 +/- 0.78 | 0/5 |
| cn2beam5_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam5_B | 70.63 +/- 0.68 | 0/5 |
| pfossil_A | 70.28 +/- 0.21 | 0/5 |
| pfossil_B | 68.23 +/- 7.84 | 0/5 |
| aqr_A | 70.28 +/- 0.21 | 0/5 |
| aqr_B | 65.70 +/- 6.26 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.470 |
| lord | 1.340 |
| pylord | 2.781 |
| pfoil_A | 0.532 |
| pfoil_B | 0.373 |
| cn2beam1_A | 0.021 |
| cn2beam1_B | 0.010 |
| cn2beam5_A | 0.077 |
| cn2beam5_B | 0.034 |
| pfossil_A | 0.175 |
| pfossil_B | 0.110 |
| aqr_A | 6.878 |
| aqr_B | 4.541 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.0 | 1.80 |
| lord | 81.0 | 4.35 |
| pylord | 84.8 | 4.59 |
| pfoil_A | 22.0 | 3.74 |
| pfoil_B | 17.6 | 3.12 |
| cn2beam1_A | 1.2 | 1.70 |
| cn2beam1_B | 0.2 | 0.20 |
| cn2beam5_A | 2.0 | 1.83 |
| cn2beam5_B | 1.0 | 0.48 |
| pfossil_A | 5.6 | 4.59 |
| pfossil_B | 3.0 | 3.11 |
| aqr_A | 20.6 | 21.81 |
| aqr_B | 13.0 | 23.87 |

(87.6s total)

---

## colic

n=368, attributes=26, A='1', B='2'

*10 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 86.42 +/- 2.97 | 0/5 |
| lord | 81.79 +/- 0.79 | 0/5 |
| pylord | 84.78 +/- 1.32 | 0/5 |
| pfoil_A | 63.04 +/- 0.49 | 0/5 |
| pfoil_B | 77.46 +/- 4.02 | 0/5 |
| cn2beam1_A | 63.04 +/- 0.49 | 0/5 |
| cn2beam1_B | 81.53 +/- 4.29 | 0/5 |
| cn2beam5_A | 63.04 +/- 0.49 | 0/5 |
| cn2beam5_B | 82.34 +/- 4.34 | 0/5 |
| pfossil_A | 63.04 +/- 0.49 | 0/5 |
| pfossil_B | 86.95 +/- 2.97 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.748 |
| lord | 1.642 |
| pylord | 28.851 |
| pfoil_A | 5.804 |
| pfoil_B | 7.159 |
| cn2beam1_A | 2.166 |
| cn2beam1_B | 1.352 |
| cn2beam5_A | 3.427 |
| cn2beam5_B | 2.070 |
| pfossil_A | 3.919 |
| pfossil_B | 2.654 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.2 | 1.97 |
| lord | 62.2 | 1.98 |
| pylord | 68.6 | 1.96 |
| pfoil_A | 15.4 | 3.02 |
| pfoil_B | 16.4 | 2.57 |
| cn2beam1_A | 12.8 | 1.40 |
| cn2beam1_B | 6.0 | 1.68 |
| cn2beam5_A | 13.4 | 1.69 |
| cn2beam5_B | 6.2 | 1.78 |
| pfossil_A | 10.8 | 2.95 |
| pfossil_B | 6.6 | 3.05 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(908.6s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

*10 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 85.65 +/- 2.48 | 0/5 |
| lord | 84.49 +/- 2.58 | 0/5 |
| pylord | 84.06 +/- 2.63 | 0/5 |
| pfoil_A | 80.43 +/- 3.07 | 0/5 |
| pfoil_B | 55.51 +/- 0.35 | 0/5 |
| cn2beam1_A | 74.06 +/- 6.27 | 0/5 |
| cn2beam1_B | 55.51 +/- 0.35 | 0/5 |
| cn2beam5_A | 76.23 +/- 4.41 | 0/5 |
| cn2beam5_B | 55.51 +/- 0.35 | 0/5 |
| pfossil_A | 84.06 +/- 2.51 | 0/5 |
| pfossil_B | 55.51 +/- 0.35 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.718 |
| lord | 1.725 |
| pylord | 25.937 |
| pfoil_A | 3.778 |
| pfoil_B | 4.763 |
| cn2beam1_A | 0.746 |
| cn2beam1_B | 1.366 |
| cn2beam5_A | 1.628 |
| cn2beam5_B | 2.963 |
| pfossil_A | 0.582 |
| pfossil_B | 1.302 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.2 | 2.05 |
| lord | 112.2 | 3.81 |
| pylord | 112.8 | 3.92 |
| pfoil_A | 27.2 | 3.46 |
| pfoil_B | 24.8 | 4.12 |
| cn2beam1_A | 8.4 | 1.96 |
| cn2beam1_B | 12.4 | 2.84 |
| cn2beam5_A | 7.8 | 2.65 |
| cn2beam5_B | 10.2 | 3.02 |
| pfossil_A | 4.4 | 3.11 |
| pfossil_B | 6.8 | 4.70 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(836.0s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*10 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 72.70 +/- 1.94 | 0/5 |
| lord | 74.30 +/- 2.52 | 0/5 |
| pylord | 73.80 +/- 2.93 | 0/5 |
| pfoil_A | 58.40 +/- 4.53 | 0/5 |
| pfoil_B | 70.00 +/- 0.00 | 0/5 |
| cn2beam1_A | 71.90 +/- 0.58 | 0/5 |
| cn2beam1_B | 70.00 +/- 0.00 | 0/5 |
| cn2beam5_A | 71.30 +/- 0.81 | 0/5 |
| cn2beam5_B | 70.00 +/- 0.00 | 0/5 |
| pfossil_A | 71.10 +/- 1.59 | 0/5 |
| pfossil_B | 70.00 +/- 0.00 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.779 |
| lord | 1.669 |
| pylord | 28.037 |
| pfoil_A | 5.385 |
| pfoil_B | 6.498 |
| cn2beam1_A | 0.380 |
| cn2beam1_B | 0.632 |
| cn2beam5_A | 1.739 |
| cn2beam5_B | 2.607 |
| pfossil_A | 0.704 |
| pfossil_B | 0.492 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.0 | 3.77 |
| lord | 223.6 | 3.81 |
| pylord | 226.2 | 3.83 |
| pfoil_A | 42.0 | 4.87 |
| pfoil_B | 47.4 | 4.54 |
| cn2beam1_A | 5.8 | 2.29 |
| cn2beam1_B | 9.0 | 2.64 |
| cn2beam5_A | 8.6 | 2.77 |
| cn2beam5_B | 15.4 | 2.78 |
| pfossil_A | 1.8 | 11.13 |
| pfossil_B | 1.6 | 7.40 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(851.3s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 74.61 +/- 2.13 | 0/5 |
| lord | 73.57 +/- 2.60 | 0/5 |
| pylord | 75.39 +/- 1.60 | 0/5 |
| pfoil_A | 65.10 +/- 0.21 | 0/5 |
| pfoil_B | 59.50 +/- 2.64 | 0/5 |
| cn2beam1_A | 65.10 +/- 0.21 | 0/5 |
| cn2beam1_B | 70.96 +/- 1.89 | 0/5 |
| cn2beam5_A | 65.10 +/- 0.21 | 0/5 |
| cn2beam5_B | 72.14 +/- 2.38 | 0/5 |
| pfossil_A | 65.10 +/- 0.21 | 0/5 |
| pfossil_B | 73.05 +/- 3.54 | 0/5 |
| aqr_A | 65.10 +/- 0.21 | 0/5 |
| aqr_B | 67.05 +/- 1.93 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.588 |
| lord | 1.336 |
| pylord | 10.279 |
| pfoil_A | 1.826 |
| pfoil_B | 1.726 |
| cn2beam1_A | 0.315 |
| cn2beam1_B | 0.225 |
| cn2beam5_A | 1.003 |
| cn2beam5_B | 0.677 |
| pfossil_A | 0.271 |
| pfossil_B | 0.201 |
| aqr_A | 15.967 |
| aqr_B | 9.860 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.0 | 2.68 |
| lord | 178.4 | 4.59 |
| pylord | 196.0 | 5.32 |
| pfoil_A | 39.8 | 4.10 |
| pfoil_B | 32.0 | 4.56 |
| cn2beam1_A | 10.8 | 2.59 |
| cn2beam1_B | 7.6 | 2.74 |
| cn2beam5_A | 10.8 | 2.91 |
| cn2beam5_B | 8.0 | 2.97 |
| pfossil_A | 5.2 | 3.21 |
| pfossil_B | 2.6 | 4.23 |
| aqr_A | 22.8 | 29.14 |
| aqr_B | 13.8 | 29.64 |

(223.3s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.33 +/- 1.01 | 0/5 |
| lord | 97.81 +/- 1.41 | 0/5 |
| pylord | 95.51 +/- 1.57 | 0/5 |
| pfoil_A | 93.62 +/- 6.09 | 0/5 |
| pfoil_B | 65.34 +/- 0.21 | 0/5 |
| cn2beam1_A | 98.33 +/- 1.01 | 0/5 |
| cn2beam1_B | 65.34 +/- 0.21 | 0/5 |
| cn2beam5_A | 98.33 +/- 1.01 | 0/5 |
| cn2beam5_B | 65.34 +/- 0.21 | 0/5 |
| pfossil_A | 80.58 +/- 4.22 | 0/5 |
| pfossil_B | 65.34 +/- 0.21 | 0/5 |
| aqr_A | 91.55 +/- 1.72 | 0/5 |
| aqr_B | 65.34 +/- 0.21 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.531 |
| lord | 1.419 |
| pylord | 5.255 |
| pfoil_A | 0.179 |
| pfoil_B | 0.117 |
| cn2beam1_A | 0.101 |
| cn2beam1_B | 0.071 |
| cn2beam5_A | 0.318 |
| cn2beam5_B | 0.215 |
| pfossil_A | 0.119 |
| pfossil_B | 0.102 |
| aqr_A | 3.152 |
| aqr_B | 3.905 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 8.0 | 3.08 |
| lord | 35.8 | 3.83 |
| pylord | 57.4 | 4.19 |
| pfoil_A | 13.0 | 3.64 |
| pfoil_B | 9.2 | 3.37 |
| cn2beam1_A | 8.4 | 3.00 |
| cn2beam1_B | 5.8 | 3.00 |
| cn2beam5_A | 8.0 | 3.00 |
| cn2beam5_B | 5.0 | 3.03 |
| pfossil_A | 7.6 | 3.74 |
| pfossil_B | 7.2 | 3.47 |
| aqr_A | 36.6 | 10.39 |
| aqr_B | 47.2 | 9.52 |

(78.3s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.61 +/- 0.54 | 0/5 |
| lord | 99.13 +/- 0.37 | 0/5 |
| pylord | 99.13 +/- 0.37 | 0/5 |
| pfoil_A | 55.54 +/- 0.08 | 0/5 |
| pfoil_B | 98.03 +/- 0.59 | 0/5 |
| cn2beam1_A | 55.54 +/- 0.08 | 0/5 |
| cn2beam1_B | 90.45 +/- 2.82 | 0/5 |
| cn2beam5_A | 55.54 +/- 0.08 | 0/5 |
| cn2beam5_B | 94.90 +/- 2.44 | 0/5 |
| pfossil_A | 55.54 +/- 0.08 | 0/5 |
| pfossil_B | 96.87 +/- 1.19 | 0/5 |
| aqr_A | 55.54 +/- 0.08 | 0/5 |
| aqr_B | 98.32 +/- 1.23 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.547 |
| lord | 1.334 |
| pylord | 4.897 |
| pfoil_A | 0.152 |
| pfoil_B | 0.140 |
| cn2beam1_A | 0.066 |
| cn2beam1_B | 0.063 |
| cn2beam5_A | 0.295 |
| cn2beam5_B | 0.241 |
| pfossil_A | 0.059 |
| pfossil_B | 0.077 |
| aqr_A | 0.747 |
| aqr_B | 1.079 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.2 | 2.30 |
| lord | 26.2 | 2.78 |
| pylord | 27.0 | 2.79 |
| pfoil_A | 13.6 | 2.94 |
| pfoil_B | 11.6 | 3.02 |
| cn2beam1_A | 6.4 | 2.32 |
| cn2beam1_B | 6.8 | 2.18 |
| cn2beam5_A | 6.6 | 2.49 |
| cn2beam5_B | 6.8 | 2.37 |
| pfossil_A | 6.2 | 2.19 |
| pfossil_B | 6.8 | 2.44 |
| aqr_A | 5.2 | 9.24 |
| aqr_B | 7.6 | 10.41 |

(49.7s total)

---

## Overall summary

### Accuracy

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 95.40 | 93.56 | 94.02 | 61.38 | 89.20 | 61.38 | 95.63 | 61.38 | 94.71 | 61.38 | 94.25 | 61.38 | 91.72 |
| breast-cancer | 73.06 | 71.32 | 72.03 | 70.28 | 52.45 | 70.28 | 69.93 | 70.28 | 70.63 | 70.28 | 68.23 | 70.28 | 65.70 |
| colic | 86.42 | 81.79 | 84.78 | 63.04 | 77.46 | 63.04 | 81.53 | 63.04 | 82.34 | 63.04 | 86.95 | nan | nan |
| credit-approval | 85.65 | 84.49 | 84.06 | 80.43 | 55.51 | 74.06 | 55.51 | 76.23 | 55.51 | 84.06 | 55.51 | nan | nan |
| credit-g | 72.70 | 74.30 | 73.80 | 58.40 | 70.00 | 71.90 | 70.00 | 71.30 | 70.00 | 71.10 | 70.00 | nan | nan |
| diabetes | 74.61 | 73.57 | 75.39 | 65.10 | 59.50 | 65.10 | 70.96 | 65.10 | 72.14 | 65.10 | 73.05 | 65.10 | 67.05 |
| tic-tac-toe | 98.33 | 97.81 | 95.51 | 93.62 | 65.34 | 98.33 | 65.34 | 98.33 | 65.34 | 80.58 | 65.34 | 91.55 | 65.34 |
| banknote-authentication | 98.61 | 99.13 | 99.13 | 55.54 | 98.03 | 55.54 | 90.45 | 55.54 | 94.90 | 55.54 | 96.87 | 55.54 | 98.32 |

### Fit time (seconds/fold)

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.473 | 1.343 | 1.305 | 0.493 | 0.124 | 0.039 | 0.083 | 0.091 | 0.239 | 0.056 | 0.037 | 0.255 | 0.234 |
| breast-cancer | 0.470 | 1.340 | 2.781 | 0.532 | 0.373 | 0.021 | 0.010 | 0.077 | 0.034 | 0.175 | 0.110 | 6.878 | 4.541 |
| colic | 0.748 | 1.642 | 28.851 | 5.804 | 7.159 | 2.166 | 1.352 | 3.427 | 2.070 | 3.919 | 2.654 | nan | nan |
| credit-approval | 0.718 | 1.725 | 25.937 | 3.778 | 4.763 | 0.746 | 1.366 | 1.628 | 2.963 | 0.582 | 1.302 | nan | nan |
| credit-g | 0.779 | 1.669 | 28.037 | 5.385 | 6.498 | 0.380 | 0.632 | 1.739 | 2.607 | 0.704 | 0.492 | nan | nan |
| diabetes | 0.588 | 1.336 | 10.279 | 1.826 | 1.726 | 0.315 | 0.225 | 1.003 | 0.677 | 0.271 | 0.201 | 15.967 | 9.860 |
| tic-tac-toe | 0.531 | 1.419 | 5.255 | 0.179 | 0.117 | 0.101 | 0.071 | 0.318 | 0.215 | 0.119 | 0.102 | 3.152 | 3.905 |
| banknote-authentication | 0.547 | 1.334 | 4.897 | 0.152 | 0.140 | 0.066 | 0.063 | 0.295 | 0.241 | 0.059 | 0.077 | 0.747 | 1.079 |

### Rule count

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 2.2 | 29.2 | 34.6 | 11.6 | 10.8 | 5.2 | 8.0 | 4.8 | 7.0 | 5.8 | 2.8 | 8.8 | 6.4 |
| breast-cancer | 1.0 | 81.0 | 84.8 | 22.0 | 17.6 | 1.2 | 0.2 | 2.0 | 1.0 | 5.6 | 3.0 | 20.6 | 13.0 |
| colic | 3.2 | 62.2 | 68.6 | 15.4 | 16.4 | 12.8 | 6.0 | 13.4 | 6.2 | 10.8 | 6.6 | nan | nan |
| credit-approval | 3.2 | 112.2 | 112.8 | 27.2 | 24.8 | 8.4 | 12.4 | 7.8 | 10.2 | 4.4 | 6.8 | nan | nan |
| credit-g | 3.0 | 223.6 | 226.2 | 42.0 | 47.4 | 5.8 | 9.0 | 8.6 | 15.4 | 1.8 | 1.6 | nan | nan |
| diabetes | 3.0 | 178.4 | 196.0 | 39.8 | 32.0 | 10.8 | 7.6 | 10.8 | 8.0 | 5.2 | 2.6 | 22.8 | 13.8 |
| tic-tac-toe | 8.0 | 35.8 | 57.4 | 13.0 | 9.2 | 8.4 | 5.8 | 8.0 | 5.0 | 7.6 | 7.2 | 36.6 | 47.2 |
| banknote-authentication | 5.2 | 26.2 | 27.0 | 13.6 | 11.6 | 6.4 | 6.8 | 6.6 | 6.8 | 6.2 | 6.8 | 5.2 | 7.6 |

### Average conditions per rule

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 1.87 | 2.94 | 3.05 | 2.44 | 3.38 | 2.11 | 3.16 | 2.15 | 3.26 | 2.49 | 3.10 | 3.44 | 5.98 |
| breast-cancer | 1.80 | 4.35 | 4.59 | 3.74 | 3.12 | 1.70 | 0.20 | 1.83 | 0.48 | 4.59 | 3.11 | 21.81 | 23.87 |
| colic | 1.97 | 1.98 | 1.96 | 3.02 | 2.57 | 1.40 | 1.68 | 1.69 | 1.78 | 2.95 | 3.05 | nan | nan |
| credit-approval | 2.05 | 3.81 | 3.92 | 3.46 | 4.12 | 1.96 | 2.84 | 2.65 | 3.02 | 3.11 | 4.70 | nan | nan |
| credit-g | 3.77 | 3.81 | 3.83 | 4.87 | 4.54 | 2.29 | 2.64 | 2.77 | 2.78 | 11.13 | 7.40 | nan | nan |
| diabetes | 2.68 | 4.59 | 5.32 | 4.10 | 4.56 | 2.59 | 2.74 | 2.91 | 2.97 | 3.21 | 4.23 | 29.14 | 29.64 |
| tic-tac-toe | 3.08 | 3.83 | 4.19 | 3.64 | 3.37 | 3.00 | 3.00 | 3.00 | 3.03 | 3.74 | 3.47 | 10.39 | 9.52 |
| banknote-authentication | 2.30 | 2.78 | 2.79 | 2.94 | 3.02 | 2.32 | 2.18 | 2.49 | 2.37 | 2.19 | 2.44 | 9.24 | 10.41 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 85.60 | 0.607 | 3.6 | 2.44 | 0 |
| lord | 84.50 | 1.476 | 93.6 | 3.51 | 0 |
| pylord | 84.84 | 13.418 | 100.9 | 3.71 | 0 |
| pfoil_A | 68.48 | 2.269 | 23.1 | 3.53 | 0 |
| pfoil_B | 70.94 | 2.612 | 21.2 | 3.58 | 0 |
| cn2beam1_A | 69.95 | 0.479 | 7.4 | 2.17 | 0 |
| cn2beam1_B | 74.92 | 0.475 | 7.0 | 2.30 | 0 |
| cn2beam5_A | 70.15 | 1.072 | 7.8 | 2.44 | 0 |
| cn2beam5_B | 75.70 | 1.131 | 7.5 | 2.46 | 0 |
| pfossil_A | 68.89 | 0.736 | 5.9 | 4.18 | 0 |
| pfossil_B | 76.27 | 0.622 | 4.7 | 3.94 | 0 |
| aqr_A | 68.77 | 5.400 | 18.8 | 14.80 | 3 |
| aqr_B | 77.63 | 3.924 | 17.6 | 15.88 | 3 |

**Average rank per criterion** (1 = best of 13; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 2.00 | 6.12 | 2.19 | 3.50 |
| lord | 3.19 | 8.62 | 11.00 | 8.00 |
| pylord | 2.88 | 11.75 | 12.25 | 8.62 |
| pfoil_A | 8.81 | 8.62 | 9.75 | 7.88 |
| pfoil_B | 9.38 | 7.62 | 9.12 | 8.75 |
| cn2beam1_A | 7.69 | 2.75 | 5.06 | 1.88 |
| cn2beam1_B | 7.50 | 2.50 | 4.62 | 3.00 |
| cn2beam5_A | 7.69 | 6.50 | 5.12 | 3.75 |
| cn2beam5_B | 6.50 | 6.38 | 4.81 | 4.88 |
| pfossil_A | 8.25 | 3.88 | 3.88 | 7.38 |
| pfossil_B | 6.88 | 3.25 | 3.38 | 8.38 |
| aqr_A | 10.44 | 11.69 | 9.75 | 12.31 |
| aqr_B | 9.81 | 11.31 | 10.06 | 12.69 |
