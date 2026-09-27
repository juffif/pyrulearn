# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-09-27 20:16:36. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. sonar, ionosphere, kr-vs-kp, mushroom excluded (pass --include-large). `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil`/`aqr` only -- `jrip`/`lord`/`pylord` need no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

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
| aqr_B | 92.87 +/- 2.66 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.754 |
| lord | 1.500 |
| pylord | 0.636 |
| pfoil_A | 0.239 |
| pfoil_B | 0.054 |
| cn2beam1_A | 0.020 |
| cn2beam1_B | 0.038 |
| cn2beam5_A | 0.044 |
| cn2beam5_B | 0.107 |
| pfossil_A | 0.028 |
| pfossil_B | 0.017 |
| aqr_A | 0.171 |
| aqr_B | 0.192 |

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
| aqr_A | 11.8 | 3.64 |
| aqr_B | 12.0 | 5.51 |

(19.4s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 73.06 +/- 5.36 | 0/5 |
| lord | 71.68 +/- 1.75 | 0/5 |
| pylord | 72.38 +/- 1.60 | 0/5 |
| pfoil_A | 70.28 +/- 0.21 | 0/5 |
| pfoil_B | 52.45 +/- 7.45 | 0/5 |
| cn2beam1_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam1_B | 69.93 +/- 0.78 | 0/5 |
| cn2beam5_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam5_B | 70.63 +/- 0.68 | 0/5 |
| pfossil_A | 70.28 +/- 0.21 | 0/5 |
| pfossil_B | 68.23 +/- 7.84 | 0/5 |
| aqr_A | 70.28 +/- 0.21 | 0/5 |
| aqr_B | 63.30 +/- 5.75 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.445 |
| lord | 1.558 |
| pylord | 2.111 |
| pfoil_A | 0.398 |
| pfoil_B | 0.280 |
| cn2beam1_A | 0.016 |
| cn2beam1_B | 0.007 |
| cn2beam5_A | 0.056 |
| cn2beam5_B | 0.024 |
| pfossil_A | 0.133 |
| pfossil_B | 0.078 |
| aqr_A | 7.095 |
| aqr_B | 6.956 |

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
| aqr_A | 26.0 | 20.72 |
| aqr_B | 25.4 | 22.29 |

(96.4s total)

---

## colic

n=368, attributes=26, A='1', B='2'

*10 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 86.42 +/- 2.97 | 0/5 |
| lord | 81.79 +/- 0.79 | 0/5 |
| pylord | 85.05 +/- 1.22 | 0/5 |
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
| jrip | 0.727 |
| lord | 1.390 |
| pylord | 12.721 |
| pfoil_A | 3.431 |
| pfoil_B | 3.901 |
| cn2beam1_A | 1.256 |
| cn2beam1_B | 0.694 |
| cn2beam5_A | 1.954 |
| cn2beam5_B | 1.138 |
| pfossil_A | 2.164 |
| pfossil_B | 1.392 |
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

(761.7s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

*5 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

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
| aqr_A | 74.88 +/- 4.03 | 2/5 |
| aqr_B | 55.07 +/- 0.00 | 3/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.611 |
| lord | 1.474 |
| pylord | 13.034 |
| pfoil_A | 1.630 |
| pfoil_B | 2.154 |
| cn2beam1_A | 0.299 |
| cn2beam1_B | 0.591 |
| cn2beam5_A | 0.730 |
| cn2beam5_B | 1.374 |
| pfossil_A | 0.283 |
| pfossil_B | 0.539 |
| aqr_A | 52.627 |
| aqr_B | 49.134 |

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
| aqr_A | 36.3 | 33.53 |
| aqr_B | 34.0 | 35.68 |

(674.3s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*10 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 72.70 +/- 1.94 | 0/5 |
| lord | 74.30 +/- 2.44 | 0/5 |
| pylord | 73.90 +/- 2.91 | 0/5 |
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
| jrip | 0.988 |
| lord | 1.691 |
| pylord | 29.883 |
| pfoil_A | 4.646 |
| pfoil_B | 6.325 |
| cn2beam1_A | 0.321 |
| cn2beam1_B | 0.646 |
| cn2beam5_A | 1.475 |
| cn2beam5_B | 2.478 |
| pfossil_A | 0.580 |
| pfossil_B | 0.501 |
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

(856.9s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

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
| aqr_A | 65.04 +/- 0.18 | 1/5 |
| aqr_B | 70.70 +/- 4.44 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.601 |
| lord | 1.428 |
| pylord | 11.459 |
| pfoil_A | 1.935 |
| pfoil_B | 1.919 |
| cn2beam1_A | 0.320 |
| cn2beam1_B | 0.227 |
| cn2beam5_A | 1.113 |
| cn2beam5_B | 0.656 |
| pfossil_A | 0.256 |
| pfossil_B | 0.186 |
| aqr_A | 50.146 |
| aqr_B | 48.259 |

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
| aqr_A | 72.5 | 26.13 |
| aqr_B | 66.4 | 26.63 |

(604.9s total)

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
| jrip | 0.564 |
| lord | 1.426 |
| pylord | 6.625 |
| pfoil_A | 0.222 |
| pfoil_B | 0.146 |
| cn2beam1_A | 0.126 |
| cn2beam1_B | 0.088 |
| cn2beam5_A | 0.418 |
| cn2beam5_B | 0.257 |
| pfossil_A | 0.145 |
| pfossil_B | 0.125 |
| aqr_A | 3.926 |
| aqr_B | 4.907 |

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

(96.0s total)

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
| aqr_B | 98.98 +/- 0.81 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.533 |
| lord | 1.326 |
| pylord | 6.239 |
| pfoil_A | 0.197 |
| pfoil_B | 0.186 |
| cn2beam1_A | 0.081 |
| cn2beam1_B | 0.084 |
| cn2beam5_A | 0.369 |
| cn2beam5_B | 0.355 |
| pfossil_A | 0.077 |
| pfossil_B | 0.098 |
| aqr_A | 2.189 |
| aqr_B | 2.153 |

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
| aqr_A | 10.6 | 9.55 |
| aqr_B | 10.4 | 9.84 |

(71.0s total)

---

## Overall summary

### Accuracy

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 95.40 | 93.56 | 94.02 | 61.38 | 89.20 | 61.38 | 95.63 | 61.38 | 94.71 | 61.38 | 94.25 | 61.38 | 92.87 |
| breast-cancer | 73.06 | 71.68 | 72.38 | 70.28 | 52.45 | 70.28 | 69.93 | 70.28 | 70.63 | 70.28 | 68.23 | 70.28 | 63.30 |
| colic | 86.42 | 81.79 | 85.05 | 63.04 | 77.46 | 63.04 | 81.53 | 63.04 | 82.34 | 63.04 | 86.95 | nan | nan |
| credit-approval | 85.65 | 84.49 | 84.06 | 80.43 | 55.51 | 74.06 | 55.51 | 76.23 | 55.51 | 84.06 | 55.51 | 74.88 | 55.07 |
| credit-g | 72.70 | 74.30 | 73.90 | 58.40 | 70.00 | 71.90 | 70.00 | 71.30 | 70.00 | 71.10 | 70.00 | nan | nan |
| diabetes | 74.61 | 73.57 | 75.39 | 65.10 | 59.50 | 65.10 | 70.96 | 65.10 | 72.14 | 65.10 | 73.05 | 65.04 | 70.70 |
| tic-tac-toe | 98.33 | 97.81 | 95.51 | 93.62 | 65.34 | 98.33 | 65.34 | 98.33 | 65.34 | 80.58 | 65.34 | 91.55 | 65.34 |
| banknote-authentication | 98.61 | 99.13 | 99.13 | 55.54 | 98.03 | 55.54 | 90.45 | 55.54 | 94.90 | 55.54 | 96.87 | 55.54 | 98.98 |

### Fit time (seconds/fold)

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.754 | 1.500 | 0.636 | 0.239 | 0.054 | 0.020 | 0.038 | 0.044 | 0.107 | 0.028 | 0.017 | 0.171 | 0.192 |
| breast-cancer | 0.445 | 1.558 | 2.111 | 0.398 | 0.280 | 0.016 | 0.007 | 0.056 | 0.024 | 0.133 | 0.078 | 7.095 | 6.956 |
| colic | 0.727 | 1.390 | 12.721 | 3.431 | 3.901 | 1.256 | 0.694 | 1.954 | 1.138 | 2.164 | 1.392 | nan | nan |
| credit-approval | 0.611 | 1.474 | 13.034 | 1.630 | 2.154 | 0.299 | 0.591 | 0.730 | 1.374 | 0.283 | 0.539 | 52.627 | 49.134 |
| credit-g | 0.988 | 1.691 | 29.883 | 4.646 | 6.325 | 0.321 | 0.646 | 1.475 | 2.478 | 0.580 | 0.501 | nan | nan |
| diabetes | 0.601 | 1.428 | 11.459 | 1.935 | 1.919 | 0.320 | 0.227 | 1.113 | 0.656 | 0.256 | 0.186 | 50.146 | 48.259 |
| tic-tac-toe | 0.564 | 1.426 | 6.625 | 0.222 | 0.146 | 0.126 | 0.088 | 0.418 | 0.257 | 0.145 | 0.125 | 3.926 | 4.907 |
| banknote-authentication | 0.533 | 1.326 | 6.239 | 0.197 | 0.186 | 0.081 | 0.084 | 0.369 | 0.355 | 0.077 | 0.098 | 2.189 | 2.153 |

### Rule count

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 2.2 | 29.2 | 34.6 | 11.6 | 10.8 | 5.2 | 8.0 | 4.8 | 7.0 | 5.8 | 2.8 | 11.8 | 12.0 |
| breast-cancer | 1.0 | 81.0 | 84.8 | 22.0 | 17.6 | 1.2 | 0.2 | 2.0 | 1.0 | 5.6 | 3.0 | 26.0 | 25.4 |
| colic | 3.2 | 62.2 | 68.6 | 15.4 | 16.4 | 12.8 | 6.0 | 13.4 | 6.2 | 10.8 | 6.6 | nan | nan |
| credit-approval | 3.2 | 112.2 | 112.8 | 27.2 | 24.8 | 8.4 | 12.4 | 7.8 | 10.2 | 4.4 | 6.8 | 36.3 | 34.0 |
| credit-g | 3.0 | 223.6 | 226.2 | 42.0 | 47.4 | 5.8 | 9.0 | 8.6 | 15.4 | 1.8 | 1.6 | nan | nan |
| diabetes | 3.0 | 178.4 | 196.0 | 39.8 | 32.0 | 10.8 | 7.6 | 10.8 | 8.0 | 5.2 | 2.6 | 72.5 | 66.4 |
| tic-tac-toe | 8.0 | 35.8 | 57.4 | 13.0 | 9.2 | 8.4 | 5.8 | 8.0 | 5.0 | 7.6 | 7.2 | 36.6 | 47.2 |
| banknote-authentication | 5.2 | 26.2 | 27.0 | 13.6 | 11.6 | 6.4 | 6.8 | 6.6 | 6.8 | 6.2 | 6.8 | 10.6 | 10.4 |

### Average conditions per rule

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 1.87 | 2.94 | 3.05 | 2.44 | 3.38 | 2.11 | 3.16 | 2.15 | 3.26 | 2.49 | 3.10 | 3.64 | 5.51 |
| breast-cancer | 1.80 | 4.35 | 4.59 | 3.74 | 3.12 | 1.70 | 0.20 | 1.83 | 0.48 | 4.59 | 3.11 | 20.72 | 22.29 |
| colic | 1.97 | 1.98 | 1.96 | 3.02 | 2.57 | 1.40 | 1.68 | 1.69 | 1.78 | 2.95 | 3.05 | nan | nan |
| credit-approval | 2.05 | 3.81 | 3.92 | 3.46 | 4.12 | 1.96 | 2.84 | 2.65 | 3.02 | 3.11 | 4.70 | 33.53 | 35.68 |
| credit-g | 3.77 | 3.81 | 3.83 | 4.87 | 4.54 | 2.29 | 2.64 | 2.77 | 2.78 | 11.13 | 7.40 | nan | nan |
| diabetes | 2.68 | 4.59 | 5.32 | 4.10 | 4.56 | 2.59 | 2.74 | 2.91 | 2.97 | 3.21 | 4.23 | 26.13 | 26.63 |
| tic-tac-toe | 3.08 | 3.83 | 4.19 | 3.64 | 3.37 | 3.00 | 3.00 | 3.00 | 3.03 | 3.74 | 3.47 | 10.39 | 9.52 |
| banknote-authentication | 2.30 | 2.78 | 2.79 | 2.94 | 3.02 | 2.32 | 2.18 | 2.49 | 2.37 | 2.19 | 2.44 | 9.55 | 9.84 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 85.60 | 0.653 | 3.6 | 2.44 | 0 |
| lord | 84.54 | 1.474 | 93.6 | 3.51 | 0 |
| pylord | 84.93 | 10.338 | 100.9 | 3.71 | 0 |
| pfoil_A | 68.48 | 1.587 | 23.1 | 3.53 | 0 |
| pfoil_B | 70.94 | 1.871 | 21.2 | 3.58 | 0 |
| cn2beam1_A | 69.95 | 0.305 | 7.4 | 2.17 | 0 |
| cn2beam1_B | 74.92 | 0.297 | 7.0 | 2.30 | 0 |
| cn2beam5_A | 70.15 | 0.770 | 7.8 | 2.44 | 0 |
| cn2beam5_B | 75.70 | 0.799 | 7.5 | 2.46 | 0 |
| pfossil_A | 68.89 | 0.458 | 5.9 | 4.18 | 0 |
| pfossil_B | 76.27 | 0.367 | 4.7 | 3.94 | 0 |
| aqr_A | 69.78 | 19.359 | 32.3 | 17.32 | 2 |
| aqr_B | 74.38 | 18.600 | 32.6 | 18.24 | 2 |

**Average rank per criterion** (1 = best of 13; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 2.12 | 7.00 | 2.12 | 3.50 |
| lord | 3.19 | 8.88 | 11.25 | 8.00 |
| pylord | 2.88 | 11.50 | 12.50 | 8.62 |
| pfoil_A | 8.75 | 8.38 | 9.00 | 7.88 |
| pfoil_B | 9.50 | 7.75 | 8.50 | 8.75 |
| cn2beam1_A | 7.75 | 2.50 | 4.94 | 1.88 |
| cn2beam1_B | 7.62 | 2.50 | 4.38 | 3.00 |
| cn2beam5_A | 7.62 | 6.38 | 5.00 | 3.75 |
| cn2beam5_B | 6.62 | 6.00 | 4.56 | 4.88 |
| pfossil_A | 8.19 | 3.62 | 3.75 | 7.38 |
| pfossil_B | 7.00 | 3.00 | 3.25 | 8.38 |
| aqr_A | 10.00 | 11.88 | 11.00 | 12.25 |
| aqr_B | 9.75 | 11.62 | 10.75 | 12.75 |
