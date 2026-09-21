# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-09-16 08:21:26. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. Only the first 1 of 5 fold(s) actually run (preview mode). sonar, ionosphere, kr-vs-kp, mushroom excluded (pass --include-large). `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil`/`aqr` only -- `jrip`/`lord`/`pylord` need no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 91.95 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 89.66 +/- 0.00 | 0/1 |
| pfoil_A | 60.92 +/- 0.00 | 0/1 |
| pfoil_B | 93.10 +/- 0.00 | 0/1 |
| cn2beam1_A | 60.92 +/- 0.00 | 0/1 |
| cn2beam1_B | 91.95 +/- 0.00 | 0/1 |
| cn2beam5_A | 60.92 +/- 0.00 | 0/1 |
| cn2beam5_B | 91.95 +/- 0.00 | 0/1 |
| pfossil_A | 60.92 +/- 0.00 | 0/1 |
| pfossil_B | 89.66 +/- 0.00 | 0/1 |
| aqr_A | 60.92 +/- 0.00 | 0/1 |
| aqr_B | 87.36 +/- 0.00 | 0/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.372 |
| lord | nan |
| pylord | 3.435 |
| pfoil_A | 1.386 |
| pfoil_B | 0.166 |
| cn2beam1_A | 0.063 |
| cn2beam1_B | 0.232 |
| cn2beam5_A | 0.187 |
| cn2beam5_B | 0.528 |
| pfossil_A | 0.174 |
| pfossil_B | 0.065 |
| aqr_A | 2.452 |
| aqr_B | 2.781 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.0 | 1.00 |
| lord | nan | nan |
| pylord | 22.0 | 2.86 |
| pfoil_A | 8.0 | 2.38 |
| pfoil_B | 6.0 | 3.33 |
| cn2beam1_A | 3.0 | 2.00 |
| cn2beam1_B | 10.0 | 2.50 |
| cn2beam5_A | 3.0 | 2.00 |
| cn2beam5_B | 8.0 | 2.88 |
| pfossil_A | 7.0 | 3.00 |
| pfossil_B | 3.0 | 2.33 |
| aqr_A | 9.0 | 7.33 |
| aqr_B | 9.0 | 13.11 |

(12.0s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 79.31 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 70.69 +/- 0.00 | 0/1 |
| pfoil_A | 70.69 +/- 0.00 | 0/1 |
| pfoil_B | 41.38 +/- 0.00 | 0/1 |
| cn2beam1_A | 70.69 +/- 0.00 | 0/1 |
| cn2beam1_B | 70.69 +/- 0.00 | 0/1 |
| cn2beam5_A | 70.69 +/- 0.00 | 0/1 |
| cn2beam5_B | 70.69 +/- 0.00 | 0/1 |
| pfossil_A | 70.69 +/- 0.00 | 0/1 |
| pfossil_B | 46.55 +/- 0.00 | 0/1 |
| aqr_A | 70.69 +/- 0.00 | 0/1 |
| aqr_B | 74.14 +/- 0.00 | 0/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.360 |
| lord | nan |
| pylord | 2.630 |
| pfoil_A | 0.467 |
| pfoil_B | 0.312 |
| cn2beam1_A | 0.014 |
| cn2beam1_B | 0.007 |
| cn2beam5_A | 0.046 |
| cn2beam5_B | 0.007 |
| pfossil_A | 0.240 |
| pfossil_B | 0.121 |
| aqr_A | 7.373 |
| aqr_B | 3.355 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.0 | 2.00 |
| lord | nan | nan |
| pylord | 81.0 | 5.25 |
| pfoil_A | 21.0 | 3.67 |
| pfoil_B | 15.0 | 3.20 |
| cn2beam1_A | 1.0 | 1.00 |
| cn2beam1_B | 0.0 | 0.00 |
| cn2beam5_A | 1.0 | 1.00 |
| cn2beam5_B | 0.0 | 0.00 |
| pfossil_A | 9.0 | 4.44 |
| pfossil_B | 5.0 | 3.80 |
| aqr_A | 21.0 | 22.33 |
| aqr_B | 9.0 | 26.22 |

(15.0s total)

---

## colic

n=368, attributes=26, A='1', B='2'

*3 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 86.49 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 81.08 +/- 0.00 | 0/1 |
| pfoil_A | 62.16 +/- 0.00 | 0/1 |
| pfoil_B | 68.92 +/- 0.00 | 0/1 |
| cn2beam1_A | 62.16 +/- 0.00 | 0/1 |
| cn2beam1_B | 74.32 +/- 0.00 | 0/1 |
| cn2beam5_A | 62.16 +/- 0.00 | 0/1 |
| cn2beam5_B | 74.32 +/- 0.00 | 0/1 |
| pfossil_A | 62.16 +/- 0.00 | 0/1 |
| pfossil_B | 81.08 +/- 0.00 | 0/1 |
| aqr_A | n/a | 1/1 |
| aqr_B | n/a | 1/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.483 |
| lord | nan |
| pylord | 18.221 |
| pfoil_A | 3.888 |
| pfoil_B | 4.189 |
| cn2beam1_A | 2.647 |
| cn2beam1_B | 0.994 |
| cn2beam5_A | 2.984 |
| cn2beam5_B | 0.897 |
| pfossil_A | 2.896 |
| pfossil_B | 1.702 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.0 | 1.33 |
| lord | nan | nan |
| pylord | 65.0 | 1.95 |
| pfoil_A | 16.0 | 2.81 |
| pfoil_B | 12.0 | 3.08 |
| cn2beam1_A | 27.0 | 1.22 |
| cn2beam1_B | 7.0 | 1.71 |
| cn2beam5_A | 20.0 | 1.55 |
| cn2beam5_B | 5.0 | 1.80 |
| pfossil_A | 10.0 | 3.60 |
| pfossil_B | 4.0 | 4.75 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(159.7s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

*3 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 81.16 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 81.16 +/- 0.00 | 0/1 |
| pfoil_A | 78.26 +/- 0.00 | 0/1 |
| pfoil_B | 55.07 +/- 0.00 | 0/1 |
| cn2beam1_A | 65.94 +/- 0.00 | 0/1 |
| cn2beam1_B | 55.07 +/- 0.00 | 0/1 |
| cn2beam5_A | 72.46 +/- 0.00 | 0/1 |
| cn2beam5_B | 55.07 +/- 0.00 | 0/1 |
| pfossil_A | 79.71 +/- 0.00 | 0/1 |
| pfossil_B | 55.07 +/- 0.00 | 0/1 |
| aqr_A | n/a | 1/1 |
| aqr_B | n/a | 1/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.473 |
| lord | nan |
| pylord | 15.175 |
| pfoil_A | 1.743 |
| pfoil_B | 2.819 |
| cn2beam1_A | 0.293 |
| cn2beam1_B | 0.681 |
| cn2beam5_A | 0.954 |
| cn2beam5_B | 1.457 |
| pfossil_A | 0.335 |
| pfossil_B | 1.027 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.0 | 1.00 |
| lord | nan | nan |
| pylord | 111.0 | 3.82 |
| pfoil_A | 25.0 | 3.32 |
| pfoil_B | 19.0 | 4.05 |
| cn2beam1_A | 7.0 | 1.86 |
| cn2beam1_B | 12.0 | 2.42 |
| cn2beam5_A | 8.0 | 3.00 |
| cn2beam5_B | 11.0 | 2.82 |
| pfossil_A | 3.0 | 4.33 |
| pfossil_B | 13.0 | 3.92 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(145.6s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*3 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 72.50 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 73.00 +/- 0.00 | 0/1 |
| pfoil_A | 60.50 +/- 0.00 | 0/1 |
| pfoil_B | 70.00 +/- 0.00 | 0/1 |
| cn2beam1_A | 72.50 +/- 0.00 | 0/1 |
| cn2beam1_B | 70.00 +/- 0.00 | 0/1 |
| cn2beam5_A | 71.00 +/- 0.00 | 0/1 |
| cn2beam5_B | 70.00 +/- 0.00 | 0/1 |
| pfossil_A | 69.50 +/- 0.00 | 0/1 |
| pfossil_B | 70.00 +/- 0.00 | 0/1 |
| aqr_A | n/a | 1/1 |
| aqr_B | n/a | 1/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.629 |
| lord | nan |
| pylord | 26.504 |
| pfoil_A | 5.248 |
| pfoil_B | 5.882 |
| cn2beam1_A | 0.332 |
| cn2beam1_B | 0.573 |
| cn2beam5_A | 2.184 |
| cn2beam5_B | 3.664 |
| pfossil_A | 0.848 |
| pfossil_B | 0.387 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.0 | 4.00 |
| lord | nan | nan |
| pylord | 227.0 | 3.70 |
| pfoil_A | 45.0 | 5.20 |
| pfoil_B | 46.0 | 4.50 |
| cn2beam1_A | 6.0 | 2.33 |
| cn2beam1_B | 10.0 | 2.30 |
| cn2beam5_A | 14.0 | 2.57 |
| cn2beam5_B | 27.0 | 2.74 |
| pfossil_A | 3.0 | 10.67 |
| pfossil_B | 1.0 | 8.00 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(167.0s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 72.08 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 73.38 +/- 0.00 | 0/1 |
| pfoil_A | 64.94 +/- 0.00 | 0/1 |
| pfoil_B | 59.09 +/- 0.00 | 0/1 |
| cn2beam1_A | 64.94 +/- 0.00 | 0/1 |
| cn2beam1_B | 72.73 +/- 0.00 | 0/1 |
| cn2beam5_A | 64.94 +/- 0.00 | 0/1 |
| cn2beam5_B | 74.03 +/- 0.00 | 0/1 |
| pfossil_A | 64.94 +/- 0.00 | 0/1 |
| pfossil_B | 73.38 +/- 0.00 | 0/1 |
| aqr_A | 64.94 +/- 0.00 | 0/1 |
| aqr_B | 68.83 +/- 0.00 | 0/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.528 |
| lord | nan |
| pylord | 10.812 |
| pfoil_A | 1.689 |
| pfoil_B | 1.469 |
| cn2beam1_A | 0.488 |
| cn2beam1_B | 0.218 |
| cn2beam5_A | 1.007 |
| cn2beam5_B | 0.695 |
| pfossil_A | 0.492 |
| pfossil_B | 0.195 |
| aqr_A | 3.467 |
| aqr_B | 23.190 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.0 | 2.50 |
| lord | nan | nan |
| pylord | 181.0 | 5.17 |
| pfoil_A | 40.0 | 4.30 |
| pfoil_B | 32.0 | 4.66 |
| cn2beam1_A | 18.0 | 2.72 |
| cn2beam1_B | 8.0 | 2.75 |
| cn2beam5_A | 12.0 | 3.33 |
| cn2beam5_B | 7.0 | 3.43 |
| pfossil_A | 11.0 | 4.45 |
| pfossil_B | 4.0 | 3.75 |
| aqr_A | 3.0 | 35.33 |
| aqr_B | 28.0 | 32.71 |

(44.5s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.44 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 95.83 +/- 0.00 | 0/1 |
| pfoil_A | 96.35 +/- 0.00 | 0/1 |
| pfoil_B | 65.62 +/- 0.00 | 0/1 |
| cn2beam1_A | 98.44 +/- 0.00 | 0/1 |
| cn2beam1_B | 65.62 +/- 0.00 | 0/1 |
| cn2beam5_A | 98.44 +/- 0.00 | 0/1 |
| cn2beam5_B | 65.62 +/- 0.00 | 0/1 |
| pfossil_A | 82.81 +/- 0.00 | 0/1 |
| pfossil_B | 65.62 +/- 0.00 | 0/1 |
| aqr_A | 94.27 +/- 0.00 | 0/1 |
| aqr_B | 65.62 +/- 0.00 | 0/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.445 |
| lord | nan |
| pylord | 4.940 |
| pfoil_A | 0.177 |
| pfoil_B | 0.097 |
| cn2beam1_A | 0.101 |
| cn2beam1_B | 0.070 |
| cn2beam5_A | 0.307 |
| cn2beam5_B | 0.116 |
| pfossil_A | 0.130 |
| pfossil_B | 0.128 |
| aqr_A | 2.370 |
| aqr_B | 3.849 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 8.0 | 3.38 |
| lord | nan | nan |
| pylord | 45.0 | 4.04 |
| pfoil_A | 14.0 | 3.64 |
| pfoil_B | 8.0 | 3.25 |
| cn2beam1_A | 9.0 | 3.00 |
| cn2beam1_B | 6.0 | 3.00 |
| cn2beam5_A | 8.0 | 3.00 |
| cn2beam5_B | 3.0 | 3.00 |
| pfossil_A | 8.0 | 3.50 |
| pfossil_B | 9.0 | 3.67 |
| aqr_A | 30.0 | 10.13 |
| aqr_B | 49.0 | 9.49 |

(12.9s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

*1 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.18 +/- 0.00 | 0/1 |
| lord | n/a | 1/1 |
| pylord | 98.55 +/- 0.00 | 0/1 |
| pfoil_A | 55.64 +/- 0.00 | 0/1 |
| pfoil_B | 97.09 +/- 0.00 | 0/1 |
| cn2beam1_A | 55.64 +/- 0.00 | 0/1 |
| cn2beam1_B | 90.91 +/- 0.00 | 0/1 |
| cn2beam5_A | 55.64 +/- 0.00 | 0/1 |
| cn2beam5_B | 92.73 +/- 0.00 | 0/1 |
| pfossil_A | 55.64 +/- 0.00 | 0/1 |
| pfossil_B | 96.00 +/- 0.00 | 0/1 |
| aqr_A | 55.64 +/- 0.00 | 0/1 |
| aqr_B | 96.00 +/- 0.00 | 0/1 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.412 |
| lord | nan |
| pylord | 5.159 |
| pfoil_A | 0.128 |
| pfoil_B | 0.124 |
| cn2beam1_A | 0.049 |
| cn2beam1_B | 0.063 |
| cn2beam5_A | 0.308 |
| cn2beam5_B | 0.181 |
| pfossil_A | 0.054 |
| pfossil_B | 0.074 |
| aqr_A | 0.420 |
| aqr_B | 0.703 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.0 | 2.40 |
| lord | nan | nan |
| pylord | 28.0 | 2.68 |
| pfoil_A | 12.0 | 3.00 |
| pfoil_B | 11.0 | 2.91 |
| cn2beam1_A | 5.0 | 2.60 |
| cn2beam1_B | 7.0 | 2.29 |
| cn2beam5_A | 7.0 | 2.43 |
| cn2beam5_B | 6.0 | 2.50 |
| pfossil_A | 6.0 | 2.17 |
| pfossil_B | 7.0 | 2.29 |
| aqr_A | 2.0 | 7.00 |
| aqr_B | 5.0 | 8.40 |

(7.9s total)

---

## Overall summary

### Accuracy

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 91.95 | nan | 89.66 | 60.92 | 93.10 | 60.92 | 91.95 | 60.92 | 91.95 | 60.92 | 89.66 | 60.92 | 87.36 |
| breast-cancer | 79.31 | nan | 70.69 | 70.69 | 41.38 | 70.69 | 70.69 | 70.69 | 70.69 | 70.69 | 46.55 | 70.69 | 74.14 |
| colic | 86.49 | nan | 81.08 | 62.16 | 68.92 | 62.16 | 74.32 | 62.16 | 74.32 | 62.16 | 81.08 | nan | nan |
| credit-approval | 81.16 | nan | 81.16 | 78.26 | 55.07 | 65.94 | 55.07 | 72.46 | 55.07 | 79.71 | 55.07 | nan | nan |
| credit-g | 72.50 | nan | 73.00 | 60.50 | 70.00 | 72.50 | 70.00 | 71.00 | 70.00 | 69.50 | 70.00 | nan | nan |
| diabetes | 72.08 | nan | 73.38 | 64.94 | 59.09 | 64.94 | 72.73 | 64.94 | 74.03 | 64.94 | 73.38 | 64.94 | 68.83 |
| tic-tac-toe | 98.44 | nan | 95.83 | 96.35 | 65.62 | 98.44 | 65.62 | 98.44 | 65.62 | 82.81 | 65.62 | 94.27 | 65.62 |
| banknote-authentication | 98.18 | nan | 98.55 | 55.64 | 97.09 | 55.64 | 90.91 | 55.64 | 92.73 | 55.64 | 96.00 | 55.64 | 96.00 |

### Fit time (seconds/fold)

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.372 | nan | 3.435 | 1.386 | 0.166 | 0.063 | 0.232 | 0.187 | 0.528 | 0.174 | 0.065 | 2.452 | 2.781 |
| breast-cancer | 0.360 | nan | 2.630 | 0.467 | 0.312 | 0.014 | 0.007 | 0.046 | 0.007 | 0.240 | 0.121 | 7.373 | 3.355 |
| colic | 0.483 | nan | 18.221 | 3.888 | 4.189 | 2.647 | 0.994 | 2.984 | 0.897 | 2.896 | 1.702 | nan | nan |
| credit-approval | 0.473 | nan | 15.175 | 1.743 | 2.819 | 0.293 | 0.681 | 0.954 | 1.457 | 0.335 | 1.027 | nan | nan |
| credit-g | 0.629 | nan | 26.504 | 5.248 | 5.882 | 0.332 | 0.573 | 2.184 | 3.664 | 0.848 | 0.387 | nan | nan |
| diabetes | 0.528 | nan | 10.812 | 1.689 | 1.469 | 0.488 | 0.218 | 1.007 | 0.695 | 0.492 | 0.195 | 3.467 | 23.190 |
| tic-tac-toe | 0.445 | nan | 4.940 | 0.177 | 0.097 | 0.101 | 0.070 | 0.307 | 0.116 | 0.130 | 0.128 | 2.370 | 3.849 |
| banknote-authentication | 0.412 | nan | 5.159 | 0.128 | 0.124 | 0.049 | 0.063 | 0.308 | 0.181 | 0.054 | 0.074 | 0.420 | 0.703 |

### Rule count

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 1.0 | nan | 22.0 | 8.0 | 6.0 | 3.0 | 10.0 | 3.0 | 8.0 | 7.0 | 3.0 | 9.0 | 9.0 |
| breast-cancer | 1.0 | nan | 81.0 | 21.0 | 15.0 | 1.0 | 0.0 | 1.0 | 0.0 | 9.0 | 5.0 | 21.0 | 9.0 |
| colic | 3.0 | nan | 65.0 | 16.0 | 12.0 | 27.0 | 7.0 | 20.0 | 5.0 | 10.0 | 4.0 | nan | nan |
| credit-approval | 2.0 | nan | 111.0 | 25.0 | 19.0 | 7.0 | 12.0 | 8.0 | 11.0 | 3.0 | 13.0 | nan | nan |
| credit-g | 2.0 | nan | 227.0 | 45.0 | 46.0 | 6.0 | 10.0 | 14.0 | 27.0 | 3.0 | 1.0 | nan | nan |
| diabetes | 2.0 | nan | 181.0 | 40.0 | 32.0 | 18.0 | 8.0 | 12.0 | 7.0 | 11.0 | 4.0 | 3.0 | 28.0 |
| tic-tac-toe | 8.0 | nan | 45.0 | 14.0 | 8.0 | 9.0 | 6.0 | 8.0 | 3.0 | 8.0 | 9.0 | 30.0 | 49.0 |
| banknote-authentication | 5.0 | nan | 28.0 | 12.0 | 11.0 | 5.0 | 7.0 | 7.0 | 6.0 | 6.0 | 7.0 | 2.0 | 5.0 |

### Average conditions per rule

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 1.00 | nan | 2.86 | 2.38 | 3.33 | 2.00 | 2.50 | 2.00 | 2.88 | 3.00 | 2.33 | 7.33 | 13.11 |
| breast-cancer | 2.00 | nan | 5.25 | 3.67 | 3.20 | 1.00 | 0.00 | 1.00 | 0.00 | 4.44 | 3.80 | 22.33 | 26.22 |
| colic | 1.33 | nan | 1.95 | 2.81 | 3.08 | 1.22 | 1.71 | 1.55 | 1.80 | 3.60 | 4.75 | nan | nan |
| credit-approval | 1.00 | nan | 3.82 | 3.32 | 4.05 | 1.86 | 2.42 | 3.00 | 2.82 | 4.33 | 3.92 | nan | nan |
| credit-g | 4.00 | nan | 3.70 | 5.20 | 4.50 | 2.33 | 2.30 | 2.57 | 2.74 | 10.67 | 8.00 | nan | nan |
| diabetes | 2.50 | nan | 5.17 | 4.30 | 4.66 | 2.72 | 2.75 | 3.33 | 3.43 | 4.45 | 3.75 | 35.33 | 32.71 |
| tic-tac-toe | 3.38 | nan | 4.04 | 3.64 | 3.25 | 3.00 | 3.00 | 3.00 | 3.00 | 3.50 | 3.67 | 10.13 | 9.49 |
| banknote-authentication | 2.40 | nan | 2.68 | 3.00 | 2.91 | 2.60 | 2.29 | 2.43 | 2.50 | 2.17 | 2.29 | 7.00 | 8.40 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 85.01 | 0.463 | 3.0 | 2.20 | 0 |
| lord | nan | nan | nan | nan | 8 |
| pylord | 82.92 | 10.859 | 95.0 | 3.69 | 0 |
| pfoil_A | 68.68 | 1.841 | 22.6 | 3.54 | 0 |
| pfoil_B | 68.79 | 1.882 | 18.6 | 3.62 | 0 |
| cn2beam1_A | 68.90 | 0.498 | 9.5 | 2.09 | 0 |
| cn2beam1_B | 73.91 | 0.355 | 7.5 | 2.12 | 0 |
| cn2beam5_A | 69.53 | 0.997 | 9.1 | 2.36 | 0 |
| cn2beam5_B | 74.30 | 0.943 | 8.4 | 2.40 | 0 |
| pfossil_A | 68.30 | 0.646 | 7.1 | 4.52 | 0 |
| pfossil_B | 72.17 | 0.462 | 5.8 | 4.06 | 0 |
| aqr_A | 69.29 | 3.216 | 13.0 | 16.43 | 3 |
| aqr_B | 78.39 | 6.776 | 20.0 | 17.99 | 3 |

**Average rank per criterion** (1 = best of 13; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 2.25 | 5.75 | 2.19 | 3.25 |
| lord | 12.62 | 12.62 | 12.62 | 12.62 |
| pylord | 3.19 | 10.88 | 11.12 | 7.88 |
| pfoil_A | 7.75 | 8.00 | 9.12 | 7.25 |
| pfoil_B | 7.38 | 6.50 | 7.69 | 7.88 |
| cn2beam1_A | 6.81 | 2.25 | 5.19 | 2.81 |
| cn2beam1_B | 6.25 | 2.88 | 5.31 | 2.94 |
| cn2beam5_A | 6.88 | 6.25 | 5.56 | 3.56 |
| cn2beam5_B | 5.75 | 5.38 | 4.31 | 4.50 |
| pfossil_A | 7.88 | 4.38 | 4.94 | 7.88 |
| pfossil_B | 6.38 | 3.62 | 4.69 | 7.06 |
| aqr_A | 9.69 | 11.00 | 8.62 | 11.62 |
| aqr_B | 8.19 | 11.50 | 9.62 | 11.75 |
