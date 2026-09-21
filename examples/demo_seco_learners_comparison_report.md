# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets

Generated 2026-09-21 13:55:25. N_FOLDS=5, MAX_INTERVALS=8, FIT_TIMEOUT_SECONDS=60. sonar, ionosphere, kr-vs-kp, mushroom excluded (pass --include-large). `{model}_A`/`{model}_B` treat each dataset's (alphabetically) first/second class as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil`/`aqr` only -- `jrip`/`lord`/`pylord` need no direction). `jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm itself. See this module's own docstring for what each model is.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

*5 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 94.94 +/- 1.72 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 93.33 +/- 2.56 | 0/5 |
| pfoil_A | 61.38 +/- 0.56 | 0/5 |
| pfoil_B | 94.25 +/- 1.26 | 0/5 |
| cn2beam1_A | 61.38 +/- 0.56 | 0/5 |
| cn2beam1_B | 92.87 +/- 1.98 | 0/5 |
| cn2beam5_A | 61.38 +/- 0.56 | 0/5 |
| cn2beam5_B | 93.56 +/- 1.38 | 0/5 |
| pfossil_A | 61.38 +/- 0.56 | 0/5 |
| pfossil_B | 94.48 +/- 2.85 | 0/5 |
| aqr_A | 61.38 +/- 0.56 | 0/5 |
| aqr_B | 92.87 +/- 3.66 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.427 |
| lord | nan |
| pylord | 6.042 |
| pfoil_A | 0.776 |
| pfoil_B | 0.327 |
| cn2beam1_A | 0.176 |
| cn2beam1_B | 0.326 |
| cn2beam5_A | 0.421 |
| cn2beam5_B | 0.886 |
| pfossil_A | 0.293 |
| pfossil_B | 0.136 |
| aqr_A | 6.570 |
| aqr_B | 4.627 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.2 | 1.60 |
| lord | nan | nan |
| pylord | 27.4 | 2.83 |
| pfoil_A | 8.2 | 2.47 |
| pfoil_B | 7.4 | 3.13 |
| cn2beam1_A | 5.2 | 2.04 |
| cn2beam1_B | 8.2 | 2.66 |
| cn2beam5_A | 4.4 | 2.09 |
| cn2beam5_B | 6.8 | 2.89 |
| pfossil_A | 6.0 | 2.76 |
| pfossil_B | 3.8 | 2.25 |
| aqr_A | 12.4 | 10.11 |
| aqr_B | 8.6 | 12.77 |

(106.3s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

*5 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 73.06 +/- 5.36 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 73.09 +/- 1.20 | 0/5 |
| pfoil_A | 70.28 +/- 0.21 | 0/5 |
| pfoil_B | 46.52 +/- 10.33 | 0/5 |
| cn2beam1_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam1_B | 69.93 +/- 0.78 | 0/5 |
| cn2beam5_A | 70.28 +/- 0.21 | 0/5 |
| cn2beam5_B | 70.98 +/- 1.37 | 0/5 |
| pfossil_A | 70.28 +/- 0.21 | 0/5 |
| pfossil_B | 60.54 +/- 12.42 | 0/5 |
| aqr_A | 70.28 +/- 0.21 | 0/5 |
| aqr_B | 66.41 +/- 5.71 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.354 |
| lord | nan |
| pylord | 4.251 |
| pfoil_A | 0.721 |
| pfoil_B | 0.565 |
| cn2beam1_A | 0.032 |
| cn2beam1_B | 0.012 |
| cn2beam5_A | 0.122 |
| cn2beam5_B | 0.058 |
| pfossil_A | 0.310 |
| pfossil_B | 0.273 |
| aqr_A | 12.664 |
| aqr_B | 7.904 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 1.0 | 1.80 |
| lord | nan | nan |
| pylord | 83.4 | 4.51 |
| pfoil_A | 19.0 | 3.77 |
| pfoil_B | 18.0 | 3.05 |
| cn2beam1_A | 1.2 | 1.70 |
| cn2beam1_B | 0.2 | 0.20 |
| cn2beam5_A | 2.0 | 1.83 |
| cn2beam5_B | 1.2 | 0.53 |
| pfossil_A | 7.8 | 4.13 |
| pfossil_B | 5.4 | 4.36 |
| aqr_A | 21.8 | 23.03 |
| aqr_B | 12.8 | 25.83 |

(137.2s total)

---

## colic

n=368, attributes=26, A='1', B='2'

*15 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 88.04 +/- 3.26 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 81.80 +/- 1.30 | 0/5 |
| pfoil_A | 63.04 +/- 0.49 | 0/5 |
| pfoil_B | 74.46 +/- 4.04 | 0/5 |
| cn2beam1_A | 63.04 +/- 0.49 | 0/5 |
| cn2beam1_B | 80.99 +/- 4.41 | 0/5 |
| cn2beam5_A | 63.04 +/- 0.49 | 0/5 |
| cn2beam5_B | 81.81 +/- 5.28 | 0/5 |
| pfossil_A | 63.04 +/- 0.49 | 0/5 |
| pfossil_B | 86.15 +/- 3.20 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.534 |
| lord | nan |
| pylord | 29.014 |
| pfoil_A | 5.457 |
| pfoil_B | 6.522 |
| cn2beam1_A | 3.120 |
| cn2beam1_B | 1.366 |
| cn2beam5_A | 3.699 |
| cn2beam5_B | 1.537 |
| pfossil_A | 3.931 |
| pfossil_B | 2.344 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.0 | 1.40 |
| lord | nan | nan |
| pylord | 67.4 | 1.96 |
| pfoil_A | 13.6 | 3.04 |
| pfoil_B | 14.8 | 2.58 |
| cn2beam1_A | 19.0 | 1.37 |
| cn2beam1_B | 6.4 | 1.64 |
| cn2beam5_A | 15.2 | 1.60 |
| cn2beam5_B | 5.2 | 1.73 |
| pfossil_A | 10.0 | 3.24 |
| pfossil_B | 4.2 | 3.89 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(894.6s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

*15 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 85.22 +/- 2.58 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 84.06 +/- 2.00 | 0/5 |
| pfoil_A | 76.38 +/- 4.75 | 0/5 |
| pfoil_B | 55.51 +/- 0.35 | 0/5 |
| cn2beam1_A | 74.06 +/- 6.27 | 0/5 |
| cn2beam1_B | 55.51 +/- 0.35 | 0/5 |
| cn2beam5_A | 75.36 +/- 4.27 | 0/5 |
| cn2beam5_B | 55.51 +/- 0.35 | 0/5 |
| pfossil_A | 84.20 +/- 3.32 | 0/5 |
| pfossil_B | 55.51 +/- 0.35 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.522 |
| lord | nan |
| pylord | 20.124 |
| pfoil_A | 2.567 |
| pfoil_B | 3.670 |
| cn2beam1_A | 0.505 |
| cn2beam1_B | 0.847 |
| cn2beam5_A | 1.144 |
| cn2beam5_B | 2.199 |
| pfossil_A | 0.390 |
| pfossil_B | 0.766 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 3.0 | 1.67 |
| lord | nan | nan |
| pylord | 114.0 | 3.89 |
| pfoil_A | 26.2 | 3.40 |
| pfoil_B | 22.4 | 4.23 |
| cn2beam1_A | 8.4 | 1.96 |
| cn2beam1_B | 11.2 | 2.89 |
| cn2beam5_A | 6.8 | 2.47 |
| cn2beam5_B | 10.6 | 3.05 |
| pfossil_A | 3.0 | 3.23 |
| pfossil_B | 7.4 | 4.54 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(768.5s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*15 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 71.60 +/- 1.07 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 73.70 +/- 2.93 | 0/5 |
| pfoil_A | 58.40 +/- 4.53 | 0/5 |
| pfoil_B | 70.00 +/- 0.00 | 0/5 |
| cn2beam1_A | 71.90 +/- 0.58 | 0/5 |
| cn2beam1_B | 70.00 +/- 0.00 | 0/5 |
| cn2beam5_A | 71.10 +/- 1.07 | 0/5 |
| cn2beam5_B | 70.00 +/- 0.00 | 0/5 |
| pfossil_A | 71.10 +/- 1.59 | 0/5 |
| pfossil_B | 70.00 +/- 0.00 | 0/5 |
| aqr_A | n/a | 5/5 |
| aqr_B | n/a | 5/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.656 |
| lord | nan |
| pylord | 32.150 |
| pfoil_A | 5.266 |
| pfoil_B | 6.936 |
| cn2beam1_A | 0.364 |
| cn2beam1_B | 0.708 |
| cn2beam5_A | 1.592 |
| cn2beam5_B | 3.559 |
| pfossil_A | 0.662 |
| pfossil_B | 0.560 |
| aqr_A | nan |
| aqr_B | nan |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.2 | 3.40 |
| lord | nan | nan |
| pylord | 226.2 | 3.83 |
| pfoil_A | 42.0 | 4.87 |
| pfoil_B | 46.6 | 4.55 |
| cn2beam1_A | 5.8 | 2.29 |
| cn2beam1_B | 9.0 | 2.64 |
| cn2beam5_A | 8.6 | 2.75 |
| cn2beam5_B | 19.6 | 2.79 |
| pfossil_A | 1.8 | 11.13 |
| pfossil_B | 1.6 | 7.40 |
| aqr_A | nan | nan |
| aqr_B | nan | nan |

(868.0s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

*6 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 75.00 +/- 3.53 | 0/5 |
| lord | n/a | 5/5 |
| pylord | 75.01 +/- 2.59 | 0/5 |
| pfoil_A | 65.10 +/- 0.21 | 0/5 |
| pfoil_B | 63.55 +/- 2.95 | 0/5 |
| cn2beam1_A | 65.10 +/- 0.21 | 0/5 |
| cn2beam1_B | 72.01 +/- 2.06 | 0/5 |
| cn2beam5_A | 65.10 +/- 0.21 | 0/5 |
| cn2beam5_B | 71.87 +/- 1.31 | 0/5 |
| pfossil_A | 65.10 +/- 0.21 | 0/5 |
| pfossil_B | 70.96 +/- 3.38 | 0/5 |
| aqr_A | 65.15 +/- 0.21 | 1/5 |
| aqr_B | 69.01 +/- 1.82 | 0/5 |

**Fit time (seconds/fold)**

| model | time |
|---|---|
| jrip | 0.482 |
| lord | nan |
| pylord | 16.989 |
| pfoil_A | 2.508 |
| pfoil_B | 2.412 |
| cn2beam1_A | 0.368 |
| cn2beam1_B | 0.300 |
| cn2beam5_A | 1.383 |
| cn2beam5_B | 0.919 |
| pfossil_A | 0.295 |
| pfossil_B | 0.439 |
| aqr_A | 31.576 |
| aqr_B | 27.965 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 2.0 | 2.80 |
| lord | nan | nan |
| pylord | 190.0 | 5.33 |
| pfoil_A | 38.8 | 4.45 |
| pfoil_B | 33.6 | 4.62 |
| cn2beam1_A | 9.4 | 2.64 |
| cn2beam1_B | 8.0 | 2.66 |
| cn2beam5_A | 10.4 | 2.96 |
| cn2beam5_B | 8.0 | 3.20 |
| pfossil_A | 4.4 | 4.06 |
| pfossil_B | 6.6 | 4.38 |
| aqr_A | 24.0 | 35.37 |
| aqr_B | 21.8 | 33.71 |

(458.8s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

*5 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.33 +/- 1.01 | 0/5 |
| lord | n/a | 5/5 |
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
| jrip | 0.496 |
| lord | nan |
| pylord | 5.977 |
| pfoil_A | 0.214 |
| pfoil_B | 0.155 |
| cn2beam1_A | 0.123 |
| cn2beam1_B | 0.104 |
| cn2beam5_A | 0.400 |
| cn2beam5_B | 0.257 |
| pfossil_A | 0.147 |
| pfossil_B | 0.137 |
| aqr_A | 3.906 |
| aqr_B | 4.533 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 8.0 | 3.08 |
| lord | nan | nan |
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

(83.1s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

*5 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| model | accuracy | failed folds |
|---|---|---|
| jrip | 98.61 +/- 0.54 | 0/5 |
| lord | n/a | 5/5 |
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
| jrip | 0.496 |
| lord | nan |
| pylord | 5.793 |
| pfoil_A | 0.178 |
| pfoil_B | 0.162 |
| cn2beam1_A | 0.079 |
| cn2beam1_B | 0.073 |
| cn2beam5_A | 0.350 |
| cn2beam5_B | 0.280 |
| pfossil_A | 0.069 |
| pfossil_B | 0.096 |
| aqr_A | 0.884 |
| aqr_B | 1.236 |

**Rule complexity**

| model | n_rules | avg_conditions |
|---|---|---|
| jrip | 5.2 | 2.30 |
| lord | nan | nan |
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

(49.8s total)

---

## Overall summary

### Accuracy

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 94.94 | nan | 93.33 | 61.38 | 94.25 | 61.38 | 92.87 | 61.38 | 93.56 | 61.38 | 94.48 | 61.38 | 92.87 |
| breast-cancer | 73.06 | nan | 73.09 | 70.28 | 46.52 | 70.28 | 69.93 | 70.28 | 70.98 | 70.28 | 60.54 | 70.28 | 66.41 |
| colic | 88.04 | nan | 81.80 | 63.04 | 74.46 | 63.04 | 80.99 | 63.04 | 81.81 | 63.04 | 86.15 | nan | nan |
| credit-approval | 85.22 | nan | 84.06 | 76.38 | 55.51 | 74.06 | 55.51 | 75.36 | 55.51 | 84.20 | 55.51 | nan | nan |
| credit-g | 71.60 | nan | 73.70 | 58.40 | 70.00 | 71.90 | 70.00 | 71.10 | 70.00 | 71.10 | 70.00 | nan | nan |
| diabetes | 75.00 | nan | 75.01 | 65.10 | 63.55 | 65.10 | 72.01 | 65.10 | 71.87 | 65.10 | 70.96 | 65.15 | 69.01 |
| tic-tac-toe | 98.33 | nan | 95.51 | 93.62 | 65.34 | 98.33 | 65.34 | 98.33 | 65.34 | 80.58 | 65.34 | 91.55 | 65.34 |
| banknote-authentication | 98.61 | nan | 99.13 | 55.54 | 98.03 | 55.54 | 90.45 | 55.54 | 94.90 | 55.54 | 96.87 | 55.54 | 98.32 |

### Fit time (seconds/fold)

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.427 | nan | 6.042 | 0.776 | 0.327 | 0.176 | 0.326 | 0.421 | 0.886 | 0.293 | 0.136 | 6.570 | 4.627 |
| breast-cancer | 0.354 | nan | 4.251 | 0.721 | 0.565 | 0.032 | 0.012 | 0.122 | 0.058 | 0.310 | 0.273 | 12.664 | 7.904 |
| colic | 0.534 | nan | 29.014 | 5.457 | 6.522 | 3.120 | 1.366 | 3.699 | 1.537 | 3.931 | 2.344 | nan | nan |
| credit-approval | 0.522 | nan | 20.124 | 2.567 | 3.670 | 0.505 | 0.847 | 1.144 | 2.199 | 0.390 | 0.766 | nan | nan |
| credit-g | 0.656 | nan | 32.150 | 5.266 | 6.936 | 0.364 | 0.708 | 1.592 | 3.559 | 0.662 | 0.560 | nan | nan |
| diabetes | 0.482 | nan | 16.989 | 2.508 | 2.412 | 0.368 | 0.300 | 1.383 | 0.919 | 0.295 | 0.439 | 31.576 | 27.965 |
| tic-tac-toe | 0.496 | nan | 5.977 | 0.214 | 0.155 | 0.123 | 0.104 | 0.400 | 0.257 | 0.147 | 0.137 | 3.906 | 4.533 |
| banknote-authentication | 0.496 | nan | 5.793 | 0.178 | 0.162 | 0.079 | 0.073 | 0.350 | 0.280 | 0.069 | 0.096 | 0.884 | 1.236 |

### Rule count

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 2.2 | nan | 27.4 | 8.2 | 7.4 | 5.2 | 8.2 | 4.4 | 6.8 | 6.0 | 3.8 | 12.4 | 8.6 |
| breast-cancer | 1.0 | nan | 83.4 | 19.0 | 18.0 | 1.2 | 0.2 | 2.0 | 1.2 | 7.8 | 5.4 | 21.8 | 12.8 |
| colic | 3.0 | nan | 67.4 | 13.6 | 14.8 | 19.0 | 6.4 | 15.2 | 5.2 | 10.0 | 4.2 | nan | nan |
| credit-approval | 3.0 | nan | 114.0 | 26.2 | 22.4 | 8.4 | 11.2 | 6.8 | 10.6 | 3.0 | 7.4 | nan | nan |
| credit-g | 2.2 | nan | 226.2 | 42.0 | 46.6 | 5.8 | 9.0 | 8.6 | 19.6 | 1.8 | 1.6 | nan | nan |
| diabetes | 2.0 | nan | 190.0 | 38.8 | 33.6 | 9.4 | 8.0 | 10.4 | 8.0 | 4.4 | 6.6 | 24.0 | 21.8 |
| tic-tac-toe | 8.0 | nan | 57.4 | 13.0 | 9.2 | 8.4 | 5.8 | 8.0 | 5.0 | 7.6 | 7.2 | 36.6 | 47.2 |
| banknote-authentication | 5.2 | nan | 27.0 | 13.6 | 11.6 | 6.4 | 6.8 | 6.6 | 6.8 | 6.2 | 6.8 | 5.2 | 7.6 |

### Average conditions per rule

| dataset | jrip | lord | pylord | pfoil_A | pfoil_B | cn2beam1_A | cn2beam1_B | cn2beam5_A | cn2beam5_B | pfossil_A | pfossil_B | aqr_A | aqr_B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 1.60 | nan | 2.83 | 2.47 | 3.13 | 2.04 | 2.66 | 2.09 | 2.89 | 2.76 | 2.25 | 10.11 | 12.77 |
| breast-cancer | 1.80 | nan | 4.51 | 3.77 | 3.05 | 1.70 | 0.20 | 1.83 | 0.53 | 4.13 | 4.36 | 23.03 | 25.83 |
| colic | 1.40 | nan | 1.96 | 3.04 | 2.58 | 1.37 | 1.64 | 1.60 | 1.73 | 3.24 | 3.89 | nan | nan |
| credit-approval | 1.67 | nan | 3.89 | 3.40 | 4.23 | 1.96 | 2.89 | 2.47 | 3.05 | 3.23 | 4.54 | nan | nan |
| credit-g | 3.40 | nan | 3.83 | 4.87 | 4.55 | 2.29 | 2.64 | 2.75 | 2.79 | 11.13 | 7.40 | nan | nan |
| diabetes | 2.80 | nan | 5.33 | 4.45 | 4.62 | 2.64 | 2.66 | 2.96 | 3.20 | 4.06 | 4.38 | 35.37 | 33.71 |
| tic-tac-toe | 3.08 | nan | 4.19 | 3.64 | 3.37 | 3.00 | 3.00 | 3.00 | 3.03 | 3.74 | 3.47 | 10.39 | 9.52 |
| banknote-authentication | 2.30 | nan | 2.79 | 2.94 | 3.02 | 2.32 | 2.18 | 2.49 | 2.37 | 2.19 | 2.44 | 9.24 | 10.41 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| jrip | 85.60 | 0.496 | 3.3 | 2.26 | 0 |
| lord | nan | nan | nan | nan | 8 |
| pylord | 84.45 | 15.043 | 99.1 | 3.67 | 0 |
| pfoil_A | 67.97 | 2.211 | 21.8 | 3.57 | 0 |
| pfoil_B | 70.96 | 2.594 | 20.4 | 3.57 | 0 |
| cn2beam1_A | 69.95 | 0.596 | 8.0 | 2.16 | 0 |
| cn2beam1_B | 74.64 | 0.467 | 7.0 | 2.23 | 0 |
| cn2beam5_A | 70.02 | 1.139 | 7.8 | 2.40 | 0 |
| cn2beam5_B | 75.50 | 1.212 | 7.9 | 2.45 | 0 |
| pfossil_A | 68.90 | 0.762 | 5.8 | 4.31 | 0 |
| pfossil_B | 74.98 | 0.594 | 5.4 | 4.09 | 0 |
| aqr_A | 68.78 | 11.120 | 20.0 | 17.63 | 3 |
| aqr_B | 78.39 | 9.253 | 19.6 | 18.45 | 3 |

**Average rank per criterion** (1 = best of 13; failed entries tie for last)

| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| jrip | 1.75 | 5.50 | 2.06 | 3.00 |
| lord | 12.62 | 12.62 | 12.62 | 12.62 |
| pylord | 2.50 | 10.62 | 11.25 | 8.25 |
| pfoil_A | 7.88 | 7.75 | 9.06 | 7.50 |
| pfoil_B | 7.88 | 7.25 | 8.50 | 8.00 |
| cn2beam1_A | 6.75 | 2.50 | 5.31 | 2.00 |
| cn2beam1_B | 7.06 | 2.75 | 5.00 | 2.75 |
| cn2beam5_A | 6.94 | 6.38 | 5.19 | 3.75 |
| cn2beam5_B | 5.75 | 6.12 | 4.75 | 4.88 |
| pfossil_A | 7.19 | 3.38 | 3.69 | 7.12 |
| pfossil_B | 6.38 | 3.38 | 3.50 | 7.75 |
| aqr_A | 9.38 | 11.50 | 9.81 | 11.62 |
| aqr_B | 8.94 | 11.25 | 10.25 | 11.75 |
