# Binarized vs. Original data preparation -- comparison across binary datasets

Generated 2026-09-01 16:19:22. N_FOLDS=5, MAX_INTERVALS=8 ('Binarized' mode only), MAX_DEPTH=4, RIPPER_K=2, N_ESTIMATORS=10 (forest), FIT_TIMEOUT_SECONDS=60. `ripper_A`/`irep_A` treat each dataset's (alphabetically) first class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit the same already-Boolean matrix under both workflows as two independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` fit-time includes JVM subprocess startup overhead, not just the algorithm itself -- see the module docstring.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 95.17 +/- 0.86 | 0/5 |
| Binarized | forest | 96.32 +/- 1.84 | 0/5 |
| Binarized | ripper_A | 94.02 +/- 1.84 | 0/5 |
| Binarized | ripper_B | 96.09 +/- 1.17 | 0/5 |
| Binarized | irep_A | 95.17 +/- 1.34 | 0/5 |
| Binarized | irep_B | 94.48 +/- 2.34 | 0/5 |
| Binarized | brl | 94.02 +/- 1.69 | 0/5 |
| Binarized | brs | 89.20 +/- 5.32 | 0/5 |
| Binarized | jrip | 94.94 +/- 1.72 | 0/5 |
| Binarized | part | 95.40 +/- 1.26 | 0/5 |
| Binarized | j48 | 96.09 +/- 1.17 | 0/5 |
| Original | tree | 94.94 +/- 1.17 | 0/5 |
| Original | forest | 94.71 +/- 2.13 | 0/5 |
| Original | ripper_A | 95.17 +/- 1.13 | 0/5 |
| Original | ripper_B | 95.17 +/- 1.52 | 0/5 |
| Original | irep_A | 93.79 +/- 1.87 | 0/5 |
| Original | irep_B | 94.94 +/- 2.00 | 0/5 |
| Original | brl | 94.02 +/- 1.69 | 0/5 |
| Original | brs | 89.20 +/- 5.32 | 0/5 |
| Original | jrip | 95.63 +/- 2.34 | 0/5 |
| Original | part | 95.63 +/- 1.34 | 0/5 |
| Original | j48 | 96.32 +/- 1.34 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.297 |
| Binarized | forest | 0.023 |
| Binarized | ripper_A | 0.114 |
| Binarized | ripper_B | 0.099 |
| Binarized | irep_A | 0.072 |
| Binarized | irep_B | 0.068 |
| Binarized | brl | 0.640 |
| Binarized | brs | 7.796 |
| Binarized | jrip | 0.290 |
| Binarized | part | 0.252 |
| Binarized | j48 | 0.249 |
| Original | tree | 0.003 |
| Original | forest | 0.014 |
| Original | ripper_A | 0.058 |
| Original | ripper_B | 0.053 |
| Original | irep_A | 0.034 |
| Original | irep_B | 0.033 |
| Original | brl | 0.530 |
| Original | brs | 7.697 |
| Original | jrip | 0.205 |
| Original | part | 0.205 |
| Original | j48 | 0.207 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.4 | 3.80 |
| Binarized | forest | 111.8 | 3.67 |
| Binarized | ripper_A | 2.0 | 1.53 |
| Binarized | ripper_B | 3.0 | 2.25 |
| Binarized | irep_A | 2.2 | 1.73 |
| Binarized | irep_B | 1.2 | 1.10 |
| Binarized | brl | 3.8 | 1.48 |
| Binarized | brs | 7.4 | 2.97 |
| Binarized | jrip | 2.2 | 1.60 |
| Binarized | part | 5.2 | 2.23 |
| Binarized | j48 | 6.8 | 3.32 |
| Original | tree | 13.4 | 3.80 |
| Original | forest | 121.0 | 3.73 |
| Original | ripper_A | 3.4 | 1.88 |
| Original | ripper_B | 3.8 | 1.95 |
| Original | irep_A | 1.8 | 1.17 |
| Original | irep_B | 1.4 | 1.10 |
| Original | brl | 3.8 | 1.48 |
| Original | brs | 7.4 | 2.97 |
| Original | jrip | 1.2 | 1.20 |
| Original | part | 5.2 | 1.42 |
| Original | j48 | 9.0 | 2.61 |

(95.8s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 72.01 +/- 2.36 | 0/5 |
| Binarized | forest | 73.43 +/- 2.76 | 0/5 |
| Binarized | ripper_A | 61.63 +/- 13.98 | 0/5 |
| Binarized | ripper_B | 75.52 +/- 2.49 | 0/5 |
| Binarized | irep_A | 74.81 +/- 2.51 | 0/5 |
| Binarized | irep_B | 72.02 +/- 4.48 | 0/5 |
| Binarized | brl | 73.09 +/- 2.26 | 0/5 |
| Binarized | brs | 71.34 +/- 3.10 | 0/5 |
| Binarized | jrip | 73.06 +/- 5.36 | 0/5 |
| Binarized | part | 67.14 +/- 4.10 | 0/5 |
| Binarized | j48 | 69.93 +/- 2.76 | 0/5 |
| Original | tree | 72.72 +/- 1.92 | 0/5 |
| Original | forest | 75.52 +/- 2.00 | 0/5 |
| Original | ripper_A | 66.79 +/- 2.79 | 0/5 |
| Original | ripper_B | 73.07 +/- 2.21 | 0/5 |
| Original | irep_A | 72.72 +/- 5.09 | 0/5 |
| Original | irep_B | 69.91 +/- 8.46 | 0/5 |
| Original | brl | 73.09 +/- 2.26 | 0/5 |
| Original | brs | 71.34 +/- 3.10 | 0/5 |
| Original | jrip | 72.02 +/- 3.56 | 0/5 |
| Original | part | 69.96 +/- 5.62 | 0/5 |
| Original | j48 | 73.42 +/- 1.45 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.004 |
| Binarized | forest | 0.018 |
| Binarized | ripper_A | 0.100 |
| Binarized | ripper_B | 0.119 |
| Binarized | irep_A | 0.059 |
| Binarized | irep_B | 0.059 |
| Binarized | brl | 0.221 |
| Binarized | brs | 7.330 |
| Binarized | jrip | 0.268 |
| Binarized | part | 0.285 |
| Binarized | j48 | 0.250 |
| Original | tree | 0.003 |
| Original | forest | 0.027 |
| Original | ripper_A | 0.048 |
| Original | ripper_B | 0.077 |
| Original | irep_A | 0.025 |
| Original | irep_B | 0.024 |
| Original | brl | 0.220 |
| Original | brs | 7.369 |
| Original | jrip | 0.198 |
| Original | part | 0.200 |
| Original | j48 | 0.197 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.4 | 3.75 |
| Binarized | forest | 126.0 | 3.78 |
| Binarized | ripper_A | 1.2 | 1.50 |
| Binarized | ripper_B | 1.4 | 2.70 |
| Binarized | irep_A | 2.2 | 1.27 |
| Binarized | irep_B | 1.4 | 2.40 |
| Binarized | brl | 3.2 | 1.42 |
| Binarized | brs | 7.6 | 2.95 |
| Binarized | jrip | 1.0 | 1.80 |
| Binarized | part | 22.0 | 4.54 |
| Binarized | j48 | 15.0 | 5.37 |
| Original | tree | 12.4 | 3.75 |
| Original | forest | 124.8 | 3.78 |
| Original | ripper_A | 3.0 | 1.55 |
| Original | ripper_B | 1.6 | 2.70 |
| Original | irep_A | 2.4 | 1.13 |
| Original | irep_B | 1.2 | 1.90 |
| Original | brl | 3.2 | 1.42 |
| Original | brs | 7.6 | 2.95 |
| Original | jrip | 2.8 | 2.22 |
| Original | part | 15.4 | 1.92 |
| Original | j48 | 8.4 | 1.77 |

(86.4s total)

---

## colic

n=368, attributes=26, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 85.06 +/- 2.66 | 0/5 |
| Binarized | forest | 86.42 +/- 2.82 | 0/5 |
| Binarized | ripper_A | 85.60 +/- 3.04 | 0/5 |
| Binarized | ripper_B | 85.07 +/- 2.79 | 0/5 |
| Binarized | irep_A | 86.68 +/- 1.03 | 0/5 |
| Binarized | irep_B | 82.63 +/- 3.68 | 0/5 |
| Binarized | brl | 84.78 +/- 1.00 | 0/5 |
| Binarized | brs | 79.89 +/- 4.57 | 0/5 |
| Binarized | jrip | 88.04 +/- 3.26 | 0/5 |
| Binarized | part | 80.70 +/- 4.83 | 0/5 |
| Binarized | j48 | 85.61 +/- 2.63 | 0/5 |
| Original | tree | 84.25 +/- 3.11 | 0/5 |
| Original | forest | 83.71 +/- 3.77 | 0/5 |
| Original | ripper_A | 81.27 +/- 3.39 | 0/5 |
| Original | ripper_B | 86.96 +/- 2.91 | 0/5 |
| Original | irep_A | 82.35 +/- 3.12 | 0/5 |
| Original | irep_B | 86.42 +/- 2.40 | 0/5 |
| Original | brl | 84.78 +/- 1.00 | 0/5 |
| Original | brs | 79.89 +/- 4.57 | 0/5 |
| Original | jrip | 87.51 +/- 3.34 | 0/5 |
| Original | part | 85.88 +/- 2.47 | 0/5 |
| Original | j48 | 82.63 +/- 5.17 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.007 |
| Binarized | forest | 0.020 |
| Binarized | ripper_A | 0.401 |
| Binarized | ripper_B | 0.469 |
| Binarized | irep_A | 0.300 |
| Binarized | irep_B | 0.314 |
| Binarized | brl | 3.917 |
| Binarized | brs | 37.010 |
| Binarized | jrip | 0.397 |
| Binarized | part | 0.361 |
| Binarized | j48 | 0.312 |
| Original | tree | 0.003 |
| Original | forest | 0.014 |
| Original | ripper_A | 0.204 |
| Original | ripper_B | 0.221 |
| Original | irep_A | 0.146 |
| Original | irep_B | 0.146 |
| Original | brl | 3.771 |
| Original | brs | 37.140 |
| Original | jrip | 0.220 |
| Original | part | 0.224 |
| Original | j48 | 0.211 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 11.4 | 3.68 |
| Binarized | forest | 121.2 | 3.74 |
| Binarized | ripper_A | 4.4 | 2.15 |
| Binarized | ripper_B | 5.2 | 1.90 |
| Binarized | irep_A | 2.0 | 1.20 |
| Binarized | irep_B | 2.4 | 1.70 |
| Binarized | brl | 4.8 | 1.46 |
| Binarized | brs | 7.8 | 2.79 |
| Binarized | jrip | 3.0 | 1.40 |
| Binarized | part | 13.6 | 4.53 |
| Binarized | j48 | 15.6 | 5.97 |
| Original | tree | 11.6 | 3.72 |
| Original | forest | 118.6 | 3.73 |
| Original | ripper_A | 6.0 | 1.61 |
| Original | ripper_B | 6.2 | 2.13 |
| Original | irep_A | 2.2 | 1.33 |
| Original | irep_B | 1.4 | 1.70 |
| Original | brl | 4.8 | 1.46 |
| Original | brs | 7.8 | 2.79 |
| Original | jrip | 3.4 | 1.83 |
| Original | part | 12.8 | 1.81 |
| Original | j48 | 8.6 | 2.17 |

(432.1s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 85.07 +/- 3.39 | 0/5 |
| Binarized | forest | 85.22 +/- 2.54 | 0/5 |
| Binarized | ripper_A | 85.36 +/- 2.69 | 0/5 |
| Binarized | ripper_B | 85.80 +/- 2.73 | 0/5 |
| Binarized | irep_A | 85.22 +/- 2.88 | 0/5 |
| Binarized | irep_B | 84.49 +/- 2.36 | 0/5 |
| Binarized | brl | 85.80 +/- 3.39 | 0/5 |
| Binarized | brs | 65.65 +/- 5.18 | 0/5 |
| Binarized | jrip | 85.22 +/- 2.58 | 0/5 |
| Binarized | part | 83.91 +/- 3.29 | 0/5 |
| Binarized | j48 | 85.22 +/- 3.32 | 0/5 |
| Original | tree | 84.78 +/- 3.89 | 0/5 |
| Original | forest | 85.65 +/- 3.25 | 0/5 |
| Original | ripper_A | 83.62 +/- 3.57 | 0/5 |
| Original | ripper_B | 83.04 +/- 3.63 | 0/5 |
| Original | irep_A | 85.51 +/- 3.30 | 0/5 |
| Original | irep_B | 85.94 +/- 3.68 | 0/5 |
| Original | brl | 85.80 +/- 3.39 | 0/5 |
| Original | brs | 65.65 +/- 5.18 | 0/5 |
| Original | jrip | 85.94 +/- 2.58 | 0/5 |
| Original | part | 84.35 +/- 3.51 | 0/5 |
| Original | j48 | 85.94 +/- 4.09 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.034 |
| Binarized | ripper_A | 0.321 |
| Binarized | ripper_B | 0.304 |
| Binarized | irep_A | 0.145 |
| Binarized | irep_B | 0.152 |
| Binarized | brl | 1.574 |
| Binarized | brs | 11.043 |
| Binarized | jrip | 0.420 |
| Binarized | part | 0.377 |
| Binarized | j48 | 0.320 |
| Original | tree | 0.004 |
| Original | forest | 0.016 |
| Original | ripper_A | 0.218 |
| Original | ripper_B | 0.231 |
| Original | irep_A | 0.077 |
| Original | irep_B | 0.081 |
| Original | brl | 1.561 |
| Original | brs | 12.178 |
| Original | jrip | 0.259 |
| Original | part | 0.264 |
| Original | j48 | 0.241 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.8 | 3.85 |
| Binarized | forest | 142.4 | 3.89 |
| Binarized | ripper_A | 5.0 | 2.78 |
| Binarized | ripper_B | 3.8 | 2.40 |
| Binarized | irep_A | 1.2 | 1.10 |
| Binarized | irep_B | 2.2 | 1.97 |
| Binarized | brl | 4.4 | 1.66 |
| Binarized | brs | 8.8 | 2.91 |
| Binarized | jrip | 3.0 | 1.67 |
| Binarized | part | 26.8 | 3.49 |
| Binarized | j48 | 17.0 | 6.31 |
| Original | tree | 14.2 | 3.87 |
| Original | forest | 134.0 | 3.84 |
| Original | ripper_A | 7.6 | 2.86 |
| Original | ripper_B | 8.8 | 3.22 |
| Original | irep_A | 1.2 | 1.20 |
| Original | irep_B | 2.0 | 1.63 |
| Original | brl | 4.4 | 1.66 |
| Original | brs | 8.8 | 2.91 |
| Original | jrip | 3.8 | 2.18 |
| Original | part | 29.6 | 2.37 |
| Original | j48 | 19.4 | 4.02 |

(151.6s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*5/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.70 +/- 0.98 | 0/5 |
| Binarized | forest | 73.50 +/- 0.89 | 0/5 |
| Binarized | ripper_A | 70.80 +/- 3.50 | 0/5 |
| Binarized | ripper_B | 58.50 +/- 6.73 | 0/5 |
| Binarized | irep_A | 71.50 +/- 0.89 | 0/5 |
| Binarized | irep_B | 70.40 +/- 2.96 | 0/5 |
| Binarized | brl | 69.80 +/- 2.23 | 0/5 |
| Binarized | brs | 45.50 +/- 4.06 | 0/5 |
| Binarized | jrip | 71.60 +/- 1.07 | 0/5 |
| Binarized | part | 72.30 +/- 3.23 | 0/5 |
| Binarized | j48 | 71.90 +/- 1.24 | 0/5 |
| Original | tree | 70.60 +/- 2.60 | 0/5 |
| Original | forest | 70.50 +/- 1.26 | 0/5 |
| Original | ripper_A | 70.20 +/- 1.03 | 0/5 |
| Original | ripper_B | 58.60 +/- 5.07 | 0/5 |
| Original | irep_A | 70.40 +/- 1.24 | 0/5 |
| Original | irep_B | 71.20 +/- 2.46 | 0/5 |
| Original | brl | 69.80 +/- 2.23 | 0/5 |
| Original | brs | 45.50 +/- 4.06 | 0/5 |
| Original | jrip | 72.40 +/- 2.40 | 0/5 |
| Original | part | 70.00 +/- 3.36 | 0/5 |
| Original | j48 | 70.80 +/- 1.91 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.007 |
| Binarized | forest | 0.024 |
| Binarized | ripper_A | 0.479 |
| Binarized | ripper_B | 0.398 |
| Binarized | irep_A | 0.183 |
| Binarized | irep_B | 0.174 |
| Binarized | brl | 2.694 |
| Binarized | brs | 11.362 |
| Binarized | jrip | 0.483 |
| Binarized | part | 0.515 |
| Binarized | j48 | 0.396 |
| Original | tree | 0.004 |
| Original | forest | 0.016 |
| Original | ripper_A | 0.257 |
| Original | ripper_B | 0.227 |
| Original | irep_A | 0.082 |
| Original | irep_B | 0.089 |
| Original | brl | 2.626 |
| Original | brs | 11.292 |
| Original | jrip | 0.273 |
| Original | part | 0.281 |
| Original | j48 | 0.243 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.0 | 3.93 |
| Binarized | forest | 146.0 | 3.91 |
| Binarized | ripper_A | 3.2 | 4.83 |
| Binarized | ripper_B | 3.8 | 2.95 |
| Binarized | irep_A | 3.0 | 3.40 |
| Binarized | irep_B | 4.2 | 1.66 |
| Binarized | brl | 4.8 | 1.67 |
| Binarized | brs | 8.6 | 2.86 |
| Binarized | jrip | 2.2 | 3.40 |
| Binarized | part | 56.8 | 4.09 |
| Binarized | j48 | 78.8 | 11.42 |
| Original | tree | 15.2 | 3.95 |
| Original | forest | 141.8 | 3.89 |
| Original | ripper_A | 5.4 | 3.64 |
| Original | ripper_B | 5.2 | 2.68 |
| Original | irep_A | 1.8 | 2.97 |
| Original | irep_B | 5.2 | 1.33 |
| Original | brl | 4.8 | 1.67 |
| Original | brs | 8.6 | 2.86 |
| Original | jrip | 3.4 | 2.69 |
| Original | part | 60.4 | 3.02 |
| Original | j48 | 77.8 | 5.57 |

(163.9s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 73.83 +/- 2.78 | 0/5 |
| Binarized | forest | 75.65 +/- 1.14 | 0/5 |
| Binarized | ripper_A | 65.37 +/- 2.70 | 0/5 |
| Binarized | ripper_B | 74.35 +/- 2.04 | 0/5 |
| Binarized | irep_A | 75.13 +/- 1.84 | 0/5 |
| Binarized | irep_B | 75.27 +/- 1.99 | 0/5 |
| Binarized | brl | 72.01 +/- 2.68 | 0/5 |
| Binarized | brs | 69.01 +/- 1.62 | 0/5 |
| Binarized | jrip | 75.00 +/- 3.53 | 0/5 |
| Binarized | part | 71.88 +/- 2.23 | 0/5 |
| Binarized | j48 | 72.78 +/- 2.34 | 0/5 |
| Original | tree | 73.45 +/- 3.14 | 0/5 |
| Original | forest | 74.88 +/- 3.45 | 0/5 |
| Original | ripper_A | 63.40 +/- 4.26 | 0/5 |
| Original | ripper_B | 72.79 +/- 3.11 | 0/5 |
| Original | irep_A | 65.22 +/- 3.84 | 0/5 |
| Original | irep_B | 74.09 +/- 2.35 | 0/5 |
| Original | brl | 72.01 +/- 2.68 | 0/5 |
| Original | brs | 69.01 +/- 1.62 | 0/5 |
| Original | jrip | 76.05 +/- 3.07 | 0/5 |
| Original | part | 75.00 +/- 2.66 | 0/5 |
| Original | j48 | 74.08 +/- 2.42 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.005 |
| Binarized | forest | 0.023 |
| Binarized | ripper_A | 0.285 |
| Binarized | ripper_B | 0.272 |
| Binarized | irep_A | 0.098 |
| Binarized | irep_B | 0.106 |
| Binarized | brl | 1.054 |
| Binarized | brs | 8.796 |
| Binarized | jrip | 0.431 |
| Binarized | part | 0.398 |
| Binarized | j48 | 0.326 |
| Original | tree | 0.004 |
| Original | forest | 0.016 |
| Original | ripper_A | 0.190 |
| Original | ripper_B | 0.208 |
| Original | irep_A | 0.064 |
| Original | irep_B | 0.062 |
| Original | brl | 1.073 |
| Original | brs | 8.805 |
| Original | jrip | 0.283 |
| Original | part | 0.249 |
| Original | j48 | 0.250 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.2 | 3.94 |
| Binarized | forest | 147.2 | 3.92 |
| Binarized | ripper_A | 3.6 | 2.69 |
| Binarized | ripper_B | 3.8 | 3.74 |
| Binarized | irep_A | 3.2 | 1.97 |
| Binarized | irep_B | 3.8 | 3.25 |
| Binarized | brl | 4.4 | 1.82 |
| Binarized | brs | 7.0 | 2.91 |
| Binarized | jrip | 2.0 | 2.80 |
| Binarized | part | 45.2 | 3.99 |
| Binarized | j48 | 37.6 | 8.62 |
| Original | tree | 15.4 | 3.96 |
| Original | forest | 143.4 | 3.89 |
| Original | ripper_A | 8.4 | 2.26 |
| Original | ripper_B | 8.6 | 2.94 |
| Original | irep_A | 5.8 | 1.62 |
| Original | irep_B | 2.4 | 1.70 |
| Original | brl | 4.4 | 1.82 |
| Original | brs | 7.0 | 2.91 |
| Original | jrip | 3.0 | 2.22 |
| Original | part | 8.0 | 2.30 |
| Original | j48 | 22.0 | 6.09 |

(116.9s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

*20 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold (n/a in the tables below). `brl`/`brs` are expected to hit this on datasets with many discretized features -- see `FIT_TIMEOUT_SECONDS`'s comment; it is a scaling limit of those algorithms, not a bug.*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.64 +/- 10.06 | 0/5 |
| Binarized | forest | 75.01 +/- 5.53 | 0/5 |
| Binarized | ripper_A | 70.24 +/- 4.63 | 0/5 |
| Binarized | ripper_B | 75.52 +/- 7.20 | 0/5 |
| Binarized | irep_A | 74.02 +/- 4.51 | 0/5 |
| Binarized | irep_B | 66.32 +/- 6.70 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 70.19 +/- 6.53 | 0/5 |
| Binarized | part | 74.53 +/- 5.71 | 0/5 |
| Binarized | j48 | 74.47 +/- 5.62 | 0/5 |
| Original | tree | 75.48 +/- 4.09 | 0/5 |
| Original | forest | 76.45 +/- 4.62 | 0/5 |
| Original | ripper_A | 62.11 +/- 11.07 | 0/5 |
| Original | ripper_B | 64.44 +/- 6.02 | 0/5 |
| Original | irep_A | 51.44 +/- 4.40 | 0/5 |
| Original | irep_B | 56.72 +/- 4.30 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | n/a | 5/5 |
| Original | jrip | 72.10 +/- 3.95 | 0/5 |
| Original | part | 75.96 +/- 5.32 | 0/5 |
| Original | j48 | 71.56 +/- 7.81 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.012 |
| Binarized | forest | 0.023 |
| Binarized | ripper_A | 0.943 |
| Binarized | ripper_B | 0.978 |
| Binarized | irep_A | 0.759 |
| Binarized | irep_B | 0.763 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.461 |
| Binarized | part | 0.422 |
| Binarized | j48 | 0.349 |
| Original | tree | 0.010 |
| Original | forest | 0.046 |
| Original | ripper_A | 0.934 |
| Original | ripper_B | 0.966 |
| Original | irep_A | 0.832 |
| Original | irep_B | 0.839 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.259 |
| Original | part | 0.244 |
| Original | j48 | 0.237 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.6 | 3.82 |
| Binarized | forest | 121.4 | 3.73 |
| Binarized | ripper_A | 3.0 | 2.25 |
| Binarized | ripper_B | 3.2 | 2.12 |
| Binarized | irep_A | 2.6 | 1.50 |
| Binarized | irep_B | 2.4 | 1.47 |
| Binarized | brl | nan | nan |
| Binarized | brs | nan | nan |
| Binarized | jrip | 3.4 | 1.74 |
| Binarized | part | 6.2 | 3.59 |
| Binarized | j48 | 16.0 | 5.73 |
| Original | tree | 13.0 | 3.76 |
| Original | forest | 114.8 | 3.67 |
| Original | ripper_A | 4.4 | 1.66 |
| Original | ripper_B | 3.4 | 2.59 |
| Original | irep_A | 1.6 | 1.20 |
| Original | irep_B | 1.4 | 0.88 |
| Original | brl | nan | nan |
| Original | brs | nan | nan |
| Original | jrip | 3.6 | 1.73 |
| Original | part | 6.4 | 2.41 |
| Original | j48 | 14.0 | 4.51 |

(1255.0s total)

---

## ionosphere

n=351, attributes=33, A='b', B='g'

*4/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

*20 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold (n/a in the tables below). `brl`/`brs` are expected to hit this on datasets with many discretized features -- see `FIT_TIMEOUT_SECONDS`'s comment; it is a scaling limit of those algorithms, not a bug.*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 90.03 +/- 2.71 | 0/5 |
| Binarized | forest | 90.31 +/- 2.78 | 0/5 |
| Binarized | ripper_A | 89.75 +/- 2.44 | 0/5 |
| Binarized | ripper_B | 87.76 +/- 3.73 | 0/5 |
| Binarized | irep_A | 88.04 +/- 5.74 | 0/5 |
| Binarized | irep_B | 88.33 +/- 5.20 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 88.89 +/- 4.17 | 0/5 |
| Binarized | part | 91.46 +/- 2.97 | 0/5 |
| Binarized | j48 | 90.31 +/- 3.05 | 0/5 |
| Original | tree | 88.61 +/- 3.24 | 0/5 |
| Original | forest | 92.02 +/- 4.10 | 0/5 |
| Original | ripper_A | 88.04 +/- 4.18 | 0/5 |
| Original | ripper_B | 75.51 +/- 2.66 | 0/5 |
| Original | irep_A | 86.63 +/- 4.31 | 0/5 |
| Original | irep_B | 74.93 +/- 2.64 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | n/a | 5/5 |
| Original | jrip | 90.89 +/- 2.31 | 0/5 |
| Original | part | 90.62 +/- 4.60 | 0/5 |
| Original | j48 | 88.33 +/- 1.33 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.010 |
| Binarized | forest | 0.027 |
| Binarized | ripper_A | 0.704 |
| Binarized | ripper_B | 0.709 |
| Binarized | irep_A | 0.532 |
| Binarized | irep_B | 0.533 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.549 |
| Binarized | part | 0.470 |
| Binarized | j48 | 0.373 |
| Original | tree | 0.010 |
| Original | forest | 0.044 |
| Original | ripper_A | 0.498 |
| Original | ripper_B | 0.488 |
| Original | irep_A | 0.355 |
| Original | irep_B | 0.369 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.307 |
| Original | part | 0.292 |
| Original | j48 | 0.297 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 8.6 | 3.46 |
| Binarized | forest | 98.6 | 3.57 |
| Binarized | ripper_A | 5.6 | 1.55 |
| Binarized | ripper_B | 3.0 | 3.20 |
| Binarized | irep_A | 2.6 | 1.37 |
| Binarized | irep_B | 1.8 | 2.43 |
| Binarized | brl | nan | nan |
| Binarized | brs | nan | nan |
| Binarized | jrip | 4.2 | 1.47 |
| Binarized | part | 6.0 | 3.09 |
| Binarized | j48 | 10.8 | 4.64 |
| Original | tree | 8.8 | 3.49 |
| Original | forest | 95.4 | 3.51 |
| Original | ripper_A | 8.0 | 2.07 |
| Original | ripper_B | 9.0 | 2.47 |
| Original | irep_A | 3.0 | 1.33 |
| Original | irep_B | 1.0 | 1.00 |
| Original | brl | nan | nan |
| Original | brs | nan | nan |
| Original | jrip | 4.2 | 1.31 |
| Original | part | 5.6 | 2.83 |
| Original | j48 | 11.4 | 5.04 |

(1242.7s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 82.36 +/- 2.06 | 0/5 |
| Binarized | forest | 77.87 +/- 1.94 | 0/5 |
| Binarized | ripper_A | 98.02 +/- 1.16 | 0/5 |
| Binarized | ripper_B | 99.48 +/- 0.33 | 0/5 |
| Binarized | irep_A | 92.91 +/- 6.19 | 0/5 |
| Binarized | irep_B | 84.66 +/- 1.44 | 0/5 |
| Binarized | brl | 80.58 +/- 3.08 | 0/5 |
| Binarized | brs | 86.74 +/- 3.65 | 0/5 |
| Binarized | jrip | 98.33 +/- 1.01 | 0/5 |
| Binarized | part | 93.84 +/- 1.85 | 0/5 |
| Binarized | j48 | 94.88 +/- 0.77 | 0/5 |
| Original | tree | 82.36 +/- 2.06 | 0/5 |
| Original | forest | 77.14 +/- 1.96 | 0/5 |
| Original | ripper_A | 97.91 +/- 0.93 | 0/5 |
| Original | ripper_B | 97.39 +/- 2.26 | 0/5 |
| Original | irep_A | 92.71 +/- 7.27 | 0/5 |
| Original | irep_B | 83.61 +/- 2.92 | 0/5 |
| Original | brl | 80.58 +/- 3.08 | 0/5 |
| Original | brs | 86.74 +/- 3.65 | 0/5 |
| Original | jrip | 97.91 +/- 0.87 | 0/5 |
| Original | part | 94.15 +/- 2.35 | 0/5 |
| Original | j48 | 87.27 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.005 |
| Binarized | forest | 0.026 |
| Binarized | ripper_A | 0.173 |
| Binarized | ripper_B | 0.179 |
| Binarized | irep_A | 0.074 |
| Binarized | irep_B | 0.061 |
| Binarized | brl | 0.837 |
| Binarized | brs | 7.263 |
| Binarized | jrip | 0.451 |
| Binarized | part | 0.341 |
| Binarized | j48 | 0.289 |
| Original | tree | 0.004 |
| Original | forest | 0.017 |
| Original | ripper_A | 0.110 |
| Original | ripper_B | 0.140 |
| Original | irep_A | 0.042 |
| Original | irep_B | 0.032 |
| Original | brl | 0.770 |
| Original | brs | 7.278 |
| Original | jrip | 0.273 |
| Original | part | 0.254 |
| Original | j48 | 0.245 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.0 | 3.86 |
| Binarized | forest | 153.4 | 3.96 |
| Binarized | ripper_A | 8.6 | 3.07 |
| Binarized | ripper_B | 9.8 | 3.22 |
| Binarized | irep_A | 7.2 | 2.91 |
| Binarized | irep_B | 6.0 | 2.65 |
| Binarized | brl | 11.8 | 1.81 |
| Binarized | brs | 7.4 | 3.00 |
| Binarized | jrip | 8.0 | 3.08 |
| Binarized | part | 29.2 | 3.23 |
| Binarized | j48 | 37.8 | 5.88 |
| Original | tree | 14.0 | 3.86 |
| Original | forest | 157.4 | 3.98 |
| Original | ripper_A | 9.4 | 3.28 |
| Original | ripper_B | 14.0 | 3.13 |
| Original | irep_A | 8.0 | 3.06 |
| Original | irep_B | 5.8 | 2.23 |
| Original | brl | 11.8 | 1.81 |
| Original | brs | 7.4 | 3.00 |
| Original | jrip | 9.6 | 3.28 |
| Original | part | 36.4 | 2.69 |
| Original | j48 | 80.6 | 4.53 |

(95.8s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 95.77 +/- 0.85 | 0/5 |
| Binarized | forest | 96.35 +/- 0.87 | 0/5 |
| Binarized | ripper_A | 98.54 +/- 0.61 | 0/5 |
| Binarized | ripper_B | 98.18 +/- 0.65 | 0/5 |
| Binarized | irep_A | 94.83 +/- 1.13 | 0/5 |
| Binarized | irep_B | 97.23 +/- 1.28 | 0/5 |
| Binarized | brl | 97.16 +/- 1.34 | 0/5 |
| Binarized | brs | 96.14 +/- 1.33 | 0/5 |
| Binarized | jrip | 98.61 +/- 0.54 | 0/5 |
| Binarized | part | 98.32 +/- 0.68 | 0/5 |
| Binarized | j48 | 98.54 +/- 0.65 | 0/5 |
| Original | tree | 95.70 +/- 1.63 | 0/5 |
| Original | forest | 97.01 +/- 1.27 | 0/5 |
| Original | ripper_A | 97.09 +/- 0.86 | 0/5 |
| Original | ripper_B | 95.63 +/- 2.12 | 0/5 |
| Original | irep_A | 90.09 +/- 2.08 | 0/5 |
| Original | irep_B | 89.21 +/- 2.12 | 0/5 |
| Original | brl | 97.16 +/- 1.34 | 0/5 |
| Original | brs | 96.14 +/- 1.33 | 0/5 |
| Original | jrip | 97.89 +/- 0.63 | 0/5 |
| Original | part | 98.76 +/- 0.75 | 0/5 |
| Original | j48 | 98.61 +/- 1.25 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.005 |
| Binarized | forest | 0.024 |
| Binarized | ripper_A | 0.133 |
| Binarized | ripper_B | 0.135 |
| Binarized | irep_A | 0.059 |
| Binarized | irep_B | 0.061 |
| Binarized | brl | 0.783 |
| Binarized | brs | 7.173 |
| Binarized | jrip | 0.396 |
| Binarized | part | 0.287 |
| Binarized | j48 | 0.292 |
| Original | tree | 0.004 |
| Original | forest | 0.019 |
| Original | ripper_A | 0.352 |
| Original | ripper_B | 0.379 |
| Original | irep_A | 0.049 |
| Original | irep_B | 0.048 |
| Original | brl | 0.793 |
| Original | brs | 7.174 |
| Original | jrip | 0.270 |
| Original | part | 0.234 |
| Original | j48 | 0.234 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.0 | 3.77 |
| Binarized | forest | 131.4 | 3.81 |
| Binarized | ripper_A | 6.6 | 2.33 |
| Binarized | ripper_B | 6.2 | 2.49 |
| Binarized | irep_A | 3.8 | 1.73 |
| Binarized | irep_B | 4.4 | 2.15 |
| Binarized | brl | 5.0 | 1.89 |
| Binarized | brs | 10.2 | 2.96 |
| Binarized | jrip | 5.2 | 2.30 |
| Binarized | part | 9.4 | 2.10 |
| Binarized | j48 | 11.4 | 3.88 |
| Original | tree | 12.2 | 3.69 |
| Original | forest | 123.4 | 3.74 |
| Original | ripper_A | 33.6 | 3.65 |
| Original | ripper_B | 31.2 | 3.85 |
| Original | irep_A | 10.2 | 2.55 |
| Original | irep_B | 8.8 | 2.86 |
| Original | brl | 5.0 | 1.89 |
| Original | brs | 10.2 | 2.96 |
| Original | jrip | 6.0 | 2.24 |
| Original | part | 7.4 | 2.11 |
| Original | j48 | 15.2 | 4.51 |

(96.4s total)

---

## kr-vs-kp

n=3196, attributes=36, A='nowin', B='won'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 94.09 +/- 1.11 | 0/5 |
| Binarized | forest | 93.71 +/- 1.63 | 0/5 |
| Binarized | ripper_A | 98.97 +/- 0.57 | 0/5 |
| Binarized | ripper_B | 98.69 +/- 0.35 | 0/5 |
| Binarized | irep_A | 98.12 +/- 0.71 | 0/5 |
| Binarized | irep_B | 95.25 +/- 3.44 | 0/5 |
| Binarized | brl | 94.09 +/- 2.48 | 0/5 |
| Binarized | brs | 84.79 +/- 1.86 | 0/5 |
| Binarized | jrip | 99.12 +/- 0.38 | 0/5 |
| Binarized | part | 98.94 +/- 0.30 | 0/5 |
| Binarized | j48 | 99.28 +/- 0.38 | 0/5 |
| Original | tree | 94.09 +/- 1.11 | 0/5 |
| Original | forest | 94.15 +/- 1.12 | 0/5 |
| Original | ripper_A | 98.65 +/- 0.65 | 0/5 |
| Original | ripper_B | 98.84 +/- 0.34 | 0/5 |
| Original | irep_A | 97.31 +/- 1.52 | 0/5 |
| Original | irep_B | 93.59 +/- 3.19 | 0/5 |
| Original | brl | 94.09 +/- 2.48 | 0/5 |
| Original | brs | 84.79 +/- 1.86 | 0/5 |
| Original | jrip | 98.81 +/- 0.70 | 0/5 |
| Original | part | 98.97 +/- 0.47 | 0/5 |
| Original | j48 | 99.34 +/- 0.38 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.013 |
| Binarized | forest | 0.038 |
| Binarized | ripper_A | 0.888 |
| Binarized | ripper_B | 0.808 |
| Binarized | irep_A | 0.270 |
| Binarized | irep_B | 0.297 |
| Binarized | brl | 10.295 |
| Binarized | brs | 12.641 |
| Binarized | jrip | 0.826 |
| Binarized | part | 0.599 |
| Binarized | j48 | 0.463 |
| Original | tree | 0.007 |
| Original | forest | 0.020 |
| Original | ripper_A | 0.552 |
| Original | ripper_B | 0.472 |
| Original | irep_A | 0.159 |
| Original | irep_B | 0.156 |
| Original | brl | 10.251 |
| Original | brs | 12.690 |
| Original | jrip | 0.409 |
| Original | part | 0.296 |
| Original | j48 | 0.294 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 7.6 | 3.43 |
| Binarized | forest | 112.4 | 3.72 |
| Binarized | ripper_A | 16.6 | 3.17 |
| Binarized | ripper_B | 9.2 | 4.48 |
| Binarized | irep_A | 7.2 | 2.80 |
| Binarized | irep_B | 6.4 | 4.39 |
| Binarized | brl | 8.4 | 1.87 |
| Binarized | brs | 7.0 | 2.89 |
| Binarized | jrip | 14.0 | 3.00 |
| Binarized | part | 20.8 | 2.93 |
| Binarized | j48 | 25.8 | 7.37 |
| Original | tree | 7.6 | 3.43 |
| Original | forest | 118.6 | 3.75 |
| Original | ripper_A | 18.4 | 3.33 |
| Original | ripper_B | 10.8 | 4.80 |
| Original | irep_A | 7.2 | 2.91 |
| Original | irep_B | 5.0 | 4.01 |
| Original | brl | 8.4 | 1.87 |
| Original | brs | 7.0 | 2.89 |
| Original | jrip | 14.6 | 3.16 |
| Original | part | 20.4 | 3.21 |
| Original | j48 | 29.0 | 7.71 |

(269.3s total)

---

## mushroom

n=8124, attributes=21, A='e', B='p'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 99.22 +/- 0.14 | 0/5 |
| Binarized | forest | 99.14 +/- 0.13 | 0/5 |
| Binarized | ripper_A | 100.00 +/- 0.00 | 0/5 |
| Binarized | ripper_B | 99.98 +/- 0.05 | 0/5 |
| Binarized | irep_A | 98.52 +/- 0.24 | 0/5 |
| Binarized | irep_B | 99.91 +/- 0.07 | 0/5 |
| Binarized | brl | 99.36 +/- 0.53 | 0/5 |
| Binarized | brs | 99.82 +/- 0.17 | 0/5 |
| Binarized | jrip | 99.98 +/- 0.05 | 0/5 |
| Binarized | part | 100.00 +/- 0.00 | 0/5 |
| Binarized | j48 | 100.00 +/- 0.00 | 0/5 |
| Original | tree | 99.22 +/- 0.14 | 0/5 |
| Original | forest | 98.66 +/- 0.38 | 0/5 |
| Original | ripper_A | 100.00 +/- 0.00 | 0/5 |
| Original | ripper_B | 99.98 +/- 0.05 | 0/5 |
| Original | irep_A | 98.52 +/- 0.24 | 0/5 |
| Original | irep_B | 96.45 +/- 0.32 | 0/5 |
| Original | brl | 99.36 +/- 0.53 | 0/5 |
| Original | brs | 99.82 +/- 0.17 | 0/5 |
| Original | jrip | 100.00 +/- 0.00 | 0/5 |
| Original | part | 100.00 +/- 0.00 | 0/5 |
| Original | j48 | 100.00 +/- 0.00 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.034 |
| Binarized | forest | 0.071 |
| Binarized | ripper_A | 1.593 |
| Binarized | ripper_B | 1.598 |
| Binarized | irep_A | 0.869 |
| Binarized | irep_B | 0.966 |
| Binarized | brl | 15.373 |
| Binarized | brs | 24.540 |
| Binarized | jrip | 1.268 |
| Binarized | part | 0.837 |
| Binarized | j48 | 0.713 |
| Original | tree | 0.019 |
| Original | forest | 0.030 |
| Original | ripper_A | 0.453 |
| Original | ripper_B | 0.346 |
| Original | irep_A | 0.182 |
| Original | irep_B | 0.176 |
| Original | brl | 15.411 |
| Original | brs | 24.102 |
| Original | jrip | 0.422 |
| Original | part | 0.324 |
| Original | j48 | 0.296 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 10.0 | 3.50 |
| Binarized | forest | 101.0 | 3.59 |
| Binarized | ripper_A | 5.8 | 2.45 |
| Binarized | ripper_B | 7.0 | 1.56 |
| Binarized | irep_A | 4.0 | 1.55 |
| Binarized | irep_B | 5.2 | 2.01 |
| Binarized | brl | 10.0 | 1.85 |
| Binarized | brs | 9.6 | 2.86 |
| Binarized | jrip | 6.0 | 1.70 |
| Binarized | part | 5.0 | 2.76 |
| Binarized | j48 | 9.8 | 3.83 |
| Original | tree | 10.0 | 3.50 |
| Original | forest | 101.8 | 3.58 |
| Original | ripper_A | 7.6 | 2.54 |
| Original | ripper_B | 8.2 | 1.50 |
| Original | irep_A | 4.0 | 1.50 |
| Original | irep_B | 4.0 | 1.00 |
| Original | brl | 10.0 | 1.85 |
| Original | brs | 9.6 | 2.86 |
| Original | jrip | 8.0 | 1.56 |
| Original | part | 10.0 | 1.85 |
| Original | j48 | 24.0 | 2.54 |

(468.8s total)

---

## Overall summary

### Accuracy

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 | fallback folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 95.17 | 96.32 | 94.02 | 96.09 | 95.17 | 94.48 | 94.02 | 89.20 | 94.94 | 95.40 | 96.09 | 94.94 | 94.71 | 95.17 | 95.17 | 93.79 | 94.94 | 94.02 | 89.20 | 95.63 | 95.63 | 96.32 | 0/5 |
| breast-cancer | 72.01 | 73.43 | 61.63 | 75.52 | 74.81 | 72.02 | 73.09 | 71.34 | 73.06 | 67.14 | 69.93 | 72.72 | 75.52 | 66.79 | 73.07 | 72.72 | 69.91 | 73.09 | 71.34 | 72.02 | 69.96 | 73.42 | 0/5 |
| colic | 85.06 | 86.42 | 85.60 | 85.07 | 86.68 | 82.63 | 84.78 | 79.89 | 88.04 | 80.70 | 85.61 | 84.25 | 83.71 | 81.27 | 86.96 | 82.35 | 86.42 | 84.78 | 79.89 | 87.51 | 85.88 | 82.63 | 0/5 |
| credit-approval | 85.07 | 85.22 | 85.36 | 85.80 | 85.22 | 84.49 | 85.80 | 65.65 | 85.22 | 83.91 | 85.22 | 84.78 | 85.65 | 83.62 | 83.04 | 85.51 | 85.94 | 85.80 | 65.65 | 85.94 | 84.35 | 85.94 | 0/5 |
| credit-g | 70.70 | 73.50 | 70.80 | 58.50 | 71.50 | 70.40 | 69.80 | 45.50 | 71.60 | 72.30 | 71.90 | 70.60 | 70.50 | 70.20 | 58.60 | 70.40 | 71.20 | 69.80 | 45.50 | 72.40 | 70.00 | 70.80 | 5/5 |
| diabetes | 73.83 | 75.65 | 65.37 | 74.35 | 75.13 | 75.27 | 72.01 | 69.01 | 75.00 | 71.88 | 72.78 | 73.45 | 74.88 | 63.40 | 72.79 | 65.22 | 74.09 | 72.01 | 69.01 | 76.05 | 75.00 | 74.08 | 0/5 |
| sonar | 70.64 | 75.01 | 70.24 | 75.52 | 74.02 | 66.32 | nan | nan | 70.19 | 74.53 | 74.47 | 75.48 | 76.45 | 62.11 | 64.44 | 51.44 | 56.72 | nan | nan | 72.10 | 75.96 | 71.56 | 0/5 |
| ionosphere | 90.03 | 90.31 | 89.75 | 87.76 | 88.04 | 88.33 | nan | nan | 88.89 | 91.46 | 90.31 | 88.61 | 92.02 | 88.04 | 75.51 | 86.63 | 74.93 | nan | nan | 90.89 | 90.62 | 88.33 | 4/5 |
| tic-tac-toe | 82.36 | 77.87 | 98.02 | 99.48 | 92.91 | 84.66 | 80.58 | 86.74 | 98.33 | 93.84 | 94.88 | 82.36 | 77.14 | 97.91 | 97.39 | 92.71 | 83.61 | 80.58 | 86.74 | 97.91 | 94.15 | 87.27 | 0/5 |
| banknote-authentication | 95.77 | 96.35 | 98.54 | 98.18 | 94.83 | 97.23 | 97.16 | 96.14 | 98.61 | 98.32 | 98.54 | 95.70 | 97.01 | 97.09 | 95.63 | 90.09 | 89.21 | 97.16 | 96.14 | 97.89 | 98.76 | 98.61 | 0/5 |
| kr-vs-kp | 94.09 | 93.71 | 98.97 | 98.69 | 98.12 | 95.25 | 94.09 | 84.79 | 99.12 | 98.94 | 99.28 | 94.09 | 94.15 | 98.65 | 98.84 | 97.31 | 93.59 | 94.09 | 84.79 | 98.81 | 98.97 | 99.34 | 0/5 |
| mushroom | 99.22 | 99.14 | 100.00 | 99.98 | 98.52 | 99.91 | 99.36 | 99.82 | 99.98 | 100.00 | 100.00 | 99.22 | 98.66 | 100.00 | 99.98 | 98.52 | 96.45 | 99.36 | 99.82 | 100.00 | 100.00 | 100.00 | 0/5 |

### Fit time (seconds/fold)

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.297 | 0.023 | 0.114 | 0.099 | 0.072 | 0.068 | 0.640 | 7.796 | 0.290 | 0.252 | 0.249 | 0.003 | 0.014 | 0.058 | 0.053 | 0.034 | 0.033 | 0.530 | 7.697 | 0.205 | 0.205 | 0.207 |
| breast-cancer | 0.004 | 0.018 | 0.100 | 0.119 | 0.059 | 0.059 | 0.221 | 7.330 | 0.268 | 0.285 | 0.250 | 0.003 | 0.027 | 0.048 | 0.077 | 0.025 | 0.024 | 0.220 | 7.369 | 0.198 | 0.200 | 0.197 |
| colic | 0.007 | 0.020 | 0.401 | 0.469 | 0.300 | 0.314 | 3.917 | 37.010 | 0.397 | 0.361 | 0.312 | 0.003 | 0.014 | 0.204 | 0.221 | 0.146 | 0.146 | 3.771 | 37.140 | 0.220 | 0.224 | 0.211 |
| credit-approval | 0.006 | 0.034 | 0.321 | 0.304 | 0.145 | 0.152 | 1.574 | 11.043 | 0.420 | 0.377 | 0.320 | 0.004 | 0.016 | 0.218 | 0.231 | 0.077 | 0.081 | 1.561 | 12.178 | 0.259 | 0.264 | 0.241 |
| credit-g | 0.007 | 0.024 | 0.479 | 0.398 | 0.183 | 0.174 | 2.694 | 11.362 | 0.483 | 0.515 | 0.396 | 0.004 | 0.016 | 0.257 | 0.227 | 0.082 | 0.089 | 2.626 | 11.292 | 0.273 | 0.281 | 0.243 |
| diabetes | 0.005 | 0.023 | 0.285 | 0.272 | 0.098 | 0.106 | 1.054 | 8.796 | 0.431 | 0.398 | 0.326 | 0.004 | 0.016 | 0.190 | 0.208 | 0.064 | 0.062 | 1.073 | 8.805 | 0.283 | 0.249 | 0.250 |
| sonar | 0.012 | 0.023 | 0.943 | 0.978 | 0.759 | 0.763 | nan | nan | 0.461 | 0.422 | 0.349 | 0.010 | 0.046 | 0.934 | 0.966 | 0.832 | 0.839 | nan | nan | 0.259 | 0.244 | 0.237 |
| ionosphere | 0.010 | 0.027 | 0.704 | 0.709 | 0.532 | 0.533 | nan | nan | 0.549 | 0.470 | 0.373 | 0.010 | 0.044 | 0.498 | 0.488 | 0.355 | 0.369 | nan | nan | 0.307 | 0.292 | 0.297 |
| tic-tac-toe | 0.005 | 0.026 | 0.173 | 0.179 | 0.074 | 0.061 | 0.837 | 7.263 | 0.451 | 0.341 | 0.289 | 0.004 | 0.017 | 0.110 | 0.140 | 0.042 | 0.032 | 0.770 | 7.278 | 0.273 | 0.254 | 0.245 |
| banknote-authentication | 0.005 | 0.024 | 0.133 | 0.135 | 0.059 | 0.061 | 0.783 | 7.173 | 0.396 | 0.287 | 0.292 | 0.004 | 0.019 | 0.352 | 0.379 | 0.049 | 0.048 | 0.793 | 7.174 | 0.270 | 0.234 | 0.234 |
| kr-vs-kp | 0.013 | 0.038 | 0.888 | 0.808 | 0.270 | 0.297 | 10.295 | 12.641 | 0.826 | 0.599 | 0.463 | 0.007 | 0.020 | 0.552 | 0.472 | 0.159 | 0.156 | 10.251 | 12.690 | 0.409 | 0.296 | 0.294 |
| mushroom | 0.034 | 0.071 | 1.593 | 1.598 | 0.869 | 0.966 | 15.373 | 24.540 | 1.268 | 0.837 | 0.713 | 0.019 | 0.030 | 0.453 | 0.346 | 0.182 | 0.176 | 15.411 | 24.102 | 0.422 | 0.324 | 0.296 |

### Rule count

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 13.4 | 111.8 | 2.0 | 3.0 | 2.2 | 1.2 | 3.8 | 7.4 | 2.2 | 5.2 | 6.8 | 13.4 | 121.0 | 3.4 | 3.8 | 1.8 | 1.4 | 3.8 | 7.4 | 1.2 | 5.2 | 9.0 |
| breast-cancer | 12.4 | 126.0 | 1.2 | 1.4 | 2.2 | 1.4 | 3.2 | 7.6 | 1.0 | 22.0 | 15.0 | 12.4 | 124.8 | 3.0 | 1.6 | 2.4 | 1.2 | 3.2 | 7.6 | 2.8 | 15.4 | 8.4 |
| colic | 11.4 | 121.2 | 4.4 | 5.2 | 2.0 | 2.4 | 4.8 | 7.8 | 3.0 | 13.6 | 15.6 | 11.6 | 118.6 | 6.0 | 6.2 | 2.2 | 1.4 | 4.8 | 7.8 | 3.4 | 12.8 | 8.6 |
| credit-approval | 13.8 | 142.4 | 5.0 | 3.8 | 1.2 | 2.2 | 4.4 | 8.8 | 3.0 | 26.8 | 17.0 | 14.2 | 134.0 | 7.6 | 8.8 | 1.2 | 2.0 | 4.4 | 8.8 | 3.8 | 29.6 | 19.4 |
| credit-g | 15.0 | 146.0 | 3.2 | 3.8 | 3.0 | 4.2 | 4.8 | 8.6 | 2.2 | 56.8 | 78.8 | 15.2 | 141.8 | 5.4 | 5.2 | 1.8 | 5.2 | 4.8 | 8.6 | 3.4 | 60.4 | 77.8 |
| diabetes | 15.2 | 147.2 | 3.6 | 3.8 | 3.2 | 3.8 | 4.4 | 7.0 | 2.0 | 45.2 | 37.6 | 15.4 | 143.4 | 8.4 | 8.6 | 5.8 | 2.4 | 4.4 | 7.0 | 3.0 | 8.0 | 22.0 |
| sonar | 13.6 | 121.4 | 3.0 | 3.2 | 2.6 | 2.4 | nan | nan | 3.4 | 6.2 | 16.0 | 13.0 | 114.8 | 4.4 | 3.4 | 1.6 | 1.4 | nan | nan | 3.6 | 6.4 | 14.0 |
| ionosphere | 8.6 | 98.6 | 5.6 | 3.0 | 2.6 | 1.8 | nan | nan | 4.2 | 6.0 | 10.8 | 8.8 | 95.4 | 8.0 | 9.0 | 3.0 | 1.0 | nan | nan | 4.2 | 5.6 | 11.4 |
| tic-tac-toe | 14.0 | 153.4 | 8.6 | 9.8 | 7.2 | 6.0 | 11.8 | 7.4 | 8.0 | 29.2 | 37.8 | 14.0 | 157.4 | 9.4 | 14.0 | 8.0 | 5.8 | 11.8 | 7.4 | 9.6 | 36.4 | 80.6 |
| banknote-authentication | 13.0 | 131.4 | 6.6 | 6.2 | 3.8 | 4.4 | 5.0 | 10.2 | 5.2 | 9.4 | 11.4 | 12.2 | 123.4 | 33.6 | 31.2 | 10.2 | 8.8 | 5.0 | 10.2 | 6.0 | 7.4 | 15.2 |
| kr-vs-kp | 7.6 | 112.4 | 16.6 | 9.2 | 7.2 | 6.4 | 8.4 | 7.0 | 14.0 | 20.8 | 25.8 | 7.6 | 118.6 | 18.4 | 10.8 | 7.2 | 5.0 | 8.4 | 7.0 | 14.6 | 20.4 | 29.0 |
| mushroom | 10.0 | 101.0 | 5.8 | 7.0 | 4.0 | 5.2 | 10.0 | 9.6 | 6.0 | 5.0 | 9.8 | 10.0 | 101.8 | 7.6 | 8.2 | 4.0 | 4.0 | 10.0 | 9.6 | 8.0 | 10.0 | 24.0 |

### Average conditions per rule

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 3.80 | 3.67 | 1.53 | 2.25 | 1.73 | 1.10 | 1.48 | 2.97 | 1.60 | 2.23 | 3.32 | 3.80 | 3.73 | 1.88 | 1.95 | 1.17 | 1.10 | 1.48 | 2.97 | 1.20 | 1.42 | 2.61 |
| breast-cancer | 3.75 | 3.78 | 1.50 | 2.70 | 1.27 | 2.40 | 1.42 | 2.95 | 1.80 | 4.54 | 5.37 | 3.75 | 3.78 | 1.55 | 2.70 | 1.13 | 1.90 | 1.42 | 2.95 | 2.22 | 1.92 | 1.77 |
| colic | 3.68 | 3.74 | 2.15 | 1.90 | 1.20 | 1.70 | 1.46 | 2.79 | 1.40 | 4.53 | 5.97 | 3.72 | 3.73 | 1.61 | 2.13 | 1.33 | 1.70 | 1.46 | 2.79 | 1.83 | 1.81 | 2.17 |
| credit-approval | 3.85 | 3.89 | 2.78 | 2.40 | 1.10 | 1.97 | 1.66 | 2.91 | 1.67 | 3.49 | 6.31 | 3.87 | 3.84 | 2.86 | 3.22 | 1.20 | 1.63 | 1.66 | 2.91 | 2.18 | 2.37 | 4.02 |
| credit-g | 3.93 | 3.91 | 4.83 | 2.95 | 3.40 | 1.66 | 1.67 | 2.86 | 3.40 | 4.09 | 11.42 | 3.95 | 3.89 | 3.64 | 2.68 | 2.97 | 1.33 | 1.67 | 2.86 | 2.69 | 3.02 | 5.57 |
| diabetes | 3.94 | 3.92 | 2.69 | 3.74 | 1.97 | 3.25 | 1.82 | 2.91 | 2.80 | 3.99 | 8.62 | 3.96 | 3.89 | 2.26 | 2.94 | 1.62 | 1.70 | 1.82 | 2.91 | 2.22 | 2.30 | 6.09 |
| sonar | 3.82 | 3.73 | 2.25 | 2.12 | 1.50 | 1.47 | nan | nan | 1.74 | 3.59 | 5.73 | 3.76 | 3.67 | 1.66 | 2.59 | 1.20 | 0.88 | nan | nan | 1.73 | 2.41 | 4.51 |
| ionosphere | 3.46 | 3.57 | 1.55 | 3.20 | 1.37 | 2.43 | nan | nan | 1.47 | 3.09 | 4.64 | 3.49 | 3.51 | 2.07 | 2.47 | 1.33 | 1.00 | nan | nan | 1.31 | 2.83 | 5.04 |
| tic-tac-toe | 3.86 | 3.96 | 3.07 | 3.22 | 2.91 | 2.65 | 1.81 | 3.00 | 3.08 | 3.23 | 5.88 | 3.86 | 3.98 | 3.28 | 3.13 | 3.06 | 2.23 | 1.81 | 3.00 | 3.28 | 2.69 | 4.53 |
| banknote-authentication | 3.77 | 3.81 | 2.33 | 2.49 | 1.73 | 2.15 | 1.89 | 2.96 | 2.30 | 2.10 | 3.88 | 3.69 | 3.74 | 3.65 | 3.85 | 2.55 | 2.86 | 1.89 | 2.96 | 2.24 | 2.11 | 4.51 |
| kr-vs-kp | 3.43 | 3.72 | 3.17 | 4.48 | 2.80 | 4.39 | 1.87 | 2.89 | 3.00 | 2.93 | 7.37 | 3.43 | 3.75 | 3.33 | 4.80 | 2.91 | 4.01 | 1.87 | 2.89 | 3.16 | 3.21 | 7.71 |
| mushroom | 3.50 | 3.59 | 2.45 | 1.56 | 1.55 | 2.01 | 1.85 | 2.86 | 1.70 | 2.76 | 3.83 | 3.50 | 3.58 | 2.54 | 1.50 | 1.50 | 1.00 | 1.85 | 2.86 | 1.56 | 1.85 | 2.54 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| algorithm | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| Binarized/brl | 85.07 | 3.739 | 6.1 | 1.69 | 2 |
| Original/brl | 85.07 | 3.701 | 6.1 | 1.69 | 2 |
| Binarized/brs | 78.81 | 13.495 | 8.1 | 2.91 | 2 |
| Original/brs | 78.81 | 13.573 | 8.1 | 2.91 | 2 |
| Binarized/forest | 85.24 | 0.029 | 126.1 | 3.77 | 0 |
| Original/forest | 85.03 | 0.023 | 124.6 | 3.76 | 0 |
| Binarized/irep_A | 86.25 | 0.285 | 3.4 | 1.88 | 0 |
| Original/irep_A | 82.22 | 0.171 | 4.1 | 1.83 | 0 |
| Binarized/irep_B | 84.25 | 0.296 | 3.4 | 2.26 | 0 |
| Original/irep_B | 81.42 | 0.171 | 3.3 | 1.78 | 0 |
| Binarized/j48 | 86.59 | 0.361 | 23.5 | 6.03 | 0 |
| Original/j48 | 85.69 | 0.246 | 26.6 | 4.25 | 0 |
| Binarized/jrip | 86.92 | 0.520 | 4.5 | 2.16 | 0 |
| Original/jrip | 87.26 | 0.282 | 5.3 | 2.14 | 0 |
| Binarized/part | 85.70 | 0.429 | 20.5 | 3.38 | 0 |
| Original/part | 86.61 | 0.256 | 18.1 | 2.33 | 0 |
| Binarized/ripper_A | 84.86 | 0.511 | 5.5 | 2.52 | 0 |
| Original/ripper_A | 83.69 | 0.323 | 9.6 | 2.53 | 0 |
| Binarized/ripper_B | 86.24 | 0.506 | 5.0 | 2.75 | 0 |
| Original/ripper_B | 83.45 | 0.317 | 9.2 | 2.83 | 0 |
| Binarized/tree | 84.50 | 0.034 | 12.3 | 3.73 | 0 |
| Original/tree | 84.68 | 0.006 | 12.3 | 3.73 | 0 |

**Average rank per criterion** (1 = best of 22; failed entries tie for last)

| algorithm | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| Binarized/brl | 14.71 | 19.58 | 11.67 | 6.75 |
| Original/brl | 14.71 | 19.25 | 11.67 | 6.75 |
| Binarized/brs | 18.67 | 21.17 | 13.00 | 13.58 |
| Original/brs | 18.67 | 21.50 | 13.00 | 13.58 |
| Binarized/forest | 9.00 | 3.67 | 21.00 | 17.92 |
| Original/forest | 10.67 | 3.25 | 20.67 | 17.33 |
| Binarized/irep_A | 10.04 | 9.42 | 3.54 | 4.46 |
| Original/irep_A | 15.88 | 6.33 | 5.08 | 4.42 |
| Binarized/irep_B | 12.54 | 10.00 | 3.62 | 7.83 |
| Original/irep_B | 14.12 | 6.17 | 3.17 | 4.92 |
| Binarized/j48 | 7.46 | 13.67 | 17.25 | 20.75 |
| Original/j48 | 7.08 | 9.92 | 17.67 | 17.75 |
| Binarized/jrip | 6.92 | 16.67 | 5.50 | 7.88 |
| Original/jrip | 5.25 | 11.33 | 7.21 | 7.50 |
| Binarized/part | 9.83 | 15.17 | 14.96 | 14.42 |
| Original/part | 7.54 | 10.75 | 15.25 | 8.58 |
| Binarized/ripper_A | 10.21 | 14.50 | 7.00 | 10.17 |
| Original/ripper_A | 13.75 | 10.67 | 11.83 | 10.25 |
| Binarized/ripper_B | 7.62 | 14.50 | 7.42 | 11.71 |
| Original/ripper_B | 12.12 | 11.08 | 12.17 | 12.04 |
| Binarized/tree | 13.00 | 3.42 | 15.04 | 17.12 |
| Original/tree | 13.21 | 1.00 | 15.29 | 17.29 |

**Binarized vs. Original: how often each was better, per model** (ties count 0.5 each per side; a dataset where both failed isn't counted for either side)

| model | accuracy (Bin / Orig) | fit time (Bin / Orig) | n_rules (Bin / Orig) | avg_conditions (Bin / Orig) |
|---|---|---|---|---|
| brl | 5.0 / 5.0 | 3.0 / 7.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| brs | 5.0 / 5.0 | 7.0 / 3.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| forest | 6.0 / 6.0 | 3.0 / 9.0 | 4.0 / 8.0 | 4.0 / 8.0 |
| irep_A | 10.5 / 1.5 | 1.0 / 11.0 | 7.5 / 4.5 | 5.0 / 7.0 |
| irep_B | 8.0 / 4.0 | 1.0 / 11.0 | 3.0 / 9.0 | 2.0 / 10.0 |
| j48 | 5.5 / 6.5 | 0.0 / 12.0 | 7.0 / 5.0 | 3.0 / 9.0 |
| jrip | 5.0 / 7.0 | 0.0 / 12.0 | 10.5 / 1.5 | 5.0 / 7.0 |
| part | 2.5 / 9.5 | 0.0 / 12.0 | 5.5 / 6.5 | 2.0 / 10.0 |
| ripper_A | 9.5 / 2.5 | 1.0 / 11.0 | 12.0 / 0.0 | 8.0 / 4.0 |
| ripper_B | 8.5 / 3.5 | 1.0 / 11.0 | 12.0 / 0.0 | 5.5 / 6.5 |
| tree | 8.5 / 3.5 | 0.0 / 12.0 | 7.5 / 4.5 | 7.5 / 4.5 |
