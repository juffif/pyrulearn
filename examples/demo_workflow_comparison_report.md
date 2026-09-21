# Binarized vs. Original data preparation -- comparison across binary datasets

Generated 2026-09-21 15:27:35. N_FOLDS=5, MAX_INTERVALS=8 ('Binarized' mode only), MAX_DEPTH=4, RIPPER_K=2, N_ESTIMATORS=10 (forest), FIT_TIMEOUT_SECONDS=60. `ripper_A`/`irep_A` treat each dataset's (alphabetically) first class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit the same already-Boolean matrix under both workflows as two independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` fit-time includes JVM subprocess startup overhead, not just the algorithm itself -- see the module docstring.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 95.17 +/- 0.86 | 0/5 |
| Binarized | forest | 96.09 +/- 1.56 | 0/5 |
| Binarized | ripper_A | 94.02 +/- 1.84 | 0/5 |
| Binarized | ripper_B | 96.09 +/- 1.17 | 0/5 |
| Binarized | irep_A | 94.94 +/- 2.37 | 0/5 |
| Binarized | irep_B | 94.71 +/- 2.68 | 0/5 |
| Binarized | jrip | 94.94 +/- 1.72 | 0/5 |
| Binarized | part | 95.40 +/- 1.26 | 0/5 |
| Binarized | j48 | 96.09 +/- 1.17 | 0/5 |
| Original | tree | 94.94 +/- 1.17 | 0/5 |
| Original | forest | 94.71 +/- 2.13 | 0/5 |
| Original | ripper_A | 95.17 +/- 1.13 | 0/5 |
| Original | ripper_B | 95.17 +/- 1.52 | 0/5 |
| Original | irep_A | 94.02 +/- 1.52 | 0/5 |
| Original | irep_B | 95.17 +/- 2.76 | 0/5 |
| Original | jrip | 95.63 +/- 2.34 | 0/5 |
| Original | part | 95.63 +/- 1.34 | 0/5 |
| Original | j48 | 96.32 +/- 1.34 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.252 |
| Binarized | forest | 0.078 |
| Binarized | ripper_A | 0.124 |
| Binarized | ripper_B | 0.125 |
| Binarized | irep_A | 0.091 |
| Binarized | irep_B | 0.086 |
| Binarized | jrip | 0.307 |
| Binarized | part | 0.302 |
| Binarized | j48 | 0.298 |
| Original | tree | 0.004 |
| Original | forest | 0.069 |
| Original | ripper_A | 0.072 |
| Original | ripper_B | 0.070 |
| Original | irep_A | 0.049 |
| Original | irep_B | 0.045 |
| Original | jrip | 0.236 |
| Original | part | 0.246 |
| Original | j48 | 0.240 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.4 | 3.80 |
| Binarized | forest | 111.8 | 3.67 |
| Binarized | ripper_A | 2.0 | 1.53 |
| Binarized | ripper_B | 3.0 | 2.25 |
| Binarized | irep_A | 2.0 | 1.50 |
| Binarized | irep_B | 1.0 | 1.00 |
| Binarized | jrip | 2.2 | 1.60 |
| Binarized | part | 5.2 | 2.23 |
| Binarized | j48 | 6.8 | 3.32 |
| Original | tree | 13.4 | 3.80 |
| Original | forest | 121.0 | 3.73 |
| Original | ripper_A | 3.4 | 1.88 |
| Original | ripper_B | 3.8 | 1.95 |
| Original | irep_A | 2.2 | 1.37 |
| Original | irep_B | 1.4 | 1.10 |
| Original | jrip | 1.2 | 1.20 |
| Original | part | 5.2 | 1.42 |
| Original | j48 | 9.0 | 2.61 |

(15.1s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 72.01 +/- 2.36 | 0/5 |
| Binarized | forest | 74.48 +/- 1.37 | 0/5 |
| Binarized | ripper_A | 61.63 +/- 13.98 | 0/5 |
| Binarized | ripper_B | 75.52 +/- 2.49 | 0/5 |
| Binarized | irep_A | 75.17 +/- 1.44 | 0/5 |
| Binarized | irep_B | 71.65 +/- 6.12 | 0/5 |
| Binarized | jrip | 73.06 +/- 5.36 | 0/5 |
| Binarized | part | 67.14 +/- 4.10 | 0/5 |
| Binarized | j48 | 69.93 +/- 2.76 | 0/5 |
| Original | tree | 72.72 +/- 1.92 | 0/5 |
| Original | forest | 75.51 +/- 3.43 | 0/5 |
| Original | ripper_A | 66.79 +/- 2.79 | 0/5 |
| Original | ripper_B | 73.07 +/- 2.21 | 0/5 |
| Original | irep_A | 75.87 +/- 2.63 | 0/5 |
| Original | irep_B | 73.75 +/- 4.68 | 0/5 |
| Original | jrip | 72.02 +/- 3.56 | 0/5 |
| Original | part | 69.96 +/- 5.62 | 0/5 |
| Original | j48 | 73.42 +/- 1.45 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.008 |
| Binarized | forest | 0.050 |
| Binarized | ripper_A | 0.129 |
| Binarized | ripper_B | 0.149 |
| Binarized | irep_A | 0.077 |
| Binarized | irep_B | 0.072 |
| Binarized | jrip | 0.304 |
| Binarized | part | 0.337 |
| Binarized | j48 | 0.295 |
| Original | tree | 0.004 |
| Original | forest | 0.056 |
| Original | ripper_A | 0.060 |
| Original | ripper_B | 0.092 |
| Original | irep_A | 0.031 |
| Original | irep_B | 0.036 |
| Original | jrip | 0.227 |
| Original | part | 0.234 |
| Original | j48 | 0.231 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.4 | 3.75 |
| Binarized | forest | 126.0 | 3.78 |
| Binarized | ripper_A | 1.2 | 1.50 |
| Binarized | ripper_B | 1.4 | 2.70 |
| Binarized | irep_A | 2.4 | 1.27 |
| Binarized | irep_B | 1.4 | 2.07 |
| Binarized | jrip | 1.0 | 1.80 |
| Binarized | part | 22.0 | 4.54 |
| Binarized | j48 | 15.0 | 5.37 |
| Original | tree | 12.4 | 3.75 |
| Original | forest | 124.8 | 3.78 |
| Original | ripper_A | 3.0 | 1.55 |
| Original | ripper_B | 1.6 | 2.70 |
| Original | irep_A | 2.2 | 1.07 |
| Original | irep_B | 2.8 | 2.25 |
| Original | jrip | 2.8 | 2.22 |
| Original | part | 15.4 | 1.92 |
| Original | j48 | 8.4 | 1.77 |

(13.3s total)

---

## colic

n=368, attributes=26, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 85.06 +/- 2.66 | 0/5 |
| Binarized | forest | 85.34 +/- 2.28 | 0/5 |
| Binarized | ripper_A | 85.60 +/- 3.04 | 0/5 |
| Binarized | ripper_B | 85.07 +/- 2.79 | 0/5 |
| Binarized | irep_A | 83.71 +/- 2.78 | 0/5 |
| Binarized | irep_B | 82.08 +/- 3.17 | 0/5 |
| Binarized | jrip | 88.04 +/- 3.26 | 0/5 |
| Binarized | part | 80.70 +/- 4.83 | 0/5 |
| Binarized | j48 | 85.61 +/- 2.63 | 0/5 |
| Original | tree | 84.25 +/- 3.11 | 0/5 |
| Original | forest | 84.26 +/- 3.31 | 0/5 |
| Original | ripper_A | 81.27 +/- 3.39 | 0/5 |
| Original | ripper_B | 86.96 +/- 2.91 | 0/5 |
| Original | irep_A | 84.53 +/- 2.57 | 0/5 |
| Original | irep_B | 82.90 +/- 4.01 | 0/5 |
| Original | jrip | 87.51 +/- 3.34 | 0/5 |
| Original | part | 85.88 +/- 2.47 | 0/5 |
| Original | j48 | 82.63 +/- 5.17 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.013 |
| Binarized | forest | 0.063 |
| Binarized | ripper_A | 0.667 |
| Binarized | ripper_B | 0.685 |
| Binarized | irep_A | 0.458 |
| Binarized | irep_B | 0.446 |
| Binarized | jrip | 0.526 |
| Binarized | part | 0.487 |
| Binarized | j48 | 0.390 |
| Original | tree | 0.004 |
| Original | forest | 0.089 |
| Original | ripper_A | 0.343 |
| Original | ripper_B | 0.366 |
| Original | irep_A | 0.247 |
| Original | irep_B | 0.249 |
| Original | jrip | 0.280 |
| Original | part | 0.278 |
| Original | j48 | 0.277 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 11.4 | 3.68 |
| Binarized | forest | 121.2 | 3.74 |
| Binarized | ripper_A | 4.4 | 2.15 |
| Binarized | ripper_B | 5.2 | 1.90 |
| Binarized | irep_A | 1.8 | 1.20 |
| Binarized | irep_B | 1.6 | 1.20 |
| Binarized | jrip | 3.0 | 1.40 |
| Binarized | part | 13.6 | 4.53 |
| Binarized | j48 | 15.6 | 5.97 |
| Original | tree | 11.6 | 3.72 |
| Original | forest | 118.6 | 3.73 |
| Original | ripper_A | 6.0 | 1.61 |
| Original | ripper_B | 6.2 | 2.13 |
| Original | irep_A | 3.0 | 1.42 |
| Original | irep_B | 2.2 | 1.37 |
| Original | jrip | 3.4 | 1.83 |
| Original | part | 12.8 | 1.81 |
| Original | j48 | 8.6 | 2.17 |

(34.3s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 85.07 +/- 3.39 | 0/5 |
| Binarized | forest | 85.65 +/- 2.80 | 0/5 |
| Binarized | ripper_A | 85.36 +/- 2.69 | 0/5 |
| Binarized | ripper_B | 85.80 +/- 2.73 | 0/5 |
| Binarized | irep_A | 85.22 +/- 2.88 | 0/5 |
| Binarized | irep_B | 85.65 +/- 2.48 | 0/5 |
| Binarized | jrip | 85.22 +/- 2.58 | 0/5 |
| Binarized | part | 83.91 +/- 3.29 | 0/5 |
| Binarized | j48 | 85.22 +/- 3.32 | 0/5 |
| Original | tree | 84.78 +/- 3.89 | 0/5 |
| Original | forest | 85.51 +/- 3.37 | 0/5 |
| Original | ripper_A | 83.62 +/- 3.57 | 0/5 |
| Original | ripper_B | 83.04 +/- 3.63 | 0/5 |
| Original | irep_A | 85.51 +/- 3.30 | 0/5 |
| Original | irep_B | 85.80 +/- 3.39 | 0/5 |
| Original | jrip | 85.94 +/- 2.58 | 0/5 |
| Original | part | 84.35 +/- 3.51 | 0/5 |
| Original | j48 | 85.94 +/- 4.09 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.104 |
| Binarized | ripper_A | 0.509 |
| Binarized | ripper_B | 0.471 |
| Binarized | irep_A | 0.223 |
| Binarized | irep_B | 0.238 |
| Binarized | jrip | 0.447 |
| Binarized | part | 0.440 |
| Binarized | j48 | 0.378 |
| Original | tree | 0.005 |
| Original | forest | 0.138 |
| Original | ripper_A | 0.316 |
| Original | ripper_B | 0.366 |
| Original | irep_A | 0.116 |
| Original | irep_B | 0.131 |
| Original | jrip | 0.314 |
| Original | part | 0.295 |
| Original | j48 | 0.265 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.8 | 3.85 |
| Binarized | forest | 142.4 | 3.89 |
| Binarized | ripper_A | 5.0 | 2.78 |
| Binarized | ripper_B | 3.8 | 2.40 |
| Binarized | irep_A | 1.2 | 1.10 |
| Binarized | irep_B | 2.4 | 2.17 |
| Binarized | jrip | 3.0 | 1.67 |
| Binarized | part | 26.8 | 3.49 |
| Binarized | j48 | 17.0 | 6.31 |
| Original | tree | 14.2 | 3.87 |
| Original | forest | 134.0 | 3.84 |
| Original | ripper_A | 7.6 | 2.86 |
| Original | ripper_B | 8.8 | 3.22 |
| Original | irep_A | 1.0 | 1.00 |
| Original | irep_B | 1.8 | 1.87 |
| Original | jrip | 3.8 | 2.18 |
| Original | part | 29.6 | 2.37 |
| Original | j48 | 19.4 | 4.02 |

(28.3s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*5/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.70 +/- 0.98 | 0/5 |
| Binarized | forest | 73.00 +/- 0.84 | 0/5 |
| Binarized | ripper_A | 70.80 +/- 3.50 | 0/5 |
| Binarized | ripper_B | 58.50 +/- 6.73 | 0/5 |
| Binarized | irep_A | 71.40 +/- 1.46 | 0/5 |
| Binarized | irep_B | 71.30 +/- 0.51 | 0/5 |
| Binarized | jrip | 71.60 +/- 1.07 | 0/5 |
| Binarized | part | 72.30 +/- 3.23 | 0/5 |
| Binarized | j48 | 71.90 +/- 1.24 | 0/5 |
| Original | tree | 70.60 +/- 2.60 | 0/5 |
| Original | forest | 70.10 +/- 1.16 | 0/5 |
| Original | ripper_A | 70.20 +/- 1.03 | 0/5 |
| Original | ripper_B | 58.60 +/- 5.07 | 0/5 |
| Original | irep_A | 69.50 +/- 1.00 | 0/5 |
| Original | irep_B | 68.00 +/- 4.93 | 0/5 |
| Original | jrip | 72.40 +/- 2.40 | 0/5 |
| Original | part | 70.00 +/- 3.36 | 0/5 |
| Original | j48 | 70.80 +/- 1.91 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.021 |
| Binarized | forest | 0.134 |
| Binarized | ripper_A | 0.834 |
| Binarized | ripper_B | 0.679 |
| Binarized | irep_A | 0.261 |
| Binarized | irep_B | 0.294 |
| Binarized | jrip | 0.587 |
| Binarized | part | 0.604 |
| Binarized | j48 | 0.449 |
| Original | tree | 0.006 |
| Original | forest | 0.199 |
| Original | ripper_A | 0.390 |
| Original | ripper_B | 0.348 |
| Original | irep_A | 0.124 |
| Original | irep_B | 0.145 |
| Original | jrip | 0.312 |
| Original | part | 0.335 |
| Original | j48 | 0.285 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.0 | 3.93 |
| Binarized | forest | 146.0 | 3.91 |
| Binarized | ripper_A | 3.2 | 4.83 |
| Binarized | ripper_B | 3.8 | 2.95 |
| Binarized | irep_A | 1.4 | 2.50 |
| Binarized | irep_B | 5.8 | 1.55 |
| Binarized | jrip | 2.2 | 3.40 |
| Binarized | part | 56.8 | 4.09 |
| Binarized | j48 | 78.8 | 11.42 |
| Original | tree | 15.2 | 3.95 |
| Original | forest | 141.8 | 3.89 |
| Original | ripper_A | 5.4 | 3.64 |
| Original | ripper_B | 5.2 | 2.68 |
| Original | irep_A | 1.4 | 1.87 |
| Original | irep_B | 3.2 | 1.16 |
| Original | jrip | 3.4 | 2.69 |
| Original | part | 60.4 | 3.02 |
| Original | j48 | 77.8 | 5.57 |

(35.2s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 73.83 +/- 2.78 | 0/5 |
| Binarized | forest | 76.30 +/- 1.01 | 0/5 |
| Binarized | ripper_A | 65.37 +/- 2.70 | 0/5 |
| Binarized | ripper_B | 74.35 +/- 2.04 | 0/5 |
| Binarized | irep_A | 75.52 +/- 1.52 | 0/5 |
| Binarized | irep_B | 73.05 +/- 3.14 | 0/5 |
| Binarized | jrip | 75.00 +/- 3.53 | 0/5 |
| Binarized | part | 71.88 +/- 2.23 | 0/5 |
| Binarized | j48 | 72.78 +/- 2.34 | 0/5 |
| Original | tree | 73.45 +/- 3.14 | 0/5 |
| Original | forest | 74.49 +/- 3.36 | 0/5 |
| Original | ripper_A | 63.40 +/- 4.26 | 0/5 |
| Original | ripper_B | 72.79 +/- 3.11 | 0/5 |
| Original | irep_A | 69.92 +/- 3.99 | 0/5 |
| Original | irep_B | 73.31 +/- 2.93 | 0/5 |
| Original | jrip | 76.05 +/- 3.07 | 0/5 |
| Original | part | 75.00 +/- 2.66 | 0/5 |
| Original | j48 | 74.08 +/- 2.42 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.112 |
| Binarized | ripper_A | 0.444 |
| Binarized | ripper_B | 0.441 |
| Binarized | irep_A | 0.152 |
| Binarized | irep_B | 0.161 |
| Binarized | jrip | 0.482 |
| Binarized | part | 0.436 |
| Binarized | j48 | 0.361 |
| Original | tree | 0.005 |
| Original | forest | 0.183 |
| Original | ripper_A | 0.280 |
| Original | ripper_B | 0.288 |
| Original | irep_A | 0.100 |
| Original | irep_B | 0.091 |
| Original | jrip | 0.279 |
| Original | part | 0.256 |
| Original | j48 | 0.257 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.2 | 3.94 |
| Binarized | forest | 147.2 | 3.92 |
| Binarized | ripper_A | 3.6 | 2.69 |
| Binarized | ripper_B | 3.8 | 3.74 |
| Binarized | irep_A | 3.0 | 1.77 |
| Binarized | irep_B | 2.4 | 2.43 |
| Binarized | jrip | 2.0 | 2.80 |
| Binarized | part | 45.2 | 3.99 |
| Binarized | j48 | 37.6 | 8.62 |
| Original | tree | 15.4 | 3.96 |
| Original | forest | 143.4 | 3.89 |
| Original | ripper_A | 8.4 | 2.26 |
| Original | ripper_B | 8.6 | 2.94 |
| Original | irep_A | 7.4 | 1.82 |
| Original | irep_B | 4.0 | 2.32 |
| Original | jrip | 3.0 | 2.22 |
| Original | part | 8.0 | 2.30 |
| Original | j48 | 22.0 | 6.09 |

(25.5s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.64 +/- 10.06 | 0/5 |
| Binarized | forest | 77.90 +/- 6.61 | 0/5 |
| Binarized | ripper_A | 70.24 +/- 4.63 | 0/5 |
| Binarized | ripper_B | 75.52 +/- 7.20 | 0/5 |
| Binarized | irep_A | 75.01 +/- 2.81 | 0/5 |
| Binarized | irep_B | 72.61 +/- 4.83 | 0/5 |
| Binarized | jrip | 70.19 +/- 6.53 | 0/5 |
| Binarized | part | 74.53 +/- 5.71 | 0/5 |
| Binarized | j48 | 74.47 +/- 5.62 | 0/5 |
| Original | tree | 75.48 +/- 4.09 | 0/5 |
| Original | forest | 78.37 +/- 4.50 | 0/5 |
| Original | ripper_A | 62.11 +/- 11.07 | 0/5 |
| Original | ripper_B | 64.44 +/- 6.02 | 0/5 |
| Original | irep_A | 49.98 +/- 3.13 | 0/5 |
| Original | irep_B | 61.09 +/- 8.82 | 0/5 |
| Original | jrip | 72.10 +/- 3.95 | 0/5 |
| Original | part | 75.96 +/- 5.32 | 0/5 |
| Original | j48 | 71.56 +/- 7.81 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.022 |
| Binarized | forest | 0.068 |
| Binarized | ripper_A | 1.450 |
| Binarized | ripper_B | 1.409 |
| Binarized | irep_A | 1.154 |
| Binarized | irep_B | 1.124 |
| Binarized | jrip | 0.500 |
| Binarized | part | 0.438 |
| Binarized | j48 | 0.384 |
| Original | tree | 0.007 |
| Original | forest | 0.076 |
| Original | ripper_A | 1.352 |
| Original | ripper_B | 1.275 |
| Original | irep_A | 1.130 |
| Original | irep_B | 1.124 |
| Original | jrip | 0.283 |
| Original | part | 0.275 |
| Original | j48 | 0.277 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.6 | 3.82 |
| Binarized | forest | 121.4 | 3.73 |
| Binarized | ripper_A | 3.0 | 2.25 |
| Binarized | ripper_B | 3.2 | 2.12 |
| Binarized | irep_A | 2.0 | 1.43 |
| Binarized | irep_B | 1.2 | 1.30 |
| Binarized | jrip | 3.4 | 1.74 |
| Binarized | part | 6.2 | 3.59 |
| Binarized | j48 | 16.0 | 5.73 |
| Original | tree | 13.0 | 3.76 |
| Original | forest | 114.8 | 3.67 |
| Original | ripper_A | 4.4 | 1.66 |
| Original | ripper_B | 3.4 | 2.59 |
| Original | irep_A | 2.4 | 1.55 |
| Original | irep_B | 2.6 | 1.27 |
| Original | jrip | 3.6 | 1.73 |
| Original | part | 6.4 | 2.41 |
| Original | j48 | 14.0 | 4.51 |

(68.8s total)

---

## ionosphere

n=351, attributes=33, A='b', B='g'

*4/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 90.03 +/- 2.71 | 0/5 |
| Binarized | forest | 91.74 +/- 2.75 | 0/5 |
| Binarized | ripper_A | 89.75 +/- 2.44 | 0/5 |
| Binarized | ripper_B | 87.76 +/- 3.73 | 0/5 |
| Binarized | irep_A | 90.03 +/- 4.52 | 0/5 |
| Binarized | irep_B | 88.89 +/- 2.75 | 0/5 |
| Binarized | jrip | 88.89 +/- 4.17 | 0/5 |
| Binarized | part | 91.46 +/- 2.97 | 0/5 |
| Binarized | j48 | 90.31 +/- 3.05 | 0/5 |
| Original | tree | 88.61 +/- 3.24 | 0/5 |
| Original | forest | 91.45 +/- 3.61 | 0/5 |
| Original | ripper_A | 88.04 +/- 4.18 | 0/5 |
| Original | ripper_B | 75.51 +/- 2.66 | 0/5 |
| Original | irep_A | 87.46 +/- 4.09 | 0/5 |
| Original | irep_B | 74.93 +/- 2.64 | 0/5 |
| Original | jrip | 90.89 +/- 2.31 | 0/5 |
| Original | part | 90.62 +/- 4.60 | 0/5 |
| Original | j48 | 88.33 +/- 1.33 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.060 |
| Binarized | ripper_A | 0.788 |
| Binarized | ripper_B | 0.801 |
| Binarized | irep_A | 0.589 |
| Binarized | irep_B | 0.562 |
| Binarized | jrip | 0.483 |
| Binarized | part | 0.429 |
| Binarized | j48 | 0.388 |
| Original | tree | 0.006 |
| Original | forest | 0.079 |
| Original | ripper_A | 0.540 |
| Original | ripper_B | 0.588 |
| Original | irep_A | 0.468 |
| Original | irep_B | 0.428 |
| Original | jrip | 0.293 |
| Original | part | 0.276 |
| Original | j48 | 0.279 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 8.6 | 3.46 |
| Binarized | forest | 98.6 | 3.57 |
| Binarized | ripper_A | 5.6 | 1.55 |
| Binarized | ripper_B | 3.0 | 3.20 |
| Binarized | irep_A | 2.0 | 1.20 |
| Binarized | irep_B | 1.8 | 2.43 |
| Binarized | jrip | 4.2 | 1.47 |
| Binarized | part | 6.0 | 3.09 |
| Binarized | j48 | 10.8 | 4.64 |
| Original | tree | 8.8 | 3.49 |
| Original | forest | 95.4 | 3.51 |
| Original | ripper_A | 8.0 | 2.07 |
| Original | ripper_B | 9.0 | 2.47 |
| Original | irep_A | 3.8 | 1.49 |
| Original | irep_B | 1.0 | 1.00 |
| Original | jrip | 4.2 | 1.31 |
| Original | part | 5.6 | 2.83 |
| Original | j48 | 11.4 | 5.04 |

(40.4s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 82.36 +/- 2.06 | 0/5 |
| Binarized | forest | 77.04 +/- 1.35 | 0/5 |
| Binarized | ripper_A | 98.02 +/- 1.16 | 0/5 |
| Binarized | ripper_B | 99.48 +/- 0.33 | 0/5 |
| Binarized | irep_A | 92.59 +/- 7.17 | 0/5 |
| Binarized | irep_B | 79.12 +/- 7.44 | 0/5 |
| Binarized | jrip | 98.33 +/- 1.01 | 0/5 |
| Binarized | part | 93.84 +/- 1.85 | 0/5 |
| Binarized | j48 | 94.88 +/- 0.77 | 0/5 |
| Original | tree | 82.36 +/- 2.06 | 0/5 |
| Original | forest | 76.83 +/- 2.17 | 0/5 |
| Original | ripper_A | 97.91 +/- 0.93 | 0/5 |
| Original | ripper_B | 97.39 +/- 2.26 | 0/5 |
| Original | irep_A | 85.80 +/- 5.60 | 0/5 |
| Original | irep_B | 80.26 +/- 5.55 | 0/5 |
| Original | jrip | 97.91 +/- 0.87 | 0/5 |
| Original | part | 94.15 +/- 2.35 | 0/5 |
| Original | j48 | 87.27 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.132 |
| Binarized | ripper_A | 0.256 |
| Binarized | ripper_B | 0.273 |
| Binarized | irep_A | 0.103 |
| Binarized | irep_B | 0.094 |
| Binarized | jrip | 0.459 |
| Binarized | part | 0.359 |
| Binarized | j48 | 0.307 |
| Original | tree | 0.005 |
| Original | forest | 0.166 |
| Original | ripper_A | 0.155 |
| Original | ripper_B | 0.191 |
| Original | irep_A | 0.052 |
| Original | irep_B | 0.048 |
| Original | jrip | 0.294 |
| Original | part | 0.265 |
| Original | j48 | 0.265 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.0 | 3.86 |
| Binarized | forest | 153.4 | 3.96 |
| Binarized | ripper_A | 8.6 | 3.07 |
| Binarized | ripper_B | 9.8 | 3.22 |
| Binarized | irep_A | 7.4 | 2.95 |
| Binarized | irep_B | 6.0 | 2.68 |
| Binarized | jrip | 8.0 | 3.08 |
| Binarized | part | 29.2 | 3.23 |
| Binarized | j48 | 37.8 | 5.88 |
| Original | tree | 14.0 | 3.86 |
| Original | forest | 157.4 | 3.98 |
| Original | ripper_A | 9.4 | 3.28 |
| Original | ripper_B | 14.0 | 3.13 |
| Original | irep_A | 5.4 | 2.76 |
| Original | irep_B | 6.4 | 2.14 |
| Original | jrip | 9.6 | 3.28 |
| Original | part | 36.4 | 2.69 |
| Original | j48 | 80.6 | 4.53 |

(19.8s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 95.77 +/- 0.85 | 0/5 |
| Binarized | forest | 96.28 +/- 0.85 | 0/5 |
| Binarized | ripper_A | 98.54 +/- 0.61 | 0/5 |
| Binarized | ripper_B | 98.18 +/- 0.65 | 0/5 |
| Binarized | irep_A | 93.73 +/- 0.90 | 0/5 |
| Binarized | irep_B | 96.94 +/- 1.36 | 0/5 |
| Binarized | jrip | 98.61 +/- 0.54 | 0/5 |
| Binarized | part | 98.32 +/- 0.68 | 0/5 |
| Binarized | j48 | 98.54 +/- 0.65 | 0/5 |
| Original | tree | 95.70 +/- 1.63 | 0/5 |
| Original | forest | 97.01 +/- 1.27 | 0/5 |
| Original | ripper_A | 97.09 +/- 0.86 | 0/5 |
| Original | ripper_B | 95.63 +/- 2.12 | 0/5 |
| Original | irep_A | 89.94 +/- 3.13 | 0/5 |
| Original | irep_B | 88.19 +/- 1.64 | 0/5 |
| Original | jrip | 97.89 +/- 0.63 | 0/5 |
| Original | part | 98.76 +/- 0.75 | 0/5 |
| Original | j48 | 98.61 +/- 1.25 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.018 |
| Binarized | forest | 0.163 |
| Binarized | ripper_A | 0.213 |
| Binarized | ripper_B | 0.215 |
| Binarized | irep_A | 0.094 |
| Binarized | irep_B | 0.100 |
| Binarized | jrip | 0.446 |
| Binarized | part | 0.341 |
| Binarized | j48 | 0.336 |
| Original | tree | 0.006 |
| Original | forest | 0.264 |
| Original | ripper_A | 0.514 |
| Original | ripper_B | 0.563 |
| Original | irep_A | 0.072 |
| Original | irep_B | 0.078 |
| Original | jrip | 0.303 |
| Original | part | 0.274 |
| Original | j48 | 0.272 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.0 | 3.77 |
| Binarized | forest | 131.4 | 3.81 |
| Binarized | ripper_A | 6.6 | 2.33 |
| Binarized | ripper_B | 6.2 | 2.49 |
| Binarized | irep_A | 3.2 | 1.68 |
| Binarized | irep_B | 4.2 | 2.08 |
| Binarized | jrip | 5.2 | 2.30 |
| Binarized | part | 9.4 | 2.10 |
| Binarized | j48 | 11.4 | 3.88 |
| Original | tree | 12.2 | 3.69 |
| Original | forest | 123.4 | 3.74 |
| Original | ripper_A | 33.6 | 3.65 |
| Original | ripper_B | 31.2 | 3.85 |
| Original | irep_A | 9.6 | 2.53 |
| Original | irep_B | 10.2 | 2.81 |
| Original | jrip | 6.0 | 2.24 |
| Original | part | 7.4 | 2.11 |
| Original | j48 | 15.2 | 4.51 |

(25.4s total)

---

## kr-vs-kp

n=3196, attributes=36, A='nowin', B='won'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 94.09 +/- 1.11 | 0/5 |
| Binarized | forest | 93.74 +/- 1.31 | 0/5 |
| Binarized | ripper_A | 98.97 +/- 0.57 | 0/5 |
| Binarized | ripper_B | 98.69 +/- 0.35 | 0/5 |
| Binarized | irep_A | 98.03 +/- 0.89 | 0/5 |
| Binarized | irep_B | 93.49 +/- 3.01 | 0/5 |
| Binarized | jrip | 99.12 +/- 0.38 | 0/5 |
| Binarized | part | 98.94 +/- 0.30 | 0/5 |
| Binarized | j48 | 99.28 +/- 0.38 | 0/5 |
| Original | tree | 94.09 +/- 1.11 | 0/5 |
| Original | forest | 94.12 +/- 1.11 | 0/5 |
| Original | ripper_A | 98.65 +/- 0.65 | 0/5 |
| Original | ripper_B | 98.84 +/- 0.34 | 0/5 |
| Original | irep_A | 97.75 +/- 1.23 | 0/5 |
| Original | irep_B | 93.81 +/- 3.36 | 0/5 |
| Original | jrip | 98.81 +/- 0.70 | 0/5 |
| Original | part | 98.97 +/- 0.47 | 0/5 |
| Original | j48 | 99.34 +/- 0.38 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.043 |
| Binarized | forest | 0.270 |
| Binarized | ripper_A | 1.398 |
| Binarized | ripper_B | 1.204 |
| Binarized | irep_A | 0.418 |
| Binarized | irep_B | 0.420 |
| Binarized | jrip | 1.013 |
| Binarized | part | 0.748 |
| Binarized | j48 | 0.518 |
| Original | tree | 0.011 |
| Original | forest | 0.353 |
| Original | ripper_A | 0.806 |
| Original | ripper_B | 0.705 |
| Original | irep_A | 0.245 |
| Original | irep_B | 0.236 |
| Original | jrip | 0.435 |
| Original | part | 0.321 |
| Original | j48 | 0.317 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 7.6 | 3.43 |
| Binarized | forest | 112.4 | 3.72 |
| Binarized | ripper_A | 16.6 | 3.17 |
| Binarized | ripper_B | 9.2 | 4.48 |
| Binarized | irep_A | 7.4 | 2.78 |
| Binarized | irep_B | 5.2 | 4.04 |
| Binarized | jrip | 14.0 | 3.00 |
| Binarized | part | 20.8 | 2.93 |
| Binarized | j48 | 25.8 | 7.37 |
| Original | tree | 7.6 | 3.43 |
| Original | forest | 118.6 | 3.75 |
| Original | ripper_A | 18.4 | 3.33 |
| Original | ripper_B | 10.8 | 4.80 |
| Original | irep_A | 7.2 | 2.80 |
| Original | irep_B | 5.4 | 4.24 |
| Original | jrip | 14.6 | 3.16 |
| Original | part | 20.4 | 3.21 |
| Original | j48 | 29.0 | 7.71 |

(59.0s total)

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
| Binarized | irep_B | 99.95 +/- 0.06 | 0/5 |
| Binarized | jrip | 99.98 +/- 0.05 | 0/5 |
| Binarized | part | 100.00 +/- 0.00 | 0/5 |
| Binarized | j48 | 100.00 +/- 0.00 | 0/5 |
| Original | tree | 99.22 +/- 0.14 | 0/5 |
| Original | forest | 98.66 +/- 0.38 | 0/5 |
| Original | ripper_A | 100.00 +/- 0.00 | 0/5 |
| Original | ripper_B | 99.98 +/- 0.05 | 0/5 |
| Original | irep_A | 98.52 +/- 0.24 | 0/5 |
| Original | irep_B | 96.45 +/- 0.32 | 0/5 |
| Original | jrip | 100.00 +/- 0.00 | 0/5 |
| Original | part | 100.00 +/- 0.00 | 0/5 |
| Original | j48 | 100.00 +/- 0.00 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.087 |
| Binarized | forest | 0.542 |
| Binarized | ripper_A | 2.327 |
| Binarized | ripper_B | 2.400 |
| Binarized | irep_A | 1.315 |
| Binarized | irep_B | 1.402 |
| Binarized | jrip | 1.602 |
| Binarized | part | 0.982 |
| Binarized | j48 | 0.848 |
| Original | tree | 0.027 |
| Original | forest | 0.879 |
| Original | ripper_A | 0.636 |
| Original | ripper_B | 0.519 |
| Original | irep_A | 0.300 |
| Original | irep_B | 0.249 |
| Original | jrip | 0.459 |
| Original | part | 0.366 |
| Original | j48 | 0.333 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 10.0 | 3.50 |
| Binarized | forest | 101.0 | 3.59 |
| Binarized | ripper_A | 5.8 | 2.45 |
| Binarized | ripper_B | 7.0 | 1.56 |
| Binarized | irep_A | 4.0 | 1.50 |
| Binarized | irep_B | 5.0 | 1.84 |
| Binarized | jrip | 6.0 | 1.70 |
| Binarized | part | 5.0 | 2.76 |
| Binarized | j48 | 9.8 | 3.83 |
| Original | tree | 10.0 | 3.50 |
| Original | forest | 101.8 | 3.58 |
| Original | ripper_A | 7.6 | 2.54 |
| Original | ripper_B | 8.2 | 1.50 |
| Original | irep_A | 4.0 | 1.50 |
| Original | irep_B | 4.0 | 1.00 |
| Original | jrip | 8.0 | 1.56 |
| Original | part | 10.0 | 1.85 |
| Original | j48 | 24.0 | 2.54 |

(111.2s total)

---

## Overall summary

### Accuracy

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_jrip | Original_part | Original_j48 | fallback folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 95.17 | 96.09 | 94.02 | 96.09 | 94.94 | 94.71 | 94.94 | 95.40 | 96.09 | 94.94 | 94.71 | 95.17 | 95.17 | 94.02 | 95.17 | 95.63 | 95.63 | 96.32 | 0/5 |
| breast-cancer | 72.01 | 74.48 | 61.63 | 75.52 | 75.17 | 71.65 | 73.06 | 67.14 | 69.93 | 72.72 | 75.51 | 66.79 | 73.07 | 75.87 | 73.75 | 72.02 | 69.96 | 73.42 | 0/5 |
| colic | 85.06 | 85.34 | 85.60 | 85.07 | 83.71 | 82.08 | 88.04 | 80.70 | 85.61 | 84.25 | 84.26 | 81.27 | 86.96 | 84.53 | 82.90 | 87.51 | 85.88 | 82.63 | 0/5 |
| credit-approval | 85.07 | 85.65 | 85.36 | 85.80 | 85.22 | 85.65 | 85.22 | 83.91 | 85.22 | 84.78 | 85.51 | 83.62 | 83.04 | 85.51 | 85.80 | 85.94 | 84.35 | 85.94 | 0/5 |
| credit-g | 70.70 | 73.00 | 70.80 | 58.50 | 71.40 | 71.30 | 71.60 | 72.30 | 71.90 | 70.60 | 70.10 | 70.20 | 58.60 | 69.50 | 68.00 | 72.40 | 70.00 | 70.80 | 5/5 |
| diabetes | 73.83 | 76.30 | 65.37 | 74.35 | 75.52 | 73.05 | 75.00 | 71.88 | 72.78 | 73.45 | 74.49 | 63.40 | 72.79 | 69.92 | 73.31 | 76.05 | 75.00 | 74.08 | 0/5 |
| sonar | 70.64 | 77.90 | 70.24 | 75.52 | 75.01 | 72.61 | 70.19 | 74.53 | 74.47 | 75.48 | 78.37 | 62.11 | 64.44 | 49.98 | 61.09 | 72.10 | 75.96 | 71.56 | 0/5 |
| ionosphere | 90.03 | 91.74 | 89.75 | 87.76 | 90.03 | 88.89 | 88.89 | 91.46 | 90.31 | 88.61 | 91.45 | 88.04 | 75.51 | 87.46 | 74.93 | 90.89 | 90.62 | 88.33 | 4/5 |
| tic-tac-toe | 82.36 | 77.04 | 98.02 | 99.48 | 92.59 | 79.12 | 98.33 | 93.84 | 94.88 | 82.36 | 76.83 | 97.91 | 97.39 | 85.80 | 80.26 | 97.91 | 94.15 | 87.27 | 0/5 |
| banknote-authentication | 95.77 | 96.28 | 98.54 | 98.18 | 93.73 | 96.94 | 98.61 | 98.32 | 98.54 | 95.70 | 97.01 | 97.09 | 95.63 | 89.94 | 88.19 | 97.89 | 98.76 | 98.61 | 0/5 |
| kr-vs-kp | 94.09 | 93.74 | 98.97 | 98.69 | 98.03 | 93.49 | 99.12 | 98.94 | 99.28 | 94.09 | 94.12 | 98.65 | 98.84 | 97.75 | 93.81 | 98.81 | 98.97 | 99.34 | 0/5 |
| mushroom | 99.22 | 99.14 | 100.00 | 99.98 | 98.52 | 99.95 | 99.98 | 100.00 | 100.00 | 99.22 | 98.66 | 100.00 | 99.98 | 98.52 | 96.45 | 100.00 | 100.00 | 100.00 | 0/5 |

### Fit time (seconds/fold)

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.252 | 0.078 | 0.124 | 0.125 | 0.091 | 0.086 | 0.307 | 0.302 | 0.298 | 0.004 | 0.069 | 0.072 | 0.070 | 0.049 | 0.045 | 0.236 | 0.246 | 0.240 |
| breast-cancer | 0.008 | 0.050 | 0.129 | 0.149 | 0.077 | 0.072 | 0.304 | 0.337 | 0.295 | 0.004 | 0.056 | 0.060 | 0.092 | 0.031 | 0.036 | 0.227 | 0.234 | 0.231 |
| colic | 0.013 | 0.063 | 0.667 | 0.685 | 0.458 | 0.446 | 0.526 | 0.487 | 0.390 | 0.004 | 0.089 | 0.343 | 0.366 | 0.247 | 0.249 | 0.280 | 0.278 | 0.277 |
| credit-approval | 0.015 | 0.104 | 0.509 | 0.471 | 0.223 | 0.238 | 0.447 | 0.440 | 0.378 | 0.005 | 0.138 | 0.316 | 0.366 | 0.116 | 0.131 | 0.314 | 0.295 | 0.265 |
| credit-g | 0.021 | 0.134 | 0.834 | 0.679 | 0.261 | 0.294 | 0.587 | 0.604 | 0.449 | 0.006 | 0.199 | 0.390 | 0.348 | 0.124 | 0.145 | 0.312 | 0.335 | 0.285 |
| diabetes | 0.015 | 0.112 | 0.444 | 0.441 | 0.152 | 0.161 | 0.482 | 0.436 | 0.361 | 0.005 | 0.183 | 0.280 | 0.288 | 0.100 | 0.091 | 0.279 | 0.256 | 0.257 |
| sonar | 0.022 | 0.068 | 1.450 | 1.409 | 1.154 | 1.124 | 0.500 | 0.438 | 0.384 | 0.007 | 0.076 | 1.352 | 1.275 | 1.130 | 1.124 | 0.283 | 0.275 | 0.277 |
| ionosphere | 0.015 | 0.060 | 0.788 | 0.801 | 0.589 | 0.562 | 0.483 | 0.429 | 0.388 | 0.006 | 0.079 | 0.540 | 0.588 | 0.468 | 0.428 | 0.293 | 0.276 | 0.279 |
| tic-tac-toe | 0.015 | 0.132 | 0.256 | 0.273 | 0.103 | 0.094 | 0.459 | 0.359 | 0.307 | 0.005 | 0.166 | 0.155 | 0.191 | 0.052 | 0.048 | 0.294 | 0.265 | 0.265 |
| banknote-authentication | 0.018 | 0.163 | 0.213 | 0.215 | 0.094 | 0.100 | 0.446 | 0.341 | 0.336 | 0.006 | 0.264 | 0.514 | 0.563 | 0.072 | 0.078 | 0.303 | 0.274 | 0.272 |
| kr-vs-kp | 0.043 | 0.270 | 1.398 | 1.204 | 0.418 | 0.420 | 1.013 | 0.748 | 0.518 | 0.011 | 0.353 | 0.806 | 0.705 | 0.245 | 0.236 | 0.435 | 0.321 | 0.317 |
| mushroom | 0.087 | 0.542 | 2.327 | 2.400 | 1.315 | 1.402 | 1.602 | 0.982 | 0.848 | 0.027 | 0.879 | 0.636 | 0.519 | 0.300 | 0.249 | 0.459 | 0.366 | 0.333 |

### Rule count

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 13.4 | 111.8 | 2.0 | 3.0 | 2.0 | 1.0 | 2.2 | 5.2 | 6.8 | 13.4 | 121.0 | 3.4 | 3.8 | 2.2 | 1.4 | 1.2 | 5.2 | 9.0 |
| breast-cancer | 12.4 | 126.0 | 1.2 | 1.4 | 2.4 | 1.4 | 1.0 | 22.0 | 15.0 | 12.4 | 124.8 | 3.0 | 1.6 | 2.2 | 2.8 | 2.8 | 15.4 | 8.4 |
| colic | 11.4 | 121.2 | 4.4 | 5.2 | 1.8 | 1.6 | 3.0 | 13.6 | 15.6 | 11.6 | 118.6 | 6.0 | 6.2 | 3.0 | 2.2 | 3.4 | 12.8 | 8.6 |
| credit-approval | 13.8 | 142.4 | 5.0 | 3.8 | 1.2 | 2.4 | 3.0 | 26.8 | 17.0 | 14.2 | 134.0 | 7.6 | 8.8 | 1.0 | 1.8 | 3.8 | 29.6 | 19.4 |
| credit-g | 15.0 | 146.0 | 3.2 | 3.8 | 1.4 | 5.8 | 2.2 | 56.8 | 78.8 | 15.2 | 141.8 | 5.4 | 5.2 | 1.4 | 3.2 | 3.4 | 60.4 | 77.8 |
| diabetes | 15.2 | 147.2 | 3.6 | 3.8 | 3.0 | 2.4 | 2.0 | 45.2 | 37.6 | 15.4 | 143.4 | 8.4 | 8.6 | 7.4 | 4.0 | 3.0 | 8.0 | 22.0 |
| sonar | 13.6 | 121.4 | 3.0 | 3.2 | 2.0 | 1.2 | 3.4 | 6.2 | 16.0 | 13.0 | 114.8 | 4.4 | 3.4 | 2.4 | 2.6 | 3.6 | 6.4 | 14.0 |
| ionosphere | 8.6 | 98.6 | 5.6 | 3.0 | 2.0 | 1.8 | 4.2 | 6.0 | 10.8 | 8.8 | 95.4 | 8.0 | 9.0 | 3.8 | 1.0 | 4.2 | 5.6 | 11.4 |
| tic-tac-toe | 14.0 | 153.4 | 8.6 | 9.8 | 7.4 | 6.0 | 8.0 | 29.2 | 37.8 | 14.0 | 157.4 | 9.4 | 14.0 | 5.4 | 6.4 | 9.6 | 36.4 | 80.6 |
| banknote-authentication | 13.0 | 131.4 | 6.6 | 6.2 | 3.2 | 4.2 | 5.2 | 9.4 | 11.4 | 12.2 | 123.4 | 33.6 | 31.2 | 9.6 | 10.2 | 6.0 | 7.4 | 15.2 |
| kr-vs-kp | 7.6 | 112.4 | 16.6 | 9.2 | 7.4 | 5.2 | 14.0 | 20.8 | 25.8 | 7.6 | 118.6 | 18.4 | 10.8 | 7.2 | 5.4 | 14.6 | 20.4 | 29.0 |
| mushroom | 10.0 | 101.0 | 5.8 | 7.0 | 4.0 | 5.0 | 6.0 | 5.0 | 9.8 | 10.0 | 101.8 | 7.6 | 8.2 | 4.0 | 4.0 | 8.0 | 10.0 | 24.0 |

### Average conditions per rule

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 3.80 | 3.67 | 1.53 | 2.25 | 1.50 | 1.00 | 1.60 | 2.23 | 3.32 | 3.80 | 3.73 | 1.88 | 1.95 | 1.37 | 1.10 | 1.20 | 1.42 | 2.61 |
| breast-cancer | 3.75 | 3.78 | 1.50 | 2.70 | 1.27 | 2.07 | 1.80 | 4.54 | 5.37 | 3.75 | 3.78 | 1.55 | 2.70 | 1.07 | 2.25 | 2.22 | 1.92 | 1.77 |
| colic | 3.68 | 3.74 | 2.15 | 1.90 | 1.20 | 1.20 | 1.40 | 4.53 | 5.97 | 3.72 | 3.73 | 1.61 | 2.13 | 1.42 | 1.37 | 1.83 | 1.81 | 2.17 |
| credit-approval | 3.85 | 3.89 | 2.78 | 2.40 | 1.10 | 2.17 | 1.67 | 3.49 | 6.31 | 3.87 | 3.84 | 2.86 | 3.22 | 1.00 | 1.87 | 2.18 | 2.37 | 4.02 |
| credit-g | 3.93 | 3.91 | 4.83 | 2.95 | 2.50 | 1.55 | 3.40 | 4.09 | 11.42 | 3.95 | 3.89 | 3.64 | 2.68 | 1.87 | 1.16 | 2.69 | 3.02 | 5.57 |
| diabetes | 3.94 | 3.92 | 2.69 | 3.74 | 1.77 | 2.43 | 2.80 | 3.99 | 8.62 | 3.96 | 3.89 | 2.26 | 2.94 | 1.82 | 2.32 | 2.22 | 2.30 | 6.09 |
| sonar | 3.82 | 3.73 | 2.25 | 2.12 | 1.43 | 1.30 | 1.74 | 3.59 | 5.73 | 3.76 | 3.67 | 1.66 | 2.59 | 1.55 | 1.27 | 1.73 | 2.41 | 4.51 |
| ionosphere | 3.46 | 3.57 | 1.55 | 3.20 | 1.20 | 2.43 | 1.47 | 3.09 | 4.64 | 3.49 | 3.51 | 2.07 | 2.47 | 1.49 | 1.00 | 1.31 | 2.83 | 5.04 |
| tic-tac-toe | 3.86 | 3.96 | 3.07 | 3.22 | 2.95 | 2.68 | 3.08 | 3.23 | 5.88 | 3.86 | 3.98 | 3.28 | 3.13 | 2.76 | 2.14 | 3.28 | 2.69 | 4.53 |
| banknote-authentication | 3.77 | 3.81 | 2.33 | 2.49 | 1.68 | 2.08 | 2.30 | 2.10 | 3.88 | 3.69 | 3.74 | 3.65 | 3.85 | 2.53 | 2.81 | 2.24 | 2.11 | 4.51 |
| kr-vs-kp | 3.43 | 3.72 | 3.17 | 4.48 | 2.78 | 4.04 | 3.00 | 2.93 | 7.37 | 3.43 | 3.75 | 3.33 | 4.80 | 2.80 | 4.24 | 3.16 | 3.21 | 7.71 |
| mushroom | 3.50 | 3.59 | 2.45 | 1.56 | 1.50 | 1.84 | 1.70 | 2.76 | 3.83 | 3.50 | 3.58 | 2.54 | 1.50 | 1.50 | 1.00 | 1.56 | 1.85 | 2.54 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| algorithm | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| Binarized/forest | 85.56 | 0.148 | 126.1 | 3.77 | 0 |
| Original/forest | 85.08 | 0.213 | 124.6 | 3.76 | 0 |
| Binarized/irep_A | 86.16 | 0.411 | 3.2 | 1.74 | 0 |
| Original/irep_A | 82.40 | 0.244 | 4.1 | 1.76 | 0 |
| Binarized/irep_B | 84.12 | 0.417 | 3.2 | 2.07 | 0 |
| Original/irep_B | 81.14 | 0.238 | 3.7 | 1.88 | 0 |
| Binarized/j48 | 86.59 | 0.413 | 23.5 | 6.03 | 0 |
| Original/j48 | 85.69 | 0.275 | 26.6 | 4.25 | 0 |
| Binarized/jrip | 86.92 | 0.596 | 4.5 | 2.16 | 0 |
| Original/jrip | 87.26 | 0.310 | 5.3 | 2.14 | 0 |
| Binarized/part | 85.70 | 0.492 | 20.5 | 3.38 | 0 |
| Original/part | 86.61 | 0.285 | 18.1 | 2.33 | 0 |
| Binarized/ripper_A | 84.86 | 0.762 | 5.5 | 2.52 | 0 |
| Original/ripper_A | 83.69 | 0.456 | 9.6 | 2.53 | 0 |
| Binarized/ripper_B | 86.24 | 0.738 | 5.0 | 2.75 | 0 |
| Original/ripper_B | 83.45 | 0.448 | 9.2 | 2.83 | 0 |
| Binarized/tree | 84.50 | 0.044 | 12.3 | 3.73 | 0 |
| Original/tree | 84.68 | 0.007 | 12.3 | 3.73 | 0 |

**Average rank per criterion** (1 = best of 18; failed entries tie for last)

| algorithm | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| Binarized/forest | 7.04 | 5.08 | 17.67 | 14.58 |
| Original/forest | 9.67 | 6.75 | 17.33 | 14.00 |
| Binarized/irep_A | 9.71 | 9.67 | 3.04 | 2.67 |
| Original/irep_A | 13.21 | 5.08 | 4.21 | 3.62 |
| Binarized/irep_B | 12.04 | 9.58 | 2.83 | 4.96 |
| Original/irep_B | 13.46 | 4.83 | 4.25 | 4.50 |
| Binarized/j48 | 7.04 | 12.92 | 14.25 | 17.42 |
| Original/j48 | 7.00 | 8.92 | 14.33 | 15.08 |
| Binarized/jrip | 7.00 | 15.67 | 4.92 | 6.17 |
| Original/jrip | 5.25 | 10.50 | 6.67 | 5.92 |
| Binarized/part | 9.08 | 14.50 | 12.25 | 11.67 |
| Original/part | 6.79 | 9.58 | 12.33 | 6.83 |
| Binarized/ripper_A | 9.58 | 15.00 | 6.12 | 8.17 |
| Original/ripper_A | 12.29 | 11.58 | 10.08 | 8.00 |
| Binarized/ripper_B | 7.29 | 15.33 | 6.50 | 9.71 |
| Original/ripper_B | 11.46 | 11.92 | 10.04 | 9.96 |
| Binarized/tree | 11.29 | 3.08 | 11.96 | 13.79 |
| Original/tree | 11.79 | 1.00 | 12.21 | 13.96 |

**Binarized vs. Original: how often each was better, per model** (ties count 0.5 each per side; a dataset where both failed isn't counted for either side)

| model | accuracy (Bin / Orig) | fit time (Bin / Orig) | n_rules (Bin / Orig) | avg_conditions (Bin / Orig) |
|---|---|---|---|---|
| forest | 8.0 / 4.0 | 11.0 / 1.0 | 4.0 / 8.0 | 4.0 / 8.0 |
| irep_A | 8.5 / 3.5 | 0.0 / 12.0 | 7.0 / 5.0 | 6.5 / 5.5 |
| irep_B | 5.0 / 7.0 | 0.0 / 12.0 | 8.0 / 4.0 | 5.0 / 7.0 |
| j48 | 5.5 / 6.5 | 0.0 / 12.0 | 7.0 / 5.0 | 3.0 / 9.0 |
| jrip | 5.0 / 7.0 | 0.0 / 12.0 | 10.5 / 1.5 | 5.0 / 7.0 |
| part | 2.5 / 9.5 | 0.0 / 12.0 | 5.5 / 6.5 | 2.0 / 10.0 |
| ripper_A | 9.5 / 2.5 | 1.0 / 11.0 | 12.0 / 0.0 | 8.0 / 4.0 |
| ripper_B | 8.5 / 3.5 | 1.0 / 11.0 | 12.0 / 0.0 | 5.5 / 6.5 |
| tree | 8.5 / 3.5 | 0.0 / 12.0 | 7.5 / 4.5 | 7.5 / 4.5 |
