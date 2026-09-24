# Binarized vs. Original data preparation -- comparison across binary datasets

Generated 2026-09-24 23:18:21. N_FOLDS=5, MAX_INTERVALS=8 ('Binarized' mode only), MAX_DEPTH=4, RIPPER_K=2, N_ESTIMATORS=10 (forest), FIT_TIMEOUT_SECONDS=60. `ripper_A`/`irep_A` treat each dataset's (alphabetically) first class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit the same already-Boolean matrix under both workflows as two independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` fit-time includes JVM subprocess startup overhead, not just the algorithm itself -- see the module docstring.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 89.43 +/- 2.85 | 0/5 |
| Binarized | forest | 95.63 +/- 1.98 | 0/5 |
| Binarized | ripper_A | 93.56 +/- 1.38 | 0/5 |
| Binarized | ripper_B | 95.63 +/- 1.52 | 0/5 |
| Binarized | irep_A | 94.25 +/- 1.26 | 0/5 |
| Binarized | irep_B | 94.71 +/- 2.13 | 0/5 |
| Binarized | brl | 94.48 +/- 1.84 | 0/5 |
| Binarized | brs | 89.89 +/- 4.20 | 0/5 |
| Binarized | jrip | 95.40 +/- 1.03 | 0/5 |
| Binarized | part | 95.17 +/- 1.13 | 0/5 |
| Binarized | j48 | 96.09 +/- 1.17 | 0/5 |
| Original | tree | 94.94 +/- 1.17 | 0/5 |
| Original | forest | 94.71 +/- 2.13 | 0/5 |
| Original | ripper_A | 95.17 +/- 1.13 | 0/5 |
| Original | ripper_B | 95.17 +/- 1.52 | 0/5 |
| Original | irep_A | 94.25 +/- 1.26 | 0/5 |
| Original | irep_B | 95.63 +/- 2.34 | 0/5 |
| Original | brl | 94.48 +/- 1.84 | 0/5 |
| Original | brs | 89.89 +/- 4.20 | 0/5 |
| Original | jrip | 95.63 +/- 2.34 | 0/5 |
| Original | part | 95.63 +/- 1.34 | 0/5 |
| Original | j48 | 96.32 +/- 1.34 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.251 |
| Binarized | forest | 0.080 |
| Binarized | ripper_A | 0.061 |
| Binarized | ripper_B | 0.054 |
| Binarized | irep_A | 0.032 |
| Binarized | irep_B | 0.031 |
| Binarized | brl | 0.593 |
| Binarized | brs | 8.250 |
| Binarized | jrip | 0.422 |
| Binarized | part | 1.283 |
| Binarized | j48 | 0.409 |
| Original | tree | 0.004 |
| Original | forest | 0.064 |
| Original | ripper_A | 0.069 |
| Original | ripper_B | 0.060 |
| Original | irep_A | 0.041 |
| Original | irep_B | 0.038 |
| Original | brl | 0.525 |
| Original | brs | 8.032 |
| Original | jrip | 0.580 |
| Original | part | 1.087 |
| Original | j48 | 0.557 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.8 | 3.59 |
| Binarized | forest | 126.4 | 3.73 |
| Binarized | ripper_A | 2.4 | 1.73 |
| Binarized | ripper_B | 3.0 | 2.33 |
| Binarized | irep_A | 2.0 | 1.70 |
| Binarized | irep_B | 1.6 | 1.43 |
| Binarized | brl | 3.2 | 1.25 |
| Binarized | brs | 7.8 | 3.00 |
| Binarized | jrip | 2.2 | 1.87 |
| Binarized | part | 6.4 | 2.09 |
| Binarized | j48 | 6.6 | 3.34 |
| Original | tree | 13.4 | 3.80 |
| Original | forest | 121.0 | 3.73 |
| Original | ripper_A | 3.4 | 1.88 |
| Original | ripper_B | 3.8 | 1.95 |
| Original | irep_A | 2.0 | 1.53 |
| Original | irep_B | 1.2 | 1.10 |
| Original | brl | 3.2 | 1.25 |
| Original | brs | 7.8 | 3.00 |
| Original | jrip | 1.2 | 1.20 |
| Original | part | 5.2 | 1.42 |
| Original | j48 | 9.0 | 2.61 |

(113.8s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 71.31 +/- 2.98 | 0/5 |
| Binarized | forest | 76.21 +/- 2.92 | 0/5 |
| Binarized | ripper_A | 61.98 +/- 14.29 | 0/5 |
| Binarized | ripper_B | 76.21 +/- 2.92 | 0/5 |
| Binarized | irep_A | 74.46 +/- 1.92 | 0/5 |
| Binarized | irep_B | 70.28 +/- 1.93 | 0/5 |
| Binarized | brl | 73.09 +/- 2.26 | 0/5 |
| Binarized | brs | 71.69 +/- 3.66 | 0/5 |
| Binarized | jrip | 73.06 +/- 5.36 | 0/5 |
| Binarized | part | 66.44 +/- 4.94 | 0/5 |
| Binarized | j48 | 70.64 +/- 3.34 | 0/5 |
| Original | tree | 72.72 +/- 1.92 | 0/5 |
| Original | forest | 75.51 +/- 3.43 | 0/5 |
| Original | ripper_A | 66.79 +/- 2.79 | 0/5 |
| Original | ripper_B | 73.07 +/- 2.21 | 0/5 |
| Original | irep_A | 74.14 +/- 2.67 | 0/5 |
| Original | irep_B | 73.05 +/- 5.71 | 0/5 |
| Original | brl | 73.09 +/- 2.26 | 0/5 |
| Original | brs | 71.69 +/- 3.66 | 0/5 |
| Original | jrip | 72.02 +/- 3.56 | 0/5 |
| Original | part | 69.96 +/- 5.62 | 0/5 |
| Original | j48 | 73.42 +/- 1.45 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.042 |
| Binarized | ripper_A | 0.103 |
| Binarized | ripper_B | 0.123 |
| Binarized | irep_A | 0.058 |
| Binarized | irep_B | 0.057 |
| Binarized | brl | 0.243 |
| Binarized | brs | 8.319 |
| Binarized | jrip | 0.459 |
| Binarized | part | 1.363 |
| Binarized | j48 | 0.441 |
| Original | tree | 0.003 |
| Original | forest | 0.054 |
| Original | ripper_A | 0.057 |
| Original | ripper_B | 0.085 |
| Original | irep_A | 0.032 |
| Original | irep_B | 0.027 |
| Original | brl | 0.242 |
| Original | brs | 8.173 |
| Original | jrip | 1.240 |
| Original | part | 0.384 |
| Original | j48 | 1.249 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.4 | 3.77 |
| Binarized | forest | 128.0 | 3.79 |
| Binarized | ripper_A | 1.4 | 1.70 |
| Binarized | ripper_B | 1.4 | 2.50 |
| Binarized | irep_A | 2.6 | 1.17 |
| Binarized | irep_B | 1.0 | 1.07 |
| Binarized | brl | 3.2 | 1.42 |
| Binarized | brs | 7.0 | 2.91 |
| Binarized | jrip | 1.0 | 1.80 |
| Binarized | part | 21.6 | 4.48 |
| Binarized | j48 | 13.8 | 5.22 |
| Original | tree | 12.4 | 3.75 |
| Original | forest | 124.8 | 3.78 |
| Original | ripper_A | 3.0 | 1.55 |
| Original | ripper_B | 1.6 | 2.70 |
| Original | irep_A | 3.2 | 1.43 |
| Original | irep_B | 1.4 | 1.70 |
| Original | brl | 3.2 | 1.42 |
| Original | brs | 7.0 | 2.91 |
| Original | jrip | 2.8 | 2.22 |
| Original | part | 15.4 | 1.92 |
| Original | j48 | 8.4 | 1.77 |

(115.0s total)

---

## colic

n=368, attributes=26, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 73.11 +/- 8.00 | 0/5 |
| Binarized | forest | 80.15 +/- 4.07 | 0/5 |
| Binarized | ripper_A | 81.24 +/- 2.83 | 0/5 |
| Binarized | ripper_B | 85.35 +/- 3.72 | 0/5 |
| Binarized | irep_A | 85.33 +/- 1.53 | 0/5 |
| Binarized | irep_B | 84.51 +/- 2.20 | 0/5 |
| Binarized | brl | 84.79 +/- 1.53 | 0/5 |
| Binarized | brs | 80.98 +/- 6.18 | 0/5 |
| Binarized | jrip | 86.42 +/- 2.97 | 0/5 |
| Binarized | part | 81.81 +/- 4.54 | 0/5 |
| Binarized | j48 | 83.43 +/- 2.59 | 0/5 |
| Original | tree | 84.25 +/- 3.11 | 0/5 |
| Original | forest | 84.26 +/- 3.31 | 0/5 |
| Original | ripper_A | 81.27 +/- 3.39 | 0/5 |
| Original | ripper_B | 86.96 +/- 2.91 | 0/5 |
| Original | irep_A | 82.08 +/- 2.81 | 0/5 |
| Original | irep_B | 86.14 +/- 1.05 | 0/5 |
| Original | brl | 84.79 +/- 1.53 | 0/5 |
| Original | brs | 80.98 +/- 6.18 | 0/5 |
| Original | jrip | 87.51 +/- 3.34 | 0/5 |
| Original | part | 85.88 +/- 2.47 | 0/5 |
| Original | j48 | 82.63 +/- 5.17 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.010 |
| Binarized | forest | 0.052 |
| Binarized | ripper_A | 0.471 |
| Binarized | ripper_B | 0.472 |
| Binarized | irep_A | 0.329 |
| Binarized | irep_B | 0.296 |
| Binarized | brl | 2.658 |
| Binarized | brs | 34.378 |
| Binarized | jrip | 0.618 |
| Binarized | part | 1.491 |
| Binarized | j48 | 0.517 |
| Original | tree | 0.004 |
| Original | forest | 0.072 |
| Original | ripper_A | 0.245 |
| Original | ripper_B | 0.262 |
| Original | irep_A | 0.173 |
| Original | irep_B | 0.182 |
| Original | brl | 2.545 |
| Original | brs | 34.786 |
| Original | jrip | 0.424 |
| Original | part | 1.306 |
| Original | j48 | 0.418 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 11.8 | 3.72 |
| Binarized | forest | 122.0 | 3.74 |
| Binarized | ripper_A | 3.6 | 2.25 |
| Binarized | ripper_B | 3.8 | 1.83 |
| Binarized | irep_A | 2.2 | 1.33 |
| Binarized | irep_B | 1.4 | 1.60 |
| Binarized | brl | 4.8 | 1.55 |
| Binarized | brs | 7.4 | 2.83 |
| Binarized | jrip | 3.2 | 1.97 |
| Binarized | part | 11.6 | 5.10 |
| Binarized | j48 | 17.0 | 6.19 |
| Original | tree | 11.6 | 3.72 |
| Original | forest | 118.6 | 3.73 |
| Original | ripper_A | 6.0 | 1.61 |
| Original | ripper_B | 6.2 | 2.13 |
| Original | irep_A | 2.0 | 1.13 |
| Original | irep_B | 1.6 | 1.60 |
| Original | brl | 4.8 | 1.55 |
| Original | brs | 7.4 | 2.83 |
| Original | jrip | 3.4 | 1.83 |
| Original | part | 12.8 | 1.81 |
| Original | j48 | 8.6 | 2.17 |

(412.8s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 84.35 +/- 2.85 | 0/5 |
| Binarized | forest | 84.06 +/- 3.83 | 0/5 |
| Binarized | ripper_A | 86.09 +/- 3.82 | 0/5 |
| Binarized | ripper_B | 85.94 +/- 2.85 | 0/5 |
| Binarized | irep_A | 85.51 +/- 3.30 | 0/5 |
| Binarized | irep_B | 84.93 +/- 2.40 | 0/5 |
| Binarized | brl | 84.20 +/- 2.65 | 0/5 |
| Binarized | brs | 62.03 +/- 5.99 | 0/5 |
| Binarized | jrip | 85.65 +/- 2.48 | 0/5 |
| Binarized | part | 83.04 +/- 3.60 | 0/5 |
| Binarized | j48 | 84.64 +/- 3.41 | 0/5 |
| Original | tree | 84.78 +/- 3.89 | 0/5 |
| Original | forest | 85.51 +/- 3.37 | 0/5 |
| Original | ripper_A | 83.62 +/- 3.57 | 0/5 |
| Original | ripper_B | 83.04 +/- 3.63 | 0/5 |
| Original | irep_A | 85.51 +/- 3.30 | 0/5 |
| Original | irep_B | 84.78 +/- 3.27 | 0/5 |
| Original | brl | 84.20 +/- 2.65 | 0/5 |
| Original | brs | 62.03 +/- 5.99 | 0/5 |
| Original | jrip | 85.94 +/- 2.58 | 0/5 |
| Original | part | 84.35 +/- 3.51 | 0/5 |
| Original | j48 | 85.94 +/- 4.09 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.011 |
| Binarized | forest | 0.076 |
| Binarized | ripper_A | 0.338 |
| Binarized | ripper_B | 0.317 |
| Binarized | irep_A | 0.156 |
| Binarized | irep_B | 0.157 |
| Binarized | brl | 1.703 |
| Binarized | brs | 11.393 |
| Binarized | jrip | 0.632 |
| Binarized | part | 1.449 |
| Binarized | j48 | 0.458 |
| Original | tree | 0.004 |
| Original | forest | 0.105 |
| Original | ripper_A | 0.238 |
| Original | ripper_B | 0.248 |
| Original | irep_A | 0.086 |
| Original | irep_B | 0.086 |
| Original | brl | 1.640 |
| Original | brs | 11.417 |
| Original | jrip | 0.396 |
| Original | part | 1.311 |
| Original | j48 | 0.408 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.4 | 3.87 |
| Binarized | forest | 138.8 | 3.87 |
| Binarized | ripper_A | 5.0 | 2.53 |
| Binarized | ripper_B | 3.6 | 2.52 |
| Binarized | irep_A | 1.2 | 1.20 |
| Binarized | irep_B | 2.6 | 2.35 |
| Binarized | brl | 4.8 | 1.58 |
| Binarized | brs | 8.4 | 2.93 |
| Binarized | jrip | 3.2 | 2.05 |
| Binarized | part | 23.6 | 4.06 |
| Binarized | j48 | 23.0 | 7.23 |
| Original | tree | 14.2 | 3.87 |
| Original | forest | 134.0 | 3.84 |
| Original | ripper_A | 7.6 | 2.86 |
| Original | ripper_B | 8.8 | 3.22 |
| Original | irep_A | 1.6 | 1.10 |
| Original | irep_B | 2.4 | 1.85 |
| Original | brl | 4.8 | 1.58 |
| Original | brs | 8.4 | 2.93 |
| Original | jrip | 3.8 | 2.18 |
| Original | part | 29.6 | 2.37 |
| Original | j48 | 19.4 | 4.02 |

(166.3s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*5/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.70 +/- 0.98 | 0/5 |
| Binarized | forest | 73.20 +/- 1.08 | 0/5 |
| Binarized | ripper_A | 70.80 +/- 3.50 | 0/5 |
| Binarized | ripper_B | 58.50 +/- 6.73 | 0/5 |
| Binarized | irep_A | 70.20 +/- 1.75 | 0/5 |
| Binarized | irep_B | 69.50 +/- 1.41 | 0/5 |
| Binarized | brl | 69.80 +/- 2.23 | 0/5 |
| Binarized | brs | 44.20 +/- 2.44 | 0/5 |
| Binarized | jrip | 72.70 +/- 1.94 | 0/5 |
| Binarized | part | 72.30 +/- 3.23 | 0/5 |
| Binarized | j48 | 71.90 +/- 1.24 | 0/5 |
| Original | tree | 70.60 +/- 2.60 | 0/5 |
| Original | forest | 70.10 +/- 1.16 | 0/5 |
| Original | ripper_A | 70.20 +/- 1.03 | 0/5 |
| Original | ripper_B | 58.60 +/- 5.07 | 0/5 |
| Original | irep_A | 69.80 +/- 1.21 | 0/5 |
| Original | irep_B | 71.40 +/- 2.08 | 0/5 |
| Original | brl | 69.80 +/- 2.23 | 0/5 |
| Original | brs | 44.20 +/- 2.44 | 0/5 |
| Original | jrip | 72.40 +/- 2.40 | 0/5 |
| Original | part | 70.00 +/- 3.36 | 0/5 |
| Original | j48 | 70.80 +/- 1.91 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.099 |
| Binarized | ripper_A | 0.602 |
| Binarized | ripper_B | 0.438 |
| Binarized | irep_A | 0.194 |
| Binarized | irep_B | 0.188 |
| Binarized | brl | 2.930 |
| Binarized | brs | 12.396 |
| Binarized | jrip | 0.715 |
| Binarized | part | 1.648 |
| Binarized | j48 | 0.564 |
| Original | tree | 0.005 |
| Original | forest | 0.144 |
| Original | ripper_A | 0.296 |
| Original | ripper_B | 0.264 |
| Original | irep_A | 0.107 |
| Original | irep_B | 0.103 |
| Original | brl | 2.919 |
| Original | brs | 12.423 |
| Original | jrip | 0.452 |
| Original | part | 1.347 |
| Original | j48 | 0.437 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.0 | 3.93 |
| Binarized | forest | 145.0 | 3.90 |
| Binarized | ripper_A | 3.2 | 4.83 |
| Binarized | ripper_B | 3.8 | 2.95 |
| Binarized | irep_A | 2.2 | 2.70 |
| Binarized | irep_B | 4.6 | 1.54 |
| Binarized | brl | 4.8 | 1.67 |
| Binarized | brs | 8.2 | 2.95 |
| Binarized | jrip | 3.0 | 3.77 |
| Binarized | part | 56.8 | 4.09 |
| Binarized | j48 | 78.8 | 11.42 |
| Original | tree | 15.2 | 3.95 |
| Original | forest | 141.8 | 3.89 |
| Original | ripper_A | 5.4 | 3.64 |
| Original | ripper_B | 5.2 | 2.68 |
| Original | irep_A | 3.2 | 2.96 |
| Original | irep_B | 5.6 | 1.59 |
| Original | brl | 4.8 | 1.67 |
| Original | brs | 8.2 | 2.95 |
| Original | jrip | 3.4 | 2.69 |
| Original | part | 60.4 | 3.02 |
| Original | j48 | 77.8 | 5.57 |

(195.2s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 61.60 +/- 8.55 | 0/5 |
| Binarized | forest | 74.48 +/- 2.17 | 0/5 |
| Binarized | ripper_A | 66.53 +/- 2.41 | 0/5 |
| Binarized | ripper_B | 73.96 +/- 1.87 | 0/5 |
| Binarized | irep_A | 75.00 +/- 1.05 | 0/5 |
| Binarized | irep_B | 74.09 +/- 2.35 | 0/5 |
| Binarized | brl | 73.70 +/- 1.26 | 0/5 |
| Binarized | brs | 68.62 +/- 0.79 | 0/5 |
| Binarized | jrip | 74.61 +/- 2.13 | 0/5 |
| Binarized | part | 74.22 +/- 1.56 | 0/5 |
| Binarized | j48 | 74.99 +/- 2.62 | 0/5 |
| Original | tree | 73.83 +/- 3.41 | 0/5 |
| Original | forest | 74.09 +/- 2.92 | 0/5 |
| Original | ripper_A | 59.62 +/- 4.74 | 0/5 |
| Original | ripper_B | 70.98 +/- 4.92 | 0/5 |
| Original | irep_A | 67.45 +/- 4.68 | 0/5 |
| Original | irep_B | 72.66 +/- 3.17 | 0/5 |
| Original | brl | 73.70 +/- 1.26 | 0/5 |
| Original | brs | 68.62 +/- 0.79 | 0/5 |
| Original | jrip | 75.14 +/- 5.03 | 0/5 |
| Original | part | 71.75 +/- 2.68 | 0/5 |
| Original | j48 | 71.74 +/- 3.49 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.011 |
| Binarized | forest | 0.085 |
| Binarized | ripper_A | 0.322 |
| Binarized | ripper_B | 0.342 |
| Binarized | irep_A | 0.116 |
| Binarized | irep_B | 0.125 |
| Binarized | brl | 1.164 |
| Binarized | brs | 9.731 |
| Binarized | jrip | 0.623 |
| Binarized | part | 1.428 |
| Binarized | j48 | 0.520 |
| Original | tree | 0.004 |
| Original | forest | 0.149 |
| Original | ripper_A | 0.216 |
| Original | ripper_B | 0.221 |
| Original | irep_A | 0.079 |
| Original | irep_B | 0.067 |
| Original | brl | 1.140 |
| Original | brs | 9.842 |
| Original | jrip | 0.431 |
| Original | part | 1.284 |
| Original | j48 | 0.394 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.6 | 3.90 |
| Binarized | forest | 145.4 | 3.91 |
| Binarized | ripper_A | 4.4 | 2.71 |
| Binarized | ripper_B | 3.8 | 3.74 |
| Binarized | irep_A | 3.4 | 1.71 |
| Binarized | irep_B | 3.2 | 3.14 |
| Binarized | brl | 4.6 | 1.79 |
| Binarized | brs | 7.0 | 2.94 |
| Binarized | jrip | 3.0 | 2.68 |
| Binarized | part | 46.2 | 4.08 |
| Binarized | j48 | 42.2 | 8.96 |
| Original | tree | 14.6 | 3.90 |
| Original | forest | 140.4 | 3.88 |
| Original | ripper_A | 8.2 | 2.46 |
| Original | ripper_B | 8.2 | 2.93 |
| Original | irep_A | 7.6 | 1.69 |
| Original | irep_B | 2.0 | 1.43 |
| Original | brl | 4.6 | 1.79 |
| Original | brs | 7.0 | 2.94 |
| Original | jrip | 2.4 | 2.40 |
| Original | part | 6.8 | 2.51 |
| Original | j48 | 22.0 | 6.13 |

(144.5s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

*20 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold (n/a in the tables below). `brl`/`brs` are expected to hit this on datasets with many discretized features -- see `FIT_TIMEOUT_SECONDS`'s comment; it is a scaling limit of those algorithms, not a bug.*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 70.64 +/- 10.06 | 0/5 |
| Binarized | forest | 77.90 +/- 6.61 | 0/5 |
| Binarized | ripper_A | 70.24 +/- 4.63 | 0/5 |
| Binarized | ripper_B | 75.52 +/- 7.20 | 0/5 |
| Binarized | irep_A | 69.76 +/- 5.15 | 0/5 |
| Binarized | irep_B | 71.65 +/- 3.75 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 70.19 +/- 6.53 | 0/5 |
| Binarized | part | 74.53 +/- 5.71 | 0/5 |
| Binarized | j48 | 74.47 +/- 5.62 | 0/5 |
| Original | tree | 75.48 +/- 4.09 | 0/5 |
| Original | forest | 78.37 +/- 4.50 | 0/5 |
| Original | ripper_A | 62.11 +/- 11.07 | 0/5 |
| Original | ripper_B | 64.44 +/- 6.02 | 0/5 |
| Original | irep_A | 58.21 +/- 5.69 | 0/5 |
| Original | irep_B | 62.96 +/- 6.45 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | n/a | 5/5 |
| Original | jrip | 72.10 +/- 3.95 | 0/5 |
| Original | part | 75.96 +/- 5.32 | 0/5 |
| Original | j48 | 71.56 +/- 7.81 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.015 |
| Binarized | forest | 0.045 |
| Binarized | ripper_A | 1.012 |
| Binarized | ripper_B | 1.011 |
| Binarized | irep_A | 0.822 |
| Binarized | irep_B | 0.813 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.643 |
| Binarized | part | 1.431 |
| Binarized | j48 | 0.510 |
| Original | tree | 0.013 |
| Original | forest | 0.147 |
| Original | ripper_A | 0.953 |
| Original | ripper_B | 0.991 |
| Original | irep_A | 0.835 |
| Original | irep_B | 0.843 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.449 |
| Original | part | 1.328 |
| Original | j48 | 0.432 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.6 | 3.82 |
| Binarized | forest | 121.4 | 3.73 |
| Binarized | ripper_A | 3.0 | 2.25 |
| Binarized | ripper_B | 3.2 | 2.12 |
| Binarized | irep_A | 1.8 | 1.87 |
| Binarized | irep_B | 3.6 | 2.15 |
| Binarized | brl | nan | nan |
| Binarized | brs | nan | nan |
| Binarized | jrip | 3.4 | 1.74 |
| Binarized | part | 6.2 | 3.59 |
| Binarized | j48 | 16.0 | 5.73 |
| Original | tree | 13.0 | 3.76 |
| Original | forest | 114.8 | 3.67 |
| Original | ripper_A | 4.4 | 1.66 |
| Original | ripper_B | 3.4 | 2.59 |
| Original | irep_A | 2.6 | 1.08 |
| Original | irep_B | 2.6 | 1.78 |
| Original | brl | nan | nan |
| Original | brs | nan | nan |
| Original | jrip | 3.6 | 1.73 |
| Original | part | 6.4 | 2.41 |
| Original | j48 | 14.0 | 4.51 |

(1275.4s total)

---

## ionosphere

n=351, attributes=33, A='b', B='g'

*4/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

*20 model-fold combination(s) timed out (> 60s) or raised and were skipped for that fold (n/a in the tables below). `brl`/`brs` are expected to hit this on datasets with many discretized features -- see `FIT_TIMEOUT_SECONDS`'s comment; it is a scaling limit of those algorithms, not a bug.*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 90.03 +/- 2.71 | 0/5 |
| Binarized | forest | 91.74 +/- 2.75 | 0/5 |
| Binarized | ripper_A | 89.75 +/- 2.44 | 0/5 |
| Binarized | ripper_B | 87.48 +/- 4.11 | 0/5 |
| Binarized | irep_A | 89.46 +/- 4.64 | 0/5 |
| Binarized | irep_B | 90.03 +/- 4.52 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 89.18 +/- 4.10 | 0/5 |
| Binarized | part | 91.74 +/- 2.75 | 0/5 |
| Binarized | j48 | 90.31 +/- 3.05 | 0/5 |
| Original | tree | 88.61 +/- 3.24 | 0/5 |
| Original | forest | 91.45 +/- 3.61 | 0/5 |
| Original | ripper_A | 88.04 +/- 4.18 | 0/5 |
| Original | ripper_B | 75.51 +/- 2.66 | 0/5 |
| Original | irep_A | 85.17 +/- 4.60 | 0/5 |
| Original | irep_B | 74.93 +/- 2.64 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | n/a | 5/5 |
| Original | jrip | 90.89 +/- 2.31 | 0/5 |
| Original | part | 90.62 +/- 4.60 | 0/5 |
| Original | j48 | 88.33 +/- 1.33 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.012 |
| Binarized | forest | 0.051 |
| Binarized | ripper_A | 0.668 |
| Binarized | ripper_B | 0.653 |
| Binarized | irep_A | 0.484 |
| Binarized | irep_B | 0.499 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.668 |
| Binarized | part | 1.406 |
| Binarized | j48 | 0.788 |
| Original | tree | 0.012 |
| Original | forest | 0.166 |
| Original | ripper_A | 0.490 |
| Original | ripper_B | 0.544 |
| Original | irep_A | 0.392 |
| Original | irep_B | 0.382 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.419 |
| Original | part | 1.444 |
| Original | j48 | 0.588 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 8.6 | 3.46 |
| Binarized | forest | 98.6 | 3.57 |
| Binarized | ripper_A | 5.6 | 1.55 |
| Binarized | ripper_B | 3.0 | 3.20 |
| Binarized | irep_A | 2.6 | 1.23 |
| Binarized | irep_B | 1.0 | 2.00 |
| Binarized | brl | nan | nan |
| Binarized | brs | nan | nan |
| Binarized | jrip | 4.2 | 1.47 |
| Binarized | part | 6.0 | 3.09 |
| Binarized | j48 | 10.8 | 4.64 |
| Original | tree | 8.8 | 3.49 |
| Original | forest | 95.4 | 3.51 |
| Original | ripper_A | 8.0 | 2.07 |
| Original | ripper_B | 9.0 | 2.47 |
| Original | irep_A | 2.2 | 1.20 |
| Original | irep_B | 1.0 | 1.00 |
| Original | brl | nan | nan |
| Original | brs | nan | nan |
| Original | jrip | 4.2 | 1.31 |
| Original | part | 5.6 | 2.83 |
| Original | j48 | 11.4 | 5.04 |

(1262.4s total)

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
| Binarized | irep_A | 93.02 +/- 6.67 | 0/5 |
| Binarized | irep_B | 82.04 +/- 4.33 | 0/5 |
| Binarized | brl | 80.58 +/- 3.08 | 0/5 |
| Binarized | brs | 86.73 +/- 6.90 | 0/5 |
| Binarized | jrip | 98.33 +/- 1.01 | 0/5 |
| Binarized | part | 93.84 +/- 1.85 | 0/5 |
| Binarized | j48 | 94.88 +/- 0.77 | 0/5 |
| Original | tree | 82.36 +/- 2.06 | 0/5 |
| Original | forest | 76.83 +/- 2.17 | 0/5 |
| Original | ripper_A | 97.91 +/- 0.93 | 0/5 |
| Original | ripper_B | 97.39 +/- 2.26 | 0/5 |
| Original | irep_A | 91.12 +/- 7.30 | 0/5 |
| Original | irep_B | 83.41 +/- 3.26 | 0/5 |
| Original | brl | 80.58 +/- 3.08 | 0/5 |
| Original | brs | 86.73 +/- 6.90 | 0/5 |
| Original | jrip | 97.91 +/- 0.87 | 0/5 |
| Original | part | 94.15 +/- 2.35 | 0/5 |
| Original | j48 | 87.27 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.011 |
| Binarized | forest | 0.097 |
| Binarized | ripper_A | 0.174 |
| Binarized | ripper_B | 0.192 |
| Binarized | irep_A | 0.076 |
| Binarized | irep_B | 0.067 |
| Binarized | brl | 0.889 |
| Binarized | brs | 7.496 |
| Binarized | jrip | 0.568 |
| Binarized | part | 1.364 |
| Binarized | j48 | 0.443 |
| Original | tree | 0.003 |
| Original | forest | 0.111 |
| Original | ripper_A | 0.113 |
| Original | ripper_B | 0.147 |
| Original | irep_A | 0.041 |
| Original | irep_B | 0.035 |
| Original | brl | 0.787 |
| Original | brs | 7.497 |
| Original | jrip | 0.439 |
| Original | part | 1.289 |
| Original | j48 | 0.375 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.0 | 3.86 |
| Binarized | forest | 153.4 | 3.96 |
| Binarized | ripper_A | 8.6 | 3.07 |
| Binarized | ripper_B | 9.8 | 3.22 |
| Binarized | irep_A | 7.2 | 2.86 |
| Binarized | irep_B | 6.0 | 2.74 |
| Binarized | brl | 11.8 | 1.81 |
| Binarized | brs | 7.4 | 3.00 |
| Binarized | jrip | 8.0 | 3.08 |
| Binarized | part | 29.2 | 3.23 |
| Binarized | j48 | 37.8 | 5.88 |
| Original | tree | 14.0 | 3.86 |
| Original | forest | 157.4 | 3.98 |
| Original | ripper_A | 9.4 | 3.28 |
| Original | ripper_B | 14.0 | 3.13 |
| Original | irep_A | 7.2 | 2.84 |
| Original | irep_B | 6.4 | 2.29 |
| Original | brl | 11.8 | 1.81 |
| Original | brs | 7.4 | 3.00 |
| Original | jrip | 9.6 | 3.28 |
| Original | part | 36.4 | 2.69 |
| Original | j48 | 80.6 | 4.53 |

(113.0s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 95.77 +/- 0.85 | 0/5 |
| Binarized | forest | 96.35 +/- 0.84 | 0/5 |
| Binarized | ripper_A | 98.54 +/- 0.61 | 0/5 |
| Binarized | ripper_B | 98.18 +/- 0.65 | 0/5 |
| Binarized | irep_A | 94.46 +/- 1.52 | 0/5 |
| Binarized | irep_B | 97.16 +/- 1.07 | 0/5 |
| Binarized | brl | 97.16 +/- 1.34 | 0/5 |
| Binarized | brs | 95.41 +/- 2.09 | 0/5 |
| Binarized | jrip | 98.61 +/- 0.54 | 0/5 |
| Binarized | part | 98.32 +/- 0.68 | 0/5 |
| Binarized | j48 | 98.54 +/- 0.65 | 0/5 |
| Original | tree | 95.70 +/- 1.63 | 0/5 |
| Original | forest | 97.01 +/- 1.27 | 0/5 |
| Original | ripper_A | 97.09 +/- 0.86 | 0/5 |
| Original | ripper_B | 95.63 +/- 2.12 | 0/5 |
| Original | irep_A | 92.35 +/- 1.25 | 0/5 |
| Original | irep_B | 90.01 +/- 2.40 | 0/5 |
| Original | brl | 97.16 +/- 1.34 | 0/5 |
| Original | brs | 95.41 +/- 2.09 | 0/5 |
| Original | jrip | 97.89 +/- 0.63 | 0/5 |
| Original | part | 98.76 +/- 0.75 | 0/5 |
| Original | j48 | 98.61 +/- 1.25 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.014 |
| Binarized | forest | 0.112 |
| Binarized | ripper_A | 0.155 |
| Binarized | ripper_B | 0.170 |
| Binarized | irep_A | 0.072 |
| Binarized | irep_B | 0.078 |
| Binarized | brl | 0.865 |
| Binarized | brs | 7.876 |
| Binarized | jrip | 0.614 |
| Binarized | part | 1.362 |
| Binarized | j48 | 0.463 |
| Original | tree | 0.004 |
| Original | forest | 0.205 |
| Original | ripper_A | 0.381 |
| Original | ripper_B | 0.412 |
| Original | irep_A | 0.054 |
| Original | irep_B | 0.055 |
| Original | brl | 0.857 |
| Original | brs | 7.929 |
| Original | jrip | 0.447 |
| Original | part | 1.292 |
| Original | j48 | 0.427 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.0 | 3.77 |
| Binarized | forest | 131.4 | 3.81 |
| Binarized | ripper_A | 6.6 | 2.33 |
| Binarized | ripper_B | 6.2 | 2.49 |
| Binarized | irep_A | 3.6 | 1.71 |
| Binarized | irep_B | 4.0 | 2.13 |
| Binarized | brl | 5.0 | 1.89 |
| Binarized | brs | 9.6 | 2.96 |
| Binarized | jrip | 5.2 | 2.30 |
| Binarized | part | 9.4 | 2.10 |
| Binarized | j48 | 11.4 | 3.88 |
| Original | tree | 12.2 | 3.69 |
| Original | forest | 123.4 | 3.74 |
| Original | ripper_A | 33.6 | 3.65 |
| Original | ripper_B | 31.2 | 3.85 |
| Original | irep_A | 11.0 | 2.69 |
| Original | irep_B | 10.4 | 3.05 |
| Original | brl | 5.0 | 1.89 |
| Original | brs | 9.6 | 2.96 |
| Original | jrip | 6.0 | 2.24 |
| Original | part | 7.4 | 2.11 |
| Original | j48 | 15.2 | 4.51 |

(122.5s total)

---

## kr-vs-kp

n=3196, attributes=36, A='nowin', B='won'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 94.09 +/- 1.11 | 0/5 |
| Binarized | forest | 94.99 +/- 0.90 | 0/5 |
| Binarized | ripper_A | 98.94 +/- 0.53 | 0/5 |
| Binarized | ripper_B | 98.75 +/- 0.41 | 0/5 |
| Binarized | irep_A | 98.22 +/- 0.79 | 0/5 |
| Binarized | irep_B | 93.77 +/- 2.92 | 0/5 |
| Binarized | brl | 94.09 +/- 2.48 | 0/5 |
| Binarized | brs | 81.63 +/- 5.22 | 0/5 |
| Binarized | jrip | 99.22 +/- 0.33 | 0/5 |
| Binarized | part | 98.94 +/- 0.30 | 0/5 |
| Binarized | j48 | 99.28 +/- 0.38 | 0/5 |
| Original | tree | 94.09 +/- 1.11 | 0/5 |
| Original | forest | 94.12 +/- 1.11 | 0/5 |
| Original | ripper_A | 98.65 +/- 0.65 | 0/5 |
| Original | ripper_B | 98.84 +/- 0.34 | 0/5 |
| Original | irep_A | 97.40 +/- 1.23 | 0/5 |
| Original | irep_B | 94.78 +/- 3.23 | 0/5 |
| Original | brl | 94.09 +/- 2.48 | 0/5 |
| Original | brs | 81.63 +/- 5.22 | 0/5 |
| Original | jrip | 98.81 +/- 0.70 | 0/5 |
| Original | part | 98.97 +/- 0.47 | 0/5 |
| Original | j48 | 99.34 +/- 0.38 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.020 |
| Binarized | forest | 0.210 |
| Binarized | ripper_A | 0.712 |
| Binarized | ripper_B | 0.581 |
| Binarized | irep_A | 0.161 |
| Binarized | irep_B | 0.149 |
| Binarized | brl | 10.942 |
| Binarized | brs | 13.437 |
| Binarized | jrip | 0.822 |
| Binarized | part | 1.561 |
| Binarized | j48 | 0.541 |
| Original | tree | 0.008 |
| Original | forest | 0.282 |
| Original | ripper_A | 0.572 |
| Original | ripper_B | 0.483 |
| Original | irep_A | 0.159 |
| Original | irep_B | 0.174 |
| Original | brl | 10.907 |
| Original | brs | 13.352 |
| Original | jrip | 0.593 |
| Original | part | 1.337 |
| Original | j48 | 0.470 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 7.6 | 3.43 |
| Binarized | forest | 112.6 | 3.73 |
| Binarized | ripper_A | 17.4 | 3.23 |
| Binarized | ripper_B | 9.6 | 4.54 |
| Binarized | irep_A | 7.0 | 2.69 |
| Binarized | irep_B | 5.2 | 4.33 |
| Binarized | brl | 8.4 | 1.87 |
| Binarized | brs | 7.0 | 2.86 |
| Binarized | jrip | 13.6 | 3.00 |
| Binarized | part | 20.8 | 2.93 |
| Binarized | j48 | 25.8 | 7.37 |
| Original | tree | 7.6 | 3.43 |
| Original | forest | 118.6 | 3.75 |
| Original | ripper_A | 18.4 | 3.33 |
| Original | ripper_B | 10.8 | 4.80 |
| Original | irep_A | 6.2 | 3.00 |
| Original | irep_B | 5.8 | 3.92 |
| Original | brl | 8.4 | 1.87 |
| Original | brs | 7.0 | 2.86 |
| Original | jrip | 14.6 | 3.16 |
| Original | part | 20.4 | 3.21 |
| Original | j48 | 29.0 | 7.71 |

(294.1s total)

---

## mushroom

n=8124, attributes=21, A='e', B='p'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 78.08 +/- 0.73 | 0/5 |
| Binarized | forest | 98.98 +/- 0.19 | 0/5 |
| Binarized | ripper_A | 98.18 +/- 0.55 | 0/5 |
| Binarized | ripper_B | 99.95 +/- 0.10 | 0/5 |
| Binarized | irep_A | 98.52 +/- 0.24 | 0/5 |
| Binarized | irep_B | 99.96 +/- 0.05 | 0/5 |
| Binarized | brl | 99.47 +/- 0.49 | 0/5 |
| Binarized | brs | 99.62 +/- 0.32 | 0/5 |
| Binarized | jrip | 99.98 +/- 0.05 | 0/5 |
| Binarized | part | 100.00 +/- 0.00 | 0/5 |
| Binarized | j48 | 100.00 +/- 0.00 | 0/5 |
| Original | tree | 99.22 +/- 0.14 | 0/5 |
| Original | forest | 98.66 +/- 0.38 | 0/5 |
| Original | ripper_A | 100.00 +/- 0.00 | 0/5 |
| Original | ripper_B | 99.98 +/- 0.05 | 0/5 |
| Original | irep_A | 98.52 +/- 0.24 | 0/5 |
| Original | irep_B | 96.45 +/- 0.32 | 0/5 |
| Original | brl | 99.47 +/- 0.49 | 0/5 |
| Original | brs | 99.62 +/- 0.32 | 0/5 |
| Original | jrip | 100.00 +/- 0.00 | 0/5 |
| Original | part | 100.00 +/- 0.00 | 0/5 |
| Original | j48 | 100.00 +/- 0.00 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.071 |
| Binarized | forest | 0.392 |
| Binarized | ripper_A | 1.783 |
| Binarized | ripper_B | 1.695 |
| Binarized | irep_A | 0.881 |
| Binarized | irep_B | 0.993 |
| Binarized | brl | 18.136 |
| Binarized | brs | 30.655 |
| Binarized | jrip | 1.420 |
| Binarized | part | 1.681 |
| Binarized | j48 | 1.087 |
| Original | tree | 0.023 |
| Original | forest | 0.827 |
| Original | ripper_A | 0.512 |
| Original | ripper_B | 0.401 |
| Original | irep_A | 0.227 |
| Original | irep_B | 0.207 |
| Original | brl | 18.229 |
| Original | brs | 30.155 |
| Original | jrip | 0.579 |
| Original | part | 1.373 |
| Original | j48 | 0.562 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 10.4 | 3.55 |
| Binarized | forest | 86.8 | 3.39 |
| Binarized | ripper_A | 7.0 | 2.02 |
| Binarized | ripper_B | 7.0 | 1.56 |
| Binarized | irep_A | 3.4 | 1.80 |
| Binarized | irep_B | 5.6 | 1.85 |
| Binarized | brl | 10.2 | 1.74 |
| Binarized | brs | 9.4 | 2.78 |
| Binarized | jrip | 5.8 | 1.64 |
| Binarized | part | 5.0 | 2.76 |
| Binarized | j48 | 9.8 | 3.83 |
| Original | tree | 10.0 | 3.50 |
| Original | forest | 101.8 | 3.58 |
| Original | ripper_A | 7.6 | 2.54 |
| Original | ripper_B | 8.2 | 1.50 |
| Original | irep_A | 4.0 | 1.50 |
| Original | irep_B | 4.0 | 1.00 |
| Original | brl | 10.2 | 1.74 |
| Original | brs | 9.4 | 2.78 |
| Original | jrip | 8.0 | 1.56 |
| Original | part | 10.0 | 1.85 |
| Original | j48 | 24.0 | 2.54 |

(588.2s total)

---

## Overall summary

### Accuracy

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 | fallback folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 89.43 | 95.63 | 93.56 | 95.63 | 94.25 | 94.71 | 94.48 | 89.89 | 95.40 | 95.17 | 96.09 | 94.94 | 94.71 | 95.17 | 95.17 | 94.25 | 95.63 | 94.48 | 89.89 | 95.63 | 95.63 | 96.32 | 0/5 |
| breast-cancer | 71.31 | 76.21 | 61.98 | 76.21 | 74.46 | 70.28 | 73.09 | 71.69 | 73.06 | 66.44 | 70.64 | 72.72 | 75.51 | 66.79 | 73.07 | 74.14 | 73.05 | 73.09 | 71.69 | 72.02 | 69.96 | 73.42 | 0/5 |
| colic | 73.11 | 80.15 | 81.24 | 85.35 | 85.33 | 84.51 | 84.79 | 80.98 | 86.42 | 81.81 | 83.43 | 84.25 | 84.26 | 81.27 | 86.96 | 82.08 | 86.14 | 84.79 | 80.98 | 87.51 | 85.88 | 82.63 | 0/5 |
| credit-approval | 84.35 | 84.06 | 86.09 | 85.94 | 85.51 | 84.93 | 84.20 | 62.03 | 85.65 | 83.04 | 84.64 | 84.78 | 85.51 | 83.62 | 83.04 | 85.51 | 84.78 | 84.20 | 62.03 | 85.94 | 84.35 | 85.94 | 0/5 |
| credit-g | 70.70 | 73.20 | 70.80 | 58.50 | 70.20 | 69.50 | 69.80 | 44.20 | 72.70 | 72.30 | 71.90 | 70.60 | 70.10 | 70.20 | 58.60 | 69.80 | 71.40 | 69.80 | 44.20 | 72.40 | 70.00 | 70.80 | 5/5 |
| diabetes | 61.60 | 74.48 | 66.53 | 73.96 | 75.00 | 74.09 | 73.70 | 68.62 | 74.61 | 74.22 | 74.99 | 73.83 | 74.09 | 59.62 | 70.98 | 67.45 | 72.66 | 73.70 | 68.62 | 75.14 | 71.75 | 71.74 | 0/5 |
| sonar | 70.64 | 77.90 | 70.24 | 75.52 | 69.76 | 71.65 | nan | nan | 70.19 | 74.53 | 74.47 | 75.48 | 78.37 | 62.11 | 64.44 | 58.21 | 62.96 | nan | nan | 72.10 | 75.96 | 71.56 | 0/5 |
| ionosphere | 90.03 | 91.74 | 89.75 | 87.48 | 89.46 | 90.03 | nan | nan | 89.18 | 91.74 | 90.31 | 88.61 | 91.45 | 88.04 | 75.51 | 85.17 | 74.93 | nan | nan | 90.89 | 90.62 | 88.33 | 4/5 |
| tic-tac-toe | 82.36 | 77.04 | 98.02 | 99.48 | 93.02 | 82.04 | 80.58 | 86.73 | 98.33 | 93.84 | 94.88 | 82.36 | 76.83 | 97.91 | 97.39 | 91.12 | 83.41 | 80.58 | 86.73 | 97.91 | 94.15 | 87.27 | 0/5 |
| banknote-authentication | 95.77 | 96.35 | 98.54 | 98.18 | 94.46 | 97.16 | 97.16 | 95.41 | 98.61 | 98.32 | 98.54 | 95.70 | 97.01 | 97.09 | 95.63 | 92.35 | 90.01 | 97.16 | 95.41 | 97.89 | 98.76 | 98.61 | 0/5 |
| kr-vs-kp | 94.09 | 94.99 | 98.94 | 98.75 | 98.22 | 93.77 | 94.09 | 81.63 | 99.22 | 98.94 | 99.28 | 94.09 | 94.12 | 98.65 | 98.84 | 97.40 | 94.78 | 94.09 | 81.63 | 98.81 | 98.97 | 99.34 | 0/5 |
| mushroom | 78.08 | 98.98 | 98.18 | 99.95 | 98.52 | 99.96 | 99.47 | 99.62 | 99.98 | 100.00 | 100.00 | 99.22 | 98.66 | 100.00 | 99.98 | 98.52 | 96.45 | 99.47 | 99.62 | 100.00 | 100.00 | 100.00 | 0/5 |

### Fit time (seconds/fold)

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.251 | 0.080 | 0.061 | 0.054 | 0.032 | 0.031 | 0.593 | 8.250 | 0.422 | 1.283 | 0.409 | 0.004 | 0.064 | 0.069 | 0.060 | 0.041 | 0.038 | 0.525 | 8.032 | 0.580 | 1.087 | 0.557 |
| breast-cancer | 0.006 | 0.042 | 0.103 | 0.123 | 0.058 | 0.057 | 0.243 | 8.319 | 0.459 | 1.363 | 0.441 | 0.003 | 0.054 | 0.057 | 0.085 | 0.032 | 0.027 | 0.242 | 8.173 | 1.240 | 0.384 | 1.249 |
| colic | 0.010 | 0.052 | 0.471 | 0.472 | 0.329 | 0.296 | 2.658 | 34.378 | 0.618 | 1.491 | 0.517 | 0.004 | 0.072 | 0.245 | 0.262 | 0.173 | 0.182 | 2.545 | 34.786 | 0.424 | 1.306 | 0.418 |
| credit-approval | 0.011 | 0.076 | 0.338 | 0.317 | 0.156 | 0.157 | 1.703 | 11.393 | 0.632 | 1.449 | 0.458 | 0.004 | 0.105 | 0.238 | 0.248 | 0.086 | 0.086 | 1.640 | 11.417 | 0.396 | 1.311 | 0.408 |
| credit-g | 0.015 | 0.099 | 0.602 | 0.438 | 0.194 | 0.188 | 2.930 | 12.396 | 0.715 | 1.648 | 0.564 | 0.005 | 0.144 | 0.296 | 0.264 | 0.107 | 0.103 | 2.919 | 12.423 | 0.452 | 1.347 | 0.437 |
| diabetes | 0.011 | 0.085 | 0.322 | 0.342 | 0.116 | 0.125 | 1.164 | 9.731 | 0.623 | 1.428 | 0.520 | 0.004 | 0.149 | 0.216 | 0.221 | 0.079 | 0.067 | 1.140 | 9.842 | 0.431 | 1.284 | 0.394 |
| sonar | 0.015 | 0.045 | 1.012 | 1.011 | 0.822 | 0.813 | nan | nan | 0.643 | 1.431 | 0.510 | 0.013 | 0.147 | 0.953 | 0.991 | 0.835 | 0.843 | nan | nan | 0.449 | 1.328 | 0.432 |
| ionosphere | 0.012 | 0.051 | 0.668 | 0.653 | 0.484 | 0.499 | nan | nan | 0.668 | 1.406 | 0.788 | 0.012 | 0.166 | 0.490 | 0.544 | 0.392 | 0.382 | nan | nan | 0.419 | 1.444 | 0.588 |
| tic-tac-toe | 0.011 | 0.097 | 0.174 | 0.192 | 0.076 | 0.067 | 0.889 | 7.496 | 0.568 | 1.364 | 0.443 | 0.003 | 0.111 | 0.113 | 0.147 | 0.041 | 0.035 | 0.787 | 7.497 | 0.439 | 1.289 | 0.375 |
| banknote-authentication | 0.014 | 0.112 | 0.155 | 0.170 | 0.072 | 0.078 | 0.865 | 7.876 | 0.614 | 1.362 | 0.463 | 0.004 | 0.205 | 0.381 | 0.412 | 0.054 | 0.055 | 0.857 | 7.929 | 0.447 | 1.292 | 0.427 |
| kr-vs-kp | 0.020 | 0.210 | 0.712 | 0.581 | 0.161 | 0.149 | 10.942 | 13.437 | 0.822 | 1.561 | 0.541 | 0.008 | 0.282 | 0.572 | 0.483 | 0.159 | 0.174 | 10.907 | 13.352 | 0.593 | 1.337 | 0.470 |
| mushroom | 0.071 | 0.392 | 1.783 | 1.695 | 0.881 | 0.993 | 18.136 | 30.655 | 1.420 | 1.681 | 1.087 | 0.023 | 0.827 | 0.512 | 0.401 | 0.227 | 0.207 | 18.229 | 30.155 | 0.579 | 1.373 | 0.562 |

### Rule count

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 12.8 | 126.4 | 2.4 | 3.0 | 2.0 | 1.6 | 3.2 | 7.8 | 2.2 | 6.4 | 6.6 | 13.4 | 121.0 | 3.4 | 3.8 | 2.0 | 1.2 | 3.2 | 7.8 | 1.2 | 5.2 | 9.0 |
| breast-cancer | 12.4 | 128.0 | 1.4 | 1.4 | 2.6 | 1.0 | 3.2 | 7.0 | 1.0 | 21.6 | 13.8 | 12.4 | 124.8 | 3.0 | 1.6 | 3.2 | 1.4 | 3.2 | 7.0 | 2.8 | 15.4 | 8.4 |
| colic | 11.8 | 122.0 | 3.6 | 3.8 | 2.2 | 1.4 | 4.8 | 7.4 | 3.2 | 11.6 | 17.0 | 11.6 | 118.6 | 6.0 | 6.2 | 2.0 | 1.6 | 4.8 | 7.4 | 3.4 | 12.8 | 8.6 |
| credit-approval | 14.4 | 138.8 | 5.0 | 3.6 | 1.2 | 2.6 | 4.8 | 8.4 | 3.2 | 23.6 | 23.0 | 14.2 | 134.0 | 7.6 | 8.8 | 1.6 | 2.4 | 4.8 | 8.4 | 3.8 | 29.6 | 19.4 |
| credit-g | 15.0 | 145.0 | 3.2 | 3.8 | 2.2 | 4.6 | 4.8 | 8.2 | 3.0 | 56.8 | 78.8 | 15.2 | 141.8 | 5.4 | 5.2 | 3.2 | 5.6 | 4.8 | 8.2 | 3.4 | 60.4 | 77.8 |
| diabetes | 14.6 | 145.4 | 4.4 | 3.8 | 3.4 | 3.2 | 4.6 | 7.0 | 3.0 | 46.2 | 42.2 | 14.6 | 140.4 | 8.2 | 8.2 | 7.6 | 2.0 | 4.6 | 7.0 | 2.4 | 6.8 | 22.0 |
| sonar | 13.6 | 121.4 | 3.0 | 3.2 | 1.8 | 3.6 | nan | nan | 3.4 | 6.2 | 16.0 | 13.0 | 114.8 | 4.4 | 3.4 | 2.6 | 2.6 | nan | nan | 3.6 | 6.4 | 14.0 |
| ionosphere | 8.6 | 98.6 | 5.6 | 3.0 | 2.6 | 1.0 | nan | nan | 4.2 | 6.0 | 10.8 | 8.8 | 95.4 | 8.0 | 9.0 | 2.2 | 1.0 | nan | nan | 4.2 | 5.6 | 11.4 |
| tic-tac-toe | 14.0 | 153.4 | 8.6 | 9.8 | 7.2 | 6.0 | 11.8 | 7.4 | 8.0 | 29.2 | 37.8 | 14.0 | 157.4 | 9.4 | 14.0 | 7.2 | 6.4 | 11.8 | 7.4 | 9.6 | 36.4 | 80.6 |
| banknote-authentication | 13.0 | 131.4 | 6.6 | 6.2 | 3.6 | 4.0 | 5.0 | 9.6 | 5.2 | 9.4 | 11.4 | 12.2 | 123.4 | 33.6 | 31.2 | 11.0 | 10.4 | 5.0 | 9.6 | 6.0 | 7.4 | 15.2 |
| kr-vs-kp | 7.6 | 112.6 | 17.4 | 9.6 | 7.0 | 5.2 | 8.4 | 7.0 | 13.6 | 20.8 | 25.8 | 7.6 | 118.6 | 18.4 | 10.8 | 6.2 | 5.8 | 8.4 | 7.0 | 14.6 | 20.4 | 29.0 |
| mushroom | 10.4 | 86.8 | 7.0 | 7.0 | 3.4 | 5.6 | 10.2 | 9.4 | 5.8 | 5.0 | 9.8 | 10.0 | 101.8 | 7.6 | 8.2 | 4.0 | 4.0 | 10.2 | 9.4 | 8.0 | 10.0 | 24.0 |

### Average conditions per rule

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 3.59 | 3.73 | 1.73 | 2.33 | 1.70 | 1.43 | 1.25 | 3.00 | 1.87 | 2.09 | 3.34 | 3.80 | 3.73 | 1.88 | 1.95 | 1.53 | 1.10 | 1.25 | 3.00 | 1.20 | 1.42 | 2.61 |
| breast-cancer | 3.77 | 3.79 | 1.70 | 2.50 | 1.17 | 1.07 | 1.42 | 2.91 | 1.80 | 4.48 | 5.22 | 3.75 | 3.78 | 1.55 | 2.70 | 1.43 | 1.70 | 1.42 | 2.91 | 2.22 | 1.92 | 1.77 |
| colic | 3.72 | 3.74 | 2.25 | 1.83 | 1.33 | 1.60 | 1.55 | 2.83 | 1.97 | 5.10 | 6.19 | 3.72 | 3.73 | 1.61 | 2.13 | 1.13 | 1.60 | 1.55 | 2.83 | 1.83 | 1.81 | 2.17 |
| credit-approval | 3.87 | 3.87 | 2.53 | 2.52 | 1.20 | 2.35 | 1.58 | 2.93 | 2.05 | 4.06 | 7.23 | 3.87 | 3.84 | 2.86 | 3.22 | 1.10 | 1.85 | 1.58 | 2.93 | 2.18 | 2.37 | 4.02 |
| credit-g | 3.93 | 3.90 | 4.83 | 2.95 | 2.70 | 1.54 | 1.67 | 2.95 | 3.77 | 4.09 | 11.42 | 3.95 | 3.89 | 3.64 | 2.68 | 2.96 | 1.59 | 1.67 | 2.95 | 2.69 | 3.02 | 5.57 |
| diabetes | 3.90 | 3.91 | 2.71 | 3.74 | 1.71 | 3.14 | 1.79 | 2.94 | 2.68 | 4.08 | 8.96 | 3.90 | 3.88 | 2.46 | 2.93 | 1.69 | 1.43 | 1.79 | 2.94 | 2.40 | 2.51 | 6.13 |
| sonar | 3.82 | 3.73 | 2.25 | 2.12 | 1.87 | 2.15 | nan | nan | 1.74 | 3.59 | 5.73 | 3.76 | 3.67 | 1.66 | 2.59 | 1.08 | 1.78 | nan | nan | 1.73 | 2.41 | 4.51 |
| ionosphere | 3.46 | 3.57 | 1.55 | 3.20 | 1.23 | 2.00 | nan | nan | 1.47 | 3.09 | 4.64 | 3.49 | 3.51 | 2.07 | 2.47 | 1.20 | 1.00 | nan | nan | 1.31 | 2.83 | 5.04 |
| tic-tac-toe | 3.86 | 3.96 | 3.07 | 3.22 | 2.86 | 2.74 | 1.81 | 3.00 | 3.08 | 3.23 | 5.88 | 3.86 | 3.98 | 3.28 | 3.13 | 2.84 | 2.29 | 1.81 | 3.00 | 3.28 | 2.69 | 4.53 |
| banknote-authentication | 3.77 | 3.81 | 2.33 | 2.49 | 1.71 | 2.13 | 1.89 | 2.96 | 2.30 | 2.10 | 3.88 | 3.69 | 3.74 | 3.65 | 3.85 | 2.69 | 3.05 | 1.89 | 2.96 | 2.24 | 2.11 | 4.51 |
| kr-vs-kp | 3.43 | 3.73 | 3.23 | 4.54 | 2.69 | 4.33 | 1.87 | 2.86 | 3.00 | 2.93 | 7.37 | 3.43 | 3.75 | 3.33 | 4.80 | 3.00 | 3.92 | 1.87 | 2.86 | 3.16 | 3.21 | 7.71 |
| mushroom | 3.55 | 3.39 | 2.02 | 1.56 | 1.80 | 1.85 | 1.74 | 2.78 | 1.64 | 2.76 | 3.83 | 3.50 | 3.58 | 2.54 | 1.50 | 1.50 | 1.00 | 1.74 | 2.78 | 1.56 | 1.85 | 2.54 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| algorithm | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| Binarized/brl | 85.14 | 4.012 | 6.1 | 1.66 | 2 |
| Original/brl | 85.14 | 3.979 | 6.1 | 1.66 | 2 |
| Binarized/brs | 78.08 | 14.393 | 7.9 | 2.92 | 2 |
| Original/brs | 78.08 | 14.361 | 7.9 | 2.92 | 2 |
| Binarized/forest | 85.06 | 0.112 | 125.8 | 3.76 | 0 |
| Original/forest | 85.05 | 0.194 | 124.3 | 3.76 | 0 |
| Binarized/irep_A | 85.68 | 0.282 | 3.3 | 1.83 | 0 |
| Original/irep_A | 83.00 | 0.186 | 4.4 | 1.85 | 0 |
| Binarized/irep_B | 84.39 | 0.288 | 3.3 | 2.19 | 0 |
| Original/irep_B | 82.18 | 0.183 | 3.7 | 1.86 | 0 |
| Binarized/j48 | 86.60 | 0.562 | 24.4 | 6.14 | 0 |
| Original/j48 | 85.50 | 0.526 | 26.6 | 4.26 | 0 |
| Binarized/jrip | 86.94 | 0.684 | 4.7 | 2.28 | 0 |
| Original/jrip | 87.19 | 0.537 | 5.2 | 2.15 | 0 |
| Binarized/part | 85.86 | 1.456 | 20.2 | 3.47 | 0 |
| Original/part | 86.33 | 1.232 | 18.0 | 2.35 | 0 |
| Binarized/ripper_A | 84.49 | 0.533 | 5.7 | 2.52 | 0 |
| Original/ripper_A | 83.37 | 0.345 | 9.6 | 2.54 | 0 |
| Binarized/ripper_B | 86.25 | 0.504 | 4.9 | 2.75 | 0 |
| Original/ripper_B | 83.30 | 0.343 | 9.2 | 2.83 | 0 |
| Binarized/tree | 80.12 | 0.037 | 12.3 | 3.72 | 0 |
| Original/tree | 84.72 | 0.007 | 12.2 | 3.73 | 0 |

**Average rank per criterion** (1 = best of 22; failed entries tie for last)

| algorithm | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| Binarized/brl | 14.50 | 18.83 | 11.62 | 6.33 |
| Original/brl | 14.50 | 18.00 | 11.62 | 6.33 |
| Binarized/brs | 18.42 | 21.25 | 13.04 | 13.83 |
| Original/brs | 18.42 | 21.42 | 13.04 | 13.83 |
| Binarized/forest | 9.88 | 5.17 | 21.08 | 17.75 |
| Original/forest | 10.54 | 6.92 | 20.58 | 17.33 |
| Binarized/irep_A | 11.04 | 7.33 | 3.17 | 4.42 |
| Original/irep_A | 14.75 | 5.00 | 5.46 | 4.75 |
| Binarized/irep_B | 12.50 | 7.17 | 3.29 | 7.46 |
| Original/irep_B | 13.00 | 4.75 | 3.92 | 5.25 |
| Binarized/j48 | 6.79 | 13.75 | 17.33 | 20.75 |
| Original/j48 | 7.42 | 12.00 | 17.58 | 17.75 |
| Binarized/jrip | 5.88 | 15.00 | 5.54 | 8.50 |
| Original/jrip | 5.21 | 12.58 | 7.04 | 7.08 |
| Binarized/part | 9.00 | 18.58 | 14.71 | 14.75 |
| Original/part | 7.92 | 17.33 | 14.92 | 8.58 |
| Binarized/ripper_A | 11.88 | 12.75 | 7.46 | 10.71 |
| Original/ripper_A | 13.25 | 9.42 | 11.88 | 10.08 |
| Binarized/ripper_B | 7.50 | 12.17 | 7.04 | 11.33 |
| Original/ripper_B | 12.04 | 9.75 | 12.17 | 11.92 |
| Binarized/tree | 16.17 | 2.75 | 15.46 | 17.12 |
| Original/tree | 12.42 | 1.08 | 15.04 | 17.12 |

**Binarized vs. Original: how often each was better, per model** (ties count 0.5 each per side; a dataset where both failed isn't counted for either side)

| model | accuracy (Bin / Orig) | fit time (Bin / Orig) | n_rules (Bin / Orig) | avg_conditions (Bin / Orig) |
|---|---|---|---|---|
| brl | 5.0 / 5.0 | 1.0 / 9.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| brs | 5.0 / 5.0 | 6.0 / 4.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| forest | 8.0 / 4.0 | 11.0 / 1.0 | 3.0 / 9.0 | 4.0 / 8.0 |
| irep_A | 10.5 / 1.5 | 2.0 / 10.0 | 8.0 / 4.0 | 4.0 / 8.0 |
| irep_B | 6.0 / 6.0 | 3.0 / 9.0 | 6.5 / 5.5 | 3.5 / 8.5 |
| j48 | 6.5 / 5.5 | 2.0 / 10.0 | 6.0 / 6.0 | 3.0 / 9.0 |
| jrip | 5.0 / 7.0 | 2.0 / 10.0 | 9.5 / 2.5 | 4.0 / 8.0 |
| part | 3.5 / 8.5 | 1.0 / 11.0 | 6.0 / 6.0 | 2.0 / 10.0 |
| ripper_A | 8.0 / 4.0 | 2.0 / 10.0 | 12.0 / 0.0 | 7.0 / 5.0 |
| ripper_B | 8.0 / 4.0 | 2.0 / 10.0 | 12.0 / 0.0 | 6.0 / 6.0 |
| tree | 4.0 / 8.0 | 1.0 / 11.0 | 5.0 / 7.0 | 5.5 / 6.5 |
