# Binarized vs. Original data preparation -- comparison across binary datasets

Generated 2026-09-27 18:54:34. N_FOLDS=5, MAX_INTERVALS=8 ('Binarized' mode only), MAX_DEPTH=4, RIPPER_K=2, N_ESTIMATORS=10 (forest), FIT_TIMEOUT_SECONDS=60. `ripper_A`/`irep_A` treat each dataset's (alphabetically) first class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit the same already-Boolean matrix under both workflows as two independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` fit-time includes JVM subprocess startup overhead, not just the algorithm itself -- see the module docstring.

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
| Binarized | irep_A | 93.56 +/- 1.17 | 0/5 |
| Binarized | irep_B | 94.25 +/- 2.06 | 0/5 |
| Binarized | brl | 94.48 +/- 1.84 | 0/5 |
| Binarized | brs | 88.05 +/- 5.12 | 0/5 |
| Binarized | jrip | 95.40 +/- 1.03 | 0/5 |
| Binarized | part | 95.17 +/- 1.13 | 0/5 |
| Binarized | j48 | 96.09 +/- 1.17 | 0/5 |
| Original | tree | 94.94 +/- 1.17 | 0/5 |
| Original | forest | 94.71 +/- 2.13 | 0/5 |
| Original | ripper_A | 95.17 +/- 1.13 | 0/5 |
| Original | ripper_B | 95.17 +/- 1.52 | 0/5 |
| Original | irep_A | 94.25 +/- 1.63 | 0/5 |
| Original | irep_B | 95.63 +/- 2.34 | 0/5 |
| Original | brl | 94.48 +/- 1.84 | 0/5 |
| Original | brs | 88.05 +/- 5.12 | 0/5 |
| Original | jrip | 95.63 +/- 2.34 | 0/5 |
| Original | part | 95.63 +/- 1.34 | 0/5 |
| Original | j48 | 96.32 +/- 1.34 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.191 |
| Binarized | forest | 0.051 |
| Binarized | ripper_A | 0.052 |
| Binarized | ripper_B | 0.047 |
| Binarized | irep_A | 0.029 |
| Binarized | irep_B | 0.028 |
| Binarized | brl | 0.547 |
| Binarized | brs | 7.176 |
| Binarized | jrip | 0.436 |
| Binarized | part | 1.265 |
| Binarized | j48 | 0.398 |
| Original | tree | 0.003 |
| Original | forest | 0.060 |
| Original | ripper_A | 0.067 |
| Original | ripper_B | 0.059 |
| Original | irep_A | 0.042 |
| Original | irep_B | 0.036 |
| Original | brl | 0.481 |
| Original | brs | 6.946 |
| Original | jrip | 1.221 |
| Original | part | 0.374 |
| Original | j48 | 1.246 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.8 | 3.59 |
| Binarized | forest | 126.4 | 3.73 |
| Binarized | ripper_A | 2.4 | 1.73 |
| Binarized | ripper_B | 3.0 | 2.33 |
| Binarized | irep_A | 2.4 | 1.47 |
| Binarized | irep_B | 1.4 | 1.07 |
| Binarized | brl | 3.2 | 1.25 |
| Binarized | brs | 7.8 | 2.98 |
| Binarized | jrip | 2.2 | 1.87 |
| Binarized | part | 6.4 | 2.09 |
| Binarized | j48 | 6.6 | 3.34 |
| Original | tree | 13.4 | 3.80 |
| Original | forest | 121.0 | 3.73 |
| Original | ripper_A | 3.4 | 1.88 |
| Original | ripper_B | 3.8 | 1.95 |
| Original | irep_A | 2.8 | 1.73 |
| Original | irep_B | 1.0 | 1.00 |
| Original | brl | 3.2 | 1.25 |
| Original | brs | 7.8 | 2.98 |
| Original | jrip | 1.2 | 1.20 |
| Original | part | 5.2 | 1.42 |
| Original | j48 | 9.0 | 2.61 |

(104.7s total)

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
| Binarized | irep_A | 72.02 +/- 1.68 | 0/5 |
| Binarized | irep_B | 73.40 +/- 4.18 | 0/5 |
| Binarized | brl | 73.09 +/- 2.26 | 0/5 |
| Binarized | brs | 68.55 +/- 3.77 | 0/5 |
| Binarized | jrip | 73.06 +/- 5.36 | 0/5 |
| Binarized | part | 66.44 +/- 4.94 | 0/5 |
| Binarized | j48 | 70.64 +/- 3.34 | 0/5 |
| Original | tree | 72.72 +/- 1.92 | 0/5 |
| Original | forest | 75.51 +/- 3.43 | 0/5 |
| Original | ripper_A | 66.79 +/- 2.79 | 0/5 |
| Original | ripper_B | 73.07 +/- 2.21 | 0/5 |
| Original | irep_A | 74.82 +/- 2.46 | 0/5 |
| Original | irep_B | 74.10 +/- 3.70 | 0/5 |
| Original | brl | 73.09 +/- 2.26 | 0/5 |
| Original | brs | 68.55 +/- 3.77 | 0/5 |
| Original | jrip | 72.02 +/- 3.56 | 0/5 |
| Original | part | 69.96 +/- 5.62 | 0/5 |
| Original | j48 | 73.42 +/- 1.45 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.039 |
| Binarized | ripper_A | 0.097 |
| Binarized | ripper_B | 0.119 |
| Binarized | irep_A | 0.058 |
| Binarized | irep_B | 0.056 |
| Binarized | brl | 0.257 |
| Binarized | brs | 7.022 |
| Binarized | jrip | 0.430 |
| Binarized | part | 1.320 |
| Binarized | j48 | 0.418 |
| Original | tree | 0.003 |
| Original | forest | 0.048 |
| Original | ripper_A | 0.055 |
| Original | ripper_B | 0.082 |
| Original | irep_A | 0.029 |
| Original | irep_B | 0.028 |
| Original | brl | 0.242 |
| Original | brs | 6.929 |
| Original | jrip | 1.213 |
| Original | part | 0.366 |
| Original | j48 | 1.227 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.4 | 3.77 |
| Binarized | forest | 128.0 | 3.79 |
| Binarized | ripper_A | 1.4 | 1.70 |
| Binarized | ripper_B | 1.4 | 2.50 |
| Binarized | irep_A | 2.4 | 1.27 |
| Binarized | irep_B | 1.0 | 1.80 |
| Binarized | brl | 3.2 | 1.42 |
| Binarized | brs | 7.0 | 2.86 |
| Binarized | jrip | 1.0 | 1.80 |
| Binarized | part | 21.6 | 4.48 |
| Binarized | j48 | 13.8 | 5.22 |
| Original | tree | 12.4 | 3.75 |
| Original | forest | 124.8 | 3.78 |
| Original | ripper_A | 3.0 | 1.55 |
| Original | ripper_B | 1.6 | 2.70 |
| Original | irep_A | 3.2 | 1.17 |
| Original | irep_B | 1.6 | 1.90 |
| Original | brl | 3.2 | 1.42 |
| Original | brs | 7.0 | 2.86 |
| Original | jrip | 2.8 | 2.22 |
| Original | part | 15.4 | 1.92 |
| Original | j48 | 8.4 | 1.77 |

(101.2s total)

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
| Binarized | irep_A | 83.98 +/- 1.72 | 0/5 |
| Binarized | irep_B | 85.33 +/- 2.86 | 0/5 |
| Binarized | brl | 84.79 +/- 1.53 | 0/5 |
| Binarized | brs | 80.17 +/- 3.45 | 0/5 |
| Binarized | jrip | 86.42 +/- 2.97 | 0/5 |
| Binarized | part | 81.81 +/- 4.54 | 0/5 |
| Binarized | j48 | 83.43 +/- 2.59 | 0/5 |
| Original | tree | 84.25 +/- 3.11 | 0/5 |
| Original | forest | 84.26 +/- 3.31 | 0/5 |
| Original | ripper_A | 81.27 +/- 3.39 | 0/5 |
| Original | ripper_B | 86.96 +/- 2.91 | 0/5 |
| Original | irep_A | 84.25 +/- 1.78 | 0/5 |
| Original | irep_B | 85.87 +/- 1.35 | 0/5 |
| Original | brl | 84.79 +/- 1.53 | 0/5 |
| Original | brs | 80.17 +/- 3.45 | 0/5 |
| Original | jrip | 87.51 +/- 3.34 | 0/5 |
| Original | part | 85.88 +/- 2.47 | 0/5 |
| Original | j48 | 82.63 +/- 5.17 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.010 |
| Binarized | forest | 0.048 |
| Binarized | ripper_A | 0.400 |
| Binarized | ripper_B | 0.425 |
| Binarized | irep_A | 0.280 |
| Binarized | irep_B | 0.281 |
| Binarized | brl | 2.278 |
| Binarized | brs | 31.069 |
| Binarized | jrip | 0.553 |
| Binarized | part | 1.411 |
| Binarized | j48 | 0.472 |
| Original | tree | 0.004 |
| Original | forest | 0.067 |
| Original | ripper_A | 0.251 |
| Original | ripper_B | 0.242 |
| Original | irep_A | 0.176 |
| Original | irep_B | 0.162 |
| Original | brl | 2.256 |
| Original | brs | 31.108 |
| Original | jrip | 0.393 |
| Original | part | 1.262 |
| Original | j48 | 0.366 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 11.8 | 3.72 |
| Binarized | forest | 122.0 | 3.74 |
| Binarized | ripper_A | 3.6 | 2.25 |
| Binarized | ripper_B | 3.8 | 1.83 |
| Binarized | irep_A | 2.2 | 1.17 |
| Binarized | irep_B | 1.4 | 1.70 |
| Binarized | brl | 4.8 | 1.55 |
| Binarized | brs | 7.8 | 2.84 |
| Binarized | jrip | 3.2 | 1.97 |
| Binarized | part | 11.6 | 5.10 |
| Binarized | j48 | 17.0 | 6.19 |
| Original | tree | 11.6 | 3.72 |
| Original | forest | 118.6 | 3.73 |
| Original | ripper_A | 6.0 | 1.61 |
| Original | ripper_B | 6.2 | 2.13 |
| Original | irep_A | 3.6 | 1.22 |
| Original | irep_B | 1.4 | 1.80 |
| Original | brl | 4.8 | 1.55 |
| Original | brs | 7.8 | 2.84 |
| Original | jrip | 3.4 | 1.83 |
| Original | part | 12.8 | 1.81 |
| Original | j48 | 8.6 | 2.17 |

(371.1s total)

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
| Binarized | irep_A | 85.51 +/- 3.24 | 0/5 |
| Binarized | irep_B | 84.20 +/- 3.59 | 0/5 |
| Binarized | brl | 84.20 +/- 2.65 | 0/5 |
| Binarized | brs | 61.16 +/- 1.75 | 0/5 |
| Binarized | jrip | 85.65 +/- 2.48 | 0/5 |
| Binarized | part | 83.04 +/- 3.60 | 0/5 |
| Binarized | j48 | 84.64 +/- 3.41 | 0/5 |
| Original | tree | 84.78 +/- 3.89 | 0/5 |
| Original | forest | 85.51 +/- 3.37 | 0/5 |
| Original | ripper_A | 83.62 +/- 3.57 | 0/5 |
| Original | ripper_B | 83.04 +/- 3.63 | 0/5 |
| Original | irep_A | 85.51 +/- 3.30 | 0/5 |
| Original | irep_B | 83.48 +/- 2.80 | 0/5 |
| Original | brl | 84.20 +/- 2.65 | 0/5 |
| Original | brs | 61.16 +/- 1.75 | 0/5 |
| Original | jrip | 85.94 +/- 2.58 | 0/5 |
| Original | part | 84.35 +/- 3.51 | 0/5 |
| Original | j48 | 85.94 +/- 4.09 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.011 |
| Binarized | forest | 0.080 |
| Binarized | ripper_A | 0.351 |
| Binarized | ripper_B | 0.349 |
| Binarized | irep_A | 0.152 |
| Binarized | irep_B | 0.162 |
| Binarized | brl | 1.579 |
| Binarized | brs | 10.222 |
| Binarized | jrip | 0.631 |
| Binarized | part | 1.436 |
| Binarized | j48 | 0.525 |
| Original | tree | 0.004 |
| Original | forest | 0.109 |
| Original | ripper_A | 0.251 |
| Original | ripper_B | 0.255 |
| Original | irep_A | 0.089 |
| Original | irep_B | 0.088 |
| Original | brl | 1.525 |
| Original | brs | 10.142 |
| Original | jrip | 0.576 |
| Original | part | 1.327 |
| Original | j48 | 0.526 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.4 | 3.87 |
| Binarized | forest | 138.8 | 3.87 |
| Binarized | ripper_A | 5.0 | 2.53 |
| Binarized | ripper_B | 3.6 | 2.52 |
| Binarized | irep_A | 1.4 | 1.30 |
| Binarized | irep_B | 2.2 | 1.85 |
| Binarized | brl | 4.8 | 1.58 |
| Binarized | brs | 8.6 | 2.94 |
| Binarized | jrip | 3.2 | 2.05 |
| Binarized | part | 23.6 | 4.06 |
| Binarized | j48 | 23.0 | 7.23 |
| Original | tree | 14.2 | 3.87 |
| Original | forest | 134.0 | 3.84 |
| Original | ripper_A | 7.6 | 2.86 |
| Original | ripper_B | 8.8 | 3.22 |
| Original | irep_A | 1.4 | 1.10 |
| Original | irep_B | 2.4 | 1.57 |
| Original | brl | 4.8 | 1.58 |
| Original | brs | 8.6 | 2.94 |
| Original | jrip | 3.8 | 2.18 |
| Original | part | 29.6 | 2.37 |
| Original | j48 | 19.4 | 4.02 |

(154.5s total)

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
| Binarized | irep_A | 70.30 +/- 4.70 | 0/5 |
| Binarized | irep_B | 71.80 +/- 0.93 | 0/5 |
| Binarized | brl | 69.80 +/- 2.23 | 0/5 |
| Binarized | brs | 45.30 +/- 3.36 | 0/5 |
| Binarized | jrip | 72.70 +/- 1.94 | 0/5 |
| Binarized | part | 72.30 +/- 3.23 | 0/5 |
| Binarized | j48 | 71.90 +/- 1.24 | 0/5 |
| Original | tree | 70.60 +/- 2.60 | 0/5 |
| Original | forest | 70.10 +/- 1.16 | 0/5 |
| Original | ripper_A | 70.20 +/- 1.03 | 0/5 |
| Original | ripper_B | 58.60 +/- 5.07 | 0/5 |
| Original | irep_A | 69.90 +/- 1.28 | 0/5 |
| Original | irep_B | 70.80 +/- 2.01 | 0/5 |
| Original | brl | 69.80 +/- 2.23 | 0/5 |
| Original | brs | 45.30 +/- 3.36 | 0/5 |
| Original | jrip | 72.40 +/- 2.40 | 0/5 |
| Original | part | 70.00 +/- 3.36 | 0/5 |
| Original | j48 | 70.80 +/- 1.91 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.014 |
| Binarized | forest | 0.102 |
| Binarized | ripper_A | 0.534 |
| Binarized | ripper_B | 0.465 |
| Binarized | irep_A | 0.186 |
| Binarized | irep_B | 0.192 |
| Binarized | brl | 2.785 |
| Binarized | brs | 11.633 |
| Binarized | jrip | 0.849 |
| Binarized | part | 1.782 |
| Binarized | j48 | 0.785 |
| Original | tree | 0.005 |
| Original | forest | 0.154 |
| Original | ripper_A | 0.297 |
| Original | ripper_B | 0.244 |
| Original | irep_A | 0.089 |
| Original | irep_B | 0.095 |
| Original | brl | 2.808 |
| Original | brs | 11.767 |
| Original | jrip | 1.108 |
| Original | part | 1.563 |
| Original | j48 | 0.708 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.0 | 3.93 |
| Binarized | forest | 145.0 | 3.90 |
| Binarized | ripper_A | 3.2 | 4.83 |
| Binarized | ripper_B | 3.8 | 2.95 |
| Binarized | irep_A | 1.8 | 2.90 |
| Binarized | irep_B | 4.0 | 1.34 |
| Binarized | brl | 4.8 | 1.67 |
| Binarized | brs | 8.6 | 2.86 |
| Binarized | jrip | 3.0 | 3.77 |
| Binarized | part | 56.8 | 4.09 |
| Binarized | j48 | 78.8 | 11.42 |
| Original | tree | 15.2 | 3.95 |
| Original | forest | 141.8 | 3.89 |
| Original | ripper_A | 5.4 | 3.64 |
| Original | ripper_B | 5.2 | 2.68 |
| Original | irep_A | 2.2 | 2.62 |
| Original | irep_B | 5.0 | 1.13 |
| Original | brl | 4.8 | 1.67 |
| Original | brs | 8.6 | 2.86 |
| Original | jrip | 3.4 | 2.69 |
| Original | part | 60.4 | 3.02 |
| Original | j48 | 77.8 | 5.57 |

(194.6s total)

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
| Binarized | irep_A | 74.74 +/- 1.77 | 0/5 |
| Binarized | irep_B | 75.65 +/- 2.07 | 0/5 |
| Binarized | brl | 73.70 +/- 1.26 | 0/5 |
| Binarized | brs | 69.40 +/- 1.50 | 0/5 |
| Binarized | jrip | 74.61 +/- 2.13 | 0/5 |
| Binarized | part | 74.22 +/- 1.56 | 0/5 |
| Binarized | j48 | 74.99 +/- 2.62 | 0/5 |
| Original | tree | 73.83 +/- 3.41 | 0/5 |
| Original | forest | 74.09 +/- 2.92 | 0/5 |
| Original | ripper_A | 59.62 +/- 4.74 | 0/5 |
| Original | ripper_B | 70.98 +/- 4.92 | 0/5 |
| Original | irep_A | 69.66 +/- 5.18 | 0/5 |
| Original | irep_B | 73.44 +/- 0.97 | 0/5 |
| Original | brl | 73.70 +/- 1.26 | 0/5 |
| Original | brs | 69.40 +/- 1.50 | 0/5 |
| Original | jrip | 75.14 +/- 5.03 | 0/5 |
| Original | part | 71.75 +/- 2.68 | 0/5 |
| Original | j48 | 71.74 +/- 3.49 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.010 |
| Binarized | forest | 0.082 |
| Binarized | ripper_A | 0.297 |
| Binarized | ripper_B | 0.313 |
| Binarized | irep_A | 0.106 |
| Binarized | irep_B | 0.121 |
| Binarized | brl | 1.133 |
| Binarized | brs | 9.509 |
| Binarized | jrip | 0.759 |
| Binarized | part | 1.423 |
| Binarized | j48 | 0.812 |
| Original | tree | 0.003 |
| Original | forest | 0.151 |
| Original | ripper_A | 0.215 |
| Original | ripper_B | 0.231 |
| Original | irep_A | 0.087 |
| Original | irep_B | 0.064 |
| Original | brl | 1.149 |
| Original | brs | 9.436 |
| Original | jrip | 1.033 |
| Original | part | 1.452 |
| Original | j48 | 0.652 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.6 | 3.90 |
| Binarized | forest | 145.4 | 3.91 |
| Binarized | ripper_A | 4.4 | 2.71 |
| Binarized | ripper_B | 3.8 | 3.74 |
| Binarized | irep_A | 3.0 | 1.52 |
| Binarized | irep_B | 4.0 | 3.16 |
| Binarized | brl | 4.6 | 1.79 |
| Binarized | brs | 7.4 | 2.92 |
| Binarized | jrip | 3.0 | 2.68 |
| Binarized | part | 46.2 | 4.08 |
| Binarized | j48 | 42.2 | 8.96 |
| Original | tree | 14.6 | 3.90 |
| Original | forest | 140.4 | 3.88 |
| Original | ripper_A | 8.2 | 2.46 |
| Original | ripper_B | 8.2 | 2.93 |
| Original | irep_A | 8.4 | 1.98 |
| Original | irep_B | 1.8 | 1.53 |
| Original | brl | 4.6 | 1.79 |
| Original | brs | 7.4 | 2.92 |
| Original | jrip | 2.4 | 2.40 |
| Original | part | 6.8 | 2.51 |
| Original | j48 | 22.0 | 6.13 |

(147.5s total)

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
| Binarized | irep_A | 72.14 +/- 4.78 | 0/5 |
| Binarized | irep_B | 70.22 +/- 7.20 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 70.19 +/- 6.53 | 0/5 |
| Binarized | part | 74.53 +/- 5.71 | 0/5 |
| Binarized | j48 | 74.47 +/- 5.62 | 0/5 |
| Original | tree | 75.48 +/- 4.09 | 0/5 |
| Original | forest | 78.37 +/- 4.50 | 0/5 |
| Original | ripper_A | 62.11 +/- 11.07 | 0/5 |
| Original | ripper_B | 64.44 +/- 6.02 | 0/5 |
| Original | irep_A | 55.30 +/- 5.48 | 0/5 |
| Original | irep_B | 63.90 +/- 7.74 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | n/a | 5/5 |
| Original | jrip | 72.10 +/- 3.95 | 0/5 |
| Original | part | 75.96 +/- 5.32 | 0/5 |
| Original | j48 | 71.56 +/- 7.81 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.013 |
| Binarized | forest | 0.045 |
| Binarized | ripper_A | 1.057 |
| Binarized | ripper_B | 1.005 |
| Binarized | irep_A | 0.839 |
| Binarized | irep_B | 0.825 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.967 |
| Binarized | part | 1.403 |
| Binarized | j48 | 0.826 |
| Original | tree | 0.006 |
| Original | forest | 0.083 |
| Original | ripper_A | 0.995 |
| Original | ripper_B | 0.953 |
| Original | irep_A | 0.833 |
| Original | irep_B | 0.818 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.436 |
| Original | part | 1.517 |
| Original | j48 | 0.659 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.6 | 3.82 |
| Binarized | forest | 121.4 | 3.73 |
| Binarized | ripper_A | 3.0 | 2.25 |
| Binarized | ripper_B | 3.2 | 2.12 |
| Binarized | irep_A | 1.8 | 1.50 |
| Binarized | irep_B | 2.8 | 1.80 |
| Binarized | brl | nan | nan |
| Binarized | brs | nan | nan |
| Binarized | jrip | 3.4 | 1.74 |
| Binarized | part | 6.2 | 3.59 |
| Binarized | j48 | 16.0 | 5.73 |
| Original | tree | 13.0 | 3.76 |
| Original | forest | 114.8 | 3.67 |
| Original | ripper_A | 4.4 | 1.66 |
| Original | ripper_B | 3.4 | 2.59 |
| Original | irep_A | 2.6 | 1.13 |
| Original | irep_B | 3.4 | 1.75 |
| Original | brl | nan | nan |
| Original | brs | nan | nan |
| Original | jrip | 3.6 | 1.73 |
| Original | part | 6.4 | 2.41 |
| Original | j48 | 14.0 | 4.51 |

(1282.5s total)

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
| Binarized | irep_A | 85.46 +/- 3.56 | 0/5 |
| Binarized | irep_B | 87.75 +/- 3.58 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | n/a | 5/5 |
| Binarized | jrip | 89.18 +/- 4.10 | 0/5 |
| Binarized | part | 91.74 +/- 2.75 | 0/5 |
| Binarized | j48 | 90.31 +/- 3.05 | 0/5 |
| Original | tree | 88.61 +/- 3.24 | 0/5 |
| Original | forest | 91.45 +/- 3.61 | 0/5 |
| Original | ripper_A | 88.04 +/- 4.18 | 0/5 |
| Original | ripper_B | 75.51 +/- 2.66 | 0/5 |
| Original | irep_A | 84.05 +/- 1.87 | 0/5 |
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
| Binarized | forest | 0.042 |
| Binarized | ripper_A | 0.534 |
| Binarized | ripper_B | 0.508 |
| Binarized | irep_A | 0.395 |
| Binarized | irep_B | 0.395 |
| Binarized | brl | nan |
| Binarized | brs | nan |
| Binarized | jrip | 0.653 |
| Binarized | part | 1.411 |
| Binarized | j48 | 0.837 |
| Original | tree | 0.006 |
| Original | forest | 0.081 |
| Original | ripper_A | 0.483 |
| Original | ripper_B | 0.431 |
| Original | irep_A | 0.335 |
| Original | irep_B | 0.320 |
| Original | brl | nan |
| Original | brs | nan |
| Original | jrip | 0.628 |
| Original | part | 1.462 |
| Original | j48 | 0.656 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 8.6 | 3.46 |
| Binarized | forest | 98.6 | 3.57 |
| Binarized | ripper_A | 5.6 | 1.55 |
| Binarized | ripper_B | 3.0 | 3.20 |
| Binarized | irep_A | 2.2 | 1.37 |
| Binarized | irep_B | 2.4 | 2.53 |
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

(1260.6s total)

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
| Binarized | irep_A | 94.46 +/- 7.23 | 0/5 |
| Binarized | irep_B | 80.80 +/- 4.97 | 0/5 |
| Binarized | brl | 80.58 +/- 3.08 | 0/5 |
| Binarized | brs | 85.91 +/- 4.74 | 0/5 |
| Binarized | jrip | 98.33 +/- 1.01 | 0/5 |
| Binarized | part | 93.84 +/- 1.85 | 0/5 |
| Binarized | j48 | 94.88 +/- 0.77 | 0/5 |
| Original | tree | 82.36 +/- 2.06 | 0/5 |
| Original | forest | 76.83 +/- 2.17 | 0/5 |
| Original | ripper_A | 97.91 +/- 0.93 | 0/5 |
| Original | ripper_B | 97.39 +/- 2.26 | 0/5 |
| Original | irep_A | 89.35 +/- 8.00 | 0/5 |
| Original | irep_B | 82.89 +/- 3.46 | 0/5 |
| Original | brl | 80.58 +/- 3.08 | 0/5 |
| Original | brs | 85.91 +/- 4.74 | 0/5 |
| Original | jrip | 97.91 +/- 0.87 | 0/5 |
| Original | part | 94.15 +/- 2.35 | 0/5 |
| Original | j48 | 87.27 +/- 2.85 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.012 |
| Binarized | forest | 0.100 |
| Binarized | ripper_A | 0.175 |
| Binarized | ripper_B | 0.179 |
| Binarized | irep_A | 0.079 |
| Binarized | irep_B | 0.062 |
| Binarized | brl | 0.905 |
| Binarized | brs | 6.819 |
| Binarized | jrip | 0.632 |
| Binarized | part | 1.393 |
| Binarized | j48 | 0.724 |
| Original | tree | 0.003 |
| Original | forest | 0.117 |
| Original | ripper_A | 0.111 |
| Original | ripper_B | 0.140 |
| Original | irep_A | 0.038 |
| Original | irep_B | 0.036 |
| Original | brl | 0.823 |
| Original | brs | 6.907 |
| Original | jrip | 0.917 |
| Original | part | 1.457 |
| Original | j48 | 0.713 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.0 | 3.86 |
| Binarized | forest | 153.4 | 3.96 |
| Binarized | ripper_A | 8.6 | 3.07 |
| Binarized | ripper_B | 9.8 | 3.22 |
| Binarized | irep_A | 7.8 | 3.01 |
| Binarized | irep_B | 5.8 | 2.63 |
| Binarized | brl | 11.8 | 1.81 |
| Binarized | brs | 8.2 | 3.00 |
| Binarized | jrip | 8.0 | 3.08 |
| Binarized | part | 29.2 | 3.23 |
| Binarized | j48 | 37.8 | 5.88 |
| Original | tree | 14.0 | 3.86 |
| Original | forest | 157.4 | 3.98 |
| Original | ripper_A | 9.4 | 3.28 |
| Original | ripper_B | 14.0 | 3.13 |
| Original | irep_A | 6.4 | 2.78 |
| Original | irep_B | 6.0 | 2.09 |
| Original | brl | 11.8 | 1.81 |
| Original | brs | 8.2 | 3.00 |
| Original | jrip | 9.6 | 3.28 |
| Original | part | 36.4 | 2.69 |
| Original | j48 | 80.6 | 4.53 |

(113.3s total)

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
| Binarized | irep_A | 94.61 +/- 1.36 | 0/5 |
| Binarized | irep_B | 96.50 +/- 1.71 | 0/5 |
| Binarized | brl | 97.16 +/- 1.34 | 0/5 |
| Binarized | brs | 96.72 +/- 2.40 | 0/5 |
| Binarized | jrip | 98.61 +/- 0.54 | 0/5 |
| Binarized | part | 98.32 +/- 0.68 | 0/5 |
| Binarized | j48 | 98.54 +/- 0.65 | 0/5 |
| Original | tree | 95.70 +/- 1.63 | 0/5 |
| Original | forest | 97.01 +/- 1.27 | 0/5 |
| Original | ripper_A | 97.09 +/- 0.86 | 0/5 |
| Original | ripper_B | 95.63 +/- 2.12 | 0/5 |
| Original | irep_A | 91.76 +/- 1.26 | 0/5 |
| Original | irep_B | 90.16 +/- 3.36 | 0/5 |
| Original | brl | 97.16 +/- 1.34 | 0/5 |
| Original | brs | 96.72 +/- 2.40 | 0/5 |
| Original | jrip | 97.89 +/- 0.63 | 0/5 |
| Original | part | 98.76 +/- 0.75 | 0/5 |
| Original | j48 | 98.61 +/- 1.25 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.013 |
| Binarized | forest | 0.110 |
| Binarized | ripper_A | 0.158 |
| Binarized | ripper_B | 0.171 |
| Binarized | irep_A | 0.070 |
| Binarized | irep_B | 0.076 |
| Binarized | brl | 0.875 |
| Binarized | brs | 6.972 |
| Binarized | jrip | 0.750 |
| Binarized | part | 1.432 |
| Binarized | j48 | 0.616 |
| Original | tree | 0.004 |
| Original | forest | 0.205 |
| Original | ripper_A | 0.388 |
| Original | ripper_B | 0.408 |
| Original | irep_A | 0.057 |
| Original | irep_B | 0.056 |
| Original | brl | 0.851 |
| Original | brs | 7.350 |
| Original | jrip | 0.788 |
| Original | part | 1.387 |
| Original | j48 | 0.645 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.0 | 3.77 |
| Binarized | forest | 131.4 | 3.81 |
| Binarized | ripper_A | 6.6 | 2.33 |
| Binarized | ripper_B | 6.2 | 2.49 |
| Binarized | irep_A | 3.4 | 1.70 |
| Binarized | irep_B | 4.4 | 2.12 |
| Binarized | brl | 5.0 | 1.89 |
| Binarized | brs | 10.2 | 2.96 |
| Binarized | jrip | 5.2 | 2.30 |
| Binarized | part | 9.4 | 2.10 |
| Binarized | j48 | 11.4 | 3.88 |
| Original | tree | 12.2 | 3.69 |
| Original | forest | 123.4 | 3.74 |
| Original | ripper_A | 33.6 | 3.65 |
| Original | ripper_B | 31.2 | 3.85 |
| Original | irep_A | 11.4 | 2.74 |
| Original | irep_B | 9.4 | 3.03 |
| Original | brl | 5.0 | 1.89 |
| Original | brs | 10.2 | 2.96 |
| Original | jrip | 6.0 | 2.24 |
| Original | part | 7.4 | 2.11 |
| Original | j48 | 15.2 | 4.51 |

(119.0s total)

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
| Binarized | irep_B | 93.09 +/- 3.39 | 0/5 |
| Binarized | brl | 94.09 +/- 2.48 | 0/5 |
| Binarized | brs | 81.48 +/- 6.57 | 0/5 |
| Binarized | jrip | 99.22 +/- 0.33 | 0/5 |
| Binarized | part | 98.94 +/- 0.30 | 0/5 |
| Binarized | j48 | 99.28 +/- 0.38 | 0/5 |
| Original | tree | 94.09 +/- 1.11 | 0/5 |
| Original | forest | 94.12 +/- 1.11 | 0/5 |
| Original | ripper_A | 98.65 +/- 0.65 | 0/5 |
| Original | ripper_B | 98.84 +/- 0.34 | 0/5 |
| Original | irep_A | 97.72 +/- 0.72 | 0/5 |
| Original | irep_B | 93.80 +/- 3.42 | 0/5 |
| Original | brl | 94.09 +/- 2.48 | 0/5 |
| Original | brs | 81.48 +/- 6.57 | 0/5 |
| Original | jrip | 98.81 +/- 0.70 | 0/5 |
| Original | part | 98.97 +/- 0.47 | 0/5 |
| Original | j48 | 99.34 +/- 0.38 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.018 |
| Binarized | forest | 0.191 |
| Binarized | ripper_A | 0.637 |
| Binarized | ripper_B | 0.566 |
| Binarized | irep_A | 0.155 |
| Binarized | irep_B | 0.142 |
| Binarized | brl | 10.441 |
| Binarized | brs | 12.960 |
| Binarized | jrip | 0.916 |
| Binarized | part | 1.508 |
| Binarized | j48 | 0.546 |
| Original | tree | 0.008 |
| Original | forest | 0.300 |
| Original | ripper_A | 0.560 |
| Original | ripper_B | 0.503 |
| Original | irep_A | 0.175 |
| Original | irep_B | 0.177 |
| Original | brl | 10.474 |
| Original | brs | 12.920 |
| Original | jrip | 0.592 |
| Original | part | 1.315 |
| Original | j48 | 0.437 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 7.6 | 3.43 |
| Binarized | forest | 112.6 | 3.73 |
| Binarized | ripper_A | 17.4 | 3.23 |
| Binarized | ripper_B | 9.6 | 4.54 |
| Binarized | irep_A | 7.2 | 2.85 |
| Binarized | irep_B | 4.4 | 3.24 |
| Binarized | brl | 8.4 | 1.87 |
| Binarized | brs | 7.2 | 2.92 |
| Binarized | jrip | 13.6 | 3.00 |
| Binarized | part | 20.8 | 2.93 |
| Binarized | j48 | 25.8 | 7.37 |
| Original | tree | 7.6 | 3.43 |
| Original | forest | 118.6 | 3.75 |
| Original | ripper_A | 18.4 | 3.33 |
| Original | ripper_B | 10.8 | 4.80 |
| Original | irep_A | 7.8 | 2.84 |
| Original | irep_B | 5.4 | 4.08 |
| Original | brl | 8.4 | 1.87 |
| Original | brs | 7.2 | 2.92 |
| Original | jrip | 14.6 | 3.16 |
| Original | part | 20.4 | 3.21 |
| Original | j48 | 29.0 | 7.71 |

(283.0s total)

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
| Binarized | irep_B | 99.91 +/- 0.07 | 0/5 |
| Binarized | brl | 99.47 +/- 0.49 | 0/5 |
| Binarized | brs | 99.43 +/- 0.62 | 0/5 |
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
| Original | brs | 99.43 +/- 0.62 | 0/5 |
| Original | jrip | 100.00 +/- 0.00 | 0/5 |
| Original | part | 100.00 +/- 0.00 | 0/5 |
| Original | j48 | 100.00 +/- 0.00 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.062 |
| Binarized | forest | 0.334 |
| Binarized | ripper_A | 1.555 |
| Binarized | ripper_B | 1.512 |
| Binarized | irep_A | 0.797 |
| Binarized | irep_B | 0.905 |
| Binarized | brl | 14.250 |
| Binarized | brs | 24.160 |
| Binarized | jrip | 1.345 |
| Binarized | part | 1.841 |
| Binarized | j48 | 0.822 |
| Original | tree | 0.021 |
| Original | forest | 0.630 |
| Original | ripper_A | 0.427 |
| Original | ripper_B | 0.334 |
| Original | irep_A | 0.179 |
| Original | irep_B | 0.175 |
| Original | brl | 14.246 |
| Original | brs | 24.382 |
| Original | jrip | 0.573 |
| Original | part | 1.341 |
| Original | j48 | 0.431 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 10.4 | 3.55 |
| Binarized | forest | 86.8 | 3.39 |
| Binarized | ripper_A | 7.0 | 2.02 |
| Binarized | ripper_B | 7.0 | 1.56 |
| Binarized | irep_A | 3.4 | 1.80 |
| Binarized | irep_B | 5.4 | 1.82 |
| Binarized | brl | 10.2 | 1.74 |
| Binarized | brs | 8.4 | 2.76 |
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
| Original | brs | 8.4 | 2.76 |
| Original | jrip | 8.0 | 1.56 |
| Original | part | 10.0 | 1.85 |
| Original | j48 | 24.0 | 2.54 |

(471.1s total)

---

## Overall summary

### Accuracy

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 | fallback folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 89.43 | 95.63 | 93.56 | 95.63 | 93.56 | 94.25 | 94.48 | 88.05 | 95.40 | 95.17 | 96.09 | 94.94 | 94.71 | 95.17 | 95.17 | 94.25 | 95.63 | 94.48 | 88.05 | 95.63 | 95.63 | 96.32 | 0/5 |
| breast-cancer | 71.31 | 76.21 | 61.98 | 76.21 | 72.02 | 73.40 | 73.09 | 68.55 | 73.06 | 66.44 | 70.64 | 72.72 | 75.51 | 66.79 | 73.07 | 74.82 | 74.10 | 73.09 | 68.55 | 72.02 | 69.96 | 73.42 | 0/5 |
| colic | 73.11 | 80.15 | 81.24 | 85.35 | 83.98 | 85.33 | 84.79 | 80.17 | 86.42 | 81.81 | 83.43 | 84.25 | 84.26 | 81.27 | 86.96 | 84.25 | 85.87 | 84.79 | 80.17 | 87.51 | 85.88 | 82.63 | 0/5 |
| credit-approval | 84.35 | 84.06 | 86.09 | 85.94 | 85.51 | 84.20 | 84.20 | 61.16 | 85.65 | 83.04 | 84.64 | 84.78 | 85.51 | 83.62 | 83.04 | 85.51 | 83.48 | 84.20 | 61.16 | 85.94 | 84.35 | 85.94 | 0/5 |
| credit-g | 70.70 | 73.20 | 70.80 | 58.50 | 70.30 | 71.80 | 69.80 | 45.30 | 72.70 | 72.30 | 71.90 | 70.60 | 70.10 | 70.20 | 58.60 | 69.90 | 70.80 | 69.80 | 45.30 | 72.40 | 70.00 | 70.80 | 5/5 |
| diabetes | 61.60 | 74.48 | 66.53 | 73.96 | 74.74 | 75.65 | 73.70 | 69.40 | 74.61 | 74.22 | 74.99 | 73.83 | 74.09 | 59.62 | 70.98 | 69.66 | 73.44 | 73.70 | 69.40 | 75.14 | 71.75 | 71.74 | 0/5 |
| sonar | 70.64 | 77.90 | 70.24 | 75.52 | 72.14 | 70.22 | nan | nan | 70.19 | 74.53 | 74.47 | 75.48 | 78.37 | 62.11 | 64.44 | 55.30 | 63.90 | nan | nan | 72.10 | 75.96 | 71.56 | 0/5 |
| ionosphere | 90.03 | 91.74 | 89.75 | 87.48 | 85.46 | 87.75 | nan | nan | 89.18 | 91.74 | 90.31 | 88.61 | 91.45 | 88.04 | 75.51 | 84.05 | 74.93 | nan | nan | 90.89 | 90.62 | 88.33 | 4/5 |
| tic-tac-toe | 82.36 | 77.04 | 98.02 | 99.48 | 94.46 | 80.80 | 80.58 | 85.91 | 98.33 | 93.84 | 94.88 | 82.36 | 76.83 | 97.91 | 97.39 | 89.35 | 82.89 | 80.58 | 85.91 | 97.91 | 94.15 | 87.27 | 0/5 |
| banknote-authentication | 95.77 | 96.35 | 98.54 | 98.18 | 94.61 | 96.50 | 97.16 | 96.72 | 98.61 | 98.32 | 98.54 | 95.70 | 97.01 | 97.09 | 95.63 | 91.76 | 90.16 | 97.16 | 96.72 | 97.89 | 98.76 | 98.61 | 0/5 |
| kr-vs-kp | 94.09 | 94.99 | 98.94 | 98.75 | 98.22 | 93.09 | 94.09 | 81.48 | 99.22 | 98.94 | 99.28 | 94.09 | 94.12 | 98.65 | 98.84 | 97.72 | 93.80 | 94.09 | 81.48 | 98.81 | 98.97 | 99.34 | 0/5 |
| mushroom | 78.08 | 98.98 | 98.18 | 99.95 | 98.52 | 99.91 | 99.47 | 99.43 | 99.98 | 100.00 | 100.00 | 99.22 | 98.66 | 100.00 | 99.98 | 98.52 | 96.45 | 99.47 | 99.43 | 100.00 | 100.00 | 100.00 | 0/5 |

### Fit time (seconds/fold)

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.191 | 0.051 | 0.052 | 0.047 | 0.029 | 0.028 | 0.547 | 7.176 | 0.436 | 1.265 | 0.398 | 0.003 | 0.060 | 0.067 | 0.059 | 0.042 | 0.036 | 0.481 | 6.946 | 1.221 | 0.374 | 1.246 |
| breast-cancer | 0.006 | 0.039 | 0.097 | 0.119 | 0.058 | 0.056 | 0.257 | 7.022 | 0.430 | 1.320 | 0.418 | 0.003 | 0.048 | 0.055 | 0.082 | 0.029 | 0.028 | 0.242 | 6.929 | 1.213 | 0.366 | 1.227 |
| colic | 0.010 | 0.048 | 0.400 | 0.425 | 0.280 | 0.281 | 2.278 | 31.069 | 0.553 | 1.411 | 0.472 | 0.004 | 0.067 | 0.251 | 0.242 | 0.176 | 0.162 | 2.256 | 31.108 | 0.393 | 1.262 | 0.366 |
| credit-approval | 0.011 | 0.080 | 0.351 | 0.349 | 0.152 | 0.162 | 1.579 | 10.222 | 0.631 | 1.436 | 0.525 | 0.004 | 0.109 | 0.251 | 0.255 | 0.089 | 0.088 | 1.525 | 10.142 | 0.576 | 1.327 | 0.526 |
| credit-g | 0.014 | 0.102 | 0.534 | 0.465 | 0.186 | 0.192 | 2.785 | 11.633 | 0.849 | 1.782 | 0.785 | 0.005 | 0.154 | 0.297 | 0.244 | 0.089 | 0.095 | 2.808 | 11.767 | 1.108 | 1.563 | 0.708 |
| diabetes | 0.010 | 0.082 | 0.297 | 0.313 | 0.106 | 0.121 | 1.133 | 9.509 | 0.759 | 1.423 | 0.812 | 0.003 | 0.151 | 0.215 | 0.231 | 0.087 | 0.064 | 1.149 | 9.436 | 1.033 | 1.452 | 0.652 |
| sonar | 0.013 | 0.045 | 1.057 | 1.005 | 0.839 | 0.825 | nan | nan | 0.967 | 1.403 | 0.826 | 0.006 | 0.083 | 0.995 | 0.953 | 0.833 | 0.818 | nan | nan | 0.436 | 1.517 | 0.659 |
| ionosphere | 0.010 | 0.042 | 0.534 | 0.508 | 0.395 | 0.395 | nan | nan | 0.653 | 1.411 | 0.837 | 0.006 | 0.081 | 0.483 | 0.431 | 0.335 | 0.320 | nan | nan | 0.628 | 1.462 | 0.656 |
| tic-tac-toe | 0.012 | 0.100 | 0.175 | 0.179 | 0.079 | 0.062 | 0.905 | 6.819 | 0.632 | 1.393 | 0.724 | 0.003 | 0.117 | 0.111 | 0.140 | 0.038 | 0.036 | 0.823 | 6.907 | 0.917 | 1.457 | 0.713 |
| banknote-authentication | 0.013 | 0.110 | 0.158 | 0.171 | 0.070 | 0.076 | 0.875 | 6.972 | 0.750 | 1.432 | 0.616 | 0.004 | 0.205 | 0.388 | 0.408 | 0.057 | 0.056 | 0.851 | 7.350 | 0.788 | 1.387 | 0.645 |
| kr-vs-kp | 0.018 | 0.191 | 0.637 | 0.566 | 0.155 | 0.142 | 10.441 | 12.960 | 0.916 | 1.508 | 0.546 | 0.008 | 0.300 | 0.560 | 0.503 | 0.175 | 0.177 | 10.474 | 12.920 | 0.592 | 1.315 | 0.437 |
| mushroom | 0.062 | 0.334 | 1.555 | 1.512 | 0.797 | 0.905 | 14.250 | 24.160 | 1.345 | 1.841 | 0.822 | 0.021 | 0.630 | 0.427 | 0.334 | 0.179 | 0.175 | 14.246 | 24.382 | 0.573 | 1.341 | 0.431 |

### Rule count

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 12.8 | 126.4 | 2.4 | 3.0 | 2.4 | 1.4 | 3.2 | 7.8 | 2.2 | 6.4 | 6.6 | 13.4 | 121.0 | 3.4 | 3.8 | 2.8 | 1.0 | 3.2 | 7.8 | 1.2 | 5.2 | 9.0 |
| breast-cancer | 12.4 | 128.0 | 1.4 | 1.4 | 2.4 | 1.0 | 3.2 | 7.0 | 1.0 | 21.6 | 13.8 | 12.4 | 124.8 | 3.0 | 1.6 | 3.2 | 1.6 | 3.2 | 7.0 | 2.8 | 15.4 | 8.4 |
| colic | 11.8 | 122.0 | 3.6 | 3.8 | 2.2 | 1.4 | 4.8 | 7.8 | 3.2 | 11.6 | 17.0 | 11.6 | 118.6 | 6.0 | 6.2 | 3.6 | 1.4 | 4.8 | 7.8 | 3.4 | 12.8 | 8.6 |
| credit-approval | 14.4 | 138.8 | 5.0 | 3.6 | 1.4 | 2.2 | 4.8 | 8.6 | 3.2 | 23.6 | 23.0 | 14.2 | 134.0 | 7.6 | 8.8 | 1.4 | 2.4 | 4.8 | 8.6 | 3.8 | 29.6 | 19.4 |
| credit-g | 15.0 | 145.0 | 3.2 | 3.8 | 1.8 | 4.0 | 4.8 | 8.6 | 3.0 | 56.8 | 78.8 | 15.2 | 141.8 | 5.4 | 5.2 | 2.2 | 5.0 | 4.8 | 8.6 | 3.4 | 60.4 | 77.8 |
| diabetes | 14.6 | 145.4 | 4.4 | 3.8 | 3.0 | 4.0 | 4.6 | 7.4 | 3.0 | 46.2 | 42.2 | 14.6 | 140.4 | 8.2 | 8.2 | 8.4 | 1.8 | 4.6 | 7.4 | 2.4 | 6.8 | 22.0 |
| sonar | 13.6 | 121.4 | 3.0 | 3.2 | 1.8 | 2.8 | nan | nan | 3.4 | 6.2 | 16.0 | 13.0 | 114.8 | 4.4 | 3.4 | 2.6 | 3.4 | nan | nan | 3.6 | 6.4 | 14.0 |
| ionosphere | 8.6 | 98.6 | 5.6 | 3.0 | 2.2 | 2.4 | nan | nan | 4.2 | 6.0 | 10.8 | 8.8 | 95.4 | 8.0 | 9.0 | 3.0 | 1.0 | nan | nan | 4.2 | 5.6 | 11.4 |
| tic-tac-toe | 14.0 | 153.4 | 8.6 | 9.8 | 7.8 | 5.8 | 11.8 | 8.2 | 8.0 | 29.2 | 37.8 | 14.0 | 157.4 | 9.4 | 14.0 | 6.4 | 6.0 | 11.8 | 8.2 | 9.6 | 36.4 | 80.6 |
| banknote-authentication | 13.0 | 131.4 | 6.6 | 6.2 | 3.4 | 4.4 | 5.0 | 10.2 | 5.2 | 9.4 | 11.4 | 12.2 | 123.4 | 33.6 | 31.2 | 11.4 | 9.4 | 5.0 | 10.2 | 6.0 | 7.4 | 15.2 |
| kr-vs-kp | 7.6 | 112.6 | 17.4 | 9.6 | 7.2 | 4.4 | 8.4 | 7.2 | 13.6 | 20.8 | 25.8 | 7.6 | 118.6 | 18.4 | 10.8 | 7.8 | 5.4 | 8.4 | 7.2 | 14.6 | 20.4 | 29.0 |
| mushroom | 10.4 | 86.8 | 7.0 | 7.0 | 3.4 | 5.4 | 10.2 | 8.4 | 5.8 | 5.0 | 9.8 | 10.0 | 101.8 | 7.6 | 8.2 | 4.0 | 4.0 | 10.2 | 8.4 | 8.0 | 10.0 | 24.0 |

### Average conditions per rule

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 3.59 | 3.73 | 1.73 | 2.33 | 1.47 | 1.07 | 1.25 | 2.98 | 1.87 | 2.09 | 3.34 | 3.80 | 3.73 | 1.88 | 1.95 | 1.73 | 1.00 | 1.25 | 2.98 | 1.20 | 1.42 | 2.61 |
| breast-cancer | 3.77 | 3.79 | 1.70 | 2.50 | 1.27 | 1.80 | 1.42 | 2.86 | 1.80 | 4.48 | 5.22 | 3.75 | 3.78 | 1.55 | 2.70 | 1.17 | 1.90 | 1.42 | 2.86 | 2.22 | 1.92 | 1.77 |
| colic | 3.72 | 3.74 | 2.25 | 1.83 | 1.17 | 1.70 | 1.55 | 2.84 | 1.97 | 5.10 | 6.19 | 3.72 | 3.73 | 1.61 | 2.13 | 1.22 | 1.80 | 1.55 | 2.84 | 1.83 | 1.81 | 2.17 |
| credit-approval | 3.87 | 3.87 | 2.53 | 2.52 | 1.30 | 1.85 | 1.58 | 2.94 | 2.05 | 4.06 | 7.23 | 3.87 | 3.84 | 2.86 | 3.22 | 1.10 | 1.57 | 1.58 | 2.94 | 2.18 | 2.37 | 4.02 |
| credit-g | 3.93 | 3.90 | 4.83 | 2.95 | 2.90 | 1.34 | 1.67 | 2.86 | 3.77 | 4.09 | 11.42 | 3.95 | 3.89 | 3.64 | 2.68 | 2.62 | 1.13 | 1.67 | 2.86 | 2.69 | 3.02 | 5.57 |
| diabetes | 3.90 | 3.91 | 2.71 | 3.74 | 1.52 | 3.16 | 1.79 | 2.92 | 2.68 | 4.08 | 8.96 | 3.90 | 3.88 | 2.46 | 2.93 | 1.98 | 1.53 | 1.79 | 2.92 | 2.40 | 2.51 | 6.13 |
| sonar | 3.82 | 3.73 | 2.25 | 2.12 | 1.50 | 1.80 | nan | nan | 1.74 | 3.59 | 5.73 | 3.76 | 3.67 | 1.66 | 2.59 | 1.13 | 1.75 | nan | nan | 1.73 | 2.41 | 4.51 |
| ionosphere | 3.46 | 3.57 | 1.55 | 3.20 | 1.37 | 2.53 | nan | nan | 1.47 | 3.09 | 4.64 | 3.49 | 3.51 | 2.07 | 2.47 | 1.33 | 1.00 | nan | nan | 1.31 | 2.83 | 5.04 |
| tic-tac-toe | 3.86 | 3.96 | 3.07 | 3.22 | 3.01 | 2.63 | 1.81 | 3.00 | 3.08 | 3.23 | 5.88 | 3.86 | 3.98 | 3.28 | 3.13 | 2.78 | 2.09 | 1.81 | 3.00 | 3.28 | 2.69 | 4.53 |
| banknote-authentication | 3.77 | 3.81 | 2.33 | 2.49 | 1.70 | 2.12 | 1.89 | 2.96 | 2.30 | 2.10 | 3.88 | 3.69 | 3.74 | 3.65 | 3.85 | 2.74 | 3.03 | 1.89 | 2.96 | 2.24 | 2.11 | 4.51 |
| kr-vs-kp | 3.43 | 3.73 | 3.23 | 4.54 | 2.85 | 3.24 | 1.87 | 2.92 | 3.00 | 2.93 | 7.37 | 3.43 | 3.75 | 3.33 | 4.80 | 2.84 | 4.08 | 1.87 | 2.92 | 3.16 | 3.21 | 7.71 |
| mushroom | 3.55 | 3.39 | 2.02 | 1.56 | 1.80 | 1.82 | 1.74 | 2.76 | 1.64 | 2.76 | 3.83 | 3.50 | 3.58 | 2.54 | 1.50 | 1.50 | 1.00 | 1.74 | 2.76 | 1.56 | 1.85 | 2.54 |
## Overview evaluation (across all datasets)

**Average performance across datasets**

| algorithm | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |
|---|---|---|---|---|---|
| Binarized/brl | 85.14 | 3.505 | 6.1 | 1.66 | 2 |
| Original/brl | 85.14 | 3.486 | 6.1 | 1.66 | 2 |
| Binarized/brs | 77.62 | 12.754 | 8.1 | 2.90 | 2 |
| Original/brs | 77.62 | 12.789 | 8.1 | 2.90 | 2 |
| Binarized/forest | 85.06 | 0.102 | 125.8 | 3.76 | 0 |
| Original/forest | 85.05 | 0.167 | 124.3 | 3.76 | 0 |
| Binarized/irep_A | 85.29 | 0.262 | 3.2 | 1.82 | 0 |
| Original/irep_A | 82.92 | 0.177 | 4.7 | 1.85 | 0 |
| Binarized/irep_B | 84.41 | 0.270 | 3.3 | 2.09 | 0 |
| Original/irep_B | 82.12 | 0.171 | 3.5 | 1.82 | 0 |
| Binarized/j48 | 86.60 | 0.648 | 24.4 | 6.14 | 0 |
| Original/j48 | 85.50 | 0.689 | 26.6 | 4.26 | 0 |
| Binarized/jrip | 86.94 | 0.743 | 4.7 | 2.28 | 0 |
| Original/jrip | 87.19 | 0.790 | 5.2 | 2.15 | 0 |
| Binarized/part | 85.86 | 1.469 | 20.2 | 3.47 | 0 |
| Original/part | 86.33 | 1.235 | 18.0 | 2.35 | 0 |
| Binarized/ripper_A | 84.49 | 0.487 | 5.7 | 2.52 | 0 |
| Original/ripper_A | 83.37 | 0.342 | 9.6 | 2.54 | 0 |
| Binarized/ripper_B | 86.25 | 0.472 | 4.9 | 2.75 | 0 |
| Original/ripper_B | 83.30 | 0.323 | 9.2 | 2.83 | 0 |
| Binarized/tree | 80.12 | 0.031 | 12.3 | 3.72 | 0 |
| Original/tree | 84.72 | 0.006 | 12.2 | 3.73 | 0 |

**Average rank per criterion** (1 = best of 22; failed entries tie for last)

| algorithm | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |
|---|---|---|---|---|
| Binarized/brl | 14.29 | 18.50 | 11.62 | 6.42 |
| Original/brl | 14.29 | 18.17 | 11.62 | 6.42 |
| Binarized/brs | 18.67 | 21.33 | 13.12 | 13.67 |
| Original/brs | 18.67 | 21.33 | 13.12 | 13.67 |
| Binarized/forest | 10.04 | 5.00 | 21.08 | 17.83 |
| Original/forest | 10.33 | 7.08 | 20.58 | 17.42 |
| Binarized/irep_A | 12.38 | 7.17 | 2.88 | 4.33 |
| Original/irep_A | 14.08 | 5.08 | 6.46 | 4.17 |
| Binarized/irep_B | 11.71 | 7.08 | 3.08 | 7.21 |
| Original/irep_B | 13.75 | 4.17 | 4.00 | 5.58 |
| Binarized/j48 | 6.62 | 13.58 | 17.29 | 20.75 |
| Original/j48 | 7.38 | 12.92 | 17.58 | 17.58 |
| Binarized/jrip | 5.96 | 14.92 | 5.29 | 8.54 |
| Original/jrip | 5.42 | 14.17 | 7.04 | 7.25 |
| Binarized/part | 9.17 | 18.50 | 14.75 | 14.83 |
| Original/part | 7.67 | 17.08 | 14.92 | 8.75 |
| Binarized/ripper_A | 11.79 | 12.17 | 7.29 | 10.50 |
| Original/ripper_A | 13.04 | 9.67 | 11.88 | 9.92 |
| Binarized/ripper_B | 7.33 | 11.83 | 6.88 | 11.67 |
| Original/ripper_B | 12.29 | 9.42 | 12.17 | 12.08 |
| Binarized/tree | 15.88 | 2.83 | 15.38 | 17.21 |
| Original/tree | 12.25 | 1.00 | 14.96 | 17.21 |

**Binarized vs. Original: how often each was better, per model** (ties count 0.5 each per side; a dataset where both failed isn't counted for either side)

| model | accuracy (Bin / Orig) | fit time (Bin / Orig) | n_rules (Bin / Orig) | avg_conditions (Bin / Orig) |
|---|---|---|---|---|
| brl | 5.0 / 5.0 | 3.0 / 7.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| brs | 5.0 / 5.0 | 5.0 / 5.0 | 5.0 / 5.0 | 5.0 / 5.0 |
| forest | 8.0 / 4.0 | 12.0 / 0.0 | 3.0 / 9.0 | 4.0 / 8.0 |
| irep_A | 8.0 / 4.0 | 2.0 / 10.0 | 10.5 / 1.5 | 4.0 / 8.0 |
| irep_B | 7.0 / 5.0 | 2.0 / 10.0 | 7.5 / 4.5 | 4.0 / 8.0 |
| j48 | 6.5 / 5.5 | 4.0 / 8.0 | 6.0 / 6.0 | 3.0 / 9.0 |
| jrip | 5.0 / 7.0 | 6.0 / 6.0 | 9.5 / 2.5 | 4.0 / 8.0 |
| part | 3.5 / 8.5 | 4.0 / 8.0 | 6.0 / 6.0 | 2.0 / 10.0 |
| ripper_A | 8.0 / 4.0 | 2.0 / 10.0 | 12.0 / 0.0 | 7.0 / 5.0 |
| ripper_B | 8.0 / 4.0 | 2.0 / 10.0 | 12.0 / 0.0 | 6.0 / 6.0 |
| tree | 4.0 / 8.0 | 0.0 / 12.0 | 5.0 / 7.0 | 5.5 / 6.5 |
