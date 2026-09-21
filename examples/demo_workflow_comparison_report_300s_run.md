# Binarized vs. Original data preparation -- comparison across binary datasets

Generated 2026-08-21 11:35:52. N_FOLDS=5, MAX_INTERVALS=8 ('Binarized' mode only), MAX_DEPTH=4, RIPPER_K=2, N_ESTIMATORS=10 (forest), FIT_TIMEOUT_SECONDS=300. `ripper_A`/`irep_A` treat each dataset's (alphabetically) first class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit the same already-Boolean matrix under both workflows as two independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` fit-time includes JVM subprocess startup overhead, not just the algorithm itself -- see the module docstring.

---

## vote

n=435, attributes=16, A='democrat', B='republican'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.949 +/- 0.012 | 0/5 |
| Binarized | forest | 0.947 +/- 0.021 | 0/5 |
| Binarized | ripper_A | 0.940 +/- 0.018 | 0/5 |
| Binarized | ripper_B | 0.961 +/- 0.012 | 0/5 |
| Binarized | irep_A | 0.945 +/- 0.022 | 0/5 |
| Binarized | irep_B | 0.947 +/- 0.027 | 0/5 |
| Binarized | brl | 0.940 +/- 0.017 | 0/5 |
| Binarized | brs | 0.903 +/- 0.042 | 0/5 |
| Binarized | jrip | 0.949 +/- 0.017 | 0/5 |
| Binarized | part | 0.954 +/- 0.013 | 0/5 |
| Binarized | j48 | 0.961 +/- 0.012 | 0/5 |
| Original | tree | 0.949 +/- 0.012 | 0/5 |
| Original | forest | 0.947 +/- 0.021 | 0/5 |
| Original | ripper_A | 0.952 +/- 0.011 | 0/5 |
| Original | ripper_B | 0.952 +/- 0.015 | 0/5 |
| Original | irep_A | 0.949 +/- 0.024 | 0/5 |
| Original | irep_B | 0.956 +/- 0.023 | 0/5 |
| Original | brl | 0.940 +/- 0.017 | 0/5 |
| Original | brs | 0.903 +/- 0.042 | 0/5 |
| Original | jrip | 0.956 +/- 0.023 | 0/5 |
| Original | part | 0.956 +/- 0.013 | 0/5 |
| Original | j48 | 0.963 +/- 0.013 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.309 |
| Binarized | forest | 0.022 |
| Binarized | ripper_A | 0.056 |
| Binarized | ripper_B | 0.053 |
| Binarized | irep_A | 0.034 |
| Binarized | irep_B | 0.034 |
| Binarized | brl | 0.623 |
| Binarized | brs | 6.891 |
| Binarized | jrip | 0.305 |
| Binarized | part | 0.269 |
| Binarized | j48 | 0.257 |
| Original | tree | 0.003 |
| Original | forest | 0.016 |
| Original | ripper_A | 0.053 |
| Original | ripper_B | 0.050 |
| Original | irep_A | 0.031 |
| Original | irep_B | 0.031 |
| Original | brl | 0.534 |
| Original | brs | 6.805 |
| Original | jrip | 0.229 |
| Original | part | 0.223 |
| Original | j48 | 0.226 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.4 | 3.80 |
| Binarized | forest | 121.0 | 3.73 |
| Binarized | ripper_A | 2.0 | 1.53 |
| Binarized | ripper_B | 3.0 | 2.25 |
| Binarized | irep_A | 2.0 | 1.30 |
| Binarized | irep_B | 1.4 | 1.20 |
| Binarized | brl | 3.8 | 1.48 |
| Binarized | brs | 7.8 | 3.00 |
| Binarized | jrip | 2.2 | 1.67 |
| Binarized | part | 5.2 | 2.23 |
| Binarized | j48 | 6.8 | 3.32 |
| Original | tree | 13.4 | 3.80 |
| Original | forest | 121.0 | 3.73 |
| Original | ripper_A | 3.4 | 1.88 |
| Original | ripper_B | 3.8 | 1.95 |
| Original | irep_A | 2.0 | 1.50 |
| Original | irep_B | 1.0 | 1.00 |
| Original | brl | 3.8 | 1.48 |
| Original | brs | 7.8 | 3.00 |
| Original | jrip | 1.2 | 1.20 |
| Original | part | 5.2 | 1.42 |
| Original | j48 | 9.0 | 2.61 |

(85.8s total)

---

## breast-cancer

n=286, attributes=9, A='no-recurrence-events', B='recurrence-events'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.727 +/- 0.019 | 0/5 |
| Binarized | forest | 0.755 +/- 0.020 | 0/5 |
| Binarized | ripper_A | 0.620 +/- 0.143 | 0/5 |
| Binarized | ripper_B | 0.738 +/- 0.025 | 0/5 |
| Binarized | irep_A | 0.734 +/- 0.029 | 0/5 |
| Binarized | irep_B | 0.724 +/- 0.053 | 0/5 |
| Binarized | brl | 0.731 +/- 0.023 | 0/5 |
| Binarized | brs | 0.720 +/- 0.038 | 0/5 |
| Binarized | jrip | 0.741 +/- 0.022 | 0/5 |
| Binarized | part | 0.671 +/- 0.041 | 0/5 |
| Binarized | j48 | 0.699 +/- 0.028 | 0/5 |
| Original | tree | 0.727 +/- 0.019 | 0/5 |
| Original | forest | 0.755 +/- 0.020 | 0/5 |
| Original | ripper_A | 0.668 +/- 0.028 | 0/5 |
| Original | ripper_B | 0.731 +/- 0.022 | 0/5 |
| Original | irep_A | 0.755 +/- 0.032 | 0/5 |
| Original | irep_B | 0.727 +/- 0.012 | 0/5 |
| Original | brl | 0.731 +/- 0.023 | 0/5 |
| Original | brs | 0.720 +/- 0.038 | 0/5 |
| Original | jrip | 0.720 +/- 0.036 | 0/5 |
| Original | part | 0.700 +/- 0.056 | 0/5 |
| Original | j48 | 0.734 +/- 0.015 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.003 |
| Binarized | forest | 0.017 |
| Binarized | ripper_A | 0.062 |
| Binarized | ripper_B | 0.096 |
| Binarized | irep_A | 0.031 |
| Binarized | irep_B | 0.031 |
| Binarized | brl | 0.238 |
| Binarized | brs | 6.992 |
| Binarized | jrip | 0.287 |
| Binarized | part | 0.312 |
| Binarized | j48 | 0.275 |
| Original | tree | 0.003 |
| Original | forest | 0.012 |
| Original | ripper_A | 0.049 |
| Original | ripper_B | 0.080 |
| Original | irep_A | 0.036 |
| Original | irep_B | 0.027 |
| Original | brl | 0.227 |
| Original | brs | 6.951 |
| Original | jrip | 0.264 |
| Original | part | 0.250 |
| Original | j48 | 0.449 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 12.4 | 3.75 |
| Binarized | forest | 124.8 | 3.78 |
| Binarized | ripper_A | 1.6 | 1.60 |
| Binarized | ripper_B | 1.6 | 2.70 |
| Binarized | irep_A | 2.2 | 1.67 |
| Binarized | irep_B | 1.6 | 2.00 |
| Binarized | brl | 3.2 | 1.42 |
| Binarized | brs | 7.6 | 3.00 |
| Binarized | jrip | 1.2 | 1.80 |
| Binarized | part | 22.0 | 4.54 |
| Binarized | j48 | 15.0 | 5.37 |
| Original | tree | 12.4 | 3.75 |
| Original | forest | 124.8 | 3.78 |
| Original | ripper_A | 3.0 | 1.55 |
| Original | ripper_B | 1.6 | 2.70 |
| Original | irep_A | 2.4 | 1.07 |
| Original | irep_B | 1.8 | 1.53 |
| Original | brl | 3.2 | 1.42 |
| Original | brs | 7.6 | 3.00 |
| Original | jrip | 2.8 | 2.22 |
| Original | part | 15.4 | 1.92 |
| Original | j48 | 8.4 | 1.77 |

(84.0s total)

---

## colic

n=368, attributes=26, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.851 +/- 0.027 | 0/5 |
| Binarized | forest | 0.850 +/- 0.015 | 0/5 |
| Binarized | ripper_A | 0.856 +/- 0.030 | 0/5 |
| Binarized | ripper_B | 0.851 +/- 0.028 | 0/5 |
| Binarized | irep_A | 0.845 +/- 0.018 | 0/5 |
| Binarized | irep_B | 0.837 +/- 0.035 | 0/5 |
| Binarized | brl | 0.848 +/- 0.010 | 0/5 |
| Binarized | brs | 0.799 +/- 0.037 | 0/5 |
| Binarized | jrip | 0.848 +/- 0.020 | 0/5 |
| Binarized | part | 0.807 +/- 0.048 | 0/5 |
| Binarized | j48 | 0.856 +/- 0.026 | 0/5 |
| Original | tree | 0.843 +/- 0.031 | 0/5 |
| Original | forest | 0.837 +/- 0.038 | 0/5 |
| Original | ripper_A | 0.813 +/- 0.034 | 0/5 |
| Original | ripper_B | 0.870 +/- 0.029 | 0/5 |
| Original | irep_A | 0.826 +/- 0.022 | 0/5 |
| Original | irep_B | 0.851 +/- 0.030 | 0/5 |
| Original | brl | 0.848 +/- 0.010 | 0/5 |
| Original | brs | 0.799 +/- 0.037 | 0/5 |
| Original | jrip | 0.875 +/- 0.033 | 0/5 |
| Original | part | 0.859 +/- 0.025 | 0/5 |
| Original | j48 | 0.826 +/- 0.052 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.010 |
| Binarized | forest | 0.021 |
| Binarized | ripper_A | 0.200 |
| Binarized | ripper_B | 0.208 |
| Binarized | irep_A | 0.123 |
| Binarized | irep_B | 0.122 |
| Binarized | brl | 3.895 |
| Binarized | brs | 29.102 |
| Binarized | jrip | 0.396 |
| Binarized | part | 0.372 |
| Binarized | j48 | 0.364 |
| Original | tree | 0.004 |
| Original | forest | 0.012 |
| Original | ripper_A | 0.198 |
| Original | ripper_B | 0.212 |
| Original | irep_A | 0.142 |
| Original | irep_B | 0.151 |
| Original | brl | 3.720 |
| Original | brs | 894.841 |
| Original | jrip | 0.308 |
| Original | part | 0.283 |
| Original | j48 | 0.270 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 11.4 | 3.68 |
| Binarized | forest | 122.6 | 3.75 |
| Binarized | ripper_A | 4.6 | 2.07 |
| Binarized | ripper_B | 5.4 | 1.88 |
| Binarized | irep_A | 2.0 | 1.20 |
| Binarized | irep_B | 2.0 | 1.37 |
| Binarized | brl | 4.8 | 1.46 |
| Binarized | brs | 7.8 | 2.86 |
| Binarized | jrip | 3.4 | 1.85 |
| Binarized | part | 13.6 | 4.53 |
| Binarized | j48 | 15.6 | 5.97 |
| Original | tree | 11.6 | 3.72 |
| Original | forest | 118.6 | 3.73 |
| Original | ripper_A | 6.0 | 1.61 |
| Original | ripper_B | 6.2 | 2.13 |
| Original | irep_A | 3.2 | 1.21 |
| Original | irep_B | 1.4 | 1.70 |
| Original | brl | 4.8 | 1.46 |
| Original | brs | 7.8 | 2.86 |
| Original | jrip | 3.4 | 1.83 |
| Original | part | 12.8 | 1.81 |
| Original | j48 | 8.6 | 2.17 |

(4676.3s total)

---

## credit-approval

n=690, attributes=15, A='+', B='-'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.849 +/- 0.033 | 0/5 |
| Binarized | forest | 0.857 +/- 0.026 | 0/5 |
| Binarized | ripper_A | 0.849 +/- 0.026 | 0/5 |
| Binarized | ripper_B | 0.857 +/- 0.026 | 0/5 |
| Binarized | irep_A | 0.852 +/- 0.029 | 0/5 |
| Binarized | irep_B | 0.852 +/- 0.028 | 0/5 |
| Binarized | brl | 0.858 +/- 0.034 | 0/5 |
| Binarized | brs | 0.625 +/- 0.053 | 0/5 |
| Binarized | jrip | 0.865 +/- 0.030 | 0/5 |
| Binarized | part | 0.839 +/- 0.033 | 0/5 |
| Binarized | j48 | 0.852 +/- 0.033 | 0/5 |
| Original | tree | 0.848 +/- 0.039 | 0/5 |
| Original | forest | 0.857 +/- 0.033 | 0/5 |
| Original | ripper_A | 0.836 +/- 0.036 | 0/5 |
| Original | ripper_B | 0.830 +/- 0.036 | 0/5 |
| Original | irep_A | 0.855 +/- 0.033 | 0/5 |
| Original | irep_B | 0.859 +/- 0.036 | 0/5 |
| Original | brl | 0.858 +/- 0.034 | 0/5 |
| Original | brs | 0.625 +/- 0.053 | 0/5 |
| Original | jrip | 0.859 +/- 0.026 | 0/5 |
| Original | part | 0.843 +/- 0.035 | 0/5 |
| Original | j48 | 0.859 +/- 0.041 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.011 |
| Binarized | forest | 0.022 |
| Binarized | ripper_A | 0.264 |
| Binarized | ripper_B | 0.227 |
| Binarized | irep_A | 0.070 |
| Binarized | irep_B | 0.084 |
| Binarized | brl | 1.780 |
| Binarized | brs | 11.084 |
| Binarized | jrip | 0.479 |
| Binarized | part | 0.404 |
| Binarized | j48 | 0.412 |
| Original | tree | 0.004 |
| Original | forest | 0.014 |
| Original | ripper_A | 0.264 |
| Original | ripper_B | 0.269 |
| Original | irep_A | 0.095 |
| Original | irep_B | 0.084 |
| Original | brl | 1.774 |
| Original | brs | 10.975 |
| Original | jrip | 0.423 |
| Original | part | 0.311 |
| Original | j48 | 0.371 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.8 | 3.85 |
| Binarized | forest | 136.4 | 3.85 |
| Binarized | ripper_A | 5.8 | 2.71 |
| Binarized | ripper_B | 4.0 | 2.57 |
| Binarized | irep_A | 1.2 | 1.10 |
| Binarized | irep_B | 1.8 | 1.67 |
| Binarized | brl | 4.4 | 1.66 |
| Binarized | brs | 7.8 | 2.83 |
| Binarized | jrip | 4.4 | 2.40 |
| Binarized | part | 26.8 | 3.49 |
| Binarized | j48 | 17.0 | 6.31 |
| Original | tree | 14.2 | 3.87 |
| Original | forest | 134.0 | 3.84 |
| Original | ripper_A | 7.6 | 2.86 |
| Original | ripper_B | 8.8 | 3.22 |
| Original | irep_A | 1.4 | 1.30 |
| Original | irep_B | 1.8 | 1.57 |
| Original | brl | 4.4 | 1.66 |
| Original | brs | 7.8 | 2.83 |
| Original | jrip | 3.8 | 2.18 |
| Original | part | 29.6 | 2.37 |
| Original | j48 | 19.4 | 4.02 |

(148.4s total)

---

## credit-g

n=1000, attributes=20, A='bad', B='good'

*5/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.705 +/- 0.011 | 0/5 |
| Binarized | forest | 0.732 +/- 0.011 | 0/5 |
| Binarized | ripper_A | 0.712 +/- 0.032 | 0/5 |
| Binarized | ripper_B | 0.597 +/- 0.067 | 0/5 |
| Binarized | irep_A | 0.725 +/- 0.004 | 0/5 |
| Binarized | irep_B | 0.713 +/- 0.030 | 0/5 |
| Binarized | brl | 0.698 +/- 0.022 | 0/5 |
| Binarized | brs | 0.436 +/- 0.027 | 0/5 |
| Binarized | jrip | 0.718 +/- 0.020 | 0/5 |
| Binarized | part | 0.723 +/- 0.032 | 0/5 |
| Binarized | j48 | 0.719 +/- 0.012 | 0/5 |
| Original | tree | 0.706 +/- 0.026 | 0/5 |
| Original | forest | 0.705 +/- 0.013 | 0/5 |
| Original | ripper_A | 0.702 +/- 0.010 | 0/5 |
| Original | ripper_B | 0.586 +/- 0.051 | 0/5 |
| Original | irep_A | 0.701 +/- 0.007 | 0/5 |
| Original | irep_B | 0.698 +/- 0.021 | 0/5 |
| Original | brl | 0.698 +/- 0.022 | 0/5 |
| Original | brs | 0.436 +/- 0.027 | 0/5 |
| Original | jrip | 0.724 +/- 0.024 | 0/5 |
| Original | part | 0.700 +/- 0.034 | 0/5 |
| Original | j48 | 0.708 +/- 0.019 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.025 |
| Binarized | ripper_A | 0.466 |
| Binarized | ripper_B | 0.324 |
| Binarized | irep_A | 0.103 |
| Binarized | irep_B | 0.098 |
| Binarized | brl | 3.122 |
| Binarized | brs | 11.381 |
| Binarized | jrip | 0.552 |
| Binarized | part | 0.556 |
| Binarized | j48 | 0.510 |
| Original | tree | 0.005 |
| Original | forest | 0.023 |
| Original | ripper_A | 0.342 |
| Original | ripper_B | 0.300 |
| Original | irep_A | 0.096 |
| Original | irep_B | 0.107 |
| Original | brl | 3.118 |
| Original | brs | 11.431 |
| Original | jrip | 0.445 |
| Original | part | 0.388 |
| Original | j48 | 0.344 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.0 | 3.93 |
| Binarized | forest | 143.6 | 3.90 |
| Binarized | ripper_A | 3.8 | 4.87 |
| Binarized | ripper_B | 4.2 | 2.86 |
| Binarized | irep_A | 2.2 | 3.67 |
| Binarized | irep_B | 4.4 | 1.40 |
| Binarized | brl | 4.8 | 1.67 |
| Binarized | brs | 8.2 | 2.83 |
| Binarized | jrip | 2.4 | 3.43 |
| Binarized | part | 56.8 | 4.09 |
| Binarized | j48 | 78.8 | 11.42 |
| Original | tree | 15.2 | 3.95 |
| Original | forest | 141.8 | 3.89 |
| Original | ripper_A | 5.4 | 3.64 |
| Original | ripper_B | 5.2 | 2.68 |
| Original | irep_A | 2.2 | 2.63 |
| Original | irep_B | 5.2 | 1.34 |
| Original | brl | 4.8 | 1.67 |
| Original | brs | 8.2 | 2.83 |
| Original | jrip | 3.4 | 2.69 |
| Original | part | 60.4 | 3.02 |
| Original | j48 | 77.8 | 5.57 |

(170.7s total)

---

## diabetes

n=768, attributes=8, A='tested_negative', B='tested_positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.737 +/- 0.025 | 0/5 |
| Binarized | forest | 0.747 +/- 0.011 | 0/5 |
| Binarized | ripper_A | 0.687 +/- 0.022 | 0/5 |
| Binarized | ripper_B | 0.746 +/- 0.019 | 0/5 |
| Binarized | irep_A | 0.737 +/- 0.026 | 0/5 |
| Binarized | irep_B | 0.734 +/- 0.033 | 0/5 |
| Binarized | brl | 0.720 +/- 0.027 | 0/5 |
| Binarized | brs | 0.687 +/- 0.018 | 0/5 |
| Binarized | jrip | 0.760 +/- 0.025 | 0/5 |
| Binarized | part | 0.719 +/- 0.022 | 0/5 |
| Binarized | j48 | 0.728 +/- 0.023 | 0/5 |
| Original | tree | 0.734 +/- 0.031 | 0/5 |
| Original | forest | 0.749 +/- 0.035 | 0/5 |
| Original | ripper_A | 0.634 +/- 0.043 | 0/5 |
| Original | ripper_B | 0.728 +/- 0.031 | 0/5 |
| Original | irep_A | 0.708 +/- 0.062 | 0/5 |
| Original | irep_B | 0.719 +/- 0.017 | 0/5 |
| Original | brl | 0.720 +/- 0.027 | 0/5 |
| Original | brs | 0.687 +/- 0.018 | 0/5 |
| Original | jrip | 0.760 +/- 0.031 | 0/5 |
| Original | part | 0.750 +/- 0.027 | 0/5 |
| Original | j48 | 0.741 +/- 0.024 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.005 |
| Binarized | forest | 0.025 |
| Binarized | ripper_A | 0.299 |
| Binarized | ripper_B | 0.269 |
| Binarized | irep_A | 0.060 |
| Binarized | irep_B | 0.061 |
| Binarized | brl | 1.293 |
| Binarized | brs | 9.353 |
| Binarized | jrip | 0.498 |
| Binarized | part | 0.422 |
| Binarized | j48 | 0.409 |
| Original | tree | 0.004 |
| Original | forest | 0.023 |
| Original | ripper_A | 0.248 |
| Original | ripper_B | 0.240 |
| Original | irep_A | 0.098 |
| Original | irep_B | 0.066 |
| Original | brl | 1.277 |
| Original | brs | 9.638 |
| Original | jrip | 0.342 |
| Original | part | 0.325 |
| Original | j48 | 0.392 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 15.2 | 3.94 |
| Binarized | forest | 145.0 | 3.91 |
| Binarized | ripper_A | 5.0 | 2.85 |
| Binarized | ripper_B | 5.0 | 4.12 |
| Binarized | irep_A | 3.6 | 1.73 |
| Binarized | irep_B | 3.4 | 2.53 |
| Binarized | brl | 4.4 | 1.82 |
| Binarized | brs | 7.0 | 3.00 |
| Binarized | jrip | 4.0 | 2.97 |
| Binarized | part | 45.2 | 3.99 |
| Binarized | j48 | 37.6 | 8.62 |
| Original | tree | 15.4 | 3.96 |
| Original | forest | 143.4 | 3.89 |
| Original | ripper_A | 8.4 | 2.26 |
| Original | ripper_B | 8.6 | 2.94 |
| Original | irep_A | 8.8 | 1.93 |
| Original | irep_B | 3.6 | 1.99 |
| Original | brl | 4.4 | 1.82 |
| Original | brs | 7.0 | 3.00 |
| Original | jrip | 3.0 | 2.22 |
| Original | part | 8.0 | 2.30 |
| Original | j48 | 22.0 | 6.09 |

(127.7s total)

---

## sonar

n=208, attributes=60, A='Mine', B='Rock'

*10 model-fold combination(s) timed out (> 300s) or raised and were skipped for that fold -- see per-model failure counts below.*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.687 +/- 0.071 | 0/5 |
| Binarized | forest | 0.779 +/- 0.028 | 0/5 |
| Binarized | ripper_A | 0.717 +/- 0.054 | 0/5 |
| Binarized | ripper_B | 0.755 +/- 0.072 | 0/5 |
| Binarized | irep_A | 0.711 +/- 0.035 | 0/5 |
| Binarized | irep_B | 0.683 +/- 0.052 | 0/5 |
| Binarized | brl | n/a | 5/5 |
| Binarized | brs | 0.741 +/- 0.064 | 0/5 |
| Binarized | jrip | 0.707 +/- 0.045 | 0/5 |
| Binarized | part | 0.745 +/- 0.057 | 0/5 |
| Binarized | j48 | 0.745 +/- 0.056 | 0/5 |
| Original | tree | 0.755 +/- 0.041 | 0/5 |
| Original | forest | 0.764 +/- 0.046 | 0/5 |
| Original | ripper_A | 0.621 +/- 0.111 | 0/5 |
| Original | ripper_B | 0.644 +/- 0.060 | 0/5 |
| Original | irep_A | 0.558 +/- 0.122 | 0/5 |
| Original | irep_B | 0.610 +/- 0.089 | 0/5 |
| Original | brl | n/a | 5/5 |
| Original | brs | 0.741 +/- 0.064 | 0/5 |
| Original | jrip | 0.721 +/- 0.040 | 0/5 |
| Original | part | 0.760 +/- 0.053 | 0/5 |
| Original | j48 | 0.716 +/- 0.078 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.020 |
| Binarized | ripper_A | 0.376 |
| Binarized | ripper_B | 0.366 |
| Binarized | irep_A | 0.296 |
| Binarized | irep_B | 0.329 |
| Binarized | brl | nan |
| Binarized | brs | 245.437 |
| Binarized | jrip | 0.404 |
| Binarized | part | 0.339 |
| Binarized | j48 | 0.317 |
| Original | tree | 0.004 |
| Original | forest | 0.013 |
| Original | ripper_A | 0.795 |
| Original | ripper_B | 0.839 |
| Original | irep_A | 0.682 |
| Original | irep_B | 0.731 |
| Original | brl | nan |
| Original | brs | 241.201 |
| Original | jrip | 0.314 |
| Original | part | 0.290 |
| Original | j48 | 0.290 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.6 | 3.82 |
| Binarized | forest | 122.2 | 3.74 |
| Binarized | ripper_A | 3.8 | 2.32 |
| Binarized | ripper_B | 3.2 | 2.12 |
| Binarized | irep_A | 2.2 | 1.37 |
| Binarized | irep_B | 2.4 | 1.83 |
| Binarized | brl | nan | nan |
| Binarized | brs | 8.6 | 2.68 |
| Binarized | jrip | 4.0 | 2.06 |
| Binarized | part | 6.2 | 3.59 |
| Binarized | j48 | 16.0 | 5.73 |
| Original | tree | 13.0 | 3.76 |
| Original | forest | 114.8 | 3.67 |
| Original | ripper_A | 4.4 | 1.66 |
| Original | ripper_B | 3.4 | 2.59 |
| Original | irep_A | 2.0 | 0.92 |
| Original | irep_B | 2.8 | 1.53 |
| Original | brl | nan | nan |
| Original | brs | 8.6 | 2.68 |
| Original | jrip | 3.6 | 1.73 |
| Original | part | 6.4 | 2.41 |
| Original | j48 | 14.0 | 4.51 |

(9165.4s total)

---

## ionosphere

n=351, attributes=33, A='b', B='g'

*4/5 fold(s) used 'Original's per-model fallback (DataSpecs didn't merge -- see the module docstring).*

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.895 +/- 0.027 | 0/5 |
| Binarized | forest | 0.920 +/- 0.021 | 0/5 |
| Binarized | ripper_A | 0.883 +/- 0.038 | 0/5 |
| Binarized | ripper_B | 0.872 +/- 0.031 | 0/5 |
| Binarized | irep_A | 0.897 +/- 0.040 | 0/5 |
| Binarized | irep_B | 0.892 +/- 0.020 | 0/5 |
| Binarized | brl | 0.855 +/- 0.027 | 0/5 |
| Binarized | brs | 0.895 +/- 0.023 | 0/5 |
| Binarized | jrip | 0.906 +/- 0.032 | 0/5 |
| Binarized | part | 0.915 +/- 0.030 | 0/5 |
| Binarized | j48 | 0.903 +/- 0.030 | 0/5 |
| Original | tree | 0.886 +/- 0.032 | 0/5 |
| Original | forest | 0.920 +/- 0.041 | 0/5 |
| Original | ripper_A | 0.880 +/- 0.042 | 0/5 |
| Original | ripper_B | 0.755 +/- 0.027 | 0/5 |
| Original | irep_A | 0.875 +/- 0.041 | 0/5 |
| Original | irep_B | 0.749 +/- 0.026 | 0/5 |
| Original | brl | 0.855 +/- 0.027 | 0/5 |
| Original | brs | 0.895 +/- 0.023 | 0/5 |
| Original | jrip | 0.909 +/- 0.023 | 0/5 |
| Original | part | 0.906 +/- 0.046 | 0/5 |
| Original | j48 | 0.883 +/- 0.013 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.006 |
| Binarized | forest | 0.016 |
| Binarized | ripper_A | 0.246 |
| Binarized | ripper_B | 0.235 |
| Binarized | irep_A | 0.170 |
| Binarized | irep_B | 0.143 |
| Binarized | brl | 79.744 |
| Binarized | brs | 44.571 |
| Binarized | jrip | 0.375 |
| Binarized | part | 0.350 |
| Binarized | j48 | 0.295 |
| Original | tree | 0.003 |
| Original | forest | 0.012 |
| Original | ripper_A | 0.306 |
| Original | ripper_B | 0.353 |
| Original | irep_A | 0.285 |
| Original | irep_B | 0.246 |
| Original | brl | 79.255 |
| Original | brs | 43.092 |
| Original | jrip | 0.293 |
| Original | part | 0.261 |
| Original | j48 | 0.288 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 8.6 | 3.46 |
| Binarized | forest | 104.4 | 3.61 |
| Binarized | ripper_A | 6.2 | 1.53 |
| Binarized | ripper_B | 3.6 | 3.17 |
| Binarized | irep_A | 2.2 | 1.10 |
| Binarized | irep_B | 1.6 | 2.17 |
| Binarized | brl | 3.2 | 1.93 |
| Binarized | brs | 1.8 | 3.00 |
| Binarized | jrip | 5.4 | 1.52 |
| Binarized | part | 6.0 | 3.09 |
| Binarized | j48 | 10.8 | 4.64 |
| Original | tree | 8.8 | 3.49 |
| Original | forest | 95.4 | 3.51 |
| Original | ripper_A | 8.0 | 2.07 |
| Original | ripper_B | 9.0 | 2.47 |
| Original | irep_A | 2.8 | 1.27 |
| Original | irep_B | 1.0 | 1.00 |
| Original | brl | 3.2 | 1.93 |
| Original | brs | 1.8 | 3.00 |
| Original | jrip | 4.2 | 1.31 |
| Original | part | 5.6 | 2.83 |
| Original | j48 | 11.4 | 5.04 |

(1254.1s total)

---

## tic-tac-toe

n=958, attributes=9, A='negative', B='positive'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.824 +/- 0.021 | 0/5 |
| Binarized | forest | 0.771 +/- 0.020 | 0/5 |
| Binarized | ripper_A | 0.981 +/- 0.010 | 0/5 |
| Binarized | ripper_B | 0.995 +/- 0.003 | 0/5 |
| Binarized | irep_A | 0.857 +/- 0.061 | 0/5 |
| Binarized | irep_B | 0.833 +/- 0.019 | 0/5 |
| Binarized | brl | 0.806 +/- 0.031 | 0/5 |
| Binarized | brs | 0.885 +/- 0.033 | 0/5 |
| Binarized | jrip | 0.978 +/- 0.009 | 0/5 |
| Binarized | part | 0.938 +/- 0.019 | 0/5 |
| Binarized | j48 | 0.949 +/- 0.008 | 0/5 |
| Original | tree | 0.824 +/- 0.021 | 0/5 |
| Original | forest | 0.771 +/- 0.020 | 0/5 |
| Original | ripper_A | 0.979 +/- 0.009 | 0/5 |
| Original | ripper_B | 0.974 +/- 0.023 | 0/5 |
| Original | irep_A | 0.887 +/- 0.073 | 0/5 |
| Original | irep_B | 0.843 +/- 0.035 | 0/5 |
| Original | brl | 0.806 +/- 0.031 | 0/5 |
| Original | brs | 0.885 +/- 0.033 | 0/5 |
| Original | jrip | 0.979 +/- 0.009 | 0/5 |
| Original | part | 0.942 +/- 0.024 | 0/5 |
| Original | j48 | 0.873 +/- 0.028 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.004 |
| Binarized | forest | 0.024 |
| Binarized | ripper_A | 0.114 |
| Binarized | ripper_B | 0.123 |
| Binarized | irep_A | 0.032 |
| Binarized | irep_B | 0.033 |
| Binarized | brl | 0.712 |
| Binarized | brs | 6.109 |
| Binarized | jrip | 0.367 |
| Binarized | part | 0.300 |
| Binarized | j48 | 0.267 |
| Original | tree | 0.003 |
| Original | forest | 0.012 |
| Original | ripper_A | 0.100 |
| Original | ripper_B | 0.135 |
| Original | irep_A | 0.032 |
| Original | irep_B | 0.041 |
| Original | brl | 0.720 |
| Original | brs | 5.944 |
| Original | jrip | 0.300 |
| Original | part | 0.232 |
| Original | j48 | 0.249 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 14.0 | 3.86 |
| Binarized | forest | 157.4 | 3.98 |
| Binarized | ripper_A | 8.8 | 3.13 |
| Binarized | ripper_B | 10.0 | 3.26 |
| Binarized | irep_A | 5.4 | 2.83 |
| Binarized | irep_B | 5.8 | 2.93 |
| Binarized | brl | 11.8 | 1.81 |
| Binarized | brs | 8.6 | 3.00 |
| Binarized | jrip | 8.6 | 3.19 |
| Binarized | part | 29.2 | 3.23 |
| Binarized | j48 | 37.8 | 5.88 |
| Original | tree | 14.0 | 3.86 |
| Original | forest | 157.4 | 3.98 |
| Original | ripper_A | 9.4 | 3.28 |
| Original | ripper_B | 14.0 | 3.13 |
| Original | irep_A | 6.8 | 2.80 |
| Original | irep_B | 6.4 | 2.32 |
| Original | brl | 11.8 | 1.81 |
| Original | brs | 8.6 | 3.00 |
| Original | jrip | 9.6 | 3.28 |
| Original | part | 36.4 | 2.69 |
| Original | j48 | 80.6 | 4.53 |

(80.1s total)

---

## banknote-authentication

n=1372, attributes=4, A='1', B='2'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.958 +/- 0.008 | 0/5 |
| Binarized | forest | 0.950 +/- 0.023 | 0/5 |
| Binarized | ripper_A | 0.985 +/- 0.006 | 0/5 |
| Binarized | ripper_B | 0.982 +/- 0.007 | 0/5 |
| Binarized | irep_A | 0.939 +/- 0.015 | 0/5 |
| Binarized | irep_B | 0.972 +/- 0.010 | 0/5 |
| Binarized | brl | 0.972 +/- 0.013 | 0/5 |
| Binarized | brs | 0.956 +/- 0.017 | 0/5 |
| Binarized | jrip | 0.986 +/- 0.006 | 0/5 |
| Binarized | part | 0.983 +/- 0.007 | 0/5 |
| Binarized | j48 | 0.985 +/- 0.007 | 0/5 |
| Original | tree | 0.957 +/- 0.016 | 0/5 |
| Original | forest | 0.970 +/- 0.013 | 0/5 |
| Original | ripper_A | 0.971 +/- 0.009 | 0/5 |
| Original | ripper_B | 0.956 +/- 0.021 | 0/5 |
| Original | irep_A | 0.907 +/- 0.010 | 0/5 |
| Original | irep_B | 0.898 +/- 0.041 | 0/5 |
| Original | brl | 0.972 +/- 0.013 | 0/5 |
| Original | brs | 0.956 +/- 0.017 | 0/5 |
| Original | jrip | 0.979 +/- 0.006 | 0/5 |
| Original | part | 0.988 +/- 0.008 | 0/5 |
| Original | j48 | 0.986 +/- 0.013 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.004 |
| Binarized | forest | 0.020 |
| Binarized | ripper_A | 0.089 |
| Binarized | ripper_B | 0.090 |
| Binarized | irep_A | 0.030 |
| Binarized | irep_B | 0.032 |
| Binarized | brl | 0.760 |
| Binarized | brs | 6.217 |
| Binarized | jrip | 0.340 |
| Binarized | part | 0.291 |
| Binarized | j48 | 0.264 |
| Original | tree | 0.004 |
| Original | forest | 0.014 |
| Original | ripper_A | 0.326 |
| Original | ripper_B | 0.349 |
| Original | irep_A | 0.042 |
| Original | irep_B | 0.041 |
| Original | brl | 0.793 |
| Original | brs | 6.064 |
| Original | jrip | 0.299 |
| Original | part | 0.248 |
| Original | j48 | 0.268 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 13.0 | 3.77 |
| Binarized | forest | 116.4 | 3.70 |
| Binarized | ripper_A | 6.8 | 2.33 |
| Binarized | ripper_B | 6.4 | 2.48 |
| Binarized | irep_A | 3.2 | 1.73 |
| Binarized | irep_B | 4.0 | 2.08 |
| Binarized | brl | 5.0 | 1.89 |
| Binarized | brs | 9.2 | 3.00 |
| Binarized | jrip | 5.4 | 2.36 |
| Binarized | part | 9.4 | 2.10 |
| Binarized | j48 | 11.4 | 3.88 |
| Original | tree | 12.2 | 3.69 |
| Original | forest | 123.4 | 3.74 |
| Original | ripper_A | 33.6 | 3.65 |
| Original | ripper_B | 31.2 | 3.85 |
| Original | irep_A | 9.8 | 2.58 |
| Original | irep_B | 8.8 | 2.76 |
| Original | brl | 5.0 | 1.89 |
| Original | brs | 9.2 | 3.00 |
| Original | jrip | 6.0 | 2.24 |
| Original | part | 7.4 | 2.11 |
| Original | j48 | 15.2 | 4.51 |

(83.8s total)

---

## kr-vs-kp

n=3196, attributes=36, A='nowin', B='won'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.941 +/- 0.011 | 0/5 |
| Binarized | forest | 0.941 +/- 0.011 | 0/5 |
| Binarized | ripper_A | 0.989 +/- 0.005 | 0/5 |
| Binarized | ripper_B | 0.988 +/- 0.004 | 0/5 |
| Binarized | irep_A | 0.980 +/- 0.009 | 0/5 |
| Binarized | irep_B | 0.950 +/- 0.034 | 0/5 |
| Binarized | brl | 0.941 +/- 0.025 | 0/5 |
| Binarized | brs | 0.811 +/- 0.086 | 0/5 |
| Binarized | jrip | 0.992 +/- 0.004 | 0/5 |
| Binarized | part | 0.989 +/- 0.003 | 0/5 |
| Binarized | j48 | 0.993 +/- 0.004 | 0/5 |
| Original | tree | 0.941 +/- 0.011 | 0/5 |
| Original | forest | 0.941 +/- 0.011 | 0/5 |
| Original | ripper_A | 0.987 +/- 0.006 | 0/5 |
| Original | ripper_B | 0.988 +/- 0.003 | 0/5 |
| Original | irep_A | 0.981 +/- 0.008 | 0/5 |
| Original | irep_B | 0.922 +/- 0.024 | 0/5 |
| Original | brl | 0.941 +/- 0.025 | 0/5 |
| Original | brs | 0.811 +/- 0.086 | 0/5 |
| Original | jrip | 0.988 +/- 0.007 | 0/5 |
| Original | part | 0.990 +/- 0.005 | 0/5 |
| Original | j48 | 0.993 +/- 0.004 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.008 |
| Binarized | forest | 0.031 |
| Binarized | ripper_A | 0.564 |
| Binarized | ripper_B | 0.495 |
| Binarized | irep_A | 0.111 |
| Binarized | irep_B | 0.120 |
| Binarized | brl | 9.857 |
| Binarized | brs | 10.928 |
| Binarized | jrip | 0.682 |
| Binarized | part | 0.444 |
| Binarized | j48 | 0.371 |
| Original | tree | 0.007 |
| Original | forest | 0.018 |
| Original | ripper_A | 0.492 |
| Original | ripper_B | 0.454 |
| Original | irep_A | 0.144 |
| Original | irep_B | 0.125 |
| Original | brl | 9.785 |
| Original | brs | 10.949 |
| Original | jrip | 0.414 |
| Original | part | 0.308 |
| Original | j48 | 0.296 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 7.6 | 3.43 |
| Binarized | forest | 118.6 | 3.75 |
| Binarized | ripper_A | 17.4 | 3.23 |
| Binarized | ripper_B | 9.8 | 4.61 |
| Binarized | irep_A | 6.6 | 2.74 |
| Binarized | irep_B | 5.2 | 4.04 |
| Binarized | brl | 8.4 | 1.87 |
| Binarized | brs | 7.0 | 2.80 |
| Binarized | jrip | 13.6 | 2.99 |
| Binarized | part | 20.8 | 2.93 |
| Binarized | j48 | 25.8 | 7.37 |
| Original | tree | 7.6 | 3.43 |
| Original | forest | 118.6 | 3.75 |
| Original | ripper_A | 18.4 | 3.33 |
| Original | ripper_B | 10.8 | 4.80 |
| Original | irep_A | 7.0 | 2.87 |
| Original | irep_B | 4.2 | 3.63 |
| Original | brl | 8.4 | 1.87 |
| Original | brs | 7.0 | 2.80 |
| Original | jrip | 14.6 | 3.16 |
| Original | part | 20.4 | 3.21 |
| Original | j48 | 29.0 | 7.71 |

(236.5s total)

---

## mushroom

n=8124, attributes=21, A='e', B='p'

**Accuracy**

| workflow | model | accuracy | failed folds |
|---|---|---|---|
| Binarized | tree | 0.992 +/- 0.001 | 0/5 |
| Binarized | forest | 0.987 +/- 0.004 | 0/5 |
| Binarized | ripper_A | 1.000 +/- 0.000 | 0/5 |
| Binarized | ripper_B | 1.000 +/- 0.000 | 0/5 |
| Binarized | irep_A | 0.985 +/- 0.002 | 0/5 |
| Binarized | irep_B | 1.000 +/- 0.000 | 0/5 |
| Binarized | brl | 0.994 +/- 0.005 | 0/5 |
| Binarized | brs | 0.997 +/- 0.004 | 0/5 |
| Binarized | jrip | 1.000 +/- 0.000 | 0/5 |
| Binarized | part | 1.000 +/- 0.000 | 0/5 |
| Binarized | j48 | 1.000 +/- 0.000 | 0/5 |
| Original | tree | 0.992 +/- 0.001 | 0/5 |
| Original | forest | 0.987 +/- 0.004 | 0/5 |
| Original | ripper_A | 1.000 +/- 0.000 | 0/5 |
| Original | ripper_B | 1.000 +/- 0.000 | 0/5 |
| Original | irep_A | 0.985 +/- 0.002 | 0/5 |
| Original | irep_B | 0.965 +/- 0.003 | 0/5 |
| Original | brl | 0.994 +/- 0.005 | 0/5 |
| Original | brs | 0.997 +/- 0.004 | 0/5 |
| Original | jrip | 1.000 +/- 0.000 | 0/5 |
| Original | part | 1.000 +/- 0.000 | 0/5 |
| Original | j48 | 1.000 +/- 0.000 | 0/5 |

**Fit time (seconds/fold)**

| workflow | model | time |
|---|---|---|
| Binarized | tree | 0.021 |
| Binarized | forest | 0.057 |
| Binarized | ripper_A | 0.756 |
| Binarized | ripper_B | 0.760 |
| Binarized | irep_A | 0.321 |
| Binarized | irep_B | 0.362 |
| Binarized | brl | 16.742 |
| Binarized | brs | 763.284 |
| Binarized | jrip | 0.964 |
| Binarized | part | 0.746 |
| Binarized | j48 | 0.682 |
| Original | tree | 0.038 |
| Original | forest | 0.039 |
| Original | ripper_A | 0.494 |
| Original | ripper_B | 0.375 |
| Original | irep_A | 0.191 |
| Original | irep_B | 0.177 |
| Original | brl | 16.946 |
| Original | brs | 24.173 |
| Original | jrip | 0.528 |
| Original | part | 0.403 |
| Original | j48 | 0.375 |

**Rule complexity**

| workflow | model | n_rules | avg_conditions |
|---|---|---|---|
| Binarized | tree | 10.0 | 3.50 |
| Binarized | forest | 101.8 | 3.58 |
| Binarized | ripper_A | 5.8 | 2.45 |
| Binarized | ripper_B | 7.0 | 1.56 |
| Binarized | irep_A | 4.0 | 1.50 |
| Binarized | irep_B | 6.0 | 1.79 |
| Binarized | brl | 10.0 | 1.85 |
| Binarized | brs | 9.2 | 2.89 |
| Binarized | jrip | 5.4 | 2.09 |
| Binarized | part | 5.0 | 2.76 |
| Binarized | j48 | 9.8 | 3.83 |
| Original | tree | 10.0 | 3.50 |
| Original | forest | 101.8 | 3.58 |
| Original | ripper_A | 7.6 | 2.54 |
| Original | ripper_B | 8.2 | 1.50 |
| Original | irep_A | 4.0 | 1.50 |
| Original | irep_B | 4.0 | 1.00 |
| Original | brl | 10.0 | 1.85 |
| Original | brs | 9.2 | 2.89 |
| Original | jrip | 8.0 | 1.56 |
| Original | part | 10.0 | 1.85 |
| Original | j48 | 24.0 | 2.54 |

(4152.1s total)

---

## Overall summary

### Accuracy

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 | fallback folds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.949 | 0.947 | 0.940 | 0.961 | 0.945 | 0.947 | 0.940 | 0.903 | 0.949 | 0.954 | 0.961 | 0.949 | 0.947 | 0.952 | 0.952 | 0.949 | 0.956 | 0.940 | 0.903 | 0.956 | 0.956 | 0.963 | 0/5 |
| breast-cancer | 0.727 | 0.755 | 0.620 | 0.738 | 0.734 | 0.724 | 0.731 | 0.720 | 0.741 | 0.671 | 0.699 | 0.727 | 0.755 | 0.668 | 0.731 | 0.755 | 0.727 | 0.731 | 0.720 | 0.720 | 0.700 | 0.734 | 0/5 |
| colic | 0.851 | 0.850 | 0.856 | 0.851 | 0.845 | 0.837 | 0.848 | 0.799 | 0.848 | 0.807 | 0.856 | 0.843 | 0.837 | 0.813 | 0.870 | 0.826 | 0.851 | 0.848 | 0.799 | 0.875 | 0.859 | 0.826 | 0/5 |
| credit-approval | 0.849 | 0.857 | 0.849 | 0.857 | 0.852 | 0.852 | 0.858 | 0.625 | 0.865 | 0.839 | 0.852 | 0.848 | 0.857 | 0.836 | 0.830 | 0.855 | 0.859 | 0.858 | 0.625 | 0.859 | 0.843 | 0.859 | 0/5 |
| credit-g | 0.705 | 0.732 | 0.712 | 0.597 | 0.725 | 0.713 | 0.698 | 0.436 | 0.718 | 0.723 | 0.719 | 0.706 | 0.705 | 0.702 | 0.586 | 0.701 | 0.698 | 0.698 | 0.436 | 0.724 | 0.700 | 0.708 | 5/5 |
| diabetes | 0.737 | 0.747 | 0.687 | 0.746 | 0.737 | 0.734 | 0.720 | 0.687 | 0.760 | 0.719 | 0.728 | 0.734 | 0.749 | 0.634 | 0.728 | 0.708 | 0.719 | 0.720 | 0.687 | 0.760 | 0.750 | 0.741 | 0/5 |
| sonar | 0.687 | 0.779 | 0.717 | 0.755 | 0.711 | 0.683 | nan | 0.741 | 0.707 | 0.745 | 0.745 | 0.755 | 0.764 | 0.621 | 0.644 | 0.558 | 0.610 | nan | 0.741 | 0.721 | 0.760 | 0.716 | 0/5 |
| ionosphere | 0.895 | 0.920 | 0.883 | 0.872 | 0.897 | 0.892 | 0.855 | 0.895 | 0.906 | 0.915 | 0.903 | 0.886 | 0.920 | 0.880 | 0.755 | 0.875 | 0.749 | 0.855 | 0.895 | 0.909 | 0.906 | 0.883 | 4/5 |
| tic-tac-toe | 0.824 | 0.771 | 0.981 | 0.995 | 0.857 | 0.833 | 0.806 | 0.885 | 0.978 | 0.938 | 0.949 | 0.824 | 0.771 | 0.979 | 0.974 | 0.887 | 0.843 | 0.806 | 0.885 | 0.979 | 0.942 | 0.873 | 0/5 |
| banknote-authentication | 0.958 | 0.950 | 0.985 | 0.982 | 0.939 | 0.972 | 0.972 | 0.956 | 0.986 | 0.983 | 0.985 | 0.957 | 0.970 | 0.971 | 0.956 | 0.907 | 0.898 | 0.972 | 0.956 | 0.979 | 0.988 | 0.986 | 0/5 |
| kr-vs-kp | 0.941 | 0.941 | 0.989 | 0.988 | 0.980 | 0.950 | 0.941 | 0.811 | 0.992 | 0.989 | 0.993 | 0.941 | 0.941 | 0.987 | 0.988 | 0.981 | 0.922 | 0.941 | 0.811 | 0.988 | 0.990 | 0.993 | 0/5 |
| mushroom | 0.992 | 0.987 | 1.000 | 1.000 | 0.985 | 1.000 | 0.994 | 0.997 | 1.000 | 1.000 | 1.000 | 0.992 | 0.987 | 1.000 | 1.000 | 0.985 | 0.965 | 0.994 | 0.997 | 1.000 | 1.000 | 1.000 | 0/5 |

### Fit time (seconds/fold)

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.309 | 0.022 | 0.056 | 0.053 | 0.034 | 0.034 | 0.623 | 6.891 | 0.305 | 0.269 | 0.257 | 0.003 | 0.016 | 0.053 | 0.050 | 0.031 | 0.031 | 0.534 | 6.805 | 0.229 | 0.223 | 0.226 |
| breast-cancer | 0.003 | 0.017 | 0.062 | 0.096 | 0.031 | 0.031 | 0.238 | 6.992 | 0.287 | 0.312 | 0.275 | 0.003 | 0.012 | 0.049 | 0.080 | 0.036 | 0.027 | 0.227 | 6.951 | 0.264 | 0.250 | 0.449 |
| colic | 0.010 | 0.021 | 0.200 | 0.208 | 0.123 | 0.122 | 3.895 | 29.102 | 0.396 | 0.372 | 0.364 | 0.004 | 0.012 | 0.198 | 0.212 | 0.142 | 0.151 | 3.720 | 894.841 | 0.308 | 0.283 | 0.270 |
| credit-approval | 0.011 | 0.022 | 0.264 | 0.227 | 0.070 | 0.084 | 1.780 | 11.084 | 0.479 | 0.404 | 0.412 | 0.004 | 0.014 | 0.264 | 0.269 | 0.095 | 0.084 | 1.774 | 10.975 | 0.423 | 0.311 | 0.371 |
| credit-g | 0.006 | 0.025 | 0.466 | 0.324 | 0.103 | 0.098 | 3.122 | 11.381 | 0.552 | 0.556 | 0.510 | 0.005 | 0.023 | 0.342 | 0.300 | 0.096 | 0.107 | 3.118 | 11.431 | 0.445 | 0.388 | 0.344 |
| diabetes | 0.005 | 0.025 | 0.299 | 0.269 | 0.060 | 0.061 | 1.293 | 9.353 | 0.498 | 0.422 | 0.409 | 0.004 | 0.023 | 0.248 | 0.240 | 0.098 | 0.066 | 1.277 | 9.638 | 0.342 | 0.325 | 0.392 |
| sonar | 0.006 | 0.020 | 0.376 | 0.366 | 0.296 | 0.329 | nan | 245.437 | 0.404 | 0.339 | 0.317 | 0.004 | 0.013 | 0.795 | 0.839 | 0.682 | 0.731 | nan | 241.201 | 0.314 | 0.290 | 0.290 |
| ionosphere | 0.006 | 0.016 | 0.246 | 0.235 | 0.170 | 0.143 | 79.744 | 44.571 | 0.375 | 0.350 | 0.295 | 0.003 | 0.012 | 0.306 | 0.353 | 0.285 | 0.246 | 79.255 | 43.092 | 0.293 | 0.261 | 0.288 |
| tic-tac-toe | 0.004 | 0.024 | 0.114 | 0.123 | 0.032 | 0.033 | 0.712 | 6.109 | 0.367 | 0.300 | 0.267 | 0.003 | 0.012 | 0.100 | 0.135 | 0.032 | 0.041 | 0.720 | 5.944 | 0.300 | 0.232 | 0.249 |
| banknote-authentication | 0.004 | 0.020 | 0.089 | 0.090 | 0.030 | 0.032 | 0.760 | 6.217 | 0.340 | 0.291 | 0.264 | 0.004 | 0.014 | 0.326 | 0.349 | 0.042 | 0.041 | 0.793 | 6.064 | 0.299 | 0.248 | 0.268 |
| kr-vs-kp | 0.008 | 0.031 | 0.564 | 0.495 | 0.111 | 0.120 | 9.857 | 10.928 | 0.682 | 0.444 | 0.371 | 0.007 | 0.018 | 0.492 | 0.454 | 0.144 | 0.125 | 9.785 | 10.949 | 0.414 | 0.308 | 0.296 |
| mushroom | 0.021 | 0.057 | 0.756 | 0.760 | 0.321 | 0.362 | 16.742 | 763.284 | 0.964 | 0.746 | 0.682 | 0.038 | 0.039 | 0.494 | 0.375 | 0.191 | 0.177 | 16.946 | 24.173 | 0.528 | 0.403 | 0.375 |

### Rule count

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 13.4 | 121.0 | 2.0 | 3.0 | 2.0 | 1.4 | 3.8 | 7.8 | 2.2 | 5.2 | 6.8 | 13.4 | 121.0 | 3.4 | 3.8 | 2.0 | 1.0 | 3.8 | 7.8 | 1.2 | 5.2 | 9.0 |
| breast-cancer | 12.4 | 124.8 | 1.6 | 1.6 | 2.2 | 1.6 | 3.2 | 7.6 | 1.2 | 22.0 | 15.0 | 12.4 | 124.8 | 3.0 | 1.6 | 2.4 | 1.8 | 3.2 | 7.6 | 2.8 | 15.4 | 8.4 |
| colic | 11.4 | 122.6 | 4.6 | 5.4 | 2.0 | 2.0 | 4.8 | 7.8 | 3.4 | 13.6 | 15.6 | 11.6 | 118.6 | 6.0 | 6.2 | 3.2 | 1.4 | 4.8 | 7.8 | 3.4 | 12.8 | 8.6 |
| credit-approval | 13.8 | 136.4 | 5.8 | 4.0 | 1.2 | 1.8 | 4.4 | 7.8 | 4.4 | 26.8 | 17.0 | 14.2 | 134.0 | 7.6 | 8.8 | 1.4 | 1.8 | 4.4 | 7.8 | 3.8 | 29.6 | 19.4 |
| credit-g | 15.0 | 143.6 | 3.8 | 4.2 | 2.2 | 4.4 | 4.8 | 8.2 | 2.4 | 56.8 | 78.8 | 15.2 | 141.8 | 5.4 | 5.2 | 2.2 | 5.2 | 4.8 | 8.2 | 3.4 | 60.4 | 77.8 |
| diabetes | 15.2 | 145.0 | 5.0 | 5.0 | 3.6 | 3.4 | 4.4 | 7.0 | 4.0 | 45.2 | 37.6 | 15.4 | 143.4 | 8.4 | 8.6 | 8.8 | 3.6 | 4.4 | 7.0 | 3.0 | 8.0 | 22.0 |
| sonar | 13.6 | 122.2 | 3.8 | 3.2 | 2.2 | 2.4 | nan | 8.6 | 4.0 | 6.2 | 16.0 | 13.0 | 114.8 | 4.4 | 3.4 | 2.0 | 2.8 | nan | 8.6 | 3.6 | 6.4 | 14.0 |
| ionosphere | 8.6 | 104.4 | 6.2 | 3.6 | 2.2 | 1.6 | 3.2 | 1.8 | 5.4 | 6.0 | 10.8 | 8.8 | 95.4 | 8.0 | 9.0 | 2.8 | 1.0 | 3.2 | 1.8 | 4.2 | 5.6 | 11.4 |
| tic-tac-toe | 14.0 | 157.4 | 8.8 | 10.0 | 5.4 | 5.8 | 11.8 | 8.6 | 8.6 | 29.2 | 37.8 | 14.0 | 157.4 | 9.4 | 14.0 | 6.8 | 6.4 | 11.8 | 8.6 | 9.6 | 36.4 | 80.6 |
| banknote-authentication | 13.0 | 116.4 | 6.8 | 6.4 | 3.2 | 4.0 | 5.0 | 9.2 | 5.4 | 9.4 | 11.4 | 12.2 | 123.4 | 33.6 | 31.2 | 9.8 | 8.8 | 5.0 | 9.2 | 6.0 | 7.4 | 15.2 |
| kr-vs-kp | 7.6 | 118.6 | 17.4 | 9.8 | 6.6 | 5.2 | 8.4 | 7.0 | 13.6 | 20.8 | 25.8 | 7.6 | 118.6 | 18.4 | 10.8 | 7.0 | 4.2 | 8.4 | 7.0 | 14.6 | 20.4 | 29.0 |
| mushroom | 10.0 | 101.8 | 5.8 | 7.0 | 4.0 | 6.0 | 10.0 | 9.2 | 5.4 | 5.0 | 9.8 | 10.0 | 101.8 | 7.6 | 8.2 | 4.0 | 4.0 | 10.0 | 9.2 | 8.0 | 10.0 | 24.0 |

### Average conditions per rule

| dataset | Binarized_tree | Binarized_forest | Binarized_ripper_A | Binarized_ripper_B | Binarized_irep_A | Binarized_irep_B | Binarized_brl | Binarized_brs | Binarized_jrip | Binarized_part | Binarized_j48 | Original_tree | Original_forest | Original_ripper_A | Original_ripper_B | Original_irep_A | Original_irep_B | Original_brl | Original_brs | Original_jrip | Original_part | Original_j48 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 3.80 | 3.73 | 1.53 | 2.25 | 1.30 | 1.20 | 1.48 | 3.00 | 1.67 | 2.23 | 3.32 | 3.80 | 3.73 | 1.88 | 1.95 | 1.50 | 1.00 | 1.48 | 3.00 | 1.20 | 1.42 | 2.61 |
| breast-cancer | 3.75 | 3.78 | 1.60 | 2.70 | 1.67 | 2.00 | 1.42 | 3.00 | 1.80 | 4.54 | 5.37 | 3.75 | 3.78 | 1.55 | 2.70 | 1.07 | 1.53 | 1.42 | 3.00 | 2.22 | 1.92 | 1.77 |
| colic | 3.68 | 3.75 | 2.07 | 1.88 | 1.20 | 1.37 | 1.46 | 2.86 | 1.85 | 4.53 | 5.97 | 3.72 | 3.73 | 1.61 | 2.13 | 1.21 | 1.70 | 1.46 | 2.86 | 1.83 | 1.81 | 2.17 |
| credit-approval | 3.85 | 3.85 | 2.71 | 2.57 | 1.10 | 1.67 | 1.66 | 2.83 | 2.40 | 3.49 | 6.31 | 3.87 | 3.84 | 2.86 | 3.22 | 1.30 | 1.57 | 1.66 | 2.83 | 2.18 | 2.37 | 4.02 |
| credit-g | 3.93 | 3.90 | 4.87 | 2.86 | 3.67 | 1.40 | 1.67 | 2.83 | 3.43 | 4.09 | 11.42 | 3.95 | 3.89 | 3.64 | 2.68 | 2.63 | 1.34 | 1.67 | 2.83 | 2.69 | 3.02 | 5.57 |
| diabetes | 3.94 | 3.91 | 2.85 | 4.12 | 1.73 | 2.53 | 1.82 | 3.00 | 2.97 | 3.99 | 8.62 | 3.96 | 3.89 | 2.26 | 2.94 | 1.93 | 1.99 | 1.82 | 3.00 | 2.22 | 2.30 | 6.09 |
| sonar | 3.82 | 3.74 | 2.32 | 2.12 | 1.37 | 1.83 | nan | 2.68 | 2.06 | 3.59 | 5.73 | 3.76 | 3.67 | 1.66 | 2.59 | 0.92 | 1.53 | nan | 2.68 | 1.73 | 2.41 | 4.51 |
| ionosphere | 3.46 | 3.61 | 1.53 | 3.17 | 1.10 | 2.17 | 1.93 | 3.00 | 1.52 | 3.09 | 4.64 | 3.49 | 3.51 | 2.07 | 2.47 | 1.27 | 1.00 | 1.93 | 3.00 | 1.31 | 2.83 | 5.04 |
| tic-tac-toe | 3.86 | 3.98 | 3.13 | 3.26 | 2.83 | 2.93 | 1.81 | 3.00 | 3.19 | 3.23 | 5.88 | 3.86 | 3.98 | 3.28 | 3.13 | 2.80 | 2.32 | 1.81 | 3.00 | 3.28 | 2.69 | 4.53 |
| banknote-authentication | 3.77 | 3.70 | 2.33 | 2.48 | 1.73 | 2.08 | 1.89 | 3.00 | 2.36 | 2.10 | 3.88 | 3.69 | 3.74 | 3.65 | 3.85 | 2.58 | 2.76 | 1.89 | 3.00 | 2.24 | 2.11 | 4.51 |
| kr-vs-kp | 3.43 | 3.75 | 3.23 | 4.61 | 2.74 | 4.04 | 1.87 | 2.80 | 2.99 | 2.93 | 7.37 | 3.43 | 3.75 | 3.33 | 4.80 | 2.87 | 3.63 | 1.87 | 2.80 | 3.16 | 3.21 | 7.71 |
| mushroom | 3.50 | 3.58 | 2.45 | 1.56 | 1.50 | 1.79 | 1.85 | 2.89 | 2.09 | 2.76 | 3.83 | 3.50 | 3.58 | 2.54 | 1.50 | 1.50 | 1.00 | 1.85 | 2.89 | 1.56 | 1.85 | 2.54 |
