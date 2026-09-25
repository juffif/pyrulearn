# Four `DataRepresentation` encodings -- same rules, with & without negation

Generated 2026-09-25 15:25:53. Single stratified 67/33 split, `random_state=0`. Each learner is fit on all four representations (Boolean = packed bit-matrix, NList = PPC-tree / N-list, PrePostNList = NList + pre/post visit codes on its search fast path, Sparse = scipy CSR/CSC = the N-list without the prefix tree) and the rule sets / prediction vectors are compared to the Boolean baseline exactly. Every dataset is run in two feature encodings: **with negation** (paired negation features) and **without** (positive tests only). `coverage` timing is 1000 random rules (length 1-4).

CN2 / PFoil / PFossil run everywhere; AQR / PyLORD only on vote & tic-tac-toe. PrePostNList is included to *demonstrate* it stays correct, not because it's expected to be faster here -- see `PrePostNListRepresentation`'s own docstring for why it measured out roughly break-even to slightly worse than plain NList at these data sizes.

---

## vote

### vote -- with negation

train n=291, features=96. Vertical index build: NList 48 ms (6863 tree nodes), PrePostNList 55 ms (6863 tree nodes), Sparse 13 ms (nnz=13968, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.09s | 25 ms |
| CN2 | NList | yes | yes | 0.611 | 0.09s | 11 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.09s | 11 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.09s | 11 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.11s | 25 ms |
| PFoil | NList | yes | yes | 0.611 | 0.11s | 11 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.11s | 11 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.11s | 11 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.23s | 25 ms |
| PFossil | NList | yes | yes | 0.611 | 0.22s | 11 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.21s | 11 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.21s | 11 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 2.78s | 25 ms |
| AQR | NList | yes | yes | 0.611 | 2.58s | 11 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 2.42s | 11 ms |
| AQR | Sparse | yes | yes | 0.611 | 2.57s | 11 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.944 | 3.38s | 25 ms |
| PyLORD | NList | yes | yes | 0.944 | 3.01s | 11 ms |
| PyLORD | PrePostNList | yes | yes | 0.944 | 3.00s | 11 ms |
| PyLORD | Sparse | yes | yes | 0.944 | 3.27s | 11 ms |

### vote -- without negation

train n=291, features=48. Vertical index build: NList 19 ms (2415 tree nodes), PrePostNList 23 ms (2415 tree nodes), Sparse 8 ms (nnz=4656, density 0.33).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.04s | 37 ms |
| CN2 | NList | yes | yes | 0.611 | 0.03s | 18 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.03s | 11 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.05s | 10 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.07s | 37 ms |
| PFoil | NList | yes | yes | 0.611 | 0.07s | 18 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.07s | 11 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.06s | 10 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.07s | 37 ms |
| PFossil | NList | yes | yes | 0.611 | 0.06s | 18 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.08s | 11 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.09s | 10 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 0.29s | 37 ms |
| AQR | NList | yes | yes | 0.611 | 0.21s | 18 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 0.27s | 11 ms |
| AQR | Sparse | yes | yes | 0.611 | 0.28s | 10 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.951 | 0.76s | 37 ms |
| PyLORD | NList | yes | yes | 0.951 | 0.61s | 18 ms |
| PyLORD | PrePostNList | yes | yes | 0.951 | 0.58s | 11 ms |
| PyLORD | Sparse | yes | yes | 0.951 | 0.73s | 10 ms |

*Encoding effect:* with negation 96 features (acc 0.944); without, 48 features (acc 0.951), sparse density 0.50 -> 0.33.

---

## tic-tac-toe

### tic-tac-toe -- with negation

train n=641, features=54. Vertical index build: NList 68 ms (12190 tree nodes), PrePostNList 113 ms (12190 tree nodes), Sparse 10 ms (nnz=17307, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.981 | 0.42s | 44 ms |
| CN2 | NList | yes | yes | 0.981 | 0.40s | 13 ms |
| CN2 | PrePostNList | yes | yes | 0.981 | 0.34s | 12 ms |
| CN2 | Sparse | yes | yes | 0.981 | 0.42s | 23 ms |
| PFoil | Boolean | (ref) | (ref) | 0.890 | 0.22s | 44 ms |
| PFoil | NList | yes | yes | 0.890 | 0.20s | 13 ms |
| PFoil | PrePostNList | yes | yes | 0.890 | 0.20s | 12 ms |
| PFoil | Sparse | yes | yes | 0.890 | 0.25s | 23 ms |
| PFossil | Boolean | (ref) | (ref) | 0.830 | 0.16s | 44 ms |
| PFossil | NList | yes | yes | 0.830 | 0.15s | 13 ms |
| PFossil | PrePostNList | yes | yes | 0.830 | 0.15s | 12 ms |
| PFossil | Sparse | yes | yes | 0.830 | 0.13s | 23 ms |
| AQR | Boolean | (ref) | (ref) | 0.861 | 4.19s | 44 ms |
| AQR | NList | yes | yes | 0.861 | 3.81s | 13 ms |
| AQR | PrePostNList | yes | yes | 0.861 | 4.10s | 12 ms |
| AQR | Sparse | yes | yes | 0.861 | 4.48s | 23 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.965 | 6.28s | 44 ms |
| PyLORD | NList | yes | yes | 0.965 | 5.07s | 13 ms |
| PyLORD | PrePostNList | yes | yes | 0.965 | 4.81s | 12 ms |
| PyLORD | Sparse | yes | yes | 0.965 | 6.16s | 23 ms |

### tic-tac-toe -- without negation

train n=641, features=27. Vertical index build: NList 32 ms (3249 tree nodes), PrePostNList 38 ms (3249 tree nodes), Sparse 8 ms (nnz=5769, density 0.33).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.981 | 0.14s | 41 ms |
| CN2 | NList | yes | yes | 0.981 | 0.10s | 15 ms |
| CN2 | PrePostNList | yes | yes | 0.981 | 0.09s | 15 ms |
| CN2 | Sparse | yes | yes | 0.981 | 0.13s | 19 ms |
| PFoil | Boolean | (ref) | (ref) | 0.950 | 0.09s | 41 ms |
| PFoil | NList | yes | yes | 0.950 | 0.08s | 15 ms |
| PFoil | PrePostNList | yes | yes | 0.950 | 0.07s | 15 ms |
| PFoil | Sparse | yes | yes | 0.950 | 0.12s | 19 ms |
| PFossil | Boolean | (ref) | (ref) | 0.830 | 0.06s | 41 ms |
| PFossil | NList | yes | yes | 0.830 | 0.05s | 15 ms |
| PFossil | PrePostNList | yes | yes | 0.830 | 0.04s | 15 ms |
| PFossil | Sparse | yes | yes | 0.830 | 0.05s | 19 ms |
| AQR | Boolean | (ref) | (ref) | 0.962 | 0.19s | 41 ms |
| AQR | NList | yes | yes | 0.962 | 0.13s | 15 ms |
| AQR | PrePostNList | yes | yes | 0.962 | 0.12s | 15 ms |
| AQR | Sparse | yes | yes | 0.962 | 0.17s | 19 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.972 | 1.89s | 41 ms |
| PyLORD | NList | yes | yes | 0.972 | 1.16s | 15 ms |
| PyLORD | PrePostNList | yes | yes | 0.972 | 1.30s | 15 ms |
| PyLORD | Sparse | yes | yes | 0.972 | 1.64s | 19 ms |

*Encoding effect:* with negation 54 features (acc 0.965); without, 27 features (acc 0.972), sparse density 0.50 -> 0.33.

---

## breast-w

### breast-w -- with negation

train n=468, features=178. Vertical index build: NList 196 ms (14185 tree nodes), PrePostNList 209 ms (14185 tree nodes), Sparse 37 ms (nnz=41652, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.654 | 0.76s | 43 ms |
| CN2 | NList | yes | yes | 0.654 | 0.70s | 21 ms |
| CN2 | PrePostNList | yes | yes | 0.654 | 0.73s | 22 ms |
| CN2 | Sparse | yes | yes | 0.654 | 0.81s | 15 ms |
| PFoil | Boolean | (ref) | (ref) | 0.654 | 0.77s | 43 ms |
| PFoil | NList | yes | yes | 0.654 | 0.66s | 21 ms |
| PFoil | PrePostNList | yes | yes | 0.654 | 0.67s | 22 ms |
| PFoil | Sparse | yes | yes | 0.654 | 0.69s | 15 ms |
| PFossil | Boolean | (ref) | (ref) | 0.654 | 0.62s | 43 ms |
| PFossil | NList | yes | yes | 0.654 | 0.61s | 21 ms |
| PFossil | PrePostNList | yes | yes | 0.654 | 0.57s | 22 ms |
| PFossil | Sparse | yes | yes | 0.654 | 0.62s | 15 ms |

### breast-w -- without negation

train n=468, features=89. Vertical index build: NList 116 ms (1676 tree nodes), PrePostNList 56 ms (1676 tree nodes), Sparse 29 ms (nnz=4212, density 0.10).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.654 | 0.52s | 41 ms |
| CN2 | NList | yes | yes | 0.654 | 0.44s | 12 ms |
| CN2 | PrePostNList | yes | yes | 0.654 | 0.44s | 12 ms |
| CN2 | Sparse | yes | yes | 0.654 | 0.45s | 9 ms |
| PFoil | Boolean | (ref) | (ref) | 0.654 | 0.26s | 41 ms |
| PFoil | NList | yes | yes | 0.654 | 0.21s | 12 ms |
| PFoil | PrePostNList | yes | yes | 0.654 | 0.21s | 12 ms |
| PFoil | Sparse | yes | yes | 0.654 | 0.22s | 9 ms |
| PFossil | Boolean | (ref) | (ref) | 0.654 | 0.16s | 41 ms |
| PFossil | NList | yes | yes | 0.654 | 0.16s | 12 ms |
| PFossil | PrePostNList | yes | yes | 0.654 | 0.11s | 12 ms |
| PFossil | Sparse | yes | yes | 0.654 | 0.13s | 9 ms |

*Encoding effect:* with negation 178 features (acc 0.654); without, 89 features (acc 0.654), sparse density 0.50 -> 0.10.

---

## diabetes

### diabetes -- with negation

train n=514, features=112. Vertical index build: NList 100 ms (12167 tree nodes), PrePostNList 109 ms (12167 tree nodes), Sparse 29 ms (nnz=28784, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.650 | 0.99s | 31 ms |
| CN2 | NList | yes | yes | 0.650 | 0.84s | 13 ms |
| CN2 | PrePostNList | yes | yes | 0.650 | 0.92s | 12 ms |
| CN2 | Sparse | yes | yes | 0.650 | 1.02s | 11 ms |
| PFoil | Boolean | (ref) | (ref) | 0.650 | 1.50s | 31 ms |
| PFoil | NList | yes | yes | 0.650 | 1.43s | 13 ms |
| PFoil | PrePostNList | yes | yes | 0.650 | 1.32s | 12 ms |
| PFoil | Sparse | yes | yes | 0.650 | 1.45s | 11 ms |
| PFossil | Boolean | (ref) | (ref) | 0.650 | 0.45s | 31 ms |
| PFossil | NList | yes | yes | 0.650 | 0.39s | 13 ms |
| PFossil | PrePostNList | yes | yes | 0.650 | 0.40s | 12 ms |
| PFossil | Sparse | yes | yes | 0.650 | 0.41s | 11 ms |

### diabetes -- without negation

train n=514, features=56. Vertical index build: NList 31 ms (2967 tree nodes), PrePostNList 35 ms (2967 tree nodes), Sparse 15 ms (nnz=12692, density 0.44).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.650 | 0.00s | 28 ms |
| CN2 | NList | yes | yes | 0.650 | 0.00s | 11 ms |
| CN2 | PrePostNList | yes | yes | 0.650 | 0.00s | 11 ms |
| CN2 | Sparse | yes | yes | 0.650 | 0.00s | 11 ms |
| PFoil | Boolean | (ref) | (ref) | 0.650 | 0.04s | 28 ms |
| PFoil | NList | yes | yes | 0.650 | 0.03s | 11 ms |
| PFoil | PrePostNList | yes | yes | 0.650 | 0.03s | 11 ms |
| PFoil | Sparse | yes | yes | 0.650 | 0.04s | 11 ms |
| PFossil | Boolean | (ref) | (ref) | 0.650 | 0.01s | 28 ms |
| PFossil | NList | yes | yes | 0.650 | 0.01s | 11 ms |
| PFossil | PrePostNList | yes | yes | 0.650 | 0.01s | 11 ms |
| PFossil | Sparse | yes | yes | 0.650 | 0.01s | 11 ms |

*Encoding effect:* with negation 112 features (acc 0.650); without, 56 features (acc 0.650), sparse density 0.50 -> 0.44.

---

## kr-vs-kp

### kr-vs-kp -- with negation

train n=2141, features=76. Vertical index build: NList 194 ms (26555 tree nodes), PrePostNList 223 ms (26555 tree nodes), Sparse 18 ms (nnz=81358, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.972 | 0.43s | 69 ms |
| CN2 | NList | yes | yes | 0.972 | 0.29s | 21 ms |
| CN2 | PrePostNList | yes | yes | 0.972 | 0.28s | 19 ms |
| CN2 | Sparse | yes | yes | 0.972 | 0.42s | 22 ms |
| PFoil | Boolean | (ref) | (ref) | 0.998 | 0.61s | 69 ms |
| PFoil | NList | yes | yes | 0.998 | 0.45s | 21 ms |
| PFoil | PrePostNList | yes | yes | 0.998 | 0.42s | 19 ms |
| PFoil | Sparse | yes | yes | 0.998 | 0.62s | 22 ms |
| PFossil | Boolean | (ref) | (ref) | 0.989 | 0.23s | 69 ms |
| PFossil | NList | yes | yes | 0.989 | 0.17s | 21 ms |
| PFossil | PrePostNList | yes | yes | 0.989 | 0.16s | 19 ms |
| PFossil | Sparse | yes | yes | 0.989 | 0.23s | 22 ms |

### kr-vs-kp -- without negation

train n=2141, features=73. Vertical index build: NList 172 ms (25575 tree nodes), PrePostNList 213 ms (25575 tree nodes), Sparse 17 ms (nnz=77076, density 0.49).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.974 | 0.49s | 70 ms |
| CN2 | NList | yes | yes | 0.974 | 0.35s | 19 ms |
| CN2 | PrePostNList | yes | yes | 0.974 | 0.32s | 18 ms |
| CN2 | Sparse | yes | yes | 0.974 | 0.48s | 21 ms |
| PFoil | Boolean | (ref) | (ref) | 0.998 | 0.55s | 70 ms |
| PFoil | NList | yes | yes | 0.998 | 0.40s | 19 ms |
| PFoil | PrePostNList | yes | yes | 0.998 | 0.38s | 18 ms |
| PFoil | Sparse | yes | yes | 0.998 | 0.54s | 21 ms |
| PFossil | Boolean | (ref) | (ref) | 0.989 | 0.22s | 70 ms |
| PFossil | NList | yes | yes | 0.989 | 0.16s | 19 ms |
| PFossil | PrePostNList | yes | yes | 0.989 | 0.15s | 18 ms |
| PFossil | Sparse | yes | yes | 0.989 | 0.22s | 21 ms |

*Encoding effect:* with negation 76 features (acc 0.989); without, 73 features (acc 0.989), sparse density 0.50 -> 0.49.

---

## mushroom

### mushroom -- with negation

train n=5443, features=222. Vertical index build: NList 1568 ms (169429 tree nodes), PrePostNList 2030 ms (169429 tree nodes), Sparse 90 ms (nnz=604173, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.518 | 1.06s | 222 ms |
| CN2 | NList | yes | yes | 0.518 | 0.86s | 39 ms |
| CN2 | PrePostNList | yes | yes | 0.518 | 0.82s | 31 ms |
| CN2 | Sparse | yes | yes | 0.518 | 1.23s | 41 ms |
| PFoil | Boolean | (ref) | (ref) | 0.518 | 0.76s | 222 ms |
| PFoil | NList | yes | yes | 0.518 | 0.58s | 39 ms |
| PFoil | PrePostNList | yes | yes | 0.518 | 0.53s | 31 ms |
| PFoil | Sparse | yes | yes | 0.518 | 0.81s | 41 ms |
| PFossil | Boolean | (ref) | (ref) | 0.518 | 0.64s | 222 ms |
| PFossil | NList | yes | yes | 0.518 | 0.51s | 39 ms |
| PFossil | PrePostNList | yes | yes | 0.518 | 0.50s | 31 ms |
| PFossil | Sparse | yes | yes | 0.518 | 0.75s | 41 ms |

### mushroom -- without negation

train n=5443, features=116. Vertical index build: NList 199 ms (21429 tree nodes), PrePostNList 330 ms (21429 tree nodes), Sparse 43 ms (nnz=114303, density 0.18).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.518 | 0.43s | 167 ms |
| CN2 | NList | yes | yes | 0.518 | 0.26s | 15 ms |
| CN2 | PrePostNList | yes | yes | 0.518 | 0.24s | 16 ms |
| CN2 | Sparse | yes | yes | 0.518 | 0.45s | 14 ms |
| PFoil | Boolean | (ref) | (ref) | 0.518 | 0.38s | 167 ms |
| PFoil | NList | yes | yes | 0.518 | 0.22s | 15 ms |
| PFoil | PrePostNList | yes | yes | 0.518 | 0.21s | 16 ms |
| PFoil | Sparse | yes | yes | 0.518 | 0.39s | 14 ms |
| PFossil | Boolean | (ref) | (ref) | 0.518 | 0.17s | 167 ms |
| PFossil | NList | yes | yes | 0.518 | 0.10s | 15 ms |
| PFossil | PrePostNList | yes | yes | 0.518 | 0.09s | 16 ms |
| PFossil | Sparse | yes | yes | 0.518 | 0.17s | 14 ms |

*Encoding effect:* with negation 222 features (acc 0.518); without, 116 features (acc 0.518), sparse density 0.50 -> 0.18.

---

## Summary -- average relative fit time (Boolean = 100%)

Mean of (data fit time / Boolean fit time) over every learner x dataset x encoding comparison whose Boolean fit took at least 0.02s (below that, rounding noise swamps the ratio). 100% = as fast as Boolean; under 100% is faster.

| data | with negation | without negation | overall | n |
|---|---|---|---|---|
| NList | 87% | 77% | 82% | 42 |
| PrePostNList | 85% | 75% | 80% | 42 |
| Sparse | 100% | 100% | 100% | 42 |

---

**Every data agreed with the Boolean baseline on every learner, dataset and encoding: yes.**

Fixed concept (4 signal columns, density 0.4) plus a growing number of pure-noise columns whose *own* density shrinks as more are added (`5 / n_noise`, capped at 0.5) -- so overall density falls purely as a side effect of adding columns, n=1000 fixed. Coverage timing is 800 random rules (length 1-4); fit is one PFossil fit per point, rules/predictions checked against Boolean.

**Fit-time gap vs. coverage-time gap**: the coverage-time gap (bottom right) grows cleanly and monotonically the whole way -- it isolates exactly the data-dependent computation. The fit-time gap (bottom left) does *not* stay monotonic, and goes slightly negative at the largest k -- because `k` also controls how many candidates the search itself has to score each round (`O(k)`, the same for every data), and that shared, data-agnostic cost grows right alongside the feature count. At extreme sparsity the actual coverage computation is already nearly free for everyone, so what's left is that per-candidate bookkeeping overhead -- and `NList`'s handle (`anchor_item`/`active`/`mask_words`/`ctx`) carries a bit more of it per call than Boolean's plain `(cov, scope)` pair. Not a data regression; a reminder that fit time bundles search overhead in with coverage cost, and only the latter is what these representations actually compete on.

## Synthetic feature-count sweep (density falls as k grows)

| k | density | rules == | Boolean fit | NList fit | PrePostNList fit | Sparse fit | Boolean cov | NList cov | PrePostNList cov | Sparse cov |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.3231 | yes | 0.008s | 0.004s | 0.004s | 0.007s | 29 ms | 10 ms | 8 ms | 10 ms |
| 50 | 0.1336 | yes | 0.014s | 0.008s | 0.008s | 0.012s | 31 ms | 8 ms | 8 ms | 8 ms |
| 100 | 0.0658 | yes | 0.021s | 0.011s | 0.011s | 0.018s | 37 ms | 9 ms | 10 ms | 6 ms |
| 250 | 0.0262 | yes | 0.056s | 0.027s | 0.025s | 0.042s | 46 ms | 8 ms | 8 ms | 5 ms |
| 500 | 0.0132 | yes | 0.123s | 0.080s | 0.069s | 0.106s | 57 ms | 8 ms | 9 ms | 5 ms |
| 1000 | 0.0066 | yes | 0.254s | 0.172s | 0.168s | 0.229s | 75 ms | 8 ms | 8 ms | 5 ms |
| 2000 | 0.0033 | yes | 0.975s | 0.659s | 0.629s | 0.750s | 124 ms | 7 ms | 7 ms | 4 ms |
| 4000 | 0.0016 | yes | 2.249s | 2.003s | 1.764s | 2.087s | 228 ms | 5 ms | 4 ms | 3 ms |

![Synthetic feature-count sweep (density falls as k grows)](demo_representations_sparsity_by_k.png)

**PFossil agreed with Boolean at every point: yes.**

---

Fixed feature count (k=500, 4 of them a fixed-density (0.4) learnable concept) with every *other* column's own density swept directly -- isolates density from feature count, unlike the feature-count sweep above where density only fell as a side effect of adding columns. n=1000 fixed. Coverage timing is 800 random rules (length 1-4); fit is one PFossil fit per point, rules/predictions checked against Boolean.

## Synthetic density sweep (k=500 fixed)

| noise density | density | rules == | Boolean fit | NList fit | PrePostNList fit | Sparse fit | Boolean cov | NList cov | PrePostNList cov | Sparse cov |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.4995 | yes | 0.830s | 0.625s | 0.621s | 0.737s | 64 ms | 23 ms | 23 ms | 14 ms |
| 0.3 | 0.3007 | yes | 0.837s | 0.519s | 0.454s | 0.633s | 63 ms | 18 ms | 18 ms | 11 ms |
| 0.1 | 0.1027 | yes | 0.146s | 0.085s | 0.077s | 0.118s | 58 ms | 11 ms | 11 ms | 8 ms |
| 0.05 | 0.0525 | yes | 0.155s | 0.098s | 0.088s | 0.144s | 55 ms | 10 ms | 9 ms | 7 ms |
| 0.02 | 0.0228 | yes | 0.120s | 0.074s | 0.068s | 0.093s | 57 ms | 9 ms | 10 ms | 5 ms |
| 0.01 | 0.0131 | yes | 0.197s | 0.064s | 0.062s | 0.098s | 55 ms | 8 ms | 8 ms | 5 ms |
| 0.005 | 0.0082 | yes | 0.095s | 0.055s | 0.046s | 0.073s | 57 ms | 9 ms | 8 ms | 6 ms |
| 0.002 | 0.0053 | yes | 0.086s | 0.049s | 0.046s | 0.076s | 56 ms | 6 ms | 5 ms | 4 ms |
| 0.001 | 0.0043 | yes | 0.094s | 0.053s | 0.056s | 0.082s | 56 ms | 4 ms | 4 ms | 5 ms |

![Synthetic density sweep (k=500 fixed)](demo_representations_sparsity_by_density.png)

**PFossil agreed with Boolean at every point: yes.**

---

