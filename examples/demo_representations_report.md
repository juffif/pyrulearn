# Four `DataRepresentation` encodings -- same rules, with & without negation

Generated 2026-09-24 22:20:51. Single stratified 67/33 split, `random_state=0`. Each learner is fit on all four representations (Boolean = packed bit-matrix, NList = PPC-tree / N-list, PrePostNList = NList + pre/post visit codes on its search fast path, Sparse = scipy CSR/CSC = the N-list without the prefix tree) and the rule sets / prediction vectors are compared to the Boolean baseline exactly. Every dataset is run in two feature encodings: **with negation** (paired negation features) and **without** (positive tests only). `coverage` timing is 1000 random rules (length 1-4).

CN2 / PFoil / PFossil run everywhere; AQR / PyLORD only on vote & tic-tac-toe. PrePostNList is included to *demonstrate* it stays correct, not because it's expected to be faster here -- see `PrePostNListRepresentation`'s own docstring for why it measured out roughly break-even to slightly worse than plain NList at these data sizes.

---

## vote

### vote -- with negation

train n=291, features=96. Vertical index build: NList 43 ms (6863 tree nodes), PrePostNList 54 ms (6863 tree nodes), Sparse 12 ms (nnz=13968, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.09s | 24 ms |
| CN2 | NList | yes | yes | 0.611 | 0.08s | 11 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.08s | 10 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.08s | 10 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.11s | 24 ms |
| PFoil | NList | yes | yes | 0.611 | 0.11s | 11 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.10s | 10 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.11s | 10 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.14s | 24 ms |
| PFossil | NList | yes | yes | 0.611 | 0.14s | 11 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.13s | 10 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.14s | 10 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 2.45s | 24 ms |
| AQR | NList | yes | yes | 0.611 | 2.31s | 11 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 2.24s | 10 ms |
| AQR | Sparse | yes | yes | 0.611 | 2.43s | 10 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.944 | 3.33s | 24 ms |
| PyLORD | NList | yes | yes | 0.944 | 2.95s | 11 ms |
| PyLORD | PrePostNList | yes | yes | 0.944 | 2.78s | 10 ms |
| PyLORD | Sparse | yes | yes | 0.944 | 3.26s | 10 ms |

### vote -- without negation

train n=291, features=48. Vertical index build: NList 26 ms (2415 tree nodes), PrePostNList 32 ms (2415 tree nodes), Sparse 10 ms (nnz=4656, density 0.33).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.05s | 37 ms |
| CN2 | NList | yes | yes | 0.611 | 0.04s | 15 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.04s | 13 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.04s | 14 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.08s | 37 ms |
| PFoil | NList | yes | yes | 0.611 | 0.07s | 15 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.06s | 13 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.06s | 14 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.07s | 37 ms |
| PFossil | NList | yes | yes | 0.611 | 0.07s | 15 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.07s | 13 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.08s | 14 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 0.28s | 37 ms |
| AQR | NList | yes | yes | 0.611 | 0.25s | 15 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 0.26s | 13 ms |
| AQR | Sparse | yes | yes | 0.611 | 0.27s | 14 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.951 | 0.74s | 37 ms |
| PyLORD | NList | yes | yes | 0.951 | 0.61s | 15 ms |
| PyLORD | PrePostNList | yes | yes | 0.951 | 0.58s | 13 ms |
| PyLORD | Sparse | yes | yes | 0.951 | 0.65s | 14 ms |

*Encoding effect:* with negation 96 features (acc 0.944); without, 48 features (acc 0.951), sparse density 0.50 -> 0.33.

---

## tic-tac-toe

### tic-tac-toe -- with negation

train n=641, features=54. Vertical index build: NList 63 ms (12190 tree nodes), PrePostNList 124 ms (12190 tree nodes), Sparse 11 ms (nnz=17307, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.981 | 0.45s | 49 ms |
| CN2 | NList | yes | yes | 0.981 | 0.42s | 16 ms |
| CN2 | PrePostNList | yes | yes | 0.981 | 0.34s | 14 ms |
| CN2 | Sparse | yes | yes | 0.981 | 0.37s | 19 ms |
| PFoil | Boolean | (ref) | (ref) | 0.890 | 0.25s | 49 ms |
| PFoil | NList | yes | yes | 0.890 | 0.21s | 16 ms |
| PFoil | PrePostNList | yes | yes | 0.890 | 0.21s | 14 ms |
| PFoil | Sparse | yes | yes | 0.890 | 0.23s | 19 ms |
| PFossil | Boolean | (ref) | (ref) | 0.830 | 0.17s | 49 ms |
| PFossil | NList | yes | yes | 0.830 | 0.14s | 16 ms |
| PFossil | PrePostNList | yes | yes | 0.830 | 0.13s | 14 ms |
| PFossil | Sparse | yes | yes | 0.830 | 0.16s | 19 ms |
| AQR | Boolean | (ref) | (ref) | 0.861 | 4.18s | 49 ms |
| AQR | NList | yes | yes | 0.861 | 3.45s | 16 ms |
| AQR | PrePostNList | yes | yes | 0.861 | 3.70s | 14 ms |
| AQR | Sparse | yes | yes | 0.861 | 4.04s | 19 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.965 | 5.97s | 49 ms |
| PyLORD | NList | yes | yes | 0.965 | 4.48s | 16 ms |
| PyLORD | PrePostNList | yes | yes | 0.965 | 4.61s | 14 ms |
| PyLORD | Sparse | yes | yes | 0.965 | 5.94s | 19 ms |

### tic-tac-toe -- without negation

train n=641, features=27. Vertical index build: NList 28 ms (3249 tree nodes), PrePostNList 29 ms (3249 tree nodes), Sparse 5 ms (nnz=5769, density 0.33).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.981 | 0.16s | 42 ms |
| CN2 | NList | yes | yes | 0.981 | 0.11s | 11 ms |
| CN2 | PrePostNList | yes | yes | 0.981 | 0.12s | 10 ms |
| CN2 | Sparse | yes | yes | 0.981 | 0.14s | 18 ms |
| PFoil | Boolean | (ref) | (ref) | 0.950 | 0.09s | 42 ms |
| PFoil | NList | yes | yes | 0.950 | 0.07s | 11 ms |
| PFoil | PrePostNList | yes | yes | 0.950 | 0.08s | 10 ms |
| PFoil | Sparse | yes | yes | 0.950 | 0.08s | 18 ms |
| PFossil | Boolean | (ref) | (ref) | 0.830 | 0.06s | 42 ms |
| PFossil | NList | yes | yes | 0.830 | 0.04s | 11 ms |
| PFossil | PrePostNList | yes | yes | 0.830 | 0.04s | 10 ms |
| PFossil | Sparse | yes | yes | 0.830 | 0.05s | 18 ms |
| AQR | Boolean | (ref) | (ref) | 0.962 | 0.18s | 42 ms |
| AQR | NList | yes | yes | 0.962 | 0.14s | 11 ms |
| AQR | PrePostNList | yes | yes | 0.962 | 0.13s | 10 ms |
| AQR | Sparse | yes | yes | 0.962 | 0.17s | 18 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.972 | 1.67s | 42 ms |
| PyLORD | NList | yes | yes | 0.972 | 1.13s | 11 ms |
| PyLORD | PrePostNList | yes | yes | 0.972 | 1.18s | 10 ms |
| PyLORD | Sparse | yes | yes | 0.972 | 1.49s | 18 ms |

*Encoding effect:* with negation 54 features (acc 0.965); without, 27 features (acc 0.972), sparse density 0.50 -> 0.33.

---

## breast-w

### breast-w -- with negation

train n=468, features=178. Vertical index build: NList 158 ms (14185 tree nodes), PrePostNList 205 ms (14185 tree nodes), Sparse 51 ms (nnz=41652, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.654 | 0.74s | 48 ms |
| CN2 | NList | yes | yes | 0.654 | 0.66s | 17 ms |
| CN2 | PrePostNList | yes | yes | 0.654 | 0.72s | 17 ms |
| CN2 | Sparse | yes | yes | 0.654 | 0.70s | 16 ms |
| PFoil | Boolean | (ref) | (ref) | 0.654 | 0.66s | 48 ms |
| PFoil | NList | yes | yes | 0.654 | 0.58s | 17 ms |
| PFoil | PrePostNList | yes | yes | 0.654 | 0.57s | 17 ms |
| PFoil | Sparse | yes | yes | 0.654 | 0.59s | 16 ms |
| PFossil | Boolean | (ref) | (ref) | 0.654 | 0.58s | 48 ms |
| PFossil | NList | yes | yes | 0.654 | 0.54s | 17 ms |
| PFossil | PrePostNList | yes | yes | 0.654 | 0.53s | 17 ms |
| PFossil | Sparse | yes | yes | 0.654 | 0.58s | 16 ms |

### breast-w -- without negation

train n=468, features=89. Vertical index build: NList 39 ms (1676 tree nodes), PrePostNList 46 ms (1676 tree nodes), Sparse 26 ms (nnz=4212, density 0.10).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.654 | 0.44s | 41 ms |
| CN2 | NList | yes | yes | 0.654 | 0.33s | 15 ms |
| CN2 | PrePostNList | yes | yes | 0.654 | 0.36s | 15 ms |
| CN2 | Sparse | yes | yes | 0.654 | 0.37s | 10 ms |
| PFoil | Boolean | (ref) | (ref) | 0.654 | 0.22s | 41 ms |
| PFoil | NList | yes | yes | 0.654 | 0.18s | 15 ms |
| PFoil | PrePostNList | yes | yes | 0.654 | 0.19s | 15 ms |
| PFoil | Sparse | yes | yes | 0.654 | 0.20s | 10 ms |
| PFossil | Boolean | (ref) | (ref) | 0.654 | 0.16s | 41 ms |
| PFossil | NList | yes | yes | 0.654 | 0.14s | 15 ms |
| PFossil | PrePostNList | yes | yes | 0.654 | 0.13s | 15 ms |
| PFossil | Sparse | yes | yes | 0.654 | 0.14s | 10 ms |

*Encoding effect:* with negation 178 features (acc 0.654); without, 89 features (acc 0.654), sparse density 0.50 -> 0.10.

---

## diabetes

### diabetes -- with negation

train n=514, features=112. Vertical index build: NList 132 ms (12167 tree nodes), PrePostNList 169 ms (12167 tree nodes), Sparse 36 ms (nnz=28784, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.650 | 1.20s | 48 ms |
| CN2 | NList | yes | yes | 0.650 | 1.06s | 18 ms |
| CN2 | PrePostNList | yes | yes | 0.650 | 1.05s | 18 ms |
| CN2 | Sparse | yes | yes | 0.650 | 1.13s | 15 ms |
| PFoil | Boolean | (ref) | (ref) | 0.650 | 1.92s | 48 ms |
| PFoil | NList | yes | yes | 0.650 | 1.78s | 18 ms |
| PFoil | PrePostNList | yes | yes | 0.650 | 1.73s | 18 ms |
| PFoil | Sparse | yes | yes | 0.650 | 1.81s | 15 ms |
| PFossil | Boolean | (ref) | (ref) | 0.650 | 0.58s | 48 ms |
| PFossil | NList | yes | yes | 0.650 | 0.55s | 18 ms |
| PFossil | PrePostNList | yes | yes | 0.650 | 0.53s | 18 ms |
| PFossil | Sparse | yes | yes | 0.650 | 0.55s | 15 ms |

### diabetes -- without negation

train n=514, features=56. Vertical index build: NList 48 ms (2967 tree nodes), PrePostNList 52 ms (2967 tree nodes), Sparse 21 ms (nnz=12692, density 0.44).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.650 | 0.01s | 42 ms |
| CN2 | NList | yes | yes | 0.650 | 0.00s | 16 ms |
| CN2 | PrePostNList | yes | yes | 0.650 | 0.00s | 16 ms |
| CN2 | Sparse | yes | yes | 0.650 | 0.01s | 17 ms |
| PFoil | Boolean | (ref) | (ref) | 0.650 | 0.05s | 42 ms |
| PFoil | NList | yes | yes | 0.650 | 0.03s | 16 ms |
| PFoil | PrePostNList | yes | yes | 0.650 | 0.05s | 16 ms |
| PFoil | Sparse | yes | yes | 0.650 | 0.05s | 17 ms |
| PFossil | Boolean | (ref) | (ref) | 0.650 | 0.01s | 42 ms |
| PFossil | NList | yes | yes | 0.650 | 0.01s | 16 ms |
| PFossil | PrePostNList | yes | yes | 0.650 | 0.01s | 16 ms |
| PFossil | Sparse | yes | yes | 0.650 | 0.01s | 17 ms |

*Encoding effect:* with negation 112 features (acc 0.650); without, 56 features (acc 0.650), sparse density 0.50 -> 0.44.

---

## kr-vs-kp

### kr-vs-kp -- with negation

train n=2141, features=76. Vertical index build: NList 209 ms (26555 tree nodes), PrePostNList 310 ms (26555 tree nodes), Sparse 24 ms (nnz=81358, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.972 | 0.59s | 91 ms |
| CN2 | NList | yes | yes | 0.972 | 0.42s | 28 ms |
| CN2 | PrePostNList | yes | yes | 0.972 | 0.37s | 26 ms |
| CN2 | Sparse | yes | yes | 0.972 | 0.52s | 31 ms |
| PFoil | Boolean | (ref) | (ref) | 0.998 | 0.84s | 91 ms |
| PFoil | NList | yes | yes | 0.998 | 0.55s | 28 ms |
| PFoil | PrePostNList | yes | yes | 0.998 | 0.54s | 26 ms |
| PFoil | Sparse | yes | yes | 0.998 | 0.82s | 31 ms |
| PFossil | Boolean | (ref) | (ref) | 0.989 | 0.30s | 91 ms |
| PFossil | NList | yes | yes | 0.989 | 0.24s | 28 ms |
| PFossil | PrePostNList | yes | yes | 0.989 | 0.21s | 26 ms |
| PFossil | Sparse | yes | yes | 0.989 | 0.30s | 31 ms |

### kr-vs-kp -- without negation

train n=2141, features=73. Vertical index build: NList 232 ms (25575 tree nodes), PrePostNList 285 ms (25575 tree nodes), Sparse 23 ms (nnz=77076, density 0.49).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.974 | 0.72s | 98 ms |
| CN2 | NList | yes | yes | 0.974 | 0.43s | 27 ms |
| CN2 | PrePostNList | yes | yes | 0.974 | 0.44s | 27 ms |
| CN2 | Sparse | yes | yes | 0.974 | 0.63s | 30 ms |
| PFoil | Boolean | (ref) | (ref) | 0.998 | 0.72s | 98 ms |
| PFoil | NList | yes | yes | 0.998 | 0.55s | 27 ms |
| PFoil | PrePostNList | yes | yes | 0.998 | 0.53s | 27 ms |
| PFoil | Sparse | yes | yes | 0.998 | 0.69s | 30 ms |
| PFossil | Boolean | (ref) | (ref) | 0.989 | 0.30s | 98 ms |
| PFossil | NList | yes | yes | 0.989 | 0.23s | 27 ms |
| PFossil | PrePostNList | yes | yes | 0.989 | 0.20s | 27 ms |
| PFossil | Sparse | yes | yes | 0.989 | 0.30s | 30 ms |

*Encoding effect:* with negation 76 features (acc 0.989); without, 73 features (acc 0.989), sparse density 0.50 -> 0.49.

---

## mushroom

### mushroom -- with negation

train n=5443, features=222. Vertical index build: NList 2211 ms (169429 tree nodes), PrePostNList 2516 ms (169429 tree nodes), Sparse 108 ms (nnz=604173, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.518 | 1.35s | 304 ms |
| CN2 | NList | yes | yes | 0.518 | 1.14s | 42 ms |
| CN2 | PrePostNList | yes | yes | 0.518 | 1.01s | 42 ms |
| CN2 | Sparse | yes | yes | 0.518 | 1.61s | 52 ms |
| PFoil | Boolean | (ref) | (ref) | 0.518 | 0.95s | 304 ms |
| PFoil | NList | yes | yes | 0.518 | 0.73s | 42 ms |
| PFoil | PrePostNList | yes | yes | 0.518 | 0.74s | 42 ms |
| PFoil | Sparse | yes | yes | 0.518 | 1.07s | 52 ms |
| PFossil | Boolean | (ref) | (ref) | 0.518 | 0.84s | 304 ms |
| PFossil | NList | yes | yes | 0.518 | 0.71s | 42 ms |
| PFossil | PrePostNList | yes | yes | 0.518 | 0.67s | 42 ms |
| PFossil | Sparse | yes | yes | 0.518 | 1.00s | 52 ms |

### mushroom -- without negation

train n=5443, features=116. Vertical index build: NList 269 ms (21429 tree nodes), PrePostNList 423 ms (21429 tree nodes), Sparse 54 ms (nnz=114303, density 0.18).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.518 | 0.53s | 183 ms |
| CN2 | NList | yes | yes | 0.518 | 0.37s | 20 ms |
| CN2 | PrePostNList | yes | yes | 0.518 | 0.33s | 19 ms |
| CN2 | Sparse | yes | yes | 0.518 | 0.60s | 18 ms |
| PFoil | Boolean | (ref) | (ref) | 0.518 | 0.51s | 183 ms |
| PFoil | NList | yes | yes | 0.518 | 0.28s | 20 ms |
| PFoil | PrePostNList | yes | yes | 0.518 | 0.29s | 19 ms |
| PFoil | Sparse | yes | yes | 0.518 | 0.48s | 18 ms |
| PFossil | Boolean | (ref) | (ref) | 0.518 | 0.24s | 183 ms |
| PFossil | NList | yes | yes | 0.518 | 0.14s | 20 ms |
| PFossil | PrePostNList | yes | yes | 0.518 | 0.14s | 19 ms |
| PFossil | Sparse | yes | yes | 0.518 | 0.23s | 18 ms |

*Encoding effect:* with negation 222 features (acc 0.518); without, 116 features (acc 0.518), sparse density 0.50 -> 0.18.

---

## Summary -- average relative fit time (Boolean = 100%)

Mean of (data fit time / Boolean fit time) over every learner x dataset x encoding comparison whose Boolean fit took at least 0.02s (below that, rounding noise swamps the ratio). 100% = as fast as Boolean; under 100% is faster.

| data | with negation | without negation | overall | n |
|---|---|---|---|---|
| NList | 86% | 76% | 81% | 42 |
| PrePostNList | 83% | 76% | 80% | 42 |
| Sparse | 98% | 93% | 96% | 42 |

---

**Every data agreed with the Boolean baseline on every learner, dataset and encoding: yes.**

Fixed concept (4 signal columns, density 0.4) plus a growing number of pure-noise columns whose *own* density shrinks as more are added (`5 / n_noise`, capped at 0.5) -- so overall density falls purely as a side effect of adding columns, n=1000 fixed. Coverage timing is 800 random rules (length 1-4); fit is one PFossil fit per point, rules/predictions checked against Boolean.

**Fit-time gap vs. coverage-time gap**: the coverage-time gap (bottom right) grows cleanly and monotonically the whole way -- it isolates exactly the data-dependent computation. The fit-time gap (bottom left) does *not* stay monotonic, and goes slightly negative at the largest k -- because `k` also controls how many candidates the search itself has to score each round (`O(k)`, the same for every data), and that shared, data-agnostic cost grows right alongside the feature count. At extreme sparsity the actual coverage computation is already nearly free for everyone, so what's left is that per-candidate bookkeeping overhead -- and `NList`'s handle (`anchor_item`/`active`/`mask_words`/`ctx`) carries a bit more of it per call than Boolean's plain `(cov, scope)` pair. Not a data regression; a reminder that fit time bundles search overhead in with coverage cost, and only the latter is what these representations actually compete on.

## Synthetic feature-count sweep (density falls as k grows)

| k | density | rules == | Boolean fit | NList fit | PrePostNList fit | Sparse fit | Boolean cov | NList cov | PrePostNList cov | Sparse cov |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 0.3231 | yes | 0.010s | 0.006s | 0.006s | 0.009s | 43 ms | 13 ms | 13 ms | 16 ms |
| 50 | 0.1336 | yes | 0.018s | 0.011s | 0.011s | 0.016s | 44 ms | 11 ms | 12 ms | 12 ms |
| 100 | 0.0658 | yes | 0.020s | 0.018s | 0.014s | 0.019s | 48 ms | 13 ms | 8 ms | 8 ms |
| 250 | 0.0262 | yes | 0.068s | 0.040s | 0.036s | 0.049s | 56 ms | 12 ms | 8 ms | 4 ms |
| 500 | 0.0132 | yes | 0.164s | 0.101s | 0.063s | 0.128s | 73 ms | 13 ms | 8 ms | 4 ms |
| 1000 | 0.0066 | yes | 0.290s | 0.224s | 0.183s | 0.280s | 106 ms | 12 ms | 11 ms | 7 ms |
| 2000 | 0.0033 | yes | 1.212s | 0.804s | 0.741s | 0.905s | 150 ms | 9 ms | 8 ms | 7 ms |
| 4000 | 0.0016 | yes | 2.798s | 2.255s | 2.122s | 2.467s | 274 ms | 7 ms | 7 ms | 4 ms |

![Synthetic feature-count sweep (density falls as k grows)](demo_representations_sparsity_by_k.png)

**PFossil agreed with Boolean at every point: yes.**

---

Fixed feature count (k=500, 4 of them a fixed-density (0.4) learnable concept) with every *other* column's own density swept directly -- isolates density from feature count, unlike the feature-count sweep above where density only fell as a side effect of adding columns. n=1000 fixed. Coverage timing is 800 random rules (length 1-4); fit is one PFossil fit per point, rules/predictions checked against Boolean.

## Synthetic density sweep (k=500 fixed)

| noise density | density | rules == | Boolean fit | NList fit | PrePostNList fit | Sparse fit | Boolean cov | NList cov | PrePostNList cov | Sparse cov |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.4995 | yes | 1.159s | 0.840s | 0.780s | 0.959s | 85 ms | 33 ms | 28 ms | 21 ms |
| 0.3 | 0.3007 | yes | 0.969s | 0.706s | 0.619s | 0.821s | 76 ms | 20 ms | 24 ms | 16 ms |
| 0.1 | 0.1027 | yes | 0.184s | 0.105s | 0.100s | 0.138s | 77 ms | 17 ms | 15 ms | 10 ms |
| 0.05 | 0.0525 | yes | 0.189s | 0.130s | 0.115s | 0.180s | 76 ms | 13 ms | 13 ms | 10 ms |
| 0.02 | 0.0228 | yes | 0.151s | 0.091s | 0.073s | 0.119s | 67 ms | 14 ms | 11 ms | 5 ms |
| 0.01 | 0.0131 | yes | 0.144s | 0.089s | 0.083s | 0.197s | 77 ms | 12 ms | 13 ms | 7 ms |
| 0.005 | 0.0082 | yes | 0.125s | 0.069s | 0.067s | 0.103s | 64 ms | 13 ms | 10 ms | 8 ms |
| 0.002 | 0.0053 | yes | 0.121s | 0.064s | 0.064s | 0.096s | 75 ms | 9 ms | 8 ms | 6 ms |
| 0.001 | 0.0043 | yes | 0.113s | 0.060s | 0.054s | 0.097s | 78 ms | 5 ms | 3 ms | 3 ms |

![Synthetic density sweep (k=500 fixed)](demo_representations_sparsity_by_density.png)

**PFossil agreed with Boolean at every point: yes.**

---

