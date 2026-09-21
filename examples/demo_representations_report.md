# Four `DataRepresentation` encodings -- same rules, with & without negation

Generated 2026-09-16 08:16:53. Single stratified 67/33 split, `random_state=0`. Each learner is fit on all four representations (Boolean = packed bit-matrix, NList = PPC-tree / N-list, PrePostNList = NList + pre/post visit codes on its search fast path, Sparse = scipy CSR/CSC = the N-list without the prefix tree) and the rule sets / prediction vectors are compared to the Boolean baseline exactly. Every dataset is run in two feature encodings: **with negation** (paired negation features) and **without** (positive tests only). `coverage` timing is 1000 random rules (length 1-4).

CN2 / PFoil / PFossil run everywhere; AQR / PyLORD only on vote & tic-tac-toe. PrePostNList is included to *demonstrate* it stays correct, not because it's expected to be faster here -- see `PrePostNListRepresentation`'s own docstring for why it measured out roughly break-even to slightly worse than plain NList at these data sizes.

---

## vote

### vote -- with negation

train n=291, features=96. Vertical index build: NList 40 ms (6863 tree nodes), PrePostNList 48 ms (6863 tree nodes), Sparse 11 ms (nnz=13968, density 0.50).

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.09s | 23 ms |
| CN2 | NList | yes | yes | 0.611 | 0.08s | 10 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.08s | 10 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.08s | 9 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.11s | 23 ms |
| PFoil | NList | yes | yes | 0.611 | 0.10s | 10 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.10s | 10 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.10s | 9 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.14s | 23 ms |
| PFossil | NList | yes | yes | 0.611 | 0.13s | 10 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.13s | 10 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.13s | 9 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 1.59s | 23 ms |
| AQR | NList | yes | yes | 0.611 | 1.54s | 10 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 1.62s | 10 ms |
| AQR | Sparse | yes | yes | 0.611 | 1.66s | 9 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.944 | 2.12s | 23 ms |
| PyLORD | NList | yes | yes | 0.944 | 1.98s | 10 ms |
| PyLORD | PrePostNList | yes | yes | 0.944 | 1.84s | 10 ms |
| PyLORD | Sparse | yes | yes | 0.944 | 2.13s | 9 ms |

### vote -- without negation

train n=291, features=48. Vertical index build: NList 17 ms (2415 tree nodes), PrePostNList 22 ms (2415 tree nodes), Sparse 7 ms (nnz=4656, density 0.33).

`.without_negations()` on the negated representations matches this from-scratch build: **yes**.

| learner | data | rules == | preds == | acc | fit | 1k coverage |
|---|---|---|---|---|---|---|
| CN2 | Boolean | (ref) | (ref) | 0.611 | 0.03s | 21 ms |
| CN2 | NList | yes | yes | 0.611 | 0.03s | 9 ms |
| CN2 | PrePostNList | yes | yes | 0.611 | 0.03s | 9 ms |
| CN2 | Sparse | yes | yes | 0.611 | 0.03s | 8 ms |
| PFoil | Boolean | (ref) | (ref) | 0.611 | 0.05s | 21 ms |
| PFoil | NList | yes | yes | 0.611 | 0.04s | 9 ms |
| PFoil | PrePostNList | yes | yes | 0.611 | 0.04s | 9 ms |
| PFoil | Sparse | yes | yes | 0.611 | 0.05s | 8 ms |
| PFossil | Boolean | (ref) | (ref) | 0.611 | 0.05s | 21 ms |
| PFossil | NList | yes | yes | 0.611 | 0.05s | 9 ms |
| PFossil | PrePostNList | yes | yes | 0.611 | 0.04s | 9 ms |
| PFossil | Sparse | yes | yes | 0.611 | 0.05s | 8 ms |
| AQR | Boolean | (ref) | (ref) | 0.611 | 0.18s | 21 ms |
| AQR | NList | yes | yes | 0.611 | 0.16s | 9 ms |
| AQR | PrePostNList | yes | yes | 0.611 | 0.16s | 9 ms |
| AQR | Sparse | yes | yes | 0.611 | 0.17s | 8 ms |
| PyLORD | Boolean | (ref) | (ref) | 0.951 | 0.49s | 21 ms |
| PyLORD | NList | yes | yes | 0.951 | 0.39s | 9 ms |
| PyLORD | PrePostNList | yes | yes | 0.951 | 0.38s | 9 ms |
| PyLORD | Sparse | yes | yes | 0.951 | 0.46s | 8 ms |

*Encoding effect:* with negation 96 features (acc 0.944); without, 48 features (acc 0.951), sparse density 0.50 -> 0.33.

---

## Summary -- average relative fit time (Boolean = 100%)

Mean of (data fit time / Boolean fit time) over every learner x dataset x encoding comparison whose Boolean fit took at least 0.02s (below that, rounding noise swamps the ratio). 100% = as fast as Boolean; under 100% is faster.

| data | with negation | without negation | overall | n |
|---|---|---|---|---|
| NList | 94% | 89% | 92% | 10 |
| PrePostNList | 92% | 85% | 88% | 10 |
| Sparse | 97% | 94% | 95% | 10 |

---

**Every data agreed with the Boolean baseline on every learner, dataset and encoding: yes.**

