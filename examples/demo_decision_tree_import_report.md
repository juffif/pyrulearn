# Decision-tree -> rule-set import demo

Generated 2026-09-16 08:22:15. N_FOLDS=5, MAX_INTERVALS=6, MAX_DEPTH=5 (covtype: 8).

Only fold 1 of each dataset is shown in full detail (tree + every rule in all four `to_string` formats); all folds contribute to the summary statistics.

---

## vote

n=435, attributes=16 (16 categorical, 0 numeric), classes=2

### Fold 1 detail

```text
Decision tree (17 leaves, depth 5):
|--- physician-fee-freeze!=y <= 0.50
|   |--- synfuels-corporation-cutback=y <= 0.50
|   |   |--- export-administration-act-south-africa!=? <= 0.50
|   |   |   |--- mx-missile!=y <= 0.50
|   |   |   |   |--- class: democrat
|   |   |   |--- mx-missile!=y >  0.50
|   |   |   |   |--- adoption-of-the-budget-resolution=y <= 0.50
|   |   |   |   |   |--- class: republican
|   |   |   |   |--- adoption-of-the-budget-resolution=y >  0.50
|   |   |   |   |   |--- class: republican
|   |   |--- export-administration-act-south-africa!=? >  0.50
|   |   |   |--- class: republican
|   |--- synfuels-corporation-cutback=y >  0.50
|   |   |--- mx-missile!=y <= 0.50
|   |   |   |--- handicapped-infants!=y <= 0.50
|   |   |   |   |--- adoption-of-the-budget-resolution!=n <= 0.50
|   |   |   |   |   |--- class: democrat
|   |   |   |   |--- adoption-of-the-budget-resolution!=n >  0.50
|   |   |   |   |   |--- class: republican
|   |   |   |--- handicapped-infants!=y >  0.50
|   |   |   |   |--- class: democrat
|   |   |--- mx-missile!=y >  0.50
|   |   |   |--- adoption-of-the-budget-resolution=y <= 0.50
|   |   |   |   |--- superfund-right-to-sue!=y <= 0.50
|   |   |   |   |   |--- class: republican
|   |   |   |   |--- superfund-right-to-sue!=y >  0.50
|   |   |   |   |   |--- class: democrat
|   |   |   |--- adoption-of-the-budget-resolution=y >  0.50
|   |   |   |   |--- water-project-cost-sharing=n <= 0.50
|   |   |   |   |   |--- class: democrat
|   |   |   |   |--- water-project-cost-sharing=n >  0.50
|   |   |   |   |   |--- class: republican
|--- physician-fee-freeze!=y >  0.50
|   |--- adoption-of-the-budget-resolution=? <= 0.50
|   |   |--- education-spending=? <= 0.50
|   |   |   |--- class: democrat
|   |   |--- education-spending=? >  0.50
|   |   |   |--- adoption-of-the-budget-resolution!=n <= 0.50
|   |   |   |   |--- class: republican
|   |   |   |--- adoption-of-the-budget-resolution!=n >  0.50
|   |   |   |   |--- class: democrat
|   |--- adoption-of-the-budget-resolution=? >  0.50
|   |   |--- education-spending!=y <= 0.50
|   |   |   |--- export-administration-act-south-africa=y <= 0.50
|   |   |   |   |--- class: republican
|   |   |   |--- export-administration-act-south-africa=y >  0.50
|   |   |   |   |--- class: democrat
|   |   |--- education-spending!=y >  0.50
|   |   |   |--- class: democrat

Extracted DisjointRuleSet (17 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 9/16

logic:      physician-fee-freeze = y ∧ mx-missile = y ∧ synfuels-corporation-cutback ≠ y ∧ export-administration-act-south-africa = ? → democrat
prolog:     democrat(X) :- physician-fee-freeze(X, y), mx-missile(X, y), synfuels-corporation-cutback(X, V), V \= y, export-administration-act-south-africa(X, ?).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: physician-fee-freeze = y, mx-missile = y, synfuels-corporation-cutback ≠ y, export-administration-act-south-africa = ?

logic:      adoption-of-the-budget-resolution ≠ y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback ≠ y ∧ export-administration-act-south-africa = ? → republican
prolog:     republican(X) :- adoption-of-the-budget-resolution(X, V), V \= y, physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, V), V \= y, export-administration-act-south-africa(X, ?).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: adoption-of-the-budget-resolution ≠ y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback ≠ y, export-administration-act-south-africa = ?

logic:      adoption-of-the-budget-resolution = y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback ≠ y ∧ export-administration-act-south-africa = ? → republican
prolog:     republican(X) :- adoption-of-the-budget-resolution(X, y), physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, V), V \= y, export-administration-act-south-africa(X, ?).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: adoption-of-the-budget-resolution = y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback ≠ y, export-administration-act-south-africa = ?

logic:      physician-fee-freeze = y ∧ synfuels-corporation-cutback ≠ y ∧ export-administration-act-south-africa ≠ ? → republican
prolog:     republican(X) :- physician-fee-freeze(X, y), synfuels-corporation-cutback(X, V), V \= y, export-administration-act-south-africa(X, V), V \= ?.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
conditions: physician-fee-freeze = y, synfuels-corporation-cutback ≠ y, export-administration-act-south-africa ≠ ?

logic:      handicapped-infants = y ∧ adoption-of-the-budget-resolution = n ∧ physician-fee-freeze = y ∧ mx-missile = y ∧ synfuels-corporation-cutback = y → democrat
prolog:     democrat(X) :- handicapped-infants(X, y), adoption-of-the-budget-resolution(X, n), physician-fee-freeze(X, y), mx-missile(X, y), synfuels-corporation-cutback(X, y).
pattern:    0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: handicapped-infants = y, adoption-of-the-budget-resolution = n, physician-fee-freeze = y, mx-missile = y, synfuels-corporation-cutback = y

logic:      handicapped-infants = y ∧ adoption-of-the-budget-resolution ≠ n ∧ physician-fee-freeze = y ∧ mx-missile = y ∧ synfuels-corporation-cutback = y → republican
prolog:     republican(X) :- handicapped-infants(X, y), adoption-of-the-budget-resolution(X, V), V \= n, physician-fee-freeze(X, y), mx-missile(X, y), synfuels-corporation-cutback(X, y).
pattern:    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: handicapped-infants = y, adoption-of-the-budget-resolution ≠ n, physician-fee-freeze = y, mx-missile = y, synfuels-corporation-cutback = y

logic:      handicapped-infants ≠ y ∧ physician-fee-freeze = y ∧ mx-missile = y ∧ synfuels-corporation-cutback = y → democrat
prolog:     democrat(X) :- handicapped-infants(X, V), V \= y, physician-fee-freeze(X, y), mx-missile(X, y), synfuels-corporation-cutback(X, y).
pattern:    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: handicapped-infants ≠ y, physician-fee-freeze = y, mx-missile = y, synfuels-corporation-cutback = y

logic:      adoption-of-the-budget-resolution ≠ y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback = y ∧ superfund-right-to-sue = y → republican
prolog:     republican(X) :- adoption-of-the-budget-resolution(X, V), V \= y, physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, y), superfund-right-to-sue(X, y).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution ≠ y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback = y, superfund-right-to-sue = y

logic:      adoption-of-the-budget-resolution ≠ y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback = y ∧ superfund-right-to-sue ≠ y → democrat
prolog:     democrat(X) :- adoption-of-the-budget-resolution(X, V), V \= y, physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, y), superfund-right-to-sue(X, V), V \= y.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution ≠ y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback = y, superfund-right-to-sue ≠ y

logic:      water-project-cost-sharing ≠ n ∧ adoption-of-the-budget-resolution = y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback = y → democrat
prolog:     democrat(X) :- water-project-cost-sharing(X, V), V \= n, adoption-of-the-budget-resolution(X, y), physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, y).
pattern:    0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: water-project-cost-sharing ≠ n, adoption-of-the-budget-resolution = y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback = y

logic:      water-project-cost-sharing = n ∧ adoption-of-the-budget-resolution = y ∧ physician-fee-freeze = y ∧ mx-missile ≠ y ∧ synfuels-corporation-cutback = y → republican
prolog:     republican(X) :- water-project-cost-sharing(X, n), adoption-of-the-budget-resolution(X, y), physician-fee-freeze(X, y), mx-missile(X, V), V \= y, synfuels-corporation-cutback(X, y).
pattern:    0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: water-project-cost-sharing = n, adoption-of-the-budget-resolution = y, physician-fee-freeze = y, mx-missile ≠ y, synfuels-corporation-cutback = y

logic:      adoption-of-the-budget-resolution ≠ ? ∧ physician-fee-freeze ≠ y ∧ education-spending ≠ ? → democrat
prolog:     democrat(X) :- adoption-of-the-budget-resolution(X, V), V \= ?, physician-fee-freeze(X, V), V \= y, education-spending(X, V), V \= ?.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution ≠ ?, physician-fee-freeze ≠ y, education-spending ≠ ?

logic:      adoption-of-the-budget-resolution = n ∧ adoption-of-the-budget-resolution ≠ ? ∧ physician-fee-freeze ≠ y ∧ education-spending = ? → republican
prolog:     republican(X) :- adoption-of-the-budget-resolution(X, n), adoption-of-the-budget-resolution(X, V), V \= ?, physician-fee-freeze(X, V), V \= y, education-spending(X, ?).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution = n, adoption-of-the-budget-resolution ≠ ?, physician-fee-freeze ≠ y, education-spending = ?

logic:      adoption-of-the-budget-resolution ≠ ? ∧ adoption-of-the-budget-resolution ≠ n ∧ physician-fee-freeze ≠ y ∧ education-spending = ? → democrat
prolog:     democrat(X) :- adoption-of-the-budget-resolution(X, V), V \= ?, adoption-of-the-budget-resolution(X, V), V \= n, physician-fee-freeze(X, V), V \= y, education-spending(X, ?).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution ≠ ?, adoption-of-the-budget-resolution ≠ n, physician-fee-freeze ≠ y, education-spending = ?

logic:      adoption-of-the-budget-resolution = ? ∧ physician-fee-freeze ≠ y ∧ education-spending = y ∧ export-administration-act-south-africa ≠ y → republican
prolog:     republican(X) :- adoption-of-the-budget-resolution(X, ?), physician-fee-freeze(X, V), V \= y, education-spending(X, y), export-administration-act-south-africa(X, V), V \= y.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
conditions: adoption-of-the-budget-resolution = ?, physician-fee-freeze ≠ y, education-spending = y, export-administration-act-south-africa ≠ y

logic:      adoption-of-the-budget-resolution = ? ∧ physician-fee-freeze ≠ y ∧ education-spending = y ∧ export-administration-act-south-africa = y → democrat
prolog:     democrat(X) :- adoption-of-the-budget-resolution(X, ?), physician-fee-freeze(X, V), V \= y, education-spending(X, y), export-administration-act-south-africa(X, y).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
conditions: adoption-of-the-budget-resolution = ?, physician-fee-freeze ≠ y, education-spending = y, export-administration-act-south-africa = y

logic:      adoption-of-the-budget-resolution = ? ∧ physician-fee-freeze ≠ y ∧ education-spending ≠ y → democrat
prolog:     democrat(X) :- adoption-of-the-budget-resolution(X, ?), physician-fee-freeze(X, V), V \= y, education-spending(X, V), V \= y.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: adoption-of-the-budget-resolution = ?, physician-fee-freeze ≠ y, education-spending ≠ y

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.920 | 0.920 | 1.000 | 17 | 9 |
| 2 | 0.966 | 0.966 | 1.000 | 18 | 10 |
| 3 | 0.954 | 0.954 | 1.000 | 19 | 12 |
| 4 | 0.954 | 0.954 | 1.000 | 20 | 11 |
| 5 | 0.897 | 0.897 | 1.000 | 18 | 11 |

**Summary** (0.1s): tree accuracy 0.938 +/- 0.026, rule-set accuracy 0.938 +/- 0.026, tree/rule-set agreement 1.0000, mean rules/fold 18.4, mean attributes used/fold 10.6 of 16 available.

---

## Overall summary

`categ.`/`numeric` = number of original attributes of each type (after dropping signal-free columns, see module docstring); `attrs used` = mean number of those original attributes actually referenced by the extracted rules, out of `categ.` + `numeric` available -- not the same as the number of *derived* Boolean features the rules are built from. Since the DisjointRuleSet is a one-to-one re-expression of the tree (one literal per split, no simplification), this is also exactly the number of attributes the tree itself splits on -- `score_fold` asserts the two counts match on every fold, so `attrs used` doubles as that count.

| dataset | n | categ. | numeric | tree acc | rule acc | agree | rules/fold | attrs used/fold | time(s) |
|---|---|---|---|---|---|---|---|---|---|
| vote | 435 | 16 | 0 | 0.938±0.026 | 0.938±0.026 | 1.000 | 18.4 | 10.6 | 0.1 |
