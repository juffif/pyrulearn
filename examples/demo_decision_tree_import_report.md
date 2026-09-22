# Decision-tree -> rule-set import demo

Generated 2026-09-22 21:39:28. N_FOLDS=5, MAX_INTERVALS=6, MAX_DEPTH=5 (covtype: 8).

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

**Summary** (0.2s): tree accuracy 0.938 +/- 0.026, rule-set accuracy 0.938 +/- 0.026, tree/rule-set agreement 1.0000, mean rules/fold 18.4, mean attributes used/fold 10.6 of 16 available.

---

## breast-cancer

n=286, attributes=9 (9 categorical, 0 numeric), classes=2

### Fold 1 detail

```text
validate_dataspec note(s) against the test fold: ["attribute 'age' has values in the data outside its known domain: ['20-29']"]

Decision tree (19 leaves, depth 5):
|--- deg-malig=3 <= 0.50
|   |--- tumor-size=10-14 <= 0.50
|   |   |--- breast-quad=right_low <= 0.50
|   |   |   |--- node-caps=no <= 0.50
|   |   |   |   |--- menopause=lt40 <= 0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |   |--- menopause=lt40 >  0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |   |--- node-caps=no >  0.50
|   |   |   |   |--- tumor-size=45-49 <= 0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |   |--- tumor-size=45-49 >  0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |--- breast-quad=right_low >  0.50
|   |   |   |--- class: no-recurrence-events
|   |--- tumor-size=10-14 >  0.50
|   |   |--- class: no-recurrence-events
|--- deg-malig=3 >  0.50
|   |--- inv-nodes=0-2 <= 0.50
|   |   |--- node-caps!=? <= 0.50
|   |   |   |--- class: no-recurrence-events
|   |   |--- node-caps!=? >  0.50
|   |   |   |--- breast!=left <= 0.50
|   |   |   |   |--- breast-quad=central <= 0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |   |   |--- breast-quad=central >  0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |--- breast!=left >  0.50
|   |   |   |   |--- tumor-size=30-34 <= 0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |   |--- tumor-size=30-34 >  0.50
|   |   |   |   |   |--- class: recurrence-events
|   |--- inv-nodes=0-2 >  0.50
|   |   |--- age=60-69 <= 0.50
|   |   |   |--- menopause!=premeno <= 0.50
|   |   |   |   |--- age=40-49 <= 0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |   |   |--- age=40-49 >  0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |--- menopause!=premeno >  0.50
|   |   |   |   |--- breast-quad=? <= 0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |   |--- breast-quad=? >  0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |--- age=60-69 >  0.50
|   |   |   |--- breast-quad!=left_low <= 0.50
|   |   |   |   |--- tumor-size=20-24 <= 0.50
|   |   |   |   |   |--- class: no-recurrence-events
|   |   |   |   |--- tumor-size=20-24 >  0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |   |--- breast-quad!=left_low >  0.50
|   |   |   |   |--- tumor-size=25-29 <= 0.50
|   |   |   |   |   |--- class: recurrence-events
|   |   |   |   |--- tumor-size=25-29 >  0.50
|   |   |   |   |   |--- class: no-recurrence-events

Extracted DisjointRuleSet (19 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 8/9

logic:      menopause ≠ lt40 ∧ tumor-size ≠ 10-14 ∧ node-caps ≠ no ∧ deg-malig ≠ 3 ∧ breast-quad ≠ right_low → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- menopause(X, V), V \= lt40, tumor-size(X, V), V \= 10-14, node-caps(X, V), V \= no, deg-malig(X, V), V \= 3, breast-quad(X, V), V \= right_low.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: menopause ≠ lt40, tumor-size ≠ 10-14, node-caps ≠ no, deg-malig ≠ 3, breast-quad ≠ right_low

logic:      menopause = lt40 ∧ tumor-size ≠ 10-14 ∧ node-caps ≠ no ∧ deg-malig ≠ 3 ∧ breast-quad ≠ right_low → recurrence-events
prolog:     'recurrence-events'(X) :- menopause(X, lt40), tumor-size(X, V), V \= 10-14, node-caps(X, V), V \= no, deg-malig(X, V), V \= 3, breast-quad(X, V), V \= right_low.
pattern:    0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: menopause = lt40, tumor-size ≠ 10-14, node-caps ≠ no, deg-malig ≠ 3, breast-quad ≠ right_low

logic:      tumor-size ≠ 10-14 ∧ tumor-size ≠ 45-49 ∧ node-caps = no ∧ deg-malig ≠ 3 ∧ breast-quad ≠ right_low → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- tumor-size(X, V), V \= 10-14, tumor-size(X, V), V \= 45-49, node-caps(X, no), deg-malig(X, V), V \= 3, breast-quad(X, V), V \= right_low.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: tumor-size ≠ 10-14, tumor-size ≠ 45-49, node-caps = no, deg-malig ≠ 3, breast-quad ≠ right_low

logic:      tumor-size = 45-49 ∧ tumor-size ≠ 10-14 ∧ node-caps = no ∧ deg-malig ≠ 3 ∧ breast-quad ≠ right_low → recurrence-events
prolog:     'recurrence-events'(X) :- tumor-size(X, 45-49), tumor-size(X, V), V \= 10-14, node-caps(X, no), deg-malig(X, V), V \= 3, breast-quad(X, V), V \= right_low.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: tumor-size = 45-49, tumor-size ≠ 10-14, node-caps = no, deg-malig ≠ 3, breast-quad ≠ right_low

logic:      tumor-size ≠ 10-14 ∧ deg-malig ≠ 3 ∧ breast-quad = right_low → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- tumor-size(X, V), V \= 10-14, deg-malig(X, V), V \= 3, breast-quad(X, right_low).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
conditions: tumor-size ≠ 10-14, deg-malig ≠ 3, breast-quad = right_low

logic:      tumor-size = 10-14 ∧ deg-malig ≠ 3 → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- tumor-size(X, 10-14), deg-malig(X, V), V \= 3.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: tumor-size = 10-14, deg-malig ≠ 3

logic:      inv-nodes ≠ 0-2 ∧ node-caps = ? ∧ deg-malig = 3 → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- inv-nodes(X, V), V \= 0-2, node-caps(X, ?), deg-malig(X, 3).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: inv-nodes ≠ 0-2, node-caps = ?, deg-malig = 3

logic:      inv-nodes ≠ 0-2 ∧ node-caps ≠ ? ∧ deg-malig = 3 ∧ breast = left ∧ breast-quad ≠ central → recurrence-events
prolog:     'recurrence-events'(X) :- inv-nodes(X, V), V \= 0-2, node-caps(X, V), V \= ?, deg-malig(X, 3), breast(X, left), breast-quad(X, V), V \= central.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
conditions: inv-nodes ≠ 0-2, node-caps ≠ ?, deg-malig = 3, breast = left, breast-quad ≠ central

logic:      inv-nodes ≠ 0-2 ∧ node-caps ≠ ? ∧ deg-malig = 3 ∧ breast = left ∧ breast-quad = central → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- inv-nodes(X, V), V \= 0-2, node-caps(X, V), V \= ?, deg-malig(X, 3), breast(X, left), breast-quad(X, central).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: inv-nodes ≠ 0-2, node-caps ≠ ?, deg-malig = 3, breast = left, breast-quad = central

logic:      tumor-size ≠ 30-34 ∧ inv-nodes ≠ 0-2 ∧ node-caps ≠ ? ∧ deg-malig = 3 ∧ breast ≠ left → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- tumor-size(X, V), V \= 30-34, inv-nodes(X, V), V \= 0-2, node-caps(X, V), V \= ?, deg-malig(X, 3), breast(X, V), V \= left.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: tumor-size ≠ 30-34, inv-nodes ≠ 0-2, node-caps ≠ ?, deg-malig = 3, breast ≠ left

logic:      tumor-size = 30-34 ∧ inv-nodes ≠ 0-2 ∧ node-caps ≠ ? ∧ deg-malig = 3 ∧ breast ≠ left → recurrence-events
prolog:     'recurrence-events'(X) :- tumor-size(X, 30-34), inv-nodes(X, V), V \= 0-2, node-caps(X, V), V \= ?, deg-malig(X, 3), breast(X, V), V \= left.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: tumor-size = 30-34, inv-nodes ≠ 0-2, node-caps ≠ ?, deg-malig = 3, breast ≠ left

logic:      age ≠ 40-49 ∧ age ≠ 60-69 ∧ menopause = premeno ∧ inv-nodes = 0-2 ∧ deg-malig = 3 → recurrence-events
prolog:     'recurrence-events'(X) :- age(X, V), V \= 40-49, age(X, V), V \= 60-69, menopause(X, premeno), inv-nodes(X, 0-2), deg-malig(X, 3).
pattern:    0 0 0 0 0 0 1 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: age ≠ 40-49, age ≠ 60-69, menopause = premeno, inv-nodes = 0-2, deg-malig = 3

logic:      age = 40-49 ∧ age ≠ 60-69 ∧ menopause = premeno ∧ inv-nodes = 0-2 ∧ deg-malig = 3 → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- age(X, 40-49), age(X, V), V \= 60-69, menopause(X, premeno), inv-nodes(X, 0-2), deg-malig(X, 3).
pattern:    0 1 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: age = 40-49, age ≠ 60-69, menopause = premeno, inv-nodes = 0-2, deg-malig = 3

logic:      age ≠ 60-69 ∧ menopause ≠ premeno ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad ≠ ? → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- age(X, V), V \= 60-69, menopause(X, V), V \= premeno, inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, V), V \= ?.
pattern:    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
conditions: age ≠ 60-69, menopause ≠ premeno, inv-nodes = 0-2, deg-malig = 3, breast-quad ≠ ?

logic:      age ≠ 60-69 ∧ menopause ≠ premeno ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad = ? → recurrence-events
prolog:     'recurrence-events'(X) :- age(X, V), V \= 60-69, menopause(X, V), V \= premeno, inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, ?).
pattern:    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: age ≠ 60-69, menopause ≠ premeno, inv-nodes = 0-2, deg-malig = 3, breast-quad = ?

logic:      age = 60-69 ∧ tumor-size ≠ 20-24 ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad = left_low → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- age(X, 60-69), tumor-size(X, V), V \= 20-24, inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, left_low).
pattern:    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: age = 60-69, tumor-size ≠ 20-24, inv-nodes = 0-2, deg-malig = 3, breast-quad = left_low

logic:      age = 60-69 ∧ tumor-size = 20-24 ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad = left_low → recurrence-events
prolog:     'recurrence-events'(X) :- age(X, 60-69), tumor-size(X, 20-24), inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, left_low).
pattern:    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: age = 60-69, tumor-size = 20-24, inv-nodes = 0-2, deg-malig = 3, breast-quad = left_low

logic:      age = 60-69 ∧ tumor-size ≠ 25-29 ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad ≠ left_low → recurrence-events
prolog:     'recurrence-events'(X) :- age(X, 60-69), tumor-size(X, V), V \= 25-29, inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, V), V \= left_low.
pattern:    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
conditions: age = 60-69, tumor-size ≠ 25-29, inv-nodes = 0-2, deg-malig = 3, breast-quad ≠ left_low

logic:      age = 60-69 ∧ tumor-size = 25-29 ∧ inv-nodes = 0-2 ∧ deg-malig = 3 ∧ breast-quad ≠ left_low → no-recurrence-events
prolog:     'no-recurrence-events'(X) :- age(X, 60-69), tumor-size(X, 25-29), inv-nodes(X, 0-2), deg-malig(X, 3), breast-quad(X, V), V \= left_low.
pattern:    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
conditions: age = 60-69, tumor-size = 25-29, inv-nodes = 0-2, deg-malig = 3, breast-quad ≠ left_low

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.741 | 0.741 | 1.000 | 19 | 8 |
| 2 | 0.737 | 0.737 | 1.000 | 20 | 9 |
| 3 | 0.719 | 0.719 | 1.000 | 18 | 7 |
| 4 | 0.702 | 0.702 | 1.000 | 19 | 7 |
| 5 | 0.719 | 0.719 | 1.000 | 19 | 7 |

**Summary** (0.2s): tree accuracy 0.724 +/- 0.014, rule-set accuracy 0.724 +/- 0.014, tree/rule-set agreement 1.0000, mean rules/fold 19.0, mean attributes used/fold 7.6 of 9 available.

---

## kr-vs-kp

n=3196, attributes=36 (36 categorical, 0 numeric), classes=2

### Fold 1 detail

```text
validate_dataspec note(s) against the test fold: ["attribute 'spcop' has values in the data outside its known domain: ['t']"]

Decision tree (11 leaves, depth 5):
|--- rimmx=f <= 0.50
|   |--- class: won
|--- rimmx=f >  0.50
|   |--- wknck=t <= 0.50
|   |   |--- bxqsq!=t <= 0.50
|   |   |   |--- class: nowin
|   |   |--- bxqsq!=t >  0.50
|   |   |   |--- wkna8!=t <= 0.50
|   |   |   |   |--- class: nowin
|   |   |   |--- wkna8!=t >  0.50
|   |   |   |   |--- bkxbq!=f <= 0.50
|   |   |   |   |   |--- class: won
|   |   |   |   |--- bkxbq!=f >  0.50
|   |   |   |   |   |--- class: won
|   |--- wknck=t >  0.50
|   |   |--- r2ar8!=t <= 0.50
|   |   |   |--- wkovl=t <= 0.50
|   |   |   |   |--- bkxcr=f <= 0.50
|   |   |   |   |   |--- class: nowin
|   |   |   |   |--- bkxcr=f >  0.50
|   |   |   |   |   |--- class: nowin
|   |   |   |--- wkovl=t >  0.50
|   |   |   |   |--- class: nowin
|   |   |--- r2ar8!=t >  0.50
|   |   |   |--- bkxcr!=f <= 0.50
|   |   |   |   |--- skrxp=f <= 0.50
|   |   |   |   |   |--- class: nowin
|   |   |   |   |--- skrxp=f >  0.50
|   |   |   |   |   |--- class: nowin
|   |   |   |--- bkxcr!=f >  0.50
|   |   |   |   |--- class: nowin

Extracted DisjointRuleSet (11 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 9/36

logic:      rimmx ≠ f → won
prolog:     won(X) :- rimmx(X, V), V \= f.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: rimmx ≠ f

logic:      bxqsq = t ∧ rimmx = f ∧ wknck ≠ t → nowin
prolog:     nowin(X) :- bxqsq(X, t), rimmx(X, f), wknck(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bxqsq = t, rimmx = f, wknck ≠ t

logic:      bxqsq ≠ t ∧ rimmx = f ∧ wkna8 = t ∧ wknck ≠ t → nowin
prolog:     nowin(X) :- bxqsq(X, V), V \= t, rimmx(X, f), wkna8(X, t), wknck(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bxqsq ≠ t, rimmx = f, wkna8 = t, wknck ≠ t

logic:      bkxbq = f ∧ bxqsq ≠ t ∧ rimmx = f ∧ wkna8 ≠ t ∧ wknck ≠ t → won
prolog:     won(X) :- bkxbq(X, f), bxqsq(X, V), V \= t, rimmx(X, f), wkna8(X, V), V \= t, wknck(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bkxbq = f, bxqsq ≠ t, rimmx = f, wkna8 ≠ t, wknck ≠ t

logic:      bkxbq ≠ f ∧ bxqsq ≠ t ∧ rimmx = f ∧ wkna8 ≠ t ∧ wknck ≠ t → won
prolog:     won(X) :- bkxbq(X, V), V \= f, bxqsq(X, V), V \= t, rimmx(X, f), wkna8(X, V), V \= t, wknck(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bkxbq ≠ f, bxqsq ≠ t, rimmx = f, wkna8 ≠ t, wknck ≠ t

logic:      bkxcr ≠ f ∧ r2ar8 = t ∧ rimmx = f ∧ wknck = t ∧ wkovl ≠ t → nowin
prolog:     nowin(X) :- bkxcr(X, V), V \= f, r2ar8(X, t), rimmx(X, f), wknck(X, t), wkovl(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0
conditions: bkxcr ≠ f, r2ar8 = t, rimmx = f, wknck = t, wkovl ≠ t

logic:      bkxcr = f ∧ r2ar8 = t ∧ rimmx = f ∧ wknck = t ∧ wkovl ≠ t → nowin
prolog:     nowin(X) :- bkxcr(X, f), r2ar8(X, t), rimmx(X, f), wknck(X, t), wkovl(X, V), V \= t.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0
conditions: bkxcr = f, r2ar8 = t, rimmx = f, wknck = t, wkovl ≠ t

logic:      r2ar8 = t ∧ rimmx = f ∧ wknck = t ∧ wkovl = t → nowin
prolog:     nowin(X) :- r2ar8(X, t), rimmx(X, f), wknck(X, t), wkovl(X, t).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0
conditions: r2ar8 = t, rimmx = f, wknck = t, wkovl = t

logic:      bkxcr = f ∧ r2ar8 ≠ t ∧ rimmx = f ∧ skrxp ≠ f ∧ wknck = t → nowin
prolog:     nowin(X) :- bkxcr(X, f), r2ar8(X, V), V \= t, rimmx(X, f), skrxp(X, V), V \= f, wknck(X, t).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bkxcr = f, r2ar8 ≠ t, rimmx = f, skrxp ≠ f, wknck = t

logic:      bkxcr = f ∧ r2ar8 ≠ t ∧ rimmx = f ∧ skrxp = f ∧ wknck = t → nowin
prolog:     nowin(X) :- bkxcr(X, f), r2ar8(X, V), V \= t, rimmx(X, f), skrxp(X, f), wknck(X, t).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bkxcr = f, r2ar8 ≠ t, rimmx = f, skrxp = f, wknck = t

logic:      bkxcr ≠ f ∧ r2ar8 ≠ t ∧ rimmx = f ∧ wknck = t → nowin
prolog:     nowin(X) :- bkxcr(X, V), V \= f, r2ar8(X, V), V \= t, rimmx(X, f), wknck(X, t).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: bkxcr ≠ f, r2ar8 ≠ t, rimmx = f, wknck = t

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.922 | 0.922 | 1.000 | 11 | 9 |
| 2 | 0.945 | 0.945 | 1.000 | 12 | 10 |
| 3 | 0.939 | 0.939 | 1.000 | 10 | 9 |
| 4 | 0.942 | 0.942 | 1.000 | 12 | 9 |
| 5 | 0.956 | 0.956 | 1.000 | 12 | 10 |

**Summary** (0.5s): tree accuracy 0.941 +/- 0.011, rule-set accuracy 0.941 +/- 0.011, tree/rule-set agreement 1.0000, mean rules/fold 11.4, mean attributes used/fold 9.4 of 36 available.

---

## hypothyroid

n=3772, attributes=27 (21 categorical, 6 numeric), classes=4

### Fold 1 detail

```text
Decision tree (13 leaves, depth 5):
|--- TSH>=6.05 <= 0.50
|   |--- TT4<51.5 <= 0.50
|   |   |--- class: negative
|   |--- TT4<51.5 >  0.50
|   |   |--- referral_source=other <= 0.50
|   |   |   |--- class: negative
|   |   |--- referral_source=other >  0.50
|   |   |   |--- sex!=M <= 0.50
|   |   |   |   |--- class: negative
|   |   |   |--- sex!=M >  0.50
|   |   |   |   |--- class: secondary_hypothyroid
|--- TSH>=6.05 >  0.50
|   |--- FTI>=61.5 <= 0.50
|   |   |--- thyroid_surgery=t <= 0.50
|   |   |   |--- T3>=2.55 <= 0.50
|   |   |   |   |--- TSH>=8.55 <= 0.50
|   |   |   |   |   |--- class: primary_hypothyroid
|   |   |   |   |--- TSH>=8.55 >  0.50
|   |   |   |   |   |--- class: primary_hypothyroid
|   |   |   |--- T3>=2.55 >  0.50
|   |   |   |   |--- class: negative
|   |   |--- thyroid_surgery=t >  0.50
|   |   |   |--- T3>=1.25 <= 0.50
|   |   |   |   |--- class: negative
|   |   |   |--- T3>=1.25 >  0.50
|   |   |   |   |--- class: primary_hypothyroid
|   |--- FTI>=61.5 >  0.50
|   |   |--- on_thyroxine=f <= 0.50
|   |   |   |--- class: negative
|   |   |--- on_thyroxine=f >  0.50
|   |   |   |--- thyroid_surgery=t <= 0.50
|   |   |   |   |--- TT4>=51.5 <= 0.50
|   |   |   |   |   |--- class: primary_hypothyroid
|   |   |   |   |--- TT4>=51.5 >  0.50
|   |   |   |   |   |--- class: compensated_hypothyroid
|   |   |   |--- thyroid_surgery=t >  0.50
|   |   |   |   |--- class: negative

Extracted DisjointRuleSet (13 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 8/27

logic:      TSH < 6.05 ∧ TT4 >= 51.5 → negative
prolog:     negative(X) :- TSH(X, V), V < 6.05, TT4(X, V), V >= 51.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: TSH < 6.05, TT4 >= 51.5

logic:      TSH < 6.05 ∧ TT4 < 51.5 ∧ referral_source ≠ other → negative
prolog:     negative(X) :- TSH(X, V), V < 6.05, TT4(X, V), V < 51.5, referral_source(X, V), V \= other.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
conditions: TSH < 6.05, TT4 < 51.5, referral_source ≠ other

logic:      sex = M ∧ TSH < 6.05 ∧ TT4 < 51.5 ∧ referral_source = other → negative
prolog:     negative(X) :- sex(X, M), TSH(X, V), V < 6.05, TT4(X, V), V < 51.5, referral_source(X, other).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: sex = M, TSH < 6.05, TT4 < 51.5, referral_source = other

logic:      sex ≠ M ∧ TSH < 6.05 ∧ TT4 < 51.5 ∧ referral_source = other → secondary_hypothyroid
prolog:     secondary_hypothyroid(X) :- sex(X, V), V \= M, TSH(X, V), V < 6.05, TT4(X, V), V < 51.5, referral_source(X, other).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
conditions: sex ≠ M, TSH < 6.05, TT4 < 51.5, referral_source = other

logic:      thyroid_surgery ≠ t ∧ TSH >= 6.05 ∧ TSH < 8.55 ∧ T3 < 2.55 ∧ FTI < 61.5 → primary_hypothyroid
prolog:     primary_hypothyroid(X) :- thyroid_surgery(X, V), V \= t, TSH(X, V), V >= 6.05, TSH(X, V), V < 8.55, T3(X, V), V < 2.55, FTI(X, V), V < 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: thyroid_surgery ≠ t, TSH >= 6.05, TSH < 8.55, T3 < 2.55, FTI < 61.5

logic:      thyroid_surgery ≠ t ∧ TSH >= 6.05 ∧ TSH >= 8.55 ∧ T3 < 2.55 ∧ FTI < 61.5 → primary_hypothyroid
prolog:     primary_hypothyroid(X) :- thyroid_surgery(X, V), V \= t, TSH(X, V), V >= 6.05, TSH(X, V), V >= 8.55, T3(X, V), V < 2.55, FTI(X, V), V < 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: thyroid_surgery ≠ t, TSH >= 6.05, TSH >= 8.55, T3 < 2.55, FTI < 61.5

logic:      thyroid_surgery ≠ t ∧ TSH >= 6.05 ∧ T3 >= 2.55 ∧ FTI < 61.5 → negative
prolog:     negative(X) :- thyroid_surgery(X, V), V \= t, TSH(X, V), V >= 6.05, T3(X, V), V >= 2.55, FTI(X, V), V < 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: thyroid_surgery ≠ t, TSH >= 6.05, T3 >= 2.55, FTI < 61.5

logic:      thyroid_surgery = t ∧ TSH >= 6.05 ∧ T3 < 1.25 ∧ FTI < 61.5 → negative
prolog:     negative(X) :- thyroid_surgery(X, t), TSH(X, V), V >= 6.05, T3(X, V), V < 1.25, FTI(X, V), V < 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: thyroid_surgery = t, TSH >= 6.05, T3 < 1.25, FTI < 61.5

logic:      thyroid_surgery = t ∧ TSH >= 6.05 ∧ T3 >= 1.25 ∧ FTI < 61.5 → primary_hypothyroid
prolog:     primary_hypothyroid(X) :- thyroid_surgery(X, t), TSH(X, V), V >= 6.05, T3(X, V), V >= 1.25, FTI(X, V), V < 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
conditions: thyroid_surgery = t, TSH >= 6.05, T3 >= 1.25, FTI < 61.5

logic:      on_thyroxine ≠ f ∧ TSH >= 6.05 ∧ FTI >= 61.5 → negative
prolog:     negative(X) :- on_thyroxine(X, V), V \= f, TSH(X, V), V >= 6.05, FTI(X, V), V >= 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: on_thyroxine ≠ f, TSH >= 6.05, FTI >= 61.5

logic:      on_thyroxine = f ∧ thyroid_surgery ≠ t ∧ TSH >= 6.05 ∧ TT4 < 51.5 ∧ FTI >= 61.5 → primary_hypothyroid
prolog:     primary_hypothyroid(X) :- on_thyroxine(X, f), thyroid_surgery(X, V), V \= t, TSH(X, V), V >= 6.05, TT4(X, V), V < 51.5, FTI(X, V), V >= 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: on_thyroxine = f, thyroid_surgery ≠ t, TSH >= 6.05, TT4 < 51.5, FTI >= 61.5

logic:      on_thyroxine = f ∧ thyroid_surgery ≠ t ∧ TSH >= 6.05 ∧ TT4 >= 51.5 ∧ FTI >= 61.5 → compensated_hypothyroid
prolog:     compensated_hypothyroid(X) :- on_thyroxine(X, f), thyroid_surgery(X, V), V \= t, TSH(X, V), V >= 6.05, TT4(X, V), V >= 51.5, FTI(X, V), V >= 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: on_thyroxine = f, thyroid_surgery ≠ t, TSH >= 6.05, TT4 >= 51.5, FTI >= 61.5

logic:      on_thyroxine = f ∧ thyroid_surgery = t ∧ TSH >= 6.05 ∧ FTI >= 61.5 → negative
prolog:     negative(X) :- on_thyroxine(X, f), thyroid_surgery(X, t), TSH(X, V), V >= 6.05, FTI(X, V), V >= 61.5.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: on_thyroxine = f, thyroid_surgery = t, TSH >= 6.05, FTI >= 61.5

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.993 | 0.993 | 1.000 | 13 | 8 |
| 2 | 0.988 | 0.988 | 1.000 | 11 | 7 |
| 3 | 0.991 | 0.991 | 1.000 | 15 | 8 |
| 4 | 0.995 | 0.995 | 1.000 | 15 | 8 |
| 5 | 0.995 | 0.995 | 1.000 | 15 | 8 |

**Summary** (1.0s): tree accuracy 0.992 +/- 0.003, rule-set accuracy 0.992 +/- 0.003, tree/rule-set agreement 1.0000, mean rules/fold 13.8, mean attributes used/fold 7.8 of 27 available.

---

## soybean

n=683, attributes=35 (35 categorical, 0 numeric), classes=19

### Fold 1 detail

```text
Decision tree (14 leaves, depth 5):
|--- leafspot-size=gt-1/8 <= 0.50
|   |--- canker-lesion!=dk-brown-blk <= 0.50
|   |   |--- fruit-spots=brown-w/blk-specks <= 0.50
|   |   |   |--- fruit-spots!=absent <= 0.50
|   |   |   |   |--- class: anthracnose
|   |   |   |--- fruit-spots!=absent >  0.50
|   |   |   |   |--- class: phytophthora-rot
|   |   |--- fruit-spots=brown-w/blk-specks >  0.50
|   |   |   |--- class: anthracnose
|   |--- canker-lesion!=dk-brown-blk >  0.50
|   |   |--- int-discolor!=brown <= 0.50
|   |   |   |--- class: brown-stem-rot
|   |   |--- int-discolor!=brown >  0.50
|   |   |   |--- stem-cankers!=below-soil <= 0.50
|   |   |   |   |--- class: rhizoctonia-root-rot
|   |   |   |--- stem-cankers!=below-soil >  0.50
|   |   |   |   |--- sclerotia!=present <= 0.50
|   |   |   |   |   |--- class: charcoal-rot
|   |   |   |   |--- sclerotia!=present >  0.50
|   |   |   |   |   |--- class: bacterial-blight
|--- leafspot-size=gt-1/8 >  0.50
|   |--- fruit-pods=diseased <= 0.50
|   |   |--- fruiting-bodies!=present <= 0.50
|   |   |   |--- class: brown-spot
|   |   |--- fruiting-bodies!=present >  0.50
|   |   |   |--- leaf-mild!=lower-surf <= 0.50
|   |   |   |   |--- class: downy-mildew
|   |   |   |--- leaf-mild!=lower-surf >  0.50
|   |   |   |   |--- date!=june <= 0.50
|   |   |   |   |   |--- class: brown-spot
|   |   |   |   |--- date!=june >  0.50
|   |   |   |   |   |--- class: alternarialeaf-spot
|   |--- fruit-pods=diseased >  0.50
|   |   |--- fruit-spots=colored <= 0.50
|   |   |   |--- germination=90-100 <= 0.50
|   |   |   |   |--- class: brown-spot
|   |   |   |--- germination=90-100 >  0.50
|   |   |   |   |--- class: frog-eye-leaf-spot
|   |   |--- fruit-spots=colored >  0.50
|   |   |   |--- class: frog-eye-leaf-spot

Extracted DisjointRuleSet (14 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 11/35

logic:      leafspot-size ≠ gt-1/8 ∧ canker-lesion = dk-brown-blk ∧ fruit-spots = absent ∧ fruit-spots ≠ brown-w/blk-specks → anthracnose
prolog:     anthracnose(X) :- leafspot-size(X, V), V \= gt-1/8, canker-lesion(X, dk-brown-blk), fruit-spots(X, absent), fruit-spots(X, V), V \= brown-w/blk-specks.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, canker-lesion = dk-brown-blk, fruit-spots = absent, fruit-spots ≠ brown-w/blk-specks

logic:      leafspot-size ≠ gt-1/8 ∧ canker-lesion = dk-brown-blk ∧ fruit-spots ≠ absent ∧ fruit-spots ≠ brown-w/blk-specks → phytophthora-rot
prolog:     'phytophthora-rot'(X) :- leafspot-size(X, V), V \= gt-1/8, canker-lesion(X, dk-brown-blk), fruit-spots(X, V), V \= absent, fruit-spots(X, V), V \= brown-w/blk-specks.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, canker-lesion = dk-brown-blk, fruit-spots ≠ absent, fruit-spots ≠ brown-w/blk-specks

logic:      leafspot-size ≠ gt-1/8 ∧ canker-lesion = dk-brown-blk ∧ fruit-spots = brown-w/blk-specks → anthracnose
prolog:     anthracnose(X) :- leafspot-size(X, V), V \= gt-1/8, canker-lesion(X, dk-brown-blk), fruit-spots(X, brown-w/blk-specks).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, canker-lesion = dk-brown-blk, fruit-spots = brown-w/blk-specks

logic:      leafspot-size ≠ gt-1/8 ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor = brown → brown-stem-rot
prolog:     'brown-stem-rot'(X) :- leafspot-size(X, V), V \= gt-1/8, canker-lesion(X, V), V \= dk-brown-blk, int-discolor(X, brown).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, canker-lesion ≠ dk-brown-blk, int-discolor = brown

logic:      leafspot-size ≠ gt-1/8 ∧ stem-cankers = below-soil ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor ≠ brown → rhizoctonia-root-rot
prolog:     'rhizoctonia-root-rot'(X) :- leafspot-size(X, V), V \= gt-1/8, stem-cankers(X, below-soil), canker-lesion(X, V), V \= dk-brown-blk, int-discolor(X, V), V \= brown.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, stem-cankers = below-soil, canker-lesion ≠ dk-brown-blk, int-discolor ≠ brown

logic:      leafspot-size ≠ gt-1/8 ∧ stem-cankers ≠ below-soil ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor ≠ brown ∧ sclerotia = present → charcoal-rot
prolog:     'charcoal-rot'(X) :- leafspot-size(X, V), V \= gt-1/8, stem-cankers(X, V), V \= below-soil, canker-lesion(X, V), V \= dk-brown-blk, int-discolor(X, V), V \= brown, sclerotia(X, present).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, stem-cankers ≠ below-soil, canker-lesion ≠ dk-brown-blk, int-discolor ≠ brown, sclerotia = present

logic:      leafspot-size ≠ gt-1/8 ∧ stem-cankers ≠ below-soil ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor ≠ brown ∧ sclerotia ≠ present → bacterial-blight
prolog:     'bacterial-blight'(X) :- leafspot-size(X, V), V \= gt-1/8, stem-cankers(X, V), V \= below-soil, canker-lesion(X, V), V \= dk-brown-blk, int-discolor(X, V), V \= brown, sclerotia(X, V), V \= present.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size ≠ gt-1/8, stem-cankers ≠ below-soil, canker-lesion ≠ dk-brown-blk, int-discolor ≠ brown, sclerotia ≠ present

logic:      leafspot-size = gt-1/8 ∧ fruiting-bodies = present ∧ fruit-pods ≠ diseased → brown-spot
prolog:     'brown-spot'(X) :- leafspot-size(X, gt-1/8), fruiting-bodies(X, present), fruit-pods(X, V), V \= diseased.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size = gt-1/8, fruiting-bodies = present, fruit-pods ≠ diseased

logic:      leafspot-size = gt-1/8 ∧ leaf-mild = lower-surf ∧ fruiting-bodies ≠ present ∧ fruit-pods ≠ diseased → downy-mildew
prolog:     'downy-mildew'(X) :- leafspot-size(X, gt-1/8), leaf-mild(X, lower-surf), fruiting-bodies(X, V), V \= present, fruit-pods(X, V), V \= diseased.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size = gt-1/8, leaf-mild = lower-surf, fruiting-bodies ≠ present, fruit-pods ≠ diseased

logic:      date = june ∧ leafspot-size = gt-1/8 ∧ leaf-mild ≠ lower-surf ∧ fruiting-bodies ≠ present ∧ fruit-pods ≠ diseased → brown-spot
prolog:     'brown-spot'(X) :- date(X, june), leafspot-size(X, gt-1/8), leaf-mild(X, V), V \= lower-surf, fruiting-bodies(X, V), V \= present, fruit-pods(X, V), V \= diseased.
pattern:    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: date = june, leafspot-size = gt-1/8, leaf-mild ≠ lower-surf, fruiting-bodies ≠ present, fruit-pods ≠ diseased

logic:      date ≠ june ∧ leafspot-size = gt-1/8 ∧ leaf-mild ≠ lower-surf ∧ fruiting-bodies ≠ present ∧ fruit-pods ≠ diseased → alternarialeaf-spot
prolog:     'alternarialeaf-spot'(X) :- date(X, V), V \= june, leafspot-size(X, gt-1/8), leaf-mild(X, V), V \= lower-surf, fruiting-bodies(X, V), V \= present, fruit-pods(X, V), V \= diseased.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: date ≠ june, leafspot-size = gt-1/8, leaf-mild ≠ lower-surf, fruiting-bodies ≠ present, fruit-pods ≠ diseased

logic:      germination ≠ 90-100 ∧ leafspot-size = gt-1/8 ∧ fruit-pods = diseased ∧ fruit-spots ≠ colored → brown-spot
prolog:     'brown-spot'(X) :- germination(X, V), V \= 90-100, leafspot-size(X, gt-1/8), fruit-pods(X, diseased), fruit-spots(X, V), V \= colored.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: germination ≠ 90-100, leafspot-size = gt-1/8, fruit-pods = diseased, fruit-spots ≠ colored

logic:      germination = 90-100 ∧ leafspot-size = gt-1/8 ∧ fruit-pods = diseased ∧ fruit-spots ≠ colored → frog-eye-leaf-spot
prolog:     'frog-eye-leaf-spot'(X) :- germination(X, 90-100), leafspot-size(X, gt-1/8), fruit-pods(X, diseased), fruit-spots(X, V), V \= colored.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: germination = 90-100, leafspot-size = gt-1/8, fruit-pods = diseased, fruit-spots ≠ colored

logic:      leafspot-size = gt-1/8 ∧ fruit-pods = diseased ∧ fruit-spots = colored → frog-eye-leaf-spot
prolog:     'frog-eye-leaf-spot'(X) :- leafspot-size(X, gt-1/8), fruit-pods(X, diseased), fruit-spots(X, colored).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: leafspot-size = gt-1/8, fruit-pods = diseased, fruit-spots = colored

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.664 | 0.664 | 1.000 | 14 | 11 |
| 2 | 0.606 | 0.606 | 1.000 | 14 | 11 |
| 3 | 0.620 | 0.620 | 1.000 | 14 | 11 |
| 4 | 0.676 | 0.676 | 1.000 | 14 | 11 |
| 5 | 0.684 | 0.684 | 1.000 | 14 | 10 |

**Summary** (0.5s): tree accuracy 0.650 +/- 0.031, rule-set accuracy 0.650 +/- 0.031, tree/rule-set agreement 1.0000, mean rules/fold 14.0, mean attributes used/fold 10.8 of 35 available.

---

## mushroom

n=8124, attributes=21 (21 categorical, 0 numeric), classes=2

### Fold 1 detail

```text
Decision tree (12 leaves, depth 5):
|--- odor!=n <= 0.50
|   |--- spore-print-color!=r <= 0.50
|   |   |--- class: p
|   |--- spore-print-color!=r >  0.50
|   |   |--- stalk-surface-below-ring!=y <= 0.50
|   |   |   |--- ring-number!=t <= 0.50
|   |   |   |   |--- class: e
|   |   |   |--- ring-number!=t >  0.50
|   |   |   |   |--- class: p
|   |   |--- stalk-surface-below-ring!=y >  0.50
|   |   |   |--- cap-surface=g <= 0.50
|   |   |   |   |--- cap-shape!=c <= 0.50
|   |   |   |   |   |--- class: p
|   |   |   |   |--- cap-shape!=c >  0.50
|   |   |   |   |   |--- class: e
|   |   |   |--- cap-surface=g >  0.50
|   |   |   |   |--- class: p
|--- odor!=n >  0.50
|   |--- stalk-root=c <= 0.50
|   |   |--- stalk-surface-below-ring!=y <= 0.50
|   |   |   |--- class: e
|   |   |--- stalk-surface-below-ring!=y >  0.50
|   |   |   |--- spore-print-color!=u <= 0.50
|   |   |   |   |--- class: e
|   |   |   |--- spore-print-color!=u >  0.50
|   |   |   |   |--- odor!=a <= 0.50
|   |   |   |   |   |--- class: e
|   |   |   |   |--- odor!=a >  0.50
|   |   |   |   |   |--- class: p
|   |--- stalk-root=c >  0.50
|   |   |--- ring-number=n <= 0.50
|   |   |   |--- class: e
|   |   |--- ring-number=n >  0.50
|   |   |   |--- class: p

Extracted DisjointRuleSet (12 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 7/21

logic:      odor = n ∧ spore-print-color = r → p
prolog:     p(X) :- odor(X, n), spore-print-color(X, r).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor = n, spore-print-color = r

logic:      odor = n ∧ stalk-surface-below-ring = y ∧ ring-number = t ∧ spore-print-color ≠ r → e
prolog:     e(X) :- odor(X, n), stalk-surface-below-ring(X, y), ring-number(X, t), spore-print-color(X, V), V \= r.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor = n, stalk-surface-below-ring = y, ring-number = t, spore-print-color ≠ r

logic:      odor = n ∧ stalk-surface-below-ring = y ∧ ring-number ≠ t ∧ spore-print-color ≠ r → p
prolog:     p(X) :- odor(X, n), stalk-surface-below-ring(X, y), ring-number(X, V), V \= t, spore-print-color(X, V), V \= r.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor = n, stalk-surface-below-ring = y, ring-number ≠ t, spore-print-color ≠ r

logic:      cap-shape = c ∧ cap-surface ≠ g ∧ odor = n ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color ≠ r → p
prolog:     p(X) :- cap-shape(X, c), cap-surface(X, V), V \= g, odor(X, n), stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, V), V \= r.
pattern:    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: cap-shape = c, cap-surface ≠ g, odor = n, stalk-surface-below-ring ≠ y, spore-print-color ≠ r

logic:      cap-shape ≠ c ∧ cap-surface ≠ g ∧ odor = n ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color ≠ r → e
prolog:     e(X) :- cap-shape(X, V), V \= c, cap-surface(X, V), V \= g, odor(X, n), stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, V), V \= r.
pattern:    0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: cap-shape ≠ c, cap-surface ≠ g, odor = n, stalk-surface-below-ring ≠ y, spore-print-color ≠ r

logic:      cap-surface = g ∧ odor = n ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color ≠ r → p
prolog:     p(X) :- cap-surface(X, g), odor(X, n), stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, V), V \= r.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: cap-surface = g, odor = n, stalk-surface-below-ring ≠ y, spore-print-color ≠ r

logic:      odor ≠ n ∧ stalk-root ≠ c ∧ stalk-surface-below-ring = y → e
prolog:     e(X) :- odor(X, V), V \= n, stalk-root(X, V), V \= c, stalk-surface-below-ring(X, y).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor ≠ n, stalk-root ≠ c, stalk-surface-below-ring = y

logic:      odor ≠ n ∧ stalk-root ≠ c ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color = u → e
prolog:     e(X) :- odor(X, V), V \= n, stalk-root(X, V), V \= c, stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, u).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor ≠ n, stalk-root ≠ c, stalk-surface-below-ring ≠ y, spore-print-color = u

logic:      odor = a ∧ odor ≠ n ∧ stalk-root ≠ c ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color ≠ u → e
prolog:     e(X) :- odor(X, a), odor(X, V), V \= n, stalk-root(X, V), V \= c, stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, V), V \= u.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor = a, odor ≠ n, stalk-root ≠ c, stalk-surface-below-ring ≠ y, spore-print-color ≠ u

logic:      odor ≠ a ∧ odor ≠ n ∧ stalk-root ≠ c ∧ stalk-surface-below-ring ≠ y ∧ spore-print-color ≠ u → p
prolog:     p(X) :- odor(X, V), V \= a, odor(X, V), V \= n, stalk-root(X, V), V \= c, stalk-surface-below-ring(X, V), V \= y, spore-print-color(X, V), V \= u.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor ≠ a, odor ≠ n, stalk-root ≠ c, stalk-surface-below-ring ≠ y, spore-print-color ≠ u

logic:      odor ≠ n ∧ stalk-root = c ∧ ring-number ≠ n → e
prolog:     e(X) :- odor(X, V), V \= n, stalk-root(X, c), ring-number(X, V), V \= n.
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor ≠ n, stalk-root = c, ring-number ≠ n

logic:      odor ≠ n ∧ stalk-root = c ∧ ring-number = n → p
prolog:     p(X) :- odor(X, V), V \= n, stalk-root(X, c), ring-number(X, n).
pattern:    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: odor ≠ n, stalk-root = c, ring-number = n

```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.995 | 0.995 | 1.000 | 12 | 7 |
| 2 | 0.996 | 0.996 | 1.000 | 12 | 7 |
| 3 | 1.000 | 1.000 | 1.000 | 12 | 7 |
| 4 | 0.999 | 0.999 | 1.000 | 12 | 7 |
| 5 | 0.999 | 0.999 | 1.000 | 12 | 7 |

**Summary** (0.9s): tree accuracy 0.998 +/- 0.002, rule-set accuracy 0.998 +/- 0.002, tree/rule-set agreement 1.0000, mean rules/fold 12.0, mean attributes used/fold 7.0 of 21 available.

---

## covtype

n=581012, attributes=54 (0 categorical, 54 numeric), classes=7

### Fold 1 detail

```text
validate_dataspec note(s) against the test fold: ["attribute 'Wilderness_Area_0' is NUMERIC but column 'Wilderness_Area_0' looks 'nominal'", "attribute 'Wilderness_Area_1' is NUMERIC but column 'Wilderness_Area_1' looks 'nominal'", "attribute 'Wilderness_Area_2' is NUMERIC but column 'Wilderness_Area_2' looks 'nominal'", "attribute 'Wilderness_Area_3' is NUMERIC but column 'Wilderness_Area_3' looks 'nominal'", "attribute 'Soil_Type_0' is NUMERIC but column 'Soil_Type_0' looks 'nominal'", "attribute 'Soil_Type_1' is NUMERIC but column 'Soil_Type_1' looks 'nominal'", "attribute 'Soil_Type_2' is NUMERIC but column 'Soil_Type_2' looks 'nominal'", "attribute 'Soil_Type_3' is NUMERIC but column 'Soil_Type_3' looks 'nominal'", "attribute 'Soil_Type_4' is NUMERIC but column 'Soil_Type_4' looks 'nominal'", "attribute 'Soil_Type_5' is NUMERIC but column 'Soil_Type_5' looks 'nominal'", "attribute 'Soil_Type_6' is NUMERIC but column 'Soil_Type_6' looks 'nominal'", "attribute 'Soil_Type_7' is NUMERIC but column 'Soil_Type_7' looks 'nominal'", "attribute 'Soil_Type_8' is NUMERIC but column 'Soil_Type_8' looks 'nominal'", "attribute 'Soil_Type_9' is NUMERIC but column 'Soil_Type_9' looks 'nominal'", "attribute 'Soil_Type_10' is NUMERIC but column 'Soil_Type_10' looks 'nominal'", "attribute 'Soil_Type_11' is NUMERIC but column 'Soil_Type_11' looks 'nominal'", "attribute 'Soil_Type_12' is NUMERIC but column 'Soil_Type_12' looks 'nominal'", "attribute 'Soil_Type_13' is NUMERIC but column 'Soil_Type_13' looks 'nominal'", "attribute 'Soil_Type_14' is NUMERIC but column 'Soil_Type_14' looks 'nominal'", "attribute 'Soil_Type_15' is NUMERIC but column 'Soil_Type_15' looks 'nominal'", "attribute 'Soil_Type_16' is NUMERIC but column 'Soil_Type_16' looks 'nominal'", "attribute 'Soil_Type_17' is NUMERIC but column 'Soil_Type_17' looks 'nominal'", "attribute 'Soil_Type_18' is NUMERIC but column 'Soil_Type_18' looks 'nominal'", "attribute 'Soil_Type_19' is NUMERIC but column 'Soil_Type_19' looks 'nominal'", "attribute 'Soil_Type_20' is NUMERIC but column 'Soil_Type_20' looks 'nominal'", "attribute 'Soil_Type_21' is NUMERIC but column 'Soil_Type_21' looks 'nominal'", "attribute 'Soil_Type_22' is NUMERIC but column 'Soil_Type_22' looks 'nominal'", "attribute 'Soil_Type_23' is NUMERIC but column 'Soil_Type_23' looks 'nominal'", "attribute 'Soil_Type_24' is NUMERIC but column 'Soil_Type_24' looks 'nominal'", "attribute 'Soil_Type_25' is NUMERIC but column 'Soil_Type_25' looks 'nominal'", "attribute 'Soil_Type_26' is NUMERIC but column 'Soil_Type_26' looks 'nominal'", "attribute 'Soil_Type_27' is NUMERIC but column 'Soil_Type_27' looks 'nominal'", "attribute 'Soil_Type_28' is NUMERIC but column 'Soil_Type_28' looks 'nominal'", "attribute 'Soil_Type_29' is NUMERIC but column 'Soil_Type_29' looks 'nominal'", "attribute 'Soil_Type_30' is NUMERIC but column 'Soil_Type_30' looks 'nominal'", "attribute 'Soil_Type_31' is NUMERIC but column 'Soil_Type_31' looks 'nominal'", "attribute 'Soil_Type_32' is NUMERIC but column 'Soil_Type_32' looks 'nominal'", "attribute 'Soil_Type_33' is NUMERIC but column 'Soil_Type_33' looks 'nominal'", "attribute 'Soil_Type_34' is NUMERIC but column 'Soil_Type_34' looks 'nominal'", "attribute 'Soil_Type_35' is NUMERIC but column 'Soil_Type_35' looks 'nominal'", "attribute 'Soil_Type_36' is NUMERIC but column 'Soil_Type_36' looks 'nominal'", "attribute 'Soil_Type_37' is NUMERIC but column 'Soil_Type_37' looks 'nominal'", "attribute 'Soil_Type_38' is NUMERIC but column 'Soil_Type_38' looks 'nominal'", "attribute 'Soil_Type_39' is NUMERIC but column 'Soil_Type_39' looks 'nominal'"]

Decision tree (233 leaves, depth 8):
|--- Elevation>=3047.5 <= 0.50
|   |--- Elevation<2510.5 <= 0.50
|   |   |--- Soil_Type_3<0.5 <= 0.50
|   |   |   |--- Elevation<2654.5 <= 0.50
|   |   |   |   |--- Horizontal_Distance_To_Roadways<1534.0 <= 0.50
|   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points<1556.5 <= 0.50
|   |   |   |   |   |   |--- Hillshade_9am<196.5 <= 0.50
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology<138.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology<138.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |--- Hillshade_9am<196.5 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology<166.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology<166.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points<1556.5 >  0.50
|   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=617.0 <= 0.50
|   |   |   |   |   |   |   |--- Hillshade_9am<227.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Hillshade_9am<227.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=617.0 >  0.50
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology>=69.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology>=69.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |--- Horizontal_Distance_To_Roadways<1534.0 >  0.50
|   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points<1058.0 <= 0.50
|   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=64.0 <= 0.50
|   |   |   |   |   |   |   |--- Hillshade_9am>=196.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Hillshade_9am>=196.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=64.0 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=1556.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=1556.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points<1058.0 >  0.50
|   |   |   |   |   |   |--- Aspect<146.5 <= 0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=440.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=440.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |--- Aspect<146.5 >  0.50
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology>=-26.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 6
|   |   |   |   |   |   |   |--- Vertical_Distance_To_Hydrology>=-26.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |--- Elevation<2654.5 >  0.50
|   |   |   |   |--- Wilderness_Area_2>=0.5 <= 0.50
|   |   |   |   |   |--- class: 2
|   |   |   |   |--- Wilderness_Area_2>=0.5 >  0.50
|   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=15.0 <= 0.50
|   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 <= 0.50
|   |   |   |   |   |   |   |--- Aspect<69.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |   |--- Aspect<69.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=617.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=617.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 6
|   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=15.0 >  0.50
|   |   |   |   |   |   |--- Aspect<69.5 <= 0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=64.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=64.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |--- Aspect<69.5 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology<166.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 3
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology<166.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 6
|   |   |--- Soil_Type_3<0.5 >  0.50
|   |   |   |--- Elevation<2942.5 <= 0.50
|   |   |   |   |--- Horizontal_Distance_To_Hydrology>=166.0 <= 0.50
|   |   |   |   |   |--- Soil_Type_22<0.5 <= 0.50
|   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 <= 0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways<570.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways<570.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=2896.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Fire_Points>=2896.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |--- Soil_Type_22<0.5 >  0.50
|   |   |   |   |   |   |--- Soil_Type_21>=0.5 <= 0.50
|   |   |   |   |   |   |   |--- Hillshade_Noon>=216.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- Hillshade_Noon>=216.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |--- Soil_Type_21>=0.5 >  0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways>=1534.0 >  0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |--- Horizontal_Distance_To_Hydrology>=166.0 >  0.50
|   |   |   |   |   |--- Hillshade_Noon>=216.5 <= 0.50
|   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=440.0 <= 0.50
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways<5347.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Horizontal_Distance_To_Roadways<5347.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |--- Horizontal_Distance_To_Hydrology>=440.0 >  0.50
|   |   |   |   |   |   |   |--- Hillshade_Noon<189.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Hillshade_Noon<189.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |--- Hillshade_Noon>=216.5 >  0.50
|   |   |   |   |   |   |--- Soil_Type_22>=0.5 <= 0.50
|   |   |   |   |   |   |   |--- Soil_Type_21>=0.5 <= 0.50
|   |   |   |   |   |   |   |   |--- class: 2
|   |   |   |   |   |   |   |--- Soil_Type_21>=0.5 >  0.50
|   |   |   |   |   |   |   |   |--- class: 1
|   |   |   |   |   |   |--- Soil_Type_22>=0.5 >  0.50
... (577 more lines omitted for brevity)

Extracted DisjointRuleSet (233 rules) -- is_disjoint=True, is_exhaustive=True, attributes used: 35/54

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Vertical_Distance_To_Hydrology >= 138.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am >= 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Vertical_Distance_To_Hydrology(X, V), V >= 138.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V >= 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Vertical_Distance_To_Hydrology >= 138.5, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am >= 196.5, Horizontal_Distance_To_Fire_Points >= 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Vertical_Distance_To_Hydrology < 138.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am >= 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1556.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Vertical_Distance_To_Hydrology(X, V), V < 138.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V >= 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Vertical_Distance_To_Hydrology < 138.5, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am >= 196.5, Horizontal_Distance_To_Fire_Points >= 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology >= 166.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am < 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V >= 166.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V < 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology >= 166.0, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am < 196.5, Horizontal_Distance_To_Fire_Points >= 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am < 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V < 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am < 196.5, Horizontal_Distance_To_Fire_Points >= 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am >= 227.5 ∧ Horizontal_Distance_To_Fire_Points < 617.0 ∧ Horizontal_Distance_To_Fire_Points < 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V >= 227.5, Horizontal_Distance_To_Fire_Points(X, V), V < 617.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am >= 227.5, Horizontal_Distance_To_Fire_Points < 617.0, Horizontal_Distance_To_Fire_Points < 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Hillshade_9am < 227.5 ∧ Horizontal_Distance_To_Fire_Points < 617.0 ∧ Horizontal_Distance_To_Fire_Points < 1556.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Hillshade_9am(X, V), V < 227.5, Horizontal_Distance_To_Fire_Points(X, V), V < 617.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Roadways >= 1534.0, Hillshade_9am < 227.5, Horizontal_Distance_To_Fire_Points < 617.0, Horizontal_Distance_To_Fire_Points < 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Vertical_Distance_To_Hydrology < 69.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 617.0 ∧ Horizontal_Distance_To_Fire_Points < 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Vertical_Distance_To_Hydrology(X, V), V < 69.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 617.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Vertical_Distance_To_Hydrology < 69.5, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points >= 617.0, Horizontal_Distance_To_Fire_Points < 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Vertical_Distance_To_Hydrology >= 69.5 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 617.0 ∧ Horizontal_Distance_To_Fire_Points < 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Vertical_Distance_To_Hydrology(X, V), V >= 69.5, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 617.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Vertical_Distance_To_Hydrology >= 69.5, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points >= 617.0, Horizontal_Distance_To_Fire_Points < 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 64.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Hillshade_9am < 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1058.0 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 64.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Hillshade_9am(X, V), V < 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 64.0, Horizontal_Distance_To_Roadways < 1534.0, Hillshade_9am < 196.5, Horizontal_Distance_To_Fire_Points >= 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 64.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Hillshade_9am >= 196.5 ∧ Horizontal_Distance_To_Fire_Points >= 1058.0 ∧ Soil_Type_3 >= 0.5 → 1
prolog:     '1'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 64.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Hillshade_9am(X, V), V >= 196.5, Horizontal_Distance_To_Fire_Points(X, V), V >= 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 64.0, Horizontal_Distance_To_Roadways < 1534.0, Hillshade_9am >= 196.5, Horizontal_Distance_To_Fire_Points >= 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology >= 64.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 1058.0 ∧ Horizontal_Distance_To_Fire_Points < 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V >= 64.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 1058.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology >= 64.0, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points >= 1058.0, Horizontal_Distance_To_Fire_Points < 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology >= 64.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 1058.0 ∧ Horizontal_Distance_To_Fire_Points >= 1556.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V >= 64.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 1058.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 1556.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology >= 64.0, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points >= 1058.0, Horizontal_Distance_To_Fire_Points >= 1556.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Aspect >= 146.5 ∧ Horizontal_Distance_To_Hydrology < 440.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 1058.0 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V >= 146.5, Horizontal_Distance_To_Hydrology(X, V), V < 440.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Aspect >= 146.5, Horizontal_Distance_To_Hydrology < 440.0, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points < 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Aspect >= 146.5 ∧ Horizontal_Distance_To_Hydrology >= 440.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 1058.0 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V >= 146.5, Horizontal_Distance_To_Hydrology(X, V), V >= 440.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Aspect >= 146.5, Horizontal_Distance_To_Hydrology >= 440.0, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points < 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Aspect < 146.5 ∧ Vertical_Distance_To_Hydrology < -26.5 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 1058.0 ∧ Soil_Type_3 >= 0.5 → 6
prolog:     '6'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V < 146.5, Vertical_Distance_To_Hydrology(X, V), V < -26.5, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Aspect < 146.5, Vertical_Distance_To_Hydrology < -26.5, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points < 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2654.5 ∧ Elevation < 3047.5 ∧ Aspect < 146.5 ∧ Vertical_Distance_To_Hydrology >= -26.5 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 1058.0 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V < 146.5, Vertical_Distance_To_Hydrology(X, V), V >= -26.5, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 1058.0, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2654.5, Elevation < 3047.5, Aspect < 146.5, Vertical_Distance_To_Hydrology >= -26.5, Horizontal_Distance_To_Roadways < 1534.0, Horizontal_Distance_To_Fire_Points < 1058.0, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Wilderness_Area_2 < 0.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Wilderness_Area_2(X, V), V < 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Wilderness_Area_2 < 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect >= 69.5 ∧ Horizontal_Distance_To_Hydrology < 15.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V >= 69.5, Horizontal_Distance_To_Hydrology(X, V), V < 15.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect >= 69.5, Horizontal_Distance_To_Hydrology < 15.0, Horizontal_Distance_To_Roadways < 1534.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect < 69.5 ∧ Horizontal_Distance_To_Hydrology < 15.0 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V < 69.5, Horizontal_Distance_To_Hydrology(X, V), V < 15.0, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect < 69.5, Horizontal_Distance_To_Hydrology < 15.0, Horizontal_Distance_To_Roadways < 1534.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 15.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 617.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 15.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 617.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 15.0, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points < 617.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 15.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 617.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 6
prolog:     '6'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 15.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 617.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 15.0, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points >= 617.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect >= 69.5 ∧ Horizontal_Distance_To_Hydrology >= 15.0 ∧ Horizontal_Distance_To_Hydrology < 64.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V >= 69.5, Horizontal_Distance_To_Hydrology(X, V), V >= 15.0, Horizontal_Distance_To_Hydrology(X, V), V < 64.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect >= 69.5, Horizontal_Distance_To_Hydrology >= 15.0, Horizontal_Distance_To_Hydrology < 64.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect >= 69.5 ∧ Horizontal_Distance_To_Hydrology >= 15.0 ∧ Horizontal_Distance_To_Hydrology >= 64.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V >= 69.5, Horizontal_Distance_To_Hydrology(X, V), V >= 15.0, Horizontal_Distance_To_Hydrology(X, V), V >= 64.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect >= 69.5, Horizontal_Distance_To_Hydrology >= 15.0, Horizontal_Distance_To_Hydrology >= 64.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect < 69.5 ∧ Horizontal_Distance_To_Hydrology >= 15.0 ∧ Horizontal_Distance_To_Hydrology >= 166.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 3
prolog:     '3'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V < 69.5, Horizontal_Distance_To_Hydrology(X, V), V >= 15.0, Horizontal_Distance_To_Hydrology(X, V), V >= 166.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect < 69.5, Horizontal_Distance_To_Hydrology >= 15.0, Horizontal_Distance_To_Hydrology >= 166.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation < 2654.5 ∧ Elevation < 3047.5 ∧ Aspect < 69.5 ∧ Horizontal_Distance_To_Hydrology >= 15.0 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Wilderness_Area_2 >= 0.5 ∧ Soil_Type_3 >= 0.5 → 6
prolog:     '6'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V < 2654.5, Elevation(X, V), V < 3047.5, Aspect(X, V), V < 69.5, Horizontal_Distance_To_Hydrology(X, V), V >= 15.0, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Wilderness_Area_2(X, V), V >= 0.5, Soil_Type_3(X, V), V >= 0.5.
pattern:    1 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation < 2654.5, Elevation < 3047.5, Aspect < 69.5, Horizontal_Distance_To_Hydrology >= 15.0, Horizontal_Distance_To_Hydrology < 166.0, Wilderness_Area_2 >= 0.5, Soil_Type_3 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2942.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Horizontal_Distance_To_Roadways >= 570.5 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Soil_Type_3 < 0.5 ∧ Soil_Type_22 >= 0.5 → 2
prolog:     '2'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2942.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Horizontal_Distance_To_Roadways(X, V), V >= 570.5, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Soil_Type_3(X, V), V < 0.5, Soil_Type_22(X, V), V >= 0.5.
pattern:    1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2942.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Horizontal_Distance_To_Roadways >= 570.5, Horizontal_Distance_To_Roadways < 1534.0, Soil_Type_3 < 0.5, Soil_Type_22 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2942.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Horizontal_Distance_To_Roadways < 570.5 ∧ Horizontal_Distance_To_Roadways < 1534.0 ∧ Soil_Type_3 < 0.5 ∧ Soil_Type_22 >= 0.5 → 1
prolog:     '1'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2942.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Horizontal_Distance_To_Roadways(X, V), V < 570.5, Horizontal_Distance_To_Roadways(X, V), V < 1534.0, Soil_Type_3(X, V), V < 0.5, Soil_Type_22(X, V), V >= 0.5.
pattern:    1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2942.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Horizontal_Distance_To_Roadways < 570.5, Horizontal_Distance_To_Roadways < 1534.0, Soil_Type_3 < 0.5, Soil_Type_22 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2942.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points < 2896.5 ∧ Soil_Type_3 < 0.5 ∧ Soil_Type_22 >= 0.5 → 1
prolog:     '1'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2942.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V < 2896.5, Soil_Type_3(X, V), V < 0.5, Soil_Type_22(X, V), V >= 0.5.
pattern:    1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2942.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points < 2896.5, Soil_Type_3 < 0.5, Soil_Type_22 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2942.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Horizontal_Distance_To_Roadways >= 1534.0 ∧ Horizontal_Distance_To_Fire_Points >= 2896.5 ∧ Soil_Type_3 < 0.5 ∧ Soil_Type_22 >= 0.5 → 1
prolog:     '1'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2942.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Horizontal_Distance_To_Roadways(X, V), V >= 1534.0, Horizontal_Distance_To_Fire_Points(X, V), V >= 2896.5, Soil_Type_3(X, V), V < 0.5, Soil_Type_22(X, V), V >= 0.5.
pattern:    1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2942.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Horizontal_Distance_To_Roadways >= 1534.0, Horizontal_Distance_To_Fire_Points >= 2896.5, Soil_Type_3 < 0.5, Soil_Type_22 >= 0.5

logic:      Elevation >= 2510.5 ∧ Elevation >= 2942.5 ∧ Elevation < 3047.5 ∧ Horizontal_Distance_To_Hydrology < 166.0 ∧ Hillshade_Noon < 216.5 ∧ Soil_Type_3 < 0.5 ∧ Soil_Type_21 < 0.5 ∧ Soil_Type_22 < 0.5 → 1
prolog:     '1'(X) :- Elevation(X, V), V >= 2510.5, Elevation(X, V), V >= 2942.5, Elevation(X, V), V < 3047.5, Horizontal_Distance_To_Hydrology(X, V), V < 166.0, Hillshade_Noon(X, V), V < 216.5, Soil_Type_3(X, V), V < 0.5, Soil_Type_21(X, V), V < 0.5, Soil_Type_22(X, V), V < 0.5.
pattern:    1 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
conditions: Elevation >= 2510.5, Elevation >= 2942.5, Elevation < 3047.5, Horizontal_Distance_To_Hydrology < 166.0, Hillshade_Noon < 216.5, Soil_Type_3 < 0.5, Soil_Type_21 < 0.5, Soil_Type_22 < 0.5

... (203 more rules omitted for brevity)
```

### Per-fold results

| fold | tree acc | rule-set acc | agreement | n rules | attrs used |
|---|---|---|---|---|---|
| 1 | 0.738 | 0.738 | 1.000 | 233 | 35 |
| 2 | 0.733 | 0.733 | 1.000 | 234 | 37 |
| 3 | 0.738 | 0.738 | 1.000 | 239 | 34 |
| 4 | 0.736 | 0.736 | 1.000 | 238 | 34 |
| 5 | 0.738 | 0.738 | 1.000 | 237 | 34 |

**Summary** (324.1s): tree accuracy 0.737 +/- 0.002, rule-set accuracy 0.737 +/- 0.002, tree/rule-set agreement 1.0000, mean rules/fold 236.2, mean attributes used/fold 34.8 of 54 available.

---

## Overall summary

`categ.`/`numeric` = number of original attributes of each type (after dropping signal-free columns, see module docstring); `attrs used` = mean number of those original attributes actually referenced by the extracted rules, out of `categ.` + `numeric` available -- not the same as the number of *derived* Boolean features the rules are built from. Since the DisjointRuleSet is a one-to-one re-expression of the tree (one literal per split, no simplification), this is also exactly the number of attributes the tree itself splits on -- `score_fold` asserts the two counts match on every fold, so `attrs used` doubles as that count.

| dataset | n | categ. | numeric | tree acc | rule acc | agree | rules/fold | attrs used/fold | time(s) |
|---|---|---|---|---|---|---|---|---|---|
| vote | 435 | 16 | 0 | 0.938±0.026 | 0.938±0.026 | 1.000 | 18.4 | 10.6 | 0.2 |
| breast-cancer | 286 | 9 | 0 | 0.724±0.014 | 0.724±0.014 | 1.000 | 19.0 | 7.6 | 0.2 |
| kr-vs-kp | 3196 | 36 | 0 | 0.941±0.011 | 0.941±0.011 | 1.000 | 11.4 | 9.4 | 0.5 |
| hypothyroid | 3772 | 21 | 6 | 0.992±0.003 | 0.992±0.003 | 1.000 | 13.8 | 7.8 | 1.0 |
| soybean | 683 | 35 | 0 | 0.650±0.031 | 0.650±0.031 | 1.000 | 14.0 | 10.8 | 0.5 |
| mushroom | 8124 | 21 | 0 | 0.998±0.002 | 0.998±0.002 | 1.000 | 12.0 | 7.0 | 0.9 |
| covtype | 581012 | 0 | 54 | 0.737±0.002 | 0.737±0.002 | 1.000 | 236.2 | 34.8 | 324.1 |
