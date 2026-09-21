# Random forest -> RuleSet import + combiner comparison demo

Generated 2026-08-30 21:14:30. N_FOLDS=5, N_ESTIMATORS=10, MAX_DEPTH=4, MAX_INTERVALS=6.

Only fold 1 of each dataset shows the top/bottom-5-by-weight rule detail; all folds contribute to the summary statistics. See the module docstring for what each method means, and why `macro-vote (~sklearn)` in particular is expected to land closest to the forest's own accuracy.

---

## vote

n=435, attributes=16, classes=2

### Fold 1 detail

```text
Combined RuleSet: 106 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.994] counts={'democrat': 187.0, 'republican': 0.0} adoption-of-the-budget-resolution = y ∧ physician-fee-freeze = n → democrat
  [w=0.994] counts={'democrat': 169.0, 'republican': 0.0} physician-fee-freeze = n ∧ aid-to-nicaraguan-contras ≠ n → democrat
  [w=0.994] counts={'democrat': 165.0, 'republican': 0.0} physician-fee-freeze = n ∧ aid-to-nicaraguan-contras = y → democrat
  [w=0.994] counts={'democrat': 171.0, 'republican': 0.0} adoption-of-the-budget-resolution ≠ ? ∧ physician-fee-freeze ≠ y ∧ el-salvador-aid ≠ y → democrat
  [w=0.994] counts={'democrat': 148.0, 'republican': 0.0} physician-fee-freeze = n ∧ el-salvador-aid = n → democrat

Bottom 5:
  [w=0.462] counts={'democrat': 6.0, 'republican': 3.0} physician-fee-freeze = y ∧ physician-fee-freeze ≠ n ∧ synfuels-corporation-cutback = y ∧ export-administration-act-south-africa ≠ y → democrat
  [w=0.400] counts={'democrat': 0.0, 'republican': 1.0} handicapped-infants = n ∧ physician-fee-freeze = n ∧ aid-to-nicaraguan-contras = n ∧ superfund-right-to-sue ≠ y → republican
  [w=0.400] counts={'democrat': 0.0, 'republican': 2.0} adoption-of-the-budget-resolution = n ∧ physician-fee-freeze = n ∧ aid-to-nicaraguan-contras ≠ y ∧ synfuels-corporation-cutback = n → republican
  [w=0.333] counts={'democrat': 1.0, 'republican': 1.0} physician-fee-freeze = y ∧ anti-satellite-test-ban = n ∧ education-spending ≠ n ∧ duty-free-exports = y → democrat
  [w=0.333] counts={'democrat': 2.0, 'republican': 3.0} water-project-cost-sharing ≠ y ∧ adoption-of-the-budget-resolution ≠ n ∧ anti-satellite-test-ban = n ∧ mx-missile ≠ n → republican
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.963 +/- 0.018 | 1.000 |
| vote (unweighted) | 0.959 +/- 0.014 | 0.995 |
| micro-vote (pooled counts) | 0.952 +/- 0.017 | 0.989 |
| macro-vote (~sklearn) | 0.963 +/- 0.018 | 1.000 |
| micro-max | 0.952 +/- 0.022 | 0.984 |
| macro-max | 0.940 +/- 0.017 | 0.959 |
| max (precision) | 0.952 +/- 0.009 | 0.979 |
| vote (precision-weighted) | 0.961 +/- 0.016 | 0.998 |
| max (laplace) | 0.954 +/- 0.018 | 0.986 |
| vote (laplace-weighted) | 0.961 +/- 0.016 | 0.998 |

mean rules/fold: 111.8 (0.4s total)

---

## breast-cancer

n=286, attributes=9, classes=2

### Fold 1 detail

```text
Combined RuleSet: 136 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.962] counts={'no-recurrence-events': 29.0, 'recurrence-events': 0.0} tumor-size = 10-14 ∧ tumor-size ≠ 30-34 → no-recurrence-events
  [w=0.960] counts={'no-recurrence-events': 24.0, 'recurrence-events': 0.0} tumor-size = 10-14 ∧ inv-nodes = 0-2 → no-recurrence-events
  [w=0.941] counts={'no-recurrence-events': 12.0, 'recurrence-events': 0.0} deg-malig ≠ 3 ∧ breast-quad = right_low → no-recurrence-events
  [w=0.929] counts={'no-recurrence-events': 15.0, 'recurrence-events': 0.0} tumor-size ≠ 20-24 ∧ tumor-size ≠ 30-34 ∧ breast-quad = left_up ∧ irradiat = yes → no-recurrence-events
  [w=0.896] counts={'no-recurrence-events': 39.0, 'recurrence-events': 1.0} tumor-size ≠ 40-44 ∧ node-caps ≠ ? ∧ deg-malig = 1 ∧ irradiat = no → no-recurrence-events

Bottom 5:
  [w=0.400] counts={'no-recurrence-events': 1.0, 'recurrence-events': 1.0} inv-nodes = 9-11 ∧ breast = left ∧ breast-quad ≠ left_up ∧ irradiat ≠ no → no-recurrence-events
  [w=0.400] counts={'no-recurrence-events': 0.0, 'recurrence-events': 1.0} age = 40-49 ∧ tumor-size = 30-34 ∧ deg-malig = 2 ∧ breast-quad = left_up → recurrence-events
  [w=0.400] counts={'no-recurrence-events': 2.0, 'recurrence-events': 2.0} tumor-size = 40-44 ∧ inv-nodes = 0-2 ∧ deg-malig ≠ 3 ∧ breast = left → no-recurrence-events
  [w=0.385] counts={'no-recurrence-events': 4.0, 'recurrence-events': 8.0} menopause ≠ lt40 ∧ tumor-size = 20-24 ∧ inv-nodes = 0-2 ∧ breast ≠ left → recurrence-events
  [w=0.333] counts={'no-recurrence-events': 2.0, 'recurrence-events': 5.0} age = 40-49 ∧ tumor-size = 30-34 ∧ deg-malig = 1 ∧ deg-malig ≠ 2 → recurrence-events
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.734 +/- 0.028 | 1.000 |
| vote (unweighted) | 0.738 +/- 0.030 | 0.975 |
| micro-vote (pooled counts) | 0.755 +/- 0.016 | 0.951 |
| macro-vote (~sklearn) | 0.734 +/- 0.028 | 1.000 |
| micro-max | 0.741 +/- 0.017 | 0.930 |
| macro-max | 0.689 +/- 0.055 | 0.857 |
| max (precision) | 0.706 +/- 0.041 | 0.874 |
| vote (precision-weighted) | 0.745 +/- 0.028 | 0.975 |
| max (laplace) | 0.755 +/- 0.019 | 0.930 |
| vote (laplace-weighted) | 0.741 +/- 0.028 | 0.979 |

mean rules/fold: 126.0 (0.4s total)

---

## colic

n=368, attributes=26, classes=2

### Fold 1 detail

```text
Combined RuleSet: 124 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.968] counts={'1': 0.0, '2': 39.0} surgery ≠ 1 ∧ pain = 1 ∧ outcome = 1 → 2
  [w=0.963] counts={'1': 37.0, '2': 0.0} pulse >= 58.0 ∧ abdominal_distension = 3 ∧ total_protein < 7.1499998569488525 → 1
  [w=0.960] counts={'1': 139.0, '2': 5.0} surgery ≠ 2 ∧ Age = 1 ∧ outcome ≠ 3 ∧ site_of_lesion ≠ 0 → 1
  [w=0.957] counts={'1': 1.0, '2': 41.0} surgery = 2 ∧ capillary_refill_time ≠ 2 ∧ peristalsis ≠ ? ∧ site_of_lesion = 0 → 2
  [w=0.952] counts={'1': 0.0, '2': 21.0} pulse < 58.0 ∧ peristalsis = 1 ∧ site_of_lesion = 0 → 2

Bottom 5:
  [w=0.429] counts={'1': 1.0, '2': 4.0} surgery = 2 ∧ peripheral_pulse ≠ 4 ∧ mucous_membranes = 4 ∧ peristalsis = 4 → 2
  [w=0.429] counts={'1': 4.0, '2': 6.0} surgery ≠ 1 ∧ rectal_examination_-_feces = 4 ∧ abdomen ≠ 4 ∧ packed_cell_volume >= 38.5 → 2
  [w=0.429] counts={'1': 4.0, '2': 6.0} pulse < 58.0 ∧ temperature_of_extremities = 3 ∧ pain ≠ 1 ∧ pain ≠ 3 → 2
  [w=0.400] counts={'1': 14.0, '2': 12.0} surgery ≠ 1 ∧ temperature_of_extremities = 3 ∧ capillary_refill_time = 1 ∧ abdomen ≠ 4 → 1
  [w=0.300] counts={'1': 1.0, '2': 1.0} pulse < 58.0 ∧ peristalsis = 1 ∧ outcome = 1 ∧ site_of_lesion ≠ 0 → 1
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.862 +/- 0.035 | 1.000 |
| vote (unweighted) | 0.864 +/- 0.045 | 0.959 |
| micro-vote (pooled counts) | 0.845 +/- 0.029 | 0.940 |
| macro-vote (~sklearn) | 0.862 +/- 0.035 | 1.000 |
| micro-max | 0.761 +/- 0.077 | 0.824 |
| macro-max | 0.802 +/- 0.048 | 0.891 |
| max (precision) | 0.834 +/- 0.023 | 0.924 |
| vote (precision-weighted) | 0.867 +/- 0.041 | 0.984 |
| max (laplace) | 0.848 +/- 0.040 | 0.954 |
| vote (laplace-weighted) | 0.862 +/- 0.041 | 0.978 |

mean rules/fold: 122.4 (0.7s total)

---

## credit-approval

n=690, attributes=15, classes=2

### Fold 1 detail

```text
Combined RuleSet: 139 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.972] counts={'+': 30.0, '-': 0.0} A8 >= 4.082499980926514 ∧ A9 = t ∧ A15 >= 538.5 → +
  [w=0.971] counts={'+': 64.0, '-': 0.0} A8 >= 0.6450000107288361 ∧ A9 ≠ f ∧ A11 >= 2.5 ∧ A14 < 129.5 → +
  [w=0.970] counts={'+': 101.0, '-': 0.0} A6 ≠ w ∧ A9 = t ∧ A11 >= 0.5 ∧ A15 >= 246.5 → +
  [w=0.963] counts={'+': 11.0, '-': 242.0} A3 >= 0.22999999672174454 ∧ A4 ≠ ? ∧ A5 ≠ gg ∧ A9 = f → -
  [w=0.963] counts={'+': 3.0, '-': 243.0} A3 >= 0.22999999672174454 ∧ A4 ≠ ? ∧ A4 ≠ l ∧ A9 ≠ t → -

Bottom 5:
  [w=0.400] counts={'+': 3.0, '-': 3.0} A6 = ff ∧ A9 = t ∧ A11 >= 0.5 ∧ A11 < 4.5 → +
  [w=0.400] counts={'+': 3.0, '-': 0.0} A1 ≠ b ∧ A6 = ? ∧ A9 = f ∧ A11 < 4.5 → +
  [w=0.375] counts={'+': 2.0, '-': 3.0} A4 ≠ y ∧ A6 = ff ∧ A9 = t ∧ A14 < 129.5 → -
  [w=0.333] counts={'+': 0.0, '-': 1.0} A6 = e ∧ A9 = t ∧ A11 >= 4.5 ∧ A12 ≠ t → -
  [w=0.333] counts={'+': 2.0, '-': 1.0} A5 ≠ g ∧ A6 = cc ∧ A8 >= 0.6450000107288361 ∧ A9 = f → +
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.861 +/- 0.021 | 1.000 |
| vote (unweighted) | 0.851 +/- 0.045 | 0.964 |
| micro-vote (pooled counts) | 0.839 +/- 0.027 | 0.943 |
| macro-vote (~sklearn) | 0.861 +/- 0.021 | 1.000 |
| micro-max | 0.806 +/- 0.023 | 0.890 |
| macro-max | 0.843 +/- 0.041 | 0.928 |
| max (precision) | 0.858 +/- 0.031 | 0.945 |
| vote (precision-weighted) | 0.852 +/- 0.033 | 0.974 |
| max (laplace) | 0.864 +/- 0.024 | 0.957 |
| vote (laplace-weighted) | 0.852 +/- 0.034 | 0.971 |

mean rules/fold: 138.8 (0.7s total)

---

## soybean

n=683, attributes=35, classes=19

### Fold 1 detail

```text
Combined RuleSet: 134 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.981] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 0.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 56.0, 'powdery-mildew': 0.0, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 0.0} plant-growth = abnorm ∧ leafspots-marg ≠ w-s-marg ∧ fruit-pods = ? → phytophthora-rot
  [w=0.981] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 0.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 44.0, 'powdery-mildew': 0.0, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 0.0} plant-stand = lt-normal ∧ leafspot-size ≠ gt-1/8 ∧ external-decay ≠ ? ∧ mold-growth = ? → phytophthora-rot
  [w=0.981] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 0.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 59.0, 'powdery-mildew': 0.0, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 0.0} germination = ? ∧ stem-cankers ≠ ? ∧ seed-discolor ≠ present → phytophthora-rot
  [w=0.981] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 0.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 56.0, 'powdery-mildew': 0.0, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 0.0} leafspots-halo ≠ no-yellow-halos ∧ canker-lesion = dk-brown-blk ∧ shriveling = ? → phytophthora-rot
  [w=0.981] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 0.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 44.0, 'powdery-mildew': 0.0, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 0.0} leafspots-halo ≠ no-yellow-halos ∧ sclerotia ≠ ? ∧ seed-size = ? → phytophthora-rot

Bottom 5:
  [w=0.213] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 3.0, 'bacterial-blight': 0.0, 'bacterial-pustule': 0.0, 'brown-spot': 0.0, 'brown-stem-rot': 25.0, 'charcoal-rot': 7.0, 'cyst-nematode': 10.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 16.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 6.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 16.0, 'powdery-mildew': 12.0, 'purple-seed-stain': 12.0, 'rhizoctonia-root-rot': 13.0} leafspots-marg ≠ w-s-marg ∧ leafspot-size ≠ lt-1/8 ∧ fruit-pods ≠ ? ∧ fruit-spots ≠ brown-w/blk-specks → brown-stem-rot
  [w=0.171] counts={'2-4-d-injury': 13.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 5.0, 'bacterial-blight': 16.0, 'bacterial-pustule': 8.0, 'brown-spot': 0.0, 'brown-stem-rot': 32.0, 'charcoal-rot': 17.0, 'cyst-nematode': 19.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 19.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 20.0, 'powdery-mildew': 15.0, 'purple-seed-stain': 4.0, 'rhizoctonia-root-rot': 14.0} leafspot-size ≠ gt-1/8 ∧ fruit-pods ≠ diseased ∧ fruit-spots ≠ colored ∧ roots ≠ rotted → brown-stem-rot
  [w=0.163] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 16.0, 'bacterial-blight': 16.0, 'bacterial-pustule': 8.0, 'brown-spot': 0.0, 'brown-stem-rot': 13.999999999999998, 'charcoal-rot': 0.0, 'cyst-nematode': 13.0, 'diaporthe-pod-&-stem-blight': 13.0, 'diaporthe-stem-canker': 21.0, 'downy-mildew': 0.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 0.0, 'powdery-mildew': 6.999999999999999, 'purple-seed-stain': 0.0, 'rhizoctonia-root-rot': 2.0} plant-stand ≠ lt-normal ∧ leafspot-size ≠ gt-1/8 ∧ canker-lesion ≠ tan ∧ seed ≠ ? → diaporthe-stem-canker
  [w=0.155] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 8.0, 'bacterial-blight': 9.0, 'bacterial-pustule': 16.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 0.0, 'cyst-nematode': 0.0, 'diaporthe-pod-&-stem-blight': 0.0, 'diaporthe-stem-canker': 15.0, 'downy-mildew': 12.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 22.0, 'powdery-mildew': 10.0, 'purple-seed-stain': 10.0, 'rhizoctonia-root-rot': 23.0} leafspots-halo ≠ no-yellow-halos ∧ int-discolor = none ∧ fruit-spots ≠ brown-w/blk-specks ∧ shriveling ≠ ? → rhizoctonia-root-rot
  [w=0.129] counts={'2-4-d-injury': 0.0, 'alternarialeaf-spot': 0.0, 'anthracnose': 5.0, 'bacterial-blight': 6.0, 'bacterial-pustule': 20.0, 'brown-spot': 0.0, 'brown-stem-rot': 0.0, 'charcoal-rot': 18.0, 'cyst-nematode': 7.0, 'diaporthe-pod-&-stem-blight': 13.0, 'diaporthe-stem-canker': 0.0, 'downy-mildew': 12.0, 'frog-eye-leaf-spot': 0.0, 'herbicide-injury': 0.0, 'phyllosticta-leaf-spot': 0.0, 'phytophthora-rot': 17.0, 'powdery-mildew': 21.0, 'purple-seed-stain': 5.0, 'rhizoctonia-root-rot': 19.0} leafspots-halo ≠ no-yellow-halos ∧ stem-cankers ≠ above-sec-nde ∧ int-discolor ≠ brown ∧ seed-size ≠ ? → powdery-mildew
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.819 +/- 0.052 | 1.000 |
| vote (unweighted) | 0.690 +/- 0.054 | 0.797 |
| micro-vote (pooled counts) | 0.663 +/- 0.060 | 0.785 |
| macro-vote (~sklearn) | 0.819 +/- 0.052 | 1.000 |
| micro-max | 0.441 +/- 0.041 | 0.483 |
| macro-max | 0.813 +/- 0.046 | 0.914 |
| max (precision) | 0.826 +/- 0.059 | 0.922 |
| vote (precision-weighted) | 0.760 +/- 0.051 | 0.868 |
| max (laplace) | 0.822 +/- 0.051 | 0.917 |
| vote (laplace-weighted) | 0.757 +/- 0.051 | 0.862 |

mean rules/fold: 126.2 (0.9s total)

---

## anneal

n=898, attributes=18, classes=5

### Fold 1 detail

```text
Combined RuleSet: 98 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.997] counts={'1': 0.0, '2': 0.0, '3': 396.0, '5': 0.0, 'U': 0.0} condition ≠ ? ∧ surface-quality ≠ ? → 3
  [w=0.997] counts={'1': 0.0, '2': 0.0, '3': 378.0, '5': 0.0, 'U': 0.0} hardness < 75.0 ∧ condition ≠ ? ∧ surface-quality ≠ ? → 3
  [w=0.996] counts={'1': 0.0, '2': 0.0, '3': 225.0, '5': 0.0, 'U': 0.0} formability = 2 ∧ strength < 375.0 ∧ surface-quality ≠ ? → 3
  [w=0.996] counts={'1': 0.0, '2': 0.0, '3': 235.0, '5': 0.0, 'U': 0.0} carbon < 3.5 ∧ hardness < 82.5 ∧ surface-quality = E ∧ enamelability = ? → 3
  [w=0.995] counts={'1': 0.0, '2': 0.0, '3': 199.0, '5': 0.0, 'U': 0.0} family = ? ∧ steel = A ∧ condition = S → 3

Bottom 5:
  [w=0.500] counts={'1': 1.0, '2': 2.0, '3': 0.0, '5': 0.0, 'U': 0.0} family = ? ∧ steel ≠ A ∧ strength >= 550.0 ∧ width < 983.0499877929688 → 2
  [w=0.461] counts={'1': 0.0, '2': 57.0, '3': 4.0, '5': 53.0, 'U': 0.0} steel ≠ K ∧ formability ≠ ? ∧ surface-quality = ? ∧ enamelability ≠ 2 → 2
  [w=0.429] counts={'1': 0.0, '2': 0.0, '3': 4.0, '5': 0.0, 'U': 3.0} hardness >= 75.0 ∧ surface-quality ≠ ? ∧ thick < 0.8005000054836273 ∧ len < 75.5 → 3
  [w=0.300] counts={'1': 6.0, '2': 6.0, '3': 2.0, '5': 0.0, 'U': 0.0} steel ≠ A ∧ carbon < 3.5 ∧ condition = ? ∧ surface-quality = ? → 1
  [w=0.281] counts={'1': 3.0, '2': 11.0, '3': 8.0, '5': 12.0, 'U': 0.0} carbon < 3.5 ∧ formability ≠ 3 ∧ surface-quality = ? ∧ shape = COIL → 5
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.962 +/- 0.018 | 1.000 |
| vote (unweighted) | 0.959 +/- 0.019 | 0.997 |
| micro-vote (pooled counts) | 0.855 +/- 0.014 | 0.886 |
| macro-vote (~sklearn) | 0.962 +/- 0.018 | 1.000 |
| micro-max | 0.785 +/- 0.028 | 0.816 |
| macro-max | 0.966 +/- 0.013 | 0.974 |
| max (precision) | 0.970 +/- 0.010 | 0.978 |
| vote (precision-weighted) | 0.961 +/- 0.016 | 0.994 |
| max (laplace) | 0.952 +/- 0.015 | 0.973 |
| vote (laplace-weighted) | 0.959 +/- 0.019 | 0.994 |

mean rules/fold: 95.4 (0.8s total)

---

## credit-g

n=1000, attributes=20, classes=2

### Fold 1 detail

```text
Combined RuleSet: 141 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.977] counts={'bad': 0.0, 'good': 54.0} duration < 8.5 ∧ credit_history ≠ all paid ∧ credit_amount < 3913.5 ∧ residence_since < 3.5 → good
  [w=0.971] counts={'bad': 0.0, 'good': 65.0} checking_status = no checking ∧ employment = >=7 ∧ num_dependents < 1.5 → good
  [w=0.960] counts={'bad': 3.0, 'good': 96.0} checking_status = no checking ∧ checking_status ≠ <0 ∧ purpose = radio/tv ∧ housing ≠ rent → good
  [w=0.944] counts={'bad': 0.0, 'good': 19.0} checking_status = no checking ∧ purpose = radio/tv ∧ savings_status = <100 ∧ installment_commitment < 2.5 → good
  [w=0.941] counts={'bad': 0.0, 'good': 16.0} checking_status ≠ 0<=X<200 ∧ checking_status ≠ <0 ∧ duration >= 34.5 ∧ employment = >=7 → good

Bottom 5:
  [w=0.429] counts={'bad': 7.0, 'good': 7.0} checking_status = 0<=X<200 ∧ duration < 15.5 ∧ other_payment_plans ≠ bank ∧ housing = rent → bad
  [w=0.400] counts={'bad': 2.0, 'good': 0.0} duration < 15.5 ∧ other_payment_plans = bank ∧ housing = rent → bad
  [w=0.400] counts={'bad': 2.0, 'good': 1.0} credit_history = critical/other existing credit ∧ installment_commitment >= 1.5 ∧ other_parties = co applicant ∧ property_magnitude = real estate → bad
  [w=0.375] counts={'bad': 4.0, 'good': 2.0} duration >= 15.5 ∧ credit_history ≠ existing paid ∧ savings_status ≠ <100 ∧ age >= 52.5 → bad
  [w=0.286] counts={'bad': 2.0, 'good': 2.0} checking_status = >=200 ∧ duration >= 8.5 ∧ credit_history ≠ existing paid ∧ savings_status = no known savings → bad
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.736 +/- 0.009 | 1.000 |
| vote (unweighted) | 0.727 +/- 0.022 | 0.947 |
| micro-vote (pooled counts) | 0.708 +/- 0.014 | 0.934 |
| macro-vote (~sklearn) | 0.736 +/- 0.009 | 1.000 |
| micro-max | 0.707 +/- 0.007 | 0.907 |
| macro-max | 0.722 +/- 0.007 | 0.910 |
| max (precision) | 0.734 +/- 0.017 | 0.940 |
| vote (precision-weighted) | 0.731 +/- 0.018 | 0.971 |
| max (laplace) | 0.732 +/- 0.013 | 0.950 |
| vote (laplace-weighted) | 0.733 +/- 0.020 | 0.969 |

mean rules/fold: 143.4 (1.0s total)

---

## cmc

n=1473, attributes=9, classes=3

### Fold 1 detail

```text
Combined RuleSet: 140 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.986] counts={'1': 72.0, '2': 0.0, '3': 0.0} Number_of_children_ever_born < 0.5 ∧ Husbands_occupation ≠ 4 → 1
  [w=0.985] counts={'1': 83.0, '2': 0.0, '3': 0.0} Number_of_children_ever_born < 0.5 ∧ Wifes_religion ≠ 0 → 1
  [w=0.983] counts={'1': 56.0, '2': 0.0, '3': 0.0} Wifes_education ≠ 2 ∧ Number_of_children_ever_born < 0.5 → 1
  [w=0.972] counts={'1': 30.0, '2': 0.0, '3': 0.0} Wifes_age >= 32.5 ∧ Wifes_education ≠ 4 ∧ Number_of_children_ever_born < 1.5 ∧ Number_of_children_ever_born < 2.5 → 1
  [w=0.960] counts={'1': 27.0, '2': 0.0, '3': 0.0} Wifes_age < 44.5 ∧ Wifes_education = 4 ∧ Number_of_children_ever_born < 0.5 → 1

Bottom 5:
  [w=0.380] counts={'1': 85.0, '2': 119.99999999999999, '3': 93.0} Wifes_education ≠ 1 ∧ Husbands_education = 4 ∧ Number_of_children_ever_born >= 0.5 ∧ Husbands_occupation = 1 → 2
  [w=0.362] counts={'1': 20.0, '2': 10.0, '3': 23.0} Number_of_children_ever_born >= 1.5 ∧ Number_of_children_ever_born < 2.5 ∧ Husbands_occupation = 1 ∧ Standard-of-living_index ≠ 3 → 3
  [w=0.350] counts={'1': 139.0, '2': 163.0, '3': 110.99999999999999} Wifes_age >= 32.5 ∧ Wifes_education ≠ 2 ∧ Number_of_children_ever_born >= 0.5 ∧ Media_exposure = 0 → 2
  [w=0.333] counts={'1': 2.0, '2': 3.0, '3': 0.0} Wifes_age < 39.5 ∧ Number_of_children_ever_born < 1.5 ∧ Wifes_now_working%3F = 0 ∧ Husbands_occupation = 4 → 2
  [w=0.264] counts={'1': 42.0, '2': 25.0, '3': 41.0} Wifes_age < 39.5 ∧ Number_of_children_ever_born >= 1.5 ∧ Wifes_now_working%3F = 0 ∧ Standard-of-living_index ≠ 2 → 1
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.527 +/- 0.029 | 1.000 |
| vote (unweighted) | 0.526 +/- 0.025 | 0.889 |
| micro-vote (pooled counts) | 0.525 +/- 0.015 | 0.873 |
| macro-vote (~sklearn) | 0.527 +/- 0.029 | 1.000 |
| micro-max | 0.439 +/- 0.018 | 0.553 |
| macro-max | 0.523 +/- 0.028 | 0.802 |
| max (precision) | 0.537 +/- 0.018 | 0.802 |
| vote (precision-weighted) | 0.525 +/- 0.019 | 0.916 |
| max (laplace) | 0.536 +/- 0.019 | 0.804 |
| vote (laplace-weighted) | 0.525 +/- 0.019 | 0.916 |

mean rules/fold: 136.8 (1.0s total)

---

## hypothyroid

n=3772, attributes=27, classes=4

### Fold 1 detail

```text
Combined RuleSet: 105 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=1.000] counts={'compensated_hypothyroid': 0.0, 'negative': 2709.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} TSH < 6.049999952316284 ∧ FTI >= 61.5 → negative
  [w=1.000] counts={'compensated_hypothyroid': 0.0, 'negative': 2213.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} TSH < 6.049999952316284 ∧ TSH < 19.5 ∧ T3 >= 1.25 ∧ FTI_measured ≠ f → negative
  [w=1.000] counts={'compensated_hypothyroid': 0.0, 'negative': 2152.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} TSH < 6.049999952316284 ∧ T3_measured ≠ f ∧ TT4 >= 51.5 → negative
  [w=0.999] counts={'compensated_hypothyroid': 0.0, 'negative': 1897.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} on_thyroxine ≠ t ∧ TSH < 6.049999952316284 ∧ TT4 >= 86.5 ∧ FTI >= 18.5 → negative
  [w=0.999] counts={'compensated_hypothyroid': 0.0, 'negative': 2706.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} TSH < 6.049999952316284 → negative

Bottom 5:
  [w=0.400] counts={'compensated_hypothyroid': 32.0, 'negative': 9.0, 'primary_hypothyroid': 24.0, 'secondary_hypothyroid': 0.0} age >= 6.5 ∧ TSH >= 19.5 ∧ T3 >= 0.8499999940395355 ∧ TT4 >= 37.5 → compensated_hypothyroid
  [w=0.400] counts={'compensated_hypothyroid': 3.0, 'negative': 0.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} sex = M ∧ TSH >= 41.5 ∧ FTI >= 61.5 → compensated_hypothyroid
  [w=0.400] counts={'compensated_hypothyroid': 0.0, 'negative': 0.0, 'primary_hypothyroid': 2.0, 'secondary_hypothyroid': 0.0} I131_treatment = t ∧ TSH >= 6.049999952316284 ∧ TSH >= 19.5 ∧ FTI >= 61.5 → primary_hypothyroid
  [w=0.389] counts={'compensated_hypothyroid': 9.0, 'negative': 6.0, 'primary_hypothyroid': 0.0, 'secondary_hypothyroid': 0.0} sick = t ∧ T3 < 1.25 ∧ TT4 < 86.5 ∧ FTI >= 18.5 → compensated_hypothyroid
  [w=0.375] counts={'compensated_hypothyroid': 0.0, 'negative': 3.0, 'primary_hypothyroid': 3.0, 'secondary_hypothyroid': 0.0} on_thyroxine = t ∧ query_hyperthyroid ≠ t ∧ TSH >= 6.049999952316284 ∧ TSH >= 41.5 → negative
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.977 +/- 0.011 | 1.000 |
| vote (unweighted) | 0.983 +/- 0.007 | 0.986 |
| micro-vote (pooled counts) | 0.945 +/- 0.004 | 0.963 |
| macro-vote (~sklearn) | 0.977 +/- 0.011 | 1.000 |
| micro-max | 0.938 +/- 0.005 | 0.956 |
| macro-max | 0.951 +/- 0.006 | 0.965 |
| max (precision) | 0.949 +/- 0.002 | 0.962 |
| vote (precision-weighted) | 0.980 +/- 0.012 | 0.994 |
| max (laplace) | 0.947 +/- 0.002 | 0.964 |
| vote (laplace-weighted) | 0.980 +/- 0.012 | 0.994 |

mean rules/fold: 117.6 (2.6s total)

---

## adult

n=48842, attributes=14, classes=2

### Fold 1 detail

```text
Combined RuleSet: 155 rules from 10 trees (sorted here by Laplace weight, the last heuristic annotate_weights ran):

Top 5:
  [w=0.998] counts={'<=50K': 0.0, '>50K': 562.0} workclass ≠ Self-emp-not-inc ∧ education-num >= 12.5 ∧ capitalgain = 4 → >50K
  [w=0.996] counts={'<=50K': 1.0, '>50K': 683.0} education-num >= 10.5 ∧ capitalgain = 4 ∧ capitalgain ≠ 0 ∧ capitalgain ≠ 1 → >50K
  [w=0.996] counts={'<=50K': 6735.0, '>50K': 30.999999999999996} age = 0 ∧ education ≠ Masters ∧ marital-status ≠ Married-civ-spouse ∧ capitalgain = 0 → <=50K
  [w=0.995] counts={'<=50K': 209.0, '>50K': 0.0} relationship = Not-in-family ∧ capitalgain = 1 ∧ capitalgain ≠ 0 → <=50K
  [w=0.994] counts={'<=50K': 3.0, '>50K': 453.0} education-num >= 10.5 ∧ marital-status = Married-civ-spouse ∧ relationship ≠ Not-in-family ∧ capitalgain = 4 → >50K

Bottom 5:
  [w=0.500] counts={'<=50K': 3.0, '>50K': 10.0} age = 3 ∧ education-num < 12.5 ∧ marital-status = Never-married ∧ capitalgain ≠ 0 → >50K
  [w=0.489] counts={'<=50K': 177.0, '>50K': 168.0} education-num >= 10.5 ∧ marital-status = Married-civ-spouse ∧ capitalgain = 0 ∧ hoursperweek = 1 → <=50K
  [w=0.471] counts={'<=50K': 61.99999999999999, '>50K': 54.0} education-num >= 10.5 ∧ marital-status = Married-civ-spouse ∧ capitalgain = 1 ∧ capitalgain ≠ 0 → <=50K
  [w=0.400] counts={'<=50K': 0.0, '>50K': 1.0} education-num >= 12.5 ∧ relationship = Husband ∧ hoursperweek = 0 ∧ native-country = Canada → >50K
  [w=0.333] counts={'<=50K': 1.0, '>50K': 3.0} education-num < 8.5 ∧ marital-status = Married-civ-spouse ∧ occupation = Tech-support ∧ relationship = Husband → >50K
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.832 +/- 0.004 | 1.000 |
| vote (unweighted) | 0.829 +/- 0.004 | 0.989 |
| micro-vote (pooled counts) | 0.806 +/- 0.007 | 0.937 |
| macro-vote (~sklearn) | 0.832 +/- 0.004 | 1.000 |
| micro-max | 0.779 +/- 0.007 | 0.887 |
| macro-max | 0.833 +/- 0.007 | 0.977 |
| max (precision) | 0.834 +/- 0.006 | 0.978 |
| vote (precision-weighted) | 0.830 +/- 0.005 | 0.992 |
| max (laplace) | 0.833 +/- 0.006 | 0.979 |
| vote (laplace-weighted) | 0.830 +/- 0.005 | 0.992 |

mean rules/fold: 155.6 (37.3s total)

---

## Overall summary

Mean accuracy per method, across all folds of each dataset.

| dataset | RF (sklearn) | vote (unweighted) | micro-vote (pooled counts) | macro-vote (~sklearn) | micro-max | macro-max | max (precision) | vote (precision-weighted) | max (laplace) | vote (laplace-weighted) | mean rules/fold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.963 | 0.959 | 0.952 | 0.963 | 0.952 | 0.940 | 0.952 | 0.961 | 0.954 | 0.961 | 111.8 |
| breast-cancer | 0.734 | 0.738 | 0.755 | 0.734 | 0.741 | 0.689 | 0.706 | 0.745 | 0.755 | 0.741 | 126.0 |
| colic | 0.862 | 0.864 | 0.845 | 0.862 | 0.761 | 0.802 | 0.834 | 0.867 | 0.848 | 0.862 | 122.4 |
| credit-approval | 0.861 | 0.851 | 0.839 | 0.861 | 0.806 | 0.843 | 0.858 | 0.852 | 0.864 | 0.852 | 138.8 |
| soybean | 0.819 | 0.690 | 0.663 | 0.819 | 0.441 | 0.813 | 0.826 | 0.760 | 0.822 | 0.757 | 126.2 |
| anneal | 0.962 | 0.959 | 0.855 | 0.962 | 0.785 | 0.966 | 0.970 | 0.961 | 0.952 | 0.959 | 95.4 |
| credit-g | 0.736 | 0.727 | 0.708 | 0.736 | 0.707 | 0.722 | 0.734 | 0.731 | 0.732 | 0.733 | 143.4 |
| cmc | 0.527 | 0.526 | 0.525 | 0.527 | 0.439 | 0.523 | 0.537 | 0.525 | 0.536 | 0.525 | 136.8 |
| hypothyroid | 0.977 | 0.983 | 0.945 | 0.977 | 0.938 | 0.951 | 0.949 | 0.980 | 0.947 | 0.980 | 117.6 |
| adult | 0.832 | 0.829 | 0.806 | 0.832 | 0.779 | 0.833 | 0.834 | 0.830 | 0.833 | 0.830 | 155.6 |
