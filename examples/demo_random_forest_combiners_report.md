# Random forest -> RuleSet import + combiner comparison demo

Generated 2026-09-24 22:19:34. N_FOLDS=5, N_ESTIMATORS=10, MAX_DEPTH=4, MAX_INTERVALS=6.

Only fold 1 of each dataset shows the top/bottom-5-by-weight rule detail; all folds contribute to the summary statistics. See the module docstring for what each method means, and why `macro-vote (~sklearn)` in particular is expected to land closest to the forest's own accuracy.

---

## vote

n=435, attributes=16, classes=2

### Fold 1 detail

```text
Combined RuleSet: 106 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.994] counts={'democrat': 175, 'republican': 0, <object object at 0x000001AF1E8AFB10>: 0} adoption-of-the-budget-resolution = y ∧ physician-fee-freeze = n → democrat
  [laplace=0.994] counts={'democrat': 172, 'republican': 0, <object object at 0x000001AF1E8AFB10>: 0} physician-fee-freeze = n ∧ aid-to-nicaraguan-contras ≠ n → democrat
  [laplace=0.994] counts={'democrat': 169, 'republican': 0, <object object at 0x000001AF1E8AFB10>: 0} physician-fee-freeze = n ∧ aid-to-nicaraguan-contras = y → democrat
  [laplace=0.994] counts={'democrat': 165, 'republican': 0, <object object at 0x000001AF1E8AFB10>: 0} adoption-of-the-budget-resolution ≠ ? ∧ physician-fee-freeze ≠ y ∧ el-salvador-aid ≠ y → democrat
  [laplace=0.994] counts={'democrat': 156, 'republican': 0, <object object at 0x000001AF1E8AFB10>: 0} physician-fee-freeze = n ∧ el-salvador-aid = n → democrat

Bottom 5:
  [laplace=0.462] counts={'democrat': 5, 'republican': 6, <object object at 0x000001AF1E8AFB10>: 0} physician-fee-freeze = y ∧ physician-fee-freeze ≠ n ∧ synfuels-corporation-cutback = y ∧ export-administration-act-south-africa ≠ y → democrat
  [laplace=0.400] counts={'republican': 1, 'democrat': 2, <object object at 0x000001AF1E8AFB10>: 0} handicapped-infants = n ∧ physician-fee-freeze = n ∧ aid-to-nicaraguan-contras = n ∧ superfund-right-to-sue ≠ y → republican
  [laplace=0.400] counts={'republican': 1, 'democrat': 2, <object object at 0x000001AF1E8AFB10>: 0} adoption-of-the-budget-resolution = n ∧ physician-fee-freeze = n ∧ aid-to-nicaraguan-contras ≠ y ∧ synfuels-corporation-cutback = n → republican
  [laplace=0.333] counts={'democrat': 1, 'republican': 3, <object object at 0x000001AF1E8AFB10>: 0} physician-fee-freeze = y ∧ anti-satellite-test-ban = n ∧ education-spending ≠ n ∧ duty-free-exports = y → democrat
  [laplace=0.333] counts={'republican': 2, 'democrat': 5, <object object at 0x000001AF1E8AFB10>: 0} water-project-cost-sharing ≠ y ∧ adoption-of-the-budget-resolution ≠ n ∧ anti-satellite-test-ban = n ∧ mx-missile ≠ n → republican
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.963 +/- 0.018 | 1.000 |
| vote (unweighted) | 0.959 +/- 0.014 | 0.995 |
| micro-vote (pooled counts) | 0.952 +/- 0.017 | 0.989 |
| macro-vote (~sklearn) | 0.961 +/- 0.016 | 0.998 |
| micro-max | 0.952 +/- 0.022 | 0.984 |
| macro-max | 0.956 +/- 0.009 | 0.979 |
| max (precision) | 0.952 +/- 0.009 | 0.979 |
| vote (precision-weighted) | 0.961 +/- 0.016 | 0.998 |
| max (laplace) | 0.954 +/- 0.018 | 0.986 |
| vote (laplace-weighted) | 0.961 +/- 0.016 | 0.998 |

mean rules/fold: 111.8 (0.5s total)

---

## breast-cancer

n=286, attributes=9, classes=2

### Fold 1 detail

```text
Combined RuleSet: 116 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.962] counts={'no-recurrence-events': 24, 'recurrence-events': 0, <object object at 0x000001AF1E8AFB10>: 0} tumor-size = 10-14 ∧ node-caps ≠ yes → no-recurrence-events
  [laplace=0.962] counts={'no-recurrence-events': 24, 'recurrence-events': 0, <object object at 0x000001AF1E8AFB10>: 0} tumor-size = 10-14 → no-recurrence-events
  [laplace=0.962] counts={'no-recurrence-events': 24, 'recurrence-events': 0, <object object at 0x000001AF1E8AFB10>: 0} tumor-size = 10-14 → no-recurrence-events
  [laplace=0.944] counts={'no-recurrence-events': 16, 'recurrence-events': 0, <object object at 0x000001AF1E8AFB10>: 0} age ≠ 50-59 ∧ tumor-size = 10-14 → no-recurrence-events
  [laplace=0.917] counts={'no-recurrence-events': 10, 'recurrence-events': 0, <object object at 0x000001AF1E8AFB10>: 0} tumor-size ≠ 10-14 ∧ deg-malig ≠ 3 ∧ breast-quad = left_up ∧ irradiat = yes → no-recurrence-events

Bottom 5:
  [laplace=0.467] counts={'recurrence-events': 6, 'no-recurrence-events': 7, <object object at 0x000001AF1E8AFB10>: 0} tumor-size = 30-34 ∧ inv-nodes = 0-2 ∧ node-caps = no ∧ deg-malig = 3 → recurrence-events
  [laplace=0.444] counts={'recurrence-events': 3, 'no-recurrence-events': 4, <object object at 0x000001AF1E8AFB10>: 0} age = 40-49 ∧ age ≠ 50-59 ∧ menopause = ge40 ∧ node-caps ≠ yes → recurrence-events
  [laplace=0.400] counts={'recurrence-events': 1, 'no-recurrence-events': 2, <object object at 0x000001AF1E8AFB10>: 0} age = 40-49 ∧ tumor-size = 30-34 ∧ inv-nodes = 0-2 ∧ irradiat = yes → recurrence-events
  [laplace=0.333] counts={'no-recurrence-events': 1, 'recurrence-events': 3, <object object at 0x000001AF1E8AFB10>: 0} age ≠ 30-39 ∧ node-caps ≠ no ∧ deg-malig = 2 ∧ breast-quad = right_up → no-recurrence-events
  [laplace=0.286] counts={'recurrence-events': 1, 'no-recurrence-events': 4, <object object at 0x000001AF1E8AFB10>: 0} age = 50-59 ∧ tumor-size = 20-24 ∧ node-caps ≠ ? ∧ breast = right → recurrence-events
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.752 +/- 0.021 | 1.000 |
| vote (unweighted) | 0.769 +/- 0.021 | 0.955 |
| micro-vote (pooled counts) | 0.738 +/- 0.027 | 0.923 |
| macro-vote (~sklearn) | 0.752 +/- 0.018 | 0.965 |
| micro-max | 0.724 +/- 0.014 | 0.909 |
| macro-max | 0.724 +/- 0.037 | 0.888 |
| max (precision) | 0.717 +/- 0.034 | 0.881 |
| vote (precision-weighted) | 0.759 +/- 0.014 | 0.986 |
| max (laplace) | 0.745 +/- 0.032 | 0.930 |
| vote (laplace-weighted) | 0.752 +/- 0.012 | 0.972 |

mean rules/fold: 126.8 (0.5s total)

---

## colic

n=368, attributes=26, classes=2

### Fold 1 detail

```text
Combined RuleSet: 114 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.973] counts={'2': 35, '1': 0, <object object at 0x000001AF1E8AFB10>: 0} respiratory_rate < 38.0 ∧ temperature_of_extremities ≠ 2 ∧ peripheral_pulse ≠ 3 ∧ site_of_lesion = 0 → 2
  [laplace=0.963] counts={'2': 25, '1': 0, <object object at 0x000001AF1E8AFB10>: 0} abdominal_distension ≠ 3 ∧ nasogastric_reflux ≠ 3 ∧ rectal_examination_-_feces = 1 ∧ site_of_lesion = 0 → 2
  [laplace=0.956] counts={'2': 42, '1': 1, <object object at 0x000001AF1E8AFB10>: 0} mucous_membranes ≠ 2 ∧ abdomen ≠ 2 ∧ outcome = 1 ∧ site_of_lesion = 0 → 2
  [laplace=0.955] counts={'2': 41, '1': 1, <object object at 0x000001AF1E8AFB10>: 0} total_protein < 63.5 ∧ site_of_lesion = 0 → 2
  [laplace=0.952] counts={'1': 19, '2': 0, <object object at 0x000001AF1E8AFB10>: 0} respiratory_rate < 35.5 ∧ pain ≠ 2 ∧ site_of_lesion = 3205 → 1

Bottom 5:
  [laplace=0.444] counts={'1': 3, '2': 4, <object object at 0x000001AF1E8AFB10>: 0} surgery ≠ 1 ∧ mucous_membranes = 5 ∧ outcome ≠ 1 ∧ site_of_lesion ≠ 5205 → 1
  [laplace=0.400] counts={'1': 1, '2': 2, <object object at 0x000001AF1E8AFB10>: 0} rectal_examination_-_feces = 4 ∧ packed_cell_volume >= 71.5 ∧ site_of_lesion ≠ 0 ∧ site_of_lesion ≠ 4124 → 1
  [laplace=0.398] counts={'2': 42, '1': 64, <object object at 0x000001AF1E8AFB10>: 0} temperature_of_extremities ≠ 3 ∧ abdominal_distension ≠ 1 ∧ abdomen ≠ 2 ∧ site_of_lesion ≠ 3205 → 2
  [laplace=0.364] counts={'2': 3, '1': 6, <object object at 0x000001AF1E8AFB10>: 0} rectal_examination_-_feces = 4 ∧ rectal_examination_-_feces ≠ 1 ∧ packed_cell_volume < 38.5 ∧ total_protein < 58.5 → 2
  [laplace=0.238] counts={'2': 4, '1': 15, <object object at 0x000001AF1E8AFB10>: 0} abdominal_distension ≠ 3 ∧ nasogastric_reflux ≠ 3 ∧ rectal_examination_-_feces = 1 ∧ site_of_lesion ≠ 0 → 2
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.861 +/- 0.013 | 1.000 |
| vote (unweighted) | 0.856 +/- 0.011 | 0.967 |
| micro-vote (pooled counts) | 0.845 +/- 0.011 | 0.940 |
| macro-vote (~sklearn) | 0.867 +/- 0.015 | 0.978 |
| micro-max | 0.731 +/- 0.043 | 0.810 |
| macro-max | 0.826 +/- 0.033 | 0.927 |
| max (precision) | 0.810 +/- 0.027 | 0.911 |
| vote (precision-weighted) | 0.867 +/- 0.015 | 0.978 |
| max (laplace) | 0.853 +/- 0.020 | 0.970 |
| vote (laplace-weighted) | 0.861 +/- 0.013 | 0.978 |

mean rules/fold: 118.8 (0.9s total)

---

## credit-approval

n=690, attributes=15, classes=2

### Fold 1 detail

```text
Combined RuleSet: 139 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.986] counts={'+': 68, '-': 0, <object object at 0x000001AF1E8AFB10>: 0} A1 = b ∧ A5 ≠ p ∧ A9 = t ∧ A15 >= 246.5 → +
  [laplace=0.979] counts={'+': 46, '-': 0, <object object at 0x000001AF1E8AFB10>: 0} A9 = t ∧ A11 >= 2.5 ∧ A11 < 9.5 ∧ A14 < 129.5 → +
  [laplace=0.976] counts={'+': 39, '-': 0, <object object at 0x000001AF1E8AFB10>: 0} A1 = b ∧ A8 >= 0.65 ∧ A11 >= 2.5 ∧ A15 >= 538.0 → +
  [laplace=0.974] counts={'+': 37, '-': 0, <object object at 0x000001AF1E8AFB10>: 0} A6 ≠ q ∧ A11 >= 4.5 ∧ A14 < 289.0 ∧ A15 >= 538.0 → +
  [laplace=0.967] counts={'+': 28, '-': 0, <object object at 0x000001AF1E8AFB10>: 0} A1 ≠ a ∧ A3 >= 4.2 ∧ A11 >= 2.5 ∧ A15 >= 246.5 → +

Bottom 5:
  [laplace=0.471] counts={'+': 7, '-': 8, <object object at 0x000001AF1E8AFB10>: 0} A2 >= 20.04 ∧ A3 < 0.23 ∧ A8 < 0.6 ∧ A14 < 369.5 → +
  [laplace=0.444] counts={'-': 27, '+': 34, <object object at 0x000001AF1E8AFB10>: 0} A3 >= 4.2 ∧ A3 < 18.8 ∧ A8 >= 1.02 ∧ A11 < 2.5 → -
  [laplace=0.400] counts={'-': 1, '+': 2, <object object at 0x000001AF1E8AFB10>: 0} A2 < 20.6 ∧ A5 = p ∧ A9 = t ∧ A15 >= 246.5 → -
  [laplace=0.400] counts={'-': 5, '+': 8, <object object at 0x000001AF1E8AFB10>: 0} A3 < 1.4 ∧ A9 = t ∧ A11 >= 0.5 ∧ A15 < 246.5 → -
  [laplace=0.300] counts={'-': 2, '+': 6, <object object at 0x000001AF1E8AFB10>: 0} A1 ≠ a ∧ A8 < 0.65 ∧ A11 >= 2.5 ∧ A15 >= 538.0 → -
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.854 +/- 0.026 | 1.000 |
| vote (unweighted) | 0.838 +/- 0.026 | 0.975 |
| micro-vote (pooled counts) | 0.829 +/- 0.015 | 0.926 |
| macro-vote (~sklearn) | 0.854 +/- 0.035 | 0.977 |
| micro-max | 0.801 +/- 0.021 | 0.870 |
| macro-max | 0.845 +/- 0.034 | 0.954 |
| max (precision) | 0.845 +/- 0.036 | 0.951 |
| vote (precision-weighted) | 0.845 +/- 0.035 | 0.983 |
| max (laplace) | 0.848 +/- 0.030 | 0.954 |
| vote (laplace-weighted) | 0.843 +/- 0.034 | 0.981 |

mean rules/fold: 139.8 (1.3s total)

---

## soybean

n=683, attributes=35, classes=19

### Fold 1 detail

```text
Combined RuleSet: 119 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.986] counts={'phytophthora-rot': 70, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 0, 'bacterial-pustule': 0, 'brown-spot': 0, 'brown-stem-rot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 0, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} plant-stand = lt-normal ∧ canker-lesion = dk-brown-blk ∧ fruit-pods ≠ diseased ∧ seed-size ≠ lt-norm → phytophthora-rot
  [laplace=0.986] counts={'phytophthora-rot': 70, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 0, 'bacterial-pustule': 0, 'brown-spot': 0, 'brown-stem-rot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 0, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} plant-stand ≠ normal ∧ plant-growth = abnorm ∧ canker-lesion = dk-brown-blk ∧ fruit-spots ≠ brown-w/blk-specks → phytophthora-rot
  [laplace=0.981] counts={'phytophthora-rot': 52, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 0, 'bacterial-pustule': 0, 'brown-spot': 0, 'brown-stem-rot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 0, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} leafspot-size ≠ gt-1/8 ∧ fruiting-bodies = ? ∧ int-discolor = none → phytophthora-rot
  [laplace=0.981] counts={'phytophthora-rot': 52, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 0, 'bacterial-pustule': 0, 'brown-spot': 0, 'brown-stem-rot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 0, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} external-decay ≠ ? ∧ seed-size = ? → phytophthora-rot
  [laplace=0.981] counts={'phytophthora-rot': 52, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 0, 'bacterial-pustule': 0, 'brown-spot': 0, 'brown-stem-rot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 0, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} fruiting-bodies = ? ∧ sclerotia ≠ ? → phytophthora-rot

Bottom 5:
  [laplace=0.264] counts={'brown-stem-rot': 28, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 0, 'bacterial-blight': 16, 'bacterial-pustule': 16, 'brown-spot': 0, 'charcoal-rot': 16, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'phytophthora-rot': 0, 'powdery-mildew': 16, 'purple-seed-stain': 16, 'rhizoctonia-root-rot': 0, <object object at 0x000001AF1E8AFB10>: 0} leafspots-halo ≠ ? ∧ leafspot-size ≠ gt-1/8 ∧ stem-cankers = absent ∧ fruiting-bodies ≠ ? → brown-stem-rot
  [laplace=0.250] counts={'brown-stem-rot': 29, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 8, 'bacterial-blight': 0, 'bacterial-pustule': 14, 'brown-spot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 12, 'diaporthe-stem-canker': 16, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'phytophthora-rot': 0, 'powdery-mildew': 16, 'purple-seed-stain': 7, 'rhizoctonia-root-rot': 16, <object object at 0x000001AF1E8AFB10>: 0} leafspots-marg ≠ w-s-marg ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor ≠ black ∧ sclerotia ≠ ? → brown-stem-rot
  [laplace=0.220] counts={'brown-spot': 74, '2-4-d-injury': 13, 'alternarialeaf-spot': 73, 'anthracnose': 8, 'bacterial-blight': 16, 'bacterial-pustule': 16, 'brown-stem-rot': 0, 'charcoal-rot': 16, 'cyst-nematode': 11, 'diaporthe-pod-&-stem-blight': 12, 'diaporthe-stem-canker': 16, 'downy-mildew': 0, 'frog-eye-leaf-spot': 30, 'herbicide-injury': 6, 'phyllosticta-leaf-spot': 16, 'phytophthora-rot': 0, 'powdery-mildew': 0, 'purple-seed-stain': 16, 'rhizoctonia-root-rot': 16, <object object at 0x000001AF1E8AFB10>: 0} leaf-mild ≠ lower-surf ∧ leaf-mild ≠ upper-surf ∧ canker-lesion ≠ dk-brown-blk ∧ int-discolor ≠ brown → brown-spot
  [laplace=0.209] counts={'brown-stem-rot': 28, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 11, 'bacterial-blight': 16, 'bacterial-pustule': 16, 'brown-spot': 0, 'charcoal-rot': 0, 'cyst-nematode': 0, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 0, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'phytophthora-rot': 18, 'powdery-mildew': 16, 'purple-seed-stain': 16, 'rhizoctonia-root-rot': 16, <object object at 0x000001AF1E8AFB10>: 0} leafspot-size ≠ gt-1/8 ∧ fruiting-bodies ≠ ? ∧ fruiting-bodies ≠ present ∧ int-discolor ≠ black → brown-stem-rot
  [laplace=0.180] counts={'brown-stem-rot': 28, '2-4-d-injury': 0, 'alternarialeaf-spot': 0, 'anthracnose': 4, 'bacterial-blight': 16, 'bacterial-pustule': 2, 'brown-spot': 0, 'charcoal-rot': 16, 'cyst-nematode': 11, 'diaporthe-pod-&-stem-blight': 0, 'diaporthe-stem-canker': 16, 'downy-mildew': 0, 'frog-eye-leaf-spot': 0, 'herbicide-injury': 0, 'phyllosticta-leaf-spot': 0, 'phytophthora-rot': 18, 'powdery-mildew': 16, 'purple-seed-stain': 16, 'rhizoctonia-root-rot': 16, <object object at 0x000001AF1E8AFB10>: 0} leafspots-marg ≠ no-w-s-marg ∧ leafspot-size ≠ gt-1/8 ∧ fruit-spots ≠ brown-w/blk-specks ∧ seed-size ≠ ? → brown-stem-rot
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.805 +/- 0.041 | 1.000 |
| vote (unweighted) | 0.652 +/- 0.025 | 0.774 |
| micro-vote (pooled counts) | 0.660 +/- 0.060 | 0.791 |
| macro-vote (~sklearn) | 0.805 +/- 0.047 | 0.960 |
| micro-max | 0.436 +/- 0.030 | 0.477 |
| macro-max | 0.810 +/- 0.032 | 0.884 |
| max (precision) | 0.813 +/- 0.035 | 0.883 |
| vote (precision-weighted) | 0.750 +/- 0.019 | 0.900 |
| max (laplace) | 0.810 +/- 0.032 | 0.886 |
| vote (laplace-weighted) | 0.751 +/- 0.018 | 0.902 |

mean rules/fold: 118.4 (1.7s total)

---

## anneal

n=898, attributes=18, classes=5

### Fold 1 detail

```text
Combined RuleSet: 101 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.996] counts={'3': 222, '1': 0, '2': 0, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} condition = S ∧ surface-quality = E → 3
  [laplace=0.996] counts={'3': 222, '1': 0, '2': 0, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} condition ≠ ? ∧ surface-quality = E → 3
  [laplace=0.994] counts={'3': 168, '1': 0, '2': 0, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} condition ≠ ? ∧ surface-quality ≠ ? ∧ surface-quality ≠ E → 3
  [laplace=0.993] counts={'3': 447, '1': 0, '2': 0, '5': 0, 'U': 2, <object object at 0x000001AF1E8AFB10>: 0} family = ? ∧ hardness < 75.0 ∧ strength < 550.0 ∧ surface-quality ≠ ? → 3
  [laplace=0.992] counts={'3': 130, '1': 0, '2': 0, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} family = ? ∧ steel ≠ A ∧ surface-quality = E → 3

Bottom 5:
  [laplace=0.544] counts={'3': 91, '1': 0, '2': 22, '5': 54, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} hardness < 75.0 ∧ formability ≠ 2 ∧ formability ≠ ? ∧ enamelability ≠ 2 → 3
  [laplace=0.522] counts={'2': 70, '1': 3, '3': 61, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} family = ? ∧ hardness < 75.0 ∧ strength < 550.0 ∧ surface-quality = ? → 2
  [laplace=0.511] counts={'5': 23, '1': 0, '2': 22, '3': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} hardness < 82.0 ∧ formability = 3 ∧ surface-quality = ? ∧ thick >= 0.7995 → 5
  [laplace=0.500] counts={'U': 2, '1': 0, '2': 0, '3': 2, '5': 0, <object object at 0x000001AF1E8AFB10>: 0} family = ? ∧ hardness >= 75.0 ∧ hardness < 82.0 ∧ strength < 550.0 → U
  [laplace=0.500] counts={'2': 1, '1': 0, '3': 1, '5': 0, 'U': 0, <object object at 0x000001AF1E8AFB10>: 0} family = ? ∧ strength >= 450.0 ∧ width >= 1095.0 ∧ len >= 76.0 → 2
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.968 +/- 0.016 | 1.000 |
| vote (unweighted) | 0.935 +/- 0.019 | 0.959 |
| micro-vote (pooled counts) | 0.840 +/- 0.016 | 0.870 |
| macro-vote (~sklearn) | 0.972 +/- 0.015 | 0.996 |
| micro-max | 0.774 +/- 0.024 | 0.804 |
| macro-max | 0.973 +/- 0.026 | 0.966 |
| max (precision) | 0.971 +/- 0.026 | 0.968 |
| vote (precision-weighted) | 0.959 +/- 0.020 | 0.991 |
| max (laplace) | 0.958 +/- 0.025 | 0.970 |
| vote (laplace-weighted) | 0.960 +/- 0.019 | 0.990 |

mean rules/fold: 96.6 (1.2s total)

---

## credit-g

n=1000, attributes=20, classes=2

### Fold 1 detail

```text
Combined RuleSet: 138 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.950] counts={'good': 112, 'bad': 5, <object object at 0x000001AF1E8AFB10>: 0} checking_status = no checking ∧ checking_status ≠ <0 ∧ credit_history = critical/other existing credit ∧ credit_amount < 10848.0 → good
  [laplace=0.942] counts={'good': 209, 'bad': 12, <object object at 0x000001AF1E8AFB10>: 0} checking_status = no checking ∧ employment ≠ <1 ∧ employment ≠ unemployed ∧ other_payment_plans = none → good
  [laplace=0.939] counts={'good': 45, 'bad': 2, <object object at 0x000001AF1E8AFB10>: 0} duration < 8.5 ∧ credit_amount < 3913.5 ∧ personal_status ≠ female div/dep/mar → good
  [laplace=0.932] counts={'good': 54, 'bad': 3, <object object at 0x000001AF1E8AFB10>: 0} duration >= 8.5 ∧ purpose = radio/tv ∧ credit_amount < 10848.0 ∧ employment = >=7 → good
  [laplace=0.931] counts={'good': 26, 'bad': 1, <object object at 0x000001AF1E8AFB10>: 0} credit_history = critical/other existing credit ∧ purpose ≠ new car ∧ property_magnitude = real estate ∧ age >= 34.5 → good

Bottom 5:
  [laplace=0.444] counts={'bad': 3, 'good': 4, <object object at 0x000001AF1E8AFB10>: 0} checking_status = no checking ∧ credit_history ≠ critical/other existing credit ∧ credit_amount < 10848.0 ∧ other_parties = co applicant → bad
  [laplace=0.442] counts={'good': 18, 'bad': 23, <object object at 0x000001AF1E8AFB10>: 0} checking_status = <0 ∧ checking_status ≠ no checking ∧ credit_history ≠ critical/other existing credit ∧ other_payment_plans ≠ none → good
  [laplace=0.400] counts={'good': 1, 'bad': 2, <object object at 0x000001AF1E8AFB10>: 0} checking_status ≠ <0 ∧ credit_amount >= 10848.0 ∧ savings_status = no known savings ∧ residence_since >= 1.5 → good
  [laplace=0.400] counts={'bad': 1, 'good': 2, <object object at 0x000001AF1E8AFB10>: 0} duration < 34.0 ∧ credit_history = critical/other existing credit ∧ other_payment_plans = stores ∧ num_dependents < 1.5 → bad
  [laplace=0.333] counts={'bad': 2, 'good': 5, <object object at 0x000001AF1E8AFB10>: 0} duration < 8.5 ∧ credit_amount < 3913.5 ∧ personal_status = female div/dep/mar ∧ property_magnitude = life insurance → bad
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.725 +/- 0.007 | 1.000 |
| vote (unweighted) | 0.724 +/- 0.016 | 0.965 |
| micro-vote (pooled counts) | 0.709 +/- 0.005 | 0.944 |
| macro-vote (~sklearn) | 0.718 +/- 0.009 | 0.973 |
| micro-max | 0.705 +/- 0.004 | 0.936 |
| macro-max | 0.719 +/- 0.010 | 0.950 |
| max (precision) | 0.718 +/- 0.009 | 0.951 |
| vote (precision-weighted) | 0.724 +/- 0.012 | 0.973 |
| max (laplace) | 0.722 +/- 0.013 | 0.961 |
| vote (laplace-weighted) | 0.725 +/- 0.011 | 0.972 |

mean rules/fold: 146.8 (1.8s total)

---

## cmc

n=1473, attributes=9, classes=3

### Fold 1 detail

```text
Combined RuleSet: 143 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.986] counts={'1': 67, '2': 0, '3': 0, <object object at 0x000001AF1E8AFB10>: 0} Husbands_education ≠ 2 ∧ Number_of_children_ever_born < 0.5 → 1
  [laplace=0.985] counts={'1': 64, '2': 0, '3': 0, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age < 40.5 ∧ Number_of_children_ever_born < 0.5 ∧ Husbands_occupation ≠ 4 → 1
  [laplace=0.981] counts={'1': 52, '2': 0, '3': 0, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age < 44.5 ∧ Wifes_education ≠ 2 ∧ Number_of_children_ever_born < 0.5 → 1
  [laplace=0.981] counts={'1': 50, '2': 0, '3': 0, <object object at 0x000001AF1E8AFB10>: 0} Husbands_education = 4 ∧ Number_of_children_ever_born < 0.5 → 1
  [laplace=0.963] counts={'1': 25, '2': 0, '3': 0, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age >= 44.5 ∧ Wifes_education = 1 ∧ Number_of_children_ever_born < 7.5 → 1

Bottom 5:
  [laplace=0.333] counts={'1': 16, '2': 20, '3': 13, <object object at 0x000001AF1E8AFB10>: 0} Number_of_children_ever_born >= 0.5 ∧ Number_of_children_ever_born >= 2.5 ∧ Number_of_children_ever_born >= 5.5 ∧ Husbands_occupation = 1 → 1
  [laplace=0.330] counts={'3': 36, '1': 40, '2': 34, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age < 39.5 ∧ Number_of_children_ever_born >= 0.5 ∧ Number_of_children_ever_born < 2.5 ∧ Husbands_occupation = 1 → 3
  [laplace=0.330] counts={'3': 28, '1': 27, '2': 31, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age < 32.5 ∧ Wifes_education = 4 ∧ Number_of_children_ever_born >= 0.5 ∧ Husbands_occupation = 1 → 3
  [laplace=0.280] counts={'3': 6, '1': 5, '2': 12, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age >= 40.5 ∧ Wifes_education = 4 ∧ Number_of_children_ever_born >= 5.5 ∧ Wifes_religion = 1 → 3
  [laplace=0.276] counts={'1': 15, '2': 27, '3': 14, <object object at 0x000001AF1E8AFB10>: 0} Wifes_age >= 40.5 ∧ Wifes_education = 4 ∧ Number_of_children_ever_born >= 1.5 ∧ Number_of_children_ever_born < 5.5 → 1
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.540 +/- 0.029 | 1.000 |
| vote (unweighted) | 0.542 +/- 0.023 | 0.931 |
| micro-vote (pooled counts) | 0.540 +/- 0.030 | 0.872 |
| macro-vote (~sklearn) | 0.545 +/- 0.027 | 0.949 |
| micro-max | 0.473 +/- 0.009 | 0.656 |
| macro-max | 0.536 +/- 0.037 | 0.820 |
| max (precision) | 0.536 +/- 0.037 | 0.820 |
| vote (precision-weighted) | 0.541 +/- 0.024 | 0.944 |
| max (laplace) | 0.538 +/- 0.036 | 0.822 |
| vote (laplace-weighted) | 0.541 +/- 0.024 | 0.944 |

mean rules/fold: 140.2 (2.1s total)

---

## hypothyroid

n=3772, attributes=27, classes=4

### Fold 1 detail

```text
Combined RuleSet: 126 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=1.000] counts={'negative': 2698, 'compensated_hypothyroid': 0, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH < 6.05 ∧ TT4 >= 51.5 → negative
  [laplace=1.000] counts={'negative': 2617, 'compensated_hypothyroid': 0, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} age >= 7.5 ∧ TSH < 6.05 ∧ T3 >= 0.85 ∧ TT4 >= 51.5 → negative
  [laplace=1.000] counts={'negative': 2544, 'compensated_hypothyroid': 0, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH < 6.05 ∧ TT4 >= 72.5 → negative
  [laplace=1.000] counts={'negative': 2508, 'compensated_hypothyroid': 0, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH < 6.05 ∧ TSH < 41.5 ∧ T3 >= 1.25 → negative
  [laplace=1.000] counts={'negative': 2440, 'compensated_hypothyroid': 0, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH < 6.05 ∧ TSH < 19.5 ∧ FTI >= 85.5 → negative

Bottom 5:
  [laplace=0.400] counts={'primary_hypothyroid': 1, 'compensated_hypothyroid': 1, 'negative': 1, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} age < 6.5 ∧ TSH < 8.55 ∧ TT4 >= 18.5 ∧ FTI_measured = f → primary_hypothyroid
  [laplace=0.400] counts={'secondary_hypothyroid': 1, 'compensated_hypothyroid': 0, 'negative': 1, 'primary_hypothyroid': 1, <object object at 0x000001AF1E8AFB10>: 0} sex = F ∧ TSH < 8.55 ∧ T3 < 1.25 ∧ FTI < 61.5 → secondary_hypothyroid
  [laplace=0.400] counts={'negative': 1, 'compensated_hypothyroid': 2, 'primary_hypothyroid': 0, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} I131_treatment = t ∧ TSH >= 6.05 ∧ TT4 >= 72.5 ∧ referral_source ≠ STMW → negative
  [laplace=0.333] counts={'compensated_hypothyroid': 1, 'negative': 1, 'primary_hypothyroid': 2, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH >= 6.05 ∧ TT4 >= 72.5 ∧ T4U < 1.075 ∧ referral_source = STMW → compensated_hypothyroid
  [laplace=0.250] counts={'compensated_hypothyroid': 1, 'negative': 1, 'primary_hypothyroid': 4, 'secondary_hypothyroid': 0, <object object at 0x000001AF1E8AFB10>: 0} TSH >= 6.05 ∧ TSH < 19.5 ∧ TSH < 32.5 ∧ TT4 < 51.5 → compensated_hypothyroid
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.972 +/- 0.004 | 1.000 |
| vote (unweighted) | 0.973 +/- 0.005 | 0.993 |
| micro-vote (pooled counts) | 0.956 +/- 0.003 | 0.971 |
| macro-vote (~sklearn) | 0.972 +/- 0.004 | 0.999 |
| micro-max | 0.949 +/- 0.008 | 0.962 |
| macro-max | 0.962 +/- 0.010 | 0.961 |
| max (precision) | 0.962 +/- 0.010 | 0.960 |
| vote (precision-weighted) | 0.973 +/- 0.003 | 0.995 |
| max (laplace) | 0.960 +/- 0.010 | 0.962 |
| vote (laplace-weighted) | 0.973 +/- 0.003 | 0.995 |

mean rules/fold: 123.8 (3.7s total)

---

## adult

n=48842, attributes=14, classes=2

### Fold 1 detail

```text
Combined RuleSet: 156 rules from 10 trees (sorted here by each leaf's own measured Laplace score):

Top 5:
  [laplace=0.998] counts={'>50K': 503, '<=50K': 0, <object object at 0x000001AF1E8AFB10>: 0} workclass ≠ Self-emp-not-inc ∧ education-num >= 10.5 ∧ marital-status ≠ Never-married ∧ capitalgain = 4 → >50K
  [laplace=0.995] counts={'>50K': 194, '<=50K': 0, <object object at 0x000001AF1E8AFB10>: 0} education ≠ Masters ∧ education-num >= 13.5 ∧ capitalgain = 4 → >50K
  [laplace=0.995] counts={'>50K': 194, '<=50K': 0, <object object at 0x000001AF1E8AFB10>: 0} education ≠ Preschool ∧ education-num >= 10.5 ∧ education-num >= 14.5 ∧ capitalgain = 4 → >50K
  [laplace=0.995] counts={'>50K': 380, '<=50K': 1, <object object at 0x000001AF1E8AFB10>: 0} occupation ≠ Protective-serv ∧ capitalgain = 3 ∧ capitalgain ≠ 4 ∧ hoursperweek ≠ 1 → >50K
  [laplace=0.994] counts={'<=50K': 2703, '>50K': 14, <object object at 0x000001AF1E8AFB10>: 0} age = 0 ∧ education-num < 10.5 ∧ occupation ≠ Exec-managerial ∧ sex = Female → <=50K

Bottom 5:
  [laplace=0.500] counts={'>50K': 51, '<=50K': 51, <object object at 0x000001AF1E8AFB10>: 0} age ≠ 2 ∧ education = Doctorate ∧ education-num >= 13.5 ∧ marital-status ≠ Married-civ-spouse → >50K
  [laplace=0.500] counts={'<=50K': 1, '>50K': 1, <object object at 0x000001AF1E8AFB10>: 0} education-num >= 13.5 ∧ occupation = Exec-managerial ∧ hoursperweek = 3 ∧ native-country = France → <=50K
  [laplace=0.495] counts={'<=50K': 47, '>50K': 48, <object object at 0x000001AF1E8AFB10>: 0} education = Prof-school ∧ education-num >= 10.5 ∧ marital-status = Never-married ∧ relationship = Not-in-family → <=50K
  [laplace=0.483] counts={'>50K': 112, '<=50K': 120, <object object at 0x000001AF1E8AFB10>: 0} education-num >= 10.5 ∧ occupation = Exec-managerial ∧ relationship = Not-in-family ∧ hoursperweek = 3 → >50K
  [laplace=0.222] counts={'<=50K': 1, '>50K': 6, <object object at 0x000001AF1E8AFB10>: 0} education ≠ Masters ∧ occupation = Protective-serv ∧ capitalgain = 3 ∧ capitalgain ≠ 4 → <=50K
```

### Method comparison (mean +/- std over folds)

| method | accuracy | agreement with RF |
|---|---|---|
| RF (sklearn) | 0.828 +/- 0.004 | 1.000 |
| vote (unweighted) | 0.825 +/- 0.003 | 0.977 |
| micro-vote (pooled counts) | 0.793 +/- 0.015 | 0.932 |
| macro-vote (~sklearn) | 0.828 +/- 0.004 | 0.999 |
| micro-max | 0.774 +/- 0.010 | 0.899 |
| macro-max | 0.828 +/- 0.004 | 0.959 |
| max (precision) | 0.828 +/- 0.004 | 0.959 |
| vote (precision-weighted) | 0.825 +/- 0.003 | 0.991 |
| max (laplace) | 0.826 +/- 0.005 | 0.958 |
| vote (laplace-weighted) | 0.825 +/- 0.003 | 0.991 |

mean rules/fold: 152.8 (61.1s total)

---

## Overall summary

Mean accuracy per method, across all folds of each dataset.

| dataset | RF (sklearn) | vote (unweighted) | micro-vote (pooled counts) | macro-vote (~sklearn) | micro-max | macro-max | max (precision) | vote (precision-weighted) | max (laplace) | vote (laplace-weighted) | mean rules/fold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vote | 0.963 | 0.959 | 0.952 | 0.961 | 0.952 | 0.956 | 0.952 | 0.961 | 0.954 | 0.961 | 111.8 |
| breast-cancer | 0.752 | 0.769 | 0.738 | 0.752 | 0.724 | 0.724 | 0.717 | 0.759 | 0.745 | 0.752 | 126.8 |
| colic | 0.861 | 0.856 | 0.845 | 0.867 | 0.731 | 0.826 | 0.810 | 0.867 | 0.853 | 0.861 | 118.8 |
| credit-approval | 0.854 | 0.838 | 0.829 | 0.854 | 0.801 | 0.845 | 0.845 | 0.845 | 0.848 | 0.843 | 139.8 |
| soybean | 0.805 | 0.652 | 0.660 | 0.805 | 0.436 | 0.810 | 0.813 | 0.750 | 0.810 | 0.751 | 118.4 |
| anneal | 0.968 | 0.935 | 0.840 | 0.972 | 0.774 | 0.973 | 0.971 | 0.959 | 0.958 | 0.960 | 96.6 |
| credit-g | 0.725 | 0.724 | 0.709 | 0.718 | 0.705 | 0.719 | 0.718 | 0.724 | 0.722 | 0.725 | 146.8 |
| cmc | 0.540 | 0.542 | 0.540 | 0.545 | 0.473 | 0.536 | 0.536 | 0.541 | 0.538 | 0.541 | 140.2 |
| hypothyroid | 0.972 | 0.973 | 0.956 | 0.972 | 0.949 | 0.962 | 0.962 | 0.973 | 0.960 | 0.973 | 123.8 |
| adult | 0.828 | 0.825 | 0.793 | 0.828 | 0.774 | 0.828 | 0.828 | 0.825 | 0.826 | 0.825 | 152.8 |
