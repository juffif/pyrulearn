# Data representations: same rules, different speed

**Full run.** Quick sanity check instead: `python demos/representations.py`.

pyrulearn's learners never look at the data directly. They ask it two things -- which examples a rule covers (`coverage(rule)`) and which features an example has (`features_of(row)`) -- so the data can be stored in any structure that answers these questions. Four such representations are implemented:

- **HBitmap** (horizontal bitmap) -- a bit matrix packed by example, one bit per example and feature. Covering a rule checks every example's row against the rule's features, whatever the density.
- **VBitmap** (vertical bitmap) -- the same bit matrix packed by feature: one bit-vector over the examples per feature. Covering a rule ANDs its features' bit-vectors, and counting is a popcount; same memory as HBitmap.
- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are inserted into a prefix tree of their true features, and each feature keeps the list of tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work grows with how often the features are true, not with the number of examples.
- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix tree.

All four must lead to exactly the same rules, so the only thing that differs is time. This demo measures that time for five learners -- CN2, PFossil, Pypper, PyLORD and ENDER, each with its default settings -- and checks the rules and the test-set predictions against HBitmap at every point. Each fit runs in a separate process and is capped at 300s; after a time-out, the larger points of that curve are skipped. Everything ran one at a time. The numerical libraries were limited to one thread (`OMP_NUM_THREADS=1` and its relatives), so each fit used one core.

**Equality check:** at 147 points where HBitmap and at least one other representation finished, 147 gave identical rules and predictions for every representation.

## 1. Training-set size

Training sets of doubling size, drawn from one shuffled pool of the `adult` dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the same 20% test set. Numeric attributes are discretized per training set (`max_intervals=8`). Two encodings: **with negation** (pyrulearn's default -- every test comes with its negation, so exactly half the features are true in every example) and **without negation** (no `x != v` features; numeric attributes keep both `x >= t` and `x < t`, so it is sparser only in the nominal attributes). The first is the bitmaps' best case and the list-based representations' worst; the second is where those should gain.

At the largest training set: with negation 278 binary features, density 0.493; without negation 182 binary features, density 0.274 (density = share of features that are true).

One pair of plots per learner: left with a logarithmic time axis (ratios -- a constant factor between two representations is a constant gap), right with a linear one (absolute seconds saved).

<img src="representations_plots/size_cn2.png" alt="CN2 fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_cn2_linear.png" alt="CN2 fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pfossil.png" alt="PFossil fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pfossil_linear.png" alt="PFossil fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pypper.png" alt="Pypper fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pypper_linear.png" alt="Pypper fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pylord.png" alt="PyLORD fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pylord_linear.png" alt="PyLORD fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_ender.png" alt="ENDER fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_ender_linear.png" alt="ENDER fit time vs. training-set size, linear scale" width="49%">

Fit time in seconds, with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.08 | 0.07 | 0.27 | 0.79 | 1.71 | 6.15 | 53.28 |
| CN2 | VBitmap | 0.06 (73%) | 0.06 (81%) | 0.24 (91%) | 0.51 (64%) | 0.90 (53%) | 2.59 (42%) | 16.36 (31%) |
| CN2 | NList | 0.10 (116%) | 0.06 (90%) | 0.38 (143%) | 0.80 (101%) | 1.74 (101%) | 3.31 (54%) | 23.22 (44%) |
| CN2 | Sparse | 0.10 (116%) | 0.06 (92%) | 0.37 (139%) | 0.51 (64%) | 1.22 (71%) | 4.24 (69%) | 31.00 (58%) |
| PFossil | HBitmap | 0.02 | 0.04 | 0.05 | 0.16 | 0.19 | 0.54 | 1.04 |
| PFossil | VBitmap | 0.02 (73%) | 0.02 (55%) | 0.03 (53%) | 0.06 (39%) | 0.08 (41%) | 0.21 (38%) | 0.28 (27%) |
| PFossil | NList | 0.03 (103%) | 0.04 (81%) | 0.04 (83%) | 0.13 (80%) | 0.12 (65%) | 0.24 (45%) | 0.46 (44%) |
| PFossil | Sparse | 0.03 (107%) | 0.04 (85%) | 0.05 (94%) | 0.11 (65%) | 0.12 (62%) | 0.28 (51%) | 0.50 (49%) |
| Pypper | HBitmap | 0.14 | 0.29 | 0.31 | 0.88 | 2.08 | 3.28 | 28.40 |
| Pypper | VBitmap | 0.04 (30%) | 0.10 (35%) | 0.09 (29%) | 0.28 (32%) | 0.29 (14%) | 0.40 (12%) | 1.68 (6%) |
| Pypper | NList | 0.08 (58%) | 0.23 (81%) | 0.16 (52%) | 0.76 (87%) | 0.74 (35%) | 0.92 (28%) | 5.40 (19%) |
| Pypper | Sparse | 0.06 (45%) | 0.17 (60%) | 0.27 (86%) | 0.97 (111%) | 1.09 (53%) | 1.34 (41%) | 13.97 (49%) |
| PyLORD | HBitmap | 1.16 | 3.17 | 12.82 | 48.97 | 139.70 | time-out |  |
| PyLORD | VBitmap | 0.89 (77%) | 2.58 (81%) | 7.93 (62%) | 22.59 (46%) | 52.29 (37%) | 149.20 | time-out |
| PyLORD | NList | 1.16 (100%) | 4.00 (126%) | 12.22 (95%) | 38.71 (79%) | 98.68 (71%) | time-out |  |
| PyLORD | Sparse | 1.22 (106%) | 4.20 (132%) | 14.88 (116%) | 48.14 (98%) | 138.45 (99%) | time-out |  |
| ENDER | HBitmap | 1.00 | 2.13 | 4.37 | 6.10 | 8.46 | 15.46 | 29.13 |
| ENDER | VBitmap | 1.14 (114%) | 1.50 (70%) | 2.81 (64%) | 4.08 (67%) | 8.16 (96%) | 13.24 (86%) | 24.12 (83%) |
| ENDER | NList | 1.24 (124%) | 1.93 (91%) | 3.04 (69%) | 5.78 (95%) | 6.98 (82%) | 11.81 (76%) | 21.22 (73%) |
| ENDER | Sparse | 1.57 (158%) | 2.29 (108%) | 3.84 (88%) | 4.69 (77%) | 7.02 (83%) | 12.46 (81%) | 22.75 (78%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.09 | 0.10 | 0.17 | 0.51 | 1.20 | 4.86 | 19.82 |
| CN2 | VBitmap | 0.08 (86%) | 0.08 (80%) | 0.12 (70%) | 0.28 (55%) | 0.64 (53%) | 2.18 (45%) | 7.72 (39%) |
| CN2 | NList | 0.10 (120%) | 0.11 (108%) | 0.16 (94%) | 0.38 (74%) | 0.86 (72%) | 3.05 (63%) | 10.05 (51%) |
| CN2 | Sparse | 0.11 (129%) | 0.12 (116%) | 0.17 (100%) | 0.42 (83%) | 0.92 (76%) | 3.25 (67%) | 11.57 (58%) |
| PFossil | HBitmap | 0.03 | 0.02 | 0.03 | 0.11 | 0.14 | 0.27 | 0.59 |
| PFossil | VBitmap | 0.03 (75%) | 0.01 (65%) | 0.02 (56%) | 0.06 (59%) | 0.06 (45%) | 0.11 (40%) | 0.24 (40%) |
| PFossil | NList | 0.03 (103%) | 0.02 (94%) | 0.03 (86%) | 0.09 (79%) | 0.09 (70%) | 0.17 (61%) | 0.35 (59%) |
| PFossil | Sparse | 0.03 (103%) | 0.02 (89%) | 0.02 (79%) | 0.08 (76%) | 0.09 (64%) | 0.15 (54%) | 0.32 (55%) |
| Pypper | HBitmap | 0.07 | 0.06 | 0.33 | 0.26 | 0.98 | 6.05 | 43.62 |
| Pypper | VBitmap | 0.02 (25%) | 0.03 (51%) | 0.09 (26%) | 0.07 (28%) | 0.18 (18%) | 0.58 (10%) | 2.13 (5%) |
| Pypper | NList | 0.02 (38%) | 0.05 (88%) | 0.19 (58%) | 0.14 (57%) | 0.40 (41%) | 1.43 (24%) | 6.15 (14%) |
| Pypper | Sparse | 0.02 (35%) | 0.05 (80%) | 0.16 (48%) | 0.15 (60%) | 0.45 (46%) | 2.51 (41%) | 18.70 (43%) |
| PyLORD | HBitmap | 0.72 | 1.97 | 6.16 | 20.59 | 69.10 | 260.49 | time-out |
| PyLORD | VBitmap | 0.55 (77%) | 1.34 (68%) | 3.67 (60%) | 10.27 (50%) | 28.04 (41%) | 87.39 (34%) | time-out |
| PyLORD | NList | 0.81 (113%) | 2.20 (112%) | 6.60 (107%) | 18.99 (92%) | 57.41 (83%) | 190.21 (73%) | time-out |
| PyLORD | Sparse | 0.84 (117%) | 2.39 (121%) | 6.38 (104%) | 20.08 (98%) | 60.24 (87%) | 210.63 (81%) | time-out |
| ENDER | HBitmap | 0.68 | 1.05 | 1.78 | 2.99 | 5.49 | 10.37 | 20.66 |
| ENDER | VBitmap | 0.61 (90%) | 0.93 (88%) | 1.53 (86%) | 2.61 (87%) | 4.97 (90%) | 9.21 (89%) | 17.91 (87%) |
| ENDER | NList | 0.75 (111%) | 1.06 (101%) | 1.63 (92%) | 2.71 (91%) | 4.72 (86%) | 9.21 (89%) | 17.02 (82%) |
| ENDER | Sparse | 0.89 (131%) | 1.19 (114%) | 1.72 (96%) | 2.77 (93%) | 5.03 (92%) | 8.76 (84%) | 17.04 (82%) |

**Growth.** How fast each learner's fit time grows with the training-set size: the exponent *b* of time ~ n^*b*, fitted by least squares to log time over log n from n=2,000 on (below that, a fit's fixed costs still dominate). 1 is linear, 2 quadratic; a curve cut short by a time-out is fitted on the points that finished. This is what decides, for example, how much a learner gains from splitting a problem into smaller ones (`demos/multiclass_decomposition.py`):

| learner | representation | with negation | without negation |
|---|---|--:|--:|
| CN2 | HBitmap | 1.82 | 1.70 |
| CN2 | VBitmap | 1.45 | 1.50 |
| CN2 | NList | 1.39 | 1.50 |
| CN2 | Sparse | 1.58 | 1.51 |
| PFossil | HBitmap | 1.03 | 0.98 |
| PFossil | VBitmap | 0.84 | 0.83 |
| PFossil | NList | 0.77 | 0.84 |
| PFossil | Sparse | 0.81 | 0.83 |
| Pypper | HBitmap | 1.49 | 1.87 |
| Pypper | VBitmap | 0.89 | 1.22 |
| Pypper | NList | 1.04 | 1.33 |
| Pypper | Sparse | 1.19 | 1.78 |
| PyLORD | HBitmap | 1.72 (3 points) | 1.80 (4 points) |
| PyLORD | VBitmap | 1.39 (4 points) | 1.52 (4 points) |
| PyLORD | NList | 1.51 (3 points) | 1.61 (4 points) |
| PyLORD | Sparse | 1.61 (3 points) | 1.67 (4 points) |
| ENDER | HBitmap | 0.68 | 0.89 |
| ENDER | VBitmap | 0.79 | 0.89 |
| ENDER | NList | 0.66 | 0.85 |
| ENDER | Sparse | 0.65 | 0.83 |

Building a vertical index is paid once per fit, on top of the fit itself:

![index build time vs. training-set size](representations_plots/size_index_build.png)

**Memory.** What each representation holds once built, measured with `tracemalloc` (the input matrix and the HBitmap matrix the others are built from don't count), and in parentheses its share of HBitmap's memory; the last row is the unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or external library working on a plain array pays on top.

![memory vs. training-set size](representations_plots/size_memory.png)

With negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 14 KiB | 31 KiB | 65 KiB | 133 KiB | 274 KiB | 547 KiB | 1.1 MiB |
| VBitmap | 15 KiB (105%) | 31 KiB (103%) | 66 KiB (102%) | 135 KiB (102%) | 271 KiB (99%) | 542 KiB (99%) | 1.1 MiB (100%) |
| NList | 828 KiB (5869%) | 1.5 MiB (4858%) | 3.2 MiB (5109%) | 6.0 MiB (4592%) | 12.0 MiB (4505%) | 21.5 MiB (4020%) | 37.7 MiB (3528%) |
| Sparse | 539 KiB (3823%) | 1.2 MiB (3858%) | 2.5 MiB (3887%) | 5.1 MiB (3953%) | 10.4 MiB (3897%) | 20.8 MiB (3897%) | 41.9 MiB (3925%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

Without negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 10 KiB | 21 KiB | 43 KiB | 90 KiB | 180 KiB | 360 KiB | 719 KiB |
| VBitmap | 10 KiB (107%) | 21 KiB (103%) | 44 KiB (102%) | 89 KiB (99%) | 178 KiB (99%) | 356 KiB (99%) | 716 KiB (100%) |
| NList | 230 KiB (2415%) | 411 KiB (1979%) | 742 KiB (1718%) | 1.3 MiB (1509%) | 2.9 MiB (1629%) | 5.0 MiB (1410%) | 8.2 MiB (1169%) |
| Sparse | 238 KiB (2503%) | 483 KiB (2330%) | 984 KiB (2278%) | 1.9 MiB (2183%) | 3.8 MiB (2184%) | 7.7 MiB (2185%) | 15.3 MiB (2185%) |
| unpacked `X` | 74 KiB | 160 KiB | 338 KiB | 699 KiB | 1.4 MiB | 2.8 MiB | 5.6 MiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass), with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.2 MiB | 2.5 MiB | 5.2 MiB | 10.8 MiB | 21.8 MiB | 43.5 MiB | 87.7 MiB |
| CN2 | VBitmap | 337 KiB (28%) | 419 KiB (17%) | 565 KiB (11%) | 891 KiB (8%) | 1.3 MiB (6%) | 2.4 MiB (6%) | 4.5 MiB (5%) |
| CN2 | NList | 1.9 MiB (165%) | 3.8 MiB (154%) | 6.2 MiB (119%) | 10.8 MiB (100%) | 21.8 MiB (100%) | 39.9 MiB (92%) | 72.1 MiB (82%) |
| CN2 | Sparse | 1.3 MiB (111%) | 2.7 MiB (111%) | 5.8 MiB (112%) | 12.1 MiB (112%) | 24.5 MiB (112%) | 48.9 MiB (112%) | 98.4 MiB (112%) |
| PFossil | HBitmap | 1.1 MiB | 2.4 MiB | 5.1 MiB | 10.7 MiB | 21.7 MiB | 43.4 MiB | 87.3 MiB |
| PFossil | VBitmap | 132 KiB (11%) | 210 KiB (8%) | 385 KiB (7%) | 636 KiB (6%) | 1.1 MiB (5%) | 2.1 MiB (5%) | 4.0 MiB (5%) |
| PFossil | NList | 1.9 MiB (170%) | 3.8 MiB (155%) | 6.2 MiB (120%) | 10.8 MiB (101%) | 21.8 MiB (101%) | 39.9 MiB (92%) | 72.1 MiB (83%) |
| PFossil | Sparse | 1.3 MiB (112%) | 2.7 MiB (112%) | 5.8 MiB (112%) | 12.0 MiB (112%) | 24.4 MiB (112%) | 48.7 MiB (112%) | 98.0 MiB (112%) |
| Pypper | HBitmap | 859 KiB | 1.7 MiB | 3.7 MiB | 7.6 MiB | 15.4 MiB | 30.7 MiB | 61.7 MiB |
| Pypper | VBitmap | 135 KiB (16%) | 209 KiB (12%) | 371 KiB (10%) | 628 KiB (8%) | 1.1 MiB (7%) | 2.1 MiB (7%) | 4.2 MiB (7%) |
| Pypper | NList | 1.7 MiB (201%) | 3.1 MiB (179%) | 6.1 MiB (167%) | 10.8 MiB (142%) | 21.8 MiB (142%) | 39.7 MiB (130%) | 71.9 MiB (116%) |
| Pypper | Sparse | 1.1 MiB (126%) | 2.2 MiB (128%) | 4.7 MiB (129%) | 9.8 MiB (129%) | 19.8 MiB (129%) | 39.6 MiB (129%) | 79.6 MiB (129%) |
| PyLORD | HBitmap | 744 KiB | 1.6 MiB | 4.3 MiB | 14.8 MiB |  |  |  |
| PyLORD | VBitmap | 353 KiB (47%) | 1.1 MiB (67%) | 3.7 MiB (88%) | 13.8 MiB (93%) | 54.0 MiB |  |  |
| PyLORD | NList | 2.0 MiB (275%) | 4.0 MiB (254%) | 7.4 MiB (174%) | 20.6 MiB (139%) | 67.7 MiB |  |  |
| PyLORD | Sparse | 1.4 MiB (188%) | 2.9 MiB (188%) | 6.2 MiB (147%) | 18.8 MiB (127%) |  |  |  |
| ENDER | HBitmap | 1014 KiB | 1.3 MiB | 2.0 MiB | 4.1 MiB | 8.2 MiB | 16.3 MiB | 32.7 MiB |
| ENDER | VBitmap | 881 KiB (87%) | 1.1 MiB (85%) | 1.5 MiB (75%) | 3.0 MiB (75%) | 6.1 MiB (74%) | 11.6 MiB (71%) | 14.1 MiB (43%) |
| ENDER | NList | 1.8 MiB (180%) | 2.8 MiB (219%) | 5.8 MiB (291%) | 11.1 MiB (273%) | 22.4 MiB (274%) | 41.1 MiB (252%) | 74.5 MiB (228%) |
| ENDER | Sparse | 1.4 MiB (142%) | 2.1 MiB (169%) | 3.6 MiB (181%) | 7.4 MiB (182%) | 15.0 MiB (183%) | 29.8 MiB (183%) | 59.9 MiB (183%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 869 KiB | 1.7 MiB | 3.5 MiB | 7.2 MiB | 14.4 MiB | 28.8 MiB | 58.0 MiB |
| CN2 | VBitmap | 241 KiB (28%) | 311 KiB (18%) | 407 KiB (11%) | 635 KiB (9%) | 989 KiB (7%) | 1.7 MiB (6%) | 3.3 MiB (6%) |
| CN2 | NList | 1.0 MiB (121%) | 2.0 MiB (121%) | 3.5 MiB (101%) | 4.6 MiB (65%) | 7.2 MiB (50%) | 12.4 MiB (43%) | 22.1 MiB (38%) |
| CN2 | Sparse | 626 KiB (72%) | 1.2 MiB (71%) | 2.4 MiB (68%) | 4.7 MiB (66%) | 9.3 MiB (65%) | 18.6 MiB (64%) | 37.2 MiB (64%) |
| PFossil | HBitmap | 811 KiB | 1.7 MiB | 3.5 MiB | 7.1 MiB | 14.4 MiB | 28.7 MiB | 57.6 MiB |
| PFossil | VBitmap | 109 KiB (13%) | 152 KiB (9%) | 270 KiB (8%) | 469 KiB (6%) | 812 KiB (6%) | 1.5 MiB (5%) | 2.9 MiB (5%) |
| PFossil | NList | 1.0 MiB (128%) | 2.0 MiB (121%) | 3.5 MiB (102%) | 4.6 MiB (64%) | 7.1 MiB (50%) | 12.4 MiB (43%) | 22.1 MiB (38%) |
| PFossil | Sparse | 606 KiB (75%) | 1.2 MiB (70%) | 2.3 MiB (68%) | 4.7 MiB (65%) | 9.2 MiB (64%) | 18.4 MiB (64%) | 36.8 MiB (64%) |
| Pypper | HBitmap | 591 KiB | 1.2 MiB | 2.5 MiB | 5.1 MiB | 10.2 MiB | 20.3 MiB | 40.8 MiB |
| Pypper | VBitmap | 106 KiB (18%) | 154 KiB (13%) | 275 KiB (11%) | 511 KiB (10%) | 992 KiB (10%) | 1.9 MiB (9%) | 3.8 MiB (9%) |
| Pypper | NList | 847 KiB (143%) | 1.6 MiB (131%) | 3.1 MiB (125%) | 4.5 MiB (89%) | 7.0 MiB (68%) | 12.3 MiB (60%) | 21.9 MiB (54%) |
| Pypper | Sparse | 500 KiB (85%) | 979 KiB (80%) | 1.9 MiB (78%) | 3.8 MiB (75%) | 7.5 MiB (74%) | 15.0 MiB (74%) | 30.0 MiB (73%) |
| PyLORD | HBitmap | 424 KiB | 1.2 MiB | 4.1 MiB | 14.2 MiB | 53.5 MiB |  |  |
| PyLORD | VBitmap | 342 KiB (81%) | 1.1 MiB (87%) | 3.7 MiB (92%) | 13.5 MiB (95%) | 52.1 MiB (97%) |  |  |
| PyLORD | NList | 1.1 MiB (259%) | 2.2 MiB (178%) | 4.8 MiB (117%) | 15.4 MiB (108%) | 56.1 MiB (105%) |  |  |
| PyLORD | Sparse | 698 KiB (165%) | 1.6 MiB (128%) | 4.7 MiB (116%) | 15.4 MiB (108%) | 55.8 MiB (104%) |  |  |
| ENDER | HBitmap | 845 KiB | 1005 KiB | 1.4 MiB | 2.9 MiB | 5.7 MiB | 11.4 MiB | 22.8 MiB |
| ENDER | VBitmap | 755 KiB (89%) | 819 KiB (82%) | 1.1 MiB (77%) | 2.2 MiB (76%) | 4.3 MiB (76%) | 8.6 MiB (76%) | 13.8 MiB (60%) |
| ENDER | NList | 1.0 MiB (127%) | 1.4 MiB (138%) | 2.1 MiB (148%) | 4.0 MiB (141%) | 7.5 MiB (132%) | 13.6 MiB (119%) | 24.5 MiB (107%) |
| ENDER | Sparse | 1007 KiB (119%) | 1.3 MiB (130%) | 1.9 MiB (132%) | 3.1 MiB (110%) | 6.2 MiB (109%) | 12.4 MiB (109%) | 24.6 MiB (108%) |

Peak memory while building (with negation) -- the vertical indexes need more than they keep:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 19 KiB | 36 KiB | 70 KiB | 138 KiB | 279 KiB | 552 KiB | 1.1 MiB |
| VBitmap | 143 KiB (742%) | 306 KiB (856%) | 645 KiB (924%) | 1.3 MiB (970%) | 2.6 MiB (972%) | 5.2 MiB (972%) | 9.5 MiB (889%) |
| NList | 4.4 MiB (23384%) | 8.5 MiB (24321%) | 17.7 MiB (26016%) | 34.5 MiB (25571%) | 69.8 MiB (25629%) | 130.9 MiB (24267%) | 255.7 MiB (23823%) |
| Sparse | 698 KiB (3625%) | 1.2 MiB (3304%) | 2.5 MiB (3601%) | 5.1 MiB (3806%) | 10.4 MiB (3825%) | 20.8 MiB (3860%) | 41.9 MiB (3907%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

## 2. Density

Synthetic data, 1000 training and 1000 test examples with 500 features: 4 of them carry a fixed learnable concept (true with probability 0.4, class = f0 and not f1, 5% label noise); every other feature is noise, true with the probability on the x-axis. Size and feature count stay fixed, so this isolates density.

![CN2 fit time vs. density](representations_plots/density_cn2.png)

![PFossil fit time vs. density](representations_plots/density_pfossil.png)

![Pypper fit time vs. density](representations_plots/density_pypper.png)

![PyLORD fit time vs. density](representations_plots/density_pylord.png)

![ENDER fit time vs. density](representations_plots/density_ender.png)

The pure data operation, 800 `coverage()` calls for random rules of 1-4 conditions, without any search around it:

![coverage() time vs. density](representations_plots/density_coverage.png)

Fit time in seconds:

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.40 | 0.53 | 0.66 | 0.59 | 0.19 | 0.12 | 0.08 | 0.06 |
| CN2 | VBitmap | 0.22 (55%) | 0.36 (67%) | 0.45 (69%) | 0.37 (62%) | 0.13 (71%) | 0.06 (55%) | 0.04 (51%) | 0.03 (43%) |
| CN2 | NList | 0.29 (71%) | 0.45 (85%) | 0.61 (92%) | 0.45 (77%) | 0.17 (88%) | 0.09 (74%) | 0.06 (69%) | 0.03 (56%) |
| CN2 | Sparse | 0.32 (79%) | 0.45 (84%) | 0.56 (85%) | 0.44 (74%) | 0.15 (81%) | 0.08 (66%) | 0.05 (62%) | 0.03 (51%) |
| PFossil | HBitmap | 0.04 | 0.04 | 0.02 | 0.02 | 0.02 | 0.01 | 0.01 | 0.01 |
| PFossil | VBitmap | 0.02 (56%) | 0.02 (54%) | 0.01 (46%) | 0.01 (46%) | 0.01 (44%) | 0.01 (43%) | 0.01 (43%) | 0.01 (42%) |
| PFossil | NList | 0.04 (95%) | 0.04 (88%) | 0.01 (89%) | 0.01 (79%) | 0.01 (77%) | 0.01 (70%) | 0.01 (66%) | 0.01 (60%) |
| PFossil | Sparse | 0.03 (81%) | 0.03 (74%) | 0.01 (55%) | 0.01 (57%) | 0.01 (51%) | 0.01 (50%) | 0.01 (50%) | 0.01 (47%) |
| Pypper | HBitmap | 0.16 | 0.02 | 0.04 | 0.02 | 0.08 | 0.03 | 0.06 | 0.03 |
| Pypper | VBitmap | 0.04 (26%) | 0.01 (55%) | 0.02 (45%) | 0.01 (56%) | 0.03 (36%) | 0.01 (48%) | 0.03 (40%) | 0.01 (52%) |
| Pypper | NList | 0.09 (57%) | 0.02 (99%) | 0.03 (77%) | 0.02 (85%) | 0.05 (57%) | 0.02 (71%) | 0.04 (62%) | 0.02 (74%) |
| Pypper | Sparse | 0.07 (43%) | 0.02 (72%) | 0.02 (55%) | 0.01 (62%) | 0.04 (44%) | 0.02 (57%) | 0.03 (47%) | 0.02 (63%) |
| PyLORD | HBitmap | 4.11 | 2.54 | 1.54 | 1.42 | 1.42 | 1.31 | 1.20 | 1.02 |
| PyLORD | VBitmap | 2.13 (52%) | 1.42 (56%) | 1.00 (65%) | 1.00 (71%) | 1.01 (71%) | 0.95 (73%) | 0.89 (74%) | 0.79 (77%) |
| PyLORD | NList | 3.64 (89%) | 2.61 (103%) | 1.96 (127%) | 1.99 (140%) | 1.98 (140%) | 1.46 (111%) | 1.36 (114%) | 1.16 (113%) |
| PyLORD | Sparse | 3.64 (89%) | 2.40 (95%) | 1.49 (97%) | 1.45 (102%) | 1.43 (101%) | 1.31 (100%) | 1.23 (102%) | 1.06 (104%) |
| ENDER | HBitmap | 1.10 | 0.91 | 0.79 | 0.75 | 0.75 | 0.79 | 0.73 | 0.74 |
| ENDER | VBitmap | 0.87 (80%) | 0.72 (79%) | 0.66 (83%) | 0.63 (84%) | 0.58 (78%) | 0.56 (71%) | 0.57 (77%) | 0.58 (77%) |
| ENDER | NList | 0.98 (89%) | 0.73 (81%) | 0.65 (82%) | 0.60 (81%) | 0.56 (75%) | 0.60 (76%) | 0.57 (78%) | 0.57 (76%) |
| ENDER | Sparse | 0.93 (85%) | 0.76 (84%) | 0.63 (79%) | 0.59 (79%) | 0.54 (72%) | 0.57 (72%) | 0.55 (75%) | 0.56 (75%) |

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

![memory vs. density](representations_plots/density_memory.png)

| representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB |
| VBitmap | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) | 63 KiB (102%) |
| NList | 16.9 MiB (27943%) | 10.1 MiB (16816%) | 3.4 MiB (5637%) | 1.7 MiB (2809%) | 709 KiB (1147%) | 376 KiB (608%) | 209 KiB (338%) | 111 KiB (180%) |
| Sparse | 2.4 MiB (3962%) | 1.4 MiB (2390%) | 509 KiB (824%) | 264 KiB (428%) | 119 KiB (193%) | 72 KiB (116%) | 48 KiB (78%) | 34 KiB (54%) |
| unpacked `X` | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 5.0 MiB | 4.9 MiB | 5.0 MiB | 5.0 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB |
| CN2 | VBitmap | 1.1 MiB (23%) | 1.1 MiB (23%) | 1.1 MiB (22%) | 1.0 MiB (20%) | 795 KiB (16%) | 678 KiB (13%) | 587 KiB (12%) | 469 KiB (9%) |
| CN2 | NList | 20.3 MiB (409%) | 13.1 MiB (264%) | 6.3 MiB (128%) | 4.6 MiB (93%) | 3.6 MiB (73%) | 1.1 MiB (22%) | 742 KiB (15%) | 484 KiB (10%) |
| CN2 | Sparse | 5.6 MiB (114%) | 3.5 MiB (70%) | 1.6 MiB (31%) | 1.2 MiB (25%) | 857 KiB (17%) | 693 KiB (14%) | 523 KiB (10%) | 372 KiB (7%) |
| PFossil | HBitmap | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB |
| PFossil | VBitmap | 406 KiB (8%) | 404 KiB (8%) | 403 KiB (8%) | 402 KiB (8%) | 402 KiB (8%) | 400 KiB (8%) | 395 KiB (8%) | 357 KiB (7%) |
| PFossil | NList | 19.7 MiB (400%) | 13.0 MiB (264%) | 6.3 MiB (127%) | 4.6 MiB (93%) | 3.6 MiB (73%) | 701 KiB (14%) | 482 KiB (10%) | 369 KiB (7%) |
| PFossil | Sparse | 5.6 MiB (114%) | 3.4 MiB (69%) | 1.2 MiB (25%) | 713 KiB (14%) | 380 KiB (8%) | 271 KiB (5%) | 244 KiB (5%) | 207 KiB (4%) |
| Pypper | HBitmap | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB |
| Pypper | VBitmap | 416 KiB (11%) | 409 KiB (11%) | 411 KiB (11%) | 408 KiB (11%) | 408 KiB (11%) | 379 KiB (11%) | 353 KiB (10%) | 351 KiB (10%) |
| Pypper | NList | 19.7 MiB (556%) | 13.0 MiB (372%) | 6.3 MiB (179%) | 4.6 MiB (131%) | 3.6 MiB (102%) | 663 KiB (19%) | 476 KiB (13%) | 363 KiB (10%) |
| Pypper | Sparse | 4.6 MiB (129%) | 2.8 MiB (80%) | 1.0 MiB (30%) | 597 KiB (17%) | 334 KiB (9%) | 256 KiB (7%) | 214 KiB (6%) | 209 KiB (6%) |
| PyLORD | HBitmap | 3.3 MiB | 2.3 MiB | 1.6 MiB | 1.9 MiB | 2.0 MiB | 2.0 MiB | 2.0 MiB | 1.8 MiB |
| PyLORD | VBitmap | 1.1 MiB (34%) | 1.1 MiB (48%) | 1.1 MiB (70%) | 1.4 MiB (75%) | 1.6 MiB (77%) | 1.6 MiB (77%) | 1.5 MiB (76%) | 1.2 MiB (65%) |
| PyLORD | NList | 20.0 MiB (613%) | 13.2 MiB (576%) | 6.4 MiB (401%) | 4.8 MiB (255%) | 3.8 MiB (186%) | 2.0 MiB (99%) | 1.8 MiB (89%) | 1.3 MiB (72%) |
| PyLORD | Sparse | 5.9 MiB (180%) | 3.6 MiB (158%) | 1.6 MiB (101%) | 1.7 MiB (88%) | 1.7 MiB (82%) | 1.6 MiB (80%) | 1.5 MiB (78%) | 1.2 MiB (66%) |
| ENDER | HBitmap | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB |
| ENDER | VBitmap | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) | 1.4 MiB (75%) |
| ENDER | NList | 19.4 MiB (1069%) | 11.9 MiB (653%) | 5.2 MiB (284%) | 3.5 MiB (190%) | 2.5 MiB (136%) | 815 KiB (44%) | 602 KiB (32%) | 475 KiB (25%) |
| ENDER | Sparse | 3.4 MiB (189%) | 2.2 MiB (118%) | 1.0 MiB (55%) | 711 KiB (38%) | 495 KiB (26%) | 419 KiB (22%) | 384 KiB (21%) | 363 KiB (19%) |

## 3. Width

Synthetic data again, 1000 training and 1000 test examples, the same concept on 4 signal features -- but now the number of *true* features per example stays fixed (on average 16 noise features, plus the signal ones) while the number of features grows. That is how sparse data usually looks: one-hot encodings of attributes with many values, or words in a text. The bitmaps grow with the number of features (one bit per cell); a sparse representation with the number of true cells, which stays the same. `Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes against 1/8 byte per cell for the bitmaps, so it should take less memory below a density of about 1/80 -- here from about 1,408 features on.

![memory vs. width](representations_plots/width_memory.png)

Memory held by each representation (share of HBitmap in parentheses; the last row is the unpacked bool matrix):

| representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| HBitmap | 13 KiB | 25 KiB | 62 KiB | 122 KiB | 244 KiB | 611 KiB | 1.2 MiB | 2.4 MiB |
| VBitmap | 13 KiB (103%) | 26 KiB (105%) | 63 KiB (102%) | 126 KiB (103%) | 251 KiB (103%) | 626 KiB (102%) | 1.2 MiB (102%) | 2.4 MiB (102%) |
| NList | 387 KiB (2996%) | 637 KiB (2586%) | 1.1 MiB (1846%) | 2.1 MiB (1754%) | 4.1 MiB (1703%) | 9.9 MiB (1652%) | 19.5 MiB (1634%) | 38.7 MiB (1624%) |
| Sparse | 179 KiB (1383%) | 179 KiB (727%) | 181 KiB (293%) | 182 KiB (149%) | 187 KiB (77%) | 199 KiB (33%) | 218 KiB (18%) | 257 KiB (11%) |
| unpacked `X` | 98 KiB | 195 KiB | 488 KiB | 977 KiB | 1.9 MiB | 4.8 MiB | 9.5 MiB | 19.1 MiB |

800 `coverage()` calls for random rules of 1-4 conditions:

![coverage() time vs. width](representations_plots/width_coverage.png)

![fit time vs. width](representations_plots/width_fit.png)

Fit time in seconds:

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 0.32 | 0.25 | 0.64 | 0.43 | 0.81 | 3.20 | 8.92 | 18.17 |
| CN2 | VBitmap | 0.25 (79%) | 0.18 (72%) | 0.43 (67%) | 0.24 (57%) | 0.37 (45%) | 0.93 (29%) | 1.41 (16%) | 2.34 (13%) |
| CN2 | NList | 0.35 (112%) | 0.25 (100%) | 0.57 (90%) | 0.27 (63%) | 0.39 (48%) | 0.90 (28%) | 1.22 (14%) | 1.91 (11%) |
| CN2 | Sparse | 0.39 (125%) | 0.23 (92%) | 0.47 (74%) | 0.26 (61%) | 0.35 (42%) | 0.99 (31%) | 1.17 (13%) | 1.86 (10%) |
| PFossil | HBitmap | 0.01 | 0.02 | 0.02 | 0.02 | 0.05 | 0.11 | 0.19 | 0.35 |
| PFossil | VBitmap | 0.01 (70%) | 0.01 (57%) | 0.01 (42%) | 0.01 (34%) | 0.01 (23%) | 0.02 (14%) | 0.02 (11%) | 0.03 (10%) |
| PFossil | NList | 0.01 (102%) | 0.01 (91%) | 0.01 (77%) | 0.01 (57%) | 0.02 (42%) | 0.03 (31%) | 0.05 (28%) | 0.10 (27%) |
| PFossil | Sparse | 0.01 (91%) | 0.01 (76%) | 0.01 (49%) | 0.01 (35%) | 0.01 (24%) | 0.01 (11%) | 0.02 (8%) | 0.02 (7%) |
| Pypper | HBitmap | 0.07 | 0.02 | 0.03 | 0.05 | 0.05 | 0.68 | 0.78 | 1.18 |
| Pypper | VBitmap | 0.03 (42%) | 0.01 (63%) | 0.01 (48%) | 0.02 (35%) | 0.02 (35%) | 0.13 (19%) | 0.11 (15%) | 0.16 (13%) |
| Pypper | NList | 0.05 (69%) | 0.02 (81%) | 0.02 (76%) | 0.03 (53%) | 0.03 (55%) | 0.16 (24%) | 0.14 (18%) | 0.19 (16%) |
| Pypper | Sparse | 0.04 (56%) | 0.01 (70%) | 0.02 (56%) | 0.02 (39%) | 0.02 (33%) | 0.11 (16%) | 0.09 (11%) | 0.12 (10%) |
| PyLORD | HBitmap | 1.56 | 1.36 | 1.47 | 1.36 | 1.23 | 1.37 | 1.58 | 2.59 |
| PyLORD | VBitmap | 1.15 (74%) | 0.95 (70%) | 1.00 (68%) | 0.95 (69%) | 0.80 (66%) | 0.77 (56%) | 0.85 (54%) | 0.98 (38%) |
| PyLORD | NList | 1.72 (110%) | 1.58 (116%) | 1.97 (134%) | 1.54 (113%) | 1.20 (98%) | 1.15 (84%) | 1.16 (74%) | 1.26 (49%) |
| PyLORD | Sparse | 1.71 (110%) | 1.45 (106%) | 1.49 (101%) | 1.36 (100%) | 1.10 (90%) | 1.00 (73%) | 1.02 (65%) | 1.06 (41%) |
| ENDER | HBitmap | 0.48 | 0.57 | 0.77 | 1.22 | 1.81 | 3.64 | 7.29 | 15.75 |
| ENDER | VBitmap | 0.46 (96%) | 0.51 (90%) | 0.61 (79%) | 0.79 (65%) | 1.15 (64%) | 2.50 (69%) | 4.37 (60%) | 8.63 (55%) |
| ENDER | NList | 0.49 (102%) | 0.51 (90%) | 0.58 (76%) | 0.68 (56%) | 0.70 (39%) | 0.86 (24%) | 1.11 (15%) | 1.69 (11%) |
| ENDER | Sparse | 0.55 (115%) | 0.56 (99%) | 0.58 (76%) | 0.61 (50%) | 0.66 (37%) | 0.81 (22%) | 1.05 (14%) | 1.66 (11%) |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | HBitmap | 1.1 MiB | 2.0 MiB | 5.0 MiB | 9.8 MiB | 19.7 MiB | 49.2 MiB | 97.7 MiB | 196.2 MiB |
| CN2 | VBitmap | 293 KiB (27%) | 436 KiB (21%) | 922 KiB (18%) | 1.4 MiB (14%) | 3.2 MiB (16%) | 7.4 MiB (15%) | 12.1 MiB (12%) | 23.9 MiB (12%) |
| CN2 | NList | 1.5 MiB (140%) | 2.7 MiB (133%) | 4.1 MiB (82%) | 3.6 MiB (36%) | 7.2 MiB (37%) | 16.8 MiB (34%) | 30.6 MiB (31%) | 60.5 MiB (31%) |
| CN2 | Sparse | 520 KiB (48%) | 610 KiB (29%) | 1.0 MiB (21%) | 1.4 MiB (15%) | 3.1 MiB (16%) | 6.9 MiB (14%) | 11.1 MiB (11%) | 21.7 MiB (11%) |
| PFossil | HBitmap | 1.0 MiB | 2.0 MiB | 4.9 MiB | 9.8 MiB | 19.6 MiB | 49.1 MiB | 97.6 MiB | 196.2 MiB |
| PFossil | VBitmap | 129 KiB (12%) | 174 KiB (9%) | 402 KiB (8%) | 688 KiB (7%) | 1.3 MiB (7%) | 3.5 MiB (7%) | 5.7 MiB (6%) | 11.1 MiB (6%) |
| PFossil | NList | 1.4 MiB (141%) | 2.6 MiB (133%) | 4.0 MiB (81%) | 2.7 MiB (28%) | 5.0 MiB (26%) | 11.9 MiB (24%) | 22.7 MiB (23%) | 45.3 MiB (23%) |
| PFossil | Sparse | 475 KiB (45%) | 479 KiB (23%) | 521 KiB (10%) | 554 KiB (6%) | 923 KiB (5%) | 2.0 MiB (4%) | 3.2 MiB (3%) | 6.6 MiB (3%) |
| Pypper | HBitmap | 758 KiB | 1.4 MiB | 3.5 MiB | 6.9 MiB | 13.8 MiB | 34.8 MiB | 69.0 MiB | 138.9 MiB |
| Pypper | VBitmap | 134 KiB (18%) | 179 KiB (12%) | 409 KiB (11%) | 685 KiB (10%) | 1.2 MiB (9%) | 2.8 MiB (8%) | 5.1 MiB (7%) | 11.1 MiB (8%) |
| Pypper | NList | 1.1 MiB (155%) | 2.1 MiB (145%) | 4.0 MiB (114%) | 2.6 MiB (38%) | 4.9 MiB (35%) | 11.7 MiB (34%) | 22.4 MiB (32%) | 45.7 MiB (33%) |
| Pypper | Sparse | 398 KiB (52%) | 398 KiB (27%) | 443 KiB (12%) | 487 KiB (7%) | 804 KiB (6%) | 1.8 MiB (5%) | 2.9 MiB (4%) | 7.0 MiB (5%) |
| PyLORD | HBitmap | 1.4 MiB | 1.4 MiB | 2.0 MiB | 2.4 MiB | 4.1 MiB | 10.1 MiB | 20.3 MiB | 40.5 MiB |
| PyLORD | VBitmap | 1.4 MiB (93%) | 1.2 MiB (86%) | 1.5 MiB (76%) | 1.5 MiB (61%) | 1.5 MiB (38%) | 2.0 MiB (20%) | 2.9 MiB (14%) | 4.6 MiB (11%) |
| PyLORD | NList | 1.8 MiB (127%) | 2.8 MiB (202%) | 4.2 MiB (213%) | 3.7 MiB (154%) | 5.6 MiB (138%) | 11.5 MiB (114%) | 21.4 MiB (106%) | 41.1 MiB (101%) |
| PyLORD | Sparse | 1.6 MiB (107%) | 1.4 MiB (101%) | 1.7 MiB (85%) | 1.6 MiB (66%) | 1.5 MiB (37%) | 1.6 MiB (16%) | 2.0 MiB (10%) | 2.4 MiB (6%) |
| ENDER | HBitmap | 509 KiB | 842 KiB | 1.8 MiB | 3.5 MiB | 6.8 MiB | 17.1 MiB | 33.5 MiB | 67.7 MiB |
| ENDER | VBitmap | 414 KiB (81%) | 654 KiB (78%) | 1.4 MiB (75%) | 2.5 MiB (73%) | 5.0 MiB (73%) | 10.7 MiB (63%) | 12.1 MiB (36%) | 25.0 MiB (37%) |
| ENDER | NList | 902 KiB (177%) | 1.4 MiB (175%) | 2.9 MiB (159%) | 2.7 MiB (79%) | 4.9 MiB (71%) | 11.6 MiB (68%) | 22.0 MiB (66%) | 44.4 MiB (66%) |
| ENDER | Sparse | 547 KiB (107%) | 564 KiB (67%) | 593 KiB (32%) | 604 KiB (17%) | 786 KiB (11%) | 1.7 MiB (10%) | 2.5 MiB (8%) | 5.7 MiB (8%) |

