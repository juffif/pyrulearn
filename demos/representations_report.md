# Data representations: same rules, different speed

**Full run.** Quick sanity check instead: `python demos/representations.py`.

pyrulearn's learners never look at the data directly. They ask it two things -- which examples a rule covers (`coverage(rule)`) and which features an example has (`features_of(row)`) -- so the data can be stored in any structure that answers these questions. Four such representations are implemented:

- **Boolean** -- a packed bit matrix, one bit per example and feature. Covering a rule is a bitwise AND over the rows, whatever the density.
- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are inserted into a prefix tree of their true features, and each feature keeps the list of tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work grows with how often the features are true, not with the number of examples.
- **PrePostNList** -- the N-list plus pre/post-order codes of the tree nodes, a second way to find which nodes are compatible with the rule so far.
- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix tree.

All four must lead to exactly the same rules, so the only thing that differs is time. This demo measures that time for four learners -- CN2, PFossil, Pypper and PyLORD, each with its default settings -- and checks the rules and the test-set predictions against the Boolean version at every point. Each fit runs in a separate process and is capped at 300s; after a time-out, the larger points of that curve are skipped.

**Equality check:** at 117 points where Boolean and at least one other representation finished, 117 gave identical rules and predictions for every representation.

## 1. Training-set size

Training sets of doubling size, drawn from one shuffled pool of the `adult` dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the same 20% test set. Numeric attributes are discretized per training set (`max_intervals=8`). Two encodings: **with negation** (pyrulearn's default -- every test comes with its negation, so exactly half the features are true in every example) and **without negation** (positive tests only, far sparser). The first is the Boolean matrix's best case and the vertical representations' worst; the second is where they should gain.

At the largest training set: with negation 278 binary features, density 0.493; without negation 140 binary features, density 0.136 (density = share of features that are true).

One pair of plots per learner: left with a logarithmic time axis (ratios -- a constant factor between two representations is a constant gap), right with a linear one (absolute seconds saved).

<img src="representations_plots/size_cn2.png" alt="CN2 fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_cn2_linear.png" alt="CN2 fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pfossil.png" alt="PFossil fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pfossil_linear.png" alt="PFossil fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pypper.png" alt="Pypper fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pypper_linear.png" alt="Pypper fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pylord.png" alt="PyLORD fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pylord_linear.png" alt="PyLORD fit time vs. training-set size, linear scale" width="49%">

Fit time in seconds, with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.04 | 0.08 | 0.25 | 0.65 | 1.82 | 6.11 | 58.76 |
| CN2 | NList | 0.06 (135%) | 0.06 (76%) | 0.21 (85%) | 0.50 (77%) | 1.16 (64%) | 2.66 (44%) | 31.25 (53%) |
| CN2 | PrePostNList | 0.06 (143%) | 0.06 (74%) | 0.19 (77%) | 0.48 (74%) | 1.18 (65%) | 2.62 (43%) | 22.18 (38%) |
| CN2 | Sparse | 0.05 (117%) | 0.06 (80%) | 0.17 (66%) | 0.47 (73%) | 1.33 (73%) | 3.23 (53%) | 29.91 (51%) |
| PFossil | Boolean | 0.01 | 0.04 | 0.06 | 0.16 | 0.33 | 0.51 | 1.09 |
| PFossil | NList | 0.02 (123%) | 0.03 (67%) | 0.04 (71%) | 0.11 (67%) | 0.17 (50%) | 0.27 (53%) | 0.43 (39%) |
| PFossil | PrePostNList | 0.02 (119%) | 0.03 (87%) | 0.03 (54%) | 0.09 (59%) | 0.17 (51%) | 0.22 (43%) | 0.41 (38%) |
| PFossil | Sparse | 0.01 (103%) | 0.03 (87%) | 0.05 (76%) | 0.09 (55%) | 0.16 (48%) | 0.24 (47%) | 0.48 (44%) |
| Pypper | Boolean | 0.10 | 0.27 | 0.20 | 0.79 | 2.36 | 2.76 | 32.17 |
| Pypper | NList | 0.05 (50%) | 0.15 (57%) | 0.13 (63%) | 0.41 (52%) | 0.79 (34%) | 0.73 (27%) | 4.38 (14%) |
| Pypper | PrePostNList | 0.05 (53%) | 0.19 (72%) | 0.15 (76%) | 0.38 (48%) | 0.81 (35%) | 0.72 (26%) | 4.37 (14%) |
| Pypper | Sparse | 0.06 (65%) | 0.16 (58%) | 0.11 (53%) | 0.49 (62%) | 1.19 (51%) | 1.10 (40%) | 10.90 (34%) |
| PyLORD | Boolean | 1.00 | 2.69 | 11.98 | 47.66 | 190.64 | time-out |  |
| PyLORD | NList | 1.26 (126%) | 3.01 (112%) | 11.52 (96%) | 38.78 (81%) | 101.21 (53%) | time-out |  |
| PyLORD | PrePostNList | 1.12 (112%) | 3.05 (113%) | 11.28 (94%) | 40.59 (85%) | 91.31 (48%) | time-out |  |
| PyLORD | Sparse | 1.02 (102%) | 2.84 (106%) | 11.91 (99%) | 47.43 (100%) | 119.04 (62%) | time-out |  |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.04 | 0.13 | 0.30 | 1.02 | 2.28 | 5.95 | 29.71 |
| CN2 | NList | 0.07 (163%) | 0.17 (128%) | 0.32 (108%) | 0.92 (90%) | 1.41 (62%) | 2.77 (47%) | 10.99 (37%) |
| CN2 | PrePostNList | 0.08 (183%) | 0.17 (128%) | 0.33 (109%) | 1.02 (100%) | 1.36 (60%) | 2.71 (46%) | 10.82 (36%) |
| CN2 | Sparse | 0.05 (131%) | 0.16 (120%) | 0.32 (106%) | 2.30 (226%) | 1.61 (71%) | 3.38 (57%) | 14.18 (48%) |
| PFossil | Boolean | 0.01 | 0.01 | 0.02 | 0.07 | 0.09 | 0.18 | 0.33 |
| PFossil | NList | 0.02 (127%) | 0.01 (91%) | 0.01 (62%) | 0.03 (38%) | 0.04 (46%) | 0.06 (33%) | 0.12 (37%) |
| PFossil | PrePostNList | 0.01 (119%) | 0.01 (92%) | 0.01 (61%) | 0.02 (28%) | 0.04 (42%) | 0.07 (36%) | 0.13 (40%) |
| PFossil | Sparse | 0.01 (110%) | 0.01 (84%) | 0.01 (52%) | 0.02 (24%) | 0.03 (35%) | 0.06 (33%) | 0.12 (35%) |
| Pypper | Boolean | 0.01 | 0.03 | 0.21 | 0.80 | 1.75 | 8.29 | 21.33 |
| Pypper | NList | 0.01 (121%) | 0.03 (93%) | 0.12 (57%) | 0.27 (34%) | 0.45 (26%) | 1.22 (15%) | 2.36 (11%) |
| Pypper | PrePostNList | 0.01 (137%) | 0.03 (99%) | 0.10 (47%) | 0.31 (39%) | 0.45 (26%) | 1.31 (16%) | 2.50 (12%) |
| Pypper | Sparse | 0.01 (97%) | 0.02 (89%) | 0.10 (46%) | 0.38 (48%) | 0.73 (42%) | 3.27 (39%) | 7.63 (36%) |
| PyLORD | Boolean | 0.43 | 1.14 | 4.34 | 15.45 | 55.65 | 158.02 | time-out |
| PyLORD | NList | 0.59 (139%) | 1.38 (120%) | 9.73 (224%) | 15.90 (103%) | 36.09 (65%) | 103.05 (65%) | time-out |
| PyLORD | PrePostNList | 0.54 (126%) | 1.51 (132%) | 4.75 (109%) | 15.95 (103%) | 34.04 (61%) | 101.62 (64%) | time-out |
| PyLORD | Sparse | 0.47 (110%) | 1.41 (124%) | 4.56 (105%) | 18.08 (117%) | 38.28 (69%) | 194.40 (123%) | time-out |

Building a vertical index is paid once per fit, on top of the fit itself:

![index build time vs. training-set size](representations_plots/size_index_build.png)

**Memory.** What each representation holds once built, measured with `tracemalloc` (the input matrix and the Boolean matrix a vertical index is built from don't count), and in parentheses its share of the Boolean representation's memory; the last row is the unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or external library working on a plain array pays on top.

![memory vs. training-set size](representations_plots/size_memory.png)

With negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 14 KiB | 31 KiB | 65 KiB | 133 KiB | 274 KiB | 547 KiB | 1.1 MiB |
| NList | 828 KiB (5876%) | 1.5 MiB (4859%) | 3.2 MiB (5110%) | 6.0 MiB (4592%) | 12.0 MiB (4505%) | 21.5 MiB (4020%) | 37.7 MiB (3528%) |
| PrePostNList | 989 KiB (7022%) | 1.7 MiB (5807%) | 3.8 MiB (5944%) | 6.9 MiB (5341%) | 14.0 MiB (5240%) | 25.0 MiB (4675%) | 43.8 MiB (4101%) |
| Sparse | 539 KiB (3827%) | 1.2 MiB (3858%) | 2.5 MiB (3887%) | 5.1 MiB (3953%) | 10.4 MiB (3897%) | 20.8 MiB (3897%) | 41.9 MiB (3925%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

Without negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 7 KiB | 16 KiB | 33 KiB | 71 KiB | 141 KiB | 281 KiB | 563 KiB |
| NList | 91 KiB (1294%) | 163 KiB (1027%) | 361 KiB (1080%) | 688 KiB (975%) | 1.4 MiB (985%) | 2.4 MiB (858%) | 4.2 MiB (763%) |
| PrePostNList | 115 KiB (1635%) | 205 KiB (1296%) | 435 KiB (1300%) | 828 KiB (1173%) | 1.6 MiB (1186%) | 2.8 MiB (1027%) | 5.0 MiB (909%) |
| Sparse | 99 KiB (1407%) | 207 KiB (1307%) | 344 KiB (1030%) | 795 KiB (1127%) | 1.7 MiB (1234%) | 3.3 MiB (1202%) | 5.9 MiB (1080%) |
| unpacked `X` | 55 KiB | 120 KiB | 256 KiB | 535 KiB | 1.1 MiB | 2.1 MiB | 4.3 MiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass), with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 1.2 MiB | 2.4 MiB | 5.2 MiB | 10.8 MiB | 21.8 MiB | 43.5 MiB | 87.7 MiB |
| CN2 | NList | 2.2 MiB (186%) | 4.2 MiB (172%) | 8.9 MiB (172%) | 17.4 MiB (161%) | 35.1 MiB (161%) | 64.7 MiB (149%) | 118.6 MiB (135%) |
| CN2 | PrePostNList | 2.3 MiB (196%) | 4.5 MiB (184%) | 9.4 MiB (182%) | 18.3 MiB (170%) | 37.1 MiB (170%) | 68.2 MiB (157%) | 124.7 MiB (142%) |
| CN2 | Sparse | 1.3 MiB (111%) | 2.7 MiB (112%) | 5.8 MiB (112%) | 12.1 MiB (112%) | 24.5 MiB (112%) | 48.9 MiB (112%) | 98.4 MiB (112%) |
| PFossil | Boolean | 1.1 MiB | 2.4 MiB | 5.1 MiB | 10.7 MiB | 21.7 MiB | 43.4 MiB | 87.3 MiB |
| PFossil | NList | 2.1 MiB (187%) | 4.2 MiB (172%) | 8.9 MiB (172%) | 17.3 MiB (162%) | 35.0 MiB (161%) | 64.6 MiB (149%) | 118.2 MiB (135%) |
| PFossil | PrePostNList | 2.3 MiB (201%) | 4.5 MiB (184%) | 9.4 MiB (183%) | 18.3 MiB (171%) | 37.0 MiB (170%) | 68.1 MiB (157%) | 124.3 MiB (142%) |
| PFossil | Sparse | 1.3 MiB (112%) | 2.7 MiB (112%) | 5.8 MiB (112%) | 12.0 MiB (112%) | 24.4 MiB (112%) | 48.7 MiB (112%) | 98.0 MiB (112%) |
| Pypper | Boolean | 860 KiB | 1.7 MiB | 3.7 MiB | 7.6 MiB | 15.4 MiB | 30.7 MiB | 61.7 MiB |
| Pypper | NList | 1.9 MiB (230%) | 3.8 MiB (215%) | 8.0 MiB (218%) | 15.5 MiB (204%) | 31.4 MiB (204%) | 57.9 MiB (189%) | 106.0 MiB (172%) |
| Pypper | PrePostNList | 2.1 MiB (247%) | 4.1 MiB (232%) | 8.5 MiB (232%) | 16.5 MiB (217%) | 33.3 MiB (217%) | 61.4 MiB (200%) | 112.2 MiB (182%) |
| Pypper | Sparse | 1.0 MiB (125%) | 2.2 MiB (128%) | 4.7 MiB (128%) | 9.8 MiB (129%) | 19.8 MiB (129%) | 39.6 MiB (129%) | 79.6 MiB (129%) |
| PyLORD | Boolean | 744 KiB | 1.6 MiB | 4.3 MiB | 15.1 MiB |  |  |  |
| PyLORD | NList | 2.2 MiB (302%) | 4.4 MiB (281%) | 9.7 MiB (228%) | 25.1 MiB (167%) | 77.0 MiB |  |  |
| PyLORD | PrePostNList | 2.4 MiB (324%) | 4.7 MiB (299%) | 10.2 MiB (240%) | 26.1 MiB (173%) | 79.3 MiB |  |  |
| PyLORD | Sparse | 1.4 MiB (186%) | 2.9 MiB (186%) | 6.5 MiB (154%) | 19.2 MiB (127%) | 64.2 MiB |  |  |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 620 KiB | 1.3 MiB | 2.7 MiB | 5.6 MiB | 11.2 MiB | 22.4 MiB | 45.0 MiB |
| CN2 | NList | 431 KiB (69%) | 807 KiB (61%) | 1.4 MiB (51%) | 2.8 MiB (51%) | 5.9 MiB (53%) | 10.8 MiB (48%) | 18.8 MiB (42%) |
| CN2 | PrePostNList | 455 KiB (73%) | 846 KiB (64%) | 1.5 MiB (54%) | 3.0 MiB (54%) | 6.2 MiB (55%) | 11.2 MiB (50%) | 19.6 MiB (44%) |
| CN2 | Sparse | 301 KiB (48%) | 594 KiB (45%) | 976 KiB (35%) | 2.1 MiB (38%) | 4.4 MiB (39%) | 8.6 MiB (38%) | 15.5 MiB (35%) |
| PFossil | Boolean | 599 KiB | 1.3 MiB | 2.6 MiB | 5.5 MiB | 11.1 MiB | 22.2 MiB | 44.6 MiB |
| PFossil | NList | 429 KiB (72%) | 757 KiB (59%) | 1.3 MiB (50%) | 2.7 MiB (50%) | 5.8 MiB (52%) | 10.5 MiB (48%) | 18.4 MiB (41%) |
| PFossil | PrePostNList | 453 KiB (76%) | 799 KiB (62%) | 1.4 MiB (53%) | 2.9 MiB (53%) | 6.1 MiB (55%) | 11.0 MiB (50%) | 19.2 MiB (43%) |
| PFossil | Sparse | 274 KiB (46%) | 546 KiB (43%) | 909 KiB (34%) | 2.0 MiB (36%) | 4.3 MiB (39%) | 8.4 MiB (38%) | 15.2 MiB (34%) |
| Pypper | Boolean | 439 KiB | 929 KiB | 1.9 MiB | 3.9 MiB | 7.9 MiB | 15.8 MiB | 31.7 MiB |
| Pypper | NList | 428 KiB (98%) | 671 KiB (72%) | 1.2 MiB (62%) | 2.4 MiB (62%) | 5.1 MiB (64%) | 9.3 MiB (59%) | 16.2 MiB (51%) |
| Pypper | PrePostNList | 452 KiB (103%) | 710 KiB (76%) | 1.2 MiB (66%) | 2.6 MiB (66%) | 5.4 MiB (68%) | 9.7 MiB (62%) | 17.0 MiB (54%) |
| Pypper | Sparse | 233 KiB (53%) | 458 KiB (49%) | 767 KiB (39%) | 1.7 MiB (42%) | 3.5 MiB (45%) | 6.9 MiB (44%) | 12.4 MiB (39%) |
| PyLORD | Boolean | 704 KiB | 1.3 MiB | 3.8 MiB | 12.5 MiB | 44.2 MiB |  |  |
| PyLORD | NList | 610 KiB (87%) | 1.6 MiB (124%) | 4.3 MiB (113%) | 13.5 MiB (108%) | 46.7 MiB (106%) |  |  |
| PyLORD | PrePostNList | 635 KiB (90%) | 1.6 MiB (127%) | 4.3 MiB (115%) | 13.6 MiB (109%) | 47.0 MiB (106%) |  |  |
| PyLORD | Sparse | 490 KiB (70%) | 1.4 MiB (109%) | 3.9 MiB (103%) | 12.8 MiB (102%) | 44.8 MiB (101%) |  |  |

Peak memory while building (with negation) -- the vertical indexes need more than they keep:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 19 KiB | 36 KiB | 70 KiB | 138 KiB | 279 KiB | 552 KiB | 1.1 MiB |
| NList | 4.4 MiB (23403%) | 8.5 MiB (24326%) | 17.7 MiB (26019%) | 35.5 MiB (26276%) | 71.8 MiB (26387%) | 143.6 MiB (26625%) | 289.2 MiB (26939%) |
| PrePostNList | 4.6 MiB (24242%) | 8.8 MiB (25137%) | 18.3 MiB (26792%) | 35.5 MiB (26294%) | 71.8 MiB (26386%) | 143.6 MiB (26625%) | 289.2 MiB (26939%) |
| Sparse | 698 KiB (3628%) | 1.2 MiB (3304%) | 2.5 MiB (3601%) | 5.1 MiB (3806%) | 10.4 MiB (3825%) | 20.8 MiB (3860%) | 41.9 MiB (3907%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

## 2. Density

Synthetic data, 1000 training and 1000 test examples with 500 features: 4 of them carry a fixed learnable concept (true with probability 0.4, class = f0 and not f1, 5% label noise); every other feature is noise, true with the probability on the x-axis. Size and feature count stay fixed, so this isolates density.

![CN2 fit time vs. density](representations_plots/density_cn2.png)

![PFossil fit time vs. density](representations_plots/density_pfossil.png)

![Pypper fit time vs. density](representations_plots/density_pypper.png)

![PyLORD fit time vs. density](representations_plots/density_pylord.png)

The pure data operation, 800 `coverage()` calls for random rules of 1-4 conditions, without any search around it:

![coverage() time vs. density](representations_plots/density_coverage.png)

Fit time in seconds:

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.40 | 0.50 | 0.71 | 0.61 | 0.20 | 0.13 | 0.09 | 0.07 |
| CN2 | NList | 0.24 (59%) | 0.32 (65%) | 0.37 (51%) | 0.29 (47%) | 0.10 (50%) | 0.05 (39%) | 0.03 (36%) | 0.02 (26%) |
| CN2 | PrePostNList | 0.23 (57%) | 0.33 (68%) | 0.35 (50%) | 0.30 (50%) | 0.10 (50%) | 0.05 (37%) | 0.03 (34%) | 0.02 (28%) |
| CN2 | Sparse | 0.22 (55%) | 0.32 (65%) | 0.35 (49%) | 0.26 (42%) | 0.10 (47%) | 0.06 (42%) | 0.03 (33%) | 0.02 (25%) |
| PFossil | Boolean | 0.05 | 0.04 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| PFossil | NList | 0.03 (68%) | 0.02 (57%) | 0.01 (38%) | 0.01 (36%) | 0.01 (36%) | 0.01 (30%) | 0.01 (33%) | 0.01 (32%) |
| PFossil | PrePostNList | 0.06 (129%) | 0.02 (55%) | 0.01 (38%) | 0.01 (36%) | 0.01 (48%) | 0.01 (29%) | 0.01 (32%) | 0.01 (32%) |
| PFossil | Sparse | 0.02 (52%) | 0.02 (45%) | 0.01 (26%) | 0.01 (27%) | 0.00 (23%) | 0.00 (22%) | 0.00 (24%) | 0.00 (25%) |
| Pypper | Boolean | 0.15 | 0.02 | 0.05 | 0.02 | 0.08 | 0.03 | 0.06 | 0.03 |
| Pypper | NList | 0.06 (44%) | 0.02 (67%) | 0.02 (38%) | 0.01 (53%) | 0.03 (39%) | 0.02 (61%) | 0.02 (37%) | 0.01 (45%) |
| Pypper | PrePostNList | 0.07 (47%) | 0.02 (70%) | 0.02 (37%) | 0.01 (56%) | 0.03 (36%) | 0.01 (44%) | 0.02 (36%) | 0.01 (44%) |
| Pypper | Sparse | 0.05 (35%) | 0.01 (46%) | 0.02 (29%) | 0.01 (41%) | 0.02 (29%) | 0.01 (34%) | 0.02 (30%) | 0.01 (39%) |
| PyLORD | Boolean | 4.13 | 2.43 | 1.03 | 0.96 | 0.90 | 0.89 | 0.73 | 0.65 |
| PyLORD | NList | 3.18 (77%) | 1.97 (81%) | 0.99 (96%) | 0.95 (99%) | 0.90 (101%) | 0.81 (91%) | 0.75 (102%) | 0.64 (98%) |
| PyLORD | PrePostNList | 3.11 (75%) | 2.01 (83%) | 0.99 (96%) | 0.94 (98%) | 0.94 (106%) | 0.86 (96%) | 0.78 (106%) | 0.67 (103%) |
| PyLORD | Sparse | 3.13 (76%) | 1.87 (77%) | 0.93 (90%) | 0.88 (92%) | 0.88 (98%) | 0.77 (86%) | 0.72 (99%) | 0.70 (108%) |

Memory held by each representation (share of Boolean in parentheses; the last row is the unpacked bool matrix):

![memory vs. density](representations_plots/density_memory.png)

| representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB |
| NList | 16.9 MiB (27947%) | 10.1 MiB (16818%) | 3.4 MiB (5638%) | 1.7 MiB (2809%) | 709 KiB (1147%) | 376 KiB (608%) | 208 KiB (337%) | 111 KiB (180%) |
| PrePostNList | 18.7 MiB (31045%) | 11.3 MiB (18679%) | 3.8 MiB (6257%) | 1.9 MiB (3114%) | 783 KiB (1268%) | 413 KiB (669%) | 227 KiB (367%) | 119 KiB (193%) |
| Sparse | 2.4 MiB (3962%) | 1.4 MiB (2390%) | 509 KiB (824%) | 264 KiB (428%) | 119 KiB (193%) | 72 KiB (116%) | 48 KiB (77%) | 34 KiB (54%) |
| unpacked `X` | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 5.0 MiB | 4.9 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 5.0 MiB | 4.9 MiB | 4.9 MiB |
| CN2 | NList | 22.8 MiB (460%) | 13.8 MiB (279%) | 5.1 MiB (102%) | 3.0 MiB (60%) | 1.6 MiB (31%) | 1.1 MiB (22%) | 742 KiB (15%) | 485 KiB (10%) |
| CN2 | PrePostNList | 24.6 MiB (497%) | 14.9 MiB (301%) | 5.4 MiB (110%) | 3.2 MiB (64%) | 1.6 MiB (33%) | 1.1 MiB (23%) | 760 KiB (15%) | 492 KiB (10%) |
| CN2 | Sparse | 5.6 MiB (113%) | 3.5 MiB (70%) | 1.5 MiB (31%) | 1.2 MiB (25%) | 857 KiB (17%) | 703 KiB (14%) | 524 KiB (10%) | 373 KiB (7%) |
| PFossil | Boolean | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB |
| PFossil | NList | 22.7 MiB (461%) | 13.8 MiB (279%) | 4.7 MiB (96%) | 2.5 MiB (50%) | 1.1 MiB (23%) | 702 KiB (14%) | 489 KiB (10%) | 377 KiB (7%) |
| PFossil | PrePostNList | 24.6 MiB (499%) | 14.9 MiB (302%) | 5.1 MiB (104%) | 2.6 MiB (54%) | 1.2 MiB (24%) | 740 KiB (15%) | 508 KiB (10%) | 385 KiB (8%) |
| PFossil | Sparse | 5.6 MiB (114%) | 3.4 MiB (69%) | 1.2 MiB (25%) | 713 KiB (14%) | 380 KiB (8%) | 272 KiB (5%) | 243 KiB (5%) | 207 KiB (4%) |
| Pypper | Boolean | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB |
| Pypper | NList | 21.8 MiB (614%) | 13.2 MiB (377%) | 4.5 MiB (130%) | 2.3 MiB (67%) | 1.1 MiB (30%) | 665 KiB (19%) | 483 KiB (13%) | 371 KiB (10%) |
| Pypper | PrePostNList | 23.6 MiB (666%) | 14.3 MiB (409%) | 4.9 MiB (140%) | 2.5 MiB (72%) | 1.1 MiB (33%) | 702 KiB (20%) | 502 KiB (14%) | 379 KiB (11%) |
| Pypper | Sparse | 4.6 MiB (129%) | 2.8 MiB (80%) | 1.0 MiB (30%) | 596 KiB (17%) | 332 KiB (9%) | 256 KiB (7%) | 212 KiB (6%) | 209 KiB (6%) |
| PyLORD | Boolean | 3.3 MiB | 2.3 MiB | 1.6 MiB | 1.9 MiB | 2.0 MiB | 2.0 MiB | 2.0 MiB | 1.7 MiB |
| PyLORD | NList | 23.0 MiB (704%) | 14.2 MiB (617%) | 5.1 MiB (318%) | 3.5 MiB (188%) | 2.4 MiB (118%) | 2.0 MiB (99%) | 1.7 MiB (88%) | 1.3 MiB (78%) |
| PyLORD | PrePostNList | 24.9 MiB (761%) | 15.1 MiB (657%) | 5.5 MiB (342%) | 3.6 MiB (191%) | 2.5 MiB (121%) | 2.1 MiB (101%) | 1.8 MiB (89%) | 1.4 MiB (86%) |
| PyLORD | Sparse | 5.9 MiB (179%) | 3.6 MiB (157%) | 1.6 MiB (99%) | 1.6 MiB (88%) | 1.7 MiB (82%) | 1.6 MiB (80%) | 1.5 MiB (77%) | 1.2 MiB (71%) |

## 3. Width

Synthetic data again, 1000 training and 1000 test examples, the same concept on 4 signal features -- but now the number of *true* features per example stays fixed (on average 16 noise features, plus the signal ones) while the number of features grows. That is how sparse data usually looks: one-hot encodings of attributes with many values, or words in a text. The packed Boolean matrix grows with the number of features (one bit per cell); a sparse representation with the number of true cells, which stays the same. `Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes against 1/8 byte per cell for Boolean, so it should take less memory below a density of about 1/80 -- here from about 1,408 features on.

![memory vs. width](representations_plots/width_memory.png)

Memory held by each representation (share of Boolean in parentheses; the last row is the unpacked bool matrix):

| representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 13 KiB | 25 KiB | 62 KiB | 122 KiB | 244 KiB | 611 KiB | 1.2 MiB | 2.4 MiB |
| NList | 387 KiB (2998%) | 637 KiB (2586%) | 1.1 MiB (1846%) | 2.1 MiB (1754%) | 4.1 MiB (1703%) | 9.9 MiB (1652%) | 19.5 MiB (1634%) | 38.7 MiB (1624%) |
| PrePostNList | 505 KiB (3908%) | 758 KiB (3074%) | 1.2 MiB (2044%) | 2.2 MiB (1855%) | 4.2 MiB (1754%) | 10.0 MiB (1673%) | 19.6 MiB (1645%) | 38.9 MiB (1629%) |
| Sparse | 179 KiB (1384%) | 179 KiB (727%) | 180 KiB (292%) | 182 KiB (149%) | 186 KiB (76%) | 199 KiB (33%) | 218 KiB (18%) | 257 KiB (11%) |
| unpacked `X` | 98 KiB | 195 KiB | 488 KiB | 977 KiB | 1.9 MiB | 4.8 MiB | 9.5 MiB | 19.1 MiB |

800 `coverage()` calls for random rules of 1-4 conditions:

![coverage() time vs. width](representations_plots/width_coverage.png)

![fit time vs. width](representations_plots/width_fit.png)

Fit time in seconds:

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.20 | 0.20 | 0.70 | 0.50 | 1.06 | 4.14 | 11.85 | 24.28 |
| CN2 | NList | 0.25 (121%) | 0.20 (98%) | 0.31 (45%) | 0.18 (36%) | 0.24 (23%) | 0.65 (16%) | 0.93 (8%) | 1.96 (8%) |
| CN2 | PrePostNList | 0.24 (119%) | 0.17 (81%) | 0.31 (45%) | 0.17 (34%) | 0.24 (23%) | 0.65 (16%) | 0.98 (8%) | 1.95 (8%) |
| CN2 | Sparse | 0.21 (105%) | 0.13 (66%) | 0.32 (46%) | 0.16 (32%) | 0.23 (22%) | 0.59 (14%) | 0.93 (8%) | 1.83 (8%) |
| PFossil | Boolean | 0.01 | 0.01 | 0.02 | 0.04 | 0.07 | 0.15 | 0.28 | 0.54 |
| PFossil | NList | 0.01 (106%) | 0.01 (72%) | 0.01 (35%) | 0.01 (26%) | 0.01 (18%) | 0.03 (17%) | 0.04 (14%) | 0.07 (13%) |
| PFossil | PrePostNList | 0.01 (103%) | 0.01 (70%) | 0.01 (36%) | 0.01 (26%) | 0.01 (18%) | 0.02 (15%) | 0.04 (15%) | 0.08 (15%) |
| PFossil | Sparse | 0.01 (90%) | 0.01 (58%) | 0.00 (24%) | 0.01 (16%) | 0.01 (10%) | 0.01 (6%) | 0.01 (5%) | 0.02 (4%) |
| Pypper | Boolean | 0.03 | 0.01 | 0.03 | 0.07 | 0.07 | 1.46 | 1.42 | 2.13 |
| Pypper | NList | 0.03 (99%) | 0.01 (79%) | 0.01 (43%) | 0.02 (25%) | 0.02 (24%) | 0.10 (7%) | 0.10 (7%) | 0.17 (8%) |
| Pypper | PrePostNList | 0.03 (87%) | 0.01 (77%) | 0.01 (44%) | 0.02 (25%) | 0.02 (24%) | 0.11 (7%) | 0.10 (7%) | 0.17 (8%) |
| Pypper | Sparse | 0.02 (67%) | 0.01 (69%) | 0.01 (35%) | 0.01 (19%) | 0.01 (15%) | 0.08 (5%) | 0.06 (5%) | 0.10 (5%) |
| PyLORD | Boolean | 0.94 | 0.82 | 0.91 | 0.97 | 0.95 | 1.26 | 3.39 | 6.57 |
| PyLORD | NList | 1.07 (114%) | 0.95 (115%) | 0.97 (106%) | 0.85 (88%) | 0.75 (79%) | 0.64 (51%) | 0.67 (20%) | 0.79 (12%) |
| PyLORD | PrePostNList | 1.09 (116%) | 0.90 (110%) | 0.98 (107%) | 0.89 (92%) | 0.71 (74%) | 0.78 (62%) | 0.71 (21%) | 0.85 (13%) |
| PyLORD | Sparse | 1.04 (111%) | 0.87 (106%) | 0.89 (98%) | 0.85 (88%) | 0.70 (73%) | 0.59 (47%) | 0.64 (19%) | 0.67 (10%) |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 1.1 MiB | 2.0 MiB | 5.0 MiB | 9.8 MiB | 19.7 MiB | 49.2 MiB | 97.7 MiB | 196.2 MiB |
| CN2 | NList | 986 KiB (90%) | 1.3 MiB (63%) | 2.2 MiB (44%) | 3.6 MiB (36%) | 7.2 MiB (37%) | 16.8 MiB (34%) | 30.6 MiB (31%) | 60.5 MiB (31%) |
| CN2 | PrePostNList | 1.1 MiB (101%) | 1.4 MiB (68%) | 2.3 MiB (46%) | 3.7 MiB (38%) | 7.3 MiB (37%) | 16.9 MiB (34%) | 30.8 MiB (32%) | 60.6 MiB (31%) |
| CN2 | Sparse | 520 KiB (48%) | 609 KiB (29%) | 1.0 MiB (21%) | 1.4 MiB (15%) | 3.1 MiB (16%) | 6.9 MiB (14%) | 11.1 MiB (11%) | 21.7 MiB (11%) |
| PFossil | Boolean | 1.0 MiB | 2.0 MiB | 4.9 MiB | 9.8 MiB | 19.6 MiB | 49.1 MiB | 97.6 MiB | 196.2 MiB |
| PFossil | NList | 935 KiB (89%) | 1.2 MiB (58%) | 1.7 MiB (34%) | 2.7 MiB (28%) | 5.0 MiB (26%) | 11.9 MiB (24%) | 22.7 MiB (23%) | 45.3 MiB (23%) |
| PFossil | PrePostNList | 1.0 MiB (100%) | 1.3 MiB (64%) | 1.8 MiB (37%) | 2.8 MiB (29%) | 5.1 MiB (26%) | 12.0 MiB (24%) | 22.9 MiB (23%) | 45.5 MiB (23%) |
| PFossil | Sparse | 476 KiB (45%) | 479 KiB (23%) | 521 KiB (10%) | 554 KiB (6%) | 922 KiB (5%) | 2.0 MiB (4%) | 3.2 MiB (3%) | 6.6 MiB (3%) |
| Pypper | Boolean | 759 KiB | 1.4 MiB | 3.5 MiB | 6.9 MiB | 13.8 MiB | 34.8 MiB | 69.0 MiB | 138.9 MiB |
| Pypper | NList | 853 KiB (112%) | 1.1 MiB (76%) | 1.6 MiB (46%) | 2.6 MiB (38%) | 4.9 MiB (35%) | 11.7 MiB (34%) | 22.4 MiB (32%) | 45.7 MiB (33%) |
| Pypper | PrePostNList | 971 KiB (128%) | 1.2 MiB (84%) | 1.7 MiB (49%) | 2.7 MiB (40%) | 5.0 MiB (36%) | 11.8 MiB (34%) | 22.5 MiB (33%) | 45.8 MiB (33%) |
| Pypper | Sparse | 401 KiB (53%) | 399 KiB (27%) | 442 KiB (12%) | 487 KiB (7%) | 802 KiB (6%) | 1.8 MiB (5%) | 2.9 MiB (4%) | 7.0 MiB (5%) |
| PyLORD | Boolean | 1.4 MiB | 1.4 MiB | 2.0 MiB | 2.6 MiB | 4.1 MiB | 10.1 MiB | 20.3 MiB | 40.5 MiB |
| PyLORD | NList | 2.0 MiB (136%) | 2.1 MiB (148%) | 2.8 MiB (143%) | 3.7 MiB (145%) | 5.6 MiB (138%) | 11.5 MiB (114%) | 21.4 MiB (106%) | 41.1 MiB (101%) |
| PyLORD | PrePostNList | 2.1 MiB (144%) | 2.2 MiB (157%) | 2.9 MiB (149%) | 3.8 MiB (150%) | 5.9 MiB (144%) | 11.6 MiB (115%) | 21.5 MiB (106%) | 41.2 MiB (102%) |
| PyLORD | Sparse | 1.6 MiB (107%) | 1.5 MiB (111%) | 1.7 MiB (84%) | 1.6 MiB (61%) | 1.5 MiB (37%) | 1.6 MiB (16%) | 1.9 MiB (10%) | 2.4 MiB (6%) |

