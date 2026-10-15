# Data representations: same rules, different speed

**Full run.** Quick sanity check instead: `python demos/representations.py`.

pyrulearn's learners never look at the data directly. They ask it two things -- which examples a rule covers (`coverage(rule)`) and which features an example has (`features_of(row)`) -- so the data can be stored in any structure that answers these questions. Four such representations are implemented:

- **Boolean** -- a packed bit matrix, one bit per example and feature. Covering a rule is a bitwise AND over the rows, whatever the density.
- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are inserted into a prefix tree of their true features, and each feature keeps the list of tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work grows with how often the features are true, not with the number of examples.
- **PrePostNList** -- the N-list plus pre/post-order codes of the tree nodes, a second way to find which nodes are compatible with the rule so far.
- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix tree.

All four must lead to exactly the same rules, so the only thing that differs is time. This demo measures that time for five learners -- CN2, PFossil, Pypper, PyLORD and ENDER, each with its default settings -- and checks the rules and the test-set predictions against the Boolean version at every point. Each fit runs in a separate process and is capped at 300s; after a time-out, the larger points of that curve are skipped. The learners ran side by side, one worker process each, every learner's fits one after another; building the representations, timing `coverage()` and measuring memory came first, with nothing else running.

**Equality check:** at 147 points where Boolean and at least one other representation finished, 147 gave identical rules and predictions for every representation.

## 1. Training-set size

Training sets of doubling size, drawn from one shuffled pool of the `adult` dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the same 20% test set. Numeric attributes are discretized per training set (`max_intervals=8`). Two encodings: **with negation** (pyrulearn's default -- every test comes with its negation, so exactly half the features are true in every example) and **without negation** (positive tests only, far sparser). The first is the Boolean matrix's best case and the vertical representations' worst; the second is where they should gain.

At the largest training set: with negation 278 binary features, density 0.493; without negation 140 binary features, density 0.136 (density = share of features that are true).

One pair of plots per learner: left with a logarithmic time axis (ratios -- a constant factor between two representations is a constant gap), right with a linear one (absolute seconds saved).

<img src="representations_plots/size_cn2.png" alt="CN2 fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_cn2_linear.png" alt="CN2 fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pfossil.png" alt="PFossil fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pfossil_linear.png" alt="PFossil fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pypper.png" alt="Pypper fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pypper_linear.png" alt="Pypper fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_pylord.png" alt="PyLORD fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_pylord_linear.png" alt="PyLORD fit time vs. training-set size, linear scale" width="49%">

<img src="representations_plots/size_ender.png" alt="ENDER fit time vs. training-set size, log scale" width="49%"> <img src="representations_plots/size_ender_linear.png" alt="ENDER fit time vs. training-set size, linear scale" width="49%">

Fit time in seconds, with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.07 | 0.07 | 0.21 | 0.54 | 1.41 | 5.42 | 48.49 |
| CN2 | NList | 0.08 (114%) | 0.06 (94%) | 0.18 (86%) | 0.44 (82%) | 1.00 (71%) | 3.05 (56%) | 21.31 (44%) |
| CN2 | PrePostNList | 0.08 (106%) | 0.06 (89%) | 0.18 (85%) | 0.44 (80%) | 0.99 (70%) | 3.02 (56%) | 20.79 (43%) |
| CN2 | Sparse | 0.08 (116%) | 0.06 (92%) | 0.19 (89%) | 0.49 (90%) | 1.14 (80%) | 3.64 (67%) | 28.42 (59%) |
| PFossil | Boolean | 0.02 | 0.04 | 0.06 | 0.11 | 0.19 | 0.40 | 0.85 |
| PFossil | NList | 0.03 (100%) | 0.03 (88%) | 0.04 (69%) | 0.08 (68%) | 0.12 (61%) | 0.23 (58%) | 0.43 (50%) |
| PFossil | PrePostNList | 0.02 (96%) | 0.03 (88%) | 0.04 (68%) | 0.08 (67%) | 0.12 (61%) | 0.24 (59%) | 0.43 (51%) |
| PFossil | Sparse | 0.03 (101%) | 0.03 (90%) | 0.04 (66%) | 0.08 (68%) | 0.13 (65%) | 0.24 (60%) | 0.46 (54%) |
| Pypper | Boolean | 0.12 | 0.28 | 0.20 | 0.70 | 1.97 | 3.02 | 27.30 |
| Pypper | NList | 0.08 (67%) | 0.21 (75%) | 0.14 (67%) | 0.34 (48%) | 0.74 (37%) | 0.90 (30%) | 5.22 (19%) |
| Pypper | PrePostNList | 0.08 (65%) | 0.22 (79%) | 0.14 (67%) | 0.34 (48%) | 0.73 (37%) | 0.88 (29%) | 5.15 (19%) |
| Pypper | Sparse | 0.08 (63%) | 0.16 (57%) | 0.12 (57%) | 0.40 (56%) | 1.03 (52%) | 1.31 (44%) | 13.60 (50%) |
| PyLORD | Boolean | 1.01 | 2.89 | 10.08 | 35.42 | 120.72 | time-out |  |
| PyLORD | NList | 1.08 (106%) | 2.86 (99%) | 9.26 (92%) | 28.78 (81%) | 86.22 (71%) | time-out |  |
| PyLORD | PrePostNList | 1.05 (104%) | 2.84 (98%) | 9.30 (92%) | 28.47 (80%) | 86.89 (72%) | 295.75 | time-out |
| PyLORD | Sparse | 1.13 (112%) | 3.13 (108%) | 10.68 (106%) | 36.39 (103%) | 121.08 (100%) | time-out |  |
| ENDER | Boolean | 0.93 | 1.59 | 2.61 | 4.48 | 8.02 | 14.54 | 27.28 |
| ENDER | NList | 1.09 (117%) | 1.56 (98%) | 2.32 (89%) | 3.66 (82%) | 6.35 (79%) | 11.32 (78%) | 20.47 (75%) |
| ENDER | PrePostNList | 1.17 (126%) | 1.52 (96%) | 2.30 (88%) | 3.65 (82%) | 6.29 (78%) | 11.35 (78%) | 20.31 (74%) |
| ENDER | Sparse | 1.26 (136%) | 1.81 (114%) | 2.62 (100%) | 4.02 (90%) | 6.72 (84%) | 11.57 (80%) | 21.07 (77%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.07 | 0.20 | 0.34 | 0.86 | 1.98 | 6.22 | 29.27 |
| CN2 | NList | 0.09 (126%) | 0.24 (121%) | 0.37 (109%) | 0.80 (93%) | 1.52 (77%) | 4.05 (65%) | 15.60 (53%) |
| CN2 | PrePostNList | 0.09 (123%) | 0.24 (119%) | 0.37 (109%) | 0.79 (92%) | 1.50 (76%) | 4.03 (65%) | 15.50 (53%) |
| CN2 | Sparse | 0.10 (136%) | 0.27 (135%) | 0.41 (120%) | 0.92 (107%) | 1.68 (85%) | 4.60 (74%) | 17.95 (61%) |
| PFossil | Boolean | 0.02 | 0.01 | 0.02 | 0.03 | 0.06 | 0.14 | 0.30 |
| PFossil | NList | 0.02 (101%) | 0.01 (95%) | 0.02 (89%) | 0.03 (81%) | 0.05 (77%) | 0.10 (73%) | 0.19 (64%) |
| PFossil | PrePostNList | 0.02 (100%) | 0.01 (94%) | 0.02 (86%) | 0.03 (81%) | 0.05 (77%) | 0.10 (73%) | 0.19 (65%) |
| PFossil | Sparse | 0.02 (106%) | 0.01 (88%) | 0.01 (78%) | 0.02 (70%) | 0.04 (65%) | 0.09 (62%) | 0.17 (57%) |
| Pypper | Boolean | 0.02 | 0.05 | 0.29 | 0.73 | 1.84 | 11.57 | 28.81 |
| Pypper | NList | 0.02 (107%) | 0.04 (93%) | 0.18 (62%) | 0.33 (44%) | 0.59 (32%) | 2.21 (19%) | 4.15 (14%) |
| Pypper | PrePostNList | 0.02 (106%) | 0.04 (97%) | 0.19 (65%) | 0.32 (44%) | 0.58 (32%) | 2.20 (19%) | 4.16 (14%) |
| Pypper | Sparse | 0.02 (102%) | 0.04 (87%) | 0.15 (52%) | 0.38 (52%) | 0.83 (45%) | 4.62 (40%) | 11.03 (38%) |
| PyLORD | Boolean | 0.68 | 1.93 | 5.11 | 16.68 | 55.88 | 196.22 | time-out |
| PyLORD | NList | 0.79 (117%) | 2.14 (111%) | 5.49 (107%) | 16.97 (102%) | 50.87 (91%) | 155.96 (79%) | time-out |
| PyLORD | PrePostNList | 0.79 (116%) | 2.12 (110%) | 5.49 (107%) | 16.98 (102%) | 50.91 (91%) | 155.23 (79%) | time-out |
| PyLORD | Sparse | 0.83 (122%) | 2.20 (114%) | 5.52 (108%) | 17.05 (102%) | 51.35 (92%) | 163.15 (83%) | time-out |
| ENDER | Boolean | 0.45 | 0.72 | 1.11 | 2.10 | 3.87 | 7.17 | 13.66 |
| ENDER | NList | 0.51 (114%) | 0.75 (104%) | 1.07 (96%) | 1.91 (91%) | 3.47 (89%) | 6.28 (88%) | 11.72 (86%) |
| ENDER | PrePostNList | 0.50 (112%) | 0.74 (103%) | 1.09 (98%) | 1.91 (91%) | 3.44 (89%) | 6.26 (87%) | 11.67 (85%) |
| ENDER | Sparse | 0.58 (130%) | 0.83 (115%) | 1.15 (103%) | 1.92 (92%) | 3.40 (88%) | 6.07 (85%) | 11.34 (83%) |

**Growth.** How fast each learner's fit time grows with the training-set size: the exponent *b* of time ~ n^*b*, fitted by least squares to log time over log n from n=2,000 on (below that, a fit's fixed costs still dominate). 1 is linear, 2 quadratic; a curve cut short by a time-out is fitted on the points that finished. This is what decides, for example, how much a learner gains from splitting a problem into smaller ones (`demos/multiclass_decomposition.py`):

| learner | representation | with negation | without negation |
|---|---|--:|--:|
| CN2 | Boolean | 1.91 | 1.57 |
| CN2 | NList | 1.66 | 1.31 |
| CN2 | PrePostNList | 1.65 | 1.31 |
| CN2 | Sparse | 1.74 | 1.32 |
| PFossil | Boolean | 0.96 | 1.02 |
| PFossil | NList | 0.85 | 0.91 |
| PFossil | PrePostNList | 0.86 | 0.92 |
| PFossil | Sparse | 0.89 | 0.91 |
| Pypper | Boolean | 1.63 | 1.73 |
| Pypper | NList | 1.19 | 1.18 |
| Pypper | PrePostNList | 1.19 | 1.17 |
| Pypper | Sparse | 1.55 | 1.60 |
| PyLORD | Boolean | 1.79 (3 points) | 1.75 (4 points) |
| PyLORD | NList | 1.61 (3 points) | 1.61 (4 points) |
| PyLORD | PrePostNList | 1.66 (4 points) | 1.61 (4 points) |
| PyLORD | Sparse | 1.75 (3 points) | 1.62 (4 points) |
| ENDER | Boolean | 0.85 | 0.90 |
| ENDER | NList | 0.79 | 0.86 |
| ENDER | PrePostNList | 0.79 | 0.86 |
| ENDER | Sparse | 0.75 | 0.83 |

Building a vertical index is paid once per fit, on top of the fit itself:

![index build time vs. training-set size](representations_plots/size_index_build.png)

**Memory.** What each representation holds once built, measured with `tracemalloc` (the input matrix and the Boolean matrix a vertical index is built from don't count), and in parentheses its share of the Boolean representation's memory; the last row is the unpacked `n x k` bool matrix that `data.X` builds on demand, which any learner or external library working on a plain array pays on top.

![memory vs. training-set size](representations_plots/size_memory.png)

With negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 14 KiB | 31 KiB | 65 KiB | 133 KiB | 274 KiB | 547 KiB | 1.1 MiB |
| NList | 828 KiB (5868%) | 1.5 MiB (4858%) | 3.2 MiB (5109%) | 6.0 MiB (4592%) | 12.0 MiB (4505%) | 21.5 MiB (4020%) | 37.7 MiB (3528%) |
| PrePostNList | 989 KiB (7013%) | 1.7 MiB (5805%) | 3.8 MiB (5944%) | 6.9 MiB (5341%) | 14.0 MiB (5240%) | 25.0 MiB (4675%) | 43.8 MiB (4101%) |
| Sparse | 539 KiB (3823%) | 1.2 MiB (3857%) | 2.5 MiB (3887%) | 5.1 MiB (3953%) | 10.4 MiB (3896%) | 20.8 MiB (3897%) | 41.9 MiB (3925%) |
| unpacked `X` | 108 KiB | 238 KiB | 508 KiB | 1.0 MiB | 2.1 MiB | 4.2 MiB | 8.5 MiB |

Without negation:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 7 KiB | 16 KiB | 33 KiB | 71 KiB | 141 KiB | 281 KiB | 563 KiB |
| NList | 91 KiB (1292%) | 163 KiB (1027%) | 361 KiB (1079%) | 688 KiB (975%) | 1.4 MiB (985%) | 2.4 MiB (858%) | 4.2 MiB (763%) |
| PrePostNList | 116 KiB (1634%) | 205 KiB (1295%) | 435 KiB (1300%) | 827 KiB (1173%) | 1.6 MiB (1186%) | 2.8 MiB (1027%) | 5.0 MiB (909%) |
| Sparse | 99 KiB (1405%) | 207 KiB (1307%) | 344 KiB (1030%) | 795 KiB (1126%) | 1.7 MiB (1234%) | 3.3 MiB (1202%) | 5.9 MiB (1080%) |
| unpacked `X` | 55 KiB | 120 KiB | 256 KiB | 535 KiB | 1.1 MiB | 2.1 MiB | 4.3 MiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass), with negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 1.2 MiB | 2.4 MiB | 5.2 MiB | 10.8 MiB | 21.8 MiB | 43.5 MiB | 87.7 MiB |
| CN2 | NList | 2.0 MiB (168%) | 3.8 MiB (155%) | 6.2 MiB (120%) | 10.8 MiB (100%) | 21.8 MiB (100%) | 39.9 MiB (92%) | 72.1 MiB (82%) |
| CN2 | PrePostNList | 2.1 MiB (178%) | 4.1 MiB (167%) | 6.7 MiB (130%) | 11.8 MiB (109%) | 23.8 MiB (109%) | 43.4 MiB (100%) | 78.2 MiB (89%) |
| CN2 | Sparse | 1.3 MiB (111%) | 2.8 MiB (112%) | 5.8 MiB (113%) | 12.1 MiB (112%) | 24.5 MiB (112%) | 48.9 MiB (112%) | 98.4 MiB (112%) |
| PFossil | Boolean | 1.1 MiB | 2.4 MiB | 5.1 MiB | 10.7 MiB | 21.7 MiB | 43.4 MiB | 87.3 MiB |
| PFossil | NList | 1.9 MiB (170%) | 3.8 MiB (155%) | 6.2 MiB (120%) | 10.8 MiB (101%) | 21.8 MiB (101%) | 39.9 MiB (92%) | 72.1 MiB (83%) |
| PFossil | PrePostNList | 2.1 MiB (184%) | 4.1 MiB (167%) | 6.7 MiB (130%) | 11.8 MiB (110%) | 23.8 MiB (110%) | 43.4 MiB (100%) | 78.2 MiB (90%) |
| PFossil | Sparse | 1.3 MiB (112%) | 2.7 MiB (112%) | 5.8 MiB (112%) | 12.0 MiB (112%) | 24.4 MiB (112%) | 48.7 MiB (112%) | 98.0 MiB (112%) |
| Pypper | Boolean | 869 KiB | 1.8 MiB | 3.7 MiB | 7.6 MiB | 15.4 MiB | 30.7 MiB | 61.7 MiB |
| Pypper | NList | 1.7 MiB (199%) | 3.1 MiB (178%) | 6.1 MiB (168%) | 10.8 MiB (142%) | 21.8 MiB (142%) | 39.7 MiB (130%) | 71.9 MiB (116%) |
| Pypper | PrePostNList | 1.8 MiB (218%) | 3.4 MiB (194%) | 6.7 MiB (182%) | 11.7 MiB (155%) | 23.7 MiB (154%) | 43.2 MiB (141%) | 78.0 MiB (126%) |
| Pypper | Sparse | 1.1 MiB (126%) | 2.3 MiB (129%) | 4.7 MiB (129%) | 9.8 MiB (129%) | 19.8 MiB (129%) | 39.6 MiB (129%) | 79.6 MiB (129%) |
| PyLORD | Boolean | 744 KiB | 1.6 MiB | 4.5 MiB | 14.8 MiB | 56.1 MiB |  |  |
| PyLORD | NList | 2.0 MiB (275%) | 4.0 MiB (254%) | 7.4 MiB (164%) | 20.6 MiB (139%) | 67.7 MiB (121%) |  |  |
| PyLORD | PrePostNList | 2.2 MiB (297%) | 4.3 MiB (273%) | 7.9 MiB (176%) | 21.5 MiB (145%) | 69.6 MiB (124%) |  |  |
| PyLORD | Sparse | 1.4 MiB (186%) | 2.9 MiB (188%) | 6.2 MiB (138%) | 18.8 MiB (127%) |  |  |  |
| ENDER | Boolean | 1014 KiB | 1.3 MiB | 2.0 MiB | 4.1 MiB | 8.2 MiB | 16.3 MiB | 32.7 MiB |
| ENDER | NList | 1.8 MiB (181%) | 2.8 MiB (219%) | 5.8 MiB (291%) | 11.1 MiB (273%) | 22.4 MiB (274%) | 41.0 MiB (252%) | 74.5 MiB (227%) |
| ENDER | PrePostNList | 1.9 MiB (196%) | 3.0 MiB (242%) | 6.3 MiB (317%) | 12.1 MiB (296%) | 24.4 MiB (298%) | 44.5 MiB (273%) | 80.6 MiB (246%) |
| ENDER | Sparse | 1.4 MiB (141%) | 2.1 MiB (170%) | 3.6 MiB (181%) | 7.4 MiB (183%) | 15.0 MiB (183%) | 29.8 MiB (183%) | 59.9 MiB (183%) |

Without negation:

| learner | representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 620 KiB | 1.3 MiB | 2.7 MiB | 5.6 MiB | 11.2 MiB | 22.4 MiB | 45.0 MiB |
| CN2 | NList | 721 KiB (116%) | 1.4 MiB (110%) | 2.9 MiB (108%) | 3.9 MiB (69%) | 5.4 MiB (48%) | 8.1 MiB (36%) | 13.0 MiB (29%) |
| CN2 | PrePostNList | 741 KiB (119%) | 1.5 MiB (114%) | 3.0 MiB (111%) | 4.0 MiB (72%) | 5.7 MiB (51%) | 8.5 MiB (38%) | 13.8 MiB (31%) |
| CN2 | Sparse | 306 KiB (49%) | 593 KiB (45%) | 981 KiB (36%) | 2.1 MiB (38%) | 4.4 MiB (40%) | 8.6 MiB (38%) | 15.5 MiB (35%) |
| PFossil | Boolean | 598 KiB | 1.3 MiB | 2.6 MiB | 5.5 MiB | 11.1 MiB | 22.2 MiB | 44.6 MiB |
| PFossil | NList | 696 KiB (116%) | 1.4 MiB (111%) | 2.9 MiB (108%) | 3.8 MiB (69%) | 5.3 MiB (48%) | 7.8 MiB (35%) | 12.6 MiB (28%) |
| PFossil | PrePostNList | 720 KiB (120%) | 1.4 MiB (114%) | 2.9 MiB (111%) | 3.9 MiB (71%) | 5.6 MiB (50%) | 8.3 MiB (37%) | 13.4 MiB (30%) |
| PFossil | Sparse | 274 KiB (46%) | 546 KiB (43%) | 909 KiB (34%) | 2.0 MiB (36%) | 4.3 MiB (39%) | 8.4 MiB (38%) | 15.2 MiB (34%) |
| Pypper | Boolean | 438 KiB | 928 KiB | 1.9 MiB | 3.9 MiB | 7.9 MiB | 15.8 MiB | 31.7 MiB |
| Pypper | NList | 531 KiB (121%) | 1.1 MiB (116%) | 2.2 MiB (115%) | 3.7 MiB (95%) | 5.2 MiB (66%) | 7.6 MiB (48%) | 12.1 MiB (38%) |
| Pypper | PrePostNList | 554 KiB (127%) | 1.1 MiB (121%) | 2.2 MiB (119%) | 3.9 MiB (98%) | 5.5 MiB (69%) | 8.1 MiB (51%) | 12.9 MiB (41%) |
| Pypper | Sparse | 233 KiB (53%) | 458 KiB (49%) | 766 KiB (39%) | 1.7 MiB (42%) | 3.5 MiB (45%) | 6.9 MiB (44%) | 12.4 MiB (39%) |
| PyLORD | Boolean | 435 KiB | 1.3 MiB | 3.9 MiB | 12.5 MiB | 44.6 MiB |  |  |
| PyLORD | NList | 768 KiB (176%) | 1.5 MiB (122%) | 4.1 MiB (105%) | 13.1 MiB (105%) | 45.4 MiB (102%) |  |  |
| PyLORD | PrePostNList | 791 KiB (182%) | 1.6 MiB (125%) | 4.2 MiB (108%) | 13.2 MiB (106%) | 45.6 MiB (102%) |  |  |
| PyLORD | Sparse | 492 KiB (113%) | 1.4 MiB (107%) | 3.9 MiB (99%) | 12.7 MiB (102%) | 44.7 MiB (100%) |  |  |
| ENDER | Boolean | 579 KiB | 729 KiB | 1.1 MiB | 2.3 MiB | 4.6 MiB | 9.2 MiB | 18.5 MiB |
| ENDER | NList | 668 KiB (115%) | 938 KiB (129%) | 1.5 MiB (129%) | 2.9 MiB (125%) | 5.7 MiB (124%) | 8.7 MiB (94%) | 14.3 MiB (77%) |
| ENDER | PrePostNList | 691 KiB (119%) | 917 KiB (126%) | 1.5 MiB (135%) | 3.0 MiB (131%) | 6.0 MiB (130%) | 9.2 MiB (99%) | 15.1 MiB (82%) |
| ENDER | Sparse | 643 KiB (111%) | 804 KiB (110%) | 975 KiB (83%) | 1.7 MiB (71%) | 3.4 MiB (73%) | 6.6 MiB (71%) | 12.2 MiB (66%) |

Peak memory while building (with negation) -- the vertical indexes need more than they keep:

| representation | n=500 | n=1000 | n=2000 | n=4000 | n=8000 | n=16000 | n=32000 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 19 KiB | 36 KiB | 70 KiB | 138 KiB | 279 KiB | 552 KiB | 1.1 MiB |
| NList | 4.4 MiB (23384%) | 8.5 MiB (24321%) | 17.7 MiB (26016%) | 34.5 MiB (25571%) | 69.8 MiB (25629%) | 130.9 MiB (24267%) | 255.7 MiB (23823%) |
| PrePostNList | 4.6 MiB (24223%) | 8.8 MiB (25132%) | 18.3 MiB (26789%) | 35.5 MiB (26292%) | 71.7 MiB (26351%) | 134.4 MiB (24916%) | 255.7 MiB (23823%) |
| Sparse | 698 KiB (3625%) | 1.2 MiB (3303%) | 2.5 MiB (3601%) | 5.1 MiB (3806%) | 10.4 MiB (3825%) | 20.8 MiB (3860%) | 41.9 MiB (3907%) |
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
| CN2 | Boolean | 0.34 | 0.50 | 0.63 | 0.56 | 0.18 | 0.12 | 0.08 | 0.06 |
| CN2 | NList | 0.29 (84%) | 0.44 (87%) | 0.52 (83%) | 0.44 (78%) | 0.16 (87%) | 0.09 (71%) | 0.06 (69%) | 0.03 (54%) |
| CN2 | PrePostNList | 0.27 (80%) | 0.43 (84%) | 0.52 (83%) | 0.44 (78%) | 0.16 (86%) | 0.08 (70%) | 0.05 (65%) | 0.03 (52%) |
| CN2 | Sparse | 0.29 (84%) | 0.42 (84%) | 0.51 (81%) | 0.41 (74%) | 0.15 (83%) | 0.08 (63%) | 0.05 (59%) | 0.03 (49%) |
| PFossil | Boolean | 0.04 | 0.04 | 0.01 | 0.02 | 0.01 | 0.01 | 0.01 | 0.01 |
| PFossil | NList | 0.03 (92%) | 0.03 (86%) | 0.01 (84%) | 0.01 (78%) | 0.01 (76%) | 0.01 (71%) | 0.01 (68%) | 0.01 (65%) |
| PFossil | PrePostNList | 0.03 (90%) | 0.03 (85%) | 0.01 (83%) | 0.01 (77%) | 0.01 (76%) | 0.01 (69%) | 0.01 (67%) | 0.01 (64%) |
| PFossil | Sparse | 0.03 (79%) | 0.03 (72%) | 0.01 (57%) | 0.01 (54%) | 0.01 (52%) | 0.01 (52%) | 0.01 (51%) | 0.01 (52%) |
| Pypper | Boolean | 0.11 | 0.02 | 0.04 | 0.02 | 0.08 | 0.03 | 0.06 | 0.03 |
| Pypper | NList | 0.09 (81%) | 0.02 (91%) | 0.03 (75%) | 0.02 (82%) | 0.05 (60%) | 0.02 (71%) | 0.04 (63%) | 0.02 (76%) |
| Pypper | PrePostNList | 0.09 (79%) | 0.02 (94%) | 0.03 (75%) | 0.02 (83%) | 0.06 (66%) | 0.02 (69%) | 0.04 (62%) | 0.02 (70%) |
| Pypper | Sparse | 0.07 (59%) | 0.02 (68%) | 0.02 (55%) | 0.01 (62%) | 0.04 (44%) | 0.02 (57%) | 0.03 (48%) | 0.02 (60%) |
| PyLORD | Boolean | 3.94 | 2.47 | 1.54 | 1.42 | 1.36 | 1.26 | 1.17 | 1.00 |
| PyLORD | NList | 3.53 (90%) | 2.56 (104%) | 1.92 (125%) | 1.92 (135%) | 1.90 (140%) | 1.41 (112%) | 1.29 (110%) | 1.12 (112%) |
| PyLORD | PrePostNList | 3.53 (89%) | 2.55 (103%) | 1.91 (125%) | 1.91 (134%) | 1.90 (140%) | 1.41 (111%) | 1.29 (110%) | 1.11 (112%) |
| PyLORD | Sparse | 3.51 (89%) | 2.26 (92%) | 1.45 (94%) | 1.41 (99%) | 1.36 (100%) | 1.26 (100%) | 1.16 (99%) | 1.03 (103%) |
| ENDER | Boolean | 1.04 | 0.86 | 0.77 | 0.74 | 0.71 | 0.71 | 0.71 | 0.71 |
| ENDER | NList | 0.86 (83%) | 0.71 (83%) | 0.62 (80%) | 0.58 (79%) | 0.55 (77%) | 0.57 (80%) | 0.56 (79%) | 0.55 (78%) |
| ENDER | PrePostNList | 0.85 (82%) | 0.69 (80%) | 0.61 (79%) | 0.58 (78%) | 0.55 (77%) | 0.56 (79%) | 0.55 (78%) | 0.54 (76%) |
| ENDER | Sparse | 0.91 (87%) | 0.71 (83%) | 0.62 (80%) | 0.57 (78%) | 0.54 (76%) | 0.53 (74%) | 0.52 (74%) | 0.52 (73%) |

Memory held by each representation (share of Boolean in parentheses; the last row is the unpacked bool matrix):

![memory vs. density](representations_plots/density_memory.png)

| representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB | 62 KiB |
| NList | 16.9 MiB (27943%) | 10.1 MiB (16816%) | 3.4 MiB (5637%) | 1.7 MiB (2810%) | 709 KiB (1148%) | 376 KiB (608%) | 208 KiB (337%) | 111 KiB (180%) |
| PrePostNList | 18.7 MiB (31041%) | 11.3 MiB (18677%) | 3.8 MiB (6256%) | 1.9 MiB (3114%) | 783 KiB (1268%) | 413 KiB (669%) | 227 KiB (368%) | 119 KiB (193%) |
| Sparse | 2.4 MiB (3961%) | 1.4 MiB (2390%) | 509 KiB (824%) | 264 KiB (427%) | 119 KiB (193%) | 72 KiB (116%) | 48 KiB (77%) | 33 KiB (54%) |
| unpacked `X` | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB | 488 KiB |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | p=0.5 | p=0.3 | p=0.1 | p=0.05 | p=0.02 | p=0.01 | p=0.005 | p=0.002 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 4.9 MiB | 4.9 MiB | 5.0 MiB | 5.0 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB |
| CN2 | NList | 20.3 MiB (410%) | 13.1 MiB (264%) | 6.3 MiB (128%) | 4.6 MiB (93%) | 3.6 MiB (73%) | 1.1 MiB (22%) | 742 KiB (15%) | 484 KiB (10%) |
| CN2 | PrePostNList | 22.1 MiB (448%) | 14.2 MiB (287%) | 6.7 MiB (135%) | 4.8 MiB (97%) | 3.7 MiB (74%) | 1.1 MiB (22%) | 760 KiB (15%) | 491 KiB (10%) |
| CN2 | Sparse | 5.6 MiB (114%) | 3.5 MiB (70%) | 1.5 MiB (31%) | 1.2 MiB (25%) | 857 KiB (17%) | 693 KiB (14%) | 523 KiB (10%) | 372 KiB (7%) |
| PFossil | Boolean | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB | 4.9 MiB |
| PFossil | NList | 19.7 MiB (400%) | 13.0 MiB (264%) | 6.3 MiB (127%) | 4.6 MiB (93%) | 3.6 MiB (73%) | 701 KiB (14%) | 481 KiB (10%) | 369 KiB (7%) |
| PFossil | PrePostNList | 21.6 MiB (438%) | 14.1 MiB (287%) | 6.7 MiB (135%) | 4.8 MiB (97%) | 3.6 MiB (74%) | 738 KiB (15%) | 500 KiB (10%) | 377 KiB (7%) |
| PFossil | Sparse | 5.6 MiB (114%) | 3.4 MiB (69%) | 1.2 MiB (25%) | 713 KiB (14%) | 380 KiB (8%) | 271 KiB (5%) | 243 KiB (5%) | 207 KiB (4%) |
| Pypper | Boolean | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB | 3.5 MiB |
| Pypper | NList | 19.7 MiB (564%) | 13.0 MiB (372%) | 6.3 MiB (179%) | 4.6 MiB (131%) | 3.6 MiB (102%) | 663 KiB (19%) | 475 KiB (13%) | 363 KiB (10%) |
| Pypper | PrePostNList | 21.6 MiB (617%) | 14.1 MiB (405%) | 6.6 MiB (190%) | 4.7 MiB (136%) | 3.6 MiB (104%) | 700 KiB (20%) | 494 KiB (14%) | 371 KiB (10%) |
| Pypper | Sparse | 4.6 MiB (131%) | 2.8 MiB (80%) | 1.0 MiB (30%) | 597 KiB (17%) | 333 KiB (9%) | 256 KiB (7%) | 213 KiB (6%) | 209 KiB (6%) |
| PyLORD | Boolean | 3.3 MiB | 2.3 MiB | 1.8 MiB | 1.9 MiB | 2.0 MiB | 2.0 MiB | 2.0 MiB | 1.7 MiB |
| PyLORD | NList | 20.0 MiB (613%) | 13.2 MiB (576%) | 6.4 MiB (362%) | 4.8 MiB (255%) | 3.8 MiB (186%) | 2.0 MiB (99%) | 1.8 MiB (89%) | 1.3 MiB (78%) |
| PyLORD | PrePostNList | 21.9 MiB (670%) | 14.3 MiB (625%) | 6.8 MiB (383%) | 5.0 MiB (264%) | 3.9 MiB (190%) | 2.1 MiB (101%) | 1.8 MiB (90%) | 1.3 MiB (79%) |
| PyLORD | Sparse | 5.9 MiB (179%) | 3.6 MiB (158%) | 1.6 MiB (91%) | 1.7 MiB (88%) | 1.7 MiB (83%) | 1.6 MiB (80%) | 1.5 MiB (78%) | 1.2 MiB (72%) |
| ENDER | Boolean | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB | 1.8 MiB |
| ENDER | NList | 19.4 MiB (1069%) | 11.9 MiB (653%) | 5.2 MiB (284%) | 3.5 MiB (190%) | 2.5 MiB (136%) | 816 KiB (44%) | 607 KiB (32%) | 476 KiB (25%) |
| ENDER | PrePostNList | 21.3 MiB (1172%) | 13.1 MiB (715%) | 5.6 MiB (304%) | 3.7 MiB (200%) | 2.5 MiB (139%) | 854 KiB (46%) | 624 KiB (33%) | 486 KiB (26%) |
| ENDER | Sparse | 3.4 MiB (188%) | 2.2 MiB (118%) | 1.0 MiB (55%) | 710 KiB (38%) | 501 KiB (27%) | 423 KiB (23%) | 383 KiB (20%) | 363 KiB (19%) |

## 3. Width

Synthetic data again, 1000 training and 1000 test examples, the same concept on 4 signal features -- but now the number of *true* features per example stays fixed (on average 16 noise features, plus the signal ones) while the number of features grows. That is how sparse data usually looks: one-hot encodings of attributes with many values, or words in a text. The packed Boolean matrix grows with the number of features (one bit per cell); a sparse representation with the number of true cells, which stays the same. `Sparse` keeps two copies of a 4-byte index plus a 1-byte value per true cell, about 10 bytes against 1/8 byte per cell for Boolean, so it should take less memory below a density of about 1/80 -- here from about 1,408 features on.

![memory vs. width](representations_plots/width_memory.png)

Memory held by each representation (share of Boolean in parentheses; the last row is the unpacked bool matrix):

| representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Boolean | 13 KiB | 25 KiB | 62 KiB | 122 KiB | 244 KiB | 611 KiB | 1.2 MiB | 2.4 MiB |
| NList | 387 KiB (2997%) | 637 KiB (2586%) | 1.1 MiB (1846%) | 2.1 MiB (1754%) | 4.1 MiB (1703%) | 9.9 MiB (1652%) | 19.5 MiB (1634%) | 38.7 MiB (1624%) |
| PrePostNList | 505 KiB (3906%) | 758 KiB (3074%) | 1.2 MiB (2044%) | 2.2 MiB (1855%) | 4.2 MiB (1754%) | 10.0 MiB (1673%) | 19.6 MiB (1645%) | 38.9 MiB (1629%) |
| Sparse | 179 KiB (1383%) | 179 KiB (727%) | 180 KiB (292%) | 182 KiB (149%) | 186 KiB (76%) | 198 KiB (32%) | 218 KiB (18%) | 257 KiB (11%) |
| unpacked `X` | 98 KiB | 195 KiB | 488 KiB | 977 KiB | 1.9 MiB | 4.8 MiB | 9.5 MiB | 19.1 MiB |

800 `coverage()` calls for random rules of 1-4 conditions:

![coverage() time vs. width](representations_plots/width_coverage.png)

![fit time vs. width](representations_plots/width_fit.png)

Fit time in seconds:

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 0.30 | 0.24 | 0.59 | 0.39 | 0.75 | 2.78 | 8.00 | 16.07 |
| CN2 | NList | 0.34 (114%) | 0.23 (98%) | 0.49 (83%) | 0.27 (69%) | 0.37 (50%) | 0.86 (31%) | 1.19 (15%) | 1.83 (11%) |
| CN2 | PrePostNList | 0.34 (112%) | 0.23 (95%) | 0.49 (82%) | 0.27 (68%) | 0.36 (48%) | 0.85 (30%) | 1.17 (15%) | 1.82 (11%) |
| CN2 | Sparse | 0.37 (122%) | 0.23 (98%) | 0.45 (77%) | 0.25 (63%) | 0.35 (46%) | 0.83 (30%) | 1.13 (14%) | 1.74 (11%) |
| PFossil | Boolean | 0.01 | 0.01 | 0.01 | 0.02 | 0.04 | 0.08 | 0.18 | 0.34 |
| PFossil | NList | 0.01 (99%) | 0.01 (90%) | 0.01 (77%) | 0.01 (58%) | 0.02 (43%) | 0.03 (34%) | 0.05 (28%) | 0.09 (27%) |
| PFossil | PrePostNList | 0.01 (98%) | 0.01 (89%) | 0.01 (76%) | 0.01 (57%) | 0.02 (44%) | 0.03 (34%) | 0.05 (28%) | 0.09 (27%) |
| PFossil | Sparse | 0.01 (93%) | 0.01 (79%) | 0.01 (53%) | 0.01 (37%) | 0.01 (24%) | 0.01 (13%) | 0.02 (8%) | 0.02 (7%) |
| Pypper | Boolean | 0.06 | 0.02 | 0.03 | 0.05 | 0.05 | 0.72 | 0.72 | 1.10 |
| Pypper | NList | 0.04 (79%) | 0.02 (92%) | 0.02 (73%) | 0.03 (52%) | 0.02 (53%) | 0.15 (21%) | 0.13 (18%) | 0.18 (16%) |
| Pypper | PrePostNList | 0.04 (78%) | 0.02 (90%) | 0.02 (75%) | 0.03 (52%) | 0.02 (54%) | 0.15 (22%) | 0.13 (18%) | 0.18 (17%) |
| Pypper | Sparse | 0.04 (64%) | 0.01 (78%) | 0.02 (57%) | 0.02 (38%) | 0.01 (32%) | 0.11 (15%) | 0.08 (12%) | 0.10 (9%) |
| PyLORD | Boolean | 1.52 | 1.31 | 1.39 | 1.35 | 1.15 | 1.21 | 1.46 | 1.98 |
| PyLORD | NList | 1.66 (109%) | 1.52 (116%) | 1.92 (138%) | 1.45 (107%) | 1.18 (103%) | 1.07 (89%) | 1.10 (76%) | 1.22 (62%) |
| PyLORD | PrePostNList | 1.66 (109%) | 1.51 (115%) | 1.92 (138%) | 1.44 (107%) | 1.16 (101%) | 1.04 (87%) | 1.10 (75%) | 1.20 (60%) |
| PyLORD | Sparse | 1.63 (107%) | 1.37 (104%) | 1.40 (100%) | 1.28 (95%) | 1.03 (89%) | 0.93 (77%) | 0.96 (66%) | 1.02 (52%) |
| ENDER | Boolean | 0.46 | 0.53 | 0.75 | 1.05 | 1.69 | 3.42 | 6.69 | 14.85 |
| ENDER | NList | 0.48 (104%) | 0.50 (94%) | 0.56 (75%) | 0.62 (59%) | 0.67 (40%) | 0.83 (24%) | 1.05 (16%) | 1.59 (11%) |
| ENDER | PrePostNList | 0.47 (103%) | 0.50 (93%) | 0.56 (74%) | 0.61 (58%) | 0.67 (40%) | 0.79 (23%) | 1.04 (15%) | 1.59 (11%) |
| ENDER | Sparse | 0.52 (114%) | 0.54 (101%) | 0.56 (75%) | 0.58 (55%) | 0.63 (37%) | 0.76 (22%) | 0.98 (15%) | 1.50 (10%) |

Memory a fit costs in total -- what the representation holds plus the peak the fit allocates on top of it, caches the representation builds during the fit included (`tracemalloc`, a separate untimed pass):

| learner | representation | k=100 | k=200 | k=500 | k=1000 | k=2000 | k=5000 | k=10000 | k=20000 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| CN2 | Boolean | 1.1 MiB | 2.0 MiB | 5.0 MiB | 9.8 MiB | 19.7 MiB | 49.2 MiB | 97.7 MiB | 196.2 MiB |
| CN2 | NList | 1.5 MiB (140%) | 2.7 MiB (133%) | 4.1 MiB (82%) | 3.6 MiB (36%) | 7.2 MiB (37%) | 16.8 MiB (34%) | 30.6 MiB (31%) | 60.5 MiB (31%) |
| CN2 | PrePostNList | 1.6 MiB (151%) | 2.8 MiB (138%) | 4.2 MiB (84%) | 3.7 MiB (38%) | 7.3 MiB (37%) | 16.9 MiB (34%) | 30.7 MiB (31%) | 60.6 MiB (31%) |
| CN2 | Sparse | 522 KiB (48%) | 610 KiB (29%) | 1.0 MiB (21%) | 1.4 MiB (15%) | 3.1 MiB (16%) | 6.9 MiB (14%) | 11.1 MiB (11%) | 21.7 MiB (11%) |
| PFossil | Boolean | 1.0 MiB | 2.0 MiB | 4.9 MiB | 9.8 MiB | 19.6 MiB | 49.1 MiB | 97.6 MiB | 196.2 MiB |
| PFossil | NList | 1.4 MiB (141%) | 2.6 MiB (133%) | 4.0 MiB (81%) | 2.7 MiB (28%) | 5.0 MiB (26%) | 11.9 MiB (24%) | 22.8 MiB (23%) | 45.3 MiB (23%) |
| PFossil | PrePostNList | 1.6 MiB (152%) | 2.8 MiB (139%) | 4.1 MiB (84%) | 2.8 MiB (29%) | 5.1 MiB (26%) | 12.0 MiB (25%) | 22.9 MiB (23%) | 45.5 MiB (23%) |
| PFossil | Sparse | 476 KiB (45%) | 479 KiB (23%) | 521 KiB (10%) | 554 KiB (6%) | 925 KiB (5%) | 2.0 MiB (4%) | 3.2 MiB (3%) | 6.6 MiB (3%) |
| Pypper | Boolean | 759 KiB | 1.4 MiB | 3.5 MiB | 6.9 MiB | 13.8 MiB | 34.8 MiB | 69.0 MiB | 138.9 MiB |
| Pypper | NList | 1.1 MiB (155%) | 2.1 MiB (145%) | 4.0 MiB (114%) | 2.6 MiB (38%) | 4.9 MiB (35%) | 11.7 MiB (34%) | 22.4 MiB (32%) | 45.7 MiB (33%) |
| Pypper | PrePostNList | 1.3 MiB (170%) | 2.2 MiB (153%) | 4.1 MiB (117%) | 2.7 MiB (40%) | 5.0 MiB (36%) | 11.8 MiB (34%) | 22.5 MiB (33%) | 45.8 MiB (33%) |
| Pypper | Sparse | 399 KiB (53%) | 398 KiB (27%) | 442 KiB (12%) | 486 KiB (7%) | 803 KiB (6%) | 1.8 MiB (5%) | 2.9 MiB (4%) | 7.0 MiB (5%) |
| PyLORD | Boolean | 1.4 MiB | 1.4 MiB | 2.0 MiB | 2.4 MiB | 4.1 MiB | 10.1 MiB | 20.3 MiB | 40.5 MiB |
| PyLORD | NList | 1.8 MiB (127%) | 2.8 MiB (202%) | 4.2 MiB (213%) | 3.7 MiB (154%) | 5.6 MiB (138%) | 11.5 MiB (114%) | 21.4 MiB (106%) | 41.1 MiB (101%) |
| PyLORD | PrePostNList | 2.0 MiB (135%) | 2.9 MiB (211%) | 4.3 MiB (219%) | 3.8 MiB (159%) | 5.7 MiB (141%) | 11.7 MiB (115%) | 21.6 MiB (106%) | 41.2 MiB (102%) |
| PyLORD | Sparse | 1.6 MiB (108%) | 1.4 MiB (101%) | 1.7 MiB (85%) | 1.6 MiB (66%) | 1.5 MiB (37%) | 1.7 MiB (16%) | 1.9 MiB (10%) | 2.4 MiB (6%) |
| ENDER | Boolean | 509 KiB | 842 KiB | 1.8 MiB | 3.5 MiB | 6.8 MiB | 17.1 MiB | 33.5 MiB | 67.7 MiB |
| ENDER | NList | 902 KiB (177%) | 1.4 MiB (175%) | 2.9 MiB (159%) | 2.7 MiB (78%) | 4.9 MiB (71%) | 11.6 MiB (68%) | 22.0 MiB (66%) | 44.4 MiB (66%) |
| ENDER | PrePostNList | 1019 KiB (200%) | 1.6 MiB (189%) | 3.0 MiB (165%) | 2.8 MiB (82%) | 5.0 MiB (73%) | 11.7 MiB (68%) | 22.1 MiB (66%) | 44.5 MiB (66%) |
| ENDER | Sparse | 542 KiB (107%) | 563 KiB (67%) | 600 KiB (32%) | 609 KiB (17%) | 791 KiB (11%) | 1.7 MiB (10%) | 2.5 MiB (8%) | 5.7 MiB (8%) |

