# Development roadmap

Plans for pyrulearn's development: what is intended for the next
releases, and why. The list of missing features for users is in the
README (*Not yet implemented*); the overhaul of the demos has its own
plan in [`examples/REVISION_PLAN.md`](examples/REVISION_PLAN.md).

## Count a numeric attribute's thresholds together

**Done for `BooleanDataRepresentation` (2026-10-07); still open for the
other representations.** The rule searches (`BeamSearch`, `HillClimbing`,
and `GainAscentHillClimbing`, which shares `HillClimbing`'s loop) used to
score a candidate condition by counting the covered positives and
negatives separately for every candidate -- one `refine_cover` +
`cover_counts` pass per threshold of a numeric attribute.

A numeric attribute's thresholds form a chain: `x >= t_k` covers exactly
the examples in intervals `k` and above, `x < t_k` those below (reversed,
the same shape). `BooleanDataRepresentation.chain_cover_counts` (approach
2 of the two once sketched here) counts a whole open run of one
attribute's chain in a single pass -- per-row count of satisfied
thresholds, histogrammed by class and summed from the top down -- instead
of one pass per threshold; `pyrulearn.learners.seco.score_children`
(shared by `BeamSearch`/`HillClimbing` until the next section) groups `rule.specialize`'s
children by chain and calls it, falling back to the original per-child
path for anything not groupable (a lone open threshold, a non-numeric
feature, or a representation without `chain_cover_counts` -- an opt-in
primitive, not a new required one, so N-list/PrePostNList/Sparse keep
working unchanged, just not yet sped up). A dead child (`tp == 0`) is now
caught from the count alone, before `refine_cover` ever builds its
handle -- *not done*: deferring handle construction further, for live
children that don't survive to the next round/beam slot, as the original
sketch's "build the covered set only for the candidates it keeps" also
proposed; nominal-attribute batching (the same idea for a multi-valued
attribute's value tests) is also still open.

**Correctness**: unweighted (plain `RemovalCovering` -- CN2/AQR/PFoil/
PFossil/Pypper/PyLORD's default) is bit-identical to the old path, every
count an exact integer; checked against the old path directly (CN2 on 5
numeric-heavy datasets: `segment`, `diabetes`, `sonar`, `ionosphere`,
`banknote-authentication`; PFoil/PFossil/Pypper on `sonar`), and the full
test suite (777 passed). Weighted (`WeightedCovering`, boosting's
per-round reweighting) can differ from the old path by float-reordering
noise (~1e-14 relative -- histogram+cumsum sums the same weights in a
different order than a direct masked sum, same issue any vectorized
reduction has) -- harmless to the actual numbers, but can rarely flip an
exact score/stopping-criterion tie to a different, equally valid rule.
Accepted as expected floating-point behavior, not a bug, after checking
it against a battery of real fits rather than assuming.

**Measured speedup** (one fit each, old vs. new, same machine): CN2 --
`segment` 1.49x, `diabetes` 2.24x, `sonar` 1.34x, `ionosphere` 1.17x,
`banknote-authentication` 1.54x; on `sonar`: PFoil 1.07x, PFossil 1.19x,
Pypper 1.12x, `WeightedCovering`+`AdditiveReweighting`+Laplace 1.10x,
Slipper 1.03x. Consistently positive, well short of the sketch's
upper-bound "~15x less counting" estimate -- that bound was for the
counting step alone; `refine_cover`'s handle construction for every live
child (unchanged here) is the rest of a round's cost, and these numbers
are the net of both.

Still open: N-list, PrePostNList and Sparse representations (each needs
its own one-pass counting primitive suited to its storage -- see the
0.3.0 plan below for why this can't be one shared implementation);
nominal-attribute batching.

**No longer on the search path (2026-10-08).** The next section's lazy
searches count every open child with one `batch_cover_counts` matmul
instead, so `chain_cover_counts` and `score_children`'s chain grouping
currently have no callers; kept, not deleted,
until it's decided whether they come back (e.g. for representations
without a matmul).

## Build only the children a search follows

**Done (2026-10-08).** `HillClimbing` (and `GainAscentHillClimbing`)
and `BeamSearch` used to call `rule.specialize`, which
builds *every* child of a node -- a `Rule`, its constraint closure
(`extend_closure`) and its mask -- and then move to one of them (or
`beam_width`). Profiled on `sonar` (1080 features), that was ~95% of a
search's time; the counting itself ~2%. Hence also why the earlier
matmul experiment (`_score_children_matmul`, below) barely moved the
needle: it sped up the 2%.

Now `count_open_children` gives `(tp, fp, fn, tn)` for every open
feature without building anything (one `batch_cover_counts` matmul on
`BooleanDataRepresentation`, one `refine_cover` + `cover_counts` per
feature on any other representation), the search scores and sorts from
those counts, and `materialize_child` builds -- closure, mask, handle --
only the child(ren) it actually follows, skipping one that turns out
contradictory exactly as `specialize` would. `BeamSearch` deduplicates
children reached from different parents by a bitmask of their condition
features (Rule equality is its condition set) before building any.
Tie-breaking is unchanged (stable sorts over `specialize`'s feature
order).

**Measured** (old eager vs. new lazy, one fit each, same machine; every
model identical except as noted):

| dataset (rows x features) | PFoil | PFossil | Pypper | Slipper | CN2 | AQR | PyLORD |
|---|--:|--:|--:|--:|--:|--:|--:|
| diabetes (768 x 112) | 5.3x | 4.1x | 2.8x | 4.0x | 7.4x | 4.5x | 3.1x |
| sonar (208 x 1080) | 19x | 17x | 15x | 22x | 42x | 4.1x | 10x |
| kr-vs-kp (3196 x 76) | 3.0x | 2.2x | 1.5x | 2.7x | 2.7x | 3.0x | 3.2x |
| spambase (4601 x 998) | 1.6x | 1.7x | 1.4x | 1.7x | -- | -- | -- |

The gain is largest with many features and few rows: building a child
cost the same per feature at every node, while the remaining counting
scales with the rows -- on `spambase` counting is now most of the time.
Full `demos/ripper_comparison.py --full` re-run (44 datasets, 10-fold):
Pypper's 440 fold results identical to the stored run; Slipper's differ
in 13 of 452, consistent with the weighted float-reordering noise
described in the previous section (Slipper's counts are weighted sums; a
matmul sums them in a different order) -- not yet confirmed, since the
stored run also predates `chain_cover_counts`.

**`CPAR` on this machinery** (`pyrulearn.learners.cpar`). `CPAR` used to
score ``(wp * cov) @ Xf`` over every feature not yet in the rule, on a
dense matrix; that version is now `DenseCPAR`. The modular `CPAR` grows
through this machinery instead -- first built (as `PropagatingCPAR`) to
test whether constraint propagation (fewer open features) pays for
itself there, and made the main `CPAR` on 2026-10-09, following the
package's components-first approach (README, *Native learning
algorithms*), with the growing heuristic, the covering strategy and the
combiner as parameters and `DecomposingLearner` for multi-class. Same
models as `DenseCPAR` (an implied condition has FOIL gain 0, below
`min_gain`, so `DenseCPAR` never picks one; checked on `diabetes`,
`sonar`, `kr-vs-kp` and in `tests/test_cpar.py`, also with exchanged
components). Built eagerly through `specialize` first, it was
3.6x (`diabetes`) to 350x (`sonar`) slower than `DenseCPAR`; lazy, 6-34x
faster than that, but still 4x (`diabetes`: 1.54s vs. 0.40s), 14x
(`sonar`: 0.82s vs. 0.06s) and 8x (`kr-vs-kp`: 1.07s vs. 0.14s) slower
than `DenseCPAR`. Profiled on `sonar`, the gap was three overheads, none
intrinsic to propagation: per-candidate Python scoring ~55%,
`batch_cover_counts`'s bool-to-float slice ~22%, recomputing the parent
closure ~21% (cProfile's estimates; see below for what fixing each one
actually measured). After all three: 2.4x (`diabetes`: 0.92s vs.
0.38s), 2.8x (`sonar`: 0.16s vs. 0.06s), 4.4x (`kr-vs-kp`: 0.56s vs.
0.13s).

Still open: AQR gains far less than CN2 on `sonar` (4x vs. 42x), so
something else dominates it -- not yet profiled.

**Closures passed down (done, 2026-10-08).** Every expanded node used to
recompute its rule's constraint closure from scratch (`parent_closure`),
although `materialize_child` had just computed it to build the child's
mask. It now returns it, and the searches carry it alongside mask and
cover handle; only a seed's is still computed. Same models, but no
measurable speedup (within +-10% everywhere) -- the profile's ~21% for
it was cProfile's per-call overhead on `propagate`'s many small Python
calls, not real time.

**Covered rows only (done, 2026-10-08).** `batch_cover_counts`
multiplied the weights of *all* rows against the open columns, copying
(and converting to float) an `n_rows x n_open` block per call, though
uncovered rows only contribute zeros. It now takes only the covered rows
(`X[covered][:, open]`, one copy), a block that shrinks with every
condition. Rejected instead: a cached float copy of `X` (8x the boolean
matrix's memory). Same models; best of 3 vs. the previous version, same
machine: `spambase` (4601 rows) 1.9-6x (CN2 23.9s -> 4.0s, PyLORD 277s
-> 145s), `kr-vs-kp` 1.2-2.3x, `sonar` 1.2-2.3x (modular `CPAR`
2.3x, now 2.8x behind `DenseCPAR`), `diabetes` ~1.1x.

**Batch scoring (done, 2026-10-08).** Each candidate used to be scored by
its own Python-level `heuristic.score` call. Every heuristic now also has
`batch_score`, which scores arrays of stats at once: the base class's
default loops over `score` (so a heuristic only *needs* `score`), every
built-in overrides it with the same formula in numpy. Two versions rather
than one array-safe `score` because numpy is slow for a single value,
and `score` stays the fast path for that; `tests/test_heuristics.py`
checks they agree (bit-identical for the arithmetic heuristics, to
rounding for the log-based ones and `Correlation`/`ChiSquare`, whose
scalar versions multiply exact Python ints). The searches score a node's
children with one `batch_score` call and rank them with one `lexsort`,
and compute everything they compare against those scores (a parent's
own score, the running best, the optimistic bound) through
`batch_score` too -- mixing the two could let a child with its parent's
exact stats win by a rounding difference. Same models in every case
measured; best of 3 fits, old vs. new, same machine: `sonar` (1080
features) 1.5-2.3x (modular `CPAR` 1.84x, now 7x behind `DenseCPAR`
instead of 14x), `diabetes` (112) 1.2-1.6x, `kr-vs-kp` (76) ~1.0x --
the gain follows the number of open features per node. AQR unchanged
(~1.0x everywhere), so its time is elsewhere.

## ENDER on the representation's primitives

**Done (2026-10-09).** `ENDER` (and `Boomer`) grow a rule the way the
lazy searches do: walk a cover handle, sum every open condition's
impurity terms with one `DataRepresentation.batch_cover_sums` call per
step (generic: one refinement per condition; `BooleanDataRepresentation`:
one matrix product over the covered rows with a nonzero term), and build
only the condition added, with constraint propagation. The impurity
criterion is a component (`ImpurityCriterion`: `ConstantStep`,
`Gradient`, `GradientBoosting`, `Simultaneous`, `Newton`), like the loss.
The dense implementation stays as `DenseENDER`/`DenseBoomer`. Same
models (tested on 24 configurations, and on every representation).

**Measured** (dense vs. modular on Boolean, 100 rules, best of 3, same
machine):

| dataset (rows x features) | constant step (default, 25% subsample) | Newton, all rows |
|---|--:|--:|
| diabetes (768 x 112) | 1.1x | 1.1x |
| sonar (208 x 1080) | 1.5x | 2.9x |
| kr-vs-kp (3196 x 76) | 0.9x | 4.4x |
| spambase (4601 x 998) | 3.3x (15.1s vs. 4.6s) | 7.7x (45.2s vs. 5.9s) |

With a subsample, the rows outside it carry zero terms and are skipped
(spambase's default went from 27.6x to 3.3x with that). Newton steps on
all rows are the expensive case: every step copies the covered rows of
the open conditions, where `DenseENDER` multiplies its precomputed float
matrix.

**Rounding could add conditions that change nothing -- fixed, in both
versions and in the searches.** A condition that drops no row carrying
weight has exactly the rule's own impurity (or tp/fp), but summing a
different set of rows -- the covered rows only, as `batch_cover_counts`
and `batch_cover_sums` do -- can round it a few ulp lower, and a search
then takes it for a strict improvement. Seen in `ENDER` on `kr-vs-kp`
and, more surprisingly, in `DenseENDER` on `spambase`, whose sums always
run over all rows: a matrix product can round different output columns
differently, and `capital_run_length_total < 693.5` was added (by 4e-15)
to a rule already holding a stricter threshold on the same attribute.
Now every search counts the contributing rows exactly (integers): a
condition keeping all of them gets exactly the rule's own score
(`count_open_children`'s `same`, `batch_cover_counts`' `keeps_all`, an
indicator column in `ENDER`'s sums). Unweighted counts were always exact;
the SeCo learners and `CPAR` learned the same models as before the fix,
a few percent slower.

Still open: batched `batch_cover_counts`/`batch_cover_sums` for sparse
data and N-lists -- sparse: slice the covered rows out of the CSR it
already keeps and `bincount` their column indices with the rows' values
(cost: the nonzeros of the covered rows); N-list: group the covered rows
by the tree node their path ends at (its path mask is the row's whole
feature set, so the depth problem that broke both `chain_cover_counts`
attempts doesn't arise), one `bincount` per node and one product over
the nodes' path bits (cost: distinct row patterns, not rows), measured
against plain horizontal counting through a CSR built from `_row_feats`.
Weighted counts on those representations get no `same` marks yet. A
dense vs. modular demo.

## Several rules per search: CPAR as a SeCo configuration

**Done (2026-10-09).** A `RuleSearch` can yield several rules
(`search_all`; default: `search`'s one), `SingleRuleLearner.learn_rules`
post-processes each, and `SeCo`'s covering loop accepts them in turn --
all judged on the scope they were found in, each updating the covering.
`GainAscentHillClimbing` got `min_gain=` (a gain of at least this,
instead of any positive gain) and `branch_similarity=` (also follow
every child within that factor of the best gain, as a copy grown on the
same way: CPAR's search). With those, `CPAR` is a `SeCo` preset --
`FoilGain`, the branching gain ascent, `WeightedCovering`, a top-k
combiner -- with no search or covering code of its own, and the
branching is available to every SeCo learner (e.g. `PFoil` with copies).
Same models as before (`CPAR` == `DenseCPAR` in `tests/test_cpar.py`;
the other SeCo learners unchanged). `max_rounds` now caps rules per
class, as `WeightedCovering` counts, not searches; `DenseCPAR` counts the
same way.

**First experiment** (3-fold CV, the 10 small datasets of the RIPPER
quick run): no accuracy effect anywhere -- every variant within about a
point (0.830-0.841 mean accuracy), well inside the noise of so few
folds. Branching mostly adds rules: `PFoil` 28 -> 67 (similarity 0.99)
-> 201 (0.9), since removal covering still accepts copies whose
examples the first rule already removed; in `CPAR`, branching at 0.99
rarely triggers (126 rules without, 135 with) and `CPAR` without it was
no worse (0.841 vs. 0.837); `Pypper` hardly changes, grow-then-prune
absorbs the copies (one outlier: `tic-tac-toe` 0.932 -> 0.981). So: a
clean generalization `CPAR` needs, no reason yet to make branching a
default anywhere -- a 10-fold run over many datasets would be the real
test (a demo).

Open: that demo; a `BeamSearch` counterpart (return the final beam
rather than its best rule).

## 0.3.0: every native learner on any data representation

**Principle.** Learners are written against the `DataRepresentation`
interface and run on any representation; which representation is used is
decided when the data is prepared, not inside a learner. This was the
design from the start, but some later learners broke it.

**Why now.** The representations demo (`demos/representations.py`,
2026-09-30) found that N-lists take about half of Boolean's fit time for
the SeCo learners and PyLORD, and the gap grows with the data: at 32,000
examples of `adult`, CN2 finishes only on N-lists. `PrePostNList` was
only marginally faster than plain N-lists (51% vs. 53% of Boolean's time)
while taking longer to build, so plain `NListRepresentation` is the
intended default.

**Status.** Everything that goes through a rule's coverage
(`covers_data_packed`, which forwards to the data's own `coverage`) is
already representation-independent: the SeCo framework with CN2, AQR,
PFoil, PFossil and Pypper, PyLORD, the rule models, pruning and
evaluation. Four implementations turn the data into a dense matrix
(`data.X`) and work on its columns with numpy instead:

- `OptimalRuleBoosting`,
- `LRI`.

`CPAR` and `ENDER` (with `Boomer`) were on this list; both now run on
every representation, built from the SeCo search primitives, and the
dense-matrix versions stay on purpose as `DenseCPAR` (same models, 2-4x
faster) and `DenseENDER`/`DenseBoomer` (same models, 1-8x faster) -- see
"Build only the children a search follows" and "ENDER on the
representation's primitives" below. (`ENDER` had been investigated on
2026-10-07/08 and kept dense: a representation-generic rewrite measured
10-500x slower -- before the lazy search, batch counting and the
primitives below existed.)

External learners reading `data.X` is fine: the external tools need a
matrix anyway.

**Shared mechanism (2026-10-08).** A learner that commits to one (or
more) representations' own storage instead of the universal
`coverage`/`initial_cover`/`refine_cover` interface declares
`NativeRuleLearner.NATIVE_REPRESENTATIONS` (`None`: any representation
works as given, unconverted -- the default, true of everything above
except the four listed here); its native fit method calls
`self.ensure_representation(data, self.max_auto_convert_cells,
purpose=...)` to convert anything else, bounded by
`max_auto_convert_cells` (raises past it -- no silent unbounded-memory
rebuild). Ties, when a learner's native set names more than one
representation and the data given matches none of them, break toward
`REPRESENTATION_PREFERENCE_ORDER` (`NListRepresentation` first, then
`BooleanDataRepresentation`, then `SparseDataRepresentation`) -- so
absent a more specific reason, N-list is this library's default
conversion target, the same preference item 1 below gives data
*preparation*. `ENDER`/`Boomer` (`NATIVE_REPRESENTATIONS =
(BooleanDataRepresentation,)`) and `pyrulearn.learners.associative.CARMiner`
(`(NListRepresentation,)`, converting in the *opposite* direction, for
mining's speed rather than `_grow`'s correctness) are the two current
users; `CARMiner`'s own `ensure_nlist` (which used to accept any other
representation as-is, just slower, since `generate_cars` only needs the
universal `coverage()`) was folded into this shared mechanism too, which
means that leniency is gone -- a `SparseDataRepresentation` now also
gets converted rather than used directly, since the mechanism doesn't
distinguish "needed" from "just much faster" and CARMiner no longer has
its own bespoke gate to make that distinction in.

**Work**, for the three remaining learners (`ENDER`/`Boomer` excluded,
see above -- expect them to need the same investigation before assuming
the work below is worth doing for them too):

1. *Data preparation*: a `representation=` choice where data is prepared
   (`pyrulearn.experiments.runner.run_cv`, the demos, the loading
   helpers), with N-lists as the default. **The primitive this needs is
   done (2026-10-08): `pyrulearn.data.io.encode`** (`binarize`'s
   sparse-intermediate cousin) builds whichever `DataRepresentation` is
   asked for -- default `NListRepresentation` -- directly from a
   `scipy.sparse` matrix, never forming the dense `(n_samples,
   n_features)` array `binarize` always does, even transiently;
   `NListRepresentation.__init__` now also accepts a `scipy.sparse`
   matrix directly (reading rows from its CSR structure instead of
   `np.flatnonzero` on a dense row -- `BooleanDataRepresentation`/
   `SparseDataRepresentation` already did). **Not done yet, on purpose,
   done step by step**: `run_cv` and every demo still call `binarize` +
   `BooleanDataRepresentation(...)` by hand, unchanged -- migrating them
   onto `encode` is the deliberate next step, kept separate so each
   demo's migration can be checked (identical rules/accuracy, not just
   "didn't crash") on its own rather than in one sweeping change.
2. *The three learners*: score candidate conditions through the
   representation's (weighted) coverage functions instead of matrix
   columns. Measure the speed on every representation -- numpy scores all
   features in one vectorized step, so this could be slower on Boolean
   data, as with the rule searches.
3. *Tests*: extend the check that all representations give identical
   rules (so far CN2/PFoil/PFossil/AQR, PyLORD, and `ENDER` (2026-10-08,
   checking its deliberate dense-conversion path rather than true
   representation independence), `tests/test_representations.py`) to
   every native learner, so the principle is enforced, not just
   intended.
4. *Optional*: build the N-list index faster (it is a Python loop per row:
   about 7 s for 32,000 rows of `adult`) and store feature indices as
   32-bit integers to halve its memory.

## Other planned work

- **Measure the SeCo-search-vs-ENDER speed gap on `demos/covering_boosting.py`
  itself, now that the Boolean-representation threshold-counting speedup
  above has landed (2026-10-07).** Not done yet -- the chain-counting
  work was checked against CN2/PFoil/PFossil/Pypper on other datasets, not
  by rerunning this demo's own `WeightedCovering`+WRAcc configuration,
  so the gap this note describes hasn't actually been re-measured.
  `demos/covering_boosting.py`
  (2026-10-06/07) found `WeightedCovering` configs (plain `SeCo`,
  `BeamSearch(beam_width=5)`) taking tens of seconds per fit
  (`Weighted-Add+WRAcc`: 86.55s) against boosting's sub-3s
  (`Boomer@100`: 2.48s) on the same datasets. The likely reason:
  `ENDER`/`Boomer`'s `_grow` scores every candidate feature in one
  vectorized `(t * c).T @ Xf` matrix multiply per round, while
  `BeamSearch`/`HillClimbing` still count each candidate condition with
  its own pass (the "High priority" item above) -- not a difference
  between weighted and removal covering, or a beam-width effect (both
  use the same beam width here). But `ENDER`'s vectorization only works
  because it commits to a dense `data.X` matrix, exactly what the
  0.3.0 plan above calls out as *not* representation-independent for
  `ENDER` itself. So two numbers worth comparing once the chain-counting
  speedup lands: how much the SeCo searches close the gap, and
  (separately, from the 0.3.0 work) how much `ENDER` slows down once
  it's made representation-independent instead of assuming a matrix --
  i.e. whether the two converge from opposite directions.
- **`CoverageDifference` is ~3x faster than `Accuracy` for the same
  ranking** (measured 2026-10-06: 1,000,000 `.score()` calls each,
  0.07s vs. 0.21s) -- confirms the docstrings' claim that
  `Accuracy = (CoverageDifference + n_neg) / (n_pos + n_neg)` exactly
  (checked over 200,000 random stats, zero floating-point difference).
  For 0.3.0, check every built-in algorithm's default heuristic for a
  hot-path use of `Accuracy` swappable to `CoverageDifference` without
  changing which rule wins. None currently defaults to `Accuracy`
  (grepped) -- start with `Pypper` specifically, since it was asked
  about directly: its defaults are `Precision` (pruning and the IREP
  stopping criterion) and `FoilGain` (growing), neither with the same
  isometrics as `CoverageDifference`, so this needs checking whether a
  cheaper-but-equivalent swap exists there too, not assuming the same
  substitution applies.
- **Inverted heuristics** (Stecher, Janssen & Fürnkranz): a refinement
  evaluated in the coverage space of its parent rule instead of the empty
  rule. Implement in `pyrulearn.heuristics`, then add to the
  `heuristic_isometrics` and `heuristic_comparison` demos. (Also listed in
  the README.)
- **README**: subsections under *Native learning algorithms* for the
  learners that are only mentioned in tables so far -- CBA, CMAR, IDS and
  RuleFit.
- **Java LORD on large data**: it timed out on every fold of `adult` and
  `connect-4` in the SeCo learners comparison. The likely reason is the
  number of binary features (about 250-280 with negation features);
  check with one fit without negation features.
- **`pyrulearn/learners/seco.py`**: the module docstring still says that
  rule-set optimization is not implemented, although `SeCo` has
  `optimization=` and Pypper uses `ReplaceReviseOptimization`.
- **Error-correcting output codes** (Dietterich & Bakiri 1995) as a
  further multi-class decomposition, next to one-vs-rest, ordered
  one-vs-rest and pairwise: each class gets a code word, one binary
  model is learned per code bit, and an example is assigned the class
  whose code word is nearest to the models' predictions. It would fit as
  another model type of `DecomposingLearner` (`fit(data, model=...)`) and
  as another method in the multi-class decomposition demo.
- **Demo plots**: on logarithmic axes, the plain-number tick labels also
  label the minor ticks, which overlap where the axis spans several
  decades (e.g. `demos/representations_plots/size_index_build.png`).

## Design decisions and known behaviour

Not to-dos: choices made deliberately, with the behaviour they imply, so
they aren't reopened by accident.

- **ENDER-style matrix-multiply counting, tried for `BooleanDataRepresentation`
  and parked, not adopted (2026-10-07).** Following up on the numeric-
  threshold-counting item above, `BooleanDataRepresentation.batch_cover_counts`
  and an experimental `pyrulearn.learners.seco._score_children_matmul`
  score *every* open feature of a round in one matrix multiply (`pos_w
  @ X[:, features]`), the way `ENDER`'s `_grow` does -- more general than
  `chain_cover_counts` (works across attributes and on nominal features
  too, not just one numeric chain), and simpler code (no monotonic-chain
  bookkeeping, no missing-value edge case -- a bug in the chain version's
  first draft, where `<t` was wrongly derived as `>=t`'s complement
  without accounting for `MissingStrategy.NEVER_COVERS` rows where
  neither holds).
  - **Where it wins cleanly**: `HillClimbing` (one lineage, one matmul
    per round, the clean ENDER-shaped case) -- consistently at or above
    `chain_cover_counts`'s speed (PFoil 1.07x, PFossil 1.19x, Pypper
    1.12x, Slipper 1.03x on `sonar`; PFossil 1.07x on `spambase`), and far
    ahead on nominal-heavy data regardless of search (`kr-vs-kp`: 1.74x
    over the old path vs. chain's 1.12x, since chain-counting can't help
    nominal features at all).
  - **Where it regresses**: `BeamSearch` with a real beam (>1) on a
    feature-rich dataset. Each of the (up to `beam_width`) surviving
    parents per round pays its own full-open-feature matmul, with no
    sharing across parents -- cost scales roughly with `beam_width x
    open_feature_count`, unlike `chain_cover_counts`'s smaller,
    per-attribute batches. Measured on `spambase` (its widest feature
    count): `beam_width=1` 1.08x (matmul wins), `beam_width=3` 0.98x
    (a wash), `beam_width=5` -- CN2's actual default -- 0.76x (matmul
    24% *slower*), `beam_width=10` 0.83x. `PyLORD`'s default
    `BeamSearch(beam_width=1)` lands in the good case; CN2/AQR/PFoil's
    beam variants, at their real default widths on feature-rich data,
    don't.
  - **GPU**: considered and rejected without prototyping, on the
    numbers already in hand. The matmul is a vector-times-matrix
    (GEMV, memory-bandwidth-bound, not the large compute-bound GEMM a
    GPU's parallelism actually pays off on), called once per beam
    member per round and synchronized back to Python immediately
    after (`tp == 0` and heuristic scoring need the result right away)
    -- exactly the per-call-overhead-dominated pattern the beam-width
    finding above already shows losing to CPU's smaller, more frequent
    calls. Would need batching across beam members into one bigger
    matrix-matrix product *and* keeping data device-resident across a
    whole search to plausibly pay off -- a real redesign, not a backend
    swap, not attempted.
  - **`batch_cover_counts`'s own fancy-indexing cost, found and then
    un-fixed (2026-10-08).** It slices `self.X[:, feature_indices]`
    before the matmul -- `feature_indices` is a list/array, so that's
    fancy indexing, which numpy always copies. Measured in isolation:
    26x slower than matmul-ing against the *whole* `self.X` and
    selecting `feature_indices` out of the small `(n_features,)` result
    afterward instead -- the same "score everything, mask the result"
    trick `pyrulearn.learners.cpar.DenseCPAR`'s (then `CPAR`'s) own native code already uses
    (see the CPAR entry below). Tried as a fix, measured *worse* in
    every real case re-checked (`spambase`, CN2/`BeamSearch` at
    `beam_width` 1/3/5/10, and `PFossil`/`HillClimbing` -- previously
    matmul's cleanest win at 1.07x, now 0.90x): the isolated benchmark
    used a case with ~98% of features still open, which is *not*
    representative of `BeamSearch`/`HillClimbing`'s own open set -- a
    `SeCo` search's open set shrinks fast, not slowly: `Rule.specialize`'s
    constraint-aware closure means fixing *one* numeric threshold forces
    every other threshold (and negation) of that same attribute out of
    the open set in one step, so most calls happen far narrower than
    98% open, where slicing-then-matmul's smaller FLOP count beats
    matmul-then-select's constant full-width cost despite the slice's
    own overhead. Reverted back to slicing; the 26x number is real and
    not wrong, just measured on a case unrepresentative of *this*
    caller, generalized from without checking the real workload first
    -- the exact mistake the "verify, don't assume" discipline elsewhere
    in this file is supposed to catch, caught here one step too late
    (after proposing the fix, not before).

    Checked directly *why* this doesn't contradict `CPAR`'s (now `DenseCPAR`'s) own native
    code using the identical "score everything, mask after" trick
    successfully (prompted by a direct question, not found
    unprompted): on `spambase` (n=4601, k=998), `CPAR`'s real
    full-matrix `_grow` is **15x faster** than a slice-first variant of
    the exact same search (102ms vs. 1547ms, identical 27 bodies) --
    the opposite result from `BeamSearch`. The difference isn't
    single-lineage-vs-branching or small-n-vs-large-n (both were
    candidate explanations, both wrong) -- it's *how fast the open set
    shrinks*. `CPAR` works directly on `data.X` with no constraint
    propagation at all: each condition removes exactly the one feature
    just used from consideration, nothing else, so with `max_length`
    typically ~5, its open set stays at roughly `k - 5` -- essentially
    full width -- for the entire search. That's genuinely the ~98%-open
    regime the isolated benchmark measured; it just isn't `SeCo`'s
    regime, where one condition can remove a dozen+ features at once.
    Same trick, opposite verdict, because the two callers' open sets
    behave completely differently -- not a property of the trick itself.
  - **Decision**: keep `chain_cover_counts` as `BooleanDataRepresentation`'s
    shipped default (safe across beam widths); keep `batch_cover_counts`/
    `_score_children_matmul` in the codebase as a validated-but-not-wired-in
    alternative, not deleted -- revisit if a beam-width/feature-count-aware
    dispatch between the two (or restricting matmul to `HillClimbing`-based
    learners specifically) turns out to be worth the complexity.
    **Superseded (2026-10-08)**: all searches now count through
    `batch_cover_counts` after all, once they stopped building every
    child eagerly -- see "Build only the children a search follows".
    These measurements were taken on top of that eager construction.
- **A `chain_cover_counts` for `NListRepresentation`: first attempt was
  wrong, reverted before being committed (2026-10-07).** Tried porting
  the Boolean trick directly: scan the loosest (shallowest) open
  threshold's own tree nodes once, then check the stricter thresholds'
  bits against those same nodes' `path_mask`. Wrong, because the
  PPC-tree orders features most-frequent-first and a monotonic chain's
  loosest threshold is, by construction, the *most frequent* member --
  so it sits *shallower* than the stricter ones, and their bits are only
  added to the path *further down*, past that node, not yet present
  there. Silent failure mode, not a crash: the undercounted thresholds
  came back `tp == 0` and were dropped into the "dead" set before ever
  reaching the per-child brute-force check, so the check -- which only
  verified *surviving* children, never *why something didn't survive* --
  reported zero mismatches on a run that was actually dropping most of
  a chain's thresholds (caught by comparing old-vs-new *full fits*
  instead, which disagreed). Scanning the *deepest* (strictest) member
  instead doesn't fix it either: its own node list only has rows
  satisfying *every* member of the chain, missing everything that
  satisfies some but not all of it -- correct for nothing shallower than
  itself. Reverted rather than shipped; the representation's existing
  per-threshold `refine_cover` path is untouched.

  **Second attempt, correct but not a win, also reverted (same day).**
  Intersect the one scan's candidate rows against each stricter
  threshold's already-precomputed, unmasked `_row_idx[feature]` directly
  (no tree traversal, no dense matrix) -- `{parent ∧ f_k} = {parent ∧
  f_1} ∩ row_idx[f_k]` holds exactly, since `f_k` implies `f_1`. Also
  had to mirror `refine_cover`'s *own* two branches for `f_1` itself
  (cheap filter vs. re-anchor scan), not just assume the expensive one
  -- a second, narrower instance of the same mistake: `f_1` is only
  safe to re-anchor-scan when it's deeper than every condition already
  in the rule, same as any other feature. Verified correct this time
  with a stricter check than the first attempt's -- explicitly
  confirming every *dropped* ("dead") candidate was brute-force-checked
  too, not just survivors, the exact gap that let the first version's
  bug through unnoticed -- across 6 datasets, 1844 search-node calls,
  zero mismatches either way; full test suite (777 passed).

  Despite that, timing (CN2, one fit each, old vs. new) was mostly a
  regression: `segment` 0.73x, `diabetes` 0.62x, `ionosphere` 0.80x,
  `banknote-authentication` 0.68x, `sonar` 1.07x (even), `kr-vs-kp`
  1.60x (the one clear win, nominal-heavy -- same pattern as
  `BooleanDataRepresentation`'s matmul vs. chain finding above). Likely
  cause: `np.isin(candidate_rows, row_idx[f])` for each of the `m - 1`
  remaining thresholds costs more than expected -- its cost scales with
  both array sizes, re-sorting `row_idx[f]` fresh on every call, which
  on a numeric-heavy dataset with many samples can exceed what it saves
  versus the tree re-scans it replaces. A pre-sorted `row_idx` (sorted
  once, cached, e.g. at `_build` time) plus `np.searchsorted` instead of
  `np.isin` might close the gap -- not tried. Reverted rather than left
  as an always-on regression (`_score_children` dispatches on whether
  `chain_cover_counts` exists at all, with no beam-width/feature-count
  gating the way the matmul decision got -- so, unlike that one, there
  was no way to "keep it but not wire it in" without also changing the
  dispatch).
- **How rules are annotated** (decided 2026-10-01). Every rule is
  annotated with the training examples its body covers, independently of
  the other rules (`annotate_rules`, `annotate_default_rule`) -- so a
  rule's counts mean the same in every model it belongs to, and don't
  change when a rule set is reordered or turned into a decision list. The
  default rule's body is empty, so it is counted over all training
  examples (Weka, by contrast, prints JRip's rules and default over the
  examples that reach them). Two alternatives were considered and not
  taken: counting each rule over the examples it actually decides in its
  model (more informative for decision lists, but model-dependent, so
  every conversion or reordering would have to recount -- needing the
  data), and counting it over the examples it was learned from (which
  would make the same rule set annotated differently depending on the
  order it was learned in).
- **The pairwise weighted vote does poorly** -- a result of the
  multi-class decomposition demo, explained in its report. It weights a
  pair model's vote by the Laplace estimate of the deciding rule; most
  pair models decide most examples by their default rule, whose counts
  (over all of its pair's examples) make its weight just its class's
  share of the pair -- about 0.5 for a balanced pair, below 0.5 when the
  default is the pair's smaller class. Counting the default over the
  examples it decides brings the weighted vote up to the plain vote, but
  not above (the demo's final check). Kept as it is, following the
  decision above.
- **The Prolog rendering of a decision list's default rule isn't
  faithful standalone Prolog** (noted 2026-10-03, building the
  tutorial). `died(X) :- true.` reads as "died holds for every X", but
  the model only means it for rows no earlier rule matched -- the
  faithful body would be `died(X) :- not(survived_rule_1(X)), not(...),
  ...`, and in general every non-first rule of a decision list would
  need the same growing prefix of negated earlier rules to be exactly
  right, which would make longer lists unreadable. A cleaner fix exists
  -- give every rule (default included) a shared head like `class(X,
  <target>)` and a trailing cut (`!`), which is faithful first-match-wins
  Prolog without any explicit negation -- but it only reads correctly if
  each rule's printed stats are also its cut-aware decided coverage, not
  its independent one, which is the "how rules are annotated" choice
  above, deliberately not reopened. No change: the current rendering
  stays informal/approximate for a decision list's non-first rules, most
  visibly its default.

## Open problems from the multi-class demo

- **Pypper with the larger class as a pair's target** learns no rules for
  many pairs (128 of 210 on `primary-tumor`), and every rule-less model
  votes for its pair's smaller class -- so the plain vote predicts the
  rarest classes and collapses (accuracy 0.003). Why Pypper learns
  nothing there is open.

## Release checklist

- The version number is in four places: `pyproject.toml`,
  `pyrulearn/__init__.py` (`__version__`), `CITATION.cff`, and the BibTeX
  entry in the README's *How to cite*. Grep for the old number before
  tagging.
- `CHANGELOG.md`: turn *Unreleased* into `## X.Y.Z (date)`.
- Tag `vX.Y.Z`.
