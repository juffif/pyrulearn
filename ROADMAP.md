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
of one pass per threshold; `pyrulearn.learners.seco._score_children`
(shared by `BeamSearch`/`HillClimbing`) groups `rule.specialize`'s
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
nominal-attribute batching; deferring handle construction past scoring.

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

- `ENDER` (and with it `Boomer`) -- **investigated and decided to stay
  this way (2026-10-07/08), not an open item any more.** A
  representation-generic rewrite (`_grow` scoring through
  `initial_cover`/`refine_cover` instead of matrix columns, the way
  `WeightedCovering` does) was prototyped and measured 10-500x slower
  even before batching, with no plausible fix -- see the "Design
  decisions" section below for the numbers and why N-list's own
  per-search caching doesn't transfer to boosting's per-round-changing
  gradient. `ENDER`'s class docstring ("Data representation") now
  documents this directly; still correct on every representation (just
  always at dense-matrix cost), still tested for identical rules across
  them.
- `OptimalRuleBoosting`,
- `CPAR`,
- `LRI`.

External learners reading `data.X` is fine: the external tools need a
matrix anyway.

**Work**, for the three remaining learners (`ENDER`/`Boomer` excluded,
see above -- expect them to need the same investigation before assuming
the work below is worth doing for them too):

1. *Data preparation*: a `representation=` choice where data is prepared
   (`pyrulearn.experiments.runner.run_cv`, the demos, the loading
   helpers), with N-lists as the default. Build the N-list directly from
   the binarized matrix, without a `BooleanDataRepresentation` in between.
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
  - **Decision**: keep `chain_cover_counts` as `BooleanDataRepresentation`'s
    shipped default (safe across beam widths); keep `batch_cover_counts`/
    `_score_children_matmul` in the codebase as a validated-but-not-wired-in
    alternative, not deleted -- revisit if a beam-width/feature-count-aware
    dispatch between the two (or restricting matmul to `HillClimbing`-based
    learners specifically) turns out to be worth the complexity.
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
