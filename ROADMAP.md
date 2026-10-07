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
evaluation. Four implementations are not -- they turn the data into a
dense matrix (`data.X`) and work on its columns with numpy, so on N-list
data they rebuild that matrix and gain nothing:

- `ENDER` (and with it `Boomer`),
- `OptimalRuleBoosting`,
- `CPAR`,
- `LRI`.

External learners reading `data.X` is fine: the external tools need a
matrix anyway.

**Work.**

1. *Data preparation*: a `representation=` choice where data is prepared
   (`pyrulearn.experiments.runner.run_cv`, the demos, the loading
   helpers), with N-lists as the default. Build the N-list directly from
   the binarized matrix, without a `BooleanDataRepresentation` in between.
2. *The four learners*: score candidate conditions through the
   representation's (weighted) coverage functions instead of matrix
   columns. Measure the speed on every representation -- numpy scores all
   features in one vectorized step, so this could be slower on Boolean
   data, as with the rule searches.
3. *Tests*: extend the check that all representations give identical
   rules (so far the SeCo learners and PyLORD, `tests/test_representations.py`)
   to every native learner, so the principle is enforced, not just
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
