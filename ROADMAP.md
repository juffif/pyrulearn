# Development roadmap

Plans for pyrulearn's development: what is intended for the next
releases, and why. The list of missing features for users is in the
README (*Not yet implemented*); the overhaul of the demos has its own
plan in [`examples/REVISION_PLAN.md`](examples/REVISION_PLAN.md).

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
