# Changelog

All notable changes to pyrulearn are listed here, newest first. The project
is in early development (alpha): until 1.0, minor versions may change the API.

## Unreleased

### Changed (may break code: statistics redesigned)

- Only rules store measurements: a rule's training statistics,
  `rule.stats()`, set once where it is produced (`fit`, an importer's
  `data=`, `annotate_rules`, or the new `SingleRule.set_stats(data)`).
  They are frozen: they are part of the model (combiners score from them,
  printing shows them), so setting them again raises instead of silently
  changing what the model predicts. `SingleRule.reset_stats(data)`,
  `annotate_rules(..., reset=True)` and `set_stats_from_counts(..., reset=True)`
  replace them deliberately. `annotate_rules(..., copy=True)` annotates
  fresh copies instead; the distillers (`CBA`, `IDS`) now use it, so they
  no longer re-annotate the shared rule pool's rules in place.
- New `RuleModel.evaluate(data)` measures any model (or rule) on any data
  and returns a `ModelStats` without storing it. It replaces
  `RuleModel.annotate`, `stats(data)` and the `split=` names, which are
  gone: containers no longer store snapshots of their own performance
  (nothing read them), and `RuleSetClassifier.fit` no longer computes an
  unused one.
- `to_string` no longer takes `data=`: it prints each rule's own stored
  statistics (`(tp/fp)`, or the class distribution for distribution
  combiners), which are exactly the numbers the model uses, and does so by
  default whenever rules carry statistics. `show_stats=False` prints the
  bare rules.
- `remap` (and `filter`, and model conversions) keep the rules' statistics,
  the default rule's included; `remap` used to drop them.

### Changed (may change results or printouts)

- Printed models whose rules predict more than one class now start with a
  `% conflict resolution: ...` line naming how conflicts are decided (e.g.
  `max Laplace`, `sum of covered class counts`, `first matching rule`),
  from the new `RuleCombiner.describe()`; `to_string(show_resolution=False)`
  omits it. Rule heuristics got a readable `repr` (`Laplace`,
  `MEstimate(m=5)`) for it.
- `print(model)` now shows the full rendering (`to_string()`) for every
  model, rules included; the short `repr` identifies it --
  `FlatRuleSet(3 rules)`, and for a rule `SingleRule(2 conditions)`
  instead of `SingleRule(1 rules)`.
- The default rule of a decomposed model (`ConceptSet`, `ConceptCascade`,
  `PairwiseModel`, as built by one-vs-rest, ordered or pairwise fitting)
  now carries training statistics like every other default rule: its
  coverage of the entire training data, i.e. the class distribution.
- Pairwise voting: `WeightedVote(heuristic=...)` makes the deciding rule's
  weight configurable (default `Laplace`, as before), and a printed
  `PairwiseModel` names its combination (`% conflict resolution: pairwise
  vote weighted by Laplace of each pair's deciding rule`). Vote ties go to
  the tied labels' duel, then training frequency (the recorded priors,
  else read from the rules' stats), then label order.
- Ties no longer depend on rule position. `max` first lets the tied
  top-scoring rules vote; any remaining tie, for every combiner, goes to
  the class more frequent in the training data, then to the one that
  sorts first. (Before, `max`, `vote` and the distribution combiners fell
  back on rule or class insertion order.) Only `"list"` is order-based.
  `EnsembleModel` vote ties follow the same convention instead of member
  order, and an ensemble prints `% conflict resolution: (weighted) vote of
  members`.

### Added

- `to_string(pretty=True)` (rules and every model): each condition on its
  own line, indented under the head, and the coverage comment on its own
  line above the head. Prolog format only so far.
### Fixed

- In the `"conditions"` and `"pattern"` formats, models now show the class
  a rule predicts where it was missing: decision lists prefix each rule
  with it (`z: ¬f0, ¬f1`), and every model's default section names it
  (`% default: x`). Before, a decision list printed in these formats
  didn't say which class any rule predicted.

## 0.1.3 (2026-09-25)

A bug-fix release: binary attributes and missing values handled
correctly, AQR's covering fixed, and several printing fixes.

### Changed (may change results or printouts)

- `Rule.to_string("prolog")` quotes a head that isn't a valid bare Prolog
  atom (`'1'(X) :- ...`, `'Yes'(X) :- ...`, `'no-recurrence-events'(X) :- ...`),
  and gives each condition that introduces a value its own variable
  (`age(X, V1), V1 < 30, income(X, V2), V2 >= 50000`) instead of reusing
  one `V`, which forced unrelated attributes to unify.
- `tree_thresholds` (decision-tree discretization) rounds each split
  point to the coarsest value that still lies strictly between the two
  neighbouring observed values -- `15.17` instead of `15.172899999999998`.
  Display only: every row stays on the same side of the split.

- `AQR().fit(data)` now returns a `DecisionList` instead of a
  `FlatRuleSet` resolved by `combiner="list"`. Predictions are identical
  (the first covering rule in learn order wins either way), but the
  model now prints in that deciding order. `fit(data, model=FlatRuleSet)`
  still gives the list-resolved rule set, e.g. to compare combiners.
- A rule set whose own combiner is `"list"` and whose rules have more
  than one head now prints in list order, like a `DecisionList`, instead
  of grouped by label, which hid the order that decides.

### Changed (may break imports)

- `pyrulearn.attributes` moved to `pyrulearn.data.attributes`, next to
  `data.spec`, which builds a `DataSpec` from it: attribute types,
  `FeatureSpec`, the feature constraints, `evaluate_feature` and
  `MissingStrategy` describe the feature space, not rules. The old module
  path no longer exists.

### Added

- Binary attributes: `AttributeType.BINARY` and
  `DataSpecBuilder.add_binary("sex", ["male", "female"])`, a closed set of
  exactly two values. They always get both features (`sex=male`,
  `sex=female`), each the other's negation, and never `!=` columns; a
  value outside the two counts as missing. This sits between `BOOLEAN`
  (one tested value, `smoker` / `not smoker`, unchanged) and `NOMINAL`
  (a possibly incomplete value list, which keeps its `!=` columns even
  for two values, since an unknown value must satisfy every `x!=v`).
- `build_dataspec(domains=...)`: declared value lists per column.
- `merge_dataspecs` merges two binary attributes over the same values as
  binary, and a binary one with a nominal one (or with a binary one over
  different values) as the nominal union. `Rule.remap` turns `x!=a` into
  `x=b` when the target attribute is binary over `{a, b}`.
- `examples/demo_covering.py`: a step-by-step separate-and-conquer teaching
  demo (PFossil's covering and hill climbing replayed on Titanic, one set
  of plots per search step).

### Changed (may change feature counts)

- `build_dataspec`, `read_csv` and `read_arff` infer a column with
  exactly two values -- declared in an ARFF header, else observed in the
  data -- as binary instead of nominal: with negation on, it gets 2
  columns instead of 4 (e.g. kr-vs-kp's 144 columns become 76). Importers
  that infer a `DataSpec` from a rule set (Weka, LORD, wittgenstein) keep
  nominal attributes, since a rule set need not mention every value.

### Fixed

- `AQR` with a target class (and in its one-vs-rest, cascade and pairwise
  decompositions) stopped learning for a class as soon as the example it
  seeded on admitted no acceptable rule -- typically a noisy example --
  leaving every other positive uncovered: with 3% label noise it often
  learned no rule at all. A failed seed is now dropped and the next
  uncovered positive tried; learning stops only once every positive is
  covered or dropped. Learners that search all rules at once (CN2, PFoil,
  PFossil) are unchanged.
- `read_arff` turned ARFF's missing marker `?` into an ordinary category
  (`sex=?`); it is now a missing value. Nominal attributes also take their
  values from the header's declared list rather than only the values
  that occur in the data.
- `examples/demo_seco_learners_comparison.py`: `lord` failed on every fold.
  Its rules were parsed against a dataspec inferred from LORD's own output,
  whose feature numbering didn't match the training data; they now bind to
  the real dataspec by column position, as `LordJar` already did.
- `examples/demo_workflow_comparison.py` and
  `demo_seco_learners_comparison.py`: pyrulearn's own binarization now gets
  the raw data with real missing values (handled by
  `MissingStrategy.NEVER_COVERS`) instead of median/"?"-imputed data, and
  Pima diabetes's zero-encoded measurements are treated as missing. Only
  the external learners still get imputed data.

## 0.1.2 (2026-09-22)

First step toward the weight-transparency framework: none of this touches
`WeightedRule` or adds any new declarative-weight storage yet, but a printed
model's coverage decoration is now meant to let a reader manually derive
`predict()`'s decision from the page, for whatever mechanism a model's own
combiner actually uses.

### Changed (may break `to_string(data=...)` output)

- Coverage decoration: `(n_covered/n_errors)` is now `(tp/fp)`, and the
  `[n_unique_covered/n_unique_errors]` bracket is gone entirely -- "unique"
  meant something different per resolution strategy (no other same-target
  rule / no earlier rule / ...), so it wasn't a stable concept to keep
  printing uniformly.
- A model whose own `combiner` is genuinely a `DistributionCombiner`
  (`micro`/`macro` × `vote`/`max`) with more than two classes now prints the
  full per-class coverage breakdown instead of `(tp/fp)` -- everything that
  combiner's `resolve()` reads, nothing it doesn't -- with a one-line
  `% classes: [...]` legend printed once above the output. Two classes make
  the vector exactly `(tp/fp)` reordered, so it stays suppressed there.
- `CN2`'s default `ConceptSet` now resolves a clash between covering rules
  by summing their covered-class distributions (`MicroVoteCombiner`), not
  the `SeCo` family's generic `combiner="max"` -- matching Clark & Boswell's
  own unordered-CN2 algorithm. `ConceptSet`'s own default, and every other
  `SeCo` member, are unchanged.

### Added

- `to_string`'s new `show_distribution`/`show_classes` (both `None` by
  default) let a caller force either the vector or the legend on or off
  independently, for any model, any class count, any combiner.
- `EnsembleModel`/`PairwiseModel` gained `to_string` -- calling it used to
  raise `AttributeError`. `PairwiseModel` forces each pair's own two-class
  legend on by default and narrows the printed data to that pair's own rows;
  `DeepModel` still has none, being a structure-only stub.
- `ClassAssociationRuleMiner` renamed to `CARMiner` (shorter; it was the
  widest cell in several README tables).

### Documentation

- README table columns: the common `pyrulearn.*`/`pyrulearn.learners`/
  `pyrulearn.interfaces` prefix repeated in every cell is now given once in
  the header instead.
- The "Not yet implemented" list consolidated from about nine granular
  bullets into five themed ones, folding in a planned `pyIDS` cross-check
  and a planned `RuleFit`-style distiller.

## 0.1.1 (2026-09-21)

### Changed (may break imports)

- The native RIPPER re-implementation is now called `Pypper`
  (`pyrulearn.learners.seco.Pypper`); the `RIPPER` class and the old `Pypper`
  alias are gone. The external wittgenstein `RIPPERk` and Weka `JRip` wrappers
  are unchanged.
- `CBA` and `CMAR` moved into `pyrulearn.learners.associative` (next to
  `ClassAssociationRuleMiner` and `RuleDistiller`); the modules
  `pyrulearn.learners.cba` and `pyrulearn.learners.cmar` no longer exist.
  `IDS` stays in `pyrulearn.learners.ids`.
- LORD is now credited to Huynh, Fürnkranz & Beck (2023), including the
  `LORDImporter` provenance source string, which changed to
  `"LORD (Huynh, Fürnkranz & Beck, 2023)"`.

### Fixed

- `rule_refinement_plot` axis labels showed `np.str_('pos')` instead of `'pos'`;
  the start label no longer collides with the plot title.
- `examples/demo_workflow_comparison.py` ran into three problems after the model
  hierarchy refactor: the forest passed `combiner=` to `EnsembleModel.predict`,
  the gitignored ARFF scratch directory was never created, and the Original
  forest lost its leaf statistics through `Rule.remap`.

### Documentation

- README restructured: separate sections for data representation and rule
  models, conflict resolution merged with the combiner descriptions, heuristics
  next to statistics, a coverage-space section with pencil / parallel / curved
  isometrics, learners before external interfaces, and a table of what each
  external tool receives. The module map now has one row per package.
- `references.bib` gains *Foundations of Rule Learning* (Fürnkranz, Gamberger &
  Lavrač, 2012) and the corrected LORD authors.
- `examples/demo.py` no longer writes the mixed-target `coverage_space.png`; all
  demo reports and plots were regenerated.

## 0.1.0

- Initial public release.
