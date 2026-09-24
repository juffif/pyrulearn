# Changelog

All notable changes to pyrulearn are listed here, newest first. The project
is in early development (alpha): until 1.0, minor versions may change the API.

## Unreleased

### Changed (may change results or printouts)

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

- `pyrulearn.data.catalog`: a catalog of about 100 benchmark datasets
  (OpenML-CC18, the LORD evaluation's datasets, classic XAI and
  rule-learning datasets), metadata and download pointers only. Select by
  task (binary/multiclass), size (small/medium/large), attribute types
  (categorical/numeric/mixed), tags or name -- `Catalog.default().select(...)`
  or `.parse("binary,small,medium")` -- and `entry.load()` downloads and
  caches the raw data with curated fixes applied (restored attribute
  names, dropped leakage columns, category codes, missing-value markers).
  `tools/build_catalog.py` maintains the catalog file.
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
