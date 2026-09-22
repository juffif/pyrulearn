# Changelog

All notable changes to pyrulearn are listed here, newest first. The project
is in early development (alpha): until 1.0, minor versions may change the API.

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
