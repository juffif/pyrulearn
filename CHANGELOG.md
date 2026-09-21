# Changelog

All notable changes to pyrulearn are listed here, newest first. The project
is in early development (alpha): until 1.0, minor versions may change the API.

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
