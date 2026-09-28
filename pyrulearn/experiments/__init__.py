"""
pyrulearn.experiments
=========================

Shared infrastructure for pyrulearn's demos and other benchmark-style
experiments: "run the following learners on all (or a selection of)
datasets, evaluated via k-fold cross-validation" -- so a demo's own code
is just picking learners, picking datasets, and deciding what to report,
not reimplementing loading, splitting, per-fit timeouts, measurement,
ranking statistics or report writing again. Needs `pandas`/`scipy`
(`matplotlib` too for `stats.critical_difference_diagram`) -- the
``experiments`` extra.

- `pyrulearn.experiments.catalog` -- `Catalog`, the 100-dataset OpenML
  catalog demos select from (see its own module docstring). Needs
  `pandas` only through `CatalogEntry.load()`, not to import the module.
- `pyrulearn.experiments.runner` -- `run_cv`, the cross-validation loop
  (dataset loading, per-fold binarization, per-fit timeout via
  `TimeoutRunner`, uniform measurement) -- returns one long-format
  result table.
- `pyrulearn.experiments.stats` -- optional, separately-callable
  statistics over a `run_cv` table: `mean_rank`, `friedman_test`,
  `critical_difference_diagram`. Never invoked automatically by `run_cv`.
- `pyrulearn.experiments.report` -- small Markdown-writing building
  blocks (`render_setup_section`, `render_results_table`) for turning a
  `run_cv` table (plus whatever `stats` a demo picked) into a report.

Nothing is re-exported here (unlike `pyrulearn.data`): every submodule
needs `pandas`/`scipy` at minimum, so a bare ``import pyrulearn``
shouldn't need them -- import the submodule you actually want, e.g.
``from pyrulearn.experiments.runner import run_cv``.
"""

from __future__ import annotations
