"""
pyrulearn.interfaces
=======================

`RuleImporter`/`ObjectRuleImporter`/`StringRuleImporter`, the shared
registry (`register_importer`/`get_importer`), and the reference
`PatternStringImporter` all live in `pyrulearn.interfaces.base` -- see
that module's docstring for the rationale behind the two shapes.

Real concrete importers live in their own submodules, one per cluster
of source models/formats that actually share implementation (not
rigidly one file per library or one per algorithm -- whichever
grouping matches the real code reuse). Each pairs an importer with the
`ExternalRuleLearner` wrapper that runs the tool:

- `sklearn` -- every sklearn estimator that's fundamentally tree-based
  (decision trees, random forests), and `RuleSetClassifier`, the other
  direction: wrapping a `RuleModel` as a scikit-learn estimator;
- `wittgenstein` -- IREP and RIPPER;
- `imodels` -- Bayesian rule lists and Bayesian rule sets;
- `weka` -- JRip, PART and J48, driven as a subprocess;
- `lord` -- the reference (Java) LORD implementation;
- `pyarc` -- the external CBA implementation.

Only `pyrulearn.interfaces.base`
(and this package's re-exports of it) have no external dependencies
beyond this package -- importing `pyrulearn.interfaces` doesn't pull in
any *concrete* format-specific importer's own dependencies; import the
specific submodule you need for that.
"""

from __future__ import annotations

from .base import (
    get_importer,
    ObjectRuleImporter,
    PatternStringImporter,
    register_importer,
    RuleImporter,
    StringRuleImporter,
)

__all__ = [
    "RuleImporter", "ObjectRuleImporter", "StringRuleImporter", "PatternStringImporter",
    "register_importer", "get_importer",
]
