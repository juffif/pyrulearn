"""
pyrulearn.interfaces.base
============================

`RuleImporter`: shared base for everything that brings rules from an
external source into pyrulearn, regardless of what shape that source
is. Two intermediate layers derive from it, matching the two shapes an
external source actually comes in:

- `ObjectRuleImporter` (`import_model(model, dataspec) -> RuleModel(s)`)
  -- for an already-fitted **live model object** (e.g. a scikit-learn
  estimator). See `pyrulearn.interfaces.sklearn.SklearnTreeImporter`.
- `StringRuleImporter` (`parse(source) -> RuleModel`) -- for a **text/
  serialized rule format** (e.g. a RIPPER/JRip rule dump, a pattern
  string). See `PatternStringImporter`, below.

Every concrete importer registers itself in one shared registry
(`register_importer`/`get_importer`) regardless of which intermediate
layer it derives from, so callers/plugins can look one up by name
(``get_importer("sklearn_tree")``) without caring which shape it is.

`RuleImporter` itself also centralizes provenance tagging
(`_stamp_rule_provenance`/`_stamp_provenance`): every rule (and the
model as a whole) a concrete importer produces gets a `pyrulearn.models.
Provenance` -- this importer's class, its params, and `SOURCE` -- so
once rules from many different importers (of either shape) are mixed
together for comparison, `rule.provenance.source` answers "which base
learner produced this rule?" uniformly.

Most concrete importers live in their own submodules, one per cluster
of source models/formats that actually share implementation (not
rigidly one file per library or one per algorithm -- whichever
grouping matches the real code reuse) -- `sklearn` for every sklearn
estimator that's fundamentally tree-based; a real text format
(RIPPER/JRip, CN2, ...), once one is added, gets its own module the
same way, since different textual formats share no parsing logic with
each other. `PatternStringImporter` is the one exception living here
rather than in its own file: it's a reference/test-fixture format
(not a real external tool's output), so it stays next to the base
classes it demonstrates rather than implying it needs the same kind of
dedicated module a genuine format would. Only the classes in this
module (and the registry) have no external dependencies beyond this
package -- importing `pyrulearn.interfaces` doesn't pull in any
*concrete* format-specific importer's own dependencies; import the
specific submodule you need for that.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Type, Union

from ..data import DataSpec
from ..models import FlatRuleSet, RuleModel
from ..rule import Rule


class RuleImporter(ABC):
    """Shared base for every adapter that brings rules from an external
    source into pyrulearn -- not instantiated directly; subclass
    `ObjectRuleImporter` or `StringRuleImporter` (or, rarely, this class itself
    for some third shape of source neither of those fits).
    """

    #: Human-readable identifier for the source library/algorithm this
    #: importer reads from (e.g. "sklearn.tree.DecisionTreeClassifier",
    #: "wittgenstein.RIPPER") -- stored under every produced rule's
    #: `.provenance.source` via `_stamp_rule_provenance`.
    SOURCE: str = NotImplemented

    def _stamp_rule_provenance(self, rules: Sequence[Rule], **params: Any) -> list:
        """Wrap each of `rules` as a `pyrulearn.models.SingleRule` (an
        already-`SingleRule` item passes through unchanged) and give it
        its own `Provenance` -- this importer's class, `params`, and
        `self.SOURCE`. A *fresh* `Provenance` per rule, not one instance
        shared across the batch, so mutating one rule's provenance (e.g.
        annotating it later) can never silently leak onto its siblings
        just because they came from the same call (e.g. one
        random-forest tree's leaves, all sharing the same `tree_index`
        value, but not the same object).

        Returns the (now `SingleRule`-wrapped) list -- a `pyrulearn.models`
        container constructor normalizes an already-`SingleRule` item as
        a no-op, so this composes directly with whichever container the
        caller builds.
        """
        from ..models import Provenance, SingleRule
        out = []
        for r in rules:
            sr = r if isinstance(r, SingleRule) else SingleRule(r)
            sr.provenance = Provenance(learner=type(self).__name__, params=dict(params), source=self.SOURCE)
            out.append(sr)
        return out

    def _stamp_rule_stats(self, rules: Sequence[Rule], data: Optional[Any] = None, split: str = "data") -> list:
        """Wrap each of `rules` as a `SingleRule` (as `_stamp_rule_provenance`
        does) and, if `data` is given, populate its measured `stats(data,
        split)` against it -- the exact rows a `fit()` round trip wrote
        out for the external tool and is now reading these rules back
        against. `data` is only ever given by that round trip
        (`ExternalRuleLearner._fit_native`, via each learner's `_import`);
        a bare `parse`/`import_model` call on captured/serialized output
        with no live data on hand (a demo or test parsing a saved rule
        dump) passes `data=None` and gets un-annotated rules, exactly as
        before -- there's nothing to measure against.

        Deliberately done here, at the exact point this importer's rules
        are bound to `data`'s dataspec (e.g. `placeholder_features=True`'s
        positional `f{i}` binding), rather than by a caller re-deriving
        that binding independently afterward: one trusted place owns the
        rule<->data correspondence, so there's no risk of two different
        reconstructions of it silently drifting apart ("no feature
        guessing"). See `pyrulearn.models.annotate_rules`.
        """
        from ..models import annotate_rules
        return annotate_rules(rules, data, split)

    def _stamp_provenance(self, model: Any, **params: Any) -> Any:
        """Set `model.provenance` (see `pyrulearn.models.Provenance`) to
        this importer's class, `params`, and `self.SOURCE` -- the
        external library/algorithm this importer reads from. Returns
        `model`, so a concrete `import_model`/`parse` can wrap its return
        statement directly: ``return self._stamp_provenance(FlatRuleSet(rules), ...)``.
        Only meaningful when the importer is used directly, without a
        `pyrulearn.learners.RuleLearner` -- `RuleLearner.fit` re-stamps
        its own return value with the *learner's* provenance instead.
        """
        from ..models import Provenance
        model.provenance = Provenance(learner=type(self).__name__, params=dict(params), source=self.SOURCE)
        return model


# -- registry (any RuleImporter subclass, of either shape) -------------------

_REGISTRY: Dict[str, Type[RuleImporter]] = {}


def register_importer(name: str, cls: Type[RuleImporter]) -> None:
    _REGISTRY[name] = cls


def get_importer(name: str) -> Type[RuleImporter]:
    if name not in _REGISTRY:
        raise KeyError(f"No importer registered for {name!r}; available: {list(_REGISTRY)}")
    return _REGISTRY[name]


# -- ObjectRuleImporter --------------------------------------------------

class ObjectRuleImporter(RuleImporter):
    """Base for adapters that convert an already-fitted external model
    object into one or more pyrulearn `RuleModel`s.
    """

    @abstractmethod
    def import_model(
        self, model: Any, dataspec: DataSpec, data: Optional[Any] = None,
    ) -> Union[RuleModel, Sequence[RuleModel]]:
        """Convert `model` -- already fitted, over `dataspec`'s Boolean
        feature space -- into one `RuleModel`, binding the
        resulting rules to `dataspec` directly (not a copy of it), or
        several `RuleModel`s if the source model isn't naturally a
        single one (e.g. a random forest producing one `DisjointRuleSet`
        per tree, since forest-level voting doesn't map onto any single
        `RuleModel.predict()`'s semantics). Concrete subclasses
        implement the actual model-specific traversal/parsing.

        `data`, when given (only by the `fit()` round trip -- see
        `ExternalRuleLearner._import`), is the exact `DataRepresentation`
        `model` was fitted on: concrete subclasses pass it to
        `_stamp_rule_stats` so each produced rule carries its own
        training-set stats. `None` (a bare, standalone `import_model`
        call with no live data) skips that -- unchanged from before.
        """
        raise NotImplementedError

    def infer_dataspec(self, model: Any, feature_names: Sequence[str]) -> DataSpec:
        """Build a `DataSpec` directly from `model`'s own fitted
        structure -- e.g. a decision tree's actual split thresholds --
        rather than requiring one built in advance from raw data (see
        `pyrulearn.data.io.build_dataspec` for that other direction).
        `feature_names[i]` names the raw input column at index `i`
        `model` was trained on.

        Default here: one plain Boolean feature per name, matching this
        module's original assume-already-Boolean convention -- concrete
        importers that can discover richer structure (numeric
        thresholds, nominal categories) from an already-fitted model
        override this, e.g. `pyrulearn.interfaces.sklearn.
        SklearnTreeImporter.infer_dataspec`. Not abstract: an importer
        that doesn't override this still works with `import_model`
        exactly as before, just without the richer discovery.
        """
        return DataSpec(list(feature_names))


# -- StringRuleImporter --------------------------------------------------

class StringRuleImporter(RuleImporter):
    """Base class for string/serialized-format-specific rule importers.

    Subclasses implement `parse`, which takes learner-specific input (a
    file path, string, or in-memory object) and returns a `RuleModel`.

    A `feature_names` list can be supplied at construction time so that
    literal names in the source format (e.g. ``"petal_length <= 2.5"``)
    can be mapped to feature indices in the target Boolean space; how
    that mapping happens is importer-specific (e.g. it may involve
    binarizing a numeric threshold into a synthetic Boolean feature).
    Internally this is kept as a plain, data-less `DataSpec` so parsed
    rules can carry it as their `dataspec` (for pretty-printing and,
    once real data is attached, coverage checks) rather than a bare name
    list.
    """

    def __init__(self, feature_names: Optional[Sequence[str]] = None):
        self.dataspec: Optional[DataSpec] = DataSpec(list(feature_names)) if feature_names is not None else None

    @abstractmethod
    def parse(self, source) -> RuleModel:
        """Parse `source` (format-specific) into a `RuleModel`."""
        raise NotImplementedError

    def _name_to_index(self, name: str) -> int:
        if self.dataspec is None:
            raise ValueError(
                "feature_names must be provided to this importer to resolve "
                f"literal name {name!r} to a feature index"
            )
        try:
            return self.dataspec.feature_index(name)
        except KeyError:
            raise KeyError(f"Unknown feature name {name!r}") from None


# -- PatternStringImporter ------------------------------------------------

class PatternStringImporter(StringRuleImporter):
    """Reference importer: one rule per line, each line a whitespace- or
    comma-separated pattern of 0/1/`-` tokens (matches
    `Rule.from_pattern_string`), optionally followed by ``=> target``.

    Example input::

        1 0 - 1 => pos
        - 1 1 0 => neg
    """

    SOURCE = "pyrulearn.interfaces.base.PatternStringImporter"

    def parse(self, source: str) -> FlatRuleSet:
        lines = source.strip().splitlines() if "\n" in source or not source.endswith((".txt", ".rules")) \
            else open(source).read().strip().splitlines()
        rules = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                pattern, target = line.split("=>")
                target = target.strip()
            else:
                pattern, target = line, None
            rules.append(Rule.from_pattern_string(
                pattern.strip(), target=target, dataspec=self.dataspec
            ))
        rules = self._stamp_rule_provenance(rules)
        return self._stamp_provenance(FlatRuleSet(rules))


register_importer("pattern_string", PatternStringImporter)
