"""
pyrulearn.models
================

The rule-model type hierarchy: `RuleModel` and its family. A *model* is
something you call `predict` on; the types below capture the different
*structures* a set of rules can have, so that (a) a learner can declare
which model types it produces and be checked against that, and (b)
conversions between model types are well-defined.

Terminology: a rule's consequent is its **head** (`Rule.target`); the
set of heads a model can output is its **labels** -- "label", not
"class", throughout, since a model need not be a classifier (a ranking
model orders the same labels). A `ConceptModel` is the definition of one
**concept** (one label, present/absent) -- the classical concept-learning
target.

The hierarchy is organised by **resolution** -- how a prediction is
decided when several rules apply:

- `RuleSet`   (ABC) -- *unordered*; covering rules are reconciled by a
  `combiner`, or a disjointness assumption. Position never matters.
    - `FlatRuleSet`      a plain bag of mixed-head rules + one combiner
                         (the generic case; counterpart to `DeepModel`).
    - `ConceptModel`     rules that all share one head -- one concept.
    - `ConceptSet`       one `ConceptModel` per label + a combiner.
    - `DisjointRuleSet`  rules assumed pairwise mutually exclusive.
- `RuleList` (ABC) -- *ordered*; the first matching rule wins.
    - `DecisionList`     a linear list.
    - `ConceptCascade`   ordered `ConceptModel`s ("peeling" one-vs-rest).
- `CompositeModel` (ABC) -- members that are themselves `RuleModel`s.
    - `EnsembleModel`   members combined by a flat (weighted) vote.
    - `PairwiseModel`   one binary member per label pair (round robin).
    - `DeepModel`       members wired output-to-input (stacking) -- STUB.
- `SingleRule` -- one rule, wrapped so it carries stats and predicts.

Cross-cutting mixins (traits, *not* hierarchy levels): `_FlatRules`
(holds a flat `SingleRule` list -- every model's flat rule list is a
list of models, not bare `Rule`s, so stats/provenance reach every leaf)
and `_ConceptIndexed` (holds a list of `ConceptModel` blocks).
Concept-indexing spans both `RuleSet` (`ConceptSet`) and `RuleList`
(`ConceptCascade`).

`default_prediction` (every model): the policy for a row the model's own
rules don't decide -- a bare label, `None` (abstain), or a
`DefaultPrediction` object (`MajorityClass`). `default_rule` is a
read-only empty-body `Rule` view of a *constant* policy, a home for
fall-through stats.

`WeightedRule` (`pyrulearn.rule`) carries the declarative per-rule
`weight` -- part of the model, usable at predict time, no dataset
needed. `stats(data, split)` carries *measured* performance instead: a
`pyrulearn.evaluation.ModelStats` snapshot (a `ConfusionMatrix` from this
model's own `predict(data)` vs `data.y`, plus `n_rules`/`n_conditions`),
`None` until annotated. Every `SingleRule` leaf has its own `stats` too
-- the recursion is structural (`CompositeModel`/`_ConceptIndexed`
members are themselves full `RuleModel`s), not a tree embedded inside
one stats object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence as _SequenceABC
from itertools import chain
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union,
)

import numpy as np

from .combiners import RuleCombiner, _resolve_combiner, _rule_stats
from .data import DataRepresentation
from .data import DataSpec
from .rule import Rule

if TYPE_CHECKING:
    from .evaluation import ConfusionMatrix, ModelStats

__all__ = [
    "RuleModel", "SingleRule",
    "RuleSet", "FlatRuleSet", "PooledRuleSet", "RuleView", "ConceptModel", "ConceptSet", "DisjointRuleSet",
    "RuleList", "DecisionList", "ConceptCascade",
    "CompositeModel", "EnsembleModel", "PairwiseModel", "DeepModel",
    "Resolution", "FirstMatch", "Exclusive", "Combine",
    "DefaultPrediction", "MajorityClass", "Provenance", "ModelStats",
    "PairwiseVote", "PairwiseCombiner", "MajorityVote", "WeightedVote", "AccuracyWeightedVote",
    "can_convert", "convert", "annotate_rules", "annotate_default_rule",
]


class Provenance:
    """What built a model: the `pyrulearn` learner or importer class that
    constructed it, its parameters, and -- for an imported model -- the
    external algorithm identifier (`ObjectRuleImporter`/`StringRuleImporter`
    `SOURCE`, e.g. ``"sklearn.tree.DecisionTreeClassifier"``; `None` for a
    native learner).

    Set once, by whichever call actually built the model:
    `RuleLearner.fit` stamps it on its return value (`learner` = the
    `RuleLearner` subclass, `params` = its constructor args plus any
    `fit(..., **model_kwargs)`, `source` = `ExternalRuleLearner.IMPORTER.
    SOURCE` if external, else `None`); an importer's `import_model`/`parse`
    stamps it the same way when called directly, without a learner
    (`learner` = the importer class, `source` = its own `SOURCE`).

    Carried through unchanged by `filter`/`remap`/a model→model
    `convert` (see `_carry_provenance`): none of those change *what*
    built the rules, only which ones are kept, which dataspec they're
    bound to, or how they're packaged -- narrowing/rebasing/repackaging
    a model doesn't make a new claim about who built it.
    """

    __slots__ = ("learner", "params", "source")

    def __init__(self, learner: str, params: Dict[str, Any], source: Optional[str] = None):
        self.learner = learner
        self.params = dict(params)
        self.source = source

    def __repr__(self) -> str:
        src = f", source={self.source!r}" if self.source is not None else ""
        return f"Provenance(learner={self.learner!r}, params={self.params!r}{src})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Provenance) and self.learner == other.learner
                and self.params == other.params and self.source == other.source)


def _sortkey(c: Any):
    """Total order for labels that keeps numbers numeric -- for
    deterministic, otherwise-arbitrary ordering of label lists."""
    return (0, c) if isinstance(c, (int, float)) else (1, str(c))


def _default_covered_by_key(rule: Rule) -> Any:
    """`covered_by`'s default within-block ranking: Laplace on the
    rule's own *measured* stats -- the same "rank by measured
    reliability" default `sort_rules`/`HeuristicMaxCombiner` use
    elsewhere (there is no longer a declarative `meta['weight']`
    fallback; `WeightedRule.weight` is declarative and predict-time-
    only, not a ranking substitute). Raises `ValueError` via
    `_heuristic_score` if the rule has no measured stats -- annotate
    first (`annotate_rules`, or a `fit()`/importer `data=` call, which
    already do)."""
    from .combiners import _heuristic_score
    from .heuristics import Laplace
    return _heuristic_score(rule, Laplace())


# =========================================================== default policy ===

class DefaultPrediction(ABC):
    """Object form of a model's `default_prediction` policy -- consulted
    per row that no rule decides, so it can vary by row. The other two
    forms need no object: a bare label, and `None` (abstain).

    A subclass whose prediction is in fact one fixed label should report
    it via `constant_target`, so `RuleModel.default_rule` can materialise
    an empty-body rule to carry its statistics.
    """

    @abstractmethod
    def predict(self, rules: Sequence[Rule], data: DataRepresentation, row: int) -> Any:
        """Fallback for row `row` of `data`, given no rule covers it.
        Called once per uncovered row (not vectorised)."""
        raise NotImplementedError

    @property
    def constant_target(self) -> Optional[Any]:
        return None


class MajorityClass(DefaultPrediction):
    """Predicts one fixed label: the most frequent among a chosen set of
    training rows, optionally restricted to candidate labels. Resolved
    once at construction against `data` (needs `data.y`), so it is
    independent of whatever data prediction later runs on.

    (Name kept for now; a `MajorityLabel` rename is on the refactor's
    breakage list -- it touches SeCo/PyLORD/multiclass/importers.)

    Parameters
    ----------
    data      training data; `data.y` supplies the counted labels.
    labels    candidate labels to choose among (`None` = all present).
    mask      boolean row mask to count over (`None` = every row); pass
              the "no rule covers this row" mask to get the majority of
              exactly the rows that fall through to the default.
    """

    def __init__(
        self,
        data: DataRepresentation,
        labels: Optional[Sequence[Any]] = None,
        mask: Optional[np.ndarray] = None,
    ):
        if data.y is None:
            raise ValueError("MajorityClass needs data.y")
        self.labels: Optional[List[Any]] = None if labels is None else list(labels)
        y = np.asarray(data.y)
        rows = y if mask is None else y[np.asarray(mask, dtype=bool)]
        pool = rows if self.labels is None else rows[np.isin(rows, self.labels)]
        if pool.size == 0:
            self._target: Any = None
        else:
            values, counts = np.unique(pool, return_counts=True)
            self._target = values[int(np.argmax(counts))]

    @property
    def constant_target(self) -> Optional[Any]:
        return self._target

    def predict(self, rules: Sequence[Rule], data: DataRepresentation, row: int) -> Any:
        return self._target


# ================================================================ resolution ===

class Resolution(ABC):
    """How `predict` turns "which rules cover row j" into a label, and
    what counts as *uniquely* one rule's coverage for stats. A model
    holds one; `RuleSet` / `RuleList` set the default."""

    @abstractmethod
    def predict(
        self,
        rules: Sequence[Rule],
        cov: np.ndarray,
        fallback: Callable[[int], Any],
        n_samples: int,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def unique_mask(self, cov: np.ndarray, rules: Sequence[Rule]) -> np.ndarray:
        """`(n_rules, n_samples)` bool: [i, j] True iff row j counts as
        uniquely rule i's under this resolution."""
        raise NotImplementedError


class FirstMatch(Resolution):
    """Rules tried in list order; the first that covers a row decides it.
    Unique coverage = the rows a rule actually *fires* on (it matches and
    no earlier rule already claimed the row)."""

    def predict(self, rules, cov, fallback, n_samples):
        preds = np.empty(n_samples, dtype=object)
        decided = np.zeros(n_samples, dtype=bool)
        for i, r in enumerate(rules):
            take = cov[i] & ~decided
            preds[take] = r.target
            decided |= take
        for j in np.flatnonzero(~decided):
            preds[j] = fallback(int(j))
        return preds

    def unique_mask(self, cov, rules):
        already = np.zeros(cov.shape[1] if cov.size else 0, dtype=bool)
        um = np.zeros_like(cov)
        for i in range(cov.shape[0]):
            um[i] = cov[i] & ~already
            already |= cov[i]
        return um


class Exclusive(FirstMatch):
    """Rules assumed pairwise disjoint (≤ 1 covers any row) -- prediction
    is `FirstMatch` (first == only), but unique coverage is the raw
    coverage, since nothing else could also cover the row."""

    def unique_mask(self, cov, rules):
        return cov.copy()


class Combine(Resolution):
    """Aggregate every rule covering a row via a `pyrulearn.combiners.
    RuleCombiner` (or a shortcut string -- `"max"`, `"vote"`, `"list"`,
    ...). Unique coverage = rows a rule covers that no other rule *of the
    same head* covers."""

    def __init__(self, combiner: Union[str, RuleCombiner] = "max"):
        self.combiner = combiner

    def predict(self, rules, cov, fallback, n_samples):
        strat = _resolve_combiner(self.combiner)
        preds = np.empty(n_samples, dtype=object)
        for j in range(n_samples):
            covering = [i for i in range(len(rules)) if cov[i, j]]
            preds[j] = strat.resolve(rules, covering) if covering else fallback(j)
        return preds

    def unique_mask(self, cov, rules):
        um = np.zeros_like(cov)
        for i, r in enumerate(rules):
            same = [j for j, r2 in enumerate(rules) if j != i and r2.target == r.target]
            if same:
                um[i] = cov[i] & ~np.any(cov[same], axis=0)
            else:
                um[i] = cov[i]
        return um


# ================================================================ base model ===

class RuleModel(ABC):
    """A collection of rules you can `predict` with. Subclasses supply
    the structure (`rules`) and the prediction (`predict`); this base
    provides the coverage matrix, the `covered_by` explanation, the
    `default_prediction` policy, on-demand `stats`, and `filter`/`remap`.
    """

    def __init__(self, *, default_prediction: Any = None):
        self._default_prediction: Any = default_prediction
        self._default_rule: Optional["SingleRule"] = None
        self._stats: Dict[str, Any] = {}
        #: what built this model -- see `Provenance`. `None` until a
        #: `RuleLearner.fit` call or a direct importer `import_model`/
        #: `parse` call stamps it.
        self.provenance: Optional[Provenance] = None

    # -- structure ------------------------------------------------------

    @property
    @abstractmethod
    def rules(self) -> Sequence["SingleRule"]:
        """Flat view of every rule in this model, each a `SingleRule`
        (pooled recursively for a `CompositeModel`, all the way down to
        the leaves) -- so every rule carries its own `predict`/`stats`/
        `provenance`, not just the model as a whole. For inspection /
        printing / complexity metrics only -- `predict` never routes
        through it.

        A read-only `Sequence` (`len`, indexing, iteration), not
        necessarily a `list`: most models return a real list, but a
        `pyrulearn.pool.PooledRuleSet` returns a lazy `RuleView` that builds
        each `SingleRule` on first access. Don't mutate it (use `add`), and
        note `list(model.rules)` on a pooled model builds every rule."""
        raise NotImplementedError

    @property
    def labels(self) -> List[Any]:
        """Sorted labels this model can output -- the union of its rules'
        heads and its constant default, if any. (Not `classes`: a model
        need not classify -- a ranking model orders these same labels.)"""
        labels = {r.target for r in self.rules if r.target is not None}
        dt = self._default_target()
        if dt is not None:
            labels.add(dt)
        return sorted(labels, key=_sortkey)

    def _rebuild_kwargs(self) -> Dict[str, Any]:
        """Constructor kwargs (beyond the rules/blocks) that `filter`
        and `remap` carry to the rebuilt instance."""
        return {"default_prediction": self._default_prediction}

    # -- prediction ---------------------------------------------------

    @abstractmethod
    def predict(self, data: DataRepresentation) -> np.ndarray:
        """One label per row of `data`; rows the model's rules don't
        decide fall back to `default_prediction`."""
        raise NotImplementedError

    def predict_distribution(self, data: DataRepresentation) -> np.ndarray:
        """`(n_samples, len(self.labels))` per-label mass array, columns
        aligned with `self.labels`. Base: all mass on `predict`'s label
        (all-zero row for an abstained example). Scored models override
        with a genuine distribution."""
        labels = self.labels
        idx = {c: k for k, c in enumerate(labels)}
        out = np.zeros((data.n_samples, len(labels)), dtype=float)
        for j, p in enumerate(self.predict(data)):
            if p in idx:
                out[j, idx[p]] = 1.0
        return out

    def covered_by(
        self,
        data: DataRepresentation,
        by: Optional[Callable[[Rule], Any]] = None,
    ) -> List[List[Rule]]:
        """For each row of `data`, the rules whose body holds for it,
        ordered as an explanation of the prediction: the predicted
        label's block first, then other labels' blocks (each ordered by
        its strongest rule), and within a block sorted by `by` (a
        per-rule key callable; default: descending Laplace-on-measured-
        stats, the same default `sort_rules`/`HeuristicMaxCombiner` use
        -- raises if a block/ranking with more than one real candidate
        needs to compare rules and one of them has no measured stats;
        annotate first). A row no rule covers yields ``[default_rule]``
        or ``[]``. A block/ranking with only one candidate is never
        scored at all -- nothing to compare, so no stats are demanded
        for it even if that lone rule has none.

        For a different `by=` -- a `RuleHeuristic` scored fresh against
        `data`, or a plain callable -- use `pyrulearn.evaluation.
        sort_rules` directly, either as `by=` here or on this method's
        own output."""
        key = by if by is not None else _default_covered_by_key
        rules = self.rules
        cov = self.coverage_matrix(data)
        preds = np.asarray(self.predict(data))
        out: List[List[Rule]] = []
        for j in range(data.n_samples):
            idx = np.flatnonzero(cov[:, j]) if cov.size else np.empty(0, dtype=int)
            if idx.size == 0:
                dr = self.default_rule
                out.append([dr] if dr is not None else [])
                continue
            blocks: Dict[Any, List[Rule]] = {}
            for i in idx:
                blocks.setdefault(rules[i].target, []).append(rules[i])
            # only call `key` where there's an actual choice to make -- a
            # singleton block/other-block list has nothing to compare, so
            # skip it rather than force every rule through `key` (which,
            # for the stats-based default, needs annotation)
            for block in blocks.values():
                if len(block) > 1:
                    block.sort(key=key, reverse=True)
            pred = preds[j]
            rest = [t for t in blocks if t != pred]
            if len(rest) > 1:
                rest.sort(key=lambda t: key(blocks[t][0]), reverse=True)
            heads = ([pred] if pred in blocks else []) + rest
            out.append([r for t in heads for r in blocks[t]])
        return out

    # -- coverage / stats -------------------------------------------

    def coverage_matrix(self, data: DataRepresentation) -> np.ndarray:
        """`(n_rules, n_samples)` bool: [i, j] True iff rule i's body
        holds for row j, in isolation. Same definition for every model
        type.

        Row *i* is `self.rules[i]` -- positional, not by any rule id
        (rules carry none, and don't need one). `self.rules` is
        deterministic for every model type (a stored list, or built from
        stored sub-structures), so `coverage_matrix` and `self.rules`
        stay aligned as long as the model isn't structurally mutated
        between the two calls. Methods that use both (`annotate`,
        `covered_by`, `predict`) fetch them together, so they're safe;
        the same `Rule` object at two positions just gets two identical
        rows -- no de-dup."""
        rules = self.rules
        if not rules:
            return np.zeros((0, data.n_samples), dtype=bool)
        return np.vstack([r.covers_data(data) for r in rules])

    def is_disjoint(self, data: DataRepresentation) -> bool:
        """True iff no example in `data` is covered by more than one
        rule. Available on any `RuleModel` -- not just `DisjointRuleSet`
        -- specifically so it can be used as a precondition check on a
        plain `FlatRuleSet` (or any other collection) before deciding to
        treat/construct it as a `DisjointRuleSet`, not only to verify one
        after the fact."""
        cov = self.coverage_matrix(data)
        return bool(np.all(cov.sum(axis=0) <= 1))

    def is_exhaustive(self, data: DataRepresentation) -> bool:
        """True iff every example in `data` is covered by some rule --
        i.e. this model's rules are a full partition of `data`, not just
        (if also `is_disjoint`) pairwise disjoint."""
        cov = self.coverage_matrix(data)
        return bool(np.all(cov.sum(axis=0) >= 1))

    def coverage_space(self, data: DataRepresentation, positive_class: Any) -> np.ndarray:
        """`(n_rules, 2)` array of (n_covered_negatives, n_covered_positives)
        -- the classic ROC/coverage-space coordinates for refinement-graph
        plots (`pyrulearn.evaluation.coverage_space_plot`). Uses the raw
        `coverage_matrix`, the same for every model type."""
        if data.y is None:
            raise ValueError("data has no labels; coverage_space needs data.y")
        cov = self.coverage_matrix(data)
        pos_mask = data.y == positive_class
        pts = np.zeros((len(self.rules), 2), dtype=int)
        pts[:, 0] = cov[:, ~pos_mask].sum(axis=1)  # covered negatives (x)
        pts[:, 1] = cov[:, pos_mask].sum(axis=1)   # covered positives (y)
        return pts

    def _unique_mask(self, cov: np.ndarray) -> np.ndarray:
        """[i, j] True iff row j is uniquely rule i's, per this model's
        resolution. `RuleSet`/`RuleList` implement it via their
        `resolution`; concept-indexed and composite models override."""
        raise NotImplementedError

    def annotate(self, data: DataRepresentation, split: str = "data") -> None:
        """Compute and store this model's measured training/test-set
        performance under `split` (call again with another `split` name
        to keep train/test side by side) -- a `ModelStats` snapshot: a
        `ConfusionMatrix` from this model's own `predict(data)` vs
        `data.y` (`None` if `data.y` isn't available), plus
        `n_rules`/`n_conditions`. Read back with `stats(split=...)`.

        Only this model's *own* prediction is scored -- a composite's
        members (or a `_FlatRules`/`_ConceptIndexed` container's
        `SingleRule`s) each carry their own `stats`/`.annotate()` too,
        called separately against whatever data is relevant to them; this
        doesn't cascade down to them automatically (a `PairwiseModel`
        member, say, is only meaningful evaluated against its own pair's
        training subset, not the whole dataset this call was given)."""
        from .evaluation import ConfusionMatrix, ModelStats  # local: avoids a load-order cycle
                                                              # (models -> evaluation -> classifier ->
                                                              # models, via classifier.py's
                                                              # still-in-progress dissolution)
        rules = self.rules
        preds = self.predict(data)
        confusion = (ConfusionMatrix.from_predictions(data.y, preds, labels=self.labels)
                    if data.y is not None else None)
        self._stats[split] = ModelStats(
            n_rows=int(data.n_samples),
            confusion=confusion,
            n_rules=len(rules),
            n_conditions=sum(len(r.conditions) for r in rules),
        )

    def stats(self, data: Optional[DataRepresentation] = None, split: str = "data") -> Optional["ModelStats"]:
        """This model's measured performance/complexity under `split`, as
        a `ModelStats`. Annotates first if `data` is given. Returns
        `None` if never annotated and no `data` passed -- callers must
        handle `None`. Uniform across every subclass."""
        if data is not None:
            self.annotate(data, split)
        return self._stats.get(split)

    # -- default-prediction policy --------------------------------

    @property
    def default_prediction(self) -> Any:
        """Policy for rows no rule decides: a bare label, `None`
        (abstain), or a `DefaultPrediction` object. Reassigning discards
        the materialised `default_rule` and any stats on it."""
        return self._default_prediction

    @default_prediction.setter
    def default_prediction(self, value: Any) -> None:
        self._default_prediction = value
        self._default_rule = None

    def _default_target(self) -> Optional[Any]:
        dp = self._default_prediction
        return dp.constant_target if isinstance(dp, DefaultPrediction) else dp

    def _fallback(self, data: DataRepresentation) -> Callable[[int], Any]:
        dp = self._default_prediction
        if isinstance(dp, DefaultPrediction):
            rules = self.rules
            return lambda j: dp.predict(rules, data, j)
        return lambda j: dp

    @property
    def default_rule(self) -> Optional["SingleRule"]:
        """Read-only empty-body `SingleRule` view of a *constant*
        `default_prediction` (bare label, or a `DefaultPrediction`'s
        `constant_target`); `None` for a `None` policy or a per-row
        strategy. A `SingleRule`, like every other leaf this model holds,
        so it carries its own `.stats`/`.provenance` uniformly -- where
        fall-through stats/provenance live."""
        if self._default_rule is None:
            target = self._default_target()
            if target is None:
                return None
            self._default_rule = SingleRule(Rule([], target=target, dataspec=self._rule_dataspec()))
        return self._default_rule

    def _rule_dataspec(self) -> Optional[DataSpec]:
        """The `DataSpec` this model's rules are bound to (that of its first
        rule; `None` if it has none). A lazy pool overrides this so asking
        doesn't build a rule."""
        rules = self.rules
        return rules[0].dataspec if len(rules) else None

    # -- transforms -------------------------------------------------

    @abstractmethod
    def filter(self, target: Any) -> "RuleModel":
        """A model of the same type keeping only rules that predict
        `target` (recursing into blocks/members)."""
        raise NotImplementedError

    @abstractmethod
    def remap(self, new_dataspec: DataSpec) -> "RuleModel":
        """A model of the same type with every rule rebuilt against
        `new_dataspec` by feature name (see `Rule.remap`)."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self):
        return iter(self.rules)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.rules)} rules)"


# =============================================== storage mixins (traits) =======

def _as_single_rule(r: Union[Rule, "SingleRule"]) -> "SingleRule":
    """Normalize one rule-list item to `_FlatRules`' storage unit,
    `SingleRule`: a plain `Rule`/`WeightedRule` (what every importer and
    native learner still constructs) is wrapped; an already-`SingleRule`
    item (from `remap`/`filter`, or a caller passing one through
    directly) is kept as-is, not re-wrapped."""
    return r if isinstance(r, SingleRule) else SingleRule(r)


def _carry_provenance(source: "RuleModel", result: "RuleModel") -> "RuleModel":
    """`filter`/`remap`/a model->model `convert` all build a genuinely
    new container object -- but none of them change WHAT built the
    rules, only which ones are kept, which dataspec they're bound to, or
    how they're packaged; `RuleModel.__init__` resets a fresh object's
    `provenance` to `None` regardless, so callers copy the source's
    provenance across explicitly. Returns `result`, so a `filter`/`remap`
    method can wrap its return statement directly."""
    result.provenance = source.provenance
    return result


def annotate_rules(
    rules: Sequence[Union[Rule, "SingleRule"]], data: Optional[DataRepresentation], split: str = "data",
) -> List["SingleRule"]:
    """Wrap each rule as a `SingleRule` (via `_as_single_rule`, so an
    already-`SingleRule` item is kept, not re-wrapped) and, if `data` is
    given, populate its `stats(data, split)` against it -- the exact
    rows it was learned or read back from. `data=None` just wraps,
    stamping no stats -- the case for an importer's `parse`/`import_model`
    call made on its own, with no live training data to measure against
    (e.g. a demo/test parsing a captured rule dump).

    Called once, at the specific point a rule is produced or imported
    (a `RuleLearner`'s covering loop, `PyLORD`'s induction, a
    `RuleImporter`'s `parse`/`import_model`) -- never as a later,
    separate recursive walk over an already-built model. Two
    consequences follow from that: (1) `data`, when given, is always
    exactly what that call already had on hand for training/reading this
    rule (a decomposition stage's own row-sliced subset, a pair's own
    2-class rows, the exact rows a `fit()` round trip wrote out for an
    external tool), never a wider or implicitly-assumed superset -- there
    is deliberately no masking/`example_mask` parameter, since every
    existing caller already narrows scope by building a genuinely
    separate `DataRepresentation` (`select_rows`), not by masking a
    shared one; (2) a composite/decomposition model (`ConceptSet`,
    `PairwiseModel`, an `EnsembleModel` of per-tree `DisjointRuleSet`s)
    needs no stats pass of its own -- its members' rules already carry
    theirs from whichever call actually produced them.

    One subtlety for an ORDERED container (`RuleList`/`DecisionList`,
    `FirstMatch` resolution): the stats this stamps are each rule's raw,
    *standalone* coverage of `data` -- the same definition every rule
    everywhere else gets, and what a `RuleHeuristic` scores. That's a
    genuinely different number from "what this rule actually decides at
    predict time", which for a first-match list is restricted to rows no
    *earlier* rule already claimed (`RuleModel._unique_mask`). The two
    coincide only when nothing upstream can have already claimed a row
    (e.g. a tree's pairwise-disjoint leaves) -- see
    `pyrulearn.interfaces.weka`'s module docstring for a concrete
    case (JRip/PART's own printed per-rule support is covering-loop-
    scoped, not standalone) and its cross-check test.
    """
    out = []
    for r in rules:
        sr = _as_single_rule(r)
        if data is not None:
            sr.stats(data, split)
        out.append(sr)
    return out


def annotate_default_rule(model: "RuleModel", data: DataRepresentation, split: str = "data") -> "RuleModel":
    """If `model.default_rule` materializes to something, populate its
    stats too (against the same `data` its sibling rules were annotated
    against), then return `model` unchanged -- lets a producer method
    chain this straight onto its return statement. A no-op for a `None`
    policy or a non-constant `DefaultPrediction`, where `default_rule`
    is `None`."""
    if model.default_rule is not None:
        model.default_rule.stats(data, split)
    return model


class _FlatRules:
    """Mixin: holds a flat `SingleRule` list -- every item passed in is
    normalized via `_as_single_rule`, so each rule is a full model with
    its own `predict`/`stats`/`provenance` (full recursion down to the
    leaf level), while every existing caller that builds a plain `Rule`
    list (importers, native learners, tests) keeps working unchanged --
    the wrapping happens here, not at those call sites. Supplies `rules`,
    `add`, and flat `filter`/`remap`. Used across both resolution
    families -- `FlatRuleSet`/`ConceptModel`/`DisjointRuleSet` (`RuleSet`)
    and `DecisionList` (`RuleList`) -- so it's a trait, not a level."""

    def __init__(self, rules: Optional[Sequence[Rule]] = None, *, default_prediction: Any = None):
        super().__init__(default_prediction=default_prediction)  # -> RuleModel.__init__
        self._rules: List["SingleRule"] = [_as_single_rule(r) for r in rules] if rules else []

    @property
    def rules(self) -> List["SingleRule"]:
        return self._rules

    def add(self, rule: Rule) -> None:
        self._rules.append(_as_single_rule(rule))

    def filter(self, target: Any) -> "RuleModel":
        return _carry_provenance(self, type(self)(  # type: ignore[call-arg]
            [r for r in self._rules if r.target == target], **self._rebuild_kwargs()))

    def remap(self, new_dataspec: DataSpec) -> "RuleModel":
        return _carry_provenance(self, type(self)(  # type: ignore[call-arg]
            [r.remap(new_dataspec) for r in self._rules], **self._rebuild_kwargs()))


class _ConceptIndexed:
    """Mixin: holds a list of `ConceptModel` blocks. Supplies the pooled
    `rules` view, `concepts` / `concept_for` / `concept_labels`, and
    block-recursing `filter`/`remap`. Spans `RuleSet` (`ConceptSet`) and
    `RuleList` (`ConceptCascade`) -- a trait, not a level."""

    def __init__(self, concepts: Sequence["ConceptModel"], *, default_prediction: Any = None):
        super().__init__(default_prediction=default_prediction)  # -> RuleModel.__init__
        self.concepts: List[ConceptModel] = list(concepts)

    @property
    def rules(self) -> List[Rule]:
        return [r for c in self.concepts for r in c.rules]

    @property
    def concept_labels(self) -> List[Any]:
        """The label each `ConceptModel` defines, in block order."""
        return [c.label for c in self.concepts]

    def concept_for(self, label: Any) -> Optional["ConceptModel"]:
        for c in self.concepts:
            if c.label == label:
                return c
        return None

    def filter(self, target: Any) -> "RuleModel":
        c = self.concept_for(target)
        return _carry_provenance(self, type(self)(  # type: ignore[call-arg]
            [c] if c is not None else [], **self._rebuild_kwargs()))

    def remap(self, new_dataspec: DataSpec) -> "RuleModel":
        return _carry_provenance(self, type(self)(  # type: ignore[call-arg]
            [c.remap(new_dataspec) for c in self.concepts], **self._rebuild_kwargs()))

    def _concepts_cover_from(self, cov: np.ndarray) -> np.ndarray:
        """`(n_concepts, n_samples)` bool, sliced out of the pooled
        `cov`: concept c covers row j iff any of its rules does."""
        rows, r0 = [], 0
        for c in self.concepts:
            n = len(c.rules)
            rows.append(cov[r0:r0 + n].any(axis=0) if n else np.zeros(cov.shape[1], dtype=bool))
            r0 += n
        return np.vstack(rows) if rows else np.zeros((0, cov.shape[1]), dtype=bool)


# =============================================================== printing ===

def _rule_coverage_dicts(model: "RuleModel", data: DataRepresentation) -> Dict[int, dict]:
    """`id(rule) -> {"n_covered", "n_unique_covered", ...}` for every rule
    in `model` (plus `model.default_rule`, if any), computed fresh against
    `data` -- the `to_string(data=...)` coverage-decoration input. Deliberately
    NOT cached onto `rule.meta` (the old `RuleClassifier.annotate_coverage`
    did that): printing shouldn't have to be preceded by a separate
    annotation call, and shouldn't mutate a rule's own stats cache either
    -- this is a pure, throwaway read, recomputed on every call, same
    spirit as `annotate_rules`'s own on-demand philosophy.

    `n_covered`/`n_covered_by_class` are each rule's own *raw* coverage
    (independent of siblings); `n_unique_covered`/`n_unique_covered_by_class`
    use `model`'s own `_unique_mask` -- exactly `RuleClassifier.
    annotate_coverage`'s old two-tier definition, just computed inline."""
    rules = model.rules
    cov = model.coverage_matrix(data)
    unique = model._unique_mask(cov)
    classes = np.unique(data.y) if data.y is not None else ()

    def _stats(covered_mask: np.ndarray, unique_mask: np.ndarray) -> dict:
        stats: dict = {"n_covered": int(covered_mask.sum())}
        if data.y is not None:
            stats["n_covered_by_class"] = {
                c: int(np.sum(covered_mask & (data.y == c))) for c in classes
            }
            stats["n_unique_covered_by_class"] = {
                c: int(np.sum(unique_mask & (data.y == c))) for c in classes
            }
        stats["n_unique_covered"] = int(unique_mask.sum())
        return stats

    out = {id(r): _stats(cov[i], unique[i]) for i, r in enumerate(rules)}
    dr = model.default_rule
    if dr is not None:
        everything = np.ones(data.n_samples, dtype=bool)
        reaches_default = ~cov.any(axis=0) if len(rules) else everything
        out[id(dr)] = _stats(everything, reaches_default)
    return out


def _decorate(rule: Rule, text: str, coverage: Optional[dict]) -> str:
    """Wrap one rule's already-rendered `text` with a trailing coverage
    comment from `coverage` (one entry of `_rule_coverage_dicts`'s
    result, or `None` to decorate nothing): ``% (n_covered/n_errors)
    [n_uniquely_covered/n_unique_errors]`` -- the classic C4.5/RIPPER
    "(covered/errors)" rule-quality notation, with a second bracket for
    this codebase's own unique-coverage extension. Error counts need
    labels to compute and a target to check correctness against; where
    either is missing, just the bare counts.

    No weight decoration here -- `WeightedRule.to_string` already
    renders its own weight natively, in every format."""
    if coverage is None:
        return text
    by_class = coverage.get("n_covered_by_class")
    unique_by_class = coverage.get("n_unique_covered_by_class")
    if by_class is not None and rule.target is not None:
        n_errors = coverage["n_covered"] - by_class.get(rule.target, 0)
        n_unique_errors = coverage["n_unique_covered"] - unique_by_class.get(rule.target, 0)
        suffix = (f"  % ({coverage['n_covered']}/{n_errors}) "
                  f"[{coverage['n_unique_covered']}/{n_unique_errors}]")
    else:
        suffix = f"  % ({coverage['n_covered']}) [{coverage['n_unique_covered']}]"
    return f"{text}{suffix}"


def _dnf_lines(rules: Sequence[Rule], ascii: bool, dec: Callable[[Rule, str], str]) -> List[str]:
    """One class's rules, collapsed into a single DNF expression for
    "logic"-format printing: each rule's `Rule.logic_body`, parenthesized
    and decorated via `dec`, on its own line (``∨``-prefixed after the
    first) so mixed ``∧``/``∨``/``¬`` stays unambiguous, followed by a
    trailing ``→ target`` line."""
    arrow_sym = "->" if ascii else "→"
    or_sym = "OR" if ascii else "∨"
    parts = [dec(r, f"({r.logic_body(ascii=ascii)})") for r in rules]
    lines = [f"    {parts[0]}"] + [f"  {or_sym} {p}" for p in parts[1:]]
    target = rules[0].target if rules else None
    lines.append(f"  {arrow_sym} {target}")
    return lines


# ============================================ resolution family bases (ABCs) ===

class RuleSet(RuleModel):
    """*Unordered* collection of rules: several rules covering one row
    are reconciled by `self.resolution` (a `Combine` over a
    `combiner`, or an `Exclusive` disjointness assumption) -- position
    never matters. Abstract: a concrete subclass supplies `rules` (via
    `_FlatRules` or `_ConceptIndexed`).
    """

    resolution: Resolution = Combine("max")

    def predict(
        self, data: DataRepresentation, combiner: Optional[Union[str, RuleCombiner]] = None,
    ) -> np.ndarray:
        """`combiner`, if given, resolves multiply-covered rows for this
        call only (a `RuleCombiner` instance or a shortcut string) --
        doesn't touch `self.resolution`/`self.combiner`, so trying
        several combiners on the same fitted model needs no mutation
        (and no risk of one comparison's override leaking into another).
        Omit it to use the model's own `self.resolution` (settable via
        `self.combiner` for a `FlatRuleSet`/`ConceptSet`)."""
        cov = self.coverage_matrix(data)
        resolution = self.resolution if combiner is None else Combine(combiner)
        return resolution.predict(self.rules, cov, self._fallback(data), data.n_samples)

    def _unique_mask(self, cov: np.ndarray) -> np.ndarray:
        return self.resolution.unique_mask(cov, self.rules)

    def to_string(
        self, fmt: Optional[str] = None, ascii: bool = False,
        data: Optional[DataRepresentation] = None,
    ) -> str:
        """Render every rule, grouped by target label -- one section per
        label, headed by ``% class: <target>``. For "logic" format, each
        label's rules collapse into a single DNF expression (see
        `_dnf_lines`); other formats list rules under the label header
        unmerged. `default_rule`, if set, gets its own trailing
        ``% default`` section.

        Every rule renders with the *same* resolved format (explicit
        `fmt=` if given, else `Rule.DEFAULT_FORMAT`), regardless of any
        individual rule's own `default_fmt`. A `WeightedRule` renders its
        own weight natively as part of that.

        `data`, if given, decorates every rule with a trailing
        ``% (n_covered/n_errors) [n_unique_covered/n_unique_errors]``
        comment, computed fresh against it (see `_rule_coverage_dicts`)
        -- no separate annotation call needed first.
        """
        resolved = fmt if fmt is not None else Rule.DEFAULT_FORMAT
        coverage = _rule_coverage_dicts(self, data) if data is not None else {}
        dec = lambda r, text: _decorate(r, text, coverage.get(id(r)))  # noqa: E731
        sections = []
        for t in sorted({r.target for r in self.rules}, key=_sortkey):
            group = [r for r in self.rules if r.target == t]
            header = f"% class: {t}"
            if resolved == "logic":
                body = "\n".join(_dnf_lines(group, ascii=ascii, dec=dec))
            else:
                body = "\n".join(dec(r, r.to_string(fmt=resolved, ascii=ascii)) for r in group)
            sections.append(f"{header}\n{body}")
        if self.default_rule is not None:
            default_text = dec(self.default_rule, self.default_rule.to_string(fmt=resolved, ascii=ascii))
            sections.append(f"% default\n{default_text}")
        return "\n\n".join(sections)

    def to_rulelist(
        self, key: Optional[Callable[[Rule], Any]] = None, reverse: bool = True,
    ) -> "DecisionList":
        """Convert to an ordered `DecisionList` by sorting `self.rules`
        via `pyrulearn.evaluation.sort_rules` (carrying `default_prediction`
        over unchanged). Defaults to descending Laplace-on-measured-stats
        -- the same default `sort_rules` itself uses; pass `key=`/`reverse=`
        for a different ordering (e.g. by `Rule.length()`).

        Unlike `flatruleset_to_decision_list` (a lossless structural
        repackaging that keeps insertion order, registered as a
        `can_convert` model converter), this picks a genuinely *new*
        order, so it's a plain method, not a converter -- call it
        directly when you want the ranking."""
        from .evaluation import sort_rules  # local: avoids a load-order cycle
        return DecisionList(sort_rules(self.rules, by=key, descending=reverse),
                            default_prediction=self._default_prediction)


class RuleList(RuleModel):
    """*Ordered* collection: rules (or, for `ConceptCascade`, concepts)
    are tried in order and the first match wins. Rule-level order --
    unrelated to `Rule.ordered` (condition order within one rule).
    Abstract: a concrete subclass supplies `rules`.
    """

    resolution: Resolution = FirstMatch()

    def predict(self, data: DataRepresentation) -> np.ndarray:
        cov = self.coverage_matrix(data)
        return self.resolution.predict(self.rules, cov, self._fallback(data), data.n_samples)

    def _unique_mask(self, cov: np.ndarray) -> np.ndarray:
        return self.resolution.unique_mask(cov, self.rules)

    def coverage_path(self, data: DataRepresentation, positive_class: Any) -> np.ndarray:
        """Cumulative (negatives, positives) coverage path for this list,
        shape ``(len(self.rules) + 1, 2)``, starting at ``(0, 0)``: point
        i is the sum of the first i rules' *unique* (fired) coverage --
        exactly what's been decided after trying the first i rules, in
        order. The final point is the list's total coverage, which falls
        short of the dataset's full ``(N, P)`` only if some rows reach no
        rule (no catch-all at the end)."""
        if data.y is None:
            raise ValueError("data has no labels; coverage_path needs data.y")
        cov = self.coverage_matrix(data)
        unique = self._unique_mask(cov)
        pos_mask = data.y == positive_class
        rules = self.rules
        pts = np.zeros((len(rules) + 1, 2), dtype=int)
        for i in range(len(rules)):
            pts[i + 1, 0] = pts[i, 0] + int(np.sum(unique[i] & ~pos_mask))
            pts[i + 1, 1] = pts[i, 1] + int(np.sum(unique[i] & pos_mask))
        return pts

    def to_string(
        self, fmt: Optional[str] = None, ascii: bool = False,
        data: Optional[DataRepresentation] = None,
    ) -> str:
        """Render this decision list in order -- no label-grouping, since
        order (not shared target) is what decision-list semantics
        actually depend on. For "logic" format, rendered as
        if/elif/else pseudocode, since order = priority there exactly
        matches `predict`'s own first-match-wins semantics; other
        formats list rules sequentially in list order, with
        `default_rule` (if set) appended as a trailing ``% default``
        section (an if/elif chain's ``else`` already says "default" for
        "logic", so no extra label is needed there).

        `data`, if given, decorates every rule the same way as
        `RuleSet.to_string` -- see there."""
        resolved = fmt if fmt is not None else Rule.DEFAULT_FORMAT
        coverage = _rule_coverage_dicts(self, data) if data is not None else {}
        dec = lambda r, text: _decorate(r, text, coverage.get(id(r)))  # noqa: E731
        rules = self.rules
        if resolved == "logic":
            arrow_sym = "->" if ascii else "→"
            lines = []
            for i, r in enumerate(rules):
                kw = "if  " if i == 0 else "elif"
                text = dec(r, f"{r.logic_body(ascii=ascii)} {arrow_sym} {r.target}")
                lines.append(f"{kw} {text}")
            if self.default_rule is not None:
                default_text = dec(self.default_rule, f"{arrow_sym} {self.default_rule.target}")
                lines.append(f"else {default_text}")
            return "\n".join(lines)
        lines = [dec(r, r.to_string(fmt=resolved, ascii=ascii)) for r in rules]
        if self.default_rule is not None:
            default_text = dec(self.default_rule, self.default_rule.to_string(fmt=resolved, ascii=ascii))
            lines.append(f"% default\n{default_text}")
        return "\n".join(lines)


# ================================================================= concrete ===

class SingleRule(RuleSet):
    """One rule, wrapped as a model so it carries stats and predicts on
    its own: its head where the body holds, else `default_prediction`. A
    degenerate `RuleSet` (`Exclusive` resolution) -- and the hierarchy's
    base case: every `_FlatRules`-based container stores its rules as
    `SingleRule`s, but `SingleRule` itself holds a raw `Rule` directly
    rather than going through that same normalization (wrapping its own
    rule in a `SingleRule` would recurse forever).

    Reads of anything not defined here (`.target`, `.conditions`,
    `.dataspec`, `.pos`, `.covers_data`, ...) fall through to the
    wrapped `Rule` via `__getattr__` -- a `SingleRule` is meant to be a
    drop-in stand-in for a bare `Rule` wherever existing coverage/combining
    code (`coverage_matrix`, `Resolution`, `pyrulearn.combiners`) or
    rule-consuming code reads one. `.weight` is the one attribute that
    needs to stay *writable* through the wrapper (mutating a fitted
    model's rule weights in place), hence the explicit property.
    `.to_string` is also explicit, not forwarded: `RuleSet.to_string`
    (inherited otherwise) groups `self.rules` by target and calls
    `.to_string` on each -- for a `SingleRule`, whose `self.rules == [self]`,
    that would recurse forever, so it's overridden to render the wrapped
    `Rule` directly instead."""

    resolution = Exclusive()

    def __init__(self, rule: Rule, *, default_prediction: Any = None):
        super().__init__(default_prediction=default_prediction)  # -> RuleModel.__init__
        self._rule: Rule = rule

    @property
    def rule(self) -> Rule:
        return self._rule

    @property
    def rules(self) -> List["SingleRule"]:
        return [self]

    def set_stats_from_counts(
        self, covered: Dict[Any, int], totals: Dict[Any, int], split: str = "data",
    ) -> "SingleRule":
        """Store this rule's measured stats from counts its producer
        already knows -- `covered[c]`: rows the rule covers with true label
        `c`; `totals[c]`: rows of the whole data with label `c` -- instead
        of `annotate`'s predict-over-the-data pass. Produces exactly the
        `ModelStats` `annotate` would (see `ConfusionMatrix.
        from_rule_counts`), for a rule with no default prediction. Use it
        where the counts fall out of the construction anyway (a CAR
        miner's per-class supports, a tree leaf's class counts): stamping
        a large pool this way is ~10x faster than `annotate_rules`.
        Returns `self`."""
        from .evaluation import ConfusionMatrix, ModelStats  # local: same load-order reason as `annotate`
        if self._default_prediction is not None:
            raise ValueError("set_stats_from_counts assumes no default prediction (the rule abstains elsewhere)")
        self._stats[split] = ModelStats(
            n_rows=int(sum(totals.values())),
            confusion=ConfusionMatrix.from_rule_counts(self._rule.target, covered, totals),
            n_rules=1,
            n_conditions=len(self._rule.conditions),
        )
        return self

    @property
    def weight(self) -> Optional[float]:
        return getattr(self._rule, "weight", None)

    @weight.setter
    def weight(self, value: float) -> None:
        self._rule.weight = value

    def __getattr__(self, name: str) -> Any:
        if name == "_rule":  # not set yet (e.g. mid-construction) -- don't recurse
            raise AttributeError(name)
        return getattr(self._rule, name)

    def _rebuild_kwargs(self) -> Dict[str, Any]:
        return {"default_prediction": self._default_prediction}

    def filter(self, target: Any) -> "RuleModel":
        if self._rule.target == target:
            return self  # already exactly this -- no need to rebuild (and lose stats/provenance)
        return _carry_provenance(self, FlatRuleSet([], default_prediction=self._default_prediction))

    def remap(self, new_dataspec: DataSpec) -> "SingleRule":
        return _carry_provenance(self, SingleRule(
            self._rule.remap(new_dataspec), default_prediction=self._default_prediction))

    def to_string(
        self, fmt: Optional[str] = None, ascii: bool = False,
        data: Optional[DataRepresentation] = None,
    ) -> str:
        """Renders the wrapped `Rule` directly -- a lone rule needs no
        per-target grouping or DNF collapsing (see the class docstring
        for why this can't just inherit `RuleSet.to_string`). `data`, if
        given, decorates with this rule's own coverage (see
        `RuleSet.to_string`) -- "unique" trivially equals "raw" here,
        there being no sibling rule to share coverage with."""
        resolved = fmt if fmt is not None else Rule.DEFAULT_FORMAT
        text = self._rule.to_string(fmt=resolved, ascii=ascii)
        if data is None:
            return text
        coverage = _rule_coverage_dicts(self, data)
        return _decorate(self._rule, text, coverage.get(id(self)))


class ConceptModel(_FlatRules, RuleSet):
    """The definition of one **concept**: rules that all share one `label`
    (their head). `predict` returns `label` for a covered row and
    `default_prediction` otherwise -- `None` ("not this concept") by
    default, but a concept learned as one side of a binary problem
    typically carries a concrete negative fallback (a bare label or a
    `MajorityClass`). The building block of `ConceptSet` and
    `ConceptCascade`, and the classical concept-learning target (AQ,
    version spaces)."""

    resolution = Combine("list")  # all rules share a head -> any covering rule -> label

    def __init__(
        self,
        rules: Optional[Sequence[Rule]] = None,
        label: Any = None,
        *,
        default_prediction: Any = None,
    ):
        rules = list(rules) if rules else []
        heads = {r.target for r in rules}
        if len(heads) > 1:
            raise ValueError(f"ConceptModel rules must share one head; got {sorted(map(str, heads))}")
        self.label = label if label is not None else (next(iter(heads)) if heads else None)
        if heads and self.label not in heads:
            raise ValueError(f"label={self.label!r} does not match the rules' head {heads}")
        super().__init__(rules, default_prediction=default_prediction)

    def _rebuild_kwargs(self) -> Dict[str, Any]:
        return {"label": self.label, "default_prediction": self._default_prediction}

    def filter(self, target: Any) -> "ConceptModel":
        return _carry_provenance(self, ConceptModel(
            self._rules if target == self.label else [], label=self.label,
            default_prediction=self._default_prediction))

    def remap(self, new_dataspec: DataSpec) -> "ConceptModel":
        return _carry_provenance(self, ConceptModel(
            [r.remap(new_dataspec) for r in self._rules], label=self.label,
            default_prediction=self._default_prediction))


class FlatRuleSet(_FlatRules, RuleSet):
    """A plain bag of rules with mixed heads, resolved by one global
    `combiner` (`pyrulearn.combiners.RuleCombiner` or a shortcut string;
    default `"max"`). The generic `RuleSet` -- the counterpart to
    `DeepModel`. Insertion order is kept but unused for prediction, which
    is what lets it convert *either* to a `DecisionList` (adopt the
    order) or a `ConceptSet` (drop it). Direct successor of the old flat
    `RuleSet`."""

    def __init__(
        self,
        rules: Optional[Sequence[Rule]] = None,
        *,
        default_prediction: Any = None,
        combiner: Union[str, RuleCombiner] = "max",
    ):
        super().__init__(rules, default_prediction=default_prediction)
        self.resolution = Combine(combiner)

    @property
    def combiner(self) -> Union[str, RuleCombiner]:
        return self.resolution.combiner  # type: ignore[attr-defined]

    @combiner.setter
    def combiner(self, value: Union[str, RuleCombiner]) -> None:
        self.resolution = Combine(value)

    def _rebuild_kwargs(self) -> Dict[str, Any]:
        return {"default_prediction": self._default_prediction, "combiner": self.combiner}


class RuleView(_SequenceABC):
    """A lazy, read-only, ordered selection of a `PooledRuleSet`'s rules --
    what `PooledRuleSet.rules` returns. `pool.rules` is the full view;
    `precedence_sorted()`, `for_target()`, `head_per_target()` and slicing
    return further views over the *same* pool (and the same rule cache),
    never copies of rules. See `PooledRuleSet`."""

    def __init__(self, pool: "PooledRuleSet", rows: Optional[np.ndarray] = None):
        self._pool = pool
        #: pool row numbers this view exposes, in view order; `None` = every row in pool order
        self._rows = rows

    def rows(self) -> np.ndarray:
        return np.arange(self._pool._n, dtype=np.int64) if self._rows is None else self._rows

    def __len__(self) -> int:
        return self._pool._n if self._rows is None else len(self._rows)

    def _row(self, k: int) -> int:
        return k if self._rows is None else int(self._rows[k])

    def __getitem__(self, k: Union[int, slice]):
        if isinstance(k, slice):
            return RuleView(self._pool, self.rows()[k])
        n = len(self)
        if not -n <= k < n:
            raise IndexError(k)
        return self._pool._rule(self._row(k % n))

    def __iter__(self) -> Iterator["SingleRule"]:
        for k in range(len(self)):
            yield self._pool._rule(self._row(k))

    def iter_transient(self) -> Iterator["SingleRule"]:
        """Each rule freshly built and *not* cached -- memory stays flat over
        a pass through millions of rules. The yielded rules are equal in
        content to `self[k]` but not the same objects; don't mutate them
        expecting the pool to remember."""
        for k in range(len(self)):
            yield self._pool._build(self._row(k))

    # -- vectorized selection: no rule is built ---------------------------

    def precedence_sorted(self) -> "RuleView":
        """CBA's precedence order (confidence desc, support desc, fewer
        conditions first; ties keep view order) -- the same order
        `pyrulearn.learners.associative.sort_by_measured_precedence` gives a
        plain list, computed from the count columns alone."""
        rows = self.rows()
        p = self._pool
        tp = p._tp(rows)
        conf = tp / p._itemset_support(rows)
        return RuleView(p, rows[np.lexsort((p._lengths(rows), -tp, -conf))])

    def for_target(self, target: Any) -> "RuleView":
        code = self._pool._code_of(target)
        rows = self.rows()
        if code is None:
            return RuleView(self._pool, rows[:0])
        return RuleView(self._pool, rows[self._pool._target_codes[rows] == code])

    def targets(self) -> List[Any]:
        """The distinct class labels this view's rules predict."""
        codes = np.unique(self._pool._target_codes[self.rows()])
        return [self._pool._classes[int(c)] for c in codes]

    def head_per_target(self, k: int) -> "RuleView":
        """The first `k` rules of each class in view order, kept in the view's
        own (interleaved) order -- what a per-class cap (IDS's `rule_cutoff`)
        needs."""
        rows = self.rows()
        codes = self._pool._target_codes[rows]
        keep = [np.flatnonzero(codes == c)[:k] for c in np.unique(codes)]
        positions = np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=np.int64)
        return RuleView(self._pool, rows[positions])

    def __repr__(self) -> str:
        return f"RuleView({len(self)} of {self._pool._n} pooled rules)"


class PooledRuleSet(FlatRuleSet):
    """A `FlatRuleSet` whose (possibly huge) rule pool is held as a few flat
    numpy columns instead of one `SingleRule` object per rule.

    A mined class-association-rule pool can hold hundreds of thousands of
    rules. As eager `SingleRule`s (each a full `RuleModel` carrying its own
    stats) that costs ~1.1 KB per rule; the same information -- the rule
    body, its class, and its per-class covered-row counts, which is
    everything its measured stats derive from -- is ~35 bytes per rule in
    columns. So the pool stores columns, and a `SingleRule` is **built the
    first time someone looks at it** (and then cached, so it stays one
    persistent object: writing `rule.weight`, stamping another `split`'s
    stats, or keying a dict on `id(rule)` all keep working). Every rule you
    do see is completely filled -- the same `ModelStats` `annotate` would
    produce, stamped from the counts by `SingleRule.set_stats_from_counts`.

    It is a `FlatRuleSet` in every other respect (`isinstance` holds;
    `predict`, `default_prediction`, `combiner`, `stats`, ... unchanged).
    Build one from mined CARs with `from_cars`. Two access paths:

    - **Ordinary**: `pool.rules[i]`, `len(pool.rules)`, `for r in pool.rules`
      -- like a list of `SingleRule`s, except `.rules` is a read-only
      `Sequence` (see `RuleModel.rules`), and touching every rule builds
      (and caches) every rule.
    - **Streaming**, for one-pass consumers (`pyrulearn.learners.associative`'s
      `coverage_select`, CMAR's significance filter, IDS's candidate cap):
      `RuleView.iter_transient()` builds each rule without caching it, and
      the vectorized `precedence_sorted()` / `for_target()` /
      `head_per_target()` reorder and filter *without building anything* --
      from the count columns alone. CBA-CB then materializes only the
      prefix of the precedence order it actually scans.

    `add` is not supported; `filter(target)` is a vectorized column
    selection returning another `PooledRuleSet`; `remap` materializes and
    returns an eager `FlatRuleSet` (a remapped rule has a different body, so
    there is nothing to share).
    """

    def __init__(
        self, dataspec: DataSpec, items: np.ndarray, offsets: np.ndarray, target_codes: np.ndarray,
        class_counts: np.ndarray, classes: Sequence[Any], totals: Dict[Any, int], *,
        default_prediction: Any = None, combiner: Union[str, RuleCombiner] = "max",
    ):
        super().__init__([], default_prediction=default_prediction, combiner=combiner)
        self._spec = dataspec
        self._items = items                    # int32, all rule bodies concatenated
        self._offsets = offsets                # int64, rule i's body = items[offsets[i]:offsets[i+1]]
        self._target_codes = target_codes      # int16, index into `classes`
        self._counts = class_counts            # int32 (n, K): covered rows per true class
        self._classes: List[Any] = list(classes)
        self._code = {c: i for i, c in enumerate(self._classes)}
        self._totals = dict(totals)
        self._n = len(target_codes)
        self._cache: Dict[int, SingleRule] = {}
        self._view = RuleView(self)

    @classmethod
    def from_cars(cls, cars: Sequence[Any], dataspec: DataSpec, totals: Dict[Any, int],
                  **kw) -> "PooledRuleSet":
        """The pool of `pyrulearn.learners.associative.CAR`s `cars`, which must
        carry their `class_counts`. `totals[c]` is the number of rows of the
        whole data with label `c` (its keys fix the class order)."""
        classes = list(totals)
        code = {c: i for i, c in enumerate(classes)}
        n = len(cars)
        offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.fromiter((len(c.items) for c in cars), dtype=np.int64, count=n), out=offsets[1:])
        items = np.fromiter(chain.from_iterable(c.items for c in cars), dtype=np.int32, count=int(offsets[-1]))
        target_codes = np.fromiter((code[c.target] for c in cars), dtype=np.int16, count=n)
        counts = np.empty((n, len(classes)), dtype=np.int32)
        seen: Dict[int, List[int]] = {}    # the per-itemset counts dict is shared by all its CARs
        for i, c in enumerate(cars):
            row = seen.get(id(c.class_counts))
            if row is None:
                row = seen[id(c.class_counts)] = [c.class_counts.get(k, 0) for k in classes]
            counts[i] = row
        return cls(dataspec, items, offsets, target_codes, counts, classes, totals, **kw)

    # -- columns ----------------------------------------------------------

    def _tp(self, rows: np.ndarray) -> np.ndarray:
        return self._counts[rows, self._target_codes[rows]].astype(np.int64)

    def _itemset_support(self, rows: np.ndarray) -> np.ndarray:
        return self._counts[rows].sum(axis=1, dtype=np.int64)

    def _lengths(self, rows: np.ndarray) -> np.ndarray:
        return (self._offsets[rows + 1] - self._offsets[rows]).astype(np.int64)

    def _code_of(self, target: Any) -> Optional[int]:
        return self._code.get(target)

    # -- building rules ---------------------------------------------------

    def _build(self, row: int) -> "SingleRule":
        body = tuple(self._items[self._offsets[row]:self._offsets[row + 1]].tolist())
        target = self._classes[int(self._target_codes[row])]
        covered = dict(zip(self._classes, self._counts[row].tolist()))
        return SingleRule(Rule(body, target=target, dataspec=self._spec)).set_stats_from_counts(
            covered, self._totals)

    def _rule(self, row: int) -> "SingleRule":
        rule = self._cache.get(row)
        if rule is None:
            rule = self._cache[row] = self._build(row)
        return rule

    # -- FlatRuleSet surface ----------------------------------------------

    @property
    def rules(self) -> RuleView:
        return self._view

    def _rule_dataspec(self) -> DataSpec:
        return self._spec

    def add(self, rule: Rule) -> None:
        raise TypeError("a PooledRuleSet is a fixed columnar pool; build a FlatRuleSet to add rules")

    def take(self, rows: np.ndarray) -> "PooledRuleSet":
        """A new pool of just `rows` (pool row numbers), in that order."""
        rows = np.asarray(rows, dtype=np.int64)
        lens = self._lengths(rows)
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        items = (np.concatenate([self._items[self._offsets[r]:self._offsets[r + 1]] for r in rows])
                 if len(rows) else np.empty(0, dtype=np.int32))
        return PooledRuleSet(self._spec, items, offsets, self._target_codes[rows], self._counts[rows],
                             self._classes, self._totals, **self._rebuild_kwargs())

    def filter(self, target: Any) -> "PooledRuleSet":
        return _carry_provenance(self, self.take(self._view.for_target(target).rows()))

    def remap(self, new_dataspec: DataSpec) -> FlatRuleSet:
        return _carry_provenance(self, FlatRuleSet(
            [r.remap(new_dataspec) for r in self._view], **self._rebuild_kwargs()))


class DisjointRuleSet(_FlatRules, RuleSet):
    """A `RuleSet` whose rules are *assumed* pairwise mutually exclusive
    -- a decision tree's leaves, whose path conditions partition the
    space by construction, so `predict` never has a genuine tie to
    resolve. Disjointness is declared, not enforced; check it with
    `is_disjoint(data)` (once that moves here from analysis)."""

    resolution = Exclusive()


class DecisionList(_FlatRules, RuleList):
    """A linearly ordered decision list (RIPPER/CN2-style sequential
    covering): rules are tried in `self.rules` order and the first that
    matches a row decides it."""


class ConceptSet(_ConceptIndexed, RuleSet):
    """One `ConceptModel` per label, unordered. A row covered by a single
    concept gets that concept's label; a row covered by several is
    resolved by `combiner` over the union of those concepts' covering
    rules. For an order-independent `combiner` (`"max"`, `"vote"`, the
    distribution combiners) prediction matches a `FlatRuleSet` with the
    same rules -- the difference is then purely structural (nested
    concepts, 3-level stats). Rows no concept covers fall back to
    `default_prediction`."""

    def __init__(
        self,
        concepts: Sequence["ConceptModel"],
        *,
        default_prediction: Any = None,
        combiner: Union[str, RuleCombiner] = "max",
    ):
        super().__init__(concepts, default_prediction=default_prediction)
        self.resolution = Combine(combiner)

    @property
    def combiner(self) -> Union[str, RuleCombiner]:
        return self.resolution.combiner  # type: ignore[attr-defined]

    @combiner.setter
    def combiner(self, value: Union[str, RuleCombiner]) -> None:
        self.resolution = Combine(value)

    def _rebuild_kwargs(self) -> Dict[str, Any]:
        return {"default_prediction": self._default_prediction, "combiner": self.combiner}

    @classmethod
    def from_rules(
        cls,
        rules: Sequence[Rule],
        *,
        default_prediction: Any = None,
        combiner: Union[str, RuleCombiner] = "max",
    ) -> "ConceptSet":
        """Group a flat rule list into one `ConceptModel` per head."""
        by_head: Dict[Any, List[Rule]] = {}
        for r in rules:
            by_head.setdefault(r.target, []).append(r)
        concepts = [ConceptModel(rs, label=h) for h, rs in by_head.items()]
        return cls(concepts, default_prediction=default_prediction, combiner=combiner)


class ConceptCascade(_ConceptIndexed, RuleList):
    """Ordered `ConceptModel`s: a row is tested against them in order and
    the first concept that fires decides it (its label); a row no concept
    claims falls back to `default_prediction` (typically the last,
    un-modelled label). The "peeling" one-vs-rest / nested decomposition
    -- concept k only sees rows concepts 1..k-1 didn't take. Since a
    concept's rules all share its label, rule-level first-match over the
    concept-ordered pooled `rules` *is* concept-level first-match, so
    `predict` is inherited from `RuleList`. Was `ClassOrderedRuleList`.
    """

    def _unique_mask(self, cov: np.ndarray) -> np.ndarray:
        # every rule of the *first firing* concept gets credit (within a
        # concept the rules are an unordered disjunction), unlike plain
        # FirstMatch which would credit only the first such rule.
        rules = self.rules
        um = np.zeros_like(cov)
        if not rules:
            return um
        ccov = self._concepts_cover_from(cov)
        claimed = np.zeros(cov.shape[1], dtype=bool)
        r0 = 0
        for c, concept in enumerate(self.concepts):
            n = len(concept.rules)
            first = ccov[c] & ~claimed
            for k in range(n):
                um[r0 + k] = cov[r0 + k] & first
            claimed |= ccov[c]
            r0 += n
        return um


# ================================================================= composite ===

class CompositeModel(RuleModel):
    """A model whose prediction aggregates the predictions of member
    `RuleModel`s rather than a flat rule list. `members` are never merged
    -- prediction always routes through them individually; `rules` is a
    read-only pooled view (recursive), for inspection/complexity only.
    `add` and in-place rule mutation are unavailable.

    `filter`/`remap` recurse into the members via `_with_members`, which
    each subclass implements to rebuild itself with a new member list and
    its own config carried over."""

    def __init__(self, members: Sequence[RuleModel], *, default_prediction: Any = None):
        super().__init__(default_prediction=default_prediction)
        self.members: List[RuleModel] = list(members)

    @property
    def rules(self) -> List[Rule]:
        return [r for m in self.members for r in m.rules]

    def add(self, rule: Rule) -> None:
        raise TypeError(
            f"{type(self).__name__} has no flat rule list to add to -- its rules live in its members"
        )

    def _unique_mask(self, cov: np.ndarray) -> np.ndarray:
        # members overlap by design (different sub-problems) -- "uniquely
        # one rule's" isn't meaningful here; report none.
        return np.zeros_like(cov)

    def _with_members(self, members: Sequence[RuleModel]) -> "CompositeModel":
        raise NotImplementedError

    def filter(self, target: Any) -> "CompositeModel":
        return _carry_provenance(self, self._with_members([m.filter(target) for m in self.members]))

    def remap(self, new_dataspec: DataSpec) -> "CompositeModel":
        return _carry_provenance(self, self._with_members([m.remap(new_dataspec) for m in self.members]))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.members)} members, {len(self.rules)} rules)"


class EnsembleModel(CompositeModel):
    """Members combined by a flat vote: each member predicts
    independently, then a per-row plurality vote -- optionally weighted
    by `member_weights` (one scalar per member) -- picks the label. Rows
    every member abstains on fall back to `default_prediction`. The
    umbrella for bagging / boosting-style rule ensembles."""

    def __init__(
        self,
        members: Sequence[RuleModel],
        *,
        default_prediction: Any = None,
        member_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(members, default_prediction=default_prediction)
        if member_weights is not None and len(member_weights) != len(self.members):
            raise ValueError(
                f"member_weights has {len(member_weights)} entries, expected {len(self.members)}"
            )
        self.member_weights = (None if member_weights is None
                               else np.asarray(member_weights, dtype=float))

    def _with_members(self, members: Sequence[RuleModel]) -> "EnsembleModel":
        return EnsembleModel(members, default_prediction=self._default_prediction,
                             member_weights=self.member_weights)

    def predict(self, data: DataRepresentation) -> np.ndarray:
        cols = [np.asarray(m.predict(data)) for m in self.members]
        fb = self._fallback(data)
        out = np.empty(data.n_samples, dtype=object)
        for j in range(data.n_samples):
            tally: Dict[Any, float] = {}
            first: Dict[Any, int] = {}
            for k, col in enumerate(cols):
                p = col[j]
                if p is None:
                    continue
                w = float(self.member_weights[k]) if self.member_weights is not None else 1.0
                tally[p] = tally.get(p, 0.0) + w
                first.setdefault(p, k)
            out[j] = (max(tally, key=lambda l: (tally[l], -first[l])) if tally else fb(j))
        return out


# -------------------------------------------------- pairwise voting combiners ---

def _pairwise_weight(rule: Rule) -> float:
    """A rule's confidence for pairwise soft voting: `Laplace` on its
    own measured stats (matching `HeuristicMaxCombiner`'s predict-time
    default and `sort_rules`' inspection-time default -- the same
    "rank/weigh by measured reliability" operation everywhere), or the
    neutral 0.5 when it carries no stats at all (0.0 would dump the
    whole vote on the *other* class)."""
    stats = _rule_stats(rule)
    if stats is None:
        return 0.5
    from .heuristics import Laplace  # local: see annotate()'s own lazy-import note
    return float(Laplace().score(stats))


class PairwiseVote(NamedTuple):
    """One member's verdict on one row: `predicted` is `positive`,
    `negative`, or `None` (abstained). `weight` in ``[0, 1]`` is its
    confidence in `predicted` -- the deciding rule's weight for
    `WeightedVote`, the member's pair accuracy for `AccuracyWeightedVote`,
    ignored by `MajorityVote`."""
    predicted: Any
    positive: Any
    negative: Any
    weight: float = 1.0


class PairwiseCombiner(ABC):
    """Turns a row's `PairwiseVote`s into a per-label score vector.
    `scores` is the reusable core (classification argmaxes it, ranking
    argsorts it); `decide` = argmax + tie-break, or `None` when nothing
    scored above zero. `weight_source` (`None`/`"rule"`/`"member"`) tells
    `PairwiseModel` whether/how to fill `PairwiseVote.weight`."""

    needs_weights: bool = False
    weight_source: Optional[str] = None

    @abstractmethod
    def scores(self, votes: Sequence[PairwiseVote], labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decide(self, votes, labels, label_priors=None) -> Optional[Any]:
        s = self.scores(votes, labels)
        best = s.max() if len(s) else 0.0
        if best <= 0:
            return None
        tied = [c for c, sc in zip(labels, s) if sc >= best]
        return tied[0] if len(tied) == 1 else self._break_tie(tied, votes, label_priors)

    def _break_tie(self, tied, votes, label_priors):
        return sorted(tied, key=_sortkey)[0]


class _VotingCombiner(PairwiseCombiner):
    """Shared `tie_break` for `MajorityVote`/`WeightedVote`: ``"direct"``
    (the tied labels' own duel; default), ``"prior"`` (most frequent in
    `label_priors`), ``"first"`` (sorts first), or a callable
    ``f(tied, votes, priors) -> label``."""

    def __init__(self, tie_break: Union[str, Callable[..., Any]] = "direct"):
        if not callable(tie_break) and tie_break not in ("direct", "prior", "first"):
            raise ValueError(f"tie_break must be 'direct'/'prior'/'first' or callable, got {tie_break!r}")
        self.tie_break = tie_break

    def _break_tie(self, tied, votes, label_priors):
        tb = self.tie_break
        if callable(tb):
            return tb(tied, votes, label_priors)
        if tb in ("direct", "prior"):
            tied_set = set(tied)
            if tb == "direct":
                pos = {c: i for i, c in enumerate(tied)}
                sub = np.zeros(len(tied), dtype=float)
                for v in votes:
                    if v.predicted in tied_set and v.positive in tied_set and v.negative in tied_set:
                        sub[pos[v.predicted]] += 1.0
                if sub.max() > 0 and (sub == sub.max()).sum() == 1:
                    return tied[int(sub.argmax())]
            if label_priors is not None:
                return sorted(tied, key=lambda c: (-label_priors.get(c, 0.0), _sortkey(c)))[0]
        return sorted(tied, key=_sortkey)[0]


class MajorityVote(_VotingCombiner):
    """One hard vote per deciding member; `scores` = vote count per label."""

    def scores(self, votes, labels):
        idx = {c: i for i, c in enumerate(labels)}
        s = np.zeros(len(labels), dtype=float)
        for v in votes:
            if v.predicted is not None and v.predicted in idx:
                s[idx[v.predicted]] += 1.0
        return s


class WeightedVote(_VotingCombiner):
    """Soft voting: each member's ``p_ij`` (the deciding rule's weight,
    clamped to ``[0, 1]``) goes to the predicted label, ``1 - p_ij`` to
    the other. `weight_source = "rule"`."""

    needs_weights = True
    weight_source = "rule"

    def scores(self, votes, labels):
        idx = {c: i for i, c in enumerate(labels)}
        s = np.zeros(len(labels), dtype=float)
        for v in votes:
            if v.predicted is None or v.predicted not in idx:
                continue
            other = v.positive if v.predicted == v.negative else v.negative
            w = float(v.weight)
            w = 0.5 if w != w else min(1.0, max(0.0, w))
            s[idx[v.predicted]] += w
            if other in idx:
                s[idx[other]] += 1.0 - w
        return s


class AccuracyWeightedVote(WeightedVote):
    """Like `WeightedVote`, but ``p_ij`` is the *member's* accuracy on its
    own pair sub-problem (one scalar per member, recorded at fit time),
    not the deciding rule's weight. `weight_source = "member"`."""

    weight_source = "member"


_PAIRWISE_COMBINER_SHORTCUTS: Dict[str, Callable[[], PairwiseCombiner]] = {
    "vote": MajorityVote,
    "weighted_vote": WeightedVote,
    "accuracy_vote": AccuracyWeightedVote,
}


def _resolve_pairwise_combiner(combiner: Union[str, PairwiseCombiner]) -> PairwiseCombiner:
    if isinstance(combiner, PairwiseCombiner):
        return combiner
    try:
        return _PAIRWISE_COMBINER_SHORTCUTS[combiner]()
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown pairwise combiner {combiner!r}; choose from "
            f"{sorted(_PAIRWISE_COMBINER_SHORTCUTS)} or pass a PairwiseCombiner"
        ) from None


class PairwiseModel(CompositeModel):
    """Round-robin: one binary member per unordered label pair (Fürnkranz,
    *Round Robin Classification*, JMLR 2002), each voting for one of its
    two labels. `members` are ``(label_a, label_b, submodel)`` triples;
    `combiner` (`PairwiseCombiner` or a shortcut -- ``"vote"`` /
    ``"weighted_vote"`` / ``"accuracy_vote"``) turns the per-row votes
    into a label, or `None` → `default_prediction`.

    Swap `combiner` on a fitted model with no retraining -- `fit` records
    both the rule weights (`"weighted_vote"`) and, via `member_weights`,
    the members' pair accuracies (`"accuracy_vote"`)."""

    def __init__(
        self,
        members: Sequence[Tuple[Any, Any, RuleModel]],
        *,
        combiner: Union[str, PairwiseCombiner] = "vote",
        default_prediction: Any = None,
        labels: Optional[Sequence[Any]] = None,
        label_priors: Optional[Dict[Any, float]] = None,
        member_weights: Optional[Sequence[float]] = None,
    ):
        self._triples = [tuple(m) for m in members]
        super().__init__([m[2] for m in self._triples], default_prediction=default_prediction)
        self.combiner = _resolve_pairwise_combiner(combiner)
        self.label_priors = label_priors
        if labels is not None:
            self._labels = list(labels)
        else:
            seen: List[Any] = []
            for a, b, _ in self._triples:
                for x in (a, b):
                    if x not in seen:
                        seen.append(x)
            self._labels = sorted(seen, key=_sortkey)
        if member_weights is not None and len(member_weights) != len(self._triples):
            raise ValueError(
                f"member_weights has {len(member_weights)} entries, expected {len(self._triples)}"
            )
        self.member_weights = (None if member_weights is None
                               else np.asarray(member_weights, dtype=float))

    @property
    def labels(self) -> List[Any]:
        return list(self._labels)

    @property
    def pairs(self) -> List[Tuple[Any, Any]]:
        return [(a, b) for a, b, _ in self._triples]

    def _with_members(self, members: Sequence[RuleModel]) -> "PairwiseModel":
        rebuilt = [(a, b, sub) for (a, b, _), sub in zip(self._triples, members)]
        return PairwiseModel(rebuilt, combiner=self.combiner,
                             default_prediction=self._default_prediction, labels=self._labels,
                             label_priors=self.label_priors, member_weights=self.member_weights)

    def predict(self, data: DataRepresentation) -> np.ndarray:
        src = (self.combiner.weight_source
               if getattr(self.combiner, "needs_weights", False) else None)
        if src == "member" and self.member_weights is None:
            raise ValueError(f"{type(self.combiner).__name__} needs member_weights (record them "
                             "at fit time)")
        if src == "rule":
            cols = [(a, b, sub.covered_by(data)) for a, b, sub in self._triples]
        else:
            cols = [(a, b, np.asarray(sub.predict(data))) for a, b, sub in self._triples]

        classes = np.asarray(self._labels, dtype=object)
        fb = self._fallback(data)
        out = np.empty(data.n_samples, dtype=object)
        for j in range(data.n_samples):
            votes: List[PairwiseVote] = []
            for k, (a, b, col) in enumerate(cols):
                if src == "rule":
                    deciders = col[j]
                    if not deciders:
                        votes.append(PairwiseVote(None, a, b)); continue
                    top = deciders[0]
                    p = top.target if top.target in (a, b) else None
                    votes.append(PairwiseVote(p, a, b, _pairwise_weight(top)))
                else:
                    p = col[j]
                    p = p if p in (a, b) else None
                    w = float(self.member_weights[k]) if src == "member" else 1.0
                    votes.append(PairwiseVote(p, a, b, w))
            decided = self.combiner.decide(votes, classes, self.label_priors)
            out[j] = decided if decided is not None else fb(j)
        return out


class DeepModel(CompositeModel):
    """Members wired into a dependency DAG -- some members consume other
    members' predictions as extra input features ("stacking"). The
    counterpart to `FlatRuleSet`.

    **STUB**: structure only. `dependencies` maps a member index to the
    indices of the members feeding it; `predict` is not implemented --
    the feature-augmentation plumbing (how an upstream prediction becomes
    a downstream feature) is still to be designed."""

    def __init__(
        self,
        members: Sequence[RuleModel],
        *,
        dependencies: Optional[Dict[int, Sequence[int]]] = None,
        default_prediction: Any = None,
    ):
        super().__init__(members, default_prediction=default_prediction)
        self.dependencies: Dict[int, List[int]] = {
            k: list(v) for k, v in (dependencies or {}).items()
        }

    def _with_members(self, members: Sequence[RuleModel]) -> "DeepModel":
        return DeepModel(members, dependencies=self.dependencies,
                         default_prediction=self._default_prediction)

    def predict(self, data: DataRepresentation) -> np.ndarray:
        raise NotImplementedError(
            "DeepModel.predict is not implemented yet -- the member-wiring / "
            "feature-augmentation design is pending (this is a stub)."
        )


# ------------------------------------------------------------------ converters ---

def flatruleset_to_decision_list(rs: FlatRuleSet) -> DecisionList:
    """Adopt the `FlatRuleSet`'s insertion order as decision-list order.
    Prediction changes: first-match instead of combiner resolution."""
    return _carry_provenance(rs, DecisionList(list(rs.rules), default_prediction=rs.default_prediction))


def flatruleset_to_conceptset(rs: FlatRuleSet) -> ConceptSet:
    """Group the `FlatRuleSet`'s rules by head into per-label concepts;
    keeps the same `combiner`, so an order-independent combiner leaves
    prediction unchanged."""
    return _carry_provenance(rs, ConceptSet.from_rules(
        list(rs.rules), default_prediction=rs.default_prediction, combiner=rs.combiner
    ))


def conceptset_to_flatruleset(cs: ConceptSet) -> FlatRuleSet:
    """Drop the per-concept structure; same rules, same combiner."""
    return _carry_provenance(cs, FlatRuleSet(
        list(cs.rules), default_prediction=cs.default_prediction, combiner=cs.combiner))


def ensemblemodel_to_flatruleset(ens: EnsembleModel, *, combiner: str = "vote") -> FlatRuleSet:
    """Pool every member's rules into one flat bag, resolved by a single
    `combiner` (default `"vote"` -- an unweighted plurality vote over all
    covering rules, the closest flat analogue of the ensemble's own
    per-member vote). Lossy: the per-member grouping, any
    `member_weights`, and each member's own resolution (e.g. a tree's
    `Exclusive`) are dropped -- a forest's soft, per-tree-distribution
    vote is *not* reproduced. Use it when a downstream consumer needs a
    single rule list and an approximate vote is acceptable."""
    return _carry_provenance(ens, FlatRuleSet(
        list(ens.rules), default_prediction=ens.default_prediction, combiner=combiner))


def conceptcascade_to_decision_list(cascade: ConceptCascade) -> DecisionList:
    """Flatten the cascade's concepts, in order, into one decision list.
    Prediction is preserved (first concept to fire == first rule to
    fire, concept-by-concept)."""
    return _carry_provenance(cascade, DecisionList(
        list(cascade.rules), default_prediction=cascade.default_prediction))


#: (source type, target type) -> converter. `can_convert` checks membership.
_CONVERTERS: Dict[Tuple[type, type], Callable[[Any], RuleModel]] = {
    (FlatRuleSet, DecisionList): flatruleset_to_decision_list,
    (FlatRuleSet, ConceptSet): flatruleset_to_conceptset,
    (ConceptSet, FlatRuleSet): conceptset_to_flatruleset,
    (ConceptCascade, DecisionList): conceptcascade_to_decision_list,
    (EnsembleModel, FlatRuleSet): ensemblemodel_to_flatruleset,
}


def can_convert(src: type, dst: type) -> bool:
    """Whether a direct model→model converter exists from `src` to
    `dst`. (No transitive closure -- chain converters explicitly.)"""
    return any((cls, dst) in _CONVERTERS for cls in src.__mro__)


def convert(model: RuleModel, dst: type) -> RuleModel:
    """Apply the `src → dst` converter. Raises if none is registered. A
    subclass of a registered source type (e.g. `pyrulearn.pool.PooledRuleSet`,
    a `FlatRuleSet`) uses its parent's converter -- the nearest class in the
    MRO that has one."""
    fn = next((_CONVERTERS[(cls, dst)] for cls in type(model).__mro__ if (cls, dst) in _CONVERTERS), None)
    if fn is None:
        raise TypeError(f"no converter from {type(model).__name__} to {dst.__name__}")
    return fn(model)
