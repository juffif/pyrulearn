"""
pyrulearn.combiners
=======================

`RuleCombiner`: strategies `RuleSet.predict` consults when more than one
rule covers the same example, to combine them down to one predicted
target. Named after the classifier-combination-rule terminology from
ensemble learning (majority vote, max, weighted vote, ...) rather than
"tie-break", since these don't just pick a single winner among competing
candidates -- several of them aggregate across every covering rule.

Two intermediate layers, matching two different kinds of per-rule
information a combiner can use:

- `HeuristicCombiner` -- a `pyrulearn.heuristics.RuleHeuristic` run
  against each covering rule's own measured stats (`SingleRule.stats()`'s
  `ConfusionMatrix`, rotated to the rule's own target via `rule_stats`).
  `HeuristicMaxCombiner`, `HeuristicVoteCombiner`. This replaced an
  earlier design (`WeightCombiner`, reading a scalar `meta['weight']`
  precomputed once at fit time by whatever heuristic the *learner*
  happened to choose) -- now the combiner computes the score itself,
  directly from stats, at predict time; which heuristic to use is a
  choice made *here*, not baked into the model beforehand. **Requires**
  every covering rule to have measured stats -- raises `ValueError`
  otherwise (see `_heuristic_score`'s docstring), rather than silently
  falling back to a scalar `meta['weight']`: a stale or hand-set weight
  is exactly the kind of quiet wrong-answer this whole stats redesign
  exists to rule out. Annotate first (`pyrulearn.models.annotate_rules`,
  or a `fit()` call / an importer's `data=` argument, which already do).
- `DistributionCombiner` -- each rule's full per-class counts: its own
  measured stats (the `ConfusionMatrix` column for its target, via
  `ConfusionMatrix.predicted_as`). **Requires** every covering rule to
  have measured stats, same as `HeuristicCombiner` -- raises `ValueError`
  otherwise (e.g. a decision-tree leaf imported via `import_model`/
  `from_random_forest` with no `data=` has nothing to score from; pass
  `data=` so `_stamp_rule_stats` measures it). Two further, orthogonal
  axes distinguish its four
  concrete combiners: *which* per-rule numbers are used (`Micro`: pool
  raw counts across covering rules, so larger leaves count for more;
  `Macro`: normalize each rule's own counts to proportions first, so
  every rule counts equally regardless of leaf size -- the same
  micro/macro-averaging distinction used for multi-class F1 scores),
  and *how* several rules' numbers are combined (`Vote`: sum, total
  support per class; `Max`: the classic ensemble "max rule", the best
  single supporting rule's confidence per class). `MicroVoteCombiner`,
  `MacroVoteCombiner`, `MicroMaxCombiner`, `MacroMaxCombiner` --
  `MacroVoteCombiner` is the one that actually matches how
  `sklearn.ensemble.RandomForestClassifier.predict()` itself averages
  trees' probabilities.

`ListCombiner`/`CountVoteCombiner` fit neither category (list order and
a plain unweighted vote, respectively -- no weight, heuristic, or
distribution involved, so nothing to be missing), so they're direct
`RuleCombiner` children.

`RuleSet.predict`'s `combiner` argument accepts either a `RuleCombiner`
instance directly, or one of the built-in shortcuts' string names
("list"/"max"/"vote"/"micro_vote"/"macro_vote"/"micro_max"/"macro_max")
-- see `_resolve_combiner`.

**Ties.** When a combiner's own criterion leaves several classes level
(`HeuristicMaxCombiner` first applies a step of its own, see there), every
combiner breaks the tie the same way, never by rule position: the class
that is more frequent in the training data wins (read from the covering
rules' frozen training stats), then the class that sorts first (the
order a printed model's class legend uses). Only `ListCombiner` is
order-based -- that is its definition. `describe()` gives the one-line
description a printed model shows as its conflict resolution; the tie
convention isn't printed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from .rule import Rule

if TYPE_CHECKING:
    from .evaluation import RuleStats
    from .heuristics import RuleHeuristic, Score


def _trivial_target(rules: Sequence[Rule], covering: Sequence[int]) -> Optional[Any]:
    """If every covering rule agrees on the target, that's the answer
    outright -- there's no actual disagreement to resolve, so nothing
    needs scoring (and nothing to raise over, even if some of those
    rules have no measured stats: requiring stats is about resolving a
    genuine tie, not a precondition for every single prediction). `None`
    if covering rules disagree and a real combine has to happen."""
    targets = {rules[i].target for i in covering}
    return next(iter(targets)) if len(targets) == 1 else None


def _rule_stats(rule: Rule) -> Optional["RuleStats"]:
    """This rule's own measured `RuleStats` -- its `.stats()` snapshot's
    `ConfusionMatrix`, rotated to its own target via `rule_stats` -- or
    `None` if it was never annotated."""
    stats_fn = getattr(rule, "stats", None)
    ms = stats_fn() if callable(stats_fn) else None
    if ms is None or ms.confusion is None:
        return None
    return ms.confusion.rule_stats(rule.target)


def _heuristic_score(rule: Rule, heuristic: "RuleHeuristic") -> "Score":
    """A rule's score: `heuristic.score(...)` on its own measured stats.
    Raises `ValueError` if the rule has none -- a hand-built rule, or an
    importer's bare `parse`/`import_model` call made without `data=`
    (see `pyrulearn.models.annotate_rules`) -- rather than silently
    reading a stale/hand-set `meta['weight']`. Annotate the rule first
    (a `fit()` call and a `fit()`-round-trip importer already do).

    Shared infrastructure: used by `HeuristicCombiner` (predict-time
    ranking) and `pyrulearn.evaluation._rule_score_fn`'s `by=None`
    default (`sort_rules`'s inspection-time ranking) -- the same "rank by
    measured reliability" operation in both places, just at different
    times."""
    stats = _rule_stats(rule)
    if stats is None:
        raise ValueError(
            f"{type(rule).__name__}(target={rule.target!r}) has no measured stats -- "
            "annotate it first (see pyrulearn.models.annotate_rules, or fit()/an "
            "importer's data= argument)"
        )
    return heuristic.score(stats)


def _training_frequencies(rules: Sequence[Rule], covering: Sequence[int]) -> Dict[Any, int]:
    """How many training rows each class has, read from the first
    covering rule that carries frozen training stats (a rule's confusion
    matrix counts every training row by its true label); empty if none
    does."""
    from .evaluation import ABSTAIN  # local: evaluation imports this module's users
    for i in covering:
        stats_fn = getattr(rules[i], "stats", None)
        ms = stats_fn() if callable(stats_fn) else None
        if ms is not None and ms.confusion is not None:
            cm = ms.confusion
            totals = cm.counts.sum(axis=1)
            return {c: int(n) for c, n in zip(cm.labels, totals) if c is not ABSTAIN}
    return {}


def _label_sortkey(c: Any):
    """Same total order as `pyrulearn.models._sortkey` (numbers numeric,
    everything else by its string) -- the class legend's order."""
    return (0, c) if isinstance(c, (int, float)) else (1, str(c))


def _break_tie(tied: Sequence[Any], rules: Sequence[Rule], covering: Sequence[int]) -> Any:
    """The shared tie-break (see the module docstring): among the `tied`
    classes, the one most frequent in the training data, then the one
    that sorts first."""
    if len(tied) == 1:
        return tied[0]
    freq = _training_frequencies(rules, covering)
    return min(tied, key=lambda c: (-freq.get(c, 0), _label_sortkey(c)))


def _argmax_classes(totals: Dict[Any, Any]) -> List[Any]:
    """Every class with the highest value in `totals`."""
    best = max(totals.values())
    return [c for c, v in totals.items() if v == best]


def _predicted_distribution(rule: Rule) -> Optional[Dict[Any, float]]:
    """The true-label distribution among rows this rule predicted its
    own target for -- the `ConfusionMatrix` column for `rule.target`,
    read via `.stats()` -- or `None` if the rule was never annotated."""
    stats_fn = getattr(rule, "stats", None)
    ms = stats_fn() if callable(stats_fn) else None
    if ms is None or ms.confusion is None:
        return None
    return ms.confusion.predicted_as(rule.target)


class RuleCombiner(ABC):
    """Base for strategies that combine several simultaneously-covering
    rules' targets into one prediction."""

    @abstractmethod
    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        """`covering` is the (always non-empty) indices into `rules`
        that cover one example; return the target to predict for it."""
        raise NotImplementedError

    def describe(self) -> str:
        """One line naming how this combiner resolves a conflict -- what a
        printed model shows (``% conflict resolution: ...``)."""
        return type(self).__name__


class ListCombiner(RuleCombiner):
    """Pick whichever covering rule comes first in `rules`' own list
    order -- the only combiner that depends on rule position rather
    than any score."""

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        return rules[covering[0]].target

    def describe(self) -> str:
        return "first matching rule"


class CountVoteCombiner(RuleCombiner):
    """Plain majority vote across every covering rule's target -- each
    covering rule counts as exactly one vote, no weight, heuristic, or
    distribution involved (see `HeuristicVoteCombiner` for a
    heuristic-weighted version). The motivating case is a random forest
    imported as one `RuleSet` (each tree contributes one always-firing
    rule per example, so exactly k rules cover every example for a
    k-tree forest) -- this reproduces forest-style *hard* voting (see
    `DistributionCombiner`'s `MacroVoteCombiner` for *soft*,
    probability-based voting, matching sklearn's own
    `RandomForestClassifier.predict()`).

    Ties go to the shared tie-break (module docstring).
    """

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        tally: Dict[Any, int] = {}
        for i in covering:
            tally[rules[i].target] = tally.get(rules[i].target, 0) + 1
        return _break_tie(_argmax_classes(tally), rules, covering)

    def describe(self) -> str:
        return "vote (one vote per covering rule)"


# -- heuristic-on-stats ---------------------------------------------------

class HeuristicCombiner(RuleCombiner):
    """Base for combiners that score each covering rule by running a
    `pyrulearn.heuristics.RuleHeuristic` against its own measured stats
    (`_heuristic_score`) -- the dataset-measured replacement for a
    scalar `meta['weight']` precomputed once at fit time. Matches this
    codebase's weight-vs-stats split: `WeightedRule.weight` stays
    declarative (part of the model, no dataset, ProbLog-facing);
    scoring covering rules for a *prediction* is measured, so it belongs
    here, computed directly from stats, with which heuristic to use
    chosen at combine time rather than baked into the model beforehand.

    Every covering rule MUST have measured stats -- `_heuristic_score`
    raises `ValueError` otherwise, rather than quietly falling back to
    whatever `meta['weight']` happens to be lying around (stale, unset,
    or hand-set to something unrelated to this heuristic -- exactly the
    kind of silent wrong answer this stats redesign exists to prevent).
    Annotate rules first: a `fit()` call and a `fit()`-round-trip
    importer already do; `pyrulearn.models.annotate_rules` for anything
    else (a hand-built `RuleSet`, or an importer's bare `parse`/
    `import_model` call made without `data=`).

    `heuristic` defaults to `Laplace()` -- the same default this
    codebase's learners used for their own (now superseded) weight
    stamping, for the same reason: bounded, prior-free, sane on
    near-zero coverage.
    """

    def __init__(self, heuristic: Optional["RuleHeuristic"] = None):
        #: `None` resolves to `Laplace()` lazily, in `_score` -- not
        #: here, so building a module-level default instance (this
        #: module's own `_COMBINER_SHORTCUTS`) never has to import
        #: `.heuristics` at combiners.py's own load time (it imports
        #: `.evaluation`, which imports `.classifier`, which imports
        #: `.combiners` -- a genuine cycle if resolved eagerly).
        self.heuristic = heuristic

    def _heuristic(self) -> "RuleHeuristic":
        if self.heuristic is not None:
            return self.heuristic
        from .heuristics import Laplace
        return Laplace()

    def _score(self, rule: Rule) -> "Score":
        return _heuristic_score(rule, self._heuristic())


class HeuristicMaxCombiner(HeuristicCombiner):
    """The classic ensemble "max rule": pick the covering rule with the
    highest heuristic score -- for each candidate class, this is
    equivalent to taking its most-confident covering rule and then
    comparing across classes, since the globally highest-scoring rule
    trivially wins its own class's comparison too.

    Ties: when several covering rules share the top score but predict
    different classes, those tied rules vote -- the class with the most of
    them wins (a rule with a lower score doesn't count). If that is level
    too, the shared tie-break applies (module docstring: training
    frequency, then class order). Never rule position.
    """

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        scores = {i: self._score(rules[i]) for i in covering}
        best = max(scores.values())
        tally: Dict[Any, int] = {}
        for i, sc in scores.items():
            if sc == best:
                tally[rules[i].target] = tally.get(rules[i].target, 0) + 1
        return _break_tie(_argmax_classes(tally), rules, covering)

    def describe(self) -> str:
        return f"max {self._heuristic()!r}"


class HeuristicVoteCombiner(HeuristicCombiner):
    """Weighted vote across *every* covering rule's target -- unlike
    `HeuristicMaxCombiner`/`ListCombiner`, which each pick one covering
    rule and use its target, this tallies all of them, each one's vote
    weighted by its heuristic score (see `CountVoteCombiner` for the
    unweighted version). Ties go to the shared tie-break (module
    docstring).
    """

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        tally: Dict[Any, float] = {}
        for i in covering:
            r = rules[i]
            tally[r.target] = tally.get(r.target, 0.0) + self._score(r)
        return _break_tie(_argmax_classes(tally), rules, covering)

    def describe(self) -> str:
        return f"vote weighted by {self._heuristic()!r}"


# -- distribution-based ---------------------------------------------------

def _combine_class_scores(
    scores_per_rule: Sequence[Dict[Any, float]], op: Callable[[List[float]], float],
    rules: Sequence[Rule], covering: Sequence[int],
) -> Any:
    """Shared aggregation for `DistributionCombiner`s: `op` is `sum`
    (vote -- total support per class) or `max` (max rule -- best single
    supporting rule's confidence per class); returns the argmax class
    over the combined per-class scores, ties to the shared tie-break."""
    from .evaluation import ABSTAIN  # local: evaluation imports this module's users
    per_class: Dict[Any, List[float]] = {}
    for scores in scores_per_rule:
        for cls, v in scores.items():
            if cls is not ABSTAIN:
                per_class.setdefault(cls, []).append(v)
    totals = {cls: op(vs) for cls, vs in per_class.items()}
    return _break_tie(_argmax_classes(totals), rules, covering)


class DistributionCombiner(RuleCombiner):
    """Base for combiners that use each covering rule's full per-class
    counts, as opposed to `HeuristicCombiner`'s single scalar score.

    `_raw_counts`/`_normalized` are the two "which numbers" variants
    concrete subclasses read from -- pooled as-is (`Micro`) or
    normalized per rule first (`Macro`) -- combined via
    `_combine_class_scores` with either `sum` (`Vote`) or `max` (`Max`).
    """

    @staticmethod
    def _raw_counts(rule: Rule) -> Dict[Any, float]:
        """This rule's per-class counts: its own measured stats (the
        `ConfusionMatrix` column for its target, via
        `_predicted_distribution`). Raises `ValueError` if the rule has
        none -- annotate it first (see `HeuristicCombiner`'s docstring)."""
        counts = _predicted_distribution(rule)
        if counts is not None:
            return counts
        raise ValueError(
            f"{type(rule).__name__}(target={rule.target!r}) has no measured stats -- "
            "DistributionCombiner needs one (see pyrulearn.models.annotate_rules, or "
            "fit()/an importer's data= argument)"
        )

    @classmethod
    def _normalized(cls, rule: Rule) -> Dict[Any, float]:
        counts = cls._raw_counts(rule)
        total = sum(counts.values())
        return {c: v / total for c, v in counts.items()} if total > 0 else counts


class MicroVoteCombiner(DistributionCombiner):
    """Pool every covering rule's *raw* class counts (larger leaves
    contribute proportionally more, since they're not normalized away
    first), sum them, argmax."""

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        scores = [self._raw_counts(rules[i]) for i in covering]
        return _combine_class_scores(scores, sum, rules, covering)

    def describe(self) -> str:
        return "sum of covered class counts"


class MacroVoteCombiner(DistributionCombiner):
    """Normalize each covering rule's own class counts to proportions
    first (every rule counts equally regardless of leaf size), sum,
    argmax -- this is what actually matches sklearn's own
    `RandomForestClassifier.predict()` mechanism: each tree contributes
    one normalized probability vector, not weighted by how many
    training examples happened to land in that leaf.
    """

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        scores = [self._normalized(rules[i]) for i in covering]
        return _combine_class_scores(scores, sum, rules, covering)

    def describe(self) -> str:
        return "sum of covered class proportions"


class MicroMaxCombiner(DistributionCombiner):
    """For each class, the highest *raw* count any single covering rule
    assigns it; argmax across classes."""

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        scores = [self._raw_counts(rules[i]) for i in covering]
        return _combine_class_scores(scores, max, rules, covering)

    def describe(self) -> str:
        return "max covered class count"


class MacroMaxCombiner(DistributionCombiner):
    """The classic ensemble "max rule" in its full, per-class-probability
    form: for each class, the highest proportion any single covering
    rule (normalized) assigns it; argmax across classes. Distinct from
    `HeuristicMaxCombiner`, which picks one whole rule globally rather
    than comparing per-class."""

    def resolve(self, rules: Sequence[Rule], covering: Sequence[int]) -> Any:
        trivial = _trivial_target(rules, covering)
        if trivial is not None:
            return trivial
        scores = [self._normalized(rules[i]) for i in covering]
        return _combine_class_scores(scores, max, rules, covering)

    def describe(self) -> str:
        return "max covered class proportion"


_COMBINER_SHORTCUTS: Dict[str, RuleCombiner] = {
    "list": ListCombiner(),
    "max": HeuristicMaxCombiner(),
    "vote": CountVoteCombiner(),
    "micro_vote": MicroVoteCombiner(),
    "macro_vote": MacroVoteCombiner(),
    "micro_max": MicroMaxCombiner(),
    "macro_max": MacroMaxCombiner(),
}


def _resolve_combiner(combiner: Union[str, RuleCombiner]) -> RuleCombiner:
    if isinstance(combiner, RuleCombiner):
        return combiner
    try:
        return _COMBINER_SHORTCUTS[combiner]
    except KeyError:
        raise ValueError(
            f"Unknown combiner {combiner!r}; choose from {sorted(_COMBINER_SHORTCUTS)} "
            "or pass a RuleCombiner instance"
        ) from None
