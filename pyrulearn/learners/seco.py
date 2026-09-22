"""
pyrulearn.learners.seco
===========================

A generic separate-and-conquer (SeCo) rule learner, built from five
composable building blocks (see Fürnkranz's SeCo survey for the general
shape this follows):

1. **search for a single rule** -- `RuleSearch`. `BeamSearch` (any
   `RuleHeuristic` except `GainHeuristic`, `beam_width` at least 1) is
   the general-purpose default; `HillClimbing` is its single-lineage
   greedy special case (`beam_width` 1 plus "stop at a local maximum of
   the heuristic"); `GainAscentHillClimbing` is a further subclass, the
   only search that can run a `GainHeuristic` (e.g. `FoilGain`) -- a
   gain score is only meaningful against its own immediate parent, and a
   single lineage is the one place "only one parent, ever" holds (see
   `BeamSearch`'s docstring for why it can't rank gain scores safely at
   any beam width).
2. **a search heuristic** -- already existed before this module:
   `pyrulearn.heuristics.RuleHeuristic`/`RuleStats`/`GainHeuristic`/
   `DeltaGain`, reused as-is. `BeamSearch`/`HillClimbing`/
   `GainAscentHillClimbing`'s own
   `filtering=`/`stopping=` criteria (`pyrulearn.pruning.
   PrePruningCriterion` and friends) live in their own module too,
   since they're equally reusable outside a SeCo-shaped loop.
3. **prepare for learning a single rule** -- `SingleRulePreparation`
   (`NoSplit` default; `GrowPruneSplit` for RIPPER/IREP-style behavior).
4. **post-process a single rule** -- `SingleRulePostProcessing`
   (`NoPostProcessing` default; `ReducedErrorPruning` for RIPPER/IREP-
   style behavior).
5. **initialize the search space** -- `SearchSpaceInit`
   (`EmptyRuleAllFeatures` default; `FeatureSubset` for a named subset;
   `SeedExample` for AQ's single-example seed -- generalize one still-
   uncovered positive rather than start from the fully-open empty rule).

These five compose into `SingleRuleLearner`. The outer `SeCo`
sequential-covering loop (subclasses `pyrulearn.learners.
NativeRuleLearner`) calls it repeatedly: learn one rule, remove every
example it covers (both classes -- see `SeCo`'s own docstring) from
further consideration, repeat until `stop_covering` (a
`pyrulearn.pruning.PrePruningCriterion`, same as `filtering=`/
`stopping=`, but consulted only via its bare `evaluate()` -- see
`SeCo`'s own docstring) says stop, the single-rule search returns
nothing (`filtering` rejected everything it reached), or no positive
examples remain uncovered. `RuleSet`-level
preparation/post-processing (e.g. RIPPER's whole-set optimization
phase, revisiting rules already added in light of each other) is a
separate, higher-level pair of hooks around that outer loop, distinct
from building blocks 3/4 above (which only ever see one rule in
isolation) -- not yet implemented; `SeCo` as built so far always
accepts every rule it finds (down to the covering criterion) without
ever reconsidering earlier ones.

Throughout, `target_class` is a fixed parameter -- every building block
only ever finds/scores rules for one predetermined class. Multi-class is
handled *around* this machinery, via the `fit(data, model=...)` switcher
(see `pyrulearn.learners.RuleLearner.fit`) -- `fit(data)` with no
`target_class` builds `_MULTICLASS_DEFAULT` (`ConceptSet`, one-vs-rest,
for the SeCo family):
- a binary decomposition (`model=ConceptSet | ConceptCascade |
  PairwiseModel` -- one-vs-rest / ordered peeling / round robin, via
  `DecomposingLearner`; `pyrulearn.learners.multiclass` wraps these as thin sugar)
  fitting copies of itself with `target_class` set; or
- `model=FlatRuleSet` (`AQR`'s own `_MULTICLASS_DEFAULT`):
  `SeCo._seed_covering_fit`, one covering loop over all classes at once,
  each rule seeded on a random uncovered example and headed with that
  example's own label -- AQ's multi-class covering.

A single global "best rule for any class" search (the early-CN2 entropy
heuristic) is a different, deferred thing; when built it would also wrap
this machinery from the outside (calling `SingleRuleLearner.
learn_one_rule` once per candidate class and picking the best via a
heuristic comparable across classes), not rework anything here.

"Masks, not splits": every one of these building blocks that needs to
restrict which examples are in scope (a grow/prune split, "which
examples are still uncovered") does so via a boolean `example_mask` over
`data`'s rows, not by constructing a smaller
`BooleanDataRepresentation`. This is already how `RuleStats.from_rule`
and `Rule.specialize` work; the pieces built here just thread that mask
through rather than reintroducing a splitting approach.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..models import (
    ConceptCascade, ConceptModel, ConceptSet, FlatRuleSet, MajorityClass, SingleRule,
    annotate_default_rule, annotate_rules,
)
from ..heuristics import (
    Correlation, CoveredNegatives, CoveredPositives, FoilGain, GainHeuristic,
    Laplace, LEF, LikelihoodRatio, MinimalLength, Precision, RuleHeuristic, RuleStats, Score,
)
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..combiners import MicroVoteCombiner
from ..pruning import AnyOf, EncodingLengthRestriction, PrePruningCriterion, ThresholdPrePruning
from ..data import BooleanDataRepresentation
from ..rule import Rule


class RuleSearch(ABC):
    """Base for "find the single best rule for `target_class`" search
    strategies -- building block 1. Concrete subclasses explore
    refinements of `initial_candidates` (each an already-constructed
    `(rule, open-feature-mask)` pair, typically from a future
    `SearchSpaceInit`), scoring candidates via `heuristic` against
    `data` restricted to `example_mask` (if given, e.g. a
    grow-set mask), and return the single best rule found -- or `None`
    if a `filtering` or `stopping` criterion is configured and no rule
    the search ever reached was in its acceptable region (the criterion
    is a hard gate on what may be returned, not just a tie-breaker; the
    caller decides what "found nothing" means -- `SeCo`, for one, ends
    its covering loop).

    `filtering` and `stopping` (both optional, independent, and usable
    together, on top of the always-on `tp == 0` floor and every search's
    `optimistic_pruning`) are the two
    different ways a `PrePruningCriterion` can be consumed -- both gate
    what may be returned; only `stopping` also halts the search -- see
    its own docstring, and each search's, for exactly
    how each applies
    both.
    """

    @abstractmethod
    def search(
        self,
        data: BooleanDataRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        raise NotImplementedError


def _handle_for(data, rule: Rule, example_mask: Optional[np.ndarray]):
    """The `data` cover handle for `rule`: `initial_cover`
    (masking is the zeroth refinement), refined in through each of
    `rule`'s own conditions. In practice this is always zero iterations
    -- every `SearchSpaceInit` seeds the empty rule -- but staying
    correct for a hypothetical non-empty seed costs nothing."""
    handle = data.initial_cover(example_mask)
    for lit in rule.conditions:
        handle = data.refine_cover(handle, lit.feature)
    return handle


def _stats_from_handle(data, target_class: Any, rule: Rule, handle) -> RuleStats:
    tp, fp, fn, tn = data.cover_counts(handle, target_class)
    return RuleStats(tp=tp, fp=fp, fn=fn, tn=tn, length=rule.length())


def _optimistic_stats(stats: RuleStats) -> RuleStats:
    """The best any refinement of a `(tp, fp)` rule could possibly cover:
    every true positive kept, every false positive shed -- `(tp, 0)`.
    `fn`/`tn` shift to hold `n_pos`/`n_neg` fixed; `length` stays at the
    parent's (a real refinement is at least one condition longer, so a
    length-penalized score here comes out no lower than anything actually
    reachable -- deliberately an upper bound). Shared by every search's
    `optimistic_pruning` (`BeamSearch`, `HillClimbing`, and
    `GainAscentHillClimbing`)."""
    return RuleStats(
        tp=stats.tp, fp=0, fn=stats.fn, tn=stats.fp + stats.tn, length=stats.length
    )


class BeamSearch(RuleSearch):
    """Standard top-down beam search: starting from `initial_candidates`,
    repeatedly specialize every rule currently in the beam by one
    literal (`Rule.specialize`), keep the `beam_width` highest-scoring
    results, and stop once no further specialization is possible, no
    beam member still looks capable of beating the best rule found so
    far (`optimistic_pruning`, on by default -- see below), or
    `max_conditions` refinement steps have happened (if given). Returns
    the single best-scoring rule seen at any point, not just whatever
    ends up in the final beam -- an earlier, shorter rule can outscore
    every one of its own descendants (ties broken toward the shorter
    rule, so a later-round rediscovery of an already-optimal candidate
    via extra, non-restricting conditions never displaces the shorter
    original in the beam). With `filtering` configured, only a
    filter-passing rule is ever eligible to be that best -- if none was
    reached, the search returns `None` (see `filtering`, below).

    `optimistic_pruning` (default `True`) bounds the search the way
    Fürnkranz's SeCo survey describes: a rule covering `tp` positives
    and `fp` negatives can, at very best, refine into one covering those
    same `tp` positives and *no* negatives, so if the heuristic's score
    at that hypothetical `(tp, 0)` point doesn't exceed the best score
    found so far, no refinement of this candidate ever will and it isn't
    specialized. This is checked against each beam member just before it
    would be expanded for the next round; once it holds for *every* rule
    in the beam at once, the whole search stops.

    "Best score found so far" means the best score of a rule that has
    actually passed `filtering` (`best_rule`'s score) -- that's the best
    result the search could return, so it's the only sound bound. With
    no `filtering` configured, that's just the plain running best. While
    nothing has passed `filtering` yet there's no bound at all and
    nothing is pruned -- so a strict `filtering` criterion makes
    optimistic pruning kick in later (a lower filtered-best is a weaker
    bound), never earlier, and can never cause the search to skip a
    branch it would otherwise have kept.

    This replaces an older unconditional "stop once some beam member has
    `fp == 0`" rule -- a pure rule's own optimistic bound is just its
    own score, which the running best already reflects, so the general
    check subsumes it -- and, on searches over interval-binarized
    numeric features, cuts out the long tail of ever-longer duplicate
    rules at the same coverage that used to pad the run out after the
    real answer was already in hand. It assumes the heuristic rewards
    covering more positives and fewer negatives (`Precision`, `Laplace`,
    `WRAcc`, `Correlation`, `Accuracy`, ... all do); turn it off for one
    that doesn't (e.g. `Entropy`, symmetric in the covered-class ratio).
    The `tp == 0` floor is separate and always on -- a rule covering no
    positives is degenerate for `target_class`, not merely unpromising,
    and every one of its descendants stays that way.

    `beam_width=1` is close to `HillClimbing` over an ordinary
    `RuleHeuristic` (`HillClimbing` adds a local-maximum stop and drops
    the running-best bookkeeping a single lineage doesn't need) -- but a
    `GainHeuristic` (e.g. `FoilGain`) can't be used here at all, at any
    `beam_width`: its score is only meaningful relative to its *own*
    parent, and as soon as more than one candidate is ever in flight
    (which `beam_width=1` doesn't prevent -- multiple children from the
    same round's single parent are still compared against each other,
    which is fine, but so is comparing scores computed against
    *different* parents the moment `initial_candidates` has more than
    one entry, or a future beam_width>1 round pools children from
    multiple beam members) there's no shared scale left to rank them
    on. Rather than carve out a narrow, guarded exception for the one
    case that happens to be safe, `GainHeuristic` support lives
    entirely in `GainAscentHillClimbing` instead, whose single lineage
    makes "only one parent, ever" a standing invariant rather than
    something that needs policing here.

    `filtering` never changes the beam itself -- every round's beam is
    selected purely by `heuristic`, regardless of `filtering`, and a
    candidate in the excluded region is still specialized like any
    other. What it changes is *which* candidate is eligible to be
    returned: each round, `best_rule` advances to the highest-scoring
    candidate that also has `filtering.accept(...)` (walking down that
    round's sorted candidates past any that don't, not just checking the
    very top one). A rule in the excluded region is never returned -- if
    the search never reaches a single filter-passing rule, it returns
    `None`, not a best-effort rejected one. `filtering` also feeds
    `optimistic_pruning`'s bound (see above): since only a
    filter-passing rule can be returned, only its score is a sound thing
    to prune against, so a strict `filtering` criterion only ever delays
    the optimistic cutoff, never brings it forward.

    `stopping` gates the result exactly as `filtering` does (a rule in
    its reject region is never returned), and *additionally* halts the
    search: `stopping.evaluate(...)` is checked once per round, from the
    first refinement onward (never against the seed), against that
    round's single best-scoring candidate (the same one that would seed
    next round's beam). The moment it fires, the search stops -- if
    `stopping.accept(...)` is also true for that candidate it's returned
    directly, otherwise the search returns `best_rule` as tracked so far
    (the best rule seen that was in every criterion's acceptable region;
    `None` if there wasn't one -- e.g. FOSSIL's `Correlation < 0.3`
    firing on the first refinement, before any rule reached 0.3, returns
    `None`, not that sub-threshold rule). "Best", not "last", since
    unlike `HillClimbing`, `BeamSearch` maintains that running best
    regardless. The only thing `stopping` does that `filtering` doesn't
    is quit early -- so it can miss a deeper rule that would have
    re-entered the acceptable region a few refinements later, which
    `filtering` (running the search out) would still find.
    """

    def __init__(
        self,
        beam_width: int = 5,
        max_conditions: Optional[int] = None,
        optimistic_pruning: bool = True,
    ):
        self.beam_width = beam_width
        self.max_conditions = max_conditions
        self.optimistic_pruning = optimistic_pruning

    def search(
        self,
        data: BooleanDataRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        if not initial_candidates:
            raise ValueError("BeamSearch.search needs at least one initial candidate")
        dataspec = data.spec

        def promises_improvement(stats: RuleStats, threshold: Optional[Score]) -> bool:
            # optimistic pruning: if even the hypothetical perfect
            # refinement (tp, 0) can't out-score `threshold` -- the score
            # of the best filter-passing rule found so far, i.e. the best
            # result the search could actually return -- no actual
            # refinement of this candidate can either, so there's no
            # point specializing it. Subsumes the old unconditional
            # "stop once a beam member has fp == 0" floor (a pure rule's
            # own optimistic bound is just its own score, already
            # reflected in `threshold`). Assumes the heuristic rewards
            # covering more positives / fewer negatives -- see this
            # class's docstring.
            if not self.optimistic_pruning or threshold is None:
                return True
            return heuristic.score(_optimistic_stats(stats)) > threshold

        def is_eligible(rule: Rule, stats: RuleStats) -> bool:
            return filtering is None or filtering.accept(rule, stats, data, target_class, example_mask)

        # beam entries carry their own already-computed handle and stats:
        # the handle so the next round's specialize step can call
        # `refine_cover` on it instead of recomputing coverage for the
        # grown rule from scratch, the stats so scoring, the tp == 0
        # floor and the optimistic bound never need a second pass either
        beam: List[Tuple[Rule, FrozenSet[int], Any, RuleStats]] = []
        for rule, mask in initial_candidates:
            handle = _handle_for(data, rule, example_mask)
            beam.append((rule, mask, handle, _stats_from_handle(data, target_class, rule, handle)))
        # best_rule tracks the best filtering-eligible rule seen *strictly
        # before* the round currently being evaluated (see the stopping
        # block below for why the timing matters) -- if none is ever seen
        # it stays None and the search returns None. best_score starts at
        # None (not a -inf sentinel) rather than assuming a plain float --
        # a LEF's score is a tuple, and tuple > float raises. Length-0
        # rules (the universal seed) are skipped: the empty conjunction is
        # never itself a meaningful "found rule" -- SeCo, ReducedError-
        # Pruning etc. all treat it specially -- so a search that stops
        # before refining anywhere should return None, not the seed.
        best_rule: Optional[Rule] = None
        best_score: Optional[Score] = None
        for rule, mask, handle, stats in beam:
            if rule.length() == 0:
                continue
            s = heuristic.score(stats)
            if is_eligible(rule, stats) and (best_score is None or s > best_score):
                best_rule, best_score = rule, s

        depth = 0
        while beam and (self.max_conditions is None or depth < self.max_conditions):
            # only specialize beam members that could still lead
            # somewhere: not degenerate (tp > 0 -- every descendant of a
            # tp == 0 rule also has tp == 0), and, under optimistic
            # pruning, still holding an optimistic (tp, 0) bound that
            # beats `best_score` -- the score of the best rule that has
            # actually passed `filtering` so far (== the plain running
            # best when there's no filtering; None, so no pruning, while
            # nothing has passed it yet). That's the best result the
            # search could return, so it's the only sound bound. When
            # *nothing* in the beam qualifies, the whole search is done
            # -- this is what stops a beam from cycling through
            # ever-longer, same-coverage duplicate rules long after the
            # real answer was found (common with interval-binarized
            # numeric features), the job the old "some beam member has
            # fp == 0" break used to do.
            refinable = [
                (rule, mask, handle, stats)
                for rule, mask, handle, stats in beam
                if stats.tp > 0 and promises_improvement(stats, best_score)
            ]
            if not refinable:
                break
            # each child's handle is derived from its *specific* parent's
            # handle by refining in the one literal that was added --
            # `rule.specialize` only ever adds exactly one, appended to
            # `conditions` (constraint closure only shrinks the returned
            # mask, never touches the child's own conditions), and
            # `Rule.__init__` preserves input order (it only ever drops
            # exact duplicates, which this can't be -- `mask` never
            # offers a feature already fixed), so `conditions[-1]` is
            # always that one new literal -- O(1), no per-child set-diff.
            proposals: List[Tuple[Rule, FrozenSet[int], Any]] = []
            for rule, mask, handle, stats in refinable:
                for child_rule, child_mask in rule.specialize(dataspec, mask):
                    added = child_rule.conditions[-1].feature
                    proposals.append((child_rule, child_mask, data.refine_cover(handle, added)))
            if not proposals:
                break

            # different beam parents can specialize to the same rule (e.g.
            # "a" refining by b and "b" refining by a both reach {a, b}) --
            # Rule's own order-independent __eq__/__hash__ (Rule.pos/neg;
            # target is fixed across every candidate here, so this reduces
            # to plain literal-set equality) already treats those as
            # identical, and their masks agree too (propagate's closure
            # depends on the *set* of fixed features, not the order they
            # were fixed in) -- so collapsing duplicates into one
            # representative here loses nothing, and is the only thing
            # standing between this and an exponential blow-up: each
            # surviving duplicate would otherwise re-explore the same
            # subtree of further refinements independently, every round.
            # (Their handles are equally interchangeable -- both refine to
            # the identical coverage -- so keeping whichever arrived first
            # is exactly as sound as the mask-only dedup this replaces.)
            deduped: Dict[Rule, Tuple[FrozenSet[int], Any]] = {}
            for child_rule, child_mask, child_handle in proposals:
                deduped.setdefault(child_rule, (child_mask, child_handle))

            scored: List[Tuple[Score, Rule, FrozenSet[int], Any, RuleStats]] = []
            for rule, (mask, handle) in deduped.items():
                stats = _stats_from_handle(data, target_class, rule, handle)
                scored.append((heuristic.score(stats), rule, mask, handle, stats))
            # ties broken toward the shorter (more general) rule -- otherwise
            # an arbitrary, sort-order-dependent longer duplicate could win a
            # beam slot over an equally-good shorter one
            scored.sort(key=lambda t: (t[0], -t[1].length()), reverse=True)
            beam = [(rule, mask, handle, stats) for _, rule, mask, handle, stats in scored[: self.beam_width]]

            top_rule, top_stats = scored[0][1], scored[0][4]

            # stopping halts the search: checked against this round's top
            # candidate only (the same rule that would seed next round).
            # Checked *before* best_rule is advanced from this round, so
            # `best_rule` still reflects only rounds strictly before the
            # one that triggered -- when stopping fires because the
            # frontier crossed the threshold, the search hands back the
            # best rule from *before* things went bad, or None if there
            # wasn't one yet (e.g. FOSSIL's `Correlation < 0.3` firing on
            # the first refinement returns None, not that sub-threshold
            # rule).
            if stopping is not None and stopping.evaluate(
                top_rule, top_stats, data, target_class, example_mask
            ):
                if stopping.accept(top_rule, top_stats, data, target_class, example_mask):
                    return top_rule
                return best_rule  # None if nothing acceptable came before this round

            # advance best_rule to this round's highest-scoring eligible
            # candidate (walking past any that fail `filtering`), not
            # necessarily the top one
            for s, rule, _, _, stats in scored:
                if is_eligible(rule, stats):
                    if best_score is None or s > best_score:
                        best_rule, best_score = rule, s
                    break

            depth += 1

        return best_rule  # None if nothing eligible was ever found


class HillClimbing(RuleSearch):
    """Single-lineage greedy search: each round, specialize the current
    rule, score every child with a plain `RuleHeuristic`, move to
    whichever scores highest, stop at a **local maximum of the
    heuristic** -- the moment no child's score exceeds the current
    rule's own. Same idea as `BeamSearch(beam_width=1)`, but with that
    local-maximum stop it halts as soon as the greedy ascent peaks
    rather than running the lineage out; on a landscape that rises
    monotonically to a single peak (the usual case for `Precision`/
    `Laplace`/`Correlation` on a conjunctive concept) the two return the
    same rule, `HillClimbing` just reaching it with less work. Kept as
    its own class because a single lineage needs no beam dedup and no
    separately-tracked running best ("the last rule reached" is also the
    highest-scoring, since the walk only ever moves to a strictly better
    one). The same-scale parent/child comparison the local-maximum stop
    makes is valid precisely because one lineage means both are always
    scored the same way. A `GainHeuristic` needs `GainAscentHillClimbing`
    instead (this class rejects one).

    `optimistic_pruning` (default `True`) is exactly `BeamSearch`'s bound
    of the same name, restated for a single lineage: the best any
    refinement of a `(tp, fp)` rule could cover is `(tp, 0)`, so if the
    heuristic's score at that projection can't beat the best rule the
    search could still return -- the most recent filter-eligible rule, or
    (with no `filtering`) just the current one -- no real child can
    either, so the walk stops without enumerating this round's children.
    It subsumes the old `fp == 0` floor (a pure rule's `(tp, 0)`
    projection is itself), and, for a plain heuristic that rewards more
    `tp` / fewer `fp`, it never changes the result -- only skips work the
    local-maximum stop would have reached the same conclusion from. Turn
    it off for a heuristic that isn't monotone that way (`Entropy`,
    symmetric in the covered-class ratio). `tp == 0` stays a separate,
    always-on floor.

    `filtering` never affects which move gets taken -- the walk always
    follows the highest-scoring child, exactly as without it. It only
    changes what's *eligible to be returned*: this class tracks the most
    recently visited rule with `filtering.accept(...)` (a non-empty seed
    counts; the length-0 universal seed never does), and falls back to
    it whenever the walk's natural endpoint isn't itself eligible. If
    the walk never visits a single filter-passing rule, the search
    returns `None` -- a configured filter is a hard gate, not a
    best-effort preference.

    `stopping` is checked from the first refinement onward (never
    against the seed) -- each round, once the walk has moved to a new
    rule, `stopping.evaluate(...)` is checked on it *before* that rule
    is recorded as the fallback. The moment it fires, `accept(...)` on
    that same rule decides what's returned: itself, if accepted,
    otherwise the most recent rule from *strictly before* this round
    that passed both `filtering` and `stopping`'s accept (`None` if
    there wasn't one -- so `stopping` never returns a rule from its own
    reject region).

    `stop_at_local_optimum` (default `True`) is the local-maximum stop
    described above. Set it `False` and the walk instead keeps adding the
    best-scoring condition -- even one that lowers the score -- toward
    consistency (`fp == 0`). It then stops early only where growth is
    genuinely blocked or bounded: no `tp`-preserving condition left,
    `max_conditions`, or a `stopping` criterion firing (e.g.
    `ThresholdPrePruning(CoveredPositives(), 2, "<")` -- RIPPER's
    `minNo`, keeping every rule on at least two positives). This is
    RIPPER's / IREP's grow phase, and it is **only useful paired with
    post-pruning** (a `SingleRulePostProcessing` like
    `ReducedErrorPruning`): on its own it just grows maximally specific,
    overfit rules. `optimistic_pruning` has no effect once this is
    `False` -- it exists to skip work the local-maximum stop would reach
    anyway, and there is no such stop then. Leave it `True` outside the
    grow-then-prune recipe.
    """

    def __init__(
        self,
        max_conditions: Optional[int] = None,
        optimistic_pruning: bool = True,
        stop_at_local_optimum: bool = True,
    ):
        self.max_conditions = max_conditions
        self.optimistic_pruning = optimistic_pruning
        self.stop_at_local_optimum = stop_at_local_optimum

    # -- hooks the gain-ascent subclass overrides; everything else in
    #    `search` is shared single-lineage machinery. The base versions
    #    below are the plain-heuristic behaviour. --------------------------

    def _reject_heuristic(self, heuristic: RuleHeuristic) -> None:
        if isinstance(heuristic, GainHeuristic):
            raise ValueError(
                f"{type(self).__name__} needs a plain RuleHeuristic, got "
                f"{type(heuristic).__name__} -- use GainAscentHillClimbing for a GainHeuristic"
            )

    def _child_score(
        self, heuristic: RuleHeuristic, child_stats: RuleStats, parent_stats: RuleStats
    ) -> Score:
        return heuristic.score(child_stats)

    def _improvement_threshold(self, heuristic: RuleHeuristic, parent_stats: RuleStats) -> Score:
        # the best child must strictly beat this to be worth moving to:
        # the current rule's own score (a local maximum of the heuristic).
        return heuristic.score(parent_stats)

    def _optimistic_stop(
        self, heuristic: RuleHeuristic, stats: RuleStats, best_stats: Optional[RuleStats],
    ) -> bool:
        # BeamSearch's optimality bound: if the heuristic's score at the
        # (tp, 0) projection can't beat the best returnable rule so far
        # (`best_stats` -- None before any exists), no child down this
        # lineage will either. Subsumes the old fp == 0 floor.
        if best_stats is None:
            return False
        return heuristic.score(_optimistic_stats(stats)) <= heuristic.score(best_stats)

    def search(
        self,
        data: BooleanDataRepresentation,
        target_class: Any,
        heuristic: RuleHeuristic,
        initial_candidates: Sequence[Tuple[Rule, FrozenSet[int]]],
        example_mask: Optional[np.ndarray] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ) -> Optional[Rule]:
        if not initial_candidates:
            raise ValueError(f"{type(self).__name__}.search needs at least one initial candidate")
        if len(initial_candidates) > 1:
            raise ValueError(
                f"{type(self).__name__}.search needs exactly one initial candidate -- "
                "it only ever tracks a single lineage"
            )
        self._reject_heuristic(heuristic)
        dataspec = data.spec

        def is_eligible(rule: Rule, stats: RuleStats) -> bool:
            # returnable only if in the acceptable region of every
            # configured criterion -- so `stopping` (like `filtering`)
            # never lets the search hand back a rule from its own reject
            # region; it just *also* halts the walk (see below).
            for crit in (filtering, stopping):
                if crit is not None and not crit.accept(
                    rule, stats, data, target_class, example_mask
                ):
                    return False
            return True

        rule, mask = initial_candidates[0]
        handle = _handle_for(data, rule, example_mask)
        stats = _stats_from_handle(data, target_class, rule, handle)
        # what `stopping` falls back to: the most recent eligible rule
        # from *before* the step that triggered it. The length-0 seed
        # never counts -- a walk that's stopped before refining anywhere
        # returns None, not the empty rule.
        last_eligible: Optional[Tuple[Rule, RuleStats]] = (
            (rule, stats) if rule.length() > 0 and is_eligible(rule, stats) else None
        )

        depth = 0
        while mask and (self.max_conditions is None or depth < self.max_conditions):
            if stats.tp == 0:
                break  # degenerate for target_class -- every descendant stays tp == 0

            if not self.stop_at_local_optimum and stats.fp == 0:
                break  # the rule is consistent -- nothing left to add

            if self.stop_at_local_optimum and self.optimistic_pruning and self._optimistic_stop(
                heuristic, stats, last_eligible[1] if last_eligible is not None else None
            ):
                break

            threshold = self._improvement_threshold(heuristic, stats)

            best_child: Optional[Tuple[Rule, FrozenSet[int], Any, RuleStats, Score]] = None
            for child_rule, child_mask in rule.specialize(dataspec, mask):
                added = child_rule.conditions[-1].feature  # see BeamSearch.search's comment
                child_handle = data.refine_cover(handle, added)
                child_stats = _stats_from_handle(data, target_class, child_rule, child_handle)
                if not self.stop_at_local_optimum and child_stats.tp == 0:
                    continue  # never grow toward a rule covering no positives
                value = self._child_score(heuristic, child_stats, stats)
                if best_child is None or value > best_child[4]:
                    best_child = (child_rule, child_mask, child_handle, child_stats, value)

            if best_child is None:
                break  # nothing left to move to
            if self.stop_at_local_optimum and best_child[4] <= threshold:
                break  # local optimum -- no child beats the current rule

            rule, mask, handle, stats, _ = best_child
            depth += 1

            if stopping is not None and stopping.evaluate(rule, stats, data, target_class, example_mask):
                if stopping.accept(rule, stats, data, target_class, example_mask):
                    return rule
                return last_eligible[0] if last_eligible is not None else None

            if is_eligible(rule, stats):
                last_eligible = (rule, stats)

        if is_eligible(rule, stats):
            return rule
        return last_eligible[0] if last_eligible is not None else None


class GainAscentHillClimbing(HillClimbing):
    """`HillClimbing` for a `GainHeuristic` (e.g. `FoilGain`). Where the
    base class ascends a *global* per-rule objective and stops at its
    local maximum, this follows the *local gradient*: each child is
    scored by `heuristic.score(child_stats, current_stats)` -- a gain
    relative to the current rule, not an absolute score -- and the walk
    stops once the best available gain is non-positive. 0 is the only
    shared reference a gain has (two gains against different parents
    aren't the same question), but it's always a meaningful one: a
    refinement that doesn't beat its own parent is never worth taking.

    This is the only search that can run a `GainHeuristic` at all --
    `BeamSearch` can hold several candidates in flight scored against
    different parents, with no shared scale to rank them (see its
    docstring); the base `HillClimbing` rejects a `GainHeuristic`
    outright. Because a gain has no cross-round scale, "the last rule the
    walk reached" is the only well-defined answer -- there's no
    "best ever seen" to track (see `GainHeuristic`'s own docstring).

    `optimistic_pruning` (default `True`) is the base-class bound
    rephrased for a gain: the best any refinement of a `(tp, fp)` rule
    could cover is `(tp, 0)`, so if the gain *there* (against the current
    rule) is already non-positive, no real child improves on it -- stop
    without enumerating children. Subsumes the old `fp == 0` floor (a
    pure rule's `(tp, 0)` projection is itself, gain 0). A gain's
    reference is always the immediate parent, so unlike the base class
    this needs no "best returnable so far" to bound against.

    `stop_at_local_optimum=False` (inherited) turns this into RIPPER's
    grow phase: keep adding the max-`FoilGain` condition -- even past the
    point gain goes non-positive -- toward consistency (`max_conditions`
    and a `stopping` criterion still bound it). Only useful feeding a
    `ReducedErrorPruning` post-processor; see `HillClimbing`'s docstring.
    """

    def _reject_heuristic(self, heuristic: RuleHeuristic) -> None:
        if not isinstance(heuristic, GainHeuristic):
            raise ValueError(
                f"GainAscentHillClimbing needs a GainHeuristic (e.g. FoilGain), got "
                f"{type(heuristic).__name__} -- use HillClimbing for a plain RuleHeuristic"
            )

    def _child_score(
        self, heuristic: RuleHeuristic, child_stats: RuleStats, parent_stats: RuleStats
    ) -> Score:
        return heuristic.score(child_stats, parent_stats)

    def _improvement_threshold(self, heuristic: RuleHeuristic, parent_stats: RuleStats) -> Score:
        return 0.0  # a gain's own shared reference point

    def _optimistic_stop(
        self, heuristic: RuleHeuristic, stats: RuleStats, best_stats: Optional[RuleStats],
    ) -> bool:
        return heuristic.score(_optimistic_stats(stats), stats) <= 0


class SearchSpaceInit(ABC):
    """Base for building block 5: where a single-rule search starts.
    Returns `(rule, open-feature-mask)` pairs, the same shape `RuleSearch.
    search`'s `initial_candidates` expects (and `Rule.specialize` itself
    returns) -- usually just one pair, but nothing here requires that.

    `example_mask` is the current covering scope (`SeCo`'s `remaining`,
    passed straight through by `SingleRuleLearner.learn_one_rule`, or
    `None` when there's no covering loop) -- most implementations ignore
    it; `SeedExample` needs it, to seed on an example that is still
    *uncovered*.
    """

    @abstractmethod
    def initial_candidates(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        raise NotImplementedError


class EmptyRuleAllFeatures(SearchSpaceInit):
    """Default: start from the empty (universal) rule, every feature open."""

    def initial_candidates(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        empty = Rule([], target=target_class, dataspec=ds)
        return [(empty, frozenset(range(ds.n_features)))]


class FeatureSubset(SearchSpaceInit):
    """Start from the empty rule, but restrict the search to a named
    subset of features (by name or index) -- e.g. to bound search cost,
    or to exclude features known to be unusable for this `target_class`.
    """

    def __init__(self, allowed_features: Sequence[Union[str, int]]):
        self.allowed_features = tuple(allowed_features)

    def initial_candidates(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        empty = Rule([], target=target_class, dataspec=ds)
        mask = frozenset(ds.feature_index(f) for f in self.allowed_features)
        return [(empty, mask)]


class SeedExample(SearchSpaceInit):
    """AQ's search-space start: pick one *positive* example still
    uncovered by the rules learned so far (the "seed"), and search the
    space of its generalizations -- the empty rule, open only to the
    features the seed itself satisfies. Every rule the search can reach
    from there is therefore guaranteed to cover the seed (a proper
    subset of a True-valued feature set stays True on that example), so
    AQ never has to check "does this still cover the seed?" explicitly --
    the open mask enforces it.

    `strategy` picks *which* uncovered positive:
    - ``"first"`` (default) -- the lowest-indexed one; fully
      deterministic, and the order examples happen to sit in the data is
      as good as any other arbitrary choice for a noise-free covering.
    - ``"random"`` -- a uniformly random uncovered positive (`random_state`
      seeds it); over several runs this explores different rules first,
      which matters when `maxstar` trimming makes the search order-
      sensitive.
    - ``"index"`` -- seed on the exact row `index`, whatever its class
      (`target_class` is ignored -- callers pass the row's own label).
      `example_mask` is ignored too. This is how `pyrulearn.learners.pylord.PyLORD`
      seeds a search on *every* example in turn, rather than one per
      covering iteration.

    Explicit-negation features make "the features the seed satisfies"
    already include every "attribute X is absent" the seed warrants
    (its negation feature is the True one), so a seed rule can still
    grow conditions like ``not smoker`` -- nothing special is needed for
    negation here.

    Raises `ValueError` if there is no uncovered positive to seed on --
    `SeCo`'s own loop never calls this in that state (it checks first),
    but a direct caller would want to know rather than get a silent
    empty result.
    """

    def __init__(
        self, strategy: str = "first", random_state: Optional[int] = None, index: Optional[int] = None,
    ):
        if strategy not in ("first", "random", "index"):
            raise ValueError(f"strategy must be 'first', 'random' or 'index', got {strategy!r}")
        if strategy == "index" and index is None:
            raise ValueError("SeedExample(strategy='index') needs index=")
        self.strategy = strategy
        self.random_state = random_state
        self.index = index

    def initial_candidates(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Sequence[Tuple[Rule, FrozenSet[int]]]:
        ds = data.spec
        if self.strategy == "index":
            seed = int(self.index)
        else:
            positive = data.y == target_class
            in_scope = positive if example_mask is None else (positive & example_mask)
            candidates = np.flatnonzero(in_scope)
            if candidates.size == 0:
                raise ValueError("SeedExample: no uncovered positive example to seed on")
            if self.strategy == "first":
                seed = int(candidates[0])
            else:
                rng = np.random.default_rng(self.random_state)
                seed = int(rng.choice(candidates))
        open_mask = frozenset(int(i) for i in data.features_of(seed))
        empty = Rule([], target=target_class, dataspec=ds)
        return [(empty, open_mask)]


class SingleRulePreparation(ABC):
    """Base for building block 3: what to do with the in-scope examples
    before searching for a single rule. `example_mask` is the incoming
    scope (e.g. a SeCo loop's "remaining" examples, or None for every
    row). Returns `(search_mask, context)`: `search_mask` is what the
    search itself is restricted to (a sub-mask of `example_mask`, or
    `example_mask` unchanged), `context` carries whatever the matching
    `SingleRulePostProcessing` needs (e.g. a held-out pruning mask).
    """

    @abstractmethod
    def prepare(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        raise NotImplementedError


class NoSplit(SingleRulePreparation):
    """Default: search on every in-scope example, no held-out set."""

    def prepare(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        return example_mask, None


class GrowPruneSplit(SingleRulePreparation):
    """RIPPER/IREP-style: split the in-scope examples into a growing set
    (what the search itself runs on) and a held-out pruning set (returned
    as `context`, meant for a matching `ReducedErrorPruning`), stratified
    by class via `sklearn.model_selection.train_test_split`.

    Confirmed as a real, not just hypothetical, failure mode: a `SeCo`
    covering loop's "remaining" scope shrinks every iteration, and can
    easily end up with too few examples of some class to stratify by the
    time a later rule is learned. Rather than let that raise and take
    down the whole covering loop, `prepare` falls back in two steps: a
    non-stratified split if stratification specifically isn't possible,
    then no split at all (search on every in-scope example, no pruning
    set -- same as `NoSplit`) if even that isn't possible (too few
    in-scope examples to split at all).
    """

    def __init__(self, prune_fraction: float = 0.33, random_state: Optional[int] = None):
        self.prune_fraction = prune_fraction
        self.random_state = random_state

    def prepare(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Any]:
        from sklearn.model_selection import train_test_split

        n = data.n_samples
        in_scope = np.arange(n) if example_mask is None else np.flatnonzero(example_mask)
        if len(in_scope) < 2:
            return example_mask, None  # too few examples to split at all

        labels = data.y[in_scope]
        try:
            grow_idx, prune_idx = train_test_split(
                in_scope, test_size=self.prune_fraction, random_state=self.random_state, stratify=labels,
            )
        except ValueError:
            try:
                grow_idx, prune_idx = train_test_split(
                    in_scope, test_size=self.prune_fraction, random_state=self.random_state,
                )
            except ValueError:
                return example_mask, None

        grow_mask = np.zeros(n, dtype=bool)
        grow_mask[grow_idx] = True
        prune_mask = np.zeros(n, dtype=bool)
        prune_mask[prune_idx] = True
        return grow_mask, prune_mask


class SingleRulePostProcessing(ABC):
    """Base for building block 4: what to do with a single freshly-found
    rule before it's accepted. `context` is whatever the matching
    `SingleRulePreparation` produced (e.g. a pruning mask)."""

    @abstractmethod
    def postprocess(
        self, rule: Rule, data: BooleanDataRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        raise NotImplementedError


class NoPostProcessing(SingleRulePostProcessing):
    """Default: accept the rule as found, unchanged."""

    def postprocess(
        self, rule: Rule, data: BooleanDataRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        return rule


class ReducedErrorPruning(SingleRulePostProcessing):
    """RIPPER/IREP-style: `context` (from `GrowPruneSplit`) is the
    held-out pruning mask. Scores every *prefix* of `rule` down to length
    1 (the full rule, one shorter, ..., a single condition) against it
    and returns whichever prefix scores best -- RIPPER's own pruning
    metric picks the single best-scoring truncation directly, not a
    greedy walk that stops at the first non-improving removal, so that's
    what this does too. Ties favor the shorter (more general) prefix.
    `context=None` (no pruning set) is a no-op, returning `rule`
    unchanged.

    Deliberately never truncates all the way to length 0 (the empty,
    unconditional rule) even if it would score best on a small/skewed
    pruning sample -- confirmed directly to be a real failure mode, not
    a hypothetical one: an "always true" rule inserted into a `SeCo`
    rule set covers every remaining example at once (both classes),
    which both terminates the covering loop early and, at prediction
    time, fires on every example alongside whatever real rules exist,
    skewing predictions. `SeCo.fit` also rejects a length-0 rule coming
    from the search itself as a second layer of defense (see its own
    docstring) -- an unconditional rule is exactly what the *default*
    rule mechanism is already for, so there's no real capability lost
    by excluding it here specifically. If `rule` itself is already
    length 0 (the search's own conclusion, not this class truncating
    down to it), it's returned unchanged -- only *truncation* to empty
    is refused, not a rule that started that way.
    """

    def __init__(self, heuristic: RuleHeuristic):
        self.heuristic = heuristic

    def postprocess(
        self, rule: Rule, data: BooleanDataRepresentation, target_class: Any, context: Any,
    ) -> Rule:
        if context is None:
            return rule

        def score(r: Rule) -> Score:
            stats = RuleStats.from_rule(r, data, target_class, example_mask=context)
            return self.heuristic.score(stats)

        best, best_score = rule, score(rule)
        for k in range(rule.length() - 1, 0, -1):  # stop at 1 -- never truncate to the empty rule
            truncated = Rule(
                rule.conditions[:k], target=rule.target, dataspec=rule.dataspec,
                n_features=rule.n_features, default_fmt=rule.default_fmt,
            )
            s = score(truncated)
            if s >= best_score:
                best, best_score = truncated, s
        return best


class SingleRuleLearner:
    """Ties the five building blocks together: `learn_one_rule` prepares
    the in-scope examples (block 3), gets a starting search space
    (block 5), runs the search (block 1, scored via block 2's
    `heuristic`), and post-processes the result (block 4). Defaults to
    the simplest configuration -- `BeamSearch`, no split, no
    post-processing, start from the empty rule with every feature open --
    RIPPER/IREP-style behavior needs `GrowPruneSplit`/`ReducedErrorPruning`
    passed explicitly.

    `heuristic` has no default -- picking one is a real modeling choice,
    not something to guess at silently. `filtering`/`stopping`
    (optional, independent, usable together, on top of `RuleSearch`'s
    own always-on `tp == 0` floor and optimistic-value pruning -- see
    `PrePruningCriterion`)
    are passed straight through to the search at call time, not baked
    into `search` itself -- keeps `RuleSearch` implementations reusable
    regardless of which policy is in effect, the same way `heuristic`
    already works.
    """

    def __init__(
        self,
        heuristic: RuleHeuristic,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
    ):
        self.heuristic = heuristic
        self.search = search if search is not None else BeamSearch()
        self.preparation = preparation if preparation is not None else NoSplit()
        self.postprocessing = postprocessing if postprocessing is not None else NoPostProcessing()
        self.space_init = space_init if space_init is not None else EmptyRuleAllFeatures()
        self.filtering = filtering
        self.stopping = stopping

    def learn_one_rule(
        self, data: BooleanDataRepresentation, target_class: Any,
        example_mask: Optional[np.ndarray] = None,
        space_init: Optional[SearchSpaceInit] = None,
    ) -> Optional[Rule]:
        """Returns the post-processed single rule, or `None` if the search
        found nothing that passed its `filtering` criterion (propagated
        straight from `RuleSearch.search` -- post-processing is skipped).

        `space_init` overrides `self.space_init` for this call only --
        used by `SeCo`'s `model=FlatRuleSet` seed-covering loop
        (`_seed_covering_fit`) to seed each rule on a specific chosen
        example."""
        si = space_init if space_init is not None else self.space_init
        search_mask, context = self.preparation.prepare(data, target_class, example_mask)
        initial = si.initial_candidates(data, target_class, search_mask)
        rule = self.search.search(
            data, target_class, self.heuristic, initial,
            example_mask=search_mask, filtering=self.filtering, stopping=self.stopping,
        )
        if rule is None:
            return None
        return self.postprocessing.postprocess(rule, data, target_class, context)


class RuleSetOptimizer(ABC):
    """The rule-set optimizer: rework a whole class's rules *after* the
    covering loop has produced them, before they become a `RuleSet`.
    Where `SingleRulePostProcessing` prunes one rule in isolation, this
    reconsiders each rule against the rest of the set -- RIPPER's
    optimization phase (`ReplaceReviseOptimization`).

    Still binary-shaped, like every other `SeCo` building block: one
    `target_class`, negatives = everything else within `example_mask`.
    RIPPER runs it per class, inside a class-ordered decomposition
    (`pyrulearn.learners.multiclass.OrderedOneVsRest`), which is exactly how
    `Pypper` composes it.
    """

    @abstractmethod
    def optimize(
        self,
        rules: List[Rule],
        data: BooleanDataRepresentation,
        target_class: Any,
        single_rule_learner: "SingleRuleLearner",
        example_mask: Optional[np.ndarray] = None,
    ) -> List[Rule]:
        """Return a (hopefully better) list of rules for `target_class`.
        `single_rule_learner` is `SeCo`'s own, handed in so alternatives
        are grown with the identical search/heuristic configuration."""
        raise NotImplementedError


# -- minimum description length (Cohen 1995, after Quinlan) ------------------

def _log2_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def _rule_theory_bits(n_conditions: int, n_possible: int) -> float:
    """Bits to transmit a rule with `n_conditions` conditions drawn from
    `n_possible` possible ones -- ``||k|| + k*log2(n/k) + (n-k)*log2(n/(n-k))``,
    halved to allow for redundancy among the features (Cohen 1995 sec. 2.2)."""
    k, n = min(n_conditions, n_possible), n_possible
    if n <= 0 or k <= 0:
        return 0.0
    p = k / n
    bits = math.log2(k) + k * math.log2(1.0 / p)
    if k < n:
        bits += (n - k) * math.log2(1.0 / (1.0 - p))
    return 0.5 * bits


def _exception_bits(n_covered: int, n_fp: int, n_uncovered: int, n_fn: int) -> float:
    """Bits to transmit which covered examples are false positives and
    which uncovered ones are false negatives -- ``log2(t+1) + log2(C(t, e))``
    per subset (the count, then which)."""
    def subset(t: int, e: int) -> float:
        return (math.log2(t + 1) + _log2_choose(t, e)) if t > 0 else 0.0
    return subset(n_covered, n_fp) + subset(n_uncovered, n_fn)


def _combined_cover(rules: Sequence[Rule], data: BooleanDataRepresentation) -> np.ndarray:
    covered = np.zeros(data.n_samples, dtype=bool)
    for r in rules:
        covered |= r.covers_data(data)
    return covered


def rule_set_description_length(
    rules: Sequence[Rule],
    data: BooleanDataRepresentation,
    target_class: Any,
    example_mask: Optional[np.ndarray] = None,
) -> float:
    """Total MDL of `rules` as a theory for the binary problem
    ``target_class`` vs. the rest (within `example_mask` if given): the
    sum of each rule's `_rule_theory_bits` plus the `_exception_bits` for
    the union of their coverage."""
    n_possible = data.spec.n_features
    theory = sum(_rule_theory_bits(r.length(), n_possible) for r in rules)
    y = np.asarray(data.y)
    scope = np.ones(data.n_samples, dtype=bool) if example_mask is None else np.asarray(example_mask, dtype=bool)
    covered = _combined_cover(rules, data) & scope
    pos = (y == target_class) & scope
    n_covered = int(covered.sum())
    n_fp = int((covered & ~pos).sum())
    n_uncovered = int((scope & ~covered).sum())
    n_fn = int((scope & ~covered & pos).sum())
    return theory + _exception_bits(n_covered, n_fp, n_uncovered, n_fn)


class ReplaceReviseOptimization(RuleSetOptimizer):
    """RIPPER's optimization phase (Cohen 1995 sec. 4). For `passes`
    rounds (RIPPER's default `k=2`), draw a fresh grow/prune split and
    walk the rules in order; for each rule `ri` build

    - a **replacement** -- a rule grown from scratch, and
    - a **revision** -- a rule grown by adding conditions to `ri`,

    each pruned (trailing conditions dropped) to minimize the *whole
    rule set's* error on the pruning split with `ri` swapped for it. Keep
    whichever of `{ri, replacement, revision}` gives the rule set the
    lowest `rule_set_description_length`. After each round, drop any rule
    whose removal lowers the total description length.

    Deliberately simplified: no separate "residual IREP*" step (Cohen
    re-covers positives left uncovered after DL-pruning) -- the input
    already comes from a full covering loop, so uncovered positives fall
    to the default like anywhere else.
    """

    def __init__(
        self,
        passes: int = 2,
        prune_fraction: float = 1.0 / 3.0,
        random_state: Optional[int] = None,
    ):
        self.passes = passes
        self.prune_fraction = prune_fraction
        self.random_state = random_state

    def optimize(
        self,
        rules: List[Rule],
        data: BooleanDataRepresentation,
        target_class: Any,
        single_rule_learner: "SingleRuleLearner",
        example_mask: Optional[np.ndarray] = None,
    ) -> List[Rule]:
        if not rules:
            return rules
        y = np.asarray(data.y)
        scope = np.ones(data.n_samples, dtype=bool) if example_mask is None else np.asarray(example_mask, dtype=bool)
        n_feat = data.spec.n_features

        for p in range(self.passes):
            rs = None if self.random_state is None else self.random_state + p
            grow_mask, prune_mask = GrowPruneSplit(self.prune_fraction, rs).prepare(data, target_class, scope)
            revised: List[Rule] = []
            for i, rule in enumerate(rules):
                context = revised + rules[i + 1:]  # the set with `rule` removed
                candidates = [rule]
                for start in (None, rule):
                    alt = self._grow_and_prune(
                        start, context, data, target_class, single_rule_learner,
                        grow_mask, prune_mask,
                    )
                    if alt is not None and alt.length() > 0:
                        candidates.append(alt)
                best = min(
                    candidates,
                    key=lambda c: rule_set_description_length(context + [c], data, target_class, scope),
                )
                revised.append(best)
            rules = self._dl_prune(revised, data, target_class, scope)
        return rules

    def _grow_and_prune(
        self, start: Optional[Rule], context: List[Rule], data: BooleanDataRepresentation,
        target_class: Any, srl: "SingleRuleLearner",
        grow_mask: Optional[np.ndarray], prune_mask: Optional[np.ndarray],
    ) -> Optional[Rule]:
        ds = data.spec
        if start is None:
            initial = EmptyRuleAllFeatures().initial_candidates(data, target_class, grow_mask)
        else:
            open_feats = frozenset(range(ds.n_features)) - {l.feature for l in start.conditions}
            initial = [(Rule(list(start.conditions), target=target_class, dataspec=ds,
                             n_features=start.n_features), open_feats)]
        grown = srl.search.search(
            data, target_class, srl.heuristic, initial,
            example_mask=grow_mask, filtering=srl.filtering, stopping=srl.stopping,
        )
        if grown is None or grown.length() == 0:
            return None
        if prune_mask is None:
            return grown

        # prune trailing conditions to minimize the whole set's error on the pruning split
        def err(rule: Rule) -> int:
            covered = _combined_cover(context + [rule], data) & prune_mask
            pos = (np.asarray(data.y) == target_class) & prune_mask
            return int((covered & ~pos).sum()) + int((~covered & pos).sum())

        best, best_err = grown, err(grown)
        for k in range(grown.length() - 1, 0, -1):
            pref = Rule(grown.conditions[:k], target=target_class, dataspec=ds, n_features=grown.n_features)
            e = err(pref)
            if e <= best_err:
                best, best_err = pref, e
        return best

    def _dl_prune(
        self, rules: List[Rule], data: BooleanDataRepresentation, target_class: Any,
        scope: np.ndarray,
    ) -> List[Rule]:
        kept = list(rules)
        for r in reversed(list(kept)):  # RIPPER: consider the last-added rules first
            without = [x for x in kept if x is not r]
            if rule_set_description_length(without, data, target_class, scope) < \
               rule_set_description_length(kept, data, target_class, scope):
                kept = without
        return kept


class SeCo(DecomposingLearner, NativeRuleLearner):
    """Sequential covering: repeatedly call `single_rule_learner` for
    `target_class`, accept what it finds, and remove every example the
    accepted rule covers -- both classes, not just positives -- from
    further consideration (the standard separate-and-conquer covering
    policy; see Fürnkranz's survey), until `stop_covering` says to stop
    or no positive examples remain uncovered. Removing negatives too
    (not just the positives just explained) matches the textbook
    definition of "separate": each subsequent rule only ever needs to
    prove itself against what's left, not the whole original dataset.

    `stop_covering` is a `pyrulearn.pruning.PrePruningCriterion` -- the
    same type a search's `filtering=`/`stopping=`
    take -- but consulted only through its bare `evaluate()`, once per
    learned rule, between `SingleRuleLearner.learn_one_rule` calls: the
    moment it fires, the just-learned rule is discarded (never appended
    to `rules`) and the whole covering loop halts. `polarity`/`reject`/
    `accept` play no role here and can be left at their defaults --
    they exist to resolve *which* rule to hand back when a search stops
    partway through refining (`BeamSearch`'s best-so-far vs. the
    hill-climbers' last-visited), a question that doesn't arise at
    this level: there's exactly one covering loop, never a population
    of candidates to filter among, so a fired criterion always simply
    means "stop, and throw away what was just found" -- see
    `PrePruningCriterion`'s own docstring for the filtering/stopping
    distinction this collapses away here. `stats` (passed to `evaluate`
    already computed, same reasoning as everywhere else it's threaded
    through rather than recomputed) is the just-learned rule's stats
    *within the current covering scope* (`remaining` -- not the full
    dataset) -- e.g. `ThresholdPrePruning(UncoveredPositives(),
    -p, ">=")` reads `stats.fn` (how many positives would still need
    explaining once this rule is added) to stop once at most `p` remain.

    Unconditional floor, independent of any criterion configured here:
    `SeCo` stops once every positive example is covered (nothing left
    to explain), `single_rule_learner` returns `None` (its search found
    nothing passing its own `filtering` -- no acceptable rule left to
    add), the just-learned rule covers no new positives at all
    (`stats.tp == 0` within the remaining scope -- adding it would be
    pure waste, and letting the loop continue could add rule after rule
    that changes nothing), or the rule is length 0 (unconditional --
    confirmed directly as a real failure mode of `ReducedErrorPruning`
    on a noisy pruning sample, see its own docstring; this is a second
    layer of defense in case the search itself ever concludes an
    unconditional rule is "best", not just pruning degrading to one).
    These are all direct checks in `fit` itself, not routed through
    `stop_covering`, for the same reason `RuleSearch` keeps its own
    floor unwrapped -- see that class's docstring. `max_rules` is a
    separate, hardcoded resource budget for the same reason a search
    keeps `max_conditions` outside the criterion system too -- a hard
    cap, not a quality judgment.

    **Which model `fit` builds** (see `pyrulearn.learners.RuleLearner.fit`):

    - `fit(data, model=ConceptModel, label="a")` -- the definition of one
      concept: the covering loop for ``"a"`` vs. everything else. Setting
      `target_class="a"` at construction makes this the default (`label`
      then comes from `target_class`).
    - `fit(data)` with no `target_class` -- the family multi-class default
      (`_MULTICLASS_DEFAULT`): `ConceptSet` (one-vs-rest) for the SeCo
      family, `FlatRuleSet` (one seed-covering loop) for `AQR`.
    - `fit(data, model=ConceptSet | ConceptCascade | PairwiseModel)` --
      one-vs-rest / ordered peeling / round robin, via `DecomposingLearner`
      (each fits per-class copies through the covering loop).
    - `fit(data, model=FlatRuleSet)` -- `_seed_covering_fit`: one covering
      loop over all classes at once (random uncovered example -> its label
      as the head -> learn -> remove -> repeat; AQ's multi-class covering).
    - `model=DecisionList` reaches via the `FlatRuleSet -> DecisionList`
      converter.

    `SeCo` takes no default-prediction or combiner arguments -- it sets
    sensible ones on the result (`MajorityClass(data)` fallback; `"max"` /
    ``"list"`` combiner) and leaves the rest to the caller. Reassign
    either on the returned model.

    Learned rules carry no declarative weight -- `combiner = "max"`
    (`HeuristicMaxCombiner`, `Laplace` by default) scores each rule from
    its own *measured* `stats()` at predict time instead, the same
    default `sort_rules`/`covered_by` use for inspection; pass
    `combiner=HeuristicMaxCombiner(SomeHeuristic())` for a different one.

    `optimization` (a `RuleSetOptimizer`, default `None`) runs once the
    covering loop is done, reworking the whole class's rules before they
    become a `FlatRuleSet` -- RIPPER's optimization phase. Only in the
    binary path (`target_class` set, or a per-class stage of a decomposition);
    ignored by `model=FlatRuleSet` seed-covering, which has no single
    target class.
    """

    #: model type `fit(data)` builds with no `model=` and no `target_class`.
    #: `ConceptSet` (one-vs-rest) for the SeCo family; `AQR` overrides to
    #: `FlatRuleSet` (one seed-covering loop).
    _MULTICLASS_DEFAULT: type = ConceptSet

    def __init__(
        self,
        single_rule_learner: SingleRuleLearner,
        target_class: Any = None,
        stop_covering: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = None,
        optimization: Optional[RuleSetOptimizer] = None,
    ):
        self.single_rule_learner = single_rule_learner
        self.target_class = target_class
        self.stop_covering = stop_covering
        self.max_rules = max_rules
        self.optimization = optimization
        self.random_state = random_state

    def _default_model(self, data: BooleanDataRepresentation) -> type:
        """`fit(data)` with no `model=`: one concept (`target_class` set)
        or the family's multi-class default (`ConceptSet`; `FlatRuleSet`
        for `AQR`)."""
        return ConceptModel if self.target_class is not None else self._MULTICLASS_DEFAULT

    def _fit_binary(self, data: BooleanDataRepresentation, positive: Any,
                    negative: Any = None) -> ConceptModel:
        """The decomposition primitive: a `ConceptModel` for `positive`,
        with `negative` (a pair sub-problem) or `MajorityClass` (one-vs-
        rest) as the fallback."""
        return self._fit_covering(data, label=positive, fallback=negative)

    def _seed_covering_fit(self, data: BooleanDataRepresentation) -> FlatRuleSet:
        """`model=FlatRuleSet`: one separate-and-conquer loop over *all*
        classes at once. Repeatedly pick a uniformly random still-uncovered
        example, take its own label as the rule head, learn the best rule
        for it (seeded on that example, `SeedExample(strategy="index")`),
        and remove everything the rule covers. This is AQ's multi-class
        covering, and the same per-example seeding `pyrulearn.learners.pylord.PyLORD`
        does -- but inside a covering loop rather than once per row. Best
        paired with a consistency-filtered learner (`AQR`): consistent
        per-class rules never conflict, so the arbitrary seed order
        doesn't matter."""
        rng = np.random.default_rng(self.random_state)
        remaining = np.ones(data.n_samples, dtype=bool)
        rules: List[Rule] = []
        while np.any(remaining):
            if self.max_rules is not None and len(rules) >= self.max_rules:
                break
            seed = int(rng.choice(np.flatnonzero(remaining)))
            target = data.y[seed]
            rule = self.single_rule_learner.learn_one_rule(
                data, target, remaining,
                space_init=SeedExample(strategy="index", index=seed),
            )
            if rule is not None and rule.length() > 0:
                stats = RuleStats.from_rule(rule, data, target, example_mask=remaining)
                stop = self.stop_covering is not None and self.stop_covering.evaluate(
                    rule, stats, data, target, remaining
                )
                if stats.tp > 0 and not stop:
                    rules.append(rule)
                    remaining = remaining & ~rule.covers_data_packed(data)
            remaining[seed] = False  # drop the seed either way -> guaranteed progress

        # each rule was consistent *within its covering scope* but can pick up
        # negatives earlier rules had already removed, so two classes' rules
        # can overlap on a later example -- `combiner="list"` breaks that by
        # learn order (the big, clean rules come first in a covering loop),
        # the honest reading of a sequential-covering result.
        rules = annotate_rules(rules, data)
        model = FlatRuleSet(rules, default_prediction=MajorityClass(data), combiner="list")
        return annotate_default_rule(model, data)

    def _covering_loop(self, data: BooleanDataRepresentation, target: Any) -> List[Rule]:
        """The separate-and-conquer covering loop for one class `target`
        vs. the rest, plus `optimization` if set. Shared by the binary
        `_fit_default` path and `_fit_covering`."""
        positive = data.y == target
        remaining = np.ones(data.n_samples, dtype=bool)
        rules: List[Rule] = []

        while True:
            if self.max_rules is not None and len(rules) >= self.max_rules:
                break
            if not np.any(positive & remaining):
                break  # every positive example is already covered

            rule = self.single_rule_learner.learn_one_rule(data, target, remaining)
            if rule is None:
                break  # the single-rule search found nothing acceptable
            stats = RuleStats.from_rule(rule, data, target, example_mask=remaining)
            if stats.tp == 0:
                break  # covers nothing new -- adding it would be pure waste
            if rule.length() == 0:
                break  # unconditional -- the default-rule mechanism covers this
            if self.stop_covering is not None and self.stop_covering.evaluate(
                rule, stats, data, target, remaining
            ):
                break

            rules.append(rule)
            remaining = remaining & ~rule.covers_data_packed(data)

        if self.optimization is not None:
            rules = self.optimization.optimize(
                rules, data, target, self.single_rule_learner, example_mask=None,
            )
        return rules

    @produces(ConceptModel)
    def _fit_covering(self, data: BooleanDataRepresentation, *,
                      label: Any = None, fallback: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")`: the definition of
        one concept -- the binary covering loop for `label` (or
        `self.target_class`). `fallback` is the negative prediction for
        uncovered rows (`MajorityClass(data)` if not given)."""
        if data.y is None:
            raise ValueError("SeCo needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        rules = annotate_rules(self._covering_loop(data, target), data)
        default = fallback if fallback is not None else MajorityClass(data)
        return annotate_default_rule(ConceptModel(rules, label=target, default_prediction=default), data)

    @produces(SingleRule)
    def _fit_one_rule(self, data: BooleanDataRepresentation, *, label: Any = None) -> SingleRule:
        """`fit(data, model=SingleRule, label="a")`: just the single best
        rule the search finds for `label` -- one `learn_one_rule` call, no
        covering loop, no optimization."""
        if data.y is None:
            raise ValueError("SeCo needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=SingleRule needs label= (or target_class set)")
        all_rows = np.ones(data.n_samples, dtype=bool)
        rule = self.single_rule_learner.learn_one_rule(data, target, all_rows)
        if rule is None:
            rule = Rule([], target=target, dataspec=data.spec)
        sr = SingleRule(rule, default_prediction=MajorityClass(data))
        sr.stats(data)
        return sr

    @produces(FlatRuleSet)
    def _fit_seed_covering(self, data: BooleanDataRepresentation, **kw) -> FlatRuleSet:
        """`fit(data, model=FlatRuleSet)`: one AQ-style covering loop over
        all classes at once (`_seed_covering_fit`)."""
        return self._seed_covering_fit(data)


class CN2(SeCo):
    """CN2 (Clark & Niblett, 1989), in the Laplace-heuristic form from
    Clark & Boswell, 1991, as a `SeCo` instantiation -- a first example
    of the pattern this module is meant to support: a named algorithm is
    just a `SeCo` subclass whose constructor picks specific building
    blocks as defaults, while still accepting every one of them as an
    override.

    Two choices define CN2 here:
    - **search heuristic**: `Laplace` -- Clark & Boswell's replacement
      for original CN2's entropy-based one.
    - **significance test, as a *stopping* criterion**: CN2's own
      description (Clark & Boswell) checks, inside the search itself,
      whether the round's best candidate is still statistically
      significant (`LikelihoodRatio`, CN2's own G-test statistic) --
      and if not, halts and returns the best significant rule found so
      far, rather than testing only the search's finished answer once
      at the end. That's exactly `BeamSearch`'s `stopping` (see
      `PrePruningCriterion`/`BeamSearch`'s own docstrings): `evaluate`
      is "no longer significant" (`LikelihoodRatio() < threshold`), and
      the default `polarity=True` reads correctly (reject the
      insignificant ones). Both `stopping` and `filtering` refuse to
      *return* an insignificant rule -- if a covering iteration never
      reaches a significant one, the search returns `None` and `SeCo`
      ends the loop, so an over-strict threshold shows up as a short or
      empty ruleset either way. They differ only in reach: `stopping`
      halts the search the moment the round's best candidate goes
      insignificant, so it can miss a rule that would have regained
      significance a few refinements deeper -- `mode="filtering"` runs
      the search out and can still find that rule. Pass
      `significance_threshold=None` to disable the criterion entirely
      (every rule the search lands on is accepted, same as calling
      `SeCo` directly with a `Laplace` heuristic and no `stopping`).

    `significance_threshold` defaults to 3.841, the chi-squared critical
    value at alpha=0.05 with 1 degree of freedom -- the significance
    level most commonly cited for CN2's own test; there's no single
    "the" CN2 default in the literature, so treat this as a reasonable
    starting point, not a fact about the original algorithm.

    `beam_width` defaults to 5 (the original paper's "star size"); pass
    `search` directly instead for anything beyond adjusting beam width
    (e.g. `max_conditions`, or a future non-beam `RuleSearch`) --
    `beam_width` is ignored if `search` is given explicitly.

    `preparation`/`postprocessing` default to `NoSplit`/
    `NoPostProcessing` -- unlike RIPPER/IREP, CN2 has no grow/prune split
    or post-hoc single-rule pruning; the significance test above is its
    only quality gate.

    `filtering`/`stopping`, if passed explicitly, override
    `significance_threshold`'s default construction entirely -- `mode`
    is a shorthand for that swap without constructing the criterion by
    hand: `mode="stopping"` (the default) wires the significance test as
    `stopping=` (gate the result *and* halt early), `mode="filtering"`
    wires the identical criterion as `filtering=` instead (gate the
    result, run the search out). Ignored once `filtering`/`stopping` is
    passed explicitly, or if `significance_threshold=None` disables the
    criterion entirely.

    `fit(data)`'s default `ConceptSet` uses `MicroVoteCombiner`, not the
    family's generic `combiner="max"` -- see `_fit_one_vs_rest` below for
    why. Measured empirically to make little difference on real data
    (rows where covering rules actually disagree in target are rare, and
    even then the two combiners' accuracy differs by a fraction of a
    point either way); the point is matching what Clark & Boswell's own
    algorithm does, not a functional improvement.
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        beam_width: int = 5,
        significance_threshold: Optional[float] = 3.841,
        mode: str = "stopping",
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if mode not in ("stopping", "filtering"):
            raise ValueError(f"mode must be 'stopping' or 'filtering', got {mode!r}")
        heuristic = heuristic if heuristic is not None else Laplace()
        search = search if search is not None else BeamSearch(beam_width=beam_width)
        if stopping is None and filtering is None and significance_threshold is not None:
            criterion = ThresholdPrePruning(LikelihoodRatio(), significance_threshold, operator="<")
            if mode == "stopping":
                stopping = criterion
            else:
                filtering = criterion
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
        )

    @produces(ConceptSet)
    def _fit_one_vs_rest(self, data: BooleanDataRepresentation, **kw) -> ConceptSet:
        """Clark & Boswell (1991)'s own unordered CN2 resolves a clash
        between rules of different classes covering the same row not by
        picking the single best-scoring rule (the family's generic
        `combiner="max"`) but by summing each rule's own covered-training-
        example class distribution and predicting the largest total --
        `pyrulearn.combiners.MicroVoteCombiner`. Only this one producer is
        overridden: `model=ConceptModel` (one concept, nothing to
        reconcile) and `model=FlatRuleSet` (AQ-style seed covering, not
        CN2's own induction shape) keep the family's `"max"` default."""
        model = super()._fit_one_vs_rest(data, **kw)
        model.combiner = MicroVoteCombiner()
        return model


class AQR(SeCo):
    """AQR (Clark & Niblett, 1989) -- their reimplementation of Michalski's
    AQ, the algorithm CN2 was designed against, here as `CN2`'s sibling
    `SeCo` subclass. Three choices define it:

    - **search-space start: a single-example seed** (`SeedExample`). Each
      rule is grown as a generalization of one still-uncovered positive
      example, never from the fully-open empty rule -- the defining
      difference from `CN2`'s general-to-specific beam. See
      `SeedExample`'s docstring; `seed_strategy`/`random_state` pick
      which uncovered positive.

    - **evaluation: a `LEF`** (Michalski's Lexicographic Evaluation
      Functional, `pyrulearn.heuristics.LEF`). Clark & Niblett describe
      AQ as ranking a complex by *the number of positive examples it
      covers* -- `CoveredPositives`. That alone is a poor way to trim the
      partial star mid-search (it can't tell a nearly-consistent complex
      from an over-general one at the same positive coverage), so the
      default LEF adds two tie-breaks that are standard in Michalski's
      own AQ LEFs and inert at the point a consistent rule is finally
      chosen: `CoveredNegatives` (minimize covered negatives -- steers
      the beam toward completable complexes, the job negative-guided
      star growth does in classical AQ) then `MinimalLength` (minimize
      selectors). Pass any `RuleHeuristic` as `heuristic=` to change it
      -- a bare `CoveredPositives()` for the letter of the paper, a
      different `LEF`, or something else entirely (this is exactly
      `CN2`'s `heuristic=` override, which also accepts a `LEF`).

    - **stopping: consistency, as a *filtering* criterion**. AQR requires
      every rule to cover **no** negative examples -- it has no
      significance test and no noise tolerance (that lack is the whole
      point of Clark & Niblett's comparison: AQR overfits noisy data,
      CN2's significance test does not). That requirement is
      `ThresholdPrePruning(CoveredNegatives(), 0, "<")` as `filtering=`:
      a rule with `fp > 0` is never returned, and if a covering
      iteration can't reach a consistent rule at all, the search returns
      `None` and `SeCo` ends -- so un-fittable (e.g. noisy) data shows up
      as a short or empty ruleset. Pass `require_consistency=False` to
      drop the gate, or `filtering=`/`stopping=` to supply your own.

    The `SeCo` covering loop needs no adjustment for AQR: because every
    accepted rule is consistent, "remove every example the rule covers"
    removes only positives, and every negative stays in scope for every
    subsequent rule -- exactly classical AQ's covering.

    **What this is not (yet):** the search here is `BeamSearch` over the
    seed-restricted feature space, not AQ's literal *star* construction
    (grow the partial star by "add a selector that excludes a specific
    still-covered negative", trim to `maxstar` after each negative). With
    the consistency filter and a negatives-aware LEF the rules found are
    materially the same, and `beam_width == maxstar`, but the classical
    star is more efficient (its per-step candidate set is only the
    selectors that distinguish the seed from some covered negative) and
    *is* by definition "the set of maximally-general consistent
    complexes". A dedicated star `RuleSearch` can be passed as `search=`
    when one exists.

    `beam_width`/`maxstar` default to 5 (the paper's star size); ignored
    if `search` is passed explicitly. `preparation`/`postprocessing`
    default to `NoSplit`/`NoPostProcessing` -- AQR (in the 1989 paper)
    has neither a grow/prune split nor rule truncation.

    **Multi-class:** `AQR`'s `fit(data)` default is `model=FlatRuleSet` --
    one covering loop over all classes (`SeCo._seed_covering_fit`): pick a
    random uncovered example, take its label as the head, learn a
    consistent rule seeded on it, remove what it covers, repeat. That's
    AQ's own multi-class covering; consistent per-class rules never
    conflict, so the random seed order is immaterial. Pass
    `model=ConceptSet` / `ConceptCascade` / `PairwiseModel` for a
    decomposition instead, or `model=ConceptModel, label=...` (or a
    `target_class`) for one binary problem. `random_state` seeds the
    example picks.
    """

    _MULTICLASS_DEFAULT = FlatRuleSet

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        maxstar: int = 5,
        seed_strategy: str = "first",
        random_state: Optional[int] = None,
        require_consistency: bool = True,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        stop_covering: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
    ):
        heuristic = heuristic if heuristic is not None else LEF(
            CoveredPositives(), CoveredNegatives(), MinimalLength()
        )
        search = search if search is not None else BeamSearch(beam_width=maxstar)
        space_init = space_init if space_init is not None else SeedExample(seed_strategy, random_state)
        if filtering is None and stopping is None and require_consistency:
            filtering = ThresholdPrePruning(CoveredNegatives(), 0, operator="<")
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            stop_covering=stop_covering,
            max_rules=max_rules,
            random_state=random_state,
        )


class PFoil(SeCo):
    """PFOIL (Mooney, 1995) -- FOIL's information-gain search heuristic
    (`FoilGain`) run propositionally via hill climbing, paired with
    Quinlan's own MDL-based encoding-length restriction
    (`pyrulearn.pruning.EncodingLengthRestriction`, see its own
    docstring and Quinlan, 1990, *Machine Learning* 5(3):239-266, p.
    251, verified directly against the primary source) as the quality
    gate, in `CN2`'s significance test's place.

    Two choices define PFoil here:
    - **search + heuristic**: `GainAscentHillClimbing` + `FoilGain` --
      `FoilGain` is a `GainHeuristic`, so `GainAscentHillClimbing` is the
      only valid search for it (see its docstring, and `BeamSearch`'s,
      for why gain scores can't be ranked across candidates in flight).
    - **stopping criterion**: `EncodingLengthRestriction`, as
      `stopping=` -- stop extending a clause once it costs more bits to
      write down than to list the positives it covers. Its default
      `polarity=True` already gives the same "leave" reading `CN2`'s
      significance test uses: `reject()==evaluate()`.

    Mooney's own PFOIL (the propositional restriction of FOIL described
    in Mooney, 1995) uses `FoilGain` as its search heuristic but does
    *not* itself apply an MDL stopping rule -- that's Quinlan's own
    contribution from the original (relational) FOIL paper, reused here
    since it needed no relational machinery to restate propositionally.
    Pass `mdl_stopping=False` (and no `filtering=`) to recover Mooney's
    own unrestricted-growth PFOIL exactly -- growth then only stops at
    `GainAscentHillClimbing`'s own unconditional floor (no move improves
    gain at all).

    `filtering`/`stopping`, if passed explicitly, override
    `mdl_stopping`'s default construction entirely -- e.g. pass
    `filtering=EncodingLengthRestriction()` to try it as a filter
    instead (only ever restricts which visited rule is eligible to be
    returned, never changes which move the walk takes -- see
    `GainAscentHillClimbing`'s own docstring).
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[GainHeuristic] = None,
        mdl_stopping: bool = True,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        heuristic = heuristic if heuristic is not None else FoilGain()
        search = search if search is not None else GainAscentHillClimbing()
        if stopping is None and filtering is None and mdl_stopping:
            stopping = EncodingLengthRestriction()
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
        )


class PFossil(SeCo):
    """FOSSIL (Fürnkranz, 1994): `Correlation` (the four-field/phi
    correlation coefficient) as the rule-refinement heuristic, with
    `correlation_threshold` (default 0.3, FOSSIL's own published value)
    as the quality gate.

    **Hill climbing, not beam search.** FOSSIL's own search is a
    top-down hill-climbing loop -- add the one condition that most
    improves correlation, repeat, stop when nothing improves it -- and
    that's what `PFossil` now does: `HillClimbing` over `Correlation()`
    directly (`HillClimbing` ranks any plain `RuleHeuristic`; no
    `DeltaGain` wrapping needed). `HillClimbing`'s local-optimum stop --
    move on only to a child that beats the current rule's own
    correlation -- *is* "stop when correlation would no longer
    increase", so the search halts at the nearest correlation maximum,
    typically after a handful of conditions.

    This replaces an earlier `BeamSearch(beam_width=5)` + `Correlation`
    default that behaved badly on two fronts, both confirmed on the
    binary-UCI benchmark suite (`examples/demo_seco_learners_comparison`):

    - *Runaway search.* `Correlation` evaluated at the beam's top
      candidate rarely drops below 0.3 as conditions are added (near-
      redundant interval literals barely move it), and its optimistic
      `(tp, 0)` bound sits near 1.0, so neither `stopping` nor
      `BeamSearch`'s optimistic pruning fired -- the beam ran to
      feature-mask exhaustion, exploring rules 40-90 conditions deep
      before returning a short one. Hill climbing stops at the
      correlation peak (~5 conditions), 5-25x faster.
    - *Over-general rules.* `Correlation` (like `WRAcc`; symmetric about
      the ROC diagonal) rewards a rule for covering the positive class
      *and* correctly excluding the negatives -- but in a first-match
      decision list only precision-on-what-fires matters. The beam
      would settle on high-coverage, ~55%-precision rules; sequential
      covering then blanketed the data with the target class and
      accuracy collapsed toward its base rate. Greedily following the
      correlation gradient from the empty rule adds negatives-removing
      conditions first and stops earlier, landing on tighter rules.

    `correlation_threshold` is wired as `filtering` by default
    (`mode="filtering"`): the local-optimum stop already decides *when to
    stop climbing*; the threshold only decides *whether the rule the
    climb lands on is worth keeping* -- if its correlation is below the
    threshold the search returns `None` and `SeCo` ends its covering
    loop. `mode="stopping"` uses the same threshold but also halts the
    climb the moment correlation dips below it (returning the rule from
    just before, or `None` if there wasn't one yet) -- earlier than the
    natural peak, so it tends to underfit (often one rule); kept for
    parity with `CN2`'s own `mode=`. `evaluate()` is
    `ThresholdPrePruning(Correlation(), correlation_threshold, "<")`
    either way -- and either way a sub-threshold rule is never returned
    (before 2026, `mode="stopping"` did append them, which is what
    wrecked accuracy). Pass `correlation_threshold=None` to drop the
    gate entirely (every rule the climb lands on is accepted).

    Beam search -- the pre-2026 default -- is still available explicitly:
    `PFossil(target_class=..., search=BeamSearch(beam_width=5),
    heuristic=Correlation())`.
    """

    def __init__(
        self,
        target_class: Any = None,
        heuristic: Optional[RuleHeuristic] = None,
        correlation_threshold: Optional[float] = 0.3,
        mode: str = "filtering",
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
        space_init: Optional[SearchSpaceInit] = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if mode not in ("stopping", "filtering"):
            raise ValueError(f"mode must be 'stopping' or 'filtering', got {mode!r}")
        search = search if search is not None else HillClimbing()
        if heuristic is None:
            heuristic = Correlation()
        if stopping is None and filtering is None and correlation_threshold is not None:
            criterion = ThresholdPrePruning(Correlation(), correlation_threshold, operator="<")
            if mode == "stopping":
                stopping = criterion
            else:
                filtering = criterion
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=preparation,
            postprocessing=postprocessing,
            space_init=space_init,
            filtering=filtering,
            stopping=stopping,
        )
        super().__init__(
            single_rule_learner=single_rule_learner,
            target_class=target_class,
            max_rules=max_rules,
            random_state=random_state,
        )


class Pypper(DecomposingLearner, NativeRuleLearner):
    """Pypper -- a re-implementation of RIPPER (Cohen, 1995, *Fast Effective
    Rule Induction*), not a port of Cohen's code: IREP\\* growth-and-pruning
    plus the `ReplaceReviseOptimization` phase, run per class. See the
    covering-loop stop below for how it differs from the original.

    `fit(data)` returns a `pyrulearn.models.ConceptCascade`: rules for the
    rarest class first, then the next, ..., the most frequent class as the
    catch-all default -- exactly RIPPER's class handling
    (least-frequent-first ordered peeling). Other model types via
    `fit(data, model=...)`:

    - `ConceptModel` -- binary Pypper for one class (`label=` / a
      `target_class`), IREP\\* + optimization, no peeling.
    - `SingleRule` -- just one grown-and-pruned rule for a class, no
      covering loop and no optimization phase.
    - `ConceptSet` -- one-vs-rest Pypper (each class its own full loop).
    - `PairwiseModel` -- round-robin Pypper.
    - `FlatRuleSet` / `DecisionList` -- via the `ConceptSet -> FlatRuleSet`
      / `ConceptCascade -> DecisionList` converters.

    (Pypper has no `FlatRuleSet` seed-covering producer -- it is not a
    seed-covering algorithm -- but still reaches `FlatRuleSet` by
    flattening its one-vs-rest `ConceptSet`.)

    Per class, a `SeCo` covering loop where each rule is:
    - **grown** by `FoilGain` on a 2/3 growing split (`GrowPruneSplit`),
      with `GainAscentHillClimbing(stop_at_local_optimum=False)` -- IREP's
      grow phase: add the max-gain condition until the rule is consistent
      (or it would drop below two covered positives, RIPPER's `minNo`,
      wired as a `stopping` criterion), *not* stopping at a gain peak,
    - **pruned** -- trailing conditions dropped by `ReducedErrorPruning`
      on the 1/3 held-out split (`Precision`, monotone with RIPPER's own
      ``(p-n)/(p+n)`` rule-value metric),
    then the whole class's rules go through `ReplaceReviseOptimization(k)`
    (`k=2` passes by default).

    The covering loop stops (`stop_covering=`) on whichever fires first
    of `pyrulearn.pruning.EncodingLengthRestriction` (FOIL's per-rule MDL
    stop) and IREP's rule (Fürnkranz & Widmer, 1994): halt once a pruned
    rule's precision on the covering scope drops below 0.5 -- a rule no
    better than a coin flip, and every rule after it, only degrades the
    set. This stands in for Cohen's exact "total ruleset DL is >64 bits
    over the minimum seen" rule (which would also need a retroactive
    trim-back the per-rule `PrePruningCriterion` interface can't express);
    the IREP stop is cruder but is itself a published stopping rule, and
    it fixes the same failure -- a covering loop that keeps tacking on
    low-precision rules until the MDL-guided optimization latches onto
    one of them. There is also no post-optimization residual IREP\\* --
    see `ReplaceReviseOptimization`.

    `k` -- optimization passes. `prune_fraction` -- held-out split size
    for grow/prune (both IREP\\* and the optimization). `grow_heuristic`/
    `search` -- override the growth heuristic / search (defaults `FoilGain`
    and `GainAscentHillClimbing(stop_at_local_optimum=False)`; pass your
    own `search` and you own its `stop_at_local_optimum` too).
    `random_state` seeds every split and the class-order tie-break.
    """

    def __init__(
        self,
        k: int = 2,
        prune_fraction: float = 1.0 / 3.0,
        random_state: Optional[int] = None,
        max_rules: Optional[int] = None,
        grow_heuristic: Optional[RuleHeuristic] = None,
        search: Optional[RuleSearch] = None,
        target_class: Any = None,
    ):
        self.k = k
        self.prune_fraction = prune_fraction
        self.random_state = random_state
        self.max_rules = max_rules
        self.grow_heuristic = grow_heuristic
        self.search = search
        self.target_class = target_class

    def _stage_learner(self) -> SeCo:
        heuristic = self.grow_heuristic if self.grow_heuristic is not None else FoilGain()
        search = (
            self.search if self.search is not None
            else GainAscentHillClimbing(stop_at_local_optimum=False)
        )
        single_rule_learner = SingleRuleLearner(
            heuristic=heuristic,
            search=search,
            preparation=GrowPruneSplit(self.prune_fraction, self.random_state),
            postprocessing=ReducedErrorPruning(Precision()),
            stopping=ThresholdPrePruning(CoveredPositives(), 2, operator="<"),
        )
        return SeCo(
            single_rule_learner=single_rule_learner,
            stop_covering=AnyOf(
                EncodingLengthRestriction(),
                # IREP's stop (Fürnkranz & Widmer, 1994): once a pruned rule
                # is no better than a coin flip on the covering scope, adding
                # it -- and every rule after it -- only hurts.
                ThresholdPrePruning(Precision(), 0.5, operator="<"),
            ),
            max_rules=self.max_rules,
            optimization=ReplaceReviseOptimization(self.k, self.prune_fraction, self.random_state),
        )

    def _default_model(self, data: BooleanDataRepresentation) -> type:
        return ConceptModel if self.target_class is not None else ConceptCascade

    @produces(ConceptModel)
    def _fit_concept(self, data: BooleanDataRepresentation, *,
                     label: Any = None, fallback: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")` -- binary Pypper
        for one class: the configured IREP* + optimization covering loop,
        no peeling."""
        target = label if label is not None else self.target_class
        return self._stage_learner()._fit_covering(data, label=target, fallback=fallback)

    @produces(SingleRule)
    def _fit_one_rule(self, data: BooleanDataRepresentation, *, label: Any = None) -> SingleRule:
        """`fit(data, model=SingleRule, label="a")` -- one grown-and-pruned
        rule for `label`, no covering loop and no optimization phase."""
        target = label if label is not None else self.target_class
        return self._stage_learner()._fit_one_rule(data, label=target)

    def _fit_binary(self, data: BooleanDataRepresentation, positive: Any,
                    negative: Any = None) -> ConceptModel:
        return self._fit_concept(data, label=positive, fallback=negative)

