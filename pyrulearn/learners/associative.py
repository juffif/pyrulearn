"""
pyrulearn.learners.associative
==================================

`CARMiner` mines class association rules (CARs); `CBA`
and `CMAR` (both defined at the bottom of this module) and `IDS`
(`pyrulearn.learners.ids`) each *consume* an already-annotated pool of
rules -- via the `RuleDistiller` mixin defined here -- to build their own,
smaller model. Mining and consuming are deliberately separate: none of
CBA/CMAR/IDS mine anything themselves. Pass `rules=` (a `FlatRuleSet` of
already-annotated `SingleRule`s -- from `CARMiner`
itself, from `pyrulearn.interfaces.sklearn.from_random_forest`, or from
anywhere else that produces one) to skip mining entirely; leave it
`None` and `RuleDistiller._resolve_rules` mines a default pool via
`CARMiner` at fit time, using the same
`min_support`/`min_confidence`/`max_len` hyperparameters CBA/CMAR/IDS
have always taken.

This split is possible because CBA-CB's own database-coverage pruning
(`coverage_select`), CMAR's significance filter/per-class pruning, and
IDS's submodular objective/optimizers never actually needed a `CAR` --
they only ever read a rule's own measured `stats()` (confidence,
support, significance) and call `data.coverage(rule)` for row-level
work, both of which work identically for a `SingleRule` pulled from
*any* `RuleModel.rules`, mined or not (`SingleRule` is an explicit
drop-in for a bare `Rule` wherever coverage/combining code reads one --
see `pyrulearn.models.SingleRule`'s docstring). The `CAR` dataclass and
its own mining-time `confidence`/`support` bookkeeping stay entirely
internal to `generate_cars`/`CARMiner` -- nothing
downstream ever sees a raw `CAR` again once it's wrapped into a `Rule`.

**Mining** (`generate_cars`) is Apriori-style level-wise frequent-itemset
mining, bounded by `max_len`: every frequent conjunction of features
("itemset"), together with its support broken down by *every* class it
was seen with (not just the majority -- so both `CBA` and `CMAR`, which
weigh confidence/significance differently, can use the same mining pass),
turned into one class association rule (CAR) per class whose confidence
clears `min_confidence`. Support counting goes through the ordinary
`data.coverage(rule)` primitive every representation implements, but
mining issues far more `coverage` calls per fit than a greedy search
ever does, so which representation you bring matters more here than for
`seco`/`pyrulearn.learners.pylord.PyLORD`:
`pyrulearn.data.representation.NListRepresentation.coverage` computes
support of an arbitrary conjunction with one vectorized word-AND over the
rarest literal's own N-list, never an `O(n_samples)` scan. `fit` therefore
*recommends* (and auto-converts to, below a size threshold) an
`NListRepresentation` -- see `ensure_nlist` -- but doesn't *require* one:
`generate_cars` only ever calls the universal `data.coverage(rule)`,
nothing about mining is algorithmically specific to `NListRepresentation`,
so `ensure_nlist` passes any other representation through as-is (with a
printed note) rather than refusing it outright -- correct, just
considerably slower without NList's near-O(1) coverage.

**Pruning** (`coverage_select`) is the shared database-coverage primitive
both CBA-CB and CMAR's own pruning are variations of: walk a
precedence-ordered rule list once, keeping a rule whenever it still
correctly classifies at least one previously-unclaimed training row
(`keep=CoveredPositives()`, CBA-CB's "M1" as printed and as `pyarc`
implements it -- see `coverage_select`; `CMAR` uses the same). `CBA` uses the
returned error trace to truncate to a single minimum-error prefix (a
`DecisionList`); `CMAR` just keeps every surviving rule for weighted
voting (a `FlatRuleSet`), so the "which prefix length is best" question
doesn't apply to it -- see each class's own docstring.

`CARMiner` itself is a full, directly-usable
`pyrulearn.learners.NativeRuleLearner`: `fit(data)` with no further
pruning returns the **raw, unpruned CAR pool** as a plain `FlatRuleSet`
-- the default combiner (`"max"`, i.e. `HeuristicMaxCombiner(Laplace())`
-- no `combiner=` passed), matching the default
`pyrulearn.interfaces.sklearn.from_random_forest`'s own `FlatRuleSet`
gets: one convention for "a raw pool of many small rules", regardless
of source.

**`RuleDistiller`** is the shared base `CBA`/`CMAR`/`IDS` mix in instead
of inheriting `CARMiner`: it owns `rules=`/
`min_support`/`min_confidence`/`max_len`/`max_auto_convert_cells`/
`target_class`, and `_resolve_rules(data, only_class=None)` -- the one
place "use the given pool, or mine a default one" is decided. See its
own docstring below.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .base import DecomposingLearner, NativeRuleLearner, produces
from ..combiners import HeuristicVoteCombiner, _rule_stats
from ..data import BooleanDataRepresentation, DataRepresentation, NListRepresentation
from ..heuristics import ChiSquare
from ..models import (
    ConceptModel, DecisionList, FlatRuleSet, MajorityClass, PooledRuleSet, RuleView, SingleRule,
    annotate_default_rule, annotate_rules,
)
from ..pruning import ThresholdPrePruning
from ..rule import Rule

if TYPE_CHECKING:
    from ..heuristics import RuleHeuristic

#: `n_samples * n_features` above which `ensure_nlist` refuses to
#: silently build an `NListRepresentation` from a `BooleanDataRepresentation`
#: -- no existing memory-footprint helper anywhere in this codebase to
#: anchor this on, so it's a fresh, tunable constant (`max_auto_convert_cells=`).
DEFAULT_MAX_AUTO_CONVERT_CELLS = 5_000_000


def ensure_nlist(data: Any, max_auto_convert_cells: int) -> Any:
    """`data`, converted to an `NListRepresentation` where that's cheap,
    else passed through unchanged -- mining is *recommended* to run on
    an `NListRepresentation` for speed, not *required* to for
    correctness (see below), so this no longer refuses other
    representations outright.

    An already-`NListRepresentation` (covers `PrePostNListRepresentation`
    too, via `isinstance`) passes through unchanged (`is` identity, no
    rebuild). A `BooleanDataRepresentation` auto-converts via
    `NListRepresentation.from_boolean` -- exactly as expensive as building
    one directly, so this is cheap and safe -- *below*
    `max_auto_convert_cells` (`n_samples * n_features`); past it, still
    raises rather than risk an unbounded-memory rebuild the caller didn't
    ask for (e.g. someone deliberately chose `SparseDataRepresentation`
    for a huge, sparse dataset specifically to avoid this
    representation's memory profile -- that memory concern is about the
    *conversion*, not about using the representation as given, so it
    doesn't apply to the case below).

    Anything else (a `SparseDataRepresentation`, or a custom subclass) is
    returned as-is, with a printed note: `generate_cars` only ever calls
    the universal `data.coverage(rule)` -- nothing about mining is
    algorithmically specific to `NListRepresentation` -- but mining
    issues far more `coverage()` calls per fit than a greedy search ever
    does, so a representation without `NListRepresentation`'s
    near-O(1) vectorized coverage will make mining considerably slower,
    never incorrect. Convert explicitly first if that matters.
    """
    if isinstance(data, NListRepresentation):
        return data
    if isinstance(data, BooleanDataRepresentation):
        cells = data.n_samples * data.spec.n_features
        if cells <= max_auto_convert_cells:
            print(
                f"{__name__}: auto-converting a {data.n_samples}x{data.spec.n_features} "
                f"({cells} cells) BooleanDataRepresentation to NListRepresentation "
                "for itemset mining."
            )
            return NListRepresentation.from_boolean(data)
        raise ValueError(
            f"refusing to silently auto-convert a {data.n_samples}x{data.spec.n_features} "
            f"({cells}-cell) BooleanDataRepresentation to NListRepresentation past "
            f"max_auto_convert_cells={max_auto_convert_cells} -- convert explicitly via "
            "NListRepresentation.from_boolean(data), or raise max_auto_convert_cells= if "
            "you know this fits in memory."
        )
    print(
        f"{__name__}: mining directly on a {type(data).__name__}, not NListRepresentation -- "
        "this will be considerably slower (mining issues far more coverage() calls per fit "
        "than a greedy search ever does); convert to NListRepresentation first if that matters."
    )
    return data


@dataclass(frozen=True)
class CAR:
    """One class association rule: `items` (sorted feature indices, a
    conjunction of positive literals -- pyrulearn's explicit-negation
    features are already separate items, so no negative literals are
    needed here) predicts `target`. `support` is this CAR's own actual
    support count (rows matching `items` AND labeled `target`);
    `itemset_support` is `items`' raw support (any class) -- `confidence
    = support / itemset_support`.
    """

    items: Tuple[int, ...]
    target: Any
    support: int
    itemset_support: int
    confidence: float
    #: `{class: rows matching `items` with that label}` for *every* class
    #: (shared by all CARs of one itemset) -- what `CARMiner`
    #: stamps a rule's measured stats from, without a coverage pass.
    class_counts: Optional[Dict[Any, int]] = field(default=None, compare=False, repr=False)

    @property
    def n_conditions(self) -> int:
        return len(self.items)


def _apriori_join(itemsets: Sequence[Tuple[int, ...]]) -> Set[Tuple[int, ...]]:
    """Classic Apriori candidate generation: two sorted `k`-itemsets that
    share their first `k-1` items join into one `(k+1)`-item candidate.
    Itemsets sorted lexicographically first, so itemsets sharing a prefix
    are contiguous -- the inner loop breaks as soon as the prefix stops
    matching, rather than comparing every pair."""
    ordered = sorted(itemsets)
    candidates: Set[Tuple[int, ...]] = set()
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a[:-1] != b[:-1]:
                break
            candidates.add(a + (b[-1],))
    return candidates


def generate_cars(
    data: DataRepresentation,
    min_support: float,
    min_confidence: float,
    max_len: int,
    only_class: Optional[Any] = None,
) -> List[CAR]:
    """CBA-RG-style rule generation: one CAR per (itemset, class) whose
    **rule support** -- rows matching the itemset *and* labeled with the
    class, i.e. the support of body + head, the standard definition --
    is at least `min_support` (a fraction of `data.n_samples`) and whose
    confidence (rule support / itemset support) clears `min_confidence`;
    itemset length <= `max_len`. More than one CAR per itemset is possible
    if several classes qualify. (An earlier version thresholded the
    itemset's support regardless of class, which admitted CARs whose
    class support was below `min_support`; `fim` and CBA-RG threshold the
    rule, as here.)

    The level-wise search prunes on the same quantity: a class-conditional
    support is anti-monotone (adding a condition can only shrink it), so
    an itemset survives to the next level only if *some* class (only
    `only_class`, if given) reaches `min_support` with it -- a "frequent
    ruleitem" in CBA-RG's terms. Consequently `only_class` now *does*
    restrict which itemsets are explored, not just which CARs are emitted.

    Only ever calls the universal `data.coverage(rule)` -- `data` can be
    any `DataRepresentation`, not just `NListRepresentation`, though
    mining issues far more `coverage` calls per fit than a greedy search
    ever does, so `NListRepresentation` (see `ensure_nlist`) matters more
    for speed here than elsewhere.
    """
    y = np.asarray(data.y)
    classes, codes = np.unique(y, return_inverse=True)
    class_list = classes.tolist()
    emit = [only_class] if only_class is not None else class_list
    min_support_count = max(1, math.ceil(min_support * data.n_samples))

    def class_counts(mask: np.ndarray) -> Dict[Any, int]:
        # every class, always (also with only_class): the full breakdown is what lets a
        # rule's measured stats be stamped from counts later, without a coverage pass
        binned = np.bincount(codes[mask], minlength=len(class_list)).tolist()
        return dict(zip(class_list, binned))

    def measure(items: Tuple[int, ...]) -> Optional[Tuple[int, Dict[Any, int]]]:
        """`(itemset support, per-class supports)` if `items` is a frequent
        ruleitem (some *emitted* class reaches `min_support_count`), else `None`."""
        mask = data.coverage(Rule(items, dataspec=data.spec))
        support = int(mask.sum())
        if support < min_support_count:  # cheap early-out: no class can reach it either
            return None
        counts = class_counts(mask)
        if max((counts.get(c, 0) for c in emit), default=0) < min_support_count:
            return None
        return support, counts

    level: Dict[Tuple[int, ...], Tuple[int, Dict[Any, int]]] = {}
    for f in range(data.spec.n_features):
        found = measure((f,))
        if found is not None:
            level[(f,)] = found

    all_frequent: Dict[Tuple[int, ...], Tuple[int, Dict[Any, int]]] = dict(level)
    k = 1
    while level and k < max_len:
        next_level: Dict[Tuple[int, ...], Tuple[int, Dict[Any, int]]] = {}
        for cand in _apriori_join(list(level.keys())):
            if not all(sub in level for sub in itertools.combinations(cand, k)):
                continue  # anti-monotone: every k-subset must already be a frequent ruleitem
            found = measure(cand)
            if found is not None:
                next_level[cand] = found
        all_frequent.update(next_level)
        level = next_level
        k += 1

    cars: List[CAR] = []
    for items, (support, counts) in all_frequent.items():
        for cls in emit:
            class_support = counts.get(cls, 0)
            if class_support < min_support_count:
                continue
            confidence = class_support / support
            if confidence >= min_confidence:
                cars.append(CAR(items=items, target=cls, support=class_support,
                                itemset_support=support, confidence=confidence,
                                class_counts=counts))
    return cars


def iterate_rules(rules: Sequence[Rule]):
    """Iterate `rules` for a one-pass consumer: a lazy `RuleView` (see
    `pyrulearn.pool`) streams each rule without caching it, so a pass over a
    pool of hundreds of thousands of rules keeps memory flat; anything else
    (a plain list) iterates as usual."""
    return rules.iter_transient() if isinstance(rules, RuleView) else iter(rules)


def coverage_select(
    rules_in_order: Sequence[Rule], data: DataRepresentation,
    keep: Optional["RuleHeuristic"] = None,
) -> Tuple[List[Rule], List[Tuple[int, Optional[Any]]]]:
    """The shared database-coverage walk CBA-CB and CMAR's own pruning
    both build on: walk `rules_in_order`, tracking the rows not yet
    claimed by an earlier **kept** rule. A rule is kept iff `keep`, a
    `pyrulearn.heuristics.RuleHeuristic` scored on the rule's stats over
    the *still-unclaimed rows only*, is positive; a kept rule claims *all*
    rows it covers, right or wrong.

    `keep` defaults to `CoveredPositives()` (`h = tp`): the rule must
    **correctly classify at least one still-unclaimed row** -- CBA-CB's
    "M1" as printed in Liu/Hsu/Ma (1998) and as implemented by `pyarc`
    (its `rule.marked`). A rule that is wrong on everything it newly
    covers is skipped and claims nothing, so a later, correct rule can
    still take those rows. (Verified against `pyarc` by a rule-by-rule
    diff -- this is the condition that reproduces its classifiers; an
    earlier version kept a rule on *any* new coverage, which I had
    mis-remembered as the paper's "temp != empty" test.)

    `keep=Coverage()` (`h = tp + fp`) is that earlier, *not*-CBA-CB
    behavior -- any new coverage, right or wrong. Nothing uses it now:
    `CMAR` briefly kept it for its per-class pruning, but switching to
    the default cut CMAR from 55 rules / 0.855 test accuracy to 8 rules /
    0.939 on `vote` (CBA: 0.947), so it's retained only as a documented
    contrast.

    Returns `(kept, trace)`: `kept` is the surviving rules, in order;
    `trace[i] = (errors_so_far, default_label)` is the total training
    error and the majority class among whatever's still uncovered,
    *after* keeping `kept[:i+1]` -- exactly what `CBA` needs to find its
    own minimum-error prefix (see `CBA`). A caller
    that pools every surviving rule for voting instead of committing to
    one first-match prefix (`CMAR`) can simply
    ignore `trace` and use `kept` as-is.
    """
    from ..evaluation import RuleStats  # local: avoids a load-order cycle, like sort_by_measured_precedence
    from ..heuristics import CoveredPositives

    if keep is None:
        keep = CoveredPositives()
    n = data.n_samples
    y = np.asarray(data.y)
    labels, label_counts = np.unique(y, return_counts=True)
    uncovered_counts: Dict[Any, int] = dict(zip(labels.tolist(), label_counts.tolist()))
    covered = np.zeros(n, dtype=bool)

    kept: List[Rule] = []
    errors_so_far = 0
    trace: List[Tuple[int, Optional[Any]]] = []

    for rule in iterate_rules(rules_in_order):
        if covered.all():
            break
        mask = data.coverage(rule) & ~covered
        tp = int(np.sum(mask & (y == rule.target)))
        fp = int(mask.sum()) - tp
        pos_left = uncovered_counts.get(rule.target, 0)
        n_left = n - int(covered.sum())
        stats = RuleStats(tp=tp, fp=fp, fn=pos_left - tp, tn=(n_left - pos_left) - fp,
                          length=len(rule.conditions))
        if keep.score(stats) <= 0:
            continue
        errors_so_far += fp
        for lbl, cnt in zip(*np.unique(y[mask], return_counts=True)):
            uncovered_counts[lbl] -= int(cnt)
        covered |= mask
        kept.append(rule)

        remaining = n - int(covered.sum())
        if remaining > 0:
            default_label = max(uncovered_counts, key=uncovered_counts.get)
            trial_error = errors_so_far + (remaining - uncovered_counts[default_label])
        else:
            default_label = None
            trial_error = errors_so_far
        trace.append((trial_error, default_label))

    return kept, trace


class CARMiner(DecomposingLearner, NativeRuleLearner):
    """Mines class association rules (CARs) via `generate_cars` and
    returns them as-is -- a large, unpruned, likely-redundant `RuleSet`,
    the raw material `CBA`/`CMAR`/`IDS` each consume (via `RuleDistiller`)
    and post-process differently. See the module docstring.

    - `min_support`/`min_confidence` (fractions in `(0, 1]`) -- the
      itemset/rule thresholds.
    - `max_len` -- hard cap on itemset length; a practical safeguard
      exhaustive mining alone doesn't provide once negation features
      roughly double the item count (see
      `pyrulearn.data.representation.NListRepresentation`'s own
      docstring).
    - `target_class` (default `None`) -- mine CARs for one class only
      (`fit(data)` then returns a `ConceptModel`) -- a one-vs-rest
      decomposition knob, not part of mining itself.
    - `max_auto_convert_cells` -- see `ensure_nlist`.
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int = 4,
        target_class: Optional[Any] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.target_class = target_class
        self.max_auto_convert_cells = max_auto_convert_cells

    def _mine_pool(
        self, data: Any, only_class: Optional[Any] = None, default_prediction: Any = None,
    ) -> Tuple[PooledRuleSet, Any]:
        """`ensure_nlist` + `generate_cars`, held as a `PooledRuleSet`: the
        mined pool as flat columns (~35 B/rule instead of ~1.1 KB), each
        rule built -- with its measured stats stamped from the CAR's own
        per-class counts, identical to what `annotate_rules` would predict
        over the data, but free -- only when it is first looked at (see
        `pyrulearn.pool`). Also returns the resolved representation
        (converted to `NListRepresentation` where that's cheap, else `data`
        unchanged -- see `ensure_nlist`) so a caller that goes on to do its
        own coverage-based pruning (`coverage_select`) never pays for -- or
        prints the auto-convert notice for -- a second conversion of the
        same `data`."""
        resolved = ensure_nlist(data, self.max_auto_convert_cells)
        target = only_class if only_class is not None else self.target_class
        cars = generate_cars(resolved, self.min_support, self.min_confidence, self.max_len,
                             only_class=target)
        classes, class_totals = np.unique(np.asarray(resolved.y), return_counts=True)
        totals = dict(zip(classes.tolist(), class_totals.tolist()))
        pool = PooledRuleSet.from_cars(cars, resolved.spec, totals, default_prediction=default_prediction)
        return pool, resolved

    def _mine(
        self, data: Any, only_class: Optional[Any] = None,
    ) -> Tuple[List[SingleRule], Any]:
        """`_mine_pool`, with every rule built up front as an ordinary list
        of measured `SingleRule`s -- for callers that want the eager form
        (e.g. a `ConceptModel`, which is stored as a plain list)."""
        pool, resolved = self._mine_pool(data, only_class)
        return list(pool.rules), resolved

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else FlatRuleSet

    @produces(FlatRuleSet)
    def _fit_native(self, data: Any, **kw) -> FlatRuleSet:
        if data.y is None:
            raise ValueError("CARMiner.fit needs data.y")
        model, _resolved = self._mine_pool(data, default_prediction=MajorityClass(data))
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("CARMiner needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        rules, _nlist = self._mine(data, only_class=target)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(rules, label=target, default_prediction=default)
        return annotate_default_rule(model, data)

    def _fit_binary(self, data: Any, positive: Any, negative: Any = None) -> ConceptModel:
        return self._fit_concept(data, label=positive, fallback=negative)


def sort_by_measured_precedence(rules: Sequence[Rule]) -> Sequence[Rule]:
    """A lazy `pyrulearn.pool.RuleView` is sorted vectorized from its count
    columns (no rule is built; returns another view). For any other
    sequence, a sorted list:

    Confidence desc, support desc, generality (fewer conditions) asc
    -- CBA's own precedence order (Liu/Hsu/Ma), generalized to read
    confidence (`Precision`) and support (`tp`) from each rule's own
    measured stats instead of `CAR`-mining-time fields. Mathematically
    identical to sorting the equivalent `CAR`s directly when a rule's
    stats happen to be measured against the data it would have been
    mined from (`CAR.confidence = class_support/itemset_support` *is*
    `Precision` on the same confusion; `CAR.support = class_support`
    *is* `tp`) -- and the only sensible version once the rule pool might
    not have come from mining at all (`RuleDistiller`'s `rules=`). Raises
    if a rule has no measured stats (`pyrulearn.combiners._rule_stats`)
    -- annotate first (`pyrulearn.models.annotate_rules`, or a `fit()`
    round trip, already does)."""
    if isinstance(rules, RuleView):
        return rules.precedence_sorted()

    from ..combiners import _rule_stats
    from ..heuristics import Precision

    def key(r: Rule) -> Tuple[float, float, int]:
        stats = _rule_stats(r)
        if stats is None:
            raise ValueError(
                f"{type(r).__name__}(target={r.target!r}) has no measured stats -- "
                "annotate it first (see pyrulearn.models.annotate_rules)"
            )
        return (-Precision().score(stats), -stats.tp, len(r.conditions))

    return sorted(rules, key=key)


class RuleDistiller:
    """Base for an algorithm (`CBA`/`CMAR`/`IDS`) that compresses or
    reorganizes a pool of class-association-rule-*shaped* rules -- an
    already-annotated `FlatRuleSet` -- into its own, smaller model.
    Never mines anything itself.

    - `rules` (default `None`) -- a `FlatRuleSet` (or any `RuleModel`)
      whose `.rules` are already-annotated `SingleRule`s. Pass one
      explicitly to skip mining entirely -- e.g.
      `pyrulearn.interfaces.sklearn.from_random_forest(...)`, often
      considerably cheaper than CAR mining, is a good first thing to
      try instead of the default below.
    - If `rules` is `None`, `_resolve_rules` mines a default pool via
      `CARMiner` at fit time, using `min_support`/
      `min_confidence`/`max_len`/`max_auto_convert_cells` -- these four
      are silently unused if `rules` is given directly; they exist only
      for this fallback.
    - `target_class` (default `None`) -- restricts `_resolve_rules` to
      one class's rules (`fit(data)` then returns a `ConceptModel`). If
      `rules` was given directly, this filters the given pool after the
      fact (it already exists); if mining a default pool, it restricts
      *what gets mined* via `CARMiner`'s own
      `target_class=` instead (cheaper: nothing is mined for other
      classes at all).
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int = 4,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
        target_class: Optional[Any] = None,
    ):
        self.rules = rules
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.max_auto_convert_cells = max_auto_convert_cells
        self.target_class = target_class

    def _resolve_rules(self, data: Any, only_class: Optional[Any] = None) -> Sequence[Rule]:
        """The candidate pool to compress: `self.rules.rules`, filtered
        to `only_class` if given (an externally-supplied pool can only
        be restricted after the fact), or -- if `self.rules` is `None`
        -- a freshly-mined default pool, restricted to `only_class` at
        mining time instead (cheaper).

        A lazy pool (`pyrulearn.pool.RuleView`, what a mined pool or any
        `PooledRuleSet` gives) is returned as the view itself, never copied
        into a list -- so nothing is built until the algorithm reaches it.
        Anything else comes back as a plain list."""
        if self.rules is not None:
            pool = self.rules.rules
            if isinstance(pool, RuleView):
                return pool.for_target(only_class) if only_class is not None else pool
            pool = list(pool)
            return [r for r in pool if r.target == only_class] if only_class is not None else pool
        miner = CARMiner(
            min_support=self.min_support, min_confidence=self.min_confidence, max_len=self.max_len,
            target_class=only_class, max_auto_convert_cells=self.max_auto_convert_cells,
        )
        pool, _resolved = miner._mine_pool(data, only_class)
        return pool.rules

    def _fit_binary(self, data: Any, positive: Any, negative: Optional[Any] = None) -> ConceptModel:
        return self._fit_concept(data, label=positive, fallback=negative)


class CBA(RuleDistiller, DecomposingLearner, NativeRuleLearner):
    """Classification Based on Associations (Liu, Hsu & Ma, KDD 1998):
    consume a pool of class association rules (a `FlatRuleSet`, via
    `RuleDistiller` -- either given directly or mined by default via
    `CARMiner`, the same shared miner `CMAR` and
    `pyrulearn.learners.ids.IDS` also consume from), sort them by
    precedence, then build a `pyrulearn.models.DecisionList` via CBA-CB's
    "M1" database-coverage selection -- a genuinely different
    classifier-building step from a covering loop or a greedy search (see
    `CBA._select`'s docstring for the exact, easy-to-get-wrong semantics).

    - **Precedence sort** (`sort_by_measured_precedence`) -- CBA's own
      criterion, specific to this algorithm (CMAR ranks by significance
      instead, never shares this): `r1` precedes `r2` iff `r1` has higher
      confidence; tie -> higher support; tie -> fewer conditions (more
      general).
    - **CBA-CB, "M1"** (`CBA._select`) -- walks the precedence-sorted rules
      via `coverage_select` (the shared database-coverage primitive: keep a
      rule only if it correctly classifies at least one still-unclaimed
      row, then claim every row it covers -- cross-checked rule-for-rule
      against `pyarc`), then truncates the kept sequence to the
      **minimum-total-training-error prefix** -- CBA's actual
      accuracy/complexity trade-off, not just "keep every rule that ever
      helped."

    Native multi-class already (each rule carries its own class head), so
    `fit(data)` with no `model=` builds a `DecisionList` directly.
    `DecomposingLearner` is mixed in for consistency with every other
    native learner here (`ConceptSet`/`ConceptCascade`/`PairwiseModel` via
    `target_class=`/`_fit_binary`, the latter provided by `RuleDistiller`).

    Constructor arguments (`rules`/`min_support`/`min_confidence`/`max_len`/
    `target_class`/`max_auto_convert_cells`) are `RuleDistiller`'s,
    unchanged.
    """

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else DecisionList

    @produces(DecisionList)
    def _fit_native(self, data: Any, **kw) -> DecisionList:
        if data.y is None:
            raise ValueError("CBA.fit needs data.y")
        ordered_rules = sort_by_measured_precedence(self._resolve_rules(data))
        kept, default_label = self._select(ordered_rules, data)
        default = default_label if default_label is not None else MajorityClass(data)
        model = DecisionList(annotate_rules(kept, data, copy=True), default_prediction=default)
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("CBA needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        ordered_rules = sort_by_measured_precedence(self._resolve_rules(data, only_class=target))
        kept, _default_label = self._select(ordered_rules, data)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(annotate_rules(kept, data, copy=True), label=target, default_prediction=default)
        return annotate_default_rule(model, data)

    @staticmethod
    def _select(ordered_rules: List[Rule], data: Any) -> Tuple[List[Rule], Optional[Any]]:
        """CBA-CB, "M1": `coverage_select` gives the kept-rule sequence
        and its error trace; the final classifier truncates to the
        prefix with the lowest recorded total error (ties -> the first,
        i.e. shortest, occurrence -- plain `min` over a list built in
        increasing-prefix-length order already does this)."""
        kept, trace = coverage_select(ordered_rules, data)
        if not trace:
            return [], None
        best_len = min(range(len(trace)), key=lambda i: trace[i][0]) + 1
        return kept[:best_len], trace[best_len - 1][1]


class CMAR(RuleDistiller, DecomposingLearner, NativeRuleLearner):
    """Classification based on Multiple Association Rules (Li, Han & Pei,
    2001): consume a pool of class association rules (a `FlatRuleSet`, via
    `RuleDistiller` -- either given directly or mined by default via
    `CARMiner`, the same shared miner `CBA` and
    `pyrulearn.learners.ids.IDS` also consume from), keep only the ones
    whose antecedent is *significantly* correlated with the class (a
    chi-square test, not merely confident), then predict by pooling every
    matching rule's vote, weighted by its own significance -- unlike
    `CBA`, which commits to a single first-match `DecisionList`.

    Deliberately not chasing every detail of the original paper -- per
    explicit direction, this reuses existing structure wherever possible,
    accepting "slight deviations" from the paper rather than reproducing
    it exactly:

    - **Significance filtering** reuses `pyrulearn.pruning.ThresholdPrePruning`
      with the `pyrulearn.heuristics.ChiSquare` heuristic -- the exact
      "score a heuristic against a critical value" machinery
      `pyrulearn.learners.seco.CN2` already uses for its own significance
      test (`LikelihoodRatio`), just applied to a flat pool of
      already-mined rules instead of gating a live search.
    - **Pruning** reuses `coverage_select` -- the same database-coverage
      walk `CBA`'s own CBA-CB step is built on -- but applied **once per
      predicted class**, not globally across the whole significant pool. A
      kept rule claims *every* row it covers, so globally, a kept rule of
      one class would claim rows a later rule of a *different* class could
      never vote on again -- the semantics a first-match `DecisionList`
      needs, and wrong for a voting ensemble (confirmed directly: reusing
      it globally measurably hurt accuracy). Run once per class, it cuts a
      same-class candidate set down to the rules that each correctly
      classify at least one still-unclaimed row of their own class (CMAR's
      paper: "correctly classifies at least one remaining object" -- the
      same condition as CBA-CB, i.e. the default `keep=CoveredPositives()`),
      while every class stays free to vote on any row its own surviving
      rules cover. (An earlier version used the any-new-coverage condition
      here, right or wrong: 55 rules / 0.855 test accuracy on `vote`, vs. 8
      rules / 0.939 with this one -- that, not the significance threshold,
      was why CMAR trailed CBA.) The paper's covering threshold delta=4 (an
      object leaves only after being covered by 4 rules) is not reproduced
      -- ours is delta=1 -- nor its subsumption pruning of a rule by a
      higher-precedence, no-worse general ancestor.
    - **Voting** reuses `pyrulearn.combiners.HeuristicVoteCombiner`, pooling
      every surviving rule and weighting each covering rule's vote by the
      same `ChiSquare` score -- CMAR's own "weighted chi-square"
      combination, essentially verbatim. The paper's own formula
      additionally normalizes by the single largest chi-square value seen
      anywhere in the rule set; not reproduced.

    Native multi-class already (each rule carries its own class head).
    `DecomposingLearner` is mixed in for consistency with every other
    native learner here, not because CMAR needs it.

    - `significance_threshold` (default `3.841`, the chi-squared
      critical value at alpha=0.05 with 1 degree of freedom -- the same
      default `pyrulearn.learners.seco.CN2` already uses for its own
      significance test) -- a rule is dropped unless its `ChiSquare`
      score clears this.
    - `yates_correction` -- passed straight to `ChiSquare`.
    - The rest (`rules`/`min_support`/`min_confidence`/`max_len`/
      `target_class`/`max_auto_convert_cells`) are `RuleDistiller`'s,
      unchanged.
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int = 4,
        significance_threshold: float = 3.841,
        yates_correction: bool = True,
        target_class: Optional[Any] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(rules, min_support, min_confidence, max_len, max_auto_convert_cells, target_class)
        self.significance_threshold = significance_threshold
        self.yates_correction = yates_correction

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else FlatRuleSet

    def _significant(self, rules: Sequence[Rule], heuristic: ChiSquare) -> List[Rule]:
        """Drop rules that aren't significant (`ThresholdPrePruning`,
        same pattern `CN2` uses for its own significance test), sorted by
        significance descending -- CMAR's own rule strength, in place of
        CBA's confidence-based precedence. One streaming pass
        (`iterate_rules`): over a lazy pool only the significant rules
        stay in memory."""
        gate = ThresholdPrePruning(heuristic, self.significance_threshold, "<")
        significant = [r for r in iterate_rules(rules) if not gate.reject(r, _rule_stats(r), None, r.target)]
        return sorted(significant, key=lambda r: -heuristic.score(_rule_stats(r)))

    def _prune(self, ordered_rules: List[Rule], data: Any, heuristic: ChiSquare) -> List[Rule]:
        """`coverage_select`, run once per predicted class -- see the
        module docstring for why per-class, not globally."""
        by_class: Dict[Any, List[Rule]] = {}
        for r in ordered_rules:
            by_class.setdefault(r.target, []).append(r)
        kept: List[Rule] = []
        for group in by_class.values():
            selected, _trace = coverage_select(group, data)
            kept.extend(selected)
        return sorted(kept, key=lambda r: -heuristic.score(_rule_stats(r)))

    @produces(FlatRuleSet)
    def _fit_native(self, data: Any, **kw) -> FlatRuleSet:
        if data.y is None:
            raise ValueError("CMAR.fit needs data.y")
        heuristic = ChiSquare(yates_correction=self.yates_correction)
        kept = self._prune(self._significant(self._resolve_rules(data), heuristic), data, heuristic)
        model = FlatRuleSet(kept, default_prediction=MajorityClass(data),
                            combiner=HeuristicVoteCombiner(heuristic))
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("CMAR needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        heuristic = ChiSquare(yates_correction=self.yates_correction)
        pool = self._resolve_rules(data, only_class=target)
        kept = self._prune(self._significant(pool, heuristic), data, heuristic)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(kept, label=target, default_prediction=default)
        return annotate_default_rule(model, data)
