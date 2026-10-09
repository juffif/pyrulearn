"""
pyrulearn.learners.cpar
=======================

`CPAR` -- Classification based on Predictive Association Rules (Yin &
Han, SDM 2003): FOIL-style rule growing on weighted examples (their PRM,
Predictive Rule Mining) that follows every condition nearly as good as
the best one, so that one search can yield several rules; covered
positives are down-weighted instead of removed; and a class is predicted
by the mean expected accuracy of its best `k` rules covering the example.

Two implementations of the same algorithm, learning the same models:

- `CPAR` is assembled from the package's components -- the growing
  criterion is a `GainHeuristic` (`FoilGain`), the covering loop a
  `pyrulearn.learners.seco.CoveringStrategy` (`WeightedCovering` with
  `MultiplicativeReweighting` and `PositiveWeightBelow`), the search the
  SeCo searches' lazy, constraint-aware primitives, the prediction a
  `pyrulearn.combiners.TopKMeanCombiner`, multi-class handling
  `DecomposingLearner`. Any of them can be exchanged, and it runs on
  every data representation.
- `DenseCPAR` is the same algorithm specialized for speed: it scores
  every feature of a search node in one matrix product over a dense
  float copy of `data.X`, without constraint propagation, so it needs a
  `BooleanDataRepresentation`. Same parameters, same models, 2-4x
  faster than `CPAR`.

The README (*Native learning algorithms*, "Components first, specialized
versions second") uses the pair as the example of how pyrulearn trades
the two off.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..combiners import RuleCombiner, TopKMeanCombiner
from ..data import BooleanDataRepresentation
from ..heuristics import FoilGain, GainHeuristic, GeneralizedMEstimate, RuleStats
from ..models import ConceptModel, ConceptSet, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import DEFAULT_MAX_AUTO_CONVERT_CELLS, DecomposingLearner, NativeRuleLearner, produces
from .seco import (
    CoveringStrategy, MultiplicativeReweighting, PositiveWeightBelow, WeightedCovering,
    handle_for, live_open_children, materialize_child, parent_closure, rank_best_first,
    stats_from_handle,
)


class _CPARParameters:
    """The parameters and the parts both implementations share."""

    def __init__(
        self,
        decay: float = 2.0 / 3.0,
        min_total_weight: float = 0.05,
        min_gain: float = 0.7,
        gain_similarity: float = 0.99,
        k: int = 5,
        max_length: Optional[int] = None,
        max_rounds: int = 1000,
        heuristic: Optional[GainHeuristic] = None,
        covering: Optional[CoveringStrategy] = None,
        combiner: Union[str, RuleCombiner, None] = None,
    ):
        if not 0.0 < gain_similarity <= 1.0:
            raise ValueError(f"gain_similarity must be in (0, 1], got {gain_similarity}")
        if heuristic is not None and not isinstance(heuristic, GainHeuristic):
            raise ValueError(
                f"CPAR needs a GainHeuristic (scored against the rule before the new condition), got "
                f"{type(heuristic).__name__} -- wrap a plain heuristic as DeltaGain({type(heuristic).__name__}())"
            )
        reweighting = getattr(covering, "reweighting", None)
        if reweighting is not None and reweighting.assigns_rule_weights:
            raise ValueError(f"{type(reweighting).__name__} fits rule weights, which CPAR's rule sets "
                             "can't hold -- use a boosting learner (e.g. Slipper)")
        self.decay = decay
        self.min_total_weight = min_total_weight
        self.min_gain = min_gain
        self.gain_similarity = gain_similarity
        self.k = k
        self.max_length = max_length
        self.max_rounds = max_rounds
        self.heuristic = heuristic
        self.covering = covering
        self.combiner = combiner

    def _heuristic(self) -> GainHeuristic:
        return self.heuristic if self.heuristic is not None else FoilGain()

    def _covering(self) -> CoveringStrategy:
        if self.covering is not None:
            return self.covering
        return WeightedCovering(MultiplicativeReweighting(self.decay), PositiveWeightBelow(self.min_total_weight),
                                max_rounds=None)

    def _combiner(self, n_classes: int) -> Union[str, RuleCombiner]:
        if self.combiner is not None:
            return self.combiner
        return TopKMeanCombiner(GeneralizedMEstimate(m=n_classes, cost=1.0 / n_classes), k=self.k)

    def _covering_rounds(self, data: Any, positive: np.ndarray, base_w: np.ndarray,
                         grow: Callable[[np.ndarray], list], covers: Callable[[Any], np.ndarray]) -> list:
        """The outer loop for one class: until the covering strategy is
        done, one search (`grow`, on the data's row weights times the
        covering weights) and, for every rule it yields, one covering
        update. `max_rounds` caps the searches, not the rules. A rule
        found again still reweights but is kept once."""
        covering = self._covering()
        state = covering.start(data, positive)
        found: Dict[Any, None] = {}
        for _ in range(self.max_rounds):
            if covering.exhausted(state):
                break
            rules = grow(base_w * state.scope)
            if not rules:
                break
            for rule in rules:
                found.setdefault(rule, None)
                covering.update(state, covers(rule))
        return list(found)

    def _admits(self, gain: float, threshold: Optional[float]) -> bool:
        """Whether a child with this `gain` is followed: at least
        `min_gain`, and once the best child is known, at least
        ``best * gain_similarity`` (`threshold`)."""
        return gain >= (self.min_gain if threshold is None else threshold)

    def _threshold(self, best: float) -> float:
        return max(best * self.gain_similarity, self.min_gain)


class CPAR(_CPARParameters, DecomposingLearner, NativeRuleLearner):
    """CPAR (Yin & Han 2003). For each class `c` against the rest:

    1. Grow rules on the current example weights, FOIL-style: starting
       from the empty rule, repeatedly add the condition with the highest
       weighted FOIL gain ``P* (log2 P*/(P*+N*) - log2 P/(P+N))`` (``P``,
       ``N``: the weights of the positives and negatives the rule covers,
       ``P*``, ``N*`` with the condition added), until no condition gains
       at least `min_gain`. Every other condition whose gain is within
       `gain_similarity` of the best (``gain >= best * gain_similarity``)
       starts a copy of the rule that is grown further the same way -- so
       one search can yield several rules.
    2. For each rule found, multiply the weights of the positives it
       covers by `decay` (`MultiplicativeReweighting`); negatives keep
       their weight.
    3. Repeat until the positives' total weight is below
       `min_total_weight` of the start (`PositiveWeightBelow`), or no rule
       is found.

    The model is a `ConceptSet` with one `ConceptModel` per class (rules
    found more than once appear once). A class is predicted by the mean
    expected accuracy ``(nc + 1) / (n + K)`` (``K`` classes; the rules'
    training stats) of its best `k` rules covering the example
    (`TopKMeanCombiner` with `GeneralizedMEstimate(m=K, cost=1/K)`); rows
    no rule covers get the majority class. `model=ConceptCascade` and
    `model=PairwiseModel` (`DecomposingLearner`) decompose differently.

    Defaults are the paper's: ``decay = 2/3``, ``min_total_weight =
    0.05``, ``min_gain = 0.7``, ``gain_similarity = 0.99``, ``k = 5``.
    `gain_similarity = 1` (only exact ties copied) is close to PRM without
    copying. Numeric attributes come already binarized
    (`pyrulearn.data.io.build_dataspec`); the data's row weights multiply
    with the covering weights. `max_length` (default none) caps rule
    length, `max_rounds` the number of searches per class.

    **Components.** Each step is a part that can be exchanged:

    - `heuristic` (default `FoilGain()`): the growing criterion. A
      `GainHeuristic`, since `min_gain` and `gain_similarity` are
      thresholds on an improvement over the rule before the condition;
      `DeltaGain(h)` makes one from any plain heuristic `h`.
    - `covering` (default ``WeightedCovering(MultiplicativeReweighting(
      decay), PositiveWeightBelow(min_total_weight))``, which is what
      `decay` and `min_total_weight` configure): how covered examples
      are reweighted, and when a class is done -- e.g.
      `AdditiveReweighting` (CN2-SD), or `RemovalCovering` for plain
      separate-and-conquer.
    - `combiner` (default the top-`k` mean above): how a prediction is
      made from the covering rules.
    - The search is built from `pyrulearn.learners.seco`'s primitives:
      `live_open_children` counts every open condition of a node at once,
      the heuristic's `batch_score` scores them in one call, and
      `materialize_child` builds only the children followed, with the
      constraint closure that removes everything a condition implies
      (fixing one numeric threshold removes the attribute's other
      thresholds too). That's why it works on every data representation.

    `DenseCPAR` is the same algorithm specialized for speed on a
    `BooleanDataRepresentation`: same parameters, same models, 2-4x
    faster. An implied condition never changes coverage, so its FOIL
    gain is 0, below `min_gain` -- `DenseCPAR` never picks one either,
    propagation only spares this version from scoring it.
    """

    def _default_model(self, data: Any) -> type:
        return ConceptSet

    def _fit_binary(self, data: Any, positive: Any, negative: Any = None) -> ConceptModel:
        """The rules for `positive` against the rest of `data` (a pair
        sub-problem: `data` holds only the two classes)."""
        if data.y is None:
            raise ValueError("CPAR needs data.y")
        y = np.asarray(data.y)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        rules = self._covering_rounds(
            data, y == positive, base_w,
            grow=lambda w: self._grow(data, positive, w),
            covers=data.coverage,
        )
        return ConceptModel(annotate_rules(rules, data), label=positive, default_prediction=negative)

    @produces(ConceptSet)
    def _fit_one_vs_rest(self, data: Any, **kw) -> ConceptSet:
        if data.y is None:
            raise ValueError("CPAR needs data.y")
        n_classes = len(np.unique(np.asarray(data.y)))
        if n_classes < 2:
            raise ValueError("CPAR needs at least two classes")
        model = super()._fit_one_vs_rest(data, **kw)
        model.combiner = self._combiner(n_classes)
        return model

    def _grow(self, data: Any, target_class: Any, w: np.ndarray) -> List[Rule]:
        """The rules one search yields on the weights `w`: the best-gain
        lineage and every copy branched off at a nearly-as-good
        condition."""
        weighted = data.with_weights(w)
        dataspec = weighted.spec
        heuristic = self._heuristic()
        results: List[Rule] = []
        seen: Set[Rule] = set()

        def grow(rule: Rule, mask, handle, closure, stats: RuleStats) -> None:
            if rule in seen:
                return
            seen.add(rule)
            if self.max_length is not None and rule.length() >= self.max_length:
                results.append(rule)
                return
            if stats.tp <= 0:
                return
            features, tps, fps, fns, tns, same, dead = live_open_children(weighted, target_class, mask, handle, stats)
            length = rule.length() + 1
            gains = heuristic.batch_score(RuleStats(tp=tps, fp=fps, fn=fns, tn=tns, length=length), stats)
            gains[same] = 0.0           # changes nothing: no gain over the rule itself
            chosen = []
            threshold = None
            for i in rank_best_first(gains).tolist():                 # stable: ties stay in feature order
                gain = float(gains[i])
                if not self._admits(gain, threshold):
                    break
                built = materialize_child(weighted, dataspec, rule, closure, mask, int(features[i]), handle)
                if built is None:
                    continue  # contradicts the rule: never a candidate
                if threshold is None:
                    threshold = self._threshold(gain)
                chosen.append((built, RuleStats(tp=tps[i], fp=fps[i], fn=fns[i], tn=tns[i], length=length)))
            if not chosen:
                if rule.length() > 0:
                    results.append(rule)
                return
            for (crule, cmask, chandle, cclosure), cstats in chosen[1:]:          # the copies
                grow(crule, cmask - dead, chandle, cclosure, cstats)
            (crule, cmask, chandle, cclosure), cstats = chosen[0]
            grow(crule, cmask - dead, chandle, cclosure, cstats)

        seed = Rule([], target=target_class, dataspec=dataspec)
        handle0 = handle_for(weighted, seed, None)
        stats0 = stats_from_handle(weighted, target_class, seed, handle0)
        grow(seed, frozenset(range(seed.n_features)), handle0, parent_closure(dataspec, seed), stats0)
        return results


class DenseCPAR(_CPARParameters, NativeRuleLearner):
    """`CPAR`, specialized for speed: same algorithm, same parameters,
    same models (see `CPAR` for both), 2-4x faster -- at the price of
    needing a `BooleanDataRepresentation` and its dense matrix, and of
    deciding multi-class one way only (one-vs-rest, a `ConceptSet`).

    **Data representation.** `NATIVE_REPRESENTATIONS = (BooleanDataRepresentation,)`
    -- `_fit_native` converts anything else via
    `NativeRuleLearner.ensure_representation`, same as `ENDER`. `_grow`
    scores every feature of a node at once, ``(wp * cov) @ Xf`` over a
    float copy of the whole matrix, and masks the ones already in the
    rule out of the *result*. It never removes an *implied* feature from
    consideration (no constraint propagation), so its open set stays
    close to full width the whole search -- exactly the regime where
    that wins: measured 15x faster than slicing to just the open
    features first, on `spambase` (`ROADMAP.md`'s "Design decisions" has
    the numbers and why this is the *opposite* regime from
    `BeamSearch`/`HillClimbing`).
    """

    #: see the "Data representation" paragraph above
    NATIVE_REPRESENTATIONS = (BooleanDataRepresentation,)

    def __init__(
        self,
        decay: float = 2.0 / 3.0,
        min_total_weight: float = 0.05,
        min_gain: float = 0.7,
        gain_similarity: float = 0.99,
        k: int = 5,
        max_length: Optional[int] = None,
        max_rounds: int = 1000,
        heuristic: Optional[GainHeuristic] = None,
        covering: Optional[CoveringStrategy] = None,
        combiner: Union[str, RuleCombiner, None] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(decay, min_total_weight, min_gain, gain_similarity, k, max_length, max_rounds,
                         heuristic, covering, combiner)
        self.max_auto_convert_cells = max_auto_convert_cells

    def _default_model(self, data: Any) -> type:
        return ConceptSet

    @produces(ConceptSet)
    def _fit_native(self, data: Any, **kw) -> ConceptSet:
        data = self.ensure_representation(data, self.max_auto_convert_cells, purpose="DenseCPAR's dense-matrix scoring")
        if data.y is None:
            raise ValueError("DenseCPAR needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("DenseCPAR needs at least two classes")
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        concepts = []
        for c in classes:
            bodies = self._rules_for(X, Xf, y == c, base_w)
            rules = [Rule(list(b), target=c, dataspec=data.spec) for b in bodies]
            concepts.append(ConceptModel(annotate_rules(rules, data), label=c))
        model = ConceptSet(concepts, default_prediction=MajorityClass(data), combiner=self._combiner(len(classes)))
        return annotate_default_rule(model, data)

    # -- one class -----------------------------------------------------------

    def _rules_for(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
                   base_w: np.ndarray) -> List[Tuple[int, ...]]:
        return self._covering_rounds(
            None, positive, base_w,
            grow=lambda w: self._grow(X, Xf, positive, w),
            covers=lambda body: np.all(X[:, list(body)], axis=1),
        )

    def _grow(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
              w: np.ndarray) -> List[Tuple[int, ...]]:
        """The rules one search yields on the weights `w`: the best-gain
        lineage and every copy branched off at a nearly-as-good condition."""
        heuristic = self._heuristic()
        wp, wn = w * positive, w * ~positive
        n_pos, n_neg = float(wp.sum()), float(wn.sum())
        results: List[Tuple[int, ...]] = []
        seen: Set[frozenset] = set()

        def grow(body: List[int], cov: np.ndarray) -> None:
            key = frozenset(body)
            if key in seen:
                return
            seen.add(key)
            if self.max_length is not None and len(body) >= self.max_length:
                results.append(tuple(body))
                return
            p, n = float(wp[cov].sum()), float(wn[cov].sum())
            if p <= 0:
                return
            p_star = (wp * cov) @ Xf
            n_star = (wn * cov) @ Xf
            parent = RuleStats(tp=p, fp=n, fn=n_pos - p, tn=n_neg - n, length=len(body))
            children = RuleStats(tp=p_star, fp=n_star, fn=n_pos - p_star, tn=n_neg - n_star, length=len(body) + 1)
            gain = np.where(p_star > 0, heuristic.batch_score(children, parent), -np.inf)
            if body:
                gain[body] = -np.inf
            best = float(gain.max())
            if not self._admits(best, None):
                if body:
                    results.append(tuple(body))
                return
            threshold = self._threshold(best)
            order = np.argsort(-gain, kind="stable")
            chosen = [int(f) for f in order if self._admits(float(gain[f]), threshold)]
            for f in chosen[1:]:                         # the copies
                grow(body + [f], cov & X[:, f])
            grow(body + [chosen[0]], cov & X[:, chosen[0]])

        grow([], np.ones(len(w), dtype=bool))
        return [tuple(sorted(b)) for b in results]
