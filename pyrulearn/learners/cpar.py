"""
pyrulearn.learners.cpar
=======================

`CPAR` -- Classification based on Predictive Association Rules (Yin &
Han, SDM 2003): FOIL-style rule growing on weighted examples (their PRM,
Predictive Rule Mining) that follows every condition nearly as good as
the best one, so that one search can yield several rules; covered
positives are down-weighted instead of removed; and a class is predicted
by the mean expected accuracy of its best `k` rules covering the example.

It reuses the weighted covering components of `pyrulearn.learners.seco`
(`MultiplicativeReweighting` for the weight decay, `PositiveWeightBelow`
as the stop) and `pyrulearn.combiners.TopKMeanCombiner` for prediction.

`PropagatingCPAR`, below, is the same algorithm grown through the SeCo
searches' constraint-aware machinery instead of a dense `data.X` matrix
-- see its own docstring and `CPAR`'s "Data representation" note.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..combiners import TopKMeanCombiner
from ..data import BooleanDataRepresentation
from ..heuristics import FoilGain, GeneralizedMEstimate, RuleStats
from ..models import ConceptModel, ConceptSet, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import DEFAULT_MAX_AUTO_CONVERT_CELLS, NativeRuleLearner, produces
from .seco import (
    CoveringState, MultiplicativeReweighting, PositiveWeightBelow,
    count_open_children, handle_for, materialize_child, parent_closure, rank_best_first,
    stats_from_handle,
)


class CPAR(NativeRuleLearner):
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
    no rule covers get the majority class.

    Defaults are the paper's: ``decay = 2/3``, ``min_total_weight =
    0.05``, ``min_gain = 0.7``, ``gain_similarity = 0.99``, ``k = 5``.
    `gain_similarity = 1` (only exact ties copied) is close to PRM without
    copying. Numeric attributes come already binarized
    (`pyrulearn.data.io.build_dataspec`); the data's row weights multiply
    with the covering weights. `max_length` (default none) caps rule
    length, `max_rounds` the number of search rounds per class.

    **Data representation.** `NATIVE_REPRESENTATIONS = (BooleanDataRepresentation,)`
    -- `_fit_native` converts anything else via
    `NativeRuleLearner.ensure_representation`, same as `ENDER`. `_grow`
    never removes an *implied* feature from consideration, only the one
    literally just added to the rule (no constraint propagation), so
    its open set stays close to full width the whole search -- exactly
    the regime where `(wp * cov) @ Xf` (score every feature in one
    matmul, mask already-used ones out of the *result*) wins: measured
    15x faster than slicing to just the open features first, on
    `spambase` (`ROADMAP.md`'s "Design decisions" has the numbers and
    why this is the *opposite* regime from `BeamSearch`/`HillClimbing`,
    where the same trick regresses). `PropagatingCPAR`, below, is the
    same algorithm with constraint propagation: same models, 4-14x
    slower.
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
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        if not 0.0 < gain_similarity <= 1.0:
            raise ValueError(f"gain_similarity must be in (0, 1], got {gain_similarity}")
        self.decay = decay
        self.min_total_weight = min_total_weight
        self.min_gain = min_gain
        self.gain_similarity = gain_similarity
        self.k = k
        self.max_length = max_length
        self.max_rounds = max_rounds
        self.max_auto_convert_cells = max_auto_convert_cells

    def _default_model(self, data: Any) -> type:
        return ConceptSet

    @produces(ConceptSet)
    def _fit_native(self, data: Any, **kw) -> ConceptSet:
        data = self.ensure_representation(data, self.max_auto_convert_cells, purpose="CPAR's dense-matrix scoring")
        if data.y is None:
            raise ValueError("CPAR needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("CPAR needs at least two classes")
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        concepts = []
        for c in classes:
            bodies = self._rules_for(X, Xf, y == c, base_w)
            rules = [Rule(list(b), target=c, dataspec=data.spec) for b in bodies]
            concepts.append(ConceptModel(annotate_rules(rules, data), label=c))
        n_classes = len(classes)
        combiner = TopKMeanCombiner(GeneralizedMEstimate(m=n_classes, cost=1.0 / n_classes), k=self.k)
        model = ConceptSet(concepts, default_prediction=MajorityClass(data), combiner=combiner)
        return annotate_default_rule(model, data)

    # -- one class -----------------------------------------------------------

    def _rules_for(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
                   base_w: np.ndarray) -> List[Tuple[int, ...]]:
        state = CoveringState(base_w.copy(), positive)
        reweighting = MultiplicativeReweighting(self.decay)
        stop = PositiveWeightBelow(self.min_total_weight)
        found: Dict[Tuple[int, ...], None] = {}
        for _ in range(self.max_rounds):
            if stop.done(state):
                break
            bodies = self._grow(X, Xf, positive, state.scope)
            if not bodies:
                break
            for body in bodies:
                found.setdefault(body, None)
                covered = np.all(X[:, list(body)], axis=1)
                state.record(covered)
                state.scope = base_w * reweighting.weights(state, covered)
        return list(found)

    def _grow(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
              w: np.ndarray) -> List[Tuple[int, ...]]:
        """The rules one search yields on the weights `w`: the best-gain
        lineage and every copy branched off at a nearly-as-good condition."""
        wp, wn = w * positive, w * ~positive
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
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = np.where(p_star > 0,
                                p_star * (np.log2(p_star / (p_star + n_star)) - math.log2(p / (p + n))),
                                -np.inf)
            if body:
                gain[body] = -np.inf
            best = float(gain.max())
            if best < self.min_gain:
                if body:
                    results.append(tuple(body))
                return
            order = np.argsort(-gain, kind="stable")
            chosen = [int(f) for f in order if gain[f] >= max(best * self.gain_similarity, self.min_gain)]
            for f in chosen[1:]:                         # the copies
                grow(body + [f], cov & X[:, f])
            grow(body + [chosen[0]], cov & X[:, chosen[0]])

        grow([], np.ones(len(w), dtype=bool))
        return [tuple(sorted(b)) for b in results]


class PropagatingCPAR(NativeRuleLearner):
    """`CPAR` (Yin & Han 2003), grown through the SeCo searches'
    constraint-aware machinery instead of a dense `data.X` matrix --
    everywhere `CPAR`-specific (the weighted FOIL-gain growing
    criterion, branching into several rules at near-ties, the outer
    weighted-covering loop) is unchanged; the growth step scores from
    `seco.count_open_children`, builds only the children it follows via
    `seco.materialize_child`, and threads the constraint-closed mask
    (`dataspec.extend_closure`: fixing one numeric threshold removes the
    attribute's other thresholds too) instead of ``(wp * cov) @ Xf``
    over every feature not yet in the rule.

    Exists to test `CPAR`'s own design choice (see its "Data
    representation" note and `ROADMAP.md`) rather than assume it. Same
    parameters, same model, same prediction as `CPAR`: an implied
    condition never changes coverage, so its FOIL gain is 0, below
    `min_gain`, and `CPAR` never picks one either -- propagation only
    saves `CPAR` from scoring it. Measured identical models on
    `diabetes`, `sonar` and `kr-vs-kp`; 4-14x slower than `CPAR`, for
    the reasons `ROADMAP.md`'s "Build only the children a search
    follows" lists as still open.
    """

    def __init__(
        self,
        decay: float = 2.0 / 3.0,
        min_total_weight: float = 0.05,
        min_gain: float = 0.7,
        gain_similarity: float = 0.99,
        k: int = 5,
        max_length: Optional[int] = None,
        max_rounds: int = 1000,
    ):
        if not 0.0 < gain_similarity <= 1.0:
            raise ValueError(f"gain_similarity must be in (0, 1], got {gain_similarity}")
        self.decay = decay
        self.min_total_weight = min_total_weight
        self.min_gain = min_gain
        self.gain_similarity = gain_similarity
        self.k = k
        self.max_length = max_length
        self.max_rounds = max_rounds

    def _default_model(self, data: Any) -> type:
        return ConceptSet

    @produces(ConceptSet)
    def _fit_native(self, data: Any, **kw) -> ConceptSet:
        if data.y is None:
            raise ValueError("PropagatingCPAR needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("PropagatingCPAR needs at least two classes")
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        concepts = []
        for c in classes:
            rules = self._rules_for(data, c, base_w)
            concepts.append(ConceptModel(annotate_rules(rules, data), label=c))
        n_classes = len(classes)
        combiner = TopKMeanCombiner(GeneralizedMEstimate(m=n_classes, cost=1.0 / n_classes), k=self.k)
        model = ConceptSet(concepts, default_prediction=MajorityClass(data), combiner=combiner)
        return annotate_default_rule(model, data)

    # -- one class -----------------------------------------------------------

    def _rules_for(self, data: Any, target_class: Any, base_w: np.ndarray) -> List[Rule]:
        positive = np.asarray(data.y) == target_class
        state = CoveringState(base_w.copy(), positive)
        reweighting = MultiplicativeReweighting(self.decay)
        stop = PositiveWeightBelow(self.min_total_weight)
        found: Dict[Rule, None] = {}
        for _ in range(self.max_rounds):
            if stop.done(state):
                break
            rules = self._grow(data, target_class, state.scope)
            if not rules:
                break
            for rule in rules:
                found.setdefault(rule, None)
                covered = data.coverage(rule)
                state.record(covered)
                state.scope = base_w * reweighting.weights(state, covered)
        return list(found)

    def _grow(self, data: Any, target_class: Any, w: np.ndarray) -> List[Rule]:
        """The rules one search yields on the weights `w`: the best-gain
        lineage and every copy branched off at a nearly-as-good
        condition -- same as `CPAR._grow`, scored from
        `count_open_children` instead of ``(wp * cov) @ Xf``, with only
        the children actually followed built (and their closure
        propagated) via `materialize_child`."""
        weighted = data.with_weights(w)
        dataspec = weighted.spec
        heuristic = FoilGain()
        results: List[Rule] = []
        seen: Set[Rule] = set()

        def grow(rule: Rule, mask, handle, stats: RuleStats) -> None:
            if rule in seen:
                return
            seen.add(rule)
            if self.max_length is not None and rule.length() >= self.max_length:
                results.append(rule)
                return
            if stats.tp <= 0:
                return
            features, tps, fps, fns, tns = count_open_children(weighted, target_class, mask, handle, stats)
            features = np.asarray(features, dtype=np.int64)
            tps, fps, fns, tns = (np.asarray(a) for a in (tps, fps, fns, tns))
            alive = tps != 0
            dead: Set[int] = set(features[~alive].tolist())
            features, tps, fps, fns, tns = features[alive], tps[alive], fps[alive], fns[alive], tns[alive]
            length = rule.length() + 1
            gains = heuristic.batch_score(RuleStats(tp=tps, fp=fps, fn=fns, tn=tns, length=length), stats)
            closure = parent_closure(dataspec, rule)
            chosen = []
            threshold = None
            for i in rank_best_first(gains).tolist():                 # stable: ties stay in feature order
                gain = float(gains[i])
                if gain < (self.min_gain if threshold is None else threshold):
                    break
                built = materialize_child(weighted, dataspec, rule, closure, mask, int(features[i]), handle)
                if built is None:
                    continue  # contradicts the rule: never a candidate
                if threshold is None:
                    threshold = max(gain * self.gain_similarity, self.min_gain)
                chosen.append((built, RuleStats(tp=tps[i], fp=fps[i], fn=fns[i], tn=tns[i], length=length)))
            if not chosen:
                if rule.length() > 0:
                    results.append(rule)
                return
            for (crule, cmask, chandle), cstats in chosen[1:]:          # the copies
                grow(crule, cmask - dead, chandle, cstats)
            (crule, cmask, chandle), cstats = chosen[0]
            grow(crule, cmask - dead, chandle, cstats)

        seed = Rule([], target=target_class, dataspec=dataspec)
        handle0 = handle_for(weighted, seed, None)
        stats0 = stats_from_handle(weighted, target_class, seed, handle0)
        grow(seed, frozenset(range(seed.n_features)), handle0, stats0)
        return results
