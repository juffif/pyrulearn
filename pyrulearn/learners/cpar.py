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
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..combiners import TopKMeanCombiner
from ..heuristics import GeneralizedMEstimate
from ..models import ConceptModel, ConceptSet, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import NativeRuleLearner, produces
from .seco import CoveringState, MultiplicativeReweighting, PositiveWeightBelow


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
