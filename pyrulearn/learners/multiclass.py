"""
pyrulearn.learners.multiclass
================================

Thin **sugar** over the `model=` switcher on a per-class-targetable
learner (`pyrulearn.learners.DecomposingLearner` -- the SeCo family,
`PyLORD`). Each class here just calls `base_learner.fit(data, model=...)`
with a fixed model type; the decomposition logic itself lives in
`DecomposingLearner`.

- `OneVsRest(base)`        == `base.fit(data, model=ConceptSet)`
- `OrderedOneVsRest(base)` == `base.fit(data, model=ConceptCascade, order=...)`
- `Pairwise(base)`         == `base.fit(data, model=PairwiseModel, positive=...)`
                             then sets the requested vote `combiner`

Prefer calling `base.fit(data, model=ConceptSet | ConceptCascade |
PairwiseModel)` directly; these wrappers exist for the older call style
and to bundle a `random_state` / non-default `combiner`.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional, Sequence, Union

from ..data import DataRepresentation
from .base import NativeRuleLearner
from ..models import (
    ConceptCascade, ConceptSet, PairwiseCombiner, PairwiseModel,
    _resolve_pairwise_combiner,
)


def _seeded(base_learner: NativeRuleLearner, random_state: Optional[int]) -> NativeRuleLearner:
    """`base_learner` (or a shallow copy of it with `random_state` set,
    if given and different) -- so the wrapper's `random_state=` reaches
    the decomposition's own RNG without mutating the caller's learner."""
    if random_state is None or getattr(base_learner, "random_state", None) == random_state:
        return base_learner
    lc = copy.copy(base_learner)
    lc.random_state = random_state
    return lc


class OneVsRest(NativeRuleLearner):
    """`OneVsRest(base).fit(data)` == `base.fit(data, model=ConceptSet)` --
    one `ConceptModel` per label, competing via a rule combiner."""

    def __init__(self, base_learner: NativeRuleLearner):
        self.base_learner = base_learner

    def fit(self, data: DataRepresentation) -> ConceptSet:
        return self.base_learner.fit(data, model=ConceptSet)


class OrderedOneVsRest(NativeRuleLearner):
    """`OrderedOneVsRest(base).fit(data)` ==
    `base.fit(data, model=ConceptCascade, order=order)` -- ordered
    "peeling" one-vs-rest; the last label in `order` is the catch-all."""

    def __init__(
        self,
        base_learner: NativeRuleLearner,
        order: Union[str, Sequence[Any]] = "least_frequent",
        random_state: Optional[int] = None,
    ):
        self.base_learner = base_learner
        self.order = order
        self.random_state = random_state

    def fit(self, data: DataRepresentation) -> ConceptCascade:
        return _seeded(self.base_learner, self.random_state).fit(
            data, model=ConceptCascade, order=self.order)


class Pairwise(NativeRuleLearner):
    """`Pairwise(base).fit(data)` ==
    `base.fit(data, model=PairwiseModel, positive=positive)`, then the
    result's `combiner` set to `combiner` (`"vote"` / `"weighted_vote"` /
    `"accuracy_vote"` / a `PairwiseCombiner`).

    `positive` picks each pair's target: ``"smaller"`` (default) /
    ``"larger"`` / ``"random"`` / ``"both"`` (double round robin) / a
    callable ``f(a, b) -> label``.
    """

    def __init__(
        self,
        base_learner: NativeRuleLearner,
        positive: Union[str, Callable[[Any, Any], Any]] = "smaller",
        combiner: Union[str, PairwiseCombiner] = "vote",
        random_state: Optional[int] = None,
    ):
        self.base_learner = base_learner
        self.positive = positive
        self.combiner = combiner
        self.random_state = random_state

    def fit(self, data: DataRepresentation) -> PairwiseModel:
        model = _seeded(self.base_learner, self.random_state).fit(
            data, model=PairwiseModel, positive=self.positive)
        model.combiner = _resolve_pairwise_combiner(self.combiner)
        return model
