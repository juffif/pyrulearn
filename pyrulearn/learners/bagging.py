"""
pyrulearn.learners.bagging
==========================

`Bagging` -- bootstrap aggregating (Breiman, *Bagging Predictors*,
Machine Learning 1996) over any rule learner: each member is learned
from a bootstrap sample of the training data, and the members vote.

A bootstrap sample draws ``n`` of the ``n`` training rows with
replacement, so a row may be drawn several times. Rather than copying
it, the sample keeps each drawn row once, with the number of times it
was drawn as its row weight (`pyrulearn.data.DataRepresentation.weights`,
multiplied with the data's own weights) -- every count a learner makes
is a sum of row weights, so this is the same as learning from the copies.
The one difference: a learner that splits its rows (`Pypper`'s grow and
prune sets) keeps a row's copies together, in one of the two sets.

The model is an `pyrulearn.models.EnsembleModel` that resolves its
members' votes with a `pyrulearn.combiners.RuleCombiner` (default
`CountVoteCombiner`, one vote per member, as in Breiman's bagging): each
member votes with the rule that decides its prediction, members whose
rules don't cover an example abstain, and examples no member decides go
to the majority class. `pyrulearn.models.ensemblemodel_to_flatruleset`
(or `fit(data, model=FlatRuleSet)`) resolves the same members at the
level of their rules instead.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..combiners import CountVoteCombiner, RuleCombiner
from ..data import DataRepresentation
from ..models import EnsembleModel, MajorityClass, RuleModel, annotate_default_rule
from .base import NativeRuleLearner, produces


class Bagging(NativeRuleLearner):
    """`n_estimators` members, each `base_learner.fit` on its own
    bootstrap sample (drawn from `random_state`), as whatever model the
    base learner builds -- configure it for the members wanted, e.g.
    `OneVsRest(Pypper())` for unordered concept sets instead of
    `Pypper`'s cascades -- combined into an `EnsembleModel` with
    `combiner` (default `CountVoteCombiner()`) and the training data's
    majority class as its default. The members' own defaults are ignored
    unless `member_defaults` (see `EnsembleModel`)."""

    def __init__(
        self,
        base_learner: NativeRuleLearner,
        n_estimators: int = 10,
        combiner: Optional[RuleCombiner] = None,
        member_defaults: bool = False,
        random_state: Optional[int] = None,
    ):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {n_estimators}")
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.combiner = combiner
        self.member_defaults = member_defaults
        self.random_state = random_state

    def _default_model(self, data: DataRepresentation) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: DataRepresentation, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("Bagging needs data.y")
        rng = np.random.default_rng(self.random_state)
        members: List[RuleModel] = [self.base_learner.fit(sample)
                                    for sample in bootstrap_samples(data, self.n_estimators, rng)]
        model = EnsembleModel(members, default_prediction=MajorityClass(data),
                              combiner=self.combiner if self.combiner is not None else CountVoteCombiner(),
                              member_defaults=self.member_defaults)
        return annotate_default_rule(model, data)


def bootstrap_samples(data: DataRepresentation, n_samples: int, rng: np.random.Generator):
    """`n_samples` bootstrap samples of `data`, one at a time: the rows
    drawn at least once, each weighted by how often it was drawn (times
    its own weight, if `data` has any)."""
    n = data.n_samples
    base = np.ones(n) if data.weights is None else data.weights
    for _ in range(n_samples):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        drawn = counts > 0
        yield data.select_rows(drawn).with_weights((counts * base)[drawn])
