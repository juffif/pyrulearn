"""
pyrulearn.learners.adaboost
===========================

`AdaBoost` -- boosting of whole models: SAMME (Zhu, Zou, Rosset & Hastie,
*Multi-class AdaBoost*, Statistics and Its Interface 2009), the
multi-class AdaBoost, over any rule learner. Each member is learned from
the training data reweighted by the errors of its predecessors, and the
members vote with weights.

With weighted training error ``err`` of a member and ``K`` classes, the
member gets the weight ``log((1 - err) / err) + log(K - 1)``, and the
examples it misclassifies are multiplied by ``exp(weight)`` before the
next member is learned. A member has to beat random guessing
(``err < 1 - 1/K``, where AdaBoost.M1 needs ``err < 1/2``); with two
classes, SAMME is AdaBoost. The weights are row weights of the data
(`pyrulearn.data.DataRepresentation.weights`), rescaled to sum to the
number of rows, so that a learner's counts stay on the scale of the
data.

The model is an `pyrulearn.models.EnsembleModel` with `member_weights`:
each member predicts, its own default included, and the members' votes
are summed with their weights. This weights *models*; the rule boosting
learners of `pyrulearn.learners.boosting` (`Slipper`, `ENDER`, `Boomer`)
weight single *rules* instead. A later member never changes an earlier
one, so `EnsembleModel.staged(n)` of a fitted model is the model after
`n` rounds.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..data import DataRepresentation
from ..models import EnsembleModel, MajorityClass, RuleModel, annotate_default_rule
from .base import NativeRuleLearner, produces


class AdaBoost(NativeRuleLearner):
    """Up to `n_estimators` members, each `base_learner.fit` on the
    reweighted training data, as whatever model the base learner builds
    (e.g. `OneVsRest(Pypper())` for concept sets), combined into an
    `EnsembleModel` weighted by SAMME. Boosting stops early at a member
    that is no better than random guessing (it is left out, unless it is
    the first) or one without training errors (it is kept)."""

    def __init__(self, base_learner: NativeRuleLearner, n_estimators: int = 10):
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {n_estimators}")
        self.base_learner = base_learner
        self.n_estimators = n_estimators

    def _default_model(self, data: DataRepresentation) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: DataRepresentation, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("AdaBoost needs data.y")
        y = np.asarray(data.y)
        n, k = data.n_samples, len(np.unique(y))
        w = np.ones(n) if data.weights is None else np.asarray(data.weights, dtype=float)
        w = w * (n / w.sum())
        members: List[RuleModel] = []
        weights: List[float] = []
        for _ in range(self.n_estimators):
            member = self.base_learner.fit(data.with_weights(w))
            wrong = np.asarray(member.predict(data)) != y
            err = float(w[wrong].sum() / w.sum())
            if err >= 1.0 - 1.0 / k - 1e-12:   # no better than random guessing (up to rounding)
                if not members:
                    members.append(member)
                    weights.append(1.0)
                break
            members.append(member)
            weights.append(samme_weight(err, k))
            if err == 0.0:
                break
            w = w * np.exp(weights[-1] * wrong)
            w = w * (n / w.sum())
        model = EnsembleModel(members, default_prediction=MajorityClass(data), member_weights=weights)
        return annotate_default_rule(model, data)


def samme_weight(err: float, n_classes: int, eps: float = 1e-10) -> float:
    """SAMME's member weight ``log((1 - err) / err) + log(K - 1)``, with
    `err` kept at least `eps` (a member without training errors)."""
    err = max(err, eps)
    return float(np.log((1.0 - err) / err) + np.log(n_classes - 1))
