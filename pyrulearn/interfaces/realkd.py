"""
pyrulearn.interfaces.realkd
===========================

`RealkdImporter` for the rule ensembles of Mario Boley's `realkd`
package (`realkd.rules.RuleBoostingEstimator`, an
`AdditiveRuleEnsemble`), and `RealkdRuleBoosting`, its learner -- the
reference implementation of optimal rule boosting (Boley, Teshuva, Le
Bodic & Webb, SDM 2021), which the native
`pyrulearn.learners.boosting.OptimalRuleBoosting` re-implements.

`realkd` fits binary targets coded ``+1``/``-1`` and describes rows by
propositions it generates from a pandas DataFrame: on a 0/1 column ``c``
those are ``c<=0`` and ``c>=1``. `RealkdRuleBoosting` therefore hands it
only the positive features of the data (columns named ``c0``, ``c1``,
... after their position), and the importer maps ``c>=1`` back to the
feature and ``c<=0`` to its paired negation feature
(`DataSpec.negation_of`) -- the rules bind to the data's own `DataSpec`,
which must have negation features. A rule ``w if q`` becomes a
`WeightedRule` for the positive class (the second label) if ``w > 0``,
for the other class with weight ``-w`` otherwise; the empty query is the
intercept; repeated queries are merged. The result is a
`pyrulearn.models.LinearRuleModel` that decides like the ensemble (which
predicts the positive class where its score is ``>= 0``; the model where
it is ``> 0`` -- only an exactly-zero score can differ).

Installing `realkd` 0.2.1 on Python 3.14 takes some care: it pins
`bitarray==1.5.3`, which doesn't build there (a current `bitarray` works:
``pip install realkd --no-deps``, then ``pip install bitarray
sortedcontainers``), and `sortednp` has no Windows wheel (it builds with
MinGW). This module requires `realkd` only when used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import ObjectRuleImporter, register_importer
from ..data import DataRepresentation, DataSpec
from ..learners import ExternalRuleLearner
from ..models import LinearRuleModel
from ..rule import WeightedRule


def _positive_columns(spec: DataSpec) -> List[int]:
    """The data's positive features -- those that aren't another feature's
    negation (for a pair, the one with the lower index)."""
    keep = []
    for j in range(spec.n_features):
        partner = spec.negation_of(j)
        if partner is None or j < partner:
            keep.append(j)
    return keep


def _query_body(query, columns: Sequence[int], spec: DataSpec) -> Optional[List[int]]:
    """A body (feature indices of `spec`) for a realkd `Conjunction` over
    the columns ``c<k>`` = feature ``columns[k]``, or `None` if no 0/1 row
    satisfies it."""
    allowed: Dict[int, set] = {}
    for prop in query.props:
        k = int(str(prop.key)[1:])
        values = {v for v in (0, 1) if prop.constraint(v)}
        allowed[k] = allowed.get(k, {0, 1}) & values
    body = []
    for k, values in allowed.items():
        f = columns[k]
        if not values:
            return None
        if values == {1}:
            body.append(f)
        elif values == {0}:
            neg = spec.negation_of(f)
            if neg is None:
                raise ValueError(f"feature {spec.feature_names[f]!r} has no negation feature")
            body.append(neg)
    return body


class RealkdImporter(ObjectRuleImporter):
    """Extracts a `LinearRuleModel` from a fitted
    `realkd.rules.RuleBoostingEstimator` (or its `rules_`, an
    `AdditiveRuleEnsemble`) fitted on the positive features of `dataspec`
    as columns ``c0``, ``c1``, ... (see the module docstring). `labels`
    are the two classes, negative (``-1``) first."""

    SOURCE = "realkd.RuleBoostingEstimator"

    def __init__(self, labels: Sequence[Any] = (-1, 1)):
        self.labels = list(labels)

    def import_model(self, model, dataspec: DataSpec,
                     data: Optional[DataRepresentation] = None) -> LinearRuleModel:
        ensemble = getattr(model, "rules_", model)
        columns = _positive_columns(dataspec)
        neg_class, pos_class = self.labels
        total: Dict[Tuple[int, ...], float] = {}
        for rule in ensemble.members:
            if float(rule.z) != 0.0:
                raise ValueError("rules with an else-value (z != 0) aren't supported")
            body = _query_body(rule.q, columns, dataspec)
            if body is None:
                continue
            key = tuple(sorted(set(body)))
            total[key] = total.get(key, 0.0) + float(rule.y)
        rules = [WeightedRule(list(body), target=pos_class if w > 0 else neg_class,
                              dataspec=dataspec, weight=abs(w))
                 for body, w in total.items() if w != 0]
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)
        classes = [c.item() if isinstance(c, np.generic) else c for c in self.labels]
        return self._stamp_provenance(LinearRuleModel(rules, classes=classes), n_rules=len(rules))


register_importer("realkd", RealkdImporter)


class RealkdRuleBoosting(ExternalRuleLearner):
    """`realkd`'s rule boosting: `realkd.rules.RuleBoostingEstimator` with
    `n_rules` rules from `XGBRuleEstimator(loss, reg, search)` base
    learners (`search="exhaustive"`: optimal rule boosting; ``"greedy"``),
    optionally preceded by an intercept (`offset=True`). Binary targets
    only. `fit(data)` -> `LinearRuleModel` (see `RealkdImporter`). The
    native re-implementation is
    `pyrulearn.learners.boosting.OptimalRuleBoosting`."""

    IMPORTER = RealkdImporter
    NATIVE_MODEL = LinearRuleModel

    def __init__(self, n_rules: int = 10, loss: str = "logistic", reg: float = 1.0,
                 search: str = "exhaustive", offset: bool = False):
        self.n_rules = n_rules
        self.loss = loss
        self.reg = reg
        self.search = search
        self.offset = offset

    def _import(self, data: DataRepresentation) -> Any:
        labels = np.unique(np.asarray(data.y))
        if len(labels) != 2:
            raise ValueError(f"RealkdRuleBoosting needs a binary target, got {len(labels)} classes")
        fitted = self.fit_external(self.prepare(data), data.y, feature_names=data.spec.feature_names)
        return RealkdImporter(labels=list(labels)).import_model(fitted, data.spec, data=data)

    def prepare(self, data: DataRepresentation) -> Any:
        import pandas as pd

        columns = _positive_columns(data.spec)
        X = np.asarray(data.X, dtype=int)[:, columns]
        return pd.DataFrame(X, columns=[f"c{k}" for k in range(len(columns))])

    def fit_external(self, X, y, feature_names=None):
        import pandas as pd
        from realkd.logic import Conjunction
        from realkd.rules import RuleBoostingEstimator, XGBRuleEstimator

        labels = np.unique(np.asarray(y))
        target = pd.Series(np.where(np.asarray(y) == labels[1], 1, -1))
        base = XGBRuleEstimator(loss=self.loss, reg=self.reg, search=self.search)
        learners = [XGBRuleEstimator(loss=self.loss, reg=self.reg, query=Conjunction([])), base] \
            if self.offset else base
        return RuleBoostingEstimator(num_rules=self.n_rules, base_learner=learners).fit(X, target)
