"""
pyrulearn.interfaces.boomer
===========================

`BoomerImporter` for the rule models of BOOMER (Rapp, Loza Mencía,
Fürnkranz, Nguyen & Hüllermeier, ECML PKDD 2020) -- gradient boosted
multi-output rules, an extension of ENDER -- as implemented by the
`mlrl-boomer` package (`mlrl.boosting.BoomerClassifier`), and
`MLRLBoomer`, its learner.

BOOMER learns an additive model of rules for several outputs (labels) at
once; pyrulearn uses it for binary classification, i.e. a single output:
each rule adds its head's score for that output where its body holds,
and the positive class (the second label) is predicted where the scores
sum to more than zero. The model's rules are read with BOOMER's own
visitor (`RuleModelVisitor`). A body is a conjunction of conditions on
the features `mlrl-boomer` was fit on -- the data's Boolean feature
columns, as 0/1 -- of the form ``x <= t`` / ``x > t`` (or ``==`` / ``!=``
for nominal features); on a 0/1 column ``x > t`` (``0 <= t < 1``) is the
feature and ``x <= t`` its negation feature (`DataSpec.negation_of`),
so the rules bind to the data's own `DataSpec`. A rule with a positive
score becomes a `WeightedRule` for the positive class, one with a
negative score one for the other class with the absolute score (the same
decision); BOOMER's default rule is the intercept, repeated bodies are
merged. The result is a `pyrulearn.models.LinearRuleModel` whose scores
equal `BoomerClassifier.decision_function` (up to BOOMER's 32-bit
floats).

The single-output case of BOOMER (logistic loss, L2-regularized Newton
steps) is ENDER's Newton method with L2 regularization -- natively
`pyrulearn.learners.boosting.Boomer`, which predicts like `MLRLBoomer` on
binary data. BOOMER's multi-label learning (several outputs at once) is
not supported: pyrulearn has no multi-label data or models yet (on the
to-do list with preference learning and label ranking, see the README).
This module requires `mlrl-boomer` only when used (it pins
scikit-learn to its supported range).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import ObjectRuleImporter, register_importer
from ..data import DataRepresentation, DataSpec
from ..learners import ExternalRuleLearner
from ..models import LinearRuleModel
from ..rule import WeightedRule

_CONDITION_KINDS = {
    "numerical_leq": lambda v, t: v <= t, "numerical_gr": lambda v, t: v > t,
    "ordinal_leq": lambda v, t: v <= t, "ordinal_gr": lambda v, t: v > t,
    "nominal_eq": lambda v, t: v == t, "nominal_neq": lambda v, t: v != t,
}


def _boomer_rules(model) -> List[Tuple[List[Tuple[str, int, float]], float]]:
    """``[(conditions, score)]`` for every rule of a fitted `BoomerClassifier`
    (first output only), read with BOOMER's visitor."""
    from mlrl.common.cython.rule_model import RuleModelVisitor

    rules: List[list] = []

    class _Visitor(RuleModelVisitor):
        def visit_empty_body(self, body):
            rules.append([[], 0.0])

        def visit_conjunctive_body(self, body):
            conditions = []
            for kind in _CONDITION_KINDS:
                idx = getattr(body, kind + "_indices")
                thr = getattr(body, kind + "_thresholds")
                if idx is not None:
                    conditions += [(kind, int(i), float(t)) for i, t in zip(idx, thr)]
            rules.append([conditions, 0.0])

        def visit_complete_head(self, head):
            rules[-1][1] = float(head.scores[0])

        def visit_partial_head(self, head):
            indices = list(head.indices)
            rules[-1][1] = float(head.scores[indices.index(0)]) if 0 in indices else 0.0

    model.model_.visit(_Visitor())
    return [(c, s) for c, s in rules]


def _body(conditions, dataspec: DataSpec) -> Optional[List[int]]:
    allowed: Dict[int, set] = {}
    for kind, col, t in conditions:
        values = {v for v in (0, 1) if _CONDITION_KINDS[kind](v, t)}
        allowed[col] = allowed.get(col, {0, 1}) & values
    body = []
    for col, values in allowed.items():
        if not values:
            return None
        if values == {1}:
            body.append(col)
        elif values == {0}:
            neg = dataspec.negation_of(col)
            if neg is None:
                raise ValueError(f"feature {dataspec.feature_names[col]!r} has no negation feature")
            body.append(neg)
    return body


class BoomerImporter(ObjectRuleImporter):
    """Extracts a `LinearRuleModel` from a fitted `mlrl.boosting.
    BoomerClassifier` with a single output, fit on `dataspec`'s features
    as 0/1 columns (see the module docstring). `labels` are the two
    classes, the one coded 0 first; `model.label_names_` (set by
    `MLRLBoomer`) takes precedence."""

    SOURCE = "mlrl.boosting.BoomerClassifier"

    def __init__(self, labels: Sequence[Any] = (0, 1)):
        self.labels = list(labels)

    def import_model(self, model, dataspec: DataSpec,
                     data: Optional[DataRepresentation] = None) -> LinearRuleModel:
        if getattr(model, "num_outputs_", 1) != 1:
            raise ValueError("BoomerImporter supports a single output (binary classification) only")
        if model.n_features_in_ != dataspec.n_features:
            raise ValueError(f"dataspec has {dataspec.n_features} features but the model was trained on "
                             f"{model.n_features_in_}")
        labels = list(getattr(model, "label_names_", self.labels))
        neg_class, pos_class = labels[0], labels[1]
        total: Dict[Tuple[int, ...], float] = {}
        for conditions, score in _boomer_rules(model):
            body = _body(conditions, dataspec)
            if body is None:
                continue
            key = tuple(sorted(set(body)))
            total[key] = total.get(key, 0.0) + score
        rules = [WeightedRule(list(body), target=pos_class if w > 0 else neg_class,
                              dataspec=dataspec, weight=abs(w))
                 for body, w in total.items() if w != 0]
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        return self._stamp_provenance(LinearRuleModel(rules, classes=classes), n_rules=len(rules))


register_importer("boomer", BoomerImporter)


class MLRLBoomer(ExternalRuleLearner):
    """BOOMER (`mlrl.boosting.BoomerClassifier`) for binary classification.
    `fit(data)` -> `LinearRuleModel` (see `BoomerImporter`). `**params` are
    its constructor arguments (`max_rules=`, `shrinkage=`, `loss=`,
    `l2_regularization_weight=`, `random_state=`, ...). Multi-label
    BOOMER isn't supported (see the module docstring); the native
    single-label version is `pyrulearn.learners.boosting.Boomer`."""

    IMPORTER = BoomerImporter
    NATIVE_MODEL = LinearRuleModel

    def __init__(self, **params):
        self.params = params

    def fit_external(self, X, y, feature_names=None):
        from mlrl.boosting import BoomerClassifier

        labels, y01 = np.unique(np.asarray(y), return_inverse=True)
        if len(labels) != 2:
            raise ValueError(f"MLRLBoomer needs a binary target, got {len(labels)} classes")
        model = BoomerClassifier(**self.params)
        model.fit(np.asarray(X, dtype=float), y01)
        model.label_names_ = labels
        return model

# -- naming ------------------------------------------------------------------

MLRLBoomer.TOOL = "MLRL"

from .base import deprecated_aliases as _deprecated_aliases  # noqa: E402

#: the 0.2.0 name, deprecated
__getattr__ = _deprecated_aliases(globals(), {"MlrlBoomer": "MLRLBoomer"})
