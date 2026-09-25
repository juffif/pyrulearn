"""
pyrulearn.learners.rulefit
==========================

`RuleFit` -- a RuleFit-style distiller (after Friedman & Popescu, 2008):
it takes a pool of candidate rules and fits an L1-regularized
(sparse) logistic regression over the rules' 0/1 coverage, keeping the
rules with a non-zero coefficient. The result is a `pyrulearn.models.
LinearRuleModel`: each kept rule carries its coefficient as its weight,
an empty-body rule per class carries the intercept, and a prediction is
the class with the highest summed weight -- exactly the regression's own
decision.

Like `CBA`/`CMAR`/`IDS` it is a `RuleDistiller`: the pool is whatever
`rules=` gives (e.g. rules extracted from a random forest,
`pyrulearn.interfaces.sklearn.from_random_forest`), else one mined by
`CARMiner`. Unlike the original RuleFit it doesn't grow its own
candidates from a tree ensemble -- candidate generation and the sparse
fit are independent steps, and this is only the latter. A rule's head in
the pool is ignored: only its body is a candidate feature (duplicate
bodies count once), and the head of a kept rule is the class its
coefficient belongs to.

The fit is `sklearn.linear_model.LogisticRegression` with the `saga`
solver: multinomial for more than two classes (one coefficient per
class and rule, the classes' scores directly comparable), a single
coefficient vector for the positive class (`classes_[1]`) with two --
the other class then scores 0, so a negative weight counts against the
positive class. `l1_ratio=1.0` (the default) is the pure L1 penalty
(lasso), which drives most coefficients to exactly zero; `0 < l1_ratio <
1` is the elastic net, which keeps groups of near-duplicate rules
together instead of picking one. `C` is the inverse regularization
strength (smaller: fewer rules); `cv=` chooses it among `Cs` by
cross-validation (`LogisticRegressionCV`, scored by `scoring`, log loss
by default). `include_features=True` adds
every single feature as a length-1 candidate -- RuleFit's "linear
terms", which for Boolean features are just the features.

After `fit`, `estimator_` is the fitted scikit-learn estimator and
`candidates_` the candidate bodies (tuples of feature indices), in its
column order.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import FlatRuleSet, LinearRuleModel, annotate_rules
from ..rule import Rule, WeightedRule
from .associative import DEFAULT_MAX_AUTO_CONVERT_CELLS, RuleDistiller
from .base import NativeRuleLearner, produces


class RuleFit(RuleDistiller, NativeRuleLearner):
    """RuleFit-style distiller: a sparse (L1) logistic regression over a
    rule pool's coverage -- see the module docstring.

    Pool options (`rules`, `min_support`, `min_confidence`, `max_len`,
    `max_auto_convert_cells`) are `RuleDistiller`'s; `min_confidence`
    defaults to 0 here, since the regression, not a confidence cut,
    decides which bodies matter. Fit options: `C`, `cv`/`Cs`/`scoring`,
    `l1_ratio`, `include_features`, `max_iter`, `random_state`.
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        min_support: float = 0.01,
        min_confidence: float = 0.0,
        max_len: int = 4,
        *,
        C: float = 1.0,
        cv: Optional[int] = None,
        Cs: Any = 10,
        scoring: Any = "neg_log_loss",
        l1_ratio: float = 1.0,
        include_features: bool = False,
        max_iter: int = 5000,
        random_state: Optional[int] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(rules, min_support, min_confidence, max_len, max_auto_convert_cells)
        if not 0.0 < l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in (0, 1] -- 0 would be ridge, not sparse; got {l1_ratio}")
        self.C = C
        self.cv = cv
        self.Cs = Cs
        self.scoring = scoring
        self.l1_ratio = l1_ratio
        self.include_features = include_features
        self.max_iter = max_iter
        self.random_state = random_state

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    def _candidate_bodies(self, data: Any) -> List[Tuple[int, ...]]:
        """The pool's distinct non-empty bodies (feature-index tuples), in
        first-seen order, plus single features if `include_features`."""
        seen: Dict[Tuple[int, ...], None] = {}
        for r in self._resolve_rules(data):
            body = tuple(sorted(lit.feature for lit in r.conditions))
            if body:
                seen.setdefault(body, None)
        if self.include_features:
            for f in range(data.spec.n_features):
                seen.setdefault((f,), None)
        return list(seen)

    def _estimator(self):
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        if self.cv is not None:
            extra = {}
            if "use_legacy_attributes" in inspect.signature(LogisticRegressionCV).parameters:
                extra["use_legacy_attributes"] = False   # scikit-learn >= 1.8's simplified attributes
            return LogisticRegressionCV(Cs=self.Cs, cv=self.cv, l1_ratios=[self.l1_ratio], solver="saga",
                                        scoring=self.scoring, max_iter=self.max_iter,
                                        random_state=self.random_state, **extra)
        return LogisticRegression(C=self.C, l1_ratio=self.l1_ratio, solver="saga",
                                  max_iter=self.max_iter, random_state=self.random_state)

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        if data.y is None:
            raise ValueError("RuleFit.fit needs data.y")
        spec = data.spec
        bodies = self._candidate_bodies(data)
        if not bodies:
            raise ValueError("RuleFit: the rule pool has no non-empty candidate rules")
        probes = [Rule(list(b), target=None, dataspec=spec) for b in bodies]
        X = FlatRuleSet(probes).coverage_matrix(data).T.astype(float)   # (n_samples, n_candidates)
        estimator = self._estimator().fit(X, np.asarray(data.y))
        self.estimator_, self.candidates_ = estimator, bodies

        classes = [c.item() if isinstance(c, np.generic) else c for c in estimator.classes_]
        if len(classes) == 2:   # one coefficient vector, for the positive class
            per_class = [(classes[1], estimator.coef_[0], estimator.intercept_[0])]
        else:
            per_class = list(zip(classes, estimator.coef_, estimator.intercept_))
        rules: List[WeightedRule] = []
        for cls, coefs, intercept in per_class:
            if intercept != 0:
                rules.append(WeightedRule([], target=cls, dataspec=spec, weight=float(intercept)))
            for j in sorted(np.flatnonzero(coefs), key=lambda j: -abs(coefs[j])):
                rules.append(WeightedRule(list(bodies[j]), target=cls, dataspec=spec, weight=float(coefs[j])))
        return LinearRuleModel(annotate_rules(rules, data), classes=classes)
