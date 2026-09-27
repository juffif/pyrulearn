"""
pyrulearn.learners.boosting
===========================

Rule learners that boost: each round reweights the training examples and
learns one rule with a fitted weight (confidence), and the model is the
sum of the rules' weights -- a `pyrulearn.models.LinearRuleModel`.

`Slipper` (Cohen & Singer, AAAI 1999) reweights with
`pyrulearn.learners.seco.AdaBoostReweighting`, the same component the
weighted covering framework uses (`pyrulearn.learners.seco.
WeightedCovering`), and grows its rules with the SeCo building blocks
(`HillClimbing`, `GrowPruneSplit`). `ENDER` (Dembczyński, Kotłowski &
Słowiński 2008/2010) is gradient boosting of rules for a pluggable
`BoostingLoss`: `LogisticLoss` (the multinomial log-likelihood -- the
MLRules variant, the default) or `ExponentialLoss` (AdaBoost's).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..heuristics import SlipperZ
from ..models import LinearRuleModel, annotate_rules
from ..rule import Rule, WeightedRule
from .base import NativeRuleLearner, produces
from .seco import AdaBoostReweighting, CoveringState, EmptyRuleAllFeatures, GrowPruneSplit, HillClimbing


class Slipper(NativeRuleLearner):
    """SLIPPER (Cohen & Singer 1999): confidence-rated boosting of rules.

    For one class `c` against the rest, every round

    1. splits the examples into a growing set and a pruning set
       (`prune_fraction`, stratified; each keeps its boosting weights);
    2. grows a rule on the growing set by greedily adding the condition
       that most increases ``sqrt(W+) - sqrt(W-)`` (`SlipperZ`, over the
       boosting weights of the covered positives and negatives), until no
       condition improves it (`HillClimbing`);
    3. prunes it: keeps the prefix minimizing the boosting loss on the
       pruning set, ``(1 - V+ - V-) + V+ exp(-C) + V- exp(C)``, with ``C``
       the prefix's confidence on the growing set and ``V+``/``V-`` the
       covered fractions of the pruning set's weight;
    4. compares it with the default rule (the empty rule, covering
       everything): whichever has the larger ``(sqrt(W+) - sqrt(W-))**2``
       on all examples is taken;
    5. gives the taken rule the confidence ``C = 1/2 ln((W+ + eps) /
       (W- + eps))`` on all examples, and reweights: every example it
       covers is multiplied by ``exp(-y C)`` (``y = +1`` for `c`, ``-1``
       otherwise), then the weights are rescaled
       (`AdaBoostReweighting`).

    After `n_rounds` rounds, the model predicts `c` where the covering
    rules' confidences sum to more than zero. A rule taken in several
    rounds appears once, with its confidences summed; the default rules
    sum into the intercept (the empty-body rule).

    The result is a `LinearRuleModel`. With two classes, one boosting run
    for `target_class` (default: the less frequent class, as RIPPER
    does); its rules all predict it, and the other class scores 0. With
    more classes, one run per class against the rest, their rules and
    intercepts together in one model: the class with the highest sum
    wins.

    Differences from Cohen & Singer: `n_rounds` is fixed (they chose it
    by internal cross-validation), and numeric attributes come already
    binarized (`pyrulearn.data.io.build_dataspec`) instead of being
    thresholded during growing. Row weights of the data are the initial
    boosting weights.
    """

    def __init__(
        self,
        n_rounds: int = 20,
        prune_fraction: float = 1.0 / 3.0,
        target_class: Any = None,
        max_conditions: Optional[int] = None,
        eps: Optional[float] = None,
        random_state: Optional[int] = None,
    ):
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be at least 1, got {n_rounds}")
        self.n_rounds = n_rounds
        self.prune_fraction = prune_fraction
        self.target_class = target_class
        self.max_conditions = max_conditions
        self.eps = eps
        self.random_state = random_state

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        if data.y is None:
            raise ValueError("Slipper needs data.y")
        y = np.asarray(data.y)
        labels, counts = np.unique(y, return_counts=True)
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        if len(classes) < 2:
            raise ValueError("Slipper needs at least two classes")
        if self.target_class is not None:
            targets = [self.target_class]
        elif len(classes) == 2:
            targets = [classes[int(np.argmin(counts))]]       # the less frequent class
        else:
            targets = classes
        rules: List[WeightedRule] = []
        for c in targets:
            rules += self._boost(data, c)
        return LinearRuleModel(annotate_rules(rules, data), classes=classes)

    # -- one boosting run -----------------------------------------------------

    def _boost(self, data: Any, target: Any) -> List[WeightedRule]:
        spec = data.spec
        base = data.with_weights(None)              # the boosting weights carry the data's weights
        positive = np.asarray(data.y) == target
        n = data.n_samples
        w = np.ones(n) if data.weights is None else data.weights.astype(float).copy()
        w *= n / w.sum()                            # mean 1: eps = 1/2 is Slipper's 1/(2n)
        reweighting = AdaBoostReweighting(eps=0.5 if self.eps is None else self.eps)
        state = CoveringState(w, positive)
        search = HillClimbing(max_conditions=self.max_conditions)
        heuristic = SlipperZ()
        everything = np.ones(n, dtype=bool)

        learned: List[Tuple[Tuple[int, ...], float]] = []
        for t in range(self.n_rounds):
            seed = None if self.random_state is None else self.random_state + t
            grow, prune = GrowPruneSplit(self.prune_fraction, seed).prepare(base, target, state.scope)
            initial = EmptyRuleAllFeatures().initial_candidates(base, target, grow)
            rule = search.search(base, target, heuristic, initial, example_mask=grow)
            if rule is not None and rule.length() > 0 and prune is not None:
                rule = self._prune(rule, base, positive, grow, prune, reweighting)

            covered = everything
            if rule is not None and rule.length() > 0:
                rule_cov = rule.covers_data_packed(base)
                if _objective(state.scope, rule_cov, positive) > _objective(state.scope, everything, positive):
                    covered = rule_cov
                else:
                    rule = None
            else:
                rule = None
            confidence = reweighting.rule_weight(state, covered)
            state.record(covered)
            state.scope = reweighting.weights(state, covered)
            body = () if rule is None else tuple(l.feature for l in rule.conditions)
            learned.append((body, confidence))

        return self._merge(learned, target, spec)

    def _prune(self, rule: Rule, base: Any, positive: np.ndarray, grow: np.ndarray,
               prune: np.ndarray, reweighting: AdaBoostReweighting) -> Rule:
        """The prefix of `rule` (at least one condition) with the lowest
        boosting loss on the pruning weights `prune`; ties go to the shorter."""
        total = float(prune.sum())
        if total <= 0:
            return rule
        best, best_loss = rule, math.inf
        for k in range(rule.length(), 0, -1):
            prefix = rule if k == rule.length() else Rule(
                rule.conditions[:k], target=rule.target, dataspec=rule.dataspec, n_features=rule.n_features)
            cov = prefix.covers_data_packed(base)
            c = reweighting.confidence(grow, cov, positive)
            v_pos = float(prune[cov & positive].sum()) / total
            v_neg = float(prune[cov & ~positive].sum()) / total
            loss = (1.0 - v_pos - v_neg) + v_pos * math.exp(-c) + v_neg * math.exp(c)
            if loss <= best_loss:
                best, best_loss = prefix, loss
        return best

    @staticmethod
    def _merge(learned: List[Tuple[Tuple[int, ...], float]], target: Any, spec: Any) -> List[WeightedRule]:
        """One rule per distinct body, its confidences summed (in first-taken
        order); the default rules' sum is the intercept, listed first."""
        total: Dict[Tuple[int, ...], float] = {}
        for body, c in learned:
            total[body] = total.get(body, 0.0) + c
        intercept = total.pop((), None)
        out = [] if intercept is None else [WeightedRule([], target=target, dataspec=spec, weight=intercept)]
        out += [WeightedRule(list(body), target=target, dataspec=spec, weight=c)
                for body, c in total.items() if c != 0]
        return out


def _objective(w: np.ndarray, covered: np.ndarray, positive: np.ndarray) -> float:
    """``(sqrt(W+) - sqrt(W-))**2`` of the rows `covered` under weights `w`
    -- the larger, the lower the boosting loss the rule can reach."""
    return (math.sqrt(float(w[covered & positive].sum())) - math.sqrt(float(w[covered & ~positive].sum()))) ** 2


# ================================================================ ENDER ===

class BoostingLoss(ABC):
    """A loss for `ENDER`. The model keeps one score per class and example,
    ``F`` (``n x K``); a rule votes for one class ``k`` by adding its
    weight ``alpha`` to that class's scores on the rows it covers.
    `derivatives` returns, per example and class, the first and second
    derivative of the loss with respect to such a vote (at ``alpha = 0``),
    already multiplied by the examples' weights ``d``; `value` is the total
    loss. ``Y`` is the one-hot class matrix."""

    #: whether the loss handles more than two classes
    multiclass: bool = True

    @abstractmethod
    def derivatives(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def value(self, F: np.ndarray, Y: np.ndarray, d: np.ndarray) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def _softmax(F: np.ndarray) -> np.ndarray:
    e = np.exp(F - F.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class LogisticLoss(BoostingLoss):
    """The multinomial negative log-likelihood, ``log sum_k exp(F_k) -
    F_y`` (MLRules; for two classes the logit loss). A vote for class
    ``k``: first derivative ``p_k - [y = k]``, second ``p_k (1 - p_k)``,
    with ``p`` the softmax of the scores."""

    def derivatives(self, F, Y, d):
        P = _softmax(F)
        return d[:, None] * (P - Y), d[:, None] * P * (1.0 - P)

    def value(self, F, Y, d):
        m = F.max(axis=1)
        lse = m + np.log(np.exp(F - m[:, None]).sum(axis=1))
        return float(np.sum(d * (lse - (F * Y).sum(axis=1))))


class ExponentialLoss(BoostingLoss):
    """AdaBoost's exponential loss ``exp(-y f)`` for two classes, with
    ``f = F_1 - F_0`` and ``y = +1`` for the second class, ``-1`` for the
    first. A vote for the second class raises ``f``, one for the first
    lowers it; the second derivative is ``exp(-y f)`` either way."""

    multiclass = False

    def derivatives(self, F, Y, d):
        y = np.where(Y[:, 1] > 0, 1.0, -1.0)
        w = d * np.exp(-y * (F[:, 1] - F[:, 0]))
        return np.stack([y * w, -y * w], axis=1), np.stack([w, w], axis=1)

    def value(self, F, Y, d):
        y = np.where(Y[:, 1] > 0, 1.0, -1.0)
        return float(np.sum(d * np.exp(-y * (F[:, 1] - F[:, 0]))))


_LOSSES = {"logistic": LogisticLoss, "exponential": ExponentialLoss}


class ENDER(NativeRuleLearner):
    """ENDER: boosting of decision rules by forward stagewise minimization
    of a loss (Dembczyński, Kotłowski & Słowiński, DMKD 2010), here as in
    its MLRules instance (ICML 2008), with a pluggable `BoostingLoss`.

    The model keeps a score per class; each rule votes for one class with
    a positive weight, and the class with the highest total wins. It
    starts from a default rule (covering everything) for the class the
    loss favours, with the weight of a Newton step. Then, for `n_rules`
    rounds:

    1. Draw a subsample (`subsample` of the rows, without replacement).
    2. Grow a rule on it: starting from the empty rule (criterion 0),
       add the condition, and choose the class, minimizing ``sum g``
       (`method="gradient"`) or ``sum g / sqrt(sum h)``
       (`method="newton"`) over the covered rows, where ``g``/``h`` are
       the first/second derivatives of the loss for a vote for that
       class; stop when no condition lowers the criterion. A rule is kept
       only if its criterion is negative (a descent direction).
    3. Give it the Newton step ``alpha = -sum g / sum h`` computed on
       *all* rows (which also regularizes it), shrink it by `shrinkage`
       (``nu``), and add it to the scores.

    With `early_stopping`, the rows left out of each subsample are a
    holdout set: a rule is acceptable if its error on the holdout rows it
    covers is below that of guessing among the classes (``1 - 1/K``), and
    the loop stops once 8 of the last 10 rules weren't acceptable (the
    paper's binomial-test choice).

    The result is a `LinearRuleModel`; a rule found in several rounds
    appears once, its weights summed, and the default rule is the
    intercept of its class. Defaults are the paper's (``M = 500``, ``nu =
    0.1``, subsample 0.5, Newton). `loss="logistic"` (default,
    `LogisticLoss`) handles any number of classes, `"exponential"`
    (`ExponentialLoss`) two. Numeric attributes come already binarized
    (`pyrulearn.data.io.build_dataspec`) instead of being thresholded
    during the search; the data's row weights weight the loss.
    """

    def __init__(
        self,
        n_rules: int = 500,
        shrinkage: float = 0.1,
        subsample: float = 0.5,
        loss: Union[str, BoostingLoss] = "logistic",
        method: str = "newton",
        early_stopping: bool = False,
        max_length: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        if n_rules < 1:
            raise ValueError(f"n_rules must be at least 1, got {n_rules}")
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in (0, 1], got {shrinkage}")
        if not 0.0 < subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {subsample}")
        if method not in ("newton", "gradient"):
            raise ValueError(f"method must be 'newton' or 'gradient', got {method!r}")
        self.n_rules = n_rules
        self.shrinkage = shrinkage
        self.subsample = subsample
        self.loss = _LOSSES[loss]() if isinstance(loss, str) else loss
        self.method = method
        self.early_stopping = early_stopping
        self.max_length = max_length
        self.random_state = random_state

    def _default_model(self, data: Any) -> type:
        return LinearRuleModel

    @produces(LinearRuleModel)
    def _fit_native(self, data: Any, **kw) -> LinearRuleModel:
        if data.y is None:
            raise ValueError("ENDER needs data.y")
        y = np.asarray(data.y)
        labels, y_idx = np.unique(y, return_inverse=True)
        classes = [c.item() if isinstance(c, np.generic) else c for c in labels]
        K = len(classes)
        if K < 2:
            raise ValueError("ENDER needs at least two classes")
        if K > 2 and not self.loss.multiclass:
            raise ValueError(f"{self.loss!r} handles two classes only")
        n = len(y)
        Y = np.zeros((n, K))
        Y[np.arange(n), y_idx] = 1.0
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        d = np.ones(n) if data.weights is None else data.weights.astype(float)
        F = np.zeros((n, K))
        rng = np.random.default_rng(self.random_state)

        learned: List[Tuple[Tuple[int, ...], int, float]] = []
        G, H = self.loss.derivatives(F, Y, d)
        k0 = int(np.argmin(self._criterion(G.sum(axis=0), H.sum(axis=0))))
        alpha0 = self._newton(G[:, k0].sum(), H[:, k0].sum())
        F[:, k0] += alpha0                                  # the default rule, not shrunk
        learned.append(((), k0, alpha0))

        size = max(1, int(round(self.subsample * n)))
        verdicts: List[bool] = []
        for _ in range(self.n_rules):
            G, H = self.loss.derivatives(F, Y, d)
            in_sample = np.zeros(n, dtype=bool)
            in_sample[rng.choice(n, size=size, replace=False)] = True
            found = self._grow(X, Xf, G * in_sample[:, None], H * in_sample[:, None])
            if found is None:
                continue
            body, k = found
            cov = np.all(X[:, list(body)], axis=1)
            alpha = self._newton(G[cov, k].sum(), H[cov, k].sum())
            if alpha <= 0:
                continue
            F[cov, k] += self.shrinkage * alpha
            learned.append((body, k, self.shrinkage * alpha))
            if self.early_stopping:
                verdicts.append(self._acceptable(cov & ~in_sample, y_idx, k, d, K))
                if len(verdicts) >= 10 and sum(not v for v in verdicts[-10:]) >= 8:
                    break

        total: Dict[Tuple[Tuple[int, ...], int], float] = {}
        for body, k, a in learned:
            key = (tuple(sorted(body)), k)
            total[key] = total.get(key, 0.0) + a
        rules = [WeightedRule(list(body), target=classes[k], dataspec=data.spec, weight=a)
                 for (body, k), a in total.items() if a != 0]
        return LinearRuleModel(annotate_rules(rules, data), classes=classes)

    # -- helpers ---------------------------------------------------------------

    def _criterion(self, g: np.ndarray, h: np.ndarray) -> np.ndarray:
        if self.method == "gradient":
            return g
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(h > 0, g / np.sqrt(h), 0.0)

    @staticmethod
    def _newton(g: float, h: float) -> float:
        return -g / h if h > 0 else 0.0

    def _grow(self, X: np.ndarray, Xf: np.ndarray, G: np.ndarray,
              H: np.ndarray) -> Optional[Tuple[Tuple[int, ...], int]]:
        """The rule (body, class) minimizing the criterion on the rows where
        `G`/`H` are non-zero, grown greedily from the empty rule; `None`
        if no condition makes the criterion negative."""
        cov = np.ones(X.shape[0], dtype=bool)
        body: List[int] = []
        best_k, current = -1, 0.0
        while self.max_length is None or len(body) < self.max_length:
            c = cov[:, None]
            crit = self._criterion((G * c).T @ Xf, (H * c).T @ Xf)      # K x n_features
            if body:
                crit[:, body] = np.inf
            k, f = np.unravel_index(int(np.argmin(crit)), crit.shape)
            value = float(crit[k, f])
            if not value < current:
                break
            current, best_k = value, int(k)
            body.append(int(f))
            cov = cov & X[:, f]
        return (tuple(body), best_k) if body and current < 0 else None

    @staticmethod
    def _acceptable(holdout_cov: np.ndarray, y_idx: np.ndarray, k: int, d: np.ndarray, K: int) -> bool:
        w = d[holdout_cov]
        if w.sum() <= 0:
            return False
        error = float(w[y_idx[holdout_cov] != k].sum() / w.sum())
        return error < 1.0 - 1.0 / K
