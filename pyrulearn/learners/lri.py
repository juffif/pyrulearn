"""
pyrulearn.learners.lri
======================

`LRI` -- Lightweight Rule Induction (Weiss & Indurkhya, ICML 2000).

Every class gets the same number of unweighted DNF rules, and a new
example goes to the class with the most satisfied rules. A DNF rule is a
`pyrulearn.models.ConceptModel` (its terms are rules with the same head;
it fires where any of them does, and abstains elsewhere), and the model
is an `pyrulearn.models.EnsembleModel` voting over all of them.

Per class, against the rest, the rules are learned one after another on
the whole training data, reweighted after each rule by the cumulative
number of errors the rules so far made on each case
(`pyrulearn.learners.seco.LRIReweighting`: weight ``1 + e**3``). A rule
is grown term by term (Table 2 of the paper): a term is a conjunction
grown greedily, condition by condition, minimizing the weighted error
``FP + k * FN``; after each term the cases it covers are removed and the
next term is grown on the rest, until `max_terms` terms or no positive is
left uncovered. There is no pruning and no default rule.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..models import ConceptModel, EnsembleModel, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import NativeRuleLearner, produces
from .seco import CoveringState, LRIReweighting


class LRI(NativeRuleLearner):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000).

    For each class `c` (every class, including both classes of a binary
    problem), `n_rules` DNF rules for `c` against the rest:

    1. Grow a conjunctive term by greedily adding the condition with the
       lowest weighted error ``err1 = FP + k * FN`` among the conditions
       that keep at least one true positive. ``k`` starts at 1 for each
       condition and is doubled while the cheapest condition overall
       would leave no true positive -- so a condition is always added
       while the term still covers negatives. The term stops at
       `max_length` conditions or when it covers no negative.
    2. Add the term to the rule. If the rule has fewer than `max_terms`
       terms and some positive is still uncovered, remove the cases the
       term covers and grow the next term on the rest.
    3. Count, for every training case, whether the finished rule errs on
       it (covers a negative, or leaves a positive uncovered), and
       reweight: ``1 + e**3`` with ``e`` the cumulative errors
       (`LRIReweighting`; halved once one exceeds 32).

    After `freeze_features_after` rules of a class, only the features its
    rules have used so far remain candidates (the paper's speed-up, 50 in
    its experiments; ``None`` never freezes). Rules found again are kept:
    they add votes, as in the paper.

    The model is an `EnsembleModel` of the DNF rules (`ConceptModel`s
    without a default, so each votes only where it fires): the class
    with the most satisfied rules wins; ties, and rows no rule fires on,
    go to the more frequent training class.

    Two readings of the paper to note. Its Table 2 grows a term "until
    FN = 0"; adding a condition can never lower FN, while a term covering
    no negative (FP = 0) can't improve any more, so the term stops at
    FP = 0 here. And "if no added condition adds a true positive, the
    cost of a false negative error is doubled" is read as above. Numeric
    attributes come already binarized (`pyrulearn.data.io.build_dataspec`)
    instead of being thresholded during the search; the data's row
    weights multiply with the error weights.
    """

    def __init__(
        self,
        n_rules: int = 50,
        max_terms: int = 4,
        max_length: int = 5,
        freeze_features_after: Optional[int] = 50,
        power: float = 3.0,
        max_errors: Optional[int] = 32,
    ):
        if n_rules < 1 or max_terms < 1 or max_length < 1:
            raise ValueError("n_rules, max_terms and max_length must be at least 1")
        self.n_rules = n_rules
        self.max_terms = max_terms
        self.max_length = max_length
        self.freeze_features_after = freeze_features_after
        self.power = power
        self.max_errors = max_errors

    def _default_model(self, data: Any) -> type:
        return EnsembleModel

    @produces(EnsembleModel)
    def _fit_native(self, data: Any, **kw) -> EnsembleModel:
        if data.y is None:
            raise ValueError("LRI needs data.y")
        y = np.asarray(data.y)
        classes = [c.item() if isinstance(c, np.generic) else c for c in np.unique(y)]
        if len(classes) < 2:
            raise ValueError("LRI needs at least two classes")
        X = np.asarray(data.X, dtype=bool)
        Xf = X.astype(float)
        base_w = np.ones(len(y)) if data.weights is None else data.weights.astype(float)
        members: List[ConceptModel] = []
        for c in classes:
            for terms in self._rules_for(X, Xf, y == c, base_w):
                rules = [Rule(list(t), target=c, dataspec=data.spec) for t in terms]
                members.append(ConceptModel(annotate_rules(rules, data), label=c))
        model = EnsembleModel(members, default_prediction=MajorityClass(data))
        return annotate_default_rule(model, data)

    # -- one class ----------------------------------------------------------

    def _rules_for(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray,
                   base_w: np.ndarray) -> List[List[Tuple[int, ...]]]:
        """`n_rules` DNF rules (each a list of terms) for `positive` vs. the rest."""
        n, n_features = X.shape
        state = CoveringState(base_w.copy(), positive)
        reweighting = LRIReweighting(power=self.power, max_errors=self.max_errors)
        allowed: Set[int] = set(range(n_features))
        used: Set[int] = set()
        rules: List[List[Tuple[int, ...]]] = []
        for r in range(self.n_rules):
            if self.freeze_features_after is not None and r == self.freeze_features_after and used:
                allowed = set(used)
            terms, covered = self._grow_rule(X, Xf, positive, state.scope, allowed)
            if not terms:
                break
            rules.append(terms)
            used.update(f for t in terms for f in t)
            state.record(covered)
            state.scope = base_w * reweighting.weights(state, covered)
        return rules

    def _grow_rule(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray, w: np.ndarray,
                   allowed: Set[int]) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
        """One DNF rule: its terms, and the rows it covers."""
        remaining = np.ones(len(w), dtype=bool)
        covered = np.zeros(len(w), dtype=bool)
        terms: List[Tuple[int, ...]] = []
        while len(terms) < self.max_terms:
            term, term_cov = self._grow_term(X, Xf, positive, w * remaining, allowed)
            if not term:
                break
            terms.append(term)
            covered |= term_cov
            if not np.any(positive & ~covered & (w > 0)):
                break                       # FN = 0: every positive covered
            remaining &= ~term_cov
        return terms, covered

    def _grow_term(self, X: np.ndarray, Xf: np.ndarray, positive: np.ndarray, w: np.ndarray,
                   allowed: Set[int]) -> Tuple[Tuple[int, ...], np.ndarray]:
        """One conjunctive term grown on the weights `w` (zero for removed
        cases), and the rows (of all) it covers."""
        wp, wn = w * positive, w * ~positive
        n_pos = float(wp.sum())
        cov = np.ones(len(w), dtype=bool)
        body: List[int] = []
        candidates = np.array(sorted(allowed), dtype=np.int64)
        if n_pos <= 0 or candidates.size == 0:
            return (), cov & False
        fp = float(wn.sum())
        while len(body) < self.max_length and fp > 0:
            # weighted TP/FP of the term with each candidate condition added
            tp_c = (wp * cov) @ Xf[:, candidates]
            fp_c = (wn * cov) @ Xf[:, candidates]
            fn_c = n_pos - tp_c
            live = tp_c > 0
            if body:
                live &= ~np.isin(candidates, body)
            if not live.any():
                break
            dead = ~live
            if body:
                dead &= ~np.isin(candidates, body)
            k = 1.0
            while True:
                err = fp_c + k * fn_c
                best_live = float(err[live].min())
                if not dead.any() or float(err[dead].min()) >= best_live:
                    break
                k *= 2.0               # the cheapest condition would drop every true positive
            j = int(np.flatnonzero(live & (err == best_live))[0])
            f = int(candidates[j])
            body.append(f)
            cov = cov & X[:, f]
            fp = float(fp_c[j])
        return tuple(body), cov
