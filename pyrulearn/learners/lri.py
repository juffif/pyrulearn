"""
pyrulearn.learners.lri
======================

`LRI` -- Lightweight Rule Induction (Weiss & Indurkhya, ICML 2000), built
from the rule-search components, and `DenseLRI`, the same learner
specialized for speed on a dense matrix.

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

from dataclasses import dataclass
from typing import Any, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from ..heuristics import RuleHeuristic, RuleStats
from ..models import ConceptModel, EnsembleModel, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule
from .base import NativeRuleLearner, produces
from .seco import (
    CoveringState, FeatureSubset, HillClimbing, LRIReweighting, Objective, Reweighting, RuleSearch,
    SearchSpaceInit, SingleRuleLearner, WeightedCovering, count_open_children, stats_from_handle,
)


class DenseLRI(NativeRuleLearner):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000), specialized for
    speed: its term search is hand-written on a dense float copy of the
    data matrix. `LRI` is the same algorithm built from the rule-search
    components, on every data representation.

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


# ======================================================= modular LRI ========

@dataclass(frozen=True)
class _TermStats(RuleStats):
    """`RuleStats` plus what `LRIError` needs along a term: `k`, the error
    weight of the step that produced them, and `body`, the features of the
    rule (of a batch of children: their parent's, and `feature` the one
    each adds)."""
    k: Any = 1.0
    body: Any = frozenset()
    feature: Any = None


def _per_child(value: Any, n: int) -> np.ndarray:
    """`value` as an array of `n` entries (a scalar repeated)."""
    return np.asarray(value) if np.ndim(value) else np.full(n, value)


def _bodies(value: Any, n: int) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = value
    return out


class LRIError(Objective):
    """The error Lightweight Rule Induction minimizes while it grows a term
    (Weiss & Indurkhya 2000, Table 2): the weighted ``FP + k * FN`` of the
    term (with the weights of the search's scope: LRI's, with the cases
    earlier terms of the rule cover removed). A search maximizes the
    score, so it is ``-(FP + k * FN)``.

    `k` defaults to the paper's adaptive choice, per step: 1, doubled while
    the cheapest condition overall -- among *all* conditions not yet in the
    term, those that would leave no true positive included -- would leave
    none. So a condition that keeps a true positive is always found, and
    ``k`` grows (towards keeping positives) only as far as that takes; a
    number fixes ``k`` for every step.

    Only conditions keeping a true positive are candidates (the others are
    dead, as for any heuristic: a child covering no positive); a term is
    consistent once it covers no negative, which ends its growth in
    `HillClimbing(stop_at_local_optimum=False)`. With the adaptive `k`,
    scores of different steps aren't comparable, so run it with a search
    that doesn't compare across steps (`HillClimbing` without the
    local-optimum stop, `BeamSearch(optimistic_pruning=False)`); with a
    fixed `k` the bound ``-k * FN`` (every false positive dropped) is
    admissible. The statistics are `RuleStats`, so filtering and stopping
    criteria work."""

    supports_criteria = True

    def __init__(self, k: Optional[float] = None):
        if k is not None and not k > 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self._universe: Optional[FrozenSet[int]] = None

    @property
    def supports_bound(self) -> bool:
        return self.k is not None

    def begin(self, data, target_class, example_mask) -> None:
        self._universe = None            # the first node of a search fixes its universe of conditions

    def rule_stats(self, data, target_class, rule, handle) -> _TermStats:
        base = stats_from_handle(data, target_class, rule, handle)
        return _TermStats(tp=base.tp, fp=base.fp, fn=base.fn, tn=base.tn, length=base.length,
                          k=1.0 if self.k is None else self.k,
                          body=frozenset(lit.feature for lit in rule.conditions))

    def _choose_k(self, tp, fp, fn) -> float:
        if self.k is not None:
            return float(self.k)
        live = tp > 0
        dead = ~live
        if not live.any() or not dead.any():
            return 1.0
        k = 1.0
        while k < 2.0 ** 60:
            err = fp + k * fn
            if float(err[dead].min()) >= float(err[live].min()):
                break
            k *= 2.0
        return k

    def children(self, data, target_class, mask, handle, stats: _TermStats):
        if self._universe is None:
            self._universe = frozenset(mask)
        open_features = self._universe - stats.body
        features, tp, fp, fn, tn, _ = count_open_children(data, target_class, open_features, handle, stats)
        features = np.asarray(features, dtype=np.int64)
        tp, fp, fn, tn = (np.asarray(a) for a in (tp, fp, fn, tn))
        k = self._choose_k(tp, fp, fn)
        in_mask = np.isin(features, np.fromiter(mask, dtype=np.int64, count=len(mask)))
        alive = (tp > 0) & in_mask
        dead = set(features[(tp <= 0) & in_mask].tolist())
        batch = _TermStats(tp=tp[alive], fp=fp[alive], fn=fn[alive], tn=tn[alive], length=stats.length + 1,
                           k=k, body=stats.body, feature=features[alive])
        # counts of children keeping every row are the parent's exactly: nothing to guard
        return features[alive], batch, np.zeros(int(alive.sum()), dtype=bool), dead

    def batch_score(self, batch: _TermStats, parent=None):
        return -(np.asarray(batch.fp, dtype=float) + batch.k * np.asarray(batch.fn, dtype=float))

    def score_one(self, stats: _TermStats, parent=None) -> float:
        return -(float(stats.fp) + float(stats.k) * float(stats.fn))

    def batch_bound(self, batch: _TermStats):
        return -(batch.k * np.asarray(batch.fn, dtype=float))

    def bound_one(self, stats: _TermStats) -> float:
        return -(float(stats.k) * float(stats.fn))

    def take(self, batch: _TermStats, index) -> _TermStats:
        n = len(np.atleast_1d(batch.tp))

        def cut(v):
            return np.asarray(v)[index] if np.ndim(v) else v
        body = batch.body if not isinstance(batch.body, np.ndarray) else batch.body[index]
        return _TermStats(tp=batch.tp[index], fp=batch.fp[index], fn=batch.fn[index], tn=batch.tn[index],
                          length=cut(batch.length), k=cut(batch.k), body=body, feature=cut(batch.feature))

    def at(self, batch: _TermStats, i: int) -> _TermStats:
        body = batch.body[i] if isinstance(batch.body, np.ndarray) else batch.body
        length = batch.length if np.ndim(batch.length) == 0 else int(np.asarray(batch.length)[i])
        k = batch.k if np.ndim(batch.k) == 0 else float(np.asarray(batch.k)[i])
        return _TermStats(tp=batch.tp[i], fp=batch.fp[i], fn=batch.fn[i], tn=batch.tn[i], length=length,
                          k=k, body=body | {int(batch.feature[i])})

    def concat(self, batches) -> _TermStats:
        sizes = [len(np.atleast_1d(b.tp)) for b in batches]
        return _TermStats(
            tp=np.concatenate([b.tp for b in batches]), fp=np.concatenate([b.fp for b in batches]),
            fn=np.concatenate([b.fn for b in batches]), tn=np.concatenate([b.tn for b in batches]),
            length=np.concatenate([_per_child(b.length, n) for b, n in zip(batches, sizes)]),
            k=np.concatenate([_per_child(b.k, n) for b, n in zip(batches, sizes)]),
            body=np.concatenate([_bodies(b.body, n) for b, n in zip(batches, sizes)]),
            feature=np.concatenate([np.asarray(b.feature) for b in batches]),
        )

    def dead(self, stats: _TermStats) -> bool:
        return stats.tp == 0

    def consistent(self, stats: _TermStats) -> bool:
        return stats.fp == 0

    def __repr__(self) -> str:
        return "LRIError()" if self.k is None else f"LRIError(k={self.k!r})"


class LRI(NativeRuleLearner):
    """Lightweight Rule Induction (Weiss & Indurkhya 2000), built from the
    rule-search components: see `DenseLRI` for the algorithm in detail --
    the same learner, specialized for speed on a dense matrix, with its
    own term search.

    For each class `c` (every class, including both classes of a binary
    problem), `n_rules` DNF rules for `c` against the rest, one after
    another on the whole training data:

    1. A **term** is what a `SingleRuleLearner` finds: `search` (default
       ``HillClimbing(max_conditions=5, stop_at_local_optimum=False)`` --
       add the best condition each step, even a worsening one, until the
       term covers no negative or has 5 conditions) over the objective
       `heuristic` (default `LRIError`: the weighted ``FP + k * FN`` with
       the paper's adaptive `k`), on the cases with their current weights.
       Any search works; `max_conditions` of the search is the term
       length.
    2. A **rule** has up to `max_terms` terms: after each, the cases it
       covers are removed and the next term is grown on the rest, until
       every positive is covered.
    3. After each rule the cases are reweighted by `reweighting` (default
       `LRIReweighting(power=3, max_errors=32)`: ``1 + e**3``, `e` the
       number of rules so far that err on a case) through a
       `WeightedCovering` of `n_rules` rounds.

    After `freeze_features_after` rules of a class, only the features its
    rules have used so far stay open (a `FeatureSubset`; ``None`` never
    freezes). Rules found again are kept: they add votes.

    The model is an `EnsembleModel` of the DNF rules (`ConceptModel`s
    without a default, so each votes only where it fires): the class with
    the most satisfied rules wins; ties, and rows no rule fires on, go to
    the more frequent training class.

    Runs on every data representation, the search walking cover handles
    with constraint propagation -- a condition the term already implies is
    never added. `DenseLRI` doesn't propagate constraints: with negation or
    threshold features it can append a redundant condition where this
    learner takes the next best real one (and without such features both
    learn identical models). The data's row weights multiply with the
    error weights.
    """

    def __init__(
        self,
        n_rules: int = 50,
        max_terms: int = 4,
        search: Optional[RuleSearch] = None,
        heuristic: Optional[Union[RuleHeuristic, Objective]] = None,
        reweighting: Optional[Reweighting] = None,
        freeze_features_after: Optional[int] = 50,
    ):
        if n_rules < 1 or max_terms < 1:
            raise ValueError("n_rules and max_terms must be at least 1")
        if search is not None and not isinstance(search, RuleSearch):
            raise ValueError(f"search must be a RuleSearch, got {search!r}")
        self.n_rules = n_rules
        self.max_terms = max_terms
        self.search = search if search is not None else HillClimbing(max_conditions=5, stop_at_local_optimum=False)
        self.heuristic = heuristic if heuristic is not None else LRIError()
        self.reweighting = reweighting if reweighting is not None else LRIReweighting()
        self.freeze_features_after = freeze_features_after

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
        term_learner = SingleRuleLearner(self.heuristic, search=self.search)
        members: List[ConceptModel] = []
        for c in classes:
            for terms in self._rules_for(data, c, term_learner):
                members.append(ConceptModel(annotate_rules(terms, data), label=c))
        model = EnsembleModel(members, default_prediction=MajorityClass(data))
        return annotate_default_rule(model, data)

    def _rules_for(self, data: Any, c: Any, term_learner: SingleRuleLearner) -> List[List[Rule]]:
        """`n_rules` DNF rules (each a list of term rules) for `c` vs. the rest."""
        positive = np.asarray(data.y) == c
        covering = WeightedCovering(self.reweighting, stop=[], max_rounds=self.n_rules)
        state = covering.start(data, positive)
        space = None
        used: Set[int] = set()
        rules: List[List[Rule]] = []
        while not covering.exhausted(state):
            if (self.freeze_features_after is not None and state.round == self.freeze_features_after and used):
                space = FeatureSubset(sorted(used))
            terms, covered = self._grow_rule(data, c, positive, state.scope, term_learner, space)
            if not terms:
                break
            rules.append(terms)
            used.update(lit.feature for t in terms for lit in t.conditions)
            covering.update(state, covered)
        return rules

    def _grow_rule(self, data: Any, c: Any, positive: np.ndarray, scope: np.ndarray,
                   term_learner: SingleRuleLearner, space: Optional[SearchSpaceInit]) -> Tuple[List[Rule], np.ndarray]:
        """One DNF rule: its terms, and the rows it covers."""
        weights = data.scope_weights(scope)
        remaining = np.ones(len(positive), dtype=bool)
        covered = np.zeros(len(positive), dtype=bool)
        terms: List[Rule] = []
        while len(terms) < self.max_terms:
            term = term_learner.learn_one_rule(data, c, scope * remaining, space_init=space)
            if term is None or term.length() == 0:
                break
            terms.append(term)
            term_cov = np.asarray(data.coverage(term), dtype=bool)
            covered |= term_cov
            if not np.any(positive & ~covered & (weights > 0)):
                break                       # every positive covered
            remaining &= ~term_cov
        return terms, covered
