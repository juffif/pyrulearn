"""
pyrulearn.learners.opus
=======================

Two learners built on `pyrulearn.learners.seco.BranchAndBoundSearch`, the
exhaustive search for optimal rules after OPUS (Webb, "OPUS: An efficient
admissible algorithm for unordered search", JAIR 1995):

- `Opus` -- separate-and-conquer with the optimal rule in each step:
  `SeCo` with `BranchAndBoundSearch`, so it is a configuration of the
  framework like `CN2` or `CPAR`. Each rule is the best one by the
  heuristic among all conjunctions (up to `max_conditions` conditions),
  not just the best one a greedy or beam search happens to reach.
- `OpusTopK` -- k-optimal rule discovery in the manner of Webb's Magnum
  Opus: the `k` best rules per class by the heuristic (default `WRAcc`,
  which is leverage for a rule predicting a class), found on all the
  data at once, without a covering loop.
"""

from __future__ import annotations

from typing import Any, Optional

from ..heuristics import Laplace, RuleHeuristic, WRAcc
from ..models import ConceptModel, ConceptSet, MajorityClass, annotate_default_rule, annotate_rules
from ..pruning import AnyOf, PrePruningCriterion, ProductiveRule
from .base import DecomposingLearner, NativeRuleLearner, produces
from .seco import BranchAndBoundSearch, CoveringStrategy, SeCo, SingleRuleLearner, _ClassCountLaplace


class Opus(_ClassCountLaplace, SeCo):
    """Separate-and-conquer with an optimal rule in each step, after Webb
    (1993, 1995), who used OPUS to search for the best classification
    rule inside a covering algorithm: `SeCo` with `BranchAndBoundSearch`
    and `heuristic` (default `Laplace` with the number of classes in the
    data -- ``(tp+1)/(tp+fp+c)``, the preference function of Webb's
    experiments, set anew by each `fit`). Webb's rules were
    conjunctions of ``attribute != value`` conditions only; here they are
    whatever features the data has. Each covering step finds the rule
    with the highest score among all conjunctions of up to
    `max_conditions` conditions --
    exhaustively, with branch-and-bound pruning, so the heuristic must
    reward more covered positives and fewer covered negatives (see
    `BranchAndBoundSearch`).

    Everything else is `SeCo`'s: removal covering by default (`covering=`
    for weighted covering), `filtering`/`stopping` criteria, multi-class
    via `ConceptSet` (`fit(data, model=...)` for the other
    decompositions). With a very fine-grained heuristic like `Laplace`,
    the optimal rule is often a narrow, pure one -- a `filtering`
    criterion (e.g. a minimum coverage) or another heuristic changes the
    trade-off.
    """

    def __init__(
        self,
        heuristic: Optional[RuleHeuristic] = None,
        max_conditions: Optional[int] = None,
        target_class: Any = None,
        filtering: Optional[PrePruningCriterion] = None,
        stopping: Optional[PrePruningCriterion] = None,
        max_rules: Optional[int] = None,
        random_state: Optional[int] = 0,
        covering: Optional[CoveringStrategy] = None,
    ):
        self.heuristic = heuristic
        self.max_conditions = max_conditions
        self.filtering = filtering
        self.stopping = stopping
        self._default_laplace = heuristic is None
        single_rule_learner = SingleRuleLearner(
            heuristic if heuristic is not None else Laplace(),
            search=BranchAndBoundSearch(max_conditions=max_conditions),
            filtering=filtering, stopping=stopping,
        )
        super().__init__(single_rule_learner, target_class=target_class, max_rules=max_rules,
                         random_state=random_state, covering=covering)



class OpusTopK(DecomposingLearner, NativeRuleLearner):
    """k-optimal rule discovery in the manner of Webb's Magnum Opus: for
    each class, the `k` rules with the highest score by `heuristic`
    (default `WRAcc` -- coverage-weighted relative accuracy, which for a
    rule predicting a class is its leverage) among all conjunctions of up
    to `max_conditions` (default 4) conditions, found by one
    `BranchAndBoundSearch` on all the data -- no covering loop, so the
    rules may overlap freely. Rules covering exactly the same rows count
    once. With `productive` (default), only productive rules qualify
    (`pyrulearn.pruning.ProductiveRule`: a rule must have a higher
    confidence than every generalization -- as Magnum Opus filters by
    default), so the `k` rules aren't padded with variants of a better,
    simpler rule. `filtering` restricts further (e.g. a minimum
    coverage). Magnum Opus's statistical tests for spurious rules are not
    reproduced.

    `fit(data)` gives a `ConceptSet`, one concept per class with its `k`
    rules (`target_class=` set: a `ConceptModel` for that class);
    `model=ConceptCascade`/`PairwiseModel` decompose the classes
    differently. The defaults for `k` and `max_conditions` are this
    implementation's, not taken from Magnum Opus.
    """

    def __init__(
        self,
        k: int = 10,
        heuristic: Optional[RuleHeuristic] = None,
        max_conditions: Optional[int] = 4,
        filtering: Optional[PrePruningCriterion] = None,
        productive: bool = True,
        target_class: Any = None,
    ):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self.k = k
        self.heuristic = heuristic
        self.max_conditions = max_conditions
        self.filtering = filtering
        self.productive = productive
        self.target_class = target_class

    def _filtering(self) -> Optional[PrePruningCriterion]:
        criteria = ([ProductiveRule()] if self.productive else []) + ([self.filtering] if self.filtering else [])
        if len(criteria) > 1:
            return AnyOf(*criteria)        # rejected if either rejects
        return criteria[0] if criteria else None

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else ConceptSet

    def _rules(self, data: Any, positive: Any) -> list:
        learner = SingleRuleLearner(
            self.heuristic if self.heuristic is not None else WRAcc(),
            search=BranchAndBoundSearch(max_conditions=self.max_conditions, k=self.k),
            filtering=self._filtering(),
        )
        return annotate_rules(learner.learn_rules(data, positive), data)

    def _fit_binary(self, data: Any, positive: Any, negative: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("OpusTopK needs data.y")
        default = negative if negative is not None else MajorityClass(data)
        return ConceptModel(self._rules(data, positive), label=positive, default_prediction=default)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")`: the `k` best rules for
        `label` (or `target_class`)."""
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        return annotate_default_rule(self._fit_binary(data, target), data)
