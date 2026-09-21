"""
pyrulearn.learners.cmar
===========================

`CMAR` -- Classification based on Multiple Association Rules (Li, Han &
Pei, 2001): consume a pool of class association rules (a `FlatRuleSet`,
via `pyrulearn.learners.associative.RuleDistiller` -- either given
directly or mined by default via `ClassAssociationRuleMiner`, the same
shared miner `pyrulearn.learners.cba.CBA` and `pyrulearn.learners.ids.
IDS` also consume from), keep only the ones whose antecedent is
*significantly* correlated with the class (a chi-square test, not
merely confident), then predict by pooling every matching rule's vote,
weighted by its own significance -- unlike `CBA`, which commits to a
single first-match `DecisionList`.

Deliberately not chasing every detail of the original paper -- per
explicit direction, this reuses existing structure wherever possible,
accepting "slight deviations" from the paper rather than reproducing it
exactly:

- **Significance filtering** reuses `pyrulearn.pruning.ThresholdPrePruning`
  with the new `pyrulearn.heuristics.ChiSquare` heuristic -- the exact
  "score a heuristic against a critical value" machinery
  `pyrulearn.learners.seco.CN2` already uses for its own significance
  test (`LikelihoodRatio`), just applied to a flat pool of already-mined
  rules instead of gating a live search.
- **Pruning** reuses `pyrulearn.learners.associative.coverage_select` --
  the same database-coverage walk `CBA`'s own CBA-CB step is built on --
  but applied **once per predicted class**, not globally across the
  whole significant pool. A kept rule claims *every* row it covers, so
  globally, a kept rule of one class would claim rows a later rule of a
  *different* class could never vote on again -- the semantics a
  first-match `DecisionList` needs, and wrong for a voting ensemble
  (confirmed directly: reusing it globally measurably hurt accuracy).
  Run once per class, it cuts a same-class candidate set down to the
  rules that each correctly classify at least one still-unclaimed
  row of their own class (CMAR's paper: "correctly classifies at least
  one remaining object" -- the same condition as CBA-CB, i.e. the
  default `keep=CoveredPositives()`), while every class stays free to
  vote on any row its own surviving rules cover. (An earlier version
  used the any-new-coverage condition here, right or wrong: 55 rules /
  0.855 test accuracy on `vote`, vs. 8 rules / 0.939 with this one --
  that, not the significance threshold, was why CMAR trailed CBA.) The
  paper's covering threshold delta=4 (an object leaves only after being
  covered by 4 rules) is not reproduced -- ours is delta=1 -- nor its
  subsumption pruning of a rule by a higher-precedence, no-worse
  general ancestor.
- **Voting** reuses `pyrulearn.combiners.HeuristicVoteCombiner`, pooling
  every surviving rule and weighting each covering rule's vote by the
  same `ChiSquare` score -- CMAR's own "weighted chi-square"
  combination, essentially verbatim. The paper's own formula
  additionally normalizes by the single largest chi-square value seen
  anywhere in the rule set; not reproduced.

Native multi-class already (each rule carries its own class head).
`DecomposingLearner` is mixed in for consistency with every other native
learner here, not because CMAR needs it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .associative import DEFAULT_MAX_AUTO_CONVERT_CELLS, RuleDistiller, coverage_select, iterate_rules
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..combiners import HeuristicVoteCombiner, _rule_stats
from ..heuristics import ChiSquare
from ..models import ConceptModel, FlatRuleSet, MajorityClass, annotate_default_rule
from ..pruning import ThresholdPrePruning
from ..rule import Rule


class CMAR(RuleDistiller, DecomposingLearner, NativeRuleLearner):
    """Classification based on Multiple Association Rules (Li, Han &
    Pei, 2001). See the module docstring for exactly which pieces are
    reused as-is vs. deliberately simplified relative to the paper.

    - `significance_threshold` (default `3.841`, the chi-squared
      critical value at alpha=0.05 with 1 degree of freedom -- the same
      default `pyrulearn.learners.seco.CN2` already uses for its own
      significance test) -- a rule is dropped unless its `ChiSquare`
      score clears this.
    - `yates_correction` -- passed straight to `ChiSquare`.
    - The rest (`rules`/`min_support`/`min_confidence`/`max_len`/
      `target_class`/`max_auto_convert_cells`) are `RuleDistiller`'s,
      unchanged.
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int = 4,
        significance_threshold: float = 3.841,
        yates_correction: bool = True,
        target_class: Optional[Any] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(rules, min_support, min_confidence, max_len, max_auto_convert_cells, target_class)
        self.significance_threshold = significance_threshold
        self.yates_correction = yates_correction

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else FlatRuleSet

    def _significant(self, rules: Sequence[Rule], heuristic: ChiSquare) -> List[Rule]:
        """Drop rules that aren't significant (`ThresholdPrePruning`,
        same pattern `CN2` uses for its own significance test), sorted by
        significance descending -- CMAR's own rule strength, in place of
        CBA's confidence-based precedence. One streaming pass
        (`iterate_rules`): over a lazy pool only the significant rules
        stay in memory."""
        gate = ThresholdPrePruning(heuristic, self.significance_threshold, "<")
        significant = [r for r in iterate_rules(rules) if not gate.reject(r, _rule_stats(r), None, r.target)]
        return sorted(significant, key=lambda r: -heuristic.score(_rule_stats(r)))

    def _prune(self, ordered_rules: List[Rule], data: Any, heuristic: ChiSquare) -> List[Rule]:
        """`coverage_select`, run once per predicted class -- see the
        module docstring for why per-class, not globally."""
        by_class: Dict[Any, List[Rule]] = {}
        for r in ordered_rules:
            by_class.setdefault(r.target, []).append(r)
        kept: List[Rule] = []
        for group in by_class.values():
            selected, _trace = coverage_select(group, data)
            kept.extend(selected)
        return sorted(kept, key=lambda r: -heuristic.score(_rule_stats(r)))

    @produces(FlatRuleSet)
    def _fit_native(self, data: Any, **kw) -> FlatRuleSet:
        if data.y is None:
            raise ValueError("CMAR.fit needs data.y")
        heuristic = ChiSquare(yates_correction=self.yates_correction)
        kept = self._prune(self._significant(self._resolve_rules(data), heuristic), data, heuristic)
        model = FlatRuleSet(kept, default_prediction=MajorityClass(data),
                            combiner=HeuristicVoteCombiner(heuristic))
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("CMAR needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        heuristic = ChiSquare(yates_correction=self.yates_correction)
        pool = self._resolve_rules(data, only_class=target)
        kept = self._prune(self._significant(pool, heuristic), data, heuristic)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(kept, label=target, default_prediction=default)
        return annotate_default_rule(model, data)
