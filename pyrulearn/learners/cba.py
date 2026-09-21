"""
pyrulearn.learners.cba
==========================

`CBA` -- Classification Based on Associations (Liu, Hsu & Ma, KDD 1998):
consume a pool of class association rules (a `FlatRuleSet`, via
`pyrulearn.learners.associative.RuleDistiller` -- either given directly or
mined by default via `ClassAssociationRuleMiner`, the same shared miner
`pyrulearn.learners.cmar.CMAR` and `pyrulearn.learners.ids.IDS` also
consume from), sort them by precedence, then build a
`pyrulearn.models.DecisionList` via CBA-CB's "M1" database-coverage
selection -- a genuinely different classifier-building step from a
covering loop or a greedy search (see `CBA._select`'s docstring for the
exact, easy-to-get-wrong semantics).

- **Precedence sort** (`pyrulearn.learners.associative.
  sort_by_measured_precedence`) -- CBA's own criterion, specific to this
  algorithm (CMAR ranks by significance instead, never shares this):
  `r1` precedes `r2` iff `r1` has higher confidence; tie -> higher
  support; tie -> fewer conditions (more general).
- **CBA-CB, "M1"** (`CBA._select`) -- walks the precedence-sorted rules
  via `pyrulearn.learners.associative.coverage_select` (the shared
  database-coverage primitive: keep a rule only if it correctly
  classifies at least one still-unclaimed row, then claim every row it
  covers -- cross-checked rule-for-rule against `pyarc`), then truncates the kept
  sequence to the **minimum-total-training-error prefix** -- CBA's
  actual accuracy/complexity trade-off, not just "keep every rule that
  ever helped."

Native multi-class already (each rule carries its own class head), so
`fit(data)` with no `model=` builds a `DecisionList` directly.
`DecomposingLearner` is mixed in for consistency with every other native
learner here (`ConceptSet`/`ConceptCascade`/`PairwiseModel` via
`target_class=`/`_fit_binary`, the latter provided by `RuleDistiller`).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .associative import RuleDistiller, coverage_select, sort_by_measured_precedence
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..models import ConceptModel, DecisionList, MajorityClass, annotate_default_rule, annotate_rules
from ..rule import Rule


class CBA(RuleDistiller, DecomposingLearner, NativeRuleLearner):
    """Classification Based on Associations (Liu, Hsu & Ma, 1998). See
    the module docstring for the precedence sort and exactly what
    CBA-CB's "M1" prefix-truncation does. Constructor arguments
    (`rules`/`min_support`/`min_confidence`/`max_len`/`target_class`/
    `max_auto_convert_cells`) are `RuleDistiller`'s, unchanged.
    """

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else DecisionList

    @produces(DecisionList)
    def _fit_native(self, data: Any, **kw) -> DecisionList:
        if data.y is None:
            raise ValueError("CBA.fit needs data.y")
        ordered_rules = sort_by_measured_precedence(self._resolve_rules(data))
        kept, default_label = self._select(ordered_rules, data)
        default = default_label if default_label is not None else MajorityClass(data)
        model = DecisionList(annotate_rules(kept, data), default_prediction=default)
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("CBA needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        ordered_rules = sort_by_measured_precedence(self._resolve_rules(data, only_class=target))
        kept, _default_label = self._select(ordered_rules, data)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(annotate_rules(kept, data), label=target, default_prediction=default)
        return annotate_default_rule(model, data)

    @staticmethod
    def _select(ordered_rules: List[Rule], data: Any) -> Tuple[List[Rule], Optional[Any]]:
        """CBA-CB, "M1": `coverage_select` gives the kept-rule sequence
        and its error trace; the final classifier truncates to the
        prefix with the lowest recorded total error (ties -> the first,
        i.e. shortest, occurrence -- plain `min` over a list built in
        increasing-prefix-length order already does this)."""
        kept, trace = coverage_select(ordered_rules, data)
        if not trace:
            return [], None
        best_len = min(range(len(trace)), key=lambda i: trace[i][0]) + 1
        return kept[:best_len], trace[best_len - 1][1]
