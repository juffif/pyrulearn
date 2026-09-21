"""
pyrulearn.learners.pylord
=============================

`PyLORD` -- a native (`pyrulearn.learners.NativeRuleLearner`), deliberately
simplified reimplementation of Huynh & Fürnkranz's LORD (*Efficient
learning of large sets of locally optimal classification rules*, Machine
Learning 112 (2023) 571-610), built from `pyrulearn.learners.seco`'s composable
pieces rather than LORD's own machinery. The reference implementation is
reachable as an external tool via `pyrulearn.interfaces.lord`
(`LORDImporter` / `run_lord`).

LORD's shape, kept:

1. **seed on every example.** For each training row, search the space of
   rules covering it (the row's own feature values, `pyrulearn.learners.seco.
   SeedExample` with ``strategy="index"``) for the best one by the chosen
   metric -- unlike a separate-and-conquer covering loop, which seeds one
   rule per iteration on a shrinking "remaining" set. Each row's rule
   targets that row's own class, so this is natively multi-class.
   (`target_class=`, off by default, restricts seeding to one class's
   rows -- a one-vs-rest knob outside LORD, see `PyLORD`.)
2. **grow, then prune, both on the full training set.** LORD grows the
   rule greedily -- add the selector that most improves the m-estimate,
   repeat -- then prunes -- drop conditions while that improves the
   m-estimate -- to a local optimum where neither helps. Here: greedy
   grow is `pyrulearn.learners.seco.BeamSearch(beam_width=1)` (a wider beam, with
   optimistic pruning, approaches LORD's *exhaustive* branch-and-bound
   variant -- same ``metric(tp, 0)`` bound -- but a finite beam isn't a
   proof); the prune is `pyrulearn.learners.seco.ReducedErrorPruning` -- RIPPER's
   own routine -- scored against the **full training set** rather than a
   held-out split, since LORD grows and prunes on the same data. That
   trims a *trailing run* of conditions, not an arbitrary middle one; for
   a greedily-grown rule (least-useful conditions added last) it's a
   close approximation of LORD's backward elimination. (`prune_fraction`
   switches this to a RIPPER/IREP held-out grow/prune split instead --
   `None`, the faithful default, keeps it train-only.)
3. **collect + coverage-filter.** Pool every rule found, deduplicate, then
   keep only the rules that "win" for at least one example: for each row,
   among pooled rules covering it whose head is that row's class, keep the
   single best (metric, then positives covered) -- exactly LORD's
   `RuleManager` filter. A rule dominated on every row it covers is
   dropped.
4. **predict best-rule-wins.** `RuleSet` + `HeuristicMaxCombiner`
   (`RuleSet.predict`'s default), with a training-majority default
   class -- LORD's `get_best_covering_rule` + `defaultClassID`. Kept
   rules carry no stored weight -- the *default* combiner scores each
   one from its own measured `stats()` with `Laplace` (unless
   overridden) at predict time instead; pass
   `combiner=HeuristicMaxCombiner(self.metric)` explicitly to have
   `predict` rank rules by this same `metric` LORD itself used to
   grow/prune/filter them.

LORD's shape, *not fully* kept -- the "Py" in `PyLORD` (see the repo
discussion):

- **N-lists.** LORD's speed comes from a PPC-tree / N-list vertical
  index that makes "support of this refined rule" an O(|N-list|)
  intersection of the *parent* rule's N-list with the added item's,
  instead of an O(n_rows) recount.
  `pyrulearn.data.NListRepresentation` builds that index --
  passing one to `PyLORD.fit` instead of a `BooleanDataRepresentation`
  runs the whole every-example search on it, with identical rules (see
  `examples/demo_representations.py`). `pyrulearn.learners.seco.BeamSearch`/
  `HillClimbing` now thread that N-list incrementally too
  (`initial_cover`/`refine_cover`/`cover_counts`), so this is
  substantively LORD's real scheme, not just a from-scratch recompute
  per candidate -- what's still simplified: `NListRepresentation` keeps
  each node's *full ancestor bitmask* rather than pre/post codes, so a
  refinement by a feature *deeper* than every one fixed so far still
  needs a fresh look at that item's own N-list (cheap, but not free);
  only a refinement no deeper than the current deepest gets the pure
  O(|current|) filter LORD's own pre/post-code scheme gets unconditionally.
- **exact backward elimination.** LORD's prune removes *any* condition;
  the reused RIPPER routine only removes a trailing run (see step 2).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

import copy

from ..heuristics import MEstimate, RuleHeuristic, RuleStats
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..models import (
    ConceptModel, FlatRuleSet, MajorityClass, SingleRule, annotate_default_rule, annotate_rules,
)
from ..data import BooleanDataRepresentation
from ..rule import Rule
from .seco import (
    BeamSearch, GrowPruneSplit, NoPostProcessing, NoSplit, ReducedErrorPruning,
    RuleSearch, SeedExample, SingleRulePostProcessing, SingleRulePreparation,
)


class PyLORD(DecomposingLearner, NativeRuleLearner):
    """Simplified locally-optimal-rules learner (after Huynh & Fürnkranz,
    2023), reimplemented on `pyrulearn.learners.seco`'s building blocks. See the
    module docstring for what's faithful and what the "Py" leaves out.

    - `m` -- the m-estimate parameter; `metric` defaults to
      `pyrulearn.heuristics.MEstimate(m)`, used for the per-example
      grow *and* prune *and* the coverage filter's per-row best-rule
      choice. Pass any `RuleHeuristic` to change it (a plain float
      score, not a `LEF`).
    - `beam_width` (default `1`) -- greedy grow, matching LORD's
      `search_for_greedy_best_rule`. A wider beam (with optimistic
      pruning) approaches LORD's exhaustive branch-and-bound variant.
    - `prune` (default `True`) -- run LORD's prune phase at all
      (`ReducedErrorPruning`, RIPPER's truncation routine, on `metric`).
      `prune=False` skips it.
    - `prune_fraction` (default `None`) -- **faithful LORD** (`None`):
      grow and prune both on the full training set, since that's what
      LORD does. A float in ``(0, 1)`` switches to **RIPPER/IREP**: split
      each seed's data into a growing set and a held-out pruning set of
      that fraction (re-drawn per seed, `random_state`-offset), grow on
      one, prune on the other.
    - `max_conditions` -- optional hard cap on rule length (passed to
      `BeamSearch`).
    - `skip_covered` (default `False`) -- **faithful LORD** (`False`)
      seeds a search on *every* row. `True` skips rows already covered by
      a pooled rule of their class, turning the loop into a
      separate-and-conquer covering loop (AQR-shaped) -- fast, but no
      longer LORD. A comparison flag, not a faithfulness knob.
    - `target_class` (default `None`) -- **faithful LORD** (`None`) seeds
      on every row for its own class, natively multi-class. Set it to
      learn a rule set for one class only (seed just that class's rows,
      exactly like `SeCo`) -- a one-vs-rest decomposition knob, not part
      of LORD.
    - `search`/`preparation`/`postprocessing` -- override the composed
      defaults wholesale.

    Like `SeCo`, `PyLORD` takes no default-prediction/combiner arguments:
    the fitted `RuleSet` gets `default_prediction = MajorityClass(data)`
    (LORD's training-majority `defaultClassID`) and `combiner = "max"`
    (`HeuristicMaxCombiner`, scored from measured stats via `Laplace` by
    default -- LORD's own `get_best_covering_rule` instead uses its own
    metric directly; pass `combiner=HeuristicMaxCombiner(self.metric)`
    to match that exactly), both reassignable on the result.

    **Do not use with `pyrulearn.learners.multiclass.OrderedOneVsRest`** (or any
    first-match decision list). LORD's every-example seeding produces a
    large pool of individually-imprecise, locally-optimal rules that are
    only accurate *in aggregate, weighted* -- exactly what `combiner =
    "max"` gives you, and exactly what a position-only `RuleList` throws
    away: an over-general rule near the top of the list captures examples
    that a lower, more precise rule of another class should have won.
    `OneVsRest(PyLORD(...))` (a weighted `RuleSet`) is fine; a
    precision-oriented learner like `CN2` is the one to feed a decision
    list.
    """

    def __init__(
        self,
        m: float = 0.1,
        beam_width: int = 1,
        prune: bool = True,
        prune_fraction: Optional[float] = None,
        max_conditions: Optional[int] = None,
        random_state: Optional[int] = None,
        skip_covered: bool = False,
        target_class: Optional[Any] = None,
        metric: Optional[RuleHeuristic] = None,
        search: Optional[RuleSearch] = None,
        preparation: Optional[SingleRulePreparation] = None,
        postprocessing: Optional[SingleRulePostProcessing] = None,
    ):
        self.m = m
        self.beam_width = beam_width
        self.prune = prune
        self.prune_fraction = prune_fraction
        self.max_conditions = max_conditions
        self.random_state = random_state
        self.skip_covered = skip_covered
        self.target_class = target_class
        self.metric = metric if metric is not None else MEstimate(m)
        self.search = search
        self.preparation = preparation
        self.postprocessing = postprocessing

    # -- the three stages ------------------------------------------------

    def _search_pool(self, rep: BooleanDataRepresentation) -> List[Rule]:
        search = self.search if self.search is not None else BeamSearch(
            beam_width=self.beam_width, max_conditions=self.max_conditions,
        )
        post = self.postprocessing if self.postprocessing is not None else (
            ReducedErrorPruning(self.metric) if self.prune else NoPostProcessing()
        )
        holdout = self.prune and self.prune_fraction is not None and self.preparation is None

        n = rep.n_samples
        all_rows = np.ones(n, dtype=bool)  # LORD prunes on the full training set
        pool: dict = {}  # Rule -> Rule (order-independent identity == LORD's signature)
        covered = np.zeros(n, dtype=bool) if self.skip_covered else None

        for i in range(n):
            if covered is not None and covered[i]:
                continue
            target = rep.y[i]
            if self.target_class is not None and target != self.target_class:
                continue  # one-class mode: only seed rows of target_class
            seed_features = rep.features_of(i)
            if seed_features.size == 0:
                continue  # all-zero row -- no rule can pin it

            if self.preparation is not None:
                prep = self.preparation
            elif holdout:
                rs = None if self.random_state is None else self.random_state + i
                prep = GrowPruneSplit(prune_fraction=self.prune_fraction, random_state=rs)
            else:
                prep = NoSplit()
            search_mask, prep_context = prep.prepare(rep, target, None)
            if search_mask is not None and not search_mask[i]:
                # the seed row must be in the growing set for its stats to
                # count while the rule is grown
                search_mask = search_mask.copy()
                search_mask[i] = True

            initial = SeedExample(strategy="index", index=i).initial_candidates(rep, target)
            rule = search.search(rep, target, self.metric, initial, example_mask=search_mask)
            if rule is None or rule.length() == 0:
                continue
            # LORD's prune: same routine RIPPER uses, but scored on the full
            # training set (`all_rows`) unless a preparation= override gave a
            # held-out mask.
            prune_context = prep_context if prep_context is not None else all_rows
            rule = post.postprocess(rule, rep, target, prune_context)
            if rule.length() == 0:
                continue
            pool.setdefault(rule, rule)
            if covered is not None:
                covered |= rule.covers_data_packed(rep)

        return list(pool.values())

    def _weights_and_pos(self, pool: List[Rule], rep: BooleanDataRepresentation):
        """Returns `(weights, pos)`, plain arrays aligned with `pool` --
        purely internal to `_coverage_filter`'s per-row best-rule choice,
        not stored on any rule (no declarative weight is kept; a kept
        rule's reliability is read from its own measured `stats()`, via
        `annotate_rules` in `_induce`)."""
        weights = np.empty(len(pool))
        pos = np.empty(len(pool), dtype=int)
        for k, r in enumerate(pool):
            st = RuleStats.from_rule(r, rep, r.target)
            weights[k] = self.metric.score(st)
            pos[k] = st.tp
        return weights, pos

    def _coverage_filter(
        self, pool: List[Rule], rep: BooleanDataRepresentation, weights: np.ndarray, pos: np.ndarray,
    ) -> List[Rule]:
        """LORD's `RuleManager` filter: keep a rule only if it's the best
        covering rule (metric, then positives) of its class for at least
        one training row."""
        if not pool:
            return []
        cov = np.stack([r.covers_data_packed(rep) for r in pool])  # (n_pool, n_rows)
        targets = np.array([r.target for r in pool], dtype=object)
        kept: dict = {}
        for e in range(rep.n_samples):
            cand = np.flatnonzero(cov[:, e] & (targets == rep.y[e]))
            if cand.size == 0:
                continue
            best = cand[np.lexsort((pos[cand], weights[cand]))[-1]]  # max weight, tie -> max pos
            kept.setdefault(pool[best], pool[best])
        return list(kept.values())

    # -- fit ----------------------------------------------------------------

    def _induce(self, data: BooleanDataRepresentation) -> List[SingleRule]:
        pool = self._search_pool(data)
        weights, pos = self._weights_and_pos(pool, data)
        kept = self._coverage_filter(pool, data, weights, pos)
        return annotate_rules(kept, data)

    def _default_model(self, data: BooleanDataRepresentation) -> type:
        return ConceptModel if self.target_class is not None else FlatRuleSet

    @produces(FlatRuleSet)
    def _fit_native(self, data: BooleanDataRepresentation, **kw) -> FlatRuleSet:
        """`fit(data, model=FlatRuleSet)` -- native LORD: seed every row
        (or every `target_class` row), a `FlatRuleSet` with `"max"`
        (`HeuristicMaxCombiner`, scored from measured stats -- LORD's own
        `get_best_covering_rule` used its own metric instead), the
        training-majority label as the default."""
        if data.y is None:
            raise ValueError("PyLORD.fit needs data.y")
        model = FlatRuleSet(self._induce(data), default_prediction=MajorityClass(data),
                            combiner="max")
        return annotate_default_rule(model, data)

    # No SingleRule producer: LORD's induced pool comes from every training
    # example essentially in parallel (one seed each), so there's no
    # unforced "first rule" the way a sequential-covering loop has one --
    # only "best of the pool by some score", which bakes in a selection
    # criterion the algorithm itself never makes. See seco.py's SeCo/RIPPER
    # `_fit_one_rule` for the contrast: a genuine single search call, not a
    # pick among already-computed candidates.

    @produces(ConceptModel)
    def _fit_concept(self, data: BooleanDataRepresentation, *,
                     label: Any = None, fallback: Any = None) -> ConceptModel:
        """`fit(data, model=ConceptModel, label="a")` -- LORD seeding
        restricted to one class."""
        if data.y is None:
            raise ValueError("PyLORD needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        lc = copy.copy(self)
        lc.target_class = target
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(lc._induce(data), label=target, default_prediction=default)
        return annotate_default_rule(model, data)

    def _fit_binary(self, data: BooleanDataRepresentation, positive: Any,
                    negative: Any = None) -> ConceptModel:
        return self._fit_concept(data, label=positive, fallback=negative)
