"""
pyrulearn.learners.ids
==========================

`IDS` -- Interpretable Decision Sets (Lakkaraju, Bach & Leskovec, KDD
2016): consume a pool of class association rules (a `FlatRuleSet`, via
`pyrulearn.learners.associative.RuleDistiller` -- either given directly or
mined by default via `ClassAssociationRuleMiner`, the same shared miner
`pyrulearn.learners.associative.CBA` and `CMAR` also
consume from), then select an *unordered* subset of them by approximately
maximizing IDS's own 7-term
submodular objective -- trading off number of rules, rule length,
same-/different-class overlap, class coverage, precision, and recall --
rather than CBA's precedence-sorted prefix or CMAR's significance-
filtered vote. Mirrors how `pyIDS` (github.com/jirifilip/pyIDS, same
author as `pyarc`) builds IDS on top of `pyarc`'s CAR mining.

**Objective** (itemset pool `S`, classes `C`, `N` training rows, `Lmax`
= longest candidate rule): for a kept subset `R`,

    f1 = (|S| - size(R)) / |S|                                    fewer rules
    f2 = (Lmax*|S| - sum_{r in R} length(r)) / (Lmax*|S|)         shorter rules
    f3 = (N*|S|^2 - sum_{i<=j, ci=cj} overlap(ri,rj)) / (N*|S|^2)  low same-class overlap
    f4 = (N*|S|^2 - sum_{i<=j, ci!=cj} overlap(ri,rj)) / (N*|S|^2) low cross-class overlap
    f5 = |{c in C : some kept rule predicts c}| / |C|             every class covered
    f6 = (N*|S| - sum_{r in R} |incorrect-cover(r)|) / (N*|S|)    precision
    f7 = |union of correct-cover(r) for r in R| / N               recall

maximized as `sum_i lambda_i * f_i(R)` over subsets of the candidate
pool. Each term is the paper's own raw reward divided by its own
natural maximum, so every term lives in `[0, 1]` and `lambda=1` for
every term is a meaningful equal-weighting default -- the paper's raw,
undivided terms differ by orders of magnitude in scale (`f3`/`f4` are
`O(N*|S|^2)`, `f5` is `O(|C|)`), which is exactly why the paper fits
lambdas by coordinate ascent rather than shipping default weights at
all; dividing each term by a fixed positive constant (independent of
which `R` is chosen) changes none of its shape -- submodularity,
non-negativity and non-monotonicity all survive positive scaling --
while making the fixed-weights mode usable without tuning first. The
paper proves the (unscaled) sum non-negative, non-monotone, and
submodular, and solves it approximately via Smooth Local Search (their
Algorithm 1); ties among several covering rules of different classes
are broken by the rule with the highest F1 score on training data, and
uncovered points get the majority-class default.

Deliberately not chasing every detail of the original paper -- per the
same "reuse existing structure, slight deviations OK" license CMAR was
built under:

- **Candidate domain** is the (precedence-capped) mined CAR pool, not
  the paper's full `S x C` (every itemset paired with every class):
  avoids materializing itemset/class pairs with ~0 confidence that
  would never survive the objective anyway. `|S|` is taken as the
  number of *distinct itemsets* among the surviving candidates, and
  `Lmax` their max length. Because a kept itemset can in principle
  appear with more than one class (not excluded by construction, just
  discouraged by the objective), `size(R)` can slightly exceed `|S|` --
  `f1`/`f2`/`f3`/`f4`/`f6` are clipped at 0 to keep them the
  non-negative rewards the paper's own construction guarantees, rather
  than letting this edge case produce a silently negative term.
- **Rule-count reduction before optimization** (`rule_cutoff`, mirrors
  pyIDS's own hyperparameter of the same name): reuses
  `pyrulearn.learners.associative.sort_by_measured_precedence` directly
  to cap each class to its top `rule_cutoff` rules before the (otherwise
  `O(|X|^2)` per round) search ever sees them.
- **Multi-cover tie-break** ("highest F1 on training data") is exactly
  `pyrulearn.combiners.HeuristicMaxCombiner` fed
  `pyrulearn.heuristics.FBeta()` -- `beta=1.0` (its default) *is* F1.
  No new heuristic or combiner code at all.
- **Smooth Local Search** (`optimizer="sls"`, the default) estimates
  each marginal contribution by averaging a fixed `sls_samples`
  Monte-Carlo draws instead of the paper's adaptive
  stop-on-standard-error rule, and caps the total number of add/remove
  moves at `sls_max_restarts` instead of relying on its
  only-in-expectation polynomial-time bound. `optimizer="greedy"` (plain
  forward submodular maximization -- no formal guarantee for a
  non-monotone objective) is offered as an explicit, much cheaper
  fallback.
- **Lambda weights**, by default (`tune_lambdas=True`), are fit by a
  simplified coordinate ascent against a held-out `validation_fraction`
  slice (`5%`, matching the paper) -- a small `tune_grid` per weight
  over `tune_passes` sweeps, in place of the paper's own ternary
  search, but *with* the paper's own feasibility constraints enforced
  (`max_rules`/`max_avg_length`/`max_overlap`/`max_uncovered`, plus
  `require_all_classes` -- softened to "every class the candidate pool
  can even reach", see `_feasible`'s docstring, since our
  confidence-gated candidate domain can't always reach the paper's own
  literal "every class in the dataset"). Coordinate ascent always
  searches via `_greedy_select` internally regardless of `self.
  optimizer` -- re-running `"sls"` at every grid point would be far too
  slow -- then the *final* fit still uses whichever `optimizer` was
  actually requested. `tune_lambdas=False` (fixed `lambda_weights`,
  default all `1.0`) is offered as an explicit, much cheaper fallback.
- **`max_len` stays at `4`** (`RuleDistiller`'s own shared default,
  unchanged) rather than the paper's own reported `10` --
  measured directly (`vote` dataset, `min_support=0.01`): mining time
  explodes well before length 10 on datasets with more than a handful
  of features (338,619 CARs / 18s at `max_len=4` vs. 33,784 / 1.2s at
  `max_len=3`, both well short of 10), a limitation of this codebase's
  own pure-Python level-wise Apriori mining (shared with CBA/CMAR), not
  a paper-fidelity choice. `min_support` stays `0.01`, matching the
  paper's own value exactly (already the shared default). Both are
  silently unused if `rules=` is given directly -- see `RuleDistiller`.

Native multi-class already (each rule carries its own class head).
`DecomposingLearner` is mixed in for consistency with every other native
learner here, not because IDS needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .associative import DEFAULT_MAX_AUTO_CONVERT_CELLS, RuleDistiller, sort_by_measured_precedence
from .base import DecomposingLearner, NativeRuleLearner, produces
from ..combiners import HeuristicMaxCombiner
from ..heuristics import FBeta
from ..models import ConceptModel, FlatRuleSet, MajorityClass, RuleView, annotate_default_rule, annotate_rules
from ..rule import Rule


@dataclass(frozen=True)
class _Candidate:
    """One candidate `(itemset, class)` rule, with its coverage cached
    as boolean masks so the objective (evaluated many times per fit)
    never re-walks `data.coverage(rule)`."""

    rule: Rule
    items: Tuple[int, ...]
    target: Any
    mask: np.ndarray
    correct_mask: np.ndarray

    @property
    def length(self) -> int:
        return len(self.items)


def _candidates_from_rules(rules: Sequence[Rule], data: Any, rule_cutoff: int) -> List[_Candidate]:
    """Sort `rules` by measured precedence (confidence/support/
    generality -- cheap, reads each rule's own cached `stats()` only),
    keep only the top `rule_cutoff` per class, then measure each
    survivor's coverage against `data` -- the candidate domain the
    objective actually searches over. `rules` is whatever
    `RuleDistiller._resolve_rules` returned: a freshly-mined default pool,
    or an externally-supplied one (e.g. `rules=` at construction) --
    this function doesn't care which."""
    ordered = sort_by_measured_precedence(rules)
    if isinstance(ordered, RuleView):
        # lazy pool: the cap is a vectorized selection, so only the survivors are ever built
        capped = list(ordered.head_per_target(rule_cutoff))
    else:
        per_class_count: Dict[Any, int] = {}
        capped = []
        for r in ordered:
            n = per_class_count.get(r.target, 0)
            if n >= rule_cutoff:
                continue
            per_class_count[r.target] = n + 1
            capped.append(r)

    y = np.asarray(data.y)
    candidates: List[_Candidate] = []
    for r in capped:
        mask = data.coverage(r)
        correct_mask = mask & (y == r.target)
        items = tuple(sorted(l.feature for l in r.conditions))
        candidates.append(_Candidate(rule=r, items=items, target=r.target, mask=mask, correct_mask=correct_mask))
    return candidates


def _slice_candidates(candidates: Sequence[_Candidate], idx: np.ndarray) -> List[_Candidate]:
    """The same candidates, with their masks restricted to `idx` rows
    -- used to search/evaluate on a train/validation split without
    re-mining or re-measuring coverage from scratch."""
    return [
        _Candidate(rule=c.rule, items=c.items, target=c.target, mask=c.mask[idx], correct_mask=c.correct_mask[idx])
        for c in candidates
    ]


def _distinct_itemset_count(candidates: Sequence[_Candidate]) -> int:
    return len({c.items for c in candidates})


def _max_length(candidates: Sequence[_Candidate]) -> int:
    return max((c.length for c in candidates), default=0)


def _candidate_f1(c: _Candidate, y: np.ndarray) -> float:
    """This candidate's own F1 score against `y` -- IDS's own
    multi-cover tie-break criterion, computed directly from cached
    masks (used only inside the coordinate-ascent tuning loop; the real
    fitted model reuses `pyrulearn.heuristics.FBeta`/
    `pyrulearn.combiners.HeuristicMaxCombiner` instead, see the module
    docstring)."""
    tp = int(np.sum(c.correct_mask))
    fp = int(np.sum(c.mask & ~c.correct_mask))
    fn = int(np.sum((y == c.target) & ~c.mask))
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _majority_label(y: np.ndarray) -> Any:
    vals, counts = np.unique(y, return_counts=True)
    return vals[int(np.argmax(counts))]


def _interpretability_metrics(kept: Sequence[_Candidate], n: int) -> Tuple[float, float, float, int]:
    """`(fraction_overlap, fraction_uncovered, avg_length, num_rules)` --
    the paper's own descriptive interpretability metrics (Section 5.2),
    used by `_feasible` to enforce its coordinate-ascent constraints.
    Distinct from the objective's own `f3`/`f4`: those include each
    rule's self-overlap term by construction (see the module
    docstring); `fraction_overlap` here is the paper's plain pairwise
    average, `2/(|R|(|R|-1)) * sum_{i<j} overlap(ri,rj)/N`.
    """
    num_rules = len(kept)
    if num_rules == 0:
        return 0.0, 1.0, 0.0, 0
    avg_length = sum(c.length for c in kept) / num_rules
    covered = np.zeros(n, dtype=bool)
    for c in kept:
        covered |= c.mask
    fraction_uncovered = 1.0 - float(covered.sum()) / max(1, n)
    if num_rules < 2:
        fraction_overlap = 0.0
    else:
        total_overlap = 0
        for a in range(num_rules):
            for b in range(a + 1, num_rules):
                total_overlap += int(np.sum(kept[a].mask & kept[b].mask))
        num_pairs = num_rules * (num_rules - 1) / 2
        fraction_overlap = (total_overlap / max(1, n)) / num_pairs
    return fraction_overlap, fraction_uncovered, avg_length, num_rules


def _feasible(
    kept: Sequence[_Candidate], n: int, achievable_classes: Set[Any],
    max_overlap: float, max_uncovered: float, max_avg_length: float, max_rules: int,
    require_all_classes: bool,
) -> bool:
    """Whether `kept` satisfies the paper's own coordinate-ascent
    constraints (Section 5.1's Parameter Selection): bounds on rule
    count, average length, overlap, and uncovered fraction, plus
    (optionally) covering every class the candidate pool can even
    reach. That last part softens the paper's own `Fraction Classes =
    1.0` -- our candidate domain is confidence-gated CARs, not the
    paper's full `S x C` (see the module docstring), so a class with no
    CAR clearing `min_confidence` at all can never be covered by any
    weight choice; requiring every *achievable* class instead of every
    class in the dataset keeps the constraint satisfiable.
    """
    if len(kept) > max_rules:
        return False
    fraction_overlap, fraction_uncovered, avg_length, _num_rules = _interpretability_metrics(kept, n)
    if avg_length > max_avg_length or fraction_overlap > max_overlap or fraction_uncovered > max_uncovered:
        return False
    if require_all_classes:
        covered = {c.target for c in kept}
        if not achievable_classes <= covered:
            return False
    return True


def _objective(
    kept: Iterable[int], candidates: Sequence[_Candidate], classes: Sequence[Any],
    n: int, s_size: int, lmax: int, weights: Sequence[float],
) -> float:
    """The 7-term weighted objective, evaluated for the subset `kept`
    (indices into `candidates`). See the module docstring for the exact
    per-term formulas, the `[0, 1]` normalization, and why the
    numerators are clipped at 0.
    """
    kept_list = list(kept)
    size_r = len(kept_list)
    s_size_safe = max(1, s_size)
    n_safe = max(1, n)

    f1 = max(0.0, s_size - size_r) / s_size_safe
    f2 = max(0.0, lmax * s_size - sum(candidates[i].length for i in kept_list)) / max(1, lmax * s_size)

    same_overlap = 0
    cross_overlap = 0
    for a in range(len(kept_list)):
        i = kept_list[a]
        same_overlap += int(candidates[i].mask.sum())  # the i==j term: a rule's own coverage
        for b in range(a + 1, len(kept_list)):
            j = kept_list[b]
            overlap = int(np.sum(candidates[i].mask & candidates[j].mask))
            if candidates[i].target == candidates[j].target:
                same_overlap += overlap
            else:
                cross_overlap += overlap
    overlap_scale = max(1, n * s_size ** 2)
    f3 = max(0.0, n * s_size ** 2 - same_overlap) / overlap_scale
    f4 = max(0.0, n * s_size ** 2 - cross_overlap) / overlap_scale

    covered_classes = {candidates[i].target for i in kept_list}
    f5 = sum(1 for c in classes if c in covered_classes) / max(1, len(classes))

    incorrect_total = sum(int(np.sum(candidates[i].mask & ~candidates[i].correct_mask)) for i in kept_list)
    f6 = max(0.0, n * s_size - incorrect_total) / max(1, n * s_size)

    if kept_list:
        correct_union = np.zeros(n, dtype=bool)
        for i in kept_list:
            correct_union |= candidates[i].correct_mask
        f7 = float(correct_union.sum()) / n_safe
    else:
        f7 = 0.0

    terms = (f1, f2, f3, f4, f5, f6, f7)
    return sum(w * f for w, f in zip(weights, terms))


def _greedy_select(
    candidates: Sequence[_Candidate], classes: Sequence[Any], n: int, s_size: int, lmax: int,
    weights: Sequence[float],
) -> Set[int]:
    """Plain forward submodular maximization: repeatedly add whichever
    remaining candidate gives the largest objective gain, stop once
    nothing improves it. Deterministic, no sampling -- the default
    optimizer (fast, no formal guarantee for a non-monotone objective).
    """
    kept: Set[int] = set()
    current = _objective(kept, candidates, classes, n, s_size, lmax, weights)
    remaining = set(range(len(candidates)))
    while remaining:
        best_idx: Optional[int] = None
        best_gain = 0.0
        best_value = current
        for idx in remaining:
            value = _objective(kept | {idx}, candidates, classes, n, s_size, lmax, weights)
            gain = value - current
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_value = value
        if best_idx is None:
            break
        kept.add(best_idx)
        remaining.discard(best_idx)
        current = best_value
    return kept


def _phi_sample(n_candidates: int, base: FrozenSet[int], delta: float, rng: np.random.Generator) -> FrozenSet[int]:
    """`Phi_X(base, delta)`: each candidate independently sampled with
    probability `(1+delta)/2` if it's in `base`, else `(1-delta)/2`."""
    p_in = (1 + delta) / 2
    p_out = (1 - delta) / 2
    draws = rng.random(n_candidates)
    sample = set()
    for idx in range(n_candidates):
        p = p_in if idx in base else p_out
        if draws[idx] < p:
            sample.add(idx)
    return frozenset(sample)


def _estimate_marginal(
    idx: int, base: FrozenSet[int], candidates: Sequence[_Candidate], classes: Sequence[Any],
    n: int, s_size: int, lmax: int, weights: Sequence[float], delta: float, samples: int,
    rng: np.random.Generator,
) -> float:
    """Monte-Carlo estimate of `E[f(Phi(base,delta) u {idx})] -
    E[f(Phi(base,delta) \\ {idx})]`, over `samples` draws -- each draw
    reused for both terms (a paired estimator) to cut variance, in
    place of the paper's adaptive stop-on-standard-error rule."""
    total = 0.0
    n_candidates = len(candidates)
    for _ in range(samples):
        sample = _phi_sample(n_candidates, base, delta, rng)
        with_idx = sample | {idx}
        without_idx = sample - {idx}
        total += (
            _objective(with_idx, candidates, classes, n, s_size, lmax, weights)
            - _objective(without_idx, candidates, classes, n, s_size, lmax, weights)
        )
    return total / samples if samples > 0 else 0.0


def _sls_select(
    candidates: Sequence[_Candidate], classes: Sequence[Any], n: int, s_size: int, lmax: int,
    weights: Sequence[float], delta: float, delta_final: float, samples: int, final_samples: int,
    max_restarts: int, rng: np.random.Generator,
) -> Tuple[FrozenSet[int], float]:
    """Smooth Local Search (Algorithm 1): repeatedly add/remove whichever
    candidate's estimated marginal contribution crosses `+-2/|X|^2 *
    OPT`, restarting the scan after every change, capped at
    `max_restarts` total moves (the paper's polynomial-time bound only
    holds in expectation; a finite run needs a hard stop). Returns the
    best-scoring subset found among `final_samples` draws of the final
    `Phi_X(A, delta_final)`, together with its true objective value.
    """
    m = len(candidates)
    if m == 0:
        return frozenset(), 0.0

    opt_sample = _phi_sample(m, frozenset(range(m)), 0.0, rng)
    opt = _objective(opt_sample, candidates, classes, n, s_size, lmax, weights)
    opt = max(opt, 1e-9)
    threshold = 2.0 / (m ** 2) * opt

    active: Set[int] = set()
    restarts = 0
    changed = True
    while changed and restarts < max_restarts:
        changed = False
        for idx in range(m):
            if idx in active:
                continue
            gain = _estimate_marginal(idx, frozenset(active), candidates, classes, n, s_size, lmax,
                                      weights, delta, samples, rng)
            if gain > threshold:
                active.add(idx)
                changed = True
                restarts += 1
                break
        if changed:
            continue
        for idx in list(active):
            gain = _estimate_marginal(idx, frozenset(active), candidates, classes, n, s_size, lmax,
                                      weights, delta, samples, rng)
            if gain < -threshold:
                active.discard(idx)
                changed = True
                restarts += 1
                break

    best_subset = frozenset(active)
    best_score = _objective(best_subset, candidates, classes, n, s_size, lmax, weights)
    for _ in range(final_samples):
        sample = _phi_sample(m, frozenset(active), delta_final, rng)
        score = _objective(sample, candidates, classes, n, s_size, lmax, weights)
        if score > best_score:
            best_score = score
            best_subset = sample
    return best_subset, best_score


class IDS(RuleDistiller, DecomposingLearner, NativeRuleLearner):
    """Interpretable Decision Sets (Lakkaraju, Bach & Leskovec, 2016).
    See the module docstring for the objective, the optimizer options,
    and exactly which pieces are reused as-is vs. deliberately
    simplified relative to the paper/`pyIDS`.

    Defaults throughout aim to match the paper's own reported settings
    (Section 5.1) as closely as this implementation's structure allows;
    `optimizer="greedy"`/`tune_lambdas=False` are offered as explicit,
    much cheaper fallbacks, not the default.

    - `max_len=4` (this codebase's own shared mining-cost limit, not the
      paper's `10` -- see the module docstring), `min_support=0.01`
      (the paper's own value).
    - `rule_cutoff` -- keep only the top `rule_cutoff` CARs per class
      (by CBA's own precedence order) as the search's candidate pool
      (a pyIDS-style knob with no direct paper equivalent, since the
      paper's own candidate domain isn't confidence-gated CARs -- see
      the module docstring).
    - `lambda_weights` -- the 7 objective weights; only used directly
      when `tune_lambdas=False`.
    - `optimizer` -- `"sls"` (default, Smooth Local Search, matches the
      paper/`pyIDS`) or `"greedy"` (fast fallback, no formal guarantee
      for a non-monotone objective).
    - `sls_samples`/`sls_final_samples`/`sls_max_restarts` -- only used
      when `optimizer="sls"`; see `_sls_select`.
    - `tune_lambdas` -- default `True`: fit a simplified coordinate
      ascent over `lambda_weights` on a held-out `validation_fraction`
      slice (`tune_passes` sweeps over `tune_grid` candidate values per
      weight), subject to `max_rules`/`max_avg_length`/`max_overlap`/
      `max_uncovered`/`require_all_classes` (the paper's own Parameter
      Selection constraints, defaults matching it exactly) -- see
      `_resolve_lambdas`/`_feasible`. `False` uses `lambda_weights`
      directly (fast fallback).
    - `validation_fraction=0.05` -- the paper's own held-out fraction.
    - `seed` -- seeds the `numpy.random.Generator` used by `"sls"` and
      by the train/validation split in `tune_lambdas`.
    - The rest (`rules`/`min_confidence`/`target_class`/
      `max_auto_convert_cells`) are `RuleDistiller`'s, unchanged.
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        min_support: float = 0.01,
        min_confidence: float = 0.5,
        max_len: int = 4,
        rule_cutoff: int = 50,
        lambda_weights: Tuple[float, ...] = (1.0,) * 7,
        optimizer: str = "sls",
        sls_samples: int = 10,
        sls_max_restarts: int = 50,
        sls_final_samples: int = 10,
        tune_lambdas: bool = True,
        validation_fraction: float = 0.05,
        tune_passes: int = 2,
        tune_grid: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0),
        max_overlap: float = 0.10,
        max_uncovered: float = 0.15,
        max_avg_length: float = 10.0,
        max_rules: int = 15,
        require_all_classes: bool = True,
        seed: Optional[int] = None,
        target_class: Optional[Any] = None,
        max_auto_convert_cells: int = DEFAULT_MAX_AUTO_CONVERT_CELLS,
    ):
        super().__init__(rules, min_support, min_confidence, max_len, max_auto_convert_cells, target_class)
        if optimizer not in ("greedy", "sls"):
            raise ValueError(f"optimizer must be 'greedy' or 'sls', got {optimizer!r}")
        if len(lambda_weights) != 7:
            raise ValueError(f"lambda_weights needs 7 values, got {len(lambda_weights)}")
        self.rule_cutoff = rule_cutoff
        self.lambda_weights = tuple(lambda_weights)
        self.optimizer = optimizer
        self.sls_samples = sls_samples
        self.sls_max_restarts = sls_max_restarts
        self.sls_final_samples = sls_final_samples
        self.tune_lambdas = tune_lambdas
        self.validation_fraction = validation_fraction
        self.tune_passes = tune_passes
        self.tune_grid = tune_grid
        self.max_overlap = max_overlap
        self.max_uncovered = max_uncovered
        self.max_avg_length = max_avg_length
        self.max_rules = max_rules
        self.require_all_classes = require_all_classes
        self.seed = seed

    def _default_model(self, data: Any) -> type:
        return ConceptModel if self.target_class is not None else FlatRuleSet

    def _build(self, data: Any, only_class: Optional[Any] = None) -> List[Rule]:
        pool = self._resolve_rules(data, only_class=only_class)
        candidates = _candidates_from_rules(pool, data, self.rule_cutoff)
        y = np.asarray(data.y)
        classes = [only_class] if only_class is not None else list(np.unique(y))
        n = data.n_samples
        s_size = _distinct_itemset_count(candidates)
        lmax = _max_length(candidates)
        rng = np.random.default_rng(self.seed)

        if self.tune_lambdas:
            weights = self._resolve_lambdas(candidates, classes, n, s_size, lmax, y, rng)
        else:
            weights = self.lambda_weights
        kept_idx = self._optimize(candidates, classes, n, s_size, lmax, weights, rng)
        return [candidates[i].rule for i in kept_idx]

    def _optimize(
        self, candidates: Sequence[_Candidate], classes: Sequence[Any], n: int, s_size: int, lmax: int,
        weights: Sequence[float], rng: np.random.Generator,
    ) -> Set[int]:
        if not candidates:
            return set()
        if self.optimizer == "greedy":
            return _greedy_select(candidates, classes, n, s_size, lmax, weights)
        subset1, score1 = _sls_select(candidates, classes, n, s_size, lmax, weights,
                                      delta=1 / 3, delta_final=1 / 3, samples=self.sls_samples,
                                      final_samples=self.sls_final_samples,
                                      max_restarts=self.sls_max_restarts, rng=rng)
        subset2, score2 = _sls_select(candidates, classes, n, s_size, lmax, weights,
                                      delta=1 / 3, delta_final=-1.0, samples=self.sls_samples,
                                      final_samples=self.sls_final_samples,
                                      max_restarts=self.sls_max_restarts, rng=rng)
        return set(subset1) if score1 >= score2 else set(subset2)

    def _score_lambda_candidate(
        self, weights: Sequence[float], candidates: Sequence[_Candidate], classes: Sequence[Any],
        train_idx: np.ndarray, val_idx: np.ndarray, y: np.ndarray, s_size: int, lmax: int,
        achievable_classes: Set[Any],
    ) -> Optional[float]:
        """Held-out accuracy of the rule set `weights` would select,
        trained on `train_idx` and scored on `val_idx` -- the inner
        scoring function coordinate ascent maximizes -- or `None` if
        that rule set violates `_feasible`'s constraints (the paper's
        own Parameter Selection bounds). Always searches via the fast
        `_greedy_select` proxy here regardless of `self.optimizer`:
        re-running full `"sls"` at every one of `tune_passes * 7 *
        len(tune_grid)` grid points would be prohibitively slow, and
        greedy's ranking of candidate weight vectors is a reasonable
        stand-in for this purpose -- the *final* fitted model still
        uses whichever `self.optimizer` was actually requested. Ties
        among multiple covering rules of different classes use each
        kept rule's own F1 on `y[train_idx]` (`_candidate_f1`), the
        same criterion the real fitted model's
        `HeuristicMaxCombiner(FBeta())` applies, computed directly here
        to avoid rebuilding a full `FlatRuleSet` for every trial
        weight."""
        train_c = _slice_candidates(candidates, train_idx)
        val_c = _slice_candidates(candidates, val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]
        default = _majority_label(y_train)
        if len(val_idx) == 0:
            return None

        kept_idx = _greedy_select(train_c, classes, len(train_idx), s_size, lmax, weights)
        kept_train = [train_c[i] for i in kept_idx]
        if not _feasible(kept_train, len(train_idx), achievable_classes,
                         self.max_overlap, self.max_uncovered, self.max_avg_length, self.max_rules,
                         self.require_all_classes):
            return None
        if not kept_idx:
            return float(np.mean(y_val == default))

        kept_val = [val_c[i] for i in kept_idx]
        preds = []
        for p in range(len(val_idx)):
            covering = [k for k in range(len(kept_val)) if kept_val[k].mask[p]]
            if not covering:
                preds.append(default)
                continue
            targets = {kept_val[k].target for k in covering}
            if len(targets) == 1:
                preds.append(kept_val[covering[0]].target)
                continue
            best_k = max(covering, key=lambda k: _candidate_f1(kept_train[k], y_train))
            preds.append(kept_val[best_k].target)
        preds_arr = np.array(preds, dtype=object)
        return float(np.mean(preds_arr == y_val))

    def _resolve_lambdas(
        self, candidates: Sequence[_Candidate], classes: Sequence[Any], n: int, s_size: int, lmax: int,
        y: np.ndarray, rng: np.random.Generator,
    ) -> Tuple[float, ...]:
        """Simplified coordinate ascent, matching the paper's own
        Parameter Selection (Section 5.1) as closely as this
        implementation's own "reuse existing structure" license allows:
        `tune_passes` sweeps over the 7 weights, each tried against
        every value in `tune_grid` (others held fixed, in place of the
        paper's ternary search), scored by held-out accuracy on a
        `validation_fraction` random split of the training rows
        (`_score_lambda_candidate`), among weight choices whose rule set
        satisfies `_feasible`'s bounds (`max_rules`/`max_avg_length`/
        `max_overlap`/`max_uncovered`/`require_all_classes` -- the
        paper's own constraints). If no candidate value is feasible for
        a dimension (including its current value), that dimension is
        left unchanged rather than forced into an infeasible choice.
        """
        if not candidates:
            return self.lambda_weights
        n_val = max(1, int(round(self.validation_fraction * n)))
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        if len(train_idx) == 0 or len(val_idx) == 0:
            return self.lambda_weights
        achievable_classes = {c.target for c in candidates}

        weights = list(self.lambda_weights)
        for _ in range(self.tune_passes):
            for dim in range(7):
                best_value = weights[dim]
                best_score = self._score_lambda_candidate(weights, candidates, classes, train_idx, val_idx,
                                                           y, s_size, lmax, achievable_classes)
                for candidate_value in self.tune_grid:
                    trial = list(weights)
                    trial[dim] = candidate_value
                    score = self._score_lambda_candidate(trial, candidates, classes, train_idx, val_idx,
                                                          y, s_size, lmax, achievable_classes)
                    if score is not None and (best_score is None or score > best_score):
                        best_score = score
                        best_value = candidate_value
                weights[dim] = best_value
        return tuple(weights)

    @produces(FlatRuleSet)
    def _fit_native(self, data: Any, **kw) -> FlatRuleSet:
        if data.y is None:
            raise ValueError("IDS.fit needs data.y")
        rules = self._build(data)
        model = FlatRuleSet(annotate_rules(rules, data), default_prediction=MajorityClass(data),
                            combiner=HeuristicMaxCombiner(FBeta()))
        return annotate_default_rule(model, data)

    @produces(ConceptModel)
    def _fit_concept(self, data: Any, *, label: Any = None, fallback: Any = None) -> ConceptModel:
        if data.y is None:
            raise ValueError("IDS needs data.y")
        target = label if label is not None else self.target_class
        if target is None:
            raise ValueError("model=ConceptModel needs label= (or target_class set)")
        rules = self._build(data, only_class=target)
        default = fallback if fallback is not None else MajorityClass(data)
        model = ConceptModel(annotate_rules(rules, data), label=target, default_prediction=default)
        return annotate_default_rule(model, data)
