"""
pyrulearn.data.discretization
=============================

Discretizers: objects that turn a numeric column's values into ascending
thresholds, each threshold ``t`` becoming the two features ``x>=t`` and
``x<t`` (`DataSpecBuilder.add_numeric`). `build_dataspec`, `read_csv`,
`read_arff` and `run_cv` take one as ``discretizer=`` (default
`TreeDiscretizer()`).

- `TreeDiscretizer` -- supervised: the split points of a single-feature
  decision tree (`pyrulearn.interfaces.sklearn.tree_thresholds`).
- `FusinterDiscretizer` -- supervised, bottom-up: FUSINTER (Zighed,
  Rabaseda & Rakotomalala, 1998), merging class-homogeneous intervals
  while a regularized quadratic entropy improves. Native.
- `EqualFrequencyDiscretizer` -- unsupervised equal-frequency
  ("quantile", "k-bins") binning, native.
- `EqualWidthDiscretizer` -- unsupervised equal-width binning, native.
- `BoundaryCorrection` -- wraps another discretizer and moves each of
  its thresholds to the nearest *boundary point* (thresholds landing on
  the same one merge): a gap between two
  adjacent observed values where the class changes, i.e. not between two
  values whose examples all have one and the same class (Fayyad & Irani,
  1993). A threshold inside such a single-class stretch is never the
  best place for it, for any rule heuristic that grows with the covered
  positives and falls with the covered negatives -- moving it to either
  end of the stretch is at least as good, also inside the subset a
  partially grown rule covers. E.g.
  ``BoundaryCorrection(EqualFrequencyDiscretizer(8))``.
- `PerClass` -- wraps a supervised discretizer and runs it once per
  class, on that class against all others, keeping the union of the
  thresholds: each class gets cuts placed for it, not only for the
  overall class mix. E.g. ``PerClass(TreeDiscretizer(3))``.

This module needs only numpy; `TreeDiscretizer` imports scikit-learn
when it is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

#: default cap on the number of intervals a numeric column is split into
#: (``max_intervals - 1`` thresholds at most)
DEFAULT_MAX_INTERVALS = 8


def prettify_threshold(t: float, sorted_unique: np.ndarray) -> float:
    """Replace a computed split point with the coarsest (fewest decimal
    digits) number that still falls in the same open gap between two
    *actually observed* values of the column -- so the resulting
    ``>=``/``<`` test partitions the real data identically, just without
    ugly floating-point artifacts. A threshold is usually the exact
    midpoint of two adjacent values, e.g.
    ``(15.1458 + 15.2) / 2 == 15.172900199890137`` -- a value nobody
    would write down by hand, and no more "correct" than ``15.17`` for
    the same purpose: any number strictly between the two nearest real
    values on *this* column produces the exact same split of the full
    dataset, so rounding here changes nothing but the display.
    """
    below = sorted_unique[sorted_unique < t]
    above = sorted_unique[sorted_unique > t]
    lo = below[-1] if len(below) else -np.inf
    hi = above[0] if len(above) else np.inf
    for decimals in range(0, 10):
        candidate = round(t, decimals)
        if lo < candidate < hi:
            return candidate
    return t  # every rounding collided with a real value -- keep the exact one


class Discretizer(ABC):
    """A numeric column's values -> ascending thresholds.

    `thresholds(values, y)` gets the column's raw values (missing values
    as NaN) and, for a supervised discretizer, the class labels of the
    same rows; it returns ascending thresholds, possibly none (a column
    with no useful split). `supervised` says whether `y` is needed.
    `params()` gives the constructor arguments, for `repr`, equality and
    `run_cv`'s cache key."""

    #: whether `thresholds` needs the class labels
    supervised: bool = True

    @abstractmethod
    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        ...

    def params(self) -> Dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params().items())
        return f"{type(self).__name__}({args})"

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.params() == other.params()

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.params().items()))))


class TreeDiscretizer(Discretizer):
    """Supervised: the internal split points of a single-feature decision
    tree fit against the class (`pyrulearn.interfaces.sklearn.
    tree_thresholds`, scikit-learn's `DecisionTreeClassifier` with
    ``max_leaf_nodes=max_intervals``), prettified. `max_intervals` is a
    ceiling, not a target: the tree's own criterion decides how much of
    it the data justifies, so a column may get fewer thresholds, or
    none. ``max_intervals=None`` grows the full tree, until every
    interval holds one class or one value -- a threshold at every
    boundary point (see below). Every split point of such a tree lies on
    a boundary point, so there is nothing to correct."""

    supervised = True

    def __init__(self, max_intervals: Optional[int] = DEFAULT_MAX_INTERVALS):
        if max_intervals is not None and max_intervals < 2:
            raise ValueError(f"max_intervals must be at least 2 (or None), got {max_intervals}")
        self.max_intervals = max_intervals

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        if y is None:
            raise ValueError("TreeDiscretizer is supervised -- it needs the class labels y")
        from ..interfaces.sklearn import tree_thresholds
        return tree_thresholds(np.asarray(values, dtype=float), y, max_intervals=self.max_intervals)

    def params(self) -> Dict[str, Any]:
        return {"max_intervals": self.max_intervals}


class _Column:
    """A column's non-missing values sorted, as the gaps between adjacent
    distinct values: `distinct` (ascending), `below[i]` -- the number of
    examples below gap i (between ``distinct[i]`` and ``distinct[i + 1]``)
    -- and, given labels, `boundaries`: the gaps where the class changes,
    and `class_counts`: per distinct value, its examples per class."""

    def __init__(self, values: np.ndarray, y: Optional[np.ndarray] = None):
        values = np.asarray(values, dtype=float).ravel()
        observed = ~np.isnan(values)
        values = values[observed]
        order = np.argsort(values, kind="stable")
        self.values = values[order]
        self.distinct, starts, counts = np.unique(self.values, return_index=True, return_counts=True)
        self.below = np.cumsum(counts)[:-1]
        self.boundaries = None
        self.class_counts = None
        if y is not None and len(self.distinct) >= 2:
            _, codes = np.unique(np.asarray(y)[observed][order], return_inverse=True)
            self.class_counts = np.zeros((len(self.distinct), codes.max() + 1))
            np.add.at(self.class_counts, (np.repeat(np.arange(len(self.distinct)), counts), codes), 1)
            lowest = np.minimum.reduceat(codes, starts)
            highest = np.maximum.reduceat(codes, starts)
            pure = lowest == highest  # every example with this value has the same class
            self.boundaries = np.flatnonzero(~(pure[:-1] & pure[1:] & (lowest[:-1] == lowest[1:])))

    def nearest_gaps(self, positions, candidates: np.ndarray) -> List[int]:
        """For each position (in examples, in order), the candidate gap
        nearest to it, measured in examples, that isn't taken yet (ties:
        the lower gap); fewer gaps once the candidates run out."""
        free = np.ones(len(candidates), dtype=bool)
        chosen = []
        for position in positions:
            if not free.any():
                break
            distance = np.where(free, np.abs(self.below[candidates] - position), np.inf)
            best = int(np.argmin(distance))
            free[best] = False
            chosen.append(int(candidates[best]))
        return chosen

    def thresholds(self, gaps) -> List[float]:
        """The prettified midpoints of `gaps`, ascending."""
        return sorted(float(prettify_threshold(float(self.distinct[i] + self.distinct[i + 1]) / 2, self.distinct))
                      for i in gaps)


class EqualFrequencyDiscretizer(Discretizer):
    """Unsupervised equal-frequency binning: up to ``max_intervals - 1``
    cuts placed so that each interval holds about the same number of
    (non-missing) examples. Native; missing values are ignored.

    The cuts are gaps between adjacent observed values: for each of the
    ``max_intervals - 1`` equal-frequency positions (in examples,
    ``k * n / max_intervals``) the nearest gap not yet taken, measured
    in examples. Each threshold is the prettified midpoint of its gap
    (`prettify_threshold`). A column with fewer distinct values than
    intervals gets a threshold in every gap. Wrap it in
    `BoundaryCorrection` to move the cuts to where the class changes."""

    supervised = False

    def __init__(self, max_intervals: int = DEFAULT_MAX_INTERVALS):
        if max_intervals < 2:
            raise ValueError(f"max_intervals must be at least 2, got {max_intervals}")
        self.max_intervals = max_intervals

    def params(self) -> Dict[str, Any]:
        return {"max_intervals": self.max_intervals}

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        column = _Column(values)
        if len(column.distinct) < 2:
            return []
        n = len(column.values)
        positions = [k * n / self.max_intervals for k in range(1, self.max_intervals)]
        return column.thresholds(column.nearest_gaps(positions, np.arange(len(column.below))))


class BoundaryCorrection(Discretizer):
    """Supervised wrapper: `discretizer`'s thresholds, each moved to the
    nearest boundary point -- a gap between two adjacent observed values
    where the class changes (see the module docstring) -- measured in
    examples: a threshold with ``m`` examples below it goes to the
    boundary whose number of examples below is closest to ``m`` (ties:
    the lower one). A threshold already on a boundary stays (up to
    prettification). Thresholds that land on the same boundary become
    one, so the column may get fewer thresholds than `discretizer` gave
    it -- where few class changes lie close together, which is where a
    second cut adds least. Missing values are ignored."""

    supervised = True

    def __init__(self, discretizer: Discretizer):
        if not isinstance(discretizer, Discretizer):
            raise TypeError(f"BoundaryCorrection wraps a Discretizer, got {discretizer!r}")
        self.discretizer = discretizer

    def params(self) -> Dict[str, Any]:
        return {"discretizer": self.discretizer}

    def __repr__(self) -> str:
        return f"BoundaryCorrection({self.discretizer!r})"

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        if y is None:
            raise ValueError("BoundaryCorrection is supervised -- it needs the class labels y")
        column = _Column(values, y)
        if column.boundaries is None or len(column.boundaries) == 0:
            return []
        proposed = self.discretizer.thresholds(values, y if self.discretizer.supervised else None)
        positions = np.searchsorted(column.values, np.asarray(proposed, dtype=float), side="left")
        below = column.below[column.boundaries]
        nearest = np.abs(below[None, :] - positions[:, None]).argmin(axis=1)  # ties: the lower boundary
        return column.thresholds(sorted({int(column.boundaries[i]) for i in nearest}))


class EqualWidthDiscretizer(Discretizer):
    """Unsupervised equal-width binning: the column's observed range split
    into `max_intervals` intervals of equal width, each cut (``min + k *
    width``) placed in the gap between observed values it falls into, as
    the prettified midpoint of that gap. Intervals that hold no example
    merge with their neighbour, so a skewed column (or one with an
    outlier) gets fewer thresholds. Native; missing values are ignored."""

    supervised = False

    def __init__(self, max_intervals: int = DEFAULT_MAX_INTERVALS):
        if max_intervals < 2:
            raise ValueError(f"max_intervals must be at least 2, got {max_intervals}")
        self.max_intervals = max_intervals

    def params(self) -> Dict[str, Any]:
        return {"max_intervals": self.max_intervals}

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        column = _Column(values)
        distinct = column.distinct
        if len(distinct) < 2:
            return []
        lo, hi = distinct[0], distinct[-1]
        cuts = lo + (hi - lo) * np.arange(1, self.max_intervals) / self.max_intervals
        # the gap with distinct[i] < cut <= distinct[i + 1]
        gaps = np.searchsorted(distinct, cuts, side="left") - 1
        return column.thresholds(sorted({int(i) for i in gaps if 0 <= i < len(distinct) - 1}))


class FusinterDiscretizer(Discretizer):
    """FUSINTER (Zighed, Rabaseda & Rakotomalala, 1998): supervised,
    bottom-up. Starts from the class-homogeneous intervals -- one per
    stretch between two boundary points (see the module docstring) --
    and repeatedly merges the two adjacent intervals whose merge lowers
    the criterion most, until no merge lowers it. The criterion sums,
    over the intervals ``j``,

        alpha * n_j / n * sum_i q_ij * (1 - q_ij)  +  (1 - alpha) * m * lam / n_j,
        q_ij = (n_ij + lam) / (n_j + m * lam),

    with ``n_ij`` the examples of class ``i`` in interval ``j``, ``n_j``
    the interval's size, ``n`` the column's and ``m`` the number of
    classes: a quadratic entropy, smoothed by `lam`, plus a penalty on
    small intervals, weighted by ``1 - alpha``. The defaults are the
    authors' recommendation. FUSINTER decides the number of intervals
    itself; `max_intervals` (default none) caps it, by going on merging
    the least harmful pair. Native; missing values are ignored."""

    supervised = True

    def __init__(self, alpha: float = 0.975, lam: float = 1.0, max_intervals: Optional[int] = None):
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if lam <= 0:
            raise ValueError(f"lam must be positive, got {lam}")
        if max_intervals is not None and max_intervals < 2:
            raise ValueError(f"max_intervals must be at least 2 (or None), got {max_intervals}")
        self.alpha = alpha
        self.lam = lam
        self.max_intervals = max_intervals

    def params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "lam": self.lam, "max_intervals": self.max_intervals}

    def _terms(self, counts: np.ndarray, n: int) -> np.ndarray:
        """Each interval's share of the criterion; `counts` is intervals x classes."""
        m = counts.shape[-1]
        size = counts.sum(axis=-1)
        q = (counts + self.lam) / (size + m * self.lam)[..., None]
        return (self.alpha * size / n * (q * (1 - q)).sum(axis=-1)
                + (1 - self.alpha) * m * self.lam / size)

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        if y is None:
            raise ValueError("FusinterDiscretizer is supervised -- it needs the class labels y")
        column = _Column(values, y)
        if column.boundaries is None or len(column.boundaries) == 0:
            return []
        cuts = list(column.boundaries)
        starts = np.concatenate([[0], column.boundaries + 1])
        counts = np.add.reduceat(column.class_counts, starts, axis=0)  # the class-homogeneous intervals
        n = len(column.values)
        terms = self._terms(counts, n)
        while len(cuts) > 0:
            merged = counts[:-1] + counts[1:]
            merged_terms = self._terms(merged, n)
            gain = terms[:-1] + terms[1:] - merged_terms  # how much each merge lowers the criterion
            best = int(np.argmax(gain))
            over_cap = self.max_intervals is not None and len(cuts) + 1 > self.max_intervals
            if gain[best] <= 1e-12 and not over_cap:
                break
            counts = np.concatenate([counts[:best], merged[best:best + 1], counts[best + 2:]])
            terms = np.concatenate([terms[:best], merged_terms[best:best + 1], terms[best + 2:]])
            del cuts[best]
        return column.thresholds(cuts)


class PerClass(Discretizer):
    """Supervised wrapper: `discretizer` run once per class ``c``, on the
    two-class problem ``c`` against all other classes, keeping the union
    of the thresholds, sorted and without duplicates: thresholds of
    different classes in the same gap between two observed values (e.g.
    1.4 and 2 between the values 1 and 3) split the data the same way,
    so only the first of them is kept. A capped discretizer then
    places its cuts for every class on its own -- with many classes and
    few intervals, a global discretization may place none for a small
    class -- at the cost of more thresholds: up to ``m`` times as many
    for ``m`` classes, e.g. ``PerClass(TreeDiscretizer(3))`` gives up to
    two per class. With two classes it is `discretizer` itself (both
    one-against-the-rest problems are the same). Wrapping an
    unsupervised discretizer changes nothing -- it gives the same
    thresholds for every class."""

    supervised = True

    def __init__(self, discretizer: Discretizer):
        if not isinstance(discretizer, Discretizer):
            raise TypeError(f"PerClass wraps a Discretizer, got {discretizer!r}")
        self.discretizer = discretizer

    def params(self) -> Dict[str, Any]:
        return {"discretizer": self.discretizer}

    def __repr__(self) -> str:
        return f"PerClass({self.discretizer!r})"

    def thresholds(self, values: np.ndarray, y: Optional[np.ndarray] = None) -> List[float]:
        if y is None:
            raise ValueError("PerClass is supervised -- it needs the class labels y")
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) <= 2:
            return list(self.discretizer.thresholds(values, y))
        union = set()
        for c in classes:
            union.update(self.discretizer.thresholds(values, (y == c).astype(int)))
        # one threshold per gap between observed values: the gap is the
        # number of distinct values below the threshold
        distinct = _Column(values).distinct
        kept: Dict[int, float] = {}
        for t in sorted(union):
            kept.setdefault(int(np.searchsorted(distinct, t, side="left")), t)
        return sorted(kept.values())
