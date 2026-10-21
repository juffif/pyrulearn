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
- `EqualFrequencyDiscretizer` -- unsupervised equal-frequency
  ("quantile", "k-bins") binning, native.
- `BoundaryCorrection` -- wraps another discretizer and moves each of
  its thresholds to the nearest *boundary point*: a gap between two
  adjacent observed values where the class changes, i.e. not between two
  values whose examples all have one and the same class (Fayyad & Irani,
  1993). A threshold inside such a single-class stretch is never the
  best place for it, for any rule heuristic that grows with the covered
  positives and falls with the covered negatives -- moving it to either
  end of the stretch is at least as good, also inside the subset a
  partially grown rule covers. E.g.
  ``BoundaryCorrection(EqualFrequencyDiscretizer(8))``.

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
    none. Every split point of such a tree lies on a boundary point, so
    there is nothing to correct."""

    supervised = True

    def __init__(self, max_intervals: int = DEFAULT_MAX_INTERVALS):
        if max_intervals < 2:
            raise ValueError(f"max_intervals must be at least 2, got {max_intervals}")
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
    -- and, given labels, `boundaries`: the gaps where the class changes."""

    def __init__(self, values: np.ndarray, y: Optional[np.ndarray] = None):
        values = np.asarray(values, dtype=float).ravel()
        observed = ~np.isnan(values)
        values = values[observed]
        order = np.argsort(values, kind="stable")
        self.values = values[order]
        self.distinct, starts, counts = np.unique(self.values, return_index=True, return_counts=True)
        self.below = np.cumsum(counts)[:-1]
        self.boundaries = None
        if y is not None and len(self.distinct) >= 2:
            _, codes = np.unique(np.asarray(y)[observed][order], return_inverse=True)
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
    nearest boundary point not yet taken -- a gap between two adjacent
    observed values where the class changes (see the module docstring)
    -- measured in examples: a threshold with ``m`` examples below it
    goes to the boundary whose number of examples below is closest to
    ``m``. A threshold already on a boundary stays (up to
    prettification). Two thresholds never share one boundary; with fewer
    boundaries than thresholds, the column gets one threshold per
    boundary. Missing values are ignored."""

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
        positions = [int(np.searchsorted(column.values, t, side="left")) for t in sorted(proposed)]
        return column.thresholds(column.nearest_gaps(positions, column.boundaries))
