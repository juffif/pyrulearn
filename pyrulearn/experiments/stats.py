"""
pyrulearn.experiments.stats
=============================

Optional, separately-callable statistics over a `run_cv` result table --
never invoked automatically by `run_cv` itself, since different demos
care about different measures (accuracy, runtime, rule/concept
complexity, ...) and a report should only include what it actually
needs. Each function here takes the same long-format `results`
DataFrame `run_cv` returns (``dataset``, ``learner``, plus whatever
measure columns), so a demo picks whichever of these (if any) its own
report needs:

- `mean_rank` -- per-learner average rank across datasets.
- `win_counts` -- per-learner count of datasets it was (tied-for-)best
  on, generalizing ``demo_workflow_comparison.py``'s pairwise
  `_better_counts` to any number of learners.
- `friedman_test` -- whether the learners' ranks differ significantly
  overall (`scipy.stats.friedmanchisquare`).
- `critical_difference_diagram` -- mean ranks plus the Nemenyi/Demsar
  (2006) critical-difference bar, plotted via `matplotlib`.

A dataset may have several rows for the same (dataset, learner) pair
(`run_cv`'s per-fold rows) -- everything here first averages `measure`
within each (dataset, learner) pair, so a fold that failed (`error` set,
`measure` NaN) pulls that pair's average down towards NaN only if
*every* fold failed; a mixed dataset (some folds ok, some not) is
scored on its successful folds only. A pair with no successful fold at
all (all-NaN average) is treated as a failure -- tied for last place in
`mean_rank`, and the whole dataset is dropped from `friedman_test`/
`critical_difference_diagram` (both need a complete dataset x learner
matrix; see their own docstrings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _pivot(results, measure: str):
    """dataset x learner matrix of `measure`, averaged within each
    (dataset, learner) pair first (see the module docstring)."""
    return results.pivot_table(index="dataset", columns="learner", values=measure, aggfunc="mean")


def mean_rank(results, measure: str, higher_is_better: bool = True):
    """Each learner's rank of `measure` (1 = best), averaged across every
    dataset in `results` -- standard average-tie ranking within a
    dataset, a learner with no successful fold on a dataset tied for
    last place on it (`_rank_dataset_by`'s convention, generalized from
    ``examples/demo_workflow_comparison.py``). Returns a `pandas.Series`
    indexed by learner, sorted best-first (ascending rank)."""
    table = _pivot(results, measure)
    n_learners = table.shape[1]
    ranks = table.rank(axis=1, ascending=not higher_is_better, method="average")
    for dataset, row in table.iterrows():
        failed = row.isna()
        n_fail = int(failed.sum())
        if n_fail == 0:
            continue
        n_ok = n_learners - n_fail
        last_rank = n_ok + (n_fail + 1) / 2
        ranks.loc[dataset, failed] = last_rank
    return ranks.mean(axis=0).sort_values()


def win_counts(results, measure: str, higher_is_better: bool = True):
    """Each learner's win count: for every dataset, whoever's `measure`
    is best gets a point -- a tie among the best split that point evenly
    among them (`demo_workflow_comparison.py`'s `_better_counts`
    convention, generalized from two learners to any number). A dataset
    where every learner failed (no successful fold at all -- see the
    module docstring) contributes no points to anyone. Returns a
    `pandas.Series` indexed by learner, sorted most-wins-first."""
    import pandas as pd

    table = _pivot(results, measure)
    wins = pd.Series(0.0, index=table.columns)
    for _, row in table.iterrows():
        ok = row.dropna()
        if ok.empty:
            continue
        best = ok.max() if higher_is_better else ok.min()
        winners = ok.index[ok == best]
        wins.loc[winners] += 1.0 / len(winners)
    return wins.sort_values(ascending=False)


@dataclass
class FriedmanResult:
    """`scipy.stats.friedmanchisquare`'s result, plus how many datasets/
    learners actually went into it (see `friedman_test`)."""

    statistic: float
    pvalue: float
    n_datasets: int
    n_learners: int


def friedman_test(results, measure: str) -> FriedmanResult:
    """The Friedman test (Friedman, 1937/1940) for whether the learners'
    `measure` ranks differ more than chance across datasets --
    `scipy.stats.friedmanchisquare` over the dataset x learner matrix of
    `measure`. Needs a *complete* matrix: datasets where any learner has
    no successful fold at all are dropped first (a Friedman test can't
    score a missing cell) -- raises `ValueError` if fewer than 3 learners
    (`friedmanchisquare`'s own requirement -- it compares 3+ treatments)
    or fewer than 2 complete datasets remain.
    """
    from scipy.stats import friedmanchisquare

    table = _pivot(results, measure).dropna(axis=0, how="any")
    if table.shape[1] < 3:
        raise ValueError(
            f"friedman_test needs >=3 learners (scipy.stats.friedmanchisquare's own "
            f"requirement); got {table.shape[1]} for {measure!r}"
        )
    if table.shape[0] < 2:
        raise ValueError(
            f"friedman_test needs >=2 datasets with a successful fold for every learner; "
            f"got {table.shape[0]} for {measure!r}"
        )
    statistic, pvalue = friedmanchisquare(*(table[col].to_numpy() for col in table.columns))
    return FriedmanResult(statistic=float(statistic), pvalue=float(pvalue),
                          n_datasets=table.shape[0], n_learners=table.shape[1])


def critical_difference(n_learners: int, n_datasets: int, alpha: float = 0.05) -> float:
    """The Nemenyi/Demsar (2006) critical difference: two learners' mean
    ranks (over `n_datasets` datasets, among `n_learners` learners
    compared) must differ by at least this much to call the difference
    significant at `alpha`. ``q_alpha * sqrt(n_learners * (n_learners + 1)
    / (6 * n_datasets))``, `q_alpha` the studentized range statistic at
    infinite degrees of freedom, divided by `sqrt(2)` (Demsar's Table
    5(b), reproduced here via `scipy.stats.studentized_range` rather than
    a literal lookup table -- confirmed to match Demsar's published
    values for `n_learners` 2 through 10)."""
    from scipy.stats import studentized_range

    q_alpha = studentized_range.ppf(1 - alpha, n_learners, np.inf) / np.sqrt(2)
    return float(q_alpha * np.sqrt(n_learners * (n_learners + 1) / (6 * n_datasets)))


def critical_difference_diagram(results, measure: str, higher_is_better: bool = True,
                                ax=None, alpha: float = 0.05):
    """Mean ranks (`mean_rank`) plotted on a single axis, best to the
    left, with the Nemenyi critical-difference bar (`critical_difference`)
    shown as a reference span and thick bars linking every group of
    learners whose mean ranks are within one CD of each other (the
    standard Demsar (2006) critical-difference diagram). Uses every
    dataset `mean_rank` does (failures tied for last, no dataset
    dropped) -- `n_datasets` for the CD itself is `results['dataset']`'s
    number of distinct values.

    Returns the `matplotlib.axes.Axes` drawn on (a new figure's, if `ax`
    is not given). Needs `matplotlib`."""
    import matplotlib.pyplot as plt

    ranks = mean_rank(results, measure, higher_is_better=higher_is_better)
    n_datasets = results["dataset"].nunique()
    n_learners = len(ranks)
    cd = critical_difference(n_learners, n_datasets, alpha=alpha)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 1.2 + 0.3 * n_learners))

    order = ranks.sort_values()
    ys = np.arange(len(order))[::-1]
    ax.scatter(order.to_numpy(), ys, color="black", zorder=3)
    for y, (name, r) in zip(ys, order.items()):
        ax.text(r, y + 0.15, name, ha="center", va="bottom", fontsize=9)

    # thick bars linking any maximal run of adjacent learners within one CD
    values = order.to_numpy()
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i:
            ax.plot([values[i], values[j]], [ys[i] - 0.3 - 0.1 * i, ys[i] - 0.3 - 0.1 * i],
                    color="black", linewidth=2)
        i = j + 1

    lo, hi = float(values.min()), float(values.max())
    ax.plot([lo, lo + cd], [ys[0] + 0.6, ys[0] + 0.6], color="black", linewidth=1.5)
    ax.text((2 * lo + cd) / 2, ys[0] + 0.75, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(f"mean rank ({measure})")
    ax.set_yticks([])
    ax.set_ylim(-0.6, ys[0] + 1.1)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    return ax
