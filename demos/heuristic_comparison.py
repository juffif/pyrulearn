"""
demos/heuristic_comparison.py
=============================

Rule learning heuristics inside a plain separate-and-conquer learner,
after Janssen & Fürnkranz, "On the Quest for Optimal Rule Learning
Heuristics" (Machine Learning, 2010).

The learner follows the paper's Algorithms 1 and 2:

- single rule: greedy top-down search that keeps adding the best
  condition until no refinement is left and returns the best rule
  encountered along the way (`BeamSearch(beam_width=1)`), no stopping
  criterion, no pruning;
- covering: stop once the newly learned rule covers no more positives
  than negatives (`stop_covering`: tp - fp <= 0);
- multi-class: ordered class binarization -- classes from least to most
  frequent, each learned against all larger classes, the largest one
  becoming the default rule (`ConceptCascade`, order "least_frequent").

The only thing that varies is the heuristic: the paper's standard
heuristics (Precision, Laplace, Accuracy, WRA, Correlation) and its
tuned parametrized ones (cost c=0.437, relative cost c_r=0.342,
F-measure beta=0.5, m-estimate m=22.466), plus an m-estimate sweep
over m in {2, 4, 8, 16, 32}. Weka's JRip (as in the paper) and our own
Pypper (both also with ordered class binarization) are included as
benchmarks. Kloesgen is left out.

Differences from the paper: numeric attributes are discretized by
`build_dataspec` (tree-based, `MAX_INTERVALS` intervals per attribute)
rather than tested at every midpoint between adjacent values, ties are
not broken randomly, and the datasets are the subset of the paper's 57
that are in `pyrulearn.experiments.catalog` (their tuning/validation
split is not kept).

Run: `python demos/heuristic_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`); its
report/plots go to ``heuristic_comparison_quick_*`` and are not
checked in. `--full` runs all `FULL_DATASETS` with `N_FOLDS` folds and
writes the canonical ``heuristic_comparison_report.md`` (plots in
``heuristic_comparison_plots/``). Needs Weka at
`WEKA_JAR`/`WEKA_JAVA` below, plus the `experiments` extra.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank, win_counts
from pyrulearn.heuristics import (
    Accuracy, Correlation, CoverageDifference, FBeta, Laplace, LinearCost, LinearCostRates,
    MEstimate, Precision, WRAcc,
)
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners.seco import BeamSearch, Pypper, SeCo, SingleRuleLearner
from pyrulearn.models import ConceptCascade, ConceptModel
from pyrulearn.pruning import ThresholdPrePruning

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 600.0  # no stopping criterion -> Precision/Laplace/Accuracy grow large theories

M_SWEEP = (2, 4, 8, 16, 32)
PAPER_M = 22.466

# The paper's datasets that are in the catalog (their names -> ours where
# they differ: cleveland-heart-disease -> heart-c, credit/credit-a ->
# credit-approval, horse-colic -> colic, krkp -> kr-vs-kp, lymphography ->
# lymph, monk1-3 -> monks-problems-1..3, house-votes-84 -> vote,
# promoters -> molecular-biology_promoters). sick (not the same data as
# their sick-euthyroid) and titanic (a different version) are left out.
FULL_DATASETS = [
    # from their 27 tuning datasets
    "anneal", "audiology", "breast-cancer", "heart-c", "credit-approval", "glass", "hepatitis",
    "colic", "hypothyroid", "iris", "kr-vs-kp", "lymph", "monks-problems-1", "monks-problems-2",
    "monks-problems-3", "mushroom", "soybean", "tic-tac-toe", "vote", "vowel", "wine",
    # from their 30 validation datasets
    "balance-scale", "breast-w", "credit-g", "diabetes", "hayes-roth", "heart-h", "heart-statlog",
    "ionosphere", "primary-tumor", "molecular-biology_promoters", "segment", "solar-flare",
    "sonar", "vehicle", "zoo",
]
FULL_DATASETS_SPEC = ",".join(FULL_DATASETS)

# Quick run (the no-args default): ten small datasets from the list above,
# binary and multi-class, 3-fold.
QUICK_DATASETS_SPEC = ("breast-cancer,hepatitis,heart-c,vote,iris,wine,glass,tic-tac-toe,lymph,"
                       "hayes-roth")
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "heuristic_comparison_report.md")
PLOTS_DIR = os.path.join(HERE, "heuristic_comparison_plots")
TRADEOFF_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_tradeoff.png")
M_SWEEP_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_m_sweep.png")
QUICK_REPORT_PATH = os.path.join(HERE, "heuristic_comparison_quick_report.md")
QUICK_TRADEOFF_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_tradeoff.png")
QUICK_M_SWEEP_PLOT_PATH = os.path.join(PLOTS_DIR, "heuristic_comparison_quick_m_sweep.png")
CACHE_DIR = os.path.join(HERE, "_heuristic_comparison_cache")


def _cost_ratio(c: float) -> float:
    """The paper's c*p - (1-c)*n ranks rules exactly like p - ((1-c)/c)*n."""
    return (1 - c) / c


class HeuristicSeCo(SeCo):
    """The paper's covering learner with one fixed heuristic, named by
    `label` in the results."""

    def __init__(self, label: str, heuristic, random_state: Optional[int] = None):
        super().__init__(
            single_rule_learner=SingleRuleLearner(heuristic=heuristic, search=BeamSearch(beam_width=1)),
            stop_covering=ThresholdPrePruning(CoverageDifference(), 0, operator="<="),
            random_state=random_state,
        )
        self.label = label

    @property
    def display_name(self) -> str:
        return self.label

    def _provenance_params(self):
        return {"label": self.label}

    def _default_model(self, data) -> type:
        return ConceptModel if self.target_class is not None else ConceptCascade


def build_heuristic_variants():
    specs = [(f"m-estimate (m={m})", MEstimate(m=m)) for m in M_SWEEP]
    specs += [
        (f"m-estimate (m={PAPER_M})", MEstimate(m=PAPER_M)),
        ("relative cost (c_r=0.342)", LinearCostRates(cost_ratio=_cost_ratio(0.342))),
        ("cost (c=0.437)", LinearCost(cost_ratio=_cost_ratio(0.437))),
        ("F-measure (beta=0.5)", FBeta(beta=0.5)),
        ("Correlation", Correlation()),
        ("WRA", WRAcc()),
        ("Precision", Precision()),
        ("Laplace", Laplace()),
        ("Accuracy", Accuracy()),
    ]
    return [HeuristicSeCo(label, h, random_state=RANDOM_STATE) for label, h in specs]


def build_learners():
    return build_heuristic_variants() + [
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
        Pypper(random_state=RANDOM_STATE),
    ]


BENCHMARKS = ("Weka:JRip", "Pypper")


def _build_description(datasets, n_folds: int, quick: bool) -> str:
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({len(FULL_DATASETS)} "
        f"datasets, {N_FOLDS}-fold): `python demos/heuristic_comparison.py --full`.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} datasets, {n_folds}-fold). Quick sanity check "
        f"instead: `python demos/heuristic_comparison.py` with no arguments.\n\n"
    )
    return f"""\
{mode_note}Rule learning heuristics compared inside one plain separate-and-conquer
learner, after Janssen & Fürnkranz (Machine Learning, 2010). The learner
follows the paper's Algorithms 1 and 2: greedy top-down search that
refines until no refinement is left and returns the best rule
encountered (`BeamSearch(beam_width=1)`), no stopping criterion and no
pruning; covering stops once a new rule covers no more positives than
negatives; multi-class problems use ordered class binarization (least
frequent class first, the largest class as the default rule).

Heuristics: the paper's standard ones (Precision, Laplace, Accuracy,
WRA, Correlation), its tuned parametrized ones (cost c=0.437, relative
cost c_r=0.342, F-measure beta=0.5, m-estimate m={PAPER_M}), and an
m-estimate sweep over m in {{{', '.join(str(m) for m in M_SWEEP)}}}.
Benchmarks: Weka's JRip and pyrulearn's Pypper, both with their own
pruning, and both also use ordered class binarization. What each
heuristic prefers is visualized by its coverage-space isometrics in the
companion demo, [heuristic_isometrics](heuristic_isometrics_report.md).

Datasets: {len(datasets)} of the paper's datasets that are in
`pyrulearn.experiments.catalog` ({len(datasets) - n_multiclass} binary,
{n_multiclass} multi-class). Numeric attributes are discretized per
training fold (`build_dataspec(max_intervals={MAX_INTERVALS})`) instead
of the paper's tests at every midpoint. Each fit is capped at
{FIT_TIMEOUT:.0f}s; a timeout counts as a failure.

Measures: test accuracy, number of rules and conditions, fit time.
"""


def write_report(results, datasets, n_folds: int, report_path: str,
                 tradeoff_plot_path: str, m_sweep_plot_path: str, quick: bool) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(os.path.dirname(tradeoff_plot_path))

    lines = ["# Rule learning heuristics in separate-and-conquer (after Janssen & Fürnkranz)\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick)))
    lines.append(f"![accuracy vs. theory size]({plots_dir}/{os.path.basename(tradeoff_plot_path)})\n\n")
    lines.append(f"![m-estimate sweep]({plots_dir}/{os.path.basename(m_sweep_plot_path)})\n\n")

    lines.append("## Summary by learner (mean across every dataset and fold)\n\n")
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[
        ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines.append("| learner | accuracy | n_rules | n_conditions | conds/rule | fit_time (s) | "
                 "wins | mean rank | failures |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|\n")
    for learner in summary["accuracy"].sort_values(ascending=False).index:
        row = summary.loc[learner]
        lines.append(f"| {learner} | {row['accuracy']:.3f} | {row['n_rules']:.1f} | "
                     f"{row['n_conditions']:.1f} | {row['conds_per_rule']:.2f} | "
                     f"{row['fit_time']:.2f} | {wins.get(learner, 0.0):.1f} | "
                     f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    lines.append("\n`wins` -- datasets where a learner's mean accuracy was (tied-for-)best, a tie "
                 "split evenly; `mean rank` -- average accuracy rank across datasets, failures "
                 "tied for last.\n\n")

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def _m_of(label: str) -> Optional[float]:
    if label.startswith("m-estimate (m="):
        return float(label[len("m-estimate (m="):-1])
    return None


def write_plots(results, tradeoff_plot_path: str, m_sweep_plot_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.dirname(tradeoff_plot_path), exist_ok=True)
    summary = results.groupby("learner")[["accuracy", "n_conditions"]].mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for learner, row in summary.iterrows():
        if learner in BENCHMARKS:
            color, marker = "tab:red", "s"
        elif _m_of(learner) is not None:
            color, marker = "tab:blue", "o"
        else:
            color, marker = "tab:gray", "^"
        ax.scatter(row["n_conditions"], row["accuracy"], color=color, marker=marker, s=50)
        ax.annotate(learner, (row["n_conditions"], row["accuracy"]), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean number of conditions (log)")
    ax.set_ylabel("mean test accuracy")
    ax.set_title("accuracy vs. theory size per heuristic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(tradeoff_plot_path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {tradeoff_plot_path}")

    sweep = sorted((m, learner) for learner in summary.index if (m := _m_of(learner)) is not None)
    ms = [m for m, _ in sweep]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(ms, [summary.loc[l, "accuracy"] for _, l in sweep], "o-", color="tab:blue")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("m (log scale)")
    ax1.set_ylabel("mean test accuracy", color="tab:blue")
    ax1.axvline(PAPER_M, color="gray", linestyle=":", label=f"paper's m={PAPER_M}")
    ax2 = ax1.twinx()
    ax2.plot(ms, [summary.loc[l, "n_conditions"] for _, l in sweep], "s--", color="tab:orange")
    ax2.set_ylabel("mean number of conditions", color="tab:orange")
    ax1.set_title("m-estimate: accuracy and theory size vs. m")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(m_sweep_plot_path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {m_sweep_plot_path}")


def main(datasets_spec: Optional[str] = None) -> None:
    quick = datasets_spec is None
    spec = QUICK_DATASETS_SPEC if quick else datasets_spec
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    tradeoff_plot_path = QUICK_TRADEOFF_PLOT_PATH if quick else TRADEOFF_PLOT_PATH
    m_sweep_plot_path = QUICK_M_SWEEP_PLOT_PATH if quick else M_SWEEP_PLOT_PATH

    datasets = Catalog.default().parse(spec, random_state=RANDOM_STATE)
    results = run_cv(build_learners(), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT,
                     max_intervals=MAX_INTERVALS, random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
    write_report(results, datasets, n_folds, report_path, tradeoff_plot_path, m_sweep_plot_path, quick)
    write_plots(results, tradeoff_plot_path, m_sweep_plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run all {len(FULL_DATASETS)} datasets, {N_FOLDS}-fold, instead of "
                             "the quick default")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    args = parser.parse_args()
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
