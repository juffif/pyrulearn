"""
examples/demo_seco_learners_comparison.py
=========================================

pyrulearn's separate-and-conquer (SeCo) rule learners side by side, with
Weka's JRip and the reference Java implementation of LORD as external
baselines, cross-validated through `pyrulearn.experiments.runner.run_cv`
on mostly symbolic binary and multi-class datasets of
`pyrulearn.experiments.catalog` (see `FULL_BINARY`/`FULL_MULTICLASS`
for which, and why). All learners
here learn a rule at a time and *remove* what it covers; learners that
instead reweight covered examples (CPAR, LRI) belong to a separate
covering vs. weighted-covering comparison.

Learners (each with its own default settings and its own default way of
handling several classes -- see `LEARNER_NOTES`):

- **CN2**      -- `pyrulearn.learners.seco.CN2` (Clark & Boswell 1991):
                  beam search (width 5) with Laplace, CN2's significance
                  test as the stopping criterion.
- **PFoil**    -- `PFoil`: FOIL's greedy gain-ascent search with FOIL
                  gain and Quinlan's (1990) encoding-length restriction.
- **PFossil**  -- `PFossil`: FOSSIL (Fürnkranz 1994), hill climbing on
                  correlation with a 0.3 correlation cutoff.
- **Pypper**   -- `Pypper`: pyrulearn's RIPPER (IREP* grow/prune plus
                  `ReplaceReviseOptimization`).
- **PyLORD**   -- `pyrulearn.learners.pylord.PyLORD`: a simplified
                  reimplementation of LORD (Huynh, Fürnkranz & Beck 2023),
                  one locally optimal rule per training example.
- **Weka:JRip** -- Weka's RIPPER, via `pyrulearn.interfaces.weka.WekaJRip`
                  (needs Weka, `WEKA_JAR`/`WEKA_JAVA` below).
- **JavaLord** -- the reference LORD implementation, via
                  `pyrulearn.interfaces.lord.JavaLord`. Only included when
                  found (`LORD_CLASSPATH` below, or `$LORD_CLASSPATH`).

**AQR** (Clark & Niblett 1989, the AQ baseline CN2 was designed to improve
on) is *not* in the main comparison: `run_preliminary_aqr_check` fits it
against CN2 on a few small datasets first, and its measured result --
long, overfit rules and far higher fit times -- is the report's argument
for leaving it out.

Run: `python examples/demo_seco_learners_comparison.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`-fold); its
report/plots go to ``demo_seco_learners_comparison_quick_*`` and are not
checked in. `--full` runs `FULL_DATASETS_SPEC`, `N_FOLDS`-fold
(`LARGE_FOLDS`-fold for the two large datasets), and writes the canonical
``demo_seco_learners_comparison_report.md`` (plots in
``demo_seco_learners_comparison_plots/``). Needs the `experiments` extra.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank, win_counts
from pyrulearn.interfaces.lord import JavaLord
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil, Pypper

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"
# the reference LORD (github.com/vqphuynh/LORD): its compiled classes plus
# its bundled Weka; $LORD_CLASSPATH / $LORD_JAVA take precedence
LORD_DIR = r"C:\Users\juffi\Github\LORD"
LORD_CLASSPATH = os.environ.get("LORD_CLASSPATH") or os.pathsep.join(
    [os.path.join(LORD_DIR, "bin"), os.path.join(LORD_DIR, "libs", "weka_3.8_stable.jar")])
LORD_JAVA = os.environ.get("LORD_JAVA") or WEKA_JAVA

RANDOM_STATE = 0
N_FOLDS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0

# Datasets whose attributes are mostly symbolic (at least half nominal) --
# the kind these learners were designed for; numeric attributes and their
# discretization are the numeric_discretization demo's topic. Left out:
# artificial concepts (monks-problems-1/2/3, mofn-3-7-10, hayes-roth, led24),
# datasets with many classes (audiology, primary-tumor, soybean, kropt --
# for the multi-class demo), and large datasets other than adult and
# connect-4.
FULL_BINARY = [
    "molecular-biology_promoters", "hepatitis", "SPECT", "heart-statlog", "breast-cancer",
    "heart-h", "heart-c", "colic", "vote", "dresses-sales", "cylinder-bands", "credit-approval",
    "tic-tac-toe", "credit-g", "kr-vs-kp", "sick", "mushroom", "adult",
]
FULL_MULTICLASS = [
    "zoo", "lymph", "analcatdata_dmft", "anneal", "solar-flare", "cmc", "car", "dna", "splice",
    "hypothyroid", "connect-4",
]
FULL_DATASETS_SPEC = ",".join(FULL_BINARY + FULL_MULTICLASS)
LARGE_FOLDS = 5  # for the catalog's "large" datasets (adult, connect-4)

# ten small, mostly symbolic datasets, two of them multi-class
QUICK_DATASETS_SPEC = ("vote,tic-tac-toe,hepatitis,breast-cancer,heart-statlog,credit-approval,"
                       "colic,SPECT,zoo,lymph")
QUICK_FOLDS = 3

HERE = os.path.dirname(__file__)
NAME = "demo_seco_learners_comparison"
REPORT_PATH = os.path.join(HERE, f"{NAME}_report.md")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
QUICK_REPORT_PATH = os.path.join(HERE, f"{NAME}_quick_report.md")
CACHE_DIR = os.path.join(HERE, "_seco_learners_comparison_cache")


def _plot_paths(quick: bool):
    tag = "_quick" if quick else ""
    return {kind: os.path.join(PLOTS_DIR, f"{NAME}{tag}_{kind}.png")
            for kind in ("accuracy", "fit_time", "cd")}


def lord_available() -> bool:
    return os.path.exists(LORD_CLASSPATH.split(os.pathsep)[0])


def build_learners():
    learners = [
        CN2(random_state=RANDOM_STATE),
        PFoil(random_state=RANDOM_STATE),
        PFossil(random_state=RANDOM_STATE),
        Pypper(random_state=RANDOM_STATE),
        PyLORD(random_state=RANDOM_STATE),
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
    ]
    if lord_available():
        learners.append(JavaLord(classpath=LORD_CLASSPATH, java=LORD_JAVA))
    return learners


# AQR vs. CN2 preliminary check -- see run_preliminary_aqr_check(): small
# datasets with class noise, one of them (credit-approval) with several
# numeric attributes, where AQR's time blows up
PRELIM_DATASETS = ["breast-cancer", "hepatitis", "heart-statlog", "credit-approval"]
PRELIM_FOLDS = 3


def run_preliminary_aqr_check():
    """Fits `AQR` against `CN2` -- the learner designed to improve on it --
    on `PRELIM_DATASETS`, and returns `(results, verdict)`: the `run_cv`
    table and a sentence built from the measured numbers."""
    datasets = Catalog.default().select(names=PRELIM_DATASETS)
    results = run_cv([CN2(random_state=RANDOM_STATE), AQR(random_state=RANDOM_STATE)], datasets,
                     n_folds=PRELIM_FOLDS, fit_timeout=FIT_TIMEOUT, max_intervals=MAX_INTERVALS,
                     random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
    r = results.copy()
    r["conds_per_rule"] = r["n_conditions"] / r["n_rules"]
    m = r.groupby("learner")[["accuracy", "n_conditions", "conds_per_rule", "fit_time"]].mean()
    aqr, cn2 = m.loc["AQR"], m.loc["CN2"]
    verdict = (
        f"Across {len(datasets)} small datasets ({PRELIM_FOLDS}-fold), AQR was "
        f"{'less' if aqr['accuracy'] < cn2['accuracy'] else 'not less'} accurate than CN2 "
        f"(mean accuracy {aqr['accuracy']:.3f} vs. {cn2['accuracy']:.3f}), learned rules "
        f"{aqr['conds_per_rule'] / cn2['conds_per_rule']:.1f}x as long "
        f"({aqr['conds_per_rule']:.1f} vs. {cn2['conds_per_rule']:.1f} conditions per rule; "
        f"{aqr['n_conditions']:.0f} vs. {cn2['n_conditions']:.0f} conditions in total), and took "
        f"{aqr['fit_time'] / cn2['fit_time']:.0f}x as long per fit ({aqr['fit_time']:.1f}s vs. "
        f"{cn2['fit_time']:.2f}s, mean) -- requiring every rule to be consistent overfits noisy "
        f"data, and on larger datasets with numeric attributes the fit times would dominate the "
        f"whole comparison."
    )
    return results, verdict


# what the report says about each learner, keyed by display_name
LEARNER_NOTES = {
    "CN2": "Beam search (width 5) with the Laplace estimate; CN2's likelihood-ratio "
           "significance test stops rules that aren't significant. Learns rules for "
           "every class (one-vs-rest rule set).",
    "PFoil": "FOIL: greedy search that adds the condition with the highest FOIL gain; "
             "Quinlan's encoding-length restriction stops rules that cost more bits "
             "than the examples they explain. Rules for every class.",
    "PFossil": "FOSSIL: hill climbing on the correlation between rule and class; rules "
               "below a correlation of 0.3 are dropped. Rules for every class.",
    "Pypper": "pyrulearn's RIPPER: grow on two thirds of the data, prune on the rest "
              "(IREP*), then optimize the rule set (replace/revise). Rules for the "
              "classes from least to most frequent, the most frequent one as default.",
    "PyLORD": "A simplified LORD: for every training example, the best rule covering it "
              "(greedy m-estimate search, then pruning); a test example is classified by "
              "the best rule that covers it. Many overlapping rules.",
    "Weka:JRip": "Weka's RIPPER, run as a subprocess on the same binarized data. Fit time "
                 "includes the JVM start.",
    "JavaLord": "The reference LORD implementation (Java), run as a subprocess on the same "
                "binarized data.",
}


def _build_description(datasets, n_folds: int, quick: bool, learner_names,
                       aqr_verdict: str) -> str:
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({len(FULL_BINARY)} binary and "
        f"{len(FULL_MULTICLASS)} multi-class datasets): `python examples/{NAME}.py --full`.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} datasets). Quick sanity check "
        f"instead: `python examples/{NAME}.py` with no arguments.\n\n"
    )
    n_multiclass = sum(1 for d in datasets if d.task == "multiclass")
    large = [d.name for d in datasets if d.size == "large"]
    folds_note = (f"{n_folds}-fold stratified cross-validation" if quick or not large else
                  f"{n_folds}-fold stratified cross-validation ({LARGE_FOLDS}-fold for the "
                  f"large {', '.join(large)})")
    lord_note = ("" if lord_available() else
                 f"\nThe reference Java LORD is not included in this run (not found at "
                 f"`{LORD_CLASSPATH.split(os.pathsep)[0]}`; set `$LORD_CLASSPATH`).\n")
    learner_lines = "\n".join(f"- **{n}** -- {LEARNER_NOTES.get(n, '')}" for n in learner_names)
    return f"""\
{mode_note}pyrulearn's separate-and-conquer rule learners compared with each other
and with two external baselines, on {len(datasets) - n_multiclass} binary and
{n_multiclass} multi-class datasets from `pyrulearn.experiments.catalog`.
All of them learn one rule at a time and remove the examples it covers;
they differ in how a rule is searched for, when it stops, and whether and
how the rule set is pruned. Every learner runs with its default settings,
including its default way of handling several classes.

The datasets have mostly symbolic attributes (at least half nominal), the
kind these learners were designed for -- numeric attributes and their
discretization are the topic of the numeric-discretization demo. Artificial
concepts (the monks problems, m-of-n, Hayes-Roth, LED) and datasets with
many classes (left for the multi-class demo) are not included.

{learner_lines}
{lord_note}
AQR, the AQ baseline CN2 was designed to improve on, was checked
separately and left out -- see "Why AQR isn't in the main comparison".
{aqr_verdict}

Protocol: {folds_note}
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals={MAX_INTERVALS})`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical Boolean
feature matrix per fold. Each fit is capped at {FIT_TIMEOUT:.0f}s; a
time-out or an error counts as a failure (and as last in the ranking),
not as the end of the run. The cap is deliberately kept for the large
datasets too: which learners scale to tens of thousands of examples is
part of the result (a single CN2 fit on adult takes about 20 minutes).

Measures: test accuracy, number of rules and conditions, fit time.
"""


def write_report(results, datasets, n_folds: int, report_path: str, plots: dict, quick: bool,
                 learner_names, aqr_results, aqr_verdict: str) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]
    plots_dir = os.path.basename(PLOTS_DIR)

    lines = ["# Separate-and-conquer rule learners compared\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick, learner_names, aqr_verdict)))
    lines.append(f"![accuracy vs. rule-set complexity]({plots_dir}/{os.path.basename(plots['accuracy'])})\n\n")
    lines.append(f"![fit time per dataset]({plots_dir}/{os.path.basename(plots['fit_time'])})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({plots_dir}/{os.path.basename(plots['cd'])})\n\n")

    lines.append("## Why AQR isn't in the main comparison\n\n")
    lines.append(f"{aqr_verdict}\n\n")
    aqr_results = aqr_results.copy()
    aqr_results["conds_per_rule"] = aqr_results["n_conditions"] / aqr_results["n_rules"]
    lines.append(render_results_table(
        aqr_results, ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

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
                 "tied for last. Means over successful fits only; `failures` counts the fits "
                 "that timed out or raised.\n\n")

    task = {d.name: d.task for d in datasets}
    if len(set(task.values())) > 1:
        lines.append("## Mean accuracy by target type\n\n")
        lines.append("| learner | binary | multi-class |\n|---|--:|--:|\n")
        by_task = results.assign(task=results["dataset"].map(task)).groupby(
            ["learner", "task"])["accuracy"].mean()
        for learner in summary["accuracy"].sort_values(ascending=False).index:
            b, m = by_task.get((learner, "binary")), by_task.get((learner, "multiclass"))
            lines.append(f"| {learner} | {b:.3f} | {m:.3f} |\n")
        lines.append("\n")

    lines.append("## Per-dataset results, per learner (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "fit_time"],
        group_by=["dataset", "learner"], include_overall=False))
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def write_plots(results, plots: dict, learner_names) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = dict(zip(learner_names, plt.get_cmap("tab10").colors))
    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for learner in learner_names:
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            ax.scatter(sub["n_conditions"], sub["accuracy"], label=learner,
                       color=colors[learner], alpha=0.7, s=40)
    ax.set_xscale("symlog")
    ax.set_xlabel("total conditions (rule set complexity)")
    ax.set_ylabel("test accuracy")
    ax.set_title("accuracy vs. rule-set complexity, one point per dataset")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots["accuracy"], dpi=110)
    plt.close(fig)
    print(f"Plot   -> {plots['accuracy']}")

    names = sorted(per_pair["dataset"].unique())
    x = np.arange(len(names))
    w = 0.8 / len(learner_names)
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names) * len(learner_names) / 4), 5.5))
    for i, learner in enumerate(learner_names):
        sub = per_pair[per_pair["learner"] == learner].set_index("dataset")
        ts = [sub["fit_time"].get(n, np.nan) for n in names]
        ax.bar(x + (i - (len(learner_names) - 1) / 2) * w, ts, w, label=learner, color=colors[learner])
    ax.set_yscale("log")
    ax.set_ylabel("mean fit time (s, log)")
    ax.set_title("fit time per dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(plots["fit_time"], dpi=110)
    plt.close(fig)
    print(f"Plot   -> {plots['fit_time']}")

    try:
        ax = critical_difference_diagram(results, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(plots["cd"], dpi=110)
        plt.close(ax.figure)
        print(f"Plot   -> {plots['cd']}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the report
        print(f"Critical-difference diagram skipped: {type(e).__name__}: {e}")


def main(datasets_spec: Optional[str] = None) -> None:
    quick = datasets_spec is None
    spec = QUICK_DATASETS_SPEC if quick else datasets_spec
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    plots = _plot_paths(quick)

    datasets = Catalog.default().parse(spec, random_state=RANDOM_STATE)
    learners = build_learners()
    learner_names = [l.display_name for l in learners]
    print("Preliminary check: AQR vs. CN2 ...")
    aqr_results, aqr_verdict = run_preliminary_aqr_check()
    print(aqr_verdict)

    groups = [(datasets, n_folds)] if quick else [
        ([d for d in datasets if d.size != "large"], n_folds),
        ([d for d in datasets if d.size == "large"], LARGE_FOLDS),
    ]
    results = pd.concat([
        run_cv(learners, group, n_folds=folds, fit_timeout=FIT_TIMEOUT, max_intervals=MAX_INTERVALS,
               random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
        for group, folds in groups if group
    ], ignore_index=True)
    write_report(results, datasets, n_folds, report_path, plots, quick, learner_names,
                 aqr_results, aqr_verdict)
    write_plots(results, plots, learner_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run the full comparison ({len(FULL_BINARY)} binary and "
                             f"{len(FULL_MULTICLASS)} multi-class datasets) instead of the quick default")
    parser.add_argument("--datasets", default=None,
                        help="Catalog.parse() spec for a full-scale run, e.g. 'vote,mushroom'")
    args = parser.parse_args()
    main(args.datasets or (FULL_DATASETS_SPEC if args.full else None))
