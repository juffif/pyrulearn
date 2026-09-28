"""
examples/demo_ripper_comparison.py
===================================

Four RIPPER-family rule learners (Slipper included -- Cohen & Singer's
own confidence-rated-boosting follow-up to RIPPER, not a separate
family), cross-validated
through `pyrulearn.experiments.runner.run_cv` (shared per-fold
binarization, per-fit timeout, uniform measurement, fold caching) on
every small/medium binary dataset in `pyrulearn.experiments.catalog`
(44 of them, by default -- `--datasets` widens or narrows the selection,
e.g. to `small,medium` for 83 datasets including multi-class, or `all`
for the full 100-dataset catalog, sized for a strong machine running
overnight rather than a quick check) -- this demo picks the learners,
the datasets, and what to report; everything else is the shared
infrastructure:

- **Weka:JRip**   -- Weka's `weka.classifiers.rules.JRip`, via
                     `pyrulearn.interfaces.weka.WekaJRip` (subprocess,
                     needs a local Weka install -- see `WEKA_JAR`/
                     `WEKA_JAVA` below).
- **Witt:RIPPER** -- the `wittgenstein` package's RIPPER, via
                     `pyrulearn.interfaces.wittgenstein.WittRIPPER`.
                     Natively binary; the local subclass below overrides
                     `_default_model` so `fit(data)` (as `run_cv` always
                     calls it, no `model=`) auto-dispatches to
                     `ConceptModel` for a binary target or `ConceptSet`
                     (one-vs-rest) for multi-class, so one learner
                     instance handles every dataset here.
- **Pypper**      -- pyrulearn's own RIPPER re-implementation
                     (`pyrulearn.learners.seco.Pypper`): IREP* growth/
                     pruning plus `ReplaceReviseOptimization`, per class
                     inside a least-frequent-first ordered decomposition.
- **Slipper**     -- pyrulearn's own SLIPPER (Cohen & Singer, 1999):
                     confidence-rated boosting of rules
                     (`pyrulearn.learners.boosting.Slipper`).

`IMod:Slipper` (`imodels`' `SlipperClassifier`, `pyrulearn.interfaces.
imodels.IModSlipper` -- despite the name, a different algorithm from the
native `Slipper` above, see its own docstring) is deliberately **not**
in that main comparison: `run_preliminary_slipper_check` fits it against
the native `Slipper` on `PRELIM_DATASETS` (a handful of small, few-
feature datasets, so this check itself stays quick) first, and its
result -- consistently much slower, for no accuracy gain (on a couple of
sampled larger datasets it was 20-60x slower than every other learner
here, once timing out outright at a 60s budget on a plain 1,000-row
dataset) -- is the report's own argument for excluding it, not an
unexplained absence. It's binary-only regardless (see `IModSlipper`'s
docstring), so it would have needed dropping from any multi-class run
anyway.

All four main-comparison learners see the identical per-fold
`BooleanDataRepresentation` (`build_dataspec`/`binarize`, `MAX_INTERVALS`
discretization) -- no learner here gets the raw, non-binarized data.
That comparison (a learner's own intrinsic numeric handling vs. our
discretization) is a deliberately separate, later demo
(`examples/REVISION_PLAN.md`'s comparison (e)); this one used to also
run Weka JRip on raw data (`jrip_native`) as a preview of that, which is
why the two are related but no longer why they're the same script.

Run: `python examples/demo_ripper_comparison.py` (needs Weka installed
at `WEKA_JAR`/`WEKA_JAVA` below, plus the `experiments`, `wittgenstein`
and `imodels` extras).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference_diagram, mean_rank
from pyrulearn.interfaces.imodels import IModSlipper
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.interfaces.wittgenstein import WittRIPPER as _WittRIPPERBase
from pyrulearn.learners.boosting import Slipper
from pyrulearn.learners.seco import Pypper
from pyrulearn.models import ConceptModel, ConceptSet

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
N_FOLDS = 10             # cheap now that IMod:Slipper -- the one slow learner -- is out of the main run
MAX_INTERVALS = 6        # numeric-feature discretization (build_dataspec)
FIT_TIMEOUT = 180.0      # seconds, per (dataset, fold, learner) -- Weka's JVM startup included

# Slipper vs. IMod:Slipper preliminary check -- see run_preliminary_slipper_check().
# Small, low-feature-count datasets (feature count, not row count, is
# what made IMod:Slipper slow in testing -- e.g. it took ~19s/fit on the
# 208-row but 60-numeric-attribute `sonar`, vs. ~5s/fit on the larger but
# 19-attribute `hepatitis`), so this stays quick even though IMod:Slipper
# is in it.
PRELIM_DATASETS = ["breast-cancer", "heart-statlog", "hepatitis", "vote"]
PRELIM_FOLDS = 3
PRELIM_TIMEOUT = 90.0

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "demo_ripper_comparison_report.md")
PLOT_PATH = os.path.join(HERE, "demo_ripper_comparison.png")
CD_PLOT_PATH = os.path.join(HERE, "demo_ripper_comparison_cd.png")
CACHE_DIR = os.path.join(HERE, "_ripper_comparison_cache")

# Default: every small/medium (<=10,000 rows) binary dataset in the
# catalog -- 44 of them as of this writing (25 small + 19 medium; see
# Catalog.default().summary()). --datasets overrides this, e.g.
# '--datasets small,medium' (83, binary and multi-class both --
# exercises Witt:RIPPER's multi-class dispatch) or '--datasets all'
# (100 -- includes >10,000-row datasets, an overnight-on-a-strong-
# machine run, not a quick one).
DEFAULT_DATASETS_SPEC = "binary,small,medium"


def _build_description(datasets, prelim_verdict: str) -> str:
    n_binary = sum(1 for d in datasets if d.task == "binary")
    n_multiclass = len(datasets) - n_binary
    dataset_count = (f"{len(datasets)} binary" if n_multiclass == 0
                    else f"{n_binary} binary and {n_multiclass} multi-class")
    return f"""\
Four RIPPER-family rule learners -- Weka:JRip, Witt:RIPPER,
Pypper, Slipper -- compared on {dataset_count} datasets from
`pyrulearn.experiments.catalog`
(default: {DEFAULT_DATASETS_SPEC!r}; `--datasets` overrides the selection,
e.g. 'small,medium' or 'all' -- see the module docstring for what those add).

A fifth, IMod:Slipper (`imodels`' SlipperClassifier), is checked
separately first against the native Slipper on a handful of small,
low-feature-count datasets, then left out of the main comparison below --
see "Why IMod:Slipper isn't in the main comparison". {prelim_verdict}

Protocol: `{N_FOLDS}`-fold stratified cross-validation
(`pyrulearn.experiments.runner.run_cv`), one `DataSpec` per training fold
(`build_dataspec(max_intervals={MAX_INTERVALS})`), the test fold binarized
against that same `DataSpec` -- every learner sees the identical
Boolean feature matrix per fold, so differences reflect the algorithms,
not the data preparation. Each fit is capped at {FIT_TIMEOUT:.0f}s; a
timeout or an exception is recorded as a failure, not fatal to the run.

Measures: test accuracy, rule count, total condition count, fit time
(seconds, JVM startup included for Weka:JRip). Witt:RIPPER auto-dispatches
one-vs-rest for multi-class (see the module docstring); every other
learner handles multi-class natively.
"""


class WittRIPPER(_WittRIPPERBase):
    """`WittRIPPER`, but `fit(data)` with no `model=` (as `run_cv` always
    calls it) picks `ConceptModel` for a binary target, `ConceptSet`
    (one-vs-rest) for multi-class -- see the module docstring."""

    def _default_model(self, data) -> type:
        return ConceptModel if len(np.unique(np.asarray(data.y))) <= 2 else ConceptSet


def build_learners():
    return [
        WekaJRip(jar=WEKA_JAR, java=WEKA_JAVA),
        WittRIPPER(random_state=RANDOM_STATE),
        Pypper(random_state=RANDOM_STATE),
        Slipper(random_state=RANDOM_STATE),
    ]


LEARNER_ORDER = ["Weka:JRip", "Witt:RIPPER", "Pypper", "Slipper"]
COLORS = dict(zip(LEARNER_ORDER, ["tab:blue", "tab:orange", "tab:green", "tab:red"]))


def select_datasets(spec: Optional[str] = None):
    return Catalog.default().parse(spec or DEFAULT_DATASETS_SPEC, random_state=RANDOM_STATE)


def run_preliminary_slipper_check():
    """Fits the native `Slipper` against `IMod:Slipper` on
    `PRELIM_DATASETS`, and returns `(results, verdict)` -- `results` the
    usual long-format `run_cv` table (just these two learners, these
    datasets), `verdict` a plain-English sentence built from the actual
    numbers (never a canned claim) comparing their mean accuracy and
    mean fit time, for both the report and the module `DESCRIPTION`.
    """
    datasets = Catalog.default().select(names=PRELIM_DATASETS)
    results = run_cv([Slipper(random_state=RANDOM_STATE), IModSlipper(random_state=RANDOM_STATE)],
                     datasets, n_folds=PRELIM_FOLDS, fit_timeout=PRELIM_TIMEOUT,
                     max_intervals=MAX_INTERVALS, random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
    by_learner = results.groupby("learner").agg(accuracy=("accuracy", "mean"), fit_time=("fit_time", "mean"))
    slipper, imod = by_learner.loc["Slipper"], by_learner.loc["IMod:Slipper"]
    ratio = imod["fit_time"] / slipper["fit_time"] if slipper["fit_time"] else float("inf")
    acc_cmp = ("no more accurate" if imod["accuracy"] <= slipper["accuracy"] else "more accurate")
    verdict = (
        f"Across {len(datasets)} datasets ({PRELIM_FOLDS}-fold), IMod:Slipper was {acc_cmp} than "
        f"Slipper (mean accuracy {imod['accuracy']:.3f} vs. {slipper['accuracy']:.3f}) while taking "
        f"{ratio:.0f}x as long per fit ({imod['fit_time']:.2f}s vs. {slipper['fit_time']:.2f}s, mean) "
        f"-- not worth its cost at the scale of the main comparison below."
    )
    return results, verdict


def write_report(results, datasets, prelim_results, prelim_verdict: str) -> None:
    task_by_name = {d.name: d.task for d in datasets}
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]

    lines = ["# RIPPER-family comparison: Weka:JRip / Witt:RIPPER / Pypper / Slipper\n\n"]
    lines.append(render_setup_section(_build_description(datasets, prelim_verdict)))
    lines.append(f"![accuracy vs. complexity, fit time per dataset]({os.path.basename(PLOT_PATH)})\n\n")
    lines.append(f"![critical-difference diagram (accuracy)]({os.path.basename(CD_PLOT_PATH)})\n\n")

    lines.append("## Why IMod:Slipper isn't in the main comparison\n\n")
    lines.append(f"{prelim_verdict}\n\n")
    lines.append(render_results_table(prelim_results, ["accuracy", "fit_time"], group_by="learner"))
    lines.append("\n")

    lines.append("## Per-dataset results (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"]))
    lines.append("\n")

    lines.append("## Mean ranks (accuracy, failures tied for last)\n\n")
    ranks = mean_rank(results, "accuracy")
    lines.append("| learner | mean rank |\n|---|--:|\n")
    for learner, rank in ranks.items():
        lines.append(f"| {learner} | {rank:.2f} |\n")
    lines.append("\n")

    binary_names = {d.name for d in datasets if task_by_name.get(d.name) == "binary"}
    multiclass_names = {d.name for d in datasets if task_by_name.get(d.name) == "multiclass"}
    if binary_names and multiclass_names:  # only worth a section when the run actually mixes both
        lines.append("## Mean accuracy by target type\n\n")
        lines.append("| learner | binary | multi-class |\n|---|--:|--:|\n")
        for learner in LEARNER_ORDER:
            sub = results[results["learner"] == learner]
            bin_acc = sub[sub["dataset"].isin(binary_names)]["accuracy"].mean()
            multi_acc = sub[sub["dataset"].isin(multiclass_names)]["accuracy"].mean()
            fmt = lambda v: f"{v:.3f}" if v == v else "n/a"  # noqa: E731 -- v == v is False for NaN
            lines.append(f"| {learner} | {fmt(bin_acc)} | {fmt(multi_acc)} |\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {REPORT_PATH}")


def write_plots(results) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for learner in LEARNER_ORDER:
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            ax1.scatter(sub["n_conditions"], sub["accuracy"], label=learner,
                       color=COLORS[learner], alpha=0.7, s=45)
    ax1.set_xlabel("total conditions (rule set complexity)")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("accuracy vs. rule-set complexity")
    ax1.set_xscale("symlog")
    ax1.legend()
    ax1.grid(alpha=0.3)

    names = sorted(per_pair["dataset"].unique())
    x = np.arange(len(names))
    w = 0.8 / len(LEARNER_ORDER)
    for i, learner in enumerate(LEARNER_ORDER):
        sub = per_pair[per_pair["learner"] == learner].set_index("dataset")
        ts = [sub["fit_time"].get(n, np.nan) for n in names]
        ax2.bar(x + (i - (len(LEARNER_ORDER) - 1) / 2) * w, ts, w, label=learner, color=COLORS[learner])
    ax2.set_yscale("log")
    ax2.set_ylabel("mean fit time (s, log)")
    ax2.set_title("fit time per dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {PLOT_PATH}")

    try:
        ax = critical_difference_diagram(results, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(CD_PLOT_PATH, dpi=110)
        plt.close(ax.figure)
        print(f"Plot   -> {CD_PLOT_PATH}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
        print(f"Critical-difference diagram skipped: {type(e).__name__}: {e}")


def main(datasets_spec: Optional[str] = None) -> None:
    print("Preliminary check: Slipper vs. IMod:Slipper ...")
    prelim_results, prelim_verdict = run_preliminary_slipper_check()
    print(prelim_verdict)

    datasets = select_datasets(datasets_spec)
    results = run_cv(build_learners(), datasets, n_folds=N_FOLDS, fit_timeout=FIT_TIMEOUT,
                     max_intervals=MAX_INTERVALS, random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
    write_report(results, datasets, prelim_results, prelim_verdict)
    write_plots(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=None,
                        help=f"Catalog.parse() spec overriding the default {DEFAULT_DATASETS_SPEC!r} "
                             "(44 datasets), e.g. 'small,medium' (83, binary and multi-class), "
                             "'all' (100 -- an overnight run) or 'vote,mushroom'")
    args = parser.parse_args()
    main(args.datasets)
