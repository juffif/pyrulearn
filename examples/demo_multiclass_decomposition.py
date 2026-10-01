"""
examples/demo_multiclass_decomposition.py
=========================================

Multi-class problems decomposed into two-class problems for a rule
learner: one-vs-rest against pairwise (round robin) classification, after
Fürnkranz, *Round Robin Classification* (JMLR 2002). The base learners
are Pypper (pyrulearn's RIPPER, standing in for the Ripper of the paper)
and PFossil, both fast.

Decompositions (`DECOMPOSITIONS`), each through the base learner's
`fit(data, model=...)`:

- **ovr**         -- one-vs-rest (`ConceptSet`): one "class vs. rest"
                     model per class, all on every example.
- **ordered**     -- ordered one-vs-rest (`ConceptCascade`): the classes
                     from least to most frequent, each against the larger
                     ones, the most frequent as default (Ripper's own).
- **pw_smaller**  -- pairwise (`PairwiseModel`): one model per pair of
                     classes, on those two classes' examples only, with
                     the pair's smaller class as the target.
- **pw_larger**   -- the same with the larger class as the target.
- **pw_both**     -- the double round robin: both classes of every pair
                     as the target, one model each.

The pairwise models are scored with three vote schemes on the same fitted
model (only prediction changes): **vote** (one vote per pair model),
**weighted_vote** (the vote split by the deciding rule's reliability) and
**accuracy_vote** (the vote split by the pair model's accuracy on its own
two classes).

Measures: accuracy, number of rules and conditions, and training and
prediction time separately. Each decomposition times its own fit inside
the worker process, so the times don't include shipping data or models
between processes. The main plot is the ratio of one-vs-rest to pairwise
training time over the number of classes: one-vs-rest learns from c * n
examples in total, pairwise from (c - 1) * n, so for a learner whose time
grows linearly with the examples the ratio is c / (c - 1), and above that
for one that grows faster.

Run: `python examples/demo_multiclass_decomposition.py` with no arguments
is the **quick** default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its
report/plots go to ``demo_multiclass_decomposition_quick_*`` and are not
checked in. `--full` runs `FULL_DATASETS` (`N_FOLDS`-fold, `LARGE_FOLDS`
for the large `kropt` and `letter`) and writes the canonical
``demo_multiclass_decomposition_report.md`` (plots in
``demo_multiclass_decomposition_plots/``). Needs the `experiments` extra.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.seco import PFossil, Pypper
from pyrulearn.models import ConceptCascade, ConceptSet, PairwiseModel, _resolve_pairwise_combiner

RANDOM_STATE = 0
N_FOLDS = 10
LARGE_FOLDS = 5
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0

BASES = ("Pypper", "PFossil")
DECOMPOSITIONS = ("ovr", "ordered", "pw_smaller", "pw_larger", "pw_both")
VOTES = ("vote", "weighted_vote", "accuracy_vote")

# small datasets with many classes (mostly symbolic, except vowel)
QUICK_DATASETS = ["audiology", "primary-tumor", "soybean", "vowel", "zoo"]
QUICK_FOLDS = 3
# plus medium ones, and the two large ones with many classes
FULL_DATASETS = QUICK_DATASETS + ["solar-flare", "segment", "yeast", "led24", "optdigits", "texture",
                                  "kropt", "letter"]

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "demo_multiclass_decomposition"
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, "_multiclass_decomposition_cache")


class Decomposition(NativeRuleLearner):
    """`base` decomposed by `method`, named ``"<base>:<method>"``; records
    the time of its own fit on the returned model (`fit_seconds`)."""

    def __init__(self, base: str, method: str, random_state: int = RANDOM_STATE):
        self.base = base
        self.method = method
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return f"{self.base}:{self.method}"

    def _provenance_params(self):
        return {"base": self.base, "method": self.method, "random_state": self.random_state}

    def fit(self, data):
        learner = {"Pypper": Pypper, "PFossil": PFossil}[self.base](random_state=self.random_state)
        kwargs = {
            "ovr": {"model": ConceptSet},
            "ordered": {"model": ConceptCascade},
            "pw_smaller": {"model": PairwiseModel, "positive": "smaller"},
            "pw_larger": {"model": PairwiseModel, "positive": "larger"},
            "pw_both": {"model": PairwiseModel, "positive": "both"},
        }[self.method]
        t0 = time.perf_counter()
        model = learner.fit(data, **kwargs)
        model.fit_seconds = time.perf_counter() - t0
        return model


def build_learners():
    return [Decomposition(base, method) for base in BASES for method in DECOMPOSITIONS]


def measure(model, test_rep) -> dict:
    """Training time from inside the worker, and prediction time and
    accuracy per vote scheme (pairwise) or for the model as it is."""
    y = np.asarray(test_rep.y)
    out = {"train_time": getattr(model, "fit_seconds", np.nan),
           "n_models": len(getattr(model, "members", []) or [])}
    if isinstance(model, PairwiseModel):
        for vote in VOTES:
            model.combiner = _resolve_pairwise_combiner(vote)
            t0 = time.perf_counter()
            pred = np.asarray(model.predict(test_rep))
            out[f"predict_time_{vote}"] = time.perf_counter() - t0
            out[f"accuracy_{vote}"] = float(np.mean(pred == y))
    else:
        t0 = time.perf_counter()
        model.predict(test_rep)
        out["predict_time"] = time.perf_counter() - t0
    return out


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, fold, variant): the pairwise results expanded
    into one row per vote scheme, with that scheme's accuracy and
    prediction time."""
    rows = []
    for _, r in results.iterrows():
        base, method = r["learner"].split(":")
        common = {"dataset": r["dataset"], "fold": r["fold"], "base": base, "method": method,
                  "n_rules": r["n_rules"], "n_conditions": r["n_conditions"],
                  "train_time": r.get("train_time"), "error": r["error"]}
        if method.startswith("pw_"):
            for vote in VOTES:
                rows.append({**common, "learner": f"{base}:{method}:{vote}", "variant": f"{method}:{vote}",
                             "accuracy": r.get(f"accuracy_{vote}"),
                             "predict_time": r.get(f"predict_time_{vote}")})
        else:
            rows.append({**common, "learner": r["learner"], "variant": method,
                         "accuracy": r["accuracy"], "predict_time": r.get("predict_time")})
    return pd.DataFrame(rows)


def _ratios(results: pd.DataFrame, n_classes: dict, pairwise: str, value: str) -> pd.DataFrame:
    """Per dataset and base learner: mean one-vs-rest time / mean pairwise time."""
    ok = results[results["error"].isna()]
    means = ok.groupby(["dataset", "learner"])[value].mean()
    rows = []
    for (ds, learner), v in means.items():
        base, method = learner.split(":")
        if method == pairwise and (ds, f"{base}:ovr") in means.index and v > 0:
            rows.append({"dataset": ds, "base": base, "classes": n_classes[ds],
                         "ratio": means[(ds, f"{base}:ovr")] / v})
    return pd.DataFrame(rows)


def write_plots(results: pd.DataFrame, n_classes: dict, quick: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    os.makedirs(PLOTS_DIR, exist_ok=True)
    tag = "_quick" if quick else ""
    paths = {}
    markers = {"Pypper": ("o", "tab:blue"), "PFossil": ("s", "tab:orange")}
    specs = [
        ("train_ratio", "train_time", "pw_smaller", "training time: one-vs-rest / pairwise",
         lambda c: c / (c - 1), "c / (c - 1): learner linear in the examples"),
        ("train_ratio_double", "train_time", "pw_both", "training time: one-vs-rest / double round robin",
         lambda c: c / (2 * (c - 1)), "c / (2(c - 1)): learner linear in the examples"),
        ("predict_ratio", "predict_time", "pw_smaller", "prediction time: one-vs-rest / pairwise (vote)",
         None, None),
    ]
    pw_results = results.copy()
    pw_results["predict_time"] = pw_results["predict_time"].fillna(pw_results.get("predict_time_vote"))
    for key, value, pairwise, title, reference, ref_label in specs:
        ratios = _ratios(pw_results, n_classes, pairwise, value)
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for base, (marker, color) in markers.items():
            sub = ratios[ratios["base"] == base]
            ax.scatter(sub["classes"], sub["ratio"], marker=marker, color=color, label=base, s=40)
            for _, r in sub.iterrows():
                ax.annotate(r["dataset"], (r["classes"], r["ratio"]), fontsize=6,
                            xytext=(3, 2), textcoords="offset points")
        if reference is not None and len(ratios):
            cs = np.linspace(max(2, ratios["classes"].min()), ratios["classes"].max(), 100)
            ax.plot(cs, [reference(c) for c in cs], "--", color="gray", label=ref_label)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_yscale("log")
        # labels at 1, 2, 5 of every decade, as plain numbers; none on the minor ticks
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("number of classes")
        ax.set_ylabel("ratio (above 1: pairwise faster)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{NAME}{tag}_{key}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths[key] = path
        print(f"Plot   -> {path}")
    return paths


def _description(datasets, quick: bool, n_folds: int) -> str:
    large = [d.name for d in datasets if d.size == "large"]
    folds = (f"{n_folds}-fold stratified cross-validation" if not large else
             f"{n_folds}-fold stratified cross-validation ({LARGE_FOLDS}-fold for the large "
             f"{', '.join(large)})")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: -d.n_classes))
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python examples/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python examples/{NAME}.py`.\n\n")
    return f"""\
{mode}A rule learner that learns rules for one class against the rest has to
decompose a problem with several classes into two-class problems. This
demo compares the ways of doing that in pyrulearn, with Pypper
(pyrulearn's RIPPER) and PFossil as base learners -- in essence the
experiments of Fürnkranz, *Round Robin Classification* (JMLR 2002), with
Pypper standing in for Ripper:

- **ovr** -- one-vs-rest: one model per class, "this class vs. all
  others", every model trained on all examples; conflicts resolved by the
  best rule.
- **ordered** -- ordered one-vs-rest: classes from least to most frequent,
  each learned against the classes not yet handled; the most frequent
  class is the default. This is Ripper's (and Pypper's) own strategy.
- **pw_smaller / pw_larger** -- pairwise (round robin): one model per pair
  of classes, trained on those two classes' examples only, with the
  pair's smaller (larger) class as the target.
- **pw_both** -- the double round robin: for every pair, a model for each
  of its two classes.

A pairwise model is scored with three vote schemes on the same fitted
model: **vote** (each pair model votes for one class), **weighted_vote**
(the vote is split by the reliability of the rule that decided it) and
**accuracy_vote** (the vote is split by the pair model's accuracy on its
own two classes). Only prediction changes, so the three share their
training.

**Why training time is interesting.** Pairwise learns many more models --
c(c-1)/2 for c classes -- but each on few examples: every example is used
in c - 1 pair models, so pairwise learns from (c - 1) * n examples in
total, against c * n for one-vs-rest. For a learner whose time grows
linearly with the examples, one-vs-rest therefore takes about c / (c - 1)
times as long; for one that grows faster, pairwise gains more with every
class. Prediction goes the other way: every pair model has to be asked.

Datasets, with their numbers of classes: {names}. Protocol: {folds}, one
`DataSpec` per training fold (`build_dataspec(max_intervals={MAX_INTERVALS})`).
Each fit is capped at {FIT_TIMEOUT:.0f}s; a time-out counts as a failure.
Training time is measured inside the worker process, so it excludes
shipping data and models between processes.
"""


def write_report(results, var, datasets, paths, quick: bool, n_folds: int, report_path: str) -> None:
    plots = os.path.basename(PLOTS_DIR)
    img = lambda k, alt: f"![{alt}]({plots}/{os.path.basename(paths[k])})\n\n"  # noqa: E731
    L = ["# Multi-class decomposition: one-vs-rest vs. pairwise\n\n"]
    L.append(render_setup_section(_description(datasets, quick, n_folds)))

    for base in BASES:
        v = var[var["base"] == base]
        ranks = mean_rank(v, "accuracy")
        means = v.groupby("learner")[["accuracy", "n_rules", "n_conditions", "train_time",
                                      "predict_time"]].mean(numeric_only=True)
        failures = v.groupby("learner")["error"].apply(lambda e: e.notna().sum())
        L.append(f"## {base}\n\n")
        L.append("| variant | accuracy | mean rank | rules | conditions | train (s) | predict (s) | failures |\n"
                 "|---|--:|--:|--:|--:|--:|--:|--:|\n")
        for learner in ranks.sort_values().index:
            m = means.loc[learner]
            L.append(f"| {learner.split(':', 1)[1]} | {m['accuracy']:.3f} | {ranks[learner]:.2f} | "
                     f"{m['n_rules']:.1f} | {m['n_conditions']:.1f} | {m['train_time']:.2f} | "
                     f"{m['predict_time']:.3f} | {int(failures.get(learner, 0))} |\n")
        L.append("\nSorted by mean rank (accuracy rank across datasets, failures last). Rules and "
                 "conditions: all of a decomposition's models together. The three vote schemes of a "
                 "pairwise decomposition share its models and training time.\n\n")

    L.append("## Training and prediction time over the number of classes\n\n")
    L.append("One point per dataset and base learner: the mean one-vs-rest time divided by the "
             "mean pairwise time. Above 1, pairwise is faster. The dashed line is what a learner "
             "whose time grows linearly with the number of examples would give.\n\n")
    L.append(img("train_ratio", "training time ratio, one-vs-rest / pairwise"))
    L.append(img("train_ratio_double", "training time ratio, one-vs-rest / double round robin"))
    L.append(img("predict_ratio", "prediction time ratio, one-vs-rest / pairwise"))

    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var.assign(learner=var["learner"]),
                                  ["accuracy", "n_conditions", "train_time", "predict_time"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    names = QUICK_DATASETS if quick else FULL_DATASETS
    datasets = Catalog.default().select(names=names)
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    groups = [(datasets, n_folds)] if quick else [
        ([d for d in datasets if d.size != "large"], N_FOLDS),
        ([d for d in datasets if d.size == "large"], LARGE_FOLDS),
    ]
    results = pd.concat([
        run_cv(build_learners(), group, n_folds=folds, fit_timeout=FIT_TIMEOUT,
               max_intervals=MAX_INTERVALS, random_state=RANDOM_STATE, cache_dir=CACHE_DIR,
               measure_fn=measure)
        for group, folds in groups if group
    ], ignore_index=True)
    var = variants(results)
    n_classes = {d.name: d.n_classes for d in datasets}
    paths = write_plots(results, n_classes, quick)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(results, var, datasets, paths, quick, n_folds, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="all datasets, including the large kropt and letter (5-fold)")
    main(quick=not parser.parse_args().full)
