"""
demos/native_vs_interfaced.py
=============================

Four native rule learners against the external reference implementations
they are modeled on, cross-validated through
`pyrulearn.experiments.runner.run_cv` on binary datasets of
`pyrulearn.experiments.catalog`:

- **RuleFit**      -- `pyrulearn.learners.rulefit.RuleFit` (a sparse
                      logistic regression over a rule pool's coverage).
                      Unlike the original RuleFit (and unlike IMod:RuleFit
                      below), it doesn't grow its own tree-ensemble pool
                      (not yet implemented -- see the module docstring);
                      left at its default it falls back to mining class
                      association rules, the same combinatorial miner
                      `CBA` uses, which is not what either RuleFit
                      actually is. This demo gives it a random forest's
                      leaves as its pool instead (`RuleFitRF` below,
                      `SKLRandomForest` + `rules=`) -- closer in kind to
                      a tree ensemble, and far cheaper than the
                      association-rule fallback (measured: 0.57s vs.
                      65.5s on one dataset).
                      vs. **IMod:RuleFit** -- `imodels.RuleFitClassifier`
                      (gradient-boosted trees as the pool). The two also
                      disagree on rule count for a different reason, left
                      unmatched: IMod:RuleFit caps its final model at
                      `max_rules=30` by searching the regularization path
                      for the sparsest fit under that ceiling; native
                      RuleFit's own knob (`C`, optionally `cv`-selected)
                      has no such cap, and neither a smaller fixed `C` nor
                      `cv=` reliably reproduces the 30-rule ceiling across
                      datasets -- see `LEARNER_NOTES["RuleFit"]`.
- **Boomer**        -- `pyrulearn.learners.boosting.Boomer` (ENDER with
                      BOOMER's logistic loss and Newton steps) vs.
                      **MLRL:Boomer** -- the reference `mlrl-boomer`
                      package, with its instance/feature sampling turned
                      off (`instance_sampling="none"`,
                      `feature_sampling="none"`) to match native Boomer's
                      full-data search -- left at its own defaults it
                      settles on ~4x the rules and conditions of native
                      Boomer for the same accuracy (measured: 534
                      rules/1942 conditions vs. 131/474 on one dataset),
                      an artifact of unmatched sampling, not a difference
                      worth showing.
- **CBA**           -- `pyrulearn.learners.associative.CBA` vs.
                      **PArc:CBA** -- the reference `pyarc` package (its
                      defaults already mirror the native ones, and the
                      two are cross-checked rule-for-rule in
                      `tests/test_pyarc_import.py`).
- **Pypper**        -- `pyrulearn.learners.seco.Pypper` (IREP*
                      growth-and-pruning plus `ReplaceReviseOptimization`)
                      vs. **Weka:JRip** -- Weka's reference RIPPER
                      implementation, via a subprocess
                      (`pyrulearn.interfaces.weka.WekaJRip`). Also
                      compared at a larger scale, and against
                      wittgenstein's RIPPER too, in
                      `demos/ripper_comparison.py`.

A fifth pair, **ORB** vs. **RKD:RuleBoosting**
(`pyrulearn.learners.boosting.ORB`, optimal rule boosting with the
library's branch-and-bound rule search, vs. the reference `realkd`
package), is checked
separately first on a couple of small, low-feature-count datasets, then
left out of the main comparison below -- both implementations' exhaustive
search stops scaling to this demo's wider, one-hot-encoded feature
counts well before either's accuracy or model size would be informative;
see "Why optimal rule boosting isn't in the main comparison"
(`run_preliminary_optimal_rule_boosting_check`).

Two more native/external pairs exist and are deliberately **not**
repeated here, each already covered by its own demo:

- **Slipper** vs. **IMod:Slipper** -- `ripper_comparison.py`'s own
  preliminary check (`run_preliminary_slipper_check`).
- **PyLORD** vs. the reference **JavaLord** --
  `demos/seco_learners_comparison.py`.

Every pair's two learners differ in implementation details (not
byte-identical rules -- unlike the four data representations, this
isn't the same computation twice), so the question here is practical
agreement: how close in accuracy and model size, and how much faster or
slower. RuleFit and Boomer are binary-only (their model is a
`LinearRuleModel`, which doesn't decompose into the `ConceptModel`s a
multiclass switcher builds) -- so this comparison stays binary
throughout, CBA and Pypper/Weka:JRip (both natively multiclass) included,
for one shared protocol.

Run: `python demos/native_vs_interfaced.py` with no arguments is
the **quick** default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its
report/plots go to ``native_vs_interfaced_quick_*`` and are not
checked in. `--full` runs `FULL_DATASETS` (`N_FOLDS`-fold, `LARGE_FOLDS`
for the large `adult`) and writes the canonical
``native_vs_interfaced_report.md`` (plots in
``native_vs_interfaced_plots/``). Needs the `boomer`, `pyarc` and
`realkd` extras (see their module docstrings for installing them on
Windows) in addition to `experiments` and `imodels`, plus a local Weka
install for `Weka:JRip` (`WEKA_JAR`/`WEKA_JAVA` below -- see
`demos/ripper_comparison.py`'s module docstring for installing Weka).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank, win_counts
from pyrulearn.interfaces.boomer import MLRLBoomer
from pyrulearn.interfaces.imodels import IModRuleFit
from pyrulearn.interfaces.pyarc import PArcCBA
from pyrulearn.interfaces.realkd import RKDRuleBoosting
from pyrulearn.interfaces.sklearn import SKLRandomForest
from pyrulearn.interfaces.weka import WekaJRip
from pyrulearn.learners.associative import CBA
from pyrulearn.learners.boosting import Boomer, ORB
from pyrulearn.learners.rulefit import RuleFit
from pyrulearn.learners.seco import BranchAndBoundSearch, Pypper
from pyrulearn.models import FlatRuleSet

RANDOM_STATE = 0
# 5 (not the usual 10) and 3 (not 5) for the large adult -- this demo's own
# pairs run far slower per fit than most (CBA/PArc:CBA's itemset mining,
# Weka:JRip's subprocess), so the usual fold counts would push a full run
# to several hours; this keeps it at roughly the other full demos' 1-2h
# (measured: ~53min for the 15 small/medium datasets at 5-fold, ~51min
# for adult at 3-fold, both from real single-fit timings with FIT_TIMEOUT
# correctly capping any run that exceeds it -- see FULL_DATASETS below).
N_FOLDS = 5
LARGE_FOLDS = 3
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0

# same Weka install as demos/ripper_comparison.py uses
WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

# itemset mining (CBA/PArc:CBA) grows fast with the number of binary
# features; RuleFit's own pool is a random forest's leaves instead (see
# RuleFitRF below), not subject to this.
QUICK_DATASETS = ["vote", "tic-tac-toe", "hepatitis", "heart-statlog", "SPECT"]
QUICK_FOLDS = 3
QUICK_MAX_INTERVALS = 3  # fewer binary features -> faster, for a sanity check, not final numbers

# molecular-biology_promoters (456 binary features at MAX_INTERVALS=8) and
# cylinder-bands (480) are excluded here -- every other learner fits them
# in seconds, but CBA/PArc:CBA's itemset mining doesn't scale to that many
# one-hot features: measured single-fit times of 730s/744s and 242s/458s,
# both past FIT_TIMEOUT, for no informative result (both implementations
# fail the same way -- not a native-vs-reference difference, same reasoning
# as ORB/RKD:RuleBoosting above).
FULL_DATASETS = [
    "vote", "breast-cancer", "tic-tac-toe", "hepatitis", "heart-statlog", "SPECT",
    "colic", "credit-approval", "dresses-sales",
    "heart-h", "heart-c", "credit-g", "kr-vs-kp", "sick", "mushroom", "adult",
]

# ORB/RKD:RuleBoosting preliminary check only (see
# run_preliminary_optimal_rule_boosting_check) -- both time out well
# before this on wider data (e.g. all 3 folds of breast-cancer, ~30
# minutes), a shared limit of exhaustive branch-and-bound search, not a
# native-vs-reference difference.
PRELIM_DATASETS = ["vote", "tic-tac-toe"]
PRELIM_FOLDS = 3

# CBA/PArc:CBA mine itemsets up to max_len; 3 instead of the default 4 is
# ~10-15x faster (measured on breast-cancer/hepatitis) at no real accuracy
# cost, while 2 starts costing accuracy -- see ROADMAP.md
ITEMSET_MAX_LEN = 3
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 4


class RuleFitRF(RuleFit):
    """Native `RuleFit` with a random forest's leaves as its candidate
    pool (`SKLRandomForest` + `RuleDistiller`'s `rules=`), fit fresh per
    call -- closer in kind to the original RuleFit's own gradient-boosted-
    tree pool (not yet implemented natively, see `pyrulearn.learners.
    rulefit`'s module docstring) than the generic class-association-rule
    fallback `RuleFit()` uses undirected. `display_name` stays "RuleFit"
    so this still pairs with `IMod:RuleFit` in the report."""

    def __init__(self, n_estimators: int = RF_N_ESTIMATORS, max_depth: int = RF_MAX_DEPTH,
                random_state: Optional[int] = None, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        super().__init__(random_state=random_state, **kwargs)

    @property
    def display_name(self) -> str:
        return "RuleFit"

    def fit(self, data, model=None, **model_kwargs):
        self.rules = SKLRandomForest(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                     random_state=self.random_state).fit(data, model=FlatRuleSet)
        return super().fit(data, model=model, **model_kwargs)


# pair key -> (native class, native kwargs, interfaced class, interfaced kwargs)
# ORB/RKDRuleBoosting are deliberately not here -- see
# run_preliminary_optimal_rule_boosting_check and the module docstring.
#
# MLRLBoomer's instance_sampling/feature_sampling default to BOOMER's own
# subsampling (unlike native Boomer, which always sees every row and
# feature, matching MLRL's shrinkage/l2/loss already); left at their
# defaults it ends up with ~4x the rules and conditions of native Boomer
# for the same accuracy (measured: 534 rules/1942 conditions vs. native's
# 131/474 on one dataset) -- "none"/"none" matches native's full-data
# search and brings it down to 116/423, now comparable.
PAIRS = {
    "RuleFit": (RuleFitRF, {"random_state": RANDOM_STATE}, IModRuleFit, {"random_state": RANDOM_STATE}),
    "Boomer": (Boomer, {"random_state": RANDOM_STATE}, MLRLBoomer,
              {"instance_sampling": "none", "feature_sampling": "none"}),
    "CBA": (CBA, {"max_len": ITEMSET_MAX_LEN}, PArcCBA, {"max_len": ITEMSET_MAX_LEN}),
    "Pypper": (Pypper, {"random_state": RANDOM_STATE}, WekaJRip, {"jar": WEKA_JAR, "java": WEKA_JAVA}),
}

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "native_vs_interfaced"
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, "_native_vs_interfaced_cache")

LEARNER_NOTES = {
    "RuleFit": "A sparse (L1) logistic regression over a random forest's leaves "
               f"({RF_N_ESTIMATORS} trees, depth {RF_MAX_DEPTH}) -- the candidate pool native "
               "RuleFit doesn't yet grow itself (see the module docstring). Its own sparsity knob "
               "is a plain regularization strength (`C`, optionally `cv`-selected by predictive "
               "score) with no cap on the final rule count -- unlike IMod:RuleFit (right), it "
               "ends up keeping noticeably more, smaller rules for similar accuracy (measured: "
               "70.8 rules/268.7 conditions vs. 25.3/59.6, mean). Neither a smaller fixed `C` nor "
               "`cv=` reproduces IMod:RuleFit's rule count reliably across datasets (tested: "
               "`cv=5` went from 67->28 rules on one dataset but 29->39, the wrong way, on "
               "another) -- it's a different sparsity *mechanism*, not a mismatched default, so "
               "left unmatched here.",
    "IMod:RuleFit": "imodels' RuleFit: gradient-boosted trees as the rule pool, then a sparse "
                    "regression over their leaves, deliberately capped at `max_rules=30` -- it "
                    "searches the regularization path for the sparsest fit under that ceiling "
                    "(`get_best_alpha_under_max_rules`), rather than picking strength by "
                    "predictive score alone.",
    "Boomer": "ENDER configured as BOOMER's single-label case: logistic loss, L2-regularized "
             "Newton steps, up to 1,000 rules.",
    "MLRL:Boomer": "The reference BOOMER implementation (`mlrl-boomer`), with its instance/feature "
                  "sampling turned off to match native Boomer's full-data search (see PAIRS above).",
    "ORB": "Optimal rule boosting: each round's rule is the exact best by branch-and-bound "
           "search (10 rounds by default), not grown greedily.",
    "RKD:RuleBoosting": "The reference optimal-rule-boosting implementation (`realkd`).",
    "CBA": "Classification Based on Associations: mine class association rules (here capped at "
          f"max_len={ITEMSET_MAX_LEN}, see ROADMAP.md), sort by precedence, select by database "
          "coverage (CBA-CB, \"M1\").",
    "PArc:CBA": "The reference CBA implementation (`pyarc`), mining with Borgelt's `fim`; its "
               "defaults mirror the native CBA's.",
    "Pypper": "IREP* growth-and-pruning plus `ReplaceReviseOptimization`, run per class -- a "
             "re-implementation of RIPPER, not a port.",
    "Weka:JRip": "Weka's reference RIPPER implementation, via a subprocess "
                "(`pyrulearn.interfaces.weka.WekaJRip`); also compared at a larger scale, and "
                "against wittgenstein's RIPPER too, in `demos/ripper_comparison.py`.",
}


def build_learners():
    learners = []
    for native_cls, native_kw, ext_cls, ext_kw in PAIRS.values():
        learners.append(native_cls(**native_kw))
        learners.append(ext_cls(**ext_kw))
    return learners


# {learner display_name -> pair key}, {pair key -> (native name, interfaced name)} -- built from
# the actual instantiated learners, not guessed from their names (RKDRuleBoosting's TOOL-stripped
# name is "RuleBoosting", not the pair key "ORB")
def _pair_index(learners):
    name_to_key, native_name, ext_name = {}, {}, {}
    i = 0
    for key in PAIRS:
        n, e = learners[i], learners[i + 1]
        name_to_key[n.display_name] = key
        name_to_key[e.display_name] = key
        native_name[key] = n.display_name
        ext_name[key] = e.display_name
        i += 2
    return name_to_key, native_name, ext_name


PAIR_OF_NAME, NATIVE_NAME, EXT_NAME = _pair_index(build_learners())


def _pair_of(learner_name: str) -> str:
    return PAIR_OF_NAME[learner_name]


def run_preliminary_optimal_rule_boosting_check():
    """Fits native `ORB` against `RKD:RuleBoosting` on
    `PRELIM_DATASETS`, and returns `(results, verdict)`: the usual
    long-format `run_cv` table (just these two learners, these datasets)
    and a plain-English sentence built from the actual numbers (never a
    canned claim), for both the report and the module docstring. Both
    learners' exhaustive branch-and-bound search times out well before
    this on wider data (see the module docstring), so this pair is kept
    out of the main comparison and checked here instead, at a scale
    where both can actually finish."""
    datasets = Catalog.default().select(names=PRELIM_DATASETS)
    # realkd's exhaustive search (its default) vs. ORB with the matching search
    results = run_cv([ORB(search=BranchAndBoundSearch()), RKDRuleBoosting()], datasets, n_folds=PRELIM_FOLDS,
                     fit_timeout=FIT_TIMEOUT, max_intervals=MAX_INTERVALS, random_state=RANDOM_STATE,
                     cache_dir=CACHE_DIR)
    by_learner = results.groupby("learner").agg(accuracy=("accuracy", "mean"), fit_time=("fit_time", "mean"))
    native, ext = by_learner.loc["ORB"], by_learner.loc["RKD:RuleBoosting"]
    acc_gap = abs(native["accuracy"] - ext["accuracy"])
    agree = "closely" if acc_gap < 0.02 else "reasonably" if acc_gap < 0.05 else "not closely"
    verdict = (
        f"Across {len(datasets)} small datasets ({PRELIM_FOLDS}-fold), ORB and "
        f"RKD:RuleBoosting agreed {agree} on accuracy (mean {native['accuracy']:.3f} vs. "
        f"{ext['accuracy']:.3f}, {acc_gap:.3f} apart; mean fit time {native['fit_time']:.2f}s vs. "
        f"{ext['fit_time']:.2f}s) -- but neither's exhaustive search scales to the main comparison's "
        f"wider, one-hot-encoded feature counts within the {FIT_TIMEOUT:.0f}s cap, a limit of the "
        f"exhaustive-search approach itself rather than of either implementation, so the pair is "
        f"checked here instead of in the main table below."
    )
    return results, verdict


def _build_description(datasets, n_folds: int, quick: bool, learner_names, prelim_verdict: str) -> str:
    mode_note = (
        f"**Quick run** ({len(datasets)} small datasets, {n_folds}-fold) -- a fast sanity "
        f"check, the default with no arguments. Full comparison ({len(FULL_DATASETS)} "
        f"datasets): `python demos/{NAME}.py --full`.\n\n" if quick else
        f"**Full run** ({len(datasets)} datasets). Quick sanity check instead: "
        f"`python demos/{NAME}.py` with no arguments.\n\n"
    )
    large = [d.name for d in datasets if d.size == "large"]
    folds_note = (f"{n_folds}-fold stratified cross-validation" if quick or not large else
                 f"{n_folds}-fold stratified cross-validation ({LARGE_FOLDS}-fold for the "
                 f"large {', '.join(large)})")
    lines = []
    for key in PAIRS:
        ext_name = EXT_NAME[key]
        lines.append(f"- **{key}** -- {LEARNER_NOTES[key]}\n  **{ext_name}** -- {LEARNER_NOTES[ext_name]}")
    pair_lines = "\n".join(lines)
    return f"""\
{mode_note}Four native rule learners against the external reference implementations
they are modeled on, on {len(datasets) - len(large)} small/medium and
{len(large)} large binary datasets from `pyrulearn.experiments.catalog`.
Two more pairs are covered by their own demos instead: Slipper vs.
IMod:Slipper (`ripper_comparison.py`'s own preliminary check), and PyLORD
vs. the reference JavaLord (`demos/seco_learners_comparison.py`). A
fifth pair, ORB vs. RKD:RuleBoosting, is checked
separately below instead of in the main comparison -- see "Why optimal
rule boosting isn't in the main comparison". {prelim_verdict}

{pair_lines}

Protocol: {folds_note} (`pyrulearn.experiments.runner.run_cv`), one
`DataSpec` per training fold (`build_dataspec(max_intervals={QUICK_MAX_INTERVALS if quick else MAX_INTERVALS})`),
the test fold binarized against that same `DataSpec`. Each fit is capped
at {FIT_TIMEOUT:.0f}s; a time-out or an error counts as a failure (and
as last in the ranking), not as the end of the run -- CBA/PArc:CBA's
itemset mining is the likeliest to hit this on wider data.

Measures: test accuracy, number of rules and conditions, fit time.
"""


def _safe_pair(name: str) -> Optional[str]:
    return PAIR_OF_NAME.get(name)


def _pair_table(results: pd.DataFrame) -> list:
    summary = results.groupby("learner")[["accuracy", "n_rules", "n_conditions", "fit_time"]].mean(
        numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    lines = ["| pair | learner | accuracy | n_rules | n_conditions | fit_time (s) | failures |\n"
             "|---|---|--:|--:|--:|--:|--:|\n"]
    for key in PAIRS:
        for name in (NATIVE_NAME[key], EXT_NAME[key]):
            if name not in summary.index:
                continue
            row = summary.loc[name]
            lines.append(f"| {key} | {name} | {row['accuracy']:.3f} | {row['n_rules']:.1f} | "
                        f"{row['n_conditions']:.1f} | {row['fit_time']:.2f} | "
                        f"{int(failures.get(name, 0))} |\n")
    return lines


def write_report(results, datasets, n_folds: int, report_path: str, plots: dict, quick: bool,
                 learner_names, prelim_results, prelim_verdict: str) -> None:
    large_names = {d.name for d in datasets if d.size == "large"}
    large = results[results["dataset"].isin(large_names)]
    results = results[~results["dataset"].isin(large_names)]
    plots_dir = os.path.basename(PLOTS_DIR)

    L = ["# Native rule learners vs. their external references\n\n"]
    L.append(render_setup_section(_build_description(datasets, n_folds, quick, learner_names, prelim_verdict)))
    L.append(f"![accuracy vs. rule-set complexity]({plots_dir}/{os.path.basename(plots['accuracy'])})\n\n")
    L.append(f"![fit time, native vs. interfaced]({plots_dir}/{os.path.basename(plots['fit_time'])})\n\n")

    L.append("## Why optimal rule boosting isn't in the main comparison\n\n")
    L.append(f"{prelim_verdict}\n\n")
    L.append(render_results_table(prelim_results, ["accuracy", "fit_time"], group_by="learner",
                                  include_overall=False))  # averaging across two different
    # algorithms isn't a meaningful "overall" -- see render_results_table's own docstring
    L.append("\n")

    L.append(f"## Native vs. interfaced, pair by pair (mean across the "
             f"{results['dataset'].nunique()} small/medium datasets and their folds)\n\n")
    L += _pair_table(results)
    L.append("\n")

    L.append("## Summary by learner (mean rank across the same datasets)\n\n")
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[["accuracy", "n_rules", "n_conditions", "fit_time"]].mean(
        numeric_only=True)
    failures = results.groupby("learner")["error"].apply(lambda e: e.notna().sum())
    L.append("| learner | accuracy | n_rules | n_conditions | fit_time (s) | wins | mean rank | "
             "failures |\n|---|--:|--:|--:|--:|--:|--:|--:|\n")
    for learner in ranks.sort_values().index:
        row = summary.loc[learner]
        L.append(f"| {learner} | {row['accuracy']:.3f} | {row['n_rules']:.1f} | "
                f"{row['n_conditions']:.1f} | {row['fit_time']:.2f} | {wins.get(learner, 0.0):.1f} | "
                f"{ranks[learner]:.2f} | {int(failures.get(learner, 0))} |\n")
    L.append("\nSorted by mean rank. `wins` -- datasets where a learner's mean accuracy was "
            "(tied-for-)best, a tie split evenly; `mean rank` -- average accuracy rank across "
            "datasets, failures tied for last.\n\n")

    if len(large):
        L.append(f"## The large dataset: {', '.join(sorted(large_names))}\n\n")
        L.append(f"{LARGE_FOLDS}-fold cross-validation, same {FIT_TIMEOUT:.0f}s cap. `folds` counts "
                "the folds that finished; the other columns are means over those folds only.\n\n")
        L.append("| learner | folds | accuracy | n_rules | n_conditions | fit_time (s) |\n"
                "|---|--:|--:|--:|--:|--:|\n")
        for learner, g in large.groupby("learner"):
            ok = g[g["error"].isna()]
            cells = ([f"{ok['accuracy'].mean():.3f}", f"{ok['n_rules'].mean():.1f}",
                     f"{ok['n_conditions'].mean():.1f}", f"{ok['fit_time'].mean():.2f}"]
                    if len(ok) else ["--"] * 4)
            L.append(f"| {learner} | {len(ok)}/{len(g)} | " + " | ".join(cells) + " |\n")
        L.append("\n")

    L.append("## Per-dataset results, per learner (mean across folds)\n\n")
    L.append(render_results_table(results, ["accuracy", "n_rules", "n_conditions", "fit_time"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def write_plots(results: pd.DataFrame, learner_names, quick: bool) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = f"{NAME}_{'quick_' if quick else ''}"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors = dict(zip(PAIRS, plt.get_cmap("tab10").colors))
    # one shape per pair as well as one color; filled = native, open = interfaced
    markers = dict(zip(PAIRS, ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]))
    per_pair = results.groupby(["dataset", "learner"], as_index=False).mean(numeric_only=True)
    paths = {}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key in PAIRS:
        native_pts = per_pair[per_pair["learner"] == NATIVE_NAME[key]].set_index("dataset")
        ext_pts = per_pair[per_pair["learner"] == EXT_NAME[key]].set_index("dataset")
        for d in native_pts.index.intersection(ext_pts.index):
            n, e = native_pts.loc[d], ext_pts.loc[d]
            if pd.isna(n["accuracy"]) or pd.isna(e["accuracy"]):
                continue
            ax.plot([n["n_conditions"], e["n_conditions"]], [n["accuracy"], e["accuracy"]],
                   color=colors[key], alpha=0.35, linewidth=1, zorder=1)
    for learner in learner_names:
        key = _safe_pair(learner)
        if key is None:
            continue
        native = learner == NATIVE_NAME[key]
        sub = per_pair[per_pair["learner"] == learner].dropna(subset=["accuracy"])
        if len(sub):
            ax.scatter(sub["n_conditions"], sub["accuracy"], label=learner, color=colors[key],
                      marker=markers[key], facecolors=colors[key] if native else "none",
                      edgecolors=colors[key], s=45, alpha=0.8, zorder=2)
    ax.set_xscale("symlog")
    ax.set_xlabel("total conditions (rule set complexity)")
    ax.set_ylabel("test accuracy")
    ax.set_title("accuracy vs. complexity (filled = native, open = interfaced,\n"
                "line = same dataset)")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths["accuracy"] = os.path.join(PLOTS_DIR, f"{stem}accuracy.png")
    fig.savefig(paths["accuracy"], dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {paths['accuracy']}")

    means = results.groupby("learner")["fit_time"].mean(numeric_only=True)
    keys = list(PAIRS)
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for i, (is_native, names) in enumerate(((True, NATIVE_NAME), (False, EXT_NAME))):
        vals = [means.get(names[k], np.nan) for k in keys]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, label="native" if is_native else "interfaced",
              color="tab:blue" if is_native else "tab:orange")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("mean fit time (s, log)")
    ax.set_title("fit time, native vs. interfaced")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    paths["fit_time"] = os.path.join(PLOTS_DIR, f"{stem}fit_time.png")
    fig.savefig(paths["fit_time"], dpi=110)
    plt.close(fig)
    print(f"Plot   -> {paths['fit_time']}")
    return paths


def main(quick: bool = True) -> None:
    names = QUICK_DATASETS if quick else FULL_DATASETS
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")

    print("Preliminary check: ORB vs. RKD:RuleBoosting ...")
    prelim_results, prelim_verdict = run_preliminary_optimal_rule_boosting_check()
    print(prelim_verdict)

    datasets = Catalog.default().select(names=names)
    learners = build_learners()
    learner_names = [l.display_name for l in learners]

    groups = [(datasets, n_folds)] if quick else [
        ([d for d in datasets if d.size != "large"], n_folds),
        ([d for d in datasets if d.size == "large"], LARGE_FOLDS),
    ]
    max_intervals = QUICK_MAX_INTERVALS if quick else MAX_INTERVALS
    results = pd.concat([
        run_cv(learners, group, n_folds=folds, fit_timeout=FIT_TIMEOUT, max_intervals=max_intervals,
              random_state=RANDOM_STATE, cache_dir=CACHE_DIR)
        for group, folds in groups if group
    ], ignore_index=True)

    plots = write_plots(results, learner_names, quick)
    write_report(results, datasets, n_folds, report_path, plots, quick, learner_names,
                prelim_results, prelim_verdict)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run all {len(FULL_DATASETS)} datasets instead of the quick default")
    args = parser.parse_args()
    main(quick=not args.full)
