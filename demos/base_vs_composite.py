"""
demos/base_vs_composite.py
==========================

Base against composite models, on a random forest -- one model, two views:
as a composite model, the trees' predictions combined by the forest's own
vote; as a base model, the union of all trees' leaves as one flat rule
set, its conflicts resolved by the four combiners of
`demos/rule_models.py`. Cross-validated through
`pyrulearn.experiments.runner.run_cv` on the datasets of that demo.

Every tree is a disjoint rule set, so that exactly one leaf of each tree
covers an example: the hard vote of the trees is the plain **voting** of
the covering rules, and the forest's soft vote (sklearn's, the trees'
class distributions averaged) is their **macro distribution voting** --
up to the leaves' class distributions, which sklearn takes from each
tree's bootstrap sample and the rule set from the whole training fold.
**Best rule** (the covering leaf with the best Laplace estimate) and
**micro distribution voting** (the covering leaves' class counts summed,
a larger leaf weighing more) have no counterpart at the level of the
trees.

On data with missing values, the equivalences hold only in part: under
pyrulearn's default (`MissingStrategy.NEVER_COVERS`), a missing value
satisfies neither a feature nor its negation, whereas a tree trained on
the Boolean features takes it as false and follows the negated branch --
so that an example with a missing value may be covered by no leaf of a
tree, and the rule set hears fewer trees than the forest.

Configurations (`DEPTHS`): a forest of `N_ESTIMATORS` trees of depth 4, of
depth 8 and of unlimited depth, each fit once per fold
(`RandomForestClassifier` on the Boolean features of the fold's
`DataSpec`, the same the rule learners of the other demos see). Each
fitted forest is scored six ways (`measure`): sklearn's own soft vote, the
hard vote of its trees, and its rule set under each of the four
combiners; for the two equivalences, also the share of test examples on
which both predict the same class.

Measures: accuracy, numbers of rules (leaves) and conditions, mean rank
across datasets.

Run: `python demos/base_vs_composite.py` with no arguments is the **quick**
default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its report goes to
``base_vs_composite_quick_report.md`` and is not checked in. `--full` runs the
multi-class datasets of `demos/rule_models.py` (`FULL_DATASETS`,
`LARGE_DATASETS`) and its two-class ones (`BINARY_DATASETS`),
`N_FOLDS`-fold, and writes ``base_vs_composite_report.md``. Needs the
`experiments` extra.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from pyrulearn.combiners import CountVoteCombiner, HeuristicMaxCombiner, MacroVoteCombiner, MicroVoteCombiner
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_results_table, \
    render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import critical_difference, critical_difference_diagram, friedman_test, mean_rank
from pyrulearn.interfaces.sklearn import RandomForestImporter, SKLRandomForest
from pyrulearn.learners.base import NativeRuleLearner

RANDOM_STATE = 0
N_FOLDS = 10
N_ESTIMATORS = 100
MAX_INTERVALS = 8
FIT_TIMEOUT = 600.0

#: name -> the trees' `max_depth`
DEPTHS = {"depth 4": 4, "depth 8": 8, "unlimited depth": None}

#: the rule set's conflict resolution, all on the same fitted forest (named as in the paper)
COMBINERS = {"best rule": HeuristicMaxCombiner, "voting": CountVoteCombiner,
             "micro distribution voting": MicroVoteCombiner, "macro distribution voting": MacroVoteCombiner}
#: the forest's own votes, and the combiner each one is equivalent to
FOREST_VOTES = {"forest (soft vote)": "macro distribution voting", "forest (hard vote)": "voting"}
#: every way a forest is scored, in the order of the report
VARIANTS = list(FOREST_VOTES) + [f"rules ({name})" for name in COMBINERS]

# small multi-class datasets
QUICK_DATASETS = ["car", "vehicle", "zoo", "soybean", "primary-tumor"]
QUICK_FOLDS = 3
# the datasets of demos/rule_models.py
FULL_DATASETS = ["balance-scale", "cmc", "dna", "hayes-roth", "iris", "splice", "wine", "car", "lymph",
                 "vehicle", "anneal", "glass", "solar-flare", "ecoli", "segment", "zoo", "led24", "yeast",
                 "optdigits", "texture", "vowel", "soybean", "primary-tumor", "audiology"]
LARGE_DATASETS = ["kropt", "letter"]
BINARY_DATASETS = ["SPECT", "Titanic", "banknote-authentication", "blood-transfusion-service-center",
                   "breast-cancer", "breast-w", "churn", "climate-model-simulation-crashes", "colic",
                   "compas-two-years", "credit-approval", "credit-g", "cylinder-bands", "diabetes",
                   "dresses-sales", "heart-c", "heart-h", "heart-statlog", "heloc", "hepatitis", "ilpd",
                   "ionosphere", "kc1", "kc2", "kr-vs-kp", "mofn-3-7-10", "molecular-biology_promoters",
                   "monks-problems-1", "monks-problems-2", "monks-problems-3", "mushroom", "ozone-level-8hr",
                   "pc1", "pc3", "pc4", "phoneme", "qsar-biodeg", "sick", "sonar", "spambase", "tic-tac-toe",
                   "vote", "wdbc", "wilt"]

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "base_vs_composite"
CACHE_DIR = os.path.join(HERE, "_base_vs_composite_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


class Forest(NativeRuleLearner):
    """A random forest of `N_ESTIMATORS` trees of the depth named by
    `depth` (`DEPTHS`), named ``"forest:<depth>"``, returned as the flat
    rule set of all its leaves (`RandomForestImporter`, each leaf's
    statistics measured on the training data). The fitted sklearn forest
    rides along as `forest`, for its own votes in `measure`; the time of
    the fit as `fit_seconds`."""

    def __init__(self, depth: str, random_state: int = RANDOM_STATE):
        self.depth = depth
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return f"forest:{self.depth}"

    def _provenance_params(self):
        # its cached results hold one accuracy per variant: a new combiner means fitting it again
        return {"depth": self.depth, "n_estimators": N_ESTIMATORS, "random_state": self.random_state,
                "combiners": list(COMBINERS)}

    def fit(self, data):
        from sklearn.ensemble import RandomForestClassifier

        data = SKLRandomForest().ensure_representation(data, None, announce=False)
        t0 = time.perf_counter()
        forest = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=DEPTHS[self.depth],
                                        random_state=self.random_state).fit(data.X, data.y)
        fit_seconds = time.perf_counter() - t0
        model = RandomForestImporter().import_model(forest, data.spec, data=data)
        model.forest = forest
        model.fit_seconds = fit_seconds
        return model


def build_learners():
    return [Forest(depth) for depth in DEPTHS]


def _hard_vote(forest, X) -> np.ndarray:
    """The plurality of the trees' predicted classes (ties to the first
    class of `forest.classes_`)."""
    votes = np.stack([tree.predict(X).astype(int) for tree in forest.estimators_], axis=1)
    counts = np.apply_along_axis(np.bincount, 1, votes, minlength=len(forest.classes_))
    return forest.classes_[counts.argmax(axis=1)]


def measure(model, test_rep) -> dict:
    """The accuracy of every variant (`VARIANTS`) of the same fitted
    forest, and for each of the forest's votes the share of test examples
    on which the equivalent combiner predicts the same class."""
    y = np.asarray(test_rep.y)
    X = SKLRandomForest().ensure_representation(test_rep, None, announce=False).X
    forest = model.forest
    predictions = {"forest (soft vote)": forest.predict(X), "forest (hard vote)": _hard_vote(forest, X)}
    for name, combiner in COMBINERS.items():
        model.combiner = combiner()
        predictions[f"rules ({name})"] = np.asarray(model.predict(test_rep))
    out = {"train_time": getattr(model, "fit_seconds", np.nan)}
    for variant, pred in predictions.items():
        out[f"accuracy_{variant}"] = float(np.mean(pred == y))
    for vote, name in FOREST_VOTES.items():
        out[f"agreement_{vote}"] = float(np.mean(predictions[vote] == predictions[f"rules ({name})"]))
    return out


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, fold, depth, variant), with that variant's
    accuracy; a forest vote's row also carries its agreement with the
    equivalent combiner."""
    rows = []
    for _, r in results.iterrows():
        depth = r["learner"].split(":", 1)[1]
        common = {"dataset": r["dataset"], "fold": r["fold"], "depth": depth, "n_rules": r["n_rules"],
                  "n_conditions": r["n_conditions"], "train_time": r.get("train_time"), "error": r["error"]}
        for variant in VARIANTS:
            rows.append({**common, "learner": f"{depth}:{variant}", "variant": variant,
                         "accuracy": r.get(f"accuracy_{variant}"),
                         "agreement": r.get(f"agreement_{variant}", np.nan)})
    return pd.DataFrame(rows)


def _description(datasets, binary, quick: bool, n_folds: int) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: d.n_classes))
    control = (f" Two-class datasets: {', '.join(f'`{d.name}`' for d in binary)}." if binary else "")
    return f"""\
{mode}A random forest combines its trees' predictions; but as every tree is a
disjoint rule set, exactly one leaf of each tree covers an example, and the
union of all leaves is a single flat rule set. This demo compares the two
views on the same fitted forests of {N_ESTIMATORS} trees, of depth 4, of depth 8 and
of unlimited depth:

- **forest (soft vote)** -- sklearn's own prediction, the trees' class
  distributions averaged;
- **forest (hard vote)** -- the plurality of the trees' predicted classes;
- **rules (...)** -- the union of the leaves as one rule set, each leaf's
  class distribution measured on the training fold, a conflict resolved by
  **best rule** (the covering leaf with the best Laplace estimate),
  **voting** (one vote per covering leaf), **micro distribution voting**
  (their class counts summed) or **macro distribution voting** (their
  class proportions summed).

The hard vote is the same as voting, and the soft vote as macro
distribution voting, except that sklearn's leaf distributions come from
each tree's bootstrap sample (and ties are broken differently); the
agreement columns give the share of test examples on which each pair
predicts the same class.

Multi-class datasets, with their numbers of classes: {names}.{control} Protocol:
{n_folds}-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals={MAX_INTERVALS})`), the forest trained on its Boolean
features. On data with missing values, the equivalences hold only in part:
a missing value satisfies neither a feature nor its negation, whereas a tree
takes it as a false feature, so that an example with a missing value may be
covered by no leaf of a tree. Each fit is capped at {FIT_TIMEOUT:.0f}s.
"""


def _summary(var: pd.DataFrame) -> list:
    """Per depth: the variants sorted by mean accuracy rank, with the
    agreement of the forest's votes with their equivalent combiners."""
    L = []
    for depth in DEPTHS:
        v = var[var["depth"] == depth]
        if not len(v):
            continue
        ranks = mean_rank(v, "accuracy")
        means = v.groupby("learner")[["accuracy", "agreement", "n_rules", "n_conditions"]].mean(numeric_only=True)
        L.append(f"### {depth[0].upper() + depth[1:]}\n\n"
                 f"Rules (leaves) per forest: {format_mean_count(v['n_rules'].mean())}, conditions: "
                 f"{format_mean_count(v['n_conditions'].mean())}.\n\n"
                 "| combination | accuracy | mean rank | agreement |\n|---|--:|--:|--:|\n")
        for learner in ranks.sort_values().index:
            m = means.loc[learner]
            variant = learner.split(":", 1)[1]
            agreement = (f"{100 * m['agreement']:.2f}% with {FOREST_VOTES[variant]}"
                         if variant in FOREST_VOTES else "")
            L.append(f"| {variant} | {format_accuracy(m['accuracy'])} | {ranks[learner]:.2f} | {agreement} |\n")
        L.append("\n")
    return L


def _overall(var: pd.DataFrame, group: str, quick: bool) -> list:
    """The variants across the depths: ranked within each (dataset, depth)
    pair, the ranks averaged over all pairs; then the Friedman test over
    those pairs and the critical-difference diagram."""
    blocks = var.assign(dataset=lambda d: d["dataset"] + " / " + d["depth"], learner=lambda d: d["variant"])
    ranks = mean_rank(blocks, "accuracy")
    means = blocks.groupby("learner")["accuracy"].mean()
    L = ["### All depths together\n\n"
         "Each combination's mean rank, ranked within each depth on each dataset, and its accuracy, "
         "averaged over the depths.\n\n| combination | accuracy | mean rank |\n|---|--:|--:|\n"]
    for learner in ranks.index:
        L.append(f"| {learner} | {format_accuracy(means[learner])} | {ranks[learner]:.2f} |\n")
    try:
        f = friedman_test(blocks, "accuracy")
        cd = critical_difference(f.n_learners, f.n_datasets)
        L.append(f"\nFriedman test over the {f.n_datasets} (dataset, depth) pairs: p = {f.pvalue:.2g} "
                 f"({f.n_learners} combinations). Nemenyi critical difference at 5%: {cd:.2f}. The pairs of one "
                 "dataset aren't independent of each other, so the test is on the optimistic side.\n\n")
    except ValueError as exc:
        L.append(f"\nNo Friedman test ({exc}).\n\n")
    path = os.path.join(PLOTS_DIR, f"{NAME}_{'quick_' if quick else ''}cd_{group}.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(PLOTS_DIR, exist_ok=True)
        ax = critical_difference_diagram(blocks, "accuracy")
        ax.figure.tight_layout()
        ax.figure.savefig(path, dpi=110)
        plt.close(ax.figure)
        L.append(f"![critical-difference diagram, {group}]({os.path.basename(PLOTS_DIR)}/{os.path.basename(path)})\n\n")
        print(f"Plot   -> {path}")
    except Exception as e:  # noqa: BLE001 -- a plot failure shouldn't sink the rest of the report
        print(f"Critical-difference diagram ({group}) skipped: {type(e).__name__}: {e}")
    return L


def write_report(var, datasets, binary, quick: bool, n_folds: int, report_path: str) -> None:
    L = ["# Base against composite models: a forest's vote against the rule set of its leaves\n\n"]
    L.append(render_setup_section(_description(datasets, binary, quick, n_folds)))
    failures = var.drop_duplicates(["dataset", "fold", "depth"])["error"].notna().sum()
    if failures:
        L.append(f"Failed folds: {failures}.\n\n")
    for title, group, sets in (("Multi-class datasets", "multiclass", datasets),
                               ("Two-class datasets", "twoclass", binary)):
        part = var[var["dataset"].isin([d.name for d in sets])]
        if not len(part):
            continue
        L.append(f"## {title}\n\nSorted by mean rank (accuracy rank across datasets). Agreement: the share of "
                 "test examples on which a forest vote and its equivalent combiner predict the same class.\n\n")
        L += _summary(part)
        L += _overall(part, group, quick)
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var, ["accuracy", "n_rules"], group_by=["dataset", "learner"],
                                  include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    catalog = Catalog.default()
    datasets = catalog.select(names=QUICK_DATASETS if quick else FULL_DATASETS + LARGE_DATASETS)
    binary = [] if quick else catalog.select(names=BINARY_DATASETS)
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    common = dict(n_folds=n_folds, fit_timeout=FIT_TIMEOUT, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=measure, n_jobs=_n_jobs())
    runs = [run_cv(build_learners(), datasets, **common)]
    if binary:
        runs.append(run_cv(build_learners(), binary, **common))
    results = pd.concat(runs, ignore_index=True)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(variants(results), list(datasets), list(binary), quick, n_folds, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS) + len(LARGE_DATASETS)} multi-class and "
                             f"{len(BINARY_DATASETS)} two-class datasets, {N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full)
