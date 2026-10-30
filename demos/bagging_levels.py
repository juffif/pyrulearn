"""
demos/bagging_levels.py
=======================

Bagged rule sets, combined at two levels: the members' predictions, or
all of their rules at once. Cross-validated through
`pyrulearn.experiments.runner.run_cv` on the datasets of
`demos/base_vs_composite.py`.

Each member of the ensemble is a concept set -- one concept description
per class, learned one-vs-rest (`OneVsRest`) by `Pypper`, `PFossil` or
`PFoil` -- from a bootstrap sample of the training fold
(`pyrulearn.learners.bagging.Bagging`). The same fitted members are
combined in two ways, each with the four conflict resolutions of
`demos/rule_models.py` (`COMBINERS`):

- **two-level**: every member resolves its own covering rules with the
  conflict resolution, and the members' predictions are combined by a
  plain vote (an `EnsembleModel` with `CountVoteCombiner`);
- **one-level**: the union of all members' rules is one rule set, its
  conflicts resolved by the conflict resolution
  (`ensemblemodel_to_flatruleset`).

Unlike the trees of a random forest, a concept set is not disjoint:
several of its rules, or none, may cover an example, so that the two
levels differ. A member whose rules don't cover an example abstains at
both levels -- the members' own defaults are ignored -- and examples no
member covers go to the majority class. The baseline is a single concept
set learned from the whole training fold, with the same conflict
resolution.

Configurations: one learner per base learner (`BASES`), fitting
`N_ESTIMATORS` members (`--n-estimators`) and the single concept set once per fold; each
fitted learner is scored twelve ways (`measure`).

Measures: accuracy, numbers of rules and conditions (of the ensemble),
mean rank across datasets.

Run: `python demos/bagging_levels.py` with no arguments is the **quick**
default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its report goes to
``bagging_levels_quick_report.md`` and is not checked in. `--full` runs the
multi-class and two-class datasets of `demos/base_vs_composite.py`,
`N_FOLDS`-fold, and writes ``bagging_levels_report.md``. Needs the
`experiments` extra.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pyrulearn.combiners import CountVoteCombiner, HeuristicMaxCombiner, MacroVoteCombiner, MicroVoteCombiner
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_results_table, \
    render_setup_section
from pyrulearn.experiments.runner import run_cv
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.learners.bagging import Bagging
from pyrulearn.learners.base import NativeRuleLearner
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.seco import PFoil, PFossil, Pypper
from pyrulearn.models import ensemblemodel_to_flatruleset

RANDOM_STATE = 0
N_FOLDS = 10
N_ESTIMATORS = 10
MAX_INTERVALS = 8
FIT_TIMEOUT = 1800.0
#: the largest datasets, with a longer cap per fit
LONG_DATASETS = ["kropt", "letter"]
LONG_FIT_TIMEOUT = 3600.0

#: name -> base learner class
BASES = {"Pypper": Pypper, "PFossil": PFossil, "PFoil": PFoil}
#: the conflict resolutions, all on the same fitted members (named as in the paper)
COMBINERS = {"best rule": HeuristicMaxCombiner, "voting": CountVoteCombiner,
             "micro distribution voting": MicroVoteCombiner, "macro distribution voting": MacroVoteCombiner}
#: how a conflict resolution is applied: to a single concept set, to each member, to all rules at once
LEVELS = ["single", "two-level", "one-level"]
#: every way a fitted learner is scored, in the order of the report
VARIANTS = [f"{level} ({name})" for name in COMBINERS for level in LEVELS]

# small datasets, multi-class and two-class
QUICK_DATASETS = ["car", "zoo", "primary-tumor", "breast-cancer", "diabetes"]
QUICK_FOLDS = 3
# the datasets of demos/base_vs_composite.py
FULL_DATASETS = ["balance-scale", "cmc", "dna", "hayes-roth", "iris", "splice", "wine", "car", "lymph",
                 "vehicle", "anneal", "glass", "solar-flare", "ecoli", "segment", "zoo", "led24", "yeast",
                 "optdigits", "texture", "vowel", "soybean", "primary-tumor", "audiology", "kropt", "letter"]
BINARY_DATASETS = ["SPECT", "Titanic", "banknote-authentication", "blood-transfusion-service-center",
                   "breast-cancer", "breast-w", "churn", "climate-model-simulation-crashes", "colic",
                   "compas-two-years", "credit-approval", "credit-g", "cylinder-bands", "diabetes",
                   "dresses-sales", "heart-c", "heart-h", "heart-statlog", "heloc", "hepatitis", "ilpd",
                   "ionosphere", "kc1", "kc2", "kr-vs-kp", "mofn-3-7-10", "molecular-biology_promoters",
                   "monks-problems-1", "monks-problems-2", "monks-problems-3", "mushroom", "ozone-level-8hr",
                   "pc1", "pc3", "pc4", "phoneme", "qsar-biodeg", "sick", "sonar", "spambase", "tic-tac-toe",
                   "vote", "wdbc", "wilt"]

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "bagging_levels"
CACHE_DIR = os.path.join(HERE, "_bagging_levels_cache")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


class BaggedConcepts(NativeRuleLearner):
    """`n_estimators` concept sets of the base learner named by `base`
    (`BASES`), each learned one-vs-rest from a bootstrap sample, as an
    `EnsembleModel`, named ``"bagging:<base>"``. The single concept set
    learned from the whole training data rides along as `single`, for
    the baseline in `measure`."""

    def __init__(self, base: str, n_estimators: int = N_ESTIMATORS, random_state: int = RANDOM_STATE):
        self.base = base
        self.n_estimators = n_estimators
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return f"bagging:{self.base}"

    def _provenance_params(self):
        # its cached results hold one accuracy per variant: a new combiner means fitting it again
        return {"base": self.base, "n_estimators": self.n_estimators, "random_state": self.random_state,
                "combiners": list(COMBINERS)}

    def fit(self, data):
        model = Bagging(OneVsRest(BASES[self.base]()), n_estimators=self.n_estimators,
                        random_state=self.random_state).fit(data)
        model.single = OneVsRest(BASES[self.base]()).fit(data)
        return model


def build_learners(n_estimators: int = N_ESTIMATORS):
    return [BaggedConcepts(base, n_estimators) for base in BASES]


def measure(model, test_rep) -> dict:
    """The accuracy of every variant (`VARIANTS`) of the same fitted
    members and single concept set."""
    y = np.asarray(test_rep.y)
    out = {"single_rules": len(model.single.rules)}
    for name, combiner in COMBINERS.items():
        for member in model.members:
            member.combiner = combiner()
        model.single.combiner = combiner()
        model.combiner = CountVoteCombiner()
        predictions = {"single": model.single.predict(test_rep), "two-level": model.predict(test_rep),
                       "one-level": ensemblemodel_to_flatruleset(model, combiner=combiner()).predict(test_rep)}
        for level, pred in predictions.items():
            out[f"accuracy_{level} ({name})"] = float(np.mean(np.asarray(pred) == y))
    return out


def variants(results: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, fold, base learner, variant), with that
    variant's accuracy."""
    rows = []
    for _, r in results.iterrows():
        base = r["learner"].split(":", 1)[1]
        common = {"dataset": r["dataset"], "fold": r["fold"], "base": base, "n_rules": r["n_rules"],
                  "n_conditions": r["n_conditions"], "single_rules": r.get("single_rules"), "error": r["error"]}
        for variant in VARIANTS:
            level, name = variant.split(" (", 1)
            rows.append({**common, "learner": f"{base}:{variant}", "variant": variant, "level": level,
                         "combiner": name[:-1], "accuracy": r.get(f"accuracy_{variant}")})
    return pd.DataFrame(rows)


def _description(datasets, binary, quick: bool, n_folds: int, n_estimators: int) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: d.n_classes))
    control = (f" Two-class datasets: {', '.join(f'`{d.name}`' for d in binary)}." if binary else "")
    return f"""\
{mode}Bagging with rule sets as members: {n_estimators} concept sets -- one concept
description per class, learned one-vs-rest by {', '.join(BASES)} -- each from a
bootstrap sample of the training fold. The same members are combined in two
ways, each with one of four conflict resolutions (**best rule**, the
covering rule with the best Laplace estimate; **voting**, one vote per
covering rule; **micro** / **macro distribution voting**, the covering
rules' class counts / class proportions summed):

- **two-level** -- every member resolves its own covering rules, and the
  members' predictions are combined by a plain vote;
- **one-level** -- the union of all members' rules as one rule set, its
  conflicts resolved directly.

A member whose rules don't cover an example abstains at both levels (the
members' defaults are ignored); examples no member covers go to the
majority class. **single** is the baseline: one concept set learned from the
whole training fold, with the same conflict resolution.

Datasets, with their numbers of classes: {names}.{control} Protocol:
{n_folds}-fold stratified cross-validation, one `DataSpec` per training fold
(`TreeDiscretizer(max_intervals={MAX_INTERVALS})`). Each fit -- the members and the
single concept set together -- is capped at {FIT_TIMEOUT:.0f}s ({LONG_FIT_TIMEOUT:.0f}s for
{', '.join(f'`{n}`' for n in LONG_DATASETS)}).
"""


def _per_base(var: pd.DataFrame) -> list:
    """Per base learner: accuracy and mean rank (among the twelve
    variants) of every conflict resolution at every level."""
    L = []
    for base in BASES:
        v = var[var["base"] == base]
        if not len(v):
            continue
        ranks = mean_rank(v, "accuracy")
        acc = v.groupby("learner")["accuracy"].mean()
        L.append(f"## {base}\n\nRules per ensemble: {format_mean_count(v['n_rules'].mean())}, conditions: "
                 f"{format_mean_count(v['n_conditions'].mean())}; rules of the single concept set: "
                 f"{format_mean_count(v['single_rules'].mean())}.\n\n"
                 "| conflict resolution | " + " | ".join(LEVELS) + " |\n|---|" + "--:|" * len(LEVELS) + "\n")
        for name in COMBINERS:
            cells = [f"{format_accuracy(acc[f'{base}:{lv} ({name})'])} ({ranks[f'{base}:{lv} ({name})']:.2f})"
                     for lv in LEVELS]
            L.append(f"| {name} | " + " | ".join(cells) + " |\n")
        L.append("\nAccuracy, and in parentheses the mean rank among the twelve variants.\n\n")
    return L


def _two_vs_one(var: pd.DataFrame) -> list:
    """Per conflict resolution, pooled over the base learners: how often
    the two-level combination is more accurate than the one-level one, and
    by how much on average."""
    acc = var.groupby(["dataset", "base", "combiner", "level"])["accuracy"].mean().unstack("level")
    diff = 100 * (acc["two-level"] - acc["one-level"])
    L = ["## Two-level against one-level\n\n"
         "Per (dataset, base learner) pair, the accuracy of the two-level combination minus that of the "
         "one-level one, in percentage points, and the number of pairs on which either is ahead.\n\n"
         "| conflict resolution | mean difference | two-level ahead | one-level ahead | ties |\n"
         "|---|--:|--:|--:|--:|\n"]
    for name in COMBINERS:
        d = diff.xs(name, level="combiner").dropna()
        L.append(f"| {name} | {d.mean():+.2f} | {(d > 0).sum()} | {(d < 0).sum()} | {(d == 0).sum()} |\n")
    L.append("\n")
    return L


def write_report(var, datasets, binary, quick: bool, n_folds: int, n_estimators: int,
                 report_path: str) -> None:
    L = ["# Bagged rule sets: two-level against one-level combination\n\n"]
    L.append(render_setup_section(_description(datasets, binary, quick, n_folds, n_estimators)))
    failed = var[var["error"].notna()].drop_duplicates(["dataset", "base"])
    if len(failed):
        L.append("Left out, as a fold failed (the whole cross-validation of that learner is then counted as "
                 "failed): " + ", ".join(f"`{r.dataset}` ({r.base}: {r.error})" for r in failed.itertuples())
                 + ".\n\n")
    done = var.merge(failed[["dataset", "base"]], how="left", indicator=True)
    done = done[done["_merge"] == "left_only"].drop(columns="_merge")
    L += _per_base(done)
    L += _two_vs_one(done)
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var, ["accuracy", "n_rules"], group_by=["dataset", "learner"],
                                  include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True, n_estimators: int = N_ESTIMATORS) -> None:
    catalog = Catalog.default()
    datasets = [d for d in catalog.select(names=QUICK_DATASETS if quick else FULL_DATASETS) if d.n_classes > 2]
    binary = [d for d in catalog.select(names=QUICK_DATASETS if quick else BINARY_DATASETS) if d.n_classes == 2]
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    common = dict(n_folds=n_folds, discretizer=TreeDiscretizer(max_intervals=MAX_INTERVALS),
                  random_state=RANDOM_STATE, cache_dir=CACHE_DIR, measure_fn=measure, n_jobs=_n_jobs())
    everything = datasets + binary
    runs = []
    for group, timeout in (([d for d in everything if d.name not in LONG_DATASETS], FIT_TIMEOUT),
                           ([d for d in everything if d.name in LONG_DATASETS], LONG_FIT_TIMEOUT)):
        if group:
            runs.append(run_cv(build_learners(n_estimators), group, fit_timeout=timeout, **common))
    results = pd.concat(runs, ignore_index=True)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(variants(results), datasets, binary, quick, n_folds, n_estimators, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS)} multi-class and {len(BINARY_DATASETS)} two-class "
                             f"datasets, {N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS,
                        help=f"members per ensemble (default {N_ESTIMATORS})")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full, n_estimators=args.n_estimators)
