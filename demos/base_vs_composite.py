"""
demos/base_vs_composite.py
==========================

Base against composite models, on bagged rule sets: the ensemble as a
composite model, the members' predictions combined, or as a base model,
all of their rules combined at once. Cross-validated through
`pyrulearn.experiments.runner.run_cv` on the multi-class datasets of
`demos/rule_models.py` (without `kropt`) and on two-class datasets.

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

Configurations: one learner per base learner (`BASES`), fitting the
largest ensemble of `SIZES` (`--sizes`) and the single concept set once per
fold. The smaller ensembles are its first members (`EnsembleModel.staged`):
bagging learns its members one after the other, so they are the ensembles
a fit with fewer members builds. Each fitted learner is scored at every
size, every level and with every conflict resolution (`measure`).

Measures: accuracy, numbers of rules (per size) and conditions, mean rank
across datasets.

Run: `python demos/base_vs_composite.py` with no arguments is the **quick**
default (`QUICK_DATASETS`, `QUICK_FOLDS`-fold); its report goes to
``base_vs_composite_quick_report.md`` and is not checked in. `--full` runs
the multi-class datasets (`FULL_DATASETS`) and the two-class ones
(`BINARY_DATASETS`), `N_FOLDS`-fold, and writes
``base_vs_composite_report.md``. Needs the `experiments` extra.
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
#: the ensemble sizes, all from one fit of the largest
SIZES = [10, 25, 50]
MAX_INTERVALS = 8
FIT_TIMEOUT = 1800.0
#: the largest datasets, with a longer cap per fit
LONG_DATASETS = ["letter"]
LONG_FIT_TIMEOUT = 3600.0

#: name -> base learner class
BASES = {"Pypper": Pypper, "PFossil": PFossil, "PFoil": PFoil}
#: the conflict resolutions, all on the same fitted members (named as in the paper)
COMBINERS = {"best rule": HeuristicMaxCombiner, "voting": CountVoteCombiner,
             "micro distribution voting": MicroVoteCombiner, "macro distribution voting": MacroVoteCombiner}
#: how an ensemble's conflict resolution is applied: to each member, or to all rules at once
LEVELS = ["two-level", "one-level"]


def variant_names(sizes):
    """Every way a fitted learner is scored: ``single (<combiner>)``, and
    ``<level>@<size> (<combiner>)`` for every ensemble size."""
    return [f"single ({name})" for name in COMBINERS] + [
        f"{level}@{n} ({name})" for n in sizes for name in COMBINERS for level in LEVELS]

# small datasets, multi-class and two-class
QUICK_DATASETS = ["car", "zoo", "primary-tumor", "breast-cancer", "diabetes"]
QUICK_FOLDS = 3
# the multi-class datasets of demos/rule_models.py, without kropt
FULL_DATASETS = ["balance-scale", "cmc", "dna", "hayes-roth", "iris", "splice", "wine", "car", "lymph",
                 "vehicle", "anneal", "glass", "solar-flare", "ecoli", "segment", "zoo", "led24", "yeast",
                 "optdigits", "texture", "vowel", "soybean", "primary-tumor", "audiology", "letter"]
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

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


class BaggedConcepts(NativeRuleLearner):
    """The largest of `sizes` concept sets of the base learner named by
    `base` (`BASES`), each learned one-vs-rest from a bootstrap sample, as
    an `EnsembleModel`, named ``"bagging:<base>"``. The single concept
    set learned from the whole training data rides along as `single`,
    and `sizes` as `sizes`, for `measure`."""

    def __init__(self, base: str, sizes=tuple(SIZES), random_state: int = RANDOM_STATE):
        self.base = base
        self.sizes = tuple(sorted(sizes))
        self.random_state = random_state

    @property
    def display_name(self) -> str:
        return f"bagging:{self.base}"

    def _provenance_params(self):
        # its cached results hold one accuracy per variant: a new combiner means fitting it again
        return {"base": self.base, "sizes": list(self.sizes), "random_state": self.random_state,
                "combiners": list(COMBINERS)}

    def fit(self, data):
        model = Bagging(OneVsRest(BASES[self.base]()), n_estimators=self.sizes[-1],
                        random_state=self.random_state).fit(data)
        model.single = OneVsRest(BASES[self.base]()).fit(data)
        model.sizes = self.sizes
        return model


def build_learners(sizes=tuple(SIZES)):
    return [BaggedConcepts(base, sizes) for base in BASES]


def measure(model, test_rep) -> dict:
    """The accuracy of every variant (`variant_names`) of the same fitted
    members and single concept set, and the number of rules at every
    size."""
    y = np.asarray(test_rep.y)
    out = {"rules_single": len(model.single.rules)}
    accuracy = lambda pred: float(np.mean(np.asarray(pred) == y))
    for n in model.sizes:
        out[f"rules@{n}"] = len(model.staged(n).rules)
    for name, combiner in COMBINERS.items():
        for member in model.members:
            member.combiner = combiner()
        model.single.combiner = combiner()
        out[f"accuracy_single ({name})"] = accuracy(model.single.predict(test_rep))
        for n in model.sizes:
            ensemble = model.staged(n)
            ensemble.combiner = CountVoteCombiner()
            out[f"accuracy_two-level@{n} ({name})"] = accuracy(ensemble.predict(test_rep))
            out[f"accuracy_one-level@{n} ({name})"] = accuracy(
                ensemblemodel_to_flatruleset(ensemble, combiner=combiner()).predict(test_rep))
    return out


def variants(results: pd.DataFrame, sizes) -> pd.DataFrame:
    """One row per (dataset, fold, base learner, variant), with that
    variant's accuracy and its size (``None`` for the single concept set)
    and number of rules."""
    rows = []
    for _, r in results.iterrows():
        base = r["learner"].split(":", 1)[1]
        common = {"dataset": r["dataset"], "fold": r["fold"], "base": base, "error": r["error"]}
        for variant in variant_names(sizes):
            level, name = variant.split(" (", 1)
            level, _, n = level.partition("@")
            size = int(n) if n else None
            rows.append({**common, "learner": f"{base}:{variant}", "variant": variant, "level": level,
                         "size": size, "combiner": name[:-1], "accuracy": r.get(f"accuracy_{variant}"),
                         "n_rules": r.get(f"rules@{n}" if n else "rules_single")})
    return pd.DataFrame(rows)


def _description(datasets, binary, quick: bool, n_folds: int, sizes) -> str:
    mode = ("**Quick run** -- a fast sanity check, the default with no arguments. Full run: "
            f"`python demos/{NAME}.py --full`.\n\n" if quick else
            f"**Full run.** Quick sanity check instead: `python demos/{NAME}.py`.\n\n")
    names = ", ".join(f"`{d.name}` ({d.n_classes})" for d in sorted(datasets, key=lambda d: d.n_classes))
    control = (f" Two-class datasets: {', '.join(f'`{d.name}`' for d in binary)}." if binary else "")
    return f"""\
{mode}Bagging with rule sets as members: {', '.join(map(str, sizes[:-1]))} and {sizes[-1]} concept sets -- one
concept description per class, learned one-vs-rest by {', '.join(BASES)} -- each from
a bootstrap sample of the training fold; the smaller ensembles are the first
members of the largest. The same members are combined in two
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


def _accuracy_table(var: pd.DataFrame, sizes, ranked: bool) -> list:
    """One row per conflict resolution: the single concept set's accuracy,
    and per size the two-level and the one-level accuracy -- with
    `ranked`, each followed by its mean rank among the eight ensemble
    variants of its size."""
    acc = var.groupby("variant")["accuracy"].mean()
    ranks = {n: mean_rank(var[var["size"] == n].assign(learner=lambda d: d["variant"]), "accuracy")
             for n in sizes} if ranked else {}
    L = ["| conflict resolution | single | " + " | ".join(f"{lv} @{n}" for n in sizes for lv in LEVELS)
         + " |\n|---|--:|" + "--:|" * (len(sizes) * len(LEVELS)) + "\n"]
    for name in COMBINERS:
        cells = [format_accuracy(acc[f"single ({name})"])]
        for n in sizes:
            for lv in LEVELS:
                v = f"{lv}@{n} ({name})"
                cells.append(format_accuracy(acc[v]) + (f" ({ranks[n][v]:.2f})" if ranked else ""))
        L.append(f"| {name} | " + " | ".join(cells) + " |\n")
    return L


def _per_base(var: pd.DataFrame, sizes) -> list:
    """Per base learner: the accuracy (and mean rank) of every conflict
    resolution at every size and level, and the numbers of rules."""
    L = []
    for base in BASES:
        v = var[var["base"] == base]
        if not len(v):
            continue
        rules = v.groupby("size", dropna=False)["n_rules"].mean()
        L.append(f"## {base}\n\nRules of the single concept set: {format_mean_count(rules[np.nan])}; "
                 "of the ensembles: " + ", ".join(f"{format_mean_count(rules[n])} ({n} members)" for n in sizes)
                 + ".\n\n")
        L += _accuracy_table(v, sizes, ranked=True)
        L.append("\nAccuracy, and in parentheses the mean rank among the eight ensemble variants of the same "
                 "size (four conflict resolutions at two levels).\n\n")
    return L


def _two_vs_one(var: pd.DataFrame, sizes) -> list:
    """Per conflict resolution and size, pooled over the base learners: how
    often the two-level combination is more accurate than the one-level
    one, and by how much on average."""
    ens = var[var["size"].notna()]
    acc = ens.groupby(["dataset", "base", "size", "combiner", "level"])["accuracy"].mean().unstack("level")
    diff = 100 * (acc["two-level"] - acc["one-level"])
    L = ["## Two-level against one-level\n\n"
         "Per (dataset, base learner) pair, the accuracy of the two-level combination minus that of the "
         "one-level one, in percentage points, and the number of pairs on which either is ahead.\n\n"
         "| conflict resolution | members | mean difference | two-level ahead | one-level ahead | ties |\n"
         "|---|--:|--:|--:|--:|--:|\n"]
    for name in COMBINERS:
        for n in sizes:
            d = diff.xs((n, name), level=("size", "combiner")).dropna()
            L.append(f"| {name} | {n} | {d.mean():+.2f} | {(d > 0).sum()} | {(d < 0).sum()} | "
                     f"{(d == 0).sum()} |\n")
    L.append("\n")
    return L


def write_report(var, datasets, binary, quick: bool, n_folds: int, sizes, report_path: str) -> None:
    L = ["# Base against composite models: bagged rule sets, combined at two levels or one\n\n"]
    L.append(render_setup_section(_description(datasets, binary, quick, n_folds, sizes)))
    failed = var[var["error"].notna()].drop_duplicates(["dataset", "base"])
    if len(failed):
        L.append("Left out, as a fold failed (the whole cross-validation of that learner is then counted as "
                 "failed): " + ", ".join(f"`{r.dataset}` ({r.base}: {r.error})" for r in failed.itertuples())
                 + ".\n\n")
    done = var.merge(failed[["dataset", "base"]], how="left", indicator=True)
    done = done[done["_merge"] == "left_only"].drop(columns="_merge")
    L.append("## All base learners\n\nAccuracy averaged over the datasets and the three base learners.\n\n")
    L += _accuracy_table(done, sizes, ranked=False)
    L.append("\n")
    L += _per_base(done, sizes)
    L += _two_vs_one(done, sizes)
    L.append("## Per dataset (mean across folds)\n\n")
    L.append(render_results_table(var, ["accuracy", "n_rules"], group_by=["dataset", "learner"],
                                  include_overall=False))
    L.append("\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True, sizes=tuple(SIZES)) -> None:
    sizes = sorted(sizes)
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
            runs.append(run_cv(build_learners(sizes), group, fit_timeout=timeout, **common))
    results = pd.concat(runs, ignore_index=True)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(variants(results, sizes), datasets, binary, quick, n_folds, sizes, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"{len(FULL_DATASETS)} multi-class and {len(BINARY_DATASETS)} two-class "
                             f"datasets, {N_FOLDS}-fold")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES,
                        help=f"ensemble sizes, all from one fit of the largest (default {SIZES})")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(quick=not args.full, sizes=args.sizes)
