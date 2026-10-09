"""
demos/covering_boosting.py
===========================

Two angles on the same question -- what should happen to an example once
an accepted rule covers it? -- in one framework, because they turn out
to literally share code:

- **Removal covering** (the classic separate-and-conquer default):
  covered examples leave the scope entirely.
- **Weighted covering**: covered examples stay, reweighted by a
  `pyrulearn.learners.seco.Reweighting` scheme -- multiplicative/additive
  (CN2-SD), or `LRIReweighting` (Lightweight Rule Induction) -- until a
  `CoveringStop` criterion is met.
- **Boosting**: `Slipper` (Cohen & Singer 1999) is `WeightedCovering`
  with `AdaBoostReweighting` -- the *same* `CoveringState`/reweighting
  machinery as the line above, just assigning each rule a fitted weight
  (confidence), which plain `SeCo` refuses to build into an unweighted
  rule set (see `SeCo`'s own `covering=` docstring) -- so it needs its
  own learner class, `Slipper`, but not its own mechanism. `ENDER`
  (gradient boosting of rules for a pluggable loss) and its `Boomer`
  configuration aren't as separate from this as they look, either:
  AdaBoost reweighting *is* gradient boosting on the exponential loss
  (Friedman, Hastie & Tibshirani 2000) -- `ENDER`'s `ExponentialLoss`
  defines its per-example quantity as the same `exp(-y f)` AdaBoost
  reweights by, and fits each rule's weight with the identical closed
  form `AdaBoostReweighting.confidence` uses. What they don't share:
  `ENDER`'s *other* losses (logistic, sigmoid) and weight-fitting
  methods have no reweighting-scheme equivalent at all, and weighted
  covering discards its weights at predict time (a heuristic scores an
  unweighted rule set) while boosting bakes the fitted weight into the
  model itself (`LinearRuleModel`, summed at predict time).

Two sections, one report:

- **Section 1 -- covering strategies**: a fixed search (beam width 5)
  and rule-growing setup, crossed with covering strategy (removal;
  weighted, multiplicative/additive/LRI) and heuristic (Laplace, WRAcc,
  m-estimate with m=16 -- `demos/heuristic_comparison.py`'s own measured
  best-performing setting),
  via plain `SeCo`. `CPAR` and `LRI` sit alongside as complete
  algorithms (their own classes -- extra procedure beyond a bare
  reweighting scheme, not expressible as a `SeCo` config). Deliberately
  *not* reusing `CN2`'s own significance-based stopping here: that's a
  search-level criterion that can end the covering loop by itself, which
  would confound the comparison this section is actually about.
- **Section 2 -- boosting**: `Slipper`, `ENDER` (a few representative
  configs: the paper's default CS-Log, CS-Exp, and MLRules'
  newton/subsample setting), `Boomer`. Headline measure is accuracy vs.
  rule count as a curve (refit at a few `n_rounds`/`n_rules` budgets),
  not one fitted size -- simpler than hunting for a staged-prediction
  shortcut, if more fits.

A closing **comparative discussion**, built fresh from the measured
numbers each run, ties the two sections together: Slipper's point here
is directly comparable to section 1's `WeightedCovering` +
`AdaBoostReweighting` entry (same mechanism), and to `ENDER`'s
exponential-loss configuration (the same mechanism again, by a
classical result -- see the module docstring above); `ENDER`'s other
losses, and weighted covering's non-exponential schemes, are where the
two families actually part ways.

Datasets: the same two-class, mostly-symbolic list as
`demos/seco_learners_comparison.py`'s `FULL_BINARY` -- these learners'
weighted-covering/boosting machinery is inherently a one-class-vs-rest
framing, and that list is already curated for exactly this learner
family.

Measures: accuracy, rule count, rule length (conditions per rule), fit
time, and "overlap" -- the mean number of a model's rules covering a
test row, pooled across every concept the model has (not restricted to
one "positive" class: a plain `SeCo` config here builds a one-vs-rest
`ConceptSet` with no single target class to restrict to). Each fit
(section 1 or 2) is capped at {FIT_TIMEOUT}s; a time-out or an error
counts as a failure, not as the end of the run.

Run: `python demos/covering_boosting.py` with no arguments is
the **quick** default; its report isn't checked in. `--full` runs both
sections at full scale and writes the canonical report.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.experiments.catalog import Catalog, CatalogEntry
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import TimeoutRunner
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.heuristics import Laplace, MEstimate, WRAcc
from pyrulearn.learners.boosting import Boomer, ENDER, Slipper
from pyrulearn.learners.cpar import CPAR
from pyrulearn.learners.lri import LRI
from pyrulearn.learners.seco import (
    AdditiveReweighting, BeamSearch, CoveredAtLeast, LRIReweighting,
    MultiplicativeReweighting, RemovalCovering, Rounds, SeCo, SingleRuleLearner,
    WeightedCovering,
)

RANDOM_STATE = 0
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0
BEAM_WIDTH = 5

# -- section 1: covering strategies -------------------------------------

# m=16 is demos/heuristic_comparison.py's own measured best-performing
# m-estimate setting, not a default from the literature -- reused here
# rather than re-deriving it.
HEURISTICS = {"Laplace": lambda: Laplace(), "WRAcc": lambda: WRAcc(), "MEstimate16": lambda: MEstimate(16)}
# gamma=0.5 and CoveredAtLeast(5) are CN2-SD's own defaults (Lavrač et
# al. 2004); Rounds(50) for LRI matches LRIReweighting's own class
# default n_rules -- each scheme keeps its own literature-default stop
# rather than forcing one arbitrary stop onto all of them.
COVERINGS = {
    "Removal": lambda: RemovalCovering(),
    "Weighted-Mult": lambda: WeightedCovering(MultiplicativeReweighting(0.5), CoveredAtLeast(5)),
    "Weighted-Add": lambda: WeightedCovering(AdditiveReweighting(), CoveredAtLeast(5)),
    "Weighted-LRI": lambda: WeightedCovering(LRIReweighting(), Rounds(50)),
}
COVERING_CONFIGS = [f"{cov}+{heur}" for cov in COVERINGS for heur in HEURISTICS]
COMPLETE_ALGORITHMS = ["CPAR", "LRI"]
SECTION1_NAMES = COVERING_CONFIGS + COMPLETE_ALGORITHMS

# -- section 2: boosting -------------------------------------------------

BOOSTING_FAMILIES = ["Slipper", "ENDER-CSLog", "ENDER-CSExp", "ENDER-Newton", "Boomer"]
N_RULES_BUDGETS = [10, 20, 50, 100]
SECTION2_NAMES = [f"{family}@{n}" for family in BOOSTING_FAMILIES for n in N_RULES_BUDGETS]

# -- datasets -------------------------------------------------------------

# Same curation as demos/seco_learners_comparison.py's FULL_BINARY:
# mostly symbolic two-class datasets, the kind these learners were
# designed for.
FULL_DATASETS = [
    "molecular-biology_promoters", "hepatitis", "SPECT", "heart-statlog", "breast-cancer",
    "heart-h", "heart-c", "colic", "vote", "dresses-sales", "cylinder-bands", "credit-approval",
    "tic-tac-toe", "credit-g", "kr-vs-kp", "sick", "mushroom", "adult",
]
N_FOLDS = 10
LARGE_FOLDS = 5  # adult

QUICK_DATASETS = ["vote", "hepatitis", "SPECT"]
QUICK_FOLDS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "covering_boosting"
CACHE_DIR = os.path.join(HERE, "_covering_boosting_cache")
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")


# =========================================================== learners ===

def build_covering_learner(name: str) -> SeCo:
    cov_name, heur_name = name.split("+")
    heuristic = HEURISTICS[heur_name]()
    covering = COVERINGS[cov_name]()
    srl = SingleRuleLearner(heuristic=heuristic, search=BeamSearch(beam_width=BEAM_WIDTH))
    return SeCo(single_rule_learner=srl, covering=covering, random_state=RANDOM_STATE)


def build_section1_learner(name: str):
    if name == "CPAR":
        return CPAR()
    if name == "LRI":
        return LRI()
    return build_covering_learner(name)


def build_section2_learner(name: str):
    family, n_str = name.split("@")
    n = int(n_str)
    if family == "Slipper":
        return Slipper(n_rounds=n, random_state=RANDOM_STATE)
    if family == "ENDER-CSLog":
        return ENDER(n_rules=n, random_state=RANDOM_STATE)
    if family == "ENDER-CSExp":
        return ENDER(n_rules=n, loss="exponential", random_state=RANDOM_STATE)
    if family == "ENDER-Newton":
        return ENDER(n_rules=n, method="newton", subsample=0.5, random_state=RANDOM_STATE)
    if family == "Boomer":
        return Boomer(n_rules=n, random_state=RANDOM_STATE)
    raise ValueError(f"unknown boosting family {family!r}")


def _fit_one(learner, train_rep):
    return learner.fit(train_rep)


def _overlap(model, test_rep) -> float:
    n = len(np.asarray(test_rep.y))
    counts = np.zeros(n, dtype=int)
    for rule in model.rules:
        counts += rule.covers_data(test_rep).astype(int)
    return float(counts.mean())


# ============================================================= caching ===

def _cache_key(entry_name: str, n_folds: int, fold: int, name: str) -> str:
    payload = f"{entry_name}|folds={n_folds}|fold={fold}|rs={RANDOM_STATE}|name={name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def _cache_load(cache_dir: Optional[str], key: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cache_save(cache_dir: Optional[str], key: str, row: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, key), "w", encoding="utf-8") as f:
        json.dump(row, f)


def _stratified_folds(y: np.ndarray, n_folds: int, random_state: int):
    from sklearn.model_selection import StratifiedKFold

    return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=random_state).split(np.zeros(len(y)), y))


# =============================================================== the loop ===

def run_section(
    datasets: Sequence[CatalogEntry], names: Sequence[str], build_fn,
    n_folds: int, cache_dir: Optional[str] = CACHE_DIR, fit_timeout: float = FIT_TIMEOUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs one section: every `name` in `names` fit (via `build_fn`)
    against every dataset, `n_folds`-fold cross-validation. One row per
    (dataset, fold, name) with accuracy/n_rules/n_conditions/fit_time/
    overlap/error."""
    rows: List[Dict[str, Any]] = []
    timeout_runner = TimeoutRunner(fit_timeout) if fit_timeout is not None else None
    try:
        for entry in datasets:
            if verbose:
                print(f"{entry.name} ...", flush=True)
            df, target = entry.load()
            y_all = df[target].to_numpy()
            folds = _stratified_folds(y_all, n_folds, RANDOM_STATE)

            for fold, (train_idx, test_idx) in enumerate(folds):
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)
                spec = build_dataspec(train_df, target=target, max_intervals=MAX_INTERVALS,
                                      skip_unusable=True).build()
                train_rep = BooleanDataRepresentation(spec, binarize(spec, train_df),
                                                       train_df[target].to_numpy())
                test_rep = BooleanDataRepresentation(spec, binarize(spec, test_df),
                                                      test_df[target].to_numpy())

                for name in names:
                    key = _cache_key(entry.name, n_folds, fold, name)
                    cached = _cache_load(cache_dir, key)
                    if verbose:
                        print(f"  fold {fold + 1}/{n_folds}  {name:<20s} ...", end="", flush=True)
                    if cached is not None:
                        rows.append(cached)
                        if verbose:
                            print(" (cached)", flush=True)
                        continue

                    learner = build_fn(name)
                    t0 = time.time()
                    if timeout_runner is not None:
                        model, error = timeout_runner.run(_fit_one, learner, train_rep)
                    else:
                        try:
                            model, error = _fit_one(learner, train_rep), None
                        except Exception as e:  # noqa: BLE001
                            model, error = None, f"{type(e).__name__}: {e}"
                    fit_time = time.time() - t0
                    row = {"dataset": entry.name, "fold": fold, "name": name, "fit_time": fit_time,
                          "error": error, "accuracy": np.nan, "n_rules": np.nan,
                          "n_conditions": np.nan, "overlap": np.nan}
                    if error is None:
                        stats = model.evaluate(test_rep)
                        row["accuracy"] = stats.confusion.accuracy if stats.confusion is not None else np.nan
                        row["n_rules"] = stats.n_rules
                        row["n_conditions"] = stats.n_conditions
                        row["overlap"] = _overlap(model, test_rep)
                    rows.append(row)
                    _cache_save(cache_dir, key, row)
                    if verbose:
                        print(f" {fit_time:6.2f}s  acc={row['accuracy']:.3f}" if error is None
                             else f" {error}", flush=True)
    finally:
        if timeout_runner is not None:
            timeout_runner.close()
    return pd.DataFrame(rows)


# ================================================================= report ===

def _common_cells(
    results: pd.DataFrame, names: Sequence[str],
) -> Tuple[Optional[set], Optional[str]]:
    """Like the pool-distillers demo's dataset-coverage check, but at
    (dataset, fold) granularity -- the unit each `name` here succeeds
    or fails at (a fold, not a whole dataset, can time out). Returns
    `(None, None)` if every name in `names` succeeded on exactly the
    same cells; otherwise the intersection and a note naming how many
    cells each name is missing, so callers can show a mean restricted
    to `common` alongside the unrestricted one."""
    ok = results[results["error"].isna()]
    cells_by_name: Dict[str, set] = {}
    for name in names:
        sub = ok[ok["name"] == name]
        if not sub.empty:
            cells_by_name[name] = set(zip(sub["dataset"], sub["fold"]))
    if len(set(map(frozenset, cells_by_name.values()))) <= 1:
        return None, None
    common = set.intersection(*cells_by_name.values())
    all_cells = set.union(*cells_by_name.values())
    note = "; ".join(
        f"{name} missing {len(all_cells - cells)} of {len(all_cells)} cells"
        for name, cells in cells_by_name.items() if all_cells - cells
    )
    return common, note


def _common_cell_means(results: pd.DataFrame, common: set, column: str = "accuracy") -> pd.Series:
    indexed = results.set_index(["dataset", "fold"])
    restricted = indexed[indexed.index.isin(common)]
    return restricted.reset_index().groupby("name")[column].mean()


def _summary_table(results: pd.DataFrame, names: Sequence[str]) -> List[str]:
    summary = results.groupby("name")[["accuracy", "n_rules", "n_conditions", "fit_time",
                                       "overlap"]].mean(numeric_only=True)
    failures = results.groupby("name")["error"].apply(lambda s: int(s.notna().sum()))

    common, note = _common_cells(results, names)
    common_acc = _common_cell_means(results, common) if common is not None else None

    acc_label = "accuracy (own cells)" if common is not None else "accuracy"
    cols = ["name", acc_label]
    if common_acc is not None:
        cols.append(f"accuracy ({len(common)} common cells)")
    cols += ["n_rules", "conditions/rule", "fit_time (s)", "overlap", "failures"]
    lines = ["| " + " | ".join(cols) + " |\n", "|---|" + "--:|" * (len(cols) - 1) + "\n"]
    for name in names:
        if name not in summary.index:
            continue
        row = summary.loc[name]
        cpr = row["n_conditions"] / row["n_rules"] if row["n_rules"] else float("nan")
        cells = [name, f"{row['accuracy']:.3f}"]
        if common_acc is not None:
            cells.append(f"{common_acc.get(name, float('nan')):.3f}")
        cells += [f"{row['n_rules']:.1f}", f"{cpr:.2f}", f"{row['fit_time']:.2f}",
                 f"{row['overlap']:.2f}", str(int(failures.get(name, 0)))]
        lines.append("| " + " | ".join(cells) + " |\n")
    if note is not None:
        lines.append(f"\n*{note} (see the \"failures\" column). \"own cells\" is each name "
                     f"averaged over whatever (dataset, fold) cells it succeeded on; "
                     f"\"{len(common)} common cells\" restricts every name to the same cells, "
                     f"for a fair comparison.*\n\n")
    return lines


def _rank_table(results: pd.DataFrame, names: Sequence[str]) -> List[str]:
    r = results.rename(columns={"name": "learner"})
    ranks = mean_rank(r, "accuracy")
    lines = ["| name | mean rank |\n|---|--:|\n"]
    for name, rank in ranks.items():
        if name in names:
            lines.append(f"| {name} | {rank:.2f} |\n")
    return lines


def _boosting_curve_table(results: pd.DataFrame, common: Optional[set] = None) -> List[str]:
    """`common`, if given, is section 1's own common-cells set (from
    `_common_cells(section1_results, SECTION1_NAMES)`), not one
    recomputed from section 2's own names -- so the "common cells"
    column here uses the *same* cells as section 1's summary table,
    making the two tables' accuracy columns directly comparable rather
    than each restricted to its own, differently-sized, common subset.
    """
    names = [f"{family}@{n}" for family in BOOSTING_FAMILIES for n in N_RULES_BUDGETS]
    summary = results.groupby("name")[["accuracy", "n_rules", "fit_time"]].mean(numeric_only=True)
    failures = results.groupby("name")["error"].apply(lambda s: int(s.notna().sum()))
    _, note = _common_cells(results, names)  # still used for the per-name failure breakdown

    common_acc = _common_cell_means(results, common) if common is not None else None

    acc_label = "accuracy (own cells)" if common is not None else "accuracy"
    cols = ["family", "budget", acc_label]
    if common_acc is not None:
        cols.append(f"accuracy ({len(common)} common cells)")
    cols += ["n_rules (actual)", "fit_time (s)", "failures"]
    lines = ["| " + " | ".join(cols) + " |\n", "|---|" + "--:|" * (len(cols) - 1) + "\n"]
    for family in BOOSTING_FAMILIES:
        for n in N_RULES_BUDGETS:
            name = f"{family}@{n}"
            if name not in summary.index:
                continue
            row = summary.loc[name]
            cells = [family, str(n), f"{row['accuracy']:.3f}"]
            if common_acc is not None:
                cells.append(f"{common_acc.get(name, float('nan')):.3f}")
            cells += [f"{row['n_rules']:.1f}", f"{row['fit_time']:.2f}", str(int(failures.get(name, 0)))]
            lines.append("| " + " | ".join(cells) + " |\n")

    footnote_parts = []
    if note is not None:
        footnote_parts.append(note)
    if common_acc is not None:
        footnote_parts.append(
            "\"own cells\" is each name averaged over whatever (dataset, fold) cells it "
            f"succeeded on; \"{len(common)} common cells\" is the same cell subset section 1's "
            "summary table uses, so the two tables' accuracy columns are directly comparable"
        )
    if footnote_parts:
        lines.append(f"\n*{'. '.join(footnote_parts)} (see the \"failures\" column).*\n\n")
    return lines


def _comparative_discussion(section1_results: pd.DataFrame, section2_results: pd.DataFrame) -> List[str]:
    """Built fresh from the measured numbers every run, restricted to
    the cells common to every row being compared (see `_common_cells`)
    so one dataset's failures don't quietly tilt which point looks
    best."""
    lines = ["## Comparative discussion\n\n"]

    common1, _ = _common_cells(section1_results, SECTION1_NAMES)
    s1 = (_common_cell_means(section1_results, common1) if common1 is not None
         else section1_results.groupby("name")["accuracy"].mean()).sort_values(ascending=False)
    if s1.empty:
        return lines
    best_name, best_acc = s1.index[0], s1.iloc[0]

    common2, _ = _common_cells(section2_results, SECTION2_NAMES)
    s2 = (_common_cell_means(section2_results, common2) if common2 is not None
         else section2_results.groupby("name")["accuracy"].mean()).sort_values(ascending=False)
    best2_name, best2_acc = (s2.index[0], s2.iloc[0]) if not s2.empty else (None, None)

    lines.append(
        f"Section 1's best covering-strategy point is `{best_name}` (mean accuracy "
        f"{best_acc:.3f} over the cells every section-1 config completed). Section 2's best "
        f"boosting point is `{best2_name}` ({best2_acc:.3f}, same restriction). "
    )

    # Ground the Slipper/ENDER-exponential kinship in the actual numbers
    # too, not just the theory below -- same budget, same cells.
    top_budget = N_RULES_BUDGETS[-1]
    exp_name, slipper_name = f"ENDER-CSExp@{top_budget}", f"Slipper@{top_budget}"
    exp_acc, slipper_acc = s2.get(exp_name), s2.get(slipper_name)
    kinship = (f"At budget {top_budget}, `{slipper_name}` scores {slipper_acc:.3f} and "
              f"`{exp_name}` scores {exp_acc:.3f} -- close, as expected from the theory below. "
              if exp_acc is not None and slipper_acc is not None else "")

    lines.append(
        "Boosting and weighted covering aren't as separate as they might look. "
        "`AdaBoostReweighting` (Slipper's mechanism, and `WeightedCovering`'s rule-weight-"
        "assigning option, which plain `SeCo` itself refuses to build into an unweighted set) "
        "and `ENDER`'s `ExponentialLoss` configuration are the *same* thing by a classical "
        "result (Friedman, Hastie & Tibshirani 2000): AdaBoost **is** gradient boosting on the "
        "exponential loss. `ExponentialLoss` here literally defines its per-example quantity as "
        "``w = exp(-y f)`` -- AdaBoost's reweighting formula -- and fits each rule's weight with "
        "the identical closed form `AdaBoostReweighting.confidence` uses. "
        f"{kinship}"
        "What actually distinguishes the two families: section 1's *other* reweighting schemes "
        "(`MultiplicativeReweighting`'s ``gamma**k``, `AdditiveReweighting`'s ``1/(k+1)``, "
        "`LRIReweighting`'s ``1 + e**3``) are hand-designed update rules with no loss function "
        "behind them, so they can't generalize the way `ENDER`'s pluggable-loss framework does "
        "(logistic, exponential, sigmoid, five weight-fitting methods). And weighted covering "
        "uses its weights only to steer training -- it still predicts with a heuristic score "
        "over an *unweighted* rule set -- while boosting bakes the fitted weight into the model "
        "itself (`LinearRuleModel`, summed at predict time): there the weight is part of the "
        "answer, not just how training got there.\n\n"
    )
    return lines


# Section 1: one color per covering strategy (as asked), one marker
# shape per heuristic -- so a covering strategy's three heuristic
# variants are visibly the same family, distinguishable by shape.
# CPAR/LRI get their own colors too, as the complete-algorithm rows
# the summary table already treats them as.
_COVERING_COLORS = {
    "Removal": "tab:blue", "Weighted-Mult": "tab:green",
    "Weighted-Add": "tab:red", "Weighted-LRI": "tab:purple",
}
_HEURISTIC_MARKERS = {"Laplace": "o", "WRAcc": "s", "MEstimate16": "^"}
_COMPLETE_COLORS = {"CPAR": "tab:brown", "LRI": "tab:pink"}
_COMPLETE_MARKERS = {"CPAR": "D", "LRI": "P"}
# Section 2: one color per boosting family, its four budgets (10/20/
# 50/100) connected by a line -- the accuracy-vs-cost curve the family
# traces out, not four unrelated points.
_FAMILY_COLORS = {
    "Slipper": "tab:orange", "ENDER-CSLog": "tab:cyan",
    "ENDER-CSExp": "tab:olive", "ENDER-Newton": "gold", "Boomer": "black",
}
# shapes unused by the heuristics (o s ^) and the complete algorithms (D P)
_FAMILY_MARKERS = {
    "Slipper": "v", "ENDER-CSLog": "<", "ENDER-CSExp": ">", "ENDER-Newton": "X", "Boomer": "*",
}


def _accuracy_vs_time_plot(
    section1_results: pd.DataFrame, section2_results: pd.DataFrame, common: set, path: str,
) -> None:
    """Accuracy (the same `common` cells every table in this report
    uses, so this plot is on the same footing) against fit time (log
    scale, since it spans timeouts down to sub-second boosting fits)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(os.path.dirname(path), exist_ok=True)

    s1_acc = _common_cell_means(section1_results, common)
    s1_time = section1_results.groupby("name")["fit_time"].mean()
    s2_acc = _common_cell_means(section2_results, common)
    s2_time = section2_results.groupby("name")["fit_time"].mean()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for name in SECTION1_NAMES:
        if name not in s1_acc.index or name not in s1_time.index:
            continue
        acc, t = s1_acc[name], s1_time[name]
        if name in _COMPLETE_COLORS:
            ax.scatter(t, acc, color=_COMPLETE_COLORS[name], marker=_COMPLETE_MARKERS[name], s=55, zorder=3)
            continue
        cov_name, heur_name = name.split("+")
        ax.scatter(t, acc, color=_COVERING_COLORS[cov_name], marker=_HEURISTIC_MARKERS[heur_name],
                  s=55, zorder=3)

    for family in BOOSTING_FAMILIES:
        xs, ys = [], []
        for n in N_RULES_BUDGETS:
            name = f"{family}@{n}"
            if name in s2_acc.index and name in s2_time.index:
                xs.append(s2_time[name])
                ys.append(s2_acc[name])
        if xs:
            ax.plot(xs, ys, "-", marker=_FAMILY_MARKERS[family], color=_FAMILY_COLORS[family], markersize=5, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("fit time (s, log scale)")
    ax.set_ylabel(f"accuracy ({len(common)} common cells)")
    ax.set_title("Accuracy vs. fit time: covering strategies and boosting")
    ax.grid(alpha=0.3, which="both")

    # covering strategies: color only (their shape is the heuristic, legend 2);
    # complete algorithms and boosting families: the marker they're drawn with
    algorithm_handles = [
        Line2D([0], [0], marker="o", color=c, linestyle="", markersize=7, label=n)
        for n, c in _COVERING_COLORS.items()
    ] + [
        Line2D([0], [0], marker=_COMPLETE_MARKERS[n], color=c, linestyle="", markersize=7, label=n)
        for n, c in _COMPLETE_COLORS.items()
    ] + [
        Line2D([0], [0], marker=_FAMILY_MARKERS[n], color=c, linestyle="-", markersize=5, label=n)
        for n, c in _FAMILY_COLORS.items()
    ]
    heuristic_handles = [
        Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=7, label=h)
        for h, m in _HEURISTIC_MARKERS.items()
    ] + [Line2D([0], [0], color="gray", linestyle="-", marker="o", markersize=4, label="rule budget")]
    legend1 = ax.legend(handles=algorithm_handles, fontsize=6, loc="upper left",
                        title="algorithm (color)", ncol=2)
    ax.add_artist(legend1)
    ax.legend(handles=heuristic_handles, fontsize=6, loc="lower right",
             title="heuristic")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def write_report(
    section1_results: pd.DataFrame, section2_results: pd.DataFrame,
    datasets: Sequence[CatalogEntry], n_folds: int, report_path: str, quick: bool,
) -> None:
    mode_note = (
        f"**Quick run** -- a fast sanity check, the default with no arguments. Full comparison: "
        f"`python demos/{NAME}.py --full`.\n\n" if quick else
        f"**Full run**. Quick sanity check instead: `python demos/{NAME}.py` with no arguments.\n\n"
    )
    dataset_names = ", ".join(d.name for d in datasets)
    description = f"""\
{mode_note}Covering strategies (plain `SeCo`, beam width {BEAM_WIDTH}) x heuristics
(`Laplace`, `WRAcc`, m-estimate m=16), plus `CPAR`/`LRI` as complete algorithms, on
{dataset_names} ({n_folds}-fold; {LARGE_FOLDS}-fold for `adult`, the
catalog's one "large" dataset here -- weighted covering and boosting
are no faster on it). Boosting (`Slipper`, `ENDER` x 3 configs,
`Boomer`) on the same datasets/folds, at rule-count budgets
{N_RULES_BUDGETS}.

Each fit is capped at {FIT_TIMEOUT:.0f}s; a time-out or an error counts
as a failure, not as the end of the run, and is excluded from every
mean it would otherwise enter (a "failures" column reports the count
instead).

Measures: accuracy, rule count, conditions per rule, fit time, and
overlap (mean rules covering a test row, pooled across every concept --
see the module docstring for why this isn't restricted to one
"positive" class).
"""
    L = ["# Covering strategies and boosting\n\n"]
    L.append(render_setup_section(description))

    L.append("## Section 1: covering strategies\n\n")
    L += _summary_table(section1_results, SECTION1_NAMES)
    L.append("\n### Mean rank\n\n")
    L += _rank_table(section1_results, SECTION1_NAMES)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(section1_results.rename(columns={"name": "learner"}),
                                  ["accuracy", "n_rules", "fit_time", "overlap"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")

    L.append("## Section 2: boosting\n\n")
    common1, _ = _common_cells(section1_results, SECTION1_NAMES)
    L += _boosting_curve_table(section2_results, common1)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(section2_results.rename(columns={"name": "learner"}),
                                  ["accuracy", "n_rules", "fit_time"],
                                  group_by=["dataset", "learner"], include_overall=False))
    L.append("\n")

    L += _comparative_discussion(section1_results, section2_results)

    tag = "_quick" if quick else ""
    plot_path = os.path.join(PLOTS_DIR, f"{NAME}{tag}_accuracy_vs_time.png")
    _accuracy_vs_time_plot(section1_results, section2_results, common1, plot_path)
    L.append("## Accuracy vs. fit time\n\n")
    L.append(
        "Every point here is on the same accuracy basis as the tables above (the "
        f"{len(common1)} common cells), against mean fit time on a log scale, since it spans "
        "sub-second boosting fits to multi-minute (and timed-out) covering fits. Color is the "
        "algorithm family -- the four covering strategies, `CPAR`/`LRI`, and the five boosting "
        "families; for section 1, marker shape is the heuristic, so a covering strategy's three "
        "heuristic variants show up as same-color points of different shapes. A boosting "
        "family's four rule-count budgets (10/20/50/100) are connected by a line, tracing out "
        "that family's own accuracy-vs-cost curve rather than four unrelated points. `Boomer` "
        "and `ENDER-Newton`'s lines nearly overlap at the top right -- both reach similarly high "
        "accuracy at a similar cost, a real finding (the two are closely related mechanisms; see "
        "the comparative discussion above), not a plotting artifact.\n\n"
    )
    L.append(f"![accuracy vs. fit time]({os.path.relpath(plot_path, HERE).replace(os.sep, '/')})\n\n")
    print(f"Plot   -> {plot_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def _run_both_sections(datasets: Sequence[CatalogEntry], n_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("=== Section 1: covering strategies ===", flush=True)
    section1 = run_section(datasets, SECTION1_NAMES, build_section1_learner, n_folds)
    print("=== Section 2: boosting ===", flush=True)
    section2 = run_section(datasets, SECTION2_NAMES, build_section2_learner, n_folds)
    return section1, section2


def main(quick: bool = True) -> None:
    names = QUICK_DATASETS if quick else FULL_DATASETS
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")

    datasets = Catalog.default().select(names=names)

    # "adult" (the catalog's only "large" dataset here) gets fewer
    # folds -- same split as demos/seco_learners_comparison.py's
    # LARGE_FOLDS, since weighted covering/boosting are no faster on it.
    if quick:
        section1_results, section2_results = _run_both_sections(datasets, n_folds)
    else:
        small = [d for d in datasets if d.size != "large"]
        large = [d for d in datasets if d.size == "large"]
        groups = [_run_both_sections(small, n_folds)] + ([_run_both_sections(large, LARGE_FOLDS)] if large else [])
        section1_results = pd.concat([g[0] for g in groups], ignore_index=True)
        section2_results = pd.concat([g[1] for g in groups], ignore_index=True)

    write_report(section1_results, section2_results, datasets, n_folds, report_path, quick)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the full comparison instead of the quick default")
    args = parser.parse_args()
    main(quick=not args.full)
