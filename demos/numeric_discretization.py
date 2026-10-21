"""
demos/numeric_discretization.py
===============================

How should numeric attributes be discretized for rule learning? The
discretizers of `pyrulearn.data.discretization`, compared with fast
native rule learners -- Pypper, PFossil and PFoil -- in two sections:

1. **Discretizers and interval counts.** Six discretizers, each at
   3, 5, 8 and 12 intervals:

   - `tree` -- `TreeDiscretizer` (a single-feature decision tree),
     also unbounded (`max_intervals=None`: the full tree, a threshold at
     every point where the class changes);
   - `fusinter` -- `FusinterDiscretizer` (Zighed et al., 1998), also
     unbounded (`max_intervals=None`: FUSINTER's own criterion decides);
   - `ew` / `ew+bc` -- `EqualWidthDiscretizer`, plain and wrapped in
     `BoundaryCorrection` (each threshold moved to the nearest point
     where the class changes);
   - `ef` / `ef+bc` -- `EqualFrequencyDiscretizer`, plain and wrapped in
     `BoundaryCorrection`.

   On `FULL_DATASETS` (every small/medium catalog dataset with only
   numeric attributes, up to `FULL_MAX_ROWS_TIMES_FEATURES`).

2. **Per class vs. all classes.** `PerClass` runs a supervised
   discretizer once per class (that class against all others) and keeps
   the union of the thresholds, so each class gets cuts placed for it.
   `tree`, `fusinter` and `ef+bc` globally (3, 5, 8, 12 intervals, and
   FUSINTER unbounded) vs. per class (2, 3, 5 intervals per class), compared over
   the number of binary features they produce, on `MULTICLASS_DATASETS`
   (small/medium catalog datasets with numeric attributes and at least
   `MIN_CLASSES` classes).

3. **Against raw numeric data.** Weka's JRip and J48, which handle
   numeric attributes themselves, on the raw data (`raw`: Weka sets its
   own thresholds) vs. discretized by `tree` and by `ef+bc`, at 3, 5, 8
   and 12 intervals, on `SECTION3_DATASETS` (every small/medium catalog
   dataset with only numeric attributes, up to
   `MULTICLASS_MAX_ROWS_TIMES_FEATURES` rows times features). Needs Weka
   (`$WEKA_JAR`, `$WEKA_JAVA`).

Every column with at least three distinct values is discretized
(`run_cv(max_ordinal_values=2)`, so none is made ordinal). Each training
fold is discretized on its own (`run_cv`); the test fold is binarized
against that fold's thresholds.

Run: `python demos/numeric_discretization.py` with no arguments is the
**quick** default (`QUICK_DATASETS_SPEC`/`QUICK_MULTICLASS_SPEC`,
`QUICK_FOLDS`-fold); its report and plots go to
``numeric_discretization_quick_*`` and are not checked in. `--full` runs
the full comparison (`N_FOLDS`-fold) and writes the canonical
``numeric_discretization_report.md`` (plots in
``numeric_discretization_plots/``). `--n-jobs N` runs N fits at a time;
`--replot` rewrites the report and plots from the last run's saved
results (for the same mode), without running anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pyrulearn.data.discretization import (
    BoundaryCorrection, Discretizer, EqualFrequencyDiscretizer, EqualWidthDiscretizer,
    FusinterDiscretizer, PerClass, TreeDiscretizer,
)
from pyrulearn.experiments.catalog import MEDIUM_MAX, SMALL_MAX, Catalog
from pyrulearn.experiments.report import format_accuracy, format_mean_count, render_setup_section
from pyrulearn.data.io import binarize, write_arff
from pyrulearn.data.representation import HorizontalBitmapRepresentation
from pyrulearn.experiments.runner import _stratified_folds, run_cv
from pyrulearn.interfaces.weka import J48Importer, JRipImporter, WekaJ48, WekaJRip, _resolve_weka, _weka_cli_options
from pyrulearn.experiments.stats import critical_difference, friedman_test, mean_rank
from pyrulearn.learners.seco import PFoil, PFossil, Pypper

NAME = "numeric_discretization"
RANDOM_STATE = 0
N_FOLDS = 10
QUICK_FOLDS = 3
FIT_TIMEOUT = 300.0  # seconds per fit

LEVELS = (3, 5, 8, 12)
UNBOUNDED = "∞"  # the level label of max_intervals=None
PER_CLASS_LEVELS = (2, 3, 5)

LEARNERS = ("Pypper", "PFossil", "PFoil")


def build_learners():
    return [Pypper(random_state=RANDOM_STATE), PFossil(random_state=RANDOM_STATE),
            PFoil(random_state=RANDOM_STATE)]


# -- section 1: the discretizers ---------------------------------------------

#: family -> (label, factory taking max_intervals, whether it has an unbounded variant)
FAMILIES = {
    "tree": ("tree", TreeDiscretizer, True),
    "fusinter": ("FUSINTER", lambda k: FusinterDiscretizer(max_intervals=k), True),
    "ew": ("equal width", EqualWidthDiscretizer, False),
    "ew+bc": ("equal width + boundary correction",
              lambda k: BoundaryCorrection(EqualWidthDiscretizer(k)), False),
    "ef": ("equal frequency", EqualFrequencyDiscretizer, False),
    "ef+bc": ("equal frequency + boundary correction",
              lambda k: BoundaryCorrection(EqualFrequencyDiscretizer(k)), False),
}


def section1_preps() -> List[Tuple[str, str, object, Discretizer]]:
    """(prep label, family, level, discretizer) for section 1."""
    preps = []
    for family, (_, factory, unbounded) in FAMILIES.items():
        for k in LEVELS:
            preps.append((f"{family}@{k}", family, k, factory(k)))
        if unbounded:
            preps.append((f"{family}@{UNBOUNDED}", family, UNBOUNDED, factory(None)))
    return preps


# Every small/medium catalog dataset with only numeric attributes ...
FULL_SELECT_KWARGS = dict(attributes="numeric", size=["small", "medium"])
# ... except those `n_instances * n_features` puts above this: that
# predicts the cost of a fold (rows times binary features) far better than
# the row count alone, and there is a natural gap in the catalog here
# (57,831 to 83,160).
FULL_MAX_ROWS_TIMES_FEATURES = 60_000
QUICK_DATASETS_SPEC = "iris,wine,glass,blood-transfusion-service-center,ionosphere"


def full_datasets():
    return [e for e in Catalog.default().select(**FULL_SELECT_KWARGS)
            if e.n_instances * e.n_features <= FULL_MAX_ROWS_TIMES_FEATURES]


# -- section 2: per class vs. all classes ------------------------------------

PER_CLASS_FAMILIES = ("tree", "fusinter", "ef+bc")
# the global unbounded variant in section 2: FUSINTER only -- the full tree
# cuts at nearly every distinct value of a continuous many-class column
# (tens of thousands of binary features on mfeat-fourier), which takes hours
# to fit and is section 1's extreme case anyway
PER_CLASS_UNBOUNDED = ("fusinter",)
MIN_CLASSES = 5
# a larger cap than section 1's: Pypper, PFossil and PFoil are fast enough,
# and many-class datasets tend to be wide; it leaves out only the three
# largest candidates (first-order-theorem-proving, GesturePhaseSegmentation-
# Processed, optdigits), where unbounded and per-class discretization
# multiply the binary features
MULTICLASS_MAX_ROWS_TIMES_FEATURES = 250_000
QUICK_MULTICLASS_SPEC = "glass,ecoli,yeast"


def section2_preps() -> List[Tuple[str, str, str, object, Discretizer]]:
    """(prep label, family, scope "global"/"per class", level, discretizer)."""
    preps = []
    for family in PER_CLASS_FAMILIES:
        _, factory, unbounded = FAMILIES[family]
        for k in LEVELS:
            preps.append((f"{family}@{k}", family, "global", k, factory(k)))
        if unbounded and family in PER_CLASS_UNBOUNDED:
            preps.append((f"{family}@{UNBOUNDED}", family, "global", UNBOUNDED, factory(None)))
        for k in PER_CLASS_LEVELS:
            preps.append((f"pc-{family}@{k}", family, "per class", k, PerClass(factory(k))))
    return preps


def multiclass_datasets():
    """Small/medium datasets with numeric attributes (only, or mixed with
    nominal ones) and at least `MIN_CLASSES` classes, up to
    `MULTICLASS_MAX_ROWS_TIMES_FEATURES` rows times features."""
    return [e for e in Catalog.default().select(attributes=["numeric", "mixed"], size=["small", "medium"],
                                                 task="multiclass")
            if (e.n_classes or 0) >= MIN_CLASSES
            and e.n_instances * e.n_features <= MULTICLASS_MAX_ROWS_TIMES_FEATURES]


# -- section 3: against JRip and J48 on the raw data -------------------------

WEKA_LEARNERS = ("Weka:JRip", "Weka:J48")
SECTION3_FAMILIES = ("tree", "ef+bc")
QUICK_SECTION3_SPEC = "iris,wine,glass,diabetes,blood-transfusion-service-center"
#: Weka's class and the importer reading its printed model, for `raw`
RAW_WEKA = {"Weka:JRip": ("weka.classifiers.rules.JRip", JRipImporter),
            "Weka:J48": ("weka.classifiers.trees.J48", J48Importer)}
#: part of the `raw` cache keys -- bump it when the `raw` procedure changes
RAW_VERSION = "raw-1"


def build_weka_learners():
    return [WekaJRip(), WekaJ48()]  # Weka from $WEKA_JAR / $WEKA_JAVA


def section3_preps() -> List[Tuple[str, str, object, Discretizer]]:
    """(prep label, family, level, discretizer) for section 3's discretized runs."""
    return [(f"{family}@{k}", family, k, FAMILIES[family][1](k)) for family in SECTION3_FAMILIES for k in LEVELS]


def section3_datasets():
    """Every small/medium catalog dataset with only numeric attributes, up to
    `MULTICLASS_MAX_ROWS_TIMES_FEATURES` rows times features."""
    return [e for e in Catalog.default().select(attributes="numeric", size=["small", "medium"])
            if e.loadable and e.n_instances * e.n_features <= MULTICLASS_MAX_ROWS_TIMES_FEATURES]


def _fit_weka_raw(algorithm: str, train_df, target: str):
    """`raw`: `train_df` written to ARFF as it is (real numeric columns) and
    Weka run on it directly -- `WekaJRip`/`WekaJ48` expect binarized data.
    The importer, given no `DataSpec`, builds one from the thresholds Weka
    printed. Returns `(model, dataspec)`; raises `subprocess.TimeoutExpired`
    or `RuntimeError`."""
    weka_class, importer_cls = RAW_WEKA[algorithm]
    jar, java = _resolve_weka(None, None)
    with tempfile.TemporaryDirectory(prefix="weka_raw_") as d:
        arff = os.path.join(d, "train.arff")
        write_arff(train_df, target, arff)
        cmd = [java, "-Duser.language=en", "-Duser.country=US", "-cp", jar, weka_class,
               "-t", arff, "-no-cv", *_weka_cli_options(None)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT)
    if result.returncode != 0 or "=== " not in result.stdout:
        raise RuntimeError(f"{weka_class} produced no usable model (exit {result.returncode}):\n"
                           f"{result.stdout}\n{result.stderr}")
    importer = importer_cls()
    model = importer.parse(result.stdout)
    return model, importer.dataspec


def _raw_cell(entry, n_folds: int, fold: int, algorithm: str, train_df, test_df, target: str) -> dict:
    """One `raw` fit and its evaluation on the test fold (binarized against
    the thresholds Weka chose), as a result row; cached in `CACHE_DIR`."""
    payload = "|".join([RAW_VERSION, entry.name, str(entry.openml_id), str(entry.openml_version),
                        str(n_folds), str(fold), str(RANDOM_STATE), algorithm])
    path = os.path.join(CACHE_DIR, f"raw_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    row = {"dataset": entry.name, "fold": fold, "algorithm": algorithm, "error": None,
           "accuracy": np.nan, "n_rules": np.nan, "n_conditions": np.nan}
    t0 = time.time()
    try:
        model, dataspec = _fit_weka_raw(algorithm, train_df, target)
        row["fit_time"] = time.time() - t0
        rep = HorizontalBitmapRepresentation(dataspec, binarize(dataspec, test_df), test_df[target].to_numpy())
        stats = model.evaluate(rep)
        row.update(accuracy=stats.confusion.accuracy if stats.confusion is not None else np.nan,
                   n_rules=stats.n_rules, n_conditions=stats.n_conditions)
    except subprocess.TimeoutExpired:
        row.update(fit_time=time.time() - t0, error="timeout")
    except Exception as exc:  # recorded, not fatal -- like run_cv
        row.update(fit_time=time.time() - t0, error=f"{type(exc).__name__}: {exc}"[:300])
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, default=float)
    return row


def run_raw(datasets, n_folds: int, verbose: bool = True) -> pd.DataFrame:
    """`raw` for JRip and J48: the same folds as `run_cv` (`_stratified_folds`,
    `RANDOM_STATE`), `N_JOBS` Weka processes at a time."""
    if verbose:
        print("== raw: Weka's own thresholds", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(N_JOBS.rstrip("!")))) as pool:
        for entry in datasets:
            df, target = entry.load()
            folds = _stratified_folds(df[target].to_numpy(), n_folds, RANDOM_STATE)
            jobs = []
            for fold, (train_idx, test_idx) in enumerate(folds):
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)
                for algorithm in WEKA_LEARNERS:
                    jobs.append(pool.submit(_raw_cell, entry, n_folds, fold, algorithm, train_df, test_df, target))
            for job in jobs:
                row = job.result()
                rows.append(row)
                if verbose:
                    outcome = row["error"] or f"acc={100 * row['accuracy']:.2f}%"
                    print(f"  {row['dataset']} fold {row['fold'] + 1}/{n_folds}  {row['algorithm']:10s} "
                          f"{row['fit_time']:6.2f}s  {outcome}", flush=True)
    out = pd.DataFrame(rows)
    out["prep"], out["family"], out["level"], out["n_features"] = "raw", "raw", "raw", np.nan
    return out


# -- running -------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, f"_{NAME}_cv_cache")

# fits run at the same time (`run_cv`'s `n_jobs`; "N!" forces N past the available cores);
# set by --n-jobs
N_JOBS = "1"


def _n_jobs():
    return N_JOBS if N_JOBS.endswith("!") else int(N_JOBS)


def _feature_count(model, test_rep) -> Dict[str, int]:
    """`run_cv`'s `measure_fn`: the number of binary features of the fold."""
    return {"n_features": test_rep.spec.n_features}


def run_preps(preps, datasets, n_folds: int, verbose: bool = True, learners=build_learners) -> pd.DataFrame:
    """One `run_cv` per preparation, all learners; the results tagged with
    the preparation's columns (`prep` and whatever else each tuple
    carries, see `section1_preps`/`section2_preps`)."""
    frames = []
    for prep in preps:
        label, discretizer = prep[0], prep[-1]
        if verbose:
            print(f"== {label}: {discretizer!r}", flush=True)
        results = run_cv(learners(), datasets, n_folds=n_folds, fit_timeout=FIT_TIMEOUT,
                         discretizer=discretizer, max_ordinal_values=2, random_state=RANDOM_STATE,
                         cache_dir=CACHE_DIR, measure_fn=_feature_count, n_jobs=_n_jobs(), verbose=verbose)
        results = results.rename(columns={"learner": "algorithm"})
        results["prep"] = label
        for column, value in zip(_PREP_COLUMNS[len(prep)], prep[1:-1]):
            results[column] = value
        frames.append(results)
    out = pd.concat(frames, ignore_index=True)
    out["level"] = out["level"].astype(str)
    return out


_PREP_COLUMNS = {4: ("family", "level"), 5: ("family", "scope", "level")}


# -- report --------------------------------------------------------------------

def _ranked(results, algorithm: str) -> pd.DataFrame:
    """`results` of one learner with `learner` = prep, for the rank statistics."""
    sub = results[results["algorithm"] == algorithm].copy()
    sub["learner"] = sub["prep"]
    return sub


def _summary_table(results, preps: List[str], extra_columns: Dict[str, str], learners=LEARNERS) -> List[str]:
    """Rows: preparations; per learner its mean accuracy and mean rank (among
    these preparations); then the mean binary features and rules."""
    stats = {a: (mean_rank(_ranked(results, a), "accuracy"),
                 _ranked(results, a).groupby("learner")["accuracy"].mean())
             for a in learners}
    head = ["prep", *extra_columns.values()]
    for a in learners:
        head += [f"{a} acc.", f"{a} rank"]
    head += ["features", "rules (mean)"]
    lines = ["| " + " | ".join(head) + " |\n",
             "|" + "|".join(["---"] * (1 + len(extra_columns)) + ["--:"] * (len(head) - 1 - len(extra_columns))) + "|\n"]
    per_prep = results.groupby("prep")
    for prep in preps:
        rows = per_prep.get_group(prep)
        cells = [prep] + [str(rows[c].iloc[0]) for c in extra_columns]
        for a in learners:
            ranks, acc = stats[a]
            cells += [format_accuracy(acc.get(prep, np.nan)), f"{ranks.get(prep, np.nan):.2f}"]
        n_features = rows["n_features"].mean()
        cells += [format_mean_count(n_features) if n_features == n_features else "--",
                  format_mean_count(rows["n_rules"].mean())]
        lines.append("| " + " | ".join(cells) + " |\n")
    return lines


def _friedman_lines(results, n_preps: int, learners=LEARNERS) -> List[str]:
    lines = []
    for a in learners:
        sub = _ranked(results, a)
        try:
            f = friedman_test(sub, "accuracy")
            cd = critical_difference(f.n_learners, f.n_datasets)
            lines.append(f"- {a}: Friedman p = {f.pvalue:.2g} over {f.n_datasets} datasets; "
                         f"critical difference of the mean ranks {cd:.2f} ({f.n_learners} preparations).\n")
        except ValueError as exc:
            lines.append(f"- {a}: no Friedman test ({exc}).\n")
    return lines


def _datasets_text(section: int, datasets, quick: bool) -> str:
    """How a section's datasets were selected, and which they are."""
    names = ", ".join(sorted(d.name for d in datasets))
    if quick:
        return f"Datasets (quick run): {len(datasets)} small datasets picked by hand for a fast check -- {names}.\n\n"
    size = (f"small or medium-sized in the catalog (fewer than {SMALL_MAX:,} rows, or {SMALL_MAX:,} to "
            f"{MEDIUM_MAX:,})")
    criteria = {
        1: (f"has only numeric attributes, and has at most {FULL_MAX_ROWS_TIMES_FEATURES:,} rows times "
            "attributes -- which predicts the cost of a fold far better than the row count alone; the "
            "catalog has a natural gap there (57,831 to 83,160)"),
        2: (f"has numeric attributes (only, or together with nominal ones), at least {MIN_CLASSES} classes, "
            f"and at most {MULTICLASS_MAX_ROWS_TIMES_FEATURES:,} rows times attributes -- which leaves out "
            "only the three largest candidates (first-order-theorem-proving, "
            "GesturePhaseSegmentationProcessed, optdigits)"),
        3: (f"has only numeric attributes, and has at most {MULTICLASS_MAX_ROWS_TIMES_FEATURES:,} rows times "
            "attributes -- section 1's selection with section 2's larger size limit"),
    }[section]
    return (f"Datasets: every catalog dataset that is {size}, {criteria}: {len(datasets)} datasets -- "
            f"{names}.\n\n")


def _setup(datasets1, datasets2, n_folds: int, quick: bool) -> str:
    mode = (f"**Quick run** (a few small datasets per section, {n_folds}-fold) -- a fast "
            f"sanity check, the default with no arguments. Full comparison: `python demos/{NAME}.py --full`.\n\n"
            if quick else
            f"**Full run** ({n_folds}-fold). Quick sanity check instead: `python demos/{NAME}.py` with no "
            "arguments.\n\n")
    concurrency = ("Fits run one at a time." if N_JOBS.rstrip("!") == "1" else
                   f"Fits run {N_JOBS.rstrip('!')} at a time (`run_cv`'s `n_jobs`), each in its own process.")
    return f"""\
{mode}How to discretize numeric attributes for rule learning: the discretizers of
`pyrulearn.data.discretization` with three fast native rule learners --
Pypper (RIPPER), PFossil and PFoil -- at their defaults (sections 1 and 2),
and the most promising of them against Weka's JRip and J48 on the raw
numeric data (section 3).

Every column with at least three distinct values is discretized (none is
made ordinal); each training fold is discretized on its own, and the test
fold is binarized against that fold's thresholds. Each threshold `t` gives
two binary features, `x>=t` and `x<t`. Each fit is capped at
{FIT_TIMEOUT:.0f}s; a timeout counts as a failure. {concurrency}
"""


def write_report(res1, res2, res3, datasets1, datasets2, datasets3, n_folds: int, quick: bool,
                 paths: Dict[str, str]) -> None:
    plots_dir = os.path.basename(PLOTS_DIR)
    lines = ["# Numeric attributes: discretizers for rule learning\n\n"]
    lines.append(render_setup_section(_setup(datasets1, datasets2, n_folds, quick)))

    # section 1
    preps1 = [p[0] for p in section1_preps()]
    lines.append("## 1. Discretizers and interval counts\n\n")
    lines.append(
        _datasets_text(1, datasets1, quick)
        + "Discretizers, each at " + ", ".join(str(k) for k in LEVELS) + " intervals: "
        + "; ".join(f"`{f}` -- {label}" for f, (label, _, _) in FAMILIES.items())
        + f". `tree` and `fusinter` also unbounded (`{UNBOUNDED}`): the full tree, which places a threshold "
        "at every point where the class changes, and FUSINTER deciding the number of intervals itself.\n\n")
    for a in LEARNERS:
        lines.append(f"![{a}: accuracy vs. intervals]({plots_dir}/{os.path.basename(paths[f's1_{a}'])})\n\n")
    lines.append("Mean accuracy over all datasets and folds, and mean rank among the "
                 f"{len(preps1)} preparations (per learner; 1 = best):\n\n")
    lines += _summary_table(res1, preps1, {})
    lines.append("\n`features` -- mean number of binary features per fold; `rules` -- mean over the three "
                 "learners.\n\n")
    lines += _friedman_lines(res1, len(preps1))
    lines.append("\n")

    # section 2
    preps2 = [p[0] for p in section2_preps()]
    lines.append("## 2. Per class vs. all classes\n\n")
    lines.append(
        _datasets_text(2, datasets2, quick)
        + "`PerClass` (`pc-`) runs a discretizer once per class, on that class against all others, and keeps "
        "the union of the thresholds (one per gap between observed values). "
        + ", ".join(f"`{f}`" for f in PER_CLASS_FAMILIES)
        + " globally at " + ", ".join(str(k) for k in LEVELS) + f" intervals (`fusinter` also unbounded, "
        f"`{UNBOUNDED}`; the full tree, which cuts at nearly every distinct value of a continuous many-class "
        "column, is left out here), and per class at " + ", ".join(str(k) for k in PER_CLASS_LEVELS)
        + " intervals per class. Plotted over the number of binary features they produce, so a per-class "
        "and a global discretization of the same size can be compared directly.\n\n")
    lines.append(f"![per class vs. all]({plots_dir}/{os.path.basename(paths['s2'])})\n\n")
    lines += _summary_table(res2, preps2, {"scope": "scope"})
    lines.append("\n")
    lines += _friedman_lines(res2, len(preps2))
    lines.append(
        "\n**Reading.** For equal frequency, per class does not pay off. It is unsupervised: per class, "
        "only the boundary correction depends on the class, so every class's cuts start from the same "
        "equal-frequency positions and end up close together -- with two intervals per class, all near the "
        "median. That gives more thresholds in about the same places, and lower accuracy than a global "
        "discretization with as many features. The decision tree and FUSINTER place their cuts for each "
        "class, and there per class is competitive and sometimes the best preparation, mostly at small "
        "numbers of intervals per class -- though not significantly, and at several times the features of a "
        "global discretization of similar accuracy.\n\n"
        "**Tentative conclusion.** Pooling the per-class thresholds may dilute their gain: a learner "
        "working on one class searches every class's cuts, most of them placed for other classes. Separate "
        "feature sets per class -- a learner with a dedicated target class seeing only the thresholds placed "
        "for it -- might pay off more. Not tested here; see *Per-class feature sets* in `ROADMAP.md`.\n")
    # section 3
    preps3 = ["raw"] + [p[0] for p in section3_preps()]
    lines.append("\n## 3. Against JRip and J48 on the raw data\n\n")
    lines.append(
        _datasets_text(3, datasets3, quick)
        + "Weka's JRip and J48 handle numeric attributes themselves: `raw` gives them the original numeric "
        "columns, and they choose their own thresholds while learning. Against that, the same data "
        "discretized per training fold by `tree` (decision tree) and by `ef+bc` (equal frequency + boundary "
        "correction) -- the strongest supervised and unsupervised discretizers of section 1 -- at "
        + ", ".join(str(k) for k in LEVELS) + " intervals, then binarized. Weka's models of `raw` are "
        "evaluated by binarizing the test fold against the thresholds they use. A missing value: Weka "
        "handles it its own way when learning; at test time it covers no threshold feature.\n\n")
    lines.append(f"![JRip and J48: raw vs. discretized]({plots_dir}/{os.path.basename(paths['s3'])})\n\n")
    lines += _summary_table(res3, preps3, {}, learners=WEKA_LEARNERS)
    lines.append("\n`features` -- mean number of binary features per fold (`--` for `raw`); `rules` -- mean "
                 "over both learners (for J48, its tree's leaves).\n\n")
    lines += _friedman_lines(res3, len(preps3), learners=WEKA_LEARNERS)
    failures = res3[res3["error"].notna()]
    if len(failures):
        lines.append(f"\nFailed fits: {len(failures)} ("
                     + ", ".join(f"{d}/{a}/{p}" for d, a, p in failures[["dataset", "algorithm", "prep"]]
                                 .drop_duplicates().itertuples(index=False)) + ").\n")

    with open(paths["report"], "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {paths['report']}")


# -- plots ---------------------------------------------------------------------

#: the discretizers' names in the plots
FAMILY_LABELS = {"tree": "decision tree", "fusinter": "FUSINTER", "ew": "equal width",
                 "ew+bc": "equal width + boundary corr.", "ef": "equal frequency",
                 "ef+bc": "equal frequency + boundary corr."}
# a base discretizer and its boundary-corrected version share a color: solid
# line for the base, dashed for the corrected one
FAMILY_COLORS = {"tree": "tab:blue", "fusinter": "tab:red", "ew": "tab:green", "ew+bc": "tab:green",
                 "ef": "tab:orange", "ef+bc": "tab:orange"}
FAMILY_STYLES = {"tree": ("-", "o"), "fusinter": ("-", "D"), "ew": ("-", "s"), "ew+bc": ("--", "s"),
                 "ef": ("-", "^"), "ef+bc": ("--", "^")}


#: the four methods, by their base family (the color and marker they share
#: with their boundary-corrected version)
METHODS = {"tree": "decision tree", "fusinter": "FUSINTER", "ew": "equal width", "ef": "equal frequency"}


def write_section1_plots(results, paths: Dict[str, str], n_attributes: Dict[str, int]) -> None:
    """One plot per learner: mean accuracy over the interval count, per
    discretizer. The unbounded points are labelled with their mean number
    of intervals per attribute, comparable to the capped levels: per
    dataset ``n_features / (2 * attributes) + 1`` -- every threshold gives
    two features (`n_attributes`: dataset -> number of attributes) --
    averaged over the datasets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    os.makedirs(PLOTS_DIR, exist_ok=True)
    positions = {str(k): i for i, k in enumerate(LEVELS)}
    positions[UNBOUNDED] = len(LEVELS) + 0.4
    intervals = None
    if "n_features" in results:
        per_dataset = results.groupby(["family", "level", "dataset"])["n_features"].mean().reset_index()
        per_dataset["intervals"] = (per_dataset["n_features"]
                                    / (2 * per_dataset["dataset"].map(n_attributes)) + 1)
        intervals = per_dataset.groupby(["family", "level"])["intervals"].mean()
    for a in LEARNERS:
        sub = results[results["algorithm"] == a]
        mean = sub.groupby(["family", "level"])["accuracy"].mean()
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for family, (_, _, unbounded) in FAMILIES.items():
            linestyle, marker = FAMILY_STYLES[family]
            color = FAMILY_COLORS[family]
            levels = [str(k) for k in LEVELS]
            xs = list(range(len(LEVELS)))
            ys = [mean.get((family, k), np.nan) for k in levels]
            ax.plot(xs, ys, linestyle=linestyle, marker=marker, color=color)
            if unbounded and (family, UNBOUNDED) in mean.index:
                x_inf, y_inf = positions[UNBOUNDED], mean[(family, UNBOUNDED)]
                ax.plot([xs[-1], x_inf], [ys[-1], y_inf], linestyle=":", color=color)
                ax.plot([x_inf], [y_inf], marker=marker, color=color, linestyle="none", markersize=8)
                levels.append(UNBOUNDED); xs.append(x_inf); ys.append(y_inf)
                if intervals is not None and (family, UNBOUNDED) in intervals.index:
                    v = intervals[(family, UNBOUNDED)]
                    ax.annotate(f"{v:.1f}" if v < 20 else f"{v:.0f}", (x_inf, y_inf), fontsize=8,
                                color=color, xytext=(7, -3), textcoords="offset points")
        ax.axvline(len(LEVELS) - 0.3, color="grey", linewidth=0.8, alpha=0.5)
        ax.set_xlim(-0.3, positions[UNBOUNDED] + 0.6)  # room for the unbounded points' labels
        ax.set_xticks(list(positions.values()), list(positions.keys()))
        ax.set_xlabel(f"intervals (max_intervals; {UNBOUNDED} = unbounded, labelled with the mean intervals "
                      "per attribute)" if intervals is not None else
                      f"intervals (max_intervals; {UNBOUNDED} = unbounded)")
        ax.set_ylabel("mean test accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(f"{a}: accuracy vs. discretization")
        # the four methods by color and marker, plain vs. boundary-corrected by
        # line style: three columns, filled column by column
        handles = [Line2D([], [], color=FAMILY_COLORS[f], marker=FAMILY_STYLES[f][1], linestyle="-", label=label)
                   for f, label in METHODS.items()]
        handles += [Line2D([], [], color="grey", linestyle="-", label="plain"),
                    Line2D([], [], color="grey", linestyle="--", label="+ boundary correction")]
        ax.legend(handles=handles, fontsize=7.5, ncol=3, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(paths[f"s1_{a}"], dpi=110)
        plt.close(fig)
        print(f"Plot   -> {paths[f's1_{a}']}")


def write_section2_plot(results, paths: Dict[str, str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter, PercentFormatter, ScalarFormatter

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, len(LEARNERS), figsize=(5.2 * len(LEARNERS), 4.8), sharey=False)
    for ax, a in zip(axes, LEARNERS):
        sub = results[results["algorithm"] == a]
        mean = sub.groupby(["family", "scope", "level"])[["accuracy", "n_features"]].mean()
        for family in PER_CLASS_FAMILIES:
            color = FAMILY_COLORS[family]
            for scope, (linestyle, marker) in (("global", ("-", "o")), ("per class", ("--", "^"))):
                levels = [str(k) for k in (LEVELS if scope == "global" else PER_CLASS_LEVELS)]
                if scope == "global" and (family, scope, UNBOUNDED) in mean.index:
                    levels.append(UNBOUNDED)
                pts = [mean.loc[(family, scope, k)] for k in levels if (family, scope, k) in mean.index]
                ax.plot([p["n_features"] for p in pts], [p["accuracy"] for p in pts],
                        linestyle=linestyle, marker=marker, color=color)
                for k, p in zip(levels, pts):
                    ax.annotate(k, (p["n_features"], p["accuracy"]), fontsize=7, color=color,
                                xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("mean binary features (log scale)")
        ax.set_ylabel("mean test accuracy")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(a)
        ax.grid(alpha=0.3, which="both")
    handles = [Line2D([], [], color=FAMILY_COLORS[f], marker="o", linestyle="none", label=FAMILY_LABELS[f])
               for f in PER_CLASS_FAMILIES]
    handles += [Line2D([], [], color="grey", linestyle="-", marker="o", label="global (intervals)"),
                Line2D([], [], color="grey", linestyle="--", marker="^", label="per class (intervals per class)")]
    axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.suptitle("per class vs. all classes: accuracy over the number of binary features")
    fig.tight_layout()
    fig.savefig(paths["s2"], dpi=110)
    plt.close(fig)
    print(f"Plot   -> {paths['s2']}")


#: section 3's plot: a color per learner, a line style per preparation
WEKA_COLORS = {"Weka:JRip": "tab:blue", "Weka:J48": "tab:orange"}
SECTION3_STYLES = {"raw": ("-", None), "tree": ("--", "o"), "ef+bc": ("-.", "^")}
SECTION3_LABELS = {"raw": "raw (Weka's own thresholds)", "tree": "decision tree",
                   "ef+bc": "equal frequency + boundary corr."}


def write_section3_plots(results, paths: Dict[str, str]) -> None:
    """One plot: per Weka learner (a color each) `tree` and `ef+bc` over the
    interval count and `raw` as a horizontal line (a line style each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    os.makedirs(PLOTS_DIR, exist_ok=True)
    mean = results.groupby(["algorithm", "family", "level"])["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for a, color in WEKA_COLORS.items():
        for family in SECTION3_FAMILIES:
            linestyle, marker = SECTION3_STYLES[family]
            ys = [mean.get((a, family, str(k)), np.nan) for k in LEVELS]
            ax.plot(range(len(LEVELS)), ys, linestyle=linestyle, marker=marker, color=color)
        ax.axhline(mean.get((a, "raw", "raw"), np.nan), color=color, linestyle=SECTION3_STYLES["raw"][0],
                   linewidth=1.5, alpha=0.8)
    ax.set_xticks(range(len(LEVELS)), [str(k) for k in LEVELS])
    ax.set_xlabel("intervals (max_intervals)")
    ax.set_ylabel("mean test accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("JRip and J48: raw numeric data vs. discretized")
    # two legend groups: the learners by color, the preparations by (grey) line style
    handles = [Line2D([], [], color=color, marker="o", linestyle="none", label=a.split(":")[-1])
               for a, color in WEKA_COLORS.items()]
    handles += [Line2D([], [], color="grey", linestyle=SECTION3_STYLES[f][0], marker=SECTION3_STYLES[f][1],
                       label=SECTION3_LABELS[f]) for f in ("raw", *SECTION3_FAMILIES)]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(paths["s3"], dpi=110)
    plt.close(fig)
    print(f"Plot   -> {paths['s3']}")


def _paths(quick: bool) -> Dict[str, str]:
    q = "_quick" if quick else ""
    paths = {"report": os.path.join(HERE, f"{NAME}{q}_report.md"),
             "s2": os.path.join(PLOTS_DIR, f"{NAME}{q}_per_class.png")}
    for a in LEARNERS:
        paths[f"s1_{a}"] = os.path.join(PLOTS_DIR, f"{NAME}{q}_{a.lower()}.png")
    paths["s3"] = os.path.join(PLOTS_DIR, f"{NAME}{q}_raw_vs_discretized.png")
    return paths


def _results_path(quick: bool, section: int) -> str:
    """Where a run keeps its results table, so `--replot` can redraw the
    report and plots without recomputing (in the cache directory, not
    checked in)."""
    return os.path.join(CACHE_DIR, f"results{'_quick' if quick else ''}_section{section}.pkl")


def main(full: bool = False, replot: bool = False) -> None:
    quick = not full
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    catalog = Catalog.default()
    datasets1 = catalog.parse(QUICK_DATASETS_SPEC) if quick else full_datasets()
    datasets2 = catalog.parse(QUICK_MULTICLASS_SPEC) if quick else multiclass_datasets()
    datasets3 = catalog.parse(QUICK_SECTION3_SPEC) if quick else section3_datasets()
    paths = _paths(quick)
    if replot:
        res1, res2, res3 = (pd.read_pickle(_results_path(quick, i)) for i in (1, 2, 3))
    else:
        res1 = run_preps(section1_preps(), datasets1, n_folds)
        res2 = run_preps(section2_preps(), datasets2, n_folds)
        res3 = pd.concat([run_raw(datasets3, n_folds),
                          run_preps(section3_preps(), datasets3, n_folds, learners=build_weka_learners)],
                         ignore_index=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        for i, res in enumerate((res1, res2, res3), start=1):
            res.to_pickle(_results_path(quick, i))
    write_report(res1, res2, res3, datasets1, datasets2, datasets3, n_folds, quick, paths)
    write_section1_plots(res1, paths, {e.name: e.n_features for e in datasets1})
    write_section2_plot(res2, paths)
    write_section3_plots(res3, paths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="run the full comparison instead of the quick default")
    parser.add_argument("--n-jobs", default="1",
                        help="fits to run at the same time (default 1); capped at the available cores "
                             "unless given as 'N!'")
    parser.add_argument("--replot", action="store_true",
                        help="rewrite the report and plots from the last run's saved results, without running")
    args = parser.parse_args()
    N_JOBS = args.n_jobs
    main(full=args.full, replot=args.replot)
