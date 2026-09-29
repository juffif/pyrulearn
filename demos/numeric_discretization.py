"""
demos/numeric_discretization.py
===============================

RIPPER-family comparison (e): does discretizing numeric attributes
ourselves cost accuracy against a learner's own intrinsic handling, and
how much does the discretization level matter? Weka's JRip and J48 --
both capable of thresholding numeric attributes internally during
learning -- run in seven data preparations per dataset:

- **raw** -- the original numeric columns, unmodified; Weka does its
  own thresholding, same as it would standalone.
- **tree@3 / tree@5 / tree@8** -- our own discretization
  (`pyrulearn.data.io.build_dataspec`'s supervised, decision-tree-based
  binning, `pyrulearn.interfaces.sklearn.tree_thresholds`) at
  `max_intervals` 3, 5, 8, then binarized (`binarize`) before fitting --
  the same pipeline every other comparison demo uses.
- **kbins@3 / kbins@5 / kbins@8** -- `sklearn.preprocessing.
  KBinsDiscretizer` (`strategy="quantile"`, unsupervised -- no target
  used, unlike `tree_thresholds`) at `n_bins` 3, 5, 8, then binarized
  the same way (`build_kbins_dataspec` below turns its `bin_edges_`
  into a `DataSpec` the rest of the pipeline treats identically to a
  tree-discretized one).

(A native learner -- Pypper, then PFossil -- was tried alongside JRip/
J48 here too, to test whether a `pyrulearn.data.attributes`-constraint-
aware search degrades less steeply with the interval count than Weka's
independent-binarized-feature search does. Dropped after confirming
Pypper's `ReplaceReviseOptimization` post-pass made it ~10x *slower*
than Weka on a high-feature-count dataset, not faster, and PFossil
(no such post-pass) was still markedly slower too -- the search itself,
not just the optimization pass, scales worse than Weka's built-in
threshold search here. Back to JRip/J48 only, as originally planned.)

Both discretizers are fit **on the training fold only**, per fold --
the test fold is binarized against that same fold's fitted thresholds,
never re-discretized from its own values (leaking test-set information
into the discretization the same way it would into a supervised
learner). This matters doubly here since it's the entire subject of the
comparison, not just incidental correctness.

`raw` bypasses `pyrulearn.learners`' `RuleLearner`/`DataRepresentation`
abstractions entirely (Weka needs the true continuous values, not a
Boolean matrix) -- `_fit_weka_raw` below writes the training fold's
DataFrame straight to ARFF and imports the printed rules back through
`JRipImporter`/`J48Importer`'s own dataspec-discovery (no `dataspec=`
passed at construction, mirroring how the pre-infra version of
`demos/ripper_comparison.py` handled its `jrip_native` variant -- this
demo is where that comparison actually belongs, per
`examples/REVISION_PLAN.md`).

Datasets: every small/medium **numeric-attribute** catalog entry (no
nominal columns -- `pyrulearn.experiments.catalog`'s
`attributes="numeric"` category; keeping every dataset's features
uniformly numeric means `build_kbins_dataspec` never has to
special-case a nominal column). Missing values are fine -- `raw`
(`write_arff`) and the tree discretizer (`DecisionTreeClassifier`
accepts NaN natively) already handle them, and `build_kbins_dataspec`
fits each column's `KBinsDiscretizer` on its own non-missing training
values only (it's the one piece here that rejects NaN outright);
`binarize`'s own `MissingStrategy` (default `NEVER_COVERS`) scores
missing values the same way regardless of which discretizer produced
the thresholds. `--full` additionally excludes any dataset whose
`n_instances * n_features` is too high
(`FULL_MAX_ROWS_TIMES_FEATURES`) -- confirmed directly against a real
run to predict this demo's fit time far better than row count alone
(a "medium" 6,118-row, 51-attribute dataset took 33 minutes across its
140 fits, ~15x the typical dataset here).

Measures: accuracy, rule count, condition count -- fit time is recorded
but secondary here (every fit is a bounded Weka subprocess call either
way; this demo's question is about representations, not runtime) --
plus, for each discretized prep, how many binary features it produced
(`n_features`) -- the direct answer to "how much does raising the
interval count actually grow the search space".

Run: `python demos/numeric_discretization.py` with no arguments
is the **quick** default (`QUICK_DATASETS_SPEC`, `QUICK_FOLDS`) --
its report/plot go to ``numeric_discretization_quick_*`` and are
*not* checked in (see `.gitignore`), the same quick/full split
`demos/ripper_comparison.py` uses. `--full` runs the full comparison
instead (every small/medium numeric-attribute catalog entry,
`N_FOLDS`-fold) and writes to the canonical
``numeric_discretization_report.md`` (plot in ``numeric_discretization_plots/``), which *are* checked
in -- a sample from a real full run. Either way needs Weka installed at
`WEKA_JAR`/`WEKA_JAVA` below, plus the `experiments` extra.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import warnings

import numpy as np

from pyrulearn.data import BooleanDataRepresentation, DataSpecBuilder
from pyrulearn.data.io import binarize, build_dataspec, write_arff
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.stats import mean_rank, win_counts
from pyrulearn.interfaces.weka import J48Importer, JRipImporter, WekaJ48, WekaJRip, _resolve_weka, _weka_cli_options

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
LEVELS = (3, 5, 8)
KBINS_STRATEGY = "quantile"
FIT_TIMEOUT = 120.0  # seconds, per (dataset, fold, algorithm, prep) -- a Weka subprocess call

N_FOLDS = 10
QUICK_FOLDS = 3
# Small, purely-numeric-attribute datasets, a spread of feature counts
# (4 to 60) so the "how many binary features does each discretization
# produce" measure actually varies -- five of the 41 candidates
# Catalog.default().select(attributes="numeric", size="small") lists.
QUICK_DATASETS_SPEC = "iris,wine,glass,blood-transfusion-service-center,ionosphere"

# Full run: every small/medium numeric-attribute catalog entry -- 41 of
# them (15 small + 26 medium; see Catalog.default().summary()). Missing
# values are fine here (see the module docstring), so this isn't
# restricted to select(missing=False) the way an earlier version was --
# not expressible as a Catalog.parse() string either way (parse() has
# no "missing" axis), so full_datasets() calls select() directly rather
# than going through parse() the way the quick default does.
FULL_SELECT_KWARGS = dict(attributes="numeric", size=["small", "medium"])

# ... except datasets `n_instances * n_features` puts above this --
# confirmed directly against a real run, not a guess: `n * features`
# predicts this demo's per-dataset fit time far better than the
# catalog's own row-count-only `size` (a "medium" 6,118-row, 51-attribute
# dataset took 33 minutes total across its 140 fits, ~15x the typical
# small/medium dataset here, because both a raw fit's cost (rows) and a
# discretized fit's binarized feature count (features, further
# multiplied by max_intervals) scale with their own factor). There's a
# natural gap in the actual data right at this cutoff (57,831 to
# 83,160) -- not an arbitrary round number.
FULL_MAX_ROWS_TIMES_FEATURES = 60_000


def _full_candidates():
    return Catalog.default().select(**FULL_SELECT_KWARGS)


def full_datasets():
    return [e for e in _full_candidates()
           if e.n_instances * e.n_features <= FULL_MAX_ROWS_TIMES_FEATURES]


HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "numeric_discretization_report.md")
PLOTS_DIR = os.path.join(HERE, "numeric_discretization_plots")
PLOT_PATH = os.path.join(PLOTS_DIR, "numeric_discretization_accuracy.png")
QUICK_REPORT_PATH = os.path.join(HERE, "numeric_discretization_quick_report.md")
QUICK_PLOT_PATH = os.path.join(PLOTS_DIR, "numeric_discretization_quick_accuracy.png")
CACHE_DIR = os.path.join(HERE, "_numeric_discretization_cache")

ALGORITHMS = {"JRip": (WekaJRip, JRipImporter, "weka.classifiers.rules.JRip"),
             "J48": (WekaJ48, J48Importer, "weka.classifiers.trees.J48")}
PREPS = ["raw"] + [f"tree@{n}" for n in LEVELS] + [f"kbins@{n}" for n in LEVELS]
DISCRETIZERS = ["tree", "kbins"]
COLORS = dict(zip(ALGORITHMS, ["tab:blue", "tab:orange"]))
LINESTYLES = dict(zip(DISCRETIZERS, ["-o", "--s"]))


def _build_description(datasets, n_folds: int, quick: bool) -> str:
    # Built from the actual datasets/n_folds a run used, not the module's
    # QUICK_*/N_FOLDS constants directly -- those can be overridden by a
    # caller (as the smoke test does), and a static string would then
    # silently print the wrong numbers.
    mode_note = (
        f"**Quick run** ({len(datasets)} datasets, {n_folds}-fold) -- a fast sanity "
        f"check, not a statistically rigorous comparison; default with no arguments. "
        f"For the full comparison ({len(full_datasets())} datasets, {N_FOLDS}-fold): "
        f"`python demos/numeric_discretization.py --full`; a sample from that "
        f"run is committed at `{os.path.basename(REPORT_PATH)}` (plus its plot) in "
        f"this directory.\n\n"
        if quick else
        f"**Full run** ({len(datasets)} small and medium-sized datasets, all attributes "
        f"numeric, {n_folds}-fold). For a quick sanity check instead "
        f"({len(QUICK_DATASETS_SPEC.split(','))} small datasets, {QUICK_FOLDS}-fold): "
        f"`python demos/numeric_discretization.py` with no arguments -- its "
        f"output isn't checked in (see `{os.path.basename(QUICK_REPORT_PATH)}` after "
        f"running it).\n\n"
    )
    return f"""\
{mode_note}Weka's JRip and J48 compared across seven data preparations per dataset: `raw`
(Weka's own intrinsic numeric thresholding), our own decision-tree-based
discretization (`build_dataspec`/`tree_thresholds`) and `sklearn.preprocessing.
KBinsDiscretizer` (`strategy={KBINS_STRATEGY!r}`, unsupervised), each of the
latter two at {', '.join(str(n) for n in LEVELS)} intervals. Both discretizers
are fit on the training fold only, per fold; the test fold is binarized
against that same fold's thresholds.

Measures: test accuracy, rule count, condition count, and (for the discretized
preps) the number of binary features that discretization produced. Fit time
is recorded but secondary here -- every fit is a single bounded Weka
subprocess call regardless of preparation. Each fit is capped at
{FIT_TIMEOUT:.0f}s; a timeout or an exception is recorded as a failure, not
fatal to the run.

Missing values: in the discretized preps a missing value makes both a
threshold feature and its negation false (e.g. neither `plas>=154.5` nor
`plas<154.5`), so a rule never covers it through that attribute. Weka's
own `raw` handling differs, which can hurt the discretized preps on
datasets with many missing values -- here mainly `diabetes` (376 of 768
rows with a missing value; `breast-w` has 16 of 699).
"""


def build_kbins_dataspec(train_df, target: str, n_bins: int, strategy: str = KBINS_STRATEGY,
                         include_negations: bool = True) -> DataSpecBuilder:
    """`build_dataspec`'s counterpart for `sklearn.preprocessing.
    KBinsDiscretizer` -- fit purely on `train_df` (the training fold
    only; see the module docstring), one `KBinsDiscretizer(n_bins=
    n_bins, strategy=strategy)` per numeric feature column, its interior
    bin edges (`bin_edges_[0][1:-1]` -- the outer two edges are just
    that column's own train-fold min/max, not real thresholds) becoming
    that column's `DataSpecBuilder.add_numeric` cut points. A column
    KBinsDiscretizer collapses to a single bin (no interior edges left,
    e.g. too many tied values) is left out, matching `build_dataspec`'s
    own `skip_unusable`.

    Missing values: unlike `tree_thresholds` (sklearn's
    `DecisionTreeClassifier` accepts NaN directly) and `binarize` (its
    own `MissingStrategy`, unaffected by any of this), `KBinsDiscretizer`
    itself rejects NaN outright -- so each column is fit on its own
    non-missing training values only, same rows `binarize` later scores
    as "missing" regardless of what threshold was learned. A column with
    fewer than two observed values (all/almost-all missing in this fold)
    is left out, same as one KBinsDiscretizer collapses to a single bin.

    Every feature column must already be numeric (no nominal handling
    here -- this demo only runs it against `attributes="numeric"`
    catalog entries)."""
    from sklearn.preprocessing import KBinsDiscretizer

    builder = DataSpecBuilder(negation=include_negations)
    feature_cols = [c for c in train_df.columns if c != target]
    for col in feature_cols:
        raw = train_df[col].to_numpy(dtype=float)
        values = raw[~np.isnan(raw)].reshape(-1, 1)
        if len(values) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn warns when it has to shrink n_bins for this column
            disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
            disc.fit(values)
        thresholds = sorted({float(e) for e in disc.bin_edges_[0][1:-1]})
        if thresholds:
            builder.add_numeric(col, thresholds)
    return builder


def _fit_weka_raw(weka_class: str, importer_cls, train_df, target: str, timeout: float):
    """`raw`: write `train_df` straight to ARFF (real numeric columns,
    no binarization) and run `weka_class` on it directly -- bypassing
    `pyrulearn.interfaces.weka.run_weka` (which assumes an
    already-Boolean matrix) and `WekaJRip`/`WekaJ48` (same assumption,
    via `ExternalRuleLearner.prepare`'s default). `importer_cls()` with
    no `dataspec=` discovers one purely from Weka's own printed
    thresholds -- see `JRipImporter`/`J48Importer`'s shared
    dataspec-discovery convention. Returns `(model, dataspec)`; raises
    `subprocess.TimeoutExpired` or `RuntimeError` on failure, same as
    `run_weka`."""
    jar, java = _resolve_weka(WEKA_JAR, WEKA_JAVA)
    with tempfile.TemporaryDirectory(prefix="weka_raw_") as d:
        arff = os.path.join(d, "train.arff")
        write_arff(train_df, target, arff)
        cmd = [java, "-Duser.language=en", "-Duser.country=US", "-cp", jar, weka_class,
              "-t", arff, "-no-cv", *_weka_cli_options(None)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or "=== " not in result.stdout:
        raise RuntimeError(
            f"{weka_class} produced no usable model (exit {result.returncode}).\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    importer = importer_cls()
    model = importer.parse(result.stdout)
    return model, importer.dataspec


def _evaluate(model, dataspec, test_df, target: str):
    test_y = test_df[target].to_numpy()
    rep = BooleanDataRepresentation(dataspec, binarize(dataspec, test_df), test_y)
    stats = model.evaluate(rep)
    return {
        "accuracy": stats.confusion.accuracy if stats.confusion is not None else np.nan,
        "n_rules": stats.n_rules,
        "n_conditions": stats.n_conditions,
    }


def _cache_key(dataset_name: str, n_folds: int, fold: int, algorithm: str, prep: str) -> str:
    import hashlib
    payload = "|".join([dataset_name, str(n_folds), str(fold), algorithm, prep, str(RANDOM_STATE)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _cache_load(key: str):
    import json
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cache_save(key: str, row: dict) -> None:
    import json
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(key), "w", encoding="utf-8") as f:
        json.dump(row, f)


def select_datasets(spec: str = QUICK_DATASETS_SPEC):
    return Catalog.default().parse(spec, random_state=RANDOM_STATE)


def run(datasets, n_folds: int, verbose: bool = True):
    """The custom cross-validation loop this comparison needs --
    `pyrulearn.experiments.runner.run_cv` hands every learner the same
    *one* per-fold representation, but this demo's entire subject is
    comparing *several* representations per fold (`raw` plus six
    discretized variants), so it can't be expressed as a `run_cv` call
    (the same reason `demos/ripper_comparison.py` dropped its analogous
    `jrip_native` variant rather than force it through `run_cv`).
    Still reuses `Catalog` for dataset selection and `pyrulearn.data.io`
    directly for discretization -- only the orchestration loop and
    timeout handling (a plain Weka subprocess timeout suffices here,
    see the module docstring) are demo-local.

    Returns one long-format `pandas.DataFrame`: `dataset`, `fold`,
    `algorithm` (``"JRip"``/``"J48"``), `prep`, `learner`
    (``f"{algorithm}:{prep}"`` -- so `pyrulearn.experiments.stats`/
    `.report`, which key off a `learner` column, work unmodified),
    `fit_time`, `error`, `accuracy`, `n_rules`, `n_conditions`,
    `n_features` (the discretized feature count; NaN for `raw`)."""
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold

    rows = []
    for entry in datasets:
        if verbose:
            print(f"{entry.name} ...", flush=True)
        df, target = entry.load()
        y_all = df[target].to_numpy()
        try:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            folds = list(splitter.split(np.zeros(len(y_all)), y_all))
        except ValueError:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
            folds = list(splitter.split(np.zeros(len(y_all))))

        for fold, (train_idx, test_idx) in enumerate(folds):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # one DataSpec per (level, discretizer) this fold -- shared by both algorithms
            specs = {}
            for n in LEVELS:
                specs[f"tree@{n}"] = build_dataspec(train_df, target=target, max_intervals=n,
                                                    skip_unusable=True).build()
                specs[f"kbins@{n}"] = build_kbins_dataspec(train_df, target=target, n_bins=n).build()
            reps = {prep: (BooleanDataRepresentation(spec, binarize(spec, train_df), train_df[target].to_numpy()),
                          BooleanDataRepresentation(spec, binarize(spec, test_df), test_df[target].to_numpy()))
                   for prep, spec in specs.items()}

            for algorithm, (learner_cls, importer_cls, weka_class) in ALGORITHMS.items():
                for prep in PREPS:
                    if verbose:
                        print(f"  fold {fold + 1}/{n_folds}  {algorithm}:{prep:9s} ...", end="", flush=True)
                    key = _cache_key(entry.name, n_folds, fold, algorithm, prep)
                    cached = _cache_load(key)
                    if cached is not None:
                        rows.append(cached)
                        if verbose:
                            print(" (cached)", flush=True)
                        continue

                    row = {"dataset": entry.name, "fold": fold, "algorithm": algorithm, "prep": prep,
                          "learner": f"{algorithm}:{prep}"}
                    t0 = time.time()
                    try:
                        if prep == "raw":
                            model, dataspec = _fit_weka_raw(weka_class, importer_cls, train_df, target,
                                                            FIT_TIMEOUT)
                            measures = _evaluate(model, dataspec, test_df, target)
                            measures["n_features"] = np.nan
                        else:
                            train_rep, test_rep = reps[prep]
                            model = learner_cls(jar=WEKA_JAR, java=WEKA_JAVA, timeout=FIT_TIMEOUT).fit(train_rep)
                            stats = model.evaluate(test_rep)
                            measures = {
                                "accuracy": stats.confusion.accuracy if stats.confusion is not None else np.nan,
                                "n_rules": stats.n_rules, "n_conditions": stats.n_conditions,
                                "n_features": specs[prep].n_features,
                            }
                        row["fit_time"], row["error"] = time.time() - t0, None
                        row.update(measures)
                    except subprocess.TimeoutExpired:
                        row.update(fit_time=time.time() - t0, error="timeout", accuracy=np.nan,
                                  n_rules=np.nan, n_conditions=np.nan, n_features=np.nan)
                    except Exception as e:  # noqa: BLE001 -- a failed fit is recorded, not fatal
                        row.update(fit_time=time.time() - t0, error=f"{type(e).__name__}: {e}",
                                  accuracy=np.nan, n_rules=np.nan, n_conditions=np.nan, n_features=np.nan)

                    _cache_save(key, row)
                    rows.append(row)
                    if verbose:
                        outcome = row["error"] if row["error"] is not None else f"acc={row['accuracy']:.3f}"
                        print(f" {row['fit_time']:6.2f}s  {outcome}", flush=True)

    return pd.DataFrame(rows)


def write_report(results, datasets, n_folds: int, report_path: str, plot_path: str, quick: bool) -> None:
    results = results.copy()
    results["conds_per_rule"] = results["n_conditions"] / results["n_rules"]

    lines = ["# Numeric attributes: intrinsic handling vs. discretization (JRip / J48)\n\n"]
    lines.append(render_setup_section(_build_description(datasets, n_folds, quick)))
    lines.append(f"![accuracy vs. discretization level]({os.path.basename(PLOTS_DIR)}/{os.path.basename(plot_path)})\n\n")

    lines.append("## Binary features produced per discretization level\n\n")
    feat = results[results["prep"] != "raw"].groupby("prep")["n_features"].mean()
    lines.append("| prep | mean binary features |\n|---|--:|\n")
    for prep in [p for p in PREPS if p != "raw"]:
        lines.append(f"| {prep} | {feat.get(prep, float('nan')):.1f} |\n")
    lines.append("\n")

    lines.append("## Summary by variant (mean across every dataset and fold)\n\n")
    ranks = mean_rank(results, "accuracy")
    wins = win_counts(results, "accuracy")
    summary = results.groupby("learner")[["accuracy", "n_rules", "conds_per_rule", "fit_time"]].mean(
        numeric_only=True)
    lines.append("| variant | accuracy | n_rules | conds/rule | fit_time (s) | wins | mean rank |\n"
                "|---|--:|--:|--:|--:|--:|--:|\n")
    for learner in ranks.index:
        row = summary.loc[learner]
        lines.append(f"| {learner} | {row['accuracy']:.3f} | {row['n_rules']:.2f} | "
                     f"{row['conds_per_rule']:.2f} | {row['fit_time']:.2f} | "
                     f"{wins.get(learner, 0.0):.1f} | {ranks[learner]:.2f} |\n")
    lines.append("\n`wins` -- datasets where a variant's mean accuracy was (tied-for-)best, a tie "
                "split evenly; `mean rank` -- average accuracy rank across datasets, failures tied "
                "for last.\n\n")

    lines.append("## Per-dataset results, per variant (mean across folds)\n\n")
    lines.append(render_results_table(
        results, ["accuracy", "n_rules", "n_conditions", "conds_per_rule", "fit_time"],
        group_by=["dataset", "algorithm", "prep"], include_overall=False))
    lines.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report -> {report_path}")


def write_plots(results, plot_path: str) -> None:
    """One line per (algorithm, discretizer) pair -- four curves --
    accuracy against discretization level; each algorithm's `raw`
    accuracy (no discretization level to place it at) as a dashed
    horizontal reference line in that algorithm's color instead of a
    fifth point on the level axis, so it reads as "the bar each curve
    would need to clear", not as an extra, artificially-ordered level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    per_variant = results.groupby(["algorithm", "prep"], as_index=False)["accuracy"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for algorithm, color in COLORS.items():
        sub = per_variant[per_variant["algorithm"] == algorithm].set_index("prep")
        for discretizer, style in LINESTYLES.items():
            ys = [sub["accuracy"].get(f"{discretizer}@{n}", np.nan) for n in LEVELS]
            ax.plot(LEVELS, ys, style, color=color, label=f"{algorithm} ({discretizer})")
        raw_acc = sub["accuracy"].get("raw", np.nan)
        if raw_acc == raw_acc:  # not NaN
            ax.axhline(raw_acc, color=color, linestyle=":", linewidth=1.5, alpha=0.8,
                      label=f"{algorithm} (raw)")
    ax.set_xticks(list(LEVELS))
    ax.set_xlabel("discretization level (intervals)")
    ax.set_ylabel("mean test accuracy")
    ax.set_title("accuracy vs. discretization level")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path, dpi=110)
    plt.close(fig)
    print(f"Plot   -> {plot_path}")


def main(full: bool = False) -> None:
    quick = not full
    datasets = select_datasets(QUICK_DATASETS_SPEC) if quick else full_datasets()
    n_folds = QUICK_FOLDS if quick else N_FOLDS
    report_path = QUICK_REPORT_PATH if quick else REPORT_PATH
    plot_path = QUICK_PLOT_PATH if quick else PLOT_PATH

    results = run(datasets, n_folds=n_folds)
    write_report(results, datasets, n_folds, report_path, plot_path, quick)
    write_plots(results, plot_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help=f"run the full comparison ({len(full_datasets())} datasets, "
                             f"{N_FOLDS}-fold) instead of the quick default; writes to the "
                             "checked-in report/plot rather than the quick ones")
    args = parser.parse_args()
    main(full=args.full)
