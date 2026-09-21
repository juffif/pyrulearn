"""
JRip vs. pyrulearn's own SeCo-based learners -- comparison across the
same binary UCI/OpenML benchmarks `demo_workflow_comparison.py` uses
(`STANDARD_DATASETS`, reused directly from that module, along with its
dataset loading/timeout/Weka-subprocess machinery -- see that module's
own docstring for what each of those pieces does and why). `sonar` and
`ionosphere` are excluded by default (`aqr`/`pylord` are very slow on
their few-hundred-feature discretizations); `--include-large` adds them.

A separate demo from `demo_workflow_comparison.py` on purpose: the
pyrulearn learners compared here (`CN2`, `PFoil`, `PFossil`, `AQR`, and
`PyLORD`) build only over a `BooleanDataRepresentation` -- unlike the
tree/RIPPER/IREP models in the other demo, they have no "fit on raw/
native data" path. So there's only ever one data-preparation workflow
here (`build_dataspec`/`binarize` first, i.e. the other demo's
"Binarized" mode), not a Binarized-vs-Original comparison -- these models
are compared against each other, not against themselves under two
preparations. `jrip` and `lord` are the external baselines: both run on
the same pre-binarized 0/1 data (placeholder feature names), one via
`weka.jar`, one via LORD's jar. `pylord` is pyrulearn's own simplified
take on LORD, run against the same reference (`lord`) directly.

Models:
- **`jrip`** -- Weka's JRip (RIPPER), via subprocess (`weka.jar`), same
  invocation `demo_workflow_comparison.py`'s Weka trio uses, on the same
  pre-binarized data with placeholder feature names (Weka's ARFF parser
  chokes on `ds1`'s own condition-style names, e.g. `"age>=30"` --
  see that module's docstring, wrinkle 3).
- **`lord`** -- LORD (Huynh, Fürnkranz & Beck, 2023), the reference Java
  implementation, via subprocess (needs `$LORD_CLASSPATH` = `<repo>/bin`
  + `<repo>/libs/weka_3.8_stable.jar`; n/a otherwise, same as `jrip`
  needs `weka.jar`). Runs `run.LordRun -mt mestimate -ma 0.1` on the
  pre-binarized 0/1 CSV; `pyrulearn.interfaces.lord.LORDImporter`
  reads its printed rule set back. Natively multi-class, no direction.
- **`pylord`** -- `pyrulearn.learners.pylord.PyLORD`: the same algorithm
  reimplemented (simplified) on `pyrulearn.learners.seco`'s pieces -- seed every
  example, greedy m-estimate grow (`beam_width=1`), prune with RIPPER's
  routine scored on the training set (LORD grows/prunes on the same
  data), coverage filter, best-rule-wins. With no N-lists the every-row
  search is ~O(n**2), so it's much slower than `lord` (and n/a on the
  large datasets -- excluded by default, see `LARGE_DATASETS`); where it
  runs it tracks `lord`'s rule sets closely. In-process through
  `TimeoutRunner`.
- **`pfoil`** -- `pyrulearn.learners.seco.PFoil`: `GainAscentHillClimbing` +
  `FoilGain` + Quinlan's (1990) MDL-based encoding-length restriction as
  `stopping=` (`mdl_stopping=True`, the default).
- **`cn2beam1`** -- `pyrulearn.learners.seco.CN2` with `beam_width=1`: `BeamSearch`
  at a beam of one is close to greedy hill-climbing over an ordinary
  heuristic (`Laplace` here, unlike `PFoil`'s `FoilGain`), CN2's own
  significance test still active as `stopping=`.
- **`cn2beam5`** -- `CN2` with its own default `beam_width=5` (the
  original paper's "star size").
- **`pfossil`** -- `pyrulearn.learners.seco.PFossil`: `HillClimbing` over
  `Correlation()` (FOSSIL's own heuristic; hill climbing, as in the
  original), with FOSSIL's published 0.3 correlation cutoff wired as
  `filtering=`. (Before 2026 this was `BeamSearch(beam_width=5)` +
  `Correlation` + a 0.3 `stopping=` threshold, which searched orders of
  magnitude harder and over-generalized badly -- see `PFossil`'s own
  docstring.)
- **`aqr`** -- `pyrulearn.learners.seco.AQR` (Clark & Niblett, 1989): the AQ
  baseline CN2 was designed to improve on. `SeedExample` + a `LEF` +
  **consistency required** (no rule may cover a negative), which on any
  dataset with class noise forces long, overfit rules -- included here
  precisely to show that failure mode against `cn2`'s significance test.
  Expect it to be the slowest model and to time out (n/a) on the larger
  noisy datasets -- rule count and rule length both blow up when every
  rule has to reach zero training errors. Uses `BeamSearch` over the
  seed-restricted space, not AQ's literal star search (see `AQR`'s
  docstring).

Like `ripper`/`irep` in the other demo, `pfoil`/`cn2beam1`/`cn2beam5`/
`pfossil`/`aqr` are all *directional*: each only ever learns rules *for*
one class, using the other as the default/passive prediction. Both
directions are fit and reported separately -- `{model}_A` treats the
(alphabetically) first class as positive, `{model}_B` the second.
`jrip`, `lord` and `pylord` need no such split -- all handle multi-class
natively.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import demo_workflow_comparison as base
from pyrulearn.data.io import binarize, build_dataspec, write_arff
from pyrulearn.interfaces.lord import LORDImporter, run_lord
from pyrulearn.interfaces.weka import JRipImporter
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil

N_FOLDS = base.N_FOLDS
RANDOM_STATE = base.RANDOM_STATE
MAX_INTERVALS = base.MAX_INTERVALS
FIT_TIMEOUT_SECONDS = base.FIT_TIMEOUT_SECONDS
STANDARD_DATASETS = base.STANDARD_DATASETS
# Datasets the slow native learners can't finish in FIT_TIMEOUT_SECONDS:
# `sonar`/`ionosphere` blow up in *feature* count (60/33 numeric attrs ->
# a few hundred discretized features -> `aqr` times out); `kr-vs-kp`/
# `mushroom` blow up in *row* count (2.5k/6.5k rows -> `pylord`'s every-row
# search is ~O(n**2) without N-lists). Excluded by default; `--include-large`
# (or `main(include_large=True)`) adds them back.
LARGE_DATASETS = ["sonar", "ionosphere", "kr-vs-kp", "mushroom"]
SMALL_DATASETS = [d for d in STANDARD_DATASETS if d not in LARGE_DATASETS]

REPORT_PATH = os.path.join(os.path.dirname(__file__), "demo_seco_learners_comparison_report.md")
ARFF_DIR = os.path.join(os.path.dirname(__file__), "_seco_learners_demo_arff")
os.makedirs(ARFF_DIR, exist_ok=True)

DIRECTIONAL_MODELS = ["pfoil", "cn2beam1", "cn2beam5", "pfossil", "aqr"]
SINGLE_MODELS = ["jrip", "lord", "pylord"]
MODEL_VARIANTS = ["jrip", "lord", "pylord", "pfoil_A", "pfoil_B", "cn2beam1_A", "cn2beam1_B",
                  "cn2beam5_A", "cn2beam5_B", "pfossil_A", "pfossil_B", "aqr_A", "aqr_B"]


# ---- module-level fit functions for TimeoutRunner (must be picklable by
# reference, so no closures/lambdas -- see base.TimeoutRunner's docstring) ----

def _fit_pfoil(rep, pos_class):
    return PFoil(target_class=pos_class).fit(rep)


def _fit_cn2_beam1(rep, pos_class):
    return CN2(target_class=pos_class, beam_width=1).fit(rep)


def _fit_cn2_beam5(rep, pos_class):
    return CN2(target_class=pos_class, beam_width=5).fit(rep)


def _fit_pfossil(rep, pos_class):
    return PFossil(target_class=pos_class).fit(rep)


def _fit_aqr(rep, pos_class):
    return AQR(target_class=pos_class, maxstar=5).fit(rep)


def _fit_pylord(rep):
    return PyLORD(m=0.1, random_state=RANDOM_STATE).fit(rep)  # greedy grow, training-set prune


DIRECTIONAL_FIT_FNS = {
    "pfoil": _fit_pfoil,
    "cn2beam1": _fit_cn2_beam1,
    "cn2beam5": _fit_cn2_beam5,
    "pfossil": _fit_pfossil,
    "aqr": _fit_aqr,
}


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, arff_prefix: str,
             runner: "base.TimeoutRunner"):
    """Returns `(metrics, failures)`. `metrics` maps a model name (a
    `MODEL_VARIANTS` entry) to a dict with `acc`/`n_rules`/`avg_cond`/
    `fit_time`, for every model that fit within `FIT_TIMEOUT_SECONDS`
    this fold. `failures` maps the same kind of key to an error string
    (`"timeout"`, or the exception stringified) for every model that
    didn't -- see `base.TimeoutRunner`/`base._run_weka_safe`.
    """
    feature_cols = [c for c in train_df.columns if c != target_col]
    nominal_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    arff_types = {c: ("nominal" if c in nominal_cols else "numeric") for c in feature_cols}

    classes = sorted(pd.unique(train_df[target_col]).tolist())
    if len(classes) != 2:
        raise ValueError(f"expected a binary target, got {len(classes)} classes: {classes}")
    class_a, class_b = classes
    pos_directions = [(class_a, "A"), (class_b, "B")]

    train_y = train_df[target_col].to_numpy()
    test_y = test_df[target_col].to_numpy()

    metrics: dict = {}
    failures: dict = {}

    def record(model, acc, n_rules, avg_cond, fit_time):
        metrics[model] = {"acc": acc, "n_rules": n_rules, "avg_cond": avg_cond, "fit_time": fit_time}

    def fail(model, error):
        failures[model] = error

    ds1 = build_dataspec(train_df, target=target_col, arff_types=arff_types, max_intervals=MAX_INTERVALS).build()
    train_rep1 = BooleanDataRepresentation(ds1, binarize(ds1, train_df), train_y)
    test_rep1 = BooleanDataRepresentation(ds1, binarize(ds1, test_df), test_y)

    for pos, tag in pos_directions:
        for name, fit_fn in DIRECTIONAL_FIT_FNS.items():
            model = f"{name}_{tag}"
            t0 = time.time()
            rules, err = runner.run(fit_fn, train_rep1, pos)
            fit_time = time.time() - t0
            if err is None:
                record(model, *base._eval(rules, test_rep1, test_y), fit_time)
            else:
                fail(model, err)

    # jrip: pre-binarized 0/1 data, placeholder feature names (Weka's ARFF
    # parser can't handle ds1's own condition-style names -- see this
    # module's docstring)
    plain_names = [f"f{i}" for i in range(ds1.n_features)]
    train_bool_df = pd.DataFrame(train_rep1.X.astype(int), columns=plain_names)
    train_bool_df[target_col] = train_y
    train_arff = os.path.join(ARFF_DIR, f"{arff_prefix}_train.arff")
    write_arff(train_bool_df, target_col, train_arff)

    t0 = time.time()
    stdout, err = base._run_weka_safe("jrip", train_arff)
    fit_time = time.time() - t0
    if err is not None:
        fail("jrip", err)
    else:
        importer = JRipImporter()
        rules = importer.parse(stdout)
        ds = importer.dataspec
        test_bool_df = pd.DataFrame(binarize(ds1, test_df).astype(int), columns=plain_names)
        rep = BooleanDataRepresentation(ds, binarize(ds, test_bool_df), test_y)
        record("jrip", *base._eval(rules, rep, test_y), fit_time)

    # lord: same pre-binarized 0/1 CSV (placeholder names, class column last),
    # driven via the LORD jar (needs the LORD_JAR env var; n/a otherwise --
    # same as jrip needs weka.jar). LORD is natively multi-class, no direction.
    header = plain_names + [target_col]
    train_rows = [list(r) + [c] for r, c in zip(train_rep1.X.astype(int), train_y)]
    t0 = time.time()
    try:
        text = run_lord(train_rows, header, metric="mestimate", metric_arg=0.1,
                        timeout=FIT_TIMEOUT_SECONDS)
        fit_time = time.time() - t0
        importer = LORDImporter()
        rules = importer.parse(text)
        ds = importer.dataspec
        rules.default_prediction = _majority_default_target(train_y)
        test_bool_df = pd.DataFrame(binarize(ds1, test_df).astype(int), columns=plain_names)
        rep = BooleanDataRepresentation(ds, binarize(ds, test_bool_df), test_y)
        record("lord", *base._eval(rules, rep, test_y), fit_time)
    except Exception as e:  # noqa: BLE001 -- report any failure back, don't crash the fold
        fail("lord", f"{type(e).__name__}: {e}")

    # pylord: pyrulearn's own simplified reimplementation (pyrulearn.learners.pylord.
    # PyLORD) -- seed every example, m-estimate, IREP-style grow/prune,
    # best-rule-wins. No N-lists, so slow on large data; through TimeoutRunner
    # like the SeCo learners.
    t0 = time.time()
    rules, err = runner.run(_fit_pylord, train_rep1)
    if err is None:
        record("pylord", *base._eval(rules, test_rep1, test_y), time.time() - t0)
    else:
        fail("pylord", err)

    return metrics, failures


def _majority_default_target(train_y):
    """The training-majority class -- LORD doesn't print its own default
    class, so the caller supplies it as the rule set's `default_prediction`."""
    values, counts = np.unique(train_y, return_counts=True)
    return values[int(np.argmax(counts))]


def run_dataset(name: str, df: pd.DataFrame, target_col: str, runner: "base.TimeoutRunner", max_folds=None):
    """Same `max_folds` semantics as `demo_workflow_comparison.run_dataset`."""
    y = df[target_col].to_numpy()
    classes = sorted(pd.unique(y).tolist())
    print(f"\n{'=' * 78}\n{name}  (n={len(df)}, attributes={df.shape[1] - 1}, "
          f"A={classes[0]!r}, B={classes[1]!r})\n{'=' * 78}")

    md = [f"## {name}\n\n", f"n={len(df)}, attributes={df.shape[1] - 1}, "
          f"A={classes[0]!r}, B={classes[1]!r}\n\n"]

    try:
        splitter = base.StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df, y))
    except ValueError as e:
        print(f"  StratifiedKFold unavailable ({e}); falling back to plain KFold")
        splitter = base.KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df))
    if max_folds is not None:
        folds = folds[:max_folds]

    all_metrics = {m: {"acc": [], "n_rules": [], "avg_cond": [], "fit_time": []} for m in MODEL_VARIANTS}
    n_failures = {m: 0 for m in MODEL_VARIANTS}
    t0 = time.time()
    for i, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        fold_metrics, failures = run_fold(train_df, test_df, target_col, arff_prefix=f"{name}_{i}", runner=runner)
        for model, v in fold_metrics.items():
            for field in ("acc", "n_rules", "avg_cond", "fit_time"):
                all_metrics[model][field].append(v[field])
        for model in failures:
            n_failures[model] += 1
        line = "  ".join(
            f"{m}={base._pct(fold_metrics[m]['acc'])}" if m in fold_metrics else f"{m}=FAIL"
            for m in MODEL_VARIANTS
        )
        note = f"  [failed/timed out: {', '.join(f'{m} ({e})' for m, e in failures.items())}]" if failures else ""
        print(f"  fold {i + 1} ({time.time() - t0:.1f}s so far): {line}{note}")
    dt = time.time() - t0

    total_failures = sum(n_failures.values())
    print(f"\n  Summary over {len(folds)} folds ({dt:.1f}s"
          f"{f', {total_failures} model-fold timeout(s)/failure(s)' if total_failures else ''}):")
    if total_failures:
        md.append(f"*{total_failures} model-fold combination(s) timed out (> {FIT_TIMEOUT_SECONDS}s) or raised "
                   f"and were skipped for that fold -- see per-model failure counts below.*\n\n")

    md.append("**Accuracy**\n\n| model | accuracy | failed folds |\n|---|---|---|\n")
    summary = {"name": name, "n": len(df), "time": dt}
    for m in MODEL_VARIANTS:
        vals = all_metrics[m]
        fails = n_failures[m]
        if vals["acc"]:
            mean, std = float(np.mean(vals["acc"])), float(np.std(vals["acc"]))
            acc_str = f"{base._pct(mean)} +/- {base._pct(std)}"
            time_mean = float(np.mean(vals["fit_time"]))
            nrules_mean = float(np.mean(vals["n_rules"]))
            avgcond_mean = float(np.mean(vals["avg_cond"]))
        else:
            mean = time_mean = nrules_mean = avgcond_mean = float("nan")
            acc_str = "n/a"
        print(f"    {m:<12} acc={acc_str}" + (f"  ({fails}/{len(folds)} failed)" if fails else ""))
        md.append(f"| {m} | {acc_str} | {fails}/{len(folds)} |\n")
        summary[f"acc_{m}"] = mean
        summary[f"time_{m}"] = time_mean
        summary[f"nrules_{m}"] = nrules_mean
        summary[f"avgcond_{m}"] = avgcond_mean

    md.append("\n**Fit time (seconds/fold)**\n\n| model | time |\n|---|---|\n")
    for m in MODEL_VARIANTS:
        md.append(f"| {m} | {summary[f'time_{m}']:.3f} |\n")

    md.append("\n**Rule complexity**\n\n| model | n_rules | avg_conditions |\n|---|---|---|\n")
    for m in MODEL_VARIANTS:
        md.append(f"| {m} | {summary[f'nrules_{m}']:.1f} | {summary[f'avgcond_{m}']:.2f} |\n")

    md.append(f"\n({dt:.1f}s total)\n\n---\n\n")

    return summary, md


def _build_overview_table(results: list) -> list:
    """Cross-dataset overview: average performance and average rank per
    model (one column per `base.CRITERIA` field, failures tied last) --
    the single-workflow analogue of `demo_workflow_comparison.
    _build_overview_tables` (no Binarized-vs-Original comparison here,
    since there's only ever one workflow)."""
    lines = ["## Overview evaluation (across all datasets)\n\n"]

    lines.append("**Average performance across datasets**\n\n")
    lines.append("| model | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for m in MODEL_VARIANTS:
        accs = np.array([r[f"acc_{m}"] for r in results], dtype=float)
        times = np.array([r[f"time_{m}"] for r in results], dtype=float)
        nrules = np.array([r[f"nrules_{m}"] for r in results], dtype=float)
        avgconds = np.array([r[f"avgcond_{m}"] for r in results], dtype=float)
        n_fully_failed = int(np.isnan(accs).sum())
        lines.append(
            f"| {m} | {base._pct(np.nanmean(accs))} | {np.nanmean(times):.3f} | "
            f"{np.nanmean(nrules):.1f} | {np.nanmean(avgconds):.2f} | {n_fully_failed} |\n"
        )

    lines.append("\n**Average rank per criterion** (1 = best of "
                  f"{len(MODEL_VARIANTS)}; failed entries tie for last)\n\n")
    lines.append("| model | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |\n")
    lines.append("|---|---|---|---|---|\n")
    avg_ranks = {m: {} for m in MODEL_VARIANTS}
    for field, higher_is_better in base.CRITERIA:
        per_dataset_ranks = {m: [] for m in MODEL_VARIANTS}
        for r in results:
            ranks = base._rank_dataset_by(r, MODEL_VARIANTS, field, higher_is_better)
            for m in MODEL_VARIANTS:
                per_dataset_ranks[m].append(ranks[m])
        for m in MODEL_VARIANTS:
            avg_ranks[m][field] = float(np.mean(per_dataset_ranks[m]))
    for m in MODEL_VARIANTS:
        ar = avg_ranks[m]
        lines.append(
            f"| {m} | {ar['acc']:.2f} | {ar['time']:.2f} | {ar['nrules']:.2f} | {ar['avgcond']:.2f} |\n"
        )

    return lines


def main(datasets=None, max_folds=None, include_large=False):
    if datasets is None:
        datasets = STANDARD_DATASETS if include_large else SMALL_DATASETS
    results = []
    fold_note = f" Only the first {max_folds} of {N_FOLDS} fold(s) actually run (preview mode)." \
        if max_folds is not None else ""
    if not include_large and datasets == SMALL_DATASETS:
        fold_note += f" {', '.join(LARGE_DATASETS)} excluded (pass --include-large)."
    report = [
        "# JRip vs. pyrulearn's SeCo-based learners -- comparison across binary datasets\n\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}. N_FOLDS={N_FOLDS}, "
        f"MAX_INTERVALS={MAX_INTERVALS}, FIT_TIMEOUT_SECONDS={FIT_TIMEOUT_SECONDS}.{fold_note} "
        f"`{{model}}_A`/`{{model}}_B` treat each dataset's (alphabetically) first/second class "
        f"as positive (`pfoil`/`cn2beam1`/`cn2beam5`/`pfossil`/`aqr` only -- `jrip`/`lord`/`pylord` need no direction). "
        f"`jrip`'s fit-time includes JVM subprocess startup overhead, not just the algorithm "
        f"itself. See this module's own docstring for what each model is.\n\n---\n\n",
    ]

    runner = base.TimeoutRunner()
    try:
        for name in datasets:
            df, target_col = base.load_openml(name)
            summary, md = run_dataset(name, df, target_col, runner=runner, max_folds=max_folds)
            results.append(summary)
            report.extend(md)
    finally:
        runner.close()

    print(f"\n{'=' * 78}\nOverall summary (mean accuracy per model)\n{'=' * 78}")
    header = f"{'dataset':<16}" + "".join(f"{m:>14}" for m in MODEL_VARIANTS)
    print(header)
    report.append("## Overall summary\n\n")
    report.append("### Accuracy\n\n")
    report.append("| dataset | " + " | ".join(MODEL_VARIANTS) + " |\n")
    report.append("|---" * (len(MODEL_VARIANTS) + 1) + "|\n")
    for r in results:
        print(f"{r['name']:<16}" + "".join(f"{base._pct(r['acc_' + m]):>14}" for m in MODEL_VARIANTS))
        report.append(f"| {r['name']} | " + " | ".join(base._pct(r['acc_' + m]) for m in MODEL_VARIANTS) + " |\n")

    report.append("\n### Fit time (seconds/fold)\n\n")
    report.append("| dataset | " + " | ".join(MODEL_VARIANTS) + " |\n")
    report.append("|---" * (len(MODEL_VARIANTS) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['time_' + m]:.3f}" for m in MODEL_VARIANTS) + " |\n")

    report.append("\n### Rule count\n\n")
    report.append("| dataset | " + " | ".join(MODEL_VARIANTS) + " |\n")
    report.append("|---" * (len(MODEL_VARIANTS) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['nrules_' + m]:.1f}" for m in MODEL_VARIANTS) + " |\n")

    report.append("\n### Average conditions per rule\n\n")
    report.append("| dataset | " + " | ".join(MODEL_VARIANTS) + " |\n")
    report.append("|---" * (len(MODEL_VARIANTS) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['avgcond_' + m]:.2f}" for m in MODEL_VARIANTS) + " |\n")

    overview_lines = _build_overview_table(results)
    report.extend(overview_lines)
    print("".join(overview_lines))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JRip vs. pyrulearn's SeCo-based learners comparison demo.")
    parser.add_argument(
        "--preview", action="store_true",
        help="Run only fold 1 of every dataset (fold sizes match a full "
             f"{N_FOLDS}-fold split -- only the loop is shortened) as a quick "
             "timing/sanity check before committing to the full run. "
             "Shorthand for --max-folds 1.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None, metavar="N",
        help="Run only the first N of the dataset's N_FOLDS folds. Overrides --preview.",
    )
    parser.add_argument(
        "--include-large", action="store_true",
        help=f"Also run the large numeric datasets ({', '.join(LARGE_DATASETS)}), "
             "excluded by default because aqr/pylord are very slow on them.",
    )
    args = parser.parse_args()
    main(
        max_folds=args.max_folds if args.max_folds is not None else (1 if args.preview else None),
        include_large=args.include_large,
    )
