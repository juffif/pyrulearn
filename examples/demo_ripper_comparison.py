"""
examples/demo_ripper_comparison.py
=================================

Four RIPPER runs, three of them on the *same* binarized feature set:

- **jrip**       -- Weka's `weka.classifiers.rules.JRip`, run as a subprocess
                    against a written ARFF, parsed back via
                    `pyrulearn.interfaces.weka.JRipImporter`. Fed the
                    shared `f0..fN` binary columns.
- **jrip_native** -- the *same* Weka JRip, but handed the raw dataset
                    (original numeric/nominal columns, JRip does its own
                    discretization). Not on the shared feature set --
                    included as the "what JRip does left to its own
                    devices" reference point.
- **wittgenstein** -- the `wittgenstein` package's `RIPPER`, via
                    `pyrulearn.interfaces.wittgenstein.RIPPERk`
                    (`pos_class`-restricted, so multi-class is a manual
                    one-vs-rest here -- one fit per class, rules pooled).
- **pypper**     -- pyrulearn's own `pyrulearn.learners.seco.Pypper` (a RIPPER
                    re-implementation): IREP* growth/pruning + `ReplaceReviseOptimization`,
                    per class inside a least-frequent-first ordered
                    decomposition.

Workflow: read each dataset, `build_dataspec` + `binarize` once, and hand
`jrip`/`wittgenstein`/`pypper` the identical `BooleanDataRepresentation`
(train + a held-out test split); `jrip_native` gets the raw train/test
DataFrames instead. Reports fit time, test accuracy, and rule complexity
(rule count, total conditions, conditions per rule).

Includes multi-class datasets this time (iris, wine, glass, ...).

Run: `python examples/demo_ripper_comparison.py` (needs Weka installed at
`WEKA_JAR` below, plus `pandas`, `scikit-learn`, `wittgenstein`).
"""

from __future__ import annotations

import os
import subprocess
import time
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from pyrulearn.models import FlatRuleSet
from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec, write_arff
from pyrulearn.interfaces.weka import JRipImporter
from pyrulearn.interfaces.wittgenstein import RIPPERk
from pyrulearn.learners.seco import Pypper

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"

RANDOM_STATE = 0
TEST_SIZE = 0.30
MAX_ROWS = 1600          # subsample larger datasets -- RIPPER scales poorly
MAX_INTERVALS = 6        # numeric-feature discretization (build_dataspec)
FIT_TIMEOUT = 180        # seconds, per (dataset, algorithm)

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "demo_ripper_comparison_report.md")
PLOT_PATH = os.path.join(HERE, "demo_ripper_comparison.png")
ARFF_DIR = os.path.join(HERE, "_ripper_demo_arff")

BINARY_DATASETS = [
    "vote", "breast-cancer", "colic", "credit-approval", "credit-g",
    "diabetes", "sonar", "ionosphere", "tic-tac-toe", "banknote-authentication",
    "hepatitis", "heart-statlog", "kr-vs-kp",
]
MULTICLASS_DATASETS = [
    "iris", "wine", "glass", "vehicle", "segment", "car",
    "balance-scale", "zoo", "ecoli", "lymph",
]


# --------------------------------------------------------------------- data ----

def _fill_missing(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    drop = [c for c in df.columns
            if c != target_col and (df[c].isna().all() or df[c].nunique(dropna=True) <= 1)]
    df = df.drop(columns=drop)
    for c in df.columns:
        if c == target_col or not df[c].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), "?")
    drop = [c for c in df.columns if c != target_col and df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=drop)


def load(name: str, max_rows=MAX_ROWS):
    """`max_rows` caps the row count (strat?-free uniform subsample, seeded);
    pass `None` to keep every row. Defaults to this module's `MAX_ROWS`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    target_col = d.target.name
    df = _fill_missing(d.frame, target_col)
    df = df.dropna(subset=[target_col])
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=RANDOM_STATE).reset_index(drop=True)
    return df, target_col


class Ctx(NamedTuple):
    name: str
    ds: object
    train_rep: BooleanDataRepresentation
    test_rep: BooleanDataRepresentation
    tr_y: np.ndarray
    te_y: np.ndarray
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    target_col: str
    arff_types: dict


def prepare(df: pd.DataFrame, target_col: str) -> Ctx:
    """One shared binary representation: split, build one DataSpec on the
    training rows, binarize both halves against it. The raw train/test
    frames ride along for `jrip_native`."""
    y = df[target_col].astype(str).to_numpy()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y if min(np.bincount(np.unique(y, return_inverse=True)[1])) >= 2 else None,
    )
    feats = [c for c in df.columns if c != target_col]
    arff_types = {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "nominal") for c in feats}
    ds = build_dataspec(train_df, target=target_col, arff_types=arff_types,
                        max_intervals=MAX_INTERVALS).build()
    tr_y = train_df[target_col].astype(str).to_numpy()
    te_y = test_df[target_col].astype(str).to_numpy()
    train_rep = BooleanDataRepresentation(ds, binarize(ds, train_df), tr_y)
    test_rep = BooleanDataRepresentation(ds, binarize(ds, test_df), te_y)
    return Ctx(name="", ds=ds, train_rep=train_rep, test_rep=test_rep, tr_y=tr_y, te_y=te_y,
               train_df=train_df, test_df=test_df, target_col=target_col, arff_types=arff_types)


# ----------------------------------------------------------------- fitters ----

def fit_pypper(c: Ctx):
    return Pypper(random_state=RANDOM_STATE).fit(c.train_rep)


def fit_wittgenstein(c: Ctx):
    classes = list(np.unique(c.tr_y))
    if len(classes) == 2:
        minority = min(classes, key=lambda k: int((c.tr_y == k).sum()))
        return RIPPERk(pos_class=minority, random_state=RANDOM_STATE).fit(c.train_rep)
    # manual one-vs-rest: one RIPPER fit per class (y relabelled to c vs "rest"),
    # rules pooled into a voting FlatRuleSet
    pooled = []
    for k in classes:
        yb = np.where(c.tr_y == k, k, "rest")
        rep_k = BooleanDataRepresentation(c.ds, c.train_rep.X, yb)
        pooled.extend(RIPPERk(pos_class=k, random_state=RANDOM_STATE).fit(rep_k).rules)
    majority = max(classes, key=lambda k: int((c.tr_y == k).sum()))
    return FlatRuleSet(pooled, default_prediction=majority, combiner="vote")


def _run_jrip(path: str):
    cmd = [WEKA_JAVA, "-cp", WEKA_JAR, "weka.classifiers.rules.JRip", "-t", path, "-no-cv"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT)
    if "===" not in out.stdout:
        raise RuntimeError(f"weka produced no model (rc {out.returncode}): {out.stderr[:400]}")
    importer = JRipImporter()
    return importer.parse(out.stdout), importer.dataspec


def fit_jrip(c: Ctx):
    """Weka JRip on the shared `f0..fN` binary columns."""
    os.makedirs(ARFF_DIR, exist_ok=True)
    names = [f"f{i}" for i in range(c.ds.n_features)]
    df = pd.DataFrame(c.train_rep.X.astype(int), columns=names)
    df["class"] = c.tr_y
    path = os.path.join(ARFF_DIR, f"{c.name}.arff")
    write_arff(df, "class", path)
    rules, jrip_ds = _run_jrip(path)
    return rules, jrip_ds, names


def fit_jrip_native(c: Ctx):
    """Weka JRip on the raw dataset -- JRip discretizes on its own."""
    os.makedirs(ARFF_DIR, exist_ok=True)
    path = os.path.join(ARFF_DIR, f"{c.name}_native.arff")
    write_arff(c.train_df, c.target_col, path, arff_types=c.arff_types)
    rules, jrip_ds = _run_jrip(path)
    return rules, jrip_ds, None  # names=None -> evaluate binarizes the raw test frame


# ---------------------------------------------------------------- evaluate ----

def evaluate(fitted, c: Ctx):
    """`(accuracy, n_rules, total_conditions, conds_per_rule)`."""
    if isinstance(fitted, tuple):  # jrip -- rules bound to their own parsed spec
        rules, jrip_ds, names = fitted
        if names is None:  # jrip_native -- rules over the raw columns
            rep = BooleanDataRepresentation(jrip_ds, binarize(jrip_ds, c.test_df), c.te_y)
        else:  # jrip -- rules over the shared f0..fN binary columns
            bool_df = pd.DataFrame(binarize(c.ds, c.test_df).astype(int), columns=names)
            rep = BooleanDataRepresentation(jrip_ds, binarize(jrip_ds, bool_df), c.te_y)
    else:
        rules, rep = fitted, c.test_rep
    preds = np.asarray(rules.predict(rep))
    acc = float(np.mean(preds == c.te_y))
    n = len(rules.rules)
    total = int(sum(len(r.conditions) for r in rules.rules))
    return acc, n, total, (total / n if n else 0.0)


ALGOS = {
    "jrip": fit_jrip,
    "jrip_native": fit_jrip_native,
    "wittgenstein": fit_wittgenstein,
    "pypper": fit_pypper,
}


def run_dataset(name: str, multiclass: bool):
    df, target_col = load(name)
    c = prepare(df, target_col)._replace(name=name)
    n_classes = len(np.unique(c.tr_y))
    print(f"\n{name}  (n={len(df)}, features->{c.ds.n_features} binary, {n_classes} classes)")
    rows = {}
    for algo, fitter in ALGOS.items():
        t0 = time.time()
        try:
            fitted = fitter(c)
            dt = time.time() - t0
            acc, n_rules, total, per_rule = evaluate(fitted, c)
            rows[algo] = dict(acc=acc, n_rules=n_rules, total=total, per_rule=per_rule, t=dt)
            print(f"  {algo:13s} acc={acc:.3f}  rules={n_rules:3d}  conds={total:4d}  "
                  f"({per_rule:.1f}/rule)  {dt:6.1f}s")
        except Exception as e:  # noqa: BLE001 -- one failure shouldn't sink the run
            rows[algo] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"  {algo:13s} FAILED: {rows[algo]['error']}")
    return dict(name=name, n=len(df), n_feat=c.ds.n_features, n_classes=n_classes,
                multiclass=multiclass, rows=rows)


# ------------------------------------------------------------------ report ----

def _fmt(v, spec):
    return "-" if v is None else format(v, spec)


def write_report(results):
    lines = ["# RIPPER comparison: jrip / jrip_native / wittgenstein / pypper", ""]
    lines.append(
        "`jrip`, `wittgenstein` and `pypper` share one binarized feature set "
        "(`build_dataspec` + `binarize`, one DataSpec per dataset); `jrip_native` "
        "is the same Weka JRip run on the raw columns instead (its own "
        f"discretization). {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} "
        f"stratified split, seed {RANDOM_STATE}, datasets capped at {MAX_ROWS} rows. "
        "wittgenstein on multi-class = manual one-vs-rest (one fit per class, rules pooled). "
        "`bin.feat` is the shared binary feature count -- not what `jrip_native` used."
    )
    lines.append("")
    lines.append("| dataset | n | bin.feat | classes | algo | acc | rules | conds | conds/rule | fit s |")
    lines.append("|---|--:|--:|--:|---|--:|--:|--:|--:|--:|")
    for r in results:
        for algo in ALGOS:
            row = r["rows"].get(algo, {})
            if "error" in row:
                lines.append(f"| {r['name']} | {r['n']} | {r['n_feat']} | {r['n_classes']} | "
                             f"{algo} | — | — | — | — | *{row['error']}* |")
                continue
            lines.append(
                f"| {r['name']} | {r['n']} | {r['n_feat']} | {r['n_classes']} | {algo} | "
                f"{_fmt(row.get('acc'), '.3f')} | {row.get('n_rules', '-')} | "
                f"{row.get('total', '-')} | {_fmt(row.get('per_rule'), '.1f')} | "
                f"{_fmt(row.get('t'), '.1f')} |"
            )

    # per-algorithm means (over datasets where the algo succeeded)
    lines += ["", "## Means (successful fits only)", "",
              "| algo | datasets | mean acc | mean rules | mean conds | mean conds/rule | mean fit s |",
              "|---|--:|--:|--:|--:|--:|--:|"]
    for algo in ALGOS:
        ok = [r["rows"][algo] for r in results if "error" not in r["rows"].get(algo, {"error": 1})]
        if not ok:
            lines.append(f"| {algo} | 0 | — | — | — | — | — |")
            continue
        m = lambda k: np.mean([x[k] for x in ok])  # noqa: E731
        lines.append(f"| {algo} | {len(ok)} | {m('acc'):.3f} | {m('n_rules'):.1f} | "
                     f"{m('total'):.1f} | {m('per_rule'):.2f} | {m('t'):.2f} |")

    # binary vs multiclass accuracy split
    lines += ["", "## Mean accuracy by target type", "",
              "| algo | binary | multi-class |", "|---|--:|--:|"]
    for algo in ALGOS:
        def mean_acc(mc):
            v = [r["rows"][algo]["acc"] for r in results
                 if r["multiclass"] == mc and "acc" in r["rows"].get(algo, {})]
            return f"{np.mean(v):.3f}" if v else "—"
        lines.append(f"| {algo} | {mean_acc(False)} | {mean_acc(True)} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport -> {REPORT_PATH}")


def write_plot(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"jrip": "tab:blue", "jrip_native": "tab:cyan",
              "wittgenstein": "tab:orange", "pypper": "tab:green"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for algo in ALGOS:
        pts = [(r["rows"][algo]["total"], r["rows"][algo]["acc"])
               for r in results if "acc" in r["rows"].get(algo, {})]
        if pts:
            xs, ys = zip(*pts)
            ax1.scatter(xs, ys, label=algo, color=colors[algo], alpha=0.7, s=45)
    ax1.set_xlabel("total conditions (rule set complexity)")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("accuracy vs. rule-set complexity")
    ax1.set_xscale("symlog")
    ax1.legend()
    ax1.grid(alpha=0.3)

    names = [r["name"] for r in results]
    x = np.arange(len(names))
    w = 0.8 / len(ALGOS)
    for i, algo in enumerate(ALGOS):
        ts = [r["rows"][algo].get("t", np.nan) for r in results]
        ax2.bar(x + (i - (len(ALGOS) - 1) / 2) * w, ts, w, label=algo, color=colors[algo])
    ax2.set_yscale("log")
    ax2.set_ylabel("fit time (s, log)")
    ax2.set_title("fit time per dataset")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=110)
    print(f"Plot   -> {PLOT_PATH}")


def main():
    results = []
    for name in BINARY_DATASETS:
        try:
            results.append(run_dataset(name, multiclass=False))
        except Exception as e:  # noqa: BLE001
            print(f"\n{name}  SKIPPED ({type(e).__name__}: {str(e)[:100]})")
    for name in MULTICLASS_DATASETS:
        try:
            results.append(run_dataset(name, multiclass=True))
        except Exception as e:  # noqa: BLE001
            print(f"\n{name}  SKIPPED ({type(e).__name__}: {str(e)[:100]})")
    write_report(results)
    write_plot(results)


if __name__ == "__main__":
    main()
