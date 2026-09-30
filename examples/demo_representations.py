"""
examples/demo_representations.py
================================

Four encodings of the same Boolean data, all behind one
`DataRepresentation` interface:

- `BooleanDataRepresentation` -- a `numpy.packbits` bit matrix (the
  baseline).
- `NListRepresentation` -- LORD's PPC-tree / N-list vertical index:
  per-feature row lists, bucketed by shared row prefix.
- `PrePostNListRepresentation` -- `NListRepresentation` plus pre/post
  visit codes, a second way for the search to find the tree nodes
  compatible with the rule so far.
- `SparseDataRepresentation` -- `scipy` CSR/CSC: per-feature row lists
  without the prefix tree.

A learner asks its data only two things -- `coverage(rule)` and
`features_of(row)` -- so the four are interchangeable and must give the
same rules; only the time differs. This demo measures that time and checks
the equality at every point:

1. **Training-set size** -- the `adult` dataset (48,842 rows), training
   sets of doubling size from one shuffled pool, one fixed test set; with
   and without negation features.
2. **Density** -- synthetic data with a fixed learnable concept and a
   fixed number of features, whose noise features get sparser and sparser.

Learners: CN2, PFossil, Pypper and PyLORD, each with its defaults. Every
fit runs in a worker process under a `FIT_TIMEOUT` cap; after a time-out
the remaining, larger points of that curve are skipped.

Run: `python examples/demo_representations.py` is the **quick** default
(sizes up to 1,000, a short density sweep); its report/plots go to
``demo_representations_quick_*`` and are not checked in. `--full` runs the
full curves and writes ``demo_representations_report.md`` (plots in
``demo_representations_plots/``). Every measurement is cached, so an
interrupted run continues where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from pyrulearn import DataSpec, Rule
from pyrulearn.data import (
    BooleanDataRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    SparseDataRepresentation,
)
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.experiments.catalog import Catalog
from pyrulearn.experiments.runner import TimeoutRunner
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.learners.seco import CN2, PFossil, Pypper

RANDOM_STATE = 0
MAX_INTERVALS = 8
FIT_TIMEOUT = 300.0
REPRESENTATIONS = ("Boolean", "NList", "PrePostNList", "Sparse")
_VERTICAL = {"NList": NListRepresentation, "PrePostNList": PrePostNListRepresentation,
             "Sparse": SparseDataRepresentation}
LEARNERS = ("CN2", "PFossil", "Pypper", "PyLORD")

SIZE_DATASET = "adult"
SIZE_TEST_FRACTION = 0.2
SIZES = [500, 1000, 2000, 4000, 8000, 16000, 32000]
QUICK_SIZES = [250, 500, 1000]

DENSITY_N = 1000
DENSITY_K = 500
DENSITY_N_SIGNAL = 4
DENSITY_SIGNAL_P = 0.4
DENSITIES = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
QUICK_DENSITIES = [0.5, 0.05, 0.005]
COVERAGE_TRIALS = 800

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "demo_representations"
PLOTS_DIR = os.path.join(HERE, f"{NAME}_plots")
CACHE_DIR = os.path.join(HERE, "_representations_cache")

STYLE = {"Boolean": ("o", "tab:blue"), "NList": ("s", "tab:orange"),
         "PrePostNList": ("^", "tab:green"), "Sparse": ("d", "tab:red")}


def make_learner(name: str):
    return {"CN2": CN2, "PFossil": PFossil, "Pypper": Pypper, "PyLORD": PyLORD}[name](
        random_state=RANDOM_STATE)


def timed_fit(learner_name: str, rep):
    """Runs in the worker process: fit, and time only the fit itself (not
    shipping the data to the worker). Falls back to CPU time when wall time
    ran far ahead of it -- the computer slept during the fit."""
    w0, c0 = time.perf_counter(), time.process_time()
    model = make_learner(learner_name).fit(rep)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return model, (cpu if wall - cpu > 30 else wall)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _build(kind: str, spec, X, y):
    """The representation of kind `kind`, and the seconds spent building
    its index from the shared Boolean matrix (0 for Boolean itself)."""
    boolean = BooleanDataRepresentation(spec, X, y)
    if kind == "Boolean":
        return boolean, 0.0
    t = time.perf_counter()
    rep = _VERTICAL[kind].from_boolean(boolean)
    return rep, time.perf_counter() - t


class Cache:
    """One JSON line per finished measurement, keyed by what was measured."""

    def __init__(self, path: str):
        self.path = path
        self.rows: Dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[row["key"]] = row

    def get(self, key: str) -> Optional[dict]:
        return self.rows.get(key)

    def put(self, row: dict) -> None:
        self.rows[row["key"]] = row
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")


def _fit_point(runner, cache, key_base: dict, learner: str, reps: dict, test_reps: dict,
               skip: set) -> List[dict]:
    """Fits `learner` on every representation in `reps` (unless its curve
    already timed out, see `skip`), checking rules and test predictions
    against Boolean."""
    rows = []
    for kind in REPRESENTATIONS:
        key = json.dumps({**key_base, "learner": learner, "rep": kind}, sort_keys=True)
        row = cache.get(key)
        if row is None:
            if (learner, key_base.get("negation"), kind) in skip:
                continue
            (result, err) = runner.run(timed_fit, learner, reps[kind])
            if err is None:
                model, seconds = result
                row = {"key": key, **key_base, "learner": learner, "rep": kind, "fit_time": seconds,
                       "error": None, "rules": _digest(str(model)),
                       "preds": _digest(",".join(map(str, model.predict(test_reps[kind])))),
                       "n_rules": model.evaluate(test_reps[kind]).n_rules}
            else:
                row = {"key": key, **key_base, "learner": learner, "rep": kind, "fit_time": None,
                       "error": err, "rules": None, "preds": None, "n_rules": None}
            cache.put(row)
            cached = ""
        else:
            cached = " (cached)"
        status = f"{row['fit_time']:.2f}s" if row["error"] is None else row["error"]
        print(f"    {learner:8s} {kind:12s} {status}{cached}", flush=True)
        if row["error"] is not None:
            skip.add((learner, key_base.get("negation"), kind))
        rows.append(row)
    return rows


# -- part 1: training-set size -------------------------------------------------------

def run_size_curve(runner, cache, sizes) -> List[dict]:
    entry = Catalog.default().select(names=[SIZE_DATASET])[0]
    df, target = entry.load()
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(df))
    n_test = int(round(SIZE_TEST_FRACTION * len(df)))
    test_df, pool = df.iloc[order[:n_test]], df.iloc[order[n_test:]]
    rows, skip = [], set()
    for negation in (True, False):
        for n in sizes:
            train_df = pool.iloc[:n]
            spec = build_dataspec(train_df, target=target, max_intervals=MAX_INTERVALS,
                                  include_negations=negation, skip_unusable=True).build()
            X, Xt = binarize(spec, train_df), binarize(spec, test_df)
            y, yt = train_df[target].astype(str).to_numpy(), test_df[target].astype(str).to_numpy()
            reps, test_reps = {}, {}
            base = {"part": "size", "negation": negation, "n": n}
            for kind in REPRESENTATIONS:
                reps[kind], build_s = _build(kind, spec, X, y)
                test_reps[kind] = _build(kind, spec, Xt, yt)[0]
                rows.append({**base, "learner": None, "rep": kind, "build_time": build_s,
                             "density": float(X.mean()), "n_features": spec.n_features})
            print(f"  {SIZE_DATASET}, {'with' if negation else 'without'} negation, n={n} "
                  f"({spec.n_features} features, density {X.mean():.3f})", flush=True)
            for learner in LEARNERS:
                rows += _fit_point(runner, cache, base, learner, reps, test_reps, skip)
    return rows


# -- part 2: density ---------------------------------------------------------------------

def _synthetic(noise_p: float, seed: int, n: int):
    rng = np.random.default_rng(seed)
    signal = rng.random((n, DENSITY_N_SIGNAL)) < DENSITY_SIGNAL_P
    X = np.concatenate([signal, rng.random((n, DENSITY_K - DENSITY_N_SIGNAL)) < noise_p], axis=1)
    y = np.where((signal[:, 0] & ~signal[:, 1]) ^ (rng.random(n) < 0.05), "pos", "neg")
    return X, y


def run_density_sweep(runner, cache, densities) -> List[dict]:
    spec = DataSpec([f"f{i}" for i in range(DENSITY_K)])
    rows, skip = [], set()
    for p in densities:
        X, y = _synthetic(p, seed=0, n=DENSITY_N)
        Xt, yt = _synthetic(p, seed=1, n=DENSITY_N)
        reps, test_reps = {}, {}
        base = {"part": "density", "noise_density": p}
        rrng = np.random.default_rng(0)
        probe = [Rule(sorted(int(f) for f in rrng.choice(DENSITY_K, size=int(rrng.integers(1, 5)),
                                                          replace=False)), target="pos", dataspec=spec)
                 for _ in range(COVERAGE_TRIALS)]
        for kind in REPRESENTATIONS:
            reps[kind], build_s = _build(kind, spec, X, y)
            test_reps[kind] = _build(kind, spec, Xt, yt)[0]
            timings = []
            for _ in range(3):  # best of three: the first pass also pays warm-up costs
                t = time.perf_counter()
                for r in probe:
                    reps[kind].coverage(r)
                timings.append(time.perf_counter() - t)
            rows.append({**base, "learner": None, "rep": kind, "build_time": build_s,
                         "coverage_ms": min(timings) * 1000, "density": float(X.mean())})
        print(f"  noise density {p} (overall {X.mean():.3f})", flush=True)
        for learner in LEARNERS:
            rows += _fit_point(runner, cache, base, learner, reps, test_reps, skip)
    return rows


# -- plots ---------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(fig, name: str, quick: bool) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}{'_quick' if quick else ''}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return path


def _plain_log_y(ax) -> None:
    """Log y axis with tick labels as plain numbers (0.6, 2, 40), not 6x10^-1."""
    from matplotlib.ticker import FuncFormatter
    ax.set_yscale("log")
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)


def write_plots(size_rows, density_rows, quick: bool) -> Dict[str, str]:
    plt = _plt()
    paths = {}
    fits = [r for r in size_rows if r["learner"] and r["error"] is None]
    sizes = sorted({r["n"] for r in size_rows})

    def size_axes(ax, kinds, log_y: bool, ylabel: str, title: str) -> None:
        # x: log-spaced, labeled with the actual sizes; legend: colors for the
        # representations and line styles for the encoding, as two groups
        from matplotlib.lines import Line2D
        from matplotlib.ticker import NullLocator
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n:,}" for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        if log_y:
            _plain_log_y(ax)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel(f"training examples ({SIZE_DATASET})")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        handles = [Line2D([], [], color=STYLE[k][1], marker=STYLE[k][0], markersize=4, label=k) for k in kinds]
        handles += [Line2D([], [], color="black", linestyle="-", label="with negation"),
                    Line2D([], [], color="black", linestyle="--", label="without negation")]
        ax.legend(handles=handles, fontsize=7)
        ax.grid(alpha=0.3, which="both")

    def size_lines(ax, rows, value: str, kinds) -> None:
        for kind in kinds:
            marker, color = STYLE[kind]
            for negation, ls in ((True, "-"), (False, "--")):
                pts = sorted((r["n"], r[value]) for r in rows if r["rep"] == kind and r["negation"] == negation)
                if pts:
                    ax.plot(*zip(*pts), ls, marker=marker, color=color, markersize=4)

    for learner in LEARNERS:
        mine = [r for r in fits if r["learner"] == learner]
        for log_y, suffix in ((True, ""), (False, "_linear")):
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            size_lines(ax, mine, "fit_time", REPRESENTATIONS)
            size_axes(ax, REPRESENTATIONS, log_y, "fit time (s)" + ("" if log_y else ", linear scale"),
                      f"{learner}: fit time vs. training-set size")
            paths[f"size_{learner}{suffix}"] = _save(fig, f"size_{learner.lower()}{suffix}", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    builds = [r for r in size_rows if r["learner"] is None and r["rep"] != "Boolean"]
    size_lines(ax, builds, "build_time", REPRESENTATIONS[1:])
    size_axes(ax, REPRESENTATIONS[1:], True, "index build time (s)",
              "building the index from the Boolean matrix")
    paths["size_build"] = _save(fig, "size_index_build", quick)

    dfits = [r for r in density_rows if r["learner"] and r["error"] is None]
    densities = sorted({r["noise_density"] for r in density_rows})

    def density_x(ax) -> None:
        from matplotlib.ticker import NullLocator
        ax.set_xscale("log")
        ax.set_xticks(densities)
        ax.set_xticklabels([f"{d:g}" for d in densities])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.invert_xaxis()

    for learner in LEARNERS:
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        for kind in REPRESENTATIONS:
            marker, color = STYLE[kind]
            pts = sorted((r["noise_density"], r["fit_time"]) for r in dfits
                         if r["learner"] == learner and r["rep"] == kind)
            if pts:
                ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
        _plain_log_y(ax)
        density_x(ax)
        ax.set_xlabel(f"density of the noise features (k={DENSITY_K}, n={DENSITY_N})")
        ax.set_ylabel("fit time (s)")
        ax.set_title(f"{learner}: fit time vs. density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        paths[f"density_{learner}"] = _save(fig, f"density_{learner.lower()}", quick)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for kind in REPRESENTATIONS:
        marker, color = STYLE[kind]
        pts = sorted((r["noise_density"], r["coverage_ms"]) for r in density_rows
                     if r["learner"] is None and r["rep"] == kind)
        ax.plot(*zip(*pts), "-", marker=marker, color=color, markersize=4, label=kind)
    _plain_log_y(ax)
    density_x(ax)
    ax.set_xlabel(f"density of the noise features (k={DENSITY_K}, n={DENSITY_N})")
    ax.set_ylabel(f"time for {COVERAGE_TRIALS} coverage() calls (ms)")
    ax.set_title("coverage() alone vs. density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    paths["density_coverage"] = _save(fig, "density_coverage", quick)
    return paths


# -- report ----------------------------------------------------------------------------------

def _agreement(rows) -> tuple:
    """(points compared, points where every representation matched Boolean,
    list of mismatches)."""
    groups: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        if r["learner"] and r["error"] is None:
            k = json.dumps({x: r[x] for x in r if x not in (
                "key", "rep", "fit_time", "error", "rules", "preds", "n_rules")}, sort_keys=True)
            groups.setdefault(k, {})[r["rep"]] = r
    compared, agreed, bad = 0, 0, []
    for k, by_rep in groups.items():
        if "Boolean" not in by_rep or len(by_rep) < 2:
            continue
        compared += 1
        ref = by_rep["Boolean"]
        diff = [kind for kind, r in by_rep.items() if (r["rules"], r["preds"]) != (ref["rules"], ref["preds"])]
        if diff:
            bad.append((json.loads(k), diff))
        else:
            agreed += 1
    return compared, agreed, bad


def _fit_table(rows, x_field: str, x_label: str, negation=None) -> str:
    fits = [r for r in rows if r["learner"] and (negation is None or r["negation"] == negation)]
    xs = sorted({r[x_field] for r in fits}, reverse=(x_field == "noise_density"))
    lines = [f"| learner | representation | " + " | ".join(f"{x_label}={x}" for x in xs) + " |\n",
             "|---|---|" + "--:|" * len(xs) + "\n"]
    for learner in LEARNERS:
        for kind in REPRESENTATIONS:
            cells = []
            for x in xs:
                r = next((r for r in fits if r["learner"] == learner and r["rep"] == kind and r[x_field] == x), None)
                cells.append("" if r is None else ("time-out" if r["error"] == "timeout" else
                                                   "error" if r["error"] else f"{r['fit_time']:.2f}"))
            lines.append(f"| {learner} | {kind} | " + " | ".join(cells) + " |\n")
    return "".join(lines)


def _relative(rows) -> str:
    """Mean fit time relative to Boolean (same learner and point), per representation."""
    groups: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r["learner"] and r["error"] is None:
            k = json.dumps({x: r.get(x) for x in ("part", "learner", "negation", "n", "noise_density")})
            groups.setdefault(k, {})[r["rep"]] = r["fit_time"]
    lines = ["| representation | " + " | ".join(LEARNERS) + " | all |\n", "|---|" + "--:|" * (len(LEARNERS) + 1) + "\n"]
    for kind in REPRESENTATIONS[1:]:
        per = {l: [] for l in LEARNERS}
        for k, t in groups.items():
            if kind in t and t.get("Boolean", 0) >= 0.05:
                per[json.loads(k)["learner"]].append(t[kind] / t["Boolean"])
        allv = [v for vs in per.values() for v in vs]
        cell = lambda vs: f"{100 * np.mean(vs):.0f}%" if vs else ""  # noqa: E731
        lines.append(f"| {kind} | " + " | ".join(cell(per[l]) for l in LEARNERS) + f" | {cell(allv)} |\n")
    return "".join(lines)


def write_report(size_rows, density_rows, paths, quick: bool, report_path: str) -> None:
    plots = os.path.basename(PLOTS_DIR)
    img = lambda key, alt: f"![{alt}]({plots}/{os.path.basename(paths[key])})\n\n"  # noqa: E731
    size_info = {(r["negation"], r["n"]): r for r in size_rows if r["learner"] is None and r["rep"] == "Boolean"}
    compared, agreed, bad = _agreement(size_rows + density_rows)
    L = ["# Data representations: same rules, different speed\n\n"]
    L.append(
        ("**Quick run** (training sets up to 1,000 examples, short density sweep) -- a fast sanity "
         "check, the default with no arguments. Full run: `python examples/demo_representations.py "
         "--full`.\n\n" if quick else
         "**Full run.** Quick sanity check instead: `python examples/demo_representations.py`.\n\n")
        + "pyrulearn's learners never look at the data directly. They ask it two things -- which "
        "examples a rule covers (`coverage(rule)`) and which features an example has "
        "(`features_of(row)`) -- so the data can be stored in any structure that answers these "
        "questions. Four such representations are implemented:\n\n"
        "- **Boolean** -- a packed bit matrix, one bit per example and feature. Covering a rule "
        "is a bitwise AND over the rows, whatever the density.\n"
        "- **NList** -- LORD's vertical index (Huynh, Fürnkranz & Beck, 2023): examples are "
        "inserted into a prefix tree of their true features, and each feature keeps the list of "
        "tree nodes it occurs in (its N-list). Covering a rule intersects these lists; the work "
        "grows with how often the features are true, not with the number of examples.\n"
        "- **PrePostNList** -- the N-list plus pre/post-order codes of the tree nodes, a second "
        "way to find which nodes are compatible with the rule so far.\n"
        "- **Sparse** -- `scipy` sparse matrices: per-feature example lists without the prefix "
        "tree.\n\n"
        "All four must lead to exactly the same rules, so the only thing that differs is time. "
        "This demo measures that time for four learners -- CN2, PFossil, Pypper and PyLORD, "
        "each with its default settings -- and checks the rules and the test-set predictions "
        "against the Boolean version at every point. Each fit runs in a separate process and is "
        f"capped at {FIT_TIMEOUT:.0f}s; after a time-out, the larger points of that curve are "
        "skipped.\n\n"
        f"**Equality check:** at {compared} points where Boolean and at least one other "
        f"representation finished, {agreed} gave identical rules and predictions for every "
        f"representation" + (".\n\n" if not bad else
                              f"; mismatches: {bad}.\n\n"))

    L.append("## 1. Training-set size\n\n")
    L.append(
        f"Training sets of doubling size, drawn from one shuffled pool of the `{SIZE_DATASET}` "
        f"dataset (48,842 examples, 14 attributes, 8 of them nominal), each evaluated on the "
        f"same {SIZE_TEST_FRACTION:.0%} test set. Numeric attributes are discretized per "
        f"training set (`max_intervals={MAX_INTERVALS}`). Two encodings: **with negation** "
        "(pyrulearn's default -- every test comes with its negation, so exactly half the "
        "features are true in every example) and **without negation** (positive tests only, "
        "far sparser). The first is the Boolean matrix's best case and the vertical "
        "representations' worst; the second is where they should gain.\n\n")
    largest = {neg: size_info[(neg, max(n for (ng, n) in size_info if ng == neg))]
               for neg in (True, False) if any(ng == neg for (ng, _) in size_info)}
    if largest:
        L.append("At the largest training set: "
                 + "; ".join(f"{'with' if neg else 'without'} negation {r['n_features']} binary "
                             f"features, density {r['density']:.3f}" for neg, r in largest.items())
                 + " (density = share of features that are true).\n\n")
    L.append("One pair of plots per learner: left with a logarithmic time axis (ratios -- a "
             "constant factor between two representations is a constant gap), right with a "
             "linear one (absolute seconds saved).\n\n")
    for learner in LEARNERS:
        pair = [f'<img src="{plots}/{os.path.basename(paths[key])}" alt="{learner} fit time vs. '
                f'training-set size, {scale}" width="49%">'
                for key, scale in ((f"size_{learner}", "log scale"), (f"size_{learner}_linear", "linear scale"))]
        L.append(" ".join(pair) + "\n\n")
    L.append("Fit time in seconds, with negation:\n\n" + _fit_table(size_rows, "n", "n", True) + "\n")
    L.append("Without negation:\n\n" + _fit_table(size_rows, "n", "n", False) + "\n")
    L.append("Building a vertical index is paid once per fit, on top of the fit itself:\n\n")
    L.append(img("size_build", "index build time vs. training-set size"))

    L.append("## 2. Density\n\n")
    L.append(
        f"Synthetic data, {DENSITY_N} training and {DENSITY_N} test examples with "
        f"{DENSITY_K} features: {DENSITY_N_SIGNAL} of them carry a fixed learnable concept "
        f"(true with probability {DENSITY_SIGNAL_P}, class = f0 and not f1, 5% label noise); "
        "every other feature is noise, true with the probability on the x-axis. Size and "
        "feature count stay fixed, so this isolates density.\n\n")
    for learner in LEARNERS:
        L.append(img(f"density_{learner}", f"{learner} fit time vs. density"))
    L.append(
        f"The pure data operation, {COVERAGE_TRIALS} `coverage()` calls for random rules of 1-4 "
        "conditions, without any search around it:\n\n")
    L.append(img("density_coverage", "coverage() time vs. density"))
    L.append("Fit time in seconds:\n\n" + _fit_table(density_rows, "noise_density", "p") + "\n")

    L.append("## Summary: fit time relative to Boolean\n\n")
    L.append("Mean of (fit time / Boolean fit time) over all points of both parts where the "
             "Boolean fit took at least 0.05s; below 100% is faster than Boolean.\n\n")
    L.append(_relative(size_rows + density_rows) + "\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = Cache(os.path.join(CACHE_DIR, f"{'quick' if quick else 'full'}.jsonl"))
    with TimeoutRunner(timeout=FIT_TIMEOUT) as runner:
        print(f"Part 1: training-set size ({SIZE_DATASET})", flush=True)
        size_rows = run_size_curve(runner, cache, QUICK_SIZES if quick else SIZES)
        print("Part 2: density", flush=True)
        density_rows = run_density_sweep(runner, cache, QUICK_DENSITIES if quick else DENSITIES)
    paths = write_plots(size_rows, density_rows, quick)
    report = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")
    write_report(size_rows, density_rows, paths, quick, report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="full curves (training sets up to 32,000, the whole density sweep)")
    main(quick=not parser.parse_args().full)
