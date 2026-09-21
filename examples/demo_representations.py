"""
Four encodings of the same data, all behind one `DataRepresentation`
interface:

- `BooleanDataRepresentation` -- a `numpy.packbits` bit-matrix (the
  baseline).
- `NListRepresentation` -- LORD's PPC-tree / N-list vertical index:
  per-feature row lists, bucketed by shared row prefix.
- `PrePostNListRepresentation` -- `NListRepresentation` plus pre/post
  visit codes, giving its search fast path a second way to answer "which
  of this feature's nodes are compatible with the rule so far" (binary
  search against the current active subtree ranges, instead of scanning
  the feature's whole N-list). Included here specifically to show it's
  *not* faster in practice at these data sizes -- see its own docstring
  for the measured story -- while still producing identical rules.
- `SparseDataRepresentation` -- `scipy` CSR/CSC: per-feature row lists,
  *not* bucketed. This is the N-list without the prefix tree (an N-list
  built on a PPC-tree that did no prefix merging would have exactly
  these lists).

Every `pyrulearn.learners.seco` learner asks a data only two things --
`coverage(rule)` and `features_of(row)` -- so all four are
interchangeable and must produce byte-identical rules. This demo checks
that on six binary UCI/OpenML benchmarks, in **both feature encodings**:

- **with negation** (pyrulearn's default): each test paired with its
  explicit negation (`not f`, `x != v`, `x < t`); every row has ~half
  the columns set.
- **without negation**: positive tests only; rows are sparser -- the
  regime where the vertical encodings actually pull ahead of the packed
  matrix. Built two ways that must agree: from scratch
  (`build_dataspec(include_negations=False)`) and by calling
  `.without_negations()` on the negated data.

Per (dataset, encoding) it reports, for each data: rules /
predictions identical to the Boolean baseline, accuracy, fit time, a
1000-random-rule `coverage` micro-benchmark, and the vertical index
size vs the dense matrix.

CN2 / PFoil / PFossil run on every dataset; AQR and PyLORD (tens of
seconds per fit on medium data) only on the two smallest, as a real-data
spot check -- full equivalence, both encodings, is in
`tests/test_representations.py`.

A final summary table averages each vertical data's relative
fit time (its fit time / the Boolean fit time for that same learner x
dataset x encoding, Boolean = 100%) over every comparison, split by
encoding and overall.

Last, two **synthetic sweeps**, well past what the six real datasets
above cover, each ending in a 2x2 plot (see `_plot_sweep`'s own
docstring): fit time and `coverage()` time, each as both a log-log
overlay (ratio-comparable) and a log-x/linear-y *gap* -- Boolean's time
minus each data's -- which is what actually answers "is the
gap growing", since a log-scaled overlay of the raw times shows ratios,
not differences, and can make an absolutely-growing gap look flat or
even shrinking.

- `run_feature_count_sweep` -- holds a fixed, learnable concept and
  grows the number of pure-noise columns, each at a density that shrinks
  as more are added (the same scaling a one-hot-encoded, increasingly
  high-cardinality attribute would have) -- so density falls only as a
  *side effect* of the feature count growing.
- `run_density_sweep` -- fixes the feature count and sweeps the noise
  columns' own density directly instead, isolating density from feature
  count. Absolute times for `NList`/`PrePostNList`/`Sparse` actually
  *fall* here as density drops (their `coverage()` cost tracks each
  feature's own support, which shrinks); Boolean's stays flat (it always
  scans the same dense `n x k/8`-byte array regardless of how many bits
  are set) -- so the gap still grows even though the non-Boolean lines
  themselves are heading toward zero, not away from it.

Skip both with ``--no-sparsity-sweep``.

Run: ``python examples/demo_representations.py`` (``--datasets a,b,c`` to
override, ``--datasets large`` for just kr-vs-kp / mushroom). Writes
``demo_representations_report.md`` (and, unless skipped,
``demo_representations_sparsity_by_k.png`` /
``demo_representations_sparsity_by_density.png``).
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import demo_workflow_comparison as base
from pyrulearn import DataSpec, Rule
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.learners.pylord import PyLORD
from pyrulearn.data import (
    BooleanDataRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    SparseDataRepresentation,
)
from pyrulearn.learners.seco import AQR, CN2, PFoil, PFossil

RANDOM_STATE = base.RANDOM_STATE
MAX_INTERVALS = base.MAX_INTERVALS
DATASETS = ["vote", "tic-tac-toe", "breast-w", "diabetes", "kr-vs-kp", "mushroom"]
LARGE_DATASETS = ["kr-vs-kp", "mushroom"]
SLOW_LEARNER_DATASETS = {"vote", "tic-tac-toe"}
VERTICAL_KINDS = ("NList", "PrePostNList", "Sparse")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "demo_representations_report.md")


def _learners(pos_class, dataset_name):
    fast = {
        "CN2": lambda: CN2(target_class=pos_class),
        "PFoil": lambda: PFoil(target_class=pos_class),
        "PFossil": lambda: PFossil(target_class=pos_class),
    }
    if dataset_name not in SLOW_LEARNER_DATASETS:
        return fast
    return {
        **fast,
        "AQR": lambda: AQR(target_class=pos_class, maxstar=5),
        "PyLORD": lambda: PyLORD(m=0.1, random_state=RANDOM_STATE),
    }


def _rule_key(ruleset):
    return sorted((r.pos, str(r.target)) for r in ruleset.rules)


def _time_coverage(rep, rules):
    t = time.perf_counter()
    for r in rules:
        rep.coverage(r)
    return time.perf_counter() - t


def _timed_fit(make, rep):
    """(rules, seconds, cpu_fallback). Falls back to CPU time when wall
    time ran far ahead of it -- the laptop suspended mid-fit and
    `perf_counter` counted the sleep. A `*` in the report marks those."""
    w0, c0 = time.perf_counter(), time.process_time()
    rules = make().fit(rep)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    if wall - cpu > 30:
        return rules, cpu, True
    return rules, wall, False


_REP_CLASSES = {
    "NList": NListRepresentation,
    "PrePostNList": PrePostNListRepresentation,
    "Sparse": SparseDataRepresentation,
}


def _build(kind, spec, df, y):
    """A (BooleanDataRepresentation, seconds) for `kind`, timing only the
    vertical index build (the Boolean matrix is shared, built once)."""
    t = time.perf_counter()
    if kind == "Boolean":
        rep = BooleanDataRepresentation(spec, binarize(spec, df), y)
    else:
        b = BooleanDataRepresentation(spec, binarize(spec, df), y)
        rep = _REP_CLASSES[kind].from_boolean(b)
    return rep, time.perf_counter() - t


def _index_size(kind, rep):
    if kind in ("NList", "PrePostNList"):
        return f"{sum(w.shape[0] for w in rep._path_words)} tree nodes"
    if kind == "Sparse":
        return f"nnz={rep._csr.nnz}, density {rep._csr.nnz / (rep.n_samples * rep.spec.n_features):.2f}"
    return "-"


def _check_without_negations(neg_reps, pos_spec, train_df):
    """`.without_negations()` on each negated data must match a
    from-scratch `include_negations=False` build. Feature names are
    identical between the two, so line the columns up by name."""
    want = binarize(pos_spec, train_df)
    for kind, rep in neg_reps.items():
        dropped = rep.without_negations()
        order = [dropped.spec.feature_index(n) for n in pos_spec.feature_names]
        if not np.array_equal(dropped.X[:, order], want):
            return f"**NO** ({kind})"
    return "yes"


#: fit times below this (seconds, Boolean baseline) are excluded from the
#: relative-fit-time summary -- too close to timer resolution / rounding
#: noise for a ratio to mean anything (e.g. a 2ms fit reported as "0.00s").
MIN_FIT_FOR_RATIO = 0.02


def run_encoding(name, md, train_df, test_df, target_col, pos_class, negation, neg_reps, rel_records):
    train_y, test_y = train_df[target_col].to_numpy(), test_df[target_col].to_numpy()
    spec = build_dataspec(train_df, target=target_col, arff_types=None,
                          max_intervals=MAX_INTERVALS, include_negations=negation).build()

    reps, builds = {}, {}
    for kind in ("Boolean",) + VERTICAL_KINDS:
        reps[kind], builds[kind] = _build(kind, spec, train_df, train_y)
    test_reps = {k: _build(k, spec, test_df, test_y)[0] for k in reps}

    tag = "with negation" if negation else "without negation"
    print(f"\n{name} [{tag}]  (train n={len(train_df)}, features={spec.n_features})")
    for k in VERTICAL_KINDS:
        print(f"  {k} build {builds[k] * 1000:.0f} ms  [{_index_size(k, reps[k])}]")

    md.append(f"### {name} -- {tag}\n\n")
    md.append(f"train n={len(train_df)}, features={spec.n_features}. Vertical index build: "
              + ", ".join(f"{k} {builds[k] * 1000:.0f} ms ({_index_size(k, reps[k])})" for k in VERTICAL_KINDS)
              + ".\n\n")
    if not negation:
        agree = _check_without_negations(neg_reps, spec, train_df)
        md.append(f"`.without_negations()` on the negated representations matches this "
                  f"from-scratch build: **{agree}**.\n\n")
        print(f"  without_negations() matches from-scratch build: {agree}")

    rng = np.random.default_rng(RANDOM_STATE)
    probe = [Rule(sorted(int(f) for f in rng.choice(spec.n_features,
                  size=int(rng.integers(1, 5)), replace=False)),
                  target=pos_class, dataspec=spec) for _ in range(1000)]
    cov_ms = {k: _time_coverage(reps[k], probe) * 1000 for k in reps}

    md.append("| learner | data | rules == | preds == | acc | fit | 1k coverage |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    ok = True
    cpu_note = False
    for lname, make in _learners(pos_class, name).items():
        base_rules, base_fit, cf = _timed_fit(make, reps["Boolean"])
        cpu_note |= cf
        base_pred = np.asarray(base_rules.predict(test_reps["Boolean"]))
        acc = float(np.mean(base_pred == test_y))
        md.append(f"| {lname} | Boolean | (ref) | (ref) | {acc:.3f} | {base_fit:.2f}s{'*' if cf else ''} "
                  f"| {cov_ms['Boolean']:.0f} ms |\n")
        for kind in VERTICAL_KINDS:
            rs, fit_s, cf = _timed_fit(make, reps[kind])
            cpu_note |= cf
            rules_same = _rule_key(rs) == _rule_key(base_rules)
            pred = np.asarray(rs.predict(test_reps[kind]))
            preds_same = np.array_equal(pred, base_pred)
            ok &= rules_same and preds_same
            flag = "" if rules_same else " *** MISMATCH ***"
            print(f"  {lname:8s} {kind:12s} rules={'==' if rules_same else '!='} "
                  f"preds={'==' if preds_same else '!='} fit {fit_s:.2f}s{flag}")
            md.append(f"| {lname} | {kind} | {'yes' if rules_same else '**NO**'} "
                      f"| {'yes' if preds_same else '**NO**'} | {acc:.3f} "
                      f"| {fit_s:.2f}s{'*' if cf else ''} | {cov_ms[kind]:.0f} ms |\n")
            if base_fit >= MIN_FIT_FOR_RATIO:
                rel_records.append({"dataset": name, "negation": negation, "learner": lname,
                                    "kind": kind, "rel": fit_s / base_fit})
    if cpu_note:
        md.append("\n\\* wall time ran ahead of CPU time (laptop suspended mid-fit); CPU time shown.\n")
    md.append("\n")
    return ok, {"acc": acc, "n_features": spec.n_features,
                "sparse_density": reps["Sparse"]._csr.nnz / (len(train_df) * spec.n_features)}


def run_dataset(name, md, rel_records):
    df, target_col = base.load_openml(name)
    classes = sorted(pd.unique(df[target_col]).tolist())
    if len(classes) != 2:
        print(f"  {name}: not binary ({classes}), skipping")
        return None
    pos_class = classes[0]
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=RANDOM_STATE,
                                         stratify=df[target_col])

    md.append(f"## {name}\n\n")
    ok_all = True
    summ = {}
    neg_spec = build_dataspec(train_df, target=target_col, arff_types=None,
                              max_intervals=MAX_INTERVALS, include_negations=True).build()
    neg_reps = {k: _build(k, neg_spec, train_df, train_df[target_col].to_numpy())[0]
                for k in ("Boolean",) + VERTICAL_KINDS}
    for negation in (True, False):
        ok, s = run_encoding(name, md, train_df, test_df, target_col, pos_class, negation, neg_reps, rel_records)
        ok_all &= ok
        summ["neg" if negation else "pos"] = s
    md.append(
        f"*Encoding effect:* with negation {summ['neg']['n_features']} features "
        f"(acc {summ['neg']['acc']:.3f}); without, {summ['pos']['n_features']} features "
        f"(acc {summ['pos']['acc']:.3f}), sparse density "
        f"{summ['neg']['sparse_density']:.2f} -> {summ['pos']['sparse_density']:.2f}.\n\n---\n\n"
    )
    return ok_all


def _rows_by_negation(rel_records, kind, negation):
    return [r for r in rel_records if r["kind"] == kind and r["negation"] == negation]


def _summary_table(rel_records):
    """Markdown lines for the average-relative-fit-time summary (also
    printed to stdout): mean of (data fit / Boolean fit) across
    every learner x dataset x encoding comparison, per data and
    per encoding -- 100% = as fast as Boolean, under 100% is faster.
    Comparisons whose Boolean fit was under `MIN_FIT_FOR_RATIO` seconds
    are excluded (a ratio of two near-zero, rounding-dominated times
    isn't informative)."""
    lines = ["## Summary -- average relative fit time (Boolean = 100%)\n\n"]
    if not rel_records:
        lines.append(f"*No fit-time comparison cleared the {MIN_FIT_FOR_RATIO}s noise floor.*\n\n")
        return lines
    lines.append(
        "Mean of (data fit time / Boolean fit time) over every learner x dataset x "
        f"encoding comparison whose Boolean fit took at least {MIN_FIT_FOR_RATIO}s (below that, "
        "rounding noise swamps the ratio). 100% = as fast as Boolean; under 100% is faster.\n\n"
    )
    lines.append("| data | with negation | without negation | overall | n |\n")
    lines.append("|---|---|---|---|---|\n")
    print(f"\nSummary -- average relative fit time (Boolean = 100%, n={len(rel_records)} comparisons):")
    for kind in VERTICAL_KINDS:
        rows = [r["rel"] for r in rel_records if r["kind"] == kind]
        with_neg = [r["rel"] for r in _rows_by_negation(rel_records, kind, True)]
        without_neg = [r["rel"] for r in _rows_by_negation(rel_records, kind, False)]

        def pct(xs):
            return f"{100 * np.mean(xs):.0f}%" if xs else "n/a"

        lines.append(f"| {kind} | {pct(with_neg)} | {pct(without_neg)} | {pct(rows)} | {len(rows)} |\n")
        print(f"  {kind:12s} with negation={pct(with_neg):>5s}  without negation={pct(without_neg):>5s}  "
              f"overall={pct(rows):>5s}  (n={len(rows)})")
    lines.append("\n")
    return lines


#: synthetic sweeps -- see run_feature_count_sweep / run_density_sweep
SWEEP_N = 1000
SWEEP_N_SIGNAL = 4
SWEEP_SIGNAL_P = 0.4
SWEEP_COVERAGE_TRIALS = 800
SWEEP_LEARNER = "PFossil"  # HillClimbing, not a beam -- doesn't blow up the way CN2 did on
                           # this data (many pure-noise columns each with *some* chance
                           # correlation, which CN2's beam search + significance test spent a
                           # long time sifting through -- up to 70s/fit at k=4000; measured,
                           # not assumed): PFossil's correlation cutoff rejects those directly.

FEATURE_COUNT_K_VALUES = [20, 50, 100, 250, 500, 1000, 2000, 4000]
FEATURE_COUNT_NOISE_TARGET = 5.0  # expected "on" noise columns per row, held ~constant as k grows

DENSITY_K = 500                   # fixed feature count for the density sweep
DENSITY_VALUES = [0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


def _sweep_dataset(n, k, noise_p, seed=0):
    """`n` rows, `k` Boolean columns: `SWEEP_N_SIGNAL` fixed "signal"
    columns at a fixed density (`SWEEP_SIGNAL_P`) carry a learnable
    concept that never changes; every other column is pure noise at
    `noise_p`. Shared by both sweeps below: `run_feature_count_sweep`
    grows `k` while holding *the expected per-row noise count* constant
    (so `noise_p` falls as a side effect -- the one-hot-encoding-an-
    increasingly-high-cardinality-attribute scaling); `run_density_sweep`
    holds `k` fixed and varies `noise_p` directly, isolating density from
    feature count. Returns `(BooleanDataRepresentation, DataSpec,
    overall density)`.
    """
    rng = np.random.default_rng(seed)
    n_signal = min(SWEEP_N_SIGNAL, k)
    signal = rng.random((n, n_signal)) < SWEEP_SIGNAL_P
    n_noise = k - n_signal
    X = np.concatenate([signal, rng.random((n, n_noise)) < noise_p], axis=1) if n_noise > 0 else signal
    concept = (signal[:, 0] & ~signal[:, 1]) if n_signal >= 2 else signal[:, 0]
    y = np.where(concept ^ (rng.random(n) < 0.05), "pos", "neg")
    ds = DataSpec([f"f{i}" for i in range(k)])
    return BooleanDataRepresentation(ds, X, y), ds, float(X.mean())


def _plot_sweep(rows, filename, xlabel, suptitle):
    """Two rows of panels, fit time and `coverage()` time in each column:

    - **top row, log-log**: the raw times, ratio-comparable across the
      whole swept range -- but a log Y axis shows *ratios*, not
      differences, so a shrinking small number next to a flat larger one
      can look deceptively like it's "catching up" when it's actually
      falling further behind in absolute terms.
    - **bottom row, log-x/linear-y**: the *gap* itself -- Boolean's time
      minus each data's, in seconds/ms, so "is the gap
      growing" is a monotonic line going up (or down), not something to
      eyeball between two overlaid curves. X stays log-scaled (the swept
      range spans 2-3 orders of magnitude either way) but Y is linear on
      purpose, per the whole point of this panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("Boolean",) + VERTICAL_KINDS
    # explicit marker *and* color per name -- matplotlib's automatic color
    # cycle restarts on every new Axes, and the bottom row plots one fewer
    # series (no Boolean-vs-itself line), so relying on cycle order would
    # silently reassign NList's color between the top and bottom rows.
    styles = {"Boolean": "o-", "NList": "s-", "PrePostNList": "^-", "Sparse": "d-"}
    colors = {"Boolean": "tab:blue", "NList": "tab:orange", "PrePostNList": "tab:green", "Sparse": "tab:red"}
    xs = [r["x"] for r in rows]

    fig, ((ax_fit, ax_cov), (ax_fit_gap, ax_cov_gap)) = plt.subplots(2, 2, figsize=(11, 8.5))
    for name in names:
        ax_fit.plot(xs, [r["fit"][name] for r in rows], styles[name], color=colors[name], label=name)
        ax_cov.plot(xs, [r["cov"][name] for r in rows], styles[name], color=colors[name], label=name)
    for ax, title, ylabel in (
        (ax_fit, f"{SWEEP_LEARNER} fit time (log-log)", "fit time (s)"),
        (ax_cov, f"coverage() time (log-log, {SWEEP_COVERAGE_TRIALS} rules)", "time (ms)"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    for name in VERTICAL_KINDS:
        ax_fit_gap.plot(xs, [r["fit"]["Boolean"] - r["fit"][name] for r in rows],
                        styles[name], color=colors[name], label=name)
        ax_cov_gap.plot(xs, [r["cov"]["Boolean"] - r["cov"][name] for r in rows],
                        styles[name], color=colors[name], label=name)
    for ax, title, ylabel in (
        (ax_fit_gap, "fit time saved vs. Boolean (linear)", "Boolean fit - data fit (s)"),
        (ax_cov_gap, "coverage() time saved vs. Boolean (linear)", "Boolean cov - data cov (ms)"),
    ):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(suptitle)
    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), filename)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def _run_synthetic_sweep(md, *, name, x_header, points, plot_filename, plot_xlabel):
    """Shared driver for both sweeps. `points`: list of
    ``(x_display, k, noise_p)`` -- `x_display` is what gets plotted/
    tabulated (`k` for the feature-count sweep, `noise_p` itself for the
    density sweep). At each point: build all four representations from
    the same `_sweep_dataset(SWEEP_N, k, noise_p)`, a `coverage()`
    micro-benchmark, and one `SWEEP_LEARNER` fit (rules/predictions
    checked against Boolean -- this doubles as a large-scale correctness
    check, not just a timing curve). Ends with a log-log 2-panel plot.
    """
    print(f"\n{'=' * 78}\n{name} (n={SWEEP_N})\n{'=' * 78}")
    md.append(f"## {name}\n\n")
    names = ("Boolean",) + VERTICAL_KINDS
    md.append(f"| {x_header} | density | rules == | " + " | ".join(f"{n} fit" for n in names) + " | "
              + " | ".join(f"{n} cov" for n in names) + " |\n")
    md.append("|---" * (3 + 2 * len(names)) + "|\n")

    rows, all_ok = [], True
    for x_display, k, noise_p in points:
        brep, ds, density = _sweep_dataset(SWEEP_N, k, noise_p)
        reps = {"Boolean": brep}
        for kind in VERTICAL_KINDS:
            reps[kind] = _REP_CLASSES[kind].from_boolean(brep)

        rng = np.random.default_rng(0)
        probe = [Rule(sorted(int(f) for f in rng.choice(k, size=int(rng.integers(1, 5)), replace=False)),
                      target="pos", dataspec=ds) for _ in range(SWEEP_COVERAGE_TRIALS)]
        cov_s = {n: _time_coverage(reps[n], probe) for n in names}

        fit_s, rules_same, base_rules = {}, True, None
        for n in names:
            rs, ft, _ = _timed_fit(lambda: PFossil(target_class="pos"), reps[n])
            fit_s[n] = ft
            if base_rules is None:
                base_rules = rs
            else:
                rules_same &= _rule_key(rs) == _rule_key(base_rules)
        all_ok &= rules_same

        print(f"  {x_header}={x_display!s:>8} density={density:.4f} rules_same={rules_same}  "
              + "  ".join(f"{n}: fit={fit_s[n]:.3f}s cov={cov_s[n] * 1000:.0f}ms" for n in names))
        md.append(f"| {x_display} | {density:.4f} | {'yes' if rules_same else '**NO**'} | "
                  + " | ".join(f"{fit_s[n]:.3f}s" for n in names) + " | "
                  + " | ".join(f"{cov_s[n] * 1000:.0f} ms" for n in names) + " |\n")
        rows.append({"x": x_display, "density": density, "fit": fit_s, "cov": {n: cov_s[n] * 1000 for n in names}})

    plot_path = _plot_sweep(rows, plot_filename, plot_xlabel, name)
    md.append(f"\n![{name}]({os.path.basename(plot_path)})\n\n")
    md.append(f"**{SWEEP_LEARNER} agreed with Boolean at every point: {'yes' if all_ok else 'NO'}.**\n\n---\n\n")
    print(f"\nPlot written to {plot_path}")
    return all_ok


def run_feature_count_sweep(md):
    """Grows the feature count `k`; each added column is pure noise at a
    density that shrinks as more are added (`FEATURE_COUNT_NOISE_TARGET
    / n_noise`, capped at 0.5, so the expected per-row noise count stays
    ~constant) -- the same scaling a one-hot encoding of an increasingly
    high-cardinality attribute would have. Density falls only as a *side
    effect* of adding columns; see `run_density_sweep` for density
    varied directly at fixed `k`."""
    md.append(
        f"Fixed concept ({SWEEP_N_SIGNAL} signal columns, density {SWEEP_SIGNAL_P}) plus a growing "
        f"number of pure-noise columns whose *own* density shrinks as more are added "
        f"(`{FEATURE_COUNT_NOISE_TARGET:g} / n_noise`, capped at 0.5) -- so overall density falls "
        f"purely as a side effect of adding columns, n={SWEEP_N} fixed. Coverage timing is "
        f"{SWEEP_COVERAGE_TRIALS} random rules (length 1-4); fit is one {SWEEP_LEARNER} fit per "
        "point, rules/predictions checked against Boolean.\n\n"
        "**Fit-time gap vs. coverage-time gap**: the coverage-time gap (bottom right) grows "
        "cleanly and monotonically the whole way -- it isolates exactly the data-"
        "dependent computation. The fit-time gap (bottom left) does *not* stay monotonic, and "
        "goes slightly negative at the largest k -- because `k` also controls how many "
        "candidates the search itself has to score each round (`O(k)`, the same for every "
        "data), and that shared, data-agnostic cost grows right alongside "
        "the feature count. At extreme sparsity the actual coverage computation is already "
        "nearly free for everyone, so what's left is that per-candidate bookkeeping overhead -- "
        "and `NList`'s handle (`anchor_item`/`active`/`mask_words`/`ctx`) carries a bit more of "
        "it per call than Boolean's plain `(cov, scope)` pair. Not a data regression;"
        " a reminder that fit time bundles search overhead in with coverage cost, and only the "
        "latter is what these representations actually compete on.\n\n"
    )
    points = []
    for k in FEATURE_COUNT_K_VALUES:
        n_noise = k - min(SWEEP_N_SIGNAL, k)
        noise_p = min(0.5, FEATURE_COUNT_NOISE_TARGET / n_noise) if n_noise > 0 else 0.0
        points.append((k, k, noise_p))
    return _run_synthetic_sweep(
        md, name="Synthetic feature-count sweep (density falls as k grows)", x_header="k",
        points=points, plot_filename="demo_representations_sparsity_by_k.png",
        plot_xlabel="k (features) -- density falls as k grows",
    )


def run_density_sweep(md):
    """Fixes the feature count (`DENSITY_K`) and sweeps the noise
    columns' own density directly over `DENSITY_VALUES` -- isolates
    density from feature count, the complement of
    `run_feature_count_sweep` (there, density only fell as a side effect
    of adding columns)."""
    md.append(
        f"Fixed feature count (k={DENSITY_K}, {SWEEP_N_SIGNAL} of them a fixed-density "
        f"({SWEEP_SIGNAL_P}) learnable concept) with every *other* column's own density swept "
        "directly -- isolates density from feature count, unlike the feature-count sweep above "
        f"where density only fell as a side effect of adding columns. n={SWEEP_N} fixed. "
        f"Coverage timing is {SWEEP_COVERAGE_TRIALS} random rules (length 1-4); fit is one "
        f"{SWEEP_LEARNER} fit per point, rules/predictions checked against Boolean.\n\n"
    )
    points = [(p, DENSITY_K, p) for p in DENSITY_VALUES]
    return _run_synthetic_sweep(
        md, name=f"Synthetic density sweep (k={DENSITY_K} fixed)", x_header="noise density",
        points=points, plot_filename="demo_representations_sparsity_by_density.png",
        plot_xlabel=f"noise column density (k={DENSITY_K} fixed)",
    )


def main(datasets=None, sparsity_sweep=True):
    datasets = datasets or DATASETS
    md = [
        "# Four `DataRepresentation` encodings -- same rules, with & without negation\n\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}. Single stratified 67/33 split, "
        f"`random_state={RANDOM_STATE}`. Each learner is fit on all four representations "
        "(Boolean = packed bit-matrix, NList = PPC-tree / N-list, PrePostNList = NList + "
        "pre/post visit codes on its search fast path, Sparse = scipy CSR/CSC = the N-list "
        "without the prefix tree) and the rule sets / prediction vectors are compared to the "
        "Boolean baseline exactly. Every dataset is run in two feature encodings: **with "
        "negation** (paired negation features) and **without** (positive tests only). "
        "`coverage` timing is 1000 random rules (length 1-4).\n\n"
        "CN2 / PFoil / PFossil run everywhere; AQR / PyLORD only on vote & tic-tac-toe. "
        "PrePostNList is included to *demonstrate* it stays correct, not because it's "
        "expected to be faster here -- see `PrePostNListRepresentation`'s own docstring for "
        "why it measured out roughly break-even to slightly worse than plain NList at these "
        "data sizes.\n\n---\n\n",
    ]
    every_ok, errors = True, []
    rel_records = []
    for nm in datasets:
        try:
            r = run_dataset(nm, md, rel_records)
            if r is not None:
                every_ok &= r
        except Exception as e:  # noqa: BLE001
            print(f"  {nm}: could not run -- {type(e).__name__}: {e}")
            md.append(f"## {nm}\n\n*could not run: {type(e).__name__}: {e}*\n\n")
            errors.append(nm)

    md.extend(_summary_table(rel_records))
    md.append("---\n\n")
    md.append(f"**Every data agreed with the Boolean baseline on every learner, "
              f"dataset and encoding: {'yes' if every_ok else 'NO -- see mismatches above'}.**\n\n")
    if errors:
        md.append(f"\n*Could not load/run: {', '.join(errors)}.*\n")

    if sparsity_sweep:
        every_ok &= run_feature_count_sweep(md)
        every_ok &= run_density_sweep(md)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"\nReport written to {REPORT_PATH}")
    print("ALL IDENTICAL" if every_ok else "*** MISMATCHES FOUND ***")
    if errors:
        print(f"(could not run: {', '.join(errors)})")
    return 0 if every_ok else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Boolean vs N-list vs sparse representations, with/without negation.")
    parser.add_argument("--datasets", type=str, default=None,
                        help="comma-separated OpenML names (default: " + ",".join(DATASETS)
                             + "); 'large' = " + ",".join(LARGE_DATASETS))
    parser.add_argument("--no-sparsity-sweep", action="store_true",
                        help="skip the synthetic growing-sparsity sweep (on by default).")
    args = parser.parse_args()
    if args.datasets == "large":
        ds = LARGE_DATASETS
    elif args.datasets:
        ds = args.datasets.split(",")
    else:
        ds = None
    sys.exit(main(ds, sparsity_sweep=not args.no_sparsity_sweep))
