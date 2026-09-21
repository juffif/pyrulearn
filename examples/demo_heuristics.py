"""
Demo: `RuleHeuristic` isometrics via `CoverageSpace`.

Two things, each rendered in both a raw coverage space and a normalized
ROC space:

1. Every defined heuristic's isometrics, plotted standalone (default
   `CoverageSpace` dimensions -- no dataset involved at all).
2. One example rule's refinement path (see `pyrulearn.evaluation.
   rule_refinement_path`), layered on top of `Precision`'s and
   `Accuracy`'s isometrics on the same space -- a concrete illustration
   of how a greedy, precision-ordered refinement moves through
   Precision's isometrics (always "uphill", since that's what it's
   greedily optimizing) versus Accuracy's (not necessarily uphill, since
   the rule isn't optimizing that heuristic at all).
3. `Precision`/`Laplace`/`MEstimate` side by side, illustrating why the
   pencil family's *pivot* matters: `Laplace`'s pivot is at (-1, -1)
   instead of `Precision`'s origin, and `MEstimate(m)`'s slides along
   that same line as `m` grows -- but a shift of a few units is
   invisible at the default 100/160-sized space (a handful of examples
   out of 100+ barely moves anything), so this uses a much smaller
   space instead, where the same absolute shift is a real fraction of
   the total.

`LengthPenalized` (needs `stats.length`, which defaults to 0 here --
indistinguishable from its wrapped heuristic on this kind of plot) is
left out of the heuristic grid for that reason. `FoilGain` (a
`GainHeuristic`) normally needs a parent `RuleStats` too (meaningless
for a bare grid point) -- made plottable here via `plot_isometrics`'s
`parent=` argument, fixed to
`RuleStats.universal(space.n_pos, space.n_neg)` (the "no rule yet"
baseline) for every point in the grid, so the plotted quantity is "gain
over the universal rule" throughout. The main grid's `MEstimate`/
`GHeuristic` instances use `m=20`/`g=30` (not smaller, more typical
values) since, unlike `Laplace`'s fixed pivot shift, their pivot- and
slope-changing effects are already visible even at the default space
size with these larger parameter values.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyrulearn import BooleanDataRepresentation, DataSpec, DataSpecBuilder, Rule
from pyrulearn.evaluation import CoverageSpace
from pyrulearn.heuristics import (
    Accuracy,
    Correlation,
    Coverage,
    CoverageDifference,
    Entropy,
    FBeta,
    FoilGain,
    GainHeuristic,
    GeneralizedMEstimate,
    GHeuristic,
    Laplace,
    LikelihoodRatio,
    LinearCost,
    LinearCostRates,
    MEstimate,
    CoveredNegatives,
    Precision,
    Recall,
    RuleStats,
    Support,
    WRAcc,
    YoudenJ,
)

OUT_DIR = os.path.dirname(__file__)

HEURISTICS = [
    ("CoveredNegatives", CoveredNegatives()),
    ("Recall", Recall()),
    ("Precision", Precision()),
    ("FBeta(1)", FBeta()),
    ("Laplace", Laplace()),
    ("MEstimate(20)", MEstimate(20)),
    ("GeneralizedMEstimate(20, 0.9)", GeneralizedMEstimate(20, 0.9)),
    ("GHeuristic(30)", GHeuristic(30)),
    ("WRAcc", WRAcc()),
    ("YoudenJ", YoudenJ()),
    ("Accuracy", Accuracy()),
    ("CoverageDifference", CoverageDifference()),
    ("Support", Support()),
    ("Coverage", Coverage()),
    ("LinearCost(2.0)", LinearCost(2.0)),
    ("LinearCostRates(2.0)", LinearCostRates(2.0)),
    ("Correlation", Correlation()),
    ("Entropy", Entropy()),
    ("LikelihoodRatio", LikelihoodRatio()),
    ("FoilGain (vs. universal rule)", FoilGain()),
]


def plot_heuristic_grid(normalized: bool, out_name: str) -> None:
    n_cols = 4
    n_rows = -(-len(HEURISTICS) // n_cols)  # ceil division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    for ax, (name, h) in zip(axes.flat, HEURISTICS):
        space = CoverageSpace(ax=ax, normalized=normalized, title=name)
        # a GainHeuristic like FoilGain needs a parent RuleStats -- fix
        # it to the universal rule (the "no rule yet" baseline) so it
        # becomes plottable at all
        extra = {"parent": RuleStats.universal(space.n_pos, space.n_neg)} if isinstance(h, GainHeuristic) else {}
        h.plot_isometrics(space=space, levels=8, **extra)
    for ax in axes.flat[len(HEURISTICS):]:
        ax.axis("off")
    kind = "ROC space (normalized)" if normalized else "coverage space (raw counts)"
    fig.suptitle(f"Heuristic isometrics -- {kind}, default dimensions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def build_example_rule():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    # ground truth: y = f0 AND NOT f1 AND (f2 OR f3), with a bit of noise
    y_clean = X[:, 0] & ~X[:, 1] & (X[:, 2] | X[:, 3])
    noise = rng.random(n) < 0.05
    y = np.where(noise, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")

    # explicit-negation feature space (the default): f_i paired with "not f_i"
    _b = DataSpecBuilder()
    for i in range(d):
        _b.add_boolean(f"f{i}")
    ds = _b.build()
    X_full = np.empty((n, 2 * d), dtype=bool)
    X_full[:, 0::2] = X
    X_full[:, 1::2] = ~X
    rep = BooleanDataRepresentation(ds, X_full, y)
    rule = Rule.from_pos_neg(
        pos=[ds.feature_index("f0"), ds.feature_index("f2")],
        neg=[ds.feature_index("f1")], target="pos", dataspec=ds,
    )
    ordered = rule.order_by_precision(rep)  # a meaningful step-by-step order to plot
    return ordered, rep


def plot_refinement_with_isometrics(normalized: bool, out_name: str) -> None:
    rule, rep = build_example_rule()
    space = CoverageSpace.from_data(
        rep, "pos", normalized=normalized,
        title="Rule refinement over Precision/Accuracy isometrics",
    )
    # thin dashed lines, in colors distinct from the refinement path's own
    # (plot_rule_refinement draws it in "C1", matplotlib's default orange)
    # -- solid same-color isometrics would be hard to tell apart from the
    # path itself
    iso_style = dict(levels=8, linestyles="dashed", linewidths=0.8)
    Precision().plot_isometrics(space=space, colors="tab:blue", **iso_style)
    Accuracy().plot_isometrics(space=space, colors="tab:green", **iso_style)
    # unlabeled contour sets don't show up in a legend on their own -- add
    # invisible proxy lines so the final legend() call (inside
    # plot_rule_refinement) can tell the two heuristics' isometrics apart
    space.ax.plot([], [], color="tab:blue", linestyle="dashed", label="Precision isometrics")
    space.ax.plot([], [], color="tab:green", linestyle="dashed", label="Accuracy isometrics")
    space.plot_rule_refinement(rule, rep, annotate=True)

    # highlight the single Precision isometric that passes exactly
    # through the final rule's own coverage -- every other point on
    # this one line is, by definition, exactly as precise as the rule
    # itself. mark_point=False since plot_rule_refinement already
    # scattered the final point.
    Precision().plot_isometric_through_rule(
        rule, rep, space=space, mark_point=False, colors="black", linewidths=1.2,
    )
    space.ax.plot([], [], color="black", label="Precision isometric through final rule")
    space.ax.legend(loc="lower right", fontsize=8)  # refresh legend with the label just added

    out_path = os.path.join(OUT_DIR, out_name)
    space.ax.figure.savefig(out_path, dpi=110)
    plt.close(space.ax.figure)
    print(f"Saved {out_path}")


# a small space, not the default 100/160: Laplace's pivot shift to
# (-1, -1) is a real fraction of a space this size, invisible at the
# default scale (see module docstring)
PRECISION_FAMILY_N_POS = 10
PRECISION_FAMILY_N_NEG = 16
PRECISION_FAMILY = [
    ("Precision", Precision(), "tab:blue"),
    ("Laplace", Laplace(), "tab:orange"),
    ("MEstimate(2)", MEstimate(2), "tab:green"),
    ("MEstimate(20)", MEstimate(20), "tab:red"),
]


def plot_precision_family_comparison(normalized: bool, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    space = CoverageSpace(
        n_pos=PRECISION_FAMILY_N_POS, n_neg=PRECISION_FAMILY_N_NEG, normalized=normalized, ax=ax,
        title=f"Precision/Laplace/MEstimate pivots ({PRECISION_FAMILY_N_POS}/{PRECISION_FAMILY_N_NEG})",
    )
    for name, h, color in PRECISION_FAMILY:
        h.plot_isometrics(space=space, levels=6, colors=color, linestyles="dashed", linewidths=0.8)
        ax.plot([], [], color=color, linestyle="dashed", label=name)
    ax.legend(fontsize=8, loc="lower right")

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    print(f"Plotting isometrics for {len(HEURISTICS)} heuristics "
          "(LengthPenalized skipped -- see module docstring).")
    plot_heuristic_grid(normalized=False, out_name="demo_heuristics_isometrics_raw.png")
    plot_heuristic_grid(normalized=True, out_name="demo_heuristics_isometrics_roc.png")
    plot_refinement_with_isometrics(normalized=False, out_name="demo_heuristics_refinement_raw.png")
    plot_refinement_with_isometrics(normalized=True, out_name="demo_heuristics_refinement_roc.png")
    plot_precision_family_comparison(normalized=False, out_name="demo_heuristics_precision_family_raw.png")
    plot_precision_family_comparison(normalized=True, out_name="demo_heuristics_precision_family_roc.png")


if __name__ == "__main__":
    main()
