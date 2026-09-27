"""
Demo: separate-and-conquer covering, step by step, on Titanic.

Illustrates `PFossil` (Fuernkranz, 1994) -- `Correlation` as the growth
heuristic, greedy hill climbing to a local optimum, a 0.3 correlation
threshold gating which rules are kept -- by *replaying* its covering loop
and single-rule search by hand, using the same primitives `HillClimbing`/
`SeCo` use internally (`Rule.specialize`, `Correlation`, `RuleStats`,
`ThresholdPrePruning`), rather than treating `PFossil(...).fit(data)` as an
opaque call. The replay is checked against the real `PFossil.fit()` output
before any plotting happens -- see `_covering_replay_matches_pfossil`.

Output, per hill-climbing step, is **three separate static PNGs** (not one
combined figure -- meant to be stepped through by hand, e.g. one triptych
per PowerPoint click) plus a companion `demo_covering.md` narrating every
step in Prolog notation (`Rule.to_string("prolog")`) with the images
embedded inline:

1. **Search view** (``*_search.png``) -- every one-condition
   specialization of the current partial rule, scored by `Correlation`
   within whatever's currently *remaining* (shrinks between rules, fixed
   within one rule's own growth); the *entire* growth path so far (not
   just this round's hop), rendered via the same
   `CoverageSpace.plot_rule_refinement`/`rule_refinement_path` machinery
   `demo_heuristics.py` uses for exactly this purpose; and the
   `Correlation = 0.3` isometric overlaid, thin and dashed like the
   random-guess diagonal but in a distinct, more visible color, so the
   acceptance cutoff reads clearly against the (differently colored,
   solid-arrow) growth path. Filtering doesn't stop the climb early (a
   rule below the line can still be specialized further), it only decides
   whether the walk's *final* rule is kept once hill climbing reaches its
   own local optimum.
2. **Population view** (``*_population.png``) -- a fixed 2-D t-SNE embedding
   of every training example (computed once, coordinates never change),
   colored and shaped by class, with rows covered by the step's *current*
   partial rule drawn vividly and everything else greyed out.
3. **Progress view** (``*_progress.png``) -- the cumulative coverage path
   across *completed* rules only, fixed to the full training set's own
   (N, P) -- never rescaled -- one leg ``(0,0) -> rule 1 -> rule 2 -> ...``
   per finished rule, the same multi-hop path style
   `RuleList.coverage_path` already uses.

The covering loop's own final attempt is a *rejection*: hill climbing
reaches a local optimum below the 0.3 threshold, so `PFossil` stops without
a 4th rule. That attempt gets search/population frames too (nothing new for
the progress view) -- seeing a real rejection against the isometric is
arguably the more instructive case, not just the three acceptances.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.evaluation import CoverageSpace, rule_refinement_path
from pyrulearn.heuristics import Correlation, RuleStats
from pyrulearn.learners.seco import PFossil
from pyrulearn.models import DecisionList, annotate_rules
from pyrulearn.pruning import ThresholdPrePruning
from pyrulearn.rule import Rule

OUT_DIR = os.path.dirname(__file__)
FRAMES_DIR = os.path.join(OUT_DIR, "demo_covering_frames")
MD_PATH = os.path.join(OUT_DIR, "demo_covering.md")
TARGET_CLASS = "survived"
CORR_THRESHOLD = 0.3
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET_COL = "survived"
MAX_INTERVALS = 6
# sex/embarked are 2-/3-valued nominals -- explicit negation features
# (the project-wide default) buy nothing for "sex" ("!= male" is just a
# second name for "= female", so every refinement would show up twice in
# the search with identical stats) and aren't needed for this demo's
# "embarked" either; numeric columns keep negation (the "<" thresholds
# most of PFossil's own rules are actually built from)
NOMINAL_FEATURES = ["sex", "embarked"]


# ------------------------------------------------------------------ data ---

def load_titanic():
    d = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto")
    df = d.frame[FEATURES + [TARGET_COL]].copy()
    df[TARGET_COL] = df[TARGET_COL].map({"0": "died", "1": "survived"})
    train_df, _test_df = train_test_split(df, test_size=0.3, random_state=0, stratify=df[TARGET_COL])
    train_df = train_df.reset_index(drop=True)
    arff_types = {c: ("nominal" if c in NOMINAL_FEATURES else "numeric") for c in FEATURES}
    negation_overrides = {c: False for c in NOMINAL_FEATURES}
    ds = build_dataspec(train_df, target=TARGET_COL, arff_types=arff_types, max_intervals=MAX_INTERVALS,
                        negation_overrides=negation_overrides).build()
    train_rep = BooleanDataRepresentation(ds, binarize(ds, train_df), train_df[TARGET_COL].to_numpy())
    return train_df, train_rep


def population_embedding(train_df: pd.DataFrame) -> np.ndarray:
    """A fixed 2-D layout of every training row, for the population view --
    entirely separate from the Boolean feature space PFossil itself learns
    on: one-hot categoricals, median-impute missing numerics, standardize,
    then t-SNE(2). Computed once; every frame reuses the same coordinates.

    t-SNE over PCA: compared side by side on this data, PCA's first two
    components come out as one diffuse blob with no visible class
    structure; t-SNE's nonlinear embedding forms visually distinct
    regions that often skew toward one class, which is what this panel
    needs to convey. (UMAP forms even tighter clusters, but those group
    by feature-combination identity, not by class, so they don't read
    any better here -- and it would add a new dependency this project
    otherwise doesn't need.)
    """
    raw = train_df[FEATURES].copy()
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(raw[c])]
    categorical = [c for c in FEATURES if c not in numeric]
    num = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(raw[numeric]), columns=numeric)
    cat = pd.get_dummies(raw[categorical].astype(str), drop_first=True)
    X = pd.concat([num, cat.reset_index(drop=True)], axis=1).to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    return TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Xs)


# --------------------------------------------------------- covering replay ---

def hill_climb_with_history(data, remaining_mask, target_class, heuristic, criterion):
    """One rule's full growth, replaying `HillClimbing`'s default settings
    (`stop_at_local_optimum=True`) exactly: greedy ascent to a local
    optimum, `filtering` tracked as a fallback-eligible rule, never an
    early stop. Returns `(accepted_rule_or_None, steps)`; each step records
    the rule the round started from, every candidate child this round
    (rule, its `RuleStats` within `remaining_mask`, its heuristic score),
    and which was chosen."""
    spec = data.spec
    current = Rule([], target=target_class, dataspec=spec, n_features=spec.n_features)
    current_mask = frozenset(range(spec.n_features))
    current_stats = RuleStats.from_rule(current, data, positive_class=target_class, example_mask=remaining_mask)
    current_score = heuristic.score(current_stats)

    best_eligible = current if criterion.accept(current, current_stats, data, target_class, remaining_mask) else None
    steps = []
    while True:
        children = current.specialize(spec, current_mask)
        if not children:
            break
        scored = []
        for child, child_mask in children:
            stats = RuleStats.from_rule(child, data, positive_class=target_class, example_mask=remaining_mask)
            scored.append((child, child_mask, stats, heuristic.score(stats)))
        best_child, best_mask, best_stats, best_score = max(scored, key=lambda t: t[3])

        steps.append({
            "parent": current, "parent_stats": current_stats, "parent_score": current_score,
            "candidates": [(c, s, sc) for c, _, s, sc in scored],
            "chosen": best_child,
        })
        if best_score <= current_score:
            break  # local optimum -- current is the walk's endpoint
        current, current_mask, current_stats, current_score = best_child, best_mask, best_stats, best_score
        if criterion.accept(current, current_stats, data, target_class, remaining_mask):
            best_eligible = current
    return best_eligible, steps


def covering_replay(data, target_class, heuristic, criterion):
    """`SeCo`'s own covering loop, PFossil's defaults (no `stop_covering`,
    no `max_rules`): grow one rule at a time on the remaining rows, stop
    once the single-rule search itself returns `None`. Returns the
    accepted rules and, for every attempt (including the final, rejected
    one), its steps and the `remaining` mask it searched against."""
    remaining = np.ones(data.n_samples, dtype=bool)
    rules, attempts = [], []
    while True:
        rule, steps = hill_climb_with_history(data, remaining, target_class, heuristic, criterion)
        attempts.append({"rule": rule, "steps": steps, "remaining": remaining.copy()})
        if rule is None:
            break
        rules.append(rule)
        remaining = remaining & ~np.asarray(rule.covers_data(data))
        if not (remaining & (data.y == target_class)).any():
            break
    return rules, attempts


def _covering_replay_matches_pfossil(data, target_class, rules) -> bool:
    """Sanity check, run once before any plotting: the replay's accepted
    rules must be exactly what `PFossil.fit()` itself produces -- same
    conditions, same targets, same order -- or the frames would be
    illustrating a subtly different algorithm."""
    ref = PFossil(target_class=target_class, random_state=0).fit(data)
    got = [(tuple(sorted(l.feature for l in r.conditions)), r.target) for r in rules]
    want = [(tuple(sorted(l.feature for l in r.conditions)), r.target) for r in ref.rules]
    return got == want


# --------------------------------------------------------------- plotting ---

def _label_growth_path(ax, path, conds, spec):
    """Label each point of a rule's growth path from a vertical column
    stacked in the plot's upper-left corner -- few negatives *and* many
    positives is the hardest combination for any rule to reach, so that
    corner is reliably empty -- each connected back to its actual point
    by a thin leader line. `plot_rule_refinement`'s own built-in
    `annotate=True` places each label right next to its own point
    instead, which overlaps badly once several refinements in a row
    barely move the point (common once a rule already has a few
    conditions and is only adding near-redundant ones)."""
    n = len(path)
    labels = ["∅"] + [Rule([c], dataspec=spec).to_string("conditions") for c in conds]
    # top-to-bottom column order follows each point's own height (tp), so
    # leader lines fan out from the column roughly in the same order they'd
    # naturally appear in, instead of crossing each other more than necessary
    order = sorted(range(n), key=lambda k: -path[k][1])
    y_slots = np.linspace(0.95, 0.20, n) if n > 1 else [0.9]
    for y_slot, k in zip(y_slots, order):
        ax.annotate(
            labels[k], xy=(path[k][0], path[k][1]), xycoords="data",
            xytext=(0.02, y_slot), textcoords="axes fraction",
            ha="left", va="center", fontsize=7,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6, shrinkA=2, shrinkB=4),
        )


def plot_search_view(out_path, rule_idx, step_idx, n_steps, data, remaining_mask, target_class, step):
    """Panel 1: this round's candidates, the rule's entire growth path so
    far (`CoverageSpace.plot_rule_refinement`, the same routine
    `demo_heuristics.py` uses for a rule's specialization path), and the
    0.3 correlation cutoff as a thin dashed isometric."""
    n_pos_remaining = int(np.sum(remaining_mask & (data.y == target_class)))
    n_neg_remaining = int(np.sum(remaining_mask & (data.y != target_class)))
    remaining_data = data.select_rows(remaining_mask)  # rule_refinement_path takes no mask

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    space = CoverageSpace(
        n_pos=n_pos_remaining, n_neg=n_neg_remaining, ax=ax, aspect="auto",
        title=f"Rule {rule_idx + 1}, condition {step_idx + 1}/{n_steps}: candidate refinements",
    )
    # thin dashed isometric, styled like the random-guess diagonal but a
    # tad more visible, and in a color distinct from the growth path
    # (plot_rule_refinement below draws that in "C1"/orange)
    Correlation().plot_isometrics(space=space, levels=[CORR_THRESHOLD], colors="crimson",
                                  linestyles="dashed", linewidths=1.0)
    ax.plot([], [], color="crimson", linestyle="dashed", label=f"Correlation = {CORR_THRESHOLD} (accept threshold)")

    cand_pts = np.array([[s.fp, s.tp] for _, s, _ in step["candidates"]])
    space.scatter(cand_pts, color="tab:gray", s=30, zorder=2, label="candidate refinements this round")

    # the *entire* growth path so far (not just this round's hop) --
    # ordered=True so plot_rule_refinement walks literals in the order
    # hill climbing actually added them, not canonical feature order
    chosen = step["chosen"]
    growth_rule = Rule(chosen.conditions, target=target_class, dataspec=data.spec,
                       n_features=data.spec.n_features, ordered=True)
    path = rule_refinement_path(growth_rule, remaining_data, positive_class=target_class)
    space.plot_rule_refinement(growth_rule, remaining_data, positive_class=target_class,
                               annotate=False, path=path)
    _label_growth_path(ax, path, growth_rule.conditions, remaining_data.spec)
    ax.plot([], [], color="C1", label="rule's growth path so far")
    ax.legend(loc="lower right", fontsize=8)  # refresh with the proxy labels just added

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_population_view(out_path, rule_idx, step_idx, n_steps, data, embedding, remaining_mask, target_class, step):
    """Panel 2: fixed t-SNE embedding, rows covered by the current partial
    rule shown vividly, other still-*remaining* rows greyed out. Rows
    already claimed by an earlier completed rule are omitted entirely --
    not just greyed, not drawn at all -- since they're no longer part of
    this rule's search space; the embedding's own coordinates stay
    exactly as computed for the full training set (recomputing t-SNE on
    a shrunk dataset would move every remaining point too, which would
    make the panels incomparable across steps)."""
    chosen = step["chosen"]
    covered = np.asarray(chosen.covers_data(data)) & remaining_mask
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for cls, marker in (("died", "^"), ("survived", "o")):
        cls_mask = data.y == cls
        vivid = cls_mask & covered
        dim = cls_mask & remaining_mask & ~covered
        color = "tab:orange" if cls == target_class else "tab:blue"
        ax.scatter(*embedding[dim].T, c="lightgray", marker=marker, s=15, alpha=0.5, zorder=1)
        ax.scatter(*embedding[vivid].T, c=color, marker=marker, s=30, zorder=2, label=f"class {cls}")
    chosen_text = chosen.to_string("conditions")
    ax.set_title(f"Rule {rule_idx + 1}, condition {step_idx + 1}/{n_steps}: covered by '{chosen_text}'\n"
                f"({int(covered.sum())} of {int(remaining_mask.sum())} still-remaining rows)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_population_overview(out_path, data, embedding, target_class, mask=None, title=None):
    """Every row in `mask` (default: all rows), shown vividly in the same
    fixed t-SNE embedding -- nothing greyed out, since this isn't about
    one rule's coverage. Used twice: once for the whole training set (the
    introduction's orientation plot, before any rule has narrowed
    anything), and once for whatever's left *uncovered* once the covering
    loop stops (an illustration of the default rule's own territory --
    what a decision list falls back to when no rule fires)."""
    if mask is None:
        mask = np.ones(data.n_samples, dtype=bool)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for cls, marker in (("died", "^"), ("survived", "o")):
        cls_mask = mask & (data.y == cls)
        color = "tab:orange" if cls == target_class else "tab:blue"
        ax.scatter(*embedding[cls_mask].T, c=color, marker=marker, s=20, zorder=2, label=f"class {cls}")
    ax.set_title(title or f"Full training set ({int(mask.sum())} rows), t-SNE embedding")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_progress_view(out_path, progress_points, n_pos_total, n_neg_total):
    """Panel 3: the cumulative coverage path over completed rules only,
    fixed to the full training set's own (N, P) -- (0,0) -> rule 1 ->
    rule 2 -> ... -- one leg per finished rule, never rescaled."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    space = CoverageSpace(n_pos=n_pos_total, n_neg=n_neg_total, ax=ax, aspect="auto",
                          title="Progress: completed rules so far")
    pts = np.array(progress_points)
    space.arrow_path(pts, color="tab:green", zorder=3)
    for i, (nx, ny) in enumerate(progress_points[1:], start=1):
        ax.annotate(f"rule {i}", (nx, ny), fontsize=8, xytext=(4, 4), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------- narration ---

def _empty_rule_prolog(target_class) -> str:
    return f"{target_class}(X) :- true."


def render_markdown(data, attempts, rules, heuristic, criterion, target_class, n_pos_total, n_neg_total,
                    n_pos_final, n_neg_final) -> str:
    """The written walkthrough: every candidate considered, which one was
    chosen and why, and whether each finished rule passed the acceptance
    threshold -- in Prolog notation (`Rule.to_string("prolog")`), so a
    reader can check every number against the rule as printed. The PNGs
    illustrate this; they carry no numbers of their own."""
    lines = [
        "# Covering demo: PFossil on Titanic",
        "",
        "Heuristic: `Correlation`. Search: greedy hill climbing to a local optimum "
        "(`HillClimbing`, `stop_at_local_optimum=True`). Acceptance filter: "
        f"`Correlation >= {CORR_THRESHOLD}` (FOSSIL's own published threshold), checked once "
        "hill climbing stops -- it never cuts a climb short, it only decides whether the "
        "climb's own final rule is kept.",
        "",
        f"Training set: {n_pos_total} positives (`{target_class}`) / "
        f"{n_neg_total} negatives (`died`), {len(FEATURES)} features (`{', '.join(FEATURES)}`).",
        "",
        f"![full training set]({os.path.basename(FRAMES_DIR)}/overview_population.png)",
        "",
        "---",
    ]

    accepted_so_far: list = []
    for rule_idx, attempt in enumerate(attempts):
        remaining_mask = attempt["remaining"]
        n_pos_r = int(np.sum(remaining_mask & (data.y == target_class)))
        n_neg_r = int(np.sum(remaining_mask & (data.y != target_class)))
        frame_prefix = f"rule{rule_idx + 1}"
        header = f"## Rule {rule_idx + 1}" if attempt["rule"] is not None else "## Rejected attempt"
        lines += [
            "",
            header,
            "",
            f"Remaining rows entering this attempt: {n_pos_r} positives / {n_neg_r} negatives.",
        ]
        steps = attempt["steps"]
        for step_idx, step in enumerate(steps):
            parent, chosen = step["parent"], step["chosen"]
            parent_pl = parent.to_string("prolog") if parent.conditions else _empty_rule_prolog(target_class)
            frame_id = f"{frame_prefix}_condition{step_idx + 1}"
            lines += [
                "",
                f"### Condition {step_idx + 1}",
                "",
                f"Growing from: `{parent_pl}` (score={step['parent_score']:.3f})",
                "",
                f"Top candidates by Correlation ({len(step['candidates'])} considered in total):",
                "",
                "| refinement (Prolog) | tp | fp | Correlation |",
                "|---|---|---|---|",
            ]
            # the chosen child is always the argmax, so it's always rank 0
            # here -- top 8 always includes it, no special-casing needed
            ranked = sorted(step["candidates"], key=lambda t: -t[2])
            for cand, stats, score in ranked[:8]:
                mark = " **<- chosen**" if cand is chosen else ""
                lines.append(f"| `{cand.to_string('prolog')}` | {stats.tp} | {stats.fp} | {score:.3f}{mark} |")
            chosen_stats = next(s for r, s, _ in step["candidates"] if r is chosen)
            chosen_score = next(sc for r, _, sc in step["candidates"] if r is chosen)
            passes = criterion.accept(chosen, chosen_stats, data, target_class, remaining_mask)
            lines += [
                "",
                f"Chosen: `{chosen.to_string('prolog')}` (tp={chosen_stats.tp}, fp={chosen_stats.fp}, "
                f"Correlation={chosen_score:.3f}) -- "
                f"{'passes' if passes else 'does not (yet) pass'} the {CORR_THRESHOLD} threshold.",
                "",
                f"![search view]({os.path.basename(FRAMES_DIR)}/{frame_id}_search.png)",
                f"![population view]({os.path.basename(FRAMES_DIR)}/{frame_id}_population.png)",
            ]
        if attempt["rule"] is not None:
            accepted_so_far.append(attempt["rule"])
            # the model so far, as an actual DecisionList, its rules annotated
            # on the training data -- to_string prints each rule's stored
            # (tp/fp), its raw coverage of that data
            model_so_far = DecisionList(annotate_rules(accepted_so_far, data))
            lines += [
                "", f"**Rule {rule_idx + 1} accepted:** `{attempt['rule'].to_string('prolog')}`", "",
                f"Model so far ({len(accepted_so_far)} rule{'s' if len(accepted_so_far) != 1 else ''}):", "",
                "```prolog", model_so_far.to_string(), "```",
            ]
        else:
            lines += [
                "", "**No rule accepted from this attempt.** Hill climbing reached a local optimum whose "
                f"Correlation never reaches {CORR_THRESHOLD}; the covering loop stops here rather than "
                "keeping a rule that fails the filter.",
            ]
        lines += ["", f"![progress view]({os.path.basename(FRAMES_DIR)}/{frame_prefix}_condition{len(steps)}_progress.png)"]

    lines += ["", "---", "", "## Final rule set", ""]
    for r in rules:
        lines.append(f"- `{r.to_string('prolog')}`")

    default_class = target_class if n_pos_final > n_neg_final else "died"
    lines += [
        "", "---", "", "## Default rule",
        "",
        f"{n_pos_final} positives / {n_neg_final} negatives are covered by none of the {len(rules)} rules "
        f"above -- the majority class among them, `{default_class}`, is what a decision list built from "
        "this ruleset would fall back to for any of these rows.",
        "",
        f"![uncovered examples]({os.path.basename(FRAMES_DIR)}/final_uncovered_population.png)",
    ]
    return "\n".join(lines) + "\n"


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    train_df, train_rep = load_titanic()
    embedding = population_embedding(train_df)
    heuristic = Correlation()
    criterion = ThresholdPrePruning(heuristic, CORR_THRESHOLD, "<")

    rules, attempts = covering_replay(train_rep, TARGET_CLASS, heuristic, criterion)
    if not _covering_replay_matches_pfossil(train_rep, TARGET_CLASS, rules):
        raise RuntimeError("replay diverged from PFossil.fit() -- do not trust these frames")
    print(f"Replay matches PFossil.fit(): {len(rules)} accepted rules, "
          f"{len(attempts)} covering-loop attempts (last one is a rejection if it has no rule).")

    plot_population_overview(os.path.join(FRAMES_DIR, "overview_population.png"), train_rep, embedding, TARGET_CLASS)
    print("Saved overview_population.png")

    n_pos_total = int(np.sum(train_rep.y == TARGET_CLASS))
    n_neg_total = int(np.sum(train_rep.y != TARGET_CLASS))
    accepted_so_far: list = []
    # DecisionList.coverage_path's own "unique (fired) coverage" logic --
    # first-match-wins under this ordering -- rather than hand-summing each
    # rule's raw, possibly-overlapping coverage
    progress_points = [(0, 0)]
    for rule_idx, attempt in enumerate(attempts):
        remaining_mask = attempt["remaining"]
        steps = attempt["steps"]
        for step_idx, step in enumerate(steps):
            frame_id = f"rule{rule_idx + 1}_condition{step_idx + 1}"
            plot_search_view(os.path.join(FRAMES_DIR, f"{frame_id}_search.png"),
                             rule_idx, step_idx, len(steps), train_rep, remaining_mask, TARGET_CLASS, step)
            plot_population_view(os.path.join(FRAMES_DIR, f"{frame_id}_population.png"),
                                 rule_idx, step_idx, len(steps), train_rep, embedding, remaining_mask,
                                 TARGET_CLASS, step)
            # the *last* step's progress frame is generated after
            # progress_points is updated below, so it can show this
            # attempt's own rule (if accepted) already added -- generating
            # it here too would show a stale, not-yet-completed state
            if step_idx < len(steps) - 1:
                plot_progress_view(os.path.join(FRAMES_DIR, f"{frame_id}_progress.png"),
                                   progress_points, n_pos_total, n_neg_total)
            print(f"Saved {frame_id}_{{search,population,progress}}.png")
        if attempt["rule"] is not None:
            accepted_so_far.append(attempt["rule"])
            progress_points = [tuple(p) for p in DecisionList(accepted_so_far).coverage_path(
                train_rep, TARGET_CLASS)]
        last_frame_id = f"rule{rule_idx + 1}_condition{len(steps)}"
        plot_progress_view(os.path.join(FRAMES_DIR, f"{last_frame_id}_progress.png"),
                           progress_points, n_pos_total, n_neg_total)

    final_remaining = attempts[-1]["remaining"]
    n_pos_final = int(np.sum(final_remaining & (train_rep.y == TARGET_CLASS)))
    n_neg_final = int(np.sum(final_remaining & (train_rep.y != TARGET_CLASS)))
    plot_population_overview(
        os.path.join(FRAMES_DIR, "final_uncovered_population.png"), train_rep, embedding, TARGET_CLASS,
        mask=final_remaining,
        title=f"Uncovered by any rule ({int(final_remaining.sum())} rows): the default rule's territory",
    )
    print("Saved final_uncovered_population.png")

    md = render_markdown(train_rep, attempts, rules, heuristic, criterion, TARGET_CLASS, n_pos_total, n_neg_total,
                         n_pos_final, n_neg_final)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved {MD_PATH}")

    print(f"\n{len(rules)} rules learned (PFossil, correlation_threshold={CORR_THRESHOLD}):")
    for r in rules:
        print(" ", r.to_string("conditions"))
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
