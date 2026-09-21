"""
Random-forest -> RuleSet import + combiner-comparison demo across ten
UCI benchmarks (a mix of small/mostly-categorical and larger/mixed
numeric+categorical datasets -- the largest, ``adult`` at ~49k rows, is
still well short of ``covtype``'s ~581k, kept out deliberately to avoid
excessive runtime).

For each dataset: 5-fold cross-validation where, per fold, a
RandomForestClassifier is fit on the training split and converted into
one combined RuleSet (RandomForestImporter -- every tree's leaves
combined into one flat, always-exhaustive RuleSet, not a sequence of
per-tree DisjointRuleSets; see its docstring). Several ways of combining
that RuleSet's predictions are then compared against the forest's own
.predict() on the held-out test split -- see pyrulearn.combiners for the
full taxonomy:

- the forest's own prediction (soft/probability-averaged voting across
  trees -- the sklearn baseline, not itself a pyrulearn combiner)
- CountVoteCombiner ("vote") -- plain hard majority vote across the
  k = n_estimators rules that always cover every example
- HeuristicMaxCombiner(heuristic) / HeuristicVoteCombiner(heuristic), for
  two different pyrulearn.heuristics.RuleHeuristics (Precision, Laplace)
  -- `RandomForestImporter.import_model` is given `data=train_rep`, so
  every leaf carries real measured stats (its own `ConfusionMatrix`) and
  each combiner scores directly from those, at predict time -- no
  annotate_weights step
- DistributionCombiner's four variants (Micro/Macro x Vote/Max), reading
  each leaf's own measured per-class distribution directly. MacroVoteCombiner
  in particular is expected to land closest to the forest's own accuracy,
  since it's the one that actually matches sklearn's mechanism (normalize
  each tree's leaf counts to a probability vector, then average across
  trees) -- this demo exists largely to check that expectation against
  real data.

None of the rule-based combiners (other than MacroVoteCombiner) are
expected to reproduce the forest's own prediction exactly: hard-voting
and weight-based combiners collapse each leaf to a single vote/scalar,
discarding the full per-class probability information sklearn's own
averaging uses. The comparison here is about how *close* each combiner
gets (both its own accuracy and its agreement with the forest), not
exact reproduction -- contrast with demo_decision_tree_import.py, where
a single tree and its extracted DisjointRuleSet agree exactly by
construction.

Missing-value handling, dataset loading, and DataSpec construction are
the same deliberately-simple, demo-only approach as
demo_decision_tree_import.py (see its module docstring for the
rationale) -- duplicated here rather than imported, so each demo stays
a self-contained, directly-runnable script.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

from pyrulearn import HeuristicMaxCombiner, HeuristicVoteCombiner
from pyrulearn.data.io import binarize, build_dataspec, validate_dataspec
from pyrulearn.heuristics import Laplace, Precision
from pyrulearn.interfaces.sklearn import RandomForestImporter
from pyrulearn.data import BooleanDataRepresentation

N_FOLDS = 5
RANDOM_STATE = 0
MAX_INTERVALS = 6  # numeric-discretization bucket cap, see DataSpecBuilder.add_numeric
N_ESTIMATORS = 10
MAX_DEPTH = 4  # shallower than the single-tree demo -- kept small since rule count scales with n_estimators too

# a mix of small/mostly-categorical (vote, breast-cancer, soybean) and
# larger/mixed numeric+categorical (credit-g, cmc, hypothyroid, adult)
# datasets -- adult (~49k rows) is the "bigger" end deliberately kept
# well short of covtype-scale to avoid excessive runtime
STANDARD_DATASETS = [
    "vote", "breast-cancer", "colic", "credit-approval", "soybean",
    "anneal", "credit-g", "cmc", "hypothyroid", "adult",
]
REPORT_PATH = os.path.join(os.path.dirname(__file__), "demo_random_forest_combiners_report.md")

HEURISTICS = {"precision": Precision(), "laplace": Laplace()}
DISTRIBUTION_METHODS = ["micro_vote", "macro_vote", "micro_max", "macro_max"]
METHODS = ["random_forest", "vote", *DISTRIBUTION_METHODS,
           "max_precision", "vote_precision", "max_laplace", "vote_laplace"]
METHOD_LABELS = {
    "random_forest": "RF (sklearn)",
    "vote": "vote (unweighted)",
    "micro_vote": "micro-vote (pooled counts)",
    "macro_vote": "macro-vote (~sklearn)",
    "micro_max": "micro-max",
    "macro_max": "macro-max",
    "max_precision": "max (precision)", "vote_precision": "vote (precision-weighted)",
    "max_laplace": "max (laplace)", "vote_laplace": "vote (laplace-weighted)",
}


def _fill_missing(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Fill missing values so the dataset survives with minimal row/column
    dropping: "?" for nominal columns, the column median for numeric ones;
    columns with no signal at all (entirely missing, or constant after
    filling) are dropped instead -- see demo_decision_tree_import.py's
    version of this helper for the full rationale.
    """
    df = df.copy()
    degenerate = [
        col for col in df.columns
        if col != target_col and (df[col].isna().all() or df[col].nunique(dropna=True) <= 1)
    ]
    if degenerate:
        print(f"  Dropping column(s) with no signal (all-missing or constant): {degenerate}")
        df = df.drop(columns=degenerate)

    for col in df.columns:
        if col == target_col or not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(object)
            df[col] = df[col].where(df[col].notna(), "?")

    still_constant = [col for col in df.columns if col != target_col and df[col].nunique(dropna=True) <= 1]
    if still_constant:
        print(f"  Dropping column(s) that became constant after filling: {still_constant}")
        df = df.drop(columns=still_constant)
    return df


def load_openml(name: str, version=1):
    d = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    target_col = d.target.name
    df = _fill_missing(d.frame, target_col)
    return df, target_col


def dataspec_from_fold(train_df: pd.DataFrame, target_col: str) -> BooleanDataRepresentation:
    """Build a DataSpec (typed attributes + thresholds) from a *training*
    fold only, then binarize into a BooleanDataRepresentation over that
    same fold's data."""
    arff_types = {
        col: "numeric" if pd.api.types.is_numeric_dtype(train_df[col]) else "nominal"
        for col in train_df.columns if col != target_col
    }
    builder = build_dataspec(train_df, target=target_col, arff_types=arff_types, max_intervals=MAX_INTERVALS)
    ds = builder.build()
    X = binarize(ds, train_df)
    y = train_df[target_col].to_numpy()
    return BooleanDataRepresentation(ds, X, y)


def score_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, capture_detail: bool):
    train_rep = dataspec_from_fold(train_df, target_col)
    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    clf.fit(train_rep.X, train_rep.y)

    rules = RandomForestImporter().import_model(clf, train_rep.spec, data=train_rep)

    validate_dataspec(train_rep.spec, test_df)  # (problems, if any, are already exercised/asserted by the tree demo)
    test_X = binarize(train_rep.spec, test_df)
    test_y = test_df[target_col].to_numpy()
    test_rep = BooleanDataRepresentation(train_rep.spec, test_X, test_y)

    rf_preds = clf.predict(test_rep.X)
    accs = {"random_forest": float(np.mean(rf_preds == test_y))}
    agreements = {"random_forest": 1.0}  # trivially agrees with itself

    # unweighted hard vote + the four DistributionCombiner variants --
    # all read straight from each leaf's own measured stats/rule.target
    for m in ["vote", *DISTRIBUTION_METHODS]:
        preds = rules.predict(test_rep, combiner=m)
        accs[m] = float(np.mean(preds == test_y))
        agreements[m] = float(np.mean(preds == rf_preds))

    for hname, heuristic in HEURISTICS.items():
        # HeuristicMaxCombiner/HeuristicVoteCombiner score each leaf from
        # its own measured stats directly -- no annotate_weights step
        # needed now that import_model was given data= above.
        max_preds = rules.predict(test_rep, combiner=HeuristicMaxCombiner(heuristic))
        accs[f"max_{hname}"] = float(np.mean(max_preds == test_y))
        agreements[f"max_{hname}"] = float(np.mean(max_preds == rf_preds))

        vote_w_preds = rules.predict(test_rep, combiner=HeuristicVoteCombiner(heuristic))
        accs[f"vote_{hname}"] = float(np.mean(vote_w_preds == test_y))
        agreements[f"vote_{hname}"] = float(np.mean(vote_w_preds == rf_preds))

    detail_md = []
    if capture_detail:
        detail_md = _build_detail_lines(rules, train_rep)

    return accs, agreements, len(rules), detail_md


def _build_detail_lines(rules, train_rep) -> list:
    """Fold-1 illustration: the 5 highest- and lowest-(Laplace-)scored
    rules (from each leaf's own measured stats), each shown with its
    per-class distribution too -- what "micro" (raw counts) vs "macro"
    (normalized proportions) actually operate on.
    """
    laplace = Laplace()
    score = lambda r: laplace.score(r.stats().confusion.rule_stats(r.target))
    ranked = sorted(rules.rules, key=score, reverse=True)
    lines = [f"Combined RuleSet: {len(rules)} rules from {N_ESTIMATORS} trees "
             f"(sorted here by each leaf's own measured Laplace score):\n\n"]
    counts = lambda r: r.stats().confusion.predicted_as(r.target)
    lines.append("Top 5:\n")
    for r in ranked[:5]:
        lines.append(f"  [laplace={score(r):.3f}] counts={counts(r)} {r.to_string('logic')}\n")
    lines.append("\nBottom 5:\n")
    for r in ranked[-5:]:
        lines.append(f"  [laplace={score(r):.3f}] counts={counts(r)} {r.to_string('logic')}\n")
    return ["### Fold 1 detail\n\n", "```text\n", *lines, "```\n\n"]


def run_dataset(name: str, df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    y = df[target_col].to_numpy()

    print(f"\n{'=' * 78}\n{name}  (n={len(df)}, attributes={len(feature_cols)}, "
          f"classes={df[target_col].nunique()})\n{'=' * 78}")

    md = [f"## {name}\n\n", f"n={len(df)}, attributes={len(feature_cols)}, "
          f"classes={df[target_col].nunique()}\n\n"]

    try:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df, y))
    except ValueError as e:
        print(f"  StratifiedKFold unavailable ({e}); falling back to plain KFold")
        md.append(f"StratifiedKFold unavailable ({e}); fell back to plain KFold.\n\n")
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df))

    all_accs = {m: [] for m in METHODS}
    all_agreements = {m: [] for m in METHODS}
    n_rules_list = []
    t0 = time.time()
    for i, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        accs, agreements, n_rules, detail_md = score_fold(train_df, test_df, target_col, capture_detail=(i == 0))
        for m in METHODS:
            all_accs[m].append(accs[m])
            all_agreements[m].append(agreements[m])
        n_rules_list.append(n_rules)
        md.extend(detail_md)
        print(f"  fold {i + 1} ({time.time() - t0:.1f}s so far): " +
              "  ".join(f"{m}={accs[m]:.3f}" for m in METHODS) + f"  n_rules={n_rules}")
    dt = time.time() - t0

    print(f"\n  Summary over {len(folds)} folds ({dt:.1f}s), mean rules/fold {np.mean(n_rules_list):.1f}:")
    for m in METHODS:
        print(f"    {METHOD_LABELS[m]:<28} acc={np.mean(all_accs[m]):.3f} +/- {np.std(all_accs[m]):.3f}"
              f"   agreement-with-RF={np.mean(all_agreements[m]):.3f}")

    md.append("### Method comparison (mean +/- std over folds)\n\n")
    md.append("| method | accuracy | agreement with RF |\n")
    md.append("|---|---|---|\n")
    for m in METHODS:
        md.append(f"| {METHOD_LABELS[m]} | {np.mean(all_accs[m]):.3f} +/- {np.std(all_accs[m]):.3f} "
                   f"| {np.mean(all_agreements[m]):.3f} |\n")
    md.append(f"\nmean rules/fold: {np.mean(n_rules_list):.1f} ({dt:.1f}s total)\n\n---\n\n")

    summary = {"name": name, "n": len(df), "mean_rules": np.mean(n_rules_list), "time": dt}
    for m in METHODS:
        summary[f"{m}_acc"] = np.mean(all_accs[m])
        summary[f"{m}_agree"] = np.mean(all_agreements[m])
    return summary, md


def main():
    results = []
    report = [
        "# Random forest -> RuleSet import + combiner comparison demo\n\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}. "
        f"N_FOLDS={N_FOLDS}, N_ESTIMATORS={N_ESTIMATORS}, MAX_DEPTH={MAX_DEPTH}, "
        f"MAX_INTERVALS={MAX_INTERVALS}.\n\n"
        "Only fold 1 of each dataset shows the top/bottom-5-by-weight rule "
        "detail; all folds contribute to the summary statistics. See the "
        "module docstring for what each method means, and why "
        "`macro-vote (~sklearn)` in particular is expected to land closest "
        "to the forest's own accuracy.\n\n---\n\n",
    ]

    for name in STANDARD_DATASETS:
        df, target_col = load_openml(name)
        summary, md = run_dataset(name, df, target_col)
        results.append(summary)
        report.extend(md)

    print(f"\n{'=' * 78}\nOverall summary (mean accuracy per method)\n{'=' * 78}")
    header = f"{'dataset':<16}" + "".join(f"{m:>16}" for m in METHODS)
    print(header)
    report.append("## Overall summary\n\n")
    report.append("Mean accuracy per method, across all folds of each dataset.\n\n")
    report.append("| dataset | " + " | ".join(METHOD_LABELS[m] for m in METHODS) + " | mean rules/fold |\n")
    report.append("|---" * (len(METHODS) + 2) + "|\n")
    for r in results:
        line = f"{r['name']:<16}" + "".join(f"{r[f'{m}_acc']:>16.3f}" for m in METHODS)
        print(line)
        report.append(
            f"| {r['name']} | " + " | ".join(f"{r[f'{m}_acc']:.3f}" for m in METHODS)
            + f" | {r['mean_rules']:.1f} |\n"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
