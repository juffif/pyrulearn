"""
Elaborate decision-tree-to-rule-set demo across standard UCI benchmarks.

For each dataset: 5-fold cross-validation where, per fold, a
DecisionTreeClassifier is fit on the training split, converted into a
DisjointRuleSet + DataSpec (pyrulearn.interfaces.sklearn.from_sklearn_tree),
the held-out test split is read in w.r.t. that same DataSpec, and the
tree's and the rule set's test predictions are compared -- they should
agree exactly, since the rule set is a faithful re-expression of the
tree.

Datasets are fetched from OpenML (network required, cached locally by
scikit-learn after the first run); ``covtype`` uses sklearn's built-in
fetcher and is included as a scale stress test. Console output stays
light (progress + per-fold/per-dataset summary lines); the full detail
(tree text, every rule in all four `to_string` formats, for fold 1 of
each dataset) goes into a Markdown report written next to this script --
see REPORT_PATH below.

Simplifications made deliberately, for this demo only (not something
pyrulearn's core does): missing values are filled -- "?" for nominal
columns (matching how several of these datasets already encode missing
values in their raw UCI form), the column median for numeric columns --
rather than dropped or modeled by pyrulearn itself (missing-value
handling in pyrulearn.data.io.binarize is explicitly out of scope for
now, see its module docstring). Columns with no signal at all (entirely
missing, or constant after filling -- e.g. hypothyroid's TBG is 100%
missing, mushroom's veil-type is a known constant) are dropped instead,
since no encoding can make them usable and pyrulearn.data.io correctly
refuses to invent thresholds for one. Tree depth is capped (deeper for
covtype) to keep the printed trees/rule sets readable and RuleSet.predict
(an O(n_samples x n_rules) Python loop) fast even on covtype's ~116k-row
test folds.
"""

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from pyrulearn.data.io import binarize, build_dataspec, validate_dataspec
from pyrulearn.interfaces.sklearn import from_sklearn_tree
from pyrulearn.data import BooleanDataRepresentation

N_FOLDS = 5
RANDOM_STATE = 0
MAX_INTERVALS = 6  # numeric-discretization bucket cap, see DataSpecBuilder.add_numeric

# tree depth per dataset -- kept small so rule counts stay readable and
# RuleSet.predict's O(n_samples x n_rules) Python loop stays fast; a few
# points deeper for the covtype stress test since it can afford it.
MAX_DEPTH = {"covtype": 8}
DEFAULT_MAX_DEPTH = 5

STANDARD_DATASETS = ["vote", "breast-cancer", "kr-vs-kp", "hypothyroid", "soybean", "mushroom"]
REPORT_PATH = os.path.join(os.path.dirname(__file__), "demo_decision_tree_import_report.md")


def _fill_missing(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Fill missing values so the dataset survives with minimal row/column
    dropping: "?" for nominal columns -- an explicit category, not
    something pyrulearn evaluates specially, matching how several of
    these datasets already spell missing values in their raw UCI form --
    and the column median for numeric columns.

    Columns that carry no signal at all -- entirely missing (a median
    fill can't help; hypothyroid's ``TBG`` is a real example, 100%
    missing) or constant after filling -- are dropped instead of filled,
    since no encoding or discretization can make a signal-free column
    usable, and pyrulearn.data.io correctly refuses to invent thresholds
    for one (see build_dataspec's "no useful split" error).
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


def load_covtype():
    d = fetch_covtype(as_frame=True)
    target_col = d.frame.columns[-1]
    df = _fill_missing(d.frame, target_col)
    return df, target_col


def dataspec_from_fold(train_df: pd.DataFrame, target_col: str) -> BooleanDataRepresentation:
    """Build a DataSpec (typed attributes + thresholds) from a *training*
    fold only, then binarize into a BooleanDataRepresentation over that
    same fold's data -- this is "reading in the tree's own feature
    space", derived without looking at the test fold at all."""
    arff_types = {
        col: "numeric" if pd.api.types.is_numeric_dtype(train_df[col]) else "nominal"
        for col in train_df.columns if col != target_col
    }
    builder = build_dataspec(train_df, target=target_col, arff_types=arff_types, max_intervals=MAX_INTERVALS)
    ds = builder.build()
    X = binarize(ds, train_df)
    y = train_df[target_col].to_numpy()
    return BooleanDataRepresentation(ds, X, y)


def score_fold(dataset_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, capture_detail: bool):
    max_depth = MAX_DEPTH.get(dataset_name, DEFAULT_MAX_DEPTH)

    train_rep = dataspec_from_fold(train_df, target_col)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    clf.fit(train_rep.X, train_rep.y)

    # bind rules straight to the fold's own DataSpec (built with explicit
    # negation features, the default) -- its feature order already matches
    # the tree's training columns 1:1
    rules = from_sklearn_tree(clf, dataspec=train_rep.spec)

    def _original_attr(feature_idx):
        spec = train_rep.spec.feature_spec(feature_idx)
        return spec.attribute if spec.attribute is not None else spec.name

    # number of *original* attributes actually referenced by the extracted
    # rules -- several derived Boolean features (e.g. "TSH>=6.05" and
    # "TSH>=8.55") can map back to the same original attribute ("TSH"),
    # so this de-duplicates by FeatureSpec.attribute, not feature index.
    used_attrs = set()
    for r in rules:
        for lit in r.conditions:
            used_attrs.add(_original_attr(lit.feature))

    # same de-duplication, but computed directly from the tree's own
    # internal split nodes rather than from the extracted rules -- since
    # from_sklearn_tree emits one literal per split with no simplification,
    # this must equal used_attrs exactly; kept separate (and asserted) as
    # an empirical check that the rule extraction is faithful.
    tree_attrs = {_original_attr(f) for f in clf.tree_.feature if f != -2}
    assert tree_attrs == used_attrs, (
        f"attributes used by rules ({used_attrs}) differ from attributes used by tree ({tree_attrs})"
    )

    # "read in the test data w.r.t. the generated DataSpec" -- reusing
    # train_rep.spec directly (not rebuilding one from its feature names)
    # guarantees identical feature indexing between train and test
    problems = validate_dataspec(train_rep.spec, test_df)
    test_X = binarize(train_rep.spec, test_df)
    test_y = test_df[target_col].to_numpy()
    test_rep = BooleanDataRepresentation(train_rep.spec, test_X, test_y)

    tree_preds = clf.predict(test_rep.X)
    rule_preds = rules.predict(test_rep)
    tree_acc = float(np.mean(tree_preds == test_y))
    rule_acc = float(np.mean(rule_preds == test_y))
    agreement = float(np.mean(tree_preds == rule_preds))

    detail_md = []
    if capture_detail:
        body = _build_detail_lines(clf, rules, train_rep, train_df, used_attrs, max_depth, problems)
        print("\n  --- fold 1 detail ---")
        print("".join(body))
        detail_md.append("### Fold 1 detail\n\n```text\n")
        detail_md.extend(body)
        detail_md.append("```\n\n")

    return tree_acc, rule_acc, agreement, len(rules), len(used_attrs), detail_md


# full detail (console + report) is printed for any fold-1 dump up to these
# sizes; beyond that it's auto-truncated with a "N more omitted" note -- not
# gated on dataset name, so it only kicks in for genuinely large trees (in
# practice: covtype), and every other dataset here stays fully verbose.
MAX_TREE_LINES_TO_PRINT = 120
MAX_RULES_TO_PRINT = 30


def _build_detail_lines(clf, rules, train_rep, train_df, used_attrs, max_depth, problems) -> list:
    lines = []
    if problems:
        lines.append(f"validate_dataspec note(s) against the test fold: {problems}\n\n")

    tree_lines = export_text(clf, feature_names=list(train_rep.spec.feature_names), max_depth=max_depth).splitlines()
    lines.append(f"Decision tree ({clf.get_n_leaves()} leaves, depth {clf.get_depth()}):\n")
    shown_tree_lines = tree_lines[:MAX_TREE_LINES_TO_PRINT]
    lines.extend(l + "\n" for l in shown_tree_lines)
    if len(tree_lines) > MAX_TREE_LINES_TO_PRINT:
        lines.append(f"... ({len(tree_lines) - MAX_TREE_LINES_TO_PRINT} more lines omitted for brevity)\n")

    lines.append(
        f"\nExtracted DisjointRuleSet ({len(rules)} rules) -- "
        f"is_disjoint={rules.is_disjoint(train_rep)}, is_exhaustive={rules.is_exhaustive(train_rep)}, "
        f"attributes used: {len(used_attrs)}/{train_df.shape[1] - 1}\n\n"
    )
    shown_rules = rules.rules[:MAX_RULES_TO_PRINT]
    for r in shown_rules:
        lines.append(f"logic:      {r.to_string('logic')}\n")
        lines.append(f"prolog:     {r.to_string('prolog')}\n")
        lines.append(f"pattern:    {r.to_string('pattern')}\n")
        lines.append(f"conditions: {r.to_string('conditions')}\n\n")
    if len(rules) > MAX_RULES_TO_PRINT:
        lines.append(f"... ({len(rules) - MAX_RULES_TO_PRINT} more rules omitted for brevity)\n")
    return lines


def run_dataset(name: str, df: pd.DataFrame, target_col: str):
    feature_cols = [c for c in df.columns if c != target_col]
    n_categorical = sum(1 for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c]))
    n_numeric = sum(1 for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]))
    y = df[target_col].to_numpy()

    print(f"\n{'=' * 78}\n{name}  (n={len(df)}, attributes={len(feature_cols)} "
          f"[{n_categorical} categorical, {n_numeric} numeric], classes={df[target_col].nunique()})\n{'=' * 78}")

    md = [
        f"## {name}\n\n",
        f"n={len(df)}, attributes={len(feature_cols)} "
        f"({n_categorical} categorical, {n_numeric} numeric), classes={df[target_col].nunique()}\n\n",
    ]

    try:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df, y))
    except ValueError as e:
        print(f"  StratifiedKFold unavailable ({e}); falling back to plain KFold")
        md.append(f"StratifiedKFold unavailable ({e}); fell back to plain KFold.\n\n")
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df))

    tree_accs, rule_accs, agreements, n_rules_list, n_attrs_list = [], [], [], [], []
    t0 = time.time()
    for i, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        tree_acc, rule_acc, agreement, n_rules, n_attrs_used, detail_md = score_fold(
            name, train_df, test_df, target_col, capture_detail=(i == 0)
        )
        tree_accs.append(tree_acc)
        rule_accs.append(rule_acc)
        agreements.append(agreement)
        n_rules_list.append(n_rules)
        n_attrs_list.append(n_attrs_used)
        md.extend(detail_md)
        print(f"  fold {i + 1}: tree acc={tree_acc:.3f}  rule-set acc={rule_acc:.3f}  "
              f"tree/rule-set agreement={agreement:.3f}  n_rules={n_rules}  attrs_used={n_attrs_used}")
    dt = time.time() - t0

    summary_lines = [
        f"\n  Summary over {len(folds)} folds ({dt:.1f}s):",
        f"    tree accuracy:      {np.mean(tree_accs):.3f} +/- {np.std(tree_accs):.3f}",
        f"    rule-set accuracy:  {np.mean(rule_accs):.3f} +/- {np.std(rule_accs):.3f}",
        f"    tree/rule-set agreement: {np.mean(agreements):.4f}",
        f"    mean rules/fold: {np.mean(n_rules_list):.1f}",
        f"    mean attributes used/fold: {np.mean(n_attrs_list):.1f} (of {len(feature_cols)} available)",
    ]
    print("\n".join(summary_lines))

    md.append("### Per-fold results\n\n")
    md.append("| fold | tree acc | rule-set acc | agreement | n rules | attrs used |\n")
    md.append("|---|---|---|---|---|---|\n")
    for i in range(len(folds)):
        md.append(f"| {i + 1} | {tree_accs[i]:.3f} | {rule_accs[i]:.3f} | {agreements[i]:.3f} "
                   f"| {n_rules_list[i]} | {n_attrs_list[i]} |\n")
    md.append(
        f"\n**Summary** ({dt:.1f}s): tree accuracy {np.mean(tree_accs):.3f} +/- {np.std(tree_accs):.3f}, "
        f"rule-set accuracy {np.mean(rule_accs):.3f} +/- {np.std(rule_accs):.3f}, "
        f"tree/rule-set agreement {np.mean(agreements):.4f}, "
        f"mean rules/fold {np.mean(n_rules_list):.1f}, "
        f"mean attributes used/fold {np.mean(n_attrs_list):.1f} of {len(feature_cols)} available.\n\n---\n\n"
    )

    summary = {
        "name": name, "n": len(df), "n_categorical": n_categorical, "n_numeric": n_numeric,
        "tree_mean": np.mean(tree_accs), "tree_std": np.std(tree_accs),
        "rule_mean": np.mean(rule_accs), "rule_std": np.std(rule_accs),
        "agreement": np.mean(agreements), "mean_rules": np.mean(n_rules_list),
        "mean_attrs_used": np.mean(n_attrs_list), "time": dt,
    }
    return summary, md


def main(include_covtype: bool = True):
    results = []
    report = [
        "# Decision-tree -> rule-set import demo\n\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}. "
        f"N_FOLDS={N_FOLDS}, MAX_INTERVALS={MAX_INTERVALS}, "
        f"MAX_DEPTH={DEFAULT_MAX_DEPTH} (covtype: {MAX_DEPTH.get('covtype')}).\n\n"
        "Only fold 1 of each dataset is shown in full detail (tree + every rule "
        "in all four `to_string` formats); all folds contribute to the summary "
        "statistics.\n\n---\n\n",
    ]

    for name in STANDARD_DATASETS:
        df, target_col = load_openml(name)
        summary, md = run_dataset(name, df, target_col)
        results.append(summary)
        report.extend(md)

    if include_covtype:
        df, target_col = load_covtype()
        summary, md = run_dataset("covtype", df, target_col)
        results.append(summary)
        report.extend(md)

    print(f"\n{'=' * 78}\nOverall summary\n{'=' * 78}")
    header = (f"{'dataset':<16}{'n':>8}{'categ.':>8}{'numeric':>8}{'tree acc':>14}"
              f"{'rule acc':>14}{'agree':>8}{'rules':>8}{'attrs used':>12}{'time(s)':>9}")
    print(header)
    report.append("## Overall summary\n\n")
    report.append(
        "`categ.`/`numeric` = number of original attributes of each type (after dropping "
        "signal-free columns, see module docstring); `attrs used` = mean number of those "
        "original attributes actually referenced by the extracted rules, out of `categ.` + "
        "`numeric` available -- not the same as the number of *derived* Boolean features the "
        "rules are built from. Since the DisjointRuleSet is a one-to-one re-expression of the "
        "tree (one literal per split, no simplification), this is also exactly the number of "
        "attributes the tree itself splits on -- `score_fold` asserts the two counts match on "
        "every fold, so `attrs used` doubles as that count.\n\n"
    )
    report.append("| dataset | n | categ. | numeric | tree acc | rule acc | agree | rules/fold "
                   "| attrs used/fold | time(s) |\n")
    report.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in results:
        line = (f"{r['name']:<16}{r['n']:>8}{r['n_categorical']:>8}{r['n_numeric']:>8}"
                f"{r['tree_mean']:>8.3f}±{r['tree_std']:.3f}"
                f"{r['rule_mean']:>8.3f}±{r['rule_std']:.3f}"
                f"{r['agreement']:>8.3f}{r['mean_rules']:>8.1f}{r['mean_attrs_used']:>12.1f}{r['time']:>9.1f}")
        print(line)
        report.append(
            f"| {r['name']} | {r['n']} | {r['n_categorical']} | {r['n_numeric']} "
            f"| {r['tree_mean']:.3f}±{r['tree_std']:.3f} | {r['rule_mean']:.3f}±{r['rule_std']:.3f} "
            f"| {r['agreement']:.3f} | {r['mean_rules']:.1f} | {r['mean_attrs_used']:.1f} "
            f"| {r['time']:.1f} |\n"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
