"""End-to-end smoke test / usage demo for pyrulearn."""
import os

import numpy as np

from pyrulearn import (
    BooleanDataRepresentation, DataSpec, DataSpecBuilder, PatternStringImporter, Rule, RuleSetClassifier,
)
from pyrulearn.models import DisjointRuleSet, FlatRuleSet, annotate_rules
from pyrulearn.interfaces.sklearn import from_sklearn_tree
from pyrulearn.evaluation import summarize, coverage_space_plot, coverage_space_auc, rule_refinement_plot

rng = np.random.default_rng(0)
n, d = 200, 6
X = rng.integers(0, 2, size=(n, d)).astype(bool)
feature_names = [f"f{i}" for i in range(d)]
# ground truth: y = f0 AND NOT f1  (with a bit of noise)
y_clean = X[:, 0] & ~X[:, 1]
noise = rng.random(n) < 0.05
y = np.where(noise, ~y_clean, y_clean)
y = np.where(y, "pos", "neg")

# Features carry explicit negations now (f0 .. f5 each paired with "not f0"
# .. "not f5" by a MutuallyExclusive constraint), so a rule condition
# meaning "feature absent" is a positive literal on the negation feature.
_b = DataSpecBuilder()  # negation=True is the default
for name in feature_names:
    _b.add_boolean(name)
ds = _b.build()
# expand the raw (n, 6) matrix into the (n, 12) positive/negation matrix
X_full = np.empty((n, 2 * d), dtype=bool)
X_full[:, 0::2] = X
X_full[:, 1::2] = ~X
rep = BooleanDataRepresentation(ds, X_full, y)

# 1. manual rule + coverage check (scalar + vectorized paths agree)
r1 = Rule.from_pos_neg(
    pos=[ds.feature_index("f0")], neg=[ds.feature_index("f1")], target="pos", dataspec=ds,
)
cov_a = r1.covers_data(rep)
cov_b = r1.covers_data_packed(rep)
assert np.array_equal(cov_a, cov_b), "packed and unpacked coverage disagree!"
assert r1.covers(X_full[0].tolist()) == bool(cov_a[0])
print("Rule:", r1)
print("  covers", int(cov_a.sum()), "of", n, "examples")

# 2. FlatRuleSet: rules annotated with their (frozen) training stats, and the
#    whole model evaluated on the data (a measurement; nothing is stored)
r2 = Rule.from_pos_neg(pos=[], neg=[ds.feature_index("f0")], target="neg", dataspec=ds)
rs = FlatRuleSet(annotate_rules([r1, r2], rep))
for r in rs:
    print(r, "stats:", r.stats())
print("model:", rs.evaluate(rep))

print("summary:", summarize(rs, rep))

# 3. sklearn interface: fit a DecisionTreeClassifier on boolean data, extract rules
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
tree_rules = from_sklearn_tree(clf, dataspec=ds)
print(f"\nExtracted {len(tree_rules)} rules from decision tree:")
for r in tree_rules:
    print(" ", r, "n_covered=", int(r.covers_data(rep).sum()))

# wrap as an sklearn-compatible estimator and cross-validate.
# NOTE: cross_val_score validates X itself before calling fit(), so it must
# be a plain array (not a BooleanDataRepresentation) -- RuleSetClassifier.fit()
# wraps it into one internally via BooleanDataRepresentation.from_xy().
# cross_val_score splits by row, so hand it the already-expanded matrix
# (the fixed rules reference the 12 positive/negation feature columns)
rsc = RuleSetClassifier(rules=tree_rules, default_class="neg")
scores = cross_val_score(rsc, X_full, y, cv=3)
print("cross_val_score (fixed rule set):", scores)


def simple_learner(train_rep: BooleanDataRepresentation) -> DisjointRuleSet:
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(train_rep.X, train_rep.y)
    return from_sklearn_tree(t, dataspec=ds)


rsc2 = RuleSetClassifier(learner=simple_learner, default_class="neg")
scores2 = cross_val_score(rsc2, X_full, y, cv=3)
print("cross_val_score (learner callback):", scores2)

# 4. importers.base: round-trip a pattern-string rule set
importer = PatternStringImporter(feature_names=feature_names)
text = "1 0 - - - - => pos\n- - 1 1 - - => neg\n"
imported = importer.parse(text)
print("\nimported rules:", list(imported))

# 5. analysis: coverage-space plots. An ordered decision list (below) and one
# rule's refinement path; uses tree rules (more of them than r1/r2)
# same rules, as an ordered decision list (most-general-first) -- a connected,
# arrow-annotated cumulative-coverage path starting at (0, 0)
tree_rulelist = tree_rules.to_rulelist(key=lambda r: int(r.covers_data(rep).sum()))
ax2 = coverage_space_plot(tree_rulelist, rep, positive_class="pos", annotate=True)
out_path2 = os.path.join(os.path.dirname(__file__), "coverage_space_rulelist.png")
ax2.figure.savefig(out_path2, dpi=120)
print(f"Saved {out_path2}")

# one rule's own specialization path, the way a greedy learner (HillClimbing /
# SeCo) builds it: order its conditions by incremental precision, then plot how
# coverage shrinks from the full space (upper right) toward the rule's own
# coverage (lower left), one condition at a time. positive_class defaults to
# the rule's own target/head, so pick a "pos" rule to match the plot above
complex_rule = max((r for r in tree_rules if r.target == "pos"), key=lambda r: r.length())
ordered_rule = complex_rule.order_by_precision(rep)
ax3 = rule_refinement_plot(ordered_rule, rep)
out_path3 = os.path.join(os.path.dirname(__file__), "rule_refinement.png")
ax3.figure.savefig(out_path3, dpi=120)
print(f"Saved {out_path3}")

# the convex hull / AUC are only meaningful for one target class at a time,
# so a mixed-target RuleSet needs one coverage_space_auc() call per class
print(f"\nAUC (pos): {coverage_space_auc(tree_rules, rep, 'pos'):.3f}")
print(f"AUC (neg): {coverage_space_auc(tree_rules, rep, 'neg'):.3f}")

print("\nAll checks passed.")
