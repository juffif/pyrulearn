from types import SimpleNamespace

import numpy as np
import pytest
from pyrulearn import BooleanDataRepresentation, DataSpec
from pyrulearn.models import RuleList, RuleSet

imodels = pytest.importorskip("imodels")

from pyrulearn.interfaces.imodels import (  # noqa: E402
    BayesianRuleList,
    BayesianRuleListImporter,
    BayesianRuleSet,
    BayesianRuleSetImporter,
)
from pyrulearn.rule import Literal  # noqa: E402

from _negation_helpers import neg_spec, neg_X  # noqa: E402

# maxlen=3 + discretization_method="randomforest" keeps fit fast for tests;
# BRS's own "fpgrowth" discretization path calls mlxtend's fpgrowth with
# kwargs (supp=/zmin=/zmax=) that mismatch its actual signature and always
# raises TypeError, so "randomforest" is the only usable method regardless.
_FAST_BRS = dict(num_iterations=30, num_chains=1, n_rules=50, maxlen=3, supp=1,
                 discretization_method="randomforest", random_state=0)


def _binary_data(n=300, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    y_clean = X[:, 0] & ~X[:, 1] & (X[:, 2] | X[:, 3])
    noise = rng.random(n) < 0.05
    y = np.where(noise, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")
    # negation-enabled feature space: BRS's "_neg" items now land on an
    # explicit negation feature (matches build_dataspec's default, i.e.
    # the real workflow-1 path these learners take)
    ds = neg_spec([f"f{i}" for i in range(d)])
    return BooleanDataRepresentation(ds, neg_X(X), y)


def test_imports_as_rulelist_and_matches_model_predictions():
    # the first importer producing an ordered RuleList rather than a
    # RuleSet -- a Bayesian Rule List's rules are only meaningful in
    # order (each implicitly conditioned on every earlier one not
    # firing), so first-match-wins is the actual semantics
    rep = _binary_data()
    learner = BayesianRuleList(random_state=0)
    rules = learner.fit(rep)

    assert isinstance(rules, RuleList)
    assert not isinstance(rules, RuleSet)
    assert len(rules.rules) > 0
    # the trailing ELSE became the list's default_rule, with no conditions
    assert rules.default_rule is not None
    assert len(rules.default_rule.conditions) == 0

    # pyrulearn's RuleList must reproduce the model's own predictions exactly
    model = learner.fit_external(rep.X, rep.y, feature_names=rep.spec.feature_names)
    model_preds = np.asarray(model.predict(np.asarray(rep.X).astype(int)))
    rule_preds = np.asarray(rules.predict(rep))
    assert np.array_equal(model_preds, rule_preds)
    print("BayesianRuleList imports as RuleList and matches model predictions exactly: OK")


def test_rule_targets_follow_probability_not_position():
    # each rule carries a posterior P(class 1) (model.theta), NOT a hard
    # label -- a rule with theta < 0.5 is a confident *negative* rule,
    # and its imported target must reflect that (not stored on the rule
    # itself -- cross-checked here against the raw fitted model's own
    # theta array directly).
    rep = _binary_data()
    learner = BayesianRuleList(random_state=0)
    model = learner.fit_external(rep.X, rep.y, feature_names=rep.spec.feature_names)
    rules = learner.fit(rep)

    imported = list(rules.rules) + [rules.default_rule]
    assert len(imported) == len(model.theta)
    neg_class, pos_class = model.classes_[0], model.classes_[1]
    for r, theta in zip(imported, model.theta):
        assert r.target == (pos_class if theta > 0.5 else neg_class)
    # this test only bites if some rule is a confident *negative* one --
    # i.e. some theta is below 0.5, so target != "always pos_class" trivially
    assert any(t < 0.5 for t in model.theta), "expected a negative-predicting rule to exercise the inversion"
    assert any(t > 0.5 for t in model.theta), "expected a positive-predicting rule too"
    print("BRL rule targets follow their posterior probability, not list position: OK")


def test_only_positive_literals():
    # BRL builds antecedents from FP-growth itemsets (sets of *present*
    # items), so an imported rule never contains a negated literal
    rep = _binary_data()
    rules = BayesianRuleList(random_state=0).fit(rep)
    for r in rules.rules:
        # every condition is a plain positive literal -- there are no
        # negative literals in the data any more, and BRL's
        # importer has no neg path at all (FP-growth "present item" sets)
        assert all(isinstance(lit, Literal) for lit in r.conditions)
        assert sorted(lit.feature for lit in r.conditions) == list(r.pos)
    print("BRL imported rules contain only positive literals: OK")


def test_provenance_tagged_on_rules_and_default():
    rep = _binary_data()
    rules = BayesianRuleList(random_state=0).fit(rep)
    for r in list(rules.rules) + [rules.default_rule]:
        assert r.provenance.source == "imodels.BayesianRuleListClassifier"
        assert r.provenance.learner == "BayesianRuleListImporter"
    print("BRL import provenance tagged on every rule incl. the default: OK")


def test_raises_on_feature_count_mismatch():
    rep = _binary_data(d=5)
    model = BayesianRuleList(random_state=0).fit_external(
        rep.X, rep.y, feature_names=rep.spec.feature_names
    )
    wrong_ds = DataSpec(["only", "three", "features"])
    try:
        BayesianRuleListImporter().import_model(model, wrong_ds)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "features" in str(e)
    print("BayesianRuleListImporter raises on feature-count mismatch: OK")


def test_nominal_attributes_via_build_dataspec_one_hot():
    # BRL needs 0/1 columns, so a nominal attribute must be ONE-HOT
    # encoded first -- raw strings and ordinal/label codes are both
    # rejected by its own fit. build_dataspec/binarize already produce
    # exactly that (one color=red/color=green/... feature per category),
    # so the ordinary workflow-1 path just works.
    import pandas as pd
    from pyrulearn.data.io import binarize, build_dataspec

    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"cat": rng.choice(["red", "green", "blue"], size=n)})
    y = np.where(df["cat"] == "red", "pos", "neg")
    df["label"] = y

    ds = build_dataspec(df, target="label", arff_types={"cat": "nominal"}).build()
    assert {"cat=red", "cat=green", "cat=blue"} <= set(ds.feature_names)
    rep = BooleanDataRepresentation(ds, binarize(ds, df), y)

    rules = BayesianRuleList(random_state=0).fit(rep)
    assert isinstance(rules, RuleList)
    # rules reference the one-hot features by their real DataSpec names
    # (pretty-printed as "cat = red" / "cat ≠ red", not the raw feature
    # name "cat=red"/"cat!=red")
    rendered = " ".join(r.to_string("logic") for r in rules.rules)
    assert "cat = " in rendered or "cat ≠ " in rendered
    print("Nominal attributes work via build_dataspec's one-hot encoding: OK")


def test_multiclass_target_raises():
    # imodels' own fit refuses >2 classes properly (unlike wittgenstein's
    # silent collapse), so no extra guard is needed on pyrulearn's side
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(200, 4)).astype(bool)
    y = rng.choice(["a", "b", "c"], size=200)
    ds = DataSpec([f"f{i}" for i in range(4)])
    rep = BooleanDataRepresentation(ds, X, y)

    try:
        BayesianRuleList(random_state=0).fit(rep)
        assert False, "expected ValueError for 3-class y"
    except ValueError as e:
        assert "binary" in str(e).lower()
    print("BayesianRuleList surfaces imodels' own binary-only error: OK")


def test_imports_as_ruleset_and_matches_model_predictions():
    # BRS is an unordered OR-of-AND pattern set (not a list): any single
    # rule firing predicts classes_[1], no rule firing predicts classes_[0]
    rep = _binary_data()
    learner = BayesianRuleSet(**_FAST_BRS)
    rules = learner.fit(rep)

    assert isinstance(rules, RuleSet)
    assert not isinstance(rules, RuleList)
    assert len(rules.rules) > 0
    assert rules.default_rule is not None
    assert len(rules.default_rule.conditions) == 0

    model = learner.fit_external(rep.X, rep.y, feature_names=rep.spec.feature_names)
    # model.predict returns raw 0/1, not decoded through classes_
    raw_preds = np.asarray(model.predict(np.asarray(rep.X).astype(int)))
    model_preds = np.where(raw_preds == 1, model.classes_[1], model.classes_[0])
    rule_preds = np.asarray(rules.predict(rep))
    assert np.array_equal(model_preds, rule_preds)
    print("BayesianRuleSet imports as RuleSet and matches model predictions exactly: OK")


def test_all_rules_target_positive_class_default_targets_negative():
    rep = _binary_data()
    rules = BayesianRuleSet(**_FAST_BRS).fit(rep)
    model = BayesianRuleSet(**_FAST_BRS).fit_external(rep.X, rep.y, feature_names=rep.spec.feature_names)
    neg_class, pos_class = model.classes_[0], model.classes_[1]

    assert all(r.target == pos_class for r in rules.rules)
    assert rules.default_rule.target == neg_class
    print("BRS rules all target the positive class, default targets the negative: OK")


def test_negative_literals_parsed_correctly():
    # unit-tests the "_neg"-suffix parsing directly against a hand-built
    # fake model, instead of relying on simulated annealing to happen to
    # select a rule with a negative literal
    ds = neg_spec(["f0", "f1", "f2"])
    fake_model = SimpleNamespace(
        rules_=[["f0", "f1_neg"], ["f2"]],
        classes_=np.array(["neg", "pos"]),
        feature_names_=list(ds.feature_names),
    )
    from pyrulearn.models import FlatRuleSet

    rules = BayesianRuleSetImporter().import_model(fake_model, ds)
    assert isinstance(rules, FlatRuleSet)
    r0, r1 = rules.rules

    # "f1_neg" becomes a positive literal on the paired negation feature
    assert {lit.feature for lit in r0.conditions} == {
        ds.feature_index("f0"), ds.feature_index("not f1"),
    }
    assert {lit.feature for lit in r1.conditions} == {ds.feature_index("f2")}
    assert r0.target == "pos" and r1.target == "pos"
    assert rules.default_rule.target == "neg"
    print("BRS negative ('_neg'-suffixed) literals parsed correctly: OK")


def test_brs_provenance_tagged_on_rules_and_default():
    rep = _binary_data()
    rules = BayesianRuleSet(**_FAST_BRS).fit(rep)
    for r in list(rules.rules) + [rules.default_rule]:
        assert r.provenance.source == "imodels.BayesianRuleSetClassifier"
        assert r.provenance.learner == "BayesianRuleSetImporter"
    print("BRS import provenance tagged on every rule incl. the default: OK")


def test_brs_raises_on_feature_count_mismatch():
    rep = _binary_data(d=5)
    model = BayesianRuleSet(**_FAST_BRS).fit_external(rep.X, rep.y, feature_names=rep.spec.feature_names)
    wrong_ds = DataSpec(["only", "three", "features"])
    try:
        BayesianRuleSetImporter().import_model(model, wrong_ds)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "features" in str(e)
    print("BayesianRuleSetImporter raises on feature-count mismatch: OK")


def test_brs_fit_external_rejects_non_boolean_input():
    # imodels itself doesn't raise cleanly for this (an opaque downstream
    # AssertionError), so BayesianRuleSet.fit_external checks it directly
    rng = np.random.default_rng(2)
    X = rng.random((100, 4))
    y = np.where(X[:, 0] > 0.5, "pos", "neg")
    try:
        BayesianRuleSet(**_FAST_BRS).fit_external(X, y, feature_names=["f0", "f1", "f2", "f3"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "boolean" in str(e).lower()
    print("BayesianRuleSet.fit_external rejects non-Boolean input with a clear error: OK")


def test_brs_fit_external_rejects_multiclass_target():
    # imodels itself doesn't raise cleanly for this either (an opaque
    # "math domain error" deep in simulated annealing)
    rng = np.random.default_rng(3)
    X = rng.integers(0, 2, size=(150, 4))
    y = rng.choice(["a", "b", "c"], size=150)
    try:
        BayesianRuleSet(**_FAST_BRS).fit_external(X, y, feature_names=["f0", "f1", "f2", "f3"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "binary" in str(e).lower()
    print("BayesianRuleSet.fit_external rejects multi-class targets with a clear error: OK")


def test_brs_nominal_attributes_via_build_dataspec_one_hot():
    import pandas as pd
    from pyrulearn.data.io import binarize, build_dataspec

    rng = np.random.default_rng(5)
    n = 300
    df = pd.DataFrame({"cat": rng.choice(["red", "green", "blue"], size=n)})
    y_clean = df["cat"] == "red"
    # a little label noise, same as _binary_data: a *perfectly* separable
    # target (plain "cat == red") makes imodels' own simulated annealing
    # reach zero training error almost immediately, which triggers the
    # "clean"-move bug documented below on nearly every random_state --
    # this isn't the scenario under test, so avoid it rather than fight it
    noise = rng.random(n) < 0.05
    y = np.where(y_clean ^ noise, "pos", "neg")
    df["label"] = y

    ds = build_dataspec(df, target="label", arff_types={"cat": "nominal"}).build()
    assert {"cat=red", "cat=green", "cat=blue"} <= set(ds.feature_names)
    rep = BooleanDataRepresentation(ds, binarize(ds, df), y)

    # imodels' own simulated-annealing "clean" move has a real bug (see the
    # module docstring) that can raise "list.remove(x): x not in list".
    # Empirically this isn't just a per-random_state coin flip: how often it
    # triggers for a given dataset also depends on things fixed once per
    # Python *process* (e.g. attr_names' dict/set iteration order, subject to
    # hash randomization) and not reset by BayesianRuleSet's own seeding, so
    # a bounded retry over random_state sometimes still can't escape it
    # within an unlucky process. Retry a generous number of times regardless
    # (cheap, and clears the vast majority of runs); if every attempt hits
    # this specific already-diagnosed *upstream* bug, skip rather than fail
    # the suite over a third-party defect unrelated to import correctness
    # (any other exception still propagates immediately, uninvolved).
    rules = None
    last_err = None
    for candidate_rs in range(10):
        try:
            rules = BayesianRuleSet(**{**_FAST_BRS, "random_state": candidate_rs}).fit(rep)
            break
        except ValueError as e:
            if "list.remove" not in str(e):
                raise
            last_err = e
    if rules is None:
        pytest.skip(f"every candidate random_state hit imodels' own clean-move bug: {last_err}")
    assert isinstance(rules, RuleSet)
    rendered = " ".join(r.to_string("logic") for r in rules.rules)
    # BRS can express "cat=red" either directly, or (thanks to the nominal
    # attribute's ExactlyOne constraint) via its negation "cat!=blue and
    # cat!=green" -- both are valid, genuine BRS output, rendered as "cat = "
    # / "cat ≠ " respectively -- so accept either form
    assert "cat = " in rendered or "cat ≠ " in rendered
    print("BRS nominal attributes work via build_dataspec's one-hot encoding: OK")


if __name__ == "__main__":
    test_imports_as_rulelist_and_matches_model_predictions()
    test_rule_targets_follow_probability_not_position()
    test_only_positive_literals()
    test_provenance_tagged_on_rules_and_default()
    test_raises_on_feature_count_mismatch()
    test_nominal_attributes_via_build_dataspec_one_hot()
    test_multiclass_target_raises()
    test_imports_as_ruleset_and_matches_model_predictions()
    test_all_rules_target_positive_class_default_targets_negative()
    test_negative_literals_parsed_correctly()
    test_brs_provenance_tagged_on_rules_and_default()
    test_brs_raises_on_feature_count_mismatch()
    test_brs_fit_external_rejects_non_boolean_input()
    test_brs_fit_external_rejects_multiclass_target()
    test_brs_nominal_attributes_via_build_dataspec_one_hot()
    print("\nAll tests passed.")
