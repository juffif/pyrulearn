import numpy as np
import pytest

from pyrulearn.data import BooleanDataRepresentation, DataSpec
from pyrulearn.learners.rulefit import RuleFit
from pyrulearn.models import FlatRuleSet, LinearRuleModel, WeightedSum, annotate_rules
from pyrulearn.rule import Rule, WeightedRule

from _negation_helpers import neg_spec, neg_X


def _rep(X, y=None, names=None):
    X = np.asarray(X, dtype=bool)
    ds = DataSpec(names or [f"f{i}" for i in range(X.shape[1])])
    return BooleanDataRepresentation(ds, X, None if y is None else np.asarray(y))


def _linear(ds):
    return LinearRuleModel([
        WeightedRule([], target="pos", dataspec=ds, weight=-0.5),
        WeightedRule([0], target="pos", dataspec=ds, weight=1.2),
        WeightedRule([1], target="pos", dataspec=ds, weight=-0.9),
    ], classes=["neg", "pos"])


# ------------------------------------------------------------ weight_format

def test_weight_format_formats_a_weighted_rule_and_is_ignored_by_plain_rules():
    ds = DataSpec(["a"])
    w = WeightedRule([0], target="x", dataspec=ds, weight=0.5)
    assert w.to_string(fmt="prolog") == "0.5::x(X) :- a(X)."
    assert w.to_string(fmt="prolog", weight_format="6.2f") == "  0.50::x(X) :- a(X)."
    assert w.to_string(fmt="logic", weight_format=".3f").endswith("[0.500]")
    plain = Rule([0], target="x", dataspec=ds)
    assert plain.to_string(fmt="prolog", weight_format="6.2f") == plain.to_string(fmt="prolog")


def test_weight_format_aligns_signed_weights_in_a_printed_model():
    m = _linear(DataSpec(["a", "b"]))
    lines = [l for l in m.to_string(fmt="prolog", show_stats=False, weight_format="6.2f").splitlines()
             if "::" in l]
    assert lines == [" -0.50::pos(X) :- true.",
                     "  1.20::pos(X) :- a(X).",
                     " -0.90::pos(X) :- b(X)."]
    assert len({l.index("::") for l in lines}) == 1


# ------------------------------------------------ LinearRuleModel / WeightedSum

def test_linear_rule_model_sums_signed_weights_and_an_intercept():
    ds = DataSpec(["a", "b"])
    m = _linear(ds)
    data = _rep([[0, 0], [1, 0], [0, 1], [1, 1]], names=["a", "b"])
    np.testing.assert_allclose(m.scores(data)[:, 1], [-0.5, 0.7, -1.4, -0.2])
    np.testing.assert_allclose(m.scores(data)[:, 0], 0.0)      # no rules: scores 0
    assert list(m.predict(data)) == ["neg", "pos", "neg", "neg"]
    assert m.labels == ["neg", "pos"]


def test_linear_rule_model_prints_its_resolution_and_keeps_per_rule_weights_in_logic():
    m = _linear(DataSpec(["a", "b"]))
    text = m.to_string(fmt="prolog")
    assert text.splitlines()[0] == "% conflict resolution: sum of rule weights per class, highest wins"
    logic = m.to_string(fmt="logic")
    assert "[1.2]" in logic and "[-0.9]" in logic          # not collapsed into one DNF


def test_linear_rule_model_validates_weights_and_heads():
    ds = DataSpec(["a"])
    with pytest.raises(ValueError, match="weighted rules"):
        LinearRuleModel([Rule([0], target="pos", dataspec=ds)], classes=["neg", "pos"])
    with pytest.raises(ValueError, match="not one of"):
        LinearRuleModel([WeightedRule([0], target="x", dataspec=ds, weight=1.0)], classes=["neg", "pos"])


def test_weighted_sum_ties_go_to_the_more_frequent_training_class():
    ds = DataSpec(["a"])
    train = _rep([[1], [1], [1], [0]], ["b", "b", "a", "b"], names=["a"])
    # both classes score +1 wherever a holds; "b" is more frequent among the rules' stats
    rules = annotate_rules([WeightedRule([0], target="a", dataspec=ds, weight=1.0),
                            WeightedRule([0], target="b", dataspec=ds, weight=1.0)], train)
    m = LinearRuleModel(rules, classes=["a", "b"])
    assert m.predict(_rep([[1]], names=["a"]))[0] == "b"
    assert "sum of rule weights" in WeightedSum(["a", "b"]).describe()


# ------------------------------------------------------------------- RuleFit

def _conjunction_data(n=400, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 5)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1], "good", "bad")
    flip = rng.random(n) < noise
    y[flip] = np.where(y[flip] == "good", "bad", "good")
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(5)]), neg_X(raw), y)


def _design(rf, data):
    probes = [Rule(list(b), target=None, dataspec=data.spec) for b in rf.candidates_]
    return FlatRuleSet(probes).coverage_matrix(data).T.astype(float)


def test_rulefit_finds_a_sparse_model_whose_predictions_are_the_regressions():
    data = _conjunction_data()
    rf = RuleFit(max_len=2, C=0.1, random_state=0)
    m = rf.fit(data)
    assert isinstance(m, LinearRuleModel)
    assert len(m.rules) < len(rf.candidates_) // 5              # sparse
    assert np.mean(np.asarray(m.predict(data)) == data.y) == 1.0
    np.testing.assert_array_equal(m.predict(data), rf.estimator_.predict(_design(rf, data)))
    # binary: every rule counts for the positive class; one intercept rule
    assert {r.target for r in m.rules} == {rf.estimator_.classes_[1]}
    assert sum(len(r.conditions) == 0 for r in m.rules) == 1
    np.testing.assert_allclose(m.scores(data)[:, 1], rf.estimator_.decision_function(_design(rf, data)))


def test_rulefit_smaller_C_keeps_fewer_rules():
    data = _conjunction_data(noise=0.2)
    sizes = [len(RuleFit(max_len=2, C=C, random_state=0).fit(data).rules) for C in (0.05, 10.0)]
    assert sizes[0] < sizes[1]


def test_rulefit_multiclass_is_multinomial_with_one_intercept_per_class():
    rng = np.random.default_rng(1)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    rf = RuleFit(max_len=2, C=1.0, random_state=0)
    m = rf.fit(data)
    assert m.labels == ["a", "b", "c"]
    assert sorted(r.target for r in m.rules if not r.conditions) == ["a", "b", "c"]
    np.testing.assert_array_equal(m.predict(data), rf.estimator_.predict(_design(rf, data)))
    assert np.mean(np.asarray(m.predict(data)) == y) > 0.95


def test_rulefit_accepts_an_external_pool_include_features_and_cv():
    data = _conjunction_data(n=200)
    pool = FlatRuleSet([Rule([0, 2], target="good", dataspec=data.spec),
                        Rule([2, 0], target="bad", dataspec=data.spec),    # same body: counted once
                        Rule([], target="bad", dataspec=data.spec)])       # empty body: skipped
    rf = RuleFit(rules=pool, include_features=True, cv=3, Cs=4, random_state=0)
    rf.fit(data)
    assert rf.candidates_[0] == (0, 2)
    assert len(rf.candidates_) == 1 + data.spec.n_features              # (0, 2), then (0,) .. (9,)
    assert hasattr(rf.estimator_, "C_")


def test_rulefit_rejects_a_ridge_l1_ratio():
    with pytest.raises(ValueError, match="l1_ratio"):
        RuleFit(l1_ratio=0.0)
