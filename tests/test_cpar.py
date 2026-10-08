import numpy as np
import pandas as pd
import pytest

from pyrulearn.combiners import TopKMeanCombiner
from pyrulearn.data import BooleanDataRepresentation, DataSpec
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.heuristics import CoveredPositives, GeneralizedMEstimate
from pyrulearn.learners.cpar import CPAR, PropagatingCPAR
from pyrulearn.models import ConceptSet, FlatRuleSet, annotate_rules
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


# ------------------------------------------------------------ TopKMeanCombiner

def _topk_fixture():
    # one feature per rule: class a's rules cover 4, 1, 1 of its rows, b's 3 and 3
    spec = DataSpec([f"f{i}" for i in range(5)])
    X = np.zeros((12, 5), dtype=bool)
    for j, rows in enumerate([range(0, 4), [4], [5], range(6, 9), range(9, 12)]):
        X[list(rows), j] = True
    y = np.array(["a"] * 6 + ["b"] * 6)
    train = BooleanDataRepresentation(spec, X, y)
    rules = annotate_rules([Rule([j], target=t, dataspec=spec)
                            for j, t in enumerate(["a", "a", "a", "b", "b"])], train)
    everything = BooleanDataRepresentation(spec, np.ones((1, 5), dtype=bool))
    return rules, everything


def test_top_k_mean_depends_on_k():
    rules, row = _topk_fixture()
    best1 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=1))
    best2 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=2))
    assert best1.predict(row)[0] == "a"          # a's best rule (4) beats b's (3)
    assert best2.predict(row)[0] == "b"          # but a's best two average 2.5 < 3
    # a class with fewer than k covering rules averages what it has
    best5 = FlatRuleSet(rules, combiner=TopKMeanCombiner(CoveredPositives(), k=5))
    assert best5.predict(row)[0] == "b"          # a: 6/3 = 2, b: 6/2 = 3
    assert "best 2 rules" in best2.combiner.describe()
    with pytest.raises(ValueError):
        TopKMeanCombiner(k=0)


# ------------------------------------------------------------ CPAR

def test_cpar_builds_a_concept_set_predicted_by_top_k_expected_accuracy():
    data = _data()
    model = CPAR().fit(data)
    assert isinstance(model, ConceptSet)
    assert isinstance(model.combiner, TopKMeanCombiner) and model.combiner.k == 5
    h = model.combiner.heuristic
    assert isinstance(h, GeneralizedMEstimate) and (h.m, h.cost) == (2, 0.5)
    for concept in model.concepts:
        bodies = [tuple(sorted(l.feature for l in r.conditions)) for r in concept.rules]
        assert len(bodies) == len(set(bodies)) > 1
        assert all(r.target == concept.label for r in concept.rules)
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    assert model.to_string().splitlines()[0].startswith("% conflict resolution: mean GeneralizedMEstimate")


def test_one_search_copies_the_rule_at_equally_good_conditions():
    # y = f0 or f1, symmetric: f0 and f1 have the same gain, so one search yields both
    rng = np.random.default_rng(0)
    raw = rng.random((400, 3)) < 0.5
    y = np.where(raw[:, 0] | raw[:, 1], "pos", "neg")
    data = BooleanDataRepresentation(neg_spec(["f0", "f1", "f2"]), neg_X(raw), y)
    X = data.X
    pos = y == "pos"
    w = np.ones(len(y))
    copying = CPAR(gain_similarity=0.5)._grow(X, X.astype(float), pos, w)
    single = CPAR(gain_similarity=1.0)._grow(X, X.astype(float), pos, w)
    assert sorted(copying) == [(0,), (2,)]                     # f0 and f1, from one search
    assert single == [(2,)]                                    # f1 is slightly better on this sample


def test_cpar_stops_once_the_positive_weight_has_decayed():
    data = _data()
    X, pos = data.X, data.y == "pos"
    bodies = CPAR()._rules_for(X, X.astype(float), pos, np.ones(len(pos)))
    covered = np.zeros(len(pos), dtype=int)
    for b in bodies:
        covered += np.all(X[:, list(b)], axis=1)
    # every rule covers positives, and most positives were covered several times
    assert all(np.any(np.all(X[:, list(b)], axis=1) & pos) for b in bodies)
    assert np.mean(covered[pos] >= 2) > 0.5


def test_cpar_multiclass_and_data_weights():
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = CPAR().fit(data)
    assert [c.label for c in model.concepts] == ["a", "b", "c"]
    assert model.combiner.heuristic.m == 3
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    weighted = CPAR().fit(data.with_weights(np.where(y == "c", 3.0, 1.0)))
    assert isinstance(weighted.concepts[0].rules[0].stats().confusion.rule_stats("a").tp, float)


def test_cpar_rejects_bad_arguments():
    with pytest.raises(ValueError):
        CPAR(gain_similarity=0.0)
    one = BooleanDataRepresentation(DataSpec(["a"]), np.ones((3, 1), bool), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        CPAR().fit(one)


# ------------------------------------------------------------ PropagatingCPAR

def _numeric_data(n=300, seed=3):
    # numeric thresholds: fixing one closes the attribute's others (propagation)
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n), "c": rng.choice(list("pqr"), n)})
    y = np.where((df["x"] > 0.3) & (df["c"] != "q") | (df["z"] < -0.8), "pos", "neg")
    df["y"] = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    spec = build_dataspec(df, target="y", max_intervals=6).build()
    return BooleanDataRepresentation(spec, binarize(spec, df), df["y"].to_numpy())


@pytest.mark.parametrize("make", [
    _data,
    _numeric_data,
    lambda: _data(noise=0.0).with_weights(np.linspace(0.5, 2.0, 600)),
])
def test_propagating_cpar_learns_cpars_model(make):
    # an implied condition has FOIL gain 0 < min_gain, so CPAR never picks one;
    # propagation only spares PropagatingCPAR from scoring it
    data = make()
    assert PropagatingCPAR().fit(data).to_string() == CPAR().fit(data).to_string()


def test_propagating_cpar_multiclass_and_bad_arguments():
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    assert PropagatingCPAR().fit(data).to_string() == CPAR().fit(data).to_string()
    with pytest.raises(ValueError):
        PropagatingCPAR(gain_similarity=0.0)
    one = BooleanDataRepresentation(DataSpec(["a"]), np.ones((3, 1), bool), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        PropagatingCPAR().fit(one)
