import math

import numpy as np
import pytest

from pyrulearn.data import BooleanDataRepresentation, NListRepresentation
from pyrulearn.evaluation import RuleStats
from pyrulearn.heuristics import SlipperZ
from pyrulearn.learners.boosting import Slipper
from pyrulearn.models import LinearRuleModel

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1, n_features=6):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, n_features)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(n_features)]), neg_X(raw), y)


def _bodies(model):
    return [tuple(sorted(l.feature for l in r.conditions)) for r in model.rules]


def test_slipper_z_heuristic():
    assert SlipperZ().score(RuleStats(tp=9, fp=4, fn=0, tn=0)) == pytest.approx(1.0)
    assert SlipperZ().score(RuleStats(tp=2.25, fp=0.0, fn=0, tn=0)) == pytest.approx(1.5)


def test_slipper_learns_a_linear_model_that_finds_the_concept():
    data = _data()
    model = Slipper(n_rounds=20, random_state=0).fit(data)
    assert isinstance(model, LinearRuleModel)
    assert model.labels == ["neg", "pos"]
    # the less frequent class is the target; the intercept comes first
    assert {r.target for r in model.rules} == {"pos"}
    assert model.rules[0].conditions == () or len(model.rules[0].conditions) == 0
    bodies = _bodies(model)
    assert len(bodies) == len(set(bodies))                          # repeated rules merged
    assert (0, 2) in bodies and (4, 6) in bodies                    # f0 & f1, f2 & f3
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    assert model.to_string().splitlines()[0] == "% conflict resolution: sum of rule weights per class, highest wins"


def test_slipper_prediction_is_the_sign_of_the_summed_confidences():
    data = _data()
    model = Slipper(n_rounds=10, random_state=0).fit(data)
    scores = model.scores(data)
    np.testing.assert_array_equal(np.asarray(model.predict(data)) == "pos", scores[:, 1] > 0)


def test_slipper_is_reproducible_and_representation_independent():
    data = _data()
    a = Slipper(n_rounds=8, random_state=3).fit(data)
    b = Slipper(n_rounds=8, random_state=3).fit(data)
    c = Slipper(n_rounds=8, random_state=3).fit(NListRepresentation.from_boolean(data))
    assert [r.to_string() for r in a.rules] == [r.to_string() for r in b.rules] == [r.to_string() for r in c.rules]


def test_slipper_target_class_and_multiclass():
    data = _data()
    neg = Slipper(n_rounds=5, target_class="neg", random_state=0).fit(data)
    assert {r.target for r in neg.rules} == {"neg"}
    rng = np.random.default_rng(2)
    raw = rng.random((400, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    multi = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = Slipper(n_rounds=5, random_state=0).fit(multi)
    assert model.labels == ["a", "b", "c"]
    assert {r.target for r in model.rules} == {"a", "b", "c"}
    assert np.mean(np.asarray(model.predict(multi)) == y) > 0.95


def test_slipper_uses_the_data_weights_as_initial_boosting_weights():
    data = _data()
    heavy_neg = np.where(data.y == "neg", 10.0, 1.0)
    plain = Slipper(n_rounds=5, random_state=0).fit(data)
    weighted = Slipper(n_rounds=5, random_state=0).fit(data.with_weights(heavy_neg))
    assert [r.to_string() for r in plain.rules] != [r.to_string() for r in weighted.rules]
    assert isinstance(weighted.rules[1].stats().confusion.rule_stats("pos").tp, float)


def test_slipper_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Slipper(n_rounds=0)
    one_class = BooleanDataRepresentation(neg_spec(["a"]), neg_X(np.ones((3, 1), bool)), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        Slipper().fit(one_class)
