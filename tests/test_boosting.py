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


# ------------------------------------------------------------------- ENDER

from pyrulearn.learners.boosting import ENDER, ExponentialLoss, LogisticLoss, SigmoidLoss  # noqa: E402


@pytest.mark.parametrize("loss", [LogisticLoss(), ExponentialLoss(), SigmoidLoss()])
def test_loss_derivatives_match_finite_differences(loss):
    rng = np.random.default_rng(0)
    n, K = 20, 2
    F = rng.normal(size=(n, K))
    Y = np.eye(K)[rng.integers(0, K, n)]
    d = rng.random(n) + 0.5
    G, H = loss.derivatives(F, Y, d)
    eps = 1e-5
    for k in range(K):
        for i in (0, 7, 13):
            step = np.zeros_like(F)
            step[i, k] = eps
            g_num = (loss.value(F + step, Y, d) - loss.value(F - step, Y, d)) / (2 * eps)
            h_num = (loss.value(F + step, Y, d) - 2 * loss.value(F, Y, d) + loss.value(F - step, Y, d)) / eps ** 2
            assert G[i, k] == pytest.approx(g_num, rel=1e-4, abs=1e-6)
            assert H[i, k] == pytest.approx(h_num, rel=1e-3, abs=1e-4)


def test_ender_learns_positive_class_votes_that_find_the_concept():
    data = _data()
    model = ENDER(n_rules=100, random_state=0).fit(data)
    assert isinstance(model, LinearRuleModel)
    assert all(r.weight > 0 for r in model.rules)                 # every rule votes for its class
    assert sum(len(r.conditions) == 0 for r in model.rules) == 1  # one default rule
    bodies = {(r.target, tuple(sorted(l.feature for l in r.conditions))) for r in model.rules}
    assert ("pos", (0, 2)) in bodies and ("pos", (4, 6)) in bodies
    assert len(bodies) == len(model.rules)                        # repeated rules merged
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85
    scores = model.scores(test)
    np.testing.assert_array_equal(np.asarray(model.predict(test)) == "pos", scores[:, 1] > scores[:, 0])


@pytest.mark.parametrize("kw", [
    dict(method="gradient"), dict(method="gradient_boosting"), dict(method="newton", subsample=0.5),
    dict(loss="exponential"), dict(loss="exponential", method="simultaneous"),
    dict(loss="sigmoid", shrinkage=1.0, subsample=0.5), dict(shrinkage=1.0), dict(beta=0.6),
])
def test_ender_variants(kw):
    data = _data()
    model = ENDER(n_rules=40, random_state=0, **kw).fit(data)
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85


def test_ender_multiclass_and_loss_restrictions():
    rng = np.random.default_rng(2)
    raw = rng.random((400, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = ENDER(n_rules=60, random_state=0).fit(data)
    assert model.labels == ["a", "b", "c"]
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    with pytest.raises(ValueError, match="two classes only"):
        ENDER(loss="exponential").fit(data)
    with pytest.raises(ValueError):
        ENDER(method="exact")


def test_ender_early_stopping_ends_on_noise():
    rng = np.random.default_rng(3)
    raw = rng.random((400, 6)) < 0.5
    noise = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw),
                                      rng.choice(["x", "y"], 400))
    full = ENDER(n_rules=200, random_state=0).fit(noise)
    stopped = ENDER(n_rules=200, early_stopping=True, random_state=0).fit(noise)
    assert len(stopped.rules) < 0.75 * len(full.rules)


def test_ender_is_reproducible_and_uses_data_weights():
    data = _data()
    a = ENDER(n_rules=30, random_state=5).fit(data)
    b = ENDER(n_rules=30, random_state=5).fit(data)
    assert [r.to_string() for r in a.rules] == [r.to_string() for r in b.rules]
    w = ENDER(n_rules=30, random_state=5).fit(data.with_weights(np.where(data.y == "pos", 4.0, 1.0)))
    assert [r.to_string() for r in w.rules] != [r.to_string() for r in a.rules]


def test_ender_defaults_are_the_papers_constant_step_logit_setting():
    e = ENDER()
    assert (e.method, e.beta, e.shrinkage, e.subsample, e.n_rules) == ("constant_step", 0.2, 0.1, 0.25, 500)
    assert isinstance(e.loss, LogisticLoss)


def test_exponential_response_is_the_smoothed_exact_minimizer():
    loss = ExponentialLoss()
    loss.eps = 0.5
    w_pos, w_neg = 6.0, 2.0                        # g = W- - W+, h = W+ + W-
    assert loss.response(w_neg - w_pos, w_pos + w_neg, 0.2) == pytest.approx(0.5 * np.log(6.5 / 2.5))
    assert SigmoidLoss().response(-3.0, 1.0, 0.7) == 0.7


def test_a_larger_constant_step_gives_rules_covering_less():
    data = _data()
    X = data.X

    def mean_coverage(beta):
        model = ENDER(n_rules=60, beta=beta, random_state=0).fit(data)
        body = [r for r in model.rules if r.conditions]
        return np.mean([np.all(X[:, [l.feature for l in r.conditions]], axis=1).sum() for r in body])

    assert mean_coverage(0.05) > mean_coverage(2.0)


def test_invalid_method_and_loss_combinations_are_refused():
    with pytest.raises(ValueError, match="exponential"):
        ENDER(method="simultaneous")
    with pytest.raises(ValueError, match="convex"):
        ENDER(loss="sigmoid", method="newton")
    with pytest.raises(ValueError):
        ENDER(beta=0.0)

