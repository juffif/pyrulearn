import numpy as np
import pytest

from pyrulearn.data import BooleanDataRepresentation, DataSpec
from pyrulearn.learners.lri import LRI
from pyrulearn.learners.seco import CoveringState, LRIReweighting
from pyrulearn.models import ConceptModel, EnsembleModel

from _negation_helpers import neg_spec, neg_X


def _data(n=600, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


def test_lri_builds_an_equal_number_of_dnf_rules_per_class():
    data = _data()
    model = LRI(n_rules=7, max_terms=3, max_length=4).fit(data)
    assert isinstance(model, EnsembleModel)
    assert all(isinstance(m, ConceptModel) and m.default_prediction is None for m in model.members)
    per_class = {}
    for m in model.members:
        per_class[m.label] = per_class.get(m.label, 0) + 1
        assert 1 <= len(m.rules) <= 3
        assert all(1 <= len(r.conditions) <= 4 for r in m.rules)
    assert per_class == {"neg": 7, "pos": 7}                      # both classes, equal counts
    assert model.to_string().splitlines()[0] == "% conflict resolution: vote of members"


def test_lri_predicts_by_counting_satisfied_rules():
    data = _data()
    model = LRI(n_rules=10, max_terms=2).fit(data)
    votes = {c: np.zeros(data.n_samples) for c in ("neg", "pos")}
    for m in model.members:
        votes[m.label] += np.asarray(m.predict(data)) == m.label
    pred = np.asarray(model.predict(data))
    clear = votes["pos"] != votes["neg"]
    np.testing.assert_array_equal(pred[clear], np.where(votes["pos"] > votes["neg"], "pos", "neg")[clear])


def test_lri_is_accurate_on_a_noisy_concept():
    model = LRI(n_rules=50, max_terms=4).fit(_data())
    test = _data(seed=1)
    assert np.mean(np.asarray(model.predict(test)) == test.y) > 0.85


def test_terms_keep_a_true_positive_and_stop_once_they_cover_no_negative():
    data = _data()
    X = data.X
    pos = data.y == "pos"
    model = LRI(n_rules=5, max_terms=1, max_length=5).fit(data)
    for m in model.members:
        cov = np.all(X[:, [l.feature for l in m.rules[0].conditions]], axis=1)
        own = data.y == m.label
        assert np.any(cov & own)                                  # TP > 0
        if len(m.rules[0].conditions) < 5:
            assert not np.any(cov & ~own)                         # stopped early only at FP = 0
    assert pos.any()


def test_freezing_restricts_later_rules_to_the_features_used_so_far():
    data = _data()
    model = LRI(n_rules=6, max_terms=1, freeze_features_after=1).fit(data)
    for label in ("neg", "pos"):
        members = [m for m in model.members if m.label == label]
        first = {l.feature for l in members[0].rules[0].conditions}
        for m in members[1:]:
            assert {l.feature for r in m.rules for l in r.conditions} <= first


def test_lri_multiclass_and_data_weights():
    rng = np.random.default_rng(2)
    raw = rng.random((300, 4)) < 0.5
    y = np.where(raw[:, 0], "a", np.where(raw[:, 1], "b", "c"))
    data = BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(4)]), neg_X(raw), y)
    model = LRI(n_rules=5, max_terms=2).fit(data)
    assert sorted({m.label for m in model.members}) == ["a", "b", "c"] and len(model.members) == 15
    assert np.mean(np.asarray(model.predict(data)) == y) > 0.95
    weighted = LRI(n_rules=5, max_terms=2).fit(data.with_weights(np.where(y == "c", 5.0, 1.0)))
    assert len(weighted.members) == 15


def test_lri_reweighting_halves_the_error_counts_past_the_cap():
    pos = np.array([True, False])
    st = CoveringState(np.ones(2), pos)
    rw = LRIReweighting(max_errors=2)
    wrong = np.array([False, True])                               # errs on both rows
    for _ in range(3):
        st.record(wrong)
        w = rw.weights(st, wrong)
    np.testing.assert_array_equal(st.extra["lri_errors"], [1, 1])  # 3 > 2 -> 3 // 2
    np.testing.assert_allclose(w, [2.0, 2.0])
    np.testing.assert_array_equal(st.errors, [3, 3])              # the plain counts


def test_lri_rejects_bad_arguments():
    with pytest.raises(ValueError):
        LRI(n_rules=0)
    one = BooleanDataRepresentation(DataSpec(["a"]), np.ones((3, 1), bool), np.array(["x"] * 3))
    with pytest.raises(ValueError, match="two classes"):
        LRI().fit(one)
