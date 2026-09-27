"""Row weights on the data representations, weighted search scopes, and
weighted covering."""

import numpy as np
import pytest

from pyrulearn.data import (
    BooleanDataRepresentation, NListRepresentation, PrePostNListRepresentation, SparseDataRepresentation,
)
from pyrulearn.evaluation import RuleStats
from pyrulearn.heuristics import WRAcc
from pyrulearn.learners.seco import (
    AQR, CN2, AdaBoostReweighting, AdditiveReweighting, CoveredAtLeast, LRIReweighting,
    MultiplicativeReweighting, PositiveWeightBelow, RemovalCovering, Rounds, WeightedCovering,
    rule_set_description_length,
)
from pyrulearn.models import DecisionList
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X

REPRESENTATIONS = [BooleanDataRepresentation, NListRepresentation, PrePostNListRepresentation,
                   SparseDataRepresentation]


def _data(n=300, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    flip = rng.random(n) < noise
    y = np.where(flip, np.where(y == "pos", "neg", "pos"), y)
    return neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y


def _counts(data, features, positive, example_mask=None):
    handle = data.initial_cover(example_mask)
    for f in features:
        handle = data.refine_cover(handle, f)
    return data.cover_counts(handle, positive)


def _rules_text(model):
    return [r.to_string() for r in model.rules]


# ------------------------------------------------------------ representations

@pytest.mark.parametrize("cls", REPRESENTATIONS)
def test_integer_weights_count_like_duplicated_rows(cls):
    spec, X, y = _data()
    w = np.random.default_rng(1).integers(1, 4, len(y))
    weighted = cls(spec, X, y, weights=w)
    duplicated = cls(spec, np.repeat(X, w, axis=0), np.repeat(y, w))
    for feats in ([], [0], [0, 2], [4, 7]):
        assert _counts(weighted, feats, "pos") == pytest.approx(_counts(duplicated, feats, "pos"))


@pytest.mark.parametrize("cls", REPRESENTATIONS)
def test_unweighted_counts_stay_integers(cls):
    spec, X, y = _data()
    counts = _counts(cls(spec, X, y), [0, 2], "pos", example_mask=np.arange(len(y)) % 2 == 0)
    assert all(isinstance(c, int) for c in counts)


@pytest.mark.parametrize("cls", REPRESENTATIONS)
def test_a_weighted_scope_multiplies_with_the_data_weights(cls):
    spec, X, y = _data()
    rng = np.random.default_rng(2)
    w, scope = rng.random(len(y)) * 3, rng.random(len(y))
    scope[::5] = 0.0                                 # some rows out of scope
    got = _counts(cls(spec, X, y, weights=w), [0, 2], "pos", example_mask=scope)
    cov = X[:, 0] & X[:, 2]
    eff, pos = w * scope, y == "pos"
    expected = (eff[cov & pos].sum(), eff[cov & ~pos].sum(), eff[~cov & pos].sum(), eff[~cov & ~pos].sum())
    assert got == pytest.approx(expected)
    # a boolean mask is the 0/1 special case of a weight vector
    mask = scope > 0.5
    assert _counts(cls(spec, X, y, weights=w), [1], "neg", example_mask=mask) == pytest.approx(
        _counts(cls(spec, X, y, weights=w), [1], "neg", example_mask=mask.astype(float)))


def test_with_weights_shares_storage_and_weights_survive_row_operations():
    spec, X, y = _data()
    w = np.arange(len(y), dtype=float) % 3
    nlist = NListRepresentation(spec, X, y)
    weighted = nlist.with_weights(w)
    assert weighted._row_idx is nlist._row_idx and nlist.weights is None
    np.testing.assert_array_equal(weighted.weights, w)
    assert weighted.with_weights(None).weights is None
    boolean = BooleanDataRepresentation(spec, X, y, weights=w)
    rows = np.arange(len(y)) < 100
    np.testing.assert_array_equal(boolean.select_rows(rows).weights, w[rows])
    np.testing.assert_array_equal(boolean.relabel(np.where(y == "pos", "a", "b")).weights, w)
    np.testing.assert_array_equal(NListRepresentation.from_boolean(boolean).weights, w)
    np.testing.assert_array_equal(SparseDataRepresentation.from_boolean(boolean).weights, w)
    np.testing.assert_array_equal(boolean.without_negations().weights, w)


def test_invalid_weights_are_rejected():
    spec, X, y = _data()
    with pytest.raises(ValueError, match="shape"):
        BooleanDataRepresentation(spec, X, y, weights=np.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        BooleanDataRepresentation(spec, X, y, weights=-np.ones(len(y)))


# ------------------------------------------------------------ stats

def test_rule_stats_and_printed_stats_are_weighted():
    spec, X, y = _data()
    w = np.full(len(y), 0.5)
    data = BooleanDataRepresentation(spec, X, y, weights=w)
    rule = Rule([0, 2], target="pos", dataspec=spec)
    st = RuleStats.from_rule(rule, data)
    plain = RuleStats.from_rule(rule, BooleanDataRepresentation(spec, X, y))
    assert (st.tp, st.fp, st.fn, st.tn) == pytest.approx((plain.tp / 2, plain.fp / 2, plain.fn / 2, plain.tn / 2))
    model = CN2(target_class="pos").fit(data)
    stats = model.rules[0].stats().confusion.rule_stats("pos")
    assert isinstance(stats.tp, float)
    assert "." in model.to_string().splitlines()[1] or stats.tp == int(stats.tp)
    ev = model.evaluate(data).confusion
    assert ev.n_total == pytest.approx(w.sum())


# ------------------------------------------------------------ learners

def test_cn2_on_integer_weights_learns_the_rules_of_duplicated_rows():
    spec, X, y = _data()
    w = np.random.default_rng(3).integers(1, 4, len(y))
    a = CN2(target_class="pos").fit(BooleanDataRepresentation(spec, X, y, weights=w))
    b = CN2(target_class="pos").fit(BooleanDataRepresentation(spec, np.repeat(X, w, axis=0), np.repeat(y, w)))
    assert _rules_text(a) == _rules_text(b)


def test_description_length_with_integer_weights_equals_duplicated_rows():
    spec, X, y = _data()
    w = np.random.default_rng(4).integers(1, 4, len(y))
    rules = [Rule([0, 2], target="pos", dataspec=spec), Rule([4, 6], target="pos", dataspec=spec)]
    a = rule_set_description_length(rules, BooleanDataRepresentation(spec, X, y, weights=w), "pos")
    b = rule_set_description_length(rules, BooleanDataRepresentation(spec, np.repeat(X, w, axis=0),
                                                                     np.repeat(y, w)), "pos")
    assert a == pytest.approx(b)


# ------------------------------------------------------------ covering strategies

def _state(covering, pos):
    return covering.start(None, pos)


def test_removal_covering_removes_everything_covered():
    pos = np.array([True, True, True, False])
    covered = np.array([True, False, True, True])
    removal = RemovalCovering()
    st = _state(removal, pos)
    assert removal.update(st, covered) is None
    np.testing.assert_array_equal(st.scope, [False, True, False, False])
    removal.update(st, np.ones(4, bool))
    assert removal.exhausted(st)


def test_multiplicative_and_additive_reweighting_touch_only_positives():
    pos = np.array([True, True, True, False])
    covered = np.array([True, False, True, True])
    mult = WeightedCovering(MultiplicativeReweighting(0.5), CoveredAtLeast(2))
    st = _state(mult, pos)
    mult.update(st, covered)
    np.testing.assert_allclose(st.scope, [0.5, 1.0, 0.5, 1.0])      # negatives keep weight 1
    add = WeightedCovering(AdditiveReweighting(), CoveredAtLeast(2))
    st2 = _state(add, pos)
    add.update(st2, covered)
    add.update(st2, covered)
    np.testing.assert_allclose(st2.scope, [1 / 3, 1.0, 1 / 3, 1.0])
    assert not mult.exhausted(st)
    mult.update(st, np.ones(4, bool))
    assert not mult.exhausted(st)                                   # the second positive: once so far
    mult.update(st, np.ones(4, bool))
    assert mult.exhausted(st)                                       # every positive covered twice
    with pytest.raises(ValueError):
        MultiplicativeReweighting(1.0)


def test_adaboost_reweighting_confidence_and_update():
    pos = np.array([True, True, True, False, False])
    covered = np.array([True, True, False, True, False])
    cov = WeightedCovering(AdaBoostReweighting(eps=0.5), Rounds(5))
    st = _state(cov, pos)
    c = cov.update(st, covered)
    assert c == pytest.approx(0.5 * np.log((2 + 0.5) / (1 + 0.5)))
    expected = np.array([np.exp(-c), np.exp(-c), 1, np.exp(c), 1])
    np.testing.assert_allclose(st.scope, expected * 5 / expected.sum())   # total kept
    assert AdaBoostReweighting.assigns_rule_weights


def test_lri_reweighting_counts_errors_of_both_kinds():
    pos = np.array([True, True, False, False])
    covered = np.array([True, False, True, False])                  # errs on rows 1 (FN) and 2 (FP)
    cov = WeightedCovering(LRIReweighting(), Rounds(10))
    st = _state(cov, pos)
    cov.update(st, covered)
    cov.update(st, covered)
    np.testing.assert_allclose(st.scope, [1, 1 + 8, 1 + 8, 1])      # 1 + e**3 with e = 2
    np.testing.assert_array_equal(st.errors, [0, 2, 2, 0])


def test_stop_criteria():
    pos = np.array([True, True, False])
    st = _state(WeightedCovering(), pos)
    assert not CoveredAtLeast(1).done(st) and not PositiveWeightBelow(0.5).done(st) and not Rounds(1).done(st)
    st.record(np.array([True, True, False]))
    st.scope = np.array([0.2, 0.2, 1.0])
    assert CoveredAtLeast(1).done(st) and PositiveWeightBelow(0.5).done(st) and Rounds(1).done(st)
    capped = WeightedCovering(stop=CoveredAtLeast(99), max_rounds=1)
    assert capped.exhausted(st)                                     # the safety net


def test_seco_refuses_a_reweighting_that_fits_rule_weights():
    spec, X, y = _data()
    with pytest.raises(ValueError, match="rule weights"):
        CN2(target_class="pos", covering=WeightedCovering(AdaBoostReweighting(), Rounds(3))).fit(
            BooleanDataRepresentation(spec, X, y))


def test_a_rule_found_again_counts_as_a_round_but_appears_once():
    spec, X, y = _data()
    data = BooleanDataRepresentation(spec, X, y)
    # a gentle decay keeps the strongest rule best for several rounds
    model = CN2(target_class="pos", heuristic=WRAcc(),
                covering=WeightedCovering(MultiplicativeReweighting(0.9), Rounds(8))).fit(data)
    texts = _rules_text(model)
    assert len(texts) == len(set(texts)) < 8


@pytest.mark.parametrize("cls", [NListRepresentation, SparseDataRepresentation])
def test_weighted_covering_is_the_same_on_every_representation(cls):
    spec, X, y = _data()
    w = np.random.default_rng(5).random(len(y)) + 0.5
    ref = CN2(target_class="pos", covering=WeightedCovering()).fit(BooleanDataRepresentation(spec, X, y, weights=w))
    got = CN2(target_class="pos", covering=WeightedCovering()).fit(cls(spec, X, y, weights=w))
    assert _rules_text(got) == _rules_text(ref)


def test_cn2_sd_recovers_the_target_concept():
    # WRAcc + weighted covering (CN2-SD): the two conjunctions of the target
    spec, X, y = _data()
    model = CN2(target_class="pos", heuristic=WRAcc(), covering=WeightedCovering()).fit(
        BooleanDataRepresentation(spec, X, y))
    bodies = {tuple(sorted(l.feature for l in r.conditions)) for r in model.rules}
    assert bodies == {(0, 2), (4, 6)}                             # f0 & f1, f2 & f3 (negations interleave)


def test_weighted_covering_lets_rules_overlap_and_never_repeats_one():
    spec, X, y = _data()
    data = BooleanDataRepresentation(spec, X, y)
    removal = CN2(target_class="pos").fit(data)
    weighted = CN2(target_class="pos", covering=WeightedCovering()).fit(data)
    assert len(set(_rules_text(weighted))) == len(weighted.rules)
    cov = np.array([r.covers_data_packed(data) for r in weighted.rules])
    pos = y == "pos"
    assert (cov[:, pos].sum(axis=0) > 1).any()                    # some positive covered twice
    assert _rules_text(removal) != _rules_text(weighted)


def test_seed_covering_decision_list_refuses_weighted_covering():
    spec, X, y = _data()
    with pytest.raises(ValueError, match="RemovalCovering"):
        AQR(covering=WeightedCovering()).fit(BooleanDataRepresentation(spec, X, y), model=DecisionList)
