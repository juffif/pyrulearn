import numpy as np
import pytest

pytest.importorskip("realkd")

from pyrulearn.data import BooleanDataRepresentation  # noqa: E402
from pyrulearn.interfaces.realkd import RealkdRuleBoosting  # noqa: E402
from pyrulearn.learners.boosting import OptimalRuleBoosting  # noqa: E402
from pyrulearn.models import LinearRuleModel  # noqa: E402

from _negation_helpers import neg_spec, neg_X  # noqa: E402


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    y = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


@pytest.mark.parametrize("search", ["greedy", "exhaustive"])
@pytest.mark.parametrize("offset", [False, True])
def test_native_optimal_rule_boosting_equals_realkd(search, offset):
    data = _data()
    ref = RealkdRuleBoosting(n_rules=5, search=search, offset=offset).fit(data)
    native = OptimalRuleBoosting(n_rules=5, search=search, offset=offset).fit(data)
    assert isinstance(ref, LinearRuleModel)
    assert sorted(r.to_string() for r in ref.rules) == sorted(r.to_string() for r in native.rules)
    a, b = ref.scores(data), native.scores(data)
    np.testing.assert_allclose(a[:, 1] - a[:, 0], b[:, 1] - b[:, 0], atol=1e-9)


def test_realkd_rules_carry_provenance_and_stats():
    data = _data()
    model = RealkdRuleBoosting(n_rules=3).fit(data)
    assert all(r.provenance.source == "realkd.RuleBoostingEstimator" for r in model.rules)
    assert all(r.stats() is not None for r in model.rules)
    assert model.labels == ["neg", "pos"]
