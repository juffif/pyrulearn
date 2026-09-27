import numpy as np
import pytest

pytest.importorskip("mlrl.boosting")

from pyrulearn.data import BooleanDataRepresentation  # noqa: E402
from pyrulearn.interfaces.boomer import MlrlBoomer  # noqa: E402
from pyrulearn.models import LinearRuleModel  # noqa: E402

from _negation_helpers import neg_spec, neg_X  # noqa: E402


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", "neg")
    y = np.where(rng.random(n) < 0.1, np.where(y == "pos", "neg", "pos"), y)
    return BooleanDataRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


@pytest.mark.parametrize("params", [dict(max_rules=10), dict(max_rules=40, l2_regularization_weight=1.0)])
def test_boomer_import_reproduces_its_decision_function(params):
    data = _data()
    learner = MlrlBoomer(random_state=0, **params)
    model = learner.fit(data)
    assert isinstance(model, LinearRuleModel) and model.labels == ["neg", "pos"]
    fitted = learner.fit_external(data.X, data.y)
    dec = np.asarray(fitted.decision_function(data.X.astype(float))).ravel()
    s = model.scores(data)
    np.testing.assert_allclose(s[:, 1] - s[:, 0], dec, atol=1e-4)
    assert all(r.weight > 0 for r in model.rules)
    assert all(r.provenance.source == "mlrl.boosting.BoomerClassifier" for r in model.rules)


def test_boomer_is_binary_only_here():
    rng = np.random.default_rng(1)
    raw = rng.random((60, 3)) < 0.5
    three = BooleanDataRepresentation(neg_spec(["a", "b", "c"]), neg_X(raw), rng.choice(["x", "y", "z"], 60))
    with pytest.raises(ValueError, match="binary"):
        MlrlBoomer(max_rules=5).fit(three)
