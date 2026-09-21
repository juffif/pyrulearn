import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("fim")    # Borgelt's pyfim -- pyarc refuses to import without it
pytest.importorskip("pyarc")

from pyarc import CBA as PyarcRawCBA
from pyarc.data_structures import TransactionDB

from pyrulearn.data import BooleanDataRepresentation, DataSpec
from pyrulearn.interfaces.pyarc import PyarcCBA, PyarcCBAImporter
from pyrulearn.learners.associative import CBA
from pyrulearn.models import DecisionList

from _negation_helpers import neg_spec, neg_X


def _rep(X, y):
    X = np.asarray(X, dtype=bool)
    return BooleanDataRepresentation(DataSpec([f"f{i}" for i in range(X.shape[1])]), X, np.asarray(y))


def _hand_worked():
    # the same dataset the native CBA-CB tests hand-derive: (f0)->a is
    # perfect, (f1)->a is wrong on both rows it newly covers (5, 6 are
    # 'b'), so CBA-CB skips it; the default is 'b'.
    X = [[1, 1]] * 3 + [[1, 0]] * 2 + [[0, 1]] * 2 + [[0, 0]] * 1
    y = ["a"] * 3 + ["a"] * 2 + ["b"] * 2 + ["b"] * 1
    return _rep(X, y)


def test_pyarc_cba_fit_returns_a_decision_list_matching_the_native_cba_on_a_hand_worked_dataset():
    data = _hand_worked()
    external = PyarcCBA(min_support=0.01, min_confidence=0.5, max_len=1).fit(data)
    native = CBA(min_support=0.01, min_confidence=0.5, max_len=1).fit(data)

    assert isinstance(external, DecisionList)
    assert [r.conditions[0].feature for r in external.rules] == [0]
    assert external.default_prediction == "b"
    assert list(external.predict(data)) == list(native.predict(data))
    print("PyarcCBA and the native CBA agree on the hand-worked dataset: OK")


def test_pyarc_cba_stamps_provenance_and_measured_stats():
    data = _hand_worked()
    model = PyarcCBA(max_len=1).fit(data)
    assert model.provenance.learner == "PyarcCBA"
    assert model.provenance.params["max_len"] == 1
    assert all(r.stats() is not None for r in model.rules)          # annotated by the fit round trip
    assert all(r.provenance.source == "pyarc.CBA" for r in model.rules)
    print("PyarcCBA stamps learner provenance, per-rule source, and training stats: OK")


def test_importer_reproduces_a_fitted_pyarc_models_own_predictions():
    import pandas as pd

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(150, 5)).astype(bool)
    y = np.where(X[:, 0] & ~X[:, 1], "x", np.where(X[:, 2], "y", "z"))
    frame = pd.DataFrame(np.where(X, 1.0, np.nan), columns=[f"f{i}" for i in range(5)])
    frame["class"] = y
    txn = TransactionDB.from_DataFrame(frame, target="class")
    raw = PyarcRawCBA(support=0.05, confidence=0.5, maxlen=3).fit(txn)

    data = _rep(X, y)
    imported = PyarcCBAImporter().import_model(raw, data.spec, data=data)
    assert list(imported.predict(data)) == list(raw.predict(txn))
    print("PyarcCBAImporter reproduces the fitted pyarc model's predictions exactly: OK")


def test_pyarc_cba_agrees_with_the_native_cba_on_a_multiclass_problem_with_negation():
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 2, size=(240, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x", np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    data = BooleanDataRepresentation(neg_spec(list("abcde")), neg_X(raw), y)

    external = PyarcCBA(min_support=0.05, min_confidence=0.5, max_len=3).fit(data)
    native = CBA(min_support=0.05, min_confidence=0.5, max_len=3).fit(data)
    acc_ext = float(np.mean(np.asarray(external.predict(data)) == y))
    acc_nat = float(np.mean(np.asarray(native.predict(data)) == y))
    assert acc_ext > 0.9 and acc_nat > 0.9
    assert abs(acc_ext - acc_nat) < 0.03, (acc_ext, acc_nat)
    print(f"PyarcCBA (acc {acc_ext:.3f}) and native CBA (acc {acc_nat:.3f}) agree on a 3-class problem: OK")


def test_max_len_means_the_rule_body_length_not_pyarcs_class_inclusive_maxlen():
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 2, size=(240, 5)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "x", np.where(raw[:, 2] & raw[:, 3], "y", "z"))
    data = BooleanDataRepresentation(neg_spec(list("abcde")), neg_X(raw), y)
    for L in (1, 2):
        model = PyarcCBA(min_support=0.02, max_len=L).fit(data)
        assert max(len(r.conditions) for r in model.rules) <= L
    print("PyarcCBA(max_len=L) bounds the rule body at L (pyarc's maxlen=L+1 handled internally): OK")


def test_importer_rejects_an_unfitted_model():
    with pytest.raises(ValueError, match="not fitted"):
        PyarcCBAImporter().import_model(PyarcRawCBA(), DataSpec(["a", "b"]))
    print("PyarcCBAImporter rejects an unfitted pyarc model: OK")
