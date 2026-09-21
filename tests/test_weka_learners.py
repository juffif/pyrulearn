"""`JRip` / `PART` / `J48` as `fit()` learners (the subprocess round-trip),
plus the `placeholder_features=True` importer mode they rely on.

The end-to-end fit tests need a real `weka.jar` -- set `$WEKA_JAR` (and,
optionally, `$WEKA_JAVA`) to run them; they skip otherwise. The
placeholder-parse tests use captured `f0..fN` stdout and always run.
"""

import os

import numpy as np
import pytest

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.interfaces.weka import (
    J48,
    J48Importer,
    JRip,
    JRipImporter,
    PART,
    PARTImporter,
)
from pyrulearn.models import (
    ConceptCascade,
    ConceptSet,
    DecisionList,
    DisjointRuleSet,
    FlatRuleSet,
    PairwiseModel,
    SingleRule,
)

from _negation_helpers import neg_spec, neg_X

_HAS_WEKA = bool(os.environ.get("WEKA_JAR"))

# A `fit()` round-trip writes ds's already-Boolean columns as placeholder
# f0..fN {False,True} ARFF attributes; this is what Weka then prints back.
_DS = neg_spec(["a", "b", "c"])  # features a,not a,b,not b,c,not c -> 0..5

_JRIP_PLACEHOLDER = """JRIP rules:
===========

(f0 = True) and (f3 = True) => class=pos (81.0/2.0)
(f4 = False) => class=pos (32.0/1.0)
 => class=neg (287.0/15.0)

Number of Rules : 3

=== Evaluation on training set ===
"""

_PART_PLACEHOLDER = """PART decision list
------------------

f0 = True AND
f2 = False: neg (125.0/8.0)

f4 = True: pos (100.0/3.0)

: pos (13.0)

Number of Rules  : 3

=== Error on training data ===
"""

_J48_PLACEHOLDER = """J48 pruned tree
------------------

f0 = False: neg (40.0/1.0)
f0 = True
|   f4 = True: pos (73.0/2.0)
|   f4 = False: neg (30.0/5.0)

Number of Leaves  : 3

=== Stratified cross-validation ===
"""


def test_jrip_placeholder_parse_binds_by_position():
    rl = JRipImporter(dataspec=_DS, placeholder_features=True).parse(_JRIP_PLACEHOLDER)
    assert [sorted(l.feature for l in r.conditions) for r in rl.rules] == [[0, 3], [5]]
    # f4 = False -> negative literal -> lands on the paired negation feature (not c = 5)
    assert rl.default_prediction == "neg"
    assert rl.rules[0].provenance.source == "weka.classifiers.rules.JRip"


def test_part_placeholder_parse_binds_by_position():
    rl = PARTImporter(dataspec=_DS, placeholder_features=True).parse(_PART_PLACEHOLDER)
    assert [sorted(l.feature for l in r.conditions) for r in rl.rules] == [[0, 3], [4]]
    assert rl.default_prediction == "pos"


def test_j48_placeholder_parse_binds_by_position():
    drs = J48Importer(dataspec=_DS, placeholder_features=True).parse(_J48_PLACEHOLDER)
    assert [sorted(l.feature for l in r.conditions) for r in drs.rules] == [[1], [0, 4], [0, 5]]


def test_placeholder_features_requires_a_dataspec():
    with pytest.raises(ValueError, match="requires an explicit dataspec"):
        JRipImporter(placeholder_features=True).parse(_JRIP_PLACEHOLDER)


@pytest.mark.parametrize("cls, native", [
    (JRip, DecisionList), (PART, DecisionList), (J48, DisjointRuleSet),
])
def test_weka_learner_capability_set(cls, native):
    caps = cls().produces()
    assert native in caps
    assert {ConceptSet, ConceptCascade, PairwiseModel, FlatRuleSet} <= caps


def test_only_jrip_has_a_single_rule_producer():
    # JRip processes one class fully before the next, so "the first rule
    # in JRip's own run for label X" is exactly what X's covering-loop
    # segment would find first -- a genuine single-rule computation, not
    # a pick among many. PART's rounds build a whole partial tree and pick
    # its best leaf for whichever class wins, not a chosen target -- no
    # such computation exists. J48 (a tree) has no covering process at all.
    assert SingleRule in JRip().produces()
    assert SingleRule not in PART().produces()
    assert SingleRule not in J48().produces()


def test_weka_learner_without_a_jar_raises_a_clear_error(monkeypatch):
    monkeypatch.delenv("WEKA_JAR", raising=False)
    rep = BooleanDataRepresentation(_DS, neg_X(np.zeros((4, 3), dtype=bool)),
                                    np.array(["a", "b", "a", "b"]))
    with pytest.raises(RuntimeError, match="WEKA_JAR"):
        JRip().fit(rep)


def _synth_rep(n=200, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n, 4)).astype(bool)
    y = np.where((raw[:, 0] & ~raw[:, 1]) | raw[:, 2], "pos", "neg")
    ds = neg_spec([f"x{i}" for i in range(4)])
    return BooleanDataRepresentation(ds, neg_X(raw), y), y


@pytest.mark.skipif(not _HAS_WEKA, reason="$WEKA_JAR not set")
@pytest.mark.parametrize("cls, native", [
    (JRip, DecisionList), (PART, DecisionList), (J48, DisjointRuleSet),
])
def test_weka_learner_fits_end_to_end(cls, native):
    rep, y = _synth_rep()
    model = cls().fit(rep)
    assert type(model) is native
    acc = float(np.mean(np.asarray(model.predict(rep)) == y))
    assert acc > 0.8, f"{cls.__name__} train accuracy only {acc:.3f}"


@pytest.mark.skipif(not _HAS_WEKA, reason="$WEKA_JAR not set")
def test_weka_learner_decomposition_path_end_to_end():
    rep, _ = _synth_rep()
    cs = JRip().fit(rep, model=ConceptSet)
    assert type(cs) is ConceptSet and len(cs.concepts) == 2


@pytest.mark.skipif(not _HAS_WEKA, reason="$WEKA_JAR not set")
def test_jrip_single_rule_is_the_first_rule_of_that_labels_own_segment():
    # 3 classes so there's a genuine non-default, non-trivial label to probe,
    # plus the majority/default class (gets zero explicit rules from JRip).
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 2, size=(300, 4)).astype(bool)
    y = np.where(raw[:, 0] & ~raw[:, 1], "pos",
                np.where(raw[:, 2] & raw[:, 3], "q", "neg"))
    ds = neg_spec([f"x{i}" for i in range(4)])
    rep = BooleanDataRepresentation(ds, neg_X(raw), y)

    full = JRip().fit(rep)
    for label in ("pos", "q", "neg"):
        sr = JRip().fit(rep, model=SingleRule, label=label)
        assert type(sr) is SingleRule
        assert sr.target == label
        first_in_full = next((r for r in full.rules if r.target == label), None)
        if first_in_full is None:
            assert len(sr.conditions) == 0    # label is JRip's own trailing default
        else:
            assert sr.pos == first_in_full.pos  # exactly the first rule of that segment

    with pytest.raises(ValueError, match="label"):
        JRip().fit(rep, model=SingleRule)
