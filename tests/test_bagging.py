import numpy as np
import pytest

from pyrulearn.combiners import (
    CountVoteCombiner, HeuristicMaxCombiner, MacroVoteCombiner, MicroVoteCombiner,
)
from pyrulearn.data import HorizontalBitmapRepresentation
from pyrulearn.learners.bagging import Bagging, bootstrap_samples
from pyrulearn.learners.multiclass import OneVsRest
from pyrulearn.learners.seco import PFoil, PFossil, Pypper
from pyrulearn.models import (
    ConceptCascade, ConceptSet, DecisionList, DisjointRuleSet, EnsembleModel, FlatRuleSet, annotate_rules,
    ensemblemodel_to_flatruleset,
)
from pyrulearn.rule import Rule

from _negation_helpers import neg_spec, neg_X


def _data(n=400, seed=0, noise=0.15):
    rng = np.random.default_rng(seed)
    raw = rng.random((n, 6)) < 0.5
    y = np.where(raw[:, 0] & raw[:, 1] | raw[:, 2] & raw[:, 3], "pos", np.where(raw[:, 4], "neg", "mid"))
    flip = rng.random(n) < noise
    y = np.where(flip, rng.choice(["pos", "neg", "mid"], n), y)
    return HorizontalBitmapRepresentation(neg_spec([f"f{i}" for i in range(6)]), neg_X(raw), y)


def test_bootstrap_samples_weight_the_drawn_rows_by_their_draws():
    data = _data()
    samples = list(bootstrap_samples(data, 3, np.random.default_rng(0)))
    assert len(samples) == 3
    for s in samples:
        assert s.weights.sum() == data.n_samples
        assert s.n_samples < data.n_samples and np.all(s.weights >= 1)


@pytest.mark.parametrize("learner", [PFossil, PFoil], ids=lambda c: c.__name__)
def test_row_weights_learn_the_same_rules_as_duplicated_rows(learner):
    data = _data(n=80)
    counts = np.random.default_rng(1).integers(1, 4, data.n_samples)
    dup = HorizontalBitmapRepresentation(data.spec, np.repeat(data.X, counts, axis=0),
                                         np.repeat(data.y, counts))
    weighted = data.with_weights(counts)
    assert ([str(r) for r in learner().fit(dup).rules]
            == [str(r) for r in learner().fit(weighted).rules])


def test_bagging_builds_an_ensemble_of_members_learned_on_bootstrap_samples():
    data = _data()
    model = Bagging(PFossil(), n_estimators=5, random_state=3).fit(data)
    assert isinstance(model, EnsembleModel) and len(model.members) == 5
    assert isinstance(model.combiner, CountVoteCombiner)
    assert len({tuple(str(r) for r in m.rules) for m in model.members}) > 1
    again = Bagging(PFossil(), n_estimators=5, random_state=3).fit(data)
    assert list(again.predict(data)) == list(model.predict(data))
    assert "over the members' deciding rules" in model.to_string()
    flat = Bagging(PFossil(), n_estimators=5, random_state=3).fit(data, model=FlatRuleSet)
    assert type(flat) is FlatRuleSet and len(flat.rules) == len(model.rules)


def test_bagging_members_are_whatever_the_base_learner_builds():
    data = _data()
    concept_sets = Bagging(OneVsRest(Pypper()), n_estimators=3, random_state=0).fit(data)
    assert all(isinstance(m, ConceptSet) for m in concept_sets.members)
    cascades = Bagging(Pypper(), n_estimators=3, random_state=0).fit(data)
    assert all(isinstance(m, ConceptCascade) for m in cascades.members)


def test_without_abstain_cascades_vote_for_their_default_class():
    data = _data(n=50)
    # three members with rules for "pos" only, leaving "neg" to their default;
    # only the first covers the rows where feature 0 holds and feature 2 doesn't
    members = [DecisionList([Rule([f], target="pos", dataspec=data.spec)], default_prediction="neg")
               for f in (0, 2, 2)]
    ignored = EnsembleModel(members, default_prediction="mid", combiner=CountVoteCombiner())
    kept = EnsembleModel(members, default_prediction="mid", combiner=CountVoteCombiner(), abstain=False)
    row = int(np.flatnonzero(data.X[:, 0].astype(bool) & ~data.X[:, 2].astype(bool))[0])
    assert ignored.predict(data)[row] == "pos"   # the two others abstain
    assert kept.predict(data)[row] == "neg"      # the two others vote for their default
    assert Bagging(Pypper(), n_estimators=2).fit(data).abstain
    assert not Bagging(Pypper(), n_estimators=2, abstain=False).fit(data).abstain


@pytest.mark.parametrize("combiner", [CountVoteCombiner, MacroVoteCombiner, MicroVoteCombiner,
                                      HeuristicMaxCombiner], ids=lambda c: c.__name__)
def test_disjoint_members_give_the_same_at_the_member_and_the_rule_level(combiner):
    data = _data()
    spec = data.spec
    trees = []
    for f in range(4):   # stumps on one feature each: disjoint and exhaustive
        pos = Rule([2 * f], target="pos" if f % 2 else "neg", dataspec=spec)
        neg = Rule([2 * f + 1], target="mid", dataspec=spec)
        trees.append(DisjointRuleSet(annotate_rules([pos, neg], data)))
    ens = EnsembleModel(trees, default_prediction="pos", combiner=combiner())
    flat = ensemblemodel_to_flatruleset(ens, combiner=combiner())
    assert list(ens.predict(data)) == list(flat.predict(data))


def test_deciding_rules_of_a_decision_list_follow_the_list_order():
    data = _data(n=50)
    first, second = Rule([0], target="pos", dataspec=data.spec), Rule([2], target="neg", dataspec=data.spec)
    dl = DecisionList([first, second], default_prediction="mid")
    deciding = dl.deciding_rules(data)
    f0, f2 = data.X[:, 0].astype(bool), data.X[:, 2].astype(bool)
    for j in range(data.n_samples):
        expected = "pos" if f0[j] else ("neg" if f2[j] else None)
        assert (deciding[j].target if deciding[j] is not None else None) == expected
    assert dl.deciding_rules(data, with_default=True)[int(np.flatnonzero(~f0 & ~f2)[0])].target == "mid"


def test_member_weights_and_a_combiner_exclude_each_other():
    m = FlatRuleSet([Rule([0], target="pos")])
    with pytest.raises(ValueError):
        EnsembleModel([m], member_weights=[1.0], combiner=CountVoteCombiner())
