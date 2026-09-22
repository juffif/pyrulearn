import numpy as np
import pytest

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data import DataSpec, DataSpecBuilder, merge_dataspecs
from pyrulearn.rule import Rule, WeightedRule
from pyrulearn.models import (
    CompositeModel, ConceptCascade, ConceptModel, ConceptSet, DecisionList,
    DeepModel, DefaultPrediction, DisjointRuleSet, EnsembleModel, FlatRuleSet, MajorityClass,
    PairwiseModel, Provenance, RuleList, RuleModel, RuleSet, SingleRule,
    can_convert, conceptcascade_to_decision_list, conceptset_to_flatruleset, convert,
    ensemblemodel_to_flatruleset,
    flatruleset_to_conceptset, flatruleset_to_decision_list,
    annotate_default_rule, annotate_rules,
)
from pyrulearn.evaluation import RuleStats

DS = DataSpec(["p", "q", "r"])
#            p  q  r
X = np.array([[1, 0, 0],   # 0
              [0, 1, 0],   # 1
              [1, 1, 0],   # 2
              [0, 0, 1],   # 3
              [0, 0, 0]],  # 4
             dtype=bool)
Y = np.array(["a", "b", "a", "b", "a"], dtype=object)
DATA = BooleanDataRepresentation(DS, X, Y)


# --------------------------------------------------------------- WeightedRule ---

def test_weighted_rule_is_rule_plus_one_declarative_field():
    wr = WeightedRule([0], target="a", dataspec=DS, weight=0.8)
    assert wr.weight == 0.8
    assert wr == Rule([0], target="a", dataspec=DS)             # weight not in identity
    assert hash(wr) == hash(Rule([0], target="a"))
    assert {wr, Rule([0], target="a")} == {wr}


def test_weighted_rule_printing():
    wr = WeightedRule([0, 2], target="a", dataspec=DS, weight=0.75)
    assert wr.to_string("prolog") == "0.75::a(X) :- p(X), r(X)."
    assert wr.to_string("logic", ascii=True) == "p AND r -> a  [0.75]"
    assert repr(wr).startswith("WeightedRule(")


def test_weighted_rule_structural_copies_keep_weight():
    wr = WeightedRule([0, 1], target="a", dataspec=DS, weight=0.9, ordered=False)
    assert wr.reorder([1, 0]).weight == 0.9
    assert wr.remap(DataSpec(["x", "p", "q", "r"])).weight == 0.9
    child, _ = wr.specialize()[0]
    assert type(child) is Rule


# ---------------------------------------------------- hierarchy: the two axes ---

def test_family_membership():
    ds = DS
    r = [Rule([0], target="a", dataspec=ds)]
    assert isinstance(FlatRuleSet(r), RuleSet) and not isinstance(FlatRuleSet(r), RuleList)
    assert isinstance(ConceptModel(r), RuleSet)
    assert isinstance(ConceptSet.from_rules(r), RuleSet)
    assert isinstance(DisjointRuleSet(r), RuleSet)
    assert isinstance(DecisionList(r), RuleList) and not isinstance(DecisionList(r), RuleSet)
    assert isinstance(ConceptCascade([ConceptModel(r, label="a")]), RuleList)
    assert isinstance(EnsembleModel([SingleRule(r[0])]), CompositeModel)
    # RuleSet / RuleList are abstract family bases -- not instantiable
    with pytest.raises(TypeError):
        RuleSet()
    with pytest.raises(TypeError):
        RuleList()


# ------------------------------------------------------------------ SingleRule ---

def test_single_rule_predicts_head_or_default():
    m = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction="b")
    assert list(m.predict(DATA)) == ["a", "b", "a", "b", "b"]
    assert m.rule.target == "a" and len(m) == 1
    assert type(m.filter("a")) is SingleRule
    assert type(m.filter("z")) is FlatRuleSet and len(m.filter("z")) == 0


# ---------------------------------------------------------------- ConceptModel ---

def test_concept_model_predicts_label_or_none():
    cm = ConceptModel([Rule([0], target="a", dataspec=DS), Rule([1], target="a", dataspec=DS)])
    assert cm.label == "a"
    assert list(cm.predict(DATA)) == ["a", "a", "a", None, None]
    assert cm.default_prediction is None
    assert len(cm.filter("a")) == 2 and len(cm.filter("b")) == 0


def test_concept_model_rejects_mixed_heads():
    with pytest.raises(ValueError):
        ConceptModel([Rule([0], target="a"), Rule([1], target="b")])


# ------------------------------------------------------------------ FlatRuleSet ---

def test_flat_rule_set_resolves_by_combiner_and_swaps():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)],
                     default_prediction="a", combiner="list")
    assert list(rs.predict(DATA)) == ["a", "b", "a", "a", "a"]  # row2 both -> list: first
    assert rs.filter("a").combiner == "list"
    rs.combiner = "vote"
    assert set(np.unique(rs.predict(DATA))) <= {"a", "b"}
    assert type(rs.remap(DataSpec(["p", "q", "r", "s"]))) is FlatRuleSet


# ------------------------------------------------------------------ DecisionList ---

def test_decision_list_is_first_match():
    dl = DecisionList([Rule([0], target="a", dataspec=DS), Rule([], target="b", dataspec=DS)],
                      default_prediction=None)
    assert list(dl.predict(DATA)) == ["a", "b", "a", "b", "b"]


# --------------------------------------------------------------- DisjointRuleSet ---

def test_disjoint_rule_set_predicts_the_one_covering_rule():
    drs = DisjointRuleSet([Rule([0], target="a", dataspec=DS), Rule([2], target="b", dataspec=DS)],
                          default_prediction="a")
    assert list(drs.predict(DATA)) == ["a", "a", "a", "b", "a"]


# ------------------------------------------------------------------- ConceptSet ---

def test_concept_set_blocks_and_matches_flat_when_no_conflict():
    rules = [Rule([0], target="a", dataspec=DS), Rule([2], target="b", dataspec=DS)]
    cs = ConceptSet.from_rules(rules, default_prediction="a", combiner="max")
    assert sorted(cs.concept_labels) == ["a", "b"]
    assert len(cs.concept_for("a")) == 1
    assert cs.labels == ["a", "b"]
    flat = FlatRuleSet(rules, default_prediction="a", combiner="max")
    assert list(cs.predict(DATA)) == list(flat.predict(DATA)) == ["a", "a", "a", "b", "a"]
    assert len(cs.filter("b").concepts) == 1
    cs.combiner = "vote"
    assert cs.filter("a").combiner == "vote"


# --------------------------------------------------------------- ConceptCascade ---

def test_concept_cascade_first_firing_concept_wins():
    a = ConceptModel([Rule([0], target="a", dataspec=DS)], label="a")
    b = ConceptModel([Rule([1], target="b", dataspec=DS)], label="b")
    assert list(ConceptCascade([a, b], default_prediction="b").predict(DATA)) == ["a", "b", "a", "b", "b"]
    assert list(ConceptCascade([b, a], default_prediction="a").predict(DATA))[2] == "b"


def test_concept_cascade_unique_mask_credits_the_whole_first_concept():
    # two rules in concept "a", both cover row2 [p,q] (p and p&q). _unique_mask
    # is no longer wired into .stats() (per-rule breakdown was dropped in favor
    # of one whole-model ConfusionMatrix -- see test_evaluation.py for that), but
    # the underlying resolution logic is still there and still worth testing
    # directly: rule-level credit, not just concept-level.
    a = ConceptModel([Rule([0], target="a", dataspec=DS), Rule([0, 1], target="a", dataspec=DS)], label="a")
    b = ConceptModel([Rule([1], target="b", dataspec=DS)], label="b")
    casc = ConceptCascade([a, b], default_prediction="b")
    cov = casc.coverage_matrix(DATA)
    um = casc._unique_mask(cov)
    # row2 -> concept "a" fires first; BOTH of a's covering rules get unique credit
    assert um[0, 2] and um[1, 2]


# ------------------------------------------------------------------------ stats ---

def test_stats_is_none_until_annotated_then_returns_a_model_stats():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="a", dataspec=DS)],
                     default_prediction="a")
    assert rs.stats() is None
    st = rs.stats(DATA)
    assert type(st).__name__ == "ModelStats"
    assert st.n_rows == 5
    assert st.n_rules == 2
    assert st.n_conditions == 2   # one condition per rule
    assert st.confusion is not None
    assert st.confusion.labels == ["a", "b"]
    assert 0.0 <= st.confusion.accuracy <= 1.0
    assert rs.stats(split="data") is st   # cached, same object back


def test_stats_confusion_is_none_without_labels():
    rep_no_y = BooleanDataRepresentation(DS, X)  # no labels available to score a confusion matrix against
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="a")
    st = rs.stats(rep_no_y)
    assert st.confusion is None
    assert st.n_rules == 1


def test_majority_class_default_and_distribution():
    m = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction=MajorityClass(DATA))
    assert list(m.predict(DATA)) == ["a", "a", "a", "a", "a"]
    dist = m.predict_distribution(DATA)
    assert dist.shape == (5, 1) and np.allclose(dist, 1.0)


# -------------------------------------------------------------------- composite ---

def test_ensemble_model_votes_over_members():
    m1 = SingleRule(Rule([0], target="a", dataspec=DS), default_prediction=None)   # p -> a
    m2 = SingleRule(Rule([1], target="b", dataspec=DS), default_prediction=None)   # q -> b
    m3 = SingleRule(Rule([], target="a", dataspec=DS), default_prediction=None)    # TRUE -> a
    ens = EnsembleModel([m1, m2, m3], default_prediction="b")
    assert list(ens.predict(DATA)) == ["a", "b", "a", "a", "a"]   # row1: tie -> earlier member (m2) -> b
    assert isinstance(ens, CompositeModel) and len(ens.rules) == 3
    with pytest.raises(TypeError):
        ens.add(Rule([0], target="a"))
    assert "3 members" in repr(ens)
    ensw = EnsembleModel([m1, m2, m3], member_weights=[1, 5, 1], default_prediction="b")
    assert list(ensw.predict(DATA))[2] == "b"   # m2->b weighted 5 beats m1+m3 -> a


def test_pairwise_model_round_robin_vote():
    ab = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    ac = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="c")
    bc = FlatRuleSet([Rule([1], target="b", dataspec=DS)], default_prediction="c")
    pm = PairwiseModel([("a", "b", ab), ("a", "c", ac), ("b", "c", bc)], default_prediction="a")
    assert pm.pairs == [("a", "b"), ("a", "c"), ("b", "c")]
    assert list(pm.predict(DATA))[:2] == ["a", "b"]
    assert type(pm.filter("a")) is PairwiseModel and pm.filter("a").pairs == pm.pairs


def test_deep_model_is_a_stub():
    dm = DeepModel([SingleRule(Rule([0], target="a", dataspec=DS))], dependencies={0: []})
    assert isinstance(dm, CompositeModel)
    with pytest.raises(NotImplementedError):
        dm.predict(DATA)


# ------------------------------------------------------------------- converters ---

def test_converters_and_can_convert():
    rs = FlatRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)],
                     default_prediction="a")
    assert can_convert(FlatRuleSet, DecisionList) and not can_convert(FlatRuleSet, ConceptCascade)
    assert type(flatruleset_to_decision_list(rs)) is DecisionList
    assert sorted(flatruleset_to_conceptset(rs).concept_labels) == ["a", "b"]
    assert type(convert(rs, DecisionList)) is DecisionList
    with pytest.raises(TypeError):
        convert(rs, ConceptCascade)
    casc = ConceptCascade([ConceptModel([Rule([0], target="a", dataspec=DS)], label="a")],
                          default_prediction="b")
    assert type(conceptcascade_to_decision_list(casc)) is DecisionList

    ens = EnsembleModel(
        [DisjointRuleSet([Rule([0], target="a", dataspec=DS), Rule([1], target="b", dataspec=DS)]),
         DisjointRuleSet([Rule([2], target="b", dataspec=DS)])],
        default_prediction="a",
    )
    assert can_convert(EnsembleModel, FlatRuleSet)
    flat = ensemblemodel_to_flatruleset(ens)
    assert type(flat) is FlatRuleSet and len(flat.rules) == 3
    assert flat.default_prediction == "a" and flat.combiner == "vote"
    assert type(convert(ens, FlatRuleSet)) is FlatRuleSet


# ------------------------------------------------------------------- provenance ---

def test_provenance_defaults_to_none_and_is_a_plain_settable_attribute():
    m = FlatRuleSet([Rule([0], target="a", dataspec=DS)])
    assert m.provenance is None
    m.provenance = Provenance(learner="CN2", params={"beam_width": 5}, source=None)
    assert m.provenance.learner == "CN2"
    assert m.provenance.params == {"beam_width": 5}
    assert m.provenance.source is None


def test_provenance_equality_and_repr():
    p1 = Provenance(learner="JRip", params={"F": 3}, source="weka.classifiers.rules.JRip")
    p2 = Provenance(learner="JRip", params={"F": 3}, source="weka.classifiers.rules.JRip")
    assert p1 == p2
    assert p1 != Provenance(learner="PART", params={"F": 3}, source="weka.classifiers.rules.JRip")
    assert "JRip" in repr(p1) and "weka.classifiers.rules.JRip" in repr(p1)
    assert "source" not in repr(Provenance(learner="CN2", params={}))  # no source -> omitted


# --------------------------------------------- provenance through filter/remap/convert ---

def _rules_with_provenance():
    """Two SingleRules, each its own fresh Provenance instance (matching
    the importers' per-rule stamping convention), targeting "a" and "b"."""
    prov_a = Provenance(learner="CN2", params={"target": "a"})
    prov_b = Provenance(learner="CN2", params={"target": "b"})
    ra = SingleRule(Rule([0], target="a", dataspec=DS))
    ra.provenance = prov_a
    rb = SingleRule(Rule([1], target="b", dataspec=DS))
    rb.provenance = prov_b
    return ra, rb, prov_a, prov_b


def test_flat_rules_filter_and_remap_carry_container_and_rule_provenance():
    ra, rb, prov_a, prov_b = _rules_with_provenance()
    fs = FlatRuleSet([ra, rb])
    fs.provenance = Provenance(learner="RandomForest", params={"n_estimators": 5})

    filtered = fs.filter("a")
    assert filtered.provenance is fs.provenance          # container-level: carried, same object
    assert filtered.rules[0].provenance is prov_a         # per-rule: same object (no rebuild)

    ds2 = DataSpec(["p", "q", "r"])                       # same names -> valid remap target
    remapped = fs.remap(ds2)
    assert remapped.provenance is fs.provenance           # container-level: carried
    assert remapped.rules[0].provenance is prov_a         # per-rule: carried across the rebuild
    assert remapped.rules[1].provenance is prov_b
    assert remapped.rules[0].dataspec is ds2              # actually remapped, not just copied
    print("FlatRuleSet.filter/remap carry both container- and rule-level provenance: OK")


def test_concept_model_filter_and_remap_carry_provenance():
    ra, _, prov_a, _ = _rules_with_provenance()
    cm = ConceptModel([ra], label="a")
    cm.provenance = Provenance(learner="SeCo", params={"target_class": "a"})

    ds2 = DataSpec(["p", "q", "r"])
    assert cm.filter("a").provenance is cm.provenance
    remapped = cm.remap(ds2)
    assert remapped.provenance is cm.provenance
    assert remapped.rules[0].provenance is prov_a
    print("ConceptModel.filter/remap carry provenance: OK")


def test_single_rule_filter_and_remap_carry_provenance():
    ra, _, prov_a, _ = _rules_with_provenance()
    assert ra.filter("a") is ra                           # matches -> same object, trivially preserved
    assert ra.filter("a").provenance is prov_a

    empty = ra.filter("zzz")                              # no match -> a fresh empty FlatRuleSet
    assert empty.provenance is prov_a                     # still carried, for consistency

    ds2 = DataSpec(["p", "q", "r"])
    remapped = ra.remap(ds2)
    assert remapped is not ra                             # remap always rebuilds
    assert remapped.provenance is prov_a
    print("SingleRule.filter/remap carry provenance: OK")


def test_concept_indexed_filter_and_remap_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    cs = ConceptSet.from_rules([ra, rb])
    cs.provenance = Provenance(learner="OneVsRest", params={})

    ds2 = DataSpec(["p", "q", "r"])
    assert cs.filter("a").provenance is cs.provenance
    assert cs.remap(ds2).provenance is cs.provenance

    cascade = ConceptCascade([ConceptModel([ra], label="a"), ConceptModel([rb], label="b")])
    cascade.provenance = Provenance(learner="OrderedOneVsRest", params={})
    assert cascade.filter("a").provenance is cascade.provenance
    assert cascade.remap(ds2).provenance is cascade.provenance
    print("ConceptSet/ConceptCascade filter/remap carry provenance: OK")


def test_composite_filter_and_remap_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    ens = EnsembleModel([SingleRule(ra.rule), SingleRule(rb.rule)])
    ens.provenance = Provenance(learner="RandomForest", params={})

    ds2 = DataSpec(["p", "q", "r"])
    assert ens.filter("a").provenance is ens.provenance
    assert ens.remap(ds2).provenance is ens.provenance
    print("CompositeModel (EnsembleModel) filter/remap carry provenance: OK")


def test_converters_carry_provenance():
    ra, rb, _, _ = _rules_with_provenance()
    fs = FlatRuleSet([ra, rb])
    fs.provenance = Provenance(learner="RandomForest", params={})
    assert flatruleset_to_decision_list(fs).provenance is fs.provenance
    assert flatruleset_to_conceptset(fs).provenance is fs.provenance

    cs = ConceptSet.from_rules([ra, rb])
    cs.provenance = Provenance(learner="OneVsRest", params={})
    assert conceptset_to_flatruleset(cs).provenance is cs.provenance

    ens = EnsembleModel([SingleRule(ra.rule), SingleRule(rb.rule)])
    ens.provenance = Provenance(learner="RandomForest", params={})
    assert ensemblemodel_to_flatruleset(ens).provenance is ens.provenance

    cascade = ConceptCascade([ConceptModel([ra], label="a"), ConceptModel([rb], label="b")])
    cascade.provenance = Provenance(learner="OrderedOneVsRest", params={})
    assert conceptcascade_to_decision_list(cascade).provenance is cascade.provenance
    print("Every model->model converter carries provenance across: OK")


# ---------------------------------------------------- SingleRule containers ---

def test_default_rule_is_a_single_rule_with_its_own_stats_and_provenance():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    dr = fs.default_rule
    assert type(dr) is SingleRule
    assert dr.target == "b" and dr.conditions == ()
    assert dr.stats() is None                   # nothing measured yet
    dr.stats(DATA)                               # fall-through stats land here, like any other leaf
    assert fs.default_rule is dr and fs.default_rule.stats() is not None
    assert fs.default_rule.provenance is None   # settable, just like any other RuleModel


def test_flat_containers_store_single_rule_not_bare_rule():
    r1 = Rule([0], target="a", dataspec=DS)
    r2 = WeightedRule([1], target="b", dataspec=DS, weight=0.7)
    fs = FlatRuleSet([r1, r2])
    assert all(type(r) is SingleRule for r in fs.rules)
    assert [r.rule for r in fs.rules] == [r1, r2]


def test_single_rule_forwards_reads_to_the_wrapped_rule():
    r = WeightedRule([0, 1], target="a", dataspec=DS, weight=0.42)
    sr = SingleRule(r)
    assert sr.target == "a"
    assert sr.conditions == r.conditions
    assert sr.dataspec is DS
    assert sr.pos == r.pos
    assert sr.weight == 0.42
    assert sr.to_string("logic") == r.to_string("logic")
    assert sr.rules == [sr]           # recursion base case


def test_single_rule_weight_is_settable_through_the_wrapper():
    r = WeightedRule([0], target="a", dataspec=DS, weight=0.5)
    fs = FlatRuleSet([r])
    fs.rules[0].weight = 0.05
    assert r.weight == 0.05           # mutation reaches the real underlying rule
    assert fs.rules[0].weight == 0.05  # and is visible again through the same wrapper


def test_single_rule_missing_attribute_still_raises_attribute_error():
    sr = SingleRule(Rule([0], target="a", dataspec=DS))
    with pytest.raises(AttributeError):
        sr.this_attribute_does_not_exist_anywhere


def test_filter_and_remap_keep_producing_single_rule_wrapped_containers():
    r1 = Rule([0], target="a", dataspec=DS)
    r2 = Rule([1], target="b", dataspec=DS)
    fs = FlatRuleSet([r1, r2])

    filtered = fs.filter("a")
    assert all(type(r) is SingleRule for r in filtered.rules)

    ds2 = DataSpec(["p", "q", "r"])
    remapped = fs.remap(ds2)
    assert all(type(r) is SingleRule for r in remapped.rules)
    assert remapped.rules[0].dataspec is ds2


def test_all_are_rule_models():
    ds = DS
    r = [Rule([0], target="a", dataspec=ds)]
    for m in (SingleRule(r[0]), ConceptModel(r), FlatRuleSet(r), DisjointRuleSet(r),
              DecisionList(r), ConceptSet.from_rules(r),
              ConceptCascade([ConceptModel(r, label="a")]),
              EnsembleModel([SingleRule(r[0])]),
              PairwiseModel([("a", "b", FlatRuleSet(r))])):
        assert isinstance(m, RuleModel)


# ------------------------------------------------------------ annotate_rules ---

def test_annotate_rules_wraps_and_populates_stats_against_given_data():
    r1 = Rule([0], target="a", dataspec=DS)          # p -> a: covers rows 0, 2
    r2 = Rule([2], target="b", dataspec=DS)          # r -> b: covers row 3
    out = annotate_rules([r1, r2], DATA)
    assert all(type(sr) is SingleRule for sr in out)
    for sr, r in zip(out, [r1, r2]):
        expected = RuleStats.from_rule(r, DATA, r.target)
        got = sr.stats().confusion.rule_stats(r.target)
        assert (got.tp, got.fp, got.fn, got.tn) == (expected.tp, expected.fp, expected.fn, expected.tn)


def test_annotate_rules_is_idempotent_on_an_already_single_rule():
    sr = SingleRule(Rule([0], target="a", dataspec=DS))
    out = annotate_rules([sr], DATA)
    assert out[0] is sr                    # not re-wrapped
    assert out[0].stats() is not None


def test_annotate_rules_with_data_none_skips_stats():
    r = Rule([0], target="a", dataspec=DS)
    out = annotate_rules([r], None)
    assert type(out[0]) is SingleRule       # still wrapped
    assert out[0].stats() is None           # but nothing measured


def test_annotate_default_rule_populates_the_materialized_default_rules_stats():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)], default_prediction="b")
    result = annotate_default_rule(fs, DATA)
    assert result is fs                     # returned unchanged, for chaining
    dr_stats = fs.default_rule.stats()
    assert dr_stats is not None and dr_stats.confusion is not None


def test_annotate_default_rule_is_a_noop_when_there_is_no_default_rule():
    fs = FlatRuleSet([Rule([0], target="a", dataspec=DS)])  # default_prediction=None -> abstain
    assert fs.default_rule is None
    result = annotate_default_rule(fs, DATA)                # must not raise
    assert result is fs


def test_stats_multiple_splits_coexist():
    X_train = np.array([[1, 0], [1, 1], [0, 0]], dtype=bool)
    X_test = np.array([[1, 0], [0, 0]], dtype=bool)
    ds = DataSpec(["a", "b"])
    train_rep = BooleanDataRepresentation(ds, X_train)
    test_rep = BooleanDataRepresentation(ds, X_test)
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    fs = FlatRuleSet([r])

    fs.annotate(train_rep, split="train")
    fs.annotate(test_rep, split="test")
    assert fs.stats(split="train").n_rows == 3
    assert fs.stats(split="test").n_rows == 2  # annotating "test" didn't clobber "train"


# ------------------------------------------------------- default_prediction ---

def test_predict_default_prediction_bare_label_and_none():
    ds = DataSpec(["a", "b"])
    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    rep = BooleanDataRepresentation(ds, X3)
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)  # covers row 0 only

    fs = FlatRuleSet([r])
    assert list(fs.predict(rep)) == ["pos", None, None]  # no default_prediction -> None
    fs.default_prediction = "neg"
    assert list(fs.predict(rep)) == ["pos", "neg", "neg"]

    dl = DecisionList([r])
    assert list(dl.predict(rep)) == ["pos", None, None]
    dl.default_prediction = "neg"
    assert list(dl.predict(rep)) == ["pos", "neg", "neg"]


def test_default_rule_reassignment_discards_materialized_rule_and_stats():
    r = Rule.from_pos_neg(pos=[0], target="a", dataspec=DS)
    fs = FlatRuleSet([r], default_prediction="fallback")
    dr = fs.default_rule
    assert dr is not None and dr.target == "fallback" and len(dr.conditions) == 0
    assert fs.default_rule is dr  # cached: same object each access

    fs.default_rule.stats(DATA)
    assert fs.default_rule.stats() is not None

    # reassigning the policy discards the materialized rule (and its stats)
    fs.default_prediction = "other"
    assert fs.default_rule is not dr
    assert fs.default_rule.target == "other"
    assert fs.default_rule.stats() is None

    fs.default_prediction = None
    assert fs.default_rule is None


class _ParityDefault(DefaultPrediction):
    def predict(self, rules, data, example_idx):
        return "even" if example_idx % 2 == 0 else "odd"


def test_object_policy_gives_per_example_fallback():
    # a fallback that can't be expressed as one static Rule: predict
    # "even" or "odd" depending on the uncovered example's own index
    ds = DataSpec(["a"])
    r = Rule.from_pos_neg(pos=[0], target="matched", dataspec=ds)  # covers only a=True
    X4 = np.array([[1], [0], [0], [0]], dtype=bool)
    data_rep = BooleanDataRepresentation(ds, X4)

    fs = FlatRuleSet([r], default_prediction=_ParityDefault())
    assert list(fs.predict(data_rep)) == ["matched", "odd", "even", "odd"]
    assert fs.default_rule is None  # non-constant policy -> no materialized view

    dl = DecisionList([r], default_prediction=_ParityDefault())
    assert list(dl.predict(data_rep)) == ["matched", "odd", "even", "odd"]


def test_majority_class_materializes_as_the_default_rules_target():
    ds = DataSpec(["a"])
    X5 = np.array([[1], [0], [0], [0]], dtype=bool)
    y5 = np.array(["a", "b", "b", "c"])
    rep = BooleanDataRepresentation(ds, X5, y5)
    r = Rule.from_pos_neg(pos=[0], target="a", dataspec=ds)  # covers row 0

    fs = FlatRuleSet([r], default_prediction=MajorityClass(rep))
    assert fs.default_rule.target == "b"  # majority over all rows
    assert list(fs.predict(rep)) == ["a", "b", "b", "b"]


def test_filter_remap_to_rulelist_preserve_default_prediction():
    r_a = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    r_b = Rule.from_pos_neg(pos=[1], target="b", n_features=2)
    fs = FlatRuleSet([r_a, r_b], default_prediction="fallback")

    # key= a stats-free callable -- ordering itself is exercised in
    # test_evaluation.py::test_ruleset_to_rulelist instead
    rl = fs.to_rulelist(key=lambda r: r.target)
    assert rl.default_prediction == "fallback"

    filtered = fs.filter("a")
    assert filtered.default_prediction == "fallback"
    assert isinstance(filtered, FlatRuleSet)


def test_filter_remap_to_rulelist_preserve_object_policy():
    strategy = _ParityDefault()
    r_a = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    fs = FlatRuleSet([r_a], default_prediction=strategy)

    # a non-constant strategy means default_rule reads as None, but the
    # policy object itself must survive filter()/to_rulelist()/remap() unchanged
    assert fs.default_rule is None
    assert fs.filter("a").default_prediction is strategy
    assert fs.to_rulelist(key=lambda r: r.target).default_prediction is strategy

    ds1 = DataSpec(["a", "b"])
    r = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds1)
    fs2 = FlatRuleSet([r], default_prediction=strategy)
    ds2 = DataSpec(["b", "a"])  # same names, different order
    assert fs2.remap(ds2).default_prediction is strategy


def test_default_prediction_is_abstract():
    with pytest.raises(TypeError):
        DefaultPrediction()


def test_remap_onto_merged_dataspec():
    # two "independently imported models" over the same conceptual
    # attributes, each with its own per-model DataSpec/feature order
    b1 = DataSpecBuilder()
    b1.add_nominal("color", ["red", "blue"])
    b1.add_numeric("age", [30])
    ds1 = b1.build()
    r1 = Rule.from_pos_neg(pos=[ds1.feature_index("color=red")], target="pos", dataspec=ds1)
    model1 = FlatRuleSet([r1], default_prediction="neg")

    b2 = DataSpecBuilder()
    b2.add_numeric("age", [40])  # different threshold than model1's
    b2.add_nominal("color", ["red", "blue", "green"])  # extra category than model1's
    ds2 = b2.build()
    r2 = Rule.from_pos_neg(pos=[ds2.feature_index("age>=40")], target="pos", dataspec=ds2)
    model2 = DisjointRuleSet([r2])

    # merge_dataspecs is the union of both models' features -- remap
    # never fails to find a match, by construction
    shared = merge_dataspecs(ds1, ds2).build()
    remapped1 = model1.remap(shared)
    remapped2 = model2.remap(shared)

    assert isinstance(remapped1, FlatRuleSet) and not isinstance(remapped1, DisjointRuleSet)
    assert isinstance(remapped2, DisjointRuleSet)
    assert remapped1.default_rule.target == "neg"
    assert remapped1.default_rule.dataspec is shared

    # only ONE shared dataset needs attaching for both remapped models
    # to be scored together
    X = np.zeros((3, shared.n_features), dtype=bool)
    X[0, shared.feature_index("color=red")] = True
    X[0, shared.feature_index("age>=30")] = True
    X[0, shared.feature_index("age>=40")] = True
    X[1, shared.feature_index("color=blue")] = True
    X[2, shared.feature_index("color=green")] = True
    shared_rep = BooleanDataRepresentation(shared, X)

    assert list(remapped1.predict(shared_rep)) == ["pos", "neg", "neg"]
    assert list(remapped2.predict(shared_rep)) == ["pos", None, None]


# --------------------------------------------------------- is_disjoint / coverage ---

def test_disjoint_rule_set_is_disjoint_and_exhaustive():
    from _negation_helpers import make_rule, neg_spec, neg_X

    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    ds = neg_spec(["a", "b"])
    rep = BooleanDataRepresentation(ds, neg_X(X3))
    r_a = make_rule(ds, pos=["a"], target="a")  # row 0 only
    r_b = make_rule(ds, pos=["b"], target="b")  # row 1 only

    not_exhaustive = DisjointRuleSet([r_a, r_b])  # row 2 uncovered
    assert not_exhaustive.is_disjoint(rep) is True
    assert not_exhaustive.is_exhaustive(rep) is False

    r_c = make_rule(ds, neg=["a", "b"], target="c")  # row 2 only
    full_partition = DisjointRuleSet([r_a, r_b, r_c])
    assert full_partition.is_disjoint(rep) is True
    assert full_partition.is_exhaustive(rep) is True

    r_overlap = make_rule(ds, neg=["b"], target="x")  # rows 0,2 -- overlaps r_a on row 0
    overlapping = DisjointRuleSet([r_a, r_overlap])
    assert overlapping.is_disjoint(rep) is False


def test_is_disjoint_usable_as_precondition_on_a_plain_ruleset():
    ds = DataSpec(["a", "b"])
    X3 = np.array([[1, 0], [0, 1], [0, 0]], dtype=bool)
    rep = BooleanDataRepresentation(ds, X3)
    r_a = Rule.from_pos_neg(pos=[0], target="a", dataspec=ds)
    r_b = Rule.from_pos_neg(pos=[1], target="b", dataspec=ds)

    fs = FlatRuleSet([r_a, r_b])
    assert fs.is_disjoint(rep) is True
    assert fs.is_exhaustive(rep) is False
    drs = DisjointRuleSet(fs.rules, default_prediction=fs.default_prediction)
    assert drs.is_disjoint(rep) is True

    dl = DecisionList([r_a, r_b])
    assert dl.is_disjoint(rep) is True  # available on RuleList too, same base method


def test_coverage_space_and_coverage_path():
    ds = DataSpec(["a", "b", "c"])
    X4 = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]], dtype=bool)
    y4 = np.array(["pos", "pos", "neg"])
    rep = BooleanDataRepresentation(ds, X4, y4)
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)   # rows 0,1
    r2 = Rule.from_pos_neg(pos=[2], target="neg", dataspec=ds)   # rows 0,2

    fs = FlatRuleSet([r1, r2])
    pts = fs.coverage_space(rep, "pos")
    assert pts.tolist() == [[0, 2], [1, 1]]  # r1: 0 neg/2 pos; r2: 1 neg/1 pos

    dl = DecisionList([r1, r2])
    path = dl.coverage_path(rep, "pos")
    assert path.tolist() == [[0, 0], [0, 2], [1, 2]]


def test_decision_list_predict_and_firing_coverage():
    X = np.array([
        [1, 0],   # only a
        [0, 1],   # only b
        [0, 0],   # neither
        [1, 1],   # both -- r1 should claim this one, not r2
    ], dtype=bool)
    ds = DataSpec(["a", "b"])
    rep = BooleanDataRepresentation(ds, X)
    r1 = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    r3 = Rule.from_pos_neg(target="default", dataspec=ds)  # no conditions -> catch-all
    dl = DecisionList([r1, r2, r3])

    assert list(dl.predict(rep)) == ["A", "B", "default", "A"]

    cov = dl.coverage_matrix(rep)          # raw coverage ignores order/position entirely
    unique = dl._unique_mask(cov)          # "fires": matches, and not already claimed
    assert list(cov.sum(axis=1)) == [2, 2, 4]
    assert list(unique.sum(axis=1)) == [2, 1, 1]


# ------------------------------------------------------------------ to_string ---

def test_flat_rule_set_to_string_grouped_by_class():
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)
    fs = FlatRuleSet([r1, r2], default_prediction="low_risk")

    prolog = fs.to_string(fmt="prolog")
    assert "% class: high_risk" in prolog
    assert "% default" in prolog
    assert prolog.index("% class: high_risk") < prolog.index("% default")
    assert "high_risk(X) :- age_gt_30(X), smoker(X)." in prolog
    assert "high_risk(X) :- high_bp(X)." in prolog
    assert "low_risk(X) :- true." in prolog  # the default rule, no conditions

    logic = fs.to_string(fmt="logic")
    # both rules share a target -> collapse into one parenthesized DNF block
    assert "(age_gt_30 ∧ smoker)" in logic
    assert "∨ (high_bp)" in logic
    assert "→ high_risk" in logic

    # every rule renders in the *same* requested format regardless of
    # each rule's own default_fmt
    r1.default_fmt = "conditions"
    assert "age_gt_30(X), smoker(X)" in fs.to_string(fmt="prolog")


def test_flat_rule_set_to_string_no_default_rule_no_section():
    r = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    fs = FlatRuleSet([r])
    assert "default" not in fs.to_string(fmt="prolog")


def test_decision_list_to_string_sequential_and_if_elif_else():
    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pos_neg(pos=[0], target="A", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="B", dataspec=ds)
    dl = DecisionList([r1, r2], default_prediction="C")

    logic = dl.to_string(fmt="logic")
    lines = logic.splitlines()
    assert lines[0].startswith("if  ") and lines[0].endswith("→ A")
    assert lines[1].startswith("elif") and lines[1].endswith("→ B")
    assert lines[2] == "else → C"

    prolog = dl.to_string(fmt="prolog")
    plines = prolog.splitlines()
    # uppercase-leading targets aren't valid bare Prolog atoms (that's a
    # variable, not a predicate name) -- quoted, same as a Rule printed alone
    assert plines[0] == "'A'(X) :- a(X)."
    assert plines[1] == "'B'(X) :- b(X)."
    assert plines[2] == "% default"
    assert plines[3] == "'C'(X) :- true."


def test_single_rule_to_string_delegates_to_the_wrapped_rule():
    r = Rule.from_pos_neg(pos=[0], target="a", n_features=2)
    sr = SingleRule(r)
    assert sr.to_string("logic") == r.to_string("logic")  # fmt is positional-first, like Rule.to_string
    assert sr.to_string(fmt="prolog") == r.to_string(fmt="prolog")


def test_to_string_coverage_decoration_needs_no_prior_annotation():
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)
    fs = FlatRuleSet([r1, r2])

    # undecorated by default -- identical to the plain rendering
    assert fs.to_string(fmt="prolog") == fs.to_string(fmt="prolog", data=None)

    X = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)
    data_rep = BooleanDataRepresentation(ds, X)
    prolog = fs.to_string(fmt="prolog", data=data_rep)
    assert "  % (1)" in prolog  # no labels -> plain "(n_covered)"


def test_to_string_coverage_correctness_counts():
    # row0: covered by r1 only, label matches target (correct)
    # row1: covered by r2 only, label does NOT match target (wrong)
    # row2: covered by both r1 and r2 (not unique to either), label matches target
    ds = DataSpec(["age_gt_30", "smoker", "high_bp"])
    r1 = Rule.from_pos_neg(pos=[0, 1], target="high_risk", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[2], target="high_risk", dataspec=ds)
    fs = FlatRuleSet([r1, r2])

    X = np.array([
        [1, 1, 0],  # row0: age_gt_30 & smoker -> covered by r1 only
        [0, 0, 1],  # row1: high_bp -> covered by r2 only
        [1, 1, 1],  # row2: all three -> covered by both
    ], dtype=bool)
    y = np.array(["high_risk", "low_risk", "high_risk"])
    data_rep = BooleanDataRepresentation(ds, X, y)

    prolog = fs.to_string(fmt="prolog", data=data_rep)
    # r1: covers rows 0,2, both high_risk (its own target) -> tp=2, fp=0
    assert "high_risk(X) :- age_gt_30(X), smoker(X).  % (2/0)" in prolog
    # r2: covers rows 1,2 -- row1 is low_risk (wrong), row2 is high_risk -> tp=1, fp=1
    assert "high_risk(X) :- high_bp(X).  % (1/1)" in prolog


def _three_class_dog_rule():
    # one rule, covering 16 training rows: 1 bird, 1 cat, 14 dog
    ds = DataSpec(["barks"])
    X = np.array([[1]] * 16 + [[0]] * 3, dtype=bool)
    y = np.array(["bird"] + ["cat"] + ["dog"] * 14 + ["bird", "cat", "dog"])
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds)
    return FlatRuleSet([rule]), BooleanDataRepresentation(ds, X, y)


def test_to_string_prints_the_full_distribution_only_for_a_distributioncombiner():
    fs, data_rep = _three_class_dog_rule()

    # a DistributionCombiner and >2 classes -> the full per-class breakdown,
    # with a printed-once legend giving its order (sorted: bird, cat, dog)
    fs.combiner = "micro_vote"
    text = fs.to_string(fmt="prolog", data=data_rep)
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text

    # "max" (the default) -- no distribution, no legend, plain (tp/fp)
    fs.combiner = "max"
    text = fs.to_string(fmt="prolog", data=data_rep)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (14/2)" in text
    print("to_string shows the full class distribution + legend only for a "
          "DistributionCombiner, plain (tp/fp) otherwise: OK")


def test_to_string_skips_the_distribution_for_a_binary_problem_even_with_a_distributioncombiner():
    # only 2 classes -- the distribution vector would just be (tp/fp) reordered,
    # so it's redundant and stays suppressed even though the combiner would want it
    ds = DataSpec(["barks"])
    X = np.array([[1]] * 5 + [[0]] * 5, dtype=bool)
    y = np.array(["dog"] * 5 + ["cat"] * 5)
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds)
    fs = FlatRuleSet([rule], combiner="micro_vote")
    data_rep = BooleanDataRepresentation(ds, X, y)

    text = fs.to_string(fmt="prolog", data=data_rep)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (5/0)" in text
    print("Binary problems skip the distribution bracket/legend even under a "
          "DistributionCombiner -- it would just be (tp/fp) reordered: OK")


def test_to_string_singlerule_default_never_prints_the_distribution():
    # SingleRule.resolution is Exclusive, never a Combine -- by default there's
    # only one rule, so no distribution-scored disagreement to make visible
    _, data_rep = _three_class_dog_rule()
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=data_rep.spec)
    sr = SingleRule(rule)
    text = sr.to_string(fmt="prolog", data=data_rep)
    assert "% classes:" not in text
    assert text == "dog(X) :- barks(X).  % (14/2)"
    print("A standalone SingleRule defaults to plain (tp/fp), never a distribution: OK")


def test_to_string_show_distribution_forces_the_choice_either_way():
    fs, data_rep = _three_class_dog_rule()  # combiner="max" (the default), 3 classes

    # show_distribution=True forces the vector even though "max" never needs it
    text = fs.to_string(fmt="prolog", data=data_rep, show_distribution=True)
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text

    # show_distribution=False suppresses it even under a genuine DistributionCombiner
    fs.combiner = "micro_vote"
    text = fs.to_string(fmt="prolog", data=data_rep, show_distribution=False)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % (14/2)" in text

    # forcing it on works even where the model structurally never has a
    # DistributionCombiner at all: a lone SingleRule, and a binary problem
    rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=data_rep.spec)
    sr_text = SingleRule(rule).to_string(fmt="prolog", data=data_rep, show_distribution=True)
    assert sr_text == "% classes: [bird, cat, dog]\n\ndog(X) :- barks(X).  % [1, 1, 14]"

    ds2 = DataSpec(["barks"])
    binary_data = BooleanDataRepresentation(
        ds2, np.array([[1]] * 5 + [[0]] * 5, dtype=bool), np.array(["dog"] * 5 + ["cat"] * 5))
    binary_rule = Rule.from_pos_neg(pos=[0], target="dog", dataspec=ds2)
    binary_fs = FlatRuleSet([binary_rule])  # default "max"
    text = binary_fs.to_string(fmt="prolog", data=binary_data, show_distribution=True)
    assert "% classes: [cat, dog]" in text
    assert "dog(X) :- barks(X).  % [0, 5]" in text  # cat=0, dog=5, class order [cat, dog]
    print("show_distribution=True/False forces the vector on or off regardless of "
          "the model's own combiner or class count: OK")


def test_to_string_show_classes_is_independent_of_show_distribution():
    fs, data_rep = _three_class_dog_rule()  # combiner="max", no rule would show a vector by default

    # show_classes=True prints the legend even though no rule shows a distribution
    text = fs.to_string(fmt="prolog", data=data_rep, show_classes=True)
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    assert "dog(X) :- barks(X).  % (14/2)" in text  # still plain (tp/fp) -- show_distribution untouched

    # show_classes=False suppresses the legend even while a distribution IS shown
    fs.combiner = "micro_vote"
    text = fs.to_string(fmt="prolog", data=data_rep, show_classes=False)
    assert "% classes:" not in text
    assert "dog(X) :- barks(X).  % [1, 1, 14]" in text  # the vector itself is untouched
    print("show_classes independently forces the legend on or off, regardless of "
          "whether any rule is actually showing a distribution vector: OK")


def _three_class_pairwise_fixture():
    # cat/dog/bird, 3 pairs, each sub-model's rule only ever explicitly
    # predicts ONE of its own two classes -- the other only ever surfaces as
    # that sub-model's own default_prediction
    ds = DataSpec(["a", "b"])
    X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1]], dtype=bool)
    y = np.array(["cat", "cat", "dog", "dog", "bird", "bird"])
    data = BooleanDataRepresentation(ds, X, y)
    cat_dog = ConceptModel([Rule.from_pos_neg(pos=[0], target="cat", dataspec=ds)],
                           label="cat", default_prediction="dog")
    cat_bird = ConceptModel([Rule.from_pos_neg(pos=[0], target="cat", dataspec=ds)],
                            label="cat", default_prediction="bird")
    dog_bird = ConceptModel([Rule.from_pos_neg(pos=[1], target="dog", dataspec=ds)],
                            label="dog", default_prediction="bird")
    return cat_dog, cat_bird, dog_bird, data


def test_pairwisemodel_to_string_forces_each_pairs_own_two_classes_by_default():
    cat_dog, cat_bird, dog_bird, data = _three_class_pairwise_fixture()
    pw = PairwiseModel(
        [("cat", "dog", cat_dog), ("cat", "bird", cat_bird), ("dog", "bird", dog_bird)],
        combiner="accuracy_vote", member_weights=[0.9, 0.75, 0.6], default_prediction="cat",
    )
    text = pw.to_string(fmt="prolog", data=data)

    # top-level legend: every class this model spans
    assert text.splitlines()[0] == "% classes: [bird, cat, dog]"
    # each pair's own header names it and, for accuracy_vote, its member weight
    assert "% pair: cat vs dog  (member weight: 0.9)" in text
    assert "% pair: cat vs bird  (member weight: 0.75)" in text
    assert "% pair: dog vs bird  (member weight: 0.6)" in text
    # each pair's own sub-model is forced to show its own two-class legend,
    # even though its one rule only ever explicitly predicts one of them
    assert "% classes: [cat, dog]" in text
    assert "% classes: [bird, cat]" in text
    assert "% classes: [bird, dog]" in text
    # data handed to each sub-model is narrowed to that pair's own rows:
    # cat_dog's rule (a=1 -> cat) sees only the 2 cat + 2 dog rows -> tp=2, fp=0
    assert "cat(X) :- a(X).  % (2/0)" in text
    # cat_bird's *same* rule, but narrowed to cat+bird rows instead -- the
    # one bird row with a=1 is now a false positive -- tp=2, fp=1
    assert "cat(X) :- a(X).  % (2/1)" in text
    print("PairwiseModel.to_string forces each pair's own two-class legend and "
          "shows accuracy_vote's per-pair member weight: OK")


def test_pairwisemodel_to_string_show_classes_false_suppresses_everything():
    cat_dog, cat_bird, dog_bird, data = _three_class_pairwise_fixture()
    pw = PairwiseModel([("cat", "dog", cat_dog), ("cat", "bird", cat_bird), ("dog", "bird", dog_bird)])
    text = pw.to_string(fmt="prolog", data=data, show_classes=False)
    assert "% classes:" not in text
    print("PairwiseModel.to_string's show_classes=False suppresses the top-level "
          "and every per-pair legend: OK")


def test_ensemblemodel_to_string_shows_member_weights_and_top_level_legend():
    cat_dog, _, dog_bird, data = _three_class_pairwise_fixture()
    ens = EnsembleModel([cat_dog, dog_bird], member_weights=[0.7, 0.3])
    text = ens.to_string(fmt="prolog", data=data)
    # .labels is rule heads + the ensemble's OWN default (unset here) --
    # "bird" never appears as either, only as a *sub-model's own* default,
    # so it's genuinely outside this model's declared label set
    assert "% classes: [cat, dog]" in text
    assert "% member 0  (weight: 0.7)" in text
    assert "% member 1  (weight: 0.3)" in text
    # members see the *full*, unfiltered data -- cat_dog's rule (a=1 -> cat)
    # over all 6 rows also covers the a=1 bird row as a false positive
    assert "cat(X) :- a(X).  % (2/1)" in text
    print("EnsembleModel.to_string shows each member's own weight, unfiltered "
          "data, and a top-level classes legend: OK")


# ------------------------------------------------------------------ covered_by ---

def _covered_by_fixture():
    ds = DataSpec(["p", "q", "r"])
    X = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    y = np.array(["a", "a", "b", "b", "b", "a"])
    data = BooleanDataRepresentation(ds, X, y)
    rp = Rule([0], target="a", dataspec=ds)       # p -> a: covers rows 0,1,2 (2/3 correct)
    rpq = Rule([0, 1], target="a", dataspec=ds)   # p & q -> a: covers rows 1,2 (1/2 correct)
    rr = Rule([2], target="b", dataspec=ds)       # r -> b: covers rows 2,3,4 (all correct)
    rq = Rule([1], target="b", dataspec=ds)       # q -> b: covers rows 1,2,3 (2/3 correct)
    # covered_by() calls predict() internally (to put the predicted
    # class's block first) -- annotate real stats so both the default
    # "max" combiner and covered_by's own default (Laplace-on-stats)
    # within-block sort have something to score. Laplace scores here:
    # rp=(2+1)/5=0.6, rpq=(1+1)/4=0.5, rr=(3+1)/5=0.8, rq=(2+1)/5=0.6
    # -- rq/rp tie exactly, broken by insertion order (rp built first).
    rules = annotate_rules([rp, rpq, rr, rq], data)
    fs = FlatRuleSet(rules, default_prediction="a")
    return ds, data, y, fs


def test_covered_by_blocks_predicted_class_first_sorted_by_score():
    ds, data, y, fs = _covered_by_fixture()
    preds = list(fs.predict(data))
    cb = fs.covered_by(data)  # default by=None -> Laplace-on-measured-stats

    assert len(cb) == data.n_samples
    for j, rules in enumerate(cb):
        if not rules:
            continue
        first_target = rules[0].target
        if preds[j] in {r.target for r in rules}:
            assert first_target == preds[j]
        targets = [r.target for r in rules]
        assert targets == sorted(targets, key=lambda t: targets.index(t))  # no interleaving

    # ex2 = [p,q,r]: covered by all 4, predicted "b" -> b-block (r:0.8, q:0.6) then a-block (p:0.6, pq:0.5)
    assert [r.pos for r in cb[2]] == [(2,), (1,), (0,), (0, 1)]
    # ex5 = [] : nothing covers -> the default rule
    assert len(cb[5]) == 1 and cb[5][0] is fs.default_rule


def test_covered_by_other_blocks_ordered_by_strongest_rule():
    ds = DataSpec(["p", "q", "s"])
    X = np.array([[1, 1, 1]], dtype=bool)
    data = BooleanDataRepresentation(ds, X, np.array(["a"]))
    ra = Rule([0], target="a", dataspec=ds)
    rb = Rule([1], target="b", dataspec=ds)
    rc = Rule([2], target="c", dataspec=ds)
    # combiner="list" so predict() (called internally by covered_by()) doesn't
    # need stats -- ra is first in list order, matching this fixture's
    # intended "a" prediction. b/c are structurally tied under any real
    # measurement on this single row, so an explicit `by=` (a plain
    # callable, no stats needed) exercises the "other blocks by their
    # strongest rule" mechanism directly rather than the stats-based default.
    fs = FlatRuleSet([ra, rb, rc], default_prediction="a", combiner="list")
    scores = {"a": 0.95, "b": 0.40, "c": 0.70}
    (block,) = fs.covered_by(data, by=lambda r: scores[r.target])
    # predicted "a" first, then the *other* blocks by their strongest rule: c (0.70) before b (0.40)
    assert [r.target for r in block] == ["a", "c", "b"]


def test_covered_by_on_a_decision_list_and_with_no_default():
    ds = DataSpec(["a", "b"])
    X = np.array([[1, 1], [0, 0]], dtype=bool)
    data = BooleanDataRepresentation(ds, X, np.array(["B", "A"]))
    dl = DecisionList([Rule([1], target="B", dataspec=ds), Rule([0], target="A", dataspec=ds)],
                     default_prediction=None)  # no constant default
    cb = dl.covered_by(data)  # singleton blocks -- no scoring needed, no stats required
    assert [r.target for r in cb[0]] == ["B", "A"]   # first-match B predicted -> its block first
    assert cb[1] == []                                # nothing covers, no default rule


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: OK")
    print("\nAll model tests passed.")
