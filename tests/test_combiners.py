import numpy as np
from pyrulearn import (
    BooleanDataRepresentation,
    DataSpec,
    DistributionCombiner,
    HeuristicCombiner,
    HeuristicMaxCombiner,
    HeuristicVoteCombiner,
    CountVoteCombiner,
    Laplace,
    ListCombiner,
    Precision,
    Rule,
    RuleCombiner,
)
from pyrulearn.models import FlatRuleSet, annotate_rules

from _negation_helpers import make_rule, neg_spec, neg_X


def test_combiner_shortcuts_and_instances_are_interchangeable():
    ds = DataSpec(["a", "b"])
    train_X = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    train_y = np.array(["first", "first", "first", "heaviest", "second"])  # a: 3/3 -> pure; b: 1/2
    train_rep = BooleanDataRepresentation(ds, train_X, train_y)

    r_first = Rule.from_pos_neg(pos=[0], target="first", dataspec=ds)
    r_heaviest = Rule.from_pos_neg(pos=[1], target="heaviest", dataspec=ds)
    r_first, r_heaviest = annotate_rules([r_first, r_heaviest], train_rep)
    rs = FlatRuleSet([r_first, r_heaviest])
    X = np.array([[1, 1]], dtype=bool)  # both fire
    data_rep = BooleanDataRepresentation(ds, X)

    assert list(rs.predict(data_rep, combiner="list")) == ["first"]
    assert list(rs.predict(data_rep, combiner=ListCombiner())) == ["first"]
    assert list(rs.predict(data_rep, combiner="max")) == ["first"]  # higher measured precision
    assert list(rs.predict(data_rep, combiner=HeuristicMaxCombiner())) == ["first"]

    try:
        rs.predict(data_rep, combiner="not_a_real_strategy")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("combiner shortcuts and RuleCombiner instances interchangeable: OK")


def test_heuristic_max_combiner_scores_from_measured_stats():
    # two rules, each annotated (via annotate_rules) against real training
    # data -- HeuristicMaxCombiner must score from that measured
    # ConfusionMatrix.
    ds = DataSpec(["a", "b"])
    X = np.array([
        [1, 0], [1, 0], [1, 0], [1, 1],   # a fires 4x, all "pos" -> precision 1.0
        [0, 1], [0, 1], [1, 0],           # b fires 3x: pos, pos, (a also fires on the 3rd)
    ], dtype=bool)
    y = np.array(["pos", "pos", "pos", "pos", "pos", "neg", "pos"])
    data_rep = BooleanDataRepresentation(ds, X, y)

    r_a = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r_b = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    r_a, r_b = annotate_rules([r_a, r_b], data_rep)  # r_a's real precision (1.0) beats r_b's (0.5)
    rs = FlatRuleSet([r_a, r_b])

    test_rep = BooleanDataRepresentation(ds, np.array([[1, 1]], dtype=bool))  # both fire
    assert list(rs.predict(test_rep, combiner=HeuristicMaxCombiner(Laplace()))) == ["pos"]  # via r_a, not r_b
    # a non-Laplace heuristic (Precision) agrees here too, just to show it's pluggable
    assert list(rs.predict(test_rep, combiner=HeuristicMaxCombiner(Precision()))) == ["pos"]
    print("HeuristicMaxCombiner scores from measured stats: OK")


def test_heuristic_vote_combiner_forest_like_scenario():
    # simulate a 3-tree "forest" imported as one RuleSet: for every
    # example, exactly one rule per tree fires, so exactly 3 rules
    # always cover each example. Trees 1/2 are noisy (25% precision on
    # this training set); tree 3 is perfect (100%) -- real, measured
    # reliability, not a hand-set weight.
    ds = neg_spec(["a", "b", "c"])
    tree1_yes = make_rule(ds, pos=["a"], target="yes")
    tree1_no = make_rule(ds, neg=["a"], target="no")
    tree2_yes = make_rule(ds, pos=["b"], target="yes")
    tree2_no = make_rule(ds, neg=["b"], target="no")
    tree3_yes = make_rule(ds, pos=["c"], target="yes")
    tree3_no = make_rule(ds, neg=["c"], target="no")

    train_raw = np.array(
        [[0, 0, 1]] * 3 +   # tree3_yes correct x3
        [[0, 0, 0]] +       # tree1_no/tree2_no/tree3_no correct x1
        [[1, 1, 0]],        # tree1_yes/tree2_yes wrong; tree3_no correct
        dtype=bool)
    train_y = np.where(train_raw[:, 2], "yes", "no")
    train_rep = BooleanDataRepresentation(ds, neg_X(train_raw), train_y)
    rules = annotate_rules(
        [tree1_yes, tree1_no, tree2_yes, tree2_no, tree3_yes, tree3_no], train_rep,
    )
    forest = FlatRuleSet(rules)

    X = neg_X([
        [1, 1, 0],  # trees vote yes, yes, no -> majority "yes" unweighted
        [0, 0, 1],  # trees vote no, no, yes -> majority "no" unweighted
    ])
    data_rep = BooleanDataRepresentation(ds, X)

    preds = forest.predict(data_rep, combiner="vote")
    assert list(preds) == ["yes", "no"]

    # weighted by measured Precision: tree3's near-perfect precision (1.0)
    # outweighs tree1/tree2's noisy 0.25 each on both examples, flipping both
    preds_weighted = forest.predict(data_rep, combiner=HeuristicVoteCombiner(Precision()))
    assert list(preds_weighted) == ["no", "yes"]
    print("HeuristicVoteCombiner, scored from measured stats, flips both examples: OK")


def test_distribution_combiners_micro_vs_macro_genuinely_diverge():
    # a "big leaf" rule (a=1: 80 "yes"/20 "no" over 100 training rows) and
    # a "small leaf" rule (b=1: 1 "yes"/9 "no" over 10 rows), covering the
    # same test example -- micro pools raw counts (big leaf dominates),
    # macro normalizes each rule first (both count equally), and these
    # two policies disagree on this example by construction.
    ds = DataSpec(["a", "b"])
    X = np.array(
        [[1, 0]] * 80 + [[1, 0]] * 20 + [[0, 1]] * 1 + [[0, 1]] * 9,
        dtype=bool,
    )
    y = np.array(["yes"] * 80 + ["no"] * 20 + ["yes"] * 1 + ["no"] * 9)
    data_rep = BooleanDataRepresentation(ds, X, y)

    r_big_leaf = Rule.from_pos_neg(pos=[0], target="yes", dataspec=ds)
    r_small_leaf = Rule.from_pos_neg(pos=[1], target="no", dataspec=ds)
    r_big_leaf, r_small_leaf = annotate_rules([r_big_leaf, r_small_leaf], data_rep)
    rs = FlatRuleSet([r_big_leaf, r_small_leaf])

    test_rep = BooleanDataRepresentation(ds, np.array([[1, 1]], dtype=bool))  # both fire

    # micro: pooled raw counts -- yes=81, no=29 -> the big leaf dominates
    assert list(rs.predict(test_rep, combiner="micro_vote")) == ["yes"]
    assert list(rs.predict(test_rep, combiner="micro_max")) == ["yes"]

    # macro: normalize each rule first -- yes=0.8+0.1=0.9, no=0.2+0.9=1.1 -> flips
    assert list(rs.predict(test_rep, combiner="macro_vote")) == ["no"]
    assert list(rs.predict(test_rep, combiner="macro_max")) == ["no"]
    print("Micro vs Macro distribution combiners genuinely diverge: OK")


def test_heuristic_combiner_raises_without_stats_on_a_genuine_disagreement():
    # two DIFFERENT-target rules covering the same row, neither annotated
    # -- a genuine tie HeuristicCombiner cannot resolve without stats, so
    # it must raise.
    ds = DataSpec(["a", "b"])
    r_pos = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r_neg = Rule.from_pos_neg(pos=[1], target="neg", dataspec=ds)
    rs = FlatRuleSet([r_pos, r_neg])
    X = np.array([[1, 1]], dtype=bool)
    data_rep = BooleanDataRepresentation(ds, X)

    try:
        rs.predict(data_rep, combiner="max")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "stats" in str(e)
    try:
        rs.predict(data_rep, combiner=HeuristicVoteCombiner())
        assert False, "expected ValueError"
    except ValueError as e:
        assert "stats" in str(e)
    print("HeuristicMaxCombiner/HeuristicVoteCombiner raise on an unannotated genuine disagreement: OK")


def test_distribution_combiner_raises_without_stats_on_a_genuine_disagreement():
    ds = DataSpec(["a", "b"])
    r_pos = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r_neg = Rule.from_pos_neg(pos=[1], target="neg", dataspec=ds)
    rs = FlatRuleSet([r_pos, r_neg])
    X = np.array([[1, 1]], dtype=bool)
    data_rep = BooleanDataRepresentation(ds, X)

    try:
        rs.predict(data_rep, combiner="macro_vote")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "stats" in str(e)
    print("DistributionCombiner raises on an unannotated genuine disagreement: OK")


def test_a_trivial_single_target_never_needs_stats():
    # every covering rule agreeing on the target is not a disagreement to
    # resolve -- neither combiner should need (or demand) stats for it.
    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="pos", dataspec=ds)
    rs = FlatRuleSet([r1, r2])
    X = np.array([[1, 1]], dtype=bool)
    data_rep = BooleanDataRepresentation(ds, X)

    assert list(rs.predict(data_rep, combiner="max")) == ["pos"]
    assert list(rs.predict(data_rep, combiner=HeuristicVoteCombiner())) == ["pos"]
    assert list(rs.predict(data_rep, combiner="macro_vote")) == ["pos"]
    print("A unanimous covering block never needs stats, from either combiner family: OK")


def test_combiner_bases_are_abstract():
    for cls in (RuleCombiner, HeuristicCombiner, DistributionCombiner):
        try:
            cls()
            assert False, f"expected TypeError instantiating {cls.__name__}"
        except TypeError:
            pass
    print("RuleCombiner/HeuristicCombiner/DistributionCombiner are all abstract: OK")


def test_count_vote_combiner_is_the_vote_shortcut():
    ds = DataSpec(["a", "b"])
    r1 = Rule.from_pos_neg(pos=[0], target="x", dataspec=ds)
    r2 = Rule.from_pos_neg(pos=[1], target="x", dataspec=ds)
    r3 = Rule.from_pos_neg(pos=[0], target="y", dataspec=ds)
    rs = FlatRuleSet([r1, r2, r3])
    X = np.array([[1, 1]], dtype=bool)  # all three fire: x, x, y -> x wins 2-1
    data_rep = BooleanDataRepresentation(ds, X)

    assert list(rs.predict(data_rep, combiner="vote")) == ["x"]
    assert list(rs.predict(data_rep, combiner=CountVoteCombiner())) == ["x"]
    print("CountVoteCombiner is exactly the 'vote' shortcut: OK")


def test_heuristic_vote_combiner_flips_a_noisy_majority():
    # r_pure always agrees with the label where it fires (precision 1.0);
    # r_noisy1/r_noisy2 are each right only 1 time in 3 (precision 1/3)
    X = np.array([
        [1, 0, 0], [1, 0, 0],                    # r_pure fires, both pos
        [0, 1, 0], [0, 1, 0], [0, 1, 0],          # r_noisy1 fires, target "neg", only row idx 4 matches
        [0, 0, 1], [0, 0, 1], [0, 0, 1],          # r_noisy2 fires, target "neg", only row idx 7 matches
    ], dtype=bool)
    y = np.array(["pos", "pos", "pos", "pos", "neg", "pos", "pos", "neg"])
    ds = DataSpec(["a", "b", "c"])
    rep = BooleanDataRepresentation(ds, X, y)

    r_pure = Rule.from_pos_neg(pos=[0], target="pos", dataspec=ds)
    r_noisy1 = Rule.from_pos_neg(pos=[1], target="neg", dataspec=ds)
    r_noisy2 = Rule.from_pos_neg(pos=[2], target="neg", dataspec=ds)
    rules = annotate_rules([r_pure, r_noisy1, r_noisy2], rep)  # real measured stats against rep
    rs = FlatRuleSet(rules)

    test_rep = BooleanDataRepresentation(DataSpec(["a", "b", "c"]), np.array([[1, 1, 1]], dtype=bool))

    # unweighted: 2 "neg" votes (r_noisy1, r_noisy2) beat 1 "pos" vote (r_pure)
    assert list(rs.predict(test_rep, combiner="vote")) == ["neg"]

    # weighted by measured Precision: r_pure's precision (1.0) now outweighs
    # the noisy rules' combined precision (1/3 + 1/3 = 2/3), flipping it
    st_pure = rs.rules[0].stats().confusion.rule_stats("pos")
    assert Precision().score(st_pure) == 1.0
    st_noisy1 = rs.rules[1].stats().confusion.rule_stats("neg")
    assert abs(Precision().score(st_noisy1) - 1 / 3) < 1e-12
    assert list(rs.predict(test_rep, combiner=HeuristicVoteCombiner(Precision()))) == ["pos"]
    print("HeuristicVoteCombiner(Precision()), scored from measured stats, flips a noisy majority: OK")


# -- ties: never by rule position ------------------------------------------

def _tie_rules(specs, totals):
    """One row covered by every rule: each spec is (target, covered_counts)."""
    from pyrulearn.models import SingleRule
    ds = DataSpec(["a", "b", "c", "d", "e"])
    rules = [SingleRule(Rule([i], target=t, dataspec=ds)).set_stats_from_counts(covered, totals)
             for i, (t, covered) in enumerate(specs)]
    row = BooleanDataRepresentation(ds, np.ones((1, 5), dtype=bool))
    return rules, row


def _predict_every_order(rules, row, combiner):
    """The prediction for every rule order -- all must agree."""
    from itertools import permutations
    return {FlatRuleSet(list(p), combiner=combiner).predict(row)[0] for p in permutations(rules)}


def test_max_ties_vote_among_the_tied_top_rules():
    totals = {"good": 300, "bad": 700}          # bad more frequent: must not matter here
    rules, row = _tie_rules([
        ("good", {"good": 8, "bad": 2}),          # Laplace 0.75
        ("good", {"good": 8, "bad": 2}),          # Laplace 0.75
        ("bad", {"bad": 8, "good": 2}),           # Laplace 0.75
        ("bad", {"bad": 1, "good": 4}),           # lower score -- doesn't vote
    ], totals)
    assert _predict_every_order(rules, row, "max") == {"good"}   # 2 tied good vs 1 tied bad


def test_remaining_ties_go_to_training_frequency_then_label_order():
    level = [("good", {"good": 8, "bad": 2}), ("bad", {"bad": 8, "good": 2})]
    for combiner in ("max", "vote", HeuristicVoteCombiner(Laplace())):
        rules, row = _tie_rules(level, {"good": 700, "bad": 300})
        assert _predict_every_order(rules, row, combiner) == {"good"}, combiner   # more frequent
        rules, row = _tie_rules(level, {"good": 300, "bad": 700})
        assert _predict_every_order(rules, row, combiner) == {"bad"}, combiner
        rules, row = _tie_rules(level, {"good": 500, "bad": 500})
        assert _predict_every_order(rules, row, combiner) == {"bad"}, combiner    # sorts first


def test_distribution_combiner_ties_use_the_same_fallback():
    # micro vote: summed counts good 5, bad 5 -> level
    level = [("good", {"good": 3, "bad": 3}), ("bad", {"good": 2, "bad": 2})]
    rules, row = _tie_rules(level, {"good": 700, "bad": 300})
    assert _predict_every_order(rules, row, "micro_vote") == {"good"}
    rules, row = _tie_rules(level, {"good": 300, "bad": 700})
    assert _predict_every_order(rules, row, "micro_vote") == {"bad"}


def test_every_combiner_describes_itself():
    from pyrulearn.combiners import _COMBINER_SHORTCUTS
    from pyrulearn.heuristics import FBeta
    assert {k: c.describe() for k, c in _COMBINER_SHORTCUTS.items()} == {
        "list": "first matching rule",
        "max": "max Laplace",
        "vote": "vote (one vote per covering rule)",
        "micro_vote": "sum of covered class counts",
        "macro_vote": "sum of covered class proportions",
        "micro_max": "max covered class count",
        "macro_max": "max covered class proportion",
    }
    assert HeuristicMaxCombiner(FBeta(beta=2.0)).describe() == "max FBeta(beta=2.0)"
    assert HeuristicVoteCombiner(Precision()).describe() == "vote weighted by Precision"


if __name__ == "__main__":
    test_combiner_shortcuts_and_instances_are_interchangeable()
    test_heuristic_max_combiner_scores_from_measured_stats()
    test_heuristic_vote_combiner_forest_like_scenario()
    test_distribution_combiners_micro_vs_macro_genuinely_diverge()
    test_heuristic_combiner_raises_without_stats_on_a_genuine_disagreement()
    test_distribution_combiner_raises_without_stats_on_a_genuine_disagreement()
    test_a_trivial_single_target_never_needs_stats()
    test_combiner_bases_are_abstract()
    test_count_vote_combiner_is_the_vote_shortcut()
    test_heuristic_vote_combiner_flips_a_noisy_majority()
    test_max_ties_vote_among_the_tied_top_rules()
    test_remaining_ties_go_to_training_frequency_then_label_order()
    test_distribution_combiner_ties_use_the_same_fallback()
    test_every_combiner_describes_itself()
    print("\nAll tests passed.")
