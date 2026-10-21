"""Numeric attributes in both directions, and ORDINAL attributes (0.4.0)."""

import io

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data import DataSpecBuilder, merge_dataspecs
from pyrulearn.data.discretization import TreeDiscretizer
from pyrulearn.data.attributes import AttributeType, MissingStrategy, NumericGroup, OrdinalGroup
from pyrulearn.data.io import binarize, build_dataspec, read_csv
from pyrulearn.rule import Rule


def _true(ds, assignment):
    return sorted(ds.feature_name(i) for i, v in assignment.items() if v)


# -- numeric attributes: always both directions -------------------------------

def test_numeric_always_has_both_directions():
    for negation in (True, False):
        b = DataSpecBuilder(negation=negation)
        idx = b.add_numeric("age", [30, 20])
        ds = b.build()
        assert ds.feature_names == ["age>=20", "age>=30", "age<20", "age<30"]
        assert idx.negative == {20: 2, 30: 3}
        assert ds.attributes["age"].negation is False
        assert [type(c) for c in ds.constraints] == [NumericGroup]
        assert ds.feature_name(ds.negation_of("age>=30")) == "age<30"


def test_numeric_le_family_always_has_its_complement():
    b = DataSpecBuilder(negation=False)
    _, le = b.add_numeric("x", le_thresholds=[5, 10])
    ds = b.build()
    assert ds.feature_names == ["x<=10", "x<=5", "x>10", "x>5"]
    assert le.negative == {10: 2, 5: 3}
    assert ds.feature_name(ds.negation_of("x<=5")) == "x>5"


def test_lt_features_are_not_negation_features():
    b = DataSpecBuilder()
    b.add_numeric("age", [30])
    ds = b.build()
    assert not ds.has_negation_features
    assert ds.without_negations()[0] is ds


def test_without_negations_keeps_numeric_lt():
    b = DataSpecBuilder()
    b.add_nominal("color", ["red", "blue", "green"])
    b.add_numeric("age", [30])
    ds = b.build()
    pos, keep = ds.without_negations()
    assert pos.feature_names == ["color=red", "color=blue", "color=green", "age>=30", "age<30"]
    assert [ds.feature_name(i) for i in keep] == pos.feature_names
    back, _, _ = pos.with_negations()
    assert back.feature_names == ds.feature_names


def test_build_dataspec_numeric_both_directions_without_negations():
    df = pd.DataFrame({"x": np.arange(40, dtype=float), "y": ["a"] * 20 + ["b"] * 20})
    ds = build_dataspec(df, target="y", include_negations=False).build()
    assert ds.feature_names == ["x>=19.5", "x<19.5"]


# -- ordinal attributes: features and constraints -----------------------------

def test_ordinal_features():
    b = DataSpecBuilder()
    idx = b.add_ordinal("size", ["s", "m", "l"])
    ds = b.build()
    assert ds.feature_names == ["size=s", "size=m", "size=l", "size<=s", "size<=m",
                                "size>s", "size>m", "size!=s", "size!=m", "size!=l"]
    assert ds.attributes["size"].type == AttributeType.ORDINAL
    assert dict(idx) == {"s": 0, "m": 1, "l": 2}
    assert idx.le == {"s": 3, "m": 4} and idx.gt == {"s": 5, "m": 6}
    assert idx.negative == {"s": 7, "m": 8, "l": 9}
    assert idx.all == list(range(10))
    assert [type(c) for c in ds.constraints] == [OrdinalGroup]
    assert ds.feature_name(ds.negation_of("size<=m")) == "size>m"
    assert ds.feature_name(ds.negation_of("size=l")) == "size!=l"


def test_ordinal_without_negation_and_numeric_values_sorted():
    b = DataSpecBuilder(negation=False)
    b.add_ordinal("kids", [2, 0, 1])
    ds = b.build()
    assert ds.feature_names == ["kids=0", "kids=1", "kids=2", "kids<=0", "kids<=1", "kids>0", "kids>1"]
    assert ds.attributes["kids"].domain == (0, 1, 2)


def test_ordinal_rejects_duplicate_or_no_values():
    with pytest.raises(ValueError):
        DataSpecBuilder().add_ordinal("x", ["a", "b", "a"])
    with pytest.raises(ValueError):
        DataSpecBuilder().add_ordinal("x", [])


def test_ordinal_propagation():
    b = DataSpecBuilder()
    b.add_ordinal("size", ["s", "m", "l"])
    ds = b.build()
    f = ds.feature_index
    assert _true(ds, ds.propagate({f("size=m"): True})) == [
        "size!=l", "size!=s", "size<=m", "size=m", "size>s"]
    # an interval of one value is that value
    assert "size=m" in _true(ds, ds.propagate({f("size<=m"): True, f("size>s"): True}))
    # two inequalities leave one value
    assert "size=l" in _true(ds, ds.propagate({f("size!=s"): True, f("size!=m"): True}))
    with pytest.raises(ValueError):
        ds.propagate({f("size<=s"): True, f("size>m"): True})
    with pytest.raises(ValueError):
        ds.propagate({f("size=s"): True, f("size=l"): True})


def test_ordinal_rule_consistency_and_implied_conditions():
    b = DataSpecBuilder()
    b.add_ordinal("size", ["s", "m", "l"])
    ds = b.build()
    f = ds.feature_index
    assert not Rule.from_pos_neg(pos=[f("size=s"), f("size>m")], target="t", dataspec=ds).is_consistent()
    implied = Rule.from_pos_neg(pos=[f("size=l")], target="t", dataspec=ds).implied_conditions()
    assert {ds.feature_name(lit.feature) for lit in implied} >= {"size>s", "size>m", "size!=s", "size!=m"}


# -- ordinal attributes: reading data ------------------------------------------

def test_binarize_ordinal_by_position():
    b = DataSpecBuilder()
    b.add_ordinal("size", ["small", "medium", "large"])
    ds = b.build()
    df = pd.DataFrame({"size": ["small", "large", None, "huge"]})
    X = binarize(ds, df)
    on = lambda row: sorted(ds.feature_name(j) for j in np.flatnonzero(X[row]))
    # alphabetically "large" < "medium" < "small"; by position it is the largest
    assert on(0) == ["size!=large", "size!=medium", "size<=medium", "size<=small", "size=small"]
    assert on(1) == ["size!=medium", "size!=small", "size=large", "size>medium", "size>small"]
    assert on(2) == [] and on(3) == []        # missing, and a value outside the order


def test_binarize_ordinal_numeric_values_compare_numerically():
    b = DataSpecBuilder(negation=False)
    b.add_ordinal("kids", [0, 1, 2])
    ds = b.build()
    X = binarize(ds, pd.DataFrame({"kids": [1, 5, np.nan]}))
    names = ds.feature_names
    assert [names[j] for j in np.flatnonzero(X[0])] == ["kids=1", "kids<=1", "kids>0"]
    assert [names[j] for j in np.flatnonzero(X[1])] == ["kids>0", "kids>1"]  # unseen 5: no "="
    assert not X[2].any()


def test_binarize_ordinal_majority_imputes_lower_median():
    b = DataSpecBuilder(negation=False)
    b.add_ordinal("size", ["s", "m", "l"])
    ds = b.build()
    df = pd.DataFrame({"size": ["s", "l", "l", "m", None]})   # sorted s, m, l, l -> lower median m
    X = binarize(ds, df, missing_strategy=MissingStrategy.MAJORITY)
    assert X[4, ds.feature_index("size=m")]


def test_binarize_ordinal_separate_missing():
    b = DataSpecBuilder()
    b.add_ordinal("size", ["s", "m", "l"], missing_name="?", missing_values=("?",))
    ds = b.build(missing_strategy=MissingStrategy.SEPARATE)
    X = binarize(ds, pd.DataFrame({"size": ["m", "?"]}))
    assert [ds.feature_name(j) for j in np.flatnonzero(X[1])] == ["size=?"]
    # "missing" is a position of its own, outside the order
    f = ds.feature_index
    assert "size<=s" not in _true(ds, ds.propagate({f("size=?"): True}))
    assert ds.propagate({f("size=?"): True})[f("size>m")] is False


def test_read_csv_infers_small_numeric_sets_as_ordinal():
    rows = ["kids,flag,income,label"]
    for i in range(30):
        rows.append(f"{i % 4},{i % 2},{1000 + 37 * i},{'pos' if i % 4 > 1 else 'neg'}")
    rep = read_csv(io.StringIO("\n".join(rows)), target="label")
    attrs = rep.spec.attributes
    assert attrs["kids"].type == AttributeType.ORDINAL and attrs["kids"].domain == (0, 1, 2, 3)
    assert attrs["flag"].type == AttributeType.BINARY
    assert attrs["income"].type == AttributeType.NUMERIC
    assert "kids!=0" in rep.spec.feature_names and "kids>1" in rep.spec.feature_names


# -- merging and display --------------------------------------------------------

def test_merge_ordinal():
    def spec(values):
        b = DataSpecBuilder()
        b.add_ordinal("kids", values)
        return b.build()
    merged = merge_dataspecs(spec([0, 2]), spec([1, 2, 3])).build()
    assert merged.attributes["kids"].domain == (0, 1, 2, 3)

    def named(values):
        b = DataSpecBuilder()
        b.add_ordinal("size", values)
        return b.build()
    assert merge_dataspecs(named(["s", "m"]), named(["s", "m"])).build().attributes["size"].domain == ("s", "m")
    with pytest.raises(ValueError):
        merge_dataspecs(named(["s", "m"]), named(["m", "s"]))


def test_ordinal_display_merges_intervals_and_value_sets():
    b = DataSpecBuilder()
    b.add_ordinal("kids", [0, 1, 2, 3])
    ds = b.build()
    f = ds.feature_index
    interval = Rule.from_pos_neg(pos=[f("kids>0"), f("kids<=2")], target="t", dataspec=ds)
    assert interval.to_string(fmt="conditions", pretty=True) == "0 < kids <= 2"
    excluded = Rule.from_pos_neg(pos=[f("kids!=0"), f("kids!=3")], target="t", dataspec=ds)
    assert excluded.to_string(fmt="conditions", pretty=True) == "kids ∉ {0, 3}"


def test_max_ordinal_values_is_configurable_and_independent_of_max_intervals():
    df = pd.DataFrame({"x": [i % 6 for i in range(60)], "y": ["a", "b"] * 30})
    assert build_dataspec(df, target="y").build().attributes["x"].type == AttributeType.ORDINAL
    assert build_dataspec(df, target="y", discretizer=TreeDiscretizer(max_intervals=3)).build().attributes["x"].type == AttributeType.ORDINAL
    numeric = build_dataspec(df, target="y", max_ordinal_values=5).build()
    assert numeric.attributes["x"].type == AttributeType.NUMERIC


# -- the constraints, checked exhaustively against the data --------------------

import itertools


def _check_constraints_against_data(ds, column, raw_values, present):
    """`ds.propagate` against the truth: the rows `binarize` makes of
    `raw_values` (`present` marks those that aren't missing values).

    1. Sound for every set of features fixed True (what a rule's
       conditions are): every forced value holds on every row that has
       all of them, and a contradiction is raised exactly when no row has.
    2. Exact for every assignment of up to three features to True/False,
       over the present rows: forced is exactly what all present rows
       agreeing with the assignment have in common (a missing row
       falsifies every feature, which no exhaustive group models)."""
    X = binarize(ds, pd.DataFrame({column: raw_values}))
    n = ds.n_features
    for r in range(1, n + 1):
        for fixed in itertools.combinations(range(n), r):
            rows = X[np.all(X[:, list(fixed)], axis=1)]
            if len(rows) == 0:
                with pytest.raises(ValueError):
                    ds.propagate({i: True for i in fixed})
                continue
            for i, v in ds.propagate({i: True for i in fixed}).items():
                assert (rows[:, i] == v).all(), (ds.feature_names, [ds.feature_names[f] for f in fixed], i)
    P = X[np.asarray(present)]
    for r in range(1, 4):
        for fixed in itertools.combinations(range(n), r):
            for values in itertools.product((True, False), repeat=r):
                rows = P[np.all(P[:, list(fixed)] == np.array(values), axis=1)]
                assignment = dict(zip(fixed, values))
                if len(rows) == 0:
                    with pytest.raises(ValueError):
                        ds.propagate(assignment)
                    continue
                expected = {i: bool(rows[0, i]) for i in range(n) if (rows[:, i] == rows[0, i]).all()}
                assert ds.propagate(assignment) == expected, (ds.feature_names, assignment)


@pytest.mark.parametrize("negation", [True, False])
@pytest.mark.parametrize("values", [[0, 1, 3], [1, 2, 3, 4]])
def test_ordinal_constraints_numeric_domain_exact(values, negation):
    # exact over the known values (an unseen number, which no training
    # example has, still compares numerically when the data is read --
    # see test_binarize_ordinal_numeric_values_compare_numerically)
    b = DataSpecBuilder(negation=negation)
    b.add_ordinal("x", values)
    ds = b.build()
    raw = list(values) + [np.nan]
    _check_constraints_against_data(ds, "x", raw, [True] * len(values) + [False])


@pytest.mark.parametrize("negation", [True, False])
# (a single named value has one position and so, like a one-value nominal
# attribute, no constraint: one would force `x=a` into every rule's closure)
@pytest.mark.parametrize("values", [["s", "m", "l"], ["a", "b"]])
def test_ordinal_constraints_named_values_exact(values, negation):
    b = DataSpecBuilder(negation=negation)
    b.add_ordinal("x", values)
    ds = b.build()
    raw = list(values) + ["unknown", None]          # both count as missing
    _check_constraints_against_data(ds, "x", raw, [True] * len(values) + [False, False])


@pytest.mark.parametrize("numeric", [True, False])
def test_ordinal_constraints_with_separate_missing_exact(numeric):
    values = [0, 1, 3] if numeric else ["s", "m", "l"]
    b = DataSpecBuilder()
    b.add_ordinal("x", values, missing_name="?", missing_values=("?",))
    ds = b.build(missing_strategy=MissingStrategy.SEPARATE)
    raw = list(values) + ["?"]
    _check_constraints_against_data(ds, "x", raw, [True] * len(raw))  # "?" is a value of its own here


@pytest.mark.parametrize("families", [("ge",), ("le",), ("ge", "le")])
def test_numeric_constraints_sound(families):
    b = DataSpecBuilder()
    b.add_numeric("x", [10, 20, 30] if "ge" in families else (),
                  le_thresholds=[10, 20, 30] if "le" in families else ())
    ds = b.build()
    raw = [5, 10, 15, 20, 25, 30, 35, np.nan]
    X = binarize(ds, pd.DataFrame({"x": raw}))
    for r in range(1, 4):
        for fixed in itertools.combinations(range(ds.n_features), r):
            rows = X[np.all(X[:, list(fixed)], axis=1)]
            try:
                closure = ds.propagate({i: True for i in fixed})
            except ValueError:
                assert len(rows) == 0, [ds.feature_names[f] for f in fixed]
                continue
            # the two families don't cross-check: a conjunction no row has
            # may pass, but nothing forced may contradict a row that has it
            for i, v in closure.items():
                assert (rows[:, i] == v).all(), ([ds.feature_names[f] for f in fixed], ds.feature_names[i])
