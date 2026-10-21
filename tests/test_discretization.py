"""Discretizers: `TreeDiscretizer`, `EqualFrequencyDiscretizer`, the
`BoundaryCorrection` wrapper, and how `build_dataspec`/`run_cv` take them."""

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data.discretization import (
    BoundaryCorrection,
    Discretizer,
    EqualFrequencyDiscretizer,
    TreeDiscretizer,
)
from pyrulearn.data.io import build_dataspec


def _is_boundary(t, values, y):
    """Whether threshold `t` separates two adjacent observed values that
    aren't both of one and the same single class."""
    below, above = values[values < t], values[values > t]
    lo, hi = below.max(), above.min()
    classes = set(y[(values == lo) | (values == hi)])
    return len(classes) > 1


def test_equal_frequency_splits_into_equally_sized_intervals():
    x = np.arange(1, 13, dtype=float)
    assert EqualFrequencyDiscretizer(3).thresholds(x) == [4.5, 8.5]
    assert EqualFrequencyDiscretizer(4).thresholds(x) == [3.5, 6.5, 9.5]


def test_equal_frequency_is_unsupervised_and_ignores_missing_values():
    disc = EqualFrequencyDiscretizer(2)
    assert not disc.supervised
    assert disc.thresholds(np.array([1, 2, np.nan, 3, 4, np.nan])) == [2.5]


def test_equal_frequency_degenerate_columns():
    disc = EqualFrequencyDiscretizer(8)
    assert disc.thresholds(np.array([5.0, 5.0, 5.0])) == []
    assert disc.thresholds(np.array([np.nan, 1.0])) == []
    # fewer distinct values than intervals: a threshold in every gap
    assert disc.thresholds(np.array([1, 1, 2, 2, 3, 3], dtype=float)) == [1.5, 2.5]


def test_equal_frequency_thresholds_are_prettified():
    x = np.array([0.1234, 0.2345, 0.3456, 0.4567])
    assert EqualFrequencyDiscretizer(2).thresholds(x) == [0.3]


def test_boundary_correction_keeps_one_cut_where_the_class_changes_once():
    x = np.arange(1, 13, dtype=float)
    y = np.array(list("aaaaaabbbbbb"))
    assert BoundaryCorrection(EqualFrequencyDiscretizer(3)).thresholds(x, y) == [6.5]


def test_boundary_correction_moves_each_cut_to_the_nearest_boundary():
    x = np.arange(1, 13, dtype=float)
    y = np.array(list("aabbaabbaabb"))
    # equal frequency proposes 3.5, 6.5, 9.5; 3.5 and 9.5 sit inside a
    # single-class pair and move to the nearest boundary (ties: lower gap)
    assert BoundaryCorrection(EqualFrequencyDiscretizer(4)).thresholds(x, y) == [2.5, 6.5, 8.5]


def test_boundary_correction_counts_a_mixed_value_as_a_boundary():
    x = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=float)
    y = np.array(list("abaaaabb"))  # value 1 has both classes
    assert BoundaryCorrection(EqualFrequencyDiscretizer(2)).thresholds(x, y) == [1.5]


def test_boundary_correction_needs_labels_and_a_discretizer():
    with pytest.raises(ValueError):
        BoundaryCorrection(EqualFrequencyDiscretizer(4)).thresholds(np.arange(5.0))
    with pytest.raises(TypeError):
        BoundaryCorrection("quantile")
    assert BoundaryCorrection(EqualFrequencyDiscretizer(4)).supervised


def test_boundary_correction_on_a_single_class_column_gives_no_threshold():
    x = np.arange(10, dtype=float)
    assert BoundaryCorrection(EqualFrequencyDiscretizer(4)).thresholds(x, np.zeros(10)) == []


def test_boundary_correction_only_returns_boundary_points():
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=300), 1)
    x[rng.random(300) < 0.05] = np.nan
    y = (np.nan_to_num(x) + rng.normal(scale=0.7, size=300) > 0).astype(int)
    for k in (3, 5, 8, 12):
        ts = BoundaryCorrection(EqualFrequencyDiscretizer(k)).thresholds(x, y)
        assert 0 < len(ts) <= k - 1 and ts == sorted(set(ts))
        observed = ~np.isnan(x)
        assert all(_is_boundary(t, x[observed], y[observed]) for t in ts)


def test_boundary_correction_leaves_tree_thresholds_alone():
    # a decision tree's split points already lie on boundary points
    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 10, size=200), 2)
    y = ((x > 3) & (x < 7)).astype(int) ^ (rng.random(200) < 0.1)
    tree = TreeDiscretizer(6)
    assert BoundaryCorrection(tree).thresholds(x, y) == tree.thresholds(x, y)


def test_discretizers_compare_and_print_by_their_parameters():
    assert TreeDiscretizer(5) == TreeDiscretizer(5) != TreeDiscretizer(6)
    assert EqualFrequencyDiscretizer(5) != TreeDiscretizer(5)
    corrected = BoundaryCorrection(EqualFrequencyDiscretizer(8))
    assert corrected == BoundaryCorrection(EqualFrequencyDiscretizer(8))
    assert len({corrected, BoundaryCorrection(EqualFrequencyDiscretizer(8))}) == 1
    assert repr(corrected) == "BoundaryCorrection(EqualFrequencyDiscretizer(max_intervals=8))"
    with pytest.raises(ValueError):
        EqualFrequencyDiscretizer(1)
    assert isinstance(corrected, Discretizer)


def _numeric_frame():
    rng = np.random.default_rng(2)
    x = rng.normal(size=120)
    return pd.DataFrame({"x": x, "y": np.where(x + rng.normal(scale=0.5, size=120) > 0, "p", "n")})


def test_build_dataspec_defaults_to_the_tree_discretizer():
    df = _numeric_frame()
    default = build_dataspec(df, target="y").build()
    explicit = build_dataspec(df, target="y", discretizer=TreeDiscretizer()).build()
    assert default.feature_names == explicit.feature_names


def test_build_dataspec_with_an_unsupervised_discretizer_needs_no_target():
    df = _numeric_frame().drop(columns="y")
    spec = build_dataspec(df, discretizer=EqualFrequencyDiscretizer(4)).build()
    assert len(spec.feature_names) == 2 * 3  # three thresholds, x>=t and x<t each
    with pytest.raises(ValueError):
        build_dataspec(df, discretizer=BoundaryCorrection(EqualFrequencyDiscretizer(4)))
    with pytest.raises(ValueError):
        build_dataspec(df)  # the default tree is supervised


def test_run_cv_cache_key_of_the_default_tree_is_unchanged():
    from pyrulearn.experiments.runner import _discretizer_token
    assert _discretizer_token(TreeDiscretizer()) == "8"  # what the key held as max_intervals
    assert _discretizer_token(TreeDiscretizer(5)) == "5"
    tokens = {_discretizer_token(d) for d in (TreeDiscretizer(8), EqualFrequencyDiscretizer(8),
                                              BoundaryCorrection(EqualFrequencyDiscretizer(8)),
                                              BoundaryCorrection(TreeDiscretizer(8)))}
    assert len(tokens) == 4
