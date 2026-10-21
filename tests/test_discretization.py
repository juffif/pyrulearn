"""Discretizers: `TreeDiscretizer`, `FusinterDiscretizer`,
`EqualFrequencyDiscretizer`, `EqualWidthDiscretizer`, the
`BoundaryCorrection` and `PerClass` wrappers, and how `build_dataspec`/`run_cv` take them."""

import numpy as np
import pandas as pd
import pytest

from pyrulearn.data.discretization import (
    BoundaryCorrection,
    Discretizer,
    EqualFrequencyDiscretizer,
    EqualWidthDiscretizer,
    FusinterDiscretizer,
    PerClass,
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


def test_boundary_correction_merges_thresholds_with_the_same_nearest_boundary():
    x = np.arange(1, 13, dtype=float)
    y = np.array(list("aaaaabbbbbba"))  # boundaries after 5 (5 below) and after 11 (11 below)
    # equal frequency proposes 4.5 (4 below) and 8.5 (8 below); both are
    # nearest to the boundary after 5 (8 is equally far from 5 and 11:
    # the lower one), so they merge -- the far boundary is not used
    assert EqualFrequencyDiscretizer(3).thresholds(x) == [4.5, 8.5]
    assert BoundaryCorrection(EqualFrequencyDiscretizer(3)).thresholds(x, y) == [5.5]


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
                                              BoundaryCorrection(TreeDiscretizer(8)), EqualWidthDiscretizer(8),
                                              FusinterDiscretizer(), FusinterDiscretizer(max_intervals=8),
                                              PerClass(TreeDiscretizer(8)))}
    assert len(tokens) == 8


def test_equal_width_splits_the_range_into_equal_parts():
    x = np.arange(1, 13, dtype=float)
    assert EqualWidthDiscretizer(4).thresholds(x) == [3.5, 6.5, 9.5]
    assert not EqualWidthDiscretizer(4).supervised


def test_equal_width_merges_empty_intervals():
    # every cut of 1..100 falls into the one gap between 3 and 100
    assert EqualWidthDiscretizer(4).thresholds(np.array([1, 2, 3, 100.0])) == [52.0]
    assert EqualWidthDiscretizer(4).thresholds(np.array([7.0, 7.0, np.nan])) == []


def _noisy_column(seed=0, n=600):
    rng = np.random.default_rng(seed)
    x = np.round(rng.normal(size=n), 2)
    y = (x + rng.normal(scale=0.8, size=n) > 0).astype(int)
    return x, y


def test_fusinter_keeps_a_single_class_change():
    x = np.arange(1, 13, dtype=float)
    assert FusinterDiscretizer().thresholds(x, np.array(list("aaaaaabbbbbb"))) == [6.5]
    assert FusinterDiscretizer().thresholds(x, np.zeros(12)) == []


def test_fusinter_only_cuts_at_boundary_points_and_stops_at_a_local_optimum():
    x, y = _noisy_column()
    disc = FusinterDiscretizer()
    ts = disc.thresholds(x, y)
    assert 0 < len(ts) and ts == sorted(set(ts))
    assert all(_is_boundary(t, x, y) for t in ts)
    assert BoundaryCorrection(disc).thresholds(x, y) == ts
    # no merge of two adjacent intervals lowers the criterion any further
    edges = [-np.inf] + ts + [np.inf]
    counts = np.array([[np.sum((x >= lo) & (x < hi) & (y == c)) for c in (0, 1)]
                       for lo, hi in zip(edges, edges[1:])], dtype=float)
    terms = disc._terms(counts, len(x))
    merged = disc._terms(counts[:-1] + counts[1:], len(x))
    assert np.all(terms[:-1] + terms[1:] - merged <= 1e-12)


def test_fusinter_max_intervals_caps_the_number_of_intervals():
    x, y = _noisy_column(1)
    free = FusinterDiscretizer().thresholds(x, y)
    for k in (2, 3, 5):
        capped = FusinterDiscretizer(max_intervals=k).thresholds(x, y)
        assert len(capped) <= k - 1
        assert len(capped) <= len(free)


def test_fusinter_smaller_alpha_gives_fewer_intervals():
    x, y = _noisy_column(2)
    many = FusinterDiscretizer(alpha=0.999).thresholds(x, y)
    few = FusinterDiscretizer(alpha=0.9).thresholds(x, y)
    assert len(few) < len(many)


def test_fusinter_parameters_are_validated():
    with pytest.raises(ValueError):
        FusinterDiscretizer(alpha=1.5)
    with pytest.raises(ValueError):
        FusinterDiscretizer(lam=0)
    with pytest.raises(ValueError):
        FusinterDiscretizer(max_intervals=1)
    with pytest.raises(ValueError):
        FusinterDiscretizer().thresholds(np.arange(5.0))
    assert repr(FusinterDiscretizer()) == "FusinterDiscretizer(alpha=0.975, lam=1.0, max_intervals=None)"


def _three_classes():
    # a large split at 45, and a small class c on 90..94
    x = np.arange(100, dtype=float)
    y = np.where(x < 45, "a", "b").astype(object)
    y[(x >= 90) & (x < 95)] = "c"
    return x, y


def test_per_class_places_cuts_for_a_small_class_a_global_tree_misses():
    x, y = _three_classes()
    assert TreeDiscretizer(2).thresholds(x, y) == [44.5]
    per_class = PerClass(TreeDiscretizer(2)).thresholds(x, y)
    assert 44.5 in per_class
    assert any(89 < t < 95 for t in per_class)
    assert per_class == sorted(set(per_class))


def test_per_class_is_the_union_of_the_one_against_the_rest_problems():
    x, y = _three_classes()
    tree = TreeDiscretizer(3)
    union = set()
    for c in ("a", "b", "c"):
        union.update(tree.thresholds(x, (y == c).astype(int)))
    assert PerClass(tree).thresholds(x, y) == sorted(union)


def test_per_class_with_two_classes_or_an_unsupervised_discretizer_changes_nothing():
    x, y = _noisy_column(3)
    assert PerClass(TreeDiscretizer(5)).thresholds(x, y) == TreeDiscretizer(5).thresholds(x, y)
    x3, y3 = _three_classes()
    assert PerClass(EqualWidthDiscretizer(4)).thresholds(x3, y3) == EqualWidthDiscretizer(4).thresholds(x3)


def test_per_class_needs_labels_and_a_discretizer():
    with pytest.raises(ValueError):
        PerClass(TreeDiscretizer()).thresholds(np.arange(5.0))
    with pytest.raises(TypeError):
        PerClass(3)
    assert PerClass(TreeDiscretizer(3)).supervised
    assert repr(PerClass(TreeDiscretizer(3))) == "PerClass(TreeDiscretizer(max_intervals=3))"


class _PerClassFixed(Discretizer):
    """Test double: a different threshold per positive class, all in the
    gap between the observed values 1 and 3, plus one shared one."""

    def thresholds(self, values, y=None):
        first = int(np.flatnonzero(np.asarray(y) == 1)[0])
        return [[1.4, 5.0], [2.0, 5.0], [2.6]][first % 3]


def test_per_class_keeps_one_threshold_per_gap_between_observed_values():
    x = np.array([1, 3, 4, 6, 1, 3, 4, 6, 7], dtype=float)
    y = np.array(["a", "b", "c", "a", "b", "c", "a", "b", "c"])
    # a's run gives 1.4 and 5, b's 2.0 and 5, c's 2.6: one per gap
    assert PerClass(_PerClassFixed()).thresholds(x, y) == [1.4, 5.0]


def test_tree_without_cap_cuts_at_exactly_the_boundary_points():
    from pyrulearn.data.discretization import _Column
    x, y = _noisy_column(4)
    full = TreeDiscretizer(max_intervals=None).thresholds(x, y)
    column = _Column(x, y)
    # the same gaps between observed values (the prettified numbers in a
    # gap may differ: scikit-learn computes its split points in float32)
    gaps = [int(np.searchsorted(column.distinct, t)) - 1 for t in full]
    assert gaps == list(column.boundaries)
    assert len(full) > len(TreeDiscretizer().thresholds(x, y))
    with pytest.raises(ValueError):
        TreeDiscretizer(max_intervals=1)
