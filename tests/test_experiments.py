"""`pyrulearn.experiments.runner`/`stats`/`report`: offline (a tiny
synthetic dataset stands in for `CatalogEntry.load()` -- nothing here
hits the network), with `Pypper`/`PFossil` as fast, deterministic-enough
real learners rather than a hand-rolled fake one."""

import time

import numpy as np
import pandas as pd
import pytest

from pyrulearn.experiments.catalog import CatalogEntry
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import TimeoutRunner, run_cv
from pyrulearn.experiments.stats import (
    critical_difference, critical_difference_diagram, friedman_test, mean_rank,
)
from pyrulearn.learners.seco import PFossil, Pypper


def _synthetic_entry(name: str, n: int = 60, seed: int = 0) -> CatalogEntry:
    """A `CatalogEntry` whose `load()` is monkeypatched to a small,
    in-memory, easily-learnable binary dataset -- no OpenML id, no
    network access. One numeric and one nominal feature, both
    informative, so `build_dataspec`/`binarize` exercise both code paths."""
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)
    nom = rng.choice(["p", "q"], size=n)
    noise = rng.normal(scale=0.2, size=n)
    y = np.where((num + np.where(nom == "p", 0.5, -0.5) + noise) > 0, "pos", "neg")
    df = pd.DataFrame({"num": num, "nom": nom, "target": y})
    entry = CatalogEntry(name=name)
    entry.load = lambda: (df, "target")
    return entry


LEARNERS = [Pypper(random_state=0), PFossil(random_state=0)]


def test_run_cv_produces_the_documented_columns():
    entries = [_synthetic_entry("synth_a", seed=0), _synthetic_entry("synth_b", n=50, seed=1)]
    results = run_cv(LEARNERS, entries, n_folds=3, fit_timeout=None, random_state=0, verbose=False)

    assert set(results["dataset"]) == {"synth_a", "synth_b"}
    assert set(results["learner"]) == {"Pypper", "PFossil"}
    assert len(results) == 2 * 3 * 2  # datasets x folds x learners
    for col in ("dataset", "fold", "learner", "fit_time", "error", "accuracy", "n_rules", "n_conditions"):
        assert col in results.columns
    # a clean, easily-separable synthetic dataset: no fit should fail
    assert results["error"].isna().all()
    assert results["accuracy"].between(0, 1).all()
    assert (results["fit_time"] >= 0).all()


def test_run_cv_measure_fn_adds_extra_columns():
    entries = [_synthetic_entry("synth_a")]

    def extra(model, test_rep):
        return {"n_predicted_pos": int((model.predict(test_rep) == "pos").sum())}

    results = run_cv(LEARNERS, entries, n_folds=2, fit_timeout=None, random_state=0,
                     measure_fn=extra, verbose=False)
    assert "n_predicted_pos" in results.columns
    assert results["n_predicted_pos"].notna().all()


def test_run_cv_caches_finished_folds(tmp_path):
    calls = {"n": 0}

    class CountingPypper(Pypper):
        # `_fit_default` (not `fit`) -- `fit(data)` with no `model=` calls
        # `_fit_default`, which itself calls back into `self.fit(data,
        # model=...)`; counting in `fit` directly would double-count.
        def _fit_default(self, data):
            calls["n"] += 1
            return super()._fit_default(data)

    entries = [_synthetic_entry("synth_a"), _synthetic_entry("synth_b", n=50, seed=1)]
    cache_dir = str(tmp_path / "cv_cache")
    learner = CountingPypper(random_state=0)

    first = run_cv([learner], entries, n_folds=3, fit_timeout=None, random_state=0,
                   cache_dir=cache_dir, verbose=False)
    assert calls["n"] == 2 * 3  # datasets x folds, once each

    second = run_cv([learner], entries, n_folds=3, fit_timeout=None, random_state=0,
                    cache_dir=cache_dir, verbose=False)
    assert calls["n"] == 2 * 3  # every row came from the cache -- no new fits
    pd.testing.assert_frame_equal(
        first.sort_values(["dataset", "fold"]).reset_index(drop=True),
        second.sort_values(["dataset", "fold"]).reset_index(drop=True),
    )


def _sleep_then_return(seconds: float) -> str:
    """Module-level (picklable by reference) -- see `TimeoutRunner`'s
    docstring on why that's required."""
    time.sleep(seconds)
    return "done"


def test_timeout_runner_records_a_timeout_instead_of_raising():
    # timeout=3.0 leaves headroom for the replacement worker's own spawn
    # (~1-2s: a fresh `python -m multiprocessing.spawn` reimporting this
    # test module and its dependencies) on top of the trivial call itself.
    runner = TimeoutRunner(timeout=3.0)
    try:
        result, error = runner.run(_sleep_then_return, 30.0)
        assert result is None
        assert error == "timeout"
        # the worker was replaced -- a fast call afterwards still works
        result, error = runner.run(_sleep_then_return, 0.0)
        assert result == "done" and error is None
    finally:
        runner.close()


def test_run_cv_records_a_failed_fit_without_raising():
    class AlwaysFails(Pypper):
        def fit(self, data, model=None, **model_kwargs):
            raise RuntimeError("synthetic failure")

    entries = [_synthetic_entry("synth_a")]
    results = run_cv([AlwaysFails()], entries, n_folds=2, fit_timeout=None, random_state=0, verbose=False)
    assert (results["error"] == "RuntimeError: synthetic failure").all()
    assert results["accuracy"].isna().all()


# ------------------------------------------------------------------- stats

@pytest.fixture(scope="module")
def cv_results():
    entries = [_synthetic_entry(f"synth_{i}", n=60, seed=i) for i in range(4)]
    return run_cv(LEARNERS, entries, n_folds=3, fit_timeout=None, random_state=0, verbose=False)


def test_mean_rank_ranks_every_learner(cv_results):
    ranks = mean_rank(cv_results, "accuracy", higher_is_better=True)
    assert set(ranks.index) == {"Pypper", "PFossil"}
    assert ranks.between(1, 2).all()
    assert list(ranks.index) == list(ranks.sort_values().index)  # best-first


def test_mean_rank_ties_a_failed_pair_for_last():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2", "d2"],
        "learner": ["A", "B", "A", "B"],
        "accuracy": [0.9, 0.8, np.nan, 0.7],
    })
    ranks = mean_rank(results, "accuracy", higher_is_better=True)
    # d1: A=1, B=2. d2: B=1 (A failed, tied for "last" among 2 -> rank 2)
    assert ranks["A"] == pytest.approx((1 + 2) / 2)
    assert ranks["B"] == pytest.approx((2 + 1) / 2)


def test_friedman_test_runs_on_a_complete_matrix():
    # friedmanchisquare needs >=3 learners (treatments) -- cv_results only
    # has Pypper/PFossil, so this one uses a synthetic 3-learner table.
    results = pd.DataFrame({
        "dataset": [d for d in ("d1", "d2", "d3", "d4", "d5") for _ in range(3)],
        "learner": ["A", "B", "C"] * 5,
        "accuracy": [0.9, 0.8, 0.7, 0.85, 0.75, 0.65, 0.95, 0.9, 0.6, 0.8, 0.82, 0.7, 0.7, 0.9, 0.75],
    })
    result = friedman_test(results, "accuracy")
    assert result.n_datasets == 5
    assert result.n_learners == 3
    assert 0.0 <= result.pvalue <= 1.0


def test_friedman_test_needs_at_least_three_learners(cv_results):
    with pytest.raises(ValueError, match="needs >=3 learners"):
        friedman_test(cv_results, "accuracy")  # only Pypper/PFossil


def test_critical_difference_matches_demsar_table():
    # Demsar (2006), Table 5(b), alpha=0.05 -- q_alpha per k: 2->1.960,
    # 3->2.343, 10->3.164 (checked to 3 decimals; our q_alpha comes from
    # scipy's studentized_range, more precise than the published table).
    for k, q_alpha in ((2, 1.960), (3, 2.343), (10, 3.164)):
        cd = critical_difference(k, n_datasets=10, alpha=0.05)
        assert cd == pytest.approx(q_alpha * np.sqrt(k * (k + 1) / 60), abs=1e-3)
    cd_k2 = critical_difference(2, 10, alpha=0.05)
    cd_k10 = critical_difference(10, 10, alpha=0.05)
    assert cd_k10 > cd_k2  # more learners compared -> wider critical difference


def test_critical_difference_diagram_returns_an_axes(cv_results):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = critical_difference_diagram(cv_results, "accuracy")
    assert ax.get_xlabel() == "mean rank (accuracy)"


# ------------------------------------------------------------------ report

def test_render_setup_section_wraps_the_description():
    out = render_setup_section("  Some experiment description.  ")
    assert out == "## Setup\n\nSome experiment description.\n\n"


def test_render_results_table_has_an_overall_row_and_marks_failures():
    results = pd.DataFrame({
        "dataset": ["d1", "d1", "d2"],
        "learner": ["A", "A", "A"],
        "accuracy": [0.8, np.nan, 0.6],
    })
    out = render_results_table(results, ["accuracy"])
    assert "| dataset | accuracy |" in out
    assert "| overall |" in out
    assert "n/a" not in out  # d1's two rows average to a real number, not NaN

    all_nan = pd.DataFrame({"dataset": ["d1"], "learner": ["A"], "accuracy": [np.nan]})
    assert "n/a" in render_results_table(all_nan, ["accuracy"])
