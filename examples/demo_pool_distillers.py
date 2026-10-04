"""
examples/demo_pool_distillers.py
=================================

Rule-pool generators crossed with rule distillers: a pool is a large,
unpruned `FlatRuleSet` of candidate rules from one of three unrelated
*generation* mechanisms --

- `pyrulearn.learners.associative.CARMiner` -- itemset mining (CBA-RG);
- `pyrulearn.interfaces.sklearn.SKLRandomForest` -- a random forest's
  leaves;
- `pyrulearn.learners.pylord.pylord_candidates` -- LORD's raw per-example
  local-search pool, before its own coverage-filter narrows it --

and a *distiller* compresses or reorganizes that pool into its own,
smaller model: `RuleFit` (sparse L1 logistic regression over the pool's
coverage), `CBA` (precedence sort + database-coverage selection), `CMAR`
(chi-square significance filter + per-class coverage pruning + weighted
voting), `IDS` (Interpretable Decision Sets -- a diverse, accurate subset
via Smooth Local Search over 7 weighted objectives, or its cheap
`optimizer="greedy"` fallback as `IDS-greedy`), and two baselines with no
distillation at all, just the whole pool wrapped in a `FlatRuleSet` and
predicted from directly (`Max`, its default Laplace-scored combiner, and
`Vote`, a plain majority vote) -- the four proper distillers share
`RuleDistiller`'s identical `rules=` constructor argument, so any pool
plugs into any of them unchanged.

Two studies, not one, because the three generators don't scale alike:

- **Study 1** (`vote`, `SPECT`, `hepatitis`, `tic-tac-toe`,
  `heart-statlog` -- all narrow, binary-feature counts kept under 60 by
  `max_intervals=3`) uses all three generators. Even here `CARMiner`'s
  pool runs into the thousands while the other two stay in the hundreds
  -- itemset mining doesn't taper off the way a bounded random forest or
  a one-search-per-row LORD pool does, even capped at `max_len=3` and a
  support raised five times over the default (0.05). On anything wider
  this gets out of hand fast: measured once on `mushroom` (220 binary
  features), `CARMiner` alone mined 588,738 rules in 57s, and
  `pylord_candidates` took 113s just for its pool, against a random
  forest's 1.8s/647 rules -- neither belongs in a demo at that scale.
- **Study 2** (`credit-g`, `kr-vs-kp`, `sick`, `spambase`, `mushroom`,
  `bank-marketing`, `adult` -- 1,000 to 48,842 rows) therefore mostly
  relies on `SKLRandomForest` -- the one generator whose pool size and
  build time stay flat regardless of the data's width or row count
  (measured: ~700-800 rules, well under 2s, from `vote`'s 435 rows up
  through `adult`'s 48,842). `pylord_candidates` is tried here too: its
  cost is one local search per training row, not flat like a forest's,
  but growing super-linearly (measured on the smaller Study 2 datasets:
  17.9s/3196 rows, 35.5s/3772 rows, 113.1s/8124 rows) -- feasible on the
  smaller end of Study 2 but expected to run long, or time out, on
  `bank-marketing`/`adult`. Pool-building therefore gets its own,
  longer timeout than fitting a distiller does (`POOL_FIT_TIMEOUT`,
  larger than `FIT_TIMEOUT`); a pool time-out is just another failure
  recorded in the pool-stats table, not a reason to skip the generator.

Each pool is built once per fold and handed to every distiller in that
study -- the whole point of `RuleDistiller.rules=` -- so the comparison
is "how differently do several distillers treat the *same* candidate
pool", not several independent fits that happen to start from similarly-sized
inputs.

Run: `python examples/demo_pool_distillers.py` with no arguments is the
**quick** default; its report isn't checked in. `--full` runs both
studies at full scale and writes the canonical report.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.experiments.catalog import Catalog, CatalogEntry
from pyrulearn.experiments.report import render_results_table, render_setup_section
from pyrulearn.experiments.runner import TimeoutRunner
from pyrulearn.experiments.stats import mean_rank
from pyrulearn.interfaces.sklearn import SKLRandomForest
from pyrulearn.learners.associative import CARMiner, CBA, CMAR
from pyrulearn.learners.base import NativeRuleLearner, produces
from pyrulearn.learners.ids import IDS
from pyrulearn.learners.pylord import pylord_candidates
from pyrulearn.learners.rulefit import RuleFit
from pyrulearn.models import FlatRuleSet, MajorityClass

RANDOM_STATE = 0
MAX_INTERVALS = 3
FIT_TIMEOUT = 300.0

# Pool-building gets its own, larger cap: pylord_candidates' one-search-
# per-row cost grows super-linearly (see the module docstring), so a
# LORD pool on bank-marketing/adult may need much longer than any
# distiller fit does. A pool time-out is recorded as a failure, same as
# a distiller time-out, not treated specially.
POOL_FIT_TIMEOUT = 1800.0

# itemset mining (CBA-RG) doesn't taper off with max_len alone; support
# raised 5x over the default (0.01 -> 0.05) still leaves CARMiner's pool
# one to two orders of magnitude bigger than the other two generators'
# -- see the module docstring
CAR_MIN_SUPPORT = 0.05
CAR_MAX_LEN = 3
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 3  # matches CAR_MAX_LEN -- a comparable complexity budget

# "IDS" is the paper-faithful Smooth Local Search optimizer (stochastic,
# no convergence guarantee for a non-monotone objective); "IDS-greedy"
# its cheap, deterministic fallback -- both kept side by side in Study 1
# since IDS's own speed/stability looks like it may depend on the
# optimizer, not just the data (see the module docstring). Study 2 drops
# plain "IDS": it was already the slowest distiller by far on Study 1's
# small datasets (SLS's cost doesn't track data size in any simple way),
# so it's not worth repeating on bigger ones -- "IDS-greedy" stays.
DISTILLER_NAMES = ["RuleFit", "CBA", "CMAR", "IDS", "IDS-greedy", "Max", "Vote"]
STUDY2_DISTILLER_NAMES = ["RuleFit", "CBA", "CMAR", "IDS-greedy", "Max", "Vote"]

STUDY1_DATASETS = ["vote", "SPECT", "hepatitis", "tic-tac-toe", "heart-statlog"]
STUDY1_POOLS = ["CAR", "LORD", "RF"]
STUDY1_FOLDS = 5
STUDY1_QUICK_DATASETS = ["vote", "hepatitis"]
STUDY1_QUICK_FOLDS = 3

STUDY2_DATASETS = ["credit-g", "kr-vs-kp", "sick", "spambase", "mushroom", "bank-marketing", "adult"]
STUDY2_POOLS = ["RF", "LORD"]
STUDY2_FOLDS = 3
STUDY2_QUICK_DATASETS = ["kr-vs-kp"]
STUDY2_QUICK_FOLDS = 3

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "demo_pool_distillers"
CACHE_DIR = os.path.join(HERE, "_pool_distillers_cache")


# =============================================================== NoDistill ===

class NoDistill(NativeRuleLearner):
    """No distillation at all: the pool wrapped directly in a
    `FlatRuleSet`, predicting by `combiner` over the whole pool as-is --
    two baselines every other distiller compresses relative to: `"max"`
    (`FlatRuleSet`'s own default, Laplace-scored) and `"vote"` (plain
    majority vote across every firing rule). Demo-local: not a
    generally reusable library feature, just two more points of
    comparison this demo needs."""

    def __init__(self, rules: Optional[FlatRuleSet] = None, combiner: str = "vote"):
        self.rules = rules
        self.combiner = combiner

    @produces(FlatRuleSet)
    def _fit_native(self, data: BooleanDataRepresentation, **kw) -> FlatRuleSet:
        return FlatRuleSet(list(self.rules.rules), default_prediction=MajorityClass(data),
                          combiner=self.combiner)


# ============================================================ generators ===

def build_pool(name: str, data: BooleanDataRepresentation) -> FlatRuleSet:
    """Module-level (picklable by reference for `TimeoutRunner`): build
    one named pool from `data`."""
    if name == "CAR":
        return CARMiner(min_support=CAR_MIN_SUPPORT, max_len=CAR_MAX_LEN).fit(data)
    if name == "RF":
        return SKLRandomForest(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                               random_state=RANDOM_STATE).fit(data, model=FlatRuleSet)
    if name == "LORD":
        return pylord_candidates(data, random_state=RANDOM_STATE)
    raise ValueError(f"unknown pool generator {name!r}")


def build_distiller(name: str, pool: FlatRuleSet):
    if name == "RuleFit":
        return RuleFit(rules=pool)
    if name == "CBA":
        return CBA(rules=pool)
    if name == "CMAR":
        return CMAR(rules=pool)
    if name == "IDS":
        return IDS(rules=pool)
    if name == "IDS-greedy":
        return IDS(rules=pool, optimizer="greedy")
    if name == "Max":
        return NoDistill(rules=pool, combiner="max")
    if name == "Vote":
        return NoDistill(rules=pool, combiner="vote")
    raise ValueError(f"unknown distiller {name!r}")


def _fit_one(learner, train_rep: BooleanDataRepresentation):
    """The one module-level function handed to `TimeoutRunner` for a
    distiller fit -- same pattern as `pyrulearn.experiments.runner`'s
    own `_fit_one`."""
    return learner.fit(train_rep)


def _build_pool_call(name: str, train_rep: BooleanDataRepresentation):
    """Same, for a pool build."""
    return build_pool(name, train_rep)


# ================================================================ caching ===

def _cache_paths(cache_dir: str, key: str) -> Tuple[str, str]:
    h = hashlib.sha256(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"{h}.json"), os.path.join(cache_dir, f"{h}.pkl")


def _cache_load_row(cache_dir: Optional[str], key: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    json_path, _ = _cache_paths(cache_dir, key)
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _cache_save_row(cache_dir: Optional[str], key: str, row: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    json_path, _ = _cache_paths(cache_dir, key)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f)


def _cache_load_pool(cache_dir: Optional[str], key: str) -> Optional[FlatRuleSet]:
    if cache_dir is None:
        return None
    _, pkl_path = _cache_paths(cache_dir, key)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


def _cache_save_pool(cache_dir: Optional[str], key: str, pool: FlatRuleSet) -> None:
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    _, pkl_path = _cache_paths(cache_dir, key)
    with open(pkl_path, "wb") as f:
        pickle.dump(pool, f)


def _stratified_folds(y: np.ndarray, n_folds: int, random_state: int):
    from sklearn.model_selection import StratifiedKFold

    return list(StratifiedKFold(n_splits=n_folds, shuffle=True,
                                random_state=random_state).split(np.zeros(len(y)), y))


# =============================================================== the loop ===

def run_study(
    datasets: Sequence[CatalogEntry], pool_names: Sequence[str], distiller_names: Sequence[str],
    n_folds: int, cache_dir: Optional[str] = CACHE_DIR, fit_timeout: float = FIT_TIMEOUT,
    pool_fit_timeout: float = POOL_FIT_TIMEOUT, verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs one study: for every dataset, every fold, every pool in
    `pool_names` is built once (cached as both its measured stats and
    the pool object itself, since a later run still needs the object,
    not just its numbers) and then every distiller in `distiller_names`
    is fit against that same pool.

    Pool-building and distiller-fitting are capped separately
    (`pool_fit_timeout` vs. `fit_timeout`) since the two costs don't
    scale alike -- see `POOL_FIT_TIMEOUT`. Both share one
    `TimeoutRunner` (one persistent worker process), with
    `pool_fit_timeout` passed as a per-call override for the
    pool-building calls only: two concurrent `spawn`-context worker
    processes were observed to corrupt each other's Windows pipe
    handles, turning an unrelated, normally sub-second fit into a
    300s timeout.

    Returns `(pool_stats, results)`: `pool_stats` has one row per
    (dataset, fold, pool) with its build time and rule count;
    `results` has one row per (dataset, fold, pool, learner) with
    accuracy/n_rules/fit_time/error.
    """
    pool_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    timeout_runner = TimeoutRunner(fit_timeout) if fit_timeout is not None else None
    try:
        for entry in datasets:
            if verbose:
                print(f"{entry.name} ...", flush=True)
            df, target = entry.load()
            y_all = df[target].to_numpy()
            folds = _stratified_folds(y_all, n_folds, RANDOM_STATE)

            for fold, (train_idx, test_idx) in enumerate(folds):
                train_df = df.iloc[train_idx].reset_index(drop=True)
                test_df = df.iloc[test_idx].reset_index(drop=True)
                spec = build_dataspec(train_df, target=target, max_intervals=MAX_INTERVALS,
                                      skip_unusable=True).build()
                train_rep = BooleanDataRepresentation(spec, binarize(spec, train_df),
                                                       train_df[target].to_numpy())
                test_rep = BooleanDataRepresentation(spec, binarize(spec, test_df),
                                                      test_df[target].to_numpy())

                for pool_name in pool_names:
                    pool_key = f"{entry.name}|folds={n_folds}|fold={fold}|rs={RANDOM_STATE}|pool={pool_name}"
                    cached_stats = _cache_load_row(cache_dir, pool_key)
                    pool = _cache_load_pool(cache_dir, pool_key) if cached_stats is not None else None
                    if verbose:
                        print(f"  fold {fold + 1}/{n_folds}  pool={pool_name:<6s} ...", end="", flush=True)
                    if pool is not None:
                        pool_rows.append({**cached_stats})
                        if verbose:
                            print(" (cached)", flush=True)
                    else:
                        t0 = time.time()
                        if timeout_runner is not None:
                            pool, error = timeout_runner.run(_build_pool_call, pool_name, train_rep,
                                                             timeout=pool_fit_timeout)
                        else:
                            try:
                                pool, error = build_pool(pool_name, train_rep), None
                            except Exception as e:  # noqa: BLE001
                                pool, error = None, f"{type(e).__name__}: {e}"
                        build_time = time.time() - t0
                        n_rules = len(pool.rules) if pool is not None else np.nan
                        stats_row = {"dataset": entry.name, "fold": fold, "pool": pool_name,
                                    "build_time": build_time, "n_rules": n_rules, "error": error}
                        pool_rows.append(stats_row)
                        if error is None:
                            _cache_save_row(cache_dir, pool_key, stats_row)
                            _cache_save_pool(cache_dir, pool_key, pool)
                        if verbose:
                            print(f" {build_time:6.2f}s  {n_rules}" if error is None
                                 else f" {error}", flush=True)
                        if error is not None:
                            continue  # no pool -- every distiller for it is a failure too

                    for learner_name in distiller_names:
                        key = (f"{entry.name}|folds={n_folds}|fold={fold}|rs={RANDOM_STATE}"
                              f"|pool={pool_name}|learner={learner_name}")
                        cached = _cache_load_row(cache_dir, key)
                        if verbose:
                            print(f"    {learner_name:<8s} ...", end="", flush=True)
                        if cached is not None:
                            result_rows.append(cached)
                            if verbose:
                                print(" (cached)", flush=True)
                            continue
                        learner = build_distiller(learner_name, pool)
                        t0 = time.time()
                        if timeout_runner is not None:
                            model, error = timeout_runner.run(_fit_one, learner, train_rep)
                        else:
                            try:
                                model, error = _fit_one(learner, train_rep), None
                            except Exception as e:  # noqa: BLE001
                                model, error = None, f"{type(e).__name__}: {e}"
                        fit_time = time.time() - t0
                        row = {"dataset": entry.name, "fold": fold, "pool": pool_name,
                              "learner": learner_name, "fit_time": fit_time, "error": error,
                              "accuracy": np.nan, "n_rules": np.nan}
                        if error is None:
                            stats = model.evaluate(test_rep)
                            row["accuracy"] = stats.confusion.accuracy if stats.confusion is not None else np.nan
                            row["n_rules"] = stats.n_rules
                        result_rows.append(row)
                        _cache_save_row(cache_dir, key, row)
                        if verbose:
                            print(f" {fit_time:6.2f}s  acc={row['accuracy']:.3f}" if error is None
                                 else f" {error}", flush=True)
    finally:
        if timeout_runner is not None:
            timeout_runner.close()
    return pd.DataFrame(pool_rows), pd.DataFrame(result_rows)


# ================================================================= report ===

def _pool_stats_table(pool_stats: pd.DataFrame, results: pd.DataFrame) -> List[str]:
    # build time/n_rules are averaged over *successful* builds only -- a
    # timed-out build's ~pool_fit_timeout elapsed seconds would otherwise
    # silently dominate the mean (see "pool-build failures" instead).
    ok = pool_stats[pool_stats["error"].isna()]
    build = ok.groupby("pool")[["build_time", "n_rules"]].mean(numeric_only=True)
    failures = pool_stats.groupby("pool")["error"].apply(lambda s: int(s.notna().sum()))
    pool_names = list(pool_stats["pool"].drop_duplicates())
    mean_acc = results.groupby("pool")["accuracy"].mean()  # each pool over its own successful datasets

    # if one pool failed to build on a dataset another succeeded on
    # (e.g. LORD timing out on bank-marketing/adult), "mean accuracy"
    # above is comparing different dataset samples per pool -- a bigger
    # distortion here than in the rank table, since accuracy (unlike
    # rank) isn't normalized per dataset, so a pool that skipped the
    # hardest datasets looks better purely for having skipped them. A
    # second column restricted to the common subset corrects for that.
    common, note = _dataset_coverage_note(results, pool_names)
    common_acc = (results[results["dataset"].isin(common)].groupby("pool")["accuracy"].mean()
                 if common is not None else None)

    acc_label = "mean accuracy (own datasets)" if common is not None else "mean accuracy (all distillers)"
    cols = ["pool", "build time (s)", "n_rules", acc_label]
    if common_acc is not None:
        cols.append(f"mean accuracy ({len(common)} common datasets)")
    cols.append("pool-build failures")
    lines = ["| " + " | ".join(cols) + " |\n", "|---|" + "--:|" * (len(cols) - 1) + "\n"]
    for pool_name in pool_names:
        build_time = f"{build.loc[pool_name, 'build_time']:.2f}" if pool_name in build.index else "--"
        n_rules = f"{build.loc[pool_name, 'n_rules']:.1f}" if pool_name in build.index else "--"
        cells = [pool_name, build_time, n_rules, f"{mean_acc.get(pool_name, float('nan')):.3f}"]
        if common_acc is not None:
            cells.append(f"{common_acc.get(pool_name, float('nan')):.3f}")
        cells.append(str(failures.get(pool_name, 0)))
        lines.append("| " + " | ".join(cells) + " |\n")
    if note is not None:
        lines.append(f"\n*{note} (see \"pool-build failures\"). \"own datasets\" is each pool "
                     f"averaged over whatever it actually succeeded on; \"{len(common)} common "
                     f"datasets\" restricts every pool to the same datasets, for a fair "
                     f"comparison.*\n\n")
    return lines


def _grid_table(results: pd.DataFrame, pool_names: Sequence[str],
                distiller_names: Sequence[str]) -> List[str]:
    summary = results.groupby(["pool", "learner"])[["accuracy", "n_rules", "fit_time"]].mean(
        numeric_only=True)
    failures = results.groupby(["pool", "learner"])["error"].apply(lambda e: e.notna().sum())
    lines = ["| pool | distiller | accuracy | n_rules | fit_time (s) | failures |\n"
             "|---|---|--:|--:|--:|--:|\n"]
    for pool_name in pool_names:
        for learner_name in distiller_names:
            if (pool_name, learner_name) not in summary.index:
                continue
            row = summary.loc[(pool_name, learner_name)]
            lines.append(f"| {pool_name} | {learner_name} | {row['accuracy']:.3f} | "
                        f"{row['n_rules']:.1f} | {row['fit_time']:.2f} | "
                        f"{int(failures.get((pool_name, learner_name), 0))} |\n")
    return lines


def _dataset_coverage_note(
    results: pd.DataFrame, pool_names: Sequence[str],
) -> Tuple[Optional[set], Optional[str]]:
    """If every pool in `pool_names` succeeded (built at all) on the same
    set of datasets, returns `(None, None)` -- nothing to flag. Otherwise
    returns `(common, note)`: the subset of datasets every pool
    succeeded on, and a note naming which pool is missing which dataset
    -- e.g. LORD timing out on `bank-marketing`/`adult` in Study 2 means
    any mean computed per pool over "whatever datasets that pool has"
    isn't comparing like with like, so callers show that number *and* a
    second one restricted to `common`."""
    datasets_by_pool = {p: set(results.loc[results["pool"] == p, "dataset"].unique())
                        for p in pool_names if not results[results["pool"] == p].empty}
    if len(set(map(frozenset, datasets_by_pool.values()))) <= 1:
        return None, None
    common = set.intersection(*datasets_by_pool.values())
    all_datasets = set.union(*datasets_by_pool.values())
    note = "; ".join(
        f"{p} has no successful pool on {sorted(all_datasets - datasets)}"
        for p, datasets in datasets_by_pool.items() if all_datasets - datasets
    )
    return common, note


def _rank_one(results: pd.DataFrame, pool_names: Sequence[str]) -> Optional[pd.DataFrame]:
    per_pool = {}
    for pool_name in pool_names:
        subset = results[results["pool"] == pool_name]
        if subset.empty:
            continue
        per_pool[pool_name] = mean_rank(subset, "accuracy")
    if not per_pool:
        return None
    table = pd.DataFrame(per_pool)
    table["overall"] = table.mean(axis=1)
    return table.sort_values("overall")


def _rank_table_lines(table: pd.DataFrame, pool_names: Sequence[str]) -> List[str]:
    lines = [f"| distiller | {' | '.join(pool_names)} | overall |\n"
             f"|---|{'--:|' * (len(pool_names) + 1)}\n"]
    for learner_name in table.index:
        cells = " | ".join(f"{table.loc[learner_name, p]:.2f}" if p in table.columns else "--"
                           for p in pool_names)
        lines.append(f"| {learner_name} | {cells} | {table.loc[learner_name, 'overall']:.2f} |\n")
    return lines


def _rank_table(results: pd.DataFrame, pool_names: Sequence[str]) -> List[str]:
    """Mean rank of the distillers, computed separately per pool (not
    pooled together) -- is a distiller's relative standing the same
    regardless of which pool feeds it, or does the ranking depend on
    the source?

    If some pool failed to build on a dataset another pool succeeded
    on (e.g. LORD timing out on `bank-marketing`/`adult` in Study 2),
    that pool's column here is averaged over fewer datasets than the
    others -- flagged explicitly, with a second table restricted to
    the datasets common to every compared pool for a fair head-to-head.
    """
    table = _rank_one(results, pool_names)
    if table is None:
        return []
    lines = _rank_table_lines(table, pool_names)

    common, note = _dataset_coverage_note(results, pool_names)
    if common is not None:
        lines.append(f"\n*Columns above don't all cover the same datasets -- {note} "
                     f"(see \"pool-build failures\" in the pool stats table). Restricted to the "
                     f"{len(common)} datasets common to every pool here, for a fair "
                     f"head-to-head:*\n\n")
        common_table = _rank_one(results[results["dataset"].isin(common)], pool_names)
        if common_table is not None:
            lines += _rank_table_lines(common_table, pool_names)
    return lines


def _ids_section(study1_results: pd.DataFrame) -> List[str]:
    """`IDS` (`optimizer=\"sls\"`) vs. `IDS-greedy` across Study 1's three
    pools, built fresh from `study1_results` every run -- the live
    evidence for why plain `IDS` is dropped from Study 2 (see the Setup
    section)."""
    ids_rows = study1_results[study1_results["learner"].isin(["IDS", "IDS-greedy"])]
    summary = ids_rows.groupby(["pool", "learner"])[["accuracy", "fit_time"]].mean(numeric_only=True)
    lines = ["## Why IDS is slow and unstable\n\n"]
    lines.append(
        "`IDS` (`optimizer=\"sls\"`, Smooth Local Search, the paper-faithful default) and "
        "`IDS-greedy` (`optimizer=\"greedy\"`, a cheap deterministic fallback) both compress "
        "the *same* pool, so any difference here is the optimizer, not the data. On Study 1's "
        "small datasets:\n\n")
    lines.append("| pool | optimizer | mean accuracy | mean fit time (s) |\n|---|---|--:|--:|\n")
    for pool_name in STUDY1_POOLS:
        for learner_name in ("IDS", "IDS-greedy"):
            if (pool_name, learner_name) not in summary.index:
                continue
            row = summary.loc[(pool_name, learner_name)]
            lines.append(f"| {pool_name} | {learner_name} | {row['accuracy']:.3f} | "
                        f"{row['fit_time']:.2f} |\n")
    plain_ids = study1_results[study1_results["learner"] == "IDS"]
    by_cell = plain_ids.groupby(["dataset", "pool"])["fit_time"].mean().sort_values()
    if len(by_cell) >= 2:
        worst_key, worst_time = by_cell.index[-1], by_cell.iloc[-1]
        best_key, best_time = by_cell.index[0], by_cell.iloc[0]
        lines.append(
            f"\nPlain `IDS`'s fit time isn't a steady cost that happens to be high -- it varies "
            f"enormously by (dataset, pool), not just by pool: slowest measured here is "
            f"`{worst_key[0]}`/`{worst_key[1]}` at {worst_time:.1f}s mean, fastest is "
            f"`{best_key[0]}`/`{best_key[1]}` at {best_time:.2f}s -- a "
            f"{(worst_time / best_time if best_time else float('inf')):.0f}x spread between "
            f"datasets this demo treats as comparably small. This is consistent with Smooth "
            f"Local Search's randomized-restart search having no "
            f"convergence guarantee on a non-monotone objective, unlike `IDS-greedy`'s single "
            f"deterministic pass -- not a cost that scales predictably with the data, but "
            f"per-instance variance. A second, independent symptom of the same instability: on "
            f"`kr-vs-kp` (Study 2's pool generator, `SKLRandomForest`, tested there before plain "
            f"`IDS` was dropped from that study), one fold's accuracy came in at 0.523 (near "
            f"chance) against 0.840-0.897 on the other two folds of the same dataset and pool. "
            f"Both the speed and the accuracy can apparently collapse independently of data size "
            f"-- why plain `IDS` stays out of Study 2 (see Setup) while `IDS-greedy`, "
            f"deterministic, stays in.\n\n")
    return lines


def write_report(
    study1_pool_stats, study1_results, study2_pool_stats, study2_results,
    study1_datasets, study2_datasets, study1_folds, study2_folds, report_path: str, quick: bool,
) -> None:
    mode_note = (
        f"**Quick run** -- a fast sanity check, the default with no arguments. Full comparison: "
        f"`python examples/{NAME}.py --full`.\n\n" if quick else
        f"**Full run**. Quick sanity check instead: `python examples/{NAME}.py` with no arguments.\n\n"
    )
    study1_names = ", ".join(d.name for d in study1_datasets)
    study2_names = ", ".join(d.name for d in study2_datasets)
    description = f"""\
{mode_note}Three rule-pool generators (`CARMiner`, `SKLRandomForest`,
`pylord_candidates`) crossed with seven rule distillers (`RuleFit`,
`CBA`, `CMAR`, `IDS`, `IDS-greedy`, `Max` and `Vote` -- the last two
with no distillation at all, just the whole pool predicted from
directly), each pool built once per fold and shared across every
distiller. `IDS`/`IDS-greedy` are the same native
`pyrulearn.learners.ids.IDS`, differing only in `optimizer=`: `"sls"`
(Smooth Local Search, the paper-faithful default -- stochastic, no
convergence guarantee for a non-monotone objective) vs. `"greedy"` (a
cheap, deterministic fallback). `Max`/`Vote` are the same `FlatRuleSet`
wrapper, differing only in `combiner=`.

Two studies, since the pool generators don't scale alike (see the
module docstring):

- **Study 1** -- {study1_names} ({study1_folds}-fold) -- uses all three
  pool generators (`CARMiner`: `min_support={CAR_MIN_SUPPORT}`,
  `max_len={CAR_MAX_LEN}`; `SKLRandomForest`: `max_depth={RF_MAX_DEPTH}`;
  `pylord_candidates`: defaults) and all seven distillers.
- **Study 2** -- {study2_names} ({study2_folds}-fold) -- uses
  `SKLRandomForest` (flat pool size/build time regardless of scale) and
  `pylord_candidates` (one local search per training row, growing
  super-linearly -- feasible on the smaller datasets here, expected to
  run long or time out on `bank-marketing`/`adult`), and only six
  distillers: plain `IDS` (`optimizer="sls"`) is dropped here -- on
  Study 1's own small datasets it was already the slowest distiller by
  a wide margin (its cost doesn't track data size in any simple way --
  see "Why IDS is slow and unstable" below), so repeating it on bigger
  data wasn't worth the time. `IDS-greedy` stays.

Discretization: `max_intervals={MAX_INTERVALS}`. Distiller fits are
capped at {FIT_TIMEOUT:.0f}s; pool building gets a longer cap of
{POOL_FIT_TIMEOUT:.0f}s, since `pylord_candidates`' cost grows
super-linearly with row count and can need much longer than any
distiller fit on the biggest Study 2 datasets. A time-out or an error
counts as a failure, not as the end of the run.

Measures: pool build time and rule count per generator; per
(pool, distiller) mean accuracy, rule count and fit time; mean rank of
the distillers, computed separately per pool; per-dataset detail
(mean across folds) for every cell.

How failures are treated, throughout: a failed cell is dropped from
every mean it would otherwise enter (accuracy, fit time, build time,
rule count), never counted as a 0 or averaged in as a NaN, and the
number of failures is reported alongside every such mean (a "failures"
or "pool-build failures" column) rather than silently disappearing. A
pool that failed to build on some dataset removes every distiller's
row for that (dataset, pool) from the per-dataset table too -- no
distiller was ever attempted there. One consequence: if a pool fails on
a dataset the others don't, that pool's own mean rank ends up averaged
over fewer datasets than the pools it's compared against in the same
table -- flagged inline wherever it actually happens, with a second,
common-datasets-only table alongside for a fair head-to-head.
"""
    L = ["# Rule pools x distillers\n\n"]
    L.append(render_setup_section(description))

    L.append("## Study 1: small datasets, all three pool generators\n\n")
    L.append("### Pool stats (mean across datasets and folds)\n\n")
    L += _pool_stats_table(study1_pool_stats, study1_results)
    L.append("\n### Pool x distiller grid\n\n")
    L += _grid_table(study1_results, STUDY1_POOLS, DISTILLER_NAMES)
    L.append("\n### Mean rank of the distillers, by pool\n\n")
    L += _rank_table(study1_results, STUDY1_POOLS)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(study1_results, ["accuracy", "n_rules", "fit_time"],
                                  group_by=["dataset", "pool", "learner"], include_overall=False))
    L.append("\n")

    L += _ids_section(study1_results)

    L.append("## Study 2: larger datasets, RandomForest and LORD pools\n\n")
    L.append("### Pool stats (mean across datasets and folds)\n\n")
    L += _pool_stats_table(study2_pool_stats, study2_results)
    L.append("\n### Distiller results\n\n")
    L += _grid_table(study2_results, STUDY2_POOLS, STUDY2_DISTILLER_NAMES)
    L.append("\n### Mean rank of the distillers\n\n")
    L += _rank_table(study2_results, STUDY2_POOLS)
    L.append("\n### Per-dataset results (mean across folds)\n\n")
    L.append(render_results_table(study2_results, ["accuracy", "n_rules", "fit_time"],
                                  group_by=["dataset", "pool", "learner"], include_overall=False))
    L.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(L)
    print(f"Report -> {report_path}")


def main(quick: bool = True) -> None:
    study1_names = STUDY1_QUICK_DATASETS if quick else STUDY1_DATASETS
    study1_folds = STUDY1_QUICK_FOLDS if quick else STUDY1_FOLDS
    study2_names = STUDY2_QUICK_DATASETS if quick else STUDY2_DATASETS
    study2_folds = STUDY2_QUICK_FOLDS if quick else STUDY2_FOLDS
    report_path = os.path.join(HERE, f"{NAME}_{'quick_' if quick else ''}report.md")

    study1_datasets = Catalog.default().select(names=study1_names)
    study2_datasets = Catalog.default().select(names=study2_names)

    print("=== Study 1 ===", flush=True)
    study1_pool_stats, study1_results = run_study(
        study1_datasets, STUDY1_POOLS, DISTILLER_NAMES, study1_folds)
    print("=== Study 2 ===", flush=True)
    study2_pool_stats, study2_results = run_study(
        study2_datasets, STUDY2_POOLS, STUDY2_DISTILLER_NAMES, study2_folds)

    write_report(study1_pool_stats, study1_results, study2_pool_stats, study2_results,
                study1_datasets, study2_datasets, study1_folds, study2_folds, report_path, quick)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run both studies at full scale instead of the quick default")
    args = parser.parse_args()
    main(quick=not args.full)
