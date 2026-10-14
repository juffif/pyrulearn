"""
pyrulearn.experiments.runner
=============================

`run_cv`: the shared cross-validation loop every demo builds on --
"run these learners on these datasets, k-fold, with a per-fit timeout,
uniformly measured" -- so a demo's own code is just picking learners,
picking datasets, and deciding what to report, not reimplementing
loading, splitting, binarization, timeouts or measurement again.

    from pyrulearn.data.catalog import Catalog
    from pyrulearn.experiments.runner import run_cv
    from pyrulearn.learners.seco import Pypper

    results = run_cv([Pypper()], Catalog.default().parse("binary,small"))

`results` is a long-format `pandas.DataFrame`, one row per (dataset,
fold, learner) -- see `run_cv`'s docstring for its columns. Nothing here
computes ranks, runs a Friedman test or writes a report; those are
`pyrulearn.experiments.stats` and `pyrulearn.experiments.report`,
deliberately separate so a demo pulls in only what it needs.

`TimeoutRunner` (moved here from ``examples/demo_workflow_comparison.py``,
generalized to not assume anything Weka-specific) runs `fn` in a
persistent worker subprocess and enforces `timeout` seconds per call --
`fn` must be a plain module-level function, picklable by reference
(no closures, no lambdas, no bound methods of a local object): the
worker process receives it across the `multiprocessing` boundary, and a
learner's own `fit` isn't picklable that way, hence `_fit_one` below,
the one module-level function `run_cv` actually hands `TimeoutRunner`.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from ..data import DataRepresentation, default_representation
from ..data.io import build_dataspec, encode
from ..learners.base import RuleLearner
from .catalog import CatalogEntry

#: default per-fit time budget, seconds -- see `TimeoutRunner`
DEFAULT_FIT_TIMEOUT = 60.0
#: default discretization cap, forwarded to `build_dataspec`
DEFAULT_MAX_INTERVALS = 8

#: the fixed measures every row carries -- see `run_cv`'s docstring.
BASE_COLUMNS = ["dataset", "fold", "learner", "fit_time", "error", "accuracy", "n_rules", "n_conditions"]


# ============================================================= TimeoutRunner


def _kill_tree(pid: Optional[int]) -> None:
    """Kill `pid` and every descendant. `multiprocessing.Process.terminate()`
    only signals the one process; a fitter that spawns its own pool
    (joblib, an external learner's own threading) would otherwise leave
    orphans thrashing the CPU for the rest of the run. Best-effort --
    never raises."""
    if pid is None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, timeout=15)
        else:
            try:
                import psutil  # optional; POSIX has no built-in tree-kill
                proc = psutil.Process(pid)
                for child in proc.children(recursive=True):
                    child.kill()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def _worker_loop(task_q, result_q) -> None:
    """Runs in a long-lived subprocess started by `TimeoutRunner`: pulls
    `(fn, args, kwargs)` off `task_q`, calls `fn(*args, **kwargs)`, and
    pushes `("ok", result)` or `("error", message)` onto `result_q` --
    forever, until it receives `None` as a stop sentinel."""
    while True:
        item = task_q.get()
        if item is None:
            return
        fn, args, kwargs = item
        try:
            result_q.put(("ok", fn(*args, **kwargs)))
        except Exception as e:  # noqa: BLE001 -- report *any* failure back, don't crash the worker
            result_q.put(("error", f"{type(e).__name__}: {e}"))


class TimeoutRunner:
    """Runs picklable `(fn, *args, **kwargs)` calls in a persistent
    worker subprocess, enforcing `timeout` seconds per call -- so one
    learner hanging (or just running unexpectedly long) can't block the
    rest of a fold, let alone the rest of the run.

    Deliberately *not* a fresh subprocess per call (that would add real
    spawn + full-module-reimport overhead, sklearn included, to *every*
    fit): the same worker process is reused across calls, and only
    killed and replaced with a fresh one when a call actually times out.
    A plain (non-timeout) exception from `fn` doesn't trigger a restart
    -- the worker's own loop already caught it and is still healthy.

    On a timeout the *whole process tree* is killed (`_kill_tree`), not
    just the worker, since some fitters spawn their own children.
    """

    def __init__(self, timeout: float = DEFAULT_FIT_TIMEOUT):
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self.task_q = None
        self.result_q = None
        self.proc = None
        self._start()

    def _start(self) -> None:
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker_loop, args=(self.task_q, self.result_q), daemon=True)
        self.proc.start()

    def _kill_worker(self) -> None:
        pid = self.proc.pid
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        _kill_tree(pid)

    def run(self, fn: Callable, *args, timeout: Optional[float] = None, **kwargs):
        """Returns `(result, None)` on success, or `(None, error_message)`
        -- for a timeout, `error_message` is exactly ``"timeout"``;
        otherwise it's `fn`'s own exception, stringified. Never raises.

        `timeout`, if given, overrides `self.timeout` for this one call
        only -- e.g. a caller that needs a longer budget for one kind of
        call (building a pool) than another (fitting a distiller) can
        share a single worker process rather than running two at once
        (two concurrent `spawn`-context `Process`/`Queue` pairs have been
        observed to corrupt each other's Windows pipe handles)."""
        self.task_q.put((fn, args, kwargs))
        try:
            status, payload = self.result_q.get(timeout=timeout if timeout is not None else self.timeout)
        except queue.Empty:
            self._kill_worker()
            self._start()
            return None, "timeout"
        if status == "ok":
            return payload, None
        return None, payload

    def close(self) -> None:
        self._kill_worker()

    def __enter__(self) -> "TimeoutRunner":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _fit_one(learner: RuleLearner, train_rep: DataRepresentation):
    """The one module-level function `run_cv` hands `TimeoutRunner`
    (must be picklable by reference -- see the module docstring)."""
    return learner.fit(train_rep)


def _timed_fit(learner: RuleLearner, train_rep: DataRepresentation):
    """`_fit_one` plus the seconds `learner.fit` took, measured inside the
    worker: with `n_jobs > 1` a cell can wait for a free worker, so the
    parent can't time a fit from the outside."""
    t0 = time.time()
    model = learner.fit(train_rep)
    return model, time.time() - t0


# =================================================================== run_cv


def _stratified_folds(y: np.ndarray, n_folds: int, random_state: int):
    from sklearn.model_selection import KFold, StratifiedKFold

    try:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(y)), y))
    except ValueError:
        # a class too small to stratify into n_folds -- fall back to plain KFold
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        return list(splitter.split(np.zeros(len(y))))


def _stable_params(value, depth: int = 0):
    """`value` as JSON-ready data that is the same in every run: nested
    component objects (e.g. a SeCo learner's `SingleRuleLearner`) by class
    name and public attributes -- their default `str` includes a memory
    address, which would change the cache key from run to run."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _stable_params(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_stable_params(v, depth + 1) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if hasattr(value, "__dict__") and depth < 8:
        attrs = {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return {"__class__": f"{type(value).__module__}.{type(value).__qualname__}",
                **{k: _stable_params(v, depth + 1) for k, v in attrs.items()}}
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _cache_key(dataset: CatalogEntry, n_folds: int, fold: int, random_state: int,
              learner: RuleLearner, representation: Type[DataRepresentation]) -> str:
    """A stable hash identifying one (dataset, fold split, representation,
    learner configuration) cell -- see `run_cv`'s "Fold caching"
    paragraph."""
    params_repr = json.dumps(_stable_params(learner._provenance_params()), sort_keys=True)
    payload = "|".join([
        dataset.name, str(dataset.openml_id), str(dataset.openml_version),
        str(n_folds), str(fold), str(random_state), representation.__name__,
        learner.display_name, params_repr,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"{key}.json")


def _cache_load(cache_dir: Optional[str], key: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cache_save(cache_dir: Optional[str], key: str, row: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, key), "w", encoding="utf-8") as f:
        json.dump(row, f)


def _measure(learner: RuleLearner, train_rep: DataRepresentation, test_rep: DataRepresentation,
            timeout_runner: Optional[TimeoutRunner],
            measure_fn: Optional[Callable[[Any, DataRepresentation], dict]]) -> Dict[str, Any]:
    t0 = time.time()
    if timeout_runner is not None:
        model, error = timeout_runner.run(_fit_one, learner, train_rep)
    else:
        try:
            model, error = _fit_one(learner, train_rep), None
        except Exception as e:  # noqa: BLE001 -- a failed fit is recorded, not fatal
            model, error = None, f"{type(e).__name__}: {e}"
    return _score(model, error, time.time() - t0, test_rep, measure_fn)


def _score(model, error: Optional[str], fit_time: float, test_rep: DataRepresentation,
           measure_fn: Optional[Callable[[Any, DataRepresentation], dict]]) -> Dict[str, Any]:
    """The measured columns of one cell, from its (possibly failed) fit."""
    row: Dict[str, Any] = {"fit_time": fit_time, "error": error,
                           "accuracy": np.nan, "n_rules": np.nan, "n_conditions": np.nan}
    if error is not None:
        return row
    stats = model.evaluate(test_rep)
    row["accuracy"] = stats.confusion.accuracy if stats.confusion is not None else np.nan
    row["n_rules"] = stats.n_rules
    row["n_conditions"] = stats.n_conditions
    if measure_fn is not None:
        row.update(measure_fn(model, test_rep))
    return row


@dataclass
class _Cell:
    """One (dataset, fold, learner) cell of `run_cv`, ready to fit."""
    index: int                      # position in the sequential order -- the row order of the result
    dataset: str
    fold: int
    n_folds: int
    learner: RuleLearner
    key: str                        # cache key
    cached: Optional[Dict[str, Any]]
    train_rep: DataRepresentation
    test_rep: DataRepresentation


def _iter_cells(learners, datasets, n_folds, max_intervals, random_state, cache_dir,
                representation, verbose):
    """Yield every cell in (dataset, fold, learner) order, loading and
    encoding each dataset and fold only when the consumer asks for its
    first cell -- so a parallel run holds just the folds in flight."""
    index = 0
    for entry in datasets:
        if verbose:
            print(f"{entry.name} ...", flush=True)
        df, target = entry.load()
        y_all = df[target].to_numpy()
        folds = _stratified_folds(y_all, n_folds, random_state)

        for fold, (train_idx, test_idx) in enumerate(folds):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            spec = build_dataspec(train_df, target=target, max_intervals=max_intervals,
                                  skip_unusable=True).build()
            train_rep = encode(spec, train_df, train_df[target].to_numpy(), representation=representation)
            test_rep = encode(spec, test_df, test_df[target].to_numpy(), representation=representation)

            for learner in learners:
                key = _cache_key(entry, n_folds, fold, random_state, learner, representation)
                yield _Cell(index, entry.name, fold, n_folds, learner, key,
                            _cache_load(cache_dir, key), train_rep, test_rep)
                index += 1


def _outcome(row: Dict[str, Any]) -> str:
    return row["error"] if row["error"] is not None else f"acc={row['accuracy']:.3f}"


def _run_cells_parallel(cells, n_jobs: int, fit_timeout: Optional[float],
                        measure_fn, cache_dir: Optional[str], verbose: bool) -> List[Dict[str, Any]]:
    """The cells of `_iter_cells`, `n_jobs` fits at a time, each in its own
    `TimeoutRunner` worker process (a thread per worker waits on it). The
    parent only encodes folds, evaluates finished fits and writes the cache.
    At most `2 * n_jobs` cells are in flight, which bounds the folds held
    in memory. Rows come back in the cells' own order."""
    import threading

    # started one after the other: concurrent spawns have corrupted each other's pipes on Windows
    runners = [TimeoutRunner(fit_timeout) for _ in range(n_jobs)]
    todo: "queue.Queue[Optional[_Cell]]" = queue.Queue()
    done: "queue.Queue[tuple]" = queue.Queue()

    def worker(runner: TimeoutRunner) -> None:
        while True:
            cell = todo.get()
            if cell is None:
                return
            t0 = time.time()
            try:
                out, error = runner.run(_timed_fit, cell.learner, cell.train_rep)
            except Exception as e:  # noqa: BLE001 -- keep the cell, don't lose the worker
                out, error = None, f"{type(e).__name__}: {e}"
            if error is None:
                model, fit_time = out
            else:
                model, fit_time = None, time.time() - t0
            done.put((cell, model, error, fit_time))

    rows: Dict[int, Dict[str, Any]] = {}

    def finish() -> None:
        cell, model, error, fit_time = done.get()
        row = {"dataset": cell.dataset, "fold": cell.fold, "learner": cell.learner.display_name}
        row.update(_score(model, error, fit_time, cell.test_rep, measure_fn))
        _cache_save(cache_dir, cell.key, row)
        rows[cell.index] = row
        if verbose:
            print(f"  {cell.dataset} fold {cell.fold + 1}/{cell.n_folds}  "
                  f"{cell.learner.display_name:<15s} {row['fit_time']:6.2f}s  {_outcome(row)}", flush=True)

    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in runners]
    try:
        for t in threads:
            t.start()
        in_flight = 0
        for cell in cells:
            if cell.cached is not None:
                rows[cell.index] = cell.cached
                if verbose:
                    print(f"  {cell.dataset} fold {cell.fold + 1}/{cell.n_folds}  "
                          f"{cell.learner.display_name:<15s} (cached)", flush=True)
                continue
            while in_flight >= 2 * n_jobs:
                finish()
                in_flight -= 1
            todo.put(cell)
            in_flight += 1
        while in_flight:
            finish()
            in_flight -= 1
    finally:
        for _ in threads:
            todo.put(None)
        for r in runners:
            r.close()
    return [rows[i] for i in sorted(rows)]


def run_cv(
    learners: Sequence[RuleLearner],
    datasets: Sequence[CatalogEntry],
    n_folds: int = 10,
    fit_timeout: Optional[float] = DEFAULT_FIT_TIMEOUT,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
    random_state: int = 0,
    cache_dir: Optional[str] = None,
    measure_fn: Optional[Callable[[Any, DataRepresentation], dict]] = None,
    verbose: bool = True,
    representation: Optional[Type[DataRepresentation]] = None,
    n_jobs: int = 1,
):
    """Run every learner in `learners` against every dataset in
    `datasets`, `n_folds`-fold cross-validation, and return one
    long-format `pandas.DataFrame`, one row per (dataset, fold, learner):

    - ``dataset`` -- `entry.name`.
    - ``fold`` -- 0-based fold index.
    - ``learner`` -- `learner.display_name`.
    - ``fit_time`` -- wall-clock seconds for this fit (whether it
      succeeded, failed, or timed out).
    - ``error`` -- `None` on success, else ``"timeout"`` or the
      exception stringified -- a failed fit is recorded, not fatal,
      matching every existing demo's convention. Every other column is
      NaN when `error` is set.
    - ``accuracy``, ``n_rules``, ``n_conditions`` -- from
      `model.evaluate(test_rep)` (`pyrulearn.models.RuleModel.evaluate`),
      the uniform measurement for any fitted model.
    - whatever extra columns `measure_fn(model, test_rep)` returns, if
      given -- for demo-specific measures (e.g. coverage overlap, a
      rule set's total weight). Keeps the *default* measures fixed and
      cheap while letting a demo ask for more.

    Per dataset: `entry.load()`, then `StratifiedKFold(n_splits=n_folds,
    shuffle=True, random_state=random_state)`, falling back to plain
    `KFold` for a class too small to stratify. Per fold, `build_dataspec`
    runs **on the training split only** (`skip_unusable=True`, since a
    column can turn constant or entirely missing within one fold), and
    both splits are then `encode`d into `representation` (default
    `pyrulearn.data.representation.default_representation()`, the first
    entry of `REPRESENTATION_PREFERENCE_ORDER` -- `NListRepresentation`,
    the fastest for every native learner measured, see
    `demos/representations_report.md`) against that same fold's `DataSpec` --
    a learner that needs another representation (an external tool's
    dense matrix, say) converts on its own, see
    `RuleLearner.ensure_representation` --
    this avoids leaking test-set discretization thresholds into training.

    Per (fold, learner): `learner.fit(train_rep)` through `TimeoutRunner`
    (`fit_timeout` seconds; `None` disables the subprocess/timeout
    machinery entirely, useful for debugging -- a plain exception is
    still caught and recorded either way), then `model.evaluate(test_rep)`
    plus wall-clock `fit_time`. `fn` handed to `TimeoutRunner` must stay
    picklable by reference -- a learner instance and a `DataRepresentation`
    both already are (plain attributes, no closures), so this holds for
    any `RuleLearner` in this package.

    Fold caching (`cache_dir`, off by default): one JSON file per
    `cache_dir`, keyed by a stable hash of (dataset name + OpenML id +
    version, `n_folds`, fold index, `random_state`, `representation`'s
    class name, `learner.display_name`, a stable repr of the learner's
    constructor params -- `learner._provenance_params()`, the same
    values `Provenance` records). A row already cached is loaded instead
    of refit, so adding one learner, or resuming after a crash, doesn't
    refit everything. This is a simple v1 -- refine later if it proves
    too coarse (e.g. it doesn't detect that `datasets`/`max_intervals`
    changed between runs, or that a learner's own *code* changed while
    its constructor params didn't -- clear `cache_dir` by hand if that
    matters; `representation` is keyed precisely because that trap was
    hit in practice).

    `verbose` (default True) prints progress as it runs: one line per
    dataset as it starts, then one line per (fold, learner) -- name and
    fold index first (so a slow fit's identity is visible immediately,
    not only once it finishes), the outcome (fit time, plus accuracy or
    the error) appended once that cell is done, or ``(cached)`` instead
    of doing any work at all for a row `cache_dir` already has. Each
    line is flushed immediately, so progress is visible live even when
    stdout is redirected to a file or piped.

    `n_jobs` (default 1) fits that many cells at the same time, each in
    its own `TimeoutRunner` worker process; the cells are independent, so
    the table has the same rows, in the same order, and the same accuracies
    and rule counts as with `n_jobs=1` -- for learners whose result is fixed
    by their `random_state`. Two things change: (1) a fit's `fit_time` is
    then measured inside its worker, around `learner.fit` alone (with
    `n_jobs=1` it includes handing the learner and the data to the worker
    and the model back); (2) progress lines come out as cells finish, each
    naming its dataset, not as they start. Concurrent fits compete for
    memory bandwidth and caches, so **do not compare `fit_time`s from runs
    with `n_jobs > 1` with each other, or with a sequential run** --
    parallelize only runs that report accuracy, sizes or counts. The
    parent keeps at most `2 * n_jobs` cells' folds in memory, and every
    worker holds its own copy of its fold; a learner passed with `n_jobs > 1`
    must be picklable by reference like any other (no class defined inside
    a function).
    """
    import pandas as pd

    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"n_jobs must be a positive integer, got {n_jobs!r}")

    resolved_representation = representation or default_representation()
    cells = _iter_cells(learners, datasets, n_folds, max_intervals, random_state, cache_dir,
                        resolved_representation, verbose)
    if n_jobs > 1:
        rows = _run_cells_parallel(cells, n_jobs, fit_timeout, measure_fn, cache_dir, verbose)
        return pd.DataFrame(rows)

    rows: List[Dict[str, Any]] = []
    timeout_runner = TimeoutRunner(fit_timeout) if fit_timeout is not None else None
    try:
        for cell in cells:
            if verbose:
                print(f"  fold {cell.fold + 1}/{n_folds}  {cell.learner.display_name:<15s} ...",
                      end="", flush=True)
            if cell.cached is not None:
                rows.append(cell.cached)
                if verbose:
                    print(" (cached)", flush=True)
                continue
            row = {"dataset": cell.dataset, "fold": cell.fold, "learner": cell.learner.display_name}
            row.update(_measure(cell.learner, cell.train_rep, cell.test_rep, timeout_runner, measure_fn))
            _cache_save(cache_dir, cell.key, row)
            rows.append(row)
            if verbose:
                print(f" {row['fit_time']:6.2f}s  {_outcome(row)}", flush=True)
    finally:
        if timeout_runner is not None:
            timeout_runner.close()

    return pd.DataFrame(rows)
