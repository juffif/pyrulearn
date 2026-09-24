"""
"Binarized" vs. "Original" data preparation, comparison across twelve
binary UCI/OpenML benchmarks, covering every rule learner imported into
pyrulearn so far:
sklearn's `DecisionTree`/`RandomForest`, wittgenstein's `RIPPERk`/`IREP`,
`imodels`' `BayesianRuleList`/`BayesianRuleSet`, and Weka's
`JRip`/`PART`/`J48` (via subprocess -- Weka isn't a Python library).

Datasets: `vote`, `breast-cancer`, `colic`, `credit-approval`, `credit-g`
(the original five), plus `diabetes`, `sonar`, `ionosphere`,
`tic-tac-toe`, `banknote-authentication` (a few hundred to ~1400 rows),
and two larger ones, `kr-vs-kp` (~3196 rows) and `mushroom` (~8124 rows)
-- included deliberately despite RIPPER's known poor scaling with
dataset size, to see where that starts to bite in practice; watch their
fit-time numbers specifically and drop them if they turn out to dominate
total runtime the way `adult` did. `adult` itself (~49k rows) stays
excluded outright: RIPPER alone measured ~155s for one fit on its ~39k-
row training fold, and this demo fits it (and now everything else) up to
several times per fold -- multi-hour territory, not "starting to bite."
Genuinely multi-class UCI sets (`soybean`, `anneal`, `cmc`,
`hypothyroid`) stay out of scope too -- `RIPPERk`/`IREP` are
fundamentally binary and refuse >2 classes outright (see
`pyrulearn.interfaces.wittgenstein`); real multi-class support is a
separate future project.

Per fold of a 5-fold cross-validation (train/test split freshly built
from *that fold's training data only* either way), `WORKFLOW_NAMES`
names the two ways training data gets prepared before fitting -- what
used to be called "workflow 1"/"workflow 2" here, renamed since those
names alone don't say which is which:

- **"Binarized"** (`pyrulearn.data.io.build_dataspec`/`binarize`
  first, i.e. the old "workflow 1"): one `DataSpec` from the training
  fold's raw data, both splits binarized against it, every model fit
  directly on the training fold's Boolean data.
- **"Original"** (fit on raw, native, not-yet-binarized data, i.e. the
  old "workflow 2"): every model fit on the training fold's *native*
  input, each `ObjectRuleImporter.infer_dataspec` discovers its own
  per-model `DataSpec` afterward, and (for the models that share one
  common raw feature space -- see below) those get merged into one
  shared `DataSpec` (`pyrulearn.data.merge_dataspecs`) so the test
  fold is binarized and predicted against just once.

For RIPPER/IREP specifically, whichever of a dataset's two classes gets
treated as "positive" isn't a neutral choice: the algorithm only learns
rules *for* `pos_class`, using the other as a passive default. Both
directions are fit and reported separately -- `ripper_A`/`irep_A` treat
the (alphabetically) first class as positive, `ripper_B`/`irep_B` the
second.

Three groups of models, by how "Binarized"/"Original" actually differ
for them:

1. **`tree`/`forest`/`ripper_*`/`irep_*`** -- genuinely two different
   fits, one per data-preparation mode, exactly as in the original
   version of this demo (see wrinkles 1/2 below for the
   sklearn-needs-one-hot and DataSpec-merge-can-fail details). `forest`
   uses `MacroVoteCombiner` for prediction -- the one that actually
   matches `RandomForestClassifier.predict()`'s own soft-voting
   mechanism (see `pyrulearn.interfaces.sklearn`).
2. **`brl`/`brs`** (`imodels.BayesianRuleListClassifier`/
   `BayesianRuleSetClassifier`) -- both *require* already-Boolean input
   (see `pyrulearn.interfaces.imodels`'s module docstring); there
   is no "fit on raw data" path at all, so "Binarized" and "Original"
   fit on the *same* already-Boolean matrix either way -- but as two
   genuinely separate calls, not one fit reused under both labels.
   "Original" rebuilds that matrix as a fresh `BooleanDataRepresentation`
   via a plain DataFrame (`train_rep1_copy`) rather than reusing
   "Binarized"'s in-memory object, so the two calls are independent (CSV
   export/import round-trip fidelity is a separate concern, already
   covered by `tests/test_data_io.py`, not re-exercised here). Both get
   a *negation-free* DataSpec (`build_dataspec(..., include_negations=
   False)`): neither model benefits from explicit negation features (BRL
   searches only "present item" sets; BRS builds its own
   `{name, name_neg}` space internally), and the ~2x extra columns from
   the default negation features push BRS's per-rule-length RandomForest
   discretization past `FIT_TIMEOUT_SECONDS` on the larger datasets. Since
   both fits are otherwise deterministic given the same `random_state`,
   matching numbers across the two columns is expected, not a
   coincidence. `BayesianRuleSetClassifier`'s own "clean"-move bug
   (documented in `imodels_rules`) is retried across a few
   `random_state`s the same way `tests/test_imodels_import.py` does,
   independently for each of the two calls.
3. **`jrip`/`part`/`j48`** -- Weka classifiers, invoked as a subprocess
   against a written `.arff` file (`run_weka`), then parsed from Weka's
   own printed text via `pyrulearn.interfaces.weka`. "Binarized"
   here can't literally reuse the same `DataSpec` object the Python-
   native models share: Weka's importers always *re-parse* condition
   text, and `ds1`'s own feature names (e.g. `"age>=30"`) would corrupt
   that re-parsing if fed straight to Weka's ARFF writer (the embedded
   `>=` collides with the parser's own operator regex). So the
   "Binarized" `.arff` for these three uses safe placeholder names
   (`f0`, `f1`, ...) for `ds1`'s already-Boolean columns, and each Weka
   importer's own `infer_dataspec` builds its own self-consistent (if
   differently named) `DataSpec` from that -- same "fit on pre-binarized
   data" semantics, just not literally the same shared object, and
   evaluated against its own re-binarization of the same 0/1 data
   rather than `ds1` directly. "Original" needs no such workaround: Weka
   accepts raw nominal/numeric columns natively, no one-hot expansion
   required the way sklearn needs. Each Weka model's `DataSpec` (either
   way) is evaluated on its own -- not merged with group 1's -- so its
   accuracy numbers are exact, just not part of the DataSpec-merge
   fallback-counting this demo tracks for group 1.

Two wrinkles group 1's "Original" mode has to deal with that
"Binarized" sidesteps by binarizing everything the same way up front:

1. **Nominal columns need one-hot, never ordinal, encoding for
   sklearn.** wittgenstein accepts a raw nominal column directly;
   scikit-learn only ever accepts numbers. One-hot (`_one_hot_expand`,
   one 0/1 column per (attribute, category) pair, named
   `f"{attribute}={category}"`) avoids inventing an order ordinal
   encoding would -- exactly `DataSpecBuilder.add_nominal`'s own naming,
   so `SklearnTreeImporter`/`RandomForestImporter`'s own discovered
   NUMERIC attribute per dummy (e.g. `"color=red>=0.5"`) never collides
   with wittgenstein's grouped NOMINAL `"color"` in `merge_dataspecs`.
2. **Models can still disagree on a column's type.** Each model's
   ruleset only references the values *its own* rules happened to need,
   so two independently-fit models can reach different NUMERIC/NOMINAL
   conclusions for the same real column. When `merge_dataspecs` hits
   this, `run_fold` doesn't crash the fold -- it falls back to
   evaluating every group-1 model against its own per-model `DataSpec`
   instead, and reports how many folds per dataset needed this.

Run-times are wall-clock seconds around each model's fit call (or, for
the Weka trio, around the whole subprocess -- JVM startup included, so
these three carry a fixed per-call overhead the in-process Python
algorithms don't; not a perfectly apples-to-apples comparison, called
out again in the runtime table itself). Rule/condition counts are read
directly off each fold's fitted rules -- `RandomForest`'s in particular
will dwarf everything else, since it's `n_estimators` trees' worth of
leaves, not a single interpretable model; included for completeness,
not as an apples-to-apples interpretability comparison either.
"""

import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from pyrulearn.combiners import MacroVoteCombiner
from pyrulearn.data import merge_dataspecs
from pyrulearn.models import FlatRuleSet, annotate_rules
from pyrulearn.data.io import binarize, build_dataspec, write_arff
# BayesianRuleList/BayesianRuleSet are imported lazily, inside _fit_brl_binarized/_fit_brs
# below (not here) -- they run much longer than the rest of the models and pull in
# imodels/mlxtend, so anything that only needs this module's lightweight pieces (e.g. the
# overview-table helpers) doesn't pay that import cost.
from pyrulearn.interfaces.sklearn import (
    DecisionTree,
    RandomForest,
    RandomForestImporter,
    SklearnTreeImporter,
)
from pyrulearn.interfaces.weka import J48Importer, JRipImporter, PARTImporter
from pyrulearn.interfaces.wittgenstein import IREP, IREPImporter, RIPPERImporter, RIPPERk
from pyrulearn.data import BooleanDataRepresentation

N_FOLDS = 5
RANDOM_STATE = 0
MAX_INTERVALS = 8  # "Binarized" mode only -- see build_dataspec's max_intervals
MAX_DEPTH = 4
RIPPER_K = 2
N_ESTIMATORS = 10
# BayesianRuleListClassifier's own defaults (n_chains=3, max_iter=50000) measured
# ~3.3s for one fit on a tiny 400-row/5-feature synthetic case; n_chains=1,
# max_iter=2000 measured ~0.12s on the same case with identical accuracy --
# ~28x faster, no correctness cost observed
BRL_PARAMS = dict(n_chains=1, max_iter=2000)
# discretization_method="randomforest" is imodels' own default. Its alternative,
# "fpgrowth" -- which would align BRS with BRL's own candidate generation (see
# extract_fpgrowth) -- is NOT usable as installed: BayesianRuleSetClassifier's
# fpgrowth branch (brs.py, in _generate_rules) calls mlxtend.frequent_patterns's
# fpgrowth with keyword arguments (supp=, zmin=, zmax=) that belong to a
# different package's fpgrowth (pyfim/fim's `fpgrowth(tracts, supp=, zmin=,
# zmax=)`) -- mlxtend's own signature is `fpgrowth(df, min_support=, max_len=)`,
# entirely different parameter names AND input shape (one-hot DataFrame, not a
# transaction list). Confirmed directly: `discretization_method="fpgrowth"`
# raises `TypeError: fpgrowth() got an unexpected keyword argument 'supp'`
# immediately on fit -- an upstream imodels bug, not something fixable from here
# without patching imodels' own internals. "randomforest" -- explosive in
# feature count (fits one RandomForestClassifier per rule length with
# n_estimators = min((2*n_features)**length, 4000), and with no random_state
# of its own -- a likely source of the run-to-run timing variance seen in this
# demo) -- is the only working option for now.
BRS_PARAMS = dict(num_iterations=30, num_chains=1, n_rules=50, maxlen=3, supp=1,
                   discretization_method="randomforest")
BRS_RANDOM_STATE_CANDIDATES = range(5)  # see BRS's "clean"-move bug, module docstring

WEKA_JAVA = r"C:\Program Files\Weka-3-8-7\jre\jre-25.0.2-full\bin\java.exe"
WEKA_JAR = r"C:\Program Files\Weka-3-8-7\weka.jar"
WEKA_CLASSES = {
    "jrip": "weka.classifiers.rules.JRip",
    "part": "weka.classifiers.rules.PART",
    "j48": "weka.classifiers.trees.J48",
}
WEKA_IMPORTERS = {"jrip": JRipImporter, "part": PARTImporter, "j48": J48Importer}

STANDARD_DATASETS = [
    "vote", "breast-cancer", "colic", "credit-approval", "credit-g",
    "diabetes", "sonar", "ionosphere", "tic-tac-toe", "banknote-authentication",
    "kr-vs-kp", "mushroom",
]
REPORT_PATH = os.path.join(os.path.dirname(__file__), "demo_workflow_comparison_report.md")
ARFF_DIR = os.path.join(os.path.dirname(__file__), "_workflow_demo_arff")

DIRECTIONAL_MODELS = ["ripper", "irep"]
SINGLE_MODELS = ["tree", "forest", "brl", "brs", "jrip", "part", "j48"]
MODEL_VARIANTS = ["tree", "forest", "ripper_A", "ripper_B", "irep_A", "irep_B", "brl", "brs", "jrip", "part", "j48"]
# Single place to turn a model off entirely -- `run_fold` skips its fit calls
# (no NaN/failure recorded, just absent), and every downstream table drops its
# row. E.g. `ENABLED_MODELS = set(MODEL_VARIANTS) - {"brl", "brs"}` to exclude
# both without touching run_fold's body or any import. Defaults to everything.
ENABLED_MODELS = set(MODEL_VARIANTS)
ACTIVE_MODELS = [m for m in MODEL_VARIANTS if m in ENABLED_MODELS]  # MODEL_VARIANTS' order, filtered
WORKFLOW_NAMES = ["Binarized", "Original"]
# "Binarized" = build_dataspec/binarize first (workflow 1)
# "Original" = fit on raw/native data, discover the DataSpec afterward (workflow 2)

# Per-algorithm-per-fold time budget. `brl`/`brs` (imodels) scale badly in
# feature count -- BRL's FP-growth candidate generation, and BRS's
# discretization fitting one RandomForest per rule length with n_estimators ~
# n_features**length -- so on the numeric-heavy datasets (`sonar`, 60
# continuous attributes -> a few hundred discretized Boolean features;
# `ionosphere`, 33) a single fit runs many minutes, well past this budget, and
# both are marked n/a for those two datasets. That's expected: they're
# fundamentally slow there, not broken (an earlier run at a 300s budget got
# `brs` on `sonar` in ~245s and `brl` on `sonar` failing outright). Every
# Python-native fit below goes through TimeoutRunner so one slow fit can't
# block the run -- but note the budget is *per call*: a dataset where all 20
# brl/brs calls (2 models x 2 workflows x 5 folds) hit it still costs ~20
# minutes of correctly-capped waiting. The Weka trio uses subprocess.run's own
# timeout= instead (see run_weka) -- a subprocess is already independently
# killable.
FIT_TIMEOUT_SECONDS = 60


def _kill_tree(pid: int) -> None:
    """Kill `pid` and every descendant. `multiprocessing.Process.terminate()`
    only signals the one process; a fitter that spawned its own pool
    (wittgenstein's beam RIPPER, joblib) would otherwise leave orphans
    thrashing the CPU for the rest of the run. Best-effort -- never raises.
    """
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


def _worker_loop(task_q, result_q):
    """Runs in a long-lived subprocess started by `TimeoutRunner`: pulls
    `(fn, args, kwargs)` off `task_q`, calls `fn(*args, **kwargs)`, and
    pushes `("ok", result)` or `("error", message)` onto `result_q` --
    forever, until it receives `None` as a stop sentinel. `fn` must be a
    module-level function (picklable by reference) for `multiprocessing`
    to send it across the process boundary at all.
    """
    while True:
        item = task_q.get()
        if item is None:
            return
        fn, args, kwargs = item
        try:
            result_q.put(("ok", fn(*args, **kwargs)))
        except Exception as e:  # noqa: BLE001 -- deliberately broad: report *any* failure back, don't crash the worker
            result_q.put(("error", f"{type(e).__name__}: {e}"))


class TimeoutRunner:
    """Runs picklable `(fn, *args, **kwargs)` calls in a persistent
    worker subprocess, enforcing `timeout` seconds per call -- so one
    algorithm hanging (or just running unexpectedly long) can't block
    the rest of a fold, let alone the rest of the run.

    Deliberately *not* a fresh subprocess per call (that would add real
    spawn + full-module-reimport overhead -- sklearn/wittgenstein/imodels
    included -- to *every* one of the ~20 fits per fold, dwarfing the
    fast ones): the same worker process is reused across calls, and only
    killed and replaced with a fresh one when a call actually times out
    (`concurrent.futures.ProcessPoolExecutor` can't do this cleanly --
    its own `shutdown()` won't forcibly kill a still-running worker, only
    stop waiting for it -- hence the lower-level `multiprocessing.Process`
    here instead). A plain (non-timeout) exception from `fn` doesn't
    trigger a restart -- the worker's own loop already caught it and is
    still healthy, ready for the next call.

    On a timeout the *whole process tree* is killed, not just the worker:
    some fitters spawn their own children (wittgenstein's beam RIPPER
    does ``mp.Pool(cpu_count())`` internally, sklearn/joblib can too), and
    a bare `Process.terminate()` leaves those orphaned -- still holding
    CPU, so the *next* fits in the fold crawl and one unlucky fold can
    balloon to hours even though every individual `run()` returned
    "timeout" on schedule. `_kill_tree` walks the children explicitly.
    """

    def __init__(self, timeout: float = FIT_TIMEOUT_SECONDS):
        self.timeout = timeout
        self.ctx = mp.get_context("spawn")
        self.task_q = None
        self.result_q = None
        self.proc = None
        self._start()

    def _start(self):
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.proc = self.ctx.Process(target=_worker_loop, args=(self.task_q, self.result_q), daemon=True)
        self.proc.start()

    def _kill_worker(self):
        pid = self.proc.pid
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join()
        _kill_tree(pid)



    def run(self, fn, *args, **kwargs):
        """Returns `(result, None)` on success, or `(None, error_message)`
        -- for a timeout, `error_message` is exactly `"timeout"`;
        otherwise it's `fn`'s own exception, stringified. Never raises.
        """
        self.task_q.put((fn, args, kwargs))
        try:
            status, payload = self.result_q.get(timeout=self.timeout)
        except queue.Empty:
            self._kill_worker()
            self._start()
            return None, "timeout"
        if status == "ok":
            return payload, None
        return None, payload

    def close(self):
        self._kill_worker()


def _drop_degenerate(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Columns that are useless regardless of missing-value strategy:
    entirely missing, or constant on their known (non-missing) values."""
    degenerate = [
        col for col in df.columns
        if col != target_col and (df[col].isna().all() or df[col].nunique(dropna=True) <= 1)
    ]
    return df.drop(columns=degenerate) if degenerate else df


# Pima diabetes records unmeasured glucose/blood pressure/skin thickness/
# insulin/BMI as a literal 0, not NaN -- a real reading of 0 is physiologically
# impossible for these (unlike `preg`, where 0 pregnancies is a real value),
# so treat it as missing before anything downstream (imputation or
# discretization) ever sees it as if it were a real measurement. See
# JF_DATA_GENERATION_ISSUES.md point 4: the discretizer was putting a cut
# right above the zeros, turning the lowest bin into a silent missing-value
# flag (`insu<14.5` held 48.2% of fold 0's training rows, matching the
# source's 374 zeros out of 768).
DIABETES_ZERO_AS_MISSING = ("plas", "pres", "skin", "insu", "mass")


def _mark_dataset_specific_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "diabetes":
        df = df.copy()
        for col in DIABETES_ZERO_AS_MISSING:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
    return df


def _fill_missing(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Impute missing values -- for learners that need a fully non-missing
    matrix (the "Original"/external-learner workflow: sklearn, wittgenstein,
    imodels). pyrulearn's own build_dataspec/binarize don't need this at
    all -- see `load_openml_raw`."""
    df = _drop_degenerate(df, target_col).copy()

    for col in df.columns:
        if col == target_col or not df[col].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(object)
            df[col] = df[col].where(df[col].notna(), "?")

    still_constant = [col for col in df.columns if col != target_col and df[col].nunique(dropna=True) <= 1]
    if still_constant:
        df = df.drop(columns=still_constant)
    return df


def _load_and_clean(name: str, version=1):
    """One `fetch_openml` call, dataset-specific missing-value markup
    applied once, then both downstream views built from that *same* frame
    (so they stay row-aligned): `imputed_df` (`_fill_missing`, for external
    learners) and `raw_df` (only degenerate columns dropped, real NaN kept,
    for pyrulearn's own build_dataspec/binarize -- see `load_openml_raw`'s
    docstring for why that's correct rather than a gap to fill)."""
    d = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    target_col = d.target.name
    marked = _mark_dataset_specific_missing(d.frame, name)
    imputed_df = _fill_missing(marked, target_col)
    raw_df = _drop_degenerate(marked, target_col)
    return imputed_df, raw_df, target_col


def load_openml(name: str, version=1):
    """Missing values imputed (`_fill_missing`) -- for the "Original"
    workflow's external learners (sklearn, wittgenstein, imodels), not all
    of which are known to tolerate raw NaN input. pyrulearn's own
    build_dataspec/binarize should use `load_openml_raw` instead."""
    df, _raw_df, target_col = _load_and_clean(name, version)
    return df, target_col


def load_openml_raw(name: str, version=1):
    """Real missing values preserved (only degenerate columns dropped, and
    `name`'s own known missing-as-a-different-value quirks fixed -- see
    `DIABETES_ZERO_AS_MISSING`) -- for pyrulearn's own build_dataspec/
    binarize, whose default `MissingStrategy.NEVER_COVERS` already makes
    every feature derived from a missing value False, for numeric and
    nominal attributes alike. `load_openml`'s imputation (silently filling
    a numeric NaN with the column median, or folding a nominal NaN into an
    ordinary-looking "?" category) exists only for external learners that
    need a fully non-missing matrix -- it was never the right input for
    pyrulearn's own binarization, which already has a correct answer for
    this and doesn't need one imposed on it upstream."""
    _df, raw_df, target_col = _load_and_clean(name, version)
    return raw_df, target_col


def _one_hot_expand(df: pd.DataFrame, nominal_cols, categories) -> pd.DataFrame:
    """Adds one real 0/1 column per (nominal attribute, category) pair,
    named `f"{col}={category}"` -- see the module docstring's wrinkle 1
    for why one-hot, not ordinal. Built as one batch `pd.concat`, not a
    per-column assignment loop -- see the original version of this demo
    for why that matters at scale.
    """
    new_cols = {
        f"{c}={cat}": (df[c] == cat).astype(float)
        for c in nominal_cols
        for cat in categories[c]
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _sklearn_feature_names(feature_cols, nominal_cols, categories):
    names = []
    for c in feature_cols:
        if c in nominal_cols:
            names.extend(f"{c}={cat}" for cat in categories[c])
        else:
            names.append(c)
    return names


# ---------------------------------------------------------------- Weka glue ----

def run_weka(classifier_key: str, arff_path: str) -> str:
    """Fit `WEKA_CLASSES[classifier_key]` on `arff_path`'s full training
    data (``-no-cv``: Weka's own cross-validation is skipped, this demo
    runs its own) via the installed `weka.jar`, and return its printed
    stdout -- exactly the text `pyrulearn.interfaces.weka` parses.

    Raises with the actual stderr included if stdout comes back without
    a rule dump -- Weka's own CLI wrapper is unhelpfully generic here
    (e.g. a malformed ARFF surfaces only as "Weka exception: Can't open
    file X", regardless of the real cause), so surfacing stderr directly
    is the difference between a five-second diagnosis and reproducing
    the failure by hand outside Python.
    """
    cmd = [WEKA_JAVA, "-cp", WEKA_JAR, WEKA_CLASSES[classifier_key], "-t", arff_path, "-no-cv"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=FIT_TIMEOUT_SECONDS)
    if "===" not in result.stdout:
        raise RuntimeError(
            f"weka.jar produced no model output for {classifier_key} on {arff_path} "
            f"(return code {result.returncode}). stderr:\n{result.stderr}"
        )
    return result.stdout


def _run_weka_safe(classifier_key: str, arff_path: str):
    """`(stdout, None)` on success, or `(None, error_message)` -- `"timeout"`
    if the subprocess ran past `FIT_TIMEOUT_SECONDS`, otherwise `run_weka`'s
    own exception, stringified. Never raises -- same `(result, error)`
    convention as `TimeoutRunner.run`, so `run_fold` can handle the Weka
    trio and the in-process models the same way; a subprocess is already
    independently killable on timeout, so no `TimeoutRunner` needed here.
    """
    try:
        return run_weka(classifier_key, arff_path), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:  # noqa: BLE001 -- report any failure back, don't crash the fold
        return None, f"{type(e).__name__}: {e}"


def _fit_brs(rep: BooleanDataRepresentation):
    """`BayesianRuleSet` with a bounded retry over `random_state` against
    imodels' own "clean"-move bug -- same reasoning as
    tests/test_imodels_import.py's version of this."""
    from pyrulearn.interfaces.imodels import BayesianRuleSet

    last_err = None
    for rs in BRS_RANDOM_STATE_CANDIDATES:
        try:
            return BayesianRuleSet(**{**BRS_PARAMS, "random_state": rs}).fit(rep)
        except ValueError as e:
            if "list.remove" not in str(e):
                raise
            last_err = e
    raise RuntimeError(f"BayesianRuleSet hit imodels' clean-move bug on every candidate random_state: {last_err}")


# ---- module-level fit functions for TimeoutRunner (must be picklable by
# reference, so no closures/lambdas -- see TimeoutRunner's docstring) ----

def _fit_tree_binarized(rep):
    return DecisionTree(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(rep)


def _fit_forest_binarized(rep):
    # the flat single-bag view (not the per-tree EnsembleModel default): its predict takes
    # a combiner=, so this is evaluated with MacroVoteCombiner exactly like the Original path
    return RandomForest(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(
        rep, model=FlatRuleSet)


def _fit_ripper_binarized(rep, pos_class):
    return RIPPERk(pos_class=pos_class, k=RIPPER_K, random_state=RANDOM_STATE).fit(rep)


def _fit_irep_binarized(rep, pos_class):
    return IREP(pos_class=pos_class).fit(rep)


def _fit_brl_binarized(rep):
    from pyrulearn.interfaces.imodels import BayesianRuleList

    return BayesianRuleList(random_state=RANDOM_STATE, **BRL_PARAMS).fit(rep)


def _fit_tree_native(X, y, feature_names):
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE).fit(X, y)
    importer = SklearnTreeImporter()
    ds = importer.infer_dataspec(tree, feature_names=feature_names)
    rules = importer.import_model(tree, ds, feature_names=feature_names)
    return rules, ds


def _fit_forest_native(X, y, feature_names):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    ).fit(X, y)
    importer = RandomForestImporter()
    ds = importer.infer_dataspec(forest, feature_names=feature_names)
    # data= measures each leaf's own training stats -- MacroVoteCombiner (see run_fold's
    # `combiners`) is a DistributionCombiner and raises on stat-less rules
    train_rep = BooleanDataRepresentation(ds, binarize(ds, pd.DataFrame(X, columns=feature_names)), y)
    rules = importer.import_model(forest, ds, feature_names=feature_names, data=train_rep)
    return rules, ds


def _fit_ripper_native(X, y, feature_names, pos_class):
    model = RIPPERk(pos_class=pos_class, k=RIPPER_K, random_state=RANDOM_STATE).fit_external(
        X, y, feature_names=feature_names
    )
    importer = RIPPERImporter()
    ds = importer.infer_dataspec(model, feature_names=feature_names)
    rules = importer.import_model(model, ds, feature_names=feature_names)
    return rules, ds


def _fit_irep_native(X, y, feature_names, pos_class):
    model = IREP(pos_class=pos_class).fit_external(X, y, feature_names=feature_names)
    importer = IREPImporter()
    ds = importer.infer_dataspec(model, feature_names=feature_names)
    rules = importer.import_model(model, ds, feature_names=feature_names)
    return rules, ds


# ------------------------------------------------------------ per-fold metrics ----

def _pct(x: float) -> str:
    """Format a 0..1 accuracy fraction as a percentage with 4 significant
    digits in the 0..100 range, e.g. ``0.995 -> "99.50"`` -- easier to
    compare at a glance than the equivalent 0..1 fraction. `nan` formats
    as the string ``"nan"`` (Python's own `float.__format__` handles this,
    no special-casing needed here)."""
    return f"{x * 100:.2f}"


def _rule_stats(rules) -> tuple:
    """`(n_rules, avg_conditions)` read directly off a fitted
    `RuleModel`."""
    n = len(rules.rules)
    avg_cond = float(np.mean([len(r.conditions) for r in rules.rules])) if n else 0.0
    return n, avg_cond


def _eval(rules, rep, y, combiner=None) -> tuple:
    """`(accuracy, n_rules, avg_conditions)` for one fitted model against
    one Boolean test data."""
    preds = rules.predict(rep, combiner=combiner) if combiner is not None else rules.predict(rep)
    acc = float(np.mean(np.asarray(preds) == y))
    n_rules, avg_cond = _rule_stats(rules)
    return acc, n_rules, avg_cond


def run_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, raw_train_df: pd.DataFrame,
             raw_test_df: pd.DataFrame, target_col: str, arff_prefix: str, runner: TimeoutRunner):
    """Returns `(metrics, merged_ok, failures)`. `metrics` maps `(workflow,
    model)` to a dict with `acc`/`n_rules`/`avg_cond`/`fit_time`, for every
    model that fit within `FIT_TIMEOUT_SECONDS` this fold. `failures` maps
    the same kind of key to an error string (`"timeout"`, or the exception
    stringified) for every model that didn't -- skipped, not raised, so one
    pathological fit can't take down the fold or the run; see
    `TimeoutRunner`/`_run_weka_safe`. `merged_ok` is False for the rare fold
    where group 1's "Original" DataSpecs couldn't be merged (see the module
    docstring's wrinkle 2).

    `train_df`/`test_df` (missing values imputed) feed the "Original"
    workflow's external learners; `raw_train_df`/`raw_test_df` (same rows,
    real missing values kept) feed "Binarized"'s own `build_dataspec`/
    `binarize` -- see `load_openml_raw`'s docstring for why that's the
    correct input for those, not `load_openml`'s imputed one."""
    os.makedirs(ARFF_DIR, exist_ok=True)  # gitignored scratch dir, absent in a fresh checkout
    feature_cols = [c for c in train_df.columns if c != target_col]
    nominal_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(train_df[c])]
    categories = {c: sorted(train_df[c].dropna().unique().tolist()) for c in nominal_cols}

    raw_feature_cols = [c for c in raw_train_df.columns if c != target_col]
    raw_nominal_cols = [c for c in raw_feature_cols if not pd.api.types.is_numeric_dtype(raw_train_df[c])]
    raw_arff_types = {c: ("nominal" if c in raw_nominal_cols else "numeric") for c in raw_feature_cols}

    classes = sorted(pd.unique(train_df[target_col]).tolist())
    if len(classes) != 2:
        raise ValueError(f"expected a binary target, got {len(classes)} classes: {classes}")
    class_a, class_b = classes
    pos_directions = [(class_a, "A"), (class_b, "B")]

    train_y = train_df[target_col].to_numpy()
    test_y = test_df[target_col].to_numpy()

    metrics: dict = {}
    failures: dict = {}

    def record(workflow, model, acc, n_rules, avg_cond, fit_time):
        metrics[(workflow, model)] = {"acc": acc, "n_rules": n_rules, "avg_cond": avg_cond, "fit_time": fit_time}

    def fail(workflow, model, error):
        failures[(workflow, model)] = error

    # ---------------- "Binarized": binarize first, one DataSpec throughout ----
    # built from raw_train_df (real missing values kept, not train_df's
    # imputed one) so NEVER_COVERS -- the correct default -- actually gets
    # to do its job; arff_types stays imputed-df-derived, still needed below
    # for the "Original" Weka path's write_arff call.
    arff_types = {c: ("nominal" if c in nominal_cols else "numeric") for c in feature_cols}
    ds1 = build_dataspec(raw_train_df, target=target_col, arff_types=raw_arff_types,
                         max_intervals=MAX_INTERVALS).build()
    train_rep1 = BooleanDataRepresentation(ds1, binarize(ds1, raw_train_df), train_y)
    test_rep1 = BooleanDataRepresentation(ds1, binarize(ds1, raw_test_df), test_y)

    # imodels' BRL/BRS scale badly in feature count (BRL's FP-growth candidate
    # generation; BRS's discretization fits one RandomForest per rule length
    # with n_estimators ~ (2*n_features)**length) -- and they gain nothing from
    # explicit negation features: BRL only searches "present item" sets, and BRS
    # builds its own {name, name_neg} column space internally before searching.
    # Since build_dataspec now adds negation features by default (~2x the
    # columns), give these two a negation-free DataSpec so they stay within
    # FIT_TIMEOUT_SECONDS on the larger datasets, same as before the redesign.
    # raw_train_df here too: brl/brs's "Original" label below still reuses
    # this same already-Boolean matrix (see the comment there) -- there's no
    # actual raw/external fit for these two either way, so both labels
    # benefit from the correct missing-value encoding equally.
    ds1_bool = build_dataspec(raw_train_df, target=target_col, arff_types=raw_arff_types,
                              max_intervals=MAX_INTERVALS, include_negations=False).build()
    train_rep1_bool = BooleanDataRepresentation(ds1_bool, binarize(ds1_bool, raw_train_df), train_y)
    test_rep1_bool = BooleanDataRepresentation(ds1_bool, binarize(ds1_bool, raw_test_df), test_y)

    # brl/brs's "Original" call fits on a fresh BooleanDataRepresentation built
    # from the already-Boolean matrix via a plain DataFrame, instead of reusing
    # "Binarized"'s in-memory fit under both labels -- a genuinely separate call,
    # not a file round-trip (write_csv/read_csv fidelity is already covered by
    # tests/test_data_io.py; no need to re-exercise it here).
    bool_train_df = pd.DataFrame(train_rep1_bool.X.astype(int), columns=ds1_bool.feature_names)
    bool_train_df[target_col] = train_y
    train_rep1_copy = BooleanDataRepresentation.from_dataframe(bool_train_df, label_col=target_col, spec=ds1_bool)

    if "tree" in ENABLED_MODELS:
        t0 = time.time()
        tree_rules1, err = runner.run(_fit_tree_binarized, train_rep1)
        if err is None:
            record("Binarized", "tree", *_eval(tree_rules1, test_rep1, test_y), time.time() - t0)
        else:
            fail("Binarized", "tree", err)

    if "forest" in ENABLED_MODELS:
        t0 = time.time()
        forest_rules1, err = runner.run(_fit_forest_binarized, train_rep1)
        if err is None:
            record("Binarized", "forest", *_eval(forest_rules1, test_rep1, test_y, combiner=MacroVoteCombiner()),
                   time.time() - t0)
        else:
            fail("Binarized", "forest", err)

    for pos, tag in pos_directions:
        if f"ripper_{tag}" in ENABLED_MODELS:
            t0 = time.time()
            r, err = runner.run(_fit_ripper_binarized, train_rep1, pos)
            if err is None:
                record("Binarized", f"ripper_{tag}", *_eval(r, test_rep1, test_y), time.time() - t0)
            else:
                fail("Binarized", f"ripper_{tag}", err)

        if f"irep_{tag}" in ENABLED_MODELS:
            t0 = time.time()
            i, err = runner.run(_fit_irep_binarized, train_rep1, pos)
            if err is None:
                record("Binarized", f"irep_{tag}", *_eval(i, test_rep1, test_y), time.time() - t0)
            else:
                fail("Binarized", f"irep_{tag}", err)

    if "brl" in ENABLED_MODELS:
        t0 = time.time()
        brl_rules, err = runner.run(_fit_brl_binarized, train_rep1_bool)
        if err is None:
            record("Binarized", "brl", *_eval(brl_rules, test_rep1_bool, test_y), time.time() - t0)
        else:
            fail("Binarized", "brl", err)

        t0 = time.time()
        brl_rules2, err = runner.run(_fit_brl_binarized, train_rep1_copy)
        if err is None:
            record("Original", "brl", *_eval(brl_rules2, test_rep1_bool, test_y), time.time() - t0)
        else:
            fail("Original", "brl", err)

    if "brs" in ENABLED_MODELS:
        # BRS's "_neg" items import as literals on paired negation features,
        # so BayesianRuleSetImporter binds its rules to a negation-augmented
        # spec (built from the model's own columns) even though we fed it the
        # negation-free ds1_bool. Predict against a matching test rep.
        def _brs_test_rep(rules):
            spec = rules.rules[0].dataspec if rules.rules else ds1_bool
            if spec.feature_names == ds1_bool.feature_names:
                # every column already has a negation partner (e.g. all
                # attributes binary), so the importer kept ds1_bool's own
                # feature space (a copy -- the fit ran in a worker process)
                return BooleanDataRepresentation(spec, test_rep1_bool.X, test_y)
            bool_df = pd.DataFrame(test_rep1_bool.X.astype(int), columns=list(ds1_bool.feature_names))
            return BooleanDataRepresentation(spec, binarize(spec, bool_df), test_y)

        t0 = time.time()
        brs_rules, err = runner.run(_fit_brs, train_rep1_bool)
        if err is None:
            record("Binarized", "brs", *_eval(brs_rules, _brs_test_rep(brs_rules), test_y), time.time() - t0)
        else:
            fail("Binarized", "brs", err)

        t0 = time.time()
        brs_rules2, err = runner.run(_fit_brs, train_rep1_copy)
        if err is None:
            record("Original", "brs", *_eval(brs_rules2, _brs_test_rep(brs_rules2), test_y), time.time() - t0)
        else:
            fail("Original", "brs", err)

    # Weka trio, "Binarized": ds1's already-Boolean data, safe placeholder names
    weka_needed = any(k in ENABLED_MODELS for k in ("jrip", "part", "j48"))
    if weka_needed:
        plain_names = [f"f{i}" for i in range(ds1.n_features)]
        train_bool_df = pd.DataFrame(train_rep1.X.astype(int), columns=plain_names)
        train_bool_df[target_col] = train_y
        train_arff_w1 = os.path.join(ARFF_DIR, f"{arff_prefix}_w1_train.arff")
        write_arff(train_bool_df, target_col, train_arff_w1)  # int columns -> inferred numeric

        for key in ("jrip", "part", "j48"):
            if key not in ENABLED_MODELS:
                continue
            t0 = time.time()
            stdout, err = _run_weka_safe(key, train_arff_w1)
            fit_time = time.time() - t0
            if err is not None:
                fail("Binarized", key, err)
                continue
            importer = WEKA_IMPORTERS[key]()
            rules = importer.parse(stdout)
            ds = importer.dataspec
            test_bool_df = pd.DataFrame(binarize(ds1, raw_test_df).astype(int), columns=plain_names)
            rep = BooleanDataRepresentation(ds, binarize(ds, test_bool_df), test_y)
            record("Binarized", key, *_eval(rules, rep, test_y), fit_time)

    # ---------------- "Original": fit raw/native, discover + merge DataSpecs ----
    sklearn_names = _sklearn_feature_names(feature_cols, nominal_cols, categories)
    train_df_oh = _one_hot_expand(train_df, nominal_cols, categories)
    train_X_sk = train_df_oh[sklearn_names].to_numpy(dtype=float)
    train_X_wt = train_df[feature_cols].to_numpy()

    per_model = []

    if "tree" in ENABLED_MODELS:
        t0 = time.time()
        result, err = runner.run(_fit_tree_native, train_X_sk, train_y, sklearn_names)
        tree_time = time.time() - t0
        if err is None:
            rules_tree, ds_tree = result
            per_model.append(("tree", rules_tree, ds_tree, tree_time))
        else:
            fail("Original", "tree", err)

    if "forest" in ENABLED_MODELS:
        t0 = time.time()
        result, err = runner.run(_fit_forest_native, train_X_sk, train_y, sklearn_names)
        forest_time = time.time() - t0
        if err is None:
            rules_forest, ds_forest = result
            per_model.append(("forest", rules_forest, ds_forest, forest_time))
        else:
            fail("Original", "forest", err)

    for pos, tag in pos_directions:
        if f"ripper_{tag}" in ENABLED_MODELS:
            t0 = time.time()
            result, err = runner.run(_fit_ripper_native, train_X_wt, train_y, feature_cols, pos)
            ripper_time = time.time() - t0
            if err is None:
                rules_r, ds_r = result
                per_model.append((f"ripper_{tag}", rules_r, ds_r, ripper_time))
            else:
                fail("Original", f"ripper_{tag}", err)

        if f"irep_{tag}" in ENABLED_MODELS:
            t0 = time.time()
            result, err = runner.run(_fit_irep_native, train_X_wt, train_y, feature_cols, pos)
            irep_time = time.time() - t0
            if err is None:
                rules_i, ds_i = result
                per_model.append((f"irep_{tag}", rules_i, ds_i, irep_time))
            else:
                fail("Original", f"irep_{tag}", err)

    test_df_oh = _one_hot_expand(test_df, nominal_cols, categories)

    combiners = {"forest": MacroVoteCombiner()}
    if not per_model:
        merged_ok = True  # nothing fit successfully this fold -- nothing to merge or evaluate
    else:
        try:
            merged = per_model[0][2]
            for _, _, ds, _ in per_model[1:]:
                merged = merge_dataspecs(merged, ds).build()
            merged_ok = True
        except ValueError:
            merged = None
            merged_ok = False

        if merged_ok:
            test_rep2 = BooleanDataRepresentation(merged, binarize(merged, test_df_oh), test_y)
            train_rep2 = None
            for name, rules, _, fit_time in per_model:
                remapped = rules.remap(merged)
                if name in combiners:
                    # remap drops measured stats; a DistributionCombiner (MacroVoteCombiner)
                    # needs them, so re-measure each leaf on the training rows under `merged`
                    if train_rep2 is None:
                        train_rep2 = BooleanDataRepresentation(merged, binarize(merged, train_df_oh), train_y)
                    remapped = FlatRuleSet(annotate_rules(remapped.rules, train_rep2))
                record("Original", name, *_eval(remapped, test_rep2, test_y, combiner=combiners.get(name)), fit_time)
        else:
            for name, rules, ds, fit_time in per_model:
                rep = BooleanDataRepresentation(ds, binarize(ds, test_df_oh), test_y)
                record("Original", name, *_eval(rules, rep, test_y, combiner=combiners.get(name)), fit_time)

    # Weka trio, "Original": raw native columns, no one-hot needed
    if weka_needed:
        train_arff_w2 = os.path.join(ARFF_DIR, f"{arff_prefix}_w2_train.arff")
        write_arff(train_df, target_col, train_arff_w2, arff_types=arff_types)

        for key in ("jrip", "part", "j48"):
            if key not in ENABLED_MODELS:
                continue
            t0 = time.time()
            stdout, err = _run_weka_safe(key, train_arff_w2)
            fit_time = time.time() - t0
            if err is not None:
                fail("Original", key, err)
                continue
            importer = WEKA_IMPORTERS[key]()
            rules = importer.parse(stdout)
            ds = importer.dataspec
            rep = BooleanDataRepresentation(ds, binarize(ds, test_df), test_y)
            record("Original", key, *_eval(rules, rep, test_y), fit_time)

    return metrics, merged_ok, failures


# ---------------------------------------------------------- overview tables ----

# (field, higher_is_better) for each of the four per-(workflow, model) criteria
# `run_dataset` already aggregates into each dataset's `summary` dict (as
# `{field}_{workflow}_{model}`). Fewer/simpler rules is treated as "better"
# here (interpretability), matching how this whole demo frames rule count and
# condition count -- not an objective fact about the data, just the same
# framing used throughout the module docstring; easy to flip if you disagree.
CRITERIA = [("acc", True), ("time", False), ("nrules", False), ("avgcond", False)]


def _rank_dataset_by(results_row: dict, cols, field: str, higher_is_better: bool) -> dict:
    """Rank every `(workflow, model)` combo in `cols` within one dataset's
    `summary` row by `field` (one of `CRITERIA`'s field names) -- `nan`
    (failed) entries tie for last place; everyone else gets standard
    average-tie ranks (1 = best). Returns `{col: rank}`."""
    s = pd.Series({c: results_row[f"{field}_{c}"] for c in cols})
    ok = s.dropna()
    ranks = ok.rank(ascending=not higher_is_better, method="average").to_dict()
    n_ok = len(ok)
    n_fail = len(cols) - n_ok
    if n_fail:
        last_rank = n_ok + (n_fail + 1) / 2
        for c in cols:
            if c not in ranks:
                ranks[c] = last_rank
    return ranks


def _better_counts(results, model: str, field: str, higher_is_better: bool):
    """Across all datasets in `results`, how many times "Binarized"'s value
    beat "Original"'s (and vice versa) for `model` on `field` -- a tie
    counts 0.5 for each side; a dataset where *both* failed isn't counted
    either way (nothing to compare); a dataset where only one side failed
    counts as a full point for whichever one actually produced a result.
    Returns `(bin_better, orig_better)`."""
    bin_better = orig_better = 0.0
    for r in results:
        a, b = r[f"{field}_Binarized_{model}"], r[f"{field}_Original_{model}"]
        a_nan, b_nan = np.isnan(a), np.isnan(b)
        if a_nan and b_nan:
            continue
        elif a_nan:
            orig_better += 1.0
        elif b_nan:
            bin_better += 1.0
        elif a == b:
            bin_better += 0.5
            orig_better += 0.5
        elif (a > b) == higher_is_better:
            bin_better += 1.0
        else:
            orig_better += 1.0
    return bin_better, orig_better


def _build_overview_tables(results: list, cols: list) -> list:
    """Cross-dataset overview: average performance per algorithm, average
    rank per algorithm (one column per `CRITERIA` field, failures tied
    last), and how often "Binarized" vs. "Original" won per model per
    field. Returns markdown lines (also printed to console verbatim by the
    caller)."""
    def sort_key(c):
        w, m = c.split("_", 1)
        return (m, 0 if w == "Binarized" else 1)

    sorted_cols = sorted(cols, key=sort_key)

    lines = ["## Overview evaluation (across all datasets)\n\n"]

    # ---- average performance per algorithm ----
    lines.append("**Average performance across datasets**\n\n")
    lines.append("| algorithm | accuracy (%) | fit time (s) | n_rules | avg_conditions | datasets fully failed |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for c in sorted_cols:
        accs = np.array([r[f"acc_{c}"] for r in results], dtype=float)
        times = np.array([r[f"time_{c}"] for r in results], dtype=float)
        nrules = np.array([r[f"nrules_{c}"] for r in results], dtype=float)
        avgconds = np.array([r[f"avgcond_{c}"] for r in results], dtype=float)
        n_fully_failed = int(np.isnan(accs).sum())
        lines.append(
            f"| {c.replace('_', '/', 1)} | {_pct(np.nanmean(accs))} | {np.nanmean(times):.3f} | "
            f"{np.nanmean(nrules):.1f} | {np.nanmean(avgconds):.2f} | {n_fully_failed} |\n"
        )

    # ---- average rank per algorithm, one column per criterion ----
    lines.append("\n**Average rank per criterion** (1 = best of "
                  f"{len(cols)}; failed entries tie for last)\n\n")
    lines.append("| algorithm | rank (accuracy) | rank (fit time) | rank (n_rules) | rank (avg_conditions) |\n")
    lines.append("|---|---|---|---|---|\n")
    avg_ranks = {c: {} for c in cols}
    for field, higher_is_better in CRITERIA:
        per_dataset_ranks = {c: [] for c in cols}
        for r in results:
            ranks = _rank_dataset_by(r, cols, field, higher_is_better)
            for c in cols:
                per_dataset_ranks[c].append(ranks[c])
        for c in cols:
            avg_ranks[c][field] = float(np.mean(per_dataset_ranks[c]))
    for c in sorted_cols:
        ar = avg_ranks[c]
        lines.append(
            f"| {c.replace('_', '/', 1)} | {ar['acc']:.2f} | {ar['time']:.2f} | "
            f"{ar['nrules']:.2f} | {ar['avgcond']:.2f} |\n"
        )

    # ---- Binarized vs. Original: which was better, per model, per criterion ----
    lines.append("\n**Binarized vs. Original: how often each was better, per model** "
                  "(ties count 0.5 each per side; a dataset where both failed isn't "
                  "counted for either side)\n\n")
    lines.append("| model | accuracy (Bin / Orig) | fit time (Bin / Orig) | "
                  "n_rules (Bin / Orig) | avg_conditions (Bin / Orig) |\n")
    lines.append("|---|---|---|---|---|\n")
    for m in sorted(ACTIVE_MODELS):
        counts = [_better_counts(results, m, field, higher_is_better) for field, higher_is_better in CRITERIA]
        lines.append(f"| {m} | " + " | ".join(f"{b:.1f} / {o:.1f}" for b, o in counts) + " |\n")

    return lines


def run_dataset(name: str, df: pd.DataFrame, raw_df: pd.DataFrame, target_col: str,
                runner: TimeoutRunner, max_folds=None):
    """`max_folds`, if given, runs only the first `max_folds` of the
    dataset's `N_FOLDS` splits -- fold *sizes* stay exactly as they'd be in
    a full run (the same `N_FOLDS`-way split is always computed; only the
    loop over it is truncated), so `max_folds=1` gives a quick per-dataset
    timing/sanity preview without changing what each fold looks like.

    `df` and `raw_df` are the same rows (same fold split applies to both --
    see `_load_and_clean`), just missing values imputed vs. kept as real
    NaN; see `run_fold`."""
    y = df[target_col].to_numpy()
    classes = sorted(pd.unique(y).tolist())
    print(f"\n{'=' * 78}\n{name}  (n={len(df)}, attributes={df.shape[1] - 1}, "
          f"A={classes[0]!r}, B={classes[1]!r})\n{'=' * 78}")

    md = [f"## {name}\n\n", f"n={len(df)}, attributes={df.shape[1] - 1}, "
          f"A={classes[0]!r}, B={classes[1]!r}\n\n"]

    try:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df, y))
    except ValueError as e:
        print(f"  StratifiedKFold unavailable ({e}); falling back to plain KFold")
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(df))
    if max_folds is not None:
        folds = folds[:max_folds]

    keys = [(w, m) for w in WORKFLOW_NAMES for m in ACTIVE_MODELS]
    all_metrics = {k: {"acc": [], "n_rules": [], "avg_cond": [], "fit_time": []} for k in keys}
    n_fallback = 0
    n_failures = {k: 0 for k in keys}
    t0 = time.time()
    for i, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        raw_train_df = raw_df.iloc[train_idx].reset_index(drop=True)
        raw_test_df = raw_df.iloc[test_idx].reset_index(drop=True)
        fold_metrics, merged_ok, failures = run_fold(
            train_df, test_df, raw_train_df, raw_test_df, target_col,
            arff_prefix=f"{name}_{i}", runner=runner
        )
        if not merged_ok:
            n_fallback += 1
        for key, v in fold_metrics.items():
            for field in ("acc", "n_rules", "avg_cond", "fit_time"):
                all_metrics[key][field].append(v[field])
        for key in failures:
            n_failures[key] += 1
        line = "  ".join(
            f"{w}/{m}={_pct(fold_metrics[(w, m)]['acc'])}" if (w, m) in fold_metrics else f"{w}/{m}=FAIL"
            for w in WORKFLOW_NAMES for m in ACTIVE_MODELS
        )
        note = "" if merged_ok else "  [Original: per-model fallback, DataSpecs didn't merge]"
        if failures:
            note += f"  [failed/timed out: {', '.join(f'{w}/{m} ({e})' for (w, m), e in failures.items())}]"
        print(f"  fold {i + 1} ({time.time() - t0:.1f}s so far): {line}{note}")
    dt = time.time() - t0

    total_failures = sum(n_failures.values())
    print(f"\n  Summary over {len(folds)} folds ({dt:.1f}s, {n_fallback} used 'Original's per-model fallback"
          f"{f', {total_failures} model-fold timeout(s)/failure(s)' if total_failures else ''}):")
    if n_fallback:
        md.append(f"*{n_fallback}/{len(folds)} fold(s) used 'Original's per-model fallback "
                   f"(DataSpecs didn't merge -- see the module docstring).*\n\n")
    if total_failures:
        md.append(f"*{total_failures} model-fold combination(s) timed out (> {FIT_TIMEOUT_SECONDS}s) or raised "
                   f"and were skipped for that fold (n/a in the tables below). `brl`/`brs` are "
                   f"expected to hit this on datasets with many discretized features -- see "
                   f"`FIT_TIMEOUT_SECONDS`'s comment; it is a scaling limit of those algorithms, not a bug.*\n\n")

    md.append("**Accuracy**\n\n| workflow | model | accuracy | failed folds |\n|---|---|---|---|\n")
    summary = {"name": name, "n": len(df), "time": dt, "n_fallback": n_fallback}
    for w in WORKFLOW_NAMES:
        for m in ACTIVE_MODELS:
            vals = all_metrics[(w, m)]
            fails = n_failures[(w, m)]
            if vals["acc"]:
                mean, std = float(np.mean(vals["acc"])), float(np.std(vals["acc"]))
                acc_str = f"{_pct(mean)} +/- {_pct(std)}"
                time_mean = float(np.mean(vals["fit_time"]))
                nrules_mean = float(np.mean(vals["n_rules"]))
                avgcond_mean = float(np.mean(vals["avg_cond"]))
            else:
                mean = time_mean = nrules_mean = avgcond_mean = float("nan")
                acc_str = "n/a"
            print(f"    {w:<10} {m:<10} acc={acc_str}" + (f"  ({fails}/{len(folds)} failed)" if fails else ""))
            md.append(f"| {w} | {m} | {acc_str} | {fails}/{len(folds)} |\n")
            summary[f"acc_{w}_{m}"] = mean
            summary[f"time_{w}_{m}"] = time_mean
            summary[f"nrules_{w}_{m}"] = nrules_mean
            summary[f"avgcond_{w}_{m}"] = avgcond_mean

    md.append("\n**Fit time (seconds/fold)**\n\n| workflow | model | time |\n|---|---|---|\n")
    for w in WORKFLOW_NAMES:
        for m in ACTIVE_MODELS:
            md.append(f"| {w} | {m} | {summary[f'time_{w}_{m}']:.3f} |\n")

    md.append("\n**Rule complexity**\n\n| workflow | model | n_rules | avg_conditions |\n|---|---|---|---|\n")
    for w in WORKFLOW_NAMES:
        for m in ACTIVE_MODELS:
            md.append(f"| {w} | {m} | {summary[f'nrules_{w}_{m}']:.1f} | {summary[f'avgcond_{w}_{m}']:.2f} |\n")

    md.append(f"\n({dt:.1f}s total)\n\n---\n\n")

    return summary, md


def main(datasets=STANDARD_DATASETS, max_folds=None):
    """`max_folds`, if given, is passed straight through to `run_dataset`
    -- e.g. `max_folds=1` runs only fold 1 of every dataset, a quick
    timing/sanity preview of all `datasets` before committing to a full
    `N_FOLDS`-way run (see the `--preview`/`--max-folds` CLI flags below)."""
    results = []
    fold_note = f" Only the first {max_folds} of {N_FOLDS} fold(s) actually run (preview mode)." \
        if max_folds is not None else ""
    report = [
        "# Binarized vs. Original data preparation -- comparison across binary datasets\n\n",
        f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}. N_FOLDS={N_FOLDS}, "
        f"MAX_INTERVALS={MAX_INTERVALS} ('Binarized' mode only), MAX_DEPTH={MAX_DEPTH}, "
        f"RIPPER_K={RIPPER_K}, N_ESTIMATORS={N_ESTIMATORS} (forest), "
        f"FIT_TIMEOUT_SECONDS={FIT_TIMEOUT_SECONDS}.{fold_note} "
        f"`ripper_A`/`irep_A` treat each dataset's (alphabetically) first "
        f"class as positive, `ripper_B`/`irep_B` the second. `brl`/`brs` fit "
        f"the same already-Boolean matrix under both workflows as two "
        f"independent calls (see the module docstring). Weka's `jrip`/`part`/`j48` "
        f"fit-time includes JVM "
        f"subprocess startup overhead, not just the algorithm itself -- "
        f"see the module docstring.\n\n---\n\n",
    ]

    runner = TimeoutRunner()
    try:
        for name in datasets:
            df, raw_df, target_col = _load_and_clean(name)
            summary, md = run_dataset(name, df, raw_df, target_col, runner=runner, max_folds=max_folds)
            results.append(summary)
            report.extend(md)
    finally:
        runner.close()

    print(f"\n{'=' * 78}\nOverall summary (mean accuracy per workflow/model)\n{'=' * 78}")
    cols = [f"{w}_{m}" for w in WORKFLOW_NAMES for m in ACTIVE_MODELS]
    header = f"{'dataset':<16}" + "".join(f"{c:>14}" for c in cols) + f"{'fallback':>10}"
    print(header)
    report.append("## Overall summary\n\n")
    report.append("### Accuracy\n\n")
    report.append("| dataset | " + " | ".join(cols) + " | fallback folds |\n")
    report.append("|---" * (len(cols) + 2) + "|\n")
    for r in results:
        print(f"{r['name']:<16}" + "".join(f"{_pct(r['acc_' + c]):>14}" for c in cols) + f"{r['n_fallback']:>10}")
        report.append(
            f"| {r['name']} | " + " | ".join(_pct(r['acc_' + c]) for c in cols) + f" | {r['n_fallback']}/{N_FOLDS} |\n"
        )

    report.append("\n### Fit time (seconds/fold)\n\n")
    report.append("| dataset | " + " | ".join(cols) + " |\n")
    report.append("|---" * (len(cols) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['time_' + c]:.3f}" for c in cols) + " |\n")

    report.append("\n### Rule count\n\n")
    report.append("| dataset | " + " | ".join(cols) + " |\n")
    report.append("|---" * (len(cols) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['nrules_' + c]:.1f}" for c in cols) + " |\n")

    report.append("\n### Average conditions per rule\n\n")
    report.append("| dataset | " + " | ".join(cols) + " |\n")
    report.append("|---" * (len(cols) + 1) + "|\n")
    for r in results:
        report.append(f"| {r['name']} | " + " | ".join(f"{r['avgcond_' + c]:.2f}" for c in cols) + " |\n")

    overview_lines = _build_overview_tables(results, cols)
    report.extend(overview_lines)
    print("".join(overview_lines))

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Binarized vs. Original workflow comparison demo.")
    parser.add_argument(
        "--preview", action="store_true",
        help="Run only fold 1 of every dataset (fold sizes match a full "
             f"{N_FOLDS}-fold split -- only the loop is shortened) as a quick "
             "timing/sanity check before committing to the full run. "
             "Shorthand for --max-folds 1.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=None, metavar="N",
        help="Run only the first N of the dataset's N_FOLDS folds. Overrides --preview.",
    )
    args = parser.parse_args()
    main(max_folds=args.max_folds if args.max_folds is not None else (1 if args.preview else None))
