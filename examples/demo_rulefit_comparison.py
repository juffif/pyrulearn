"""
RuleFit: imodels' implementation vs. pyrulearn's native distiller
=================================================================

RuleFit has two independent steps -- candidate rules from a tree
ensemble, then a sparse (L1) logistic regression over them -- and
imodels and pyrulearn differ in both. The variants swap one step at a
time; `DESCRIPTION` below describes the setup and each variant, and is
written into the report too.

Run from the repo root: ``python examples/demo_rulefit_comparison.py``.
Writes `demo_rulefit_comparison_report.md` next to this file.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from pyrulearn.data import BooleanDataRepresentation
from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.interfaces.imodels import ImodelsRuleFit, RuleFitImporter, rulefit_candidates
from pyrulearn.interfaces.sklearn import from_random_forest
from pyrulearn.learners.associative import CARMiner
from pyrulearn.learners.rulefit import RuleFit
from pyrulearn.models import FlatRuleSet
from pyrulearn.learners.seco import Pypper

RANDOM_STATE = 0
N_FOLDS = 5
MAX_ROWS = 1600
MAX_INTERVALS = 6

HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "demo_rulefit_comparison_report.md")

DATASETS = [
    "vote", "breast-cancer", "colic", "credit-approval", "credit-g",
    "diabetes", "sonar", "ionosphere", "tic-tac-toe", "banknote-authentication",
    "hepatitis", "heart-statlog", "kr-vs-kp",
]
VARIANTS = ["imodels", "imodels, f > 0", "trees + native", "trees + native +lin", "CAR + native +lin",
            "RF d3 + native +lin", "RF + native +lin", "random forest", "Pypper"]

DESCRIPTION = f"""\
RuleFit (Friedman & Popescu 2008) has two independent steps: it
generates candidate rules from the nodes of a tree ensemble, and then
fits a sparse (L1-regularized) logistic regression over the rules'
0/1 coverage, keeping the rules with a non-zero weight. imodels'
`RuleFitClassifier` and pyrulearn's native `RuleFit` distiller differ in
both steps, so the variants below change one step at a time, and add
two benchmarks that aren't linear rule models.

**Data and protocol.** {len(DATASETS)} binary-class OpenML datasets,
stratified {N_FOLDS}-fold cross-validation (at most {MAX_ROWS} rows per
dataset, uniformly subsampled). Every fold binarizes its training part
with `build_dataspec` (numeric attributes into at most {MAX_INTERVALS}
intervals, every value feature paired with a negation feature) and
applies that to the test part. All learners see the same binarized
features.

**Variants.**

- *imodels*: `imodels.RuleFitClassifier` with its defaults: 100
  gradient-boosted regression trees with about 4 leaves each (all their
  nodes are candidates), the features as linear terms, and C chosen as
  the least regularization that keeps at most 30 terms, by
  cross-validated accuracy (liblinear). Imported exactly with
  `ImodelsRuleFit`, predicting as imodels' own `predict` does, i.e. the
  positive class where the logistic output f > 0.5.
- *imodels, f > 0*: the same fitted models with the correct logistic
  threshold f > 0 (`imodels_threshold=False`).
- *trees + native*: imodels' candidate generator with the same
  parameters and seed (`rulefit_candidates`), fitted by the native
  `RuleFit`: C chosen from 10 values by 5-fold cross-validated log loss,
  no cap on the number of terms.
- *trees + native +lin*: the same, plus every single feature as a
  candidate (RuleFit's linear terms).
- *CAR + native +lin*: mined class association rules instead of trees
  (support >= 0.05, confidence >= 0.6, length <= 3, or <= 2 above 200
  features, where the frequent negation features make length-3 mining
  explode), of which the 500 most confident are kept (the full pool runs
  to tens of thousands), plus the single features; native fit.
- *RF d3 + native +lin*: the leaves of a shallow random forest (100
  trees, depth <= 3) as candidates, plus the single features; native fit.
- *RF + native +lin*: the leaves of the benchmark random forest below
  (100 fully grown trees) as candidates, plus the single features;
  native fit. Only leaves, not the trees' inner nodes.

**Benchmarks.**

- *random forest*: sklearn's `RandomForestClassifier` with its defaults
  (100 fully grown trees).
- *Pypper*: pyrulearn's re-implementation of RIPPER (a decision list).

**Measures.** Test accuracy; the number of terms (rules with a non-empty
body and a non-zero weight -- leaves for the forest, rules for Pypper);
their mean length (conditions per rule, leaf depth for the forest); and
fit time per fold, including candidate generation."""


def load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    target = d.target.name
    df = d.frame.dropna(subset=[target])
    for c in df.columns:
        if c != target and df[c].isna().any():
            df[c] = (df[c].fillna(df[c].median()) if pd.api.types.is_numeric_dtype(df[c])
                     else df[c].astype(object).where(df[c].notna(), "?"))
    df = df[[c for c in df.columns if c == target or df[c].nunique() > 1]]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    return df, target


def split(df, target, train_idx, test_idx):
    feats = [c for c in df.columns if c != target]
    types = {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "nominal") for c in feats}
    tr, te = df.iloc[train_idx], df.iloc[test_idx]
    ds = build_dataspec(tr, target=target, arff_types=types, max_intervals=MAX_INTERVALS).build()
    mk = lambda part: BooleanDataRepresentation(ds, binarize(ds, part), part[target].astype(str).to_numpy())
    return mk(tr), mk(te)


def size(model):
    body = [r for r in model.rules if r.conditions and getattr(r, "weight", 1) != 0]
    return len(body), (np.mean([len(r.conditions) for r in body]) if body else 0.0)


def car_pool(train, k=500):
    """The `k` most confident mined class association rules (ties: higher support)."""
    max_len = 3 if train.spec.n_features <= 200 else 2
    rules = list(CARMiner(min_support=0.05, max_len=max_len, min_confidence=0.6).fit(train).rules)
    cov = FlatRuleSet(rules).coverage_matrix(train)
    y = np.asarray(train.y)
    hits = np.array([np.sum(cov[i] & (y == r.target)) for i, r in enumerate(rules)])
    n = cov.sum(axis=1)
    order = np.lexsort((-hits, -hits / np.maximum(n, 1)))[:k]
    return FlatRuleSet([rules[i] for i in order])


def forest_size(rf):
    """Total leaves and their mean depth."""
    depths = []
    for est in rf.estimators_:
        t = est.tree_
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            if t.children_left[node] == -1:
                depths.append(d)
            else:
                stack += [(t.children_left[node], d + 1), (t.children_right[node], d + 1)]
    return len(depths), float(np.mean(depths))


def run_fold(train, test):
    out = {}

    def record(name, model, seconds, sizes=None):
        pred = model.predict(test.X) if isinstance(model, RandomForestClassifier) else model.predict(test)
        acc = float(np.mean(np.asarray(pred) == test.y))
        out[name] = (acc, *(sizes or size(model)), seconds)

    t = time.perf_counter()
    learner = ImodelsRuleFit(random_state=RANDOM_STATE)
    ext = learner.fit_external(train.X, train.y, feature_names=train.spec.feature_names)
    fit_s = time.perf_counter() - t
    record("imodels", RuleFitImporter(True).import_model(ext, train.spec, data=train), fit_s)
    record("imodels, f > 0", RuleFitImporter(False).import_model(ext, train.spec, data=train), fit_s)

    t = time.perf_counter()
    pool = rulefit_candidates(train, random_state=RANDOM_STATE)
    pool_s = time.perf_counter() - t
    for name, lin in (("trees + native", False), ("trees + native +lin", True)):
        t = time.perf_counter()
        m = RuleFit(rules=pool, include_features=lin, cv=5, random_state=RANDOM_STATE).fit(train)
        record(name, m, pool_s + time.perf_counter() - t)

    t = time.perf_counter()
    m = RuleFit(rules=car_pool(train), include_features=True, cv=5,
                random_state=RANDOM_STATE).fit(train)
    record("CAR + native +lin", m, time.perf_counter() - t)

    t = time.perf_counter()
    shallow = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
    shallow.fit(train.X, train.y)
    pool = from_random_forest(shallow, dataspec=train.spec)
    m = RuleFit(rules=pool, include_features=True, cv=5, random_state=RANDOM_STATE).fit(train)
    record("RF d3 + native +lin", m, time.perf_counter() - t)

    t = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE).fit(train.X, train.y)
    rf_s = time.perf_counter() - t
    record("random forest", rf, rf_s, forest_size(rf))

    t = time.perf_counter()
    pool = from_random_forest(rf, dataspec=train.spec)
    m = RuleFit(rules=pool, include_features=True, cv=5, random_state=RANDOM_STATE).fit(train)
    record("RF + native +lin", m, rf_s + time.perf_counter() - t)

    t = time.perf_counter()
    m = Pypper(random_state=RANDOM_STATE).fit(train)
    record("Pypper", m, time.perf_counter() - t)
    return out


def main():
    warnings.filterwarnings("ignore")
    rows = []
    for name in DATASETS:
        df, target = load(name)
        y = df[target].astype(str).to_numpy()
        folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(df, y)
        per = {v: [] for v in VARIANTS}
        for tr_idx, te_idx in folds:
            train, test = split(df, target, tr_idx, te_idx)
            for v, res in run_fold(train, test).items():
                per[v].append(res)
        for v in VARIANTS:
            a = np.array(per[v])
            rows.append(dict(dataset=name, variant=v, acc=a[:, 0].mean(), acc_sd=a[:, 0].std(),
                             terms=a[:, 1].mean(), len=a[:, 2].mean(), seconds=a[:, 3].mean()))
        print(name, "  ".join(f"{v}: {np.mean([r[0] for r in per[v]]):.3f}" for v in VARIANTS), flush=True)
    write_report(pd.DataFrame(rows))


def write_report(res):
    acc = res.pivot(index="dataset", columns="variant", values="acc")[VARIANTS].loc[DATASETS]
    terms = res.pivot(index="dataset", columns="variant", values="terms")[VARIANTS].loc[DATASETS]
    length = res.pivot(index="dataset", columns="variant", values="len")[VARIANTS].loc[DATASETS]
    secs = res.pivot(index="dataset", columns="variant", values="seconds")[VARIANTS].loc[DATASETS]
    ranks = acc.rank(axis=1, ascending=False).mean()

    def table(frame, fmt):
        head = "| dataset | " + " | ".join(VARIANTS) + " |\n|---|" + "---:|" * len(VARIANTS) + "\n"
        body = "".join(f"| {d} | " + " | ".join(format(frame.loc[d, v], fmt) for v in VARIANTS) + " |\n"
                       for d in frame.index)
        foot = "| **mean** | " + " | ".join(format(frame[v].mean(), fmt) for v in VARIANTS) + " |\n"
        return head + body + foot

    text = f"""# RuleFit: imodels vs. native

Generated by `examples/demo_rulefit_comparison.py`.

## Setup

{DESCRIPTION}

## Accuracy

{table(acc, ".3f")}
Mean rank (1 = best): {", ".join(f"{v} {ranks[v]:.2f}" for v in VARIANTS)}.

## Model size: terms (non-empty rules with a non-zero weight; leaves for the forest)

{table(terms, ".1f")}
## Mean rule length

{table(length, ".2f")}
## Fit time (seconds per fold)

{table(secs, ".2f")}"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
