"""
examples/export_binary_cv_folds.py
===================================

An ad-hoc export for feeding discretized, binarized cross-validation
data to a tool outside this package, not a demo of this package's own
features. Its output (`_export_binary_cv_folds/`) is regenerated, not
checked in.

For every two-class dataset in the catalog: 10-fold stratified
cross-validation, each fold's training split discretized (decision-tree
thresholds, max 6 intervals per numeric attribute) and both splits
binarized against it, written out as plain CSV (one 0/1 column per
Boolean feature, the class label last).

Negation columns are left out by default -- see NOMINAL_NEGATIONS below
for exactly what that means and how to turn it on.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pyrulearn.data.io import binarize, build_dataspec
from pyrulearn.experiments.catalog import Catalog

# ---------------------------------------------------------------- settings --

N_FOLDS = 10
MAX_INTERVALS = 6
RANDOM_STATE = 0
OUT_DIR = os.path.join(os.path.dirname(__file__), "_export_binary_cv_folds")

# A numeric attribute always gets both "attr>=t" and "attr<t" -- that
# pair isn't what this flag controls. What it does control: a nominal
# (or ordinal) attribute with more than two values (e.g. a "job" column
# with a dozen categories) gets one "attr=v" column per value, and with
# negations *also* one "attr!=v" column per value. (A binary attribute
# -- exactly two values -- never gets "!=" columns regardless, since
# "x!=a" already means "x=b" there.)
NOMINAL_NEGATIONS = False  # set True to also get "attr!=v" columns


def export_dataset(entry) -> None:
    df, target = entry.load()
    y = df[target].to_numpy()
    out_dir = os.path.join(OUT_DIR, entry.name)
    os.makedirs(out_dir, exist_ok=True)

    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE).split(np.zeros(len(y)), y)
    for fold, (train_idx, test_idx) in enumerate(folds):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # skip_unusable=True drops a column that turned constant or
        # missing within this training fold.
        spec = build_dataspec(train_df, target=target, max_intervals=MAX_INTERVALS,
                              include_negations=NOMINAL_NEGATIONS, skip_unusable=True).build()
        names = spec.feature_names

        train_X = binarize(spec, train_df).astype(int)
        test_X = binarize(spec, test_df).astype(int)

        train_out = pd.DataFrame(train_X, columns=names)
        train_out[target] = train_df[target].to_numpy()
        test_out = pd.DataFrame(test_X, columns=names)
        test_out[target] = test_df[target].to_numpy()

        train_out.to_csv(os.path.join(out_dir, f"fold{fold}_train.csv"), index=False)
        test_out.to_csv(os.path.join(out_dir, f"fold{fold}_test.csv"), index=False)

    print(f"{entry.name}: {len(names)} features, {N_FOLDS} folds -> {out_dir}", flush=True)


def main() -> None:
    datasets = Catalog.default().select(task="binary")
    print(f"{len(datasets)} two-class datasets", flush=True)
    for entry in datasets:
        try:
            export_dataset(entry)
        except Exception as e:  # noqa: BLE001 -- one bad dataset shouldn't stop the rest
            print(f"{entry.name}: SKIPPED ({type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
