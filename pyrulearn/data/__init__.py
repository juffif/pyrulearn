"""
pyrulearn.data
=================

Everything about representing data, split by concern:

- `pyrulearn.data.spec` -- `DataSpec`, a pure feature-space specification
  (names, typed attributes, constraints, missing-value policy), no data.
- `pyrulearn.data.representation` -- `DataRepresentation` (ABC) /
  `BooleanDataRepresentation` / `NListRepresentation` /
  `PrePostNListRepresentation` / `SparseDataRepresentation`, the actual
  data bound to a `DataSpec`.
- `pyrulearn.data.io` -- `read_arff`/`read_csv`/`write_arff`/`write_csv`/
  `binarize`/`build_dataspec`, reading/writing a `BooleanDataRepresentation`
  from/to ARFF or CSV. Requires `pandas` -- **not** re-exported here, so a
  bare `import pyrulearn` (or `import pyrulearn.data`) still doesn't need
  it; `import pyrulearn.data.io` (or the functions it's actually used
  through) does.

Both `spec` and `representation` are re-exported here, so
``from pyrulearn.data import DataSpec, BooleanDataRepresentation`` keeps
working as a single import.
"""

from __future__ import annotations

from .spec import DataSpec, DataSpecBuilder, merge_dataspecs
from .representation import (
    BooleanDataRepresentation,
    DataRepresentation,
    NListRepresentation,
    PrePostNListRepresentation,
    SparseDataRepresentation,
)

__all__ = [
    "DataSpec", "DataSpecBuilder", "merge_dataspecs",
    "DataRepresentation", "BooleanDataRepresentation",
    "NListRepresentation", "PrePostNListRepresentation", "SparseDataRepresentation",
]
