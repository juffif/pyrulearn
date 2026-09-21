"""
pyrulearn.data
=================

Everything about representing data, split by concern:

- `pyrulearn.data.spec` -- `DataSpec`, a pure feature-space specification
  (names, typed attributes, constraints, missing-value policy), no data.
- `pyrulearn.data.representation` -- `DataRepresentation` (ABC) and its
  three base representations, the actual data bound to a `DataSpec`:
  `BooleanDataRepresentation` (bit-packed matrix), `SparseDataRepresentation`
  (scipy CSR/CSC) and `NListRepresentation` (PPC-tree / N-list index;
  `PrePostNListRepresentation` is an opt-in variant). All implement the
  same `coverage(rule)` interface, so every rule learner runs on any of them.
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
