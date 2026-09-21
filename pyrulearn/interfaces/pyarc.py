"""
pyrulearn.interfaces.pyarc
==============================

`PyarcCBAImporter` (an `ObjectRuleImporter`) and `PyarcCBA` (its
`ExternalRuleLearner`) for the *external* `pyarc` package's CBA -- the
same algorithm `pyrulearn.learners.cba.CBA` implements natively. The two
exist side by side on purpose: `PyarcCBA` is the reference to cross-check
the native one against (runtime and soundness; agreement on the same
input has been verified rule for rule, see `memory`/README), and a fast
drop-in when mining speed matters (`fim`, the C library `pyarc` mines
with, is roughly 10x faster than this package's pure-Python miner).

**Dependencies.** `pyarc` itself is pure Python but refuses to import
without Christian Borgelt's `fim` C extension (`pyfim`), which has no
Windows wheels; on Windows it has to be built with a compiler (MinGW
`g++` works, one missing `#include <time.h>` in `pyfim.c` needs adding
first). Importing *this* module needs neither -- `pyarc` is imported
lazily inside `PyarcCBA.fit_external` -- and `PyarcCBAImporter` only
reads attributes of an already-fitted `pyarc.CBA`.

**Input encoding.** Rows go to `pyarc` as transactions containing *only
the features that are True* (False cells are turned into NaN, which
`TransactionDB.from_DataFrame` drops), plus the class. That is exactly
this package's own item semantics -- rule bodies are conjunctions of
positive literals, negation being a separate paired feature -- so a
`DataSpec` built with negation features works unchanged and the imported
rules are again plain positive-literal `Rule`s. (Feeding `pyarc` the full
0/1 matrix instead would make it mine `f=0` items too, duplicating what
the explicit negation features already express.)

**Two conventions that differ from the native `CBA`**, both handled here
so the constructor arguments mean the same thing in both:

- `pyarc`'s `maxlen` counts the *class item* too (it is passed to
  `fim`'s `zmax`): `maxlen=L+1` mines antecedents of length <= `L`.
  `PyarcCBA(max_len=L)` therefore passes `maxlen=L+1`, matching
  `CBA(max_len=L)`.
- `pyarc` (like `fim` and CBA-RG) thresholds the support of the whole
  rule, body + head -- which `generate_cars` now does too.

Native multi-class (each rule carries its own class), so no
`DecomposingLearner`; `fit(data)` gives a `pyrulearn.models.DecisionList`
whose default prediction is `pyarc`'s own default class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import ObjectRuleImporter, register_importer
from ..data import DataRepresentation, DataSpec
from ..learners.base import ExternalRuleLearner
from ..models import DecisionList
from ..rule import Rule


def _label_lookup(data: Optional[DataRepresentation]) -> Dict[str, Any]:
    """`pyarc` only ever sees (and returns) string class labels; map them
    back to the original labels of `data.y` when it's available."""
    if data is None or data.y is None:
        return {}
    return {str(c): c for c in np.unique(np.asarray(data.y))}


def _rule_from_pyarc(car, dataspec: DataSpec, labels: Dict[str, Any]) -> Rule:
    """One `Rule` from one `pyarc.ClassAssocationRule`: antecedent items
    are `("f<i>", "1.0")` (only True features are ever transactions), the
    consequent `("class", label)`."""
    features: List[int] = []
    for attribute, _value in car.antecedent:
        idx = int(str(attribute)[1:])
        if not 0 <= idx < dataspec.n_features:
            raise ValueError(
                f"pyarc rule mentions feature f{idx} but dataspec has only {dataspec.n_features} "
                "features -- the model was fitted over a different feature space"
            )
        features.append(idx)
    label = str(car.consequent[1])
    return Rule.from_pos_neg(pos=sorted(set(features)), neg=[], target=labels.get(label, label),
                             dataspec=dataspec)


class PyarcCBAImporter(ObjectRuleImporter):
    """Extracts a `pyrulearn.models.DecisionList` from a fitted
    `pyarc.CBA` -- one `Rule` per rule of `model.clf.rules` **in the
    classifier's own order** (first match wins, as `pyarc` predicts), plus
    `model.clf.default_class` as the list's default prediction. See the
    module docstring for the input encoding the model must have been
    fitted on (feature `i` is the attribute named ``f{i}``).
    """

    SOURCE = "pyarc.CBA"

    def import_model(self, model, dataspec: DataSpec, data: Optional[DataRepresentation] = None) -> DecisionList:
        clf = getattr(model, "clf", None)
        if clf is None:
            raise ValueError("model is not fitted -- call pyarc.CBA.fit first")
        labels = _label_lookup(data)
        rules = [_rule_from_pyarc(car, dataspec, labels) for car in clf.rules]
        default = clf.default_class
        default = labels.get(str(default), default)

        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)
        rule_list = DecisionList(rules, default_prediction=default)
        if default is not None:
            self._stamp_rule_provenance([rule_list.default_rule], n_rules=len(rules))
        return self._stamp_provenance(rule_list, n_rules=len(rules))


register_importer("pyarc_cba", PyarcCBAImporter)


class PyarcCBA(ExternalRuleLearner):
    """`pyarc.CBA`, fitted on a `DataRepresentation` and read back as a
    `pyrulearn.models.DecisionList`.

    - `min_support` / `min_confidence` -- fractions in `[0, 1]`; support is
      the support of body + head (see the module docstring).
    - `max_len` -- longest rule *body* (converted to `pyarc`'s
      class-item-inclusive `maxlen` internally), same meaning as
      `pyrulearn.learners.cba.CBA(max_len=)`.
    - `algorithm` -- `"m1"` (default) or `"m2"`, `pyarc`'s two
      classifier-building variants.

    Defaults mirror the native `CBA`'s (`0.01`, `0.5`, `4`) so the two are
    interchangeable in a comparison.
    """

    IMPORTER = PyarcCBAImporter
    NATIVE_MODEL = DecisionList

    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5, max_len: int = 4,
                 algorithm: str = "m1"):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len
        self.algorithm = algorithm

    def fit_external(self, X, y, feature_names: Optional[Sequence[str]] = None):
        try:
            import pandas as pd
            from pyarc import CBA as _CBA
            from pyarc.data_structures import TransactionDB
        except ImportError as e:  # pyarc raises a bare Exception when fim is missing, see below
            raise ImportError(f"PyarcCBA needs pandas and pyarc (which needs Borgelt's fim): {e}") from e
        except Exception as e:
            raise ImportError(f"PyarcCBA needs pyarc (which needs Borgelt's pyfim): {e}") from e

        if y is None:
            raise ValueError("PyarcCBA needs labels (y) to fit")
        X = np.asarray(X, dtype=bool)
        frame = pd.DataFrame(np.where(X, 1.0, np.nan), columns=[f"f{i}" for i in range(X.shape[1])])
        frame["class"] = np.asarray(y).astype(str)
        transactions = TransactionDB.from_DataFrame(frame, target="class")
        return _CBA(support=self.min_support, confidence=self.min_confidence,
                    maxlen=self.max_len + 1, algorithm=self.algorithm).fit(transactions)
