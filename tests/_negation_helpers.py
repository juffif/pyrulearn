"""Shared helpers for tests that need explicit-negation Boolean dataspecs.

Since the big redesign, negative literals no longer exist -- a rule condition
that means "feature f is absent" is expressed as a separate negation feature
paired with ``f`` by a ``MutuallyExclusive`` constraint. These helpers build
such specs and the matching (doubled) coverage matrices so existing tests that
were written against ``Rule.from_pos_neg(neg=...)`` keep working.
"""

import numpy as np

from pyrulearn.data import DataSpecBuilder
from pyrulearn.rule import Rule


def neg_spec(names):
    """A DataSpec with, for every name, a positive feature and its negation.

    Feature layout is interleaved: ``names[i]`` is feature ``2*i`` and
    ``not names[i]`` is feature ``2*i + 1``.
    """
    builder = DataSpecBuilder(negation=True)
    for name in names:
        builder.add_boolean(name)
    return builder.build()


def neg_X(rows):
    """Expand an (n, k) 0/1 matrix into the (n, 2k) positive/negation matrix
    that matches :func:`neg_spec` (column ``2i`` = value, ``2i+1`` = its
    complement)."""
    base = np.asarray(rows, dtype=bool)
    full = np.empty((base.shape[0], 2 * base.shape[1]), dtype=bool)
    full[:, 0::2] = base
    full[:, 1::2] = ~base
    return full


def make_rule(dataspec, pos=(), neg=(), **kwargs):
    """``Rule.from_pos_neg`` addressed by feature name against a :func:`neg_spec`."""
    return Rule.from_pos_neg(
        pos=[dataspec.feature_index(n) for n in pos],
        neg=[dataspec.feature_index(n) for n in neg],
        dataspec=dataspec,
        **kwargs,
    )
