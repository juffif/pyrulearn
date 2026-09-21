"""
pyrulearn.data.representation
=================================

`DataRepresentation`: a concrete encoding of a dataset bound to a
`pyrulearn.data.spec.DataSpec`. `DataSpec` itself is pure schema (feature
names/order, typed attributes, constraints, missing-value policy) -- no
data; a `DataRepresentation` is where the actual values live, always
pointing back at the `DataSpec` it was built against.

`BooleanDataRepresentation` is the canonical one: a `numpy.packbits`-
packed Boolean matrix (plus labels). `NListRepresentation` is the same
data as a PPC-tree / N-list vertical index (the one Huynh &
Fürnkranz's LORD builds); `SparseDataRepresentation` is a `scipy`
CSR/CSC encoding -- the N-list without the prefix tree. All three sit
behind the identical interface, so every `pyrulearn.learners.seco` learner runs
on any of them. All point at an ordinary `DataSpec`; further encodings
some future algorithm wants directly are further `DataRepresentation`
subclasses.

Every representation implements the primitives the rule-learning code
is written against: `coverage(rule)` (the boolean covered-rows vector)
and `features_of(row)` (a seed example's True features) for one-shot
use outside a search (prediction, pruning, `RuleStats.from_rule` --
`Rule.covers_data`/`Rule.covers_data_packed` just forward to
`data.coverage(self)`); and, for `pyrulearn.learners.seco.BeamSearch`/
`HillClimbing`'s incremental search fast path, `initial_cover`,
`refine_cover`, `cover_counts`, and `cover_rows`, built around an
opaque, representation-specific "cover handle" so each encoding can
make refining a rule by one more literal as cheap as its own storage
allows (see `NListRepresentation`'s own docstring for the interesting
case).

Because a `DataSpec` no longer carries any data of its own, the same
`DataSpec` instance can be shared verbatim across multiple
`DataRepresentation`s built from it (e.g. a train split and a test
split), guaranteeing identical feature indexing between them by
construction rather than by convention.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import numpy as np

from .spec import DataSpec

if TYPE_CHECKING:
    from ..rule import Rule


class DataRepresentation(ABC):
    """Base for a concrete encoding of a dataset bound to a `DataSpec`.

    Two abstract primitives make every representation a drop-in for the
    rule-learning machinery, regardless of how it stores the data
    internally:

    - `coverage(rule)` -- the boolean "which rows does this rule cover"
      vector every heuristic, search and covering loop is ultimately
      defined over (`RuleStats.from_rule` does the ``& example_mask &
      (y == class)`` counting on top of it). `Rule.covers_data` /
      `Rule.covers_data_packed` just forward here.
    - `features_of(row)` -- the feature indices that are True in one row,
      needed to seed a search from an example (`pyrulearn.learners.seco.
      SeedExample`).
    """

    def __init__(self, spec: DataSpec):
        self.spec = spec
        self.y: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def n_samples(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def coverage(self, rule: "Rule") -> np.ndarray:
        """Boolean array of shape ``(n_samples,)`` -- ``[i]`` True iff
        every one of `rule`'s conditions holds for row ``i``."""
        raise NotImplementedError

    @abstractmethod
    def features_of(self, row: int) -> np.ndarray:
        """The integer feature indices that are True in `row` (ascending)
        -- the "open" features a rule seeded on that row can still add."""
        raise NotImplementedError

    # -- incremental coverage (search fast path) -----------------------

    @abstractmethod
    def initial_cover(self, example_mask: Optional[np.ndarray] = None) -> Any:
        """A representation-specific, opaque "cover handle" for the empty
        rule, restricted to `example_mask` if given. Masking a search's
        scope (a SeCo covering loop's shrinking "remaining" set, an
        IREP/RIPPER grow/prune split) is just the zeroth refinement --
        applying it once here, rather than re-applying it on every
        downstream `cover_counts` call the way `RuleStats.from_rule`'s
        `example_mask` does, is what lets a representation that keeps a
        compressed handle (`NListRepresentation`) narrow its own index
        once instead of masking a full row vector at every search node.
        """
        raise NotImplementedError

    @abstractmethod
    def refine_cover(self, handle: Any, feature: int) -> Any:
        """The handle for `handle`'s rule with one more literal
        (`feature`) added -- i.e. `handle` narrowed to rows/nodes where
        `feature` also holds. `pyrulearn.learners.seco.BeamSearch`/`HillClimbing`
        call this once per specialization step instead of recomputing
        `coverage` for the whole grown rule from scratch, so each
        representation can make refining as cheap as its own storage
        allows (see each subclass's implementation)."""
        raise NotImplementedError

    @abstractmethod
    def cover_counts(self, handle: Any, positive_class: Any) -> Tuple[int, int, int, int]:
        """``(tp, fp, fn, tn)`` for the rows covered by `handle`, against
        `self.y`, treating `positive_class` as positive. A plain tuple,
        not a `RuleStats` -- `length` is rule/search context, not
        something a coverage handle has, and `RuleStats` lives in
        `pyrulearn.heuristics`, which imports this module (a
        `RuleStats`-returning method here would be a circular import)."""
        raise NotImplementedError

    @abstractmethod
    def cover_rows(self, handle: Any) -> np.ndarray:
        """Boolean array of shape ``(n_samples,)``: which rows `handle`
        covers -- `coverage(rule)` restated for a handle instead of a
        `Rule`. Materializing the full array is usually only worth doing
        once a search has settled on a final handle, not at every node,
        which is why this is a separate call from `cover_counts` rather
        than something every refinement step pays for."""
        raise NotImplementedError

    # -- negation-feature toggling -----------------------------------

    def without_negations(self) -> "DataRepresentation":
        """A representation of the same concrete type over this data's
        *positive-only* feature space -- every ``not f`` / ``x != v`` /
        ``x < t`` / ``x > t`` feature dropped, via
        `DataSpec.without_negations`. Returns `self` unchanged if there
        are no negation features. No data is recomputed: the surviving
        columns are exactly the columns that were already there."""
        new_spec, keep = self.spec.without_negations()
        if new_spec is self.spec:
            return self
        return self._reindexed(new_spec, keep, None)

    def with_negations(self) -> "DataRepresentation":
        """A representation of the same concrete type with a negation
        feature added for every test that lacks one, via
        `DataSpec.with_negations`. The added columns are synthesized as
        the logical complement of their positive partner -- see that
        method for the (missing-value) assumption that carries. Returns
        `self` unchanged if every feature already has its negation.

        For `SparseDataRepresentation` / `NListRepresentation` the added
        columns are near-50%-dense, so the result is much less compact
        than the input -- correct, but usually not what you want; prefer
        building the negated version from the raw data."""
        new_spec, source, is_complement = self.spec.with_negations()
        if new_spec.n_features == self.spec.n_features and not is_complement.any():
            return self
        return self._reindexed(new_spec, source, is_complement)

    def _reindexed(
        self, new_spec: DataSpec, source: np.ndarray, is_complement: Optional[np.ndarray]
    ) -> "DataRepresentation":
        """Build a representation of the same concrete type over
        `new_spec`, where new column ``j`` is old column ``source[j]``,
        logically negated where ``is_complement[j]`` (``is_complement is
        None`` -> plain column selection, no negation). Subclasses
        override with a storage-appropriate implementation; this fallback
        goes through the dense `X`."""
        cols = self.X[:, source]
        if is_complement is not None:
            cols = np.where(is_complement, ~cols, cols)
        return type(self)(new_spec, cols, self.y)

    # -- row subsetting ---------------------------------------------------

    def select_rows(self, mask: np.ndarray) -> "DataRepresentation":
        """A representation of the same concrete type over just the rows
        `mask` selects (a boolean mask or an integer index array),
        rebuilt from scratch -- `NListRepresentation` re-derives its
        tree, etc. The `DataSpec` (feature space, so rule feature indices)
        is unchanged, so rules learned on the subset still apply to the
        full data. Used to train one stage of a class-ordered
        decomposition on just the classes not yet peeled off."""
        y = None if self.y is None else self.y[mask]
        return type(self)(self.spec, self.X[mask], y)

    def relabel(self, new_y: Any) -> "DataRepresentation":
        """Same rows and feature space, a different `y` -- e.g.
        ``data.relabel(np.where(data.y == "a", "a", "rest"))`` for a
        one-vs-rest sub-problem given to a learner that can't fold the
        rest in itself (external binary learners)."""
        return type(self)(self.spec, self.X, np.asarray(new_y))


class BooleanDataRepresentation(DataRepresentation):
    """The canonical Boolean bit-matrix encoding: a `numpy.packbits`-packed
    feature matrix (shape ``(n_samples, ceil(spec.n_features / 8))``) plus
    optional labels `y`. `coverage`/`Rule.covers_data_packed` run against
    the packed form (`packed_rows()`) without ever touching `X`; the
    unpacked ``(n_samples, spec.n_features)`` bool matrix `X` is
    reconstructed lazily on first access (see the property) -- for handing
    a plain array to external learners, and (once) by `refine_cover`'s
    incremental search fast path.
    """

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None):
        super().__init__(spec)
        try:
            from scipy.sparse import issparse
        except ImportError:  # pragma: no cover
            issparse = lambda _o: False  # noqa: E731
        if issparse(X):
            X = X.toarray()
        X = np.asarray(X)
        if X.dtype != bool:
            X = X.astype(bool)
        if X.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        self._n_samples: int = X.shape[0]
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._packed: np.ndarray = np.packbits(X, axis=1)

    @functools.cached_property
    def X(self) -> np.ndarray:
        """The unpacked ``(n_samples, n_features)`` bool matrix, rebuilt
        from the packed cache on first access and held thereafter.
        `coverage`/`Rule.covers_data_packed` use `packed_rows()` and never
        trigger this; it's here for handing a plain array to external
        learners (sklearn / wittgenstein / imodels), for `Rule.covers_data`,
        and for `refine_cover`'s per-column access."""
        n_features = self.spec.n_features
        return np.unpackbits(self._packed, axis=1)[:, :n_features].astype(bool)

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def packed_rows(self) -> np.ndarray:
        return self._packed

    def coverage(self, rule: "Rule") -> np.ndarray:
        """Vectorized bitmask subset check across all rows, against the
        `numpy.packbits` cache -- no per-feature Python loop over the
        data, and `X` is never materialized."""
        n_bytes = self._packed.shape[1]
        n_features = self.spec.n_features
        want_full = np.zeros(n_features, dtype=bool)
        for lit in rule.conditions:
            want_full[lit.feature] = True
        want = np.pad(np.packbits(want_full), (0, n_bytes - (n_features + 7) // 8))
        return np.all((self._packed & want) == want, axis=1)

    def features_of(self, row: int) -> np.ndarray:
        return np.flatnonzero(self.X[row])

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (cov, scope): `cov` is the current rule's coverage,
    # already ANDed with the original `example_mask` (so it's zero
    # outside it from the very first refinement on); `scope` is that
    # same mask (or `None`), carried through unchanged. `scope` has to be
    # tracked separately from `cov` -- `~cov` alone is True for *every*
    # out-of-scope row regardless of the rule, which would wrongly count
    # them as false negatives/negatives without it (see `cover_counts`).

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        if example_mask is None:
            return np.ones(self._n_samples, dtype=bool), None
        mask = np.array(example_mask, dtype=bool, copy=True)
        return mask, mask

    def refine_cover(self, handle, feature: int):
        cov, scope = handle
        # `X` materializes here on first use -- a one-time O(n*k) unpack,
        # paid back many times over versus `coverage`'s O(n*k/8) *every*
        # call regardless of how short the rule being grown is.
        return cov & self.X[:, feature], scope

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        cov, scope = handle
        pos_mask = self.y == positive_class
        in_scope = np.ones(self._n_samples, dtype=bool) if scope is None else scope
        tp = int(np.sum(cov & pos_mask))
        fp = int(np.sum(cov & ~pos_mask))
        fn = int(np.sum(~cov & pos_mask & in_scope))
        tn = int(np.sum(~cov & ~pos_mask & in_scope))
        return tp, fp, fn, tn

    def cover_rows(self, handle) -> np.ndarray:
        cov, _ = handle
        return cov

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "BooleanDataRepresentation":
        """Build from a pandas DataFrame of 0/1 (or bool) feature columns.
        `spec` binds against an existing `DataSpec` (e.g. a training
        split's, when building a test representation); if omitted, one
        is built fresh from `df`'s (non-label) columns."""
        if label_col is not None:
            y = df[label_col].to_numpy()
            feat_df = df.drop(columns=[label_col])
        else:
            y = None
            feat_df = df
        if spec is None:
            spec = DataSpec(list(feat_df.columns), name=name)
        return cls(spec, feat_df.to_numpy(dtype=bool), y)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "BooleanDataRepresentation":
        """Build from a `scipy.sparse` matrix already matching `spec`'s
        feature order -- densified into the packed bit-matrix. (For a
        genuinely sparse dataset, `SparseDataRepresentation.from_scipy`
        keeps it sparse.)"""
        return cls(spec, matrix, y)

    @classmethod
    def from_xy(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "BooleanDataRepresentation":
        """Build from an array-like ``X`` (plain numpy, not necessarily
        already Boolean) -- the ordinary ``X, y`` convention shared by
        scikit-learn and most other array-based ML tooling.

        If ``X`` is already 0/1-valued, it's used as-is. Otherwise pass
        ``threshold`` to binarize continuous features as ``X > threshold``
        (a single scalar, or an array broadcastable to a feature-wise
        threshold, e.g. per-feature medians). `spec` binds against an
        existing `DataSpec`; if omitted, one is built fresh, with
        `feature_names` defaulting to ``f0, f1, ...`` if not given.
        """
        X = np.asarray(X)
        Xb = (X > threshold) if threshold is not None else X.astype(bool)
        if spec is None:
            names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(Xb.shape[1])]
            spec = DataSpec(names, name=name)
        return cls(spec, Xb, y)

    def __repr__(self) -> str:
        return f"BooleanDataRepresentation(spec={self.spec!r}, n_samples={self.n_samples})"


class _PPCNode:
    """A node of the PPC- (pre/post-code) prefix tree -- an FP-tree-like
    trie over each row's set of True features, ordered most-frequent
    first so common prefixes are shared. Only what
    `NListRepresentation` needs during construction lives here; the
    finished index is flat numpy arrays, not a graph of these.

    No pre/post visit numbers on the node itself -- `PrePostNListRepresentation`
    (see its own docstring, and the comment above `NListRepresentation.
    refine_cover`) derives them in a side dict instead, precisely so this
    class -- shared by both -- doesn't carry that cost for the common
    case that doesn't use them."""

    __slots__ = ("item", "children", "rows")

    def __init__(self, item: int):
        self.item = item          # feature index this node stands for (-1 = root)
        self.children: dict = {}   # feature index -> _PPCNode
        self.rows: list = []       # row indices whose path passes through here


class _NListMaskContext:
    """Per-search state shared, unchanged, by every handle
    `NListRepresentation.initial_cover`/`refine_cover` produce for one
    search: the `example_mask` (or `None`) plus a lazily-filled cache of
    per-item masked node class-counts. Every handle from the same
    `initial_cover` call carries a reference to the *same* instance, so
    an item that becomes the anchor for several different candidates
    within that one search only pays the mask-intersection cost the
    first time -- the cache is purely additive (never invalidated,
    never wrong to reuse), since the mask itself never changes within
    one search."""

    __slots__ = ("mask", "_node_counts", "_scope_totals")

    def __init__(self, mask: Optional[np.ndarray]):
        self.mask = mask
        self._node_counts: dict = {}
        self._scope_totals: Optional[np.ndarray] = None

    def scope_totals(self, rep: "NListRepresentation") -> np.ndarray:
        """Per-class row counts within this context's scope (all rows,
        or just the masked ones) -- computed once, reused for every
        `cover_counts` call in this search."""
        if self.mask is None:
            return rep._class_totals
        if self._scope_totals is None:
            self._scope_totals = np.bincount(rep._y_idx[self.mask], minlength=rep._n_classes)
        return self._scope_totals

    def node_counts(self, rep: "NListRepresentation", item: int) -> np.ndarray:
        """Per-node, per-class counts for `item`, restricted to this
        context's mask -- `rep`'s own precomputed (unmasked) counts if
        there's no mask, otherwise derived once (by intersecting that
        item's row list with the mask) and cached."""
        if self.mask is None:
            return rep._node_class_counts[item]
        cached = self._node_counts.get(item)
        if cached is None:
            m = rep._path_words[item].shape[0]
            counts = np.zeros((m, rep._n_classes), dtype=np.int64)
            row_idx, entry_node = rep._row_idx[item], rep._entry_node[item]
            if row_idx.size:
                in_mask = self.mask[row_idx]
                np.add.at(counts, (entry_node[in_mask], rep._y_idx[row_idx[in_mask]]), 1)
            cached = counts
            self._node_counts[item] = cached
        return cached


class NListRepresentation(DataRepresentation):
    """A PPC-tree / N-list encoding of a Boolean dataset -- the vertical
    index Huynh & Fürnkranz's LORD builds before learning -- exposed
    behind the same `DataRepresentation` interface as
    `BooleanDataRepresentation`, so every `pyrulearn.learners.seco` learner (CN2,
    AQR, `pyrulearn.learners.pylord.PyLORD`, ...) runs on it unchanged.

    How it works
    ------------
    Each row is the set of features that are True in it. Sort those
    features most-frequent-first and insert the resulting path into a
    trie (`_PPCNode`); rows with a shared prefix share tree nodes. Every
    node records the exact set of row indices that pass through it
    (`rows`) and -- derived once, up front -- the bitmask of every
    feature on its root-to-here path (`path_mask`).

    A rule ``f1 ∧ ... ∧ fm`` covers exactly the rows whose path contains
    a node for every ``fi``. Equivalently: take ``iy``, the deepest
    (least frequent) of the ``fi``; among the nodes registering ``iy``,
    keep those whose ``path_mask`` is a superset of the rule's feature
    set; the rule's coverage is the union of those nodes' row sets. That
    superset test is a vectorized ``uint64`` word-AND over the item's
    N-list -- `coverage` never materializes an ``(n_rows, n_features)``
    matrix and never falls back to a packed bit-matrix.

    `coverage`/`features_of` (used outside a search: prediction, pruning,
    `RuleStats.from_rule`) always recompute from scratch this way.
    `pyrulearn.learners.seco.BeamSearch`/`HillClimbing` instead go through
    `initial_cover`/`refine_cover`/`cover_counts` -- see their own
    docstrings just below `_build` for the real LORD-style incremental
    scheme: a handle tracks the deepest feature fixed so far (its
    "anchor") and the still-shrinking set of that item's nodes still
    satisfying the rule, so refining by a feature no deeper than the
    anchor is just a filter over the *current* (already narrow) set --
    cheaper the further a search has already gone -- and only refining
    by a strictly deeper feature needs a fresh look at that item's own
    N-list (see the comment above `refine_cover` for why that fresh look
    stays a plain bitmask scan here rather than a pre/post-code shortcut
    -- tried, measured, and kept as the opt-in `PrePostNListRepresentation`
    subclass instead, not folded in here). `example_mask` (a
    grow/prune split, a SeCo loop's shrinking "remaining" set) is handled
    the same way as masking is everywhere in this module: applied once,
    in `initial_cover`, not re-applied on every downstream call --
    `_NListMaskContext` also memoizes the masked per-item node counts it
    derives, shared by every handle from one search, so an item that
    anchors several different candidates only pays that cost once.

    Negation hurts this representation too, for a different reason than
    `SparseDataRepresentation`: pairing every feature with its negation
    doubles each row's path length (every original feature *or* its
    negation is now True, so both occupy a tree position), which lengthens
    every root-to-leaf path without adding information -- fewer rows end
    up sharing a full prefix, so the tree gets more nodes, not fewer, for
    the same underlying data (e.g. one demo dataset: 6863 tree nodes with
    negation vs. 2415 without, same rows). `without_negations()` before
    building (or on an already-negated representation) is the fix here
    too.

    Build it from an existing `BooleanDataRepresentation` with
    `from_boolean`, or from raw arrays via the constructor / `from_xy` /
    `from_dataframe`, mirroring `BooleanDataRepresentation`.
    """

    def __init__(self, spec: DataSpec, X: np.ndarray, y: Optional[np.ndarray] = None):
        super().__init__(spec)
        X = np.asarray(X)
        if X.dtype != bool:
            X = X.astype(bool)
        if X.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        self._n_samples: int = X.shape[0]
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._build(X)

    # -- construction ---------------------------------------------------

    def _build(self, X: np.ndarray) -> "_PPCNode":
        """Builds the trie and every per-item array this class needs,
        and returns the finished root node -- not kept as `self._root`,
        since nothing here needs the node graph once the flat arrays
        exist; a subclass that does (e.g. to derive more structure from
        it, via `_prepare_extra_per_item`) gets it as the return value
        instead, rather than paying to retain it unconditionally."""
        n, k = X.shape
        self._n_features = k
        self._word_count = (k + 63) // 64 or 1

        # most-frequent feature first -> shared prefixes near the root
        freq = X.sum(axis=0)
        order = np.argsort(-freq, kind="stable")
        rank = np.empty(k, dtype=np.int64)
        rank[order] = np.arange(k)
        self._rank = rank

        root = _PPCNode(-1)
        row_feats: list = [None] * n          # per row: its features, ascending
        leaf_of: list = [root] * n            # per row: the node its path ends at
        for i in range(n):
            feats = np.flatnonzero(X[i])
            row_feats[i] = feats
            node = root
            for f in sorted((int(v) for v in feats), key=lambda v: rank[v]):
                child = node.children.get(f)
                if child is None:
                    child = _PPCNode(f)
                    node.children[f] = child
                child.rows.append(i)
                node = child
            leaf_of[i] = node
        self._row_feats = row_feats
        self._leaf_of = leaf_of

        # extension point (see PrePostNListRepresentation): tree-wide
        # preprocessing that needs the finished trie but runs before the
        # per-item traversal below -- e.g. assigning pre/post visit codes,
        # and setting up per-item accumulator lists for `_extra_per_item`
        # to fill in during that same traversal (guaranteeing they end up
        # in the identical per-item node order as `_path_words` etc.,
        # rather than depending on two separate traversals staying in
        # sync). No-op here.
        self._prepare_extra_per_item(root, k)

        # one traversal: per item, the N-list of (path_mask, row set)
        masks_by_item: list = [[] for _ in range(k)]
        rows_by_item: list = [[] for _ in range(k)]
        stack = [(root, 0)]
        while stack:
            node, pmask = stack.pop()
            for child in node.children.values():
                cmask = pmask | (1 << child.item)
                masks_by_item[child.item].append(cmask)
                rows_by_item[child.item].append(child.rows)
                self._extra_per_item(child, cmask)
                stack.append((child, cmask))

        W = self._word_count
        empty_rows = np.empty(0, dtype=np.int64)
        self._path_words: list = []   # per item: uint64 (n_nodes_i, W)
        self._row_idx: list = []      # per item: int64  (total_i,)  rows, node-grouped
        self._entry_node: list = []   # per item: int64  (total_i,)  node id per row entry
        for it in range(k):
            masks = masks_by_item[it]
            m = len(masks)
            words = np.zeros((m, W), dtype=np.uint64)
            for ni, mm in enumerate(masks):
                wj = 0
                while mm:
                    words[ni, wj] = np.uint64(mm & 0xFFFFFFFFFFFFFFFF)
                    mm >>= 64
                    wj += 1
            self._path_words.append(words)
            node_rows = rows_by_item[it]
            if node_rows:
                self._row_idx.append(
                    np.concatenate([np.asarray(r, dtype=np.int64) for r in node_rows])
                )
                self._entry_node.append(
                    np.concatenate([
                        np.full(len(r), ni, dtype=np.int64) for ni, r in enumerate(node_rows)
                    ])
                )
            else:
                self._row_idx.append(empty_rows)
                self._entry_node.append(empty_rows)

        self._finalize_extra_per_item(k)  # extension point -- no-op here

        # per-node, per-class counts -- built once so an *unmasked*
        # refine_cover's stats are a sum over the (already-shrinking)
        # active node set, never touching a row-id array at all. Under a
        # mask, `_NListMaskContext` derives the masked equivalent lazily,
        # once per (item, search) -- see `refine_cover`/`cover_counts`.
        if self.y is not None:
            classes, y_idx = np.unique(self.y, return_inverse=True)
            self._classes = classes
            self._class_index = {c: i for i, c in enumerate(classes)}
            self._y_idx = y_idx.astype(np.int64)
            self._n_classes = len(classes)
            self._class_totals = np.bincount(self._y_idx, minlength=self._n_classes)
            self._node_class_counts: list = []
            for it in range(k):
                m = self._path_words[it].shape[0]
                counts = np.zeros((m, self._n_classes), dtype=np.int64)
                entry_node = self._entry_node[it]
                if entry_node.size:
                    np.add.at(counts, (entry_node, self._y_idx[self._row_idx[it]]), 1)
                self._node_class_counts.append(counts)
        else:
            self._classes = self._class_index = self._y_idx = None
            self._n_classes = self._class_totals = self._node_class_counts = None

        return root

    def _prepare_extra_per_item(self, root: "_PPCNode", k: int) -> None:
        """Extension hook, called once, right after the trie is built and
        before the per-item traversal below -- for tree-wide
        preprocessing (e.g. `PrePostNListRepresentation` assigning pre/
        post visit codes) and setting up whatever per-item accumulator
        lists `_extra_per_item` will fill in. No-op here."""

    def _extra_per_item(self, node: "_PPCNode", path_mask: int) -> None:
        """Extension hook, called once per registered tree node, at
        exactly the point `masks_by_item[node.item]`/`rows_by_item[
        node.item]` are appended to -- so a subclass's own per-item
        accumulator lists end up in the identical node order as
        `_path_words`/`_row_idx`/`_entry_node`, by construction rather
        than by two traversals happening to agree. No-op here."""

    def _finalize_extra_per_item(self, k: int) -> None:
        """Extension hook, called once the per-item arrays this class
        keeps (`_path_words` etc.) are finished -- for a subclass to
        convert its own per-item accumulator lists (built up via
        `_extra_per_item`) into its final form. No-op here."""

    # -- the two DataRepresentation primitives -------------------------

    def _feature_words(self, features: Sequence[int]) -> np.ndarray:
        w = np.zeros(self._word_count, dtype=np.uint64)
        for f in features:
            w[f >> 6] |= np.uint64(1 << (int(f) & 63))
        return w

    def _coverage_of_features(self, feats: Sequence[int]) -> np.ndarray:
        """The from-scratch N-list computation `coverage` and (for now)
        `cover_counts`/`cover_rows` share: look only at the *deepest*
        feature's own N-list and keep the nodes whose full root-to-node
        path contains every one of `feats`. `coverage(rule)` is just this
        applied to `rule`'s conditions."""
        feats = list(feats)
        out = np.zeros(self._n_samples, dtype=bool)
        if not feats:
            out[:] = True
            return out
        iy = max(feats, key=lambda f: self._rank[f])   # deepest item in the tree
        words = self._path_words[iy]
        if words.shape[0] == 0:
            return out
        want = self._feature_words(feats)
        hit = np.all((words & want) == want, axis=1)     # nodes whose path has every fi
        if not hit.any():
            return out
        sel = hit[self._entry_node[iy]]
        out[self._row_idx[iy][sel]] = True
        return out

    def coverage(self, rule) -> np.ndarray:
        return self._coverage_of_features([int(l.feature) for l in rule.conditions])

    def features_of(self, row: int) -> np.ndarray:
        return np.sort(np.asarray(self._row_feats[row], dtype=np.int64))

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (anchor_item, active, mask_words, ctx):
    #
    # - `anchor_item` is the *deepest* (least frequent) feature fixed so
    #   far, or `None` for the empty rule. `active` indexes into that
    #   item's own N-list (`_path_words[anchor_item]` etc.) -- the nodes
    #   whose path already satisfies every feature fixed so far.
    # - Refining by a feature no deeper than the anchor only needs to
    #   test the *new* bit against the already-narrow `active` set --
    #   every node still in it already satisfies every earlier bit, so
    #   there's nothing to recheck. `active` only ever shrinks, so this
    #   gets cheaper as the rule grows, the actual LORD-style win.
    # - Refining by a feature *deeper* than the anchor re-anchors: no
    #   node registered under the old anchor can tell us anything about
    #   an item that only appears *below* it, so this re-scans the new
    #   item's own global N-list, filtered by the full accumulated mask.
    #   Exactly `coverage`'s cost for that one item, never worse, and it
    #   only happens when a strictly deeper feature is introduced.
    #
    #   Pre/post visit codes would, *in principle*, make this cheaper
    #   still: `active`'s nodes -- no two of the same item can be
    #   ancestor/descendant of each other -- occupy disjoint subtree
    #   intervals, so binary-searching the new item's pre-sorted
    #   occurrences against those (few) intervals only touches
    #   occurrences that could possibly matter, instead of the new
    #   item's *entire* N-list. This was built, correctness-tested
    #   exhaustively, then measured, and kept out of *this* class: at the
    #   data scales this library actually deals with (thousands of rows,
    #   hundreds of features), the win is real in elements-touched but
    #   not in wall-clock time -- the interval-join path needs roughly
    #   4-5x as many separate vectorized numpy calls as the plain bitmask
    #   scan, and each call's own fixed dispatch overhead is on the same
    #   order as the work being saved, so the measured net effect across
    #   realistic feature orderings is roughly break-even to slightly
    #   negative -- including on data deliberately built to favor it. See
    #   `PrePostNListRepresentation`, below, for that version -- kept as
    #   a separate, explicitly opt-in subclass (not a flag here) for the
    #   regime it hasn't been measured on: a single feature's N-list
    #   running into the tens of thousands while `active` stays tiny.
    # - `mask_words` is the accumulated feature bitmask (needed for the
    #   re-anchor case's full-mask test). `ctx` (`_NListMaskContext`)
    #   carries `example_mask`, shared unchanged by every handle from one
    #   `initial_cover` call, memoizing per-item masked node counts so an
    #   item that becomes the anchor for several candidates in the same
    #   search only pays the mask/count derivation once.

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        mask = None if example_mask is None else np.array(example_mask, dtype=bool, copy=True)
        return None, None, np.zeros(self._word_count, dtype=np.uint64), _NListMaskContext(mask)

    def refine_cover(self, handle, feature: int):
        anchor_item, active, mask_words, ctx = handle
        feature = int(feature)
        new_mask_words = mask_words.copy()
        new_mask_words[feature >> 6] |= np.uint64(1 << (feature & 63))

        if anchor_item is None or self._rank[feature] > self._rank[anchor_item]:
            words = self._path_words[feature]
            hit = np.all((words & new_mask_words) == new_mask_words, axis=1)
            return feature, np.flatnonzero(hit), new_mask_words, ctx

        wj, bit = feature >> 6, np.uint64(1 << (feature & 63))
        keep = (self._path_words[anchor_item][active, wj] & bit) == bit
        return anchor_item, active[keep], new_mask_words, ctx

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        anchor_item, active, _, ctx = handle
        pos_idx = self._class_index[positive_class]
        scope_totals = ctx.scope_totals(self)
        n_pos = int(scope_totals[pos_idx])
        n_neg = int(scope_totals.sum()) - n_pos

        if anchor_item is None:
            tp, fp = n_pos, n_neg  # the empty rule covers everything in scope
        else:
            active_counts = ctx.node_counts(self, anchor_item)[active].sum(axis=0)
            tp = int(active_counts[pos_idx])
            fp = int(active_counts.sum()) - tp
        return tp, fp, n_pos - tp, n_neg - fp

    def cover_rows(self, handle) -> np.ndarray:
        anchor_item, active, _, ctx = handle
        if anchor_item is None:
            return np.ones(self._n_samples, dtype=bool) if ctx.mask is None else ctx.mask.copy()
        row_idx, entry_node = self._row_idx[anchor_item], self._entry_node[anchor_item]
        out = np.zeros(self._n_samples, dtype=bool)
        out[row_idx[np.isin(entry_node, active)]] = True
        if ctx.mask is not None:
            out &= ctx.mask
        return out

    # -- parity with BooleanDataRepresentation ------------------------

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @functools.cached_property
    def X(self) -> np.ndarray:
        """The unpacked ``(n_samples, n_features)`` bool matrix, rebuilt
        from the stored per-row feature sets on first access -- for
        handing a plain array to external tooling. The coverage path
        never triggers this."""
        X = np.zeros((self._n_samples, self._n_features), dtype=bool)
        for i, feats in enumerate(self._row_feats):
            X[i, feats] = True
        return X

    @classmethod
    def from_boolean(cls, data: "BooleanDataRepresentation") -> "NListRepresentation":
        """Build the N-list index for an existing
        `BooleanDataRepresentation`, reusing its `DataSpec` and labels."""
        return cls(data.spec, data.X, data.y)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "NListRepresentation":
        """Build the PPC-tree / N-list index from a `scipy.sparse` matrix
        matching `spec`'s feature order (densified during the build)."""
        from scipy.sparse import issparse

        return cls(spec, matrix.toarray() if issparse(matrix) else matrix, y)

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "NListRepresentation":
        return cls.from_boolean(
            BooleanDataRepresentation.from_dataframe(df, label_col=label_col, spec=spec, name=name)
        )

    @classmethod
    def from_xy(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "NListRepresentation":
        return cls.from_boolean(
            BooleanDataRepresentation.from_xy(
                X, y, feature_names=feature_names, threshold=threshold, spec=spec, name=name
            )
        )

    def __repr__(self) -> str:
        n_nodes = sum(w.shape[0] for w in self._path_words)
        return (
            f"{type(self).__name__}(spec={self.spec!r}, n_samples={self.n_samples}, "
            f"n_tree_nodes={n_nodes})"
        )

    # `_reindexed` uses the base's dense-projection fallback: the PPC-tree
    # has to be rebuilt over the new feature set either way.


class PrePostNListRepresentation(NListRepresentation):
    """`NListRepresentation` plus pre/post visit codes on every trie
    node, giving `refine_cover`'s "deeper feature" case (see the base
    class's own comment above `refine_cover`) a second way to answer
    "which of this feature's nodes are compatible with the rule so far":
    instead of scanning the feature's *entire* N-list, binary-search its
    pre-order-sorted occurrences against the current active nodes'
    ``[pre, subtree_end]`` subtree intervals (disjoint, since no two
    nodes of the same item can be ancestor/descendant of each other) and
    keep only what falls inside one -- skipping everything outside every
    active subtree entirely. `_reanchor` picks whichever of the two
    (plain scan vs. interval join) should be cheaper for the actual
    sizes involved, falling back to the plain scan -- identical to the
    base class -- whenever the active set isn't meaningfully smaller
    than the feature's own N-list.

    **Opt-in, not the default**, and not because it's unfinished: it was
    built, correctness-tested exhaustively (see
    `tests/test_representations.py`), and then *measured* -- at the data
    scales this library's own demos exercise (thousands of rows,
    hundreds of features) it comes out roughly break-even to slightly
    *worse* than `NListRepresentation`'s plain scan, including on data
    deliberately built to favor it (a handful of features present in 95%
    of rows, refined into from a small, already-narrowed active set).
    The interval-join path needs roughly 4-5x as many separate
    vectorized numpy calls (two `searchsorted`s, a `cumsum`, several
    `repeat`s, the final gather) as the plain scan (AND + compare +
    reduce), and each call's own fixed dispatch overhead is on the same
    order as the work being saved at these sizes -- the "touches fewer
    elements" argument is real, but numpy's per-call overhead eats it
    before it shows up in wall-clock time. It's kept as this separate,
    opt-in subclass -- rather than deleted, or folded into
    `NListRepresentation` behind a flag -- for the case it hasn't been
    measured on: a single feature's N-list running into the tens of
    thousands of rows while the active set stays tiny, where the
    element-count saving should eventually be large enough to clear that
    per-call overhead. Try it there; don't reach for it by default.
    """

    @staticmethod
    def _assign_pre_post(root: "_PPCNode") -> Tuple[dict, dict]:
        """Number every node with a pre-order visit number (`pre`) and,
        bottom-up, the largest `pre` among its own descendants
        (`subtree_end`) -- so a node's whole subtree is exactly the
        contiguous range ``[pre, subtree_end]``, and "is A an ancestor of
        B" reduces to ``A.pre <= B.pre <= A.subtree_end``. Returns
        ``({id(node): pre}, {id(node): subtree_end})`` rather than
        storing these on the node itself -- `_PPCNode.__slots__` stays
        exactly what the base class needs, so `NListRepresentation`
        doesn't pay for this even indirectly; only this one opt-in
        subclass, and only during its own one-time build, ever computes
        it. Iterative (not recursive), so a wide/deep tree can't hit
        Python's recursion limit. Nodes stay referenced by the tree/the
        traversal lists throughout, so `id()` is safe to use as a key
        here (no premature reuse)."""
        order: list = []
        stack = [root]
        pre_of: dict = {}
        counter = 0
        while stack:
            node = stack.pop()
            pre_of[id(node)] = counter
            counter += 1
            order.append(node)
            for child in node.children.values():
                stack.append(child)
        end_of: dict = {}
        for node in reversed(order):
            end = pre_of[id(node)]
            for child in node.children.values():
                cend = end_of[id(child)]
                if cend > end:
                    end = cend
            end_of[id(node)] = end
        return pre_of, end_of

    def _prepare_extra_per_item(self, root: "_PPCNode", k: int) -> None:
        self._pre_of, self._end_of = self._assign_pre_post(root)
        self._pre_by_item: list = [[] for _ in range(k)]
        self._end_by_item: list = [[] for _ in range(k)]

    def _extra_per_item(self, node: "_PPCNode", path_mask: int) -> None:
        self._pre_by_item[node.item].append(self._pre_of[id(node)])
        self._end_by_item[node.item].append(self._end_of[id(node)])

    def _finalize_extra_per_item(self, k: int) -> None:
        self._pre: list = []
        self._subtree_end: list = []
        self._pre_order: list = []
        self._pre_sorted: list = []
        for it in range(k):
            pre = np.asarray(self._pre_by_item[it], dtype=np.int64)
            self._pre.append(pre)
            self._subtree_end.append(np.asarray(self._end_by_item[it], dtype=np.int64))
            po = np.argsort(pre, kind="stable").astype(np.int64)
            self._pre_order.append(po)
            self._pre_sorted.append(pre[po])
        del self._pre_by_item, self._end_by_item, self._pre_of, self._end_of

    def _reanchor(self, anchor_item: int, active: np.ndarray, feature: int, mask_words: np.ndarray) -> np.ndarray:
        """Which of `feature`'s own nodes satisfy every bit in
        `mask_words`, given that `active` already lists the `anchor_item`
        nodes satisfying every bit *except* `feature`'s. See this class's
        own docstring for the two strategies and why the choice between
        them is a simple, conservative size comparison rather than a
        precise cost model -- it just needs to guarantee this never picks
        the interval join in a case where the plain scan would clearly
        have been cheaper, not to pick the optimal strategy every time.
        """
        words = self._path_words[feature]
        m_feature = words.shape[0]
        if active.size == 0 or active.size >= m_feature:
            hit = np.all((words & mask_words) == mask_words, axis=1)
            return np.flatnonzero(hit).astype(np.int64)

        pre_sorted, order = self._pre_sorted[feature], self._pre_order[feature]
        lo = np.searchsorted(pre_sorted, self._pre[anchor_item][active], side="left")
        hi = np.searchsorted(pre_sorted, self._subtree_end[anchor_item][active], side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64)
        # vectorized ragged-range gather (no Python loop over `active` --
        # that loop's own overhead was the first thing measured to erase
        # the saving this is supposed to buy)
        group_start = np.concatenate(([0], np.cumsum(counts)[:-1]))
        offset_in_group = np.arange(total) - np.repeat(group_start, counts)
        return order[np.repeat(lo, counts) + offset_in_group]

    def refine_cover(self, handle, feature: int):
        anchor_item, active, mask_words, ctx = handle
        feature = int(feature)
        new_mask_words = mask_words.copy()
        new_mask_words[feature >> 6] |= np.uint64(1 << (feature & 63))

        if anchor_item is None:
            m = self._path_words[feature].shape[0]
            return feature, np.arange(m, dtype=np.int64), new_mask_words, ctx

        if self._rank[feature] > self._rank[anchor_item]:
            new_active = self._reanchor(anchor_item, active, feature, new_mask_words)
            return feature, new_active, new_mask_words, ctx

        wj, bit = feature >> 6, np.uint64(1 << (feature & 63))
        keep = (self._path_words[anchor_item][active, wj] & bit) == bit
        return anchor_item, active[keep], new_mask_words, ctx


class SparseDataRepresentation(DataRepresentation):
    """A `scipy.sparse` encoding: the feature matrix as CSR (row slices
    for `features_of`) and CSC (column index-sets for `coverage`), plus
    optional labels `y`. Same `DataRepresentation` interface as
    `BooleanDataRepresentation` / `NListRepresentation`, so every
    `pyrulearn.learners.seco` learner runs on it unchanged and produces identical
    rules.

    `coverage(rule)` intersects the rule's features' CSC column
    index-sets (each a sorted list of the rows where that one feature is
    True), rarest feature first -- Eclat's "vertical" tid-list
    intersection, ``O(sum of the involved features' supports)``, and no
    ``(n_rows, n_features)`` matrix is ever built. This is the N-list
    without the prefix tree: an `NListRepresentation` whose PPC-tree did
    no prefix merging (every row its own root-to-leaf path) would have
    exactly these per-feature lists as its N-lists.

    Best when the data is genuinely sparse -- one-hot expansions of
    high-cardinality nominals, bag-of-words. Negation actively works
    against that: whenever every feature has a paired negation (`x`, `not
    x`), each pair contributes exactly one True value to every row, so
    density is *exactly* 50% regardless of how skewed the underlying
    attributes are -- not "sparse" at all, and worse than the packed
    matrix has any reason to be. `without_negations()` (which drops
    exactly the redundant complement half) is what makes this
    representation worth using; build with it, or call it on an
    already-negated representation, before reaching for `Sparse`.
    """

    def __init__(self, spec: DataSpec, X, y: Optional[np.ndarray] = None):
        super().__init__(spec)
        from scipy import sparse

        if sparse.issparse(X):
            csr = X.tocsr()
        else:
            csr = sparse.csr_matrix(np.asarray(X, dtype=bool))
        if csr.dtype != bool:
            csr = csr.astype(bool)
        csr.eliminate_zeros()
        if csr.shape[1] != spec.n_features:
            raise ValueError(
                f"X has {csr.shape[1]} columns but this DataSpec has "
                f"{spec.n_features} features"
            )
        csr.sort_indices()
        self._n_samples: int = csr.shape[0]
        self.y: Optional[np.ndarray] = None if y is None else np.asarray(y)
        self._csr = csr
        self._csc = csr.tocsc()
        self._csc.sort_indices()

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def coverage(self, rule: "Rule") -> np.ndarray:
        feats = [int(l.feature) for l in rule.conditions]
        out = np.zeros(self._n_samples, dtype=bool)
        if not feats:
            out[:] = True
            return out
        indptr, indices = self._csc.indptr, self._csc.indices
        feats.sort(key=lambda f: indptr[f + 1] - indptr[f])  # rarest first
        rows = indices[indptr[feats[0]]:indptr[feats[0] + 1]]
        for f in feats[1:]:
            if rows.size == 0:
                break
            rows = np.intersect1d(
                rows, indices[indptr[f]:indptr[f + 1]], assume_unique=True
            )
        out[rows] = True
        return out

    def features_of(self, row: int) -> np.ndarray:
        s, e = self._csr.indptr[row], self._csr.indptr[row + 1]
        return np.sort(self._csr.indices[s:e].astype(np.int64))

    @functools.cached_property
    def X(self) -> np.ndarray:
        return np.asarray(self._csr.todense(), dtype=bool)

    # -- incremental coverage (search fast path) -----------------------
    #
    # A handle is (rows, scope): `rows` is the current rule's row-id
    # array -- a tid-list, exactly what `coverage` builds by intersecting
    # one feature's column at a time, except now the intersection carries
    # across search steps instead of restarting from the full column set
    # for every candidate. `scope` is the original `example_mask` (or
    # `None`), carried through unchanged -- needed to get fn/tn's *total*
    # positive/negative counts right when a mask is in play (`rows`
    # alone, being already narrowed to the rule's own coverage, can't
    # answer "how many masked rows are there in total").

    def initial_cover(self, example_mask: Optional[np.ndarray] = None):
        if example_mask is None:
            return np.arange(self._n_samples, dtype=np.int64), None
        mask = np.array(example_mask, dtype=bool, copy=True)
        return np.flatnonzero(mask).astype(np.int64), mask

    def refine_cover(self, handle, feature: int):
        rows, mask = handle
        indptr, indices = self._csc.indptr, self._csc.indices
        col = indices[indptr[feature]:indptr[feature + 1]]
        return np.intersect1d(rows, col, assume_unique=True), mask

    def cover_counts(self, handle, positive_class: Any) -> Tuple[int, int, int, int]:
        if self.y is None:
            raise ValueError("cover_counts needs labels (self.y)")
        rows, mask = handle
        tp = int(np.sum(self.y[rows] == positive_class))
        fp = int(rows.size) - tp
        y_scope = self.y if mask is None else self.y[mask]
        n_pos = int(np.sum(y_scope == positive_class))
        n_neg = y_scope.size - n_pos
        return tp, fp, n_pos - tp, n_neg - fp

    def cover_rows(self, handle) -> np.ndarray:
        rows, _ = handle
        out = np.zeros(self._n_samples, dtype=bool)
        out[rows] = True
        return out

    def _reindexed(self, new_spec, source, is_complement):
        from scipy import sparse

        if is_complement is None:
            new = self._csc[:, source]
            return SparseDataRepresentation(new_spec, new, self.y)
        cols = []
        for j, s in enumerate(np.asarray(source)):
            col = self._csc[:, int(s)]
            if is_complement[j]:
                dense = np.ones((self._n_samples, 1), dtype=bool)
                dense[col.indices] = False
                cols.append(sparse.csc_matrix(dense))
            else:
                cols.append(col)
        return SparseDataRepresentation(new_spec, sparse.hstack(cols, format="csc"), self.y)

    @classmethod
    def from_scipy(cls, matrix, spec: DataSpec, y: Optional[np.ndarray] = None) -> "SparseDataRepresentation":
        """Wrap an existing `scipy.sparse` matrix (any format) that
        already matches `spec`'s feature order."""
        return cls(spec, matrix, y)

    @classmethod
    def from_boolean(cls, data: "BooleanDataRepresentation") -> "SparseDataRepresentation":
        return cls(data.spec, data.X, data.y)

    @classmethod
    def from_dataframe(
        cls, df, label_col: Optional[str] = None, spec: Optional[DataSpec] = None, name: Optional[str] = None
    ) -> "SparseDataRepresentation":
        return cls.from_boolean(
            BooleanDataRepresentation.from_dataframe(df, label_col=label_col, spec=spec, name=name)
        )

    @classmethod
    def from_xy(
        cls,
        X,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
        threshold: Optional[float] = None,
        spec: Optional[DataSpec] = None,
        name: Optional[str] = None,
    ) -> "SparseDataRepresentation":
        return cls.from_boolean(
            BooleanDataRepresentation.from_xy(
                X, y, feature_names=feature_names, threshold=threshold, spec=spec, name=name
            )
        )

    def __repr__(self) -> str:
        nnz = self._csr.nnz
        cells = self._n_samples * self.spec.n_features
        density = nnz / cells if cells else 0.0
        return (
            f"SparseDataRepresentation(spec={self.spec!r}, n_samples={self.n_samples}, "
            f"nnz={nnz}, density={density:.2f})"
        )
