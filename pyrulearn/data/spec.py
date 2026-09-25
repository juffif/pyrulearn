"""
pyrulearn.data.spec
======================

`DataSpec` is a pure feature-space specification (names, count, typed
attributes, constraints, missing-value policy) -- no data. See
`pyrulearn.data.representation` for `DataRepresentation`/
`BooleanDataRepresentation`, which hold the actual values (a
`DataRepresentation` always points back at the `DataSpec` it was built
against). Because a `DataSpec` carries no data, the same instance can be
shared verbatim across multiple representations built from it (e.g. a
train split and a test split), guaranteeing identical feature indexing
between them by construction rather than by convention -- useful when
hand-writing rules, or importing rules from an external learner, before
any dataset has even been loaded.

Beyond plain Boolean feature names, a DataSpec can optionally know the
**typed attributes** (boolean/nominal/numeric/set/hierarchical/
relational) its features derive from, via `feature_specs` -- and the
**constraints** those attributes imply among their derived features
(mutual exclusion for nominal equality tests, monotonic chains for
numeric threshold tests, sibling exclusion + upward implication for
hierarchy nodes; see `pyrulearn.data.attributes`). Building a DataSpec this
way is what lets a `Rule` ask whether it's internally consistent, or what
it implies beyond its explicit conditions -- see `DataSpec.propagate` and
`Rule.is_consistent` / `Rule.implied_conditions`.

Two ways to build one:
- Plain: ``DataSpec(feature_names=[...])`` -- every feature is
  attribute-less (the original, simplest use case).
- Typed: ``DataSpecBuilder().add_nominal(...).add_numeric(...).build()``
  -- generates feature names/specs/constraints from attribute
  declarations automatically.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

from .attributes import (
    Attribute,
    AttributeType,
    Constraint,
    ExactlyOne,
    FeatureSpec,
    FeatureValues,
    Hierarchy,
    Implies,
    MissingStrategy,
    MutuallyExclusive,
    NominalGroup,
    NumericGroup,
    ThresholdChain,
)

#: FeatureSpec.op values that mark a "positive" feature vs. its negation.
_POSITIVE_OPS = (None, "==", ">=", "<=")
_NEGATION_OPS = ("not", "!=", "<", ">")


class DataSpec:
    """Pure feature-space specification -- no data. See
    `pyrulearn.data.BooleanDataRepresentation` for binding
    actual values to a DataSpec.

    Parameters
    ----------
    feature_names : sequence of str
        Names of the Boolean features, in index order. Canonical source
        of truth for `n_features` and for resolving names <-> indices.
    name : str, optional
        Human-readable label for this spec (e.g. dataset name) -- cosmetic
        only.
    feature_specs : sequence of FeatureSpec, optional
        Provenance for each feature (which attribute/test it derives
        from), parallel to `feature_names`. Usually set via
        `DataSpecBuilder` rather than by hand. Defaults to plain,
        attribute-less specs if omitted.
    attributes : dict of str -> Attribute, optional
        The typed attributes referenced by `feature_specs`, keyed by name.
    constraints : sequence of Constraint, optional
        Relationships auto-derived from `attributes` (mutual exclusion /
        threshold chains / hierarchy edges). See `DataSpec.propagate`.
    missing_strategy : MissingStrategy, optional
        This spec's own default for how `pyrulearn.data.io.binarize`
        should handle missing raw values when reading data against it
        -- falls back to `DataSpec.DEFAULT_MISSING_STRATEGY`
        (``NEVER_COVERS``) if unset. An explicit `missing_strategy=`
        passed to `binarize` itself always wins over both.
    """

    #: Class-wide fallback for missing-value handling when neither an
    #: explicit `missing_strategy=` (to `pyrulearn.data.io.binarize`)
    #: nor a DataSpec's own `missing_strategy` is set.
    DEFAULT_MISSING_STRATEGY: MissingStrategy = MissingStrategy.NEVER_COVERS

    def __init__(
        self,
        feature_names: Sequence[str],
        name: Optional[str] = None,
        feature_specs: Optional[Sequence[FeatureSpec]] = None,
        attributes: Optional[Dict[str, Attribute]] = None,
        constraints: Optional[Sequence[Constraint]] = None,
        missing_strategy: Optional[MissingStrategy] = None,
    ):
        self.feature_names: List[str] = list(feature_names)
        self._name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
        if len(self._name_to_idx) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        self.name = name

        self.feature_specs: Tuple[FeatureSpec, ...] = (
            tuple(feature_specs) if feature_specs is not None
            else tuple(FeatureSpec(i, n) for i, n in enumerate(self.feature_names))
        )
        if len(self.feature_specs) != len(self.feature_names):
            raise ValueError("feature_specs must be parallel to feature_names")
        self.attributes: Dict[str, Attribute] = dict(attributes) if attributes else {}
        self.constraints: Tuple[Constraint, ...] = tuple(constraints) if constraints else ()
        self.missing_strategy: Optional[MissingStrategy] = missing_strategy
        self._negation_map: Dict[int, int] = self._build_negation_map()

    # -- feature spec --------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def feature_index(self, name_or_idx: Union[str, int]) -> int:
        if isinstance(name_or_idx, str):
            return self._name_to_idx[name_or_idx]
        return int(name_or_idx)

    def feature_name(self, idx: int) -> str:
        return self.feature_names[idx]

    def feature_spec(self, idx: int) -> FeatureSpec:
        return self.feature_specs[idx]

    def _build_negation_map(self) -> Dict[int, int]:
        """Pair each feature with its exact-negation feature, if one
        exists (``color=red`` <-> ``color!=red``, ``age>=30`` <->
        ``age<30``, ``smoker`` <-> ``not smoker``). Both directions.

        A BINARY attribute's two ``==`` features are paired with each
        other (``sex=male`` <-> ``sex=female``): its value list is
        declared complete, so each is the other's exact negation."""
        def key(spec: FeatureSpec):
            if spec.op in ("==", "!="):
                return (spec.attribute, spec.value, "eq")
            if spec.op in (">=", "<"):
                return (spec.attribute, spec.value, "ge")
            if spec.op in ("<=", ">"):
                return (spec.attribute, spec.value, "le")
            if spec.attribute is not None and spec.op in (None, "not"):  # boolean
                return (spec.attribute, None, "bool")
            return None

        pos: Dict[Any, int] = {}
        neg: Dict[Any, int] = {}
        for spec in self.feature_specs:
            k = key(spec)
            if k is None:
                continue
            (neg if spec.op in _NEGATION_OPS else pos)[k] = spec.index
        out: Dict[int, int] = {}
        for k, pi in pos.items():
            ni = neg.get(k)
            if ni is not None:
                out[pi] = ni
                out[ni] = pi

        binary = {n for n, a in self.attributes.items() if a.type == AttributeType.BINARY}
        if binary:
            eq_by_attr: Dict[str, List[int]] = {}
            for spec in self.feature_specs:
                if spec.op == "==" and spec.attribute in binary:
                    eq_by_attr.setdefault(spec.attribute, []).append(spec.index)
            for i, j in (idxs for idxs in eq_by_attr.values() if len(idxs) == 2):
                out[i] = j
                out[j] = i
        return out

    def negation_of(self, name_or_idx: Union[str, int]) -> Optional[int]:
        """The feature index of `name_or_idx`'s exact negation
        (``color=red`` -> ``color!=red``, and back), or `None` if the
        attribute was declared without negation (or the feature has no
        negation, e.g. a plain attribute-less one, a ``<=`` feature, a
        hierarchy node)."""
        return self._negation_map.get(self.feature_index(name_or_idx))

    # -- negation-feature toggling -------------------------------------

    @property
    def has_negation_features(self) -> bool:
        """True if any feature is an explicit negation (``not f``,
        ``x != v``, ``x < t``, ``x > t``)."""
        return any(fs.op in _NEGATION_OPS for fs in self.feature_specs)

    def without_negations(self) -> Tuple["DataSpec", np.ndarray]:
        """This spec with every explicit negation feature removed, plus
        an int array `keep` mapping each surviving feature to the old
        index it came from (matched by name), so a `DataRepresentation`
        can do ``new_X = old_X[:, keep]``. Constraints are regenerated for the
        reduced feature set (``NominalGroup`` -> ``ExactlyOne``,
        ``NumericGroup`` -> ``ThresholdChain``, the ``not``-pair
        ``MutuallyExclusive`` constraints simply drop out). Returns
        ``(self, arange(n_features))`` unchanged if there are no negation
        features to remove.
        """
        if not self.has_negation_features:
            return self, np.arange(self.n_features)
        if self.attributes:
            new_spec = _rebuilt_with_negation(self, negation=False)
            keep = np.array([self.feature_index(n) for n in new_spec.feature_names], dtype=int)
            return new_spec, keep
        keep = [fs.index for fs in self.feature_specs if fs.op in _POSITIVE_OPS]
        new_spec = DataSpec([self.feature_names[i] for i in keep], name=self.name)
        return new_spec, np.asarray(keep, dtype=int)

    def with_negations(self) -> Tuple["DataSpec", np.ndarray, np.ndarray]:
        """This spec with a negation feature added for every attribute /
        test that lacks one, plus two parallel arrays over the *new*
        feature set: `source` (the old feature index each new feature
        draws its data from, by name) and `is_complement` (whether that
        column must be logically negated). A `DataRepresentation`
        rebuilds its data as::

            new_X[:, j] = ~old_X[:, source[j]] if is_complement[j] else old_X[:, source[j]]

        This assumes each added negation feature is the exact complement
        of its positive partner -- correct for data binarized without
        missing-value routing (``MissingStrategy.SEPARATE`` and friends
        break the assumption; rebuild from the raw data in that case).
        Features that already have a negation are copied straight across
        (never re-derived). Returns identity arrays if every feature
        already has its negation.
        """
        target = _with_negation_spec(self)
        old_names = set(self.feature_names)
        source = np.empty(target.n_features, dtype=int)
        is_complement = np.zeros(target.n_features, dtype=bool)
        for j, nm in enumerate(target.feature_names):
            if nm in old_names:
                source[j] = self.feature_index(nm)
            else:
                partner = target.negation_of(j)
                if partner is None:
                    raise ValueError(
                        f"with_negations: new feature {nm!r} has no positive partner to derive from"
                    )
                source[j] = self.feature_index(target.feature_name(partner))
                is_complement[j] = True
        return target, source, is_complement

    # -- constraint propagation ------------------------------------------

    def propagate(self, fixed: FeatureValues) -> FeatureValues:
        """Compute the closure of a partial assignment (`fixed`:
        feature_index -> True/False) under this spec's constraints,
        iterating to a fixed point.

        Returns a new dict (superset of `fixed`) with every additional
        forced assignment included. Raises `ValueError` if `fixed` is
        contradictory under the constraints (e.g. two different nominal
        values fixed True for the same attribute).

        A no-op (returns `fixed` unchanged) when this spec has no
        constraints -- so it's always safe to call, even on a plain
        attribute-less DataSpec.
        """
        assignment = dict(fixed)
        if not self.constraints:
            return assignment
        changed = True
        while changed:
            changed = False
            for c in self.constraints:
                forced = c.propagate(assignment)
                for i, v in forced.items():
                    if i in assignment:
                        if assignment[i] != v:
                            raise ValueError(
                                f"Contradiction: feature {i} ({self.feature_names[i]}) "
                                f"forced to both {assignment[i]} and {v}"
                            )
                    else:
                        assignment[i] = v
                        changed = True
        return assignment

    def __repr__(self) -> str:
        label = f" {self.name!r}" if self.name else ""
        con_info = f", {len(self.constraints)} constraints" if self.constraints else ""
        return f"DataSpec({self.n_features} features{label}{con_info})"


class NumericFeatureIndices(NamedTuple):
    """`DataSpecBuilder.add_numeric`'s return value once `le_thresholds`
    is given: `ge`/`le` are each ``{threshold: feature_index}``, same
    shape as the plain-`>=`-only return value. Still unpacks
    positionally (``ge_idx, le_idx = add_numeric(...)``) for callers
    that don't need the field names.
    """
    ge: Dict[float, int]
    le: Dict[float, int]


class _WithNegation(dict):
    """A ``{key: positive-feature-index}`` dict (unchanged for existing
    callers) that also carries `.negative` -- the parallel
    ``{key: negation-feature-index}`` dict, or `None` when the attribute
    was built without negation -- and `.all`, every feature index the
    attribute contributed (sorted)."""

    def __init__(self, positive: Dict[Any, int], negative: Optional[Dict[Any, int]] = None):
        super().__init__(positive)
        self.negative: Optional[Dict[Any, int]] = negative
        idxs = set(positive.values())
        if negative:
            idxs |= set(negative.values())
        self.all: List[int] = sorted(idxs)


class NominalFeatureIndices(_WithNegation):
    """`add_nominal`'s return: ``{value: '=' feature index}`` plus
    `.negative` (``{value: '!=' feature index}`` or `None`) and `.all`."""


class NumericGeIndices(_WithNegation):
    """`add_numeric`'s return: ``{threshold: '>=' feature index}`` plus
    `.negative` (``{threshold: '<' feature index}`` or `None`), `.all`,
    and `.le` (``{threshold: '<=' feature index}`` or `None`, the
    separate `le_thresholds` family)."""

    le: Optional[Dict[float, int]] = None


class BoolFeatureIndex(int):
    """`add_boolean`'s return: the positive feature index (behaves as a
    plain `int` everywhere) with `.negative` (the ``not name`` feature
    index, or `None`) and `.all` attached."""

    negative: Optional[int]
    all: List[int]

    def __new__(cls, positive: int, negative: Optional[int] = None):
        obj = super().__new__(cls, positive)
        obj.negative = negative
        obj.all = sorted({positive} | ({negative} if negative is not None else set()))
        return obj


class DataSpecBuilder:
    """Incrementally build a DataSpec's feature space from typed
    attributes, auto-generating the derived Boolean features (one per
    nominal value / numeric threshold / set value / hierarchy node) and
    the constraints among them.

    By default (``negation=True``) every attribute also gets explicit
    negation features -- ``not smoker``, ``color!=red``, ``age<20`` --
    each paired with its positive counterpart by a `MutuallyExclusive` /
    `NominalGroup` / `NumericGroup` constraint, so a rule condition
    meaning "feature absent" is a positive literal on the negation
    feature rather than a negative literal. Pass ``negation=False`` (or
    per-attribute ``negation=False``) for the positive-only feature
    space.

    Example
    -------
    >>> b = DataSpecBuilder(negation=False)
    >>> b.add_boolean("smoker")
    0
    >>> color_idx = b.add_nominal("color", ["red", "green", "blue"])
    >>> age_idx = b.add_numeric("age", [20, 30, 40])
    >>> dv = b.build()
    >>> dv.feature_names
    ['smoker', 'color=red', 'color=green', 'color=blue', 'age>=20', 'age>=30', 'age>=40']
    """

    def __init__(self, negation: bool = True):
        self._names: List[str] = []
        self._specs: List[FeatureSpec] = []
        self._constraints: List[Constraint] = []
        self._attributes: Dict[str, Attribute] = {}
        #: builder-wide default for whether attributes get negation
        #: features -- per-attribute `negation=` overrides it.
        self._negation_default = negation

    def _neg(self, override: Optional[bool]) -> bool:
        return self._negation_default if override is None else override

    def add_boolean(
        self, name: str, missing_values: Sequence[Any] = (), *, negation: Optional[bool] = None
    ) -> BoolFeatureIndex:
        """Add a Boolean attribute. Without negation: one plain feature,
        no constraint (the original behavior). With negation: also a
        ``not {name}`` feature, mutually exclusive with it (both False =
        the value is missing/unknown). `missing_values` are raw values
        (besides `None`/NaN) to also treat as missing when reading data
        -- see `MissingStrategy`. Returns the positive feature index (an
        `int` carrying `.negative`/`.all`)."""
        neg_on = self._neg(negation)
        pos_idx = len(self._names)
        self._names.append(name)
        self._specs.append(
            FeatureSpec(pos_idx, name, attribute=name, op=None) if neg_on
            else FeatureSpec(pos_idx, name)
        )
        neg_idx: Optional[int] = None
        if neg_on:
            neg_idx = len(self._names)
            self._names.append(f"not {name}")
            self._specs.append(FeatureSpec(neg_idx, f"not {name}", attribute=name, op="not"))
            self._constraints.append(MutuallyExclusive([pos_idx, neg_idx]))
        self._attributes[name] = Attribute(
            name, AttributeType.BOOLEAN, missing_values=tuple(missing_values), negation=neg_on
        )
        return BoolFeatureIndex(pos_idx, neg_idx)

    def add_nominal(
        self,
        attribute: str,
        domain: Sequence[Any],
        *,
        missing_name: Optional[Any] = None,
        missing_values: Sequence[Any] = (),
        negation: Optional[bool] = None,
    ) -> NominalFeatureIndices:
        """Add a nominal attribute. Generates one equality feature per
        value (``"{attribute}={value}"``) and, with negation on, one
        inequality feature per value (``"{attribute}!={value}"``) -- also
        for exactly two values: the domain may be incomplete (an unseen
        value must satisfy every ``!=``), so ``x!=a`` is not ``x=b``. For
        a closed two-value attribute use `add_binary`. The constraint is
        `ExactlyOne` over the equality features without negation,
        `NominalGroup` (which subsumes it, plus the ``!=`` logic) with.
        Returns a `NominalFeatureIndices` -- a ``{value: '=' index}``
        dict, with `.negative` (the ``!=`` dict, or `None` if negation is
        off) and `.all`.

        `missing_name`, if given, adds one more value *in the same
        group* -- what lets `MissingStrategy.SEPARATE` route a missing
        raw value into its own feature. `missing_values` are raw values
        (besides `None`/NaN) to also treat as missing when reading data.
        """
        neg_on = self._neg(negation)
        all_values = list(domain) + ([missing_name] if missing_name is not None else [])
        eq: Dict[Any, int] = {}
        eq_idxs: List[int] = []
        for value in all_values:
            idx = len(self._names)
            name = f"{attribute}={value}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="==", value=value))
            eq[value] = idx
            eq_idxs.append(idx)
        ne: Optional[Dict[Any, int]] = None
        ne_idxs: List[int] = []
        if neg_on:
            ne = {}
            for value in all_values:
                idx = len(self._names)
                name = f"{attribute}!={value}"
                self._names.append(name)
                self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="!=", value=value))
                ne[value] = idx
                ne_idxs.append(idx)
        self._attributes[attribute] = Attribute(
            attribute, AttributeType.NOMINAL, tuple(domain),
            missing_name=missing_name, missing_values=tuple(missing_values), negation=neg_on,
        )
        if len(eq_idxs) > 1:
            self._constraints.append(
                NominalGroup(eq_idxs, ne_idxs) if neg_on else ExactlyOne(eq_idxs)
            )
        return NominalFeatureIndices(eq, ne)

    def add_binary(
        self,
        attribute: str,
        values: Sequence[Any],
        *,
        missing_values: Sequence[Any] = (),
    ) -> NominalFeatureIndices:
        """Add a binary attribute: a *closed* set of exactly two values
        (``sex`` in ``("male", "female")``). Generates the two equality
        features ``"{attribute}={v}"``, each the other's negation -- no
        ``!=`` columns, whatever the builder's negation setting, since
        with the value list known to be complete ``x!=v1`` is ``x=v2``.
        Both are False for a missing value and for any value outside the
        two (treated as missing); they're linked by `MutuallyExclusive`,
        not `ExactlyOne`, since "neither" is a valid state.

        Returns a `NominalFeatureIndices`: ``{value: index}``, with
        `.negative` mapping each value to the *other* value's index.
        `missing_values` are raw values (besides `None`/NaN) to also
        treat as missing when reading data.
        """
        values = list(values)
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError(f"add_binary({attribute!r}) needs exactly two distinct values, got {values!r}")
        eq: Dict[Any, int] = {}
        for value in values:
            idx = len(self._names)
            name = f"{attribute}={value}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="==", value=value))
            eq[value] = idx
        v1, v2 = values
        self._attributes[attribute] = Attribute(
            attribute, AttributeType.BINARY, tuple(values), missing_values=tuple(missing_values),
        )
        self._constraints.append(MutuallyExclusive([eq[v1], eq[v2]]))
        return NominalFeatureIndices(eq, {v1: eq[v2], v2: eq[v1]})

    def add_numeric(
        self,
        attribute: str,
        ge_thresholds: Sequence[float] = (),
        *,
        le_thresholds: Sequence[float] = (),
        missing_name: Optional[Any] = None,
        missing_values: Sequence[Any] = (),
        negation: Optional[bool] = None,
    ) -> Union["NumericGeIndices", "NumericFeatureIndices"]:
        """Add a numeric attribute binarized at the given cut points.
        `ge_thresholds` generates one ``>=`` threshold-test feature per
        cut point (named ``"{attribute}>={threshold}"``, sorted
        ascending), with a `ThresholdChain` constraint over them if
        there's more than one -- as before (just renamed from the
        original bare `thresholds`, for symmetry with `le_thresholds`
        now that there are two).

        `le_thresholds` (default: none, a complete no-op -- existing
        callers see no change at all) *separately* generates ``<=``
        features (``"{attribute}<={threshold}"``) with their own
        `ThresholdChain`. It's a genuinely different family, not the
        complement of the ``>=`` one: ``x<=t`` is the exact logical
        complement of ``x>t``, not of ``x>=t`` (``x<=t`` and
        ``NOT(x>=t)`` disagree exactly at ``x==t``), so folding it into
        the same chain would get that boundary wrong. Internally this
        still reuses `ThresholdChain` completely unchanged for the
        ``<=`` family too -- its propagation logic only assumes "index 0
        is easiest to satisfy, later indices are monotonically harder,"
        which is generic enough for either direction; feeding it the
        ``<=`` feature indices in *descending* threshold order (rather
        than the ``>=`` family's ascending order) is what makes that
        already-correct for ``<=``'s mirrored monotonicity, with no
        changes to the class itself. The two families are built the same
        way internally, each just walked in the order that's "easiest to
        satisfy first" for its own direction.

        The two families are independent and not cross-checked against
        each other -- nothing stops a rule from stating both
        ``age>=30`` and ``age<=25`` (impossible to satisfy, but not
        flagged inconsistent; it will simply never cover any real
        example, which for rules read in from an external source, as
        opposed to hand-built, is the normal and sufficient outcome).
        Giving the *same* value to both `ge_thresholds` and
        `le_thresholds` is exactly how to represent ``attribute==value``
        as a conjunction of the two resulting literals.

        With negation on, each ``>=`` cut point also gets its exact
        complement ``"{attribute}<{threshold}"``, and a `NumericGroup`
        (rather than `ThresholdChain`) owns the ``>=`` family plus the
        ``>=``/``<`` complement links. The ``<=`` (`le_thresholds`)
        family is unaffected by `negation`.

        Returns a `NumericGeIndices` -- a ``{threshold: '>=' index}``
        dict (unchanged for existing callers) carrying `.negative` (the
        ``{threshold: '<' index}`` dict, or `None`), `.all`, and `.le`
        (the ``<=`` dict, or `None`). Once `le_thresholds` is given,
        returns a `NumericFeatureIndices` ``(ge, le)`` named pair
        instead (``ge`` still a `NumericGeIndices`), unpackable
        positionally (``ge_idx, le_idx = add_numeric(...)``).

        `missing_name`, if given, adds one more feature (named
        ``"{attribute}={missing_name}"``) for `MissingStrategy.SEPARATE`
        to route into -- deliberately *outside* either `ThresholdChain`
        (and with no constraint linking it to either), since a numeric
        value being "missing" is independent of whether it would pass
        any `>=`/`<=` test. `missing_values` are raw values (besides
        `None`/NaN) to also treat as missing when reading data for this
        attribute.
        """
        neg_on = self._neg(negation)
        ge_sorted = sorted(set(ge_thresholds))
        le_sorted = sorted(set(le_thresholds), reverse=True)
        ge_indices: Dict[float, int] = {}
        lt_indices: Dict[float, int] = {}
        le_indices: Dict[float, int] = {}
        ge_feature_idxs: List[int] = []
        lt_feature_idxs: List[int] = []
        le_feature_idxs: List[int] = []

        for t in ge_sorted:
            idx = len(self._names)
            name = f"{attribute}>={t}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op=">=", value=t))
            ge_indices[t] = idx
            ge_feature_idxs.append(idx)
        if neg_on:
            for t in ge_sorted:  # ascending, parallel to the >= family
                idx = len(self._names)
                name = f"{attribute}<{t}"
                self._names.append(name)
                self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="<", value=t))
                lt_indices[t] = idx
                lt_feature_idxs.append(idx)
        gt_indices: Dict[float, int] = {}
        gt_feature_idxs: List[int] = []
        for t in le_sorted:
            idx = len(self._names)
            name = f"{attribute}<={t}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="<=", value=t))
            le_indices[t] = idx
            le_feature_idxs.append(idx)
        if neg_on:
            for t in le_sorted:  # descending, parallel to the <= family
                idx = len(self._names)
                name = f"{attribute}>{t}"
                self._names.append(name)
                self._specs.append(FeatureSpec(idx, name, attribute=attribute, op=">", value=t))
                gt_indices[t] = idx
                gt_feature_idxs.append(idx)

        if missing_name is not None:
            idx = len(self._names)
            name = f"{attribute}={missing_name}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="==", value=missing_name))
            ge_indices[missing_name] = idx

        self._attributes[attribute] = Attribute(
            attribute, AttributeType.NUMERIC, tuple(ge_sorted) if ge_sorted else None,
            le_domain=tuple(sorted(le_sorted)) if le_sorted else None,
            missing_name=missing_name, missing_values=tuple(missing_values), negation=neg_on,
        )
        if neg_on and lt_feature_idxs:
            self._constraints.append(NumericGroup(ge_feature_idxs, lt_feature_idxs))
        elif len(ge_feature_idxs) > 1:
            self._constraints.append(ThresholdChain(ge_feature_idxs))
        if neg_on and gt_feature_idxs:
            self._constraints.append(NumericGroup(le_feature_idxs, gt_feature_idxs))
        elif len(le_feature_idxs) > 1:
            self._constraints.append(ThresholdChain(le_feature_idxs))

        ge_ret = NumericGeIndices(ge_indices, lt_indices if neg_on else None)
        ge_ret.le = le_indices if le_thresholds else None
        if not le_thresholds:
            return ge_ret
        le_ret = _WithNegation(le_indices, gt_indices if neg_on else None)
        return NumericFeatureIndices(ge_ret, le_ret)

    def add_set(self, attribute: str, domain: Sequence[Any], missing_values: Sequence[Any] = ()) -> Dict[Any, int]:
        """Add a set-valued attribute: an example's value is a subset of
        `domain` (e.g. ``tags`` with domain ``{urgent, billing, bug}``,
        where an example can have any combination of tags). Generates one
        membership-test feature per possible value (named
        ``"{attribute} has {value}"``). Unlike `add_nominal`, values are
        independent -- no mutual-exclusion or exhaustiveness constraint is
        generated, since any number of them (including zero or all) may
        hold at once. `missing_values` are raw values (besides `None`/NaN)
        to also treat as missing when reading data for this attribute
        (there's no `missing_name`/dedicated-feature option here --
        `MissingStrategy.SEPARATE` isn't supported for SET attributes).
        Returns ``{value: feature_index}``.
        """
        indices: Dict[Any, int] = {}
        for value in domain:
            idx = len(self._names)
            name = f"{attribute} has {value}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="has", value=value))
            indices[value] = idx
        self._attributes[attribute] = Attribute(
            attribute, AttributeType.SET, tuple(domain), missing_values=tuple(missing_values)
        )
        return indices

    def add_hierarchical(
        self, attribute: str, tree: Dict[Any, Any], missing_values: Sequence[Any] = ()
    ) -> Dict[Any, int]:
        """Add a hierarchical attribute from a nested-dict tree, e.g.

        >>> b.add_hierarchical("region", {
        ...     "Europe": {"France": {"Paris": {}, "Lyon": {}}, "Germany": {}},
        ...     "Asia": {},
        ... })

        Generates one equality-test feature per node (named
        ``"{attribute}={node}"``, same convention as `add_nominal`), a
        `MutuallyExclusive` constraint among siblings at each level (two
        children of the same parent can't both hold), and an `Implies`
        constraint from each node to its parent -- so e.g. ``region=Paris``
        forces ``region=France``, which forces ``region=Europe``. Unlike
        `add_nominal`, sibling groups are *not* assumed exhaustive: an
        example can be known only down to ``region=Europe`` without any of
        Europe's children being set. `missing_values` are raw values
        (besides `None`/NaN) to also treat as missing when reading data
        for this attribute (no `missing_name` option here -- see
        `add_nominal`/`add_numeric`). Returns ``{node: feature_index}``.
        """
        return self._add_hierarchy(attribute, Hierarchy.from_nested(tree), missing_values=missing_values)

    def _add_hierarchy(
        self, attribute: str, hierarchy: Hierarchy, missing_values: Sequence[Any] = ()
    ) -> Dict[Any, int]:
        """Same as `add_hierarchical`, but taking an already-built
        `Hierarchy` directly instead of a nested-dict tree -- used
        internally (e.g. by `merge_dataspecs`) to reuse an existing
        attribute's hierarchy without re-parsing it.
        """
        indices: Dict[Any, int] = {}
        for node in hierarchy.nodes:
            idx = len(self._names)
            name = f"{attribute}={node}"
            self._names.append(name)
            self._specs.append(FeatureSpec(idx, name, attribute=attribute, op="==", value=node))
            indices[node] = idx

        siblings: Dict[Any, List[Any]] = {}
        for node in hierarchy.nodes:
            siblings.setdefault(hierarchy.parent_of(node), []).append(node)
        for group in siblings.values():
            if len(group) > 1:
                self._constraints.append(MutuallyExclusive([indices[n] for n in group]))

        for node in hierarchy.nodes:
            parent = hierarchy.parent_of(node)
            if parent is not None:
                self._constraints.append(Implies(indices[node], indices[parent]))

        self._attributes[attribute] = Attribute(
            attribute, AttributeType.HIERARCHICAL, tuple(hierarchy.nodes), hierarchy=hierarchy,
            missing_values=tuple(missing_values),
        )
        return indices

    def add_relational(self, name: str, attributes: Sequence[str], expression: Optional[str] = None) -> int:
        """Add a single relational feature derived from multiple existing
        attributes (e.g. ``income_gt_age_scaled`` reading ``income`` and
        ``age``, with ``expression="income > age * 1000"``). This builder
        does not compute the feature's value itself -- it only records
        provenance (which source `attributes` it reads) and an optional
        human-readable `expression` for display; the actual Boolean column
        must already exist in the data passed to `attach_data`/`build`.
        No constraints are generated, since the relationship between a
        relational feature and its sources isn't inferrable in general.
        Returns the feature index.
        """
        idx = len(self._names)
        self._names.append(name)
        self._specs.append(FeatureSpec(idx, name, attributes=tuple(attributes), op="expr", expression=expression))
        self._attributes[name] = Attribute(name, AttributeType.RELATIONAL, sources=tuple(attributes))
        return idx

    def build(self, name: Optional[str] = None, missing_strategy: Optional[MissingStrategy] = None) -> DataSpec:
        return DataSpec(
            self._names, name=name,
            feature_specs=self._specs, attributes=self._attributes, constraints=self._constraints,
            missing_strategy=missing_strategy,
        )


# ---------------------------------------------------------------------------
# Merging two DataSpecs over the same underlying attributes
# ---------------------------------------------------------------------------

def merge_dataspecs(a: DataSpec, b: DataSpec) -> DataSpecBuilder:
    """Merge two `DataSpec`s describing the same underlying attributes
    into one `DataSpecBuilder` with the union of nominal categories,
    numeric thresholds, and set values for attributes present in both
    (e.g. two different discretizations of the same numeric attribute).
    Attributes present in only one side are carried over as-is.

    Deliberately strict about anything that isn't a straightforward
    union: an attribute present in both with a different `AttributeType`,
    a different `Hierarchy`, or (for relational attributes) different
    `sources`/`expression` raises rather than guessing at a resolution --
    such conflicts should be reconciled by hand before merging.

    Returns a builder (not yet `.build()`ed) -- this only merges the
    *spec*; it doesn't attach data, and existing `Rule`s built against
    `a`/`b` don't automatically carry over to the merged feature space
    (their feature indices refer to the old spec(s), not this one).
    """
    builder = DataSpecBuilder()
    for name in sorted(set(a.attributes) | set(b.attributes)):
        attr_a = a.attributes.get(name)
        attr_b = b.attributes.get(name)
        if attr_a is not None and attr_b is not None:
            if _widens_to_nominal(attr_a, attr_b):
                _merge_as_nominal(builder, name, attr_a, attr_b)
                continue
            if attr_a.type != attr_b.type:
                raise ValueError(
                    f"Cannot merge attribute {name!r}: type mismatch "
                    f"({attr_a.type} vs {attr_b.type}) -- reconcile by hand before merging"
                )
            _merge_attribute(builder, name, attr_a, attr_b, a, b)
        else:
            attr, src = (attr_a, a) if attr_a is not None else (attr_b, b)
            _copy_attribute(builder, name, attr, src)
    return builder


def _rebuilt_with_negation(spec: DataSpec, negation: bool) -> DataSpec:
    """Rebuild `spec` from its typed attributes, in their original order,
    with negation features forced on (`True`) or off (`False`).
    `spec.attributes` must be populated -- an attribute-less DataSpec has
    nothing to rebuild, and its callers handle that case separately.
    SET / HIERARCHICAL / RELATIONAL attributes have no negation features
    either way, so the flag is a no-op for them."""
    builder = DataSpecBuilder(negation=negation)
    for name, attr in spec.attributes.items():
        t = attr.type
        if t == AttributeType.BOOLEAN:
            builder.add_boolean(name, missing_values=attr.missing_values)
        elif t == AttributeType.BINARY:
            builder.add_binary(name, list(attr.domain), missing_values=attr.missing_values)
        elif t == AttributeType.NOMINAL:
            builder.add_nominal(name, list(attr.domain), missing_name=attr.missing_name,
                                missing_values=attr.missing_values)
        elif t == AttributeType.NUMERIC:
            builder.add_numeric(name, list(attr.domain) if attr.domain else (),
                                le_thresholds=list(attr.le_domain) if attr.le_domain else (),
                                missing_name=attr.missing_name, missing_values=attr.missing_values)
        elif t == AttributeType.SET:
            builder.add_set(name, list(attr.domain), missing_values=attr.missing_values)
        elif t == AttributeType.HIERARCHICAL:
            builder._add_hierarchy(name, attr.hierarchy, missing_values=attr.missing_values)
        elif t == AttributeType.RELATIONAL:
            fs = spec.feature_spec(spec.feature_index(name))
            builder.add_relational(name, list(attr.sources), expression=fs.expression)
        else:
            raise ValueError(f"Unknown AttributeType {t!r} for attribute {name!r}")
    return builder.build(name=spec.name, missing_strategy=spec.missing_strategy)


def _with_negation_spec(spec: DataSpec) -> DataSpec:
    """The negation-augmented twin of `spec` -- via the typed attributes
    when it has them, otherwise by treating each plain feature as a
    Boolean attribute and adding its ``not {name}`` partner."""
    if spec.attributes:
        return _rebuilt_with_negation(spec, negation=True)
    builder = DataSpecBuilder(negation=True)
    for nm in spec.feature_names:
        builder.add_boolean(nm)
    return builder.build(name=spec.name, missing_strategy=spec.missing_strategy)


def _copy_attribute(builder: DataSpecBuilder, name: str, attr: Attribute, src: DataSpec) -> None:
    """Add `attr` (from `src`, one-sided) to `builder` unchanged,
    including its `missing_name`/`missing_values` if any."""
    if attr.type == AttributeType.BOOLEAN:
        builder.add_boolean(name, missing_values=attr.missing_values, negation=attr.negation)
    elif attr.type == AttributeType.BINARY:
        builder.add_binary(name, list(attr.domain), missing_values=attr.missing_values)
    elif attr.type == AttributeType.NOMINAL:
        builder.add_nominal(name, list(attr.domain), negation=attr.negation,
                             missing_name=attr.missing_name, missing_values=attr.missing_values)
    elif attr.type == AttributeType.NUMERIC:
        builder.add_numeric(name, list(attr.domain) if attr.domain else (), negation=attr.negation,
                             le_thresholds=list(attr.le_domain) if attr.le_domain else (),
                             missing_name=attr.missing_name, missing_values=attr.missing_values)
    elif attr.type == AttributeType.SET:
        builder.add_set(name, list(attr.domain), missing_values=attr.missing_values)
    elif attr.type == AttributeType.HIERARCHICAL:
        builder._add_hierarchy(name, attr.hierarchy, missing_values=attr.missing_values)
    elif attr.type == AttributeType.RELATIONAL:
        spec = src.feature_spec(src.feature_index(name))
        builder.add_relational(name, list(attr.sources), expression=spec.expression)
    else:
        raise ValueError(f"Unknown AttributeType {attr.type!r} for attribute {name!r}")


def _merged_missing_name(name: str, attr_a: Attribute, attr_b: Attribute) -> Optional[Any]:
    """Reconcile two attributes' `missing_name`: identical or one-sided
    is fine, a genuine conflict (both set, and different) raises rather
    than picking one silently."""
    if attr_a.missing_name is not None and attr_b.missing_name is not None \
            and attr_a.missing_name != attr_b.missing_name:
        raise ValueError(
            f"Cannot merge attribute {name!r}: different missing_name values "
            f"({attr_a.missing_name!r} vs {attr_b.missing_name!r}) -- reconcile by hand before merging."
        )
    return attr_a.missing_name if attr_a.missing_name is not None else attr_b.missing_name


def _widens_to_nominal(attr_a: Attribute, attr_b: Attribute) -> bool:
    """A BINARY attribute merged with a NOMINAL one, or with a BINARY one
    over different values: the union no longer is a closed two-value
    set, so the result is NOMINAL (see `_merge_as_nominal`)."""
    kinds = {attr_a.type, attr_b.type}
    if AttributeType.BINARY not in kinds or not kinds <= {AttributeType.BINARY, AttributeType.NOMINAL}:
        return False
    return kinds != {AttributeType.BINARY} or set(attr_a.domain) != set(attr_b.domain)


def _merge_as_nominal(builder: DataSpecBuilder, name: str, attr_a: Attribute, attr_b: Attribute) -> None:
    """The NOMINAL union of a BINARY attribute with a NOMINAL one (or
    with a BINARY one over different values). Negation follows the
    NOMINAL side(s), else the builder's default."""
    nominal = [a for a in (attr_a, attr_b) if a.type == AttributeType.NOMINAL]
    builder.add_nominal(
        name, sorted(set(attr_a.domain) | set(attr_b.domain), key=str),
        negation=any(a.negation for a in nominal) if nominal else None,
        missing_name=_merged_missing_name(name, attr_a, attr_b),
        missing_values=tuple(sorted(set(attr_a.missing_values) | set(attr_b.missing_values), key=str)),
    )


def _merge_attribute(
    builder: DataSpecBuilder, name: str, attr_a: Attribute, attr_b: Attribute, a: DataSpec, b: DataSpec
) -> None:
    """Add the union (or, for hierarchical/relational, the strictly-equal
    shared form) of `attr_a`/`attr_b` -- both sides already confirmed to
    share the same `AttributeType` -- to `builder`. `missing_values` are
    unioned; `missing_name` must agree if both sides declare one (see
    `_merged_missing_name`)."""
    t = attr_a.type
    missing_values = tuple(sorted(set(attr_a.missing_values) | set(attr_b.missing_values), key=str))
    negation = attr_a.negation or attr_b.negation
    if t == AttributeType.BOOLEAN:
        builder.add_boolean(name, missing_values=missing_values, negation=negation)
    elif t == AttributeType.BINARY:  # same two values -- see _widens_to_nominal
        builder.add_binary(name, list(attr_a.domain), missing_values=missing_values)
    elif t == AttributeType.NOMINAL:
        builder.add_nominal(name, sorted(set(attr_a.domain) | set(attr_b.domain)), negation=negation,
                             missing_name=_merged_missing_name(name, attr_a, attr_b),
                             missing_values=missing_values)
    elif t == AttributeType.NUMERIC:
        ge = list(attr_a.domain or ()) + list(attr_b.domain or ())
        le = list(attr_a.le_domain or ()) + list(attr_b.le_domain or ())
        builder.add_numeric(name, ge, le_thresholds=le, negation=negation,
                             missing_name=_merged_missing_name(name, attr_a, attr_b),
                             missing_values=missing_values)
    elif t == AttributeType.SET:
        builder.add_set(name, sorted(set(attr_a.domain) | set(attr_b.domain)), missing_values=missing_values)
    elif t == AttributeType.HIERARCHICAL:
        if set(attr_a.hierarchy.edges) != set(attr_b.hierarchy.edges):
            raise ValueError(
                f"Cannot merge attribute {name!r}: hierarchies differ -- merging "
                "different hierarchy structures isn't supported; reconcile them "
                "by hand before merging."
            )
        builder._add_hierarchy(name, attr_a.hierarchy, missing_values=missing_values)
    elif t == AttributeType.RELATIONAL:
        spec_a = a.feature_spec(a.feature_index(name))
        spec_b = b.feature_spec(b.feature_index(name))
        if attr_a.sources != attr_b.sources or spec_a.expression != spec_b.expression:
            raise ValueError(
                f"Cannot merge attribute {name!r}: relational attributes differ in "
                "sources/expression -- reconcile them by hand before merging."
            )
        builder.add_relational(name, list(attr_a.sources), expression=spec_a.expression)
    else:
        raise ValueError(f"Unknown AttributeType {t!r} for attribute {name!r}")
