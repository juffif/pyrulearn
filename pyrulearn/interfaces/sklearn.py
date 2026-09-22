"""
pyrulearn.interfaces.sklearn
================================

This package's two-way relationship with scikit-learn.

`RuleSetClassifier` wraps a `pyrulearn.models.FlatRuleSet` (or any
`RuleModel`) as a proper `sklearn.base.BaseEstimator`/`ClassifierMixin`,
so it can be used in `cross_val_score`, `GridSearchCV`, sklearn
`Pipeline`s, etc. This is what makes scikit-learn a hard dependency of
`pyrulearn` overall (not just an interface you opt into).

(`RuleSetClassifier` is named for the compatibility layer it lives in,
not the other way around -- a naming pass on the class itself is
expected later.)

The rest of this module is the other direction -- `ObjectRuleImporter`s
that extract rules *out of* an already-fitted sklearn model:
`SklearnTreeImporter` for a single `sklearn.tree.DecisionTreeClassifier`,
`RandomForestImporter` for a `sklearn.ensemble.RandomForestClassifier`
(further tree ensembles -- `GradientBoostingClassifier`,
`AdaBoostClassifier`, ... -- belong in this same module too, when added,
since they'd share the same per-tree leaf-extraction logic,
`_rules_from_tree` below -- grouped by shared implementation, not one
file per algorithm).

`RandomForestImporter` combines every tree's rules into one flat
`RuleSet`, not a sequence of one `DisjointRuleSet` per tree: each tree
is individually exhaustive+disjoint, so a k-tree forest always has
exactly k covering rules per example, and `pyrulearn.combiners` is what
turns that into forest-style prediction -- `CountVoteCombiner` for hard
majority voting (or `HeuristicVoteCombiner` to weight that vote by a
`pyrulearn.heuristics.RuleHeuristic` score instead of counting every
rule equally), or `DistributionCombiner`'s `MacroVoteCombiner` for
soft, per-class-probability voting -- the one that actually matches
sklearn's own `RandomForestClassifier.predict()` mechanism, using each
leaf's own *measured* stats (pass `data=` to `import_model`) rather
than a single scalar weight.

`DecisionTree`/`RandomForest` are the `pyrulearn.learners.ExternalRuleLearner`
counterparts of `SklearnTreeImporter`/`RandomForestImporter` -- thin
wrappers that fit a fresh sklearn estimator on a `DataRepresentation`
and hand it to the importer right next to them, so a comparative
experiment can drive this algorithm the same way as any other learner
(``learner.fit(rep)``) without the caller fitting sklearn and calling
the importer by hand. Deliberately *not* a separate hierarchy: reusing
`IMPORTER` means there's no second copy of the leaf-extraction logic to
keep in sync.

Also `tree_thresholds`, a numeric-discretization helper used by
`pyrulearn.data.io` (fits a single-feature decision tree and harvests
its split points) -- unrelated to rule *import* as such, but it reuses
the same sklearn tree machinery, so it lives here rather than forcing a
separate module just for one function.

Both importers also implement `infer_dataspec` (the base class's
generic hook, see `pyrulearn.interfaces.base`): fit `model` directly on
raw, un-pre-binarized input -- numeric, already-encoded categorical, or
a mix, whatever sklearn itself can be fit on -- and this discovers a
`DataSpec` from the thresholds the tree(s) actually used, rather than
requiring them fixed in advance by a separate, weaker discretizer --
see `_rules_from_tree`'s docstring for exactly how. This is
`pyrulearn.learners`'s "workflow 2": since `data.X` (the only
existing `DataRepresentation`, `BooleanDataRepresentation`, casts to
bool in its own constructor) can't hold raw data, workflow 2 doesn't go
through `fit`/`prepare` (workflow-1-only) at all -- call
`DecisionTree(**params).fit_external(raw_X, y, feature_names=names)`
directly, then `importer.infer_dataspec(model, names)`, then
`importer.import_model(model, ds, feature_names=names)`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .base import ObjectRuleImporter, register_importer
from ..combiners import RuleCombiner
from ..data import BooleanDataRepresentation, DataRepresentation
from ..data import DataSpec, DataSpecBuilder
from ..learners import ExternalRuleLearner, RelabelingExternalLearner
from .. import models
from ..models import FlatRuleSet, RuleModel
from ..rule import Rule


class RuleSetClassifier(BaseEstimator, ClassifierMixin):
    """scikit-learn-compatible classifier wrapping a `pyrulearn.models`
    rule model.

    Two usage modes:

    1. **Fixed rule set** -- pass `rules=` at construction; `fit` just
       annotates the given data (useful for evaluating externally-mined
       rules via `cross_val_score`).
    2. **Learner callback** -- pass `learner=` a callable
       `(DataRepresentation) -> RuleModel` that mines rules from
       training data; `fit` calls it fresh each time, so this plugs any
       rule-learning algorithm into sklearn's model-selection tools.

    `fit`/`predict` accept a pre-built `DataRepresentation` of any
    concrete type; a raw array/DataFrame is wrapped in a
    `BooleanDataRepresentation`.

    Parameters
    ----------
    rules : RuleModel or list[Rule], optional
        A ready `RuleModel` (any `pyrulearn.models` type) is used as-is;
        a plain list of `Rule`s is wrapped in a `FlatRuleSet`.
    learner : callable, optional
    default_class : Any, optional
        Prediction for examples covered by no rule. Set as the resulting
        `FlatRuleSet`'s `default_prediction` (a bare label) at fit time.
    combiner : str or RuleCombiner, optional
        How to combine multiple covering rules -- see `FlatRuleSet.predict`.
    """

    def __init__(
        self,
        rules: Optional[Any] = None,
        learner: Optional[Callable[[DataRepresentation], RuleModel]] = None,
        default_class: Any = None,
        combiner: Union[str, RuleCombiner] = "max",
    ):
        if rules is None and learner is None:
            raise ValueError("Provide either `rules` or `learner`")
        self.rules = rules
        self.learner = learner
        self.default_class = default_class
        self.combiner = combiner
        self.rule_set_: Optional[RuleModel] = None
        self.classes_: Optional[np.ndarray] = None

    # get_params / set_params / __sklearn_tags__ / clone() are all inherited
    # from BaseEstimator (which introspects __init__'s signature), so we
    # don't redefine them here.

    def fit(self, X, y=None):
        rep = X if isinstance(X, DataRepresentation) else BooleanDataRepresentation.from_xy(X, y)
        if self.learner is not None:
            self.rule_set_ = self.learner(rep)
        else:
            self.rule_set_ = self.rules if isinstance(self.rules, RuleModel) else FlatRuleSet(self.rules)
        if self.default_class is not None:
            self.rule_set_.default_prediction = self.default_class
        self.rule_set_.annotate(rep)
        if rep.y is not None:
            self.classes_ = np.unique(rep.y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.rule_set_ is None:
            raise RuntimeError("Call fit() before predict()")
        rep = X if isinstance(X, DataRepresentation) else BooleanDataRepresentation.from_xy(X)
        return self.rule_set_.predict(rep, combiner=self.combiner)

    def score(self, X, y) -> float:
        preds = self.predict(X)
        y = np.asarray(y)
        return float(np.mean(preds == y))


# ===================================================== tree importers =========

def _thresholds_by_column(tree) -> Dict[int, set]:
    """Every distinct threshold actually used at an internal split node
    of a fitted sklearn `Tree`, grouped by raw column index -- the
    "what did this tree actually look at" a `DataSpec` built via
    `infer_dataspec` needs, as opposed to a fixed cap chosen ahead of
    time by a separate discretizer (see `tree_thresholds`, below, for
    that other use case). `feature[i] == -2` (sklearn's
    `TREE_UNDEFINED`) marks a leaf -- no split there.
    """
    by_col: Dict[int, set] = {}
    for feat, threshold in zip(tree.feature, tree.threshold):
        if feat == -2:
            continue
        by_col.setdefault(int(feat), set()).add(float(threshold))
    return by_col


def _dataspec_from_thresholds(by_col: Dict[int, set], feature_names: Sequence[str]) -> DataSpec:
    """Shared `infer_dataspec` body for `SklearnTreeImporter`/
    `RandomForestImporter`: one NUMERIC attribute per raw column that
    was actually split on somewhere (`add_numeric`, with exactly the
    thresholds discovered), named via `feature_names[i]`. A column never
    split on contributes no attribute at all -- there's nothing to build
    a feature from, and no rule can ever reference it anyway.
    """
    # negation on: a tree's "left" branch (attr <= threshold) becomes a
    # positive literal on the paired "attr < threshold" feature.
    builder = DataSpecBuilder(negation=True)
    for i, name in enumerate(feature_names):
        thresholds = sorted(by_col.get(i, ()))
        if thresholds:
            builder.add_numeric(name, thresholds)
    return builder.build()


def _boolean_dataspec_with_negation(names: Sequence[str]) -> DataSpec:
    """A Boolean `DataSpec` with, for every already-Boolean tree column,
    a positive feature and its paired negation -- a tree's "left" branch
    (``column <= 0.5``) becomes a positive literal on that negation
    feature, so rules need no negative literals of their own.
    """
    builder = DataSpecBuilder(negation=True)
    for name in names:
        builder.add_boolean(name)
    return builder.build()


def _rules_from_tree(
    tree_model,
    dataspec: DataSpec,
    classes: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> List[Rule]:
    """One `Rule` per leaf of a fitted binary sklearn tree-based
    estimator (a plain `DecisionTreeClassifier`, or one entry from a
    `RandomForestClassifier`'s `estimators_`). Shared leaf-extraction
    logic -- no provenance tagging here, since that's caller-specific (a
    lone tree vs. one tree of many in a forest); callers call
    `self._stamp_rule_provenance(...)` themselves.

    A tree "left" branch (``feature <= threshold``) means the feature is
    absent; since rules carry no negative literals it becomes a positive
    literal on that feature's paired negation feature, so `dataspec` must
    be negation-enabled (`DataSpecBuilder`'s default) -- `from_sklearn_tree`
    / `from_random_forest` build one that way automatically.

    Two modes, chosen by whether `feature_names` is given:

    - Omitted (default): the tree's own -- already-Boolean -- training
      column `j` is `dataspec`'s `j`-th *positive* feature (its own index
      directly when `dataspec` has exactly one feature per column; every
      other index when `dataspec` also carries the paired negation
      features), for a `dataspec` built by hand or via
      `pyrulearn.data.io.build_dataspec`/`binarize` *before* fitting.
    - Given: `feature_names[i]` names the tree's raw column `i` (its
      *original* attribute, not a pre-derived Boolean one); each split's
      real threshold is looked up as `f"{attribute}>={threshold}"` in
      `dataspec` (or the bare name, for an already-Boolean column) --
      built by `infer_dataspec` from this *same* fitted model, so every
      threshold the tree actually used is guaranteed to already be there.

    `classes` overrides `tree_model.classes_` -- needed for
    `RandomForestImporter`: sklearn encodes labels once at the *forest*
    level and fits each individual tree on those encoded values, so
    ``rf.estimators_[i].classes_`` is just ``[0., 1., ...]``, not the
    original labels -- the caller must pass the *forest's* `classes_`
    explicitly to get real target values back.

    Each leaf's full per-class breakdown decides `target` (the majority
    class) but isn't kept beyond that -- `pyrulearn.combiners.
    DistributionCombiner` (micro/macro vote, matching a forest's own
    soft-voting `predict()`) reads a rule's *measured* `stats()` instead,
    which needs `data=` passed through to `import_model`/
    `RandomForestImporter` (see their docstrings).
    """
    tree = tree_model.tree_
    if feature_names is None and tree.n_features not in (dataspec.n_features, dataspec.n_features // 2):
        raise ValueError(
            f"dataspec has {dataspec.n_features} features but the tree was trained "
            f"on {tree.n_features} -- they must describe the same Boolean feature space, "
            "one positive feature per tree column (a negation-enabled dataspec may also "
            "carry the paired negation features) "
            "(pass feature_names= if dataspec was built via infer_dataspec instead)"
        )
    if feature_names is not None and tree.n_features != len(feature_names):
        raise ValueError(
            f"feature_names has {len(feature_names)} entries but the tree was trained "
            f"on {tree.n_features} columns"
        )
    classes = classes if classes is not None else tree_model.classes_
    rules: List[Rule] = []

    # column-index mode: the tree's raw column `j` is the `j`-th *positive*
    # feature of `dataspec` (its own index directly, for a plain Boolean
    # dataspec; every other feature -- 2*j, for a negation-enabled one
    # built one add_boolean per column -- when negations are interleaved).
    positive_feats = [
        i for i, s in enumerate(dataspec.feature_specs)
        if s.op in (None, "==", ">=", "<=")
    ]
    use_positive_map = feature_names is None and len(positive_feats) == tree.n_features

    def recurse(node: int, pos: list, neg: list):
        if tree.children_left[node] == tree.children_right[node] == -1:
            # leaf -- majority class. tree.value[node] holds normalized
            # per-class proportions, but argmax is scale-invariant, so no
            # need to rescale by node sample weight just to pick a target
            # (DistributionCombiner's own micro/macro leaf-size distinction
            # comes from each rule's *measured* stats(), not from here).
            target = classes[int(np.argmax(tree.value[node][0]))]
            rules.append(Rule.from_pos_neg(
                pos=tuple(pos), neg=tuple(neg), target=target, dataspec=dataspec,
            ))
            return
        feat = tree.feature[node]
        if feature_names is None:
            feat_idx = positive_feats[feat] if use_positive_map else feat
        else:
            attr_name = feature_names[feat]
            threshold = float(tree.threshold[node])
            ge_name = f"{attr_name}>={threshold}"
            # numeric attribute if a matching threshold feature exists,
            # otherwise the column was already Boolean and named directly
            feat_idx = (dataspec.feature_index(ge_name)
                        if ge_name in dataspec.feature_names
                        else dataspec.feature_index(attr_name))
        # left child = "feature <= 0.5" / "attr <= threshold" => our feature is False
        recurse(tree.children_left[node], pos, neg + [feat_idx])
        # right child = "feature > 0.5" / "attr > threshold" => our feature is True
        recurse(tree.children_right[node], pos + [feat_idx], neg)

    recurse(0, [], [])
    return rules


class SklearnTreeImporter(ObjectRuleImporter):
    """Extracts one `Rule` per leaf from a fitted binary
    `sklearn.tree.DecisionTreeClassifier`. See `from_sklearn_tree` for
    the ergonomic wrapper most callers want; use this class directly if
    you want to inspect/reuse the importer instance itself (e.g.
    alongside other `RuleImporter`s).

    Two ways to get a matching `dataspec`: build one in advance (by hand,
    or via `pyrulearn.data.io.build_dataspec`/`binarize` before fitting
    -- the tree only ever sees already-Boolean columns then) and pass it
    to `import_model` as before; or fit the tree directly on raw,
    un-pre-binarized input (numeric, already-encoded categorical, or a
    mix) and call `infer_dataspec` afterward, which discovers the tree's
    *actual* thresholds instead of requiring them decided beforehand by
    a separate, weaker discretizer.
    """

    SOURCE = "sklearn.tree.DecisionTreeClassifier"

    def import_model(self, model, dataspec: DataSpec, feature_names: Optional[Sequence[str]] = None,
                     data: Optional[DataRepresentation] = None) -> "models.DisjointRuleSet":
        """Without `feature_names`: `dataspec`'s feature order must
        match the Boolean columns `model` was trained on. With it:
        `dataspec` must be one `infer_dataspec` built from this same
        `model` and `feature_names` (or an equivalent) -- see
        `_rules_from_tree`'s docstring for the full distinction. Rules
        bind to `dataspec` directly either way.

        Returns a `pyrulearn.models.DisjointRuleSet`, not a plain
        `FlatRuleSet`: a decision tree's leaves are pairwise disjoint by
        construction (every example reaches exactly one leaf), so that
        guarantee is free to declare here.
        """
        rules = _rules_from_tree(model, dataspec, feature_names=feature_names)
        max_depth = model.get_params().get("max_depth")
        rules = self._stamp_rule_provenance(rules, max_depth=max_depth)
        rules = self._stamp_rule_stats(rules, data)
        return self._stamp_provenance(models.DisjointRuleSet(rules), max_depth=max_depth)

    def infer_dataspec(self, model, feature_names: Sequence[str]) -> DataSpec:
        """Discover a NUMERIC attribute per raw column `model`'s fitted
        tree actually split on, with exactly the thresholds it actually
        used -- no cap chosen ahead of time, no separate discretizer.
        Pass the result (and the same `feature_names`) to `import_model`.
        """
        return _dataspec_from_thresholds(_thresholds_by_column(model.tree_), feature_names)


def from_sklearn_tree(
    tree_clf,
    feature_names: Optional[Sequence[str]] = None,
    dataspec: Optional[DataSpec] = None,
) -> "models.DisjointRuleSet":
    """Convenience wrapper around `SklearnTreeImporter`.

    Prefer passing `dataspec=` when you already have one (e.g. the one
    you trained `tree_clf` from) -- rules bind to it directly, so they
    get its typed display/constraints and share its identity with the
    rest of your pipeline (no separate object to `Rule.remap` later).
    If `dataspec` is omitted, one is built from `feature_names` (or
    ``f0, f1, ...`` if that's omitted too).
    """
    if dataspec is None:
        names = list(feature_names) if feature_names is not None else \
            [f"f{i}" for i in range(tree_clf.tree_.n_features)]
        dataspec = _boolean_dataspec_with_negation(names)
    return SklearnTreeImporter().import_model(tree_clf, dataspec)


register_importer("sklearn_tree", SklearnTreeImporter)


class DecisionTree(RelabelingExternalLearner):
    """`ExternalRuleLearner` for `sklearn.tree.DecisionTreeClassifier`:
    fits a fresh tree with `**params` (the usual `DecisionTreeClassifier`
    constructor arguments, e.g. `max_depth=`, `random_state=`), then
    extracts it via `SklearnTreeImporter`. `prepare` isn't overridden --
    sklearn's native input format already is `data.X`.

    ``DecisionTree(**params).fit(data)`` (workflow 1, an
    existing `DataSpec` + already-Boolean data) is the uniform-`fit`
    counterpart to calling `from_sklearn_tree` by hand -- it produces a
    `models.DisjointRuleSet` (a lone tree's leaves are pairwise disjoint).
    A single sklearn tree is already inherently multi-class, but
    `DecomposingLearner` is mixed in anyway so `fit(data,
    model=ConceptSet | ConceptCascade | PairwiseModel)` can build a
    one-tree-per-class decomposition, which is a genuinely different
    model. For workflow 2 (fit directly on raw native data, discover the
    `DataSpec` afterward), call `fit_external`/
    `SklearnTreeImporter.infer_dataspec` directly instead -- see
    `pyrulearn.learners`'s module docstring.
    """

    IMPORTER = SklearnTreeImporter
    NATIVE_MODEL = models.DisjointRuleSet

    def __init__(self, **params):
        self.params = params

    def fit_external(self, X, y, feature_names=None):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**self.params).fit(X, y)


class RandomForestImporter(ObjectRuleImporter):
    """Extracts every tree's leaves from a fitted
    `sklearn.ensemble.RandomForestClassifier` and combines them into one
    flat `RuleSet` -- not a sequence of one `DisjointRuleSet` per tree.
    Each tree is individually exhaustive+disjoint, so the combined
    `RuleSet` always has exactly `n_estimators` rules covering any given
    example; `pyrulearn.combiners.CountVoteCombiner` on that `RuleSet`
    reproduces hard forest-style voting at prediction time (plain
    majority vote; `HeuristicVoteCombiner` for a heuristic-weighted
    one), while `MacroVoteCombiner` reproduces *soft* voting -- the
    mechanism sklearn's own `.predict()` actually uses -- via each
    leaf's own measured stats. Pass `data=` (see below) so those stats
    exist -- `DistributionCombiner` raises otherwise. No `default_rule`
    is needed: since every tree always fires, the combined set is always
    exhaustive too.

    See `from_random_forest` for the ergonomic wrapper most callers
    want; use this class directly if you want to inspect/reuse the
    importer instance itself.
    """

    SOURCE = "sklearn.ensemble.RandomForestClassifier"

    def import_model(self, model, dataspec: DataSpec, feature_names: Optional[Sequence[str]] = None,
                     data: Optional[DataRepresentation] = None) -> "models.FlatRuleSet":
        """Without `feature_names`: `dataspec`'s feature order must
        match the Boolean columns `model` was trained on. With it:
        `dataspec` must be one `infer_dataspec` built from this same
        `model` and `feature_names` -- see `_rules_from_tree`'s
        docstring for the full distinction. Every extracted rule binds
        to `dataspec` directly either way; each rule's `.provenance.
        params` additionally records which tree (`tree_index`) it came
        from. Pass `data=` (the training `DataRepresentation`) to measure
        each leaf's own stats -- needed for `DistributionCombiner`
        (`"micro_vote"`/`"macro_vote"`/`"micro_max"`/`"macro_max"`).
        """
        max_depth = model.get_params().get("max_depth")
        n_estimators = len(model.estimators_)
        all_rules: List[Rule] = []
        for i, estimator in enumerate(model.estimators_):
            # model.classes_, not estimator.classes_ -- see _rules_from_tree's docstring
            tree_rules = _rules_from_tree(estimator, dataspec, classes=model.classes_, feature_names=feature_names)
            tree_rules = self._stamp_rule_provenance(
                tree_rules, tree_index=i, n_estimators=n_estimators, max_depth=max_depth,
            )
            tree_rules = self._stamp_rule_stats(tree_rules, data)
            all_rules.extend(tree_rules)
        return self._stamp_provenance(models.FlatRuleSet(all_rules), n_estimators=n_estimators, max_depth=max_depth)

    def infer_dataspec(self, model, feature_names: Sequence[str]) -> DataSpec:
        """Discover a NUMERIC attribute per raw column *any* tree in the
        fitted forest actually split on, with the *union* of thresholds
        actually used across all of them (so every tree's rules can
        share one feature space) -- no cap chosen ahead of time, no
        separate discretizer. Pass the result (and the same
        `feature_names`) to `import_model`.
        """
        by_col: Dict[int, set] = {}
        for estimator in model.estimators_:
            for col, thresholds in _thresholds_by_column(estimator.tree_).items():
                by_col.setdefault(col, set()).update(thresholds)
        return _dataspec_from_thresholds(by_col, feature_names)


def from_random_forest(
    rf_clf,
    feature_names: Optional[Sequence[str]] = None,
    dataspec: Optional[DataSpec] = None,
) -> "models.FlatRuleSet":
    """Convenience wrapper around `RandomForestImporter` -- same
    `dataspec=`/`feature_names=` resolution as `from_sklearn_tree`."""
    if dataspec is None:
        n_features = rf_clf.estimators_[0].tree_.n_features
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]
        dataspec = _boolean_dataspec_with_negation(names)
    return RandomForestImporter().import_model(rf_clf, dataspec)


register_importer("random_forest", RandomForestImporter)


class RandomForest(ExternalRuleLearner):
    """`ExternalRuleLearner` for `sklearn.ensemble.RandomForestClassifier`:
    fits a fresh forest with `**params` (the usual
    `RandomForestClassifier` constructor arguments, e.g. `n_estimators=`,
    `max_depth=`, `random_state=`), then extracts it via
    `RandomForestImporter`. `prepare` isn't overridden -- sklearn's
    native input format already is `data.X`.

    ``RandomForest(**params).fit(data)`` (workflow 1) is the
    uniform-`fit` counterpart to calling `from_random_forest` by hand --
    it produces a `models.EnsembleModel` with one `DisjointRuleSet` member
    per tree (its honest structure: a bagged vote over per-tree leaf
    partitions; a forest is inherently multi-class). `fit(data,
    model=FlatRuleSet)` gives the flattened single-bag view instead (via
    `ensemblemodel_to_flatruleset` -- an unweighted plurality vote over
    the pooled leaves, *not* sklearn's soft per-tree-distribution vote).
    See `DecisionTree`'s docstring for workflow 2.
    """

    IMPORTER = RandomForestImporter
    NATIVE_MODEL = models.EnsembleModel

    def __init__(self, **params):
        self.params = params

    def _fit_native(self, data):
        """One `DisjointRuleSet` per tree (each tree's leaves partition
        the space), combined into an `EnsembleModel`'s flat vote -- built
        here rather than taken as-is from `RandomForestImporter.
        import_model`, which returns one plain `FlatRuleSet` bag with no
        per-tree structure. Trees are recovered from each rule's own
        `.provenance.params["tree_index"]`
        (`RandomForestImporter` gives every leaf a fresh `Provenance` per
        rule, sharing the same `tree_index` value within one tree)."""
        imported = self._import(data)
        groups: Dict[int, List[Rule]] = {}
        for r in imported.rules:
            groups.setdefault(r.provenance.params["tree_index"], []).append(r)
        members = [models.DisjointRuleSet(rs) for _, rs in sorted(groups.items())]
        return models.EnsembleModel(members)

    def fit_external(self, X, y, feature_names=None):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**self.params).fit(X, y)


def _prettify_threshold(t: float, sorted_unique: np.ndarray) -> float:
    """Replace a computed split point with the coarsest (fewest decimal
    digits) number that still falls in the same open gap between two
    *actually observed* values of the column -- so the resulting
    ``>=``/``<`` test partitions the real data identically, just without
    ugly floating-point artifacts. sklearn's tree computes each
    threshold as the exact midpoint of two adjacent values in whatever
    (possibly already-split) subset reached that node, e.g.
    ``(15.1458 + 15.2) / 2 == 15.172900199890137`` -- a value nobody
    would write down by hand, and no more "correct" than ``15.17`` for
    the same purpose: any number strictly between the two nearest real
    values on *this* column produces the exact same split of the full
    dataset, so rounding here changes nothing but the display.
    """
    below = sorted_unique[sorted_unique < t]
    above = sorted_unique[sorted_unique > t]
    lo = below[-1] if len(below) else -np.inf
    hi = above[0] if len(above) else np.inf
    for decimals in range(0, 10):
        candidate = round(t, decimals)
        if lo < candidate < hi:
            return candidate
    return t  # every rounding collided with a real value -- keep the exact one


def tree_thresholds(values: np.ndarray, y: np.ndarray, max_intervals: int = 8) -> List[float]:
    """Supervised numeric discretization: fit a single-feature decision
    tree against `values`/`y` and return its internal split points as
    ascending thresholds, prettified (see `_prettify_threshold`) against
    the column's own observed values.

    The default mirrors `pyrulearn.data.io.DEFAULT_MAX_INTERVALS` (kept
    as a separate literal here rather than imported, so this module
    doesn't pick up a pandas dependency just to read a constant).

    `max_intervals` is a ceiling on the number of resulting buckets
    (``max_intervals - 1`` thresholds at most), not a target -- it's
    passed straight through as the tree's `max_leaf_nodes`, so the
    tree's own splitting criterion decides how much of that budget the
    data actually justifies; a column with little signal may come back
    with fewer thresholds, or none.
    """
    from sklearn.tree import DecisionTreeClassifier

    flat_values = np.asarray(values, dtype=float).ravel()
    sorted_unique = np.unique(flat_values)
    y = np.asarray(y)
    tree = DecisionTreeClassifier(max_leaf_nodes=max(2, max_intervals))
    tree.fit(flat_values.reshape(-1, 1), y)
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    # feature[i] == -2 (sklearn's TREE_UNDEFINED) marks a leaf, i.e. no split there.
    raw = sorted(float(t) for f, t in zip(feature, threshold) if f != -2)
    return [_prettify_threshold(t, sorted_unique) for t in raw]
