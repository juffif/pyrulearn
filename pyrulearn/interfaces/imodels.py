"""
pyrulearn.interfaces.imodels
=====================================

`ObjectRuleImporter`s for the `imodels` package's rule models --
`BayesianRuleListImporter` for `imodels.BayesianRuleListClassifier`
(Letham et al.'s Bayesian Rule Lists) and `BayesianRuleSetImporter` for
`imodels.BayesianRuleSetClassifier` (the "BOA"/Wang et al. Bayesian
Or-of-And algorithm), and `RuleFitImporter` for
`imodels.RuleFitClassifier` (Friedman & Popescu's RuleFit, see its own
section at the end); `imodels`' other rule models
(`BoostedRulesClassifier`, ...) belong in this same module when added,
per this package's group-by-shared-implementation convention.

**The first importer producing a `DecisionList`, not a `FlatRuleSet`.** A
Bayesian Rule List is genuinely an ordered decision list -- its own
`__str__` prints ``IF ... THEN ... ELSE IF ... THEN ... ELSE ...`` --
so first-match-wins ordering is the semantics, not an artifact:
`pyrulearn.models.DecisionList` is the exact match, with the trailing
ELSE becoming the list's `default_prediction`.

Input requirements, confirmed by reading `BayesianRuleListClassifier.
fit`'s own source rather than assumed:

- **Already-Boolean features -- categorical ones included.** `fit`
  raises ``ValueError("All numeric features must be discretized prior
  to fitting!")`` unless every entry of `X` is 0 or 1. Checked
  directly, that rules out *both* raw string categoricals (which fail
  even earlier, at `check_X_y`: ``could not convert string to float``)
  and ordinal/label-encoded ones (0/1/2/... hits the same
  discretization error) -- so a nominal attribute must be **one-hot**
  encoded before fitting, which is exactly what
  `pyrulearn.data.io.build_dataspec`/`binarize` already produce for a
  NOMINAL attribute (one ``color=red``/``color=green``/... feature
  each, via `DataSpecBuilder.add_nominal`). A
  `pyrulearn.data.BooleanDataRepresentation` is therefore
  already in exactly the right shape.

  Unlike `pyrulearn.interfaces.wittgenstein`, there is
  consequently no internal discretization to reverse-engineer and no
  numeric/nominal parsing here at all: `infer_dataspec` is inherited
  unchanged from `ObjectRuleImporter` (one plain Boolean feature per
  name), which is already correct for this model, and workflows 1 and 2
  (see `pyrulearn.learners`) collapse to the same thing. (`imodels`
  does ship its own discretizers -- `BRLDiscretizer`, `MDLPDiscretizer`
  -- but `BayesianRuleListClassifier` has no discretizer parameter and
  never applies one itself; pyrulearn's `build_dataspec`/`binarize`
  fills that role here, keeping the resulting features tied to a real
  `DataSpec`.)
- **Binary targets only.** `fit` raises ``ValueError("Only binary
  classification is supported at this time!")`` for anything else --
  properly, unlike wittgenstein's silent collapse (see
  `pyrulearn.interfaces.wittgenstein`'s module docstring), so no
  extra guard is needed on this side. There is no built-in one-vs-rest
  path either; genuine multi-class support would mean pyrulearn adding
  its own cascade on top, the same open question wittgenstein raises.

Two semantic details worth knowing when reading imported rules:

- **Rules carry probabilities, not labels.** `model.theta[i]` is that
  rule's posterior P(class 1) -- e.g. a rule with `theta=0.11` is a
  confident *negative* rule. Each imported `Rule`'s `target` is the
  *majority* class at that node (`classes_[1]` if `theta > 0.5`, else
  `classes_[0]`) -- BRL's own posterior/credible-interval numbers
  themselves aren't kept on the rule; `pyrulearn.combiners`'
  `HeuristicCombiner`s (and everything else that ranks/combines rules)
  read each rule's own *measured* `stats()` instead, populated by the
  `fit()`/`data=` round trip (see `_stamp_rule_stats` below).
- **Only positive literals.** BRL builds its antecedents from FP-growth
  itemsets, which are sets of *present* items, so every condition is
  "feature is True" -- an imported rule never contains a negated
  literal. Not a conversion limitation; it's what the algorithm
  searches over.

`BayesianRuleSetImporter` (for `imodels.BayesianRuleSetClassifier`,
a.k.a. BOA/"Bayesian Or-of-and") produces a plain `RuleSet`, not a
`RuleList`: it fits a genuinely *unordered* pattern set via simulated
annealing, and predicts OR-of-AND -- any one rule firing predicts the
positive class, `classes_[1]`, regardless of which -- so, unlike BRL,
rules from this model never actually disagree when several cover the
same example. Only the fallback needs a `default_prediction`
(`classes_[0]`) for examples no rule covers.

Confirmed by direct experimentation (`BayesianRuleSetClassifier`'s own
input validation is much thinner than BRL's, so its failure modes are
correspondingly less legible):

- **Boolean input, same as BRL, but not enforced with a clear error.**
  Fitting on continuous columns doesn't raise an informative message
  the way BRL's does -- it fails downstream with `AssertionError: Only
  0 potential rules found, change hyperparams to allow for more`, since
  its `1 - X` negation and support-based rule screening only make sense
  for already-0/1 columns. `BayesianRuleSet.fit_external` checks this
  itself before calling into `imodels`, raising a clearer error, since
  the library doesn't.
- **Binary targets only, but not enforced either.** A >2-class `y`
  isn't rejected up front -- it fails deep inside simulated annealing
  with an opaque `ValueError: math domain error` (a `log` of a
  quantity that only makes sense for a 0/1 target). Same fix:
  `BayesianRuleSet.fit_external` checks `y` has exactly two classes
  itself and raises a clear message before that can happen.
- **Items can be feature-present or feature-absent tests**, unlike BRL:
  each entry of `model.rules_` is a list of item names, each either a
  real feature name or that name with a `"_neg"` suffix -- `model`'s own
  `predict` builds exactly this `{name, name_neg}` column space before
  evaluating rules against it, so parsing does the same: try the item as
  a feature name directly before considering it a suffixed negation,
  matching `predict`'s own resolution order. A `"_neg"` item is
  translated to a positive literal on that feature's paired **negation
  feature** (via `DataSpec.negation_of`), so the imported rule still has
  no negative literals -- which means the bound `DataSpec` must be
  negation-enabled (`DataSpecBuilder`'s default, and what
  `build_dataspec`/`binarize` produce). (This inherits BRS's own ambiguity if a dataset
  genuinely has both a feature named e.g. `"foo"` and one named
  literally `"foo_neg"` -- not something this importer can resolve any
  better than the model itself already does.)
- **No natural per-rule weight.** Unlike BRL's `theta`, BRS's
  simulated-annealing search optimizes the *rule set* jointly, not a
  posterior per individual rule, so there's nothing principled to record
  beyond provenance and (via the `fit()` round trip) each rule's own
  measured stats, same as any other importer's rules. Moot for
  `HeuristicMaxCombiner`/`HeuristicVoteCombiner` either way, though,
  since firing rules never disagree in target to begin with.
- Feature names are looked up by **name** via `dataspec.feature_index`
  (`model.feature_names_`, real names throughout -- no `X_0`-style
  placeholder indirection the way BRL's `feature_placeholders` needs),
  same convention `pyrulearn.interfaces.wittgenstein` uses.

**Two separate, genuine bugs in `imodels` itself, discovered while
testing this importer against real 0/1 data** (neither is something
this module can fully work around):

- `BayesianRuleSetClassifier.fit`'s simulated-annealing "clean" move
  (in `_propose`, the branch that drops redundant rules once a
  candidate set already classifies training data perfectly) computes
  `remove.append(i)` for *positions* in `rules_norm`, then does
  `rules_norm.remove(x)` for `x` in that list -- which removes by
  *value*, not position. This raises a bare `ValueError:
  list.remove(x): x not in list` whenever the annealing search happens
  to reach a perfectly-classifying candidate set at all -- nothing to
  do with whether the input was properly prepared. Worth reporting
  upstream.
- `fit` mixes `np.random` (seeded from its own `random_state=`) with
  the stdlib `random.sample`/`random.random` (`from random import
  sample` in `imodels/rule_set/brs.py`), which `random_state` never
  seeds. So a given `random_state` alone is *not* reproducible -- the
  same call can hit the bug above or not depending on unrelated earlier
  code's use of the global `random` module. `BayesianRuleSet.
  fit_external` seeds `random.seed(random_state)` itself before calling
  in, to actually make results (and whether this run hits the "clean"
  bug) reproducible.

`RuleFitImporter` (for `imodels.RuleFitClassifier`) produces a
`pyrulearn.models.LinearRuleModel`, the same model type the native
`pyrulearn.learners.rulefit.RuleFit` returns, so the two compare
directly. RuleFit generates candidate rules from the paths of a
gradient-boosted tree ensemble (fit as a *regressor* on the 0/1 label,
even for classification), adds the input features as "linear terms",
and fits an L1-regularized logistic regression over both (`liblinear`,
its `C` picked from a grid as the least regularization keeping at most
`max_rules` terms, by cross-validated accuracy with `cv=True`). Binary
targets only (`fit` raises for more). The conversion is exact on 0/1
input, confirmed against the model's own `predict`:

- **Tree-path rules** are ``X_3 > 0.5 and X_1 <= 0.5``-style
  conjunctions. On a Boolean column, ``> t`` (0 <= t < 1) is "feature
  True" and ``<= t`` is "feature False", which imports as a positive
  literal on the paired **negation feature** (`DataSpec.negation_of`),
  as for BRS; a spec without negation features is rebuilt with them the
  same way.
- **Linear terms** go through imodels' winsorizing/scaling
  (`lin_standardise`, `lin_trim_quantile`) before the fit. On a Boolean
  column that is an affine map ``a + b * x``: `a` folds into the
  intercept and ``b * coef`` becomes the weight of a length-1 rule on the
  feature -- e.g. a feature True in fewer than `lin_trim_quantile` of the
  rows is trimmed to a constant 0, so its term disappears.
- A tree-path rule and a linear term with the same body are merged, by
  summing their weights.
- **A bug in imodels: the decision threshold.** The regression is
  logistic, so its output `f` is a log-odds and the positive class
  should win where ``f > 0``. But `RuleFitClassifier.predict_proba`
  takes the softmax of ``[1 - f, f]``, i.e. ``sigmoid(2f - 1)``, and
  `predict` returns the positive class only where ``f > 0.5``.
  `RuleFitImporter(imodels_threshold=True)` (the default) reproduces
  `predict` exactly, and makes the offset visible in the model: a rule
  ``0.5::<negative class>(X) :- true.`` beside the intercept.
  `imodels_threshold=False` imports the logistic model as fitted
  (``f > 0``).

This module requires the `imodels` package (an optional dependency --
only importing this specific module pulls it in, per
`pyrulearn.interfaces`'s own dependency-isolation convention).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import ObjectRuleImporter, register_importer
from ..data import DataSpec, DataSpecBuilder
from ..learners import ExternalRuleLearner, RelabelingExternalLearner
from ..models import ConceptModel, DecisionList, FlatRuleSet, LinearRuleModel
from ..data import DataRepresentation
from ..rule import Rule, WeightedRule


def _placeholder_to_column(model) -> Dict[str, int]:
    """`{"X_0": 0, "X_1": 1, ...}` -- `model.feature_placeholders` is in
    the same order as the columns `model` was fit on, so a rule term's
    placeholder resolves straight to a column index (and thus, by this
    package's usual convention, to the matching `dataspec` feature
    index). Used rather than `model.feature_dict_`'s *names* so that a
    model fit without `feature_names=` -- whose names are then just
    "X_0", "X_1", ... -- still imports against a real `DataSpec`.
    """
    return {str(p): i for i, p in enumerate(model.feature_placeholders)}


def _rule_from_terms(terms, placeholder_cols: Dict[str, int], target: Any, dataspec: DataSpec) -> Rule:
    """One pyrulearn `Rule` from one `imodels.util.rule.Rule`'s `terms`
    -- a list of ``[placeholder, op, threshold]`` triples, e.g.
    ``[["X_0", ">", "0.5"], ["X_1", ">", "0.5"]]``.

    Since `BayesianRuleListClassifier` only accepts 0/1 input (see the
    module docstring), every term is necessarily a ``> 0.5`` test, which
    is just "this Boolean feature is True" -- so each becomes one
    *positive* literal. Anything else raises rather than being silently
    reinterpreted, since it would mean the model was fit on input this
    importer's assumptions don't cover.
    """
    pos: List[int] = []
    for term in terms:
        placeholder, op, threshold = term[0], term[1], float(term[2])
        if op != ">" or threshold != 0.5:
            raise ValueError(
                f"unexpected BRL rule term {term!r}: only '> 0.5' tests (i.e. plain Boolean "
                "features) are supported, which is all BayesianRuleListClassifier can produce "
                "from its required 0/1 input"
            )
        if placeholder not in placeholder_cols:
            raise ValueError(f"rule term references unknown feature placeholder {placeholder!r}")
        pos.append(placeholder_cols[placeholder])
    return Rule.from_pos_neg(pos=pos, target=target, dataspec=dataspec)


def _target_for(model, theta: float) -> Any:
    """The target for one rule (or the trailing default): `theta` is
    P(`classes_[1]`), so the rule's own prediction is whichever class
    that favors -- see the module docstring. The posterior itself isn't
    kept anywhere on the rule; rank/combine by measured `stats()`
    instead (`pyrulearn.combiners.HeuristicMaxCombiner`/
    `HeuristicVoteCombiner`, populated by the `fit()`/`data=` round trip)."""
    neg_class, pos_class = model.classes_[0], model.classes_[1]
    return pos_class if theta > 0.5 else neg_class


class BayesianRuleListImporter(ObjectRuleImporter):
    """Extracts a `pyrulearn.models.DecisionList` from a fitted
    `imodels.BayesianRuleListClassifier` -- one `Rule` per antecedent in
    `model.rules_`, **in the model's own order** (first match wins,
    matching how the model itself predicts), plus the trailing ELSE as
    the list's `default_prediction`.

    See the module docstring for the 0/1-input requirement, why
    `infer_dataspec` needs no override here, and how each rule's
    posterior probability maps onto its `target`.
    """

    SOURCE = "imodels.BayesianRuleListClassifier"

    def import_model(self, model, dataspec: DataSpec, data: Optional[DataRepresentation] = None) -> DecisionList:
        """`dataspec`'s feature order must match the Boolean columns
        `model` was trained on -- rules bind to `dataspec` directly.
        Returns a `DecisionList` (ordered, first-match-wins), not a
        `FlatRuleSet`: a Bayesian Rule List's rules are meaningful only in
        order, each implicitly conditioned on every earlier rule having
        *not* fired.
        """
        if len(model.theta) != len(model.rules_) + 1:
            raise ValueError(
                f"model has {len(model.rules_)} rules but {len(model.theta)} theta entries -- "
                "expected exactly one more theta than rules (the trailing ELSE/default)"
            )
        placeholder_cols = _placeholder_to_column(model)
        if len(placeholder_cols) != dataspec.n_features:
            raise ValueError(
                f"dataspec has {dataspec.n_features} features but the model was trained on "
                f"{len(placeholder_cols)} -- they must describe the same Boolean feature space"
            )

        rules: List[Rule] = []
        for i, wrule in enumerate(model.rules_):
            target = _target_for(model, model.theta[i])
            rules.append(_rule_from_terms(wrule.terms, placeholder_cols, target, dataspec))

        # the trailing ELSE -> the list's default_prediction (a bare label)
        default_target = _target_for(model, model.theta[-1])
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)

        rule_list = DecisionList(rules, default_prediction=default_target)
        if default_target is not None:
            self._stamp_rule_provenance([rule_list.default_rule], n_rules=len(rules))
        return self._stamp_provenance(rule_list, n_rules=len(rules))


register_importer("bayesian_rule_list", BayesianRuleListImporter)


class BayesianRuleList(RelabelingExternalLearner):
    """`imodels.BayesianRuleListClassifier`. `fit(data)` ->
    `pyrulearn.models.DecisionList` (BRL is a 2-class decision list).
    `**params` are its constructor args (`minsupport=`, `maxcardinality=`,
    `n_chains=`, `random_state=`). Multi-class via `fit(data,
    model=ConceptSet | ConceptCascade | PairwiseModel)`.
    """

    IMPORTER = BayesianRuleListImporter
    NATIVE_MODEL = DecisionList

    def __init__(self, **params):
        self.params = params

    def fit_external(self, X, y, feature_names=None):
        from imodels import BayesianRuleListClassifier

        if y is None:
            raise ValueError("BayesianRuleList needs labels (y) to fit")
        model = BayesianRuleListClassifier(**self.params)
        # X must be 0/1 ints, not bools: BRL's own fit does
        # `X_df[col].replace({1: col, 0: ''})` on a DataFrame built from
        # X, which doesn't match numpy bools
        model.fit(np.asarray(X).astype(int), y, feature_names=list(feature_names) if feature_names else None)
        return model


def _brs_dataspec(model, dataspec: DataSpec) -> DataSpec:
    """The `DataSpec` BRS rules should bind to.

    BRS emits ``"_neg"``-suffixed items, which import as positive literals
    on a *paired negation feature* -- so every one of the model's columns
    needs one. If `dataspec` already provides that (it was built with
    negation on), it's returned unchanged. Otherwise -- e.g. BRS was
    deliberately fit on a negation-free space to keep its feature count
    (and its per-rule-length RandomForest discretization) small -- a
    plain Boolean dataspec with one ``add_boolean`` per model column
    (negation on) is built so the ``"_neg"`` items resolve.
    """
    return _negation_enabled_dataspec(list(model.feature_names_), dataspec)


def _negation_enabled_dataspec(names: Sequence[str], dataspec: DataSpec) -> DataSpec:
    """`dataspec` if every feature in `names` has a paired negation
    feature, else a plain Boolean spec over `names` with negation on (see
    `_brs_dataspec`)."""
    names = list(names)
    missing = [n for n in names if n not in dataspec.feature_names]
    if missing:
        raise ValueError(
            f"dataspec is missing {len(missing)} of the model's {len(names)} features "
            f"(e.g. {missing[0]!r}) -- it must describe the same Boolean feature space"
        )
    if all(dataspec.negation_of(dataspec.feature_index(n)) is not None for n in names):
        return dataspec
    builder = DataSpecBuilder(negation=True)
    for name in names:
        builder.add_boolean(name)
    return builder.build()


def _name_to_idx(model, dataspec: DataSpec) -> Dict[str, int]:
    """`{"age": 3, "color=red": 5, ...}` -- BRS's rule items are real
    feature names (optionally `_neg`-suffixed), unlike BRL's `X_0`-style
    placeholders, so this maps straight through `dataspec.feature_index`
    rather than needing an intermediate placeholder table.
    """
    return {name: dataspec.feature_index(name) for name in model.feature_names_}


def _rule_from_items(items, name_to_idx: Dict[str, int], target: Any, dataspec: DataSpec) -> Rule:
    """One pyrulearn `Rule` from one entry of `model.rules_` -- a list of
    item strings, each either a real feature name or that name
    `_neg`-suffixed (feature-absent). The direct name is tried first,
    matching `BayesianRuleSetClassifier.predict`'s own resolution order.
    A `_neg` item becomes a positive literal on the feature's paired
    negation feature (`Rule.from_pos_neg`'s `neg=` translation), so
    `dataspec` must be negation-enabled -- see the module docstring.
    """
    pos: List[int] = []
    neg: List[int] = []
    for item in items:
        if item in name_to_idx:
            pos.append(name_to_idx[item])
        elif item.endswith("_neg") and item[:-4] in name_to_idx:
            neg.append(name_to_idx[item[:-4]])
        else:
            raise ValueError(
                f"BRS rule item {item!r} doesn't match any known feature name (or its "
                "'_neg'-suffixed negation) -- dataspec wasn't built from this same model"
            )
    return Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec)


class BayesianRuleSetImporter(ObjectRuleImporter):
    """Extracts a `pyrulearn.models.FlatRuleSet` from a fitted
    `imodels.BayesianRuleSetClassifier` (BOA) -- one `Rule` per pattern
    in `model.rules_`, every one predicting `model.classes_[1]` (the
    model's OR-of-AND structure: any single rule firing is enough), plus
    a `default_prediction` of `model.classes_[0]` for examples no rule
    covers. See the module docstring for the Boolean-
    input/binary-target requirements this importer doesn't itself
    enforce (that's `BayesianRuleSet.fit_external`'s job) and why
    imported rules carry no per-rule weight/posterior.
    """

    SOURCE = "imodels.BayesianRuleSetClassifier"

    def import_model(self, model, dataspec: DataSpec, data: Optional[DataRepresentation] = None) -> FlatRuleSet:
        rebuilt_dataspec = _brs_dataspec(model, dataspec)
        # _brs_dataspec rebuilds dataspec (adding the negation columns BRS's
        # own "_neg" items need) when the caller's dataspec didn't already
        # have them -- see its docstring. When that happens, `data`'s own X
        # no longer matches the rules' actual (bigger) feature space, so
        # computing coverage against it would misread rule conditions
        # against the wrong columns ("feature guessing") -- skip stats in
        # that case rather than risk it; it only arises for a dataspec
        # deliberately built negation-free (a documented BRS-only workaround
        # for its exponential column blowup), not the normal fit() path.
        stats_data = data if rebuilt_dataspec is dataspec else None
        dataspec = rebuilt_dataspec
        name_to_idx = _name_to_idx(model, dataspec)
        neg_class, pos_class = model.classes_[0], model.classes_[1]

        rules = [_rule_from_items(items, name_to_idx, pos_class, dataspec) for items in model.rules_]
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, stats_data)

        rule_set = FlatRuleSet(rules, default_prediction=neg_class)
        self._stamp_rule_provenance([rule_set.default_rule], n_rules=len(rules))
        return self._stamp_provenance(rule_set, n_rules=len(rules))


register_importer("bayesian_rule_set", BayesianRuleSetImporter)


class BayesianRuleSet(RelabelingExternalLearner):
    """`imodels.BayesianRuleSetClassifier`. `fit(data)` ->
    `pyrulearn.models.FlatRuleSet`. `**params` are its constructor args
    (`n_rules=`, `supp=`, `maxlen=`, `discretization_method=`,
    `random_state=`). Non-Boolean input / non-binary target are checked
    here (imodels itself fails opaquely). Multi-class via `fit(data,
    model=ConceptSet | ...)`.
    """

    IMPORTER = BayesianRuleSetImporter
    NATIVE_MODEL = FlatRuleSet

    def __init__(self, **params):
        self.params = params

    def fit_external(self, X, y, feature_names=None):
        from imodels import BayesianRuleSetClassifier

        if y is None:
            raise ValueError("BayesianRuleSet needs labels (y) to fit")
        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(
                f"BayesianRuleSet only supports binary targets, got {len(classes)} classes: "
                f"{list(classes)}"
            )
        X = np.asarray(X)
        if not np.all((X == 0) | (X == 1)):
            raise ValueError(
                "BayesianRuleSet needs already-Boolean (0/1) input -- binarize categorical/"
                "numeric attributes first, e.g. via pyrulearn.data.io.build_dataspec/binarize"
            )
        # BayesianRuleSetClassifier.fit mixes np.random (seeded from its own
        # random_state=) with the *unseeded* stdlib random.sample/random.random
        # (`from random import sample` in imodels/rule_set/brs.py) -- without
        # seeding that too, results (and the "clean"-move bug documented in
        # the module docstring) depend on ambient global random state left by
        # unrelated earlier code, not just this call's own random_state
        random_state = self.params.get("random_state")
        if random_state is not None:
            import random as _random
            _random.seed(random_state)
        model = BayesianRuleSetClassifier(**self.params)
        model.fit(X.astype(int), y, feature_names=list(feature_names) if feature_names else None)
        return model


# ================================================================ RuleFit ===

_OPS = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal,
        "==": np.equal}


def _boolean_values(terms) -> set:
    """The values in {0, 1} a Boolean column may take to satisfy every
    ``(op, threshold)`` test in `terms`."""
    return {v for v in (0, 1) if all(_OPS[op](v, float(t)) for op, t in terms)}


def _rulefit_body(agg_dict, cols: Dict[str, int], idx: Sequence[int], dataspec: DataSpec):
    """A pyrulearn body (feature indices) from an `imodels.util.rule.Rule`'s
    `agg_dict` (``{(placeholder, op): threshold}``) on 0/1 columns, or
    `None` if no 0/1 row can satisfy it. A "False" test becomes the
    column's negation feature; a column both values satisfy drops out."""
    by_col: Dict[int, list] = {}
    for (placeholder, op), t in agg_dict.items():
        by_col.setdefault(cols[placeholder], []).append((op, t))
    body = []
    for c, terms in by_col.items():
        allowed = _boolean_values(terms)
        if not allowed:
            return None
        if allowed == {1}:
            body.append(idx[c])
        elif allowed == {0}:
            body.append(dataspec.negation_of(idx[c]))
    return body


def _rulefit_linear_map(model, j: int):
    """``(a, b)`` with imodels' transformed linear term for column `j`
    equal to ``a + b * x`` on a 0/1 column `x`."""
    x = np.array([[0.0] * model.n_features_, [1.0] * model.n_features_])
    v = model.friedscale.scale(x)[:, j] if model.lin_standardise else x[:, j]
    return float(v[0]), float(v[1] - v[0])


class RuleFitImporter(ObjectRuleImporter):
    """Extracts a `pyrulearn.models.LinearRuleModel` from a fitted
    `imodels.RuleFitClassifier` -- one `WeightedRule` per kept tree-path
    rule and linear term (each with its coefficient as the weight, on
    `classes_[1]`), plus the intercept as an empty-body rule. Assumes the
    model was fit on 0/1 columns matching `dataspec`'s features. See the
    module docstring for the conversion and `imodels_threshold`.
    """

    SOURCE = "imodels.RuleFitClassifier"

    def __init__(self, imodels_threshold: bool = True):
        self.imodels_threshold = imodels_threshold

    def import_model(self, model, dataspec: DataSpec,
                     data: Optional[DataRepresentation] = None) -> LinearRuleModel:
        cols = _placeholder_to_column(model)
        if len(cols) != dataspec.n_features:
            raise ValueError(
                f"dataspec has {dataspec.n_features} features but the model was trained on "
                f"{len(cols)} -- they must describe the same Boolean feature space"
            )
        names = [str(n) for n in model.feature_names]
        rebuilt = _negation_enabled_dataspec(names, dataspec)
        stats_data = data if rebuilt is dataspec else None     # see BayesianRuleSetImporter
        dataspec = rebuilt
        idx = [dataspec.feature_index(n) for n in names]       # model column -> feature
        neg_class, pos_class = model.classes_[0], model.classes_[1]

        weights: Dict[tuple, float] = {}                       # body (feature tuple) -> weight
        intercept = float(np.ravel(model.intercept)[0])

        def add(body, w):
            nonlocal intercept
            if not body:
                intercept += w
            else:
                key = tuple(sorted(set(body)))
                weights[key] = weights.get(key, 0.0) + w

        n_linear = len(model.coef) - len(model.rules_without_feature_names_)
        for j in range(n_linear):                              # linear terms: a + b * x
            if model.coef[j] != 0:
                a, b = _rulefit_linear_map(model, j)
                add((), a * model.coef[j])
                add((idx[j],), b * model.coef[j])
        for r in model.rules_without_feature_names_:          # tree-path rules
            body = _rulefit_body(r.agg_dict, cols, idx, dataspec)
            if body is not None:
                add(body, float(r.args[0]))

        rules: List[WeightedRule] = [WeightedRule([], target=pos_class, dataspec=dataspec, weight=intercept)]
        if self.imodels_threshold:   # predict's f > 0.5, as a visible offset for the other class
            rules.append(WeightedRule([], target=neg_class, dataspec=dataspec, weight=0.5))
        for body in sorted((b for b, w in weights.items() if w != 0), key=lambda b: -abs(weights[b])):
            rules.append(WeightedRule(list(body), target=pos_class, dataspec=dataspec, weight=weights[body]))
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, stats_data)
        classes = [c.item() if isinstance(c, np.generic) else c for c in model.classes_]
        return self._stamp_provenance(LinearRuleModel(rules, classes=classes), n_rules=len(rules))


register_importer("rulefit", RuleFitImporter)


class ImodelsRuleFit(ExternalRuleLearner):
    """`imodels.RuleFitClassifier`. `fit(data)` ->
    `pyrulearn.models.LinearRuleModel` (binary targets only -- a linear
    model doesn't decompose into the `ConceptModel`s the multiclass
    switchers build). `**params` are its constructor args (`max_rules=`,
    `n_estimators=`, `tree_size=`, `include_linear=`, `alpha=`, `cv=`,
    `random_state=`); `imodels_threshold` is `RuleFitImporter`'s. Named
    apart from the native `pyrulearn.learners.rulefit.RuleFit`.
    """

    IMPORTER = RuleFitImporter
    NATIVE_MODEL = LinearRuleModel

    def __init__(self, imodels_threshold: bool = True, **params):
        self.imodels_threshold = imodels_threshold
        self.params = params

    def _import(self, data: DataRepresentation) -> Any:
        fitted = self.fit_external(self.prepare(data), data.y, feature_names=data.spec.feature_names)
        return RuleFitImporter(self.imodels_threshold).import_model(fitted, data.spec, data=data)

    def fit_external(self, X, y, feature_names=None):
        from imodels import RuleFitClassifier

        if y is None:
            raise ValueError("ImodelsRuleFit needs labels (y) to fit")
        X = np.asarray(X)
        if not np.all((X == 0) | (X == 1)):
            raise ValueError(
                "ImodelsRuleFit needs already-Boolean (0/1) input -- binarize categorical/"
                "numeric attributes first, e.g. via pyrulearn.data.io.build_dataspec/binarize"
            )
        model = RuleFitClassifier(**self.params)
        model.fit(X.astype(float), np.asarray(y), feature_names=list(feature_names) if feature_names else None)
        return model


def rulefit_candidates(data: DataRepresentation, n_estimators: int = 100, tree_size: int = 4,
                       memory_par: float = 0.01, exp_rand_tree_size: bool = True,
                       sample_fract: Any = "default", random_state: Optional[int] = None) -> FlatRuleSet:
    """RuleFit's candidate generation on its own: the distinct rules from
    the node paths of imodels' gradient-boosted tree ensemble
    (`imodels.util.extract.extract_rulefit`, the step
    `RuleFitClassifier.fit` runs before its regression; same parameters
    and defaults), as a pool for a distiller -- e.g.
    ``RuleFit(rules=rulefit_candidates(data))`` runs RuleFit's two steps
    with pyrulearn's own fit. Binary targets only, like RuleFit (the
    trees are regressors on the 0/1 label); `data` must be Boolean with
    negation features. Each rule's head is the majority class of the
    training rows it covers (`RuleFit` ignores heads; `CBA`/`IDS` use
    them), and it carries its training stats.
    """
    from imodels.util.extract import extract_rulefit
    from imodels.util.rule import Rule as ImodelsRule
    from ..models import annotate_rules

    y = np.asarray(data.y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"rulefit_candidates needs a binary target, got {len(classes)} classes")
    X = np.asarray(data.X).astype(float)
    spec = data.spec
    placeholders = [f"X_{i}" for i in range(spec.n_features)]
    cols = {p: i for i, p in enumerate(placeholders)}
    idx = list(range(spec.n_features))
    strings = extract_rulefit(X, (y == classes[1]).astype(float), feature_names=placeholders,
                              n_estimators=n_estimators, tree_size=tree_size, memory_par=memory_par,
                              exp_rand_tree_size=exp_rand_tree_size, sample_fract=sample_fract,
                              random_state=random_state)
    bodies: Dict[tuple, None] = {}
    for text in strings:
        body = _rulefit_body(ImodelsRule(text).agg_dict, cols, idx, spec)
        if body:
            bodies.setdefault(tuple(sorted(set(body))), None)
    rules = []
    for body in bodies:
        covered = np.all(np.asarray(data.X)[:, list(body)], axis=1)
        n1 = int(np.sum(y[covered] == classes[1]))
        head = classes[1] if 2 * n1 > int(covered.sum()) else classes[0]
        rules.append(Rule(list(body), target=head.item() if isinstance(head, np.generic) else head,
                          dataspec=spec))
    return FlatRuleSet(annotate_rules(rules, data))
