"""
pyrulearn.learners.base
===========================

`RuleLearner`: the shared entry point for turning data into rules.

`fit(data, model=None, **model_kwargs) -> RuleModel`
    `model` selects the **return type** (a `pyrulearn.models` class); with
    `model=None` the learner produces its own default (`_fit_default`).
    A learner declares which model types it can build by decorating
    production methods with `@produces(SomeModel, ...)`; `fit` collects
    those across the class hierarchy (`_producers`), and falls back to a
    registered model→model converter over any producible type. `produces()`
    reports the whole set -- the capability matrix.

Two concrete bases:

- `ExternalRuleLearner` -- runs an external algorithm and converts its
  output via an `ObjectRuleImporter` (`IMPORTER` class attribute). On the
  `fit` switcher too: `_fit_native` produces `NATIVE_MODEL`, returning the
  importer's `pyrulearn.models` output directly. Mix in
  `RelabelingExternalLearner` for the decomposition producers.
- `NativeRuleLearner` -- induces rules directly; uses the `fit` switcher.

`DecomposingLearner` (mixin) adds the binary-decomposition producers --
`ConceptSet` (one-vs-rest), `ConceptCascade` (ordered peeling),
`PairwiseModel` (round robin) -- for any learner that can be pointed at
one class at a time. Mix it in; override an individual producer to
customise, or set it to `None` to disable just that one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Type, Union

import numpy as np

from ..data import DataRepresentation
from ..interfaces.base import ObjectRuleImporter
from ..models import (
    ConceptCascade, ConceptModel, ConceptSet, MajorityClass, PairwiseModel,
    Provenance, RuleModel, annotate_default_rule, can_convert, convert,
)


def produces(*model_types: type) -> Callable:
    """Mark a method as a *producer* of the given `pyrulearn.models`
    types. `fit` dispatches `model=` to it. `@produces()` with no
    arguments (on an override) unregisters the inherited producer."""
    def deco(fn):
        fn._produces = model_types
        return fn
    return deco


class RuleLearner(ABC):
    """Base for anything that turns a `DataRepresentation` into a
    `pyrulearn.models.RuleModel`."""

    # -- producer registry -------------------------------------------

    def _producers(self) -> Dict[type, Callable]:
        """`{model type -> bound producer method}`, merged across the
        MRO. A subclass method of the same name overrides the parent's
        entry; an override without `@produces` (or set to `None`)
        disables it."""
        names = {n for k in type(self).__mro__ for n, f in vars(k).items()
                 if hasattr(f, "_produces")}
        out: Dict[type, Callable] = {}
        for n in names:
            fn = getattr(self, n, None)
            for t in getattr(getattr(fn, "__func__", None), "_produces", ()):
                out[t] = fn
        return out

    def produces(self) -> Set[type]:
        """Every model type this learner can deliver -- registered
        producers, plus one-hop conversions of them."""
        direct = set(self._producers())
        extra = {d for n in direct for d in _CONVERTIBLE_FROM.get(n, ())}
        return direct | extra

    # -- provenance --------------------------------------------------

    def _provenance_params(self) -> Dict[str, Any]:
        """Constructor parameters recorded on a built model's
        `Provenance`. Default: every public instance attribute -- every
        learner in this package assigns one attribute per constructor
        argument (sklearn-style, no extra state), so `vars(self)` already
        *is* "the learner's parameters". A `self.params` dict (this
        package's `__init__(self, **params): self.params = params`
        convention for external-library wrappers) is flattened into the
        result rather than nested one level down, so e.g.
        `DecisionTree(max_depth=2)`'s provenance reads `{"max_depth": 2}`,
        not `{"params": {"max_depth": 2}}`. Override if a learner stores
        something else."""
        attrs = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        params = attrs.pop("params", None)
        if isinstance(params, dict):
            return {**params, **attrs}
        if params is not None:
            attrs["params"] = params
        return attrs

    def _provenance_source(self) -> Optional[str]:
        """The external algorithm identifier for this learner's
        `Provenance` (`Provenance.source`) -- `None` for a native
        learner. `ExternalRuleLearner` overrides with its `IMPORTER.SOURCE`."""
        return None

    # -- fit -------------------------------------------------------

    def fit(self, data: DataRepresentation, model: Optional[type] = None, **model_kwargs) -> RuleModel:
        if model is None:
            if model_kwargs:
                raise TypeError(f"model_kwargs {list(model_kwargs)} given without model=")
            return self._fit_default(data)
        result = self._dispatch(data, model, model_kwargs)
        if result.provenance is None:
            result.provenance = Provenance(
                learner=type(self).__name__,
                params={**self._provenance_params(), **model_kwargs},
                source=self._provenance_source(),
            )
        return result

    def _dispatch(self, data: DataRepresentation, model: type, model_kwargs: Dict[str, Any]) -> RuleModel:
        prod = self._producers()
        if model in prod:
            return prod[model](data, **model_kwargs)
        for native, build in prod.items():
            if can_convert(native, model):
                if model_kwargs:
                    raise TypeError(
                        f"{type(self).__name__} reaches {model.__name__} only by converting "
                        f"{native.__name__}; no room for {list(model_kwargs)}"
                    )
                return convert(build(data), model)
        raise TypeError(
            f"{type(self).__name__} cannot produce {model.__name__} "
            f"(can: {sorted(t.__name__ for t in self.produces())})"
        )

    def _default_model(self, data: DataRepresentation) -> type:
        """The model type `fit(data)` builds with no `model=`. Default:
        the sole registered producer's type, if there is exactly one;
        otherwise a subclass must override (SeCo does, keyed on
        `target_class` and the label count)."""
        prod = self._producers()
        if len(prod) == 1:
            return next(iter(prod))
        raise NotImplementedError(
            f"{type(self).__name__} has several producers "
            f"({sorted(t.__name__ for t in prod)}); override _default_model or pass model="
        )

    def _fit_default(self, data: DataRepresentation) -> RuleModel:
        """`fit(data)` with no `model=` -- dispatches `_default_model`."""
        return self.fit(data, model=self._default_model(data))


class ExternalRuleLearner(RuleLearner):
    """Runs an external algorithm against data, then hands the fitted
    model to an `ObjectRuleImporter` (`IMPORTER` class attribute).

    On the `model=` switcher: `_fit_native` (prepare -> `fit_external` ->
    import -> adapt to `NATIVE_MODEL`) is the sole native producer; mix in
    `DecomposingLearner` and implement `_fit_binary` (relabel to
    `{c, "rest"}` -- an external binary learner can't fold "the rest" in
    itself) to gain `ConceptSet` / `ConceptCascade` / `PairwiseModel`.
    """

    IMPORTER: Type[ObjectRuleImporter]
    #: the `pyrulearn.models` type a plain `fit(data)` produces
    NATIVE_MODEL: type

    def prepare(self, data: DataRepresentation) -> Any:
        """(1a) Convert `data` into the algorithm's native input format.
        Default: `data.X` as-is."""
        return data.X

    @abstractmethod
    def fit_external(self, X: Any, y: Any, feature_names: Optional[Sequence[str]] = None) -> Any:
        """(1b/2b) Call the external algorithm and return its fitted
        model object (not yet converted to rules)."""
        raise NotImplementedError

    def _import(self, data: DataRepresentation) -> Any:
        X = self.prepare(data)
        fitted = self.fit_external(X, data.y, feature_names=data.spec.feature_names)
        return self.IMPORTER().import_model(fitted, data.spec, data=data)

    def _fit_native(self, data: DataRepresentation) -> RuleModel:
        # self._import already stamped the *importer's* own Provenance
        # (via ObjectRuleImporter.import_model -> _stamp_provenance) --
        # clear it so RuleLearner.fit's `if result.provenance is None`
        # check fires and re-stamps the *learner's* own Provenance
        # instead (this fit() call's actual params, e.g. max_depth=,
        # not import_model's own bookkeeping params).
        model = self._import(data)
        model.provenance = None
        return model

    def _producers(self) -> Dict[type, Callable]:
        out = dict(super()._producers())          # @produces methods (DecomposingLearner, if mixed in)
        out[self.NATIVE_MODEL] = self._fit_native
        return out

    def _default_model(self, data: DataRepresentation) -> type:
        return self.NATIVE_MODEL

    def _provenance_source(self) -> Optional[str]:
        return getattr(self.IMPORTER, "SOURCE", None)


class NativeRuleLearner(RuleLearner):
    """Base for learners that induce rules directly against a
    `DataRepresentation` -- they use the `fit` switcher and register
    `@produces` methods."""


# ============================================== decomposition mixin ==========

def _ordered_labels(y: np.ndarray, order: Union[str, Sequence[Any]],
                    rng: np.random.Generator) -> List[Any]:
    labels, counts = np.unique(y, return_counts=True)
    if not isinstance(order, str):
        order = list(order)
        if set(order) != set(labels.tolist()):
            raise ValueError(f"order {order} is not a permutation of {labels.tolist()}")
        return order
    if order == "least_frequent":
        idx = np.argsort(counts, kind="stable")
    elif order == "most_frequent":
        idx = np.argsort(-counts, kind="stable")
    elif order == "random":
        idx = rng.permutation(len(labels))
    else:
        raise ValueError("order must be a label sequence or 'least_frequent'/"
                         f"'most_frequent'/'random', got {order!r}")
    return list(labels[idx])


def _pairwise_directions(a: Any, b: Any, positive: Union[str, Callable],
                         counts: Dict[Any, int], rng: np.random.Generator) -> List[tuple]:
    """The ``(positive, negative)`` target(s) for the unordered pair
    ``{a, b}`` -- two for ``positive="both"`` (double round robin), one
    otherwise: ``"smaller"``/``"larger"``/``"random"`` or a callable
    ``f(a, b) -> positive``."""
    if positive == "both":
        return [(a, b), (b, a)]
    if callable(positive):
        pos = positive(a, b)
    elif positive == "smaller":
        pos = a if counts[a] <= counts[b] else b
    elif positive == "larger":
        pos = a if counts[a] >= counts[b] else b
    elif positive == "random":
        pos = a if rng.random() < 0.5 else b
    else:
        raise ValueError("positive must be 'smaller'/'larger'/'random'/'both' or a callable, "
                         f"got {positive!r}")
    return [(pos, b if pos == a else a)]


class DecomposingLearner:
    """Mixin: adds the binary-decomposition producers to a learner that
    can fit a model for one class at a time (`_fit_binary`). Provides
    `_fit_one_vs_rest` -> `ConceptSet`, `_fit_ordered` -> `ConceptCascade`
    (peeling), `_fit_pairwise` -> `PairwiseModel` (round robin). Override
    an individual `_fit_*` to customise, or set it to `None` to disable
    just that one.
    """

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(getattr(self, "random_state", None))

    def _fit_binary(self, data: DataRepresentation, positive: Any,
                    negative: Optional[Any] = None) -> RuleModel:
        """Fit a model for `positive` -- vs. `negative` if given (a pair
        sub-problem), else vs. everything else in `data.y` (one-vs-rest).
        The learner-specific primitive every decomposition strategy uses;
        the SeCo family / PyLORD implement it by setting `target_class`."""
        raise NotImplementedError(
            f"{type(self).__name__} mixes in DecomposingLearner but has no _fit_binary"
        )

    @produces(ConceptSet)
    def _fit_one_vs_rest(self, data: DataRepresentation, **kw) -> ConceptSet:
        y = np.asarray(data.y)
        concepts = [ConceptModel(list(self._fit_binary(data, c).rules), label=c)
                    for c in np.unique(y)]
        return annotate_default_rule(ConceptSet(concepts, default_prediction=MajorityClass(data)), data)

    @produces(ConceptCascade)
    def _fit_ordered(self, data: DataRepresentation, *,
                     order: Union[str, Sequence[Any]] = "least_frequent", **kw) -> ConceptCascade:
        y = np.asarray(data.y)
        labels = _ordered_labels(y, order, self._rng())
        concepts: List[ConceptModel] = []
        in_scope = np.ones(data.n_samples, dtype=bool)
        for c in labels[:-1]:
            stage = data if in_scope.all() else data.select_rows(in_scope)
            concepts.append(ConceptModel(list(self._fit_binary(stage, c).rules), label=c))
            in_scope = in_scope & (y != c)
        return annotate_default_rule(ConceptCascade(concepts, default_prediction=labels[-1]), data)

    @produces(PairwiseModel)
    def _fit_pairwise(self, data: DataRepresentation, *,
                      positive: Union[str, Callable] = "smaller", **kw) -> PairwiseModel:
        y = np.asarray(data.y)
        labels, counts = np.unique(y, return_counts=True)
        cnt = dict(zip(labels.tolist(), counts.tolist()))
        rng = self._rng()
        members: List[tuple] = []
        mweights: List[float] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                rows = (y == a) | (y == b)
                stage = data if rows.all() else data.select_rows(rows)
                for pos, neg in _pairwise_directions(a, b, positive, cnt, rng):
                    sub = self._fit_binary(stage, pos, negative=neg)
                    members.append((pos, neg, sub))
                    mweights.append(float(np.mean(
                        np.asarray(sub.predict(stage)) == np.asarray(stage.y))))
        model = PairwiseModel(members, default_prediction=MajorityClass(data),
                              label_priors=cnt, member_weights=mweights)
        return annotate_default_rule(model, data)


class RelabelingExternalLearner(DecomposingLearner, ExternalRuleLearner):
    """An `ExternalRuleLearner` that is natively *binary only* (no
    `pos_class` / internal negative handling) and folds "the rest" in by
    relabelling `y` to ``{c, "rest"}``. Mixing in `DecomposingLearner`
    gives it `ConceptSet` / `ConceptCascade` / `PairwiseModel`; this
    `_fit_binary` is the shared bridge the external Bayesian models and
    the sklearn single-tree learner all use.
    """

    def _fit_binary(self, data: DataRepresentation, positive: Any,
                    negative: Optional[Any] = None) -> RuleModel:
        y = np.asarray(data.y)
        src = data if negative is not None else data.relabel(
            np.where(y == positive, positive, "rest"))
        imported = self._import(src)
        pos_rules = [r for r in imported.rules if r.target == positive]
        return ConceptModel(pos_rules, label=positive, default_prediction=negative)


#: model type -> types reachable from it by a single converter (for `produces()`)
_CONVERTIBLE_FROM: Dict[type, Set[type]] = {}


def _init_convertible_from() -> None:
    from ..models import _CONVERTERS
    for (src, dst) in _CONVERTERS:
        _CONVERTIBLE_FROM.setdefault(src, set()).add(dst)


_init_convertible_from()
