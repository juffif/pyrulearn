# pyrulearn

[![CI](https://github.com/juffif/pyrulearn/actions/workflows/ci.yml/badge.svg)](https://github.com/juffif/pyrulearn/actions/workflows/ci.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

A Python library for **representing, learning, combining and analyzing
propositional rule models** over Boolean data. It provides a common model
hierarchy (rule sets, decision lists, concept sets, ensembles), a shared
data representation, pluggable rule-evaluation heuristics and combiners,
[native implementations of a range of rule learners](#algorithms) (a
configurable separate-and-conquer (SeCo) framework and the algorithms that
are instantiations of it, such as CN2, AQR, PFOIL, FOSSIL and Pypper, a
RIPPER re-implementation; plus locally optimal rules and the associative
classifiers CBA, CMAR and IDS), and
[interfaces to external learners](#algorithms) (scikit-learn, wittgenstein,
imodels, Weka, LORD, pyarc), so that native and external algorithms can be
run and compared through one API.

> **Status: early development (alpha).** The APIs are still changing, and
> some demos need revising. Changes between versions are listed in the
> [changelog](CHANGELOG.md).

## Install

```
pip install git+https://github.com/juffif/pyrulearn.git
```

Requires Python ≥ 3.10, `numpy` and `scikit-learn` (the latter because
`RuleSetClassifier` inherits from `sklearn.base.BaseEstimator`, hence
`import pyrulearn` itself needs it). Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `data` | `pandas`, `scipy` | `pyrulearn.data.io` (reading ARFF/CSV; `scipy` for ARFF and sparse representations) |
| `plot` | `matplotlib`, `networkx` | coverage-space and refinement-graph plotting in `pyrulearn.evaluation` |
| `wittgenstein` | `wittgenstein` | `pyrulearn.interfaces.wittgenstein` (IREP, RIPPER) |
| `imodels` | `imodels` | `pyrulearn.interfaces.imodels` (Bayesian rule lists / sets) |
| `pyarc` | `pyarc` | `pyrulearn.interfaces.pyarc` (CBA); `pyarc` itself also needs Borgelt's `pyfim` C extension, which must be built separately (no Windows wheels) |
| `test` | `pytest` | running the test suite |
| `all` | all of the above except `pyarc` | |

For example `pip install "pyrulearn[data,plot] @ git+https://github.com/juffif/pyrulearn.git"`.
For development, clone the repository and run `pip install -e ".[all]"`, then `pytest`.
The Weka, LORD and pyarc interfaces additionally need those external
tools (Java and `weka.jar`, LORD's Java implementation, `pyarc` with `pyfim`);
their tests skip themselves when the tool is not available.

## Algorithms

All learners below share the `learner.fit(data)` entry point (see
*Learning algorithms*); multiclass decompositions
(`model=ConceptSet | ConceptCascade | PairwiseModel`) work for the native
and most of the external learners. Importers on their own convert an
already-fitted external model (or its text output) into a `RuleModel`.

### Natively implemented

| Algorithm | Class (in `pyrulearn.learners`) | Notes | Reference |
|---|---|---|---|
| **SeCo framework** | `seco.SeCo` | the separate-and-conquer (covering) engine that the next five entries are instantiations of: a per-class covering loop around composable building blocks (search, heuristics, pruning, stopping, optimization) | Fürnkranz, Gamberger & Lavrač 2012; Fürnkranz & Flach 2005 |
| **CN2** | `seco.CN2` | a `SeCo` instantiation: Laplace heuristic and likelihood-ratio significance test; unlike the rest of the family, its default `ConceptSet` resolves a clash between firing rules by summing their covered-class distributions (`MicroVoteCombiner`), not the family's generic `combiner="max"`, matching Clark & Boswell's own unordered-CN2 | Clark & Niblett 1989; Clark & Boswell 1991 |
| **AQR** | `seco.AQR` | a `SeCo` instantiation: Clark & Niblett's reimplementation of Michalski's AQ; the literal *star* search is approximated by a seed-restricted beam search | Clark & Niblett 1989 |
| **PFOIL** | `seco.PFoil` | a `SeCo` instantiation: propositional FOIL with information gain, hill climbing, MDL-based encoding-length restriction | Mooney 1995; Quinlan 1990 |
| **FOSSIL** | `seco.PFossil` | a `SeCo` instantiation: correlation heuristic with a quality threshold | Fürnkranz 1994 |
| **Pypper** | `seco.Pypper` | a `SeCo` instantiation: a re-implementation of RIPPER, not a port of Cohen's code. IREP\* growth and pruning plus the replace/revise optimization phase, per class, least-frequent class first. It differs from the original in places: the covering loop stops on FOIL's MDL restriction or IREP's precision below 0.5 instead of Cohen's 64-bit description-length rule, and there is no residual IREP\* pass after optimization | Cohen 1995; Fürnkranz & Widmer 1994 |
| **LORD** (simplified, `PyLORD`) | `pylord.PyLORD` | locally optimal rules, built from the `SeCo` building blocks but not a covering loop: every training example seeds a rule search. A simplified reimplementation, not the reference one (see *Interfaced* for that) | Huynh, Fürnkranz & Beck 2023 |
| **Class association rule mining** | `associative.CARMiner` | Apriori-style CBA-RG; returns a compact, lazily materialized `PooledRuleSet` | Liu et al. 1998; Agrawal & Srikant 1994 |
| **CBA** | `associative.CBA` | CBA-CB (M1) classifier building on top of a rule pool; cross-checked rule-for-rule against `pyarc` | Liu et al. 1998 |
| **CMAR** | `associative.CMAR` | simplified: chi-square significance filter, per-class coverage pruning, weighted chi-square voting | Li et al. 2001 |
| **IDS** | `ids.IDS` | interpretable decision sets: submodular objective, smooth local search or greedy optimization, optional coordinate-ascent tuning of the weights | Lakkaraju et al. 2016 |
| **Multiclass decomposition** | `multiclass.OneVsRest`, `OrderedOneVsRest`, `Pairwise` | one-vs-rest, ordered (peeling) and round-robin decomposition for any binary-capable learner | Fürnkranz 2002 |

`CBA`, `CMAR` and `IDS` are *rule distillers*: each consumes any pool of
rules given as a `FlatRuleSet` (`rules=`) — mined by
`CARMiner` by default, but equally one extracted from a
random forest — and returns its own, much smaller model.

### Interfaced (external implementations)

| Algorithm | Class(es) (in `pyrulearn.interfaces`) | Requires | Reference |
|---|---|---|---|
| **Decision tree** (CART) | `sklearn.DecisionTree`, `SklearnTreeImporter` | `scikit-learn` | Breiman et al. 1984 |
| **Random forest** | `sklearn.RandomForest`, `RandomForestImporter` | `scikit-learn` | Breiman 2001 |
| **IREP** | `wittgenstein.IREP`, `IREPImporter` | `wittgenstein` | Fürnkranz & Widmer 1994 |
| **RIPPER** | `wittgenstein.RIPPERk`, `RIPPERImporter` | `wittgenstein` | Cohen 1995 |
| **Bayesian Rule Lists** | `imodels.BayesianRuleList`, `BayesianRuleListImporter` | `imodels` | Letham et al. 2015 |
| **Bayesian Rule Sets** | `imodels.BayesianRuleSet`, `BayesianRuleSetImporter` | `imodels` | Wang et al. 2017 |
| **JRip** (Weka's RIPPER) | `weka.JRip`, `JRipImporter` | Java, `weka.jar` (`$WEKA_JAR`) | Cohen 1995 |
| **PART** | `weka.PART`, `PARTImporter` | Java, `weka.jar` | Frank & Witten 1998 |
| **J48** (Weka's C4.5) | `weka.J48`, `J48Importer` | Java, `weka.jar` | Quinlan 1993 |
| **LORD** (reference implementation) | `lord.LordJar`, `LORDImporter` | LORD's Java implementation ([vqphuynh/LORD](https://github.com/vqphuynh/LORD)) | Huynh, Fürnkranz & Beck 2023 |
| **CBA** | `pyarc.PyarcCBA`, `PyarcCBAImporter` | `pyarc` and Borgelt's `pyfim` | Liu et al. 1998 |

Many of the ideas behind this library, such as the separate-and-conquer
algorithms and rule-evaluation heuristics, are described in Fürnkranz,
Gamberger & Lavrač, *Foundations of Rule Learning* (Springer, 2012). Complete bibliographic entries are in
[`references.bib`](references.bib).

## Module map

One row per package or top-level module, in alphabetical order; submodules
are named inline. The module docstrings and the sections below carry the
details.

| Module (in `pyrulearn`) | What it's for |
|---|---|
| `attributes` | Typed attributes (boolean, nominal, numeric, set, hierarchical, relational) and the derived Boolean features they generate (`color=red`, `age>=30`, ...). Also the **constraints** among those features (`ExactlyOne`, `ThresholdChain`, `MutuallyExclusive`, `Implies`), which record what is impossible or already implied. Rule search uses them to skip contradictory refinements and to drop features an added condition already determines, which **reduces the search space**; they also let a rule check its own consistency. Also `evaluate_feature` (raw value to bit) and `MissingStrategy`. |
| `combiners` | `RuleCombiner`: how a `RuleSet` resolves an example covered by several rules. List order, plain majority vote, heuristic-scored max or vote, and per-class-distribution combiners (`MacroVoteCombiner` reproduces scikit-learn's soft voting). |
| `data` | Everything about data. **Three base representations**, all behind the same `coverage(rule)` / `features_of(row)` interface, so every rule learner runs on any of them and finds identical rules: `BooleanDataRepresentation` (a bit-packed Boolean matrix, the default), `SparseDataRepresentation` (scipy CSR/CSC, Eclat-style tid-lists) and `NListRepresentation` (the PPC-tree / N-list index of LORD; `PrePostNListRepresentation` is an opt-in variant). Submodules: `data.spec` (`DataSpec`, `DataSpecBuilder`, `merge_dataspecs`: the feature space, no data), `data.representation` (the three representations above) and `data.io` (ARFF/CSV reading and writing, `binarize`, `build_dataspec`; needs `pandas`). |
| `evaluation` | Measured statistics (`RuleStats`, `ConfusionMatrix`, `ModelStats`), `sort_rules`, `summarize`, and coverage-space plotting (`CoverageSpace`, `coverage_space_plot`, `coverage_space_auc`, `rule_refinement_plot`, `build_refinement_graph`). |
| `heuristics` | `RuleHeuristic`: pluggable rule-evaluation heuristics (`Precision`, `Laplace`, `MEstimate`, `WRAcc`, `FoilGain`, `Correlation`, `Entropy`, `LikelihoodRatio`, ...), the composable `LEF`, and `plot_isometrics` for drawing a heuristic into a `CoverageSpace`. |
| `interfaces` | Bringing external rule models in. `interfaces.base` has the shared `RuleImporter` machinery (`ObjectRuleImporter`, `StringRuleImporter`, the importer registry, `PatternStringImporter`); each external tool then has its own submodule, pairing an importer with a learner wrapper: `interfaces.sklearn` (decision trees, random forests, and `RuleSetClassifier`, which wraps any `RuleModel` as a scikit-learn estimator), `interfaces.wittgenstein` (IREP, RIPPER), `interfaces.imodels` (Bayesian rule lists and sets), `interfaces.weka` (JRip, PART, J48), `interfaces.lord` (the reference LORD implementation) and `interfaces.pyarc` (CBA). |
| `learners` | Turning data into rules through one `fit(data, model=None) -> RuleModel`. `learners.base` has the shared `RuleLearner` classes, including the `DecomposingLearner` multiclass switcher. Native algorithms: `learners.seco` (the `SeCo` framework and `CN2`, `AQR`, `PFoil`, `PFossil`, `Pypper`), `learners.pylord` (`PyLORD`), `learners.associative` (`CARMiner`, the `RuleDistiller` mixin, and the `CBA` and `CMAR` classifiers built on it), `learners.ids` (`IDS`), and `learners.multiclass` (`OneVsRest`, `OrderedOneVsRest`, `Pairwise`). |
| `models` | The `RuleModel` hierarchy, organised by how a prediction is resolved: `RuleSet` (`FlatRuleSet`, `ConceptModel`, `ConceptSet`, `DisjointRuleSet`, and the memory-compact `PooledRuleSet` that `CARMiner` returns), `RuleList` (`DecisionList`, `ConceptCascade`), `CompositeModel` (`EnsembleModel`, `PairwiseModel`, `DeepModel`) and `SingleRule`. Also the `default_prediction` policy, per-model `stats`, `Provenance`, `annotate_rules`, and the model-to-model converters. |
| `pruning` | `PrePruningCriterion`: one per-candidate test (`ThresholdPrePruning`, `EncodingLengthRestriction`, ...) that a search can use as a filter, as a stopping trigger, or that the covering loop can use as its stop condition. |
| `rule` | `Rule`: a conjunction of Boolean literals, with optional condition order, several output formats and constraint-aware consistency checks. No dependencies beyond numpy. |

## Data representation

A `DataSpec` (pure schema: feature names, typed attributes, constraints,
missing-value policy) defines the Boolean feature space; a
`DataRepresentation` binds actual data to one. There are **three base
representations**, all implementing the same `coverage(rule)` /
`features_of(row)` interface, so every native rule learner
(`pyrulearn.learners.seco`'s `CN2`, `PFoil`, `PFossil`, `AQR` and `Pypper`,
`pyrulearn.learners.pylord.PyLORD`, ...) runs on any of them unchanged and
produces byte-identical rules:

- `BooleanDataRepresentation` -- the default: a `numpy.packbits`-packed copy
  of the feature matrix; every example is checked with vectorized bitwise
  NumPy ops instead of per-feature Python loops.
- `SparseDataRepresentation` -- `scipy` CSR/CSC; a rule's coverage is
  the intersection of its features' column index-sets (Eclat's vertical
  tid-lists), rarest feature first. This is the N-list *without* the
  prefix tree.
- `NListRepresentation` -- the FP-tree / N-list vertical index Huynh,
  Fürnkranz & Beck's LORD builds. Each row's true-feature set is inserted
  (most-frequent-first) into a prefix trie with shared prefixes; a
  rule's coverage is a vectorized `uint64` word-AND over one item's
  N-list, no `(n_rows, n_features)` matrix ever materialized.
  `PrePostNListRepresentation` adds pre/post visit codes, giving its search
  fast path a second way to narrow a common feature's occurrences down to
  the current search branch. A deliberate opt-in, not a flag on
  `NListRepresentation`: it's correct (see `examples/demo_representations.py`,
  which runs it alongside the other three) but measured to be roughly
  break-even to slightly *slower* than plain `NListRepresentation` at the
  data scales this library deals with -- see its own docstring for the
  numbers and when it might actually help.

Coverage dispatches on the `DataRepresentation` subclass:
`Rule.covers_data` / `Rule.covers_data_packed` both forward to
`data.coverage(rule)`, so a `Rule` needs to know nothing about how the data
is stored.

Build any of the three with `.from_boolean(bool_rep)` (or `from_xy` /
`from_dataframe`); see `examples/demo_representations.py`, which also
runs each dataset in both feature encodings (paired negation features
vs. positive tests only). Any `DataRepresentation` can switch between
those encodings in place: `rep.without_negations()` drops every
`not f` / `x != v` / `x < t` feature (a column slice; regenerates the
constraints), and `rep.with_negations()` adds the missing ones back
(synthesized as complements -- exact for data with no missing-value
routing). `DataSpec.without_negations()` / `.with_negations()` are the
schema-only versions. `rep.select_rows(mask)` gives a same-type
representation over a row subset (feature space unchanged, so learned
rules still apply to the full data) -- rebuilt from scratch, used by
`OrderedOneVsRest` to train each stage on the not-yet-peeled classes.

### Typed attributes

Beyond plain Boolean features, a `DataSpec` can know the **typed
attributes** its features derive from, via `DataSpecBuilder`:

```python
from pyrulearn import DataSpecBuilder

b = DataSpecBuilder()
b.add_boolean("smoker")
b.add_nominal("color", ["red", "green", "blue"])
b.add_numeric("age", [20, 30, 40])
b.add_set("tags", ["urgent", "billing", "bug"])
b.add_hierarchical("region", {"Europe": {"France": {"Paris": {}}}, "Asia": {}})
b.add_relational("income_gt_age_scaled", ["income", "age"], expression="income > age * 1000")
ds = b.build()
```

Nominal/numeric/set/hierarchical attributes each generate one derived
Boolean feature per value/threshold/node; relational attributes register
a single feature derived from several source attributes (the value
itself is still supplied externally, via a `BooleanDataRepresentation`).
The
constraints implied among derived features (mutual exclusion, threshold
chains, hierarchy sibling-exclusion + upward implication) are tracked
automatically, which is what lets a `Rule` ask whether it's internally
consistent (`Rule.is_consistent`) or what it implies beyond its explicit
conditions (`Rule.implied_conditions`).

`add_numeric`'s `ge_thresholds=` generates ``>=`` features, as above; a
separate `le_thresholds=` argument (default: none, a complete no-op)
generates ``<=`` features instead -- a second, independent
`ThresholdChain`, not a reinterpretation of the first. The two aren't
the same family in disguise: ``x<=t`` is the exact logical complement of
``x>t``, *not* of ``x>=t`` (they disagree exactly at ``x==t``), so
treating a `<=` condition as `NOT(x>=t)` would get that boundary wrong.
Both directions binarize straight from the same raw numeric column with
no extra step either way, and giving the same value to both arguments
(`add_numeric("age", [30], le_thresholds=[30])`) is how to represent
`age==30` as a conjunction of the two resulting literals. The two
families aren't cross-checked against each other, though (deliberately
not built): nothing stops a rule from combining `age>=30` and `age<=25`
into an impossible-to-satisfy conjunction without `Rule.is_consistent`
flagging it -- such a rule simply never covers any real example, which
is a sufficient outcome for rules read in from an external source (see
`pyrulearn.interfaces.weka.JRipImporter`, below, for exactly this
use case).

### Merging DataSpecs

`merge_dataspecs(a, b)` (in `pyrulearn.data`) merges two `DataSpec`s
describing the same underlying attributes -- e.g. two different
discretizations of the same numeric attribute -- into one
`DataSpecBuilder` with the union of nominal categories, numeric
thresholds, and set values for attributes present in both; attributes
present in only one side are carried over as-is. It's deliberately
strict about anything that isn't a straightforward union: a type
mismatch, a different `Hierarchy`, or (for relational attributes)
different sources/expression all raise rather than guessing -- reconcile
such conflicts by hand before merging. It returns a builder (spec only
-- `DataSpec` never carries data to begin with); existing `Rule`s built
against `a`/`b` don't automatically carry over to the merged feature
space -- see `remap` below for that.

#### Rebasing rules onto a different DataSpec

`Rule.remap(new_dataspec)` rebuilds a rule's conditions against a
different, compatible `DataSpec`, translating each condition's feature
index **by name** (`old_idx -> self.dataspec.feature_name(old_idx) ->
new_dataspec.feature_index(name)`) -- the same name-based identity
`merge_dataspecs` already relies on. It's a pure structural translation
(no data touched, nothing recomputed) and raises rather than silently
dropping a condition if a name has no match in `new_dataspec`.
`RuleModel.remap(new_dataspec)` applies it to every rule and carries the
`default_prediction` policy (and `.provenance`) over unchanged (the
materialized `default_rule` re-derives against the remapped rules'
dataspec on demand, dropping any stats measured against the old one --
re-annotate against whatever data comes next), returning a new instance
of the same concrete class.

This is the fix for reading in several rule-based models over "the same"
dataset when each was imported against its own per-model `DataSpec`
(different attribute/threshold choices, different feature order): build
the shared target with `merge_dataspecs`, `remap` each model onto it,
then binarize the real dataset **once** against that shared `DataSpec`
and score/compare/combine all the remapped models against it --

```python
from pyrulearn.data import BooleanDataRepresentation

shared = merge_dataspecs(model1.rules[0].dataspec, model2.rules[0].dataspec).build()
model1 = model1.remap(shared)
model2 = model2.remap(shared)
shared_rep = BooleanDataRepresentation(shared, binarize(shared, df))
```

Only `shared` ever needs a full Boolean data matrix built for it; the
per-model DataSpecs used during import don't (`remap` needs nothing but
their feature *names*), which keeps memory to one dataset's worth
regardless of how many models get combined. `remap` deliberately drops
any stats measured before the rebase (a rule's `stats()` was computed
against the *old* dataspec's rows, which no longer applies) --
re-annotate against `shared_rep` (`annotate_rules`, or a fresh
`to_string(data=shared_rep)`/`stats(shared_rep)` call) for fresh numbers.

### Reading ARFF / CSV data

```python
from pyrulearn.data.io import read_arff, read_csv

# infer a DataSpec from the file's own column types, discretizing numeric
# columns against `target` with a decision tree (max_intervals - 1 thresholds,
# capped at DEFAULT_MAX_INTERVALS = 8 by default -- a full 3-level binary tree)
rep = read_arff("weather.arff", target="play")
rep = read_csv("data.csv", target="label", max_intervals=4)  # override the cap

# or binarize against a DataSpec you already have (thresholds and all)
rep = read_csv("data.csv", dataspec=my_dataspec, target="label")
# rep.spec is my_dataspec; rep.X / rep.y hold the binarized data
```

Numeric columns without pre-given thresholds need a `target` column to
discretize against (decision-tree splits are the only strategy
implemented so far -- equal-width/equal-frequency and FUSINTER are
planned). `validate_dataspec(dataspec, df)` checks an existing `DataSpec`
against a file's header without reading data (missing attributes, type
mismatches, unknown nominal categories); `strict=True` (the default on
`read_arff`/`read_csv`) raises on anything `validate_dataspec` reports
rather than binarizing against a `DataSpec` that doesn't actually match.

Not yet handled: set-valued/hierarchical/relational attributes (ARFF/CSV
headers can't declare them, so they're never inferred, and relational
features can't be evaluated from raw data at all -- see
`pyrulearn.attributes.evaluate_feature`).

### Missing values

A missing raw value (`None`/NaN, or an attribute's declared
`missing_values`, e.g. `("?",)` for the common ARFF/UCI convention) is
handled per `pyrulearn.attributes.MissingStrategy`, resolved (explicit
`binarize`/`read_arff`/`read_csv` argument > the `DataSpec`'s own
`missing_strategy` > `DataSpec.DEFAULT_MISSING_STRATEGY`) the same way
`Rule.to_string`'s `fmt` resolves:

- **`NEVER_COVERS`** (the default) -- every feature derived from the
  missing attribute is False for that example, so it never satisfies
  *any* literal conditioning on it, positive or negated. This is a
  closed-world "we don't know" reading, not "the negation holds" -- the
  cheaper alternative to genuinely tracking a third "unknown" state
  through `Rule`'s coverage checks, which isn't implemented.
- **`MAJORITY`** -- impute the raw value before evaluating: the column
  median for a NUMERIC attribute, the most frequent value (mode)
  otherwise.
- **`RANDOM`** -- impute with another, uniformly randomly chosen
  non-missing example's value from the same column (`random_state=`
  seeds this).
- **`SEPARATE`** -- route into a dedicated feature, treating "missing"
  as its own value alongside the attribute's declared domain. Requires
  the attribute to have been built with `missing_name=`
  (`add_nominal`/`add_numeric`) -- raises if none was declared. For a
  NOMINAL attribute the dedicated feature rides in the same exhaustive
  `ExactlyOne` group as the declared domain (missing is just one more
  category); for a NUMERIC attribute it's a standalone feature outside
  the `ThresholdChain`, deliberately with no constraint linking the two
  (a numeric value can be simultaneously "missing" and fail every `>=`
  test).

```python
b = DataSpecBuilder()
b.add_nominal("color", ["red", "green", "blue"], missing_name="<missing>", missing_values=("?",))
b.add_numeric("age", [20, 30, 40], missing_name="<missing>")
ds = b.build(missing_strategy=MissingStrategy.SEPARATE)
```

`add_boolean`/`add_set`/`add_hierarchical` accept `missing_values=` too
(for sentinel recognition) but not `missing_name=` -- `SEPARATE` isn't
supported for those attribute types.

## Rule models

A `Rule` is a conjunction of Boolean literals over a fixed feature space
(`Literal(feature)` -- a literal is just a feature index; there are no
negative literals). "Feature absent" is expressed by conditioning on a
separate *negation feature* (`not f`, `age<30`, `color!=red`), paired
with its positive counterpart by a `MutuallyExclusive` / `NominalGroup`
/ `NumericGroup` constraint -- `DataSpecBuilder` generates these by
default (`negation=False` opts out). A rule reads back out as one tuple
of feature indices (`pos`, the features that must be True) plus a single
integer bitmask, so coverage of one example is a genuine O(1)-ish subset
check:

```python
covers = (x_bits & mask) == mask
```

`Rule.from_pos_neg(pos=..., neg=...)` is still the convenient
constructor: each `neg` index is translated to its negation feature via
`DataSpec.negation_of`, so a negation-enabled `dataspec` is required
when `neg` is non-empty.

A rule can optionally carry the order its conditions were added/tested
in (`ordered=True`), and renders via `to_string(fmt=...)` as logic
notation, a Prolog-style clause, a positional pattern string, or a bare
condition list. `fmt` is optional: omitted, `to_string` (and `__repr__`,
so this is also what you see printed) falls back to that rule's own
`default_fmt` if set at construction (`Rule(..., default_fmt="logic")`),
else the class-wide `Rule.DEFAULT_FORMAT` (`"prolog"`) -- an explicit
`fmt=` always overrides both. `reorder`/`generalize` carry a rule's
`default_fmt` over to the result. A bare `Rule` carries no weight or
free-form metadata; measured per-rule statistics live on `SingleRule.stats()`,
and what built a rule on `SingleRule.provenance`.

`WeightedRule` is the `Rule` subclass that adds a weight: one declarative
number that is *part of the model* (e.g. a ProbLog-style probability), as
opposed to statistics measured against a dataset. Weights are meant to be
used when they are part of a model, but nothing in prediction uses them yet:
a clear strategy for that is still to be found.

`RuleModel` (in `pyrulearn.models`) is the shared abstract base for
collections of `Rule`s that make predictions together. The hierarchy is
organised by **resolution** -- how a prediction is decided when several
rules apply:

- **`SingleRule`** -- one rule, wrapped so it carries its own `stats`/
  `provenance` and predicts on its own; the hierarchy's base case, and
  what every container actually stores (`.rules` is a list of
  `SingleRule`s, never bare `Rule`s).
- **`RuleSet`** (abstract) -- *unordered*; covering rules are reconciled
  by `self.resolution` (a `Combine` over a `combiner`, or an `Exclusive`
  disjointness assumption). Concrete subclasses:
  - `FlatRuleSet` -- a plain bag of mixed-head rules plus one `combiner`
    (the generic, directly-instantiable case). `PooledRuleSet` is its
    memory-compact variant for huge mined rule pools.
  - `ConceptModel` -- rules that all share one head (one **concept**);
    `predict` returns that label where covered, `default_prediction`
    otherwise.
  - `ConceptSet` -- one `ConceptModel` per label, `combiner`-resolved
    where several concepts cover the same row.
  - `DisjointRuleSet` -- rules assumed pairwise mutually exclusive --
    e.g. a decision tree's leaves, whose path conditions partition the
    feature space by construction, so there's never actually a tie to
    break. `interfaces.sklearn.from_sklearn_tree` returns one
    directly.
- **`RuleList`** (abstract) -- *ordered*; the first matching rule wins
  (RIPPER/CN2-style sequential covering). This rule-level order is
  unrelated to `Rule.ordered`, which is about condition order *within*
  one rule. Concrete subclasses:
  - `DecisionList` -- a linear list of rules; the first that matches
    decides.
  - `ConceptCascade` -- ordered `ConceptModel`s, "peeling" one-vs-rest:
    the first concept that fires decides, and concept k only sees the
    rows concepts 1..k-1 didn't take.
- **`CompositeModel`** (abstract) -- members that are themselves
  `RuleModel`s, combined at prediction time. Concrete subclasses:
  - `EnsembleModel` -- a flat, optionally weighted vote over independent
    members (bagging / boosting-style ensembles, e.g. one member per
    random-forest tree).
  - `PairwiseModel` -- round robin: one binary member per label pair,
    each voting for one of its two labels.
  - `DeepModel` -- members wired into a dependency DAG ("stacking");
    currently a stub, structure only, with no `predict` yet.

### Printing a model

`to_string(fmt=None, ascii=False, data=None)` renders a whole model,
applying one resolved format (explicit `fmt=`, else `Rule.DEFAULT_FORMAT`)
to *every* rule, regardless of any individual rule's own `default_fmt`.
This is what keeps a printed model internally consistent even when its
rules weren't all built with the same `default_fmt`. Structure differs by
family:

- `RuleSet` (`FlatRuleSet`/`ConceptModel`/`DisjointRuleSet`/`ConceptSet`)
  groups rules by target label, each section headed by `% class:
  <target>`. For `fmt="logic"`, a label's rules collapse into one DNF
  expression -- each rule's conjunction parenthesized on its own line,
  `∨`-prefixed after the first, e.g. `(age_gt_30 ∧ smoker)` /
  `∨ (high_bp)` / `→ high_risk`. `fmt="prolog"` doesn't need that
  treatment: consecutive clauses sharing a head already read as a
  disjunction, so each rule just stays its own clause under the header.
  `"pattern"`/`"conditions"` have no natural merged form, so rules are
  listed under the label header unmerged.
- `RuleList` (`DecisionList`/`ConceptCascade`) doesn't group by label --
  order, not shared target, is what decision-list semantics depend on.
  `fmt="logic"` renders as if/elif/else pseudocode (order = priority,
  matching `predict`'s own first-match-wins semantics exactly); other
  formats list rules sequentially in list order.
- `default_rule` (when the policy has a constant label) gets its own
  trailing section: a `% default` header for `RuleSet`/non-logic
  `RuleList` output, or the pseudocode's own `else` line for `RuleList` +
  `fmt="logic"`.

A `WeightedRule` prints its weight as part of the rule: in front of it in
the default Prolog format (`0.8::head :- body`), and appended in the other
formats (`[0.8]` for `"logic"`, `% 0.8` otherwise). The one optional
decoration is `data=<a DataRepresentation>`, which suffixes every rule with
a trailing coverage comment, computed *fresh* against it (no separate
annotation call needed first):

- Ordinarily `% (tp/fp)` -- covered rows that are, or aren't, actually this
  rule's own target (the classic C4.5/RIPPER rule-quality notation; 0 `fp`
  reads as a perfect rule). Labels (`data.y`) are needed to compute it and a
  target to check correctness against; where either is missing, just the
  bare covered count, `% (n_covered)`.
- For a model whose own `combiner` is genuinely a `DistributionCombiner`
  (`"micro_vote"`/`"macro_vote"`/`"micro_max"`/`"macro_max"`) *and* has more
  than two classes, the full per-class breakdown instead -- `% [n0, n1, ...]`
  -- everything that combiner's own `resolve()` reads, nothing it doesn't.
  A one-line `% classes: [...]` legend, printed once above the rest of the
  output (never repeated per rule), gives that vector's order. With exactly
  two classes this vector would just be `(tp/fp)` reordered, so it's
  suppressed in favor of the plain form.

Both are choices, not hard rules -- `to_string`'s `show_distribution`/
`show_classes` (`None` by default) force either one independently, for any
model, any class count, any combiner: the raw per-class counts are always
computable from `data.y`, whether or not a given model's own resolution
actually consults them. `show_distribution=True`/`False` forces the vector
on or off outright (e.g. showing it for a plain `combiner="max"` model, or
suppressing it for a genuine `DistributionCombiner`); `show_classes=True`/
`False` independently forces the legend on or off, regardless of whether any
rule ends up showing a vector at all -- useful for naming a model's relevant
classes as a label on its own.

`EnsembleModel`/`PairwiseModel` also have `to_string` (`CompositeModel`'s
other concrete subclass, `DeepModel`, doesn't -- it's a structure-only stub
with no member-wiring semantics yet to print). Both render every member/pair
in turn plus a top-level `% classes: [...]` naming the model's own `labels`
(from its own declared structure, not `data` -- see `_container_legend`).
`EnsembleModel` headers each member `% member <k>`, with `(weight: ...)`
appended where `member_weights` is set -- exactly the number `predict`'s
plurality vote weighs that member's verdict by, so the one thing beyond each
member's own rules a reader needs to redo the vote by hand; `data`/
`show_distribution`/`show_classes` pass through unchanged to every member,
since they all cover the same overall multiclass problem. `PairwiseModel`
headers each pair `% pair: a vs b`, with `(member weight: ...)` appended for
`"accuracy_vote"` specifically -- `"weighted_vote"`'s own per-row deciding-
rule weight is exactly `Laplace` on that rule's own measured stats, already
fully reconstructable from its own printed `(tp/fp)`, so nothing extra is
needed for that combiner. Unlike `EnsembleModel`, each pair's `data` is
narrowed to just its own two classes' rows first (via `select_rows`), and
its own `show_classes` defaults to forced-on (`None` here means "force", not
"auto") -- a sub-model's rules may only ever explicitly predict *one* of its
two classes (the other only ever surfacing as its own `default_prediction`),
so without this a reader may have no way to tell which two classes a given
pair is even about.

### Converting between model types

`pyrulearn.models.convert(model, dst)` converts a model to another type when
a direct converter exists (`can_convert(src, dst)` checks; there is no
transitive closure, so chain converters explicitly). The registered ones:

- `FlatRuleSet` to `DecisionList` -- adopts the insertion order, so
  prediction changes from combiner resolution to first-match.
- `FlatRuleSet` to `ConceptSet` -- groups the rules by head; with an
  order-independent combiner, prediction is unchanged.
- `ConceptSet` to `FlatRuleSet` -- drops the per-concept structure.
- `ConceptCascade` to `DecisionList` -- prediction is preserved.
- `EnsembleModel` to `FlatRuleSet` -- lossy: pools every member's rules under
  one combiner (default `"vote"`), dropping the per-member grouping and any
  member weights.

Any `RuleSet` also has `to_rulelist(key=None, reverse=True)`, which orders
its rules with `pyrulearn.evaluation.sort_rules` (default: descending
Laplace-on-measured-stats -- pass `key=`/`reverse=` for another order) into
a `DecisionList`.

### Conflict resolution

When several rules cover the same row, a `RuleSet` reconciles them with its
`combiner` (a `RuleCombiner` in `pyrulearn.combiners`, or a string shortcut).
Pass either the string shortcut or a `RuleCombiner` instance directly to the
constructor, or override it per call via `predict(data, combiner=...)`.
Omitted, `predict` falls back to the set's own `self.combiner` (settable at
construction, itself defaulting to `"max"`), so a `RuleSet` built with a
particular strategy keeps it without every `predict` call re-passing it.

`RuleCombiner.resolve(rules, covering)` is the one abstract method
(`covering`: non-empty indices into `rules`). `ListCombiner` and
`CountVoteCombiner` stand alone; everything else falls under one of two
intermediate bases, matching two different kinds of per-rule information a
combiner can use:

- `"list"` -- `ListCombiner`: rule position, i.e. list order.
- `"vote"` -- `CountVoteCombiner`: plain, unweighted majority vote.
- `HeuristicCombiner` -- scores a `RuleHeuristic` against each rule's own
  measured stats (`SingleRule.stats()`'s `ConfusionMatrix`, rotated to the
  rule's own target), computed fresh at combine time, not baked into the
  model beforehand. Both take a `heuristic=` (default `Laplace()`):
  - `"max"` -- `HeuristicMaxCombiner`: the classic ensemble "max rule",
    which picks the single covering rule with the highest heuristic score.
  - `HeuristicVoteCombiner`: majority vote across every covering rule, each
    vote weighted by that rule's heuristic score.
- `DistributionCombiner` -- for rules scored by a full per-class breakdown
  rather than one scalar: each rule's own measured stats
  (`ConfusionMatrix.predicted_as(rule.target)`, the true-label distribution
  among the rows it actually fired on). Two independent axes give four
  concrete combiners. *Which* per-rule numbers get used: `Micro` pools every
  covering rule's raw counts (larger leaves count for more), `Macro`
  normalizes each rule's own counts to proportions first (every rule counts
  equally regardless of leaf size) -- the same micro/macro distinction used
  for multi-class F1 scores. *How* the rules' numbers are combined: `Vote`
  sums them (total support per class), `Max` takes the highest per class
  across covering rules (the classic ensemble "max rule" in its full
  per-class form, distinct from `HeuristicMaxCombiner`, which picks one whole
  rule rather than comparing per class).
  - `"micro_vote"` -- `MicroVoteCombiner`.
  - `"macro_vote"` -- `MacroVoteCombiner`. **This is the one that matches
    `sklearn.ensemble.RandomForestClassifier.predict()`'s own mechanism**:
    each tree contributes one class-probability vector (its own leaf counts
    normalized), averaged (equivalently, summed) across trees -- unlike every
    `HeuristicCombiner`/`CountVoteCombiner`, which collapses each rule down
    to a single hard vote/score before combining, discarding how confident a
    leaf actually was.
  - `"micro_max"` -- `MicroMaxCombiner`.
  - `"macro_max"` -- `MacroMaxCombiner`.

The distribution combiners are the ones to reach for when importing a whole
random forest as one `FlatRuleSet`
(`pyrulearn.interfaces.sklearn.RandomForestImporter`): each tree contributes
one always-firing rule per example, so a k-tree forest always has exactly k
covering rules, and combining across all of them is what actually reproduces
the forest's own prediction (`MacroVoteCombiner` in particular matches
sklearn's own soft-voting mechanism almost exactly).

`HeuristicMaxCombiner`, `HeuristicVoteCombiner` and `DistributionCombiner`
all **require** every covering rule to have measured stats when the covering
rules genuinely disagree in target, and raise `ValueError` otherwise (an
unmeasured rule contributing a silently wrong score is exactly the kind of
quiet wrong answer this stats-based design exists to rule out). Annotate
first: `pyrulearn.models.annotate_rules` (see *Statistics*, below), or a
`fit()`/importer `data=` call, which already do. A *unanimous* covering block
(every covering rule already agreeing on the target) never needs stats at all,
as there is no actual disagreement to resolve. `DistributionCombiner` also
raises if a covering rule has neither measured stats nor `class_counts` --
there is no further fallback.

Whether conflicts can occur at all is a check you can run: `is_disjoint(data)`
is available on any `RuleModel`, not just `DisjointRuleSet`, so a plain
`FlatRuleSet` can be checked *before* deciding whether it's safe to
convert, not only verified after the fact. Disjointness isn't enforced at
construction, the same way `Rule.is_consistent()` is a check you call, not a
built-in guarantee.

### Default predictions

A model isn't necessarily *exhaustive*: some rows may match no rule.
`is_exhaustive(data)` checks that against real data (like `is_disjoint`, on
any `RuleModel`).

Every `RuleModel` has a `default_prediction` (settable at construction or
afterward), the *policy* for rows no rule in `self.rules` decides. It is one
of:

- a bare label -- `predict` returns it;
- `None` -- abstain, `predict` returns `None`;
- a `DefaultPrediction` object -- consulted per uncovered row via
  `predict(rules, data, example_idx)`. This is not vectorized, so different
  rows can genuinely get different fallbacks. `MajorityClass` (the built-in
  one) resolves a fixed majority label at construction from a chosen row
  subset; "predict via whichever rule is most similar to this row" would be
  another.

`filter`/`to_rulelist`/`remap` all carry the policy over unchanged, and so
does `.provenance` (what built this model -- a `RuleLearner`/`RuleImporter`
stamps it; narrowing/rebasing/repackaging doesn't change who built the
rules, so it survives all three).

`default_rule` is a read-only, lazily-materialized `SingleRule` view of that
policy: its `target` is the policy's constant label (a bare label, or a
`DefaultPrediction`'s `constant_target`), or `None` for a `None` policy or a
non-constant strategy. Being a real `SingleRule`, it carries its own
`stats`/`provenance` like any other rule. Reassigning `default_prediction`
discards the materialized rule and its stats.

### Explaining a prediction

`RuleModel.covered_by(data, by=None)` returns, per row, every rule whose
body holds, ordered as an explanation of the prediction: the predicted
label's block first, other labels' blocks after (each ordered by its
strongest rule), rules within a block sorted by `by` (a per-rule key
callable; default: descending Laplace on the rule's own *measured* stats --
raises if a genuine multi-rule block has a rule with none). Uncovered rows
yield `[default_rule]` or `[]`. For a richer heuristic-based ranking
(needing `data=` to score against), use
`pyrulearn.evaluation.sort_rules(rules, by=<a RuleHeuristic>, data=data)`
directly on `model.covered_by(data)`'s output, or as `covered_by`'s own
`by=` argument.

### Statistics

`stats(data=None, split="data")`: every `RuleModel` (down to each
`SingleRule` leaf) can measure its own performance against `data`. It
returns a `pyrulearn.evaluation.ModelStats` snapshot (a `ConfusionMatrix`
from `predict(data)` vs `data.y`, plus `n_rules`/`n_conditions`), cached
under `split` (pass a different `split` name, e.g. `"train"` then
`"test"`, to keep several side by side).

`pyrulearn.models.annotate_rules(rules, data)` wraps a plain rule list and
stats each one against `data` in one call. This is what every native
learner and importer `fit()`/`data=` round trip already does, so
combiners/`sort_rules`/`covered_by` have real measured stats to score from.

### Rule-evaluation heuristics

`RuleHeuristic` (in `pyrulearn.heuristics`) is a pluggable rule-quality
score, evaluated against `RuleStats` -- the confusion-matrix quartet
`tp`/`fp`/`fn`/`tn` (treating "covers" as "predicts positive"), plus
optional `length` and `parent` -- rather than positional args, so adding
a new stat later doesn't force every heuristic's signature to change.
`length`/`parent` deliberately aren't part of `Rule` itself: they're
heuristic-search context (a bare rule has no inherent notion of its own
refinement history), not coverage information, which is why `RuleStats`
lives here rather than in `pyrulearn.rule`.

```python
class RuleHeuristic(ABC):
    @abstractmethod
    def score(self, stats: RuleStats) -> float: ...   # higher = more preferred
    def score_rule(self, rule, data, positive_class=None) -> float: ...  # convenience
```

Every heuristic follows the same "higher = more preferred" convention,
so any of them can be dropped in as a ranking key without the caller
caring which one is active -- `Rule.order_by_precision`'s greedy
criterion is one hardcoded instance of this general idea. `pyrulearn.
evaluation.sort_rules(rules, by=None, data=None, descending=True)` is
the general-purpose consumer: `by=None` ranks by descending
Laplace-on-measured-stats (the same default `HeuristicMaxCombiner`
uses), a `RuleHeuristic` scores fresh against `data` (unless
`needs_data` is `False`, e.g. `MinimalLength`), or pass a plain callable.
`RuleStats.from_rule(rule, data, positive_class=None, example_mask=None)`
computes `tp`/`fp`/`fn`/`tn` from a rule's actual coverage
(`positive_class` defaulting to `rule.target`) and `length` from
`rule.length()`. A gain-style heuristic needing a *parent* rule's stats
too (see `GainHeuristic`, below) just calls this a second time on the
parent rule -- there's no dedicated parameter for it here.

Built in (all in `pyrulearn.heuristics`). The four confusion-matrix quadrant
counts ("covered/uncovered" x "positives/negatives"); the sign follows whether
the quadrant is a correct outcome (`CoveredPositives`/`UncoveredNegatives`,
scored directly) or an error (`CoveredNegatives`/`UncoveredPositives`,
negated), so more of it is never accidentally "better":

- `CoveredPositives` -- tp alone; ignores fp entirely.
- `CoveredNegatives` -- -fp alone; ignores tp entirely.
- `UncoveredPositives` -- -fn.
- `UncoveredNegatives` -- tn.

Precision-like heuristics:

- `Precision` -- tp/(tp+fp); also called Confidence in association-rule-mining
  terminology.
- `Recall` -- tp/n_pos, a.k.a. sensitivity/TPR/hit rate. `CoveredPositives`'s
  own rate-normalized twin (the same relationship `Support` has to
  `Coverage`); not to be confused with raw `CoveredPositives`, a count, not a
  rate.
- `FBeta(beta=1.0)` -- the weighted harmonic mean of `Precision` and `Recall`;
  `beta=1`, the default, is the standard F1 score, weighing them equally,
  `beta<1` favors precision, `beta>1` favors recall. Rewritten as
  `(1+beta**2)*tp / (tp+fp+beta**2*n_pos)`, its isometrics turn out to be a
  pencil pivoting at `(-beta**2*n_pos, 0)`, the same fp-axis sub-family as
  `GHeuristic`, just tied to the dataset's actual class balance instead of a
  free constant.
- `Laplace` -- Laplace-smoothed precision.
- `MEstimate(m)` -- generalizes `Precision` at m=0, pulled toward the prior
  positive rate as m grows. Isometrics: a pencil pivoting at
  `(-m*(1-p0), -m*p0)`.
- `GeneralizedMEstimate(m, cost)` -- replaces `MEstimate`'s prior `p0` with a
  free `cost` parameter (`MEstimate(m)` is exactly
  `GeneralizedMEstimate(m, cost=p0)`); freeing `cost` lets the pivot land
  anywhere on the line `fp+tp=-m`, not just at the single point the dataset's
  own prior would put it.
- `GHeuristic(g)` -- Gamberger & Lavrač's expert-guided subgroup discovery "g
  heuristic", used in CN2-SD: tp/(fp+g), where `g` is meant to be a positive
  constant; larger `g` tolerates more covered negatives, favoring more
  general rules.

Accuracy-, coverage- and cost-style heuristics:

- `WRAcc` -- weighted relative accuracy.
- `YoudenJ` -- Youden's J statistic (Youden, 1950), tpr - fpr =
  tp/n_pos - fp/n_neg; an alternative to `WRAcc` that trades off true/false
  positive *rates* rather than raw coverage-weighted counts. Also known as
  informedness, the vertical distance above the random-guess diagonal in ROC
  space.
- `Accuracy`.
- `CoverageDifference` -- tp - fp; ranks identically to `Accuracy` (same
  isometrics) but skips the normalizing division, so prefer it when only
  relative order matters and `Accuracy` when the actual percentage does. Also
  exactly `LinearCost(cost_ratio=1.0)`, kept as its own name for
  discoverability.
- `Support` -- coverage rate, ignoring purity.
- `Coverage` -- tp + fp; the unnormalized version of `Support`, the same
  relationship `CoverageDifference` has to `Accuracy`.
- `LinearCost(cost_ratio=1.0)` -- tp - cost_ratio*fp.
- `LinearCostRates(cost_ratio=1.0)` -- tpr - cost_ratio*fpr; the rate-space
  counterpart to `LinearCost`, the same way `YoudenJ` is to
  `CoverageDifference` (`YoudenJ` is exactly `LinearCostRates(cost_ratio=1.0)`).

Statistical and information-theoretic heuristics:

- `Correlation` -- FOSSIL's four-field/Matthews correlation coefficient
  between "rule covers this example" and "example is positive",
  `(tp*tn - fp*fn) / sqrt((tp+fp)(tp+fn)(fp+tn)(fn+tn))`, in `[-1, 1]`.
- `ChiSquare` -- Pearson's chi-square statistic for the same 2x2 table
  (`n * phi**2`, optionally Yates-corrected); CMAR's significance test.
- `Entropy` -- CN2's original heuristic: the *negated* binary entropy of the
  covered class distribution, `p*log2(p) + (1-p)*log2(1-p)` with
  `p = tp/(tp+fp)`. Negated so a pure rule scores 0, the best, and an even
  50/50 split scores -1 bit, the worst, matching this module's convention
  where CN2 itself minimizes plain entropy directly.
- `LikelihoodRatio` -- CN2's significance-testing statistic: the G-test /
  log-likelihood-ratio comparing a rule's covered class counts against what
  the dataset's prior alone would predict, `2*(tp*ln(tp/e_tp) + fp*ln(fp/e_fp))`
  with `e_tp`/`e_fp` the prior-implied expected counts. Higher means further
  from chance, already matching the higher-is-better convention with no sign
  flip.

Heuristics that read past `tp`/`fp`/`fn`/`tn`, or compose others:

- `LengthPenalized(base, penalty)` -- wraps another heuristic and subtracts
  `penalty * stats.length` (a simple Occam's-razor-style complexity penalty).
- `MinimalLength` -- `-length`: shorter rules score higher. Meant purely as a
  lexicographic tie-break inside a `LEF`, not as a heuristic on its own.
- `LEF` -- Michalski's Lexicographic Evaluation Functional (AQ): an ordered
  list of heuristics, where the first one decides unless it ties. It is itself
  a `RuleHeuristic`, so it works anywhere one is expected.
- `FoilGain` -- Quinlan's information gain, `tp * (log2(tp/(tp+fp)) -
  log2(parent.tp/(parent.tp+parent.fp)))`, the one built-in `GainHeuristic`.
- `DeltaGain(base)` -- turns any ordinary heuristic into a `GainHeuristic`:
  scores a refinement by how much `base`'s own score improved relative to its
  parent.

A `GainHeuristic`'s `score(stats, parent_stats)` takes the parent rule's
`RuleStats` as a *mandatory second argument*, not an optional field read off
`stats`: a gain score is only meaningful relative to one specific parent, not
on a shared scale comparable across different parents/search steps, so a
caller needs to know it's holding a `GainHeuristic` (`isinstance(heuristic,
GainHeuristic)`) and supply the parent explicitly; calling
`FoilGain().score(stats)` with one argument raises `TypeError`.
`RuleStats.universal(n_pos, n_neg)`/`RuleStats.empty(n_pos, n_neg)` give the
"covers everyone"/"covers no one" degenerate stats directly from a dataset's
totals, without needing an actual `Rule` on hand -- the natural parent when
there's no specific prior refinement to compare against, e.g. for
`RuleHeuristic.plot_isometrics`'s `parent=` argument, below.

How heuristics differ geometrically -- whether their isometrics (curves of
constant score in coverage space) are a pencil of lines, parallel lines, or
curves -- is described under *Isometrics* in *Coverage space*, below.

### Coverage space

Rules and rule models can be analysed in coverage space (Fürnkranz & Flach,
2005): a rule is a point (negatives covered, positives covered), so a rule
set, a decision list or a single rule's refinement can be read off a plot.
Every `RuleModel` gives the coordinates, and `pyrulearn.evaluation` adds the
plots and the convex-hull AUC:

- `coverage_space(data, positive_class)` gives the classic (negatives
  covered, positives covered) coordinates for one-vs-rest analysis, using
  the raw, order-independent coverage, the same for every type.
- `RuleList.coverage_path(data, positive_class)` gives the *cumulative*
  (negatives, positives) coverage as rules are tried in order, starting at
  `(0, 0)`: point i is what's been decided after the first i rules, using
  each one's unique/fired coverage.

`coverage_space_plot(ruleset, data, positive_class)` (in
`pyrulearn.evaluation`) adapts to the classifier type: a `RuleSet` is drawn
as a scatter of *every* rule's own raw coverage w.r.t. `positive_class`
(regardless of the rule's own target), optionally with
`build_refinement_graph` edges connecting rules that differ by one
literal. `show_convex_hull=True` additionally draws the boundary of the
area under the ROC-convex-hull curve -- `(0, 0)` to `(N, 0)` to `(N, P)`
and back to `(0, 0)` via the hull itself, *not* a bare closing edge
straight back to `(0, 0)` (which would just sit on top of the
random-guess diagonal) -- restricted to just the rules whose own
`target == positive_class`: the hull/AUC are only meaningful for one
target class at a time, so a mixed-target `RuleSet` needs one call per
class rather than one hull mixing them (`coverage_space_auc(ruleset,
data, positive_class)` gives the number directly -- exactly the
enclosed area normalized by `N * P` -- and is also shown in the
boundary's legend label). A `RuleList` is instead drawn as
`coverage_path`'s points connected by directional **arrows**, in order,
starting at `(0, 0)` (`show_refinements`/`show_convex_hull` don't apply
in this mode -- the arrows already show the sequential structure). Pass
`aspect="square"` (old behavior), `"rectangular"` (box width:height =
negatives:positives, unclamped), or the default `"auto"` (same, but
clamped to `[1/max_aspect_ratio, max_aspect_ratio]` so a very skewed
dataset still gets a readable plot rather than a sliver).

`rule_refinement_plot(rule, data, positive_class=None)` plots a
single `Rule`'s own specialization path instead, also as directional
arrows: starting at `(N, P)` (the empty rule, covering everyone) and
adding one condition at a time in the rule's own order -- monotonically
non-increasing in both coordinates, ending at the rule's actual coverage.
`positive_class` defaults to `rule.target` -- for a single rule,
"positive" naturally means the class *it* predicts (its head), so
plotting a rule that predicts `"neg"` without overriding `positive_class`
correctly counts `"neg"` examples as the true positives, not whichever
class happens to be the caller's usual default. A rule needs a
meaningful order for this to say much; `Rule.order_by_precision(data)`
computes one greedily (FOIL/CN2-style: at each step, add whichever
remaining condition gives the highest precision *in combination with*
what's already chosen) if the rule doesn't already have one --
`Rule.reorder(...)` sets one explicitly instead.

#### Isometrics: pencil, parallel and curved

An *isometric* is a curve of constant heuristic score in coverage space
(`RuleHeuristic.plot_isometrics`, below, draws them), and its shape sorts the
heuristics into three groups (see Fürnkranz & Flach, *"ROC 'n' Rule
Learning"*, Machine Learning 2005, for the full analysis). In their
terminology the isometrics of the first two groups, pencil and parallel, are
both *linear* (straight lines), and those of the third are *non-linear*.

- **Pencil: straight lines through a common pivot point.** `Precision`,
  `Laplace`, `MEstimate`, `GeneralizedMEstimate`, `GHeuristic` and `FBeta`.
  The pivot is the origin for `Precision`, `(-1,-1)` for `Laplace`, a
  prior-dependent sliding point on the line `fp+tp=-m` for `MEstimate`
  (freely placeable on that same line via `GeneralizedMEstimate`'s `cost`
  parameter), `(-g, 0)` on the fp-axis for `GHeuristic`, and
  `(-beta**2*n_pos, 0)`, also on the fp-axis, for `FBeta`. Because the lines
  are not parallel, these heuristics can prefer a lucky low-coverage rule
  over a robust one. `Entropy` belongs here too -- same pivot as `Precision`
  (it also depends only on `p = tp/(tp+fp)`, constant along any ray from the
  origin) -- but each score value is a symmetric *pair* of rays, not one:
  `h(p) == h(1-p)`, so entropy alone can't tell a 90%-positive rule from a
  90%-*negative* one apart.
- **Parallel: parallel lines.** `CoveredPositives`, `CoveredNegatives`,
  `UncoveredPositives`, `UncoveredNegatives`, `Recall`, `WRAcc`, `Accuracy`,
  `CoverageDifference`, `Support`, `Coverage`, `LinearCost`, `YoudenJ` and
  `LinearCostRates`. `CoveredPositives`/`CoveredNegatives` are this group's
  two degenerate limits, horizontal/vertical, i.e. zero weight on the other
  variable, with `UncoveredPositives`/`UncoveredNegatives` in the same two
  groups, just affine-shifted by the constant `n_pos`/`n_neg`; `Recall` is
  `CoveredPositives`'s rate-normalized twin. A rule is optimal for *some* such
  heuristic if and only if it lies on `coverage_space_plot`'s convex hull.
- **Curved (non-linear).** `Correlation` (hyperbolic isometrics) and
  `LikelihoodRatio` (isometrics that bow away from the origin, since it
  depends on absolute covered counts, not just their ratio); and `FoilGain`,
  which, once made plottable via a fixed `parent`, isn't even a single smooth
  curve -- its `tp *` factor can make a low-tp, low-precision point score the
  same as a higher-tp, higher-precision one, producing a visibly non-convex,
  "hooked" isometric.

#### CoverageSpace: layering heuristics, rulesets, and refinement paths

`coverage_space_plot`/`rule_refinement_plot` are thin, standalone-figure
wrappers around `CoverageSpace` (also in `pyrulearn.evaluation`) -- the
reusable, composable version of the same idea. A `CoverageSpace` owns
the plot's dimensions (`n_pos`/`n_neg`) and its matplotlib `Axes`;
`plot_ruleset`/`plot_rule_refinement` draw a layer into it (the same
content as the two standalone functions, minus the one-off figure
setup), and `RuleHeuristic.plot_isometrics` (in `pyrulearn.heuristics`,
inherited by every concrete heuristic since it only needs `score`) draws
that heuristic's isometrics -- curves of constant score -- into one too,
so several layers can share one plot:

```python
from pyrulearn.evaluation import CoverageSpace
from pyrulearn.heuristics import WRAcc

space = CoverageSpace.from_data(train_rep, positive_class="pos")
WRAcc().plot_isometrics(space=space, levels=10, filled=True)
space.plot_ruleset(rules, train_rep, "pos", show_refinements=False, annotate=True)
```

`n_pos`/`n_neg` are either given explicitly, taken from real data via
`CoverageSpace.from_data(data, positive_class)`, or
-- if both are omitted -- default to `100`/`160` (the smaller value on
Pos), a fixed, dataset-agnostic space for looking at a heuristic's
isometrics in the abstract (e.g. bare `Precision().plot_isometrics()`,
no `data` needed at all). `normalized=True` switches to the
classic **ROC space** (`FPR`/`TPR` in `[0, 1] x [0, 1]`) instead of raw
counts -- the heuristic is still evaluated against the space's real
`n_pos`/`n_neg` internally (most heuristics, e.g. `WRAcc`, genuinely
depend on the class prior, not just the displayed rate), only the
*display* coordinates are normalized. A normalized space's box is always
displayed as a perfect square, regardless of `n_pos`/`n_neg` or
`aspect=` -- the conventional ROC-space shape; a non-normalized space's
box genuinely reflects the true negatives:positives ratio (`aspect=
"rectangular"` for the unclamped ratio, the default `"auto"` for the
same ratio clamped to `[1/max_aspect_ratio, max_aspect_ratio]` so a
heavily skewed dataset doesn't render as an unreadable sliver, or
`"square"` to ignore the ratio entirely).

`show_labels` controls whether *concrete numbers* -- tick labels beyond
the two endpoints, and isometric value labels (`ax.clabel`) -- are drawn
at all. `None` (the default) means "yes, if `n_pos`/`n_neg` were
actually given", "no" otherwise, since the default space's `100`/`160`
are arbitrary placeholders and printing them as if they meant something
would be misleading; pass `True`/`False` explicitly to override either
way. Axis text (`"negatives covered"`/`"positives covered"`, or
`"FPR"`/`"TPR"` when normalized) and the two endpoint ticks are always
shown regardless -- only the tick *labels* change: symbolic `"0"`/`"N"`
and `"0"`/`"P"` when `show_labels` is `False` and the space isn't
normalized (the real counts aren't meaningful), literal `"0"`/`"1"` when
it is normalized (that endpoint is always true), or the real numbers
(with matplotlib's normal, denser default ticks) when `show_labels` is
`True`. A light gray minor-tick grid is always drawn too, styled like
the random-guess diagonal, for visual reference regardless of whether
concrete numbers are shown. The plot's `title` (if given) is shown
regardless of `show_labels`.

`plot_isometrics(levels=10, filled=False, cmap="viridis")` draws line
contours (label text via `ax.clabel`, skipped when `show_labels` is
`False`) by default, or filled bands with `filled=True`; `levels` is
passed straight to matplotlib's `contour`/`contourf`, as is everything
else passed via `**kwargs` -- `colors=`, `linestyles=`, `linewidths=`,
etc. are all genuinely configurable this way, e.g. `colors="tab:blue",
linestyles="dashed", linewidths=0.8` to keep one heuristic's isometrics
visually distinct from another's (or from an overlaid rule's own
refinement-path color) when layering several on the same space --
`cmap` is dropped automatically when `colors` is given, since matplotlib
rejects both together. `parent=` fixes the parent `RuleStats` passed to
a `GainHeuristic`'s `score` across every grid point -- the same value
throughout, not one per point -- which is what makes a gain-style
heuristic like `FoilGain` plottable at all here (it otherwise raises
without one); `RuleStats.universal(space.n_pos,
space.n_neg)` is the natural choice, reading the result as "gain over
the rule that covers everyone." Plotting several heuristics' isometrics
side by side makes their families concrete -- `Precision`/`Laplace` fan
out from the origin, `WRAcc` is parallel lines, `Correlation` bows into
hyperbolic curves -- and, empirically, that `Entropy`'s isometrics are
*also* a pencil through the origin (same pivot as `Precision`), just
with each score value split into a symmetric pair of rays, and that
`FoilGain`'s (with a fixed parent) aren't even a single smooth curve
(see *Isometrics*, above). See `examples/demo_heuristics.py`
for all of this in practice, including layering `Precision`/`Accuracy`
isometrics under a rule's own refinement path.

`plot_isometric_through_point(neg, pos, space=None, mark_point=True,
...)` draws just the *one* isometric that passes through a specific raw
coverage point -- every other point on that curve scores exactly as well,
under this heuristic, as `(neg, pos)` does -- and scatters the point
itself so it's clear which one. `plot_isometric_through_rule(rule,
data, positive_class=None, space=None, ...)` is the common-case
convenience: the point is `rule`'s own raw coverage (`positive_class`
defaulting to `rule.target`, same resolution as `RuleStats.from_rule`),
and `space` defaults to `CoverageSpace.from_data(data,
positive_class)` -- sized to the real data the rule was scored against,
not the dataset-agnostic default. Both accept the same styling `**kwargs`
as `plot_isometrics` (they're built on it).

## Learning algorithms

`RuleLearner` (in `pyrulearn.learners`) is the shared entry point for
running a comparative experiment across several rule-learning
algorithms without the driver caring which is which:

```python
class RuleLearner(ABC):
    def fit(self, data: DataRepresentation, model: Optional[type] = None, **model_kwargs) -> RuleModel: ...
```

**`ExternalRuleLearner`** -- runs an external algorithm (e.g. a
scikit-learn estimator) on `data`'s data, then hands the
fitted model to an existing `ObjectRuleImporter` (declared via the
`IMPORTER` class attribute) to convert it into rules. This is
deliberately *not* a second hierarchy parallel to `RuleImporter`: a
learner reuses its importer's rule-extraction logic via composition
rather than duplicating it, and lives in the *same file* as that
importer -- `pyrulearn.interfaces.sklearn.DecisionTree`/
`RandomForest` sit right next to `SklearnTreeImporter`/
`RandomForestImporter`. So adding support for one more external
algorithm means adding one file (or, if its importer already exists,
one small class in it) -- never two separately registered/maintained
class hierarchies to keep in sync.

Breaks down into three named steps: `prepare` (converts a
`DataRepresentation` into the algorithm's own native input format; default
is `data.X` as-is, right for most estimators) -> `fit_external` (calls the
algorithm) -> `import_model`, bound to the *existing* `DataSpec` (no new
features, just lookups). `fit(data)` composes them automatically and is the
only method most callers need. Concrete subclasses implement `fit_external`
(required) and `prepare` (only if the native format genuinely differs from
`data.X`).

```python
from pyrulearn.interfaces.sklearn import DecisionTree, RandomForest

learners = [
    DecisionTree(max_depth=4, random_state=0),
    RandomForest(n_estimators=10, max_depth=4, random_state=0),
]
for learner in learners:
    rules = learner.fit(train_rep)   # DisjointRuleSet or RuleSet, depending on the algorithm
    print(type(rules).__name__, len(rules), "rules")
```

An algorithm can also be fit directly on its own raw, un-pre-binarized
input (numeric, already-encoded categorical, or a mix -- whatever sklearn
itself was trained on; no threshold cap chosen ahead of time), with no
`DataSpec` yet: call `fit_external` yourself, discover a `DataSpec` from the
fitted model (`ObjectRuleImporter.infer_dataspec`, see below), then
`import_model` against it, this time actually adding features as they're
found:

```python
from pyrulearn.interfaces.sklearn import DecisionTree, SklearnTreeImporter

learner = DecisionTree(max_depth=4, random_state=0)
model = learner.fit_external(X_raw, y, feature_names=["age", "income", "score"])

importer = SklearnTreeImporter()
ds = importer.infer_dataspec(model, feature_names=["age", "income", "score"])
rules = importer.import_model(model, ds, feature_names=["age", "income", "score"])
```

`pyrulearn.interfaces.wittgenstein.IREP`/`RIPPERk` are the same
idea, wrapping `wittgenstein.IREP`/`wittgenstein.RIPPER`. `pos_class` is
required (pyrulearn never guesses which class is "positive"); `neg_class`
is optional and, if omitted, auto-resolved from `data.y` when
there's exactly one other label besides `pos_class` -- with more than
one other label present (real multi-class, which wittgenstein itself
doesn't support natively), `default_prediction` is left `None` unless
`neg_class` is given explicitly:

```python
from pyrulearn.interfaces.wittgenstein import IREP, RIPPERk

learners = [
    IREP(pos_class="pos"),
    RIPPERk(pos_class="pos", k=2, random_state=0),
]
for learner in learners:
    rules = learner.fit(train_rep)  # RuleSet, default_prediction wired up automatically
```

**`NativeRuleLearner`** -- for a rule-induction algorithm implemented
directly in pyrulearn (e.g. `pyrulearn.learners.seco`'s from-scratch Pypper/CN2/
AQR, or `pyrulearn.learners.pylord.PyLORD`): `fit` induces straight against
`data`, with no external algorithm call and no `RuleImporter`
round-trip at all.

### Multiclass classification

Multiclass support is one `fit(data, model=...)` switcher
(`pyrulearn.learners.DecomposingLearner`) shared by every learner family
-- native (the SeCo family, `PyLORD`) and external alike
(`RelabelingExternalLearner` -- sklearn's `DecisionTree`/`RandomForest`;
`_WittgensteinLearner` -- wittgenstein's `IREP`/`RIPPERk`). `fit(data,
model=ConceptSet | ConceptCascade | PairwiseModel)` decomposes into
per-class binary sub-fits, each through the learner's own `_fit_binary`.
`pyrulearn.learners.multiclass` wraps these three as thin sugar over the same
calls, for the older call style or to bundle a `random_state`/non-default
`combiner`:

- `model=ConceptSet` (`OneVsRest(base_learner)`) — one `ConceptModel`
  per class, pooled into a `ConceptSet`; `default_prediction =
  MajorityClass(data)` (fires only for a row no class's rules cover),
  `combiner = "max"`. Reassign either on the result.
- `model=ConceptCascade, order="least_frequent"`
  (`OrderedOneVsRest(base_learner, order=...)`) — peel the classes off
  one at a time (`least_frequent` / `most_frequent` / `"random"` / an
  explicit sequence): model c1 vs. the rest, then c2 vs. `{c3..cn}` on
  the rows not yet peeled (`DataRepresentation.select_rows`), … the
  last class becomes the cascade's catch-all `default_prediction`.
  `least_frequent` (default) leaves the majority class as the catch-all
  and tends to give the most compact list.
- `model=PairwiseModel, positive="smaller"`
  (`Pairwise(base_learner, positive=...)`) — one binary model per
  unordered class pair ("round robin"; Fürnkranz, JMLR 2002), each
  trained on just that pair's rows, `positive` picking the pair's
  positive label: `"smaller"` (default) / `"larger"` / `"random"` /a
  callable `f(a, b) -> pos`, or `"both"` (double round robin — one
  model per *ordered* pair, `2·C(n,2)` models, so an asymmetric learner
  gets a member for every class). At predict time each member votes and
  a `PairwiseCombiner` (`pyrulearn.models`) turns the votes into a
  label:
  - `combiner="vote"` → `MajorityVote` — one hard vote per member;
    `tie_break` `"direct"`/`"prior"`/`"first"`/callable.
  - `combiner="weighted_vote"` → `WeightedVote` — the deciding rule's
    weight `p_ij` (`Laplace` on its own *measured* stats, computed
    fresh at predict time, 0.5 if it has none) goes to the predicted
    label, `1 - p_ij` to the other (clamped to `[0, 1]`; pulled per row
    via `covered_by`). Swap it onto an already-fitted model
    (`pm.combiner = WeightedVote()`) --
    no retraining, just re-predict. `WeightedVote` is a real gain when
    per-rule reliability varies (`PyLORD`'s m-estimate scoring;
    imprecise base learners on many-class sets) but roughly neutral
    otherwise — `MajorityVote` stays the default.
  - `combiner="accuracy_vote"` → `AccuracyWeightedVote` — same
    `p_ij` / `1 - p_ij` split, but `p_ij` is the *member's* accuracy on
    its own two-class sub-problem (one scalar per member, constant over
    rows, recorded once at fit time as `PairwiseModel.member_weights`
    by `DecomposingLearner._fit_pairwise`), so a cleaner pair pulls
    harder -- also a no-retrain combiner swap.

  `MajorityVote.scores()` / `WeightedVote.scores()` return a per-label
  score vector, so the same aggregation feeds a future label-ranking
  layer (argsort instead of argmax).

The `SeCo` family reaches for this itself: `fit(data)` with no
`target_class` and no `model=` builds each learner's own multi-class
default (`_MULTICLASS_DEFAULT`) — `ConceptSet` (one-vs-rest) for
`CN2`/`PFoil`/`PFossil`/`Pypper`, so `CN2().fit(iris_rep)` just works;
`FlatRuleSet` for `AQR` (one seed-covering loop over all classes at
once, each rule seeded on a random uncovered example and headed with
its own label — AQ's multi-class covering, the same per-example seeding
`PyLORD` does). Pass `model=ConceptCascade`/`PairwiseModel` explicitly
for the other decompositions on any of them.

## Interfacing with external learners

An external learner is a two-way bridge. `fit(data)` first hands the
`DataRepresentation` to the tool in the format it expects
(`ExternalRuleLearner.prepare`, then `fit_external`), and then an importer
reads the fitted model, or the tool's printed output, back as rules bound to
the same `DataSpec`. This section covers both directions: first what each tool
receives, then the importers.

### What each tool receives

| Package | Learner classes | Rules imported | What is handed over | Feature names |
|---|---|---|---|---|
| scikit-learn | `DecisionTree`, `RandomForest` | directly | `data.X` as-is: the Boolean feature matrix, in memory | not needed to fit; the importer binds the thresholds back to the `DataSpec` |
| wittgenstein | `IREP`, `RIPPERk` | directly | `data.X` and the labels, in memory; binary only, so `pos_class` is required | the `DataSpec`'s own feature names |
| imodels | `BayesianRuleList`, `BayesianRuleSet` | directly | `data.X` as a 0/1 integer matrix, in memory (a Boolean dtype breaks BRL); BRS refuses anything that isn't 0/1; binary only | the `DataSpec`'s own feature names |
| Weka | `JRip`, `PART`, `J48` | by parsing text | an ARFF file in a temporary directory: every feature a `{False,True}` nominal attribute, plus a `class` column; run as `java -cp weka.jar <classifier> -t train.arff -no-cv` | placeholders `f0..fN` |
| LORD | `LordJar` | by parsing text | a CSV in a temporary directory (`data_train_01.csv`, plus the same rows as the test file LORD insists on): 0/1 columns with the class as the last column; run as `java -cp <classpath> run.LordRun` | placeholders `f0..fN` |
| pyarc | `PyarcCBA` | directly | transactions (via `TransactionDB.from_DataFrame`) holding only each row's True features plus the class; False cells are dropped, which matches this library's own item semantics | placeholders `f0..fN` |

Rules are imported either directly from the fitted model object (an
`ObjectRuleImporter`) or by parsing text the tool printed or wrote (a
`StringRuleImporter`); see *Importer base classes* and *Text formats*, below.

Tools that can't cope with a `DataSpec`'s feature names (`age>=30`,
`color=red`, ...) get the placeholders instead, and the matching importer is
then created with `placeholder_features=True` so that the rules bind back to
the real `DataSpec` by column position. Files are written to temporary
directories that are removed after the run. The Weka and LORD runners can also
be used on their own (`run_weka`, `run_lord`).

### Importer base classes

`RuleImporter` (in `pyrulearn.interfaces`) is the shared base for
everything that brings rules into pyrulearn from an external source --
not instantiated directly. Two intermediate layers derive from it,
matching the two shapes a source actually comes in, and every concrete
importer (of either shape) registers itself in one shared registry:

```python
class RuleImporter(ABC):
    SOURCE: str  # e.g. "sklearn.tree.DecisionTreeClassifier"
    def _stamp_rule_provenance(self, rules, **params): ...   # wraps as SingleRule, sets .provenance
    def _stamp_provenance(self, model, **params): ...         # sets the returned model's own Provenance

class ObjectRuleImporter(RuleImporter):   # an already-fitted live model object
    @abstractmethod
    def import_model(self, model, dataspec: DataSpec) -> RuleModel | Sequence[RuleModel]: ...

class StringRuleImporter(RuleImporter):   # a string/serialized rule format
    @abstractmethod
    def parse(self, source) -> RuleModel: ...

register_importer(name: str, cls)         # e.g. register_importer("sklearn_tree", SklearnTreeImporter)
get_importer(name: str) -> Type[RuleImporter]
```

**`ObjectRuleImporter`** -- `pyrulearn.interfaces.sklearn` has two
concrete ones so far, both grouped in the same module since they share
the same per-tree leaf-extraction logic (`_rules_from_tree`) -- concrete
importers are grouped by shared implementation, not rigidly one file
per library or one per algorithm. `dataspec` is the caller's *own*
`DataSpec` -- rules bind to it directly rather than a throwaway shadow
copy built from just its feature names, so imported rules share typed
display/constraints and object identity with the rest of the pipeline
(no separate object to `remap` later, if you're already using one
shared `DataSpec`).

### Text formats: `StringRuleImporter`

**`StringRuleImporter`** -- for a string/serialized rule format (RIPPER/JRip,
CN2, FOIL, CBA/CMAR-style association rules, decision-list text dumps,
...); named after the Python-level shape of the input, same as
`ObjectRuleImporter` (`object` vs. `str`), so a hypothetical future
importer of some other shape (e.g. an image) wouldn't be ambiguously
grouped in with either. `pyrulearn.interfaces.base.PatternStringImporter`
is the reference implementation -- one rule per line, a whitespace-/
comma-separated pattern of 0/1/`-` tokens (`Rule.from_pattern_string`),
where `1` at position i means feature i is a condition of the rule and
`0`/`-` mean it is not (there is no "required-False" token any more --
negation is its own feature), optionally followed by `=> target`:

```python
from pyrulearn import PatternStringImporter

importer = PatternStringImporter(feature_names=["f0", "f1", "f2"])
rules = importer.parse("1 - - => pos\n- 1 1 => neg\n")
```

### scikit-learn: decision trees and random forests

`SklearnTreeImporter` extracts one `DecisionTreeClassifier`'s leaves as
a `DisjointRuleSet` (a tree's leaves are pairwise disjoint by
construction):

```python
from pyrulearn.interfaces.sklearn import SklearnTreeImporter, from_sklearn_tree

rules = SklearnTreeImporter().import_model(fitted_tree, my_dataspec)
# or the convenience wrapper (also accepts feature_names= if you don't
# have a DataSpec yet):
rules = from_sklearn_tree(fitted_tree, dataspec=my_dataspec)
```

Both `SklearnTreeImporter` and `RandomForestImporter` also implement
`infer_dataspec(model, feature_names)` (the generic `ObjectRuleImporter`
hook -- see `pyrulearn.interfaces.base`): fit directly on *raw* numeric
data and this discovers a `DataSpec` from the thresholds actually used,
rather than requiring them fixed in advance by a separate discretizer.
Pass `feature_names=` to `import_model` too when using a discovered
`DataSpec` this way -- see the raw-input example under *Learning
algorithms*, above.

`RandomForestImporter` extracts every tree's leaves from a
`RandomForestClassifier` and combines them all into one flat `RuleSet`
-- not a sequence of one `DisjointRuleSet` per tree. Each tree is
individually exhaustive+disjoint, so a k-tree forest always has exactly
k covering rules per example; `pyrulearn.combiners` is what turns that
into forest-style prediction (see *Conflict resolution*, above) -- either
`CountVoteCombiner` for hard voting (or `HeuristicVoteCombiner` for a
heuristic-weighted one), or `MacroVoteCombiner` for soft voting straight
from each leaf's own measured per-class distribution (requires
`import_model(..., data=...)` so every leaf has stats --
`DistributionCombiner` raises otherwise), which is the one that
actually matches sklearn's own averaging mechanism. No `default_rule`
is needed: since
every tree always fires, the combined set is always exhaustive too.
(Note: sklearn
encodes labels once at the *forest* level and fits each tree on those
encoded values, so `rf.estimators_[i].classes_` is just `[0., 1., ...]`,
not the original labels -- `RandomForestImporter` passes the forest's
own `classes_` through explicitly to get real targets back.)

```python
from pyrulearn.interfaces.sklearn import RandomForestImporter, from_random_forest
from pyrulearn import BooleanDataRepresentation, MacroVoteCombiner

rules = RandomForestImporter().import_model(fitted_forest, my_dataspec)
my_rep = BooleanDataRepresentation(my_dataspec, X, y)
preds = rules.predict(my_rep, combiner=MacroVoteCombiner())  # soft voting, ~matches sklearn
```

This module also has the reverse direction: `RuleSetClassifier` wraps any
`pyrulearn.models` rule model as an `sklearn.base.BaseEstimator`/
`ClassifierMixin`, so it isn't an importer at all and doesn't fit the table
above. Two modes -- a fixed `rules=` (annotate given data, e.g. to evaluate
externally-mined rules via `cross_val_score`), or a `learner=` callback
mining rules fresh on every `fit` call, which plugs any rule-learning
algorithm into `cross_val_score`/`GridSearchCV`/sklearn pipelines.

### wittgenstein: IREP and RIPPER

`IREPImporter`/`RIPPERImporter` (`pyrulearn.interfaces.wittgenstein`,
an optional dependency -- only importing that module pulls in
`wittgenstein`) extract a `RuleSet` from a fitted `wittgenstein.IREP`/
`wittgenstein.RIPPER` model. Both algorithms share the same `Ruleset`/
`Rule`/`Cond` structure, hence one shared conversion helper and one
file for both, same grouping-by-shared-implementation rule as
`pyrulearn.interfaces.sklearn`. Every extracted rule predicts the model's single
`pos_class` -- IREP/RIPPER are fundamentally binary, confirmed directly
against the library rather than assumed: no internal one-vs-rest for
more classes; `fit` raises without an explicit `pos_class` for anything
beyond two/boolean-like labels, but given an explicit one it silently
proceeds on *any* number of classes instead, collapsing everything that
isn't `pos_class` into one undifferentiated negative with no warning --
`IREP`/`RIPPERk` (below) refuse this outright rather than let it happen
unnoticed. So rules never actually
conflict when several cover the same example (any `RuleCombiner`
agrees on the answer). What *does* need resolving is only the
*fallback* for what's uncovered, and the fitted model alone doesn't
retain what label that should be (its own `.predict()` only returns
booleans, "matched `pos_class`" or not) -- used directly,
`default_prediction` is therefore left `None`:

```python
from pyrulearn.interfaces.wittgenstein import RIPPERImporter
import wittgenstein as lw

model = lw.RIPPER(k=2, random_state=0)
model.fit(my_rep.X, y=my_rep.y, pos_class="pos", feature_names=my_rep.spec.feature_names)
rules = RIPPERImporter().import_model(model, my_dataspec)  # rules.default_rule is None
```

Use `IREP`/`RIPPERk` (below) instead of the bare importer to get that
fallback wired up automatically from training labels.

Fit on **raw** numeric/categorical columns and there are two ways to
still get a correct conversion, the same two as for
`pyrulearn.interfaces.sklearn`:

- **Binarize first**, via `pyrulearn.data.io.
  build_dataspec`/`binarize` (the same pipeline the
  `RandomForestImporter` demo above uses), so wittgenstein only ever
  sees already-Boolean columns named after real `DataSpec` features
  (`age>=20`, `color=red`, ...); every `Cond` it then produces is
  already a plain Boolean literal on a feature already in the target
  `DataSpec`, complete with `ThresholdChain`/`ExactlyOne` provenance:

  ```python
  from pyrulearn.data.io import build_dataspec, binarize
  from pyrulearn.data import BooleanDataRepresentation

  ds = build_dataspec(df, target="label", arff_types={...}).build()
  X = binarize(ds, df)
  rep = BooleanDataRepresentation(ds, X, df["label"].to_numpy())
  rules = RIPPERk(pos_class="pos").fit(rep)
  ```

- **Fit on raw columns** directly and let
  `IREPImporter`/`RIPPERImporter.infer_dataspec` parse wittgenstein's
  *own* discretization/nominal output back into typed attributes: a
  bin-range `Cond` like `num_feat=0.53 - 0.84` becomes two `<=`-family
  threshold literals (`num_feat<=0.53` false, `num_feat<=0.84` true),
  an unbounded one like `num_feat=>1.32` becomes one, and a bare
  category like `color=red` becomes a nominal-equality literal --
  decided per attribute from the `Cond.val`s actually seen for it, not
  globally. Both bounds go through the `<=` family, not `>=`:
  wittgenstein's own bins are right-closed/left-open, `(lo, hi]` --
  confirmed directly against `wittgenstein.discretize.BinTransformer`
  (its own docstring, its `closed='right'` interval construction, and
  empirically, a value exactly at a bin's lower edge lands in the
  *previous* bin) rather than assumed:

  ```python
  from pyrulearn.interfaces.wittgenstein import RIPPERk, RIPPERImporter

  learner = RIPPERk(pos_class="pos", k=2, random_state=0)
  model = learner.fit_external(df.to_numpy(), y, feature_names=["age", "color"])

  importer = RIPPERImporter()
  ds = importer.infer_dataspec(model, feature_names=["age", "color"])
  rules = importer.import_model(model, ds, feature_names=["age", "color"])
  ```

### imodels: Bayesian rule lists and rule sets

`BayesianRuleListImporter` (`pyrulearn.interfaces.imodels`, an
optional dependency on `imodels`) extracts Letham et al.'s Bayesian
Rule Lists from a fitted `imodels.BayesianRuleListClassifier` — and is
the one importer so far producing a **`RuleList`**, not a `RuleSet`. A
Bayesian Rule List is genuinely ordered (its own `__str__` prints
`IF ... ELSE IF ... ELSE ...`), each rule implicitly conditioned on
every earlier one *not* firing, so first-match-wins is the real
semantics and the trailing `ELSE` becomes the list's `default_prediction`.

No discretization wrinkle here, unlike wittgenstein: `imodels`' own
`fit` *requires* strictly 0/1 input (it raises `"All numeric features
must be discretized prior to fitting!"` otherwise) — exactly what a
`BooleanDataRepresentation` already holds — so `infer_dataspec` is
inherited unchanged and both approaches collapse to the same thing.
Confirmed directly (not just from the error message): a categorical
column has to go through `build_dataspec`/`binarize`'s **one-hot**
nominal encoding first, same as any numeric attribute — a raw string
column fails even earlier (`could not convert string to float`), and
an *ordinal*-encoded one (`0`/`1`/`2`, no one-hot) hits the exact same
"must be discretized" error, since a 3-valued int column still isn't
0/1. Its `fit` also refuses >2 classes properly (`"Only binary
classification is supported at this time!"`, with no built-in
one-vs-rest fallback), so no extra guard is needed on pyrulearn's side
— genuine multi-class support would mean pyrulearn adding its own
cascade on top, same open question wittgenstein raises.

One thing to know when reading imported rules: each rule carries a
posterior *probability* rather than a label, so a rule with
`theta=0.11` is a confident **negative** rule — `target` is read off as
whichever class `theta` favors (`> 0.5` -> positive class, else
negative). The raw `theta`/confidence-interval numbers aren't kept on
the rule; `pyrulearn.combiners`' `HeuristicCombiner`s score each rule
from its own measured `stats()` instead (pass `data=` to
`import_model`/`fit` to get them). Antecedents come from FP-growth
itemsets, so rules only ever contain **positive** literals.

```python
from pyrulearn.interfaces.imodels import BayesianRuleList

rules = BayesianRuleList(random_state=0).fit(train_rep)   # a RuleList
print(rules.to_string("logic"))
# if   f0 ∧ f1 → neg
# elif f0 ∧ f2 → pos
# else → neg
```

`BayesianRuleSetImporter` (same module) extracts the "BOA"/Bayesian
Or-of-And algorithm from a fitted `imodels.BayesianRuleSetClassifier`
into a plain **`RuleSet`**, not a `RuleList`: it fits a genuinely
*unordered* pattern set via simulated annealing and predicts OR-of-AND
(any one rule firing predicts `classes_[1]`, regardless of which), so
rules never actually disagree in target when several cover the same
example. Same Boolean-input requirement as BRL, but `imodels` doesn't
raise a clear error for either that or a non-binary target here (an
opaque downstream `AssertionError`/`ValueError` instead), so
`BayesianRuleSet.fit_external` checks both itself first. Rule items are
real feature names, optionally `_neg`-suffixed for a negative literal
(`model.rules_`, unlike BRL, can and does contain both), looked up via
`dataspec.feature_index` directly (no `X_0`-placeholder indirection).
There's no per-rule posterior the way BRL's `theta` is (the annealing
search optimizes the whole rule set jointly), so imported rules carry
no per-rule confidence score — moot anyway here, since firing rules
never disagree. **Two genuine bugs found in `imodels` itself** while testing
this importer (both documented in the module's docstring, not fixable
from this side): its own `fit` mixes seeded `np.random` with
*unseeded* stdlib `random.sample` (worked around in
`BayesianRuleSet.fit_external` by seeding that too), and its
simulated-annealing "clean" move has a genuine off-by-value-vs-index
bug that can raise a bare `list.remove(x): x not in list`.

```python
from pyrulearn.interfaces.imodels import BayesianRuleSet

rules = BayesianRuleSet(random_state=0, maxlen=3).fit(train_rep)   # a RuleSet
```

### weka: JRip, PART and J48

`pyrulearn.interfaces.weka.JRipImporter` is a real, non-reference
example of the shape: it parses `weka.classifiers.rules.JRip`'s printed
rule-set text (Weka is not a Python library, so there's no live model
object to hand an `ObjectRuleImporter` -- this is genuinely the
`StringRuleImporter` case, not just a stand-in for one). Verified
against a real `weka.jar` run, not assumed:

```python
from pyrulearn.interfaces.weka import JRipImporter

text = """
(color = red) and (age >= 41) => label=pos (81.0/2.0)
(income <= 19992) => label=pos (32.0/1.0)
 => label=neg (287.0/15.0)
"""
rules = JRipImporter().parse(text)   # a RuleList, not a RuleSet
```

Produces a **`RuleList`**, unlike wittgenstein's IREP/RIPPER importers
(`RuleSet`): JRip is genuinely multi-class (classic RIPPER sequential
covering -- one class's rules at a time, in printed order, falling back
to a default/majority class), so two rules with different targets can
both cover the same new example, and only trying them in the printed
order reproduces JRip's own predictions correctly -- confirmed by
`.predict()` on the underlying data matching JRip's rule logic exactly
*and* Weka's own reported training accuracy. No `dataspec=` needed up
front: `parse` discovers and caches a typed `DataSpec` from the rule
text itself, with numeric attributes getting a proper `ThresholdChain`
on *both* sides: `>=`/`<` conditions go through `add_numeric`'s usual
`ge_thresholds=` family, `<=`/`>` conditions through its `le_thresholds=`
family (see `DataSpecBuilder.add_numeric`'s own docstring for why
that's a second, genuinely independent family rather than a
reinterpretation of the first) -- so both binarize straight from a raw
numeric column with no manual precomputation needed either way. Pass
`dataspec=` at construction to reuse an already-built one instead, e.g.
across a train/test pair of Weka runs.

`PARTImporter` (same module) parses `weka.classifiers.rules.PART`'s
decision-list text -- Frank & Witten's "partial decision trees" learner,
separate-and-conquer over a pruned partial C4.5 tree each round. Same
`RuleList` result and `dataspec=`/auto-discovery convention as
`JRipImporter`, but a genuinely different text format underneath (also
verified against a real `weka.jar` run):

```python
from pyrulearn.interfaces.weka import PARTImporter

text = """
income > 19992 AND
color = green: neg (125.0/8.0)

age > 39: pos (100.0/3.0)

: pos (13.0)
"""
rules = PARTImporter().parse(text)   # a RuleList
```

No `=>`, no parentheses, one condition per line ending in a trailing
`` AND`` (except a rule's last line, which ends in `` : target
(stats)`` instead), and rules are blank-line-delimited blocks rather
than one line each -- so `PARTImporter` parses at the block level
(`_collect_part_conditions`), unrelated to JRip's line regex, but hands
off to the *same* shared condition-parsing/`DataSpec`-building code once
it has `(conditions, target, covered, errors)` tuples, since that part
only cares about the six recognized operator strings, not which
classifier produced them. Two format quirks worth knowing: PART's
numeric splits are classic C4.5-style `<=`/`>` (`>=`/`<` never appear),
and its trailing stats can omit the error count entirely when it's zero
(`(13.0)`, not JRip's always-both-numbers `(13.0/0.0)`) -- both handled
already.

`J48Importer` (same module) parses `weka.classifiers.trees.J48` --
Weka's C4.5 reimplementation -- and produces a **`DisjointRuleSet`**,
not a `RuleList`: a decision tree's leaves already partition the
feature space disjointly and exhaustively by construction, the same
reasoning `pyrulearn.interfaces.sklearn.SklearnTreeImporter` uses
for its own trees, so there's no rule-order/default-rule question here
at all -- confirmed, not just assumed (`rules.is_disjoint(rep)` and
`rules.is_exhaustive(rep)` both hold on real data):

```python
from pyrulearn.interfaces.weka import J48Importer

text = """J48 pruned tree
------------------

income <= 19992: pos (40.0/1.0)
income > 19992
|   color = red
|   |   age <= 40: neg (45.0/3.0)
|   |   age > 40: pos (73.0/2.0)
|   color = green: neg (125.0/8.0)
|   color = blue: neg (117.0/4.0)
"""
rules = J48Importer().parse(text)   # a DisjointRuleSet
```

The text is an indented tree, not a flat list: each `` |   `` group is
one nesting level, a line ending in `` : target (stats)`` is a leaf
(its rule is the conjunction of every still-open ancestor condition
plus its own), and a line without that suffix is an internal split
whose condition applies to everything nested under it -- walked with a
depth-indexed stack (`_collect_j48_conditions`), pushing on a deeper
split and truncating back to the current depth before every line (the
backtrack, whenever a sibling appears at the same or a shallower
depth). Because C4.5 (unlike CART) allows a nominal split to fan out
into more than two branches, `color` above genuinely has three
siblings under one parent -- handled without any special-casing, since
each sibling is just another line at that depth; verified against a
deliberately deep, 33-leaf/depth-5 tree too, not only the shallow
example above. One thing this importer needs that JRip/PART's don't:
scanning a *whole* raw console dump line-by-line for tree-shaped
content isn't safe here, since the confusion matrix's own column
layout can produce a line reading as both a `` |   ``-indented node
*and* a valid `` attr = value`` condition (e.g. `` 110  15 |   a =
pos``) -- `parse` instead anchors explicitly on the `` J48 ... tree``
header and its `` ---- `` divider before reading the tree body.

### LORD: the reference implementation

`pyrulearn.interfaces.lord` runs the Java implementation of LORD (Huynh,
Fürnkranz & Beck, 2023; [vqphuynh/LORD](https://github.com/vqphuynh/LORD)) as
a subprocess. `LordJar` is the `fit()` learner (`variant=` selects
`run.LordRun`, `LordStarRun` or `LordLoopRun`; `metric=`/`metric_arg=` are
LORD's `-mt`/`-ma`), `LORDImporter` parses the `eg_output.txt` it writes, and
`run_lord` is the function that drives the jar. `classpath=` (else
`$LORD_CLASSPATH`, or `$LORD_JAR` for a fat jar) and `java=` (else
`$LORD_JAVA`) locate the build.

The result is a `FlatRuleSet`: LORD predicts with the single covering rule
with the highest metric value, so pass the same metric to
`HeuristicMaxCombiner` if you want an identical ranking (its default is
`Laplace`). LORD does not print its default class, so `fit` sets the
training-majority class. A condition `(f3=0)` imports as a positive literal on
the paired negation feature of `f3`, so the bound `DataSpec` must have
negation features (`DataSpecBuilder`'s default). LORD is natively multi-class.

### pyarc: CBA

`pyrulearn.interfaces.pyarc.PyarcCBA` fits the external `pyarc` package's CBA
and returns a `DecisionList`; `PyarcCBAImporter` reads an already-fitted
`pyarc.CBA`. It is the reference that the native
`pyrulearn.learners.associative.CBA` is cross-checked against, and a faster
drop-in (the C library `fim` that `pyarc` mines with is roughly 10x faster than
this package's pure-Python miner). Its defaults (`min_support=0.01`,
`min_confidence=0.5`, `max_len=4`) mirror the native `CBA`'s, and `max_len`
means the same in both (`pyarc`'s own `maxlen` also counts the class item,
which `PyarcCBA` adjusts for); `algorithm=` picks `pyarc`'s `"m1"` (default)
or `"m2"` classifier builder. `pyarc` needs Borgelt's `pyfim` C extension,
which has no Windows wheels and must be built separately; importing the module
needs neither.

### Provenance of imported rules

Either shape's `import_model`/`parse` should end by calling
`self._stamp_rule_provenance(rules, **params)` and
`self._stamp_provenance(model, **params)`, which give every produced
rule (and the model itself) their own `Provenance(source=..., learner=...,
params={...})` -- how you tell which base learner produced which rule
once you're comparing many imported models from many sources at once.

## Try it

```
python examples/demo.py
python -m pytest                       # the whole test suite
python -m pytest tests/test_models.py  # or a single file
```

`examples/demo.py` exercises the whole surface: manual rules, dataset
coverage annotation, decision-tree rule extraction, `cross_val_score`,
a pattern-string import round-trip, and two
coverage-space plots -- a rule set as a `DecisionList`'s arrow-connected
cumulative path (`coverage_space_rulelist.png`), and one rule's own
precision-ordered refinement path (`rule_refinement.png`).

Other demos: `examples/demo_heuristics.py` (heuristic isometrics),
`examples/demo_workflow_comparison.py` and
`examples/demo_seco_learners_comparison.py` (cross-benchmark model
comparisons, each writing a `*_report.md`), and
`examples/demo_representations.py` (checks `NListRepresentation`,
`PrePostNListRepresentation` and `SparseDataRepresentation` all give
every SeCo learner byte-identical rules to `BooleanDataRepresentation`,
in both feature encodings, times all four, and finishes with two
synthetic sweeps -- one growing the feature count with density falling
as a side effect, one fixing the feature count and sweeping density
directly -- each plotting fit/`coverage()` time per representation
against it).

## Not yet implemented (left as clear extension points)

- **More rule learners and importers.** Beyond what's already interfaced
  (decision trees, random forests, wittgenstein's IREP/RIPPER, imodels'
  Bayesian Rule Lists/Rule Sets, Weka's JRip/PART/J48, LORD, and pyarc's
  CBA): more `StringRuleImporter` text formats (e.g. FOIL output); more
  `ObjectRuleImporter`s for other sklearn tree ensembles (gradient
  boosting, AdaBoost -- lower priority) and `imodels`'
  `BoostedRulesClassifier`; and `pyIDS` (github.com/jirifilip/pyIDS) as an
  external cross-check for the native `IDS` -- its selected rules read back through
  the same `pyarc` CAR shape `PyarcCBAImporter` already parses, but the
  package is unmaintained since 2021 and its declared `sklearn` dependency
  is a now-broken PyPI package name (installs with `--no-deps`; a full
  compatibility check on a modern Python is still open).
  `imodels.RuleFitClassifier` and `SlipperClassifier` are addressed below
  instead, not here -- both are better served by native machinery than by
  a direct import.

- **Closing gaps between native algorithms and their reference papers.**
  `AQR`'s literal *star* search and `PyLORD`'s exhaustive branch-and-bound
  are both currently approximated by a seed-restricted `BeamSearch` (see
  `pyrulearn.learners.seco`/`pyrulearn.learners.pylord`); `Pypper`'s
  residual IREP\* re-growth pass is the one piece still missing from its
  optimization phase (`ReplaceReviseOptimization` itself is already done).

- **Weighted and additive rule models**, at two different points in the
  pipeline. `SeCo`'s covering step is currently hard-coded inline
  (`SeCo._covering_loop`), with a boolean `example_mask` as the only
  notion of "scope" threaded through `learn_one_rule`, `RuleStats.from_rule`,
  pruning/filtering/stopping, and the incremental cover handles. Plan: a
  `CoveringStrategy` alongside `SeCo`'s other pluggable strategies, with
  `RemovalCovering` (today's behavior) and `WeightedCovering` (covered
  examples down-weighted instead of dropped) -- the real cost is
  downstream, since `RuleStats` would need weighted `tp`/`fp`/`fn`/`tn`
  (heuristics work unchanged) and the `NListRepresentation` popcount fast
  path doesn't extend to weights for free. Three things motivate it:
  - **CPAR** (Yin & Han, 2003) and `imodels`' **`SlipperClassifier`**
    (Cohen & Singer, 1999) both need it *during induction* -- CPAR via a
    fixed per-round decay, Slipper via boosting-style reweighting -- and
    CPAR additionally needs a new `TopKMeanCombiner` (average Laplace
    accuracy of the top-k rules per class) for prediction.
  - **Additive boosting of rules** (the ENDER family, BOOMER) needs the
    same example-reweighting machinery, but driven by gradient/residual
    updates after each added rule, with additive rather than list/set
    prediction.
  - A planned **RuleFit-style distiller** needs weights at the *opposite*
    end instead: given any rule pool (mined or externally supplied, the
    same `RuleDistiller` pattern as `CBA`/`CMAR`/`IDS`), fit a LASSO over
    the rules' coverage-indicator features (`RuleModel.coverage_matrix`
    already gives that matrix) and keep the rules with a nonzero weight.
    Preferred over importing `imodels.RuleFitClassifier` directly, since
    its own rule-generation step is an unrelated `GradientBoostingRegressor`
    detour, not the LASSO fit that's actually its contribution. Needs a
    new `RuleModel` type (`intercept + sum(weight * indicator)`
    prediction, not covering-rule combination) and `WeightedRule.weight`'s
    docstring loosened to admit signed coefficients -- no code change
    there, since nothing reads `.weight` at prediction time yet.

- **Representation and analysis extensions.** Rule bodies beyond pure
  conjunctions (e.g. general CNF/DNF rules); native readers for sparse
  ARFF (`{index value, ...}`) and libsvm files (`SparseDataRepresentation`
  and `from_scipy` exist, but `pyrulearn.data.io` and the demos' OpenML
  loader still go through a dense DataFrame, so a genuinely sparse dataset
  like OpenML's `connect-4` densifies on the way in); and a logic/SAT
  interface (export to `sympy`/DIMACS, SAT-based analysis) -- an earlier
  version was removed from this release and will return in a reworked
  form.

- **Project housekeeping.** Publishing to PyPI (installation is currently
  from GitHub); revising the demos in `examples/`, some of which were
  written against earlier versions of the API.

## Authors and contributions

**Johannes Fürnkranz** ([Johannes Kepler University Linz](https://www.jku.at),
`juffi@faw.jku.at`) is the author of pyrulearn and responsible for the project.

The programming was done by **Claude** (Anthropic's AI model, used through
[Claude Code](https://claude.com/claude-code)) in continuous dialogue with the
author, who directed the design and reviewed the results. Contributions, following
the [CRediT](https://credit.niso.org) contributor roles:

| Role | Contributor |
|---|---|
| Conceptualization | Johannes Fürnkranz |
| Software design (architecture, APIs, choice and adaptation of algorithms) | Johannes Fürnkranz, Claude |
| Software (implementation, tests, documentation) | Claude |
| Validation (review; cross-checks against reference implementations) | Johannes Fürnkranz, Claude |
| Supervision and project administration | Johannes Fürnkranz |

Since an AI system cannot take responsibility for a work, Claude is not
listed as an author in the citation metadata, in line with the guidance of
publishers and bodies such as COPE and ICMJE on AI tools; its contribution is
documented here and through `Co-Authored-By: Claude` trailers on commits.

## How to cite

If you use pyrulearn, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button), or as

```bibtex
@software{furnkranz2026pyrulearn,
  author  = {F{\"u}rnkranz, Johannes},
  title   = {pyrulearn},
  year    = {2026},
  version = {0.1.2},
  url     = {https://github.com/juffif/pyrulearn}
}
```

Please also cite the original publications of the algorithms you use
(see the tables above and [`references.bib`](references.bib)).

## License

pyrulearn is free software, released under the
[GNU General Public License, version 3 or (at your option) any later version](LICENSE).
Copyright © 2026 Johannes Fürnkranz.

## Acknowledgements

pyrulearn re-implements algorithms published by many others (see
[`references.bib`](references.bib)) and interfaces to scikit-learn,
`wittgenstein`, `imodels`, `pyarc` (and Christian Borgelt's `pyfim`), Weka and
LORD. These are used through their public interfaces under their own
licenses; none of them is bundled with pyrulearn. Datasets used in the demos
come from public repositories such as OpenML and the UCI repository and are not
redistributed here.
