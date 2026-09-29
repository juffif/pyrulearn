# Demo revision: plan and notes

Notes for the demo overhaul on this branch.

## Overall plan (agreed 2026-09-27, to be done later)

1. **Shared experiment infrastructure first**, replacing the loading,
   splitting and measuring code each demo now copies: datasets through the
   catalog (`Catalog.default().select(...)`); stratified CV with
   binarization per fold; a time limit per fit and failures recorded
   instead of aborting; measures for every model type (accuracy, number of
   rules and conditions -- also for linear models and ensembles -- fit
   time); result tables with `display_name` columns and mean ranks;
   caching of finished folds; the report skeleton with its Setup section
   from `DESCRIPTION`; a quick mode (few datasets) and a full mode.
2. **The comparisons**, each with its own report:
   - (a) native vs. external: each native algorithm against its external
     counterpart (Pypper vs. WekaJRip/WittRIPPER, Slipper vs. IModSlipper,
     RuleFit vs. IModRuleFit, CBA vs. PArcCBA, Boomer vs. MLRLBoomer,
     OptimalRuleBoosting vs. RKDRuleBoosting, PyLORD vs. JavaLord) --
     replaces the ripper, seco-learners and workflow demos;
   - (b) rule pools x distillers (below) -- replaces the RuleFit and
     random-forest-combiner demos;
   - (c) covering strategies and reweighting schemes (below);
   - (d) rule ensembles: Slipper, LRI, CPAR, the ENDER variants, Boomer,
     optimal rule boosting, as accuracy against number of rules;
   - (e) numeric attributes: intrinsic handling vs. discretization (below);
   - (f) multiclass decompositions (refresh of the pairwise demo).
3. **Tour demos** (overview, representations, heuristics/isometrics,
   decision-tree import) stay as short illustrations, updated to the new
   API and names.

Suggested order: infrastructure, (a), (d), (b), (c), (e), (f), tour demos.

Decided (2026-09-28): infrastructure landed as `pyrulearn.experiments`
(`runner`/`stats`/`report`/`catalog`, the last moved from
`pyrulearn.data.catalog`). `demo_ripper_comparison.py` stays its own
separate demo rather than folding into (a) -- migrated onto the new
infra, `jrip_native` (JRip on raw data) dropped, and the two Slipper
variants (`Slipper`, `IModSlipper`) added to it. **Next up after that:
(e), also kept as its own demo** -- native numeric-attribute handling
vs. discretization.

Open questions: infrastructure in `examples/` or as a library module
(e.g. `pyrulearn.experiments`; leaning towards the library); statistics
(mean ranks only, or Friedman tests with critical-difference diagrams);
the default catalog selection for full runs and a fixed small set for
quick mode; the order.

## Rule pools and distillers: one systematic comparison

Merge the RuleFit comparison (`demo_rulefit_comparison.py`, on branch
`weight-framework` so far) and possibly the random-forest demo
(`demo_random_forest_combiners.py`) into one demo that compares
**rule-pool generators** and **rule distillers** systematically: every
generator crossed with every distiller, on the same data and protocol.

- **Pool generators:** mined class association rules (`CARMiner`),
  random-forest leaves (shallow and fully grown, `from_random_forest`),
  RuleFit's boosted trees (`rulefit_candidates`), perhaps rules from
  other learners.
- **Distillers:** `RuleFit`, `CBA`, `CMAR`, `IDS`, and the pool used
  directly with a combiner (as in the random-forest demo).
- **Benchmarks:** the random forest itself, one or two direct rule
  learners (e.g. Pypper), and imodels' `RuleFitClassifier`.
- **Measures:** accuracy, model size and rule length (compared against
  each other, not only accuracy), and fit time, split into pool
  generation and distillation.

Things learned from the RuleFit demo:

- Compare accuracy at comparable model sizes. imodels caps RuleFit at 30
  terms and the native `RuleFit` has no such cap, so accuracy alone
  favors the larger models.
- Pools need comparable sizes. A mined pool runs to tens of thousands of
  rules unless capped: the RuleFit demo keeps the 500 most confident,
  and mines only up to length 2 above 200 features, where the frequent
  negation features make length-3 mining explode.
- A shallow forest (depth <= 3) was as good a pool as the fully grown
  one, with far fewer and shorter rules.

## Covering strategies: removal vs. weighted reweighting schemes

A demo comparing how the covering loop treats covered examples: removal
covering (classic separate-and-conquer) against the weighted covering
variants, each a reweighting scheme plus its stopping criterion.

- **Schemes:** removal; multiplicative (gamma, CN2-SD) and additive
  (1/(k+1), CN2-SD); CPAR's decay; AdaBoost-style reweighting (Slipper);
  Lightweight Rule Induction's cumulative-error weights (1 + e^3); later
  gradient-based weights (ENDER).
- **Across heuristics:** at least Laplace and WRAcc, since the
  heuristic and the reweighting interact (CN2-SD is WRAcc + weighted
  covering; Laplace + weighted covering gives many overlapping rules).
- **Measures:** accuracy, number and length of rules, overlap (how often
  a positive is covered), fit time; for rule sets without weights also
  the effect of the combiner (max vs. vote).
- The branches with the machinery: `weighted-covering` (row weights,
  `CoveringStrategy`), and whatever follows it (Slipper, LRI).

## Numeric attributes: intrinsic handling vs. our discretization

Compare learners that handle numeric attributes themselves (thresholds
chosen during learning: Weka's JRip and J48, also on the raw data) with
the same or similar learners on data discretized by `build_dataspec`
beforehand, at several discretization levels (`max_intervals`, e.g. 2,
4, 6, 10, 20).

- **Learners:** JRip and J48 on raw data vs. on our binarized data;
  the native learners (Pypper, CN2, Slipper, LRI, ...) on the binarized
  data at each level.
- **Discretizers:** the current `build_dataspec` binning at each level
  (supervised, decision-tree-based, `tree_thresholds`); `sklearn.
  preprocessing.KBinsDiscretizer` (unsupervised -- `strategy=`
  `"uniform"`/`"quantile"`/`"kmeans"`, no target needed -- its
  `bin_edges_` feed straight into `DataSpecBuilder.add_numeric
  (thresholds=...)`, which takes cut points regardless of how they were
  derived); later FUSINTER (Zighed, Rabaséda & Rakotomalala 1998), a
  supervised discretization, once implemented.
- **Datasets:** ones with many numeric attributes (the catalog's
  numeric-heavy selection: diabetes, sonar, ionosphere,
  banknote-authentication, ...).
- **Measures:** accuracy, number and length of rules, fit time, and the
  number of binary features each discretization produces.
- `demo_ripper_comparison.py` already runs JRip on both the raw data
  (`jrip_native`) and our binarization -- a starting point.

## For every demo

- The generated report describes its own experiment (data, protocol,
  variants and their settings, measures) in a Setup section, taken from
  one `DESCRIPTION` string in the script, rather than referring to the
  script's docstring.
