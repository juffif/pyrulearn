# Demo revision: plan and notes

Notes for the demo overhaul on this branch.

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

## For every demo

- The generated report describes its own experiment (data, protocol,
  variants and their settings, measures) in a Setup section, taken from
  one `DESCRIPTION` string in the script, rather than referring to the
  script's docstring.
