# Rule learning heuristics: isometrics in coverage space

A rule learning heuristic scores a rule by the positives `p` and negatives `n` it covers, given the totals `P` and `N`. Its *isometrics* -- the lines of equal score in coverage space (x = negatives covered, y = positives covered) -- show its preference structure at a glance: how it trades off consistency against coverage (Fürnkranz & Flach, 2005). This demo plots them for every heuristic in `pyrulearn.heuristics`, each in raw coverage space (counts) and normalized ROC space (rates), for an abstract dataset with P = 100 and N = 160. The dashed gray diagonal is the random-guess line; brighter isometrics are better.

All plots come straight from the library, no plotting code of their own: `pyrulearn.evaluation.CoverageSpace` draws the space, and every heuristic can draw its own isometrics into it.

```python
from pyrulearn.evaluation import CoverageSpace
from pyrulearn.heuristics import MEstimate

space = CoverageSpace(normalized=False)          # or CoverageSpace.from_data(rep, target)
MEstimate(20).plot_isometrics(space=space, levels=8)
space.plot_rule_refinement(rule, rep)            # a rule's refinement path on top
```

How the heuristics actually perform as search heuristics -- accuracy and theory size inside one separate-and-conquer learner, after Janssen & Fürnkranz (Machine Learning, 2010) -- is tested in the heuristic comparison demo: see the [full results](heuristic_comparison_report.md).

Run: `python demos/heuristic_isometrics.py` (no data or extras needed beyond matplotlib).

## Isometrics per heuristic

Left: coverage space (counts), right: ROC space (rates). `p`/`n`: covered positives/negatives, `P`/`N`: all positives/negatives. `LengthPenalized` is left out: it needs a rule length, which a point in coverage space doesn't have.

### CoveredNegatives

h = -n. Only consistency counts: vertical isometrics.

<img src="heuristic_isometrics_plots/isometrics_coverednegatives_raw.png" alt="CoveredNegatives, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_coverednegatives_roc.png" alt="CoveredNegatives, ROC space" width="49%">

### Recall

h = p/P. Only positive coverage counts: horizontal isometrics.

<img src="heuristic_isometrics_plots/isometrics_recall_raw.png" alt="Recall, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_recall_roc.png" alt="Recall, ROC space" width="49%">

### Precision

h = p/(p+n). Lines through the origin: purity, regardless of coverage.

<img src="heuristic_isometrics_plots/isometrics_precision_raw.png" alt="Precision, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_precision_roc.png" alt="Precision, ROC space" width="49%">

### FBeta(1)

Harmonic mean of precision and recall. Precision-like lines, but pivoting at (-beta^2 * P, 0) left of the origin, so coverage gains weight.

<img src="heuristic_isometrics_plots/isometrics_fbeta_1_raw.png" alt="FBeta(1), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_fbeta_1_roc.png" alt="FBeta(1), ROC space" width="49%">

### Laplace

h = (p+1)/(p+n+2). Precision with its pivot moved to (-1, -1).

<img src="heuristic_isometrics_plots/isometrics_laplace_raw.png" alt="Laplace, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_laplace_roc.png" alt="Laplace, ROC space" width="49%">

### MEstimate(20)

h = (p + m * P/(P+N))/(p+n+m). The pivot moves out along the negative diagonal as m grows; from precision (m = 0) towards WRAcc (m -> infinity).

<img src="heuristic_isometrics_plots/isometrics_mestimate_20_raw.png" alt="MEstimate(20), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_mestimate_20_roc.png" alt="MEstimate(20), ROC space" width="49%">

### GeneralizedMEstimate(20, 0.9)

h = (p + m * c)/(p+n+m): the m-estimate with a free prior c instead of P/(P+N).

<img src="heuristic_isometrics_plots/isometrics_generalizedmestimate_20_0_9_raw.png" alt="GeneralizedMEstimate(20, 0.9), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_generalizedmestimate_20_0_9_roc.png" alt="GeneralizedMEstimate(20, 0.9), ROC space" width="49%">

### GHeuristic(30)

h = p/(n+g) (Gamberger & Lavrac). Precision-like lines pivoting at (-g, 0).

<img src="heuristic_isometrics_plots/isometrics_gheuristic_30_raw.png" alt="GHeuristic(30), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_gheuristic_30_roc.png" alt="GHeuristic(30), ROC space" width="49%">

### WRAcc

Weighted relative accuracy, coverage times precision gain over P/(P+N). Lines parallel to the diagonal.

<img src="heuristic_isometrics_plots/isometrics_wracc_raw.png" alt="WRAcc, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_wracc_roc.png" alt="WRAcc, ROC space" width="49%">

### YoudenJ

h = p/P - n/N. Lines parallel to the diagonal; ranks rules exactly like WRAcc.

<img src="heuristic_isometrics_plots/isometrics_youdenj_raw.png" alt="YoudenJ, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_youdenj_roc.png" alt="YoudenJ, ROC space" width="49%">

### Accuracy

h = (p + N - n)/(P+N). Parallel lines of slope 1.

<img src="heuristic_isometrics_plots/isometrics_accuracy_raw.png" alt="Accuracy, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_accuracy_roc.png" alt="Accuracy, ROC space" width="49%">

### CoverageDifference

h = p - n. Ranks rules exactly like Accuracy.

<img src="heuristic_isometrics_plots/isometrics_coveragedifference_raw.png" alt="CoverageDifference, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_coveragedifference_roc.png" alt="CoverageDifference, ROC space" width="49%">

### Support

h = (p+n)/(P+N). Anti-diagonal lines: coverage regardless of class.

<img src="heuristic_isometrics_plots/isometrics_support_raw.png" alt="Support, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_support_roc.png" alt="Support, ROC space" width="49%">

### Coverage

h = p + n. Ranks rules exactly like Support.

<img src="heuristic_isometrics_plots/isometrics_coverage_raw.png" alt="Coverage, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_coverage_roc.png" alt="Coverage, ROC space" width="49%">

### LinearCost(2.0)

h = p - c * n (the paper's cost measure, up to scaling). Parallel lines of slope c.

<img src="heuristic_isometrics_plots/isometrics_linearcost_2_0_raw.png" alt="LinearCost(2.0), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_linearcost_2_0_roc.png" alt="LinearCost(2.0), ROC space" width="49%">

### LinearCostRates(2.0)

h = p/P - c * n/N (the relative cost measure, up to scaling). Parallel lines of slope c in ROC space.

<img src="heuristic_isometrics_plots/isometrics_linearcostrates_2_0_raw.png" alt="LinearCostRates(2.0), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_linearcostrates_2_0_roc.png" alt="LinearCostRates(2.0), ROC space" width="49%">

### Correlation

The phi coefficient between 'rule covers' and 'is positive' (FOSSIL). Curved isometrics, symmetric around the diagonal.

<img src="heuristic_isometrics_plots/isometrics_correlation_raw.png" alt="Correlation, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_correlation_roc.png" alt="Correlation, ROC space" width="49%">

### Entropy

Negated class entropy among covered examples (CN2's original heuristic). Symmetric: pure negative rules score as well as pure positive ones.

<img src="heuristic_isometrics_plots/isometrics_entropy_raw.png" alt="Entropy, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_entropy_roc.png" alt="Entropy, ROC space" width="49%">

### LikelihoodRatio

CN2's significance statistic: how far the covered class distribution is from the prior. Grows with coverage, in both directions away from the diagonal.

<img src="heuristic_isometrics_plots/isometrics_likelihoodratio_raw.png" alt="LikelihoodRatio, coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_likelihoodratio_roc.png" alt="LikelihoodRatio, ROC space" width="49%">

### FoilGain (vs. universal rule)

FOIL's information gain, here always relative to the universal rule (a gain heuristic needs a parent rule): p * (log2 precision - log2 P/(P+N)).

<img src="heuristic_isometrics_plots/isometrics_foilgain_vs_universal_rule_raw.png" alt="FoilGain (vs. universal rule), coverage space" width="49%"> <img src="heuristic_isometrics_plots/isometrics_foilgain_vs_universal_rule_roc.png" alt="FoilGain (vs. universal rule), ROC space" width="49%">

## A rule's refinement path over the isometrics

One rule on a small synthetic dataset, its conditions added in order of precision (`Rule.order_by_precision`), drawn as a path through coverage space (`CoverageSpace.plot_rule_refinement`) over Precision's (blue) and Accuracy's (green) isometrics. The path always moves to better Precision isometrics -- that is what the order optimizes -- but not necessarily to better Accuracy ones. The black line is the Precision isometric through the final rule (`plot_isometric_through_rule`): every point on it is exactly as precise.

<img src="heuristic_isometrics_plots/refinement_raw.png" alt="Rule refinement path, coverage space" width="49%"> <img src="heuristic_isometrics_plots/refinement_roc.png" alt="Rule refinement path, ROC space" width="49%">

## Where the precision family pivots

Precision, Laplace and the m-estimate all have isometrics that are lines through one pivot point: the origin for Precision, (-1, -1) for Laplace, and further out along the negative diagonal for the m-estimate as `m` grows. In a space of realistic size a shift of one or two examples is invisible, so this plot uses a small space (P = 10, N = 16).

<img src="heuristic_isometrics_plots/precision_family_raw.png" alt="Precision family, coverage space" width="49%"> <img src="heuristic_isometrics_plots/precision_family_roc.png" alt="Precision family, ROC space" width="49%">

