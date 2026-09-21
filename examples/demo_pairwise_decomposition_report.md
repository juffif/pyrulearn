# Pairwise vs. one-vs-rest decomposition for Pypper

One inner learner (`RIPPER._stage_learner()`), shared binarized features, 70/30 stratified split, seed 0. 5 fitted models; the 3 pairwise ones are each scored under `MajorityVote` (`:vote`), `WeightedVote` (`:wv`), `AccuracyWeightedVote` (`:awv`) on the *same fit* -- their `train s` is 0 (shared). `pw_min`/`pw_maj`/`pw_both` = `Pairwise(positive=)` `smaller`/`larger`/`both`.

## Accuracy -- all 11 results

| dataset | n | classes | ovr | ovr_ord | pw_min:vote | pw_min:wv | pw_min:awv | pw_maj:vote | pw_maj:wv | pw_maj:awv | pw_both:vote | pw_both:wv | pw_both:awv |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| iris | 150 | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wine | 178 | 3 | 0.944 | 0.889 | 0.907 | 0.870 | 0.907 | 0.907 | 0.389 | 0.907 | 0.907 | 0.907 | 0.907 |

| result | datasets | mean acc | mean predict s |
|---|--:|--:|--:|
| ovr | 2 | 0.972 | 0.00 |
| ovr_ord | 2 | 0.944 | 0.00 |
| pw_min:vote | 2 | 0.954 | 0.00 |
| pw_min:wv | 2 | 0.935 | 0.00 |
| pw_min:awv | 2 | 0.954 | 0.00 |
| pw_maj:vote | 2 | 0.954 | 0.00 |
| pw_maj:wv | 2 | 0.694 | 0.00 |
| pw_maj:awv | 2 | 0.954 | 0.00 |
| pw_both:vote | 2 | 0.954 | 0.00 |
| pw_both:wv | 2 | 0.954 | 0.00 |
| pw_both:awv | 2 | 0.954 | 0.00 |

### predict time (s) -- all 11 results

| dataset | ovr | ovr_ord | pw_min:vote | pw_min:wv | pw_min:awv | pw_maj:vote | pw_maj:wv | pw_maj:awv | pw_both:vote | pw_both:wv | pw_both:awv |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| iris | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| wine | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 |

## Train vs. predict time (s) -- the 5 fitted models

| dataset | ovr train / pred | ovr_ord train / pred | pw_min train / pred | pw_maj train / pred | pw_both train / pred |
|---|--:|--:|--:|--:|--:|
| iris | 0.1 / 0.00 | 0.0 / 0.00 | 0.0 / 0.00 | 0.0 / 0.00 | 0.1 / 0.00 |
| wine | 0.5 / 0.00 | 0.2 / 0.00 | 0.3 / 0.00 | 0.2 / 0.00 | 0.5 / 0.00 |

| model | mean train s | mean predict s (vote) |
|---|--:|--:|
| ovr | 0.28 | 0.00 |
| ovr_ord | 0.11 | 0.00 |
| pw_min | 0.16 | 0.00 |
| pw_maj | 0.11 | 0.00 |
| pw_both | 0.28 | 0.00 |

*(train time for `:wv` / `:awv` is 0 -- same fit as `:vote`. `:wv` roughly doubles predict time -- it re-resolves every member's covering rules; `:awv` is as cheap as `:vote`. Per-result predict times are in the section above.)*

## Rule complexity -- the 5 fitted models

| dataset | ovr models/rules/conds | ovr_ord models/rules/conds | pw_min models/rules/conds | pw_maj models/rules/conds | pw_both models/rules/conds |
|---|--:|--:|--:|--:|--:|
| iris | 3/4/7 | 2/2/3 | 3/3/4 | 3/3/4 | 6/7/8 |
| wine | 3/7/12 | 2/2/3 | 3/4/5 | 3/4/5 | 6/8/10 |

| model | mean models | mean rules | mean conds |
|---|--:|--:|--:|
| ovr | 3.0 | 5.5 | 9.5 |
| ovr_ord | 2.0 | 2.0 | 3.0 |
| pw_min | 3.0 | 3.5 | 4.5 |
| pw_maj | 3.0 | 3.5 | 4.5 |
| pw_both | 6.0 | 7.5 | 9.0 |
