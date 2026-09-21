# RIPPER comparison: jrip / jrip_native / wittgenstein / pypper

`jrip`, `wittgenstein` and `pypper` share one binarized feature set (`build_dataspec` + `binarize`, one DataSpec per dataset); `jrip_native` is the same Weka JRip run on the raw columns instead (its own discretization). 70/30 stratified split, seed 0, datasets capped at 1600 rows. wittgenstein on multi-class = manual one-vs-rest (one fit per class, rules pooled). `bin.feat` is the shared binary feature count -- not what `jrip_native` used.

| dataset | n | bin.feat | classes | algo | acc | rules | conds | conds/rule | fit s |
|---|--:|--:|--:|---|--:|--:|--:|--:|--:|
| vote | 435 | 96 | 2 | jrip | 0.931 | 2 | 4 | 2.0 | 0.4 |
| vote | 435 | 96 | 2 | jrip_native | 0.931 | 3 | 7 | 2.3 | 1.2 |
| vote | 435 | 96 | 2 | wittgenstein | 0.908 | 3 | 9 | 3.0 | 0.1 |
| vote | 435 | 96 | 2 | pypper | 0.947 | 1 | 1 | 1.0 | 0.2 |
| iris | 150 | 40 | 3 | jrip | 0.956 | 3 | 5 | 1.7 | 0.3 |
| iris | 150 | 40 | 3 | jrip_native | 1.000 | 2 | 4 | 2.0 | 1.2 |
| iris | 150 | 40 | 3 | wittgenstein | 0.956 | 5 | 7 | 1.4 | 0.1 |
| iris | 150 | 40 | 3 | pypper | 1.000 | 2 | 3 | 1.5 | 0.0 |

## Means (successful fits only)

| algo | datasets | mean acc | mean rules | mean conds | mean conds/rule | mean fit s |
|---|--:|--:|--:|--:|--:|--:|
| jrip | 2 | 0.943 | 2.5 | 4.5 | 1.83 | 0.38 |
| jrip_native | 2 | 0.966 | 2.5 | 5.5 | 2.17 | 1.24 |
| wittgenstein | 2 | 0.932 | 4.0 | 8.0 | 2.20 | 0.10 |
| pypper | 2 | 0.973 | 1.5 | 2.0 | 1.25 | 0.11 |

## Mean accuracy by target type

| algo | binary | multi-class |
|---|--:|--:|
| jrip | 0.931 | 0.956 |
| jrip_native | 0.931 | 1.000 |
| wittgenstein | 0.908 | 0.956 |
| pypper | 0.947 | 1.000 |
