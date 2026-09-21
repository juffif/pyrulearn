"""
examples/demo_pairwise_decomposition.py
======================================

Multi-class decomposition for **one** rule learner -- Pypper
(`Pypper._stage_learner()`: native IREP* growth + replace/revise
optimization) -- across **5 fitted models** and, for the pairwise ones,
**3 voting schemes** = **11 accuracy results**.

Fitted models (5):

- **ovr**     -- `OneVsRest`: `n` "class c vs. the rest" models on all
                 rows, max-weight vote.
- **ovr_ord** -- `OrderedOneVsRest` (least-frequent-first): `n-1` peeling
                 models, a decision list. This is *native Pypper*.
- **pw_min**  -- `Pairwise(positive="smaller")`: `C(n,2)` models, each
                 targeting the pair's **minority** class.
- **pw_maj**  -- `Pairwise(positive="larger")`: `C(n,2)` models,
                 **majority** target.
- **pw_both** -- `Pairwise(positive="both")`: `2*C(n,2)` models (one per
                 *ordered* pair) -- a rule set for every class.

Voting -- `pw_min` / `pw_maj` / `pw_both` are each scored 3 ways on the
**same fitted models** (no refit; only the combiner is swapped):

- **:vote** -- `MajorityVote`: one hard vote per member.
- **:wv**   -- `WeightedVote`: the deciding rule's Laplace-on-measured-
               stats score splits the vote `p_ij` / `1 - p_ij` (resolved
               per example, computed fresh -- not a stored weight).
- **:awv**  -- `AccuracyWeightedVote`: the *member's* own accuracy on its
               two-class sub-problem splits the vote (one scalar per
               member, recorded at fit time).

`2 + 3*3 = 11` results. **Train and predict time are reported
separately** -- predict dominates for pairwise (many sub-models), and
`:wv` in particular re-resolves every member's covering rules.

- The **accuracy** view shows all 11.
- The **runtime** and **rule-complexity** views show only the 5 fitted
  models -- the `:wv` / `:awv` variants share both training and rules.

Shared binarized feature set (`build_dataspec` + `binarize` once per
dataset), 70/30 stratified split, seed 0. Uncapped by default; `--cap N`
subsamples larger sets. Reuses `demo_ripper_comparison`'s OpenML loader.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

import examples.demo_ripper_comparison as drc
from pyrulearn.models import AccuracyWeightedVote, MajorityVote, WeightedVote
from pyrulearn.learners.multiclass import OneVsRest, OrderedOneVsRest, Pairwise
from pyrulearn.learners.seco import Pypper

RANDOM_STATE = 0
MAX_ROWS = None
HERE = os.path.dirname(__file__)
REPORT_PATH = os.path.join(HERE, "demo_pairwise_decomposition_report.md")
PLOT_PREFIX = os.path.join(HERE, "demo_pairwise_decomposition")

DATASETS = [
    "iris", "wine", "glass", "vehicle", "segment", "car",
    "balance-scale", "zoo", "ecoli", "lymph", "yeast", "letter",
]


def pypper():
    """A fresh per-class-targetable Pypper stage learner."""
    return Pypper(random_state=RANDOM_STATE)._stage_learner()


PAIRWISE_FITS = [("pw_min", "smaller"), ("pw_maj", "larger"), ("pw_both", "both")]
VOTERS = [("vote", MajorityVote), ("wv", WeightedVote), ("awv", AccuracyWeightedVote)]

FIT_KEYS = ["ovr", "ovr_ord"] + [k for k, _ in PAIRWISE_FITS]           # the 5
RESULT_KEYS = ["ovr", "ovr_ord"] + [f"{k}:{v}" for k, _ in PAIRWISE_FITS
                                    for v, _ in VOTERS]                  # the 11

# minority = blues, majority = greens, both = purples; light->dark = vote/wv/awv
_COLORS = {
    "ovr": "#bdbdbd", "ovr_ord": "#636363",
    "pw_min:vote": "#c6dbef", "pw_min:wv": "#6baed6", "pw_min:awv": "#08519c",
    "pw_maj:vote": "#c7e9c0", "pw_maj:wv": "#74c476", "pw_maj:awv": "#006d2c",
    "pw_both:vote": "#dadaeb", "pw_both:wv": "#9e9ac8", "pw_both:awv": "#54278f",
}
_FIT_COLORS = {"ovr": "#bdbdbd", "ovr_ord": "#636363",
               "pw_min": "#6baed6", "pw_maj": "#74c476", "pw_both": "#9e9ac8"}


def _complexity(model, n_classes, kind):
    rules = model.rules
    if hasattr(model, "members"):
        n_models = len(model.members)
    else:
        n_models = n_classes if kind == "ovr" else n_classes - 1
    return dict(rules=len(rules), conds=int(sum(len(r.conditions) for r in rules)),
                models=n_models)


def run_dataset(name):
    df, target = drc.load(name, max_rows=MAX_ROWS)
    c = drc.prepare(df, target)._replace(name=name)
    n = len(np.unique(c.tr_y))
    print(f"\n{name}  (n={len(df)}, {c.ds.n_features} bin.feat, {n} classes)")
    row = {"name": name, "n": len(df), "n_classes": n, "res": {}, "cx": {}}
    res, cx = row["res"], row["cx"]

    def evaluate(model):
        t0 = time.time()
        acc = float(np.mean(np.asarray(model.predict(c.test_rep)) == c.te_y))
        return acc, time.time() - t0

    # -- ovr / ovr_ord -----------------------------------------------------
    for key, build in (
        ("ovr", lambda: OneVsRest(pypper())),
        ("ovr_ord", lambda: OrderedOneVsRest(pypper(), order="least_frequent",
                                             random_state=RANDOM_STATE)),
    ):
        try:
            t0 = time.time()
            model = build().fit(c.train_rep)
            train_s = time.time() - t0
            acc, pred_s = evaluate(model)
            res[key] = dict(acc=acc, train_s=train_s, predict_s=pred_s)
            cx[key] = _complexity(model, n, key)
            print(f"  {key:14s} acc={acc:.3f}  train={train_s:8.1f}s  pred={pred_s:7.2f}s  "
                  f"models={cx[key]['models']:3d}  rules={cx[key]['rules']:4d}")
        except Exception as e:  # noqa: BLE001
            res[key] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"  {key:14s} FAILED: {res[key]['error']}")

    # -- pairwise: fit ONCE per target, score under 3 voters --------------
    for pkey, positive in PAIRWISE_FITS:
        try:
            t0 = time.time()
            pw = Pairwise(pypper(), positive=positive, combiner="vote",
                          random_state=RANDOM_STATE).fit(c.train_rep)
            train_s = time.time() - t0
            cx[pkey] = _complexity(pw, n, pkey)
            for vkey, Voter in VOTERS:
                pw.combiner = Voter()
                acc, pred_s = evaluate(pw)
                # train time is charged once (to :vote); the others reuse the fit
                res[f"{pkey}:{vkey}"] = dict(
                    acc=acc, train_s=train_s if vkey == "vote" else 0.0, predict_s=pred_s)
                tag = f"{train_s:8.1f}s" if vkey == "vote" else " (shared)"
                print(f"  {pkey + ':' + vkey:14s} acc={acc:.3f}  train={tag}  pred={pred_s:7.2f}s  "
                      f"models={cx[pkey]['models']:3d}  rules={cx[pkey]['rules']:4d}")
        except Exception as e:  # noqa: BLE001
            for vkey, _ in VOTERS:
                res[f"{pkey}:{vkey}"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"  {pkey:14s} FAILED: {res[f'{pkey}:vote']['error']}")
    return row


# -- report -----------------------------------------------------------------

def _mean(rows, key, field):
    vals = [r["res"][key][field] for r in rows
            if key in r["res"] and field in r["res"][key]]
    return np.mean(vals) if vals else float("nan")


def write_report(results):
    L = ["# Pairwise vs. one-vs-rest decomposition for Pypper", "",
         "One inner learner (`Pypper._stage_learner()`), shared binarized features, 70/30 "
         "stratified split, seed 0. 5 fitted models; the 3 pairwise ones are each scored under "
         "`MajorityVote` (`:vote`), `WeightedVote` (`:wv`), `AccuracyWeightedVote` (`:awv`) on "
         "the *same fit* -- their `train s` is 0 (shared). `pw_min`/`pw_maj`/`pw_both` = "
         "`Pairwise(positive=)` `smaller`/`larger`/`both`.", ""]

    L += ["## Accuracy -- all 11 results", "",
          "| dataset | n | classes | " + " | ".join(RESULT_KEYS) + " |",
          "|---|--:|--:|" + "--:|" * len(RESULT_KEYS)]
    for r in results:
        cells = []
        for k in RESULT_KEYS:
            a = r["res"].get(k, {})
            cells.append("—" if "acc" not in a else f"{a['acc']:.3f}")
        L.append(f"| {r['name']} | {r['n']} | {r['n_classes']} | " + " | ".join(cells) + " |")
    L += ["", "| result | datasets | mean acc | mean predict s |", "|---|--:|--:|--:|"]
    for k in RESULT_KEYS:
        ok = [r for r in results if "acc" in r["res"].get(k, {})]
        L.append(f"| {k} | {len(ok)} | {np.mean([r['res'][k]['acc'] for r in ok]):.3f} | "
                 f"{_mean(results, k, 'predict_s'):.2f} |" if ok else f"| {k} | 0 | — | — |")

    L += ["", "### predict time (s) -- all 11 results", "",
          "| dataset | " + " | ".join(RESULT_KEYS) + " |",
          "|---|" + "--:|" * len(RESULT_KEYS)]
    for r in results:
        cells = []
        for k in RESULT_KEYS:
            a = r["res"].get(k, {})
            cells.append("—" if "predict_s" not in a else f"{a['predict_s']:.2f}")
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")

    L += ["", "## Train vs. predict time (s) -- the 5 fitted models", "",
          "| dataset | " + " | ".join(f"{k} train / pred" for k in FIT_KEYS) + " |",
          "|---|" + "--:|" * len(FIT_KEYS)]
    for r in results:
        cells = []
        for k in FIT_KEYS:
            a = r["res"].get(k) or r["res"].get(f"{k}:vote", {})
            cells.append("—" if "train_s" not in a else f"{a['train_s']:.1f} / {a['predict_s']:.2f}")
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    L += ["", "| model | mean train s | mean predict s (vote) |", "|---|--:|--:|"]
    for k in FIT_KEYS:
        vote_key = k if k in ("ovr", "ovr_ord") else f"{k}:vote"
        L.append(f"| {k} | {_mean(results, vote_key, 'train_s'):.2f} | "
                 f"{_mean(results, vote_key, 'predict_s'):.2f} |")
    L += ["", "*(train time for `:wv` / `:awv` is 0 -- same fit as `:vote`. `:wv` roughly "
          "doubles predict time -- it re-resolves every member's covering rules; `:awv` is as "
          "cheap as `:vote`. Per-result predict times are in the section above.)*"]

    L += ["", "## Rule complexity -- the 5 fitted models", "",
          "| dataset | " + " | ".join(f"{k} models/rules/conds" for k in FIT_KEYS) + " |",
          "|---|" + "--:|" * len(FIT_KEYS)]
    for r in results:
        cells = []
        for k in FIT_KEYS:
            x = r["cx"].get(k)
            cells.append("—" if not x else f"{x['models']}/{x['rules']}/{x['conds']}")
        L.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    L += ["", "| model | mean models | mean rules | mean conds |", "|---|--:|--:|--:|"]
    for k in FIT_KEYS:
        xs = [r["cx"][k] for r in results if k in r["cx"]]
        if xs:
            L.append(f"| {k} | {np.mean([x['models'] for x in xs]):.1f} | "
                     f"{np.mean([x['rules'] for x in xs]):.1f} | "
                     f"{np.mean([x['conds'] for x in xs]):.1f} |")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"\nReport -> {REPORT_PATH}")


# -- plots ----------------------------------------------------------------

def _grouped_bar(ax, results, keys, colors, field, getter):
    names = [r["name"] for r in results]
    x = np.arange(len(names))
    w = 0.9 / len(keys)
    for i, k in enumerate(keys):
        vals = [getter(r, k) for r in results]
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, vals, w, label=k, color=colors[k])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")


def write_plots(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # 1. accuracy -- all 11
    fig, ax = plt.subplots(figsize=(20, 6))
    _grouped_bar(ax, results, RESULT_KEYS, _COLORS, "acc",
                 lambda r, k: r["res"].get(k, {}).get("acc", np.nan))
    ax.set_ylabel("test accuracy"); ax.set_ylim(0.35, 1.0)
    ax.set_title("Pypper decomposition -- accuracy (all 11)")
    ax.legend(fontsize=8, ncol=11, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(f"{PLOT_PREFIX}_accuracy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot   -> {PLOT_PREFIX}_accuracy.png")

    # 2. runtime -- 5 fitted models, train vs predict
    def _fit_res(r, k):
        return r["res"].get(k) or r["res"].get(f"{k}:vote", {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    _grouped_bar(ax1, results, FIT_KEYS, _FIT_COLORS, "train_s",
                 lambda r, k: _fit_res(r, k).get("train_s", np.nan))
    _grouped_bar(ax2, results, FIT_KEYS, _FIT_COLORS, "predict_s",
                 lambda r, k: _fit_res(r, k).get("predict_s", np.nan))
    ax1.set_yscale("log"); ax1.set_ylabel("train time (s, log)"); ax1.set_title("training")
    ax2.set_yscale("log"); ax2.set_ylabel("predict time (s, log)")
    ax2.set_title("prediction (MajorityVote)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PREFIX}_runtime.png", dpi=110)
    plt.close(fig)
    print(f"Plot   -> {PLOT_PREFIX}_runtime.png")

    # 3. rule complexity -- 5 fitted models
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    _grouped_bar(ax1, results, FIT_KEYS, _FIT_COLORS, "rules",
                 lambda r, k: r["cx"].get(k, {}).get("rules", np.nan))
    _grouped_bar(ax2, results, FIT_KEYS, _FIT_COLORS, "conds",
                 lambda r, k: r["cx"].get(k, {}).get("conds", np.nan))
    ax1.set_yscale("log"); ax1.set_ylabel("rules (log)"); ax1.set_title("rule count")
    ax2.set_yscale("log"); ax2.set_ylabel("total conditions (log)"); ax2.set_title("total conditions")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{PLOT_PREFIX}_complexity.png", dpi=110)
    plt.close(fig)
    print(f"Plot   -> {PLOT_PREFIX}_complexity.png")


def main():
    warnings.simplefilter("ignore")
    print(f"row cap: {MAX_ROWS if MAX_ROWS is not None else 'none (uncapped)'}")
    results = []
    for name in DATASETS:
        try:
            results.append(run_dataset(name))
        except Exception as e:  # noqa: BLE001
            print(f"\n{name}  SKIPPED ({type(e).__name__}: {str(e)[:100]})")
    write_report(results)
    write_plots(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=None, metavar="N",
                    help="subsample datasets larger than N rows (default: no cap)")
    args = ap.parse_args()
    MAX_ROWS = args.cap
    main()
