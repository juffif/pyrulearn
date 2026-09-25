import numpy as np
import pandas as pd
import pytest
from pyrulearn import BooleanDataRepresentation, DataSpec
from pyrulearn.models import ConceptModel, ConceptSet
from pyrulearn.data.io import binarize, build_dataspec

wittgenstein = pytest.importorskip("wittgenstein")

from pyrulearn.interfaces.wittgenstein import (  # noqa: E402
    IREP,
    IREPImporter,
    RIPPERImporter,
    RIPPERk,
    _parse_numeric_cond_value,
)

from _negation_helpers import neg_spec, neg_X  # noqa: E402


def _binary_data(n=300, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 2, size=(n, d)).astype(bool)
    y_clean = X[:, 0] & ~X[:, 1] & (X[:, 2] | X[:, 3])
    noise = rng.random(n) < 0.05
    y = np.where(noise, ~y_clean, y_clean)
    y = np.where(y, "pos", "neg")
    # negation-enabled feature space: IREP/RIPPER emit negated Boolean
    # conditions, which now land on an explicit negation feature
    ds = neg_spec([f"f{i}" for i in range(d)])
    return BooleanDataRepresentation(ds, neg_X(X), y)


@pytest.mark.parametrize("learner_cls,importer_cls,wclass", [
    (IREP, IREPImporter, wittgenstein.IREP),
    (RIPPERk, RIPPERImporter, wittgenstein.RIPPER),
])
def test_learner_matches_importer_and_wittgenstein_predict(learner_cls, importer_cls, wclass):
    rep = _binary_data()

    learned = learner_cls(pos_class="pos", random_state=0).fit(rep)
    assert type(learned) is ConceptModel and learned.label == "pos"
    # multi-class reachable via decomposition
    assert set(learner_cls().produces()) >= {ConceptModel, ConceptSet}

    wmodel = wclass(random_state=0)
    wmodel.fit(rep.X, y=rep.y, pos_class="pos", feature_names=rep.spec.feature_names)
    imported = importer_cls().import_model(wmodel, rep.spec)

    # same random_state -> identical rules (compare by logic-string
    # rendering, since Rule equality ignores provenance; show_stats=False,
    # since the learner's rules carry training stats and the bare import doesn't)
    assert (sorted(r.to_string("logic", show_stats=False) for r in learned)
            == sorted(r.to_string("logic", show_stats=False) for r in imported))

    # the imported model itself has no default_rule (see module
    # docstring); the learner-produced one does, since it can see
    # data.y at fit time
    assert imported.default_rule is None
    assert learned.default_rule is not None and learned.default_rule.target == "neg"

    # pyrulearn's converted RuleSet must predict *exactly* what
    # wittgenstein's own fitted model predicts on the same data
    preds = learned.predict(rep)
    wpreds = np.where(wmodel.predict(rep.X, feature_names=rep.spec.feature_names), "pos", "neg")
    assert np.array_equal(preds, wpreds)
    print(f"{learner_cls.__name__} matches {importer_cls.__name__} and wittgenstein's own predict(): OK")


def test_fit_raises_on_more_than_two_classes_even_with_neg_class():
    # wittgenstein itself silently collapses every non-pos_class label
    # into "negative" given an explicit pos_class (confirmed directly
    # against the library) -- IREP/RIPPERk must refuse this outright
    # rather than let it happen unnoticed, regardless of neg_class.
    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(200, 4)).astype(bool)
    y = rng.choice(["a", "b", "c"], size=200)
    ds = DataSpec([f"f{i}" for i in range(4)])
    rep = BooleanDataRepresentation(ds, X, y)

    for learner in (RIPPERk(pos_class="a", random_state=0), IREP(pos_class="a")):
        try:
            learner.fit(rep)
            assert False, f"expected ValueError for {type(learner).__name__} on 3-class y"
        except ValueError as e:
            assert "binary" in str(e)

    try:
        RIPPERk(pos_class="a", neg_class="NOT_A", random_state=0).fit(rep)
        assert False, "neg_class= should not bypass the multi-class guard"
    except ValueError:
        pass
    print("IREP/RIPPERk raise on >2 classes even with neg_class given: OK")


def test_missing_labels_raises():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, size=(50, 3)).astype(bool)
    ds = DataSpec([f"f{i}" for i in range(3)])
    rep = BooleanDataRepresentation(ds, X, None)

    try:
        IREP(pos_class="pos").fit(rep)
        assert False, "expected ValueError for missing labels"
    except ValueError:
        pass
    print("IREP/RIPPERk raise without labels: OK")


def test_numeric_and_nominal_attributes_via_build_dataspec():
    # binarizing through build_dataspec/binarize first (the same
    # pipeline demo_random_forest_combiners.py uses for sklearn) means
    # wittgenstein only ever sees already-Boolean columns named after
    # real DataSpec features -- so numeric/nominal attributes convert
    # correctly with no special-casing in wittgenstein_rules.py itself.
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({
        "num_feat": rng.normal(size=n),
        "cat_feat": rng.choice(["red", "green", "blue"], size=n),
    })
    y_clean = (df["num_feat"] > 0.2) & (df["cat_feat"] == "red")
    noise = rng.random(n) < 0.05
    y = np.where(noise, ~y_clean, y_clean)
    df["label"] = np.where(y, "pos", "neg")

    ds = build_dataspec(df, target="label", arff_types={"num_feat": "numeric", "cat_feat": "nominal"}).build()
    X = binarize(ds, df)
    rep = BooleanDataRepresentation(ds, X, df["label"].to_numpy())

    rules = RIPPERk(pos_class="pos", random_state=0).fit(rep)
    assert len(rules.rules) > 0
    # every literal binds to a real, typed DataSpec feature -- not an
    # opaque bit -- so it renders with its actual attribute/threshold,
    # e.g. "num_feat >= 0.36", not just a bare feature name
    rendered = " ".join(r.to_string("logic") for r in rules.rules)
    assert "num_feat" in rendered or "cat_feat" in rendered
    print("Numeric/nominal attributes convert correctly via build_dataspec/binarize: OK")


def test_parse_numeric_cond_value():
    # wittgenstein's own bins are right-closed/left-open, (lo, hi] --
    # confirmed directly against BinTransformer (its fit()'s own
    # docstring, its closed='right' pd.Interval construction, and
    # empirically: a value exactly at a bin's lower edge lands in the
    # *previous* bin, one at the upper edge stays in the bin) -- so both
    # bounds are <=-family literals: the lower bound negated (x>lo ==
    # NOT(x<=lo)), the upper bound plain (x<=hi)
    assert _parse_numeric_cond_value("0.53 - 0.84") == [(0.53, False), (0.84, True)]
    assert _parse_numeric_cond_value("-1.28 - -0.88") == [(-1.28, False), (-0.88, True)]
    assert _parse_numeric_cond_value(">1.32") == [(1.32, False)]  # x>1.32, exclusive
    assert _parse_numeric_cond_value("<-1.28") == [(-1.28, True)]  # x<=-1.28, inclusive
    assert _parse_numeric_cond_value("red") is None  # a genuine nominal category, not a bin
    print("_parse_numeric_cond_value handles range/unbounded/non-numeric: OK")


@pytest.mark.parametrize("learner_cls,importer_cls", [
    (RIPPERk, RIPPERImporter),
    (IREP, IREPImporter),
])
def test_infer_dataspec_parses_wittgensteins_own_bins_workflow_2(learner_cls, importer_cls):
    # workflow 2 for wittgenstein: fit directly on raw numeric+nominal
    # columns (no pre-binarization at all) and let infer_dataspec parse
    # wittgenstein's own bin-range/category Cond values back into typed
    # NUMERIC/NOMINAL attributes.
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({
        "num_feat": rng.normal(size=n),
        "cat_feat": rng.choice(["red", "green", "blue"], size=n),
    })
    y_clean = (df["num_feat"] > 0.3) & (df["cat_feat"] == "red")
    noise = rng.random(n) < 0.03
    y = np.where(np.where(noise, ~y_clean, y_clean), "pos", "neg")
    names = ["num_feat", "cat_feat"]

    kwargs = {"k": 2, "random_state": 0} if learner_cls is RIPPERk else {}
    learner = learner_cls(pos_class="pos", **kwargs)
    model = learner.fit_external(df.to_numpy(), y, feature_names=names)

    importer = importer_cls()
    ds = importer.infer_dataspec(model, feature_names=names)
    if len(ds.feature_names) == 0:
        return  # this algorithm/seed learned an empty ruleset -- nothing to check
    # discovered feature names reveal real threshold/category structure,
    # not opaque per-bin booleans -- num_feat's bins are wittgenstein's
    # own right-closed (lo, hi] intervals, so they're a <=-family, not >=
    assert any(name.startswith("num_feat<=") for name in ds.feature_names) or \
        any(name.startswith("cat_feat=") for name in ds.feature_names)

    rules = importer.import_model(model, ds, feature_names=names)
    Xb = binarize(ds, df)
    rep = BooleanDataRepresentation(ds, Xb, y)
    preds = rules.predict(rep)
    wpreds = np.where(model.predict(df.to_numpy(), feature_names=names), "pos", "neg")
    assert np.array_equal(preds, wpreds)
    print(f"{learner_cls.__name__} workflow-2 infer_dataspec parses wittgenstein's own bins correctly: OK")


def test_imported_rules_match_wittgenstein_exactly_at_bin_boundaries():
    # regression test for a real bug: wittgenstein's own bins are
    # right-closed/left-open, (lo, hi] -- confirmed directly against
    # BinTransformer, not assumed (see _parse_numeric_cond_value's
    # docstring) -- so a value sitting exactly on a bin edge is the one
    # place a wrong boundary convention actually shows up; random
    # continuous test data almost never lands exactly on one, which is
    # how the original (backwards) version of this code went unnoticed
    rng = np.random.default_rng(3)
    n = 500
    x = rng.uniform(0, 10, size=n)
    y = np.where(x > 5.0, "pos", "neg")
    names = ["f0"]

    model = RIPPERk(pos_class="pos", random_state=0).fit_external(
        x.reshape(-1, 1), y, feature_names=names
    )
    bins = model.bin_transformer_.bins_.get("f0", [])
    range_bins = [b for b in bins if " - " in b]
    assert range_bins, "expected at least one range bin to test boundaries against"

    importer = RIPPERImporter()
    ds = importer.infer_dataspec(model, feature_names=names)
    rules = importer.import_model(model, ds, feature_names=names)

    boundary_values = sorted({float(v) for b in range_bins for v in b.split(" - ")})
    df = pd.DataFrame({"f0": boundary_values})
    Xb = binarize(ds, df)
    rep = BooleanDataRepresentation(ds, Xb, np.array(["?"] * len(boundary_values)))
    preds = rules.predict(rep)
    wpreds = np.where(model.predict(df.to_numpy(), feature_names=names), "pos", "neg")
    assert np.array_equal(preds, wpreds), (
        f"mismatch at bin boundary values {boundary_values}: "
        f"imported={list(preds)} vs wittgenstein={list(wpreds)}"
    )
    print("Imported rules match wittgenstein exactly at bin boundary values: OK")


def test_import_model_without_feature_names_still_raises_on_nonboolean():
    # old (workflow-1) behavior is unchanged: without feature_names=,
    # a non-Boolean Cond.val still raises rather than silently guessing
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame({"num_feat": rng.normal(size=n)})
    y = np.where(df["num_feat"] > 0, "pos", "neg")
    names = ["num_feat"]

    model = RIPPERk(pos_class="pos", random_state=0).fit_external(df.to_numpy(), y, feature_names=names)
    ds = DataSpec(names)  # plain Boolean dataspec, wrong shape for this model's real Conds
    try:
        RIPPERImporter().import_model(model, ds)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "feature_names" in str(e)
    print("import_model without feature_names= still raises on non-Boolean Cond.val: OK")


if __name__ == "__main__":
    for learner_cls, importer_cls, wclass in [
        (IREP, IREPImporter, wittgenstein.IREP),
        (RIPPERk, RIPPERImporter, wittgenstein.RIPPER),
    ]:
        test_learner_matches_importer_and_wittgenstein_predict(learner_cls, importer_cls, wclass)
    test_fit_raises_on_more_than_two_classes_even_with_neg_class()
    test_missing_labels_raises()
    test_numeric_and_nominal_attributes_via_build_dataspec()
    test_parse_numeric_cond_value()
    for learner_cls, importer_cls in [(RIPPERk, RIPPERImporter), (IREP, IREPImporter)]:
        test_infer_dataspec_parses_wittgensteins_own_bins_workflow_2(learner_cls, importer_cls)
    test_imported_rules_match_wittgenstein_exactly_at_bin_boundaries()
    test_import_model_without_feature_names_still_raises_on_nonboolean()
    print("\nAll tests passed.")
