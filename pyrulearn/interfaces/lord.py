"""
pyrulearn.interfaces.lord
================================

`LORDImporter` -- a `StringRuleImporter` for the printed rule set of
**LORD** (Huynh, Q.-V.P. & Fürnkranz, J., *Efficient learning of large
sets of locally optimal classification rules*, Machine Learning 112
(2023) 571-610; Java implementation at https://github.com/vqphuynh/LORD).

Like `pyrulearn.interfaces.weka`, LORD is not a Python library:
there is no live fitted object to inspect, only the text its
`run.LordRun` main class writes to ``<output_dir>/fold_NN/eg_output.txt``.
`run_lord` (also here) drives it as a subprocess and returns that text;
`LORDImporter.parse` turns it into a `pyrulearn.models.FlatRuleSet`.
`LordJar` (also here) packages the whole round-trip -- write CSV, run
LORD, parse -- as a `pyrulearn.learners` learner with a uniform
`fit(data, model=...)`, the subprocess-backed counterpart of the native
`pyrulearn.learners.pylord.PyLORD`.

LORD isn't distributed as a fat jar -- it's the committed ``bin/``
classes in the repo plus ``libs/weka_3.8_stable.jar``. Clone the repo
and point ``$LORD_CLASSPATH`` at ``<repo>/bin<sep><repo>/libs/
weka_3.8_stable.jar`` (``<sep>`` is ``;`` on Windows, ``:`` elsewhere);
``$LORD_JAVA`` overrides the ``java`` executable. Any JRE 8+ works.

Rule format
-----------
LORD searches, for *every* training example, the best rule (by the
chosen metric -- m-estimate by default) covering it, then keeps the
deduplicated union. Its `RuleInfo.content(selectors)` prints one line
per rule (captured from a real run, not assumed)::

    ------------------------------------------------------------------------------------
    Rule set:
    IF (f3=1) & (f7=0) THEN (Class=pos)	(p=45, n=3, heuristic_value=0.873)
    IF (f1=1) THEN (Class=neg)	(p=88, n=12, heuristic_value=0.79)

    ------------------------------------------------------------------------------------

Each condition is an atom selector ``(attribute=value)``; the body is
joined by ``" & "``; the head is itself a selector ``(classAttr=class)``;
a tab separates the rule from ``(p=<pos covered>, n=<neg covered>,
heuristic_value=<metric>)``. `parse` scans every line matching that
shape (the ``heuristic_value=`` token makes a false positive on other
console output effectively impossible -- no need to anchor on the
``Rule set:`` header), so the whole raw ``eg_output.txt`` can be handed
in directly.

Boolean input, the intended workflow
------------------------------------
LORD does its own discretization of ARFF numeric attributes and treats
every CSV column as nominal. The workflow this importer is built for is
the latter: pre-binarize with `pyrulearn.data.io.build_dataspec`/
`binarize`, write the resulting 0/1 matrix as a CSV with placeholder
column names (LORD, like Weka, would choke on a `DataSpec`'s own
``age>=30``-style names), run LORD, read the rules back. A condition
``(f3=1)`` is then a positive literal on feature ``f3``; ``(f3=0)`` is
"``f3`` is absent", imported -- since rules carry no negative literals --
as a positive literal on ``f3``'s paired negation feature (so the bound
`DataSpec` must be negation-enabled, `DataSpecBuilder`'s default). An
attribute whose values aren't just ``{0, 1}`` (e.g. LORD run on an
already-nominal CSV) is treated as genuinely nominal: ``(color=red)``
becomes a positive literal on the ``color=red`` feature.

**Produces a `FlatRuleSet`, not a `DecisionList`.** LORD's own prediction picks
the single covering rule with the highest metric value (its
`get_best_covering_rule`), falling back to a default class when none
matches. `FlatRuleSet.predict`'s default combiner,
`pyrulearn.combiners.HeuristicMaxCombiner`, only reproduces this exactly
if given the *same* metric LORD was run with (`HeuristicMaxCombiner(
lord_metric)` -- its own default is `Laplace`, which may rank
differently than, say, an entropy or cosine run) -- and needs `data=`
passed to `parse` so its rules have measured stats to score from (a bare
`parse` call with no `data=` leaves rules unable to combine at all
beyond a unanimous covering block; LORD's own printed `heuristic_value`
per rule isn't kept). The rule order in the file carries no prediction
semantics. LORD does not print its default class, so `parse` leaves
`default_prediction=None`; `run_lord`'s caller (which has the labels)
should set the training-majority class, e.g.
``ruleset.default_prediction = MajorityClass(train_rep)``.

Multi-class: LORD is natively multi-class -- each rule's target is read
straight from its head selector, no positive-class restriction.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..attributes import AttributeType
from ..data import DataSpec, DataSpecBuilder
from ..learners import RelabelingExternalLearner
from ..models import FlatRuleSet, MajorityClass
from ..rule import Rule
from .base import StringRuleImporter, register_importer

#: one atom selector, ``(attribute=value)``
_SELECTOR_RE = re.compile(r"\(\s*([^()=|]+?)\s*=\s*([^()|]+?)\s*\)")
#: a full printed rule line
_RULE_RE = re.compile(
    r"^IF\s+(?P<body>.+?)\s+THEN\s+"
    r"\(\s*(?P<head_attr>[^()=|]+?)\s*=\s*(?P<head_val>[^()|]+?)\s*\)"
    r"\s*\(\s*p=(?P<p>[\d.eE+-]+)\s*,\s*n=(?P<n>[\d.eE+-]+)\s*,"
    r"\s*heuristic_value=(?P<h>[\d.eE+-]+)\s*\)\s*$"
)

#: ``(conditions, target)``; conditions are ``(attr, value)`` pairs. The
#: trailing ``(p=..., n=..., heuristic_value=...)`` must still match (it's
#: what makes the line-shape check unambiguous -- see the module
#: docstring), but the numbers themselves aren't kept.
ParsedLORDRule = Tuple[List[Tuple[str, str]], str]


def _parse_rules(source: str) -> List[ParsedLORDRule]:
    parsed: List[ParsedLORDRule] = []
    for line in source.splitlines():
        line = line.strip()
        m = _RULE_RE.match(line)
        if not m:
            continue
        body = m.group("body")
        if "||" in body:
            raise ValueError(
                f"LORD rule uses a disjunctive selector (not supported yet): {line!r}"
            )
        conditions = _SELECTOR_RE.findall(body)
        if not conditions:
            raise ValueError(f"line looks like a LORD rule but has no parseable conditions: {line!r}")
        parsed.append((
            [(a.strip(), v.strip()) for a, v in conditions],
            m.group("head_val").strip(),
        ))
    return parsed


def _infer_dataspec(parsed: Sequence[ParsedLORDRule]) -> DataSpec:
    """One attribute per name seen in a condition: an attribute whose
    values are only ``{"0", "1"}`` (or a subset) becomes a Boolean
    feature (negation on, so ``(x=0)`` has somewhere to land); anything
    else becomes a nominal attribute with every value seen for it.
    Head attributes are the class -- not features -- and are skipped.
    """
    values: Dict[str, Set[str]] = {}
    for conditions, _target in parsed:
        for attr, value in conditions:
            values.setdefault(attr, set()).add(value)

    builder = DataSpecBuilder(negation=True)
    for attr in sorted(values):
        vals = values[attr]
        if vals <= {"0", "1"}:
            builder.add_boolean(attr)
        else:
            builder.add_nominal(attr, sorted(vals))
    return builder.build()


def _rule_from_conditions(conditions: Sequence[Tuple[str, str]], target: str, dataspec: DataSpec) -> Rule:
    pos: List[int] = []
    neg: List[int] = []
    for attr, value in conditions:
        if attr in dataspec.attributes and dataspec.attributes[attr].type in (AttributeType.NOMINAL, AttributeType.BINARY):
            pos.append(dataspec.feature_index(f"{attr}={value}"))
        elif value == "1":
            pos.append(dataspec.feature_index(attr))
        elif value == "0":
            neg.append(dataspec.feature_index(attr))
        else:
            # a bare nominal attribute not declared as one in the given
            # dataspec -- treat "attr=value" as an equality feature lookup
            pos.append(dataspec.feature_index(f"{attr}={value}"))
    return Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec)


def _placeholder_rule_from_conditions(
    conditions: Sequence[Tuple[str, str]], target: str, dataspec: DataSpec,
) -> Rule:
    """`_rule_from_conditions`'s counterpart for a `fit()` round-trip:
    `run_lord` wrote the columns with safe placeholder names `f0/f1/...`,
    so each condition ``(f{i}, "1"|"0")`` binds to `dataspec` feature
    ``i`` **by position** (`Rule.remap` is by-name, so it can't do this).
    ``=1`` -> positive literal, ``=0`` -> negative (paired negation
    feature). LORD's binarized workflow only ever emits ``0``/``1``.
    """
    pos: List[int] = []
    neg: List[int] = []
    for attr, value in conditions:
        if not (attr.startswith("f") and attr[1:].isdigit()):
            raise ValueError(
                f"placeholder_features=True expects f0/f1/... attribute names, got {attr!r}"
            )
        idx = int(attr[1:])
        if value == "1":
            pos.append(idx)
        elif value == "0":
            neg.append(idx)
        else:
            raise ValueError(f"placeholder feature {attr!r} tested against non-Boolean value {value!r}")
    return Rule.from_pos_neg(pos=pos, neg=neg, target=target, dataspec=dataspec)


class LORDImporter(StringRuleImporter):
    """Parses LORD's ``eg_output.txt`` rule set into a `FlatRuleSet`. Pass
    `dataspec=` to bind rules to an existing `DataSpec` (the one the data
    was binarized with); omit it to have one inferred from the conditions
    (Boolean for ``{0,1}``-valued attributes, nominal otherwise). See the
    module docstring for the format and the boolean-input workflow.

    `placeholder_features=True` (used by the `LordJar` learner's `fit()`
    round-trip) requires an explicit `dataspec` and binds each ``f{i}``
    condition to its column ``i`` by position instead of by name.
    """

    SOURCE = "LORD (Huynh, Fürnkranz & Beck, 2023)"

    def __init__(self, dataspec: Optional[DataSpec] = None, placeholder_features: bool = False):
        self.dataspec = dataspec
        self.placeholder_features = placeholder_features

    def infer_dataspec(self, source: str) -> DataSpec:
        return _infer_dataspec(_parse_rules(source))

    def parse(self, source: str, data: Optional[Any] = None) -> FlatRuleSet:
        """`data`, when given (only by the `LordJar` learner's `fit()`
        round trip), is the exact `DataRepresentation` written out for
        LORD; each returned rule gets its own measured `stats` against
        it. `None` (a bare `parse` call on a captured `eg_output.txt`)
        leaves rules un-annotated, unchanged from before.
        """
        parsed = _parse_rules(source)
        if self.placeholder_features:
            if self.dataspec is None:
                raise ValueError("placeholder_features=True requires an explicit dataspec=")
            build = _placeholder_rule_from_conditions
        else:
            if self.dataspec is None:
                self.dataspec = _infer_dataspec(parsed)
            build = _rule_from_conditions
        dataspec = self.dataspec

        rules = [build(conditions, target, dataspec) for conditions, target in parsed]
        rules = self._stamp_rule_provenance(rules, n_rules=len(rules))
        rules = self._stamp_rule_stats(rules, data)
        return self._stamp_provenance(FlatRuleSet(rules), n_rules=len(rules))


register_importer("lord", LORDImporter)


# -- driving the jar --------------------------------------------------------

def _lord_classpath(classpath: Optional[str], java: Optional[str]) -> Tuple[str, str]:
    """Resolve LORD's Java classpath. LORD isn't shipped as a fat jar --
    it's the committed ``bin/`` classes plus ``libs/weka_3.8_stable.jar``
    -- so this is a full ``-cp`` string, not a single file. Taken from the
    argument, else ``$LORD_CLASSPATH``, else ``$LORD_JAR`` (the fat-jar
    case, if you built one). The Java executable is the argument, else
    ``$LORD_JAVA``, else ``java`` on PATH.
    """
    classpath = classpath or os.environ.get("LORD_CLASSPATH") or os.environ.get("LORD_JAR")
    if not classpath:
        raise ValueError(
            "LORD classpath not set -- pass classpath=, or set $LORD_CLASSPATH "
            "(e.g. '<repo>/bin"
            f"{os.pathsep}<repo>/libs/weka_3.8_stable.jar'), or $LORD_JAR if you built a fat jar"
        )
    return classpath, (java or os.environ.get("LORD_JAVA") or "java")


def run_lord(
    train_rows: Sequence[Sequence[Any]],
    header: Sequence[str],
    *,
    classpath: Optional[str] = None,
    java: Optional[str] = None,
    main_class: str = "run.LordRun",
    metric: str = "mestimate",
    metric_arg: float = 0.1,
    timeout: Optional[float] = None,
    test_rows: Optional[Sequence[Sequence[Any]]] = None,
) -> str:
    """Run LORD's ``run.LordRun`` on one train/(test) split and return the
    text of its ``eg_output.txt``.

    `header` is the CSV column names -- **the last column is the class**.
    `train_rows` / `test_rows` are row sequences matching `header`; if
    `test_rows` is omitted the training rows are reused (LORD needs a
    test file to run; its own accuracy number is ignored here). Use
    placeholder feature names (``f0, f1, ...``) -- LORD, like Weka, can't
    parse a `DataSpec`'s ``>=``/``=``-laden feature names as CSV headers.

    `classpath` is LORD's Java classpath (see `_lord_classpath`);
    `main_class` selects the variant (``run.LordRun`` / ``run.LordStarRun``
    / ``run.LordLoopRun``). `metric` is one of LORD's ``-mt`` values
    (``precision``, ``laplace``, ``entropy``, ``mestimate``,
    ``linear_cost``, ``relative_cost``, ``cosine``); `metric_arg` is its
    ``-ma``. Raises if the classpath isn't set, the subprocess fails, or
    no rule output is produced.
    """
    classpath, java_exe = _lord_classpath(classpath, java)
    test_rows = train_rows if test_rows is None else test_rows

    def _write_csv(path: str, rows: Sequence[Sequence[Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(str(c) for c in header) + "\n")
            for row in rows:
                fh.write(",".join(str(c) for c in row) + "\n")

    with tempfile.TemporaryDirectory(prefix="lord_in_") as in_dir, \
         tempfile.TemporaryDirectory(prefix="lord_out_") as out_dir:
        _write_csv(os.path.join(in_dir, "data_train_01.csv"), train_rows)
        _write_csv(os.path.join(in_dir, "data_test_01.csv"), test_rows)

        cmd = [
            java_exe,
            # pin the locale: LORD formats its *summary* stats locale-aware
            # (a German JVM prints "Accuracy: 1,000000"), and parses -ma the
            # same way -- force en-US so both are dot-decimal and stable.
            "-Duser.language=en", "-Duser.country=US",
            "-cp", classpath, main_class,
            "-id", in_dir, "-od", out_dir,
            "-mt", metric, "-ma", str(metric_arg),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"LORD exited with code {result.returncode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        eg_output: Optional[str] = None
        for root, _dirs, files in os.walk(out_dir):
            if "eg_output.txt" in files:
                with open(os.path.join(root, "eg_output.txt"), encoding="utf-8") as fh:
                    eg_output = fh.read()
                break
        if eg_output is None:
            raise RuntimeError(
                f"LORD produced no eg_output.txt under {out_dir}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        if "IF " not in eg_output:
            raise RuntimeError(f"LORD's eg_output.txt has no rules:\n{eg_output}")
        return eg_output


# =============================================== LORD as a fit() learner ======

class LordJar(RelabelingExternalLearner):
    """LORD (Huynh, Fürnkranz & Beck, 2023) driven as a subprocess against its
    committed Java build -- the reference-implementation counterpart of
    the native `pyrulearn.learners.pylord.PyLORD`.

    `fit(data)` writes `data`'s already-Boolean matrix to a temporary CSV
    (safe ``f0..fN`` column names), runs LORD's `run.LordRun` (or
    `LordStarRun` / `LordLoopRun` -- `variant=`), and reads its
    ``eg_output.txt`` rules back through `LORDImporter`
    (`placeholder_features=True`, so rules bind to `data.spec` by column
    position). Produces a `pyrulearn.models.FlatRuleSet`; `fit(data,
    model=ConceptSet | ConceptCascade | PairwiseModel)` for the
    binary-decomposition variants.

    `classpath=` (else ``$LORD_CLASSPATH`` / ``$LORD_JAR``) and `java=`
    (else ``$LORD_JAVA``) locate the build -- see `run_lord`. `metric` /
    `metric_arg` are LORD's ``-mt`` / ``-ma``.
    """

    IMPORTER = LORDImporter
    NATIVE_MODEL = FlatRuleSet
    _MAIN = {"lord": "run.LordRun", "lord*": "run.LordStarRun", "lord-loop": "run.LordLoopRun"}

    def __init__(self, variant: str = "lord", *, classpath: Optional[str] = None,
                 java: Optional[str] = None, timeout: Optional[float] = None,
                 metric: str = "mestimate", metric_arg: float = 0.1):
        if variant not in self._MAIN:
            raise ValueError(f"variant must be one of {sorted(self._MAIN)}, got {variant!r}")
        self.variant = variant
        self.classpath = classpath
        self.java = java
        self.timeout = timeout
        self.metric = metric
        self.metric_arg = metric_arg

    def fit_external(self, X, y, feature_names=None):
        import numpy as np

        X = np.asarray(X).astype(int)
        y = np.asarray(y)
        header = [f"f{i}" for i in range(X.shape[1])] + ["Class"]
        rows = [[*X[i].tolist(), y[i]] for i in range(X.shape[0])]
        return run_lord(rows, header, classpath=self.classpath, java=self.java,
                        main_class=self._MAIN[self.variant], metric=self.metric,
                        metric_arg=self.metric_arg, timeout=self.timeout)

    def _import(self, data):
        text = self.fit_external(data.X, data.y, feature_names=data.spec.feature_names)
        return LORDImporter(dataspec=data.spec, placeholder_features=True).parse(text, data=data)

    def _fit_native(self, data):
        model = super()._fit_native(data)
        if model.default_prediction is None:
            model.default_prediction = MajorityClass(data)
        return model
