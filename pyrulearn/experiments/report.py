"""
pyrulearn.experiments.report
==============================

Small Markdown-writing building blocks for a demo's report -- not one
do-everything report generator (measures differ per demo, so a report
assembles only the sections it actually needs; see
`pyrulearn.experiments.stats`'s own docstring for the same reasoning).
A demo composes these (plus whatever else it writes directly) into one
Markdown string and saves it, the way every ``examples/demo_*.py``
already does.
"""

from __future__ import annotations

from typing import Sequence, Union


def render_setup_section(description: str) -> str:
    """The report's Setup section, from one `DESCRIPTION` string a demo
    writes about its own experiment (data, protocol, variants and their
    settings, measures) -- replacing "refer to the script's docstring"
    (``examples/REVISION_PLAN.md``'s "For every demo" convention)."""
    return f"## Setup\n\n{description.strip()}\n\n"


def render_results_table(results, columns: Sequence[str], group_by: Union[str, Sequence[str]] = "dataset",
                         float_format: str = "{:.3f}", include_overall: bool = True) -> str:
    """A plain Markdown table over `results` (a `run_cv`-shaped
    long-format DataFrame): one row per `group_by` value (default:
    per dataset) then an ``overall`` row averaging across all of them,
    one column per `columns` entry averaged within each group -- e.g.
    ``render_results_table(results, ["accuracy", "fit_time"])`` for a
    per-dataset-then-overall accuracy/runtime table. Values are the
    plain mean across whatever rows fall in that group (`run_cv`'s
    per-fold rows, most commonly) -- NaN (every fold failed) prints as
    ``n/a``. `results` is not mutated.

    `group_by` can also be a list of columns, e.g. ``["dataset",
    "learner"]`` for one row per (dataset, learner) pair -- **the usual
    choice for a per-dataset table that also compares algorithms**; a
    single `group_by` averages every learner (and everything else)
    together within each group, which silently erases the very
    comparison most reports want.

    `include_overall` (default True) adds that final row, spanning every
    row regardless of how many `group_by` columns there are -- pass
    `False` when `group_by` includes ``"learner"`` (or anything else
    naming a different *thing being compared*, not just a different
    *sample* of the same thing): averaging accuracy across several
    algorithms isn't a meaningful number the way averaging across
    datasets or folds is, so that row would misrepresent the table
    rather than summarize it.
    """
    import pandas as pd

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    grouped = results.groupby(group_cols)[list(columns)].mean(numeric_only=True)
    table = grouped
    if include_overall:
        overall = results[list(columns)].mean(numeric_only=True)
        overall.name = "overall"
        table = pd.concat([grouped, overall.to_frame().T])

    header_cols = group_cols + list(columns)
    lines = ["| " + " | ".join(header_cols) + " |",
            "|" + "---|" * len(header_cols)]
    for name, row in table.iterrows():
        # a plain (single-column) group_by gives a scalar `name`; several
        # columns give a tuple -- except the appended "overall" row,
        # always a scalar, padded out to the same column count.
        key_parts = list(name) if isinstance(name, tuple) else [name] + [""] * (len(group_cols) - 1)
        cells = [str(k) for k in key_parts] + \
            [(float_format.format(v) if v == v else "n/a") for v in row]  # v == v is False for NaN
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
