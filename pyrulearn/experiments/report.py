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

from typing import Sequence


def render_setup_section(description: str) -> str:
    """The report's Setup section, from one `DESCRIPTION` string a demo
    writes about its own experiment (data, protocol, variants and their
    settings, measures) -- replacing "refer to the script's docstring"
    (``examples/REVISION_PLAN.md``'s "For every demo" convention)."""
    return f"## Setup\n\n{description.strip()}\n\n"


def render_results_table(results, columns: Sequence[str], group_by: str = "dataset",
                         float_format: str = "{:.3f}") -> str:
    """A plain Markdown table over `results` (a `run_cv`-shaped
    long-format DataFrame): one row per `group_by` value (default:
    per dataset) then an ``overall`` row averaging across all of them,
    one column per `columns` entry averaged within each group -- e.g.
    ``render_results_table(results, ["accuracy", "fit_time"])`` for a
    per-dataset-then-overall accuracy/runtime table. Values are the
    plain mean across whatever rows fall in that group (`run_cv`'s
    per-fold rows, most commonly) -- NaN (every fold failed) prints as
    ``n/a``. `results` is not mutated."""
    import pandas as pd

    grouped = results.groupby(group_by)[list(columns)].mean(numeric_only=True)
    overall = results[list(columns)].mean(numeric_only=True)
    overall.name = "overall"
    table = pd.concat([grouped, overall.to_frame().T])

    lines = [f"| {group_by} | " + " | ".join(columns) + " |",
            "|" + "---|" * (len(columns) + 1)]
    for name, row in table.iterrows():
        cells = [(float_format.format(v) if v == v else "n/a") for v in row]  # v == v is False for NaN
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
