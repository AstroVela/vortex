#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Flag benchmarks that exceed the per-iteration budget, using CodSpeed's own PR comment.

``docs/developer-guide/benchmarking.md`` asks that a single benchmark iteration stay under
1 ms. CodSpeed already measures and publishes exactly that number: its sticky PR comment
lists every benchmark the pull request added or changed, with the per-iteration time under
``HEAD``. This script reads that comment and re-reports the rows that blow the budget, so
nobody has to eyeball a 20-row table of microsecond values to notice a 123 ms benchmark.

Nothing is rebuilt and nothing is re-run: the input is a comment body, so the check costs
one API-free job and a few milliseconds.

Two consequences of that trade fall out of the source data, and the rendered report says
both out loud:

* Only benchmarks CodSpeed reports as new or changed appear in the comment. An untouched
  benchmark that was already over budget is invisible here.
* CodSpeed truncates its table to the first 20 rows, so a pull request that adds many slow
  benchmarks can hide some of them behind the truncation marker.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

DEFAULT_MAX_NS = 1_000_000
"""One millisecond, per the benchmarking guide."""

CODSPEED_MARKER = "__CODSPEED_PERFORMANCE_REPORT_COMMENT__"
"""Hidden marker CodSpeed puts at the top of the comment it owns."""

GUIDE_URL = (
    "https://github.com/vortex-data/vortex/blob/develop/docs/developer-guide/benchmarking.md"
    "#keep-per-iteration-execution-time-under-1-ms"
)

MAX_ROWS = 20
"""Cap the rendered table. CodSpeed itself never reports more rows than this."""

TRUNCATION_MARKER = "Only the first"
"""Substring of CodSpeed's own "only the first N benchmarks are displayed" note."""

UNITS_NS = {"ns": 1, "us": 1_000, "µs": 1_000, "μs": 1_000, "ms": 1_000_000, "s": 1_000_000_000}
"""Divan/CodSpeed duration suffixes. Both micro sign variants appear in the wild."""

STATUS_BY_EMOJI = {
    "🆕": "new",
    "⚡": "improved",
    "❌": "regressed",
    "⚠️": "regressed",
}

FLAGGED_BY_DEFAULT = frozenset({"new", "regressed", "changed"})
"""A benchmark this pull request made *faster* is not a reason to open a complaint."""

DURATION_RE = re.compile(r"^([0-9][0-9,]*(?:\.[0-9]+)?)\s*(ns|us|µs|μs|ms|s)$")
NAME_RE = re.compile(r"``\s*(.+?)\s*``", re.DOTALL)
SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")


@dataclass(frozen=True)
class Benchmark:
    """One row of CodSpeed's "Performance Changes" table."""

    uri: str
    """Fully qualified benchmark URI, e.g. ``vortex-geo/benches/x.rs::contains::points``."""

    name: str
    """Short display name, as CodSpeed renders it."""

    mode: str
    """CodSpeed instrument that produced the number, e.g. ``Simulation``."""

    status: str
    """One of ``new``, ``improved``, ``regressed``, or ``changed``."""

    head_ns: float
    """Per-iteration time on the pull request's head commit, in nanoseconds."""


def parse_duration(text: str) -> float | None:
    """Parse a CodSpeed duration such as ``1,182.5 µs`` into nanoseconds.

    Returns ``None`` for anything that is not a duration, including the ``N/A`` CodSpeed
    prints for the base side of a new benchmark.
    """
    match = DURATION_RE.match(text.strip())
    if match is None:
        return None
    return float(match.group(1).replace(",", "")) * UNITS_NS[match.group(2)]


def format_duration(nanos: float) -> str:
    """Render a nanosecond count using the same units as CodSpeed's own output."""
    for limit, unit, scale in (
        (1_000, "ns", 1),
        (1_000_000, "µs", 1_000),
        (1_000_000_000, "ms", 1_000_000),
    ):
        if nanos < limit:
            return f"{nanos / scale:.4g} {unit}"
    return f"{nanos / 1_000_000_000:.4g} s"


def split_row(line: str) -> list[str]:
    """Split a Markdown table row into its cells."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def parse_benchmark_cell(cell: str) -> tuple[str, str]:
    """Pull ``(uri, name)`` out of the linked benchmark cell.

    The link target carries a percent-encoded ``uri`` query parameter, which is the only
    unambiguous identifier -- the visible label omits the file, so two benchmarks in
    different crates can share it. Falls back to the label when the link is missing.
    """
    name_match = NAME_RE.search(cell)
    name = name_match.group(1) if name_match else cell
    uri = ""
    link_match = re.search(r"\]\((https?://[^)\s]+)\)", cell)
    if link_match:
        query = parse_qs(urlparse(link_match.group(1)).query)
        if query.get("uri"):
            uri = unquote(query["uri"][0])
    return uri or name, name


def parse_comment(body: str) -> tuple[list[Benchmark], bool]:
    """Parse CodSpeed's comment into benchmarks plus whether its table was truncated.

    Rows CodSpeed emits that carry no measurement -- the header, the ``| --- |``
    separator, and the ``| ... |`` truncation row -- are skipped rather than guessed at.
    """
    benchmarks = []
    for line in body.splitlines():
        cells = split_row(line)
        # marker, mode, benchmark, base, head, efficiency
        if len(cells) != 6 or SEPARATOR_RE.match(line.strip()):
            continue
        head_ns = parse_duration(cells[4])
        if head_ns is None:
            continue
        uri, name = parse_benchmark_cell(cells[2])
        status = next(
            (status for emoji, status in STATUS_BY_EMOJI.items() if emoji in cells[0]),
            "changed",
        )
        benchmarks.append(Benchmark(uri=uri, name=name, mode=cells[1] or "unknown", status=status, head_ns=head_ns))

    return benchmarks, TRUNCATION_MARKER in body


def over_budget(benchmarks: list[Benchmark], max_ns: int, include_improved: bool = False) -> list[Benchmark]:
    """Select the benchmarks that exceed the budget, slowest first."""
    flagged = FLAGGED_BY_DEFAULT | ({"improved"} if include_improved else set())
    violations = [b for b in benchmarks if b.head_ns > max_ns and b.status in flagged]
    return sorted(violations, key=lambda b: b.head_ns, reverse=True)


def render_report(violations: list[Benchmark], reported: int, max_ns: int, truncated: bool) -> str:
    """Render the body of the sticky comment."""
    budget = format_duration(max_ns)
    lines = ["## ⏱️ Benchmark iteration budget", ""]

    if not violations:
        lines += [
            f"`✅ {reported}` benchmark(s) changed by this PR are within the **{budget}** per-iteration budget.",
        ]
    else:
        lines += [
            f"`⚠️ {len(violations)}` of the `{reported}` benchmark(s) changed by this PR "
            f"exceed the **{budget}** per-iteration budget.",
            "",
            "| Benchmark | Per-iteration | Over budget | |",
            "| --- | --- | --- | --- |",
        ]
        for violation in violations[:MAX_ROWS]:
            marker = "🆕" if violation.status == "new" else "❌"
            lines.append(
                f"| `{violation.uri}` | {format_duration(violation.head_ns)} "
                f"| {violation.head_ns / max_ns:.1f}× | {marker} |"
            )
        lines += [
            "",
            "Each iteration of a benchmarked closure should finish in under "
            f"{budget}. CodSpeed's simulation instrument runs each benchmark exactly once, "
            "so a slow iteration spends CI time without buying extra signal. Shrink the "
            "input size, or gate the benchmark with `#[cfg(not(codspeed))]`. See "
            f"[the benchmarking guide]({GUIDE_URL}).",
        ]

    lines += [
        "",
        "<details><summary>How this is measured</summary>",
        "",
        "These numbers are CodSpeed's own, read straight from its performance report on "
        "this PR -- nothing is rebuilt or re-run here.",
        "",
        "That means only benchmarks CodSpeed reports as **new or changed** are checked. A "
        "benchmark this PR did not touch is not listed in its report, so an existing "
        "benchmark that is already over budget will not show up here.",
    ]
    if truncated:
        lines += [
            "",
            "> ⚠️ _CodSpeed truncated its own table, so this PR may change more benchmarks "
            "than were checked. [Open the full report in CodSpeed]"
            "(https://app.codspeed.io/vortex-data/vortex) to see the rest._",
        ]
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def read_comment(args: argparse.Namespace) -> str:
    if args.comment_file is not None:
        return args.comment_file.read_text()
    body = os.environ.get(args.comment_env)
    if body is None:
        raise SystemExit(f"environment variable {args.comment_env} is not set")
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--comment-file", type=Path, help="file holding the CodSpeed comment body")
    source.add_argument(
        "--comment-env",
        default="CODSPEED_COMMENT_BODY",
        help="environment variable holding the CodSpeed comment body",
    )
    parser.add_argument(
        "--max-ns",
        type=int,
        default=DEFAULT_MAX_NS,
        help=f"per-iteration budget in nanoseconds (default: {DEFAULT_MAX_NS})",
    )
    parser.add_argument(
        "--include-improved",
        action="store_true",
        help="also flag over-budget benchmarks that this PR made faster",
    )
    parser.add_argument("--output", type=Path, required=True, help="path to write the comment to")
    parser.add_argument(
        "--github-output",
        type=Path,
        default=os.environ.get("GITHUB_OUTPUT"),
        help="path to append the `violations` step output to",
    )
    parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="exit non-zero when a benchmark is over budget (default: report only)",
    )
    args = parser.parse_args()

    body = read_comment(args)
    if CODSPEED_MARKER not in body:
        print(f"comment does not contain {CODSPEED_MARKER}; refusing to parse", file=sys.stderr)
        return 1

    benchmarks, truncated = parse_comment(body)
    violations = over_budget(benchmarks, args.max_ns, args.include_improved)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(violations, len(benchmarks), args.max_ns, truncated))

    if args.github_output is not None:
        with Path(args.github_output).open("a") as f:
            f.write(f"violations={len(violations)}\n")

    print(
        f"{len(benchmarks)} benchmark(s) reported by CodSpeed, "
        f"{len(violations)} over the {format_duration(args.max_ns)} budget"
    )
    for violation in violations:
        print(f"  {format_duration(violation.head_ns):>10}  {violation.uri}")

    return 1 if violations and args.fail_on_violation else 0


if __name__ == "__main__":
    sys.exit(main())
