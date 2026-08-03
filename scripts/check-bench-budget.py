#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Check that CodSpeed benchmarks stay under a per-iteration wall-clock budget.

CodSpeed itself runs in ``simulation`` mode, which executes each benchmark exactly once
and estimates cycles from an instruction trace -- it never reports wall-clock time, so it
cannot enforce the "keep per-iteration execution time under ~1 ms" rule from
``docs/developer-guide/benchmarking.md``.

The ``divan`` dependency is really ``codspeed-divan-compat``. Built *without* ``--cfg
codspeed`` it re-exports CodSpeed's patched divan, which writes one JSON file per
benchmark to ``target/codspeed/walltime/raw_results/divan/`` whenever ``CODSPEED_ENV`` is
set. Those files carry per-iteration statistics (the harness already divides each round by
its iteration count), which is exactly the quantity the budget is written against.

Two subcommands:

``check``
    Read the raw walltime results for one shard, restrict them to the benchmarks CodSpeed
    actually measures, and emit a JSON verdict.

``report``
    Merge the per-shard verdicts into a single Markdown comment body.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_MAX_NS = 1_000_000
"""One millisecond, per the benchmarking guide."""

GUIDE_URL = (
    "https://github.com/vortex-data/vortex/blob/develop/docs/developer-guide/benchmarking.md"
    "#keep-per-iteration-execution-time-under-1-ms"
)

MAX_ROWS = 30
"""Cap the table so a wholesale regression cannot produce an unreadable comment."""


def format_duration(nanos: float) -> str:
    """Render a nanosecond count using the same units as divan's own output."""
    for limit, unit, scale in (
        (1_000, "ns", 1),
        (1_000_000, "µs", 1_000),
        (1_000_000_000, "ms", 1_000_000),
    ):
        if nanos < limit:
            return f"{nanos / scale:.3g} {unit}"
    return f"{nanos / 1_000_000_000:.3g} s"


def load_raw_results(raw_results: Path) -> list[dict]:
    """Load every per-benchmark JSON file written by the patched divan harness."""
    benchmarks = []
    for path in sorted(raw_results.glob("**/*.json")):
        with path.open() as f:
            benchmarks.append(json.load(f))
    return benchmarks


def load_scope(scope: Path | None) -> set[str] | None:
    """Load the set of benchmark URIs CodSpeed measures, or ``None`` for "all of them".

    The URIs are produced by the analysis-mode binaries, which print ``Measured: <uri>``
    (instrumented) or ``Checked: <uri>`` (not instrumented) for every benchmark they run.
    Both modes build that URI with identical code, so the strings match the ``uri`` field
    in the walltime results exactly.
    """
    if scope is None:
        return None
    uris = set()
    for line in scope.read_text().splitlines():
        line = line.strip()
        for prefix in ("Measured: ", "Checked: "):
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        # Instrumented runs append a group suffix that the URI itself does not carry.
        line = line.split(" (group: ", 1)[0].strip()
        if line:
            uris.add(line)
    return uris


def check(args: argparse.Namespace) -> int:
    benchmarks = load_raw_results(args.raw_results)
    scope = load_scope(args.scope)

    in_scope, violations = [], []
    for bench in benchmarks:
        uri = bench["uri"]
        if scope is not None and uri not in scope:
            continue
        in_scope.append(uri)
        nanos = bench["stats"][args.metric]
        if nanos > args.max_ns:
            violations.append({"uri": uri, "name": bench["name"], "ns": nanos})

    violations.sort(key=lambda v: v["ns"], reverse=True)
    verdict = {
        "shard": args.shard,
        "max_ns": args.max_ns,
        "metric": args.metric,
        "measured": len(benchmarks),
        "in_scope": len(in_scope),
        "violations": violations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2) + "\n")

    print(
        f"shard {args.shard}: {len(in_scope)}/{len(benchmarks)} benchmarks in scope, "
        f"{len(violations)} over the {format_duration(args.max_ns)} budget"
    )
    for violation in violations:
        print(f"  {format_duration(violation['ns']):>10}  {violation['uri']}")

    return 1 if violations and args.fail_on_violation else 0


def render_report(verdicts: list[dict]) -> str:
    """Render the merged verdicts as the body of a sticky PR comment."""
    in_scope = sum(v["in_scope"] for v in verdicts)
    violations = [v for verdict in verdicts for v in verdict["violations"]]
    violations.sort(key=lambda v: v["ns"], reverse=True)

    # Every shard is configured identically; fall back to the default if none ran.
    max_ns = verdicts[0]["max_ns"] if verdicts else DEFAULT_MAX_NS
    metric = verdicts[0]["metric"] if verdicts else "min_ns"
    budget = format_duration(max_ns)

    lines = ["## ⏱️ Benchmark iteration budget", ""]

    if not violations:
        lines += [
            f"`✅ {in_scope}` CodSpeed benchmarks are within the **{budget}** "
            "per-iteration budget.",
        ]
    else:
        lines += [
            f"`⚠️ {len(violations)}` of `{in_scope}` CodSpeed benchmarks exceed the "
            f"**{budget}** per-iteration budget.",
            "",
            "CodSpeed's simulation instrument runs each benchmark exactly once, so a slow "
            "iteration costs CI time without buying any extra signal. Shrink the input "
            f"size, or gate the benchmark with `#[cfg(not(codspeed))]`. See [the "
            f"benchmarking guide]({GUIDE_URL}).",
            "",
            "| Benchmark | Fastest iteration | Over budget |",
            "| --- | --- | --- |",
        ]
        for violation in violations[:MAX_ROWS]:
            over = violation["ns"] / max_ns
            lines.append(
                f"| `{violation['uri']}` | {format_duration(violation['ns'])} | {over:.1f}× |"
            )
        if len(violations) > MAX_ROWS:
            lines += [
                "",
                f"> ℹ️ _Only the first {MAX_ROWS} of {len(violations)} benchmarks are "
                "displayed._",
            ]

    lines += [
        "",
        "<details><summary>How this is measured</summary>",
        "",
        "Benchmarks are rebuilt in CodSpeed's walltime mode and run once outside the "
        f"CodSpeed runner. The reported number is `{metric}` -- the fastest observed "
        "iteration, which is the estimate least contaminated by runner noise, so a shared "
        "CI machine cannot make this check flaky.",
        "",
        "Only benchmarks that CodSpeed actually measures are checked. That set comes from "
        "the analysis-mode binaries built by `cargo codspeed build`, which enumerate every "
        "benchmark they run, so anything behind `#[cfg(not(codspeed))]` is excluded "
        "automatically rather than by an allowlist that can drift.",
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


def report(args: argparse.Namespace) -> int:
    verdicts = []
    for path in sorted(args.inputs.glob("**/*.json")):
        with path.open() as f:
            verdicts.append(json.load(f))

    if not verdicts:
        print(f"no verdicts found under {args.inputs}", file=sys.stderr)
        return 1

    body = render_report(verdicts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body)
    print(body)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="check one shard's walltime results against the budget"
    )
    check_parser.add_argument(
        "--raw-results",
        type=Path,
        default=Path("target/codspeed/walltime/raw_results/divan"),
        help="directory of per-benchmark JSON files written by the divan harness",
    )
    check_parser.add_argument(
        "--scope",
        type=Path,
        help="file listing the benchmark URIs CodSpeed measures; all are checked if omitted",
    )
    check_parser.add_argument(
        "--max-ns",
        type=int,
        default=DEFAULT_MAX_NS,
        help=f"per-iteration budget in nanoseconds (default: {DEFAULT_MAX_NS})",
    )
    check_parser.add_argument(
        "--metric",
        choices=["min_ns", "median_ns", "mean_ns", "max_ns"],
        default="min_ns",
        help="statistic to compare against the budget (default: min_ns)",
    )
    check_parser.add_argument("--shard", default="", help="shard name, for reporting")
    check_parser.add_argument(
        "--output", type=Path, required=True, help="path to write the JSON verdict to"
    )
    check_parser.add_argument(
        "--fail-on-violation",
        action="store_true",
        help="exit non-zero when a benchmark is over budget (default: report only)",
    )
    check_parser.set_defaults(func=check)

    report_parser = subparsers.add_parser(
        "report", help="merge per-shard verdicts into a Markdown comment"
    )
    report_parser.add_argument(
        "--inputs", type=Path, required=True, help="directory of JSON verdicts"
    )
    report_parser.add_argument(
        "--output", type=Path, required=True, help="path to write the Markdown body to"
    )
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
