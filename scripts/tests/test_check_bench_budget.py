# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/check-bench-budget.py`.

The fixtures below are trimmed copies of real CodSpeed comments on this repository, so the
parser is tested against the markup CodSpeed actually posts rather than an idealised
version of it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_bench_budget", Path(__file__).parent.parent / "check-bench-budget.py"
)
assert _SPEC is not None and _SPEC.loader is not None
budget = importlib.util.module_from_spec(_SPEC)
sys.modules["check_bench_budget"] = budget
_SPEC.loader.exec_module(budget)


def _row(marker: str, name: str, uri: str, base: str, head: str, efficiency: str) -> str:
    link = f"https://app.codspeed.io/vortex-data/vortex/branches/x?uri={uri}&runnerMode=Simulation"
    return f"| {marker} | Simulation | [`` {name} ``]({link}) | {base} | {head} | {efficiency} |"


NEW_BENCHMARKS_COMMENT = "\n".join(
    [
        "<!-- __CODSPEED_PERFORMANCE_REPORT_COMMENT__ -->",
        "## Merging this PR will **not alter performance**",
        "",
        "`✅ 1885` untouched benchmarks  ",
        "`🆕 75` new benchmarks  ",
        "",
        "### Performance Changes",
        "",
        "|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |",
        "| --- | ---- | --------- | ------ | ------ | ---------- |",
        _row(
            "🆕",
            "inline[4096]",
            "vortex-array%2Fbenches%2Fbyte_length.rs%3A%3Ainline%5B4096%5D",
            "N/A",
            "61.8 µs",
            "N/A",
        ),
        _row(
            "🆕",
            "like_per_row_distinct_patterns",
            "vortex-array%2Fbenches%2Flike.rs%3A%3Alike_per_row_distinct_patterns",
            "N/A",
            "1.1 ms",
            "N/A",
        ),
        _row(
            "🆕",
            "column_x_column_polygons",
            "vortex-geo%2Fbenches%2Fbinary_predicates.rs%3A%3Acontains%3A%3Acolumn_x_column_polygons",
            "N/A",
            "23.8 ms",
            "N/A",
        ),
        _row(
            "🆕",
            "constant_x_polygons_overlapping",
            "vortex-geo%2Fbenches%2Fbinary_predicates.rs%3A%3Acontains%3A%3Aconstant_x_polygons_overlapping",
            "N/A",
            "123.4 ms",
            "N/A",
        ),
        "| ... | ... | ... | ... | ... | ... |",
        "",
        "> :information_source: _Only the first 20 benchmarks are displayed._",
        "",
        "<sub>Comparing <code>a</code> (9755708) with <code>develop</code> (c2288dc)</sub>",
    ]
)

IMPROVED_COMMENT = "\n".join(
    [
        "<!-- __CODSPEED_PERFORMANCE_REPORT_COMMENT__ -->",
        "## Merging this PR will **improve performance by 27.76%**",
        "",
        "`⚡ 2` improved benchmarks  ",
        "`✅ 1840` untouched benchmarks  ",
        "",
        "### Performance Changes",
        "",
        "|     | Mode | Benchmark | `BASE` | `HEAD` | Efficiency |",
        "| --- | ---- | --------- | ------ | ------ | ---------- |",
        _row(
            "⚡",
            "take_map[(0.1, 1.0)]",
            "vortex-array%2Fbenches%2Ftake_patches.rs%3A%3Atake_map",
            "2.2 ms",
            "1.6 ms",
            "+34.9%",
        ),
        _row(
            "⚡",
            "take_map[(0.1, 0.5)]",
            "vortex-array%2Fbenches%2Ftake_patches.rs%3A%3Atake_map2",
            "1,182.5 µs",
            "977.3 µs",
            "+20.99%",
        ),
    ]
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("61.8 µs", 61_800.0),
        ("61.8 μs", 61_800.0),  # micro sign vs greek mu
        ("61.8 us", 61_800.0),
        ("1.1 ms", 1_100_000.0),
        ("1,182.5 µs", 1_182_500.0),
        ("794 ns", 794.0),
        ("1.5 s", 1_500_000_000.0),
        ("N/A", None),
        ("", None),
        ("...", None),
        ("+34.9%", None),
    ],
)
def test_parse_duration(text: str, expected: float | None) -> None:
    assert budget.parse_duration(text) == expected


@pytest.mark.parametrize(
    ("nanos", "expected"),
    [(794, "794 ns"), (61_800, "61.8 µs"), (23_800_000, "23.8 ms"), (1_500_000_000, "1.5 s")],
)
def test_format_duration(nanos: float, expected: str) -> None:
    assert budget.format_duration(nanos) == expected


def test_parse_comment_reads_new_benchmarks() -> None:
    benchmarks, truncated = budget.parse_comment(NEW_BENCHMARKS_COMMENT)

    assert truncated
    assert [b.name for b in benchmarks] == [
        "inline[4096]",
        "like_per_row_distinct_patterns",
        "column_x_column_polygons",
        "constant_x_polygons_overlapping",
    ]
    assert {b.status for b in benchmarks} == {"new"}
    assert benchmarks[0].mode == "Simulation"
    # The URI is percent-decoded, and identifies the file the benchmark lives in.
    assert benchmarks[2].uri == ("vortex-geo/benches/binary_predicates.rs::contains::column_x_column_polygons")
    assert benchmarks[3].head_ns == 123_400_000


def test_parse_comment_skips_header_separator_and_truncation_rows() -> None:
    benchmarks, _ = budget.parse_comment(NEW_BENCHMARKS_COMMENT)
    assert len(benchmarks) == NEW_BENCHMARKS_COMMENT.count("app.codspeed.io")


def test_parse_comment_without_a_table() -> None:
    body = "<!-- __CODSPEED_PERFORMANCE_REPORT_COMMENT__ -->\nNo performance changes.\n"
    assert budget.parse_comment(body) == ([], False)


def test_over_budget_flags_only_slow_benchmarks() -> None:
    benchmarks, _ = budget.parse_comment(NEW_BENCHMARKS_COMMENT)
    violations = budget.over_budget(benchmarks, budget.DEFAULT_MAX_NS)

    # Sorted slowest first; the 61.8 µs benchmark is comfortably inside the budget.
    assert [b.name for b in violations] == [
        "constant_x_polygons_overlapping",
        "column_x_column_polygons",
        "like_per_row_distinct_patterns",
    ]


def test_over_budget_ignores_improvements_by_default() -> None:
    benchmarks, _ = budget.parse_comment(IMPROVED_COMMENT)

    assert budget.over_budget(benchmarks, budget.DEFAULT_MAX_NS) == []
    # Only the 1.6 ms row is over budget; its 977.3 µs sibling is inside it either way.
    included = budget.over_budget(benchmarks, budget.DEFAULT_MAX_NS, include_improved=True)
    assert [b.name for b in included] == ["take_map[(0.1, 1.0)]"]


def test_over_budget_honours_a_custom_budget() -> None:
    benchmarks, _ = budget.parse_comment(NEW_BENCHMARKS_COMMENT)
    violations = budget.over_budget(benchmarks, 50_000_000)
    assert [b.name for b in violations] == ["constant_x_polygons_overlapping"]


def test_render_report_lists_violations() -> None:
    benchmarks, truncated = budget.parse_comment(NEW_BENCHMARKS_COMMENT)
    violations = budget.over_budget(benchmarks, budget.DEFAULT_MAX_NS)
    report = budget.render_report(violations, len(benchmarks), budget.DEFAULT_MAX_NS, truncated)

    assert "`⚠️ 3` of the `4` benchmark(s)" in report
    assert "123.4 ms" in report
    assert "123.4×" in report
    assert "vortex-geo/benches/binary_predicates.rs::contains::constant_x_polygons_overlapping" in report
    assert "#[cfg(not(codspeed))]" in report
    assert "CodSpeed truncated its own table" in report
    # A benchmark inside the budget is never named.
    assert "inline[4096]" not in report


def test_render_report_when_everything_is_within_budget() -> None:
    benchmarks, truncated = budget.parse_comment(IMPROVED_COMMENT)
    report = budget.render_report([], len(benchmarks), budget.DEFAULT_MAX_NS, truncated)

    assert "`✅ 2` benchmark(s)" in report
    assert "| Benchmark |" not in report
    assert "CodSpeed truncated its own table" not in report


def test_main_writes_comment_and_step_output(tmp_path: Path) -> None:
    comment = tmp_path / "codspeed.md"
    comment.write_text(NEW_BENCHMARKS_COMMENT)
    output, step_output = tmp_path / "comment.md", tmp_path / "github_output"

    argv = [
        "check-bench-budget.py",
        "--comment-file",
        str(comment),
        "--output",
        str(output),
        "--github-output",
        str(step_output),
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        assert budget.main() == 0

    assert "123.4 ms" in output.read_text()
    assert step_output.read_text() == "violations=3\n"


def test_main_reads_the_comment_from_the_environment(tmp_path: Path) -> None:
    output = tmp_path / "comment.md"
    argv = ["check-bench-budget.py", "--output", str(output), "--github-output", str(tmp_path / "o")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        patch.setenv("CODSPEED_COMMENT_BODY", IMPROVED_COMMENT)
        assert budget.main() == 0

    assert "within the **1 ms** per-iteration budget" in output.read_text()


def test_main_rejects_a_comment_that_is_not_codspeeds(tmp_path: Path) -> None:
    output = tmp_path / "comment.md"
    argv = ["check-bench-budget.py", "--output", str(output), "--github-output", str(tmp_path / "o")]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        patch.setenv("CODSPEED_COMMENT_BODY", "This benchmark has a too long runtime")
        assert budget.main() == 1

    assert not output.exists()


def test_main_can_fail_the_job(tmp_path: Path) -> None:
    output = tmp_path / "comment.md"
    argv = [
        "check-bench-budget.py",
        "--output",
        str(output),
        "--github-output",
        str(tmp_path / "o"),
        "--fail-on-violation",
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        patch.setenv("CODSPEED_COMMENT_BODY", NEW_BENCHMARKS_COMMENT)
        assert budget.main() == 1
