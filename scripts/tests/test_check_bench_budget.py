# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_SCRIPT = REPO_ROOT / "scripts" / "check-bench-budget.py"


def load_module():
    spec = importlib.util.spec_from_file_location("check_bench_budget", BUDGET_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


budget = load_module()


def write_raw_result(directory: Path, uri: str, min_ns: float, name: str | None = None) -> None:
    """Write a raw result in the shape the patched divan harness emits."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name or uri.rsplit("::", 1)[-1],
        "uri": uri,
        "config": {},
        "stats": {
            "min_ns": min_ns,
            "max_ns": min_ns * 1.2,
            "mean_ns": min_ns * 1.1,
            "median_ns": min_ns * 1.05,
            "stdev_ns": 0.0,
            "q1_ns": min_ns,
            "q3_ns": min_ns * 1.1,
            "rounds": 3,
            "total_time": 0.1,
            "iqr_outlier_rounds": 0,
            "stdev_outlier_rounds": 0,
            "iter_per_round": 1,
            "warmup_iters": 0,
        },
    }
    (directory / f"{abs(hash(uri))}.json").write_text(json.dumps(payload))


def run_check(tmp_path: Path, **overrides):
    raw = tmp_path / "raw"
    output = tmp_path / "verdict.json"
    args = {
        "raw_results": raw,
        "scope": None,
        "max_ns": budget.DEFAULT_MAX_NS,
        "metric": "min_ns",
        "shard": "1",
        "output": output,
        "fail_on_violation": False,
    }
    args.update(overrides)
    code = budget.check(_Namespace(**args))
    return code, json.loads(output.read_text())


class _Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.mark.parametrize(
    ("nanos", "expected"),
    [(36, "36 ns"), (61_600, "61.6 µs"), (14_100_000, "14.1 ms"), (2_000_000_000, "2 s")],
)
def test_format_duration_matches_divan_units(nanos, expected):
    assert budget.format_duration(nanos) == expected


def test_check_flags_only_over_budget_benchmarks(tmp_path):
    raw = tmp_path / "raw"
    write_raw_result(raw, "vortex-mask/benches/rank.rs::fast", 36)
    write_raw_result(raw, "vortex-geo/benches/envelope.rs::slow", 65_500_000)

    code, verdict = run_check(tmp_path)

    assert code == 0
    assert verdict["in_scope"] == 2
    assert [v["uri"] for v in verdict["violations"]] == ["vortex-geo/benches/envelope.rs::slow"]


def test_check_ignores_benchmarks_codspeed_does_not_measure(tmp_path):
    raw = tmp_path / "raw"
    write_raw_result(raw, "vortex/benches/throughput.rs::gated", 65_500_000)
    write_raw_result(raw, "vortex-mask/benches/rank.rs::fast", 36)
    scope = tmp_path / "scope.txt"
    scope.write_text("Measured: vortex-mask/benches/rank.rs::fast\n")

    code, verdict = run_check(tmp_path, scope=scope)

    assert code == 0
    assert verdict["measured"] == 2
    assert verdict["in_scope"] == 1
    assert verdict["violations"] == []


def test_check_can_fail_the_job(tmp_path):
    raw = tmp_path / "raw"
    write_raw_result(raw, "vortex-geo/benches/envelope.rs::slow", 65_500_000)

    code, _ = run_check(tmp_path, fail_on_violation=True)

    assert code == 1


def test_check_orders_violations_worst_first(tmp_path):
    raw = tmp_path / "raw"
    write_raw_result(raw, "a.rs::mid", 14_100_000)
    write_raw_result(raw, "a.rs::worst", 2_000_000_000)
    write_raw_result(raw, "a.rs::least", 1_000_001)

    _, verdict = run_check(tmp_path)

    assert [v["uri"] for v in verdict["violations"]] == ["a.rs::worst", "a.rs::mid", "a.rs::least"]


@pytest.mark.parametrize(
    "line",
    [
        "Measured: vortex-mask/benches/rank.rs::fast",
        "Checked: vortex-mask/benches/rank.rs::fast",
        "Measured: vortex-mask/benches/rank.rs::fast (group: outer/inner)",
        "  Measured: vortex-mask/benches/rank.rs::fast  ",
    ],
)
def test_load_scope_parses_harness_output(tmp_path, line):
    scope = tmp_path / "scope.txt"
    scope.write_text(f"{line}\n")

    assert budget.load_scope(scope) == {"vortex-mask/benches/rank.rs::fast"}


def test_load_scope_of_none_means_check_everything():
    assert budget.load_scope(None) is None


def test_report_merges_shards_and_sorts_globally():
    body = budget.render_report(
        [
            {
                "shard": "1",
                "max_ns": budget.DEFAULT_MAX_NS,
                "metric": "min_ns",
                "measured": 2,
                "in_scope": 2,
                "violations": [{"uri": "a.rs::mid", "name": "mid", "ns": 14_100_000}],
            },
            {
                "shard": "2",
                "max_ns": budget.DEFAULT_MAX_NS,
                "metric": "min_ns",
                "measured": 3,
                "in_scope": 3,
                "violations": [{"uri": "b.rs::worst", "name": "worst", "ns": 2_000_000_000}],
            },
        ]
    )

    assert "`⚠️ 2` of `5` CodSpeed benchmarks exceed" in body
    assert body.index("b.rs::worst") < body.index("a.rs::mid")
    assert "| `b.rs::worst` | 2 s | 2000.0× |" in body


def test_report_is_reassuring_when_clean():
    body = budget.render_report(
        [
            {
                "shard": "1",
                "max_ns": budget.DEFAULT_MAX_NS,
                "metric": "min_ns",
                "measured": 7,
                "in_scope": 7,
                "violations": [],
            }
        ]
    )

    assert "`✅ 7` CodSpeed benchmarks are within the **1 ms** per-iteration budget." in body
    assert "| Benchmark |" not in body


def test_report_truncates_a_wholesale_regression():
    violations = [{"uri": f"a.rs::b{i}", "name": f"b{i}", "ns": 2_000_000 + i} for i in range(50)]
    body = budget.render_report(
        [
            {
                "shard": "1",
                "max_ns": budget.DEFAULT_MAX_NS,
                "metric": "min_ns",
                "measured": 50,
                "in_scope": 50,
                "violations": violations,
            }
        ]
    )

    assert body.count("| `a.rs::") == budget.MAX_ROWS
    assert f"Only the first {budget.MAX_ROWS} of 50 benchmarks are displayed." in body
