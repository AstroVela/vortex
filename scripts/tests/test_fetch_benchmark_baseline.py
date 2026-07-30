# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch-benchmark-baseline.py"


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_benchmark_baseline", FETCH_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeCursor:
    def __init__(self, columns: list[str], rows: list[tuple[object, ...]]):
        self.description = [(column,) for column in columns]
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, responses: list[FakeCursor]):
        self.responses = responses
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...]):
        self.calls.append((sql, params))
        return self.responses.pop(0)


def query_record(
    commit: str,
    query_idx: int,
    engine: str,
    file_format: str,
) -> dict[str, object]:
    return {
        "kind": "query_measurement",
        "commit_sha": commit,
        "dataset": "tpch",
        "scale_factor": "10",
        "query_idx": query_idx,
        "storage": "nvme",
        "engine": engine,
        "format": file_format,
        "value_ns": 100,
        "all_runtimes_ns": [90, 100, 110],
    }


def test_query_baseline_selects_latest_commit_in_pr_scope() -> None:
    fetch = load_fetch_module()
    pr_records = [
        query_record("pr-sha", 1, "datafusion", "parquet"),
        query_record("pr-sha", 1, "datafusion", "vortex-file-compressed"),
    ]
    columns = [
        "kind",
        "commit_sha",
        "dataset",
        "dataset_variant",
        "scale_factor",
        "query_idx",
        "storage",
        "engine",
        "format",
        "value_ns",
        "all_runtimes_ns",
        "peak_physical",
        "peak_virtual",
        "physical_delta",
        "virtual_delta",
        "env_triple",
    ]
    baseline_row = (
        "query_measurement",
        "base-new",
        "tpch",
        None,
        "10",
        1,
        "nvme",
        "datafusion",
        "parquet",
        95,
        [85, 95, 105],
        None,
        None,
        None,
        None,
        "x86_64-linux-gnu",
    )
    conn = FakeConnection(
        [
            FakeCursor(["commit_sha"], [("base-new",)]),
            FakeCursor(columns, [baseline_row]),
        ]
    )

    commit_sha, records = fetch.fetch_baseline_records(conn, pr_records)

    assert commit_sha == "base-new"
    assert records == [dict(zip(columns, baseline_row, strict=True))]
    assert len(conn.calls) == 2
    candidate_sql, candidate_params = conn.calls[0]
    assert "FROM query_measurements q" in candidate_sql
    assert "ORDER BY c.timestamp DESC, c.commit_sha DESC" in candidate_sql
    assert candidate_params == ("tpch", "10", "nvme")
    rows_sql, rows_params = conn.calls[1]
    assert "q.commit_sha = %s" in rows_sql
    assert rows_params == ("base-new", "tpch", "10", "nvme")


def test_compression_baseline_reads_times_and_sizes_from_same_commit() -> None:
    fetch = load_fetch_module()
    pr_records = [
        {
            "kind": "compression_time",
            "commit_sha": "pr-sha",
            "dataset": "taxi",
            "format": "vortex-file-compressed",
            "op": "encode",
            "value_ns": 200,
            "all_runtimes_ns": [200],
        },
        {
            "kind": "compression_size",
            "commit_sha": "pr-sha",
            "dataset": "taxi",
            "format": "vortex-file-compressed",
            "value_bytes": 400,
        },
    ]
    time_columns = [
        "kind",
        "commit_sha",
        "dataset",
        "dataset_variant",
        "format",
        "op",
        "value_ns",
        "all_runtimes_ns",
        "env_triple",
    ]
    size_columns = [
        "kind",
        "commit_sha",
        "dataset",
        "dataset_variant",
        "format",
        "value_bytes",
    ]
    time_row = ("compression_time", "base-sha", "taxi", None, "vortex-file-compressed", "encode", 180, [180], None)
    size_row = ("compression_size", "base-sha", "taxi", None, "vortex-file-compressed", 390)
    conn = FakeConnection(
        [
            FakeCursor(["commit_sha"], [("base-sha",)]),
            FakeCursor(time_columns, [time_row]),
            FakeCursor(size_columns, [size_row]),
        ]
    )

    commit_sha, records = fetch.fetch_baseline_records(conn, pr_records)

    assert commit_sha == "base-sha"
    assert records == [
        dict(zip(time_columns, time_row, strict=True)),
        dict(zip(size_columns, size_row, strict=True)),
    ]
    assert "FROM compression_times t" in conn.calls[1][0]
    assert "FROM compression_sizes s" in conn.calls[2][0]
    assert all(call[1][0] == "base-sha" for call in conn.calls[1:])


def test_mixed_benchmark_families_are_rejected() -> None:
    fetch = load_fetch_module()

    with pytest.raises(ValueError, match="multiple benchmark families"):
        fetch.benchmark_family(
            [
                query_record("pr-sha", 1, "datafusion", "parquet"),
                {
                    "kind": "random_access_time",
                    "commit_sha": "pr-sha",
                    "dataset": "taxi",
                    "format": "parquet",
                    "value_ns": 10,
                    "all_runtimes_ns": [10],
                },
            ]
        )


def test_missing_baseline_is_reported() -> None:
    fetch = load_fetch_module()
    conn = FakeConnection([FakeCursor(["commit_sha"], [])])

    with pytest.raises(ValueError, match="No RDS baseline"):
        fetch.fetch_baseline_records(
            conn,
            [query_record("pr-sha", 1, "datafusion", "parquet")],
        )
