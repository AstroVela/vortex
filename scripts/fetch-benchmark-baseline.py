#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "boto3",
#   "psycopg[binary]",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Fetch the latest matching benchmark baseline from RDS as v3 JSONL."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_KIND_FAMILY = {
    "query_measurement": "query",
    "compression_time": "compression",
    "compression_size": "compression",
    "random_access_time": "random_access",
    "vector_search_run": "vector_search",
}

_FAMILY_SCOPE = {
    "query": ("dataset", "dataset_variant", "scale_factor", "storage"),
    "compression": ("dataset", "dataset_variant"),
    "random_access": ("dataset",),
    "vector_search": ("dataset", "layout", "threshold"),
}

_FAMILY_CANDIDATE = {
    "query": ("query_measurements", "q"),
    "compression": ("compression_times", "t"),
    "random_access": ("random_access_times", "r"),
    "vector_search": ("vector_search_runs", "v"),
}

_SELECTS = {
    "query": (
        (
            "query_measurements",
            "q",
            """
            SELECT 'query_measurement' AS kind,
                   q.commit_sha, q.dataset, q.dataset_variant, q.scale_factor,
                   q.query_idx, q.storage, q.engine, q.format,
                   q.value_ns, q.all_runtimes_ns,
                   q.peak_physical, q.peak_virtual,
                   q.physical_delta, q.virtual_delta, q.env_triple
              FROM query_measurements q
            """,
            ("dataset", "dataset_variant", "scale_factor", "storage"),
            ("q.query_idx", "q.engine", "q.format"),
        ),
    ),
    "compression": (
        (
            "compression_times",
            "t",
            """
            SELECT 'compression_time' AS kind,
                   t.commit_sha, t.dataset, t.dataset_variant, t.format, t.op,
                   t.value_ns, t.all_runtimes_ns, t.env_triple
              FROM compression_times t
            """,
            ("dataset", "dataset_variant"),
            ("t.dataset", "t.dataset_variant", "t.format", "t.op"),
        ),
        (
            "compression_sizes",
            "s",
            """
            SELECT 'compression_size' AS kind,
                   s.commit_sha, s.dataset, s.dataset_variant, s.format,
                   s.value_bytes
              FROM compression_sizes s
            """,
            ("dataset", "dataset_variant"),
            ("s.dataset", "s.dataset_variant", "s.format"),
        ),
    ),
    "random_access": (
        (
            "random_access_times",
            "r",
            """
            SELECT 'random_access_time' AS kind,
                   r.commit_sha, r.dataset, r.format,
                   r.value_ns, r.all_runtimes_ns, r.env_triple
              FROM random_access_times r
            """,
            ("dataset",),
            ("r.dataset", "r.format"),
        ),
    ),
    "vector_search": (
        (
            "vector_search_runs",
            "v",
            """
            SELECT 'vector_search_run' AS kind,
                   v.commit_sha, v.dataset, v.layout, v.flavor, v.threshold,
                   v.value_ns, v.all_runtimes_ns,
                   v.matches, v.rows_scanned, v.bytes_scanned,
                   v.iterations, v.env_triple
              FROM vector_search_runs v
            """,
            ("dataset", "layout", "threshold"),
            ("v.dataset", "v.layout", "v.threshold", "v.flavor"),
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_jsonl", type=Path, help="PR v3 ingest JSONL used to identify the benchmark scope.")
    parser.add_argument("--postgres", metavar="DSN", required=True, help="RDS Postgres libpq DSN.")
    parser.add_argument("--region", default=None, help="AWS region used to mint the RDS IAM token.")
    parser.add_argument("--output", type=Path, required=True, help="Destination v3 baseline JSONL.")
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    """Read non-empty JSON objects from a v3 JSONL file."""

    records = []
    with path.open(encoding="utf-8") as lines:
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"{path}: no benchmark records")
    return records


def benchmark_family(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the single fact-table family represented by ``records``."""

    unknown = sorted({record.get("kind") for record in records if record.get("kind") not in _KIND_FAMILY})
    if unknown:
        raise ValueError(f"unknown benchmark record kinds: {unknown}")

    families = {_KIND_FAMILY[str(record["kind"])] for record in records}
    if len(families) != 1:
        raise ValueError(f"PR records contain multiple benchmark families: {sorted(families)}")
    return next(iter(families))


def benchmark_scopes(records: Sequence[Mapping[str, Any]], family: str) -> list[tuple[Any, ...]]:
    """Return sorted, unique database scopes for a PR benchmark family."""

    columns = _FAMILY_SCOPE[family]
    scopes = {tuple(record.get(column) for column in columns) for record in records}
    return sorted(scopes, key=lambda scope: tuple("" if value is None else str(value) for value in scope))


def _scope_predicate(
    alias: str,
    columns: Sequence[str],
    scopes: Sequence[tuple[Any, ...]],
) -> tuple[str, tuple[Any, ...]]:
    clauses = []
    params: list[Any] = []
    for scope in scopes:
        terms = []
        for column, value in zip(columns, scope, strict=True):
            if value is None:
                terms.append(f"{alias}.{column} IS NULL")
            else:
                terms.append(f"{alias}.{column} = %s")
                params.append(value)
        clauses.append(f"({' AND '.join(terms)})")
    return f"({' OR '.join(clauses)})", tuple(params)


def _column_name(description: Any) -> str:
    if hasattr(description, "name"):
        return str(description.name)
    if isinstance(description, Sequence) and description:
        return str(description[0])
    raise TypeError(f"unsupported cursor description entry: {description!r}")


def _dict_rows(cursor: Any) -> list[dict]:
    rows = cursor.fetchall()
    if rows and isinstance(rows[0], Mapping):
        return [dict(row) for row in rows]
    columns = [_column_name(column) for column in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _candidate_commit(conn: Any, family: str, scopes: Sequence[tuple[Any, ...]]) -> str | None:
    table, alias = _FAMILY_CANDIDATE[family]
    predicate, params = _scope_predicate(alias, _FAMILY_SCOPE[family], scopes)
    cursor = conn.execute(
        f"""
        SELECT {alias}.commit_sha
          FROM {table} {alias}
          JOIN commits c USING (commit_sha)
         WHERE {predicate}
         ORDER BY c.timestamp DESC, c.commit_sha DESC
         LIMIT 1
        """,
        params,
    )
    row = cursor.fetchone()
    if row is None:
        return None
    if isinstance(row, Mapping):
        return str(row["commit_sha"])
    return str(row[0])


def fetch_baseline_records(
    conn: Any,
    pr_records: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict]]:
    """Fetch the newest matching commit and its scoped v3 fact rows."""

    family = benchmark_family(pr_records)
    scopes = benchmark_scopes(pr_records, family)
    commit_sha = _candidate_commit(conn, family, scopes)
    if commit_sha is None:
        raise ValueError(f"No RDS baseline found for {family} scopes {scopes!r}")

    records = []
    for _table, alias, select_sql, scope_columns, order_columns in _SELECTS[family]:
        predicate, scope_params = _scope_predicate(alias, scope_columns, scopes)
        order_by = ", ".join(order_columns)
        cursor = conn.execute(
            f"""
            {select_sql}
             WHERE {alias}.commit_sha = %s
               AND {predicate}
             ORDER BY {order_by}
            """,
            (commit_sha, *scope_params),
        )
        records.extend(_dict_rows(cursor))

    if not records:
        raise ValueError(f"RDS selected baseline commit {commit_sha}, but it contained no scoped records")
    return commit_sha, records


def write_records(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write v3 records as compact JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for record in records:
                output.write(json.dumps(record, separators=(",", ":")))
                output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _post_ingest_module():
    path = Path(__file__).resolve().with_name("post-ingest.py")
    spec = importlib.util.spec_from_file_location("post_ingest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    pr_records = read_records(args.pr_jsonl)
    post_ingest = _post_ingest_module()
    conn = post_ingest.connect_postgres(args.postgres, args.region)
    try:
        commit_sha, records = fetch_baseline_records(conn, pr_records)
    finally:
        conn.close()
    write_records(args.output, records)
    print(json.dumps({"commit_sha": commit_sha, "records": len(records)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
