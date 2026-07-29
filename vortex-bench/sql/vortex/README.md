# Vortex queries benchmark

A small suite of microbenchmark queries targeting Vortex-specific scan paths, run on
**every PR commit** (unlike the label-gated suites) with a high iteration count, so it is
the most sensitive — and most frequently seen — benchmark comment.

[`init.sql`](./init.sql) generates a 25M-row two-column table; the numbered queries each
pin down one scan behavior.

## CI variant

CI runs this as the `Vortex queries` PR comment (see
[`.github/workflows/sql-vortex-pr.yml`](../../../.github/workflows/sql-vortex-pr.yml)),
comparing DataFusion and DuckDB over Parquet and Vortex files with 100 iterations.

## Verification

Golden files for benchmarks live in `results/duckdb/{query_idx}.csv`. If you're
adding a new query, generate its golden file with the following command,
replacing `query_idx` with next number:

```sql
CREATE OR REPLACE VIEW test AS SELECT (...);
COPY (SELECT sum(col) FROM test) TO 'results/duckdb/{query_idx}.csv'
(HEADER, DELIMITER '|');
```

## Running locally

```bash
vx-bench run vortex --engine datafusion,duckdb --format parquet,vortex
```
