# SpatialBench benchmark

The [Apache Sedona SpatialBench](https://sedona.apache.org/spatialbench/) spatial
analytics benchmark has twelve queries over a trips/zones schema. The queries exercise spatial
predicates and functions such as `ST_DWithin`, `ST_Intersects`, and `ST_Distance`.

[`spatialbench/duckdb.sql`](./spatialbench/duckdb.sql) contains the DuckDB dialect.
[`spatialbench/datafusion.sql`](./spatialbench/datafusion.sql) contains the equivalent DataFusion
dialect. The query logic matches upstream `sedona-spatialbench`.

Engine dialects use the `sql/<benchmark>/<engine>.sql` path.
The harness selects the matching file automatically.

The harness lives in [`src/spatialbench`](../src/spatialbench).

## Local use

```bash
vx-bench run spatialbench
```

The default command compares Parquet and Vortex with DataFusion and DuckDB. To run the native
Vortex spatial representation explicitly:

```bash
vx-bench run spatialbench --engine duckdb --format vortex-spatial-native
```

To compare all three representations in one run:

```bash
vx-bench run spatialbench --engine duckdb --format parquet,vortex,vortex-spatial-native
```

To compare DataFusion over Parquet and Vortex:

```bash
vx-bench run spatialbench --engine datafusion --format parquet,vortex
```
