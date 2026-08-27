# SpatialBench benchmark

The [Apache Sedona SpatialBench](https://sedona.apache.org/spatialbench/) benchmark has twelve spatial analytics queries over a trips/zones schema.

[`spatialbench/duckdb.sql`](./spatialbench/duckdb.sql) contains the DuckDB dialect.
The query logic matches upstream `sedona-spatialbench`.

Engine dialects use the `sql/<benchmark>/<engine>.sql` path.
The harness selects the matching file automatically.

The harness lives in [`src/spatialbench`](../src/spatialbench).

## Local use

```bash
vx-bench run spatialbench
```

The default command compares the Parquet and Vortex WKB representations with DuckDB.

Run the native Vortex spatial representation:

```bash
vx-bench run spatialbench --engine duckdb --format vortex-spatial-native
```

Compare all three representations:

```bash
vx-bench run spatialbench --engine duckdb --format parquet,vortex,vortex-spatial-native
```
