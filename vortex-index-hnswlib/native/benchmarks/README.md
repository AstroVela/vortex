# Filtered search microbenchmark

```sh
c++ -std=c++17 -O3 -mavx2 -mfma -mf16c -pthread -I "$VORTEX_HNSWLIB_SOURCE" \
  vortex-index-hnswlib/native/benchmarks/filterable.cpp -o /tmp/filterable-bench
/tmp/filterable-bench > results.csv
```

The fixture uses 10,000 16-dimensional Gaussian vectors, 100 equality groups,
80 queries, k=10, base M=8, ef_construction=100, payload_m=8 and search ef=32.
One workload assigns attributes independently of vector distances; the second
shifts the first vector component according to the attribute. It measures
upstream callback filtering and payload filtering on the same base graph at
1%, 2%, 10%, 50% and 100% selectivity. Brute-force filtered truth is computed
outside the timed region. Measurement order alternates to reduce cache bias.
All measured queries use an explicit filter, including the 100% case; unrestricted
provider queries retain the upstream unfiltered path.

[Sample results](sample.csv), measured on 2026-10-29 with GCC 13 on an Intel Xeon
E5-2687W v4 @ 3.00GHz, show **higher latency for payload search** at the same ef.
For independent attributes at 50% selectivity, recall@10 increased from 0.950 to
0.965 while p50 increased from 44.678 to 84.106 microseconds. At 1%, both attained
1.0 recall with p50 649.071 versus 876.229 microseconds. The correlated workload
attained recall 1.0 for both methods at every measured selectivity, with added
latency for the payload graph.

The independent/correlated overlays contained 82,086/31,236 directed edges and
408,352/204,952 CSR bytes. Base construction took 0.650/0.333 seconds and payload
construction added 0.122/0.106 seconds in this run. These are one-run observations
on a shared development host, not reproducible latency thresholds or production
performance qualification.

The initial algorithm preserves hnswlib's rejected-node navigation semantics.
Supplementary edges can improve low-ef recall but also add distance evaluations
and heap work. Further work must compare latency at matched recall and include
skew, ranges, compound predicates, larger data, RSS and representative workloads.
There is no automatic scan path or ACORN expansion in this benchmark or backend.
