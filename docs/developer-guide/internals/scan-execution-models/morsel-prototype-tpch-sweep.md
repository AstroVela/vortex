# Morsel Prototype: Scaling Sweep Output

Raw output of
`TPCH_SWEEP=1 cargo run --release -p vortex-morsel --features _test-harness --bin tpch-eval -- 1`.

Four sweeps: driving threads against physical cores (including oversubscription), V1's
concurrent-unit count, morsel size with morsel counts, and a single-morsel run that isolates
scheduling overhead from kernel work. Analysis in
[`morsel-prototype-tpch-findings.md`](morsel-prototype-tpch-findings.md).


lineitem SF=1: 6001215 rows (6001215 generated), 16 columns,          733 natural splits; generated in 3640.354ms, written in 5731.767ms
written through the btrblocks compressing pipeline (repartition 8192 rows -> coalesce 1048576B -> compress -> buffer -> chunk -> flat); no zone maps, no dict layout
host: 4 logical cores; segments in memory; 5 alternating iterations, median reported

schema: {l_orderkey=i64, l_partkey=i64, l_suppkey=i64, l_linenumber=i32, l_quantity=decimal(15,2), l_extendedprice=decimal(15,2), l_discount=decimal(15,2), l_tax=decimal(15,2), l_returnflag=utf8, l_linestatus=utf8, l_shipdate=vortex.date[days](i32), l_commitdate=vortex.date[days](i32), l_receiptdate=vortex.date[days](i32), l_shipinstruct=utf8, l_shipmode=utf8, l_comment=utf8}

## Driving threads vs cores (4 physical cores, 1 thread per core)

Morsel driver: one morsel in flight per thread. `x4` is one thread per physical core; beyond that the host is oversubscribed.

| query | D x1 | D x2 | D x4 | D x8 | D x16 | best | vs D x4 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q6 | 38.438ms | 20.209ms | 10.809ms | 11.676ms | 12.467ms | x4 | 1.00x |
| Q1 | 11.958ms | 6.723ms | 3.952ms | 4.358ms | 4.764ms | x4 | 1.00x |
| Q14 | 14.459ms | 7.897ms | 4.418ms | 4.948ms | 5.530ms | x4 | 1.00x |
| Q15 | 14.514ms | 7.740ms | 4.665ms | 4.882ms | 5.531ms | x4 | 1.00x |
| Q12 | 34.169ms | 19.173ms | 10.030ms | 10.607ms | 11.679ms | x4 | 1.00x |
| Q19 | 23.613ms | 20.093ms | 11.435ms | 10.903ms | 11.004ms | x8 | 0.95x |
| scan-6col | 1.979ms | 1.673ms | 1.196ms | 1.529ms | 2.368ms | x4 | 1.00x |
| selective | 18.671ms | 10.704ms | 5.659ms | 5.959ms | 7.134ms | x4 | 1.00x |

## V1 concurrent units: 4 workers x per-worker split concurrency

V1's parallelism is workers x concurrency. This sweeps the second factor to check the baseline is not simply mis-tuned.

| query | V1 x1 | tok4 c=1 | tok4 c=2 | tok4 c=4 | tok4 c=8 | tok4 c=16 | best |
|---|--:|--:|--:|--:|--:|--:|--:|
| Q6 | 42.788ms | 18.316ms | 15.701ms | 14.052ms | 14.474ms | 13.078ms | 13.078ms |
| Q1 | 14.174ms | 8.583ms | 7.596ms | 6.613ms | 6.586ms | 6.268ms | 6.268ms |
| Q14 | 16.157ms | 7.817ms | 6.932ms | 6.497ms | 5.908ms | 5.761ms | 5.761ms |
| Q15 | 16.026ms | 8.570ms | 7.027ms | 6.157ms | 6.061ms | 6.255ms | 6.061ms |
| Q12 | 38.416ms | 18.331ms | 14.973ms | 13.823ms | 13.001ms | 13.013ms | 13.001ms |
| Q19 | 45.641ms | 29.090ms | 25.441ms | 23.286ms | 23.180ms | 22.534ms | 22.534ms |
| scan-6col | 4.040ms | 4.593ms | 3.862ms | 3.591ms | 3.449ms | 3.699ms | 3.449ms |
| selective | 21.580ms | 10.381ms | 9.056ms | 8.175ms | 8.229ms | 7.610ms | 7.610ms |

## Morsel size at 4 threads: is there enough work to fill the cores?

`n` is the morsel count, `n/core` is morsels per driving thread. Below roughly 4 morsels per core the tail of the last morsel is a large fraction of the run and load balance falls apart, whatever the per-morsel overhead is.

| query | splits | 16k | 32k | 64k | 128k | 256k | 1M | single |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Q6 | 13.907ms n=92 (23.0/core) | 18.780ms n=367 (91.8/core) | 13.866ms n=184 (46.0/core) | 10.603ms n=92 (23.0/core) | 11.114ms n=46 (11.5/core) | 10.851ms n=23 (5.8/core) | 13.538ms n=6 (1.5/core) | 38.582ms n=1 (0.2/core) |
| Q1 | 3.786ms n=92 (23.0/core) | 10.958ms n=367 (91.8/core) | 7.950ms n=184 (46.0/core) | 4.027ms n=92 (23.0/core) | 3.758ms n=46 (11.5/core) | 3.714ms n=23 (5.8/core) | 4.363ms n=6 (1.5/core) | 11.283ms n=1 (0.2/core) |
| Q14 | 4.511ms n=92 (23.0/core) | 8.206ms n=367 (91.8/core) | 5.897ms n=184 (46.0/core) | 4.303ms n=92 (23.0/core) | 4.443ms n=46 (11.5/core) | 4.403ms n=23 (5.8/core) | 5.463ms n=6 (1.5/core) | 15.378ms n=1 (0.2/core) |
| Q15 | 4.357ms n=92 (23.0/core) | 8.148ms n=367 (91.8/core) | 5.891ms n=184 (46.0/core) | 4.273ms n=92 (23.0/core) | 4.721ms n=46 (11.5/core) | 4.280ms n=23 (5.8/core) | 5.523ms n=6 (1.5/core) | 15.151ms n=1 (0.2/core) |
| Q12 | 9.615ms n=92 (23.0/core) | 18.898ms n=367 (91.8/core) | 13.087ms n=184 (46.0/core) | 9.471ms n=92 (23.0/core) | 12.394ms n=46 (11.5/core) | 13.328ms n=23 (5.8/core) | 18.707ms n=6 (1.5/core) | 142.082ms n=1 (0.2/core) |
| Q19 | 11.577ms n=366 (91.5/core) | 11.377ms n=367 (91.8/core) | 7.890ms n=184 (46.0/core) | 4.579ms n=92 (23.0/core) | 4.551ms n=46 (11.5/core) | 4.425ms n=23 (5.8/core) | 5.490ms n=6 (1.5/core) | 14.958ms n=1 (0.2/core) |
| scan-6col | 1.388ms n=92 (23.0/core) | 5.439ms n=367 (91.8/core) | 3.394ms n=184 (46.0/core) | 1.310ms n=92 (23.0/core) | 1.115ms n=46 (11.5/core) | 1.009ms n=23 (5.8/core) | 1.065ms n=6 (1.5/core) | 1.981ms n=1 (0.2/core) |
| selective | 5.992ms n=92 (23.0/core) | 12.101ms n=367 (91.8/core) | 8.349ms n=184 (46.0/core) | 5.882ms n=92 (23.0/core) | 5.691ms n=46 (11.5/core) | 5.587ms n=23 (5.8/core) | 6.702ms n=6 (1.5/core) | 18.974ms n=1 (0.2/core) |

## Where the single-threaded time goes

`1 morsel` drives the whole scan as one unit: no per-morsel reset, planning, cutting or emission, and no cross-morsel sharing to do. Whatever it costs is decode plus predicate and gather kernels — work identical to V1's. The gap between it and the per-split column is the executor's entire scheduling overhead.

| query | V1 x1 | D x1 per-split | D x1 one morsel | scheduling overhead | kernel-bound share |
|---|--:|--:|--:|--:|--:|
| Q6 | 40.798ms | 37.758ms | 37.970ms | -0.6% | 93% |
| Q1 | 15.184ms | 12.334ms | 10.718ms | 13.1% | 71% |
| Q14 | 16.450ms | 14.493ms | 15.091ms | -4.1% | 92% |
| Q15 | 16.162ms | 14.231ms | 15.052ms | -5.8% | 93% |
| Q12 | 39.396ms | 33.758ms | 98.785ms | -192.6% | 251% |
| Q19 | 41.890ms | 23.429ms | 14.690ms | 37.3% | 35% |
| scan-6col | 3.814ms | 2.007ms | 1.674ms | 16.6% | 44% |
| selective | 21.378ms | 18.745ms | 19.429ms | -3.7% | 91% |

