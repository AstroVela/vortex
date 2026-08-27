# Morsel Prototype: TPC-H Evaluation Output

Raw output of
`cargo run --release -p vortex-morsel --features _test-harness --bin tpch-eval -- 1`.
Analysis in [`morsel-prototype-tpch-findings.md`](morsel-prototype-tpch-findings.md).


lineitem SF=1: 6001215 rows (6001215 generated), 16 columns,          733 natural splits; generated in 3583.409ms, written in 5617.826ms
written through the btrblocks compressing pipeline (repartition 8192 rows -> coalesce 1048576B -> compress -> buffer -> chunk -> flat); no zone maps, no dict layout
host: 4 logical cores; segments in memory; 5 alternating iterations, median reported

schema: {l_orderkey=i64, l_partkey=i64, l_suppkey=i64, l_linenumber=i32, l_quantity=decimal(15,2), l_extendedprice=decimal(15,2), l_discount=decimal(15,2), l_tax=decimal(15,2), l_returnflag=utf8, l_linestatus=utf8, l_shipdate=vortex.date[days](i32), l_commitdate=vortex.date[days](i32), l_receiptdate=vortex.date[days](i32), l_shipinstruct=utf8, l_shipmode=utf8, l_comment=utf8}

### Q6 — 114160 rows out (1.90% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 45.163ms | 1.00x | 9.268ms | — | — | — | — |
| A' V1 (tokio x4) | 14.499ms | 0.32x | 1.569ms | — | — | — | — |
| D  morsel (x1, 128k) | 41.557ms | 0.92x | 1.050ms | 46 | 368 | 368 | 276 |
| D  morsel (x1, 128k, no-reuse) | 41.720ms | 0.92x | 1.071ms | 46 | 368 | 644 | 0 |
| D  morsel (x4, 128k) | 12.565ms | 0.28x | 1.667ms | 46 | 368 | 368 | 276 |
| D  morsel (x4, splits) | 11.692ms | 0.26x | 0.763ms | 92 | 368 | 368 | 276 |

### Q1 — 5916591 rows out (98.59% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 16.172ms | 1.00x | 3.988ms | — | — | — | — |
| A' V1 (tokio x4) | 7.504ms | 0.46x | 2.279ms | — | — | — | — |
| D  morsel (x1, 128k) | 13.440ms | 0.83x | 0.444ms | 46 | 644 | 644 | 0 |
| D  morsel (x1, 128k, no-reuse) | 12.987ms | 0.80x | 0.461ms | 46 | 644 | 644 | 0 |
| D  morsel (x4, 128k) | 5.042ms | 0.31x | 0.814ms | 46 | 644 | 644 | 0 |
| D  morsel (x4, splits) | 4.759ms | 0.29x | 0.597ms | 92 | 644 | 644 | 0 |

### Q14 — 75983 rows out (1.27% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 16.717ms | 1.00x | 3.548ms | — | — | — | — |
| A' V1 (tokio x4) | 7.170ms | 0.43x | 1.920ms | — | — | — | — |
| D  morsel (x1, 128k) | 14.850ms | 0.89x | 0.588ms | 46 | 368 | 368 | 92 |
| D  morsel (x1, 128k, no-reuse) | 15.214ms | 0.91x | 0.390ms | 46 | 368 | 460 | 0 |
| D  morsel (x4, 128k) | 6.147ms | 0.37x | 0.861ms | 46 | 368 | 368 | 92 |
| D  morsel (x4, splits) | 5.595ms | 0.33x | 0.468ms | 92 | 368 | 368 | 92 |

### Q15 — 225954 rows out (3.77% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 16.209ms | 1.00x | 3.469ms | — | — | — | — |
| A' V1 (tokio x4) | 7.257ms | 0.45x | 1.761ms | — | — | — | — |
| D  morsel (x1, 128k) | 14.477ms | 0.89x | 0.536ms | 46 | 368 | 368 | 92 |
| D  morsel (x1, 128k, no-reuse) | 14.676ms | 0.91x | 0.412ms | 46 | 368 | 460 | 0 |
| D  morsel (x4, 128k) | 5.432ms | 0.34x | 0.799ms | 46 | 368 | 368 | 92 |
| D  morsel (x4, splits) | 4.429ms | 0.27x | 0.416ms | 92 | 368 | 368 | 92 |

### Q12 — 108434 rows out (1.81% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 40.280ms | 1.00x | 7.885ms | — | — | — | — |
| A' V1 (tokio x4) | 14.849ms | 0.37x | 1.838ms | — | — | — | — |
| D  morsel (x1, 128k) | 46.894ms | 1.16x | 1.365ms | 46 | 460 | 460 | 460 |
| D  morsel (x1, 128k, no-reuse) | 47.270ms | 1.17x | 1.216ms | 46 | 460 | 920 | 0 |
| D  morsel (x4, 128k) | 13.959ms | 0.35x | 1.840ms | 46 | 460 | 460 | 460 |
| D  morsel (x4, splits) | 10.217ms | 0.25x | 0.829ms | 92 | 460 | 460 | 460 |

### Q19 — 3599028 rows out (59.97% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 45.608ms | 1.00x | 4.729ms | — | — | — | — |
| A' V1 (tokio x4) | 27.261ms | 0.60x | 4.292ms | — | — | — | — |
| D  morsel (x1, 128k) | 15.721ms | 0.34x | 0.762ms | 46 | 826 | 826 | 184 |
| D  morsel (x1, 128k, no-reuse) | 15.774ms | 0.35x | 0.570ms | 46 | 826 | 1010 | 0 |
| D  morsel (x4, 128k) | 5.553ms | 0.12x | 1.118ms | 46 | 826 | 826 | 184 |
| D  morsel (x4, splits) | 12.173ms | 0.27x | 0.491ms | 366 | 1893 | 1072 | 1856 |

### scan-6col — 6001215 rows out (100.00% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 4.403ms | 1.00x | 1.512ms | — | — | — | — |
| A' V1 (tokio x4) | 4.232ms | 0.96x | 1.187ms | — | — | — | — |
| D  morsel (x1, 128k) | 2.364ms | 0.54x | 0.156ms | 46 | 552 | 552 | 0 |
| D  morsel (x1, 128k, no-reuse) | 2.062ms | 0.47x | 0.064ms | 46 | 552 | 552 | 0 |
| D  morsel (x4, 128k) | 1.595ms | 0.36x | 0.397ms | 46 | 552 | 552 | 0 |
| D  morsel (x4, splits) | 1.437ms | 0.33x | 0.280ms | 92 | 552 | 552 | 0 |

### selective — 260 rows out (0.00% selectivity)

| executor | wall | vs V1 | ttfb | morsels | reqs | decodes | reuses |
|---|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 21.903ms | 1.00x | 4.375ms | — | — | — | — |
| A' V1 (tokio x4) | 8.937ms | 0.41x | 2.255ms | — | — | — | — |
| D  morsel (x1, 128k) | 20.291ms | 0.93x | 0.695ms | 46 | 460 | 460 | 92 |
| D  morsel (x1, 128k, no-reuse) | 20.060ms | 0.92x | 0.572ms | 46 | 460 | 552 | 0 |
| D  morsel (x4, 128k) | 7.467ms | 0.34x | 1.085ms | 46 | 460 | 460 | 92 |
| D  morsel (x4, splits) | 6.529ms | 0.30x | 0.599ms | 92 | 460 | 446 | 92 |

Every configuration reproduced V1's dtype, row count and ordered content exactly.
