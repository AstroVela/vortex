# Morsel Prototype: P1 Evaluation Output

Raw output of `cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval`.
The analysis, and the list of what this run does *not* establish, is in
[`morsel-prototype-p1-findings.md`](morsel-prototype-p1-findings.md).


host: 4 logical cores; segments in memory; 1000000 rows per workload; 5 alternating iterations, median reported

## string-heavy — FineWeb-shaped: wide text plus scalars, five disagreeing chunkings

250000 rows, 62 natural splits

### SH1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 31.154ms | 1.00x | 250000 | 9.155ms | — | — | — | — | — |
| A' V1 (tokio x4) | 9.591ms | 0.31x | 250000 | 1.955ms | — | — | — | — | — |
| D  morsel (x1, splits) | 15.166ms | 0.49x | 250000 | 0.525ms | 62 | 310 | 121 | 121 | 189 |
| D  morsel (x1, splits, no cache) | 31.243ms | 1.00x | 250000 | 0.508ms | 62 | 310 | 121 | 310 | 0 |
| D  morsel (x4, splits) | 8.226ms | 0.26x | 250000 | 0.743ms | 62 | 310 | 209 | 209 | 101 |
| D  morsel (x4, 65536r) | 5.401ms | 0.17x | 250000 | 3.558ms | 4 | 121 | 121 | 121 | 0 |
| D  morsel (x4, splits, parallel) | 6.560ms | 0.21x | 250000 | 0.702ms | 62 | 310 | 229 | 229 | 81 |

### SH2 lowcard-eq

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 10.897ms | 1.00x | 31301 | 5.843ms | — | — | — | — | — |
| A' V1 (tokio x4) | 4.490ms | 0.41x | 31301 | 2.474ms | — | — | — | — | — |
| D  morsel (x1, splits) | 5.796ms | 0.53x | 31301 | 0.407ms | 31 | 93 | 55 | 55 | 38 |
| D  morsel (x1, splits, no cache) | 10.218ms | 0.94x | 31301 | 0.322ms | 31 | 93 | 55 | 93 | 0 |
| D  morsel (x4, splits) | 3.549ms | 0.33x | 31301 | 0.549ms | 31 | 93 | 88 | 88 | 5 |
| D  morsel (x4, 65536r) | 2.646ms | 0.24x | 31301 | 1.418ms | 4 | 55 | 55 | 55 | 0 |
| D  morsel (x4, splits, parallel) | 3.367ms | 0.31x | 31301 | 0.507ms | 31 | 93 | 88 | 88 | 5 |

### SH3 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 34.866ms | 1.00x | 2496 | 9.023ms | — | — | — | — | — |
| A' V1 (tokio x4) | 10.909ms | 0.31x | 2496 | 3.120ms | — | — | — | — | — |
| D  morsel (x1, splits) | 17.334ms | 0.50x | 2496 | 0.657ms | 62 | 248 | 117 | 116 | 130 |
| D  morsel (x1, splits, no cache) | 33.280ms | 0.95x | 2496 | 0.529ms | 62 | 248 | 117 | 246 | 0 |
| D  morsel (x4, splits) | 9.503ms | 0.27x | 2496 | 0.895ms | 62 | 248 | 197 | 195 | 51 |
| D  morsel (x4, 65536r) | 7.287ms | 0.21x | 2496 | 4.002ms | 4 | 117 | 117 | 117 | 0 |
| D  morsel (x4, splits, parallel) | 8.760ms | 0.25x | 2496 | 0.699ms | 62 | 248 | 197 | 195 | 51 |

### SH4 selective

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 10.097ms | 1.00x | 40 | 2.249ms | — | — | — | — | — |
| A' V1 (tokio x4) | 4.597ms | 0.46x | 40 | 1.891ms | — | — | — | — | — |
| D  morsel (x1, splits) | 8.387ms | 0.83x | 40 | 0.423ms | 62 | 310 | 113 | 69 | 139 |
| D  morsel (x1, splits, no cache) | 9.568ms | 0.95x | 40 | 0.477ms | 62 | 310 | 113 | 208 | 0 |
| D  morsel (x4, splits) | 2.953ms | 0.29x | 40 | 0.615ms | 62 | 310 | 173 | 113 | 95 |
| D  morsel (x4, 65536r) | 4.429ms | 0.44x | 40 | 3.871ms | 4 | 117 | 113 | 113 | 4 |
| D  morsel (x4, splits, parallel) | 2.938ms | 0.29x | 40 | 0.565ms | 62 | 310 | 176 | 115 | 93 |

### SH5 empty

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.891ms | 1.00x | 0 | — | — | — | — | — | — |
| A' V1 (tokio x4) | 1.620ms | 1.82x | 0 | — | — | — | — | — | — |
| D  morsel (x1, splits) | 0.622ms | 0.70x | 0 | — | 62 | 186 | 97 | 4 | 58 |
| D  morsel (x1, splits, no cache) | 0.711ms | 0.80x | 0 | — | 62 | 186 | 97 | 62 | 0 |
| D  morsel (x4, splits) | 0.721ms | 0.81x | 0 | — | 62 | 186 | 136 | 15 | 47 |
| D  morsel (x4, 65536r) | 0.325ms | 0.37x | 0 | — | 4 | 97 | 97 | 4 | 0 |
| D  morsel (x4, splits, parallel) | 0.529ms | 0.59x | 0 | — | 62 | 186 | 136 | 14 | 48 |

### SH6 narrow-project

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.539ms | 1.00x | 125382 | 0.529ms | — | — | — | — | — |
| A' V1 (tokio x4) | 0.680ms | 1.26x | 125382 | 0.655ms | — | — | — | — | — |
| D  morsel (x1, splits) | 0.404ms | 0.75x | 125382 | 0.076ms | 16 | 32 | 20 | 20 | 12 |
| D  morsel (x1, splits, no cache) | 0.393ms | 0.73x | 125382 | 0.025ms | 16 | 32 | 20 | 32 | 0 |
| D  morsel (x4, splits) | 0.498ms | 0.92x | 125382 | 0.255ms | 16 | 32 | 30 | 30 | 2 |
| D  morsel (x4, 65536r) | 0.354ms | 0.66x | 125382 | 0.235ms | 4 | 20 | 20 | 20 | 0 |
| D  morsel (x4, splits, parallel) | 0.385ms | 0.71x | 125382 | 0.139ms | 16 | 32 | 29 | 29 | 3 |

## wide-numeric — ClickBench-shaped: 20 narrow integer columns, five disagreeing chunkings

1000000 rows, 228 natural splits

### WN1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 22.432ms | 1.00x | 1000000 | 6.218ms | — | — | — | — | — |
| A' V1 (tokio x4) | 25.715ms | 1.15x | 1000000 | 4.511ms | — | — | — | — | — |
| D  morsel (x1, splits) | 9.465ms | 0.42x | 1000000 | 0.436ms | 228 | 4560 | 1332 | 1332 | 3228 |
| D  morsel (x1, splits, no cache) | 10.512ms | 0.47x | 1000000 | 0.163ms | 228 | 4560 | 1332 | 4560 | 0 |
| D  morsel (x4, splits) | 6.757ms | 0.30x | 1000000 | 0.989ms | 228 | 4560 | 3908 | 3908 | 652 |
| D  morsel (x4, 65536r) | 2.644ms | 0.12x | 1000000 | 0.916ms | 16 | 1476 | 1476 | 1476 | 0 |
| D  morsel (x4, splits, parallel) | 6.468ms | 0.29x | 1000000 | 0.443ms | 228 | 4560 | 3848 | 3848 | 712 |

### WN2 point-filter

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 2.561ms | 1.00x | 2 | 1.331ms | — | — | — | — | — |
| A' V1 (tokio x4) | 3.128ms | 1.22x | 2 | 1.745ms | — | — | — | — | — |
| D  morsel (x1, splits) | 1.701ms | 0.66x | 2 | 0.657ms | 147 | 588 | 193 | 53 | 100 |
| D  morsel (x1, splits, no cache) | 1.684ms | 0.66x | 2 | 0.620ms | 147 | 588 | 193 | 153 | 0 |
| D  morsel (x4, splits) | 1.282ms | 0.50x | 2 | 0.531ms | 147 | 588 | 403 | 133 | 20 |
| D  morsel (x4, 65536r) | 0.849ms | 0.33x | 2 | 0.453ms | 16 | 276 | 215 | 81 | 8 |
| D  morsel (x4, splits, parallel) | 1.087ms | 0.42x | 2 | 0.432ms | 147 | 588 | 425 | 142 | 11 |

### WN3 dashboard

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 11.811ms | 1.00x | 874895 | 1.891ms | — | — | — | — | — |
| A' V1 (tokio x4) | 8.711ms | 0.74x | 874895 | 1.817ms | — | — | — | — | — |
| D  morsel (x1, splits) | 5.060ms | 0.43x | 874895 | 0.172ms | 204 | 1428 | 455 | 455 | 973 |
| D  morsel (x1, splits, no cache) | 5.899ms | 0.50x | 874895 | 0.096ms | 204 | 1428 | 455 | 1428 | 0 |
| D  morsel (x4, splits) | 3.056ms | 0.26x | 874895 | 0.389ms | 204 | 1428 | 1115 | 1115 | 313 |
| D  morsel (x4, 65536r) | 1.827ms | 0.15x | 874895 | 0.492ms | 16 | 555 | 493 | 493 | 62 |
| D  morsel (x4, splits, parallel) | 2.905ms | 0.25x | 874895 | 0.403ms | 204 | 1428 | 1108 | 1108 | 320 |

### WN4 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 38.300ms | 1.00x | 15441 | 8.849ms | — | — | — | — | — |
| A' V1 (tokio x4) | 27.836ms | 0.73x | 15441 | 6.649ms | — | — | — | — | — |
| D  morsel (x1, splits) | 16.052ms | 0.42x | 15441 | 0.595ms | 228 | 5016 | 1332 | 1332 | 3684 |
| D  morsel (x1, splits, no cache) | 19.873ms | 0.52x | 15441 | 0.350ms | 228 | 5016 | 1332 | 5016 | 0 |
| D  morsel (x4, splits) | 10.236ms | 0.27x | 15441 | 1.207ms | 228 | 5016 | 3476 | 3476 | 1540 |
| D  morsel (x4, 65536r) | 4.250ms | 0.11x | 15441 | 1.453ms | 16 | 1569 | 1476 | 1476 | 93 |
| D  morsel (x4, splits, parallel) | 9.033ms | 0.24x | 15441 | 0.526ms | 228 | 5016 | 3908 | 3908 | 1108 |

### WN5 selective-wide

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 11.363ms | 1.00x | 10 | 3.323ms | — | — | — | — | — |
| A' V1 (tokio x4) | 13.310ms | 1.17x | 10 | 3.466ms | — | — | — | — | — |
| D  morsel (x1, splits) | 5.860ms | 0.52x | 10 | 0.328ms | 228 | 5016 | 1332 | 311 | 334 |
| D  morsel (x1, splits, no cache) | 5.212ms | 0.46x | 10 | 0.252ms | 228 | 5016 | 1332 | 645 | 0 |
| D  morsel (x4, splits) | 3.683ms | 0.32x | 10 | 0.651ms | 228 | 5016 | 3744 | 562 | 83 |
| D  morsel (x4, 65536r) | 2.193ms | 0.19x | 10 | 0.812ms | 16 | 1629 | 1476 | 755 | 70 |
| D  morsel (x4, splits, parallel) | 4.026ms | 0.35x | 10 | 0.682ms | 228 | 5016 | 3264 | 521 | 135 |

### WN6 packed

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 4.230ms | 1.00x | 250021 | 0.791ms | — | — | — | — | — |
| A' V1 (tokio x4) | 4.250ms | 1.00x | 250021 | 0.930ms | — | — | — | — | — |
| D  morsel (x1, splits) | 2.612ms | 0.62x | 250021 | 0.074ms | 147 | 441 | 193 | 193 | 248 |
| D  morsel (x1, splits, no cache) | 2.646ms | 0.63x | 250021 | 0.057ms | 147 | 441 | 193 | 441 | 0 |
| D  morsel (x4, splits) | 1.580ms | 0.37x | 250021 | 0.278ms | 147 | 441 | 402 | 402 | 39 |
| D  morsel (x4, 65536r) | 1.008ms | 0.24x | 250021 | 0.293ms | 16 | 215 | 215 | 215 | 0 |
| D  morsel (x4, splits, parallel) | 1.565ms | 0.37x | 250021 | 0.208ms | 147 | 441 | 435 | 435 | 6 |

## narrow-analytic — TPC-H Q6/Q1-shaped: conjunctive range filter, narrow projection

1000000 rows, 49 natural splits

### NA1 q6-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 5.393ms | 1.00x | 30093 | 1.937ms | — | — | — | — | — |
| A' V1 (tokio x4) | 3.202ms | 0.59x | 30093 | 1.384ms | — | — | — | — | — |
| D  morsel (x1, splits) | 4.359ms | 0.81x | 30093 | 0.216ms | 49 | 294 | 78 | 78 | 216 |
| D  morsel (x1, splits, no cache) | 4.231ms | 0.78x | 30093 | 0.135ms | 49 | 294 | 78 | 294 | 0 |
| D  morsel (x4, splits) | 1.990ms | 0.37x | 30093 | 0.355ms | 49 | 294 | 185 | 185 | 109 |
| D  morsel (x4, 65536r) | 1.710ms | 0.32x | 30093 | 0.407ms | 16 | 168 | 100 | 100 | 68 |
| D  morsel (x4, splits, parallel) | 1.882ms | 0.35x | 30093 | 0.273ms | 49 | 294 | 180 | 180 | 114 |

### NA2 q1-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 2.422ms | 1.00x | 857083 | 1.006ms | — | — | — | — | — |
| A' V1 (tokio x4) | 2.015ms | 0.83x | 857083 | 0.906ms | — | — | — | — | — |
| D  morsel (x1, splits) | 1.446ms | 0.60x | 857083 | 0.078ms | 49 | 196 | 78 | 78 | 118 |
| D  morsel (x1, splits, no cache) | 1.474ms | 0.61x | 857083 | 0.048ms | 49 | 196 | 78 | 196 | 0 |
| D  morsel (x4, splits) | 1.108ms | 0.46x | 857083 | 0.229ms | 49 | 196 | 166 | 166 | 30 |
| D  morsel (x4, 65536r) | 0.883ms | 0.36x | 857083 | 0.273ms | 16 | 100 | 100 | 100 | 0 |
| D  morsel (x4, splits, parallel) | 0.991ms | 0.41x | 857083 | 0.238ms | 49 | 196 | 175 | 175 | 21 |

### NA3 scan-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | cache hits |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 1.212ms | 1.00x | 1000000 | 0.632ms | — | — | — | — | — |
| A' V1 (tokio x4) | 1.867ms | 1.54x | 1000000 | 0.621ms | — | — | — | — | — |
| D  morsel (x1, splits) | 0.370ms | 0.31x | 1000000 | 0.038ms | 49 | 196 | 78 | 78 | 118 |
| D  morsel (x1, splits, no cache) | 0.405ms | 0.33x | 1000000 | 0.012ms | 49 | 196 | 78 | 196 | 0 |
| D  morsel (x4, splits) | 0.792ms | 0.65x | 1000000 | 0.202ms | 49 | 196 | 186 | 186 | 10 |
| D  morsel (x4, 65536r) | 0.377ms | 0.31x | 1000000 | 0.167ms | 16 | 100 | 100 | 100 | 0 |
| D  morsel (x4, splits, parallel) | 0.622ms | 0.51x | 1000000 | 0.107ms | 49 | 196 | 173 | 173 | 23 |

All configurations matched the V1 oracle.
