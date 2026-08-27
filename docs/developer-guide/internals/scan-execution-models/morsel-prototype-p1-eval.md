# Morsel Prototype: P1 Evaluation Output

Raw output of `cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval`.
The analysis, and the list of what this run does *not* establish, is in
[`morsel-prototype-p1-findings.md`](morsel-prototype-p1-findings.md).


host: 4 logical cores; segments in memory; 1000000 rows per workload; 5 alternating iterations, median reported

## string-heavy — FineWeb-shaped: wide text plus scalars, five disagreeing chunkings

250000 rows, 62 natural splits

### SH1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 30.998ms | 1.00x | 250000 | 8.671ms | — | — | — | — |
| A' V1 (tokio x4) | 10.018ms | 0.32x | 250000 | 2.481ms | — | — | — | — |
| D  morsel (x1, splits) | 30.007ms | 0.97x | 250000 | 0.547ms | 62 | 310 | 310 | 310 |
| D  morsel (x4, splits) | 9.638ms | 0.31x | 250000 | 0.818ms | 62 | 310 | 310 | 310 |
| D  morsel (x4, 65536r) | 5.122ms | 0.17x | 250000 | 2.869ms | 4 | 121 | 121 | 121 |
| D  morsel (x4, splits, parallel) | 7.752ms | 0.25x | 250000 | 0.593ms | 62 | 310 | 310 | 310 |

### SH2 lowcard-eq

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 9.884ms | 1.00x | 31301 | 5.138ms | — | — | — | — |
| A' V1 (tokio x4) | 3.577ms | 0.36x | 31301 | 1.948ms | — | — | — | — |
| D  morsel (x1, splits) | 9.776ms | 0.99x | 31301 | 0.322ms | 31 | 93 | 93 | 93 |
| D  morsel (x4, splits) | 2.819ms | 0.29x | 31301 | 0.496ms | 31 | 93 | 93 | 93 |
| D  morsel (x4, 65536r) | 1.778ms | 0.18x | 31301 | 1.468ms | 4 | 55 | 55 | 55 |
| D  morsel (x4, splits, parallel) | 2.715ms | 0.27x | 31301 | 0.521ms | 31 | 93 | 93 | 93 |

### SH3 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 32.165ms | 1.00x | 2496 | 8.535ms | — | — | — | — |
| A' V1 (tokio x4) | 9.532ms | 0.30x | 2496 | 2.593ms | — | — | — | — |
| D  morsel (x1, splits) | 30.166ms | 0.94x | 2496 | 0.540ms | 62 | 248 | 248 | 246 |
| D  morsel (x4, splits) | 8.884ms | 0.28x | 2496 | 0.810ms | 62 | 248 | 248 | 246 |
| D  morsel (x4, 65536r) | 4.599ms | 0.14x | 2496 | 3.881ms | 4 | 117 | 117 | 117 |
| D  morsel (x4, splits, parallel) | 8.245ms | 0.26x | 2496 | 0.700ms | 62 | 248 | 248 | 246 |

### SH4 selective

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 9.306ms | 1.00x | 40 | 1.972ms | — | — | — | — |
| A' V1 (tokio x4) | 3.944ms | 0.42x | 40 | 1.387ms | — | — | — | — |
| D  morsel (x1, splits) | 7.469ms | 0.80x | 40 | 0.321ms | 62 | 310 | 248 | 208 |
| D  morsel (x4, splits) | 2.602ms | 0.28x | 40 | 0.471ms | 62 | 310 | 248 | 208 |
| D  morsel (x4, 65536r) | 4.057ms | 0.44x | 40 | 3.149ms | 4 | 117 | 113 | 117 |
| D  morsel (x4, splits, parallel) | 2.574ms | 0.28x | 40 | 0.456ms | 62 | 310 | 248 | 208 |

### SH5 empty

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.834ms | 1.00x | 0 | — | — | — | — | — |
| A' V1 (tokio x4) | 1.337ms | 1.60x | 0 | — | — | — | — | — |
| D  morsel (x1, splits) | 0.426ms | 0.51x | 0 | — | 62 | 186 | 186 | 62 |
| D  morsel (x4, splits) | 0.522ms | 0.63x | 0 | — | 62 | 186 | 186 | 62 |
| D  morsel (x4, 65536r) | 0.300ms | 0.36x | 0 | — | 4 | 97 | 97 | 4 |
| D  morsel (x4, splits, parallel) | 0.514ms | 0.62x | 0 | — | 62 | 186 | 186 | 62 |

### SH6 narrow-project

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 0.548ms | 1.00x | 125382 | 0.538ms | — | — | — | — |
| A' V1 (tokio x4) | 0.570ms | 1.04x | 125382 | 0.405ms | — | — | — | — |
| D  morsel (x1, splits) | 0.357ms | 0.65x | 125382 | 0.049ms | 16 | 32 | 32 | 32 |
| D  morsel (x4, splits) | 0.419ms | 0.76x | 125382 | 0.148ms | 16 | 32 | 32 | 32 |
| D  morsel (x4, 65536r) | 0.320ms | 0.58x | 125382 | 0.201ms | 4 | 20 | 20 | 20 |
| D  morsel (x4, splits, parallel) | 0.298ms | 0.54x | 125382 | 0.100ms | 16 | 32 | 32 | 32 |

## wide-numeric — ClickBench-shaped: 20 narrow integer columns, five disagreeing chunkings

1000000 rows, 228 natural splits

### WN1 select-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 23.804ms | 1.00x | 1000000 | 6.682ms | — | — | — | — |
| A' V1 (tokio x4) | 28.371ms | 1.19x | 1000000 | 5.728ms | — | — | — | — |
| D  morsel (x1, splits) | 12.243ms | 0.51x | 1000000 | 0.363ms | 228 | 4560 | 4560 | 4560 |
| D  morsel (x4, splits) | 5.813ms | 0.24x | 1000000 | 0.429ms | 228 | 4560 | 4560 | 4560 |
| D  morsel (x4, 65536r) | 2.173ms | 0.09x | 1000000 | 0.575ms | 16 | 1476 | 1476 | 1476 |
| D  morsel (x4, splits, parallel) | 5.573ms | 0.23x | 1000000 | 0.421ms | 228 | 4560 | 4560 | 4560 |

### WN2 point-filter

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 2.758ms | 1.00x | 2 | 1.372ms | — | — | — | — |
| A' V1 (tokio x4) | 3.357ms | 1.22x | 2 | 2.000ms | — | — | — | — |
| D  morsel (x1, splits) | 1.769ms | 0.64x | 2 | 0.723ms | 147 | 588 | 441 | 153 |
| D  morsel (x4, splits) | 0.967ms | 0.35x | 2 | 0.464ms | 147 | 588 | 441 | 153 |
| D  morsel (x4, 65536r) | 0.693ms | 0.25x | 2 | 0.469ms | 16 | 276 | 215 | 89 |
| D  morsel (x4, splits, parallel) | 0.871ms | 0.32x | 2 | 0.372ms | 147 | 588 | 441 | 153 |

### WN3 dashboard

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 9.916ms | 1.00x | 874895 | 1.766ms | — | — | — | — |
| A' V1 (tokio x4) | 8.563ms | 0.86x | 874895 | 1.749ms | — | — | — | — |
| D  morsel (x1, splits) | 5.149ms | 0.52x | 874895 | 0.195ms | 204 | 1428 | 1224 | 1428 |
| D  morsel (x4, splits) | 2.765ms | 0.28x | 874895 | 0.415ms | 204 | 1428 | 1224 | 1428 |
| D  morsel (x4, 65536r) | 1.492ms | 0.15x | 874895 | 0.438ms | 16 | 555 | 493 | 555 |
| D  morsel (x4, splits, parallel) | 2.570ms | 0.26x | 874895 | 0.301ms | 204 | 1428 | 1224 | 1428 |

### WN4 two-conjuncts

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 37.404ms | 1.00x | 15441 | 7.314ms | — | — | — | — |
| A' V1 (tokio x4) | 25.908ms | 0.69x | 15441 | 4.939ms | — | — | — | — |
| D  morsel (x1, splits) | 19.526ms | 0.52x | 15441 | 0.505ms | 228 | 5016 | 4560 | 5016 |
| D  morsel (x4, splits) | 8.186ms | 0.22x | 15441 | 0.641ms | 228 | 5016 | 4560 | 5016 |
| D  morsel (x4, 65536r) | 3.684ms | 0.10x | 15441 | 1.080ms | 16 | 1569 | 1476 | 1569 |
| D  morsel (x4, splits, parallel) | 7.310ms | 0.20x | 15441 | 0.537ms | 228 | 5016 | 4560 | 5016 |

### WN5 selective-wide

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 9.940ms | 1.00x | 10 | 3.109ms | — | — | — | — |
| A' V1 (tokio x4) | 13.985ms | 1.41x | 10 | 4.065ms | — | — | — | — |
| D  morsel (x1, splits) | 6.331ms | 0.64x | 10 | 0.390ms | 228 | 5016 | 4560 | 645 |
| D  morsel (x4, splits) | 3.122ms | 0.31x | 10 | 0.519ms | 228 | 5016 | 4560 | 645 |
| D  morsel (x4, 65536r) | 2.384ms | 0.24x | 10 | 0.722ms | 16 | 1629 | 1476 | 825 |
| D  morsel (x4, splits, parallel) | 3.745ms | 0.38x | 10 | 0.486ms | 228 | 5016 | 4560 | 656 |

### WN6 packed

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 3.760ms | 1.00x | 250021 | 0.711ms | — | — | — | — |
| A' V1 (tokio x4) | 4.347ms | 1.16x | 250021 | 1.219ms | — | — | — | — |
| D  morsel (x1, splits) | 2.597ms | 0.69x | 250021 | 0.097ms | 147 | 441 | 441 | 441 |
| D  morsel (x4, splits) | 1.702ms | 0.45x | 250021 | 0.304ms | 147 | 441 | 441 | 441 |
| D  morsel (x4, 65536r) | 1.224ms | 0.33x | 250021 | 0.323ms | 16 | 215 | 215 | 215 |
| D  morsel (x4, splits, parallel) | 1.831ms | 0.49x | 250021 | 0.227ms | 147 | 441 | 441 | 441 |

## narrow-analytic — TPC-H Q6/Q1-shaped: conjunctive range filter, narrow projection

1000000 rows, 49 natural splits

### NA1 q6-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 5.399ms | 1.00x | 30093 | 1.985ms | — | — | — | — |
| A' V1 (tokio x4) | 3.217ms | 0.60x | 30093 | 1.446ms | — | — | — | — |
| D  morsel (x1, splits) | 4.416ms | 0.82x | 30093 | 0.229ms | 49 | 294 | 196 | 294 |
| D  morsel (x4, splits) | 2.144ms | 0.40x | 30093 | 0.343ms | 49 | 294 | 196 | 294 |
| D  morsel (x4, 65536r) | 1.837ms | 0.34x | 30093 | 0.433ms | 16 | 168 | 100 | 168 |
| D  morsel (x4, splits, parallel) | 2.109ms | 0.39x | 30093 | 0.326ms | 49 | 294 | 196 | 294 |

### NA2 q1-shape

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 2.323ms | 1.00x | 857083 | 0.939ms | — | — | — | — |
| A' V1 (tokio x4) | 2.186ms | 0.94x | 857083 | 1.003ms | — | — | — | — |
| D  morsel (x1, splits) | 1.371ms | 0.59x | 857083 | 0.093ms | 49 | 196 | 196 | 196 |
| D  morsel (x4, splits) | 0.934ms | 0.40x | 857083 | 0.232ms | 49 | 196 | 196 | 196 |
| D  morsel (x4, 65536r) | 0.860ms | 0.37x | 857083 | 0.242ms | 16 | 100 | 100 | 100 |
| D  morsel (x4, splits, parallel) | 0.900ms | 0.39x | 857083 | 0.143ms | 49 | 196 | 196 | 196 |

### NA3 scan-all

| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| A  V1 (1 thread) | 1.029ms | 1.00x | 1000000 | 0.434ms | — | — | — | — |
| A' V1 (tokio x4) | 1.679ms | 1.63x | 1000000 | 0.698ms | — | — | — | — |
| D  morsel (x1, splits) | 0.358ms | 0.35x | 1000000 | 0.027ms | 49 | 196 | 196 | 196 |
| D  morsel (x4, splits) | 0.559ms | 0.54x | 1000000 | 0.111ms | 49 | 196 | 196 | 196 |
| D  morsel (x4, 65536r) | 0.338ms | 0.33x | 1000000 | 0.092ms | 16 | 100 | 100 | 100 |
| D  morsel (x4, splits, parallel) | 0.477ms | 0.46x | 1000000 | 0.101ms | 49 | 196 | 196 | 196 |

All configurations matched the V1 oracle.
