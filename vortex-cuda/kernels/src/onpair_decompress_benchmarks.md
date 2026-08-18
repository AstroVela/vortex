# OnPair CUDA decompression benchmarks

## Result

On the eight-workload bits12 matrix, the packed-length/high-plane candidate is
**1.86% faster than the legacy kernel by paired geomean**. The sum of the eight
combined median times improves by 1.55%. This is a modest aggregate win, not a
universal win: five workloads improve and three regress.

A positive percentage means `old_ms / packed_ms - 1`.

| dataset | old ms | packed ms | old GiB/s | packed GiB/s | paired speedup | old-first | packed-first |
|---|---:|---:|---:|---:|---:|---:|---:|
| TPC-H `l_comment` | 0.927107 | 0.847870 | 1004.5 | 1098.4 | +9.33% | +9.37% | +9.29% |
| TPC-H `l_shipmode` | 0.943464 | 1.010568 | 1015.1 | 947.7 | -6.69% | -6.79% | -6.59% |
| ClickBench `URL` | 1.195731 | 1.129241 | 778.9 | 824.7 | +5.80% | +5.75% | +5.85% |
| FineWeb | 1.320650 | 1.349755 | 705.2 | 690.0 | -2.16% | -2.27% | -2.05% |
| GH Archive | 1.227504 | 1.186384 | 758.7 | 785.0 | +3.41% | +3.33% | +3.49% |
| HDFS | 1.010725 | 0.933562 | 921.4 | 997.6 | +8.18% | +8.14% | +8.21% |
| CodeSearchNet | 1.317375 | 1.286479 | 707.0 | 723.9 | +2.38% | +2.31% | +2.46% |
| FineWeb2 Chinese | 1.313822 | 1.371469 | 708.9 | 679.1 | -4.19% | -4.22% | -4.17% |

Aggregate order results:

- old then packed: +1.803% paired geomean;
- packed then old: +1.914% paired geomean;
- both orders combined: +1.859% paired geomean;
- ratio of summed combined medians: +1.547%.

The maximum per-row coefficient of variation across the ten fresh-process
samples was 0.21%. The two order aggregates differ by 0.11 percentage points.
Treating each of the ten order/run positions as one aggregate repeat across all
eight datasets gives a 95% t-interval of +1.804% to +1.913% around the +1.859%
geomean. This interval measures repeat noise on this machine, not uncertainty
across all possible workloads or GPU architectures.

This supports a small real aggregate effect on this machine, while the result
is close enough to zero that it should not be generalized to other GPUs or
workload mixes without measurement.

## Protocol

- Date: 2026-08-17 UTC.
- GPU: NVIDIA GH200 480GB, sm_90.
- Driver: 595.71.05.
- CUDA toolkit: 13.1, nvcc 13.1.115.
- Cells: the same eight saved `bits12_chunk1000mb_thr0.20` cells used by the
  previous old/new comparison.
- Per cell and order: one untabulated warm-up process, then five fresh measured
  processes.
- Per process: 100 timed iterations of each kernel.
- Orders: `onpair_old_2,onpair_decompress` and the reverse.
- Execution: serial; no benchmark processes ran concurrently.
- Validation: full GPU output copied back and compared byte-for-byte with CPU
  decode after timing.
- Validation result: 80/80 measured processes verified, and both kernels
  verified in every process.
- TPC-H `l_shipmode` repeats its one small Vortex part four times, matching the
  prior 1,028,346,692-byte decoded workload. Every other row decodes about
  1,000,000,000 bytes once.

Combined medians use all ten measured processes per dataset. The paired row
speedup is the geometric mean of `old_ms / packed_ms` from each of those ten
processes. The overall result is the geometric mean of the eight row ratios.

Raw JSON and stderr logs are in:

```text
/tmp/onpair-packed-20260817/old-first/
/tmp/onpair-packed-20260817/new-first/
/tmp/onpair-packed-20260817/metadata/
```

The timing binary SHA-256 was
`78892f6cd5b1a1955a7898861b77d883cfc87e9a7eeeedb6439bed084b2b1312`.
The later metadata-field-only rebuild is
`3eea4ab02cf31fb4ee3818ac7cdbde1ef1fa3028dd683c56b0f0451272d2b6a0`.

## Saved memory counters

The metadata pass now emits the code stream and each hot table separately.

| dataset | code bytes (streaming) | low plane | high plane | packed lengths | hot tables total |
|---|---:|---:|---:|---:|---:|
| TPC-H `l_comment` | 255,867,710 | 32,768 | 32,768 | 2,048 | 67,584 |
| TPC-H `l_shipmode` | 479,888,416 | 8,800 | 8,800 | 552 | 18,152 |
| ClickBench `URL` | 428,121,346 | 32,768 | 32,768 | 2,048 | 67,584 |
| FineWeb | 594,881,558 | 32,768 | 32,768 | 2,048 | 67,584 |
| GH Archive | 448,975,578 | 32,768 | 32,768 | 2,048 | 67,584 |
| HDFS | 302,167,820 | 32,768 | 32,768 | 2,048 | 67,584 |
| CodeSearchNet | 532,199,242 | 32,768 | 32,768 | 2,048 | 67,584 |
| FineWeb2 Chinese | 632,145,574 | 32,768 | 32,768 | 2,048 | 67,584 |

The code arrays are hundreds of megabytes and stream through cache; they do not
reside in L1. For saturated bits12 dictionaries, the reusable low/high/length
working set is 67,584 bytes.

## Reproduction command

The measured process command was:

```bash
ONPAIR_FAST=1 \
ONPAIR_KERNELS=onpair_old_2,onpair_decompress \
/tmp/vortex-core8-impl/target/release/onpair-chunk-bench \
  gpu-decode-vortex \
  --vortex <saved-cell-or-file> \
  --column <column> \
  --gpu-iters 100 \
  --gpu-validate
```

The reverse pass swapped the two names in `ONPAIR_KERNELS`.

## Build evidence

- packed CUDA source SHA-256:
  `69dff7561772fa226fc2f4ecf88beacd91386f6ed34796c6f409dc08f7aaa8c1`;
- legacy CUDA source SHA-256:
  `c239aaab30685e7333eb16d1fc23a1886a7191d796eec39461ed14a076994abf`;
- release PTX SHA-256:
  `5ca99cca78d82fa85908859918ffb04bd50569ff62283dbd39a1916a967573c8`;
- release PTX size: 27,048 bytes;
- `cargo build -p vortex-cuda --release`: passed;
- `clang-format --dry-run --Werror --style=file`: passed for all three CUDA
  sources.

GPU clocks were not locked. These numbers prove behavior on this GH200 and this
saved workload matrix; they are not a cross-architecture performance guarantee.

## 4/6/8 tokens-per-thread experiment

The packed-split decoder was generalized without changing its dictionary
representation. Six TPT owns 192 tokens per warp; eight TPT owns 256. Prefix
scans remain packed in groups of four 16-bit fields, so 6 and 8 TPT execute a
second scan group. Both variants use 256-thread blocks.

Protocol matched the earlier matrix: eight saved bits12 cells, two execution
orders (4→6→8 and 8→6→4), one warm-up plus five fresh measured processes per
order, 100 iterations per kernel, serial execution, and full byte validation.
All 80 measured processes verified every kernel.

| dataset | 4 TPT ms | 6 TPT ms | 8 TPT ms | 6 vs 4 | 8 vs 4 |
|---|---:|---:|---:|---:|---:|
| TPC-H `l_comment` | 0.848562 | 0.815888 | 1.016772 | +3.99% | -16.83% |
| TPC-H `l_shipmode` | 1.010452 | 0.881332 | 0.973080 | +14.67% | +3.79% |
| ClickBench `URL` | 1.128991 | 1.024548 | 1.164947 | +10.23% | -3.01% |
| FineWeb | 1.350680 | 1.194911 | 1.354946 | +12.97% | -0.35% |
| GH Archive | 1.186568 | 1.072169 | 1.224839 | +10.69% | -3.17% |
| HDFS | 0.933784 | 0.880300 | 0.994868 | +6.06% | -6.14% |
| CodeSearchNet | 1.287054 | 1.135667 | 1.270494 | +13.32% | +1.31% |
| FineWeb2 Chinese | 1.371110 | 1.217380 | 1.391460 | +12.63% | -1.45% |

Six TPT improves every row: +10.514% paired geomean and +10.885% by summed
medians. Eight TPT is -3.419% paired geomean and -2.920% by summed medians.

Compiler resource usage on sm_90:

| TPT | registers/thread | spills | static shared/block |
|---:|---:|---:|---:|
| 4 | 64 | 0 B | 20,736 B |
| 6 | 64 | 0 B | 30,976 B |
| 8 | 64 | 0 B | 41,216 B |

The expected register-growth regression did not occur: all variants compile to
64 registers with no spills. Shared memory grows linearly because each warp
needs decoded staging plus a high-byte request queue.

### Nsight Systems

A 20-iteration `l_comment` trace is stored at
`/tmp/onpair-tpt-lcomment.nsys-rep`.

| TPT | median kernel time | grid blocks | block threads | registers | static shared |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.853392 ms | 124,936 | 256 | 64 | 20,736 B |
| 6 | 0.809040 ms | 83,291 | 256 | 64 | 30,976 B |
| 8 | 1.006592 ms | 62,468 | 256 | 64 | 41,216 B |

### Nsight Compute: why 6 wins and 8 loses

One representative `l_comment` invocation per variant:

| TPT | instructions | L1 hit | L2 read sectors | DRAM read | long-scoreboard stall |
|---:|---:|---:|---:|---:|---:|
| 4 | 555.0M | 85.35% | 23.33M | 264.0 MB | 18.88% |
| 6 | 495.1M | 85.37% | 22.89M | 261.3 MB | 18.66% |
| 8 | 512.9M | 66.82% | 79.34M | 260.0 MB | 38.04% |

Six TPT reduces total warp instructions by about 11% while preserving cache hit
rate, occupancy, and scoreboard behavior. Eight TPT's 41 KB/block shared
allocation consumes more of Hopper's unified L1/shared capacity: L1 hit rate
falls by 18.5 percentage points, L2 sector traffic grows 3.4×, and long
scoreboard stalls double. DRAM bytes stay similar because the extra traffic
mostly hits L2.

The tiny seven-code `l_shipmode` dictionary is the counterexample: capacity is
irrelevant, so 8 TPT's lower instruction count can win. Six TPT is still best
because it reduces instructions without 8 TPT's larger scoreboard penalty.

Raw timing JSON is under `/tmp/onpair-tpt-20260817/`. Nsight Compute CSVs are
`/tmp/onpair-ncu-{4tpt,6tpt,8tpt}.csv` and
`/tmp/onpair-shipmode-ncu-*.csv`.

## Split-dictionary cache-sharing experiment

An aligned 8-byte low-entry load at address `dict + code * 8` never straddles
a 32-byte sector: every start is 8-byte aligned and every sector contains four
complete entries. The old 16-byte stride similarly places two complete entries
per sector.

The GPU automatically coalesces lanes in one warp instruction that address the
same 32-byte sector. To measure the opportunity directly, an env-gated CPU
diagnostic replayed the actual 4-TPT access grouping: four consecutive 32-lane
load instructions per 128-token warp. It counted unique 32-byte sectors and
128-byte lines for both address mappings:

```text
stride 16: sector = code / 2,  line = code / 8
stride  8: sector = code / 4,  line = code / 16
```

| dataset | sectors saved by stride 8 | 128-B lines saved | stride-16 loads/sector | stride-8 loads/sector |
|---|---:|---:|---:|---:|
| TPC-H `l_comment` | 0.839% | 3.051% | 1.017 | 1.025 |
| TPC-H `l_shipmode` | 14.184% | 39.740% | 4.605 | 5.366 |
| ClickBench `URL` | 1.598% | 4.389% | 1.225 | 1.245 |
| FineWeb | 0.923% | 3.460% | 1.055 | 1.065 |
| GH Archive | 1.825% | 4.858% | 1.199 | 1.222 |
| HDFS | 0.845% | 1.903% | 1.258 | 1.269 |
| CodeSearchNet | 1.482% | 3.957% | 1.207 | 1.225 |
| FineWeb2 Chinese | 1.282% | 3.891% | 1.107 | 1.122 |

For ordinary text, halving the stride improves same-instruction sector sharing
by only 0.84–1.83%; nearby-code accesses are not common enough to approach the
theoretical 2×. The large benefit of the split dictionary instead comes from
halving the common load width and hot working-set footprint. The tiny,
seven-code `l_shipmode` column has extreme code repetition and realizes much
more sharing.

A software sharing implementation would look roughly like:

```cpp
uint32_t sector = code >> 2;               // four 8-byte entries / sector
uint32_t peers = __match_any_sync(mask, sector);
int leader = __ffs(peers) - 1;
uint4 line = lane == leader
    ? *reinterpret_cast<const uint4 *>(dict_s8_lo + sector * 32)
    : make_uint4(0, 0, 0, 0);
// Shuffle the selected two words from leader to each peer.
```

That is not implemented as a candidate because it duplicates hardware
coalescing, fetches the same 32-byte sector, and adds match, branch, and shuffle
instructions. The direct code counts show too little typical sector reduction
to compensate. The original stride-16 versus split-8 NCU comparison confirms
the useful hardware effect is a modest L1 improvement (83.15%→85.35%) and a
small L2-sector reduction (23.82M→23.33M), not a large coalescing multiplier.

Raw diagnostic logs are under `/tmp/onpair-cache-sharing-20260817/`.

## 4/5/6/7 tokens-per-thread experiment

Five- and seven-token wrappers instantiate the same packed-split decoder as
the six- and eight-token experiments. They own 160 and 224 tokens per warp,
respectively; the low/high dictionary planes and packed four-bit lengths are
unchanged.

The requested original-paper-family matrix uses every currently saved
representative text column for TPC-H, ClickBench, and FineWeb, plus the original
English Wikipedia cell. Protocol: GH200 sm_90; bits12; serial execution; orders
4→5→6→7 and 7→6→5→4; one discarded warm-up plus five fresh measured processes
per order and row; 100 timed iterations per kernel. All 70 measured processes
and all 280 kernel results passed full byte-for-byte GPU/CPU validation.

The table reports the median of ten measured processes per row. Throughput
changes use the corresponding median latency ratio versus four TPT.

| dataset / column | decoded bytes | 4 TPT ms | 5 TPT ms | 6 TPT ms | 7 TPT ms | 5 vs 4 | 6 vs 4 | 7 vs 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TPC-H `l_comment` | 999,999,993 | 0.848832 | 0.946770 | **0.814891** | 0.893232 | -10.34% | **+4.17%** | -4.97% |
| TPC-H `l_shipmode` | 1,028,346,692 | 1.009532 | 1.071761 | **0.880466** | 0.918790 | -5.81% | **+14.66%** | +9.88% |
| ClickBench `URL` | 999,999,995 | 1.130920 | 1.226655 | **1.023760** | 1.119617 | -7.81% | **+10.47%** | +1.01% |
| ClickBench `SearchPhrase` | 739,636,342 | 0.727938 | 0.791021 | **0.675710** | 0.748519 | -7.98% | **+7.73%** | -2.75% |
| FineWeb `text` | 999,998,604 | 1.349977 | 1.453529 | **1.194626** | 1.285338 | -7.12% | **+13.00%** | +5.03% |
| FineWeb `url` | 229,657,845 | 0.336665 | 0.359367 | **0.296261** | 0.321212 | -6.32% | **+13.64%** | +4.81% |
| Wikipedia `text` | 299,998,376 | 0.429892 | 0.460022 | **0.381346** | 0.409817 | -6.55% | **+12.73%** | +4.90% |

Throughput in decimal GB/s from those same medians:

| dataset / column | 4 TPT | 5 TPT | 6 TPT | 7 TPT |
|---|---:|---:|---:|---:|
| TPC-H `l_comment` | 1,178.1 | 1,056.2 | **1,227.2** | 1,119.5 |
| TPC-H `l_shipmode` | 1,018.6 | 959.5 | **1,168.0** | 1,119.2 |
| ClickBench `URL` | 884.2 | 815.2 | **976.8** | 893.2 |
| ClickBench `SearchPhrase` | 1,016.1 | 935.0 | **1,094.6** | 988.1 |
| FineWeb `text` | 740.8 | 688.0 | **837.1** | 778.0 |
| FineWeb `url` | 682.2 | 639.1 | **775.2** | 715.0 |
| Wikipedia `text` | 697.8 | 652.1 | **786.7** | 732.0 |

Geomean of the seven per-row median throughput ratios versus four TPT:

| TPT | overall | wins vs 4 |
|---:|---:|---:|
| 5 | -7.428% | 0/7 |
| 6 | **+10.857%** | **7/7** |
| 7 | +2.447% | 5/7 |

Family geomeans `(5, 6, 7)` versus four TPT are TPC-H `(-8.103%, +9.286%,
+2.184%)`, ClickBench `(-7.890%, +9.090%, -0.888%)`, FineWeb `(-6.722%,
+13.321%, +4.920%)`, and Wikipedia `(-6.550%, +12.730%, +4.898%)`.

Compiler resource usage on sm_90 shows a five-TPT cost and bounds the
seven-TPT tradeoff:

| TPT | registers/thread | spills | static shared/block |
|---:|---:|---:|---:|
| 4 | 64 | 0 B | 20,736 B |
| 5 | 72 | 0 B | 25,856 B |
| 6 | 64 | 0 B | 30,976 B |
| 7 | 64 | 0 B | 36,096 B |

Five TPT is the only variant that increases register allocation, to 72, and it
loses every row; that resource increase is consistent with, but does not alone
prove, the cause of its regression. Seven TPT keeps 64 registers but allocates
5,120 more shared bytes per block than six TPT; its lower chunk count helps
five rows, yet still loses to six TPT everywhere. Six TPT remains the best
balance.

Measured-process command (with the saved cell path and column substituted):

```bash
ONPAIR_FAST=1 \
ONPAIR_KERNELS=onpair_decompress,onpair_decompress_5tpt,\
onpair_decompress_6tpt,onpair_decompress_7tpt \
/tmp/vortex-core8-impl/target/release/onpair-chunk-bench \
  gpu-decode-vortex --vortex <saved-cell> --column <column> \
  --gpu-iters 100 --gpu-validate
```

The reverse pass reversed the four names in `ONPAIR_KERNELS`.

Raw timing JSON and stderr logs are under
`/tmp/onpair-tpt4567-20260817/{4-5-6-7,7-6-5-4}/`. The measured binary SHA-256
is `7ac8a4ba2c9c67b0653887c49d8e2ed01891e6cffee936b5c21597109eb8c199`.

### Odd-count resource diagnosis

One representative Nsight Compute launch on the 1 GB TPC-H `l_comment` cell
used the same counter set as the earlier 4/6/8 analysis:

| TPT | registers | register block limit | achieved occupancy | instructions | L1 hit | L2 read sectors | long-scoreboard stall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 64 | 4 | 45.08% | 555.0M | 85.35% | 23.33M | 18.88% |
| 5 | 72 | 3 | 33.45% | 545.1M | 85.52% | 23.14M | 19.62% |
| 6 | 64 | 4 | 45.88% | 495.1M | 85.37% | 22.89M | 18.66% |
| 7 | 64 | 4 | 45.08% | 532.0M | 81.19% | 37.88M | 25.14% |

There is no general odd-number penalty. Five TPT is a resource-count case: it
is the only variant compiled to 72 registers, reducing the register-limited
resident-block ceiling from four to three and measured occupancy from about
46% to 33%. Its cache counters remain comparable, so the occupancy loss is the
strongest observed signal.

Seven TPT disproves a universal odd-count rule: it compiles to 64 registers and
retains four-block and 45% occupancy. Relative to six TPT, however, it executes
7.5% more instructions, its larger shared allocation leaves more pressure on
Hopper's unified L1/shared capacity, L1 hit rate drops 4.2 points, L2 sectors
rise 65%, and long-scoreboard stalls rise 35%. Six TPT is the sweet spot.

The code shape also disfavors five: every TPT value above four pays for a
second four-field warp-scan group, but five fills only one of those four fields;
six fills two and seven fills three. That fixed scan cost and the compiler's
72-register allocation together account for the observed five-TPT weakness.

Raw profiles are `/tmp/onpair-ncu-5tpt-resource.csv` and
`/tmp/onpair-ncu-7tpt-resource.csv`; comparable 4/6 profiles are
`/tmp/onpair-ncu-{4tpt,6tpt}.csv`.

## Five TPT four-block launch bound

The five-TPT wrapper now overrides the generic two-block launch bound with
`__launch_bounds__(256, 4)`. On sm_90 this changes ptxas allocation from 72 to
64 registers/thread with zero spills and unchanged 25,856-byte static shared
memory. The kernel code and data layout are otherwise identical.

Protocol: the same seven bits12 cells as the 4/5/6/7 experiment; orders
baseline→candidate→4TPT and candidate→baseline→4TPT; one discarded warm-up plus
five fresh measured processes per order and row; 100 iterations per kernel;
serial execution; full byte validation. All 70 measured processes and all 210
kernel results verified.

| dataset / column | 72-reg 5 TPT GB/s | 64-reg 5 TPT GB/s | candidate speedup | 4 TPT GB/s | candidate vs 4 |
|---|---:|---:|---:|---:|---:|
| TPC-H `l_comment` | 1,055.2 | **1,181.7** | +11.99% | 1,178.5 | +0.27% |
| TPC-H `l_shipmode` | 959.1 | **1,084.6** | +13.08% | 1,018.2 | +6.52% |
| ClickBench `URL` | 814.4 | **909.8** | +11.72% | 885.3 | +2.77% |
| ClickBench `SearchPhrase` | 933.3 | **1,040.7** | +11.50% | 1,017.5 | +2.28% |
| FineWeb `text` | 687.5 | **770.1** | +12.02% | 740.1 | +4.05% |
| FineWeb `url` | 639.0 | **704.9** | +10.31% | 681.9 | +3.37% |
| Wikipedia `text` | 652.0 | **725.8** | +11.32% | 697.9 | +4.00% |

The geomean of per-row median throughput ratios is **+11.702%**; the geomean
of all 70 process-paired ratios is **+11.651%**. The candidate wins every row
against both the old 72-register five-TPT kernel and four TPT. Six TPT remains
faster on every row.

A representative Nsight Compute `l_comment` launch confirms the mechanism:

| metric | 72-reg baseline | 64-reg launch bound |
|---|---:|---:|
| register-limited blocks/SM | 3 | 4 |
| achieved occupancy | 33.45% | 45.78% |
| instructions | 545.1M | 544.3M |
| L1 hit | 85.52% | 85.54% |
| L2 read sectors | 23.14M | 23.07M |
| long-scoreboard stall | 19.62% | 18.63% |

Instruction and cache work are essentially unchanged; the performance gain
tracks the restored fourth resident block and 12.3-point occupancy increase.
This makes register scheduling, rather than the high-plane data representation,
the sufficient fix for five TPT.

Raw timings are under
`/tmp/onpair-5tpt-lb4-20260817/{base-first,lb4-first}/`; the candidate profile
is `/tmp/onpair-ncu-5tpt-lb4.csv` and the baseline profile is
`/tmp/onpair-ncu-5tpt-resource.csv`.

## Six TPT compact 12-byte drain

The follow-on `onpair_decompress_6tpt_cap12_lb5` kernel reduces the per-warp
drain allocation from the 16-byte/token maximum to 12 bytes/token and uses a
correct direct-write fallback for overflowing warps.

Compiler and representative `l_comment` profile:

| metric | six-TPT baseline | compact 12-byte |
|---|---:|---:|
| registers/thread | 64 | 48 |
| shared bytes/block | 30,976 | 24,832 |
| spills | 0 | 0 |
| register block limit | 4 | 5 |
| shared block limit | 4 | 5 |
| achieved occupancy | 45.87% | 56.72% |
| long-scoreboard stall | 18.75% | 15.83% |
| L1 hit rate | 85.36% | 85.36% |
| executed instructions | 495.1M | 517.1M |

Protocol: seven saved bits12 cells; 100 iterations/kernel; baseline-first and
candidate-first orders; one discarded warm-up plus five fresh measured
processes per order and row; serial execution; byte validation in all 70
measured processes. Values below are medians in decimal GB/s.

| dataset / column | baseline | compact 12-byte | change |
|---|---:|---:|---:|
| TPC-H `l_comment` | 1,225.8 | **1,243.8** | +1.47% |
| TPC-H `l_shipmode` | 1,169.2 | **1,193.1** | +2.04% |
| ClickBench `URL` | 977.2 | **994.0** | +1.72% |
| ClickBench `SearchPhrase` | 1,096.7 | **1,122.4** | +2.35% |
| FineWeb `text` | 836.6 | **846.5** | +1.18% |
| FineWeb `url` | **775.8** | 763.5 | -1.58% |
| Wikipedia `text` | 787.2 | **790.1** | +0.37% |
| geomean | 966.0 | **976.3** | **+1.07%** |

The two order-specific geomean changes were +1.05% and +1.08%. Raw timing
artifacts are under `/tmp/onpair-cap12-20260817/`. The partial-drain and
shared-destination follow-ups are documented in
`onpair_decompression_design.md`; neither beat this kernel.

## Adaptive cap-9, one-low-register drain

`onpair_decompress_6tpt_cap9_keep1_lb6` is a fixed-code follow-up to the compact
cap-12 kernel. It keeps one of six low `uint2` values live and reloads the other
five during emission. It is eligible only when every 192-token output chunk is
at most 1,728 bytes; otherwise the adaptive policy retains cap-12.

The saved-cell chunk histogram explains where whole-column selection applies:

| dataset / column | cap-9-fitting chunks | total chunks | fit rate | maximum chunk bytes | selected kernel |
|---|---:|---:|---:|---:|---|
| TPC-H `l_comment` | 666,322 | 666,323 | 99.9998% | 1,735 | cap-12 |
| TPC-H `l_shipmode` | 312,428 | 312,428 | 100% | 909 | cap-9 |
| ClickBench `URL` | 1,110,638 | 1,114,900 | 99.6177% | 3,072 | cap-12 |
| ClickBench `SearchPhrase` | 690,325 | 690,554 | 99.9668% | 2,186 | cap-12 |
| FineWeb `text` | 1,549,171 | 1,549,171 | 100% | 969 | cap-9 |
| FineWeb `url` | 353,095 | 353,095 | 100% | 864 | cap-9 |
| Wikipedia `text` | 482,187 | 482,187 | 100% | 1,494 | cap-9 |

A per-chunk two-kernel dispatch was also measured, but its chunk-ID loads and
split scheduling lost on the mixed columns. The selected policy therefore uses
one kernel for the whole column.

Final controlled results below use the exact PTX generated by this branch.
Protocol: GH200 sm_90; two orders; one discarded warm-up plus five fresh
processes/order and row; 100 iterations/kernel; serial execution; full GPU/CPU
byte validation. All 40 measured processes verified both kernels. Throughput
is decimal GB/s from the median of the ten measured processes.

| eligible dataset / column | cap-12 GB/s | cap-9 GB/s | change | cap-12 first | cap-9 first |
|---|---:|---:|---:|---:|---:|
| TPC-H `l_shipmode` | 1,195.55 | **1,213.20** | +1.48% | +1.45% | +1.53% |
| FineWeb `text` | 844.69 | **879.91** | +4.17% | +4.03% | +4.32% |
| FineWeb `url` | 763.99 | **783.95** | +2.61% | +2.70% | +2.66% |
| Wikipedia `text` | 789.93 | **819.45** | +3.74% | +3.83% | +3.66% |

The eligible-row geomean improvement is **+2.99%**. Treating the three
ineligible rows as unchanged cap-12 launches, the seven-row adaptive-policy
geomean is **+1.70%**.

A representative FineWeb `text` Nsight Compute launch isolates the resource
effect:

| metric | cap-12 | cap-9 keep-one |
|---|---:|---:|
| registers/thread | 48 | 40 |
| static shared bytes/block | 24,832 | 20,224 |
| spills | 0 B | 0 B |
| register block limit | 5 | 6 |
| shared-memory block limit | 5 | 6 |
| achieved occupancy | 55.75% | 67.58% |
| L1 sector hit rate | 90.48% | 90.48% |
| L2 read sectors | 26.89M | 26.91M |
| executed instructions | 954.24M | 953.65M |
| long-scoreboard stall | 11.18% | 12.73% |

The cache hit rate, L2 traffic, and instruction count are essentially unchanged.
The measured speedup tracks the sixth resident block and 11.83-point achieved
occupancy increase, not a code-format or cache-sharing change.

Raw final timings are under
`/tmp/onpair-cap9-final-controlled-20260817/`. Final profiles are
`/tmp/onpair-cap9-final-ncu-{base,candidate}.csv`. The candidate PTX SHA-256 is
`716966dc5ba9dbd2ff89cbe30715c7165ee72df67e43dd7fe2cd9d2793624c26`.

## Final 12-bit and 16-bit selector search

The final search covered 32 non-empty columns from Amazon book reviews,
ClickBench, FineWeb, TPC-H SF10, and Wikipedia at both 12 and 16 bits. Each cell
used ten fresh measured processes (five in each kernel order), 100 timed
iterations/process, serial execution, and full GPU/CPU byte validation. Each
bit width had 320 measured processes; all 640 processes validated. Results
apply to the GH200 sm_90 system described above.

The capacity-safe 12-bit selector reduced the sum of per-column medians from
12.288316 ms to 10.751556 ms, a 14.29% throughput improvement. Its aggregate
output rate was 1.004 TB/s. It won 27 of 32 columns.

For 16-bit data, the general direct-high kernel compiled to 48 registers and
24,832 bytes shared memory. Removing the dense high-request queue improved the
32-column output rate from 0.823 TB/s for plain 6-TPT to 0.833 TB/s. Selecting
the 40-register keep-one variant only for dictionaries with at most 384 entries
and at most 1% token-weighted long codes reached 0.834 TB/s, 1.38% faster than
plain 6-TPT by total time.

Representative output rates are decimal TB/s:

| column | 12-bit legacy | 12-bit selected | 16-bit legacy | 16-bit reported best |
|---|---:|---:|---:|---:|
| ClickBench `URL` | 0.835 | 0.974 | 0.900 | 1.006 |
| TPC-H `l_comment` | 1.078 | 1.245 | 0.621 | 0.714 |
| Amazon book-reviews `text` | 0.786 | 0.945 | 0.570 | 0.570 |
| FineWeb `text` | 0.757 | 0.881 | 0.518 | 0.621 |
| Wikipedia `text` | 0.707 | 0.817 | 0.510 | 0.591 |

The 16-bit reported-best column uses two measured overrides that are not
encoded in the generic selector: legacy for Amazon book reviews and
`onpair_decompress_6tpt_directhi_keep3_lb4` for TPC-H `l_comment`. The
data-only tree intentionally avoids dispatching on dataset names.

The semantically renamed kernels reproduce the measured experimental aliases:

| committed kernel | measured alias | registers | shared/block | spills |
|---|---|---:|---:|---:|
| `onpair_decompress_6tpt_directhi_lb5` | `onpair_decompress_6tpt_window32_lb5` | 48 | 24,832 B | 0 B |
| `onpair_decompress_6tpt_directhi_keep1_lb6` | `onpair_decompress_6tpt_cap10_lb6` | 40 | 24,832 B | 0 B |
| `onpair_decompress_6tpt_directhi_keep3_lb4` | `onpair_decompress_6tpt_lb2` | 64 | 24,832 B | 0 B |

Raw controlled results are in `/tmp/onpair16-search/` and
`/tmp/onpair-allcols-stages-20260817/`.

## Wikipedia code distribution and 16-bit high-plane cache policy

The final Wikipedia-specific pass used the same GH200, saved Vortex cells, and
original u16 code streams as the selector search. A read-only staging
diagnostic counted all code occurrences before GPU upload; it did not relabel
codes or change dictionary storage.

| statistic | 12-bit | 16-bit |
|---|---:|---:|
| tokens | 92,579,734 | 53,459,252 |
| distinct codes | 4,041 | 65,355 |
| token length 1--4 | 80.189% | 44.135% |
| token length 5--8 | 18.051% | 38.486% |
| token length 9--12 | 1.605% | 14.075% |
| token length 13--16 | 0.156% | 3.304% |
| hottest 16 codes | 6.446% | 2.300% |
| hottest 256 codes | 41.779% | 13.074% |
| hottest 4,096 codes | 100.000% | 46.490% |
| long codes per 192-token warp, p50 / p99 / max | 3 / 11 / 69 | 34 / 63 / 116 |
| decoded bytes per 192-token warp, p50 / p99 / max | 634 / 760 / 1,494 | 1,097 / 1,361 / 1,813 |

Moving from 16-byte to split 8-byte dictionary spacing saved only 1.080% of
requested sectors at 12 bits and 0.330% at 16 bits for this stream, so further
cache-line sharing was not a useful lever. At 16 bits, 17.379% of tokens use
the high plane and 81.0% of those suffixes are at most four bytes.

Nsight Compute on the 16-bit direct-high baseline reported 48
registers/thread, 25.86 KiB allocated shared/block, 56.38% active-warps
occupancy, 37.57% long-scoreboard stalls, 43.52% L1 sector hit rate, and
93.37% L2 sector hit rate. Explicit L1 prefetching regressed throughput by
9.7%; loading four bytes for short suffixes was neutral. Marking only
high-plane loads cache-global (`__ldcg`) retained the same 48-register and
24,832-byte static-shared resource shape and won.

Ten order-balanced processes (five per order, 100 timed iterations/process)
measured:

| Wikipedia 16-bit kernel | mean runtime | output rate | change |
|---|---:|---:|---:|
| `onpair_decompress_6tpt_directhi_lb5` | 0.524678 ms | 571.78 GB/s | baseline |
| `onpair_decompress_6tpt_directhi_highcg_lb5` | **0.516133 ms** | **581.24 GB/s** | **+1.66%** |

All candidates passed full GPU/CPU byte validation. The same policy improved
FineWeb text by 0.48% and FineWeb URL by 0.82%. It made ClickBench OriginalURL,
whose long-token share is 62.03%, 12.9% slower by runtime. The selector
therefore uses high-CG only for dictionaries with at least 32,768 entries and
at most 25% long tokens. The 12-bit cap-8 follow-up was noise-equivalent to
cap-9 (818.64 versus 818.57 GB/s), so cap-9 remains selected.
