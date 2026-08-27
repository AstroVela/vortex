# Morsel Prototype: Handoff

Everything needed to re-run the morsel-executor evaluation on other hardware and interpret what
comes back. All the code is on branch `claude/morsel-executor-prototype-vvrscx`.

The numbers already recorded came off a **4-core Intel Xeon @ 2.10 GHz with 15 GB RAM, no
hyperthreading, segments held in memory**. That box is small enough that several conclusions are
provisional; the "what to look for" section below says which ones and why.

## 1. Get it running

```bash
git fetch origin claude/morsel-executor-prototype-vvrscx
git checkout claude/morsel-executor-prototype-vvrscx
cargo build --release -p vortex-morsel --features _test-harness --bins
```

Needs nothing external: TPC-H data is generated in-process by `tpchgen`, which was already a
workspace dependency.

```bash
# Correctness. 18 differential tests against the V1 LayoutReader.
cargo test -p vortex-morsel

# Real TPC-H at SF=1. ~1 min including generation and write.
./target/release/tpch-eval 1

# Bigger. Memory scales roughly 1.5 GB per scale factor; SF=10 wants ~24 GB.
./target/release/tpch-eval 10

# Thread scaling, V1 concurrency tuning, morsel-size sweep.
TPCH_SWEEP=1 ./target/release/tpch-eval 1

# The synthetic workloads (string-heavy / wide-numeric / narrow-analytic).
MORSEL_EVAL_ROWS=1000000 ./target/release/morsel-eval
```

Knobs: `TPCH_SCALE`, `TPCH_ROW_BLOCK` (default 8192, the write pipeline's repartition size),
`TPCH_BLOCK_BYTES` (default 1 MiB, the coalescing target — **this is what decides how many
natural splits the file has**, so it is the first thing to vary if you want more morsels),
`MORSEL_EVAL_ROWS`.

## 2. What the eval guarantees

`tpch-eval` validates **before it times anything**: every configuration's output is compared to
V1's on dtype, row count and ordered content, and a mismatch aborts the run rather than quietly
dropping a row from the table. If you see a timing table, the exactness check passed for every
row in it. Then five alternating iterations, median reported.

The morsel executor rejects at build time anything outside its scope (nested structs, non-struct
roots, nullable root structs, non-flat/non-chunked columns), so an unsupported query can never be
timed as if it had run.

## 3. Results to expect, and what would falsify them

Geomeans over the 8 TPC-H scan queries at SF=1, ratios against V1 single-threaded:

| configuration | expected |
|---|--:|
| V1, 4 tokio workers, default concurrency | 0.48x |
| morsel, 1 thread | 0.75x |
| morsel, 1 thread, decode sharing off | 0.76x |
| morsel, 4 threads | 0.31x |
| morsel, 4 threads, 64k morsels | 0.26x |

Against V1 tuned to its *best* concurrency, the morsel executor at 4 threads is **0.61x geomean
(1.64x faster)**, decomposing as ~0.73x single-thread base advantage × ~1.19x better scaling.

**Claims that should hold on any hardware:**

- The morsel executor beats V1 at equal core count on every query.
- `decodes + reuses` with sharing enabled equals `decodes` with it disabled, exactly, per query.
- Time to first batch is an order of magnitude lower (structural: D emits on the first completed
  morsel, V1 after the pipeline fills).

**Claims that are host-specific and worth re-testing:**

- **One driving thread per physical core is optimal.** Measured on 4 cores with no
  hyperthreading, where x8 costs ~10% and x16 ~20%. On a many-core box, or one with SMT, or with
  real storage latency to hide, the optimum may move. This is the single most valuable thing to
  re-measure.
- **Scaling efficiency (2.93x on 4 cores).** Will degrade at higher core counts; where it breaks
  is unknown and matters for P2's admission design.
- **Morsel coalescing is neutral.** Excluding Q19 it is 1.02x — no effect. It only helps Q19
  (0.42x) because Q19's string columns land on different block boundaries, giving it 366 natural
  splits where other queries have 92. Past 64k rows it hurts (Q12: 9.3 → 22.7 ms at 1M).
- **Decode sharing is neutral** (0.75x vs 0.76x), because the real write pipeline repartitions
  every column onto the same row blocks. Q19 again is the exception (0.54x vs 0.69x). A schema
  with more width divergence than `lineitem` would show more.

**Measurement noise on the recorded host was significant** — Q15's V1 single-thread time varied
17.1 ms to 23.9 ms between runs, ~40%. Treat single-query differences under ~20% as noise unless
they reproduce. A quieter machine is one of the main reasons to re-run this.

## 4. What is not covered

- **Zone maps and dictionary layout are disabled for both executors.** P1 supports neither; V1
  supports both. Writing them would compare a pruning executor against a non-pruning one. This is
  a real capability gap in the prototype and is prerequisite to any production comparison — on
  the selective queries V1-with-zone-maps would skip blocks the prototype must read.
- **Segments are in memory.** No IO latency, so nothing here says how either executor behaves on
  object storage. The prototype plan's gate E2 (a latency grid of {0,1,10,50} ms) is not built.
- **Gate E1 as written cannot be evaluated in this repository.** It requires rows B and C — the
  self-paced graph/reactor and pipeline executors — and neither exists at any commit reachable
  here (`self_paced`, `morsel`, `vortex-scan-v2` all find nothing). If those exist on a branch
  elsewhere, running row C against these same fixtures is the highest-value next measurement.
- **`lineitem` only.** The joins in Q12/Q14/Q15/Q19 are above the scan.
- **V1 exposes no IO counters**, so the cold-scan IO invariant is unverified for V1. Output
  equality is proven; IO equality is not.
- ClickBench and FineWeb still need multi-gigabyte downloads and remain synthetic
  (`morsel-eval`). Their absolute times are not comparable to any published suite number.

## 5. Code map

| path | what |
|---|---|
| `vortex-morsel/src/node.rs` | The `ExecNode` contract, the arena, `drive_morsel` |
| `vortex-morsel/src/nodes/` | FLAT, CHUNKED, STRUCT, CONJUNCT (cascade/parallel), FILTER |
| `vortex-morsel/src/io.rs` | The IO plane: keyed cells, tickets, registration |
| `vortex-morsel/src/cells.rs` | Leased shared decoded cells (lease counts from the morsel cut) |
| `vortex-morsel/src/build.rs` | `ExecPlan`: immutable blueprint, per-thread instantiation |
| `vortex-morsel/src/driver.rs` | Atomic-cursor morsel scheduling, order restoration |
| `vortex-morsel/src/tpch.rs` | Real TPC-H generation, queries, write strategy |
| `vortex-morsel/src/harness.rs` | Fair-comparison harness, V1 and morsel runners |
| `vortex-morsel/src/bin/tpch-eval.rs` | The TPC-H evaluation and sweep |
| `vortex-morsel/src/bin/morsel-eval.rs` | The synthetic evaluation |

Design context: [morsel-based plan execution](morsel-based-plan-execution.md),
[graph model](scan-execution-graph-model.md). Results:
[TPC-H findings](morsel-prototype-tpch-findings.md),
[P1 findings](morsel-prototype-p1-findings.md).

## 6. If you are picking this up

In rough order of value:

1. **Re-run `TPCH_SWEEP=1` on a many-core box.** Where thread scaling breaks, and whether one
   thread per core is still optimal, are the two facts P2's admission loop needs and the two this
   host was least able to answer.
2. **Find rows B and C.** The gate this was written against is unmeasurable without them.
3. **Add zone-map support** (a pass-through node ignoring stats is enough to start), then
   re-enable them in the write strategy so the comparison includes pruning.
4. **Build the latency-injection segment source** for gate E2. The IO plane already carries
   `source_range`, `extent`, `producer` and `estimated_bytes`; nothing reads them yet, and the
   latency grid is what makes them earn their place.
5. **A wider schema than `lineitem`.** Decode sharing and morsel coalescing both looked neutral
   here specifically because the write pipeline aligns every column. Q19 shows what happens when
   it does not.
