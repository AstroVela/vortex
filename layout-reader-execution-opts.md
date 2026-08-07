# LayoutReader (v1) scan execution optimizations — task briefs

Three independent, non-overlapping optimizations for the v1 `LayoutReader` scan execution
path, back-ported from work done on the plan-native (v2) scan in
`vortex-scan-v2` / `vortex-layout/src/plan` (branch `claude/plan-execution-perf-v2-xylqjd`,
commit `872c5a0`). Each task touches a different file, so all three can be implemented in
parallel and merged without conflicts.

- Task 1: `vortex-layout/src/scan/tasks.rs` — eager registration of pruning reads
- Task 2: `vortex-layout/src/layouts/zoned/reader.rs` — uniform-zone fast path
- Task 3: `vortex-layout/src/layouts/chunked/reader.rs` — avoid redundant mask slices

## Shared background

### How segment reads work

`vortex-file`'s `FileSegmentSource` (`vortex-file/src/segments/source.rs`) services all leaf
reads. A read future has four states: **registered** (created, not yet polled), **requested**
(polled), **in-flight** (sent to storage), **resolved**. Two properties drive everything
below:

1. A *registered* request triggers no I/O by itself, but is **eligible to be coalesced**
   with neighboring requests into a single larger read. Requests are processed in
   registration order.
2. A read future dropped before completion is **canceled** when possible.

Consequence: the earlier a scan registers its segment reads — ideally for *all* splits
before any split task is polled — the more the driver can coalesce. Registration is cheap
and safe: reads for splits that later turn out to be fully pruned are simply dropped
unpolled and canceled.

This is exactly what fixed v2: moving all plan executions out of the polled task future
took a wide TPC-H sf=1 projection from **1535 preads to 115** (v1 baseline: 119) and turned
an 18% execution-time deficit into a ~9% win. See `vortex-scan-v2/src/tasks.rs` after
commit `872c5a0` for the reference structure.

### v1 split execution today

`vortex-layout/src/scan/tasks.rs::split_exec` builds one task per row split:

- The **projection** evaluation is built *outside* the returned future
  (`ctx.reader.projection_evaluation(...)` around line 135), so projection segment reads
  for every split are registered at task-build time. This is why v1's pread counts looked
  good in benchmarks.
- The **pruning and filter** evaluations run *inside* the `MaskFuture` async block
  (roughly lines 73–130): a fixed-order loop of per-conjunct
  `reader.pruning_evaluation(...)` calls, then an adaptive loop that picks the next
  conjunct at runtime (`FilterExpr::next_conjunct`, selectivity-ordered) and calls
  `reader.filter_evaluation(...)` with the mask accumulated so far. None of these register
  until the task is polled — the same "trickle" problem v2 had.

This goes unnoticed when the filter columns are also projected (their segments are already
registered by the projection). For a query that **filters on a column it does not
project**, v1 loses coalescing on those reads entirely.

### Benchmark harness

`vortex-scan-v2/examples/clickbench_plan_perf.rs` runs the same scan through v1
(`SCAN_VERSION=v1`, `file.scan()`) or v2 (`SCAN_VERSION=v2`) and prints median execution
time. Environment variables:

- `DATASET=<dir>` — directory of `.vortex` files (default: clickbench path)
- `QUERY=lineitem | lineitem_prune | lineitem_and | lineitem_wide` — TPC-H lineitem query
  shapes (see `query_expressions()` in the example)
- `PLAN_ITERS`, `EXEC_ITERS` — iteration counts

Generate TPC-H data (fully local, no downloads):

```bash
cargo build --profile release_debug -p vortex-scan-v2 --examples -p vortex-bench --bin data-gen
target/release_debug/data-gen tpch --formats vortex --opt scale-factor=1   # ~180MB lineitem
target/release_debug/data-gen tpch --formats vortex --opt scale-factor=10  # ~1.8GB lineitem
mkdir -p /tmp/tpch1-lineitem && ln -sf "$PWD/vortex-bench/data/tpch/1/vortex-file-compressed/lineitem.vortex" /tmp/tpch1-lineitem/
mkdir -p /tmp/tpch10-lineitem && ln -sf "$PWD/vortex-bench/data/tpch/10/vortex-file-compressed/lineitem.vortex" /tmp/tpch10-lineitem/
```

Run, e.g.:

```bash
QUERY=lineitem DATASET=/tmp/tpch1-lineitem SCAN_VERSION=v1 PLAN_ITERS=1 EXEC_ITERS=15 \
  target/release_debug/examples/clickbench_plan_perf
```

**Measurement protocol** (this container is noisy): run v1-before vs v1-after interleaved,
3+ rounds each, compare medians; treat <3% as noise. For I/O claims, count syscalls:

```bash
QUERY=... DATASET=... SCAN_VERSION=v1 PLAN_ITERS=1 EXEC_ITERS=1 \
  strace -f -c -e trace=pread64 target/release_debug/examples/clickbench_plan_perf 2>&1 | grep -E "pread64|RESULT"
```

### Ground rules (all tasks)

- v1-only: do **not** modify `vortex-scan-v2/` or `vortex-layout/src/plan/` (already
  optimized on this branch).
- Behavior must be identical: same result arrays, same masks. The scan-v2 example
  `tpch_scan` and `assert_arrays_eq!`-based tests are the correctness oracle.
- Verify: `cargo nextest run -p vortex-layout -p vortex-file -p vortex-scan-v2`, then
  `cargo +nightly fmt --all` and
  `cargo clippy -p vortex-layout -p vortex-scan-v2 --all-targets --all-features`.
- Base your work on `claude/plan-execution-perf-v2-xylqjd`.
- Commits need `Signed-off-by` (see repo `CLAUDE.md`).

---

## Task 1 — Eagerly register per-conjunct pruning evaluations

**File:** `vortex-layout/src/scan/tasks.rs` (`split_exec`)

### Problem

The pruning loop runs inside the filter `MaskFuture`:

```rust
for (idx, conjunct) in filter.conjuncts().iter().enumerate() {
    if mask.all_false() { return Ok(mask); }
    ...
    let conjunct_mask = reader.pruning_evaluation(&row_range, conjunct, mask.clone())?.await?;
    mask = mask.bitand(&conjunct_mask);
}
```

`pruning_evaluation` is what registers zone-map segment reads (and, via recursion, any
nested pruning reads). Because this runs only when the task is polled, zone-map reads
trickle in split-by-split instead of coalescing across splits.

### Change

Hoist the per-conjunct pruning evaluations out of the async block and build them at
task-construction time, one per conjunct, each fed the **original** split `row_mask`
instead of the accumulated mask:

```rust
// outside the future, per conjunct (fixed order):
let pruning_evals: Vec<MaskFuture> = filter.conjuncts().iter()
    .map(|c| reader.pruning_evaluation(&row_range, c, row_mask.clone()))
    .collect::<VortexResult<_>>()?;
// inside the future:
//   mask = row_mask ∧ eval_0 ∧ eval_1 ∧ ... (await sequentially, early-exit on all_false)
```

This is semantically equivalent: each pruning evaluation intersects its zone mask with the
input mask, and intersection is associative/commutative, so feeding every conjunct the
original mask and folding with `bitand` yields the same final mask as feeding each the
accumulated one. Two accepted trade-offs, matching what v2 does:

- A conjunct's pruning evaluation no longer sees the narrowed mask, so it can't
  early-exit on a mask another conjunct already zeroed. Keep the `all_false` early exit
  *between awaits* in the combining future so later evaluations are dropped (= canceled)
  when possible.
- Dynamic-expression re-pruning (the `dynamic_versions` / `filter.dynamic_updates(idx)`
  logic) must stay inside the future exactly as it is today — it depends on runtime
  version checks. Only the initial fixed-order pruning pass moves out.

Do **not** try to hoist the adaptive `filter_evaluation` loop: the conjunct order and the
mask fed to each conjunct are decided at runtime from selectivity statistics
(`FilterExpr::next_conjunct`, `report_selectivity`). That restructuring is out of scope.

### Verification

- Add a query shape to `clickbench_plan_perf.rs` that filters on a **non-projected**
  column, e.g. `QUERY=lineitem_filter_only`: filter `l_linenumber > 5`, projection
  `select(["l_extendedprice"], root())`. This is the case where filter-column reads are
  not pre-registered by the projection.
- Expect: pread count for `SCAN_VERSION=v1` on that query drops (strace protocol above);
  no regression on `lineitem`, `lineitem_and`, `lineitem_prune`, `lineitem_wide` at sf=1
  and sf=10.
- Tests: full protocol from Ground rules. Pay attention to
  `vortex-layout/src/scan` tests and any test asserting *when* segments are requested —
  if one counts registration calls, apply the same fix as
  `vortex-scan-v2/src/tests.rs::TrackingSource` on this branch: record a segment only
  when its read future is first polled, since "registered but never read" is now expected.

---

## Task 2 — Uniform-zone fast path in `ZonedReader::pruning_evaluation`

**File:** `vortex-layout/src/layouts/zoned/reader.rs` (~lines 136–207)

### Problem

The zone-level pruning mask is already cached per expression (`PruningState`), but every
split still pays, inside the returned `MaskFuture`:

1. a `zone_lengths` `Vec` allocation (built even before the future runs),
2. a `BitBufferMut` of the full split length filled via `append_n` per zone,
3. `Mask::from(builder.freeze())` — a popcount pass,
4. `mask.bitand(&stats_mask)`.

For the overwhelming majority of splits the covered zones are **uniform** — either none
pruned (typical un-prunable query) or all pruned (typical sorted-column range query) — and
all of that work reduces to a constant. On many-split scans (sf=10 lineitem is ~916
splits/iteration) this is pure per-split overhead. The v2 equivalent of this fix is the
`ConstantArray` fast path in `vortex-layout/src/plan/plans/zoned.rs` on this branch.

### Change

After resolving `pruning_mask` (zone-level `Mask`, `true` = zone pruned), inspect only the
zones covering this split (`zone_range`), before building anything:

- **No relevant zone pruned** → the stats mask would be all-true: skip the builder and the
  bitand entirely, and continue with `mask` unchanged (still forward to the data child
  evaluation as today).
- **All relevant zones pruned** → `stats_mask` is all-false: return `Mask::new_false(mask.len())`
  immediately (today's code already skips the data-child await when
  `stats_mask.all_false()`; this path just avoids materializing the buffer first).
- Otherwise fall through to the existing expansion code.

Uniformity check should be cheap: slice or `count_range` over the relevant zone bits (a
handful of bits per split) — do not scan the whole zone mask. Also move the
`zone_lengths` construction behind the non-uniform path so uniform splits allocate
nothing.

### Verification

- `lineitem_prune` (sf=1 and sf=10) is the pruning-heavy shape; `lineitem` is the
  no-pruning shape (falsifier exists, zero zones pruned — exercises the all-kept path).
  Expect a small v1 improvement at sf=10 (v1 baseline ~9.5–10.3ms on `lineitem_prune`),
  and strictly no regression elsewhere.
- Zoned reader unit tests in `zoned/reader.rs` cover mixed masks; add a case where the
  requested row range covers only pruned zones and only kept zones if not already present.

---

## Task 3 — Avoid redundant mask slices in `ChunkedReader`

**File:** `vortex-layout/src/layouts/chunked/reader.rs`

### Problem

All three evaluation paths slice the mask per chunk unconditionally:

- `pruning_evaluation` (~line 228): `mask.slice(mask_range)` on a resolved `Mask`
- `filter_evaluation` (~line 274): `mask.slice(mask_range)` on a `MaskFuture`
- `projection_evaluation` (~line 313): `mask.slice(mask_range)` on a `MaskFuture`

`MaskFuture::slice` allocates a new boxed + `Shared` future per call, and `Mask::slice`
copies mask state. In the common case — splits align with chunk boundaries, so exactly one
chunk covers the entire range — the slice is the identity and the allocation is wasted.
This happens once per chunk × per evaluation × per split.

### Change

In each of the three sites, when the computed `mask_range` covers the whole mask, pass a
clone instead of slicing:

```rust
let child_mask = if mask_range.start == 0 && mask_range.end == mask.len() {
    mask.clone()
} else {
    mask.slice(mask_range)
};
```

(`Mask::clone` / `MaskFuture::clone` are cheap Arc-style clones.) The v2 equivalent is in
`ChunkedPlan::execute` (`vortex-layout/src/plan/plans/chunked.rs`) on this branch.

### Verification

- Behavior-neutral by construction; run the full test protocol from Ground rules
  (chunked reader tests live at the bottom of the same file).
- Sanity benchmark: `lineitem_wide` at sf=1, v1 before vs after (expect equal-or-slightly
  better; this is a micro-optimization, so no regression is the bar).
