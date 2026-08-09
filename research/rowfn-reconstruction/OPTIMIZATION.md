<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn optimization guide

## Performance model

RowFn is fast when the hot loop contains only work that changes for each row. These operations can
stay outside the loop:

- Array and dtype dispatch.
- Decoding and downcasts.
- Batch-constant detection.
- Input length validation.
- Output allocation and array construction.
- Validity policy.
- Rich error construction.
- Work derived only from constant operands.

The design also gives LLVM concrete types and independent indexed lanes. A short row closure is not
enough by itself. The generic plumbing must disappear after monomorphization.

## Optimization history

### Stage 0: sink-only output

The first shared executor required every row function to write through a sink. This model supported
runtime-shaped output, but it hid the independence of primitive output values.

The checked primitive loop was slower for several wide integer types. Signed `i64` multiply was
about 29% slower than the baseline. Unsigned `u64` multiply was about 59% slower.

### Stage 1: owned output

The next design let the row closure return `(Output, Failure)`. Shared execution owned the final
store and reduced failure evidence.

This change improved wide integer multiplication, but it did not give LLVM a simple input source.
For example, `i32` multiplication remained about 18% slower in the measured matrix.

This stage proved that output ownership mattered. It also proved that output ownership alone was
not sufficient.

### Stage 2: typed indexed input

`IndexedElementTuple` added an all-varying source. A primitive pair becomes
`LaneZip<&[Left], &[Right]>`. Shared execution validates both lengths once and calls
`map_checked_into`.

This stage restored varying and nullable multiplication to approximately baseline performance. It
also removed hot bounds checks from the inspected production monomorphs.

The trait is separate from `ElementTuple`. Many element types do not have a contiguous source.
Stable Rust cannot combine a blanket fallback with a more specific primitive implementation
without specialization.

### Stage 3: remove the `Output: Copy` bound

The executor needs only one property from owned output: abandoning initialized spare capacity on
unwind must not leak a required destructor. `Output: Copy` was stronger than this property.

On Rust 1.91.0 and LLVM 21.1.2, adding the public `Copy` bound changed the production `i32` checked
multiply monomorph from about 18.7 microseconds to about 29.9 microseconds. An inert marker bound
did not cause the loss. One codegen unit did not remove it.

The selected design uses a compile-time `!needs_drop::<Output>()` assertion. It does not expose a
`Copy` bound that the executor does not need.

The exact compiler mechanism remains unknown. Standalone reduced loops did not reproduce the
effect. The real trait, closure, vector, and monomorphization context was necessary.

### Stage 4: preserve mixed-constant code placement

Commit `5c02036a2` deduplicated length validation:

```rust
let varying = Args::varying(&columns);
ensure_decoded_lengths(&columns, varying.as_ref(), row_count)?;

if let Some(varying) = varying {
    // All-varying loop.
} else {
    // Mixed loop.
}
```

This source-only change made constant add and subtract about 3.3 times slower at that revision. It
did not change the all-varying cases.

The selected form keeps the view and proof in the selected branch:

```rust
if let Some(varying) = Args::varying(&columns) {
    validate_varying_lengths(&varying, row_count)?;
    // All-varying loop.
} else {
    validate_mixed_lengths(&columns, row_count)?;
    // Mixed loop.
}
```

This change restored constant add and subtract to about 9.2 microseconds. Constant `i32` multiply
returned to about 18.9 microseconds. The all-varying controls did not move.

The source placement is a measured constraint for the current toolchain. Rust semantics do not
require it. The source ablation proves the performance relationship, but it does not identify the
LLVM pass that causes it.

The sink executors retain the shared validator. Moving their proof into each branch did not improve
the cosine or spatial benchmarks.

### Stage 5: typed tensor rows

The old tensor row accessor repeated a ptype check and buffer downcast for every output row. The
new `TensorRows<T>` representation performs these operations once during decode.

Each row access uses a typed flat buffer, width, and stride. A constant-backed tensor uses stride
zero, so `index * stride` selects row zero without a branch.

This representation makes the tensor inner loop ordinary slice arithmetic. It also keeps constant
input storage compact.

### Stage 6: prepared tensor and spatial constants

Prepared visits expose batch constants before the loop. Cosine similarity computes a constant norm
once. Spatial predicates compute constant bounding boxes and relation helpers once.

This optimization does not require a new array kernel. The same row declaration handles both
constant and varying operands.

## Source-placement constraints

### Decode before the loop

The `InputElement::decode` method must contain dtype checks, array execution, downcasts, and buffer
extraction. Calling these operations through `get` makes the loop pay batch work for every row.

### Prepare before the loop

`Args::constants` and the prepare closure run once after decode. The prepared value is borrowed by
the row closure. It must not be rebuilt for each row.

### Validate lengths before the loop

Unchecked input reads are sound only after each varying source proves that it contains
`row_count` rows. The output slice must also contain `row_count` slots.

The validations must execute before the loop. A check in the loop keeps bounds control flow in the
hot path and can prevent bounds-check elimination.

### Keep the owned varying proof in its branch

The owned executor must not pass `Option<&VaryingColumns>` through the shared generic helper on the
measured toolchain. The option construction, proof, and consumer stay in one branch.

This rule is intentionally narrow. Applying it to every executor adds duplication without measured
benefit.

### Borrow sink rows once

`sink.rows()` runs before the loop. The loop receives a stable row view instead of repeatedly
borrowing the sink object. This keeps the buffer descriptor and output shape invariant.

### Keep rich errors cold

The row closure computes a small failure word. A `#[cold]` and `#[inline(never)]` helper creates the
`VortexError` after the loop or on the immediate failure path.

This arrangement prevents formatting, allocation, and error branches from entering successful
checked-arithmetic loops.

### Use inlining evidence, not a blanket attribute

The public wrappers use ordinary `#[inline]` only where a caller must see captured constants or a
small adapter. The implementation does not apply `#[inline(always)]` to checked arithmetic.

The lane-kernel module contains small internal chunk helpers with stronger attributes. Those
helpers were measured as part of the pre-existing lane-kernel work. A new strong inlining attribute
requires separate assembly or benchmark evidence.

## Why the loop can autovectorize

The optimized all-varying primitive loop presents these facts to LLVM:

1. The element types are concrete because `dispatch` selected `T` before execution.
2. The input sources are typed slices or a typed `LaneZip`.
3. Input and output lengths match.
4. Unchecked reads follow one pre-loop proof.
5. Each iteration reads and writes an independent row.
6. Failure combines with bitwise OR.
7. The closure is concrete and can inline into the loop.
8. Error construction and validity are outside the loop.

The generated loop can use SIMD when LLVM has a legal and profitable lowering. Checked add and
small-width arithmetic often fit this model.

The word _autovectorize_ must not describe every result. The inspected `i64` and `u64` widened
multiply loops remained scalar on x86. They recovered performance because RowFn matched the
handwritten scalar loop, not because LLVM found SIMD.

The tensor outer loop returns one scalar for each tensor row. SIMD commonly appears in the inner
loop over each tensor slice. The outer RowFn loop does not need to vectorize across variable slice
references.

## Rejected or incomplete alternatives

### Keep every output behind a sink

This model supports more output shapes, but it loses the independent owned-value contract that
primitive code generation needs.

### Add a numeric `reduce_encoded` fast path

This path recovered speed by duplicating shared null and constant policy inside the numeric
function. It made RowFn a slow fallback instead of making shared execution fast.

### Add a numeric-specific visitor seam

This design moved the same specialization into generic execution under a different name. It did
not establish a reusable capability for nonnumeric row functions.

### Use safe zipped iterators

The tested iterator forms caused 3x to 9x losses for narrow integer types. They did not preserve the
same indexed source shape across all monomorphs.

### Depend on per-row bounds checks

Unchecked access improved some cases, but it did not solve the original output and source-shape
problems. It also regressed some `u8` cases when applied without the final indexed design.

### Scan output for failures

The selected loop returns failure evidence directly. Scanning a finished output adds another pass
and cannot represent every error condition.

### Use `Copy` as the no-drop proof

`Copy` is stronger than required and triggered a measured compiler regression. The compile-time
no-drop assertion expresses the actual safety condition.

### Apply branch-local validation to sinks

This change did not improve cosine or spatial performance. The shared helper remains in those
paths.

## Unrelated benchmark movement

An unrelated benchmark can move after a RowFn source edit even when it never calls RowFn. The
source edit rebuilds `vortex-array` and the benchmark executable. This rebuild can change:

- Codegen-unit partitioning.
- Inlining decisions in affected monomorphs.
- Function order and address alignment.
- Instruction-cache and decoded-instruction-cache set placement.
- Branch target placement.
- Linker layout of code that remains reachable through the shared session.

These are code-generation dependencies, not semantic dependencies.

[CodSpeed CPU simulation] measures executed instructions and models cache and memory access. It
can therefore report a different result when the instruction sequence or binary layout changes.
Local wall time can differ from the simulated ratio because it uses a real AMD processor instead
of the CodSpeed CPU model.

CodSpeed documents [function alignment] as one reason an unchanged microbenchmark can move after
a rebuild. The correct diagnostic is the simulated instruction and cache counts in the
differential flame graph.

An unrelated recovery does not prove that an algorithmic problem was fixed. The result is stable
only after source ablation, machine-code inspection, and repeated measurements agree on a cause.

## Current `take_filter_list` evidence

The [CodSpeed check at `4c936447a`] reports 31 regressions. Several `take_filter_list_*`
benchmarks are 14% to 16% slower than develop in CPU simulation.

The [CodSpeed check at `892717f30`] already reported the same benchmarks as 15% to 16% slower. The
final mixed-constant fix did not bring them back. Most of their simulated times improved by less
than 2% between the two checks. The fix removed larger constant-arithmetic regressions, so the
unchanged take/filter entries became more prominent in the ordered report.

Every retained RowFn CodSpeed summary from `0e5c19c00` through `4c936447a` that contains a
performance table also contains `take_filter_list_*` regressions. Some GitHub views show only the
20 largest changes, and the bot edits one current PR comment. Either behavior can make a persistent
regression appear to leave and return.

The compared list, filter, and take source files are identical between develop and the branch.
The measured benchmark has no runtime call to RowFn. Therefore, the change is not an algorithmic
regression in list take or filter execution.

AVX2 wall-time runs on CPU 4 provide a separate native-runtime observation:

| Revision | Typical list/filter median | Difference from develop |
| --- | ---: | ---: |
| Develop `66d096b5d` | 6.2 to 7.0 us | Baseline |
| Before latest push `892717f30` | 7.9 to 8.8 us | About 25% to 31% slower |
| Latest push `4c936447a` | 8.0 to 8.9 us | About 25% to 31% slower |

The latest push changes most local cases by only 0% to 2%. The branch already contains a native
wall-time gap before that push. This result does not explain the CodSpeed simulation result.

Changing the bench profile from 16 codegen units to one did not remove the native gap. For one
representative case, the candidate and develop medians were 7.86 and 6.21 microseconds. The same
case measured 8.25 and 6.41 microseconds with 16 codegen units.

The main filter-take and list-take function sizes are identical across the three AVX2 binaries.
Normalized disassembly of the list-take function has the same instructions. Relative addresses and
link layout differ. This native evidence points to linked-code layout or a called function outside
the compared symbol. It does not identify a specific cache or branch mechanism. The CodSpeed
differential flame graph and its instruction and cache counters are the correct evidence for the
simulation result.

Do not fix this result with arbitrary padding or an unrelated source edit. Such a change can move
the report without removing the cause.

## Current unresolved work

- Reduce the mixed-constant LLVM sensitivity while preserving the production monomorph.
- Identify the linked-code cause of the list/filter wall-time gap.
- Identify the spatial `envelope` regression that begins when numeric RowFn code enters the linked
  binary.
- Compare current CodSpeed flame graphs for list/filter and `envelope` against develop.
- Repeat the key results on a second compiler version before filing a compiler issue.

[CodSpeed check at `4c936447a`]: https://github.com/vortex-data/vortex/runs/93181527671
[CodSpeed check at `892717f30`]: https://github.com/vortex-data/vortex/runs/93169735961
[CodSpeed CPU simulation]: https://codspeed.io/docs/instruments/cpu
[function alignment]: https://codspeed.io/docs/instruments/cpu/regression-causes#function-alignment
