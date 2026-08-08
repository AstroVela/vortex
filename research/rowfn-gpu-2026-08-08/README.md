<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn GPU execution research

`RowFn` can drive GPU execution for new scalar functions, and the evidence is stronger than
expected. A replica of the `visit_prepared_deferred` shape compiles to correct PTX for
`nvptx64-nvidia-cuda` with the generic trait machinery fully erased, producing a loop body that
matches what a hand-written CUDA kernel emits for the same operation.

The rest of the executor has a working precedent in the repository. `filter_primitive` in
`vortex-cuda/src/kernel/filter/primitive.rs` is already a compute kernel over an already-canonical
array: it destructures to a device buffer handle, launches, handles validity separately on the host,
and rebuilds through `PrimitiveArray::from_buffer_handle`. That is the shape a GPU `RowFn` executor
needs, and it splits values from validity the same way `lift.rs` does.

FoR decode is the existing proof that the shape runs. `ForOp<T>{reference}` plus `scalar_kernel` is
`add(column, constant)` with `prepare` and `apply` split exactly as `RowFn` splits them, so a row
body already executes on the GPU, written in C++ by hand. Compiling the same row body from Rust
reproduces its instruction sequence.

What this covers is the elementwise fixed-width family, not every CPU scalar function. The columnar
family does not need `RowFn`, and the variable-work family is hard on a GPU regardless.

One gap in the generated code is real. A GPU thread issues one 8-byte load where the CUDA kernels
ask for 16, because the executor visits one row per iteration and `&[T]` hides the buffer's real
alignment. No compiler flag changes it, and on x86 the loop vectorizer makes the same shape free, so
nothing has needed it until now. It is a known gap with a known fix rather than a blocker, and
nothing here shows the wider access is worth anything.

The remaining blockers are the element types that are not flat buffers, the null-strategy
thresholds, and the toolchain.

## Scope and limits of this record

No GPU and no CUDA toolkit were available on the research machine. `nvcc` and `nvidia-smi` are both
absent.

- The PTX in [`codegen`](codegen) is real `rustc` output, emitted for `nvptx64-nvidia-cuda`. It was
  not assembled by `ptxas` and it was not executed.
- **This record contains no timings.** Every performance statement below is either arithmetic from
  published bandwidth figures, labeled as such, or a structural claim about memory traffic. None of
  it is a measurement, and none of it substitutes for one.

## The layering this question sits in

Two separate concerns meet on the GPU, and conflating them produces the wrong answer.

**Decoding an encoded array is the encoding implementor's job.** `hybrid_dispatch` collapses an
encoding tree into one launch, falling back through standalone kernel, full fusion, partial fusion,
and per-node execution. Its output is a canonical array. Writing fused decode kernels belongs to
whoever adds the encoding.

**Computing a scalar function over canonical arrays is a separate layer, and it does not exist
yet.** `CudaExecute::execute(&self, array: ArrayRef, ctx) -> Canonical` is per-array and returns
canonical, so it is an encoding trait. Nothing in `vortex-cuda` takes N canonical input arrays and
produces one output array. The only scalar function reachable on the GPU today is `Cast`, and it is
special-cased inside the plan builder because it appears inside encoding trees.

The dispatch plan cannot be extended to cover the second layer. `PackedStage` holds a single
`input_ptr`, `PlanHeader` a single `output_ptype`, and every `ScalarOp` variant (`FOR`, `ZIGZAG`,
`ALP`, `DICT`, `CAST`) is unary over one register array. There is nowhere to put a second operand,
so a binary function such as `hypot(x, y)` has no representation in it. A `RowFn` stage is not a new
`ScalarOp`.

The layering is therefore the same as on the CPU. Decode, then compute. A `RowFn` on the GPU
receives canonical device arrays and never touches the decode path.

## The row body compiles to PTX today

[`row-body-replica-rs.md`](codegen/row-body-replica-rs.md) reproduces the
`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs` structure: a `CheckedPrimitiveOp` trait with
an associated `Failure` type, generic monomorphization over the primitive width, a `prepare` step
producing batch state, and an `apply` closure passed as `impl Fn` to a generic row loop.

`rustup target add nvptx64-nvidia-cuda` succeeds on the repository's pinned 1.91.0 stable toolchain
and ships `core`, `alloc`, and `compiler_builtins`. Codegen itself succeeds on stable. Declaring the
kernel entry point needs `feature(abi_ptx)`, so the emitted PTX below used nightly.

The generated main loop for `numeric_add_i64`:

```ptx
$L__BB1_2:
	ld.global.b64 	%rd13, [%rd12];        // lhs
	ld.global.b64 	%rd15, [%rd14];        // rhs
	add.s64 	%rd16, %rd15, %rd13;
	xor.b64 	%rd17, %rd16, %rd13;       // branchless overflow evidence
	xor.b64 	%rd18, %rd16, %rd15;
	and.b64 	%rd19, %rd17, %rd18;
	shr.u64 	%rd20, %rd19, 63;
	st.global.b64 	[%rd21], %rd16;
	or.b64 	%rd24, %rd20, %rd24;           // failure accumulates in a register
	setp.lt.u64 	%p2, %rd23, %rd7;      // loop control is the only branch
	@%p2 bra 	$L__BB1_2;
```

Four properties matter:

1. The trait, the associated type, the generic `row_loop`, and the `impl Fn` closure all
   monomorphize away. Nothing survives into the loop. This is the same result the x86 audit
   recorded, from the same cause.
2. The only branch in the loop is loop control. The deferred failure word stays in a register and
   reduces once after the loop.
3. `prepare` constant-folds. In `scale_f32_by_constant` the prepared `2.0 * PI` becomes the
   immediate `0f40C90FDB`, and the always-zero failure word is eliminated entirely.
4. The checked-arithmetic overflow evidence costs four ALU instructions and no branch. On a
   bandwidth-bound kernel those instructions are free.

The design decisions that made `RowFn` autovectorize on x86 are the same ones that make it
SIMT-friendly. A branchless body, an OR-reducible failure word instead of a per-row `Result`, and
`const { assert!(!needs_drop::<O>()) }` forcing POD outputs are all GPU requirements that the CPU
work already paid for.

Two gaps in the generated code:

- The loads are scalar `ld.global.b64` rather than the 16-byte access `scalar_kernel.cuh` uses. This
  reproduces under the tiled loop shape as well, and is treated as its own finding below.
- The replica reduces the failure word per thread and then has every failing thread store `1`. The
  stores are benign because the value is identical, but a real implementation needs `atomicOr` or a
  block reduction.

<details>
<summary>Full replica source, emitted PTX, and reproduction</summary>

Sources: [`row-body-replica-rs.md`](codegen/row-body-replica-rs.md),
[`row-body-replica-ptx.md`](codegen/row-body-replica-ptx.md).

```bash
rustup target add nvptx64-nvidia-cuda
rustc +nightly --edition 2021 --crate-type cdylib \
  --target nvptx64-nvidia-cuda -O --emit=asm -o out.ptx src/lib.rs
```

The emitted module declares `.version 7.0`, `.target sm_70`, `.address_size 64`, and exposes both
kernels as `.visible .entry`. Building through `cargo` instead reaches codegen and then fails at the
`llvm-bitcode-linker` component, which was not installed. The object file for the target is produced
regardless, so the failure is in linking and not in compilation.

</details>

## FoR decode is the existing proof

No scalar function executes on the GPU today. `Cast` is the only one reachable, and only because the
plan builder special-cases it inside encoding trees.

FoR decode is the closest thing, and it is closer than it looks. `for.cu` is:

```cpp
template <typename T> struct ForOp {
    T reference;
    __device__ inline T operator()(T value) const { return value + reference; }
};
scalar_kernel(input, output, array_len, ForOp<Type>{reference});
```

That is `add(column, constant)`. `ForOp<T>{reference}` is `prepare` hoisting the constant operand,
`operator()` is `apply`, and `scalar_kernel` is the executor. The same operation on the CPU is the
existing `NumericBinary` `RowFn` at `Add` with a constant right-hand side. A `RowFn` row body
already executes on the GPU, written in C++ by hand.

[`for-decode-rs.md`](codegen/for-decode-rs.md) writes that row body as Rust generics and compiles it
for `nvptx64`. The grid-stride loop:

```ptx
$L__BB1_2:
	ld.global.b64 	%rd12, [%rd11];
	add.s64 	%rd13, %rd12, %rd5;    // %rd5 is the reference, hoisted before the loop
	st.global.b64 	[%rd14], %rd13;
	setp.lt.u64 	%p2, %rd16, %rd6;
	@%p2 bra 	$L__BB1_2;
```

The `RowOp<T>` trait, the generic executor, and the constant hoist all erase. The reference is
loaded once into a register and reused, which is what `ForOp<T>{reference}` does. This is the
instruction sequence `scalar_kernel` produces for the same operation.

## Vector loads: the Rust path does not emit them

The tiled kernel in [`for-decode-rs.md`](codegen/for-decode-rs.md) mirrors `scalar_kernel.cuh`
exactly: 2048 elements per block, 64 threads, `VALUES_PER_LOOP = 16 / size_of::<T>()`. The unroll
works and the addressing is adjacent, but the loads stay scalar:

```ptx
	ld.global.b64 	%rd15, [%rd14];
	add.s64 	%rd16, %rd15, %rd5;
	ld.global.b64 	%rd17, [%rd14+8];
	add.s64 	%rd18, %rd17, %rd5;
	st.global.b64 	[%rd19], %rd16;
	st.global.b64 	[%rd19+8], %rd18;
```

No `ld.global.v2.b64` appears anywhere in either module.

### It is not a flag

`opt-level` 2 and 3, crossed with the default target, `sm_80`, and `sm_90`, all emit zero vector
loads for the scalar-access kernel. Six combinations, one result.

### It is not missing alignment metadata

[`vectorization-probes.md`](codegen/vectorization-probes.md) isolates the cause with four kernels
that differ only in what the compiler is told:

| Probe | Emitted |
| --- | --- |
| plain `*const i64` | `ld.global.b64` x2 |
| `assert_unchecked(ptr % 16 == 0)` | `ld.global.b64` x2 |
| `#[repr(align(16))] Pair([i64; 2])` | `ld.global.v2.b64` |
| `#[repr(align(16))] Quad([i32; 4])` | `ld.global.v4.b32` |

Asserting 16-byte alignment on the base pointers changes nothing. `core::hint::assert_unchecked`
emits an `llvm.assume`, which does not reach the alignment analysis the LoadStoreVectorizer uses.

The pass does merge adjacent scalar accesses when alignment is provable from the *type*. Reading the
two lanes as separate scalar field accesses through a `#[repr(align(16))]` pair still produces
`ld.global.v2.b64`:

```rust
let a = (*input.add(i)).0[0];
let b = (*input.add(i)).0[1];
(*output.add(i)).0[0] = a + reference;
(*output.add(i)).0[1] = b + reference;
```

So the merging is not the obstacle. Two conditions have to hold together, and the current shape
misses both:

1. **One iteration must touch adjacent elements.** The LoadStoreVectorizer merges accesses within an
   iteration. It does not restructure a loop. A grid-stride or block-tiled loop visits elements
   `stride` apart, so consecutive iterations are never adjacent and there is nothing to merge.
2. **The pointer must carry provable alignment.** `Alignment::DEFAULT_ALIGNMENT` is 256 bytes, so
   Vortex buffers are over-aligned far beyond the 16 a vector access needs, but `&[T]` erases that
   to `align_of::<T>()`. The alignment is real and invisible.

The earlier tiled kernel satisfied the first and not the second, which is why it stayed scalar.

### Why x86 does not have this problem

`InputElement::get(column, index) -> Elem` reads one element, and under it
`IndexedSource::get_unchecked(&self, i) -> Item` reads one lane.
`vortex-compute/src/lane_kernels/source.rs` states why: the trait exists in that shape "so that lane
reads carry no inter-iteration data dependency, the autovectorizer treats each lane independently".

That works on x86 because LLVM's **loop vectorizer** restructures the loop, turning one element per
iteration into four or eight in a SIMD register. It supplies both conditions above by itself, so the
one-element API costs nothing.

No such pass runs for the GPU, and none is missing. On a GPU the lane parallelism already exists
across the warp's threads, so there is nothing for a loop vectorizer to create. The only remaining
question is how many bytes one thread moves per instruction, and the LoadStoreVectorizer answers
that only for accesses that are already adjacent inside one iteration. A one-row-per-iteration loop
never presents any.

This is the same conclusion the CUDA code reached. `scalar_kernel.cuh` fixes
`VALUES_PER_LOOP = 16 / sizeof(InputT)`, and `dynamic_dispatch.cu` applies each scalar op to
`VALUES_PER_TILE` values in registers. Both write the loop N-at-a-time on purpose.

### Which half is the executor's and which is the API's

The two conditions have different owners, and only one of them reaches the trait.

**Adjacency belongs entirely to the executor.** `execute_row_output_prepared` already owns the loop,
and `A::get(&columns, i)` and `A::get(&columns, i + 1)` are both already callable. Visiting two rows
per iteration is a local change with no trait involvement.

**Alignment cannot be solved from the executor alone.** The executor is generic over
`InputElement`, so it cannot reinterpret an opaque `Varying<'a>`. Moving the alignment into the
element's own `Varying` type does not rescue it either, and the reason is worth recording because it
rules out the whole category. Two attempts, both keeping today's
`get_varying(&Varying, row_index) -> Elem` signature and putting a `#[repr(align(16))]` pair behind
the pointer:

1. Index by row with `i = tid() * 2`, incrementing by `stride() * 2`. The `i / 2` and `i % 2`
   inside the accessor fold away correctly, but LLVM flattens the address to `base + i * 8` and
   then cannot prove `i` is even, because `stride()` is opaque.
2. Index by chunk with `i = c * 2`, which makes `i` provably even at the source level. This also
   fails. LLVM strength-reduces the loop to a running byte offset incremented by an opaque stride,
   and the evenness is gone by the time the load is selected.

Both emit `ld.global.b64` at `[%rd10]` and `[%rd10+8]`: adjacent, correct, and unmerged.

The working shape indexes a pointer whose *pointee* is the 16-byte chunk. Then `ptr.add(c)` has a
type-derived stride of 16, so the alignment holds for any index and survives every loop transform.
That is a property of the access granularity, which the element owns, not of the loop, which the
executor owns.

So adjacency needs no API change and alignment needs a small one: a chunk-granular accessor on
`InputElement`, defaulting to one lane so every existing element is unaffected.

### What closing it would take

[`chunked-executor.md`](codegen/chunked-executor.md) models the proposed change: the element type
names how many values it reads as one aligned aggregate, and the executor applies the row body once
per lane over registers. The row body stays an ordinary scalar closure, and the executor stays
generic over both. `chunked_for_i64` with the body `|r, v| v + *r` emits:

```ptx
$L__BB2_2:
	add.s64 	%rd10, %rd1, %rd16;
	ld.global.v2.b64 	{%rd11, %rd12}, [%rd10];
	add.s64 	%rd13, %rd11, %rd6;
	add.s64 	%rd14, %rd12, %rd6;
	add.s64 	%rd15, %rd2, %rd16;
	st.global.v2.b64 	[%rd15], {%rd13, %rd14};
	add.s64 	%rd17, %rd17, %rd4;
	add.s64 	%rd16, %rd16, %rd5;
	setp.lt.u64 	%p2, %rd17, %rd3;
	@%p2 bra 	$L__BB2_2;
```

One vector load, one row body per lane, one vector store, strength-reduced increments, and loop
control. `chunked_affine_i32` at `|s, v| v * *s + 7` emits `ld.global.v4.b32` with four
`mad.lo.s32`, so this is not one lucky shape.

A default of one lane per chunk preserves the current behavior exactly, which keeps the change
additive and leaves the x86 results untouched.

This is a documented gap with a known fix, not a change to make now. Nothing here shows that the
wider transaction is worth anything. A warp already coalesces 32 scalar loads into the same memory
transactions, so the vector access buys instruction count and memory-level parallelism rather than
bandwidth, and whether that is measurable on a bandwidth-bound kernel is exactly what has not been
tested. The useful position is knowing where the gap is and that closing it does not disturb any row
body, so it can wait for a profile that asks for it.

### The chunked path is an x86 question too

[`x86-chunked.md`](codegen/x86-chunked.md) compiles the same two shapes for x86-64. Under AVX-512
the chunked shape reaches `zmm` where the one-lane-at-a-time loop stays on `ymm`, and its loads
become `vmovdqa` rather than `vmovdqu`. Under AVX2 both reach `ymm` and the difference is small.

Do not read that as a win yet. LLVM's preference for 256-bit registers is frequently a deliberate
heuristic rather than a missed optimization, and forcing 512-bit is slower on parts that downclock
for it. Zen 4 is not one of those parts, so the question is worth settling on the bench machine.

The checked `i64` multiply does not vectorize under either shape, since x86 has no vector 64-bit
multiply producing overflow evidence. Chunking only unrolls it further. That is consistent with
`mul_i64` and `mul_u64` being the shapes the x86 record had the most trouble with, and it means the
chunked path is not a fix for them.

So the addition is not GPU-specific. It is an access-shape question that the GPU forced into view
because a GPU has no autovectorizer to hide it.

### One artifact of the replica

Reading a special register through `core::arch::asm!` leaves the read inside the loop, because plain
inline assembly is not hoistable. Adding `options(pure, nomem, nostack)` hoists it and produces the
loop shown above. This is a property of the test harness, not of `RowFn`, but any real
implementation needs it or the `core::arch::nvptx` intrinsics.

## Which scalar functions this actually covers

`RowFn` on the GPU covers the elementwise fixed-width family and does not cover the rest. The
boundary is predictable and it mostly matches the boundary `RowFn` already draws on the CPU.

**Elementwise fixed-width.** Numeric `binary`, comparison, `between`, `case_when`, `fill_null`, and
`cast`. These port, and they are the reason to do the work at all.

**Columnar and bitmap.** `not`, `is_null`, `is_not_null`, `mask`, `list_length`, and `byte_length`.
`RowFn` excludes these by design, and they do not need it. They are straightforward hand-written GPU
kernels that arrive through a different path.

**Zero-copy and structural.** `ext_storage`, `get_item`, `pack`, `select`, and `literal`. No kernel
on either device.

**Variable work per row.** `like`, `list_contains`, `list_sum`, `variant_get`, and the geo
predicates. Warp divergence rules these out, and `RowFn` already excludes most of them on the CPU
for related reasons.

**Variable-length output.** String transforms and other `VarBinView` sinks. Not reachable without a
two-pass count-then-fill or an atomic bump allocator.

So porting every CPU scalar function to the GPU through `RowFn` is not the outcome. The accurate
statement is narrower and still worth having. The elementwise fixed-width family comes essentially
for free once the executor exists, that family is the largest and the most used, the columnar family
is easy on a GPU but arrives through a different path, and the variable-work family is hard on a GPU
whether or not `RowFn` is involved.

## What the executor looks like

`scalar_kernel.cuh` is already a generic elementwise executor parameterized by a functor, and its
split matches `RowFn` one-to-one:

| `RowFn` | `vortex-cuda` |
| --- | --- |
| `prepare(A::ConstElems) -> P` | `ForOp<T> { reference }`, the functor's captured state |
| `apply(&P, elems) -> O` | `T operator()(T value) const` |
| `RowFn::dispatch` with `match_each_native_ptype!` | `FOR_EACH_INTEGER(GENERATE_FOR_KERNEL)` |
| `execute_row_output_prepared` | `scalar_kernel<InputT, OutputT, Op>` |
| `OutputElement` with no drop glue | the POD output buffer |

`for.cu` uses that executor for a decode kernel, but the executor itself is not decode-specific. The
correspondence means a GPU row executor is a second implementation of a split the codebase already
has, not a new concept.

The three framework steps map onto existing device machinery:

- `A::decode` becomes a destructure of each canonical input into a device buffer handle and a
  length, as `filter_primitive` does with `PrimitiveDataParts`. For flat primitive elements this is
  cheaper than the host decode, not more expensive, because there is nothing to canonicalize.
- The row loop becomes one kernel launch, monomorphized on the same element tuple that
  `RowFn::dispatch` already selects.
- `O::build` becomes `PrimitiveArray::from_buffer_handle`, which already exists and already accepts
  a device `BufferHandle`.

Validity needs no new design. `filter_primitive` computes `validity.filter(&mask)` separately from
the values kernel, and `execute_validity_cuda` handles the array case. That is the same separation
`lift.rs` makes when it applies input validity to the output after the row loop.

Constant operands keep their value. `ArgColumn` collapses a constant to one decoded row read at
index 0, which on a GPU is a broadcast from a kernel argument instead of a column of global memory
traffic. That saves more on a bandwidth-bound device than it does on the CPU.

## What does not port

`visit_prepared_deferred` is the GPU-portable visit: indexed tuples, POD output, OR-reducible
failure, no per-row control flow. The parts that do not port are specific.

- `InputElement::get` returns a GAT borrow. For a primitive that lowers to a load. For an element
  that follows an offset or parses bytes it becomes data-dependent addressing, which is warp
  divergence.
- `OutputSink` with `Row<'_>`, shared builders, and `SinkResult::accumulate` returning
  `VortexResult` is the least portable piece. A `VarBinView` sink needs a two-pass count-then-fill
  or an atomic bump allocator on device, and `accumulate` reintroduces a per-row error path.
  `visit_prepared_into` therefore does not port in general, though a fixed-width sink such as a
  tensor output is tractable.
- The geo predicates are the worst case. `contains` and `intersects` do data-dependent work per row
  behind a bounding-box early-out. The early-out saves nothing on a GPU, because a warp executes
  both sides of the divergence, and the exact predicate walks variable-length geometry. These stay
  on the CPU.

## The null-strategy cost model inverts

`lift.rs` picks between `Dense`, `DenseWithRetry`, `Filter`, and `BranchAndSkip` using
`ONE_DECODE_BRANCH_MIN_SURVIVING_FRACTION` and `MULTI_DECODE_BRANCH_MIN_SURVIVING_FRACTION`. Those
thresholds are x86 measurements and none of them transfer.

- `BranchAndSkip` saves nothing on a GPU. The warp executes the skipped lanes anyway, so the branch
  buys divergence and no work reduction.
- `Filter` costs a full stream compaction pass, which is a real technique with a real cost.
- `Dense` is almost always correct on a GPU, because the compute it wastes is free against the
  memory traffic it avoids.

The strategy *set* is shared and the *policy* is device-specific. `lift.rs` already separates the
two, so this is a second policy implementation rather than a redesign. `DENSE_SAFE` keeps its exact
meaning and becomes more valuable, since it is what admits the strategy a GPU wants by default.

## Performance against a hand-written kernel

The comparison that matters is a `RowFn`-generated kernel against a hand-written CUDA kernel for the
same new scalar function, both over canonical device arrays.

On that comparison the PTX above is the answer for the arithmetic. The abstraction erases
completely, the loop body is what a hand author would write, and an elementwise scalar function is
bandwidth-bound, so the ALU work the row body performs is not the limiting factor. The FoR
comparison makes this concrete: the Rust row body produces the same instruction sequence as the
functor `scalar_kernel` already runs for that operation.

The one identified gap is memory access width, and it is an API gap rather than a compiler one. The
current element API reads one value at a time, which is right for the CPU autovectorizer and cannot
express a wider transaction for a GPU. Adding a chunked access path recovers `ld.global.v2.b64` and
`ld.global.v4.b32` without changing any row body. How much the difference is worth still needs
`ptxas` and a GPU.

Two costs sit outside the kernel and both are worth stating.

**Residency.** At 100M `i64` rows a column is 800 MB. PCIe 4.0 x16 moves roughly 25 GB/s in
practice, so a host round trip for one column costs about 32 ms, against roughly 1.2 ms of device
memory time for a binary operation reading two columns and writing one at 2 TB/s. A GPU scalar
function only pays inside an already device-resident pipeline. `device_read_at.rs` and
`pooled_read_at` mean that pipeline exists.

**Expression-level fusion.** Each `RowFn` is one launch and one round trip through global memory, so
`hypot` decomposed into a tree of numeric expressions costs one round trip per node. This cost
belongs to the scalar-function layer rather than to the encoding layer. It is also not a
GPU-specific objection to `RowFn`, because the CPU path round-trips through memory per node in
exactly the same way. It is an argument for expression-level fusion in general, and the GPU raises
the stakes rather than changing the shape of the problem.

## Two routes to a shipped kernel

The access-width finding does not block GPU execution. The scalar-load kernels are correct, and a
warp coalesces their accesses into the same memory transactions the vector form uses, so the gap is
per-thread instruction count with an unmeasured cost. The open question is not whether a `RowFn`
row body can run on the GPU but which build route produces the kernel.

Neither route asks the compiler to decide where code runs. Vortex owns dispatch, as
`filter_primitive` does today, and the compiler only has to produce the kernel body.

**Direct compilation.** `rustc` compiles the row body for `nvptx64-nvidia-cuda` and the build embeds
the PTX exactly as `build.rs` embeds `nvcc` output today. Every codegen question this record raises
is answered for this route: the trait machinery erases, `prepare` constant-folds, the failure word
stays in registers, and the chunk-granular accessor closes the access-width gap where a profile
asks for it. The costs are a nightly toolchain for the device crate alone, since
`extern "ptx-kernel"` is feature-gated while the host workspace stays on 1.91.0, the
`llvm-bitcode-linker` rustup component, and tier 2 target risk. This is the only route that
delivers one source for both targets, which is the property that makes `RowFn` worth extending to
the GPU at all.

**Codegen.** A Rust generator emits CUDA C functors and `nvcc` compiles them, which is the pattern
`bit_unpack_gen.rs` already ships for the FastLanes kernels. This route adds no toolchain
requirement, because `vortex-cuda` already requires `nvcc`, and it inherits nvcc's optimizer,
including whatever vectorization `#pragma unroll` recovers. Its limit is the source of truth.
Arbitrary Rust row bodies cannot be transpiled, so the body either lives in a restricted form that
a macro or generator can print in both languages, or it is written twice with differential tests
keeping the pair honest. `RowFn` remains the specification either way, but only as a contract, not
as the single implementation.

The milestone is the same under either route: `NumericBinary` at `i64` over canonical device
arrays, compared against `for.cu`, whose column-plus-constant case computes the same operation.
Only the provenance of the functor differs between the routes, so both can be evaluated against
that one milestone on a single GPU machine, and the abandoned route still validates the other's
output.

## What remains between the PTX and production

Compiling the row body is the finished part, not the whole part. The remaining work is host
plumbing, and all of it is enumerable. Most of it is the `filter_primitive` shape applied again.

**Mirroring monomorphizations across two compilations.** The enumeration itself is not new.
`match_each_native_ptype!` inside `dispatch` already names every supported width, and rustc
compiles every arm of that match into the host binary, so the CPU pays the same enumeration today
and generic code never executes on either target. What changes is where the list lives. The host
match cannot cause nvptx codegen, because the PTX module is a separate compilation, so the same
width list has to appear a second time in the device crate, one `extern "ptx-kernel"` entry per
signature, plus a host table from the dispatched signature to the entry name. One macro can stamp
out both sides from one list. Drift between the two is a silent CPU fallback rather than an error,
which is why the sync belongs in a test: `dispatch` must be a pure function of options and dtypes,
so a probe visitor can enumerate every signature `dispatch` can select and assert that each one
has an entry or a recorded exclusion.

The sealed `RowVisitor` is the insertion point for all of this. A GPU executor is a second
framework-owned visitor that runs the same `dispatch`, the same `prepare`, and the same
`finish_failure` on the host, and replaces only the row loop with a launch of the named entry. Two
constraints fall out. The row computation must be a nameable function whose only batch state is
the prepared `P`, because a device entry cannot see host closure captures, and `NumericBinary`
already satisfies this by delegating to `Operator::apply`. And a third-party `RowFn` still ships
its own PTX, since no host compilation can produce device code for it.

**`prepare` output becomes kernel parameters.** The prepare step still runs on the host, once per
batch, but its output crosses the launch boundary, so the prepared state must be plain data. `()`
for the numeric operators and a precomputed norm for cosine both qualify. A prepared state holding
references or heap structure does not, and such a function stays on the CPU.

**The executor is `CudaExecute` for the `ScalarFn` encoding.** Scalar functions already appear in
an array tree as the lazy `ScalarFn` encoding, and `vortex-cuda` executes encodings through
per-encoding `CudaExecute` implementations, so the integration point exists and
`try_gpu_dispatch` already routes trees through it. The implementation is the `filter_primitive`
shape: verify every input is canonical and device-resident, destructure to buffer handles, look up
the entry, launch, read the failure word back, and rebuild through `from_buffer_handle` with
conjoined validity.

**Fallback needs a residency policy.** A dispatch with no GPU entry has to run on the CPU, and the
wrong way to do that is copying device inputs back across PCIe. Whether an unsupported function
forces its subtree onto the CPU or pays the copy is a planner decision, and it is the same
decision `hybrid_dispatch` already makes for unfusable encodings.

**Failure words and retry.** `visit_prepared_deferred` reduces per thread. The device kernel needs
an `atomicOr` into a device word, one copy back after the launch, and the same deferred-error
mapping `lift.rs` applies, including the valid-only retry.

**The build must degrade.** `build.rs` already produces an empty PTX table when `nvcc` is absent,
and the session falls back to the CPU. The rustc route needs identical behavior when the nightly
toolchain or `llvm-bitcode-linker` is missing, plus a CI job that builds the device crate so the
fallback stays a choice rather than an accident.

None of these blocks the milestone. `NumericBinary` at `i64` needs one table entry, `()` prepared
state, one `atomicOr`, and the executor, which is why it is the first step.

## What else this knowledge is good for

The result that Rust generics, trait dispatch, and closures survive to clean PTX is not confined to
scalar functions. Four consequences are worth separating from the `RowFn` question.

**Encodings are the larger prize.** `vortex-cuda/src/bit_unpack_gen.rs` is Rust that consumes the
`fastlanes` crate and writes CUDA C source strings, which `build.rs` then feeds to `nvcc`. The
FastLanes logic is already Rust, and the build transliterates it into another language to reach the
GPU. Compiling the Rust to PTX directly removes that step. Encodings are also where `vortex-cuda`'s
throughput work actually is, so the payoff is larger than the scalar-function layer, and the
mechanism is identical.

**Agreement between CPU and GPU becomes structural.** Today a kernel exists twice, once in Rust and
once in CUDA C++, and nothing but tests keeps the two honest. One source compiled for two targets
makes divergence impossible rather than merely detectable. That matters most for the encodings,
where the decode has to be bit-exact.

**A second execution target is a forcing function for the API.** The access-width gap was invisible
while x86 was the only target, because the loop vectorizer covered for it. Any further execution
target is likely to expose API assumptions the same way.

**Buffer alignment is invisible to the compiler.** Vortex allocates at
`Alignment::DEFAULT_ALIGNMENT`, which is 256 bytes, and then hands the data out as `&[T]`, which
promises only `align_of::<T>()`. Every downstream optimization that depends on alignment has to
rediscover it or go without. Asserting the alignment with `core::hint::assert_unchecked` does not
recover it, because that does not reach the analysis the LoadStoreVectorizer uses. Only the type
does. That applies to any Vortex kernel where access width matters, not only to the GPU.

## Toolchain and practicality

Rust-to-PTX is no longer the risky part, but it is not free either.

- `nvptx64-nvidia-cuda` is a tier 2 rustc target. Rust 1.97 raised the baseline PTX ISA and GPU
  architecture, which fixed defects where valid Rust triggered compiler crashes or miscompilations.
- NVIDIA released [cuda-oxide](https://github.com/NVlabs/cuda-oxide) in May 2026, a rustc codegen
  backend compiling `#[kernel]` Rust to PTX through MIR, Pliron IR, then LLVM. It supports generic
  kernels with monomorphization and closures with captures, and its own `map<T: Copy, F>` example is
  close to the `RowFn` shape. It is explicitly alpha.
- [Rust-CUDA](https://github.com/Rust-GPU/rust-cuda) with `rustc_codegen_nvvm` is the older route
  through NVIDIA's NVVM.
- [CubeCL](https://github.com/tracel-ai/cubecl) reaches CUDA, ROCm, Vulkan, and Metal from one
  kernel, at the cost of a restricted DSL rather than arbitrary Rust.

Four practical costs, in rough order of how much work they are:

1. Row bodies currently sit beside host code that uses `ArrayRef`, `ExecutionCtx`, and
   `vortex_error`. Splitting them into `no_std` POD-only units is mechanical for the numeric and
   tensor functions and has no path for the rest.
2. The workspace pins stable 1.91.0. `abi_ptx` is nightly. Device code compiles as a separate unit
   producing PTX at build time, which is what `vortex-cuda/build.rs` already does for `.cu`, so this
   is an additive build step rather than a workspace toolchain change.
3. CI already has `cuda.yaml` and `pr-bench-gpu-compress.yml`, so GPU runners exist.
4. Third-party row functions do not get GPU execution for free. They need to ship PTX. This is a
   real narrowing of the epic's goal that anyone can add their own scalar function, and it belongs
   in the epic rather than being discovered later.

The smallest useful first step is one monomorphization end to end: `NumericBinary` at `i64`, over
two canonical device arrays, with the failure word reduced on device and validity applied by the
existing lifting. That exercises the PTX build step, the launch path, the buffer handle round trip,
and the deferred-error retry, and it needs no new plan machinery.

## Open questions

- How much is the chunked access path worth? It recovers the vector transactions, but sizing the
  difference needs `ptxas` and a GPU. `nvcc` may not vectorize the equivalent C++ either, since
  `scalar_kernel.cuh` leaves that to `#pragma unroll` rather than an explicit `int4` cast.
- Does the chunk width belong on `InputElement`, or on the executor as a per-batch choice? An
  element knows its own width, but the useful chunk depends on the whole argument tuple.
- Does the chunked shape reaching `zmm` on AVX-512 make the x86 numbers better or worse on Zen 4?
  The code generation differs, the direction is unmeasured, and LLVM's `ymm` preference is often
  deliberate.
- Which is the right device null policy: always `Dense` where `DENSE_SAFE` permits it, or is there a
  survivor fraction below which stream compaction wins?
- Does the failure word reduce per block with `atomicOr`, or per warp with a ballot, and do the
  deferred-error retry semantics in `lift.rs` survive either?
- Is `visit_prepared_deferred` alone enough to be worth the work, given it excludes every sink-based
  function including the tensor outputs?
- Does the GPU scalar-function layer dispatch per `RowFn`, or does it need an expression-level entry
  point from the start to avoid a round trip per node?
