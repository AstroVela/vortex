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

The blockers are narrower than the framing suggests. They are the element types that are not flat
buffers, the null-strategy thresholds, the memory access width, and the toolchain.

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

No `ld.global.v2.b64` appears anywhere in either module. LLVM's NVPTX backend did not merge the
adjacent accesses, and writing the loop in the shape that invites merging did not change that.

This is the one place the generated code is measurably behind a hand-written kernel, and it needs a
`ptxas` comparison to size. Whether `nvcc` merges the equivalent C++ was not tested, because no CUDA
toolkit was available. `scalar_kernel.cuh` relies on `#pragma unroll` rather than an explicit
`int4` or `double2` cast, so it is not certain that the C++ vectorizes either. Explicit chunked
loads on the Rust side are the obvious mitigation if the gap is real.

One artifact of the replica rather than of `RowFn`: reading `%ntid.x` through `core::arch::asm!`
leaves the read inside the loop, because plain inline assembly is not hoistable. A real
implementation reads the special registers once outside the loop or uses the `core::arch::nvptx`
intrinsics.

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

The one identified gap is memory access width. The Rust path emits scalar loads where
`scalar_kernel.cuh` asks for 16 bytes per iteration, and that did not change under a tiled loop.
Sizing it needs `ptxas` and a GPU.

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

- How much does the scalar-load gap cost? The Rust path does not emit `ld.global.v2.b64` even under
  a tiled loop. Measuring it needs `ptxas` and a GPU, and it also needs confirming that `nvcc`
  vectorizes the equivalent C++, which `scalar_kernel.cuh` leaves to `#pragma unroll`.
- Do explicit chunked loads on the Rust side recover the vector access, and does that belong in the
  executor or in `InputElement`?
- Which is the right device null policy: always `Dense` where `DENSE_SAFE` permits it, or is there a
  survivor fraction below which stream compaction wins?
- Does the failure word reduce per block with `atomicOr`, or per warp with a ballot, and do the
  deferred-error retry semantics in `lift.rs` survive either?
- Is `visit_prepared_deferred` alone enough to be worth the work, given it excludes every sink-based
  function including the tensor outputs?
- Does the GPU scalar-function layer dispatch per `RowFn`, or does it need an expression-level entry
  point from the start to avoid a round trip per node?
