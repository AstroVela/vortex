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

The blockers are narrower than the framing suggests. They are the element types that are not flat
buffers, the null-strategy thresholds, and the toolchain.

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

- The loads are scalar `ld.global.b64`. `scalar_kernel.cuh` deliberately processes 16 bytes per
  iteration. A warp coalesces the scalar loads into the same transactions, so this costs instruction
  count and latency hiding rather than raw bandwidth, but it is a real difference and it needs
  measuring rather than assuming.
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

On that comparison the PTX above is the answer. The abstraction erases completely, the loop body is
what a hand author would write, and an elementwise scalar function is bandwidth-bound, so the ALU
work the row body performs is not the limiting factor. There is no structural reason for a gap. The
one open item is the vectorized-access difference noted above, which needs measuring.

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

- Does the vectorized 16-byte access in `scalar_kernel.cuh` beat the scalar grid-stride loop the
  Rust path emits, on a bandwidth-bound elementwise operation?
- Which is the right device null policy: always `Dense` where `DENSE_SAFE` permits it, or is there a
  survivor fraction below which stream compaction wins?
- Does the failure word reduce per block with `atomicOr`, or per warp with a ballot, and do the
  deferred-error retry semantics in `lift.rs` survive either?
- Is `visit_prepared_deferred` alone enough to be worth the work, given it excludes every sink-based
  function including the tensor outputs?
- Does the GPU scalar-function layer dispatch per `RowFn`, or does it need an expression-level entry
  point from the start to avoid a round trip per node?
