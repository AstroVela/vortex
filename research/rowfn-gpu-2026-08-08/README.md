<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn GPU execution research

`RowFn` can drive GPU execution. A replica of the `visit_prepared_deferred` shape compiles to
correct PTX on the `nvptx64` target with the generic trait machinery fully erased, and the loop body
it produces matches what a hand-written CUDA kernel emits. The language and codegen side of the
question is settled.

The obstacle is elsewhere. `RowFn` is row-at-a-time over a *decoded* column, and `vortex-cuda` earns
its throughput by never decoding to a canonical column at all. A `RowFn` executed as its own kernel
launch over canonical inputs is slower than the current GPU path on encoded data, not because the
row body is slow, but because it forces a materialization and a global memory round trip that
`dynamic_dispatch.cu` exists to avoid.

The useful target is therefore not a GPU `RowFn` executor. It is a `RowFn` row body admitted as a
new `ScalarOp` inside the existing dispatch plan.

## Scope and limits of this record

No GPU and no CUDA toolkit were available on the research machine. `nvcc` and `nvidia-smi` are both
absent.

- The PTX in [`codegen`](codegen) is real `rustc` output, emitted for `nvptx64-nvidia-cuda`. It was
  not assembled by `ptxas` and it was not executed.
- **This record contains no timings.** Every performance statement below is either arithmetic from
  published bandwidth figures, labeled as such, or a structural claim about memory traffic. None of
  it is a measurement, and none of it substitutes for one.
- The bandwidth arithmetic uses 100M elements, the size in
  `vortex-cuda/benches/bench_config/mod.rs`.

## The C++ kernels already have RowFn's shape

`vortex-cuda/kernels/src/for.cu` and `scalar_kernel.cuh` split the work exactly the way `RowFn`
does. The correspondence is one-to-one:

| `RowFn` | `vortex-cuda` |
| --- | --- |
| `prepare(A::ConstElems) -> P` | `ForOp<T> { reference }`, the functor's captured state |
| `apply(&P, elems) -> O` | `T operator()(T value) const` |
| `RowFn::dispatch` with `match_each_native_ptype!` | `FOR_EACH_INTEGER(GENERATE_FOR_KERNEL)` |
| `execute_row_output_prepared` | `scalar_kernel<InputT, OutputT, Op>` |
| `OutputElement` with no drop glue | the POD output buffer |

The two arrived at the same factoring independently. `RowFn` is the abstraction the CUDA code
already writes by hand in C++ templates. The question is only whether one source can serve both.

## The row body compiles to PTX today

[`row-body-replica-rs.md`](codegen/row-body-replica-rs.md) reproduces the
`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs` structure: a `CheckedPrimitiveOp` trait with
an associated `Failure` type, generic monomorphization over the primitive width, a `prepare` step
producing batch state, and an `apply` closure passed as `impl Fn` to a generic row loop.

`rustup target add nvptx64-nvidia-cuda` succeeds on the repository's pinned 1.91.0 stable toolchain
and ships `core`, `alloc`, and `compiler_builtins`. Codegen itself succeeds on stable. Declaring
the kernel entry point needs `feature(abi_ptx)`, so the emitted PTX below used nightly.

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
  iteration. A warp coalesces the scalar loads into the same transactions, so this costs
  instruction count and latency hiding rather than raw bandwidth, but it is a real difference and
  it needs measuring rather than assuming.
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
kernels as `.visible .entry`. Building through `cargo` instead reaches codegen and then fails at
the `llvm-bitcode-linker` component, which was not installed. The object file for the target is
produced regardless, so the failure is in linking and not in compilation.

</details>

## What does not port

The row body is the portable part. The machinery around it is host-shaped throughout.

- `InputElement::decode(ArrayRef, &mut ExecutionCtx) -> Column` runs on the host and produces host
  buffers. The GPU equivalent already exists and is not this function. It is the source and scalar
  op chain in `dynamic_dispatch.cu`, which decodes `BitPacked`, `ALP`, `FoR`, `Dict`, and `RunEnd`
  on device.
- `OutputElement::build(values: Vec<Self>) -> ArrayRef` allocates a host `Vec`.
- `InputElement::get` returns a GAT borrow. For primitives that lowers to a load. For an element
  that follows a pointer or parses bytes it becomes data-dependent addressing, which is warp
  divergence.
- `OutputSink` with `Row<'_>`, shared builders, and `SinkResult::accumulate` returning
  `VortexResult` is the least portable piece. A `VarBinView` sink needs a two-pass count-then-fill
  or an atomic bump allocator on device, and `accumulate` reintroduces a per-row error path.

This splits the trait cleanly. `visit_prepared_deferred` is the GPU-portable visit: indexed tuples,
POD output, OR-reducible failure, no per-row control flow. `visit_prepared_into` is not, except for
fixed-width sinks such as the tensor outputs.

The geo predicates are the worst case and are worth naming explicitly. `contains` and `intersects`
do data-dependent work per row with a bounding-box early-out. On a GPU the early-out saves nothing,
because a warp executes both sides of the divergence, and the exact predicate walks variable-length
geometry. These stay on CPU.

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
meaning and becomes considerably more valuable, since it is what admits the strategy a GPU wants by
default.

## Where the performance actually goes

At 100M `i64` rows a column is 800 MB. A binary operation reads two and writes one, so 2.4 GB of
traffic. At roughly 2 TB/s that is about 1.2 ms of unavoidable memory time, against a kernel launch
of a few microseconds. Launch overhead is under 1% and is not the thing to optimize.

Two costs dominate instead, and both are framework-level rather than row-level.

**Fusion.** Every unfused stage pays another read and write of the full column. `hybrid_dispatch`
exists to collapse an entire encoding tree into one launch, and its documented fallback order
prefers a standalone kernel, then full fusion, then partial fusion. A `RowFn` compiled to its own
kernel over canonical inputs opts out of all of it: the encoded input must first be materialized to
a canonical device buffer by a separate launch, and only then read back. That is one extra full
round trip plus the materialization, against a fused plan that reads the compressed bytes once.

**Residency.** PCIe 4.0 x16 moves roughly 25 GB/s in practice, so a host round trip for that same
800 MB column costs about 32 ms, which is more than twenty times the kernel. `RowFn` on GPU only
makes sense inside an already device-resident pipeline. `device_read_at.rs` and `pooled_read_at`
mean that pipeline exists, and it is the only context in which any of this pays.

So the answer to whether a generated kernel matches a hand-written one depends on what
"hand-written" means. Against a standalone elementwise CUDA kernel, the PTX above says yes, and
there is no reason to expect otherwise on a bandwidth-bound operation. Against what `vortex-cuda`
actually does, which is a fused decode-and-compute plan, a per-`RowFn` kernel loses on memory
traffic no matter how good its body is.

## Three designs

**A. Shared Rust source compiled to PTX.** Row bodies move into `no_std` POD-only modules compiled
twice, once for the host and once for `nvptx64`, with a build step emitting PTX the way `build.rs`
already emits it from `.cu`. `RowFn::dispatch` is already a pure function of options and dtypes, so
the same visitor selects the monomorphization at runtime and looks up the matching PTX entry point.
This delivers what the epic actually wants, which is one definition and no duplicated semantics.

**B. Device functor as a new `ScalarOp`.** The row body is written once more in CUDA C++ as a
functor, registered in the `ScalarOp` enum that already carries `FOR`, `ZIGZAG`, `ALP`, `DICT`, and
`CAST`, and executed by the existing `scalar_op<T, N>` over N values in registers. `RowFn` is the
specification and differential tests enforce that the two agree. This fuses into the dispatch plan
for free and needs no new toolchain.

**C. An expression IR or DSL.** Row bodies are rewritten in a restricted language that emits both
CPU and GPU code, in the shape of CubeCL. This gives the best fusion and cross-vendor reach, and it
gives up writing row bodies in ordinary Rust, which is most of why `RowFn` is pleasant to
implement against.

A and B are less different than they look. Both need a device-callable function of identical shape
admitted as a stage in the dispatch plan, and that plan integration is the larger and riskier half
of the work in both cases. They differ only in who writes the functor.

That suggests the sequencing. Do the plan integration first, against one or two hand-written
functors, because it is required either way and it is where the throughput actually lives. Treat
the source-sharing question as a later swap of how the functor is produced, once the row bodies are
split out of the host-typed crates. Do not start with the toolchain work, which is the part with
the least uncertainty and the least payoff.

## Toolchain and practicality

Rust-to-PTX is no longer the risky part, but it is not free either.

- `nvptx64-nvidia-cuda` is a tier 2 rustc target. Rust 1.97 raised the baseline PTX ISA and GPU
  architecture, which fixed defects where valid Rust triggered compiler crashes or
  miscompilations.
- NVIDIA released [cuda-oxide](https://github.com/NVlabs/cuda-oxide) in May 2026, a rustc codegen
  backend compiling `#[kernel]` Rust to PTX through MIR, Pliron IR, then LLVM. It supports generic
  kernels with monomorphization and closures with captures, and its own `map<T: Copy, F>` example
  is close to the `RowFn` shape. It is explicitly alpha.
- [Rust-CUDA](https://github.com/Rust-GPU/rust-cuda) with `rustc_codegen_nvvm` is the older route
  through NVIDIA's NVVM.
- [CubeCL](https://github.com/tracel-ai/cubecl) is the design-C option and reaches CUDA, ROCm,
  Vulkan, and Metal from one kernel, at the cost of a restricted DSL rather than arbitrary Rust.

Four practical costs, in rough order of how much work they are:

1. Row bodies currently sit beside host code that uses `ArrayRef`, `ExecutionCtx`, and
   `vortex_error`. Splitting them into `no_std` POD-only units is mechanical for the numeric and
   tensor functions and has no path for the rest.
2. The workspace pins stable 1.91.0. `abi_ptx` is nightly. Device code compiles as a separate unit
   producing PTX at build time, which is what `vortex-cuda/build.rs` already does for `.cu`, so
   this is an additive build step rather than a workspace toolchain change.
3. CI already has `cuda.yaml` and `pr-bench-gpu-compress.yml`, so GPU runners exist.
4. Third-party row functions do not get GPU execution for free under any of the three designs. They
   need to ship PTX or a functor. This is a real narrowing of the epic's goal that anyone can add
   their own scalar function, and it belongs in the epic rather than being discovered later.

## Open questions

- Does the vectorized 16-byte access in `scalar_kernel.cuh` beat the scalar grid-stride loop the
  Rust path emits, on a bandwidth-bound elementwise operation? This needs a measurement before
  either design is chosen.
- Which is the right device null policy: always `Dense` where `DENSE_SAFE` permits it, or is there
  a survivor fraction below which stream compaction wins?
- Can a `RowFn` stage participate in the fused plan when its inputs have different encodings, or
  does fusion require all operands to reach the same stage boundary?
- Is `visit_prepared_deferred` alone enough to be worth the work, given it excludes every sink-based
  function including the tensor outputs?
- Does the failure word reduce per block with `atomicOr`, or per warp with a ballot, and do the
  deferred-error retry semantics in `lift.rs` survive either?
