<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn GPU execution research

This record answers whether `RowFn` can drive GPU execution for scalar functions over canonical
device arrays, what a shipped version needs, and what to measure first on a GPU machine.

What the evidence supports: a replica of the `visit_prepared_deferred` shape compiles for
`nvptx64-nvidia-cuda` with the generic trait machinery fully erased, and the emitted hot loop has
the intended shape, with loop control as the only branch and the failure word accumulating in a
register. The executor around it has close in-repo precedents, including one three-input compute
kernel over canonical device arrays that already ships.

What the evidence does not support: no PTX in this record was assembled by `ptxas` or executed,
no `nvcc` output existed to compare against, and there are no timings. Parity with the
hand-written kernels is a source-level reading, not a measured result. Every performance sentence
below is either labeled arithmetic or a structural claim, and the first hour on a GPU machine can
invalidate any of them.

## Scope and limits of this record

No GPU and no CUDA toolkit were available on the research machine. `nvcc` and `nvidia-smi` are
both absent.

- The PTX in [`codegen`](codegen) is real `rustc` output for `nvptx64-nvidia-cuda`, from the
  nightly toolchain named in each file. None of it was assembled or executed.
- The two headline PTX excerpts below elide address computations and induction increments, marked
  with `...`. The appendix files hold the unedited loops.
- **This record contains no timings.** The bandwidth arithmetic names its assumed hardware inline.
- The bandwidth arithmetic uses 100M-element columns, the size in
  `vortex-cuda/benches/bench_config/mod.rs`. Production scan splits default to about 8k rows, a
  regime where launch latency competes with memory time, and no claim below is validated there.

## Verdicts in one place

**Plausible through a GPU `RowFn` executor, pending measurement.** The elementwise fixed-width
family: the numeric operators, comparison, `between`, `case_when`, `fill_null`, and `cast` over
primitives. One qualification the coverage claim previously hid: of these, only `NumericBinary`
is a `RowFn` today. The others execute through `ScalarFnVTable` directly, so each needs its CPU
`RowFn` port before any GPU route applies to it. The current `RowFn` set is `NumericBinary`, the
three tensor functions, and the three spatial predicates.

**Workable on a GPU, but not through this executor.** The columnar family (`not`, `is_null`,
`mask`, `list_length`, `byte_length`) wants plain hand-written kernels, and `literal` already has
one (`constant_numeric.cu` through `ConstantNumericExecutor`). `list_sum` is a segmented
reduction, a standard GPU primitive. `like` has shipped GPU implementations in cuDF. These are
excluded from the `RowFn` route by its one-thread-per-row shape, not from the device.

**Not workable through this route as designed.** Sink-based `visit_prepared_into` with
variable-length output, without a two-pass count-then-fill or an atomic bump allocator. Prepared
state that is not plain data, since it must cross the launch boundary as kernel parameters. The
spatial predicates, which are `RowFn`s on the CPU and whose bbox early-out and variable-length
exact predicate fit warp execution badly. Third-party `RowFn`s that do not ship device code,
since no host compilation can produce it for them.

**Unknown until measured.** Whether generated kernels match `nvcc` output on wall clock. Whether
the access-width gap exists against the real `nvcc` baseline at all. The right null strategy per
survivor fraction and validity clustering. Whether any of this survives at scan-split batch
sizes rather than 100M-row benchmark sizes.

## What to run on a GPU machine

Ordered so each step gates the next. Steps 1 through 3 fit in a day.

1. **Validate the PTX.** `ptxas -arch=sm_70 --verbose` over the modules in [`codegen`](codegen).
   Rejection or register blowup invalidates the codegen conclusions and stops the line cheaply.
2. **Launch and check correctness.** Load the replica kernels through `cudarc` the way
   `vortex-cuda` loads embedded PTX, run `numeric_add_i64` and the bool-failure variant against
   the CPU path over randomized inputs, including overflow rows. This proves the tier 2 backend
   on real hardware, which matters because miscompilation defects were being fixed as late as
   Rust 1.97.
3. **The route decider.** Benchmark the Rust-emitted FoR kernel against `for.cu`'s
   `for_in_out_i64` using the existing `for_cuda.rs` bench harness. Same shape, same data, at
   100M rows and at 8k, 64k, and 1M rows. Within noise at large sizes means direct compilation is
   viable and the remaining questions are plumbing. A stable gap means inspect SASS, and if the
   gap is access width, run step 4 before concluding.
4. **Access width.** Time the scalar grid-stride, scalar tiled, and `ld.global.v2.b64` chunked
   kernels from [`chunked-executor.md`](codegen/chunked-executor.md) against each other, and
   profile sector utilization. Also compile `scalar_kernel.cuh` with `nvcc` and read whether it
   emits vector loads at all, which decides whether the "gap" exists against the real baseline.
5. **Null strategy sweep.** Dense against filter against branch-and-skip at survivor fractions
   from 10% to 99%, crossed with random and run-clustered validity. Clustered validity can make
   whole warps skip coherently, so branch-and-skip stays in the matrix despite the divergence
   argument.
6. **Failure word cost.** The `or.pred` accumulation plus one `atomicOr` per block against the
   unchecked kernel, to price checked semantics.

Each step maps to a decision: 1 and 2 gate everything, 3 picks the build route, 4 decides whether
the chunk accessor is worth an API change, 5 replaces the x86 thresholds, 6 prices `FALLIBLE`.

## The layering this question sits in

Two concerns meet on the GPU. Decoding an encoded array belongs to the encoding implementor:
`hybrid_dispatch` collapses an encoding tree into one launch and its output is canonical. A
scalar function over canonical inputs is a separate layer, and it is the layer in question.

The previous revision of this record claimed that layer does not exist in `vortex-cuda`. That
was wrong, and the correction strengthens the case:

- `DateTimePartsExecutor` executes all three children to canonical device arrays, computes a
  divisor from the dtype options on the host, and launches one kernel over three device inputs
  plus that constant as a kernel parameter, producing one output. That is prepare-state, N-ary
  input, and elementwise math, with `match_each_signed_integer_ptype!` monomorphization on the
  host mirrored by C macros on the device. It is the closest template for a `RowFn` executor.
- The Dict executor launches `dict_<V>_<I>` over two canonical device inputs.
- `Cast` executes on the GPU today inside dispatch plans, special-cased by the plan builder.
  Whether it later migrates into a general scalar-function layer or stays a plan special case is
  an open design point.

What does not exist is only the general entry point: nothing routes an arbitrary `ScalarFnId`
over canonical device inputs. Since scalar functions appear in trees as the lazy `ScalarFn`
encoding and `vortex-cuda` executes per-encoding through `CudaExecute`, the entry point is a
`CudaExecute` impl for the `ScalarFn` encoding.

The dispatch plan is not that entry point. Plans do carry multiple inputs across stages, and the
`DICT` op reads a second data source, but every non-output stage must decode wholly into a
shared-memory tile. A full-length row-aligned second operand has no representation, so a binary
function such as `hypot(x, y)` cannot be a plan stage. The layering is decode, then compute, the
same as on the CPU.

## The row body compiles to PTX with the intended shape

[`row-body-replica-rs.md`](codegen/row-body-replica-rs.md) reproduces the
`vortex-array/src/scalar_fn/fns/binary/numeric/row.rs` structure: a `CheckedPrimitiveOp` trait
with an associated `Failure` type, generic monomorphization over the width, a `prepare` step, and
an `apply` closure passed as `impl Fn` to a generic row loop.

The toolchain reality: the emitted PTX used nightly, and a launchable kernel cannot compile on
stable, since both `extern "ptx-kernel"` (`feature(abi_ptx)`) and the special-register access
(`feature(asm_experimental_arch)`, or the unstable `core::arch::nvptx` intrinsics) are gated. The
cost is a nightly toolchain for the device crate alone, with the host workspace unchanged.

The main loop for `numeric_add_i64`, elided to the operative instructions:

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
	// ... address and induction increments elided
	setp.lt.u64 	%p2, %rd23, %rd7;      // loop control is the only branch
	@%p2 bra 	$L__BB1_2;
```

The full loop is in [`row-body-replica-ptx.md`](codegen/row-body-replica-ptx.md). The properties
that matter:

1. The trait, the associated type, the generic loop, and the `impl Fn` closure all monomorphize
   away. This is the x86 audit's result again, from the same cause.
2. The only branch is loop control, and the failure word reduces once after the loop.
3. `prepare` constant-folds: in `scale_f32_by_constant` the prepared `2.0 * PI` becomes the
   immediate `0f40C90FDB`.
4. The replica used a `u64` failure word, but production `CheckedAdd` declares `bool`.
   [`bool-failure.md`](codegen/bool-failure.md) compiles the production shape: the evidence
   lowers to `setp.lt.s64` with `or.pred` accumulation, still branchless. The clean-loop result
   holds for the failure type the real function uses.

Two caveats on the replica itself. The failing-threads-store-`1` finish is a formal data race
even though every write is identical, so a real kernel uses `atomicOr`, for which the repository
already has the pattern: `arrow_offsets.cu` and `list_view.cu` accumulate `atomicMax` into a
device status word that `canonical.rs` copies back and maps to a `VortexResult`. And special
registers read through plain `asm!` stay inside the loop unless marked
`options(pure, nomem, nostack)`.

## FoR decode, the nearest hand-written comparison

One scalar operation already runs on the GPU under `RowFn`'s exact split, written in C++.
`for.cu` is:

```cpp
template <typename T> struct ForOp {
    T reference;
    __device__ inline T operator()(T value) const { return value + reference; }
};
scalar_kernel(input, output, array_len, ForOp<Type>{reference});
```

`ForOp<T>{reference}` is `prepare` hoisting the constant operand, `operator()` is `apply`, and
`scalar_kernel<InputT, OutputT, Op>` is the executor, generic over a functor exactly as the Rust
executor is generic over a closure. [`for-decode-rs.md`](codegen/for-decode-rs.md) writes that
row body as Rust generics, and the emitted grid-stride loop
([`for-decode-ptx.md`](codegen/for-decode-ptx.md)) is:

```ptx
$L__BB1_2:
	ld.global.b64 	%rd12, [%rd11];
	add.s64 	%rd13, %rd12, %rd5;    // %rd5 is the reference, hoisted before the loop
	st.global.b64 	[%rd14], %rd13;
	// ... address and induction increments elided
	setp.lt.u64 	%p2, %rd16, %rd6;
	@%p2 bra 	$L__BB1_2;
```

This matches a source-level reading of what `scalar_kernel` computes per element. Whether it
matches what `nvcc` emits is unknown, because no `nvcc` was available, and that comparison is
step 3 of the GPU plan. The `for_cuda.rs` bench and the `cuda.yaml` GPU CI job already exist as
the vehicles for it.

## Vector memory access: two conditions, two owners

The Rust-emitted loops above issue one 8-byte load per element. An ideal kernel for this shape
issues 16-byte vector loads. Two things are true about that gap and neither is what the first
revision of this record claimed:

- `scalar_kernel.cuh` contains no vector access either. It processes 16 bytes per iteration at
  the source level through `#pragma unroll` over plain element pointers, so `nvcc` faces the same
  alignment-provability constraint. Whether the shipped CUDA kernels issue vector loads is
  unverified, so the deficit is against an ideal kernel, not an established baseline.
- For the one-element-per-thread grid-stride shape, a warp's 32 scalar loads coalesce into the
  same transactions the vector form uses, so the cost is instruction count. For the tiled
  multiple-elements-per-thread shape, consecutive threads sit 16 bytes apart, each scalar load
  achieves about half sector utilization, and the scalar form issues roughly twice the
  transactions while leaning on cache hits. The two shapes must be profiled separately.

What the probes established about getting the vector form
([`vectorization-probes.md`](codegen/vectorization-probes.md),
[`alignment-expression.md`](codegen/alignment-expression.md),
[`element-carried-alignment.md`](codegen/element-carried-alignment.md),
[`chunked-executor.md`](codegen/chunked-executor.md)):

1. **One iteration must touch adjacent elements.** The LoadStoreVectorizer merges accesses within
   an iteration and does not restructure loops. LLVM's loop vectorizer is present in the nvptx64
   pipeline but declines through the target cost model, and rightly so, since the warp already
   supplies the lane parallelism it would create. A one-row-per-iteration loop presents nothing
   to merge. This half is owned by the executor: visiting two rows per iteration is a local loop
   change with no trait involvement.
2. **The access must carry provable alignment in its type.** In these probes, neither
   `core::hint::assert_unchecked` nor a runtime `align_offset` check recovered the vector form,
   and carrying alignment in the element's `Varying` type behind today's row-indexed accessor
   also failed, under both a `tid() * 2` and a provably even `c * 2` index, because strength
   reduction lowers the loop to a byte offset advanced by an opaque stride. The only shape that
   merged indexes a pointer whose pointee is the 16-byte chunk, so the alignment holds for any
   index. The probes cannot rule out that some `llvm.assume` alignment pattern reaches the
   analysis, but every executor-shaped attempt failed and the type-carried shape worked on the
   first try. This half touches the element API: a chunk-granular accessor, whose exact home
   (`InputElement` or the executor, per argument tuple) is an open design point.

Two constraints the fix must carry, found by review rather than by the probes. Vortex buffers are
256-byte aligned at allocation, but slicing is `range.start * byte_width`, so an `i64` array
sliced at an odd offset is 8-byte aligned and a `#[repr(align(16))]` cast over it is undefined
behavior. The chunked path therefore needs a runtime alignment check with a scalar fallback, plus
a remainder loop for lengths that are not a chunk multiple. And on the CPU side the executor
already runs on `lane_kernels` machinery with `CHUNK_LEN = 64` tiling, so a chunk accessor nests
inside existing tiles rather than introducing chunking as a new concept.

None of this is scheduled work. The wider access buys instruction count and sector utilization
whose value is unmeasured, which is why it is step 4 of the GPU plan and not part of the
milestone.

[`x86-chunked.md`](codegen/x86-chunked.md) compiles chunked and scalar shapes for x86: under
AVX-512 the chunked shape reaches `zmm` where the scalar loop stays on `ymm`. That is a codegen
observation on a different chunk width than the GPU probes, LLVM's 256-bit preference is often
deliberate, and the checked `i64` multiply vectorizes under neither shape, so chunking is not a
fix for the `mul_i64` regressions in the x86 record.

## The null-strategy thresholds do not transfer

`lift.rs` picks between `Dense`, `DenseWithRetry`, `Filter`, and `BranchAndSkip` with thresholds
measured on x86. On a GPU the costs move: branch-and-skip loses its advantage when lanes within a
warp diverge, filtering pays a real compaction pass, and dense execution wastes compute that a
bandwidth-bound kernel may not notice.

The working hypothesis for a device policy is dense-by-default where `DENSE_SAFE` permits. It is
a hypothesis, not a conclusion. Run-clustered validity lets whole warps skip coherently, which
revives branch-and-skip, and an expensive row body changes the arithmetic for filtering. All
three strategies stay in the step 5 sweep. The structure transfers as-is: the strategy set is
shared, the policy is device-specific, and `lift.rs` already separates the two.

## Performance, with the arithmetic labeled

Assumed hardware for the arithmetic: PCIe 4.0 x16 at roughly 25 GB/s effective, and an A100-class
device at roughly 2 TB/s. On an L4 at roughly 300 GB/s the device-side numbers are about 7x
larger and the conclusions compress accordingly.

At 100M `i64` rows a column is 800 MB. A binary operation reads two columns and writes one, about
2.4 GB of device traffic, roughly 1.2 ms at 2 TB/s, against launch overhead in the microseconds.
Moving that operation's data over PCIe costs about 32 ms per column each way, so about 96 ms for
two inputs up and one output back. The ratio survives any current hardware choice: a GPU scalar
function pays only inside an already device-resident pipeline, which `device_read_at.rs` and
`pooled_read_at` provide.

The batch-size caveat applies to all of it. At the default scan split of about 8k rows, a column
is 64 KB, kernel time is microseconds, and launch latency is a first-order cost rather than
noise. The 100M-row arithmetic says nothing about that regime, and step 3 measures both.

Per-node execution is also not symmetric with the CPU, as this record previously claimed. A tree
of `RowFn` nodes costs one launch and one global-memory round trip per node on the GPU. On the
CPU at scan batch sizes the intermediate stays cache-resident, so the per-node penalty is smaller
there. Expression-level fusion is consequently more valuable on the GPU, and per-node execution
is the right first shape but not a long-term equivalence.

## What remains between the PTX and production

**The entry point.** A `CudaExecute` impl for the `ScalarFn` encoding, modeled on
`DateTimePartsExecutor`: execute children to canonical, host-side prepare to kernel parameters,
launch, rebuild through `from_buffer_handle`. Validity conjoins on the host as `lift.rs` does.
`filter_primitive` shows the values-and-validity split for a mask-shaped operation, and
`execute_validity_cuda` exists for array-backed validity, used today by the FSST and Arrow export
paths.

**Keeping host and device monomorphizations in sync.** The enumeration itself already exists in
`dispatch`, and the repository already solves the sync problem twice: the device side stamps
entries per width with the `FOR_EACH_*` X-macros in `types.cuh`, deliberately mirroring the host
match macros, and the host derives kernel names by convention through
`ctx.load_function("for", &[P::PTYPE])`, so a missing entry is a loud load error rather than a
silent fallback. A Rust device crate reuses the same design with the shared width list in a
`no_std` crate both compilations depend on, or with build-time generation as `bit_unpack_gen.rs`
already does. A probe-visitor conformance test can additionally assert every dispatch-selectable
signature has an entry or a recorded exclusion.

**Prepared state as kernel parameters.** `prepare` runs on the host once per batch, and its
output crosses the launch boundary, so it must be plain data. The `DateTimeParts` divisor is the
in-repo precedent. A prepared state holding references stays on the CPU.

**Failure words.** Reuse the existing device status-word machinery from the Arrow kernels, with
`atomicOr` semantics and the `lift.rs` deferred-error mapping including the valid-only retry.

**Fallback policy.** The current residency gate in `executor.rs` refuses CPU fallback when
buffers are device-resident, erroring instead of paying the copy. A scalar function with no GPU
entry needs an actual decision, made at plan level: keep the subtree on the CPU or accept the
transfer. This is genuinely new policy, not a reuse of an existing decision.

**Build degradation.** `build.rs` already embeds an empty PTX table when `nvcc` is absent and the
session falls back to CPU. The rustc route needs identical behavior when the nightly toolchain or
`llvm-bitcode-linker` is missing. The `cuda.yaml` workflow already has both a no-GPU compile job
and a GPU-runner test job to extend.

**The milestone**, defined once: checked `NumericBinary` add at `i64`, column times constant, as
a `CudaExecute`-backed execution over canonical device inputs, benchmarked with the existing
`for_cuda.rs` harness against `for.cu`'s `for_in_out_i64`. The baselines differ by the checked
semantics, so the delta prices the failure word, and the bbox for success is parity within the
failure-word cost. The column-times-column shape runs second, with no direct CUDA baseline, per
the repository's rule that operand shapes are benchmarked separately.

## What else this knowledge is good for

**Encodings are the larger prize.** `bit_unpack_gen.rs` is Rust consuming the `fastlanes` crate
to write CUDA C source strings for `nvcc`. The FastLanes logic is already Rust, transliterated to
reach the GPU. Direct compilation removes the transliteration, and encodings are where
`vortex-cuda`'s throughput work is.

**Agreement between CPU and GPU becomes structural.** A kernel that exists once in Rust and once
in CUDA C++ is kept honest only by tests. One source compiled for two targets removes the
divergence class, which matters most for bit-exact decode.

**A second execution target audits the API.** The access-width question was invisible while x86
was the only target because the loop vectorizer restructured the loop for free. The buffer
alignment lesson is the same shape: Vortex allocates at 256 bytes and hands out `&[T]`, which
erases the guarantee to 8, and in these probes only a type at the load site carried it back.

## Two routes to a shipped kernel

Neither route asks the compiler to decide where code runs. Vortex owns dispatch, and the compiler
produces the kernel body.

**Direct compilation.** `rustc` emits PTX and the build embeds it exactly as `build.rs` embeds
`nvcc` output. The codegen questions this record could answer without hardware are answered:
erasure, constant folding, register-resident failure words, and both failure-word types. The
open ones need hardware: `ptxas` acceptance, execution correctness, and wall clock. Costs are the
nightly device crate, the `llvm-bitcode-linker` component, and tier 2 risk. This is the only
route with one source for both targets, which is the property that makes the exercise worth
anything.

**Codegen.** A Rust generator emits CUDA C functors, the `bit_unpack_gen.rs` pattern. No new
toolchain, `nvcc`'s optimizer, and the existing X-macro sync design. The limit is the source of
truth: arbitrary Rust bodies cannot be transpiled, so bodies live in a generator-printable form
or get written twice with differential tests, with `RowFn` as the contract rather than the
implementation.

Both routes converge on the same milestone, so they can be raced on one machine and the losing
route still validates the winner's output.

## Open questions

- Does `ptxas` accept the recorded modules, and does the generated FoR kernel match `for.cu` on
  wall clock at both benchmark and scan-split sizes? (Steps 1 through 3.)
- Does `nvcc` emit vector loads for `scalar_kernel.cuh` at all, and how much does the vector form
  buy in each loop shape? (Step 4.)
- Where does the chunk-granular accessor live, on `InputElement` or on the executor, given the
  useful width depends on the whole argument tuple, and what does its slice-alignment fallback
  cost?
- What is the device null policy as a function of survivor fraction and validity clustering?
  (Step 5.)
- Does the deferred-error retry survive the device failure word intact? (Step 6.)
- Does `Cast` migrate into the scalar-function layer or remain a plan special case?
- What does the GPU story look like at 8k-row scan splits, where launch latency is first-order?
