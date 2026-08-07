<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn owned-output and x86 numeric research

This is the durable record for the investigation that produced the owned-output RowFn path. The
result is not that RowFn is inherently difficult to optimize. The declaration must distinguish an
independent returned value from a stateful output sink, and dense primitive inputs must cross a
validated indexed-source boundary that shared execution can lower directly.

The selected implementation restores `i64` and `u64` varying multiplication to within about 1% of
the actual merge-base throughput. It does so without a numeric array downcast, `reduce_encoded`
override, numeric-owned allocation, or numeric-specific null and constant policy.

## Revisions and environment

- Merge-base baseline: `19f771f2a426103aa7d1bf7153a258bb1bab1e19`.
- Untouched sink-only RowFn: `35098c72118f1b555a24bd2f9b58b0400fa46dc5`.
- Selected implementation: `1a0a055c752b54448c8e1d54af032fe43acf8517`.
- Selected diff fingerprint:
  `928e7a0baa2895609d102c98d110c21fb7a12e079b04195b85903277c71537a2`.

The research branch has older tensor and spatial RowFn users. The result was ported rather than
rebased so that history remains intact. Its port also backports `map_checked_into`, which already
exists at the mergeable branch's base.

```text
AMD Ryzen 9 7950X
1 socket, 16 physical cores, 32 threads
benchmark logical CPU: 8; SMT sibling: 24
Linux CTCachyDesktop 7.1.6-1-cachyos, x86_64
rustc 1.91.0, LLVM 21.1.2
cargo 1.91.0
```

The CPU reports AVX2 and AVX-512F/DQ/BW/VL. Builds used the default repository target and bench
profile without LTO, `target-cpu=native`, profile changes, or forced inlining. The scaling governor
was `performance`. Timed executions were pinned to CPU 8 and never overlapped compilation.

```bash
taskset -c 8 "$BENCH" --bench --sample-count 100 --max-time 0.5 --color never \
  mul_i8_nonnull mul_u8_nonnull mul_i16_nonnull mul_u16_nonnull \
  mul_i32_nonnull mul_u32_nonnull mul_i64_nonnull mul_u64_nonnull \
  add_i64_nonnull add_i64_constant sub_i64_constant \
  mul_i32_constant mul_i32_nullable div_i64_nonnull
```

Every file in [`benchmarks`](benchmarks) is unedited Divan output wrapped in Markdown. It includes
fastest, slowest, median, mean, samples, and iterations rather than only the selected medians.

## Stage 0: reproduction

Order: baseline, candidate, baseline, candidate. Values are median microseconds.

| Benchmark | Baseline 1 / 2 | Candidate 1 / 2 | Candidate/baseline |
| --- | ---: | ---: | ---: |
| `add_i64_constant` | 8.449 / 8.399 | 9.269 / 9.290 | 1.097 / 1.106 |
| `add_i64_nonnull` | 9.205 / 9.149 | 9.455 / 9.490 | 1.027 / 1.037 |
| `div_i64_nonnull` | 44.850 / 44.800 | 45.090 / 45.160 | 1.005 / 1.008 |
| `mul_i8_nonnull` | 6.184 / 6.199 | 4.694 / 4.699 | 0.759 / 0.758 |
| `mul_i16_nonnull` | 4.099 / 4.119 | 4.269 / 4.269 | 1.041 / 1.036 |
| `mul_i32_constant` | 26.420 / 26.430 | 18.880 / 18.870 | 0.715 / 0.714 |
| `mul_i32_nonnull` | 26.410 / 26.420 | 28.390 / 28.350 | 1.075 / 1.073 |
| `mul_i32_nullable` | 27.350 / 27.400 | 29.180 / 29.150 | 1.067 / 1.064 |
| `mul_i64_nonnull` | 23.220 / 23.200 | 30.020 / 30.080 | **1.293 / 1.297** |
| `mul_u8_nonnull` | 3.319 / 3.329 | 3.539 / 3.529 | 1.066 / 1.060 |
| `mul_u16_nonnull` | 2.599 / 2.599 | 2.429 / 2.429 | 0.935 / 0.935 |
| `mul_u32_nonnull` | 6.939 / 6.949 | 7.069 / 7.059 | 1.019 / 1.016 |
| `mul_u64_nonnull` | 19.210 / 19.190 | 30.430 / 30.490 | **1.584 / 1.589** |
| `sub_i64_constant` | 8.255 / 8.239 | 9.099 / 9.099 | 1.102 / 1.104 |

The x86 regression reproduced. Raw runs are the four `stage0-*` files.

## Stage 1: owned output without indexed input

The closure returned `(output, failure)`, shared execution owned the store, and failure remained a
loop-local OR. This removed the numeric checked sink and materially improved 64-bit cases, but did
not solve the general problem.

| Benchmark | Baseline 1 / 2 | Owned 1 / 2 | Owned/baseline |
| --- | ---: | ---: | ---: |
| `mul_i64_nonnull` | 23.20 / 23.26 | 25.65 / 25.59 | 1.106 / 1.100 |
| `mul_u64_nonnull` | 19.18 / 19.21 | 19.41 / 19.41 | 1.012 / 1.010 |
| `mul_i32_constant` | 26.43 / 26.44 | 32.36 / 32.38 | 1.224 / 1.225 |
| `mul_i32_nonnull` | 26.42 / 26.41 | 31.24 / 31.23 | 1.182 / 1.183 |
| `mul_i32_nullable` | 27.36 / 27.35 | 32.04 / 32.04 | 1.171 / 1.171 |

The six `stage1-*` files contain the full matrix. This falsifies output ownership as a complete
explanation: it matters, but does not give LLVM the specialized kernel's input representation.

## Stage 2: indexed dense input

`IndexedElementTuple` lets a primitive pair expose `LaneZip<&[Left], &[Right]>` after shared
execution validates both varying lengths once. The generic owned executor calls
`map_checked_into`; numeric code still declares only row types, operation, failure, and error.

| Benchmark | Baseline 1 / 2 | Indexed 1 / 2 | Candidate 1 / 2 |
| --- | ---: | ---: | ---: |
| `mul_i32_nonnull` | 26.39 / 26.41 | 26.58 / 26.60 | 28.34 / 28.36 |
| `mul_i32_nullable` | 27.37 / 27.38 | 27.41 / 27.43 | 29.20 / 29.17 |
| `mul_i64_nonnull` | 23.22 / 23.24 | 23.43 / 23.44 | 30.02 / 30.10 |
| `mul_u64_nonnull` | 19.22 / 19.21 | 19.41 / 19.42 | 30.41 / 30.43 |
| `div_i64_nonnull` | 44.84 / 44.87 | 45.07 / 45.03 | 45.07 / 45.12 |
| `mul_i32_constant` | 26.42 / 26.43 | 32.38 / 32.39 | 18.88 / 18.88 |

The indexed source closed the varying and nullable gap. It did not affect mixed constants, which
exposed the next compiler-sensitive detail.

## Compiler ablations: `Copy`, source order, and whole-function sensitivity

The completed ablation matrix isolates the public `Output: Copy` bound as a reliable trigger, while
falsifying the simpler explanations considered during the initial investigation:

| Variant | `mul_i32_constant` run 1 / 2 |
| --- | ---: |
| No `Copy` bound | 18.77 / 18.72 us |
| Inert private marker bound | 18.77 / 18.72 us |
| `Output: Copy` | 29.94 / 29.93 us |
| `Output: Copy`, `codegen-units=1` | 29.87 / 29.89 us |

The `i64` and `u64` controls did not move. The inert private marker is important: an arbitrary
where-clause or source perturbation is insufficient to trigger the loss. The result is specific to
the optimizer-visible `Copy` constraint, though the mechanism is not yet known.

The default-CGU DWARF ranges show large whole-function differences for the exact `i32 CheckedMul`
monomorph. The `Copy` function spans `0xe58c90..0xe59adc` (`0xe4c` bytes); no-Copy spans
`0xe7a1c0..0xe7b6d0` (`0x1510` bytes). The `Copy` hot loop at `0xe58f90` is only 16-byte aligned and
computes the low multiply before the widened chain. The no-Copy loop at `0xe7b260` is 32-byte
aligned, computes the widened chain first, and delays the low multiply. LLVM-MCA nevertheless
predicts the smaller `Copy` loop slightly better, 2.5 versus 2.7 cycles. Alignment and final loop
scheduling therefore do not explain the measured direction.

A fresh `Copy` plus `codegen-units=1` build makes this conclusion stronger: its optimized IR already
has store-before-OR, yet the linked benchmark remains at about 29.9 microseconds. Store-before-OR is
neither sufficient nor established as causal. The earlier source-order edit changed production
performance, but it must be described only as another trigger for a whole-function compiler
interaction. An isolated exact-loop hardware ablation also found OR-before-store slightly faster
(0.75-0.77 ns/row) than store-before-OR (0.823-0.825 ns/row), while LLVM-MCA rated both at 2.7
cycles. The loop's local instruction order cannot explain the production result.

A standalone generic `MaybeUninit` loop emits identical optimized IR and assembly with and without
`Copy`. The sensitivity therefore needs the real trait, closure, `Vec`, and monomorphization context.
This is currently evidence of compiler phase-order or code-quality sensitivity, not enough to claim
a rustc correctness bug or a specific LLVM bug. The next upstream step is to reduce the real
monomorph while retaining both the timing and whole-function delta, then bisect MIR/LLVM passes and
compiler versions. The executor needs only the no-drop property, so the selected API continues to
enforce `!needs_drop::<Output>()` without exposing the harmful, unnecessary `Copy` bound. See the
compact [Copy-ablation evidence](codegen/copy-ablation.md).

## Final results

Order: baseline, final, candidate, repeated twice. Values are median microseconds.

| Benchmark | Baseline 1 / 2 | Candidate 1 / 2 | Final 1 / 2 | Final/baseline |
| --- | ---: | ---: | ---: | ---: |
| `add_i64_constant` | 8.399 / 8.449 | 9.310 / 9.289 | 9.269 / 9.279 | 1.104 / 1.098 |
| `add_i64_nonnull` | 9.159 / 9.239 | 9.374 / 9.449 | 9.379 / 9.389 | 1.024 / 1.016 |
| `div_i64_nonnull` | 44.820 / 44.860 | 45.040 / 45.080 | 45.020 / 45.060 | 1.004 / 1.004 |
| `mul_i8_nonnull` | 6.209 / 6.199 | 4.719 / 4.699 | 6.389 / 6.409 | 1.029 / 1.034 |
| `mul_i16_nonnull` | 4.099 / 4.109 | 4.269 / 4.269 | 4.265 / 4.299 | 1.040 / 1.046 |
| `mul_i32_constant` | 26.440 / 26.440 | 18.890 / 18.840 | 18.690 / 18.700 | **0.707 / 0.707** |
| `mul_i32_nonnull` | 26.410 / 26.420 | 28.350 / 28.390 | 26.590 / 26.640 | 1.007 / 1.008 |
| `mul_i32_nullable` | 27.380 / 27.360 | 29.170 / 29.170 | 27.400 / 27.440 | 1.001 / 1.003 |
| `mul_i64_nonnull` | 23.200 / 23.350 | 30.010 / 30.050 | 23.460 / 23.430 | **1.011 / 1.003** |
| `mul_u8_nonnull` | 3.319 / 3.319 | 3.545 / 3.519 | 3.514 / 3.549 | 1.059 / 1.069 |
| `mul_u16_nonnull` | 2.609 / 2.609 | 2.429 / 2.429 | 2.789 / 2.810 | 1.069 / 1.077 |
| `mul_u32_nonnull` | 6.949 / 6.959 | 7.060 / 7.059 | 7.129 / 7.149 | 1.026 / 1.027 |
| `mul_u64_nonnull` | 19.180 / 19.210 | 30.370 / 30.400 | 19.370 / 19.380 | **1.010 / 1.009** |
| `sub_i64_constant` | 8.239 / 8.259 | 9.114 / 9.079 | 9.149 / 9.159 | 1.110 / 1.109 |

The six `land-*` logs preserve every final run. Narrow widths avoid the rejected zipped-iterator
experiment's 3x to 9x losses. Constant add/sub retain the untouched candidate's roughly 10% gap;
constant multiplication is faster than merge base. Division stays at parity.

## Generated code: confirmed evidence

```bash
CARGO_TARGET_DIR="$TARGET" cargo rustc -p vortex-array --lib --profile bench -- \
  --emit=llvm-ir,asm -C codegen-units=1 -C remark=loop-vectorize
```

Full output was about 1.85 GiB IR plus 1.02 GiB assembly and was deleted after extracting exact
production monomorphs into [`codegen`](codegen). These are not fixture or benchmark control loops.

Baseline, candidate, owned, and final signed `i64` use a scalar one-lane loop: one high/low `imulq`,
one store, `sarq`/`xorq` overflow evidence, register OR, and one backedge. Unsigned `u64` uses two
independent scalar `mulq` groups per backedge plus an odd remainder. Neither final loop has a hot
call, panic edge, bounds check, runtime alias check, or vector body. The second input length check is
an `llvm.assume`; loads and stores carry disjoint alias metadata; failure is a register `phi`.

Therefore host SIMD did not hide a deficient loop. The default build did not enable optional native
AVX features, and LLVM selected the same essential scalar high-half strategy as merge base. See
[`base summary`](codegen/base-codegen-summary.md),
[`final i64 assembly`](codegen/final-i64-mul-dense-s.md), and
[`final u64 assembly`](codegen/final-u64-mul-dense-s.md).

A separate minimal `target-cpu=native` experiment did form `<8 x i128>` operations in LLVM IR for
the widened `u64` product. The x86 backend still scalarized them into eight `mulq`/`imulq`
instructions, then used ZMM registers only to pack and reduce the scalar results. x86 has no true
wide 64-by-64-to-128 integer multiply here. Seeing a vector IR type or ZMM instruction is therefore
not evidence that the expensive multiply itself executed as SIMD.

`-C remark=loop-vectorize` emitted no remark attributable to the exact dense production loop. The
constant fallback source line had successes for other monomorphs and duplicated cost-model misses,
but diagnostics lacked function identity. Exact IR proves the measured specialization is scalar;
it cannot assign those remarks to it. The merge-base focused remark rebuild was cancelled, so no
merge-base missed-vectorization reason is claimed.

## Findings

Confirmed:

- Bounds checks are not the all-varying blocker; candidate dense multiply had no hot bounds edge.
- `SinkResult` already reduced to a register OR and did not impose a per-row `Result`.
- Output ownership materially helped but was insufficient alone.
- A typed indexed source restored stable parity for varying primitive tuples.
- An `Output: Copy` bound reliably triggers slower LLVM 21.1.2 production codegen; an inert marker
  does not. Source store/OR order is neither sufficient nor established as causal. The mechanism is
  an unresolved whole-function compiler interaction.
- The default x86 target prefers scalar high-half 64-bit multiply; SIMD is not the recovered speed.

Still inference:

- No single alias defect explains the original gap. Baseline and candidate had useful metadata too.
- The nearly identical dense inner loops do not explain all end-to-end timing. Surrounding control
  flow, placement, and instruction-cache effects remain candidates.
- Unattributed source-line remarks do not prove a missed-vectorization reason for one monomorph.

Rejected controls: checked unchecked access only partially helped and regressed some `u8` runs;
direct failure accumulation matched existing IR; safe zipped iterators caused 3x to 9x narrow
losses; a numeric `reduce_encoded` fast path recovered speed by duplicating shared policy; and a
primitive-binary visitor seam moved that specialization into generic execution. Earlier Apple work,
including the non-affine `index & mask` failure, remains in
[`NUMERIC_ROWFN_PLAN.md`](../../NUMERIC_ROWFN_PLAN.md).

## Why both visitor methods exist

`visit_prepared_deferred` represents an independent owned value and OR-reducible failure per row.
The executor allocates contiguous output, owns the store, and can use a typed indexed source. It is
intentionally limited to indexed inputs, fixed no-drop output, and a batch-deferred row error.

`visit_prepared_into` represents stateful construction: shared buffers, runtime-shaped layouts,
multiple coordinated builders, skip-capable output, drop-requiring values, non-indexed tuples, and
ordinary immediate or deferred `SinkResult` forms. Encoding those through the owned method would
either hide a mutable builder reference inside a supposed value, allocate a temporary per row,
forbid legitimate output, or duplicate lifting. Encoding numeric output only through the sink loses
the fact that each value and store are independent. These are distinct capabilities.

## Indexed source, specialization, and safety

`InputElement` is open and many elements are not contiguous. Sealed `ElementTuple` is the safe
composition point for unchecked reads after one length validation. Stable Rust cannot overlap a
blanket fallback for every tuple with a more specific associated dense source without
specialization. Runtime erasure would obscure the source type LLVM needs. The indexed capability is
therefore explicit and opt-in; only the proven primitive pair implements it today.

The executor reserves `row_count` slots and exposes exactly that many `MaybeUninit` values. It
validates varying lengths before `LaneZip`; `map_checked_into` validates output length. Either loop
writes every slot exactly once before `set_len`. On panic the vector length remains zero, and the
compile-time no-drop assertion makes abandoning initialized slots safe. Deferred errors are examined
only after initialization. Nullable lifting retries a deferred error over valid rows, so a failure
shaped value behind null cannot surface.

## Open improvements

- Investigate infallible owned output only with a measured caller; avoid a speculative result tree.
- Revisit constant add/sub only with exact production IR and a stable regression.
- Add indexed tuple/element families only for real consumers with a safe source.
- Re-run the store-order and `Copy` ablations after LLVM upgrades.
- Produce an upstream LLVM reproducer for those compiler sensitivities.
- Preserve assembly checks because throughput can hide compensating target-specific instructions.

The selected branch passed focused checks, 87 numeric tests, 3,385 nextest tests with one skipped,
73 doctests with 13 ignored, nightly formatting, all-target/all-feature clippy, and `diff --check`.
One intermediate 1.85 GiB IR copy hit `ENOSPC`; exact-final codegen later completed. The requested
`ROWFN_FIRST_PR_PROMPT.md` was absent from the repository, fetched refs, home tree, and worktrees.
