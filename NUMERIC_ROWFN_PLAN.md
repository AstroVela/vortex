<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Plan: fit the numeric binary operators onto `RowFn`

Working note, branch-only, like `SCALAR_FN_HANDOFF.md`. Written so this survives a conversation
compaction: everything needed to start is here, and nothing below depends on chat history.

## Where things stand

Branch `claude/strict-scalar-fn-abstraction-ah88x3`, pushed, five commits past `a8b6cd52a1`:

```
7dbd9044cb Take byte_length off the row framework
926f4590c6 Attach the scatter mask as validity instead of masking again
8942e9ace4 Export ElementRow, the slot a row closure writes through
48e260e002 Allocate row output through a zeroable placeholder
d0b938aacf Pin decode fallibility to the argument witness
```

Issues 9128, 9129 and 9130 are updated to the sink-only API and contain no stale design. PRs #9138
(`ct/l2-denorm-encoding` to develop) and #9136 (`ct/scalar-fn-baselines`, stacked on it) are both
mergeable; #9136 is blocked only on #9138 landing. CodSpeed baselines do not exist until both merge.

`byte_length` is no longer a row function, and `Bytes`/`BytesLen` are deleted. It measured 7.6-7.7x
slower than develop and is the case #9128 already excludes.

## Goal of this spike

Prove, or disprove, that the four arithmetic operators can move onto `RowFn` without changing the
`RowFn` API, without a second scalar function ID, and without touching serialization. Doing this
first is deliberate: it is the change most likely to force an API change, and discovering that after
tensor and geo are ported would mean reworking them.

## The design

`Binary` keeps everything and delegates only execution:

```rust
Operator::Add => ScalarFnVTable::execute(&NumericBinary, &NumericOperator::Add, args, ctx),
```

`NumericBinary` is a `RowFn` with `Options = NumericOperator` and `FALLIBLE = true`. It is **not**
registered as a public scalar function, so it needs no ID in the registry and appears in no
serialized expression. It is reached through the `ScalarFnVTable::execute` that the blanket impl
already provides.

Why this works, and each of these was verified against the code rather than assumed:

- **Nothing is lost.** `BooleanKernel` and `CompareKernel` exist with per-encoding pushdown; there is
  no `NumericKernel`. Unlike `not`, a numeric port gives up no encoding fast path.
- **The seam is already numeric-only.** All four arithmetic arms of `Binary::execute` funnel into
  `execute_numeric(lhs, rhs, NumericOperator, ctx)`, and `NumericOperator` is already its own enum in
  `crate::scalar`, so it is a ready-made `RowFn::Options`.
- **Fallibility is uniform.** `Binary::is_fallible` is false for the six comparisons plus `And`/`Or`
  and true for exactly the four arithmetic operators, so `FALLIBLE = true` on a numeric-only `RowFn`
  is exactly right. The options-independence of `RowFn::is_fallible` only bites when one function
  spans both families.
- **Strictness stays where it belongs.** `Binary::is_strict` is `!matches!(op, And | Or)` because
  Kleene `false AND null` is a valid `false`. `Binary` keeps owning that; `NumericBinary` never sees
  a boolean operator.
- **Decimal fits.** `OutputSink::sink_dtype(args)` sees the input dtypes, which is what
  `numeric_op_result_decimal_dtype(decimal_dtype, op)` needs.

## Steps

1. **Primitive path only, `Add` only.** A `NumericBinary` `RowFn` over `(T, T)` for one integer
   width, with a deferred-error sink that writes the wrapping sum and ORs an overflow bit. Delegate
   only `Operator::Add` from `Binary::execute` and leave the other three on `execute_numeric`.
   Success is: the existing `binary/numeric/tests.rs` suite passes unchanged.
2. **Widen to every primitive ptype**, through `match_each_native_ptype!` in `dispatch`. Confirm the
   compile-time witness check tolerates it, as it does for tensor widths.
3. **Add `Sub`, `Mul`, `Div`.** `Div` is the awkward one: see the risk below.
4. **Decide decimal.** Either a decimal input element plus a sink that carries the result precision
   and scale, or leave `DType::Decimal` on `execute_numeric` and delegate only the primitive path.
   Leaving it is a legitimate outcome for the spike and possibly for the first PR.
5. **Delete the replaced code** only once benchmarks agree, not before.

## Risks, in the order they are likely to bite

- **`Div` already has a per-type strategy.** `primitive.rs` carries `CHECKED_VALUE_LOOP` and
  `DIV_CHECKS_IN_VALUE_LOOP`, set per type, so division checking is not uniform. A single row closure
  may not express it, and `Div` may have to stay behind.
- **The existing implementation is tuned, not naive.** `checked.rs` has `checked_lanes` and
  `checked_apply_lanes` taking a `valid_rows: &Mask` and returning `Result<Buffer<T>, usize>` with the
  failing index. The port is replacing real engineering, so parity is not a given. This is the reason
  the CodSpeed gate on the `binary_ops` names from #9136 matters.
- **Two declarations of the result dtype must agree.** `Binary::return_dtype` is what the expression
  layer uses, while `reconcile_return` checks the kernel output against `NumericBinary`'s
  sink-derived dtype. Cover every operator and dtype pair with a test that asserts they match.
- **Error messages are part of the contract.** `primitive.rs` defines `ERROR` per operator, such as
  `"integer overflow in checked add"`, and `numeric/tests.rs` asserts on failures. The deferred-error
  sink reports once from `finish`, so the message must be preserved and the error must still be
  raised for the same inputs.
- **Overflow behind a null row must stay invisible.** `numeric/tests.rs` has
  `test_decimal_overflow_on_null_lane_ignored`. The lifting's deferred-error retry over valid rows is
  exactly this behavior, so the test should pass, but it is the first thing to check.

## Verification

```bash
cargo nextest run -p vortex-array
cargo clippy --all-targets --all-features -p vortex-array
cargo +nightly fmt --all
cargo test --doc -p vortex-array
```

The numeric suite specifically:

```bash
cargo nextest run -p vortex-array scalar_fn::fns::binary
```

Performance gate is CodSpeed on the `binary_ops` names from #9136, once it lands on develop. Locally,
`cargo bench -p vortex-array --bench binary_ops` with two runs, fastest and median, machine stated.

## What this spike is not

Not a PR. Not a deletion of `execute_numeric`. Not decimal support unless step 4 turns out easy. The
output is an answer to "does this fit cleanly", plus whatever the answer implies for #9129's API.

## Outcome

It fits, with no change to the `RowFn` API and one change to the machinery.

Steps 1 through 3 landed together rather than in sequence: once the sink existed, widening it through
`match_each_native_ptype!` and adding the other three operators was the same code. Step 4 leaves
decimal on `execute_numeric_decimal`, which the delegation makes easy since `execute_numeric` still
owns the dtype split. Step 5 deleted the replaced primitive execution, which the measurements below
justify.

### What the design turned out to be

`Binary::execute` is untouched. `execute_numeric` keeps its validation, its error messages, its empty
short circuit, and its primitive/decimal split, and only `execute_numeric_primitive` changed: it
builds a `VecExecutionArgs` and calls `ScalarFnVTable::execute(&NumericBinary, &op, ..)`. Everything
the old implementation did around the arithmetic (decoding, the constant-operand collapse, the
all-constant fold, the null-constant short circuit, output allocation, nullability widening, masking,
and the valid-row retry after an overflow behind a null) is now the lifting's.

`NumericOperator` became the options type, which needed `Hash` and a `PersistableOptions` impl.
Nothing serializes it, since `NumericBinary` is unregistered, but encoding it as `vortex.binary` does
keeps a future registration wire-compatible.

Three things the old code carried are gone because the row framework removes the distinction they
existed for:

- `CHECKED_VALUE_LOOP` and `DIV_CHECKS_IN_VALUE_LOOP` chose between a split value/error scan and a
  one-pass early-exit kernel, because for integer division the split loop only added a second scan.
  A row kernel produces the value and the error bit in the same pass, so there is one loop shape and
  no choice to make. `div_i64` got 1.11x faster.
- `checked_apply_lanes` had no caller left. `checked_lanes` stays for decimal.
- `PrimitiveOperand` moved to `compare/primitive.rs`, its only remaining user.

### The one machinery change: `DeferredError` is a byte, not an `i64`

This was not predicted and it is the substantive finding. `DeferredError` held an `i64` so a kernel
could OR raw words and read failure off the sign bit. The bit is reduced once per row alongside the
kernel's own arithmetic, so a 64-bit accumulator caps how many rows a vector of the reduction covers
regardless of the element width. Measured against `Mul`, which is where the value work is widest:

| width | `i64` accumulator | byte accumulator |
| --- | --- | --- |
| `i8` | 3.5x slower than baseline | 1.05x |
| `i16` | 2.05x | 1.03x |
| `i32` | 1.28x | 1.01x |
| `i64` | 1.13x | 1.12x |

`from_sign_bit` went with it. Its premise (that reusing an already-computed word is free) is what the
measurement contradicts, and its only caller was the `row_fn_executor` benchmark, now on
`DeferredError::new`. `row_checked_add` is unchanged at 13.99 µs, matching `specialized_checked_add`
exactly, so nothing was given up at `i64`.

### Benchmarks

`binary_ops`, 65536 rows, divan fastest of 100 samples, best of two runs, Apple M4 Max. The six
benchmarks over unchanged code (`add_decimal_*`, `and_bool`, `eq_i64`, `lt_i64`, `or_bool`) all moved
by less than 1%, which is what makes the rest comparable.

| benchmark | before | after | |
| --- | --- | --- | --- |
| `div_i64_nonnull` | 38.95 µs | 35.24 µs | 1.11x faster |
| `add_i64_nonnull` | 13.45 µs | 13.83 µs | 1.03x slower |
| `add_i64_nullable` | 14.24 µs | 14.37 µs | 1.01x slower |
| `sub_i64_constant` | 12.87 µs | 13.33 µs | 1.04x slower |
| `mul_i8_nonnull` | 2.665 µs | 2.790 µs | 1.05x slower |
| `mul_i16_nonnull` | 4.957 µs | 5.082 µs | 1.03x slower |
| `mul_i32_nonnull` | 7.582 µs | 7.666 µs | 1.01x slower |
| `mul_i32_nullable` | 8.29 µs | 8.12 µs | 1.02x faster |
| `mul_i32_constant` | 7.499 µs | 7.999 µs | 1.07x slower |
| `mul_i64_nonnull` | 27.16 µs | 30.45 µs | 1.12x slower |
| `mul_u8_nonnull` | 22.70 µs | 27.08 µs | 1.19x slower |
| `mul_u16_nonnull` | 20.33 µs | 26.33 µs | 1.30x slower |
| `mul_u32_nonnull` | 24.16 µs | 29.37 µs | 1.22x slower |

### The open regression: unsigned `Mul` and `mul_i64`

Every regression that survives is on a loop that does not vectorize, and there are two separate
causes stacked on top of each other.

**Underneath: unsigned narrow `mul_error` never vectorizes.** Not in Vortex, and not in a twelve-line
standalone program with no Vortex code in it. Compiled to ARM64, the signed check
(`p < MIN || p > MAX` over the widened product) emits `smlal.8h` and `cmhi.8h` and runs sixteen lanes
at a time; the unsigned check (`p > MAX`) emits `ldrb`/`mul`/`strb` and runs one. Eight rewrites of
the unsigned check (shift, mask, round-trip truncation, `checked_mul`, `overflowing_mul`, widening to
`u32`, mirroring the signed two-sided form) all produced byte-identical scalar code, so LLVM is
canonicalizing every spelling to `umul.with.overflow`, which has no vector lowering. Removing the
check entirely takes `mul_u8_nonnull` from 27.1 µs to 1.5 µs.

This is why `mul_u8` costs 22.7 µs on develop against `mul_i8`'s 2.7 µs for the same shape of code,
and why `mul_i64` (which uses `overflowing_mul` explicitly) sits in the same 20-30 µs band at every
width. It predates this work and affects develop identically. It deserves its own issue.

**On top: the row loop pays for bounds checks.** A scalar loop has nothing to hide them behind.
Replacing the two input reads and the one output write with `get_unchecked` (ABBA, two rounds,
medians, both rounds agreeing):

| benchmark | row loop | with unchecked indexing |
| --- | --- | --- |
| `mul_u8_nonnull` | 27.41 / 27.70 µs | 25.12 / 25.62 µs |
| `mul_u16_nonnull` | 27.04 / 26.87 µs | 24.66 / 24.60 µs |
| `mul_u32_nonnull` | 30.58 / 29.83 µs | 27.10 / 27.66 µs |
| `mul_i64_nonnull` | 31.20 / 31.24 µs | 29.41 / 29.79 µs |
| `mul_i8_nonnull` | 2.832 / 2.915 µs | 2.915 / 2.874 µs |
| `add_i64_nonnull` | 14.04 / 14.41 µs | 13.66 / 13.62 µs |

So bounds checking is 5-9% on the scalar loops and nothing measurable on the vectorized ones, which
accounts for about half the remaining gap. The other half is the rest of the per-row shape against
`map_checked_into`, which is a `get_unchecked` loop with a register-resident `bool`.

`execute_row_sink_prepared` already tries to enable this: `varying_len_matches` and
`row_count_matches` are asserted before the loop for exactly this reason, and the doc on
`row_count_matches` says so. They are not achieving it. An isolated reproduction of the same shape
(nested references, generic accessors, a `&mut` row slot) does eliminate its bounds checks, so the
mechanism is not simply the double indirection and is not yet pinned down.

The sound fix is an API change rather than an `unsafe` block: `InputElement::get_varying` and
`OutputSink::row` are safe public methods, so neither may skip its check on the framework's promise.
Giving `varying(column, row_count)` the row count and having it return a view already sliced to that
length would make `column[index]` provably in bounds, delete `varying_len_matches` and
`varying_len`, and move a runtime assert into the type. That is a change to #9129's surface and to
every element implementation, so it belongs to that issue rather than to this port.

### What this implies for #9129 and #9130

- The `RowFn` API needed nothing. No new visit method, no options-aware `sink_dtype`, no return
  witness. The options-independence of `RowFn::is_fallible` does not bite, because fallibility is
  uniform across the four arithmetic operators.
- `DeferredError`'s width is part of #9130's machinery contract and should be recorded there: a
  deferred bit is only free when the accumulator is no wider than the element.
- `varying_len_matches` and `row_count_matches` exist to buy bounds-check elimination and do not buy
  it. Worth 5-9% on any row kernel whose arithmetic does not vectorize. Slicing the varying view to
  the row count inside `varying()` would get it without `unsafe`.
- A batch-constant operand still costs a per-row `ArgColumn` branch, since `ElementTuple::varying`
  declines whenever any argument is constant and `prepare` cannot remove the read. Worth 1.07x on
  `mul_i32_constant` and nothing measurable on `sub_i64_constant`. Not a blocker, but it is the
  remaining structural gap against a hand-written kernel that hoists its constant into a register.

### Verification

`cargo nextest run -p vortex-array` (3156 passed), `cargo nextest run --workspace`,
`cargo clippy --all-targets --all-features`, `cargo +nightly fmt --all`, `cargo test --doc -p
vortex-array`. The whole of `binary/numeric/tests.rs` passes unchanged, including
`test_decimal_overflow_on_null_lane_ignored` and the three integer-error tests that pin the valid-row
retry.
