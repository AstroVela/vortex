<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Handoff: the row scalar-function framework

This is the concise source of truth for the branch. `STRICT_SCALAR_FN_RESEARCH.md` keeps the full
design history, rejected alternatives, measurements, and generated-code evidence.
`NUMERIC_ROWFN_PLAN.md` records the numeric-binary migration and its narrower performance boundary.
All three are branch-only working notes for agents. They are not intended to land with the API.

The public design lives in these tracking issues, which now match the implementation:

- [#9128, Row-oriented scalar functions](https://github.com/vortex-data/vortex/issues/9128)
- [#9129, Define the `RowFn` API](https://github.com/vortex-data/vortex/issues/9129)
- [#9130, Execute `RowFn` over Vortex arrays](https://github.com/vortex-data/vortex/issues/9130)

The branch is `claude/strict-scalar-fn-abstraction-ah88x3`. It is publicly linked from #9128, so do
not rewrite or delete its history. Commit `4becc863ae` contains the final API simplification. Push
only when explicitly requested.

## Next action: rerun the benchmarks on x86

The next session will run on an x86 machine. Rerun the performance comparison there before treating
the implementation as complete. Do not reuse the Apple timings as the final runtime result.

The production benchmark baseline from #9136 is on `develop` at `9a482c0230`. Fetch the latest
`origin/develop`, record the exact baseline and candidate commits, and run the same public benchmark
binaries at both revisions:

```bash
cargo bench -p vortex-array --bench binary_ops
cargo bench -p vortex-array --bench like
cargo bench -p vortex-tensor --bench l2_norm
cargo bench -p vortex-tensor --bench inner_product
cargo bench -p vortex-tensor --bench cosine_similarity
cargo bench -p vortex-tensor --bench normalized
cargo bench -p vortex-geo --bench binary_predicates
cargo bench -p vortex-geo --bench distance
cargo bench -p vortex-geo --bench envelope
cargo bench -p vortex-geo --bench predicate_bbox
```

Run each revision at least twice in alternating order. If the host allows it, pin the process to one
core. Record the timer and CPU configuration, and compare both fastest and median values. The
benchmark binaries and public names are now shared with `develop`, so the comparison no longer
needs a frozen benchmark-local implementation as its primary control.

Also run the branch-only `vortex-geo` `null_strategies` diagnostic. It forces branch-and-skip and
filter-and-scatter for the measured nullable geometry shapes. Confirm that automatic selection uses
the faster mechanism for one costly decode at 50% survivors and for two costly decodes at about 81%
survivors. This is the x86 runtime check that remains after the LLVM comparison.

```bash
cargo bench -p vortex-geo --bench null_strategies
```

If a stable benchmark regresses, inspect optimized LLVM IR again. The previous cross-compile proves
that the API cleanup preserved the x86_64-v3 loop shape. The x86 run must confirm runtime effects
from the revised null selector and the target CPU's vectorizer and branch predictor.

## The API in one screen

`RowFn` is the author-facing function trait. A function gives the framework its exact argument
names, a conservative fallibility declaration, function-owned persistence, and a value-blind
dispatch over concrete input and sink types:

```rust
impl RowFn for Example {
    type Options = ExampleOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.example");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(encode(options)?))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        decode(metadata, session)
    }

    fn dispatch<V: RowVisitor>(
        &self,
        options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        validate_options(options, args)?;
        visitor.visit_prepared_into::<(InputA, InputB), ElementSink<Output>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| {
                *output = compute(lhs, rhs);
            },
        )
    }
}
```

There are no argument or return witness types. The dispatched tuple is the argument declaration,
the sink owns the output representation, and the row result names the error behavior. Planning
runs the same dispatch as execution and checks the selected types against the function constants.

## The extension boundary

The framework is deliberately not sealed wholesale. Function authors need to add decode and output
primitives for their own scalar functions. Only the executor mechanics are closed.

| API | Boundary | Why |
| --- | --- | --- |
| `RowFn` | open | Defines a scalar function and selects concrete execution types. |
| `InputElement` | open | Adds a new scalar decode primitive, including crate-local domain types. |
| `OutputElement` | open | Adds an ordinary one-value-per-row output primitive. |
| `OutputSink` | open | Adds a custom output representation or builder. |
| `RowVisitor` | sealed | Executor-owned dispatch mechanism with one supported implementation. |
| `ElementTuple` | sealed | Executor-owned tuple recursion, with built-ins through arity 12. |
| `SinkResult` | sealed | Executor-owned loop and error facts trusted by the blanket vtable. |

`ElementTuple` being sealed does not prevent a function from adding a decode primitive. Implement
`InputElement` and use it inside one of the supplied tuples. Likewise, a function with two logical
outputs should define one `OutputSink` whose state has two fields. The framework does not need a
second tuple or composite-sink abstraction.

The supplied `SinkResult` forms are:

- `()` for infallible rows;
- `VortexResult<()>` for an error that must stop immediately; and
- `bool`, `u8`, `u16`, `u32`, or `u64` for error evidence OR-reduced after the loop.

The unsigned evidence widths let each kernel choose a word no wider than its element type. That is
load-bearing for vectorization, particularly for checked unsigned multiplication.

## Function-owned persistence

Persistence belongs to the function ID, not to the Rust options type. `RowFn::Options` has no
serialization supertrait. The `RowFn::serialize` and `RowFn::deserialize` hooks have conservative
defaults, and registered functions override them when their existing wire contract requires it.

This has three useful consequences:

- two functions may reuse an options type while choosing different formats;
- a function may deliberately be nonserializable even if another function serializes the same
  options type; and
- an unregistered helper such as `NumericBinary` needs no dummy persistence implementation.

Tensor and geo functions keep their explicit existing formats. Do not introduce a blanket options
wire format or infer serializability from `Options`.

## One sink abstraction

`OutputSink` is the complete output contract. It owns the output dtype, allocation, row storage,
row lookup, length proof, and final array construction. `ElementSink<T>` covers the common case. Its
row type is `&mut T`, so the closure writes with ordinary assignment.

Custom sinks remain available for a real output shape that cannot use `ElementSink`. The unused
public `TensorSink` was removed. No current tensor row function returns tensor-valued rows, and a
90-line public runtime-shaped sink was not justified without a user. Add a custom sink when a real
function needs one, using one sink struct even when it owns several builders.

Every current sink produces an all-valid child column. The blanket vtable can therefore derive the
function result validity from the input validities. Nullable row outputs remain out of scope. A
sink that emits its own nulls must change that derivation in the same change.

`OutputSink::sink_dtype` must return a non-nullable dtype. `SUPPORTS_SKIPPED_ROWS` says whether
branch-and-skip may leave placeholder rows behind the result validity. `ERRORS_ARE_DEFERRED` says
whether the sink accepts accumulated error evidence at `finish`.

## Dispatch and fallibility

`dispatch` must be pure in `(options, args)`. It sees dtypes, not array values. Planning and
execution both call it, so value-dependent preparation belongs inside `visit_prepared_into`.

The executor statically checks each dispatched visit:

- the tuple arity equals `ARG_NAMES.len()`;
- a fallible decoder, early-error result, or deferred result implies `RowFn::FALLIBLE`;
- deferred evidence requires both `RowFn::FALLIBLE` and a sink with
  `ERRORS_ARE_DEFERRED = true`; and
- the sink and result agree about their error contract.

The implications are intentionally one-way. `FALLIBLE = true` is a conservative function-level
claim, while a particular dtype dispatch arm may be infallible.

`prepare` must not be load-bearing for validation. Empty batches may bypass value preparation, and
the executor needs its safety and fallibility facts before it runs the closure.

## Null execution policy

The old public `NullHandling` enum and argument witness were removed. Authors do not select an
execution mechanism. The executor derives a private row policy from the dispatched input and result
types:

- `Dense` may execute over garbage behind nulls and masks afterward;
- `DenseWithRetry` may execute densely, then retry valid rows when deferred evidence reports an
  error; and
- `ValidOnly { filtered_decode_cost }` guarantees that the row closure sees only valid rows.

An early-failing row or a decoder that is not dense-safe must use valid-only execution. A deferred
kernel may use dense execution because it writes a legal provisional value for every row. If only
garbage behind nulls reports an error, the valid-row retry discards it.

Valid-only execution has two mechanisms. Filter-and-scatter shrinks inputs before decoding.
Branch-and-skip decodes the original batch and visits set bits from the conjoined validity mask. A
sink that does not support skipped rows automatically falls back to filter-and-scatter.

The selector needs more than a boolean "decode shrinks" flag. Every `InputElement` declares an
additive `FILTERED_DECODE_COST`, defaulting to zero. `ElementTuple` sums the costs across arguments:

- cost 0 always prefers branch-and-skip;
- cost 1 prefers branch-and-skip at 50% or more surviving rows; and
- cost 2 or greater prefers branch-and-skip at 85% or more surviving rows.

This distinction comes from the x86 measurement in #9128. One nullable geometry input at 50% nulls
favored branching, while two independently nullable geometry inputs at 10% nulls each, about 81%
survivors, favored filtering. OR-ing a per-argument flag loses exactly that distinction.

The values are still a coarse heuristic. There is no evidence yet to separate cost 2 from cost 3,
and the batch-size crossover has not been measured. `NullStrategy` remains only as a test-harness
seam for forcing a mechanism. Do not expose the private row policy as an author contract.

## Performance and generated-code evidence

The older Ryzen 9 7950X AVX-512 measurements remain the production-performance record in the
[#9128 follow-up](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802). They
also supplied the per-argument null-selection evidence above.

The final API cleanup was checked separately against its parent, `53c51d803c`, by cross-compiling
the optimized `row_fn_executor` benchmark for `x86_64-apple-darwin` with `target-cpu=x86-64-v3`.
After normalizing symbol names and metadata, the vector and reduction blocks were identical for all
three executor shapes:

- ordinary wrapping add through `ElementSink`;
- checked add with deferred evidence; and
- wrapping add through a custom sink.

The wrapping loops retain 256-bit `<4 x i64>` loads, adds, and stores. The checked loop retains the
same vector loads and adds, derives overflow with vector xor/and/compare operations, accumulates
`<4 x i1>` with vector OR, and reduces after the loop. None of the vector bodies contains a call or
panic path. Scalar tails are unchanged.

The production tensor benchmarks were also cross-compiled before and after the cleanup. Normalized
arithmetic sequences and counts match for `l2_norm`, inner product, and cosine similarity. Their
ordered floating-point reductions are scalar-unrolled in both revisions because LLVM preserves the
strict reduction order. The cleanup did not remove vectorization because those reductions were not
vectorized before it.

Native Apple M4 Max timings used 65,536 rows, two alternating before/after runs, 100 samples, and a
0.5-second minimum per arm. RowFn median deltas ranged from 1.11% faster to 0.94% slower. Fastest
deltas stayed within about 0.17%, while specialized controls drifted by as much as 3.7% in their
medians. There is no measurable native regression from the API cleanup.

This does not replace the required x86 runtime run above. Cross-target IR proves that the hot loop
shape survived, not that the revised null selector has the expected branch-predictor behavior on
x86.

## Current implementation and checks

The implementation includes production users in `vortex-array`, `vortex-tensor`, and `vortex-geo`.
`NumericBinary` is an unregistered `RowFn` used only for primitive arithmetic execution. Decimal
arithmetic keeps its existing path. The stable public-path benchmark baseline landed as #9136.

The checks recorded for the final API state are:

- 67 focused RowFn tests;
- 179 `vortex-tensor` tests;
- 230 `vortex-geo` tests;
- `cargo +nightly fmt --all`; and
- full workspace clippy, with `PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1` because the host
  `/usr/bin/python3` is 3.9 while the workspace requires the Python 3.11 stable ABI.

The generated-code comparison and native timing evidence are described above and in the final
section of `STRICT_SCALAR_FN_RESEARCH.md`.

## Remaining boundaries

- Complete the required x86 production and forced-null-strategy benchmark run above before treating
  the thresholds or overall performance as settled.
- Keep nullable outputs separate until the first real function can define the validity contract.
- Do not add another sink composition abstraction. Put multiple builders in one custom sink.
- Do not add a general runtime-shaped sink until a production function needs one.
- Keep pattern compilation and other state shared across rows outside `RowFn` when it cannot be
  represented as batch preparation.
- Use emitted optimized IR as a gate for numeric changes near LLVM's vectorization boundary, then
  use the stable #9136 benchmark names for runtime confirmation.

## Repository rules for the next agent

Follow `AGENTS.md`. Keep public APIs small, run narrow checks before workspace-wide checks, and
report blocked checks separately from passing ones. Preserve unrelated working-tree and staging
state. Every commit must include the required `Signed-off-by` trailer.
