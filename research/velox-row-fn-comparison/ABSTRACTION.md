# Where the abstractions differ

This document compares the two designs one dimension at a time. Code citations name the file in
either tree. Vortex paths are relative to the [`ct/row-fn-*` stack](https://github.com/vortex-data/vortex/tree/28d61448e). Velox
paths are relative to
[`facebookincubator/velox` at `54fea71cc`](https://github.com/facebookincubator/velox/tree/54fea71cc).

## 1. The execution pipeline and who owns each step

Both frameworks implement the same pipeline. The table maps each step to its owner.

| Step | Vortex `RowFn` | Velox simple function |
| --- | --- | --- |
| Validate input types | `InputElement::validate`, called from planning | Registry signature binding at plan time |
| Resolve the output type | `dispatch` selects it, `OutputElement::element_dtype` or `OutputSink::return_dtype` supplies it | Registration declares it, `SignatureBinder` re-derives it |
| Decode each input once | `InputElement::decode` into `Column`, `view()` before the loop | `DecodedVector`, or direct flat/constant readers on the fast path |
| Handle degenerate batches | Batch executor: all-null, null constant, all-constant broadcast | `Expr` constant folding, `removeSureNulls`, empty-selection check |
| Hoist constant work | `prepare` closure over `ConstElems` per batch | `initialize` per query and thread, constant readers per row |
| Run the row loop | `execute_owned` / `execute_sink`, framework-owned | `iterate` picks one of five loop variants, framework-owned |
| Build the output | `OutputElement::build` or `OutputSink::finish` | `VectorWriter` over a vector from `ensureWritable`, or a reused input |
| Apply null semantics | Strict validity masked on after the loop | Nulls decided per row or deselected before the call |
| Validate the result | Length, dtype, all-valid check at the function boundary | None (the adapter trusts itself) |

The last row is a real difference in posture. Vortex treats the row kernel as untrusted at the
batch boundary: `finalize_kernel_output` re-checks length, dtype, and that every produced row is
valid, and `finalize_reduced` additionally proves an encoded reduction introduced no nulls on valid
rows. Velox has no equivalent because the adapter and the kernel are one compiled unit with no
boundary between them.

## 2. Where types bind

**Velox: registration-time binding, registry-side dispatch.** `registerFunction<Fn, Ret, Args...>`
instantiates `UDFHolder<Fn<VectorExec>, Ret, Args...>` and stores a factory in a map keyed by name
and signature. At [plan time](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/expression/SimpleFunctionRegistry.cpp), `SignatureBinder` matches call-site types against stored signatures,
a priority lattice breaks ties (concrete signatures beat variadic, variadic beats generic, generic
beats variadic-of-generic, then most concrete types wins), and the chosen entry constructs a
monomorphic `VectorFunction`. After compilation the engine cannot tell the function was authored
row-at-a-time. All type dispatch is finished before execution starts.

**Vortex: execution-time binding, function-side dispatch.** A `RowFn` is one registered object.
Its `dispatch` receives `&[DType]` and must call the visitor at concrete Rust types:

```rust
fn dispatch<V: RowVisitor<Self::Options>>(
    &self,
    options: &Self::Options,
    args: &[DType],
    visitor: V,
) -> VortexResult<V::VisitResult>;
```

The visitor is the interesting part. It is sealed, and its three required methods are generic in
the element tuple and output type, which makes `dispatch` a rank-2 function: the framework supplies
a polymorphic continuation, and the function picks the types. Three framework visitors implement
it: `BatchPlanner` (returns a `BatchPlan`: output dtype plus null policy), `ExecuteRows` (dense),
and `ExecuteValidRows` (skip-invalid). The same `dispatch` body therefore serves planning and both
execution shapes. The executor re-runs it per phase and verifies that the runs agree (`ensure_plan`
in `visitor/execute.rs`), because a `dispatch` that depends on anything but `(options, args)`
breaks the plan.

Consequences of the two choices:

- Velox's registry must model genericity explicitly (`Generic<T1>`, physical signatures, the
  priority lattice) because binding happens where the function body is out of reach. Vortex needs
  none of that machinery: genericity is a `match` in `dispatch`.
- Vortex pays a per-batch dispatch cost (two visitor walks, `match_each_native_ptype!` and friends)
  where Velox pays once per compiled expression. At 64K-row batches the Vortex evidence attributes
  visible small-batch overhead to exactly this fixed cost, with parity at 1M rows.
- Velox can hold multiple independent implementations of one logical signature (custom types over
  the same physical layout, disambiguated by physical signature). Vortex holds one function object
  per `ScalarFnId` and any such splitting happens inside `dispatch`.

## 3. Null handling

This is the deepest design difference between the two frameworks.

**Velox encodes null semantics in which method exists.**

| Author writes | Derived property | Engine behavior |
| --- | --- | --- |
| `call` only | `is_default_null_behavior = true` | `Expr::removeSureNulls` deselects known-null rows before `apply`, adapter re-checks `isSet(row)` per row for the rest |
| `callNullable` | `is_default_null_behavior = false` | Function receives pointers, null pointer means null input, function sees every selected row |
| `callNullFree` only | `is_default_contains_nulls_behavior = true` | Any null anywhere in an input, including inside nested values, yields null. The function is not called |
| `call` plus `callNullFree` | fast path | Adapter scans the batch once, uses the null-free loop when no input can contain nulls |

Two levels cooperate: the expression evaluator removes rows it can prove null when the function
propagates nulls
([`Expr::removeSureNulls`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/expression/Expr.cpp#L1191)),
and the adapter picks one of its loop variants (`allNotNull`, null-free, ASCII, general) once per
batch. In the mixed case the null check is a `LIKELY` branch per row, and a null
row's slot is skipped (nulls were optimistically cleared up front).

**Vortex encodes null semantics in the planner.** The function has no choice: a `RowFn` is strict,
and stricter than `ScalarFnVTable::is_strict`, because valid inputs must produce valid outputs.
What varies is execution strategy, derived from capabilities the element and sink types declare
([`plan.rs`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/visitor/plan.rs#L145)):

```rust
pub(crate) const fn for_owned_output<Args: ElementTuple>() -> Self {
    if Args::DENSE_SAFE && Args::DECODE_INFALLIBLE {
        Self::Dense
    } else {
        Self::ValidOnly
    }
}
```

- `Dense`: run every row, including the unspecified values behind nulls, then mask the output.
  Requires every element to be `DENSE_SAFE` (payloads behind nulls are readable garbage, as for
  primitives and tensors) and decoding to be infallible.
- `DenseWithRetry`: same, for kernels that defer failure evidence. A deferred failure can come from
  the garbage behind a null row, so the executor filters to valid rows and re-runs before reporting.
- `ValidOnly`: never evaluate an invalid row. At runtime this tries skip-invalid first (needs a
  null-tolerant decode for every input and a `skipped_rows_initializer` on the sink, as
  [geometry columns](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-spatial/src/scalar_fn/row.rs) provide for points and polygons), and otherwise filters every input, runs dense, and
  scatters the compact result back under a null mask.

The philosophical difference: **Velox does not evaluate a null row unless the function asks to see
nulls (`callNullable`). Vortex prefers to evaluate it.** Velox's
choice keeps semantics simple (no observable effects from garbage) at the price of a per-row branch
or a row-set shrink. Vortex's choice keeps the hot loop branch-free and vectorizable at the price
of real machinery: `DENSE_SAFE` as a trust contract on element types, the three-state
`RowExecution` result, the deferred retry, and the output-side rule that a kernel handed garbage
must still terminate without a panic. The Vortex evidence branch shows why the
machinery pays: dense checked-add over two nullable columns runs within a few percent of the
non-null case, because validity never enters the loop.

One place the systems converge: both special-case the all-valid batch to a loop with no null
handling at all, and both special-case all-null to a constant result without calling the kernel.

**Nullable outputs.** Velox rows can declare their own null (`bool call`). Vortex rows cannot, and
the framework exploits that: output validity is exactly the conjunction of input validities, which
`validity()` can push down as an expression without executing anything. See `EXAMPLES.md`
section 6 for the trade.

## 4. Constants

Velox has three constant mechanisms at three lifetimes:

1. `Constant<T>` in a registration signature makes constancy part of the function's type contract
   (`rand(Constant<int32_t>)`).
2. `initialize` receives plan-time constant values once per query and thread, and the function
   caches derived state in member fields.
3. At row level, a constant vector is read through `ConstantVectorReader`, whose index multiplier
   is 0, so every row reads element zero branchlessly.

Vortex has two mechanisms at two lifetimes, both per batch:

1. Whole-batch: if every input is constant and non-null, the executor evaluates one row and
   broadcasts the result as a `ConstantArray` (`broadcast_one_row`).
2. Per argument: [`ArgColumn::decode`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/types/element/tuple/element_tuple.rs) detects a batch-constant input (looking through masked and
   extension wrappers), decodes exactly one row of it, and the tuple's `ConstElems` hands the
   `prepare` closure `Some(value)` per constant argument. The row loop then either drops constant
   checks entirely (`views_if_no_consts`, when nothing is constant) or keeps the constant selection
   visible so LLVM unswitches it, and element types can encode broadcast structurally, as
   `TensorRows` does with a stride of 0.

The scoping difference matters in both directions. Vortex sees constants that only exist at
execution time, because Vortex arrays carry their encoding into the kernel, and constant-ness is an
encoding. Velox sees constants earlier and once, which supports expensive per-query preparation
(compiled regexes) without re-deriving per batch. Vortex's `SpatialContains` needed a lazy
`OnceCell` to get similar economics per batch. Neither subsumes the other.

## 5. Encodings and decoding

Velox has three vector encodings that reach a function: flat, constant, and dictionary. The
expression evaluator peels dictionaries above the function when it can (evaluating on base vectors
and rewrapping), `DecodedVector` flattens whatever remains into a base-plus-indices view, and the
adapter specializes the whole row loop per flat/constant combination for up to three arguments
(2^N instantiations, guarded by `specializeForAllEncodings`). Functions that want more, such as
subscript returning a dictionary over its input, leave the simple framework.

Vortex has an open-ended encoding zoo, so "decode" is a real step with real cost, and the framework
makes it a typed contract: [`InputElement::decode`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/types/element/input.rs) canonicalizes one column into `Column` once per
kernel invocation, and `View` re-borrows it so the loop reads through loop-invariant pointers.
Because decoding can be expensive and encoding-aware answers can be cheap, the escape hatch lives
inside the function (`reduce_encoded`, probed once on the original arrays before any decode or
filter), and encoding-awareness can also live in decode itself (`TensorRows` reading constant
storage with stride 0). The skip-invalid strategy adds a second decode contract,
`decode_null_tolerant`, for representations whose ordinary decode fails on the garbage behind
nulls.

The structural summary: Velox normalizes encodings away above the function and compensates with a
separate vector-function tier. Vortex lets encodings reach the function boundary and gives the
function typed hooks at three depths (reduce the whole call, specialize the decode, or just read
rows).

## 6. Outputs

Velox writes through per-type writers into a vector the adapter prepared (`ensureWritable`), with
three notable optimizations: fixed-width results write through a raw pointer with nulls
pre-cleared, a singly-referenced flat input of the right type can become the output in place, and
string results can alias input buffers (`setNoCopy` plus `reuse_strings_from_arg`). Complex writers
(`ArrayWriter`, `MapWriter`, `RowWriter`) append directly into child vectors.

Vortex splits the output contract in two, and the split is measured rather than aesthetic:

- `OutputElement`: the row closure returns an owned value, the executor writes it into
  uninitialized spare capacity, and `build(Vec<Self>)` makes the column. Requires no drop glue
  (a `const` assertion) so an unwind can abandon the buffer.
- `OutputSink`: the executor allocates once (`with_capacity`), borrows a `Rows` view so buffer
  metadata is loop-invariant, hands each closure a `Row` handle, and consumes the sink in an
  `unsafe fn finish`. `UninitElementSink` exposes `MaybeUninit` slots and demands the
  `InitializedElement` token back.

The [history](https://github.com/vortex-data/vortex/blob/2beac64a4/research/rowfn-reconstruction/OPTIMIZATION.md) explains the split: the first shared executor was sink-only, and it cost 29% on
signed and 59% on unsigned 64-bit checked multiply because it hid the independence of owned
primitive outputs. The owned form was reinstated on that evidence. Velox's single writer model does
not hit the same wall because its adapter writes fixed-width results through a raw pointer anyway,
which is the owned form in all but name.

What Vortex's sinks lack today: in-place input reuse, buffer aliasing for variable-width data, and
any complex-type writer. What Velox's writers lack: uninitialized output (nulls are pre-cleared and
slots always exist) and the write-token proof that makes skipping initialization sound.

## 7. Errors

| Property | Velox | Vortex `RowFn` |
| --- | --- | --- |
| Row error channel | Throw, or return `Status` | `SinkResult` of `VortexResult<_>` (immediate) |
| Batch error channel | none (errors are per row) | Deferred evidence, OR-reduced, one error after the loop |
| Granularity reported | The failing row, per row | The batch |
| Recoverable per row | Yes, `TRY` converts a row error to null | No |
| Nulls versus errors | Function never sees sure-null rows, no interaction | Dense can compute garbage rows, retry suppresses their errors |
| Infrastructure errors | Exceptions | `Err` at any layer, never suppressed or retried |

Velox's model is Presto's: an error belongs to a row, `TRY` must be able to catch it, and rows
after a failing row still evaluate. The adapter wraps loops in `applyToSelectedNoThrow` and records
exceptions in an error vector. This is expressive and matches SQL semantics, and it prices every
fallible function at a try/catch region plus per-row branching.

Vortex's model is shaped by two constraints Velox does not have: the loop must vectorize, and dense
execution can evaluate garbage. The result is the three-tier design (immediate `Err`, OR-reduced
`FailureEvidence` with a width cap, `RowExecution::DeferredError` resolved by a valid-rows retry).
It reports one error per batch and cannot express `TRY`. If Vortex ever needs row-granular error
recovery, it will need something like an error mask, and the evidence branch's warning applies: the
accumulator must stay out of sink storage or it becomes a loop-carried dependency.

## 8. Metadata: derived versus declared

Velox derives nearly everything from the C++ surface: [null behavior](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/core/SimpleFunctionMetadata.h#L983) from `callNullable`'s
existence, null-output capability from a `bool` return type, ASCII support from `callAscii`,
determinism and ASCII propagation from opt-in constants, priority from the shape of the signature.
The author states almost nothing twice. The failure mode is silence: after a signature typo, the
SFINAE probe does not find the method, and the error surfaces as a static assertion that at least
one `call` flavor must exist. Historically it was worse: a template `initialize` matched every
probe, and detection needed a dummy-type argument.

Vortex declares capabilities as constants (`FALLIBLE` on the function, `DENSE_SAFE` and
`DECODE_INFALLIBLE` on elements) and then cross-checks declarations against use with `const`
[assertions](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/visitor/check.rs) evaluated at monomorphization time: the visited tuple's arity must equal
`ARG_NAMES.len()`, fallible decode requires `FALLIBLE`, deferred visits require `FALLIBLE`,
evidence must fit in the output width, owned outputs must not need drop. Redundancy is the point:
`is_fallible` must be answerable without input dtypes (dictionary push-down evaluates values no row
references), so the function-wide constant exists, and the assertions keep it honest per dispatch.

## 9. Identity, registration, and persistence

A Velox function's identity is its name plus signature in a process-global registry, and that is
sufficient because plans are built and executed in the same ecosystem. Aliases are strings.
Function options do not exist as a concept: option-like behavior comes from constant arguments or
query config.

A Vortex scalar function has a `ScalarFnId`, an `Options` associated type, and a wire format
(`serialize`/`deserialize` on the function), because expressions embed in files that outlive any
process. This is why `RowFn` carries persistence methods that look out of place next to Velox: the
Vortex function object is a serialization boundary, not only a kernel. It is also why
`OutputSink::return_dtype` takes options: a decimal operator's output dtype depends on them.

One consequence surfaced in the stack: because every `RowFn` gets the vtable via a blanket impl, a
type cannot implement both traits, and existing public functions (`Binary`) keep their registered
identity by delegating to a private `RowFn` (`NumericBinary`) through the
[`execute_rows` and `row_fn_return_dtype`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/vtable.rs) free functions. Velox has no analogous friction: the simple function was
never the registered identity in the first place, the adapter-produced `VectorFunction` is.

## 10. Stated scope limits

Both projects wrote down what the row form is not for, and the lists agree almost item for item.

Velox ([`scalar-functions.rst`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/docs/develop/scalar-functions.rst)): functions that return an inner vector or buffer unchanged
(`map_keys`, `cardinality` historically), encoding tricks (`element_at` as a dictionary), lambda
functions, and anything needing a demonstrated benchmark win for the vector form.

Vortex ([#9128](https://github.com/vortex-data/vortex/issues/9128)): columnar and zero-copy kernels (`not`, `list_length`), kernels with cross-row state
(`like`), heterogeneous variadic kernels, and non-strict functions. The early experiment confirmed
the list the hard way: `byte_length`, `not`, `list_length`, and `list_sum` were all ported onto the
prototype and then reverted off it.
