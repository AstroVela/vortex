# RowFn and Velox simple functions

This directory compares the Vortex `RowFn` framework
([#9128](https://github.com/vortex-data/vortex/issues/9128)) with the simple scalar function
framework in [Velox](https://github.com/facebookincubator/velox). Both systems let an author define
a scalar function as an operation on one typed row. Both lift that definition into batch execution
over columnar data. Both make the same core bet: a monomorphized, fully inlined row closure inside
a framework-owned loop matches hand-written kernels. Almost every other choice differs, and those
differences are the subject of these documents.

## How to read this

Start with the table below. Each row links to the file that treats that dimension in depth:

- [`EXAMPLES.md`](./EXAMPLES.md) shows the same functions written in both systems.
- [`ABSTRACTION.md`](./ABSTRACTION.md) compares the designs one dimension at a time.
- [`LANGUAGE.md`](./LANGUAGE.md) separates Rust-versus-C++ effects from genuine design choices.

## The comparison in one table

Context for the table: a Velox simple function is a templated C++ struct with a `call` method,
lifted into a `VectorFunction` by
[`SimpleFunctionAdapter`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/expression/SimpleFunctionAdapter.h).
A Vortex `RowFn` is a Rust trait implementation whose `dispatch` method selects typed row code
through a sealed
[`RowVisitor`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/visitor/row_visitor.rs),
and a blanket impl turns it into a
[`ScalarFnVTable`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/unstable/row/vtable.rs).

| Dimension | Velox simple function | Vortex `RowFn` | Detail |
| --- | --- | --- | --- |
| Author writes | A struct with `call(result, args...)` | A trait impl: consts plus `dispatch` into a visitor | [EXAMPLES §1](./EXAMPLES.md#1-the-minimal-function-hypot) |
| Type binding | At registration, one call per signature. The registry resolves by name and types at plan time | At plan and execution time. `dispatch` matches runtime `DType`s and selects monomorphized code | [ABSTRACTION §2](./ABSTRACTION.md#2-where-types-bind), [LANGUAGE §2](./LANGUAGE.md#2-why-rowfn-has-a-visitor-and-velox-has-nothing-like-it) |
| Null semantics | Chosen by method: `call`, `callNullable`, or `callNullFree` | Strict only. The planner picks `Dense`, `DenseWithRetry`, or `ValidOnly` from element capabilities | [ABSTRACTION §3](./ABSTRACTION.md#3-null-handling) |
| Null rows | Not evaluated under default null behavior. Per-row `isSet` branch otherwise | `Dense` evaluates the garbage behind nulls branch-free, then masks the output | [ABSTRACTION §3](./ABSTRACTION.md#3-null-handling) |
| Nullable output | `bool call` marks a null result per row | Not expressible. Open question in [#9129](https://github.com/vortex-data/vortex/issues/9129) | [EXAMPLES §6](./EXAMPLES.md#6-nullable-outputs-the-open-question-answered-two-ways) |
| Constants | `initialize` once per query and thread. Plan constants only | `prepare` closure once per batch. Sees runtime `ConstantArray`s | [EXAMPLES §5](./EXAMPLES.md#5-hoisting-constant-argument-work), [ABSTRACTION §4](./ABSTRACTION.md#4-constants) |
| Encodings | Flattened above the function (peeling, `DecodedVector`) | Reach the function: `InputElement::decode`, plus the `reduce_encoded` hook | [ABSTRACTION §5](./ABSTRACTION.md#5-encodings-and-decoding) |
| Errors | Throw or return `Status` per row. `TRY` recovers per row | Three tiers: immediate, OR-reduced evidence, deferred with valid-row retry. Batch granularity | [EXAMPLES §3](./EXAMPLES.md#3-checked-arithmetic-where-the-error-channel-lives), [ABSTRACTION §7](./ABSTRACTION.md#7-errors) |
| Output | Writers into a prepared vector. In-place input reuse. Zero-copy strings | `OutputElement` (owned values) or `OutputSink` (uninitialized slots plus a write token) | [ABSTRACTION §6](./ABSTRACTION.md#6-outputs) |
| Feature surface | Strings, arrays, maps, rows, variadic, generics, ASCII fast path | Fixed arity (12 max). Primitives, bool, extension rows | [EXAMPLES §7](./EXAMPLES.md#7-beyond-the-current-rowfn-surface) |
| Escape hatch | A separate `VectorFunction` under the same name wins resolution | `reduce_encoded` inside the function, plus ad hoc columnar fallbacks | [EXAMPLES §8](./EXAMPLES.md#8-escape-hatches) |
| Metadata | Derived from which methods exist (SFINAE probes) | Declared consts, cross-checked by `const` assertions | [ABSTRACTION §8](./ABSTRACTION.md#8-metadata-derived-versus-declared), [LANGUAGE §1](./LANGUAGE.md#1-how-the-framework-learns-the-kernels-shape) |
| Safety | Raw pointers, unchecked by convention | `unsafe` contracts: one pre-loop check, write tokens, `compile_fail` tests | [LANGUAGE §4](./LANGUAGE.md#4-safety-what-unsafe-buys-and-what-ub-by-default-hides) |
| Shared concession | SIMD comparisons stay columnar ([`ComparisonSimdFunction`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/functions/prestosql/Comparisons.cpp#L148)) | x86 comparisons keep the [fused columnar path](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/fns/binary/compare/primitive.rs#L137), about 38% faster for `compare_u64_constant` | [EXAMPLES §8](./EXAMPLES.md#8-escape-hatches) |

Two rows deserve one extra sentence each. The null-rows row is the deepest contrast: the Vortex
choice to evaluate garbage keeps the loop branch-free, and it is what forces the deferred-error and
retry machinery that Velox does not need. The escape-hatch row shows both projects landing on the
same policy, stated in the
[Velox docs](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/docs/develop/scalar-functions.rst):
prefer the row form unless a benchmark demonstrates a significant gain for the columnar form.

## Candidate takeaways for #9128

These are observations, not decisions.

1. Velox's `bool call(...)` convention answers the nullable-output question in
   [#9129](https://github.com/vortex-data/vortex/issues/9129) with row-level granularity
   ([`array_min`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/functions/prestosql/ArrayFunctions.h#L131)).
   The bill: a `notNull` branch per row even on the fast path, and exclusion from result reuse and
   the flat-no-nulls fast path. A `RowFn` analogue also invalidates the current
   `validity() = union_child_validities` derivation.
2. Velox reuses a singly-referenced flat input of the right type as the output vector, in place.
   `RowFn` owned outputs have no analogue, and it is worth measuring.
3. Velox resolves a vector function ahead of a simple function per name and signature. That is a
   registry-level version of what
   [`compare_primitive_with_path`](https://github.com/vortex-data/vortex/blob/28d61448e/vortex-array/src/scalar_fn/fns/binary/compare/primitive.rs)
   does ad hoc. If more migrations keep columnar fallbacks, a shared selection mechanism can earn
   its place.
4. Velox derives metadata from which methods exist. Vortex declares constants and cross-checks
   them at dispatch time. The Vortex form is heavier and fails earlier. The Velox form is lighter
   and fails as "no method detected" at registration.
5. Velox string support rests on a per-batch input scan with a specialized loop (`callAscii`) and
   buffer-sharing outputs (`setNoCopy`,
   [`reuse_strings_from_arg`](https://github.com/facebookincubator/velox/blob/54fea71cc/velox/functions/prestosql/StringFunctions.h#L60)).
   Both map onto `OutputSink`, and neither needs nullable outputs.

## Sources

Vortex:

- Issues [#9128](https://github.com/vortex-data/vortex/issues/9128) (epic),
  [#9129](https://github.com/vortex-data/vortex/issues/9129) (API),
  [#9130](https://github.com/vortex-data/vortex/issues/9130) (execution).
- The PR stack: [#9353](https://github.com/vortex-data/vortex/pull/9353) (row execution),
  [#9450](https://github.com/vortex-data/vortex/pull/9450) (batch execution),
  [#9345](https://github.com/vortex-data/vortex/pull/9345) through
  [#9351](https://github.com/vortex-data/vortex/pull/9351) (migrations and bench tooling), and the
  merged [#9358](https://github.com/vortex-data/vortex/pull/9358) and
  [#9386](https://github.com/vortex-data/vortex/pull/9386).
- Code as of the stack top,
  [`ct/row-fn-benchmark-tools` at `28d61448e`](https://github.com/vortex-data/vortex/tree/28d61448e).
- History: the [#9255](https://github.com/vortex-data/vortex/pull/9255) umbrella, plus the
  research docs preserved on
  [`ct/row-fn-evidence`](https://github.com/vortex-data/vortex/tree/2beac64a4/research),
  [`ct/row-fn-history`](https://github.com/vortex-data/vortex/blob/34d36b11a/STRICT_SCALAR_FN_RESEARCH.md),
  and the umbrella's first parent
  [`5c302bce6`](https://github.com/vortex-data/vortex/tree/5c302bce6/research/rowfn-review-followup).

Velox: [`facebookincubator/velox` at `54fea71cc`](https://github.com/facebookincubator/velox/tree/54fea71cc),
mainly `velox/expression/SimpleFunctionAdapter.h`, `velox/core/SimpleFunctionMetadata.h`, and
`velox/docs/develop/scalar-functions.rst`.

A caveat on numbers: the Vortex figures are transcribed from the pinned methodology on the
evidence branches and from the PR bodies. The Velox docs assert near-parity with vector functions
but publish no equivalent methodology. Nothing here compares Vortex and Velox wall-clock numbers
directly.
