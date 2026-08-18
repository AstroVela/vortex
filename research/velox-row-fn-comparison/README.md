# RowFn and Velox simple functions

This directory compares the Vortex `RowFn` framework (#9128) with Velox's simple scalar function
framework. Both systems let an author define a scalar function as an operation on one typed row.
Both then lift that definition into batch execution over columnar data. They make different choices
at almost every layer, and those differences are the subject of these documents.

Sources for the Vortex side: issues #9128, #9129, and #9130, the PR stack (#9353, #9450, #9345
through #9351, and the merged prerequisites #9358 and #9386), the code on `ct/row-fn-benchmark-tools`
(the top of the stack), and the historical experimental branches `ct/row-fn`, `ct/row-fn-history`,
and `ct/row-fn-evidence`. Sources for the Velox side: `facebookincubator/velox` at commit
`54fea71cc` (2026-08-17), mainly `velox/expression/SimpleFunctionAdapter.h`,
`velox/core/SimpleFunctionMetadata.h`, `velox/functions/Macros.h`, and
`velox/docs/develop/scalar-functions.rst`.

## How to read this

- [`EXAMPLES.md`](./EXAMPLES.md) shows the same functions written in both systems. Read this first
  to get a feel for both APIs.
- [`ABSTRACTION.md`](./ABSTRACTION.md) compares the two designs one dimension at a time: type
  dispatch, null handling, constants, encodings, outputs, errors, and escape hatches.
- [`LANGUAGE.md`](./LANGUAGE.md) separates the differences that come from Rust versus C++ from the
  differences that are genuine design choices.

## The two systems in one paragraph each

A Velox **simple function** is a templated C++ struct with a `call` method that computes one row:
`void call(T& result, const T& a, const T& b)`. Registration binds the template to concrete types,
one `registerFunction<Fn, Ret, Args...>` call per signature. A template adapter
(`SimpleFunctionAdapter`) detects which methods the struct defines through SFINAE, derives null
semantics and metadata from those methods, and compiles the whole package into a `VectorFunction`.
The adapter owns decoding, per-batch null and ASCII decisions, constant inputs, result allocation,
result reuse, and error capture. The engine resolves functions by name and argument types at plan
time, and a hand-written `VectorFunction` under the same name wins resolution when one exists.

A Vortex **`RowFn`** is a Rust type that implements one trait. The author declares argument names
and fallibility as constants, then writes a `dispatch` method that inspects the runtime input
`DType`s and calls back into a sealed `RowVisitor` with concrete element types and a row closure:
`visitor.visit::<(f64, f64), f64>(|(x, y)| x.hypot(y))`. A blanket implementation turns every
`RowFn` into a `ScalarFnVTable`. The batch executor owns validation, decoding through
`InputElement`, constant collapsing, a planner-selected null policy (`Dense`, `DenseWithRetry`, or
`ValidOnly`), output construction through `OutputElement` or `OutputSink`, deferred error
reduction, output validation, and strict validity. An encoding-aware `reduce_encoded` hook can
bypass the row loop from inside the same function.

## Summary of findings

**The core bet is the same.** Both frameworks bet that a monomorphized, fully inlined row closure
inside a framework-owned loop matches hand-written kernels, so the batch machinery can be written
once. Both back the bet the same way: no virtual calls in the loop, batch-level decisions hoisted
out of the loop, and specialization per type and per operand arrangement.

**The biggest abstraction difference is where types bind.** Velox binds types at registration time.
One C++ template becomes N independent vector functions, and the plan-time resolver picks one by
name and argument types. Vortex binds types at plan and execution time. One registered function
object receives runtime `DType`s and selects monomorphized code through the `dispatch`/`RowVisitor`
rank-2 callback. Velox moves the dispatch problem into the registry. Vortex moves it into the
function. This one choice explains most of the API-shape differences, including why `RowFn` needs a
visitor at all. See `ABSTRACTION.md` section 2 and `LANGUAGE.md` section 2.

**Null handling is method choice in Velox and capability planning in Vortex.** A Velox author picks
null semantics by choosing which method to write: `call` (null in, null out), `callNullable` (sees
nulls), or `callNullFree` (never sees nulls, even inside nested values). The engine deselects
sure-null rows above the function and the adapter re-checks per row. A Vortex `RowFn` is strict by
contract, full stop. The planner derives one of three execution policies from declared element
capabilities (`DENSE_SAFE`, `DECODE_INFALLIBLE`, sink fallibility), and `ValidOnly` further splits
into skip-invalid and filter-and-scatter at runtime. The deepest contrast: under default null
behavior, Velox never evaluates a null row (only a `callNullable` function sees one, by request).
Vortex's `Dense` policy deliberately evaluates the garbage behind nulls to keep the loop
branch-free, then masks the output. That choice is what forces the deferred-error and retry
machinery that Velox does not need. See `ABSTRACTION.md` section 3.

**Vortex can express things Velox cannot.** Deferred failure evidence that OR-reduces inside a
vectorized loop and retries only valid rows. Skip-invalid execution over placeholder-initialized
sinks. Encoding-aware shortcuts (`reduce_encoded`) inside the same function rather than a separate
registration. Per-batch constant preparation with typed `Option<elem>` values. Function options
that serialize into files.

**Velox can express things Vortex cannot, yet.** Row-level nullable outputs (`bool call(...)`), the
exact capability that #9129 leaves as an open question. Strings with zero-copy results and an
ASCII fast path. Arrays, maps, and rows through lazy views and writers. Variadic signatures.
Generic (type-parametric) signatures with orderability constraints. In-place result reuse of an
input vector. Per-row errors recoverable by `TRY`.

**Both frameworks concede the same limitation.** A row-at-a-time API cannot express kernels whose
output shape is not one value per row. Velox keeps hand-written vector functions for `is_null`
(bulk bit flip), subscript (dictionary wrap), and notably a SIMD `ComparisonSimdFunction` that
outranks its own simple comparison functions at resolution time. Vortex keeps the fused
compare-and-bitpack columnar path for measured x86 cases (about 38% faster for
`compare_u64_constant`), keeps decimal arithmetic columnar, and once had to opt `reset_offsets` out
of `RowFn` entirely. The Velox docs state the policy both projects converged on: prefer the row
form unless a benchmark demonstrates a significant gain for the vector form.

**The safety story is the real Rust-versus-C++ divide.** The Velox adapter reads through raw
pointers and trusts its own indexing. The Vortex executor gets the same machine code only by
building an explicit proof chain: one pre-loop length check, `unsafe` traits whose contracts state
what the check proved, a zero-sized write token that safe code cannot forge (pinned by a
`compile_fail` doctest), and `const` assertions on drop glue and failure-evidence width. The cost
is API weight. The return is that the contracts are machine-checked at the boundary and the row
closure itself stays safe. See `LANGUAGE.md` sections 4 and 5.

## Candidate takeaways for #9128

These are observations, not decisions.

1. Velox's `bool call(...)` return convention answers the nullable-output open question in #9129
   with row-level granularity, and its `can_produce_null_output` metadata shows the derived
   consequences (no result-vector reuse, no flat-no-nulls fast path). A `RowFn` analogue needs an
   output form that builds validity, which invalidates the current
   `validity() = union_child_validities` derivation. Velox pays for the capability with a `notNull`
   branch per row, even in its fast path.
2. Velox's result-vector reuse (compute in place over a singly-referenced flat input of the same
   type) has no Vortex analogue and is worth measuring for `RowFn` owned outputs.
3. The Velox resolution rule (vector function outranks simple function per name and signature) is a
   registry-level version of what `compare_primitive_with_path` does ad hoc. If more `RowFn`
   migrations keep columnar fallbacks, a shared selection mechanism can earn its place.
4. Velox derives all function metadata from which methods exist. Vortex declares `FALLIBLE` and
   element capabilities as constants and then cross-checks them with `const` assertions at dispatch
   time. The Vortex form is more verbose and more checkable. The Velox form is lighter and fails
   later (a signature mismatch surfaces as "no method detected" at registration).
5. Velox's string support rests on two ideas `RowFn` does not have: a per-batch input property scan
   with a specialized loop (`callAscii`), and buffer-sharing outputs (`setNoCopy` plus
   `reuse_strings_from_arg`). Both map naturally onto `OutputSink`, and neither requires nullable
   outputs.

## A note on performance claims

The numbers quoted in these documents come from two different worlds. Vortex numbers come from the
pinned two-worktree methodology in `research/rowfn-*` on the evidence branches and from the PR
bodies (#9345, #9346), measured on specific rustc/LLVM versions with stated CGU and LTO
configurations. Velox has no equivalent published methodology for its adapter overhead. Its docs
assert near-parity with vector functions and back specific cases with in-tree benchmarks
(`CardinalityBenchmark.cpp`, `SimpleSubscriptBenchmark.cpp`). Treat cross-system statements here as
qualitative. Nothing in these documents compares Vortex and Velox wall-clock numbers directly.
