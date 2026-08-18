# Rust versus C++: which differences are the language's fault

The two frameworks look very different at the API surface. Some of that is design taste, but much
of it is forced by the host language. This document separates the two, because a fair comparison
must not credit or blame a design for what its language made inevitable.

## 1. How the framework learns the kernel's shape

**Velox: duck typing plus SFINAE.** The author writes any method named `call` whose signature is
compatible, and the framework probes for it:

```cpp
DECLARE_METHOD_RESOLVER(call_method_resolver, call);

static constexpr bool udf_has_call_return_bool = util::has_method<
    Fun, call_method_resolver, bool, exec_return_type,
    const exec_arg_type<TArgs>&...>::value;
```

(`velox/core/SimpleFunctionMetadata.h`.) There is no interface to implement. The same probing
detects `callNullable`, `callNullFree`, `callAscii`, and `initialize`, and derives the function's
null semantics from which probes succeed. The benefits are real: the author writes one method with
natural types, methods can themselves be templates or overload sets (`CeilFunction::call` is
templated over the input type, `date_trunc` has four `initialize` overloads matched by signature),
and no trait vocabulary exists to learn. The costs are also real: a signature typo means a probe
silently fails, and the framework needed a dummy-type probe only to distinguish a templated
`initialize` from a matching one, because a template matches every probe:

```cpp
// Detects if initialize() is a template method using SFINAE.
// Template methods can match any signature via template parameter deduction,
// causing false positives in trait detection. We probe with a dummy type ...
struct DummyProbeType {};
```

**Vortex: explicit traits.** `RowFn` is a trait with named items, the compiler checks every
implementation against the declaration, and there is nothing to probe. The equivalent of "which
call flavor did the author write" becomes "which visitor method did `dispatch` call", which is a
value-level choice checked at monomorphization time by `const` assertions rather than inferred from
a method's existence. Rust can imitate the Velox style only through macros, and the framework
deliberately has no macro front end: the umbrella-branch research rejected a generated-adapter
design and kept plain trait implementations.

Verdict: mostly language. C++ templates make signature-driven inference natural and interface
declarations optional. Rust makes interfaces cheap and inference of this kind nearly impossible.
Each framework leans into its language's grain.

## 2. Why `RowFn` has a visitor and Velox has nothing like it

The visitor is the piece of `RowFn` with no Velox counterpart, and it exists for a precise reason:
a Vortex function receives its types at runtime.

In Velox, by the time function code runs, types were fixed at registration. The C++ template *is*
the "for all element types" quantifier, and `registerFunction` discharges it per signature. Velox
therefore never needs to pass a polymorphic continuation anywhere.

In Vortex, `dispatch` inspects `&[DType]` and must then enter monomorphized code for types chosen
by that inspection. A function cannot return "some `ElementTuple` chosen at runtime" (the choices
have different types), so the framework inverts control: the caller passes a visitor that is
generic over the choice, and the function calls it at concrete types. This is rank-2 polymorphism,
and the visitor trait's generic methods are the only stable way Rust expresses it.

The design history shows the alternative that failed. An earlier iteration generated
generic-associated-type families with a `row_family!` macro and hit a language wall:

> Rust cannot abstract over a GAT's bound (`type Args<T: Self::Bound>` is rejected), so that
> approach needed a trait *and* an adapter per width class, hand-written or macro-stamped. The
> rank-2 visitor sidesteps the limit rather than writing around it.

(`STRICT_SCALAR_FN_RESEARCH.md` on `ct/row-fn-history`.)

Verdict: the visitor is a design consequence of one genuine design choice (execution-time type
binding, section 2 of `ABSTRACTION.md`) implemented in the only shape Rust offers. C++ avoids the
problem by construction, not by cleverness. A C++ system with runtime type binding grows the same
structure, as a virtual dispatch table over instantiations.

## 3. Arity

Velox handles arity with parameter packs. `doApply<POSITION>` peels one reader per recursion level
and builds the argument list at compile time, `Variadic<T>` folds a tail of arguments into a view,
and nothing anywhere states a maximum arity (Presto's `concat` accepts 252 arguments).

Rust has no variadic generics, so `ElementTuple` is implemented by a macro for tuples of arity 1
through 12, and `ARG_NAMES.len()` fixes exact arity per function. Variadic signatures are out of
scope for `RowFn`, and #9128 lists heterogeneous variadic kernels as a non-goal.

Verdict: pure language. This is the clearest case in the comparison where C++ is better equipped.

## 4. Safety: what `unsafe` buys and what UB-by-default hides

The hot loops in both frameworks compile to the same shape: unchecked indexed loads, a computation,
an unchecked indexed store. The difference is what stands between the source and that machine code.

The Velox adapter indexes raw pointers (`rawValues_[offset * indexMultiple_]`,
`data[row] = out`) and its correctness is a property of the surrounding code that nothing checks.
This is normal C++, and the adapter is careful, well-reviewed code. But the contract "every index
the loop produces is in bounds for every reader" exists only in the authors' heads.

Vortex has to earn each removed bounds check, and the framework turns that obligation into an
explicit chain, documented as such in the code:

1. `InputElement` is an `unsafe trait` whose contract states that every index below `ViewLen::len`
   is valid for `get_from_view_unchecked`.
2. The executor performs one pre-loop check (`view_lens_match(&views, row_count)`), placed
   deliberately beside the loop so LLVM sees the dominating equality.
3. `indexed_source` is an `unsafe fn` whose contract consumes that check.
4. The lane kernel's `get_unchecked` needs only `i < len`, guaranteed by the loop bound.
5. On the output side, `UninitElementSink` keeps `Vec::len` at zero until `finish`, a `const`
   assertion proves the element type needs no drop glue, and the `InitializedElement` token, whose
   only constructor is `unsafe`, is the per-row proof of initialization. A `compile_fail` doctest
   pins that safe code cannot forge it.

The practical differences this produces:

- The Vortex contracts are auditable at the trait boundary. A third-party `InputElement` (the
  stated extension point for crates like `vortex-tensor`) signs the same contract the framework's
  own elements sign, and review can check one implementation against one documented obligation.
  A third-party Velox function never touches the dangerous layer at all, because only the adapter
  indexes buffers. The exposure difference is real: Vortex externalizes an unsafe surface because
  external crates add element types, while Velox keeps all indexing internal because its type
  system is closed.
- The Rust framework must also be drop-safe at every intermediate point (a sink and its borrowed
  rows must tolerate abandonment after any callback prefix), a requirement C++ code with exceptions
  has too but rarely states. The `OutputSink` safety comment states it.
- The proof has a compile-time budget. The umbrella research measured an unchecked-access variant
  that removed 16 panic sites and recovered only 398 instructions, and rejected it: "The unsafe API
  was not justified." That cost-benefit sentence has no C++ analogue, because in C++ the checks it
  removed never existed.

Verdict: language, but with a design dividend. Rust forces the ceremony, and the ceremony produced
contracts (`DENSE_SAFE`, the write token, the skipped-rows initializer) precise enough that the
planner can make strategy decisions from them. Velox's equivalent knowledge is implicit in adapter
code paths.

## 5. Vectorization strategy

Both frameworks want the same loops out of the compiler, and neither uses intrinsics in the row
framework itself. The strategies differ.

**Velox specializes loops by hand and tolerates per-row branches.** The adapter builds five loop
variants (null-free with recursive check, null-free without, ASCII, all-not-null, general) and
selects one per batch, explodes flat-versus-constant reader combinations for up to three arguments,
and forces inlining through the whole chain (`FOLLY_ALWAYS_INLINE`, `INLINE_LAMBDA`). Inside the
loop it accepts branches: `isSet(row)` under `LIKELY`, a `notNull` check on every write, and the
try/catch region from `applyToSelectedNoThrow`. Where branch-free SIMD matters, Velox leaves the
simple framework entirely (the xsimd comparison functions).

**Vortex aims the whole design at branch-free autovectorizable loops** and treats the generated
code as part of the contract. The visible commitments in the stack:

- Row closures are `Fn`, not `FnMut`: a mutable capture is a loop-carried dependency, measured at
  8 to 11% (`ct/row-fn-history`).
- Failure evidence reduces in a loop-local accumulator with a `const`-asserted width cap, because a
  wider accumulator bounds the vector width and sink storage adds a memory dependency.
- Views exist so buffer descriptors are loop invariants. Error formatting is hoisted into `#[cold]`
  `#[inline(never)]` functions because formatting inside the branch takes the address of the loop
  bound and blocks vectorization.
- Length validations are placed *inside* the branch they protect, with a comment admitting "the
  exact pass interaction is unknown", because a shared helper produced 3.3x slower mixed-constant
  code under one LLVM version.

That last item names the tax Rust pays here that C++ mostly does not: sensitivity to CGU
partitioning, LTO configuration, and LLVM version. The evidence branches document four cases. A
five-line refactor made mixed-constant loops 4.6 to 8.5x slower under LLVM 21, and an
output-iterator fix recovered them under LLVM 22. A `T: Copy` bound, unused by codegen, cost 60% on
one benchmark. A loop that crossed a cache-line boundary cost 26% with identical instruction mixes.
Velox's
header-only, always-inline, single-TU-per-signature adapter model gives the C++ compiler one big
function per signature and makes placement comparatively stable. The Vortex answer is procedural
rather than structural: a pinned two-worktree benchmark harness (`scripts/benchmark-rowfn.sh`),
IR and assembly inspection as review evidence, and columnar fallbacks kept where the generated code
still loses.

One more contrast worth naming: Velox's per-row output for booleans is a bit write
(`bits::setBit`), while Vortex's `bool` output element materializes a byte per row and bit-packs in
`build`. Neither row framework can express the fused compare-to-movemask loop, which is exactly why
both keep columnar comparison kernels (`ComparisonSimdFunction` there, the fused
compare-and-bitpack path here).

Verdict: mixed. The branch-free-dense ambition is a Vortex design choice Velox did not make. The
toolchain fragility that ambition exposes is substantially a rustc/LLVM artifact, and the
mitigation (measure generated code, keep fallbacks) is process, not language.

## 6. Errors: exceptions versus values

Velox leans on C++ exceptions: kernels throw, the adapter catches per row, `TRY` consumes recorded
errors, and a throwing `initialize` is replayed lazily so that an all-null input still folds to
null instead of raising. This gives row-granular error semantics with zero API surface on the
kernel, priced at try/catch regions around every loop and full unwinding on the failing row. The
newer `Status` return is a value-based refinement of the same per-row channel.

Rust has no exceptions to lean on, and a `Result` branch per row is exactly the branch the
vectorization strategy forbids. The three-tier error design (`SinkResult` for immediate,
`FailureEvidence` for deferred, `RowExecution` for the encoded path) is therefore as much a
language consequence as a design one: it is what per-row fallibility looks like when both unwinding
and per-row branching are off the table. The panic path still exists underneath (Rust bounds checks
and assertions can panic), and the framework treats "callback must not panic" as a documented
requirement plus drop-safety obligations rather than pretending panics cannot happen.

Verdict: mixed, mostly language. The granularity difference (per row versus per batch) is the
design consequence to keep in view.

## 7. Where hoisted state lives

Velox `initialize` caches into mutable member fields (`unit_`, `timeZone_`), made safe by one
function instance per compiled expression and thread. Idiomatic, and invisible in signatures: any
method can read any field, and nothing marks which state is per-query versus per-batch.

Vortex `prepare` returns a value whose type the author chooses (`ConstNorms<T>`,
`PreparedOperand`), which the executor threads into the row closure by shared reference. The row
closure cannot mutate it (it is `Fn`), so per-batch state is immutable by construction, and lazy
mutation needs an explicit interior-mutability cell (`OnceCell` in `SpatialContains`). `prepare` is
also infallible by design, a rule the history states crisply: it refines values the row loop can
compute itself, and fallibility is read off the types before dispatch, so a failing prepare has
nowhere to be declared.

Verdict: language-shaped design. Rust's ownership rules push state into visible, typed channels.
The same discipline is available in C++ and Velox did not need it, because its per-instance model
already isolates the state.

## 8. Compile-time and code-size economics

Both frameworks multiply code aggressively, and both installed guards.

Velox instantiates one adapter per registered signature, times up to 2^3 encoding combinations,
times five loop variants. The guard is explicit: `specializeForAllEncodings = num_args <= 3`.
Registration cost also repeats per translation unit that registers functions.

Vortex monomorphizes per ptype, times operator, times three visitors (planner, dense, valid-only),
times constant arrangements. The guards are different in kind: sealed arities cap tuple expansion,
the module split was tuned against measured symbol counts (the framework refactor cut out-of-line
`NumericBinary` functions from 147 to 94), and unreachable monomorphs were pruned when evidence
showed them costing layout (removing eight unreachable fallback monomorphs shifted an unrelated
benchmark 11%).

Verdict: shared problem, same order of magnitude, different instruments. The one asymmetry is
again the CGU lottery: C++'s per-TU model makes "who inlines with whom" predictable, while Rust's
CGU partitioning made it a measured variable in this work.

## 9. Summary table

| Difference | Language or design? |
| --- | --- |
| `call` detected by probe versus trait implementation | Language |
| Visitor / rank-2 dispatch | Design (runtime type binding), shaped by language |
| Registration-per-signature versus `dispatch` on dtypes | Design |
| Arity 12 cap and no variadics | Language |
| `unsafe` contracts, write tokens, `const` assertions | Language, with design dividends |
| Never-evaluate-nulls versus dense-then-mask | Design |
| Deferred evidence versus per-row throw plus `TRY` | Mixed: granularity is design, mechanism is language |
| Member-field state versus prepared values | Language-shaped design |
| CGU/LTO/LLVM-version sensitivity | Toolchain |
| Columnar escape hatches for comparisons | Neither: the row abstraction's shared limit |
