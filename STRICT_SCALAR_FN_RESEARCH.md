# A layered authoring API for strict scalar functions

**Status: proposal, feature-complete on this branch.** Most scalar functions in Vortex are
[strict](https://github.com/vortex-data/vortex/pull/8930): a null input row forces a null output row,
and non-null outputs depend only on non-null inputs. The null propagation, constant folding, validity
bookkeeping, and nullability derivation that follow are identical in every implementation, and right
now each function re-derives them by hand. This branch lifts them out into two authoring traits and
an open element vocabulary, and ports eleven functions onto them as proof: `byte_length`,
`list_length`, `not`, `list_sum`, the four `vortex-tensor` functions, and the three `vortex-geo`
functions.

Every pre-existing test passes unchanged, plus new regression tests for the invariants the framework
now enforces. Along the way it turned up three problems that are not about the framework, filed as
#9091, #9090 and #9092 and discussed in
[Problems to extract](#problems-to-extract-onto-develop).

---

## The design in one screen

```text
RowFn ──────────blanket──▶ StrictScalarFnVTable ──────blanket──▶ ScalarFnVTable
(row at a time, types                (null / constant / validity          (full control)
 chosen per batch)                    lifting for a columnar kernel)
```

Two authoring traits, one for each axis a strict function actually varies on, plus a third axis (*how
a row is typed*) factored into an open element vocabulary that neither trait mentions.

### `StrictScalarFnVTable`, the null/validity lifting

Write the structural metadata plus one **columnar** kernel that ignores validity. A blanket impl
derives:

- `is_strict = true`, and a mirrored `validity` a kernel can answer with the conjunction of its child
  validities when it never turns a wholly non-null row into a null (see
  [Strictness is not totality](#strictness-is-not-totality)), so the planner knows which rows are null
  without executing the function.
- `return_dtype` = `return_element_dtype` widened to nullable iff any input is nullable, so the
  strictness dtype contract holds by construction rather than per function.
- `execute` = the shared cases before the kernel runs: a null-constant input short-circuits to an
  all-null constant, all-constant inputs evaluate one row and broadcast, and partially-null inputs
  are handled per `NullHandling` (`Dense` masks after a full pass, `Filter` filters then scatters).
- Options serde, from `PersistableOptions` on the options type.

This is the layer for a function whose kernel is columnar rather than row-at-a-time: `not` (one `!`
per 64-bit word), `list_length` (a difference of offset buffers), `list_sum` (a grouped accumulator over
the elements child), `l2_denorm` (broadcast over flat tensor storage). See
[Why three concepts and not fewer](#why-three-concepts-and-not-fewer) for why it cannot be folded
away.

### `RowFn`, one row with element types chosen per batch

Name a witness argument tuple and return type, then in `dispatch` pick the concrete element types for
a batch and hand the framework a row closure through a rank-2 visitor. A blanket impl derives the
whole `StrictScalarFnVTable` (and array serde) from it. When the element types are fixed, `dispatch`
is a single `visit` at those types. When one ID spans several widths (`l2_norm` accepts f16/f32/f64),
`dispatch` matches on the input dtypes and visits at the chosen width.

Everything structural follows from the argument tuple and return type: arity, per-argument dtype
validation, the output dtype, null handling, and fallibility. There is nothing for an implementor to
declare twice or get wrong, because the framework reads it off the types (see
[Properties, not conventions](#properties-not-conventions)). A constant operand is decoded once and
read at stride 0, so a broadcast argument costs one decode rather than one per row.

Note that `RowFn` does not *require* totality, it just cannot currently express its absence: every
`OutputElement` builds an all-valid column, so a row kernel has no way to say "this row is null". One
`impl OutputElement for Option<T>` would lift that. No function needs it yet, so it is not there.

### The element vocabulary, how a row is typed

`InputElement` and `OutputElement` are open traits. A `NativePType`, `bool`, `Bytes` (a resolved
`&[u8]`), and `BytesLen` (a length read from a view without resolving it) ship in the framework, and
`vortex-tensor` adds `TensorRow<T>`, reaching through the extension wrapper into flat storage, in one
impl in its own crate. Adding `&str`, decimals, or a list row is one impl that every row function
gains, with no framework change.

---

## Why three concepts and not fewer

The standard applied here: every trait, and every member of every trait, has to have a purpose
nothing else can provide. Testing each against that standard is what the bulk of this research was.

### `RowFn` and the witnesses are forced, not chosen

A scalar function's *signature*, meaning its arity and fallibility, is a property of
`(function, options)` with **no input dtypes**: `ScalarFnVTable::arity(&self, options)` and
`is_fallible(&self, options)`, and `ScalarFnSignature` above them, take none. So any framework that
derives arity and fallibility from element types has to be able to name element types *without seeing
dtypes*, which is exactly what `ArgsWitness` / `RetWitness` are. Because `dispatch` *does* see dtypes
and could choose otherwise, some check has to tie the two together, which is the compile-time witness
check below. This cost is not a consequence of the rank-2 encoding: **any** design that derives a
dtype-free signature from per-batch types pays it.

A previous iteration made the width choice a generic-associated-type family generated by a
`row_family!` macro. Rust cannot abstract over a GAT's bound (`type Args<T: Self::Bound>` is
rejected), so that approach needed a trait *and* an adapter per width class, hand-written or
macro-stamped. The rank-2 visitor sidesteps the limit rather than writing around it: the kernel owns
the width `match`, where `T: Float` appears literally inside a `match_each_*_ptype!` arm, and the
framework method `RowVisitor::visit<A: ElementTuple, R: ApplyResult>` is generic only over bounds it
owns. The macro, its family traits, and its generated adapters are all deleted. Note that `dispatch`
is not even per-*width*: it can pick different element *kinds* per dtype, which no
bound-parameterized family could.

### `ElementwiseFn` was not forced, so it is gone

An earlier revision had a third trait, `ElementwiseFn`, for the fixed-element-type case: name `Args`
and `Ret`, write `apply`. It read cleanly, but it failed the standard. `RowFn` already covers the
fixed case (the dispatch is a single constant `visit`), so `ElementwiseFn` bought roughly seven lines
on exactly one production function (`byte_length`) at the cost of 114 framework lines and a third
link in the blanket-impl chain. The probes settled it: of the functions examined, `not` and `list_sum`
turned out not to be row functions at all, and `list_length` needed the encoding-aware
`reduce_encoded` hook that `ElementwiseFn` never exposed. So the constituency I expected it to have
never materialized, and it is deleted. `byte_length` writes a two-line `dispatch` instead.

The one-trait-with-defaults alternative (a single `RowFn` with `dispatch` defaulted to visit the
witnesses and `apply` defaulted to `unimplemented!()`) was rejected because it converts a compile
error into a runtime panic: a type implementing neither method compiles, registers, and answers
signature queries with a plausible shape, then panics on first execution. `dispatch` is therefore
required.

### `StrictScalarFnVTable` cannot be folded into `RowFn`

`RowFn`'s type surface is *closed*. The output dtype is `OutputElement::element_dtype()`, drawn from
the finite set of `OutputElement` impls, `ElementTuple` exists only for arities 1 to 3, and the loop
is one `apply` per row. Three whole classes of strict function are therefore inexpressible as a
`RowFn` at any cost:

- **Output dtype outside the element set.** `ext_storage`'s output is an extension array's storage
  dtype, so `vortex.geo.box` is a struct and `vortex.uuid` is a `FixedSizeList(u8,16)`. `vortex-geo`'s
  zone-map pruning calls `ext_storage` on a `geo.box` statistic, and a row-function port breaks it at
  plan time.
- **Variadic arity.** `merge` and `select` take an unbounded number of children, while `RowFn` fixes
  `Arity::Exact(n <= 3)`.
- **Sub-row-granular kernels.** `not` negates one 64-bit word at a time, so a row loop over `bool` is
  ~64x the memory traffic and, measured, 406x slower at a 64Ki batch (see
  [Measurements](#measurements)).

So the middle layer has a genuine, disjoint constituency: `l2_denorm`, `not`, `list_length`,
`list_sum`, and prospectively `select`, `merge`, `json_to_variant`. "Just a visitor" collapses three
concepts to two rather than to one.

### Every remaining member earns its place

A member-by-member audit, with call sites found by grep rather than by guess, turned up nothing
deletable. The non-obvious cases are worth recording:

- **`RowVisitor::Out`** is what lets one `dispatch` `match` serve both plan time (`Out = DType`,
  validate and name the output dtype) and run time (`Out = ArrayRef`, decode and run the loop). The
  alternatives, a `{DType, ArrayRef}` enum unwrapped at each site or two separate dispatch hooks,
  either add unwrap-panics or duplicate the width `match` in every width-polymorphic function with no
  compiler check that the two copies agree.
- **A plan-time visit is unavoidable.** `l2_norm` declares `RetWitness = f64` but dispatches over
  f16/f32/f64, so the output dtype read off the witness would be wrong for two of three widths. Also
  `TensorRow<T>::validate` rejects an `f32` column against an `f64` witness, and the visit is what
  gives cross-argument uniformity for free (`int_max(i16_col, i64_col)` is rejected by
  `(T, T)::validate`, not by any `dispatch` body, which only inspects `args[0]`).
- **`ApplyResult` distinct from `OutputElement`** is what lets one trait serve both infallible
  (`Ret = f64`) and fallible (`Ret = VortexResult<f64>`) kernels without a wrapper. `f64` cannot be
  simultaneously fallible and infallible, so the fallibility bit lives on the return *shape* rather
  than on the element.

---

## Properties, not conventions

The framework's real value beyond line count is that two invariants an implementor used to have to
get right are now derived from the types, so an unsound combination cannot be written.

### Null handling follows from the arguments and the return type

`NullHandling::Dense` runs the kernel over every row including those behind nulls, then masks. It is
cheaper than filtering and the only option that leaves inputs at their original encoding, so it is
right whenever it is sound. Soundness needs two things, every argument readable behind a null row and
an infallible computation, and both are already visible in the types:

```rust
const fn row_null_handling<A: ElementTuple, R: ApplyResult>() -> NullHandling {
    if A::DENSE_SAFE && !row_is_fallible::<A, R>() { NullHandling::Dense } else { NullHandling::Filter }
}
```

Whether a dense read is safe is a property of the *element*, not of the function: reading a whole
value out of a flat buffer is safe (`NativePType`, `bool`, `TensorRow`, `BytesLen`), while following a
stored offset into a data buffer is not (`Bytes`), because arrays only validate the views of their
*valid* rows. This caught a real bug in this branch's own `byte_length`, see
[Problems to extract](#problems-to-extract-onto-develop).

### Fallibility comes from the return type *and* the element decode

A function is fallible if its computation can fail (`Ret = VortexResult<T>`) **or** if decoding an
argument can fail on legal data. The second source is real and was missing: `geo_distance`'s row
computation cannot fail, but parsing WKB bytes into a geometry can, for a *valid* row holding
malformed bytes. So `InputElement` carries `DECODE_FALLIBLE`, and fallibility is the disjunction:

```rust
const fn row_is_fallible<A: ElementTuple, R: ApplyResult>() -> bool { A::DECODE_FALLIBLE || R::FALLIBLE }
```

`is_fallible` gates dict-value pushdown (`arrays/dict/compute/rules.rs`), which speculatively
evaluates a function over *unreferenced* dictionary values, so a function that under-reports
fallibility fails a query on rows it never needed.

### The witness is checked at compile time

Arity, dense-safety and fallibility must not vary between the choices `dispatch` makes, because the
framework acts on them before dispatching. Since (with `ElementwiseFn` gone) *every* function names
its element tuple twice, once as `ArgsWitness` and once in the `visit`, the check that the two agree
is load-bearing, and it is a compile-time `const` assert inside each visit:

```rust
const fn assert_witness_agrees<F: RowFn, A: ElementTuple, R: ApplyResult>() {
    assert!(A::ARITY == <F::ArgsWitness as ElementTuple>::ARITY, "…");
    assert!(A::DENSE_SAFE == <F::ArgsWitness as ElementTuple>::DENSE_SAFE, "…");
    assert!(row_is_fallible::<A, R>() == row_is_fallible::<F::ArgsWitness, F::RetWitness>(), "…");
}
```

Monomorphizing any dispatch arm evaluates it, so even a `match` arm that never runs at a given width
is checked, and a disagreement fails the build pointing at the exact `visit::<…>` call. It compares
the raw arity/dense-safety/fallibility rather than the derived `NullHandling`, which collapses
dense-safety and fallibility together and would miss an arm that flipped both. A `compile_fail`
doctest pins that a lying witness does not compile. This replaced a runtime check that ran three
times per array (plan, execute, deserialize).

---

## Strictness is not totality

This is the finding that decides what the middle layer may derive. Note that
[#9033](https://github.com/vortex-data/vortex/pull/9033) reached the same conclusion independently and
has since landed, so this section is no longer the argument for the finding, only for the API that
follows from it.

Before #9033, the `is_strict` documentation stated the validity-equivariance law,
`f(…, mask(aⱼ, m), …) == mask(f(…, aⱼ, …), m)`, and then asserted as "consequence 1" that output
validity is the conjunction of input validities. **Consequence 1 does not follow from the law.** It
needs an extra premise: that the kernel never turns a wholly non-null row into a null. #9033 replaced
that equality with a one-sided bound, `valid(f(a₁, …, aₖ)) ⊆ valid(a₁) ∧ … ∧ valid(aₖ)`, which is the
vocabulary this branch uses. `docs/strictness-and-validity-pushdown.typ` proves the law and the
null-propagation reading are the same property, and separates what does not follow from either.

`list_sum` is the counterexample. Summing a valid *empty* list yields null. It still satisfies the law
(a null it introduces at a valid row appears identically on both sides of the equation and cancels),
so it is genuinely strict, but its output validity is *narrower* than its input validity.

Two properties, then, not one:

| property | what needs it |
| --- | --- |
| **strict** (null propagation, equivalently validity equivariance) | every validity push-down, the thing we actually want |
| **total** (non-null in implies non-null out) | upgrading the `⊆` bound to `=`, so validity is precomputable |

The old blanket impl derived `validity = union_child_validities` for *every* implementor, which needs
totality while the trait only requires strictness. Every current implementor happens to be total, so
nothing was broken, but a partial function joining the layer would get a `validity` that contradicts
what it computes: `arr.validity()` would report all-valid while `arr.execute()` yields the null, since
`ValidityVTable<ScalarFn>::validity` evaluates the derived expression. `list_sum` was about to be
exactly that, and is now ported onto the layer as the first non-total member.

#9033 says a function satisfying the stronger equality "can advertise that through
`ScalarFnVTable::validity`". That is the same idea as `is_total`, moved from a hand-written method to a
boolean, because a blanket impl cannot hand-write `validity` per function: it needs the property as
data in order to decide whether to derive one.

The fix needs no new property. `validity` is mirrored on `StrictScalarFnVTable` alongside `reduce`,
defaulting to `None`, and a kernel that satisfies the equality answers it with
`union_child_validities`. The unsound direction is the one that now takes work, and the safe default is
what a function gets for free.

An earlier revision of this branch instead added an `is_total` method and derived `validity` from it.
That was strictly worse: it introduced a concept the codebase did not have, in order to compute
something a function can just say directly. It is gone. The `RowFn` blanket impl answers `validity`
for every row function, justified by its own output vocabulary (no `OutputElement` is nullable, so no
row kernel can introduce a null), which keeps the row layer at zero boilerplate.

Note that strictness rather than totality gates membership either way: `is_null` is total but
disqualified, because it inspects validity and so does not propagate nulls. That is also why the trait
is not called `TotalFnVTable`.

> **A related latent issue, deliberately not fixed here.** Four functions declare `is_strict = true`
> and are strict-but-not-total: `get_item` (a nullable field under a non-null struct), `mask`,
> `variant_get`, `geo_envelope`. None is broken today, since `get_item` leaves `validity` at the
> default and `mask` overrides it correctly, but any that grows a conjunction-shaped `validity`
> derivation would be wrong. This predates the branch and belongs in its own investigation.

---

## Problems to extract onto develop

The framework surfaced three problems that are not really about the framework. Each is filed
separately and I think each should land as its own PR rather than riding in on this one. Note that
none of them is a live miscompute on `develop` today, which is worth saying plainly, because the
branch's own commit messages describe fixes to *this branch's* code.

1. **Strict-but-non-total validity derivation ([#9091]).** The `is_strict` documentation presents
   totality as a consequence of strictness when it is an independent premise (see above). Nothing
   derives validity from `is_strict` automatically, so nothing is wrong today, but the doc invites the
   next strict-but-partial function to write `validity: union_child_validities` and be silently wrong.
   **Superseded by [#9033], which lands the documentation correction on `develop`.** This branch needs
   nothing beyond that, since it now mirrors `validity` rather than deriving it from a property.

2. **Views behind null rows are unvalidated ([#9090]).** `VarBinViewArray::validate_views` only
   validates the views of *valid* rows, so a legal array can hold a view behind a null row naming a
   buffer that does not exist, and resolving it densely panics (`index out of bounds: the len is 1 but
   the index is 9`). On this branch, expressing byte length as "a function of the row's bytes" quietly
   changed *what gets decoded* and hit that panic. The fix here reads the length out of the view
   (`BytesLen`) and never resolves the row, and
   `test_byte_length_ignores_unresolvable_views_behind_nulls` pins it (verified to panic without the
   fix). `develop`'s `byte_length` was already immune, since it also read `view.len()`, so the
   extraction is that regression test rather than a code change. The doc half is also covered by
   [#9033], which deletes the dense-evaluation "consequence 2" outright rather than narrowing it. That
   leaves `InputElement::DENSE_SAFE` as the only place the licence is written down, per element rather
   than as a blanket claim, which is where it belongs.

3. **Bit-at-a-time bool packing ([#9092]).** `OutputElement for bool` used `BitBuffer::from_iter`,
   where the `Vec<bool>` is already owned and contiguous so `BitBuffer::from` routes to the
   multiversioned SIMD packer. Measured **6.6 to 7.9x faster** on the packing step, for every
   bool-returning row function. Note that `OutputElement` only exists on this branch, so the
   develop-side instance of the same pattern is a different call site:
   `encodings/sequence/src/compute/compare.rs` builds an n-bit result with a per-row predicate when it
   already knows the single set index. I have not benchmarked that site.

[#9033]: https://github.com/vortex-data/vortex/pull/9033
[#9090]: https://github.com/vortex-data/vortex/issues/9090
[#9091]: https://github.com/vortex-data/vortex/issues/9091
[#9092]: https://github.com/vortex-data/vortex/issues/9092

---

## Audit: can the four `StrictScalarFnVTable` impls really not be `RowFn`?

There are exactly four in production. Auditing each against the two questions that matter, rather than
repeating the earlier verdicts, **not one of them is structurally impossible**. Every "cannot" in this
document was really "cannot with the trait signed as it is today", and the required changes are already
named elsewhere in these notes. Recording the distinction because it is the difference between a limit
and a decision.

| function | signature expressible? | kernel row-shaped? | what it would take |
| --- | --- | --- | --- |
| `not` | **yes**, `(bool,) -> bool`, both elements exist | **no** | nothing. It can be a `RowFn` today and should not be: `!bits` is one `!` per 64-bit word, in place when unshared, against 16k closure calls and a `Vec<bool>` repack |
| `list_length` | output is a fixed `U64`; input needs a `ListLen` element | **no** | one new element. Still should not: the answer is a child array or one constant |
| `list_sum` | no, twice over: dtype varies with the element type, and a valid empty list sums to null | yes | `element_dtype(args)` plus `impl OutputElement for Option<T>` |
| `l2_denorm` | no: output dtype is `arg_dtypes[0]` | yes, per-row scaling | `element_dtype(args)` plus a write-into-buffer output element |

Two readings follow.

**The honest framing is "can, and here is whether it is worth it."** For `not` and `list_length` the
answer is a flat no on performance grounds, and those are settled. For `list_sum` and `l2_denorm` the
answer is yes-with-changes, and the changes are shared: both want `element_dtype` to see the input
dtypes, which is one signature widening that `validate_row_args` is already positioned to pass through.

**`l2_denorm` is the one worth doing**, because its kernel genuinely is per-row scaling and because it
still carries eight `unsafe` blocks that the port would remove, exactly as it did for the other three
tensor kernels. The sequence:

1. Widen `OutputElement::element_dtype()` to `element_dtype(args: &[DType]) -> VortexResult<DType>`,
   updating the three existing impls to ignore the argument. Mechanical, but on its own it enables
   nothing, so it should land together with step 2 rather than alone.
2. Add a write-into-buffer output element and the visit method that feeds it. This is the real design:
   the row closure becomes `Fn(A::Elems<'_>, &mut [T])`, the executor preallocates `rows * width` from
   the output dtype rather than collecting a `Vec` per row, and wraps the flat buffer once at the end.
   Without this the port allocates once per row and loses to the columnar kernel outright.
3. Move the constant-norms fast path to `reduce_encoded`, which already sees argument values.
4. Benchmark against `vortex-tensor/benches/l2_norm.rs`'s pattern before keeping it, since the whole
   point is that the row form is not slower.

Step 2 is also what a `str -> str` string library needs, so it has two prospective users rather than
one, which is the argument for building it properly rather than special-casing tensors.

---

## Is there anything left to port?

Asked directly: could the remaining hand-written vtables move onto `RowFn` if the element vocabulary
covered more types? Classifying all ~30 of them says no, and says the vocabulary is not what is
stopping them.

| blocker | count | members |
| --- | --- | --- |
| **Not strict.** `RowFn` implies strict, so these cannot reach it at all. | 12 | `between`, `case_when`, `cast`, `dynamic`, `fill_null`, `is_null`, `is_not_null`, `list_contains`, `pack`, `stat`, `row_size`, `zip` |
| **The answer already exists in bulk.** Zero-copy child projection, a metadata field, or a vectorized slice kernel. A row loop would be strictly slower. | 12 | `not`, `list_length`, `binary`, `mask`, `ext_storage`, `get_item`, `select`, `merge`, `variant_get`, `geo.envelope`, `json_to_variant`, `row_encode` |
| **No element rows to read.** Zero-arity, or a type-erasure adapter. | 5 | `literal`, `root`, `row_idx`, `row_count`, `ForeignScalarFnVTable` |
| **Output side.** Nullable output, or an output dtype that depends on runtime data. | 3 | `list_sum`, `l2_denorm`, `geo.envelope` |
| **Value-dependent per-batch setup.** | 1 | `like` |

`geo.envelope` is the one function counted twice: its output is a struct-of-four extension type *and*
its fast paths hand back existing child arrays untouched.

`binary` deserves a note, since on strictness alone it looks portable: only its Kleene `And`/`Or` are
non-strict, and `is_strict` already varies by operator, so comparison and arithmetic go through the
strict lifting today. What keeps it columnar is the kernel. `collect_zip_bits` and `LaneZip` run over
`as_slice()` pairs as tight vectorizable loops, with a separate constant-operand path
(`collect_bits(lhs, |a| a.is_eq(rhs))`). Routing that through a per-row closure and `ArgColumn::get`
would give up the slice-level vectorization for nothing.

Three things follow.

**The porting well is dry.** The seven functions already on `RowFn` (`byte_length`, the three tensor
kernels, the three geo kernels) are the complete set in this repository that wants a row loop. Every
remaining one is blocked, and forcing any of them onto `RowFn` would cost performance rather than
save lines.

**Missing elements are not the constraint.** Only `list_contains` would need new input vocabulary, and
it is independently blocked by non-strictness, so a list element would not unblock a single function
today. A list *input* element is nonetheless easy (`Bytes` already proves the shape: `Elem<'a>` is a
GAT, so `&'a [T]` works), and `list_length` could even be a `RowFn` given a `ListLen` element in the
style of `BytesLen`. It should not be, because its answer is a child array or one constant.

**`like` is a new gap, and the sharpest one.** It is strict, infallible, `(Utf8, Utf8) -> Bool`: on
signature alone it is the ideal `RowFn`. Two things block it, and measuring both is what settled where
it belongs.

Its constant-pattern path is fine. `reduce_encoded` already sees the argument arrays before the row
loop, so compiling the pattern once and evaluating in bulk has a home, and a constant operand stays
constant even through a filtered batch. No new hook needed for that case.

Its *per-row* pattern path is what blocks it. That path memoizes the compiled pattern across
consecutive rows carrying the same one, and a `RowFn` closure is `impl Fn`, so it can hold no such
state. Defeating the cache costs **5.7x** (`like_per_row_distinct_patterns` 249.1 µs against
`like_per_row_patterns` 44.03 µs, 2048 rows, same matching work in both), which is the same shape of
regression the constant-operand stride fixed for geo.

Relaxing the closure to `impl FnMut` would restore the cache, and it compiles as a one-word change.
It is not free. Measured on `byte_length_element`, `fastest` column, both configurations run twice:

| case | `Fn` | `FnMut` | delta |
| --- | --- | --- | --- |
| `long_strings_bytes_len` 4096 | 11.15 µs | 12.08 µs | +8.3% |
| `long_strings_bytes_len` 65536 | 166.4 µs | 181.7 µs | +9.2% |
| `long_strings_bytes_slice` 4096 | 14.75 µs | 15.97 µs | +8.3% |
| `short_strings_bytes_len` 65536 | 166.2 µs | 180.4 µs | +8.5% |
| `short_strings_bytes_slice` 65536 | 180.9 µs | 200.3 µs | +10.7% |

Capturing the closure by `&mut` inhibits the vectorization the shared capture allows, so `FnMut`
taxes every row function 8 to 11% to enable state that one function wants. Keep `visit` on `Fn`.

The conclusion is that `like` does not want a row loop at all: its general path needs cross-row state,
and its fast path is bulk. What it wants is to declare `(Utf8, Utf8) -> Bool` through the element
vocabulary and keep its own kernel, which is the missing cell below. A per-batch setup hook would not
have been enough on its own, since the state `like` needs is mutable *across* rows rather than fixed
before them.

A second, smaller thing blocks `like` too: it renders custom SQL through `fmt_sql`, and neither
`StrictScalarFnVTable` nor `RowFn` forwards that, so today porting any function with bespoke SQL
rendering would silently lose it.

---

## Known gaps and future work

Found by the porting probes, left unfixed here because each is a larger change with its own review
surface. Recorded so they are decisions rather than surprises.

- **~~No constant-operand affordance.~~ Fixed.** A partially-constant call used to decode the constant
  column in full, so a broadcast operand cost one decode per row (measured: a broadcast query vector
  cost the same as a genuine column, 234 ms vs 226 ms at 50k x 256). That was what kept the geo
  functions off `RowFn`, since porting them would have traded one geometry parse for *n*. Each decoded
  column now carries a stride, 0 for a constant, and the geo functions are row functions.
- **`NullHandling::Dense` is chosen on safety alone, with no cost input.** For a fixed-width element
  (`TensorRow`) dense is unambiguously cheaper. For an unbounded-width row (a nested list) the garbage
  behind a null row need only be *in bounds*, so it can span the whole elements array, which is
  pathologically O(nulls x elements). No current function hits this, but the choice should consider
  width.
- **`OutputElement::build(Vec<Self>)` forces materialization.** A row function's output is always a
  freshly built `Vec` turned into a `PrimitiveArray`, so it cannot return a `ConstantArray` or a lazy
  child. This is why `list_length` is a columnar `StrictScalarFnVTable` rather than a `RowFn`, since a
  row port would materialize one `u64` per row and lose the `FixedSizeList` constant. A columnar output
  escape that stays inside the framework ("given the decoded columns, can you produce the whole output
  at once?") would let `list_length`, `byte_length` and `not` share one abstraction.
- **The missing cell.** The two authoring traits cover *declare-signature-once + row-loop* (`RowFn`)
  and *hand-write-signature + own-kernel* (`StrictScalarFnVTable`). The cell for
  *declare-signature-once + own-kernel* is empty, so a columnar function hand-writes five signature
  methods (`arity`, `child_name`, `return_element_dtype`, `null_handling`, `is_fallible`) that are all
  mechanically derivable from an element tuple.

  **It is buildable.** The obvious worry is coherence, since `RowFn` already blanket-impls
  `StrictScalarFnVTable` and a second blanket impl of the same trait is a hard E0119 conflict. The way
  through is to layer rather than branch, putting the new trait *between* the two:

  ```text
  StrictScalarFnVTable  <-blanket-  StrictSignature  <-blanket-  RowFn
  ```

  One blanket impl per edge, so nothing overlaps, and a columnar function hand-writes `StrictSignature`
  while a row function reaches it through `RowFn`. Compiling the shape confirms a hand-written impl
  coexists with the blanket one, including from a *downstream* crate, because within the crate that owns
  the type rustc can see the blanket impl's bound does not hold. This is not a new trick here:
  `impl<V: StrictScalarFnVTable> ScalarFnVTable for V` already coexists with `Like`'s and `Between`'s
  hand-written `ScalarFnVTable` impls the same way.

  **The user count is 3, not 12, and 2 of those need an element first.** Being in the columnar category
  is not enough: the function's *signature* has to be expressible in the vocabulary, and
  `element_dtype()` taking no arguments rules out every function whose return dtype is derived from its
  input at runtime. That is most of them: `mask` returns `arg_dtypes[0].as_nullable()`, `ext_storage`
  returns `ext_dtype.storage_dtype()`, `get_item` and `select` a projection of the input struct,
  `variant_get` an options-derived dtype, `binary` a width negotiated between operands. What is left is
  `not` (`(bool,) -> bool`, usable today), `like` (`(Bytes, Bytes) -> bool`, usable today once `fmt_sql`
  forwards), and `list_length` (needs a `ListLen` element in the style of `BytesLen`).

  So this is worth building *after* the elements that give it a third user, not before. Against ~140
  lines of new trait and blanket impl it would save roughly 20 lines per function, which at one usable
  caller is a wrapper with one impl. The cheap interim is to make `validate_row_args`,
  `row_null_handling` and `row_is_fallible` public, which turns each hand-written signature method into
  a one-liner and removes the *logic* duplication (each function currently rolling its own dtype check
  and asserting rather than deriving its null handling) without adding a layer.
- **No nullable output element, so no non-total `RowFn`.** `OutputElement::build` always produces an
  all-valid column, so a row kernel cannot return a null from a valid row. `impl OutputElement for
  Option<T>` is the whole fix. Left out because nothing needs it *yet*: `list_sum` would need it, but
  is columnar for independent reasons too (the grouped-accumulator path and the `FixedSizeList`
  constant).
- **No borrowed output element, so no zero-copy row function.** A row closure returns an
  `ApplyResult`, which is `'static`, so its result cannot borrow from the input columns. Note the
  asymmetry with the input side, where `InputElement::Elem<'a>` is a GAT and borrows freely. Every
  `str -> str` function therefore copies: `OutputElement for String` allocates one `String` per row
  and then rebuilds views from them. A string library would hit this on its first `upper`. Two
  distinct fixes, of increasing scope:
  - `upper`, `lower` and `replace` genuinely allocate, and want a `Cow<'a, str>` output element. That
    needs `OutputElement` to grow its own lifetime GAT and `build` to take an iterator rather than a
    `Vec`, so a borrowed row passes through without a copy and an owned one is built in place.
  - `trim`, `substring`, `left` and `right` want more than a `Cow` can give. Their result is a
    *slice* of the input, so the right kernel keeps the input's data buffer entirely and rewrites
    only the views, copying no bytes. That stays columnar whatever the output element can express.

  Predicates and measurements (`starts_with`, `contains`, `byte_length`) have none of this problem
  and are already the best case for `RowFn`, so the split for a string library falls along the return
  type rather than the argument type.

  **A plain higher-ranked bound does not get there,** which is worth recording because it looks like
  it should. Writing the visit as `impl for<'a> Fn(A::Elems<'a>) -> R::Elem<'a>` fails with
  [E0582]: the `Fn` sugar puts `R::Elem<'a>` in an `Output` binding, and rustc requires the bound
  lifetime to appear *structurally* in the trait's input types before a binding may reference it. An
  opaque projection `A::Elems<'a>` does not count, even though it plainly mentions `'a`. Three routes
  around it, measured by compiling each:

  | route | works | cost |
  | --- | --- | --- |
  | concrete input type instead of `A::Elems<'a>` | yes | gives up the element abstraction |
  | custom callable trait with a generic `apply` method | yes | callers write a struct per kernel, not a closure, and the impl must spell `<Bytes as InputElement>::Elem<'a>` rather than `&'a str`, or hit [E0195] |
  | pass a zero-sized `Row<'a>(PhantomData<&'a ()>)` token beside the row | yes | closures survive, but every row closure grows an ignored parameter |

  The third is the one to build on: the token makes `'a` appear structurally in the `Fn`'s inputs,
  which satisfies E0582 and lets the `Output` binding reference it, and plain closures still infer.
  The ignored parameter is a tax on *every* row function though, so the shape to prefer is a second
  visit method for lending kernels, leaving today's `visit` untouched for the `'static` majority.

  [E0582]: https://doc.rust-lang.org/error_codes/E0582.html
  [E0195]: https://doc.rust-lang.org/error_codes/E0195.html
- **`OutputElement::element_dtype()` takes no arguments,** so a row function's output dtype is a
  property of its Rust type and cannot depend on runtime data. This is the other thing keeping
  `l2_denorm` columnar: it returns whole tensor rows, and a tensor's dtype carries its shape.

  **This one is a signature choice, not a law, and calling it a law was wrong.** `l2_denorm`'s
  `return_element_dtype` just returns `arg_dtypes[0]` with nullability unioned in, and the blanket
  `RowFn` path already *has* the input dtypes at that point: `validate_row_args` does
  `A::validate(args)?; Ok(R::Out::element_dtype())`, discarding `args` for the output. Widening to
  `element_dtype(args: &[DType]) -> VortexResult<DType>` would let a `TensorRowOut<T>` hand back
  `args[0]`, and `l2_denorm`'s input side needs nothing new at all, since `(TensorRow<T>, T)` are both
  existing elements.

  What still blocks it is the *other* signature. `build(values: Vec<Self>)` with `Self = Vec<T>` means
  one heap allocation per row and then a flatten, against a columnar kernel that scales the flat
  storage buffer in a single pass. At 16k rows that is 16k allocations versus zero, and no amount of
  dtype plumbing fixes it. Making `l2_denorm` a fast row function needs an output element that writes
  into a preallocated flat buffer (`fn apply(row, out: &mut [T])`) rather than returning an owned row,
  which is a genuinely different output shape and the thing to design if this is wanted.

  Note also what *not* to do on the input side: replacing the generic `TensorRow<T>` with a
  non-generic element whose `Elem<'a>` is an enum over `f16`/`f32`/`f64` would move the width choice
  from monomorphization into a branch inside the row loop. That is precisely what
  `match_each_float_ptype!` plus a generic element exists to avoid, so it would cost every tensor
  kernel its inner-loop specialization.
- **~~The witness carries four scalars through two associated types.~~ Not a gap.** This looked like
  the framework's weakest joint, since `ArgsWitness` and `RetWitness` are read *only* for `ARITY`,
  `DENSE_SAFE`, `DECODE_FALLIBLE` and `FALLIBLE`, and for a multi-dispatch function the witness names
  an arbitrary representative (`L2Norm` says `f64` for no reason a reader can see). The plan was to
  collapse them into three consts.

  Checking the signatures says no. `arity`, `null_handling` and `is_fallible` on
  `StrictScalarFnVTable` all take *only* the options, with no input dtypes, while `dispatch` needs
  dtypes to choose. So those three answers **must** be dtype-independent, which means they cannot be
  read off whatever element types a batch picks, which is exactly why a separate declaration has to
  exist. The witness is not redundant bookkeeping; it is the only place those facts can live.

  Given that, types beat consts. With types, dense-safety and fallibility are *derived* from the
  element types, so the only available mistake is a witness that disagrees with the dispatch, and that
  is a build error. With three hand-written consts an implementor could state a fact wrongly *and*
  visit consistently with their mistake. Converting would be a notation change that removes a
  derivation, not a fragility fix. Left alone, with the reason now recorded on `ArgsWitness` so the
  next reader does not re-open it.

  What is left of the original complaint is presentational: the arbitrary representative reads oddly.
  A doc line on each multi-dispatch implementor saying why the width shown is arbitrary is the whole
  fix.
- **`InputElement` is an open trait with required consts.** Adding `DECODE_FALLIBLE` broke every
  out-of-crate element (`TensorRow`) until updated. If elements are a real extension point for other
  crates, `DENSE_SAFE` / `DECODE_FALLIBLE` should carry conservative defaults.
- **`DENSE_SAFE`'s doc guidance is subtly wrong for lists.** It says `false` for "any element that
  follows an offset," but a list element *is* dense-safe, because list arrays validate
  `offsets[i] + sizes[i] <= elements.len()` for every row including nulls. Following the doc literally
  would put `list_length` on `Filter` and lose its encoding fast paths.

---

## What the ports bought

**Not line count.** That was the first justification I reached for and it does not hold up: `row/` is
514 code lines and `strict/` is 269, against roughly 470 lines saved across six kernels. Near
break-even. Nor is it bug fixes, since none of the three extracted problems is a live miscompute on
`develop`.

**It is `unsafe`.** Every hand-written kernel in `vortex-tensor` ended the same way:

```rust
// SAFETY: The buffer length equals `len`, which matches the source validity length.
Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
```

A kernel that computes its own values *and* carries its input's validity has to assert that the two
lengths agree, and the only tool for that is `new_unchecked`. The framework never pairs them:
[`OutputElement::build`] returns a non-nullable column, and the strict lifting applies validity
afterwards by masking. The invariant stops being asserted and becomes unrepresentable.

Counting production `unsafe` blocks, test modules excluded, gives a clean natural experiment. The
three functions moved onto the row layer lost all of theirs; `l2_denorm`, which stayed columnar on
`StrictScalarFnVTable`, kept all of its own:

| function | layer it moved to | `unsafe` on `develop` | `unsafe` now |
| --- | --- | --- | --- |
| `l2_norm` | `RowFn` | 1 | 0 |
| `inner_product` | `RowFn` | 3 | 0 |
| `cosine_similarity` | `RowFn` | 3 | 0 |
| `l2_denorm` | `StrictScalarFnVTable` | 8 | 8 |

Same crate, same reviewers, same standards, so the row layer is what removes them rather than the
strict lifting or the port itself. `develop`'s `l2_norm` also hand-rolled a 25-line constant-array
fast path that the strict lifting now does generically for every function, and computed its output
nullability by hand.

This is the justification to carry onto a clean branch. It also bounds the claim: a `vortex-tensor`
local helper owning the same invariant would remove the same `unsafe`, so what earns the *generic*
placement in `vortex-array` is that `vortex-geo`'s three predicates and `byte_length` use it too,
over three different element types. Two downstream crates plus core is the second-caller test met, not
anticipated.

### What it costs

Removing that `unsafe` is not free, because `new_unchecked` was buying something: the old kernel paired
its freshly built buffer with the input's validity in one step, so a nullable input cost it nothing
extra. The framework builds a non-nullable column and the lifting applies validity afterwards, which
for `Validity::Array` means materializing a mask and running a separate pass.

That pass is `O(rows)` while the kernel is `O(rows * width)`, so width amortizes it. Measured on
`vortex-tensor/benches/l2_norm.rs`, 16384 rows, `fastest` column:

| width | non-nullable | nullable | cost of the extra pass |
| --- | --- | --- | --- |
| 2 | 68.87 µs | 70.44 µs | +2.3% |
| 32 | 241.4 µs | 243.9 µs | +1.0% |
| 256 | 2.513 ms | 2.529 ms | +0.6% |

So 1 to 2% on nullable input, worst at the narrowest vector anyone would store, and nothing at all on
non-nullable input where no mask is applied. Trading that for seven `unsafe` blocks is the right side
of the deal.

### The like-for-like comparison, and a fixed cost worth chasing

The table above compares the framework against itself, so it isolates the masking pass but says nothing
about the rest of the machinery. `PrePortL2Norm` in the same benchmark closes that: a bench-local
`ScalarFnVTable` running the identical arithmetic, indexing the flat slice directly into a `Buffer` and
attaching validity in one step. `fastest` column, non-nullable, 16384 rows:

| width | framework | pre-port | delta |
| --- | --- | --- | --- |
| 2 | 69.44 µs | 32.87 µs | **2.11x slower** |
| 32 | 276.7 µs | 273.0 µs | +1.4% |
| 256 | 2.758 ms | 2.802 ms | parity |

**The overhead is fixed per batch, not per row.** In absolute terms the gap is 36.6 µs at width 2,
3.7 µs at 32, and negative at 256. A per-row cost would hold roughly constant per row across widths;
one that shrinks as total work grows is a constant being amortized. So the framework carries something
like tens of microseconds of fixed setup that a hand-written kernel does not.

**Where it most likely comes from, and why this is not yet an indictment of the row layer.**
`PrePortL2Norm` implements `ScalarFnVTable` *directly*, so the engine calls its `execute` and nothing
else. The framework path first runs the whole strict lifting: collecting inputs, cloning arg dtypes,
`return_dtype`, the null-constant and all-constant checks, conjoining validities, choosing null
handling, then `execute_strict`, then `reduce_encoded`'s encoding probe, then `dispatch`'s width match,
and only then the row loop. Most of that is the *strict* layer rather than the row layer, and some of
it (constant folding, validity derivation) is work the hand-written kernel simply does not do, not work
it does faster.

So the honest statement is: the framework costs a fixed tens of microseconds per batch, which is
invisible at production tensor widths and 2x at width 2. Before this goes in a PR description it needs
decomposing, because "the row layer is slow" and "the strict lifting does more bookkeeping" have very
different remedies. The cheap next step is a third bench arm implementing `StrictScalarFnVTable` with
the pre-port kernel body, which shares the lifting but skips the row layer and splits the difference in
two.

[`OutputElement::build`]: vortex-array/src/scalar_fn/row/element/mod.rs

Production lines, before and after:

| function | layer | before | after |
| --- | --- | --- | --- |
| `byte_length` | `RowFn` (fixed) | n/a | 23 (impl) |
| `list_length` | `StrictScalarFnVTable` | 189 | 143 |
| `not` | `StrictScalarFnVTable` | 76 (impl) | 53 (impl) |
| `list_sum` | `StrictScalarFnVTable` | 78 (impl) | 56 (impl) |
| `l2_norm` | `RowFn` (width) | 254 | 96 |
| `inner_product` | `RowFn` (width) | 277 | 112 |
| `cosine_similarity` | `RowFn` (width) | 309 | 203 |
| `l2_denorm` | `StrictScalarFnVTable` | 731 | 706 |
| geo x 3 | `RowFn` (fixed) | 51 each (impl) | 15 each (impl), plus one shared element |

Nothing outside the functions' own crates changed: the `L2DenormScheme` compressor and every
`ExactScalarFn` matcher are untouched, because the encoding-aware push-downs key off the function
*type* rather than its vtable layer.

The line-count case does not close on its own. The framework is ~1330 production lines and removes
~760 across the ported functions, so **net this branch adds lines**, amortizing around the fourteenth
function against ~20 strict candidates in the tree. To be honest, the case for merging is the marginal
cost of the *next* function (~15 lines, and the invariants above enforced rather than reviewed), plus
the correctness the type-derived properties buy, rather than the diff.

---

## Measurements

`vortex-array/benches/byte_length_element.rs`, element choice for `byte_length`, whole-execution
medians:

| input | `BytesLen` | `Bytes` | |
| --- | --- | --- | --- |
| 64Ki non-inlined rows | **206 µs** | 256 µs | 24% faster |
| 64Ki inlined rows | **207 µs** | 215 µs | 4% faster |

`vortex-array/benches/strict_validity.rs`, how the `Dense` path applies validity, same kernel in both
arms:

| | `lazy` | `eager` | |
| --- | --- | --- | --- |
| 64Ki, one call | **9.0 µs** | 75.3 µs | 8.3x faster |
| 1Mi, one call | 1.357 ms | 1.357 ms | parity |
| 64Ki, chain of 3 | **28.3 µs** | 30.6 µs | 7% faster |

`Validity::and` is already lazy, so the conjunction is never materialized to be applied. Only
`NullHandling::Filter` needs positions, and only it pays for them.

`not`, word-wise kernel against the row loop it would have if it were a `RowFn` (release, identical
outputs asserted):

| len | word-wise `!` | row loop + `bool::build` |
| --- | --- | --- |
| 64Ki | 927 ns | 376 µs (**406x**) |
| 1Mi | 10.3 µs | 5.83 ms (**569x**) |

This is why `not` is a columnar `StrictScalarFnVTable` rather than a row function.

---

## Rejected alternatives

- **A wrapper type instead of a blanket impl** (`Strict<MyFn>`): forces churn at every call site,
  meaning matchers, kernel registrations, and expression constructors. The blanket impl means a port
  edits only the function's own impl block.
- **A `row_family!` macro, a per-crate GAT family, or a framework GAT family**: three encodings of
  "element types as a function of the width," all paying for the same limit (the width bound has to
  appear literally in a GAT), so each width class needed its own trait *and* adapter. The rank-2
  visitor replaces the whole lineage with one non-generic trait method and no generated code.
- **`ElementwiseFn` as a third trait**: subsumed by `RowFn` with a constant dispatch, see above.
- **One `RowFn` with defaulted `dispatch` and `apply`**: converts "define nothing" from a compile
  error into a runtime panic.
- **Renaming `StrictScalarFnVTable` to `TotalFnVTable`**: the trait admits non-total members on
  purpose, so the name would be wrong.
- **An `is_total` method feeding a derived `validity`**: a new concept to compute what a function can
  state directly. Mirroring `validity` with a `None` default makes the unsound answer the one that
  takes work.
- **Macro-generated per-type constructors**: a bespoke API per function, where the general
  `ScalarFnFactoryExt::try_new_array` is what every other scalar function already uses.
- **A separate `FallibleElementwiseFn`**: an associated return type (`ApplyResult`) costs one line per
  function instead of a whole trait and a spent coherence slot.
