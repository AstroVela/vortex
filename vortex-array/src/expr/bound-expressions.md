<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Bound Expressions

## Context

We want an explicit **bind** step: a phase that resolves scope-dependent references, type-checks the
tree, and records a dtype on every node — run once, before an expression is applied.

None of that exists as a phase today.

- **Types are derived on demand, repeatedly.** `return_dtype` re-walks the whole subtree on every
  call with no memoization, so any pass wanting types at more than one node re-walks overlapping
  subtrees: `transform/coerce.rs`, `analysis/referenced_field_paths.rs`, `transform/partition.rs`,
  `stats/rewrite.rs`. `SimplifyCache` in `expr/optimize.rs` is a local patch for exactly this,
  applied inside the optimizer and nowhere else.
- **A dtype only becomes durable at the physical layer.** It is first materialized when the
  expression is *applied*, where `ScalarFnArray` hands it to `ArrayParts`. Note the asymmetry that
  creates: the array tree memoizes as it is built, because each child array carries its own dtype;
  the expression tree never did.
- **There is no validated form.** Nothing sits between "a tree someone constructed" and "arrays", so
  there is no type at which well-typedness is already established, and no single place where a
  scope-dependent reference is resolved. That place is a prerequisite for lambda parameters later,
  which cannot be typed from the tree alone.

Moving `Root` out of `ScalarFnVTable` and making `Expression` an enum is a **supporting change**,
not a second goal. Binding needs a node it can resolve against the scope, and `Root` is that node —
its dtype comes from the scope rather than from its children. A scalar function cannot express that:
`ScalarFnVTable`'s two central methods are `return_dtype` and `execute`, and `Root`'s vtable bailed
from both while `child_name` was `unreachable!`. Ten `is::<Root>()` special cases existed to work
around it. Tidying that up is a welcome side effect rather than the motivation.

## What changed

The bind step:

```rust
pub struct Scope {
    root: DType,
}

pub struct BoundExpression {
    kind: BoundKind,
    dtype: DType,
}

pub enum BoundKind {
    Scalar {
        scalar_fn: ScalarFnRef,
        children: Arc<Vec<BoundExpression>>,
    },
    Root,
}

impl Expression {
    /// Type-check the whole tree in one walk, resolving `Root` against the scope.
    pub fn bind(&self, scope: &Scope) -> VortexResult<BoundExpression>;
}
```

`return_dtype(&DType)` survives as a shim over `bind`, so its call sites are untouched. Callers
wanting types at more than one node should bind once and read fields instead:

```rust
pub fn return_dtype(&self, scope: &DType) -> VortexResult<DType> {
    Ok(self.bind(&Scope::new(scope.clone()))?.dtype().clone())
}
```

To give binding something to resolve, `Expression` becomes an enum with `Root` as a variant:

```rust
pub enum Expression {
    Scalar {
        scalar_fn: ScalarFnRef,
        children: Arc<Vec<Expression>>,
    },
    Root,
}
```

`Root`'s `ScalarFnVTable` impl and its session registration are deleted. The ten special cases
become match arms or `is_root()` calls.

## What binding is, and is not

Binding is **purely logical**. `Scope` and `BoundExpression` hold nothing but `DType`s; the walk
never sees an array, a length, an encoding or a `PType`. It is type-checking and name resolution.

| Stage | Representation | Side |
| --- | --- | --- |
| build | `Expression` | logical |
| `bind` | `BoundExpression` | logical |
| `apply` | `ScalarFnArray` — lengths and encodings present | physical |
| `execute` | canonical `ArrayRef` | physical |

A `ScalarFnRef` spans the first three: of `ScalarFnVTable`'s 15 methods only `execute` is physical,
dispatching into the kernels under `scalar_fn/fns/*/kernel.rs`. A scalar function is not "the
physical half" of an expression; it is a logical contract with one physical entry point.

**`bind` and `apply` answer different questions and can disagree.** `bind(scope)` asks whether the
expression type-checks against a *declared* scope; `apply(array)` asks whether it type-checks
against an *actual* array, which it already verifies implicitly — `ScalarFnArray::try_new_with_len`
calls `return_dtype` on the real child array dtypes and propagates the error. Nothing connects the
two, so a `BoundExpression` is **not** proof that `apply` will succeed.

That is why `apply` still takes `&Expression` and no `apply_bound` exists yet. See
*Deliberate omissions*.

## Design decisions

### `BoundExpression` is `{ kind, dtype }`, not a flat enum

The alternative is `enum BoundExpression { Scalar { .., dtype }, Root { dtype } }`. Hoisting `dtype`
above the variant makes "every bound node has a dtype" **structural** — a variant cannot be added
without one — where the flat form makes it per-variant discipline and turns `dtype()` into a match.
Since the entire point of the bound tree is replacing a subtree walk with a guaranteed field read,
encoding that guarantee in the shape is worth the extra type name.

This mirrors `ArrayParts` (`array/typed.rs:55`), which hoists `dtype` and `len` above the
encoding-specific `data` for the same reason.

The cost is `match bound.kind()` rather than `match bound`, mitigated by `as_scalar()`,
`children()` and `is_root()` so most consumers never match at all.

### Bound children are behind an `Arc`

`BoundExpression` needs an iterative `Drop` impl, because dropping a deep tree recursively
overflows the stack. `Drop` in turn makes the type non-destructurable by value (E0509), so a
consumer that rebuilds a tree — the optimizer, once it runs on bound input — must clone rather than
move. With a bare `Vec` that clone is a **deep copy**; with an `Arc` it is a refcount bump, matching
`Expression::Scalar`.

Sharing affects clone cost only, not construction: `bind` walks structurally and produces a distinct
node per occurrence, so two occurrences of the same subtree keep their own dtypes. That matters for
future work where the same subtree can appear under different scopes. Both properties are pinned by
tests (`clone_shares_children`, `repeated_subtree_is_bound_per_occurrence`).

### `Scope` is opaque and holds only a root dtype

With no lexical bindings there is nothing else to hold. It is a struct rather than a bare `DType` so
that frames can be added later without changing `bind`'s signature. The field is private, so adding
one is not a breaking change.

Named `Scope` rather than `ExpressionScope` because ~138 existing call sites already use
`scope: &DType` as a parameter name; a longer type name reads worse than the collision costs.

### Supporting: `Root` is a variant, not a scalar function

Its dtype comes from the scope and it has no execution. Making it a variant means both variants are
honest, deletes the vtable that lied, and turns the ten special cases into match arms the compiler
checks.

### Decision sites match the variant, rather than funnelling through `Option`

Seven sites need an answer for "what does this mean for a non-scalar node": four in `optimize.rs`
(`try_optimize` plus the untyped/typed/reduce rule helpers), plus `apply`, `stats/rewrite.rs` and
`transform/coerce.rs`. Each matches `Expression` exhaustively instead of going through
`as_scalar() -> Option<_>`.

The distinction matters for future variants. `as_scalar()` collapses "everything that isn't
`Scalar`" into one `None`, so a new variant silently inherits Root's answer. An exhaustive match
breaks. Verified empirically by adding a throwaway variant: 14 sites became compile errors, at
exactly the semantic decision points — the bind walk, `as_scalar`, `children`, `with_children`,
`validity`, `fmt_sql`, `ExactExpr`'s identity, all four optimizer entry points, `coerce`, `apply`
and `stats/rewrite`.

It correctly did *not* flag the ~116 `is::<V>()` / `as_opt::<V>()` downcast sites or the dict
pushdown's `as_scalar()`, because for those "not a scalar function" is the right answer for any
future variant.

`Ok(None)` at those sites is not an unchecked fallback — it is the answer. Previously the same
answer came from `ScalarFnVTable::simplify_untyped`'s default impl, which `Root` inherited. The
change moves where it is written, and makes the compiler ask for it.

## Migration notes

`impl Deref for Expression` is removed, since a `Root` has no `ScalarFnRef` to deref to. The
replacements:

| Before (via `Deref`) | After |
| --- | --- |
| `scalar_fn() -> &ScalarFnRef` | `as_scalar() -> Option<&ScalarFnRef>` |
| `id()` | `scalar_fn_id() -> Option<ScalarFnId>` |
| `signature()`, `options()` | same names, now `Option`-returning |
| `children() -> &Arc<Vec<Expression>>` | `children() -> &[Expression]` |

`is::<V>()`, `as_opt::<V>()` and `as_::<V>()` keep their signatures, returning `false`/`None` for
non-scalar variants.

**A panicking accessor is the wrong shim.** An interim `scalar_fn() -> &ScalarFnRef` that panicked
on `Root` let four genuinely broken sites compile — `try_optimize`, the deprecated
`simplify_untyped`, `stats/rewrite`, and `ReduceNode::scalar_fn` — and only one test caught it.
Deleting the accessor surfaced all four at compile time immediately. There is no panicking accessor
in the final shape, and there should not be one.

**`is_strict` and `is_fallible` need opposite defaults for `Root`**, and the compiler caught this
only because `signature()` became `Option`-returning. `Root` is strict (a null scope row yields a
null result) and infallible. A uniform `unwrap_or(false)` would have silently dropped Root's
strictness and changed mask-hoisting behaviour with no error.

**Wire compatibility.** `Root` still serializes as id `vortex.root` with empty metadata. Because it
is no longer in the scalar-function registry, `Expression::from_proto` resolves that id *before* the
registry lookup; otherwise every persisted expression containing a root would fail to load.

**`ExactExpr`** keyed identity on `(scalar_fn, Arc::as_ptr(children))`. `Root` has neither, so its
`PartialEq`/`Hash` now match the variant and give `Root` its own discriminant.

**Measured cost.** Outside `vortex-array`, the change produced **three** compile errors, all in
`vortex-layout/src/layouts/dict/reader.rs`. An earlier estimate of ~116 downcast sites plus ~80
signature sites across nine crates was wrong: it conflated `ScalarFnRef`-side method calls with
`Expression`-side ones, and did not account for `is::<V>()`/`as_opt::<V>()` keeping their signatures
or for `&Arc<Vec<T>>` → `&[T]` being source-compatible through deref coercion.

The `dict/reader.rs` site is the dictionary pushdown predicate. Its behaviour is preserved exactly:
`Root` was never pushdown-eligible, because `is_negative_cost` admits only `ByteLength`,
`ExtStorage`, `GetItem` and `Literal`.

## Deliberate omissions

- **`apply_bound`.** It would have no caller, and no advantage over `apply`, which already
  type-checks against real array dtypes. Adding it is only worthwhile alongside the next item.
- **Reusing bound dtypes during `apply`.** A `try_new_with_dtype` that trusts the bound dtype would
  save one `return_dtype` call and one `Vec<DType>` allocation per node per chunk. It is sound —
  binding against the array's own dtype provably yields the dtypes the array layer derives, by
  induction over the tree — but it **must** land in the same commit as promoting the optimizer's
  dtype check from `#[cfg(debug_assertions)]` to unconditional. That check
  (`optimizer/rules.rs`, `reduced.dtype() == parent.dtype()`) is debug-only precisely because
  `try_new_with_len` re-derives in release; removing the derivation without promoting the check
  leaves release builds with no dtype validation in the apply path at all.
- **Migrating the optimizer onto `BoundExpression`.** `SimplifyCtx::return_dtype` takes
  `&Expression` and is implemented against 39 `simplify` impls, so the ctx would need to map an
  arbitrary `&Expression` back to a bound node. Retaining a source expression on each bound node
  solves it, but that is a separate change with its own rewrite-invalidation policy.
- **Lambdas and higher-order functions.** The variants and scope frames they need are additive to
  everything above.

## Known gaps

`analysis/strict.rs` and `analysis/fallible.rs` still funnel through `signature() -> Option`, so a
future variant would silently inherit `strict = true` and `fallible = false`. Both are wrong in the
unsound direction — the first licenses mask-hoisting through unknown semantics, the second licenses
dictionary pushdown over values that may fail. Root's answers are correct today, so this is a gap in
future safety rather than current correctness. Two exhaustive matches close it.

The tree-display label for `Root` is the literal string `"vortex.root()"`, preserved so that a pure
refactor does not change output. There is no longer a function by that name.

## Verification

```bash
cargo build --workspace          # eight crates beyond vortex-array are in scope
cargo nextest run -p vortex-array
cargo nextest run -p vortex-layout -p vortex-file -p vortex-scan
cargo test --doc -p vortex-array
cargo +nightly fmt --all
cargo clippy --all-targets --all-features
git diff --check
```

All pass. 3067 tests in `vortex-array`, 343 across `vortex-layout`/`-file`/`-scan`, 72 doctests.
The diff is 313 insertions and 251 deletions across 26 files, including the 87-line deletion of
`scalar_fn/fns/root.rs`.

The workspace build is the load-bearing check: removing `Deref` changes method signatures rather
than merely adding variants, so a missing accessor surfaces outside `vortex-array` rather than in a
narrow build.
