# `list_transform` — Higher-Order List Functions as Vortex Expressions

*Goal: support `list_transform(l, x -> f(x))` — apply an expression to every element of a
list, preserving list structure — as a first-class Vortex expression.*

---

## 1. Motivation

`list_transform` is the gateway to the whole higher-order function (HOF) family — `list_filter`, `list_reduce`, `element_at` — because it introduces the one
thing Vortex expressions don't have yet: **a sub-expression evaluated in a different scope** (the
list's elements) **at a different cardinality** (element count, not row count).

Why it's worth doing well:

1. **Structure-preserving execution.** `list_transform` never changes list *shape* — offsets and
list validity pass through untouched. Only the elements child is rewritten. In Vortex terms,
the transform of a `ListArray` is `ListArray::try_new(apply(elements, body), offsets, validity)` — a deferred, nearly-free wrapper.
2. **Compression-aware element evaluation.** Because the body is applied to the elements child via the normal deferred expression machinery, all existing encoded-array kernels fire: a
dictionary-encoded elements child evaluates the body over dictionary *values* only; constant
elements fold to a constant. Eager engines (DuckDB) flatten to vectors first and can't do this.
3. **Layout pushdown.** With the shredded `ListLayout` (elements / offsets / validity as
independent sub-layout trees), `list_transform(col, x -> x.a)` should read **only the `a`
sub-layout of the elements subtree** — offsets and validity stream through unchanged. This is the `len`/`element_at` pushdown frontier from `list-storage-formats.md`, extended to arbitrary element expressions.

## 2. Survey: how other engines implement lambda HOFs

Every vectorized engine converges on the same two ideas. **(a)** The lambda is a *bind-time*
construct, never a runtime value: the body is bound, stashed somewhere the executor can reach, and the lambda node disappears from the expression tree before execution. **(b)** Execution evaluates the body **once over the flattened elements**, then reattaches the original offsets; outer-column references ("captures") are replicated to element cardinality, usually zero-copy.

### DuckDB (the closest model; verified against source @ `36c64ee8`, 2026-07)

- **Representation.** Parse-level `LambdaExpression` (shared with the JSON `->` operator;
disambiguated in the binder, which is why DuckDB 1.3 introduced Python-style `lambda x: ...` syntax and is deprecating `->`). The binder pushes a `DummyBinding` for the parameters onto a scope stack (`lambda_bindings`, innermost-first — nested lambdas shadow), binds the body, and rewrites every leaf to a physical slot index (`BoundReferenceExpression`) against a fixed input chunk layout: `[params, reversed][outer-lambda params][captures]`.
- **Capture hoisting is the load-bearing idea.** Column refs from the enclosing row scope are
hoisted out of the body and appended as *ordinary arguments* of the surrounding call:
`list_transform(l, x -> x + col)` physically becomes `list_transform(l, col)` with the body in bind data. Captures thus get all standard per-argument optimizer machinery (folding, CSE, projection pushdown) for free.
- **The body lives in `FunctionData` (bind data), not as a child.** Each HOF supplies a
`bind_lambda` callback mapping parameter index → type (param 0 = list child type, param 1 =
`BIGINT` index). The `BoundLambdaExpression` child is erased after binding. Known footgun:
generic expression-tree walks (e.g. volatility for constant-folding) don't see the body, so
foldability must special-case it (`BoundFunctionExpression::IsFoldable`; bug history #8108).
- **Execution** (`ExecuteLambda`, post-#9395 revamp, ~10³× faster than the naive first cut):
iterate rows, gathering element positions into a selection vector; every non-constant capture
gets a parallel selection vector with `sel[k] = row_idx` — a **dictionary-vector expansion of
row values to element cardinality, zero copy**. When the batch hits `STANDARD_VECTOR_SIZE`
(2048), run a plain `ExpressionExecutor` over `[index?][element slice][captures...]`. Batches deliberately cross list boundaries (a 10k-element list = ~5 invocations; hundreds of small lists = 1). Transform output reuses input lengths with re-densified offsets; filter gathers surviving input elements rather than re-evaluating.
- **Null semantics.** `SPECIAL_HANDLING`: NULL list → NULL result row (body never runs on its
elements); NULL *element* flows through the body's own null semantics (`x + 1` → NULL);
empty list → empty list, no evaluation.
- **Optimizer.** Lambda functions are const-foldable unless the body is volatile; subqueries and
UNNEST are banned inside bodies; a dedicated rewrite collapses the 3-stage list-comprehension desugaring into `list_transform(list_filter(l, cond), expr)`.

### ClickHouse

`arrayMap(x -> e, arr)` compiles the lambda to a `ColumnFunction`; captured columns are attached with `appendArguments` and **replicated to element granularity via the array's offsets**
(`ColumnFunction::replicate(offsets)`), then the body is evaluated once over the flattened nested data column. Same flatten-evaluate-reattach shape as DuckDB, replication instead of selection
vectors.

### Spark / Catalyst (the counter-example)

`transform` is a `HigherOrderFunction` whose lambda is a real expression node:
`LambdaFunction(body, Seq(NamedLambdaVariable))`, resolved by `bindLambdaFunction`. Execution is **row-at-a-time**: for each row, loop over elements, mutate the `NamedLambdaVariable`'s `AtomicReference` value, evaluate the body. No vectorization, no codegen for the loop — a well-known perf cliff. Useful lesson: named-variable nodes with interpreter-mutable state is the design to avoid in a columnar engine.

### Velox, Trino, StarRocks, Polars

- **Velox**: lambdas are first-class *vector* functions — a `FunctionVector` carries (body, capture
rows); vector functions receive it and `apply` it to element-cardinality vectors; captures are
wrapped in dictionaries to align cardinality. Fully vectorized, same core trick.
- **Trino/Presto**: lambdas compile to JVM bytecode (`LambdaDefinitionExpression`); arrays are
processed per row-block. Vectorized in blocks but not offset-preserving in the DuckDB sense.
- **StarRocks**: `array_map` with explicit capture-column replication (`array_map_expr.cpp`) —
the ClickHouse approach.
- **Polars**: `list.eval(pl.element() * 2)` — notably, the *element* is addressed by a dedicated
root-like expression (`pl.element()`) evaluated against the exploded child, with an elementwise fast path that skips per-list grouping. This is the closest existing design to what's proposed below: **the lambda "variable" is just the root of a sub-expression whose scope is the elements array.**

### DataFusion (lambda support is NEW and shipped)

DataFusion merged lambda support in **PR #21679 (2026-04-28)**, closing the long-standing HOF issue (#14205, closed as completed 2026-04-29) — and it shipped in **DataFusion 54, the exact version Vortex pins today** (verified: the `lambda.rs` / `lambda_variable.rs` / `higher_order_function.rs` modules are present in the `datafusion-physical-expr-54.0.0` crate Vortex builds against).

The representation (Spark-style named variables, not DuckDB-style bind data):

- Logical: three new `Expr` variants — `HigherOrderFunction { func: Arc<dyn HigherOrderUDF>, args }`, `Lambda { params: Vec<String>, body }`, and `LambdaVariable { name, field }`. `HigherOrderUDF` is the lambda-accepting analogue of `ScalarUDFImpl`.
- Physical: `HigherOrderFunctionExpr` (`datafusion-physical-expr/src/higher_order_function.rs:70`; exposes `fun()`, `name()`, `args()`, `return_type()`, `nullable()`), with lambda arguments as
`LambdaExpr { params: Vec<String>, body: Arc<dyn PhysicalExpr> }`
(`expressions/lambda.rs:41`) containing `LambdaVariable { index: usize, field }` leaves
(`expressions/lambda_variable.rs:35`) — physical variables are resolved to positional indices.
- One HOF exists so far: `array_transform` in `datafusion-functions-nested`
(`array_transform_higher_order_function()`), i.e. exactly our target. SQL:
`SELECT array_transform([2, 3], v -> v != 2)`. Nested lambdas are supported.
- **Column capture was deferred upstream** (removed from the initial PR; tracked in
#21172) — DataFusion lambdas are capture-free today, which aligns exactly with the capture-free v1 proposed here.

Consequence: the DataFusion converter arm is **in scope for v1** (§6.2), same tier as the
expression itself — this is now the same story as `list_length`, where DF pushdown was the
motivating consumer.

## 3. Where Vortex is today

- An expression is `Expression { scalar_fn: ScalarFnRef, children: Arc<Vec<Expression>> }` (`vortex-array/src/expr/expression.rs:27`). Operations implement `ScalarFnVTable`(`vortex-array/src/scalar_fn/vtable.rs:39`) with typed, hashable, serializable `Options`.
- **Single scope.** Expressions evaluate against one `ArrayRef`; `root()` (`fns/root.rs`) is the
scope reference and every leaf sees the same root (`vortex-array/src/expression.rs:18`). There is no variable/binding machinery of any kind.
- **Children are evaluated eagerly, at row cardinality.** `ScalarFnVTable::execute` receives
already-evaluated child arrays (`ExecutionArgs`, `vtable.rs:320`). A lambda body therefore
**cannot be an ordinary child**: it must run at element cardinality against a scope that only
exists inside `execute` (the elements child of the evaluated list argument).
- **Deferred execution.** `ArrayRef::apply(expr)` builds a lazy `ScalarFnArray`; computation
happens via `execute_until` / canonicalization with encoded-array kernels. `list_length`'s
`execute` (`fns/list_length.rs:84`) is the model: `execute_until::<AnyList>` to reach a
`List`/`ListView`/`FixedSizeList`, then answer from structure.
- **Serialization.** Options serialize to bytes inside `pb::Expr` metadata; `deserialize` receives
a `&VortexSession` (`vtable.rs:56`) — which is exactly what's needed to deserialize a *nested expression* via `Expression::from_proto` (`expr/proto.rs:41`).
- `Expression` is `Clone + Debug + Display + PartialEq + Eq + Hash` — it satisfies every bound `ScalarFnVTable::Options` requires. Nothing blocks embedding an expression in options.

## 4. Design

### 4.1 The lambda body lives in `Options`; its scope is the elements array

```rust
/// Options for the `list_transform` scalar function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ListTransformOptions {
    /// The lambda body. Evaluated with the list's *elements* array as its root scope:
    /// within this expression, `root()` refers to the element, not the row.
    pub body: Expression,
}

pub struct ListTransform;

impl ScalarFnVTable for ListTransform {
    type Options = ListTransformOptions;
    // id: "vortex.list.transform"
    // arity: Exact(1), child 0 = "input" (the list)
    ...
}
```

Two decisions in one, each with a rejected alternative:

**(a) Body in `Options`, not as a child** — DuckDB's bind-data approach, adapted.

|  | body in `Options` (chosen) | body as child + "scoped child" support |
| --- | --- | --- |
| Framework change | none — `Options` already supports nested exprs (Eq/Hash/serde all work) | evaluator, `ExecutionArgs`, arity checking, and every visitor must learn "child N evaluates lazily in scope S" |
| Scope confusion | impossible by construction: row-scope passes (field analysis, validity rewrites, DataFusion conversion of siblings) never descend into the body with the wrong root | every existing traversal must be audited to skip/rescope the body child |
| Blind spots | scope-*independent* properties must delegate explicitly (see below) | none |
| Precedent | DuckDB bind data; Polars `list.eval` | Spark `LambdaFunction` node |

The blind-spot cost is real but small and enumerable — the same footgun DuckDB hit with
foldability (#8108), avoided here by delegation:

- `is_fallible` → `true` iff any node of `body` is fallible (walk `body` in the impl).
- `is_null_sensitive` → `false` for the *list* dimension (masking list validity commutes with
transform, same argument as `list_length`); the body's own null sensitivity is internal to it.
- `serialize`/`deserialize` → body encoded as `pb::Expr` bytes in options metadata; deserialization uses the session's scalar-fn registry, which is why `deserialize` takes a session.
- Display/`fmt_sql` → render as `vortex.list.transform(input, x -> <body>)`, printing the body with its root shown as the lambda parameter to avoid two meanings of `$` in one line.
- Expression-tree transforms (`expr/transform`, analysis passes) will not rewrite inside the body. That is *correct* for scope-dependent passes and acceptable for the rest; if a pass ever needs to see inside (e.g. a global "collect all referenced encodings"), add an explicit
`ScalarFnVTable` hook rather than flattening the scope distinction.

**(b) `root()` *is* the lambda parameter** — no variable nodes, no de Bruijn indices, no name
resolution. `x -> x + 1` is simply `add(root(), lit(1))` evaluated with the elements array as
root. This is Polars' `pl.element()` insight expressed with machinery Vortex already has:

- The elements array **is** a complete evaluation scope; the body is an ordinary expression
against it. `x -> x.a + 1` = `add(get_item("a", root()), lit(1))` — struct element access works with zero new code.
- Nesting composes: the body of a transform over `List<List<i32>>` may itself contain a
`list_transform`, whose own body's root is the inner element. Each level rebinds root naturally.
- What this deliberately *cannot* express: **captures** (outer row columns in the body) and the
**index parameter** `(x, i)`. Both have a clean forward path that changes no v1 code — the scope generalizes from "the element" to "a struct of the element plus expanded companions" (§8) — exactly DuckDB's input-chunk layout `[index][element][captures]`, expressed as a struct scope instead of positional slots.

### 4.2 Typing

```
return_dtype(options, args):
    match args[0]:
        List(elem, n)              => List(body.return_dtype(elem)?, n)
        FixedSizeList(elem, sz, n) => FixedSizeList(body.return_dtype(elem)?, sz, n)
        other                      => bail
```

`Expression::return_dtype(scope)` (`expression.rs:95`) already threads a scope dtype through `root()`, so the body types itself against the element dtype with no new machinery. Note `list_transform` preserves the *fixed-size* structure of FSL inputs (unlike DuckDB, which casts ARRAY→LIST at bind time) — the output needs no offsets, and the FSL layout fast path (no offsets IO) survives the transform.

Nullability: list-level nullability passes through (transform of a null list is a null list).
Element-level nullability is whatever `body.return_dtype(elem)` says — the body sees the element dtype's nullability and its own semantics decide the output's.

### 4.3 Execution

```rust
fn execute(&self, options, args, ctx) -> VortexResult<ArrayRef> {
    let input = args.get(0)?;

    // 1. Constant list: transform the scalar's elements once, wrap in ConstantArray.
    if let Some(scalar) = input.as_constant() { ... }

    // 2. Reach a structural list encoding (same matcher as list_length).
    let any_list = input.execute_until::<AnyList>(ctx)?;

    // 3. Rewrap: transform elements, keep structure.
    //    List:          ListArray::try_new(elements.apply(&options.body)?, offsets, validity)
    //    FixedSizeList: FixedSizeListArray::new(elements.apply(&options.body)?, size, validity, len)
    //    ListView:      canonicalize to List first (see below), then as List.
}
```

The key property: **`execute` does no element work.** `elements.apply(body)` builds a deferred
`ScalarFnArray` over the elements child; evaluation happens on demand with full access to
encoded-array kernels. If the elements child is dictionary-encoded, the body runs over the
dictionary values; if constant, it folds. This is the structural advantage over DuckDB's
flatten-into-2048-row-batches loop: DuckDB pays O(total elements) through the executor no matter how the elements were encoded; Vortex pays O(encoded elements).

**Unreferenced elements and fallibility.** Evaluating the *whole* elements child computes results
for positions no list references. For a canonical `List`, monotonic offsets tile
`[offsets[0], offsets[len])` with no gaps, so unreferenced positions exist only (a) outside
`[first, last)` after slicing, and (b) *under null lists*, whose ranges Arrow allows to hold
arbitrary values. Policy, mirroring the existing dictionary-pushdown rule (`vtable.rs:227`):

- If the body is **infallible** (`x + 1` on wrapping semantics, `get_item`, casts that can't fail):
evaluate the full child (or the `[first, last)` slice with offsets rebased by `offsets - first`
when the input was sliced — same arithmetic as `list_length_from_offsets` in reverse). Wasted work on null-list ranges is bounded and harmless.
- If the body is **fallible** (checked arithmetic, narrowing cast): evaluate only referenced
elements. v1 does this the blunt way — mask elements under null lists before applying the body (element validity = list validity repeated by lengths), or gather-compact when offsets don't start at zero. This is a correctness requirement, not an optimization: a checked cast must not error on garbage under a null list that the query never observes.

**ListView** ranges may overlap, gap, and reorder — per-position evaluation is still positionally
correct (the body is elementwise), so an infallible body can be applied to the ListView's elements
child directly, preserving the view structure (and its element-sharing win: shared elements are
transformed once). Fallible bodies canonicalize to `List` first. Start with
canonicalize-always-for-ListView if simpler; it's a performance knob, not semantics.

**Null semantics** (matches DuckDB/ClickHouse/Spark):

| input row | output row |
| --- | --- |
| `null` list | `null` list (body never observably runs on its elements) |
| `[]` | `[]` |
| `[4, null, 5]` with body `x+1` | `[5, null, 6]` — null elements flow through the body's own null semantics |

`validity(expr)` = `expr.child(0).validity()` (identical to `list_length.rs:101`);
`is_null_sensitive` = `false`.

### 4.4 Algebraic rewrites (`simplify_untyped`)

These fall out of the representation almost for free and are where the expression-level design
pays rent:

1. **Identity elimination**: `list_transform(l, x -> x)` → `l` (body is exactly `root()`).
2. **Fusion**: `list_transform(list_transform(l, f), g)` → `list_transform(l, g[root := f])` — substitute `f` for every `root()` in `g`. One pass over the elements instead of two, and one deferred wrapper instead of a materialized intermediate list. (DuckDB needs a bespoke list-comprehension optimizer rule for a special case of this; here it's a three-line
substitution because bodies are ordinary expressions.)
3. **Length preservation**: `list_length(list_transform(l, f))` → `list_length(l)` — implemented in `ListLength::simplify_untyped` (or a shared reduce rule); a transform never changes offsets, so the composition should never fetch elements at all.
4. **Constant folding** happens through the normal machinery: constant list input takes the
scalar fast path; constant *elements* fold inside the deferred body evaluation.

Fusion + length-preservation are also the safety net for engines that push down enthusiastically:
a pathological `list_length(list_transform(...))` from a generated query costs nothing.

### 4.5 Public API

```rust
/// Transform every element of a list with `body`, preserving list structure.
///
/// `body` is evaluated with the list's elements as its root scope: within `body`,
/// `root()` refers to the element. `list_transform(col("tags"), add(root(), lit(1)))`
/// is DuckDB's `list_transform(tags, lambda x: x + 1)`.
pub fn list_transform(input: Expression, body: Expression) -> Expression {
    ListTransform.new_expr(ListTransformOptions { body }, [input])
}
```

in `vortex-array/src/expr/exprs.rs` next to `list_length` (`:765`); vtable registered in
`scalar_fn/session.rs` (`:61-79`) and declared in `fns/mod.rs`. Python bindings expose it as
`vx.expr.list_transform(expr, body)` where the element is addressed by the existing root/column builder — worth a follow-up to give Python an ergonomic `element()` alias.

## 5. Layout pushdown

Day one, nothing is required: the FSL layout reader materializes the list and `.apply()`s the whole
expression (`fixed_size_list/reader.rs:312`), which now simply works for `list_transform` — and because `execute` returns a deferred wrapper, "materializes" still touches only what the body needs.

The real win lands with the shredded `ListLayout` (elements / offsets / validity sub-layouts):

- `projection_evaluation(row_range, list_transform(root(), body), mask)` routes **`body` to the elements child reader** as its own projection over the mapped element range, while offsets and validity stream through untouched — structurally identical to how the struct layout partitions expressions per field (`struct_/reader.rs:316` + `partition_expr`) and how the FSL reader maps row ranges to element ranges in `register_splits` (`fixed_size_list/reader.rs:162`).
- Composition is where it gets good: `list_transform(col, x -> x.a)` — the elements subtree is a struct layout, `body = get_item("a", root())` partitions to the `a` sub-layout, and the scan reads offsets + `elements.a` only. `list_length(list_transform(...))` after rewrite 4.4(3) reads offsets only. Neither Parquet nor ORC can express either.
- The scope alignment is exact by construction: the elements reader's dtype *is* the body's root
scope. No adapter layer.

This section is future work gated on `ListLayout` itself, but it constrains v1: the body must
remain an inspectable `Expression` (which options-embedding preserves) so the layout reader can extract and route it.

## 6. Engine integration

### 6.1 DuckDB (near-term, real lambdas today)

`vortex-duckdb/src/convert/expr.rs` already converts list functions (`list_length` at `:70-73`). For `list_transform`, DuckDB hands us a `BoundFunctionExpression` whose children are
`[list, capture...]` and whose **bind data holds the bound body** with slot-indexed leaf
references (input chunk layout `[index?][element][captures...]`). Conversion for v1:

- function name `list_transform` / `array_transform`, `has_index == false`, zero captures →
convert the body by mapping `BoundReferenceExpression{slot 0}` → `root()` and recursing through
the supported-function whitelist; wrap as `list_transform(input, body)`.
- any capture, index parameter, or unsupported body node → don't push down (DuckDB executes it).

### 6.2 DataFusion (v1 scope — DF 54 ships `array_transform` + lambdas)

The converter follows the `ArrayLength` recipe (`vortex-datafusion/src/convert/exprs.rs:166-195`

- `can_scalar_fn_be_pushed_down:612`), with one structural difference: the physical node is a
`HigherOrderFunctionExpr`, not a `ScalarFunctionExpr`, so it needs its own downcast arm in
`convert()` (`:279`) and `can_be_pushed_down_impl` (`:489`) rather than a new entry in the
scalar-function whitelist.

Conversion for `array_transform(list, LambdaExpr { params: [v], body })`:

1. Convert the list argument as usual (column / `get_item` chain); require it to resolve to
`List | LargeList | FixedSizeList` (same gating shape as `can_array_length_be_pushed_down`).
2. Convert `body` recursively with one extra rule: `LambdaVariable` with the innermost lambda's
index → `root()`. A `LambdaVariable` referring to an *outer* lambda's parameter is a
capture-across-scopes — not expressible in v1, don't push down. (Upstream DF is capture-free today anyway, #21172.)
3. Nested `HigherOrderFunctionExpr` inside the body recurses through the same arm — nested
transforms convert naturally because each level rebinds `root()` (§4.1b).
4. Multi-parameter lambdas (index form), or any body node outside the supported set → no
pushdown; DataFusion executes the expression itself, unchanged behavior.
5. Wrap as `cast(list_transform(input, body), return_dtype)` if DF's declared return field
disagrees with Vortex's (mirroring the `list_length` cast).

Add an slt suite mirroring `list_length_pushdown.slt`.

### 6.3 Direct consumers

`vortex-scan`/`vortex-file` users and the Python bindings can use the expression immediately —
worth a benchmark in `vortex-file/benches/` next to the FSL layout bench to keep the
offsets-passthrough property honest.

## 7. Risks and open questions

1. **DataFusion's HOF API is brand new** (merged 2026-04, one function so far, capture support
still open upstream as #21172) — expect the `HigherOrderUDF`/`LambdaExpr` surface to move across DF upgrades; keep the converter arm thin and conservative.
2. **Fallible bodies over unreferenced elements** (§4.3) is the subtlest correctness point. The
proposed policy reuses the existing `is_fallible` contract, but the null-list-garbage case
needs a dedicated test with a checked cast as the body.
3. **Options-embedded expressions are a new pattern.** Nothing in the vtable contract forbids it
(all trait bounds are satisfied, serde has the session hook), but `list_transform` would be the first — expect to touch display and possibly `expr/arbitrary.rs` (proptest generation) to
account for it. Any pass discovered to *need* body visibility gets an explicit vtable hook.
4. **`ScalarFnArray` over elements whose parent list is lazy**: `execute_until::<AnyList>` may
itself canonicalize a compressed list encoding; confirm the elements child it exposes preserves encodings (it should — that's the point of `execute_until` stopping at the matcher).
5. **Display ambiguity**: two nested transforms print two scopes; the `x ->` rendering must pick
distinct parameter names per nesting depth (trivial counter) or accept `$` meaning
nearest-enclosing scope, documented.

## 8. Future work (in likely order)

1. **`list_filter`** — same options-embedded body returning `Bool`; execution computes an element selection then rebuilds offsets by per-list surviving counts (DuckDB gathers surviving input elements rather than re-evaluating — same trick applies). Requires the offsets-recompute helper that `list_length_from_offsets` is the inverse of.
2. **Index parameter and captures via struct scope.** Generalize the body's scope from `element` to `struct { elem, idx?, captures... }`: positions from offsets
(`idx = iota(total) - repeat(starts, lengths)`), captures replicated to element cardinality by `repeat(col, lengths)` — ClickHouse's `replicate(offsets)`, which for Vortex is a `take` with a run-length index array and therefore *stays lazy*. Body addresses them as `get_item("elem", root())` etc. No v1 code changes shape; `ListTransformOptions` grows a `captures: Vec<Expression>` (row-scope, ordinary children semantics — hoisted like DuckDB) and
`with_index: bool`.
3. **`list_reduce`** — fundamentally different loop (sequential across positions, parallel across
rows; accumulator type widening forced DuckDB into double-binding). Do not force it into the
transform mold; design separately.
4. **`ListLayout` pushdown** per §5, once the layout exists.
5. **Pruning through transforms**: `list_max(list_transform(l, f)) > c` ⇒ for monotone `f`, prune with `f`-transformed element zone maps. Speculative; note only.

## 9. Implementation checklist (v1)

- [ ] `vortex-array/src/scalar_fn/fns/list_transform/mod.rs` — `ListTransform` vtable:
`id`, `arity Exact(1)`, `child_name "input"`, `return_dtype` (§4.2), `execute` (§4.3),
`validity`/`is_null_sensitive` (as `list_length`), `is_fallible` (delegate to body),
`serialize`/`deserialize` (body as `pb::Expr`), `fmt_sql`, `simplify_untyped` (§4.4 rules 1–2).
- [ ] Rule 4.4(3) in `ListLength::simplify_untyped`.
- [ ] Constructor in `expr/exprs.rs`; registration in `scalar_fn/session.rs`; decl in `fns/mod.rs`.
- [ ] Tests mirroring `list_length.rs`: rstest over offset widths; nullable lists; null elements;
empty lists; ListView (incl. overlapping views); FSL (structure preserved); sliced + taken
inputs (offset rebase); constant input; fallible body over a list with null-list garbage
ranges; nested `list_transform` over `List<List<i32>>`; fusion/identity rewrites via
`display_tree` snapshots; serde round-trip through proto.
- [ ] DuckDB converter arm (§6.1) + sqllogictest mirroring
`vortex-sqllogictest/slt/datafusion/list_length_pushdown.slt` for the DuckDB path.
- [ ] Bench in `vortex-file/benches/` (transform-of-projection vs full decode).

---

## Appendix: sources

- DuckDB: PRs #3694 (original lambdas),
#8851 (n-ary + index, `bind_lambda` callback),
#9395 (vectorized revamp, ~10³× speedups),
#9909 (`list_reduce`),
#17235 (`lambda x:` syntax, `->` deprecation);
blog: Friendly Lists and Their Buddies, the Lambdas;
source: `src/planner/binder/expression/bind_lambda.cpp`,
`extension/core_functions/lambda_functions.cpp` @ `36c64ee8`.
- ClickHouse: `ColumnFunction::replicate` / `arrayMap` in
`src/Functions/array/FunctionArrayMapped.h`.
- Spark: `org.apache.spark.sql.catalyst.expressions.higherOrderFunctions.scala`
(`HigherOrderFunction`, `LambdaFunction`, `NamedLambdaVariable`, `ArrayTransform`).
- Velox: lambda functions doc.
- Polars: `list.eval` / `pl.element()`.
- Vortex: `vortex-array/src/scalar_fn/{vtable.rs,fns/list_length.rs}`,
`vortex-array/src/expr/{expression.rs,proto.rs,exprs.rs}`,
`vortex-layout/src/layouts/fixed_size_list/reader.rs`,
`vortex-layout/src/layouts/list/list-storage-formats.md`,
`vortex-datafusion/src/convert/exprs.rs`, `vortex-duckdb/src/convert/expr.rs`.
