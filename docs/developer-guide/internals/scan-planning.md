# Expression Pushdown in Scan Plans

:::{note}
This is a provisional design, not a description of the current scan planner.
:::

## Expression scan plan

Add a physical plan that applies an expression to the output of another scan plan:

```rust
pub struct ExpressionScanPlan {
    expression: Expression,
    child: ScanPlanRef,
}
```

`ExpressionScanPlan` inherits its row domain from `child` and derives its output dtype from
`expression`. Any part of the expression that cannot be pushed down remains in this plan and is
evaluated over the child's result.

Scalar functions remain nodes in `Expression`; they do not each become physical scan-plan nodes.
Instead, pushdown behavior is supplied by pluggable kernels registered for a scalar function (or
other expression node) and a concrete scan-plan type.

Composite scan plans expose ordered logical children to this optimizer:

```text
StructScanPlan.children = [field(0), ..., field(n - 1), validity?]
DictScanPlan.children   = [codes, values]
```

## Pushdown

Optimizing `ExpressionScanPlan(expression, child)` has three phases.

1. **Annotate dependencies.** Walk the expression and annotate every node with the indices of the
   immediate scan-plan children needed to evaluate it. A registered
   `(expression node, scan plan) -> [child index]` kernel provides plan-specific dependencies;
   ordinary scalar functions can otherwise take the union of their expression children's
   annotations. For a struct, `get_item($, "a")` needs field `a` and, when present, the struct
   validity child. `is_not_null($)` and `is_null($)` need only the top-level validity child.

2. **Partition and group.** Only a subexpression annotated with exactly one scan-plan child is
   eligible for pushdown. Cut maximal eligible subexpressions, group all cuts for the same child,
   and build one packed expression for that child. Replace the cuts in the remaining expression
   with references to the group results; this remaining expression is the combination expression.
   Expressions needing zero or multiple children stay above the current plan. This follows the
   grouping model of the existing expression partitioner and ensures each child group is evaluated
   once.

3. **Lower into each child.** Rewrite every root reference in a group from the current plan's scope
   into the selected child's scope. This uses a second pluggable
   `(expression node, scan plan, child index) -> expression` kernel. For a non-nullable struct,
   `get_item($, "a")` becomes `$` when lowering into field `a`; targeting another field rejects that
   pushdown. When lowering into a struct's validity child, `is_not_null($)` becomes `$` and
   `is_null($)` becomes `not($)`. The lowered group is installed as an `ExpressionScanPlan` over
   that child, and pushdown then continues recursively.

The combination expression is retained above the grouped child plans. Missing kernels or failed
lowering leave the affected expression at the current level, preserving the generic execution
fallback.

## Examples

`@name` denotes a reference from the combination expression to a grouped child result.

### Struct without validity

```text
plan = StructScanPlan.children = [a, b]
expr = (get_item($, "a") + 1) * get_item($, "b")

annotations:
  get_item($, "a") + 1 -> {a}
  get_item($, "b")     -> {b}
  expr                  -> {a, b}

groups before lowering:
  @a = get_item($, "a") + 1
  @b = get_item($, "b")

groups after lowering:
  @a = $ + 1
  @b = $

combine = @a * @b
```

The current non-nullable struct shape can therefore install `$ + 1` over child `a`, read child
`b` directly, and evaluate only the multiplication above the grouped results.

### Struct with validity

```text
plan = StructScanPlan.children = [a, b, validity]
expr = is_not_null($) && (get_item($, "a") > 0)

annotations:
  is_not_null($)        -> {validity}
  get_item($, "a")     -> {a, validity}
  get_item($, "a") > 0 -> {a, validity}
  expr                  -> {a, validity}

groups before lowering:
  @valid = is_not_null($)

groups after lowering:
  @valid = $

combine = @valid && (get_item($, "a") > 0)
```

`is_not_null($)` moves to the validity child. `get_item($, "a")` remains above the struct because
masking the field requires both `a` and `validity`.

### Dictionary

```text
plan = DictScanPlan.children = [codes, values]
expr = byte_length($)

annotations:
  byte_length($) -> {values}

groups before lowering:
  @values = byte_length($)

groups after lowering:
  @values = byte_length($)

combine = $

result = DictScanPlan.children = [
  codes,
  ExpressionScanPlan(byte_length($), values),
]
```

This is valid for the same strict, infallible, negative-cost functions accepted by the current
dictionary pushdown. The dictionary plan reuses `codes` and applies the function once to
`values`.

## Future work

The same optimizer may eventually support `ScanPlan x ScanPlan -> ScanPlan` transforms. Those
plan-to-plan rewrites are outside the scope of this proposal; this design covers only expression
and scalar-function pushdown through scan plans.
