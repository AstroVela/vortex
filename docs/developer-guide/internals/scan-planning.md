# Expression Pushdown in Scan Plans

:::{note}
This is a provisional design, not a description of the current scan planner.
:::

## Expression plan

Add a physical plan that applies an expression to the output of another scan plan:

```rust
pub struct ExpressionPlan {
    expression: Expression,
    child: PlanRef,
}
```

`ExpressionPlan` inherits its row domain from `child` and derives its output dtype from
`expression`. Any part of the expression that cannot be pushed down remains in this plan and is
evaluated over the child's result.

Scalar functions remain nodes in `Expression`; they do not each become physical scan-plan nodes.
Instead, pushdown behavior is supplied by pluggable kernels registered for a scalar function (or
other expression node) and a concrete scan-plan type.

Composite scan plans expose ordered logical children to this optimizer:

```text
StructPlan.children = [field(0), ..., field(n - 1), validity?]
DictPlan.children   = [codes, values]
```

## Pushdown

Optimizing `ExpressionPlan(expression, child)` has three phases.

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
   `is_null($)` becomes `not($)`. The lowered group is installed as an `ExpressionPlan` over
   that child, and pushdown then continues recursively. Every rule must prove equivalence across
   the plan boundary; lowering is not a generic replacement of `$`.

The combination expression is retained above the grouped child plans. Missing kernels or failed
lowering leave the affected expression at the current level, preserving the generic execution
fallback.

## Examples

`@name` denotes a reference from the combination expression to a grouped child result.

### Struct without validity

```text
plan = StructPlan.children = [a, b]
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
`b` directly, and evaluate only the multiplication above the grouped results. The `+` needs no
struct-specific rule: it is rebuilt after its `get_item` child is lowered.

### Struct with validity

```text
plan = StructPlan.children = [a, b, validity]
expr = is_not_null($) && (get_item($, "a") > 0)

annotations:
  is_not_null($)        -> {validity}
  get_item($, "a")      -> {a, validity}
  get_item($, "a") > 0 -> {a, validity}
  expr                  -> {a, validity}

groups before lowering:
  @valid = is_not_null($)

groups after lowering:
  @valid = $

combine = @valid && (get_item($, "a") > 0)
```

The direct `is_not_null($) -> $` lowering is valid because `$` on the validity child is exactly the
top-level struct validity. The analogous direct lowering of `get_item` is not valid:

```text
get_item(nullable_struct, "a") = mask(a, validity)

invalid lowering:
  get_item($, "a") -> $

counterexample:
  a = 7, validity = false
  get_item(nullable_struct, "a") = null
  lowered result                 = 7
```

The singleton rule prevents this mistake: `get_item($, "a")` is annotated with `{a, validity}` and
therefore remains above the struct. A stronger validity-aware rule may lower a strict function by
factoring out the mask:

```text
(get_item($, "a") + 1)
  = mask(a, validity) + 1
  = mask(a + 1, validity)

groups after lowering:
  @a     = $ + 1
  @valid = $

combine = mask(@a, @valid)
```

This requires `+` to be strict. It must also be infallible, or execute under `@valid`, so that
evaluating values hidden by the mask cannot introduce a new error. Non-strict functions require
different combination expressions:

```text
is_not_null(get_item($, "a"))
  = @valid && is_not_null(@a)

is_null(get_item($, "a"))
  = !@valid || is_null(@a)
```

Lowering through validity is therefore a proven factorization into child expressions and a
residual combination expression, not just scope substitution.

### Dictionary

```text
plan = DictPlan.children = [codes, values]
expr = byte_length($)

annotations:
  byte_length($) -> {values}

groups before lowering:
  @values = byte_length($)

groups after lowering:
  @values = byte_length($)

combine = $

result = DictPlan.children = [
  codes,
  ExpressionPlan(byte_length($), values),
]
```

This is valid for the same strict, infallible, negative-cost functions accepted by the current
dictionary pushdown. The dictionary plan reuses `codes` and applies the function once to
`values`.

## Reader materialization

Plans do not own sources, sessions, or readers. A layout first creates a runtime-independent plan;
only the optimized plan creates the reader tree:

```rust
let plan = layout.new_plan()?;
let plan = ExpressionPlan::new_ref(expression, plan)?.optimize()?;
let reader = plan.new_reader(name, segment_source, session, ctx)?;
```

Each surviving plan node creates its corresponding reader node. Composite readers are constructed
from their optimized child plans' readers, never by reopening the original layout children. If
optimization replaces a `StructPlan` with one of its field plans, materialization therefore creates
the field reader directly and no stale struct reader remains.

## Future work

The same optimizer may eventually support `Plan x Plan -> Plan` transforms. Those
plan-to-plan rewrites are outside the scope of this proposal; this design covers only expression
and scalar-function pushdown through scan plans.
