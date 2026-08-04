# Layout Scan Plans

A layout scan plan is the physical plan for satisfying one scan query from a layout tree. It
describes the layout nodes and derived operations needed to produce that query's result.

The stored layout tree describes all physical data in a file. A scan plan is query-specific: it is
built from that tree for one projection, filter, and row domain. Different queries over the same
file can therefore produce different plans.

For example, a plan can:

- select only the referenced fields of a struct layout;
- select only chunks that overlap the requested rows;
- evaluate an expression against dictionary values while retaining the codes layout; and
- generate row-index values without reading an unrelated data layout.

Plan optimization rewrites the initial tree so that each expression is evaluated as close as
possible to the physical layout that can satisfy it. The optimized plan should retain only the
layout reads and derived work required by the query.

Planning does not read segment data. It constructs and optimizes a description of the work that a
later execution stage will perform. A plan is therefore neither the stored layout itself nor a
general logical query plan.

Every rewrite must preserve the query result, including its dtype, row domain, row order, row
identity, null behavior, and observable errors.

## Future execution

Plans currently stop at construction and optimization. A future PR will add a method for executing
an optimized plan. That method will walk the physical plan, read the referenced layout data,
evaluate its expressions, and return the result of the query. The execution API and return type
will be defined as part of that integration rather than fixed by the planning IR today.
