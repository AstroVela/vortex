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

## Execution

Each plan node can execute a row range and selection mask. Leaf plans read their referenced
segments, structural plans combine their children, and expression plans evaluate the remaining
derived work. The separate `vortex-scan-v2` crate copies the existing scan orchestration around
this API so the original `LayoutReader` scanner remains unchanged while the plan-native path is
developed.
