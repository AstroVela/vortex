# Row-index plan domain handover

## Context

This note captures the unresolved row-domain question found while reviewing the physical-plan
row-index work.

- Base branch: `vortex-plan-rules` at `5361eb1858` (`Add plan optimizer rules and push
  expressions`).
- Execution branch inspected: `vortex-plan-scan-v2` at `a4d7778cbd` (`Add plan-native scan
  execution`).
- No production code was changed as part of this handover.

## Current model

On `vortex-plan-rules`, `RowIdxPlan::new` takes only a row count. `RowIdxData` is empty, so the
plan node does not contain an absolute or root-relative offset:

```rust
pub fn new(row_count: u64) -> Self
```

The stacked execution branch supplies the root offset through `PlanExecutionContext`. A
`RowIdxPlan` produces:

```text
execution_context.row_offset + requested_row_range.start + [0, 1, ...]
```

`ConcatPlan` derives cumulative child offsets from the child row counts. At execution, it creates
a child context using `parent_context.row_offset + chunk_offset`. Other same-domain operators such
as `EvalPlan` and `PackPlan` pass the context and row range through unchanged.

This means `RowIdxPlan::new(2)` does **not** intrinsically mean `[0, 1]`. It means "two row-index
values in whichever execution domain contains this node."

## Existing coverage

The stacked `vortex-plan-scan-v2` branch contains these tests in
`vortex-layout/src/plan/tests.rs`:

- `row_idx_source_adds_the_execution_range_start` executes a six-row source with root offset 100
  over range `2..5` and expects `[102, 103, 104]`.
- `row_idx_source_uses_concat_child_row_domains` constructs
  `Concat(RowIdx(rows=2), RowIdx(rows=3))`, executes root range `1..4` with root offset 100, and
  expects `[101, 102, 103]`.

The second test confirms that `ConcatPlan` rebases context-relative children. It does not confirm
that concatenating independently rooted plans preserves their previous row indices.

## Unresolved semantic choice

The current implementation is composable only if a `PlanRef` is explicitly defined as relative
to the execution domain supplied by its parent.

Under that definition:

- `Pack(left, right)` places both children in the same row domain, so both observe the same row
  offset.
- `Concat(first, second)` creates a new contiguous row domain. `first` starts at the parent's
  offset and `second` starts at `parent offset + first.row_count()`.
- Reusing the same `RowIdxPlan` under two different parents can deliberately produce different
  values.

That definition does not support combining two plans whose existing absolute origins must be
preserved. For example, plans rooted at offsets 100 and 1,000 cannot be represented faithfully by
one `ConcatPlan` plus a single execution-context offset. `ConcatPlan` will rebase the second plan
immediately after the first.

This matters because context-relative row indices make plan output depend on placement, not only
on the `PlanRef`. Optimizer rules, result caching, DAG reuse, and any future arbitrary plan
composition must preserve or account for the row-domain mapping.

## Decision required

Choose and document one of these contracts before treating the existing concat test as complete
coverage:

1. **Context-relative plans.** `ConcatPlan` intentionally creates a fresh contiguous row domain.
   Independently rooted plans must be explicitly rebased before composition, and caches cannot be
   keyed only by `PlanRef` when a plan contains `RowIdxPlan`.
2. **Source-origin-preserving plans.** A plan or wrapper must carry its row origin. Concatenation
   must preserve or validate per-child origins rather than deriving every origin solely from child
   lengths. This likely needs an explicit `RowOffset`/`Rebase` plan operator or an equivalent
   row-domain descriptor.

The existing scan behavior naturally fits option 1 for chunks belonging to one file. Option 2 is
needed if public plan composition is expected to retain the identities of independently rooted
inputs.

## Suggested follow-up tests

After deciding the contract, add tests for these cases:

1. Reuse the same `RowIdxPlan` twice under `ConcatPlan` and assert either deliberate rebasing or
   preserved origins.
2. Put two `RowIdxPlan` children side by side under `PackPlan` and confirm they observe the same
   domain.
3. Exercise a row-index expression before and after an optimizer rewrite that moves work across a
   `ConcatPlan` boundary.
4. Cover independently rooted inputs. Either verify an explicit origin-preserving operator or
   verify that unsupported composition is rejected rather than silently renumbered.
5. If results are cached, execute one shared `RowIdxPlan` in two different domains and ensure the
   cache key includes the execution domain.

## Relevant implementation locations

- `vortex-layout/src/plan/plans/row_idx.rs`: context-relative row-index source construction and
  expression partitioning.
- `vortex-layout/src/plan/plans/concat.rs` on `vortex-plan-scan-v2`: cumulative child offsets and
  `child_row_domain` propagation.
- `vortex-layout/src/plan/execution.rs` on `vortex-plan-scan-v2`: root row offset and child-domain
  derivation.
- `vortex-scan-v2/src/scan_builder.rs` on `vortex-plan-scan-v2`: scan-level root-offset setup.
- `vortex-layout/src/layout.rs`: `LayoutChildType` distinguishes row-preserving, chunk, and
  auxiliary child relationships.
