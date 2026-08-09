<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn investigation handoff

This file records the exact state at the end of the 2026-08-09 investigation. Start with this
file, then read [`DESIGN.md`](DESIGN.md), [`OPTIMIZATION.md`](OPTIMIZATION.md), and
[`REPRODUCE.md`](REPRODUCE.md).

## Branch state

- Branch: `ct/row-fn`.
- Code head before this documentation commit: `4c936447a`.
- Comparison revision: develop at `66d096b5d`.
- No RowFn code changed during this final investigation.
- The only intended new branch changes are the files in `research/rowfn-reconstruction`.

Two temporary remote refs exist for a future CodSpeed ablation:

- `ct/row-fn-codspeed-framework` points to `0a0ad0db1`.
- `ct/row-fn-codspeed-numeric` points to `89fd28bc1`.

The refs contain exact historical code. They do not contain the uncommitted focused-workflow edits
that were made only in temporary local worktrees.

## Corrected CodSpeed history

The latest push did not bring back the `take_filter_list_*` regressions.

- The [CodSpeed check at `892717f30`] already reports the cases as about 15% to 16% slower.
- The [CodSpeed check at `4c936447a`] reports the same cases as about 14% to 16% slower.
- Most take/filter simulated times improve by less than 2% between those checks.
- `4c936447a` fixes the much larger constant add, subtract, and multiply regressions. This moves the
  persistent take/filter entries higher in the ordered list of the 20 largest changes.
- Every retained RowFn CodSpeed summary from `0e5c19c00` through `4c936447a` that has a performance
  table also contains take/filter regressions.

The PR bot edits one current comment, and GitHub displays only the 20 largest changes. These two
details can make a persistent regression appear to leave and return.

## What is known about take/filter

The list, filter, and take source files are identical between develop and `4c936447a`. The
`take_filter_list` benchmark does not execute a RowFn operation.

The linked AVX2 benchmark binaries still differ. Native inspection found:

- The main filter-take function has the same `0x41cc` byte size on develop, `892717f30`, and
  `4c936447a`.
- The main list `TakeExecute::take` function has the same `0x40ac` byte size.
- Normalized list-take disassembly has the same instructions.
- Function addresses, relative call targets, and linked layout differ.

This evidence is consistent with a linked-layout effect or a changed callee outside the inspected
symbol. It does not prove which cache, branch, or callee causes the result.

CodSpeed documents [function alignment] as a reason unchanged microbenchmarks can move after a
rebuild. Its differential flame graph is the correct next source of evidence. Inspect the
instruction, cache, and memory components separately.

## Native measurements are separate evidence

Pinned AVX2 wall-time runs on an AMD Ryzen 9 7950X found both `892717f30` and `4c936447a` about 25%
to 31% slower than develop for the tested take/filter list cases. The final push changes those
native medians by only 0% to 2%.

Changing the bench profile from 16 codegen units to one did not remove that native gap. One
representative median pair was:

| Profile | `4c936447a` | Develop |
| --- | ---: | ---: |
| 16 codegen units | 8.25 us | 6.41 us |
| One codegen unit | 7.86 us | 6.21 us |

These measurements do not explain the CodSpeed simulation result. Do not use local wall time as a
proxy for CodSpeed CPU simulation.

## Incomplete CodSpeed ablation

Two `workflow_dispatch` runs were started and then canceled:

- Framework only: [run `31289620637`].
- Numeric RowFn: [run `31289622392`].

This approach was not sufficient. A workflow-dispatch run has no pull-request context, so it does
not update PR #9255's comment or create the PR comparison check needed for an inspectable result.
The framework array shard also reached an unrelated cancellation in
`take_slices_to_buffer_matrix`. Do not use either run as performance evidence.

## Recommended next steps

1. Open one affected `take_filter_list_*` benchmark in the existing `4c936447a` CodSpeed check.
2. Compare its differential flame graph with develop. Record executed instruction, cache, and
   memory costs for the changed stack.
3. If the cost is extra instructions or a changed call path, follow that stack into assembly and
   source.
4. If the cost is only instruction-cache placement, do not add arbitrary padding or unrelated
   source edits. Determine whether a stable alignment or build-level remedy exists.
5. To locate the first bad revision, run focused `take_filter` simulations for `0a0ad0db1` and
   `89fd28bc1` in a pull-request context. A dedicated temporary PR is less disruptive than moving
   the head of PR #9255. Run only `cargo codspeed run --bench take_filter`.
6. If framework-only is clean and numeric RowFn is bad, compare those two profiles. If both are
   clean, continue through `5c02036a2`, `a236e0b9d`, and `f4617a2b5`.
7. Recheck native wall time only after finding a CodSpeed cause. Keep the two result types labeled
   separately.

## Mixed-constant optimization

Keep `4c936447a`. It fixes a real RowFn regression.

For two varying inputs, `Args::varying` returns typed slices and selects the indexed lane source.
For an array plus a constant, one argument returns `None`, so the tuple returns `None`. Here,
`None` means "not every input varies," not "no input varies." The mixed loop reads the array at
`index` and the one-row constant at zero.

The measured compiler requires the varying match and its length proof to remain inside the selected
owned-executor branch. Moving the proof through one shared `Option` helper made constant add and
subtract about 3.3 times slower. The branch-local form restored them. The semantic reason for the
source-placement sensitivity remains unknown.

[CodSpeed check at `892717f30`]: https://github.com/vortex-data/vortex/runs/93169735961
[CodSpeed check at `4c936447a`]: https://github.com/vortex-data/vortex/runs/93181527671
[function alignment]: https://codspeed.io/docs/instruments/cpu/regression-causes#function-alignment
[run `31289620637`]: https://github.com/vortex-data/vortex/actions/runs/31289620637
[run `31289622392`]: https://github.com/vortex-data/vortex/actions/runs/31289622392
