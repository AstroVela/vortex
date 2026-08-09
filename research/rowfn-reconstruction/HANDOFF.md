<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# RowFn investigation handoff

This file records the current state of the 2026-08-09 investigation. Start with this file, then
read [`DESIGN.md`](DESIGN.md), [`OPTIMIZATION.md`](OPTIMIZATION.md), and
[`REPRODUCE.md`](REPRODUCE.md).

## Branch state

- Branch: `ct/row-fn`.
- Last RowFn code commit: `4c936447a`.
- Documentation head before the offsets fix: `bdf95a77e`.
- Comparison revision: develop at `66d096b5d`.
- The offsets fix does not change the RowFn API or implementation.

Three temporary remote refs exist for the CodSpeed ablation:

- `ct/row-fn-codspeed-framework` points to `0a0ad0db1`.
- `ct/row-fn-codspeed-numeric` points to `89fd28bc1`.
- `ct/row-fn-codspeed-take-filter` is the head of temporary draft PR #9298.

The first two refs contain exact historical code. The third ref adds a PR-only workflow that runs
only `cargo codspeed run --bench take_filter`.

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

## Verified take/filter cause

The list, filter, and take source files are identical between develop and `4c936447a`. The
benchmark still reaches RowFn through an indirect call:

```text
take_filter
  -> list_view_from_list
    -> ListArrayExt::reset_offsets
      -> binary(Sub) on offsets and the first offset
        -> numeric RowFn
```

The differential profile therefore corrects the earlier claim that the benchmark does not execute
RowFn. `reset_offsets` creates a constant array and runs generic numeric subtraction. Numeric RowFn
adds batch planning, dispatch, argument decoding, and output reconciliation to this small operation.

The representative benchmark is
`take_filter_list_small_uncached_random_mask_random_indices[256, 10]`. The current PR report gives
233.737 microseconds for develop and 280.793 microseconds for `bdf95a77e`. This is a 16.76%
regression.

CodSpeed creates the downloadable callgraph in a separate profiling execution. Its total can
differ slightly from the aggregate report. The callgraph components are:

| Revision | Instructions | Cache | Memory | Total |
| --- | ---: | ---: | ---: | ---: |
| Develop `66d096b5d` | 21.312 us | 83.443 us | 133.294 us | 238.050 us |
| RowFn `bdf95a77e` | 26.210 us | 104.293 us | 155.172 us | 285.675 us |
| Increase | 4.898 us | 20.850 us | 21.878 us | 47.626 us |

The extra instructions and the changed stack rule out a cache-only layout explanation. Cache and
memory costs also increase, but they occur on newly executed RowFn work.

The largest changed functions in the focused numeric profile are:

| Function | Base self / total | Head self / total |
| --- | ---: | ---: |
| Old `execute_numeric_primitive` | 0.741 / 18.639 us | absent |
| RowFn `execute_numeric_primitive` | absent | 0.430 / 71.156 us |
| `Batch::execute` | absent | 1.033 / 49.972 us |
| `Batch::execute_dense` | absent | 0.634 / 45.781 us |
| `NumericBinary::dispatch` | absent | 1.316 / 45.736 us |
| `(A, B)::decode` | absent | 0.539 / 37.501 us |
| `ArgColumn<T>::decode` | absent | 0.968 / 36.254 us |
| `list_view_from_list` | 3.543 / 79.144 us | 2.592 / 108.951 us |
| `Batch::new` | absent | 1.797 / 10.794 us |

These totals are inclusive callgraph costs. A function can appear in more than one caller stack.

The linked AVX2 benchmark binaries still differ. Native inspection found:

- The main filter-take function has the same `0x41cc` byte size on develop, `892717f30`, and
  `4c936447a`.
- The main list `TakeExecute::take` function has the same `0x40ac` byte size.
- Normalized list-take disassembly has the same instructions.
- Function addresses, relative call targets, and linked layout differ.

That native inspection covered the large take and filter functions. It missed the changed numeric
callee reached during list offset normalization.

CodSpeed documents [function alignment] as a reason unchanged microbenchmarks can move after a
rebuild. That warning remains useful, but alignment is not the cause of this simulation regression.

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

## Focused CodSpeed ablation

Two `workflow_dispatch` runs were started and then canceled:

- Framework only: [run `31289620637`].
- Numeric RowFn: [run `31289622392`].

This approach was not sufficient. A workflow-dispatch run has no pull-request context, so it does
not create the needed comparison. Do not use either run as performance evidence.

Draft PR [#9298] provides the required pull-request context. Its workflow builds and runs only the
`take_filter` benchmark.

- [Focused framework check] at `0a0ad0db1`: 232.542 microseconds against 233.737 microseconds for
  develop. This is a 0.51% improvement and CodSpeed classifies it as no change.
- [Focused numeric check] at `89fd28bc1`: 279.491 microseconds against 233.737 microseconds for
  develop. This is a 16.37% regression.

`89fd28bc1` is the first bad revision. It is the direct child of clean revision `0a0ad0db1`.

The numeric revision's callgraph totals are 25.835 microseconds for instructions, 103.531
microseconds for cache, and 154.728 microseconds for memory. Develop's totals are 21.312, 83.443,
and 133.294 microseconds. The total increases from 238.050 to 284.093 microseconds.

## Focused fix

`ListArrayExt::reset_offsets` now decodes offsets once and subtracts the first offset in a typed
loop. It no longer allocates a constant array or invokes the generic scalar-function path.

The AVX2 release binary auto-vectorizes the benchmark's `u16` loop. The loop uses two packed
`psubw` instructions per iteration and processes 16 offsets. This is code-generation evidence,
not a local timing result.

A new test covers nonzero `u16` offsets. The existing list and list-view tests cover other offset
types and conversion behavior. A push to PR #9255 is still required for CodSpeed validation.

## Recommended next steps

1. Push the focused offsets fix to `ct/row-fn` so PR #9255 creates a CodSpeed comparison.
2. Verify the representative benchmark's report value and callgraph components.
3. Check the remaining `take_filter_list_*` cases for a consistent recovery.
4. Keep local wall time separate from CodSpeed CPU simulation.
5. Continue investigating the native wall-time gap only if it remains after the measured call path
   is removed.

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
[#9298]: https://github.com/vortex-data/vortex/pull/9298
[Focused framework check]: https://github.com/vortex-data/vortex/actions/runs/31316492455
[Focused numeric check]: https://github.com/vortex-data/vortex/actions/runs/31316710479
[run `31289620637`]: https://github.com/vortex-data/vortex/actions/runs/31289620637
[run `31289622392`]: https://github.com/vortex-data/vortex/actions/runs/31289622392
