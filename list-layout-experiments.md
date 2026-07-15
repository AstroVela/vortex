# List-layout performance experiments

Running journal for `mk/list-layout-refactor` in the `list-whole-chunk` worktree.

## Benchmark conventions

- Dataset: StatPopGen scale factor 100 (`100,000` outer rows).
- Engine/format: DuckDB over `vortex-file-compressed`.
- List layout: `VORTEX_EXPERIMENTAL_LIST_LAYOUT=1` unless marked “no list”.
- Builds do **not** enable `unstable_encodings`.
- Reported full-suite values are medians of 20 iterations.
- The local q01 is `SELECT array_length("GT") FROM statpopgen LIMIT 1`; it differs from the
  original benchmark query, so local q01 values are not directly comparable with the original CI
  q01.
- Benchmark measurements are run serially, not concurrently.

## Stable no-list control

Fresh back-to-back control from 2026-07-14:

| Query | Median (ms) |
| --- | ---: |
| q00 | 5.734 |
| q01 | 17.742 |
| q02 | 167.701 |
| q03 | 436.722 |
| q04 | 445.957 |
| q05 | 207.088 |
| q06 | 705.961 |
| q07 | 57.645 |
| q08 | 63.045 |
| q09 | 324.962 |
| q10 | 1128.000 |

Artifacts:

- Results: `/private/tmp/statpopgen-all-list-off-back-to-back-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-list-off.vortex` (`1,942,718,052` bytes)

## Experiment history

### 1. Initial whole-column list layout

Configuration:

- One structural list layout spans the entire outer column.
- Elements go through the ordinary leaf pipeline with its approximately 1 MiB pre-compression
  repartition target.
- The list reader originally did not provide useful element-derived root-row scan boundaries.

Observed in CI:

- Overall reported regression: `6.234x`.
- q01: `49.15x`; q02: `6.74x`; q03–q10 were generally `4.7x–8.4x` slower.

Learning:

- A whole-column list changes the scan/task and compression shape substantially. The initial
  reader path also did too much element work for selective queries.

Status: superseded by reader fixes and later splitting experiments.

### 2. Remove redundant chunked-reader check

Configuration:

- Treat the elements layout as valid by construction instead of requiring a chunked-layout check
  before taking the bounded reader path.

Learning:

- The construction-time invariant is sufficient; the runtime check prevented the intended path
  for otherwise valid layouts.

Status: retained in the branch history; committed and pushed before later experiments.

### 3. Store GT values as `u8` instead of `u64`

Hypothesis:

- GT values are only 0, 1, or 2. Building them as `u64` might steer compression toward a poor
  encoding and cause extreme compression ratios/fragmentation.

Outcome:

- Changing the builder type did not materially improve the benchmark.

Learning:

- Logical input width was not the main cause. The compressor already represents the tiny value
  domain compactly; scan partitioning and physical chunk count dominated.

Status: reverted.

### 4. Await the outer-row mask and expand it into element space

Configuration:

- Resolve the row mask with offsets.
- Crop leading/trailing unselected rows.
- Convert selected list ranges into an elements-space mask and pass it to the elements reader.

Learning:

- This avoids materializing elements belonging to unselected lists and is necessary for selective
  list reads, but it does not solve poor global scan partitioning by itself.

Status: active reader change.

### 5. Register every 1 MiB element-chunk boundary in outer-row space

Configuration:

- Keep ordinary approximately 1 MiB pre-compression element repartitioning.
- Translate every physical element chunk endpoint through the global list offsets and register it
  as an outer-row boundary.

Selected results:

| Query | Median (ms) |
| --- | ---: |
| q02 | 120.9 |
| q07 | ~1820 |
| q08 | ~1820 |

Shape:

- Roughly 3,100 scan tasks/boundaries for GT-scale data.

Learning:

- Fine element alignment helps the dense GT projection, but raw boundary registration explodes
  task count and is disastrous for selective, wide `SELECT *` queries.

Artifacts:

- Results: `/private/tmp/statpopgen-all-element-splits-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-element-splits-raw.vortex`

Status: superseded.

### 6. Coalesce mapped boundaries in outer-row space

Configuration:

- Start from mapped element boundaries, then coalesce them into larger outer-row ranges.

Selected results:

| Query | Median (ms) |
| --- | ---: |
| q02 | 148.0 |
| q07 | 78.5 |
| q08 | 81.0 |

Shape:

- 64 all-field scan ranges.

Learning:

- Bounding global task count recovers most selective-query performance while retaining some dense
  projection benefit. This was the first strong evidence that element boundaries should be inputs
  to a global/coalescing policy rather than mandatory splits.

Artifacts:

- Results: `/private/tmp/statpopgen-all-element-splits-coalesced-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-element-splits-coalesced.vortex`

Status: superseded.

### 7. Use a shared 8,192-row outer grid

Configuration:

- Persist the same `8,192, 16,384, ...` outer-row boundaries for every list column that has
  physical element chunks.
- Element chunks remain on the ordinary approximately 1 MiB pre-compression target.

Results versus its matched no-list run:

| Query | No list (ms) | Shared grid (ms) | Ratio |
| --- | ---: | ---: | ---: |
| q00 | 5.778 | 5.883 | 1.02x |
| q01 | 18.736 | 46.188 | 2.47x |
| q02 | 179.535 | 155.203 | 0.86x |
| q03 | 442.440 | 447.588 | 1.01x |
| q04 | 449.191 | 446.721 | 0.99x |
| q05 | 206.754 | 200.664 | 0.97x |
| q06 | 706.302 | 698.290 | 0.99x |
| q07 | 59.587 | 61.284 | 1.03x |
| q08 | 64.925 | 62.898 | 0.97x |
| q09 | 335.014 | 307.367 | 0.92x |
| q10 | 1124.042 | 1144.094 | 1.02x |

Shape and size:

- 13 all-field ranges.
- File size: `1,752,293,052` bytes.

Learning:

- A common grid avoids cross-column split-union explosion and makes q07/q08 healthy. It leaves a
  single task responsible for many element chunks, however, and q01 still exposed reader overhead.

Artifacts:

- Results: `/private/tmp/statpopgen-all-list-shared-grid-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-list-shared-grid.vortex`

Status: superseded.

### 8. Use 10 MiB element chunks without element-derived splits

Configuration:

- Increase only the list-elements pre-compression repartition target from 1 MiB to 10 MiB.
- Do not add mapped element boundaries to the scan split set.

Focused outcome:

- Logical element requests fell from roughly 3,169 to 323.
- Physical reads fell from 64 to 51.
- The focused q02 measurement moved from about 1071.6 ms to 1145.6 ms (`+6.9%`) in that diagnostic
  setup.

Learning:

- Larger chunks reduce fragmentation, but this isolated run remained under-split and therefore did
  not test the useful combination of larger physical chunks plus aligned scan work.

Status: superseded.

### 9. Use 10 MiB element chunks and register every mapped boundary

Configuration:

- Ordinary non-list columns keep the 1 MiB target.
- Leaves below list elements use a 10 MiB pre-compression target.
- Every physical element endpoint is mapped through offsets and persisted as an outer-row split.

Results versus the fresh back-to-back no-list control:

| Query | No list (ms) | 10 MiB + mapped (ms) | Ratio |
| --- | ---: | ---: | ---: |
| q00 | 5.734 | 5.830 | 1.02x |
| q01 | 17.742 | 7.361 | 0.41x |
| q02 | 167.701 | 100.740 | 0.60x |
| q03 | 436.722 | 302.344 | 0.69x |
| q04 | 445.957 | 301.132 | 0.68x |
| q05 | 207.088 | 131.842 | 0.64x |
| q06 | 705.961 | 444.219 | 0.63x |
| q07 | 57.645 | 284.648 | 4.94x |
| q08 | 63.045 | 295.968 | 4.69x |
| q09 | 324.962 | 215.325 | 0.66x |
| q10 | 1128.000 | 786.655 | 0.70x |

Shape and size:

- GT has 321 physical element chunks and 320 interior mapped boundaries.
- Compressed GT chunk sizes: 41.4 KiB minimum, 344 KiB median, 603.6 KiB mean, 1.43 MiB p90,
  2.50 MiB maximum.
- File size: `1,864,830,564` bytes.
- Overall geomean: `0.940x`; excluding q07/q08: `0.654x`.

Learning:

- Larger chunks are strongly beneficial for dense list work.
- Registering every locally natural boundary remains incompatible with the current global union:
  q07/q08 select all 1,190 columns and become hundreds of tiny, very wide tasks.
- The 10 MiB input target is not a stable physical-size target; compression produces a wide size
  distribution averaging only about 604 KiB.

Artifacts:

- Results: `/private/tmp/statpopgen-all-list-elements-10m-mapped-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-list-elements-10m-mapped.vortex`

Status: superseded; the implementation was removed before experiment 10.

### 10. Chunked outer list as the ordinary leaf strategy

Configuration:

```text
repartition_outer_rows(
  chunked(
    list(
      offsets = existing leaf strategy,
      elements = descended TableStrategy / existing leaf strategy,
      validity = shared validity strategy,
    )
  )
)
```

- Repartition the list in outer-row space into 8,192-row blocks.
- Wrap the resulting list layouts in one outer `ChunkedLayout`.
- Decompose each outer chunk independently; elements still use the ordinary leaf strategy and its
  approximately 1 MiB pre-compression chunks.
- Use only the outer `chunked` layout's natural row boundaries; do not persist translated element
  boundaries.
- Keep `vortex-file/src/strategy.rs` identical to base; the experiment is implemented in
  `TableStrategy`.

Results versus the fresh back-to-back no-list control:

| Query | No list (ms) | Chunked outer list (ms) | Ratio |
| --- | ---: | ---: | ---: |
| q00 | 5.734 | 5.866 | 1.02x |
| q01 | 17.742 | 40.596 | 2.29x |
| q02 | 167.701 | 147.233 | 0.88x |
| q03 | 436.722 | 432.776 | 0.99x |
| q04 | 445.957 | 431.264 | 0.97x |
| q05 | 207.088 | 196.598 | 0.95x |
| q06 | 705.961 | 691.215 | 0.98x |
| q07 | 57.645 | 558.407 | 9.69x |
| q08 | 63.045 | 567.476 | 9.00x |
| q09 | 324.962 | 300.424 | 0.92x |
| q10 | 1128.000 | 1137.626 | 1.01x |

Shape and size:

- GT has 13 outer chunks: twelve 8,192-row chunks and one 1,696-row chunk.
- Each outer child is a self-contained `ListLayout`.
- A full GT list chunk covers 34,004,992 elements and contains 260 ordinary physical element
  chunks, mostly 131,072 `u64` values before compression.
- File size: `1,789,091,412` bytes.
- Overall geomean: `1.576x`; excluding q07/q08: `1.061x`.

Learning:

- The initial implementation made q07/q08 9–10x slower than no-list despite having only 13 root
  splits. Experiment 11 established that this was caused by the list reader's full-local-range
  shortcut, not by root split count or element chunk count alone.
- Dense queries are otherwise close to neutral, with modest improvements on q02/q05/q09.
- A scan task exactly covers one outer list child, so that child sees `0..row_count` even when the
  task's filter mask selects only one row. Treating that range as an unconditional dense read
  fetched all roughly 34 million elements before applying the mask.

Artifacts:

- Generation smoke: `/private/tmp/statpopgen-chunked-outer-list-generate.jsonl`
- Results: `/private/tmp/statpopgen-all-chunked-outer-list-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-chunked-outer-mask-aware.vortex`

Status: superseded by experiment 12.

### 11. Mask-aware full-local-range list reads

Configuration:

- Keep experiment 10's file and layout unchanged.
- For a strict list-row subrange, continue using the bounded offsets-to-elements path.
- For a full local list range, resolve the row mask before choosing the read path:
  - all true: preserve the concurrent full-elements read;
  - selective or all false: use the bounded path and translate the mask into element space.

Focused regression test:

- Five two-element chunks plus one offsets segment are stored for a five-row list.
- A full-range mask selecting only the first row requested all six segments before the fix.
- The mask-aware implementation requests two segments: offsets and the one selected element chunk.

Results versus the same no-list control and the initial experiment 10 reader:

| Query | No list (ms) | Before (ms) | Mask-aware (ms) | Mask-aware / no list |
| --- | ---: | ---: | ---: | ---: |
| q00 | 5.734 | 5.866 | 5.912 | 1.03x |
| q01 | 17.742 | 40.596 | 41.066 | 2.31x |
| q02 | 167.701 | 147.233 | 146.622 | 0.87x |
| q03 | 436.722 | 432.776 | 431.819 | 0.99x |
| q04 | 445.957 | 431.264 | 435.785 | 0.98x |
| q05 | 207.088 | 196.598 | 192.757 | 0.93x |
| q06 | 705.961 | 691.215 | 687.741 | 0.97x |
| q07 | 57.645 | 558.407 | 87.958 | 1.53x |
| q08 | 63.045 | 567.476 | 91.718 | 1.45x |
| q09 | 324.962 | 300.424 | 312.641 | 0.96x |
| q10 | 1128.000 | 1137.626 | 1121.233 | 0.99x |

Learning:

- q07/q08 improve by 6.3–6.4x in the all-query run without changing the file or scan splits,
  confirming that the catastrophic regression was excess element reads.
- Dense-query timings remain essentially unchanged because all-true masks retain the concurrent
  full-elements path.
- Selective reads are still 45–53% slower than no-list and roughly 44–46% slower than the
  shared-grid experiment. The remaining gap is separate from the full-range mask bug.
- Overall geomean versus no-list improves from `1.576x` to `1.131x`; excluding q07/q08 it is
  `1.064x`.

Artifacts:

- Focused q07/q08: `/private/tmp/statpopgen-q07-q08-chunked-outer-mask-aware-20.jsonl`
- All queries: `/private/tmp/statpopgen-all-chunked-outer-mask-aware-20.jsonl`
- File: `/private/tmp/statpopgen-sf100-chunked-outer-mask-aware.vortex`

Status: the mask-aware reader change remains active; the outer-chunk topology is superseded.

### 12. Whole-column list with shared splits and mask-aware reader

Configuration:

```text
list(
  offsets = existing leaf strategy,
  elements = descended TableStrategy / existing leaf strategy,
  validity = shared validity strategy,
)
```

- Remove `Repartition(Chunked(ListLayout(...)))` from `TableStrategy`.
- Write one whole-column `ListLayout`; its elements and offsets retain the ordinary leaf strategy.
- Persist interior outer-row boundaries at `8,192, 16,384, ...` in `ListLayout` metadata and
  register only those shared boundaries in root row space.
- Keep experiment 11's mask-aware full-local-range reader behavior.
- Keep `vortex-file/src/strategy.rs` identical to base.

Results versus the same no-list control and experiment 11's outer-chunk layout:

| Query | No list (ms) | Outer chunk (ms) | Whole list + shared grid (ms) | Ratio |
| --- | ---: | ---: | ---: | ---: |
| q00 | 5.734 | 5.912 | 5.858 | 1.02x |
| q01 | 17.742 | 41.066 | 42.294 | 2.38x |
| q02 | 167.701 | 146.622 | 142.514 | 0.85x |
| q03 | 436.722 | 431.819 | 428.960 | 0.98x |
| q04 | 445.957 | 435.785 | 433.429 | 0.97x |
| q05 | 207.088 | 192.757 | 195.598 | 0.94x |
| q06 | 705.961 | 687.741 | 687.349 | 0.97x |
| q07 | 57.645 | 87.958 | 57.419 | 1.00x |
| q08 | 63.045 | 91.718 | 61.460 | 0.97x |
| q09 | 324.962 | 312.641 | 310.021 | 0.95x |
| q10 | 1128.000 | 1121.233 | 1092.469 | 0.97x |

Shape and size:

- GT is one `ListLayout` with 100,000 outer rows and 12 interior shared splits, producing 13 scan
  ranges.
- GT elements contain 415.1 million rows in 3,167 ordinary physical chunks.
- File size: `1,752,328,092` bytes.
- Overall geomean versus no-list: `1.045x`; excluding q07/q08: `1.059x`.

Learning:

- Removing outer list chunking recovers q07/q08 completely: both are at or slightly faster than
  the no-list control and roughly 1.5x faster than the mask-aware outer-chunk topology.
- q02–q10 are neutral or faster than no-list. q01 remains the only large ratio regression, at
  2.38x, though its absolute difference is about 24.6 ms.
- A single structural list plus a shared root-row grid is the strongest overall configuration
  tested with the ordinary element chunk target.

Artifacts:

- Generation smoke: `/private/tmp/statpopgen-whole-list-shared-grid-mask-aware-generate.jsonl`
- Focused q07/q08: `/private/tmp/statpopgen-q07-q08-whole-list-shared-grid-mask-aware-20.jsonl`
- All queries: `/private/tmp/statpopgen-all-whole-list-shared-grid-mask-aware-20.jsonl`
- File: `/private/tmp/vortex-list-element-splits/vortex-bench/data/statpopgen/100000/vortex-file-compressed/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex`

Status: completed; currently present in the worktree.

### 13. Outer-row zones before whole-column list decomposition

Configuration:

```text
repartition(outer rows, row_block_size)
  -> zoned(
       data = one whole-column list(
         offsets = existing leaf strategy,
         elements = descended TableStrategy / existing leaf strategy,
         validity = shared validity strategy,
       ),
       zones = compressed outer-list statistics,
     )
```

- Repartition lists into the configured file `row_block_size` before structural decomposition.
- Compute zone statistics on the list chunks in outer-row space, before shredding.
- Feed the complete repartitioned stream into one `ListLayout`; do not create an outer
  `ChunkedLayout`.
- Record the cumulative boundaries actually observed by `ListLayoutStrategy`, rather than
  synthesizing an 8,192-row grid inside the list writer.
- Benchmark with only `VORTEX_EXPERIMENTAL_LIST_LAYOUT=1`; unstable encodings are disabled.

Results versus the same no-list control and experiment 12's unzoned whole list:

| Query | No list (ms) | Unzoned whole list (ms) | Outer-zoned whole list (ms) | Zoned / no list |
| --- | ---: | ---: | ---: | ---: |
| q00 | 5.734 | 5.858 | 5.619 | 0.98x |
| q01 | 17.742 | 42.294 | 21.934 | 1.24x |
| q02 | 167.701 | 142.514 | 145.523 | 0.87x |
| q03 | 436.722 | 428.960 | 434.275 | 0.99x |
| q04 | 445.957 | 433.429 | 430.618 | 0.97x |
| q05 | 207.088 | 195.598 | 195.020 | 0.94x |
| q06 | 705.961 | 687.349 | 688.551 | 0.98x |
| q07 | 57.645 | 57.419 | 59.173 | 1.03x |
| q08 | 63.045 | 61.460 | 63.045 | 1.00x |
| q09 | 324.962 | 310.021 | 311.964 | 0.96x |
| q10 | 1128.000 | 1092.469 | 1122.582 | 1.00x |

Focused selective-query run:

| Query | Outer-zoned whole list (ms) | Zoned / no list |
| --- | ---: | ---: |
| q07 | 56.596 | 0.98x |
| q08 | 60.554 | 0.96x |

Controlled back-to-back q01 run with the same binary and SQL (50 iterations):

| File | q01 (ms) |
| --- | ---: |
| Unzoned whole list | 43.537 |
| Pre-shredding repartition + outer zones | 22.818 |

Shape and size:

- GT is a 13-zone `ZonedLayout` whose data child is one 100,000-row `ListLayout` and whose zone
  child contains 13 rows of outer-list statistics.
- The list metadata contains the 12 actual interior repartition boundaries.
- GT elements still contain 415.1 million rows in 3,167 ordinary physical chunks.
- File size: `1,752,839,708` bytes, an increase of `511,616` bytes (`0.029%`) over experiment 12.
- Overall geomean: `0.948x` versus experiment 12 and `0.991x` versus no-list.
- Excluding q01, geomean is `1.007x` versus experiment 12 and `0.970x` versus no-list.

Learning:

- The pre-shredding repartition + outer-zone configuration recovers most of q01's remaining
  regression: q01 is 1.93x faster in the all-query run and 1.91x faster in a controlled
  back-to-back run. This experiment does not isolate whether that gain comes from the upstream
  repartitioning, the transparent zone wrapper, or their interaction.
- The other queries are effectively neutral versus the unzoned whole-list design, so the zone-map
  benefit does not require reintroducing the costly outer list chunking.
- The physical topology is compatible with the mask-aware list reader: zoning is a transparent
  wrapper around one list, and the list's internal leaves retain their independent chunking.
- Recording observed repartition boundaries makes the shared split grid follow the configured
  `row_block_size` without duplicating that configuration in the list writer.

Artifacts:

- Generation smoke: `/private/tmp/statpopgen-whole-list-outer-zones-generate.jsonl`
- Focused q07/q08: `/private/tmp/statpopgen-q07-q08-whole-list-outer-zones-20.jsonl`
- Controlled q01, zoned: `/private/tmp/statpopgen-q01-whole-list-outer-zones-50.jsonl`
- Controlled q01, unzoned: `/private/tmp/statpopgen-q01-whole-list-unzoned-50.jsonl`
- All queries: `/private/tmp/statpopgen-all-whole-list-outer-zones-20.jsonl`
- Previous unzoned file: `/private/tmp/statpopgen-sf100-whole-list-shared-grid-no-outer-zones.vortex`
- Current file: `/private/tmp/vortex-list-element-splits/vortex-bench/data/statpopgen/100000/vortex-file-compressed/gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex`

Status: completed; currently present in the worktree.

## Current synthesis

- Whole-column list decomposition gives the element reader useful structure but divorces physical
  element chunks from the root-space task model.
- Bigger element chunks help dense list scans substantially.
- Per-leaf natural boundaries cannot be blindly unioned across projected columns.
- A common/coalesced root grid works well for selective wide scans.
- A list reader must choose between dense and selective element reads from the resolved row mask,
  not from whether the requested range happens to cover a complete local layout child.
- A single whole-column list wrapped in outer-row zones is the best overall ordinary-chunk
  configuration tested; chunking the list itself adds avoidable selective-read overhead.
- Zone the outer list before shredding, and derive the shared row splits from those same
  repartitioned chunks so pruning and scan task boundaries stay aligned.
- The long-term splitter likely needs to treat leaf boundaries and estimated work as candidate
  inputs, then globally balance work while capping task count.
