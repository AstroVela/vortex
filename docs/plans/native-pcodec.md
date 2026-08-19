# Native PCodec-inspired numeric compression

## Goal

Improve default numeric compression where ALP and ALP-RD lose to Pco.

Keep native Vortex arrays, bounded metadata, fast canonical decode, and bounded random access.

Pco remains a compression oracle. The new arrays do not preserve Pco byte compatibility.

The production target is the default compressor, not Compact.

Use Compact to identify numeric columns where the default compressor leaves a material size gap.

For each repeated gap, identify the Pco model that creates the gain.

Transfer that model into a native array only when it preserves these Default properties:

- Full decode remains close to the displaced encoding.
- Compression throughput remains close to the displaced encoding.
- Scalar access remains O(1) or bounded O(log N).
- Rejected candidates add little selector cost.
- The size gain remains material after native array overhead.

Compact can use experimental schemes during evaluation. Default selection remains the release decision.

Measure each candidate against both the prior Default tree and the Compact tree.

Use Compact gap recovery as the size metric. Use the displaced Default tree as the speed baseline.

## Current decision

Focus the production work on these encodings:

- `OrderedFloatArray`.
- `BlockResidualArray` for all integer types, with one reference per 1,024-value block.
- `FloatQuantArray`.

Keep `FloatQuantScheme`, `OrderedBlockResidualScheme`, and `BlockResidualScheme` as BtrBlocks candidates.

Remove `FloatMultArray` and `FloatMultScheme` from the focused branch.

Remove `RangeEntropyArray`, `RangeEntropyScheme`, and `BitSplitCodec` from the focused branch.

The `wm/pcodec-entropy-experiments` branch preserves the complete entropy and bit-split prototypes.

Keep `RangePackedArray` and its manual benchmark as experimental work.

Do not add RangePacked to the default or Compact selector.

Do not add adjacent Delta, Delta-of-delta, Delta with lookback, or convolution Delta.

## Current state

The branch implements `OrderedFloatArray`, `BlockResidualArray`, and `FloatQuantArray`.

The default candidate set includes their three BtrBlocks schemes.

`OrderedFloat(BlockResidual)` now supports `f32` and `f64` inputs.

The float scheme applies `OrderedFloat` first, then `BlockResidual` to the unsigned child.

The serialized tree uses `OrderedFloat(BlockResidual(...))` because the outer array restores the float dtype.

The FloatQuant candidate accepts native `f32` and `f64` inputs.

Direct integer BlockResidual supports every integer type. The default selector accepts only 32-bit and 64-bit inputs.

The retained schemes win on specific structures. They do not replace ALP or ALP-RD across general float data.

FloatQuant now passes the speed gates for zero-secondary and one-bit-secondary inputs.

BlockResidual now passes the direct speed gates for 32-bit and 64-bit integers.

The GloVe result identifies a separate entropy gap inside the ALP integer child.

Generic quotient and remainder trees do not close that gap.

The current selector factors and nonlinear patch cost remain provisional.

Final calibration requires the complete corpus and selected-tree evidence.

The current branch recovers a small share of Compact's aggregate numeric advantage.

The remaining work targets repeated Compact mechanisms with native, bounded-access trees.

## OrderedFloatArray

`OrderedFloatArray` maps IEEE float bits to unsigned integers with the same order.

The transform preserves every bit pattern. It also preserves nulls and signed zero values.

The array stores one unsigned child. Empty metadata identifies the transform.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

## BlockResidualArray

`BlockResidualArray` divides integers into independent blocks of 1,024 values.

Unsigned values retain their bit pattern. Signed values first flip the sign bit to preserve numeric order.

Each block stores one minimum value. Packed residuals store the difference from that minimum.

Rare wide residuals use sorted positions and packed high bits.

Scalar access uses one packed read and one binary search over the patch positions.

One aligned parent buffer stores references, widths, offsets, packed words, patch positions, and patch high bits.

Fixed metadata stores only lengths and slice bounds.

Typed zero-copy views point into the parent buffer after a file read.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

The `OrderedFloat(BlockResidual)` execute kernel combines residual decode with the inverse float transform.

The residual payload uses the logical integer width. A 16-bit array uses 16-bit FastLanes pack and unpack operations.

The serialized residual payload remains a `u64` section. The logical type defines the packed word interpretation.

The production codec uses one reference per block. The multi-reference prototype did not justify its encode and decode costs.

## FloatQuantArray

`FloatQuantArray` splits each float into a primary quantum and a secondary adjustment.

Metadata stores only the split width. The array dtype stores the source type.

One or two child arrays store the integer latents.

The BtrBlocks scheme uses a fixed `FoR(BitPacked)` primary tree.

A common path uses `FloatQuant(FoR(BitPacked))` for `f32` values stored in `f64` columns.
An absent secondary child represents zero low bits.

Nonzero low bits use `FloatQuant(FoR(BitPacked), BitPacked)`.

The kernel unpacks both aligned children and reconstructs each float in one pass.

The array supports exact IEEE bit-pattern round trips, nulls, slices, scalar access, and serialization.

The automatic scheme accepts `f32` and `f64` inputs.

Native `f32` and two-child selection meet the selected-column and rejected-column throughput limits.

Final selector factors still require broad corpus validation.

## Selection policy

`FloatQuantScheme` uses the normal sample comparison against ALP, ALP-RD, dictionary, sparse, and RLE schemes.

The sample builds the exact one-child or two-child fixed tree.

The prefilter tracks the bitwise union of all low float bits.

For each split, it compares the removed bit count with the required secondary width.

A candidate must save at least two fixed bits through the split.

The prefilter uses a stratified one-percent sample.

If the sample qualifies, the estimator builds the exact fixed tree on that sample.

The final encoder maps float blocks directly into one or two FastLanes packers.

This fused path does not allocate full primary or secondary integer buffers.

The scheme does not call the recursive integer selector.

`OrderedBlockResidualScheme` uses eight locality-preserving sample blocks.

The ordinary BtrBlocks sample does not preserve the block-local float structure.

The estimator runs the exact production width planner on native integer or float slices.

It does not create packed payloads, patch payloads, temporary ordered integers, or child arrays.

All-valid samples use direct block copies. Nullable samples retain the validity mask.

The residual scheme requires a 1.05 compression ratio. Its adjusted score includes a 1.02 decode-cost factor.

`BlockResidualScheme` uses the same locality probe for integer arrays.

The default scheme accepts only 32-bit and 64-bit integers. Direct 8-bit and 16-bit candidates cannot save enough absolute space.

The scheme does not run inside trial compression for an outer scheme. Generic 64-row samples do not preserve 1,024-row locality.

The selected outer scheme can still choose BlockResidual for its full child.

The integer selector divides the measured compression ratio by these decode-cost factors:

- 1.10 for 32-bit integers.
- 1.20 for 64-bit integers.

A 1.20 factor requires about 16.7 percent fewer estimated bytes. It does not require 20 percent fewer bytes.

The block planner adds 16 synthetic cost bits per patch.

This cost selects the physical residual width. It preserves useful sparse-patch trees.

A global 96-bit patch cost prevented the synthetic dense-patch regressions. It also removed useful sparse-patch compression on HashTags.

The BtrBlocks estimator adds a separate nonlinear decode cost.

For `p` patches and `n` values, it adds `320 * p * p / n` synthetic bits.

This formula adds 80 bits per patch at 25 percent density. It adds 32 bits per patch at 10 percent density.

The physical encoding remains size-optimal. The adjusted size only affects selection.

The selector excludes BlockResidual from dictionary-code children. A complete BlockResidual tree can still displace a complete dictionary tree.

Both schemes remain eligible to displace ALP or ALP-RD when their sample size scores win.

## Performance requirements

A default candidate must meet these limits:

- Compression throughput can regress by at most 20 percent on a selected column.
- Full-decode throughput can regress by at most 20 percent.
- Scalar access must remain bounded and competitive with the displaced encoding.
- The selected candidate must reduce size materially.
- Rejected candidates must add little analysis cost.

Use at least two million logical rows for throughput and scalar-access decisions.

Exclude source-array construction and storage input from codec throughput measurements.

## Evidence for the retained candidates

### FloatQuant on widened f32 values

The input contains two million arbitrary `f32` values stored in an `f64` column.

| Configuration | Bytes | Compression MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Prior default | 14,057,966 | 534.4 | 10,072.7 | 208.7 |
| Default with FloatQuant | 8,753,920 | 565.4 | 14,032.5 | 145.5 |
| Compact | 6,051,737 | 272.2 | 3,912.7 | Not measured |

FloatQuant reduced size by 37.7 percent. It remained 44.7 percent larger than Compact.

Full compressor throughput increased by 5.8 percent. Decode throughput increased by 39.3 percent.

Scalar access latency decreased by 30.3 percent.

The selected tree was `FloatQuant(FoR(BitPacked))` with an implicit-zero secondary.

The fused FloatQuant scheme compressed at 4,799 MB/s.

It decoded between 14,480 and 14,810 MB/s.

Compact Pco compressed the same input at 272 MB/s and decoded it at 3,913 MB/s.

The scheme exceeded Compact throughput by 17.6 times for compression and 3.6 times for decode.

FloatQuant recovered 66.2 percent of the size gap between the prior default and Compact Pco.

The direct sample tree removed the recursive integer selector from the estimate and final compression paths.

FloatQuant meets the selected-column throughput limit on this input.

### FloatQuant with a nonzero secondary

The input changed the lowest bit for ten percent of the widened-`f32` values.

The fixed tree was `FloatQuant(FoR(BitPacked), BitPacked)`. The secondary used one bit per value.

| Configuration | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Prior ALP-RD default | 14,057,966 | 558.0 | 11,420 | 250 |
| Default with FloatQuant | 9,004,032 | 603.2 | 13,060 | 209 |
| Compact Pco | 6,171,139 | 255.5 | 2,914 | Not measured |

FloatQuant reduced size by 36.0 percent. It was 2.9 percent larger than the zero-secondary tree.

FloatQuant recovered 64.1 percent of the size gap between ALP-RD and Compact Pco.

The first generic decode path reached 8,375 MB/s.

The fused pair kernel increased decode throughput by 59.4 percent.

The complete default encoded 8.1 percent faster and decoded 14.4 percent faster than the prior default.

Scalar access latency decreased by 16.4 percent.

The direct two-child scheme compressed at 6,298 MB/s.

The fused kernel supports aligned, patch-free BitPacked secondary children of any width.

Decode throughput was 13,010 MB/s with a one-bit secondary.

It was 12,410 MB/s with a 16-bit secondary.

The real profile selected this tree only for `HashTags_1.twitter#id`.

It reduced that column by 6.1 percent against ALP-RD.

OrderedFloat with BlockResidual reduced the column by 14.5 percent and won the final comparison.

No other tested Pcodec or Public BI column selected the two-child tree.

Rejected analysis changed encode throughput by less than one percent on most measured datasets.

Retain the two-child form in the default candidate set for final corpus calibration.

### OrderedFloat with BlockResidual on random walks

| Configuration | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Prior default | 12,255,488 | 589.8 | 12,438.1 | 207.7 |
| Default with the scheme | 10,425,690 | 529.6 | 21,728.1 | 166.7 |
| Compact Pco | 9,342,749 | 318.1 | 4,559.8 | Not measured |

The residual scheme saved 14.9 percent against the prior default. It remained 11.6 percent larger than Compact.

Decode throughput increased by 74.7 percent. Random access latency decreased by 19.7 percent.

Compression throughput decreased by 10.2 percent on the selected column.

The isolated tree compressed at 2,969 MB/s and decoded at 20,870 MB/s.

Compact Pco compressed the same input at 318 MB/s and decoded it at 4,560 MB/s.

Ordered BlockResidual recovered 62.8 percent of the size gap between ALP-RD and Compact Pco.

The selector rejected the scheme on Gaussian, lognormal, decimal, widened-f32, and four-cluster inputs.

The rejected locality probe reduced compression throughput by 0.5 to 2.2 percent.

The HashTags dataset selected this scheme for two columns.

It reduced the numeric subset by 6.0 percent with no isolated compressor regression.

### Integer BlockResidual throughput

The direct benchmark uses two million block-local values.

| Logical type and tree | Encode GB/s | Decode GB/s | Scalar access ns |
| --- | ---: | ---: | ---: |
| `u64` BlockResidual | 5.11 | 37.39 | 84 |
| `u64` FoR plus BitPacked | 4.51 | 36.22 | 125 |
| `i16` BlockResidual | 1.39 | 31.86 | 125 |
| `i16` FoR plus BitPacked | 1.15 | 42.14 | 125 |

The first `i16` implementation unpacked through `u64` residuals. It decoded at 11.43 GB/s.

Native-width unpack increased `i16` decode throughput by 2.78 times.

The synthetic `i16` BlockResidual tree uses 1,793,784 bytes. The FoR plus BitPacked tree uses 3,500,000 bytes.

BlockResidual is 48.7 percent smaller on that input.

Its `i16` decode throughput is 24.4 percent lower. The default selector therefore excludes direct 8-bit and 16-bit candidates.

### Narrow BlockResidual in Compact

The Compact comparison uses the same two million block-local `i16` values.

| Tree | Bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| FoR plus BitPacked | 3,501,568 | 1,129 | 40,970 |
| BlockResidual | 1,793,784 | 1,361 | 31,330 |
| Compact Pco | 241,832 | 391 | 1,680 |

BlockResidual uses 7.4 times as many bytes as Pco.

It encodes 3.5 times faster and decodes 18.6 times faster than Pco.

Compact gives priority to the Pco size point. Narrow BlockResidual therefore has no Compact-only role.

The 48.8 percent saving against FoR plus BitPacked supports one more narrow decode optimization pass.

The BlockResidual estimator measures its sampled tree exactly, with all payload sections.

The incumbent and outer tree estimates remain approximate. Trial compression previously mis-ranked Dict and ALP on Taxi tips.

The outer-sample exclusion removed that error from the measured tree.

### Broad numeric revalidation before patch-density calibration

This table predates the nonlinear patch cost. The later patch-density section contains the current HashTags result.

The focused run uses two million rows when the source contains that many rows.

The integer-only configuration adds `BlockResidualScheme` to the prior default.

| Dataset | Prior bytes | Integer BlockResidual bytes | Complete default bytes | Integer size change | Complete size change |
| --- | ---: | ---: | ---: | ---: | ---: |
| California Housing | 307,427 | 301,125 | 301,125 | -2.0 percent | -2.0 percent |
| NYC Taxi | 52,407,972 | 52,407,972 | 52,407,972 | 0.0 percent | 0.0 percent |
| CMS Payments | 30,061,670 | 28,472,040 | 28,472,040 | -5.3 percent | -5.3 percent |
| Arade | 26,937,026 | 26,937,026 | 26,937,026 | 0.0 percent | 0.0 percent |
| Euro2016 | 44,698,131 | 39,566,892 | 39,566,892 | -11.5 percent | -11.5 percent |
| Food | 13,579,790 | 13,579,790 | 13,579,790 | 0.0 percent | 0.0 percent |
| HashTags | 22,602,522 | 22,193,089 | 21,018,949 | -1.8 percent | -7.0 percent |

Across these numeric inputs, integer BlockResidual reduced size by 3.7 percent.

Its aggregate encode throughput decreased by 0.4 percent. Its aggregate decode throughput increased by 1.7 percent.

California Housing contains only 20,433 rows. Fixed analysis cost dominates its encode result.

The complete default reduced aggregate size by 4.4 percent. Encode throughput decreased by 1.2 percent.

Aggregate decode throughput increased by 2.3 percent.

Euro2016 reduced size by 11.5 percent and increased decode throughput by 10.2 percent.

CMS reduced size by 5.3 percent and increased decode throughput by 5.5 percent.

HashTags reduced size by 7.0 percent and increased decode throughput by 4.9 percent.

The direct narrow-type exclusion changed no selected tree in this corpus.

Before that exclusion, an earlier 1.40 trial factor rejected each direct `i16` candidate.

A FastLanes fused FoR decode trial did not improve `u64` throughput. It reduced `i16` throughput, so the implementation retains native unpack.

The zero-width residual path now writes base values and patches directly. It skips the scratch residual block.

### Native f32 OrderedFloat with BlockResidual

The input contains two million f32 values with narrow ordered-bit ranges inside each 1,024-value block.

| Operation | Result |
| --- | ---: |
| Encode | 2.45 GB/s |
| Decode | 30.30 GB/s |
| Scalar access | 167 ns |

The fused decode now handles `OrderedFloat(BlockResidual)` trees with u32 latents.

The BtrBlocks candidate now accepts f32 and f64 inputs.

These results remove the prior f32 type exclusion. Corpus comparisons against ALP and ALP-RD remain necessary.

### Native f32 FloatQuant

The input contains two million native `f32` values with eight zero low mantissa bits.

| Configuration | Selected tree | Bytes | Encode MB/s | Decode MB/s |
| --- | --- | ---: | ---: | ---: |
| Prior default | `ALP-RD(BitPacked, BitPacked)` | 5,752,576 | 664.7 | 10,284.4 |
| Default with FloatQuant | `FloatQuant(FoR(BitPacked))` | 3,751,680 | 745.4 | 18,461.6 |
| FloatQuant only | `FloatQuant(FoR(BitPacked))` | 3,751,680 | 746.6 | 18,169.8 |
| Compact Pco | `Pco` | 201,374 | 294.2 | 2,885.0 |

FloatQuant reduced size by 34.8 percent against ALP-RD.

Decode throughput increased by 79.5 percent.

Full compression throughput increased by 12.1 percent in the interleaved compressor benchmark.

The fused FloatQuant scheme compressed at 2.33 GB/s.

The prior materialized tree compressed at 1.25 GB/s in the same direct benchmark.

The proposed default compressed at 753 MB/s in the Divan benchmark.

The prior default compressed at 776 MB/s in that benchmark. The difference is 2.9 percent.

On rejected general `f32` data, the proposed default compressed at 390 MB/s.

The prior default compressed at 394 MB/s. The difference is 1.2 percent.

The Compact size is specific to this synthetic pattern. It does not establish a general real-float result.

### Direct integer comparison after decode specialization

The direct benchmark compares two million block-local values against a whole-column FoR plus BitPacked tree.

The `u32` decode path now unpacks residuals and adds the block base directly into the output buffer.

Signed integers and ordered floats still require a transform after residual decode.

| Type and tree | Decode GB/s | Difference |
| --- | ---: | ---: |
| `u32` BlockResidual | 46.27 | +16.1 percent |
| `u32` FoR plus BitPacked | 39.85 | Baseline |
| `i32` BlockResidual | 33.86 | +31.8 percent |
| `i32` FoR plus BitPacked | 25.69 | Baseline |
| `u64` BlockResidual | 36.88 | +0.6 percent |
| `u64` FoR plus BitPacked | 36.65 | Baseline |
| `i16` BlockResidual | 32.34 | -22.2 percent |
| `i16` FoR plus BitPacked | 41.59 | Baseline |

The specialization removes the prior `u32` decode disadvantage.

The narrow integer exclusion remains valid. Native-width unpack does not close the `i16` gap.

One width factor does not predict both signed and unsigned results.

Retain the current factor until the final corpus calibration uses selected-tree evidence.

### Serialized BlockResidual topology

The first serialized layout stored nine primitive child arrays per BlockResidual array.

The codec kernels remained fast, but the file reader executed nine extra array nodes for each chunk.

An intermediate layout replaced those children with nine direct parent buffers.

That layout improved file decode, but each buffer still produced a separate file segment.

The final layout stores every table and payload in one aligned parent buffer.

The section order preserves native alignment without padding:

1. Block references and residual words use 64-bit sections.
2. Residual and patch offsets use 32-bit sections.
3. Patch positions use one 16-bit section.
4. Widths and packed patch high bits use byte sections.

Typed zero-copy views share the same allocation. Scalar access and bulk decode read those views directly.

The single-encoding benchmark uses two million block-local values.

| Logical type | Encode GB/s | Decode GB/s | Scalar access ns |
| --- | ---: | ---: | ---: |
| `u64` | 5.23 | 36.87 | 83 |
| `u32` | 2.79 | 46.69 | 38 |
| `i32` | 2.78 | 32.23 | 41 |
| `i16` | 1.42 | 32.18 | 83 |

The direct 8-bit and 16-bit selector exclusion remains valid.

The broad file benchmark uses three iterations across eight datasets.

Positive throughput changes indicate faster execution.

| Dataset | Size change | Encode throughput change | Decode throughput change |
| --- | ---: | ---: | ---: |
| Taxi | -3.24 percent | -1.06 percent | +2.51 percent |
| GloVe | 0.00 percent | +2.45 percent | +0.53 percent |
| Arade | -0.19 percent | +0.47 percent | +1.72 percent |
| Bimbo | -1.38 percent | +0.57 percent | +0.05 percent |
| CMSprovider | -1.41 percent | +0.20 percent | -0.06 percent |
| Euro2016 | -2.68 percent | +1.68 percent | -5.74 percent |
| Food | -1.76 percent | +0.50 percent | -3.50 percent |
| HashTags | -0.63 percent | +1.77 percent | -9.71 percent |
| Geometric mean | -1.42 percent | +0.82 percent | -1.86 percent |

Five datasets decoded at parity or faster.

The focused five-iteration runs showed smaller HashTags decode losses of 3.7 to 5.6 percent.

The focused Euro2016 decode loss ranged from 3.8 to 5.1 percent.

Focused encode throughput remained within one percent of the prior default.

This layout recovers most of the Compact-like size gain without a material aggregate throughput loss.

The remaining threshold calibration must use selected-column evidence and the complete corpus.

### Compact transfer baseline

The local corpus comparison uses three iterations on all 16 available datasets.

The proposed Default includes FloatQuant and BlockResidual. The prior Default excludes all new numeric schemes.

The numeric scope contains eight numeric datasets. The real scope adds two TPC-H comment variants.

Positive throughput changes indicate faster execution.

| Scope | Size change | Encode throughput change | Decode throughput change | Proposed gap above Compact | Proposed gap above Parquet with Zstd |
| --- | ---: | ---: | ---: | ---: | ---: |
| Numeric, 8 datasets | -0.84 percent | -0.35 percent | -2.10 percent | +40.84 percent | +1.46 percent |
| Real, 10 datasets | -1.12 percent | -0.47 percent | -1.85 percent | +37.92 percent | +2.28 percent |
| All, 16 datasets | -0.70 percent | -1.45 percent | -1.85 percent | +22.26 percent | -0.84 percent |

The proposed Default preserves the aggregate speed and Parquet size constraints.

It recovers little of Compact's numeric advantage. Compact remains 40.84 percent smaller across the numeric scope.

Taxi gains 3.23 percent, CMS gains 1.34 percent, and Euro2016 gains 2.42 percent against the prior Default.

GloVe size does not change. The remaining datasets change by less than one percent.

This result defines the next objective. Recover repeated Compact gains without a large loss in Default throughput or scalar access.

### Column attribution for Compact wins

The profile reads up to two million numeric rows from seven Public BI files and GloVe.

Twenty-nine float columns give Compact a size advantage of at least ten percent.

The prior Default uses 190,686,999 bytes across those columns. Compact uses 143,939,147 bytes.

Compact saves 24.52 percent, or 46,747,852 bytes.

The Pco trees use these main modes:

- Classic bins without Delta cover 20,200,768 profiled values.
- Classic bins with consecutive Delta cover 9,311,364 values.
- `IntMult` with consecutive Delta covers 5,048,574 values.
- `IntMult` without Delta covers 4,469,863 values.
- `FloatMult` with consecutive Delta covers 3,999,998 values.
- FloatQuant variants cover about four million values.
- Classic bins with lookback Delta cover 1,345,593 values.

Classic bins without Delta form the largest transferable target.

`IntMult` forms a separate target for ALP integer children. GloVe demonstrates this target with `IntMult(10)`.

Consecutive Delta explains several Arade wins. It conflicts with the Default random-access and decode goals.

The largest single gap is Euro2016 `subjectivity_confidence`.

The prior Default uses 14,234,326 bytes. Compact uses 6,860,457 bytes with classic bins and no Delta.

CMS standard-deviation columns also use classic bins without Delta.

CMS payment fields use `IntMult`. CMS submitted-charge fields use FloatQuant.

Food `volume_total_bytes` uses classic bins inside an ALP child.

The column attribution separates numeric Compact gains from file-level string compression gains.

### Selector calibration miss on a Compact gap

The CMS ALP integer child exposes a current BlockResidual threshold miss.

The current Default tree uses 11,064,308 bytes and decodes at 24.98 GB/s.

`ALP(BlockResidual)` uses 10,716,519 bytes and decodes at 22.27 GB/s.

The candidate is 3.1 percent smaller and 10.8 percent slower to decode.

The current 1.20 factor rejects it. Final threshold calibration must decide whether this trade belongs in Default.

### Patch-density sweep

The u32 input uses one large outlier at each configured stride.

| Outlier stride | Approximate patch share | Decode GB/s | Scalar access ns |
| ---: | ---: | ---: | ---: |
| 256 | 0.4 percent | 57.88 | 84 |
| 64 | 1.6 percent | 49.31 | 86 |
| 16 | 6.3 percent | 26.32 | 104 |
| 4 | 25.0 percent | 9.57 | 118 |
| 1 | Packed residuals | 33.02 | 99 |

Scalar access remains bounded across the sweep.

Bulk decode has a severe patch-density cliff before the planner changes to packed residuals.

Before the decode specialization, the complete selector exposed two failures with the 16-bit patch cost.

| Outlier stride | Prior bytes | BlockResidual bytes | Prior decode GB/s | BlockResidual decode GB/s |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 5,508,488 | 3,072,310 | 18.11 | 9.01 |
| 1 | 5,252,352 | 2,543,221 | 44.73 | 33.39 |

The stride-4 tree remains a bad selection because one quarter of the values use patches.

A 96-bit cost selected a near-full residual width for both cases.

The complete selector then retained BitPacked at stride 4. It retained patch-free BlockResidual at stride 1.

With the direct-output path, decode reached 18.06 GB/s at stride 4 and 45.94 GB/s at stride 1.

These results combine the 96-bit planner with the decode specialization. The next sweep must isolate both effects.

The global 96-bit cost also increased proposed-default size on real data:

| Dataset | Size change from 16-bit cost |
| --- | ---: |
| Euro2016 | +0.1 percent |
| HashTags | +3.2 percent |
| Air Quality | +0.04 percent |

On HashTags, size increased from 21,306,418 bytes to 21,989,476 bytes.

Decode throughput changed from 22.54 GB/s to 22.77 GB/s. Encode throughput did not change materially.

The global 96-bit cost is rejected.

The first per-block density gate also rejected the two useful HashTags trees.

The selector-level cost uses total sample density instead. It preserves the physical encoding and compares the complete tree against its incumbent.

The dense synthetic tree now retains BitPacked.

| Synthetic input | Prior tree | Proposed tree | Proposed bytes | Result |
| --- | --- | --- | ---: | --- |
| 25 percent patches | BitPacked | BitPacked | 5,508,488 | Reject BlockResidual |
| Packed residuals | FoR with BitPacked | BlockResidual | 2,543,221 | Select BlockResidual |

The packed-residual prior tree uses 5,252,352 bytes.

BlockResidual reduced its size by 51.6 percent. Encode throughput decreased by 9.4 percent, and decode throughput increased by 8.0 percent.

Two HashTags trees explain the real-data tradeoff:

| Column and candidate | Prior bytes | Candidate bytes | Encode change | Decode change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `twitter#in_reply_to_user_id` BlockResidual | 701,203 | 399,111 | -23.4 percent | -3.0 percent | Reject |
| `interaction#received_at` Ordered BlockResidual | 3,263,939 | 3,018,443 | -38.3 percent | -28.5 percent | Reject |

The first tree fails the selected-column encode gate. The second tree fails both throughput gates.

With the nonlinear cost, complete HashTags size is 21,874,126 bytes.

The prior default uses 22,602,522 bytes. The unpenalized new default used 21,306,418 bytes but selected both failing trees.

The gated default reduces size by 3.2 percent against the prior default.

Its measured aggregate encode throughput increased by 0.9 percent. Decode throughput increased by 5.6 percent.

BlockResidual also composes with outer encodings. Sparse and RunEnd children selected BlockResidual in HashTags and the synthetic low-density sweep.

### BlockResidual analysis cost

California Housing contains 20,433 rows and nine nonnullable `f32` columns.

Every new scheme rejects every column. The output remains 290,737 bytes.

The first eight-block estimator reduced complete encode throughput by 14 to 15 percent.

The exact width planner and direct all-valid copies reduced that cost to 6.3 percent.

| Configuration | Encode MB/s |
| --- | ---: |
| Prior default | 214.4 |
| FloatQuant only | 216.4 |
| Ordered BlockResidual only | 208.1 |
| Integer BlockResidual only | 213.2 |
| Complete proposed default | 200.8 |

A four-block trial reduced analysis cost further. It mis-ranked the California longitude column.

The selected `ALP(BlockResidual)` tree saved only 5.7 percent against the prior tree.

That result did not meet the intended 32-bit speed-adjusted margin. The production candidate retains eight sample blocks.

The two-million-value random walk retained `OrderedFloat(BlockResidual)`.

It used 10,428,451 bytes, encoded at 526.7 MB/s, and decoded at 22.29 GB/s.

The prior default used 12,255,488 bytes, encoded at 595.8 MB/s, and decoded at 12.55 GB/s.

The selected candidate remained inside both throughput gates.

### Complete parent compositions

The finalized benchmark decodes every nested child to recursive canonical form.

Each numeric case contains two million source values. The FSST case contains 500,000 strings.

| Complete tree | Prior bytes | Proposed bytes | Encode change | Decode change |
| --- | ---: | ---: | ---: | ---: |
| Timestamp storage with direct BlockResidual | 9,254,235 | 7,293,370 | -5.1 percent | +348.9 percent |
| `List(BlockResidual)` | 12,755,712 | 2,543,268 | +1.0 percent | +24.8 percent |
| FSST with two BlockResidual children | 4,675,014 | 4,118,592 | -0.4 percent | +8.1 percent |
| `ALP(BlockResidual)` | 7,753,472 | 2,543,268 | -0.2 percent | +1.0 percent |
| `Sparse(BlockResidual, BlockResidual)` | 1,070,594 | 383,297 | +1.9 percent | +6.3 percent |
| `RunEnd(Sequence, BlockResidual)` | 739,968 | 159,124 | +0.5 percent | +1.6 percent |

Each composition reduced size and passed both throughput gates.

The timestamp candidate displaced `DateTimeParts`. It reduced size by 21.2 percent and decoded 4.5 times faster.

The list, ALP, Sparse, and RunEnd cases reduced size by 64.2 to 80.1 percent.

The FSST case reduced size by 11.9 percent with no material encode cost.

A separate float case selected direct `OrderedFloat(BlockResidual)`.

It reduced size by 69.9 percent, increased encode throughput by 128.7 percent, and increased decode throughput by 56.4 percent.

The attempted ALP-RD case did not select ALP-RD. Evidence for BlockResidual under an ALP-RD child remains missing.

Focused selection tests now cover ALP, Sparse, and RunEnd child composition.

Golden snapshots cover temporal and FSST parent trees.

### GloVe embeddings

The corpus now includes the real GloVe dataset with 100,000 rows and 200 f32 values per row.

The complete proposed and previous Vortex defaults both use 68,375,768 bytes.

The Vortex file uses 0.92 times the bytes of Parquet with Zstd.

The first two million embedding values use `ALP(ZigZag(BitPacked))` under both default configurations.

That tree uses 6,274,834 bytes for 8,000,000 raw bytes.

Compact uses `ALP(Pco)` and 5,287,510 bytes. Compact is 15.7 percent smaller than the default tree.

FloatQuant and OrderedFloat with BlockResidual both lose to ALP on this dataset.

Pco receives the `i32` primary child from ALP. Each 262,144-value Pco chunk selects `IntMult(10)` with no Delta encoding.

`IntMult(10)` splits each integer into a quotient and a remainder modulo ten.

The quotient uses 23 to 31 entropy bins. Its average encoded cost is 16.95 to 16.99 bits per value.

The remainder uses ten entropy bins and no offset bits. Its average encoded cost is about 1.08 bits per value.

The complete Pco child uses 4,518,870 bytes for two million values. This result includes model metadata and page overhead.

The size gap comes from a decimal quotient and remainder split plus entropy coding. Pco does not use a Delta or float mode here.

GloVe therefore identifies an ALP-child compression gap. It does not identify a failure in the ALP float transform.

The current Vortex tree does not leave the embedding values uncompressed.

It uses ALP for the float transform, then ZigZag and BitPacked for the integer child.

The Pco advantage comes from a different encoding of the same ALP integer child.

### GloVe quotient and remainder prototypes

The first prototype splits the exact ALP integer child with a constant base.

The estimator stores each child at the source latent width. GloVe `i32` latents produce `u32` quotient and remainder children.

The native-width correction did not change the GloVe byte totals. Each selected child still uses the same packed bit width.

| Candidate | Bytes |
| --- | ---: |
| Pco `IntMult(10)` child | 4,518,870 |
| Base ten with ordinary integer children | 6,002,688 |
| Best tested ordinary base, base 100 | 5,572,096 |
| Base ten with bitmap-patched quotient and mode bitmap remainder | 5,221,476 |
| Direct bitmap patches on the unsplit child | 5,541,496 |
| Direct position patches on the unsplit child | 5,507,862 |

Ordinary integer trees lose materially to Pco on the quotient and remainder.

Bitmap patches improve size, but they do not match the Pco child.

The results reject a general quotient and remainder array with ordinary children.

They support a small-alphabet entropy experiment and a fixed integer split experiment.

### Exact ALP child comparison

The benchmark replaces a Compact `ALP(Pco)` child with the same latent integers in BlockResidual.

This comparison separates float transform quality from integer child compression.

| Input | Current default bytes | `ALP(BlockResidual)` bytes | Current decode MB/s | Candidate decode MB/s | Candidate encode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMS Payments | 4,146,124 | 3,918,239 | 14,202 | 20,280 | 2,605 |
| GloVe | 6,274,834 | 6,298,751 | 16,738 | 19,310 | 1,201 |

The CMS candidate is 5.5 percent smaller and 42.8 percent faster to decode.

The current selector rejects it because the 64-bit BlockResidual score includes a 1.20 factor.

This result makes the CMS miss a final selector calibration issue.

The GloVe candidate is 0.4 percent larger than the current default.

It does not recover the 15.7 percent Pco size advantage.

GloVe still requires a better integer split or small-alphabet backend for the ALP child.

### Fixed-bin range backend experiments

The experimental branch compares ANS symbols with fixed-width bin identifiers.

The packed variants retain variable-width offsets inside each bin.

| Input and backend | Bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| GloVe ALP child, ANS | 5,264,329 | 414.2 | 3,476.5 |
| GloVe ALP child, packed bins | 5,573,968 | 765.8 | 7,700.0 |
| GloVe `IntMult(10)`, ANS | 4,716,002 | 457.8 | 1,944.6 |
| GloVe `IntMult(10)`, packed bins | 5,562,181 | 821.1 | 3,661.7 |
| CMS ALP child, ANS | 3,293,477 | 406.6 | 3,823.0 |
| CMS ALP child, packed bins | 3,647,593 | 642.7 | 8,358.2 |
| Food ALP child, ANS | 5,405,851 | 392.5 | 3,936.6 |
| Food ALP child, packed bins | 5,525,544 | 607.4 | 8,525.6 |

ANS approaches Pco size, but it retains an expensive decode path.

Packed bins decode faster, but they lose part of the size benefit.

Packed bins remain plausible for Compact on selected columns.

Fast scalar access requires checkpoints for the variable offset stream.

A checkpoint interval of 32 values bounds scalar work to 31 offset widths.

This design adds about 0.5 bits per value with 16-bit checkpoints.

### Fixed-bin checkpoint prototype

The checkpoint prototype uses 1,024-value blocks and one 16-bit offset checkpoint per 32 values.

The bin optimizer scores fixed-width identifiers. It restricts bin counts to powers of two.

The bulk decoder uses one narrow-offset specialization for bins with widths of at most 57 bits.

| Input | Default bytes | Pco bytes | Fixed-bin bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic uniform f64 | 14,104,090 | 13,768,605 | 14,037,682 | 1,194 | 10,534 | 48 |
| GloVe ALP child | 6,274,834 | 5,287,510 | 6,177,168 | 539 | 6,600 | 25 |
| CMS ALP child | 4,146,124 | 3,261,563 | 3,515,002 | 1,280 | 13,554 | 23 |

The GloVe fixed-bin tree saves only 1.6 percent against default.

The CMS fixed-bin tree saves 15.2 percent against default. It remains 7.8 percent larger than Compact Pco.

The first CMS decoder reached 8,057 MB/s. The specialized decoder increased throughput by 68.2 percent.

The complete CMS default tree decodes at 14,575 MB/s. The fixed-bin child is 6.1 percent slower.

The benchmark now wraps the codec in a serialized Vortex array.

This wrapper permits complete ALP decode and scalar measurements without a storage-format commitment.

| Input | Default decode MB/s | Compact decode MB/s | Fixed-bin decode MB/s | Fused decode MB/s |
| --- | ---: | ---: | ---: | ---: |
| GloVe | 17,523 | 1,853 | 4,991 | 4,984 |
| CMS Payments | 14,575 | 5,588 | 9,342 | 9,791 |
| Food | 24,000 | 5,569 | 9,431 | 11,195 |

| Input | Default scalar ns | Compact scalar ns | Fixed-bin scalar ns |
| --- | ---: | ---: | ---: |
| GloVe | 824 | 21,374 | 535 |
| CMS Payments | 903 | 14,473 | 383 |
| Food | 535 | 13,569 | 140 |

The fused path writes final floats during range decode. It retains the ALP patch step.

Fusion increased CMS decode by 4.8 percent and Food decode by 18.7 percent. It did not improve GloVe decode.

The fixed-bin tree fails the default decode gate on all three inputs.

CMS and Food show that fixed bins can trade size for faster access than Pco.

CMS uses 7.8 percent more bytes than Compact. Generic decode is 67.2 percent faster, and scalar access is 37.8 times faster.

Food uses 5.3 percent more bytes than Compact. Generic decode is 69.4 percent faster, and scalar access is 97.3 times faster.

The fused Food decode is 2.0 times as fast as Compact. The generic composition already gives most of the benefit.

GloVe uses 16.8 percent more bytes than Compact because Pco also uses `IntMult(10)`.

The evidence does not justify a fused ALP and fixed-bin array.

The temporary Compact selector increased GloVe encode time by 7.8 percent when it rejected RangePacked.

The selector also increased aggregate CMS bytes by 1.1 percent after it replaced Pco.

RangePacked therefore remains outside all writer selectors.

### Nullable and Delta-heavy fixed-bin cases

The optimized nullable prototype stores only valid values in the range stream.

It stores canonical validity bits and one 32-bit rank checkpoint per 256 logical values.

Scalar access tests one validity bit. It then scans at most four words to find the dense value index.

The complete Euro subjectivity tree uses `OrderedFloat(RangePacked)`.

| Input | Default bytes | Compact bytes | Fixed-bin bytes | Default decode MB/s | Compact decode MB/s | Fixed-bin decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Euro subjectivity | 14,234,326 | 6,860,457 | 7,270,094 | 16,822 | 2,475 | 2,487 |
| Euro polarity | 13,505,943 | 11,638,679 | 12,796,702 | 11,517 | 1,487 | 4,767 |
| Arade F8 | 6,380,852 | 5,617,973 | 6,257,548 | 17,944 | 4,371 | 8,350 |

Euro subjectivity uses classic Pco bins without Delta. Fixed bins use 6.0 percent more bytes and match Compact decode.

Its scalar access takes 215 ns. Default takes 657 ns, and Compact takes 19,045 ns.

The complete subjectivity candidate encodes at 440 MB/s.

Euro polarity mixes lookback Delta and no-Delta chunks. Fixed bins use 10.0 percent more bytes and decode 3.2 times faster.

Arade F8 uses consecutive Delta in Pco. Fixed bins use 11.4 percent more bytes and decode 1.9 times faster.

These results do not support Delta in the native candidate.

Fixed bins retain much of Pco's size benefit without Delta. They also preserve bounded scalar access.

A ten-percent Compact size allowance includes CMS, Food, Euro subjectivity, and Euro polarity.

Arade F8 remains outside that threshold. GloVe also remains outside because `IntMult(10)` gives Pco another size advantage.

### Alternate fixed-bin layouts

The byte-aligned prototype rounds every bin offset width to a byte boundary.

Euro2016 `subjectivity_confidence` provides the comparison.

| Backend | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Serial bit-packed bins | 7,238,842 | 541.8 | 3,824.4 | 31.1 |
| Serial byte-aligned bins | 7,835,311 | 474.4 | 3,528.3 | 25.0 |

Byte alignment improves scalar access. It worsens size, encode throughput, and bulk decode throughput.

Cross-byte offset extraction is not the main bulk decode cost.

The grouped-bin prototype stores bit-sliced identifiers and one offset stream per bin.

It uses rank checkpoints every 256 values. Scalar access scans at most four 64-bit words.

| Input and backend | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Euro serial bins | 7,238,842 | 555.2 | 3,949.5 | 32.1 |
| Euro grouped bins | 7,257,069 | 461.0 | 5,665.8 | 25.0 |
| CMS serial bins | 10,650,059 | 1,256.0 | 13,699.0 | 36.8 |
| CMS grouped bins | 10,670,330 | 929.9 | 6,283.7 | 30.3 |

Grouped bins improve Euro bulk decode by 43 percent. The complete Default tree still decodes about four times faster.

CMS uses similar offset widths across its bins. The grouped scatter path then loses more than half of serial-bin throughput.

Grouped bins do not generalize. The prototype remains outside production code.

### Multi-reference block prototype

The prototype stores up to four quantile references per 1,024-value block.

Two-bit FastLanes identifiers select one reference for each value. FastLanes stores the residuals.

The final experiment uses packed `u8` identifiers and a direct four-entry reference lookup.

| Euro candidate | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Default ALP | 14,234,326 | Not isolated | 21,759.7 | 523.6 |
| One-reference BlockResidual | 13,293,263 | 2,171.1 | 24,382.6 | 198.9 |
| Four references with patches | 11,353,335 | 1,026.5 | 11,215.0 | 19.2 |
| Four references without patches | 13,117,584 | 1,357.9 | 12,960.4 | 7.4 |
| Compact Pco | 6,860,457 | Not isolated | 2,467.8 | 18,027.2 |

Packed `u8` identifiers materially improve the multi-reference decoder.

The patched form is 20.2 percent smaller than Default. Its decode throughput is 48.5 percent lower.

The patch-free form is 7.8 percent smaller. Its decode throughput is 40.4 percent lower.

Patch application is not the main cost. Reference-ID expansion and per-value reference lookup dominate.

The one-reference BlockResidual tree is both faster and simpler. The multi-reference prototype is rejected for Default.

### Pcodec corpus gap analysis

The focused benchmark reads the first two million numeric rows from each source.

Air Quality, r/place, and the Twitter graph contain no float columns in the benchmark schema.

The float gap corpus includes California Housing, NYC Taxi, CMS Payments, and four Public BI datasets.

Thirty float columns use at least ten percent fewer bytes with Compact Pco than with the prior default.

Pco uses four principal mechanisms on these columns:

- Entropy bins on ALP integer children.
- First-order Delta plus bins on raw floats or ALP integer children.
- FloatMult plus bins on Taxi tips.
- IntMult splits on selected integer children.

No Pcodec paper gap used lookback Delta. Some Public BI columns used lookback Delta.

GloVe, CMS Payments, and Food include important no-Delta wins.

GloVe uses `IntMult(10)` plus entropy bins. CMS Payments and Food use classic entropy bins.

Arade relies mainly on first-order consecutive Delta. Two columns combine `IntMult` with Delta.

The current CMS source revision differs from the paper source snapshot.

The Twitter source uses the first two million official edges. The paper used an unspecified ID sort.

## General real-float follow-up

Target float columns where Compact Pco materially beats the complete default cascade.

The default baseline includes ALP, ALP-RD, and compression of their children.

Use a ten-percent Pco size advantage as the initial corpus filter. This filter is not a product threshold.

Measure gap recovery as `(default bytes - candidate bytes) / (default bytes - Pco bytes)`.

Retain a prototype only if it recovers a material gap across unrelated real datasets.

The retained schemes cover lower-precision values in wider floats and locally narrow ordered or integer ranges.

The multi-reference estimator tested one, two, and four quantile references per block on thirty gap columns.

The estimate includes reference identifiers, packed residuals, patch positions, patch highs, and block metadata.

One reference won 25 columns. Four references won five columns. Two references won no columns.

The best estimate recovered 29.4 percent of the aggregate Pco size gap.

It reduced default bytes by 7.9 percent across the gap columns.

The extra references did not justify a new array. Generic integer BlockResidual became the next prototype.

Test these secondary candidates after the quotient and remainder prototype:

- Use ordered-bit XOR suffixes instead of arithmetic residuals.
- Use sign and exponent prefixes as fixed reference candidates.
- Use block-local ALP-RD split widths.
- Use independent entropy microblocks only in Compact.

Classify the other gaps by exponent count, prefix count, local range, low-bit entropy, and outlier rate.

Require size, compression, decode, and scalar-access results on the same two-million-row inputs.

Do not add a general codec to the default until it wins across several unrelated real datasets.

## Removed candidates

### FloatMult

The useful exact-multiple form duplicated an ALP candidate that the upstream Rust search omitted.

The upstream ALP fix now includes equal exponent and factor pairs such as `{0,0}`.

The two-child form with nonzero adjustments decoded 64 percent slower than the prior default.

Its scalar access took more than seven times longer in the large-row test.

Vortex will take the corrected ALP dependency after its release. This release does not block other work.

### RangeEntropy overlap

`RangeEntropy` and `OrderedFloat(BlockResidual)` both compress ordered unsigned latents.

`BlockResidual` models one local range per block. It uses a minimum, packed residuals, and sparse high-bit patches.

`RangeEntropy` models many ranges across the input. It entropy-codes range identifiers and stores fixed-width offsets inside each range.

The entropy model can represent separated clusters better than one local reference.

That benefit requires bin search, model construction, two encoded streams, and ANS state transitions.

The measured size gains did not compensate for those costs under the default writer priorities.

| Dataset | Default bytes | RangeEntropy bytes | Compact bytes |
| --- | ---: | ---: | ---: |
| California Housing | 339,682 | 319,705 | 230,197 |
| April 2023 HVFHV Taxi | 28,540,793 | 23,331,922 | 21,127,765 |
| Air Quality | 6,135,414 | 5,149,775 | 2,311,526 |

RangeEntropy reduced Taxi size by 18.3 percent and Air Quality size by 16.1 percent.

It remained larger than Compact on every measured dataset.

RangeEntropy reduced compression throughput by 42 to 49 percent.

Its decode path was 5.7 to 17.2 times slower than the lightweight default.

Pco already occupies the compact, slower-encode design point with equal or better measured size.

RangeEntropy therefore has no current production role in the default or Compact compressor.

### BitSplit

The prototype used one prefix dictionary per block and fixed-width suffixes.

It handled separated clusters better than multiple residual references.

It remained 1.4 percent larger than ALP-RD on the four-cluster input.

Its full decode was much slower than the current ALP-RD path.

The experimental branch retains the prototype for possible ALP-RD decomposition research.

## Revalidation after the ALP release

The workspace now uses ALP 0.0.3.

The corrected ALP selected `ALP(FoR(BitPacked))` for the integer-valued float control.

The control used 5,002,240 bytes with or without the new schemes.

This round completed these steps:

- Replaced the recursive FloatQuant child search with one fixed sample tree.
- Added scalar-access benchmarks on two million values.
- Added nonzero-secondary FloatQuant benchmarks.
- Added all six Pcodec paper datasets.
- Fixed the high-value defects from three adversarial reviews.
- Extended BlockResidual to every integer type.
- Added native-width residual pack and unpack operations.
- Added integer BlockResidual to `single_encoding_throughput`.
- Measured the Pco gap across thirty float columns.
- Rejected the multi-reference residual design.
- Added the outer-sample exclusion and width-specific factors.
- Added the patch-count decode cost.
- Excluded direct 8-bit and 16-bit candidates.
- Excluded BlockResidual from dictionary-code children.
- Added fused f32 OrderedFloat with BlockResidual decode and selection.
- Added u32, i32, f32, and patch-density benchmarks.
- Added complete-compressor patch-density datasets and a column filter to the profiling benchmark.
- Added complete temporal, FSST, Sparse, RunEnd, list, and ALP composition benchmarks.
- Added selection tests for BlockResidual under ALP, Sparse, and RunEnd.
- Added exact allocation-free BlockResidual size estimates.
- Replaced nine BlockResidual child arrays with one aligned parent payload buffer.
- Removed eight file segments from each serialized BlockResidual array.
- Revalidated size and throughput across eight file datasets.
- Added an all-valid fast path for locality sample copies.
- Rejected a four-block short-column estimate after a real mis-ranking.
- Added patch-density statistics to the BlockResidual profile.
- Replaced the global patch cost experiment with a nonlinear selector cost.
- Removed the full patch scan from scalar access.
- Made null payload bits neutral for the new default candidates.
- Excluded integer BlockResidual from the CUDA-compatible preset.
- Added the real GloVe embeddings dataset to the compression corpus.
- Added native `f32` FloatQuant selection and scheme benchmarks.
- Replaced the full primary buffer with fused FloatQuant and FastLanes packing.
- Replaced the generic FloatQuant sample with a direct fixed-tree estimator.
- Added fixed-tree analysis for nonzero FloatQuant secondary values.
- Added direct paired FastLanes packing for both FloatQuant children.
- Added fused two-child FloatQuant decode and width-sweep benchmarks.
- Validated selected and rejected two-child FloatQuant paths.
- Added Compact as a first-class format in the compression benchmark.
- Compared the prior Default, proposed Default, Compact, and Parquet across all 16 local datasets.
- Attributed the largest Compact numeric gaps to Pco modes at the column level.
- Rejected byte-aligned fixed bins after a direct size and throughput comparison.
- Rejected grouped fixed bins after Euro2016 and CMS comparisons.
- Rejected packed multi-reference blocks after patched and patch-free comparisons.

The Pco mode profile and quotient and remainder experiments are complete.

The nonzero-secondary FloatQuant implementation and focused validation are complete.

Complete these remaining steps:

1. Prototype a lightweight integer split for Pco `IntMult` wins under ALP.
2. Prototype one bounded-access scheme for repeated classic-bin wins without Delta.
3. Add a true ALP-RD child composition case if a real selected tree exposes one.
4. Validate the remaining rejected analysis cost after the candidate set stabilizes.
5. Add real embedding datasets beyond GloVe when licenses and loaders permit them.
6. Classify more float columns where Pco beats ALP, ALP-RD, and the new schemes.
7. Run the complete compression corpus after the candidate set stabilizes.
8. Compare the final geometric mean against Compact and Parquet with Zstd.
9. Calibrate all selector thresholds from complete corpus and selected-tree evidence.
10. Revisit the 32-bit and 64-bit BlockResidual factors with selected-tree evidence.
11. Keep RangePacked outside writer selectors unless a new decode design changes the result.
12. Update this plan after each experiment.

## Pull request structure

Prepare one focused stack for `OrderedFloatArray`, `BlockResidualArray`, and `OrderedBlockResidualScheme`.

Prepare a separate stack for `FloatQuantArray` and `FloatQuantScheme`.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
