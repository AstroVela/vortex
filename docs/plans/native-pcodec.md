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
- Selector cost remains proportionate to measured gains.
- Cheap rejection lowers the evidence bar, but it is not required.
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

Remove `RangePackedArray` and its manual benchmark from the focused branch.

The Git history and this plan preserve the fixed-bin experiment.

The fixed-bin experiment used this composed tree:

- `Dict(BitPacked(bin_codes), bin_starts)` reconstructs one reference per value.
- `BlockResidual(offsets)` stores each distance from the selected reference.
- `IntMult(base=1, references, offsets)` adds both components.

The prototype accepted any bin count from one through 64.

The experiment also tested an `IntegerFixedBinsScheme` below ALP.

The complete corpus did not justify either scheme. The focused branch no longer contains either scheme.

Do not add the fused RangePacked array to the Default or Compact selector.

Do not add adjacent Delta, Delta-of-delta, Delta with lookback, or convolution Delta.

## Current state

The branch implements `OrderedFloatArray`, `BlockResidualArray`, and `FloatQuantArray`.

The evaluation set includes BtrBlocks schemes for all three arrays.

`OrderedFloat(BlockResidual)` now supports `f16`, `f32`, and `f64` inputs.

The float scheme applies `OrderedFloat` first, then `BlockResidual` to the unsigned child.

The serialized tree uses `OrderedFloat(BlockResidual(...))` because the outer array restores the float dtype.

The FloatQuant candidate accepts native `f16`, `f32`, and `f64` inputs.

Direct integer BlockResidual supports every integer type. The Default selector accepts every integer width.

The retained schemes win on specific structures. They do not replace ALP or ALP-RD across general float data.

FloatQuant now passes the speed gates for zero-secondary and one-bit-secondary inputs.

The complete file corpus justifies FloatQuant in Default.

OpenAI-on-C4 f64 embeddings select zero-secondary FloatQuant and shrink by 40.5 percent.

BlockResidual now passes the direct speed gates for every integer width.

The GloVe result identifies a separate entropy gap inside the ALP integer child.

The previously tested quotient and remainder trees do not close that gap.

The final calibration retains the 1.05 BlockResidual ratio floor.

It retains the 1.02 ordered-float factor, 1.10 FloatQuant factor, 1.12 8-bit factor, and nonlinear patch cost.

The range decomposition prototype used `IntMult(base=1)` as generic addition.

Its fused decode path skipped multiplication and avoided dictionary reference materialization.

No tested IntMult tree passed the Default evidence bar. The focused branch no longer contains `IntMultArray`.

Decomposed fixed bins do not participate in the Default selector.

The final calibration includes the complete corpus and selected-tree evidence.

The Opportunistic bundle cuts aggregate size by 11.7 percent against Prior Default.

The three array encodings belong to the draft `core:2026.08.4` edition.

The default session still writes the frozen `core:2026.08.1` edition.

Focused file tests and benchmarks enable the draft numeric edition explicitly.

Future work targets repeated Compact mechanisms with native, bounded-access trees.

## Array support and validation

The array API and the Default selector use separate type policies.

An array supports each natural logical type unless the transform has a structural restriction.

The Default selector can exclude a supported type when measured costs do not justify selection.

| Array | Supported logical types | Default policy |
| --- | --- | --- |
| OrderedFloat | `f16`, `f32`, `f64` | All float types remain eligible. |
| BlockResidual | `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64` | All widths are eligible. |
| FloatQuant | `f16`, `f32`, `f64` | All float types remain eligible. |

OrderedFloat validates the logical float type, unsigned child width, child nullability, child length, and empty metadata.

FloatQuant validates the metadata version, split width, latent child types, child nullability, and child lengths.

BlockResidual validates every payload offset table before decode or scalar access.

It also validates bases, bit widths, packed word counts, patch counts, patch order, patch bounds, and validity length.

Bit-exact tests cover signed zero values, infinities, and NaN payloads for every float width.

Round-trip tests cover every signed and unsigned integer width.

## OrderedFloatArray

`OrderedFloatArray` maps IEEE float bits to unsigned integers with the same order.

The transform preserves every bit pattern. It also preserves nulls and signed zero values.

The array stores one unsigned child. Empty metadata identifies the transform.

The array supports `f16`, `f32`, and `f64` values.

It supports canonical decode, scalar access, slice reduction, serialization, and validation.

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

The automatic scheme accepts `f16`, `f32`, and `f64` inputs.

Native `f32` and two-child selection meet the selected-column and rejected-column throughput limits.

The final corpus pass validates the retained selector factors.

## IntMult prototype

The prototype reconstructed each integer as `base * primary + secondary`.

It supported every signed and unsigned integer type.

The primary and secondary children used compatible Vortex array encodings.

The primary child owned validity. The secondary child was nonnullable.

The prototype supported canonical decode, scalar access, slices, serialization, and validation.

`IntMult::from_primitive` created quotient and remainder children with the source integer width.

The tested selectors compressed both children through specialized or recursive integer schemes.

IntMult did not own exception positions or exception payloads.

Child encodings used `PatchArray`, `BlockResidual`, or other integer arrays when those trees won.

When the base equaled one, bulk decode and scalar access skipped multiplication.

For a dictionary primary, the fused path decoded the secondary into the output buffer.

It then unpacked dictionary codes and added the selected reference directly.

This path avoided a full materialized array of references.

## Selection policy

Fast rejection is a selection advantage for schemes in normal Default recursion.

A cheap rejection path reads a bounded sample and avoids child compression when the model does not fit.

This path lets Default test more encodings with little throughput loss on rejected columns.

Fast rejection is not a hard gate.

Do not use fast rejection as an admission criterion.

A scheme with costly analysis can enter Default when stronger corpus evidence justifies its analysis cost.

Specialized schemes remain useful when they bypass depth limits, recursive search, or an unfused common tree.

Measure rejected-column cost separately from selected-column encode cost.

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

`BlockResidualScheme` requires at least a 1.05 compression ratio before outer comparison.

The nonlinear patch cost represents the measured decode cliff for dense patches.

The selector does not apply another width-specific size factor.

Direct 32-bit and 64-bit BlockResidual decode now matches or exceeds the main FoR paths.

BlockResidual does not occur below ZigZag. That composition lost decode throughput on GloVe for a small size reduction.

The ordered-float residual scheme requires a 1.05 compression ratio.

Its adjusted score includes a 1.02 decode-cost factor.

`BlockResidualScheme` uses the same locality probe for integer arrays.

The Default scheme accepts every integer width.

The scheme does not run inside trial compression for an outer scheme. Generic 64-row samples do not preserve 1,024-row locality.

The selected outer scheme can still choose BlockResidual for its full child.

The integer BlockResidual scheme does not use a width-specific decode-cost factor.

The block planner adds 16 synthetic cost bits per patch.

This cost selects the physical residual width. It preserves useful sparse-patch trees.

A global 96-bit patch cost prevented the synthetic dense-patch regressions. It also removed useful sparse-patch compression on HashTags.

The BtrBlocks estimator adds a separate nonlinear decode cost.

For `p` patches and `n` values, it adds `320 * p * p / n` synthetic bits.

This formula adds 80 bits per patch at 25 percent density. It adds 32 bits per patch at 10 percent density.

The physical encoding remains size-optimal. The adjusted size only affects selection.

The selector excludes BlockResidual from dictionary-code children. A complete BlockResidual tree can still displace a complete dictionary tree.

Both schemes remain eligible to displace ALP or ALP-RD when their sample size scores win.

## Single-encoding performance ledger

This snapshot dates from 2026-08-20.

Each benchmark uses two million logical values and 100 Divan samples.

Encode and decode results report median logical input throughput.

Scalar results report median random-access latency.

### Primitive transforms

These cases use canonical primitive children. They isolate each outer array transform.

| Array and input | Encode | Decode | Scalar access |
| --- | ---: | ---: | ---: |
| OrderedFloat `f16` | 59.11 GB/s | 59.09 GB/s | 77 ns |
| OrderedFloat `f32` | 52.38 GB/s | 60.47 GB/s | 82 ns |
| OrderedFloat `f64` | 58.78 GB/s | 42.78 GB/s | 90 ns |
| IntMult base-ten `i8` | 0.72 GB/s | 29.98 GB/s | 125 ns |
| IntMult base-ten `i16` | 1.41 GB/s | 29.68 GB/s | 105 ns |
| IntMult base-ten `i32` | 2.82 GB/s | 29.84 GB/s | 100 ns |
| IntMult base-ten `i64` | 5.57 GB/s | 24.80 GB/s | 169 ns |
| IntMult base-ten `u8` | 0.70 GB/s | 29.63 GB/s | 125 ns |
| IntMult base-ten `u16` | 1.42 GB/s | 30.59 GB/s | 160 ns |
| IntMult base-ten `u32` | 2.84 GB/s | 30.51 GB/s | 136 ns |
| IntMult base-ten `u64` | 5.66 GB/s | 24.57 GB/s | 150 ns |
| FloatQuant zero-secondary `f16` | 7.62 GB/s | 33.02 GB/s | 83 ns |
| FloatQuant zero-secondary `f32` | 13.63 GB/s | 33.58 GB/s | 75 ns |
| FloatQuant zero-secondary `f64` | 20.90 GB/s | 26.88 GB/s | 82 ns |
| FloatQuant one-bit-secondary `f16` | 2.41 GB/s | 2.98 GB/s | 118 ns |
| FloatQuant one-bit-secondary `f32` | 4.87 GB/s | 4.57 GB/s | 136 ns |
| FloatQuant one-bit-secondary `f64` | 9.18 GB/s | 8.23 GB/s | 148 ns |

IntMult encode includes quotient and remainder creation. It excludes compression of both children.

FloatQuant zero-secondary encode uses separate validation and transform passes.

The prior fallible per-value loop reached 1.99 GB/s for `f32` and 3.72 GB/s for `f64`.

The two-pass implementation improved those results by 5.9 times and 5.1 times.

### Production child trees

These cases include the current compressed children and fused decode paths.

| Array tree and input | Encode | Decode | Scalar access |
| --- | ---: | ---: | ---: |
| BlockResidual `i8` | 0.56 GB/s | 30.53 GB/s | 44 ns |
| BlockResidual `i16` | 1.40 GB/s | 31.31 GB/s | 39 ns |
| BlockResidual `i32` | 2.72 GB/s | 33.89 GB/s | 40 ns |
| BlockResidual `i64` | 4.93 GB/s | 35.88 GB/s | 42 ns |
| BlockResidual `u8` | 0.55 GB/s | 30.14 GB/s | 35 ns |
| BlockResidual `u16` | 1.31 GB/s | 32.55 GB/s | 61 ns |
| BlockResidual `u32` | 2.67 GB/s | 46.61 GB/s | 48 ns |
| BlockResidual `u64` | 4.97 GB/s | 35.56 GB/s | 40 ns |
| IntMult with packed children `u8` | 0.70 GB/s | 24.81 GB/s | 125 ns |
| IntMult with packed children `u16` | 1.38 GB/s | 22.90 GB/s | 125 ns |
| IntMult with packed children `u32` | 2.64 GB/s | 22.62 GB/s | 140 ns |
| IntMult with packed children `u64` | 4.34 GB/s | 17.96 GB/s | 125 ns |
| OrderedFloat with BlockResidual `f16` | 1.26 GB/s | 26.36 GB/s | 79 ns |
| OrderedFloat with BlockResidual `f32` | 2.43 GB/s | 29.64 GB/s | 78 ns |
| OrderedFloat with BlockResidual `f64` | 2.70 GB/s | 20.44 GB/s | 125 ns |
| FloatQuant with packed primary `f16` | 3.21 GB/s | 20.05 GB/s | 125 ns |
| FloatQuant with packed primary `f32` | 5.72 GB/s | 19.79 GB/s | 129 ns |
| FloatQuant with packed primary `f64` | 7.76 GB/s | 16.19 GB/s | 125 ns |
| FloatQuant with two packed children `f16` | 3.11 GB/s | 16.75 GB/s | 174 ns |
| FloatQuant with two packed children `f32` | 5.17 GB/s | 16.30 GB/s | 183 ns |
| FloatQuant with two packed children `f64` | 6.72 GB/s | 13.13 GB/s | 208 ns |
| Decomposed fixed bins `u64` | 1.20 GB/s | 14.27 GB/s | 332 ns |

The FloatQuant production encode rows use the single-scheme compressor.

They include sample analysis and direct packing of the final child tree.

The implicit-zero `f64` decoder maps FastLanes output directly to reconstructed floats.

A controlled five-second comparison measured 14.08 GB/s before fusion and 16.19 GB/s after it.

The fused path improves median decode throughput by 15.0 percent.

The same trial reduced `f16` and `f32` throughput by 2 to 3 percent. Those types retain the generic path.

The paired decoder remains within 18 percent of the zero-secondary decoder for every float width.

The packed IntMult tree uses a base of ten and patch-free BitPacked children.

The fused pair decoder unpacks both children directly into the reconstructed output.

It removes the full secondary buffer and the separate multiplication pass.

The paired `u64` benchmark improved decode throughput from 13.10 GB/s to 17.73 GB/s.

This change increased decode throughput by 35.3 percent.

The decomposed fixed-bin row uses ten separated clusters and a BlockResidual offset child.

The fixed-bin result still lacks complete Default and Compact comparisons on real columns.

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

### All-width two-child FloatQuant revalidation

The August 20 pass added two-child benchmarks for `f16` and `f32`.

Each case changes the lowest bit for ten percent of two million quantized values.

| Input | Incumbent bytes | Candidate bytes | Size change | Encode change | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: |
| `f16` | 1,751,040 | 1,751,040 | 0.0 percent | -0.3 percent | +0.2 percent |
| `f32` | 5,752,576 | 4,001,792 | -30.4 percent | +29.6 percent | +43.1 percent |
| `f64` | 14,057,966 | 9,004,032 | -36.0 percent | +10.0 percent | +14.2 percent |

The `f16` selector retains `Dict(BitPacked, Primitive)`.

The `f32` and `f64` selectors choose `FloatQuant(FoR(BitPacked), BitPacked)`.

The paired decoder stays within 18 percent of the zero-secondary decoder for all three widths.

Paired scalar access uses 174 ns for `f16`, 183 ns for `f32`, and 208 ns for `f64`.

The two selected complete trees improve size, encode throughput, and decode throughput.

This evidence supports the opportunistic FloatQuant bundle despite the zero-selection real corpus pass.

### FloatQuant rejection cost

The August 20 pass added isolated Default controls that differ only by FloatQuant.

Each benchmark compresses two million values with 100 samples.

| Input | Default without FloatQuant | Default with FloatQuant | Time change |
| --- | ---: | ---: | ---: |
| General `f16` | 4.383 ms | 4.348 ms | Noise |
| General `f32` | 17.84 ms | 18.03 ms | +1.1 percent |
| General `f64` | 23.62 ms | 23.43 ms | Noise |
| Near-miss `f32` | 23.77 ms | 23.78 ms | Noise |
| Near-miss `f64` | 20.67 ms | 20.58 ms | Noise |

The near-miss inputs pass FloatQuant transform analysis but fail the adjusted ratio test.

The estimator now computes the exact padded FastLanes buffer size and validity size from the analysis.

It no longer packs the sample payload only to count its bytes.

A focused test verifies the estimate against complete `f16`, `f32`, and `f64` trees.

The test includes a nullable two-child case.

The optimization did not produce a measurable complete-compressor change on the near-miss controls.

Sample construction, statistics, and the other schemes dominate those complete times.

The isolated controls place the marginal rejected cost between noise and 1.1 percent.

The 16-file pass remains the broader result. It measured a 0.9 percent write-throughput cost.

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

Its `i16` decode throughput is 24.4 percent lower, but it still exceeds 31 GB/s.

The absolute decode rate makes this a useful Default trade. The selector now includes 16-bit integers.

### 16-bit selector revalidation

The complete synthetic control uses the production compressor without experimental Delta.

| Configuration | Bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Prior Default | 3,501,568 | 599.4 | 38,415 |
| Default with 16-bit BlockResidual | 1,793,269 | 653.8 | 31,600 |

The selected tree reduces size by 48.8 percent and increases encode throughput by 9.1 percent.

Decode throughput decreases by 17.7 percent, but it remains 31.6 GB/s.

Three real files selected direct or nested 16-bit BlockResidual trees.

| Dataset | 32-bit and 64-bit bundle | Bundle with 16-bit | Size change | Encode change | Decode change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bimbo | 10,573,121 | 9,671,397 | -8.53 percent | -1.01 percent | -2.20 percent |
| Food | 12,297,725 | 11,443,009 | -6.95 percent | -1.28 percent | -5.66 percent |
| HashTags | 21,283,222 | 21,254,341 | -0.14 percent | +1.78 percent | +8.81 percent |
| Geometric mean | — | — | -5.27 percent | -0.18 percent | +0.13 percent |

The throughput changes come from adjacent runs and include ordinary benchmark noise.

The exact size gains and high absolute decode rates support 16-bit Default eligibility.

### 8-bit selector validation

The direct benchmark already showed 30.53 GB/s decode and 44 ns scalar access for `i8`.

The complete synthetic control uses two million block-local `i8` values.

| Configuration | Bytes | Encode MB/s |
| --- | ---: | ---: |
| Prior Default with Primitive | 2,000,000 | 820.6 |
| Default with BlockResidual | 1,043,448 | 316.3 |

BlockResidual reduces size by 47.8 percent and decodes at 29.76 GB/s.

The selected tree has a material encode cost because the prior Default stores the input without compression.

A uniform `i8` control rejects BlockResidual and retains Primitive at 2,000,000 bytes.

The rejected analysis changes encode throughput from 720.5 MB/s to 706.3 MB/s, a 2.0 percent decrease.

The original eight numeric files contain no 8-bit columns.

A second pass quantizes 20 million real GloVe values into six `i8` and `u8` variants.

It contains these quantizers:

- Symmetric quantization uses the maximum absolute corpus value. The unsigned form adds 128.
- Global affine quantization maps the corpus minimum and maximum to 0 and 255. The signed form subtracts 128.
- Per-vector affine quantization uses each 200-value vector minimum and maximum. The signed form subtracts 128.

The global affine `i8` and `u8` variants select BlockResidual. The unsigned symmetric variant also selects BlockResidual.

The signed symmetric variant retains ZigZag with BitPacked. Both per-vector affine variants retain Primitive.

| Selected variant | Prior bytes | BlockResidual bytes | Size change | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global affine `i8` | 20,000,000 | 16,775,243 | -16.12 percent | 277.8 | 21,791 | 80 |
| Global affine `u8` | 20,000,000 | 16,775,243 | -16.12 percent | 278.3 | 20,745 | 82 |
| Symmetric `u8` | 20,000,000 | 15,558,250 | -22.21 percent | 294.4 | 32,870 | 78 |

The isolated Primitive encoders reach 754 to 777 MB/s on these selected columns.

The complete writer reduces that isolated encode gap:

| File | Size change | Write time change | Read throughput |
| --- | ---: | ---: | ---: |
| Global affine `i8` only | -16.12 percent | +10.06 percent | 17.3 GB/s |
| Two `i8` variants | -8.06 percent | +0.21 percent | 30.7 GB/s |
| Four `u8` variants | -10.19 percent | +1.62 to +2.25 percent | 35.4 GB/s |

The single-column result isolates the maximum measured writer cost. The multi-column results represent mixed accepted and rejected candidates.

A patch-free boundary control uses one seven-bit local range per block.

It reduces complete file size by 10.35 percent and increases write time by 1.43 percent.

Its complete read throughput is 31.7 GB/s. Scalar access takes 81 ns for `u8` and 110 ns for `i8`.

Residual widths change in whole bits. Therefore, the first patch-free 8-bit win already saves about ten percent after file overhead.

The patch-density adjustment protects cases between those discrete widths.

The bulk and writer results support Default eligibility for `i8` and `u8`.

The file random-access pass adds a material selection trade:

| Input | Size change | Prior correlated | BlockResidual correlated | Prior uniform | BlockResidual uniform |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global affine `i8` | -16.12 percent | 78.6 us | 156.1 us | 497.7 us | 835.6 us |
| Symmetric `u8` | -22.16 percent | 79.5 us | 136.4 us | 490.6 us | 708.1 us |
| Seven-bit boundary | -10.35 percent | 112.7 us | 174.3 us | 785.3 us | 1,014.3 us |

Each pattern requests about 100 rows through a cached file handle.

The candidate increases correlated latency by 55 to 99 percent. It increases uniform latency by 29 to 68 percent.

Every measured lookup remains below 1.1 ms. Sparse patches make the global affine case the slowest relative result.

The selector keeps the common 1.05 raw-ratio floor. It applies a 1.12 access cost factor to 8-bit estimates.

The factor rejects the seven-bit boundary case. That case saved 10.35 percent and increased random-access latency by 29 to 55 percent.

The factor retains the global affine `i8` and `u8` variants. It also retains the unsigned symmetric variant.

These retained variants save 16.12 to 22.21 percent. The measured file-access latency remains below one millisecond for about 100 requested rows.

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

### Native f16 coverage

The audit added `f16` support to OrderedFloat, OrderedFloat with BlockResidual, FloatQuant, and FloatQuantScheme.

The test corpus covers zero low bits, nonzero secondary bits, signed zero values, infinities, and NaN payloads.

The quantized input contains two million `f16` values with four zero low mantissa bits.

| Configuration | Bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Prior default | 1,500,800 | 368.1 | 7,268 |
| Default with FloatQuant | 1,500,672 | 1,018 | 19,450 |

The selected Default tree is `FloatQuant(FoR(BitPacked))`.

It uses nearly the same space as the prior dictionary-like tree.

Compression throughput increased by 2.77 times. Decode throughput increased by 2.68 times.

On general rejected `f16` values, median compression throughput decreased by 1.8 percent.

The first `f16` BlockResidual tree decoded at 20.50 GB/s.

A later same-benchmark comparison measured 24.31 GB/s before a fused narrow transform and 26.36 GB/s after it.

The fused path improves median decode throughput by 8.4 percent. It reconstructs residuals and float bits in one output pass.

Retain `f16` eligibility. Revisit its selector factors during final corpus calibration.

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

Native-width unpack leaves a percentage gap for `i16`, but its absolute decode rate remains high.

One width factor does not predict both signed and unsigned results. The integer selector uses no width-specific factor.

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
| `i8` | 0.56 | 30.53 | 44 |

The 16-bit path now meets the revised size and absolute-throughput preference.

The 8-bit path also clears the decode and scalar-access gates.

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

### BlockResidual selector calibration

The final numeric trial removes the prior 1.10 and 1.20 width factors.

The 1.05 minimum ratio and nonlinear patch cost remain active.

An initial trial selected `ALP(ZigZag(BlockResidual))` on GloVe.

That tree reduced size by 1.2 percent and reduced decode throughput by 19.9 percent.

No other selected corpus tree placed BlockResidual below ZigZag. An ancestor exclusion removes that composition.

The final run reads two million rows from seven Public BI files and GloVe.

| Dataset | Size change | Encode throughput change | Decode throughput change |
| --- | ---: | ---: | ---: |
| Arade | -3.13 percent | -4.44 percent | +8.80 percent |
| Bimbo | -2.97 percent | -0.36 percent | +1.78 percent |
| CMSprovider 1 | -1.63 percent | -3.42 percent | -1.29 percent |
| CMSprovider 2 | -1.56 percent | -3.32 percent | -0.65 percent |
| Euro2016 | -11.47 percent | -3.61 percent | +6.23 percent |
| Food | -6.89 percent | -0.85 percent | -4.00 percent |
| HashTags | -5.83 percent | -0.47 percent | +4.36 percent |
| GloVe | 0.00 percent | +0.43 percent | +3.63 percent |
| Geometric mean | -4.25 percent | -2.02 percent | +2.28 percent |

The prior width factors reduced size by 2.28 percent in the same eight-dataset scope.

They changed encode throughput by -0.49 percent and decode throughput by +3.30 percent.

The new policy recovers another 1.97 percent of geometric-mean size.

It keeps every dataset throughput change far inside the 20-percent gate.

The complete file corpus remains necessary before the policy becomes final.

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

### Resolved selector calibration miss on a Compact gap

The CMS ALP integer child exposed a historical BlockResidual threshold miss.

The current Default tree uses 11,064,308 bytes and decodes at 24.98 GB/s.

`ALP(BlockResidual)` uses 10,716,519 bytes and decodes at 22.27 GB/s.

The candidate is 3.1 percent smaller and 10.8 percent slower to decode.

The historical 1.20 factor rejected it.

Later calibration removed the width factors.

The current integer scheme uses a 1.05 raw-ratio floor and no decode-cost factor.

The current compressor selects `ALP(BlockResidual)` at 10,716,519 bytes.

This result resolves the threshold miss. The nonlinear patch cost still protects dense-patch decode.

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

### Bounded IntMult prototypes

The direct prototypes use 1,024-value blocks and exact quotient and remainder splits.

Each quotient block stores one reference, FastLanes-packed low bits, and bounded bitmap patches.

The GloVe variants tested three remainder layouts:

- A mode value with bitmap exceptions.
- Gap positions with a maximum scan of 1,024 values.
- One exception bit per value pair with one payload byte per exceptional pair.

| GloVe base-ten child | Bytes | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: |
| Bitmap remainder | 5,197,126 | 10,400 to 11,200 | 14 |
| Gap positions | 5,348,130 | 10,800 to 12,400 | 78 to 88 |
| Pair bytes | 5,184,309 | 10,600 to 10,700 | Not measured |

The current Default GloVe tree decodes at 16,700 to 17,500 MB/s.

The bitmap variant encodes at 650 to 666 MB/s. It applies 142,861 quotient patches and 298,831 remainder exceptions.

The gap layout loses size and scalar speed. The pair layout does not improve bulk decode.

The CMS prototype stores every remainder with one dense FastLanes stream. This layout avoids sparse remainder expansion.

| CMS payment child, base ten | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Pco child | 9,141,872 | Not isolated | Not isolated | Not isolated |
| Dense remainder prototype | 9,811,885 | 879 | 8,094 | 19 |

The complete Default CMS tree uses 10,494,404 bytes and decodes at 25,226 MB/s.

The dense prototype applies 853,533 quotient patches across two million values.

A diagnostic decode without quotient patches reaches 12,628 MB/s. The split and reconstruction costs still miss the Default decode target.

These results reject the tested bespoke IntMult layouts for Default.

The layouts retain bounded scalar access, but their bulk decode cost is too high.

They do not reject a pure IntMult transform with independently compressed children.

The result does not reject an IntMult backend for Compact. Compact accepts a different throughput and random-access tradeoff.

### Exact ALP child comparison

The benchmark replaces a Compact `ALP(Pco)` child with the same latent integers in BlockResidual.

This comparison separates float transform quality from integer child compression.

| Input | Current default bytes | `ALP(BlockResidual)` bytes | Current decode MB/s | Candidate decode MB/s | Candidate encode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMS Payments | 4,146,124 | 3,918,239 | 14,202 | 20,280 | 2,605 |
| GloVe | 6,274,834 | 6,298,751 | 16,738 | 19,310 | 1,201 |

The CMS candidate is 5.5 percent smaller and 42.8 percent faster to decode.

The selector at the time rejected it because the 64-bit BlockResidual score included a 1.20 factor.

The current selector removed that factor. The resolved calibration section records the selected 10,716,519-byte tree.

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

### Composable fixed-bin prototype

The prior RangePacked prototypes fused bins, offset streams, checkpoints, and reconstruction into one codec.

The current prototype separates the model from storage.

`RangeDecomposition` fits any number of bins through 64.

It produces three logical components:

- A small array of bin starts.
- One fixed-width bin code per value.
- One integer offset from the selected start per value.

The manual benchmark builds this exact tree:

`IntMult(base=1, Dict(BitPacked(codes), starts), BlockResidual(offsets))`.

The first IntMult decode materialized every dictionary reference before addition.

A generic dictionary-add path increased decode throughput from 12.06 GB/s to 12.62 GB/s.

A packed-code specialization increased median decode throughput to 14.27 GB/s.

The specialization unpacks codes and adds starts directly into the decoded offset buffer.

It does not multiply because the IntMult base equals one.

The benchmark uses two million `u64` values in ten separated clusters.

| Operation | Result |
| --- | ---: |
| Encode | 1.20 GB/s |
| Decode | 14.27 GB/s |
| Scalar access | 332 ns |

A separate base-ten IntMult benchmark uses two million `i32` values and primitive children.

| Operation | Result |
| --- | ---: |
| Split encode | 2.88 GB/s |
| Decode | 30.67 GB/s |
| Scalar access | 125 ns |

These values establish low transform overhead and bounded scalar access.

They do not establish a Default win because the range benchmark lacks incumbent size and throughput comparisons.

The next experiment must compare complete trees on the real Pco gap columns.

### Decomposed fixed-bin real-data results

The benchmark now builds `IntMult(base=1, Dict(BitPacked(codes), starts), offsets)` on four real float columns.

It tests `BlockResidual` and `FoR(BitPacked)` as the offset child.

The GloVe benchmark reader flattens numeric `List`, `LargeList`, and `FixedSizeList` columns.

| Input | Default bytes | Compact bytes | Decomposed BR bytes | Decomposed FoR bytes | Default decode MB/s | Decomposed BR decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Euro subjectivity | 14,234,326 | 6,860,457 | 13,000,391 | 14,006,288 | 16,915 | 4,217 |
| Food volume | 7,529,728 | 5,368,140 | 7,223,283 | 10,504,832 | 24,104 | 10,060 |
| GloVe embeddings | 6,274,834 | 5,287,510 | 7,105,298 | 8,160,260 | 17,388 | 7,125 |
| CMS payment | 11,064,308 | 9,288,800 | 11,073,385 | 12,735,216 | 22,542 | 6,570 |

The `BlockResidual` form saves 8.7 percent on Euro and 4.1 percent on Food against Default.

It grows GloVe by 13.2 percent and matches Default size on CMS payment.

Its decode throughput falls by 58 to 75 percent against Default on all four columns.

The `FoR(BitPacked)` form encodes and decodes faster than the `BlockResidual` form.

Its global offset range loses more size on every input.

The offset stream interleaves values from bins with different widths.

A generic offset child then pays for the widest offsets in each local block.

The fused prototype avoids this cost because each value uses the width for its selected bin.

The tested composition therefore does not pass the Default evidence bar.

A competitive fixed-bin design requires a reusable primitive for code-dependent widths or grouped offsets with rank data.

Earlier grouped and checkpoint prototypes did not pass the Default decode gate.

The generic fixed-bin composition remains paused for Default.

### Pure IntMult real-data results

The next prototype applies ALP first and splits its integer child with one global IntMult base.

Default compresses the quotient and remainder independently through normal integer recursion.

The benchmark tests bases 5, 10, 100, and 1,000.

| Input | Default bytes | BlockResidual bytes | Best IntMult bytes | Default decode MB/s | Best IntMult decode MB/s | Best IntMult encode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GloVe embeddings | 6,274,834 | 6,298,751 | 6,665,400 at base 1,000 | 17,504 | 10,188 | 445 |
| CMS payment | 11,064,308 | 10,716,519 | 10,442,347 at base 10 | 22,715 | 10,212 | 412 |

GloVe grows by 6.2 percent against Default.

CMS shrinks by 5.6 percent against Default and 2.6 percent against `ALP(BlockResidual)`.

The CMS IntMult tree decodes 55 percent slower than Default and 51 percent slower than `ALP(BlockResidual)`.

Its encode throughput falls 77 percent against the direct `ALP(BlockResidual)` candidate.

CMS scalar access remains close to Default, but it remains slower than the BlockResidual candidate.

Pure IntMult with generic children therefore does not pass the Default evidence bar.

Pco gains more from its bin and entropy backend than from the multiplication split alone on GloVe.

The CMS standard-deviation payment column provides a no-Delta IntMult target.

Pco uses `IntMult(10)` on seven of eight chunks. The remaining chunk uses classic bins.

| Tree | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Default | 10,494,404 | Not isolated | 24,593 | 866 |
| `ALP(BlockResidual)` | 10,220,895 | 2,124 | 21,889 | 551 |
| Best IntMult, base 1,000 | 10,300,986 | 530 | 10,704 | 727 |
| Compact Pco | 9,325,132 | Not isolated | 1,977 | 25,987 |

The IntMult tree saves 1.8 percent against Default. It remains larger than the BlockResidual tree.

Its encode throughput is four times lower than BlockResidual. Its decode throughput is more than twice as low.

This result removes Delta as a confounder. Pure IntMult remains outside Default.

### Equal-width prefix bins

The equal-width prototype uses only `IntMult`, `Dict`, and `BitPacked`.

The base is a power of two. A dictionary stores at most 64 observed quotients.

The remainder uses one fixed suffix width for every value.

This tree provides fast rejection when more than 64 quotients appear in the sample.

| Input | Best suffix width | Candidate bytes | Default bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| GloVe embeddings | 20 | 6,909,476 | 6,274,834 | 1,348 | 7,538 |
| CMS payment | 44 | 11,984,848 | 11,064,308 | 2,487 | 8,709 |

The encoder is fast because it builds one dictionary and two packed streams.

The fixed suffix grows both columns by 8 to 10 percent against Default.

The generic decoder is also more than twice as slow as Default.

Equal-width prefix bins do not justify a Default scheme.

### Frequency-ranked dictionary codes

Dictionary codes previously followed hash-table iteration order.

The prototype assigns the lowest codes to the most frequent values. It then compresses both children through normal recursion.

This transform uses existing `Dict`, `BitPacked`, `Sparse`, and BlockResidual arrays. It adds no array format.

Integer statistics already store frequency counts. Float statistics now retain counts during the existing distinct-value pass.

| Column and tree | Bytes | Decode MB/s | Scalar ns | Candidate construction MB/s |
| --- | ---: | ---: | ---: | ---: |
| Bimbo `Venta_hoy`, old Default | 3,544,154 | 26,689 | 595 | Not isolated |
| Bimbo `Venta_hoy`, ranked proposed Default | 3,197,655 | 25,490 | 577 | 1,480 |
| Bimbo `Dev_proxima`, ranked prior Default | 280,812 | 26,902 | 3,834 | Not isolated |
| Bimbo `Dev_proxima`, ranked Dict with BlockResidual descendants | 234,832 | 27,134 | 3,823 | 1,367 |

Frequency ranking reduces `Venta_hoy` by 9.8 percent against the old Default.

The complete proposed tree keeps decode and scalar throughput close to the old tree.

The `Dev_proxima` gain depends on BlockResidual below the dictionary code child.

The former ancestor exclusion prevents that useful composition. The corpus trial therefore removes the integer and float Dict exclusions.

The scheme retains exclusions for string and binary dictionaries.

The current eight-dataset numeric trial compares the ranked prior Default with the ranked proposed Default.

| Dataset | Size change | Encode throughput change | Decode throughput change |
| --- | ---: | ---: | ---: |
| Arade | -3.19 percent | -5.17 percent | +1.93 percent |
| Bimbo | -3.57 percent | -0.37 percent | +0.23 percent |
| CMSprovider 1 | -2.26 percent | -6.51 percent | -6.34 percent |
| CMSprovider 2 | -3.99 percent | -6.97 percent | -9.06 percent |
| Euro2016 | -11.47 percent | -3.63 percent | +11.68 percent |
| Food | -7.14 percent | -3.91 percent | -10.18 percent |
| HashTags | -5.68 percent | +0.43 percent | +5.46 percent |
| GloVe | 0.00 percent | +0.11 percent | +1.85 percent |
| Geometric mean | -4.72 percent | -3.29 percent | -0.80 percent |

Every dataset remains inside the 20-percent throughput gate.

Frequency ranking needs no separate selector. Dictionary analysis already computes the required distinct values.

Cheap rejection remains useful guidance for other schemes. A costly scheme can enter Default only with stronger corpus evidence.

### BlockResidual patch positions

The patch prototype replaces sorted ALP patch indices and chunk offsets with BlockResidual when it reduces their byte size.

It leaves patch values unchanged.

| Input and tree | Original bytes | Candidate bytes | Original decode MB/s | Candidate decode MB/s | Original scalar ns | Candidate scalar ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GloVe Default | 6,274,834 | 6,071,376 | 18,077 | 17,178 | 812 | 2,076 |
| GloVe `ALP(BlockResidual)` | 6,298,751 | 6,095,293 | 19,233 | 18,735 | 503 | 1,766 |
| CMS Default | 11,064,308 | 11,040,803 | 24,168 | 23,111 | 934 | 1,551 |
| CMS `ALP(BlockResidual)` | 10,716,519 | 10,693,014 | 20,678 | 19,982 | 656 | 1,241 |
| Euro subjectivity Default | 14,234,326 | 11,211,722 | 21,055 | 16,583 | 631 | 9,801 |

Food volume has no ALP patches on this input.

BlockResidual saves 3.2 percent on GloVe patch trees and 21.2 percent on Euro Default.

CMS saves only 0.2 percent.

Patch lookup uses binary search over the index array.

Compressed indices require one generic scalar read for each comparison.

This path increases scalar latency by 1.7 to 15.5 times across the measured trees.

Direct BlockResidual patch positions therefore do not pass the random-access gate.

A future general patch backend can retain this size opportunity through a fast sorted-search kernel or a bitmap with bounded rank data.

Do not add a BlockResidual special case to ALP or IntMult patch handling.

### Direct unsigned BlockResidual decode

The array decoder now writes unpacked unsigned values directly into the final output buffer.

The prior path used a temporary 1,024-value block for u8, u16, and u64 values.

| Type | Prior decode GB/s | Direct decode GB/s | Change |
| --- | ---: | ---: | ---: |
| u8 | 30.14 | 40.37 | +34.0 percent |
| u16 | 32.55 | 45.86 | +40.9 percent |
| u32 | 46.61 | 47.23 | +1.3 percent |
| u64 | 35.56 | 39.37 | +10.7 percent |

u32 already used the direct path.

The signed prototype unpacked into the final unsigned buffer and flipped the sign bit in place.

That extra pass reduced i16 throughput from about 31.4 GB/s to 27.4 GB/s.

It also reduced throughput for i8, i32, and i64.

The implementation retains the prior signed path.

Current i16 BlockResidual decode reaches 31.4 GB/s. The comparable `FoR(BitPacked)` tree reaches 42.0 GB/s.

At that stage, this result retained the direct i8 and i16 selector exclusions.

At that stage, the u16 result lacked real selected-tree evidence.

Later absolute-throughput and complete-file results re-enabled all narrow integer widths.

The faster unsigned path also improves the BlockResidual patch-position bulk decoder.

GloVe patch decode now stays within 2 percent of Default. Euro patch decode remains 12 percent slower than Default.

Scalar patch lookup remains the blocking cost.

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

### Precomputed offset masks

The serial decoder previously constructed one low-bit mask for each nonzero offset.

The 8-lane decoder now stores one mask per bin in its 1 KiB decode table.

For offset widths from 41 through 57 bits, each lane performs one load, one shift, and one mask operation.

This path removes the unpredictable zero-width branch and the per-lane mask construction.

The narrow path retains its zero-width branch because an unconditional load reduced Euro throughput.

| Input | Fixed-bin bytes | Direct decode MB/s | Complete decode MB/s | Fused decode MB/s | Default decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Euro subjectivity | 7,270,094 | 15,932 | 5,121 | 5,965 | 17,829 to 20,600 |
| Food volume | 5,653,643 | 14,244 | 9,025 | 12,114 | 24,903 |
| CMS `LINE_SRVC_CNT` | 2,463,791 | 14,031 | 5,731 | 4,071 | 26,249 |
| CMS average payment | 10,845,743 | 15,390 | 3,545 | 3,746 | 23,861 |

Euro direct decode previously reached 3,721 to 3,979 MB/s.

The precomputed mask increases its direct throughput by about four times.

The 500-iteration rerun produced 15,932 MB/s.

Food and CMS `LINE_SRVC_CNT` use offset widths of at most 40 bits, so they do not use the new path.

CMS average payment uses the new path, but BlockResidual gives a smaller complete tree.

The optimized codec establishes a much higher fixed-bin decode ceiling.

The complete dense candidate still fails the Default decode gate because validity checks and float reconstruction dominate its final path.

An in-place reverse validity expansion reduced complete Euro decode to 3,116 MB/s.

Its reverse bit iteration cost exceeded the allocation and copy cost that it removed.

The implementation retains the forward expansion with a separate logical output vector.

### Full-position null storage

The full-position mode stores physical values at null positions instead of a dense valid-value stream.

The validity child remains present, but the rank child is absent.

The missing rank child identifies full-position storage without a metadata flag.

Bulk decode then writes one value per logical position and skips validity expansion.

Scalar access uses the logical index directly.

| Input | Mode | Bytes | Encode MB/s | Fused decode MB/s | Scalar ns |
| --- | --- | ---: | ---: | ---: | ---: |
| Euro subjectivity | Dense | 7,270,094 | 461 | 5,965 | 232 |
| Euro subjectivity | Full-position | 8,180,506 | 654 | 15,180 | 206 |
| CMS `LINE_SRVC_CNT` | Dense | 2,463,791 | 875 | 4,071 | 465 |
| CMS `LINE_SRVC_CNT` | Full-position | 2,470,956 | 942 | 10,623 | 445 |

The Euro result uses 200 complete-tree decode iterations.

Default uses 14,234,326 bytes and decodes at 20,924 MB/s on the same run.

Full-position RangePacked reduces Euro size by 42.5 percent and reduces decode throughput by 27.5 percent.

Compact uses 6,860,457 bytes and decodes at 2,574 MB/s.

The full-position candidate retains 82.1 percent of the Compact size gain and decodes 5.9 times faster than Compact.

CMS `LINE_SRVC_CNT` still favors `ALP(BlockResidual)` for Default.

That tree uses 2,725,541 bytes and decodes at 23,493 MB/s.

The Euro candidate now warrants a complete corpus and selector-cost evaluation.

CMS uses similar offset widths across its bins. The grouped scatter path then loses more than half of serial-bin throughput.

Grouped bins do not generalize. The prototype remains outside production code.

### Fused parent kernels and the experimental selector

RangePacked now provides parent kernels for `OrderedFloat(RangePacked)` and `ALP(RangePacked)`.

The session registry selects these kernels from the child encoding and the parent encoding.

The outer arrays remain generic composable arrays.

On Euro subjectivity, normal recursive execution reaches 14,956 MB/s.

The manual fused benchmark reaches 15,492 MB/s on the same run.

The registered kernel is within 3.5 percent of the manual helper.

An experimental `OrderedFloatRangePackedScheme` remains outside `ALL_SCHEMES`.

The scheme tests two complete trees on the existing stratified one-percent sample:

- `OrderedFloat(RangePacked)`
- `ALP(RangePacked)`

The full encoder builds only the smaller sampled tree.

The selector divides the sample ratio by a provisional 1.20 decode cost factor.

This factor does not produce the intended complete-tree choices on CMS or Food.

The one-percent sample overstates the RangePacked advantage against BlockResidual on those columns.

| Input | Default bytes | Candidate bytes | Default encode MB/s | Candidate encode MB/s | Default decode MB/s | Candidate decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Euro subjectivity | 14,234,326 | 8,180,506 | 362.4 | 376.4 | 21,795 | 14,849 |
| CMS `LINE_SRVC_CNT` | 2,725,541 | 2,436,604 | 1,054.8 | 630.5 | 24,227 | 10,354 |
| Food volume | 6,599,221 | 5,653,643 | 491.3 | 538.2 | 19,747 | 11,920 |

Euro is the target win.

CMS and Food do not justify their decode losses at the current size gains.

The rejected-candidate cost is also material.

On uniform random floats, the candidate reduces total encode throughput by 6.8 percent.

On widened `f32` values, FloatQuant still wins, but the candidate reduces encode throughput by 18.8 percent.

These results justify a cheap rejection test, if that test retains the Euro model.

Fast rejection remains an advantage, not an admission rule.

A candidate with costly analysis needs stronger corpus evidence.

### RangePacked estimate and rejection cost

A focused Samply profile attributed 11.79 percent of CPU time to the rejected RangePacked callback.

Bin construction consumed almost all of that time. Exact size estimation alone did not reduce the cost.

The estimator now fits the bins and counts exact child-buffer bytes. It does not construct the packed payload.

A coarse model uses the existing one-percent sample. It compares fixed prefix bins with a single global range.

The model also compares 64-value local ranges with the global range model.

A large local advantage identifies inputs that favor BlockResidual. The selector then skips RangePacked before ALP analysis.

The locality test retains Euro and rejects the random walk, Food, and CMS candidates.

| Input | Default encode MB/s | Candidate encode MB/s | Write change | Selected RangePacked |
| --- | ---: | ---: | ---: | --- |
| Uniform floats | 639.5 | 629.7 | -1.5 percent | No |
| Widened `f32` values | 680.3 | 662.9 | -2.6 percent | No |
| Random walk | 581.0 | 576.2 | -0.8 percent | No |
| Nonzero-secondary FloatQuant | 677.8 | 659.8 | -2.7 percent | No |
| Quantized `f32` | 1,410.8 | 1,338.7 | -5.1 percent | No |
| Euro subjectivity | 357.8 | 359.9 | +0.6 percent | Yes |

These focused paired results contain benchmark noise. The complete corpus will determine the final policy.

On Euro, RangePacked uses 8,180,506 bytes instead of 14,234,326 bytes.

Its decode throughput is 15,179 MB/s instead of 19,716 MB/s.

Food and CMS now retain their existing BlockResidual trees. Their prior RangePacked choices lost too much decode throughput.

### Broad RangePacked selector pass

The next pass covered seven Public BI files and the GloVe file.

Only Euro and HashTags selected RangePacked. The other six files retained every existing tree.

| Dataset | Default bytes | Candidate bytes | Default encode MB/s | Candidate encode MB/s | Default decode MB/s | Candidate decode MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arade | 26,078,866 | 26,078,866 | 549.8 | 539.6 | 21,853 | 23,772 |
| Bimbo | 10,573,121 | 10,573,121 | 767.8 | 764.8 | 11,181 | 11,342 |
| CMS provider 1 | 70,490,567 | 70,490,567 | 492.0 | 480.3 | 16,387 | 16,394 |
| CMS provider 2 | 70,640,453 | 70,640,453 | 483.9 | 475.0 | 15,641 | 15,623 |
| Euro2016 | 39,571,884 | 33,518,064 | 506.8 | 493.7 | 20,356 | 18,969 |
| Food | 12,297,725 | 12,297,725 | 542.7 | 541.8 | 19,633 | 19,650 |
| HashTags | 21,283,222 | 20,147,808 | 709.8 | 631.6 | 22,303 | 22,207 |
| GloVe | 6,274,834 | 6,274,834 | 363.0 | 342.8 | 19,341 | 19,440 |

The geometric mean gives 2.7 percent less size and 3.3 percent less write throughput.

The geometric-mean read throughput increases by 0.4 percent. This small change is within benchmark noise.

HashTags `interaction#received_at` is the second selected column.

RangePacked uses 2,128,525 bytes instead of 3,263,939 bytes. It reduces size by 34.8 percent.

Its focused decode throughput is 14,658 MB/s instead of 15,678 MB/s.

Its focused write throughput is 231 MB/s instead of 1,254 MB/s.

The direct RangePacked tree encoder reaches 660 MB/s. Repeated scheme analysis accounts for part of the complete write cost.

This HashTags trade needs the final threshold review. It does not yet justify default inclusion.

The ALP RangePacked branch selected no column in this pass. Its rejected analysis contributes to the GloVe and synthetic costs.

An OrderedFloat-only estimator reduced rejected write costs on every affected control.

| Input | General selector loss | OrderedFloat-only loss |
| --- | ---: | ---: |
| Widened `f32` | 2.6 percent | 0.5 percent |
| Nonzero-secondary FloatQuant | 2.7 percent | 1.3 percent |
| Quantized `f32` | 5.1 percent | 2.4 percent |
| GloVe | 5.6 percent | 1.3 percent |

The same estimator increased HashTags write throughput from 231 to 294 MB/s.

A direct OrderedFloat encoder increased it again to 379 MB/s. The Default baseline reached 1,316 MB/s.

The direct form kept the Euro write result neutral. It also retained both selected sizes.

The Default candidate can therefore use a specialized OrderedFloat selector.

The ALP candidate remains experimental until independent columns justify its analysis cost.

This split does not reject all selectors with expensive analysis. It reflects the current selection evidence for these two transforms.

### Corrected RangePacked bundle

The first controlled RangePacked pass exposed another constant-sample error.

A constant HashTags sample produced an infinite coarse range ratio.

The exact candidate then lost to the complete incumbent, but it blocked the existing Sparse tree.

The prefilter now rejects non-finite range ratios.

| Dataset | Current bytes | RangePacked bytes | Size change | Write time change | Read time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Euro2016 | 159,053,012 | 153,090,172 | -3.75 percent | +1.60 percent | -1.30 percent |
| HashTags | 179,382,828 | 178,141,132 | -0.69 percent | +0.89 percent | +0.69 percent |

The corrected 16-file geometric mean reduces size by 0.28 percent.

The timing differences in this two-file repeat remain within benchmark noise.

The focused Euro tree still decodes about 30 percent slower than its incumbent.

RangePacked therefore remains an experimental Default option.

### Decomposed fixed-bin bundle

The storage prototype now uses only composable arrays:

`OrderedFloat(IntMult(Dict(BitPacked(codes), starts), BlockResidual(offsets)))`

The scheme permits 1 through 64 bins.

The `IntMult` base is one, so decode uses addition without multiplication.

The exact sample estimator constructs this fixed child tree.

The writer therefore avoids generic integer recursion and its depth limit.

The first decomposed HashTags tree used 1,991,606 bytes.

The incumbent ALP-RD tree used 3,263,939 bytes.

This change reduced the column size by 39.0 percent.

The generic child execution reached 4,476 MB/s.

A nullable dictionary fast path reached 8,383 MB/s after it removed validity work from normal payloads.

An outer in-place decoder now preserves direct BlockResidual unpack.

It combines bin addition and ordered-float restoration in one pass.

The final decomposed tree reached 10,148 to 10,389 MB/s across two focused runs.

The direct fused RangePacked prototype reached 14,658 MB/s.

The ALP-RD incumbent reached 15,578 to 16,060 MB/s.

Scalar access used 312 to 323 ns for the decomposed tree and 150 to 158 ns for ALP-RD.

The decomposed tree retains bounded random access, but the constant factor remains material.

The matched 16-dataset compression pass produced these geometric-mean changes:

| Metric | Change versus Current Default |
| --- | ---: |
| File size | -0.056 percent |
| Write time | +1.172 percent |
| Read time | -0.268 percent |

HashTags file size fell from 179,382,860 bytes to 178,180,812 bytes.

Its matched write time fell by 0.82 percent, and its matched read time fell by 2.38 percent.

Those complete-file timing changes conflict with the slower selected column.

Treat them as aggregate noise until a larger repeated pass confirms them.

The first two million rows selected the new tree only for HashTags `interaction#received_at`.

The initial full files also changed size for Bimbo, CMSprovider, Euro2016, and Taxi.

A later chunk-level trace identified completed incumbent cascades as the cause.

The current broad size gain does not justify Default inclusion by itself.

The prototype temporarily added IntMult to a draft edition for file tests.

The focused branch removed IntMult and that edition membership.

Fast rejection remains a cost advantage, not an admission rule.

A scheme without fast rejection can enter Default when corpus evidence justifies its analysis cost.

### Completed incumbent cascade comparison

The first decomposed selector compared its complete fixed tree against each incumbent root estimate.

Several incumbents compressed their children after root selection.

A full Bimbo chunk trace found eight false fixed-bin selections in `Dev_proxima`.

The fixed-bin trees used 5.3 to 239.7 percent more bytes than the completed incumbent trees.

The selector now compares each retained candidate against the incumbent sample cascade.

The coarse model runs first. Clear losses do not pay for a second sample compression.

This comparison preserves the current cascade depth and excludes only the fixed-bin scheme.

The corrected Bimbo file uses 391,860,906 bytes in both configurations.

Its matched write throughput changes from 515.6 to 509.6 MB/s.

Its matched read throughput changes from 11,143.6 to 11,130.7 MB/s.

The focused HashTags file retains the useful selection:

| Metric | Current | Fixed-bin candidate | Change |
| --- | ---: | ---: | ---: |
| File size | 21,252,695 bytes | 20,718,724 bytes | -2.51 percent |
| Write throughput | 626.6 MB/s | 610.3 MB/s | -2.60 percent |
| Read throughput | 21,294.0 MB/s | 20,913.0 MB/s | -1.79 percent |

The matched 16-dataset file pass selected the fixed-bin tree only on HashTags.

| Metric | Geometric-mean change |
| --- | ---: |
| File size | -0.027 percent |
| Write time | +0.301 percent |
| Read time | -0.298 percent |

The read result is benchmark noise because only one file changes its encoded tree.

The corpus size result does not justify Default inclusion with the current decode and scalar costs.

Fast rejection remains an advantage. It does not replace evidence from complete candidate trees.

### Complete bundle controls

The compression benchmark exposes four numeric bundles:

- `prior-default`
- `block-residual`
- `current-default`
- `range-packed`

All bundles use the same file strategy construction.

The complete pass covered 16 datasets with three iterations per operation.

The BlockResidual bundle produced this geometric-mean change against Prior Default:

| Scope | Size change | Write throughput change | Read throughput change |
| --- | ---: | ---: | ---: |
| Numeric files | -2.6 percent | +0.0 percent | +2.0 percent |
| Real files | -2.5 percent | +0.0 percent | +0.9 percent |
| All files | -1.6 percent | -0.9 percent | +1.3 percent |

This result supports BlockResidual as the primary Default bundle.

The first FloatQuant pass exposed two selector errors.

HashTags produced a constant FloatQuant sample and an estimated ratio of 8,193.

Full analysis rejected the transform. The failed winner then blocked a compact Sparse tree.

FloatQuant now rejects a constant sample. The main compressor already handles true constant arrays.

Food selected FloatQuant from a 1.94 sample ratio. Its full ratio was 1.73.

ALP had a 1.80 sample ratio and a 2.51 full ratio on the same chunk.

A 1.10 FloatQuant factor retained every focused FloatQuant selection test and rejected this Food choice.

The corrected FloatQuant bundle produced this change against the BlockResidual bundle:

| Scope | Size change | Write throughput change | Read throughput change |
| --- | ---: | ---: | ---: |
| 16-file corpus | 0.0 percent | -0.9 percent | -0.8 percent |

Every file size matched exactly.

No selected-tree evidence supports a read difference. Treat the measured read change as noise.

The write result measures FloatQuant analysis without a size win in this corpus.

FloatQuant remains a production array and a BtrBlocks candidate.

Default inclusion requires evidence that justifies its analysis cost.

### Real Pco-gap FloatQuant pass

The current benchmark tested 33 real float columns across five inputs:

- CMS Open Payments.
- California Housing.
- NYC Taxi.
- CMS Provider 1.
- CMS Provider 2.

The pass compared `block-residual-bundle` with `proposed-default`.

FloatQuant selected no column.

Every proposed file size matched the BlockResidual bundle.

The geometric-mean encode throughput changed by +0.12 percent.

The geometric-mean decode throughput changed by -0.06 percent.

Both throughput changes are benchmark noise because the selected trees are identical.

Compact retains material size gains on many float columns in these files.

The Pco mode profile attributes those gaps to classic bins, IntMult, and FloatMult.

This corpus does not expose a FloatQuant win under the current factor.

The complete synthetic `f32` and `f64` trees produce strict size and throughput wins.

The complete corpus shows no selected FloatQuant tree and no stable analysis-cost signal.

Retain FloatQuant as an opportunistic option under the calibrated 1.10 factor.

### Wider-integer BlockResidual factor sweep

The final sweep compares wider-integer access cost factors of 1.00, 1.02, and 1.05.

Each compression pass uses six complete Public BI files and three iterations per operation.

The access pass requests about 100 rows through cached file handles. Each pattern runs for five seconds.

| Factor | Size geometric mean | Total bytes | Compression time geometric mean | Decompression time geometric mean | Correlated access | Uniform access |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.02 | -0.25 percent | +0.44 percent | -0.46 percent | +2.17 percent | +3.93 percent | +1.88 percent |
| 1.05 | +0.92 percent | +1.49 percent | -0.05 percent | +5.04 percent | +1.21 percent | +0.03 percent |

The first candidate access passes contain large Bimbo and CMS correlated outliers.

Reverse-order repeats remove those outliers. The table uses the repeated correlated results.

The 1.02 factor diverts enough Food trees to reduce geometric-mean size by 0.25 percent.

It increases total bytes by 0.44 percent. It also reduces bulk decode and file-access throughput.

The 1.05 factor increases both geometric-mean size and total bytes. It provides no stable access gain.

Retain the 1.00 wider-integer factor. The raw ratio floor and patch cost remain sufficient.

### Final selector factors

The integer BlockResidual scheme keeps a 1.05 raw-ratio floor. It applies a 1.12 access cost factor to 8-bit estimates.

This policy resolves the CMS threshold miss and produces a 2.94 percent complete-corpus size gain.

The Ordered BlockResidual scheme keeps its 1.05 raw floor and 1.02 decode factor.

The nonlinear patch cost rejects the known dense-patch and slow HashTags trees.

The 1.02 and 1.05 trials provide no evidence for a wider-integer factor.

The FloatQuant scheme keeps its 1.10 factor.

The Food sample mismatch requires a factor above 1.078 to retain its smaller ALP tree.

The 1.10 value adds a small margin and retains every strict synthetic `f32` and `f64` win.

The 33-column Pco-gap pass exposes no missed real FloatQuant candidate.

The 8-bit factor rejects the weakest measured gain. It retains three real quantized GloVe variants with 16 to 22 percent savings.

### Default bundle options

The final numeric pass uses complete files and three iterations per operation.

It covers Taxi, GloVe, Arade, Bimbo, CMSprovider, Euro2016, Food, and HashTags.

The BlockResidual bundle produces these changes against Prior Default:

| Dataset | Size | Write throughput | Read throughput |
| --- | ---: | ---: | ---: |
| Taxi | -6.35 percent | -0.81 percent | +7.23 percent |
| GloVe | 0.00 percent | -11.09 percent | +0.15 percent |
| Arade | -2.39 percent | -1.71 percent | +5.70 percent |
| Bimbo | -8.90 percent | -1.56 percent | -5.49 percent |
| CMSprovider | -7.94 percent | -1.40 percent | -8.51 percent |
| Euro2016 | -2.71 percent | +8.26 percent | -1.98 percent |
| Food | -7.79 percent | +1.95 percent | +6.08 percent |
| HashTags | -3.22 percent | +10.67 percent | -7.45 percent |
| Geometric mean | -4.96 percent | +0.34 percent | -0.71 percent |

Total bytes decrease by 6.18 percent across the eight files.

Total write time increases by 0.49 percent. Total read time increases by 3.39 percent.

The GloVe write result conflicts with adjacent runs that reverse the difference.

GloVe retains the same tree and size. Treat its write result as benchmark noise.

The remaining eight files contain two TPC-H comment layouts and six nested wide-table cases.

Both TPC-H files become 3.45 percent smaller through BlockResidual descendants.

All six wide-table files retain identical sizes.

Across all 16 files, the BlockResidual bundle produces these changes:

| Metric | Geometric-mean change | Aggregate change |
| --- | ---: | ---: |
| Size | -2.94 percent | -5.30 percent |
| Write throughput | -1.22 percent | -0.60 percent |
| Read throughput | -1.21 percent | -2.62 percent |

The evidence supports three bundle options:

| Option | Schemes | Scope | Comparison | Size | Write throughput | Read throughput |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Conservative | BlockResidual | 16 files | Prior Default | -2.94 percent | -1.22 percent | -1.21 percent |
| Opportunistic | BlockResidual and FloatQuant | 16 files | Conservative | 0.00 percent | Noise | Noise |
| Fixed-bin experiment | Opportunistic plus RangePacked | 8 numeric files | Opportunistic | -0.054 percent | -1.03 percent | Noise |

The opportunistic option is the leading Default bundle.

It adds strict synthetic FloatQuant wins without a measured corpus size regression.

The opportunistic option selects no FloatQuant tree in these eight files.

Focused synthetic `f32` and `f64` inputs still produce strict FloatQuant size and throughput wins.

The fixed-bin option changes only HashTags `interaction#received_at`.

Its numeric subset is 5.99 percent smaller, but the selected tree has material access costs.

The tree decodes at about 10.1 GB/s instead of 15.6 to 16.1 GB/s.

Scalar access takes 312 to 323 ns instead of 150 to 158 ns.

The absolute decode rate remains valid for Default consideration.

RangePacked remains an active experiment until the broad corpus establishes its coverage and total cost.

### Default bundle random access

The random-access benchmark now writes one Vortex file for each numeric bundle.

The cached-handle pass uses correlated and uniform patterns with about 100 requested rows.

| Dataset and pattern | Prior Default latency | Current Default latency | Change |
| --- | ---: | ---: | ---: |
| Taxi correlated | 0.609 ms | 0.624 ms | +2.45 percent |
| Taxi uniform | 2.760 ms | 2.807 ms | +1.68 percent |
| Nested lists correlated | 0.154 ms | 0.148 ms | -3.95 percent |
| Nested lists uniform | 0.922 ms | 0.922 ms | -0.06 percent |
| Nested structs correlated | 0.216 ms | 0.220 ms | +1.94 percent |
| Nested structs uniform | 0.532 ms | 0.544 ms | +2.20 percent |
| Geometric mean | — | — | +0.69 percent |

The six standard patterns remain within four percent of Prior Default.

Taxi file size decreases by 6.35 percent. Nested-list file size decreases by 0.71 percent.

The nested-struct files have identical sizes.

The legacy six-index Taxi case improves from 1.587 ms to 1.325 ms.

This small target is more sensitive to run noise than the 100-row patterns.

BlockResidual, Current Default, and RangePacked remain within 2.5 percent on both Taxi 100-row patterns.

The Public BI pass uses six complete files and cached handles. Each pattern requests about 100 rows.

The first pass exposed repeated BlockResidual payload validation during each slice operation.

The source array already passed validation. The slice only changes its logical bounds.

The slice path now constructs the narrowed array without repeated payload validation.

The valid patch-position branch now avoids a non-inlined fallible call for each patch.

A focused CMS correlated pass decreased from 1.127 ms to 0.964 ms after both changes.

The Prior Default control changed from 0.991 ms to 0.954 ms across the same runs.

The CMS gap therefore decreased from 13.7 percent to 1.1 percent.

Two new passes use five-second target windows. The second pass uses reverse dataset order.

| Dataset | File size | Correlated latency | Uniform latency |
| --- | ---: | ---: | ---: |
| Arade | -2.39 percent | -15.75 to -6.11 percent | -5.11 to +2.75 percent |
| Bimbo | -7.54 percent | -8.28 to +1.96 percent | -1.09 to +7.99 percent |
| CMSprovider | -7.86 percent | -2.22 to +9.22 percent | -6.64 to +11.82 percent |
| Euro2016 | -2.44 percent | +1.55 to +2.64 percent | -1.76 to +3.54 percent |
| Food | -6.19 percent | -29.89 to -15.70 percent | -2.83 to +6.97 percent |
| HashTags | -2.56 percent | +3.23 to +6.85 percent | +8.34 to +13.26 percent |

The file-size geometric mean still decreases by 4.86 percent. Aggregate bytes still decrease by 5.81 percent.

Across both passes, correlated latency decreases by 5.05 percent geometrically. Uniform latency increases by 2.91 percent.

The combined latency geometric mean decreases by 1.15 percent.

The individual file values remain noisy. The aggregate result removes the earlier random-access trade for the conservative bundle.

The selector factors remain unchanged.

### Broad fixed-bin coverage pass

The Pco mode scan covers 53 float columns across 12 real datasets.

It uses 524,288 rows per large column. This limit includes two complete Pco chunks.

Pco selects classic bins without Delta on 41 chunks. It selects classic bins with consecutive Delta on another 17 chunks.

Four columns use direct classic bins without Delta:

| Dataset and column | Compact gap | Maximum Pco bins |
| --- | ---: | ---: |
| HashTags `interaction#received_at` | 65.2 percent | 36 |
| Taxi `airport_fee` | 53.6 percent | 4 |
| Euro2016 `subjectivity_confidence` | 51.1 percent | 39 |
| Taxi `tips` | 45.4 percent | 17 |

The current corpus therefore contains the direct fixed-bin distribution.

Another 21 columns use classic bins only below an outer transform such as ALP.

Fifteen of these columns have a Compact gap of at least ten percent.

Twelve also use at most 64 Pco bins. Six have a Compact gap of at least 20 percent.

The complete bundle pass uses 18 files and three iterations per operation.

It adds the CMS Payments and Twitter Pco inputs to the standard 16-file corpus.

| Fixed-bin change against opportunistic | Geometric mean | Aggregate |
| --- | ---: | ---: |
| Size | -0.024 percent | -0.034 percent |
| Write throughput | -0.45 percent | Noise |
| Read throughput | -0.65 percent | Noise |

Only HashTags changes its file size. The other 17 files retain identical sizes.

The HashTags file becomes 0.429 percent smaller.

Its selected `interaction#received_at` tree changes from 3,263,939 bytes to 1,991,606 bytes.

The column becomes 39.0 percent smaller. Decode throughput changes from 15.79 GB/s to 11.44 GB/s.

Scalar access changes from 165 ns to 322 ns.

Fresh files from the same run produce this cached file-access result:

| Pattern | Opportunistic | Fixed-bin | Change |
| --- | ---: | ---: | ---: |
| Correlated | 1.978 ms | 2.009 ms | +1.58 percent |
| Uniform | 12.142 ms | 12.018 ms | -1.02 percent |

The selected tree passes the absolute decode and bounded-access bar.

The current direct scheme remains niche. Its full-suite size gain does not justify Default inclusion alone.

The nested Pco evidence motivated the pure integer fixed-bin prototype.

The experiment added a pure integer scheme that constructs this tree:

`IntMult(base=1, Dict(BitPacked(codes), starts), BlockResidual(offsets))`

ALP selects it through normal child recursion.

Keep the black-box RangePacked array outside Default.

### Pure integer fixed-bin result

The prototype constructs this complete integer tree:

`IntMult(base=1, Dict(BitPacked(codes), starts), BlockResidual(offsets))`

It supports every signed and unsigned integer type.

Signed inputs flip the sign bit before the bin fit. The tree restores starts and offsets with modular addition.

Tests cover every integer width, nullable values, signed-domain boundaries, and composition below ALP.

The bin trainer previously used one fixed stride. Periodic inputs could alias that stride and hide complete clusters.

A deterministic offset within each sample stratum now covers periodic phases. One regression test covers four interleaved clusters.

The first selector version examined every integer recursion site.

Across nine available datasets, this version reduced write throughput by 6.85 percent geometric mean.

An ALP-child gate reduced the write loss to 2.78 percent geometric mean.

The focused pass used 524,288 rows per dataset.

It covered Arade, Bimbo, both CMS Provider files, Euro2016, Food, HashTags, CMS Payments, and GloVe embeddings.

The fixed-bin scheme selected no tree. Every encoded size matched the Opportunistic bundle exactly.

This result agrees with the earlier forced-tree measurements.

One generic offset child mixes residuals from bins with different local widths.

BlockResidual then pays for local width variation and occasional cross-bin outliers.

The Pco bin model retains one offset width per bin. The generic tree loses that advantage.

The pure integer fixed-bin scheme therefore does not enter Default.

Keep the implementation as an experimental reference. Do not spend more selector time on this tree without new evidence.

The direct OrderedFloat fixed-bin scheme remains a separate experiment.

It still reduces HashTags `interaction#received_at` by 39.0 percent. Its complete file gain remains 0.429 percent.

### Compact and Parquet size constraint

The format pass uses three iterations per operation.

| Scope and comparison for Current Default | Size | Write time | Read time |
| --- | ---: | ---: | ---: |
| 8 numeric files versus Compact | +38.66 percent | -13.42 percent | -66.63 percent |
| 8 numeric files versus Parquet with Zstd | -1.11 percent | -65.37 percent | -96.08 percent |
| Complete 16 files versus Compact | +21.85 percent | -11.69 percent | -55.35 percent |
| Complete 16 files versus Parquet with Zstd | -1.67 percent | -34.80 percent | -85.82 percent |

Current Default remains 1.67 percent smaller than Parquet with Zstd across the complete corpus.

It writes 1.5 times faster and reads 7.1 times faster than Parquet with Zstd.

Compact remains 17.9 percent smaller than Current Default across the complete corpus.

Current Default writes 13.2 percent faster and reads 2.2 times faster than Compact.

Quick rejection reduces analysis cost and strengthens a candidate.

It is not an admission rule. A costly candidate requires stronger size and throughput evidence.

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

### Centered block residual prototype

Euro2016 `subjectivity_confidence` contains a large cluster near `1.0`.

About 36.5 percent of the first two million values equal `1.0`. Many other values sit close to that value.

The prototype selects one median reference per 1,024-value block. It ZigZag-encodes each ordered-float distance from that reference.

BlockResidual stores the transformed distances. The decoder fuses residual reconstruction, inverse ZigZag, and the ordered-float inverse.

Null values use the block reference as their payload. They produce zero residuals and keep logical positions.

The prototype tested extra patch costs from 16 through 96 bits. A 32-bit cost gives the best measured size and speed tradeoff.

| Euro subjectivity tree | Bytes | Encode MB/s | Decode MB/s | Scalar ns |
| --- | ---: | ---: | ---: | ---: |
| Current Default | 14,234,326 | 358 | 22,664 | 511 |
| `OrderedFloat(BlockResidual)` | 13,293,263 | 2,171 | 24,128 | 196 |
| Centered block residual | 13,071,137 | 1,068 | 14,028 | Direct codec: 12 |
| Compact Pco | 6,860,457 | 424 | 2,711 | 17,582 |

The centered form uses 1.7 percent fewer bytes than ordinary BlockResidual.

Its decode throughput is 41.9 percent lower than ordinary BlockResidual. Its encode throughput is also about half as fast.

The result rejects centered BlockResidual for Default. The extra transform does not recover enough of the Compact gap.

Ordinary BlockResidual gives a strict size and throughput win on this column. The final selector calibration must include this case.

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
- Initially excluded direct 8-bit and 16-bit candidates.
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
- Added two-child FloatQuant benchmarks for `f16`, `f32`, and `f64`.
- Confirmed strict complete-tree wins for the selected `f32` and `f64` cases.
- Confirmed that the `f16` selector retains its smaller dictionary tree.
- Added Compact as a first-class format in the compression benchmark.
- Compared the prior Default, proposed Default, Compact, and Parquet across all 16 local datasets.
- Attributed the largest Compact numeric gaps to Pco modes at the column level.
- Rejected byte-aligned fixed bins after a direct size and throughput comparison.
- Rejected grouped fixed bins after Euro2016 and CMS comparisons.
- Rejected packed multi-reference blocks after patched and patch-free comparisons.
- Rejected three bounded IntMult remainder layouts on GloVe.
- Rejected the dense-remainder IntMult layout on CMS Payments.
- Rejected pure IntMult on a no-Delta CMS standard-deviation column.
- Rejected centered block residuals after a complete Euro subjectivity comparison.
- Tested dictionary codes in descending value frequency, then removed the policy with IntMult.
- Tested float frequency counts, then restored the existing distinct-value statistics.
- Reopened BlockResidual descendants for integer and float dictionary codes.
- Validated the ranked Dict policy across eight numeric datasets.
- Audited constructor and deserialization validation for all three production arrays.
- Added round-trip tests for every supported integer and float type.
- Added single-encoding benchmarks for every supported integer and float type.
- Added packed-child IntMult benchmarks for every unsigned integer type.
- Added fused IntMult decode for aligned, patch-free BitPacked children.
- Added `f16` support to OrderedFloat, FloatQuant, and both Default schemes.
- Verified zero-secondary and nonzero-secondary `f16` FloatQuant trees.
- Added exact RangePacked size estimates without packed payload construction.
- Added a coarse RangePacked rejection model over the existing one-percent sample.
- Added a locality rejection test that retains Euro and rejects clear BlockResidual fits.
- Updated stale golden trees after the BlockResidual payload and selector changes.
- Added four numeric bundle controls to the complete compression benchmark.
- Tested explicit winner result events, then removed the unused trace event.
- Added optional chunk-level encoding trees to the focused compressor benchmark.
- Rejected constant FloatQuant samples before exact sample encoding.
- Added and calibrated a 1.10 FloatQuant selection factor.
- Replaced FloatQuant sample packing with an exact buffer-size estimate.
- Added isolated FloatQuant rejection controls for all float widths.
- Added adjusted-ratio near-miss controls for `f32` and `f64`.
- Re-enabled 16-bit BlockResidual after absolute-throughput and corpus revalidation.
- Enabled an 8-bit BlockResidual selector trial with selected and rejected synthetic controls.
- Removed the Food and HashTags FloatQuant size regressions.
- Revalidated the BlockResidual and FloatQuant bundles across 16 files.
- Rejected constant RangePacked samples before exact sample encoding.
- Revalidated RangePacked against the corrected current bundle.
- Compared retained fixed-bin candidates against the completed incumbent sample cascade.
- Removed every known Bimbo, CMSprovider, Euro2016, and Taxi fixed-bin false win.
- Tested FloatQuant on 33 real float columns from five Pco-gap inputs.
- Revalidated the final bundles across eight numeric files and all 16 corpus files.
- Revalidated Current Default against Compact and Parquet with Zstd across all 16 files.
- Added numeric bundle controls to the random-access benchmark.
- Compared cached random access across Taxi and two nested datasets.
- Added `i8` and `u8` support to the focused Parquet and scalar benchmarks.
- Validated 8-bit selection with quantized real GloVe values.
- Added local Parquet inputs to the complete compression benchmark.
- Added local Parquet inputs to the random-access benchmark.
- Measured 8-bit BlockResidual file access on accepted and boundary cases.
- Measured complete-file random access for six Public BI datasets.
- Added a fused `f16` decoder for `OrderedFloat(BlockResidual)`.
- Added a fused implicit-zero `f64` decoder for `FloatQuant(FoR(BitPacked))`.
- Rejected 1.02 and 1.05 wider-integer BlockResidual access cost factors.
- Added a fast Pco mode-only profile path.
- Profiled Pco modes across 53 float columns from 12 real datasets.
- Compared opportunistic and fixed-bin bundles across 18 complete files.
- Measured the selected HashTags fixed-bin tree and fresh cached file access.
- Added the pure integer fixed-bin scheme for ALP child recursion.
- Fixed alias errors in the deterministic fixed-bin sample.
- Rejected the integer fixed-bin tree after a focused nine-dataset pass.

The Pco mode profile and the quotient and remainder experiments are complete.

The composable IntMult follow-up is complete. The tested IntMult trees remain outside Default.

The nonzero-secondary FloatQuant implementation and focused validation are complete.

The specialized OrderedFloat and ALP trees now use registered fused parent kernels.

The pure integer fixed-bin experiment is complete.

It selected no integer tree and reduced write throughput. The focused branch no longer contains either fixed-bin scheme.

## Final calibration

The final pass compared 17 datasets with five timed iterations per operation.

The corpus included GloVe `f32` embeddings and OpenAI-on-C4 `f64` embeddings.

| Bundle or format | Aggregate bytes | Encode time | Decode time |
| --- | ---: | ---: | ---: |
| Prior Default | 3,025,406,960 | 10.928 seconds | 0.424 seconds |
| Opportunistic | 2,672,321,904 | 10.487 seconds | 0.422 seconds |
| Compact | 2,027,179,944 | 11.655 seconds | 1.082 seconds |
| Parquet with Zstd | 2,718,861,135 | 24.830 seconds | 5.635 seconds |

Against Prior Default, Opportunistic reduced aggregate size by 11.67 percent.

It reduced aggregate encode time by 4.04 percent and aggregate decode time by 0.56 percent.

The geometric means improved by 5.69 percent for size, 10.49 percent for encode time, and 4.99 percent for decode time.

Opportunistic produced 1.71 percent fewer aggregate bytes than Parquet with Zstd.

Its geometric-mean size ratio against Parquet with Zstd was 0.980.

The focused cached-access corpus included five tabular datasets and the OpenAI vector dataset.

Opportunistic reduced geometric-mean scalar latency by 10.97 percent and aggregate scalar latency by 10.20 percent.

OpenAI-on-C4 selected `FloatQuant(FoR(BitPacked))`.

That tree reduced size by 40.5 percent and improved bulk decode time against Prior Default.

GloVe retained its prior tree. Its result still identifies an entropy gap inside the ALP integer child.

The fixed-bin bundle reduced aggregate size by only 0.025 percent against Opportunistic.

It changed only the HashTags file in the complete corpus, where it reduced file size by 0.43 percent.

The selected HashTags column was 35.0 percent smaller, but that narrow win did not justify the scheme cost.

## Final production decision

Use the Opportunistic bundle in Default.

The bundle contains these schemes:

- `BlockResidualScheme` for integers.
- `OrderedBlockResidualScheme` for floating-point values.
- `FloatQuantScheme` for floating-point values.

Retain these selector controls:

- A 1.05 minimum ratio for integer BlockResidual.
- A 1.05 minimum ratio and 1.02 decode factor for ordered BlockResidual.
- A 1.10 FloatQuant factor.
- A 1.12 factor for 8-bit BlockResidual.
- The nonlinear BlockResidual patch cost.
- The one-block rejection for integer BlockResidual.

Remove RangePacked and both fixed-bin schemes from the focused branch.

Keep RangeEntropy, BitSplit, and alternate residual models on the experimental branch.

Use ordinary child arrays for patches and exception payloads.

Test these items in future research:

1. Test nonzero-secondary FloatQuant on more real distributions.
2. Add more real float datasets where Pco beats ALP and ALP-RD.
3. Test a true ALP-RD child composition when a real selected tree exposes one.
4. Revisit fixed bins only when the corpus includes more applicable distributions.

## Pull request structure

Prepare one pull request for the three arrays, their schemes, fused FastLanes paths, and benchmark support.

Split the pull request only when review reveals a clear independent boundary.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## Adversarial review checkpoint

The review followed the merge from `origin/develop`.

It preserved the frozen August editions and added `core:2026.08.4` as a draft.

It added the array conformance suite to BlockResidual, OrderedFloat, and FloatQuant.

It added empty-buffer validation to OrderedFloat and FloatQuant deserialization.

It added signed coverage for the fused nonzero-secondary FloatQuant encoder.

It changed invalid FloatQuant references from a debug panic to an error.

It removed the Pco probe, the custom Pco API, abandoned compressor hooks, and public BlockResidual codec internals.

It removed unrelated dictionary frequency ranking and float frequency statistics.

It retained the three production candidates, their selectors, and their focused benchmarks.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
