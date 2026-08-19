# Native PCodec-inspired numeric compression

## Goal

Improve default numeric compression where ALP and ALP-RD lose to Pco.

Keep native Vortex arrays, bounded metadata, fast canonical decode, and bounded random access.

Pco remains a compression oracle. The new arrays do not preserve Pco byte compatibility.

## Current decision

Focus the production work on these encodings:

- `OrderedFloatArray`.
- `BlockResidualArray` for all integer types, with one reference per 1,024-value block.
- `FloatQuantArray`.

Keep `FloatQuantScheme`, `OrderedBlockResidualScheme`, and `BlockResidualScheme` as BtrBlocks candidates.

Remove `FloatMultArray` and `FloatMultScheme` from the focused branch.

Remove `RangeEntropyArray`, `RangeEntropyScheme`, and `BitSplitCodec` from the focused branch.

The `wm/pcodec-entropy-experiments` branch preserves the complete entropy and bit-split prototypes.

Do not add adjacent Delta, Delta-of-delta, Delta with lookback, or convolution Delta.

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

Child arrays store references, widths, offsets, packed words, patch positions, and patch high bits.

Fixed metadata stores only lengths and slice bounds. Variable tables remain in child arrays.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

The `OrderedFloat(BlockResidual)` execute kernel combines residual decode with the inverse float transform.

The residual payload uses the logical integer width. A 16-bit array uses 16-bit FastLanes pack and unpack operations.

The serialized residual payload remains a `u64` child. The logical type defines the packed word interpretation.

The production codec uses one reference per block. The multi-reference prototype did not justify its encode and decode costs.

## FloatQuantArray

`FloatQuantArray` splits each float into a primary quantum and a secondary adjustment.

Metadata stores only the split width. The array dtype stores the source type.

One or two child arrays store the integer latents.

The current BtrBlocks scheme accepts only a constant secondary. It uses a fixed `FoR(BitPacked)` primary tree.

A common path uses `FloatQuant(FoR(BitPacked))` for `f32` values stored in `f64` columns.
An absent secondary child represents zero low bits.

The array supports exact IEEE bit-pattern round trips, nulls, slices, scalar access, and serialization.

The automatic scheme accepts only `f64`. Native `f32` columns remain with ALP, ALP-RD, and existing schemes.

## Selection policy

`FloatQuantScheme` uses the normal sample comparison against ALP, ALP-RD, dictionary, sparse, and RLE schemes.

The sample builds a direct `FoR(BitPacked)` primary and uses an implicit-zero secondary.

The scheme rejects samples with a nonzero secondary. This avoids the recursive integer selector.

`OrderedBlockResidualScheme` uses eight locality-preserving sample blocks.

The ordinary BtrBlocks sample does not preserve the block-local float structure.

The residual scheme requires a 1.05 compression ratio. Its adjusted score includes a 1.02 decode-cost factor.

`BlockResidualScheme` uses the same locality probe for integer arrays.

The default scheme accepts only 32-bit and 64-bit integers. Direct 8-bit and 16-bit candidates cannot save enough absolute space.

The scheme does not run inside trial compression for an outer scheme. Generic 64-row samples do not preserve 1,024-row locality.

The selected outer scheme can still choose BlockResidual for its full child.

The integer selector divides the measured compression ratio by these decode-cost factors:

- 1.10 for 32-bit integers.
- 1.20 for 64-bit integers.

A 1.20 factor requires about 16.7 percent fewer estimated bytes. It does not require 20 percent fewer bytes.

The block planner adds 16 synthetic cost bits per patch. This cost favors wider packed residuals when many patches slow decode.

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
| Prior default | 14,057,966 | 553.6 | 10,944.5 | 208.7 |
| Default with FloatQuant | 8,753,920 | 504.9 | 14,795.4 | 145.5 |
| Compact | 6,051,737 | 280.0 | 4,022.0 | Not measured |

FloatQuant reduced size by 37.7 percent. It remained 44.7 percent larger than Compact.

Full compressor throughput decreased by 8.8 percent. Decode throughput increased by 35.2 percent.

Scalar access latency decreased by 30.3 percent.

The selected tree was `FloatQuant(FoR(BitPacked))` with an implicit-zero secondary.

The isolated tree compressed between 2,402 and 2,469 MB/s.

It decoded between 14,480 and 14,810 MB/s.

Compact Pco compressed the same input at 280 MB/s and decoded it at 4,022 MB/s.

The tree itself exceeded Compact throughput by at least 8.6 times for compression and 3.6 times for decode.

FloatQuant recovered 66.2 percent of the size gap between the prior default and Compact Pco.

The direct sample tree removed the recursive integer selector from the estimate and final compression paths.

FloatQuant meets the selected-column throughput limit on this input.

### FloatQuant with a nonzero secondary

The prototype changed the lowest bit for ten percent of the widened-`f32` values.

The fixed tree was `FloatQuant(FoR(BitPacked), BitPacked)`. The secondary used one bit per value.

| Configuration | Bytes | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: |
| Prior ALP-RD default | 14,057,966 | 11,430 | 208.5 |
| Two-child FloatQuant prototype | 9,004,032 | 8,375 | 192.2 |

The prototype reduced size by 36.0 percent. It was 2.9 percent larger than the zero-secondary tree.

The prototype recovered 64.1 percent of the size gap between ALP-RD and Compact Pco.

The prototype reduced decode throughput by 26.7 percent against ALP-RD. It reduced decode throughput by 43.6 percent against zero-secondary FloatQuant.

Scalar access remained competitive. The direct prototype tree compressed at 3,301 MB/s.

The default scheme rejects this form. A fused one-bit-secondary decode is the next experiment.

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
| `u64` BlockResidual | 5.00 | 36.43 | 125 |
| `u64` FoR plus BitPacked | 4.25 | 35.37 | 115 |
| `i16` BlockResidual | 1.38 | 31.12 | 125 |
| `i16` FoR plus BitPacked | 1.12 | 41.57 | 104 |

The first `i16` implementation unpacked through `u64` residuals. It decoded at 11.43 GB/s.

Native-width unpack increased `i16` decode throughput by 2.78 times.

The synthetic `i16` BlockResidual tree uses 1,793,784 bytes. The FoR plus BitPacked tree uses 3,500,000 bytes.

BlockResidual is 48.7 percent smaller on that input.

Its `i16` decode throughput is 25.1 percent lower. The default selector therefore excludes direct 8-bit and 16-bit candidates.

The BlockResidual estimator measures its sampled tree exactly, with all child arrays.

The incumbent and outer tree estimates remain approximate. Trial compression previously mis-ranked Dict and ALP on Taxi tips.

The outer-sample exclusion removed that error from the measured tree.

### Broad numeric revalidation

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

The direct narrow-type exclusion changed no selected tree in this corpus. The earlier 1.40 factor already rejected each direct `i16` candidate.

A FastLanes fused FoR decode trial did not improve `u64` throughput. It reduced `i16` throughput, so the implementation retains native unpack.

The zero-width residual path now writes base values and patches directly. It skips the scratch residual block.

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

Test these secondary candidates after the multi-reference prototype:

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

Complete these remaining steps:

1. Run the complete `bench-vortex` compression corpus.
2. Compare the geometric mean against Parquet with Zstd.
3. Reduce the analysis cost on short rejected columns.
4. Test a fused nonzero-secondary FloatQuant decode on selected gap columns.
5. Evaluate nonzero-secondary FloatQuant for the default compressor.
6. Investigate new schemes for real floats that Pco compresses better than ALP and ALP-RD.
7. Update this plan after each experiment.

## Pull request structure

Prepare one focused stack for `OrderedFloatArray`, `BlockResidualArray`, and `OrderedBlockResidualScheme`.

Prepare a separate stack for `FloatQuantArray` and `FloatQuantScheme`.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
