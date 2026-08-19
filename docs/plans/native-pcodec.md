# Native PCodec-inspired numeric compression

## Goal

Improve default numeric compression where ALP and ALP-RD lose to Pco.

Keep native Vortex arrays, bounded metadata, fast canonical decode, and bounded random access.

Pco remains a compression oracle. The new arrays do not preserve Pco byte compatibility.

## Current decision

Focus the production work on these encodings:

- `OrderedFloatArray`.
- `BlockResidualArray` with one reference per 1,024-value block.
- `FloatQuantArray`.

Keep `FloatQuantScheme` and `OrderedBlockResidualScheme` as BtrBlocks candidates.

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

`BlockResidualArray` divides unsigned integers into independent blocks of 1,024 values.

Each block stores one minimum value. Packed residuals store the difference from that minimum.

Rare wide residuals use sorted positions and packed high bits.

Scalar access uses one packed read and one binary search over the patch positions.

Child arrays store references, widths, offsets, packed words, patch positions, and patch high bits.

Fixed metadata stores only lengths and slice bounds. Variable tables remain in child arrays.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

The `OrderedFloat(BlockResidual)` execute kernel combines residual decode with the inverse float transform.

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

The residual scheme requires a 1.05 compression ratio and a 1.02 win over the incumbent.

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

### Broad dataset revalidation

The earlier file benchmark compared the corrected default against the new schemes.

| Dataset | Prior bytes | Proposed bytes | Size change |
| --- | ---: | ---: | ---: |
| Taxi | 439,806,364 | 439,806,364 | 0.000 percent |
| Air Quality | 15,848,420 | 15,848,420 | 0.000 percent |
| Arade | 143,343,828 | 143,343,828 | 0.000 percent |
| Euro2016 | 164,673,900 | 164,674,236 | +0.0002 percent |
| Food | 38,894,360 | 38,922,408 | +0.072 percent |
| HashTags | 195,035,692 | 194,753,372 | -0.145 percent |

Separate benchmark processes introduced low single-digit throughput noise.

The focused two-million-row run selected no new schemes in Taxi, Air Quality, Arade, Euro2016, or Food.

Ordered BlockResidual selected two HashTags columns. FloatQuant selected no columns in these datasets.

The earlier FloatQuant probe reduced HashTags compression throughput by 2.4 percent without a selection.

The direct sample tree removed that recursive analysis cost.

### Pcodec paper corpus

The focused benchmark reads the first two million numeric rows from each source.

Timestamps use their integer representation. The California Housing size uses its original 20,640 rows.

| Dataset | Input bytes | Prior bytes | Proposed bytes | Compact bytes | New selection |
| --- | ---: | ---: | ---: | ---: | --- |
| Air Quality | 42,834,636 | 16,645,494 | 16,645,494 | 4,298,741 | None |
| California Housing | 743,040 | 307,427 | 307,427 | 227,970 | None |
| r/place | 56,000,000 | 16,266,697 | 16,266,697 | 8,891,886 | None |
| NYC Taxi | 240,000,000 | 52,407,972 | 52,407,972 | 33,148,991 | None |
| Twitter follower graph | 32,000,000 | 7,226,752 | 7,226,752 | 3,661,121 | None |
| CMS Payments | 160,000,000 | 30,061,670 | 30,061,670 | 22,300,792 | None |

Air Quality, California Housing, and r/place match the paper input sizes.

The Taxi logical input matches the paper size. Vortex arrays also retain column validity.

The Twitter source uses the first two million official edges. The paper used an unspecified ID sort.

The Twitter result is not an exact size comparison. Independent column sort produced a 1,212,617-byte Compact result.

The six inputs selected no new schemes. Four inputs contain no eligible `f64` column.

Taxi contains `f64` columns, but both new schemes lost their sample comparisons.

CMS Payments contains one `f64` column. ALP won that column.

The current CMS source revision differs from the paper source snapshot.

The CMS prior and proposed compressors both encoded at approximately 1,252 MB/s.

Their decode results differed by less than two percent. They produced identical trees.

## General real-float follow-up

Target float columns where Compact Pco materially beats the complete default cascade.

The default baseline includes ALP, ALP-RD, and compression of their children.

Use a ten-percent Pco size advantage as the initial corpus filter. This filter is not a product threshold.

Measure gap recovery as `(default bytes - candidate bytes) / (default bytes - Pco bytes)`.

Retain a prototype only if it recovers a material gap across unrelated real datasets.

The retained schemes cover lower-precision values in wider floats and locally narrow ordered ranges.

They do not replace ALP-RD for generic non-decimal floats. No measured column moved from ALP to a new scheme.

The first general prototype will use bounded multiple references per 1,024-value block.

Each block will test one, two, or four references. Each value will store a small reference ID and one packed residual.

All references will share one residual width per block. Sparse high-bit patches will contain rare outliers.

Direct access will read one reference ID, one residual, and an optional patch. The one-reference form matches `BlockResidualArray`.

This design approximates Pco bins without entropy coding. It also avoids a rank query into separate variable-width bin streams.

Test these secondary candidates after the multi-reference prototype:

- Use ordered-bit XOR suffixes instead of arithmetic residuals.
- Use sign and exponent prefixes as fixed reference candidates.
- Use block-local ALP-RD split widths.
- Use independent entropy microblocks only in Compact.

Build a gap corpus from columns where Compact Pco materially beats ALP-RD.

Classify each gap by exponent count, prefix count, local range, low-bit entropy, and outlier rate.

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

Complete these remaining steps:

1. Run the broad Vortex compression corpus after the selector change.
2. Compare the geometric mean against Parquet with Zstd.
3. Build the Pco-versus-ALP-RD gap corpus.
4. Prototype the bounded multi-reference residual codec.
5. Test a fused nonzero-secondary decode on qualifying gap columns.
6. Update this plan after each experiment.

## Pull request structure

Prepare one focused stack for `OrderedFloatArray`, `BlockResidualArray`, and `OrderedBlockResidualScheme`.

Prepare a separate stack for `FloatQuantArray` and `FloatQuantScheme`.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
