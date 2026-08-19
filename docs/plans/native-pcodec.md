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

Bounded metadata stores the source type and split width. One or two child arrays store the integer latents.

The BtrBlocks scheme compresses each present child through the normal cascade.

A common path uses `FloatQuant(FoR(BitPacked))` for `f32` values stored in `f64` columns.
An absent secondary child represents zero low bits.

The array supports exact IEEE bit-pattern round trips, nulls, slices, scalar access, and serialization.

The automatic scheme accepts only `f64`. Native `f32` columns remain with ALP, ALP-RD, and existing schemes.

## Selection policy

`FloatQuantScheme` uses normal sample comparison against ALP, ALP-RD, dictionary, sparse, and RLE schemes.

A strong constant split can use a direct `FoR(BitPacked)` primary and an implicit-zero secondary.

Other splits compress both children through the normal integer cascade.

`OrderedBlockResidualScheme` uses eight locality-preserving sample blocks.

The ordinary BtrBlocks sample does not preserve the block-local float structure.

The residual scheme requires a 1.05 compression ratio and a 1.02 win over the incumbent.

Both schemes remain eligible to displace ALP or ALP-RD when their adjusted size scores win.

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

| Configuration | Bytes | Compression MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Prior default | 14,057,966 | 550.4 | 11,310.8 |
| Default with FloatQuant | 8,753,920 | 244.9 | 14,664.3 |
| Compact | 6,051,640 | 175.4 | 4,019.0 |

FloatQuant reduced size by 37.7 percent. It remained 44.7 percent larger than Compact.

Full compressor throughput decreased by 55.5 percent. Decode throughput increased by 29.6 percent.

The selected tree was `FloatQuant(FoR(BitPacked))` with an implicit-zero secondary.

The isolated tree compressed at 2,503 MB/s and decoded at 15,070 MB/s.

Pco compressed the same input at 533 MB/s and decoded it at 3,973 MB/s.

The tree itself exceeded Pco throughput by 4.7 times during compression and 3.8 times during decode.

The BtrBlocks analysis and candidate path caused most of the full compressor cost.

FloatQuant cannot enter the default set until its selection path meets the compression throughput limit.

### OrderedFloat with BlockResidual on random walks

| Configuration | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Prior default | 12,255,488 | 577.9 | 12,869.9 | 161.24 |
| Default with the scheme | 10,425,690 | 516.1 | 18,019.7 | 129.06 |
| Compact Pco | 9,342,749 | 316.3 | 4,611.7 | Not measured |

The residual scheme saved 14.9 percent against the prior default. It remained 11.6 percent larger than Compact.

Decode throughput increased by 40.0 percent. Random access latency decreased by 20.0 percent.

Compression throughput decreased by 10.7 percent on the selected column.

The isolated tree compressed at 2,504 MB/s and decoded at 18,920 MB/s.

Pco compressed the same input at 648 MB/s and decoded it at 4,635 MB/s.

The selector rejected the scheme on Gaussian, lognormal, decimal, widened-f32, and four-cluster inputs.

The rejected locality probe reduced compression throughput by 0.5 to 2.2 percent.

The HashTags dataset selected this scheme for two columns.

It reduced the numeric subset by 6.0 percent with no isolated compressor regression.

### Broad dataset revalidation

The file benchmark compared the corrected default against the new schemes.

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

The FloatQuant probe reduced HashTags compression throughput by 2.4 percent without a selection.

This analysis cost needs optimization before FloatQuant can enter the default set.

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

Complete these remaining steps:

1. Optimize the FloatQuant selection path.
2. Repeat the broad file benchmark after that change.
3. Measure random access on two million values for every retained tree.
4. Add CMS Payments, r/place, and Twitter to the paper dataset matrix.
5. Compare the geometric mean against Parquet with Zstd.

## Pull request structure

Prepare one focused stack for `OrderedFloatArray`, `BlockResidualArray`, and `OrderedBlockResidualScheme`.

Prepare a separate stack for `FloatQuantArray` and `FloatQuantScheme`.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
