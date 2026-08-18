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

Bounded metadata stores the source type and split width. Two child arrays store the integer latents.

The BtrBlocks scheme compresses both children through the normal cascade.

A common path uses `FloatQuant(FoR(BitPacked), Constant)` for `f32` values stored in `f64` columns.

The array supports exact IEEE bit-pattern round trips, nulls, slices, scalar access, and serialization.

The automatic scheme accepts only `f64`. Native `f32` columns remain with ALP, ALP-RD, and existing schemes.

## Selection policy

`FloatQuantScheme` uses normal sample comparison against ALP, ALP-RD, dictionary, sparse, and RLE schemes.

A strong constant split can use a direct `FoR(BitPacked)` primary and a constant secondary.

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

The input contains 262,144 arbitrary `f32` values stored in an `f64` column.

| Configuration | Bytes | Compression MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Default without FloatQuant | 1,638,436 | 762.4 | 12,932.1 |
| Default with FloatQuant | 688,130 | 635.0 | 13,636.4 |
| Compact | 658,758 | 336.8 | 5,117.1 |

FloatQuant reduced size by 58.0 percent. It remained 4.5 percent larger than Compact.

Compression throughput decreased by 16.7 percent. Decode throughput increased by 5.4 percent.

The selected tree was `FloatQuant(FoR(BitPacked), Constant)`.

This result needs a larger throughput test. The compression-ratio result remains strong.

### OrderedFloat with BlockResidual on random walks

| Configuration | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Default without the scheme | 1,853,564 | 713.4 | 12,486.1 | 161.24 |
| Default with the scheme | 1,663,831 | 635.6 | 17,355.8 | 129.06 |
| Compact Pco | 1,551,142 | 296.2 | 3,777.2 | Not measured |

The residual scheme saved 10.2 percent against ALP-RD. It remained 7.3 percent larger than Pco.

Decode throughput increased by 39.0 percent. Random access latency decreased by 20.0 percent.

Compression throughput decreased by 10.9 percent on the selected column.

The selector rejected the scheme on Gaussian, lognormal, decimal, widened-f32, and four-cluster inputs.

The rejected locality probe reduced compression throughput by 0.5 to 2.2 percent.

This result needs several larger random walks and real time-series columns.

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

After the corrected ALP release, perform these steps:

1. Update the workspace ALP dependency.
2. Remove any remaining FloatMult compatibility code.
3. Confirm that corrected ALP replaces the historical Housing FloatMult selections.
4. Recheck FloatQuant on widened-f32 columns.
5. Recheck OrderedFloat with BlockResidual on random walks and time-series columns.
6. Repeat throughput and scalar-access measurements with at least two million rows.
7. Compare each candidate against corrected default, Compact, and Pco Auto.
8. Measure rejected-candidate analysis cost on every dataset.
9. Add CMS Payments, r/place, and Twitter to the paper dataset matrix.

Record size, encode throughput, decode throughput, and scalar-access latency for each selected column.

## Pull request structure

Prepare one focused stack for `OrderedFloatArray`, `BlockResidualArray`, and `OrderedBlockResidualScheme`.

Prepare a separate stack for `FloatQuantArray` and `FloatQuantScheme`.

Keep entropy, bit-split, and alternate residual models on `wm/pcodec-entropy-experiments`.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
