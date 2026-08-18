# Native PCodec-inspired numeric compression

## Goal

Improve default numeric compression where ALP and ALP-RD lose to Pco.

Keep native Vortex arrays, bounded metadata, child arrays, and fast canonical decode.

Pco remains a compression oracle. The new arrays do not preserve Pco byte compatibility.

## Decision

Add `FloatQuantArray` and `FloatQuantScheme` to default BtrBlocks compression for `f64` arrays.

Add `OrderedFloatArray`, `BlockResidualArray`, and their BtrBlocks scheme to the default compressor.

Keep `BitSplitArray` and `FloatMultArray` as prototypes.

Keep `RangeEntropyArray` and `RangeEntropyScheme` experimental. Their current full-decode and compression costs fail the throughput gate.

Keep FastLanes Delta unchanged. Its current lane-stride transform does not replace Pco adjacent Delta on the measured data.

Do not add Delta-of-delta, Delta with lookback, convolution Delta, or Pco byte compatibility.

## Implemented arrays

### OrderedFloatArray

`OrderedFloatArray` maps IEEE float bits to unsigned integers with the same order.

The transform preserves every bit pattern. It also preserves nulls and signed zero values.

The array stores one unsigned child. An empty metadata record identifies the transform.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

### BlockResidualArray

`BlockResidualArray` divides unsigned integers into independent blocks of 1,024 values.

Each block stores one minimum value. Packed residuals store the difference from that minimum.

Rare wide residuals use sorted positions and packed high bits. Scalar access uses one packed read and one binary search.

Nine child arrays store references, widths, offsets, packed words, patch positions, and patch high bits.

The fixed metadata record stores only lengths and slice bounds. Variable tables do not use metadata.

The array supports canonical decode, scalar access, slice reduction, serialization, and validation.

The `OrderedFloat(BlockResidual)` execute kernel fuses residual decode with the inverse float transform.

### RangeEntropyArray

`RangeEntropyArray` stores entropy-coded range-bin identifiers and fixed-width offsets.

The logical dtype remains the source primitive dtype. The codec maps each value to an ordered unsigned latent.

The fixed metadata record stores these fields:

- Physical type.
- ANS table log.
- Restart block length.
- Logical slice bounds.

Child arrays store these variable tables:

- Bin lower bounds.
- Bin offset widths.
- Quantized ANS weights.
- Restart block byte offsets.
- Validity.

One payload buffer stores the ANS stream and packed offsets. Independent restart blocks bound scalar access work.

The array supports canonical decode, slice reduction, scalar access, serialization, and validation.

### FloatQuantArray

`FloatQuantArray` splits each float into a primary quantum and a secondary adjustment.

Bounded metadata stores the source type and split width. Two child arrays store the integer latents.

The BtrBlocks scheme compresses both children through the normal cascade.

A common path uses `FloatQuant(FoR(BitPacked), Constant)`. This path targets `f32` values stored in `f64` columns.

The array supports exact IEEE bit-pattern round trips, nulls, slices, scalar access, and serialization.

The automatic scheme currently accepts only `f64`. Native `f32` columns keep the current ALP and dictionary choices.

## Selection policy

FloatQuant uses normal sample comparison against ALP, ALP-RD, dictionary, sparse, and RLE schemes.

The scheme does not claim an unconditional win. This rule prevents poor choices on low-cardinality integer-valued floats.

The full FloatQuant analysis runs only after the sample selects the scheme. Unselected `f64` columns pay only the sample cost.

Callers can disable the default scheme with this builder configuration:

```rust
let compressor = BtrBlocksCompressorBuilder::default()
    .exclude_schemes([FloatQuantScheme.id()])
    .build();
```

BlockResidual uses eight locality-preserving sample blocks. The ordinary BtrBlocks sample destroys the block-local float structure.

The scheme requires a 5 percent compression ratio. It also requires a 2 percent win over the current best scheme.

Callers can disable this scheme with this builder configuration:

```rust
let compressor = BtrBlocksCompressorBuilder::default()
    .exclude_schemes([OrderedBlockResidualScheme.id()])
    .build();
```

RangeEntropy also uses sample comparison. It remains outside `ALL_SCHEMES`.

## Performance gate

The default candidate must meet these limits:

- Compression throughput can regress by at most 20 percent.
- Full-decode throughput can regress by at most 20 percent.
- A selected candidate must reduce size materially.
- The result must approach Compact size on its target data.

The release benchmark excludes source-array construction and storage I/O.

## Synthetic target result

The widened-f32 input contains 262,144 arbitrary `f32` values stored as `f64`.

| Configuration | Bytes | Compression MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Default without FloatQuant | 1,638,436 | 762.4 | 12,932.1 |
| Default with FloatQuant | 688,130 | 635.0 | 13,636.4 |
| Compact | 658,758 | 336.8 | 5,117.1 |

FloatQuant reduces size by 58.0 percent. It remains 4.5 percent larger than Compact.

Compression throughput regresses by 16.7 percent. Decode throughput improves by 5.4 percent.

The selected tree is `FloatQuant(FoR(BitPacked), Constant)`.

### FloatQuant and RangeEntropy composition

The stacked configuration leaves the widened-f32 tree and byte size unchanged.

RangeEntropy loses selection for both FloatQuant children. The primary already uses FoR and bitpacking, while the secondary is constant zero.

A repeated synthetic run reduced compression throughput by 4.9 percent because RangeEntropy still incurred sample work.

No measured dataset selected RangeEntropy beneath FloatQuant. RangeEntropy sometimes wins as a root alternative when FloatQuant loses.

## Pco paper data

The benchmark reads source columns before timing and limits Parquet inputs to two million rows.

The aggregate values below cover every supported numeric column in each downloaded input.

### Size

| Dataset | Without new float schemes | Default | Default plus RangeEntropy | Compact | Pco Auto |
| --- | ---: | ---: | ---: | ---: | ---: |
| California Housing | 339,682 | 339,682 | 319,705 | 230,197 | 242,414 |
| April 2023 HVFHV Taxi | 28,540,793 | 28,540,793 | 23,331,922 | 21,127,765 | 21,994,290 |
| Air Quality | 6,135,414 | 6,135,414 | 5,149,775 | 2,311,526 | 2,303,523 |

FloatQuant and OrderedFloat with BlockResidual make no selection on these inputs. Their size and decode paths remain unchanged.

RangeEntropy reduces Taxi size by 18.3 percent and Air Quality size by 16.1 percent.

RangeEntropy remains 10.4 percent larger than Compact on Taxi. It remains 122.8 percent larger on Air Quality.

### Compression throughput

| Dataset | Without new float schemes | Without BlockResidual | Default | Default plus RangeEntropy | Compact | Pco Auto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| California Housing | 205.8 MB/s | 208.5 MB/s | 210.0 MB/s | 114.8 MB/s | 72.1 MB/s | 206.3 MB/s |
| April 2023 HVFHV Taxi | 886.2 MB/s | 838.4 MB/s | 828.8 MB/s | 421.7 MB/s | 432.3 MB/s | 574.0 MB/s |
| Air Quality | 548.2 MB/s | 549.1 MB/s | 550.5 MB/s | 318.3 MB/s | 303.7 MB/s | 415.2 MB/s |

The BlockResidual locality probe changes compression throughput by 1.2 percent or less on these inputs.

RangeEntropy reduces compression throughput by 42 to 49 percent. This result fails the gate.

### Decode throughput

| Dataset | Without new float schemes | Without BlockResidual | Default | Default plus RangeEntropy | Compact | Pco Auto |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| California Housing | 12,392.7 MB/s | 12,531.9 MB/s | 12,558.4 MB/s | 2,221.6 MB/s | 2,336.9 MB/s | 2,413.8 MB/s |
| April 2023 HVFHV Taxi | 23,963.4 MB/s | 23,879.0 MB/s | 24,200.2 MB/s | 3,809.3 MB/s | 4,825.0 MB/s | 3,644.8 MB/s |
| Air Quality | 25,013.7 MB/s | 24,807.0 MB/s | 25,371.2 MB/s | 1,471.8 MB/s | 2,088.0 MB/s | 2,092.8 MB/s |

RangeEntropy decode is 5.7 to 17.2 times slower than the current lightweight default.

RangeEntropy beats Pco Auto decode on Taxi. It loses to Pco Auto decode on Housing and Air Quality.

The native implementation does not meet the default full-decode requirement.

## Pco encode cost

The probe separates Pco into a prepare stage and an emit stage.

The prepare stage selects the mode and Delta transform. It also builds the final bins and ANS model.

The emit stage finds each value's bin, creates offsets, runs reverse ANS, and writes the bit stream.

The forced test supplies the mode and Delta transform that `Auto` selected. Both paths produce identical bytes.

| Dataset | Auto MB/s | Forced MB/s | Auto time used for search |
| --- | ---: | ---: | ---: |
| Gaussian | 647.2 | 872.0 | 25.8 percent |
| Lognormal | 686.5 | 860.3 | 20.2 percent |
| Decimal | 604.5 | 747.3 | 19.1 percent |
| Widened f32 | 604.5 | 747.5 | 19.1 percent |
| Random walk | 572.3 | 702.6 | 18.5 percent |

Mode and Delta search uses 18 to 26 percent of Pco's default encode time on these inputs.

The forced path still trains the full bin model. This model and final emission use most of the time.

The complete prepare stage uses 58 to 71 percent of automatic encode time. Final emission uses the remaining 29 to 42 percent.

Preparation includes the selected transform, Delta application, histograms, bin optimization, weight quantization, and ANS model construction.

Compression level 4 keeps most of level 8's size benefit with much higher throughput.

| Dataset | Level 0 bytes | Level 0 MB/s | Level 4 bytes | Level 4 MB/s | Level 8 bytes | Level 8 MB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gaussian | 1,671,181 | 5,195.8 | 1,618,359 | 1,226.0 | 1,609,324 | 872.0 |
| Random walk | 2,097,165 | 3,322.7 | 1,611,991 | 956.8 | 1,551,142 | 702.6 |

Level 0 proves that Pco's mode transforms do not require a slow encoder.

The expensive part is the richer bin model. Random-walk data still needs that model or an adjacent Delta transform for good size.

## Block residual prototype

The block residual codec uses one or more references for each 1,024-value block.

The single-reference form stores the block minimum. It subtracts that minimum from every ordered integer.

FastLanes stores the low `w` residual bits for every value. A width histogram selects `w` without repeated residual scans.

Patches store sorted 16-bit positions and the remaining high bits. Scalar access uses one packed read and one binary search.

The multi-reference form also tests two and four references. It stores a one-bit or two-bit reference selector for each value.

| Dataset | Form | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | --- | ---: | ---: | ---: | ---: |
| Gaussian | One reference | 1,643,520 | 4,194.7 | 15,942.8 | 3.99 |
| Gaussian | Up to four references | 1,643,520 | 1,399.9 | 15,983.3 | 4.05 |
| Random walk | One reference | 1,663,307 | 3,752.7 | 15,582.5 | 7.38 |
| Random walk | Up to four references | 1,660,912 | 1,267.9 | 14,716.9 | 7.52 |
| Four clusters | One reference | 2,102,272 | 4,667.7 | 14,492.3 | 3.01 |
| Four clusters | Up to four references | 1,978,898 | 1,151.3 | 9,348.4 | 17.21 |

Multiple references save 0.14 percent on random-walk data and 5.9 percent on four-cluster data.

The four-cluster result remains 17.9 percent larger than ALP-RD. It remains 44.2 percent larger than RangeEntropy.

One local reference plus patches is the current block-residual winner. It is not the current winner for all float distributions.

This result narrows the role of multiple references. A prefix split or range tag still handles distinct float clusters better.

### Default candidate on random-walk floats

| Configuration | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Default without the scheme | 1,853,564 | 713.4 | 12,486.1 | 161.24 |
| Default with the scheme | 1,663,831 | 635.6 | 17,355.8 | 129.06 |
| Compact Pco | 1,551,142 | 296.2 | 3,777.2 | Not measured |

The default scheme saves 10.2 percent against ALP-RD. It remains 7.3 percent larger than Pco.

Decode throughput improves by 39.0 percent. Random access latency improves by 20.0 percent.

Compression throughput regresses by 10.9 percent on the selected column.

The direct decoder reads child buffers without copies. It also fuses the inverse ordered-float transform.

The selector rejected the scheme on Gaussian, lognormal, decimal, widened-f32, and four-cluster inputs.

The locality probe reduced compression throughput by 0.5 to 2.2 percent on those unselected inputs.

## BitSplit prototype

The `BitSplit` prototype uses one prefix dictionary per 1,024-value block. Fixed-width suffixes store the remaining bits.

The encoder sorts each block once. Adjacent XOR widths evaluate every split point without repeated value scans.

Scalar access uses two packed reads and one prefix lookup. No patch search is necessary.

| Dataset | Bytes | Encode MB/s | Decode MB/s | Scalar access ns |
| --- | ---: | ---: | ---: | ---: |
| Gaussian | 1,648,800 | 1,046.7 | 5,989.0 | 5.33 |
| Random walk | 1,664,696 | 1,057.8 | 6,260.2 | 9.85 |
| Four clusters | 1,702,464 | 900.3 | 5,898.5 | 6.19 |

The four-cluster result is 1.4 percent larger than ALP-RD. It is 14.0 percent smaller than the multiple-reference prototype.

`BitSplit` handles separated clusters better than residual references. Full decode remains much slower than the current ALP-RD path.

## FloatMult composition

Pco selects FloatMult on seven Taxi columns. The selected source columns contain 112 MB of logical values.

| Backend for FloatMult children | Aggregate bytes | Encode MB/s | Decode MB/s |
| --- | ---: | ---: | ---: |
| Normal BtrBlocks integer compression | 22,340,941 | Not isolated | 5,079.1 |
| Block residual packing | 19,555,263 | 1,826.0 | 3,628.8 |
| BitSplit | 24,105,632 | 814.7 | 2,487.2 |
| Fixed range tags | 20,860,940 | 703.5 | 3,057.3 |
| Two-level range tags | 19,052,586 | 477.3 | 2,088.3 |
| RangeEntropy | 16,730,631 | 461.9 | 1,884.1 |
| Current default on source floats | 20,104,147 | Not isolated | Not isolated |
| Pco Auto on source floats | 15,061,783 | Not isolated | Not isolated |

FloatMult does not create conventional integer distributions. Normal integer compression loses 11.1 percent against the current default.

Block residual packing saves only 2.7 percent against the current default on the Taxi source columns.

RangeEntropy saves 16.8 percent against the current default. It remains 11.1 percent larger than Pco on these columns.

The direct decode test includes child decode, FloatMult reconstruction, and output allocation.

The isolated encode test excludes FloatMult base search and the split from floats to latent integers.

California Housing gives a different result. Block residual packing uses 170,645 bytes for five FloatMult columns.

The current default uses 209,265 bytes for those source columns. Block residual packing saves 18.5 percent.

The full block residual path decodes at 2,941.6 MB/s and encodes its existing latent integers at 937.6 MB/s.

Normal BtrBlocks children use 172,046 bytes and decode at 6,991.6 MB/s on those columns.

The current `RangeEntropy` decoder is too slow for the default compressor. FloatMult therefore needs a faster entropy-like child codec.

Block residual packing is the fastest specialized encoder. Its decode path is still too slow for the default compressor.

The next FloatMult prototype will use normal integer children first. Its selector must reject Taxi-like inputs with larger encoded children.

## Why Compact still wins

Pco does not use FloatQuant on the measured paper inputs.

Pco selects FloatMult on five Housing columns and seven Taxi columns.

Pco selects adjacent Delta on three Housing columns and thirteen Air Quality columns.

These transforms explain most of the remaining Compact size gap.

FloatMult requires a new transform array. The current prototype scope includes this array.

Pco adjacent Delta also differs from current FastLanes lane-stride Delta.

The FastLanes Delta diagnostic selected one Air Quality column. It reduced aggregate size by 2.6 percent.

That diagnostic reduced compression throughput by 26 percent and decode throughput by 36 percent.

An adjacent-Delta array requires a separate design decision. This work does not add it.

## Dataset status

The completed matrix includes these paper inputs:

- Air Quality.
- California Housing.
- April 2023 high-volume for-hire Taxi.

The matrix does not include CMS Payments, r/place, or Twitter.

The RangeEntropy throughput failure appears on synthetic floats, Taxi floats, Housing floats, and Air Quality integers.

More datasets cannot satisfy the gate without a faster codec implementation.

## Default writer integration

`FloatQuantScheme` belongs to `ALL_SCHEMES`. `BtrBlocksCompressor::default()` now evaluates it for `f64` arrays.

The default file session registers `FloatQuantArray`. The August 2026 core edition permits the array.

The CUDA-compatible builder excludes FloatQuant because no CUDA decode kernel exists.

The default writer test writes, reads, and verifies an actual FloatQuant tree.

## Current prototype scope

Do not start RangeEntropy pushdowns before its full-decode cost improves substantially.

The current prototype scope includes these transforms:

- Ordered float bits.
- A high-bit and low-bit split.
- FloatMult with an arbitrary common multiplier.
- Block residual packing with optional patches.

The current scope excludes these transforms:

- Block-local adjacent Delta with fast random access.

Delta-of-delta, Delta with lookback, and convolution Delta also remain excluded.

## References

- [PCodec paper](https://arxiv.org/html/2502.06112v2)
- [PCodec repository](https://github.com/pcodec/pcodec)
