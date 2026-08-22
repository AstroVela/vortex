# Delta / delta-of-delta encoding analysis

A reproducible study of two questions:

1. Which cheap, sampleable statistics predict that **Delta** encoding will pay off?
2. Is **delta-of-delta** ever worth encoding in Vortex?

Everything here runs on public real-world data. No synthetic arrays are used, because the whole
question is which *real* shapes carry delta structure.

## Running it

```bash
scripts/delta-analysis/fetch.sh                       # ~1.5 GB of public parquet/CSV
cd scripts/delta-analysis
uv run --with pyarrow --with pandas --with numpy python study.py    # writes results.csv
uv run --with pandas --with numpy python analyze.py                 # prints the tables below
```

`study.py` models the compressor; the end-to-end numbers in the last section come from running the
real `BtrBlocksCompressor` over the same corpus.

## Corpus

| dataset | rows used | what it is | integer columns |
| --- | --- | --- | --- |
| `hits` | 1 M | ClickBench web analytics, sorted by counter/user | 77 |
| `airquality` | 824 K | air-sensor telemetry, Kafka offsets, ns timestamps | 16 |
| `taxi` | 3.3 M | NYC yellow taxi trips, near-sorted timestamps | 8 |
| `rplace` | 2 M | r/place canvas events, sorted ns timestamps | 6 |
| `btc_1s` / `btc_1m` | 2.7 M / 45 K | exchange klines: **exactly periodic** timestamps, fixed-point prices | 11 each |
| `btc_trades` | 2 M | raw trades: sequential ids, irregular ms timestamps | 5 |
| `power` | 2.1 M | UCI household power, one-minute meter readings | 8 |

That is 142 integer columns, analysed in 64 Ki-value blocks: 1966 (column, block) units.

## What the compressor actually does

Three facts pinned down from the code, which the cost model relies on:

* **FastLanes Delta is lag-1 in the original order.** The transpose permutes a 1024-element chunk
  and the delta kernel walks each lane in `FL_ORDER`; composing the two mappings gives, for every
  residual, a predecessor exactly one index earlier. Only the `1024 / T` lane heads per chunk
  become bases. (`transpose.rs` + `macros.rs::iterate!` in `fastlanes 0.6.1`; reproduced in
  `study.py`.) The earlier estimate in `DeltaScheme` assumed a lane-stride difference instead,
  which is why it measured residuals by encoding the whole array.
* **Bases cost exactly one bit per value**: `1024 / T` bases of `T` bits per 1024 values, for every
  width `T`.
* **The layer below Delta is FoR or ZigZag, then BitPacking**, and BitPacking picks its width by
  minimising `packed_bits + exceptions × (byte_width + 4) × 8`. FoR and ZigZag fail on opposite
  inputs: one extreme outlier ruins FoR (it shifts every value up by `|min|`), while ZigZag leaves
  the outlier as a cheap exception. The model takes the better of the two, which is what the
  cascade does.

## Finding 1: delta structure is local, so sampling is nearly free and nearly exact

A residual only exists between neighbouring values, so the sample must be **contiguous runs**;
index-wise random sampling measures nothing. With 16 runs of 64 values (1024 values, the same
shape as the compressor's own sampler), the predicted BitPacking width for the residuals matches
the width computed over the full 64 Ki block:

| sample | within 1 bit of truth | mean error |
| --- | --- | --- |
| 16 × 64 (1 Ki) | 93.0 % | +0.21 bits |
| 64 × 64 (4 Ki) | 94.0 % | +0.27 bits |
| 16 × 256 (4 Ki) | 94.2 % | +0.23 bits |
| 8 × 1024 (8 Ki) | 94.2 % | +0.25 bits |

Quadrupling the sample buys 1 percentage point. This is the asymmetry worth remembering: the
residual width is a *local* property that a sample nails, whereas the *value range* that FoR needs
is a global property that a sample under-estimates (mean −0.28 bits at 1 Ki).

## Finding 2: the stats that predict it

All of these accumulate in one pass over the sampled runs, in the same style as `IntegerStats`:

| statistic | cost | what it decides |
| --- | --- | --- |
| bit-width histogram of the residuals, as the cascade sees them (ZigZag for signed, raw for unsigned) | 65 counters | the packed width, exceptions included |
| `min` / `max` of the residuals | 2 registers | the FoR width, i.e. residuals in a narrow band away from zero |
| zero count | 1 counter | run-heavy data, where RunEnd/Dict usually win |
| decreasing count | 1 counter | unsigned wrap-around, where Delta cannot help at all |
| span == 0 | derived | arithmetic sequence, where `SequenceScheme` wins |

Decision quality, choosing Delta when the sampled estimate beats the sampled alternative by a
factor of `t` (ground truth = full-block cost):

| `t` | picks delta | precision | recall | mean regret |
| --- | --- | --- | --- | --- |
| 1.00 | 40.3 % | 0.977 | 0.917 | 0.021 B/value |
| 1.05 | 37.8 % | 0.987 | 0.870 | 0.024 B/value |
| 1.25 | 27.4 % | 0.996 | 0.635 | 0.059 B/value |
| 1.50 | 21.5 % | 1.000 | 0.501 | 0.110 B/value |

Cheaper rules were tried and are worse: widths alone without the exception model (precision 0.93),
"more than half the residuals are zero" (0.43), "the sample is sorted" (0.34). The histogram is
worth its 65 counters.

## Finding 3: delta-of-delta never pays — not once in 1966 real blocks

| | |
| --- | --- |
| blocks where delta-of-delta beats delta | **0 / 1966** |
| best width the second delta layer ever saved | **0 bits** |
| median cost of the second layer | +0.20 bytes/value |

The reason is structural, and it is worth stating because it is not what the Gorilla-style
literature suggests:

* Gorilla stores deltas with a variable-length code and no framing layer, so removing the constant
  part of the rate is the *only* way to shrink them. In Vortex the layer under Delta is already
  FoR (or ZigZag) + BitPacking, and **FoR subtracts the mean rate for free**. Constant-rate
  timestamps do not need a second delta layer; they come out at width 0 (`btc_1s.open_time`:
  3.25 B/value → 0.125 B/value, a 26× win from one Delta layer, and `SequenceScheme` catches the
  perfectly regular case even more cheaply).
* What remains after the first layer is jitter. Differencing jitter *doubles* its span, costing
  about a bit, and the second layer of bases costs another bit per value. So delta-of-delta needs
  the delta *rate to drift* by more than the local jitter across a whole block to break even — and
  no real column in this corpus does that. Measured over the corpus, the second layer is one bit
  *wider* than the first in 554 blocks, exactly as wide in 1254, and never narrower.

The conclusion drawn in the code: expose delta-of-delta as a *statistic*
(`DeltaStats::delta_of_delta_bits_per_value`) so the claim stays falsifiable on new corpora, but do
not add a scheme that would never fire. `delta_of_delta_is_never_narrower_than_delta` in
`vortex-btrblocks/src/schemes/integer/delta_stats.rs` pins it as a test.

## Finding 4: what this is worth end-to-end

Every integer column of the corpus compressed with the real `BtrBlocksCompressor`, comparing the
compressor with Delta disabled, with the previous full-encode estimate, and with the sampled-stats
estimate (`nbytes` of the compressed tree, 8 blocks per dataset):

| dataset | no Delta | previous estimate | sampled stats | vs. no Delta | vs. previous |
| --- | --- | --- | --- | --- | --- |
| airquality | 9.80 MB | 5.73 MB | 5.55 MB | −43.3 % | −3.1 % |
| taxi | 6.32 MB | 6.17 MB | 3.94 MB | −37.6 % | −36.1 % |
| rplace | 4.25 MB | 4.25 MB | 3.19 MB | −25.0 % | −25.0 % |
| btc_1m | 1.93 MB | 1.93 MB | 1.58 MB | −18.2 % | −18.2 % |
| btc_trades | 6.40 MB | 5.63 MB | 5.63 MB | −12.0 % | 0.0 % |
| hits | 20.54 MB | 20.46 MB | 18.87 MB | −8.2 % | −7.8 % |
| btc_1s | 16.56 MB | 16.34 MB | 15.93 MB | −3.8 % | −2.5 % |
| power | 3.35 MB | 3.35 MB | 3.34 MB | −0.2 % | −0.1 % |
| **total** | **69.15 MB** | **63.87 MB** | **58.04 MB** | **−16.1 %** | **−9.1 %** |

Delta changes the output on 366 of 1059 blocks: smaller on 355, larger on 11. Compression
throughput is unchanged (the estimate got cheaper, and Delta gets selected more often, which
roughly cancel), while the estimate is now O(sample) rather than O(array).

The two golden-corpus snapshots that moved are both wins for the same reason — monotone offsets,
the shape the previous estimate was blindest to:

* `string_fsst_structured`: 151 382 → 121 774 bytes (FSST `codes_offsets` 36 992 → 8 732).
* `list_of_int_runs`: 11 146 → 6 026 bytes (list offsets 7 178 → 2 058).

## Known limitations

* **Run-heavy columns can still be over-selected.** 11 blocks regress, all in columns dominated by
  repeats (`power.timestamp`, `power.Sub_metering_3`), where RunEnd-shaped encodings beat Delta by
  a little and the residual estimate is slightly optimistic. Total cost 26 KB against 5.8 MB
  gained. The zero-count statistic is a candidate guard, but on its own it is a poor predictor
  (precision 0.43) — `rplace.red` has 98.5 % zero residuals and Delta wins there by 1.9×.
* **Unsigned columns cannot use ZigZag.** Vortex subtracts in the unsigned domain, so a decrease
  wraps to a near-maximal value that only BitPacking exceptions can absorb. A signed column with
  the same shape keeps narrow residuals. Reinterpreting unsigned residuals as signed before the
  cascade would recover this; it is not done today, and the stats report the width Vortex actually
  achieves rather than the one it could.
* `study.py` compares Delta against FoR/ZigZag + BitPacking only. It does not model Dict, RunEnd or
  Sparse, so its "delta wins" rate overstates what the full cascade picks; the end-to-end table is
  the one to trust for selection rates.
* Nulls are treated the way Delta treats them (fill-forward), and each dataset contributes a single
  partition.
