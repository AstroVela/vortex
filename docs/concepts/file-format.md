# File Format

The Vortex file format is an opinionated implementation of the concepts we have seen thus far.
The writer accepts a stream of Vortex arrays, applies a layout strategy to organize them into a layout tree,
and serializes the layout and its segments into a single file.

The bulk of the file format specification describes the representation of the footer bytes such that the
layout tree can be reconstructed for scans.

See the [Vortex File Format Specification](../specification/file-format.md) for full details, or
[Reading a File](../specification/reading-a-file.md) for a step-by-step walkthrough of the read path.

## Layout Strategy

The default layout strategy for Vortex files is roughly:

* Struct Layout at the top-level to partition by columns
  * Zoned Layout to store pruning statistics for every 8k rows
      * Chunked Layout to partition the column into chunks
        * Compressor Layout to apply a compression strategy
          * Buffered Layout to localize up to 2 MB of compressed chunks per column.
              * Flat Layout to serialize each individual array chunk

This strategy optimizes for analytical query patterns: column pruning avoids reading unused columns,
zone statistics enable skipping irrelevant row ranges, and buffered chunks allow efficient I/O
with parallel decompression. The 8k row zones and buffered chunk locality balance pruning
granularity against metadata overhead.

## Compression Strategies

The Vortex file format supports two compression strategies out-of-the-box.

### BtrBlocks

BtrBlocks is a paper that describes a compression technique that uses cascading lightweight compression
techniques to achieve high compression ratios with fast decompression speeds.

The Vortex implementation is tuned to achieve a good balance of compression ratio and read performance for analytical
data. For other workloads, different compression strategies may be more appropriate.

### Compact

The Compact compression strategy is a strategy that aims to minimize the size of data on disk at the expense of
read performance. It uses more aggressive compression techniques that may require more CPU time to decompress.

The main encodings used by this strategy are ZStd for binary data and PCodec for numeric data.
