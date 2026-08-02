// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexSparkSession;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * {@link Scan} answering pushed-down aggregations ({@code MIN}/{@code MAX}/{@code SUM}/{@code COUNT}/{@code COUNT(*)})
 * with Vortex's native streaming accumulators instead of decoding rows into Spark.
 *
 * <p>The pushdown is partial: the scan plans one input partition per file, each emitting a single row of per-file
 * aggregate values, and Spark re-aggregates the partials ({@code MIN}/{@code MAX}/{@code SUM} of the emitted column;
 * counts are summed). Pushed predicates are applied natively before aggregation. A scan that resolves to zero files
 * emits a single identity row (zero counts, null min/max/sum) so the final aggregation still sees input.
 */
final class VortexAggregateScan implements Scan, Batch {

    private final List<String> paths;
    private final Map<String, String> formatOptions;
    private final List<PushedAggregate> aggregates;
    private final Predicate[] pushedPredicates;

    /**
     * Creates a native aggregate scan.
     *
     * @param paths the file or directory paths of the table
     * @param formatOptions object-store properties used to open the files
     * @param aggregates the accepted aggregates, in Spark's requested order
     * @param pushedPredicates predicates applied natively before aggregation; may be empty
     */
    VortexAggregateScan(
            List<String> paths,
            Map<String, String> formatOptions,
            List<PushedAggregate> aggregates,
            Predicate[] pushedPredicates) {
        this.paths = List.copyOf(paths);
        this.formatOptions = Map.copyOf(formatOptions);
        this.aggregates = List.copyOf(aggregates);
        this.pushedPredicates = pushedPredicates == null ? new Predicate[0] : pushedPredicates.clone();
    }

    @Override
    public StructType readSchema() {
        StructField[] fields = new StructField[aggregates.size()];
        for (int i = 0; i < aggregates.size(); i++) {
            PushedAggregate aggregate = aggregates.get(i);
            fields[i] = new StructField("agg_" + i, aggregate.resultType(), aggregate.nullable(), Metadata.empty());
        }
        return new StructType(fields);
    }

    @Override
    public String description() {
        return String.format(
                "VortexAggregateScan{paths=%s, aggregates=%s, pushedPredicates=%s}",
                paths, aggregates, Arrays.toString(pushedPredicates));
    }

    @Override
    public Batch toBatch() {
        return this;
    }

    @Override
    public InputPartition[] planInputPartitions() {
        List<String> resolved =
                VortexBatchExec.resolveVortexPaths(VortexSparkSession.get(formatOptions), paths, formatOptions);
        if (resolved.isEmpty()) {
            // Emit a single identity row: zero counts and null min/max/sum, matching an
            // aggregation over an empty table once Spark folds the partial values.
            return new InputPartition[] {new VortexAggregatePartition(List.of(), formatOptions)};
        }
        return resolved.stream()
                .map(path -> new VortexAggregatePartition(List.of(path), formatOptions))
                .toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        return new VortexAggregateReaderFactory(aggregates, pushedPredicates);
    }

    /** One file (or nothing, for the zero-file case) whose aggregate values become a single output row. */
    record VortexAggregatePartition(List<String> paths, Map<String, String> formatOptions)
            implements InputPartition, Serializable {}

    private static final class VortexAggregateReaderFactory implements PartitionReaderFactory, Serializable {
        private static final long serialVersionUID = 1L;

        private final List<PushedAggregate> aggregates;
        private final Predicate[] pushedPredicates;

        VortexAggregateReaderFactory(List<PushedAggregate> aggregates, Predicate[] pushedPredicates) {
            this.aggregates = aggregates;
            this.pushedPredicates = pushedPredicates;
        }

        @Override
        public PartitionReader<InternalRow> createReader(InputPartition partition) {
            return new VortexAggregatePartitionReader(
                    (VortexAggregatePartition) partition, aggregates, pushedPredicates);
        }

        @Override
        public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
            throw new UnsupportedOperationException("aggregate scans produce rows, not columnar batches");
        }
    }
}
