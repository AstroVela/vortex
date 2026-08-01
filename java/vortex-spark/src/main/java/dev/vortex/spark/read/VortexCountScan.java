// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexSparkSession;
import java.io.Serializable;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Metadata-only {@link Scan} answering pushed-down row-count aggregations ({@code COUNT(*)} and {@code COUNT(col)} over
 * non-nullable data columns) from Vortex file footers.
 *
 * <p>The pushdown is partial: the scan plans one input partition per file, each emitting a single row whose columns all
 * carry that file's row count, and Spark sums the per-file counts. A scan that resolves to zero files emits a single
 * zero row so that the final aggregation still sees input and returns 0 rather than null.
 */
final class VortexCountScan implements Scan, Batch {

    private final List<String> paths;
    private final Map<String, String> formatOptions;
    private final int countColumns;

    /**
     * Creates a metadata-only count scan.
     *
     * @param paths the file or directory paths of the table
     * @param formatOptions object-store properties used to open the files
     * @param countColumns how many pushed aggregate expressions the scan must emit; every emitted column carries the
     *     same per-file row count
     */
    VortexCountScan(List<String> paths, Map<String, String> formatOptions, int countColumns) {
        this.paths = List.copyOf(paths);
        this.formatOptions = Map.copyOf(formatOptions);
        this.countColumns = countColumns;
    }

    @Override
    public StructType readSchema() {
        StructField[] fields = new StructField[countColumns];
        for (int i = 0; i < countColumns; i++) {
            fields[i] = new StructField("count_" + i, DataTypes.LongType, false, Metadata.empty());
        }
        return new StructType(fields);
    }

    @Override
    public String description() {
        return String.format("VortexCountScan{paths=%s, countColumns=%d}", paths, countColumns);
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
            // Emit a single zero-count row: with a partial aggregate pushdown Spark computes the
            // final count as a sum of the scan output, and a sum over no rows would yield null.
            return new InputPartition[] {new VortexCountPartition(List.of(), formatOptions, countColumns)};
        }
        return resolved.stream()
                .map(path -> new VortexCountPartition(List.of(path), formatOptions, countColumns))
                .toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        return new VortexCountReaderFactory();
    }

    /** One file (or nothing, for the zero-file case) whose footer row count becomes a single output row. */
    record VortexCountPartition(List<String> paths, Map<String, String> formatOptions, int countColumns)
            implements InputPartition, Serializable {}

    private static final class VortexCountReaderFactory implements PartitionReaderFactory, Serializable {
        private static final long serialVersionUID = 1L;

        @Override
        public PartitionReader<InternalRow> createReader(InputPartition partition) {
            return new VortexCountPartitionReader((VortexCountPartition) partition);
        }

        @Override
        public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
            throw new UnsupportedOperationException("count scans produce rows, not columnar batches");
        }
    }
}
