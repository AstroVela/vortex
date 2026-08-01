// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.util.OptionalLong;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.connector.read.PartitionReader;

/**
 * Emits a single row carrying the row count of one Vortex file, for pushed-down count aggregations.
 *
 * <p>The count is taken from the file footer when it is exact; otherwise the reader falls back to iterating the scan's
 * native partitions, preferring their recorded row counts and only decoding batches as a last resort.
 */
final class VortexCountPartitionReader implements PartitionReader<InternalRow> {

    private final VortexCountScan.VortexCountPartition partition;
    private boolean emitted;

    VortexCountPartitionReader(VortexCountScan.VortexCountPartition partition) {
        this.partition = partition;
    }

    @Override
    public boolean next() {
        if (emitted) {
            return false;
        }
        emitted = true;
        return true;
    }

    @Override
    public InternalRow get() {
        long count = countRows();
        Object[] values = new Object[partition.countColumns()];
        for (int i = 0; i < values.length; i++) {
            values[i] = count;
        }
        return new GenericInternalRow(values);
    }

    @Override
    public void close() {}

    private long countRows() {
        if (partition.paths().isEmpty()) {
            return 0L;
        }
        Session session = VortexSparkSession.get(partition.formatOptions());
        DataSource dataSource = DataSource.open(session, partition.paths(), partition.formatOptions());
        if (dataSource.rowCount() instanceof DataSource.RowCount.Exact exact) {
            return exact.value();
        }
        return countByScanning(dataSource);
    }

    private long countByScanning(DataSource dataSource) {
        long total = 0;
        Scan scan = dataSource.scan(ScanOptions.of());
        while (scan.hasNext()) {
            Partition nativePartition = scan.next();
            OptionalLong partitionRows = nativePartition.rowCount();
            if (partitionRows.isPresent()) {
                total += partitionRows.getAsLong();
                continue;
            }
            try (ArrowReader reader = nativePartition.scanArrow(ArrowAllocation.rootAllocator())) {
                while (reader.loadNextBatch()) {
                    total += reader.getVectorSchemaRoot().getRowCount();
                }
            } catch (IOException e) {
                throw new RuntimeException("failed to count rows in " + partition.paths(), e);
            }
        }
        return total;
    }
}
