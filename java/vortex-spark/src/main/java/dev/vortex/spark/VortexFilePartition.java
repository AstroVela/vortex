// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.HasPartitionKey;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.types.StructType;

/**
 * An {@link InputPartition} describing a group of Vortex files that a single reader should handle together.
 *
 * <p>Each executor opens a single Vortex {@code Session}, {@code DataSource} and {@code Scan} over the partition's
 * {@link #paths()} and consumes every Vortex partition produced by that scan before moving on to the next Spark
 * {@code InputPartition}.
 *
 * <p>The requested output schema is carried as a {@link StructType} rather than a list of {@code Column} objects:
 * {@code StructType} is the stable serialization surface in Spark and survives shipping to executors reliably.
 *
 * @param paths the Vortex file paths (or globs) belonging to this input partition
 * @param readSchema the requested output schema (data columns + partition columns)
 * @param formatOptions object-store properties used to open the files
 * @param partitionValues Hive-style partition column values shared by all {@link #paths()}
 * @param partitionKey the typed partition values in table-partitioning order, or {@code null} when the scan does not
 *     report a key-grouped partitioning
 */
public record VortexFilePartition(
        List<String> paths,
        StructType readSchema,
        Map<String, String> formatOptions,
        Map<String, String> partitionValues,
        InternalRow partitionKey)
        implements InputPartition, HasPartitionKey, Serializable {

    /** Creates a partition without a key-grouped partitioning key. */
    public VortexFilePartition(
            List<String> paths,
            StructType readSchema,
            Map<String, String> formatOptions,
            Map<String, String> partitionValues) {
        this(paths, readSchema, formatOptions, partitionValues, null);
    }

    @Override
    public InternalRow partitionKey() {
        return partitionKey;
    }
}
