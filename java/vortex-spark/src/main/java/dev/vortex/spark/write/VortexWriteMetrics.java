// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import org.apache.spark.sql.connector.metric.CustomMetric;
import org.apache.spark.sql.connector.metric.CustomSumMetric;
import org.apache.spark.sql.connector.metric.CustomTaskMetric;

/**
 * Custom metrics reported by Vortex writes in the Spark SQL UI.
 *
 * <p>Every metric is a sum over tasks: files created, dynamic partition directories written, rows written, and Arrow
 * buffer bytes handed to the native writer (before Vortex compression).
 */
public final class VortexWriteMetrics {

    /** Number of Vortex files created by write tasks. */
    public static final String FILES_WRITTEN = "filesWritten";

    /** Number of Hive-style partition directories written to (partitioned writes only). */
    public static final String PARTITIONS_WRITTEN = "partitionsWritten";

    /** Number of rows written. */
    public static final String ROWS_WRITTEN = "rowsWritten";

    /**
     * Arrow buffer bytes flushed to the native writer, before Vortex compression. Spark collects the final task metrics
     * just before {@code DataWriter.commit()}, so rows still buffered in the last partial batch are not reflected in
     * this metric.
     */
    public static final String BYTES_BUFFERED = "bytesBuffered";

    private VortexWriteMetrics() {}

    /** The write metrics advertised through {@code Write.supportedCustomMetrics()}, as a fresh array. */
    public static CustomMetric[] supportedMetrics() {
        return new CustomMetric[] {
            new SumMetric(FILES_WRITTEN, "number of Vortex files written"),
            new SumMetric(PARTITIONS_WRITTEN, "number of partition directories written"),
            new SumMetric(ROWS_WRITTEN, "number of rows written"),
            new SumMetric(BYTES_BUFFERED, "Arrow bytes buffered for writing (before compression)")
        };
    }

    /** Creates a task-level metric value. */
    public static CustomTaskMetric taskMetric(String name, long value) {
        return new TaskMetric(name, value);
    }

    private static final class SumMetric extends CustomSumMetric {
        private final String name;
        private final String description;

        SumMetric(String name, String description) {
            this.name = name;
            this.description = description;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public String description() {
            return description;
        }
    }

    private record TaskMetric(String name, long value) implements CustomTaskMetric {}
}
