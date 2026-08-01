// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.connector.metric.CustomMetric;
import org.apache.spark.sql.connector.metric.CustomSumMetric;
import org.apache.spark.sql.connector.metric.CustomTaskMetric;

/**
 * Custom metrics reported by Vortex scans in the Spark SQL UI.
 *
 * <p>Every metric is a sum over tasks: files opened, native Vortex splits processed, Arrow record batches decoded, and
 * rows produced (after any pushed filter).
 */
public final class VortexScanMetrics {

    /** Number of Vortex files opened by scan tasks. */
    public static final String FILES_READ = "filesRead";

    /** Number of native Vortex splits (scan partitions) processed. */
    public static final String SPLITS_PROCESSED = "splitsProcessed";

    /** Number of Arrow record batches decoded. */
    public static final String BATCHES_READ = "batchesRead";

    /** Number of rows produced by the scan, after any pushed filter. */
    public static final String ROWS_READ = "rowsRead";

    private VortexScanMetrics() {}

    /** The scan metrics advertised through {@code Scan.supportedCustomMetrics()}, as a fresh array. */
    public static CustomMetric[] supportedMetrics() {
        return new CustomMetric[] {
            new SumMetric(FILES_READ, "number of Vortex files read"),
            new SumMetric(SPLITS_PROCESSED, "number of Vortex splits processed"),
            new SumMetric(BATCHES_READ, "number of Arrow batches read"),
            new SumMetric(ROWS_READ, "number of rows read")
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
