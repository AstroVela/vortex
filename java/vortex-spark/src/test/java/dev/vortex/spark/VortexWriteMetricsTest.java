// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.QueryExecution;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.metric.SQLMetric;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.QueryExecutionListener;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that Vortex writes report custom task metrics (files, partitions, rows, buffered bytes) through the Spark
 * metric system.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexWriteMetricsTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("region", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexWriteMetricsTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    private Dataset<Row> testData(int rows) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            data.add(RowFactory.create(i, "region-" + (i % 2)));
        }
        return spark.createDataFrame(data, SCHEMA);
    }

    /** Runs a write while capturing the executed plan of the write query through a QueryExecutionListener. */
    private SparkPlan captureWritePlan(Runnable write) throws Exception {
        AtomicReference<SparkPlan> captured = new AtomicReference<>();
        QueryExecutionListener listener = new QueryExecutionListener() {
            @Override
            public void onSuccess(String funcName, QueryExecution qe, long durationNs) {
                SparkPlan plan = qe.executedPlan();
                if (findMetric(plan, "rowsWritten") != null) {
                    captured.set(plan);
                }
            }

            @Override
            public void onFailure(String funcName, QueryExecution qe, Exception exception) {}
        };
        spark.listenerManager().register(listener);
        try {
            write.run();
            // The listener bus delivers events asynchronously; poll briefly.
            for (int i = 0; i < 100 && captured.get() == null; i++) {
                Thread.sleep(100);
            }
        } finally {
            spark.listenerManager().unregister(listener);
        }
        SparkPlan plan = captured.get();
        assertTrue(plan != null, "did not observe a write plan reporting Vortex write metrics");
        return plan;
    }

    /** Walks the plan tree looking for a node exposing the named SQL metric. */
    @SuppressWarnings("deprecation") // JavaConverters works in both Scala 2.12 and 2.13
    private static SQLMetric findMetric(SparkPlan plan, String name) {
        scala.Option<SQLMetric> metric = plan.metrics().get(name);
        if (metric.isDefined()) {
            return metric.get();
        }
        List<SparkPlan> children = scala.collection.JavaConverters.seqAsJavaListConverter(plan.children())
                .asJava();
        for (SparkPlan child : children) {
            SQLMetric found = findMetric(child, name);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    @Test
    @DisplayName("Unpartitioned writes report file, row, and byte metrics")
    public void testWriteMetrics() throws Exception {
        Path outputPath = tempDir.resolve("write_metrics");
        SparkPlan plan = captureWritePlan(() -> testData(100)
                .repartition(2)
                .write()
                .format("vortex")
                // Metrics are pulled before commit(), which flushes the last partial batch; a small
                // batch size makes the byte metric observable for this small dataset.
                .option("vortex.write.batch.size", "16")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save());

        assertEquals(100L, findMetric(plan, "rowsWritten").value());
        assertEquals(2L, findMetric(plan, "filesWritten").value());
        assertTrue(findMetric(plan, "bytesBuffered").value() > 0L, "expected non-zero buffered bytes");
    }

    @Test
    @DisplayName("Partitioned writes additionally report partition directory counts")
    public void testPartitionedWriteMetrics() throws Exception {
        Path outputPath = tempDir.resolve("write_metrics_partitioned");
        SparkPlan plan = captureWritePlan(() -> testData(20)
                .repartition(1)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save());

        assertEquals(20L, findMetric(plan, "rowsWritten").value());
        assertEquals(2L, findMetric(plan, "partitionsWritten").value());
        assertEquals(2L, findMetric(plan, "filesWritten").value());
    }
}
