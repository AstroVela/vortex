// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.SparkPlan;
import org.apache.spark.sql.execution.datasources.v2.BatchScanExec;
import org.apache.spark.sql.execution.metric.SQLMetric;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that Vortex scans report custom task metrics (files, splits, batches, rows) through the Spark metric system.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexScanMetricsTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexScanMetricsTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    private Dataset<Row> writeAndRead(String dir, int rows, int numFiles) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            data.add(RowFactory.create(i, "name-" + i));
        }
        Path outputPath = tempDir.resolve(dir);
        spark.createDataFrame(data, SCHEMA)
                .repartition(numFiles)
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();
        return spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();
    }

    @SuppressWarnings("deprecation") // JavaConverters works in both Scala 2.12 and 2.13
    private static long metricValue(Dataset<Row> executed, String name) {
        SparkPlan plan = executed.queryExecution().executedPlan();
        List<SparkPlan> leaves = scala.collection.JavaConverters.seqAsJavaListConverter(plan.collectLeaves())
                .asJava();
        for (SparkPlan leaf : leaves) {
            if (leaf instanceof BatchScanExec batchScan) {
                SQLMetric metric = batchScan.metrics().apply(name);
                assertNotNull(metric, "missing metric " + name);
                return metric.value();
            }
        }
        throw new AssertionError("no BatchScanExec found in plan: " + plan);
    }

    @Test
    @DisplayName("Scan tasks report files, splits, batches, and row counts")
    public void testScanMetrics() {
        Dataset<Row> readDf = writeAndRead("metrics_basic", 60, 3);

        Dataset<Row> selected = readDf.select("id");
        assertEquals(60, selected.collectAsList().size());

        assertEquals(3L, metricValue(selected, "filesRead"));
        assertEquals(60L, metricValue(selected, "rowsRead"));
        assertTrue(metricValue(selected, "splitsProcessed") >= 3L, "each file produces at least one split");
        assertTrue(metricValue(selected, "batchesRead") >= 3L, "each file produces at least one batch");
    }

    @Test
    @DisplayName("Row metrics count post-filter rows when a predicate is pushed")
    public void testScanMetricsWithPushedFilter() {
        Dataset<Row> readDf = writeAndRead("metrics_filtered", 100, 2);

        Dataset<Row> filtered = readDf.filter(readDf.col("id").lt(10));
        assertEquals(10, filtered.collectAsList().size());

        assertEquals(2L, metricValue(filtered, "filesRead"));
        assertEquals(10L, metricValue(filtered, "rowsRead"), "rowsRead counts rows surviving the pushed filter");
    }
}
