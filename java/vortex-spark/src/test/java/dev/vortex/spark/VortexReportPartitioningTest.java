// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that Vortex scans report a key-grouped output partitioning over Hive-style partition columns, enabling
 * storage-partitioned execution: joins and aggregations keyed on the partition columns run without a shuffle when
 * {@code spark.sql.sources.v2.bucketing.enabled} is set.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexReportPartitioningTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("region", DataTypes.StringType, true),
            DataTypes.createStructField("amount", DataTypes.LongType, false)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexReportPartitioningTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.sources.v2.bucketing.enabled", "true")
                .config("spark.sql.autoBroadcastJoinThreshold", "-1")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    private Dataset<Row> writePartitionedAndRead(String dir, int rows) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            data.add(RowFactory.create(i, "region-" + (i % 4), i * 10L));
        }
        Path outputPath = tempDir.resolve(dir);
        spark.createDataFrame(data, SCHEMA)
                .repartition(1)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();
        return spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();
    }

    @Test
    @DisplayName("A join on the partition column runs without a shuffle (storage-partitioned join)")
    public void testStoragePartitionedJoin() {
        Dataset<Row> left = writePartitionedAndRead("spj_left", 40);
        Dataset<Row> right = writePartitionedAndRead("spj_right", 20);

        Dataset<Row> joined = left.join(right, "region");
        String plan = joined.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("Exchange"), "expected a shuffle-free storage-partitioned join: " + plan);

        // Each of the 4 regions has 10 rows on the left and 5 on the right.
        assertEquals(4 * 10 * 5, joined.count());
    }

    @Test
    @DisplayName("An aggregation keyed on the partition column runs without a shuffle")
    public void testAggregationWithoutShuffle() {
        Dataset<Row> readDf = writePartitionedAndRead("kgp_agg", 40);

        Dataset<Row> aggregated = readDf.groupBy("region").count().orderBy("region");
        String plan = aggregated.queryExecution().executedPlan().toString();
        // The orderBy needs a range exchange; the aggregation itself must not.
        assertFalse(
                plan.contains("Exchange hashpartitioning"),
                "the aggregation must reuse the reported key-grouped partitioning: " + plan);

        List<Row> rows = aggregated.collectAsList();
        assertEquals(4, rows.size());
        for (Row row : rows) {
            assertEquals(10L, row.getLong(1));
        }
    }

    @Test
    @DisplayName("Scans that prune away the partition column fall back to unknown partitioning")
    public void testPrunedPartitionColumnFallsBack() {
        Dataset<Row> readDf = writePartitionedAndRead("kgp_pruned", 40);

        // The scan output does not contain "region", so the scan cannot report a key-grouped
        // partitioning; the aggregation shuffles as usual and results stay correct.
        Dataset<Row> aggregated = readDf.groupBy("id").count();
        assertEquals(40, aggregated.count());
        assertTrue(
                aggregated.queryExecution().executedPlan().toString().contains("Exchange"),
                "grouping on a non-partition column requires a shuffle");
    }
}
