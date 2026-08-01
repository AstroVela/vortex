// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that row-count aggregations push down into the Vortex datasource and are answered from file footers.
 *
 * <p>Pushed-down counts plan a {@code VortexCountScan}; queries that cannot be answered from metadata (filters,
 * grouping, nullable columns, distinct) must fall back to a regular scan and still return correct results.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexCountPushdownTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexCountPushdownTest")
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

    private Dataset<Row> writeAndRead(String dir, int rows, int numFiles) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            data.add(RowFactory.create(i, i % 10 == 0 ? null : "name-" + i));
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

    @Test
    @DisplayName("count() over a multi-file dataset is answered from footers via VortexCountScan")
    public void testCountStarPushdown() {
        Dataset<Row> readDf = writeAndRead("count_multi", 100, 4);

        Dataset<Row> counted = readDf.agg(functions.count(functions.lit(1)).as("cnt"));
        String plan = counted.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("VortexCountScan"), "Expected a metadata-only count scan in the plan: " + plan);
        assertEquals(100L, counted.collectAsList().get(0).getLong(0));

        assertEquals(100L, readDf.count());
    }

    @Test
    @DisplayName("COUNT(col) on a non-nullable column pushes down; on a nullable column it falls back")
    public void testCountColumn() {
        Dataset<Row> readDf = writeAndRead("count_columns", 50, 2);

        Dataset<Row> countedId = readDf.agg(functions.count(readDf.col("id")).as("cnt"));
        assertTrue(
                countedId.queryExecution().executedPlan().toString().contains("VortexCountScan"),
                "COUNT of a non-nullable column should be answered from metadata");
        assertEquals(50L, countedId.collectAsList().get(0).getLong(0));

        Dataset<Row> countedName =
                readDf.agg(functions.count(readDf.col("name")).as("cnt"));
        assertFalse(
                countedName.queryExecution().executedPlan().toString().contains("VortexCountScan"),
                "COUNT of a nullable column must not be answered from metadata");
        assertEquals(45L, countedName.collectAsList().get(0).getLong(0));
    }

    @Test
    @DisplayName("Several counts in one aggregation all push down together")
    public void testMultipleCounts() {
        Dataset<Row> readDf = writeAndRead("count_several", 30, 3);

        Dataset<Row> counted = readDf.agg(
                functions.count(functions.lit(1)).as("all_rows"),
                functions.count(readDf.col("id")).as("ids"));
        assertTrue(counted.queryExecution().executedPlan().toString().contains("VortexCountScan"));
        Row row = counted.collectAsList().get(0);
        assertEquals(30L, row.getLong(0));
        assertEquals(30L, row.getLong(1));
    }

    @Test
    @DisplayName("count() with a filter falls back to a regular scan and stays correct")
    public void testCountWithFilterFallsBack() {
        Dataset<Row> readDf = writeAndRead("count_filter", 40, 2);

        Dataset<Row> counted = readDf.filter(readDf.col("id").lt(10)).agg(functions.count(functions.lit(1)));
        assertFalse(
                counted.queryExecution().executedPlan().toString().contains("VortexCountScan"),
                "a filtered count cannot be answered from footer metadata");
        assertEquals(10L, counted.collectAsList().get(0).getLong(0));
    }

    @Test
    @DisplayName("Grouped counts fall back to a regular scan and stay correct")
    public void testGroupedCountFallsBack() {
        Dataset<Row> readDf = writeAndRead("count_grouped", 20, 2);

        Dataset<Row> grouped =
                readDf.groupBy(functions.col("id").mod(2).as("parity")).count().orderBy("parity");
        assertFalse(grouped.queryExecution().executedPlan().toString().contains("VortexCountScan"));
        List<Row> rows = grouped.collectAsList();
        assertEquals(2, rows.size());
        assertEquals(10L, rows.get(0).getLong(1));
        assertEquals(10L, rows.get(1).getLong(1));
    }

    @Test
    @DisplayName("count() over a directory with no Vortex files returns 0, not null")
    public void testCountEmptyDirectory() throws IOException {
        Path emptyDir = tempDir.resolve("count_empty");
        Files.createDirectories(emptyDir);

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .schema(SCHEMA)
                .option("path", emptyDir.toUri().toString())
                .load();

        assertEquals(0L, readDf.count());
    }
}
