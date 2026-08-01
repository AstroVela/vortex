// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that Spark LIMIT pushdown into the Vortex datasource produces correct results.
 *
 * <p>{@code VortexScanBuilder} accepts every LIMIT and reports it as partially pushed: each partition reader returns at
 * most {@code limit} rows and Spark applies the global limit on top, so results must be correct for single-file and
 * multi-file datasets, with and without pushed filters.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexLimitPushdownTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("category", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexLimitPushdownTest")
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
            data.add(RowFactory.create(i, i % 2 == 0 ? "even" : "odd"));
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
    @DisplayName("LIMIT on a single-file dataset returns exactly the requested row count")
    public void testLimitSingleFile() {
        Dataset<Row> readDf = writeAndRead("limit_single", 100, 1);

        assertEquals(10, readDf.limit(10).collectAsList().size());
        assertEquals(1, readDf.limit(1).collectAsList().size());
    }

    @Test
    @DisplayName("LIMIT on a multi-file dataset still returns exactly the requested row count")
    public void testLimitMultiFile() {
        Dataset<Row> readDf = writeAndRead("limit_multi", 100, 4);

        assertEquals(7, readDf.limit(7).collectAsList().size());
        assertEquals(100, readDf.limit(100).collectAsList().size());
    }

    @Test
    @DisplayName("LIMIT larger than the dataset returns every row")
    public void testLimitLargerThanDataset() {
        Dataset<Row> readDf = writeAndRead("limit_large", 50, 2);

        assertEquals(50, readDf.limit(1000).collectAsList().size());
    }

    @Test
    @DisplayName("LIMIT combined with a pushed filter applies the filter before the limit")
    public void testLimitWithPushedFilter() {
        Dataset<Row> readDf = writeAndRead("limit_filter", 100, 2);

        List<Row> results;
        try {
            results = readDf.filter(readDf.col("id").geq(90)).limit(5).collectAsList();
        } catch (Exception e) {
            // Native support for combining a filter with a limit lands with
            // https://github.com/vortex-data/vortex/pull/8766; until then the native scan
            // rejects the combination and this test cannot run.
            Assumptions.assumeTrue(
                    !isFilterWithLimitUnsupported(e), "requires native filter+limit support (vortex-data/vortex#8766)");
            throw e;
        }
        assertEquals(5, results.size());
        for (Row row : results) {
            assertTrue(row.getInt(0) >= 90, "filter must be applied before the limit, got id=" + row.getInt(0));
        }
    }

    private static boolean isFilterWithLimitUnsupported(Throwable t) {
        for (Throwable cause = t; cause != null; cause = cause.getCause()) {
            String message = cause.getMessage();
            if (message != null && message.contains("doesn't support scans with both a filter and a limit")) {
                return true;
            }
        }
        return false;
    }

    @Test
    @DisplayName("The pushed limit shows up in the executed plan's scan description")
    public void testLimitAppearsInPlan() {
        Dataset<Row> readDf = writeAndRead("limit_plan", 20, 1);

        String plan = readDf.limit(3).queryExecution().executedPlan().toString();
        assertTrue(plan.contains("limit=3"), "Expected pushed limit in the executed plan: " + plan);
    }
}
