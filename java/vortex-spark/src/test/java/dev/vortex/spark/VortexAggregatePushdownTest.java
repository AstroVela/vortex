// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.math.BigDecimal;
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
 * Tests that MIN/MAX/SUM/COUNT aggregations push down into the Vortex datasource and are evaluated by Vortex's native
 * accumulators.
 *
 * <p>Pushed-down aggregations plan a {@code VortexAggregateScan}; cases with semantics Vortex cannot match (float
 * min/max NaN ordering, long-sum overflow, distinct, grouping) must fall back to a regular scan and still return
 * correct results.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexAggregatePushdownTest {

    private static final int ROWS = 100;
    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true),
            DataTypes.createStructField("qty", DataTypes.IntegerType, true),
            DataTypes.createStructField("price", DataTypes.createDecimalType(10, 2), true),
            DataTypes.createStructField("ratio", DataTypes.DoubleType, true),
            DataTypes.createStructField("big", DataTypes.LongType, false)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexAggregatePushdownTest")
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

    /**
     * Rows {@code i} in {@code [0, 100)}: {@code id=i}; {@code name} is null every 10th row, otherwise
     * {@code name-%02d}; {@code qty} is null every 5th row, otherwise {@code i}; {@code price=i.25};
     * {@code ratio=i*0.5}; {@code big=i*1000}.
     */
    private Dataset<Row> writeAndRead(String dir, int numFiles) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < ROWS; i++) {
            data.add(RowFactory.create(
                    i,
                    i % 10 == 0 ? null : String.format("name-%02d", i),
                    i % 5 == 0 ? null : i,
                    new BigDecimal(i + ".25"),
                    i * 0.5,
                    i * 1000L));
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

    private static void assertNativeAggregate(Dataset<Row> df) {
        String plan = df.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("VortexAggregateScan"), "Expected a native aggregate scan in the plan: " + plan);
    }

    private static void assertNoNativeAggregate(Dataset<Row> df) {
        String plan = df.queryExecution().executedPlan().toString();
        assertFalse(plan.contains("VortexAggregateScan"), "Expected fallback to a regular scan in the plan: " + plan);
    }

    @Test
    @DisplayName("MIN/MAX/SUM/COUNT over a multi-file dataset push down together")
    public void testMinMaxSumCountPushdown() {
        Dataset<Row> readDf = writeAndRead("agg_basic", 4);

        Dataset<Row> agg = readDf.agg(
                functions.min("id"), functions.max("id"), functions.sum("id"), functions.count(readDf.col("qty")));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals(0, row.getInt(0));
        assertEquals(99, row.getInt(1));
        assertEquals(4950L, row.getLong(2));
        assertEquals(80L, row.getLong(3));
    }

    @Test
    @DisplayName("Aggregates with a pushed filter are evaluated natively on the filtered rows")
    public void testAggregateWithFilter() {
        Dataset<Row> readDf = writeAndRead("agg_filter", 3);

        Dataset<Row> agg = readDf.filter(readDf.col("id").lt(50))
                .agg(functions.sum("id"), functions.count(functions.lit(1)), functions.min("qty"));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals(1225L, row.getLong(0));
        assertEquals(50L, row.getLong(1));
        assertEquals(1, row.getInt(2));
    }

    @Test
    @DisplayName("MIN/MAX over strings and COUNT of a nullable column push down")
    public void testStringMinMaxAndNullableCount() {
        Dataset<Row> readDf = writeAndRead("agg_string", 2);

        Dataset<Row> agg =
                readDf.agg(functions.min("name"), functions.max("name"), functions.count(readDf.col("name")));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals("name-01", row.getString(0));
        assertEquals("name-99", row.getString(1));
        assertEquals(90L, row.getLong(2));
    }

    @Test
    @DisplayName("SUM over decimal and double columns pushes down and keeps exact results")
    public void testSumDecimalAndDouble() {
        Dataset<Row> readDf = writeAndRead("agg_decimal", 3);

        Dataset<Row> agg = readDf.agg(
                functions.sum("price"), functions.min("price"), functions.max("price"), functions.sum("ratio"));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals(0, new BigDecimal("4975.00").compareTo(row.getDecimal(0)));
        assertEquals(0, new BigDecimal("0.25").compareTo(row.getDecimal(1)));
        assertEquals(0, new BigDecimal("99.25").compareTo(row.getDecimal(2)));
        assertEquals(2475.0, row.getDouble(3));
    }

    @Test
    @DisplayName("avg() pushes down as its SUM and COUNT parts")
    public void testAveragePushesAsSumAndCount() {
        Dataset<Row> readDf = writeAndRead("agg_avg", 2);

        Dataset<Row> agg = readDf.agg(functions.avg("id"));
        assertNativeAggregate(agg);
        assertEquals(49.5, agg.collectAsList().get(0).getDouble(0));
    }

    @Test
    @DisplayName("A filter matching no rows yields SQL identities: zero counts, null min/max/sum")
    public void testAggregateOverNoMatchingRows() {
        Dataset<Row> readDf = writeAndRead("agg_empty_result", 2);

        Dataset<Row> agg = readDf.filter(readDf.col("id").lt(0))
                .agg(
                        functions.count(functions.lit(1)),
                        functions.count(readDf.col("qty")),
                        functions.sum("id"),
                        functions.min("id"),
                        functions.max("name"));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals(0L, row.getLong(0));
        assertEquals(0L, row.getLong(1));
        assertTrue(row.isNullAt(2), "sum over no rows must be null");
        assertTrue(row.isNullAt(3), "min over no rows must be null");
        assertTrue(row.isNullAt(4), "max over no rows must be null");
    }

    @Test
    @DisplayName("Aggregating a directory with no Vortex files yields the SQL identities")
    public void testAggregateEmptyDirectory() throws IOException {
        Path emptyDir = tempDir.resolve("agg_empty_dir");
        Files.createDirectories(emptyDir);

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .schema(SCHEMA)
                .option("path", emptyDir.toUri().toString())
                .load();

        Dataset<Row> agg = readDf.agg(functions.sum("id"), functions.count(readDf.col("qty")));
        Row row = agg.collectAsList().get(0);
        assertTrue(row.isNullAt(0), "sum over an empty table must be null");
        assertEquals(0L, row.getLong(1));
    }

    @Test
    @DisplayName("Semantics Vortex cannot match fall back to a regular scan and stay correct")
    public void testUnsupportedAggregatesFallBack() {
        Dataset<Row> readDf = writeAndRead("agg_fallback", 2);

        // Spark orders NaN above all other doubles; Vortex does not, so float min/max must not push.
        Dataset<Row> minDouble = readDf.agg(functions.min("ratio"));
        assertNoNativeAggregate(minDouble);
        assertEquals(0.0, minDouble.collectAsList().get(0).getDouble(0));

        // Spark wraps on long-sum overflow; Vortex saturates to null, so sum(long) must not push.
        Dataset<Row> sumLong = readDf.agg(functions.sum("big"));
        assertNoNativeAggregate(sumLong);
        assertEquals(4_950_000L, sumLong.collectAsList().get(0).getLong(0));

        Dataset<Row> distinct = readDf.agg(functions.count_distinct(functions.col("qty")));
        assertNoNativeAggregate(distinct);
        assertEquals(80L, distinct.collectAsList().get(0).getLong(0));

        Dataset<Row> grouped = readDf.groupBy(functions.col("id").mod(2).as("parity"))
                .agg(functions.sum("id"))
                .orderBy("parity");
        assertNoNativeAggregate(grouped);
        List<Row> rows = grouped.collectAsList();
        assertEquals(2450L, rows.get(0).getLong(1));
        assertEquals(2500L, rows.get(1).getLong(1));
    }

    @Test
    @DisplayName("MIN/MAX over every row including nulls skips the nulls")
    public void testMinMaxSkipsNulls() {
        Dataset<Row> readDf = writeAndRead("agg_nulls", 2);

        Dataset<Row> agg = readDf.agg(functions.min("qty"), functions.max("qty"), functions.sum("qty"));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertEquals(1, row.getInt(0));
        assertEquals(99, row.getInt(1));
        assertEquals(4000L, row.getLong(2));
    }

    @Test
    @DisplayName("An all-null slice keeps SUM null while COUNT(*) sees its rows")
    public void testAllNullSliceSumIsNull() {
        Dataset<Row> readDf = writeAndRead("agg_all_null", 2);

        // Every 5th qty is null, so this filter selects rows whose qty values are all null.
        Dataset<Row> agg = readDf.filter(readDf.col("id").isin(0, 5, 10, 15))
                .agg(functions.sum("qty"), functions.count(readDf.col("qty")), functions.count(functions.lit(1)));
        assertNativeAggregate(agg);
        Row row = agg.collectAsList().get(0);
        assertNull(row.get(0), "sum of an all-null column must be null");
        assertEquals(0L, row.getLong(1));
        assertEquals(4L, row.getLong(2));
    }
}
