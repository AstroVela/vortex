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
 * Tests TABLESAMPLE pushdown into the Vortex datasource: Bernoulli samples are answered through a native row selection
 * (the Sample operator disappears from the plan), results are repeatable for a fixed seed, and unsupported sampling
 * modes fall back to Spark.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexTableSampleTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexTableSampleTest")
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

    @Test
    @DisplayName("Bernoulli samples push into the scan and are repeatable per seed")
    public void testSamplePushdown() {
        Dataset<Row> readDf = writeAndRead("sample_basic", 2000, 2);

        Dataset<Row> sampled = readDf.sample(0.5, 42L);
        String plan = sampled.queryExecution().executedPlan().toString();
        // The scan description mentions the pushed VortexTableSample; strip it before checking
        // that no Sample OPERATOR is left in the plan.
        assertTrue(plan.contains("sample=VortexTableSample"), "expected the pushed sample in the scan: " + plan);
        assertFalse(
                plan.replace("VortexTableSample", "").contains("Sample"),
                "the Sample operator must be absorbed by the scan: " + plan);

        long first = sampled.count();
        assertTrue(first > 600 && first < 1400, "a 50% sample of 2000 rows should be near 1000, got " + first);
        assertEquals(first, readDf.sample(0.5, 42L).count(), "sampling must be repeatable for a fixed seed");

        long otherSeed = readDf.sample(0.5, 43L).count();
        assertTrue(otherSeed > 600 && otherSeed < 1400, "unexpected sample size " + otherSeed);
    }

    @Test
    @DisplayName("Sampling with replacement falls back to Spark")
    public void testSampleWithReplacementFallsBack() {
        Dataset<Row> readDf = writeAndRead("sample_replacement", 500, 1);

        Dataset<Row> sampled = readDf.sample(true, 0.5, 42L);
        String plan = sampled.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("Sample"), "with-replacement sampling must stay in Spark: " + plan);
        long count = sampled.count();
        assertTrue(count > 100 && count < 500, "unexpected with-replacement sample size " + count);
    }

    @Test
    @DisplayName("A pushed sample composes with a pushed filter")
    public void testSampleWithFilter() {
        Dataset<Row> readDf = writeAndRead("sample_filter", 2000, 2);

        List<Row> rows =
                readDf.sample(0.5, 42L).filter(readDf.col("id").lt(1000)).collectAsList();
        assertTrue(rows.size() > 300 && rows.size() < 700, "expected roughly half of 1000 rows, got " + rows.size());
        for (Row row : rows) {
            assertTrue(row.getInt(0) < 1000, "filter must apply to the sampled rows");
        }
    }

    @Test
    @DisplayName("TABLESAMPLE in SQL pushes into the scan")
    public void testSqlTableSample() {
        Dataset<Row> readDf = writeAndRead("sample_sql", 2000, 1);
        readDf.createOrReplaceTempView("sample_sql_view");

        Dataset<Row> sampled = spark.sql("SELECT * FROM sample_sql_view TABLESAMPLE (25 PERCENT) REPEATABLE (7)");
        String plan = sampled.queryExecution().executedPlan().toString();
        assertTrue(plan.contains("sample=VortexTableSample"), "expected the pushed sample in the scan: " + plan);
        assertFalse(
                plan.replace("VortexTableSample", "").contains("Sample"),
                "the Sample operator must be absorbed by the scan: " + plan);
        long count = sampled.count();
        assertTrue(count > 300 && count < 700, "a 25% sample of 2000 rows should be near 500, got " + count);
    }
}
