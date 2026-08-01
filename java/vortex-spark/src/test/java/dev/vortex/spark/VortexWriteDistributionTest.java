// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
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
 * Tests that partitioned Vortex writes request a clustered distribution on the partition columns, so every Hive
 * partition directory receives exactly one file regardless of the input parallelism.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexWriteDistributionTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("region", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexWriteDistributionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "4")
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

    private static List<Path> vortexFilesUnder(Path dir) throws IOException {
        try (Stream<Path> stream = Files.walk(dir)) {
            return stream.filter(p -> p.toString().endsWith(".vortex")).collect(Collectors.toList());
        }
    }

    @Test
    @DisplayName("Partitioned writes cluster rows so each partition directory holds exactly one file")
    public void testPartitionedWriteClustering() throws IOException {
        Path outputPath = tempDir.resolve("clustered");
        // Many input partitions: without a clustered write distribution this would create one
        // file per (task, region) pair; with it, all rows of a region reach a single task.
        testData(100)
                .repartition(8)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        for (String region : List.of("region-0", "region-1")) {
            List<Path> files = vortexFilesUnder(outputPath.resolve("region=" + region));
            assertEquals(1, files.size(), "expected exactly one file for " + region + ", got " + files);
        }

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();
        assertEquals(100, readDf.count());
        assertEquals(50, readDf.filter(readDf.col("region").equalTo("region-0")).count());
    }

    @Test
    @DisplayName("Unpartitioned writes keep the input parallelism")
    public void testUnpartitionedWriteUnchanged() throws IOException {
        Path outputPath = tempDir.resolve("unpartitioned");
        testData(30)
                .repartition(3)
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        assertEquals(3, vortexFilesUnder(outputPath).size(), "unspecified distribution must not add a shuffle");
        assertEquals(
                30,
                spark.read()
                        .format("vortex")
                        .option("path", outputPath.toUri().toString())
                        .load()
                        .count());
    }
}
