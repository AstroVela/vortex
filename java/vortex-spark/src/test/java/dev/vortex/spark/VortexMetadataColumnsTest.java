// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
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
 * Tests the {@code _file} and {@code _pos} metadata columns of Vortex tables.
 *
 * <p>{@code _file} must carry the path of the file each row was read from; {@code _pos} must carry the row's position
 * within that file, assigned before any pushed filter. A data column with the same name shadows the metadata column.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexMetadataColumnsTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexMetadataColumnsTest")
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

    private Dataset<Row> writeOrderedAndRead(String dir, int rows) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < rows; i++) {
            data.add(RowFactory.create(i, "name-" + i));
        }
        Path outputPath = tempDir.resolve(dir);
        // coalesce(1) keeps the input order, so row i has id == i == position in the single file.
        spark.createDataFrame(data, SCHEMA)
                .coalesce(1)
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
    @DisplayName("_file carries the source file path and _pos the row position within it")
    public void testFileAndPos() {
        Dataset<Row> readDf = writeOrderedAndRead("meta_basic", 50);

        List<Row> rows = readDf.select("id", "_file", "_pos").collectAsList();
        assertEquals(50, rows.size());
        for (Row row : rows) {
            assertTrue(row.getString(1).endsWith(".vortex"), "unexpected _file: " + row.getString(1));
            assertEquals(
                    row.getInt(0),
                    (int) row.getLong(2),
                    "with an ordered single-file write, _pos must equal the row's id");
        }
    }

    @Test
    @DisplayName("_pos reflects pre-filter row positions when a filter is pushed down")
    public void testPosWithPushedFilter() {
        Dataset<Row> readDf = writeOrderedAndRead("meta_filtered", 100);

        List<Row> rows =
                readDf.select("id", "_pos").filter(readDf.col("id").geq(95)).collectAsList();
        assertEquals(5, rows.size());
        for (Row row : rows) {
            assertEquals(
                    row.getInt(0), (int) row.getLong(1), "_pos must keep the original file position of surviving rows");
        }
    }

    @Test
    @DisplayName("_file distinguishes files of a multi-file dataset")
    public void testFileAcrossMultipleFiles() {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            data.add(RowFactory.create(i, "name-" + i));
        }
        Path outputPath = tempDir.resolve("meta_multi");
        spark.createDataFrame(data, SCHEMA)
                .repartition(4)
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        List<Row> rows = readDf.select("_file", "_pos").collectAsList();
        assertEquals(40, rows.size());
        Set<String> files = rows.stream().map(r -> r.getString(0)).collect(Collectors.toSet());
        assertEquals(4, files.size(), "expected rows from four distinct files");

        // Within each file, positions must start at 0 and be dense.
        for (String file : files) {
            List<Long> positions = rows.stream()
                    .filter(r -> file.equals(r.getString(0)))
                    .map(r -> r.getLong(1))
                    .sorted()
                    .collect(Collectors.toList());
            for (int i = 0; i < positions.size(); i++) {
                assertEquals(i, positions.get(i), "positions within " + file + " must be dense from 0");
            }
        }
    }

    @Test
    @DisplayName("Metadata columns work together with partition columns")
    public void testMetadataWithPartitionColumns() {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            data.add(RowFactory.create(i, "region-" + (i % 2)));
        }
        Path outputPath = tempDir.resolve("meta_partitioned");
        spark.createDataFrame(
                        data,
                        DataTypes.createStructType(List.of(
                                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                                DataTypes.createStructField("region", DataTypes.StringType, true))))
                .repartition(1)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        List<Row> rows = readDf.select("region", "_file", "_pos").collectAsList();
        assertEquals(20, rows.size());
        for (Row row : rows) {
            assertTrue(
                    row.getString(1).contains("region=" + row.getString(0)),
                    "_file must point into the row's partition directory: " + row.getString(1));
        }
    }

    @Test
    @DisplayName("A data column named _file shadows the metadata column")
    public void testDataColumnShadowsMetadata() {
        List<Row> data = List.of(RowFactory.create(1, "data-value"));
        Path outputPath = tempDir.resolve("meta_shadow");
        spark.createDataFrame(
                        data,
                        DataTypes.createStructType(List.of(
                                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                                DataTypes.createStructField("_file", DataTypes.StringType, true))))
                .write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        Dataset<Row> readDf = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load();

        List<Row> rows = readDf.select("_file").collectAsList();
        assertEquals(1, rows.size());
        assertEquals("data-value", rows.get(0).getString(0), "the data column must shadow the metadata column");

        // The unshadowed metadata column is still available.
        HashSet<String> posValues = new HashSet<>();
        for (Row row : readDf.select("_pos").collectAsList()) {
            posValues.add(Long.toString(row.getLong(0)));
        }
        assertEquals(Set.of("0"), posValues);
    }
}
