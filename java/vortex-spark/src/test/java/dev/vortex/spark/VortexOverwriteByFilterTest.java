// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
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
 * Tests overwrite-by-filter for Vortex tables: {@code DataFrameWriterV2.overwrite(condition)} and {@code INSERT
 * OVERWRITE ... PARTITION (...)} replace only the partitions selected by the condition. Conditions on data columns
 * cannot be applied at file granularity and must be rejected.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexOverwriteByFilterTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("region", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    static Path warehouseDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexOverwriteByFilterTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.catalog.spark_catalog", VortexSessionCatalog.class.getName())
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    private void createAndSeed(String table) {
        spark.sql(String.format("CREATE TABLE %s (id INT, region STRING) USING vortex PARTITIONED BY (region)", table));
        spark.sql(String.format(
                "INSERT INTO %s VALUES (1, 'east'), (2, 'east'), (3, 'west'), (4, 'west'), (5, 'north')", table));
        assertEquals(5, spark.table(table).count());
    }

    @Test
    @DisplayName("DataFrameWriterV2.overwrite replaces only the partitions matching the condition")
    public void testOverwriteByPartitionCondition() throws Exception {
        createAndSeed("owf_writer_v2");

        Dataset<Row> replacement = spark.createDataFrame(
                Arrays.asList(RowFactory.create(1000, "west"), RowFactory.create(1001, "west")), SCHEMA);
        replacement.writeTo("owf_writer_v2").overwrite(functions.col("region").equalTo("west"));

        Dataset<Row> result = spark.table("owf_writer_v2");
        assertEquals(5, result.count(), "east keeps 2 rows, north keeps 1, west is replaced by 2 new rows");
        assertEquals(2, result.filter(result.col("region").equalTo("east")).count());
        assertEquals(1, result.filter(result.col("region").equalTo("north")).count());
        for (Row row : result.filter(result.col("region").equalTo("west")).collectAsList()) {
            assertTrue(row.getInt(0) >= 1000, "west must only contain new rows, got id=" + row.getInt(0));
        }
        spark.sql("DROP TABLE owf_writer_v2");
    }

    @Test
    @DisplayName("INSERT OVERWRITE with a static partition spec replaces only that partition")
    public void testInsertOverwriteStaticPartition() {
        createAndSeed("owf_static_partition");

        spark.sql("INSERT OVERWRITE owf_static_partition PARTITION (region = 'east') VALUES (2000), (2001), (2002)");

        Dataset<Row> result = spark.table("owf_static_partition");
        assertEquals(6, result.count(), "east is replaced by 3 rows; west keeps 2, north keeps 1");
        assertEquals(3, result.filter(result.col("region").equalTo("east")).count());
        assertEquals(2, result.filter(result.col("region").equalTo("west")).count());
        for (Row row : result.filter(result.col("region").equalTo("east")).collectAsList()) {
            assertTrue(row.getInt(0) >= 2000, "east must only contain new rows, got id=" + row.getInt(0));
        }
        spark.sql("DROP TABLE owf_static_partition");
    }

    @Test
    @DisplayName("Overwrite conditions on data columns are rejected")
    public void testOverwriteByDataColumnRejected() {
        createAndSeed("owf_rejected");

        Dataset<Row> replacement = spark.createDataFrame(Arrays.asList(RowFactory.create(1000, "west")), SCHEMA);
        Exception e = assertThrows(
                Exception.class,
                () -> replacement
                        .writeTo("owf_rejected")
                        .overwrite(functions.col("id").geq(3)),
                "a row-level overwrite condition cannot be applied at file granularity");
        assertTrue(
                e.getMessage().contains("does not support overwrite by expression"),
                "unexpected rejection: " + e.getMessage());

        assertEquals(5, spark.table("owf_rejected").count(), "the rejected overwrite must not modify the table");
        spark.sql("DROP TABLE owf_rejected");
    }
}
