// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests dynamic partition overwrite for Vortex tables: with {@code spark.sql.sources.partitionOverwriteMode=dynamic},
 * {@code INSERT OVERWRITE} only replaces the partition directories that receive new data and leaves other partitions
 * untouched.
 *
 * <p>Dynamic overwrite is only planned by Spark for catalog tables, so these tests go through
 * {@link VortexSessionCatalog} managed tables rather than path-based writes.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexDynamicOverwriteTest {

    private SparkSession spark;

    @TempDir
    static Path warehouseDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexDynamicOverwriteTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.catalog.spark_catalog", VortexSessionCatalog.class.getName())
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .getOrCreate();
    }

    @AfterEach
    public void resetConf() {
        spark.conf().unset("spark.sql.sources.partitionOverwriteMode");
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
    @DisplayName("Dynamic INSERT OVERWRITE replaces only the partitions receiving new data")
    public void testDynamicOverwrite() {
        createAndSeed("dyn_overwrite");

        spark.conf().set("spark.sql.sources.partitionOverwriteMode", "dynamic");
        spark.sql("INSERT OVERWRITE dyn_overwrite VALUES (1000, 'west'), (1001, 'west')");

        Dataset<Row> result = spark.table("dyn_overwrite");
        assertEquals(5, result.count(), "east keeps 2 rows, north keeps 1, west is replaced by 2 new rows");
        assertEquals(2, result.filter(result.col("region").equalTo("east")).count());
        assertEquals(1, result.filter(result.col("region").equalTo("north")).count());

        List<Row> westRows = result.filter(result.col("region").equalTo("west")).collectAsList();
        assertEquals(2, westRows.size());
        for (Row row : westRows) {
            assertTrue(row.getInt(0) >= 1000, "west must only contain the newly written rows, got id=" + row.getInt(0));
        }
        spark.sql("DROP TABLE dyn_overwrite");
    }

    @Test
    @DisplayName("Static INSERT OVERWRITE still replaces the whole table")
    public void testStaticOverwriteUnchanged() {
        createAndSeed("static_overwrite");

        spark.sql("INSERT OVERWRITE static_overwrite VALUES (1000, 'west')");

        Dataset<Row> result = spark.table("static_overwrite");
        assertEquals(1, result.count(), "static overwrite drops every previous partition");
        assertEquals(0, result.filter(result.col("region").equalTo("east")).count());
        spark.sql("DROP TABLE static_overwrite");
    }

    @Test
    @DisplayName("Repeated dynamic overwrites of the same partition converge to the last write")
    public void testRepeatedDynamicOverwrite() {
        createAndSeed("dyn_repeated");

        spark.conf().set("spark.sql.sources.partitionOverwriteMode", "dynamic");
        spark.sql("INSERT OVERWRITE dyn_repeated VALUES (100, 'west')");
        spark.sql("INSERT OVERWRITE dyn_repeated VALUES (200, 'west'), (201, 'west')");

        Dataset<Row> result = spark.table("dyn_repeated");
        assertEquals(5, result.count(), "east keeps 2 rows, north keeps 1, west holds the 2 rows of the last write");
        List<Row> westRows = result.filter(result.col("region").equalTo("west")).collectAsList();
        assertEquals(2, westRows.size());
        for (Row row : westRows) {
            assertTrue(row.getInt(0) >= 200 && row.getInt(0) < 300, "unexpected id " + row.getInt(0));
        }
        spark.sql("DROP TABLE dyn_repeated");
    }
}
