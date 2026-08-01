// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsRuntimeV2Filtering;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests runtime (dynamic partition pruning style) filtering of Vortex file partitions.
 *
 * <p>{@code VortexScan} reports partition columns as filterable attributes; runtime predicates installed through
 * {@link SupportsRuntimeV2Filtering#filter} must skip files whose Hive-style partition values fail the predicates, and
 * must never drop files the predicates cannot decide.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexRuntimeFilteringTest {

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexRuntimeFilteringTest")
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

    private String writePartitioned(String dir) {
        List<Row> data = new ArrayList<>();
        for (int i = 0; i < 40; i++) {
            data.add(RowFactory.create(i, "region-" + (i % 4), i * 10L));
        }
        Path outputPath = tempDir.resolve(dir);
        spark.createDataFrame(
                        data,
                        DataTypes.createStructType(List.of(
                                DataTypes.createStructField("id", DataTypes.IntegerType, false),
                                DataTypes.createStructField("region", DataTypes.StringType, true),
                                DataTypes.createStructField("amount", DataTypes.LongType, false))))
                // One writer task, so each region directory holds exactly one file.
                .repartition(1)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();
        return outputPath.toUri().toString();
    }

    private Scan buildScan(String path) {
        VortexDataSourceV2 provider = new VortexDataSourceV2();
        CaseInsensitiveStringMap options = new CaseInsensitiveStringMap(Map.of("path", path));
        var table = provider.getTable(provider.inferSchema(options), provider.inferPartitioning(options), options);
        ScanBuilder builder = ((SupportsRead) table).newScanBuilder(options);
        return builder.build();
    }

    @Test
    @DisplayName("Runtime IN predicates on the partition column prune whole files")
    public void testRuntimePruning() {
        String path = writePartitioned("runtime_pruning");

        Scan scan = buildScan(path);
        Batch unfiltered = scan.toBatch();
        int allPartitions = unfiltered.planInputPartitions().length;
        assertEquals(4, allPartitions, "expected one input partition per region directory");

        SupportsRuntimeV2Filtering filterable = (SupportsRuntimeV2Filtering) scan;
        NamedReference[] attributes = filterable.filterAttributes();
        assertEquals(1, attributes.length);
        assertEquals("region", attributes[0].fieldNames()[0]);

        NamedReference region = Expressions.column("region");
        Predicate in = new Predicate("IN", new Expression[] {
            region,
            LiteralValue.apply(UTF8String.fromString("region-1"), DataTypes.StringType),
            LiteralValue.apply(UTF8String.fromString("region-3"), DataTypes.StringType)
        });
        filterable.filter(new Predicate[] {in});

        InputPartition[] pruned = scan.toBatch().planInputPartitions();
        assertEquals(2, pruned.length, "expected only the region-1 and region-3 files to survive");
    }

    @Test
    @DisplayName("Unevaluable runtime predicates keep every file")
    public void testUnknownPredicatesKeepFiles() {
        String path = writePartitioned("runtime_unknown");

        Scan scan = buildScan(path);
        SupportsRuntimeV2Filtering filterable = (SupportsRuntimeV2Filtering) scan;

        // Predicate on a non-partition column: must not prune anything.
        Predicate onData = new Predicate(
                "IN", new Expression[] {Expressions.column("id"), LiteralValue.apply(1, DataTypes.IntegerType)});
        // Unsupported operator on the partition column: must not prune anything.
        Predicate unsupported = new Predicate("STARTS_WITH", new Expression[] {
            Expressions.column("region"), LiteralValue.apply(UTF8String.fromString("region"), DataTypes.StringType)
        });
        filterable.filter(new Predicate[] {onData, unsupported});

        assertEquals(4, scan.toBatch().planInputPartitions().length);
    }

    @Test
    @DisplayName("A join on the partition column returns correct results with DPP enabled")
    public void testJoinCorrectnessWithDynamicPruning() {
        String path = writePartitioned("runtime_join");

        Dataset<Row> fact = spark.read().format("vortex").option("path", path).load();
        Dataset<Row> dim = spark.createDataFrame(
                Arrays.asList(RowFactory.create("region-2", "west")),
                DataTypes.createStructType(List.of(
                        DataTypes.createStructField("region", DataTypes.StringType, true),
                        DataTypes.createStructField("label", DataTypes.StringType, true))));

        List<Row> joined = fact.join(dim, "region").select("id", "label").collectAsList();
        assertEquals(10, joined.size(), "10 of 40 rows live in region-2");
        assertTrue(joined.stream().allMatch(r -> "west".equals(r.getString(1))));
    }
}
