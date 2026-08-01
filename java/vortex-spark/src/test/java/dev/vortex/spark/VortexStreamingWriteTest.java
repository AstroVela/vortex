// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.apache.spark.sql.streaming.Trigger;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests Structured Streaming writes to the Vortex sink: micro-batches append epoch-named Vortex files, restarts resume
 * from the checkpoint, and partitioned streams write Hive-style directories.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexStreamingWriteTest {

    private static final StructType SCHEMA = DataTypes.createStructType(List.of(
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("region", DataTypes.StringType, true)));

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexStreamingWriteTest")
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

    private void writeInputFile(Path inputDir, String name, String... jsonLines) throws Exception {
        Files.createDirectories(inputDir);
        Files.writeString(inputDir.resolve(name), String.join("\n", jsonLines));
    }

    private void runStream(Path inputDir, Path outputDir, Path checkpointDir) throws Exception {
        Dataset<Row> stream =
                spark.readStream().schema(SCHEMA).json(inputDir.toUri().toString());
        StreamingQuery query = stream.writeStream()
                .format("vortex")
                .option("path", outputDir.toUri().toString())
                .option("checkpointLocation", checkpointDir.toUri().toString())
                .trigger(Trigger.AvailableNow())
                .start();
        assertTrue(query.awaitTermination(120_000), "the streaming query did not finish in time");
    }

    private Dataset<Row> readOutput(Path outputDir) {
        return spark.read()
                .format("vortex")
                .option("path", outputDir.toUri().toString())
                .load();
    }

    @Test
    @DisplayName("Micro-batches append epoch-named Vortex files and restarts resume from the checkpoint")
    public void testStreamingAppend() throws Exception {
        Path inputDir = tempDir.resolve("stream_in");
        Path outputDir = tempDir.resolve("stream_out");
        Path checkpointDir = tempDir.resolve("stream_ck");

        writeInputFile(inputDir, "batch-0.json", "{\"id\":1,\"region\":\"east\"}", "{\"id\":2,\"region\":\"west\"}");
        runStream(inputDir, outputDir, checkpointDir);
        assertEquals(2, readOutput(outputDir).count());

        // A second run with the same checkpoint only processes the newly arrived file.
        writeInputFile(inputDir, "batch-1.json", "{\"id\":3,\"region\":\"east\"}");
        runStream(inputDir, outputDir, checkpointDir);

        Dataset<Row> result = readOutput(outputDir);
        assertEquals(3, result.count());
        assertEquals(
                List.of(1, 2, 3),
                result.orderBy("id").collectAsList().stream()
                        .map(r -> r.getInt(0))
                        .collect(Collectors.toList()));

        try (var files = Files.walk(outputDir)) {
            List<String> names = files.map(p -> p.getFileName().toString())
                    .filter(n -> n.endsWith(".vortex"))
                    .collect(Collectors.toList());
            assertTrue(
                    names.stream().allMatch(n -> n.contains("-epoch-")),
                    "streaming files must carry the epoch in their name: " + names);
        }
    }

    @Test
    @DisplayName("Streaming appends to an existing partitioned dataset follow its directory layout")
    public void testPartitionedStreamingWrite() throws Exception {
        Path inputDir = tempDir.resolve("stream_part_in");
        Path outputDir = tempDir.resolve("stream_part_out");
        Path checkpointDir = tempDir.resolve("stream_part_ck");

        // Seed a partitioned dataset with a batch write; the streaming sink infers the partition
        // layout from it. (Spark does not forward writeStream.partitionBy to DSv2 sinks.)
        spark.createDataFrame(List.of(RowFactory.create(1, "east"), RowFactory.create(2, "west")), SCHEMA)
                .write()
                .format("vortex")
                .partitionBy("region")
                .option("path", outputDir.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        writeInputFile(inputDir, "batch-0.json", "{\"id\":3,\"region\":\"east\"}", "{\"id\":4,\"region\":\"west\"}");
        runStream(inputDir, outputDir, checkpointDir);

        assertTrue(Files.isDirectory(outputDir.resolve("region=east")), "expected a region=east directory");
        assertTrue(Files.isDirectory(outputDir.resolve("region=west")), "expected a region=west directory");

        Dataset<Row> result = readOutput(outputDir);
        assertEquals(4, result.count());
        assertEquals(2, result.filter(result.col("region").equalTo("east")).count());

        // The streamed rows landed inside the partition directories, not at the dataset root.
        try (var files = Files.list(outputDir)) {
            assertTrue(
                    files.filter(f -> f.toString().endsWith(".vortex"))
                            .findAny()
                            .isEmpty(),
                    "streamed files must be placed in partition directories");
        }
    }
}
