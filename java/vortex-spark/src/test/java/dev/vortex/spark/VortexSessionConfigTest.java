// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Tests that {@code spark.datasource.vortex.*} session configurations propagate into Vortex read and write options via
 * {@link org.apache.spark.sql.connector.catalog.SessionConfigSupport}.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSessionConfigTest {

    /**
     * Session provider that counts instantiations. Selected through the {@code vortex.session.provider} option, which
     * this test supplies only via a session configuration rather than a per-read option.
     */
    public static final class RecordingSessionProvider implements VortexSessionProvider {
        private static final AtomicInteger CALLS = new AtomicInteger();

        @Override
        public Session get() {
            CALLS.incrementAndGet();
            return VortexSparkSession.get();
        }
    }

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexSessionConfigTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.shuffle.partitions", "2")
                .config("spark.ui.enabled", "false")
                .config("spark.datasource.vortex.vortex.session.provider", RecordingSessionProvider.class.getName())
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    @DisplayName("spark.datasource.vortex.* session configs reach reads and writes as options")
    public void testSessionConfigPropagation() {
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(RowFactory.create(1, "a"), RowFactory.create(2, "b")),
                DataTypes.createStructType(List.of(
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true))));

        Path outputPath = tempDir.resolve("session_config");
        df.write()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .mode(SaveMode.Overwrite)
                .save();

        List<Row> rows = spark.read()
                .format("vortex")
                .option("path", outputPath.toUri().toString())
                .load()
                .orderBy("id")
                .collectAsList();

        assertEquals(2, rows.size());
        assertEquals(1, rows.get(0).getInt(0));

        // The provider is configured exclusively through the session configuration, so it can only
        // have been reached if SessionConfigSupport injected vortex.session.provider into the options.
        assertTrue(
                RecordingSessionProvider.CALLS.get() >= 1,
                "expected the session-config provider to be used at least once, calls="
                        + RecordingSessionProvider.CALLS.get());
    }
}
