// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Aggregate pushdown over a single file of {@code {i: int32?, d: float64?}}. */
public final class AggregatePushdownTest {
    @TempDir
    Path tempDir;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    private static Schema schema() {
        return new Schema(List.of(
                Field.nullable("i", new ArrowType.Int(32, true)),
                Field.nullable("d", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE))));
    }

    /** Writes {@code i = [1, null, 3, 2]} and {@code d = [1.5, NaN, null, 0.5]}. */
    private DataSource source(Session session, BufferAllocator allocator) throws IOException {
        String path = tempDir.resolve("aggregates.vortex").toAbsolutePath().toString();
        try (VortexWriter writer =
                        VortexWriter.builder(session, path, schema(), allocator).build();
                VectorSchemaRoot root = VectorSchemaRoot.create(schema(), allocator)) {
            IntVector i = (IntVector) root.getVector("i");
            Float8Vector d = (Float8Vector) root.getVector("d");
            i.allocateNew(4);
            d.allocateNew(4);

            i.setSafe(0, 1);
            i.setNull(1);
            i.setSafe(2, 3);
            i.setSafe(3, 2);
            d.setSafe(0, 1.5);
            d.setSafe(1, Double.NaN);
            d.setNull(2);
            d.setSafe(3, 0.5);
            root.setRowCount(4);

            try (ArrowArray array = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
                writer.writeBatch(array.memoryAddress(), arrowSchema.memoryAddress());
            }
            writer.finish();
        }
        return DataSource.open(session, path);
    }

    /** Assertions run against the single aggregate row a partition produces. */
    @FunctionalInterface
    private interface RowAssertions {
        void check(VectorSchemaRoot row);
    }

    /** Reads the single aggregate row every partition of an aggregate scan produces. */
    private static void aggregateRow(
            DataSource source,
            AggregatePlan plan,
            BufferAllocator allocator,
            Optional<Expression> filter,
            RowAssertions assertions)
            throws IOException {
        ImmutableScanOptions.Builder options = ScanOptions.builder().aggregates(plan);
        filter.ifPresent(options::filter);

        Scan scan = source.scan(options.build());
        assertTrue(scan.hasNext(), "aggregate scan produces at least one partition");
        Partition partition = scan.next();
        assertFalse(scan.hasNext(), "one file produces one partition");

        try (ArrowReader reader = partition.scanArrow(allocator)) {
            assertTrue(reader.loadNextBatch(), "aggregates produce one batch");
            VectorSchemaRoot row = reader.getVectorSchemaRoot();
            assertEquals(1, row.getRowCount(), "aggregates produce one row");
            assertions.check(row);
            assertFalse(reader.loadNextBatch(), "aggregates produce a single batch");
        }
    }

    @Test
    public void aggregatesAreComputedFromTheScan() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        List<Aggregate> aggregates = List.of(
                Aggregate.countStar(),
                Aggregate.count("i"),
                Aggregate.min("i"),
                Aggregate.max("i"),
                Aggregate.sum("i"));
        AggregatePlan plan =
                AggregatePlan.tryCreate(source, aggregates).orElseThrow(() -> new AssertionError("pushable"));

        assertEquals(aggregates, plan.aggregates());
        Schema aggregateSchema = plan.arrowSchema(allocator);
        assertEquals(5, aggregateSchema.getFields().size());
        assertEquals(
                new ArrowType.Int(64, true), aggregateSchema.getFields().get(0).getType());
        assertEquals(
                new ArrowType.Int(32, true), aggregateSchema.getFields().get(2).getType());
        assertEquals(
                new ArrowType.Int(64, true), aggregateSchema.getFields().get(4).getType());

        aggregateRow(source, plan, allocator, Optional.empty(), row -> {
            assertEquals(aggregateSchema.getFields(), row.getSchema().getFields());
            assertEquals(4L, ((BigIntVector) row.getVector(0)).get(0), "count(*) counts null rows");
            assertEquals(3L, ((BigIntVector) row.getVector(1)).get(0), "count(i) skips nulls");
            assertEquals(1, ((IntVector) row.getVector(2)).get(0), "min(i)");
            assertEquals(3, ((IntVector) row.getVector(3)).get(0), "max(i)");
            assertEquals(6L, ((BigIntVector) row.getVector(4)).get(0), "sum(i) widens to int64");
        });
    }

    @Test
    public void filteredCountStarCountsSurvivingRows() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        AggregatePlan plan = AggregatePlan.tryCreate(source, List.of(Aggregate.countStar()))
                .orElseThrow(() -> new AssertionError("count(*) is always pushable"));

        Expression filter = Expression.binary(Expression.BinaryOp.GT, Expression.column("i"), Expression.literal(1));
        aggregateRow(
                source,
                plan,
                allocator,
                Optional.of(filter),
                row -> assertEquals(2L, ((BigIntVector) row.getVector(0)).get(0), "i > 1 selects 3 and 2"));
    }

    @Test
    public void unfilteredCountStarComesFromMetadata() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        AggregatePlan plan = AggregatePlan.tryCreate(source, List.of(Aggregate.countStar(), Aggregate.countStar()))
                .orElseThrow(() -> new AssertionError("count(*) is always pushable"));

        // Answered from the file's row count, without decoding any column.
        aggregateRow(source, plan, allocator, Optional.empty(), row -> {
            assertEquals(4L, ((BigIntVector) row.getVector(0)).get(0));
            assertEquals(4L, ((BigIntVector) row.getVector(1)).get(0), "repeated count(*)");
        });
    }

    @Test
    public void nanIsAValueForCountMaxAndSum() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        AggregatePlan plan = AggregatePlan.tryCreate(
                        source, List.of(Aggregate.count("d"), Aggregate.max("d"), Aggregate.sum("d")))
                .orElseThrow(() -> new AssertionError("pushable"));

        aggregateRow(source, plan, allocator, Optional.empty(), row -> {
            assertEquals(3L, ((BigIntVector) row.getVector(0)).get(0), "NaN is a value, null is not");
            assertTrue(Double.isNaN(((Float8Vector) row.getVector(1)).get(0)), "NaN is the largest value");
            assertTrue(Double.isNaN(((Float8Vector) row.getVector(2)).get(0)), "NaN poisons the sum");
        });
    }

    @Test
    public void unpushableAggregatesReturnEmpty() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        assertTrue(
                AggregatePlan.tryCreate(source, List.of(Aggregate.min("d"))).isEmpty(),
                "min over a float column cannot reproduce engine NaN ordering");
        assertTrue(
                AggregatePlan.tryCreate(source, List.of(Aggregate.max("absent")))
                        .isEmpty(),
                "a column outside the files is not pushable");
        assertTrue(
                AggregatePlan.tryCreate(source, List.of(Aggregate.countStar(), Aggregate.min("d")))
                        .isEmpty(),
                "one unpushable aggregate rejects the whole aggregation");
    }

    @Test
    public void aggregatesRejectAProjection() throws IOException {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource source = source(session, allocator);

        AggregatePlan plan = AggregatePlan.tryCreate(source, List.of(Aggregate.countStar()))
                .orElseThrow(() -> new AssertionError("pushable"));

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () -> ScanOptions.builder()
                .aggregates(plan)
                .projection(Expression.select(new String[] {"i"}, Expression.root()))
                .build());
        assertTrue(error.getMessage().contains("mutually exclusive"), error.getMessage());
    }

    @Test
    public void aggregateFactoriesValidateTheirColumn() {
        assertThrows(NullPointerException.class, () -> Aggregate.min(null));
        assertThrows(IllegalArgumentException.class, () -> Aggregate.sum(""));
        assertTrue(Aggregate.countStar().column().isEmpty());
        assertEquals(Optional.of("i"), Aggregate.max("i").column());
        assertEquals(Aggregate.count("i"), Aggregate.count("i"));
        assertFalse(Aggregate.count("i").equals(Aggregate.count("d")));
    }
}
