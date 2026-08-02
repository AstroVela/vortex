// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Aggregate;
import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.GenericInternalRow;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * Emits a single row of natively-computed aggregate values for one Vortex file, for pushed-down aggregations.
 *
 * <p>Delegates to {@link DataSource#aggregate}, which streams the (optionally filtered) file through Vortex's
 * accumulators and returns a one-row Arrow batch. A partition with no paths (the zero-file table case) emits the
 * aggregation identity instead: zero counts and null min/max/sum.
 */
final class VortexAggregatePartitionReader implements PartitionReader<InternalRow> {

    private final VortexAggregateScan.VortexAggregatePartition partition;
    private final List<PushedAggregate> aggregates;
    private final Predicate[] pushedPredicates;
    private boolean emitted;

    VortexAggregatePartitionReader(
            VortexAggregateScan.VortexAggregatePartition partition,
            List<PushedAggregate> aggregates,
            Predicate[] pushedPredicates) {
        this.partition = partition;
        this.aggregates = aggregates;
        this.pushedPredicates = pushedPredicates;
    }

    @Override
    public boolean next() {
        if (emitted) {
            return false;
        }
        emitted = true;
        return true;
    }

    @Override
    public InternalRow get() {
        Object[] values = new Object[aggregates.size()];
        if (partition.paths().isEmpty()) {
            for (int i = 0; i < values.length; i++) {
                values[i] = aggregates.get(i).nullable() ? null : 0L;
            }
            return new GenericInternalRow(values);
        }

        Session session = VortexSparkSession.get(partition.formatOptions());
        DataSource dataSource = DataSource.open(session, partition.paths(), partition.formatOptions());
        List<Aggregate> nativeAggregates =
                aggregates.stream().map(PushedAggregate::toApi).collect(Collectors.toList());
        Expression filter =
                VortexPartitionReader.buildFilterExpression(pushedPredicates).orElse(null);

        try (ArrowReader reader = dataSource.aggregate(nativeAggregates, filter, ArrowAllocation.rootAllocator())) {
            if (!reader.loadNextBatch()) {
                throw new IllegalStateException("native aggregate returned no batch for " + partition.paths());
            }
            VectorSchemaRoot root = reader.getVectorSchemaRoot();
            for (int i = 0; i < aggregates.size(); i++) {
                ColumnVector vector =
                        new VortexArrowColumnVector(root.getFieldVectors().get(i));
                values[i] = readValue(vector, aggregates.get(i).resultType());
            }
        } catch (IOException e) {
            throw new RuntimeException("failed to aggregate " + partition.paths(), e);
        }
        return new GenericInternalRow(values);
    }

    @Override
    public void close() {}

    /**
     * Copies row 0 of the vector out as the boxed value Spark's {@link GenericInternalRow} expects for {@code type}.
     * Values are copied because the backing Arrow memory is released when the reader closes.
     */
    private static Object readValue(ColumnVector vector, DataType type) {
        if (vector.isNullAt(0)) {
            return null;
        }
        if (type instanceof BooleanType) {
            return vector.getBoolean(0);
        }
        if (type instanceof ByteType) {
            return vector.getByte(0);
        }
        if (type instanceof ShortType) {
            return vector.getShort(0);
        }
        if (type instanceof IntegerType || type instanceof DateType) {
            return vector.getInt(0);
        }
        if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            return vector.getLong(0);
        }
        if (type instanceof FloatType) {
            return vector.getFloat(0);
        }
        if (type instanceof DoubleType) {
            return vector.getDouble(0);
        }
        if (type instanceof StringType) {
            return vector.getUTF8String(0).copy();
        }
        if (type instanceof DecimalType decimal) {
            return vector.getDecimal(0, decimal.precision(), decimal.scale());
        }
        throw new UnsupportedOperationException("unsupported aggregate result type: " + type);
    }
}
