// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeAggregate;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * A group-by-free aggregation that Vortex can evaluate inside a scan.
 *
 * <p>Attach a plan to a scan through {@link ScanOptions#aggregates()}. Every partition of that scan then produces a
 * single row holding <em>that partition's</em> aggregates, with one column per requested {@link Aggregate}, in request
 * order and typed as {@link #arrowSchema(BufferAllocator)} describes. Callers that scan more than one partition must
 * combine the partial rows themselves: sum the counts and sums, take the minimum of the minima and the maximum of the
 * maxima.
 *
 * <p>A plan reads only the columns its aggregates need, so it cannot be combined with a projection. A
 * {@code count(*)}-only plan over an unfiltered scan is answered from file metadata without reading any column data.
 * {@link Partition#rowCount()} keeps reporting the rows a partition reads, not the single row it returns.
 *
 * <p>Native resources are released automatically via {@link VortexCleaner} when the plan becomes unreachable.
 */
public final class AggregatePlan {
    private final Session session;
    private final long pointer;
    private final List<Aggregate> aggregates;

    private AggregatePlan(Session session, long pointer, List<Aggregate> aggregates) {
        Preconditions.checkArgument(pointer != 0, "invalid aggregate plan pointer");
        this.session = session;
        this.pointer = pointer;
        this.aggregates = aggregates;
        VortexCleaner.register(this, () -> NativeAggregate.free(pointer));
    }

    /**
     * Plan {@code aggregates} against {@code source}, or return empty when at least one of them cannot be computed with
     * the semantics a SQL engine expects, in which case the caller must scan the rows and aggregate them itself.
     *
     * <p>An aggregate is not pushable when:
     *
     * <ul>
     *   <li>its column is not a top-level column of the source — for example a Spark partition column, whose values
     *       live in the file path rather than in the file;
     *   <li>it is {@code min} over a floating-point column, because Vortex cannot reproduce "smallest non-NaN value,
     *       unless every value is NaN" in a single pass;
     *   <li>it is {@code min} or {@code max} over a column whose type has no ordering, such as a struct, list, or map;
     *   <li>it is {@code sum} over a 64-bit integer column, whose partial sum can overflow;
     *   <li>it is {@code sum} over an unsigned or boolean column, which has no signed counterpart to widen into.
     * </ul>
     *
     * <p>Otherwise the aggregates are computed exactly, with these output types: counts are non-nullable {@code int64};
     * {@code min}/{@code max} keep the column's type, made nullable; {@code sum} widens to nullable {@code int64} for
     * integers, {@code float64} for floating-point columns, and by 10 digits of precision for decimals.
     *
     * @param source the data source the aggregates read
     * @param aggregates at least one aggregate, evaluated in the order given
     */
    public static Optional<AggregatePlan> tryCreate(DataSource source, List<Aggregate> aggregates) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(aggregates, "aggregates");
        Preconditions.checkArgument(!aggregates.isEmpty(), "at least one aggregate is required");

        int[] kinds = new int[aggregates.size()];
        String[] columns = new String[aggregates.size()];
        for (int i = 0; i < aggregates.size(); i++) {
            Aggregate aggregate = aggregates.get(i);
            Objects.requireNonNull(aggregate, "aggregates must not contain null values");
            kinds[i] = aggregate.kind().code();
            columns[i] = aggregate.column().orElse(null);
        }

        long pointer = NativeAggregate.plan(source.nativePointer(), kinds, columns);
        if (pointer == 0) {
            return Optional.empty();
        }
        return Optional.of(new AggregatePlan(source.session(), pointer, ImmutableList.copyOf(aggregates)));
    }

    /** The aggregates this plan computes, in output column order. */
    public List<Aggregate> aggregates() {
        return aggregates;
    }

    /** Arrow schema of the single row each partition of an aggregate scan produces. */
    public Schema arrowSchema(BufferAllocator allocator) {
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeAggregate.arrowSchema(session.nativePointer(), pointer, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        }
    }

    long nativePointer() {
        return pointer;
    }
}
