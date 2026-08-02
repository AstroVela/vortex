// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Aggregate;
import java.io.Serializable;
import java.util.Locale;
import org.apache.spark.sql.types.DataType;

/**
 * One aggregate accepted by {@link VortexScanBuilder#pushAggregation} for native evaluation.
 *
 * @param kind the aggregate function
 * @param column the aggregated data column, or {@code null} for {@link Aggregate.Kind#COUNT_STAR}
 * @param resultType the Spark type of the value the scan emits for this aggregate
 */
record PushedAggregate(Aggregate.Kind kind, String column, DataType resultType) implements Serializable {

    /** Whether the emitted value may be null. Counts are always non-null; min/max/sum are null on empty input. */
    boolean nullable() {
        return kind != Aggregate.Kind.COUNT && kind != Aggregate.Kind.COUNT_STAR;
    }

    /** Converts to the JNI aggregate specification. */
    Aggregate toApi() {
        return switch (kind) {
            case MIN -> Aggregate.min(column);
            case MAX -> Aggregate.max(column);
            case SUM -> Aggregate.sum(column);
            case COUNT -> Aggregate.count(column);
            case COUNT_STAR -> Aggregate.countStar();
        };
    }

    @Override
    public String toString() {
        return kind == Aggregate.Kind.COUNT_STAR
                ? "count(*)"
                : kind.name().toLowerCase(Locale.ROOT) + "(" + column + ")";
    }
}
