// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

/**
 * One aggregate function to push into a scan, over at most one top-level column.
 *
 * <p>Aggregates are combined into an {@link AggregatePlan}, which reports whether Vortex can compute all of them. See
 * {@link AggregatePlan#tryCreate} for the semantics each function guarantees.
 */
public final class Aggregate {
    private final Kind kind;
    private final String column;

    private Aggregate(Kind kind, String column) {
        this.kind = kind;
        this.column = column;
    }

    /** {@code count(*)}: the number of rows the scan returns, including rows that are null in every column. */
    public static Aggregate countStar() {
        return new Aggregate(Kind.COUNT_STAR, null);
    }

    /** {@code count(column)}: the number of non-null values. NaN counts as a value. */
    public static Aggregate count(String column) {
        return of(Kind.COUNT, column);
    }

    /** {@code min(column)}, ignoring nulls. Not pushable over floating-point columns. */
    public static Aggregate min(String column) {
        return of(Kind.MIN, column);
    }

    /** {@code max(column)}, ignoring nulls. NaN is ordered above every other value. */
    public static Aggregate max(String column) {
        return of(Kind.MAX, column);
    }

    /** {@code sum(column)}, ignoring nulls. Null when the column holds no values. */
    public static Aggregate sum(String column) {
        return of(Kind.SUM, column);
    }

    private static Aggregate of(Kind kind, String column) {
        Objects.requireNonNull(column, "column");
        Preconditions.checkArgument(!column.isEmpty(), "column must not be empty");
        return new Aggregate(kind, column);
    }

    /** The aggregate function. */
    public Kind kind() {
        return kind;
    }

    /** The aggregated column, empty for {@link Kind#COUNT_STAR}. */
    public Optional<String> column() {
        return Optional.ofNullable(column);
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof Aggregate)) {
            return false;
        }
        Aggregate that = (Aggregate) other;
        return kind == that.kind && Objects.equals(column, that.column);
    }

    @Override
    public int hashCode() {
        return Objects.hash(kind, column);
    }

    @Override
    public String toString() {
        return kind == Kind.COUNT_STAR ? "count(*)" : kind.name().toLowerCase(Locale.ROOT) + "(" + column + ")";
    }

    /** The pushable aggregate functions. The codes are shared with the native side. */
    public enum Kind {
        /** See {@link Aggregate#countStar()}. */
        COUNT_STAR(0),
        /** See {@link Aggregate#count(String)}. */
        COUNT(1),
        /** See {@link Aggregate#min(String)}. */
        MIN(2),
        /** See {@link Aggregate#max(String)}. */
        MAX(3),
        /** See {@link Aggregate#sum(String)}. */
        SUM(4);

        private final int code;

        Kind(int code) {
            this.code = code;
        }

        /** Wire code passed to the native side. */
        public int code() {
            return code;
        }
    }
}
