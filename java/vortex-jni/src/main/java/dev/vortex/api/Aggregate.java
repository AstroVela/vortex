// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.Locale;
import java.util.Objects;

/**
 * Specification of one aggregate to evaluate natively over a {@link DataSource}, via
 * {@link DataSource#aggregate(java.util.List, Expression, org.apache.arrow.memory.BufferAllocator)}.
 *
 * <p>Null handling follows SQL semantics: {@code MIN}/{@code MAX}/{@code SUM} ignore null values and return null when
 * every value is null (or the source is empty); {@code COUNT} counts non-null values and {@code COUNT_STAR} counts
 * rows, both returning a non-null 64-bit integer. Floating point NaN values follow Spark/Java ordering and arithmetic:
 * {@code MIN}/{@code MAX} order NaN above every other value, {@code SUM} propagates NaN, and {@code COUNT} counts NaN
 * as an ordinary non-null value.
 */
public final class Aggregate {
    /** The kind of aggregate function. Codes must match the native side. */
    public enum Kind {
        /** Minimum non-null value of a column. */
        MIN((byte) 0),
        /** Maximum non-null value of a column. */
        MAX((byte) 1),
        /** Sum of the non-null values of a column. */
        SUM((byte) 2),
        /** Count of non-null values of a column. */
        COUNT((byte) 3),
        /** Count of rows. */
        COUNT_STAR((byte) 4);

        private final byte code;

        Kind(byte code) {
            this.code = code;
        }

        /** Wire code passed over the JNI boundary. */
        public byte code() {
            return code;
        }
    }

    private final Kind kind;
    private final String column;

    private Aggregate(Kind kind, String column) {
        this.kind = kind;
        this.column = column;
    }

    /** Minimum of {@code column}. */
    public static Aggregate min(String column) {
        return new Aggregate(Kind.MIN, Objects.requireNonNull(column, "column"));
    }

    /** Maximum of {@code column}. */
    public static Aggregate max(String column) {
        return new Aggregate(Kind.MAX, Objects.requireNonNull(column, "column"));
    }

    /** Sum of {@code column}. */
    public static Aggregate sum(String column) {
        return new Aggregate(Kind.SUM, Objects.requireNonNull(column, "column"));
    }

    /** Count of non-null values of {@code column}. */
    public static Aggregate count(String column) {
        return new Aggregate(Kind.COUNT, Objects.requireNonNull(column, "column"));
    }

    /** Count of rows. */
    public static Aggregate countStar() {
        return new Aggregate(Kind.COUNT_STAR, null);
    }

    /** The aggregate function kind. */
    public Kind kind() {
        return kind;
    }

    /** The aggregated column name, or {@code null} for {@link Kind#COUNT_STAR}. */
    public String column() {
        return column;
    }

    @Override
    public String toString() {
        return kind == Kind.COUNT_STAR ? "count(*)" : kind.name().toLowerCase(Locale.ROOT) + "(" + column + ")";
    }
}
