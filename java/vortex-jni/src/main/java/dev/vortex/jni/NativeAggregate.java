// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for {@link dev.vortex.api.AggregatePlan}. */
public final class NativeAggregate {
    static {
        NativeLoader.loadJni();
    }

    private NativeAggregate() {}

    /**
     * Plan an aggregate pushdown against a data source.
     *
     * @param dataSourcePointer pointer from {@link NativeDataSource#open}
     * @param kinds one {@link dev.vortex.api.Aggregate.Kind} code per aggregate
     * @param columns the aggregated column name per aggregate, parallel to {@code kinds}, with a null entry for
     *     {@code count(*)}
     * @return a plan pointer, or {@code 0} when the aggregation cannot be pushed down
     */
    public static native long plan(long dataSourcePointer, int[] kinds, String[] columns);

    /** Free a plan pointer. */
    public static native void free(long pointer);

    /**
     * Export the plan's output schema into the Arrow C Data Interface struct at {@code schemaAddress}: one field per
     * requested aggregate, in request order.
     *
     * @param sessionPointer pointer from {@link NativeSession#newSession()}
     */
    public static native void arrowSchema(long sessionPointer, long pointer, long schemaAddress);
}
