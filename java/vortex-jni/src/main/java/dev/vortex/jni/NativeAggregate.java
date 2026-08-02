// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for pushed-down aggregate evaluation over a {@link dev.vortex.api.DataSource}. */
public final class NativeAggregate {
    static {
        NativeLoader.loadJni();
    }

    private NativeAggregate() {}

    /**
     * Evaluate aggregates over the (optionally filtered) data source and write a single-row Arrow record batch — one
     * column per aggregate, in request order — to the {@code FFI_ArrowArrayStream} at {@code streamAddress}.
     *
     * @param sessionPointer native session pointer used for execution context
     * @param dataSourcePointer data source to aggregate; the pointer is borrowed, not consumed
     * @param aggKinds aggregate kind codes, one per aggregate; see {@link dev.vortex.api.Aggregate.Kind#code()}
     * @param aggColumns column name per aggregate; entries may be {@code null} for count(*)
     * @param filterPointer borrowed native expression pointer to filter rows, or {@code 0} for none
     * @param streamAddress address of an allocated {@code FFI_ArrowArrayStream} struct
     */
    public static native void compute(
            long sessionPointer,
            long dataSourcePointer,
            byte[] aggKinds,
            String[] aggColumns,
            long filterPointer,
            long streamAddress);
}
