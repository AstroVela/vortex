// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.spark.read.PartitionPredicateEvaluator;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.SupportsDynamicOverwrite;
import org.apache.spark.sql.connector.write.SupportsOverwriteV2;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriteBuilder;

/**
 * Builder for configuring Vortex write operations.
 *
 * <p>This class is responsible for creating BatchWrite instances that execute the actual write operations to create
 * Vortex files from Spark DataFrames.
 */
public final class VortexWriteBuilder implements WriteBuilder, SupportsOverwriteV2, SupportsDynamicOverwrite {

    private final String paths;
    private final LogicalWriteInfo writeInfo;
    private final Map<String, String> options;
    private final Transform[] partitionTransforms;
    private VortexBatchWrite.Mode mode = VortexBatchWrite.Mode.APPEND;
    private Predicate[] overwritePredicates = new Predicate[0];

    /**
     * Creates a new VortexWriteBuilder.
     *
     * @param paths root path for write
     * @param writeInfo logical information about the write operation
     * @param options additional write options
     * @param partitionTransforms partition transforms (may be empty)
     */
    public VortexWriteBuilder(
            String paths, LogicalWriteInfo writeInfo, Map<String, String> options, Transform[] partitionTransforms) {
        this.paths = paths;
        this.writeInfo = writeInfo;
        this.options = options;
        this.partitionTransforms = partitionTransforms;
    }

    /**
     * Builds a Write for executing the write operation.
     *
     * @return a new VortexBatchWrite configured with this builder's settings
     */
    @Override
    public Write build() {
        return new VortexBatchWrite(paths, writeInfo.schema(), options, mode, overwritePredicates, partitionTransforms);
    }

    /**
     * Configures the write operation to truncate existing data.
     *
     * <p>When truncate is enabled, existing Vortex files at the output path will be removed before writing new data.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder truncate() {
        this.mode = VortexBatchWrite.Mode.TRUNCATE;
        this.overwritePredicates = new Predicate[0];
        return this;
    }

    /**
     * Configures the write operation to dynamically overwrite partitions: at commit time, only the partition
     * directories that received new data have their previous files removed; untouched partitions are preserved.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder overwriteDynamicPartitions() {
        this.mode = VortexBatchWrite.Mode.DYNAMIC_OVERWRITE;
        this.overwritePredicates = new Predicate[0];
        return this;
    }

    /**
     * Reports whether the overwrite condition can be applied at file granularity: every predicate must be decidable
     * from Hive-style partition values alone ({@code =}, {@code IN}, {@code IS_NULL}, {@code IS_NOT_NULL} and boolean
     * combinations over partition columns). Row-level conditions on data columns are rejected, which makes Spark fail
     * the query instead of silently deleting too much or too little.
     */
    @Override
    public boolean canOverwrite(Predicate[] predicates) {
        Set<String> partitionColumns = partitionColumnNames();
        for (Predicate predicate : predicates) {
            if (!PartitionPredicateEvaluator.isPartitionPredicate(predicate, partitionColumns)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Configures the write to first delete the files of every partition whose values match the given predicates, then
     * write the new data. An always-true condition is equivalent to {@link #truncate()}.
     *
     * @return this builder for method chaining
     */
    @Override
    public WriteBuilder overwrite(Predicate[] predicates) {
        this.mode = VortexBatchWrite.Mode.OVERWRITE_BY_FILTER;
        this.overwritePredicates = predicates == null ? new Predicate[0] : predicates.clone();
        return this;
    }

    private Set<String> partitionColumnNames() {
        Set<String> names = new HashSet<>();
        for (Transform transform : partitionTransforms) {
            if ("identity".equals(transform.name())) {
                for (NamedReference ref : transform.references()) {
                    if (ref.fieldNames().length == 1) {
                        names.add(ref.fieldNames()[0]);
                    }
                }
            }
        }
        return names;
    }
}
