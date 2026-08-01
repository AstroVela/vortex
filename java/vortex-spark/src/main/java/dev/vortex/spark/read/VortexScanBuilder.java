// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.expressions.aggregate.AggregateFunc;
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation;
import org.apache.spark.sql.connector.expressions.aggregate.Count;
import org.apache.spark.sql.connector.expressions.aggregate.CountStar;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsPushDownAggregates;
import org.apache.spark.sql.connector.read.SupportsPushDownLimit;
import org.apache.spark.sql.connector.read.SupportsPushDownRequiredColumns;
import org.apache.spark.sql.connector.read.SupportsPushDownV2Filters;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;

/** Spark V2 {@link ScanBuilder} for table scans over Vortex files. */
public final class VortexScanBuilder
        implements ScanBuilder,
                SupportsPushDownRequiredColumns,
                SupportsPushDownV2Filters,
                SupportsPushDownLimit,
                SupportsPushDownAggregates {
    private final ImmutableList.Builder<String> paths;
    private final List<Column> tableColumns;
    private final List<Column> readColumns;
    private final Map<String, String> formatOptions;
    private final Set<String> partitionColumnNames;
    private Predicate[] pushedPredicates = new Predicate[0];
    private int limit = VortexScan.NO_LIMIT;
    private int pushedCountColumns = 0;

    /** Creates a new VortexScanBuilder with empty paths and columns. */
    public VortexScanBuilder(Map<String, String> formatOptions) {
        this(formatOptions, new Transform[0]);
    }

    /**
     * Creates a new VortexScanBuilder with empty paths and columns and the supplied partition transforms. Filters that
     * reference partition columns are not pushed down, since the partition columns are not stored inside the Vortex
     * files.
     */
    public VortexScanBuilder(Map<String, String> formatOptions, Transform[] partitionTransforms) {
        this.paths = ImmutableList.builder();
        Map<String, String> options = Maps.newHashMap();
        options.put("vortex.workerThreads", "4");
        options.putAll(formatOptions);
        this.tableColumns = new ArrayList<>();
        this.readColumns = new ArrayList<>();
        this.formatOptions = options;
        this.partitionColumnNames = collectPartitionColumnNames(partitionTransforms);
    }

    /**
     * Adds a file path to scan.
     *
     * @param path the file path to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addPath(String path) {
        this.paths.add(path);
        return this;
    }

    /**
     * Adds a column to read.
     *
     * @param column the column to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addColumn(Column column) {
        this.tableColumns.add(column);
        this.readColumns.add(column);
        return this;
    }

    /**
     * Adds multiple file paths to scan.
     *
     * @param paths the iterable of file paths to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addAllPaths(Iterable<String> paths) {
        this.paths.addAll(paths);
        return this;
    }

    /**
     * Adds multiple columns to read.
     *
     * @param columns the iterable of columns to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addAllColumns(Iterable<Column> columns) {
        for (Column column : columns) {
            addColumn(column);
        }
        return this;
    }

    /**
     * Builds a VortexScan with the configured paths and columns.
     *
     * @return a new VortexScan instance
     * @throws IllegalStateException if no paths or columns have been added
     */
    @Override
    public Scan build() {
        var paths = this.paths.build();

        checkState(!paths.isEmpty(), "paths cannot be empty");
        // Allow empty columns for operations like count() that don't need actual column data
        // If no columns are specified, we'll read the minimal schema needed

        if (pushedCountColumns > 0) {
            return new VortexCountScan(paths, this.formatOptions, pushedCountColumns);
        }

        return new VortexScan(
                paths,
                List.copyOf(this.tableColumns),
                List.copyOf(this.readColumns),
                pushedPredicates,
                this.formatOptions,
                partitionColumnNames,
                limit);
    }

    /**
     * Accepts a LIMIT to push into the scan. The limit is applied by each partition reader after any pushed filter, so
     * every Spark input partition returns at most {@code limit} rows.
     *
     * @return always {@code true}; the pushdown is reported as partial via {@link #isPartiallyPushed()}
     */
    @Override
    public boolean pushLimit(int limit) {
        this.limit = limit;
        return true;
    }

    /**
     * Reports the LIMIT pushdown as partial: a scan may span several files, each of which enforces the limit
     * independently, so Spark must still apply the global limit on top.
     */
    @Override
    public boolean isPartiallyPushed() {
        return true;
>>>>>>> claude/spark-datasource-extensions-pmk8mc-limit-pushdown
    }

    /**
     * Accepts aggregations that count rows: any combination of {@code COUNT(*)} and {@code COUNT(col)} over
     * non-nullable data columns, without grouping. Every accepted aggregate evaluates to the file row count recorded in
     * the Vortex footer, so the scan can answer from metadata without decoding any data. The pushdown is partial (see
     * {@link SupportsPushDownAggregates#supportCompletePushDown}): the scan emits one count per file and Spark sums
     * them.
     *
     * <p>Rejected cases fall back to a regular scan: grouped aggregations, distinct counts, counts of nullable or
     * partition columns (their null counts are not recorded), other aggregate functions, and scans that already pushed
     * predicates (the footer count does not reflect filtering).
     */
    @Override
    public boolean pushAggregation(Aggregation aggregation) {
        if (pushedPredicates.length > 0) {
            return false;
        }
        if (aggregation.groupByExpressions().length > 0) {
            return false;
        }
        AggregateFunc[] funcs = aggregation.aggregateExpressions();
        if (funcs.length == 0) {
            return false;
        }
        for (AggregateFunc func : funcs) {
            if (func instanceof CountStar) {
                continue;
            }
            if (func instanceof Count count && !count.isDistinct() && isNonNullableDataColumn(count.column())) {
                continue;
            }
            return false;
        }
        this.pushedCountColumns = funcs.length;
        return true;
    }

    private boolean isNonNullableDataColumn(org.apache.spark.sql.connector.expressions.Expression expression) {
        if (!(expression instanceof NamedReference ref) || ref.fieldNames().length != 1) {
            return false;
        }
        String name = ref.fieldNames()[0];
        if (partitionColumnNames.contains(name)) {
            // Partition values may be null (__HIVE_DEFAULT_PARTITION__), and null counts for them
            // are not recorded anywhere, so COUNT(partitionColumn) cannot be answered from metadata.
            return false;
        }
        for (Column column : tableColumns) {
            if (column.name().equals(name)) {
                return !column.nullable();
            }
        }
        return false;
    }

    /**
     * Prunes the columns to only include those specified in the required schema.
     *
     * <p>This method clears the current column list and replaces it with columns derived from the required schema.
     * Currently only supports top-level schema pruning - deeply nested schema pruning is not yet implemented.
     *
     * @param requiredSchema the schema specifying which columns are required
     */
    @Override
    public void pruneColumns(StructType requiredSchema) {
        readColumns.clear();
        readColumns.addAll(Arrays.asList(CatalogV2Util.structTypeToV2Columns(requiredSchema)));
    }

    /**
     * Splits the supplied predicates into pushed and not-pushed sets.
     *
     * <p>A predicate is pushed when it references only data columns (not partition columns) and uses operators and
     * literal types that {@link SparkPredicateToVortexExpression} can map to Vortex expressions. Predicates that
     * reference partition columns or use unsupported features are returned to Spark for post-scan evaluation.
     *
     * @return the predicates that Spark must still evaluate
     */
    @Override
    public Predicate[] pushPredicates(Predicate[] predicates) {
        Map<String, DataType> dataColumnTypes = new HashMap<>();
        for (Column column : readColumns) {
            if (!partitionColumnNames.contains(column.name())) {
                dataColumnTypes.put(column.name(), column.dataType());
            }
        }
        List<Predicate> pushed = new ArrayList<>();
        List<Predicate> postScan = new ArrayList<>();
        for (Predicate predicate : predicates) {
            if (SparkPredicateToVortexExpression.isPushable(predicate, dataColumnTypes)) {
                pushed.add(predicate);
            } else {
                postScan.add(predicate);
            }
        }
        this.pushedPredicates = pushed.toArray(new Predicate[0]);
        return postScan.toArray(new Predicate[0]);
    }

    /** Returns the predicates this scan promises to apply. */
    @Override
    public Predicate[] pushedPredicates() {
        return Arrays.copyOf(pushedPredicates, pushedPredicates.length);
    }

    private static Set<String> collectPartitionColumnNames(Transform[] transforms) {
        if (transforms == null || transforms.length == 0) {
            return Collections.emptySet();
        }
        Set<String> names = new HashSet<>();
        for (Transform transform : transforms) {
            for (NamedReference ref : transform.references()) {
                String[] parts = ref.fieldNames();
                if (parts.length == 1) {
                    names.add(parts[0]);
                }
            }
        }
        return names;
    }
}
