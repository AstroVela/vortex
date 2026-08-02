// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import dev.vortex.api.Aggregate;
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
import org.apache.spark.sql.connector.expressions.aggregate.Max;
import org.apache.spark.sql.connector.expressions.aggregate.Min;
import org.apache.spark.sql.connector.expressions.aggregate.Sum;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsPushDownAggregates;
import org.apache.spark.sql.connector.read.SupportsPushDownLimit;
import org.apache.spark.sql.connector.read.SupportsPushDownRequiredColumns;
import org.apache.spark.sql.connector.read.SupportsPushDownTableSample;
import org.apache.spark.sql.connector.read.SupportsPushDownV2Filters;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;

/** Spark V2 {@link ScanBuilder} for table scans over Vortex files. */
public final class VortexScanBuilder
        implements ScanBuilder,
                SupportsPushDownRequiredColumns,
                SupportsPushDownV2Filters,
                SupportsPushDownLimit,
                SupportsPushDownAggregates,
                SupportsPushDownTableSample {
    private final ImmutableList.Builder<String> paths;
    private final List<Column> tableColumns;
    private final List<Column> readColumns;
    private final Map<String, String> formatOptions;
    private final Set<String> partitionColumnNames;
    private final List<String> orderedPartitionColumnNames;
    private Predicate[] pushedPredicates = new Predicate[0];
    private int limit = VortexScan.NO_LIMIT;
    private int pushedCountColumns = 0;
    private VortexTableSample tableSample;
    private List<PushedAggregate> pushedAggregates;

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
        this.orderedPartitionColumnNames = orderedIdentityPartitionColumns(partitionTransforms);
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
        if (pushedAggregates != null) {
            return new VortexAggregateScan(paths, this.formatOptions, pushedAggregates, pushedPredicates);
        }

        return new VortexScan(
                paths,
                List.copyOf(this.tableColumns),
                List.copyOf(this.readColumns),
                pushedPredicates,
                this.formatOptions,
                partitionColumnNames,
                metadataColumnNames(),
                tableSample,
                limit,
                partitionSchema());
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
    }

    /**
     * Accepts ungrouped aggregations that Vortex can answer without decoding rows into Spark. The pushdown is partial
     * (see {@link SupportsPushDownAggregates#supportCompletePushDown}): the scan emits one row of per-file aggregate
     * values and Spark re-aggregates them.
     *
     * <p>Two strategies are used, in order of preference:
     *
     * <ul>
     *   <li><b>Footer counts:</b> when no predicates were pushed and every aggregate is {@code COUNT(*)} or
     *       {@code COUNT(col)} over a non-nullable data column, the counts are answered from the file footers via
     *       {@link VortexCountScan} without opening the data at all.
     *   <li><b>Native aggregation:</b> otherwise, {@code MIN}/{@code MAX}/{@code SUM}/{@code COUNT}/{@code COUNT(*)}
     *       over supported data column types are streamed through Vortex's native accumulators via
     *       {@link VortexAggregateScan}, applying any pushed predicates before aggregation.
     * </ul>
     *
     * <p>Rejected cases fall back to a regular scan: grouped aggregations, distinct aggregates, aggregates over
     * partition or nested columns, {@code MIN}/{@code MAX} over floating point columns (Spark orders NaN above every
     * value, Vortex does not), {@code SUM} over longs (Spark wraps on overflow, Vortex returns null), unsupported
     * column types, and scans that already pushed a TABLESAMPLE or LIMIT (neither strategy reflects them).
     */
    @Override
    public boolean pushAggregation(Aggregation aggregation) {
        if (tableSample != null || limit != VortexScan.NO_LIMIT) {
            // A pushed TABLESAMPLE or LIMIT changes the row set; neither the footer row counts
            // nor the native aggregate scan reflect it, so aggregates must fall back to a
            // regular scan.
            return false;
        }
        if (aggregation.groupByExpressions().length > 0) {
            return false;
        }
        AggregateFunc[] funcs = aggregation.aggregateExpressions();
        if (funcs.length == 0) {
            return false;
        }

        if (pushedPredicates.length == 0 && allCountableFromFooters(funcs)) {
            this.pushedCountColumns = funcs.length;
            return true;
        }

        List<PushedAggregate> translated = new ArrayList<>(funcs.length);
        for (AggregateFunc func : funcs) {
            PushedAggregate aggregate = translateAggregate(func);
            if (aggregate == null) {
                return false;
            }
            translated.add(aggregate);
        }
        this.pushedAggregates = List.copyOf(translated);
        return true;
    }

    /** Whether every aggregate is answerable from footer row counts alone. */
    private boolean allCountableFromFooters(AggregateFunc[] funcs) {
        for (AggregateFunc func : funcs) {
            if (func instanceof CountStar) {
                continue;
            }
            if (func instanceof Count count && !count.isDistinct() && isNonNullableDataColumn(count.column())) {
                continue;
            }
            return false;
        }
        return true;
    }

    /** Maps one Spark aggregate onto a native Vortex aggregate, or returns {@code null} if it cannot be pushed. */
    private PushedAggregate translateAggregate(AggregateFunc func) {
        if (func instanceof CountStar) {
            return new PushedAggregate(Aggregate.Kind.COUNT_STAR, null, DataTypes.LongType);
        }
        if (func instanceof Count count && !count.isDistinct()) {
            String column = dataColumnName(count.column());
            return column == null ? null : new PushedAggregate(Aggregate.Kind.COUNT, column, DataTypes.LongType);
        }
        if (func instanceof Min min) {
            String column = dataColumnName(min.column());
            DataType type = column == null ? null : columnType(column);
            return type == null || !isMinMaxSupported(type)
                    ? null
                    : new PushedAggregate(Aggregate.Kind.MIN, column, type);
        }
        if (func instanceof Max max) {
            String column = dataColumnName(max.column());
            DataType type = column == null ? null : columnType(column);
            return type == null || !isMinMaxSupported(type)
                    ? null
                    : new PushedAggregate(Aggregate.Kind.MAX, column, type);
        }
        if (func instanceof Sum sum && !sum.isDistinct()) {
            String column = dataColumnName(sum.column());
            DataType type = column == null ? null : columnType(column);
            DataType resultType = type == null ? null : sumResultType(type);
            return resultType == null ? null : new PushedAggregate(Aggregate.Kind.SUM, column, resultType);
        }
        return null;
    }

    /**
     * Resolves an aggregate input to the name of a top-level data column stored in the Vortex files, or {@code null}
     * when the reference is nested, unknown, or a partition column (partition values live in paths, not files).
     */
    private String dataColumnName(org.apache.spark.sql.connector.expressions.Expression expression) {
        if (!(expression instanceof NamedReference ref) || ref.fieldNames().length != 1) {
            return null;
        }
        String name = ref.fieldNames()[0];
        if (partitionColumnNames.contains(name)) {
            return null;
        }
        return columnType(name) == null ? null : name;
    }

    private DataType columnType(String name) {
        for (Column column : tableColumns) {
            if (column.name().equals(name)) {
                return column.dataType();
            }
        }
        return null;
    }

    /**
     * Types whose native min/max ordering matches Spark's. Floating point columns are excluded because Spark orders NaN
     * above every other value while Vortex does not define a total order over NaNs.
     */
    private static boolean isMinMaxSupported(DataType type) {
        return type instanceof BooleanType
                || type instanceof ByteType
                || type instanceof ShortType
                || type instanceof IntegerType
                || type instanceof LongType
                || type instanceof DateType
                || type instanceof TimestampType
                || type instanceof TimestampNTZType
                || type instanceof StringType
                || type instanceof DecimalType;
    }

    /**
     * The Spark result type of a pushed {@code SUM}, mirroring {@code Sum.dataType}, or {@code null} when the input
     * type cannot be pushed. {@code SUM(long)} is rejected because Spark wraps on overflow while Vortex saturates to
     * null; decimal sums that Spark would widen beyond precision 38 are rejected for the same overflow-semantics
     * reason.
     */
    private static DataType sumResultType(DataType type) {
        if (type instanceof ByteType || type instanceof ShortType || type instanceof IntegerType) {
            return DataTypes.LongType;
        }
        if (type instanceof FloatType || type instanceof DoubleType) {
            return DataTypes.DoubleType;
        }
        if (type instanceof DecimalType decimal && decimal.precision() + 10 <= DecimalType.MAX_PRECISION()) {
            return DataTypes.createDecimalType(decimal.precision() + 10, decimal.scale());
        }
        return null;
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
     * Names of the requested read columns that are served as metadata columns ({@code _file}, {@code _pos}) rather than
     * read from file data. A name only counts as a metadata column when the table schema does not contain it — a data
     * or partition column with the same name shadows the metadata column.
     */
    private Set<String> metadataColumnNames() {
        Set<String> tableColumnNames = new HashSet<>();
        for (Column column : tableColumns) {
            tableColumnNames.add(column.name());
        }
        Set<String> names = new HashSet<>();
        for (Column column : readColumns) {
            String name = column.name();
            if ((VortexMetadataColumns.FILE_PATH.equals(name) || VortexMetadataColumns.ROW_POSITION.equals(name))
                    && !tableColumnNames.contains(name)) {
                names.add(name);
            }
        }
        return names;
    }

    /**
     * Accepts a Bernoulli (without replacement) TABLESAMPLE. Each partition reader draws a deterministic pseudo-random
     * value per row, seeded by the sampling seed and the file path, and pushes the accepted row positions into the
     * native scan as a row selection, so skipped rows are never decoded. Sampling with replacement is rejected and left
     * for Spark to evaluate.
     */
    @Override
    public boolean pushTableSample(double lowerBound, double upperBound, boolean withReplacement, long seed) {
        if (withReplacement) {
            return false;
        }
        this.tableSample = new VortexTableSample(lowerBound, upperBound, seed);
        return true;
    }

    /**
     * Schema of the identity partition columns in table-partitioning order, resolved against the table columns. Used to
     * report a key-grouped partitioning; empty for unpartitioned tables.
     */
    private StructType partitionSchema() {
        StructType schema = new StructType();
        for (String name : orderedPartitionColumnNames) {
            for (Column column : tableColumns) {
                if (column.name().equals(name)) {
                    schema = schema.add(name, column.dataType(), column.nullable());
                    break;
                }
            }
        }
        return schema;
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

    private static List<String> orderedIdentityPartitionColumns(Transform[] transforms) {
        List<String> names = new ArrayList<>();
        if (transforms == null) {
            return names;
        }
        for (Transform transform : transforms) {
            if (!"identity".equals(transform.name())) {
                continue;
            }
            for (NamedReference ref : transform.references()) {
                String[] parts = ref.fieldNames();
                if (parts.length == 1 && !names.contains(parts[0])) {
                    names.add(parts[0]);
                }
            }
        }
        return names;
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
