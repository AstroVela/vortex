// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.Iterables;
import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexSparkSession;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

/**
 * Per-{@link VortexFilePartition} columnar reader.
 *
 * <p>Opens a single Vortex {@link Session}, {@link DataSource} and {@link Scan} spanning all of
 * {@link VortexFilePartition#paths()} and streams every Vortex partition's record batches through the
 * {@link PartitionReader} interface.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition spark;
    private final BufferAllocator allocator;
    private final Set<String> metadataColumnNames;

    // Held so the DataSource/Scan stay reachable even if the JVM-wide singleton is
    // ever reset during a task; the actual native session is owned by
    // {@link VortexSparkSession} and is not released when this reader closes.
    private Session session;
    private DataSource dataSource;
    private Scan scan;

    private Partition currentPartition;
    private ArrowReader currentReader;
    private boolean currentBatchLoaded;
    private boolean exhausted;

    VortexPartitionReader(
            VortexFilePartition spark,
            List<String> dataColumnNames,
            Map<String, String> formatOptions,
            Predicate[] pushedPredicates,
            Set<String> metadataColumnNames,
            VortexTableSample tableSample,
            int limit) {
        this.spark = spark;
        this.allocator = ArrowAllocation.rootAllocator();
        this.metadataColumnNames = metadataColumnNames;

        session = VortexSparkSession.get(formatOptions);
        dataSource = DataSource.open(session, spark.paths(), formatOptions);

        var options = ScanOptions.builder();
        buildProjection(dataColumnNames).ifPresent(options::projection);
        if (pushedPredicates != null && pushedPredicates.length > 0) {
            buildFilterExpression(pushedPredicates).ifPresent(options::filter);
        }
        if (tableSample != null) {
            options.selectionRoaringBitmap(sampleSelection(tableSample));
            options.selectionMode(ScanOptions.SelectionMode.INCLUDE_ROARING);
        }
        if (limit >= 0) {
            options.limit(limit);
        }
        scan = dataSource.scan(options.build());
    }

    /**
     * Builds the scan's projection expression. Without metadata columns this is a plain field selection. When the
     * {@code _pos} metadata column is requested, the projection instead packs the data columns and the row-index
     * expression into a struct, in the order the fields appear in the requested read schema, so that the Arrow output
     * lines up with the batch assembly in {@link #get()}. The {@code _file} column needs nothing from the scan; it is
     * materialized as a per-file constant.
     */
    private Optional<Expression> buildProjection(List<String> dataColumnNames) {
        if (!metadataColumnNames.contains(VortexMetadataColumns.ROW_POSITION)) {
            if (dataColumnNames.isEmpty()) {
                return Optional.empty();
            }
            return Optional.of(Expression.select(dataColumnNames.toArray(new String[0]), Expression.root()));
        }

        List<String> fieldNames = new ArrayList<>();
        List<Expression> fields = new ArrayList<>();
        for (StructField field : spark.readSchema().fields()) {
            String name = field.name();
            if (VortexMetadataColumns.ROW_POSITION.equals(name) && metadataColumnNames.contains(name)) {
                fieldNames.add(name);
                fields.add(Expression.rowIdx());
            } else if (dataColumnNames.contains(name)) {
                fieldNames.add(name);
                fields.add(Expression.column(name));
            }
        }
        return Optional.of(Expression.pack(
                fieldNames.toArray(new String[0]), fields.toArray(new Expression[0]), /* nullable= */ false));
    }

    /**
     * Draws the sampled row positions for this partition's files as a serialized Roaring bitmap. The pseudo-random
     * sequence is seeded from the sampling seed and the file paths, so the same seed always selects the same rows.
     * Sampling is applied to the file's original row positions, before any pushed filter, matching SQL TABLESAMPLE
     * semantics.
     */
    private byte[] sampleSelection(VortexTableSample tableSample) {
        if (!(dataSource.rowCount() instanceof DataSource.RowCount.Exact exact)) {
            throw new IllegalStateException(
                    "cannot sample " + spark.paths() + ": the exact file row count is not available");
        }
        Random random =
                new Random(tableSample.seed() ^ String.join(",", spark.paths()).hashCode());
        Roaring64NavigableMap selection = new Roaring64NavigableMap();
        for (long row = 0; row < exact.value(); row++) {
            double draw = random.nextDouble();
            if (draw >= tableSample.lowerBound() && draw < tableSample.upperBound()) {
                selection.addLong(row);
            }
        }
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                DataOutputStream output = new DataOutputStream(bytes)) {
            selection.serializePortable(output);
            output.flush();
            return bytes.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Optional<Expression> buildFilterExpression(Predicate[] predicates) {
        Expression combined = null;
        for (Predicate predicate : predicates) {
            Optional<Expression> expr = SparkPredicateToVortexExpression.convert(predicate);
            if (expr.isEmpty()) {
                continue;
            }
            combined = combined == null ? expr.get() : Expression.and(combined, expr.get());
        }
        return Optional.ofNullable(combined);
    }

    @Override
    public boolean next() {
        if (exhausted) {
            return false;
        }
        while (true) {
            if (currentReader != null) {
                try {
                    if (currentReader.loadNextBatch()) {
                        currentBatchLoaded = true;
                        return true;
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                closeCurrentReader();
            }
            if (!scan.hasNext()) {
                exhausted = true;
                return false;
            }
            currentPartition = scan.next();
            currentReader = currentPartition.scanArrow(allocator);
        }
    }

    @Override
    public ColumnarBatch get() {
        if (!currentBatchLoaded) {
            throw new IllegalStateException("no batch loaded; call next() first");
        }
        currentBatchLoaded = false;

        VectorSchemaRoot root;
        try {
            root = currentReader.getVectorSchemaRoot();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        int rowCount = root.getRowCount();
        Map<String, String> partVals = spark.partitionValues();
        if (partVals.isEmpty() && metadataColumnNames.isEmpty()) {
            ColumnVector[] vectors = new ColumnVector[root.getFieldVectors().size()];
            for (int i = 0; i < vectors.length; i++) {
                vectors[i] = new VortexArrowColumnVector(root.getFieldVectors().get(i));
            }
            return new ColumnarBatch(vectors, rowCount);
        }

        StructField[] fields = spark.readSchema().fields();
        ColumnVector[] combined = new ColumnVector[fields.length];
        int dataIdx = 0;
        for (int i = 0; i < fields.length; i++) {
            StructField field = fields[i];
            String name = field.name();
            if (VortexMetadataColumns.FILE_PATH.equals(name) && metadataColumnNames.contains(name)) {
                combined[i] = PartitionPathUtils.createConstantVector(
                        rowCount, DataTypes.StringType, Iterables.getOnlyElement(spark.paths()));
                continue;
            }
            String partValue = partVals.get(name);
            if (partValue != null && !metadataColumnNames.contains(name)) {
                combined[i] = PartitionPathUtils.createConstantVector(rowCount, field.dataType(), partValue);
            } else {
                // Data columns and the _pos metadata column both come from the Arrow output, in
                // read-schema order (see buildProjection).
                combined[i] = new VortexArrowColumnVector(root.getFieldVectors().get(dataIdx++));
            }
        }
        return new ColumnarBatch(combined, rowCount);
    }

    @Override
    public void close() {
        closeCurrentReader();
        // Scan and DataSource native resources are released by VortexCleaner once
        // references are dropped. Session is the JVM-wide singleton and outlives this reader.
        scan = null;
        dataSource = null;
        session = null;
    }

    private void closeCurrentReader() {
        if (currentReader != null) {
            try {
                currentReader.close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            currentReader = null;
        }
        currentPartition = null;
    }
}
