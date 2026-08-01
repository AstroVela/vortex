// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import com.google.common.collect.ImmutableList;
import dev.vortex.api.Session;
import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.VortexSparkSession;
import dev.vortex.spark.read.PartitionPathUtils;
import dev.vortex.spark.read.PartitionPredicateEvaluator;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.distributions.Distribution;
import org.apache.spark.sql.connector.distributions.Distributions;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.SortDirection;
import org.apache.spark.sql.connector.expressions.SortOrder;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.metric.CustomMetric;
import org.apache.spark.sql.connector.write.BatchWrite;
import org.apache.spark.sql.connector.write.DataWriterFactory;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.RequiresDistributionAndOrdering;
import org.apache.spark.sql.connector.write.Write;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.connector.write.streaming.StreamingWrite;
import org.apache.spark.sql.types.StructType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Manages the batch write operation for creating Vortex files.
 *
 * <p>This class coordinates the distributed write operation across Spark executors, handling the creation of data
 * writers and managing commits/aborts.
 *
 * <p>Partitioned writes request a clustered distribution and in-task ordering on the identity partition columns (see
 * {@link RequiresDistributionAndOrdering}), so all rows of one Hive partition arrive in a single task and each
 * partition directory receives one file per write instead of one file per (task, partition) pair.
 */
public final class VortexBatchWrite implements Write, BatchWrite, RequiresDistributionAndOrdering, Serializable {

    /** How the write treats data already present at the output path. */
    enum Mode {
        /** Keep existing files; only add new ones. */
        APPEND,
        /** Delete every existing file under the output path before writing. */
        TRUNCATE,
        /** At commit, replace previous files only in partition directories that received new data. */
        DYNAMIC_OVERWRITE,
        /** Delete the files of partitions matching the overwrite predicates before writing. */
        OVERWRITE_BY_FILTER
    }

    private static final Logger log = LoggerFactory.getLogger(VortexBatchWrite.class);
    private final String outputPath;
    private final StructType schema;
    private final Map<String, String> options;
    private final Mode mode;
    private final Predicate[] overwritePredicates;
    // Resolved eagerly so that Spark Transform objects (Scala case classes that are not
    // Java-serializable) never reach the DataWriterFactory serialization boundary.
    private final PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms;
    // Top-level identity partition column names, used to request a clustered write distribution.
    private final ImmutableList<String> identityPartitionColumns;

    /**
     * Creates a new VortexBatchWrite.
     *
     * @param outputPath the base path where Vortex files will be written
     * @param schema the schema of the data to write
     * @param options additional write options
     * @param mode how to treat data already present at the output path
     * @param overwritePredicates partition predicates selecting the files to replace (only for
     *     {@link Mode#OVERWRITE_BY_FILTER})
     * @param partitionTransforms partition transforms (may be empty)
     */
    VortexBatchWrite(
            String outputPath,
            StructType schema,
            Map<String, String> options,
            Mode mode,
            Predicate[] overwritePredicates,
            Transform[] partitionTransforms) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
        this.mode = mode;
        this.overwritePredicates = overwritePredicates == null ? new Predicate[0] : overwritePredicates.clone();
        this.resolvedTransforms = PartitionedVortexDataWriter.resolveTransforms(partitionTransforms, schema);
        this.identityPartitionColumns = identityPartitionColumns(partitionTransforms);
    }

    private static ImmutableList<String> identityPartitionColumns(Transform[] partitionTransforms) {
        ImmutableList.Builder<String> names = ImmutableList.builder();
        for (Transform transform : partitionTransforms) {
            if ("identity".equals(transform.name())) {
                for (NamedReference ref : transform.references()) {
                    if (ref.fieldNames().length == 1) {
                        names.add(ref.fieldNames()[0]);
                    }
                }
            }
        }
        return names.build();
    }

    /**
     * Requests that rows be clustered by the identity partition columns, so every Hive partition is written by exactly
     * one task. Non-identity transforms (years/months/days/hours/bucket) are not used for clustering because their
     * evaluation requires a Spark function catalog; unpartitioned writes leave the distribution unspecified.
     */
    @Override
    public Distribution requiredDistribution() {
        if (identityPartitionColumns.isEmpty()) {
            return Distributions.unspecified();
        }
        Expression[] clustering =
                identityPartitionColumns.stream().map(Expressions::column).toArray(Expression[]::new);
        return Distributions.clustered(clustering);
    }

    /**
     * Requests an in-task sort on the identity partition columns, so a task writing several partitions sees each
     * partition's rows contiguously and keeps only one file open at a time.
     */
    @Override
    public SortOrder[] requiredOrdering() {
        return identityPartitionColumns.stream()
                .map(name -> Expressions.sort(Expressions.column(name), SortDirection.ASCENDING))
                .toArray(SortOrder[]::new);
    }

    /**
     * Returns this object as a BatchWrite.
     *
     * <p>This method is required by the Write interface to support batch writes.
     *
     * @return this object
     */
    @Override
    public BatchWrite toBatch() {
        return this;
    }

    /**
     * Declares the custom metrics Vortex write tasks report: files written, partition directories written, rows
     * written, and Arrow bytes buffered. Spark sums the per-task values and shows them in the SQL UI.
     */
    @Override
    public CustomMetric[] supportedCustomMetrics() {
        return VortexWriteMetrics.supportedMetrics();
    }

    /**
     * Returns the streaming variant of this write. Epochs append files named after (partition, task, epoch); when the
     * write was configured to truncate (streaming Complete output mode), each epoch commit removes the files of earlier
     * epochs so the output always reflects the latest result.
     */
    @Override
    public StreamingWrite toStreaming() {
        return new VortexStreamingWrite(outputPath, schema, options, mode == Mode.TRUNCATE, resolvedTransforms);
    }

    /**
     * Creates a DataWriterFactory for producing data writers on executors.
     *
     * <p>This method is called once at the start of the write operation, making it the right place to handle overwrite
     * cleanup.
     *
     * @return a new VortexDataWriterFactory
     */
    @Override
    public DataWriterFactory createBatchWriterFactory(PhysicalWriteInfo info) {
        // Handle truncate and overwrite-by-filter cleanup BEFORE writing starts. Dynamic
        // overwrite defers all deletion to commit(), once the set of partitions that received
        // new data is known.
        if (mode == Mode.TRUNCATE || mode == Mode.OVERWRITE_BY_FILTER) {
            Session session = VortexSparkSession.get(options);
            List<String> uris = NativeFiles.listFiles(session, outputPath, options);
            if (mode == Mode.OVERWRITE_BY_FILTER) {
                // Only replace files whose partition values definitively match every predicate.
                // canOverwrite() already restricted the predicates to decidable partition
                // predicates, so nothing that should be replaced can be missed here.
                uris = uris.stream()
                        .filter(uri -> PartitionPredicateEvaluator.definitelyMatches(
                                PartitionPathUtils.parsePartitionValues(uri), overwritePredicates))
                        .collect(Collectors.toList());
            }
            // Deleting the existing files is destructive and happens before the new data is written:
            // if the subsequent write fails, abort() only removes the newly written files and cannot
            // restore what was deleted here. Log loudly so operators can see what was removed.
            log.warn(
                    "Deleting {} existing file(s) under {} because of overwrite, before writing new data; "
                            + "this cannot be undone if the subsequent write fails",
                    uris.size(),
                    outputPath);
            NativeFiles.delete(session, uris.toArray(new String[0]), options);
        }

        return new VortexDataWriterFactory(outputPath, schema, options, resolvedTransforms);
    }

    /**
     * Called when a single data writer task completes successfully.
     *
     * <p>This is called for each successful task but individual file commits are handled in the data writer itself.
     *
     * @param message commit message from a successful data writer task
     */
    @Override
    public void onDataWriterCommit(WriterCommitMessage message) {
        // Individual file commits are handled in the data writer
        // This is called for each successful task
        log.debug("Committing DataWriter");
    }

    /**
     * Commits the entire write job after all tasks complete successfully.
     *
     * <p>This finalizes the write operation and ensures all Vortex files are properly written. For dynamic partition
     * overwrite, the previous files of every partition directory that received new data are deleted here, once the new
     * files are fully written; partitions that received no data are left untouched.
     *
     * @param messages commit messages from all successful write tasks
     */
    @Override
    public void commit(WriterCommitMessage[] messages) {
        List<String> writtenFiles = extractFilePaths(messages);

        if (mode == Mode.DYNAMIC_OVERWRITE) {
            replaceDynamicPartitions(writtenFiles);
        }

        if (!writtenFiles.isEmpty()) {
            log.info("Successfully wrote {} Vortex files to {}", writtenFiles.size(), outputPath);
        }
    }

    /**
     * Deletes the pre-existing files of every directory that received new files. Files are compared by name within
     * their directory, so URI scheme differences between the writer paths and the listing cannot cause false
     * mismatches. New files are already durable at this point; a failure here can leave old files behind but never lose
     * new data.
     */
    private void replaceDynamicPartitions(List<String> writtenFiles) {
        Map<String, Set<String>> newFileNamesByDir = new HashMap<>();
        for (String file : writtenFiles) {
            int slash = file.lastIndexOf('/');
            newFileNamesByDir
                    .computeIfAbsent(file.substring(0, slash), dir -> new HashSet<>())
                    .add(file.substring(slash + 1));
        }

        Session session = VortexSparkSession.get(options);
        List<String> toDelete = new ArrayList<>();
        for (Map.Entry<String, Set<String>> entry : newFileNamesByDir.entrySet()) {
            for (String existing : NativeFiles.listFiles(session, entry.getKey(), options)) {
                String name = existing.substring(existing.lastIndexOf('/') + 1);
                if (!entry.getValue().contains(name)) {
                    toDelete.add(existing);
                }
            }
        }
        if (!toDelete.isEmpty()) {
            log.info(
                    "Dynamic partition overwrite: deleting {} replaced file(s) across {} partition director(ies) "
                            + "under {}",
                    toDelete.size(),
                    newFileNamesByDir.size(),
                    outputPath);
            NativeFiles.delete(session, toDelete.toArray(new String[0]), options);
        }
    }

    /**
     * Aborts the write job due to failures.
     *
     * <p>This method cleans up any partially written files.
     *
     * @param messages commit messages from write tasks (may include failures)
     */
    @Override
    public void abort(WriterCommitMessage[] messages) {
        for (String filePath : extractFilePaths(messages)) {
            try {
                Path path = Paths.get(filePath);
                if (Files.exists(path)) {
                    Files.delete(path);
                }
            } catch (IOException e) {
                log.error("Failed to clean up file: {}", filePath, e);
            }
        }
    }

    private static List<String> extractFilePaths(WriterCommitMessage[] messages) {
        return Arrays.stream(messages)
                .flatMap(msg -> {
                    if (msg instanceof VortexWriterCommitMessage) {
                        return Stream.of(((VortexWriterCommitMessage) msg).filePath());
                    } else if (msg instanceof PartitionedVortexDataWriter.PartitionedWriterCommitMessage) {
                        return ((PartitionedVortexDataWriter.PartitionedWriterCommitMessage) msg)
                                .getPartitionMessages().stream().map(VortexWriterCommitMessage::filePath);
                    }
                    return Stream.empty();
                })
                .collect(Collectors.toList());
    }
}
