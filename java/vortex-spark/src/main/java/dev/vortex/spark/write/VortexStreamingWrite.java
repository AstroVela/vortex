// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.Session;
import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.write.DataWriter;
import org.apache.spark.sql.connector.write.PhysicalWriteInfo;
import org.apache.spark.sql.connector.write.WriterCommitMessage;
import org.apache.spark.sql.connector.write.streaming.StreamingDataWriterFactory;
import org.apache.spark.sql.connector.write.streaming.StreamingWrite;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Streaming (micro-batch) write of Vortex files.
 *
 * <p>Every epoch appends files named {@code part-<partition>-<task>-epoch-<epoch>.vortex}. Files become visible as each
 * task commits, so the sink provides at-least-once semantics: a failed and replayed epoch can leave duplicate rows,
 * like other non-transactional file sinks.
 *
 * <p>In truncate mode (streaming Complete output mode) each epoch commit deletes the files of earlier epochs, so the
 * output directory converges to the result of the latest epoch.
 */
final class VortexStreamingWrite implements StreamingWrite, Serializable {

    private static final Logger log = LoggerFactory.getLogger(VortexStreamingWrite.class);

    private final String outputPath;
    private final StructType schema;
    private final Map<String, String> options;
    private final boolean truncate;
    private final PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms;

    VortexStreamingWrite(
            String outputPath,
            StructType schema,
            Map<String, String> options,
            boolean truncate,
            PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms) {
        this.outputPath = outputPath;
        this.schema = schema;
        this.options = options;
        this.truncate = truncate;
        this.resolvedTransforms = resolvedTransforms.clone();
    }

    @Override
    public StreamingDataWriterFactory createStreamingWriterFactory(PhysicalWriteInfo info) {
        return new Factory(outputPath, schema, options, resolvedTransforms);
    }

    @Override
    public void commit(long epochId, WriterCommitMessage[] messages) {
        List<String> writtenFiles = extractFilePaths(messages);
        if (truncate) {
            replaceEarlierEpochs(epochId);
        }
        log.info("Committed epoch {} with {} Vortex file(s) under {}", epochId, writtenFiles.size(), outputPath);
    }

    @Override
    public void abort(long epochId, WriterCommitMessage[] messages) {
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

    /** Deletes every file that does not belong to the given epoch, for truncate (Complete) mode. */
    private void replaceEarlierEpochs(long epochId) {
        String keepSuffix = epochSuffix(epochId);
        Session session = VortexSparkSession.get(options);
        List<String> toDelete = new ArrayList<>();
        for (String existing : NativeFiles.listFiles(session, outputPath, options)) {
            if (!existing.endsWith(keepSuffix)) {
                toDelete.add(existing);
            }
        }
        if (!toDelete.isEmpty()) {
            log.info(
                    "Streaming truncate: deleting {} file(s) of earlier epochs under {} after committing epoch {}",
                    toDelete.size(),
                    outputPath,
                    epochId);
            NativeFiles.delete(session, toDelete.toArray(new String[0]), options);
        }
    }

    private static String epochSuffix(long epochId) {
        return String.format("-epoch-%d.vortex", epochId);
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

    private static final class Factory implements StreamingDataWriterFactory, Serializable {
        private static final long serialVersionUID = 1L;

        private final String outputUri;
        private final StructType schema;
        private final Map<String, String> options;
        private final PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms;

        Factory(
                String outputUri,
                StructType schema,
                Map<String, String> options,
                PartitionedVortexDataWriter.ResolvedTransform[] resolvedTransforms) {
            this.outputUri = outputUri;
            this.schema = schema;
            this.options = options;
            this.resolvedTransforms = resolvedTransforms;
        }

        @Override
        public DataWriter<InternalRow> createWriter(int partitionId, long taskId, long epochId) {
            CaseInsensitiveStringMap optionsMap = new CaseInsensitiveStringMap(options);
            String fileName = String.format("part-%05d-%d%s", partitionId, taskId, epochSuffix(epochId));

            if (resolvedTransforms.length > 0) {
                return new PartitionedVortexDataWriter(
                        outputUri, schema, optionsMap, resolvedTransforms, partitionId, taskId, fileName);
            }

            String base = outputUri.endsWith("/") ? outputUri : outputUri + "/";
            return new VortexDataWriter(base + fileName, schema, optionsMap);
        }
    }
}
