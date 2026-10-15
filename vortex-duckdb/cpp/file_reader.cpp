// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "error.hpp"
#include "vortex_duckdb.h"
#include "duckdb/common/file_system.hpp"
#include "duckdb/main/client_context.hpp"

#include <limits>

using namespace duckdb;

namespace {
struct VortexFileReader {
    VortexFileReader(ClientContext &context, unique_ptr<FileHandle> handle, uint64_t size)
        : database(context.db), context(context.shared_from_this()), handle(std::move(handle)), size(size) {
    }

    // Keep the underlying filesystem alive without a prepared-statement/context cycle.
    shared_ptr<DatabaseInstance> database;
    weak_ptr<ClientContext> context;
    unique_ptr<FileHandle> handle;
    uint64_t size;
};
} // namespace

extern "C" duckdb_vx_file_reader duckdb_vx_file_reader_create(void *client_context,
                                                              const char *file_path,
                                                              uint64_t *size_out,
                                                              duckdb_vx_error *error_out) {
    if (!client_context || !file_path || !size_out) {
        SetError(error_out, "Cannot open Vortex input without a context, path and size output");
        return nullptr;
    }
    try {
        auto &context = *reinterpret_cast<ClientContext *>(client_context);
        if (context.IsInterrupted()) {
            throw InterruptException();
        }
        auto &file_system = FileSystem::GetFileSystem(context);
        auto flags = FileFlags::FILE_FLAGS_READ;
        if (file_system.IsRemoteFile(file_path)) {
            // Vortex already coalesces ranges; avoid a second layer of speculative read-ahead.
            flags |= FileFlags::FILE_FLAGS_DIRECT_IO;
        }
        auto handle = file_system.OpenFile(file_path, flags);
        auto size = file_system.GetFileSize(*handle);
        if (size < 0 || !handle->CanSeek()) {
            throw IOException("Vortex input must be a seekable file with a known size");
        }
        auto reader = make_uniq<VortexFileReader>(context, std::move(handle), uint64_t(size));
        *size_out = reader->size;
        *error_out = nullptr;
        return reinterpret_cast<duckdb_vx_file_reader>(reader.release());
    } catch (const std::exception &error) {
        SetError(error_out, error.what());
        return nullptr;
    }
}

extern "C" duckdb_state duckdb_vx_file_reader_read(duckdb_vx_file_reader ffi_reader,
                                                   uint64_t offset,
                                                   uint8_t *data,
                                                   size_t size,
                                                   duckdb_vx_error *error_out) {
    if (!ffi_reader || (!data && size != 0)) {
        return SetError(error_out, "Cannot read Vortex input through a null reader or buffer");
    }
    try {
        auto &reader = *reinterpret_cast<VortexFileReader *>(ffi_reader);
        auto context = reader.context.lock();
        if (!context || context->IsInterrupted()) {
            throw InterruptException();
        }
        if (offset > reader.size || size > reader.size - offset ||
            size > uint64_t(std::numeric_limits<int64_t>::max())) {
            throw IOException("Vortex input range is outside the file");
        }
        reader.handle->Read(QueryContext(*context), data, size, offset);
        if (context->IsInterrupted()) {
            throw InterruptException();
        }
        *error_out = nullptr;
        return DuckDBSuccess;
    } catch (const std::exception &error) {
        return SetError(error_out, error.what());
    }
}

extern "C" void duckdb_vx_file_reader_destroy(duckdb_vx_file_reader reader) {
    delete reinterpret_cast<VortexFileReader *>(reader);
}
