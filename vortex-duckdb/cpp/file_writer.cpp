// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "error.hpp"
#include "vortex_duckdb.h"
#include "duckdb/common/file_system.hpp"
#include "duckdb/main/client_context.hpp"

using namespace duckdb;

namespace {
struct VortexFileWriter {
    explicit VortexFileWriter(unique_ptr<FileHandle> handle) : handle(std::move(handle)) {
    }

    unique_ptr<FileHandle> handle;
};

VortexFileWriter &GetVortexFileWriter(duckdb_vx_file_writer writer) {
    return *reinterpret_cast<VortexFileWriter *>(writer);
}

struct VortexFileWriterAbort {};

struct VortexFileWriterAbortGuard {
    explicit VortexFileWriterAbortGuard(VortexFileWriter *writer) : writer(writer) {
    }

    ~VortexFileWriterAbortGuard() {
        delete writer;
    }

    VortexFileWriter *writer;
};
} // namespace

extern "C" duckdb_vx_file_writer
duckdb_vx_file_writer_create(void *client_context, const char *file_path, duckdb_vx_error *error_out) {
    if (!client_context) {
        SetError(error_out, "Cannot open Vortex output without a DuckDB client context");
        return nullptr;
    }
    if (!file_path) {
        SetError(error_out, "Cannot open Vortex output with a null file path");
        return nullptr;
    }

    try {
        auto &context = *reinterpret_cast<ClientContext *>(client_context);
        auto &file_system = FileSystem::GetFileSystem(context);
        auto handle =
            file_system.OpenFile(file_path,
                                 FileFlags::FILE_FLAGS_WRITE | FileFlags::FILE_FLAGS_FILE_CREATE_NEW);
        auto writer = make_uniq<VortexFileWriter>(std::move(handle));
        *error_out = nullptr;
        return reinterpret_cast<duckdb_vx_file_writer>(writer.release());
    } catch (const std::exception &error) {
        SetError(error_out, error.what());
        return nullptr;
    }
}

extern "C" duckdb_state duckdb_vx_file_writer_write(duckdb_vx_file_writer ffi_writer,
                                                    const uint8_t *data,
                                                    size_t size,
                                                    duckdb_vx_error *error_out) {
    if (!ffi_writer) {
        return SetError(error_out, "Cannot write through a null Vortex file writer");
    }
    if (!data && size != 0) {
        return SetError(error_out, "Cannot write non-empty Vortex output from a null buffer");
    }

    try {
        auto &writer = GetVortexFileWriter(ffi_writer);
        if (!writer.handle) {
            return SetError(error_out, "Cannot write through a closed Vortex file writer");
        }
        if (size != 0) {
            const auto written = writer.handle->Write(const_cast<uint8_t *>(data), size);
            if (written != static_cast<int64_t>(size)) {
                throw IOException("DuckDB wrote only %lld of %llu Vortex output bytes",
                                  static_cast<long long>(written),
                                  static_cast<unsigned long long>(size));
            }
        }
        *error_out = nullptr;
        return DuckDBSuccess;
    } catch (const std::exception &error) {
        return SetError(error_out, error.what());
    }
}

extern "C" duckdb_state duckdb_vx_file_writer_flush(duckdb_vx_file_writer ffi_writer,
                                                    duckdb_vx_error *error_out) {
    if (!ffi_writer) {
        return SetError(error_out, "Cannot flush a null Vortex file writer");
    }

    try {
        auto &writer = GetVortexFileWriter(ffi_writer);
        if (!writer.handle) {
            return SetError(error_out, "Cannot flush a closed Vortex file writer");
        }
        // FileHandle writes are synchronous. Scheme-specific buffers, such as S3 multipart
        // buffers, must remain open until shutdown because VortexWrite permits more writes after
        // flush.
        *error_out = nullptr;
        return DuckDBSuccess;
    } catch (const std::exception &error) {
        return SetError(error_out, error.what());
    }
}

extern "C" duckdb_state duckdb_vx_file_writer_close(duckdb_vx_file_writer ffi_writer,
                                                    duckdb_vx_error *error_out) {
    if (!ffi_writer) {
        return SetError(error_out, "Cannot close a null Vortex file writer");
    }

    try {
        auto &writer = GetVortexFileWriter(ffi_writer);
        if (!writer.handle) {
            return SetError(error_out, "Cannot close an already closed Vortex file writer");
        }
        auto handle = std::move(writer.handle);
        handle->Close();
        *error_out = nullptr;
        return DuckDBSuccess;
    } catch (const std::exception &error) {
        return SetError(error_out, error.what());
    }
}

extern "C" void duckdb_vx_file_writer_destroy(duckdb_vx_file_writer ffi_writer) {
    delete reinterpret_cast<VortexFileWriter *>(ffi_writer);
}

extern "C" void duckdb_vx_file_writer_abort(duckdb_vx_file_writer ffi_writer) {
    if (!ffi_writer) {
        return;
    }

    // DuckDB's remote file handles detect exception unwinding and avoid completing an unfinished
    // multipart upload. Rust reports write failures as values, so destroy the handle while a local
    // exception is unwinding to preserve the same fail-closed behavior as a native DuckDB writer.
    try {
        VortexFileWriterAbortGuard guard(reinterpret_cast<VortexFileWriter *>(ffi_writer));
        throw VortexFileWriterAbort {};
    } catch (const VortexFileWriterAbort &) {
    }
}
