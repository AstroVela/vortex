// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/array.hpp"
#include "vortex/dtype.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <memory>
#include <string_view>
#include <thread>

namespace vortex {

// Summary of a written Vortex file
struct WriteSummary {
    uint64_t row_count;
    // File size in bytes
    uint64_t file_size;
};

/**
 * Write Arrays into a .vortex file.
 *
 * finish() writes the footer and finalizes the file.
 * Not calling finish() leaves file corrupted.
 *
 * Writer methods are thread-safe.
 */
class Writer {
public:
    /*
     * Open a writer for a file at "path". "path" is copied.
     * "dtype" is used to validate pushed arrays so they all have same schema.
     *
     * "concurrent_array_limit" is an upper limit on how many pushed arrays are
     * buffered before push() blocks. This caps RAM used for buffering.
     */
    static Writer open(const Session &session,
                       std::string_view path,
                       const DataType &dtype,
                       size_t concurrent_array_limit = std::thread::hardware_concurrency());

    Writer(const Writer &) = delete;
    Writer &operator=(const Writer &) = delete;
    Writer(Writer &&) noexcept = default;
    Writer &operator=(Writer &&) noexcept = default;

    /*
     * Append Array to output file.
     * Throws if "array"'s DataType doesn't match writer's DataType.
     */
    void push(std::span<const Array> arrays);
    void push(const Array &array);
    void push(std::initializer_list<Array> arrays);

    /*
     * Write footer and finalize the file.
     * Throws on failure. Writer is closed afterwards and further uses throws.
     */
    WriteSummary finish();

private:
    explicit Writer(vx_writer *writer) : handle_(writer) {
    }

    struct Deleter {
        void operator()(vx_writer *ptr) const noexcept;
    };
    std::unique_ptr<vx_writer, Deleter> handle_;
};
} // namespace vortex
