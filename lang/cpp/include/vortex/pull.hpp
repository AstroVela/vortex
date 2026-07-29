// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "vortex/array.hpp"
#include "vortex/common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/expression.hpp"
#include "vortex/scan.hpp"
#include "vortex/session.hpp"

#include <vortex.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

/**
 * Pull-based (inverted-IO) scanning: the caller performs all reads.
 *
 * Unlike DataSource/Scan, nothing here touches the built-in Vortex runtime or
 * performs IO. A coroutine (PullFooter for the open path, PullScan for data)
 * tells the caller which byte ranges it needs and hands out pre-allocated,
 * correctly aligned destination buffers; the caller fills them with its own IO
 * machinery (an engine file system, io_uring, plain pread) and hands them
 * back. Decoding happens inside advance() on the calling thread.
 *
 * These objects are not thread-safe: drive each from one thread. To scan one
 * file on many threads, create one PullScan per disjoint row range aligned to
 * Footer::split_points(), so no two scans read the same segment.
 */
namespace vortex {

/**
 * A single byte-range read the caller must perform.
 *
 * Fill data() with the len() bytes at file offset offset(), then return the
 * read to the coroutine's complete(). The buffer behind data() is owned by
 * the coroutine that issued the read and stays valid until the read is
 * completed or the coroutine is destroyed.
 */
class PullRead {
public:
    uint64_t offset() const {
        return raw_.offset;
    }

    // Destination buffer to fill; also the identity of this read.
    std::span<uint8_t> data() const {
        return {raw_.dst, static_cast<size_t>(raw_.len)};
    }

private:
    friend struct detail::Access;
    friend class PullFooter;
    friend class PullScan;
    explicit PullRead(vx_pull_read raw) : raw_(raw) {
    }

    vx_pull_read raw_;
};

/**
 * A parsed Vortex file footer: segment map, layout tree, dtype and statistics.
 *
 * Obtained from PullFooter. Calling methods of a moved-out Footer is UB.
 */
class Footer {
public:
    Footer(const Footer &) = delete;
    Footer &operator=(const Footer &) = delete;
    Footer(Footer &&) noexcept = default;
    Footer &operator=(Footer &&) noexcept = default;

    DataType dtype() const;
    uint64_t row_count() const;

    /**
     * Chunk-aligned row split points for sharding this file's scan across
     * threads.
     *
     * PullScans over disjoint row ranges aligned to these points never share
     * data segments for the fields referenced by the projection and filter,
     * so they can be driven independently without reading any segment twice.
     * Performs no IO. Pass the same projection/filter the scans will use.
     */
    std::vector<uint64_t> split_points(const Session &session,
                                       const std::optional<Expression> &projection = std::nullopt,
                                       const std::optional<Expression> &filter = std::nullopt) const;

private:
    friend struct detail::Access;
    explicit Footer(vx_footer *owned) : handle_(owned) {
    }

    struct Deleter {
        void operator()(vx_footer *ptr) const noexcept;
    };
    std::unique_ptr<vx_footer, Deleter> handle_;
};

/**
 * A pull coroutine that parses a file footer; the caller performs the reads.
 *
 * Footer reads are sequential, so exactly one read is outstanding at a time:
 *
 * PullFooter pf(session, file_size);
 * while (auto read = pf.next_read()) {
 *     my_read_at(fd, read->data(), read->offset());
 *     pf.complete(*read);
 * }
 * Footer footer = std::move(pf).take();
 */
class PullFooter {
public:
    PullFooter(const Session &session, uint64_t file_size);

    PullFooter(const PullFooter &) = delete;
    PullFooter &operator=(const PullFooter &) = delete;
    PullFooter(PullFooter &&) noexcept = default;
    PullFooter &operator=(PullFooter &&) noexcept = default;

    // The next read to perform, or nullopt once the footer is parsed.
    std::optional<PullRead> next_read();

    // Hand back the filled read issued by the previous next_read().
    void complete(const PullRead &read);

    // The parsed footer. Call after next_read() returned nullopt.
    Footer take() &&;

private:
    struct Deleter {
        void operator()(vx_pull_footer *ptr) const noexcept;
    };
    std::unique_ptr<vx_pull_footer, Deleter> handle_;
    vx_footer *footer_ = nullptr;
};

/**
 * A pull-based scan of a single Vortex file; the caller performs all reads.
 *
 * advance() returns either reads to perform (complete them in any order — many
 * may be in flight at once), a decoded batch, or nullopt when the scan is
 * exhausted:
 *
 * PullScan scan(session, footer, {.filter = expr});
 * while (auto event = scan.advance()) {
 *     if (auto *reads = std::get_if<PullScan::Reads>(&*event)) {
 *         for (auto &read : *reads) {
 *             my_read_at(fd, read.data(), read.offset());
 *             scan.complete(read);
 *         }
 *     } else {
 *         consume(std::get<Array>(std::move(*event)));
 *     }
 * }
 *
 * An empty Reads vector means the in-flight window is full: complete an
 * outstanding read before calling advance() again.
 */
class PullScan {
public:
    using Reads = std::vector<PullRead>;
    using Event = std::variant<Reads, Array>;

    /**
     * Per-file scanning context: builds the reader tree once so many PullScans
     * (e.g. one per chunk-aligned shard) reuse it. Scans created from one File
     * must be driven by one thread, one at a time.
     */
    class File {
    public:
        File(const Session &session, const Footer &footer);
        File(const File &) = delete;
        File &operator=(const File &) = delete;
        File(File &&) noexcept = default;
        File &operator=(File &&) noexcept = default;

        PullScan scan(const ScanOptions &options = {}, uint64_t max_inflight = 0) const;

    private:
        struct Deleter {
            void operator()(vx_pull_file *ptr) const noexcept;
        };
        std::unique_ptr<vx_pull_file, Deleter> handle_;
    };

    /**
     * Create a pull scan of the file described by "footer".
     *
     * options.row_range selects the shard of the file this scan decodes (see
     * Footer::split_points). "max_inflight" bounds how many reads may be
     * outstanding, which also bounds destination-buffer memory; 0 means no
     * bound.
     */
    PullScan(const Session &session, const Footer &footer, const ScanOptions &options = {},
             uint64_t max_inflight = 0);

    PullScan(const PullScan &) = delete;
    PullScan &operator=(const PullScan &) = delete;
    PullScan(PullScan &&) noexcept = default;
    PullScan &operator=(PullScan &&) noexcept = default;

    // One coroutine step: reads to perform, a decoded batch, or nullopt when
    // exhausted. Decoding happens inside this call, on the calling thread.
    std::optional<Event> advance();

    // Hand back a filled read. Completions may arrive in any order.
    void complete(const PullRead &read);

private:
    friend class File;
    explicit PullScan(vx_pull_scan *owned) : handle_(owned) {
    }

    struct Deleter {
        void operator()(vx_pull_scan *ptr) const noexcept;
    };
    std::unique_ptr<vx_pull_scan, Deleter> handle_;
};

} // namespace vortex
