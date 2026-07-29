// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// Pull-based scan: DuckDB threads read file bytes through the DuckDB FileSystem and drive
// Vortex pull coroutines (vortex/pull.hpp); Rust decodes inside advance() and exports batches
// into DataChunks. One file is sharded on chunk-aligned split points so many threads can scan
// it without ever reading the same segment twice.

#include "pull_table_function.h"

// vortex::cxx C++ API over the base Vortex C ABI (vortex-ffi's vortex.h). DuckDB headers above
// already define the Arrow C data interface structs, so tell vortex.h not to redefine them.
#define USE_OWN_ARROW
typedef struct ArrowSchema FFI_ArrowSchema;
typedef struct ArrowArray FFI_ArrowArray;
typedef struct ArrowArrayStream FFI_ArrowArrayStream;
#include <vortex/pull.hpp>

// vortex-duckdb's generated header (duckdb_pull_* helpers implemented in Rust). Included by
// explicit relative path: its name collides with vortex-ffi's <vortex.h> above.
#include "vortex_duckdb.h"
#include "../include/vortex.h"

#include "error.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>
#include <variant>
#include <vector>

namespace vortex_pull {

using duckdb::ClientContext;
using duckdb::DataChunk;
using duckdb::FileHandle;
using duckdb::FileSystem;
using duckdb::GlobalTableFunctionState;
using duckdb::LocalTableFunctionState;
using duckdb::DConstants;
using duckdb::TableFunctionInitInput;
using duckdb::TableFunctionInput;
using duckdb::idx_t;
using duckdb::make_uniq;
using duckdb::unique_ptr;
using duckdb::vector;

namespace {

// A file bigger than this is split into several shards so one huge file still scans on
// многих threads; smaller files scan as a single shard.
constexpr uint64_t TARGET_SHARD_BYTES = 16ULL << 20;

bool pull_enabled() {
    const char *env = std::getenv("VORTEX_DUCKDB_PULL");
    return env == nullptr || std::string_view(env) != "0";
}

void throw_ffi_error(duckdb_vx_error error) {
    throw duckdb::InvalidInputException(IntoErrString(error));
}

void throw_vx_error(vx_error *error) {
    const vx_view message = vx_error_message(error);
    std::string text(message.ptr, message.len);
    vx_error_free(error);
    throw duckdb::InvalidInputException(text);
}

struct PullFile {
    std::string path;
    void *cache; // duckdb_pull_cache_new

    // The footer, reader tree, and split points are shared by every scan of this file across
    // threads (the reader tree caches decoded dictionaries, statistics, and rewritten
    // expressions), and are built lazily by the first worker that touches the file.
    std::mutex init_mutex;
    std::shared_ptr<vortex::Footer> footer;
    std::shared_ptr<vortex::PullScan::File> ctx;
    std::shared_ptr<const std::vector<uint64_t>> points;
    // File-level statistics prove the filter matches nothing: skip the file entirely.
    bool pruned = false;
};

// Shard "part/parts" of a file: the worker maps it onto chunk-aligned row split points after
// it has read the footer, so global init performs no IO at all.
struct Shard {
    idx_t file_index;
    uint32_t part;
    uint32_t parts;
};

struct PullGlobalState final : GlobalTableFunctionState {
    explicit PullGlobalState(vortex::Session session) : session(std::move(session)) {
    }

    std::atomic<idx_t> batch_id {0};

    ~PullGlobalState() override {
        for (auto &file : files) {
            duckdb_pull_cache_free(file->cache);
        }
    }

    idx_t MaxThreads() const override {
        return shards.size();
    }

    vortex::Session session;
    std::optional<vortex::Expression> projection;
    std::optional<vortex::Expression> filter;
    std::vector<duckdb::unique_ptr<PullFile>> files;
    std::vector<Shard> shards;
    std::atomic<idx_t> next_shard {0};
    std::atomic<idx_t> shards_done {0};
    FileSystem *fs = nullptr;
};

struct PullLocalState final : LocalTableFunctionState {
    ~PullLocalState() override {
        if (exporter != nullptr) {
            duckdb_pull_exporter_free(exporter);
        }
    }

    // Batch id used as the DuckDB partition index; one exported batch = one partition.
    idx_t partition_index = 0;
    // This thread's own copies of the pushdown expressions: sharing one expression tree
    // across workers turns every per-shard scan setup into refcount contention on the same
    // cache lines.
    std::optional<vortex::Expression> projection;
    std::optional<vortex::Expression> filter;
    std::optional<vortex::PullScan> scan;
    idx_t file_index = DConstants::INVALID_INDEX;
    unique_ptr<FileHandle> handle;
    // The batch currently being exported; the exporter borrows it.
    std::optional<vortex::Array> batch;
    void *exporter = nullptr;
};

// Parse a footer by reading the file tail through the DuckDB file handle.
vortex::Footer read_footer(const vortex::Session &session, FileHandle &handle,
                           uint64_t file_size) {
    vortex::PullFooter pf(session, file_size);
    while (auto read = pf.next_read()) {
        handle.Read(read->data().data(), read->data().size(), read->offset());
        pf.complete(*read);
    }
    return std::move(pf).take();
}

// Footers are immutable per (path, size) and expensive to parse (the whole segment map), so
// they are shared process-wide across threads and queries. Keyed by size as a cheap staleness
// check; a rewritten same-size file would need an mtime key instead.
std::shared_ptr<vortex::Footer> cached_footer(const vortex::Session &session, FileHandle &handle,
                                              const std::string &path, uint64_t file_size) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<vortex::Footer>> cache;
    const std::string key = path + "#" + std::to_string(file_size);
    {
        std::lock_guard<std::mutex> guard(mutex);
        if (auto it = cache.find(key); it != cache.end()) {
            return it->second;
        }
    }
    auto footer = std::make_shared<vortex::Footer>(read_footer(session, handle, file_size));
    std::lock_guard<std::mutex> guard(mutex);
    return cache.emplace(key, std::move(footer)).first->second;
}

// Build the file's shared footer, reader tree, and (when the file is split) split points on
// first touch; later workers reuse them.
void ensure_file(PullGlobalState &gs, PullFile &file, FileHandle &handle, bool need_points) {
    std::lock_guard<std::mutex> guard(file.init_mutex);
    if (!file.footer) {
        file.footer = cached_footer(gs.session, handle, file.path,
                                    static_cast<uint64_t>(handle.GetFileSize()));
        if (gs.filter.has_value()) {
            vx_error *error = nullptr;
            file.pruned = vx_footer_can_prune(
                vortex::detail::Access::c_ptr(gs.session),
                vortex::detail::Access::c_ptr(*file.footer),
                vortex::detail::Access::c_ptr(*gs.filter), &error);
            if (error != nullptr) {
                throw_vx_error(error);
            }
        }
        if (!file.pruned) {
            file.ctx = std::make_shared<vortex::PullScan::File>(gs.session, *file.footer);
        }
    }
    if (need_points && !file.pruned && !file.points) {
        file.points = std::make_shared<const std::vector<uint64_t>>(
            file.footer->split_points(gs.session, gs.projection, gs.filter));
    }
}

// The chunk-aligned row range of shard "part" of "parts": an equal slice of the file's split
// points, so concurrent parts never share a data segment.
vortex::RowRange part_range(const std::vector<uint64_t> &points, uint32_t part, uint32_t parts) {
    if (points.size() < 2) {
        return vortex::RowRange {0, 0};
    }
    const size_t spans = points.size() - 1;
    const size_t lo = spans * part / parts;
    const size_t hi = spans * (part + 1) / parts;
    return vortex::RowRange {points[lo], points[hi]};
}

} // namespace

unique_ptr<GlobalTableFunctionState> try_init_global(ClientContext &context,
                                                     TableFunctionInitInput &,
                                                     const vector<duckdb::string> &patterns,
                                                     const duckdb_vx_tfunc_init_input &ffi_input) {
    if (!pull_enabled() || patterns.empty()) {
        return nullptr;
    }

    duckdb_vx_pull_plan plan {};
    duckdb_vx_error error = nullptr;
    duckdb_pull_plan(&ffi_input, &plan, &error);
    if (error != nullptr) {
        throw_ffi_error(error);
    }
    if (!plan.supported) {
        return nullptr;
    }

    auto session = vortex::detail::Access::adopt<vortex::Session>(duckdb_vortex_session());
    auto state = make_uniq<PullGlobalState>(std::move(session));
    state->projection =
        vortex::detail::Access::adopt<vortex::Expression>(const_cast<const vx_expression *>(plan.projection));
    if (plan.filter != nullptr) {
        state->filter =
            vortex::detail::Access::adopt<vortex::Expression>(const_cast<const vx_expression *>(plan.filter));
    }

    auto &fs = FileSystem::GetFileSystem(context);
    state->fs = &fs;
    const auto threads =
        static_cast<uint64_t>(duckdb::TaskScheduler::GetScheduler(context).NumberOfThreads());
    for (const auto &pattern : patterns) {
        for (auto &entry : fs.GlobFiles(pattern)) {
            const idx_t file_index = state->files.size();
            auto file = make_uniq<PullFile>();
            file->path = entry.path;
            file->cache = duckdb_pull_cache_new(file_index);
            state->files.push_back(std::move(file));
            const auto file_size = static_cast<uint64_t>(fs.GetFileSize(
                *fs.OpenFile(entry.path, duckdb::FileFlags::FILE_FLAGS_READ)));
            const auto parts = static_cast<uint32_t>(std::clamp<uint64_t>(
                file_size / TARGET_SHARD_BYTES, 1, threads * 2));
            for (uint32_t part = 0; part < parts; ++part) {
                state->shards.push_back(Shard {file_index, part, parts});
            }
        }
    }
    return state;
}

bool is_pull_state(const GlobalTableFunctionState &state) {
    return dynamic_cast<const PullGlobalState *>(&state) != nullptr;
}

unique_ptr<LocalTableFunctionState> init_local(const duckdb_vx_tfunc_init_input &ffi_input) {
    auto state = make_uniq<PullLocalState>();
    duckdb_vx_pull_plan plan {};
    duckdb_vx_error error = nullptr;
    duckdb_pull_plan(&ffi_input, &plan, &error);
    if (error != nullptr) {
        throw_ffi_error(error);
    }
    D_ASSERT(plan.supported); // the global state took the pull path with the same input
    state->projection = vortex::detail::Access::adopt<vortex::Expression>(
        const_cast<const vx_expression *>(plan.projection));
    if (plan.filter != nullptr) {
        state->filter = vortex::detail::Access::adopt<vortex::Expression>(
            const_cast<const vx_expression *>(plan.filter));
    }
    return state;
}

void scan(TableFunctionInput &input, DataChunk &output) {
    auto &gs = input.global_state->Cast<PullGlobalState>();
    auto &ls = input.local_state->Cast<PullLocalState>();

    for (;;) {
        // Drain the current exporter first: one DataChunk per call.
        if (ls.exporter != nullptr) {
            duckdb_vx_error error = nullptr;
            const bool has_more = duckdb_pull_exporter_next(
                ls.exporter, reinterpret_cast<duckdb_data_chunk>(&output), &error);
            if (error != nullptr) {
                throw_ffi_error(error);
            }
            if (!has_more) {
                duckdb_pull_exporter_free(ls.exporter);
                ls.exporter = nullptr;
                ls.batch.reset();
            }
            if (output.size() > 0) {
                return;
            }
            continue;
        }

        // Pick up the next shard when idle.
        if (!ls.scan.has_value()) {
            const idx_t shard_idx = gs.next_shard.fetch_add(1);
            if (shard_idx >= gs.shards.size()) {
                return; // empty chunk: this thread is done
            }
            const Shard &shard = gs.shards[shard_idx];
            PullFile &file = *gs.files[shard.file_index];
            if (ls.file_index != shard.file_index || ls.handle == nullptr) {
                ls.handle = gs.fs->OpenFile(file.path, duckdb::FileFlags::FILE_FLAGS_READ);
                ls.file_index = shard.file_index;
            }
            ensure_file(gs, file, *ls.handle, shard.parts > 1);
            if (file.pruned) {
                gs.shards_done.fetch_add(1);
                continue;
            }
            vortex::ScanOptions options;
            options.projection = ls.projection;
            options.filter = ls.filter;
            if (shard.parts > 1) {
                options.row_range = part_range(*file.points, shard.part, shard.parts);
            }
            // A bounded read window keeps time-to-first-batch low: LIMIT queries stop pulling
            // after a few chunks, so the whole shard must not be read up front.
            ls.scan.emplace(gs.files[shard.file_index]->ctx->scan(options, /*max_inflight=*/16));
        }

        // Drive the coroutine until it yields a batch or finishes the shard.
        bool have_batch = false;
        while (auto event = ls.scan->advance()) {
            if (auto *reads = std::get_if<vortex::PullScan::Reads>(&*event)) {
                if (reads->empty()) {
                    // A live split waits on a segment another thread's scan is reading (e.g. a
                    // shared dictionary); let that thread run.
                    std::this_thread::yield();
                    continue;
                }
                for (auto &read : *reads) {
                    ls.handle->Read(read.data().data(), read.data().size(), read.offset());
                    ls.scan->complete(read);
                }
            } else {
                ls.batch.emplace(std::get<vortex::Array>(std::move(*event)));
                ls.partition_index = gs.batch_id.fetch_add(1);
                duckdb_vx_error error = nullptr;
                ls.exporter = duckdb_pull_exporter_new(
                    reinterpret_cast<const vx_array *>(vortex::detail::Access::c_ptr(*ls.batch)),
                    gs.files[ls.file_index]->cache, &error);
                if (error != nullptr) {
                    throw_ffi_error(error);
                }
                have_batch = true;
                break;
            }
        }
        if (!have_batch) {
            ls.scan.reset();
            gs.shards_done.fetch_add(1);
        }
    }
}

duckdb::OperatorPartitionData partition_data(duckdb::TableFunctionGetPartitionInput &input) {
    const auto &ls = input.local_state->Cast<PullLocalState>();
    duckdb::OperatorPartitionData out(ls.partition_index);
    if (!input.partition_info.partition_columns.empty()) {
        throw duckdb::InternalException(
            "pull scan: partition columns are not constant per partition");
    }
    return out;
}

double progress(const GlobalTableFunctionState &state) {
    const auto &gs = state.Cast<PullGlobalState>();
    if (gs.shards.empty()) {
        return 100.0;
    }
    return 100.0 * static_cast<double>(gs.shards_done.load()) /
           static_cast<double>(gs.shards.size());
}

} // namespace vortex_pull
