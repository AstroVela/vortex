// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "bridge.h"
#include "posting_io.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "inc/Core/BKT/Index.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Helper/SimpleIniReader.h"

namespace SPTAG::SPANN {
extern std::function<std::shared_ptr<Helper::DiskIO>()> f_createAsyncIO;
}

namespace {
std::mutex native_mutex;
std::once_flag logger_initialized;

// The native boundary serializes this temporary loader-factory override.
struct PostingReaderFactory {
    decltype(SPTAG::SPANN::f_createAsyncIO) previous;
    PostingReaderFactory(const PostingReaderFactory &) = delete;
    PostingReaderFactory &operator=(const PostingReaderFactory &) = delete;
    PostingReaderFactory() : previous(SPTAG::SPANN::f_createAsyncIO) {
        if (vortex_spfresh::posting_views_enabled()) {
            SPTAG::SPANN::f_createAsyncIO = [] {
                return std::shared_ptr<SPTAG::Helper::DiskIO>(
                    std::make_shared<vortex_spfresh::PostingFileIO>());
            };
        }
    }
    ~PostingReaderFactory() {
        SPTAG::SPANN::f_createAsyncIO = std::move(previous);
    }
};

// Both upstream workspaces are shared by every Float32 handle on this thread.
struct Workspaces {
    Workspaces() {
        clear();
    }
    ~Workspaces() {
        clear();
    }
    static void clear() {
        SPTAG::SPANN::Index<float>::m_workspace.reset();
        SPTAG::BKT::Index<float>::m_workspace.reset();
    }
};

struct Handle {
    SPTAG::SPANN::Index<float> index;
    uint32_t dimension;
    uint32_t rows;
    uint32_t posting_pages;
    std::shared_ptr<SPTAG::SPANN::ExtraWorkSpace> spann_workspace;
    std::shared_ptr<SPTAG::COMMON::WorkSpace> bkt_workspace;
    uint32_t workspace_max_check = 0;
    uint32_t workspace_internal_results = 0;
};

// A serialized, synchronous call lends this handle's buffers to the current TLS.
// Keeping only one pair per handle also bounds retention when worker threads change.
struct SearchWorkspaces {
    Handle &handle;

    SearchWorkspaces(Handle &owner, uint32_t max_check, uint32_t internal_results) : handle(owner) {
        // Dimension, page size and hash configuration are fixed by the owning handle.
        // Upstream only initializes capacities when TLS is empty, not on option changes.
        if (handle.workspace_max_check != max_check ||
            handle.workspace_internal_results != internal_results) {
            handle.spann_workspace.reset();
            handle.bkt_workspace.reset();
        }
        handle.workspace_max_check = max_check;
        handle.workspace_internal_results = internal_results;
        SPTAG::SPANN::Index<float>::m_workspace = std::move(handle.spann_workspace);
        SPTAG::BKT::Index<float>::m_workspace = std::move(handle.bkt_workspace);
    }

    SearchWorkspaces(const SearchWorkspaces &) = delete;
    SearchWorkspaces &operator=(const SearchWorkspaces &) = delete;

    ~SearchWorkspaces() {
        Workspaces::clear();
    }

    void retain() noexcept {
        handle.spann_workspace = std::move(SPTAG::SPANN::Index<float>::m_workspace);
        handle.bkt_workspace = std::move(SPTAG::BKT::Index<float>::m_workspace);
    }
};

void check(SPTAG::ErrorCode result, const char *operation) {
    if (result != SPTAG::ErrorCode::Success) {
        throw std::runtime_error(std::string(operation) + " failed (" +
                                 std::to_string(static_cast<int>(result)) + ")");
    }
}

template <class F>
int boundary(char *error, F &&function) noexcept {
    try {
        std::lock_guard<std::mutex> lock(native_mutex);
        std::call_once(logger_initialized, [] {
            SPTAG::g_pLogger =
                std::make_shared<SPTAG::Helper::SimpleLogger>(SPTAG::Helper::LogLevel::LL_Warning);
        });
        Workspaces workspaces;
        function();
        return 0;
    } catch (const std::exception &exception) {
        std::snprintf(error, 1024, "%s", exception.what());
    } catch (...) {
        std::snprintf(error, 1024, "Unknown SPFresh exception");
    }
    return -1;
}
} // namespace

extern "C" int vortex_spfresh_build(const char *root,
                                    const float *vectors,
                                    uint32_t dimension,
                                    uint32_t rows,
                                    uint32_t heads,
                                    uint32_t posting_pages,
                                    uint32_t replicas,
                                    char *error) {
    return boundary(error, [&] {
        const uint64_t components = static_cast<uint64_t>(dimension) * rows;
        if (vectors == nullptr || dimension == 0 || dimension > 4096 || rows < 64 || rows > INT32_MAX ||
            heads < 32 || heads >= rows || posting_pages == 0 || posting_pages > 4096 || replicas == 0 ||
            replicas > 8 || components * sizeof(float) > INT32_MAX) {
            throw std::runtime_error("Invalid SPFresh static build options");
        }
        for (uint64_t component = 0; component < components; ++component) {
            if (!std::isfinite(vectors[component])) {
                throw std::runtime_error("Non-finite build vector component");
            }
        }
        const std::filesystem::path directory(root);
        if (!directory.is_absolute() || !std::filesystem::create_directory(directory)) {
            throw std::runtime_error("SPFresh build requires a new absolute directory");
        }
        {
            SPTAG::SPANN::Index<float> index;
            const auto set = [&](const char *section, const char *name, const std::string &value) {
                check(index.SetParameter(name, value.c_str(), section), name);
            };
            set("Base", "IndexAlgoType", "BKT");
            set("Base", "ValueType", "Float");
            set("Base", "DistCalcMethod", "L2");
            set("Base", "IndexDirectory", directory.string());
            set("Base", "HeadVectorIDs", "head_ids.bin");
            set("Base", "SSDIndex", "postings.bin");
            set("Base", "SSDIndexFileNum", "1");
            set("Base", "DataBlockSize", "1024");
            set("Base", "DataCapacity", std::to_string(rows));
            set("SelectHead", "isExecute", "true");
            set("SelectHead", "SelectHeadType", "Random");
            set("SelectHead", "Count", std::to_string(heads));
            set("SelectHead", "NumberOfThreads", "1");
            set("BuildHead", "isExecute", "true");
            set("BuildHead", "DistCalcMethod", "L2");
            set("BuildHead", "NumberOfThreads", "1");
            set("BuildHead", "BKTKmeansK", "4");
            set("BuildHead", "TPTNumber", "4");
            set("BuildHead", "TPTLeafSize", "32");
            set("BuildHead", "NeighborhoodSize", "16");
            set("BuildHead", "RefineIterations", "2");
            set("BuildHead", "DataBlockSize", "1024");
            set("BuildHead", "DataCapacity", std::to_string(rows));
            set("BuildSSDIndex", "isExecute", "true");
            set("BuildSSDIndex", "BuildSsdIndex", "true");
            set("BuildSSDIndex", "NumberOfThreads", "1");
            set("BuildSSDIndex", "IOThreadsPerHandler", "1");
            set("BuildSSDIndex", "UseKV", "false");
            set("BuildSSDIndex", "UseSPDK", "false");
            set("BuildSSDIndex", "ExcludeHead", "true");
            set("BuildSSDIndex", "EnableDataCompression", "false");
            set("BuildSSDIndex", "EnableDeltaEncoding", "false");
            set("BuildSSDIndex", "EnablePostingListRearrange", "false");
            set("BuildSSDIndex", "OutputEmptyReplicaID", "false");
            set("BuildSSDIndex", "PostingPageLimit", std::to_string(posting_pages));
            set("BuildSSDIndex", "SearchPostingPageLimit", std::to_string(posting_pages));
            set("BuildSSDIndex", "ReplicaCount", std::to_string(replicas));
            set("BuildSSDIndex", "InternalResultNum", "32");
            set("BuildSSDIndex", "SearchInternalResultNum", "32");
            set("BuildSSDIndex", "TmpDir", directory.string());
            // The synchronous L2 build borrows the immutable input without normalization.
            check(index.BuildIndex(vectors, rows, dimension, false, true), "BuildIndex");
        }
        for (const auto *name : {"vectors.bin", "tree.bin", "graph.bin", "deletes.bin"}) {
            std::filesystem::rename(directory / "HeadIndex" / name, directory / name);
        }
    });
}

extern "C" int vortex_spfresh_open(const char *root,
                                   uint32_t dimension,
                                   uint32_t rows,
                                   uint32_t posting_pages,
                                   void **output,
                                   char *error) {
    *output = nullptr;
    return boundary(error, [&] {
        if (dimension == 0 || dimension > 4096 || rows == 0 || rows > INT32_MAX || posting_pages == 0 ||
            posting_pages > 4096) {
            throw std::runtime_error("Invalid SPFresh bundle dimensions or page limit");
        }
        auto handle = std::make_unique<Handle>();
        PostingReaderFactory posting_reader;
        handle->dimension = dimension;
        handle->rows = rows;
        handle->posting_pages = posting_pages;
        SPTAG::Helper::IniReader config;
        const auto set = [&](const char *section, const char *name, const std::string &value) {
            config.SetParameter(section, name, value);
        };
        set("Base", "IndexAlgoType", "BKT");
        set("Base", "ValueType", "Float");
        set("Base", "DistCalcMethod", "L2");
        set("Base", "Dim", std::to_string(dimension));
        set("Base", "VectorSize", std::to_string(rows));
        set("Base", "IndexDirectory", root);
        set("Base", "SSDIndex", "postings.bin");
        set("Base", "DataBlockSize", "1024");
        set("Base", "DataCapacity", std::to_string(rows));
        set("BuildHead", "DistCalcMethod", "L2");
        set("BuildHead", "DataBlockSize", "1024");
        set("BuildHead", "DataCapacity", std::to_string(rows));
        set("BuildHead", "NumberOfThreads", "1");
        set("BuildSSDIndex", "NumberOfThreads", "1");
        set("BuildSSDIndex", "IOThreadsPerHandler", "1");
        set("BuildSSDIndex", "UseKV", "false");
        set("BuildSSDIndex", "UseSPDK", "false");
        set("BuildSSDIndex", "ExcludeHead", "true");
        set("BuildSSDIndex", "PostingPageLimit", std::to_string(posting_pages));
        set("BuildSSDIndex", "SearchPostingPageLimit", std::to_string(posting_pages));
        set("BuildSSDIndex", "EnableDataCompression", "false");
        set("BuildSSDIndex", "EnableDeltaEncoding", "false");
        set("BuildSSDIndex", "EnablePostingListRearrange", "false");
        check(handle->index.LoadConfig(config), "LoadConfig");
        std::vector<std::shared_ptr<SPTAG::Helper::DiskIO>> streams;
        for (const auto *name : {"vectors.bin", "tree.bin", "graph.bin", "deletes.bin", "head_ids.bin"}) {
            auto stream = SPTAG::f_createIO();
            const auto path = (std::filesystem::path(root) / name).string();
            if (!stream || !stream->Initialize(path.c_str(), std::ios::binary | std::ios::in)) {
                throw std::runtime_error("Cannot open native bundle file: " + path);
            }
            streams.push_back(std::move(stream));
        }
        check(handle->index.LoadIndexData(streams), "LoadIndexData");
        if (handle->index.GetFeatureDim() != static_cast<int>(dimension) ||
            handle->index.GetMemoryIndex()->GetNumDeleted() != 0) {
            throw std::runtime_error("Native schema mismatch or deleted heads");
        }
        handle->index.SetReady(true);
        *output = handle.release();
    });
}

extern "C" int vortex_spfresh_search(void *opaque,
                                     const float *queries,
                                     uint32_t dimension,
                                     uint32_t count,
                                     uint32_t k,
                                     uint32_t max_check,
                                     uint32_t internal_results,
                                     uint32_t search_pages,
                                     int32_t *ids,
                                     float *distances,
                                     char *error) {
    return boundary(error, [&] {
        auto &handle = *static_cast<Handle *>(opaque);
        if (dimension != handle.dimension || count == 0 || k == 0 || k > handle.rows ||
            internal_results < k || internal_results > 4096 || max_check < internal_results ||
            max_check > 1048576 || search_pages != handle.posting_pages) {
            throw std::runtime_error("Invalid SPFresh query options");
        }
        SearchWorkspaces workspaces(handle, max_check, internal_results);
        const auto set = [&](const char *name, uint32_t value, const char *section) {
            check(handle.index.SetParameter(name, std::to_string(value).c_str(), section), name);
        };
        set("MaxCheck", max_check, "BuildHead");
        set("MaxCheck", max_check, "BuildSSDIndex");
        set("SearchInternalResultNum", internal_results, "BuildSSDIndex");
        set("SearchPostingPageLimit", search_pages, "BuildSSDIndex");
        for (uint32_t query = 0; query < count; ++query) {
            const float *vector = queries + static_cast<size_t>(query) * dimension;
            for (uint32_t component = 0; component < dimension; ++component) {
                if (!std::isfinite(vector[component])) {
                    throw std::runtime_error("Non-finite query component");
                }
            }
            SPTAG::QueryResult result(vector, static_cast<int>(k), false);
            check(handle.index.SearchIndex(result), "SearchIndex");
            for (uint32_t rank = 0; rank < k; ++rank) {
                const auto &hit = *result.GetResult(static_cast<int>(rank));
                if (hit.VID < -1 || hit.VID >= static_cast<int>(handle.rows) ||
                    (hit.VID >= 0 && (!std::isfinite(hit.Dist) || hit.Dist < 0))) {
                    throw std::runtime_error("Invalid native search result");
                }
                const auto offset = static_cast<size_t>(query) * k + rank;
                ids[offset] = hit.VID;
                distances[offset] = hit.Dist;
            }
        }
        workspaces.retain();
    });
}

extern "C" void vortex_spfresh_close(void *opaque) {
    std::lock_guard<std::mutex> lock(native_mutex);
    Workspaces workspaces;
    delete static_cast<Handle *>(opaque);
}

#ifdef VORTEX_SPFRESH_BENCHMARK
SPTAG::SPANN::Index<float> &vortex_spfresh_benchmark_index(void *opaque) {
    return static_cast<Handle *>(opaque)->index;
}
#endif

#ifdef VORTEX_SPFRESH_TESTING
SpFreshWorkspaceStats vortex_spfresh_test_workspaces(void *opaque) {
    std::lock_guard<std::mutex> lock(native_mutex);
    SpFreshWorkspaceStats stats;
    stats.thread_detached =
        !SPTAG::SPANN::Index<float>::m_workspace && !SPTAG::BKT::Index<float>::m_workspace;
    stats.live_posting_workspaces = SPTAG::SPANN::ExtraWorkSpace::g_spaceCount.load();
    if (opaque != nullptr) {
        const auto &handle = *static_cast<Handle *>(opaque);
        stats.postings = handle.spann_workspace.get();
        stats.heads = handle.bkt_workspace.get();
        if (handle.bkt_workspace) {
            stats.head_check_capacity = handle.bkt_workspace->nodeCheckStatus.MaxCheck();
        }
        if (handle.spann_workspace) {
            auto &workspace = *handle.spann_workspace;
            stats.internal_results = static_cast<uint32_t>(workspace.m_pageBuffers.size());
            stats.check_capacity = workspace.m_deduper.MaxCheck();
            stats.posting_buffer_bytes = workspace.m_pageBuffers[0].GetPageSize();
        }
    }
    return stats;
}
#endif
