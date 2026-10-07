// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
struct Handle {
    hnswlib::L2Space space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    std::mutex mutex;
    Handle(const char *path, uint32_t dimension) : space(dimension) {
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, path, false);
    }
};

template <class F>
int checked(char *error, F operation) noexcept {
    try {
        operation();
        return 0;
    } catch (const std::exception &exception) {
        std::snprintf(error, 1024, "%s", exception.what());
    } catch (...) {
        std::snprintf(error, 1024, "Unknown hnswlib exception");
    }
    return 1;
}
} // namespace

extern "C" int vortex_hnswlib_build(const char *path,
                                    const float *vectors,
                                    uint32_t dimension,
                                    uint32_t rows,
                                    uint32_t m,
                                    uint32_t construction,
                                    uint32_t seed,
                                    uint32_t threads,
                                    char *error) noexcept {
    return checked(error, [&] {
        hnswlib::L2Space space(dimension);
        hnswlib::HierarchicalNSW<float> index(&space, rows, m, construction, seed);
        index.addPoint(vectors, 0);
        std::exception_ptr failure;
        std::mutex failure_mutex;
#pragma omp parallel for schedule(static) num_threads(threads)
        for (uint32_t row = 1; row < rows; ++row) {
            try {
                index.addPoint(vectors + static_cast<size_t>(row) * dimension, row);
            } catch (...) {
                std::lock_guard<std::mutex> guard(failure_mutex);
                if (!failure)
                    failure = std::current_exception();
            }
        }
        if (failure)
            std::rethrow_exception(failure);
        if (index.getCurrentElementCount() != rows)
            throw std::runtime_error("Incomplete hnswlib build");
        index.saveIndex(path);
    });
}

extern "C" int vortex_hnswlib_open(const char *path,
                                   uint32_t dimension,
                                   uint32_t rows,
                                   uint32_t m,
                                   void **output,
                                   char *error) noexcept {
    return checked(error, [&] {
        auto handle = std::make_unique<Handle>(path, dimension);
        if (handle->index->getCurrentElementCount() != rows || handle->index->M_ != m) {
            throw std::runtime_error("hnswlib index shape mismatch");
        }
        *output = handle.release();
    });
}

extern "C" int vortex_hnswlib_search(void *raw,
                                     const float *query,
                                     uint32_t k,
                                     uint32_t ef,
                                     uint32_t *ids,
                                     float *distances,
                                     uint32_t *count,
                                     char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        std::lock_guard<std::mutex> guard(handle.mutex);
        handle.index->setEf(ef);
        auto hits = handle.index->searchKnn(query, k);
        if (hits.size() > k) {
            throw std::runtime_error("hnswlib returned too many results");
        }
        *count = static_cast<uint32_t>(hits.size());
        for (uint32_t rank = *count; rank > 0; --rank) {
            distances[rank - 1] = hits.top().first;
            ids[rank - 1] = static_cast<uint32_t>(hits.top().second);
            hits.pop();
        }
    });
}

extern "C" void vortex_hnswlib_close(void *raw) noexcept {
    delete static_cast<Handle *>(raw);
}
