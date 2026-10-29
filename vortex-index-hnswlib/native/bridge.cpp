// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "hnswlib/payload_graph.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace {
float l2_avx_residuals(const void *left, const void *right, const void *parameter) {
    const size_t dimension = *static_cast<const size_t *>(parameter);
    const size_t aligned = dimension / 16 * 16;
    const size_t tail = dimension - aligned;
    return hnswlib::L2SqrSIMD16ExtAVX(left, right, &aligned) +
           hnswlib::L2Sqr(static_cast<const float *>(left) + aligned,
                          static_cast<const float *>(right) + aligned,
                          &tail);
}

// L2Space mutates a global SIMD pointer, also read by its residual kernel. Bind
// directly to stateless kernels; Rust checks the required AVX2 CPU support.
class FixedL2Space final : public hnswlib::SpaceInterface<float> {
public:
    explicit FixedL2Space(size_t dimension) : dimension_(dimension), distance_(select(dimension)) {
    }
    size_t get_data_size() override {
        return dimension_ * sizeof(float);
    }
    hnswlib::DISTFUNC<float> get_dist_func() override {
        return distance_;
    }
    void *get_dist_func_param() override {
        return &dimension_;
    }

private:
    static hnswlib::DISTFUNC<float> select(size_t dimension) {
        if (dimension % 16 == 0) {
            return hnswlib::L2SqrSIMD16ExtAVX;
        }
        if (dimension % 4 == 0) {
            return hnswlib::L2SqrSIMD4Ext;
        }
        if (dimension > 16) {
            return l2_avx_residuals;
        }
        if (dimension > 4) {
            return hnswlib::L2SqrSIMD4ExtResiduals;
        }
        return hnswlib::L2Sqr;
    }

    size_t dimension_;
    const hnswlib::DISTFUNC<float> distance_;
};

struct Handle {
    FixedL2Space space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index;
    hnswlib::PayloadGraph payload;
    std::unique_ptr<hnswlib::PayloadGraphBuilder> payload_builder;
    std::mutex mutex;
    Handle(const char *path, uint32_t dimension) : space(dimension) {
        index = std::make_unique<hnswlib::HierarchicalNSW<float>>(&space, path, false);
        payload.offsets.resize(index->getCurrentElementCount() + 1, 0);
    }
};

class LabelMask final : public hnswlib::BaseFilterFunctor {
public:
    LabelMask(const uint64_t *words, size_t rows) : words_(words), rows_(rows) {
    }
    bool operator()(hnswlib::labeltype label) override {
        if (label >= rows_) {
            throw std::runtime_error("Filter label outside dense mapping");
        }
        return (words_[label / 64] & (uint64_t(1) << (label % 64))) != 0;
    }

private:
    const uint64_t *words_;
    size_t rows_;
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
        // Upstream addPoint draws from an unsynchronized per-index RNG.
        if (threads != 1) {
            throw std::invalid_argument("hnswlib construction requires threads=1");
        }
        FixedL2Space space(dimension);
        hnswlib::HierarchicalNSW<float> index(&space, rows, m, construction, seed);
        for (uint32_t row = 0; row < rows; ++row) {
            index.addPoint(vectors + static_cast<size_t>(row) * dimension, row);
        }
        if (index.getCurrentElementCount() != rows) {
            throw std::runtime_error("Incomplete hnswlib build");
        }
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
                                     const uint64_t *mask,
                                     uint64_t mask_words,
                                     uint32_t *ids,
                                     float *distances,
                                     uint32_t *count,
                                     char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        std::lock_guard<std::mutex> guard(handle.mutex);
        handle.index->setEf(ef);
        const auto rows = handle.index->getCurrentElementCount();
        if ((mask && mask_words != (rows + 63) / 64) || (!mask && mask_words != 0)) {
            throw std::invalid_argument("Invalid filter mask length");
        }
        LabelMask filter(mask, rows);
        auto hits =
            hnswlib::searchKnnWithPayload(*handle.index, handle.payload, query, k, mask ? &filter : nullptr);
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

extern "C" int vortex_hnswlib_payload_begin(void *raw,
                                            uint32_t m,
                                            uint32_t ef,
                                            uint64_t max_edges,
                                            uint32_t max_degree,
                                            uint32_t max_block_rows,
                                            uint64_t max_memberships,
                                            char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        handle.payload_builder = std::make_unique<hnswlib::PayloadGraphBuilder>(
            *handle.index,
            m,
            ef,
            hnswlib::PayloadBuildLimits {max_edges, max_degree, max_block_rows, max_memberships});
    });
}

extern "C" int
vortex_hnswlib_payload_block(void *raw, const uint32_t *labels, uint32_t count, char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        if (!handle.payload_builder) {
            throw std::runtime_error("Payload build not started");
        }
        handle.payload_builder->add_block(std::vector<hnswlib::labeltype>(labels, labels + count));
    });
}

extern "C" int vortex_hnswlib_payload_finish(void *raw, uint64_t *edges, char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        if (!handle.payload_builder) {
            throw std::runtime_error("Payload build not started");
        }
        handle.payload = handle.payload_builder->finish();
        handle.payload_builder.reset();
        *edges = handle.payload.links.size();
    });
}

extern "C" int vortex_hnswlib_payload_export(void *raw,
                                             uint64_t *offsets,
                                             uint32_t rows,
                                             uint32_t *links,
                                             uint64_t edges,
                                             char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        if (handle.payload.offsets.size() != size_t(rows) + 1 || handle.payload.links.size() != edges) {
            throw std::invalid_argument("Payload export shape mismatch");
        }
        std::copy(handle.payload.offsets.begin(), handle.payload.offsets.end(), offsets);
        std::copy(handle.payload.links.begin(), handle.payload.links.end(), links);
    });
}

extern "C" int vortex_hnswlib_payload_load(void *raw,
                                           const uint64_t *offsets,
                                           uint32_t rows,
                                           const uint32_t *links,
                                           uint64_t edges,
                                           char *error) noexcept {
    return checked(error, [&] {
        auto &handle = *static_cast<Handle *>(raw);
        if (handle.index->getCurrentElementCount() != rows) {
            throw std::invalid_argument("Payload load shape mismatch");
        }
        hnswlib::PayloadGraph payload;
        payload.offsets.assign(offsets, offsets + size_t(rows) + 1);
        payload.links.assign(links, links + edges);
        payload.validate(rows);
        handle.payload = std::move(payload);
    });
}
