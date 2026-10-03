// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "bridge.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "inc/Core/BKT/Index.h"
#include "inc/Core/SPANN/Index.h"

namespace {
using Clock = std::chrono::steady_clock;

uint32_t number(const char *text) {
    const std::string value(text);
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("Expected an unsigned integer");
    }
    const auto parsed = std::stoull(value);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("Integer out of range");
    }
    return static_cast<uint32_t>(parsed);
}

void check(SPTAG::ErrorCode code) {
    if (code != SPTAG::ErrorCode::Success) {
        throw std::runtime_error("Native SPFresh operation failed");
    }
}

void clear_workspace() {
    SPTAG::SPANN::Index<float>::m_workspace.reset();
    SPTAG::BKT::Index<float>::m_workspace.reset();
}

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Close {
    void operator()(void *handle) const {
        vortex_spfresh_close(handle);
    }
};
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 12) {
            throw std::runtime_error(
                "Usage: spfresh_benchmark MODE NATIVE_ROOT QUERIES.f32bin ROWS K MAX_CHECK "
                "INTERNAL_RESULTS POSTING_PAGES WARMUP_ROUNDS ROUNDS OUTPUT.csv");
        }
        const std::string mode(argv[1]);
        const auto rows = number(argv[4]);
        const auto k = number(argv[5]);
        const auto max_check = number(argv[6]);
        const auto internal_results = number(argv[7]);
        const auto pages = number(argv[8]);
        const auto warmup = number(argv[9]);
        const auto rounds = number(argv[10]);
        if ((mode != "cpp-reset" && mode != "cpp-reuse" && mode != "bridge") || k == 0 || k > rows ||
            k > 4096 || internal_results < k || internal_results > 4096 || max_check < internal_results ||
            max_check > 1048576 || pages == 0 || pages > 4096 || rounds == 0 || rounds > 100 ||
            warmup > 100 || uint64_t(pages) * internal_results * 4096 > 256 * 1024 * 1024) {
            throw std::runtime_error("Invalid benchmark options");
        }
        std::ifstream input(argv[3], std::ios::binary);
        uint32_t count = 0;
        uint32_t dimension = 0;
        // The qualified platform is little-endian Linux x86_64, matching f32bin.
        input.read(reinterpret_cast<char *>(&count), sizeof(count));
        input.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
        if (!input || count == 0 || count > 256 || dimension == 0 || dimension > 4096) {
            throw std::runtime_error("Invalid query shape");
        }
        std::vector<float> queries(size_t(count) * dimension);
        input.read(reinterpret_cast<char *>(queries.data()), queries.size() * sizeof(float));
        if (!input || input.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Invalid query file length");
        }
        for (float component : queries) {
            if (!std::isfinite(component)) {
                throw std::runtime_error("Non-finite query component");
            }
        }
        if (std::filesystem::exists(argv[11])) {
            throw std::runtime_error("Benchmark output already exists");
        }
        std::ofstream output(argv[11]);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output << std::setprecision(std::numeric_limits<double>::max_digits10);
        output << "event,round,query,rank,native_id,distance,elapsed_ms\n";
        char error[1024] = {};
        void *opaque = nullptr;
        auto start = Clock::now();
        if (vortex_spfresh_open(argv[2], dimension, rows, pages, &opaque, error) != 0) {
            throw std::runtime_error(error);
        }
        std::unique_ptr<void, Close> handle(opaque);
        output << "open,,,,,," << elapsed(start) << '\n';
        auto &index = vortex_spfresh_benchmark_index(handle.get());
        if (mode != "bridge") {
            for (const char *section : {"BuildHead", "BuildSSDIndex"}) {
                check(index.SetParameter("MaxCheck", std::to_string(max_check).c_str(), section));
            }
            check(index.SetParameter("SearchInternalResultNum",
                                     std::to_string(internal_results).c_str(),
                                     "BuildSSDIndex"));
            check(
                index.SetParameter("SearchPostingPageLimit", std::to_string(pages).c_str(), "BuildSSDIndex"));
        }
        std::vector<int32_t> ids(k);
        std::vector<float> distances(k);
        for (uint32_t round = 0; round < warmup + rounds; ++round) {
            for (uint32_t query = 0; query < count; ++query) {
                const float *vector = queries.data() + size_t(query) * dimension;
                start = Clock::now();
                if (mode == "bridge") {
                    if (vortex_spfresh_search(handle.get(),
                                              vector,
                                              dimension,
                                              1,
                                              k,
                                              max_check,
                                              internal_results,
                                              pages,
                                              ids.data(),
                                              distances.data(),
                                              error) != 0) {
                        throw std::runtime_error(error);
                    }
                } else {
                    if (mode == "cpp-reset") {
                        clear_workspace();
                    }
                    {
                        SPTAG::QueryResult result(vector, static_cast<int>(k), false);
                        check(index.SearchIndex(result));
                        for (uint32_t rank = 0; rank < k; ++rank) {
                            const auto &hit = *result.GetResult(static_cast<int>(rank));
                            ids[rank] = hit.VID;
                            distances[rank] = hit.Dist;
                        }
                    }
                    if (mode == "cpp-reset") {
                        clear_workspace();
                    }
                }
                const auto ms = elapsed(start);
                for (uint32_t rank = 0; rank < k; ++rank) {
                    if (ids[rank] < 0 || uint32_t(ids[rank]) >= rows || !std::isfinite(distances[rank]) ||
                        distances[rank] < 0) {
                        throw std::runtime_error("Invalid native result");
                    }
                    output << "sample," << round << ',' << query << ',' << rank << ',' << ids[rank] << ','
                           << distances[rank] << ',' << ms << '\n';
                }
            }
        }
        start = Clock::now();
        handle.reset();
        output << "close,,,,,," << elapsed(start) << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
