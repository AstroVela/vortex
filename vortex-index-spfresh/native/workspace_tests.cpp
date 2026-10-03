// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "bridge.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct Scratch {
    std::filesystem::path path;
    Scratch() {
        const auto pattern = (std::filesystem::temp_directory_path() / "spfresh-workspace-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        require(mkdtemp(name.data()) != nullptr, "Cannot create private test scratch");
        path = name.data();
    }
    ~Scratch() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

struct Close {
    void operator()(void *handle) const {
        vortex_spfresh_close(handle);
    }
};
using NativeHandle = std::unique_ptr<void, Close>;
using Hits = std::pair<std::vector<int32_t>, std::vector<float>>;

struct Fixture {
    std::filesystem::path path;
    uint32_t dimension;
    uint32_t rows;
    uint32_t pages;
    std::vector<float> vectors;

    Fixture(const std::filesystem::path &root, uint32_t dimension_, uint32_t rows_, uint32_t pages_)
        : path(root), dimension(dimension_), rows(rows_), pages(pages_), vectors(size_t(rows) * dimension) {
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t component = 0; component < dimension; ++component) {
                vectors[size_t(row) * dimension + component] = float((row * 17 + component * 31) % 1009);
            }
        }
        char error[1024] = {};
        if (vortex_spfresh_build(path.c_str(), vectors.data(), dimension, rows, 64, pages, 4, error) != 0) {
            throw std::runtime_error(error);
        }
    }

    NativeHandle open() const {
        char error[1024] = {};
        void *handle = nullptr;
        if (vortex_spfresh_open(path.c_str(), dimension, rows, pages, &handle, error) != 0) {
            throw std::runtime_error(error);
        }
        return NativeHandle(handle);
    }

    Hits search(void *handle,
                uint32_t row,
                uint32_t max_check = 4096,
                uint32_t internal_results = 64,
                uint32_t k = 5) const {
        Hits hits {std::vector<int32_t>(k), std::vector<float>(k)};
        char error[1024] = {};
        if (vortex_spfresh_search(handle,
                                  vectors.data() + size_t(row) * dimension,
                                  dimension,
                                  1,
                                  k,
                                  max_check,
                                  internal_results,
                                  pages,
                                  hits.first.data(),
                                  hits.second.data(),
                                  error) != 0) {
            throw std::runtime_error(error);
        }
        require(hits.first[0] == static_cast<int32_t>(row) && hits.second[0] == 0.0F,
                "Search did not return the original nearest row");
        return hits;
    }
};

SpFreshWorkspaceStats
cached(void *handle, const Fixture &fixture, uint32_t max_check = 4096, uint32_t internal_results = 64) {
    const auto stats = vortex_spfresh_test_workspaces(handle);
    require(stats.thread_detached, "Upstream TLS retained a workspace after return");
    require(stats.postings && stats.heads, "Search did not retain handle-owned workspaces");
    // Upstream rounds the hash capacity to the next strictly larger power of two.
    require(stats.check_capacity > max_check && stats.check_capacity <= max_check * 2 &&
                stats.internal_results == internal_results &&
                stats.posting_buffer_bytes == fixture.pages * 4096,
            "Retained workspace has the wrong query capacity");
    return stats;
}

void same_workspace(const SpFreshWorkspaceStats &first, const SpFreshWorkspaceStats &second) {
    require(first.postings == second.postings && first.heads == second.heads,
            "Unchanged query options reallocated workspaces");
}

void test_reuse_and_handle_isolation(const Fixture &first, const Fixture &second) {
    auto a = first.open();
    auto b = second.open();
    const auto expected = first.search(a.get(), 7);
    const auto original = cached(a.get(), first);
    require(first.search(a.get(), 7) == expected, "Repeated search changed ranked hits");
    same_workspace(original, cached(a.get(), first));
    second.search(b.get(), 9);
    const auto other = cached(b.get(), second);
    require(other.postings != original.postings && other.heads != original.heads,
            "Different handles shared workspaces");
    require(first.search(a.get(), 7) == expected, "Another handle contaminated results");
    same_workspace(original, cached(a.get(), first));
    same_workspace(other, cached(b.get(), second));
    first.search(a.get(), 11, 4096, 64, 10);
    same_workspace(original, cached(a.get(), first));
}

void test_option_changes(const Fixture &fixture) {
    auto handle = fixture.open();
    for (const auto [max_check, probes] : std::array<std::pair<uint32_t, uint32_t>, 9> {{{1024, 16},
                                                                                         {8192, 16},
                                                                                         {8192, 128},
                                                                                         {65536, 128},
                                                                                         {8192, 128},
                                                                                         {8192, 32},
                                                                                         {4096, 64},
                                                                                         {4096, 128},
                                                                                         {1024, 128}}}) {
        auto control = fixture.open();
        const auto expected = fixture.search(control.get(), 7, max_check, probes);
        const auto control_state = cached(control.get(), fixture, max_check, probes);
        require(fixture.search(handle.get(), 7, max_check, probes) == expected,
                "Changed query options differ from a freshly opened control");
        const auto state = cached(handle.get(), fixture, max_check, probes);
        require(state.head_check_capacity == control_state.head_check_capacity,
                "Changed max_check retained a stale BKT hash capacity");
        require(fixture.search(handle.get(), 7, max_check, probes) == expected,
                "Changed query options do not repeat");
        same_workspace(state, cached(handle.get(), fixture, max_check, probes));
    }
}

void test_error_cleanup(const Fixture &fixture) {
    auto handle = fixture.open();
    const auto expected = fixture.search(handle.get(), 7);
    cached(handle.get(), fixture);
    std::vector<float> queries(fixture.vectors.begin() + 7 * fixture.dimension,
                               fixture.vectors.begin() + 9 * fixture.dimension);
    queries[fixture.dimension] = std::numeric_limits<float>::quiet_NaN();
    std::array<int32_t, 10> ids {};
    std::array<float, 10> distances {};
    char error[1024] = {};
    require(vortex_spfresh_search(handle.get(),
                                  queries.data(),
                                  fixture.dimension,
                                  2,
                                  5,
                                  4096,
                                  64,
                                  fixture.pages,
                                  ids.data(),
                                  distances.data(),
                                  error) != 0 &&
                std::string(error).find("Non-finite") != std::string::npos,
            "Invalid later batch vector was accepted");
    const auto failed = vortex_spfresh_test_workspaces(handle.get());
    require(!failed.postings && !failed.heads && failed.thread_detached,
            "Failed search retained a partially used workspace");
    require(fixture.search(handle.get(), 7) == expected, "Failed search contaminated the next call");
    const auto before = cached(handle.get(), fixture);
    require(vortex_spfresh_search(handle.get(),
                                  queries.data(),
                                  fixture.dimension,
                                  1,
                                  5,
                                  4096,
                                  64,
                                  fixture.pages - 1,
                                  ids.data(),
                                  distances.data(),
                                  error) != 0,
            "A smaller posting buffer page limit was accepted");
    same_workspace(before, cached(handle.get(), fixture));
}

void test_thread_migration_and_destruction(const Fixture &fixture) {
    auto handle = fixture.open();
    const auto expected = fixture.search(handle.get(), 7);
    const auto original = cached(handle.get(), fixture);
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            require(fixture.search(handle.get(), 7) == expected, "Thread migration changed hits");
            same_workspace(original, cached(handle.get(), fixture));
        } catch (...) {
            error = std::current_exception();
        }
    });
    worker.join();
    if (error) {
        std::rethrow_exception(error);
    }
    require(fixture.search(handle.get(), 7) == expected, "Worker TLS contaminated the caller");
    same_workspace(original, cached(handle.get(), fixture));
    const auto live = original.live_posting_workspaces;
    std::thread closer([owned = std::move(handle)]() mutable { owned.reset(); });
    closer.join();
    require(vortex_spfresh_test_workspaces(nullptr).live_posting_workspaces == live - 1,
            "Closing on another thread leaked the handle workspace");
    for (int repeat = 0; repeat < 8; ++repeat) {
        auto fresh = fixture.open();
        const auto unopened = vortex_spfresh_test_workspaces(fresh.get());
        require(!unopened.postings && !unopened.heads, "New handle inherited a previous cache");
        require(fixture.search(fresh.get(), 7) == expected, "Reopened handle changed hits");
        cached(fresh.get(), fixture);
    }
}

void test_concurrent_handles(const Fixture &first, const Fixture &second) {
    auto a = first.open();
    auto b = second.open();
    std::array<std::exception_ptr, 4> errors {};
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < errors.size(); ++worker) {
        threads.emplace_back([&, worker] {
            try {
                const auto &fixture = worker % 2 == 0 ? first : second;
                auto *handle = worker % 2 == 0 ? a.get() : b.get();
                for (uint32_t repeat = 0; repeat < 20; ++repeat) {
                    fixture.search(handle,
                                   7 + worker,
                                   repeat % 3 == 0 ? 8192 : 4096,
                                   repeat % 2 == 0 ? 16 : 128);
                    require(vortex_spfresh_test_workspaces(handle).thread_detached,
                            "Concurrent call left a workspace in its worker TLS");
                }
            } catch (...) {
                errors[worker] = std::current_exception();
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
    for (const auto &error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}
} // namespace

int main() {
    try {
        Scratch scratch;
        const Fixture first(scratch.path / "first", 8, 256, 12);
        const Fixture second(scratch.path / "second", 16, 320, 4);
        test_reuse_and_handle_isolation(first, second);
        test_option_changes(first);
        test_error_cleanup(first);
        test_thread_migration_and_destruction(first);
        test_concurrent_handles(first, second);
        const auto final = vortex_spfresh_test_workspaces(nullptr);
        require(final.thread_detached && final.live_posting_workspaces == 0,
                "Tests leaked thread-local or handle-owned workspaces");
        std::cout << "5 workspace lifecycle checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
