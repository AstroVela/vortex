// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

extern "C" int vortex_hnswlib_build(const char *,
                                    const float *,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    uint32_t,
                                    char *) noexcept;
extern "C" int vortex_hnswlib_open(const char *, uint32_t, uint32_t, uint32_t, void **, char *) noexcept;
extern "C" int vortex_hnswlib_search(void *,
                                     const float *,
                                     uint32_t,
                                     uint32_t,
                                     uint32_t *,
                                     float *,
                                     uint32_t *,
                                     char *) noexcept;
extern "C" void vortex_hnswlib_close(void *) noexcept;

namespace {
constexpr uint32_t ROWS = 1024;
using Handle = std::unique_ptr<void, decltype(&vortex_hnswlib_close)>;

void check(int status, const char *error) {
    if (status != 0) {
        throw std::runtime_error(error);
    }
}

void build(const std::string &path, const std::vector<float> &values, uint32_t dimension) {
    char error[1024] = {};
    const int status =
        vortex_hnswlib_build(path.c_str(), values.data(), dimension, ROWS, 8, 64, 100, 1, error);
    check(status, error);
}

Handle open(const std::string &path, uint32_t dimension) {
    char error[1024] = {};
    void *raw = nullptr;
    const int status = vortex_hnswlib_open(path.c_str(), dimension, ROWS, 8, &raw, error);
    check(status, error);
    return Handle(raw, vortex_hnswlib_close);
}

std::vector<std::pair<uint32_t, float>> search(const Handle &handle, const float *query) {
    char error[1024] = {};
    uint32_t ids[10], count = 0;
    float distances[10];
    const int status = vortex_hnswlib_search(handle.get(), query, 10, 64, ids, distances, &count, error);
    check(status, error);
    if (count != 10) {
        throw std::runtime_error("Unexpected result count");
    }
    std::vector<std::pair<uint32_t, float>> hits;
    for (uint32_t i = 0; i < count; ++i) {
        hits.emplace_back(ids[i], distances[i]);
    }
    return hits;
}

void exercise(const std::filesystem::path &root, uint32_t dimension) {
    const auto path = (root / (std::to_string(dimension) + ".bin")).string();
    std::vector<float> values(ROWS * dimension);
    for (uint32_t row = 0; row < ROWS; ++row) {
        for (uint32_t col = 0; col < dimension; ++col) {
            values[row * dimension + col] = ((row * 97 + col * 53 + row * col) % 65521) / 65521.0f;
        }
    }

    for (uint32_t threads : {0, 2, 4, 8}) {
        char error[1024] = {};
        const int status =
            vortex_hnswlib_build(path.c_str(), values.data(), dimension, ROWS, 8, 64, 100, threads, error);
        if (status == 0 || std::string(error).find("threads=1") == std::string::npos ||
            std::filesystem::exists(path)) {
            throw std::runtime_error("Unsupported construction threads must fail before writing");
        }
    }
    build(path, values, dimension);
    const auto handle = open(path, dimension);
    const float *query = values.data() + 7 * dimension;
    const auto expected = search(handle, query);
    if (expected.front() != std::make_pair(uint32_t(7), 0.0f)) {
        throw std::runtime_error("Self query mismatch");
    }

    std::atomic<bool> start {false};
    std::atomic<unsigned> failures {0};
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < 4; ++worker) {
        workers.emplace_back([&, worker] {
            while (!start.load()) {
                std::this_thread::yield();
            }
            try {
                if (worker < 2) {
                    const auto built = path + ".worker-" + std::to_string(worker);
                    for (unsigned i = 0; i < 8; ++i) {
                        build(built, values, dimension);
                        if (search(open(built, dimension), query) != expected) {
                            throw std::runtime_error("Independent build mismatch");
                        }
                    }
                } else if (worker == 2) {
                    for (unsigned i = 0; i < 80; ++i) {
                        if (search(open(path, dimension), query) != expected) {
                            throw std::runtime_error("Reopened search mismatch");
                        }
                    }
                } else {
                    for (unsigned i = 0; i < 4000; ++i) {
                        if (search(handle, query) != expected) {
                            throw std::runtime_error("Concurrent search mismatch");
                        }
                    }
                }
            } catch (const std::exception &error) {
                std::cerr << error.what() << '\n';
                ++failures;
            }
        });
    }
    start.store(true);
    for (auto &worker : workers) {
        worker.join();
    }
    if (failures.load() != 0) {
        throw std::runtime_error("Concurrent native operation failed");
    }
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2 || !std::filesystem::is_directory(argv[1])) {
            throw std::runtime_error("Expected a scratch directory argument");
        }
        for (uint32_t dimension : {17, 128, 129}) {
            exercise(argv[1], dimension);
        }
        std::cout << "Native concurrency regression passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
