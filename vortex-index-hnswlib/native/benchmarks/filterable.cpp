// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "../hnswlib/payload_graph.h"
#include <chrono>
#include <iostream>
#include <random>
#include <set>

using Clock = std::chrono::steady_clock;
using Hits = std::priority_queue<std::pair<float, hnswlib::labeltype>>;

struct Filter final : hnswlib::BaseFilterFunctor {
    size_t groups;
    bool operator()(hnswlib::labeltype label) override {
        return label % 100 < groups;
    }
};

struct Samples {
    std::vector<double> micros;
    size_t correct = 0;
    size_t expected = 0;

    template <class F>
    void query(F search, const std::set<hnswlib::labeltype> &truth, Filter &filter) {
        const auto start = Clock::now();
        auto hits = search();
        micros.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        expected += truth.size();
        std::set<hnswlib::labeltype> unique;
        while (!hits.empty()) {
            const auto label = hits.top().second;
            if (!filter(label) || !unique.insert(label).second) {
                throw std::runtime_error("Invalid filtered benchmark result");
            }
            correct += truth.count(label);
            hits.pop();
        }
    }

    void print(const char *workload, size_t groups, const char *algorithm) {
        std::sort(micros.begin(), micros.end());
        std::cout << workload << ',' << groups << ',' << algorithm << ',' << double(correct) / expected << ','
                  << micros[micros.size() / 2] << ',' << micros[micros.size() * 95 / 100] << '\n';
    }
};

int main() {
    constexpr size_t rows = 10000, dimension = 16, query_count = 80, k = 10;
    std::cout << "workload,selectivity_percent,algorithm,recall_at_10,p50_us,p95_us\n";
    for (bool correlated : {false, true}) {
        const char *workload = correlated ? "correlated" : "independent";
        std::mt19937 random(42);
        std::normal_distribution<float> normal;
        std::vector<float> vectors(rows * dimension), queries(query_count * dimension);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < dimension; ++col) {
                vectors[row * dimension + col] = normal(random);
            }
            if (correlated) {
                vectors[row * dimension] += float(row % 100) * 3;
            }
        }
        for (size_t query = 0; query < query_count; ++query) {
            for (size_t col = 0; col < dimension; ++col) {
                queries[query * dimension + col] = normal(random);
            }
            if (correlated) {
                queries[query * dimension] += float(query % 100) * 3;
            }
        }
        hnswlib::L2Space space(dimension);
        const auto start = Clock::now();
        hnswlib::HierarchicalNSW<float> base(&space, rows, 8, 100, 17);
        for (size_t row = 0; row < rows; ++row) {
            base.addPoint(vectors.data() + row * dimension, row);
        }
        const auto built = Clock::now();
        hnswlib::PayloadGraphBuilder builder(base, 8, 100, {1000000, 256, 1000, rows});
        for (size_t group = 0; group < 100; ++group) {
            std::vector<hnswlib::labeltype> labels;
            for (size_t row = group; row < rows; row += 100) {
                labels.push_back(row);
            }
            builder.add_block(labels);
        }
        const auto payload = builder.finish();
        payload.validate(rows);
        std::cerr << workload << ": base_build_s=" << std::chrono::duration<double>(built - start).count()
                  << ", payload_build_s=" << std::chrono::duration<double>(Clock::now() - built).count()
                  << ", payload_edges=" << payload.links.size()
                  << ", csr_bytes=" << payload.offsets.size() * 8 + payload.links.size() * 4 << '\n';
        base.setEf(32);
        for (size_t groups : {1, 2, 10, 50, 100}) {
            Filter filter;
            filter.groups = groups;
            Samples ordinary, augmented;
            for (size_t query = 0; query < query_count; ++query) {
                const auto *point = queries.data() + query * dimension;
                Hits exact;
                for (size_t row = 0; row < rows; ++row) {
                    if (!filter(row)) {
                        continue;
                    }
                    exact.emplace(space.get_dist_func()(point,
                                                        vectors.data() + row * dimension,
                                                        space.get_dist_func_param()),
                                  row);
                    if (exact.size() > k) {
                        exact.pop();
                    }
                }
                std::set<hnswlib::labeltype> truth;
                while (!exact.empty()) {
                    truth.insert(exact.top().second);
                    exact.pop();
                }
                const auto run_base = [&] {
                    ordinary.query([&] { return base.searchKnn(point, k, &filter); }, truth, filter);
                };
                const auto run_payload = [&] {
                    augmented.query(
                        [&] { return hnswlib::searchKnnWithPayload(base, payload, point, k, &filter); },
                        truth,
                        filter);
                };
                // Alternate order to reduce systematic cache bias. Truth work is not timed.
                if (query % 2 == 0) {
                    run_base();
                    run_payload();
                } else {
                    run_payload();
                    run_base();
                }
            }
            ordinary.print(workload, groups, "base_filter");
            augmented.print(workload, groups, "payload_filter");
        }
    }
}
