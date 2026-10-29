// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "../hnswlib/payload_graph.h"
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <set>

struct Filter final : hnswlib::BaseFilterFunctor {
    std::set<hnswlib::labeltype> labels;
    bool operator()(hnswlib::labeltype label) override {
        return labels.count(label) != 0;
    }
};

template <class F>
void rejects(F operation) {
    bool failed = false;
    try {
        operation();
    } catch (const std::exception &) {
        failed = true;
    }
    assert(failed);
}

int main() {
    constexpr size_t rows = 1000;
    constexpr size_t dimension = 8;
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float> base(&space, rows, 4, 64, 17);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-1, 1);
    std::vector<float> vectors(rows * dimension);
    for (auto &value : vectors) {
        value = distribution(random);
    }
    for (size_t row = 0; row < rows; ++row) {
        // Exercise label/internal-ID translation, including labels above rows.
        base.addPoint(vectors.data() + row * dimension, 2 * (rows - row));
    }
    const std::vector<char> before(base.data_level0_memory_,
                                   base.data_level0_memory_ + rows * base.size_data_per_element_);
    const hnswlib::PayloadBuildLimits limits {100000, 1000, 1000, 10000};
    hnswlib::PayloadGraphBuilder builder(base, 8, 64, limits);
    Filter filter;
    for (size_t group = 0; group < 10; ++group) {
        std::vector<hnswlib::labeltype> labels;
        for (size_t row = group; row < rows; row += 10) {
            labels.push_back(base.getExternalLabel(row));
        }
        if (group == 3) {
            filter.labels.insert(labels.begin(), labels.end());
        }
        builder.add_block(labels);
        // Repeated/overlapping blocks must not duplicate the merged adjacency.
        builder.add_block(labels);
    }
    auto payload = builder.finish();
    payload.validate(rows);
    assert(!payload.links.empty());
    assert(std::equal(before.begin(), before.end(), base.data_level0_memory_));
    for (size_t row = 0; row < rows; ++row) {
        for (auto edge = payload.offsets[row]; edge < payload.offsets[row + 1]; ++edge) {
            assert(payload.links[edge] % 10 == row % 10);
        }
    }
    base.setEf(rows);
    const auto start = std::chrono::steady_clock::now();
    for (size_t query = 0; query < 100; ++query) {
        const auto *point = vectors.data() + query * dimension;
        auto actual = hnswlib::searchKnnWithPayload(base, payload, point, 10, &filter);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> exact;
        for (size_t row = 0; row < rows; ++row) {
            const auto label = base.getExternalLabel(row);
            if (filter(label)) {
                exact.emplace(space.get_dist_func()(point,
                                                    vectors.data() + row * dimension,
                                                    space.get_dist_func_param()),
                              label);
                if (exact.size() > 10) {
                    exact.pop();
                }
            }
        }
        assert(actual.size() == exact.size());
        while (!exact.empty()) {
            assert(actual.top() == exact.top());
            actual.pop();
            exact.pop();
        }
        auto ordinary = base.searchKnn(point, 10);
        auto unfiltered = hnswlib::searchKnnWithPayload(base, payload, point, 10, nullptr);
        assert(ordinary.size() == unfiltered.size());
        while (!ordinary.empty()) {
            assert(ordinary.top() == unfiltered.top());
            ordinary.pop();
            unfiltered.pop();
        }
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "100 filtered queries + exact oracle: " << elapsed
              << "s; payload edges: " << payload.links.size() << '\n';
    Filter empty;
    assert(hnswlib::searchKnnWithPayload(base, payload, vectors.data(), 10, &empty).empty());
    Filter singleton;
    singleton.labels.insert(200);
    const auto one = hnswlib::searchKnnWithPayload(base, payload, vectors.data(), 10, &singleton);
    assert(one.size() == 1 && one.top().second == 200);

    // Prove the search traverses payload edges and rejected navigation nodes.
    hnswlib::HierarchicalNSW<float> isolated(&space, 3, 2, 16, 1);
    for (size_t row = 0; row < 3; ++row) {
        isolated.addPoint(vectors.data() + row * dimension, row);
        isolated.setListCount(isolated.get_linklist0(row), 0);
    }
    for (size_t row = 0; row < 3; ++row) {
        isolated.setListCount(isolated.get_linklist0(row), 0);
    }
    isolated.maxlevel_ = 0;
    isolated.enterpoint_node_ = 0;
    hnswlib::PayloadGraph chain {{0, 1, 2, 2}, {1, 2}};
    chain.validate(3);
    Filter last;
    last.labels.insert(2);
    const auto reached = hnswlib::searchKnnWithPayload(isolated, chain, vectors.data(), 1, &last);
    assert(reached.size() == 1 && reached.top().second == 2);

    rejects([&] { builder.add_block({99999}); });
    rejects([&] { builder.add_block({2, 2}); });
    rejects([&] {
        hnswlib::PayloadGraphBuilder limited(base, 8, 64, {1, 1, 1000, 10000});
        limited.add_block({2, 200, 400, 600, 800, 1000});
    });
    rejects([&] {
        hnswlib::PayloadGraphBuilder limited(base, 8, 64, {1000, 100, 2, 2});
        limited.add_block({2, 4, 6});
    });
    rejects([] { hnswlib::PayloadGraph {{0, 2, 1}, {1}}.validate(2); });
    rejects([] { hnswlib::PayloadGraph {{0, 1, 1}, {0}}.validate(2); });
    rejects([] { hnswlib::PayloadGraph {{0, 2, 2}, {1, 1}}.validate(2); });
    rejects([] { hnswlib::PayloadGraph {{0, 1, 1}, {2}}.validate(2); });
    std::cout << "payload graph tests passed\n";
}
