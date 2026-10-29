// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <hnswlib/hnswlib.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace hnswlib {

// An immutable overlay on level zero, addressed by INTERNAL node IDs. The base
// graph and its labels must remain unchanged for the lifetime of this graph.
struct PayloadGraph {
    std::vector<uint64_t> offsets;
    std::vector<tableint> links;

    void validate(size_t rows) const {
        if (offsets.size() != rows + 1 || offsets.front() != 0 || offsets.back() != links.size()) {
            throw std::invalid_argument("Invalid payload CSR shape");
        }
        for (size_t row = 0; row < rows; ++row) {
            if (offsets[row] > offsets[row + 1] || offsets[row + 1] > links.size()) {
                throw std::invalid_argument("Invalid payload CSR offsets");
            }
            tableint previous = 0;
            bool first = true;
            for (uint64_t edge = offsets[row]; edge < offsets[row + 1]; ++edge) {
                const auto target = links[edge];
                if (target >= rows || target == row || (!first && target <= previous)) {
                    throw std::invalid_argument("Invalid payload CSR neighbor");
                }
                previous = target;
                first = false;
            }
        }
    }
};

struct PayloadBuildLimits {
    uint64_t max_edges;
    uint32_t max_degree;
    uint32_t max_block_rows;
    uint64_t max_memberships;
};

// Builds true single-level NSW graphs inside attribute blocks. It borrows the
// base vectors, does not call addPoint (level=0 there still draws random levels),
// and never changes the base adjacency. No predicate combinations are built.
class PayloadGraphBuilder {
    using Candidate = std::pair<float, tableint>;
    using Nearest = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;
    using Farthest = std::priority_queue<Candidate>;

public:
    PayloadGraphBuilder(const HierarchicalNSW<float> &base,
                        uint32_t m,
                        uint32_t ef,
                        PayloadBuildLimits limits)
        : base_(base), m_(m), ef_(ef), limits_(limits), edges_(base.cur_element_count) {
        if (base.num_deleted_ != 0 || m == 0 || m > 64 || ef < m || ef > 4096 || limits.max_degree == 0 ||
            limits.max_edges == 0 || limits.max_block_rows < 2 || limits.max_memberships == 0) {
            throw std::invalid_argument("Invalid payload construction parameters");
        }
    }

    // Labels may be in any order, but each block must contain distinct labels.
    // Blocks may overlap. A failed build must be discarded, never published.
    void add_block(const std::vector<labeltype> &labels) {
        if (labels.size() > limits_.max_block_rows ||
            labels.size() > limits_.max_memberships - memberships_) {
            throw std::runtime_error("Payload block/membership limit exceeded");
        }
        memberships_ += labels.size();
        std::vector<tableint> ids;
        ids.reserve(labels.size());
        for (auto label : labels) {
            const auto found = base_.label_lookup_.find(label);
            if (found == base_.label_lookup_.end()) {
                throw std::invalid_argument("Unknown payload label");
            }
            ids.push_back(found->second);
        }
        std::sort(ids.begin(), ids.end());
        if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
            throw std::invalid_argument("Duplicate payload label");
        }
        std::vector<std::vector<tableint>> local(ids.size());
        std::vector<size_t> visited(ids.size(), 0);
        for (tableint next = 1; next < ids.size(); ++next) {
            Nearest pending;
            Farthest best;
            const Candidate entry {distance(ids[next], ids[0]), 0};
            pending.push(entry);
            best.push(entry);
            visited[0] = next;
            while (!pending.empty()) {
                const auto current = pending.top();
                pending.pop();
                if (best.size() >= ef_ && current.first > best.top().first) {
                    break;
                }
                for (auto neighbor : local[current.second]) {
                    if (visited[neighbor] == next) {
                        continue;
                    }
                    visited[neighbor] = next;
                    const Candidate candidate {distance(ids[next], ids[neighbor]), neighbor};
                    if (best.size() < ef_ || candidate < best.top()) {
                        pending.push(candidate);
                        best.push(candidate);
                        if (best.size() > ef_) {
                            best.pop();
                        }
                    }
                }
            }
            std::vector<tableint> candidates;
            while (!best.empty()) {
                candidates.push_back(best.top().second);
                best.pop();
            }
            local[next] = select(next, candidates, ids, m_);
            for (auto neighbor : local[next]) {
                local[neighbor].push_back(next);
                if (local[neighbor].size() > 2 * m_) {
                    local[neighbor] = select(neighbor, local[neighbor], ids, 2 * m_);
                }
            }
        }
        for (size_t row = 0; row < ids.size(); ++row) {
            auto merged = edges_[ids[row]];
            auto *base_links = base_.get_linklist0(ids[row]);
            const auto *neighbors = reinterpret_cast<const tableint *>(base_links + 1);
            const auto count = base_.getListCount(base_links);
            for (auto neighbor : local[row]) {
                const auto target = ids[neighbor];
                if (std::find(neighbors, neighbors + count, target) == neighbors + count) {
                    merged.push_back(target);
                }
            }
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            const auto added = merged.size() - edges_[ids[row]].size();
            if (merged.size() > limits_.max_degree || added > limits_.max_edges - edge_count_) {
                throw std::runtime_error("Payload degree/edge limit exceeded");
            }
            edge_count_ += added;
            edges_[ids[row]] = std::move(merged);
        }
    }

    PayloadGraph finish() const {
        PayloadGraph graph;
        graph.offsets.reserve(edges_.size() + 1);
        graph.links.reserve(edge_count_);
        graph.offsets.push_back(0);
        for (const auto &neighbors : edges_) {
            graph.links.insert(graph.links.end(), neighbors.begin(), neighbors.end());
            graph.offsets.push_back(graph.links.size());
        }
        return graph;
    }

private:
    float distance(tableint left, tableint right) const {
        const auto value = base_.fstdistfunc_(base_.getDataByInternalId(left),
                                              base_.getDataByInternalId(right),
                                              base_.dist_func_param_);
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite payload build distance");
        }
        return value;
    }

    std::vector<tableint> select(tableint center,
                                 const std::vector<tableint> &candidates,
                                 const std::vector<tableint> &ids,
                                 size_t capacity) const {
        std::vector<Candidate> ordered;
        for (auto candidate : candidates) {
            ordered.emplace_back(distance(ids[center], ids[candidate]), candidate);
        }
        std::sort(ordered.begin(), ordered.end());
        std::vector<tableint> selected;
        for (const auto &candidate : ordered) {
            bool diverse = true;
            for (auto previous : selected) {
                if (distance(ids[candidate.second], ids[previous]) < candidate.first) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                selected.push_back(candidate.second);
                if (selected.size() == capacity) {
                    break;
                }
            }
        }
        return selected;
    }

    const HierarchicalNSW<float> &base_;
    uint32_t m_;
    uint32_t ef_;
    PayloadBuildLimits limits_;
    std::vector<std::vector<tableint>> edges_;
    uint64_t edge_count_ = 0;
    uint64_t memberships_ = 0;
};

// Same filter semantics as upstream: rejected nodes can navigate, only matching
// labels enter the result heap. Extra links participate only in level zero.
// The caller validates the CSR once and freezes both graphs before searching.
inline std::priority_queue<std::pair<float, labeltype>>
searchKnnWithPayload(const HierarchicalNSW<float> &base,
                     const PayloadGraph &payload,
                     const void *query,
                     size_t k,
                     BaseFilterFunctor *filter) {
    if (filter == nullptr) {
        return base.searchKnn(query, k);
    }
    std::priority_queue<std::pair<float, labeltype>> result;
    if (k == 0 || base.cur_element_count == 0) {
        return result;
    }
    const auto distance = [&](tableint id) {
        const auto value = base.fstdistfunc_(query, base.getDataByInternalId(id), base.dist_func_param_);
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite filtered search distance");
        }
        return value;
    };
    auto entry = base.enterpoint_node_;
    auto nearest = distance(entry);
    for (int level = base.maxlevel_; level > 0; --level) {
        bool improved = true;
        while (improved) {
            improved = false;
            auto *links = base.get_linklist(entry, level);
            const auto *neighbors = reinterpret_cast<const tableint *>(links + 1);
            for (size_t offset = 0; offset < base.getListCount(links); ++offset) {
                const auto candidate = distance(neighbors[offset]);
                if (candidate < nearest) {
                    nearest = candidate;
                    entry = neighbors[offset];
                    improved = true;
                }
            }
        }
    }
    const auto release = [&](VisitedList *list) {
        base.visited_list_pool_->releaseVisitedList(list);
    };
    std::unique_ptr<VisitedList, decltype(release)> visited(base.visited_list_pool_->getFreeVisitedList(),
                                                            release);
    auto *marks = visited->mass;
    const auto tag = visited->curV;
    using Candidate = std::pair<float, tableint>;
    std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> pending;
    std::priority_queue<Candidate> best;
    const auto allowed = [&](tableint id) {
        return !base.isMarkedDeleted(id) && (*filter)(base.getExternalLabel(id));
    };
    pending.emplace(nearest, entry);
    marks[entry] = tag;
    if (allowed(entry)) {
        best.emplace(nearest, entry);
    }
    const auto ef = std::max(k, base.ef_);
    const auto visit = [&](tableint neighbor) {
        if (marks[neighbor] == tag) {
            return;
        }
        marks[neighbor] = tag;
        const Candidate candidate {distance(neighbor), neighbor};
        if (best.size() < ef || candidate < best.top()) {
            pending.push(candidate);
            if (allowed(neighbor)) {
                best.push(candidate);
                if (best.size() > ef) {
                    best.pop();
                }
            }
        }
    };
    while (!pending.empty()) {
        const auto current = pending.top();
        pending.pop();
        if (best.size() >= ef && current.first > best.top().first) {
            break;
        }
        auto *links = base.get_linklist0(current.second);
        const auto *neighbors = reinterpret_cast<const tableint *>(links + 1);
        for (size_t offset = 0; offset < base.getListCount(links); ++offset) {
            visit(neighbors[offset]);
        }
        for (auto offset = payload.offsets[current.second]; offset < payload.offsets[current.second + 1];
             ++offset) {
            visit(payload.links[offset]);
        }
    }
    while (best.size() > k) {
        best.pop();
    }
    while (!best.empty()) {
        result.emplace(best.top().first, base.getExternalLabel(best.top().second));
        best.pop();
    }
    return result;
}

} // namespace hnswlib
