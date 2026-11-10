// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "rabitq.h"
#include "rabitq_io.hpp"
#include "rabitq_kernel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "rabitqlib/core/estimator.hpp"
#include "rabitqlib/core/rotator.hpp"
#include "rabitqlib/fastscan/encoder.hpp"
#include "rabitqlib/fastscan/estimator.hpp"
#include "rabitqlib/utils/cpu_features.hpp"

namespace {
constexpr char magic[8] = {'V', 'X', 'O', 'R', 'B', 'Q', '0', '4'};
template <class T>
T read(std::istream &in) {
    T value {};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Truncated official RaBitQ model");
    }
    return value;
}
template <class T>
std::vector<T> array(std::istream &in, size_t count) {
    std::vector<T> values(count);
    in.read(reinterpret_cast<char *>(values.data()), count * sizeof(T));
    if (!in) {
        throw std::runtime_error("Truncated official RaBitQ array");
    }
    return values;
}
template <class T>
void write(std::ostream &out, const T &value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}
template <class T>
void write_array(std::ostream &out, const std::vector<T> &values) {
    out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
}
template <class F>
int boundary(char *error, F &&function) {
    try {
        function();
        return 0;
    } catch (const std::exception &e) {
        std::snprintf(error, 1024, "%s", e.what());
    } catch (...) {
        std::snprintf(error, 1024, "Unknown official RaBitQ exception");
    }
    return -1;
}
void validate_vector(const float *values, uint32_t dimension) {
    double norm = 0;
    for (uint32_t i = 0; i < dimension; ++i) {
        const double value = values[i];
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite RaBitQ vector");
        }
        norm += value * value;
    }
    if (norm > double(std::numeric_limits<float>::max()) / 64) {
        throw std::runtime_error("RaBitQ vector norm exceeds safe Float32 encoding range");
    }
}
struct Model {
    uint32_t D = 0, N = 0, H = 0, P = 0, rotation_kind = 0;
    uint32_t bits = 0, centering = 0;
    uint64_t batches = 0, extra_stride = 0;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator;
    std::vector<float> centers;
    std::vector<uint64_t> batch_offsets;
    std::vector<uint32_t> offsets, ids, owner, position;
    std::vector<char> base, extra;
    std::vector<float> query;
    std::vector<float> scratch;
    rabitqlib::SplitBatchQuery<float> prepared;
    float (*extra_ip)(const float *, const uint8_t *, size_t) = nullptr;
    bool batch_extra = false;
    size_t batch_bytes = 0;
    Model(const char *path, uint32_t dimension, uint32_t rows, uint32_t heads) {
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) {
            throw std::runtime_error("Cannot open official model");
        }
        const auto file_size = uint64_t(in.tellg());
        in.seekg(0);
        const auto m = read<std::array<char, 8>>(in);
        if (std::memcmp(m.data(), magic, 8)) {
            throw std::runtime_error("Invalid official model magic");
        }
        D = read<uint32_t>(in);
        N = read<uint32_t>(in);
        H = read<uint32_t>(in);
        bits = read<uint32_t>(in);
        centering = read<uint32_t>(in);
        P = read<uint32_t>(in);
        rotation_kind = read<uint32_t>(in);
        if (!D || D > 4096 || !N || N > uint32_t(INT32_MAX) || !H || H > N || P < D || P > 4096 || P % 32 ||
            D != dimension || N != rows || H != heads || P != (D + 31) / 32 * 32 ||
            rotation_kind != uint32_t(D >= 64) || bits != 2 || centering != 1 || read<uint32_t>(in)) {
            throw std::runtime_error("Invalid official shape/options");
        }
        batches = read<uint64_t>(in);
        const auto rotation_bytes = read<uint64_t>(in);
        extra_stride = read<uint64_t>(in);
        const auto type =
            rotation_kind ? rabitqlib::RotatorType::FhtKacRotator : rabitqlib::RotatorType::MatrixRotator;
        rotator = rabitqlib::rotator_impl::make_rotator_for_loading(D, type, P);
        batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(P);
        if (rotation_bytes != rotator->dump_bytes() || !batches || batches > N ||
            extra_stride != (bits == 2 ? rabitqlib::ExDataMap<float>::data_bytes(P, 1) : 0)) {
            throw std::runtime_error("Invalid official layout");
        }
        const uint64_t expected_size = 64 + rotation_bytes + uint64_t(H) * P * 4 + uint64_t(H + 1) * 12 +
                                       uint64_t(N) * 4 + batches * (batch_bytes + 32 * extra_stride);
        if (expected_size != file_size) {
            throw std::runtime_error("Invalid official file size");
        }
        query.resize(P);
        scratch.resize(size_t(P) * 4);
        auto rotation = array<char>(in, rotation_bytes);
        if (!rotation_kind) {
            for (size_t i = 0; i < rotation.size(); i += 4) {
                float value;
                std::memcpy(&value, rotation.data() + i, 4);
                if (!std::isfinite(value)) {
                    throw std::runtime_error("Non-finite saved RaBitQ rotation");
                }
            }
        }
        rotator->load(rotation.data());
        centers = array<float>(in, size_t(H) * P);
        if (std::any_of(centers.begin(), centers.end(), [](float v) { return !std::isfinite(v); })) {
            throw std::runtime_error("Non-finite RaBitQ centers");
        }
        batch_offsets = array<uint64_t>(in, H + 1);
        offsets = array<uint32_t>(in, H + 1);
        ids = array<uint32_t>(in, N);
        base = array<char>(in, batches * batch_bytes);
        extra = array<char>(in, batches * 32 * extra_stride);
        for (uint64_t batch = 0; batch < batches; ++batch) {
            rabitqlib::ConstBatchDataMap<float> record(base.data() + batch * batch_bytes, P);
            for (uint32_t lane = 0; lane < 32; ++lane) {
                rabitqlib::ConstExDataMap<float> ex(extra.data() + (batch * 32 + lane) * extra_stride, P, 1);
                if (!std::isfinite(record.f_add()[lane]) || !std::isfinite(record.f_rescale()[lane]) ||
                    !std::isfinite(record.f_error()[lane]) || !std::isfinite(ex.f_add_ex()) ||
                    !std::isfinite(ex.f_rescale_ex())) {
                    throw std::runtime_error("Non-finite official RaBitQ factors");
                }
            }
        }
        if (in.peek() != std::char_traits<char>::eof() || offsets.front() || offsets.back() != N ||
            batch_offsets.front() || batch_offsets.back() != batches) {
            throw std::runtime_error("Invalid official model extents");
        }
        owner.assign(N, UINT32_MAX);
        position.resize(N);
        for (uint32_t h = 0; h < H; ++h) {
            if (offsets[h] >= offsets[h + 1] ||
                batch_offsets[h + 1] - batch_offsets[h] != (offsets[h + 1] - offsets[h] + 31) / 32) {
                throw std::runtime_error("Invalid official group");
            }
            for (uint32_t i = offsets[h]; i < offsets[h + 1]; ++i) {
                const auto id = ids[i];
                if (id >= N || owner[id] != UINT32_MAX) {
                    throw std::runtime_error("Invalid official ID mapping");
                }
                owner[id] = h;
                position[id] = i - offsets[h];
            }
        }
        if (bits == 2) {
            extra_ip = rabitqlib::select_excode_ipfunc(1);
#if defined(__x86_64__) || defined(__i386__)
            batch_extra = rabitqlib::cpu::has_avx2() && !rabitqlib::cpu::has_avx512_core();
#endif
        }
    }
    void prepare(const float *raw) {
        for (uint32_t j = 0; j < D; ++j) {
            if (!std::isfinite(raw[j])) {
                throw std::runtime_error("Non-finite official query");
            }
        }
        rotator->rotate(raw, query.data());
        prepared.reset(query.data(), P, bits - 1, scratch.data(), nullptr, rabitqlib::METRIC_L2, false);
    }
    void center_query(uint32_t h) {
        const auto norm =
            std::sqrt(rabitqlib::euclidean_sqr(query.data(), centers.data() + size_t(h) * P, P));
        prepared.set_g_add(norm);
    }
    void scores(const uint32_t *input, size_t count, double *output) {
        size_t cursor = 0;
        while (cursor < count) {
            if (input[cursor] >= N) {
                throw std::runtime_error("Invalid official candidate");
            }
            const auto h = owner[input[cursor]], n = offsets[h + 1] - offsets[h];
            if (cursor + n > count ||
                !std::equal(input + cursor, input + cursor + n, ids.data() + offsets[h])) {
                throw std::runtime_error("Official scan expects complete ordered posting groups");
            }
            center_query(h);
            for (uint32_t first = 0; first < n; first += 32) {
                const auto batch = batch_offsets[h] + first / 32;
                const auto logical = std::min<uint32_t>(32, n - first);
                std::array<float, 32> est {}, lower {}, products {};
                rabitqlib::split_batch_estdist(base.data() + batch * batch_bytes,
                                               prepared,
                                               P,
                                               est.data(),
                                               lower.data(),
                                               products.data(),
                                               false);
                std::array<float, 32> extra_products;
                const auto *extra_records = bits == 2 ? extra.data() + batch * 32 * extra_stride : nullptr;
#if defined(__x86_64__) || defined(__i386__)
                if (batch_extra) {
                    extra_inner_products(prepared.rotated_query(),
                                         extra_records,
                                         extra_stride,
                                         P,
                                         logical,
                                         extra_products.data());
                }
#endif
                for (uint32_t lane = 0; lane < logical; ++lane) {
                    auto value = est[lane];
                    if (bits == 2 && batch_extra) {
                        rabitqlib::ConstExDataMap<float> record(extra_records + lane * extra_stride, P, 1);
                        value = record.f_add_ex() + prepared.g_add() +
                                record.f_rescale_ex() *
                                    (2.0F * products[lane] + extra_products[lane] + prepared.g_kbxsumq());
                    } else if (bits == 2) {
                        value = rabitqlib::split_distance_boosting(extra_records + lane * extra_stride,
                                                                   extra_ip,
                                                                   prepared,
                                                                   P,
                                                                   1,
                                                                   products[lane]);
                    }
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("Non-finite official estimate");
                    }
                    output[cursor + first + lane] = value;
                }
            }
            cursor += n;
        }
    }
    float reference(uint32_t id) {
        if (id >= N) {
            throw std::runtime_error("Invalid reference ID");
        }
        const auto h = owner[id], p = position[id];
        const auto batch = batch_offsets[h] + p / 32;
        const auto lane = p % 32;
        center_query(h);
        rabitqlib::ConstBatchDataMap<float> record(base.data() + batch * batch_bytes, P);
        std::vector<uint8_t> raw(32 * P / 8);
        rabitqlib::fastscan::unpack_codes(P, record.bin_code(), 32, raw.data());
        uint32_t sum = 0;
        const auto *code = raw.data() + lane * (P / 8);
        const auto *lut = prepared.lut();
        for (uint32_t j = 0; j < P / 8; ++j) {
            sum += lut[(2 * j) * 16 + (code[j] >> 4)] + lut[(2 * j + 1) * 16 + (code[j] & 15)];
        }
        const float ip = prepared.delta() * float(sum) + prepared.sum_vl_lut();
        if (bits == 2) {
            return rabitqlib::split_distance_boosting(extra.data() + (batch * 32 + lane) * extra_stride,
                                                      extra_ip,
                                                      prepared,
                                                      P,
                                                      1,
                                                      ip);
        }
        return record.f_add()[lane] + prepared.g_add() +
               record.f_rescale()[lane] * (ip + prepared.g_k1xsumq());
    }
};

void build(const char *destination,
           const float *vectors,
           uint32_t D,
           uint32_t N,
           uint32_t H,
           const uint32_t *offsets,
           const uint32_t *ids,
           uint32_t seed) {
    if (!D || D > 4096 || !N || N > uint32_t(INT32_MAX) || !H || H > N || offsets[0] || offsets[H] != N) {
        throw std::runtime_error("Invalid official RaBitQ build shape");
    }
    std::vector<uint8_t> seen(N);
    for (uint32_t h = 0; h < H; ++h) {
        if (offsets[h] >= offsets[h + 1] || offsets[h + 1] > N) {
            throw std::runtime_error("Invalid official RaBitQ build group");
        }
        for (uint32_t i = offsets[h]; i < offsets[h + 1]; ++i) {
            if (ids[i] >= N || seen[ids[i]]++) {
                throw std::runtime_error("Official RaBitQ requires replica-1 routing");
            }
            validate_vector(vectors + size_t(ids[i]) * D, D);
        }
    }
    const uint32_t P = (D + 31) / 32 * 32;
    const uint32_t rotation_kind = D >= 64 ? 1 : 0;
    const auto type =
        rotation_kind ? rabitqlib::RotatorType::FhtKacRotator : rabitqlib::RotatorType::MatrixRotator;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator(rabitqlib::choose_rotator<float>(D, type, P, seed));
    std::vector<float> centers(size_t(H) * P);
    std::vector<uint64_t> batch_offsets {0};
    for (uint32_t h = 0; h < H; ++h) {
        std::vector<double> mean(D);
        for (uint32_t i = offsets[h]; i < offsets[h + 1]; ++i) {
            for (uint32_t j = 0; j < D; ++j) {
                mean[j] += vectors[size_t(ids[i]) * D + j];
            }
        }
        std::vector<float> raw(D);
        for (uint32_t j = 0; j < D; ++j) {
            raw[j] = float(mean[j] / (offsets[h + 1] - offsets[h]));
        }
        rotator->rotate(raw.data(), centers.data() + size_t(h) * P);
        batch_offsets.push_back(batch_offsets.back() + (offsets[h + 1] - offsets[h] + 31) / 32);
    }
    const auto batches = batch_offsets.back();
    const auto batch_bytes = rabitqlib::BatchDataMap<float>::data_bytes(P);
    const auto stride = rabitqlib::ExDataMap<float>::data_bytes(P, 1);
    std::vector<char> base(batches * batch_bytes), extra(batches * 32 * stride);
    for (uint32_t h = 0; h < H; ++h) {
        const auto n = offsets[h + 1] - offsets[h];
        for (uint32_t first = 0; first < n; first += 32) {
            const auto logical = std::min<uint32_t>(32, n - first);
            const auto batch = batch_offsets[h] + first / 32;
            std::vector<float> rotated(size_t(32) * P);
            for (uint32_t i = 0; i < logical; ++i) {
                rotator->rotate(vectors + size_t(ids[offsets[h] + first + i]) * D,
                                rotated.data() + size_t(i) * P);
            }
            rabitqlib::quant::quantize_split_batch(rotated.data(),
                                                   centers.data() + size_t(h) * P,
                                                   logical,
                                                   P,
                                                   1,
                                                   base.data() + batch * batch_bytes,
                                                   extra.data() + batch * 32 * stride);
        }
    }
    if (std::filesystem::exists(destination)) {
        throw std::runtime_error("Refusing to overwrite official RaBitQ model");
    }
    std::vector<char> rotation(rotator->dump_bytes());
    rotator->save(rotation.data());
    std::ofstream out(destination, std::ios::binary);
    out.write(magic, 8);
    for (auto value : {D, N, H, uint32_t(2), uint32_t(1), P, rotation_kind, uint32_t(0)}) {
        write(out, value);
    }
    write(out, batches);
    write(out, uint64_t(rotation.size()));
    write(out, uint64_t(stride));
    write_array(out, rotation);
    write_array(out, centers);
    write_array(out, batch_offsets);
    out.write(reinterpret_cast<const char *>(offsets), size_t(H + 1) * 4);
    out.write(reinterpret_cast<const char *>(ids), size_t(N) * 4);
    write_array(out, base);
    write_array(out, extra);
    out.flush();
    if (!out) {
        throw std::runtime_error("Cannot write official RaBitQ model");
    }
    Model check(destination, D, N, H);
}

struct Hit {
    uint32_t id;
    double distance;
};
struct State {
    Model model;
    uint32_t dimension;
    std::vector<uint64_t> locations;
    std::vector<std::array<uint64_t, 2>> rows;
    vortex_rabitq::OriginalReader originals;
    bool diagnostics;
    State(const char *path,
          const char *root,
          uint32_t D,
          uint32_t N,
          uint32_t H,
          const uint32_t *offsets,
          const uint32_t *ids,
          const uint64_t *positions,
          const uint64_t *row_pairs,
          bool diagnostic)
        : model(path, D, N, H), dimension(D), locations(positions, positions + N), rows(N),
          originals(std::filesystem::path(root)), diagnostics(diagnostic) {
        if (!std::equal(model.offsets.begin(), model.offsets.end(), offsets) ||
            !std::equal(model.ids.begin(), model.ids.end(), ids)) {
            throw std::runtime_error("Official RaBitQ routing differs from native files");
        }
        for (uint32_t id = 0; id < N; ++id) {
            rows[id] = {row_pairs[size_t(id) * 2], row_pairs[size_t(id) * 2 + 1]};
            const auto head = locations[id] >> 63;
            const auto offset = locations[id] & ~(uint64_t(1) << 63);
            const auto length = originals.length(uint32_t(head));
            if (offset % 4 || offset > length || uint64_t(D) * 4 > length - offset) {
                throw std::runtime_error("Invalid original Float32 vector location");
            }
        }
    }
    bool less(Hit left, Hit right) const {
        return left.distance != right.distance ? left.distance < right.distance
                                               : rows[left.id] < rows[right.id];
    }
    std::vector<uint32_t> candidates(const uint32_t *heads, uint32_t count) const {
        if (count > model.H) {
            throw std::runtime_error("Too many RaBitQ routing heads");
        }
        std::vector<uint8_t> seen(model.H);
        std::vector<uint32_t> result;
        for (uint32_t h = 0; h < count; ++h) {
            const auto head = heads[h];
            if (head >= model.H || seen[head]++) {
                throw std::runtime_error("Invalid or duplicate official RaBitQ routing head");
            }
            result.insert(result.end(),
                          model.ids.begin() + model.offsets[head],
                          model.ids.begin() + model.offsets[head + 1]);
        }
        return result;
    }
    float exact(size_t batch_index,
                const std::vector<double> &query,
                std::vector<float> &vector,
                uint64_t &fallbacks) const {
        originals.copy(batch_index, vector.data(), size_t(dimension) * 4);
        std::array<double, 4> sums {};
        uint32_t j = 0;
        for (; j + 4 <= dimension; j += 4) {
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const double delta = double(vector[j + lane]) - query[j + lane];
                sums[lane] += delta * delta;
            }
        }
        double sum = -0.0;
        for (double part : sums) {
            sum += part;
        }
        for (; j < dimension; ++j) {
            const double delta = double(vector[j]) - query[j];
            sum += delta * delta;
        }
        // Match the scalar Float64 API after rounding, including halfway cases.
        const double slack = 4.0 * dimension * std::numeric_limits<double>::epsilon() * sum;
        const float lower = float(sum - slack), upper = float(sum + slack);
        if (lower == upper) {
            return lower;
        }
        ++fallbacks;
        sum = -0.0;
        for (uint32_t i = 0; i < dimension; ++i) {
            const double delta = double(vector[i]) - query[i];
            sum += delta * delta;
        }
        return float(sum);
    }
};

uint64_t now_ns(bool enabled) {
    return enabled ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count())
                   : 0;
}
template <class F>
uint64_t fingerprint(size_t count, F id) {
    uint64_t value = 14695981039346656037ULL;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t current = id(i);
        for (uint32_t shift = 0; shift < 32; shift += 8) {
            value = (value ^ uint8_t(current >> shift)) * 1099511628211ULL;
        }
    }
    return value;
}
} // namespace

extern "C" int vortex_rabitq_build(const char *destination,
                                   const float *vectors,
                                   uint32_t dimension,
                                   uint32_t rows,
                                   uint32_t heads,
                                   const uint32_t *offsets,
                                   const uint32_t *ids,
                                   uint32_t seed,
                                   char *error) {
    return boundary(error, [&] { build(destination, vectors, dimension, rows, heads, offsets, ids, seed); });
}
extern "C" int vortex_rabitq_open(const char *model,
                                  const char *native_root,
                                  uint32_t dimension,
                                  uint32_t rows,
                                  uint32_t heads,
                                  const uint32_t *offsets,
                                  const uint32_t *ids,
                                  const uint64_t *locations,
                                  const uint64_t *row_pairs,
                                  uint32_t diagnostics,
                                  void **handle,
                                  char *error) {
    *handle = nullptr;
    return boundary(error, [&] {
        if (diagnostics > 1) {
            throw std::runtime_error("Invalid official RaBitQ diagnostic flag");
        }
        *handle = new State(model,
                            native_root,
                            dimension,
                            rows,
                            heads,
                            offsets,
                            ids,
                            locations,
                            row_pairs,
                            diagnostics != 0);
    });
}
extern "C" void vortex_rabitq_close(void *handle) {
    delete static_cast<State *>(handle);
}
extern "C" int vortex_rabitq_search(void *handle,
                                    const float *query,
                                    const uint32_t *heads,
                                    uint32_t head_count,
                                    uint32_t k,
                                    uint32_t rerank,
                                    const uint8_t *allowed,
                                    uint32_t allowed_bytes,
                                    uint32_t *output,
                                    float *distances,
                                    uint32_t *count,
                                    uint64_t *stats,
                                    char *error) {
    return boundary(error, [&] {
        auto &state = *static_cast<State *>(handle);
        auto &model = state.model;
        if (!k || k > model.N || rerank < k || rerank > model.N || head_count > model.H ||
            (allowed ? allowed_bytes != (model.N + 7) / 8 : allowed_bytes != 0) ||
            (allowed && model.N % 8 && (allowed[allowed_bytes - 1] >> (model.N % 8)))) {
            throw std::runtime_error("Invalid official RaBitQ query options");
        }
        validate_vector(query, model.D);
        *count = 0;
        std::fill(stats, stats + 12, uint64_t(0));
        auto ids = state.candidates(heads, head_count);
        stats[0] = ids.size();
        if (ids.empty()) {
            return;
        }
        if (state.diagnostics) {
            stats[6] = fingerprint(ids.size(), [&](size_t i) { return ids[i]; });
        }
        const auto estimate_begin = now_ns(state.diagnostics);
        model.prepare(query);
        std::vector<double> estimates(ids.size());
        model.scores(ids.data(), ids.size(), estimates.data());
        const auto select_begin = now_ns(state.diagnostics);
        stats[9] = select_begin - estimate_begin;
        std::vector<Hit> candidates;
        candidates.reserve(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!allowed || (allowed[ids[i] / 8] & (1u << (ids[i] % 8)))) {
                candidates.push_back({ids[i], estimates[i]});
            }
        }
        const auto less = [&](Hit left, Hit right) {
            return state.less(left, right);
        };
        rerank = std::min<uint32_t>(rerank, uint32_t(candidates.size()));
        if (rerank < candidates.size()) {
            std::nth_element(candidates.begin(), candidates.begin() + rerank, candidates.end(), less);
        }
        candidates.resize(rerank);
        stats[1] = stats[2] = rerank;
        if (state.diagnostics) {
            stats[7] = fingerprint(candidates.size(), [&](size_t i) { return candidates[i].id; });
        }
        stats[10] = now_ns(state.diagnostics) - select_begin;
        std::vector<double> exact_query(query, query + model.D);
        std::vector<float> vector(model.D);
        vortex_rabitq::IoStats io;
        for (size_t first = 0; first < candidates.size(); first += vortex_rabitq::rerank_batch) {
            const auto count = std::min(vortex_rabitq::rerank_batch, candidates.size() - first);
            std::vector<uint64_t> positions;
            positions.reserve(count);
            for (size_t i = 0; i < count; ++i) {
                positions.push_back(state.locations[candidates[first + i].id]);
            }
            const auto read_begin = now_ns(state.diagnostics);
            state.originals.prepare(positions, size_t(model.D) * sizeof(float), io);
            const auto exact_begin = now_ns(state.diagnostics);
            stats[8] += exact_begin - read_begin;
            for (size_t i = 0; i < count; ++i) {
                auto &candidate = candidates[first + i];
                candidate.distance = state.exact(i, exact_query, vector, stats[3]);
                if (!std::isfinite(candidate.distance) || candidate.distance < 0) {
                    throw std::runtime_error("Non-finite official RaBitQ exact distance");
                }
            }
            stats[11] += now_ns(state.diagnostics) - exact_begin;
        }
        stats[4] = io.requests;
        stats[5] = io.bytes;
        std::sort(candidates.begin(), candidates.end(), less);
        *count = std::min(k, rerank);
        for (uint32_t i = 0; i < *count; ++i) {
            output[i] = candidates[i].id;
            distances[i] = float(candidates[i].distance);
        }
    });
}

extern "C" int vortex_rabitq_estimates(void *handle,
                                       const float *query,
                                       const uint32_t *heads,
                                       uint32_t head_count,
                                       uint32_t capacity,
                                       uint32_t *output,
                                       double *scores,
                                       uint32_t scalar_reference,
                                       uint32_t *count,
                                       char *error) {
    return boundary(error, [&] {
        auto &state = *static_cast<State *>(handle);
        validate_vector(query, state.dimension);
        auto ids = state.candidates(heads, head_count);
        if (ids.size() > capacity || scalar_reference > 2) {
            throw std::runtime_error("Invalid RaBitQ estimate output capacity");
        }
        state.model.prepare(query);
        if (scalar_reference == 1) {
            for (size_t i = 0; i < ids.size(); ++i) {
                scores[i] = state.model.reference(ids[i]);
            }
        } else if (scalar_reference == 2) {
            const bool optimized = state.model.batch_extra;
            state.model.batch_extra = false;
            try {
                state.model.scores(ids.data(), ids.size(), scores);
            } catch (...) {
                state.model.batch_extra = optimized;
                throw;
            }
            state.model.batch_extra = optimized;
        } else {
            state.model.scores(ids.data(), ids.size(), scores);
        }
        std::copy(ids.begin(), ids.end(), output);
        *count = uint32_t(ids.size());
    });
}
