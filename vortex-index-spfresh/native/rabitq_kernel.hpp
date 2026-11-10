// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

constexpr auto make_extra_masks() {
    std::array<std::array<uint32_t, 8>, 256> result {};
    for (uint32_t byte = 0; byte < 256; ++byte) {
        for (uint32_t bit = 0; bit < 8; ++bit) {
            result[byte][bit] = byte & (1u << bit) ? UINT32_MAX : 0;
        }
    }
    return result;
}
alignas(32) static constexpr auto extra_masks = make_extra_masks();

// Preserve the official AVX2 kernel's eight accumulation chains and reduction tree.
// Four independent records share each query load and avoid per-record dispatch.
__attribute__((target("avx2"))) inline void extra_inner_products(const float *query,
                                                                 const char *records,
                                                                 size_t stride,
                                                                 size_t padded_dim,
                                                                 uint32_t count,
                                                                 float *output) {
    uint32_t first = 0;
    for (; first + 4 <= count; first += 4) {
        __m256 sums[4] {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
#pragma GCC unroll 1
        for (uint32_t column = 0; column < padded_dim / 8; ++column) {
            const auto q = _mm256_loadu_ps(query + column * 8);
#pragma GCC unroll 4
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const auto byte = static_cast<uint8_t>(records[(first + lane) * stride + column]);
                const auto mask = _mm256_castsi256_ps(
                    _mm256_load_si256(reinterpret_cast<const __m256i *>(extra_masks[byte].data())));
                sums[lane] = _mm256_add_ps(sums[lane], _mm256_and_ps(q, mask));
            }
        }
        for (uint32_t lane = 0; lane < 4; ++lane) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(sums[lane]), _mm256_extractf128_ps(sums[lane], 1));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            output[first + lane] = _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
        }
    }
    for (; first < count; ++first) {
        auto sum = _mm256_setzero_ps();
        for (uint32_t column = 0; column < padded_dim / 8; ++column) {
            const auto q = _mm256_loadu_ps(query + column * 8);
            const auto byte = static_cast<uint8_t>(records[first * stride + column]);
            const auto mask = _mm256_castsi256_ps(
                _mm256_load_si256(reinterpret_cast<const __m256i *>(extra_masks[byte].data())));
            sum = _mm256_add_ps(sum, _mm256_and_ps(q, mask));
        }
        auto reduced = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        reduced = _mm_add_ps(reduced, _mm_movehl_ps(reduced, reduced));
        output[first] = _mm_cvtss_f32(_mm_add_ss(reduced, _mm_movehdup_ps(reduced)));
    }
}
#endif
