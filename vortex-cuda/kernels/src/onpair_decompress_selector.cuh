// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include <stdint.h>

enum class OnPairDecompressKernel : uint8_t {
    Legacy,
    SixTpt,
    Cap12Lb5,
    Cap9Keep1Lb6,
    DirectHighLb5,
    DirectHighCgLb5,
    DirectHighKeep1Lb6,
    DirectHighKeep3Lb4,
};

struct OnPairDecompressStats {
    uint32_t code_bits;
    uint32_t dictionary_entries;
    uint32_t max_chunk_bytes_192;
    uint64_t token_count;
    uint64_t gt8_token_count;
};

// Data-only production candidate. The 12-bit branch must use the exact maximum
// for the launch's 192-code chunking; an average cannot prove compact-drain
// safety. The 16-bit branch uses token-weighted long-code frequency and
// dictionary footprint. Dataset-specific benchmark overrides remain explicit
// caller choices instead of being hidden in this function.
constexpr OnPairDecompressKernel select_onpair_decompress_kernel(const OnPairDecompressStats stats) {
    if (stats.token_count == 0u) {
        return OnPairDecompressKernel::Legacy;
    }
    if (stats.code_bits == 12u) {
        if (stats.max_chunk_bytes_192 <= 1728u) {
            return OnPairDecompressKernel::Cap9Keep1Lb6;
        }
        if (stats.max_chunk_bytes_192 <= 2304u) {
            return OnPairDecompressKernel::Cap12Lb5;
        }
        return OnPairDecompressKernel::SixTpt;
    }
    if (stats.code_bits == 16u) {
        const bool at_most_one_percent_long = stats.gt8_token_count <= stats.token_count / 100u;
        if (stats.dictionary_entries <= 384u && at_most_one_percent_long) {
            return OnPairDecompressKernel::DirectHighKeep1Lb6;
        }
        const bool large_split_dictionary = stats.dictionary_entries >= 32768u;
        const bool at_most_twenty_five_percent_long = stats.gt8_token_count <= stats.token_count / 4u;
        if (large_split_dictionary && at_most_twenty_five_percent_long) {
            return OnPairDecompressKernel::DirectHighCgLb5;
        }
        return OnPairDecompressKernel::DirectHighLb5;
    }
    return OnPairDecompressKernel::Legacy;
}

constexpr const char *onpair_decompress_kernel_symbol(const OnPairDecompressKernel kernel) {
    switch (kernel) {
    case OnPairDecompressKernel::Legacy:
        return "onpair_old_2";
    case OnPairDecompressKernel::SixTpt:
        return "onpair_decompress_6tpt";
    case OnPairDecompressKernel::Cap12Lb5:
        return "onpair_decompress_6tpt_cap12_lb5";
    case OnPairDecompressKernel::Cap9Keep1Lb6:
        return "onpair_decompress_6tpt_cap9_keep1_lb6";
    case OnPairDecompressKernel::DirectHighLb5:
        return "onpair_decompress_6tpt_directhi_lb5";
    case OnPairDecompressKernel::DirectHighCgLb5:
        return "onpair_decompress_6tpt_directhi_highcg_lb5";
    case OnPairDecompressKernel::DirectHighKeep1Lb6:
        return "onpair_decompress_6tpt_directhi_keep1_lb6";
    case OnPairDecompressKernel::DirectHighKeep3Lb4:
        return "onpair_decompress_6tpt_directhi_keep3_lb4";
    }
    return "onpair_old_2";
}

static_assert(select_onpair_decompress_kernel({12u, 4096u, 1728u, 192u, 0u}) ==
              OnPairDecompressKernel::Cap9Keep1Lb6);
static_assert(select_onpair_decompress_kernel({12u, 4096u, 2304u, 192u, 192u}) ==
              OnPairDecompressKernel::Cap12Lb5);
static_assert(select_onpair_decompress_kernel({12u, 4096u, 2305u, 192u, 192u}) ==
              OnPairDecompressKernel::SixTpt);
static_assert(select_onpair_decompress_kernel({16u, 384u, 3072u, 1000u, 10u}) ==
              OnPairDecompressKernel::DirectHighKeep1Lb6);
static_assert(select_onpair_decompress_kernel({16u, 385u, 3072u, 1000u, 0u}) ==
              OnPairDecompressKernel::DirectHighLb5);
static_assert(select_onpair_decompress_kernel({16u, 384u, 3072u, 1000u, 11u}) ==
              OnPairDecompressKernel::DirectHighLb5);
static_assert(select_onpair_decompress_kernel({16u, 65536u, 3072u, 1000u, 250u}) ==
              OnPairDecompressKernel::DirectHighCgLb5);
static_assert(select_onpair_decompress_kernel({16u, 65536u, 3072u, 1000u, 251u}) ==
              OnPairDecompressKernel::DirectHighLb5);
