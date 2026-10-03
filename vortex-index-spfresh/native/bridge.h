// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <stdint.h>

// Buffers and handle are caller-owned. Errors are NUL-terminated in 1024 bytes.
// Only trusted, validated static Float32/L2 files in the closed bundle are accepted.
// Calls are serialized; the caller must not close a handle still in use.
#ifdef __cplusplus
extern "C" {
#endif

// Build into a new, private directory. Input remains live and immutable until return.
int vortex_spfresh_build(const char *root,
                         const float *vectors,
                         uint32_t dimension,
                         uint32_t rows,
                         uint32_t heads,
                         uint32_t posting_pages,
                         uint32_t replicas,
                         char *error);
int vortex_spfresh_open(const char *root,
                        uint32_t dimension,
                        uint32_t rows,
                        uint32_t posting_pages,
                        void **handle,
                        char *error);
int vortex_spfresh_search(void *handle,
                          const float *queries,
                          uint32_t dimension,
                          uint32_t count,
                          uint32_t k,
                          uint32_t max_check,
                          uint32_t internal_results,
                          uint32_t search_pages,
                          int32_t *ids,
                          float *distances,
                          char *error);
void vortex_spfresh_close(void *handle);
#ifdef __cplusplus
}
#endif

#if defined(VORTEX_SPFRESH_BENCHMARK) && defined(__cplusplus)
namespace SPTAG::SPANN {
template <typename T>
class Index;
}
// Benchmark-only access to an opened handle; never built into the production bridge.
SPTAG::SPANN::Index<float> &vortex_spfresh_benchmark_index(void *handle);
#endif

#if defined(VORTEX_SPFRESH_TESTING) && defined(__cplusplus)
struct SpFreshWorkspaceStats {
    const void *postings = nullptr;
    const void *heads = nullptr;
    uint32_t internal_results = 0;
    uint32_t check_capacity = 0;
    uint32_t head_check_capacity = 0;
    uint32_t posting_buffer_bytes = 0;
    int live_posting_workspaces = 0;
    bool thread_detached = false;
};
// Test-only inspection; never built into the production bridge.
SpFreshWorkspaceStats vortex_spfresh_test_workspaces(void *handle);
#endif
