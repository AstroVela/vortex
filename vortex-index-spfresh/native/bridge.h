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
