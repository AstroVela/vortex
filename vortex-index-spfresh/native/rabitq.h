// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once
#include <stdint.h>

// All buffers live through the synchronous call. Error buffers contain 1024 bytes.
// Group offsets have heads+1 entries; IDs, locations and row pairs have rows entries.
// Each group is nonempty, begins with its head, and the groups partition all rows.
// The caller serializes operations on each handle and retains an immutable native bundle.
// Original vectors are read synchronously with buffered pread.
// Diagnostics: 0=off, 1=record hashes/timings. Search stats has 12 entries:
// scanned, retained, reranked, exact rounding fallbacks, read requests, aligned bytes,
// candidate hash, shortlist hash,
// prepare/read ns, estimate ns, shortlist ns, copy/exact ns. Hashes and times are zero when off.
#ifdef __cplusplus
extern "C" {
#endif
int vortex_rabitq_build(const char *destination,
                        const float *vectors,
                        uint32_t dimension,
                        uint32_t rows,
                        uint32_t heads,
                        const uint32_t *offsets,
                        const uint32_t *ids,
                        uint32_t seed,
                        char *error);
int vortex_rabitq_open(const char *model,
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
                       char *error);
int vortex_rabitq_search(void *handle,
                         const float *query,
                         const uint32_t *heads,
                         uint32_t head_count,
                         uint32_t k,
                         uint32_t rerank,
                         const uint8_t *allowed,
                         uint32_t allowed_bytes,
                         uint32_t *ids,
                         float *distances,
                         uint32_t *count,
                         uint64_t *stats,
                         char *error);
void vortex_rabitq_close(void *handle);
// Bounded scoring: reference 0=optimized, 1=scalar LUT, 2=unmodified official FastScan.
int vortex_rabitq_estimates(void *handle,
                            const float *query,
                            const uint32_t *heads,
                            uint32_t head_count,
                            uint32_t capacity,
                            uint32_t *ids,
                            double *scores,
                            uint32_t scalar_reference,
                            uint32_t *count,
                            char *error);
#ifdef __cplusplus
}
#endif
