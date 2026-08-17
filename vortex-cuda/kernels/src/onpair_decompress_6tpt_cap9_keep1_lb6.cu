// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// The caller must prove every 192-token output chunk is at most 1,728 bytes
// and select the general cap-12 kernel otherwise.
#define TOKENS_PER_THREAD          6u
#define ONPAIR_BUF_BYTES_PER_TOKEN 9u
#define ONPAIR_KEEP_LO_COUNT       1
#define ONPAIR_ASSUME_BUFFER_FITS
#define ONPAIR_KERNEL_NAME   onpair_decompress_6tpt_cap9_keep1_lb6
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 6)
#include "onpair_decompress_tpt.cuh"
