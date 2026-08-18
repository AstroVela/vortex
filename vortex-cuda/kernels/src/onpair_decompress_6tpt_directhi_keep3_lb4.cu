// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#define TOKENS_PER_THREAD          6u
#define ONPAIR_BUF_BYTES_PER_TOKEN 16u
#define ONPAIR_DIRECT_HIGH
#define ONPAIR_KEEP_LO_COUNT 3
#define ONPAIR_KERNEL_NAME   onpair_decompress_6tpt_directhi_keep3_lb4
#define ONPAIR_LAUNCH_BOUNDS __launch_bounds__(256, 4)
#include "onpair_decompress_tpt.cuh"
