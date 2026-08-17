// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#define TOKENS_PER_THREAD          6u
#define ONPAIR_BUF_BYTES_PER_TOKEN 12u
#define ONPAIR_KERNEL_NAME         onpair_decompress_6tpt_cap12_lb5
#define ONPAIR_LAUNCH_BOUNDS       __launch_bounds__(256, 5)
#include "onpair_decompress_tpt.cuh"
