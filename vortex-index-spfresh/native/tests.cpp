// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "inc/Core/Common/CommonUtils.h"

int main() {
    try {
        // On LP64, an eight-byte atomic operation on value also touches its neighbor.
        struct alignas(8) Counter {
            float value = 0;
            uint32_t neighbor = 0x12345678;
        } counter;
        if (SPTAG::COMMON::Utils::atomic_float_add(&counter.value, 1.0F) != 1.0F || counter.value != 1.0F ||
            counter.neighbor != 0x12345678) {
            throw std::runtime_error("Float atomic update overwrote adjacent data");
        }
#pragma omp parallel for num_threads(4)
        for (int i = 0; i < 1000; ++i) {
            SPTAG::COMMON::Utils::atomic_float_add(&counter.value, 1.0F);
        }
        if (counter.value != 1001.0F || counter.neighbor != 0x12345678) {
            throw std::runtime_error("Float atomic update lost increments or overwrote adjacent data");
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
