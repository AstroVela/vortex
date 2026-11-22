// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include "inc/Core/VectorIndex.h"
#include "inc/Helper/ThreadPool.h"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

int main(int argc, char **argv) {
    try {
        if (argc > 2) {
            throw std::invalid_argument("Expected an optional shutdown iteration count");
        }
        const int iterations = argc == 2 ? std::stoi(argv[1]) : 10000;
        if (iterations < 1) {
            throw std::invalid_argument("Shutdown iteration count must be positive");
        }
        for (int i = 0; i < iterations; ++i) {
            SPTAG::Helper::ThreadPool pool;
            pool.init(1);
            std::this_thread::yield();
        }
        std::cout << iterations << " idle thread-pool shutdowns completed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
