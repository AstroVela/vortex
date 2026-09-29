// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
// Test-only native builder; deliberately not exposed as a provider capability.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "inc/Core/SPANN/Index.h"

int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: spfresh_fixture input.f32bin output-directory");
        }
        int32_t rows = 0;
        int32_t dimension = 0;
        std::ifstream input(argv[1], std::ios::binary);
        input.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char *>(&dimension), sizeof(dimension));
        if (!input || rows < 64 || rows > 100000 || dimension < 1 || dimension > 4096) {
            throw std::runtime_error("Invalid test fixture shape");
        }
        std::vector<float> vectors(static_cast<size_t>(rows) * dimension);
        input.read(reinterpret_cast<char *>(vectors.data()), vectors.size() * sizeof(float));
        if (!input || input.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Invalid test fixture length");
        }
        for (float value : vectors) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Invalid test fixture component");
            }
        }
        const auto root = std::filesystem::absolute(argv[2]);
        std::filesystem::create_directory(root);
        {
            SPTAG::SPANN::Index<float> index;
            const auto set = [&](const char *section, const char *name, const std::string &value) {
                if (index.SetParameter(name, value.c_str(), section) != SPTAG::ErrorCode::Success) {
                    throw std::runtime_error("Failed setting fixture parameter");
                }
            };
            set("Base", "ValueType", "Float");
            set("Base", "DistCalcMethod", "L2");
            set("Base", "IndexDirectory", root.string());
            set("Base", "HeadVectorIDs", "head_ids.bin");
            set("Base", "SSDIndex", "postings.bin");
            set("Base", "DataBlockSize", "1024");
            set("Base", "DataCapacity", std::to_string(rows));
            set("SelectHead", "isExecute", "true");
            set("SelectHead", "SelectHeadType", "Random");
            set("SelectHead", "Ratio", "0.25");
            set("SelectHead", "NumberOfThreads", "1");
            set("BuildHead", "isExecute", "true");
            set("BuildHead", "NumberOfThreads", "1");
            set("BuildHead", "BKTKmeansK", "4");
            set("BuildHead", "TPTNumber", "4");
            set("BuildHead", "TPTLeafSize", "32");
            set("BuildHead", "NeighborhoodSize", "16");
            set("BuildHead", "RefineIterations", "2");
            set("BuildHead", "DataBlockSize", "1024");
            set("BuildHead", "DataCapacity", std::to_string(rows));
            set("BuildSSDIndex", "isExecute", "true");
            set("BuildSSDIndex", "BuildSsdIndex", "true");
            set("BuildSSDIndex", "NumberOfThreads", "1");
            set("BuildSSDIndex", "IOThreadsPerHandler", "1");
            set("BuildSSDIndex", "PostingPageLimit", "12");
            set("BuildSSDIndex", "SearchPostingPageLimit", "12");
            set("BuildSSDIndex", "ReplicaCount", "4");
            set("BuildSSDIndex", "InternalResultNum", "32");
            set("BuildSSDIndex", "SearchInternalResultNum", "32");
            set("BuildSSDIndex", "TmpDir", root.string());
            if (index.BuildIndex(vectors.data(), rows, dimension, false, false) !=
                SPTAG::ErrorCode::Success) {
                throw std::runtime_error("Failed building native test fixture");
            }
        }
        for (const auto *name : {"vectors.bin", "tree.bin", "graph.bin", "deletes.bin"}) {
            std::filesystem::copy_file(root / "HeadIndex" / name, root / name);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
