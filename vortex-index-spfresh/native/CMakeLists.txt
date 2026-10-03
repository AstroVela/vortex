# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
cmake_minimum_required(VERSION 3.20)
project(vortex_spfresh LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    message(FATAL_ERROR "Only Linux x86_64 is qualified")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_library(ZSTD_LIBRARY NAMES zstd REQUIRED)
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h REQUIRED)

file(GLOB_RECURSE core CONFIGURE_DEPENDS
    "${SPFRESH_SOURCE}/AnnService/src/Core/*.cpp"
    "${SPFRESH_SOURCE}/AnnService/src/Helper/*.cpp")
list(FILTER core EXCLUDE REGEX "(DistanceUtils|SIMDUtils|InstructionUtils|ExtraSPDKController)\\.cpp$")
add_library(spfresh_core STATIC ${core})
target_include_directories(spfresh_core PUBLIC "${SPFRESH_SOURCE}/AnnService")
target_include_directories(spfresh_core PRIVATE "${ZSTD_INCLUDE_DIR}")
target_compile_definitions(spfresh_core PUBLIC SPTAG_STATIC_ONLY SPTAG_DISABLE_NUMA)
target_link_libraries(spfresh_core PUBLIC OpenMP::OpenMP_CXX Threads::Threads ${ZSTD_LIBRARY})

add_library(spfresh_distance STATIC
    "${SPFRESH_SOURCE}/AnnService/src/Core/Common/DistanceUtils.cpp"
    "${SPFRESH_SOURCE}/AnnService/src/Core/Common/SIMDUtils.cpp"
    "${SPFRESH_SOURCE}/AnnService/src/Core/Common/InstructionUtils.cpp")
target_include_directories(spfresh_distance PRIVATE "${SPFRESH_SOURCE}/AnnService")
# Patched per-function ISA targets keep dispatch/SSE paths baseline-compatible.
target_compile_definitions(spfresh_distance PRIVATE SPTAG_STATIC_ONLY)
target_link_libraries(spfresh_core PUBLIC spfresh_distance)

# Actually link a shared object to check PIC, not just an archive's compiler flags.
add_library(vortex_spfresh_pic_check SHARED bridge.cpp)
target_link_libraries(vortex_spfresh_pic_check PRIVATE spfresh_core)
target_link_options(vortex_spfresh_pic_check PRIVATE -Wl,--no-undefined)
add_executable(spfresh_fixture fixture.cpp)
target_link_libraries(spfresh_fixture PRIVATE spfresh_core)

# Opt-in benchmark; uses the same loader and bridge, without changing production ABI.
add_executable(spfresh_benchmark EXCLUDE_FROM_ALL benchmark.cpp bridge.cpp)
target_compile_definitions(spfresh_benchmark PRIVATE VORTEX_SPFRESH_BENCHMARK)
target_link_libraries(spfresh_benchmark PRIVATE spfresh_core)

enable_testing()
add_executable(spfresh_native_tests tests.cpp)
target_link_libraries(spfresh_native_tests PRIVATE spfresh_core)
add_test(NAME spfresh_native_tests COMMAND spfresh_native_tests)
add_executable(spfresh_workspace_tests workspace_tests.cpp bridge.cpp)
target_compile_definitions(spfresh_workspace_tests PRIVATE VORTEX_SPFRESH_TESTING)
target_link_libraries(spfresh_workspace_tests PRIVATE spfresh_core)
add_test(NAME spfresh_workspace_tests COMMAND spfresh_workspace_tests)

file(WRITE "${CMAKE_BINARY_DIR}/revision.expected" "5893eb61ee3b18610b6b00f1939be7dae1af8904\n")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/static-only.patch" patch_hash)
file(WRITE "${CMAKE_BINARY_DIR}/patch-sha256.expected" "${patch_hash}\n")
file(WRITE "${CMAKE_BINARY_DIR}/source.expected" "${SPFRESH_SOURCE}\n")
file(WRITE "${CMAKE_BINARY_DIR}/zstd-library.expected" "${ZSTD_LIBRARY}\n")
