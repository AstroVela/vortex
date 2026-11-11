#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-"$here/../../target/rabitq-native"}
mkdir -p "$root"
root=$(cd "$root" && pwd)
revision=dabfdb8c200a6f155df4edd0b9c825e6728b14a1
source="$root/source"
if [[ ! -d "$source" ]]; then
    git init "$source"
    git -C "$source" fetch --depth 1 https://github.com/VectorDB-NTU/RaBitQ-Library.git "$revision"
    git -C "$source" checkout --detach FETCH_HEAD
fi
if [[ $(git -C "$source" rev-parse HEAD) != "$revision" ]] ||
    [[ -n $(git -C "$source" status --porcelain --untracked-files=all) ]]; then
    echo 'Expected a clean pinned RaBitQ checkout; use a fresh build directory.' >&2
    exit 1
fi
cmake -S "$source" -B "$root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DRABITQ_BUILD_FASTSCAN=ON -DRABITQ_BUILD_INDEX=OFF \
    -DRABITQ_BUILD_SAMPLES=OFF -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_BENCHMARKS=OFF -DRABITQ_USE_OPENMP=OFF \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build "$root/build" --target rabitq_core rabitq_fastscan \
    --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
printf '%s\n' "$revision" > "$root/build/revision.txt"
printf '%s\n' "$source" > "$root/build/source.txt"
printf '\nexport VORTEX_RABITQ_NATIVE=%q\n' "$root/build"
