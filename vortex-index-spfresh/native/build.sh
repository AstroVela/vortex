#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=${1:-"$here/../../target/spfresh-native"}
mkdir -p "$root"
root=$(cd "$root" && pwd)
revision=5893eb61ee3b18610b6b00f1939be7dae1af8904
source="$root/source"

if [[ ! -d "$source" ]]; then
    git init "$source"
    git -C "$source" fetch --depth 1 https://github.com/SPFresh/SPFresh.git "$revision"
    git -C "$source" checkout --detach FETCH_HEAD
fi
[[ $(git -C "$source" rev-parse HEAD) == "$revision" ]] || {
    echo 'Unexpected source revision; use a fresh build directory.' >&2
    exit 1
}
# Never reset an existing checkout. A changed patch requires a new build directory.
if [[ -n $(git -C "$source" ls-files --others) ]]; then
    echo 'Unexpected untracked source files; use a fresh build directory.' >&2
    exit 1
fi
if git -C "$source" diff HEAD --quiet; then
    git -C "$source" apply "$here/static-only.patch"
elif ! diff -u "$here/static-only.patch" <(git -C "$source" diff HEAD --binary); then
    echo 'Unexpected source modifications; use a fresh build directory.' >&2
    exit 1
fi
cmake -S "$here" -B "$root/build" -DSPFRESH_SOURCE="$source" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
for stamp in revision patch-sha256 source zstd-library; do
    cp "$root/build/$stamp.expected" "$root/build/$stamp.txt"
done
printf '\nexport VORTEX_SPFRESH_NATIVE=%q\n' "$root/build"
