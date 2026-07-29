#!/usr/bin/env bash

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <duckdb-bench-binary>" >&2
    exit 2
fi

binary="$1"
if [[ ! -x "$binary" ]]; then
    echo "DuckDB benchmark binary is not executable: $binary" >&2
    exit 1
fi

duckdb_library="$(
    ldd "$binary" \
        | awk '$1 == "libduckdb.so" && $2 == "=>" { print $3; exit }'
)"
if [[ -z "$duckdb_library" || ! -f "$duckdb_library" ]]; then
    echo "Could not resolve libduckdb.so used by $binary" >&2
    exit 1
fi

binary_symbols="$(nm -D --defined-only "$binary")"
missing_symbols=()
for symbol in malloc calloc realloc free aligned_alloc posix_memalign; do
    if ! awk -v symbol="$symbol" '$NF == symbol { found = 1 } END { exit !found }' \
        <<< "$binary_symbols"
    then
        missing_symbols+=("$symbol")
    fi
done

if (( ${#missing_symbols[@]} > 0 )); then
    echo "duckdb-bench does not export jemalloc overrides for: ${missing_symbols[*]}" >&2
    exit 1
fi

duckdb_definitions="$(nm --defined-only "$duckdb_library")"
if awk '$NF == "duckdb_je_malloc" { found = 1 } END { exit !found }' \
    <<< "$duckdb_definitions"
then
    echo "DuckDB still contains its private jemalloc allocator: $duckdb_library" >&2
    exit 1
fi

duckdb_imports="$(nm -D --undefined-only "$duckdb_library")"
for symbol in malloc realloc free; do
    if ! awk -v symbol="$symbol" '
        {
            name = $NF
            sub(/@.*/, "", name)
            if (name == symbol) {
                found = 1
            }
        }
        END { exit !found }
    ' <<< "$duckdb_imports"
    then
        echo "DuckDB does not import the expected process allocator symbol: $symbol" >&2
        exit 1
    fi
done

echo "Verified DuckDB allocations resolve through process-wide jemalloc"
