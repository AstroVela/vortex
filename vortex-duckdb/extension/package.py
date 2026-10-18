# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Package the compiled library as an unsigned DuckDB extension.

Version/platform must match the target DuckDB (SELECT version(); PRAGMA platform;).
The output filename must be vortex.duckdb_extension to match the exported entrypoint.
"""
import argparse
from pathlib import Path
import shutil


def package(library: Path, output: Path, version: str, platform: str) -> None:
    if output.name != "vortex.duckdb_extension":
        raise ValueError("Output must be named vortex.duckdb_extension")
    if library.resolve() == output.resolve():
        raise ValueError("Output cannot replace the compiled library")
    fields = ["", "", "", "CPP", "tantivy-v1", version, platform, "4"]
    encoded = [value.encode("ascii") for value in fields]
    if any(len(value) > 32 or b"\0" in value for value in encoded):
        raise ValueError("Metadata fields must be NUL-free ASCII of at most 32 bytes")
    # DuckDB's C++ ABI footer: eight 32-byte fields in reverse order and a
    # 256-byte empty signature. This is intentionally an unsigned local build.
    with library.open("rb") as source, output.open("xb") as destination:
        shutil.copyfileobj(source, destination)
        destination.write(b"".join(value.ljust(32, b"\0") for value in encoded) + bytes(256))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duckdb-version", required=True)
    parser.add_argument("--platform", required=True)
    args = parser.parse_args()
    package(args.library, args.output, args.duckdb_version, args.platform)
