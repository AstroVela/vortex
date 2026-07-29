# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Run random-access-bench once per (dataset, format, pattern, open-mode)
then merge the per-combination outputs
"""

import argparse
import glob
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

SCRIPT_DIR = Path(__file__).resolve().parent
BINARY = "target/release_debug/random-access-bench"
PARTS_DIR = Path("parts")

DATASETS = ["taxi", "feature-vectors", "nested-lists", "nested-structs"]
FORMATS = ["parquet", "lance", "vortex"]
PATTERNS = ["correlated", "uniform"]
OPEN_MODES = ["cached", "reopen"]


class Args(argparse.Namespace):
    emit_ingest_records: bool

    def __init__(self) -> None:
        super().__init__()
        self.emit_ingest_records = False


def run_combinations(emit_ingest_records: bool) -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    i = 0
    for dataset in DATASETS:
        for fmt in FORMATS:
            for pattern in PATTERNS:
                for open_mode in OPEN_MODES:
                    args = [
                        "bash",
                        str(SCRIPT_DIR / "bench-taskset.sh"),
                        BINARY,
                        "--datasets",
                        dataset,
                        "--formats",
                        fmt,
                        "--patterns",
                        pattern,
                        "--open-mode",
                        open_mode,
                        "-d",
                        "gh-json",
                        "-o",
                        str(PARTS_DIR / f"{i}.gh.json"),
                    ]
                    if emit_ingest_records:
                        args += ["--ingest-jsonl", str(PARTS_DIR / f"{i}.ingest.jsonl")]

                    profile_env = os.environ.copy()
                    profile_env["HEAP_PROFILE_ENGINE"] = "vortex"
                    profile_env["HEAP_PROFILE_FORMAT"] = fmt
                    profiled_args = [str(SCRIPT_DIR / "bench-heap-profile.sh"), *args]

                    print("+", " ".join(profiled_args), flush=True)
                    _ = subprocess.run(profiled_args, check=True, env=profile_env)
                    i += 1


"""
This function exists only because of taxi-legacy.

Every taxi invocation re-emits the pattern-less legacy taxi rows, so we need
the merge to drop the duplicates. Otherwise we could just merge JSONL lines.
"""


def merge(pattern: str, key: Callable[[dict[str, object]], object], out_path: str) -> None:
    seen: set[object] = set()
    lines: list[str] = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = cast(dict[str, object], json.loads(line))
                identity = key(record)
                if identity in seen:
                    continue
                seen.add(identity)
                lines.append(line)
    _ = Path(out_path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--emit-ingest-records",
        action="store_true",
        help="merge --ingest-jsonl records into results.ingest.jsonl",
    )
    args = cast(Args, parser.parse_args())

    run_combinations(args.emit_ingest_records)
    merge(f"{PARTS_DIR}/*.gh.json", lambda record: record["name"], "results.json")
    if args.emit_ingest_records:
        merge(
            f"{PARTS_DIR}/*.ingest.jsonl",
            lambda record: (record["kind"], record["dataset"], record["format"]),
            "results.ingest.jsonl",
        )


if __name__ == "__main__":
    main()
