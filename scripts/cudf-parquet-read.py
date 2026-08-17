#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Times a full GPU Parquet read with cuDF.

`cudf.read_parquet` performs the entire read on the device: page header decode,
codec decompression, dictionary/RLE/plain decoding and column assembly. That makes it
the like-for-like opponent for the Vortex GPU backend, which also decodes all the way
to canonical arrays on device.

Timing excludes interpreter start, `import cudf`, CUDA context creation and any JIT
warm-up, all of which are paid once per process and are not part of a read. A warm-up
read runs first for exactly that reason.

Emits one JSON object on stdout so the benchmark can parse it.
"""

import argparse
import json
import sys
import time
from datetime import date


def synchronize() -> None:
    """Block until queued device work finishes.

    `cudf.read_parquet` returns a materialized DataFrame, but synchronizing explicitly
    keeps the measurement honest if that ever stops being true.
    """
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except ImportError:
        pass


def normalize(frame):
    """Collapses representation differences that are not value differences.

    A Parquet DATE column comes back from pyarrow as a column of `datetime.date`
    objects but from cuDF as `datetime64[s]`. Those hold the same instants, yet
    `check_dtype=False` does not bridge them because one side is `object`, so the
    comparison reports every row as different. Coercing both sides to datetime64
    compares the dates themselves.
    """
    import pandas as pd

    for name in frame.columns:
        column = frame[name]
        if column.dtype == object and len(column) and isinstance(column.iloc[0], date):
            frame[name] = pd.to_datetime(column)
    return frame


def verify(path: str, row_group: int, frame) -> None:
    """Fails unless one GPU-decoded row group matches the CPU Parquet decoder."""
    import pyarrow.parquet as pq
    from pandas.testing import assert_frame_equal

    expected = normalize(pq.ParquetFile(path).read_row_group(row_group).to_pandas())
    actual = normalize(frame.to_pandas())

    # cuDF and pyarrow can land on different-but-equivalent dtypes (nullable vs numpy
    # backed, for instance), so compare values and leave dtype policy out of it.
    assert_frame_equal(actual, expected, check_dtype=False)


def read_all_row_groups(path: str, verify_output: bool) -> tuple[int, int]:
    """Materializes and optionally verifies every row group independently."""
    import cudf
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    rows = 0
    columns = 0
    for row_group in range(parquet.num_row_groups):
        frame = cudf.read_parquet(path, row_groups=[row_group])
        synchronize()
        if verify_output:
            verify(path, row_group, frame)
        rows += len(frame)
        columns = len(frame.columns)
        del frame
    return rows, columns


def read_all_chunks(path: str) -> tuple[int, int]:
    """Scans one open Parquet source through libcudf's bounded chunked reader."""
    import pylibcudf as plc

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo([path])
    ).build()
    reader = plc.io.parquet.ChunkedParquetReader(
        options,
        chunk_read_limit=8 * 1024**3,
        pass_read_limit=1024**3,
    )

    rows = 0
    columns = 0
    while reader.has_next():
        chunk = reader.read_chunk().tbl
        synchronize()
        rows += chunk.num_rows()
        columns = chunk.num_columns()
        del chunk
    return rows, columns


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Parquet file to read")
    parser.add_argument("--iterations", type=int, default=1, help="timed reads to perform")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="cross-check the GPU read against a CPU Parquet read",
    )
    args = parser.parse_args()

    # Warm-up: pays CUDA context creation and any first-call JIT so they stay out of
    # the timed reads below. Reading one row group at a time keeps scans larger than device
    # memory bounded while still materializing every row and column on the GPU.
    if args.verify:
        rows, columns = read_all_row_groups(args.path, True)
    else:
        rows, columns = read_all_chunks(args.path)

    runs_ns = []
    for _ in range(max(args.iterations, 1)):
        start = time.perf_counter_ns()
        timed_rows, timed_columns = read_all_chunks(args.path)
        runs_ns.append(time.perf_counter_ns() - start)
        if (timed_rows, timed_columns) != (rows, columns):
            raise RuntimeError(
                "timed scan shape differs from warm-up: "
                f"{(timed_rows, timed_columns)} != {(rows, columns)}"
            )

    json.dump(
        {
            "min_ns": min(runs_ns),
            "runs_ns": runs_ns,
            "rows": int(rows),
            "columns": int(columns),
            "verified": bool(args.verify),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
