// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Microbenchmark for the `preadv2(RWF_NOWAIT)` page-cache fast path in [`FileReadAt`].
//!
//! Run with `VORTEX_IO_NOWAIT=0` to measure the plain `spawn_blocking` + `pread` baseline.
//!
//! ```text
//! cargo run --release --example nowait_bench -p vortex-io
//! ```

use std::io::Write;
use std::time::Instant;

use futures::StreamExt;
use futures::stream;
use vortex_buffer::Alignment;
use vortex_error::VortexExpect;
use vortex_io::VortexReadAt;
use vortex_io::runtime::Handle;
use vortex_io::std_file::FileReadAt;

const FILE_SIZE: usize = 512 << 20;
const READS: usize = 200_000;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let read_size: usize = std::env::var("READ_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let concurrency: usize = std::env::var("CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);

    let mut tmp = tempfile::NamedTempFile::new()?;
    let chunk = vec![0xABu8; 1 << 20];
    for _ in 0..(FILE_SIZE >> 20) {
        tmp.write_all(&chunk)?;
    }
    tmp.flush()?;

    let handle = Handle::find().vortex_expect("example must run on a tokio runtime");
    let reader = FileReadAt::open(tmp.path(), handle)?;

    // Deterministic pseudo-random offsets, aligned to the read size.
    let slots = (FILE_SIZE / read_size) as u64;
    let offsets: Vec<u64> = (0..READS)
        .scan(0x243F6A8885A308D3u64, |s, _| {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            Some((*s % slots) * read_size as u64)
        })
        .collect();

    // Warm the page cache so we are measuring the hit path.
    for _ in 0..2 {
        run(&reader, &offsets, read_size, concurrency).await?;
    }

    let start = Instant::now();
    run(&reader, &offsets, read_size, concurrency).await?;
    let elapsed = start.elapsed();

    println!(
        "nowait={} read_size={read_size} concurrency={concurrency} reads={READS} \
         elapsed={:.3}s throughput={:.0} reads/s mean_latency={:.2}us",
        std::env::var("VORTEX_IO_NOWAIT").unwrap_or_else(|_| "1".into()),
        elapsed.as_secs_f64(),
        READS as f64 / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1e6 / READS as f64,
    );
    Ok(())
}

async fn run(
    reader: &FileReadAt,
    offsets: &[u64],
    read_size: usize,
    concurrency: usize,
) -> anyhow::Result<()> {
    stream::iter(offsets.iter().copied())
        .map(|offset| reader.read_at(offset, read_size, Alignment::none()))
        .buffer_unordered(concurrency)
        .for_each(|r| async move {
            r.vortex_expect("read failed");
        })
        .await;
    Ok(())
}
