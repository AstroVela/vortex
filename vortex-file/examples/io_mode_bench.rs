// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares the local-file I/O modes on scan throughput and on read amplification.
//!
//! Throughput alone cannot tell direct I/O apart from a warm page cache, so this reports the bytes
//! the kernel actually fetched from the storage layer (`/proc/self/io`) alongside wall time, and
//! drops the page cache between runs. Run as root so that the cache drop takes effect:
//!
//! ```text
//! cargo run --release -p vortex-file --example io_mode_bench -- <path-to-vortex-file>
//! ```
//!
//! With no path, a synthetic file is generated in a temporary directory.

use std::fs;
use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use futures::StreamExt;
use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::memory::MemorySessionExt;
use vortex_array::session::ArraySessionExt;
use vortex_array::stream::ArrayStreamExt;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_edition::Edition;
use vortex_edition::EditionId;
use vortex_edition::EditionInclusion;
use vortex_edition::EditionSession;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::VortexReadAt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_io::std_file::FileIoMode;
use vortex_io::std_file::FileReadAt;
use vortex_io::std_file::FileReadAtOptions;
use vortex_layout::session::LayoutSession;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

/// Repetitions per measurement. The median is reported.
const ITERATIONS: usize = 5;

/// Number of small reads issued when measuring per-operation cost.
const SMALL_READS: usize = 20_000;

const MODES: [(&str, FileIoMode); 3] = [
    ("buffered", FileIoMode::Buffered),
    ("direct", FileIoMode::Direct),
    ("direct_uring", FileIoMode::DirectUring),
];

fn session() -> VortexResult<VortexSession> {
    let session = array_session()
        .with::<EditionSession>()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    enable_registered_encodings(&session)?;
    Ok(session)
}

/// Enable every registered encoding for writing by declaring a private edition that includes them.
///
/// The real editions live in the `vortex` umbrella crate, which this crate cannot depend on, and a
/// benchmark wants whatever the writer would normally choose rather than a curated subset.
fn enable_registered_encodings(session: &VortexSession) -> VortexResult<()> {
    const BENCH_EDITION: EditionId = EditionId::new("bench", 2026, 8, 0);

    let editions = session.editions();
    editions
        .declare_edition(Edition {
            id: BENCH_EDITION,
            min_vortex_version: None,
        })
        .map_err(|error| vortex_err!("{error}"))?;
    let ids = session
        .arrays()
        .registry()
        .read(|map| map.keys().copied().collect::<Vec<_>>());
    for id in ids {
        editions
            .declare_inclusion(EditionInclusion::new(&id, BENCH_EDITION))
            .map_err(|error| vortex_err!("{error}"))?;
    }
    session
        .enable_edition(BENCH_EDITION)
        .map_err(|error| vortex_err!("{error}"))?;
    Ok(())
}

/// Bytes this process has actually fetched from the storage layer.
///
/// This is the only measurement that distinguishes a real device read from a page-cache hit, and
/// it captures readahead that the application never asked for.
fn device_bytes_read() -> u64 {
    fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|io| {
            io.lines()
                .find_map(|line| line.strip_prefix("read_bytes:")?.trim().parse().ok())
        })
        .unwrap_or(0)
}

/// Drop the page cache so that a buffered run has to go to the device like a direct run does.
///
/// Returns false when the drop failed, which makes every subsequent buffered number a
/// cache-warm measurement rather than a cold one.
fn drop_page_cache() -> bool {
    drop(std::process::Command::new("sync").status());
    fs::write("/proc/sys/vm/drop_caches", "3").is_ok()
}

/// Pull this executable's pages back into the page cache.
///
/// `drop_caches` is indiscriminate: it evicts the benchmark's own text pages along with the data
/// file, so a "cold" iteration otherwise pays to re-fault the binary from disk before it can run
/// any of the work being measured. That cost lands on every I/O mode equally, including the direct
/// ones that have no page cache of their own to lose, and it is not what we are trying to measure.
fn warm_executable() {
    if let Ok(mut exe) = fs::File::open("/proc/self/exe") {
        drop(io::copy(&mut exe, &mut io::sink()));
    }
}

struct Measurement {
    elapsed: Duration,
    device_bytes: u64,
    rows: u64,
}

/// Open `path` with an explicitly chosen I/O mode, and confirm the mode actually took effect.
///
/// Going through `open_path` would consult the process-wide default, which is resolved once and
/// cached, so a benchmark that switched modes that way would silently measure the first mode
/// three times.
fn reader(session: &VortexSession, path: &Path, mode: FileIoMode) -> VortexResult<Arc<FileReadAt>> {
    let reader = FileReadAt::open_with_options(
        path,
        session.handle(),
        session.allocator(),
        FileReadAtOptions::default().with_io_mode(mode),
    )?;
    assert_eq!(
        reader.io_mode(),
        mode,
        "requested {mode:?} but the platform resolved {:?}; measurements would not compare",
        reader.io_mode()
    );
    Ok(Arc::new(reader))
}

async fn scan_all(
    session: &VortexSession,
    path: &Path,
    mode: FileIoMode,
) -> VortexResult<Measurement> {
    let source = reader(session, path, mode)?;
    let before = device_bytes_read();
    let start = Instant::now();

    let file = session.open_options().open(source).await?;
    let mut rows = 0u64;
    let stream = file.scan()?.into_array_stream()?;
    futures::pin_mut!(stream);
    while let Some(chunk) = stream.try_next().await? {
        rows += chunk.len() as u64;
    }

    Ok(Measurement {
        elapsed: start.elapsed(),
        device_bytes: device_bytes_read().saturating_sub(before),
        rows,
    })
}

async fn random_access(
    session: &VortexSession,
    path: &Path,
    indices: &[u64],
    mode: FileIoMode,
) -> VortexResult<Measurement> {
    let source = reader(session, path, mode)?;
    let before = device_bytes_read();
    let start = Instant::now();

    let file = session.open_options().open(source).await?;
    let buffer = Buffer::from_iter(indices.iter().copied());
    let mut rows = 0u64;
    let stream = file
        .scan()?
        .with_row_indices(StrictSortedBuffer::try_new(buffer)?)
        .into_array_stream()?;
    futures::pin_mut!(stream);
    while let Some(chunk) = stream.try_next().await? {
        rows += chunk.len() as u64;
    }

    Ok(Measurement {
        elapsed: start.elapsed(),
        device_bytes: device_bytes_read().saturating_sub(before),
        rows,
    })
}

async fn generate(
    session: &VortexSession,
    path: &Path,
    block: Option<Alignment>,
) -> VortexResult<()> {
    let columns = 16usize;
    let chunks = 64usize;
    let chunk_rows = 65_536usize;

    let mut fields: Vec<(String, ArrayRef)> = Vec::new();
    for c in 0..columns {
        let chunked: Vec<ArrayRef> = (0..chunks)
            .map(|k| {
                let values: Buffer<i64> = (0..chunk_rows)
                    .map(|i| ((i as u64).wrapping_mul(2_654_435_761) >> (c % 17)) as i64 + k as i64)
                    .collect();
                values.into_array()
            })
            .collect();
        fields.push((
            format!("c{c}"),
            ChunkedArray::from_iter(chunked).into_array(),
        ));
    }
    let refs: Vec<(&str, ArrayRef)> = fields
        .iter()
        .map(|(n, a)| (n.as_str(), a.clone()))
        .collect();
    let array = StructArray::from_fields(&refs)?.into_array();

    let file = tokio::fs::File::create(path).await?;
    let mut options = session.write_options();
    if let Some(block) = block {
        options = options.with_block_alignment(block);
    }
    options.write(file, array.to_array_stream()).await?;
    Ok(())
}

/// Median of repeated measurements, which is what makes buffered-vs-direct comparable: a single
/// run is dominated by scheduling noise and by whichever file was touched first.
fn median(mut measurements: Vec<Measurement>) -> Measurement {
    measurements.sort_by_key(|m| m.elapsed);
    measurements.swap_remove(measurements.len() / 2)
}

fn report(label: &str, measurement: &Measurement, logical: u64) {
    let seconds = measurement.elapsed.as_secs_f64();
    let mb = logical as f64 / (1024.0 * 1024.0);
    let device_mb = measurement.device_bytes as f64 / (1024.0 * 1024.0);
    let amplification = if logical > 0 {
        measurement.device_bytes as f64 / logical as f64
    } else {
        0.0
    };
    println!(
        "{label:<28} {:>8.0} ms   {:>8.1} MB/s   device {:>8.1} MB   amp {amplification:>5.2}x   rows {}",
        seconds * 1000.0,
        mb / seconds,
        device_mb,
        measurement.rows,
    );
}

/// Issue `count` small reads at random offsets and report the achieved rate.
///
/// This isolates per-operation cost from bandwidth. The full-scan numbers are dominated by a
/// handful of large coalesced transfers, which hides whatever each mode charges per request; a
/// query that touches many small segments pays that per-request cost instead.
async fn small_reads(
    session: &VortexSession,
    path: &Path,
    mode: FileIoMode,
    size: usize,
    count: usize,
    concurrency: usize,
) -> VortexResult<Duration> {
    let source = reader(session, path, mode)?;
    let file_len = source.size().await?;
    let offsets: Vec<u64> = (0..count)
        .map(|i| {
            let span = file_len - size as u64;
            (i as u64).wrapping_mul(2_654_435_761) % span
        })
        .collect();

    let start = Instant::now();
    futures::stream::iter(offsets)
        .map(|offset| {
            let source = Arc::clone(&source);
            async move { source.read_at(offset, size, Alignment::new(256)).await }
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await?;
    Ok(start.elapsed())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> VortexResult<()> {
    let session = session()?;
    let args: Vec<String> = std::env::args().collect();

    let (paths, _tempdir): (Vec<(String, PathBuf)>, Option<tempfile::TempDir>) =
        if let Some(path) = args.get(1) {
            (vec![("file".to_string(), PathBuf::from(path))], None)
        } else {
            let dir = tempfile::tempdir()?;
            let plain = dir.path().join("plain.vortex");
            let aligned = dir.path().join("aligned.vortex");
            println!("generating synthetic files in {}", dir.path().display());
            generate(&session, &plain, None).await?;
            generate(&session, &aligned, Some(Alignment::new(4096))).await?;
            (
                vec![
                    ("segments@256".to_string(), plain),
                    ("segments@4096".to_string(), aligned),
                ],
                Some(dir),
            )
        };

    if !drop_page_cache() {
        println!("WARNING: could not drop the page cache; buffered numbers are cache-warm");
    }

    for (name, path) in &paths {
        let size = fs::metadata(path)?.len();
        println!(
            "\n=== {name}: {} ({:.1} MB)",
            path.display(),
            size as f64 / 1048576.0
        );

        // Row indices are fixed across modes so every mode does identical logical work.
        let row_count = {
            let file = session.open_options().open_path(path).await?;
            file.row_count()
        };
        // Few enough rows that zone-map pruning can skip most chunks. A uniform sample of a
        // thousand rows would touch every chunk of every column and measure a full scan instead.
        let sample: Vec<u64> = (0..32)
            .map(|i| (i as u64).wrapping_mul(2_654_435_761) % row_count)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();

        println!("-- full scan, cold cache incl. cold binary (median of {ITERATIONS})");
        for (label, mode) in MODES {
            let mut runs = Vec::new();
            for _ in 0..ITERATIONS {
                drop_page_cache();
                runs.push(scan_all(&session, path, mode).await?);
            }
            report(label, &median(runs), size);
        }

        println!("-- full scan, cold data only, binary re-warmed (median of {ITERATIONS})");
        for (label, mode) in MODES {
            let mut runs = Vec::new();
            for _ in 0..ITERATIONS {
                drop_page_cache();
                warm_executable();
                runs.push(scan_all(&session, path, mode).await?);
            }
            report(label, &median(runs), size);
        }

        println!("-- full scan, warm cache (median of {ITERATIONS})");
        for (label, mode) in MODES {
            scan_all(&session, path, mode).await?;
            let mut runs = Vec::new();
            for _ in 0..ITERATIONS {
                runs.push(scan_all(&session, path, mode).await?);
            }
            report(label, &median(runs), size);
        }

        println!(
            "-- random access, {} rows, cold cache (median of {ITERATIONS})",
            sample.len()
        );
        for (label, mode) in MODES {
            let mut runs = Vec::new();
            for _ in 0..ITERATIONS {
                drop_page_cache();
                runs.push(random_access(&session, path, &sample, mode).await?);
            }
            let measurement = median(runs);
            report(label, &measurement, measurement.device_bytes.max(1));
        }

        for concurrency in [1usize, 8, 128] {
            println!("-- 4KiB random reads x{SMALL_READS} at concurrency {concurrency} (warm)");
            for (label, mode) in MODES {
                let mut runs = Vec::new();
                for _ in 0..3 {
                    runs.push(
                        small_reads(&session, path, mode, 4096, SMALL_READS, concurrency).await?,
                    );
                }
                runs.sort();
                let elapsed = runs[runs.len() / 2];
                let per_op = elapsed.as_secs_f64() * 1e6 / SMALL_READS as f64;
                println!(
                    "{label:<28} {:>8.0} ms   {:>9.0} reads/s   {per_op:>7.1} us/read",
                    elapsed.as_secs_f64() * 1000.0,
                    SMALL_READS as f64 / elapsed.as_secs_f64(),
                );
            }
        }
    }

    Ok(())
}
