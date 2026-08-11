// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Decompose a random-access take into its phases to see where the time goes.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Exp;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;
use vortex_bench::SESSION;

const NUM_CLUSTERS: usize = 5;
const CLUSTER_SIZE: usize = 20;
const POISSON_EXPECTED_COUNT: usize = 100;

fn correlated(row_count: u64) -> Vec<u64> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut indices = Vec::with_capacity(NUM_CLUSTERS * CLUSTER_SIZE);
    for _ in 0..NUM_CLUSTERS {
        let start = rng.random_range(0..row_count.saturating_sub(CLUSTER_SIZE as u64));
        for offset in 0..CLUSTER_SIZE as u64 {
            indices.push(start + offset);
        }
    }
    indices.sort_unstable();
    indices
}

fn uniform(row_count: u64) -> anyhow::Result<Vec<u64>> {
    let mut rng = StdRng::seed_from_u64(42);
    let rate = POISSON_EXPECTED_COUNT as f64 / row_count as f64;
    let exp = Exp::new(rate)?;
    let mut indices = Vec::with_capacity(POISSON_EXPECTED_COUNT);
    let mut pos = 0.0_f64;
    loop {
        let gap: f64 = exp.sample(&mut rng);
        pos += gap;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "positions below row_count fit in u64"
        )]
        let idx = pos as u64;
        if idx >= row_count {
            break;
        }
        indices.push(idx);
    }
    Ok(indices)
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort_unstable();
    v[v.len() / 2]
}

/// Read syscalls issued by this process so far.
fn syscr() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("syscr: ").map(|v| v.trim().parse().ok()))
        })
        .flatten()
        .unwrap_or(0)
}

/// Bytes this process has pulled through read syscalls so far.
fn rchar() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("rchar: ").map(|v| v.trim().parse().ok()))
        })
        .flatten()
        .unwrap_or(0)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(path), Some(pattern)) = (args.get(1), args.get(2)) else {
        anyhow::bail!("usage: take_phases <file.vortex> <correlated|uniform> [reps] [max-indices]")
    };
    let path = PathBuf::from(path);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);

    let file = SESSION
        .open_options()
        .with_layout_reader_cache()
        .open_path(path.as_path())
        .await?;
    let row_count = file.row_count();

    let mut indices = match pattern.as_str() {
        "correlated" => correlated(row_count),
        _ => uniform(row_count)?,
    };
    // Optional 5th arg: keep only the first N indices, to see how cost scales with index count.
    if let Some(n) = args.get(4).and_then(|s| s.parse::<usize>().ok()) {
        indices.truncate(n);
    }
    let indices_len = indices.len();

    // Phase timings, per repetition.
    let mut t_scan = Vec::new(); // file.scan() -- ScanBuilder construction
    let mut t_rowidx = Vec::new(); // with_row_indices(StrictSortedBuffer)
    let mut t_prepare = Vec::new(); // ScanBuilder::prepare()
    let mut t_stream = Vec::new(); // RepeatedScan::execute_array_stream (task construction)
    let mut t_read = Vec::new(); // read_all() -- IO + decode
    let mut t_total = Vec::new();
    let mut n_tasks = 0usize;
    let mut io_bytes: Vec<u64> = Vec::new();
    let mut io_calls: Vec<u64> = Vec::new();

    for _ in 0..reps {
        let total = Instant::now();

        let s = Instant::now();
        let builder = file.scan()?;
        t_scan.push(s.elapsed());

        let s = Instant::now();
        let indices_buf: Buffer<u64> = Buffer::from(indices.clone());
        let builder = builder.with_row_indices(StrictSortedBuffer::try_new(indices_buf)?);
        t_rowidx.push(s.elapsed());

        let s = Instant::now();
        let prepared = builder.prepare()?;
        t_prepare.push(s.elapsed());

        n_tasks = prepared.execute(None)?.len();

        let s = Instant::now();
        let stream = prepared.execute_array_stream(None)?;
        t_stream.push(s.elapsed());

        let io_before = rchar();
        let calls_before = syscr();
        let s = Instant::now();
        let array = stream.read_all().await?;
        t_read.push(s.elapsed());
        io_bytes.push(rchar() - io_before);
        io_calls.push(syscr() - calls_before);

        t_total.push(total.elapsed());
        black_box(array);
    }

    let name = path.file_name().map_or_else(
        || path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    println!("== {name} / {pattern} ==");
    println!("  rows in file : {row_count}");
    println!("  indices      : {indices_len}");
    println!("  split tasks  : {n_tasks}");
    io_bytes.sort_unstable();
    let med_io = io_bytes[io_bytes.len() / 2];
    println!(
        "  read syscalls: {:.2} MiB per take  ({:.0} bytes per row returned)",
        med_io as f64 / (1024.0 * 1024.0),
        med_io as f64 / indices_len as f64
    );
    io_calls.sort_unstable();
    let med_calls = io_calls[io_calls.len() / 2];
    println!(
        "  read calls   : {med_calls} per take  ({:.1} KiB average read size)",
        med_io as f64 / med_calls.max(1) as f64 / 1024.0
    );
    let m_total = median(t_total.clone());
    let phase = |label: &str, v: Vec<Duration>| {
        let m = median(v);
        println!(
            "  {label:<24} {:>9.1} us  ({:>5.2}% of take)",
            m.as_secs_f64() * 1e6,
            100.0 * m.as_secs_f64() / m_total.as_secs_f64()
        );
    };
    phase("file.scan()", t_scan);
    phase("with_row_indices()", t_rowidx);
    phase("prepare()", t_prepare);
    phase("execute_array_stream()", t_stream);
    phase("read_all()", t_read);
    println!("  {:<24} {:>9.1} us", "TOTAL", m_total.as_secs_f64() * 1e6);

    Ok(())
}
