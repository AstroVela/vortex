// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Read-path I/O benchmark: writes a wide multi-column file, then times projected
//! scans with a cold and hot page cache.
//!
//! Run with: `cargo run -p vortex-file --release --example ioscan -- <mode> <runs>`

use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use vortex::VortexSessionDefault;
use vortex::array::IntoArray;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::expr::BoundExpression;
use vortex::array::expr::Expression;
use vortex::array::expr::root;
use vortex::array::expr::select;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::session::VortexSession;
use vortex_scan::strict_sorted_buffer::StrictSortedBuffer;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

static PATH: LazyLock<String> = LazyLock::new(|| {
    std::env::var("IOSCAN_PATH").unwrap_or_else(|_| "/tmp/ioscan.vortex".to_string())
});
const NUM_COLS: usize = 40;
const CHUNK_ROWS: usize = 65_536;
const NUM_CHUNKS: usize = 64;

/// Build one chunk of a wide struct array: mostly i64 columns with a couple of
/// string columns, so segment sizes vary the way they do in real data.
fn chunk(seed: u64) -> vortex::array::ArrayRef {
    let mut fields: Vec<(String, vortex::array::ArrayRef)> = Vec::with_capacity(NUM_COLS);
    for c in 0..NUM_COLS {
        let name = format!("c{c:02}");
        if c % 10 == 7 {
            // A string column: wider segments.
            let vals: Vec<String> = (0..CHUNK_ROWS)
                .map(|r| format!("value-{}-{}", seed.wrapping_add(r as u64), c))
                .collect();
            let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
            fields.push((name, VarBinArray::from(refs).into_array()));
        } else {
            // Pseudo-random i64 so the compressor cannot collapse it to a constant.
            let vals: Buffer<i64> = (0..CHUNK_ROWS)
                .map(|r| {
                    (seed.wrapping_add(r as u64).wrapping_mul(0x9E3779B97F4A7C15)
                        ^ (c as u64).wrapping_mul(0x100000001B3)) as i64
                        % 1_000_000
                })
                .collect();
            fields.push((name, vals.into_array()));
        }
    }
    let refs: Vec<(&str, vortex::array::ArrayRef)> = fields
        .iter()
        .map(|(n, a)| (n.as_str(), a.clone()))
        .collect();
    StructArray::from_fields(&refs)
        .vortex_expect("struct")
        .into_array()
}

async fn write_file() -> VortexResult<()> {
    if Path::new(PATH.as_str()).exists() {
        return Ok(());
    }
    eprintln!("writing {NUM_CHUNKS} chunks x {CHUNK_ROWS} rows x {NUM_COLS} cols...");
    let chunks: Vec<_> = (0..NUM_CHUNKS).map(|i| chunk(i as u64 * 7919)).collect();
    let array = ChunkedArray::from_iter(chunks).into_array();
    let mut out = tokio::fs::File::create(PATH.as_str()).await?;
    SESSION
        .write_options()
        .write(&mut out, array.to_array_stream())
        .await?;
    let len = std::fs::metadata(PATH.as_str())?.len();
    eprintln!("wrote {:.1} MiB", len as f64 / (1u64 << 20) as f64);
    Ok(())
}

fn drop_caches() {
    std::process::Command::new("sync")
        .status()
        .vortex_expect("sync");
    match File::create("/proc/sys/vm/drop_caches") {
        Ok(mut f) => {
            f.write_all(b"3").vortex_expect("write drop_caches");
            std::thread::sleep(Duration::from_millis(200));
        }
        Err(e) => {
            eprintln!("warning: cannot drop caches ({e}); COLD results will be warm");
        }
    }
}

fn projection(file: &VortexFile, cols: &[&str]) -> BoundExpression {
    let names: Vec<std::sync::Arc<str>> = cols.iter().map(|c| (*c).into()).collect();
    let expr: Expression = select(names, root());
    expr.bind(file.dtype()).vortex_expect("bind projection")
}

/// One scan: open the file fresh and stream the projection to completion.
///
/// `stride` selects one row every `stride` rows, which skips whole chunks and so
/// leaves real gaps between the segments the scan needs.
async fn scan_once(cols: &[&str], stride: Option<u64>) -> VortexResult<usize> {
    let file = SESSION.open_options().open_path(PATH.as_str()).await?;
    let proj = projection(&file, cols);
    let mut scan = file.scan()?.with_projection(proj);
    if let Some(stride) = stride {
        let indices: Buffer<u64> = (0..file.row_count()).step_by(stride as usize).collect();
        scan = scan.with_row_indices(
            StrictSortedBuffer::try_new(indices).vortex_expect("strictly increasing"),
        );
    }
    let array = scan.into_array_stream()?.read_all().await?;
    Ok(array.len())
}

/// Device-level read counters, so we can see whether a change actually altered
/// the I/O issued rather than just the wall clock.
fn diskstat() -> (u64, u64) {
    let s = std::fs::read_to_string("/sys/block/vda/stat").unwrap_or_default();
    let f: Vec<u64> = s
        .split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect();
    (
        f.first().copied().unwrap_or(0),
        f.get(2).copied().unwrap_or(0) * 512,
    )
}

fn stats(mut xs: Vec<f64>) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = xs[xs.len() / 2];
    (xs[0], median, xs[xs.len() - 1])
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> VortexResult<()> {
    write_file().await?;

    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("all");
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    // Sparse projection: 3 of 40 columns, so the wanted segments are far apart.
    let sparse: Vec<&str> = vec!["c03", "c17", "c31"];
    // Dense projection: adjacent columns, so segments are near each other.
    let dense: Vec<&str> = vec!["c10", "c11", "c12"];

    let cases: Vec<(&str, &Vec<&str>, Option<u64>)> = vec![
        ("sparse 3/40 cols", &sparse, None),
        ("dense 3/40 adjacent", &dense, None),
        ("selective 1/4096 rows", &sparse, Some(4096)),
        ("selective 1/65536 rows", &sparse, Some(65536)),
    ];

    for (name, cols, stride) in cases {
        for cold in [true, false] {
            if mode != "all" && mode != if cold { "cold" } else { "hot" } {
                continue;
            }
            let mut times = Vec::with_capacity(runs);
            let mut ios = Vec::with_capacity(runs);
            let mut bytes = Vec::with_capacity(runs);
            let mut rows = 0usize;
            for _ in 0..runs {
                if cold {
                    drop_caches();
                } else {
                    // Warm the cache with a full scan we do not time.
                    scan_once(cols, stride).await?;
                }
                let d0 = diskstat();
                let t0 = Instant::now();
                rows = scan_once(cols, stride).await?;
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                let d1 = diskstat();
                ios.push((d1.0 - d0.0) as f64);
                bytes.push((d1.1 - d0.1) as f64);
            }
            let (min, med, max) = stats(times);
            let (_, med_ios, _) = stats(ios);
            let (_, med_bytes, _) = stats(bytes);
            println!(
                "{:>22} {:>5}  runs={} rows={}  min={:.1}ms median={:.1}ms max={:.1}ms  \
                 dev_ios={:.0} dev_read={:.1}MiB",
                name,
                if cold { "COLD" } else { "HOT" },
                runs,
                rows,
                min,
                med,
                max,
                med_ios,
                med_bytes / (1u64 << 20) as f64
            );
        }
    }
    Ok(())
}
