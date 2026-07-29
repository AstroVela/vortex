// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare the RunEnd and RLE compression stacks column by column.
//!
//! RunEnd and RLE encode the same information — a sequence of runs — with different positional
//! representations. RunEnd stores one end position per run; RLE stores one chunk-local run index
//! per element. They are therefore direct competitors, and the compressor picks between them on a
//! sampled size estimate. This example makes that choice inspectable: for every column it
//! compresses each block twice, once with RLE excluded and once with RunEnd excluded, and prints
//! both resulting stacks with their on-disk bytes.
//!
//! A third arm runs the unmodified compressor, which has both schemes available and picks between
//! them from a ~1% sample. Comparing that arm against the better of the two forced arms shows
//! whether the sampled estimate actually agrees with the compressed sizes it is predicting.
//!
//! Only columns where at least one variant actually reaches for RunEnd or RLE are reported.
//!
//! Usage: `cargo run --release -p vortex-tui --features unstable_encodings --example runend_vs_rle -- <file.parquet> [block_rows]`

use std::path::PathBuf;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::VortexSessionDefault;
use vortex::array::ArrayContext;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_native_ptype;
use vortex::array::serde::SerializeOptions;
use vortex::session::VortexSession;
use vortex_arrow::FromArrowArray;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::float::FloatRLEScheme;
use vortex_btrblocks::schemes::integer::IntRLEScheme;
use vortex_btrblocks::schemes::integer::RunEndScheme;

/// One variant's measurement of a single column.
struct Measured {
    disk: u64,
    nodes: u64,
    depth: u32,
    /// Compact stack of the first block, e.g. `dict(rle(primitive, delta(..), sequence), onpair)`.
    stack: String,
    uses_target: bool,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next().map(PathBuf::from) else {
        anyhow::bail!("usage: runend_vs_rle <file.parquet> [block_rows]");
    };
    let block_rows: usize = match args.next() {
        Some(arg) => arg.parse()?,
        None => 524_288,
    };

    let session = VortexSession::default();
    let array_ctx = ArrayContext::empty();

    // Two compressors that differ only in which of the two run encodings is available.
    let runend_only = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([IntRLEScheme.id(), FloatRLEScheme.id()])
        .build();
    let rle_only = BtrBlocksCompressorBuilder::default()
        .exclude_schemes([RunEndScheme.id()])
        .build();
    // Both available: this is what the shipped compressor does, choosing from a sample.
    let both = BtrBlocksCompressorBuilder::default().build();

    let file = std::fs::File::open(&path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(65_536)
        .build()?;

    let mut names: Vec<String> = Vec::new();
    let mut columns: Vec<Vec<ArrayRef>> = Vec::new();
    for batch in reader {
        let batch = batch?;
        let schema = batch.schema();
        if names.is_empty() {
            names = schema.fields().iter().map(|f| f.name().clone()).collect();
            columns.resize_with(names.len(), Vec::new);
        }
        for (i, field) in schema.fields().iter().enumerate() {
            columns[i].push(ArrayRef::from_arrow(
                batch.column(i).as_ref(),
                field.is_nullable(),
            )?);
        }
    }

    let mut totals = [(0u64, 0u64); 2];
    let mut reported = 0usize;

    for (name, chunks) in names.iter().zip(&columns) {
        let dtype = chunks[0].dtype().clone();
        // SAFETY: all chunks come from the same Parquet column, so they share a dtype.
        let whole = unsafe { ChunkedArray::new_unchecked(chunks.clone(), dtype) };

        let re = measure(
            &runend_only,
            whole.as_array(),
            block_rows,
            "vortex.runend",
            &session,
            &array_ctx,
        )?;
        let rle = measure(
            &rle_only,
            whole.as_array(),
            block_rows,
            "fastlanes.rle",
            &session,
            &array_ctx,
        )?;

        let picked = measure(
            &both,
            whole.as_array(),
            block_rows,
            "fastlanes.rle",
            &session,
            &array_ctx,
        )?;

        if !re.uses_target && !rle.uses_target {
            continue;
        }
        reported += 1;
        totals[0].0 += re.disk;
        totals[0].1 += re.nodes;
        totals[1].0 += rle.disk;
        totals[1].1 += rle.nodes;

        let avg_run = avg_run_len(whole.as_array(), &session)?;
        let best = re.disk.min(rle.disk);
        let winner = if rle.disk < re.disk { "RLE" } else { "RunEnd" };
        let pct = 100.0 * (rle.disk as f64 - re.disk as f64) / re.disk.max(1) as f64;
        // How far the sampled estimate landed above the better of the two forced arms.
        let regret = 100.0 * (picked.disk as f64 - best as f64) / best.max(1) as f64;
        println!(
            "=== {name}   avg_run={avg_run}   smaller: {winner} ({pct:+.1}% RLE vs RunEnd)   estimator picked {} ({regret:+.1}% vs best)",
            if picked.uses_target {
                "RLE"
            } else {
                "RunEnd/other"
            }
        );
        println!(
            "  RunEnd  {:>11} B  nodes={:<4} depth={}  {}",
            re.disk,
            re.nodes,
            re.depth,
            if re.uses_target {
                re.stack
            } else {
                format!("{} (RunEnd not selected)", re.stack)
            }
        );
        println!(
            "  chosen  {:>11} B  nodes={:<4} depth={}  {}",
            picked.disk, picked.nodes, picked.depth, picked.stack
        );
        println!(
            "  RLE     {:>11} B  nodes={:<4} depth={}  {}",
            rle.disk,
            rle.nodes,
            rle.depth,
            if rle.uses_target {
                rle.stack
            } else {
                format!("{} (RLE not selected)", rle.stack)
            }
        );
    }

    println!(
        "\ncolumns of this form: {reported}\n  RunEnd total {:>11} B  nodes={}\n  RLE    total {:>11} B  nodes={}",
        totals[0].0, totals[0].1, totals[1].0, totals[1].1
    );
    Ok(())
}

/// Average run length of the column, or `-` for dtypes this example does not measure.
///
/// This is the property that decides which decode kernel is cheaper: RunEnd walks one splat
/// fill per run, so it gets faster as runs lengthen, while RLE gathers one index per element
/// at a cost independent of run structure. Null slots are compared by their stored value.
fn avg_run_len(array: &ArrayRef, session: &VortexSession) -> anyhow::Result<String> {
    if !array.dtype().is_primitive() {
        return Ok("-".to_string());
    }
    let mut ctx = session.create_execution_ctx();
    let primitive = array.clone().execute::<PrimitiveArray>(&mut ctx)?;
    if primitive.is_empty() {
        return Ok("-".to_string());
    }
    let runs = match_each_native_ptype!(primitive.ptype(), |P| {
        let values = primitive.as_slice::<P>();
        1 + values.windows(2).filter(|w| w[0] != w[1]).count()
    });
    Ok(format!("{:.1}", primitive.len() as f64 / runs as f64))
}

fn measure(
    compressor: &BtrBlocksCompressor,
    array: &ArrayRef,
    block_rows: usize,
    target: &str,
    session: &VortexSession,
    array_ctx: &ArrayContext,
) -> anyhow::Result<Measured> {
    let mut disk = 0u64;
    let mut nodes = 0u64;
    let mut depth_max = 0u32;
    let mut stack = String::new();
    let mut uses_target = false;

    let len = array.len();
    let mut offset = 0usize;
    while offset < len {
        let end = (offset + block_rows).min(len);
        let mut ctx = session.create_execution_ctx();
        let canonical = array.slice(offset..end)?.execute::<Canonical>(&mut ctx)?;
        let compressed = compressor.compress(&canonical.into_array(), &mut ctx)?;
        for buffer in
            compressed
                .clone()
                .serialize(array_ctx, session, &SerializeOptions::default())?
        {
            disk += buffer.len() as u64;
        }
        nodes += count_nodes(&compressed);
        depth_max = depth_max.max(depth(&compressed));
        uses_target |= contains(&compressed, target);
        if stack.is_empty() {
            stack = render(&compressed);
        }
        offset = end;
    }
    Ok(Measured {
        disk,
        nodes,
        depth: depth_max,
        stack,
        uses_target,
    })
}

/// Renders the encoding tree as `parent(child, child)`, dropping the vendor prefix.
fn render(array: &ArrayRef) -> String {
    let id = array.encoding_id().to_string();
    let short = id.split('.').next_back().unwrap_or(&id).to_string();
    let children = array.children();
    if children.is_empty() {
        return short;
    }
    let names = array.children_names();
    let inner: Vec<String> = names
        .into_iter()
        .zip(children.iter())
        .map(|(n, c)| format!("{n}={}", render(c)))
        .collect();
    format!("{short}({})", inner.join(", "))
}

fn contains(array: &ArrayRef, target: &str) -> bool {
    array.encoding_id().to_string() == target
        || array.children().iter().any(|c| contains(c, target))
}

fn count_nodes(array: &ArrayRef) -> u64 {
    1 + array.children().iter().map(count_nodes).sum::<u64>()
}

fn depth(array: &ArrayRef) -> u32 {
    1 + array.children().iter().map(depth).max().unwrap_or(0)
}
