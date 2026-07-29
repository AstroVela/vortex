// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measure the compressor's on-disk output and encoding-tree complexity, per column.
//!
//! Reads a Parquet file and, for every column, compresses it one block at a time with the
//! BtrBlocks compressor, then *serializes* each compressed block and sums the buffer lengths.
//! That is the number of bytes the block actually occupies on disk, with no layout, footer or
//! chunking in the way. Alongside it, reports the encoding tree's node count and maximum depth,
//! since a smaller file bought with a much deeper decode tree is not obviously a good trade.
//!
//! Usage: `cargo run --release -p vortex-tui --features unstable_encodings --example array_size -- <file.parquet> [block_rows]`

use std::path::PathBuf;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::VortexSessionDefault;
use vortex::array::ArrayContext;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::serde::SerializeOptions;
use vortex::compressor::BtrBlocksCompressor;
use vortex::session::VortexSession;
use vortex_arrow::FromArrowArray;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next().map(PathBuf::from) else {
        anyhow::bail!("usage: array_size <file.parquet> [block_rows]");
    };
    let block_rows: usize = match args.next() {
        Some(arg) => arg.parse()?,
        None => 524_288,
    };

    let session = VortexSession::default();
    let file = std::fs::File::open(&path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
        .with_batch_size(65_536)
        .build()?;

    // Accumulate each column's batches so we can compress whole blocks. The schema is fixed
    // across batches, so column position is a stable key.
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

    let compressor = BtrBlocksCompressor::default();
    let array_ctx = ArrayContext::empty();
    let mut total_canonical = 0u64;
    let mut total_disk = 0u64;
    let mut total_nodes = 0u64;
    let mut max_depth_overall = 0u32;

    println!(
        "{:<24}{:>13}{:>13}{:>8}{:>7}{:>7}  encodings",
        "column", "canonical", "on_disk", "ratio", "nodes", "depth"
    );
    for (name, chunks) in names.iter().zip(&columns) {
        let dtype = chunks[0].dtype().clone();
        // SAFETY: all chunks come from the same Parquet column, so they share a dtype.
        let whole = unsafe { ChunkedArray::new_unchecked(chunks.clone(), dtype) };

        let mut canonical_bytes = 0u64;
        let mut disk_bytes = 0u64;
        let mut nodes = 0u64;
        let mut max_depth = 0u32;
        let mut encodings: Vec<String> = Vec::new();

        let len = whole.len();
        let mut offset = 0usize;
        while offset < len {
            let end = (offset + block_rows).min(len);
            let block = whole.as_array().slice(offset..end)?;
            let mut ctx = session.create_execution_ctx();
            let canonical = block.execute::<Canonical>(&mut ctx)?.into_array();
            canonical_bytes += canonical.nbytes();
            let compressed = compressor.compress(&canonical, &mut ctx)?;
            for buffer in
                compressed
                    .clone()
                    .serialize(&array_ctx, &session, &SerializeOptions::default())?
            {
                disk_bytes += buffer.len() as u64;
            }
            nodes += count_nodes(&compressed);
            max_depth = max_depth.max(depth(&compressed));
            collect_encodings(&compressed, &mut encodings);
            offset = end;
        }

        encodings.sort();
        encodings.dedup();
        total_canonical += canonical_bytes;
        total_disk += disk_bytes;
        total_nodes += nodes;
        max_depth_overall = max_depth_overall.max(max_depth);
        println!(
            "{:<24}{:>13}{:>13}{:>8.2}{:>7}{:>7}  {}",
            name,
            canonical_bytes,
            disk_bytes,
            canonical_bytes as f64 / disk_bytes.max(1) as f64,
            nodes,
            max_depth,
            encodings.join(",")
        );
    }

    println!(
        "\n{:<24}{:>13}{:>13}{:>8.2}{:>7}{:>7}",
        "TOTAL",
        total_canonical,
        total_disk,
        total_canonical as f64 / total_disk.max(1) as f64,
        total_nodes,
        max_depth_overall
    );
    Ok(())
}

/// Total number of arrays in the encoding tree.
fn count_nodes(array: &ArrayRef) -> u64 {
    1 + array.children().iter().map(count_nodes).sum::<u64>()
}

/// Longest root-to-leaf path in the encoding tree, counting the root as depth 1.
fn depth(array: &ArrayRef) -> u32 {
    1 + array.children().iter().map(depth).max().unwrap_or(0)
}

fn collect_encodings(array: &ArrayRef, out: &mut Vec<String>) {
    let id = array.encoding_id().to_string();
    if let Some(short) = id.split('.').next_back() {
        out.push(short.to_string());
    }
    for child in array.children() {
        collect_encodings(&child, out);
    }
}
