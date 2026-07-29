// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Measure what the compressor does to *array* size, independent of file layout.
//!
//! Reads a Parquet file, and for every column compresses it one block at a time with the
//! BtrBlocks compressor, reporting canonical `nbytes` against compressed `nbytes`. Unlike
//! `vx tree array` this never goes through a layout, so no `vortex.slice` node inflates the
//! numbers by reporting a whole block for a single chunk.
//!
//! Usage: `cargo run --release -p vortex-tui --features unstable_encodings --example array_size -- <file.parquet> [block_rows]`

use std::path::PathBuf;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
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
    let mut total_canonical = 0u64;
    let mut total_compressed = 0u64;

    println!(
        "{:<26}{:>14}{:>14}{:>9}  encodings",
        "column", "canonical", "compressed", "ratio"
    );
    for (name, chunks) in names.iter().zip(&columns) {
        let dtype = chunks[0].dtype().clone();
        // SAFETY: all chunks come from the same Parquet column, so they share a dtype.
        let whole = unsafe { ChunkedArray::new_unchecked(chunks.clone(), dtype) };

        let mut canonical_bytes = 0u64;
        let mut compressed_bytes = 0u64;
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
            compressed_bytes += compressed.nbytes();
            collect_encodings(&compressed, &mut encodings);
            offset = end;
        }

        encodings.sort();
        encodings.dedup();
        total_canonical += canonical_bytes;
        total_compressed += compressed_bytes;
        println!(
            "{:<26}{:>14}{:>14}{:>9.2}  {}",
            name,
            canonical_bytes,
            compressed_bytes,
            canonical_bytes as f64 / compressed_bytes.max(1) as f64,
            encodings.join(",")
        );
    }

    println!(
        "\n{:<26}{:>14}{:>14}{:>9.2}",
        "TOTAL",
        total_canonical,
        total_compressed,
        total_canonical as f64 / total_compressed.max(1) as f64
    );
    Ok(())
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
