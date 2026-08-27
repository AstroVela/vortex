// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout fixtures for the correctness suites and the comparison harness.
//!
//! Layouts are assembled by hand rather than through a writer strategy, because the strategies
//! chunk every column on the same boundaries and the interesting cases are the misaligned ones —
//! a morsel whose range cuts column `a` mid-chunk and column `b` on a boundary.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::layout_children;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::segments::SegmentSource;
use vortex_layout::segments::TestSegments;
use vortex_layout::sequence::SequenceId;
use vortex_layout::sequence::SequentialArrayStreamExt;
use vortex_session::VortexSession;

/// One column of a fixture: a name and the chunks it is stored in.
pub struct Column {
    /// The field name.
    pub name: FieldName,
    /// The chunks, in row order. Chunk boundaries need not agree with any other column's.
    pub chunks: Vec<ArrayRef>,
}

impl Column {
    /// Build a column from a name and its chunks.
    pub fn new(name: impl Into<FieldName>, chunks: Vec<ArrayRef>) -> Self {
        Self {
            name: name.into(),
            chunks,
        }
    }
}

/// A written fixture: the segments holding it, the layout over them, and the whole table as one
/// in-memory array for oracle comparisons.
pub struct Fixture {
    /// The segment source the layout reads from.
    pub segments: Arc<dyn SegmentSource>,
    /// The struct-of-chunked-flat layout.
    pub layout: LayoutRef,
    /// The complete table, unchunked.
    pub table: ArrayRef,
    /// The number of rows.
    pub row_count: u64,
}

/// Write a struct-of-chunked-flat fixture with per-column chunking.
///
/// Every column must cover the same total number of rows; their chunk boundaries need not agree.
pub async fn write_fixture(columns: Vec<Column>, session: &VortexSession) -> VortexResult<Fixture> {
    let segments = Arc::new(TestSegments::default());
    let strategy = FlatLayoutStrategy::default();
    let ctx = vortex_array::ArrayContext::empty();

    let mut row_count = None;
    let mut field_layouts = Vec::with_capacity(columns.len());
    let mut field_names = Vec::with_capacity(columns.len());
    let mut field_dtypes = Vec::with_capacity(columns.len());
    let mut table_fields: Vec<ArrayRef> = Vec::with_capacity(columns.len());

    for column in &columns {
        let dtype = column
            .chunks
            .first()
            .map(|chunk| chunk.dtype().clone())
            .ok_or_else(|| vortex_err!("a column needs at least one chunk"))?;

        let mut chunk_layouts = Vec::with_capacity(column.chunks.len());
        let mut rows = 0u64;
        for chunk in &column.chunks {
            let (ptr, eof) = SequenceId::root().split();
            let layout = strategy
                .write_stream(
                    ctx.clone().into(),
                    Arc::<TestSegments>::clone(&segments),
                    chunk.clone().to_array_stream().sequenced(ptr),
                    eof,
                    session,
                )
                .await?;
            rows += chunk.len() as u64;
            chunk_layouts.push(layout);
        }

        match row_count {
            None => row_count = Some(rows),
            Some(expected) if expected == rows => {}
            Some(expected) => {
                vortex_bail!("columns must have equal row counts: {expected} vs {rows}")
            }
        }

        let chunked =
            ChunkedLayout::new(rows, dtype.clone(), layout_children(chunk_layouts)).into_layout();
        field_layouts.push(chunked);
        field_names.push(column.name.clone());
        field_dtypes.push(dtype);

        // The oracle copy of the column, concatenated.
        table_fields.push(concat_chunks(&column.chunks)?);
    }

    let rows = row_count.unwrap_or(0);
    let struct_dtype = DType::Struct(
        StructFields::new(field_names.clone().into(), field_dtypes),
        Nullability::NonNullable,
    );
    let layout = StructLayout::new(rows, struct_dtype, field_layouts).into_layout();

    let table = StructArray::try_new(
        field_names.into(),
        table_fields,
        usize::try_from(rows).map_err(|_| vortex_err!("row count exceeds usize"))?,
        vortex_array::validity::Validity::NonNullable,
    )?
    .into_array();

    Ok(Fixture {
        segments,
        layout,
        table,
        row_count: rows,
    })
}

fn concat_chunks(chunks: &[ArrayRef]) -> VortexResult<ArrayRef> {
    if chunks.len() == 1 {
        return Ok(chunks[0].clone());
    }
    let dtype = chunks[0].dtype().clone();
    Ok(vortex_array::arrays::ChunkedArray::try_new(chunks.to_vec(), dtype)?.into_array())
}
