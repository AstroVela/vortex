// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Restricted physical layout used by the self-paced execution experiment.

use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::struct_::StructStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// Writes exactly the `Struct<Chunked<Flat<i64>>>` shape supported by the experiment.
#[derive(Clone)]
pub struct SelfPacedLayoutStrategy {
    delegate: StructStrategy,
}

impl Default for SelfPacedLayoutStrategy {
    fn default() -> Self {
        let flat = FlatLayoutStrategy::default();
        let chunked = ChunkedLayoutStrategy::new(flat.clone()).with_preserve_single_child();
        Self {
            delegate: StructStrategy::new(Arc::new(flat), Arc::new(chunked)),
        }
    }
}

#[async_trait]
impl LayoutStrategy for SelfPacedLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype();
        let Some(fields) = dtype.as_struct_fields_opt() else {
            vortex_bail!("self-paced layout requires a Struct root, got {dtype}");
        };
        if dtype.is_nullable() {
            vortex_bail!("self-paced layout does not support nullable Struct roots");
        }
        let supported = DType::Primitive(PType::I64, Nullability::NonNullable);
        if let Some((index, field)) = fields
            .fields()
            .enumerate()
            .find(|(_, field)| *field != supported)
        {
            vortex_bail!("self-paced layout field {index} must be non-nullable i64, got {field}");
        }
        self.delegate
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }
}
