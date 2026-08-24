// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;

#[derive(Clone)]
pub struct ChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
}

impl ChunkedLayoutStrategy {
    pub fn new<S: LayoutStrategy>(chunk_strategy: S) -> Self {
        Self {
            chunk_strategy: Arc::new(chunk_strategy),
        }
    }
}

#[async_trait]
impl LayoutStrategy for ChunkedLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let dtype2 = dtype.clone();
        let chunk_strategy = Arc::clone(&self.chunk_strategy);
        let handle = session.handle();
        let child_context = ctx.clone();

        // We spawn each child to allow parallelism when processing chunks.
        let stream = stream! {
            let mut stream = stream;
            while let Some(chunk) = stream.next().await {
                let chunk_eof = eof.split_off();

                let chunk_strategy = Arc::clone(&chunk_strategy);
                let (child_ctx, child_info) = child_context.child_context();
                let segment_sink = Arc::clone(&segment_sink);
                let dtype = dtype2.clone();
                let session = session.clone();

                yield handle.spawn_nested(move |handle| async move {
                    let session = session.with_handle(handle);
                    let layout = chunk_strategy
                        .write_stream(
                            child_ctx,
                            segment_sink,
                            SequentialStreamAdapter::new(
                                dtype,
                                stream::iter([chunk]),
                            )
                            .sendable(),
                            chunk_eof,
                            &session,
                        )
                        .await?;
                    Ok::<_, vortex_error::VortexError>((layout, child_info))
                })
            }
        };

        // Poll all of our children concurrently to accumulate their layouts.
        let child_results: Vec<_> = stream.buffered(usize::MAX).try_collect().await?;
        let mut child_layouts = Vec::with_capacity(child_results.len());
        let mut chunk_boundaries = Vec::new();
        let mut row_offset = 0;

        for (layout, child_info) in child_results {
            let row_count = layout.row_count();
            chunk_boundaries.extend(
                child_info
                    .chunk_boundaries()
                    .into_iter()
                    .filter(|&boundary| boundary != 0 && boundary < row_count)
                    .map(|boundary| row_offset + boundary),
            );
            row_offset += row_count;
            child_layouts.push(layout);
        }
        chunk_boundaries.extend(
            child_layouts
                .iter()
                .map(|layout| layout.row_count())
                .scan(0, |row_offset, row_count| {
                    *row_offset += row_count;
                    Some(*row_offset)
                })
                .take(child_layouts.len().saturating_sub(1)),
        );
        ctx.report_chunk_boundaries(chunk_boundaries);

        if child_layouts.len() == 1 {
            Ok(child_layouts.pop().vortex_expect("must have one child"))
        } else {
            let row_count = child_layouts.iter().map(|layout| layout.row_count()).sum();
            Ok(ChunkedLayout::new(
                row_count,
                dtype,
                OwnedLayoutChildren::layout_children(child_layouts),
            )
            .into_layout())
        }
    }
}
