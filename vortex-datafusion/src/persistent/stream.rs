// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::task::Context;
use std::task::Poll;

use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_pruning::FilePruner;
use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;

/// Utility to end a stream early when it reaches its limit or its backing
/// [`PartitionedFile`] can be pruned by an updated dynamic expression.
///
/// [`PartitionedFile`]: datafusion_datasource::PartitionedFile
pub(crate) struct PrunableStream {
    file_pruner: Option<FilePruner>,
    remaining: Option<usize>,
    stream: BoxStream<'static, DFResult<RecordBatch>>,
}

impl PrunableStream {
    pub fn new(
        file_pruner: Option<FilePruner>,
        limit: Option<usize>,
        stream: BoxStream<'static, DFResult<RecordBatch>>,
    ) -> Self {
        Self {
            file_pruner,
            remaining: limit,
            stream,
        }
    }
}

impl Stream for PrunableStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.remaining == Some(0) {
            Poll::Ready(None)
        } else if let Some(file_pruner) = self.file_pruner.as_mut()
            && file_pruner.should_prune()?
        {
            Poll::Ready(None)
        } else {
            match self.stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => match &mut self.remaining {
                    Some(remaining) if batch.num_rows() > *remaining => {
                        let batch = batch.slice(0, *remaining);
                        *remaining = 0;
                        Poll::Ready(Some(Ok(batch)))
                    }
                    Some(remaining) => {
                        *remaining -= batch.num_rows();
                        Poll::Ready(Some(Ok(batch)))
                    }
                    None => Poll::Ready(Some(Ok(batch))),
                },
                poll => poll,
            }
        }
    }
}
