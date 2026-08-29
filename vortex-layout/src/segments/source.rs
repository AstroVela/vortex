// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
pub use vortex_io::ReadAtNowait;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Attempt to resolve a segment synchronously without waiting on storage.
    ///
    /// Sources that cannot guarantee non-blocking behavior return
    /// [`ReadAtNowait::Unsupported`].
    fn request_nowait(&self, _id: SegmentId) -> VortexResult<ReadAtNowait> {
        Ok(ReadAtNowait::Unsupported)
    }

    /// Whether planned reads should be submitted to a background driver immediately.
    ///
    /// Remote and file-backed sources use this to overlap IO and expose neighboring requests to
    /// coalescing before execution asks for their bytes. In-memory sources should retain the
    /// default so execution can resolve them inline without scheduler round trips.
    fn prefers_background_reads(&self) -> bool {
        false
    }
}
