// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

/// A future that resolves to a mask.
///
/// Masks that are known up-front (e.g. scans without a filter) are stored resolved, so awaiting,
/// cloning, and slicing them never allocates a boxed future.
#[derive(Clone)]
pub struct MaskFuture {
    inner: Inner,
    len: usize,
}

#[derive(Clone)]
enum Inner {
    /// The mask is already resolved.
    Ready(Mask),
    /// The mask is still being computed.
    Pending(Shared<BoxFuture<'static, SharedVortexResult<Mask>>>),
}

impl MaskFuture {
    /// Create a new MaskFuture from a future that returns a mask.
    pub fn new<F>(len: usize, fut: F) -> Self
    where
        F: Future<Output = VortexResult<Mask>> + Send + 'static,
    {
        Self {
            inner: Inner::Pending(
                fut.inspect(move |r| {
                    if let Ok(mask) = r
                        && mask.len() != len {
                            vortex_panic!("MaskFuture created with future that returned mask of incorrect length (expected {}, got {})", len, mask.len());
                        }
                })
                .map_err(Arc::new)
                .boxed()
                .shared(),
            ),
            len,
        }
    }

    /// Returns the length of the mask.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Create a MaskFuture from a ready mask.
    pub fn ready(mask: Mask) -> Self {
        Self {
            len: mask.len(),
            inner: Inner::Ready(mask),
        }
    }

    /// Create a MaskFuture that resolves to a mask with all values set to true.
    pub fn new_true(row_count: usize) -> Self {
        Self::ready(Mask::new_true(row_count))
    }

    /// Create a MaskFuture that resolves to a slice of the original mask.
    pub fn slice(&self, range: Range<usize>) -> Self {
        // Slicing the whole mask is the identity. Cloning shares the existing state instead of
        // allocating another boxed, shared future that would await it only to hand the mask back.
        if range.start == 0 && range.end == self.len {
            return self.clone();
        }

        match &self.inner {
            Inner::Ready(mask) => Self::ready(mask.slice(range)),
            Inner::Pending(inner) => {
                let inner = inner.clone();
                Self::new(range.len(), async move { Ok(inner.await?.slice(range)) })
            }
        }
    }

    /// Observe the resolved mask.
    ///
    /// An already-resolved mask calls `f` immediately rather than deferring it to the first poll.
    pub fn inspect(
        self,
        f: impl FnOnce(&SharedVortexResult<Mask>) + 'static + Send + Sync,
    ) -> Self {
        let len = self.len;

        match self.inner {
            Inner::Ready(mask) => {
                f(&Ok(mask.clone()));
                Self {
                    inner: Inner::Ready(mask),
                    len,
                }
            }
            Inner::Pending(inner) => Self {
                inner: Inner::Pending(inner.inspect(f).boxed().shared()),
                len,
            },
        }
    }
}

impl Future for MaskFuture {
    type Output = VortexResult<Mask>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        match &mut self.inner {
            Inner::Ready(mask) => std::task::Poll::Ready(Ok(mask.clone())),
            Inner::Pending(inner) => inner.poll_unpin(cx).map_err(VortexError::from),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;

    use vortex_buffer::BitBuffer;

    use super::*;

    /// Slicing resolves to the same mask the equivalent [`Mask::slice`] would produce, for both
    /// the full range (which takes the identity fast path) and a sub-range.
    #[test]
    fn slice_resolves_to_sliced_mask() -> VortexResult<()> {
        futures::executor::block_on(async {
            let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
            let fut = MaskFuture::ready(mask.clone());

            let full = fut.slice(0..mask.len());
            assert_eq!(full.len(), mask.len());
            assert_eq!(full.await?, mask);

            let partial = fut.slice(0..mask.len() - 1);
            assert_eq!(partial.len(), mask.len() - 1);
            assert_eq!(partial.await?, mask.slice(0..mask.len() - 1));
            Ok(())
        })
    }

    /// Ready and pending variants must behave identically through slice and await.
    #[test]
    fn pending_slice_matches_ready_slice() -> VortexResult<()> {
        futures::executor::block_on(async {
            let mask = Mask::from_buffer(BitBuffer::from_iter([true, false, true, true, false]));
            let ready = MaskFuture::ready(mask.clone());
            let pending = {
                let mask = mask.clone();
                MaskFuture::new(mask.len(), async move { Ok(mask) })
            };

            assert_eq!(ready.slice(1..4).await?, pending.slice(1..4).await?);
            assert_eq!(ready.clone().await?, pending.clone().await?);
            Ok(())
        })
    }

    /// Inspect fires for an already-resolved mask and preserves the resolved value.
    #[test]
    fn inspect_fires_on_ready_mask() -> VortexResult<()> {
        futures::executor::block_on(async {
            let mask = Mask::from_buffer(BitBuffer::from_iter([true, false]));
            let fired = Arc::new(AtomicBool::new(false));
            let fut = MaskFuture::ready(mask.clone()).inspect({
                let fired = Arc::clone(&fired);
                move |r| {
                    assert!(r.is_ok());
                    fired.store(true, Ordering::Relaxed);
                }
            });

            assert!(fired.load(Ordering::Relaxed));
            assert_eq!(fut.await?, mask);
            Ok(())
        })
    }
}
