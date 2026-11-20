// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(target_arch = "wasm32"))]
#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::task::Poll;
use std::time::Duration;

use futures::FutureExt;
use futures::future;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_file::SegmentSpec;
use vortex_file::segments::FileSegmentSource;
use vortex_file::segments::RequestMetrics;
use vortex_io::VortexReadAt;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::runtime::current::CurrentThreadRuntime;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_metrics::DefaultMetricsRegistry;

struct PendingRead {
    started: Arc<AtomicBool>,
}
impl VortexReadAt for PendingRead {
    fn concurrency(&self) -> usize {
        1
    }
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        future::ready(Ok(4)).boxed()
    }
    fn read_at(
        &self,
        _: u64,
        _: usize,
        _: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        self.started.store(true, Ordering::SeqCst);
        future::pending().boxed()
    }
}

fn segments() -> Arc<[SegmentSpec]> {
    Arc::from([SegmentSpec {
        offset: 0,
        length: 4,
        alignment: Alignment::none(),
    }])
}
fn metrics() -> RequestMetrics {
    RequestMetrics::new(&DefaultMetricsRegistry::default(), vec![])
}

#[test]
fn idle_source_release_and_request_outlives_source() -> VortexResult<()> {
    let runtime = CurrentThreadRuntime::new();
    let reader = Arc::new(ByteBuffer::from(vec![1_u8, 2, 3, 4]));
    let weak = Arc::downgrade(&reader);
    let source = FileSegmentSource::open(segments(), reader, runtime.handle(), metrics());
    let request = source.request(SegmentId::from(0));
    drop(source);
    assert!(weak.upgrade().is_some());
    assert_eq!(
        runtime.block_on(request)?.try_into_host_sync()?.as_slice(),
        &[1, 2, 3, 4]
    );
    assert_eq!(weak.strong_count(), 0);
    Ok(())
}

#[test]
fn cancelling_inflight_async_request_releases_reader_without_extra_poll() {
    let runtime = CurrentThreadRuntime::new();
    let started = Arc::new(AtomicBool::new(false));
    let reader = Arc::new(PendingRead {
        started: Arc::clone(&started),
    });
    let weak = Arc::downgrade(&reader);
    let source = FileSegmentSource::open(segments(), reader, runtime.handle(), metrics());
    let mut request = source.request(SegmentId::from(0));
    runtime.block_on(future::poll_fn(|cx| {
        assert!(request.poll_unpin(cx).is_pending());
        if started.load(Ordering::SeqCst) {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }));
    drop(source);
    drop(request);
    let retained = weak.strong_count();
    assert_eq!(
        retained, 0,
        "cancelled in-flight read leaves reader in stopped executor"
    );
}

struct BlockingRead {
    lease: Arc<()>,
    runtime: Handle,
    started: Arc<AtomicBool>,
    release: Arc<Barrier>,
    finished: mpsc::Sender<()>,
}
impl VortexReadAt for BlockingRead {
    fn concurrency(&self) -> usize {
        1
    }
    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        future::ready(Ok(4)).boxed()
    }
    fn read_at(
        &self,
        _: u64,
        _: usize,
        _: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let lease = Arc::clone(&self.lease);
        let release = Arc::clone(&self.release);
        let finished = self.finished.clone();
        let started = Arc::clone(&self.started);
        let runtime = self.runtime.clone();
        async move {
            runtime
                .spawn_blocking(move || {
                    started.store(true, Ordering::SeqCst);
                    release.wait();
                    drop(lease);
                    assert!(finished.send(()).is_ok());
                    Ok(BufferHandle::new_host(ByteBuffer::from(vec![
                        1_u8, 2, 3, 4,
                    ])))
                })
                .await
        }
        .boxed()
    }
}

#[test]
fn completed_blocking_io_releases_reader_after_request_cancellation() {
    let runtime = CurrentThreadRuntime::new();
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let started = Arc::new(AtomicBool::new(false));
    let release = Arc::new(Barrier::new(2));
    let (finished, wait_finished) = mpsc::channel();
    let reader = Arc::new(BlockingRead {
        lease,
        runtime: runtime.handle(),
        started: Arc::clone(&started),
        release: Arc::clone(&release),
        finished,
    });
    let source = FileSegmentSource::open(segments(), reader, runtime.handle(), metrics());
    let mut request = source.request(SegmentId::from(0));
    runtime.block_on(future::poll_fn(|cx| {
        assert!(request.poll_unpin(cx).is_pending());
        if started.load(Ordering::SeqCst) {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }));
    drop(source);
    drop(request);
    assert!(
        weak.strong_count() > 0,
        "a live blocking read must own its lease"
    );
    release.wait();
    assert!(wait_finished.recv_timeout(Duration::from_secs(3)).is_ok());
    let retained = weak.strong_count();
    assert_eq!(
        retained, 0,
        "finished blocking I/O still needs an unrelated executor poll to release reader lease"
    );
}

#[test]
fn idle_source_and_unpolled_cancel_release_reader() {
    let runtime = CurrentThreadRuntime::new();
    let reader = Arc::new(ByteBuffer::from(vec![1_u8, 2, 3, 4]));
    let weak = Arc::downgrade(&reader);
    let source = FileSegmentSource::open(segments(), reader, runtime.handle(), metrics());
    drop(source);
    assert_eq!(weak.strong_count(), 0);

    let reader = Arc::new(ByteBuffer::from(vec![1_u8, 2, 3, 4]));
    let weak = Arc::downgrade(&reader);
    let source = FileSegmentSource::open(segments(), reader, runtime.handle(), metrics());
    let request = source.request(SegmentId::from(0));
    drop(source);
    assert!(weak.upgrade().is_some());
    drop(request);
    assert_eq!(weak.strong_count(), 0);
}
