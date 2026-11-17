// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(target_arch = "wasm32"))]
#![expect(clippy::tests_outside_test_module)]

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::mpsc;
use std::task::Context;
use std::task::Poll;
use std::time::Duration;

use futures::StreamExt;
use futures::future;
use futures::stream;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Task;
use vortex_io::runtime::current::CurrentThreadRuntime;

struct Release(Option<oneshot::Sender<()>>);

impl Drop for Release {
    fn drop(&mut self) {
        if let Some(send) = self.0.take() {
            drop(send.send(()));
        }
    }
}

fn parent_join() {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = peer.handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    let parent = caller.handle().spawn(async move { drop(child) });
    caller.block_on(parent);
    assert_eq!(weak.strong_count(), 0);
}

fn live_parent() {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let (send, receive) = oneshot::channel();
    let release = Release(Some(send));
    let child = peer.handle().spawn(async move {
        let _release = release;
        future::pending::<()>().await;
    });
    let parent = caller.handle().spawn(async move {
        drop(child);
        assert!(receive.into_future().await.is_ok());
    });
    caller.block_on(parent);
}

fn last_iterator(started: bool) {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = peer.handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    let mut iterator = caller.block_on_stream_thread_safe(move |_| {
        stream::once(future::ready(1)).chain(stream::once(async move {
            let _child = child;
            future::pending::<usize>().await
        }))
    });
    if started {
        assert_eq!(iterator.next(), Some(1));
    }
    drop(iterator);
    assert_eq!(weak.strong_count(), 0);
}

fn nested_peers() {
    let runtimes: Vec<_> = (0..32).map(|_| CurrentThreadRuntime::new()).collect();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let mut child = runtimes[31].handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    for runtime in runtimes[..31].iter().rev() {
        child = runtime.handle().spawn(async move {
            let _child = child;
            future::pending::<()>().await;
        });
    }
    drop(child);
    assert_eq!(weak.strong_count(), 0);
}

fn idle_runtime_teardown() {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = peer.handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    let (send, receive) = oneshot::channel();
    let parent = caller.handle().spawn(async move {
        drop(child);
        assert!(send.send(()).is_ok());
        future::pending::<()>().await;
    });
    assert!(caller.block_on(receive.into_future()).is_ok());
    drop(caller);
    drop(peer);
    drop(parent);
    assert_eq!(weak.strong_count(), 0);
}

struct PeerQueue {
    parent: Option<Task<()>>,
    resource: Weak<()>,
    _unrelated: Vec<Task<()>>,
}

fn peer_queue(caller: &CurrentThreadRuntime, peer: &CurrentThreadRuntime) -> PeerQueue {
    let mut unrelated = Vec::new();
    for _ in 0..300 {
        unrelated.push(peer.handle().spawn(async {}));
    }
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = peer.handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    for _ in 0..700 {
        unrelated.push(peer.handle().spawn(async {}));
    }
    PeerQueue {
        parent: Some(caller.handle().spawn(async move {
            let _child = child;
            future::pending::<()>().await;
        })),
        resource: weak,
        _unrelated: unrelated,
    }
}

fn root_future_poll() {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let mut queued = peer_queue(&caller, &peer);
    let mut first = true;
    peer.block_on(future::poll_fn(|cx| {
        if first {
            first = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        drop(queued.parent.take());
        Poll::Ready(())
    }));
    assert_eq!(queued.resource.strong_count(), 0);
}

struct DropParent {
    parent: Option<Task<()>>,
    first: bool,
    panic: bool,
}

impl Future for DropParent {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.first {
            self.first = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        assert!(!self.panic, "root-future-panic");
        Poll::Ready(())
    }
}

impl Drop for DropParent {
    fn drop(&mut self) {
        drop(self.parent.take());
    }
}

fn root_future_drop(panic: bool) {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let mut queued = peer_queue(&caller, &peer);
    let root = DropParent {
        parent: queued.parent.take(),
        first: true,
        panic,
    };
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| peer.block_on(root)));
    if panic {
        assert!(result.is_err_and(|payload| {
            payload
                .downcast_ref::<String>()
                .is_some_and(|s| s.contains("root-future-panic"))
                || payload.downcast_ref::<&str>() == Some(&"root-future-panic")
        }));
    } else {
        assert!(result.is_ok());
    }
    assert_eq!(queued.resource.strong_count(), 0);
}

fn borrowed_stream_poll() {
    let caller = CurrentThreadRuntime::new();
    let peer = CurrentThreadRuntime::new();
    let mut queued = peer_queue(&caller, &peer);
    let mut first = true;
    let stream = stream::poll_fn(|cx| {
        if first {
            first = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        drop(queued.parent.take());
        Poll::Ready(Some(1))
    });
    let mut iterator = peer.block_on_stream(stream);
    assert_eq!(iterator.next(), Some(1));
    assert_eq!(queued.resource.strong_count(), 0);
}

fn run_case(name: &'static str, case: impl FnOnce() + Send + 'static) {
    let (send, receive) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        case();
        let _ = send.send(());
    });
    assert!(
        receive.recv_timeout(Duration::from_secs(10)).is_ok(),
        "cross-runtime cleanup did not complete: {name}"
    );
    assert!(thread.join().is_ok());
}

// A separate test binary lets the watchdog fail without retaining blocked test workers.
#[test]
fn cancellation_drives_idle_peer_runtimes() {
    run_case("normal parent join", parent_join);
    run_case("live parent", live_parent);
    run_case("unpolled last iterator", || last_iterator(false));
    run_case("started last iterator", || last_iterator(true));
    run_case("nested peer runtimes", nested_peers);
    run_case("idle runtime teardown", idle_runtime_teardown);
    run_case("root future poll", root_future_poll);
    run_case("root future destructor", || root_future_drop(false));
    run_case("root future unwind", || root_future_drop(true));
    run_case("borrowed stream poll", borrowed_stream_poll);
}
