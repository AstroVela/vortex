// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(target_arch = "wasm32"))]
#![expect(clippy::tests_outside_test_module)]

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;
use std::thread;

use futures::future;
use futures::stream;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Task;
use vortex_io::runtime::current::CurrentThreadRuntime;

#[test]
fn direct_resource_drop() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::once(async move {
            let _resource = resource;
            future::pending::<usize>().await
        })
    });
    drop(iterator);
    assert_eq!(weak.strong_count(), 0);
}

#[test]
fn partial_and_concurrent_final_drop() {
    let runtime = CurrentThreadRuntime::new();
    for _ in 0..64 {
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::unfold(resource, |resource| async move { Some((42, resource)) })
        });
        assert_eq!(iterator.next(), Some(42));
        let other = iterator.clone();
        let barrier = Arc::new(Barrier::new(2));
        let worker_barrier = Arc::clone(&barrier);
        let worker = thread::spawn(move || {
            worker_barrier.wait();
            drop(other);
        });
        barrier.wait();
        drop(iterator);
        assert!(worker.join().is_ok());
        assert_eq!(weak.strong_count(), 0);
    }
}

#[test]
fn completion_and_panic_propagation() {
    let runtime = CurrentThreadRuntime::new();
    let iterator = runtime.block_on_stream_thread_safe(|_| stream::iter([1, 2, 3]));
    assert_eq!(iterator.collect::<Vec<_>>(), vec![1, 2, 3]);
    let mut iterator = runtime.block_on_stream_thread_safe(|_| {
        stream::once(async {
            panic!("stream panic payload");
            #[allow(unreachable_code)]
            0
        })
    });
    let result = catch_unwind(AssertUnwindSafe(|| iterator.next()));
    assert!(result.is_err());
    drop(iterator);
}

#[test]
fn reentrant_drop_from_another_executor_task() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::once(async move {
            let _held = resource;
            future::pending::<usize>().await
        })
    });
    let dropping_task = runtime.handle().spawn(async move {
        drop(iterator);
    });
    runtime.block_on(dropping_task);
    assert_eq!(weak.strong_count(), 0);
}

#[test]
fn pending_children_are_released_before_iterator_drop_returns() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let started = Arc::new(AtomicUsize::new(0));
    let mut children = Vec::new();
    for _ in 0..512 {
        let held = Arc::clone(&resource);
        let started = Arc::clone(&started);
        children.push(runtime.handle().spawn(async move {
            let _held = held;
            started.fetch_add(1, Ordering::SeqCst);
            future::pending::<()>().await;
        }));
    }
    drop(resource);
    let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::unfold(children, |children| async move { Some((42, children)) })
    });
    assert_eq!(iterator.next(), Some(42));
    assert_eq!(started.load(Ordering::SeqCst), 512);
    drop(iterator);
    let retained = weak.strong_count();
    assert_eq!(
        retained, 0,
        "iterator drop returned before its child task resources were released"
    );
}

#[test]
fn nested_children_are_released_before_iterator_drop_returns() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let mut child = runtime.handle().spawn(async move {
        let _held = resource;
        future::pending::<()>().await;
    });
    for _ in 0..400 {
        child = runtime.handle().spawn(async move {
            let _child = child;
            future::pending::<()>().await;
        });
    }
    let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::unfold(child, |child| async move { Some((42, child)) })
    });
    assert_eq!(iterator.next(), Some(42));
    drop(iterator);
    let retained = weak.strong_count();
    assert_eq!(
        retained, 0,
        "iterator drop returned before nested cancellation completed"
    );
}

#[test]
fn one_child_with_unrelated_executor_work_releases_before_drop_returns() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = runtime.handle().spawn(async move {
        let _held = resource;
        future::pending::<()>().await;
    });
    let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::unfold((child, false), |(child, emitted)| async move {
            if emitted {
                future::pending::<()>().await;
            }
            Some((42, (child, true)))
        })
    });
    assert_eq!(iterator.next(), Some(42));
    let unrelated = (0..199)
        .map(|_| runtime.handle().spawn(future::pending::<()>()))
        .collect::<Vec<_>>();
    drop(iterator);
    let retained = weak.strong_count();
    drop(unrelated);
    assert_eq!(
        retained, 0,
        "unrelated ready tasks can defer the only child's cancellation"
    );
}

#[test]
fn early_drop_after_buffered_item_does_not_raise_background_panic() {
    let runtime = CurrentThreadRuntime::new();
    let mut iterator = runtime.block_on_stream_thread_safe(|_| {
        stream::unfold(false, |emitted| async move {
            assert!(!emitted, "background stream failed after buffered item");
            Some((42, true))
        })
    });
    assert_eq!(iterator.next(), Some(42));
    let outcome = catch_unwind(AssertUnwindSafe(|| drop(iterator)));
    assert!(outcome.is_ok(), "cleanup re-raised a stored stream panic");
}

#[test]
fn cancelled_child_destructor_drops_nested_thread_safe_iterator() {
    let runtime = CurrentThreadRuntime::new();
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let inner = runtime.block_on_stream_thread_safe(move |_| {
        stream::once(async move {
            let _held = resource;
            future::pending::<usize>().await
        })
    });
    let child = runtime.handle().spawn(async move {
        let _inner = inner;
        future::pending::<()>().await;
    });
    let mut outer = runtime.block_on_stream_thread_safe(move |_| {
        stream::unfold(child, |child| async move { Some((42, child)) })
    });
    assert_eq!(outer.next(), Some(42));
    drop(outer);
    assert_eq!(weak.strong_count(), 0);
}

#[test]
fn early_drop_during_consumer_unwind_keeps_the_consumer_panic() {
    let runtime = CurrentThreadRuntime::new();
    let mut iterator = runtime.block_on_stream_thread_safe(|_| {
        stream::unfold(false, |emitted| async move {
            assert!(!emitted, "background stream panic");
            Some((42, true))
        })
    });
    assert_eq!(iterator.next(), Some(42));
    let outcome = catch_unwind(AssertUnwindSafe(move || {
        let _iterator = iterator;
        panic!("consumer panic");
    }));
    let panic = outcome
        .err()
        .and_then(|panic| panic.downcast::<&str>().ok());
    assert_eq!(panic.as_deref().copied(), Some("consumer panic"));
}

#[test]
fn cancellation_finishes_with_unrelated_self_waking_work() {
    let runtime = CurrentThreadRuntime::new();
    let unrelated = runtime.handle().spawn(future::poll_fn(|cx| {
        cx.waker().wake_by_ref();
        Poll::<()>::Pending
    }));
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    let child = runtime.handle().spawn(async move {
        let _resource = resource;
        future::pending::<()>().await;
    });
    let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
        stream::unfold(child, |child| async move { Some((42, child)) })
    });
    assert_eq!(iterator.next(), Some(42));
    drop(iterator);
    assert_eq!(weak.strong_count(), 0);
    drop(unrelated);
}

#[test]
fn a_task_can_drop_its_own_handle() {
    let runtime = CurrentThreadRuntime::new();
    let finished = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&finished);
    let (send, receive) = oneshot::channel::<Task<()>>();
    let task = runtime.handle().spawn(async move {
        if let Ok(task) = receive.into_future().await {
            drop(task);
            observed.store(1, Ordering::SeqCst);
        }
    });
    assert!(send.send(task).is_ok());
    runtime.block_on(future::poll_fn(|cx| {
        if finished.load(Ordering::SeqCst) == 1 {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }));
}

#[test]
fn worker_pool_cancellation_releases_descendants_before_returning() {
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(2);
    for _ in 0..16 {
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let children = (0..512)
            .map(|_| {
                let resource = Arc::clone(&resource);
                runtime.handle().spawn(async move {
                    let _resource = resource;
                    future::pending::<()>().await;
                })
            })
            .collect::<Vec<_>>();
        drop(resource);
        let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::unfold(children, |children| async move { Some((42, children)) })
        });
        assert_eq!(iterator.next(), Some(42));
        drop(iterator);
        assert_eq!(weak.strong_count(), 0);
    }
}
