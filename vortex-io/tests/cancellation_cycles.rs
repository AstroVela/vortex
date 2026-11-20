// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(target_arch = "wasm32"))]
#![expect(clippy::tests_outside_test_module)]

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;

use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Task;
use vortex_io::runtime::current::CurrentThreadRuntime;

fn completed(receive: &mpsc::Receiver<()>, count: usize) {
    for _ in 0..count {
        assert!(
            receive.recv_timeout(Duration::from_secs(10)).is_ok(),
            "running tasks remained blocked while cancelling each other"
        );
    }
}

fn parent_and_running_child() {
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(2);
    let (send_parent, receive_parent) = oneshot::channel::<Task<()>>();
    let (done, receive_done) = mpsc::channel();
    let barrier = Arc::new(Barrier::new(2));
    let handle = runtime.handle();
    let parent = runtime.handle().spawn(async move {
        let Ok(parent) = receive_parent.into_future().await else {
            return;
        };
        let child_barrier = Arc::clone(&barrier);
        let child_done = done.clone();
        let child = handle.spawn(async move {
            child_barrier.wait();
            drop(parent);
            assert!(child_done.send(()).is_ok());
        });
        barrier.wait();
        drop(child);
        assert!(done.send(()).is_ok());
    });
    assert!(send_parent.send(parent).is_ok());
    completed(&receive_done, 2);
    pool.set_workers(0);
}

fn parent_blocking_on_child() {
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(2);
    let (send_parent, receive_parent) = oneshot::channel::<Task<()>>();
    let (done, receive_done) = mpsc::channel();
    let barrier = Arc::new(Barrier::new(2));
    let handle = runtime.handle();
    let nested = runtime.clone();
    let parent = runtime.handle().spawn(async move {
        let Ok(parent) = receive_parent.into_future().await else {
            return;
        };
        let child_barrier = Arc::clone(&barrier);
        let child_done = done.clone();
        let child = handle.spawn(async move {
            child_barrier.wait();
            drop(parent);
            assert!(child_done.send(()).is_ok());
        });
        barrier.wait();
        nested.block_on(child);
        assert!(done.send(()).is_ok());
    });
    assert!(send_parent.send(parent).is_ok());
    completed(&receive_done, 2);
    pool.set_workers(0);
}

fn peers_on_different_runtimes() {
    let left = CurrentThreadRuntime::new();
    let right = CurrentThreadRuntime::new();
    let left_pool = left.new_pool();
    let right_pool = right.new_pool();
    left_pool.set_workers(1);
    right_pool.set_workers(1);
    let (send_left, receive_left) = oneshot::channel::<Task<()>>();
    let (send_right, receive_right) = oneshot::channel::<Task<()>>();
    let (done, receive_done) = mpsc::channel();
    let barrier = Arc::new(Barrier::new(2));
    let right_barrier = Arc::clone(&barrier);
    let right_done = done.clone();
    let left_task = left.handle().spawn(async move {
        let Ok(right) = receive_right.into_future().await else {
            return;
        };
        barrier.wait();
        drop(right);
        assert!(done.send(()).is_ok());
    });
    let right_task = right.handle().spawn(async move {
        let Ok(left) = receive_left.into_future().await else {
            return;
        };
        right_barrier.wait();
        drop(left);
        assert!(right_done.send(()).is_ok());
    });
    assert!(send_left.send(left_task).is_ok());
    assert!(send_right.send(right_task).is_ok());
    completed(&receive_done, 2);
    left_pool.set_workers(0);
    right_pool.set_workers(0);
}

struct CycleMember {
    next: Option<Task<()>>,
    done: mpsc::Sender<()>,
}

impl Drop for CycleMember {
    fn drop(&mut self) {
        drop(self.next.take());
        let _ = self.done.send(());
    }
}

fn transitive_cycle_with_two_workers() {
    let runtime = CurrentThreadRuntime::new();
    let pool = runtime.new_pool();
    pool.set_workers(2);
    let (done, receive_done) = mpsc::channel();
    let started = Arc::new(AtomicUsize::new(0));
    let mut senders = Vec::new();
    let mut tasks = Vec::new();
    for _ in 0..3 {
        let (send, receive) = oneshot::channel::<Task<()>>();
        senders.push(send);
        let done = done.clone();
        let started = Arc::clone(&started);
        tasks.push(runtime.handle().spawn(async move {
            let Ok(next) = receive.into_future().await else {
                return;
            };
            let member = CycleMember {
                next: Some(next),
                done,
            };
            started.fetch_add(1, Ordering::AcqRel);
            while started.load(Ordering::Acquire) != 3 {
                smol::future::yield_now().await;
            }
            drop(member);
        }));
    }
    tasks.rotate_left(1);
    for (send, task) in senders.into_iter().zip(tasks) {
        assert!(send.send(task).is_ok());
    }
    completed(&receive_done, 3);
    pool.set_workers(0);
}

// Isolation keeps watchdog failure from leaving blocked workers in another regression's process.
#[test]
fn cancellation_cycles_finish_across_workers_and_runtimes() {
    parent_and_running_child();
    parent_blocking_on_child();
    peers_on_different_runtimes();
    transitive_cycle_with_two_workers();
}
