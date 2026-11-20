// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg(not(target_arch = "wasm32"))]
#![expect(clippy::tests_outside_test_module)]

use std::alloc::GlobalAlloc;
use std::alloc::Layout;
use std::alloc::System;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use futures::future;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::current::CurrentThreadRuntime;

struct CountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every allocation operation forwards the caller's unchanged pointer and layout to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: GlobalAlloc's caller supplies a valid layout; System receives that same layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: The caller's pointer and its original layout are forwarded without modification.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller's live pointer, original layout and requested size pass through unchanged.
        let next = unsafe { System.realloc(pointer, layout, size) };
        if !next.is_null() {
            if size >= layout.size() {
                LIVE_BYTES.fetch_add(size - layout.size(), Ordering::Relaxed);
            } else {
                LIVE_BYTES.fetch_sub(layout.size() - size, Ordering::Relaxed);
            }
        }
        next
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

// This binary has one test so other test cases cannot perturb the live-allocation measurements.
#[test]
fn completed_cancellations_do_not_accumulate_in_a_live_parent() {
    let runtime = CurrentThreadRuntime::new();
    let finished = Arc::new(AtomicBool::new(false));
    let parent_finished = Arc::clone(&finished);
    let warm_bytes = Arc::new(AtomicUsize::new(0));
    let parent_warm_bytes = Arc::clone(&warm_bytes);
    let end_bytes = Arc::new(AtomicUsize::new(0));
    let parent_end_bytes = Arc::clone(&end_bytes);
    let handle = runtime.handle();
    let parent = runtime.handle().spawn(async move {
        for count in 1..=100_000 {
            drop(handle.spawn(future::pending::<()>()));
            smol::future::yield_now().await;
            if count == 10_000 {
                parent_warm_bytes.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
            }
        }
        parent_end_bytes.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
        parent_finished.store(true, Ordering::Release);
        future::pending::<()>().await;
    });
    runtime.block_on(future::poll_fn(|cx| {
        if finished.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }));
    let warmed = warm_bytes.load(Ordering::Relaxed);
    let retained = end_bytes.load(Ordering::Relaxed);
    let growth = retained.saturating_sub(warmed);
    drop(parent);
    assert!(
        growth <= 256 * 1024,
        "completed cancellations retained allocations while the parent stayed alive: {warmed} -> {retained}"
    );
}
