// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::future::join_all;
use futures::future::poll_fn;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use smol::block_on;
use vortex_error::VortexExpect;
use vortex_utils::parallelism::get_available_parallelism;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::BlockingRuntime;
use crate::runtime::Executor;
use crate::runtime::Handle;
pub use crate::runtime::pool::CurrentThreadWorkerPool;

/// A current thread runtime allows callers to much more explicitly drive Vortex futures than with
/// a Tokio runtime.
///
/// Blocking calls drive work on their calling thread. Cancelling owned tasks also drives their
/// cleanup and the cancellation of child tasks before the outermost cleanup returns.
/// Nested task cancellation registers its completion so the calling task can finish its poll.
/// The outer close boundary joins these completions and their descendants.
/// Cancelling an active ancestor requests cancellation without joining its current poll,
/// including when the dependency chain runs across worker threads.
/// Started blocking I/O retains its own resources until the operation finishes.
///
/// It's also possible to clone the runtime onto other threads, each of which can call `block_on`
/// to drive work on that thread. Each thread shares the same underlying executor with the same
/// set of tasks, allowing work to be driven in parallel.
///
/// For automatic driving of work, a [`CurrentThreadWorkerPool`] can be created from the runtime
/// by calling [`new_pool`](CurrentThreadRuntime::new_pool). The returned pool can be configured
/// with the desired number of worker threads that will drive work on behalf of the runtime.
#[derive(Clone)]
pub struct CurrentThreadRuntime {
    executor: Arc<CurrentThreadExecutor>,
}

struct CurrentThreadExecutor {
    inner: Arc<smol::Executor<'static>>,
    weak: Weak<CurrentThreadExecutor>,
}

impl Default for CurrentThreadRuntime {
    fn default() -> Self {
        Self {
            executor: Arc::new_cyclic(|weak| CurrentThreadExecutor {
                inner: Arc::new(smol::Executor::new()),
                weak: Weak::clone(weak),
            }),
        }
    }
}

thread_local! {
    static CANCELLATIONS: RefCell<Cancellations> = RefCell::new(Cancellations::default());
    static EXECUTING_TASKS: RefCell<Vec<Arc<TaskCleanup>>> = const { RefCell::new(Vec::new()) };
    static DRIVING_CLEANUP: Cell<bool> = const { Cell::new(false) };
}

static CANCELLATION_GRAPH: Mutex<()> = Mutex::new(());

struct TaskCleanup {
    executor: Weak<smol::Executor<'static>>,
    children: Mutex<Vec<ChildCleanup>>,
}

impl TaskCleanup {
    fn new(executor: Weak<smol::Executor<'static>>) -> Self {
        Self {
            executor,
            children: Mutex::new(Vec::new()),
        }
    }
}

struct CleanupDrive;

impl CleanupDrive {
    fn enter() -> Option<Self> {
        DRIVING_CLEANUP.with(|driving| {
            if driving.replace(true) {
                None
            } else {
                Some(Self)
            }
        })
    }
}

impl Drop for CleanupDrive {
    fn drop(&mut self) {
        DRIVING_CLEANUP.with(|driving| driving.set(false));
    }
}

#[derive(Clone)]
struct ChildCleanup {
    token: Weak<TaskCleanup>,
    completion: Shared<BoxFuture<'static, ()>>,
}

async fn join_child(mut child: ChildCleanup, driving: Weak<smol::Executor<'static>>) {
    poll_fn(|cx| {
        if Pin::new(&mut child.completion).poll(cx).is_ready() {
            return Poll::Ready(());
        }
        // A peer runtime may have no worker. Borrow its executor only for this poll.
        let target = child
            .token
            .upgrade()
            .map(|token| Weak::clone(&token.executor));
        if let Some(target) = target {
            if Weak::ptr_eq(&target, &driving) {
                return Poll::Pending;
            }
            let Some(executor) = target.upgrade() else {
                return Poll::Pending;
            };
            if let Some(_drive) = CleanupDrive::enter() {
                if executor.try_tick() {
                    cx.waker().wake_by_ref();
                }
            } else {
                // An outer tick drives this runnable; defer further executor re-entry.
                cx.waker().wake_by_ref();
            }
        }
        Poll::Pending
    })
    .await;
}

fn depends_on(start: &Arc<TaskCleanup>, target: &Arc<TaskCleanup>) -> bool {
    let mut pending = vec![Arc::clone(start)];
    let mut visited = BTreeSet::new();
    while let Some(current) = pending.pop() {
        if Arc::ptr_eq(&current, target) {
            return true;
        }
        if !visited.insert(Arc::as_ptr(&current)) {
            continue;
        }
        pending.extend(
            current
                .children
                .lock()
                .iter()
                .filter(|child| child.completion.peek().is_none())
                .filter_map(|child| child.token.upgrade()),
        );
    }
    false
}

fn retire_receipt(parent: Option<&Weak<TaskCleanup>>, completion: &Shared<BoxFuture<'static, ()>>) {
    if let Some(parent) = parent.and_then(Weak::upgrade) {
        parent
            .children
            .lock()
            .retain(|child| !child.completion.ptr_eq(completion));
    }
}

#[derive(Default)]
struct Cancellations {
    draining: bool,
    joins: VecDeque<Box<dyn FnOnce()>>,
}

struct ExecutingTask;

impl ExecutingTask {
    fn enter(token: &Arc<TaskCleanup>) -> Self {
        EXECUTING_TASKS.with(|tasks| tasks.borrow_mut().push(Arc::clone(token)));
        Self
    }
}

impl Drop for ExecutingTask {
    fn drop(&mut self) {
        EXECUTING_TASKS.with(|tasks| tasks.borrow_mut().pop());
    }
}

struct CurrentThreadFuture<F: Future + ?Sized> {
    future: Option<Pin<Box<F>>>,
    token: Arc<TaskCleanup>,
}

impl<F: Future + ?Sized> Future for CurrentThreadFuture<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let _executing = ExecutingTask::enter(&this.token);
        this.future
            .as_mut()
            .vortex_expect("Current-thread task future is present until drop")
            .as_mut()
            .poll(cx)
    }
}

impl<F: Future + ?Sized> Drop for CurrentThreadFuture<F> {
    fn drop(&mut self) {
        let _executing = ExecutingTask::enter(&self.token);
        drop(self.future.take());
    }
}

struct RootCleanup {
    executor: Arc<CurrentThreadExecutor>,
    token: Arc<TaskCleanup>,
}

impl Drop for RootCleanup {
    fn drop(&mut self) {
        cancel_and_join(Arc::clone(&self.executor), None, Arc::clone(&self.token));
    }
}

impl CurrentThreadExecutor {
    fn block_on<F: Future>(self: &Arc<Self>, future: F) -> F::Output {
        let cleanup = RootCleanup {
            executor: Arc::clone(self),
            token: Arc::new(TaskCleanup::new(Arc::downgrade(&self.inner))),
        };
        let future = CurrentThreadFuture {
            future: Some(Box::pin(future)),
            token: Arc::clone(&cleanup.token),
        };
        // Root future destruction registers children too. Join after run() has also
        // returned its prefetched local tasks, including during panic unwinding.
        let output = block_on(self.inner.run(future));
        drop(cleanup);
        output
    }
}

impl Executor for CurrentThreadExecutor {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        let token = Arc::new(TaskCleanup::new(Arc::downgrade(&self.inner)));
        let task = self.inner.spawn(CurrentThreadFuture {
            future: Some(future),
            token: Arc::clone(&token),
        });
        Box::new(CurrentThreadAbortHandle {
            task: Some(task),
            executor: Weak::clone(&self.weak),
            token,
        })
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.spawn(async move { task() }.boxed())
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // Blocking I/O owns its live handle and can finish independently of this executor.
        self.inner.spawn_blocking_io(task)
    }
}

struct CurrentThreadAbortHandle {
    task: Option<smol::Task<()>>,
    executor: Weak<CurrentThreadExecutor>,
    token: Arc<TaskCleanup>,
}

impl AbortHandle for CurrentThreadAbortHandle {
    fn abort(mut self: Box<Self>) {
        let task = self.task.take().vortex_expect("Abort handle owns its task");
        if let Some(executor) = self.executor.upgrade() {
            cancel_and_join(executor, Some(task), Arc::clone(&self.token));
        }
        // Dropping the task also requests cancellation when its own poll or destructor is active.
    }
}

impl Drop for CurrentThreadAbortHandle {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.detach();
        }
    }
}

fn cancel_and_join(
    executor: Arc<CurrentThreadExecutor>,
    task: Option<smol::Task<()>>,
    token: Arc<TaskCleanup>,
) {
    let executing = EXECUTING_TASKS.with(|tasks| {
        tasks
            .borrow()
            .iter()
            .any(|active| Arc::ptr_eq(active, &token))
    });
    if executing {
        // A task can cancel its own handle while it is polling or releasing its future.
        drop(task);
        return;
    }
    let cleanup_token = Arc::clone(&token);
    let future = async move {
        if let Some(task) = task {
            // Normal next() still propagates a driver panic. Cleanup consumes a late panic.
            drop(AssertUnwindSafe(task.cancel()).catch_unwind().await);
        }
        // Keep pending links visible while joining so another worker can detect a cycle.
        let children = cleanup_token.children.lock().clone();
        join_all(
            children
                .into_iter()
                .map(|child| join_child(child, Weak::clone(&cleanup_token.executor))),
        )
        .await;
    }
    .boxed();
    let cancellation = CurrentThreadFuture {
        future: Some(future),
        token: Arc::clone(&token),
    }
    .boxed()
    .shared();
    let parent = EXECUTING_TASKS.with(|tasks| tasks.borrow().last().map(Arc::clone));
    let cyclic = if let Some(parent) = &parent {
        // Serialize graph additions so concurrent workers cannot both introduce a cycle.
        let _graph = CANCELLATION_GRAPH.lock();
        if depends_on(&token, parent) {
            true
        } else {
            parent.children.lock().push(ChildCleanup {
                token: Arc::downgrade(&token),
                completion: cancellation.clone(),
            });
            false
        }
    } else {
        false
    };
    if cyclic {
        // An ancestor on another worker cannot be joined from its own dependency chain.
        // Dropping the unpolled cancellation still requests cancellation of its raw task.
        return;
    }
    let parent = parent.as_ref().map(Arc::downgrade);
    // Request cancellation immediately, including when another cancellation is being joined.
    // Keep the original Shared unpolled for receipt identity comparison after completion.
    let ready = {
        let mut request = cancellation.clone();
        let mut context = Context::from_waker(futures::task::noop_waker_ref());
        Pin::new(&mut request).poll(&mut context).is_ready()
    };
    if ready {
        retire_receipt(parent.as_ref(), &cancellation);
        return;
    }
    if parent.is_some() {
        // Progress on the active caller's runtime, including peers with no background worker.
        let caller = parent
            .as_ref()
            .and_then(Weak::upgrade)
            .and_then(|token| token.executor.upgrade());
        if let Some(caller) = caller {
            let driving = Arc::downgrade(&caller);
            let child = ChildCleanup {
                token: Arc::downgrade(&token),
                completion: cancellation.clone(),
            };
            caller
                .spawn(async move {
                    join_child(child, driving).await;
                    retire_receipt(parent.as_ref(), &cancellation);
                })
                .detach();
        }
        return;
    }
    let drain = CANCELLATIONS.with(|cancellations| {
        let mut cancellations = cancellations.borrow_mut();
        cancellations.joins.push_back(Box::new(move || {
            block_on(executor.inner.run(cancellation.clone()));
            // Completed receipts must not accumulate in a long-lived parent.
            retire_receipt(parent.as_ref(), &cancellation);
        }));
        !std::mem::replace(&mut cancellations.draining, true)
    });
    if drain {
        loop {
            let join =
                CANCELLATIONS.with(|cancellations| cancellations.borrow_mut().joins.pop_front());
            match join {
                Some(join) => join(),
                None => break,
            }
        }
        CANCELLATIONS.with(|cancellations| cancellations.borrow_mut().draining = false);
    }
}

impl CurrentThreadRuntime {
    /// Create a new current thread runtime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new worker pool for driving the runtime in the background.
    ///
    /// This pool can be used to offload work from the current thread to a set of worker threads
    /// that will drive the runtime's executor.
    ///
    /// By default, the pool has no worker threads; the caller must set the desired number of
    /// worker threads using the `set_workers` method on the returned pool.
    pub fn new_pool(&self) -> CurrentThreadWorkerPool {
        CurrentThreadWorkerPool::new(Arc::clone(&self.executor.inner))
    }

    /// Returns an iterator wrapper around a stream, blocking the current thread for each item.
    ///
    /// ## Multi-threaded Usage
    ///
    /// To drive the iterator from multiple threads, simply clone it and call `next()` on each
    /// clone. Results on each thread are ordered with respect to the stream, but there is no
    /// ordering guarantee between threads.
    pub fn block_on_stream_thread_safe<F, S, R>(&self, f: F) -> ThreadSafeIterator<R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + 'static,
        R: Send + 'static,
    {
        let stream = f(self.handle());

        // We create an MPMC result channel and spawn a task to drive the stream and send results.
        // This allows multiple worker threads to drive the execution while all waiting for results
        // on the channel. Channel buffers up to one item per core so calling threads can get a
        // ready item without every one of them having to become an executor.
        let capacity = get_available_parallelism().unwrap_or(1).max(1);
        let (result_tx, result_rx) = kanal::bounded_async(capacity);
        let token = Arc::new(TaskCleanup::new(Arc::downgrade(&self.executor.inner)));
        let future = async move {
            futures::pin_mut!(stream);
            while let Some(item) = stream.next().await {
                // If all receivers are dropped, we stop driving the stream.
                if let Err(e) = result_tx.send(item).await {
                    tracing::trace!("all receivers dropped, stopping stream: {}", e);
                    break;
                }
            }
        }
        .boxed();
        let driver = self.executor.inner.spawn(CurrentThreadFuture {
            future: Some(future),
            token: Arc::clone(&token),
        });

        ThreadSafeIterator {
            executor: Arc::clone(&self.executor),
            results: result_rx,
            driver: Arc::new(StreamDriver {
                executor: Arc::clone(&self.executor),
                task: Mutex::new(Some(driver)),
                token,
            }),
        }
    }
}

impl BlockingRuntime for CurrentThreadRuntime {
    type BlockingIterator<'a, R: 'a> = CurrentThreadIterator<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = Arc::clone(&self.executor) as Arc<dyn Executor>;
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        self.executor.block_on(fut)
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        CurrentThreadIterator {
            executor: Arc::clone(&self.executor),
            stream: stream.boxed(),
        }
    }
}

/// An iterator that wraps up a stream to drive it using the current thread execution.
pub struct CurrentThreadIterator<'a, T> {
    executor: Arc<CurrentThreadExecutor>,
    stream: BoxStream<'a, T>,
}

impl<T> Iterator for CurrentThreadIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.executor.block_on(self.stream.next())
    }
}

/// An iterator that drives a stream from multiple threads.
/// Dropping the last clone cancels and joins the task driving its stream.
pub struct ThreadSafeIterator<T> {
    executor: Arc<CurrentThreadExecutor>,
    results: kanal::AsyncReceiver<T>,
    /// Handle to the task driving the stream. Once the stream ends, the first consumer to
    /// observe it joins the task so a panic raised while driving the stream is re-raised rather
    /// than silently ending the iterator.
    driver: Arc<StreamDriver>,
}

struct StreamDriver {
    executor: Arc<CurrentThreadExecutor>,
    task: Mutex<Option<smol::Task<()>>>,
    token: Arc<TaskCleanup>,
}

impl Drop for StreamDriver {
    fn drop(&mut self) {
        cancel_and_join(
            Arc::clone(&self.executor),
            self.task.get_mut().take(),
            Arc::clone(&self.token),
        );
    }
}

// Manual clone implementation since `T` does not need to be `Clone`.
impl<T> Clone for ThreadSafeIterator<T> {
    fn clone(&self) -> Self {
        Self {
            executor: Arc::clone(&self.executor),
            results: self.results.clone(),
            driver: Arc::clone(&self.driver),
        }
    }
}

impl<T> Iterator for ThreadSafeIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // If driver already has an item, take it without driving the executor
        match self.results.try_recv() {
            Ok(Some(item)) => return Some(item),
            Ok(None) => {}
            Err(_) => return self.get_task_error(),
        }

        match self.executor.block_on(self.results.recv()) {
            Ok(item) => Some(item),
            Err(_) => self.get_task_error(),
        }
    }
}

impl<T> ThreadSafeIterator<T> {
    // Join current task so panics are re-raised
    fn get_task_error(&self) -> Option<T> {
        let task = self.driver.task.lock().take();
        if let Some(task) = task {
            self.executor.block_on(task);
        }
        None
    }
}

#[expect(clippy::if_then_some_else_none)] // Clippy is wrong when if/else has await.
#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Poll;
    use std::thread;
    use std::time::Duration;

    use futures::StreamExt;
    use futures::stream;
    use parking_lot::Mutex;

    use super::*;

    #[test]
    fn test_worker_thread() {
        let runtime = CurrentThreadRuntime::new();

        // We spawn a future that sets a value on a separate thread.
        let value = Arc::new(AtomicUsize::new(0));
        let value2 = Arc::clone(&value);
        runtime
            .handle()
            .spawn(async move {
                value2.store(42, Ordering::SeqCst);
            })
            .detach();

        // By default, nothing has driven the executor, so the value should still be 0.
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // An empty pool still does nothing.
        let pool = runtime.new_pool();
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // Adding a worker thread should drive the executor.
        pool.set_workers(1);
        for _ in 0..10 {
            if value.load(Ordering::SeqCst) == 42 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(value.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_block_on_stream_single_thread() {
        let mut iter =
            CurrentThreadRuntime::new().block_on_stream(stream::iter(vec![1, 2, 3, 4, 5]).boxed());

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_thread_safe_iterator_drop_releases_unpolled_stream() {
        let runtime = CurrentThreadRuntime::new();
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::once(async move {
                let _resource = resource;
                futures::future::pending::<usize>().await
            })
        });
        drop(iterator);
        assert!(
            weak.upgrade().is_none(),
            "Dropping the last iterator must release its stream without another runtime poll"
        );
    }

    #[test]
    fn test_thread_safe_iterator_drop_releases_partial_stream() {
        let runtime = CurrentThreadRuntime::new();
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::unfold(resource, |resource| async move { Some((42, resource)) })
        });
        assert_eq!(iterator.next(), Some(42));
        drop(iterator);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_thread_safe_iterator_last_clone_releases_stream() {
        let runtime = CurrentThreadRuntime::new();
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::unfold(resource, |resource| async move { Some((42, resource)) })
        });
        assert_eq!(iterator.next(), Some(42));
        let clone = iterator.clone();
        drop(iterator);
        assert!(weak.upgrade().is_some());
        assert!(thread::spawn(move || drop(clone)).join().is_ok());
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_thread_safe_iterator_drop_releases_child_task() {
        let runtime = CurrentThreadRuntime::new();
        let resource = Arc::new(());
        let weak = Arc::downgrade(&resource);
        let child = runtime.handle().spawn(async move {
            let _resource = resource;
            futures::future::pending::<()>().await;
        });
        let mut iterator = runtime.block_on_stream_thread_safe(move |_| {
            stream::unfold(child, |child| async move { Some((42, child)) })
        });
        assert_eq!(iterator.next(), Some(42));
        drop(iterator);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_thread_safe_iterator_concurrent_last_drop_releases_stream() {
        let runtime = CurrentThreadRuntime::new();
        for _ in 0..32 {
            let resource = Arc::new(());
            let weak = Arc::downgrade(&resource);
            let iterator = runtime.block_on_stream_thread_safe(move |_| {
                stream::once(async move {
                    let _resource = resource;
                    futures::future::pending::<usize>().await
                })
            });
            let clone = iterator.clone();
            let barrier = Arc::new(Barrier::new(2));
            let other = Arc::clone(&barrier);
            let consumer = thread::spawn(move || {
                other.wait();
                drop(clone);
            });
            barrier.wait();
            drop(iterator);
            assert!(consumer.join().is_ok());
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn test_block_on_stream_multiple_threads() {
        let counter = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;
        let items_per_thread = 25;
        let total_items = 100;

        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..total_items).boxed());

        let barrier = Arc::new(Barrier::new(num_threads));
        let results = Arc::new(Mutex::new(Vec::new()));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let mut iter = iter.clone();
                let counter = Arc::clone(&counter);
                let barrier = Arc::clone(&barrier);
                let results = Arc::clone(&results);

                thread::spawn(move || {
                    barrier.wait();
                    let mut local_results = Vec::new();

                    for _ in 0..items_per_thread {
                        if let Some(item) = iter.next() {
                            counter.fetch_add(1, Ordering::SeqCst);
                            local_results.push(item);
                        }
                    }

                    results.lock().push(local_results);
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), total_items);

        let all_results = results.lock();
        let mut collected: Vec<_> = all_results.iter().flatten().copied().collect();
        collected.sort();
        assert_eq!(collected, (0..total_items).collect::<Vec<_>>());
    }

    #[test]
    fn test_block_on_stream_thread_safe_propagates_driver_panic() {
        let runtime = CurrentThreadRuntime::new();
        let mut iter = runtime.block_on_stream_thread_safe(|_h| {
            stream::poll_fn(|_| -> Poll<Option<usize>> {
                panic!("stream driver panic");
            })
            .boxed()
        });

        let panic = std::panic::catch_unwind(AssertUnwindSafe(|| iter.next()))
            .expect_err("stream panic must propagate through iterator");
        let message = panic
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<unknown panic>");
        assert!(message.contains("stream driver panic"));
    }

    fn panic_message(panic: &(dyn Any + Send)) -> &str {
        panic
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<unknown panic>")
    }

    // A driver panic must propagate on *every* run, regardless of executor scheduling. Running
    // the scenario many times guards against a return to timing-dependent propagation.
    #[test]
    fn test_block_on_stream_thread_safe_panic_propagation_is_deterministic() {
        for i in 0..2000 {
            let mut iter = CurrentThreadRuntime::new().block_on_stream_thread_safe(|_h| {
                stream::poll_fn(|_| -> Poll<Option<usize>> {
                    panic!("deterministic driver panic");
                })
                .boxed()
            });

            let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| iter.next()));
            assert!(
                outcome.is_err(),
                "driver panic was swallowed on iteration {i}: next() returned {:?}",
                outcome.ok().flatten(),
            );
        }
    }

    // A panic after some items were already produced must still surface, not be seen as a clean
    // end of stream.
    #[test]
    fn test_block_on_stream_thread_safe_panic_after_items() {
        let mut emitted = 0usize;
        let iter = CurrentThreadRuntime::new().block_on_stream_thread_safe(move |_h| {
            stream::poll_fn(move |_| -> Poll<Option<usize>> {
                if emitted < 3 {
                    emitted += 1;
                    Poll::Ready(Some(emitted))
                } else {
                    panic!("driver panic after items");
                }
            })
            .boxed()
        });

        // Drain the iterator. The terminal event must be a propagated panic, never a clean
        // `None`. This avoids depending on exactly how many buffered items survive channel close.
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(move || iter.collect::<Vec<_>>()));
        match outcome {
            Ok(items) => panic!("driver panic was swallowed; stream ended cleanly with {items:?}"),
            Err(panic) => assert!(panic_message(&*panic).contains("driver panic after items")),
        }
    }

    // With multiple consumers, a driver panic must reach *every* consumer that observes the end
    // of the stream; it must never be swallowed by all of them. Exactly one consumer joins the
    // driver and observes the panic; the rest see the stream end.
    #[test]
    fn test_block_on_stream_thread_safe_multi_consumer_panic_surfaced() {
        let iter = CurrentThreadRuntime::new().block_on_stream_thread_safe(|_h| {
            stream::poll_fn(|_| -> Poll<Option<usize>> {
                panic!("multi consumer driver panic");
            })
            .boxed()
        });

        let num_threads = 4;
        let barrier = Arc::new(Barrier::new(num_threads));
        let panics = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let mut iter = iter.clone();
                let barrier = Arc::clone(&barrier);
                let panics = Arc::clone(&panics);
                thread::spawn(move || {
                    barrier.wait();
                    match std::panic::catch_unwind(AssertUnwindSafe(|| iter.next())) {
                        // The driver panicked before producing anything, so a clean end is the
                        // only non-panic outcome a consumer may observe.
                        Ok(None) => {}
                        Ok(Some(_)) => panic!("no item was produced before the driver panicked"),
                        Err(panic) => {
                            assert!(panic_message(&*panic).contains("multi consumer driver panic"));
                            panics.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("consumer thread panicked uncaught");
        }

        // The panic surfaces to exactly one consumer and is never swallowed by all of them.
        assert_eq!(panics.load(Ordering::SeqCst), 1);
    }

    // Clean completion must return `None` with no spurious panic.
    #[test]
    fn test_block_on_stream_thread_safe_clean_completion_returns_none() {
        let mut iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(vec![1usize, 2, 3]).boxed());

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_block_on_stream_concurrent_clone_and_drive() {
        let num_items = 50;
        let num_threads = 3;

        let iter = CurrentThreadRuntime::new().block_on_stream_thread_safe(|h| {
            stream::unfold(0, move |state| {
                let h = h.clone();
                async move {
                    if state < num_items {
                        h.spawn_cpu(move || {
                            thread::sleep(Duration::from_micros(10));
                            state
                        })
                        .await;
                        Some((state, state + 1))
                    } else {
                        None
                    }
                }
            })
        });

        let collected = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(num_threads));

        let threads: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let iter = iter.clone();
                let collected = Arc::clone(&collected);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();
                    let mut local_items = Vec::new();

                    for item in iter {
                        local_items.push((thread_id, item));
                        if local_items.len() >= 5 {
                            break;
                        }
                    }

                    collected.lock().extend(local_items);
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        let results = collected.lock();
        let mut values: Vec<_> = results.iter().map(|(_, v)| *v).collect();
        values.sort();
        values.dedup();

        assert!(values.len() >= 5);
        assert!(values.iter().all(|&v| v < num_items));
    }

    #[test]
    fn test_block_on_stream_async_work() {
        let runtime = CurrentThreadRuntime::new();
        let handle = runtime.handle();
        let iter = runtime.block_on_stream({
            stream::unfold((handle, 0), |(h, state)| async move {
                if state < 10 {
                    let value = h
                        .spawn(async move { futures::future::ready(state * 2).await })
                        .await;
                    Some((value, (h, state + 1)))
                } else {
                    None
                }
            })
        });

        let results: Vec<_> = iter.collect();
        assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_block_on_stream_drop_receivers_early() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);

        let mut iter = CurrentThreadRuntime::new().block_on_stream({
            stream::unfold(0, move |state| {
                let c = Arc::clone(&c);
                async move {
                    (state < 100).then(|| {
                        c.fetch_add(1, Ordering::SeqCst);
                        (state, state + 1)
                    })
                }
            })
            .boxed()
        });

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));

        drop(iter);

        let final_count = counter.load(Ordering::SeqCst);
        assert!(
            final_count < 100,
            "Stream should stop when all receivers are dropped"
        );
    }

    #[test]
    fn test_block_on_stream_interleaved_access() {
        let barrier = Arc::new(Barrier::new(2));
        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..20).boxed());

        let iter1 = iter.clone();
        let iter2 = iter;
        let barrier1 = Arc::clone(&barrier);
        let barrier2 = barrier;

        let thread1 = thread::spawn(move || {
            let mut iter = iter1;
            let mut results = Vec::new();
            barrier1.wait();

            for _ in 0..5 {
                if let Some(val) = iter.next() {
                    results.push(val);
                    thread::sleep(Duration::from_micros(50));
                }
            }
            results
        });

        let thread2 = thread::spawn(move || {
            let mut iter = iter2;
            let mut results = Vec::new();
            barrier2.wait();

            for _ in 0..5 {
                if let Some(val) = iter.next() {
                    results.push(val);
                    thread::sleep(Duration::from_micros(50));
                }
            }
            results
        });

        let results1 = thread1.join().unwrap();
        let results2 = thread2.join().unwrap();

        let mut all_results = results1;
        all_results.extend(results2);
        all_results.sort();

        assert_eq!(all_results, (0..10).collect::<Vec<_>>());

        for i in 0..10 {
            assert_eq!(all_results.iter().filter(|&&x| x == i).count(), 1);
        }
    }

    #[test]
    fn test_block_on_stream_stress_test() {
        let num_threads = 10;
        let num_items = 1000;

        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..num_items).boxed());

        let received = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(num_threads));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let iter = iter.clone();
                let received = Arc::clone(&received);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();
                    for val in iter {
                        received.lock().push(val);
                    }
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        let mut results = received.lock().clone();
        results.sort();

        assert_eq!(results.len(), num_items);
        assert_eq!(results, (0..num_items).collect::<Vec<_>>());
    }
}
