// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::io;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use compio::runtime::JoinHandle;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future::Abortable;
use futures::future::BoxFuture;
use futures::future::LocalBoxFuture;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_error::vortex_panic;
use vortex_io::runtime::AbortHandle;
use vortex_io::runtime::AbortHandleRef;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Executor;
use vortex_io::runtime::Handle;

/// A Vortex runtime backed by a thread-local Compio runtime.
///
/// The runtime does no work unless [`BlockingRuntime::block_on`] or an iterator returned by
/// [`BlockingRuntime::block_on_stream`] is being driven. For thread-per-core execution, construct
/// one `CompioRuntime` on each worker thread.
pub struct CompioRuntime {
    sender: Arc<Sender>,
    runtime: compio::runtime::Runtime,
}

impl CompioRuntime {
    /// Create a Compio-backed Vortex runtime using Compio's default platform driver.
    pub fn new() -> io::Result<Self> {
        let runtime = compio::runtime::Runtime::new()?;
        let sender = Arc::new(Sender::new(&runtime));
        Ok(Self { sender, runtime })
    }

    /// Return the Compio driver selected for this runtime.
    pub fn driver_type(&self) -> compio::driver::DriverType {
        self.runtime.driver_type()
    }

    /// Return a handle for scheduling operations that must remain local to this Compio runtime.
    pub fn compio_handle(&self) -> CompioHandle {
        CompioHandle {
            sender: Arc::clone(&self.sender),
        }
    }
}

/// A thread-safe handle for scheduling local operations on a [`CompioRuntime`].
///
/// Unlike a Vortex [`Handle`], this handle accepts a factory rather than an already-created future.
/// This ensures Compio's `!Send` I/O futures are created and polled only on their owning runtime.
#[derive(Clone)]
pub struct CompioHandle {
    sender: Arc<Sender>,
}

impl CompioHandle {
    pub(crate) fn spawn_local<F, Fut, R>(&self, make_future: F) -> LocalTask<R>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        self.sender.spawn_local(make_future)
    }
}

impl BlockingRuntime for CompioRuntime {
    type BlockingIterator<'a, R: 'a> = CompioBlockingIterator<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = Arc::clone(&self.sender) as Arc<dyn Executor>;
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<Fut, R>(&self, future: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        self.runtime.block_on(future)
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        CompioBlockingIterator {
            runtime: self.runtime.clone(),
            stream: stream.boxed(),
        }
    }
}

/// Run a future to completion on a new [`CompioRuntime`].
///
/// The closure receives a Vortex [`Handle`] associated with the runtime.
pub fn block_on<F, Fut, R>(f: F) -> io::Result<R>
where
    F: FnOnce(Handle) -> Fut,
    Fut: Future<Output = R>,
{
    let runtime = CompioRuntime::new()?;
    let handle = runtime.handle();
    Ok(runtime.block_on(f(handle)))
}

/// An iterator that drives a stream with a Compio runtime on the current thread.
pub struct CompioBlockingIterator<'a, T> {
    runtime: compio::runtime::Runtime,
    stream: BoxStream<'a, T>,
}

impl<T> Iterator for CompioBlockingIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.runtime.block_on(self.stream.next())
    }
}

struct Sender {
    tasks: kanal::Sender<Spawn>,
    runtime_id: usize,
}

impl Sender {
    fn new(runtime: &compio::runtime::Runtime) -> Self {
        let (send, recv) = kanal::unbounded::<Spawn>();
        let runtime_id = runtime_identity(runtime);

        runtime
            .spawn(async move {
                while let Ok(spawn) = recv.as_async().recv().await {
                    spawn.schedule();
                }
            })
            .detach();

        Self {
            tasks: send,
            runtime_id,
        }
    }

    fn is_current_runtime(&self) -> bool {
        compio::runtime::Runtime::try_with_current(|runtime| {
            runtime_identity(runtime) == self.runtime_id
        })
        .unwrap_or(false)
    }

    fn schedule(&self, spawn: Spawn) {
        // Vortex's coalescing driver runs on this runtime, so this is the hot path for local file
        // reads. Bypass the synchronized cross-thread queue when scheduling from the owner.
        if self.is_current_runtime() {
            spawn.schedule();
        } else if let Err(error) = self.tasks.send(spawn) {
            vortex_panic!("Compio executor missing: {error}");
        }
    }

    fn send(&self, spawn: Spawn) -> AbortHandleRef {
        let (task_send, task_recv) = oneshot::channel();
        let spawn = spawn.with_callback(task_send);
        self.schedule(spawn);
        Box::new(LazyAbortHandle {
            task: Mutex::new(task_recv),
        })
    }

    fn spawn_local<F, Fut, R>(&self, make_future: F) -> LocalTask<R>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (result_send, result_recv) = oneshot::channel();
        let (abort_handle, abort_registration) = futures::future::AbortHandle::new_pair();
        let factory = Box::new(move || {
            async move {
                let future = async move { make_future().await };
                let output =
                    Abortable::new(AssertUnwindSafe(future).catch_unwind(), abort_registration)
                        .await;
                if let Ok(output) = output {
                    // The receiver may have been dropped immediately after the operation completed.
                    drop(result_send.send(output));
                }
            }
            .boxed_local()
        });

        self.schedule(Spawn::Local { factory });

        LocalTask {
            result: result_recv.into_future(),
            abort_handle: Some(abort_handle),
        }
    }
}

fn runtime_identity(runtime: &compio::runtime::Runtime) -> usize {
    std::ptr::from_ref(&**runtime).addr()
}

impl Executor for Sender {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.send(Spawn::Future {
            future,
            callback: None,
        })
    }

    fn spawn_io(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.send(Spawn::Future {
            future,
            callback: None,
        })
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.send(Spawn::Cpu {
            task,
            callback: None,
        })
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.send(Spawn::Blocking {
            task,
            callback: None,
        })
    }
}

enum Spawn {
    Future {
        future: BoxFuture<'static, ()>,
        callback: Option<oneshot::Sender<AbortHandleRef>>,
    },
    Cpu {
        task: Box<dyn FnOnce() + Send + 'static>,
        callback: Option<oneshot::Sender<AbortHandleRef>>,
    },
    Blocking {
        task: Box<dyn FnOnce() + Send + 'static>,
        callback: Option<oneshot::Sender<AbortHandleRef>>,
    },
    Local {
        factory: LocalFutureFactory,
    },
}

type LocalFutureFactory = Box<dyn FnOnce() -> LocalBoxFuture<'static, ()> + Send + 'static>;

impl Spawn {
    fn with_callback(self, callback: oneshot::Sender<AbortHandleRef>) -> Self {
        match self {
            Spawn::Future { future, .. } => Spawn::Future {
                future,
                callback: Some(callback),
            },
            Spawn::Cpu { task, .. } => Spawn::Cpu {
                task,
                callback: Some(callback),
            },
            Spawn::Blocking { task, .. } => Spawn::Blocking {
                task,
                callback: Some(callback),
            },
            Spawn::Local { .. } => {
                vortex_panic!("local Compio tasks manage cancellation directly")
            }
        }
    }

    fn schedule(self) {
        if let Spawn::Local { factory } = self {
            compio::runtime::spawn(factory()).detach();
            return;
        }

        let (task, callback) = match self {
            Spawn::Future { future, callback } => (compio::runtime::spawn(future), callback),
            Spawn::Cpu { task, callback } => {
                (compio::runtime::spawn(async move { task() }), callback)
            }
            Spawn::Blocking { task, callback } => (compio::runtime::spawn_blocking(task), callback),
            Spawn::Local { .. } => unreachable!(),
        };

        let abort_handle: AbortHandleRef = Box::new(CompioAbortHandle { task: Some(task) });
        if let Some(callback) = callback {
            // A failed send means the caller dropped or aborted its task before it was scheduled.
            drop(callback.send(abort_handle));
        }
    }
}

type LocalTaskOutput<T> = Result<T, Box<dyn Any + Send>>;

pub(crate) struct LocalTask<T> {
    result: oneshot::AsyncReceiver<LocalTaskOutput<T>>,
    abort_handle: Option<futures::future::AbortHandle>,
}

impl<T> Future for LocalTask<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match ready!(this.result.poll_unpin(cx)) {
            Ok(Ok(output)) => {
                this.abort_handle.take();
                Poll::Ready(output)
            }
            Ok(Err(panic)) => {
                this.abort_handle.take();
                std::panic::resume_unwind(panic)
            }
            Err(error) => vortex_panic!("Compio local task was cancelled: {error}"),
        }
    }
}

impl<T> Drop for LocalTask<T> {
    fn drop(&mut self) {
        if let Some(abort_handle) = self.abort_handle.take() {
            abort_handle.abort();
        }
    }
}

struct CompioAbortHandle {
    task: Option<JoinHandle<()>>,
}

impl AbortHandle for CompioAbortHandle {
    fn abort(mut self: Box<Self>) {
        // Dropping a Compio join handle cancels its task.
        drop(self.task.take());
    }
}

impl Drop for CompioAbortHandle {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.detach();
        }
    }
}

struct LazyAbortHandle {
    task: Mutex<oneshot::Receiver<AbortHandleRef>>,
}

impl AbortHandle for LazyAbortHandle {
    fn abort(self: Box<Self>) {
        if let Ok(task) = self.task.lock().try_recv() {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::stream;
    use vortex_error::VortexResult;
    use vortex_io::runtime::BlockingRuntime;

    use super::CompioRuntime;

    #[test]
    fn drives_spawned_work() -> VortexResult<()> {
        let runtime = CompioRuntime::new()?;
        let handle = runtime.handle();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = Arc::clone(&counter);

        let result = runtime.block_on(async move {
            let task = handle.spawn_io(async move {
                counter_clone.fetch_add(1, Ordering::SeqCst);
                42
            });
            task.await
        });

        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn drives_streams() -> VortexResult<()> {
        let runtime = CompioRuntime::new()?;
        let values = runtime
            .block_on_stream(stream::iter([1, 2, 3]))
            .collect::<Vec<_>>();
        assert_eq!(values, [1, 2, 3]);
        Ok(())
    }

    #[test]
    fn identifies_the_owning_runtime() -> VortexResult<()> {
        let runtime = CompioRuntime::new()?;
        assert!(!runtime.sender.is_current_runtime());
        runtime.block_on(async {
            assert!(runtime.sender.is_current_runtime());
        });
        Ok(())
    }
}
