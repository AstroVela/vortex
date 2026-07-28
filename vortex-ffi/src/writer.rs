// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::SinkExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::channel::mpsc::Sender;
use parking_lot::RwLock;
use vortex::array::ArrayRef;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::dtype::DType;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::file::WriteSummary;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::Task;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutStrategy;
use vortex::session::VortexSession;

use crate::RUNTIME;
use crate::array::vx_array;
use crate::dtype::vx_dtype;
use crate::error::try_or_default;
use crate::error::vx_error;
use crate::session::vx_session;
use crate::string::vx_view;

struct Inner {
    sender: Option<Sender<VortexResult<ArrayRef>>>,
    task: Option<Task<VortexResult<WriteSummary>>>,
}

/// vx_writer allows concurrently writing vx_array's into a .vortex file
pub struct vx_writer {
    session: VortexSession,
    inner: RwLock<Inner>,
    dtype: DType,
}

impl vx_writer {
    fn take_error(&self) -> VortexError {
        let task = self.inner.write().task.take();
        match task {
            Some(task) => match RUNTIME.block_on(task) {
                Ok(_) => vortex_err!("writer is closed"),
                Err(e) => e,
            },
            None => vortex_err!("writer is closed"),
        }
    }
}

/// Summary of a written .vortex file
#[repr(C)]
#[cfg_attr(test, derive(Debug, Clone, Copy))]
pub struct vx_write_summary {
    /// Number of rows
    pub row_count: u64,
    /// File size in bytes
    pub file_size: u64,
}

/// Open a writer for a file at "path" with explicit write strategy. "path"
/// is copied.
///
/// "dtype" is used to validate pushed arrays so they would all have the same
/// schema.
///
/// "concurrent_array_limit" bounds how many pushed arrays may be buffered in
/// flight before `vx_writer_push` blocks (channel capacity / backpressure). It
/// caps RAM used for buffering; the actual encoding parallelism is governed by
/// the write strategy.
///
/// # Safety
///
/// session and dtype must be non-null pointers to valid objects.
/// path's pointer must be NULL only on len = 0.
pub unsafe fn vx_writer_open_with_strategy(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    concurrent_array_limit: usize,
    strategy: Arc<dyn LayoutStrategy>,
) -> VortexResult<*mut vx_writer> {
    let session = vx_session::as_ref(session).clone();
    vortex_ensure!(!path.ptr.is_null());
    let path = unsafe { path.as_str() }?.to_string();

    let file_dtype = vx_dtype::as_ref(dtype);
    let (sender, receiver) = mpsc::channel(concurrent_array_limit);
    let dtype = file_dtype.clone();
    let array_stream = ArrayStreamAdapter::new(dtype.clone(), receiver.into_stream());

    let writer = Box::new(vx_writer {
        session,
        inner: RwLock::new(Inner {
            sender: Some(sender),
            task: None,
        }),
        dtype,
    });

    // Spawn the write task on the writer's own session so the task and the
    // driving block_on (in push/close) share one executor.
    let task_session = writer.session.clone();
    let task = writer.session.handle().spawn(async move {
        let mut file = async_fs::File::create(path).await?;
        task_session
            .write_options()
            .with_strategy(strategy)
            .write(&mut file, array_stream)
            .await
    });
    writer.inner.write().task = Some(task);

    Ok(Box::into_raw(writer))
}

/// Open a writer for a file at "path". "path" is copied.
///
/// "dtype" is used to validate pushed arrays so they would all have the same
/// schema.
///
/// "concurrent_array_limit" is an upper limit on how many pushed vx_array's
/// may be buffered before vx_writer_push blocks.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_open(
    session: *const vx_session,
    path: vx_view,
    dtype: *const vx_dtype,
    concurrent_array_limit: usize,
    error_out: *mut *mut vx_error,
) -> *mut vx_writer {
    let strategy = WriteStrategyBuilder::default().build();
    try_or_default(error_out, || unsafe {
        vx_writer_open_with_strategy(session, path, dtype, concurrent_array_limit, strategy)
    })
}

/// Push an array into a writer. Does not take ownership of array.
///
/// Array ordering across concurrent calls to this function is
/// non-deterministic: vx_writer_push(array1) called concurrently with
/// vx_writer_push(array2) may write array2 first.
///
/// Errors if array's dtype and writer's initialized dtype are different.
/// Errors if writer has already been closed.
///
/// Thread safe.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_push(
    writer: *mut vx_writer,
    array: *const vx_array,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        vortex_ensure!(!writer.is_null());

        let array = vx_array::as_ref(array);
        let writer = unsafe { &*writer };

        vortex_ensure!(
            *array.dtype() == writer.dtype,
            "array dtype {} does not match writer dtype {}",
            array.dtype(),
            writer.dtype
        );

        let send_result = {
            let inner = writer.inner.read();
            let mut sender = inner
                .sender
                .clone()
                .ok_or_else(|| vortex_err!("writer is closed"))?;
            RUNTIME.block_on(sender.send(Ok(array.clone())))
        };

        match send_result {
            Ok(_) => Ok(()),
            Err(_) => Err(writer.take_error()),
        }
    })
}

/// Close a writer.
///
/// Call to ensure all values pushed to the writer are indeed written. This
/// call writes the footer to the file. If you don't call this function, file
/// will be left corrupted.
///
/// You need to call vx_writer_free after this function even if it returned an
/// error.
///
/// "summary_out" may be NULL. If it's non-NULL, it is filled with file's row
/// count and size.
///
/// If this function is called concurrently with vx_writer_push, it will block
/// until vx_writer_push finishes.
///
/// Errors if writer was already closed.
///
/// Thread unsafe. Must be called exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_close(
    writer: *mut vx_writer,
    summary_out: *mut vx_write_summary,
    error_out: *mut *mut vx_error,
) {
    try_or_default(error_out, || {
        vortex_ensure!(!writer.is_null());
        let writer = unsafe { &*writer };

        let task = {
            let mut inner = writer.inner.write();
            drop(inner.sender.take());
            inner.task.take()
        };
        let task = task.ok_or_else(|| vortex_err!("writer is closed"))?;

        let summary = RUNTIME.block_on(task)?;
        if !summary_out.is_null() {
            unsafe {
                *summary_out = vx_write_summary {
                    row_count: summary.row_count(),
                    file_size: summary.size(),
                };
            }
        }
        VortexResult::Ok(())
    })
}

/// Free the writer.
///
/// Thread unsafe. Must be called exactly once.
///
/// If vx_writer_close wasn't called before this function, file is left
/// corrupted.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_writer_free(writer: *mut vx_writer) {
    if writer.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(writer) });
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::thread::spawn;

    use tempfile::NamedTempFile;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::DType;

    use super::*;
    use crate::array::vx_array;
    use crate::array::vx_array_free;
    use crate::dtype::vx_dtype;
    use crate::dtype::vx_dtype_free;
    use crate::error::vx_error_free;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer() {
        let temp_file = NamedTempFile::new().unwrap();
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);

        unsafe {
            let session = vx_session_new();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());
            let dtype = vx_dtype::new(dtype);

            let mut error = ptr::null_mut();
            let writer = vx_writer_open(session, path, dtype, 1, &raw mut error);
            assert!(error.is_null());
            assert!(!writer.is_null());

            let array = vx_array::new(array.into_array());
            vx_writer_push(writer, array, &raw mut error);
            assert!(error.is_null());

            let mut summary = vx_write_summary {
                row_count: 0,
                file_size: 0,
            };
            vx_writer_close(writer, &raw mut summary, &raw mut error);
            assert!(error.is_null());
            assert_eq!(summary.row_count, 3);
            assert!(summary.file_size > 0);

            vx_writer_free(writer);
            vx_array_free(array);
            vx_dtype_free(dtype);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_holds_session() {
        let temp_file = NamedTempFile::new().unwrap();
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        let array = PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::NonNullable);

        unsafe {
            let session = vx_session_new();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());
            let dtype = vx_dtype::new(dtype);

            let mut error = ptr::null_mut();
            let writer = vx_writer_open(session, path, dtype, 1, &raw mut error);
            assert!(error.is_null());

            vx_session_free(session);

            let array = vx_array::new(array.into_array());
            vx_writer_push(writer, array, &raw mut error);
            assert!(error.is_null());

            vx_writer_close(writer, ptr::null_mut(), &raw mut error);
            assert!(error.is_null());

            vx_writer_free(writer);
            vx_array_free(array);
            vx_dtype_free(dtype);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_concurrent() {
        let temp_file = NamedTempFile::new().unwrap();
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        unsafe {
            let session = vx_session_new();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());
            let dtype = vx_dtype::new(dtype);

            let mut error = ptr::null_mut();
            let writer = vx_writer_open(session, path, dtype, 4, &raw mut error);
            assert!(error.is_null());

            let addr = writer as usize;
            let pool: Vec<_> = (0..4)
                .map(|t| {
                    spawn(move || {
                        let writer = addr as *mut vx_writer;
                        let mut err = ptr::null_mut();
                        for i in 0..4 {
                            let v = t * 4 + i;
                            let array = PrimitiveArray::new(buffer![v], Validity::NonNullable);
                            let array = vx_array::new(array.into_array());
                            vx_writer_push(writer, array, &raw mut err);
                            assert!(err.is_null());
                            vx_array_free(array);
                        }
                    })
                })
                .collect();
            for handle in pool {
                handle.join().unwrap();
            }

            let mut summary = vx_write_summary {
                row_count: 0,
                file_size: 0,
            };
            vx_writer_close(writer, &raw mut summary, &raw mut error);
            assert!(error.is_null());
            assert_eq!(summary.row_count, u64::try_from(16).unwrap());

            vx_writer_free(writer);
            vx_dtype_free(dtype);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_null_path() {
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        unsafe {
            let session = vx_session_new();
            let vx_dtype_ptr = vx_dtype::new(dtype);

            let mut error = ptr::null_mut();
            let writer = vx_writer_open(session, vx_view::null(), vx_dtype_ptr, 1, &raw mut error);

            assert!(writer.is_null());
            assert!(!error.is_null());

            vx_error_free(error);
            vx_dtype_free(vx_dtype_ptr);
            vx_session_free(session);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_writer_push_after_close() {
        let temp_file = NamedTempFile::new().unwrap();
        let dtype = DType::Primitive(vortex::dtype::PType::I32, false.into());
        let array = PrimitiveArray::new(buffer![1i32], Validity::NonNullable);
        unsafe {
            let session = vx_session_new();
            let path = vx_view::from_str(temp_file.path().to_str().unwrap());
            let dtype = vx_dtype::new(dtype);

            let mut error = ptr::null_mut();
            let writer = vx_writer_open(session, path, dtype, 1, &raw mut error);
            assert!(error.is_null());

            let array = vx_array::new(array.into_array());
            vx_writer_push(writer, array, &raw mut error);
            assert!(error.is_null());

            vx_writer_close(writer, ptr::null_mut(), &raw mut error);
            assert!(error.is_null());

            vx_writer_push(writer, array, &raw mut error);
            assert!(!error.is_null());
            let message = crate::error::vx_error_message(error);
            assert!(message.as_str().unwrap().contains("closed"));
            vx_error_free(error);

            vx_writer_close(writer, ptr::null_mut(), &raw mut error);
            assert!(!error.is_null());
            vx_error_free(error);

            vx_writer_free(writer);

            vx_array_free(array);
            vx_dtype_free(dtype);
            vx_session_free(session);
        }
    }
}
