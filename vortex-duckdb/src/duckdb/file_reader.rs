// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::future::ready;
use std::ptr;
use std::sync::Arc;

use futures::future::BoxFuture;
use parking_lot::Mutex;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::BlockingRuntime;

use crate::RUNTIME;
use crate::cpp;

struct ReadHandle(cpp::duckdb_vx_file_reader);

// SAFETY: The C++ handle owns its file and keeps the database alive. It checks a weak client
// context before each read. All access is serialized by DuckDBFileReader's mutex, and a running
// blocking task owns an Arc until it has finished, even if its future is cancelled.
unsafe impl Send for ReadHandle {}

impl Drop for ReadHandle {
    fn drop(&mut self) {
        // SAFETY: This value uniquely owns the opaque handle and no reads remain.
        unsafe { cpp::duckdb_vx_file_reader_destroy(self.0) };
    }
}

/// An exact-file reader using DuckDB's connection-scoped filesystem.
pub(crate) struct DuckDBFileReader {
    handle: Arc<Mutex<ReadHandle>>,
    size: u64,
}

impl DuckDBFileReader {
    /// Open a literal path without URL normalization or wildcard expansion.
    ///
    /// # Safety
    /// `context` must point to a live DuckDB ClientContext for this call.
    pub(crate) unsafe fn open(context: *mut c_void, path: &str) -> VortexResult<Self> {
        let path = CString::new(path)
            .map_err(|_| vortex_err!("Vortex input path contains a null byte"))?;
        let mut error = ptr::null_mut();
        let mut size = 0;
        // SAFETY: The caller guarantees the context lifetime; path and outputs live through the call.
        let handle = unsafe {
            cpp::duckdb_vx_file_reader_create(context, path.as_ptr(), &raw mut size, &raw mut error)
        };
        check_error(error)?;
        if handle.is_null() {
            vortex_bail!("DuckDB returned a null Vortex reader without an error");
        }
        Ok(Self {
            handle: Arc::new(Mutex::new(ReadHandle(handle))),
            size,
        })
    }
}

impl VortexReadAt for DuckDBFileReader {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::new(64 * 1024, 4 * 1024 * 1024))
    }

    fn concurrency(&self) -> usize {
        // DuckDB file handles are serialized; independent files can still read in parallel.
        1
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        Box::pin(ready(Ok(self.size)))
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let handle = Arc::clone(&self.handle);
        let size = self.size;
        Box::pin(async move {
            if offset > size || length as u64 > size - offset {
                vortex_bail!("Vortex input range is outside the file");
            }
            RUNTIME
                .handle()
                .spawn_blocking(move || {
                    let handle = handle.lock();
                    let mut buffer = ByteBufferMut::zeroed_aligned(length, alignment);
                    let mut error = ptr::null_mut();
                    // SAFETY: The mutex exclusively owns the live handle; buffer has length initialized bytes.
                    let state = unsafe {
                        cpp::duckdb_vx_file_reader_read(
                            handle.0,
                            offset,
                            buffer.as_mut_ptr(),
                            length,
                            &raw mut error,
                        )
                    };
                    check_error(error)?;
                    if state != cpp::duckdb_state::DuckDBSuccess {
                        vortex_bail!("DuckDB failed to read Vortex input without an error message");
                    }
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        })
    }
}

fn check_error(error: cpp::duckdb_vx_error) -> VortexResult<()> {
    if error.is_null() {
        return Ok(());
    }
    // SAFETY: A non-null bridge error owns a terminated message until it is freed below.
    let message = unsafe { CStr::from_ptr(cpp::duckdb_vx_error_value(error)) }
        .to_string_lossy()
        .into_owned();
    unsafe { cpp::duckdb_vx_error_free(error) };
    Err(vortex_err!("{message}"))
}
