// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_void;
use std::future::Future;
use std::future::ready;
use std::io;
use std::ptr;

use vortex::io::IoBuf;
use vortex::io::VortexWrite;

use crate::cpp;

/// A Vortex writer backed by DuckDB's client-context-aware file system.
pub(crate) struct DuckDBFileWriter {
    writer: cpp::duckdb_vx_file_writer,
    closed: bool,
}

// SAFETY: The C++ bridge uses the client context only while opening the file. The resulting opaque
// file handle is exclusively owned by this value, and all access requires `&mut self`, so it may
// move with the write future but is never accessed concurrently.
unsafe impl Send for DuckDBFileWriter {}

impl DuckDBFileWriter {
    /// Open `file_path` using the settings and secrets from `client_context`.
    ///
    /// # Safety
    ///
    /// `client_context` must point to the live `duckdb::ClientContext` that owns the COPY query.
    pub(crate) unsafe fn open(client_context: *mut c_void, file_path: &str) -> io::Result<Self> {
        if client_context.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot open Vortex output without a DuckDB client context",
            ));
        }
        let file_path = CString::new(file_path).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Vortex output path contains a null byte",
            )
        })?;
        let mut error = ptr::null_mut();
        let writer = unsafe {
            cpp::duckdb_vx_file_writer_create(client_context, file_path.as_ptr(), &raw mut error)
        };
        cpp_result(error, "failed to open Vortex output")?;
        if writer.is_null() {
            return Err(io::Error::other(
                "failed to open Vortex output: DuckDB returned a null writer",
            ));
        }
        Ok(Self {
            writer,
            closed: false,
        })
    }

    fn ensure_open(&self) -> io::Result<()> {
        if self.closed {
            return Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "Vortex output writer is already closed",
            ));
        }
        Ok(())
    }
}

impl VortexWrite for DuckDBFileWriter {
    fn write_all<B: IoBuf>(&mut self, buffer: B) -> impl Future<Output = io::Result<B>> {
        let result = self.ensure_open().and_then(|()| {
            let mut error = ptr::null_mut();
            let state = unsafe {
                cpp::duckdb_vx_file_writer_write(
                    self.writer,
                    buffer.as_slice().as_ptr(),
                    buffer.as_slice().len(),
                    &raw mut error,
                )
            };
            cpp_state_result(state, error, "failed to write Vortex output")
        });
        ready(result.map(|()| buffer))
    }

    fn flush(&mut self) -> impl Future<Output = io::Result<()>> {
        let result = self.ensure_open().and_then(|()| {
            let mut error = ptr::null_mut();
            let state = unsafe { cpp::duckdb_vx_file_writer_flush(self.writer, &raw mut error) };
            cpp_state_result(state, error, "failed to flush Vortex output")
        });
        ready(result)
    }

    fn shutdown(&mut self) -> impl Future<Output = io::Result<()>> {
        let result = self.ensure_open().and_then(|()| {
            self.closed = true;
            let mut error = ptr::null_mut();
            let state = unsafe { cpp::duckdb_vx_file_writer_close(self.writer, &raw mut error) };
            cpp_state_result(state, error, "failed to close Vortex output")
        });
        ready(result)
    }
}

impl Drop for DuckDBFileWriter {
    fn drop(&mut self) {
        if self.closed {
            unsafe { cpp::duckdb_vx_file_writer_destroy(self.writer) };
        } else {
            unsafe { cpp::duckdb_vx_file_writer_abort(self.writer) };
        }
    }
}

fn cpp_result(error: cpp::duckdb_vx_error, operation: &str) -> io::Result<()> {
    if error.is_null() {
        return Ok(());
    }

    let message = unsafe { CStr::from_ptr(cpp::duckdb_vx_error_value(error)) }
        .to_string_lossy()
        .into_owned();
    unsafe { cpp::duckdb_vx_error_free(error) };
    Err(io::Error::other(format!("{operation}: {message}")))
}

fn cpp_state_result(
    state: cpp::duckdb_state,
    error: cpp::duckdb_vx_error,
    operation: &str,
) -> io::Result<()> {
    cpp_result(error, operation)?;
    if state == cpp::duckdb_state::DuckDBSuccess {
        Ok(())
    } else {
        Err(io::Error::other(format!(
            "{operation}: DuckDB returned an error without a message"
        )))
    }
}
