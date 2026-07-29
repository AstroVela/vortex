// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(non_camel_case_types)]
#![deny(missing_docs)]

//! Pull-based (inverted-IO) scanning: the caller performs all reads.
//!
//! Unlike [`vx_data_source_scan`](crate::scan), nothing here touches the shared FFI runtime or
//! performs IO. The coroutine tells the caller which byte ranges it needs and into which
//! buffers to read them; decoding happens inside `advance` on the calling thread.
//!
//! Protocol, for both [`vx_pull_footer`] (file open) and [`vx_pull_scan`] (data):
//!
//! 1. `advance` returns [`VX_PULL_READS`](vx_pull_state::VX_PULL_READS) with reads to perform.
//!    Each read carries a pre-allocated, correctly aligned destination buffer `dst`. Fill
//!    `dst` with the `len` bytes at file offset `offset`, then call `complete(dst)`.
//!    Completions may arrive in any order, so many reads can be in flight at once.
//! 2. `advance` returns [`VX_PULL_BATCH`](vx_pull_state::VX_PULL_BATCH) when a decoded array
//!    (or a parsed footer) is ready.
//! 3. `advance` returns [`VX_PULL_DONE`](vx_pull_state::VX_PULL_DONE) when exhausted.
//!
//! These objects are not thread-safe: drive each from one thread. To scan one file on many
//! threads, create one `vx_pull_scan` per disjoint row range aligned to
//! [`vx_footer_split_points`], so no two scans read the same segment.

use std::ffi::c_int;
use std::ptr;

use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::file::Footer;
use vortex::file::pull::FooterEvent;
use vortex::file::pull::PullEvent;
use vortex::file::pull::PullFile;
use vortex::file::pull::PullFooter;
use vortex::file::pull::PullRead;
use vortex::file::pull::PullScan;
use vortex::file::pull::chunk_split_points;
use vortex::file::pull::footer_can_prune;
use vortex::utils::aliases::hash_map::HashMap;

use crate::array::vx_array;
use crate::box_wrapper;
use crate::dtype::vx_dtype;
use crate::error::try_or;
use crate::error::vx_error;
use crate::expression::vx_expression;
use crate::scan::scan_request;
use crate::scan::vx_scan_options;
use crate::session::vx_session;
use crate::session::vx_session_ref;

box_wrapper!(
    /// A parsed Vortex file footer: the segment map, layout tree, dtype, and statistics.
    ///
    /// Obtained from a [`vx_pull_footer`] coroutine. Free with [`vx_footer_free`].
    Footer,
    vx_footer
);

/// A byte-range read the caller must perform.
///
/// `dst` points to a buffer of `len` bytes owned by the coroutine that issued this read; it is
/// valid until the read is completed or the coroutine is freed. Fill it with the bytes at file
/// offset `offset` and pass `dst` back to the matching `complete` function. `dst` is the
/// identity of the read: return exactly the pointer you were given.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct vx_pull_read {
    /// Destination buffer to fill; also the identity of this read.
    pub dst: *mut u8,
    /// Absolute file offset to read from.
    pub offset: u64,
    /// Number of bytes to read.
    pub len: u64,
}

/// The state returned by a pull coroutine step.
#[repr(C)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum vx_pull_state {
    /// Reads were issued (possibly zero when the in-flight window is full): perform them and
    /// call `complete` for each, then `advance` again.
    VX_PULL_READS = 0,
    /// A result is ready (a batch for scans, a footer for footer coroutines).
    VX_PULL_BATCH = 1,
    /// The coroutine is exhausted.
    VX_PULL_DONE = 2,
    /// An error occurred; inspect the `vx_error` out-parameter.
    VX_PULL_ERROR = 3,
}

pub struct FfiPullFooter {
    inner: PullFooter,
    outstanding: Option<PullRead>,
}
box_wrapper!(
    /// A pull coroutine that parses a file footer without Vortex performing any IO.
    ///
    /// Footer reads are sequential, so at most one read is outstanding at a time.
    FfiPullFooter,
    vx_pull_footer
);

box_wrapper!(
    /// A per-file pull scanning context: builds the file's reader tree once so many
    /// [`vx_pull_scan`]s (e.g. one per chunk-aligned shard) can reuse it.
    ///
    /// Not thread-safe: scans created from one context must be driven by one thread.
    PullFile,
    vx_pull_file
);

/// Create a pull scanning context for the file described by "footer".
///
/// On error, returns NULL and sets "err". Free with [`vx_pull_file_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_file_new(
    session: *const vx_session,
    footer: *const vx_footer,
    err: *mut *mut vx_error,
) -> *mut vx_pull_file {
    try_or(err, ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session) }?;
        Ok(vx_pull_file::new(PullFile::try_new(
            vx_footer::as_ref(footer).clone(),
            session,
        )?))
    })
}

/// Create a pull scan over the file of "pull_file", reusing its reader tree.
///
/// See [`vx_pull_scan_new`] for "options" and "max_inflight". On error, returns NULL and
/// sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_file_scan(
    pull_file: *const vx_pull_file,
    options: *const vx_scan_options,
    max_inflight: u64,
    err: *mut *mut vx_error,
) -> *mut vx_pull_scan {
    try_or(err, ptr::null_mut(), || {
        let request = scan_request(options)?;
        let scan = vx_pull_file::as_ref(pull_file).scan(
            usize::try_from(max_inflight).unwrap_or(usize::MAX),
            move |mut b| {
                b = b
                    .with_projection(request.projection)
                    .with_some_filter(request.filter)
                    .with_selection(request.selection)
                    .with_ordered(request.ordered)
                    .with_some_limit(request.limit);
                if let Some(range) = request.row_range {
                    b = b.with_row_range(range);
                }
                b
            },
        )?;
        Ok(vx_pull_scan::new(FfiPullScan {
            inner: scan,
            outstanding: HashMap::default(),
            scratch: Vec::new(),
        }))
    })
}

pub struct FfiPullScan {
    inner: PullScan,
    outstanding: HashMap<usize, PullRead>,
    scratch: Vec<vx_pull_read>,
}
box_wrapper!(
    /// A pull coroutine that scans a single Vortex file; the caller performs all reads.
    FfiPullScan,
    vx_pull_scan
);

fn stash_read(read: &mut PullRead) -> vx_pull_read {
    vx_pull_read {
        dst: read.data().as_mut_ptr(),
        offset: read.offset(),
        len: read.len() as u64,
    }
}

/// Create a footer coroutine for a file of `file_size` bytes.
///
/// On error, returns NULL and sets "err". Free with [`vx_pull_footer_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_footer_new(
    session: *const vx_session,
    file_size: u64,
    err: *mut *mut vx_error,
) -> *mut vx_pull_footer {
    try_or(err, ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session) }?;
        vortex_ensure!(file_size > 0, "empty file");
        Ok(vx_pull_footer::new(FfiPullFooter {
            inner: PullFooter::new(session.clone(), file_size),
            outstanding: None,
        }))
    })
}

/// Advance the footer coroutine.
///
/// On VX_PULL_READS writes the single read to perform into "*out_read".
/// On VX_PULL_BATCH writes an owned footer handle into "*out_footer"; the coroutine is then
/// exhausted. On error returns VX_PULL_ERROR and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_footer_advance(
    footer: *mut vx_pull_footer,
    out_read: *mut vx_pull_read,
    out_footer: *mut *mut vx_footer,
    err: *mut *mut vx_error,
) -> vx_pull_state {
    try_or(err, vx_pull_state::VX_PULL_ERROR, || {
        let this = vx_pull_footer::as_mut(footer);
        match this.inner.advance()? {
            FooterEvent::Read(mut read) => {
                unsafe { out_read.write(stash_read(&mut read)) };
                this.outstanding = Some(read);
                Ok(vx_pull_state::VX_PULL_READS)
            }
            FooterEvent::Done(parsed) => {
                unsafe { out_footer.write(vx_footer::new(parsed)) };
                Ok(vx_pull_state::VX_PULL_BATCH)
            }
        }
    })
}

/// Hand the filled read buffer "dst" back to the footer coroutine.
///
/// Returns 0 on success. On error returns 1 and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_footer_complete(
    footer: *mut vx_pull_footer,
    dst: *const u8,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let this = vx_pull_footer::as_mut(footer);
        let mut read = this
            .outstanding
            .take()
            .ok_or_else(|| vortex_err!("no outstanding footer read"))?;
        vortex_ensure!(
            ptr::eq(read.data().as_ptr(), dst),
            "unknown read: dst does not match the outstanding read"
        );
        this.inner.complete(read)?;
        Ok(0)
    })
}

/// The dtype of the file described by "footer".
///
/// The caller owns the returned dtype and must free it with vx_dtype_free.
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_footer_dtype(footer: *const vx_footer) -> *mut vx_dtype {
    vx_dtype::new(vx_footer::as_ref(footer).dtype().clone())
}

/// The number of rows in the file described by "footer".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_footer_row_count(footer: *const vx_footer) -> u64 {
    vx_footer::as_ref(footer).row_count()
}

/// Returns true if "footer"'s file-level statistics prove that "filter" cannot match any rows
/// in the file, so the whole file can be skipped. Performs no IO. On error returns false and
/// sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_footer_can_prune(
    session: *const vx_session,
    footer: *const vx_footer,
    filter: *const vx_expression,
    err: *mut *mut vx_error,
) -> bool {
    try_or(err, false, || {
        let session = unsafe { vx_session_ref(session) }?;
        vortex_ensure!(!filter.is_null(), "null filter");
        footer_can_prune(
            vx_footer::as_ref(footer),
            session,
            vx_expression::as_ref(filter),
        )
    })
}

/// Chunk-aligned row split points for sharding a file scan across threads.
///
/// Scans over disjoint row ranges aligned to these points never share data segments for the
/// fields referenced by "projection"/"filter" (both may be NULL, meaning all fields), so they
/// can be driven as independent [`vx_pull_scan`]s without reading any segment twice. Performs
/// no IO.
///
/// Writes at most "capacity" points to "out_points" and the total number of points to
/// "*out_len"; if "*out_len" exceeds "capacity", call again with a larger buffer. Returns 0 on
/// success. On error returns 1 and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_footer_split_points(
    session: *const vx_session,
    footer: *const vx_footer,
    projection: *const vx_expression,
    filter: *const vx_expression,
    out_points: *mut u64,
    capacity: usize,
    out_len: *mut usize,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let session = unsafe { vx_session_ref(session) }?;
        let footer = vx_footer::as_ref(footer);
        let projection = if projection.is_null() {
            vortex::expr::root()
        } else {
            vx_expression::as_ref(projection).clone()
        };
        let filter = (!filter.is_null()).then(|| vx_expression::as_ref(filter).clone());
        let points = chunk_split_points(footer, session, &projection, filter.as_ref())?;
        unsafe { out_len.write(points.len()) };
        let n = points.len().min(capacity);
        vortex_ensure!(n == 0 || !out_points.is_null(), "null out_points");
        unsafe { ptr::copy_nonoverlapping(points.as_ptr(), out_points, n) };
        Ok(0)
    })
}

/// Create a pull scan of the file described by "footer".
///
/// "options" may be NULL (scan everything); its row_range selects the shard of the file this
/// scan decodes. "max_inflight" bounds how many reads may be outstanding, which also bounds
/// destination-buffer memory; pass 0 for no bound.
///
/// On error, returns NULL and sets "err". Free with [`vx_pull_scan_free`].
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_scan_new(
    session: *const vx_session,
    footer: *const vx_footer,
    options: *const vx_scan_options,
    max_inflight: u64,
    err: *mut *mut vx_error,
) -> *mut vx_pull_scan {
    try_or(err, ptr::null_mut(), || {
        let session = unsafe { vx_session_ref(session) }?;
        let footer = vx_footer::as_ref(footer).clone();
        let request = scan_request(options)?;
        let scan = PullScan::try_new(
            footer,
            session,
            usize::try_from(max_inflight).unwrap_or(usize::MAX),
            move |mut b| {
                b = b
                    .with_projection(request.projection)
                    .with_some_filter(request.filter)
                    .with_selection(request.selection)
                    .with_ordered(request.ordered)
                    .with_some_limit(request.limit);
                if let Some(range) = request.row_range {
                    b = b.with_row_range(range);
                }
                b
            },
        )?;
        Ok(vx_pull_scan::new(FfiPullScan {
            inner: scan,
            outstanding: HashMap::default(),
            scratch: Vec::new(),
        }))
    })
}

/// Advance the scan coroutine.
///
/// On VX_PULL_READS writes a pointer to an array of reads into "*out_reads" and its length
/// into "*out_reads_len"; the array is valid until the next advance or free. A zero length
/// means the in-flight window is full: complete an outstanding read first.
/// On VX_PULL_BATCH writes an owned array handle into "*out_batch" (free with vx_array_free).
/// On error returns VX_PULL_ERROR and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_scan_advance(
    scan: *mut vx_pull_scan,
    out_reads: *mut *const vx_pull_read,
    out_reads_len: *mut usize,
    out_batch: *mut *mut vx_array,
    err: *mut *mut vx_error,
) -> vx_pull_state {
    try_or(err, vx_pull_state::VX_PULL_ERROR, || {
        let this = vx_pull_scan::as_mut(scan);
        match this.inner.advance()? {
            PullEvent::Reads(reads) => {
                this.scratch.clear();
                for mut read in reads {
                    let ffi = stash_read(&mut read);
                    this.scratch.push(ffi);
                    this.outstanding.insert(ffi.dst as usize, read);
                }
                unsafe {
                    out_reads.write(this.scratch.as_ptr());
                    out_reads_len.write(this.scratch.len());
                }
                Ok(vx_pull_state::VX_PULL_READS)
            }
            PullEvent::Batch(batch) => {
                unsafe { out_batch.write(vx_array::new(batch)) };
                Ok(vx_pull_state::VX_PULL_BATCH)
            }
            PullEvent::Done => Ok(vx_pull_state::VX_PULL_DONE),
        }
    })
}

/// Hand a filled read buffer back to the scan. Completions may arrive in any order.
///
/// Returns 0 on success. On error returns 1 and sets "err".
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn vx_pull_scan_complete(
    scan: *mut vx_pull_scan,
    dst: *const u8,
    err: *mut *mut vx_error,
) -> c_int {
    try_or(err, 1, || {
        let this = vx_pull_scan::as_mut(scan);
        let read = this
            .outstanding
            .remove(&(dst as usize))
            .ok_or_else(|| vortex_err!("unknown read: dst was not issued by this scan"))?;
        this.inner.complete(read)?;
        Ok(0)
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use vortex::error::VortexExpect;

    use super::*;
    use crate::array::vx_array_free;
    use crate::array::vx_array_len;
    use crate::session::vx_session_free;
    use crate::session::vx_session_new;
    use crate::tests::assert_no_error;
    use crate::tests::write_sample;

    unsafe fn serve(bytes: &[u8], read: &vx_pull_read) {
        let start = usize::try_from(read.offset).unwrap();
        unsafe {
            ptr::copy_nonoverlapping(
                bytes[start..].as_ptr(),
                read.dst,
                usize::try_from(read.len).unwrap(),
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn pull_scan_round_trip() {
        unsafe {
            let session = vx_session_new();
            let (file, struct_array) = write_sample(session);
            let bytes = std::fs::read(file.path()).vortex_expect("read sample");

            let mut error = ptr::null_mut();
            let pf = vx_pull_footer_new(session, bytes.len() as u64, &raw mut error);
            assert_no_error(error);

            let mut footer = ptr::null_mut();
            let footer = loop {
                let mut read = vx_pull_read {
                    dst: ptr::null_mut(),
                    offset: 0,
                    len: 0,
                };
                match vx_pull_footer_advance(pf, &raw mut read, &raw mut footer, &raw mut error) {
                    vx_pull_state::VX_PULL_READS => {
                        serve(&bytes, &read);
                        assert_eq!(vx_pull_footer_complete(pf, read.dst, &raw mut error), 0);
                        assert_no_error(error);
                    }
                    vx_pull_state::VX_PULL_BATCH => break footer,
                    other => {
                        assert_no_error(error);
                        panic!("unexpected footer state {}", other as i32);
                    }
                }
            };
            vx_pull_footer_free(pf);

            assert_eq!(vx_footer_row_count(footer), struct_array.len() as u64);

            let mut points = vec![0u64; 16];
            let mut len = 0usize;
            let rc = vx_footer_split_points(
                session,
                footer,
                ptr::null(),
                ptr::null(),
                points.as_mut_ptr(),
                points.len(),
                &raw mut len,
                &raw mut error,
            );
            assert_no_error(error);
            assert_eq!(rc, 0);
            assert!(len >= 2);

            let scan = vx_pull_scan_new(session, footer, ptr::null(), 0, &raw mut error);
            assert_no_error(error);

            let mut rows = 0usize;
            loop {
                let mut reads: *const vx_pull_read = ptr::null();
                let mut reads_len = 0usize;
                let mut batch = ptr::null_mut();
                match vx_pull_scan_advance(
                    scan,
                    &raw mut reads,
                    &raw mut reads_len,
                    &raw mut batch,
                    &raw mut error,
                ) {
                    vx_pull_state::VX_PULL_READS => {
                        assert!(reads_len > 0, "sync driver must always receive reads");
                        for i in 0..reads_len {
                            let read = *reads.add(i);
                            serve(&bytes, &read);
                            assert_eq!(vx_pull_scan_complete(scan, read.dst, &raw mut error), 0);
                            assert_no_error(error);
                        }
                    }
                    vx_pull_state::VX_PULL_BATCH => {
                        rows += vx_array_len(batch);
                        vx_array_free(batch);
                    }
                    vx_pull_state::VX_PULL_DONE => break,
                    vx_pull_state::VX_PULL_ERROR => {
                        assert_no_error(error);
                        unreachable!()
                    }
                }
            }
            assert_eq!(rows, struct_array.len());

            vx_pull_scan_free(scan);
            vx_footer_free(footer);
            vx_session_free(session);
        }
    }
}
