// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The only unsafe boundary; callers validate bundles before handing them to C++.

use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::SpFreshBuildOptions;
use crate::SpFreshBundle;

unsafe extern "C" {
    fn vortex_spfresh_build(
        root: *const c_char,
        vectors: *const f32,
        dimension: u32,
        rows: u32,
        heads: u32,
        posting_pages: u32,
        replicas: u32,
        error: *mut c_char,
    ) -> i32;
    fn vortex_spfresh_open(
        root: *const c_char,
        dimension: u32,
        rows: u32,
        posting_pages: u32,
        handle: *mut *mut c_void,
        error: *mut c_char,
    ) -> i32;
    fn vortex_spfresh_search(
        handle: *mut c_void,
        queries: *const f32,
        dimension: u32,
        count: u32,
        k: u32,
        max_check: u32,
        internal_results: u32,
        search_pages: u32,
        ids: *mut i32,
        distances: *mut f32,
        error: *mut c_char,
    ) -> i32;
    fn vortex_spfresh_close(handle: *mut c_void);
}

#[derive(Debug)]
pub(crate) struct Native {
    handle: NonNull<c_void>,
    bundle: SpFreshBundle,
}

// SAFETY: the C++ bridge serializes all entrypoints with a process-wide mutex and
// detaches handle-owned workspaces from upstream TLS at every call boundary.
// Synchronous IO leaves no outstanding tasks or owner-thread affinity.
unsafe impl Send for Native {}
// SAFETY: shared Rust access only invokes those serialized C++ entrypoints. Drop
// requires exclusive ownership, so no call can race handle destruction.
unsafe impl Sync for Native {}

fn result(code: i32, error: &[u8; 1024]) -> VortexResult<()> {
    if code != 0 {
        let end = error
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(error.len());
        vortex_bail!("SPFresh native: {}", String::from_utf8_lossy(&error[..end]));
    }
    Ok(())
}

impl Native {
    pub(crate) fn build(
        root: &Path,
        vectors: &[f32],
        bundle: SpFreshBundle,
        options: &SpFreshBuildOptions,
    ) -> VortexResult<()> {
        bundle.validate()?;
        options.validate(bundle.rows)?;
        if options.dimension != bundle.dimension
            || options.posting_page_limit != bundle.posting_page_limit
            || (bundle.rows as usize).checked_mul(bundle.dimension as usize) != Some(vectors.len())
        {
            vortex_bail!("Invalid native build vector shape");
        }
        let path = CString::new(root.as_os_str().as_bytes()).map_err(|err| vortex_err!("{err}"))?;
        let mut error = [0u8; 1024];
        // SAFETY: dimensions match the live input slice. The synchronous, serialized
        // L2 build borrows but never mutates or retains it. Scratch is private and
        // the error buffer is 1024 bytes; the bridge catches native exceptions.
        let code = unsafe {
            vortex_spfresh_build(
                path.as_ptr(),
                vectors.as_ptr(),
                bundle.dimension,
                bundle.rows,
                options.head_count,
                options.posting_page_limit,
                options.replicas,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)
    }

    pub(crate) fn open(root: &Path, bundle: SpFreshBundle) -> VortexResult<Self> {
        bundle.validate()?;
        let path = CString::new(root.as_os_str().as_bytes()).map_err(|err| vortex_err!("{err}"))?;
        let mut error = [0u8; 1024];
        let mut handle = std::ptr::null_mut();
        // SAFETY: path and output buffers live across the call. The caller holds
        // a verified lease of the validated native bundle; C++ catches exceptions.
        let code = unsafe {
            vortex_spfresh_open(
                path.as_ptr(),
                bundle.dimension,
                bundle.rows,
                bundle.posting_page_limit,
                &raw mut handle,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        Ok(Self {
            handle: NonNull::new(handle).ok_or_else(|| vortex_err!("Null SPFresh handle"))?,
            bundle,
        })
    }

    pub(crate) fn search(
        &self,
        queries: &[f32],
        k: u32,
        max_check: u32,
        internal_results: u32,
        search_pages: u32,
    ) -> VortexResult<Vec<Vec<(u32, f32)>>> {
        if queries.is_empty()
            || !queries.len().is_multiple_of(self.bundle.dimension as usize)
            || k == 0
            || k > self.bundle.rows
        {
            vortex_bail!("Invalid native query shape");
        }
        let count = u32::try_from(queries.len() / self.bundle.dimension as usize)
            .map_err(|err| vortex_err!("SPFresh batch size: {err}"))?;
        let len = (count as usize)
            .checked_mul(k as usize)
            .ok_or_else(|| vortex_err!("SPFresh result size overflow"))?;
        let mut ids = vec![-1i32; len];
        let mut distances = vec![0f32; len];
        let mut error = [0u8; 1024];
        // SAFETY: the handle is live, input length is count*dimension, both output
        // arrays have count*k elements, and the error buffer has exactly 1024 bytes.
        let code = unsafe {
            vortex_spfresh_search(
                self.handle.as_ptr(),
                queries.as_ptr(),
                self.bundle.dimension,
                count,
                k,
                max_check,
                internal_results,
                search_pages,
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        ids.chunks_exact(k as usize)
            .zip(distances.chunks_exact(k as usize))
            .map(|(ids, distances)| {
                ids.iter()
                    .zip(distances)
                    .filter(|(id, _)| **id != -1)
                    .map(|(id, distance)| {
                        let id = u32::try_from(*id)
                            .map_err(|err| vortex_err!("Invalid native ID: {err}"))?;
                        if id >= self.bundle.rows || !distance.is_finite() || *distance < 0.0 {
                            vortex_bail!("Invalid SPFresh native result");
                        }
                        Ok((id, *distance))
                    })
                    .collect()
            })
            .collect()
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        // SAFETY: this handle was returned by open, is owned solely by self, and
        // cannot have outstanding calls when its final Rust owner is dropped.
        unsafe { vortex_spfresh_close(self.handle.as_ptr()) };
    }
}
