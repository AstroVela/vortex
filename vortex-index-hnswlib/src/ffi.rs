// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Owned, serialized native handles and the only unsafe boundary.

use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::ptr::null_mut;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::HnswBuildOptions;
use crate::HnswBundle;
use crate::validate;

unsafe extern "C" {
    fn vortex_hnswlib_build(
        path: *const c_char,
        vectors: *const f32,
        dimension: u32,
        rows: u32,
        m: u32,
        construction: u32,
        seed: u32,
        threads: u32,
        error: *mut c_char,
    ) -> i32;
    fn vortex_hnswlib_open(
        path: *const c_char,
        dimension: u32,
        rows: u32,
        m: u32,
        output: *mut *mut c_void,
        error: *mut c_char,
    ) -> i32;
    fn vortex_hnswlib_search(
        handle: *mut c_void,
        query: *const f32,
        k: u32,
        ef: u32,
        ids: *mut u32,
        distances: *mut f32,
        count: *mut u32,
        error: *mut c_char,
    ) -> i32;
    fn vortex_hnswlib_close(handle: *mut c_void);
}

fn supported_cpu() -> VortexResult<()> {
    if !std::is_x86_feature_detected!("avx2")
        || !std::is_x86_feature_detected!("fma")
        || !std::is_x86_feature_detected!("f16c")
    {
        vortex_bail!("hnswlib native requires AVX2, FMA and F16C");
    }
    Ok(())
}

fn result(code: i32, error: &[u8; 1024]) -> VortexResult<()> {
    if code != 0 {
        let end = error
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(error.len());
        vortex_bail!("hnswlib native: {}", String::from_utf8_lossy(&error[..end]));
    }
    Ok(())
}

#[derive(Debug)]
pub(crate) struct Native {
    handle: NonNull<c_void>,
    bundle: HnswBundle,
}

// SAFETY: each handle's C++ mutex serializes ef changes and synchronous searches.
// Distance kernels have no shared mutable state. No native operation retains
// borrowed inputs or has owner-thread affinity.
unsafe impl Send for Native {}
// SAFETY: the same handle mutex guards shared access; final drop cannot race borrows.
unsafe impl Sync for Native {}

impl Native {
    pub(crate) fn build(
        path: &Path,
        vectors: &[f32],
        bundle: HnswBundle,
        options: &HnswBuildOptions,
    ) -> VortexResult<()> {
        supported_cpu()?;
        bundle.validate()?;
        options.validate()?;
        if options.bundle(bundle.rows) != bundle
            || (bundle.rows as usize).checked_mul(bundle.dimension as usize) != Some(vectors.len())
        {
            vortex_bail!("Invalid hnswlib native build shape");
        }
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|err| vortex_err!("{err}"))?;
        let mut error = [0u8; 1024];
        // SAFETY: live slice is exactly rows*dimension; native code only borrows it,
        // catches all exceptions and writes at most 1024 bytes to the error buffer.
        let code = unsafe {
            vortex_hnswlib_build(
                path.as_ptr(),
                vectors.as_ptr(),
                bundle.dimension,
                bundle.rows,
                bundle.m,
                bundle.ef_construction,
                options.seed,
                options.threads,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)
    }

    pub(crate) fn open(path: &Path, bundle: HnswBundle) -> VortexResult<Self> {
        supported_cpu()?;
        validate::native_file(path, bundle)?;
        let path = CString::new(path.as_os_str().as_bytes()).map_err(|err| vortex_err!("{err}"))?;
        let mut output = null_mut();
        let mut error = [0u8; 1024];
        // SAFETY: private lease holds a structurally validated, immutable native
        // file. All output pointers live through the call and exceptions are caught.
        let code = unsafe {
            vortex_hnswlib_open(
                path.as_ptr(),
                bundle.dimension,
                bundle.rows,
                bundle.m,
                &raw mut output,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        Ok(Self {
            handle: NonNull::new(output).ok_or_else(|| vortex_err!("Null hnswlib handle"))?,
            bundle,
        })
    }

    pub(crate) fn search(&self, query: &[f32], k: u32, ef: u32) -> VortexResult<Vec<(u32, f32)>> {
        if query.len() != self.bundle.dimension as usize
            || k == 0
            || k > self.bundle.rows
            || ef < k
            || ef > 1_048_576
            || query.iter().any(|value| !value.is_finite())
        {
            vortex_bail!("Invalid hnswlib native query shape or options");
        }
        let mut ids = vec![u32::MAX; k as usize];
        let mut distances = vec![f32::NAN; k as usize];
        let mut count = 0u32;
        let mut error = [0u8; 1024];
        // SAFETY: live owned handle, dimension-sized input and k-sized output
        // buffers. The C++ mutex serializes this synchronous call and ef mutation.
        let code = unsafe {
            vortex_hnswlib_search(
                self.handle.as_ptr(),
                query.as_ptr(),
                k,
                ef,
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
                &raw mut count,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        if count > k {
            vortex_bail!("hnswlib returned more than k results");
        }
        ids.truncate(count as usize);
        distances.truncate(count as usize);
        ids.into_iter()
            .zip(distances)
            .map(|(id, distance)| {
                if id >= self.bundle.rows || !distance.is_finite() || distance < 0.0 {
                    vortex_bail!("Invalid hnswlib native result");
                }
                Ok((id, distance))
            })
            .collect()
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        // SAFETY: unique native ownership and no outstanding borrowed operations.
        unsafe { vortex_hnswlib_close(self.handle.as_ptr()) };
    }
}
