// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Safe owner of the pinned official encoder, FastScan query state and original-vector reader.

#[cfg(test)]
mod tests;

use std::env;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_index::RowAddress;

use super::storage::Routing;
use super::storage::routing;
use crate::SpFreshBundle;

unsafe extern "C" {
    fn vortex_rabitq_build(
        destination: *const c_char,
        vectors: *const f32,
        dimension: u32,
        rows: u32,
        heads: u32,
        offsets: *const u32,
        ids: *const u32,
        seed: u32,
        error: *mut c_char,
    ) -> i32;
    fn vortex_rabitq_open(
        model: *const c_char,
        native_root: *const c_char,
        dimension: u32,
        rows: u32,
        heads: u32,
        offsets: *const u32,
        ids: *const u32,
        locations: *const u64,
        row_pairs: *const u64,
        diagnostics: u32,
        handle: *mut *mut c_void,
        error: *mut c_char,
    ) -> i32;
    fn vortex_rabitq_search(
        handle: *mut c_void,
        query: *const f32,
        heads: *const u32,
        head_count: u32,
        k: u32,
        rerank: u32,
        allowed: *const u8,
        allowed_bytes: u32,
        ids: *mut u32,
        distances: *mut f32,
        count: *mut u32,
        stats: *mut u64,
        error: *mut c_char,
    ) -> i32;
    fn vortex_rabitq_close(handle: *mut c_void);
}

fn path(path: &Path) -> VortexResult<CString> {
    CString::new(path.as_os_str().as_bytes()).map_err(|err| vortex_err!("{err}"))
}

fn result(code: i32, error: &[u8; 1024]) -> VortexResult<()> {
    if code != 0 {
        let end = error
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(error.len());
        vortex_bail!(
            "Official RaBitQ: {}",
            String::from_utf8_lossy(&error[..end])
        );
    }
    Ok(())
}

fn groups(route: &Routing, bundle: SpFreshBundle) -> VortexResult<(Vec<u32>, Vec<u32>)> {
    let count = route
        .postings
        .iter()
        .try_fold(route.head_ids.len(), |count, list| {
            count.checked_add(list.len())
        })
        .ok_or_else(|| vortex_err!("Official RaBitQ routing length overflow"))?;
    if count != bundle.rows as usize || route.locations.len() != count {
        vortex_bail!("Official RaBitQ requires complete replica-1 routing");
    }
    let mut offsets = vec![0];
    let mut ids = Vec::with_capacity(count);
    for (head, list) in route.head_ids.iter().zip(&route.postings) {
        ids.push(*head);
        ids.extend_from_slice(list);
        offsets.push(u32::try_from(ids.len())?);
    }
    Ok((offsets, ids))
}

pub(crate) fn build_official_sidecar(
    root: &Path,
    bundle: SpFreshBundle,
    vectors: &[f32],
    seed: u32,
) -> VortexResult<()> {
    let route = routing(root, bundle)?;
    let (offsets, ids) = groups(&route, bundle)?;
    if (bundle.rows as usize).checked_mul(bundle.dimension as usize) != Some(vectors.len()) {
        vortex_bail!("Invalid official RaBitQ input shape");
    }
    let destination = path(&root.join("rabitq-official.bin"))?;
    let mut error = [0; 1024];
    // SAFETY: the immutable input has rows*dimension floats; offsets and IDs have
    // heads+1 and rows entries. All buffers live until the synchronous encoder returns.
    // C++ validates the replica-1 permutation, bounds and numeric range, and catches exceptions.
    let code = unsafe {
        vortex_rabitq_build(
            destination.as_ptr(),
            vectors.as_ptr(),
            bundle.dimension,
            bundle.rows,
            u32::try_from(route.head_ids.len())?,
            offsets.as_ptr(),
            ids.as_ptr(),
            seed,
            error.as_mut_ptr().cast(),
        )
    };
    result(code, &error)
}

#[derive(Debug)]
struct OfficialHandle(NonNull<c_void>);

// SAFETY: this owner has no thread affinity or borrowed buffers. Its C++ objects
// use ISA-independent allocation and synchronous calls. Mutex<OfficialHandle>
// serializes every mutation of the official prepared query and scratch storage.
unsafe impl Send for OfficialHandle {}

impl Drop for OfficialHandle {
    fn drop(&mut self) {
        // SAFETY: this is the unique owner of a successfully opened handle;
        // the enclosing mutex cannot be dropped while a search still borrows it.
        unsafe { vortex_rabitq_close(self.0.as_ptr()) };
    }
}

#[derive(Debug)]
pub(crate) struct OfficialIndex {
    handle: Mutex<OfficialHandle>,
    bundle: SpFreshBundle,
}

#[derive(Debug, Default)]
pub(crate) struct OfficialSearchStats {
    pub scanned: usize,
    pub retained: usize,
    pub reranked: usize,
    pub exact_fallbacks: usize,
    pub original_bytes: usize,
    pub original_read_calls: usize,
    pub original_read_bytes: usize,
    pub candidate_hash: u64,
    pub shortlist_hash: u64,
    pub original_read_ns: u64,
    pub estimate_ns: u64,
    pub shortlist_ns: u64,
    pub exact_ns: u64,
}

impl OfficialIndex {
    pub(crate) fn open(
        root: &Path,
        model: &Path,
        bundle: SpFreshBundle,
        rows: &[RowAddress],
    ) -> VortexResult<Self> {
        Self::open_with_diagnostics(
            root,
            model,
            bundle,
            rows,
            env::var_os("VORTEX_SPFRESH_TRACE").is_some(),
        )
    }

    fn open_with_diagnostics(
        root: &Path,
        model: &Path,
        bundle: SpFreshBundle,
        rows: &[RowAddress],
        diagnostics: bool,
    ) -> VortexResult<Self> {
        if rows.len() != bundle.rows as usize {
            vortex_bail!("Invalid official RaBitQ row map");
        }
        let route = routing(root, bundle)?;
        let (offsets, ids) = groups(&route, bundle)?;
        let row_pairs = rows
            .iter()
            .map(|row| [row.file_id, row.row_offset])
            .collect::<Vec<_>>();
        let model = path(model)?;
        let root = path(root)?;
        let mut error = [0; 1024];
        let mut handle = std::ptr::null_mut();
        // SAFETY: validated routing and row buffers match bundle.rows. Each [u64;2]
        // contains adjacent integers; C++ copies every borrowed array before return.
        // The provider retains its immutable private native-file lease until drop.
        let code = unsafe {
            vortex_rabitq_open(
                model.as_ptr(),
                root.as_ptr(),
                bundle.dimension,
                bundle.rows,
                u32::try_from(route.head_ids.len())?,
                offsets.as_ptr(),
                ids.as_ptr(),
                route.locations.as_ptr(),
                row_pairs.as_ptr().cast(),
                u32::from(diagnostics),
                &raw mut handle,
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        Ok(Self {
            handle: Mutex::new(OfficialHandle(
                NonNull::new(handle).ok_or_else(|| vortex_err!("Null official RaBitQ handle"))?,
            )),
            bundle,
        })
    }

    pub(crate) fn search(
        &self,
        query: &[f32],
        heads: &[u32],
        k: u32,
        rerank: u32,
        allowed: Option<&[u8]>,
    ) -> VortexResult<(Vec<(u32, f32)>, OfficialSearchStats)> {
        if query.len() != self.bundle.dimension as usize
            || query.iter().any(|x| !x.is_finite())
            || k == 0
            || rerank < k
            || rerank > self.bundle.rows
            || allowed.is_some_and(|mask| mask.len() != (self.bundle.rows as usize).div_ceil(8))
        {
            vortex_bail!("Invalid official RaBitQ query shape or budget");
        }
        let mut ids = vec![0; k as usize];
        let mut distances = vec![0.0; k as usize];
        let mut count = 0;
        let mut stats = [0; 12];
        let mut error = [0; 1024];
        let handle = self.handle.lock();
        // SAFETY: the mutex exclusively lends a live handle. Query, routing and mask
        // slices have validated extents; each output has k entries and stats has 12.
        // C++ borrows them only until return and catches exceptions.
        let code = unsafe {
            vortex_rabitq_search(
                handle.0.as_ptr(),
                query.as_ptr(),
                heads.as_ptr(),
                u32::try_from(heads.len())?,
                k,
                rerank,
                allowed.map_or(std::ptr::null(), |mask| mask.as_ptr()),
                allowed.map_or(0, |_| self.bundle.rows.div_ceil(8)),
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
                &raw mut count,
                stats.as_mut_ptr(),
                error.as_mut_ptr().cast(),
            )
        };
        result(code, &error)?;
        if count > k || stats[2] > u64::from(rerank) {
            vortex_bail!("Invalid official RaBitQ output extent");
        }
        let hits = ids
            .into_iter()
            .zip(distances)
            .take(count as usize)
            .map(|(id, distance)| {
                if id >= self.bundle.rows || !distance.is_finite() || distance < 0.0 {
                    vortex_bail!("Invalid official RaBitQ result");
                }
                Ok((id, distance))
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok((
            hits,
            OfficialSearchStats {
                scanned: usize::try_from(stats[0])?,
                retained: usize::try_from(stats[1])?,
                reranked: usize::try_from(stats[2])?,
                exact_fallbacks: usize::try_from(stats[3])?,
                original_bytes: usize::try_from(stats[2])? * self.bundle.dimension as usize * 4,
                original_read_calls: usize::try_from(stats[4])?,
                original_read_bytes: usize::try_from(stats[5])?,
                candidate_hash: stats[6],
                shortlist_hash: stats[7],
                original_read_ns: stats[8],
                estimate_ns: stats[9],
                shortlist_ns: stats[10],
                exact_ns: stats[11],
            },
        ))
    }
}
