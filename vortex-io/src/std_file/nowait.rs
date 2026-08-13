// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Non-blocking page-cache reads via `preadv2(2)` with `RWF_NOWAIT`.
//!
//! `RWF_NOWAIT` asks the kernel to return `EAGAIN` rather than sleeping whenever a read cannot
//! be satisfied without initiating I/O. That lets us serve page-cache hits directly on the
//! calling thread and only pay the cost of handing the request to a blocking pool when the data
//! is actually cold.

use std::fs::File;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

/// Whether the non-blocking read fast path is enabled at all.
///
/// Set `VORTEX_IO_NOWAIT=0` to disable it, primarily so that the fast path can be A/B benchmarked
/// against the plain `spawn_blocking` + `pread` path.
#[cfg(target_os = "linux")]
fn enabled() -> bool {
    use std::sync::LazyLock;

    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        !matches!(
            std::env::var("VORTEX_IO_NOWAIT").as_deref(),
            Ok("0") | Ok("false")
        )
    });
    *ENABLED
}

/// Largest read served on the calling thread.
///
/// The cap is not about syscall cost, which is flat in the read size. It is about where the
/// `memcpy` out of the page cache runs. Taking the fast path moves that copy onto the thread
/// polling the read, so concurrent reads that the blocking pool would have copied in parallel
/// become serial. The saved hand-off is a fixed benefit; the lost parallelism grows with the
/// copy. The two cross over once the copy is large enough to dominate.
///
/// Measured on a warm page cache, as fast path versus blocking pool (>1 favours the fast path):
///
/// | read size | 4 concurrent | 16 concurrent |
/// |-----------|--------------|---------------|
/// | 8KiB      | 6.1x         | 3.4x          |
/// | 128KiB    | 1.43x        | 1.29x         |
/// | 256KiB    | 0.76x        | 1.37x         |
/// | 1MiB      | 0.82x        | 0.80x         |
/// | 4MiB      | 0.51x        | —             |
///
/// 128KiB is the largest size that still wins at every concurrency measured; past it the result
/// depends on how many reads are in flight, which this code cannot know. Machines with more cores
/// have more parallelism to lose and will cross over sooner, so tune with
/// `VORTEX_IO_NOWAIT_MAX_BYTES` rather than assuming this default transfers.
pub fn max_nowait_bytes() -> usize {
    #[cfg(target_os = "linux")]
    {
        use std::sync::LazyLock;

        static MAX: LazyLock<usize> = LazyLock::new(|| {
            std::env::var("VORTEX_IO_NOWAIT_MAX_BYTES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(128 * 1024)
        });
        *MAX
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Cleared permanently the first time the kernel tells us `preadv2`/`RWF_NOWAIT` is unavailable,
/// so that unsupported kernels and filesystems pay the syscall cost only once.
#[cfg(target_os = "linux")]
static SUPPORTED: AtomicBool = AtomicBool::new(true);

// Keep the static referenced on non-Linux builds to avoid a dead-code warning.
#[cfg(not(target_os = "linux"))]
static SUPPORTED: AtomicBool = AtomicBool::new(false);

/// Attempt to fill `buffer` from `file` at `offset` without ever blocking.
///
/// Returns the number of bytes filled. A return value shorter than `buffer` means the remainder
/// was not resident in the page cache (or the file ended); the caller must complete the read on a
/// blocking thread. Errors are deliberately not surfaced: the blocking fallback re-issues the read
/// and reports the failure with its usual semantics.
pub fn read_at_nowait(file: &File, buffer: &mut [u8], offset: u64) -> usize {
    #[cfg(target_os = "linux")]
    {
        if buffer.len() > max_nowait_bytes() || !enabled() || !SUPPORTED.load(Ordering::Relaxed) {
            return 0;
        }

        let mut filled = 0;
        while filled < buffer.len() {
            match preadv2_nowait(file, &mut buffer[filled..], offset + filled as u64) {
                Some(0) | None => break,
                Some(n) => filled += n,
            }
        }
        filled
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, buffer, offset);
        0
    }
}

/// A single `preadv2(.., RWF_NOWAIT)` call. `None` means "could not be served from cache", which
/// covers `EAGAIN` as well as any error we prefer to re-surface from the blocking path.
#[cfg(target_os = "linux")]
fn preadv2_nowait(file: &File, buffer: &mut [u8], offset: u64) -> Option<usize> {
    use std::os::fd::AsRawFd;

    let iov = libc::iovec {
        iov_base: buffer.as_mut_ptr().cast::<libc::c_void>(),
        iov_len: buffer.len(),
    };
    let iov_ptr = std::ptr::addr_of!(iov);
    // SAFETY: `iov` describes exactly the `buffer` slice, which is valid and writable for the
    // duration of the call, and `file` owns a valid file descriptor. `preadv2` does not modify
    // the file offset, so concurrent readers of the same `File` are unaffected.
    let read = unsafe {
        libc::preadv2(
            file.as_raw_fd(),
            iov_ptr,
            1,
            offset as libc::off_t,
            libc::RWF_NOWAIT,
        )
    };

    if read >= 0 {
        return Some(read as usize);
    }

    let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    // ENOSYS: kernel predates preadv2. EOPNOTSUPP/EINVAL: this kernel or filesystem does not
    // implement RWF_NOWAIT. Any of these means the flag will never work here, so stop trying.
    if matches!(errno, libc::ENOSYS | libc::EOPNOTSUPP | libc::EINVAL) {
        SUPPORTED.store(false, Ordering::Relaxed);
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn reads_cached_data() -> std::io::Result<()> {
        let mut tmp = tempfile::NamedTempFile::new()?;
        let data = (0..4096usize)
            .map(|i| i.to_le_bytes()[0])
            .collect::<Vec<_>>();
        tmp.write_all(&data)?;
        tmp.flush()?;

        let file = File::open(tmp.path())?;
        let mut buffer = vec![0u8; 1024];
        let filled = read_at_nowait(&file, &mut buffer, 64);

        // We cannot guarantee the page cache is warm, but whatever was returned must be correct.
        assert!(filled <= buffer.len());
        assert_eq!(&buffer[..filled], &data[64..64 + filled]);
        Ok(())
    }

    #[test]
    fn short_read_at_eof() -> std::io::Result<()> {
        let mut tmp = tempfile::NamedTempFile::new()?;
        tmp.write_all(&[7u8; 16])?;
        tmp.flush()?;

        let file = File::open(tmp.path())?;
        let mut buffer = vec![0u8; 64];
        let filled = read_at_nowait(&file, &mut buffer, 0);
        assert!(filled <= 16);
        Ok(())
    }
}
