// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Borrow;
use std::fs::File;
use std::io;
#[cfg(all(not(unix), not(windows)))]
use std::io::Read;
#[cfg(all(not(unix), not(windows)))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_array::memory::WritableHostBuffer;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = file.seek_read(&mut buffer[bytes_read..], offset + bytes_read as u64)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += read;
        }
        Ok(())
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

/// Read as many leading bytes of `buffer` as the page cache can serve without blocking.
///
/// Uses `preadv2(RWF_NOWAIT)`, which fails with `EAGAIN` as soon as it reaches a page that is not
/// already resident, so the return value is the length of the cached prefix of the range. Any
/// condition we cannot serve from cache — including kernels or filesystems without `RWF_NOWAIT`
/// support, and genuine I/O errors — is reported as a short read, leaving the caller to complete
/// (and, for an error, re-encounter) the rest of the range on the blocking pool.
#[cfg(target_os = "linux")]
fn read_cached_at(file: &File, buffer: &mut [u8], offset: u64) -> usize {
    use std::io::IoSliceMut;
    use std::sync::atomic::AtomicBool;

    use rustix::io::Errno;
    use rustix::io::ReadWriteFlags;

    /// Latches off the first time the kernel tells us `RWF_NOWAIT` is not available, so we stop
    /// paying for a syscall that can never succeed.
    static SUPPORTED: AtomicBool = AtomicBool::new(true);

    if !SUPPORTED.load(Ordering::Relaxed) {
        return 0;
    }

    let mut filled = 0;
    while filled < buffer.len() {
        let mut bufs = [IoSliceMut::new(&mut buffer[filled..])];
        match rustix::io::preadv2(
            file,
            &mut bufs,
            offset + filled as u64,
            ReadWriteFlags::NOWAIT,
        ) {
            // End of file. `read_exact_at` turns this into an `UnexpectedEof` error.
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(Errno::INTR) => continue,
            Err(Errno::NOSYS | Errno::OPNOTSUPP) => {
                SUPPORTED.store(false, Ordering::Relaxed);
                break;
            }
            // `EAGAIN` (the next page is not resident) and any other error: hand the remainder
            // to the blocking path.
            Err(_) => break,
        }
    }
    filled
}

#[cfg(not(target_os = "linux"))]
fn read_cached_at(_file: &File, _buffer: &mut [u8], _offset: u64) -> usize {
    0
}

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// Smallest request length for which we attempt a page-cache-only read before dispatching to the
/// blocking pool.
///
/// Below this size the read is dominated by fixed per-request overhead, and the extra `preadv2`
/// syscall is not worth it.
pub const NOWAIT_MIN_READ_LENGTH: usize = 64 * 1024;

/// Largest request length for which we attempt a page-cache-only read.
///
/// The fast path trades a blocking-pool round trip for copying the range on the calling task. That
/// is a good trade only while the copy is cheap: the pool spreads concurrent copies over several
/// threads, whereas the calling task runs them one after another. Past this size the copy costs
/// far more than the hop it saves, so large reads keep going to the pool.
pub const NOWAIT_MAX_READ_LENGTH: usize = 256 * 1024;

/// The request-length window within which we try the page cache before the blocking pool.
///
/// Defaults to [`NOWAIT_MIN_READ_LENGTH`]`..=`[`NOWAIT_MAX_READ_LENGTH`]. Both bounds can be
/// overridden with the `VORTEX_IO_NOWAIT_MIN_READ_LENGTH` and `VORTEX_IO_NOWAIT_MAX_READ_LENGTH`
/// environment variables, which is useful for measuring the trade-off on a given storage stack.
/// Setting the maximum to `0` disables the fast path entirely.
fn nowait_read_window() -> (usize, usize) {
    static WINDOW: OnceLock<(usize, usize)> = OnceLock::new();
    *WINDOW.get_or_init(|| {
        let bound = |name, default| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(default)
        };
        (
            bound("VORTEX_IO_NOWAIT_MIN_READ_LENGTH", NOWAIT_MIN_READ_LENGTH),
            bound("VORTEX_IO_NOWAIT_MAX_READ_LENGTH", NOWAIT_MAX_READ_LENGTH),
        )
    })
}

/// Counts of how the page-cache fast path has fared, process-wide.
///
/// Whether a range is resident is a property of the machine, not of Vortex, so the only way to
/// know whether the fast path is earning its keep on a given workload is to count. Snapshot these
/// with [`nowait_stats`].
#[derive(Debug, Default)]
struct NowaitCounters {
    /// Reads whose length fell inside the fast-path window, so an attempt was made.
    attempted: AtomicU64,
    /// Attempts the page cache served in full, skipping the blocking pool entirely.
    hit: AtomicU64,
    /// Attempts the page cache served in part, leaving a tail for the blocking pool.
    partial: AtomicU64,
    /// Attempts that returned nothing, costing one syscall before the blocking pool ran.
    miss: AtomicU64,
    /// Reads whose length fell outside the window, so no attempt was made.
    skipped: AtomicU64,
    /// Bytes copied by the fast path rather than the blocking pool.
    bytes_served: AtomicU64,
    /// Bytes the fast path could not serve and the blocking pool had to read.
    bytes_missed: AtomicU64,
    /// Reads that reached the window check, whether or not an attempt followed. Drives logging.
    decisions: AtomicU64,
}

static NOWAIT_COUNTERS: NowaitCounters = NowaitCounters::new();

impl NowaitCounters {
    const fn new() -> Self {
        Self {
            attempted: AtomicU64::new(0),
            hit: AtomicU64::new(0),
            partial: AtomicU64::new(0),
            miss: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            bytes_served: AtomicU64::new(0),
            bytes_missed: AtomicU64::new(0),
            decisions: AtomicU64::new(0),
        }
    }
}

/// A snapshot of how the page-cache fast path has fared, process-wide.
///
/// See [`nowait_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NowaitStats {
    /// Reads whose length fell inside the fast-path window, so an attempt was made.
    pub attempted: u64,
    /// Attempts the page cache served in full, skipping the blocking pool entirely.
    pub hit: u64,
    /// Attempts the page cache served in part, leaving a tail for the blocking pool.
    pub partial: u64,
    /// Attempts that returned nothing, costing one syscall before the blocking pool ran.
    pub miss: u64,
    /// Reads whose length fell outside the window, so no attempt was made.
    pub skipped: u64,
    /// Bytes copied by the fast path rather than the blocking pool.
    pub bytes_served: u64,
    /// Bytes the fast path could not serve and the blocking pool had to read.
    pub bytes_missed: u64,
}

impl NowaitStats {
    /// The fraction of attempts the page cache served in full, or `None` if none were made.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.attempted > 0).then(|| self.hit as f64 / self.attempted as f64)
    }
}

/// Snapshot the process-wide [`NowaitStats`] for the page-cache fast path.
///
/// The counters are relaxed and read independently, so a snapshot taken while reads are in flight
/// is approximate. Take it once the work being measured has finished.
pub fn nowait_stats() -> NowaitStats {
    let c = &NOWAIT_COUNTERS;
    NowaitStats {
        attempted: c.attempted.load(Ordering::Relaxed),
        hit: c.hit.load(Ordering::Relaxed),
        partial: c.partial.load(Ordering::Relaxed),
        miss: c.miss.load(Ordering::Relaxed),
        skipped: c.skipped.load(Ordering::Relaxed),
        bytes_served: c.bytes_served.load(Ordering::Relaxed),
        bytes_missed: c.bytes_missed.load(Ordering::Relaxed),
    }
}

/// How many read decisions pass before the running totals are logged.
const NOWAIT_LOG_EVERY: u64 = 1024;

/// Record that a read reached the window check, and periodically log the running totals.
///
/// Counting decisions rather than attempts means the log still appears when the window excludes
/// every read, which is exactly the case worth noticing.
fn record_nowait_decision() {
    let decisions = NOWAIT_COUNTERS.decisions.fetch_add(1, Ordering::Relaxed) + 1;
    if !decisions.is_multiple_of(NOWAIT_LOG_EVERY) {
        return;
    }
    let stats = nowait_stats();
    let (min, max) = nowait_read_window();
    tracing::debug!(
        target: "vortex_io::nowait",
        window_min = min,
        window_max = max,
        attempted = stats.attempted,
        hit = stats.hit,
        partial = stats.partial,
        miss = stats.miss,
        skipped = stats.skipped,
        bytes_served = stats.bytes_served,
        bytes_missed = stats.bytes_missed,
        hit_rate = stats.hit_rate().unwrap_or(0.0),
        "RWF_NOWAIT page-cache fast path"
    );
}

/// Read exactly `length` bytes from `file` at `offset` into a buffer from `allocator`.
///
/// Reads inside the [`nowait_read_window`] first try to take the range straight from the OS page
/// cache on the calling task. A resident range is only a memcpy, so serving it here avoids a round
/// trip through the blocking pool entirely. Anything the cache cannot serve — the whole range in
/// the common miss case — is read on the blocking pool as before, starting from the first byte the
/// fast path did not fill.
pub async fn read_exact_at_pooled<F>(
    handle: &Handle,
    file: F,
    allocator: HostAllocatorRef,
    length: usize,
    alignment: Alignment,
    offset: u64,
) -> VortexResult<WritableHostBuffer>
where
    F: Borrow<File> + Send + 'static,
{
    record_nowait_decision();

    let (min, max) = nowait_read_window();
    if (min..=max).contains(&length) {
        let mut buffer = allocator.allocate(length, alignment)?;
        let filled = read_cached_at(file.borrow(), buffer.as_mut_slice(), offset);

        let counters = &NOWAIT_COUNTERS;
        counters.attempted.fetch_add(1, Ordering::Relaxed);
        counters
            .bytes_served
            .fetch_add(filled as u64, Ordering::Relaxed);
        counters
            .bytes_missed
            .fetch_add((length - filled) as u64, Ordering::Relaxed);
        let outcome = if filled == length {
            &counters.hit
        } else if filled == 0 {
            &counters.miss
        } else {
            &counters.partial
        };
        outcome.fetch_add(1, Ordering::Relaxed);

        if filled == length {
            return Ok(buffer);
        }
        // A short read keeps the bytes we already copied; the pool reads only the tail.
        return handle
            .spawn_blocking(move || {
                read_exact_at(
                    file.borrow(),
                    &mut buffer.as_mut_slice()[filled..],
                    offset + filled as u64,
                )?;
                Ok(buffer)
            })
            .await;
    }

    // Outside the window, allocate on the pool too, so a read that cannot use the fast path is
    // exactly as it was before it existed.
    NOWAIT_COUNTERS.skipped.fetch_add(1, Ordering::Relaxed);
    handle
        .spawn_blocking(move || {
            let mut buffer = allocator.allocate(length, alignment)?;
            read_exact_at(file.borrow(), buffer.as_mut_slice(), offset)?;
            Ok(buffer)
        })
        .await
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    allocator: HostAllocatorRef,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_with_allocator(path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Open a file for reading using a custom writable buffer allocator.
    pub fn open_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            allocator,
        })
    }
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(&self.file);
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        async move {
            let buffer =
                read_exact_at_pooled(&handle, file, allocator, length, alignment, offset).await?;
            Ok(BufferHandle::new_host(buffer.freeze()))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    // Test offsets and lengths are small literals, so narrowing casts cannot lose information.
    #![expect(clippy::cast_possible_truncation)]

    use std::io::Write;

    use rstest::rstest;
    use tempfile::NamedTempFile;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::single::block_on;

    /// A file whose contents are deterministic per byte, and long enough to exercise reads on both
    /// sides of [`NOWAIT_MIN_READ_LENGTH`].
    fn temp_file(len: usize) -> VortexResult<(NamedTempFile, Vec<u8>)> {
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let mut file = NamedTempFile::new()?;
        file.write_all(&data)?;
        file.flush()?;
        Ok((file, data))
    }

    /// Whatever prefix the page cache serves must match the file contents exactly.
    #[rstest]
    #[case(0, NOWAIT_MIN_READ_LENGTH)]
    #[case(1, NOWAIT_MIN_READ_LENGTH)]
    #[case(4096, 4 << 20)]
    fn read_cached_at_matches_file(#[case] offset: u64, #[case] length: usize) -> VortexResult<()> {
        let (file, data) = temp_file(offset as usize + length)?;
        let file = File::open(file.path())?;

        // Warm the page cache so the fast path has something to hit.
        let mut warm = vec![0u8; data.len()];
        read_exact_at(&file, &mut warm, 0)?;

        let mut buffer = vec![0u8; length];
        let filled = read_cached_at(&file, &mut buffer, offset);
        assert!(filled <= length, "read_cached_at over-filled the buffer");
        assert_eq!(
            &buffer[..filled],
            &data[offset as usize..][..filled],
            "cached prefix does not match the file contents"
        );
        Ok(())
    }

    /// A cached prefix must never be reported for a range the file does not contain.
    #[test]
    fn read_cached_at_stops_at_eof() -> VortexResult<()> {
        let length = 4 * NOWAIT_MIN_READ_LENGTH;
        let (file, data) = temp_file(length)?;
        let file = File::open(file.path())?;

        let mut warm = vec![0u8; data.len()];
        read_exact_at(&file, &mut warm, 0)?;

        let mut buffer = vec![0u8; length];
        let filled = read_cached_at(&file, &mut buffer, (length / 2) as u64);
        assert!(
            filled <= length / 2,
            "read_cached_at reported {filled} bytes past the end of a {length} byte file"
        );
        Ok(())
    }

    /// Reads inside and outside the fast-path window must return identical bytes, whether they are
    /// served from the page cache or by the blocking fallback.
    #[rstest]
    #[case(0, 1024)]
    #[case(3, NOWAIT_MIN_READ_LENGTH - 1)]
    #[case(0, NOWAIT_MIN_READ_LENGTH)]
    #[case(7, NOWAIT_MIN_READ_LENGTH + 13)]
    #[case(0, NOWAIT_MAX_READ_LENGTH)]
    #[case(5, NOWAIT_MAX_READ_LENGTH + 1)]
    #[case(1 << 20, 2 << 20)]
    fn read_at_returns_file_contents(
        #[case] offset: u64,
        #[case] length: usize,
    ) -> VortexResult<()> {
        let (file, data) = temp_file(offset as usize + length + 17)?;
        let path = file.path().to_path_buf();

        let buffer = block_on(|handle| async move {
            FileReadAt::open(&path, handle)?
                .read_at(offset, length, Alignment::none())
                .await
        })?;

        assert_eq!(buffer.len(), length);
        assert_eq!(
            buffer.try_to_host_sync()?.as_ref(),
            &data[offset as usize..][..length]
        );
        Ok(())
    }

    /// A fast-path read that runs off the end of the file must still surface `UnexpectedEof` from
    /// the blocking fallback rather than silently returning a short buffer.
    #[test]
    fn read_at_past_eof_is_an_error() -> VortexResult<()> {
        let length = 2 * NOWAIT_MIN_READ_LENGTH;
        let (file, _data) = temp_file(length)?;
        let path = file.path().to_path_buf();

        let result = block_on(|handle| async move {
            FileReadAt::open(&path, handle)?
                .read_at(0, length + 1, Alignment::none())
                .await
        });

        assert!(result.is_err(), "expected a read past EOF to fail");
        Ok(())
    }
}
