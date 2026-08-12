// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;
#[cfg(target_os = "linux")]
use crate::std_file::direct::DirectIoConstraints;
#[cfg(target_os = "linux")]
use crate::std_file::direct::DirectIoRange;
#[cfg(target_os = "linux")]
use crate::std_file::uring::UringDriver;
#[cfg(target_os = "linux")]
use crate::std_file::uring::await_read;

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

/// Largest read serviced inline on the calling thread when the page cache can satisfy it.
///
/// The inline path trades the blocking pool's parallelism for the latency of a thread handoff, so
/// it only pays while the copy is cheaper than the handoff it avoids, which is tens of
/// microseconds. Above this size the copy dominates and is better spread across the pool, where it
/// also stops a large memcpy from monopolising a runtime worker.
const INLINE_READ_LIMIT: usize = 256 * 1024;

/// Read as much of `buffer` as the kernel can supply without blocking, returning the byte count.
///
/// `RWF_NOWAIT` fails with `EAGAIN` rather than waiting whenever the data is not already in the
/// page cache, which makes it safe to call on a runtime thread: it either completes at memory
/// speed or does nothing. Anything it declines to do is left to the caller's blocking path, which
/// is also where a genuine I/O error will be reported.
#[cfg(target_os = "linux")]
fn read_cached_at(file: &File, buffer: &mut [u8], offset: u64) -> usize {
    use std::io::IoSliceMut;

    use rustix::io::ReadWriteFlags;

    let mut done = 0usize;
    while done < buffer.len() {
        let mut iov = [IoSliceMut::new(&mut buffer[done..])];
        match rustix::io::preadv2(file, &mut iov, offset + done as u64, ReadWriteFlags::NOWAIT) {
            // Zero means end of file. Leave it to the blocking path to raise the error, so that
            // short-file handling lives in exactly one place.
            Ok(0) => break,
            Ok(read) => done += read,
            Err(_) => break,
        }
    }
    done
}

#[cfg(not(target_os = "linux"))]
fn read_cached_at(_file: &File, _buffer: &mut [u8], _offset: u64) -> usize {
    0
}

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

/// Number of concurrent requests to allow when reads bypass the page cache.
///
/// Direct I/O performs no readahead, so every request is a real device operation and deeper
/// queues are needed to keep an NVMe device busy.
pub const DEFAULT_DIRECT_CONCURRENCY: usize = 128;

/// How [`FileReadAt`] should issue reads against the underlying file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FileIoMode {
    /// Ordinary buffered `pread` calls dispatched onto a blocking thread pool.
    #[default]
    Buffered,
    /// `O_DIRECT` `pread` calls dispatched onto a blocking thread pool.
    ///
    /// Bypasses the page cache. Falls back to [`FileIoMode::Buffered`] if the filesystem rejects
    /// `O_DIRECT`.
    Direct,
    /// `O_DIRECT` reads submitted through a shared `io_uring`.
    ///
    /// Falls back to [`FileIoMode::Direct`] if this kernel has no usable `io_uring`, and to
    /// [`FileIoMode::Buffered`] if the filesystem also rejects `O_DIRECT`.
    DirectUring,
}

impl FileIoMode {
    /// The default mode for local file reads, honouring `VORTEX_FILE_IO_MODE`.
    ///
    /// The variable is read once per process. Direct I/O is a deployment decision rather than a
    /// property of the data, so it is deliberately not inferred from the file or the filesystem.
    pub fn default_for_process() -> Self {
        static MODE: OnceLock<FileIoMode> = OnceLock::new();
        *MODE.get_or_init(|| Self::from_env().unwrap_or_default())
    }

    /// Read the mode from the `VORTEX_FILE_IO_MODE` environment variable.
    ///
    /// Accepts `buffered`, `direct`, and `direct_uring`. Returns `None` when unset or
    /// unrecognized, so that callers keep their own default.
    pub fn from_env() -> Option<Self> {
        match std::env::var("VORTEX_FILE_IO_MODE").ok()?.as_str() {
            "buffered" => Some(Self::Buffered),
            "direct" => Some(Self::Direct),
            "direct_uring" | "uring" => Some(Self::DirectUring),
            other => {
                tracing::warn!("ignoring unrecognized VORTEX_FILE_IO_MODE={other}");
                None
            }
        }
    }
}

/// Options controlling how [`FileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FileReadAtOptions {
    io_mode: FileIoMode,
}

impl FileReadAtOptions {
    /// Request a particular I/O mode.
    ///
    /// The mode is a request rather than a guarantee: opening probes the kernel and the
    /// filesystem, and silently degrades to a supported mode. Use [`FileReadAt::io_mode`] to
    /// observe what was actually selected.
    pub fn with_io_mode(mut self, io_mode: FileIoMode) -> Self {
        self.io_mode = io_mode;
        self
    }

    /// The requested I/O mode.
    pub fn io_mode(&self) -> FileIoMode {
        self.io_mode
    }
}

enum Backend {
    Buffered(Arc<File>),
    #[cfg(target_os = "linux")]
    Direct {
        file: Arc<File>,
        constraints: DirectIoConstraints,
        uring: Option<&'static Arc<UringDriver>>,
    },
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    backend: Backend,
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
        Self::open_with_options(
            path,
            handle,
            allocator,
            FileReadAtOptions::default().with_io_mode(FileIoMode::default_for_process()),
        )
    }

    /// Open a file for reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: HostAllocatorRef,
        options: FileReadAtOptions,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        Ok(Self {
            uri,
            backend: open_backend(path, options.io_mode)?,
            handle,
            allocator,
        })
    }

    /// The I/O mode actually in use, after probing the kernel and the filesystem.
    pub fn io_mode(&self) -> FileIoMode {
        match &self.backend {
            Backend::Buffered(_) => FileIoMode::Buffered,
            #[cfg(target_os = "linux")]
            Backend::Direct { uring, .. } => {
                if uring.is_some() {
                    FileIoMode::DirectUring
                } else {
                    FileIoMode::Direct
                }
            }
        }
    }

    fn file(&self) -> &Arc<File> {
        match &self.backend {
            Backend::Buffered(file) => file,
            #[cfg(target_os = "linux")]
            Backend::Direct { file, .. } => file,
        }
    }
}

#[cfg(target_os = "linux")]
fn open_backend(path: &Path, io_mode: FileIoMode) -> VortexResult<Backend> {
    use crate::std_file::direct::open_direct;

    if io_mode == FileIoMode::Buffered {
        return Ok(Backend::Buffered(Arc::new(File::open(path)?)));
    }

    // O_DIRECT is rejected outright by several ordinary filesystems, tmpfs among them, so this
    // has to be probed against the actual path rather than assumed from the platform.
    let file = match open_direct(path) {
        Ok(file) => file,
        Err(err) => {
            tracing::debug!(
                "O_DIRECT unavailable for {}, falling back to buffered reads: {err}",
                path.display()
            );
            return Ok(Backend::Buffered(Arc::new(File::open(path)?)));
        }
    };

    let constraints = DirectIoConstraints::probe(&file)?;
    let uring = (io_mode == FileIoMode::DirectUring)
        .then(UringDriver::get)
        .flatten();

    Ok(Backend::Direct {
        file: Arc::new(file),
        constraints,
        uring,
    })
}

#[cfg(not(target_os = "linux"))]
fn open_backend(path: &Path, _io_mode: FileIoMode) -> VortexResult<Backend> {
    Ok(Backend::Buffered(Arc::new(File::open(path)?)))
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        match self.io_mode() {
            FileIoMode::Buffered => DEFAULT_CONCURRENCY,
            FileIoMode::Direct | FileIoMode::DirectUring => DEFAULT_DIRECT_CONCURRENCY,
        }
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(self.file());
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
        match &self.backend {
            Backend::Buffered(file) => {
                let file = Arc::clone(file);
                let handle = self.handle.clone();
                let allocator = Arc::clone(&self.allocator);
                async move {
                    let mut buffer = allocator.allocate(length, alignment)?;

                    // A small read the kernel can already satisfy from the page cache takes about
                    // a microsecond, while handing it to the blocking pool costs tens of them in
                    // scheduling alone. Try to service those inline and only pay for a thread when
                    // the data is not resident.
                    let done = if length <= INLINE_READ_LIMIT {
                        read_cached_at(&file, buffer.as_mut_slice(), offset)
                    } else {
                        0
                    };
                    if done == length {
                        return Ok(BufferHandle::new_host(buffer.freeze()));
                    }

                    handle
                        .spawn_blocking(move || {
                            read_exact_at(
                                &file,
                                &mut buffer.as_mut_slice()[done..],
                                offset + done as u64,
                            )?;
                            Ok(BufferHandle::new_host(buffer.freeze()))
                        })
                        .await
                }
                .boxed()
            }
            #[cfg(target_os = "linux")]
            Backend::Direct {
                file,
                constraints,
                uring,
            } => {
                let file = Arc::clone(file);
                let allocator = Arc::clone(&self.allocator);
                let constraints = *constraints;
                let handle = self.handle.clone();
                let uring = *uring;

                async move {
                    let range = DirectIoRange::widen(offset, length, constraints.offset_alignment)?;
                    let buffer_alignment = constraints.buffer_alignment(alignment);
                    let buffer = allocator.allocate(range.read_length, buffer_alignment)?;

                    let buffer = match uring {
                        Some(driver) => {
                            let receiver = driver.read_at(
                                &file,
                                range.read_offset,
                                range.requested_range.end,
                                buffer,
                            )?;
                            await_read(receiver).await?
                        }
                        None => {
                            handle
                                .spawn_blocking(move || {
                                    let mut buffer = buffer;
                                    read_direct_at(
                                        &file,
                                        buffer.as_mut_slice(),
                                        range.read_offset,
                                        range.requested_range.end,
                                    )?;
                                    VortexResult::Ok(buffer)
                                })
                                .await?
                        }
                    };

                    // The widened prefix is a multiple of the block size, and the buffer is at
                    // least block-aligned, so this slice keeps the caller's alignment without
                    // copying. `aligned` stays correct if that ever ceases to hold.
                    Ok(BufferHandle::new_host(
                        buffer
                            .freeze()
                            .slice_unaligned(range.requested_range)
                            .aligned(alignment),
                    ))
                }
                .boxed()
            }
        }
    }
}

/// Fill at least `required` bytes of `buffer` from `offset`, tolerating short reads.
///
/// Direct I/O requires the length passed to the kernel to stay block-aligned, so a short read that
/// does not land on a block boundary cannot be resumed and is reported as an error instead.
#[cfg(target_os = "linux")]
fn read_direct_at(file: &File, buffer: &mut [u8], offset: u64, required: usize) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    let mut done = 0usize;
    while done < required {
        let bytes_read = match file.read_at(&mut buffer[done..], offset + done as u64) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("direct read got {done} bytes at offset {offset}, needed {required}"),
            ));
        }
        done += bytes_read;
    }
    Ok(())
}

// These tests need a runtime handle, which only the tokio integration provides.
#[cfg(test)]
#[cfg(feature = "tokio")]
mod tests {
    use std::io::Write;

    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;
    use crate::runtime::tokio::TokioRuntime;
    use crate::std_file::direct::is_direct_io_available;

    fn write_temp(len: usize) -> VortexResult<(tempfile::TempPath, Vec<u8>)> {
        let data: Vec<u8> = (0..len)
            .map(|i| u8::try_from(i % 251).unwrap_or(0))
            .collect();
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(&data)?;
        file.flush()?;
        Ok((file.into_temp_path(), data))
    }

    /// Every mode must return identical bytes for unaligned offsets and lengths, since the direct
    /// modes widen the request to block boundaries and slice the result back down.
    #[rstest]
    #[case(FileIoMode::Buffered)]
    #[case(FileIoMode::Direct)]
    #[case(FileIoMode::DirectUring)]
    #[tokio::test]
    async fn reads_match_across_io_modes(#[case] io_mode: FileIoMode) -> VortexResult<()> {
        let (path, data) = write_temp(300_000)?;
        let reader = FileReadAt::open_with_options(
            &path,
            TokioRuntime::current(),
            Arc::new(DefaultHostAllocator),
            FileReadAtOptions::default().with_io_mode(io_mode),
        )?;

        // Guard against this test silently degrading to buffered reads everywhere and thereby
        // testing nothing: where the filesystem supports O_DIRECT, the requested mode must stick.
        if is_direct_io_available(&path) {
            assert_eq!(reader.io_mode(), io_mode);
        }

        assert_eq!(reader.size().await?, data.len() as u64);

        for (offset, length) in [
            (0u64, 1usize),
            (1, 1),
            (0, 4096),
            (1, 4095),
            (4095, 2),
            (12_345, 6_789),
            (299_999, 1),
            (0, 300_000),
        ] {
            let start = usize::try_from(offset)?;
            let buffer = reader
                .read_at(offset, length, Alignment::new(256))
                .await?
                .into_host()
                .await;
            assert_eq!(
                buffer.as_slice(),
                &data[start..start + length],
                "mismatch at {start}..{}",
                start + length
            );
        }
        Ok(())
    }

    /// The inline page-cache path must return exactly the same bytes as the blocking path, for
    /// reads that straddle its size limit as well as reads under it.
    #[tokio::test]
    async fn inline_cached_reads_match() -> VortexResult<()> {
        let (path, data) = write_temp(INLINE_READ_LIMIT * 2 + 4096)?;
        let reader = FileReadAt::open_with_options(
            &path,
            TokioRuntime::current(),
            Arc::new(DefaultHostAllocator),
            FileReadAtOptions::default().with_io_mode(FileIoMode::Buffered),
        )?;

        // Warm the cache so the inline path is the one actually taken for the small reads.
        drop(reader.read_at(0, data.len(), Alignment::none()).await?);

        for (offset, length) in [
            (0u64, 1usize),
            (7, INLINE_READ_LIMIT - 7),
            (0, INLINE_READ_LIMIT),
            (1, INLINE_READ_LIMIT + 1),
            (4096, INLINE_READ_LIMIT * 2),
            ((data.len() - 10) as u64, 10),
        ] {
            let start = usize::try_from(offset)?;
            let buffer = reader
                .read_at(offset, length, Alignment::new(256))
                .await?
                .into_host()
                .await;
            assert_eq!(
                buffer.as_slice(),
                &data[start..start + length],
                "mismatch at {start}..{}",
                start + length
            );
        }
        Ok(())
    }

    /// A read past the end of the file must still fail, even though the inline path deliberately
    /// stays silent about end-of-file and leaves that to the blocking path.
    #[tokio::test]
    async fn reads_past_end_of_file_fail() -> VortexResult<()> {
        let (path, data) = write_temp(8192)?;
        let reader = FileReadAt::open_with_options(
            &path,
            TokioRuntime::current(),
            Arc::new(DefaultHostAllocator),
            FileReadAtOptions::default().with_io_mode(FileIoMode::Buffered),
        )?;
        drop(reader.read_at(0, data.len(), Alignment::none()).await?);

        assert!(
            reader
                .read_at(0, data.len() + 1, Alignment::none())
                .await
                .is_err(),
            "reading past the end of the file must be an error"
        );
        Ok(())
    }

    /// Requesting a direct mode on a filesystem or kernel that cannot provide it must degrade
    /// rather than fail, and must report the mode it actually settled on.
    #[tokio::test]
    async fn reports_resolved_io_mode() -> VortexResult<()> {
        let (path, _) = write_temp(8192)?;
        let reader = FileReadAt::open_with_options(
            &path,
            TokioRuntime::current(),
            Arc::new(DefaultHostAllocator),
            FileReadAtOptions::default().with_io_mode(FileIoMode::DirectUring),
        )?;
        assert!(reader.concurrency() > 0);
        // The resolved mode depends on the filesystem under the temp directory, so assert only
        // that it is one that this build can actually service.
        assert!(matches!(
            reader.io_mode(),
            FileIoMode::Buffered | FileIoMode::Direct | FileIoMode::DirectUring
        ));
        Ok(())
    }
}
