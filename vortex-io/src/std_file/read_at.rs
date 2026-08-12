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
use crate::std_file::nowait::max_nowait_bytes;
use crate::std_file::nowait::read_at_nowait;

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

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

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

        // Fast path: try to serve the read straight from the page cache on the calling thread.
        // `read_at_nowait` never blocks, so the only cost of a miss is one extra syscall, while a
        // hit avoids a task spawn, a thread hand-off, and the wake-up that comes back with it.
        // Larger reads skip this entirely so that their copy still fans out across the blocking
        // pool; see `max_nowait_bytes`.
        if length <= max_nowait_bytes() {
            let mut buffer = match allocator.allocate(length, alignment) {
                Ok(buffer) => buffer,
                Err(e) => return futures::future::ready(Err(e.into())).boxed(),
            };
            let filled = read_at_nowait(&file, buffer.as_mut_slice(), offset);
            if filled == length {
                return futures::future::ready(Ok(BufferHandle::new_host(buffer.freeze()))).boxed();
            }

            // Miss (or partial hit): finish the read on a blocking thread, keeping whatever the
            // fast path already filled.
            return async move {
                handle
                    .spawn_blocking(move || {
                        read_exact_at(
                            &file,
                            &mut buffer.as_mut_slice()[filled..],
                            offset + filled as u64,
                        )?;
                        Ok(BufferHandle::new_host(buffer.freeze()))
                    })
                    .await
            }
            .boxed();
        }

        async move {
            handle
                .spawn_blocking(move || {
                    let mut buffer = allocator.allocate(length, alignment)?;
                    read_exact_at(&file, buffer.as_mut_slice(), offset)?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}
