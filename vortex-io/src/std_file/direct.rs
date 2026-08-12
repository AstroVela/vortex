// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct I/O (`O_DIRECT`) support for local files.
//!
//! Direct I/O bypasses the operating system page cache, which removes a kernel-to-userspace copy
//! and stops large scans from evicting the rest of the machine's working set. In exchange, the
//! kernel enforces alignment constraints on the file offset, the transfer length, and the address
//! of the userspace buffer, and it performs no readahead.
//!
//! Callers issue logical reads at arbitrary offsets and lengths. [`DirectIoRange::widen`] expands
//! each request out to the filesystem's block boundaries, and the extra bytes are sliced away once
//! the transfer completes.

use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;

use rustix::fs::AtFlags;
use rustix::fs::Mode;
use rustix::fs::OFlags;
use rustix::fs::StatxFlags;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

/// Alignment assumed when Linux cannot report the filesystem's direct-I/O constraints.
///
/// A page-sized fallback is accepted by common block devices and filesystems. If the real
/// requirement is stricter, reads fail with the underlying `EINVAL` rather than silently
/// corrupting data.
pub const FALLBACK_DIRECT_IO_ALIGNMENT: usize = 4096;

/// The alignment constraints a filesystem imposes on direct I/O.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectIoConstraints {
    /// Required alignment of the address of the userspace I/O buffer.
    pub memory_alignment: usize,
    /// Required alignment of both the file offset and the I/O length.
    pub offset_alignment: usize,
}

impl Default for DirectIoConstraints {
    fn default() -> Self {
        Self {
            memory_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
            offset_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
        }
    }
}

impl DirectIoConstraints {
    /// Probe the direct-I/O constraints of the filesystem backing `file`.
    ///
    /// Falls back to [`FALLBACK_DIRECT_IO_ALIGNMENT`] when the kernel does not report
    /// `STATX_DIOALIGN`, which is the case before Linux 6.1.
    pub fn probe(file: &File) -> VortexResult<Self> {
        let Ok(stat) = rustix::fs::statx(
            file,
            c"",
            AtFlags::EMPTY_PATH | AtFlags::STATX_DONT_SYNC,
            StatxFlags::DIOALIGN,
        ) else {
            return Ok(Self::default());
        };
        if stat.stx_mask & StatxFlags::DIOALIGN.bits() == 0 {
            return Ok(Self::default());
        }

        let (Ok(memory_alignment), Ok(offset_alignment)) = (
            usize::try_from(stat.stx_dio_mem_align),
            usize::try_from(stat.stx_dio_offset_align),
        ) else {
            return Ok(Self::default());
        };
        if memory_alignment == 0 || offset_alignment == 0 {
            return Ok(Self::default());
        }
        vortex_ensure!(
            memory_alignment.is_power_of_two(),
            "direct I/O memory alignment must be a power of two, got {memory_alignment}"
        );
        vortex_ensure!(
            offset_alignment.is_power_of_two(),
            "direct I/O offset alignment must be a power of two, got {offset_alignment}"
        );

        Ok(Self {
            memory_alignment,
            offset_alignment,
        })
    }

    /// The buffer alignment to request from a [`HostAllocator`][vortex_array::memory::HostAllocator]
    /// so that the resulting buffer address satisfies this filesystem.
    ///
    /// The read path slices the requested range back out of the widened buffer, so the buffer must
    /// also be aligned strongly enough for the caller's own alignment to survive that slice. See
    /// [`DirectIoRange::widen`] for why taking the maximum of the two is sufficient.
    pub fn buffer_alignment(&self, requested: Alignment) -> Alignment {
        Alignment::new(self.memory_alignment.max(*requested))
    }
}

/// A logical read widened to the block boundaries that direct I/O requires.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectIoRange {
    /// Block-aligned file offset to read from.
    pub read_offset: u64,
    /// Block-aligned number of bytes to transfer.
    pub read_length: usize,
    /// Range within the transferred buffer holding the originally requested bytes.
    pub requested_range: Range<usize>,
}

impl DirectIoRange {
    /// Widen a logical `offset..offset + length` read out to `alignment` boundaries.
    ///
    /// The prefix skipped by `requested_range` is a multiple of `alignment`. Provided the buffer
    /// itself is `alignment`-aligned, the requested bytes therefore remain aligned to any power of
    /// two up to `alignment`, so the slice back out of the buffer stays zero-copy.
    pub fn widen(offset: u64, length: usize, alignment: usize) -> VortexResult<Self> {
        vortex_ensure!(alignment > 0, "direct I/O alignment must be non-zero");
        if length == 0 {
            return Ok(Self {
                read_offset: offset,
                read_length: 0,
                requested_range: 0..0,
            });
        }

        let alignment_u64 = u64::try_from(alignment)?;
        let requested_end = offset
            .checked_add(u64::try_from(length)?)
            .ok_or_else(|| vortex_err!("direct I/O range overflow: {offset}+{length}"))?;
        let read_offset = offset - offset % alignment_u64;
        let read_end = requested_end
            .checked_next_multiple_of(alignment_u64)
            .ok_or_else(|| vortex_err!("direct I/O aligned end overflow"))?;
        let slice_start = usize::try_from(offset - read_offset)?;

        Ok(Self {
            read_offset,
            read_length: usize::try_from(read_end - read_offset)?,
            requested_range: slice_start..slice_start + length,
        })
    }
}

/// Open `path` for reading with `O_DIRECT`.
pub fn open_direct(path: &Path) -> io::Result<File> {
    rustix::fs::open(
        path,
        OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECT,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(io::Error::from)
}

/// Whether direct I/O can actually be used for `path`.
///
/// `O_DIRECT` is rejected outright by several filesystems that are otherwise perfectly ordinary to
/// run against, most notably tmpfs, so this must be probed per path rather than assumed from the
/// platform.
pub fn is_direct_io_available(path: &Path) -> bool {
    open_direct(path).is_ok()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty(0, 0, 4096, 0, 0, 0)]
    #[case::empty_at_offset(5, 0, 4096, 5, 0, 0)]
    #[case::widens_both_ends(5, 10, 4096, 0, 4096, 5)]
    #[case::spans_two_blocks(4090, 20, 4096, 0, 8192, 4090)]
    #[case::already_aligned(4096, 4096, 4096, 4096, 4096, 0)]
    #[case::smaller_block(513, 1, 512, 512, 512, 1)]
    #[case::multi_block(4096, 8193, 4096, 4096, 12288, 0)]
    fn widens_to_block_boundaries(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
        #[case] expected_offset: u64,
        #[case] expected_length: usize,
        #[case] expected_prefix: usize,
    ) -> VortexResult<()> {
        assert_eq!(
            DirectIoRange::widen(offset, length, alignment)?,
            DirectIoRange {
                read_offset: expected_offset,
                read_length: expected_length,
                requested_range: expected_prefix..expected_prefix + length,
            }
        );
        Ok(())
    }

    #[rstest]
    #[case::offset_overflow(u64::MAX, 2, 4096)]
    #[case::zero_alignment(0, 1, 0)]
    fn rejects_invalid_range(#[case] offset: u64, #[case] length: usize, #[case] alignment: usize) {
        assert!(DirectIoRange::widen(offset, length, alignment).is_err());
    }

    /// The slice back out of a widened read must preserve the caller's alignment, because
    /// `CoalescedRequest::resolve` panics rather than copies when it does not.
    #[rstest]
    #[case(256, 4096)]
    #[case(64, 4096)]
    #[case(4096, 4096)]
    fn preserves_requested_alignment(
        #[case] requested: usize,
        #[case] block: usize,
    ) -> VortexResult<()> {
        for multiple in 0..32u64 {
            let offset = multiple * requested as u64;
            let range = DirectIoRange::widen(offset, 1024, block)?;
            assert!(
                range.requested_range.start.is_multiple_of(requested),
                "prefix {} is not a multiple of {requested}",
                range.requested_range.start
            );
        }
        Ok(())
    }
}
