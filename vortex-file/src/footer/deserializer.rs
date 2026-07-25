// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use flatbuffers::root;
use vortex_array::ArrayId;
use vortex_array::dtype::DType;
use vortex_array::session::ArraySessionExt;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_flatbuffers::FlatBuffer;
use vortex_flatbuffers::ReadFlatBuffer;
use vortex_session::VortexSession;

use crate::EOF_SIZE;
use crate::Footer;
use crate::MAGIC_BYTES;
use crate::VERSION;
use crate::footer::FileStatistics;
use crate::footer::kernels::EmbeddedKernel;
use crate::footer::kernels::EmbeddedKernelSession;
use crate::footer::postscript::Postscript;
use crate::footer::postscript::PostscriptKernel;
use crate::footer::postscript::PostscriptSegment;

/// Deserialize a footer from the end of a Vortex file or created from a
/// [`crate::footer::FooterSerializer`].
///
/// The deserializer is incremental because callers may initially read only the tail of a file. Call
/// [`deserialize`](Self::deserialize) until it returns [`DeserializeStep::Done`]. If it asks for
/// [`DeserializeStep::NeedMoreData`], prefix the requested bytes with [`prefix_data`](Self::prefix_data).
/// If it asks for [`DeserializeStep::NeedFileSize`], call [`with_size`](Self::with_size) and retry.
pub struct FooterDeserializer {
    // A buffer representing the end of a Vortex file.
    // During deserialization, we may need to expand this buffer by requesting more data from
    // the caller.
    buffer: ByteBuffer,
    // The session to use for deserialization.
    session: VortexSession,
    // The DType, if provided externally.
    dtype: Option<DType>,

    // Internal state that we accumulate

    // The file size, possibly provided externally.
    file_size: Option<u64>,
    // The postscript, once we've parsed it.
    postscript: Option<Postscript>,
}

impl FooterDeserializer {
    pub(super) fn new(initial_read: ByteBuffer, session: VortexSession) -> Self {
        Self {
            buffer: initial_read,
            session,
            dtype: None,
            file_size: None,
            postscript: None,
        }
    }

    /// Provide the file dtype externally.
    ///
    /// This is required for files written with [`VortexWriteOptions::exclude_dtype`](crate::VortexWriteOptions::exclude_dtype).
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    /// Provide or clear the externally known file dtype.
    pub fn with_some_dtype(mut self, dtype: Option<DType>) -> Self {
        self.dtype = dtype;
        self
    }

    /// Provide the total file size.
    pub fn with_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    /// Provide or clear the total file size.
    pub fn with_some_size(mut self, file_size: Option<u64>) -> Self {
        self.file_size = file_size;
        self
    }

    /// Prefix more data to the existing buffer when requested by the deserializer.
    pub fn prefix_data(&mut self, more_data: ByteBuffer) {
        let mut buffer = ByteBufferMut::with_capacity(self.buffer.len() + more_data.len());
        buffer.extend_from_slice(&more_data);
        buffer.extend_from_slice(&self.buffer);
        self.buffer = buffer.freeze();
    }

    /// The session used for deserialization.
    ///
    /// Once [`deserialize`](Self::deserialize) has returned [`DeserializeStep::Done`], this is the
    /// session the footer was parsed with — which, for a file embedding decoder kernels, is a
    /// file-scoped session extended with those kernels' encodings. Readers must use it for
    /// subsequent scans of the file, not the session they passed in.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Advance footer deserialization.
    ///
    /// Returns the next missing input requirement or the finished [`Footer`].
    pub fn deserialize(&mut self) -> VortexResult<DeserializeStep> {
        let postscript = if let Some(postscript) = &self.postscript {
            postscript
        } else {
            self.postscript = Some(self.parse_postscript(&self.buffer)?);
            self.postscript
                .as_ref()
                .vortex_expect("Just set postscript")
        };

        // If we haven't been provided a DType, we must read one from the file.
        let dtype_segment = self
            .dtype
            .is_none()
            .then(|| {
                postscript.dtype.as_ref().ok_or_else(|| {
                    vortex_err!(
                        "Vortex file doesn't embed a DType and none provided to VortexOpenOptions"
                    )
                })
            })
            .transpose()?
            .cloned();

        // Kernels for encodings we can already decode natively are never fetched: a native decoder
        // supersedes an embedded one, so their bytes would be dead weight.
        let wanted_kernels = self.wanted_kernels(postscript);

        // Copy the remaining segment locations out so the borrow of `self.postscript` ends here;
        // loading kernels below replaces `self.session`.
        let stats_segment = postscript.statistics.clone();
        let layout_segment = postscript.layout.clone();
        let footer_segment = postscript.footer.clone();

        // The other postscript segments are required, so now we figure out our the offset that
        // contains all the required segments.

        // The initial offset is the file size - the size of our initial read.
        let Some(file_size) = self.file_size else {
            return Ok(DeserializeStep::NeedFileSize);
        };
        let initial_offset = file_size - (self.buffer.len() as u64);

        let mut read_more_offset = initial_offset;
        if let Some(dtype_segment) = &dtype_segment {
            read_more_offset = read_more_offset.min(dtype_segment.offset);
        }
        if let Some(stats_segment) = &stats_segment {
            read_more_offset = read_more_offset.min(stats_segment.offset);
        }
        for kernel in &wanted_kernels {
            read_more_offset = read_more_offset.min(kernel.segment.offset);
        }
        read_more_offset = read_more_offset.min(layout_segment.offset);
        read_more_offset = read_more_offset.min(footer_segment.offset);

        // Read more bytes if necessary.
        if read_more_offset < initial_offset {
            tracing::trace!(
                "Initial read from {initial_offset} did not cover all footer segments, reading from {read_more_offset}"
            );
            return Ok(DeserializeStep::NeedMoreData {
                offset: read_more_offset,
                len: usize::try_from(initial_offset - read_more_offset)?,
            });
        }

        // Register the file's decoder kernels before anything resolves an encoding id against the
        // session, so the encodings they supply are visible to the layout and footer below.
        self.load_kernels(initial_offset, &wanted_kernels)?;

        // Now we read our initial segments.
        let dtype = dtype_segment
            .map(|segment| self.parse_dtype(initial_offset, &self.buffer, &segment))
            .transpose()?
            .unwrap_or_else(|| self.dtype.clone().vortex_expect("DType was provided"));
        let file_stats = stats_segment
            .map(|segment| {
                self.parse_file_statistics(
                    initial_offset,
                    &self.buffer,
                    &segment,
                    &dtype,
                    &self.session,
                )
            })
            .transpose()?;

        Ok(DeserializeStep::Done(self.parse_footer(
            initial_offset,
            &self.buffer,
            &footer_segment,
            &layout_segment,
            dtype,
            file_stats,
        )?))
    }

    /// The postscript's kernels for encodings this session has no native decoder for.
    fn wanted_kernels(&self, postscript: &Postscript) -> Vec<PostscriptKernel> {
        let registry = self.session.arrays().registry();
        postscript
            .wasm_kernels
            .iter()
            .filter(|kernel| registry.find(&ArrayId::new(&kernel.id)).is_none())
            .cloned()
            .collect()
    }

    /// Hand the wanted kernels' bytes to the session's [`EmbeddedKernelLoader`], replacing
    /// `self.session` with the file-scoped session it returns.
    ///
    /// Does nothing if the file embeds no kernels this reader needs, or if no loader is installed
    /// — in the latter case the encodings simply stay unknown, and the layout below fails (or
    /// yields foreign placeholders) exactly as for any other unknown encoding.
    fn load_kernels(
        &mut self,
        initial_offset: u64,
        wanted: &[PostscriptKernel],
    ) -> VortexResult<()> {
        if wanted.is_empty() {
            return Ok(());
        }
        let Some(loader) = self
            .session
            .get_opt::<EmbeddedKernelSession>()
            .and_then(|kernels| kernels.loader())
            .cloned()
        else {
            tracing::debug!(
                "File embeds {} decoder kernel(s) for encodings this session cannot decode, but no EmbeddedKernelLoader is installed",
                wanted.len()
            );
            return Ok(());
        };

        let kernels = wanted
            .iter()
            .map(|kernel| {
                let offset = usize::try_from(kernel.segment.offset - initial_offset)?;
                let end = offset + kernel.segment.length as usize;
                // The postscript is untrusted, so slice defensively rather than panicking.
                let module = self
                    .buffer
                    .as_slice()
                    .get(offset..end)
                    .ok_or_else(|| {
                        vortex_err!(
                            "Embedded kernel segment for {} is out of bounds of the footer read",
                            kernel.id
                        )
                    })
                    .map(ByteBuffer::copy_from)?;
                Ok(EmbeddedKernel::new(
                    kernel.id.clone(),
                    kernel.abi_version,
                    module,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        self.session = loader.load(&self.session, &kernels)?;
        Ok(())
    }

    /// The current buffer being used for deserialization.
    pub fn buffer(&self) -> &ByteBuffer {
        &self.buffer
    }

    /// Parse the postscript from the initial read.
    fn parse_postscript(&self, initial_read: &[u8]) -> VortexResult<Postscript> {
        if initial_read.len() < EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least EOF_SIZE ({}) bytes",
                EOF_SIZE
            );
        }
        let eof_loc = initial_read.len() - EOF_SIZE;
        let magic_bytes_loc = eof_loc + (EOF_SIZE - MAGIC_BYTES.len());

        let magic_number = &initial_read[magic_bytes_loc..];
        if magic_number != MAGIC_BYTES {
            vortex_bail!("Malformed file, invalid magic bytes, got {magic_number:?}")
        }

        let version = u16::from_le_bytes(
            initial_read[eof_loc..eof_loc + 2]
                .try_into()
                .map_err(|e| vortex_err!("Version was not a u16 {e}"))?,
        );
        if version != VERSION {
            vortex_bail!("Malformed file, unsupported version {version}")
        }

        let ps_size = u16::from_le_bytes(
            initial_read[eof_loc + 2..eof_loc + 4]
                .try_into()
                .map_err(|e| vortex_err!("Postscript size was not a u16 {e}"))?,
        ) as usize;

        if initial_read.len() < ps_size + EOF_SIZE {
            vortex_bail!(
                "Initial read must be at least {} bytes to include the Postscript",
                ps_size + EOF_SIZE
            );
        }

        Postscript::read_flatbuffer_bytes(&initial_read[eof_loc - ps_size..eof_loc])
    }

    /// Parse the DType from the initial read.
    fn parse_dtype(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
    ) -> VortexResult<DType> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);
        DType::from_flatbuffer(sliced_buffer, &self.session)
    }

    /// Parse the [`FileStatistics`] from the initial read buffer.
    fn parse_file_statistics(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        segment: &PostscriptSegment,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<FileStatistics> {
        let offset = usize::try_from(segment.offset - initial_offset)?;
        let sliced_buffer =
            FlatBuffer::copy_from(&initial_read[offset..offset + (segment.length as usize)]);

        let fb = root::<vortex_flatbuffers::footer::FileStatistics>(&sliced_buffer)?;
        FileStatistics::from_flatbuffer(&fb, dtype, session)
    }

    /// Parse the rest of the footer from the initial read.
    fn parse_footer(
        &self,
        initial_offset: u64,
        initial_read: &[u8],
        footer_segment: &PostscriptSegment,
        layout_segment: &PostscriptSegment,
        dtype: DType,
        file_stats: Option<FileStatistics>,
    ) -> VortexResult<Footer> {
        let footer_offset = usize::try_from(footer_segment.offset - initial_offset)?;
        let footer_bytes = FlatBuffer::copy_from(
            &initial_read[footer_offset..footer_offset + (footer_segment.length as usize)],
        );

        let layout_offset = usize::try_from(layout_segment.offset - initial_offset)?;
        let layout_bytes = FlatBuffer::copy_from(
            &initial_read[layout_offset..layout_offset + (layout_segment.length as usize)],
        );

        Footer::from_flatbuffer(footer_bytes, layout_bytes, dtype, file_stats, &self.session)
    }
}

#[derive(Debug)]
/// Result of one [`FooterDeserializer::deserialize`] step.
pub enum DeserializeStep {
    /// Additional data needed to continue deserialization.
    NeedMoreData {
        /// Absolute file offset to read from.
        offset: u64,
        /// Number of bytes to read and prefix into the deserializer.
        len: usize,
    },
    /// The total file size is required before offsets can be resolved.
    NeedFileSize,
    /// Footer deserialization is complete.
    Done(Footer),
}
