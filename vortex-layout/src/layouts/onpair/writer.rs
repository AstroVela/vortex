// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join;
use futures::future::try_join_all;
use futures::stream::BoxStream;
use futures::stream::once;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_onpair::Config;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::OnPair as OnPairArrayEncoding;
use vortex_onpair::Parser;
use vortex_onpair::onpair_encode;
use vortex_onpair::onpair_train;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::compressed::CompressorPlugin;
use crate::layouts::onpair::OnPairLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Whether the default file writer should shred OnPair-eligible string columns
/// into an [`OnPairLayout`].
///
/// Off by default. The layout writes a `vortex.onpair` layout node, which the
/// edition system does not cover — a file written with it contains no
/// `vortex.onpair` *array*, so the writer's array-encoding edition check never
/// fires. Enable with `VORTEX_EXPERIMENTAL_ONPAIR_LAYOUT=1`, or per-writer with
/// `WriteStrategyBuilder::with_onpair_layout`.
pub fn use_experimental_onpair_layout() -> bool {
    static USE_EXPERIMENTAL_ONPAIR_LAYOUT: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_ONPAIR_LAYOUT").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_ONPAIR_LAYOUT
}

/// Item carried on each child sub-stream: a sequenced, materialized chunk.
type ChildChunk = VortexResult<(SequenceId, ArrayRef)>;

/// Options for OnPair layout encoding.
#[derive(Clone)]
pub struct OnPairLayoutOptions {
    /// Training configuration for the shared dictionary.
    pub config: Config,
}

impl Default for OnPairLayoutOptions {
    fn default() -> Self {
        Self {
            config: DEFAULT_CONFIG,
        }
    }
}

/// A layout strategy that shreds a string column into one shared OnPair
/// dictionary plus a chunked code stream, with a fallback for columns OnPair does
/// not suit.
///
/// The dictionary is trained on the first chunk and every subsequent chunk is
/// encoded against it, so the column pays for one dictionary rather than one per
/// chunk. Reusing a dictionary is always correct — an OnPair dictionary contains
/// all 256 single-byte tokens, so any string encodes under any dictionary — only
/// the compression ratio varies.
///
/// The stream is transposed into six sub-streams:
///  1. `dict_bytes` and `dict_offsets` carry the trained dictionary, emitted once
///     with the first chunk's sequence ids so they land next to its data.
///  2. `codes_offsets` are rebased onto the running token total, so the single
///     `codes` child is indexed by whole-column positions.
///  3. every sub-stream is written by its own strategy, concurrently.
///
/// Columns whose dtype is not `Utf8`/`Binary`, whose stream is empty, or whose
/// first chunk the probe compressor does not choose OnPair for are forwarded to
/// `fallback` unchanged.
#[derive(Clone)]
pub struct OnPairStrategy {
    dict: Arc<dyn LayoutStrategy>,
    chunks: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    options: OnPairLayoutOptions,
    probe_compressor: Arc<dyn CompressorPlugin>,
}

impl OnPairStrategy {
    /// Create a strategy writing the two dictionary children through `dict`, the
    /// four per-chunk children through `chunks`, and ineligible columns through
    /// `fallback`.
    ///
    /// `probe_compressor` decides eligibility: the layout applies only when this
    /// compressor chooses OnPair for the column's first chunk.
    pub fn new<D: LayoutStrategy, C: LayoutStrategy, F: LayoutStrategy>(
        dict: D,
        chunks: C,
        fallback: F,
        options: OnPairLayoutOptions,
        probe_compressor: Arc<dyn CompressorPlugin>,
    ) -> Self {
        Self {
            dict: Arc::new(dict),
            chunks: Arc::new(chunks),
            fallback: Arc::new(fallback),
            options,
            probe_compressor,
        }
    }
}

#[async_trait]
impl LayoutStrategy for OnPairStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !onpair_layout_supported(&dtype) {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Peek the first chunk: it decides eligibility and trains the dictionary.
        let (stream, first_chunk) = peek_first_chunk(stream).await?;
        let stream = SequentialStreamAdapter::new(dtype.clone(), stream).sendable();

        // Nothing to train on: either no chunks at all, or a leading chunk with no rows. A zero-row
        // array still streams one empty chunk, and training on it would write a minimal dictionary
        // (the 256 single-byte tokens) for a column that has no strings to encode. A non-empty stream
        // that merely *starts* with an empty chunk also falls back, which costs the layout but is
        // always correct.
        if first_chunk.as_ref().is_none_or(|chunk| chunk.is_empty()) {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }
        let first_chunk = first_chunk.vortex_expect("first chunk is present and non-empty");

        // Defer to the compressor's scheme selection rather than duplicating the
        // OnPair-vs-FSST-vs-zstd policy here. Conservative by construction: the
        // probe charges the first chunk for a dictionary this layout amortizes
        // over the whole column.
        let probe_compressor = Arc::clone(&self.probe_compressor);
        let config = self.options.config;
        let probe_session = session.clone();
        let train_chunk = first_chunk.clone();
        let trained = session
            .handle()
            .spawn_cpu(move || {
                let mut exec_ctx = probe_session.create_execution_ctx();
                if !probe_compressor
                    .compress_chunk(&train_chunk, &mut exec_ctx)?
                    .is::<OnPairArrayEncoding>()
                {
                    return Ok(None);
                }
                onpair_train(&train_chunk, config, &mut exec_ctx).map(Some)
            })
            .await?;

        let Some(parser) = trained else {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        };

        let is_nullable = dtype.is_nullable();
        // Cumulative over the whole column, so wider than any chunk's own offsets.
        let codes_offsets_dtype = non_nullable(PType::U64);

        let (dict_bytes_tx, dict_bytes_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (dict_offsets_tx, dict_offsets_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (codes_tx, codes_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (codes_offsets_tx, codes_offsets_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (lengths_tx, lengths_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (validity_tx, validity_rx) = if is_nullable {
            let (tx, rx) = kanal::bounded_async::<ChildChunk>(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let parser = Arc::new(parser);
        // Kept joined with the child writers below so producer errors surface
        // rather than being hidden as an early channel close.
        let fanout_fut = transpose_onpair_column(
            stream,
            Arc::clone(&parser),
            session.clone(),
            is_nullable,
            ChildSenders {
                dict_bytes: dict_bytes_tx,
                dict_offsets: dict_offsets_tx,
                codes: codes_tx,
                codes_offsets: codes_offsets_tx,
                lengths: lengths_tx,
                validity: validity_tx,
            },
        );

        let uncompressed_lengths_dtype = non_nullable(PType::I32);
        let mut child_specs: Vec<(
            DType,
            Arc<dyn LayoutStrategy>,
            kanal::AsyncReceiver<ChildChunk>,
        )> = vec![
            (
                non_nullable(PType::U8),
                Arc::clone(&self.dict),
                dict_bytes_rx,
            ),
            (
                non_nullable(PType::U32),
                Arc::clone(&self.dict),
                dict_offsets_rx,
            ),
            (non_nullable(PType::U16), Arc::clone(&self.chunks), codes_rx),
            (
                codes_offsets_dtype,
                Arc::clone(&self.chunks),
                codes_offsets_rx,
            ),
            (
                uncompressed_lengths_dtype,
                Arc::clone(&self.chunks),
                lengths_rx,
            ),
        ];
        if let Some(validity_rx) = validity_rx {
            child_specs.push((
                DType::Bool(Nullability::NonNullable),
                Arc::clone(&self.chunks),
                validity_rx,
            ));
        }

        let handle = session.handle();
        let layout_futures: Vec<_> = child_specs
            .into_iter()
            .map(|(child_dtype, strategy, rx)| {
                let child_stream =
                    SequentialStreamAdapter::new(child_dtype, rx.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(ctx, segment_sink, child_stream, child_eof, &session)
                        .await
                })
            })
            .collect();

        let (_, layouts) = try_join(fanout_fut, try_join_all(layout_futures)).await?;
        let mut layouts = layouts.into_iter();
        let dict_bytes = layouts.next().vortex_expect("dict_bytes layout present");
        let dict_offsets = layouts.next().vortex_expect("dict_offsets layout present");
        let codes = layouts.next().vortex_expect("codes layout present");
        let codes_offsets = layouts.next().vortex_expect("codes_offsets layout present");
        let uncompressed_lengths = layouts
            .next()
            .vortex_expect("uncompressed_lengths layout present");
        let validity = is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));

        Ok(OnPairLayout::new(
            dtype,
            dict_bytes,
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity,
        )
        .into_layout())
    }
}

/// The producer half of every child sub-stream.
struct ChildSenders {
    dict_bytes: kanal::AsyncSender<ChildChunk>,
    dict_offsets: kanal::AsyncSender<ChildChunk>,
    codes: kanal::AsyncSender<ChildChunk>,
    codes_offsets: kanal::AsyncSender<ChildChunk>,
    lengths: kanal::AsyncSender<ChildChunk>,
    validity: Option<kanal::AsyncSender<ChildChunk>>,
}

/// Encode every chunk against the shared dictionary and fan the results out to
/// the child sub-streams, rebasing each chunk's local `codes_offsets` onto the
/// running token total so the single `codes` child is indexed by whole-column
/// positions.
///
/// The dictionary children are emitted once, with the first chunk's sequence ids.
async fn transpose_onpair_column(
    mut stream: SendableSequentialStream,
    parser: Arc<Parser>,
    session: VortexSession,
    is_nullable: bool,
    senders: ChildSenders,
) -> VortexResult<()> {
    let handle = session.handle();
    let mut token_base: u64 = 0;
    let mut first = true;

    while let Some(chunk) = stream.next().await {
        let (sequence_id, array) = chunk?;

        // Allocate every child's sequence id up front and drop the pointer before
        // the first await: a SequencePointer must not be held across await points.
        let mut sp = sequence_id.descend();
        let dict_ids = first.then(|| (sp.advance(), sp.advance()));
        let codes_id = sp.advance();
        let codes_offsets_id = sp.advance();
        let lengths_id = sp.advance();
        let validity_id = is_nullable.then(|| sp.advance());
        drop(sp);

        let row_count = array.len();
        let encode_parser = Arc::clone(&parser);
        let encode_session = session.clone();
        // TODO(francesco): encoding is serial. `onpair_encode` only borrows the
        // parser, so this could run `buffered` over the stream — `buffered`
        // preserves order, and only the offset rebasing below needs to stay
        // sequential.
        let (encoded, validity) = handle
            .spawn_cpu(move || {
                let mut exec_ctx = encode_session.create_execution_ctx();
                let encoded = onpair_encode(&encode_parser, &array, &mut exec_ctx)?;
                let validity = is_nullable
                    .then(|| {
                        encoded
                            .validity
                            .execute_mask(row_count, &mut exec_ctx)
                            .map(|mask| mask.into_array())
                    })
                    .transpose()?;
                Ok::<_, vortex_error::VortexError>((encoded, validity))
            })
            .await?;

        let n_codes = u64::try_from(encoded.codes.len())?;
        let codes_offsets = global_codes_offsets(&encoded.codes_offsets, token_base, first);
        token_base += n_codes;

        if let Some((dict_bytes_id, dict_offsets_id)) = dict_ids {
            // The trainer's blob is already read-padded, which is the form the
            // reader hands to `CompactDictionary::validate_safety`.
            let dict_bytes = Buffer::copy_from(parser.dict.bytes()).into_array();
            let dict_offsets = Buffer::copy_from(parser.dict.offsets()).into_array();
            if senders
                .dict_bytes
                .send(Ok((dict_bytes_id, dict_bytes)))
                .await
                .is_err()
                || senders
                    .dict_offsets
                    .send(Ok((dict_offsets_id, dict_offsets)))
                    .await
                    .is_err()
            {
                vortex_bail!("OnPair dictionary writer finished before the dictionary was sent");
            }
        }

        if senders
            .codes
            .send(Ok((codes_id, encoded.codes.into_array())))
            .await
            .is_err()
            || senders
                .codes_offsets
                .send(Ok((codes_offsets_id, codes_offsets.into_array())))
                .await
                .is_err()
            || senders
                .lengths
                .send(Ok((lengths_id, encoded.uncompressed_lengths.into_array())))
                .await
                .is_err()
        {
            vortex_bail!("OnPair child writer finished before all chunks were sent");
        }

        if let (Some(validity_tx), Some(validity_id), Some(validity)) =
            (&senders.validity, validity_id, validity)
            && validity_tx.send(Ok((validity_id, validity))).await.is_err()
        {
            vortex_bail!("OnPair validity writer finished before all chunks were sent");
        }

        first = false;
    }

    Ok(())
}

/// Rebase a chunk's local token boundaries onto the whole-column code stream.
///
/// Every offset is shifted by `token_base`, the number of codes already emitted,
/// and the duplicated leading boundary is dropped on every chunk after the first,
/// so the concatenation of all chunks is a single monotonic `[0, .., total]` array
/// of length `row_count + 1`.
fn global_codes_offsets(local: &Buffer<u64>, token_base: u64, first: bool) -> Buffer<u64> {
    let skip = usize::from(!first);
    local
        .as_slice()
        .iter()
        .skip(skip)
        .map(|&offset| offset + token_base)
        .collect()
}

/// Whether a column's dtype can be OnPair-encoded.
pub fn onpair_layout_supported(dtype: &DType) -> bool {
    matches!(dtype, DType::Utf8(_) | DType::Binary(_))
}

fn non_nullable(ptype: PType) -> DType {
    DType::Primitive(ptype, Nullability::NonNullable)
}

/// Take the first chunk from `stream` without consuming it, so it can both decide
/// eligibility and train the dictionary before the stream is written.
async fn peek_first_chunk(
    stream: SendableSequentialStream,
) -> VortexResult<(BoxStream<'static, ChildChunk>, Option<ArrayRef>)> {
    let mut stream = stream.boxed();
    match stream.next().await {
        None => Ok((stream.boxed(), None)),
        Some(Err(e)) => Err(e),
        Some(Ok((sequence_id, chunk))) => {
            let peeked = chunk.clone();
            let reconstructed = once(async move { Ok((sequence_id, chunk)) }).chain(stream);
            Ok((reconstructed.boxed(), Some(peeked)))
        }
    }
}
