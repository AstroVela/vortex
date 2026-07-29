// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_alp::ALP;
use vortex_alp::ALPRD;
use vortex_array::ArrayContext;
use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Null;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::Variant;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::dtype::FieldPath;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fsst::FSST;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::LayoutStrategyEncodingValidator;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::collect::CollectStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::compressed::CompressorPlugin;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::writer::ListLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::table::use_experimental_list_layout;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
#[cfg(feature = "unstable_encodings")]
use vortex_onpair::OnPair;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_session::VortexSession;
use vortex_sparse::Sparse;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_zigzag::ZigZag;
#[cfg(feature = "zstd")]
use vortex_zstd::Zstd;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
use vortex_zstd::ZstdBuffers;

const ONE_MEG: u64 = 1 << 20;

/// Static registry of all allowed array encodings for file writing.
///
/// This includes all canonical encodings from vortex-array plus all compressed
/// encodings from the various encoding crates.
pub static ALLOWED_ENCODINGS: LazyLock<HashSet<ArrayId>> = LazyLock::new(|| {
    let mut allowed = HashSet::new();

    // Canonical encodings from vortex-array
    allowed.insert(Null.id());
    allowed.insert(Bool.id());
    allowed.insert(Primitive.id());
    allowed.insert(Decimal.id());
    allowed.insert(VarBin.id());
    allowed.insert(VarBinView.id());
    allowed.insert(List.id());
    allowed.insert(ListView.id());
    allowed.insert(FixedSizeList.id());
    allowed.insert(Struct.id());
    allowed.insert(Extension.id());
    allowed.insert(Chunked.id());
    allowed.insert(Constant.id());
    allowed.insert(Masked.id());
    allowed.insert(Dict.id());
    allowed.insert(Variant.id());

    // Compressed encodings from encoding crates
    allowed.insert(ALP.id());
    allowed.insert(ALPRD.id());
    allowed.insert(BitPacked.id());
    allowed.insert(ByteBool.id());
    allowed.insert(DateTimeParts.id());
    allowed.insert(DecimalByteParts.id());
    allowed.insert(Delta.id());
    allowed.insert(FoR.id());
    allowed.insert(FSST.id());
    #[cfg(feature = "unstable_encodings")]
    allowed.insert(OnPair.id());
    allowed.insert(Pco.id());
    allowed.insert(RLE.id());
    allowed.insert(RunEnd.id());
    allowed.insert(Sequence.id());
    allowed.insert(Sparse.id());
    allowed.insert(ZigZag.id());

    // Experimental encodings

    if use_experimental_patches() {
        allowed.insert(Patched.id());
    }

    #[cfg(feature = "zstd")]
    allowed.insert(Zstd.id());
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    allowed.insert(ZstdBuffers.id());

    allowed
});

/// The array encodings a writer may emit when writing with `session`: [`ALLOWED_ENCODINGS`]
/// gated by the session's enabled editions.
///
/// An encoding survives the gate only when an enabled edition includes it. A session that
/// enables no editions therefore writes nothing; the default Vortex session enables the current
/// `core` edition and loses the encodings declared solely by `unstable` editions.
pub fn writable_encodings(session: &VortexSession) -> HashSet<ArrayId> {
    let enabled: HashSet<ArrayId> = session.enabled_encoding_ids().into_iter().collect();
    ALLOWED_ENCODINGS
        .iter()
        .copied()
        .filter(|id| enabled.contains(id))
        .collect()
}

/// Normalizes every chunk down to the encodings the *writing session* may emit, resolved from
/// the session at write time.
///
/// The set is deliberately not resolved when the strategy is built: enabling an edition mutates
/// the session, so a set captured at construction time can disagree with the one the writer's
/// [`ArrayContext`] enforces, and the write then fails on an encoding
/// the compressor was still allowed to produce. [`LayoutStrategy::write_stream`] receives the
/// session, which is the point the gate is actually enforced, so it is resolved there.
///
/// Gated encodings are executed back to an allowed representation rather than rejected. The
/// compressor is free to pick any scheme, so a scheme whose output an edition does not cover
/// costs compression ratio on a narrowed session instead of failing the file.
struct EditionGatedStrategy {
    child: Arc<dyn LayoutStrategy>,
    /// The static allow-list, intersected with the session's writable set at write time.
    allow_encodings: HashSet<ArrayId>,
}

impl EditionGatedStrategy {
    fn new<S: LayoutStrategy>(child: S, allow_encodings: HashSet<ArrayId>) -> Self {
        Self {
            child: Arc::new(child),
            allow_encodings,
        }
    }
}

#[async_trait]
impl LayoutStrategy for EditionGatedStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let enabled: HashSet<ArrayId> = session.enabled_encoding_ids().into_iter().collect();
        let writable: HashSet<ArrayId> = self
            .allow_encodings
            .iter()
            .copied()
            .filter(|id| enabled.contains(id))
            .collect();

        // Nothing to gate: skip the per-chunk traversal entirely. This is the case for every
        // session whose enabled editions cover the writer's allow-list, which is all of them
        // by default.
        if writable.len() == self.allow_encodings.len() {
            return self
                .child
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let dtype = stream.dtype().clone();
        let writable = Arc::new(writable);
        let exec_session = session.clone();
        let stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut exec_ctx = exec_session.create_execution_ctx();
            let chunk = chunk.normalize(&mut NormalizeOptions {
                allowed: &writable,
                operation: Operation::Execute(&mut exec_ctx),
            })?;
            Ok((sequence_id, chunk))
        });

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                session,
            )
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        self.child.buffered_bytes()
    }
}

/// How the compressor was configured on [`WriteStrategyBuilder`].
enum CompressorConfig {
    /// A [`BtrBlocksCompressorBuilder`] that [`WriteStrategyBuilder::build`] will finalize.
    /// `IntDictScheme` is automatically excluded from the data compressor to prevent recursive
    /// dictionary encoding.
    BtrBlocks(BtrBlocksCompressorBuilder),
    /// An opaque compressor used as-is for both data and stats compression.
    Opaque(Arc<dyn CompressorPlugin>),
}

/// Build a new [writer strategy](LayoutStrategy) to compress and reorganize chunks of a Vortex
/// file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
///
/// The default pipeline first splits struct columns, repartitions rows into fixed-size row blocks,
/// computes zoned statistics, applies dictionary encoding where useful, coalesces chunks toward
/// segment-sized blocks, compresses arrays, buffers nearby chunks, and finally writes flat leaf
/// layouts.
pub struct WriteStrategyBuilder {
    compressor: CompressorConfig,
    row_block_size: usize,
    data_block_target_bytes: Option<u64>,
    field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    allow_encodings: Option<HashSet<ArrayId>>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
    probe_compressor: Option<Arc<dyn CompressorPlugin>>,
    /// Whether to write list fields using [`ListLayoutStrategy`].
    ///
    /// [`ListLayoutStrategy`]: vortex_layout::layouts::list::writer::ListLayoutStrategy
    use_list_layout: bool,
}

impl Default for WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured,
    /// and then finally built yielding the [`LayoutStrategy`].
    fn default() -> Self {
        Self {
            compressor: CompressorConfig::BtrBlocks(BtrBlocksCompressorBuilder::default()),
            row_block_size: 8192,
            data_block_target_bytes: Some(ONE_MEG),
            field_writers: HashMap::new(),
            allow_encodings: Some(ALLOWED_ENCODINGS.clone()),
            flat_strategy: None,
            probe_compressor: None,
            use_list_layout: use_experimental_list_layout(),
        }
    }
}

impl WriteStrategyBuilder {
    /// Override the row block size used for row repartitioning and zoned statistics.
    ///
    /// Larger blocks reduce footer/statistics overhead. Smaller blocks can improve pruning and
    /// random-access locality.
    pub fn with_row_block_size(mut self, row_block_size: usize) -> Self {
        self.row_block_size = row_block_size;
        self
    }

    /// Override the target uncompressed byte size used to coalesce data blocks.
    ///
    /// Passing `None` disables byte-size coalescing, so blocks retain the row granularity set by
    /// [`Self::with_row_block_size`].
    pub fn with_data_block_target_bytes(mut self, target_bytes: Option<u64>) -> Self {
        self.data_block_target_bytes = target_bytes;
        self
    }

    /// Enable writing list fields with [`ListLayoutStrategy`].
    ///
    /// **Note**: this is an unstable and experimental layout that is expected to change.
    /// Using it may lead to unreadable files in the future.
    ///
    /// [`ListLayoutStrategy`]: vortex_layout::layouts::list::writer::ListLayoutStrategy
    pub fn with_list_layout(mut self) -> Self {
        self.use_list_layout = true;
        self
    }

    /// Override the write layout for a specific field somewhere in the nested schema tree.
    ///
    /// The field path is matched after the root struct is split into columns. This is useful when a
    /// column needs a custom compression/layout policy while the rest of the file uses defaults.
    pub fn with_field_writer(
        mut self,
        field: impl Into<FieldPath>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(field.into(), writer);
        self
    }

    /// Override the allowed array encodings for normalization.
    ///
    /// The configured flat leaf strategy is wrapped in a [`LayoutStrategyEncodingValidator`]
    /// that recursively checks every chunk before passing it to the leaf writer.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allow_encodings = Some(allow_encodings);
        self
    }

    /// Override the flat layout strategy used for leaf chunks.
    ///
    /// By default, this uses [`FlatLayoutStrategy`]. This can be used to substitute a custom
    /// layout strategy, e.g. one that inlines constant array buffers for GPU reads.
    pub fn with_flat_strategy(mut self, flat: Arc<dyn LayoutStrategy>) -> Self {
        self.flat_strategy = Some(flat);
        self
    }

    /// Override the default [`BtrBlocksCompressorBuilder`] used for compression.
    ///
    /// The builder is finalized during [`build`](Self::build), producing two compressors: one for
    /// data (with `IntDictScheme` excluded) and one for stats.
    pub fn with_btrblocks_builder(mut self, builder: BtrBlocksCompressorBuilder) -> Self {
        self.compressor = CompressorConfig::BtrBlocks(builder);
        self
    }

    /// Set the compressor to an opaque [`CompressorPlugin`].
    ///
    /// The compressor is used as-is for both data and stats compression. Use this when the
    /// compressor is already fully configured and should not be modified by the builder.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = CompressorConfig::Opaque(Arc::new(compressor));
        self
    }

    /// Override the compressor used to probe whether a column is dict-eligible.
    pub fn with_probe_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.probe_compressor = Some(Arc::new(compressor));
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let flat: Arc<dyn LayoutStrategy> = if let Some(flat) = self.flat_strategy {
            flat
        } else {
            Arc::new(FlatLayoutStrategy::default())
        };
        let flat: Arc<dyn LayoutStrategy> = if let Some(allow_encodings) = &self.allow_encodings {
            // The session gate runs first, normalizing away any encoding the writing session's
            // editions do not cover; the validator then asserts the static allow-list holds.
            Arc::new(EditionGatedStrategy::new(
                LayoutStrategyEncodingValidator::new(flat, allow_encodings.clone()),
                allow_encodings.clone(),
            ))
        } else {
            flat
        };

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB

        // 5. compress each chunk.
        // Exclude IntDictScheme from the data compressor because DictStrategy (step 3) already
        // dictionary-encodes columns. Allowing IntDictScheme here would redundantly
        // dictionary-encode the integer codes produced by that earlier step.
        let data_compressor: Arc<dyn CompressorPlugin> = match &self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(
                builder
                    .clone()
                    .exclude_schemes([IntDictScheme.id()])
                    .build(),
            ),
            CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
        };
        let compressing = CompressingStrategy::new(buffered, data_compressor);

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                // Write stream partitions roughly become segments. Because Vortex never reads less
                // than one segment, the size of segments and, therefore, partitions, must be small
                // enough to both (1) allow fine-grained random access reads and (2) allow
                // sufficient read concurrency for the desired throughput. One megabyte is small
                // enough to achieve this for S3 (Durner et al., "Exploiting Cloud Object Storage for
                // High-Performance Analytics", VLDB Vol 16, Iss 11).
                block_size_minimum: self.data_block_target_bytes.unwrap_or(0),
                block_len_multiple: self.row_block_size,
                block_size_target: self.data_block_target_bytes,
                canonicalize: true,
            },
        );

        // 2.1. | 3.1. compress stats tables and dict values.
        let stats_compressor: Arc<dyn CompressorPlugin> = match self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(builder.build()),
            CompressorConfig::Opaque(compressor) => compressor,
        };
        let compress_then_flat = CompressingStrategy::new(flat, Arc::clone(&stats_compressor));

        // 3. apply dict encoding or fallback
        let probe_compressor = if let Some(probe_compressor) = self.probe_compressor {
            probe_compressor
        } else {
            Arc::clone(&stats_compressor)
        };
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
            probe_compressor,
        );

        let row_block_size = NonZeroUsize::new(self.row_block_size).vortex_expect("must be non 0");

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: row_block_size,
                ..Default::default()
            },
        );

        // 1. repartition each column to fixed row counts
        let repartition = RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: self.row_block_size,
                block_size_target: None,
                canonicalize: false,
            },
        );

        // 0. start with splitting columns
        let validity_strategy = CollectStrategy::new(compress_then_flat.clone());

        // Take any field overrides from the builder and apply them to the final strategy.
        let mut table_strategy =
            TableStrategy::new(Arc::new(validity_strategy), Arc::new(repartition))
                .with_field_writers(self.field_writers);

        if self.use_list_layout {
            // We need a closure here to enable recursive application of list layout.
            table_strategy = table_strategy.with_list_layout_factory(
                move |list_layout: ListLayoutStrategy| -> Arc<dyn LayoutStrategy> {
                    let zoned = ZonedStrategy::new(
                        list_layout,
                        compress_then_flat.clone(),
                        ZonedLayoutOptions {
                            block_size: row_block_size,
                            ..Default::default()
                        },
                    );
                    Arc::new(RepartitionStrategy::new(
                        zoned,
                        RepartitionWriterOptions {
                            block_size_minimum: 0,
                            block_len_multiple: row_block_size.get(),
                            block_size_target: None,
                            canonicalize: false,
                        },
                    ))
                },
            );
        }

        Arc::new(table_strategy)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use vortex_array::array_session;
    use vortex_edition::Edition;
    use vortex_edition::EditionDeclaration;
    use vortex_edition::EditionError;
    use vortex_edition::EditionId;

    use super::*;

    /// An enabled edition that adds nothing, so nothing it declares is writable beyond what
    /// other editions contribute.
    const ENABLED: EditionId = EditionId::new("strategytest", 2026, 1, 0);
    /// A registered but never enabled edition, whose members must be gated out.
    const DISABLED: EditionId = EditionId::new("strategydraft", 2026, 1, 0);

    static DECLARATIONS: &[EditionDeclaration] = &[
        EditionDeclaration {
            edition: Edition {
                id: ENABLED,
                min_vortex_version: Some("0.1.0"),
            },
            // The structural encodings are what a file is built from, so an edition that did
            // not cover them could not write at all, gated compression or otherwise.
            added: &[
                &"vortex.primitive",
                &"vortex.struct",
                &"vortex.chunked",
                &"vortex.constant",
            ],
        },
        EditionDeclaration {
            edition: Edition {
                id: DISABLED,
                min_vortex_version: None,
            },
            added: &[&"fastlanes.bitpacked", &"fastlanes.for"],
        },
    ];

    /// Register the test editions on `session` and enable the one covering `vortex.primitive`,
    /// leaving the edition that declares the FastLanes encodings registered but disabled.
    pub(crate) fn gate_session(session: &VortexSession) -> Result<(), EditionError> {
        for declaration in DECLARATIONS {
            session.register_edition(declaration)?;
        }
        session.enable_edition(ENABLED)
    }

    fn gated_session() -> Result<VortexSession, EditionError> {
        let session = array_session();
        gate_session(&session)?;
        Ok(session)
    }

    /// Only what an enabled edition includes survives. An encoding no edition mentions at all
    /// is not writable either: the gate admits, it does not merely exclude.
    #[test]
    fn writable_encodings_keeps_only_enabled_edition_members() -> Result<(), EditionError> {
        let writable = writable_encodings(&gated_session()?);

        assert!(writable.contains(&Primitive.id()), "enabled edition member");
        assert!(!writable.contains(&FSST.id()), "no edition declares FSST");
        assert!(!writable.contains(&BitPacked.id()), "declared, not enabled");
        assert!(!writable.contains(&FoR.id()), "declared, not enabled");
        Ok(())
    }

    /// The builder no longer resolves the session; the gate happens at write time. What the
    /// builder still owns is the static allow-list, which must stay untouched by any session.
    #[test]
    fn the_builder_allow_list_is_independent_of_any_session() {
        let builder = WriteStrategyBuilder::default();
        assert_eq!(builder.allow_encodings, Some(ALLOWED_ENCODINGS.clone()));
    }

    /// A custom allow-list is narrowed by the session, not replaced by it: the gate can only
    /// ever remove encodings the caller already permitted.
    #[test]
    fn a_custom_allow_list_is_narrowed_by_the_session() -> Result<(), EditionError> {
        let session = gated_session()?;
        let custom: HashSet<ArrayId> = HashSet::from_iter([Primitive.id(), FoR.id()]);
        let enabled: HashSet<ArrayId> = session.enabled_encoding_ids().into_iter().collect();
        let writable: HashSet<ArrayId> = custom
            .iter()
            .copied()
            .filter(|id| enabled.contains(id))
            .collect();

        assert_eq!(writable, HashSet::from_iter([Primitive.id()]));
        Ok(())
    }
}
