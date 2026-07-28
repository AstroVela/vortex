// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_alp::ALP;
use vortex_alp::ALPRD;
use vortex_array::ArrayId;
use vortex_array::VTable;
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
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_edition::EditionSessionExt;
use vortex_error::VortexExpect;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fsst::FSST;
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
/// An encoding survives the gate when an enabled edition includes it, or when no registered
/// edition declares it at all — see
/// [`retain_writable_encodings`](EditionSessionExt::retain_writable_encodings). A session with no
/// editions registered is therefore not gated and gets [`ALLOWED_ENCODINGS`] unchanged, while the
/// default Vortex session, which enables only the current `core` edition, loses the encodings
/// declared solely by `unstable` editions.
pub fn writable_encodings(session: &VortexSession) -> HashSet<ArrayId> {
    session
        .retain_writable_encodings(ALLOWED_ENCODINGS.iter().copied())
        .into_iter()
        .collect()
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
    /// that recursively checks every chunk before passing it to the leaf writer, and the
    /// compressor's schemes are filtered down to those whose output the allow-list covers.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allow_encodings = Some(allow_encodings);
        self
    }

    /// Gate the allowed array encodings by the editions `session` has enabled for writing.
    ///
    /// This removes every allowed encoding that a registered edition declares but the session has
    /// not enabled, so the file cannot contain an encoding outside the read-compatibility
    /// guarantee the session opted into. Encodings no registered edition declares are left alone;
    /// see [`writable_encodings`].
    ///
    /// The gate applies to compression as well as validation: [`build`](Self::build) drops every
    /// compression scheme that could produce a gated-out encoding.
    pub fn with_session_editions(mut self, session: &VortexSession) -> Self {
        if let Some(allow_encodings) = self.allow_encodings.take() {
            self.allow_encodings = Some(
                session
                    .retain_writable_encodings(allow_encodings)
                    .into_iter()
                    .collect(),
            );
        }
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
            Arc::new(LayoutStrategyEncodingValidator::new(
                flat,
                allow_encodings.clone(),
            ))
        } else {
            flat
        };

        // Filter the compressor to the schemes whose output the allow-list covers. Without this,
        // a scheme excluded from the allow-list — for instance one only an unstable edition
        // declares — would compress a chunk into an encoding the validator then rejects, failing
        // the write instead of choosing the next best scheme. This runs on the finished builder,
        // so schemes added after construction, including the compact ones from
        // [`BtrBlocksCompressorBuilder::with_compact`], are filtered too.
        let compressor = match self.compressor {
            CompressorConfig::BtrBlocks(builder) => {
                CompressorConfig::BtrBlocks(match &self.allow_encodings {
                    Some(allowed) => builder.retain_allowed_encodings(allowed),
                    None => builder,
                })
            }
            opaque @ CompressorConfig::Opaque(_) => opaque,
        };

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB

        // 5. compress each chunk.
        // Exclude IntDictScheme from the data compressor because DictStrategy (step 3) already
        // dictionary-encodes columns. Allowing IntDictScheme here would redundantly
        // dictionary-encode the integer codes produced by that earlier step.
        let data_compressor: Arc<dyn CompressorPlugin> = match &compressor {
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
        let stats_compressor: Arc<dyn CompressorPlugin> = match compressor {
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
            added: &[&"vortex.primitive"],
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

    #[test]
    fn writable_encodings_drops_disabled_editions() -> Result<(), EditionError> {
        let writable = writable_encodings(&gated_session()?);

        assert!(writable.contains(&Primitive.id()), "enabled edition member");
        assert!(writable.contains(&FSST.id()), "no edition declares FSST");
        assert!(!writable.contains(&BitPacked.id()));
        assert!(!writable.contains(&FoR.id()));
        Ok(())
    }

    #[test]
    fn session_editions_gate_the_allow_list() -> Result<(), EditionError> {
        let builder = WriteStrategyBuilder::default().with_session_editions(&gated_session()?);
        let allowed = builder
            .allow_encodings
            .as_ref()
            .vortex_expect("default builder sets an allow-list");

        assert!(allowed.contains(&Primitive.id()));
        assert!(!allowed.contains(&BitPacked.id()));
        assert!(!allowed.contains(&FoR.id()));
        Ok(())
    }

    #[test]
    fn an_ungated_session_keeps_every_allowed_encoding() {
        // `array_session` registers no editions, so there is nothing to gate against.
        let session = array_session();
        assert_eq!(writable_encodings(&session), *ALLOWED_ENCODINGS);

        let builder = WriteStrategyBuilder::default().with_session_editions(&session);
        assert_eq!(builder.allow_encodings, Some(ALLOWED_ENCODINGS.clone()));
    }

    #[test]
    fn a_custom_allow_list_is_narrowed_not_replaced() -> Result<(), EditionError> {
        let builder = WriteStrategyBuilder::default()
            .with_allow_encodings(HashSet::from_iter([Primitive.id(), FoR.id()]))
            .with_session_editions(&gated_session()?);

        assert_eq!(
            builder.allow_encodings,
            Some(HashSet::from_iter([Primitive.id()]))
        );
        Ok(())
    }
}
