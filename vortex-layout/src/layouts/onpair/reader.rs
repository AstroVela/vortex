// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::OnceLock;

use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_onpair::OnPair as OnPairEncoding;
use vortex_onpair::OnPairData;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::onpair::DICT_OFFSETS_PTYPE;
use crate::layouts::onpair::OnPairLayout;
use crate::layouts::onpair::expr::OnPairChildrenNeeded;
use crate::layouts::onpair::expr::get_necessary_onpair_children;
use crate::layouts::onpair::expr::rewrite_lengths_expr;
use crate::layouts::onpair::expr::rewrite_validity_expr;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// The threshold of mask density below which we push the input mask into projection evaluation,
/// and above which we evaluate the expression over all rows and intersect afterward.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// The column's one dictionary, read and safety-validated once.
///
/// `data` carries the memoized [`onpair::CompactDictionary`], so cloning it into every reassembled
/// chunk shares both the dictionary bytes and the validation.
///
/// [`onpair::CompactDictionary`]: https://docs.rs/onpair
#[derive(Clone)]
struct SharedDict {
    data: OnPairData,
    /// The canonical `dict_offsets` child, kept so each reassembled array can hold it directly
    /// rather than re-decoding it.
    dict_offsets: ArrayRef,
}

type SharedDictFuture = Shared<BoxFuture<'static, SharedVortexResult<SharedDict>>>;

/// Reader for [`OnPairLayout`].
///
/// The dictionary is auxiliary to the column's rows: it is read once, on the first row range that
/// needs the strings themselves, and reused for every row range after. Each range then reassembles
/// an ordinary [`OnPairArray`] from that dictionary and the range's own codes, so every OnPair
/// kernel — compressed-domain compare, dictionary-preserving filter, `byte_length` — applies without
/// the layout knowing about any of them.
///
/// [`OnPairArray`]: vortex_onpair::OnPairArray
#[derive(Clone)]
pub struct OnPairReader {
    layout: OnPairLayout,
    name: Arc<str>,
    session: VortexSession,
    /// Shared across clones so the dictionary is fetched and validated at most once.
    dict: Arc<OnceLock<SharedDictFuture>>,
    dict_bytes: LayoutReaderRef,
    dict_offsets: LayoutReaderRef,
    codes: LayoutReaderRef,
    codes_offsets: LayoutReaderRef,
    uncompressed_lengths: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl OnPairReader {
    pub(super) fn try_new(
        layout: OnPairLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<Self> {
        let child_reader = |child: crate::LayoutRef, suffix: &str| {
            child.new_reader(
                format!("{name}.{suffix}").into(),
                Arc::clone(&segment_source),
                &session,
                ctx,
            )
        };

        let dict_bytes = child_reader(layout.dict_bytes()?, "dict_bytes")?;
        let dict_offsets = child_reader(layout.dict_offsets()?, "dict_offsets")?;
        let codes = child_reader(layout.codes()?, "codes")?;
        let codes_offsets = child_reader(layout.codes_offsets()?, "codes_offsets")?;
        let uncompressed_lengths =
            child_reader(layout.uncompressed_lengths()?, "uncompressed_lengths")?;
        let validity = layout
            .validity()?
            .map(|v| child_reader(v, "validity"))
            .transpose()?;

        Ok(Self {
            layout,
            name,
            session,
            dict: Default::default(),
            dict_bytes,
            dict_offsets,
            codes,
            codes_offsets,
            uncompressed_lengths,
            validity,
        })
    }

    /// Read the shared dictionary, or return the in-flight/completed read of it.
    ///
    /// Both dictionary children are auxiliary and read whole. Safety-validating them here — rather
    /// than lazily, per array — means the whole column pays for one validation.
    fn shared_dict(&self) -> SharedDictFuture {
        self.dict
            .get_or_init(|| {
                let bytes_len = usize::try_from(self.dict_bytes.row_count())
                    .vortex_expect("dictionary blob length fits in usize");
                let offsets_len = usize::try_from(self.dict_offsets.row_count())
                    .vortex_expect("dictionary offsets length fits in usize");
                let bytes_fut = self
                    .dict_bytes
                    .projection_evaluation(
                        &(0..self.dict_bytes.row_count()),
                        &root(),
                        MaskFuture::new_true(bytes_len),
                    )
                    .vortex_expect("must construct OnPair dict_bytes evaluation");
                let offsets_fut = self
                    .dict_offsets
                    .projection_evaluation(
                        &(0..self.dict_offsets.row_count()),
                        &root(),
                        MaskFuture::new_true(offsets_len),
                    )
                    .vortex_expect("must construct OnPair dict_offsets evaluation");
                let session = self.session.clone();

                async move {
                    let (bytes, offsets) = try_join!(bytes_fut, offsets_fut)?;
                    let mut ctx = session.create_execution_ctx();
                    // The blob is stored read-padded, which is the form
                    // `CompactDictionary::validate_safety` requires; taking the buffer handle keeps
                    // it zero-copy when the child read it back uncompressed.
                    let bytes = bytes
                        .execute::<PrimitiveArray>(&mut ctx)?
                        .buffer_handle()
                        .clone();
                    let offsets = offsets
                        .cast(DType::Primitive(
                            DICT_OFFSETS_PTYPE,
                            Nullability::NonNullable,
                        ))?
                        .execute::<PrimitiveArray>(&mut ctx)?
                        .into_buffer::<u32>();
                    let data = OnPairData::try_new_with_dictionary(bytes, offsets.clone())?;
                    Ok(SharedDict {
                        data,
                        dict_offsets: offsets.into_array(),
                    })
                }
                .map_err(Arc::new)
                .boxed()
                .shared()
            })
            .clone()
    }

    /// Projection for [`OnPairChildrenNeeded::Validity`] expressions. Reads only the validity child,
    /// synthesizing all-valid for a non-nullable column, and touches neither the dictionary nor the
    /// codes.
    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
        // Evaluate the rewritten expression against the validity bool array (true == valid row).
        let rewritten = rewrite_validity_expr(expr)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let out_len = if mask.all_true() {
                row_count
            } else {
                mask.true_count()
            };

            let validity_array = match validity_reader.as_ref() {
                Some(v) => Some(
                    v.projection_evaluation(&row_range, &root(), MaskFuture::ready(mask))?
                        .await?,
                ),
                None => None,
            };

            let validity = create_validity(validity_array, nullability).to_array(out_len);

            validity.apply(&rewritten)
        }
        .boxed())
    }

    /// Projection for [`OnPairChildrenNeeded::LengthsAndValidity`] expressions. Reads only
    /// `uncompressed_lengths` and validity.
    ///
    /// `uncompressed_lengths` already holds the decoded byte length of every row, so unlike a list's
    /// offsets it needs no differencing — just the widening and validity that make it match what
    /// `byte_length` would have returned.
    fn project_lengths_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let lengths_fut = self.fetch_raw_lengths(row_range)?;
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
        let rewritten = rewrite_lengths_expr(expr)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;

            let validity_mask = if mask.all_true() {
                MaskFuture::new_true(row_count)
            } else {
                MaskFuture::ready(mask.clone())
            };
            let validity_fut = fetch_validity(validity_reader.as_ref(), &row_range, validity_mask)?;

            let lengths = lengths_fut.await?;
            let lengths = if mask.all_true() {
                lengths
            } else {
                lengths.filter(mask)?
            };
            let validity = validity_fut.await?;
            let lengths = apply_lengths_validity(lengths, validity, nullability)?;

            lengths.apply(&rewritten)
        }
        .boxed())
    }

    /// Projection for [`OnPairChildrenNeeded::All`] expressions.
    ///
    /// An all-true mask over the full local range reads every child concurrently. Otherwise the read
    /// is bounded to the codes of the first and last selected row.
    fn project_all(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let is_full_range = row_range.start == 0 && row_range.end == self.layout.row_count();
        let reader = self.clone();
        let row_range = row_range.clone();
        let expr = expr.clone();
        Ok(async move {
            let mask = mask.await?;
            if is_full_range && mask.all_true() {
                reader.project_all_full(&expr)?.await
            } else {
                reader.project_all_bounded(&row_range, &expr, mask)?.await
            }
        }
        .boxed())
    }

    /// Fetch the dictionary and every per-row child concurrently, reading the whole code stream.
    fn project_all_full(&self, expr: &Expression) -> VortexResult<ArrayFuture> {
        let row_count = self.layout.row_count();
        let row_range = 0..row_count;
        let dtype = self.layout.dtype().clone();
        let nullability = dtype.nullability();
        let expr = expr.clone();

        let dict_fut = self.shared_dict();
        let codes_offsets_fut = self.fetch_raw_codes_offsets(&row_range)?;
        let codes_fut = self.fetch_raw_codes(&(0..self.codes.row_count()))?;
        let lengths_fut = self.fetch_raw_lengths(&row_range)?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &row_range,
            MaskFuture::new_true(usize::try_from(row_count)?),
        )?;

        Ok(async move {
            let (dict, codes_offsets, codes, lengths, validity) = try_join!(
                dict_fut.map_err(VortexError::from),
                codes_offsets_fut,
                codes_fut,
                lengths_fut,
                validity_fut
            )?;
            // The full range starts at token zero, so the offsets already index `codes` directly.
            let SharedDict { data, dict_offsets } = dict;
            OnPairEncoding::try_new_with_data(
                dtype,
                data,
                dict_offsets,
                codes,
                codes_offsets,
                lengths,
                create_validity(validity, nullability),
            )?
            .into_array()
            .apply(&expr)
        }
        .boxed())
    }

    /// Bounded read for a sub-range or selective mask.
    ///
    /// Crops leading and trailing unselected rows, reads their code boundaries, and translates the
    /// first and last boundary into the token range to fetch. Any holes in the selection are filtered
    /// after reassembling the array — OnPair's filter preserves the dictionary, so that costs no
    /// decode.
    fn project_all_bounded(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: Mask,
    ) -> VortexResult<ArrayFuture> {
        // Crop to the smallest contiguous row range containing every selected row.
        let Some(selected_rows) = selected_row_range(&mask) else {
            let empty = Canonical::empty(self.layout.dtype()).into_array();
            let expr = expr.clone();
            return Ok(async move { empty.apply(&expr) }.boxed());
        };

        let selected_mask = mask.slice(selected_rows.clone());
        let selected_row_range = (row_range.start + u64::try_from(selected_rows.start)?)
            ..(row_range.start + u64::try_from(selected_rows.end)?);

        let dtype = self.layout.dtype().clone();
        let nullability = dtype.nullability();
        let expr = expr.clone();
        let reader = self.clone();
        let dict_fut = self.shared_dict();
        let codes_offsets_fut = self.fetch_raw_codes_offsets(&selected_row_range)?;

        Ok(async move {
            let codes_offsets = codes_offsets_fut.await?;

            let codes_range = codes_range_from_offsets(&codes_offsets, &reader.session)?;
            let codes_fut = reader.fetch_raw_codes(&codes_range)?;
            let lengths_fut = reader.fetch_raw_lengths(&selected_row_range)?;
            let validity_fut = fetch_validity(
                reader.validity.as_ref(),
                &selected_row_range,
                MaskFuture::new_true(selected_mask.len()),
            )?;
            let (dict, codes, lengths, validity) = try_join!(
                dict_fut.map_err(VortexError::from),
                codes_fut,
                lengths_fut,
                validity_fut
            )?;

            let codes_offsets = rebase_offsets(codes_offsets, codes_range.start)?;
            let SharedDict { data, dict_offsets } = dict;
            let array = OnPairEncoding::try_new_with_data(
                dtype,
                data,
                dict_offsets,
                codes,
                codes_offsets,
                lengths,
                create_validity(validity, nullability),
            )?
            .into_array();
            let array = if selected_mask.all_true() {
                array
            } else {
                array.filter(selected_mask)?
            };
            array.apply(&expr)
        }
        .boxed())
    }

    /// Fire the `codes_offsets` read for `row_range`. The child has an extra entry, so reading
    /// `row_range` maps to boundaries in `[row_range.start..row_range.end + 1)`.
    fn fetch_raw_codes_offsets(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;
        self.codes_offsets.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true(offsets_count),
        )
    }

    /// Fire the `codes` read for `codes_range`, in token space.
    fn fetch_raw_codes(&self, codes_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let count = usize::try_from(codes_range.end - codes_range.start)?;
        self.codes
            .projection_evaluation(codes_range, &root(), MaskFuture::new_true(count))
    }

    /// Fire the `uncompressed_lengths` read for `row_range`, unmasked so it stays aligned with the
    /// code boundaries.
    fn fetch_raw_lengths(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let count = usize::try_from(row_range.end - row_range.start)?;
        self.uncompressed_lengths.projection_evaluation(
            row_range,
            &root(),
            MaskFuture::new_true(count),
        )
    }
}

impl LayoutReader for OnPairReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        // Every per-chunk child is written from the same chunk boundaries, so the row-space
        // `uncompressed_lengths` child already reports exactly the boundaries the codes were written
        // at. No translation out of token space is needed, unlike a list layout's elements.
        self.uncompressed_lengths
            .register_splits(field_mask, split_range, splits)
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // Reading the dictionary just to approximate a filter is not worth it, and stats-based
        // pruning has already happened upstream of this reader.
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let len = mask.len();
        let reader = self.clone();
        let row_range = row_range.clone();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(len, async move {
            let mask = mask.await?;

            if mask.all_false() {
                return Ok(mask);
            }

            if mask.density() < EXPR_EVAL_THRESHOLD {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
                    .await?;
                let predicate_mask = predicate_array_to_mask(predicate, &session)?;
                Ok(mask.intersect_by_rank(&predicate_mask))
            } else {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::new_true(len))?
                    .await?;
                let predicate_mask = predicate_array_to_mask(predicate, &session)?;
                Ok(mask & &predicate_mask)
            }
        }))
    }

    /// Reads only the children needed to evaluate `expr`.
    ///
    /// Validity-only expressions avoid the dictionary, the codes, and the lengths; `byte_length`
    /// expressions read lengths and validity; everything else reassembles an OnPair array from all
    /// applicable children.
    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        match get_necessary_onpair_children(expr) {
            OnPairChildrenNeeded::Validity => self.project_validity(row_range, expr, mask),
            OnPairChildrenNeeded::LengthsAndValidity => {
                self.project_lengths_validity(row_range, expr, mask)
            }
            OnPairChildrenNeeded::All => self.project_all(row_range, expr, mask),
        }
    }
}

fn selected_row_range(mask: &Mask) -> Option<Range<usize>> {
    Some(mask.first()?..mask.last()? + 1)
}

fn create_validity(validity_array: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity_array {
        Some(arr) => Validity::Array(arr),
        None => match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        },
    }
}

/// Fetch the validity child for `row_range` under `mask`, yielding `None` for a non-nullable column
/// (which has no validity child).
fn fetch_validity(
    validity: Option<&LayoutReaderRef>,
    row_range: &Range<u64>,
    mask: MaskFuture,
) -> VortexResult<OptionalArrayFuture> {
    let fut = validity
        .map(|v| v.projection_evaluation(row_range, &root(), mask))
        .transpose()?;
    Ok(async move {
        match fut {
            Some(f) => f.await.map(Some),
            None => Ok(None),
        }
    }
    .boxed())
}

/// Read `codes_offsets[0]` and `codes_offsets[-1]` and return the token range they bound.
fn codes_range_from_offsets(
    offsets: &ArrayRef,
    session: &VortexSession,
) -> VortexResult<Range<u64>> {
    if offsets.is_empty() {
        return Ok(0..0);
    }
    let mut exec_ctx = session.create_execution_ctx();
    let start = offsets
        .execute_scalar(0, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("code boundary fits in u64");
    let end = offsets
        .execute_scalar(offsets.len() - 1, &mut exec_ctx)?
        .as_primitive()
        .as_::<u64>()
        .vortex_expect("code boundary fits in u64");
    Ok(start..end)
}

/// Subtract `first` from every boundary so they index into a `codes[first..]` slice that starts at
/// zero.
fn rebase_offsets(offsets: ArrayRef, first: u64) -> VortexResult<ArrayRef> {
    if first == 0 {
        return Ok(offsets);
    }
    let constant = ConstantArray::new(first, offsets.len())
        .into_array()
        .cast(offsets.dtype().clone())?;
    offsets.binary(constant, Operator::Sub)
}

/// Widen the stored lengths to what `byte_length` returns — `u64` carrying the column's nullability
/// — and null out the rows the column marks null.
fn apply_lengths_validity(
    lengths: ArrayRef,
    validity: Option<ArrayRef>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = lengths.len();
    let lengths = lengths.cast(DType::Primitive(PType::U64, nullability))?;

    if matches!(nullability, Nullability::Nullable) {
        lengths.mask(create_validity(validity, nullability).to_array(len))
    } else {
        Ok(lengths)
    }
}

fn predicate_array_to_mask(array: ArrayRef, session: &VortexSession) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    array.null_as_false().execute(&mut ctx)
}
