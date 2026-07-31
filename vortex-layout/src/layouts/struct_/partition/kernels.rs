// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushdown kernels describing how a scalar function decomposes over a struct layout.
//!
//! A struct layout stores `n + 1` children: one per struct field, plus a validity child that is
//! present only when the struct is nullable. Any expression evaluated against such a layout
//! ultimately reads some subset of those children — its *slots* — and combines them back together.
//!
//! Which slots a given expression reads, and how the slot values recombine, is a property of the
//! *pair* `(scalar function, layout)`: `is_null` over a struct layout reads only the validity
//! child, whereas `is_null` over, say, a list layout would read something else entirely. This
//! module holds the struct-layout half of that pairing as a small registry of
//! [`StructPartitionKernel`]s keyed by [`ScalarFnId`].
//!
//! The registry is a process-wide static rather than a session variable. That is deliberate for
//! now: the rules encoded here are properties of the built-in scalar functions themselves, and
//! nothing yet needs to override them per session. When a second layout grows pushdown rules the
//! key should become `(LayoutId, ScalarFnId)` and the registry should move onto the session.
//!
//! # Fallback
//!
//! A scalar function without a rule is *not* an error. [`WholeScope`] is the safe default: the
//! expression is assumed to read every field and the validity, the whole struct is reconstructed
//! from its children, and the function is evaluated on top of it. Registering a rule only ever
//! narrows what has to be read.

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::col;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::mask;
use vortex_array::expr::not;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::transform::replace;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::byte_length::ByteLength;
use vortex_array::scalar_fn::fns::case_when::CaseWhen;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::fill_null::FillNull;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::scalar_fn::fns::list_sum::ListSum;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::merge::Merge;
use vortex_array::scalar_fn::fns::not::Not;
use vortex_array::scalar_fn::fns::pack::Pack;
use vortex_array::scalar_fn::fns::root::Root;
use vortex_array::scalar_fn::fns::select::Select;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::zip::Zip;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::HashMap;

/// A logical child of a struct layout.
///
/// A struct layout with `n` fields has `n + 1` slots: [`StructSlot::Validity`] holds the
/// non-nullable boolean validity of the struct itself and exists only when the struct is nullable,
/// and [`StructSlot::Field`] holds the (unmasked) values of the field at that index.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StructSlot {
    /// The validity child of a nullable struct, a non-nullable boolean where `true` means valid.
    Validity,
    /// The child holding the raw, unmasked values of the struct field at this index.
    Field(usize),
}

impl StructSlot {
    /// The synthetic name under which this slot's partition is referenced by the root expression.
    ///
    /// The names live in a scope that is built solely out of partitions, so they can never collide
    /// with the layout's own field names.
    pub fn partition_name(&self) -> FieldName {
        match self {
            StructSlot::Validity => FieldName::from("$validity"),
            StructSlot::Field(idx) => FieldName::from(format!("${idx}")),
        }
    }
}

impl Display for StructSlot {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            StructSlot::Validity => write!(f, "validity"),
            StructSlot::Field(idx) => write!(f, "field[{idx}]"),
        }
    }
}

/// The struct layout that an expression is being partitioned over.
pub struct StructScope<'a> {
    fields: &'a StructFields,
    nullability: Nullability,
    field_lookup: Option<&'a HashMap<FieldName, usize>>,
}

impl<'a> StructScope<'a> {
    /// Create a scope over the fields of a struct layout.
    ///
    /// `field_lookup`, when present, is used to resolve field names in place of a linear scan.
    pub fn new(
        fields: &'a StructFields,
        nullability: Nullability,
        field_lookup: Option<&'a HashMap<FieldName, usize>>,
    ) -> Self {
        Self {
            fields,
            nullability,
            field_lookup,
        }
    }

    /// The fields of the struct.
    pub fn fields(&self) -> &StructFields {
        self.fields
    }

    /// Whether the struct itself is nullable, and therefore has a validity slot.
    pub fn is_nullable(&self) -> bool {
        self.nullability.is_nullable()
    }

    /// Resolve a field name to its slot.
    pub fn find_field(&self, name: &FieldName) -> VortexResult<StructSlot> {
        self.field_lookup
            .and_then(|lookup| lookup.get(name).copied())
            .or_else(|| self.fields.find(name))
            .map(StructSlot::Field)
            .ok_or_else(|| vortex_err!("Field {name} not found in struct layout"))
    }

    /// The dtype of the child reader backing `slot`.
    pub fn slot_dtype(&self, slot: StructSlot) -> VortexResult<DType> {
        match slot {
            StructSlot::Validity => Ok(DType::Bool(Nullability::NonNullable)),
            StructSlot::Field(idx) => self
                .fields
                .field_by_index(idx)
                .ok_or_else(|| vortex_err!("Field index {idx} out of bounds")),
        }
    }

    /// Decompose a read of the entire struct scope into a read of every slot.
    ///
    /// The combining expression rebuilds the struct exactly as it is stored: a pack of the raw
    /// field values, masked by the layout validity when the struct is nullable.
    pub fn reconstruct(&self) -> SlotDecomposition {
        let mut parts: Vec<(StructSlot, Expression)> = (0..self.fields.nfields())
            .map(|idx| (StructSlot::Field(idx), root()))
            .collect();

        let packed = pack(
            self.fields
                .names()
                .iter()
                .cloned()
                .zip((0..parts.len()).map(part_ref)),
            Nullability::NonNullable,
        );

        if self.is_nullable() {
            let validity = part_ref(parts.len());
            parts.push((StructSlot::Validity, root()));
            SlotDecomposition::general(parts, mask(packed, validity))
        } else {
            SlotDecomposition::general(parts, packed)
        }
    }

    /// Select a subset of the struct's fields, preserving the struct's own validity.
    fn project(
        &self,
        names: impl IntoIterator<Item = FieldName>,
    ) -> VortexResult<SlotDecomposition> {
        let mut parts = Vec::new();
        let mut packed = Vec::new();
        for name in names {
            let slot = self.find_field(&name)?;
            packed.push((name, part_ref(parts.len())));
            parts.push((slot, root()));
        }

        let packed = pack(packed, Nullability::NonNullable);
        if self.is_nullable() {
            let validity = part_ref(parts.len());
            parts.push((StructSlot::Validity, root()));
            Ok(SlotDecomposition::general(parts, mask(packed, validity)))
        } else {
            Ok(SlotDecomposition::general(parts, packed))
        }
    }
}

/// The placeholder used inside a [`SlotDecomposition`]'s combining expression to reference part
/// `idx`. Substituted for the real partition reference by [`SlotDecomposition::combine`].
fn part_ref(idx: usize) -> Expression {
    col(format!("$part{idx}"))
}

/// How a decomposition's parts recombine, which controls how far the partitioner can push
/// enclosing expressions down into a single child.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum CombineKind {
    /// The value is exactly the single part, so any enclosing expression can be rewritten into
    /// that part's slot by substitution.
    Single,
    /// The value is `mask(part 0, part 1)`, where part 1 reads the layout validity as-is.
    ///
    /// `mask` commutes with strict scalar functions — `f(mask(x, v)) == mask(f(x), v)` when `f`
    /// maps null inputs to null outputs — so the partitioner may hoist the mask above strict
    /// ancestors and keep pushing them into part 0's slot.
    ValidityMasked,
    /// Anything else. Enclosing expressions stay in the root expression.
    General,
}

/// How an expression that reads the struct scope decomposes into reads of the layout's slots.
///
/// Each part is an expression evaluated in the scope of a single child reader; the combining
/// expression rebuilds the original value from the part results.
#[derive(Clone, Debug)]
pub struct SlotDecomposition {
    parts: Vec<(StructSlot, Expression)>,
    combine: Expression,
    kind: CombineKind,
}

impl SlotDecomposition {
    /// A value that does not read the struct at all, such as `is_null` of a non-nullable struct.
    pub fn constant(value: Expression) -> Self {
        Self {
            parts: Vec::new(),
            combine: value,
            kind: CombineKind::General,
        }
    }

    /// The value is `expr` evaluated against `slot`'s child reader, with nothing to recombine.
    pub fn single(slot: StructSlot, expr: Expression) -> Self {
        Self {
            parts: vec![(slot, expr)],
            combine: part_ref(0),
            kind: CombineKind::Single,
        }
    }

    /// The value is `expr` evaluated against `slot`'s child reader, masked by the struct validity.
    pub fn validity_masked(slot: StructSlot, expr: Expression) -> Self {
        Self {
            parts: vec![(slot, expr), (StructSlot::Validity, root())],
            combine: mask(part_ref(0), part_ref(1)),
            kind: CombineKind::ValidityMasked,
        }
    }

    /// A general decomposition.
    ///
    /// `combine` must reference part `i` as [`part_ref`] and must not otherwise reference the
    /// struct scope.
    pub fn general(parts: Vec<(StructSlot, Expression)>, combine: Expression) -> Self {
        Self {
            parts,
            combine,
            kind: CombineKind::General,
        }
    }

    /// The per-slot expressions this value is built from.
    pub fn parts(&self) -> &[(StructSlot, Expression)] {
        &self.parts
    }

    /// The distinct slots read by this value.
    pub fn slots(&self) -> impl Iterator<Item = StructSlot> + '_ {
        self.parts.iter().map(|(slot, _)| *slot)
    }

    /// Rebuild the value from expressions yielding each part's result.
    pub fn combine(&self, part_exprs: &[Expression]) -> Expression {
        part_exprs
            .iter()
            .enumerate()
            .fold(self.combine.clone(), |acc, (idx, part)| {
                replace(acc, &part_ref(idx), part.clone())
            })
    }

    /// The expression to evaluate against `slot` when an enclosing expression is being lowered
    /// into that slot, or `None` when this value cannot be read from `slot` alone.
    ///
    /// `hoist_validity` allows a [`CombineKind::ValidityMasked`] value to drop its mask, on the
    /// understanding that the caller re-applies the struct validity above the lowered expression.
    pub(super) fn lowerable(&self, slot: StructSlot, hoist_validity: bool) -> Option<&Expression> {
        if self.parts.is_empty() {
            // Reads nothing, so the value is valid in any child's scope.
            return Some(&self.combine);
        }
        match self.kind {
            CombineKind::Single if self.parts[0].0 == slot => Some(&self.parts[0].1),
            CombineKind::ValidityMasked if hoist_validity && self.parts[0].0 == slot => {
                Some(&self.parts[0].1)
            }
            _ => None,
        }
    }
}

/// Describes how a scalar function reading the struct scope decomposes over a struct layout's
/// children.
///
/// Kernels are only consulted for expressions that read the scope *directly*: either [`Root`]
/// itself, or a function whose first child is [`Root`]. Deeper reads are reached by recursion,
/// so an implementation must never inspect the scope through any child other than the first.
pub trait StructPartitionKernel: Debug + Send + Sync {
    /// Decompose `expr` into per-slot reads of `scope`.
    ///
    /// Returns `Ok(None)` when this function has no pushdown rule over a struct layout, in which
    /// case the partitioner falls back to reconstructing the whole struct.
    fn decompose(
        &self,
        expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>>;
}

/// The safe default: the function is opaque, so the whole struct is reconstructed beneath it.
#[derive(Debug)]
pub struct WholeScope;

impl StructPartitionKernel for WholeScope {
    fn decompose(
        &self,
        _expr: &Expression,
        _scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        Ok(None)
    }
}

/// `root()` — reads every field and the validity, and rebuilds the struct.
#[derive(Debug)]
struct RootKernel;

impl StructPartitionKernel for RootKernel {
    fn decompose(
        &self,
        _expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        Ok(Some(scope.reconstruct()))
    }
}

/// `root().field` — reads one field, intersected with the struct validity.
#[derive(Debug)]
struct GetItemKernel;

impl StructPartitionKernel for GetItemKernel {
    fn decompose(
        &self,
        expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        let slot = scope.find_field(expr.as_::<GetItem>())?;
        Ok(Some(if scope.is_nullable() {
            SlotDecomposition::validity_masked(slot, root())
        } else {
            SlotDecomposition::single(slot, root())
        }))
    }
}

/// `root(){a, b}` — reads the selected fields, keeping the struct's own validity.
///
/// Unlike `get_item`, a selection does not push the struct validity into the field values, so the
/// mask applies to the packed struct rather than to each field.
#[derive(Debug)]
struct SelectKernel;

impl StructPartitionKernel for SelectKernel {
    fn decompose(
        &self,
        expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        let included = expr
            .as_::<Select>()
            .normalize_to_included_fields(scope.fields().names())?;
        scope.project(included.iter().cloned()).map(Some)
    }
}

/// `is_null(root())` — reads only the validity.
#[derive(Debug)]
struct IsNullKernel;

impl StructPartitionKernel for IsNullKernel {
    fn decompose(
        &self,
        _expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        Ok(Some(if scope.is_nullable() {
            SlotDecomposition::single(StructSlot::Validity, not(root()))
        } else {
            SlotDecomposition::constant(lit(false))
        }))
    }
}

/// `is_not_null(root())` — reads only the validity.
#[derive(Debug)]
struct IsNotNullKernel;

impl StructPartitionKernel for IsNotNullKernel {
    fn decompose(
        &self,
        _expr: &Expression,
        scope: &StructScope<'_>,
    ) -> VortexResult<Option<SlotDecomposition>> {
        Ok(Some(if scope.is_nullable() {
            SlotDecomposition::single(StructSlot::Validity, root())
        } else {
            SlotDecomposition::constant(lit(true))
        }))
    }
}

/// The struct-layout pushdown rule for every built-in scalar function that can appear directly
/// above a struct scope.
///
/// Functions mapped to [`WholeScope`] are listed explicitly so that the set of functions that have
/// been considered is visible; they behave exactly as an unregistered function does.
static KERNELS: LazyLock<HashMap<ScalarFnId, Arc<dyn StructPartitionKernel>>> =
    LazyLock::new(|| {
        let whole_scope: Arc<dyn StructPartitionKernel> = Arc::new(WholeScope);
        let mut kernels: HashMap<ScalarFnId, Arc<dyn StructPartitionKernel>> = HashMap::new();

        // Functions that can be answered from a subset of the layout's children.
        kernels.insert(ScalarFnVTable::id(&Root), Arc::new(RootKernel));
        kernels.insert(ScalarFnVTable::id(&GetItem), Arc::new(GetItemKernel));
        kernels.insert(ScalarFnVTable::id(&Select), Arc::new(SelectKernel));
        kernels.insert(ScalarFnVTable::id(&IsNull), Arc::new(IsNullKernel));
        kernels.insert(ScalarFnVTable::id(&IsNotNull), Arc::new(IsNotNullKernel));

        // Functions that consume the struct opaquely, or that never see a struct at all.
        for id in [
            ScalarFnVTable::id(&Pack),
            ScalarFnVTable::id(&Merge),
            ScalarFnVTable::id(&Mask),
            ScalarFnVTable::id(&Cast),
            ScalarFnVTable::id(&Binary),
            ScalarFnVTable::id(&Not),
            ScalarFnVTable::id(&Between),
            ScalarFnVTable::id(&Like),
            ScalarFnVTable::id(&FillNull),
            ScalarFnVTable::id(&CaseWhen),
            ScalarFnVTable::id(&Zip),
            ScalarFnVTable::id(&ListContains),
            ScalarFnVTable::id(&ListLength),
            ScalarFnVTable::id(&ListSum),
            ScalarFnVTable::id(&ByteLength),
            ScalarFnVTable::id(&Literal),
            ScalarFnVTable::id(&VariantGet),
        ] {
            kernels.insert(id, Arc::clone(&whole_scope));
        }

        kernels
    });

/// The struct-layout partition kernel for a scalar function.
///
/// Falls back to [`WholeScope`] for functions without a registered rule.
pub fn struct_partition_kernel(id: ScalarFnId) -> &'static dyn StructPartitionKernel {
    static DEFAULT: WholeScope = WholeScope;
    KERNELS
        .get(&id)
        .map(|kernel| kernel.as_ref())
        .unwrap_or(&DEFAULT)
}

/// Reference the sub-expression at `idx` within a slot's partition.
pub(super) fn sub_expr_ref(slot: StructSlot, idx: usize) -> Expression {
    get_item(sub_expr_name(idx), col(slot.partition_name()))
}

/// The synthetic name of the `idx`th sub-expression packed into a slot's partition.
pub(super) fn sub_expr_name(idx: usize) -> FieldName {
    FieldName::from(format!("$sub{idx}"))
}
