// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definitions of Vortex *editions*: named, frozen sets of serializable objects that a
//! writer may put in a file, carrying a forever read-compatibility guarantee.
//!
//! Editions cover every kind of object whose identifier can appear in serialized Vortex
//! data: array encodings, layout encodings, aggregate functions (stored in zone maps and
//! file statistics), extension dtypes (stored in every serialized `DType`), and the scalar
//! functions named by serialized expressions. Each member is an [`EditionInclusion`]
//! identified by an [`ObjectKind`] plus an object id; ids are unique within a kind, and
//! the same id may name different objects of different kinds (`vortex.dict` is both an
//! array encoding and a layout encoding).
//!
//! Editions live on the session, like encodings do: [`EditionSession`] holds the registered
//! editions and [`EnabledEditions`] selects which of them a writer may emit. Declarations
//! are plain constants — an [`EditionId`] plus an [`Edition`] record, and one
//! [`EditionInclusion`] per object stating that it is a member of an edition *and every
//! later edition of the same family*. Any crate can register declarations into a session,
//! so inclusions can live next to the object they describe.
//!
//! An edition is a **draft** until its [`Edition::min_vortex_version`] is recorded —
//! recording it is the act of freezing. The per-edition member sets are computed from the
//! registered declarations by [`EditionSession::members_in`], and correctness is enforced
//! by unit tests: [`EditionSession::validate`] checks a whole registry, and
//! [`test_harness::validate_edition`] validates one edition's constraints — call it once in
//! the `#[cfg(test)]` module of each edition definition.
//!
//! The first-party edition declarations live in the public `vortex` crate, which registers
//! and enables them on the default session. See the published spec at
//! <https://docs.vortex.dev/specs/editions.html>, which also documents how the serialized
//! form of each member object is allowed to evolve between editions.

mod session;
pub mod test_harness;
#[cfg(test)]
mod tests;

use std::error::Error;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;

pub use session::EditionSession;
pub use session::EditionSessionExt;
pub use session::EnabledEditions;
use vortex_session::registry::Id;

/// The identifier of an edition, e.g. `core2026.07.0`.
///
/// The `family` names an independently versioned, additive group of objects (`core` is the
/// set the default writer emits). The date components record when the edition was frozen and
/// order editions chronologically *within* a family; there is no ordering across families.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EditionId {
    /// The edition family, e.g. `core`.
    pub family: &'static str,
    /// Year the edition was cut.
    pub year: u16,
    /// Month the edition was cut.
    pub month: u8,
    /// Distinguishes editions cut in the same month; normally `0`.
    pub version: u8,
}

impl EditionId {
    /// Create an edition identifier. Validated by [`EditionId::validate`], which
    /// [`test_harness::validate_edition`] exercises
    /// per edition in unit tests.
    pub const fn new(family: &'static str, year: u16, month: u8, version: u8) -> Self {
        Self {
            family,
            year,
            month,
            version,
        }
    }

    /// Returns true if `self` is the same edition as `other` or an earlier edition of the
    /// same family. Editions of different families are never ordered.
    pub fn is_at_or_before(&self, other: &EditionId) -> bool {
        self.family == other.family
            && (self.year, self.month, self.version) <= (other.year, other.month, other.version)
    }

    /// Validate the identifier's form: a non-empty lowercase family, a four-digit year,
    /// and a month in 01-12. Checked for every declared edition by
    /// [`EditionSession::validate`] and per edition by
    /// [`test_harness::validate_edition`].
    pub fn validate(&self) -> Result<(), EditionError> {
        if self.family.is_empty() || !self.family.chars().all(|c| c.is_ascii_lowercase()) {
            return Err(EditionError::new(format!(
                "edition {self} must have a non-empty lowercase family, e.g. `core`"
            )));
        }
        if !(1000..=9999).contains(&self.year) {
            return Err(EditionError::new(format!(
                "edition {self} must have a four-digit year"
            )));
        }
        if !(1..=12).contains(&self.month) {
            return Err(EditionError::new(format!(
                "edition {self} must have a month in 01-12"
            )));
        }
        Ok(())
    }
}

impl Display for EditionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}.{:02}.{}",
            self.family, self.year, self.month, self.version
        )
    }
}

/// The kind of serializable object an edition member is.
///
/// Serialized Vortex data references objects of several registries, each with its own id
/// namespace. An edition member is identified by its kind *and* its id: ids are unique
/// within a kind, while the same id may name different objects of different kinds
/// (`vortex.dict` is both an array encoding and a layout encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectKind {
    /// An array encoding, e.g. `vortex.alp`.
    Array,
    /// A layout encoding, e.g. `vortex.zoned`.
    Layout,
    /// An aggregate function, e.g. `vortex.sum`. Aggregate function ids and options are
    /// serialized into files by zone maps and file statistics.
    Aggregation,
    /// A scalar function named by a serialized expression, e.g. `vortex.tensor.l2_norm`.
    ///
    /// Expressions are usually transient (scan predicates cross process boundaries but not
    /// storage), so most scalar functions never join an edition. Functions whose serialized
    /// form can reach durable data carry the same guarantee as the other kinds.
    Expression,
    /// An extension dtype, e.g. `vortex.timestamp`. Extension dtype ids and metadata are
    /// serialized wherever a [`DType`](https://docs.vortex.dev/specs/dtype-format.html)
    /// is, including every file's schema.
    ExtensionDType,
}

impl ObjectKind {
    /// Every object kind, in declaration order.
    pub const ALL: [ObjectKind; 5] = [
        ObjectKind::Array,
        ObjectKind::Layout,
        ObjectKind::Aggregation,
        ObjectKind::Expression,
        ObjectKind::ExtensionDType,
    ];
}

impl Display for ObjectKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ObjectKind::Array => "array",
            ObjectKind::Layout => "layout",
            ObjectKind::Aggregation => "aggregation",
            ObjectKind::Expression => "expression",
            ObjectKind::ExtensionDType => "extension dtype",
        })
    }
}

/// An edition: a named set of serializable objects with a read-compatibility guarantee,
/// registered with [`EditionSession::declare_edition`]. The set itself is computed from the
/// registered [`EditionInclusion`]s by [`EditionSession::members_in`].
#[derive(Clone, Copy, Debug)]
pub struct Edition {
    /// The edition identifier. Also carries the freeze date: `core2026.07.0` freezes in
    /// 2026-07.
    pub id: EditionId,
    /// The minimum Vortex version whose reader supports every member of this edition, in
    /// the serialized form writers emit as of the freeze.
    ///
    /// Recording this is the act of freezing: an edition with `None` is a **draft** — being
    /// assembled, carrying no guarantee, free to change, never the default write target.
    /// Validated against the members' [`EditionInclusion::required_vortex_release`] values:
    /// no member may require a version newer than the edition declares.
    pub min_vortex_version: Option<&'static str>,
}

impl Edition {
    /// A draft is an edition whose `min_vortex_version` has not been recorded yet.
    pub fn is_draft(&self) -> bool {
        self.min_vortex_version.is_none()
    }
}

/// Declares that an object is a member of an edition — and of every later edition of the
/// same family. Registered with [`EditionSession::declare_inclusion`].
#[derive(Clone, Copy, Debug)]
pub struct EditionInclusion {
    /// The kind of object this inclusion covers.
    pub kind: ObjectKind,
    /// The interned object id, e.g. `vortex.alp`. Unique within `kind`; ids of different
    /// kinds never conflict, so `vortex.dict` may join as an array encoding and again as a
    /// layout encoding.
    pub object_id: Id,
    /// The first edition this object is a member of.
    pub since: EditionId,
    /// The earliest Vortex release able to read this object in the serialized form emitted
    /// by `since`-targeting writers, recorded from evidence (e.g. compat-fixture history).
    /// `None` until recorded.
    pub required_vortex_release: Option<&'static str>,
}

/// A source of an object id for edition declarations.
///
/// Implemented for raw id strings (`"vortex.alp"`) and interned [`Id`]s here; encoding and
/// function vtables can implement it where they are defined, so a declaration can name the
/// vtable (`&Primitive`) instead of spelling its id.
pub trait AsObjectId: Debug + Send + Sync {
    /// The interned object id.
    fn object_id(&self) -> Id;
}

impl AsObjectId for str {
    #[expect(
        clippy::disallowed_methods,
        reason = "interning a dynamic object id at declaration time"
    )]
    fn object_id(&self) -> Id {
        Id::new(self)
    }
}

impl AsObjectId for Id {
    fn object_id(&self) -> Id {
        *self
    }
}

// `str` is unsized and cannot be a trait object, so declaration blocks (slices of
// `&dyn AsObjectId`) name objects as `&"vortex.alp"` through this impl.
impl AsObjectId for &'static str {
    fn object_id(&self) -> Id {
        (**self).object_id()
    }
}

/// Declares an edition together with the objects that join the family at it, in one block.
/// Registered with [`EditionSession::declare`], which derives each object's membership
/// (`since` = the declared edition) from the block structure.
///
/// Members of earlier editions are inherited and never restated. Each `added_*` slice
/// covers one [`ObjectKind`]; objects are named by id string or by vtable.
#[derive(Clone, Copy, Debug)]
pub struct EditionDeclaration {
    /// The edition being declared.
    pub edition: Edition,
    /// The array encodings that join the family at this edition.
    pub added_arrays: &'static [&'static dyn AsObjectId],
    /// The layout encodings that join the family at this edition.
    pub added_layouts: &'static [&'static dyn AsObjectId],
    /// The aggregate functions that join the family at this edition.
    pub added_aggregations: &'static [&'static dyn AsObjectId],
    /// The scalar functions (named by serialized expressions) that join the family at this
    /// edition.
    pub added_expressions: &'static [&'static dyn AsObjectId],
    /// The extension dtypes that join the family at this edition.
    pub added_extension_dtypes: &'static [&'static dyn AsObjectId],
}

impl EditionInclusion {
    /// Declare that an object of `kind` is a member of `since` and every later edition of
    /// the same family. The object can be named by id string or by vtable.
    pub fn new<E: AsObjectId + ?Sized>(kind: ObjectKind, object: &E, since: EditionId) -> Self {
        Self {
            kind,
            object_id: object.object_id(),
            since,
            required_vortex_release: None,
        }
    }

    /// Validate the declaration's form: a lowercase `namespace.name` object id and, if
    /// recorded, a well-formed `major.minor.patch` release. Checked for every declared
    /// inclusion by [`EditionSession::validate`].
    pub fn validate(&self) -> Result<(), EditionError> {
        let id = self.object_id.as_str();
        let well_formed = !id.starts_with('.')
            && !id.ends_with('.')
            && id.contains('.')
            && id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c));
        if !well_formed {
            return Err(EditionError::new(format!(
                "invalid {} id {id:?}: expected lowercase `namespace.name`, e.g. `vortex.alp`",
                self.kind
            )));
        }
        if let Some(release) = self.required_vortex_release
            && parse_release(release).is_none()
        {
            return Err(EditionError::new(format!(
                "{} {id} declares malformed required_vortex_release {release:?}",
                self.kind
            )));
        }
        Ok(())
    }
}

/// Parse a `major.minor.patch` release string into a comparable key.
pub(crate) fn parse_release(release: &str) -> Option<Vec<u64>> {
    let parts: Vec<u64> = release
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect::<Option<_>>()?;
    (parts.len() == 3).then_some(parts)
}

/// Error raised when edition declarations are inconsistent.
#[derive(Debug)]
pub struct EditionError(String);

impl EditionError {
    /// Create an error with the given message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl Display for EditionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for EditionError {}
