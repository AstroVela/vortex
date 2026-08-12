// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The proptest strategies that generate cases.
//!
//! Coordinates are drawn from small lattices instead of the full float range, so generated
//! geometries often share vertices, touch boundaries, or have collinear edges. These are the
//! configurations where geometry bugs live, and uniform random floats would almost never
//! produce them.
//!
//! All geometries are well-formed: lines have at least two vertices and multi geometries at
//! least one member, because degenerate shapes panic inside the `geo` oracle. Null rows are
//! not generated either; null propagation is the executor's job, covered by each kernel's
//! unit tests.

use std::ops::Range;

use proptest::prelude::*;

use super::fixture::BinaryInput;
use super::fixture::ConstSide;
use super::fixture::FAMILIES;
use super::fixture::Family;
use super::fixture::Fixture;
use super::fixture::GeometryColumn;
use super::fixture::Vertices;

/// A finite ordinate, biased onto small integer and half-integer lattices so that generated
/// geometries frequently share vertices, touch boundaries, and have collinear edges.
fn ordinate() -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => (-4i32..=4).prop_map(f64::from),
        2 => (-8i32..=8).prop_map(|v| f64::from(v) / 2.0),
        1 => -100.0..100.0f64,
    ]
}

/// An open path of `min` to eight vertices.
fn path(min: usize) -> impl Strategy<Value = Vertices> {
    prop::collection::vec((ordinate(), ordinate()), min..=8)
}

/// A ring of three to six vertices, mostly closed but occasionally left unclosed: the decode
/// path closes rings implicitly, and both forms must behave alike.
fn ring() -> impl Strategy<Value = Vertices> {
    (
        prop::collection::vec((ordinate(), ordinate()), 3..=6),
        prop::bool::weighted(0.875),
    )
        .prop_map(|(mut ring, close)| {
            if close {
                ring.push(ring[0]);
            }
            ring
        })
}

/// One to three rings: an exterior plus up to two holes.
fn rings() -> impl Strategy<Value = Vec<Vertices>> {
    prop::collection::vec(ring(), 1..=3)
}

/// One well-formed fixture of `family`.
fn fixture(family: Family) -> BoxedStrategy<Fixture> {
    match family {
        Family::Point => (ordinate(), ordinate())
            .prop_map(|(x, y)| Fixture::Point(x, y))
            .boxed(),
        Family::LineString => path(2).prop_map(Fixture::LineString).boxed(),
        Family::MultiPoint => path(1).prop_map(Fixture::MultiPoint).boxed(),
        Family::Polygon => rings().prop_map(Fixture::Polygon).boxed(),
        Family::MultiLineString => prop::collection::vec(path(2), 1..=3)
            .prop_map(Fixture::MultiLineString)
            .boxed(),
        Family::MultiPolygon => prop::collection::vec(rings(), 1..=3)
            .prop_map(Fixture::MultiPolygon)
            .boxed(),
        Family::Rect => (ordinate(), ordinate(), ordinate(), ordinate())
            .prop_map(|(x1, y1, x2, y2)| Fixture::Rect(x1, y1, x2, y2))
            .boxed(),
    }
}

/// A column of `len` rows of `family`.
fn column(family: Family, len: usize) -> impl Strategy<Value = GeometryColumn> {
    prop::collection::vec(fixture(family), len..=len)
        .prop_map(move |rows| GeometryColumn { family, rows })
}

/// A unary invocation over one fixed `family`: a column of up to 16 rows, plus a slice of it.
pub(super) fn unary_input(family: Family) -> impl Strategy<Value = (GeometryColumn, Range<usize>)> {
    (0..=16usize).prop_flat_map(move |len| {
        (
            column(family, len),
            (0..=len).prop_flat_map(move |start| (start..=len).prop_map(move |end| start..end)),
        )
    })
}

/// A binary invocation: two equal-length columns, plus which side (if any) is constant.
///
/// The operands share a family half the time ("coupled"), and each coupled row of `b` has a
/// one-in-four chance of being an exact clone of `a`'s row. This produces comparisons of a
/// geometry with itself, which independent generation would almost never create.
pub(super) fn binary_input() -> impl Strategy<Value = BinaryInput> {
    (
        prop::sample::select(FAMILIES.to_vec()),
        prop::sample::select(FAMILIES.to_vec()),
        0..=12usize,
        any::<bool>(),
    )
        .prop_flat_map(|(fa, fb, len, coupled)| {
            let fb = if coupled { fa } else { fb };
            let clone_mask = prop::collection::vec(prop::bool::weighted(0.25), len..=len);
            let constant = if len == 0 {
                Just(ConstSide::Neither).boxed()
            } else {
                prop_oneof![
                    2 => Just(ConstSide::Neither),
                    1 => (0..len).prop_map(ConstSide::Left),
                    1 => (0..len).prop_map(ConstSide::Right),
                ]
                .boxed()
            };
            (column(fa, len), column(fb, len), clone_mask, constant).prop_map(
                move |(a, mut b, clone_mask, constant)| {
                    if coupled {
                        for (row, clone) in clone_mask.into_iter().enumerate() {
                            if clone {
                                b.rows[row] = a.rows[row].clone();
                            }
                        }
                    }
                    BinaryInput { a, b, constant }
                },
            )
        })
}
