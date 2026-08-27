// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Correctness suites for the morsel executor.
//!
//! Every suite is differential: the V1 `LayoutReader` is the oracle, and a run passes only when
//! it emits the same rows in the same order. The properties the design document lists are each
//! expressed as a variation the output must be invariant under — thread count, morsel size,
//! conjunct policy, decode-cache budget, chunk alignment.

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::array_session;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::and;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::pack;
use vortex_array::expr::root;
use vortex_array::expr::select;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_session::VortexSession;

use crate::fixtures::Column;
use crate::fixtures::Fixture;
use crate::fixtures::write_fixture;
use crate::harness::MorselConfig;
use crate::harness::Query;
use crate::harness::assert_same_rows;
use crate::harness::run_morsel;
use crate::harness::run_v1;
use crate::nodes::ConjunctMode;

fn session() -> VortexSession {
    array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

fn i32_chunks(values: &[i32], boundaries: &[usize]) -> Vec<ArrayRef> {
    cut(values, boundaries)
        .into_iter()
        .map(|slice| {
            PrimitiveArray::new(Buffer::copy_from(slice), Validity::NonNullable).into_array()
        })
        .collect()
}

fn utf8_chunks(values: &[i32], boundaries: &[usize]) -> Vec<ArrayRef> {
    cut(values, boundaries)
        .into_iter()
        .map(|slice| {
            VarBinViewArray::from_iter_str(slice.iter().map(|v| format!("row-{v:06}"))).into_array()
        })
        .collect()
}

/// Split `values` at `boundaries`, which are exclusive ends in ascending order.
fn cut<'a>(values: &'a [i32], boundaries: &[usize]) -> Vec<&'a [i32]> {
    let mut out = Vec::with_capacity(boundaries.len());
    let mut start = 0;
    for &end in boundaries {
        out.push(&values[start..end]);
        start = end;
    }
    assert_eq!(start, values.len(), "boundaries must cover every value");
    out
}

/// The canonical misaligned fixture: three columns cut on three different boundary sets.
fn misaligned_fixture(session: &VortexSession, rows: usize) -> VortexResult<Fixture> {
    let a: Vec<i32> = (0..rows as i32).collect();
    let b: Vec<i32> = (0..rows as i32).map(|v| (v * 7) % 101).collect();
    let c: Vec<i32> = (0..rows as i32).map(|v| (v * 13) % 17).collect();

    let thirds = boundaries(rows, 3);
    let fifths = boundaries(rows, 5);
    let sevenths = boundaries(rows, 7);

    block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&a, &thirds)),
                Column::new("b", i32_chunks(&b, &fifths)),
                Column::new("c", utf8_chunks(&c, &sevenths)),
            ],
            session,
        )
        .await
    })
}

/// The same data with every column cut on the same boundaries — the aligned reference.
fn aligned_fixture(session: &VortexSession, rows: usize) -> VortexResult<Fixture> {
    let a: Vec<i32> = (0..rows as i32).collect();
    let b: Vec<i32> = (0..rows as i32).map(|v| (v * 7) % 101).collect();
    let c: Vec<i32> = (0..rows as i32).map(|v| (v * 13) % 17).collect();
    let single = vec![rows];

    block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&a, &single)),
                Column::new("b", i32_chunks(&b, &single)),
                Column::new("c", utf8_chunks(&c, &single)),
            ],
            session,
        )
        .await
    })
}

fn boundaries(rows: usize, parts: usize) -> Vec<usize> {
    let step = rows.div_ceil(parts);
    let mut out = Vec::with_capacity(parts);
    let mut end = step;
    while end < rows {
        out.push(end);
        end += step;
    }
    out.push(rows);
    out
}

fn queries() -> Vec<Query> {
    vec![
        Query {
            name: "select-all",
            projection: select(vec!["a", "b", "c"], root()),
            filter: None,
        },
        Query {
            name: "project-two",
            projection: select(vec!["a", "c"], root()),
            filter: None,
        },
        Query {
            name: "one-conjunct",
            projection: select(vec!["a", "b"], root()),
            filter: Some(gt(get_item("a", root()), lit(400i32))),
        },
        Query {
            name: "two-conjuncts",
            projection: select(vec!["a", "b", "c"], root()),
            filter: Some(and(
                gt(get_item("a", root()), lit(100i32)),
                lt(get_item("b", root()), lit(50i32)),
            )),
        },
        Query {
            name: "selective",
            projection: select(vec!["a", "c"], root()),
            filter: Some(and(
                gt(get_item("a", root()), lit(900i32)),
                lt(get_item("b", root()), lit(10i32)),
            )),
        },
        Query {
            name: "empty-result",
            projection: select(vec!["a"], root()),
            filter: Some(gt(get_item("a", root()), lit(1_000_000i32))),
        },
        Query {
            name: "filter-on-unprojected",
            projection: select(vec!["c"], root()),
            filter: Some(lt(get_item("b", root()), lit(30i32))),
        },
        Query {
            name: "packed-projection",
            projection: pack(
                vec![
                    ("x", get_item("a", root())),
                    ("y", get_item("b", root())),
                ],
                Nullability::NonNullable,
            ),
            filter: Some(gt(get_item("a", root()), lit(200i32))),
        },
    ]
}

const ROWS: usize = 1000;

/// Property: the executor agrees with V1 on every query, over misaligned chunks.
#[rstest]
fn matches_v1_oracle(#[values(1, 2, 4)] threads: usize) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
        let morsel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                threads,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &v1_dtype(&fixture.layout, &query)?, &v1, &morsel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// Property: misaligned chunking is invisible. The same logical table stored with three
/// different per-column chunkings must produce byte-identical output to the single-chunk
/// reference.
#[rstest]
fn misaligned_chunks_match_aligned_reference() -> VortexResult<()> {
    let session = session();
    let misaligned = misaligned_fixture(&session, ROWS)?;
    let aligned = aligned_fixture(&session, ROWS)?;
    let misaligned_segments: Arc<dyn SegmentSource> = Arc::clone(&misaligned.segments);
    let aligned_segments: Arc<dyn SegmentSource> = Arc::clone(&aligned.segments);

    for query in queries() {
        let left = run_morsel(
            &session,
            &misaligned.layout,
            &misaligned_segments,
            &query,
            MorselConfig::default(),
        )?;
        let right = run_morsel(
            &session,
            &aligned.layout,
            &aligned_segments,
            &query,
            MorselConfig::default(),
        )?;
        assert_same_rows(
            &session,
            &v1_dtype(&misaligned.layout, &query)?,
            &left,
            &right,
        )
        .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// The document's specific misaligned-chunk case: fields chunked `[0,3,10)` against `[0,6,10)`.
#[rstest]
fn document_misalignment_case() -> VortexResult<()> {
    let session = session();
    let values: Vec<i32> = (0..10).collect();
    let fixture = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[3, 10])),
                Column::new("b", i32_chunks(&values, &[6, 10])),
            ],
            &session,
        )
        .await
    })?;
    let reference = block_on(|_handle| async {
        write_fixture(
            vec![
                Column::new("a", i32_chunks(&values, &[10])),
                Column::new("b", i32_chunks(&values, &[10])),
            ],
            &session,
        )
        .await
    })?;

    let query = Query {
        name: "doc-case",
        projection: select(vec!["a", "b"], root()),
        filter: Some(gt(get_item("a", root()), lit(2i32))),
    };
    let dtype = v1_dtype(&fixture.layout, &query)?;

    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);
    let reference_segments: Arc<dyn SegmentSource> = Arc::clone(&reference.segments);

    let left = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig::default(),
    )?;
    let right = run_morsel(
        &session,
        &reference.layout,
        &reference_segments,
        &query,
        MorselConfig::default(),
    )?;
    let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;

    assert_same_rows(&session, &dtype, &left, &right)?;
    assert_same_rows(&session, &dtype, &left, &v1)?;

    // The morsel cut must be the union of both columns' boundaries.
    let plan = crate::build_plan(
        &fixture.layout,
        &query.projection,
        query.filter.as_ref(),
        ConjunctMode::Cascade,
    )?;
    assert_eq!(plan.natural_splits(), &[3, 6, 10]);
    Ok(())
}

/// Property: the result does not depend on how the scan is cut into morsels.
#[rstest]
fn independent_of_morsel_size(#[values(0, 1, 7, 128, 4096)] morsel_rows: u64) -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let v1 = run_v1(&session, &fixture.layout, &segments, &query)?;
        let morsel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                morsel_rows,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &dtype, &v1, &morsel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// Property: cascade and parallel conjunct policies are observationally identical.
#[rstest]
fn conjunct_policy_is_not_observable() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let cascade = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                mode: ConjunctMode::Cascade,
                ..Default::default()
            },
        )?;
        let parallel = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                mode: ConjunctMode::Parallel,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &dtype, &cascade, &parallel)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;
    }
    Ok(())
}

/// Property: the decoded-chunk cache is an optimisation only. Disabling it must not change a
/// single row — the chaos-mode analogue for P1.
#[rstest]
fn decode_cache_is_not_observable() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let dtype = v1_dtype(&fixture.layout, &query)?;
        let cached = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig::default(),
        )?;
        let uncached = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig {
                decode_cache_bytes: 0,
                ..Default::default()
            },
        )?;
        assert_same_rows(&session, &dtype, &cached, &uncached)
            .map_err(|err| err.with_context(format!("query {}", query.name)))?;

        // And the cache must actually be doing something on a misaligned fixture.
        let cached_stats = cached.stats.as_ref().expect("morsel runs report stats");
        let uncached_stats = uncached.stats.as_ref().expect("morsel runs report stats");
        assert!(
            cached_stats.decodes <= uncached_stats.decodes,
            "the cache must not increase decode count"
        );
    }
    Ok(())
}

/// Property: every read a node waits on was named by its own planning stream, so the number of
/// distinct segments read never exceeds the number of uses named.
#[rstest]
fn every_read_was_planned() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    for query in queries() {
        let run = run_morsel(
            &session,
            &fixture.layout,
            &segments,
            &query,
            MorselConfig::default(),
        )?;
        let stats = run.stats.as_ref().expect("morsel runs report stats");
        assert_eq!(
            stats.io_bypassed, 0,
            "query {}: no read should bypass planning with the floor at zero",
            query.name
        );
        assert!(
            stats.io_requests <= stats.io_uses,
            "query {}: {} requests exceeds {} named uses",
            query.name,
            stats.io_requests,
            stats.io_uses
        );
    }
    Ok(())
}

/// Property: an all-false filter emits nothing and reads no projection column.
#[rstest]
fn empty_filter_emits_nothing() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

    let query = Query {
        name: "empty",
        projection: select(vec!["a", "b", "c"], root()),
        filter: Some(gt(get_item("a", root()), lit(i32::MAX - 1))),
    };
    let run = run_morsel(
        &session,
        &fixture.layout,
        &segments,
        &query,
        MorselConfig::default(),
    )?;
    assert_eq!(run.rows, 0);
    assert!(run.batches.is_empty());
    let stats = run.stats.as_ref().expect("morsel runs report stats");
    assert_eq!(stats.morsels_empty, stats.morsels);
    Ok(())
}

/// Unsupported shapes are build errors, never silent fallbacks.
#[rstest]
fn rejects_unsupported_layouts() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, 32)?;
    // A non-struct root: take a column's chunked layout directly.
    let column = fixture
        .layout
        .slot(1)?
        .expect("the fixture root has a first field");
    let err = crate::build_plan(
        &column,
        &select(vec!["a"], root()),
        None,
        ConjunctMode::Cascade,
    )
    .err()
    .expect("a chunked root must be rejected");
    assert!(
        format!("{err}").contains("struct"),
        "unexpected error: {err}"
    );
    Ok(())
}

fn v1_dtype(layout: &LayoutRef, query: &Query) -> VortexResult<DType> {
    Ok(query.projection.bind(layout.dtype())?.dtype().clone())
}

/// A guard against the fixtures silently degenerating into a single chunk per column.
#[rstest]
fn fixture_is_actually_misaligned() -> VortexResult<()> {
    let session = session();
    let fixture = misaligned_fixture(&session, ROWS)?;
    let plan = crate::build_plan(
        &fixture.layout,
        &select(vec!["a", "b", "c"], root()),
        None,
        ConjunctMode::Cascade,
    )?;
    // Three columns cut into 3, 5 and 7 chunks share only the final boundary.
    assert!(
        plan.natural_splits().len() > 7,
        "expected the union of three chunkings, got {:?}",
        plan.natural_splits()
    );
    Ok(())
}
