// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::thread;

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::*;
use crate::SpFreshBuildOptions;
use crate::ffi::Native;
use crate::rabitq::OfficialRaBitQBuildOptions;

unsafe extern "C" {
    fn vortex_rabitq_estimates(
        handle: *mut c_void,
        query: *const f32,
        heads: *const u32,
        head_count: u32,
        capacity: u32,
        ids: *mut u32,
        scores: *mut f64,
        scalar_reference: u32,
        count: *mut u32,
        error: *mut c_char,
    ) -> i32;
}

fn fixture(
    root: &Path,
    dimension: u32,
) -> VortexResult<(SpFreshBundle, Vec<f32>, Vec<RowAddress>)> {
    fixture_rows(root, dimension, 97)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Synthetic components are reduced modulo 1009 before conversion"
)]
fn fixture_rows(
    root: &Path,
    dimension: u32,
    count: u32,
) -> VortexResult<(SpFreshBundle, Vec<f32>, Vec<RowAddress>)> {
    let bundle = SpFreshBundle {
        dimension,
        rows: count,
        posting_page_limit: 4096,
    };
    let vectors = (0..bundle.rows)
        .flat_map(|id| {
            (0..dimension).map(move |j| {
                // Integer-valued Float32s avoid cancellation ambiguity in the exact oracle.
                ((id * 37 + j * 17 + id * j * 3) % 1009) as f32 - 504.0
            })
        })
        .collect::<Vec<_>>();
    Native::build(
        root,
        &vectors,
        bundle,
        &SpFreshBuildOptions {
            format_version: 3,
            dimension,
            head_count: 32,
            posting_page_limit: 4096,
            replicas: 1,
            rabitq: None,
            rabitq_official: Some(OfficialRaBitQBuildOptions { seed: 42 }),
        },
    )?;
    build_official_sidecar(root, bundle, &vectors, 42)?;
    let rows = (0..bundle.rows)
        .map(|id| RowAddress {
            file_id: u64::from(17 + id % 2),
            row_offset: u64::from(id / 2),
        })
        .collect();
    Ok((bundle, vectors, rows))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Independent oracle rounds to the common Float32 distance API"
)]
fn oracle(
    vectors: &[f32],
    query: &[f32],
    rows: &[RowAddress],
    ids: impl Iterator<Item = u32>,
    k: usize,
) -> Vec<(u32, f32)> {
    let d = query.len();
    let mut hits = ids
        .map(|id| {
            let distance = vectors[id as usize * d..(id as usize + 1) * d]
                .iter()
                .zip(query)
                .map(|(x, q)| (f64::from(*x) - f64::from(*q)).powi(2))
                .sum::<f64>() as f32;
            (id, distance)
        })
        .collect::<Vec<_>>();
    hits.sort_unstable_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| rows[left.0 as usize].cmp(&rows[right.0 as usize]))
    });
    hits.truncate(k);
    hits
}

fn estimates(
    index: &OfficialIndex,
    query: &[f32],
    reference: u32,
) -> VortexResult<(Vec<u32>, Vec<f64>)> {
    let mut ids = vec![0; index.bundle.rows as usize];
    let mut scores = vec![0.0; ids.len()];
    let heads = (0..32).collect::<Vec<_>>();
    let mut count = 0;
    let mut error = [0; 1024];
    let handle = index.handle.lock();
    // SAFETY: exclusive handle; dimension-sized query, 32 valid heads and rows-sized
    // output arrays. The diagnostic call validates capacity and catches exceptions.
    let code = unsafe {
        vortex_rabitq_estimates(
            handle.0.as_ptr(),
            query.as_ptr(),
            heads.as_ptr(),
            32,
            index.bundle.rows,
            ids.as_mut_ptr(),
            scores.as_mut_ptr(),
            reference,
            &raw mut count,
            error.as_mut_ptr().cast(),
        )
    };
    result(code, &error)?;
    ids.truncate(count as usize);
    scores.truncate(count as usize);
    Ok((ids, scores))
}

#[test]
fn test_official_dimensions_scalar_reference_shortlist_and_exact_reranking() -> VortexResult<()> {
    for dimension in [
        1, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 191, 192, 384, 768, 769, 1025, 4095, 4096,
    ] {
        let directory = tempfile::tempdir()?;
        let root = directory.path().join("native");
        let (bundle, vectors, rows) = fixture(&root, dimension)?;
        let model = root.join("rabitq-official.bin");
        let index = OfficialIndex::open(&root, &model, bundle, &rows)?;
        for query in [
            vectors[..dimension as usize].to_vec(),
            vec![0.0; dimension as usize],
        ] {
            let (ids, scores) = estimates(&index, &query, 0)?;
            let (reference_ids, reference) = estimates(&index, &query, 1)?;
            let (stock_ids, stock) = estimates(&index, &query, 2)?;
            assert_eq!(ids, stock_ids);
            assert_eq!(
                scores
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>(),
                stock
                    .iter()
                    .map(|score| score.to_bits())
                    .collect::<Vec<_>>(),
                "optimized extension-bit kernel differs from official: dimension={dimension}"
            );
            assert_eq!(ids, reference_ids);
            assert_eq!(ids.len(), bundle.rows as usize);
            let query_norm = query.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
            for (position, (score, expected)) in scores.iter().zip(reference).enumerate() {
                let start = ids[position] as usize * dimension as usize;
                let vector_norm = vectors[start..start + dimension as usize]
                    .iter()
                    .map(|x| f64::from(*x).powi(2))
                    .sum::<f64>();
                // Near-zero estimates subtract large Float32 terms. Bound rounding
                // by their input scale; the optimized and stock SIMD paths above
                // must still agree bit for bit, independently of this scalar bound.
                let slack = 1e-4 + 8.0 * f64::from(f32::EPSILON) * (query_norm + vector_norm);
                assert!(
                    score.is_finite() && (score - expected).abs() <= slack,
                    "dimension={dimension}, id={}, score={score}, reference={expected}, query_norm={}",
                    ids[position],
                    query_norm
                );
            }
            let mut shortlist = ids.iter().copied().zip(scores).collect::<Vec<_>>();
            shortlist.sort_unstable_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| rows[left.0 as usize].cmp(&rows[right.0 as usize]))
            });
            let heads = (0..32).collect::<Vec<_>>();
            for budget in [17, bundle.rows] {
                let expected = oracle(
                    &vectors,
                    &query,
                    &rows,
                    shortlist.iter().take(budget as usize).map(|hit| hit.0),
                    10,
                );
                let (actual, stats) = index.search(&query, &heads, 10, budget, None)?;
                assert_eq!(actual, expected);
                assert_eq!(stats.scanned, bundle.rows as usize);
                assert_eq!(stats.reranked, budget as usize);
            }
            let mut mask = vec![0; (bundle.rows as usize).div_ceil(8)];
            for id in (0..bundle.rows).step_by(3) {
                mask[id as usize / 8] |= 1 << (id % 8);
            }
            assert_eq!(
                index
                    .search(&query, &heads, 10, bundle.rows, Some(&mask))?
                    .0,
                oracle(&vectors, &query, &rows, (0..bundle.rows).step_by(3), 10)
            );
            assert!(
                index
                    .search(&query, &heads, 10, bundle.rows, Some(&vec![0; mask.len()]))?
                    .0
                    .is_empty()
            );
        }
        drop(index);
        let reopened = OfficialIndex::open(&root, &model, bundle, &rows)?;
        assert_eq!(
            reopened
                .search(
                    &vectors[..dimension as usize],
                    &(0..32).collect::<Vec<_>>(),
                    10,
                    bundle.rows,
                    None
                )?
                .0,
            oracle(
                &vectors,
                &vectors[..dimension as usize],
                &rows,
                0..bundle.rows,
                10
            )
        );
    }
    Ok(())
}

#[test]
fn test_official_corruption_and_query_errors_preserve_loaded_index() -> VortexResult<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path().join("native");
    let (bundle, vectors, rows) = fixture(&root, 65)?;
    let model = root.join("rabitq-official.bin");
    let index = OfficialIndex::open(&root, &model, bundle, &rows)?;
    let original = fs::read(&model)?;
    let mut bad = original.clone();
    bad[0] ^= 1;
    let mut wrong_dimension = original.clone();
    wrong_dimension[8..12].copy_from_slice(&64u32.to_le_bytes());
    let mut wrong_bits = original.clone();
    wrong_bits[20..24].copy_from_slice(&1u32.to_le_bytes());
    let mut wrong_padding = original.clone();
    wrong_padding[28..32].copy_from_slice(&128u32.to_le_bytes());
    let malformed = root.join("malformed.bin");
    for bytes in [
        bad,
        wrong_dimension,
        wrong_bits,
        wrong_padding,
        original[..original.len() - 1].to_vec(),
    ] {
        fs::write(&malformed, bytes)?;
        assert!(OfficialIndex::open(&root, &malformed, bundle, &rows).is_err());
    }
    let heads = (0..32).collect::<Vec<_>>();
    let query = &vectors[..65];
    assert!(index.search(query, &heads, 10, 9, None).is_err());
    assert!(index.search(query, &heads, 10, 98, None).is_err());
    assert!(index.search(query, &[0, 0], 10, 20, None).is_err());
    assert!(index.search(query, &[32], 10, 20, None).is_err());
    assert!(index.search(&[f32::NAN; 65], &heads, 10, 20, None).is_err());
    assert!(index.search(&[f32::MAX; 65], &heads, 10, 20, None).is_err());
    assert_eq!(
        index.search(query, &heads, 10, 97, None)?.0,
        oracle(&vectors, query, &rows, 0..97, 10)
    );
    Ok(())
}

#[test]
fn test_official_query_state_is_serialized_across_threads() -> VortexResult<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path().join("native");
    let (bundle, vectors, rows) = fixture(&root, 65)?;
    let index = OfficialIndex::open(&root, &root.join("rabitq-official.bin"), bundle, &rows)?;
    thread::scope(|scope| -> VortexResult<()> {
        let handles = (0..4)
            .map(|offset| {
                let index = &index;
                let vectors = &vectors;
                let rows = &rows;
                scope.spawn(move || -> VortexResult<()> {
                    for id in [offset, offset + 7, offset + 23] {
                        let query = &vectors[id * 65..(id + 1) * 65];
                        let expected = oracle(vectors, query, rows, 0..97, 10);
                        assert_eq!(
                            index
                                .search(query, &(0..32).collect::<Vec<_>>(), 10, 97, None)?
                                .0,
                            expected
                        );
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle
                .join()
                .map_err(|_| vortex_err!("Query thread panicked"))??;
        }
        Ok(())
    })
}

#[test]
fn test_original_reader_preserves_filtered_shortlists_and_distances() -> VortexResult<()> {
    for dimension in [1, 31, 65, 128, 768, 1025, 4096] {
        let directory = tempfile::tempdir()?;
        let root = directory.path().join("native");
        let (bundle, vectors, rows) = fixture(&root, dimension)?;
        let model = root.join("rabitq-official.bin");
        let heads = (0..32).collect::<Vec<_>>();
        let query = &vectors[..dimension as usize];
        let index = OfficialIndex::open_with_diagnostics(&root, &model, bundle, &rows, true)?;
        let (ids, scores) = estimates(&index, query, 0)?;
        let mut candidates = ids.into_iter().zip(scores).collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| rows[left.0 as usize].cmp(&rows[right.0 as usize]))
        });
        let mut mask = vec![0; (bundle.rows as usize).div_ceil(8)];
        for id in (0..bundle.rows).step_by(3) {
            mask[id as usize / 8] |= 1 << (id % 8);
        }
        for budget in [17, bundle.rows] {
            for filter in [None, Some(mask.as_slice())] {
                let shortlist = candidates
                    .iter()
                    .map(|hit| hit.0)
                    .filter(|id| {
                        filter.is_none_or(|mask| mask[*id as usize / 8] & (1 << (id % 8)) != 0)
                    })
                    .take(budget as usize)
                    .collect::<Vec<_>>();
                let expected = oracle(&vectors, query, &rows, shortlist.iter().copied(), 10);
                let (actual, stats) = index.search(query, &heads, 10, budget, filter)?;
                assert_eq!(actual, expected);
                assert_eq!(stats.reranked, shortlist.len());
                assert!(stats.original_read_calls > 0);
                assert!(stats.original_read_bytes >= stats.original_bytes);
            }
        }
        thread::scope(|scope| -> VortexResult<()> {
            let (actual, _) = scope
                .spawn(|| index.search(query, &heads, 10, bundle.rows, None))
                .join()
                .map_err(|_| vortex_err!("IO query thread panicked"))??;
            assert_eq!(actual, oracle(&vectors, query, &rows, 0..bundle.rows, 10));
            Ok(())
        })?;
    }
    Ok(())
}

#[test]
fn test_original_reader_rejects_truncation_and_can_retry_after_repair() -> VortexResult<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path().join("native");
    let (bundle, vectors, rows) = fixture(&root, 768)?;
    let path = root.join("postings.bin");
    let bytes = fs::read(&path)?;
    let heads = (0..32).collect::<Vec<_>>();
    let index = OfficialIndex::open(&root, &root.join("rabitq-official.bin"), bundle, &rows)?;
    fs::OpenOptions::new().write(true).open(&path)?.set_len(0)?;
    assert!(index.search(&vectors[..768], &heads, 10, 97, None).is_err());
    fs::write(&path, &bytes)?;
    assert_eq!(
        index.search(&vectors[..768], &heads, 10, 97, None)?.0,
        oracle(&vectors, &vectors[..768], &rows, 0..97, 10)
    );
    Ok(())
}

#[test]
fn test_original_reader_handles_multiple_bounded_batches() -> VortexResult<()> {
    let directory = tempfile::tempdir()?;
    let root = directory.path().join("native");
    let (bundle, vectors, rows) = fixture_rows(&root, 65, 1057)?;
    let query = &vectors[..65];
    let expected = oracle(&vectors, query, &rows, 0..bundle.rows, 100);
    let index = OfficialIndex::open_with_diagnostics(
        &root,
        &root.join("rabitq-official.bin"),
        bundle,
        &rows,
        true,
    )?;
    let (hits, stats) =
        index.search(query, &(0..32).collect::<Vec<_>>(), 100, bundle.rows, None)?;
    assert_eq!(hits, expected);
    assert_eq!(stats.reranked, 1057);
    Ok(())
}
