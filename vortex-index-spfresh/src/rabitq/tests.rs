// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::io::Write;
use std::os::unix::fs::FileExt;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::StandardNormal;
use vortex_error::VortexResult;
use vortex_index::RowAddress;
use vortex_index::RowFilter;
use vortex_index::SearchHit;
use vortex_index::Snapshot;
use vortex_index::SourceFile;

use super::RaBitQ;
use super::RaBitQErrorBound;
use super::distance::ExactQuery;
use super::storage::QuantizedIndex;
use super::storage::build_sidecar;
use crate::SpFreshBundle;

#[test]
fn test_code_size_and_deterministic_bounds() -> VortexResult<()> {
    for dimension in [1, 7, 8, 9, 64, 1536] {
        let quantizer = RaBitQ::new(vec![1.0; dimension], 42)?;
        let vector: Vec<f32> = (0..dimension)
            .map(|i| ((i * 17 % 31) as f32 - 15.0) / 7.0)
            .collect();
        let code = quantizer.encode(&vector)?;
        assert_eq!(code.code().len(), dimension.div_ceil(8));
        for query in [vec![1.0; dimension], vector.clone(), vec![-2.0; dimension]] {
            let interval = quantizer
                .query(&query, RaBitQErrorBound::Deterministic)?
                .estimate(&code)?;
            let exact = vector
                .iter()
                .zip(query)
                .map(|(a, b)| (f64::from(*a) - f64::from(b)).powi(2))
                .sum::<f64>();
            assert!(
                interval.lower() <= exact && exact <= interval.upper(),
                "{dimension}: {interval:?} versus {exact}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_zero_residual_and_invalid_inputs() -> VortexResult<()> {
    let quantizer = RaBitQ::new(vec![2.0; 9], 2)?;
    let zero = quantizer.encode(&[2.0; 9])?;
    let result = quantizer
        .query(&[3.0; 9], RaBitQErrorBound::Probabilistic { epsilon: 4.0 })?
        .estimate(&zero)?;
    assert_eq!(result.estimate, 9.0);
    assert!(result.error < 1e-10);
    assert!(quantizer.encode(&[f32::NAN; 9]).is_err());
    assert!(quantizer.encode(&[1.0; 8]).is_err());
    assert!(
        quantizer
            .query(&[1.0; 9], RaBitQErrorBound::Probabilistic { epsilon: 0.0 })
            .is_err()
    );
    assert!(RaBitQ::new(Vec::new(), 0).is_err());
    assert!(RaBitQ::new(vec![f64::INFINITY], 0).is_err());
    Ok(())
}

#[test]
fn test_prepared_bound_constants_preserve_error_bits() -> VortexResult<()> {
    for dimension in [1, 7, 128] {
        let quantizer = RaBitQ::new(vec![1.0; dimension], 42)?;
        let vector: Vec<f32> = (0..dimension).map(|i| (i % 11) as f32 / 3.0).collect();
        let query: Vec<f32> = (0..dimension).map(|i| (i % 7) as f32 / 13.0).collect();
        let code = quantizer.encode(&vector)?;
        let norm_squared = query
            .iter()
            .map(|x| (f64::from(*x) - 1.0).powi(2))
            .sum::<f64>();
        for bound in [
            RaBitQErrorBound::Deterministic,
            RaBitQErrorBound::Probabilistic { epsilon: 1.0 },
        ] {
            let prepared = quantizer.query(&query, bound)?;
            let expected = 2.0 * code.residual * norm_squared.sqrt() * bound.factor(dimension)
                + 128.0
                    * (dimension as f64).powi(2)
                    * f64::EPSILON
                    * (code.norm_squared
                        + norm_squared
                        + 2.0 * code.scale * norm_squared.sqrt()
                        + 1.0);
            assert_eq!(
                prepared.estimate(&code)?.error.to_bits(),
                expected.to_bits()
            );
        }
    }
    Ok(())
}

// A real static posting layout with one head and every other vector in its list.
fn fixture(
    values: &[Vec<f32>],
) -> VortexResult<(tempfile::TempDir, SpFreshBundle, Vec<RowAddress>, RowFilter)> {
    let dir = tempfile::tempdir()?;
    let dimension = u32::try_from(values[0].len())?;
    let rows = u32::try_from(values.len())?;
    let count = rows - 1;
    let pages = (count * (4 + 4 * dimension)).div_ceil(4096);
    let bundle = SpFreshBundle {
        dimension,
        rows,
        posting_page_limit: pages.max(1),
    };
    let mut heads = fs::File::create(dir.path().join("vectors.bin"))?;
    heads.write_all(&1u32.to_le_bytes())?;
    heads.write_all(&dimension.to_le_bytes())?;
    for value in &values[0] {
        heads.write_all(&value.to_le_bytes())?;
    }
    fs::write(dir.path().join("head_ids.bin"), 0u64.to_le_bytes())?;
    let mut postings = vec![0u8; (4096 * (pages + 1)) as usize];
    for (offset, value) in [
        (0, 1),
        (4, rows),
        (8, dimension),
        (12, 1),
        (16, 0),
        (22, count),
    ] {
        postings[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }
    postings[26..28].copy_from_slice(&u16::try_from(pages)?.to_le_bytes());
    let mut offset = 4096;
    for (id, vector) in values.iter().enumerate().skip(1) {
        postings[offset..offset + 4].copy_from_slice(&u32::try_from(id)?.to_le_bytes());
        offset += 4;
        for value in vector {
            postings[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            offset += 4;
        }
    }
    fs::write(dir.path().join("postings.bin"), postings)?;
    let snapshot = Snapshot {
        dataset_id: "rabitq".into(),
        version: "v1".into(),
        schema_fingerprint: "f32".into(),
        files: vec![SourceFile {
            id: 7,
            uri: "fixture".into(),
            version: "v1".into(),
            row_count: u64::from(rows),
        }],
    };
    let addresses = (0..rows)
        .map(|id| RowAddress {
            file_id: 7,
            row_offset: u64::from(rows - id - 1),
        })
        .collect();
    let filter = RowFilter::try_new(snapshot, None, Default::default())?;
    Ok((dir, bundle, addresses, filter))
}

#[test]
fn test_persistence_pruning_and_exact_rerank() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..128).map(|i| vec![i as f32; 9]).collect();
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    for bound in [
        RaBitQErrorBound::Deterministic,
        RaBitQErrorBound::Probabilistic { epsilon: 4.0 },
    ] {
        let (hits, stats) = index.search(&[0.0; 9], &[0, 0], &rows, &filter, 3, bound)?;
        assert_eq!(hits.iter().map(|h| h.row).collect::<Vec<_>>(), rows[..3]);
        assert_eq!(
            hits.iter().map(|h| h.distance).collect::<Vec<_>>(),
            [0.0, 9.0, 36.0]
        );
        assert_eq!(stats.scanned, 128);
        assert!(stats.reranked < stats.scanned);
        assert_eq!(stats.original_bytes, stats.reranked * 9 * 4);
    }
    // Exact originals are fetched at query time and failures propagate.
    fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("postings.bin"))?
        .set_len(4096)?;
    assert!(
        index
            .search(
                &[64.0; 9],
                &[0],
                &rows,
                &filter,
                3,
                RaBitQErrorBound::Deterministic
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn test_original_distance_matches_ordered_f64_sum() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..17)
        .map(|i| (0..9).map(|j| (i * 7 + j * 13) as f32 / 11.0).collect())
        .collect();
    let (dir, bundle, ..) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    let mut reader = index.original_reader(0)?;
    for query in [&values[3], &vec![-0.0; 9], &vec![f32::MAX; 9]] {
        for (id, value) in values.iter().enumerate() {
            let expected = value
                .iter()
                .zip(query)
                .map(|(left, right)| {
                    let delta = f64::from(*left) - f64::from(*right);
                    delta * delta
                })
                .sum::<f64>();
            assert_eq!(
                reader.exact_distance(u32::try_from(id)?, query)?.to_bits(),
                expected.to_bits()
            );
        }
    }
    Ok(())
}

#[test]
fn test_parallel_distance_preserves_f32_rounding() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(91);
    for dimension in [1, 3, 4, 7, 128, 257, 4096] {
        let query: Vec<f32> = (0..dimension)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        let prepared = ExactQuery::new(&query)?;
        for vector in [
            query.clone(),
            vec![-0.0; dimension],
            vec![f32::MAX; dimension],
            vec![f32::from_bits(1); dimension],
            (0..dimension)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect(),
        ] {
            let expected = vector
                .iter()
                .zip(&query)
                .map(|(left, right)| {
                    let delta = f64::from(*left) - f64::from(*right);
                    delta * delta
                })
                .sum::<f64>();
            assert_eq!(
                prepared.score(&vector)?.0.to_bits(),
                super::search::score(expected).to_bits()
            );
        }
    }
    Ok(())
}

#[test]
fn test_parallel_distance_falls_back_at_rounding_boundary() -> VortexResult<()> {
    let query = vec![0.0; 130];
    let mut vector = vec![2.0f32.powi(-28); 130];
    vector[0] = 1.0;
    vector[1] = 2.0f32.powi(-12);
    let (distance, fallback) = ExactQuery::new(&query)?.score(&vector)?;
    assert_eq!(distance, 1.0);
    assert!(fallback);
    Ok(())
}

#[test]
fn test_parallel_distance_rejects_invalid_inputs() -> VortexResult<()> {
    assert!(ExactQuery::new(&[]).is_err());
    assert!(ExactQuery::new(&[f32::NAN]).is_err());
    assert!(ExactQuery::new(&[0.0; 4097]).is_err());
    let query = ExactQuery::new(&[0.0; 4])?;
    assert!(query.score(&[0.0; 3]).is_err());
    assert!(query.score(&[f32::INFINITY; 4]).is_err());
    assert!(query.score(&[f32::NAN; 4]).is_err());
    Ok(())
}

#[test]
fn test_parallel_distance_with_mixed_exponents() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(407);
    for dimension in [1, 2, 3, 4, 5, 7, 8, 127, 128, 129, 4095, 4096] {
        for trial in 0..16 {
            let mut query = Vec::with_capacity(dimension);
            let mut vector = Vec::with_capacity(dimension);
            for values in [&mut query, &mut vector] {
                for component in 0..dimension {
                    let value: f32 = StandardNormal.sample(&mut rng);
                    let exponent = u32::try_from((component * 17 + trial * 7) % 80 + 85)?;
                    values.push(f32::from_bits(
                        (value.to_bits() & 0x807f_ffff) | (exponent << 23),
                    ));
                }
            }
            let expected = vector
                .iter()
                .zip(&query)
                .map(|(left, right)| {
                    let delta = f64::from(*left) - f64::from(*right);
                    delta * delta
                })
                .sum::<f64>();
            assert_eq!(
                ExactQuery::new(&query)?.score(&vector)?.0.to_bits(),
                super::search::score(expected).to_bits(),
                "dimension={dimension}, trial={trial}"
            );
        }
    }
    Ok(())
}

#[test]
fn test_non_finite_original_after_open_is_rejected() -> VortexResult<()> {
    let values = vec![vec![0.0; 9]; 17];
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("vectors.bin"))?
        .write_all_at(&f32::NAN.to_le_bytes(), 8)?;
    assert!(
        index
            .search(
                &[0.0; 9],
                &[0],
                &rows,
                &filter,
                3,
                RaBitQErrorBound::Deterministic
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn test_original_reads_are_amortized_over_candidates() -> VortexResult<()> {
    let values = vec![vec![0.0; 128]; 512];
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    let (hits, stats) = index.search(
        &[0.0; 128],
        &[0],
        &rows,
        &filter,
        3,
        RaBitQErrorBound::Deterministic,
    )?;
    assert_eq!(
        hits.iter().map(|hit| hit.row).collect::<Vec<_>>(),
        rows.iter().rev().take(3).copied().collect::<Vec<_>>()
    );
    assert_eq!(stats.reranked, values.len());
    assert!(
        stats.original_read_calls < stats.reranked / 4,
        "{} reads for {} candidates",
        stats.original_read_calls,
        stats.reranked
    );
    Ok(())
}

#[test]
fn test_original_reader_boundaries_and_cache_limit() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..512).map(|id| vec![id as f32; 128]).collect();
    let (dir, bundle, ..) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    let mut reader = index.original_reader(32 * 1024)?;
    for id in [0u32, 1, 24, 31, 55, 87, 120, 128, 255, 256, 510, 511] {
        let delta = f64::from(id) - 121.0;
        assert_eq!(
            reader.exact_distance(id, &[121.0; 128])?,
            128.0 * delta * delta
        );
        assert!(reader.cached_bytes <= 32 * 1024);
    }
    Ok(())
}

#[test]
fn test_packed_codes_match_public_encoding() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..17)
        .map(|id| (0..65).map(|i| (id * 7 + i * 13) as f32 / 17.0).collect())
        .collect();
    let (dir, bundle, ..) = fixture(&values)?;
    let path = dir.path().join("postings.bin");
    let mut postings = fs::read(&path)?;
    let stride = 4 + values[0].len() * 4;
    let end = 4096 + (values.len() - 1) * stride;
    let payload = postings[4096..end].to_vec();
    for (record, original) in postings[4096..end]
        .chunks_exact_mut(stride)
        .zip(payload.chunks_exact(stride).rev())
    {
        record.copy_from_slice(original);
    }
    fs::write(path, postings)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    assert_eq!(index.codes.len(), values.len());
    for query in [&values[0], &values[8], &values[16]] {
        for bound in [
            RaBitQErrorBound::Deterministic,
            RaBitQErrorBound::Probabilistic { epsilon: 1.0 },
        ] {
            let prepared = index.quantizer.query(query, bound)?;
            for (id, value) in values.iter().enumerate() {
                let expected = prepared.estimate(&index.quantizer.encode(value)?)?;
                let actual = index.codes.estimate(&prepared, id)?;
                assert_eq!(actual.estimate.to_bits(), expected.estimate.to_bits());
                assert_eq!(actual.error.to_bits(), expected.error.to_bits());
            }
            for start in [0u32, 4, 8, 12] {
                let ids = [start, start + 1, start + 2, start + 3];
                let batch = index.codes.estimate_four(&prepared, ids)?;
                for (id, actual) in ids.into_iter().zip(batch) {
                    let expected =
                        prepared.estimate(&index.quantizer.encode(&values[id as usize])?)?;
                    assert_eq!(actual.estimate.to_bits(), expected.estimate.to_bits());
                    assert_eq!(actual.error.to_bits(), expected.error.to_bits());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn test_randomized_estimator_and_bounds() -> VortexResult<()> {
    let dimension = 65;
    let vector: Vec<f32> = (0..dimension)
        .map(|i| ((i * 41 % 113) as f32 - 53.0) / 21.0)
        .collect();
    let query: Vec<f32> = (0..dimension)
        .map(|i| ((i * 73 % 127) as f32 - 64.0) / 23.0)
        .collect();
    let exact = vector
        .iter()
        .zip(&query)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>();
    let mut error_sum = 0.0;
    let mut misses = 0;
    for seed in 0..128 {
        let quantizer = RaBitQ::new(vec![0.0; dimension], seed)?;
        let code = quantizer.encode(&vector)?;
        let strict = quantizer
            .query(&query, RaBitQErrorBound::Deterministic)?
            .estimate(&code)?;
        assert!(strict.lower() <= exact && exact <= strict.upper());
        let interval = quantizer
            .query(&query, RaBitQErrorBound::Probabilistic { epsilon: 4.0 })?
            .estimate(&code)?;
        error_sum += interval.estimate - exact;
        if interval.lower() > exact || exact > interval.upper() {
            misses += 1;
        }
    }
    assert!(error_sum.abs() / (128.0 * exact) < 0.03);
    assert!(misses <= 1);
    Ok(())
}

#[test]
fn test_filtered_topk_ties_replicas_and_empty_candidates() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..96)
        .map(|i| {
            (0..17)
                .map(|j| ((i / 2 * 31 + j * 17) % 61) as f32 - 30.0)
                .collect()
        })
        .collect();
    let (dir, bundle, rows, all) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 22)?;
    let mut index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    index.postings[0].extend([1, 1, 2, 0]);
    let restricted = RowFilter::try_new(
        all.snapshot().clone(),
        Some(rows.iter().step_by(3).copied().collect()),
        [rows[0]].into(),
    )?;
    let empty = RowFilter::try_new(
        all.snapshot().clone(),
        Some(Default::default()),
        Default::default(),
    )?;
    for filter in [&all, &restricted, &empty] {
        for query in [&values[0], &values[31], &vec![0.0; 17]] {
            let mut expected: Vec<_> = values
                .iter()
                .zip(&rows)
                .filter(|(_, row)| filter.includes(**row))
                .map(|(vector, row)| {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "Match common f32 distance semantics"
                    )]
                    let distance = vector
                        .iter()
                        .zip(query)
                        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                        .sum::<f64>() as f32;
                    SearchHit {
                        row: *row,
                        distance,
                    }
                })
                .collect();
            expected.sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.row.cmp(&b.row)));
            for k in [1, 7, 128] {
                let (hits, stats) = index.search(
                    query,
                    &[0, 0],
                    &rows,
                    filter,
                    k,
                    RaBitQErrorBound::Deterministic,
                )?;
                assert_eq!(hits, expected[..expected.len().min(k)]);
                assert!(stats.reranked <= stats.scanned);
                if expected.is_empty() {
                    assert_eq!(stats.original_bytes, 0);
                }
            }
        }
    }
    assert!(
        index
            .search(
                &values[0],
                &[],
                &rows,
                &all,
                3,
                RaBitQErrorBound::Deterministic
            )?
            .0
            .is_empty()
    );
    Ok(())
}

#[test]
fn test_sidecar_rejects_corruption_and_preserves_original_handles() -> VortexResult<()> {
    let values: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32; 9]).collect();
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let path = dir.path().join("rabitq.bin");
    let valid = fs::read(&path)?;
    // Header, shape, length, non-unit rotation and non-zero final code padding.
    for offset in [0, 8, 16, 20, 24, 32 + 9 * 8 + 6, valid.len() - 1] {
        let mut bytes = valid.clone();
        bytes[offset] ^= 0x80;
        fs::write(&path, bytes)?;
        assert!(
            QuantizedIndex::open(dir.path(), &path, bundle).is_err(),
            "accepted corruption at {offset}"
        );
    }
    fs::write(&path, &valid[..valid.len() - 1])?;
    assert!(QuantizedIndex::open(dir.path(), &path, bundle).is_err());
    fs::write(&path, &valid)?;
    let index = QuantizedIndex::open(dir.path(), &path, bundle)?;
    // Replacement of the path cannot redirect reads away from owned descriptors.
    fs::rename(dir.path().join("postings.bin"), dir.path().join("old.bin"))?;
    fs::write(dir.path().join("postings.bin"), [])?;
    let (hits, _) = index.search(
        &values[5],
        &[0],
        &rows,
        &filter,
        1,
        RaBitQErrorBound::Deterministic,
    )?;
    assert_eq!(hits[0].row, rows[5]);
    Ok(())
}

#[test]
fn test_extreme_finite_components_and_distance_overflow() -> VortexResult<()> {
    let values = vec![vec![f32::MAX; 3]; 17];
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 9)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    let (hits, _) = index.search(
        &values[0],
        &[0],
        &rows,
        &filter,
        3,
        RaBitQErrorBound::Deterministic,
    )?;
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|hit| hit.distance == 0.0));
    assert!(
        index
            .search(
                &[-f32::MAX; 3],
                &[0],
                &rows,
                &filter,
                1,
                RaBitQErrorBound::Deterministic
            )
            .is_err()
    );
    Ok(())
}

#[test]
fn test_high_dimension_confidence_prunes_original_reads() -> VortexResult<()> {
    let mut rng = StdRng::seed_from_u64(1234);
    let values: Vec<Vec<f32>> = (0..256)
        .map(|_| (0..128).map(|_| StandardNormal.sample(&mut rng)).collect())
        .collect();
    let (dir, bundle, rows, filter) = fixture(&values)?;
    build_sidecar(dir.path(), bundle, 7)?;
    let index = QuantizedIndex::open(dir.path(), &dir.path().join("rabitq.bin"), bundle)?;
    let (strict, strict_stats) = index.search(
        &values[0],
        &[0],
        &rows,
        &filter,
        1,
        RaBitQErrorBound::Deterministic,
    )?;
    let (confidence, stats) = index.search(
        &values[0],
        &[0],
        &rows,
        &filter,
        1,
        RaBitQErrorBound::Probabilistic { epsilon: 4.0 },
    )?;
    assert_eq!(confidence, strict);
    assert_eq!(confidence[0].row, rows[0]);
    assert_eq!(confidence[0].distance, 0.0);
    assert_eq!(stats.scanned, 256);
    assert!(stats.reranked < stats.scanned / 4);
    assert!(stats.reranked <= strict_stats.reranked);
    // On this separated fixture all non-head originals are pruned. Removing the
    // posting payload proves that coarse search itself performs no original IO.
    assert_eq!(stats.reranked, 1);
    fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("postings.bin"))?
        .set_len(4096)?;
    assert_eq!(
        index
            .search(
                &values[0],
                &[0],
                &rows,
                &filter,
                1,
                RaBitQErrorBound::Probabilistic { epsilon: 4.0 }
            )?
            .0,
        confidence
    );
    Ok(())
}

fn inverse_rotation(quantizer: &RaBitQ, mut vector: Vec<f64>) -> Vec<f64> {
    let dimension = vector.len();
    for (value, sign) in vector.iter_mut().zip(&quantizer.signs) {
        *value *= sign;
    }
    for start in (0..dimension).rev() {
        let offset = dimension * start - start * start.saturating_sub(1) / 2;
        let reflector = &quantizer.reflectors[offset..offset + dimension - start];
        let dot = vector[start..]
            .iter()
            .zip(reflector)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        for (value, direction) in vector[start..].iter_mut().zip(reflector) {
            *value -= 2.0 * dot * direction;
        }
    }
    vector
}

#[test]
fn test_near_codebook_vertices_do_not_underestimate_residual() -> VortexResult<()> {
    for seed in 0..32 {
        let quantizer = RaBitQ::new(vec![0.0; 17], seed)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Test original f32 values near a codebook vertex"
        )]
        let vector: Vec<f32> = inverse_rotation(&quantizer, vec![1.0; 17])
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let code = quantizer.encode(&vector)?;
        let reconstructed = inverse_rotation(&quantizer, vec![code.scale / 17.0f64.sqrt(); 17]);
        let error: Vec<f64> = vector
            .iter()
            .zip(reconstructed)
            .map(|(a, b)| f64::from(*a) - b)
            .collect();
        let norm = error.iter().map(|v| v * v).sum::<f64>().sqrt();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The query API accepts f32 components"
        )]
        let query: Vec<f32> = error.iter().map(|v| (v / norm) as f32).collect();
        let interval = quantizer
            .query(&query, RaBitQErrorBound::Deterministic)?
            .estimate(&code)?;
        let exact = vector
            .iter()
            .zip(query)
            .map(|(a, b)| (f64::from(*a) - f64::from(b)).powi(2))
            .sum::<f64>();
        assert!(
            interval.lower() <= exact && exact <= interval.upper(),
            "{seed}: {interval:?} vs {exact}"
        );
    }
    Ok(())
}
