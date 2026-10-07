// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_index::IndexMetadata;
use vortex_index::RowAddress;
use vortex_index::Snapshot;
use vortex_index::SourceFile;

use crate::HNSWLIB_ID;
use crate::HnswBundle;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;

#[cfg(feature = "native")]
mod native;

fn metadata(count: u64) -> IndexMetadata {
    IndexMetadata {
        format_version: 1,
        name: "embedding".into(),
        generation: "generation-1".into(),
        backend: HNSWLIB_ID.into(),
        backend_version: 1,
        fields: vec!["embedding".into()],
        covered_files: vec![11, 29],
        artifacts: Vec::new(),
        snapshot: Snapshot {
            dataset_id: "test".into(),
            version: "v1".into(),
            schema_fingerprint: "schema".into(),
            files: vec![
                SourceFile {
                    id: 11,
                    uri: "first".into(),
                    version: "a".into(),
                    row_count: count / 2,
                },
                SourceFile {
                    id: 29,
                    uri: "second".into(),
                    version: "b".into(),
                    row_count: count - count / 2,
                },
            ],
        },
    }
}

fn rows(count: u32) -> Vec<RowAddress> {
    (0..count)
        .map(|id| RowAddress {
            file_id: if id % 2 == 0 { 29 } else { 11 },
            row_offset: u64::from(id / 2),
        })
        .collect()
}

#[test]
fn mapping_requires_complete_coverage_not_native_order() -> VortexResult<()> {
    let metadata = metadata(3);
    let bundle = HnswBundle {
        dimension: 8,
        rows: 3,
        m: 8,
        ef_construction: 64,
    };
    validate_metadata(&metadata)?;
    validate_rows(&metadata, bundle, &rows(3))?;
    assert!(validate_rows(&metadata, bundle, &rows(2)).is_err());
    let mut duplicate = rows(3);
    duplicate[0] = duplicate[1];
    assert!(validate_rows(&metadata, bundle, &duplicate).is_err());
    let mut wrong = metadata;
    wrong.backend_version = 2;
    assert!(validate_metadata(&wrong).is_err());
    Ok(())
}

#[rstest]
#[case(0, 256, 8, 64)]
#[case(4097, 256, 8, 64)]
#[case(8, 0, 8, 64)]
#[case(8, u32::MAX, 8, 64)]
#[case(8, 256, 1, 64)]
#[case(8, 256, 65, 64)]
#[case(8, 256, 8, 7)]
#[case(8, 256, 8, 4097)]
fn unsupported_bundle_shape(
    #[case] dimension: u32,
    #[case] rows: u32,
    #[case] m: u32,
    #[case] ef_construction: u32,
) {
    assert!(
        HnswBundle {
            dimension,
            rows,
            m,
            ef_construction
        }
        .validate()
        .is_err()
    );
}
