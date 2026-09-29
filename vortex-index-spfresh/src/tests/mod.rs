// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_index::IndexMetadata;
use vortex_index::RowAddress;
use vortex_index::Snapshot;
use vortex_index::SourceFile;

use crate::SPFRESH_FORMAT_VERSION;
use crate::SPFRESH_ID;
use crate::SpFreshBundle;
use crate::bundle::validate_metadata;
use crate::bundle::validate_rows;

#[cfg(feature = "native")]
mod native;

fn metadata(snapshot: Snapshot) -> IndexMetadata {
    IndexMetadata {
        format_version: 1,
        name: "embedding".into(),
        generation: "generation-1".into(),
        backend: SPFRESH_ID.into(),
        backend_version: SPFRESH_FORMAT_VERSION,
        covered_files: snapshot.files.iter().map(|file| file.id).collect(),
        snapshot,
        fields: vec!["embedding".into()],
        artifacts: Vec::new(),
    }
}

#[test]
fn test_mapping_requires_exact_coverage_but_not_native_id_order() -> VortexResult<()> {
    let snapshot = Snapshot {
        dataset_id: "test".into(),
        version: "v1".into(),
        schema_fingerprint: "schema".into(),
        files: vec![
            SourceFile {
                id: 11,
                uri: "first".into(),
                version: "a".into(),
                row_count: 2,
            },
            SourceFile {
                id: 29,
                uri: "second".into(),
                version: "b".into(),
                row_count: 1,
            },
        ],
    };
    let metadata = metadata(snapshot);
    let bundle = SpFreshBundle {
        dimension: 8,
        rows: 3,
        posting_page_limit: 12,
    };
    let rows = vec![
        RowAddress {
            file_id: 29,
            row_offset: 0,
        },
        RowAddress {
            file_id: 11,
            row_offset: 1,
        },
        RowAddress {
            file_id: 11,
            row_offset: 0,
        },
    ];
    validate_metadata(&metadata)?;
    validate_rows(&metadata, bundle, &rows)?;
    assert!(validate_rows(&metadata, bundle, &rows[..2]).is_err());
    let mut duplicate = rows.clone();
    duplicate[0] = rows[1];
    assert!(validate_rows(&metadata, bundle, &duplicate).is_err());
    let mut outside = rows.clone();
    outside[0].row_offset = 1;
    assert!(validate_rows(&metadata, bundle, &outside).is_err());
    let mut uncovered = metadata.clone();
    uncovered.covered_files = vec![11];
    assert!(validate_rows(&uncovered, bundle, &rows).is_err());
    let mut wrong_backend = metadata;
    wrong_backend.backend_version += 1;
    assert!(validate_metadata(&wrong_backend).is_err());
    Ok(())
}

#[test]
fn test_bundle_rejects_unsupported_shape() {
    for bundle in [
        SpFreshBundle {
            dimension: 0,
            rows: 256,
            posting_page_limit: 12,
        },
        SpFreshBundle {
            dimension: 4097,
            rows: 256,
            posting_page_limit: 12,
        },
        SpFreshBundle {
            dimension: 8,
            rows: 0,
            posting_page_limit: 12,
        },
        SpFreshBundle {
            dimension: 8,
            rows: u32::MAX,
            posting_page_limit: 12,
        },
        SpFreshBundle {
            dimension: 8,
            rows: 256,
            posting_page_limit: 0,
        },
    ] {
        assert!(bundle.validate().is_err());
    }
}
