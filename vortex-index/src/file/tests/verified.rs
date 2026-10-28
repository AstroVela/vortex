// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use futures::TryStreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::single::block_on;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;

use super::BUDGET;
use super::Fixture;
use super::assert_error;
use super::data;
use super::fields;
use super::row;
use crate::IndexSource;
use crate::file::LocalFileSource;
use crate::file::VerifiedFile;
use crate::file::file_version;
use crate::file::schema_fingerprint;

fn verified(fixture: &Fixture) -> VortexResult<BTreeMap<u64, VerifiedFile>> {
    fixture
        .snapshot
        .files
        .iter()
        .map(|entry| {
            let bytes = ByteBuffer::from(fs::read(&entry.uri)?);
            Ok((
                entry.id,
                VerifiedFile::from_buffer(entry.clone(), bytes, BUDGET, fixture.session.clone())?,
            ))
        })
        .collect()
}

#[test]
fn test_verified_files_retain_readers_through_assembly_and_path_removal() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let files = verified(&fixture)?;
        let total = files
            .values()
            .map(VerifiedFile::pinned_bytes)
            .sum::<usize>();
        for file in files.values() {
            assert_eq!(file.dtype(), &fixture.dtype);
        }
        fs::remove_dir_all(fixture.dir.path())?;
        let source = LocalFileSource::from_verified(fixture.snapshot, fixture.dtype, files, total)?;
        assert_eq!(source.pinned_bytes(), total);
        let selected = [row(30, 1), row(10, 4), row(30, 1), row(10, 0)];
        let taken = source
            .take(&selected, &fields(&["id", "embedding", "note"]))
            .await?;
        assert_eq!(taken.rows, selected);
        assert_arrays_eq!(
            taken.data,
            data(&[31, 14, 31, 10])?.into_array(),
            &mut fixture.session.create_execution_ctx()
        );
        let batches = source
            .scan(&[30, 10], &fields(&["id"]))?
            .try_collect::<Vec<_>>()
            .await?;
        assert_eq!(
            batches
                .into_iter()
                .flat_map(|batch| batch.rows)
                .collect::<Vec<_>>(),
            (0..3)
                .map(|offset| row(30, offset))
                .chain((0..5).map(|offset| row(10, offset)))
                .collect::<Vec<_>>()
        );
        Ok(())
    })
}

#[test]
fn test_verified_files_reject_rebinding_inventories_schema_and_aggregate_budget() -> VortexResult<()>
{
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let open = |snapshot, dtype, files, budget| {
            LocalFileSource::from_verified(snapshot, dtype, files, budget)
        };
        for change in 0..4 {
            let mut snapshot = fixture.snapshot.clone();
            match change {
                0 => snapshot.files[0].id = 9,
                1 => snapshot.files[0].uri.push_str(".replaced"),
                2 => snapshot.files[0].version = format!("sha256:{}", "0".repeat(64)),
                _ => snapshot.files[0].row_count += 1,
            }
            assert!(open(snapshot, fixture.dtype.clone(), verified(&fixture)?, BUDGET).is_err());
        }
        let mut missing = verified(&fixture)?;
        missing.remove(&10);
        assert_error(
            open(
                fixture.snapshot.clone(),
                fixture.dtype.clone(),
                missing,
                BUDGET,
            ),
            "inventory",
        )?;
        let mut unknown = verified(&fixture)?;
        let first = unknown
            .remove(&10)
            .ok_or_else(|| vortex_err!("Missing fixture file"))?;
        unknown.insert(99, first);
        assert_error(
            open(
                fixture.snapshot.clone(),
                fixture.dtype.clone(),
                unknown,
                BUDGET,
            ),
            "Missing verified",
        )?;
        let files = verified(&fixture)?;
        let total = files
            .values()
            .map(VerifiedFile::pinned_bytes)
            .sum::<usize>();
        assert_error(
            open(
                fixture.snapshot.clone(),
                fixture.dtype.clone(),
                files,
                total - 1,
            ),
            "pinned byte budget",
        )?;
        let types = fixture
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_err!("Expected struct"))?
            .fields()
            .collect();
        let dtype = DType::Struct(
            StructFields::new(["different_id", "embedding", "note"].into(), types),
            Nullability::NonNullable,
        );
        let mut snapshot = fixture.snapshot.clone();
        snapshot.schema_fingerprint = schema_fingerprint(&dtype)?;
        assert_error(
            open(snapshot, dtype, verified(&fixture)?, BUDGET),
            "schema or row count mismatch",
        )?;
        let mut snapshot = fixture.snapshot.clone();
        snapshot.schema_fingerprint = "wrong".into();
        assert_error(
            open(snapshot, fixture.dtype.clone(), verified(&fixture)?, BUDGET),
            "schema fingerprint",
        )?;
        Ok(())
    })
}

#[test]
fn test_verified_file_checks_contents_rows_budget_and_descriptor() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let entry = fixture.snapshot.files[0].clone();
        let bytes = ByteBuffer::from(fs::read(&entry.uri)?);
        let open = |entry, bytes, budget| {
            VerifiedFile::from_buffer(entry, bytes, budget, fixture.session.clone())
        };
        assert_error(
            open(entry.clone(), ByteBuffer::from(vec![0_u8; 32]), BUDGET),
            "content version mismatch",
        )?;
        assert_error(
            open(entry.clone(), bytes.clone(), bytes.len() - 1),
            "pinned byte budget",
        )?;
        let mut changed = entry.clone();
        changed.row_count += 1;
        assert_error(open(changed, bytes.clone(), BUDGET), "row count mismatch")?;
        let mut changed = entry.clone();
        changed.uri = "relative.vortex".into();
        assert_error(open(changed, bytes.clone(), BUDGET), "absolute path")?;
        let mut changed = entry;
        changed.version = "sha256:bad".into();
        assert_error(open(changed, bytes, BUDGET), "lowercase SHA-256")?;
        Ok(())
    })
}

#[test]
fn test_verified_file_requires_embedded_dtype_without_path_retry() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let mut entry = fixture.snapshot.files[1].clone();
        let mut bytes = Vec::new();
        fixture
            .session
            .write_options()
            .with_strategy(Arc::new(FlatLayoutStrategy::default()))
            .exclude_dtype()
            .write(
                &mut bytes,
                data(&[30, 31, 32])?.into_array().to_array_stream(),
            )
            .await?;
        entry.version = file_version(&bytes);
        assert!(
            VerifiedFile::from_buffer(entry, ByteBuffer::from(bytes), BUDGET, fixture.session)
                .is_err()
        );
        Ok(())
    })
}
