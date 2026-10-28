// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs;

use futures::TryStreamExt;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;

use super::BUDGET;
use super::Fixture;
use super::assert_error;
use super::data;
use super::fields;
use super::row;
use crate::IndexSource;
use crate::file::LocalFileSource;
use crate::file::schema_fingerprint;

fn buffers(fixture: &Fixture) -> VortexResult<BTreeMap<u64, ByteBuffer>> {
    fixture
        .snapshot
        .files
        .iter()
        .map(|file| Ok((file.id, ByteBuffer::from(fs::read(&file.uri)?))))
        .collect()
}

#[test]
fn test_buffers_preserve_scan_take_and_ownership_without_paths() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let bytes = buffers(&fixture)?;
        let expected_bytes = bytes.values().map(|buffer| buffer.len()).sum::<usize>();
        fs::remove_dir_all(fixture.dir.path())?;
        let source = LocalFileSource::from_buffers(
            fixture.snapshot.clone(),
            fixture.dtype.clone(),
            bytes.clone(),
            BUDGET,
            fixture.session.clone(),
        )?;
        drop(bytes);
        assert_eq!(source.pinned_bytes(), expected_bytes);
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
fn test_buffers_reject_incomplete_unknown_corrupt_and_over_budget() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let bytes = buffers(&fixture)?;
        let open = |bytes, budget| {
            LocalFileSource::from_buffers(
                fixture.snapshot.clone(),
                fixture.dtype.clone(),
                bytes,
                budget,
                fixture.session.clone(),
            )
        };
        let mut missing = bytes.clone();
        missing.remove(&10);
        assert_error(open(missing, BUDGET), "inventory")?;
        let mut unknown = bytes.clone();
        let first = unknown
            .remove(&10)
            .ok_or_else(|| vortex_error::vortex_err!("Missing fixture bytes"))?;
        unknown.insert(99, first);
        assert_error(open(unknown, BUDGET), "Missing source buffer")?;
        let mut corrupt = bytes.clone();
        corrupt.insert(30, ByteBuffer::from(vec![0_u8; 32]));
        assert_error(open(corrupt, BUDGET), "content version mismatch")?;
        let total = bytes.values().map(|bytes| bytes.len()).sum::<usize>();
        assert_error(open(bytes, total - 1), "pinned byte budget")?;
        Ok(())
    })
}

#[test]
fn test_buffers_validate_snapshot_schema_and_rows() -> VortexResult<()> {
    block_on(|handle| async move {
        let fixture = Fixture::new(handle).await?;
        let bytes = buffers(&fixture)?;
        let mut changed = fixture.snapshot.clone();
        changed.files[0].row_count += 1;
        assert_error(
            LocalFileSource::from_buffers(
                changed,
                fixture.dtype.clone(),
                bytes.clone(),
                BUDGET,
                fixture.session.clone(),
            ),
            "schema or row count mismatch",
        )?;
        let mut changed = fixture.snapshot.clone();
        changed.schema_fingerprint = "wrong".into();
        assert_error(
            LocalFileSource::from_buffers(
                changed,
                fixture.dtype.clone(),
                bytes.clone(),
                BUDGET,
                fixture.session.clone(),
            ),
            "schema fingerprint",
        )?;
        let types = fixture
            .dtype
            .as_struct_fields_opt()
            .ok_or_else(|| vortex_error::vortex_err!("Expected struct"))?
            .fields()
            .collect();
        let changed_dtype = DType::Struct(
            StructFields::new(["different_id", "embedding", "note"].into(), types),
            Nullability::NonNullable,
        );
        let mut changed = fixture.snapshot.clone();
        changed.schema_fingerprint = schema_fingerprint(&changed_dtype)?;
        assert_error(
            LocalFileSource::from_buffers(
                changed,
                changed_dtype,
                bytes.clone(),
                BUDGET,
                fixture.session.clone(),
            ),
            "schema or row count mismatch",
        )?;
        let mut changed = fixture.snapshot.clone();
        changed.files[0].uri = "relative.vortex".into();
        assert_error(
            LocalFileSource::from_buffers(
                changed,
                fixture.dtype.clone(),
                bytes,
                BUDGET,
                fixture.session.clone(),
            ),
            "absolute path",
        )?;
        let mut empty = fixture.snapshot.clone();
        empty.files.clear();
        let source = LocalFileSource::from_buffers(
            empty,
            fixture.dtype,
            BTreeMap::new(),
            0,
            fixture.session,
        )?;
        assert_eq!(source.pinned_bytes(), 0);
        Ok(())
    })
}
