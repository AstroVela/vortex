// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Build a bounded fvecs fixture through the public static builder without changing SQL defaults.

use std::env;
use std::fs;
use std::io::BufReader;
use std::io::Error as IoError;
use std::io::Read;
use std::io::Write;
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use rustix::fs as unix_fs;
use serde::Deserialize;
use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_file::WriteOptionsSessionExt;
use vortex_index::IndexBuildRequest;
use vortex_index::IndexBuilder;
use vortex_index::IndexMetadata;
use vortex_index::IndexProvider;
use vortex_index::IndexStore;
use vortex_index::Snapshot;
use vortex_index::SourceFile;
use vortex_index::file::LocalFileSource;
use vortex_index::file::file_version;
use vortex_index::file::schema_fingerprint;
use vortex_index::store::LocalIndexStore;
use vortex_index::store::LocalStoreLimits;
use vortex_index_spfresh::SPFRESH_ID;
use vortex_index_spfresh::SpFreshBuildLimits;
use vortex_index_spfresh::SpFreshBuildOptions;
use vortex_index_spfresh::SpFreshIndexBuilder;
use vortex_index_spfresh::SpFreshLimits;
use vortex_index_spfresh::SpFreshProvider;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

const INPUT_LIMIT: u64 = 512 * 1024 * 1024;
const SOURCE_LIMIT: usize = 512 * 1024 * 1024;
const STORE_LIMITS: LocalStoreLimits = LocalStoreLimits {
    max_artifact_bytes: 1024 * 1024 * 1024,
    max_manifest_bytes: 16 * 1024 * 1024,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    base: PathBuf,
    output_dir: PathBuf,
    rows: u32,
    options: SpFreshBuildOptions,
}

fn array(start: u64, rows: usize, dimension: u32, values: Vec<f32>) -> VortexResult<ArrayRef> {
    let vectors = FixedSizeListArray::try_new(
        PrimitiveArray::new(Buffer::from(values), Validity::NonNullable).into_array(),
        dimension,
        Validity::NonNullable,
        rows,
    )?;
    let labels = (start..start + rows as u64)
        .map(|id| format!("row-{id}"))
        .collect::<Vec<_>>();
    Ok(StructArray::try_from_iter([
        (
            "id",
            PrimitiveArray::from_iter(start..start + rows as u64).into_array(),
        ),
        ("embedding", vectors.into_array()),
        (
            "label",
            VarBinArray::from_strs(labels.iter().map(String::as_str).collect()).into_array(),
        ),
    ])?
    .into_array())
}

fn read_vectors(reader: &mut impl Read, rows: usize, dimension: u32) -> VortexResult<Vec<f32>> {
    let mut record = vec![0u8; dimension as usize * 4];
    let mut values = Vec::with_capacity(rows * dimension as usize);
    for _ in 0..rows {
        let mut header = [0; 4];
        reader.read_exact(&mut header)?;
        if u32::from_le_bytes(header) != dimension {
            vortex_bail!("fvecs record dimension changed");
        }
        reader.read_exact(&mut record)?;
        for component in record.chunks_exact(4) {
            let value = f32::from_le_bytes(
                component
                    .try_into()
                    .map_err(|err| vortex_err!("fvecs component: {err}"))?,
            );
            if !value.is_finite() {
                vortex_bail!("Non-finite fvecs component");
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn open_vectors(path: &Path, rows: u32, dimension: u32) -> VortexResult<fs::File> {
    let file = unix_fs::open(
        path,
        unix_fs::OFlags::RDONLY
            | unix_fs::OFlags::NONBLOCK
            | unix_fs::OFlags::NOCTTY
            | unix_fs::OFlags::CLOEXEC,
        unix_fs::Mode::empty(),
    )
    .map_err(IoError::from)?;
    let file = fs::File::from(file);
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != u64::from(rows) * (4 + u64::from(dimension) * 4) {
        vortex_bail!("Fixture requires the complete regular fvecs file, not a subset");
    }
    Ok(file)
}

#[derive(Serialize)]
struct Reference {
    format_version: u32,
    snapshot: Snapshot,
    dtype: DType,
    generation: vortex_index::store::LocalGeneration,
}

fn main() -> VortexResult<()> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let [config_path] = args.as_slice() else {
        vortex_bail!("Usage: bench_sift_fixture CONFIG.json");
    };
    let config: Config = serde_json::from_slice(&fs::read(config_path)?)
        .map_err(|err| vortex_err!("Fixture config: {err}"))?;
    let dimension = config.options.dimension;
    if !config.base.is_absolute()
        || !config.output_dir.is_absolute()
        || !(64..=1_000_000).contains(&config.rows)
        || !(1..=4096).contains(&dimension)
        || u64::from(config.rows) * u64::from(dimension) * 4 > INPUT_LIMIT
        || config.options.format_version != 1
        || !(32..config.rows).contains(&config.options.head_count)
        || !(1..=4096).contains(&config.options.posting_page_limit)
        || !(1..=8).contains(&config.options.replicas)
    {
        vortex_bail!("Invalid bounded fixture configuration");
    }
    let file = open_vectors(&config.base, config.rows, dimension)?;
    fs::DirBuilder::new()
        .mode(0o700)
        .create(&config.output_dir)?;
    block_on(|handle| async move {
        let session = VortexSession::default().with_handle(handle);
        let reference_path = config.output_dir.join("index.json");
        let dtype = array(0, 0, dimension, vec![])?.dtype().clone();
        let mut reader = BufReader::new(file);
        let mut inventory = Vec::new();
        let mut start = 0u64;
        for (index, count) in [config.rows / 2, config.rows - config.rows / 2]
            .into_iter()
            .enumerate()
        {
            let mut bytes = ByteBufferMut::empty();
            let mut writer = session.write_options().writer(&mut bytes, dtype.clone());
            let mut remaining = count as usize;
            while remaining > 0 {
                let count = remaining.min(16_384);
                let values = read_vectors(&mut reader, count, dimension)?;
                writer.push(array(start, count, dimension, values)?).await?;
                start += count as u64;
                remaining -= count;
            }
            writer.finish().await?;
            let path = config.output_dir.join(format!("part-{index}.vortex"));
            let mut output = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            output.write_all(&bytes)?;
            output.sync_all()?;
            inventory.push(SourceFile {
                id: index as u64 + 1,
                uri: path.to_string_lossy().into_owned(),
                version: file_version(&bytes),
                row_count: u64::from(count),
            });
        }
        if reader.read(&mut [0u8; 1])? != 0 {
            vortex_bail!("Unexpected trailing fvecs data");
        }
        let snapshot = Snapshot {
            dataset_id: reference_path.to_string_lossy().into_owned(),
            version: file_version(
                &serde_json::to_vec(&inventory).map_err(|err| vortex_err!("{err}"))?,
            ),
            schema_fingerprint: schema_fingerprint(&dtype)?,
            files: inventory,
        };
        let source = Arc::new(
            LocalFileSource::open(
                snapshot.clone(),
                dtype.clone(),
                SOURCE_LIMIT,
                session.clone(),
            )
            .await?,
        );
        let scratch = tempfile::Builder::new()
            .prefix("fixture-scratch-")
            .permissions(fs::Permissions::from_mode(0o700))
            .tempdir_in(&config.output_dir)?;
        let builder = SpFreshIndexBuilder::try_new(
            scratch.path().to_owned(),
            session.clone(),
            SpFreshBuildLimits {
                max_vector_bytes: INPUT_LIMIT,
                ..SpFreshBuildLimits::default()
            },
        )?;
        let store = Arc::new(LocalIndexStore::create(
            &config.output_dir,
            "generation-1",
            STORE_LIMITS,
        )?);
        let start = Instant::now();
        let metadata = builder
            .build(
                IndexBuildRequest {
                    metadata: IndexMetadata {
                        format_version: 1,
                        name: "embedding".into(),
                        generation: "generation-1".into(),
                        backend: SPFRESH_ID.into(),
                        backend_version: 1,
                        snapshot: snapshot.clone(),
                        fields: vec!["embedding".into()],
                        covered_files: snapshot.files.iter().map(|file| file.id).collect(),
                        artifacts: vec![],
                    },
                    backend_options: Bytes::from(
                        serde_json::to_vec(&config.options).map_err(|err| vortex_err!("{err}"))?,
                    ),
                },
                source,
                Arc::clone(&store) as Arc<dyn IndexStore>,
            )
            .await?;
        let generation = store.seal(&metadata)?;
        let (reopened, metadata) =
            LocalIndexStore::open(&config.output_dir, &generation, &snapshot, STORE_LIMITS)?;
        let provider =
            SpFreshProvider::try_new(scratch.path().to_owned(), SpFreshLimits::default())?;
        drop(provider.open(&metadata, Arc::new(reopened)).await?);
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        drop(LocalFileSource::open(snapshot.clone(), dtype.clone(), SOURCE_LIMIT, session).await?);
        let reference = Reference {
            format_version: 1,
            snapshot,
            dtype,
            generation,
        };
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&reference_path)?;
        output.write_all(&serde_json::to_vec(&reference).map_err(|err| vortex_err!("{err}"))?)?;
        output.sync_all()?;
        fs::write(config.output_dir.join("build.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "format_version": 1, "rows": config.rows, "dimension": dimension,
            "build_ms": build_ms, "max_vector_bytes": INPUT_LIMIT,
            "max_native_artifact_bytes": SpFreshBuildLimits::default().max_native_artifact_bytes,
            "options": config.options, "reference": reference_path,
        })).map_err(|err| vortex_err!("{err}"))?)?;
        println!(
            "Built {} rows x {} in {:.2} ms: {}",
            config.rows,
            dimension,
            build_ms,
            reference_path.display()
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_fvecs_preserves_values_and_rejects_changed_dimensions() -> VortexResult<()> {
        let mut bytes = Vec::new();
        for values in [[1.0f32, 2.0], [3.0, 4.0]] {
            bytes.extend_from_slice(&2u32.to_le_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        assert_eq!(
            read_vectors(&mut Cursor::new(&bytes), 2, 2)?,
            [1.0, 2.0, 3.0, 4.0]
        );
        bytes[12..16].copy_from_slice(&3u32.to_le_bytes());
        assert!(read_vectors(&mut Cursor::new(&bytes), 2, 2).is_err());
        assert!(read_vectors(&mut Cursor::new(&bytes[..11]), 1, 2).is_err());
        Ok(())
    }

    #[test]
    fn test_fvecs_rejects_non_finite_components() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bytes = 1u32.to_le_bytes().to_vec();
            bytes.extend_from_slice(&value.to_le_bytes());
            assert!(read_vectors(&mut Cursor::new(bytes), 1, 1).is_err());
        }
    }

    #[test]
    fn test_input_open_requires_a_complete_regular_file() -> VortexResult<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("base.fvecs");
        let bytes = [1u32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        fs::write(&path, &bytes)?;
        let mut file = open_vectors(&path, 1, 1)?;
        assert_eq!(read_vectors(&mut file, 1, 1)?, [2.0]);
        assert!(open_vectors(&path, 2, 1).is_err());
        assert!(open_vectors(root.path(), 1, 1).is_err());
        Ok(())
    }

    #[test]
    fn test_input_open_rejects_a_fifo_without_waiting_for_a_writer() -> VortexResult<()> {
        let root = tempfile::tempdir()?;
        let path = root.path().join("base.fvecs");
        unix_fs::mkfifoat(
            unix_fs::CWD,
            &path,
            unix_fs::Mode::RUSR | unix_fs::Mode::WUSR,
        )
        .map_err(IoError::from)?;
        assert!(open_vectors(&path, 1, 1).is_err());
        Ok(())
    }
}
