// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

use serde::Serialize;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::FieldPath;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_file::OpenOptionsSessionExt;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::scan::split_by::SplitBy;
use vortex_layout::session::LayoutSession;

#[derive(Serialize)]
struct Catalog {
    files: Vec<CatalogFile>,
}

#[derive(Serialize)]
struct CatalogFile {
    path: String,
    row_count: u64,
    fields: BTreeMap<String, Vec<u64>>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> VortexResult<()> {
    let mut args = std::env::args_os().skip(1);
    let Some(output) = args.next().map(PathBuf::from) else {
        vortex_bail!("usage: natural_split_catalog OUTPUT.json FILE.vortex...");
    };
    let paths = args.map(PathBuf::from).collect::<Vec<_>>();
    if paths.is_empty() {
        vortex_bail!("at least one Vortex file is required");
    }
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
        .with_tokio();
    vortex_file::register_default_encodings(&session);
    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let file = session.open_options().open_path(&path).await?;
        let reader = file.layout_reader()?;
        let mut fields = BTreeMap::new();
        for name in file.dtype().as_struct_fields().names().iter() {
            let mask = FieldMask::Exact(FieldPath::from(Field::from(name.clone())));
            fields.insert(
                name.to_string(),
                SplitBy::natural_splits(reader.as_ref(), &(0..file.row_count()), &[mask])?,
            );
        }
        files.push(CatalogFile {
            path: path.display().to_string(),
            row_count: file.row_count(),
            fields,
        });
    }

    serde_json::to_writer(File::create(output)?, &Catalog { files })
        .map_err(|error| vortex_error::vortex_err!(External: error))?;
    Ok(())
}
