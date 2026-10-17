// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::error::VortexResult;
    use vortex::error::vortex_err;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::session::VortexSession;

    #[tokio::test]
    async fn test_cli_build_search_inspect_and_failure() -> VortexResult<()> {
        let session = VortexSession::default();
        let temporary = tempfile::tempdir()?;
        let source = temporary.path().join("source.vortex");
        let index = temporary.path().join("index");
        let array = StructArray::from_fields(&[(
            "body",
            VarBinViewArray::from_iter_str(["full text search", "columnar storage"]).into_array(),
        )])?;
        let mut bytes = Vec::new();
        session
            .write_options()
            .write(&mut bytes, array.into_array().to_array_stream())
            .await?;
        fs::write(&source, bytes)?;
        let build = Command::new(env!("CARGO_BIN_EXE_vortex-fts"))
            .arg("build")
            .arg("--source")
            .arg(&source)
            .arg("--index")
            .arg(&index)
            .args(["--columns", "body"])
            .output()?;
        assert!(
            build.status.success(),
            "{}",
            String::from_utf8_lossy(&build.stderr)
        );
        let search = Command::new(env!("CARGO_BIN_EXE_vortex-fts"))
            .arg("search")
            .arg("--source")
            .arg(&source)
            .arg("--index")
            .arg(&index)
            .args(["--query", "\"full text\"", "--limit", "1"])
            .output()?;
        assert!(
            search.status.success(),
            "{}",
            String::from_utf8_lossy(&search.stderr)
        );
        let result: serde_json::Value =
            serde_json::from_slice(&search.stdout).map_err(|err| vortex_err!(External: err))?;
        assert_eq!(result["total_hits"], 1);
        assert_eq!(result["hits"][0]["row_id"], 0);
        let inspect = Command::new(env!("CARGO_BIN_EXE_vortex-fts"))
            .arg("inspect")
            .arg("--index")
            .arg(&index)
            .output()?;
        assert!(inspect.status.success());
        let metadata: serde_json::Value =
            serde_json::from_slice(&inspect.stdout).map_err(|err| vortex_err!(External: err))?;
        assert_eq!(metadata["row_count"], 2);
        let failed = Command::new(env!("CARGO_BIN_EXE_vortex-fts"))
            .arg("search")
            .arg("--source")
            .arg(&source)
            .arg("--index")
            .arg(&index)
            .args(["--query", "\"unterminated"])
            .output()?;
        assert!(!failed.status.success());
        assert!(!failed.stderr.is_empty());
        assert!(failed.stdout.is_empty());
        Ok(())
    }
}
