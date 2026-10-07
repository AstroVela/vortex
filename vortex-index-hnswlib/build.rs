// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "native")]
use std::env;
#[cfg(feature = "native")]
use std::error::Error;
#[cfg(feature = "native")]
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::process::Command;
#[cfg(feature = "native")]
use std::str::from_utf8;

#[cfg(not(feature = "native"))]
fn main() {}

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=VORTEX_HNSWLIB_SOURCE");
    println!("cargo:rerun-if-changed=native");
    if env::var("CARGO_CFG_TARGET_OS")? != "linux"
        || env::var("CARGO_CFG_TARGET_ARCH")? != "x86_64"
        || env::var("HOST")? != env::var("TARGET")?
    {
        return Err("hnswlib native requires a native Linux x86_64 build".into());
    }
    let source = PathBuf::from(
        env::var("VORTEX_HNSWLIB_SOURCE")
            .map_err(|_| "set VORTEX_HNSWLIB_SOURCE to a clean hnswlib v0.9.0 checkout")?,
    )
    .canonicalize()?;
    for (arguments, expected) in [
        (
            vec!["rev-parse", "HEAD"],
            "d9b3608c83d83b46c96e25088cb1d729b29dcfe9",
        ),
        (vec!["status", "--porcelain", "--untracked-files=all"], ""),
    ] {
        let output = Command::new("git")
            .arg("-C")
            .arg(&source)
            .args(arguments)
            .output()?;
        if !output.status.success() || from_utf8(&output.stdout)?.trim() != expected {
            return Err("hnswlib source must be the clean pinned v0.9.0 revision".into());
        }
    }
    println!(
        "cargo:rerun-if-changed={}",
        source.join("hnswlib").display()
    );
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .pic(true)
        .warnings(false)
        .include(&source)
        .file("native/bridge.cpp")
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mf16c")
        .flag("-fopenmp")
        .compile("vortex_hnswlib_bridge");
    println!("cargo:rustc-link-lib=gomp");
    Ok(())
}
