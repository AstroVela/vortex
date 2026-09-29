// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "native")]
use std::env;
#[cfg(feature = "native")]
use std::error::Error;
#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::PathBuf;
#[cfg(feature = "native")]
use std::process::Command;

#[cfg(feature = "native")]
use sha2::Digest;
#[cfg(feature = "native")]
use sha2::Sha256;

#[cfg(not(feature = "native"))]
fn main() {}

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=VORTEX_SPFRESH_NATIVE");
    println!("cargo:rerun-if-changed=native");
    if env::var("CARGO_CFG_TARGET_OS")? != "linux"
        || env::var("CARGO_CFG_TARGET_ARCH")? != "x86_64"
        || env::var("HOST")? != env::var("TARGET")?
    {
        return Err("SPFresh native currently requires a native Linux x86_64 build".into());
    }
    let build = PathBuf::from(env::var("VORTEX_SPFRESH_NATIVE").map_err(|_| {
        "run bash vortex-index-spfresh/native/build.sh, then set VORTEX_SPFRESH_NATIVE to its build directory"
    })?);
    let revision = fs::read_to_string(build.join("revision.txt"))?;
    if revision.trim() != "5893eb61ee3b18610b6b00f1939be7dae1af8904" {
        return Err("SPFresh source revision mismatch; rebuild native libraries".into());
    }
    let patch_hash =
        base16ct::lower::encode_string(&Sha256::digest(fs::read("native/static-only.patch")?));
    if fs::read_to_string(build.join("patch-sha256.txt"))?.trim() != patch_hash {
        return Err("SPFresh patch mismatch; rebuild native libraries".into());
    }
    let source = fs::read_to_string(build.join("source.txt"))?;
    for (args, expected) in [
        (
            vec!["rev-parse", "HEAD"],
            revision.trim().as_bytes().to_vec(),
        ),
        (
            vec!["diff", "HEAD", "--binary"],
            fs::read("native/static-only.patch")?,
        ),
        (vec!["ls-files", "--others"], Vec::new()),
    ] {
        let output = Command::new("git")
            .arg("-C")
            .arg(source.trim())
            .args(args)
            .output()?;
        if !output.status.success() || output.stdout.trim_ascii() != expected.trim_ascii() {
            return Err(
                "SPFresh source changed since qualification; use a clean pinned build".into(),
            );
        }
    }
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(source.trim()).join("AnnService").display()
    );
    for name in [
        "revision.txt",
        "patch-sha256.txt",
        "source.txt",
        "zstd-library.txt",
        "libspfresh_core.a",
        "libspfresh_distance.a",
    ] {
        println!("cargo:rerun-if-changed={}", build.join(name).display());
    }
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .pic(true)
        .warnings(false)
        .define("SPTAG_STATIC_ONLY", None)
        .define("SPTAG_DISABLE_NUMA", None)
        .include(PathBuf::from(source.trim()).join("AnnService"))
        .file("native/bridge.cpp")
        .flag("-fopenmp")
        .compile("vortex_spfresh_bridge");
    println!("cargo:rustc-link-search=native={}", build.display());
    let zstd = PathBuf::from(fs::read_to_string(build.join("zstd-library.txt"))?.trim());
    let zstd_dir = zstd.parent().ok_or("Invalid qualified zstd library path")?;
    println!("cargo:rustc-link-search=native={}", zstd_dir.display());
    for library in ["spfresh_core", "spfresh_distance"] {
        println!("cargo:rustc-link-lib=static={library}");
    }
    for library in ["stdc++", "gomp", "zstd", "rt"] {
        println!("cargo:rustc-link-lib={library}");
    }
    println!(
        "cargo:rustc-env=VORTEX_SPFRESH_FIXTURE={}",
        build.join("spfresh_fixture").display()
    );
    Ok(())
}
