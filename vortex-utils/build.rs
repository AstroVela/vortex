// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Detects whether the compiling toolchain has stable algebraic float operations.
//!
//! `f32::algebraic_add` and friends stabilized in Rust 1.98. The workspace MSRV is 1.95, so
//! [`AlgebraicFloat`] falls back to ordinary IEEE operations on older toolchains and this script
//! decides which of the two impl blocks compiles.
//!
//! [`AlgebraicFloat`]: vortex_utils::algebraic::AlgebraicFloat

use std::env;
use std::process::Command;

/// First stable release of the `float_algebraic` library feature.
const ALGEBRAIC_FLOAT_MINOR: u32 = 98;

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-check-cfg=cfg(vortex_float_algebraic)");

    if rustc_minor_version().is_some_and(|minor| minor >= ALGEBRAIC_FLOAT_MINOR) {
        println!("cargo:rustc-cfg=vortex_float_algebraic");
    }
}

/// Returns the minor version of the `rustc` that Cargo invoked us with.
///
/// Returns `None` if `rustc` cannot be run or its `-vV` output cannot be parsed, in which case the
/// caller conservatively assumes the feature is unavailable.
fn rustc_minor_version() -> Option<u32> {
    let rustc = env::var_os("RUSTC")?;
    let output = Command::new(rustc).arg("-vV").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;

    // `release: 1.98.0` for stable, `release: 1.99.0-nightly` for nightly.
    let release = stdout
        .lines()
        .find_map(|line| line.strip_prefix("release:"))?
        .trim();
    let mut components = release.split('.');
    let major: u32 = components.next()?.parse().ok()?;
    let minor: u32 = components.next()?.parse().ok()?;

    (major == 1).then_some(minor)
}
