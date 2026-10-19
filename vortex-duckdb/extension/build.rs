// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/loader.cpp");
    let include = env::var("DEP_DUCKDB_INCLUDE_DIR")?;
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .include(include)
        .file("src/loader.cpp")
        .compile("vortex_fts_loader");
    if let Ok(directory) = env::var("DEP_DUCKDB_LIB_DIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{directory}");
    }
    Ok(())
}
