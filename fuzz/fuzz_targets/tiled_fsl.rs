// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![no_main]

use libfuzzer_sys::Corpus;
use libfuzzer_sys::fuzz_target;
use vortex_error::vortex_panic;
use vortex_fuzz::FuzzTiledFsl;
use vortex_fuzz::run_tiled_fsl;

fuzz_target!(|input: FuzzTiledFsl| -> Corpus {
    match run_tiled_fsl(input) {
        Ok(()) => Corpus::Keep,
        Err(error) => vortex_panic!("{error}"),
    }
});
