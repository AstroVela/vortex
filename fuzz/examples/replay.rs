// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Replay a saved `array_ops` fuzz input outside of `cargo-fuzz`.
//!
//! Mirrors what `libfuzzer-sys` does for a typed `fuzz_target!`: build an `Unstructured` over the
//! raw crash bytes and decode the action with `arbitrary_take_rest`.

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_error::vortex_err;
use vortex_fuzz::FuzzArrayAction;
use vortex_fuzz::error::Backtrace;
use vortex_fuzz::error::VortexFuzzError;
use vortex_fuzz::run_fuzz_action;

/// Boxed so the large fuzz error enum does not bloat the `Result` returned from `main`.
fn setup_error(message: String) -> Box<VortexFuzzError> {
    Box::new(VortexFuzzError::VortexError(
        vortex_err!("{}", message),
        Backtrace::capture(),
    ))
}

fn main() -> Result<(), Box<VortexFuzzError>> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| setup_error("usage: replay <path/to/artifact>".to_string()))?;
    let data =
        std::fs::read(&path).map_err(|e| setup_error(format!("failed to read {path}: {e}")))?;

    let Ok(action) = FuzzArrayAction::arbitrary_take_rest(Unstructured::new(&data)) else {
        println!("input does not decode into a FuzzArrayAction");
        return Ok(());
    };

    let kept = run_fuzz_action(action).map_err(Box::new)?;
    println!("replayed without error (corpus kept: {kept})");
    Ok(())
}
