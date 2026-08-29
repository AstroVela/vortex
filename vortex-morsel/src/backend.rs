// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stable selection boundary between independently maintained scan readers.

use std::str::FromStr;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// Scan implementation selected by SQL and benchmark integrations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ScanBackend {
    /// The original asynchronous `LayoutReader` implementation.
    V1,
    /// The optimized recursive pull-morsel crate.
    #[default]
    Pull,
    /// The independently imported physical push-morsel crate.
    Push,
}

impl FromStr for ScanBackend {
    type Err = vortex_error::VortexError;

    fn from_str(value: &str) -> VortexResult<Self> {
        match value {
            "v1" => Ok(Self::V1),
            "pull" | "morsel-pull" => Ok(Self::Pull),
            "push" | "morsel-push" => Ok(Self::Push),
            _ => vortex_bail!("scan backend must be v1, pull, or push; received {value:?}"),
        }
    }
}

/// Read [`ScanBackend`] from `VORTEX_SCAN_BACKEND`, defaulting to pull morsels.
pub fn scan_backend_from_env() -> VortexResult<ScanBackend> {
    match std::env::var("VORTEX_SCAN_BACKEND") {
        Ok(value) => value.parse(),
        Err(std::env::VarError::NotPresent) => Ok(ScanBackend::default()),
        Err(err) => vortex_bail!("VORTEX_SCAN_BACKEND is not valid Unicode: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use vortex_error::VortexResult;

    use super::ScanBackend;

    #[test]
    fn parses_stable_backend_labels() -> VortexResult<()> {
        assert_eq!(ScanBackend::from_str("v1")?, ScanBackend::V1);
        assert_eq!(ScanBackend::from_str("pull")?, ScanBackend::Pull);
        assert_eq!(ScanBackend::from_str("push")?, ScanBackend::Push);
        assert!(ScanBackend::from_str("other").is_err());
        Ok(())
    }
}
