// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stable selection boundary between the V1, pull-morsel, and push-morsel readers.

use std::str::FromStr;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ExecutionMode;

/// Scan implementation selected by SQL and benchmark integrations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScanBackend {
    /// The original asynchronous `LayoutReader` implementation.
    V1,
    /// The morsel executor using the selected value-flow implementation.
    Morsel(ExecutionMode),
}

impl Default for ScanBackend {
    fn default() -> Self {
        Self::Morsel(ExecutionMode::Pull)
    }
}

impl FromStr for ScanBackend {
    type Err = vortex_error::VortexError;

    fn from_str(value: &str) -> VortexResult<Self> {
        match value {
            "v1" => Ok(Self::V1),
            "pull" | "morsel-pull" => Ok(Self::Morsel(ExecutionMode::Pull)),
            "push" | "morsel-push" => Ok(Self::Morsel(ExecutionMode::Push)),
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
    use crate::ExecutionMode;

    #[test]
    fn parses_stable_backend_labels() -> VortexResult<()> {
        assert_eq!(ScanBackend::from_str("v1")?, ScanBackend::V1);
        assert_eq!(
            ScanBackend::from_str("pull")?,
            ScanBackend::Morsel(ExecutionMode::Pull)
        );
        assert_eq!(
            ScanBackend::from_str("push")?,
            ScanBackend::Morsel(ExecutionMode::Push)
        );
        assert!(ScanBackend::from_str("other").is_err());
        Ok(())
    }
}
