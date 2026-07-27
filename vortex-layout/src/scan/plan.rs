// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Heap-allocated physical scan planning over the established layout-reader execution API.
//!
//! # Design direction
//!
//! Planning should begin with a heap-allocated source plan produced by a layout. Applying an
//! expression to that source returns another plan, and optimizing that derived plan returns another
//! plan. Execution receives only a row range and mask; expressions do not cross the
//! planning/execution boundary.
//!
//! A scan retains its existing N+1 decomposition: one independently pushed and optimized plan for
//! each filter conjunct, plus one independently pushed and optimized projection plan. These plans
//! must remain separate so their reads can be scheduled concurrently. They are not combined into a
//! single filter-then-projection operator tree.
//!
//! The current [`LayoutReaderScanPlan`](crate::LayoutReaderScanPlan) is a compatibility source plan:
//! it supports the plan-to-plan API but delegates execution to the existing [`crate::LayoutReader`].
//! Layout-specific source plans can replace it incrementally.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// Environment variable selecting the scan planning implementation.
pub const SCAN_IMPL_ENV: &str = "VORTEX_SCAN_IMPL";

/// Returns whether heap-allocated planning is enabled for this process.
///
/// Heap-allocated planning is enabled by default. Set `VORTEX_SCAN_IMPL=v1` to use the legacy
/// expression-passing path.
pub fn planned_scan_enabled() -> VortexResult<bool> {
    match std::env::var(SCAN_IMPL_ENV) {
        Ok(value) => scan_impl_or_default(Some(&value)),
        Err(std::env::VarError::NotPresent) => scan_impl_or_default(None),
        Err(std::env::VarError::NotUnicode(value)) => {
            vortex_bail!("{SCAN_IMPL_ENV} must be valid unicode, got {value:?}")
        }
    }
}

fn scan_impl_or_default(value: Option<&str>) -> VortexResult<bool> {
    value.map_or(Ok(true), parse_scan_impl)
}

fn parse_scan_impl(value: &str) -> VortexResult<bool> {
    match value {
        "" | "v1" | "legacy" | "layout-reader" => Ok(false),
        "v2" | "planned" | "scan-plan" => Ok(true),
        other => vortex_bail!(
            "{SCAN_IMPL_ENV} must be one of v1, legacy, layout-reader, v2, planned, or scan-plan, got {other:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_impl_defaults_to_planned() -> VortexResult<()> {
        assert!(scan_impl_or_default(None)?);
        Ok(())
    }

    #[test]
    fn scan_impl_accepts_v1_and_v2_values() -> VortexResult<()> {
        for value in ["", "v1", "legacy", "layout-reader"] {
            assert!(!parse_scan_impl(value)?);
        }
        for value in ["v2", "planned", "scan-plan"] {
            assert!(parse_scan_impl(value)?);
        }
        Ok(())
    }

    #[test]
    fn scan_impl_rejects_unknown_value() {
        assert!(parse_scan_impl("unknown").is_err());
    }
}
