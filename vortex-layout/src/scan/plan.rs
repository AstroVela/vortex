// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Heap-allocated physical scan planning over the established layout-reader execution API.

use std::sync::Arc;

use vortex_array::expr::Expression;
use vortex_array::expr::forms::conjuncts;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::LayoutReaderRef;
use crate::LayoutReaderScanPlan;
use crate::ScanPlanRef;

/// Environment variable selecting the scan planning implementation.
pub const SCAN_IMPL_ENV: &str = "VORTEX_SCAN_IMPL";

/// A request-level physical plan prepared once before split execution.
pub struct PreparedScanPlan {
    projection: ScanPlanRef,
    predicates: Vec<ScanPlanRef>,
}

/// Shared handle to a prepared request-level physical scan plan.
pub type PreparedScanPlanRef = Arc<PreparedScanPlan>;

impl PreparedScanPlan {
    /// Bind a projection and each filter conjunct to heap-allocated physical plan nodes.
    pub fn try_new(
        reader: LayoutReaderRef,
        projection: Expression,
        filter: Option<&Expression>,
    ) -> VortexResult<Self> {
        let projection = Arc::new(LayoutReaderScanPlan::try_new(
            Arc::clone(&reader),
            projection,
        )?) as ScanPlanRef;
        let predicates = filter
            .map(conjuncts)
            .unwrap_or_default()
            .into_iter()
            .map(|expr| {
                Ok(
                    Arc::new(LayoutReaderScanPlan::try_new(Arc::clone(&reader), expr)?)
                        as ScanPlanRef,
                )
            })
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(Self {
            projection,
            predicates,
        })
    }

    /// Returns the bound projection plan.
    pub fn projection(&self) -> &ScanPlanRef {
        &self.projection
    }

    /// Returns one bound plan per filter conjunct.
    pub fn predicates(&self) -> &[ScanPlanRef] {
        &self.predicates
    }
}

/// Returns whether heap-allocated planning is enabled for this process.
///
/// The existing V1 path remains the default on this extraction branch. Set
/// `VORTEX_SCAN_IMPL=v2` to exercise the planning path with the same execution implementation.
pub fn planned_scan_enabled() -> VortexResult<bool> {
    match std::env::var(SCAN_IMPL_ENV) {
        Ok(value) => parse_scan_impl(&value),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(std::env::VarError::NotUnicode(value)) => {
            vortex_bail!("{SCAN_IMPL_ENV} must be valid unicode, got {value:?}")
        }
    }
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
