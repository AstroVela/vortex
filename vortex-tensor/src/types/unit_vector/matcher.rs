// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDTypeRef;
use vortex_array::dtype::extension::Matcher;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::types::unit_vector::UnitVector;
use crate::types::vector::VectorMatcherMetadata;

/// Matches exactly the [`UnitVector`] extension type.
pub struct AnyUnitVector;

impl Matcher for AnyUnitVector {
    type Match<'a> = VectorMatcherMetadata;

    fn try_match<'a>(ext_dtype: &'a ExtDTypeRef) -> Option<Self::Match<'a>> {
        if !ext_dtype.is::<UnitVector>() {
            return None;
        }

        let DType::FixedSizeList(element_dtype, dimensions, _) = ext_dtype.storage_dtype() else {
            vortex_panic!("UnitVector dtype must have FixedSizeList storage")
        };
        let (PType::F16 | PType::F32 | PType::F64) = element_dtype.as_ptype() else {
            vortex_panic!("UnitVector dtype must have float elements")
        };

        Some(
            VectorMatcherMetadata::try_new(element_dtype.as_ptype(), *dimensions)
                .vortex_expect("UnitVector dtype validation established float elements"),
        )
    }
}
