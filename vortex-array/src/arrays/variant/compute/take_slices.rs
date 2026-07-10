// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::TakeSlicesArray;
use crate::arrays::Variant;
use crate::arrays::VariantArray;
use crate::arrays::take_slices::TakeSlicesReduce;
use crate::arrays::variant::VariantArrayExt;

impl TakeSlicesReduce for Variant {
    fn take_slices(
        array: ArrayView<'_, Variant>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let core_storage = TakeSlicesArray::try_new(
            array.core_storage().clone(),
            starts.clone(),
            lengths.clone(),
            output_len,
        )?
        .into_array();
        let shredded = array
            .shredded()
            .map(|shredded| {
                TakeSlicesArray::try_new(
                    shredded.clone(),
                    starts.clone(),
                    lengths.clone(),
                    output_len,
                )
                .map(IntoArray::into_array)
            })
            .transpose()?;

        Ok(Some(
            VariantArray::try_new(core_storage, shredded)?.into_array(),
        ))
    }
}
