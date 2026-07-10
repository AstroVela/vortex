// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::arrays::TakeSlicesArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::take_slices::TakeSlicesReduce;

impl TakeSlicesReduce for Extension {
    fn take_slices(
        array: ArrayView<'_, Extension>,
        starts: &ArrayRef,
        lengths: &ArrayRef,
        output_len: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let storage = TakeSlicesArray::try_new(
            array.storage_array().clone(),
            starts.clone(),
            lengths.clone(),
            output_len,
        )?
        .into_array();

        Ok(Some(
            ExtensionArray::new(array.ext_dtype().clone(), storage).into_array(),
        ))
    }
}
