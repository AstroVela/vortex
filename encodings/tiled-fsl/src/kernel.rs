// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::builders::builder_with_capacity;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::TiledFixedSizeList;
use crate::TiledFixedSizeListArrayExt;
use crate::TiledFixedSizeListArraySlotsExt;
use crate::gather::plan_physical_row_tile_spans;
use crate::geometry::geometry_usizes;
use crate::transpose::decode_visible_elements;

pub(crate) fn initialize(session: &VortexSession) {
    session.kernels().register_execute_parent_kernel(
        Slice.id(),
        TiledFixedSizeList,
        SliceExecuteAdaptor(TiledFixedSizeList),
    );
}

impl SliceKernel for TiledFixedSizeList {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // Span planning below uses geometry relative to a complete, zero-offset backing array.
        vortex_ensure!(
            array.row_offset() == 0 && array.backing_rows() == array.len(),
            InvalidArgument: "tiled fixed-size-list slice kernel requires a non-view array"
        );
        let (tile_rows, _) = geometry_usizes(array.geometry())?;
        let retained_start = range.start / tile_rows * tile_rows;
        let retained_end = range.end.div_ceil(tile_rows) * tile_rows;
        let retained_end = retained_end.min(array.len());
        let retained_range = retained_start..retained_end;
        let retained_rows = retained_range.len();
        let retained_element_count = retained_rows * array.list_size() as usize;
        let mut elements = builder_with_capacity(array.elements().dtype(), retained_element_count);
        for span in plan_physical_row_tile_spans(array, retained_range)? {
            let span_elements = array
                .elements()
                .slice(span)?
                .execute::<PrimitiveArray>(ctx)?;
            span_elements.append_to_builder(elements.as_mut(), ctx)?;
        }
        let elements = elements.finish();
        let decoded = decode_visible_elements(
            elements.as_::<Primitive>(),
            range.len(),
            array.list_size() as usize,
            array.geometry(),
            range.start - retained_start,
            retained_rows,
            ctx,
        )?;

        Ok(Some(
            FixedSizeListArray::new(
                decoded.into_array(),
                array.list_size(),
                array.array_validity().slice(range.clone())?,
                range.len(),
            )
            .into_array(),
        ))
    }
}
