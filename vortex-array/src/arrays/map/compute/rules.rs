// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Map;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::map::MapArrayExt;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const PARENT_RULES: ParentRuleSet<Map> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&MapGetItemRule),
    ParentRuleSet::lift(&FilterReduceAdaptor(Map)),
    ParentRuleSet::lift(&CastReduceAdaptor(Map)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Map)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Map)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Map)),
]);

const MAP_ENTRIES_FIELD: &str = "entries";

#[derive(Debug)]
struct MapGetItemRule;

impl ArrayParentReduceRule<Map> for MapGetItemRule {
    type Parent = ExactScalarFn<GetItem>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Map>,
        parent: ScalarFnArrayView<'_, GetItem>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if parent.options != MAP_ENTRIES_FIELD {
            return Ok(None);
        }

        Ok(Some(array.entries().into_owned().into_array()))
    }
}
