// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::scalar_fn::fns::mask::MaskReduceAdaptor;

use crate::TiledFixedSizeList;

pub(crate) static RULES: ParentRuleSet<TiledFixedSizeList> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&MaskReduceAdaptor(TiledFixedSizeList)),
    ParentRuleSet::lift(&SliceReduceAdaptor(TiledFixedSizeList)),
]);
