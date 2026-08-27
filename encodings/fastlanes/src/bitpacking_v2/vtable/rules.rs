// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::BitPackedV2;

pub(crate) const RULES: ParentRuleSet<BitPackedV2> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(BitPackedV2))]);
