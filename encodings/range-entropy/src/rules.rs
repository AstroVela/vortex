// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::RangeEntropy;

pub(crate) static RULES: ParentRuleSet<RangeEntropy> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(RangeEntropy))]);
