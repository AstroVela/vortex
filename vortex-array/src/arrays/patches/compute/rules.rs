// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::Patches;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(crate) const PARENT_RULES: ParentRuleSet<Patches> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FilterReduceAdaptor(Patches)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Patches)),
]);
