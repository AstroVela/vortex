// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::slice::SliceReduceAdaptor;
use vortex_array::optimizer::rules::ParentRuleSet;

use crate::FloatMult;
use crate::FloatQuant;

pub(crate) static RULES: ParentRuleSet<FloatQuant> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(FloatQuant))]);

pub(crate) static FLOAT_MULT_RULES: ParentRuleSet<FloatMult> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(FloatMult))]);
