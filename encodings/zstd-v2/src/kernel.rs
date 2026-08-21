// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_session::VortexSession;

use crate::ZstdV2;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Filter.id(), ZstdV2, FilterExecuteAdaptor(ZstdV2));
    // Random access arrives as a dict parent: the codes are the rows being read.
    kernels.register_execute_parent_kernel(Dict.id(), ZstdV2, TakeExecuteAdaptor(ZstdV2));
}
