// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayVTable;
use vortex_array::arrays::Filter;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_session::VortexSession;

use crate::Zstd;

/// Registers the parent execution kernels the [`Zstd`] encoding implements.
pub fn register_kernels(session: &VortexSession) {
    session
        .kernels()
        .register_execute_parent_kernel(Filter.id(), Zstd, FilterExecuteAdaptor(Zstd));
}
