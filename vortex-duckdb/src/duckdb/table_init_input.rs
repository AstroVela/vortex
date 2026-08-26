// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Formatter;
use std::fmt::Result;

use crate::cpp;
use crate::duckdb::TableFilterSet;
use crate::duckdb::TableFilterSetRef;

pub struct TableInitInput<'a> {
    pub input: &'a cpp::duckdb_vx_tfunc_init_input,
}

impl Debug for TableInitInput<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_struct("TableInitInput")
            .field("column_ids", &self.column_ids())
            .field("table_filter_set", &self.table_filter_set())
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn new(input: &'a cpp::duckdb_vx_tfunc_init_input) -> Self {
        Self { input }
    }

    pub fn column_ids(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.input.column_ids, self.input.column_ids_count) }
    }

    /// Number of threads DuckDB will run for this scan.
    ///
    /// DuckDB sizes its task scheduler from `std::thread::hardware_concurrency`, which
    /// ignores the process CPU affinity mask, so this can differ from
    /// [`vortex_utils::parallelism::get_available_parallelism`] whenever the process is
    /// pinned (`taskset`, `numactl`, a container cpuset). Scan-side scheduling decisions
    /// need the number of threads that actually contend for splits, which is this one.
    pub fn num_threads(&self) -> usize {
        if self.input.client_context.is_null() {
            return 1;
        }
        let threads =
            unsafe { cpp::duckdb_vx_client_context_num_threads(self.input.client_context) };
        usize::try_from(threads).unwrap_or(1).max(1)
    }

    /// Returns the table filter set for the table function.
    pub fn table_filter_set(&self) -> Option<&TableFilterSetRef> {
        let ptr = self.input.filters;
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { TableFilterSet::borrow(ptr) })
        }
    }
}
