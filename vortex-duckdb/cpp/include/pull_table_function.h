// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#pragma once

#include "duckdb.h"
#include "duckdb/function/table_function.hpp"
#include "table_function.h"

// Pull-based (inverted-IO) scan path: DuckDB owns file listing, file handles, and every read;
// Vortex decodes via the vx_pull_scan coroutine (see vortex-ffi and lang/cpp vortex/pull.hpp).
// The legacy path (Rust-side IO through object_store) remains the fallback for pushdown
// features the pull path does not cover, and can be forced with VORTEX_DUCKDB_PULL=0.
namespace vortex_pull {

// Build the pull-based global state, or nullptr when the pull path does not apply and the
// caller must build the legacy state. "ffi_input" is the same init input handed to the legacy
// Rust init_global.
duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
try_init_global(duckdb::ClientContext &context, duckdb::TableFunctionInitInput &input,
                const duckdb::vector<duckdb::string> &patterns,
                const duckdb_vx_tfunc_init_input &ffi_input);

// Whether "state" was produced by try_init_global.
bool is_pull_state(const duckdb::GlobalTableFunctionState &state);

duckdb::unique_ptr<duckdb::LocalTableFunctionState> init_local(const duckdb_vx_tfunc_init_input &ffi_input);

void scan(duckdb::TableFunctionInput &input, duckdb::DataChunk &output);

double progress(const duckdb::GlobalTableFunctionState &state);

duckdb::OperatorPartitionData partition_data(duckdb::TableFunctionGetPartitionInput &input);

} // namespace vortex_pull
