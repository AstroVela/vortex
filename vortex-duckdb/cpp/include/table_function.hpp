// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#pragma once

#include "duckdb.h"
#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"

using namespace duckdb;

static_assert(sizeof(idx_t) == 8);

bool is_vortex_scan(const TableFunction &function);

// We need this exposed to compare function addresses in optimizer.cpp
unique_ptr<FunctionData> duckdb_vx_table_function_bind(ClientContext &context,
                                                       TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types,
                                                       vector<string> &names);

struct TableFunctionUngroupedAggregateInput {
    const LogicalGet &get;
    // Column scan index -> aggregate expression
    const vector<std::pair<idx_t, const Expression &>> &projections;
};

bool aggregate_pushdown(ClientContext &context, const TableFunctionUngroupedAggregateInput &input);
