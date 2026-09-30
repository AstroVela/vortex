// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex_duckdb.h"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/relation.hpp"

// Linked only by Rust tests that need the relation API absent from DuckDB's C API.
extern "C" duckdb_vx_error vortex_index_test_bind_relation(duckdb_connection connection, const char *sql) {
    try {
        auto relation = reinterpret_cast<duckdb::Connection *>(connection)->RelationFromQuery(sql);
        return nullptr;
    } catch (const std::exception &error) {
        auto message = duckdb::string(error.what());
        return duckdb_vx_error_create(message.data(), message.size());
    }
}
