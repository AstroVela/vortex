// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

extern "C" void vortex_fts_initialize(void *loader, void (*initialize)(void *)) {
    auto &db = static_cast<duckdb::ExtensionLoader *>(loader)->GetDatabaseInstance();
    duckdb::DatabaseWrapper wrapper;
    wrapper.database = duckdb::make_shared_ptr<duckdb::DuckDB>(db);
    // The C API borrows this wrapper throughout registration. Its DuckDB object
    // shares ownership of the existing instance rather than opening a new database.
    initialize(&wrapper);
}
