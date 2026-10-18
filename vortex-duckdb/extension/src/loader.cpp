// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "duckdb/main/extension/extension_loader.hpp"

extern "C" void *vortex_fts_loader_database(void *loader) {
    return &static_cast<duckdb::ExtensionLoader *>(loader)->GetDatabaseInstance();
}
