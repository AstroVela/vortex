// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "data.hpp"
#include "error.hpp"
#include "table_function.hpp"
#include "expr.h"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"

using namespace duckdb;

namespace {

void CheckAccess(ClientContext &context) {
    Value enabled;
    if (!context.TryGetCurrentSetting("enable_external_access", enabled) || !enabled.GetValue<bool>()) {
        throw PermissionException("Vortex indexes require external access");
    }
}

void CheckError(duckdb_vx_error error) {
    if (error) {
        throw InvalidInputException(IntoErrString(error));
    }
}

struct IndexBind final : FunctionData {
    explicit IndexBind(void *request) : request(request) {
    }
    ~IndexBind() override {
        vortex_index_bind_free(request);
    }
    unique_ptr<FunctionData> Copy() const override {
        return make_uniq<IndexBind>(vortex_index_bind_copy(request));
    }
    bool Equals(const FunctionData &other) const override {
        return this == &other;
    }
    void *request;
};

struct IndexState final : GlobalTableFunctionState {
    explicit IndexState(void *exporter) : exporter(exporter) {
    }
    ~IndexState() override {
        vortex_index_state_free(exporter);
    }
    idx_t MaxThreads() const override {
        return 1;
    }
    void *exporter;
};

unique_ptr<FunctionData> BindIndex(ClientContext &context,
                                   TableFunctionBindInput &input,
                                   vector<LogicalType> &types,
                                   vector<string> &names) {
    CheckAccess(context);
    const bool build = input.table_function.name == "vortex_index_build";
    auto values = input.inputs;
    if (!build) {
        auto options = input.named_parameters.find("backend_options");
        values.push_back(options == input.named_parameters.end() ? Value("") : options->second);
    }
    vector<duckdb_value> pointers;
    for (auto &value : values) {
        pointers.push_back(reinterpret_cast<duckdb_value>(&value));
    }
    duckdb_logical_type result_type = nullptr;
    duckdb_vx_error error = nullptr;
    auto request = vortex_index_bind(build, pointers.data(), pointers.size(), &result_type, &error);
    CheckError(error);
    auto bind = make_uniq<IndexBind>(request);
    unique_ptr<LogicalType> result(reinterpret_cast<LogicalType *>(result_type));
    for (auto &child : StructType::GetChildTypes(*result)) {
        names.push_back(child.first);
        types.push_back(child.second);
    }
    return bind;
}

unique_ptr<GlobalTableFunctionState> InitIndex(ClientContext &context, TableFunctionInitInput &input) {
    CheckAccess(context);
    auto &bind = input.bind_data->Cast<IndexBind>();
    duckdb_vx_error error = nullptr;
    auto exporter = vortex_index_execute(bind.request, &error);
    CheckError(error);
    return make_uniq<IndexState>(exporter);
}

void ScanIndex(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
    CheckAccess(context);
    auto &state = input.global_state->Cast<IndexState>();
    duckdb_vx_error error = nullptr;
    vortex_index_scan(state.exporter, reinterpret_cast<duckdb_data_chunk>(&output), &error);
    CheckError(error);
}

vector<TableFunction> Functions() {
    TableFunction build("vortex_index_build",
                        {LogicalType::LIST(LogicalType::VARCHAR),
                         LogicalType::VARCHAR,
                         LogicalType::VARCHAR,
                         LogicalType::VARCHAR,
                         LogicalType::VARCHAR},
                        ScanIndex,
                        BindIndex,
                        InitIndex);
    TableFunction search("vortex_index_search",
                         {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT), LogicalType::BIGINT},
                         ScanIndex,
                         BindIndex,
                         InitIndex);
    search.named_parameters["backend_options"] = LogicalType::VARCHAR;
    return {build, search};
}

} // namespace

void RegisterVortexIndexFunctions(DatabaseInstance &db) {
    auto &catalog = Catalog::GetSystemCatalog(db);
    auto transaction = CatalogTransaction::GetSystemTransaction(db);
    for (auto &function : Functions()) {
        CreateTableFunctionInfo info(function);
        catalog.CreateFunction(transaction, info);
    }
}

#ifdef VORTEX_VANE_DISTRIBUTED
void RegisterVortexIndexFunctions(ExtensionLoader &loader) {
    for (auto &function : Functions()) {
        loader.RegisterFunction(function);
    }
}
#endif
