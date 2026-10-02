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
#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/extension_callback.hpp"
#include "duckdb/planner/operator/logical_prepare.hpp"
#include "duckdb/planner/planner_extension.hpp"

using namespace duckdb;

namespace {

void CheckAccess(ClientContext &context) {
    Value enabled;
    if (!context.TryGetCurrentSetting("enable_external_access", enabled) || !enabled.GetValue<bool>()) {
        throw PermissionException("Vortex indexes require external access");
    }
    // Rust and native backends perform local I/O outside DuckDB's filesystem.
    if (FileSystem::GetFileSystem(context).SubSystemIsDisabled("LocalFileSystem")) {
        throw PermissionException("File system LocalFileSystem has been disabled by configuration");
    }
}

void CheckStrings(const Value &value) {
    if (value.IsNull()) {
        return;
    }
    if (value.type().id() == LogicalTypeId::VARCHAR) {
        // Validate the full VARCHAR before Rust's NUL-terminated C API extraction.
        if (StringValue::Get(value).find('\0') != string::npos) {
            throw InvalidInputException("Index string arguments must not contain NUL");
        }
    } else if (value.type().id() == LogicalTypeId::LIST) {
        for (auto &child : ListValue::GetChildren(value)) {
            CheckStrings(child);
        }
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

const char *const PIN_DEPENDENCY = "vortex_index_reference_pins";

struct CacheBudget final {
    CacheBudget() : budget(vortex_index_cache_budget_new()) {
    }
    ~CacheBudget() {
        vortex_index_cache_budget_free(budget);
    }
    void *budget;
};

template <class CALLBACK>
void VisitIndexRequests(const PhysicalOperator &op, CALLBACK &&callback) {
    if (op.type == PhysicalOperatorType::TABLE_SCAN) {
        auto &scan = op.Cast<PhysicalTableScan>();
        if (scan.function.name == "vortex_index_search") {
            callback(scan.bind_data->Cast<IndexBind>().request);
        }
    }
    for (auto &child : op.GetChildren()) {
        VisitIndexRequests(child.get(), callback);
    }
}

struct ReferencePins final {
    explicit ReferencePins(const CacheBudget &budget) : pins(vortex_index_pins_new(budget.budget)) {
    }
    ~ReferencePins() {
        vortex_index_pins_free(pins);
    }
    void Remember(const PhysicalOperator &op) {
        VisitIndexRequests(op, [this](const void *request) {
            duckdb_vx_error error = nullptr;
            const void *groups[] = {pins};
            vortex_index_pins_record(groups, 1, request, &error);
            CheckError(error);
        });
    }
    void *pins;
};

struct PreparedPinLifetime final {
    explicit PreparedPinLifetime(shared_ptr<ReferencePins> pins) : pins(std::move(pins)) {
    }
    shared_ptr<ReferencePins> pins;
};

struct PreparedPinOwner final : PhysicalOperator {
    PreparedPinOwner(PhysicalPlan &plan,
                     const vector<LogicalType> &types,
                     shared_ptr<PreparedPinLifetime> lifetime)
        : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, types, 0), lifetime(std::move(lifetime)) {
    }
    void BuildPipelines(Pipeline &, MetaPipeline &) override {
        throw InternalException("Vortex index reference owner is not an executable plan");
    }
    shared_ptr<PreparedPinLifetime> lifetime;
};

struct IndexPreparedState final : ClientContextState {
    bool CanRequestRebind() override {
        // Both supported engines call this before each prepare pass, but not
        // for RelationFromQuery's standalone schema bind.
        auto owner = std::move(execute_owner);
        EndPrepare();
        preparing = true;
        if (owner) {
            borrowed.push_back(std::move(owner));
        }
        return true;
    }
    void EndPrepare() {
        borrowed.clear();
        execute_owner.reset();
        pending.reset();
        sql_prepares.clear();
        sql_execute = false;
        preparing = false;
    }
    void PruneOwners() {
        for (auto it = owners.begin(); it != owners.end();) {
            if (it->second.expired()) {
                it = owners.erase(it);
            } else {
                ++it;
            }
        }
    }
    void Own(ClientContext &context, PreparedStatementData &prepared, shared_ptr<ReferencePins> pins) {
        // The original prepared handle owns its pins, independent of AST type.
        // Automatic rebinds execute temporary data without replacing this owner.
        bool needs_rebind = !prepared.physical_plan;
        if (needs_rebind) {
            prepared.physical_plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
            prepared.properties.always_require_rebind = true;
        }
        // Pins can be shared with an inner SQL owner. Cache validity must track
        // this handle's plan, not shared pins that survive its destruction.
        auto lifetime = make_shared_ptr<PreparedPinLifetime>(std::move(pins));
        auto &owner = prepared.physical_plan->Make<PreparedPinOwner>(prepared.types, lifetime);
        if (needs_rebind) {
            prepared.physical_plan->SetRoot(owner);
        }
        PruneOwners();
        owners[&prepared] = lifetime;
    }
    void CaptureSqlPrepares(LogicalOperator &op) {
        if (!preparing || !pending) {
            return;
        }
        if (op.type == LogicalOperatorType::LOGICAL_PREPARE) {
            // Physical planning can replace SQL PREPARE's inner physical plan.
            // The enclosing handle also owns the successful bind's identity.
            sql_prepares.push_back({op.Cast<LogicalPrepare>().prepared, pending});
        }
        for (auto &child : op.children) {
            CaptureSqlPrepares(*child);
        }
    }
    RebindQueryInfo OnFinalizePrepare(ClientContext &context,
                                      PreparedStatementData &prepared,
                                      PreparedStatementMode) override {
        bool rebind_execute = sql_execute && pending;
        auto captures = std::move(sql_prepares);
        auto pins = std::move(pending);
        EndPrepare();
        for (auto &capture : captures) {
            Own(context, *capture.prepared, std::move(capture.pins));
        }
        if (pins) {
            Own(context, prepared, std::move(pins));
        }
        if (rebind_execute) {
            // A cached EXECUTE can borrow the SQL owner's physical plan without
            // retaining it. Rebind the wrapper before using it after DEALLOCATE.
            prepared.properties.always_require_rebind = true;
        }
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    RebindQueryInfo OnPlanningError(ClientContext &, SQLStatement &, ErrorData &) override {
        EndPrepare();
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    shared_ptr<ReferencePins> PinsFor(ClientContext &context, PreparedStatementData &prepared) {
        PruneOwners();
        auto owner = owners.find(&prepared);
        if (owner != owners.end()) {
            if (auto lifetime = owner->second.lock()) {
                return lifetime->pins;
            }
        }
        auto pins = make_shared_ptr<ReferencePins>(cache_budget);
        if (prepared.physical_plan) {
            pins->Remember(prepared.physical_plan->Root());
        }
        Own(context, prepared, pins);
        return pins;
    }
    RebindQueryInfo
    OnExecutePrepared(ClientContext &context, PreparedStatementCallbackInfo &info, RebindQueryInfo) override {
        // C API execution invokes this before CreatePreparedStatement. Only
        // the next prepare pass may borrow this owner, never a standalone bind.
        execute_owner = PinsFor(context, info.prepared_statement);
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    RebindQueryInfo OnRebindPreparedStatement(ClientContext &context,
                                              BindPreparedStatementCallbackInfo &info,
                                              RebindQueryInfo) override {
        if (preparing) {
            // Keep both an outer C API wrapper and its inner SQL EXECUTE owner.
            auto owner = PinsFor(context, info.prepared_statement);
            InheritPins(*owner);
            borrowed.push_back(std::move(owner));
            sql_execute = true;
        }
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    void QueryEnd() override {
        EndPrepare();
        PruneOwners();
    }
    vector<const void *> PinGroups(const ReferencePins &capture) {
        vector<const void *> groups {capture.pins};
        for (auto &owner : borrowed) {
            groups.push_back(owner->pins);
        }
        return groups;
    }
    void InheritPins(const ReferencePins &source) {
        // An optimized-away scan still has a saved identity. Copy only the
        // inner owner's pins outward, not unrelated outer pins into the owner.
        auto capture = pending ? pending : make_shared_ptr<ReferencePins>(cache_budget);
        auto groups = PinGroups(*capture);
        duckdb_vx_error error = nullptr;
        bool inherited = vortex_index_pins_inherit(source.pins, groups.data(), groups.size(), &error);
        CheckError(error);
        if (inherited) {
            pending = std::move(capture);
        }
    }
    void RecordBind(void *request) {
        if (!preparing) {
            return;
        }
        auto capture = pending ? pending : make_shared_ptr<ReferencePins>(cache_budget);
        auto groups = PinGroups(*capture);
        duckdb_vx_error error = nullptr;
        vortex_index_pins_record(groups.data(), groups.size(), request, &error);
        CheckError(error);
        // Rebinds borrow the original prepared owner's cache. Independent binds
        // get a new cache, and the request holds only a weak reference to it.
        auto &owner = borrowed.empty() ? *capture : *borrowed.front();
        vortex_index_bind_cache(request, owner.pins);
        pending = std::move(capture);
    }
    CacheBudget cache_budget;
    vector<shared_ptr<ReferencePins>> borrowed;
    shared_ptr<ReferencePins> execute_owner;
    shared_ptr<ReferencePins> pending;
    struct SqlPrepareCapture {
        shared_ptr<PreparedStatementData> prepared;
        shared_ptr<ReferencePins> pins;
    };
    vector<SqlPrepareCapture> sql_prepares;
    bool preparing = false;
    bool sql_execute = false;
    unordered_map<const PreparedStatementData *, weak_ptr<PreparedPinLifetime>> owners;
};

void SavePreparedPins(PlannerExtensionInput &input, BoundStatement &statement) {
    if (statement.plan) {
        auto state = input.context.registered_state->Get<IndexPreparedState>(PIN_DEPENDENCY);
        state->CaptureSqlPrepares(*statement.plan);
    }
}

struct IndexConnectionCallback final : ExtensionCallback {
    void OnConnectionOpened(ClientContext &context) override {
        context.registered_state->GetOrCreate<IndexPreparedState>(PIN_DEPENDENCY);
    }
};

void RegisterPreparedState(DatabaseInstance &db) {
    ExtensionCallback::Register(DBConfig::GetConfig(db), make_shared_ptr<IndexConnectionCallback>());
    PlannerExtension extension;
    extension.post_bind_function = SavePreparedPins;
    PlannerExtension::Register(DBConfig::GetConfig(db), std::move(extension));
    // LOAD also needs the callback state on connections that already exist.
    for (auto &context : ConnectionManager::Get(db).GetConnectionList()) {
        context->registered_state->GetOrCreate<IndexPreparedState>(PIN_DEPENDENCY);
    }
}

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
        CheckStrings(value);
        pointers.push_back(reinterpret_cast<duckdb_value>(&value));
    }
    duckdb_logical_type result_type = nullptr;
    duckdb_vx_error error = nullptr;
    auto prepared = context.registered_state->Get<IndexPreparedState>(PIN_DEPENDENCY);
    auto request = vortex_index_bind(build, pointers.data(), pointers.size(), &result_type, &error);
    CheckError(error);
    auto bind = make_uniq<IndexBind>(request);
    unique_ptr<LogicalType> result(reinterpret_cast<LogicalType *>(result_type));
    if (!build && prepared) {
        prepared->RecordBind(request);
    }
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
    RegisterPreparedState(db);
    auto &catalog = Catalog::GetSystemCatalog(db);
    auto transaction = CatalogTransaction::GetSystemTransaction(db);
    for (auto &function : Functions()) {
        CreateTableFunctionInfo info(function);
        catalog.CreateFunction(transaction, info);
    }
}

#ifdef VORTEX_VANE_DISTRIBUTED
void RegisterVortexIndexFunctions(ExtensionLoader &loader) {
    RegisterPreparedState(loader.GetDatabaseInstance());
    for (auto &function : Functions()) {
        loader.RegisterFunction(function);
    }
}
#endif
