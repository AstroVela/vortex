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
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/query_node/recursive_cte_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/query_node/set_operation_node.hpp"
#include "duckdb/parser/statement/call_statement.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/delete_statement.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/statement/merge_into_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/statement/update_statement.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
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

struct ReferencePins final : DependencyItem {
    ReferencePins() : pins(vortex_index_pins_new()) {
    }
    ~ReferencePins() override {
        vortex_index_pins_free(pins);
    }
    void Remember(const PhysicalOperator &op) {
        if (op.type == PhysicalOperatorType::TABLE_SCAN) {
            auto &scan = op.Cast<PhysicalTableScan>();
            if (scan.function.name == "vortex_index_search") {
                duckdb_vx_error error = nullptr;
                vortex_index_pins_record(pins, scan.bind_data->Cast<IndexBind>().request, &error);
                CheckError(error);
            }
        }
        for (auto &child : op.GetChildren()) {
            Remember(child.get());
        }
    }
    void *pins;
};

struct PreparedPinOwner final : PhysicalOperator {
    PreparedPinOwner(PhysicalPlan &plan, const vector<LogicalType> &types, shared_ptr<ReferencePins> pins)
        : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, types, 0), pins(std::move(pins)) {
    }
    void BuildPipelines(Pipeline &, MetaPipeline &) override {
        throw InternalException("Vortex index reference owner is not an executable plan");
    }
    shared_ptr<ReferencePins> pins;
};

optional_ptr<TableRef> PinAnchor(QueryNode &node) {
    switch (node.type) {
    case QueryNodeType::SELECT_NODE:
        return node.Cast<SelectNode>().from_table.get();
    case QueryNodeType::SET_OPERATION_NODE:
        return PinAnchor(*node.Cast<SetOperationNode>().children.front());
    case QueryNodeType::RECURSIVE_CTE_NODE:
        return PinAnchor(*node.Cast<RecursiveCTENode>().left);
    default:
        return nullptr;
    }
}

optional_ptr<TableRef> PinAnchor(unique_ptr<SQLStatement> &statement) {
    if (statement->type == StatementType::CALL_STATEMENT) {
        auto &function = statement->Cast<CallStatement>().function;
        if (function->GetExpressionClass() != ExpressionClass::FUNCTION ||
            !StringUtil::CIEquals(function->Cast<FunctionExpression>().function_name,
                                  "vortex_index_search")) {
            return nullptr;
        }
        // Match DuckDB's CALL-to-SELECT rewrite so the unbound statement has a
        // dependency owner even when parameters prevent an initial bind.
        auto select = make_uniq<SelectStatement>();
        auto node = make_uniq<SelectNode>();
        auto table = make_uniq<TableFunctionRef>();
        table->function = std::move(function);
        node->from_table = std::move(table);
        node->select_list.push_back(make_uniq<StarExpression>());
        select->node = std::move(node);
        select->named_param_map = statement->named_param_map;
        select->query = statement->query;
        select->stmt_location = statement->stmt_location;
        select->stmt_length = statement->stmt_length;
        statement = std::move(select);
    }
    switch (statement->type) {
    case StatementType::SELECT_STATEMENT:
        return PinAnchor(*statement->Cast<SelectStatement>().node);
    case StatementType::INSERT_STATEMENT: {
        auto &select = statement->Cast<InsertStatement>().select_statement;
        return select ? PinAnchor(*select->node) : nullptr;
    }
    case StatementType::COPY_STATEMENT: {
        auto &select = statement->Cast<CopyStatement>().info->select_statement;
        return select ? PinAnchor(*select) : nullptr;
    }
    case StatementType::DELETE_STATEMENT:
        return statement->Cast<DeleteStatement>().table.get();
    case StatementType::UPDATE_STATEMENT:
        return statement->Cast<UpdateStatement>().table.get();
    case StatementType::MERGE_INTO_STATEMENT:
        return statement->Cast<MergeIntoStatement>().target.get();
    case StatementType::EXPLAIN_STATEMENT:
        return PinAnchor(statement->Cast<ExplainStatement>().stmt);
    case StatementType::CREATE_STATEMENT: {
        auto &info = *statement->Cast<CreateStatement>().info;
        if (info.type == CatalogType::TABLE_ENTRY) {
            auto &query = info.Cast<CreateTableInfo>().query;
            return query ? PinAnchor(*query->node) : nullptr;
        }
        if (info.type == CatalogType::VIEW_ENTRY) {
            auto &query = info.Cast<CreateViewInfo>().query;
            return query ? PinAnchor(*query->node) : nullptr;
        }
        return nullptr;
    }
    default:
        return nullptr;
    }
}

struct IndexPreparedState final : ClientContextState {
    bool CanRequestRebind() override {
        // DuckDB only invokes OnFinalizePrepare for states with this capability.
        return true;
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
    void Save(ClientContext &context, PreparedStatementData &prepared) {
        if (!pending) {
            return;
        }
        auto pins = std::move(pending);
        if (prepared.unbound_statement) {
            auto anchor = PinAnchor(prepared.unbound_statement);
            if (!anchor) {
                throw InternalException("Index statement has no reference dependency owner");
            }
            if (!anchor->external_dependency) {
                anchor->external_dependency = make_shared_ptr<ExternalDependency>();
            }
            anchor->external_dependency->AddDependency(PIN_DEPENDENCY, std::move(pins));
        } else {
            // C API prepare assigns the unbound statement AFTER this callback.
            // An arena-owned operator keeps the first-bind pins alive even when
            // DuckDB discarded the query plan. A root owner is never executed.
            bool needs_rebind = !prepared.physical_plan;
            if (needs_rebind) {
                prepared.physical_plan = make_uniq<PhysicalPlan>(Allocator::Get(context));
                prepared.properties.always_require_rebind = true;
            }
            auto &owner = prepared.physical_plan->Make<PreparedPinOwner>(prepared.types, pins);
            if (needs_rebind) {
                prepared.physical_plan->SetRoot(owner);
            }
            PruneOwners();
            owners[&prepared] = pins;
        }
    }
    RebindQueryInfo OnFinalizePrepare(ClientContext &context,
                                      PreparedStatementData &prepared,
                                      PreparedStatementMode) override {
        Save(context, prepared);
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    RebindQueryInfo OnPlanningError(ClientContext &, SQLStatement &, ErrorData &) override {
        pending.reset();
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    void Activate(PreparedStatementData &prepared) {
        active.reset();
        PruneOwners();
        auto owner = owners.find(&prepared);
        if (owner != owners.end()) {
            active = owner->second.lock();
        }
        if (!prepared.unbound_statement) {
            return;
        }
        auto anchor = PinAnchor(prepared.unbound_statement);
        if (!anchor) {
            return;
        }
        auto &dependencies = anchor->external_dependency;
        if (!dependencies) {
            dependencies = make_shared_ptr<ExternalDependency>();
        }
        auto existing = dependencies->GetDependency(PIN_DEPENDENCY);
        if (existing) {
            active = shared_ptr_cast<DependencyItem, ReferencePins>(existing);
        } else {
            if (!active) {
                active = make_shared_ptr<ReferencePins>();
                if (prepared.physical_plan) {
                    active->Remember(prepared.physical_plan->Root());
                }
            }
            dependencies->AddDependency(PIN_DEPENDENCY, active);
        }
    }
    RebindQueryInfo
    OnExecutePrepared(ClientContext &, PreparedStatementCallbackInfo &info, RebindQueryInfo) override {
        Activate(info.prepared_statement);
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    RebindQueryInfo OnRebindPreparedStatement(ClientContext &,
                                              BindPreparedStatementCallbackInfo &info,
                                              RebindQueryInfo) override {
        Activate(info.prepared_statement);
        return RebindQueryInfo::DO_NOT_REBIND;
    }
    void QueryEnd() override {
        active.reset();
        pending.reset();
        PruneOwners();
    }
    shared_ptr<ReferencePins> BindPins() {
        if (active) {
            return active;
        }
        if (!pending) {
            pending = make_shared_ptr<ReferencePins>();
        }
        return pending;
    }
    shared_ptr<ReferencePins> active;
    shared_ptr<ReferencePins> pending;
    unordered_map<const PreparedStatementData *, weak_ptr<ReferencePins>> owners;
};

void SavePreparedPins(PlannerExtensionInput &input, BoundStatement &statement) {
    if (statement.plan && statement.plan->type == LogicalOperatorType::LOGICAL_PREPARE) {
        // SQL PREPARE has its own planner, including partial binds that discarded
        // the inner plan. Its unbound statement is already available here.
        auto state = input.context.registered_state->Get<IndexPreparedState>(PIN_DEPENDENCY);
        state->Save(input.context, *statement.plan->Cast<LogicalPrepare>().prepared);
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
    auto pins = !build && prepared ? prepared->BindPins() : nullptr;
    auto request = vortex_index_bind(build,
                                     pointers.data(),
                                     pointers.size(),
                                     pins ? pins->pins : nullptr,
                                     &result_type,
                                     &error);
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
