// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "data.hpp"
#include "error.hpp"
#include "table_function.hpp"
#include "expr.h"
#include "vortex_duckdb.h"
#include "table_function.h"
#include "vortex.h"

#include "duckdb.h"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/insertion_order_preserving_map.hpp"
#include "duckdb/common/multi_file/multi_file_reader.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#ifdef VORTEX_VANE_DISTRIBUTED
#include <algorithm>

#include "duckdb/common/exception.hpp"
#include "duckdb/common/limits.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#endif

using namespace std::string_literals;
constexpr column_t COLUMN_IDENTIFIER_FILE_INDEX = MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
constexpr column_t COLUMN_IDENTIFIER_FILE_ROW_NUMBER = MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;

unique_ptr<FunctionData> VortexBindData::Copy() const {
#ifdef VORTEX_VANE_DISTRIBUTED
    unique_ptr<CData> ffi_data_copy;
    if (ffi_data) {
        const auto copied_ffi_data = duckdb_table_function_bind_data_clone(ffi_data->DataPtr());
        ffi_data_copy = unique_ptr<CData>(reinterpret_cast<CData *>(copied_ffi_data));
    }
    auto result = make_uniq<VortexBindData>(std::move(ffi_data_copy), types, names);
    result->scan_split_set_id = scan_split_set_id;
    result->portable_bind = portable_bind;
    result->distributed_files = distributed_files;
    result->aggregate_scan = aggregate_scan;
    result->explicit_split_mode = explicit_split_mode;
    result->splits_applied = splits_applied;
    result->eligible_file_indexes = eligible_file_indexes;
    result->assigned_fragments = assigned_fragments;
    return result;
#else
    const auto copied_ffi_data = duckdb_table_function_bind_data_clone(ffi_data->DataPtr());
    auto ffi_data_p = unique_ptr<CData>(reinterpret_cast<CData *>(copied_ffi_data));
    return make_uniq<VortexBindData>(std::move(ffi_data_p), types);
#endif
}

bool VortexBindData::Equals(const FunctionData &other_base) const {
    const VortexBindData &other = other_base.Cast<VortexBindData>();
#ifdef VORTEX_VANE_DISTRIBUTED
    if (ffi_data || other.ffi_data) {
        // Runtime bind equality retains the upstream pointer-identity semantics.
        return ffi_data.get() == other.ffi_data.get();
    }
    return types == other.types && names == other.names && scan_split_set_id == other.scan_split_set_id &&
           portable_bind == other.portable_bind && distributed_files == other.distributed_files &&
           aggregate_scan == other.aggregate_scan && explicit_split_mode == other.explicit_split_mode &&
           splits_applied == other.splits_applied && eligible_file_indexes == other.eligible_file_indexes &&
           assigned_fragments == other.assigned_fragments;
#else
    // if "types" are different, "ffi_data" would also be different as it
    // contains types inside, so omit "types" from comparison.
    return ffi_data.get() == other.ffi_data.get();
#endif
}

#ifdef VORTEX_VANE_DISTRIBUTED
VortexBindData::PortableSnapshot VortexBindData::CreatePortableSnapshot() const {
    hugeint_t parsed_scan_id;
    if (!UUID::FromString(scan_split_set_id, parsed_scan_id, true) ||
        UUID::ToString(parsed_scan_id) != scan_split_set_id) {
        throw SerializationException("Vortex bind has an invalid distributed scan identity");
    }
    if (!ffi_data) {
        if (portable_bind.empty()) {
            throw SerializationException("Deserialized Vortex bind has no portable state");
        }
        return {portable_bind, distributed_files, aggregate_scan};
    }

    PortableSnapshot result;
    duckdb_vx_error error_out = nullptr;
    auto portable_data = duckdb_table_function_distributed_bind_serialize(ffi_data->DataPtr(), &error_out);
    if (error_out) {
        throw SerializationException(IntoErrString(error_out));
    }
    if (!portable_data) {
        throw SerializationException("Vortex failed to serialize distributed bind state");
    }
    auto portable = unique_ptr<CData>(reinterpret_cast<CData *>(portable_data));
    size_t portable_size = 0;
    const auto portable_bytes =
        duckdb_table_function_distributed_bind_bytes(portable->DataPtr(), &portable_size);
    if (!portable_bytes || portable_size == 0) {
        throw SerializationException("Vortex produced empty distributed bind state");
    }
    result.portable_bind.assign(reinterpret_cast<const char *>(portable_bytes), portable_size);

    auto file_count = duckdb_table_function_distributed_file_count(portable->DataPtr());
    result.aggregate_scan = duckdb_table_function_distributed_is_aggregate(portable->DataPtr());
    vector<DistributedFile> files;
    files.reserve(file_count);
    for (idx_t file_index = 0; file_index < file_count; file_index++) {
        VortexDistributedFileView view {};
        if (!duckdb_table_function_distributed_file_at(portable->DataPtr(), file_index, &view) ||
            !view.source_url || view.source_url_len == 0 || !view.path || view.path_len == 0 ||
            view.size == DConstants::INVALID_INDEX) {
            throw SerializationException("Vortex produced an invalid distributed file at index %llu",
                                         static_cast<unsigned long long>(file_index));
        }
        DistributedFile file;
        file.source_url.assign(reinterpret_cast<const char *>(view.source_url), view.source_url_len);
        file.path.assign(reinterpret_cast<const char *>(view.path), view.path_len);
        file.size = view.size;
        files.push_back(std::move(file));
    }
    result.distributed_files = std::move(files);
    return result;
}

static vector<VortexBindData::DistributedFragment> PlanVortexFragments(const string &portable_bind,
                                                                       const vector<idx_t> &file_indexes,
                                                                       idx_t target_fragment_count) {
    if (!std::is_sorted(file_indexes.begin(), file_indexes.end()) ||
        std::adjacent_find(file_indexes.begin(), file_indexes.end()) != file_indexes.end()) {
        throw InvalidInputException("Vortex fragment file indexes are not in canonical order");
    }
    duckdb_vx_error error_out = nullptr;
    auto fragment_plan_data = duckdb_table_function_distributed_plan_fragments(
        reinterpret_cast<const uint8_t *>(portable_bind.data()),
        portable_bind.size(),
        file_indexes.data(),
        file_indexes.size(),
        target_fragment_count,
        &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
    if (!fragment_plan_data) {
        throw InvalidInputException("Vortex failed to plan distributed scan fragments");
    }
    auto fragment_plan = unique_ptr<CData>(reinterpret_cast<CData *>(fragment_plan_data));
    const auto fragment_count = duckdb_table_function_distributed_fragment_count(fragment_plan->DataPtr());
    vector<VortexBindData::DistributedFragment> fragments;
    fragments.reserve(fragment_count);
    for (idx_t fragment_index = 0; fragment_index < fragment_count; fragment_index++) {
        VortexDistributedFragmentView view {};
        if (!duckdb_table_function_distributed_fragment_at(fragment_plan->DataPtr(), fragment_index, &view) ||
            !std::binary_search(file_indexes.begin(), file_indexes.end(), view.file_index) ||
            view.row_start > view.row_end || view.estimated_bytes == DConstants::INVALID_INDEX) {
            throw InvalidInputException("Vortex produced an invalid distributed fragment at index %llu",
                                        static_cast<unsigned long long>(fragment_index));
        }
        VortexBindData::DistributedFragment fragment {view.file_index,
                                                      view.row_start,
                                                      view.row_end,
                                                      view.estimated_bytes};
        if (!fragments.empty()) {
            const auto &previous = fragments.back();
            if (previous.file_index > fragment.file_index ||
                (previous.file_index == fragment.file_index &&
                 (previous.row_start >= fragment.row_start || previous.row_end > fragment.row_start))) {
                throw InvalidInputException("Vortex produced fragments outside canonical order at index %llu",
                                            static_cast<unsigned long long>(fragment_index));
            }
        }
        fragments.push_back(fragment);
    }
    return fragments;
}
#endif

// This is a flaw of Duckdb API which doesn't allow passing non-const
// expressions. We never modify the value on Rust side.
static duckdb_vx_expr get_ffi_expr(const Expression &expr) {
    return reinterpret_cast<duckdb_vx_expr>(const_cast<Expression *>(&expr));
}

static void *get_ffi_bind(const FunctionData *bind_data) {
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &bind = bind_data->Cast<VortexBindData>();
    if (!bind.ffi_data) {
        throw InternalException("Vortex runtime bind state is not initialized");
    }
    return bind.ffi_data->DataPtr();
#else
    return bind_data->Cast<VortexBindData>().ffi_data->DataPtr();
#endif
}

static void *get_ffi_global(GlobalTableFunctionState *state) {
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &global = state->Cast<VortexGlobalData>();
    auto ffi_global = global.ffi_data->DataPtr();
    if (global.distributed) {
        return duckdb_table_function_distributed_global_data(ffi_global);
    }
    return ffi_global;
#else
    return state->Cast<VortexGlobalData>().ffi_data->DataPtr();
#endif
}

static void *get_ffi_local(LocalTableFunctionState *state) {
    return state->Cast<VortexLocalData>().ffi_data->DataPtr();
}

double
table_scan_progress(ClientContext &, const FunctionData *, const GlobalTableFunctionState *global_state) {
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &global = global_state->Cast<VortexGlobalData>();
    auto c_global_state = global.ffi_data->DataPtr();
    if (global.distributed) {
        c_global_state = duckdb_table_function_distributed_global_data(c_global_state);
    }
#else
    void *const c_global_state = global_state->Cast<VortexGlobalData>().ffi_data->DataPtr();
#endif
    return duckdb_table_function_scan_progress(c_global_state);
}

static Value &UnwrapValue(duckdb_value value) {
    return *(reinterpret_cast<Value *>(value));
}

#ifdef VORTEX_VANE_DISTRIBUTED
struct OwnedColumnStatistics {
    ~OwnedColumnStatistics() {
        if (value.min) {
            duckdb_destroy_value(&value.min);
        }
        if (value.max) {
            duckdb_destroy_value(&value.max);
        }
    }

    duckdb_column_statistics value {};
};
#endif

unique_ptr<BaseStatistics> numeric_stats(duckdb_column_statistics &stats, LogicalType type) {
#ifdef VORTEX_VANE_DISTRIBUTED
    BaseStatistics out = NumericStats::CreateUnknown(type);
#else
    BaseStatistics out = StringStats::CreateUnknown(type);
#endif
    if (stats.min) {
        NumericStats::SetMin(out, UnwrapValue(stats.min));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        NumericStats::SetMax(out, UnwrapValue(stats.max));
        duckdb_destroy_value(&stats.max);
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> string_stats(duckdb_column_statistics &stats, LogicalType type) {
    BaseStatistics out = StringStats::CreateUnknown(type);
    if (stats.min) {
        StringStats::SetMin(out, StringValue::Get(UnwrapValue(stats.min)));
        duckdb_destroy_value(&stats.min);
    }
    if (stats.max) {
        StringStats::SetMax(out, StringValue::Get(UnwrapValue(stats.max)));
        duckdb_destroy_value(&stats.max);
    }
    if (stats.max_string_length >> 63) {
        StringStats::SetMaxStringLength(out, uint32_t(stats.max_string_length));
    }
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }

    return out.ToUnique();
}

unique_ptr<BaseStatistics> base_stats(duckdb_column_statistics &stats, LogicalType type) {
#ifdef VORTEX_VANE_DISTRIBUTED
    BaseStatistics out = BaseStatistics::CreateUnknown(type);
#else
    BaseStatistics out = StringStats::CreateUnknown(type);
#endif
    if (!stats.has_null) {
        out.Set(StatsInfo::CANNOT_HAVE_NULL_VALUES);
    }
    return out.ToUnique();
}

unique_ptr<BaseStatistics> statistics(ClientContext &, const FunctionData *bind_data, column_t column_index) {
    if (IsVirtualColumn(column_index)) {
        return {};
    }

    const auto &bind = bind_data->Cast<VortexBindData>();
#ifdef VORTEX_VANE_DISTRIBUTED
    if (!bind.ffi_data) {
        return {};
    }
#endif
    const void *const ffi_bind = get_ffi_bind(bind_data);

#ifdef VORTEX_VANE_DISTRIBUTED
    OwnedColumnStatistics owned_statistics;
    auto &statistics = owned_statistics.value;
#else
    duckdb_column_statistics statistics = {};
#endif
    if (!duckdb_table_function_statistics(ffi_bind, column_index, &statistics)) {
        return {};
    }

    const LogicalType type = bind.types[column_index];

    switch (type.id()) {
    case LogicalTypeId::BOOLEAN:
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE:
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
    case LogicalTypeId::UHUGEINT:
    case LogicalTypeId::HUGEINT: {
        return numeric_stats(statistics, type);
    }
    case LogicalTypeId::VARCHAR:
    case LogicalTypeId::BLOB: {
        return string_stats(statistics, type);
    }
    case LogicalTypeId::STRUCT: {
        // TODO(myrrc)
        // Duckdb's has_null has a different semantics for structs.
        // If we propagate our has_null, this breaks Duckdb optimizer.
        // You can reproduce it in struct.slt test in vortex-sqllogictests:
        return {};
    }
    default:
        return base_stats(statistics, type);
    }
}

bool projection_expression_pushdown(ClientContext &, const TableFunctionProjectionExpressionInput &input) {
#ifdef VORTEX_VANE_DISTRIBUTED
    // A logical-plan round trip reconstructs the bind from portable owned data.
    // There is deliberately no live Rust reader until a worker receives splits
    // and enters init_global with its real ClientContext. Keep any newly
    // discovered projection expression above the scan in that detached state.
    if (!input.get.bind_data->Cast<VortexBindData>().ffi_data) {
        return false;
    }
#endif
    duckdb_vx_expr ffi_expr = get_ffi_expr(input.expression);
    void *const ffi_bind = get_ffi_bind(input.get.bind_data.get());
    duckdb_vx_error error_out = nullptr;

    const bool ret = duckdb_table_function_pushdown_projection_expression( //
        ffi_bind,
        ffi_expr,
        input.projection_idx,
        &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    return ret;
}

extern "C" {
idx_t duckdb_vx_aggregate_len(duckdb_vx_agg_input ffi_input) {
    return reinterpret_cast<const TableFunctionUngroupedAggregateInput *>(ffi_input)->projections.size();
}

duckdb_vx_expr duckdb_vx_aggregate_at(duckdb_vx_agg_input ffi_input, idx_t i, idx_t *proj_idx) {
    const auto &input = *reinterpret_cast<const TableFunctionUngroupedAggregateInput *>(ffi_input);
    const auto &[scan_index, expr] = input.projections[i];
    *proj_idx = scan_index == COUNT_STAR_PROJ_IDX ? scan_index
                                                  : input.get.GetColumnIds()[scan_index].GetPrimaryIndex();
    return get_ffi_expr(expr);
}
}

bool aggregate_pushdown(ClientContext &, const TableFunctionUngroupedAggregateInput &input) {
#ifdef VORTEX_VANE_DISTRIBUTED
    // TryReplaceAggregate rewrites the LogicalGet's column IDs to aggregate
    // result positions. DuckDB table filters still refer to the original scan
    // columns, so retaining them would either attach a filter to the wrong
    // field or make physical planning fail (in particular for virtual file
    // columns). Keep the upper aggregate whenever table filters remain.
    if (!input.get.table_filters.filters.empty()) {
        return false;
    }
    // Rust aggregate pushdown indexes physical bind fields. Virtual columns
    // have no valid physical projection index and must stay in the upper
    // aggregate for both live and detached Vane binds.
    for (const auto &projection : input.projections) {
        const auto scan_index = projection.first;
        if (scan_index != COUNT_STAR_PROJ_IDX && input.get.GetColumnIds()[scan_index].IsVirtualColumn()) {
            return false;
        }
    }
    auto &bind_data = input.get.bind_data->Cast<VortexBindData>();
    if (!bind_data.ffi_data) {
        duckdb_vx_error error_out = nullptr;
        const auto ffi_input =
            reinterpret_cast<duckdb_vx_agg_input>(const_cast<TableFunctionUngroupedAggregateInput *>(&input));
        auto portable_data = duckdb_table_function_distributed_bind_pushdown_projection_aggregates(
            reinterpret_cast<const uint8_t *>(bind_data.portable_bind.data()),
            bind_data.portable_bind.size(),
            ffi_input,
            &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        if (!portable_data) {
            return false;
        }
        auto portable = unique_ptr<CData>(reinterpret_cast<CData *>(portable_data));
        size_t portable_size = 0;
        const auto portable_bytes =
            duckdb_table_function_distributed_bind_bytes(portable->DataPtr(), &portable_size);
        if (!portable_bytes || portable_size == 0 ||
            !duckdb_table_function_distributed_is_aggregate(portable->DataPtr())) {
            throw SerializationException("Vortex produced invalid aggregate bind state");
        }
        bind_data.portable_bind.assign(reinterpret_cast<const char *>(portable_bytes), portable_size);
        bind_data.aggregate_scan = true;
        return true;
    }
#endif
    void *const ffi_bind = get_ffi_bind(input.get.bind_data.get());
    duckdb_vx_error error_out = nullptr;
    const auto ffi_input =
        reinterpret_cast<duckdb_vx_agg_input>(const_cast<TableFunctionUngroupedAggregateInput *>(&input));
    const bool res = duckdb_table_function_pushdown_projection_aggregates(ffi_bind, ffi_input, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }
    return res;
}

unique_ptr<FunctionData> duckdb_vx_table_function_bind(ClientContext &,
                                                       TableFunctionBindInput &input,
                                                       vector<LogicalType> &return_types,
                                                       vector<string> &names) {
#ifdef VORTEX_VANE_DISTRIBUTED
    VortexBindResults bind_results = {return_types, names};
#else
    VortexBindResults result = {return_types, names};
#endif

    duckdb_vx_error error_out = nullptr;
    duckdb_vx_tfunc_bind_input bind_input = reinterpret_cast<duckdb_vx_tfunc_bind_input>(&input);
#ifdef VORTEX_VANE_DISTRIBUTED
    duckdb_vx_tfunc_bind_result bind_result = reinterpret_cast<duckdb_vx_tfunc_bind_result>(&bind_results);
#else
    duckdb_vx_tfunc_bind_result bind_result = reinterpret_cast<duckdb_vx_tfunc_bind_result>(&result);
#endif
    duckdb_vx_data ffi_bind_data = duckdb_table_function_bind(bind_input, bind_result, &error_out);
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_bind_data));
#ifdef VORTEX_VANE_DISTRIBUTED
    auto result = make_uniq<VortexBindData>(std::move(cdata), return_types, names);
    result->scan_split_set_id = UUID::ToString(UUID::GenerateRandomUUID());
    return result;
#else
    return make_uniq<VortexBindData>(std::move(cdata), return_types);
#endif
}

unique_ptr<GlobalTableFunctionState> init_global(ClientContext &context, TableFunctionInitInput &input) {
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &bind_data = input.bind_data->Cast<VortexBindData>();
    if (bind_data.explicit_split_mode && !bind_data.splits_applied) {
        throw InvalidInputException(
            "Detached distributed Vortex worker bind has no explicit split assignment");
    }
#else
    const void *const ffi_bind = get_ffi_bind(input.bind_data.get());
#endif

    duckdb_vx_tfunc_init_input ffi_input = {
#ifdef VORTEX_VANE_DISTRIBUTED
        .bind_data = bind_data.ffi_data ? bind_data.ffi_data->DataPtr() : nullptr,
#else
        .bind_data = ffi_bind,
#endif
        .column_ids = input.column_ids.data(),
        .column_ids_count = input.column_ids.size(),
        .projection_ids = input.projection_ids.data(),
        .projection_ids_count = input.projection_ids.size(),
        .filters = reinterpret_cast<duckdb_vx_table_filter_set>(input.filters.get()),
        .client_context = reinterpret_cast<duckdb_client_context>(&context),
    };

    duckdb_vx_error error_out = nullptr;
#ifdef VORTEX_VANE_DISTRIBUTED
    duckdb_vx_data ffi_global_data;
    bool distributed = false;
    if (!bind_data.ffi_data) {
        const auto snapshot = bind_data.CreatePortableSnapshot();
        vector<VortexBindData::DistributedFragment> native_fragments;
        const vector<VortexBindData::DistributedFragment> *runtime_fragments = &bind_data.assigned_fragments;
        if (!bind_data.explicit_split_mode) {
            vector<idx_t> native_file_indexes;
            native_file_indexes.reserve(snapshot.distributed_files.size());
            for (idx_t file_index = 0; file_index < snapshot.distributed_files.size(); file_index++) {
                native_file_indexes.push_back(file_index);
            }
            native_fragments =
                PlanVortexFragments(snapshot.portable_bind, native_file_indexes, native_file_indexes.size());
            runtime_fragments = &native_fragments;
        }
        vector<VortexDistributedFragmentView> runtime_fragment_views;
        runtime_fragment_views.reserve(runtime_fragments->size());
        for (const auto &fragment : *runtime_fragments) {
            runtime_fragment_views.push_back(
                {fragment.file_index, fragment.row_start, fragment.row_end, fragment.estimated_bytes});
        }
        // Optional filters (for example TopN's dynamic bound) are maintained
        // by an upstream operator that is absent from Vane's detached scan
        // plan. They are pruning hints, so ignoring them is correctness-safe;
        // required table filters still pass through unchanged.
        ffi_global_data = duckdb_table_function_init_global_distributed(
            reinterpret_cast<const uint8_t *>(snapshot.portable_bind.data()),
            snapshot.portable_bind.size(),
            runtime_fragment_views.data(),
            runtime_fragment_views.size(),
            bind_data.explicit_split_mode,
            &ffi_input,
            &error_out);
        distributed = true;
    } else {
        ffi_global_data = duckdb_table_function_init_global(&ffi_input, &error_out);
    }
#else
    duckdb_vx_data ffi_global_data = duckdb_table_function_init_global(&ffi_input, &error_out);
#endif
    if (error_out) {
        throw BinderException(IntoErrString(error_out));
    }

    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_global_data));
#ifdef VORTEX_VANE_DISTRIBUTED
    bool force_empty_output = false;
    force_empty_output = distributed && bind_data.explicit_split_mode && bind_data.splits_applied &&
                         bind_data.assigned_fragments.empty();
    return make_uniq<VortexGlobalData>(std::move(cdata), distributed, force_empty_output);
#else
    return make_uniq<VortexGlobalData>(std::move(cdata));
#endif
}

unique_ptr<LocalTableFunctionState>
init_local(ExecutionContext &, TableFunctionInitInput &input, GlobalTableFunctionState *global_state) {
#ifdef VORTEX_VANE_DISTRIBUTED
    const void *ffi_bind;
    auto &global = global_state->Cast<VortexGlobalData>();
    if (global.distributed) {
        ffi_bind = duckdb_table_function_distributed_bind_data(global.ffi_data->DataPtr());
    } else {
        ffi_bind = get_ffi_bind(input.bind_data.get());
    }
#else
    const void *const ffi_bind = get_ffi_bind(input.bind_data.get());
#endif
    void *const ffi_global = get_ffi_global(global_state);

    duckdb_vx_data ffi_local_data = duckdb_table_function_init_local(ffi_bind, ffi_global);
    auto cdata = unique_ptr<CData>(reinterpret_cast<CData *>(ffi_local_data));
    return make_uniq<VortexLocalData>(std::move(cdata));
}

void function(ClientContext &, TableFunctionInput &input, DataChunk &output) {
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &global = input.global_state->Cast<VortexGlobalData>();
    if (global.force_empty_output) {
        output.SetCardinality(0);
        return;
    }
#endif
    void *const ffi_global = get_ffi_global(input.global_state.get());
    void *const ffi_local = get_ffi_local(input.local_state.get());

    duckdb_data_chunk chunk = reinterpret_cast<duckdb_data_chunk>(&output);
    duckdb_vx_error error_out = nullptr;
    duckdb_table_function_scan(ffi_global, ffi_local, chunk, &error_out);
    if (error_out) {
        throw InvalidInputException(IntoErrString(error_out));
    }
}

using FilterVec = vector<unique_ptr<Expression>>;

void pushdown_complex_filter(const FunctionData &bind_data, FilterVec &filters) {
#ifdef VORTEX_VANE_DISTRIBUTED
    // Leave filters in DuckDB's plan when optimizing a detached bind. The
    // worker will still apply already-serialized Vortex filters and its table
    // filters, without constructing a reader during deserialization.
    if (!bind_data.Cast<VortexBindData>().ffi_data) {
        return;
    }
#endif
    void *const ffi_bind = get_ffi_bind(&bind_data);
    duckdb_vx_error error_out = nullptr;

    for (auto iter = filters.begin(); iter != filters.end();) {
        duckdb_vx_expr ffi_expr = reinterpret_cast<duckdb_vx_expr>(iter->get());

        const bool pushed = duckdb_table_function_pushdown_complex_filter(ffi_bind, ffi_expr, &error_out);
        if (error_out) {
            throw BinderException(IntoErrString(error_out));
        }
        iter = pushed ? filters.erase(iter) : std::next(iter);
    }
}

unique_ptr<NodeStatistics> cardinality(ClientContext &, const FunctionData *bind_data) {
#ifdef VORTEX_VANE_DISTRIBUTED
    if (!bind_data->Cast<VortexBindData>().ffi_data) {
        return make_uniq<NodeStatistics>();
    }
#endif
    const void *const ffi_bind = get_ffi_bind(bind_data);

    duckdb_vx_node_statistics stats = {};
    duckdb_table_function_cardinality(ffi_bind, &stats);

    auto out = make_uniq<NodeStatistics>();
    out->has_estimated_cardinality = stats.has_estimated_cardinality;
    out->estimated_cardinality = stats.estimated_cardinality;
    out->has_max_cardinality = stats.has_max_cardinality;
    out->max_cardinality = stats.max_cardinality;

    return out;
}

extern "C" duckdb_value duckdb_vx_tfunc_bind_input_get_parameter(duckdb_vx_tfunc_bind_input ffi_input,
                                                                 size_t index) {
    D_ASSERT(ffi_input);
    const TableFunctionBindInput &input = *reinterpret_cast<TableFunctionBindInput *>(ffi_input);
    return reinterpret_cast<duckdb_value>(new Value(input.inputs[index]));
}

extern "C" void duckdb_vx_tfunc_bind_result_add_column(duckdb_vx_tfunc_bind_result ffi_result,
                                                       const char *name_str,
                                                       size_t name_len,
                                                       duckdb_logical_type ffi_type) {
    D_ASSERT(ffi_result);
    D_ASSERT(name_str);
    D_ASSERT(ffi_type);
    const VortexBindResults &result = *reinterpret_cast<VortexBindResults *>(ffi_result);
    const LogicalType logical_type = *reinterpret_cast<LogicalType *>(ffi_type);

    result.names.emplace_back(name_str, name_len);
    result.return_types.emplace_back(logical_type);
}

/**
 * Called at planning time to determine whether data is partitioned by a
 * given set of columns. Requested columns are GROUP BY parameters i.e. columns
 * over which the query aggregates.
 */
TablePartitionInfo get_partition_info(ClientContext &, TableFunctionPartitionInput &input) {
    const vector<column_t> &ids = input.partition_ids;
    // Our data is partitioned by array exporters. Each exporter processes a
    // single Array which belongs to a single file. If data is partitioned only
    // by file_index, there is one unique value for an Array. Otherwise there
    // may be multiple values.
    return (ids.size() == 1 && ids[0] == COLUMN_IDENTIFIER_FILE_INDEX)
               ? TablePartitionInfo::SINGLE_VALUE_PARTITIONS
               : TablePartitionInfo::NOT_PARTITIONED;
}

OperatorPartitionData get_partition_data(ClientContext &, TableFunctionGetPartitionInput &input) {
    void *const ffi_global = get_ffi_global(input.global_state.get());
    void *const ffi_local = get_ffi_local(input.local_state.get());
    duckdb_vx_partition_data partition_data;
    duckdb_table_function_get_partition_data(ffi_global, ffi_local, &partition_data);

    OperatorPartitionData out(partition_data.partition_index);

    // file_index_column_pos may be INVALID_IDX, but column_index will never
    // be INVALID_IDX, so we can compare directly
    for (const column_t column_index : input.partition_info.partition_columns) {
        if (column_index == partition_data.file_index_column_pos) {
            out.partition_data.emplace_back(Value::UBIGINT(partition_data.file_index));
        } else {
            throw InternalException(StringUtil::Format(
                "get_partition_data: requested column_index %d is not constant for given partition",
                column_index));
        }
    }
    return out;
}

extern "C" void duckdb_vx_string_map_insert(duckdb_vx_string_map map, const char *key, const char *value) {
    D_ASSERT(map);
    D_ASSERT(key);
    D_ASSERT(value);
    reinterpret_cast<InsertionOrderPreservingMap<string> *>(map)->insert(key, value);
}

InsertionOrderPreservingMap<string> to_string(TableFunctionToStringInput &input) {
    InsertionOrderPreservingMap<string> result;
#ifdef VORTEX_VANE_DISTRIBUTED
    auto &bind_data = input.bind_data->Cast<VortexBindData>();
    if (!bind_data.ffi_data) {
        result.insert("Function", "Vortex Scan");
        if (bind_data.explicit_split_mode) {
            result.insert("Assigned fragments", std::to_string(bind_data.assigned_fragments.size()));
        } else {
            result.insert("Distributed files", std::to_string(bind_data.distributed_files.size()));
        }
        return result;
    }
#endif
    duckdb_vx_string_map ffi_map = reinterpret_cast<duckdb_vx_string_map>(&result);
    const void *const ffi_bind = get_ffi_bind(input.bind_data.get());
    duckdb_table_function_to_string(ffi_bind, ffi_map);
    return result;
}

#ifdef VORTEX_VANE_DISTRIBUTED
namespace {

static constexpr uint8_t VORTEX_SPLIT_PAYLOAD_VERSION = 1;
static constexpr uint8_t VORTEX_BIND_SERDE_VERSION = 1;
static constexpr idx_t VORTEX_MIN_FRAGMENT_PAYLOAD_BYTES = sizeof(uint64_t) * 7;
static constexpr const char *VORTEX_SPLIT_CODEC = "vane.vortex-file-fragment-split";

static bool IsCanonicalVortexScanId(const string &scan_split_set_id) {
    hugeint_t parsed;
    return UUID::FromString(scan_split_set_id, parsed, true) && UUID::ToString(parsed) == scan_split_set_id;
}

static bool IsCanonicalVortexFilePath(const string &path) {
    if (path.empty() || path.front() == '/' || path.back() == '/' || path.find('\0') != string::npos) {
        return false;
    }
    idx_t segment_start = 0;
    while (segment_start < path.size()) {
        auto segment_end = path.find('/', segment_start);
        if (segment_end == string::npos) {
            segment_end = path.size();
        }
        const auto segment_size = segment_end - segment_start;
        if (segment_size == 0 || (segment_size == 1 && path[segment_start] == '.') ||
            (segment_size == 2 && path[segment_start] == '.' && path[segment_start + 1] == '.')) {
            return false;
        }
        segment_start = segment_end + 1;
    }
    return true;
}

static void AppendSplitByte(string &result, uint8_t value) {
    result.push_back(static_cast<char>(value));
}

static void AppendSplitU64(string &result, uint64_t value) {
    for (idx_t byte_index = 0; byte_index < sizeof(value); byte_index++) {
        AppendSplitByte(result, static_cast<uint8_t>((value >> (byte_index * 8U)) & 0xffU));
    }
}

static void AppendSplitString(string &result, const string &value) {
    AppendSplitU64(result, value.size());
    result.append(value);
}

// Binary fragment payload v1:
//   "VXFR" | u8 version | string scan_split_set_id | u64 fragment_count |
//   repeated(u64 stable_file_index | string source_url | string path |
//            u64 immutable_size | u64 row_start | u64 row_end | u64 estimated_bytes)
// Normal scans encode one independently assignable fragment. Aggregate-pushed
// scans encode one full-file fragment per selected file in one complete-set split.
static string EncodeVortexSplit(const vector<VortexBindData::DistributedFragment> &fragments,
                                const vector<VortexBindData::DistributedFile> &distributed_files,
                                const string &scan_split_set_id) {
    if (fragments.empty()) {
        throw InternalException("Cannot encode an empty distributed Vortex split");
    }
    if (!IsCanonicalVortexScanId(scan_split_set_id)) {
        throw InternalException("Cannot encode a distributed Vortex split without a canonical scan identity");
    }
    string result("VXFR", 4);
    AppendSplitByte(result, VORTEX_SPLIT_PAYLOAD_VERSION);
    AppendSplitString(result, scan_split_set_id);
    AppendSplitU64(result, fragments.size());
    for (const auto &fragment : fragments) {
        if (fragment.file_index >= distributed_files.size()) {
            throw InternalException("Cannot encode unknown distributed Vortex file index %llu",
                                    static_cast<unsigned long long>(fragment.file_index));
        }
        if (fragment.row_start > fragment.row_end || fragment.estimated_bytes == DConstants::INVALID_INDEX) {
            throw InternalException("Cannot encode an invalid distributed Vortex fragment range or estimate");
        }
        const auto &file = distributed_files[fragment.file_index];
        if (fragment.estimated_bytes > file.size) {
            throw InternalException("Cannot encode a distributed Vortex fragment larger than its file");
        }
        AppendSplitU64(result, fragment.file_index);
        AppendSplitString(result, file.source_url);
        AppendSplitString(result, file.path);
        AppendSplitU64(result, file.size);
        AppendSplitU64(result, fragment.row_start);
        AppendSplitU64(result, fragment.row_end);
        AppendSplitU64(result, fragment.estimated_bytes);
    }
    return result;
}

class VortexSplitDecoder {
public:
    explicit VortexSplitDecoder(const string &payload_p) : payload(payload_p) {
    }

    uint8_t ReadByte() {
        if (offset >= payload.size()) {
            throw InvalidInputException("Truncated distributed Vortex split payload");
        }
        return static_cast<uint8_t>(payload[offset++]);
    }

    uint64_t ReadU64() {
        uint64_t result = 0;
        for (idx_t byte_index = 0; byte_index < sizeof(result); byte_index++) {
            result |= static_cast<uint64_t>(ReadByte()) << (byte_index * 8U);
        }
        return result;
    }

    string ReadString() {
        auto size = ReadU64();
        if (size > payload.size() - offset) {
            throw InvalidInputException("Invalid string length in distributed Vortex split payload");
        }
        auto result = payload.substr(offset, size);
        offset += size;
        return result;
    }

    void Finish() const {
        if (offset != payload.size()) {
            throw InvalidInputException("Distributed Vortex split payload contains trailing bytes");
        }
    }

    idx_t RemainingBytes() const {
        return payload.size() - offset;
    }

private:
    const string &payload;
    idx_t offset = 0;
};

struct DecodedVortexFragment {
    idx_t file_index;
    VortexBindData::DistributedFile file;
    idx_t row_start;
    idx_t row_end;
    idx_t estimated_bytes;
};

struct DecodedVortexSplit {
    string scan_split_set_id;
    vector<DecodedVortexFragment> fragments;
};

static DecodedVortexSplit DecodeVortexSplit(const string &payload) {
    if (payload.size() < 5 || payload.compare(0, 4, "VXFR") != 0) {
        throw InvalidInputException("Invalid distributed Vortex split payload magic");
    }
    VortexSplitDecoder decoder(payload);
    for (idx_t magic_index = 0; magic_index < 4; magic_index++) {
        decoder.ReadByte();
    }
    auto version = decoder.ReadByte();
    if (version != VORTEX_SPLIT_PAYLOAD_VERSION) {
        throw InvalidInputException("Unsupported distributed Vortex split payload version %u", version);
    }
    DecodedVortexSplit result;
    result.scan_split_set_id = decoder.ReadString();
    if (!IsCanonicalVortexScanId(result.scan_split_set_id)) {
        throw InvalidInputException("Distributed Vortex split contains an invalid scan identity");
    }
    auto fragment_count = decoder.ReadU64();
    if (fragment_count == 0 ||
        fragment_count > decoder.RemainingBytes() / VORTEX_MIN_FRAGMENT_PAYLOAD_BYTES) {
        throw InvalidInputException("Invalid fragment count in distributed Vortex split payload");
    }
    result.fragments.reserve(fragment_count);
    for (idx_t fragment_offset = 0; fragment_offset < fragment_count; fragment_offset++) {
        DecodedVortexFragment decoded;
        decoded.file_index = decoder.ReadU64();
        decoded.file.source_url = decoder.ReadString();
        decoded.file.path = decoder.ReadString();
        decoded.file.size = decoder.ReadU64();
        decoded.row_start = decoder.ReadU64();
        decoded.row_end = decoder.ReadU64();
        decoded.estimated_bytes = decoder.ReadU64();
        if (decoded.file.size == DConstants::INVALID_INDEX) {
            throw InvalidInputException("Distributed Vortex split contains an invalid file size");
        }
        if (decoded.row_start > decoded.row_end || decoded.estimated_bytes == DConstants::INVALID_INDEX ||
            decoded.estimated_bytes > decoded.file.size) {
            throw InvalidInputException("Distributed Vortex split contains an invalid fragment range");
        }
        if (decoded.file.source_url.empty() || !IsCanonicalVortexFilePath(decoded.file.path)) {
            throw InvalidInputException("Distributed Vortex split contains an invalid file identity");
        }
        result.fragments.push_back(std::move(decoded));
    }
    decoder.Finish();
    return result;
}

static string CanonicalVortexSplitId(const vector<VortexBindData::DistributedFragment> &fragments) {
    string result;
    for (const auto &fragment : fragments) {
        if (!result.empty()) {
            result += ',';
        }
        result += std::to_string(fragment.file_index);
        result += ':';
        result += std::to_string(fragment.row_start);
        result += '-';
        result += std::to_string(fragment.row_end);
    }
    return result;
}

static idx_t SaturatingVortexSplitEstimate(idx_t left, idx_t right) {
    // idx_t(-1) is reserved by optional_idx as its invalid sentinel.
    const auto maximum = NumericLimits<idx_t>::Maximum() - 1;
    return right > maximum - left ? maximum : left + right;
}

static bool SameDistributedFile(const VortexBindData::DistributedFile &left,
                                const VortexBindData::DistributedFile &right) {
    return left.source_url == right.source_url && left.path == right.path && left.size == right.size;
}

static bool IsCompleteAggregateVortexAssignment(const vector<VortexBindData::DistributedFragment> &fragments,
                                                const vector<idx_t> &eligible_file_indexes,
                                                const vector<VortexBindData::DistributedFile> &files) {
    if (fragments.size() != eligible_file_indexes.size()) {
        return false;
    }
    for (idx_t fragment_index = 0; fragment_index < fragments.size(); fragment_index++) {
        const auto file_index = eligible_file_indexes[fragment_index];
        const auto &fragment = fragments[fragment_index];
        if (file_index >= files.size() || fragment.file_index != file_index || fragment.row_start != 0 ||
            fragment.estimated_bytes != files[file_index].size) {
            return false;
        }
    }
    // The worker checks row_end against the immutable file's actual row count when it opens the
    // reader. Keeping that storage-dependent check there avoids reopening every aggregate file
    // while applying or deserializing owned split state.
    return true;
}

static void ValidatePortableVortexBind(const string &portable_bind,
                                       const vector<VortexBindData::DistributedFile> &files,
                                       bool aggregate_scan,
                                       const vector<LogicalType> &types,
                                       const vector<string> &names) {
    duckdb_vx_error error_out = nullptr;
    auto decoded_data = duckdb_table_function_distributed_bind_deserialize(
        reinterpret_cast<const uint8_t *>(portable_bind.data()),
        portable_bind.size(),
        &error_out);
    if (error_out) {
        throw SerializationException(IntoErrString(error_out));
    }
    if (!decoded_data) {
        throw SerializationException("Vortex failed to decode distributed bind state");
    }
    auto decoded = unique_ptr<CData>(reinterpret_cast<CData *>(decoded_data));
    if (duckdb_table_function_distributed_is_aggregate(decoded->DataPtr()) != aggregate_scan ||
        duckdb_table_function_distributed_file_count(decoded->DataPtr()) != files.size()) {
        throw SerializationException("Serialized Vortex bind metadata does not match its portable state");
    }
    const auto field_count = duckdb_table_function_distributed_field_count(decoded->DataPtr());
    if (field_count != types.size() || field_count != names.size()) {
        throw SerializationException("Serialized Vortex schema does not match its portable state");
    }
    for (idx_t field_index = 0; field_index < field_count; field_index++) {
        VortexDistributedFieldView view {};
        if (!duckdb_table_function_distributed_field_at(decoded->DataPtr(), field_index, &view) ||
            !view.name || !view.logical_type ||
            string(reinterpret_cast<const char *>(view.name), view.name_len) != names[field_index] ||
            *reinterpret_cast<LogicalType *>(view.logical_type) != types[field_index]) {
            throw SerializationException("Serialized Vortex field metadata differs at index %llu",
                                         static_cast<unsigned long long>(field_index));
        }
    }
    for (idx_t file_index = 0; file_index < files.size(); file_index++) {
        VortexDistributedFileView view {};
        if (!duckdb_table_function_distributed_file_at(decoded->DataPtr(), file_index, &view) ||
            !view.source_url || !view.path || view.size == DConstants::INVALID_INDEX) {
            throw SerializationException("Portable Vortex bind contains an invalid file at index %llu",
                                         static_cast<unsigned long long>(file_index));
        }
        VortexBindData::DistributedFile portable_file;
        portable_file.source_url.assign(reinterpret_cast<const char *>(view.source_url), view.source_url_len);
        portable_file.path.assign(reinterpret_cast<const char *>(view.path), view.path_len);
        portable_file.size = view.size;
        if (!SameDistributedFile(portable_file, files[file_index])) {
            throw SerializationException("Serialized Vortex file metadata differs from its portable bind");
        }
    }
}

static void VortexScanSerialize(Serializer &serializer,
                                const optional_ptr<FunctionData> bind_data,
                                const TableFunction &) {
    if (!bind_data) {
        throw SerializationException("Distributed Vortex scan requires bind data");
    }
    auto &data = bind_data->Cast<VortexBindData>();
    if (!IsCanonicalVortexScanId(data.scan_split_set_id)) {
        throw SerializationException("Distributed Vortex bind has an invalid scan identity");
    }
    auto snapshot = data.CreatePortableSnapshot();
    vector<string> source_urls;
    vector<string> paths;
    vector<uint64_t> sizes;
    source_urls.reserve(snapshot.distributed_files.size());
    paths.reserve(snapshot.distributed_files.size());
    sizes.reserve(snapshot.distributed_files.size());
    for (const auto &file : snapshot.distributed_files) {
        if (file.source_url.empty() || file.source_url.find('\0') != string::npos ||
            !IsCanonicalVortexFilePath(file.path) || file.size == DConstants::INVALID_INDEX) {
            throw SerializationException("Distributed Vortex bind contains an invalid file identity");
        }
        source_urls.push_back(file.source_url);
        paths.push_back(file.path);
        sizes.push_back(file.size);
    }
    serializer.WriteProperty(99, "format_version", VORTEX_BIND_SERDE_VERSION);
    serializer.WriteProperty(100, "types", data.types);
    serializer.WriteProperty(101, "names", data.names);
    serializer.WriteProperty(102, "portable_bind", snapshot.portable_bind);
    serializer.WriteProperty(103, "source_urls", source_urls);
    serializer.WriteProperty(104, "paths", paths);
    serializer.WriteProperty(105, "sizes", sizes);
    serializer.WriteProperty(106, "scan_split_set_id", data.scan_split_set_id);
    serializer.WriteProperty(107, "explicit_split_mode", data.explicit_split_mode);
    serializer.WriteProperty(108, "splits_applied", data.splits_applied);
    vector<idx_t> assigned_file_indexes;
    vector<idx_t> assigned_row_starts;
    vector<idx_t> assigned_row_ends;
    vector<idx_t> assigned_estimated_bytes;
    assigned_file_indexes.reserve(data.assigned_fragments.size());
    assigned_row_starts.reserve(data.assigned_fragments.size());
    assigned_row_ends.reserve(data.assigned_fragments.size());
    assigned_estimated_bytes.reserve(data.assigned_fragments.size());
    for (const auto &fragment : data.assigned_fragments) {
        assigned_file_indexes.push_back(fragment.file_index);
        assigned_row_starts.push_back(fragment.row_start);
        assigned_row_ends.push_back(fragment.row_end);
        assigned_estimated_bytes.push_back(fragment.estimated_bytes);
    }
    serializer.WriteProperty(109, "assigned_file_indexes", assigned_file_indexes);
    serializer.WriteProperty(110, "assigned_row_starts", assigned_row_starts);
    serializer.WriteProperty(111, "assigned_row_ends", assigned_row_ends);
    serializer.WriteProperty(112, "assigned_estimated_bytes", assigned_estimated_bytes);
    serializer.WriteProperty(113, "aggregate_scan", snapshot.aggregate_scan);
    serializer.WriteProperty(114, "eligible_file_indexes", data.eligible_file_indexes);
}

static unique_ptr<FunctionData> VortexScanDeserialize(Deserializer &deserializer, TableFunction &) {
    auto format_version = deserializer.ReadProperty<uint8_t>(99, "format_version");
    if (format_version != VORTEX_BIND_SERDE_VERSION) {
        throw SerializationException("Unsupported distributed Vortex bind format version %u", format_version);
    }
    auto types = deserializer.ReadProperty<vector<LogicalType>>(100, "types");
    auto names = deserializer.ReadProperty<vector<string>>(101, "names");
    auto portable_bind = deserializer.ReadProperty<string>(102, "portable_bind");
    auto source_urls = deserializer.ReadProperty<vector<string>>(103, "source_urls");
    auto paths = deserializer.ReadProperty<vector<string>>(104, "paths");
    auto sizes = deserializer.ReadProperty<vector<uint64_t>>(105, "sizes");
    auto scan_split_set_id = deserializer.ReadProperty<string>(106, "scan_split_set_id");
    auto explicit_split_mode = deserializer.ReadProperty<bool>(107, "explicit_split_mode");
    auto splits_applied = deserializer.ReadProperty<bool>(108, "splits_applied");
    auto assigned_file_indexes = deserializer.ReadProperty<vector<idx_t>>(109, "assigned_file_indexes");
    auto assigned_row_starts = deserializer.ReadProperty<vector<idx_t>>(110, "assigned_row_starts");
    auto assigned_row_ends = deserializer.ReadProperty<vector<idx_t>>(111, "assigned_row_ends");
    auto assigned_estimated_bytes = deserializer.ReadProperty<vector<idx_t>>(112, "assigned_estimated_bytes");
    auto aggregate_scan = deserializer.ReadProperty<bool>(113, "aggregate_scan");
    auto eligible_file_indexes = deserializer.ReadProperty<vector<idx_t>>(114, "eligible_file_indexes");
    if (types.size() != names.size() || portable_bind.empty() ||
        !IsCanonicalVortexScanId(scan_split_set_id) || source_urls.size() != paths.size() ||
        source_urls.size() != sizes.size()) {
        throw SerializationException("Invalid serialized Vortex bind state");
    }
    vector<VortexBindData::DistributedFile> files;
    files.reserve(paths.size());
    for (idx_t file_index = 0; file_index < paths.size(); file_index++) {
        if (source_urls[file_index].empty() || source_urls[file_index].find('\0') != string::npos ||
            !IsCanonicalVortexFilePath(paths[file_index]) || sizes[file_index] == DConstants::INVALID_INDEX) {
            throw SerializationException("Invalid serialized Vortex file at index %llu",
                                         static_cast<unsigned long long>(file_index));
        }
        VortexBindData::DistributedFile file;
        file.source_url = std::move(source_urls[file_index]);
        file.path = std::move(paths[file_index]);
        file.size = sizes[file_index];
        files.push_back(std::move(file));
    }
    ValidatePortableVortexBind(portable_bind, files, aggregate_scan, types, names);
    unordered_set<idx_t> eligible;
    optional_idx previous_eligible;
    for (auto file_index : eligible_file_indexes) {
        if (file_index >= files.size() || !eligible.insert(file_index).second ||
            (previous_eligible.IsValid() && previous_eligible.GetIndex() >= file_index)) {
            throw SerializationException("Invalid eligible Vortex file index %llu",
                                         static_cast<unsigned long long>(file_index));
        }
        previous_eligible = optional_idx(file_index);
    }
    if (assigned_file_indexes.size() != assigned_row_starts.size() ||
        assigned_file_indexes.size() != assigned_row_ends.size() ||
        assigned_file_indexes.size() != assigned_estimated_bytes.size()) {
        throw SerializationException("Serialized Vortex fragment vectors have different lengths");
    }
    vector<VortexBindData::DistributedFragment> assigned_fragments;
    assigned_fragments.reserve(assigned_file_indexes.size());
    bool has_previous_fragment = false;
    idx_t previous_file_index = 0;
    idx_t previous_row_start = 0;
    idx_t previous_row_end = 0;
    for (idx_t fragment_index = 0; fragment_index < assigned_file_indexes.size(); fragment_index++) {
        const auto file_index = assigned_file_indexes[fragment_index];
        const auto row_start = assigned_row_starts[fragment_index];
        const auto row_end = assigned_row_ends[fragment_index];
        const auto estimated_bytes = assigned_estimated_bytes[fragment_index];
        const bool same_file = has_previous_fragment && previous_file_index == file_index;
        if (file_index >= files.size() || !eligible.count(file_index) || row_start > row_end ||
            estimated_bytes == DConstants::INVALID_INDEX || estimated_bytes > files[file_index].size ||
            (has_previous_fragment && previous_file_index > file_index) ||
            (same_file && previous_row_start >= row_start) || (same_file && previous_row_end > row_start)) {
            throw SerializationException("Invalid assigned Vortex fragment at index %llu",
                                         static_cast<unsigned long long>(fragment_index));
        }
        assigned_fragments.push_back({file_index, row_start, row_end, estimated_bytes});
        has_previous_fragment = true;
        previous_file_index = file_index;
        previous_row_start = row_start;
        previous_row_end = row_end;
    }
    if (!explicit_split_mode &&
        (splits_applied || !eligible_file_indexes.empty() || !assigned_fragments.empty())) {
        throw SerializationException("Native Vortex bind contains distributed split state");
    }
    if (!splits_applied && !assigned_fragments.empty()) {
        throw SerializationException(
            "Detached Vortex bind contains assigned fragments without an applied split batch");
    }
    if (aggregate_scan && splits_applied && !assigned_fragments.empty() &&
        !IsCompleteAggregateVortexAssignment(assigned_fragments, eligible_file_indexes, files)) {
        throw SerializationException(
            "Distributed aggregate Vortex bind contains an incomplete file assignment");
    }

    auto result = make_uniq<VortexBindData>(nullptr, types, names);
    result->scan_split_set_id = std::move(scan_split_set_id);
    result->portable_bind = std::move(portable_bind);
    result->distributed_files = std::move(files);
    result->aggregate_scan = aggregate_scan;
    result->explicit_split_mode = explicit_split_mode;
    result->splits_applied = splits_applied;
    result->eligible_file_indexes = std::move(eligible_file_indexes);
    result->assigned_fragments = std::move(assigned_fragments);
    return result;
}

static vector<idx_t> SelectDistributedVortexFiles(const TableFunctionDistributedScanInput &input,
                                                  idx_t file_count) {
    vector<idx_t> column_ids;
    column_ids.reserve(input.column_ids.size());
    for (const auto &column_id : input.column_ids) {
        column_ids.push_back(column_id.GetPrimaryIndex());
    }
    auto filters =
        reinterpret_cast<duckdb_vx_table_filter_set>(const_cast<TableFilterSet *>(input.table_filters.get()));
    vector<idx_t> selected_file_indexes;
    selected_file_indexes.reserve(file_count);
    for (idx_t file_index = 0; file_index < file_count; file_index++) {
        duckdb_vx_error error_out = nullptr;
        auto selected = duckdb_table_function_distributed_file_is_selected(filters,
                                                                           column_ids.data(),
                                                                           column_ids.size(),
                                                                           file_index,
                                                                           &error_out);
        if (error_out) {
            throw InvalidInputException(IntoErrString(error_out));
        }
        if (selected) {
            selected_file_indexes.push_back(file_index);
        }
    }
    return selected_file_indexes;
}

static const VortexBindData &
RequireDistributedVortexBindData(const TableFunctionDistributedScanInput &input) {
    if (!input.bind_data) {
        throw InvalidInputException("Distributed Vortex scan requires table-function bind data");
    }
    return input.bind_data->Cast<VortexBindData>();
}

static vector<DistributedScanSplit>
VortexPlanDistributedScanSplits(const TableFunctionDistributedScanPlanningInput &input) {
    auto &bind_data = RequireDistributedVortexBindData(input);
    if (bind_data.explicit_split_mode) {
        throw InvalidInputException("Distributed Vortex splits cannot be planned from a worker bind");
    }
    // Vane may serialize a logical coordinator plan before generating its
    // physical scan. Such a bind is intentionally detached from the original
    // connection, but its owned portable state and immutable file identities
    // are sufficient for deterministic split planning.
    auto snapshot = bind_data.CreatePortableSnapshot();
    auto selected_file_indexes = SelectDistributedVortexFiles(input, snapshot.distributed_files.size());
    vector<DistributedScanSplit> result;
    if (selected_file_indexes.empty()) {
        return result;
    }
    const auto fragments =
        PlanVortexFragments(snapshot.portable_bind,
                            selected_file_indexes,
                            snapshot.aggregate_scan ? selected_file_indexes.size()
                                                    : MaxValue<idx_t>(input.target_split_count, 1));
    if (fragments.empty()) {
        throw InvalidInputException("Vortex produced no fragments for a non-empty distributed scan");
    }
    if (snapshot.aggregate_scan) {
        idx_t total_bytes = 0;
        for (const auto &fragment : fragments) {
            total_bytes = SaturatingVortexSplitEstimate(total_bytes, fragment.estimated_bytes);
        }
        DistributedScanSplit split;
        split.split_id = CanonicalVortexSplitId(fragments);
        split.payload = EncodeVortexSplit(fragments, snapshot.distributed_files, bind_data.scan_split_set_id);
        split.estimated_bytes = optional_idx(total_bytes);
        split.estimated_cardinality = optional_idx(1);
        result.push_back(std::move(split));
        return result;
    }
    result.reserve(fragments.size());
    for (const auto &fragment : fragments) {
        vector<VortexBindData::DistributedFragment> split_fragments {fragment};
        DistributedScanSplit split;
        split.split_id = CanonicalVortexSplitId(split_fragments);
        split.payload =
            EncodeVortexSplit(split_fragments, snapshot.distributed_files, bind_data.scan_split_set_id);
        split.estimated_bytes = optional_idx(fragment.estimated_bytes);
        split.estimated_cardinality = optional_idx(fragment.row_end - fragment.row_start);
        result.push_back(std::move(split));
    }
    return result;
}

static unique_ptr<FunctionData>
VortexCreateDistributedWorkerBind(const TableFunctionDistributedScanInput &input) {
    auto &coordinator_bind = RequireDistributedVortexBindData(input);
    if (coordinator_bind.explicit_split_mode) {
        throw InvalidInputException(
            "Distributed Vortex worker bind cannot be created from another worker bind");
    }
    auto snapshot = coordinator_bind.CreatePortableSnapshot();
    auto eligible_file_indexes = SelectDistributedVortexFiles(input, snapshot.distributed_files.size());
    auto worker_bind = make_uniq<VortexBindData>(nullptr, coordinator_bind.types, coordinator_bind.names);
    worker_bind->scan_split_set_id = coordinator_bind.scan_split_set_id;
    worker_bind->portable_bind = std::move(snapshot.portable_bind);
    worker_bind->distributed_files = std::move(snapshot.distributed_files);
    worker_bind->aggregate_scan = snapshot.aggregate_scan;
    worker_bind->explicit_split_mode = true;
    worker_bind->splits_applied = false;
    worker_bind->eligible_file_indexes = std::move(eligible_file_indexes);
    return worker_bind;
}

static void VortexApplyDistributedSplits(optional_ptr<FunctionData> worker_bind_data,
                                         const vector<DistributedScanSplit> &splits) {
    if (!worker_bind_data) {
        throw InvalidInputException("Distributed Vortex scan requires worker bind data");
    }
    auto &bind_data = worker_bind_data->Cast<VortexBindData>();
    if (bind_data.ffi_data || !bind_data.explicit_split_mode) {
        throw InvalidInputException("Vortex distributed splits require a detached worker bind");
    }
    if (bind_data.aggregate_scan && splits.size() > 1) {
        throw InvalidInputException("Distributed aggregate Vortex scans require one complete file-set split");
    }
    unordered_set<string> split_ids;
    unordered_set<idx_t> eligible_file_indexes(bind_data.eligible_file_indexes.begin(),
                                               bind_data.eligible_file_indexes.end());
    vector<VortexBindData::DistributedFragment> assigned;
    for (const auto &split : splits) {
        split.Validate();
        if (split.payload.empty()) {
            throw InvalidInputException("Distributed Vortex split '%s' has an empty payload", split.split_id);
        }
        if (!split_ids.insert(split.split_id).second) {
            throw InvalidInputException("Duplicate distributed Vortex split id '%s'", split.split_id);
        }
        auto decoded = DecodeVortexSplit(split.payload);
        if (decoded.scan_split_set_id != bind_data.scan_split_set_id) {
            throw InvalidInputException("Distributed Vortex split '%s' belongs to a different scan identity",
                                        split.split_id);
        }
        if (!bind_data.aggregate_scan && decoded.fragments.size() != 1) {
            throw InvalidInputException(
                "Non-aggregate distributed Vortex splits must reference exactly one fragment");
        }
        vector<VortexBindData::DistributedFragment> decoded_fragments;
        decoded_fragments.reserve(decoded.fragments.size());
        for (const auto &decoded_fragment : decoded.fragments) {
            if (decoded_fragment.file_index >= bind_data.distributed_files.size()) {
                throw InvalidInputException("Distributed Vortex split '%s' references an unknown file index",
                                            split.split_id);
            }
            if (!eligible_file_indexes.count(decoded_fragment.file_index)) {
                throw InvalidInputException(
                    "Distributed Vortex split '%s' references file index %llu outside the planned file set",
                    split.split_id,
                    static_cast<unsigned long long>(decoded_fragment.file_index));
            }
            if (!SameDistributedFile(decoded_fragment.file,
                                     bind_data.distributed_files[decoded_fragment.file_index])) {
                throw InvalidInputException(
                    "Distributed Vortex split '%s' does not match the bound file identity",
                    split.split_id);
            }
            decoded_fragments.push_back({decoded_fragment.file_index,
                                         decoded_fragment.row_start,
                                         decoded_fragment.row_end,
                                         decoded_fragment.estimated_bytes});
        }
        if (split.split_id != CanonicalVortexSplitId(decoded_fragments)) {
            throw InvalidInputException(
                "Distributed Vortex split id '%s' does not match its fragment payload",
                split.split_id);
        }
        if (!bind_data.aggregate_scan &&
            (!split.estimated_cardinality.IsValid() || !split.estimated_bytes.IsValid() ||
             split.estimated_cardinality.GetIndex() !=
                 decoded_fragments[0].row_end - decoded_fragments[0].row_start ||
             split.estimated_bytes.GetIndex() != decoded_fragments[0].estimated_bytes)) {
            throw InvalidInputException(
                "Distributed Vortex split '%s' estimates do not match its fragment payload",
                split.split_id);
        }
        assigned.insert(assigned.end(), decoded_fragments.begin(), decoded_fragments.end());
    }
    // A batch is a set of elementary scan fragments. Canonicalize its assignment
    // so transport or retry code may reorder splits without changing scan meaning.
    std::sort(assigned.begin(), assigned.end(), [](const auto &left, const auto &right) {
        if (left.file_index != right.file_index) {
            return left.file_index < right.file_index;
        }
        if (left.row_start != right.row_start) {
            return left.row_start < right.row_start;
        }
        return left.row_end < right.row_end;
    });
    for (idx_t fragment_index = 1; fragment_index < assigned.size(); fragment_index++) {
        const auto &previous = assigned[fragment_index - 1];
        const auto &current = assigned[fragment_index];
        if (previous.file_index == current.file_index &&
            (previous.row_start >= current.row_start || previous.row_end > current.row_start)) {
            throw InvalidInputException(
                "Distributed Vortex fragment assignment overlaps within file index %llu",
                static_cast<unsigned long long>(current.file_index));
        }
    }
    if (bind_data.aggregate_scan && !splits.empty()) {
        if (!IsCompleteAggregateVortexAssignment(assigned,
                                                 bind_data.eligible_file_indexes,
                                                 bind_data.distributed_files)) {
            throw InvalidInputException(
                "Distributed aggregate Vortex split does not contain the complete planned file set");
        }
        idx_t expected_bytes = 0;
        for (const auto &fragment : assigned) {
            expected_bytes = SaturatingVortexSplitEstimate(expected_bytes, fragment.estimated_bytes);
        }
        if (!splits[0].estimated_cardinality.IsValid() || !splits[0].estimated_bytes.IsValid() ||
            splits[0].estimated_cardinality.GetIndex() != 1 ||
            splits[0].estimated_bytes.GetIndex() != expected_bytes) {
            throw InvalidInputException(
                "Distributed aggregate Vortex split estimates do not match its fragment payload");
        }
    }
    if (bind_data.splits_applied) {
        if (assigned != bind_data.assigned_fragments) {
            throw InvalidInputException(
                "Distributed Vortex bind already has a different explicit split assignment");
        }
        return;
    }
    bind_data.assigned_fragments = std::move(assigned);
    bind_data.splits_applied = true;
}

static TableFunctionDistributedScanCallbacks VortexDistributedScanCallbacks() {
    TableFunctionDistributedScanCallbacks callbacks;
    callbacks.protocol_version = 1;
    callbacks.split_codec = {VORTEX_SPLIT_CODEC, 1};
    callbacks.bind_data_mode = TableFunctionDistributedBindDataMode::REQUIRED;
    callbacks.plan_splits = VortexPlanDistributedScanSplits;
    callbacks.create_worker_bind = VortexCreateDistributedWorkerBind;
    callbacks.apply_splits = VortexApplyDistributedSplits;
    return callbacks;
}

} // namespace
#endif

#ifdef VORTEX_VANE_DISTRIBUTED
static TableFunction CreateDistributedVortexTableFunction(LogicalType parameter, const std::string &name) {
    TableFunction tf(name, {}, function, duckdb_vx_table_function_bind, init_global, init_local);

    tf.projection_pushdown = true;
    tf.filter_pushdown = true;
    tf.filter_prune = true;
    tf.sampling_pushdown = false;
    tf.supports_pushdown_type = [](const FunctionData &, idx_t column_id) {
        // file_index is constant per partition and is evaluated exactly before
        // readers are created. Other virtual-column filters stay in a DuckDB
        // PhysicalFilter so unsupported predicates cannot be silently ignored.
        return !IsVirtualColumn(column_id) || column_id == COLUMN_IDENTIFIER_FILE_INDEX;
    };

    tf.pushdown_expression = [](auto &, const auto &, Expression &expression) {
        return duckdb_table_function_pushdown_expression(reinterpret_cast<duckdb_vx_expr>(&expression));
    };
    tf.pushdown_complex_filter = [](auto &, auto &, FunctionData *bind_data, FilterVec &filters) {
        pushdown_complex_filter(*bind_data, filters);
    };
    tf.cardinality = cardinality;
    tf.get_partition_info = get_partition_info;
    tf.get_partition_data = get_partition_data;
    tf.to_string = to_string;
    tf.table_scan_progress = table_scan_progress;
    tf.statistics = statistics;

    tf.serialize = VortexScanSerialize;
    tf.deserialize = VortexScanDeserialize;
    tf.SetDistributedScanCallbacks(VortexDistributedScanCallbacks());

    // DuckDB's late-materialization rewrite duplicates the table scan and
    // joins the copies by their virtual row identifiers. Vane assigns explicit
    // splits to every physical scan independently, so the two sides are not a
    // co-partitioned unit and can otherwise observe disjoint file assignments.
    // Keep the distributed-capable function as one explicit scan until Vane
    // has a grouped multi-scan split contract.
    tf.late_materialization = false;
    // Columns that uniquely identify a row for deferred re-fetch in a multi
    // file scan: (file index, row number in file).
    tf.get_row_id_columns = [](auto &, auto) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    tf.get_virtual_columns = [](auto &, auto) -> virtual_column_map_t {
        return {
            {COLUMN_IDENTIFIER_EMPTY, {"", LogicalTypeId::BOOLEAN}},
            {COLUMN_IDENTIFIER_FILE_INDEX, {"file_index", LogicalType::UBIGINT}},
            // MultiFileReader's file_row_number column is BIGINT.
            // row_idx() is UBIGINT. Use UBIGINT since there's no difference to
            // Duckdb what to compare.
            {COLUMN_IDENTIFIER_FILE_ROW_NUMBER, {"file_row_number", LogicalType::UBIGINT}},
        };
    };

    tf.arguments.resize(1);
    tf.arguments[0] = parameter;

    return tf;
}

void RegisterVortexTableFunctions(ExtensionLoader &loader) {
    for (const std::string &name : {"read_vortex"s, "vortex_scan"s}) {
        TableFunctionSet functions(name);
        functions.AddFunction(CreateDistributedVortexTableFunction(LogicalType::VARCHAR, name));
        functions.AddFunction(
            CreateDistributedVortexTableFunction(LogicalType::LIST(LogicalType::VARCHAR), name));
        loader.RegisterFunction(std::move(functions));
    }
}
#endif

duckdb_state register_table_function(DatabaseInstance &db, LogicalType parameter, const std::string &name) {
    TableFunction tf(name, {}, function, duckdb_vx_table_function_bind, init_global, init_local);

    tf.projection_pushdown = true;
    tf.filter_pushdown = true;
    tf.filter_prune = true;
    tf.sampling_pushdown = false;

    tf.pushdown_expression = [](auto &, const auto &, Expression &expression) {
        return duckdb_table_function_pushdown_expression(reinterpret_cast<duckdb_vx_expr>(&expression));
    };
    tf.pushdown_complex_filter = [](auto &, auto &, FunctionData *bind_data, FilterVec &filters) {
        pushdown_complex_filter(*bind_data, filters);
    };
    tf.cardinality = cardinality;
    tf.get_partition_info = get_partition_info;
    tf.get_partition_data = get_partition_data;
    tf.to_string = to_string;
    tf.table_scan_progress = table_scan_progress;
    tf.statistics = statistics;

    tf.late_materialization = true;
    // Columns that uniquely identify a row for deferred re-fetch in a multi
    // file scan: (file index, row number in file).
    tf.get_row_id_columns = [](auto &, auto) -> vector<column_t> {
        return {COLUMN_IDENTIFIER_FILE_INDEX, COLUMN_IDENTIFIER_FILE_ROW_NUMBER};
    };

    tf.get_virtual_columns = [](auto &, auto) -> virtual_column_map_t {
        return {
            {COLUMN_IDENTIFIER_EMPTY, {"", LogicalTypeId::BOOLEAN}},
            {COLUMN_IDENTIFIER_FILE_INDEX, {"file_index", LogicalType::UBIGINT}},
            // MultiFileReader's file_row_number column is BIGINT.
            // row_idx() is UBIGINT. Use UBIGINT since there's no difference to
            // Duckdb what to compare.
            {COLUMN_IDENTIFIER_FILE_ROW_NUMBER, {"file_row_number", LogicalType::UBIGINT}},
        };
    };

    tf.arguments.resize(1);
    tf.arguments[0] = parameter;

    try {
        auto &system_catalog = Catalog::GetSystemCatalog(db);
        auto data = CatalogTransaction::GetSystemTransaction(db);
        CreateTableFunctionInfo tf_info(tf);
        tf_info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
        system_catalog.CreateFunction(data, tf_info);
    } catch (const std::exception &e) {
        ErrorData data(e);
        DUCKDB_LOG_ERROR(db, "Failed to create Vortex table function:\t" + data.Message());
        return DuckDBError;
    }
    return DuckDBSuccess;
}

extern "C" duckdb_state duckdb_vx_register_table_functions(duckdb_database ffi_db) {
    D_ASSERT(ffi_db);
    const DatabaseWrapper &wrapper = *reinterpret_cast<DatabaseWrapper *>(ffi_db);
    DatabaseInstance &db = *wrapper.database->instance;

    for (LogicalType type : {LogicalType(LogicalType::VARCHAR), LogicalType::LIST(LogicalType::VARCHAR)}) {
        for (const std::string &name : {"read_vortex"s, "vortex_scan"s}) {
            if (register_table_function(db, type, name) == DuckDBError) {
                return DuckDBError;
            }
        }
    }
    return DuckDBSuccess;
}
