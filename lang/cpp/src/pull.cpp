// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/pull.hpp"

#include "vortex/error.hpp"

#include <vortex.h>

#include <utility>

namespace vortex {

using detail::Access;
using detail::throw_on_error;

void Footer::Deleter::operator()(vx_footer *ptr) const noexcept {
    vx_footer_free(ptr);
}

DataType Footer::dtype() const {
    return Access::adopt<DataType>(vx_footer_dtype(handle_.get()));
}

uint64_t Footer::row_count() const {
    return vx_footer_row_count(handle_.get());
}

std::vector<uint64_t> Footer::split_points(const Session &session,
                                           const std::optional<Expression> &projection,
                                           const std::optional<Expression> &filter) const {
    const vx_expression *proj = projection.has_value() ? Access::c_ptr(*projection) : nullptr;
    const vx_expression *filt = filter.has_value() ? Access::c_ptr(*filter) : nullptr;

    std::vector<uint64_t> points(64);
    for (;;) {
        size_t len = 0;
        vx_error *error = nullptr;
        vx_footer_split_points(Access::c_ptr(session), handle_.get(), proj, filt, points.data(),
                               points.size(), &len, &error);
        throw_on_error(error);
        if (len <= points.size()) {
            points.resize(len);
            return points;
        }
        points.resize(len);
    }
}

void PullFooter::Deleter::operator()(vx_pull_footer *ptr) const noexcept {
    vx_pull_footer_free(ptr);
}

PullFooter::PullFooter(const Session &session, uint64_t file_size) {
    vx_error *error = nullptr;
    vx_pull_footer *handle = vx_pull_footer_new(Access::c_ptr(session), file_size, &error);
    throw_on_error(error);
    handle_.reset(handle);
}

std::optional<PullRead> PullFooter::next_read() {
    vx_pull_read read {};
    vx_error *error = nullptr;
    const vx_pull_state state = vx_pull_footer_advance(handle_.get(), &read, &footer_, &error);
    throw_on_error(error);
    if (state == VX_PULL_READS) {
        return PullRead(read);
    }
    return std::nullopt;
}

void PullFooter::complete(const PullRead &read) {
    vx_error *error = nullptr;
    vx_pull_footer_complete(handle_.get(), read.raw_.dst, &error);
    throw_on_error(error);
}

Footer PullFooter::take() && {
    return Access::adopt<Footer>(std::exchange(footer_, nullptr));
}

void PullScan::Deleter::operator()(vx_pull_scan *ptr) const noexcept {
    vx_pull_scan_free(ptr);
}

namespace {
vx_scan_options to_raw_options(const ScanOptions &options) {
    vx_scan_options raw {};
    raw.projection = options.projection.has_value() ? Access::c_ptr(*options.projection) : nullptr;
    raw.filter = options.filter.has_value() ? Access::c_ptr(*options.filter) : nullptr;
    if (options.row_range.has_value()) {
        raw.row_range_begin = options.row_range->begin;
        raw.row_range_end = options.row_range->end;
    }
    if (options.selection.has_value()) {
        raw.selection.idx = options.selection->indices.data();
        raw.selection.idx_len = options.selection->indices.size();
        raw.selection.include = static_cast<vx_scan_selection_include>(options.selection->kind);
    } else {
        raw.selection.include = VX_SELECTION_INCLUDE_ALL;
    }
    raw.limit = options.limit;
    raw.ordered = options.ordered;
    return raw;
}
} // namespace

void PullScan::File::Deleter::operator()(vx_pull_file *ptr) const noexcept {
    vx_pull_file_free(ptr);
}

PullScan::File::File(const Session &session, const Footer &footer) {
    vx_error *error = nullptr;
    vx_pull_file *handle = vx_pull_file_new(Access::c_ptr(session), Access::c_ptr(footer), &error);
    throw_on_error(error);
    handle_.reset(handle);
}

PullScan PullScan::File::scan(const ScanOptions &options, uint64_t max_inflight) const {
    const vx_scan_options raw = to_raw_options(options);
    vx_error *error = nullptr;
    vx_pull_scan *handle = vx_pull_file_scan(handle_.get(), &raw, max_inflight, &error);
    throw_on_error(error);
    return PullScan(handle);
}

PullScan::PullScan(const Session &session, const Footer &footer, const ScanOptions &options,
                   uint64_t max_inflight) {
    vx_scan_options raw {};
    raw.projection = options.projection.has_value() ? Access::c_ptr(*options.projection) : nullptr;
    raw.filter = options.filter.has_value() ? Access::c_ptr(*options.filter) : nullptr;
    if (options.row_range.has_value()) {
        raw.row_range_begin = options.row_range->begin;
        raw.row_range_end = options.row_range->end;
    }
    if (options.selection.has_value()) {
        raw.selection.idx = options.selection->indices.data();
        raw.selection.idx_len = options.selection->indices.size();
        raw.selection.include = static_cast<vx_scan_selection_include>(options.selection->kind);
    } else {
        raw.selection.include = VX_SELECTION_INCLUDE_ALL;
    }
    raw.limit = options.limit;
    raw.ordered = options.ordered;

    vx_error *error = nullptr;
    vx_pull_scan *handle = vx_pull_scan_new(Access::c_ptr(session), Access::c_ptr(footer), &raw,
                                            max_inflight, &error);
    throw_on_error(error);
    handle_.reset(handle);
}

std::optional<PullScan::Event> PullScan::advance() {
    const vx_pull_read *reads = nullptr;
    size_t reads_len = 0;
    vx_array *batch = nullptr;
    vx_error *error = nullptr;
    const vx_pull_state state =
        vx_pull_scan_advance(handle_.get(), &reads, &reads_len, &batch, &error);
    throw_on_error(error);
    switch (state) {
    case VX_PULL_READS: {
        Reads out;
        out.reserve(reads_len);
        for (size_t i = 0; i < reads_len; ++i) {
            out.push_back(PullRead(reads[i]));
        }
        return Event(std::move(out));
    }
    case VX_PULL_BATCH:
        return Event(Access::adopt<Array>(batch));
    case VX_PULL_DONE:
    default:
        return std::nullopt;
    }
}

void PullScan::complete(const PullRead &read) {
    vx_error *error = nullptr;
    vx_pull_scan_complete(handle_.get(), read.raw_.dst, &error);
    throw_on_error(error);
}

} // namespace vortex
