// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <vortex/pull.hpp>

#include "common.hpp"

using namespace vortex;
using vortex_test::SAMPLE_ROWS;
using vortex_test::TempPath;
using vortex_test::write_sample;

namespace {

std::vector<char> read_whole_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void serve(const std::vector<char> &bytes, const PullRead &read) {
    const auto dst = read.data();
    std::copy_n(bytes.data() + read.offset(), dst.size(), reinterpret_cast<char *>(dst.data()));
}

Footer pull_footer(const Session &session, const std::vector<char> &bytes) {
    PullFooter pf(session, bytes.size());
    while (auto read = pf.next_read()) {
        serve(bytes, *read);
        pf.complete(*read);
    }
    return std::move(pf).take();
}

TEST_CASE("Pull scan round trip", "[pull]") {
    Session session;
    TempPath path = write_sample(session);
    const std::vector<char> bytes = read_whole_file(path.string());

    Footer footer = pull_footer(session, bytes);
    REQUIRE(footer.row_count() == SAMPLE_ROWS);
    REQUIRE(footer.dtype().variant() == DataTypeVariant::Struct);

    const std::vector<uint64_t> points = footer.split_points(session);
    REQUIRE(points.size() >= 2);
    REQUIRE(points.front() == 0);
    REQUIRE(points.back() == SAMPLE_ROWS);

    PullScan scan(session, footer);
    size_t rows = 0;
    while (auto event = scan.advance()) {
        if (auto *reads = std::get_if<PullScan::Reads>(&*event)) {
            REQUIRE(!reads->empty());
            for (auto &read : *reads) {
                serve(bytes, read);
                scan.complete(read);
            }
        } else {
            rows += std::get<Array>(std::move(*event)).size();
        }
    }
    REQUIRE(rows == SAMPLE_ROWS);
}

TEST_CASE("Pull scan bad complete", "[pull]") {
    Session session;
    TempPath path = write_sample(session);
    const std::vector<char> bytes = read_whole_file(path.string());

    Footer footer = pull_footer(session, bytes);
    PullScan scan(session, footer);
    uint8_t bogus = 0;
    REQUIRE_THROWS_AS(scan.complete(detail::Access::adopt<PullRead>(vx_pull_read {
                          .dst = &bogus,
                          .offset = 0,
                          .len = 1,
                      })),
                      VortexException);
}

} // namespace
