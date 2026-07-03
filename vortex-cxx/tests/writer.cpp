// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#include <catch2/catch_test_macros.hpp>
#include <vortex/scan.hpp>

#include "common.hpp"
#include "vortex/dtype.hpp"
#include "vortex/error.hpp"

using namespace vortex;
using vortex_test::SAMPLE_ROWS;
using vortex_test::TempPath;
using vortex_test::write_sample;

TEST_CASE("Unfinished writer", "[writer]") {
    Session session;
    TempPath path = TempPath::unique();

    {
        Writer writer = Writer::open(session, path.string(), vortex_test::sample_dtype());
        writer.push(vortex_test::sample_array());
    }

    REQUIRE_THROWS_AS(DataSource::open(session, {path.string()}), VortexException);
}

TEST_CASE("Write finish() called twice", "[writer]") {
    Session session;
    TempPath path = TempPath::unique();

    Writer writer = Writer::open(session, path.string(), vortex_test::sample_dtype());
    writer.push(vortex_test::sample_array());
    writer.finish();
    REQUIRE_THROWS_AS(writer.finish(), VortexException);
    REQUIRE_THROWS_AS(writer.push(vortex_test::sample_array()), VortexException);
}

TEST_CASE("Writer push with invalid dtype", "[writer]") {
    Session session;
    TempPath path = TempPath::unique();
    Writer writer = Writer::open(session, path.string(), dtype::null());
    REQUIRE_THROWS_AS(writer.push(vortex_test::sample_array()), VortexException);
    REQUIRE_THROWS_AS(writer.push(vortex_test::sample_array()), VortexException);
}
