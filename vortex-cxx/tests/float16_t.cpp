// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/dtype.hpp"
#include <vortex/scalar.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace vortex;

namespace {
TEST_CASE("float16_t to_float", "[float]") {
    REQUIRE(float(1.0f16) == 1.0f);
    REQUIRE(float(-2.0f16) == -2.0f);
    REQUIRE(float(0.0f16) == 0.0f);
}

TEST_CASE("F16 scalar", "[scalar]") {
    float16_t float16t = 1.0f16;
    Scalar scalar = scalar::of(float16t);
    REQUIRE(scalar.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE(scalar.dtype().primitive_type() == vortex::PType::F16);
}
} // namespace
