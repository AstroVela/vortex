// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// This is UB but no other way to test compatibility macro, fine
// as we're running tests
#undef __STDCPP_FLOAT16_T__
#include <catch2/catch_test_macros.hpp>
#include <vortex/array.hpp>

using namespace vortex;

namespace {
TEST_CASE("float16_t to_float (compatibility)", "[float]") {
    REQUIRE(float(float16_t {0x3C00}) == 1.0F);
    REQUIRE(float(float16_t {0xC000}) == -2.0F);
    REQUIRE(float(float16_t {0}) == 0.0F);
}

TEST_CASE("F16 scalar (compatibility)", "[scalar]") {
    const float16_t float16t {0x3C00};
    Scalar scalar = scalar::of(float16t);
    REQUIRE(scalar.dtype().variant() == DataTypeVariant::Primitive);
    REQUIRE(scalar.dtype().primitive_type() == vortex::PType::F16);
}
} // namespace
