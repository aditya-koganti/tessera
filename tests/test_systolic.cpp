// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <random>
#include <tuple>
#include <vector>

#include "tessera/reference.hpp"
#include "tessera/systolic.hpp"

namespace tessera {
namespace {

std::vector<std::int8_t> random_int8(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(-128, 127);
  std::vector<std::int8_t> v(n);
  for (auto& x : v) x = static_cast<std::int8_t>(d(rng));
  return v;
}

// (array rows, array cols, m, k, n)
using Shape = std::tuple<int, int, int, int, int>;
class SystolicShapes : public ::testing::TestWithParam<Shape> {};

TEST_P(SystolicShapes, MatchesGoldenGemmAndAnalyticalTiming) {
  const auto [R, C, m, k, n] = GetParam();
  const auto a = random_int8(static_cast<std::size_t>(m) * k, 1);
  const auto w = random_int8(static_cast<std::size_t>(k) * n, 2);

  SystolicArray arr(R, C);
  std::vector<std::int32_t> out(static_cast<std::size_t>(m) * n, -1);
  const MatMulTiming t = arr.run(a.data(), k, w.data(), n, m, k, n, out.data(), n, false);

  EXPECT_EQ(out, gemm_ref(a, w, m, k, n));
  const MatMulTiming ref = SystolicArray::analytical(R, C, m, k, n);
  EXPECT_EQ(t.total(), ref.total());
  EXPECT_EQ(t.stream_cycles, static_cast<std::int64_t>(m) + n + R - 2);
  EXPECT_EQ(t.useful_macs, static_cast<std::int64_t>(m) * k * n);
  EXPECT_EQ(arr.pe_updates(), t.stream_cycles * R * n);
}

INSTANTIATE_TEST_SUITE_P(Shapes, SystolicShapes,
                         ::testing::Values(Shape{1, 1, 1, 1, 1}, Shape{4, 4, 1, 4, 4}, Shape{4, 4, 7, 3, 2},
                                           Shape{8, 4, 9, 8, 4}, Shape{4, 8, 5, 2, 8}, Shape{16, 16, 33, 16, 16},
                                           Shape{16, 16, 3, 5, 11}, Shape{32, 8, 64, 17, 8}));

TEST(Systolic, AccumulatesAcrossPasses) {
  const int R = 8, C = 8, m = 6, k = 16, n = 5;
  const auto a = random_int8(static_cast<std::size_t>(m) * k, 3);
  const auto w = random_int8(static_cast<std::size_t>(k) * n, 4);
  SystolicArray arr(R, C);
  std::vector<std::int32_t> out(static_cast<std::size_t>(m) * n, 0);
  // K = 16 split into two passes of 8 rows: strides address the sub-blocks in place.
  arr.run(a.data(), k, w.data(), n, m, 8, n, out.data(), n, false);
  arr.run(a.data() + 8, k, w.data() + 8 * n, n, m, 8, n, out.data(), n, true);
  EXPECT_EQ(out, gemm_ref(a, w, m, k, n));
}

TEST(Systolic, RejectsShapesThatDoNotFit) {
  SystolicArray arr(4, 4);
  std::vector<std::int8_t> buf(64, 1);
  std::vector<std::int32_t> out(64, 0);
  EXPECT_THROW(arr.run(buf.data(), 5, buf.data(), 4, 2, 5, 4, out.data(), 4, false), std::invalid_argument);
  EXPECT_THROW(arr.run(buf.data(), 4, buf.data(), 5, 2, 4, 5, out.data(), 5, false), std::invalid_argument);
  EXPECT_THROW(SystolicArray::analytical(4, 4, 0, 1, 1), std::invalid_argument);
}

TEST(Reference, RequantizeRoundsHalfUpAndSaturates) {
  EXPECT_EQ(requantize_int8(5, 1), 3);     // 2.5 -> 3
  EXPECT_EQ(requantize_int8(-5, 1), -2);   // -2.5 -> -2 (half up)
  EXPECT_EQ(requantize_int8(-6, 1), -3);
  EXPECT_EQ(requantize_int8(1000, 0), 127);
  EXPECT_EQ(requantize_int8(-1000, 2), -128);
  EXPECT_EQ(requantize_int8(256, 8), 1);
  EXPECT_EQ(requantize_int8(127, 8), 0);
  EXPECT_EQ(requantize_int8(128, 8), 1);
}

}  // namespace
}  // namespace tessera
