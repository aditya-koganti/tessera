// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <tuple>
#include <vector>

#include "tessera/compiler.hpp"
#include "tessera/reference.hpp"
#include "tessera/simulator.hpp"

namespace tessera {
namespace {

std::vector<std::int8_t> random_int8(std::size_t n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(-128, 127);
  std::vector<std::int8_t> v(n);
  for (auto& x : v) x = static_cast<std::int8_t>(d(rng));
  return v;
}

std::vector<std::int32_t> read_int32(Simulator& sim, std::size_t addr, std::size_t count) {
  std::vector<std::int32_t> out(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::uint8_t* p = sim.dram().at(addr + 4 * i, 4);
    std::uint32_t u = 0;
    for (int b = 3; b >= 0; --b) u = (u << 8) | p[b];
    out[i] = static_cast<std::int32_t>(u);
  }
  return out;
}

struct GemmRun {
  SimStats stats;
  std::vector<std::int32_t> got;
  std::vector<std::int32_t> expected;
  LayerPlan plan;
};

GemmRun run_gemm(const HwConfig& cfg, int m, int k, int n, int buffers, int tile_m = 0, bool relu = false) {
  GemmLayer L;
  L.m = m;
  L.k = k;
  L.n = n;
  L.a_dram = 0;
  L.w_dram = static_cast<std::size_t>(m) * k;
  L.c_dram = L.w_dram + static_cast<std::size_t>(k) * n;
  L.relu = relu;
  const auto a = random_int8(static_cast<std::size_t>(m) * k, 11);
  const auto w = random_int8(static_cast<std::size_t>(k) * n, 12);

  GemmRun r;
  const Program prog = compile_gemm(cfg, L, CompileOptions{buffers, tile_m}, &r.plan);
  Simulator sim(cfg);
  std::memcpy(sim.dram().at(L.a_dram, a.size()), a.data(), a.size());
  std::memcpy(sim.dram().at(L.w_dram, w.size()), w.data(), w.size());
  r.stats = sim.run(prog);
  r.got = read_int32(sim, L.c_dram, static_cast<std::size_t>(m) * n);
  r.expected = epilogue_int32(gemm_ref(a, w, m, k, n), relu);
  return r;
}

HwConfig cfg8() {
  HwConfig cfg;
  cfg.array_rows = 8;
  cfg.array_cols = 8;
  cfg.sbuf_bytes = 16 * 1024;
  cfg.psum_bytes = 4 * 1024;
  cfg.dram_bytes = 1 << 20;
  return cfg;
}

// (m, k, n, buffers)
using Case = std::tuple<int, int, int, int>;
class CompiledGemm : public ::testing::TestWithParam<Case> {};

TEST_P(CompiledGemm, MatchesGoldenModelWithoutRacesOrDeadlock) {
  const auto [m, k, n, buffers] = GetParam();
  const GemmRun r = run_gemm(cfg8(), m, k, n, buffers);
  EXPECT_EQ(r.got, r.expected);
  EXPECT_EQ(r.stats.useful_macs, static_cast<std::int64_t>(m) * k * n);
  EXPECT_GT(r.stats.pe_utilization(cfg8()), 0.0);
  EXPECT_LE(r.stats.pe_utilization(cfg8()), 1.0);
}

INSTANTIATE_TEST_SUITE_P(Shapes, CompiledGemm,
                         ::testing::Values(Case{1, 1, 1, 1}, Case{1, 1, 1, 2}, Case{8, 8, 8, 2}, Case{9, 17, 23, 2},
                                           Case{64, 40, 24, 1}, Case{64, 40, 24, 2}, Case{100, 64, 64, 3},
                                           Case{3, 100, 5, 2}, Case{50, 7, 70, 4}));

TEST(Compiler, DoubleBufferingOverlapsDmaWithCompute) {
  HwConfig cfg = cfg8();
  cfg.cycle_accurate_pe = false;
  const GemmRun serial = run_gemm(cfg, 64, 128, 64, 1);
  const GemmRun overlapped = run_gemm(cfg, 64, 128, 64, 2);
  EXPECT_EQ(serial.got, serial.expected);
  EXPECT_EQ(overlapped.got, overlapped.expected);
  EXPECT_LT(overlapped.stats.total_cycles, serial.stats.total_cycles * 3 / 4);
}

TEST(Compiler, CycleAccurateAndAnalyticalPeModelsAgree) {
  HwConfig fast = cfg8();
  fast.cycle_accurate_pe = false;
  const GemmRun a = run_gemm(cfg8(), 33, 20, 17, 2);
  const GemmRun b = run_gemm(fast, 33, 20, 17, 2);
  EXPECT_EQ(a.got, b.got);
  EXPECT_EQ(a.stats.total_cycles, b.stats.total_cycles);
}

TEST(Compiler, PlanReflectsTiling) {
  const GemmRun r = run_gemm(cfg8(), 40, 20, 17, 2, 16);
  EXPECT_EQ(r.plan.tile_m, 16);
  EXPECT_EQ(r.plan.tiles_m, 3);
  EXPECT_EQ(r.plan.tiles_k, 3);
  EXPECT_EQ(r.plan.tiles_n, 3);
  EXPECT_EQ(r.plan.matmuls, 27);
  EXPECT_LE(r.plan.sbuf_bytes_used, cfg8().sbuf_bytes);
  EXPECT_EQ(r.got, r.expected);
}

TEST(Compiler, ReluEpilogue) {
  const GemmRun r = run_gemm(cfg8(), 12, 24, 10, 2, 0, true);
  EXPECT_EQ(r.got, r.expected);
}

TEST(Compiler, RejectsTilesThatDoNotFitOnChip) {
  HwConfig cfg = cfg8();
  cfg.psum_bytes = 256;  // 2 slots x 8 cols x 4 B = 64 B per row -> at most 4 rows
  GemmLayer L;
  L.m = 64;
  L.k = 8;
  L.n = 8;
  L.c_dram = 4096;
  EXPECT_THROW(compile_gemm(cfg, L, CompileOptions{2, 16}), CompileError);
  LayerPlan plan;
  compile_gemm(cfg, L, CompileOptions{2, 0}, &plan);
  EXPECT_EQ(plan.tile_m, 4);
  EXPECT_THROW(compile_gemm(cfg, L, CompileOptions{0, 0}), CompileError);
}

TEST(Compiler, TwoLayerMlpChainsThroughDram) {
  const HwConfig cfg = cfg8();
  const int M = 20, D0 = 30, D1 = 26, D2 = 9;
  GemmLayer l1;
  l1.m = M;
  l1.k = D0;
  l1.n = D1;
  l1.a_dram = 0;
  l1.w_dram = 1024;
  l1.c_dram = 2048;
  l1.relu = true;
  l1.shift = 9;
  l1.out_int8 = true;
  GemmLayer l2;
  l2.m = M;
  l2.k = D1;
  l2.n = D2;
  l2.a_dram = l1.c_dram;
  l2.w_dram = 4096;
  l2.c_dram = 8192;

  const auto a = random_int8(static_cast<std::size_t>(M) * D0, 21);
  const auto w1 = random_int8(static_cast<std::size_t>(D0) * D1, 22);
  const auto w2 = random_int8(static_cast<std::size_t>(D1) * D2, 23);

  Simulator sim(cfg);
  std::memcpy(sim.dram().at(l1.a_dram, a.size()), a.data(), a.size());
  std::memcpy(sim.dram().at(l1.w_dram, w1.size()), w1.data(), w1.size());
  std::memcpy(sim.dram().at(l2.w_dram, w2.size()), w2.data(), w2.size());
  std::vector<LayerPlan> plans;
  sim.run(compile(cfg, {l1, l2}, CompileOptions{}, &plans));
  ASSERT_EQ(plans.size(), 2u);

  const auto hidden = epilogue_int8(gemm_ref(a, w1, M, D0, D1), true, 9);
  const auto expected = gemm_ref(hidden, w2, M, D1, D2);
  EXPECT_EQ(read_int32(sim, l2.c_dram, expected.size()), expected);
}

TEST(Compiler, DroppingASynchronizationEdgeIsCaught) {
  // Remove the "PSUM slot drained" dependency and the race detector must fire.
  // Fast DMA and a slow activation engine make the matmul of tile j+2 reach
  // PSUM slot j%2 while the epilogue of tile j is still reading it.
  HwConfig cfg = cfg8();
  cfg.dma_latency_cycles = 0;
  cfg.dma_bytes_per_cycle = 64;
  cfg.act_lanes = 1;
  GemmLayer L;
  L.m = 32;
  L.k = 8;
  L.n = 32;
  L.w_dram = 1024;
  L.c_dram = 2048;
  Program p = compile_gemm(cfg, L, CompileOptions{2, 8});
  {
    Simulator ok(cfg);
    EXPECT_NO_THROW(ok.run(p));
  }
  for (Instruction& in : p.instrs) {
    if (in.engine == Engine::Pe) in.waits.erase(in.waits.begin() + 2, in.waits.end());
  }
  Simulator sim(cfg);
  EXPECT_THROW(sim.run(p), RaceError);
}

}  // namespace
}  // namespace tessera
