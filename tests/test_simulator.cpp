// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <cstring>
#include <sstream>

#include "tessera/simulator.hpp"

namespace tessera {
namespace {

HwConfig small_config() {
  HwConfig cfg;
  cfg.array_rows = 4;
  cfg.array_cols = 4;
  cfg.sbuf_bytes = 4096;
  cfg.psum_bytes = 4096;
  cfg.dram_bytes = 1 << 16;
  cfg.dma_bytes_per_cycle = 8;
  cfg.dma_latency_cycles = 10;
  return cfg;
}

Instruction dma(Engine e, Space from, std::size_t src, Space to, std::size_t dst, int rows, int row_bytes,
                int src_stride, int dst_stride, const char* label) {
  Instruction in;
  in.engine = e;
  in.op = DmaOp{from, to, src, dst, rows, row_bytes, src_stride, dst_stride};
  in.label = label;
  return in;
}

TEST(Simulator, StridedDmaCopiesSubmatrixAndChargesLatencyPlusBandwidth) {
  Simulator sim(small_config());
  for (int i = 0; i < 64; ++i) *sim.dram().at(static_cast<std::size_t>(i), 1) = static_cast<std::uint8_t>(i);

  Program p;
  // Copy the 3x2 block starting at row 1, col 2 of an 8x8 byte matrix.
  p.instrs.push_back(dma(Engine::DmaIn, Space::Dram, 1 * 8 + 2, Space::Sbuf, 0, 3, 2, 8, 2, "block"));
  const SimStats s = sim.run(p);

  const std::uint8_t expected[] = {10, 11, 18, 19, 26, 27};
  EXPECT_EQ(std::memcmp(sim.sbuf().at(0, 6), expected, 6), 0);
  EXPECT_EQ(s.total_cycles, 10 + 1);  // latency + ceil(6 / 8)
  EXPECT_EQ(s.dma_bytes_in, 6);
  EXPECT_EQ(s.engines[static_cast<int>(Engine::DmaIn)].instructions, 1);
}

TEST(Simulator, EnginesOverlapUnlessSemaphoresOrderThem) {
  Program p;
  const int done = p.add_semaphore("done");
  p.instrs.push_back(dma(Engine::DmaIn, Space::Dram, 0, Space::Sbuf, 0, 1, 80, 80, 80, "in"));
  p.instrs.push_back(dma(Engine::DmaOut, Space::Sbuf, 512, Space::Dram, 4096, 1, 80, 80, 80, "out"));
  {
    Simulator sim(small_config());
    EXPECT_EQ(sim.run(p).total_cycles, 20);  // both run concurrently: 10 + 80/8
  }
  p.instrs[0].signal = done;
  p.instrs[1].waits.push_back(Wait{done, 1});
  Simulator sim(small_config());
  EXPECT_EQ(sim.run(p).total_cycles, 40);  // serialized by the semaphore
  ASSERT_EQ(sim.trace().size(), 2u);
  EXPECT_EQ(sim.trace()[1].start, 20);
}

TEST(Simulator, ReportsDeadlockWithBlockedEngineAndSemaphore) {
  Program p;
  const int never = p.add_semaphore("never_signaled");
  Instruction in = dma(Engine::DmaIn, Space::Dram, 0, Space::Sbuf, 0, 1, 8, 8, 8, "stuck");
  in.waits.push_back(Wait{never, 1});
  p.instrs.push_back(in);
  Simulator sim(small_config());
  try {
    sim.run(p);
    FAIL() << "expected DeadlockError";
  } catch (const DeadlockError& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("stuck"), std::string::npos);
    EXPECT_NE(msg.find("never_signaled>=1 (is 0)"), std::string::npos);
  }
}

TEST(Simulator, DetectsUnsynchronizedWriteToBufferBeingRead) {
  // DMA-in overwrites the SBUF tile the store queue is still reading.
  Program p;
  p.instrs.push_back(dma(Engine::DmaOut, Space::Sbuf, 0, Space::Dram, 0, 1, 64, 64, 64, "store tile"));
  p.instrs.push_back(dma(Engine::DmaIn, Space::Dram, 1024, Space::Sbuf, 32, 1, 64, 64, 64, "prefetch"));
  Simulator sim(small_config());
  EXPECT_THROW(sim.run(p), RaceError);

  HwConfig unchecked = small_config();
  unchecked.detect_races = false;
  Simulator lax(unchecked);
  EXPECT_NO_THROW(lax.run(p));
}

TEST(Simulator, ConcurrentReadsAreNotRaces) {
  Program p;
  p.instrs.push_back(dma(Engine::DmaOut, Space::Sbuf, 0, Space::Dram, 0, 1, 64, 64, 64, "store a"));
  p.instrs.push_back(dma(Engine::DmaIn, Space::Sbuf, 0, Space::Sbuf, 128, 1, 64, 64, 64, "copy a"));
  Simulator sim(small_config());
  EXPECT_NO_THROW(sim.run(p));
}

TEST(Simulator, OutOfBoundsAccessThrows) {
  Program p;
  p.instrs.push_back(dma(Engine::DmaIn, Space::Dram, 0, Space::Sbuf, 4000, 2, 64, 64, 64, "overflow"));
  Simulator sim(small_config());
  EXPECT_THROW(sim.run(p), MemoryError);
}

TEST(Simulator, RejectsOpOnWrongEngine) {
  Program p;
  Instruction in = dma(Engine::Pe, Space::Dram, 0, Space::Sbuf, 0, 1, 8, 8, 8, "wrong engine");
  p.instrs.push_back(in);
  Simulator sim(small_config());
  EXPECT_THROW(sim.run(p), std::invalid_argument);
}

TEST(Simulator, ActivationEngineAppliesReluAndRequantization) {
  Simulator sim(small_config());
  const std::int32_t vals[] = {-300, -1, 0, 255, 256, 100000};
  std::memcpy(sim.psum().at(0, 6), vals, sizeof(vals));
  Program p;
  Instruction act;
  act.engine = Engine::Act;
  act.op = ActOp{0, 6, 0, 6, 1, 6, true, 8, true};
  act.label = "relu+requant";
  p.instrs.push_back(act);
  sim.run(p);
  const std::int8_t expected[] = {0, 0, 0, 1, 1, 127};
  EXPECT_EQ(std::memcmp(sim.sbuf().at(0, 6), expected, 6), 0);
}

TEST(Simulator, ChromeTraceListsEveryInstruction) {
  Program p;
  p.instrs.push_back(dma(Engine::DmaIn, Space::Dram, 0, Space::Sbuf, 0, 1, 8, 8, 8, "load \"x\""));
  Simulator sim(small_config());
  sim.run(p);
  std::ostringstream os;
  sim.write_chrome_trace(os);
  const std::string json = os.str();
  EXPECT_NE(json.find("\"ph\":\"X\""), std::string::npos);
  EXPECT_NE(json.find("load \\\"x\\\""), std::string::npos);
  EXPECT_NE(json.find("\"name\":\"dma_in\""), std::string::npos);
}

}  // namespace
}  // namespace tessera
