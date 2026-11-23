// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <cstdint>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <vector>

#include "tessera/config.hpp"
#include "tessera/isa.hpp"
#include "tessera/memory.hpp"
#include "tessera/systolic.hpp"

namespace tessera {

// No engine can make progress but instructions remain: a synchronization bug.
class DeadlockError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Two in-flight instructions touch overlapping memory and at least one writes.
class RaceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct EngineStats {
  std::int64_t busy_cycles = 0;
  std::int64_t instructions = 0;
};

struct SimStats {
  std::int64_t total_cycles = 0;
  std::array<EngineStats, kNumEngines> engines{};
  std::int64_t useful_macs = 0;
  std::int64_t dma_bytes_in = 0;
  std::int64_t dma_bytes_out = 0;

  double occupancy(Engine e) const;
  // Fraction of peak MAC throughput (R x C MACs per cycle) actually used.
  double pe_utilization(const HwConfig& cfg) const;
};

struct TraceEvent {
  Engine engine;
  std::string label;
  std::int64_t start;
  std::int64_t end;
};

class Simulator {
 public:
  explicit Simulator(HwConfig cfg);

  ByteMemory& dram() { return dram_; }
  ByteMemory& sbuf() { return sbuf_; }
  PsumMemory& psum() { return psum_; }
  const HwConfig& config() const { return cfg_; }

  // Discrete-event execution: engines start an instruction as soon as its
  // semaphore waits are satisfied and the engine is idle.
  SimStats run(const Program& program);

  const std::vector<TraceEvent>& trace() const { return trace_; }
  // Chrome trace-event JSON; open in https://ui.perfetto.dev or chrome://tracing.
  void write_chrome_trace(std::ostream& os) const;

 private:
  std::int64_t execute(const Instruction& in, SimStats& stats);

  HwConfig cfg_;
  ByteMemory dram_;
  ByteMemory sbuf_;
  PsumMemory psum_;
  SystolicArray array_;
  std::vector<TraceEvent> trace_;
};

}  // namespace tessera
