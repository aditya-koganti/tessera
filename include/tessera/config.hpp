// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

namespace tessera {

// Hardware parameters of the simulated accelerator core.
//
// The core has four engines that run concurrently and synchronize through
// semaphores: two DMA queues (DRAM -> SBUF and SBUF -> DRAM), a weight-stationary
// systolic tensor engine (PE) that accumulates into PSUM, and an activation
// engine (ACT) that applies the epilogue (ReLU, requantization) PSUM -> SBUF.
struct HwConfig {
  int array_rows = 16;  // R: reduction (K) depth handled per matmul pass
  int array_cols = 16;  // C: output columns (N) produced per matmul pass

  std::size_t sbuf_bytes = 256 * 1024;        // on-chip scratchpad (int8 tiles)
  std::size_t psum_bytes = 64 * 1024;         // on-chip int32 accumulators
  std::size_t dram_bytes = 64 * 1024 * 1024;  // off-chip memory

  int dma_bytes_per_cycle = 32;  // per DMA queue
  int dma_latency_cycles = 64;   // fixed cost per descriptor

  int act_lanes = 16;  // elements processed per cycle by the activation engine
  int act_latency_cycles = 4;

  // true: simulate every PE register every cycle (slow, used to validate timing).
  // false: use the closed-form timing model and a direct GEMM (fast).
  bool cycle_accurate_pe = true;

  // Detect concurrent conflicting accesses between in-flight instructions.
  bool detect_races = true;

  // Throws std::invalid_argument if a parameter is out of range.
  void validate() const;
};

}  // namespace tessera
