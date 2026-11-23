// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "tessera/config.hpp"
#include "tessera/isa.hpp"

namespace tessera {

class CompileError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// C[m x n] = epilogue(A[m x k] * W[k x n]); all matrices row-major in DRAM.
struct GemmLayer {
  int m = 0;
  int k = 0;
  int n = 0;
  std::size_t a_dram = 0;  // int8
  std::size_t w_dram = 0;  // int8
  std::size_t c_dram = 0;  // int8 if out_int8, else int32
  bool relu = false;
  int shift = 0;
  bool out_int8 = false;
};

struct CompileOptions {
  int buffers = 2;  // SBUF/PSUM slots per stream: 1 = serial, 2 = double buffering
  int tile_m = 0;   // rows per output tile; 0 = largest that fits on chip
};

struct LayerPlan {
  int tile_m = 0;
  int tiles_m = 0;
  int tiles_k = 0;
  int tiles_n = 0;
  int matmuls = 0;
  std::size_t sbuf_bytes_used = 0;
  std::size_t psum_bytes_used = 0;
};

// Lower a chain of GEMM layers (e.g. an MLP) to a multi-engine program.
// Tiles are N-major, then M, then K; partial sums over K accumulate in PSUM.
// With buffers >= 2 the DMA of tile i+1 overlaps the matmul of tile i and the
// epilogue/store of output tile j overlaps the matmuls of tile j+1.
Program compile(const HwConfig& cfg, const std::vector<GemmLayer>& layers,
                const CompileOptions& opts, std::vector<LayerPlan>* plans = nullptr);

inline Program compile_gemm(const HwConfig& cfg, const GemmLayer& layer,
                            const CompileOptions& opts, LayerPlan* plan = nullptr) {
  std::vector<LayerPlan> plans;
  Program p = compile(cfg, {layer}, opts, &plans);
  if (plan != nullptr) *plan = plans.front();
  return p;
}

}  // namespace tessera
