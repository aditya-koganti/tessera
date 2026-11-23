// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace tessera {

enum class Engine : int { DmaIn = 0, DmaOut = 1, Pe = 2, Act = 3 };
inline constexpr int kNumEngines = 4;

const char* engine_name(Engine e);

enum class Space : std::uint8_t { Dram, Sbuf };

// Block the issuing engine until semaphore `sem` holds a value >= `value`.
struct Wait {
  int sem = 0;
  std::int64_t value = 0;
};

// 2-D strided byte copy between DRAM and SBUF.
struct DmaOp {
  Space src_space = Space::Dram;
  Space dst_space = Space::Sbuf;
  std::size_t src = 0;
  std::size_t dst = 0;
  int rows = 0;
  int row_bytes = 0;
  int src_stride = 0;  // bytes between consecutive rows
  int dst_stride = 0;
};

// PSUM[m x n] (+)= SBUF_A[m x k] (int8) * SBUF_W[k x n] (int8), k <= R, n <= C.
struct MatMulOp {
  std::size_t act_addr = 0;  // SBUF byte address of A
  int act_stride = 0;        // bytes
  std::size_t w_addr = 0;    // SBUF byte address of W
  int w_stride = 0;          // bytes
  std::size_t psum_addr = 0;  // PSUM element index
  int psum_stride = 0;        // elements
  int m = 0;
  int k = 0;
  int n = 0;
  bool accumulate = false;
};

// SBUF[rows x cols] = epilogue(PSUM[rows x cols]).
// epilogue: optional ReLU, then either requantize to int8 (rounding shift +
// saturation) or store raw int32 (little-endian, 4 bytes per element).
struct ActOp {
  std::size_t psum_addr = 0;
  int psum_stride = 0;  // elements
  std::size_t dst_addr = 0;
  int dst_stride = 0;  // bytes
  int rows = 0;
  int cols = 0;
  bool relu = false;
  int shift = 0;
  bool out_int8 = false;
};

using Op = std::variant<DmaOp, MatMulOp, ActOp>;

struct Instruction {
  Engine engine = Engine::DmaIn;
  Op op;
  std::vector<Wait> waits;
  int signal = -1;  // semaphore incremented on completion, -1 for none
  std::string label;
};

// A program is a flat list of instructions. Each engine executes the
// instructions addressed to it in list order; engines only coordinate through
// semaphores, the same model used by multi-engine ML accelerator cores.
struct Program {
  std::vector<Instruction> instrs;
  std::vector<std::string> semaphore_names;

  int add_semaphore(std::string name) {
    semaphore_names.push_back(std::move(name));
    return static_cast<int>(semaphore_names.size()) - 1;
  }
  int num_semaphores() const { return static_cast<int>(semaphore_names.size()); }
};

}  // namespace tessera
