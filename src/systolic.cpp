// SPDX-License-Identifier: MIT
#include "tessera/systolic.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tessera {

SystolicArray::SystolicArray(int rows, int cols) : rows_(rows), cols_(cols) {
  if (rows < 1 || cols < 1) throw std::invalid_argument("systolic array needs rows, cols >= 1");
  const auto cells = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  weight_.assign(cells, 0);
  act_reg_.assign(cells, 0);
  psum_reg_.assign(cells, 0);
}

MatMulTiming SystolicArray::analytical(int rows, int cols, int m, int k, int n) {
  if (m < 1 || k < 1 || n < 1 || k > rows || n > cols) {
    throw std::invalid_argument("matmul shape m=" + std::to_string(m) + " k=" + std::to_string(k) +
                                " n=" + std::to_string(n) + " does not fit a " +
                                std::to_string(rows) + "x" + std::to_string(cols) + " array");
  }
  MatMulTiming t;
  t.weight_load_cycles = k;
  // Last output C[m-1][n-1] leaves column n-1 at cycle (m-1) + (n-1) + (rows-1).
  t.stream_cycles = static_cast<std::int64_t>(m) + n + rows - 2;
  t.useful_macs = static_cast<std::int64_t>(m) * k * n;
  return t;
}

MatMulTiming SystolicArray::run(const std::int8_t* a, int a_stride, const std::int8_t* w,
                                int w_stride, int m, int k, int n, std::int32_t* out,
                                int out_stride, bool accumulate) {
  const MatMulTiming timing = analytical(rows_, cols_, m, k, n);
  const int R = rows_;
  const int C = cols_;

  // Weight load: rows beyond k (and columns beyond n) hold zero and just pass data through.
  std::fill(weight_.begin(), weight_.end(), 0);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) weight_[i * C + j] = w[i * w_stride + j];
  }
  std::fill(act_reg_.begin(), act_reg_.end(), 0);
  std::fill(psum_reg_.begin(), psum_reg_.end(), 0);

  std::int64_t cycles = 0;
  for (std::int64_t t = 0; t < timing.stream_cycles; ++t) {
    // Update in place from bottom-right to top-left so every PE reads its
    // neighbours' values from the previous cycle.
    for (int i = R - 1; i >= 0; --i) {
      for (int j = n - 1; j >= 0; --j) {
        std::int32_t a_in = 0;
        if (j == 0) {
          const std::int64_t row = t - i;  // skewed injection
          if (i < k && row >= 0 && row < m) a_in = a[row * a_stride + i];
        } else {
          a_in = act_reg_[i * C + (j - 1)];
        }
        act_reg_[i * C + j] = a_in;
        const std::int32_t p_in = (i == 0) ? 0 : psum_reg_[(i - 1) * C + j];
        psum_reg_[i * C + j] = p_in + weight_[i * C + j] * a_in;
      }
    }
    // Collect outputs leaving the bottom row.
    for (int j = 0; j < n; ++j) {
      const std::int64_t row = t - j - (R - 1);
      if (row < 0 || row >= m) continue;
      std::int32_t& dst = out[row * out_stride + j];
      const std::int32_t v = psum_reg_[(R - 1) * C + j];
      dst = accumulate ? dst + v : v;
    }
    pe_updates_ += static_cast<std::int64_t>(R) * n;
    ++cycles;
  }
  if (cycles != timing.stream_cycles) throw std::logic_error("systolic timing mismatch");
  return timing;
}

}  // namespace tessera
