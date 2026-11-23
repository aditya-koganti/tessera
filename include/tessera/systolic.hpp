// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <vector>

namespace tessera {

struct MatMulTiming {
  std::int64_t weight_load_cycles = 0;  // shift k weight rows into the array
  std::int64_t stream_cycles = 0;       // skewed activation stream + drain
  std::int64_t useful_macs = 0;
  std::int64_t total() const { return weight_load_cycles + stream_cycles; }
};

// Register-level model of a weight-stationary systolic array with R rows and C
// columns. PE(i, j) holds W[i][j]; activations move left-to-right one PE per
// cycle and partial sums move top-to-bottom one PE per cycle. Activation
// A[m][i] enters row i at cycle m + i, so C[m][j] leaves the bottom of column j
// at cycle m + j + R - 1.
class SystolicArray {
 public:
  SystolicArray(int rows, int cols);

  // out[m x n] (+)= a[m x k] * w[k x n]; requires 1 <= k <= R and 1 <= n <= C.
  MatMulTiming run(const std::int8_t* a, int a_stride, const std::int8_t* w, int w_stride, int m,
                   int k, int n, std::int32_t* out, int out_stride, bool accumulate);

  // Closed-form timing for the same dataflow; run() must agree with it exactly.
  static MatMulTiming analytical(int rows, int cols, int m, int k, int n);

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  std::int64_t pe_updates() const { return pe_updates_; }

 private:
  int rows_;
  int cols_;
  std::vector<std::int32_t> weight_;
  std::vector<std::int32_t> act_reg_;
  std::vector<std::int32_t> psum_reg_;
  std::int64_t pe_updates_ = 0;
};

}  // namespace tessera
