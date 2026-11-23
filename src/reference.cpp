// SPDX-License-Identifier: MIT
#include "tessera/reference.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace tessera {

std::vector<std::int32_t> gemm_ref(const std::vector<std::int8_t>& a,
                                   const std::vector<std::int8_t>& w, int m, int k, int n) {
  if (a.size() != static_cast<std::size_t>(m) * k || w.size() != static_cast<std::size_t>(k) * n) {
    throw std::invalid_argument("gemm_ref: size mismatch");
  }
  std::vector<std::int32_t> c(static_cast<std::size_t>(m) * n, 0);
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const std::int32_t av = a[static_cast<std::size_t>(i) * k + p];
      for (int j = 0; j < n; ++j) {
        c[static_cast<std::size_t>(i) * n + j] += av * w[static_cast<std::size_t>(p) * n + j];
      }
    }
  }
  return c;
}

std::int8_t requantize_int8(std::int32_t x, int shift) {
  std::int64_t v = x;
  if (shift > 0) {
    v += std::int64_t{1} << (shift - 1);
    // Floor division by 2^shift without relying on signed right-shift semantics.
    const std::int64_t d = std::int64_t{1} << shift;
    v = (v >= 0) ? v / d : -((-v + d - 1) / d);
  }
  return static_cast<std::int8_t>(std::clamp<std::int64_t>(v, -128, 127));
}

std::vector<std::int8_t> epilogue_int8(const std::vector<std::int32_t>& x, bool relu, int shift) {
  std::vector<std::int8_t> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    y[i] = requantize_int8(relu ? std::max(x[i], 0) : x[i], shift);
  }
  return y;
}

std::vector<std::int32_t> epilogue_int32(const std::vector<std::int32_t>& x, bool relu) {
  std::vector<std::int32_t> y(x);
  if (relu) {
    for (auto& v : y) v = std::max(v, 0);
  }
  return y;
}

}  // namespace tessera
