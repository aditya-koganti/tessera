// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <vector>

namespace tessera {

// Golden model used to validate simulator output. Row-major, contiguous.
std::vector<std::int32_t> gemm_ref(const std::vector<std::int8_t>& a,
                                   const std::vector<std::int8_t>& w, int m, int k, int n);

// Round-half-up arithmetic right shift followed by saturation to int8.
std::int8_t requantize_int8(std::int32_t x, int shift);

// Apply the activation-engine epilogue to a whole matrix.
std::vector<std::int8_t> epilogue_int8(const std::vector<std::int32_t>& x, bool relu, int shift);
std::vector<std::int32_t> epilogue_int32(const std::vector<std::int32_t>& x, bool relu);

}  // namespace tessera
