// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tessera {

class MemoryError : public std::out_of_range {
 public:
  using std::out_of_range::out_of_range;
};

// Bytes covered by `rows` rows of `row_len` units placed `stride` units apart.
inline std::size_t region_span(int rows, int row_len, int stride) {
  if (rows <= 0 || row_len <= 0) return 0;
  return static_cast<std::size_t>(rows - 1) * static_cast<std::size_t>(stride) +
         static_cast<std::size_t>(row_len);
}

template <typename T>
class Memory {
 public:
  Memory(std::string name, std::size_t elements) : name_(std::move(name)), data_(elements, T{}) {}

  std::size_t size() const { return data_.size(); }
  const std::string& name() const { return name_; }

  void check(std::size_t addr, std::size_t len) const {
    if (len > data_.size() || addr > data_.size() - len) {
      throw MemoryError(name_ + ": access [" + std::to_string(addr) + ", " +
                        std::to_string(addr + len) + ") is outside [0, " +
                        std::to_string(data_.size()) + ")");
    }
  }

  T* at(std::size_t addr, std::size_t len) {
    check(addr, len);
    return data_.data() + addr;
  }
  const T* at(std::size_t addr, std::size_t len) const {
    check(addr, len);
    return data_.data() + addr;
  }

 private:
  std::string name_;
  std::vector<T> data_;
};

using ByteMemory = Memory<std::uint8_t>;
using PsumMemory = Memory<std::int32_t>;

}  // namespace tessera
