#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace needle {
namespace matmul_detail {

// Validate before narrowing index arithmetic or reading a possibly strided view.
inline void ValidateView(const cpu::AlignedArray& source, uint32_t rows, uint32_t columns,
                         const std::vector<int64_t>& strides, int64_t offset) {
  if (strides.size() != 2 || offset < 0) {
    throw std::invalid_argument("Matmul requires two strides and a nonnegative offset");
  }
  __int128 low = offset, high = offset;
  for (size_t axis = 0; axis < 2; axis++) {
    const __int128 extent = static_cast<__int128>(axis ? columns - 1 : rows - 1) * strides[axis];
    if (extent < 0) low += extent;
    else high += extent;
  }
  if (low < 0 || high >= static_cast<__int128>(source.size) ||
      high > std::numeric_limits<int64_t>::max()) {
    throw std::invalid_argument("Matmul view exceeds its source buffer");
  }
}

inline void PackView(const cpu::AlignedArray& source, cpu::AlignedArray* packed,
                     uint32_t rows, uint32_t columns,
                     const std::vector<int64_t>& strides, int64_t offset) {
  const size_t tile_columns = columns / TILE;
  for (size_t tr = 0; tr < rows / TILE; tr++) {
    for (size_t tc = 0; tc < tile_columns; tc++) {
      float* tile = packed->ptr + (tr * tile_columns + tc) * TILE * TILE;
      for (size_t i = 0; i < TILE; i++) {
        const int64_t index = offset + static_cast<int64_t>(tr * TILE + i) * strides[0]
                                      + static_cast<int64_t>(tc * TILE) * strides[1];
        if (strides[1] == 1) {
          std::memcpy(tile + i * TILE, source.ptr + index, TILE * cpu::ELEM_SIZE);
        } else {
          for (int64_t j = 0; j < TILE; j++) tile[i * TILE + j] = source.ptr[index + j * strides[1]];
        }
      }
    }
  }
}

inline void Unpack(const cpu::AlignedArray& packed, cpu::AlignedArray* output,
                   uint32_t rows, uint32_t columns) {
  const size_t tile_columns = columns / TILE;
  for (size_t tr = 0; tr < rows / TILE; tr++) {
    for (size_t tc = 0; tc < tile_columns; tc++) {
      const float* tile = packed.ptr + (tr * tile_columns + tc) * TILE * TILE;
      for (size_t i = 0; i < TILE; i++) {
        std::memcpy(output->ptr + (tr * TILE + i) * columns + tc * TILE,
                    tile + i * TILE, TILE * cpu::ELEM_SIZE);
      }
    }
  }
}

inline void Strided(const cpu::AlignedArray& a, const cpu::AlignedArray& b,
                    cpu::AlignedArray* output, uint32_t m, uint32_t n, uint32_t p,
                    const std::vector<int64_t>& a_strides, const std::vector<int64_t>& b_strides,
                    int64_t a_offset, int64_t b_offset) {
  const size_t a_cells = static_cast<size_t>(m) * n;
  const size_t b_cells = static_cast<size_t>(n) * p;
  const size_t out_cells = static_cast<size_t>(m) * p;
  const size_t max_cells = std::numeric_limits<size_t>::max() / cpu::ELEM_SIZE;
  if (!m || !n || !p || m % TILE || n % TILE || p % TILE ||
      a_cells > max_cells || b_cells > max_cells || out_cells > max_cells ||
      output->size != out_cells || output->ptr == a.ptr || output->ptr == b.ptr) {
    throw std::invalid_argument("Strided tiled matmul requires positive aligned dimensions and a distinct matching output");
  }
  ValidateView(a, m, n, a_strides, a_offset);
  ValidateView(b, n, p, b_strides, b_offset);
  cpu::AlignedArray packed_a(a_cells), packed_b(b_cells), packed_output(out_cells);
  PackView(a, &packed_a, m, n, a_strides, a_offset);
  PackView(b, &packed_b, n, p, b_strides, b_offset);
  cpu::MatmulTiled(packed_a, packed_b, &packed_output, m, n, p);
  Unpack(packed_output, output, m, p);
}

}  // namespace matmul_detail
}  // namespace needle
