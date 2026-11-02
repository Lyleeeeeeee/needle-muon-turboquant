#pragma once

#include <limits>

namespace needle {
namespace muon_detail {

// These helpers copy layouts only; all matrix arithmetic reuses HW3 kernels.
inline void Pack(const cpu::AlignedArray& source, cpu::AlignedArray* target,
                 uint32_t rows, uint32_t columns) {
  const size_t tile_columns = columns / TILE;
  for (size_t tr = 0; tr < rows / TILE; tr++) {
    for (size_t tc = 0; tc < tile_columns; tc++) {
      float* block = target->ptr + (tr * tile_columns + tc) * TILE * TILE;
      for (size_t i = 0; i < TILE; i++) {
        const float* row = source.ptr + (tr * TILE + i) * columns + tc * TILE;
        for (size_t j = 0; j < TILE; j++) block[i * TILE + j] = row[j];
      }
    }
  }
}

inline void Unpack(const cpu::AlignedArray& source, cpu::AlignedArray* target,
                   uint32_t rows, uint32_t columns) {
  const size_t tile_columns = columns / TILE;
  for (size_t tr = 0; tr < rows / TILE; tr++) {
    for (size_t tc = 0; tc < tile_columns; tc++) {
      const float* block = source.ptr + (tr * tile_columns + tc) * TILE * TILE;
      for (size_t i = 0; i < TILE; i++) {
        float* row = target->ptr + (tr * TILE + i) * columns + tc * TILE;
        for (size_t j = 0; j < TILE; j++) row[j] = block[i * TILE + j];
      }
    }
  }
}

inline void TransposePacked(const cpu::AlignedArray& source, cpu::AlignedArray* target,
                            uint32_t rows, uint32_t columns) {
  const size_t tile_rows = rows / TILE, tile_columns = columns / TILE;
  for (size_t tr = 0; tr < tile_rows; tr++) {
    for (size_t tc = 0; tc < tile_columns; tc++) {
      const float* input = source.ptr + (tr * tile_columns + tc) * TILE * TILE;
      float* output = target->ptr + (tc * tile_rows + tr) * TILE * TILE;
      for (size_t i = 0; i < TILE; i++) {
        for (size_t j = 0; j < TILE; j++) output[j * TILE + i] = input[i * TILE + j];
      }
    }
  }
}

inline void NewtonSchulzTiled(const cpu::AlignedArray& normalized,
                              cpu::AlignedArray* output,
                              uint32_t rows, uint32_t columns, int32_t steps) {
  const size_t cells = static_cast<size_t>(rows) * columns;
  if (!rows || !columns || rows > columns || rows % TILE || columns % TILE || steps <= 0 ||
      cells > std::numeric_limits<size_t>::max() / cpu::ELEM_SIZE ||
      normalized.size != cells || output->size != cells) {
    throw std::invalid_argument("Tiled NS requires aligned normalized rows<=columns and matching buffers");
  }
  const size_t gram_cells = static_cast<size_t>(rows) * rows;
  cpu::AlignedArray x(cells), transpose(cells), product(cells);
  cpu::AlignedArray gram(gram_cells), square(gram_cells);
  Pack(normalized, &x, rows, columns);
  for (int32_t step = 0; step < steps; step++) {
    TransposePacked(x, &transpose, rows, columns);
    cpu::MatmulTiled(x, transpose, &gram, rows, columns, rows);
    cpu::MatmulTiled(gram, gram, &square, rows, rows, rows);

    // Keep the reference's separate float32 multiplication/addition stages.
    cpu::ScalarMul(gram, -4.7750f, &gram);
    cpu::ScalarMul(square, 2.0315f, &square);
    cpu::EwiseAdd(gram, square, &gram);
    cpu::MatmulTiled(gram, x, &product, rows, rows, columns);
    cpu::ScalarMul(x, 3.4445f, &x);
    cpu::EwiseAdd(x, product, &x);
  }
  Unpack(x, output, rows, columns);
}

}  // namespace muon_detail
}  // namespace needle
