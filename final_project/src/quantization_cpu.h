// CPU kernels for real low-bit storage. Included after AlignedArray is defined.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>

namespace needle {
namespace cpu {
namespace quantization_detail {
namespace py = pybind11;
using Bytes = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using Floats = py::array_t<float, py::array::c_style | py::array::forcecast>;

inline size_t RowBytes(size_t dim, int bits) {
  if ((bits != 2 && bits != 4) || dim == 0 ||
      dim > std::numeric_limits<size_t>::max() - 3)
    throw std::invalid_argument("Expected positive dimension and 2 or 4 bits");
  return (dim + (8 / bits) - 1) / (8 / bits);
}

inline void CheckCodebook(const Floats& centroids, int bits) {
  if (centroids.ndim() != 1 || centroids.size() != (1 << bits))
    throw std::invalid_argument("Codebook length must equal 2**bits");
  const float* c = centroids.data();
  for (ssize_t i = 0; i < centroids.size(); ++i)
    if (!std::isfinite(c[i]) || (i && c[i] <= c[i - 1]))
      throw std::invalid_argument("Codebook must be finite and strictly increasing");
}

inline Bytes PackCodes(const Bytes& codes, int bits) {
  if (codes.ndim() != 2) throw std::invalid_argument("Codes must be a matrix");
  const size_t rows = codes.shape(0), dim = codes.shape(1);
  const size_t stride = RowBytes(dim, bits);
  Bytes packed({static_cast<ssize_t>(rows), static_cast<ssize_t>(stride)});
  const uint8_t* input = codes.data();
  uint8_t* output = packed.mutable_data();
  py::gil_scoped_release release;
  std::fill(output, output + rows * stride, uint8_t{0});
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < dim; ++j) {
      const uint8_t index = input[i * dim + j];
      if (index >= (1 << bits)) throw std::invalid_argument("Code exceeds bit width");
      output[i * stride + j / (8 / bits)] |= index << ((j % (8 / bits)) * bits);
    }
  return packed;
}

inline Bytes QuantizePack(const AlignedArray& values, const Floats& centroids,
                          size_t rows, size_t dim, int bits) {
  const size_t stride = RowBytes(dim, bits);
  CheckCodebook(centroids, bits);
  if (rows > values.size / dim || rows * dim != values.size)
    throw std::invalid_argument("Input buffer size does not match shape");
  std::vector<float> boundaries((1 << bits) - 1);
  const float* c = centroids.data();
  for (size_t j = 0; j < boundaries.size(); ++j)
    boundaries[j] = (c[j] + c[j + 1]) * 0.5f;
  Bytes packed({static_cast<ssize_t>(rows), static_cast<ssize_t>(stride)});
  uint8_t* output = packed.mutable_data();
  py::gil_scoped_release release;
  std::fill(output, output + rows * stride, uint8_t{0});
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < dim; ++j) {
      const float value = values.ptr[i * dim + j];
      if (!std::isfinite(value)) throw std::invalid_argument("Non-finite input");
      const uint8_t index = std::upper_bound(boundaries.begin(), boundaries.end(), value) - boundaries.begin();
      output[i * stride + j / (8 / bits)] |= index << ((j % (8 / bits)) * bits);
    }
  return packed;
}

inline void CheckDecode(const Bytes& packed, const Floats& first,
                        size_t start, size_t stop, size_t dim, int bits,
                        const AlignedArray* output) {
  const size_t stride = RowBytes(dim, bits);
  if (packed.ndim() != 2 || static_cast<size_t>(packed.shape(1)) != stride ||
      first.ndim() != 1 || first.size() != packed.shape(0) ||
      start > stop || stop > static_cast<size_t>(packed.shape(0)) ||
      (stop - start) > output->size / dim || (stop - start) * dim != output->size)
    throw std::invalid_argument("Invalid packed block shape or output buffer");
}

inline void DecodeCentroids(const Bytes& packed, const Floats& centroids,
                            const Floats& norms, size_t start, size_t stop,
                            size_t dim, int bits, AlignedArray* output) {
  CheckDecode(packed, norms, start, stop, dim, bits, output);
  CheckCodebook(centroids, bits);
  const size_t stride = RowBytes(dim, bits);
  const uint8_t* input = packed.data();
  const float* c = centroids.data();
  const float* n = norms.data();
  const uint8_t mask = (1 << bits) - 1;
  py::gil_scoped_release release;
  for (size_t i = start; i < stop; ++i)
    for (size_t j = 0; j < dim; ++j) {
      const uint8_t index = (input[i * stride + j / (8 / bits)] >> ((j % (8 / bits)) * bits)) & mask;
      output->ptr[(i - start) * dim + j] = c[index] * n[i];
    }
}

inline void DecodeUniform(const Bytes& packed, const Floats& minimum,
                          const Floats& scale, size_t start, size_t stop,
                          size_t dim, int bits, AlignedArray* output) {
  CheckDecode(packed, minimum, start, stop, dim, bits, output);
  if (scale.ndim() != 1 || scale.size() != minimum.size())
    throw std::invalid_argument("Invalid uniform scales");
  const size_t stride = RowBytes(dim, bits);
  const uint8_t* input = packed.data();
  const float* lo = minimum.data();
  const float* step = scale.data();
  const uint8_t mask = (1 << bits) - 1;
  py::gil_scoped_release release;
  for (size_t i = start; i < stop; ++i)
    for (size_t j = 0; j < dim; ++j) {
      const uint8_t index = (input[i * stride + j / (8 / bits)] >> ((j % (8 / bits)) * bits)) & mask;
      output->ptr[(i - start) * dim + j] = lo[i] + step[i] * index;
    }
}

inline void Bind(py::module_& module) {
  module.def("pack_codes", &PackCodes);
  module.def("quantize_pack", &QuantizePack);
  module.def("decode_centroids", &DecodeCentroids);
  module.def("decode_uniform", &DecodeUniform);
}
}  // namespace quantization_detail
}  // namespace cpu
}  // namespace needle
