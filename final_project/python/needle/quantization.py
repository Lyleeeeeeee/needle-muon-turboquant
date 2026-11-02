"""TurboQuant-MSE (Algorithm 1), real packed storage, and CPU vector retrieval.

Codebooks use the exact sphere-coordinate distribution, not a Gaussian fit.
SciPy is used only for offline scalar codebook construction. Runtime rotations,
decoding outputs and search matrix products use Needle. Numeric byte counts
include shared metadata once, but exclude Python objects and allocator overhead.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np
from scipy.special import betainc, betaincinv

from .autograd import Tensor
from .backend_selection import BACKEND
from . import backend_ndarray as nd


def _parameters(dim, bits):
    if not isinstance(dim, (int, np.integer)) or dim < 2:
        raise ValueError("Dimension must be an integer >= 2")
    if not isinstance(bits, (int, np.integer)) or bits not in (2, 4):
        raise ValueError("Only 2-bit and 4-bit storage is supported")


@lru_cache(maxsize=32)
def sphere_codebook(dim, bits):
    """Solve Lloyd-Max updates with exact Beta cell masses and first moments."""
    _parameters(dim, bits)
    alpha = (dim - 1) / 2
    levels = 1 << bits
    centers = 2 * betaincinv(alpha, alpha, (np.arange(levels) + .5) / levels) - 1
    constant = math.exp(math.lgamma(dim / 2) - .5 * math.log(math.pi) - math.lgamma(alpha))
    for _ in range(4000):
        edges = np.r_[-1., (centers[:-1] + centers[1:]) * .5, 1.]
        lo, hi = edges[:-1], edges[1:]
        mass = betainc(alpha, alpha, (hi + 1) / 2) - betainc(alpha, alpha, (lo + 1) / 2)
        right = lo >= 0
        mass[right] = betainc(alpha, alpha, (1 - lo[right]) / 2) - betainc(alpha, alpha, (1 - hi[right]) / 2)
        moment = constant / (dim - 1) * ((1 - lo**2)**alpha - (1 - hi**2)**alpha)
        if np.any(mass <= 0):
            raise ArithmeticError("Codebook cell has zero numerical probability")
        updated = moment / mass
        updated = (updated - updated[::-1]) * .5
        if np.max(abs(updated - centers)) < 1e-11:
            result = updated.astype('float32')
            result.setflags(write=False)
            return result
        centers = updated
    raise ArithmeticError("Lloyd-Max codebook did not converge")


def _device(device):
    if BACKEND != 'nd':
        raise ValueError("Quantization requires NEEDLE_BACKEND=nd")
    if device is None:
        device = nd.cpu() if nd.cpu().enabled() else nd.cpu_numpy()
    if device not in (nd.cpu(), nd.cpu_numpy()) or not device.enabled():
        raise ValueError("Select the native CPU or cpu_numpy device")
    return device


def _matrix(values, dim=None):
    if isinstance(values, Tensor):
        values = values.numpy()
    values = np.asarray(values, dtype='float32')
    if values.ndim != 2 or min(values.shape) == 0 or (dim is not None and values.shape[1] != dim):
        raise ValueError("Expected a nonempty matrix with the configured feature dimension")
    if not np.isfinite(values).all():
        raise ValueError("Input values must be finite float32 numbers")
    return np.ascontiguousarray(values)


def _tensor(values, device):
    return Tensor(values, device=device, dtype='float32', requires_grad=False)


def _unit_tensor_rows(tensor):
    """Stable L2 normalization using only the selected Needle NDArray backend."""
    values = tensor.realize_cached_data()
    scale = values.maximum(-values).max(axis=1, keepdims=True)
    safe_scale = scale + (scale == 0)
    scaled = values / safe_scale.broadcast_to(values.shape)
    norm = (scaled * scaled).sum(axis=1, keepdims=True) ** .5
    safe_norm = norm + (norm == 0)
    return Tensor.make_const(scaled / safe_norm.broadcast_to(values.shape))


def _pack(codes, bits, device):
    if device == nd.cpu():
        return device.pack_codes(codes, bits)
    rows, dim = codes.shape
    packed = np.zeros((rows, (dim * bits + 7) // 8), dtype='uint8')
    for offset in range(8 // bits):
        part = codes[:, offset::8 // bits]
        packed[:, :part.shape[1]] |= part << (offset * bits)
    return packed


def _indices(packed, dim, bits):
    columns = np.arange(dim)
    return (packed[:, columns // (8 // bits)] >> ((columns % (8 // bits)) * bits)) & ((1 << bits) - 1)


@dataclass
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray
    max_decoded_block_bytes: int
    max_score_block_bytes: int


class _VectorStore:
    @property
    def nbytes(self):
        return sum(self.storage_bytes.values())

    def _query_tensor(self, values):
        return values

    def _bounds(self, start, stop):
        if not (0 <= start < stop <= self.rows):
            raise ValueError("Invalid nonempty row range")

    def search(self, queries, k=10, block_size=256, *, metric='inner_product'):
        """Blockwise raw-IP or decoded-cosine top-k; zero cosine rows score zero."""
        queries = _matrix(queries, self.dim)
        if metric not in ('inner_product', 'cosine'):
            raise ValueError("metric must be 'inner_product' or 'cosine'")
        if not isinstance(k, int) or not 1 <= k <= self.rows:
            raise ValueError("k must lie between 1 and the gallery size")
        if not isinstance(block_size, int) or block_size <= 0:
            raise ValueError("block_size must be positive")
        q = _tensor(queries, self.device)
        if metric == 'cosine':
            # Normalize before any rotation, including extreme finite queries.
            q = _unit_tensor_rows(q)
        q = self._query_tensor(q)
        best_scores = np.empty((len(queries), 0), dtype='float32')
        best_ids = np.empty((len(queries), 0), dtype='int64')
        max_decode = max_scores = 0
        for start in range(0, self.rows, block_size):
            stop = min(start + block_size, self.rows)
            block = self.decode_block(start, stop, rotated=True)
            if metric == 'cosine':
                block = _unit_tensor_rows(block)
            scores = (q @ block.transpose()).numpy()
            max_decode = max(max_decode, (stop - start) * self.dim * 4)
            max_scores = max(max_scores, scores.nbytes)
            ids = np.broadcast_to(np.arange(start, stop), scores.shape)
            candidates = np.concatenate((best_scores, scores), axis=1)
            candidate_ids = np.concatenate((best_ids, ids), axis=1)
            # Stable row IDs resolve equal-score ties independently of block size.
            order = np.lexsort((candidate_ids, -candidates), axis=1)[:, :k]
            best_scores = np.take_along_axis(candidates, order, axis=1)
            best_ids = np.take_along_axis(candidate_ids, order, axis=1)
            del block, scores
        return SearchResult(best_ids, best_scores, max_decode, max_scores)


class TurboQuantMSE:
    def __init__(self, dim, bits, seed=0, device=None):
        _parameters(dim, bits)
        self.dim, self.bits, self.device = dim, bits, _device(device)
        started = time.perf_counter()
        self.codebook = sphere_codebook(dim, bits)
        self.codebook_setup_seconds = time.perf_counter() - started
        started = time.perf_counter()
        matrix = np.random.default_rng(seed).normal(size=(dim, dim))
        rotation, triangular = np.linalg.qr(matrix)
        rotation *= np.where(np.diag(triangular) >= 0, 1., -1.)
        self.rotation = _tensor(rotation.astype('float32'), self.device)
        self.rotation_setup_seconds = time.perf_counter() - started

    def encode(self, values):
        values = _matrix(values, self.dim)
        norm64 = np.linalg.norm(values.astype('float64'), axis=1)
        if np.any(norm64 > np.finfo('float32').max):
            raise ValueError("Row norm exceeds float32 range")
        norms = norm64.astype('float32')
        unit = np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] != 0)
        rotated = (_tensor(unit, self.device) @ self.rotation).realize_cached_data().compact()
        if self.device == nd.cpu():
            packed = self.device.quantize_pack(rotated._handle, self.codebook, len(values), self.dim, self.bits)
        else:
            boundaries = (self.codebook[:-1] + self.codebook[1:]) * .5
            codes = np.searchsorted(boundaries, rotated.numpy(), side='right').astype('uint8')
            packed = _pack(codes, self.bits, self.device)
        return PackedVectorStore(self, packed, norms)


class PackedVectorStore(_VectorStore):
    def __init__(self, quantizer, packed, norms):
        self.quantizer = quantizer
        self.dim, self.bits, self.device = quantizer.dim, quantizer.bits, quantizer.device
        self.packed, self.norms = packed, norms
        self.rows = len(norms)

    @property
    def storage_bytes(self):
        return dict(codes=self.packed.nbytes, norms=self.norms.nbytes,
                    rotation=self.dim * self.dim * 4, codebook=self.quantizer.codebook.nbytes)

    def _query_tensor(self, values):
        return values @ self.quantizer.rotation

    def decode_block(self, start, stop, rotated=False):
        self._bounds(start, stop)
        if self.device == nd.cpu():
            values = nd.empty((stop - start, self.dim), device=self.device)
            self.device.decode_centroids(self.packed, self.quantizer.codebook, self.norms,
                                          start, stop, self.dim, self.bits, values._handle)
            block = Tensor.make_const(values)
        else:
            codes = _indices(self.packed[start:stop], self.dim, self.bits)
            block = _tensor(self.quantizer.codebook[codes] * self.norms[start:stop, None], self.device)
        return block if rotated else block @ self.quantizer.rotation.transpose()

    def save(self, path):
        np.savez(path, format_version=np.int32(1), bits=np.int32(self.bits), packed=self.packed,
                 norms=self.norms, rotation=self.quantizer.rotation.numpy(), codebook=self.quantizer.codebook)

    @classmethod
    def load(cls, path, device=None):
        with np.load(path, allow_pickle=False) as source:
            version, bits = int(source['format_version']), int(source['bits'])
            packed, norms = source['packed'].copy(), source['norms'].copy()
            rotation, codebook = source['rotation'].copy(), source['codebook'].copy()
        if version != 1 or rotation.ndim != 2 or rotation.shape[0] != rotation.shape[1]:
            raise ValueError("Unsupported serialized quantizer")
        dim = len(rotation)
        _parameters(dim, bits)
        if (norms.dtype != np.float32 or norms.ndim != 1 or norms.size == 0
                or packed.dtype != np.uint8 or packed.shape != (norms.size, (dim * bits + 7) // 8)
                or codebook.shape != (1 << bits,) or codebook.dtype != np.float32
                or rotation.dtype != np.float32 or not np.isfinite(rotation).all()
                or not np.isfinite(norms).all() or np.any(norms < 0)
                or not np.isfinite(codebook).all() or not np.all(np.diff(codebook) > 0)
                or not np.allclose(rotation.T @ rotation, np.eye(dim), atol=2e-5)):
            raise ValueError("Invalid serialized buffers")
        quantizer = TurboQuantMSE.__new__(TurboQuantMSE)
        quantizer.dim, quantizer.bits, quantizer.device = dim, bits, _device(device)
        quantizer.rotation = _tensor(rotation, quantizer.device)
        quantizer.codebook = codebook
        quantizer.codebook_setup_seconds = quantizer.rotation_setup_seconds = 0.
        return cls(quantizer, packed, norms)


class UniformQuantizer:
    """Per-row affine quantization with min/scale metadata and identical packing."""
    def __init__(self, dim, bits, device=None):
        _parameters(dim, bits)
        self.dim, self.bits, self.device = dim, bits, _device(device)

    def encode(self, values):
        values = _matrix(values, self.dim)
        minimum, maximum = values.min(axis=1), values.max(axis=1)
        scale = (maximum - minimum) / ((1 << self.bits) - 1)
        normalized = np.divide(values - minimum[:, None], scale[:, None],
                               out=np.zeros_like(values), where=scale[:, None] != 0)
        codes = np.clip(np.rint(normalized), 0, (1 << self.bits) - 1).astype('uint8')
        return UniformVectorStore(self, _pack(codes, self.bits, self.device), minimum, scale)


class UniformVectorStore(_VectorStore):
    def __init__(self, quantizer, packed, minimum, scale):
        self.dim, self.bits, self.device = quantizer.dim, quantizer.bits, quantizer.device
        self.packed, self.minimum, self.scale = packed, minimum, scale
        self.rows = len(minimum)

    @property
    def storage_bytes(self):
        return dict(codes=self.packed.nbytes, minimum=self.minimum.nbytes, scale=self.scale.nbytes)

    def decode_block(self, start, stop, rotated=False):
        self._bounds(start, stop)
        if self.device == nd.cpu():
            values = nd.empty((stop - start, self.dim), device=self.device)
            self.device.decode_uniform(self.packed, self.minimum, self.scale,
                                       start, stop, self.dim, self.bits, values._handle)
            return Tensor.make_const(values)
        codes = _indices(self.packed[start:stop], self.dim, self.bits)
        return _tensor(self.minimum[start:stop, None] + self.scale[start:stop, None] * codes, self.device)


class FloatVectorStore(_VectorStore):
    def __init__(self, values, device=None):
        values = _matrix(values)
        self.rows, self.dim = values.shape
        self.device = _device(device)
        self.values = _tensor(values.copy(), self.device).realize_cached_data()

    @property
    def storage_bytes(self):
        return dict(vectors=self.rows * self.dim * 4)

    def decode_block(self, start, stop, rotated=False):
        self._bounds(start, stop)
        return Tensor.make_const(self.values[start:stop, :])
