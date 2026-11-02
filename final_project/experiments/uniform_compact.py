"""Uniform baselines with shared or derived row metadata and native decoding.

Mode 0 retains affine minima; mode 1 shares an identical minimum; mode 2 uses
a symmetric grid whose minimum is derived from its row scale. Derived arrays
are temporary decoder inputs, never retained gallery buffers.
"""
from pathlib import Path
import numpy as np
import needle as ndl
from needle.quantization import UniformQuantizer, UniformVectorStore, _matrix, _pack


class CompactUniformQuantizer(UniformQuantizer):
    def __init__(self, dim, bits, device=None, symmetric=False):
        super().__init__(dim, bits, device=device)
        self.symmetric = symmetric

    def encode(self, values):
        if not self.symmetric:
            original = super().encode(values)
            shared = np.all(original.minimum == original.minimum[0])
            payload = original.minimum[:1].copy() if shared else original.minimum
            return CompactUniformStore(self, original.packed, original.scale, payload, 1 if shared else 0)
        values = _matrix(values, self.dim)
        levels = (1 << self.bits) - 1
        scale = (2 * np.abs(values).max(axis=1) / levels).astype('float32')
        minimum = -scale * np.float32(levels / 2)
        scaled = np.divide(values - minimum[:, None], scale[:, None], out=np.zeros_like(values),
                           where=scale[:, None] != 0)
        codes = np.clip(np.rint(scaled), 0, levels).astype('uint8')
        return CompactUniformStore(self, _pack(codes, self.bits, self.device), scale,
                                   np.empty(0, dtype='float32'), 2)


class CompactUniformStore(UniformVectorStore):
    def __init__(self, quantizer, packed, scale, payload, mode):
        self.dim, self.bits, self.device = quantizer.dim, quantizer.bits, quantizer.device
        self.packed, self.scale, self.payload = packed, scale, payload
        self.mode = np.array([mode], dtype='uint8')
        self.rows = len(scale)

    @property
    def minimum(self):
        if self.mode[0] == 1:
            return np.full(self.rows, self.payload[0], dtype='float32')
        if self.mode[0] == 2:
            return -self.scale * np.float32(((1 << self.bits) - 1) / 2)
        return self.payload

    @property
    def storage_bytes(self):
        return dict(codes=self.packed.nbytes, scale=self.scale.nbytes,
                    minimum_payload=self.payload.nbytes, mode=self.mode.nbytes)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, dim=np.int32(self.dim), bits=np.int32(self.bits), packed=self.packed,
                 scale=self.scale, minimum_payload=self.payload, mode=self.mode)

    @classmethod
    def load(cls, path, device=None):
        with np.load(path, allow_pickle=False) as data:
            quantizer = CompactUniformQuantizer(int(data['dim']), int(data['bits']), device=device)
            packed, scale, payload, mode = [data[k].copy() for k in ('packed', 'scale', 'minimum_payload', 'mode')]
        if (scale.ndim != 1 or not len(scale) or scale.dtype != np.float32 or not np.isfinite(scale).all()
                or (scale < 0).any() or mode.shape != (1,) or mode.dtype != np.uint8 or mode[0] > 2
                or packed.dtype != np.uint8 or packed.shape != (len(scale), (quantizer.dim*quantizer.bits+7)//8)
                or payload.dtype != np.float32 or not np.isfinite(payload).all()
                or payload.shape != ((len(scale),) if mode[0] == 0 else (1,) if mode[0] == 1 else (0,))):
            raise ValueError('Compact uniform archive shape, dtype or metadata mode differs')
        return cls(quantizer, packed, scale, payload, int(mode[0]))
