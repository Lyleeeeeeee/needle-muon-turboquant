"""Pickle-free inference checkpoints for the shared Needle encoder."""
from pathlib import Path

import numpy as np
import needle as ndl

from models import SmallEncoder


LAYERS = ('input', 'hidden', 'classifier')


def save_encoder(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{layer}.{name}': getattr(getattr(model, layer), name).numpy()
              for layer in LAYERS for name in ('weight', 'bias')}
    np.savez(path, format_version=np.int32(1), **arrays)


def load_encoder(path, device):
    with np.load(path, allow_pickle=False) as archive:
        if archive['format_version'].shape != () or int(archive['format_version']) != 1:
            raise ValueError('Unsupported encoder checkpoint version')
        arrays = {f'{layer}.{name}': archive[f'{layer}.{name}'].copy()
                  for layer in LAYERS for name in ('weight', 'bias')}
    weights = [arrays[f'{layer}.weight'] for layer in LAYERS]
    if any(w.ndim != 2 or min(w.shape) <= 0 for w in weights):
        raise ValueError('Encoder weights must be nonempty matrices')
    if weights[0].shape[1] != weights[1].shape[0] or weights[1].shape[1] != weights[2].shape[0]:
        raise ValueError('Encoder layer dimensions do not match')
    for layer, weight in zip(LAYERS, weights):
        bias = arrays[f'{layer}.bias']
        if bias.shape != (weight.shape[1],):
            raise ValueError('Encoder bias shape does not match its weight')
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Checkpoint parameters must be finite float32 arrays')
    model = SmallEncoder(input_dim=weights[0].shape[0], hidden_dim=weights[0].shape[1],
                         feature_dim=weights[1].shape[1], num_classes=weights[2].shape[1],
                         device=device)
    for layer in LAYERS:
        for name in ('weight', 'bias'):
            getattr(getattr(model, layer), name).data = ndl.Tensor(
                arrays[f'{layer}.{name}'], device=device, requires_grad=False)
    model.eval()
    return model
