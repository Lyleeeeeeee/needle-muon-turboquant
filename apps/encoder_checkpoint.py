"""Pickle-free inference checkpoints for the shared Needle encoder."""
from pathlib import Path

import numpy as np
import needle as ndl

from models import SmallEncoder


LAYERS = ('input', 'hidden', 'classifier')


def layer_items(model):
    return ([('input', model.input)] +
            [(f'extra.{i}', layer) for i, layer in enumerate(model.extra)] +
            [('hidden', model.hidden), ('classifier', model.classifier)])


def save_encoder(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{prefix}.{name}': getattr(layer, name).numpy()
              for prefix, layer in layer_items(model) for name in ('weight', 'bias')}
    metadata = {'format_version': np.int32(2 if model.extra_hidden_layers else 1)}
    if model.extra_hidden_layers:
        metadata['extra_hidden_layers'] = np.int32(model.extra_hidden_layers)
    np.savez(path, **metadata, **arrays)


def load_encoder(path, device):
    with np.load(path, allow_pickle=False) as archive:
        version = archive['format_version']
        if version.shape != () or version.dtype.kind not in 'iu' or int(version) not in (1, 2):
            raise ValueError('Unsupported encoder checkpoint version')
        extra = 0
        if int(version) == 2:
            count = archive['extra_hidden_layers']
            if count.shape != () or count.dtype.kind not in 'iu' or not 0 < int(count) <= len(archive.files) // 2:
                raise ValueError('Invalid checkpoint extra depth')
            extra = int(count)
        layers = ['input'] + [f'extra.{i}' for i in range(extra)] + ['hidden', 'classifier']
        arrays = {f'{layer}.{name}': archive[f'{layer}.{name}'].copy()
                  for layer in layers for name in ('weight', 'bias')}
    weights = [arrays[f'{layer}.weight'] for layer in layers]
    if any(w.ndim != 2 or min(w.shape) <= 0 for w in weights):
        raise ValueError('Encoder weights must be nonempty matrices')
    if any(a.shape[1] != b.shape[0] for a, b in zip(weights, weights[1:])):
        raise ValueError('Encoder layer dimensions do not match')
    if any(w.shape != (weights[0].shape[1], weights[0].shape[1]) for w in weights[1:1 + extra]):
        raise ValueError('Extra encoder layer shape does not match its hidden width')
    for layer, weight in zip(layers, weights):
        bias = arrays[f'{layer}.bias']
        if bias.shape != (weight.shape[1],):
            raise ValueError('Encoder bias shape does not match its weight')
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Checkpoint parameters must be finite float32 arrays')
    model = SmallEncoder(input_dim=weights[0].shape[0], hidden_dim=weights[0].shape[1],
                         feature_dim=weights[-2].shape[1], num_classes=weights[-1].shape[1],
                         device=device, extra_hidden_layers=extra)
    for prefix, layer in layer_items(model):
        for name in ('weight', 'bias'):
            getattr(layer, name).data = ndl.Tensor(
                arrays[f'{prefix}.{name}'], device=device, requires_grad=False)
    model.eval()
    return model
