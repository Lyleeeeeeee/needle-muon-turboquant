"""Small MLP-Mixer-inspired image encoder using existing native Needle operators.

This is a CPU pilot adaptation with ReLU, not the paper's pretrained models.
"""
import json
from pathlib import Path
import numpy as np
import needle as ndl
from needle import nn


class MixerBlock(nn.Module):
    def __init__(self, tokens, width, token_hidden, channel_hidden, device):
        super().__init__()
        self.tokens, self.width = tokens, width
        self.token_norm = nn.LayerNorm1d(width, device=device)
        self.channel_norm = nn.LayerNorm1d(width, device=device)
        self.token_in = nn.Linear(tokens, token_hidden, device=device)
        self.token_out = nn.Linear(token_hidden, tokens, device=device)
        self.channel_in = nn.Linear(width, channel_hidden, device=device)
        self.channel_out = nn.Linear(channel_hidden, width, device=device)
        self.activation = nn.ReLU()

    def forward(self, x):
        batch, tokens, width = x.shape
        y = self.token_norm(x.reshape((batch * tokens, width))).reshape(x.shape)
        y = y.transpose((1, 2)).reshape((batch * width, tokens))
        y = self.token_out(self.activation(self.token_in(y)))
        x = x + y.reshape((batch, width, tokens)).transpose((1, 2))
        y = self.channel_norm(x.reshape((batch * tokens, width)))
        y = self.channel_out(self.activation(self.channel_in(y)))
        return x + y.reshape(x.shape)

    def muon_parameters(self):
        return [layer.weight for layer in (self.token_in, self.token_out, self.channel_in, self.channel_out)]


class PatchMixer(nn.Module):
    def __init__(self, patch_dim=192, tokens=16, width=64, depth=2,
                 token_hidden=32, channel_hidden=128, classes=100, device=None):
        super().__init__()
        self.config = dict(patch_dim=patch_dim, tokens=tokens, width=width, depth=depth,
                           token_hidden=token_hidden, channel_hidden=channel_hidden, classes=classes)
        if any(type(v) is not int or v <= 0 for v in self.config.values()):
            raise ValueError('Patch model dimensions must be positive integers')
        self.patch = nn.Linear(patch_dim, width, device=device)
        self.blocks = [MixerBlock(tokens, width, token_hidden, channel_hidden, device) for _ in range(depth)]
        self.final_norm = nn.LayerNorm1d(width, device=device)
        self.classifier = nn.Linear(width, classes, device=device)

    def features(self, x):
        batch, tokens, dim = x.shape
        if (tokens, dim) != (self.config['tokens'], self.config['patch_dim']):
            raise ValueError('Patch input dimensions differ from the encoder')
        width = self.config['width']
        y = self.patch(x.reshape((batch * tokens, dim))).reshape((batch, tokens, width))
        for block in self.blocks:
            y = block(y)
        y = self.final_norm(y.reshape((batch * tokens, width))).reshape((batch, tokens, width))
        return y.sum(axes=(1,)) / tokens

    def muon_parameters(self):
        return [p for block in self.blocks for p in block.muon_parameters()]

    def forward(self, x):
        return self.classifier(self.features(x))


def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, format=np.array('cifar_patch_mixer_v1'), config=np.array(json.dumps(model.config, sort_keys=True)),
             **{f'p{i}': p.numpy() for i, p in enumerate(model.parameters())})


def load_checkpoint(path, device):
    with np.load(path, allow_pickle=False) as archive:
        if str(archive['format']) != 'cifar_patch_mixer_v1':
            raise ValueError('Unsupported CIFAR checkpoint format')
        model = PatchMixer(**json.loads(str(archive['config'])), device=device)
        params = model.parameters()
        if set(archive.files) != {'format', 'config', *[f'p{i}' for i in range(len(params))]}:
            raise ValueError('CIFAR checkpoint parameter coverage differs')
        for i, p in enumerate(params):
            values = archive[f'p{i}']
            if values.shape != p.shape or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError('CIFAR checkpoint shape, dtype or finiteness differs')
            p.data = ndl.Tensor(values.copy(), device=device, requires_grad=False)
    model.eval()
    return model
