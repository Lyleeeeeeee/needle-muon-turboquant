"""CIFAR parsing, model/reference gradients and checkpoint contracts."""
"""CIFAR binary parsing, native patch model and reproducible bounded pilot."""

import hashlib

import json

from pathlib import Path

import subprocess

import sys

import numpy as np

import pytest

import needle as ndl

from needle import nn

ROOT = Path(__file__).resolve().parents[1]

def synthetic_binary(path, classes=3, per_class=6):
    rng = np.random.default_rng(11)
    labels = np.repeat(np.arange(classes, dtype='uint8'), per_class)
    pixels = rng.integers(0, 256, (len(labels), 3, 32, 32), dtype='uint8')
    rows = np.column_stack([labels // 5, labels, pixels.reshape(len(labels), -1)])
    path.write_bytes(rows.tobytes())
    return pixels, labels

def test_cifar_binary_and_patch_layout_preserve_channels_pixels_and_labels(tmp_path):
    from cifar_data import read_binary, patches
    path = tmp_path / 'train.bin'
    pixels, labels = synthetic_binary(path)
    x, y = read_binary(path)
    np.testing.assert_array_equal(x, pixels)
    np.testing.assert_array_equal(y, labels)
    z = patches(x, np.zeros(3, dtype='float32'), np.ones(3, dtype='float32'), patch_size=8)
    assert z.shape == (18, 16, 192) and z.dtype == np.float32
    np.testing.assert_allclose(z[0, 0], pixels[0, :, :8, :8].transpose(1, 2, 0).reshape(-1) / 255, atol=1e-7)
    np.testing.assert_allclose(z[0, 1], pixels[0, :, :8, 8:16].transpose(1, 2, 0).reshape(-1) / 255, atol=1e-7)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError): read_binary(path)

def test_stratified_pilot_split_is_disjoint_balanced_and_deterministic():
    from cifar_data import stratified_split
    labels = np.repeat(np.arange(3), 9)
    train, val = stratified_split(labels, 4, 2, classes=3, seed=17)
    assert not set(train) & set(val)
    np.testing.assert_array_equal(np.bincount(labels[train]), [4, 4, 4])
    np.testing.assert_array_equal(np.bincount(labels[val]), [2, 2, 2])
    assert all(np.array_equal(a, b) for a, b in zip((train, val), stratified_split(labels, 4, 2, classes=3, seed=17)))
    with pytest.raises(ValueError): stratified_split(labels, 8, 2, classes=3)

def test_native_patch_mixer_reference_forward_and_all_hidden_gradients():
    from cifar_model import PatchMixer
    np.random.seed(3)
    model = PatchMixer(patch_dim=6, tokens=4, width=8, depth=1,
                       token_hidden=8, channel_hidden=16, classes=3, device=ndl.cpu())
    x = np.random.default_rng(8).normal(size=(5, 4, 6)).astype('float32')
    def linear(x, layer): return x @ layer.weight.numpy() + layer.bias.numpy()
    def norm(x, layer):
        x = x - x.mean(axis=-1, keepdims=True)
        return x / np.sqrt((x*x).mean(axis=-1, keepdims=True) + layer.eps) * layer.weight.numpy() + layer.bias.numpy()
    y = linear(x, model.patch)
    b = model.blocks[0]
    t = norm(y, b.token_norm).transpose(0, 2, 1)
    y += linear(np.maximum(linear(t, b.token_in), 0), b.token_out).transpose(0, 2, 1)
    c = norm(y, b.channel_norm)
    y += linear(np.maximum(linear(c, b.channel_in), 0), b.channel_out)
    expected = norm(y, model.final_norm).mean(axis=1)
    tensor = ndl.Tensor(x, device=ndl.cpu(), requires_grad=False)
    np.testing.assert_allclose(model.features(tensor).numpy(), expected, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(model(tensor).numpy(), linear(expected, model.classifier), atol=3e-5, rtol=3e-5)
    assert (expected < 0).any() and (expected > 0).any()
    labels = ndl.Tensor(np.arange(5) % 3, device=ndl.cpu(), dtype='float32', requires_grad=False)
    nn.SoftmaxLoss()(model(tensor), labels).backward()
    hidden = model.muon_parameters()
    assert len(hidden) == 4
    assert all(p.grad is not None and np.isfinite(p.grad.numpy()).all() and np.linalg.norm(p.grad.numpy()) > 0 for p in hidden)
    assert all(p is not model.patch.weight and p is not model.classifier.weight for p in hidden)
    full = PatchMixer(device=ndl.cpu())
    assert sum(p.numpy().size for p in full.muon_parameters()) / sum(p.numpy().size for p in full.parameters()) > .6

def test_patch_checkpoint_exact_roundtrip_and_shape_rejection(tmp_path):
    from cifar_model import PatchMixer, load_checkpoint, save_checkpoint
    model = PatchMixer(patch_dim=6, tokens=4, width=8, depth=1,
                       token_hidden=8, channel_hidden=16, classes=3, device=ndl.cpu())
    path = tmp_path / 'model.npz'
    save_checkpoint(model, path)
    restored = load_checkpoint(path, ndl.cpu())
    for a, b in zip(model.parameters(), restored.parameters()): np.testing.assert_array_equal(a.numpy(), b.numpy())
    with np.load(path, allow_pickle=False) as source: arrays = {k: source[k].copy() for k in source.files}
    arrays['p0'] = np.zeros((1, 1), dtype='float32')
    np.savez(path, **arrays)
    with pytest.raises(ValueError): load_checkpoint(path, ndl.cpu())

def test_patch_mixer_gradient_matches_finite_differences():
    from cifar_model import PatchMixer
    np.random.seed(14)
    model = PatchMixer(patch_dim=6, tokens=4, width=8, depth=1,
                       token_hidden=8, channel_hidden=16, classes=3, device=ndl.cpu())
    x = ndl.Tensor(np.random.default_rng(9).normal(size=(3, 4, 6)).astype('float32'), device=ndl.cpu(), requires_grad=False)
    y = ndl.Tensor(np.arange(3), dtype='float32', device=ndl.cpu(), requires_grad=False)
    loss = nn.SoftmaxLoss()(model(x), y)
    loss.backward()
    for p in [model.patch.weight, model.blocks[0].token_in.weight, model.final_norm.weight]:
        values, gradient = p.numpy().copy(), p.grad.numpy().copy()
        index = np.unravel_index(np.abs(gradient).argmax(), gradient.shape)
        samples = []
        for delta in [-.001, .001]:
            perturbed = values.copy();perturbed[index] += delta
            p.data = ndl.Tensor(perturbed, device=ndl.cpu(), requires_grad=False)
            samples.append(float(nn.SoftmaxLoss()(model(x), y).numpy().item()))
        p.data = ndl.Tensor(values, device=ndl.cpu(), requires_grad=False)
        assert (samples[1] - samples[0]) / .002 == pytest.approx(float(gradient[index]), rel=.03, abs=.003)
