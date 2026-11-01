import json
import gzip
from pathlib import Path
import subprocess
import struct
import sys

import numpy as np
import pytest
import needle as ndl
from needle import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps'))


def test_deep_encoder_groups_all_hidden_matrices_and_backpropagates():
    from models import SmallEncoder

    np.random.seed(3)
    model = SmallEncoder(input_dim=4, hidden_dim=8, feature_dim=4,
                         num_classes=3, extra_hidden_layers=2, device=ndl.cpu())
    matrices = model.muon_parameters()
    assert [p.shape for p in matrices] == [(8, 8), (8, 8), (8, 4)]
    assert len({id(p) for p in matrices}) == 3
    params = model.parameters()
    assert all(any(p is q for q in params) for p in matrices)
    assert not any(p is model.input.weight or p is model.classifier.weight for p in matrices)
    x = ndl.Tensor(np.random.default_rng(3).normal(size=(16, 4)).astype('float32'),
                   device=ndl.cpu(), requires_grad=False)
    y = ndl.Tensor(np.arange(16) % 3, dtype='float32', device=ndl.cpu(), requires_grad=False)
    nn.SoftmaxLoss()(model(x), y).backward()
    for p in matrices:
        assert p.grad is not None
        assert np.linalg.norm(p.grad.numpy()) > 0
    full = SmallEncoder(extra_hidden_layers=2, device=ndl.cpu())
    assert sum(np.prod(p.shape) for p in full.parameters()) == 142410
    assert sum(np.prod(p.shape) for p in full.muon_parameters()) == 40960


def test_deep_checkpoint_restores_every_layer_and_rejects_bad_shapes(tmp_path):
    from models import SmallEncoder
    from encoder_checkpoint import save_encoder, load_encoder

    model = SmallEncoder(input_dim=4, hidden_dim=8, feature_dim=4,
                         num_classes=3, extra_hidden_layers=2, device=ndl.cpu())
    path = tmp_path / 'deep.npz'
    save_encoder(model, path)
    restored = load_encoder(path, ndl.cpu())
    assert restored.extra_hidden_layers == 2
    for a, b in zip(model.parameters(), restored.parameters()):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
    x = ndl.Tensor(np.random.default_rng(1).normal(size=(16, 4)).astype('float32'),
                   device=ndl.cpu(), requires_grad=False)
    np.testing.assert_array_equal(model.features(x).numpy(), restored.features(x).numpy())
    with np.load(path, allow_pickle=False) as archive:
        assert int(archive['format_version']) == 2
        assert int(archive['extra_hidden_layers']) == 2
        content = {key: archive[key].copy() for key in archive.files}
    content['extra.0.weight'] = np.zeros((8, 7), dtype='float32')
    np.savez(path, **content)
    with pytest.raises(ValueError, match='dimensions|shape'):
        load_encoder(path, ndl.cpu())


def test_shallow_checkpoint_stays_version_one_and_depth_is_validated(tmp_path):
    from models import SmallEncoder
    from encoder_checkpoint import save_encoder, load_encoder

    model = SmallEncoder(device=ndl.cpu())
    assert model.extra_hidden_layers == 0
    path = tmp_path / 'old.npz'
    save_encoder(model, path)
    with np.load(path, allow_pickle=False) as archive:
        assert int(archive['format_version']) == 1
        assert 'extra_hidden_layers' not in archive.files
    assert load_encoder(path, ndl.cpu()).extra_hidden_layers == 0
    for depth in (-1, .5, True):
        with pytest.raises(ValueError):
            SmallEncoder(extra_hidden_layers=depth)


def test_deep_training_cli_records_depth_and_all_muon_parameters(tmp_path):
    output, checkpoint = tmp_path / 'train.json', tmp_path / 'best.npz'
    data = tmp_path / 'data'
    data.mkdir()
    pixels = np.random.default_rng(4).integers(0, 256, (40, 28, 28), dtype=np.uint8)
    (data / 'train-images-idx3-ubyte.gz').write_bytes(
        gzip.compress(struct.pack('>IIII', 2051, 40, 28, 28) + pixels.tobytes()))
    (data / 'train-labels-idx1-ubyte.gz').write_bytes(
        gzip.compress(struct.pack('>II', 2049, 40) + (np.arange(40, dtype=np.uint8) % 10).tobytes()))
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--device', 'cpu',
         '--dataset', 'Fashion-MNIST', '--data-dir', str(data),
         '--optimizer', 'muon', '--extra-hidden-layers', '2',
         '--train-limit', '24', '--validation-size', '16', '--eval-limit', '16',
         '--epochs', '1', '--seed', '7', '--lr', '.001', '--muon-lr', '.007',
         '--best-checkpoint-output', str(checkpoint), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    r = json.loads(output.read_text())
    assert r['extra_hidden_layers'] == 2
    assert r['muon_parameter_shapes'] == [[128, 128], [128, 128], [128, 64]]
    assert r['total_parameter_count'] == 142410 and r['muon_parameter_count'] == 40960
    assert r['muon_parameter_fraction'] == pytest.approx(40960 / 142410)
