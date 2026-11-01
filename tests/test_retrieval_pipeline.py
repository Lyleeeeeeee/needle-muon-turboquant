import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'apps'), str(ROOT / 'experiments')]


def test_fashion_export_excludes_validation_and_identifies_own_encoder(tmp_path):
    import needle as ndl
    from encoder_checkpoint import save_encoder
    from models import SmallEncoder

    data = tmp_path / 'fashion'
    data.mkdir()
    rng = np.random.default_rng(3)
    for prefix, count in [('train', 32), ('t10k', 12)]:
        images = rng.integers(0, 256, (count, 28, 28), dtype=np.uint8)
        labels = np.arange(count, dtype=np.uint8) % 10
        (data / f'{prefix}-images-idx3-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
        (data / f'{prefix}-labels-idx1-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    checkpoint, output = tmp_path / 'model.npz', tmp_path / 'features.npz'
    save_encoder(SmallEncoder(hidden_dim=8, feature_dim=4, device=ndl.cpu()), checkpoint)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/export_features.py'), '--checkpoint', str(checkpoint),
         '--dataset', 'Fashion-MNIST', '--data-dir', str(data), '--validation-size', '8',
         '--split-seed', '17', '--reference-train', '5', '--reference-queries', '3',
         '--gallery-size', '24', '--queries', '8', '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads(output.with_suffix('.json').read_text())
    split = np.random.default_rng(17).permutation(32)
    with np.load(output, allow_pickle=False) as features:
        assert set(features['gallery_ids']) == set(split[8:])
        assert not set(features['gallery_ids']) & set(split[:8])
        assert features['gallery'].shape == (24, 4)
        assert features['queries'].shape == (8, 4)
        assert str(features['source']).startswith('Fashion-MNIST_frozen_encoder_')
        assert metadata['gallery_ids_sha256'] == hashlib.sha256(features['gallery_ids'].astype('int64').tobytes()).hexdigest()
    assert metadata['dataset'] == 'Fashion-MNIST'
    assert metadata['validation_size_excluded'] == 8
    assert metadata['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert metadata['feature_archive_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert set(metadata['data_files_sha256']) == {p.name for p in data.iterdir()}


def test_benchmark_can_run_only_selected_methods(tmp_path):
    rng = np.random.default_rng(2)
    features, output = tmp_path / 'features.npz', tmp_path / 'report.json'
    np.savez(features, gallery=rng.normal(size=(32, 4)).astype('float32'),
             queries=rng.normal(size=(5, 4)).astype('float32'))
    command = [sys.executable, str(ROOT / 'apps/benchmark_quantization.py'),
               '--features', str(features), '--k', '3', '--block-size', '16',
               '--repeats', '1', '--metric', 'cosine', '--methods', 'fp32', 'turbo_2',
               '--output', str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert [r['method'] for r in record['results']] == ['fp32', 'turbo_2']
    assert record['results'][0]['recall_at_k'] == 1.
    assert all(r['search_metric'] == 'cosine' for r in record['results'])


def test_benchmark_rejects_duplicate_methods(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/benchmark_quantization.py'), '--methods', 'fp32', 'fp32'],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'unique' in result.stderr


def test_joint_summary_averages_rotations_before_training_seeds():
    from common import aggregate_quality

    rows = []
    for encoder in ('adamw', 'muon'):
        for seed, baseline in zip((8, 9, 10), (.8, .9, 1.)):
            if encoder == 'muon':
                baseline -= .1
            common = dict(encoder=encoder, training_seed=seed, stored_bytes=100,
                          query_seconds_median=1., recall_at_k=1.)
            rows.append(dict(common, method='fp32', rotation_seed=7, label_precision_at_k=baseline))
            drops = (.1, .2, .3) if encoder == 'adamw' else (.05, .1, .15)
            for rotation, drop in zip((7, 8, 9), drops):
                rows.append(dict(common, method='turbo_2', rotation_seed=rotation,
                                 label_precision_at_k=baseline-drop, recall_at_k=.6))
    result = aggregate_quality(rows)
    a = result['encoders']['adamw']['turbo_2']
    assert a['label_precision_at10']['mean'] == pytest.approx(.7)
    assert a['label_precision_at10']['sample_std'] == pytest.approx(.1)
    assert len(a['label_precision_at10']['values']) == 3
    assert a['precision_change_vs_own_fp32']['mean'] == pytest.approx(-.2)
    assert result['interaction']['turbo_2']['mean'] == pytest.approx(.1)
