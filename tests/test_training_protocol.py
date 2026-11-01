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
sys.path.insert(0, str(ROOT / 'experiments'))


def test_fashion_cli_records_dataset_and_feature_provenance(tmp_path):
    data = tmp_path / 'fashion'
    data.mkdir()
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8)
    labels = np.arange(40, dtype=np.uint8) % 10
    image_path = data / 'train-images-idx3-ubyte.gz'
    label_path = data / 'train-labels-idx1-ubyte.gz'
    image_path.write_bytes(gzip.compress(struct.pack('>IIII', 2051, 40, 28, 28) + images.tobytes()))
    label_path.write_bytes(gzip.compress(struct.pack('>II', 2049, 40) + labels.tobytes()))
    output, features = tmp_path / 'record.json', tmp_path / 'features.npz'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--device', 'cpu',
         '--dataset', 'Fashion-MNIST', '--data-dir', str(data), '--train-limit', '30',
         '--validation-size', '10', '--eval-limit', '10', '--epochs', '1',
         '--batch-size', '8', '--hidden-dim', '16', '--feature-dim', '8',
         '--feature-output', str(features), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record['dataset'] == 'Fashion-MNIST'
    assert record['validation_source'] == 'train_holdout'
    assert record['data_files_sha256'] == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (image_path, label_path)}
    with np.load(features, allow_pickle=False) as feature_data:
        assert str(feature_data['source']) == 'Fashion-MNIST_encoder'


def test_candidate_selection_uses_validation_loss_not_accuracy_or_test():
    from common import select_configuration

    records = [
        {'candidate': 'high', 'best_validation_loss': .2, 'best_validation_accuracy': .99,
         'test_accuracy': 1.},
        {'candidate': 'low', 'best_validation_loss': .1, 'best_validation_accuracy': .8,
         'test_accuracy': .1},
        {'candidate': 'medium', 'best_validation_loss': .1, 'best_validation_accuracy': .9}]
    assert select_configuration(records)['candidate'] == 'low'
    assert select_configuration(list(reversed(records)))['candidate'] == 'low'


def test_candidate_selection_rejects_nonfinite_or_duplicate_records():
    from common import select_configuration

    with pytest.raises(ValueError):
        select_configuration([{'candidate': 'low', 'best_validation_loss': float('nan')}])
    with pytest.raises(ValueError):
        select_configuration([{'candidate': 'low', 'best_validation_loss': .1}] * 2)


def test_pair_validation_rejects_data_or_order_mismatch():
    from common import validate_pair

    shared = dict(initial_parameter_sha256='init', train_indices_sha256='train',
                  validation_indices_sha256='validation', training_order_sha256='order',
                  data_files_sha256={'train': 'data'}, dataset='MNIST', seed=8,
                  train_samples=55000, eval_samples=5000, hidden_dim=128, feature_dim=64,
                  batch_size=64, validation_source='train_holdout',
                  initial_validation_loss=2.5, initial_validation_accuracy=.1)
    validate_pair(shared, dict(shared))
    for key, value in [('training_order_sha256', 'other'), ('dataset', 'Fashion-MNIST'),
                       ('data_files_sha256', {'train': 'different'})]:
        with pytest.raises(ValueError, match=key):
            validate_pair(shared, dict(shared, **{key: value}))


def test_repeat_summary_uses_selected_checkpoints_and_sample_deviation():
    from common import summarize_runs

    records = [dict(best_validation_accuracy=accuracy, best_validation_loss=loss,
                    best_validation_epoch=1, optimizer_numeric_state_bytes=100,
                    epochs=[dict(training_seconds=seconds, optimizer_step_seconds=seconds / 4,
                                 optimizer_steps=10, validation_accuracy=.99)])
               for accuracy, loss, seconds in [(.8, .3, 2.), (.9, .2, 4.), (1., .1, 6.)]]
    result = summarize_runs(records)
    assert result['validation_accuracy']['mean'] == pytest.approx(.9)
    assert result['validation_accuracy']['sample_std'] == pytest.approx(.1)
    assert result['training_seconds']['mean'] == 4.
    assert result['mean_update_ms']['mean'] == 100.
    assert result['selected_epochs'] == [1, 1, 1]
