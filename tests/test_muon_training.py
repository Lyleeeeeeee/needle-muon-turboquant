import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps'))


def test_holdout_split_is_fixed_disjoint_and_legacy_rng_is_preserved():
    from train_baseline import select_training_indices

    a, b, source = select_training_indices(1000, 200, 400, 80, np.random.default_rng(7), 100, 17)
    c, d, _ = select_training_indices(1000, 200, 400, 80, np.random.default_rng(8), 100, 17)
    assert source == 'train_holdout'
    assert not set(a) & set(b)
    np.testing.assert_array_equal(b, d)
    assert not np.array_equal(a, c)
    rng, reference = np.random.default_rng(7), np.random.default_rng(7)
    a, b, source = select_training_indices(1000, 200, 400, 80, rng)
    np.testing.assert_array_equal(a, reference.choice(1000, 400, replace=False))
    np.testing.assert_array_equal(b, reference.choice(200, 80, replace=False))
    np.testing.assert_array_equal(rng.permutation(19), reference.permutation(19))
    assert source == 'official_test_exploratory'


def test_muon_training_cli_uses_holdout_and_saves_best_weights(tmp_path):
    data = ROOT.parent / 'hw0/data'
    if not (data / 'train-images-idx3-ubyte.gz').is_file():
        import pytest
        pytest.skip('Optional local MNIST integration data is unavailable')
    output, best = tmp_path / 'training.json', tmp_path / 'best.npz'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--device', 'cpu',
         '--optimizer', 'muon', '--train-limit', '128', '--validation-size', '500',
         '--eval-limit', '64', '--epochs', '1', '--hidden-dim', '16', '--feature-dim', '8',
         '--best-checkpoint-output', str(best), '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record['optimizer'] == 'Muon+AdamW'
    assert record['validation_source'] == 'train_holdout'
    assert record['train_validation_disjoint'] is True
    assert record['muon_parameter_shapes'] == [[16, 8]]
    assert record['epochs'][0]['optimizer_steps'] == 2
    assert record['epochs'][0]['optimizer_step_seconds'] > 0
    assert record['optimizer_numeric_state_bytes'] > 0
    assert record['best_validation_epoch'] == 1
    assert len(record['initial_parameter_sha256']) == 64
    assert best.is_file()
