import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps'))


def test_deadline_charges_validation_and_saves_minimum_loss():
    from train_timed import run_timed_updates

    now = [0.]
    losses = iter([.6, .7, .4, .5])
    saved = []

    def step():
        now[0] += 1.
        return {'samples': 4, 'loss_sum': 2., 'optimizer_seconds': .3}

    def evaluate():
        now[0] += .2
        return next(losses), .8

    def save():
        now[0] += .1
        saved.append(now[0])

    result = run_timed_updates(step, evaluate, save, 8., checkpoint_count=4,
                               clock=lambda: now[0])
    assert result['optimizer_steps'] == 8
    assert result['samples_seen'] == 32
    assert result['timed_seconds'] == pytest.approx(8.8)
    assert result['budget_overrun_seconds'] == pytest.approx(.8)
    assert result['budget_overrun_seconds'] <= result['last_step_seconds']
    assert len(result['checkpoints']) == 4
    assert result['best_checkpoint_index'] == 2
    assert len(saved) == 2
    assert result['best_validation_loss'] == .4
    assert result['finalization_seconds'] == pytest.approx(.2)
    assert result['optimizer_step_seconds'] == pytest.approx(2.4)


def test_terminal_candidate_can_win_and_ties_keep_earlier_candidate():
    from train_timed import run_timed_updates

    for losses, expected in [([.4, .3], 1), ([.3, .3], 0)]:
        now, saved = [0.], []
        iterator = iter(losses)

        def step():
            now[0] += .4
            return {'samples': 1, 'loss_sum': 1., 'optimizer_seconds': .1}

        result = run_timed_updates(step, lambda: (next(iterator), .7),
                                   lambda: saved.append(1), 1.2, 2, lambda: now[0])
        assert result['best_checkpoint_index'] == expected
        assert len(result['checkpoints']) == 2
        assert result['timed_seconds'] >= 1.2


@pytest.mark.parametrize('budget', [0, -1, math.inf, math.nan])
def test_invalid_time_budget_rejected_before_training(budget):
    from train_timed import run_timed_updates

    with pytest.raises(ValueError, match='budget'):
        run_timed_updates(None, None, None, budget)


def test_partial_epoch_pairing_checks_order_prefix_not_total_length():
    sys.path.insert(0, str(ROOT / 'experiments'))
    from common import validate_timed_pair

    shared = {'initial_parameter_sha256': 'init', 'train_indices_sha256': 'train',
              'validation_indices_sha256': 'val', 'data_files_sha256': {'x': 'data'},
              'seed': 8, 'batch_size': 64, 'time_budget_seconds': 220.,
              'architecture': [784, 128, 128, 128, 64, 10]}
    a = dict(shared, epoch_order_sha256=['a', 'b', 'c'])
    b = dict(shared, epoch_order_sha256=['a', 'b'])
    validate_timed_pair(a, b)
    b['epoch_order_sha256'] = ['a', 'wrong']
    with pytest.raises(ValueError, match='order'):
        validate_timed_pair(a, b)


def test_timed_native_cli_checkpoint_roundtrip(tmp_path):
    import gzip
    import json
    import struct
    import subprocess
    import numpy as np
    import needle as ndl
    from encoder_checkpoint import load_encoder

    images = np.random.default_rng(3).integers(0, 256, (96, 28, 28), dtype='uint8')
    labels = np.arange(96, dtype='uint8') % 10
    for name, header, body in [
        ('train-images-idx3-ubyte.gz', struct.pack('>IIII', 2051, 96, 28, 28), images.tobytes()),
        ('train-labels-idx1-ubyte.gz', struct.pack('>II', 2049, 96), labels.tobytes())]:
        with gzip.open(tmp_path / name, 'wb') as out:
            out.write(header + body)
    output, best, last = [tmp_path / name for name in ('record.json', 'best.npz', 'last.npz')]
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/train_timed.py'), '--optimizer', 'muon',
         '--budget-seconds', '.3', '--data-dir', str(tmp_path), '--train-limit', '64',
         '--validation-size', '32', '--batch-size', '16', '--hidden-dim', '16',
         '--feature-dim', '8', '--output', str(output), '--best-checkpoint-output', str(best),
         '--checkpoint-output', str(last)], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record['optimizer_steps'] > 0 and record['timed_seconds'] >= .3
    assert record['budget_overrun_seconds'] <= record['last_step_seconds'] + .001
    assert record['best_validation_loss'] == min(r['validation_loss'] for r in record['checkpoints'])
    assert record['architecture'] == [784, 16, 16, 16, 8, 10]
    assert record['muon_parameter_shapes'] == [[16, 16], [16, 16], [16, 8]]
    for path in (best, last):
        restored = load_encoder(path, ndl.cpu())
        x = ndl.Tensor(images[:4].reshape(4, -1).astype('float32') / 255,
                       device=ndl.cpu(), requires_grad=False)
        assert restored.features(x).shape == (4, 8)
        assert np.isfinite(restored(x).numpy()).all()
