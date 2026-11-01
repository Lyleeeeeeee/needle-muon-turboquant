import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'apps'))


def test_query_prefix_quality_uses_only_the_selected_queries():
    from benchmark_quantization import retrieval_quality

    found = np.array([[0, 2], [1, 3], [4, 2]])
    truth = np.array([[0, 1], [1, 3], [0, 1]])
    labels = (np.array([0, 1, 0, 1, 2]), np.array([0, 1, 0]))
    prefix = retrieval_quality(found, truth, labels, 2, 2)
    complete = retrieval_quality(found, truth, labels, 2, 3)
    assert prefix == {'query_count': 2, 'recall_at_k': .75, 'label_precision_at_k': 1.}
    assert complete['recall_at_k'] == .5
    assert complete['label_precision_at_k'] == 5 / 6
    assert retrieval_quality(found, truth, None, 2, 3)['label_precision_at_k'] is None


def test_retrieval_indices_preserve_reference_and_nested_samples():
    from export_features import retrieval_indices

    arguments = (101, 79, 17, 13)
    small_gallery, small_queries = retrieval_indices(*arguments, 31, 23, 7)
    large_gallery, large_queries = retrieval_indices(*arguments, 101, 59, 7)
    rng = np.random.default_rng(7)
    original_gallery = rng.choice(101, 17, replace=False)
    original_queries = rng.choice(79, 13, replace=False)
    np.testing.assert_array_equal(small_gallery[:17], original_gallery)
    np.testing.assert_array_equal(small_queries[:13], original_queries)
    np.testing.assert_array_equal(large_gallery[:31], small_gallery)
    np.testing.assert_array_equal(large_queries[:23], small_queries)
    assert len(set(large_gallery)) == 101
    assert len(set(large_queries)) == 59


def test_checkpoint_export_preserves_features_when_retrieval_grows(tmp_path):
    data = ROOT.parent / 'hw0/data'
    if not (data / 'train-images-idx3-ubyte.gz').is_file():
        import pytest
        pytest.skip('Optional local MNIST integration data is unavailable')
    original, expanded = tmp_path / 'original.npz', tmp_path / 'expanded.npz'
    checkpoint = tmp_path / 'encoder.npz'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--data-dir', str(data),
         '--device', 'cpu', '--train-limit', '128', '--eval-limit', '64', '--epochs', '1',
         '--hidden-dim', '16', '--feature-dim', '8', '--feature-output', str(original),
         '--checkpoint-output', str(checkpoint), '--output', str(tmp_path / 'train.json')],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/export_features.py'), '--data-dir', str(data),
         '--checkpoint', str(checkpoint), '--reference-train', '128',
         '--reference-queries', '64', '--gallery-size', '256', '--queries', '96',
         '--output', str(expanded)],
        cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    with np.load(original, allow_pickle=False) as a, np.load(expanded, allow_pickle=False) as b:
        np.testing.assert_array_equal(a['gallery'], b['gallery'][:128])
        np.testing.assert_array_equal(a['queries'], b['queries'][:64])
        np.testing.assert_array_equal(a['gallery_labels'], b['gallery_labels'][:128])
        np.testing.assert_array_equal(a['query_labels'], b['query_labels'][:64])
        assert b['gallery'].shape == (256, 8)
        assert b['queries'].shape == (96, 8)
    metadata = json.loads(expanded.with_suffix('.json').read_text())
    assert metadata['device'] == 'cpu()'
    assert len(metadata['checkpoint_sha256']) == 64
    assert 0 <= metadata['query_classification_accuracy'] <= 1
