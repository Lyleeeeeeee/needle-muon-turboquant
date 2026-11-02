"""Stronger uniform baselines retain real metadata savings and native decoding."""
from pathlib import Path
import sys
import json
import subprocess
import numpy as np
import pytest
import needle as ndl
from needle.quantization import UniformQuantizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


@pytest.mark.parametrize('bits', [2, 4])
@pytest.mark.parametrize('device', [ndl.cpu(), ndl.cpu_numpy()])
def test_shared_zero_minimum_matches_affine_decode_and_search_exactly(bits, device, tmp_path):
    from uniform_compact import CompactUniformQuantizer, CompactUniformStore
    rng = np.random.default_rng(12)
    x = np.maximum(rng.normal(size=(64, 16)).astype('float32'), 0)
    x[:, 0] = 0
    x[0] = 0
    ordinary = UniformQuantizer(16, bits, device=device).encode(x)
    compact = CompactUniformQuantizer(16, bits, device=device).encode(x)
    assert compact.nbytes == 64 * 16 * bits // 8 + 64 * 4 + 5
    np.testing.assert_array_equal(ordinary.decode_block(3, 19).numpy(), compact.decode_block(3, 19).numpy())
    queries = rng.normal(size=(4, 16)).astype('float32')
    np.testing.assert_array_equal(ordinary.search(queries, k=5, block_size=16, metric='cosine').indices,
                                  compact.search(queries, k=5, block_size=16, metric='cosine').indices)
    path = tmp_path / 'packed.npz'
    compact.save(path)
    restored = CompactUniformStore.load(path, device=device)
    assert restored.nbytes == compact.nbytes
    np.testing.assert_array_equal(restored.decode_block(0, 64).numpy(), compact.decode_block(0, 64).numpy())


def test_nonconstant_minimum_keeps_ordinary_affine_information():
    from uniform_compact import CompactUniformQuantizer
    x = np.arange(32, dtype='float32').reshape(4, 8)
    ordinary = UniformQuantizer(8, 2, device=ndl.cpu()).encode(x)
    compact = CompactUniformQuantizer(8, 2, device=ndl.cpu()).encode(x)
    assert compact.nbytes == ordinary.nbytes + 1
    np.testing.assert_array_equal(ordinary.decode_block(0, 4).numpy(), compact.decode_block(0, 4).numpy())


@pytest.mark.parametrize('bits', [2, 4])
def test_symmetric_uniform_uses_one_row_scale_and_roundtrips(bits, tmp_path):
    from uniform_compact import CompactUniformQuantizer, CompactUniformStore
    x = np.random.default_rng(3).normal(size=(32, 64)).astype('float32')
    x[0] = 0
    store = CompactUniformQuantizer(64, bits, device=ndl.cpu(), symmetric=True).encode(x)
    assert store.nbytes == 32 * 64 * bits // 8 + 32 * 4 + 1
    assert np.isfinite(store.decode_block(0, 32).numpy()).all()
    np.testing.assert_array_equal(store.decode_block(0, 1).numpy(), np.zeros((1, 64), dtype='float32'))
    path = tmp_path / 'symmetric.npz'
    store.save(path)
    loaded = CompactUniformStore.load(path, device=ndl.cpu())
    np.testing.assert_array_equal(loaded.decode_block(3, 17).numpy(), store.decode_block(3, 17).numpy())


def test_isolated_gallery_comparison_keeps_all_formats_rotations_and_real_archives(tmp_path):
    rng = np.random.default_rng(4)
    gallery = np.maximum(rng.normal(size=(32, 16)).astype('float32'), 0)
    gallery[:, 0] = 0
    features = tmp_path / 'features.npz'
    np.savez(features, gallery=gallery, queries=rng.normal(size=(4, 16)).astype('float32'),
             gallery_labels=np.arange(32) % 3, query_labels=np.arange(4) % 3)
    output = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(ROOT / 'experiments/compare_galleries.py'),
                             '--features', str(features), '--output', str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert len(report['results']) == 13
    for row in report['results']:
        assert len(row['query_seconds']) == 3 and min(row['query_seconds']) > 0
        assert row['stored_bytes'] == sum(row['storage_breakdown'].values())
        assert Path(row['serialized_path']).exists() and len(row['serialized_sha256']) == 64
        assert Path(row['neighbor_archive']).exists() and len(row['neighbor_archive_sha256']) == 64
    rows = {r['method']: r for r in report['results'] if r['rotation_seed'] == 7}
    for bits in [2, 4]:
        assert rows[f'uniform_{bits}']['label_precision_at_k'] == rows[f'compact_{bits}']['label_precision_at_k']
        assert rows[f'uniform_{bits}']['recall_at_k'] == rows[f'compact_{bits}']['recall_at_k']
        assert rows[f'uniform_{bits}']['neighbor_indices_sha256'] == rows[f'compact_{bits}']['neighbor_indices_sha256']
        assert rows[f'compact_{bits}']['stored_bytes'] < rows[f'uniform_{bits}']['stored_bytes']


def test_gallery_rotations_are_averaged_within_each_training_seed():
    from compare_galleries import summarize_galleries
    rows = []
    for encoder in ['adamw', 'muon']:
        for method, trials in [('fp32', [(7, .5)]), ('turbo_2', [(7, .4), (8, .5), (9, .6)])]:
            for rotation, precision in trials:
                rows.append(dict(encoder=encoder, training_seed=8, method=method, rotation_seed=rotation,
                                 label_precision_at_k=precision, recall_at_k=.5,
                                 query_seconds_median=.1, stored_bytes=100 if method == 'fp32' else 10))
    summary = summarize_galleries(rows, [8])
    assert summary['encoders']['adamw']['turbo_2']['label_precision_at10']['values'] == [.5]
    assert summary['encoders']['adamw']['turbo_2']['label_precision_at10']['sample_std'] == 0
    with pytest.raises(ValueError): summarize_galleries(rows[:-1], [8])
    with pytest.raises(ValueError): summarize_galleries(rows+[rows[0]], [8])
