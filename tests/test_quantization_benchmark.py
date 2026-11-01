import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_reports_storage_quality_and_memory_scope(tmp_path):
    output = tmp_path / 'comparison.json'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'apps/benchmark_quantization.py'),
         '--gallery-size', '128', '--queries', '8', '--dim', '16',
         '--block-size', '31', '--repeats', '1', '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record['source'] == 'synthetic_sphere'
    assert record['similarity'] == 'inner_product_on_l2_normalized_inputs'
    assert {row['method'] for row in record['results']} == {'fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4'}
    for row in record['results']:
        assert row['stored_bytes'] == sum(row['storage_breakdown'].values())
        assert 0 <= row['recall_at_k'] <= 1
        assert row['query_seconds_median'] > 0
        assert row['max_decoded_block_bytes'] <= 31 * 16 * 4
        assert row['max_score_block_bytes'] <= 8 * 31 * 4
        assert row['process_peak_rss_bytes'] > 0
    exact = next(row for row in record['results'] if row['method'] == 'fp32')
    assert exact['stored_bytes'] == 128 * 16 * 4
    assert exact['recall_at_k'] == 1
    assert exact['relative_reconstruction_mse'] == 0
