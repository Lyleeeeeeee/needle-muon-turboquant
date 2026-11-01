"""The report table must be complete, honest about protocols, and reproducible."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'experiments')]


def evidence():
    return json.loads((ROOT / 'report/results.json').read_text())


def test_table_keeps_every_optimizer_store_and_negative_equal_time_result():
    from build_report import collect_rows

    rows = collect_rows(evidence())
    primary = [r for r in rows if r['study'] == '10 epochs / original CPU']
    assert {(r['optimizer'], r['store']) for r in primary} == {
        (optimizer, store) for optimizer in ('AdamW', 'Muon + aux AdamW')
        for store in ('FP32', 'Uniform 2-bit', 'Uniform 4-bit', 'TurboQuant 2-bit', 'TurboQuant 4-bit')}
    adam = next(r for r in primary if r['optimizer'] == 'AdamW' and r['store'] == 'TurboQuant 2-bit')
    assert adam['precision_mean'] == pytest.approx(.856)
    # Packed 2-bit codes, one float32 norm per row, a 64x64 rotation and four centroids.
    assert adam['gallery_bytes'] == 55000 * 64 * 2 // 8 + 55000 * 4 + 64 * 64 * 4 + 4 * 4
    terminal = [r for r in rows if r['study'] == '220s terminal / original CPU (posthoc)']
    assert len(terminal) == 2
    assert terminal[0]['precision_mean'] > terminal[1]['precision_mean']
    system = [r for r in rows if r['study'].startswith('128 updates /')]
    assert len(system) == 4
    assert all(r['precision_mean'] is None and r['gallery_bytes'] is None for r in system)


@pytest.mark.parametrize('mutation', ['missing_quant_trial', 'duplicate_terminal', 'wrong_training_pair',
                                    'different_system_output', 'wrong_precision_range'])
def test_report_refuses_incomplete_or_incompatible_measurements(mutation):
    from build_report import collect_rows

    data = copy.deepcopy(evidence())
    if mutation == 'missing_quant_trial': data['fixed_epochs']['retrieval_rows'].pop()
    if mutation == 'duplicate_terminal': data['equal_time']['terminal_rows'].append(data['equal_time']['terminal_rows'][0])
    if mutation == 'wrong_training_pair': data['fixed_epochs']['training_records'][0]['initial_parameter_sha256'] = 'wrong'
    if mutation == 'different_system_output': data['system']['runs'][0]['metrics']['training_adamw']['output_sha256'] = 'wrong'
    if mutation == 'wrong_precision_range': data['fixed_epochs']['retrieval_rows'][0]['label_precision_at_k'] = 1.1
    with pytest.raises(ValueError):
        collect_rows(data)


def test_readme_and_csv_are_one_generated_table():
    from build_report import collect_rows, render_markdown, render_csv

    rows = collect_rows(evidence())
    readme = (ROOT / 'README.md').read_text()
    assert readme.count('| Study | Optimizer |') == 1
    assert sum(line.startswith('|---') for line in readme.splitlines()) == 1
    actual = readme.split('<!-- RESULTS:START -->\n')[1].split('\n<!-- RESULTS:END -->')[0]
    assert actual == render_markdown(rows)
    assert (ROOT / 'report/results.csv').read_text() == render_csv(rows)


def test_history_validation_rejects_plausible_but_changed_measurements():
    from build_report import verify_history

    data = evidence()
    data['fixed_epochs']['retrieval_rows'][0]['label_precision_at_k'] += .001
    with pytest.raises(ValueError, match='differ from archive'):
        verify_history(data)


@pytest.mark.parametrize('study,checkpoint', [('epochs', 'best'), ('timed', 'terminal')])
def test_unified_pipeline_trains_both_optimizers_and_all_stores(tmp_path, study, checkpoint):
    import gzip
    import struct
    import subprocess
    import numpy as np

    data = tmp_path / 'data'
    data.mkdir()
    for prefix, count in (('train', 768), ('t10k', 32)):
        images = np.random.default_rng(count).integers(0, 256, (count, 28, 28), dtype='uint8')
        labels = np.arange(count, dtype='uint8') % 10
        (data / f'{prefix}-images-idx3-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
        (data / f'{prefix}-labels-idx1-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    output = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(ROOT / 'experiments/run_report.py'), '--smoke',
                             '--study', study, '--checkpoint', checkpoint,
                             '--data-dir', str(data), '--output-dir', str(output)],
                            cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((output / 'run.json').read_text())
    assert record['protocol']['smoke'] is True
    assert len(record['training_records']) == 2
    assert record['common_image_ids_verified'] is True
    assert {r['method'] for r in record['retrieval_rows']} == {'fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4'}
    assert {r['encoder'] for r in record['retrieval_rows']} == {'adamw', 'muon'}
    assert len(record['retrieval_rows']) == 18
