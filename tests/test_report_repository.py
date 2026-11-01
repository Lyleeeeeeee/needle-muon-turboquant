"""The report table must be complete, honest about protocols, and reproducible."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'experiments')]


def fresh_evidence():
    current = json.loads((ROOT / 'report/results.json').read_text())
    if current['schema_version'] == 2:
        return current
    base = ROOT / 'results/fresh_study_20261101'
    return dict(schema_version=2,
                studies={name: json.loads((base / name / 'run.json').read_text())
                         for name in ('epochs', 'timed')},
                system=json.loads((base / 'system.json').read_text()))


def test_fresh_table_has_complete_cross_products_and_explicit_resource_costs():
    from build_report import collect_fresh_rows

    rows = collect_fresh_rows(fresh_evidence())
    data = fresh_evidence()
    assert len(rows) == (28 + (8 if 'time_to_quality' in data else 0)
                         + (6 if 'batch_timing' in data else 0) + (8 if 'tuned_batch512' in data else 0))
    for name in ('10 epochs / final CPU', '220s / final CPU'):
        main = [r for r in rows if r['study'] == name]
        assert {(r['optimizer'], r['store']) for r in main} == {
            (m, q) for m in ('AdamW', 'Muon + aux AdamW')
            for q in ('FP32', 'Uniform 2-bit', 'Uniform 4-bit', 'TurboQuant 2-bit', 'TurboQuant 4-bit')}
        for row in main:
            assert row['updates_mean'] > 0
            assert row['optimizer_state_bytes'] == (1139280 if row['optimizer'] == 'AdamW' else 975440)
            exact = next(r for r in main if r['optimizer'] == row['optimizer'] and r['store'] == 'FP32')
            assert row['precision_change_mean'] == pytest.approx(row['precision_mean'] - exact['precision_mean'])
    checks = [r for r in rows if 'terminal check' in r['study']]
    assert len(checks) == 4 and all(r['store'] == 'FP32' for r in checks)


@pytest.mark.parametrize('mutation', ['mixed_binary', 'missing_store', 'different_images',
                                    'missing_terminal', 'wrong_budget', 'wrong_state', 'zero_time'])
def test_fresh_table_rejects_incompatible_or_missing_trials(mutation):
    from build_report import collect_fresh_rows

    data = copy.deepcopy(fresh_evidence())
    epochs, timed = data['studies']['epochs'], data['studies']['timed']
    if mutation == 'mixed_binary': timed['protocol']['backend']['extension_sha256'] = '0' * 64
    if mutation == 'missing_store': epochs['retrieval_rows'].pop()
    if mutation == 'different_images': timed['feature_metadata'][0]['gallery_ids_sha256'] = 'different'
    if mutation == 'missing_terminal': timed['checkpoint_check_rows'].pop()
    if mutation == 'wrong_budget': timed['training_records'][0]['time_budget_seconds'] = 221.
    if mutation == 'wrong_state': epochs['training_records'][0]['optimizer_numeric_state_bytes'] = 1
    if mutation == 'zero_time': epochs['training_records'][0]['epochs'][0]['training_seconds'] = 0.
    with pytest.raises(ValueError):
        collect_fresh_rows(data)


def test_readme_and_csv_are_one_generated_table():
    from build_report import collect_rows, render_markdown, render_csv

    rows = collect_rows(json.loads((ROOT / 'report/results.json').read_text()))
    readme = (ROOT / 'README.md').read_text()
    assert readme.count('| Study | Optimizer |') == 1
    assert sum(line.startswith('|---') for line in readme.splitlines()) == 1
    actual = readme.split('<!-- RESULTS:START -->\n')[1].split('\n<!-- RESULTS:END -->')[0]
    assert actual == render_markdown(rows)
    assert (ROOT / 'report/results.csv').read_text() == render_csv(rows)


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
                             '--checkpoint-check',
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
    assert len(record['checkpoint_check_rows']) == 2
    assert {r['method'] for r in record['checkpoint_check_rows']} == {'fp32'}
    assert len(record['protocol']['backend']['extension_sha256']) == 64
    training_path = output / 'adamw_seed8_train.json'
    before = training_path.read_bytes()
    command = [sys.executable, str(ROOT / 'experiments/run_report.py'), '--smoke',
               '--study', study, '--checkpoint', checkpoint, '--checkpoint-check', '--resume',
               '--data-dir', str(data), '--output-dir', str(output)]
    resumed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert training_path.read_bytes() == before
    frozen = output / 'protocol.json'
    changed = json.loads(frozen.read_text())
    changed['rates']['muon'] = .99
    frozen.write_text(json.dumps(changed))
    rejected = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert 'protocol' in (rejected.stdout + rejected.stderr).lower()


def test_fresh_history_verification_rejects_changed_rate_selection():
    from build_report import verify_history

    data = copy.deepcopy(fresh_evidence())
    data['locked_rate_selection']['selected']['adamw']['candidate'] = 'changed'
    with pytest.raises(ValueError, match='selection differs'):
        verify_history(data)
