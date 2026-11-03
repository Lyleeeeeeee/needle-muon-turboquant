"""The cleaned project must preserve cohorts, raw evidence and portable entry points."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def tables_module():
    path = ROOT / 'tools/make_tables.py'
    assert path.is_file(), 'The standalone table generator is missing'
    spec = importlib.util.spec_from_file_location('curated_tables', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence():
    path = ROOT / 'report/evidence.json'
    assert path.is_file(), 'Compact evidence for the two shared-cohort tables is missing'
    return json.loads(path.read_text())


def test_two_tables_use_all_six_models_and_keep_the_unreached_goal():
    module = tables_module()
    training, quantization = module.build_tables(evidence())
    assert [r['target'] for r in training] == [.08, .10, .12, .14]
    assert [r['method'] for r in quantization] == ['fp32', 'uniform_2', 'symmetric_2', 'turbo_2',
                                                  'uniform_4', 'symmetric_4', 'turbo_4']
    for row in training[:3]:
        assert row['adamw']['reached_count'] == row['muon']['reached_count'] == 3
    assert training[1]['adamw']['training_seconds']['mean'] == pytest.approx(52.77295140891025)
    assert training[1]['muon']['training_seconds']['mean'] == pytest.approx(25.66606739303097)
    for method in ['adamw', 'muon']:
        assert training[-1][method]['reached_count'] == 0
        assert training[-1][method]['training_seconds'] is None
    assert quantization[0]['adamw']['precision']['mean'] == pytest.approx(.11883333333333333)
    assert quantization[0]['muon']['precision']['mean'] == pytest.approx(.12513333333333334)
    assert quantization[0]['stored_bytes'] == 2560000
    assert quantization[3]['stored_bytes'] == 216400


@pytest.mark.parametrize('mutation', ['missing_model', 'duplicate_rotation', 'missing_control',
                                     'wrong_feature', 'wrong_bytes', 'wrong_terminal', 'mixed_start'])
def test_tables_reject_mixed_or_incomplete_evidence(mutation):
    module = tables_module()
    data = copy.deepcopy(evidence())
    if mutation == 'missing_model': data['training']['trajectories'].pop()
    if mutation == 'duplicate_rotation': data['quantization']['trials'].append(copy.deepcopy(data['quantization']['trials'][0]))
    if mutation == 'missing_control': data['quantization']['trials'].pop()
    if mutation == 'wrong_feature': data['quantization']['trials'][0]['feature_archive_sha256'] = 'bad'
    if mutation == 'wrong_bytes': data['quantization']['trials'][0]['stored_bytes'] += 1
    if mutation == 'wrong_terminal': data['training']['trajectories'][0]['curve'][-1]['precision'] = .99
    if mutation == 'mixed_start': data['training']['trajectories'][0]['initial_parameter_sha256'] = 'bad'
    with pytest.raises(ValueError): module.build_tables(data)


def test_partial_reach_never_reports_a_success_only_mean():
    module = tables_module()
    data = copy.deepcopy(evidence())
    # Pure crossing aggregation, without altering a real quantization cohort.
    records = copy.deepcopy(data['training']['trajectories'])
    chosen = next(r for r in records if (r['optimizer'], r['seed']) == ('muon', 11))
    for point in chosen['curve']: point['precision'] = min(point['precision'], .11)
    row = module.training_rows(records, [.12])[0]
    assert row['muon']['reached_count'] == 2
    assert row['muon']['training_seconds'] is None
    assert len(row['muon']['per_seed']) == 3


def test_readme_has_exactly_the_two_generated_tables_and_valid_local_links():
    import re
    module = tables_module()
    data = evidence(); training, quantization = module.build_tables(data)
    readme = (ROOT / 'README.md').read_text()
    assert sum(line.startswith('|---') for line in readme.splitlines()) == 2
    assert readme.split('<!-- TRAINING:START -->\n')[1].split('\n<!-- TRAINING:END -->')[0] == module.render_training(training)
    assert readme.split('<!-- QUANTIZATION:START -->\n')[1].split('\n<!-- QUANTIZATION:END -->')[0] == module.render_quantization(quantization)
    for link in re.findall(r'\]\(([^)]+)\)', readme):
        if not link.startswith(('https:', 'http:', '#')):
            assert (ROOT / link.split('#')[0]).exists(), link


def test_standalone_cli_requires_no_parent_reports_or_experiment_modules(tmp_path):
    import numpy as np
    path = ROOT / 'experiments/workflow.py'
    assert path.exists(), 'Portable CIFAR workflow is missing'
    labels = np.repeat(np.arange(3, dtype='uint8'), 6)
    pixels = np.random.default_rng(17).integers(0, 256, (18, 3072), dtype='uint8')
    data = tmp_path / 'train.bin'
    data.write_bytes(np.column_stack([labels, labels, pixels]).tobytes())
    output = tmp_path / 'run'
    command = [sys.executable, str(path), '--smoke', '--train-file', str(data), '--output-dir', str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    run = json.loads((output / 'run.json').read_text())
    assert len(run['trajectories']) == 2
    for row in run['trajectories']:
        assert row['status'] == 'complete' and row['training_seconds'] >= .03
        assert (output / f'seed{row["seed"]}' / row['feature_archive']).is_file()
    resumed = subprocess.run(command + ['--resume'], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
