"""Existing tuned models must retain their provenance through every gallery format."""
import copy
import gzip
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'tools')]


def parent_fixture():
    from test_tuned_batch512 import full_study_fixture
    return full_study_fixture()[0]


def reseal(parent):
    from build_report import json_hash
    for record in parent['final']['trajectories']:
        raw = {k: v for k, v in record.items() if k not in (
            'curve', 'gallery_ids_sha256', 'query_ids_sha256', 'query_source', 'training_record_sha256')}
        record['training_record_sha256'] = json_hash(raw)
    parent['final_run_sha256'] = json_hash(parent['final'])


def test_existing_model_selection_always_uses_fixed_terminal_not_best_quality():
    from evaluate_tuned import select_terminal_snapshots
    parent = parent_fixture()
    for record in parent['final']['trajectories']:
        record['curve'][1]['precision'] = 1.
    reseal(parent)
    chosen = select_terminal_snapshots(parent)
    assert len(chosen) == 6
    assert {r['steps'] for r in chosen} == {10800}
    assert all(r['expected_fp32_precision'] == .9 for r in chosen)
    assert chosen[0]['checkpoint'] == parent['final']['trajectories'][0]['snapshots'][-1]['checkpoint']
    assert chosen[0]['training_seconds'] == 10800 * .032


@pytest.mark.parametrize('change', ['missing', 'diverged', 'early', 'rate', 'order', 'hash'])
def test_terminal_selection_rejects_incomplete_or_mismatched_source_models(change):
    from evaluate_tuned import select_terminal_snapshots
    parent = parent_fixture()
    records = parent['final']['trajectories']
    if change == 'missing': records.pop()
    if change == 'diverged': records[0]['status'] = 'diverged'
    if change == 'early': records[0]['snapshots'].pop()
    if change == 'rate': records[0]['candidate']['lr'] = .99
    if change == 'order': records[0]['training_order_sha256'] = 'different'
    if change == 'hash': parent['final_run_sha256'] = '0' * 64
    if change != 'hash': reseal(parent)
    with pytest.raises(ValueError):
        select_terminal_snapshots(parent)


def linked_fixture():
    from build_report import json_hash
    from common import aggregate_quality
    from evaluate_tuned import make_protocol, select_terminal_snapshots
    parent = parent_fixture()
    chosen = select_terminal_snapshots(parent)
    protocol = make_protocol(parent, ROOT / 'results/example/run.json', chosen)
    original = json.loads((ROOT / 'report/results.json').read_text())['studies']['epochs']
    metadata = copy.deepcopy(original['feature_metadata'])
    selected = {(r['optimizer'], r['seed']): r for r in chosen}
    stamp = dict(protocol_sha256=json_hash(protocol))
    for feature in metadata:
        source = selected[feature['encoder'], feature['training_seed']]
        feature.update(checkpoint_sha256=source['checkpoint_sha256'], runner_provenance=stamp,
                       export_wall_seconds=1.)
    features = {(f['encoder'], f['training_seed']): f for f in metadata}
    rows = copy.deepcopy(original['retrieval_rows'])
    old_precision = {(r['encoder'], r['training_seed']): r['label_precision_at_k']
                     for r in rows if r['method'] == 'fp32'}
    for row in rows:
        key = row['encoder'], row['training_seed']
        row['label_precision_at_k'] += .9 - old_precision[key]
        row['query_prefix_quality'][-1]['label_precision_at_k'] = row['label_precision_at_k']
        row['runner_provenance'] = dict(stamp, feature_archive_sha256=features[key]['feature_archive_sha256'])
    return dict(protocol=protocol, selected_snapshots=chosen, feature_metadata=metadata,
                retrieval_rows=rows, quality_summary=aggregate_quality(rows), common_image_ids_verified=True), parent


def test_joint_report_contains_all_ten_formats_and_own_model_costs():
    from build_report import collect_integrated_rows
    run, parent = linked_fixture()
    rows = collect_integrated_rows(run, parent)
    assert len(rows) == 10
    assert all('100 epochs' in r['study'] and 'Batch 512' in r['study'] for r in rows)
    assert rows[0]['training_cost'] == pytest.approx(345.6)
    assert rows[5]['training_cost'] == pytest.approx(432.)
    assert rows[0]['precision_mean'] == .9
    assert rows[0]['updates_mean'] == 10800
    assert rows[3]['gallery_bytes'] == 1116400


def test_current_table_emphasizes_joint_results_and_retains_supporting_systems():
    from build_report import collect_fresh_rows
    run, parent = linked_fixture()
    data = json.loads((ROOT / 'report/results.json').read_text())
    data.pop('compact_uniform', None)
    data.pop('cifar_pilot', None)
    data.pop('cifar_time_to_quality', None)
    data.update(integrated=run, tuned_batch512=parent)
    rows = collect_fresh_rows(data)
    assert len(rows) == 28
    assert all('100 epochs' in r['study'] for r in rows[:10])
    assert sum(r['training_unit'] == 'ms/step' for r in rows) == 4
    assert sum(r['training_unit'] == 'µs/image' for r in rows) == 6
    assert sum('First P@10' in r['study'] for r in rows) == 8
    assert not any(r['study'] == '10 epochs / final CPU' for r in rows)


@pytest.mark.parametrize('change', ['model', 'images', 'missing', 'duplicate', 'fp32',
                                    'summary', 'timing', 'source', 'budget'])
def test_joint_report_rejects_cross_model_or_incomplete_comparisons(change):
    from build_report import collect_integrated_rows
    run, parent = linked_fixture()
    if change == 'model': run['feature_metadata'][0]['checkpoint_sha256'] = 'different'
    if change == 'images': run['feature_metadata'][0]['query_ids_sha256'] = 'different'
    if change == 'missing': run['retrieval_rows'].pop()
    if change == 'duplicate': run['retrieval_rows'].append(run['retrieval_rows'][0])
    if change == 'fp32': run['retrieval_rows'][0]['label_precision_at_k'] = .1
    if change == 'summary': run['quality_summary']['encoders']['muon']['fp32']['label_precision_at10']['mean'] = .1
    if change == 'timing': run['retrieval_rows'][0]['query_seconds'][0] = -1.
    if change == 'source': run['protocol']['source_sha256']['apps/models.py'] = '0' * 64
    if change == 'budget': run['selected_snapshots'][0]['training_seconds'] = 1.
    with pytest.raises(ValueError):
        collect_integrated_rows(run, parent)


def test_native_existing_checkpoint_pipeline_all_stores_and_resume(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    for prefix, count in [('train', 640), ('t10k', 32)]:
        images = np.random.default_rng(count).integers(0, 256, (count, 28, 28), dtype='uint8')
        labels = np.arange(count, dtype='uint8') % 10
        (data / f'{prefix}-images-idx3-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
        (data / f'{prefix}-labels-idx1-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    parent = tmp_path / 'tuned'
    train = subprocess.run([sys.executable, str(ROOT / 'experiments/tune_batch512.py'), '--smoke',
                            '--data-dir', str(data), '--output-dir', str(parent)],
                           cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert train.returncode == 0, train.stdout + train.stderr
    source = parent / 'final/adamw_seed8_train.json'
    before = source.read_bytes()
    output = tmp_path / 'quant'
    command = [sys.executable, str(ROOT / 'experiments/evaluate_tuned.py'),
               '--tuned-run', str(parent / 'run.json'), '--data-dir', str(data), '--output-dir', str(output)]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert done.returncode == 0, done.stdout + done.stderr
    run = json.loads((output / 'run.json').read_text())
    assert run['protocol']['smoke'] is True
    assert len(run['retrieval_rows']) == 18
    assert len(run['feature_metadata']) == 2
    assert all(r['steps'] == 3 for r in run['selected_snapshots'])
    assert source.read_bytes() == before
    artifact = output / 'adamw_seed8_rotation7.json'
    unchanged = artifact.read_bytes()
    resumed = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert artifact.read_bytes() == unchanged
    bad = json.loads(artifact.read_text())
    bad['runner_provenance']['feature_archive_sha256'] = 'changed'
    artifact.write_text(json.dumps(bad))
    rejected = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert 'provenance' in (rejected.stdout + rejected.stderr).lower()
