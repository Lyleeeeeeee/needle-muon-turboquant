"""Time-to-retrieval-quality must measure crossings, censoring and real train time."""
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
sys.path.insert(0, str(ROOT / 'experiments'))


def test_checkpoint_schedule_covers_early_learning_and_exact_terminal_budget():
    from time_to_quality import checkpoint_schedule

    points = checkpoint_schedule(43000)
    assert points[:4] == [0, 430, 860, 1290]
    assert points[20:23] == [8600, 9460, 10320]
    assert points[-1] == 43000 and len(points) == 61
    assert checkpoint_schedule(7, early_until=4, early_interval=2, late_interval=3) == [0, 2, 4, 7]


def test_training_clock_excludes_snapshot_io_and_includes_every_update():
    from time_to_quality import run_fixed_updates

    now = [0.]
    seen = []

    def step():
        now[0] += 2.
        seen.append(len(seen) + 1)
        return {'samples': 4, 'optimizer_seconds': .5}

    def save(row):
        now[0] += 7.  # Deliberately larger than the training itself.
        return {'checkpoint_sha256': str(row['steps'])}

    result = run_fixed_updates(step, save, [0, 2, 4], clock=lambda: now[0])
    assert seen == [1, 2, 3, 4]
    assert [r['training_seconds'] for r in result['snapshots']] == [0., 4., 8.]
    assert [r['samples_seen'] for r in result['snapshots']] == [0, 8, 16]
    assert result['training_seconds'] == 8.
    assert result['optimizer_seconds'] == 2.
    assert result['checkpoint_seconds'] == 21.
    assert result['acquisition_wall_seconds'] == 29.


def curve():
    return [dict(steps=s, training_seconds=t, precision=p)
            for s, t, p in [(0, 0., .2), (100, 1., .8), (200, 2., .79),
                            (300, 3., .85), (400, 4., .86)]]


def test_first_crossing_is_observed_without_interpolation_and_may_later_revert():
    from time_to_quality import first_crossing

    first = first_crossing(curve(), .8)
    assert first['reached'] is True
    assert (first['steps'], first['training_seconds'], first['precision']) == (100, 1., .8)
    assert (first['previous_steps'], first['previous_training_seconds']) == (0, 0.)
    stable = first_crossing(curve(), .8, consecutive=2)
    assert (stable['steps'], stable['training_seconds']) == (400, 4.)
    assert stable['first_qualifying_steps'] == 300
    assert first_crossing(curve(), .9)['reached'] is False
    assert first_crossing(curve(), .9)['observed_until_steps'] == 400
    assert first_crossing(curve(), .1)['steps'] == 0


@pytest.mark.parametrize('mutation', ['unordered', 'nonfinite', 'time_backwards', 'invalid_precision'])
def test_crossing_rejects_invalid_observations(mutation):
    from time_to_quality import first_crossing

    rows = curve()
    if mutation == 'unordered': rows[2]['steps'] = 100
    if mutation == 'nonfinite': rows[1]['training_seconds'] = float('nan')
    if mutation == 'time_backwards': rows[2]['training_seconds'] = .5
    if mutation == 'invalid_precision': rows[2]['precision'] = 1.1
    with pytest.raises(ValueError):
        first_crossing(rows, .8)


def paired_trajectories():
    records = []
    for seed in (8, 9, 10):
        for method, multiplier in [('adamw', 1.), ('muon', 2.)]:
            rows = copy.deepcopy(curve())
            for r in rows:
                r['training_seconds'] *= multiplier
            records.append(dict(optimizer=method, seed=seed, curve=rows,
                                protocol_sha256='locked', initial_parameter_sha256=str(seed),
                                train_indices_sha256='train', training_order_sha256=str(seed),
                                gallery_ids_sha256='gallery', query_ids_sha256='queries'))
    return records


def test_paired_summary_reports_all_seeds_and_does_not_average_censored_runs():
    from time_to_quality import summarize_trajectories

    records = paired_trajectories()
    result = summarize_trajectories(records, targets=[.8, .9])
    row = result['first_observed'][0]
    assert row['adamw']['steps']['mean'] == 100
    assert row['muon']['training_seconds']['mean'] == 2.
    assert row['paired_muon_over_adamw_time']['mean'] == 2.
    assert row['adamw']['reached_count'] == 3
    assert result['first_observed'][1]['adamw']['steps'] is None
    records[-1]['curve'][-1]['precision'] = .7
    records[-1]['curve'][-2]['precision'] = .7
    records[-1]['curve'][1]['precision'] = .7
    censored = summarize_trajectories(records, targets=[.8])['first_observed'][0]
    assert censored['muon']['reached_count'] == 2
    assert censored['muon']['steps'] is None
    assert censored['paired_muon_over_adamw_time'] is None


@pytest.mark.parametrize('mutation', ['missing_pair', 'duplicate', 'images', 'initialization', 'order', 'protocol'])
def test_trajectory_summary_rejects_unpaired_or_incompatible_runs(mutation):
    from time_to_quality import summarize_trajectories

    records = paired_trajectories()
    if mutation == 'missing_pair': records.pop()
    if mutation == 'duplicate': records.append(copy.deepcopy(records[0]))
    if mutation == 'images': records[-1]['query_ids_sha256'] = 'other'
    if mutation == 'initialization': records[-1]['initial_parameter_sha256'] = 'other'
    if mutation == 'order': records[-1]['training_order_sha256'] = 'other'
    if mutation == 'protocol': records[-1]['protocol_sha256'] = 'other'
    with pytest.raises(ValueError):
        summarize_trajectories(records, targets=[.8])


def test_small_native_trajectory_pipeline_and_checked_resume(tmp_path):
    data = tmp_path / 'fashion'
    data.mkdir()
    for prefix, count in [('train', 96), ('t10k', 24)]:
        images = np.random.default_rng(count).integers(0, 256, (count, 28, 28), dtype='uint8')
        labels = np.arange(count, dtype='uint8') % 10
        (data / f'{prefix}-images-idx3-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
        (data / f'{prefix}-labels-idx1-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    output = tmp_path / 'trajectory'
    command = [sys.executable, str(ROOT / 'experiments/time_to_quality.py'), '--smoke',
               '--data-dir', str(data), '--output-dir', str(output)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    run = json.loads((output / 'run.json').read_text())
    assert run['protocol']['smoke'] is True
    assert len(run['trajectories']) == 2
    assert {r['optimizer'] for r in run['trajectories']} == {'adamw', 'muon'}
    for r in run['trajectories']:
        assert [p['steps'] for p in r['curve']] == [0, 2, 4]
        assert r['curve'][-1]['training_seconds'] > 0
        for point in r['curve']:
            assert point['precision'] == point['same_category_matches'] / (16 * 10)
        assert r['training_seconds'] <= r['acquisition_wall_seconds']
        assert len(r['curve'][-1]['checkpoint_sha256']) == 64
    before = (output / 'adamw_seed8_train.json').read_bytes()
    resumed = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / 'adamw_seed8_train.json').read_bytes() == before
    # A corrupt archived snapshot must not be silently accepted by resume.
    snapshot = next((output / 'adamw_seed8').glob('*.npz'))
    snapshot.write_bytes(b'corrupted')
    corrupt = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert corrupt.returncode != 0
    assert 'checkpoint' in (corrupt.stdout + corrupt.stderr).lower()


def full_report_fixture():
    """A complete synthetic trajectory with real locked baseline provenance."""
    from common import digest
    from time_to_quality import checkpoint_schedule, summarize_trajectories
    sys.path.insert(0, str(ROOT / 'tools'))
    from build_report import json_hash

    evidence = json.loads((ROOT / 'report/results.json').read_text())
    baseline = evidence['studies']['epochs']
    old = {(r['optimizer'], r['seed']): r for r in baseline['training_records']}
    p = copy.deepcopy(baseline['protocol'])
    p.update(study='time_to_retrieval_quality', max_steps=43000,
             targets=[.80, .85, .87, .88], snapshot_steps=checkpoint_schedule(43000),
             gallery_format='fp32', search_metric='cosine', k=10, block_size=256,
             no_metric_based_stopping=True,
             source_sha256={**p['source_sha256'], 'experiments/time_to_quality.py':
                            digest(ROOT / 'experiments/time_to_quality.py')})
    p['reference_checkpoints'] = {
        f'{m}_seed{s}': dict(sha256=old['AdamW' if m == 'adamw' else 'Muon+AdamW', s]['terminal_checkpoint_sha256'])
        for s in (8, 9, 10) for m in ('adamw', 'muon')}
    records = []
    for seed in (8, 9, 10):
        for method in ('adamw', 'muon'):
            before = old['AdamW' if method == 'adamw' else 'Muon+AdamW', seed]
            multiplier = .005 if method == 'adamw' else .010
            snapshots = [dict(steps=step, training_seconds=step * multiplier,
                              optimizer_seconds=step * multiplier / 2,
                              samples_seen=(step // 860) * 55000 + (step % 860) * 64,
                              acquisition_wall_seconds=step * multiplier + .001 * (i + 1),
                              checkpoint=f'{method}_seed{seed}/step{step:06d}.npz',
                              checkpoint_sha256=f'{step:064x}', parameter_sha256=f'{step:064x}',
                              training_order_prefix_sha256=before['training_order_sha256'])
                         for i, step in enumerate(p['snapshot_steps'])]
            snapshots[0]['parameter_sha256'] = before['initial_parameter_sha256']
            next(r for r in snapshots if r['steps'] == 8600)['reference_terminal_arrays_equal'] = True
            training = dict(optimizer=method, seed=seed, protocol_sha256=json_hash(p),
                            **{k: before[k] for k in ('initial_parameter_sha256', 'train_indices_sha256',
                                                       'validation_indices_sha256', 'total_parameter_count',
                                                       'muon_parameter_count')},
                            training_order_sha256=f'{seed:064x}', epoch_order_sha256=[f'{seed:064x}'] * 50,
                            optimizer_numeric_state_bytes=before['optimizer_numeric_state_bytes'],
                            snapshots=snapshots, steps=43000, samples_seen=2750000,
                            training_seconds=43000 * multiplier, optimizer_seconds=21500 * multiplier,
                            checkpoint_seconds=.061, acquisition_wall_seconds=43000 * multiplier + .061)
            curve = []
            for snap in snapshots:
                matches = (7 if snap['steps'] < 1000 else 9) * 1000
                if snap['steps'] == 8600:
                    reference = next(r for r in baseline['checkpoint_check_rows']
                                     if r['encoder'] == method and r['training_seed'] == seed)
                    matches = round(reference['label_precision_at_k'] * 10000)
                quotient, remainder = divmod(matches, 1000)
                histogram = [0] * 11
                histogram[quotient] = 1000 - remainder
                if remainder:
                    histogram[quotient + 1] = remainder
                curve.append(dict(snap, precision=matches / 10000, same_category_matches=matches,
                                  per_query_match_histogram=histogram,
                                  neighbor_indices_sha256='f' * 64, evaluation_seconds=.1))
            ids = baseline['feature_metadata'][0]
            records.append(dict(training, curve=curve, training_record_sha256=json_hash(training),
                                gallery_ids_sha256=ids['gallery_ids_sha256'],
                                query_ids_sha256=ids['query_ids_sha256']))
    return dict(protocol=p, trajectories=records,
                summary=summarize_trajectories(records, p['targets'])), baseline


def test_threshold_report_rows_are_complete_and_use_crossing_costs():
    sys.path.insert(0, str(ROOT / 'tools'))
    from build_report import collect_time_to_quality_rows, render_markdown

    run, baseline = full_report_fixture()
    rows = collect_time_to_quality_rows(run, baseline)
    assert len(rows) == 8
    assert all(r['updates_mean'] == 1290 for r in rows)
    assert rows[0]['training_cost'] == pytest.approx(6.45)
    assert rows[1]['training_cost'] == pytest.approx(12.9)
    assert rows[0]['precision_mean'] == .9
    assert '80%' in render_markdown(rows)
    assert '3/3' in render_markdown(rows)


@pytest.mark.parametrize('mutation', ['source', 'binary', 'images', 'counts', 'summary',
                                    'schedule', 'reference', 'clock', 'initialization', 'targets', 'legacy_precision'])
def test_threshold_report_rejects_invalid_or_incompatible_evidence(mutation):
    sys.path.insert(0, str(ROOT / 'tools'))
    from build_report import collect_time_to_quality_rows

    run, baseline = full_report_fixture()
    if mutation == 'source': run['protocol']['source_sha256']['apps/models.py'] = '0' * 64
    if mutation == 'binary': run['protocol']['backend']['extension_sha256'] = '0' * 64
    if mutation == 'images': run['trajectories'][0]['query_ids_sha256'] = '0' * 64
    if mutation == 'counts': run['trajectories'][0]['curve'][0]['same_category_matches'] += 1
    if mutation == 'summary': run['summary']['first_observed'][0]['adamw']['steps']['mean'] += 1
    if mutation == 'schedule': run['trajectories'][0]['curve'].pop()
    if mutation == 'reference': run['trajectories'][0]['snapshots'][20]['reference_terminal_arrays_equal'] = False
    if mutation == 'clock': run['trajectories'][0]['curve'][1]['training_seconds'] += 1
    if mutation == 'initialization': run['trajectories'][0]['initial_parameter_sha256'] = '0' * 64
    if mutation == 'targets': run['protocol']['targets'] = [.8]
    if mutation == 'legacy_precision':
        point = run['trajectories'][0]['curve'][20]
        point.update(precision=.9, same_category_matches=9000,
                     per_query_match_histogram=[0] * 9 + [1000, 0])
    with pytest.raises(ValueError):
        collect_time_to_quality_rows(run, baseline)


def test_threshold_report_preserves_censored_run_instead_of_using_success_only_mean():
    sys.path.insert(0, str(ROOT / 'tools'))
    from build_report import collect_time_to_quality_rows, render_markdown
    from time_to_quality import summarize_trajectories

    run, baseline = full_report_fixture()
    for point in run['trajectories'][-1]['curve']:
        if point['steps'] == 8600:
            continue
        point.update(precision=.7, same_category_matches=7000,
                     per_query_match_histogram=[0] * 7 + [1000] + [0] * 3)
    run['summary'] = summarize_trajectories(run['trajectories'])
    rows = collect_time_to_quality_rows(run, baseline)
    assert rows[-1]['training_cost'] is None and rows[-1]['updates_mean'] is None
    assert rows[-1]['precision_change_mean'] is None and rows[-1]['recall_mean'] is None
    assert '2/3' in render_markdown(rows)
