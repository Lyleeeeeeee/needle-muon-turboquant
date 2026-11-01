"""Validation-only rate selection and censored, paired batch512 learning curves."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'tools')]


def test_rate_grid_tunes_both_muon_and_auxiliary_with_equal_candidate_counts():
    from tune_batch512 import learning_rate_grid, final_schedule

    grid = learning_rate_grid()
    assert [c['lr'] for c in grid['adamw']] == [.001, .003, .01]
    assert [(c['muon_lr'], c['lr']) for c in grid['muon']] == [(.007, .001), (.021, .003), (.07, .01)]
    points = final_schedule()
    assert points[:3] == [0, 54, 108]
    assert points[20:23] == [1080, 1296, 1512]
    assert points[-1] == 10800 and len(points) == 66


def candidates():
    from tune_batch512 import learning_rate_grid

    records = []
    for method, configs in learning_rate_grid().items():
        for i, config in enumerate(configs):
            values = ([.5, .8, .8], [.5, .7, .85], [.5, .6, .9])[i]
            records.append(dict(optimizer=method, candidate=config, status='complete',
                                validation_curve=[dict(steps=s, precision=p) for s, p in zip([0, 1, 2], values)],
                                test_precision=1. if i == 2 else 0.))
    return records


def test_selection_uses_validation_learning_curve_not_terminal_peak_or_test():
    from tune_batch512 import choose_rates, validation_auc

    selected = choose_rates(candidates())
    assert selected['adamw']['candidate']['id'] == 'adamw_0'
    assert selected['muon']['candidate']['id'] == 'muon_0'
    assert validation_auc([dict(steps=0, precision=.8), dict(steps=100, precision=.8)]) == pytest.approx(.8)
    rows = candidates()
    rows[0].update(status='diverged', validation_curve=[])
    assert choose_rates(rows)['adamw']['candidate']['id'] == 'adamw_1'
    with pytest.raises(ValueError):
        choose_rates(rows[:-1])


def test_training_clock_excludes_snapshots_and_records_numerical_failure_without_padding():
    from tune_batch512 import train_updates

    now, calls = [0.], [0]

    def step():
        now[0] += 2.
        calls[0] += 1
        return dict(samples=4, optimizer_seconds=.5)

    def save(row):
        now[0] += 7.
        return dict(checkpoint_sha256=str(row['steps']))

    good = train_updates(step, save, [0, 2, 4], clock=lambda: now[0])
    assert good['status'] == 'complete' and good['steps'] == 4
    assert good['training_seconds'] == 8. and good['checkpoint_seconds'] == 21.
    assert good['acquisition_wall_seconds'] == 29.

    def fails():
        value = step()
        if calls[0] == 3:
            raise FloatingPointError('Non-finite training loss')
        return value

    now[0], calls[0] = 0., 0
    failed = train_updates(fails, save, [0, 2, 4], clock=lambda: now[0])
    assert failed['status'] == 'diverged' and failed['steps'] == 2
    assert failed['training_seconds'] == 6.
    assert [r['steps'] for r in failed['snapshots']] == [0, 2]


def paired_curves():
    return [dict(optimizer=m, seed=s, status='complete', protocol_sha256='phase',
                 initial_parameter_sha256=str(s), train_indices_sha256=str(s),
                 validation_indices_sha256='val', gallery_ids_sha256='gallery', query_ids_sha256='queries',
                 epoch_order_sha256=['same'], curve=[dict(steps=n, training_seconds=t * scale, precision=p)
                    for n, t, p in [(0, 0., .5), (100, 1., .8), (200, 2., .9)]])
            for s in (8, 9, 10) for m, scale in [('adamw', 1.), ('muon', 1.2)]]


def test_summary_retains_unreached_or_failed_seeds_and_pairing():
    from tune_batch512 import summarize_runs

    rows = paired_curves()
    summary = summarize_runs(rows, targets=[.8, .9])
    assert summary['first_observed'][0]['muon']['training_seconds']['mean'] == 1.2
    rows[-1].update(status='diverged', curve=rows[-1]['curve'][:1])
    failed = summarize_runs(rows, targets=[.8])['first_observed'][0]
    assert failed['muon']['reached_count'] == 2 and failed['muon']['steps'] is None
    with pytest.raises(ValueError):
        summarize_runs(rows[:-1])


def test_native_tuning_then_locked_final_run_and_checked_resume(tmp_path):
    data = tmp_path / 'fashion'
    data.mkdir()
    for prefix, count in [('train', 640), ('t10k', 32)]:
        images = np.random.default_rng(count).integers(0, 256, (count, 28, 28), dtype='uint8')
        labels = np.arange(count, dtype='uint8') % 10
        (data / f'{prefix}-images-idx3-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
        (data / f'{prefix}-labels-idx1-ubyte.gz').write_bytes(
            gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    output = tmp_path / 'study'
    command = [sys.executable, str(ROOT / 'experiments/tune_batch512.py'), '--smoke',
               '--data-dir', str(data), '--output-dir', str(output)]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert done.returncode == 0, done.stdout + done.stderr
    run = json.loads((output / 'run.json').read_text())
    assert len(run['tuning']['candidates']) == 6
    assert len(run['final']['trajectories']) == 2
    assert run['tuning']['protocol']['evaluation_split'] == 'training_holdout'
    assert run['final']['protocol']['evaluation_split'] == 'official_test_exploratory'
    locked = (output / 'selection.json').read_bytes()
    assert hashlib.sha256(locked).hexdigest() == run['final']['protocol']['selection_sha256']
    for r in run['final']['trajectories']:
        assert r['candidate'] == run['selection']['selected'][r['optimizer']]['candidate']
        assert r['steps'] == 3 and len(r['curve']) == 4
    before = (output / 'final/adamw_seed8_train.json').read_bytes()
    resumed = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / 'selection.json').read_bytes() == locked
    assert (output / 'final/adamw_seed8_train.json').read_bytes() == before
    document = json.loads(locked)
    document['selected']['adamw']['candidate']['lr'] = 99.
    (output / 'selection.json').write_text(json.dumps(document))
    corrupt = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert corrupt.returncode != 0


def full_study_fixture():
    from build_report import json_hash
    from common import digest
    from tune_batch512 import choose_rates, final_schedule, learning_rate_grid, summarize_runs
    from time_to_quality import THREADS

    baseline = json.loads((ROOT / 'report/results.json').read_text())['studies']['epochs']
    old = {('adamw' if r['optimizer'] == 'AdamW' else 'muon', r['seed']): r
           for r in baseline['training_records']}
    sha = lambda text: hashlib.sha256(str(text).encode()).hexdigest()
    common = dict(dataset='Fashion-MNIST', study='tuned_batch512', smoke=False, batch_size=512,
                  split_seed=17, train_rows=55000, validation_rows=5000, gallery_rows=55000, query_rows=1000,
                  architecture=[784, 128, 128, 128, 64, 10], weight_decay=.01,
                  candidate_grid=learning_rate_grid(), targets=[.8, .85, .87, .88],
                  gallery_format='fp32', search_metric='cosine', k=10, block_size=256,
                  no_metric_based_stopping=True, offline_quality_workers=3, thread_limits=THREADS,
                  selection_rule='max_normalized_trapezoidal_validation_P_at10_over_updates;tie_terminal_precision_then_id',
                  backend=copy.deepcopy(baseline['protocol']['backend']),
                  data_files_sha256=copy.deepcopy(baseline['protocol']['data_files_sha256']),
                  source_sha256={**baseline['protocol']['source_sha256'],
                                 'experiments/time_to_quality.py': digest(ROOT / 'experiments/time_to_quality.py'),
                                 'experiments/tune_batch512.py': digest(ROOT / 'experiments/tune_batch512.py')})
    tuning_p = dict(common, phase='tuning', training_seeds=[7], evaluation_split='training_holdout',
                    max_steps=1620, snapshot_steps=[0, 108, 216, 324, 540, 864, 1080, 1620])
    ids = baseline['feature_metadata'][0]

    def record(method, seed, config, p):
        ref = old[method, seed] if seed != 7 else old[method, 8]
        initial = ref['initial_parameter_sha256'] if seed != 7 else sha('tuning_initial')
        train_hash = ref['train_indices_sha256'] if seed != 7 else sha('tuning_train')
        seconds = .032 if method == 'adamw' else .04
        snaps = [dict(steps=s, training_seconds=s * seconds, optimizer_seconds=s * .003,
                      samples_seen=(s // 108) * 55000 + (s % 108) * 512,
                      checkpoint=f'{method}_seed{seed}/step{s:06d}.npz', checkpoint_sha256=sha((method, seed, s)),
                      parameter_sha256=initial if s == 0 else sha((config['id'], seed, s)),
                      training_order_prefix_sha256=ref['training_order_sha256'],
                      acquisition_wall_seconds=s * seconds + .01 * (i + 1))
                 for i, s in enumerate(p['snapshot_steps'])]
        training = dict(optimizer=method, seed=seed, candidate=copy.deepcopy(config), status='complete',
                        protocol_sha256=json_hash(p), initial_parameter_sha256=initial,
                        train_indices_sha256=train_hash, validation_indices_sha256=ref['validation_indices_sha256'],
                        total_parameter_count=142410, muon_parameter_count=40960 if method == 'muon' else 0,
                        optimizer_numeric_state_bytes=1139280 if method == 'adamw' else 975440,
                        steps=p['max_steps'], samples_seen=(p['max_steps'] // 108) * 55000,
                        training_seconds=p['max_steps'] * seconds, optimizer_seconds=p['max_steps'] * .003,
                        checkpoint_seconds=.01 * len(snaps),
                        acquisition_wall_seconds=p['max_steps'] * seconds + .01 * len(snaps), snapshots=snaps,
                        epoch_order_sha256=[sha(seed)] * (p['max_steps'] // 108), training_order_sha256=sha((seed, p['max_steps'])))
        curve = []
        for snap in snaps:
            precision = .7 if snap['steps'] == 0 else (
                .80 + .01 * int(config['id'][-1]) if p['phase'] == 'tuning' else .81 if snap['steps'] < 108 else .9)
            matches = round(precision * 10000)
            q, remainder = divmod(matches, 1000)
            histogram = [0] * 11
            histogram[q] = 1000 - remainder
            if remainder:
                histogram[q + 1] = remainder
            curve.append(dict(snap, precision=matches / 10000, same_category_matches=matches,
                              per_query_match_histogram=histogram, neighbor_indices_sha256=sha(matches), evaluation_seconds=.1))
        key = 'validation_curve' if p['phase'] == 'tuning' else 'curve'
        return dict(training, gallery_ids_sha256=ids['gallery_ids_sha256'],
                    query_ids_sha256=sha('validation_queries') if p['phase'] == 'tuning' else ids['query_ids_sha256'],
                    query_source=p['evaluation_split'], training_record_sha256=json_hash(training), **{key: curve})

    grid = learning_rate_grid()
    tuning_records = [record(m, 7, c, tuning_p) for m, configs in grid.items() for c in configs]
    chosen = choose_rates(tuning_records)
    tuning = dict(protocol=tuning_p, candidates=tuning_records, selected=chosen)
    selection = dict(selected=chosen, tuning_run_sha256=json_hash(tuning), locked_at_utc='2026-11-01T12:00:00+00:00')
    final_p = dict(common, phase='final', training_seeds=[8, 9, 10], max_steps=10800,
                   snapshot_steps=final_schedule(), evaluation_split='official_test_exploratory',
                   selected=chosen, selection_sha256=json_hash(selection), tuning_run_sha256=json_hash(tuning))
    records = [record(m, s, chosen[m]['candidate'], final_p) for s in (8, 9, 10) for m in ('adamw', 'muon')]
    final = dict(protocol=final_p, trajectories=records, summary=summarize_runs(records))
    return dict(tuning=tuning, selection=selection, final=final, selection_sha256=json_hash(selection),
                tuning_run_sha256=json_hash(tuning), final_run_sha256=json_hash(final)), baseline


def test_tuned_report_contains_all_targets_and_locked_rates():
    from build_report import collect_tuned512_rows

    run, baseline = full_study_fixture()
    rows = collect_tuned512_rows(run, baseline)
    assert len(rows) == 8
    assert all('Batch 512' in r['study'] for r in rows)
    assert rows[0]['updates_mean'] == 54 and rows[2]['updates_mean'] == 108
    assert rows[0]['training_cost'] == pytest.approx(54 * .032)


@pytest.mark.parametrize('mutation', ['selection', 'rates', 'validation_leak', 'missing', 'source', 'samples', 'summary'])
def test_tuned_report_rejects_unlocked_or_incompatible_runs(mutation):
    from build_report import collect_tuned512_rows

    run, baseline = full_study_fixture()
    if mutation == 'selection': run['selection']['selected']['adamw']['candidate']['lr'] = 99.
    if mutation == 'rates': run['final']['trajectories'][0]['candidate']['lr'] = 99.
    if mutation == 'validation_leak': run['tuning']['candidates'][0]['query_source'] = 'official_test_exploratory'
    if mutation == 'missing': run['final']['trajectories'].pop()
    if mutation == 'source': run['final']['protocol']['source_sha256']['apps/models.py'] = '0' * 64
    if mutation == 'samples': run['final']['trajectories'][0]['snapshots'][1]['samples_seen'] -= 1
    if mutation == 'summary': run['final']['summary']['first_observed'][0]['adamw']['steps']['mean'] += 1
    with pytest.raises(ValueError):
        collect_tuned512_rows(run, baseline)


def test_tuned_report_retains_unreached_high_target_without_success_only_mean():
    from build_report import collect_tuned512_rows, json_hash
    from tune_batch512 import summarize_runs

    run, baseline = full_study_fixture()
    for point in run['final']['trajectories'][-1]['curve']:
        point.update(precision=.7, same_category_matches=7000,
                     per_query_match_histogram=[0] * 7 + [1000] + [0] * 3)
    run['final']['summary'] = summarize_runs(run['final']['trajectories'])
    run['final_run_sha256'] = json_hash(run['final'])
    rows = collect_tuned512_rows(run, baseline)
    assert rows[-1]['training_cost'] is None and '2/3' in rows[-1]['study']
