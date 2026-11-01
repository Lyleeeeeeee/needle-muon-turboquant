"""Validate the fresh study and build one consolidated README/CSV table."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from common import (SEEDS, aggregate_quality, distribution, select_configuration,
                    validate_pair, validate_timed_pair)
from benchmark_system import validate_measurements
from benchmark_batch import summarize_batches
from time_to_quality import TARGETS, THREADS, checkpoint_schedule, first_crossing, summarize_trajectories
from tune_batch512 import choose_rates, final_schedule, learning_rate_grid, summarize_runs
from evaluate_tuned import make_protocol, select_terminal_snapshots
from cifar_pilot import choose_candidate, final_epochs, model_config, rate_grid
from compare_galleries import METHODS as GALLERY_METHODS, summarize_galleries
from cifar_time_to_quality import TARGETS as CIFAR_TARGETS, SEEDS as CIFAR_SEEDS, observation_due, summarize_targets

STORES = {'fp32': 'FP32', 'uniform_2': 'Uniform 2-bit', 'uniform_4': 'Uniform 4-bit',
          'turbo_2': 'TurboQuant 2-bit', 'turbo_4': 'TurboQuant 4-bit'}
OPTIMIZERS = {'adamw': 'AdamW', 'muon': 'Muon + aux AdamW'}
METHOD_KEYS = {'adamw': 'adamw', 'muon': 'muon', 'AdamW': 'adamw', 'Muon+AdamW': 'muon'}
HEADERS = ['Study', 'Optimizer', 'Gallery format', 'P@10 (%)', 'Δ vs FP32 (pp)',
           'Own-FP32 R@10 (%)', 'Gallery MB', 'Compression', 'Training cost',
           'Updates', 'Optimizer state MB', 'Query s / 1k']


def validate_retrieval(rows):
    for row in rows:
        if row['device'] != 'cpu()' or row['search_metric'] != 'cosine' or not row['normalization_timed']:
            raise ValueError('Expected native cosine retrieval with normalization timed')
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in ('label_precision_at_k', 'recall_at_k')):
            raise ValueError('Invalid retrieval quality')
        times = row['query_seconds']
        if len(times) != 3 or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('Expected three positive query timings')
        if stats.median(times) != row['query_seconds_median']:
            raise ValueError('Query median does not match raw timings')
        if row['stored_bytes'] != sum(row['storage_breakdown'].values()):
            raise ValueError('Storage components do not match')


def training_by_seed(records, timed=False):
    indexed = {(METHOD_KEYS[r['optimizer']], r['seed']): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(m, s) for m in OPTIMIZERS for s in SEEDS}:
        raise ValueError('Expected all six unique paired training records')
    for seed in SEEDS:
        (validate_timed_pair if timed else validate_pair)(indexed['adamw', seed], indexed['muon', seed])
    return indexed


def quality_rows(study, quality, costs, stores=None, fp32_bytes=14080000):
    result = []
    for optimizer in OPTIMIZERS:
        for method in (STORES if stores is None else stores):
            if method not in quality['encoders'][optimizer]:
                continue
            metric = quality['encoders'][optimizer][method]
            if len(set(metric['stored_bytes']['values'])) != 1:
                raise ValueError('Persistent format bytes changed across training seeds')
            byte_count = metric['stored_bytes']['mean']
            result.append(dict(study=study, optimizer=OPTIMIZERS[optimizer], store=(STORES if stores is None else stores)[method],
                               precision_mean=metric['label_precision_at10']['mean'],
                               precision_sd=metric['label_precision_at10']['sample_std'],
                               recall_mean=metric['neighbor_recall_at10']['mean'],
                               recall_sd=metric['neighbor_recall_at10']['sample_std'],
                               gallery_bytes=byte_count, compression=fp32_bytes / byte_count,
                               training_cost=costs[optimizer]['mean'],
                               training_sd=costs[optimizer]['sample_std'],
                               training_unit='s', speedup=None,
                               query_mean=metric['query_batch_seconds']['mean'],
                               query_sd=metric['query_batch_seconds']['sample_std']))
    return result


def json_hash(protocol):
    return hashlib.sha256((json.dumps(protocol, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def threshold_rows(summary, prefix=''):
    rows = []
    for result in summary['first_observed']:
        for method, label in OPTIMIZERS.items():
            hit = result[method]
            cost, steps, precision = (hit[key] for key in ('training_seconds', 'steps', 'precision'))
            rows.append(dict(
                study=f"{prefix}First P@10 ≥ {100*result['target']:.0f}% ({hit['reached_count']}/3 reached)",
                optimizer=label, store='FP32',
                precision_mean=precision['mean'] if precision else None,
                precision_sd=precision['sample_std'] if precision else None,
                recall_mean=1. if precision else None, recall_sd=0. if precision else None,
                gallery_bytes=14080000, compression=1.,
                training_cost=cost['mean'] if cost else None,
                training_sd=cost['sample_std'] if cost else None, training_unit='s', speedup=None,
                query_mean=None, query_sd=None, precision_change_mean=0. if precision else None,
                precision_change_sd=0. if precision else None,
                updates_mean=steps['mean'] if steps else None,
                updates_sd=steps['sample_std'] if steps else None,
                optimizer_state_bytes=1139280 if method == 'adamw' else 975440))
    return rows


def validate_fresh_study(run, name):
    """Validate complete paired runs without requiring ignored model artifacts."""
    p = run['protocol']
    expected = dict(dataset='Fashion-MNIST', study=name, smoke=False, checkpoint='best',
                    checkpoint_check=True, epochs=10, budget_seconds=220., train_rows=55000,
                    validation_rows=5000, gallery_rows=55000, query_rows=1000, split_seed=17,
                    architecture=[784, 128, 128, 128, 64, 10], batch_size=64,
                    rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                    rotation_seeds=[7, 8, 9], training_seeds=list(SEEDS), methods=list(STORES))
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Fresh study protocol differs from the locked comparison')
    if {k: p['backend'].get(k) for k in ('name', 'dtype', 'cpu_matmul', 'tiled_ns')} != dict(
            name='native_cpu', dtype='float32', cpu_matmul='strided', tiled_ns=True):
        raise ValueError('Fresh runs require the same final native CPU backend')
    indexed = training_by_seed(run['training_records'], timed=name == 'timed')
    stamp = {'protocol_sha256': json_hash(p)}
    for (method, seed), r in indexed.items():
        if (r.get('runner_provenance') != stamp or r['train_samples'] != 55000
                or r['eval_samples'] != 5000 or r['extra_hidden_layers'] != 2
                or r['device'] != 'cpu()' or r['backend'] != 'nd'
                or r['total_parameter_count'] != 142410 or r['learning_rate'] != .001
                or r['muon_parameter_count'] != (40960 if method == 'muon' else 0)
                or r['muon_learning_rate'] != (.007 if method == 'muon' else None)
                or r['weight_decay'] != .01
                or r['optimizer_numeric_state_bytes'] != (1139280 if method == 'adamw' else 975440)):
            raise ValueError('Fresh training provenance, shape or numeric state differs')
        if name == 'epochs':
            epochs = r['epochs']
            if (len(epochs) != 10 or any(e['optimizer_steps'] != 860 for e in epochs)
                    or any(not math.isfinite(e['training_seconds']) or e['training_seconds'] <= 0 for e in epochs)
                    or r['best_validation_epoch'] != 1 + min(range(10), key=lambda i: epochs[i]['validation_loss'])):
                raise ValueError('Incomplete epoch budget or incorrect checkpoint rule')
        elif (r['time_budget_seconds'] != 220. or not 220 <= r['timed_seconds'] <= 220 + r['last_step_seconds'] + .001
              or r['optimizer_steps'] <= 0
              or r['best_checkpoint_index'] != min(range(len(r['checkpoints'])),
                                                  key=lambda i: r['checkpoints'][i]['validation_loss'])):
            raise ValueError('Invalid wall-clock budget or checkpoint selection')
    expected_models = {(m, s) for m in OPTIMIZERS for s in SEEDS}
    common_ids = None
    for key, checkpoint_field in (('feature_metadata', 'best_checkpoint_sha256'),
                                  ('checkpoint_check_metadata', 'terminal_checkpoint_sha256')):
        metadata = run[key]
        if len(metadata) != 6 or {(r['encoder'], r['training_seed']) for r in metadata} != expected_models:
            raise ValueError('Incomplete own-model feature provenance')
        for r in metadata:
            if (r['checkpoint_sha256'] != indexed[r['encoder'], r['training_seed']][checkpoint_field]
                    or r.get('runner_provenance') != stamp or r['gallery_shape'] != [55000, 64]
                    or r['query_shape'] != [1000, 64] or r['validation_size_excluded'] != 5000
                    or r['split_seed'] != 17 or r['data_files_sha256'] != p['data_files_sha256']):
                raise ValueError('Own-model checkpoint, data or feature dimensions differ')
            ids = r['gallery_ids_sha256'], r['query_ids_sha256']
            if common_ids is None:
                common_ids = ids
            if ids != common_ids:
                raise ValueError('Fresh models used different retrieval images')
    for key, sensitivity in (('retrieval_rows', False), ('checkpoint_check_rows', True)):
        rows = run[key]
        expected_keys = {(m, s, q, rotation) for m in OPTIMIZERS for s in SEEDS
                         for q in (['fp32'] if sensitivity else STORES)
                         for rotation in ((7, 8, 9) if q.startswith('turbo') else (7,))}
        keys = {(r['encoder'], r['training_seed'], r['method'], r['rotation_seed']) for r in rows}
        if len(rows) != len(expected_keys) or keys != expected_keys:
            raise ValueError('Missing or duplicate optimizer/store/rotation trial')
        validate_retrieval(rows)
    quality, check = aggregate_quality(run['retrieval_rows']), aggregate_quality(run['checkpoint_check_rows'])
    if quality != run['quality_summary'] or check != run['checkpoint_check_summary']:
        raise ValueError('Fresh aggregates differ from raw trials')
    return quality, check, indexed, common_ids


def collect_time_to_quality_rows(run, baseline):
    """Validate sampled crossings and keep censored runs visible in the same table."""
    p, before = run['protocol'], baseline['protocol']
    expected = dict(dataset='Fashion-MNIST', study='time_to_retrieval_quality', smoke=False,
                    training_seeds=list(SEEDS), max_steps=43000, targets=list(TARGETS),
                    snapshot_steps=checkpoint_schedule(43000), train_rows=55000,
                    validation_rows=5000, gallery_rows=55000, query_rows=1000,
                    architecture=[784, 128, 128, 128, 64, 10], batch_size=64, split_seed=17,
                    rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                    gallery_format='fp32', search_metric='cosine', k=10, block_size=256,
                    no_metric_based_stopping=True)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Time-to-quality protocol differs from the locked comparison')
    if p['backend'] != before['backend'] or p['data_files_sha256'] != before['data_files_sha256']:
        raise ValueError('Time-to-quality backend or data differs from the original study')
    for name, sha in before['source_sha256'].items():
        if name != 'experiments/run_report.py' and p['source_sha256'].get(name) != sha:
            raise ValueError('Time-to-quality shared source differs: ' + name)
    original = training_by_seed(baseline['training_records'])
    originals = {(r['encoder'], r['training_seed']): r for r in baseline['checkpoint_check_rows']}
    image_ids = tuple(baseline['feature_metadata'][0][key] for key in ('gallery_ids_sha256', 'query_ids_sha256'))
    records = run['trajectories']
    if (len(records) != 6 or {(r['optimizer'], r['seed']) for r in records}
            != {(m, s) for m in OPTIMIZERS for s in SEEDS}):
        raise ValueError('Expected six complete unique time-to-quality trajectories')
    for r in records:
        key = r['optimizer'], r['seed']
        old = original[key]
        if (r['protocol_sha256'] != json_hash(p)
                or tuple(r[k] for k in ('gallery_ids_sha256', 'query_ids_sha256')) != image_ids
                or any(r[k] != old[k] for k in ('initial_parameter_sha256', 'train_indices_sha256',
                                               'validation_indices_sha256', 'total_parameter_count',
                                               'muon_parameter_count', 'optimizer_numeric_state_bytes'))):
            raise ValueError('Time-to-quality model, image selection or provenance differs')
        training = {k: v for k, v in r.items() if k not in
                    ('curve', 'gallery_ids_sha256', 'query_ids_sha256', 'training_record_sha256')}
        if json_hash(training) != r['training_record_sha256']:
            raise ValueError('Time-to-quality training record differs from its sealed original')
        if (r['steps'] != 43000 or r['samples_seen'] != 2750000
                or len(r['epoch_order_sha256']) != 50
                or r['snapshots'][0]['parameter_sha256'] != r['initial_parameter_sha256']
                or r['training_seconds'] != r['curve'][-1]['training_seconds']
                or not 0 < r['optimizer_seconds'] <= r['training_seconds'] <= r['acquisition_wall_seconds']
                or not 0 <= r['checkpoint_seconds'] <= r['acquisition_wall_seconds']):
            raise ValueError('Incomplete time-to-quality training budget or invalid clock')
        for rows in (r['snapshots'], r['curve']):
            if [s['steps'] for s in rows] != p['snapshot_steps']:
                raise ValueError('Missing time-to-quality observation')
        for snap, point in zip(r['snapshots'], r['curve']):
            if any(point[k] != value for k, value in snap.items()):
                raise ValueError('Retrieval observation refers to a different training snapshot')
            histogram, matches = point['per_query_match_histogram'], point['same_category_matches']
            if (len(histogram) != 11 or any(type(n) is not int or n < 0 for n in histogram)
                    or sum(histogram) != 1000 or type(matches) is not int
                    or matches != sum(i * n for i, n in enumerate(histogram))
                    or point['precision'] != matches / 10000
                    or not math.isfinite(point['evaluation_seconds']) or point['evaluation_seconds'] <= 0
                    or snap['samples_seen'] != (snap['steps'] // 860) * 55000 + (snap['steps'] % 860) * 64):
                raise ValueError('Retrieval precision or sample counts do not match raw observations')
            for field in ('checkpoint_sha256', 'parameter_sha256', 'neighbor_indices_sha256'):
                if len(point[field]) != 64 or set(point[field]) - set('0123456789abcdef'):
                    raise ValueError('Invalid archived snapshot or search hash')
        reference = next(s for s in r['curve'] if s['steps'] == 8600)
        if (reference.get('reference_terminal_arrays_equal') is not True
                or reference['training_order_prefix_sha256'] != old['training_order_sha256']
                or reference['precision'] != originals[key]['label_precision_at_k']
                or p['reference_checkpoints'][f'{key[0]}_seed{key[1]}']['sha256'] != old['terminal_checkpoint_sha256']):
            raise ValueError('8600-update parameter, order or retrieval reference check failed')
    summary = summarize_trajectories(records, p['targets'], p['training_seeds'])
    if summary != run['summary']:
        raise ValueError('Time-to-quality crossing summaries differ from raw trajectories')
    return threshold_rows(summary)


def collect_batch_rows(run, baseline):
    """Keep fixed-image throughput separate from convergence and query timing."""
    p, before = run['protocol'], baseline['protocol']
    expected = dict(dataset='Fashion-MNIST', study='batch_training_cost', smoke=False,
                    blocks=5, batch_sizes=[64, 256, 512], seed=8, split_seed=17,
                    train_rows=55000, validation_rows=5000, measured_samples=32768, warmup_samples=2048,
                    architecture=[784, 128, 128, 128, 64, 10],
                    rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                    thread_limits=THREADS)
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Batch timing protocol differs from the fixed-image diagnostic')
    train_files = {name: sha for name, sha in before['data_files_sha256'].items() if name.startswith('train-')}
    if p['backend'] != before['backend'] or p['data_files_sha256'] != train_files:
        raise ValueError('Batch timing used a different backend or dataset')
    for path, sha in before['source_sha256'].items():
        if path != 'experiments/run_report.py' and p['source_sha256'].get(path) != sha:
            raise ValueError('Batch timing shared source differs: ' + path)
    original = training_by_seed(baseline['training_records'])
    for r in run['runs']:
        old = original[r['optimizer'], 8]
        if (r['protocol_sha256'] != json_hash(p)
                or r['native_extension_sha256'] != p['backend']['extension_sha256']
                or r['measured_samples'] != 32768 or r['warmup_samples'] != 2048
                or any(r[key] != old[key] for key in ('initial_parameter_sha256', 'train_indices_sha256',
                                                      'total_parameter_count', 'muon_parameter_count',
                                                      'optimizer_numeric_state_bytes'))):
            raise ValueError('Batch worker start, work, state or provenance differs')
    summary = summarize_batches(run['runs'], p['blocks'], p['batch_sizes'])
    if summary != run['summary']:
        raise ValueError('Batch timing summaries differ from raw measurements')
    rows = []
    for batch in p['batch_sizes']:
        for method, label in OPTIMIZERS.items():
            metric = summary[str(batch)][method]
            cost = metric['training_us_per_image']
            rows.append(dict(study=f'Batch {batch} / 32,768 images', optimizer=label, store='—',
                             precision_mean=None, precision_sd=None, recall_mean=None, recall_sd=None,
                             gallery_bytes=None, compression=None, training_cost=cost['mean'],
                             training_sd=cost['sample_std'], training_unit='µs/image', speedup=None,
                             query_mean=None, query_sd=None, precision_change_mean=None, precision_change_sd=None,
                             updates_mean=metric['updates'], updates_sd=0,
                             optimizer_state_bytes=1139280 if method == 'adamw' else 975440))
    return rows


def validate_tuned_curve(record, protocol, key):
    """Validate complete or honestly truncated trajectories without ignored files."""
    r, p = record, protocol
    planned = p['snapshot_steps']
    snapshots, curve = r['snapshots'], r[key]
    steps = [s['steps'] for s in snapshots]
    expected_state = 1139280 if r['optimizer'] == 'adamw' else 975440
    training = {name: value for name, value in r.items() if name not in
                ('curve', 'validation_curve', 'gallery_ids_sha256', 'query_ids_sha256',
                 'query_source', 'training_record_sha256')}
    if (r['protocol_sha256'] != json_hash(p) or r['training_record_sha256'] != json_hash(training)
            or r['query_source'] != p['evaluation_split'] or not snapshots
            or r['status'] not in ('complete', 'diverged') or steps != planned[:len(steps)]
            or len(curve) != len(snapshots) or r['steps'] > p['max_steps']
            or r['samples_seen'] != (r['steps'] // 108) * 55000 + (r['steps'] % 108) * 512
            or r['total_parameter_count'] != 142410
            or r['muon_parameter_count'] != (40960 if r['optimizer'] == 'muon' else 0)
            or r['optimizer_numeric_state_bytes'] not in ([expected_state] if r['steps'] else [0, expected_state])
            or snapshots[0]['parameter_sha256'] != r['initial_parameter_sha256']):
        raise ValueError('Tuned trajectory budget, model, query source or provenance differs')
    if r['status'] == 'complete' and (steps != planned or r['steps'] != p['max_steps']
                                     or len(r['epoch_order_sha256']) != p['max_steps'] // 108):
        raise ValueError('A complete tuned run did not finish the fixed training budget')
    if (not math.isfinite(r['training_seconds']) or not math.isfinite(r['optimizer_seconds'])
            or not 0 <= r['optimizer_seconds'] <= r['training_seconds'] <= r['acquisition_wall_seconds']
            or not 0 <= r['checkpoint_seconds'] <= r['acquisition_wall_seconds']
            or r['training_seconds'] < curve[-1]['training_seconds']):
        raise ValueError('Invalid tuned training clock')
    first_crossing(curve, 0.)  # Validate every observation, including ordered clocks.
    for snap, point in zip(snapshots, curve):
        histogram, matches = point['per_query_match_histogram'], point['same_category_matches']
        if (any(point[name] != value for name, value in snap.items())
                or snap['samples_seen'] != (snap['steps'] // 108) * 55000 + (snap['steps'] % 108) * 512
                or len(histogram) != 11 or any(type(n) is not int or n < 0 for n in histogram)
                or sum(histogram) != 1000 or type(matches) is not int
                or matches != sum(i * n for i, n in enumerate(histogram))
                or point['precision'] != matches / 10000
                or not math.isfinite(point['evaluation_seconds']) or point['evaluation_seconds'] <= 0):
            raise ValueError('Tuned retrieval metrics differ from the saved training snapshot or match counts')
        for field in ('checkpoint_sha256', 'parameter_sha256', 'neighbor_indices_sha256'):
            if len(point[field]) != 64 or set(point[field]) - set('0123456789abcdef'):
                raise ValueError('Invalid tuned checkpoint/search hash')


def collect_tuned512_rows(run, baseline):
    tuning, final, selection = run['tuning'], run['final'], run['selection']
    before = baseline['protocol']
    if (run['tuning_run_sha256'] != json_hash(tuning) or run['final_run_sha256'] != json_hash(final)
            or run['selection_sha256'] != json_hash(selection)
            or selection['tuning_run_sha256'] != run['tuning_run_sha256']):
        raise ValueError('Tuned phase or immutable selection differs from its sealed original')
    expected = dict(dataset='Fashion-MNIST', study='tuned_batch512', smoke=False, batch_size=512,
                    split_seed=17, train_rows=55000, validation_rows=5000, gallery_rows=55000, query_rows=1000,
                    architecture=[784, 128, 128, 128, 64, 10], weight_decay=.01,
                    candidate_grid=learning_rate_grid(), targets=list(TARGETS), gallery_format='fp32',
                    search_metric='cosine', k=10, block_size=256, no_metric_based_stopping=True,
                    offline_quality_workers=3, thread_limits=THREADS,
                    selection_rule='max_normalized_trapezoidal_validation_P_at10_over_updates;tie_terminal_precision_then_id')
    for name, phase in (('tuning', tuning), ('final', final)):
        p = phase['protocol']
        if (any(p.get(k) != value for k, value in expected.items()) or p['phase'] != name
                or p['backend'] != before['backend'] or p['data_files_sha256'] != before['data_files_sha256']):
            raise ValueError('Tuned batch512 protocol differs from the fixed experiment')
        for path, sha in before['source_sha256'].items():
            if path != 'experiments/run_report.py' and p['source_sha256'].get(path) != sha:
                raise ValueError('Tuned study shared source differs: ' + path)
    tp, fp = tuning['protocol'], final['protocol']
    if (tp['source_sha256'] != fp['source_sha256'] or tp['training_seeds'] != [7]
            or tp['evaluation_split'] != 'training_holdout' or tp['max_steps'] != 1620
            or tp['snapshot_steps'] != [0, 108, 216, 324, 540, 864, 1080, 1620]
            or fp['training_seeds'] != [8, 9, 10] or fp['max_steps'] != 10800
            or fp['snapshot_steps'] != final_schedule() or fp['evaluation_split'] != 'official_test_exploratory'
            or fp['selection_sha256'] != run['selection_sha256']
            or fp['tuning_run_sha256'] != run['tuning_run_sha256']):
        raise ValueError('Tuning/final phases or locked selection are incompatible')
    chosen = choose_rates(tuning['candidates'])
    if chosen != tuning['selected'] or chosen != selection['selected'] or chosen != fp['selected']:
        raise ValueError('Final rates were not selected by the frozen validation-only rule')
    ids = baseline['feature_metadata'][0]
    candidates = tuning['candidates']
    for r in candidates:
        validate_tuned_curve(r, tp, 'validation_curve')
        if (r['seed'] != 7 or r['gallery_ids_sha256'] != ids['gallery_ids_sha256']
                or r['query_ids_sha256'] == ids['query_ids_sha256']):
            raise ValueError('Tuning used the wrong gallery or final test-query selection')
    for key in ('initial_parameter_sha256', 'train_indices_sha256', 'validation_indices_sha256', 'query_ids_sha256'):
        if len({r[key] for r in candidates}) != 1:
            raise ValueError('Tuning candidates used different starts or images')
    original = training_by_seed(baseline['training_records'])
    records = final['trajectories']
    if len(records) != 6 or {(r['optimizer'], r['seed']) for r in records} != set(original):
        raise ValueError('Expected six fresh paired final tuned runs')
    for r in records:
        validate_tuned_curve(r, fp, 'curve')
        old = original[r['optimizer'], r['seed']]
        if (r['candidate'] != chosen[r['optimizer']]['candidate']
                or any(r[k] != old[k] for k in ('initial_parameter_sha256', 'train_indices_sha256', 'validation_indices_sha256'))
                or r['gallery_ids_sha256'] != ids['gallery_ids_sha256'] or r['query_ids_sha256'] != ids['query_ids_sha256']):
            raise ValueError('Final tuned learning rates, starts or retrieval IDs differ')
        prefix = next((s for s in r['snapshots'] if s['steps'] == 1080), None)
        if prefix and prefix['training_order_prefix_sha256'] != old['training_order_sha256']:
            raise ValueError('Tuned data-order prefix differs from the original ten epochs')
    summary = summarize_runs(records, fp['targets'], fp['training_seeds'])
    if summary != final['summary']:
        raise ValueError('Tuned target summaries differ from the raw final curves')
    return threshold_rows(summary, 'Batch 512 / ')


def collect_integrated_rows(run, parent):
    """Bind every final gallery trial to its own fixed terminal tuned checkpoint."""
    selected = select_terminal_snapshots(parent)
    p = run['protocol']
    if (p != make_protocol(parent, Path(p['parent_run']), selected)
            or p['smoke'] or p['max_steps'] != 10800 or p['training_seeds'] != list(SEEDS)
            or run['selected_snapshots'] != selected or not run['common_image_ids_verified']):
        raise ValueError('Integrated checkpoint selection or evaluation protocol differs')
    sources = {(r['optimizer'], r['seed']): r for r in selected}
    metadata = run['feature_metadata']
    features = {(r['encoder'], r['training_seed']): r for r in metadata}
    if len(metadata) != 6 or set(features) != set(sources):
        raise ValueError('Expected six own-model embedding archives')
    stamp = dict(protocol_sha256=json_hash(p))
    for key, feature in features.items():
        source = sources[key]
        if (feature['checkpoint_sha256'] != source['checkpoint_sha256']
                or feature['runner_provenance'] != stamp or feature['device'] != 'cpu()'
                or feature['gallery_shape'] != [55000, 64] or feature['query_shape'] != [1000, 64]
                or feature['gallery_ids_sha256'] != source['gallery_ids_sha256']
                or feature['query_ids_sha256'] != source['query_ids_sha256']
                or feature['data_files_sha256'] != p['data_files_sha256']
                or feature['validation_size_excluded'] != 5000 or feature['split_seed'] != 17
                or not math.isfinite(feature['export_wall_seconds']) or feature['export_wall_seconds'] <= 0):
            raise ValueError('Integrated own-model feature provenance or image selection differs')
    trials = run['retrieval_rows']
    expected = {(m, s, q, rotation) for m, s in sources for q in STORES
                for rotation in ((7, 8, 9) if q.startswith('turbo') else (7,))}
    keys = {(r['encoder'], r['training_seed'], r['method'], r['rotation_seed']) for r in trials}
    if len(trials) != 54 or keys != expected:
        raise ValueError('Missing or duplicate optimizer/store/rotation trial')
    validate_retrieval(trials)
    sizes = dict(fp32=14080000, uniform_2=1320000, uniform_4=2200000,
                 turbo_2=1116400, turbo_4=1996448)
    for row in trials:
        key = row['encoder'], row['training_seed']
        expected_stamp = dict(stamp, feature_archive_sha256=features[key]['feature_archive_sha256'])
        if (row['runner_provenance'] != expected_stamp or row['stored_bytes'] != sizes[row['method']]
                or row['compression_ratio_vs_fp32'] != 14080000 / row['stored_bytes']
                or any(not math.isfinite(row[k]) or row[k] < 0
                       for k in ('setup_seconds', 'build_seconds', 'relative_reconstruction_mse'))):
            raise ValueError('Integrated trial source, persistent bytes or build cost differs')
        if row['method'] == 'fp32' and (row['label_precision_at_k'] != sources[key]['expected_fp32_precision']
                                      or row['recall_at_k'] != 1.):
            raise ValueError('FP32 result differs from the source terminal trajectory')
    quality = aggregate_quality(trials)
    if quality != run['quality_summary']:
        raise ValueError('Integrated aggregate differs from complete raw trials')
    costs = {m: distribution([sources[m, seed]['training_seconds'] for seed in SEEDS]) for m in OPTIMIZERS}
    rows = quality_rows('Batch 512 / 100 epochs / tuned terminal', quality, costs)
    for row in rows:
        method = next(m for m, label in OPTIMIZERS.items() if label == row['optimizer'])
        store = next(q for q, label in STORES.items() if label == row['store'])
        delta = quality['encoders'][method][store]['precision_change_vs_own_fp32']
        states = {sources[method, seed]['optimizer_numeric_state_bytes'] for seed in SEEDS}
        if len(states) != 1:
            raise ValueError('Optimizer numeric state differs across paired seeds')
        row.update(precision_change_mean=delta['mean'], precision_change_sd=delta['sample_std'],
                   updates_mean=10800, updates_sd=0, optimizer_state_bytes=states.pop())
    return rows


def validate_gallery_reports(reports, previous, features, methods, seeds, gallery_rows):
    indexed = {(r['encoder'], r['training_seed']): r for r in reports}
    if len(indexed) != len(reports) or set(indexed) != set(features):
        raise ValueError('Missing unique own-model gallery report')
    rows = []
    for key, report in indexed.items():
        p, f = report['protocol'], features[key]
        expected = dict(gallery_shape=[gallery_rows, 64], query_shape=[1000, 64], methods=methods,
                        k=10, block_size=256, repeats=3, rotations=[7, 8, 9], metric='cosine',
                        backend=previous['protocol']['backend'], feature_archive_sha256=f['feature_archive_sha256'])
        if any(p.get(k) != v for k, v in expected.items()):
            raise ValueError('Gallery feature identity, format or query protocol differs')
        for path, sha in previous['protocol']['source_sha256'].items():
            if p['source_sha256'].get(path) != sha:
                raise ValueError('Shared gallery source differs: ' + path)
        for path in ('experiments/compare_galleries.py', 'experiments/uniform_compact.py'):
            if p['source_sha256'].get(path) != hashlib.sha256((ROOT / path).read_bytes()).hexdigest():
                raise ValueError('Measured gallery comparison source changed: ' + path)
        validate_retrieval(report['results'])
        for row in report['results']:
            if row['runner_provenance'] != dict(protocol_sha256=json_hash(p), feature_archive_sha256=f['feature_archive_sha256']):
                raise ValueError('Gallery worker feature provenance differs')
            method = row['method']
            if method == 'fp32':
                allowed = {gallery_rows * 64 * 4}
                if abs(row['label_precision_at_k'] - f['precision']) > 1e-12 or row['recall_at_k'] != 1:
                    raise ValueError('FP32 retrieval differs from its own model')
            else:
                bits = int(method.rsplit('_', 1)[1])
                codes = gallery_rows * 64 * bits // 8
                if method.startswith('uniform'): allowed = {codes + gallery_rows * 8}
                elif method.startswith('compact'): allowed = {codes + gallery_rows * 8 + 1, codes + gallery_rows * 4 + 5}
                elif method.startswith('symmetric'): allowed = {codes + gallery_rows * 4 + 1}
                else: allowed = {codes + gallery_rows * 4 + 64*64*4 + (1 << bits)*4}
            if row['stored_bytes'] not in allowed or row['compression_ratio_vs_fp32'] != gallery_rows*64*4/row['stored_bytes']:
                raise ValueError('Gallery storage excludes required metadata')
            rows.append(dict(row, encoder=key[0], training_seed=key[1]))
    summary = summarize_galleries(rows, seeds)
    return rows, summary


def gallery_table_rows(title, quality, selected, stores, seeds, gallery_rows):
    costs = {m: distribution([selected[m, s]['training_seconds'] for s in seeds]) for m in OPTIMIZERS}
    rows = quality_rows(title, quality, costs, stores, gallery_rows*64*4)
    for row in rows:
        method = next(m for m, label in OPTIMIZERS.items() if label == row['optimizer'])
        store = next(q for q, label in stores.items() if label == row['store'])
        delta = quality['encoders'][method][store]['precision_change_vs_own_fp32']
        updates = distribution([selected[method, s]['steps'] for s in seeds])
        states = {selected[method, s]['optimizer_numeric_state_bytes'] for s in seeds}
        if len(states) != 1:
            raise ValueError('Optimizer state changed across seed replicas')
        row.update(precision_change_mean=delta['mean'], precision_change_sd=delta['sample_std'],
                   updates_mean=updates['mean'], updates_sd=updates['sample_std'], optimizer_state_bytes=states.pop())
        if len(seeds) == 1:
            for key in ('precision_sd', 'recall_sd', 'training_sd', 'query_sd', 'precision_change_sd'):
                row[key] = None
    return rows


def collect_compact_rows(bundle, previous):
    if bundle['parent_integrated_sha256'] != json_hash(previous):
        raise ValueError('Compact metadata follow-up uses a different parent')
    features = {(f['encoder'], f['training_seed']): dict(f, precision=next(
        r['label_precision_at_k'] for r in previous['retrieval_rows']
        if (r['encoder'], r['training_seed'], r['method']) == (f['encoder'], f['training_seed'], 'fp32')))
        for f in previous['feature_metadata']}
    methods = ['fp32', 'uniform_2', 'uniform_4', 'compact_2', 'compact_4']
    raw, quality = validate_gallery_reports(bundle['reports'], previous, features, methods, [8, 9, 10], 55000)
    if quality != bundle['quality_summary']:
        raise ValueError('Compact uniform summary differs from raw measurements')
    for encoder, seed in features:
        for bits in (2, 4):
            affine, compact = [next(r for r in raw if (r['encoder'], r['training_seed'], r['method']) ==
                                        (encoder, seed, prefix+str(bits))) for prefix in ('uniform_', 'compact_')]
            if (compact['stored_bytes'] != 55000*64*bits//8 + 55000*4 + 5
                    or any(compact[k] != affine[k] for k in ('label_precision_at_k', 'recall_at_k'))
                    or ('neighbor_indices_sha256' in affine and compact.get('neighbor_indices_sha256') != affine['neighbor_indices_sha256'])):
                raise ValueError('Compact minimum storage changed decoded retrieval or omitted metadata')
            old = next(r for r in previous['retrieval_rows'] if
                       (r['encoder'], r['training_seed'], r['method']) == (encoder, seed, 'uniform_'+str(bits)))
            if any(old[k] != affine[k] for k in ('label_precision_at_k', 'recall_at_k')):
                raise ValueError('Fresh uniform reference differs from the parent model')
    selected = {(r['optimizer'], r['seed']): r for r in previous['selected_snapshots']}
    return gallery_table_rows('Fashion / shared-min follow-up / same terminal models', quality, selected,
                              {'compact_2': 'Uniform shared-min 2-bit', 'compact_4': 'Uniform shared-min 4-bit'},
                              [8, 9, 10], 55000)


def collect_cifar_rows(bundle, previous):
    pilot = bundle['pilot'];p = pilot['protocol']
    expected = dict(dataset='CIFAR-100', study='bounded_cifar_pilot', smoke=False, model=model_config(),
                    train_rows=10000, validation_rows=2000, train_per_class=100, validation_per_class=20,
                    query_per_class=10, batch_size=256, split_seed=17, training_seed=7, final_seeds=[8],
                    official_test_used=False, tuning_epochs=5, rate_grid=rate_grid(), weight_decay=.01, k=10, metric='cosine')
    if any(p.get(k) != v for k, v in expected.items()) or p['backend'] != previous['protocol']['backend']:
        raise ValueError('CIFAR pilot protocol or scope differs')
    for path, sha in previous['protocol']['source_sha256'].items():
        if p['source_sha256'].get(path) != sha:
            raise ValueError('CIFAR shared framework source differs: ' + path)
    for path in ('apps/cifar_model.py', 'experiments/cifar_data.py', 'experiments/cifar_pilot.py'):
        if p['source_sha256'].get(path) != hashlib.sha256((ROOT / path).read_bytes()).hexdigest():
            raise ValueError('Measured CIFAR source changed: ' + path)
    cost = max(r['median_seconds'] for r in pilot['probe'].values())
    if p['final_epochs'] != final_epochs(cost):
        raise ValueError('Pilot training budget was changed after the cost probe')
    tuning, final = pilot['tuning'], pilot['final']
    configs = {c['id']: (m, c) for m in p['rate_grid'] for c in p['rate_grid'][m]}
    if len(tuning) != 6 or {r['candidate']['id'] for r in tuning} != set(configs):
        raise ValueError('CIFAR tuning candidates were omitted or duplicated')
    if len(final) != 2 or {(r['optimizer'], r['seed']) for r in final} != {(m, 8) for m in OPTIMIZERS}:
        raise ValueError('Expected one complete paired final CIFAR pilot seed')
    for phase, records in (('tuning', tuning), ('final', final)):
        epochs = 5 if phase == 'tuning' else p['final_epochs']
        planned = [40*e for e in ([0, 1, 3, 5] if phase == 'tuning' else [0, 1, 5, 10, 15, 20]) if e <= epochs]
        for r in records:
            method = r['optimizer']
            raw = {k: v for k, v in r.items() if k not in ('curve', 'training_record_sha256', 'feature_archive',
                         'feature_archive_sha256', 'gallery_ids_sha256', 'query_ids_sha256')}
            snaps = r['snapshots'];curve = r['curve']
            if (r['training_record_sha256'] != json_hash(raw) or r['protocol_sha256'] != json_hash(p)
                    or r['candidate'] != configs[r['candidate']['id']][1]
                    or method != configs[r['candidate']['id']][0]
                    or r['seed'] != (7 if phase == 'tuning' else 8)
                    or r['total_parameter_count'] != 54788 or r['muon_eligible_parameter_count'] != 34816
                    or r['muon_parameter_count'] != (34816 if method == 'muon' else 0)
                    or r['optimizer_numeric_state_bytes'] != (438304 if method == 'adamw' else 299040)
                    or r['train_ids_sha256'] != p['train_ids_sha256'] or r['validation_ids_sha256'] != p['validation_ids_sha256']
                    or len(curve) != len(snaps) or not snaps
                    or [s['steps'] for s in snaps] != planned[:len(snaps)]):
                raise ValueError('CIFAR model, rate, checkpoint or raw training provenance differs')
            if r['status'] == 'complete':
                if (len(snaps) != len(planned) or r['steps'] != epochs*40 or r['samples_seen'] != epochs*10000
                        or r['training_seconds'] != snaps[-1]['training_seconds']):
                    raise ValueError('CIFAR completed training budget or clock differs')
            elif r['status'] != 'diverged':
                raise ValueError('Unknown CIFAR training outcome')
            for snap, point in zip(snaps, curve):
                if (any(point.get(k) != v for k, v in snap.items())
                        or not math.isfinite(point['precision']) or not 0 <= point['precision'] <= 1
                        or abs(point['precision'] - point['same_category_matches']/10000) > 1e-12
                        or snap['samples_seen'] != snap['steps']//40*10000):
                    raise ValueError('CIFAR retrieval curve differs from saved training observations')
    chosen = {m: choose_candidate([r for r in tuning if r['optimizer'] == m]) for m in OPTIMIZERS}
    selection = dict(selected=chosen, tuning_sha256=hashlib.sha256(json.dumps(tuning, sort_keys=True).encode()).hexdigest())
    if pilot['selection'] != selection or pilot['selection_sha256'] != json_hash(selection):
        raise ValueError('CIFAR rates were not locked from the full validation search')
    for r in final:
        if r['status'] != 'complete' or r['candidate'] != chosen[r['optimizer']]['candidate']:
            raise ValueError('Final CIFAR pilot must retain both locked complete models')
    for records in (tuning, final):
        complete = [r for r in records if r['status'] == 'complete']
        for key in ('initial_parameter_sha256', 'training_order_sha256', 'augmentation_sha256',
                    'train_ids_sha256', 'validation_ids_sha256', 'gallery_ids_sha256', 'query_ids_sha256'):
            if len({r[key] for r in complete}) != 1:
                raise ValueError('CIFAR paired starts, image IDs or augmented data order differ')
    features = {(r['optimizer'], r['seed']): dict(feature_archive_sha256=r['feature_archive_sha256'],
                    precision=r['curve'][-1]['precision']) for r in final}
    raw, quality = validate_gallery_reports(bundle['reports'], previous, features, GALLERY_METHODS, [8], 10000)
    if quality != bundle['quality_summary']:
        raise ValueError('CIFAR gallery aggregate differs from all raw trials')
    stores = {m: ({'compact_2': 'Uniform compact 2-bit', 'compact_4': 'Uniform compact 4-bit',
                  'symmetric_2': 'Uniform symmetric 2-bit', 'symmetric_4': 'Uniform symmetric 4-bit'}.get(m) or STORES[m])
              for m in GALLERY_METHODS}
    return gallery_table_rows(f'CIFAR-100 pilot / {p["final_epochs"]} epochs / seed 8', quality,
                              { (r['optimizer'], r['seed']): r for r in final}, stores, [8], 10000)


def collect_cifar_time_rows(run,parent):
    p=run['protocol'];before=parent['protocol']
    expected=dict(dataset='CIFAR-100',study='cifar_time_to_quality',smoke=False,targets=CIFAR_TARGETS,
                  budget_seconds=600.,final_seeds=CIFAR_SEEDS,selected=parent['selection']['selected'],
                  parent_pilot_sha256=json_hash(parent),parent_selection_sha256=parent['selection_sha256'],
                  observation_rule='every20_updates_through400_then_every40;always_terminal',
                  offline_quality_workers=3,no_metric_based_stopping=True)
    for key in ('model','batch_size','train_rows','validation_rows','train_per_class','validation_per_class',
                'query_per_class','train_ids_sha256','validation_ids_sha256','data_sha256','mean','std',
                'split_seed','augmentation','weight_decay','official_test_used','backend'):
        expected[key]=before[key]
    if any(p.get(k)!=v for k,v in expected.items()):
        raise ValueError('CIFAR matched-quality targets, budget, selected rates or model/data changed')
    for path,sha in before['source_sha256'].items():
        if p['source_sha256'].get(path)!=sha:raise ValueError('CIFAR matched-quality shared source differs: '+path)
    path='experiments/cifar_time_to_quality.py'
    if p['source_sha256'].get(path)!=hashlib.sha256((ROOT/path).read_bytes()).hexdigest():
        raise ValueError('Measured CIFAR trajectory runner changed')
    records=run['trajectories']
    if len(records)!=6 or {(r['optimizer'],r['seed']) for r in records}!={(m,s) for m in OPTIMIZERS for s in CIFAR_SEEDS}:
        raise ValueError('Expected all six fresh paired CIFAR trajectories')
    ids=parent['final'][0]
    for r in records:
        raw={k:v for k,v in r.items() if k not in ('curve','training_record_sha256','feature_archive',
             'feature_archive_sha256','gallery_ids_sha256','query_ids_sha256')}
        method=r['optimizer'];complete=r['status']=='complete'
        states=438304 if method=='adamw' else 299040
        if (r['protocol_sha256']!=json_hash(p) or r['training_record_sha256']!=json_hash(raw)
                or r['candidate']!=p['selected'][method]['candidate']
                or r['total_parameter_count']!=54788 or r['muon_eligible_parameter_count']!=34816
                or r['muon_parameter_count']!=(34816 if method=='muon' else 0)
                or r['optimizer_numeric_state_bytes'] not in ({states} if complete or r['steps'] else {0,states})
                or r['train_ids_sha256']!=p['train_ids_sha256'] or r['validation_ids_sha256']!=p['validation_ids_sha256']
                or any(r[k]!=ids[k] for k in ('gallery_ids_sha256','query_ids_sha256'))
                or r['status'] not in ('complete','diverged')
                or len(r['training_losses'])!=r['steps']
                or any(not math.isfinite(x) for x in r['training_losses'])):
            raise ValueError('CIFAR trajectory raw data, model, rate or training record differs')
        if complete and (r['stop_reason']!='training_time_budget'
                or not 600<=r['training_seconds']<=600+r['last_step_seconds']+1e-6
                or r['data_attempts']!=r['steps']):
            raise ValueError('CIFAR trajectory elapsed budget or complete-update clock differs')
        snapshots=r['snapshots'];curve=r['curve']
        planned=[0,*[n for n in range(1,r['steps']+1) if observation_due(n)]]
        if complete and planned[-1]!=r['steps']:planned.append(r['steps'])
        if (not snapshots or len(snapshots)!=len(curve)
                or [s['steps'] for s in snapshots]!=(planned if complete else planned[:len(snapshots)])
                or (complete and snapshots[-1]['training_seconds']!=r['training_seconds'])
                or len(r['epoch_order_sha256'])!=(r['data_attempts']+39)//40):
            raise ValueError('CIFAR trajectory observation schedule or terminal budget differs')
        for snap,point in zip(snapshots,curve):
            n=snap['steps'];samples=n//40*10000+(n%40)*256
            if (any(point.get(k)!=v for k,v in snap.items()) or snap['samples_seen']!=samples
                    or not math.isfinite(point['precision']) or not 0<=point['precision']<=1
                    or abs(point['precision']-point['same_category_matches']/10000)>1e-12):
                raise ValueError('CIFAR retrieval observations differ from actual training snapshots')
    summary=summarize_targets(records,p['targets'],p['final_seeds'])
    if summary!=run['summary']:raise ValueError('CIFAR matched-quality summary differs from all raw curves')
    for seed in CIFAR_SEEDS:
        pair=[next(r for r in records if (r['seed'],r['optimizer'])==(seed,m)) for m in OPTIMIZERS]
        if any(pair[0]['curve'][0][k]!=pair[1]['curve'][0][k] for k in ('precision','same_category_matches','neighbor_indices_sha256')):
            raise ValueError('Paired identical initial CIFAR models have different retrieval results')
    rows=threshold_rows(summary,'CIFAR / 600s budget / ')
    for row in rows:
        method=next(m for m,label in OPTIMIZERS.items() if label==row['optimizer'])
        values={r['optimizer_numeric_state_bytes'] for r in records if r['optimizer']==method}
        row.update(gallery_bytes=2560000,optimizer_state_bytes=values.pop() if len(values)==1 else None)
    return rows


def collect_fresh_rows(data):
    """One table: all current-backend methods, terminal checks, and total systems cost."""
    if set(data['studies']) != {'epochs', 'timed'}:
        raise ValueError('Expected both equal-epoch and equal-time studies')
    rows, checks, backends, images, sources, datasets = [], [], [], [], [], []
    for name in ('epochs', 'timed'):
        run = data['studies'][name]
        quality, terminal, indexed, common_ids = validate_fresh_study(run, name)
        p = run['protocol']
        backends.append(p['backend'])
        images.append(common_ids)
        sources.append(p['source_sha256'])
        datasets.append(p['data_files_sha256'])
        costs, updates, states = {}, {}, {}
        for m in OPTIMIZERS:
            records = [indexed[m, seed] for seed in SEEDS]
            costs[m] = distribution([sum(e['training_seconds'] for e in r['epochs'])
                                     if name == 'epochs' else r['timed_seconds'] for r in records])
            updates[m] = distribution([sum(e['optimizer_steps'] for e in r['epochs'])
                                       if name == 'epochs' else r['optimizer_steps'] for r in records])
            states[m] = records[0]['optimizer_numeric_state_bytes']
        title = '10 epochs' if name == 'epochs' else '220s'
        for destination, summary, label in ((rows, quality, title + ' / final CPU'),
                                            (checks, terminal, title + ' / terminal check')):
            for row in quality_rows(label, summary, costs):
                m = next(m for m, label in OPTIMIZERS.items() if label == row['optimizer'])
                q = next(q for q, label in STORES.items() if label == row['store'])
                delta = summary['encoders'][m][q]['precision_change_vs_own_fp32']
                row.update(precision_change_mean=delta['mean'], precision_change_sd=delta['sample_std'],
                           updates_mean=updates[m]['mean'], updates_sd=updates[m]['sample_std'],
                           optimizer_state_bytes=states[m])
                destination.append(row)
    if any(v != backends[0] for v in backends) or any(v != sources[0] for v in sources):
        raise ValueError('Fresh studies used different native binaries or source code')
    if any(v != images[0] for v in images) or any(v != datasets[0] for v in datasets):
        raise ValueError('Fresh studies used different images or data files')
    by_budget = [training_by_seed(data['studies'][name]['training_records'], timed=name == 'timed')
                 for name in ('epochs', 'timed')]
    for key in by_budget[0]:
        for field in ('initial_parameter_sha256', 'train_indices_sha256', 'validation_indices_sha256'):
            if by_budget[0][key][field] != by_budget[1][key][field]:
                raise ValueError('Training initialization or image selections differ across budgets')
    system = data['system']
    summary = validate_measurements(system['runs'])
    if summary != system['summary'] or len(system['runs']) != 2 * system['protocol']['blocks']:
        raise ValueError('System paired timing differs from raw measurements')
    new_binary = next(r['native_extension_sha256'] for r in system['runs'] if r['variant'] == 'new')
    if new_binary != backends[0]['extension_sha256']:
        raise ValueError('Systems and quality runs used different final binaries')
    for path in set(system['protocol']['source_sha256']) & set(sources[0]):
        if system['protocol']['source_sha256'][path] != sources[0][path]:
            raise ValueError('Systems and quality runs used different shared source code')
    for variant in ('old', 'new'):
        for m in OPTIMIZERS:
            metric = summary['training_' + m]
            checks.append(dict(study='128 updates / ' + ('original CPU' if variant == 'old' else 'final CPU'),
                               optimizer=OPTIMIZERS[m], store='—', precision_mean=None, precision_sd=None,
                               recall_mean=None, recall_sd=None, gallery_bytes=None, compression=None,
                               training_cost=1000 * metric[variant + '_seconds']['median'], training_sd=None,
                               training_unit='ms/step', speedup=metric['speedup']['median'] if variant == 'new' else None,
                               query_mean=None, query_sd=None, precision_change_mean=None, precision_change_sd=None,
                               updates_mean=128, updates_sd=0, optimizer_state_bytes=None))
    if 'time_to_quality' in data:
        rows.extend(collect_time_to_quality_rows(data['time_to_quality'], data['studies']['epochs']))
    batch_rows = collect_batch_rows(data['batch_timing'], data['studies']['epochs']) if 'batch_timing' in data else []
    tuned_rows = collect_tuned512_rows(data['tuned_batch512'], data['studies']['epochs']) if 'tuned_batch512' in data else []
    if 'integrated' in data:
        current = collect_integrated_rows(data['integrated'], data['tuned_batch512'])
        # Superseded batch-64 studies remain fully validated above and in the JSON.
        systems = [r for r in checks if r['training_unit'] == 'ms/step']
        compact = collect_compact_rows(data['compact_uniform'], data['integrated']) if 'compact_uniform' in data else []
        cifar = collect_cifar_rows(data['cifar_pilot'], data['integrated']) if 'cifar_pilot' in data else []
        cifar_time=collect_cifar_time_rows(data['cifar_time_to_quality'],data['cifar_pilot']['pilot']) if 'cifar_time_to_quality' in data else []
        return current + compact + cifar + cifar_time + systems + batch_rows + tuned_rows
    return rows + checks + batch_rows + tuned_rows


def collect_rows(data):
    if data['schema_version'] != 2:
        raise ValueError('Expected the fresh study evidence schema')
    return collect_fresh_rows(data)


def display_rows(rows):
    def pm(mean, sd, scale=1, digits=2):
        if mean is None:
            return '—'
        return f'{scale*mean:.{digits}f}' if sd is None else f'{scale*mean:.{digits}f} ± {scale*sd:.{digits}f}'
    for r in rows:
        cost = ('—' if r['training_cost'] is None else
                f"{r['training_cost']:.3f} {r['training_unit']}" if r['training_sd'] is None else
                pm(r['training_cost'], r['training_sd'], digits=3) + ' ' + r['training_unit'])
        if r['speedup'] is not None:
            cost += f" ({r['speedup']:.3f}×)"
        delta = ('—' if r['precision_change_mean'] is None else f"{100*r['precision_change_mean']:+.2f}")
        if r['precision_change_mean'] is not None and r['precision_change_sd'] is not None:
            delta += f" ± {100*r['precision_change_sd']:.2f}"
        updates = '—' if r['updates_mean'] is None else f"{r['updates_mean']:,.0f}"
        if r['updates_sd']:
            updates += f" ± {r['updates_sd']:,.0f}"
        yield [r['study'], r['optimizer'], r['store'], pm(r['precision_mean'], r['precision_sd'], 100),
               delta, pm(r['recall_mean'], r['recall_sd'], 100),
               '—' if r['gallery_bytes'] is None else f"{r['gallery_bytes']/1e6:.4f}",
               '—' if r['compression'] is None else f"{r['compression']:.2f}×", cost, updates,
               '—' if r['optimizer_state_bytes'] is None else f"{r['optimizer_state_bytes']/1e6:.4f}",
               pm(r['query_mean'], r['query_sd'], digits=3)]


def render_markdown(rows):
    lines = ['| ' + ' | '.join(HEADERS) + ' |', '|' + '|'.join(['---'] * len(HEADERS)) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(row) + ' |' for row in display_rows(rows)])


def render_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(HEADERS)
    writer.writerows(display_rows(rows))
    return output.getvalue()


def verify_history(data):
    revision = data['provenance']['previous_report_commit']
    previous = json.loads(subprocess.check_output(['git', 'show', f'{revision}:report/results.json'], cwd=ROOT))
    if (data['needle_origin'] != previous['needle_origin']
            or data['locked_rate_selection'] != previous['fixed_epochs']['selection']):
        raise ValueError('Origin or locked hyperparameter selection differs from history')
    baseline = data['system']['protocol']
    for path, expected in baseline['baseline_source_sha256'].items():
        raw = subprocess.check_output(['git', 'show', f'{baseline["baseline_commit"]}:{path}'], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original-backend source differs from pinned history')
    return


def verify_artifacts(data):
    """Independently reload all selected/terminal models and their own feature samples.

    Run after retained timings: no training, search timing or checkpoint selection
    occurs here. Local ignored artifacts are required for this optional audit.
    """
    if data['schema_version'] != 2:
        raise ValueError('Artifact audit requires the fresh study dataset')
    import sysconfig
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn
    from encoder_checkpoint import load_encoder
    from train_baseline import select_training_indices

    def check_hash(path, expected):
        path = Path(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Local artifact hash mismatch: ' + str(path))

    extension = ROOT / 'python/needle/backend_ndarray' / ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    check_hash(extension, data['studies']['epochs']['protocol']['backend']['extension_sha256'])
    directory = ROOT / 'data/fashion-mnist'
    train = ndl.data.MNISTDataset(str(directory / 'train-images-idx3-ubyte.gz'),
                                 str(directory / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(directory / 't10k-images-idx3-ubyte.gz'),
                                str(directory / 't10k-labels-idx1-ubyte.gz'))
    proof = dict(trained_models=0, checkpoint_validation_reloads=0,
                 feature_archives=0, feature_sample_rows=0, raw_query_reports=0)
    query_environment = None
    criterion = nn.SoftmaxLoss()
    for name, run in data['studies'].items():
        p = run['protocol']
        stamp = {'protocol_sha256': json_hash(p)}
        for path, expected in p['data_files_sha256'].items():
            check_hash(directory / path, expected)
        check_hash(ROOT / data['provenance']['raw_run_paths'][name], data['provenance']['raw_run_sha256'][name])
        indexed = training_by_seed(run['training_records'], timed=name == 'timed')
        for (method, seed), record in indexed.items():
            root = Path(record['best_checkpoint_export']).parent
            disk_record = json.loads((root / f'{method}_seed{seed}_train.json').read_text())
            if disk_record != record or json.loads((root / 'protocol.json').read_text()) != p:
                raise ValueError('Training record or frozen protocol differs from local originals')
            check_hash(record['best_checkpoint_export'], record['best_checkpoint_sha256'])
            check_hash(record['checkpoint_export'], record['terminal_checkpoint_sha256'])
            train_ids, val_ids, _ = select_training_indices(len(train), len(train), 55000, 5000,
                                                           np.random.default_rng(seed), 5000, 17)
            for ids, field in ((train_ids, 'train_indices_sha256'), (val_ids, 'validation_indices_sha256')):
                if hashlib.sha256(ids.astype('int64').tobytes()).hexdigest() != record[field]:
                    raise ValueError('Training/validation selection cannot be reproduced')
            proof['trained_models'] += 1
        for metadata_key, rows_key, terminal in (('feature_metadata', 'retrieval_rows', False),
                                                  ('checkpoint_check_metadata', 'checkpoint_check_rows', True)):
            for metadata in run[metadata_key]:
                method, seed = metadata['encoder'], metadata['training_seed']
                r = indexed[method, seed]
                checkpoint = Path(r['checkpoint_export'] if terminal else r['best_checkpoint_export'])
                features = Path(metadata['feature_archive'])
                check_hash(checkpoint, metadata['checkpoint_sha256'])
                check_hash(features, metadata['feature_archive_sha256'])
                if json.loads(features.with_suffix('.json').read_text()) != {
                        k: v for k, v in metadata.items() if k not in ('encoder', 'training_seed')}:
                    raise ValueError('Curated feature metadata differs from original')
                model = load_encoder(checkpoint, ndl.cpu())
                model.eval()
                val_ids = np.random.default_rng(17).permutation(len(train))[:5000]
                total_loss, correct = 0., 0
                for start in range(0, len(val_ids), 64):
                    ids = val_ids[start:start + 64]
                    x = ndl.Tensor(train.X[ids], device=ndl.cpu(), requires_grad=False)
                    y = ndl.Tensor(train.y[ids], device=ndl.cpu(), requires_grad=False)
                    scores = model(x)
                    total_loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * len(ids)
                    correct += int((scores.numpy().argmax(1) == train.y[ids]).sum())
                expected = (r['epochs'][-1] if name == 'epochs' else r['checkpoints'][-1]) if terminal else {
                    'validation_loss': r['best_validation_loss'], 'validation_accuracy': r['best_validation_accuracy']}
                if (abs(total_loss / 5000 - expected['validation_loss']) > 1e-7
                        or correct / 5000 != expected['validation_accuracy']):
                    raise ValueError('Reloaded checkpoint does not reproduce full validation')
                proof['checkpoint_validation_reloads'] += 1
                with np.load(features, allow_pickle=False) as arrays:
                    gallery_ids, query_ids = arrays['gallery_ids'], arrays['query_ids']
                    if (len(np.unique(gallery_ids)) != 55000 or np.intersect1d(gallery_ids, val_ids).size
                            or len(np.unique(query_ids)) != 1000):
                        raise ValueError('Retrieval IDs overlap validation or contain duplicates')
                    for ids, field in ((gallery_ids, 'gallery_ids_sha256'), (query_ids, 'query_ids_sha256')):
                        if hashlib.sha256(ids.astype('int64').tobytes()).hexdigest() != metadata[field]:
                            raise ValueError('Retrieval image IDs differ from frozen hashes')
                    np.testing.assert_array_equal(arrays['gallery_labels'], train.y[gallery_ids])
                    np.testing.assert_array_equal(arrays['query_labels'], test.y[query_ids])
                    for key, dataset, ids in (('gallery', train, gallery_ids), ('queries', test, query_ids)):
                        values = arrays[key]
                        if not np.isfinite(values).all():
                            raise ValueError('Feature archive contains nonfinite values')
                        sample = model.features(ndl.Tensor(dataset.X[ids[:64]], device=ndl.cpu(), requires_grad=False)).numpy()
                        np.testing.assert_array_equal(sample, values[:64])
                        proof['feature_sample_rows'] += 64
                proof['feature_archives'] += 1
                prefix = f'{method}_seed{seed}' + ('_terminal' if terminal else '')
                for rotation in ((7,) if terminal else (7, 8, 9)):
                    report = json.loads((features.parent / f'{prefix}_rotation{rotation}.json').read_text())
                    if query_environment is None:
                        query_environment = report['environment']
                    if report['environment'] != query_environment:
                        raise ValueError('Query workers used different software environments')
                    if report['runner_provenance'] != dict(stamp, feature_archive_sha256=metadata['feature_archive_sha256']):
                        raise ValueError('Query report refers to a different feature archive or protocol')
                    original = [dict(row, encoder=method, training_seed=seed, rotation_seed=rotation)
                                for row in report['results']]
                    retained = [row for row in run[rows_key] if row['encoder'] == method
                                and row['training_seed'] == seed and row['rotation_seed'] == rotation]
                    if original != retained:
                        raise ValueError('Curated query rows differ from original worker output')
                    proof['raw_query_reports'] += 1
    proof['query_environment'] = query_environment
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--verify-artifacts', action='store_true', help='Audit original local models/features/query reports after retained timings')
    parser.add_argument('--time-to-quality-run', type=Path,
                        help='Attach a complete, validated time-to-quality run to the same evidence/table')
    parser.add_argument('--batch-run', type=Path, help='Attach fixed-image batch timing to the same evidence/table')
    parser.add_argument('--tuned512-run', type=Path, help='Attach validation-tuned batch512 target comparisons')
    parser.add_argument('--integrated-run', type=Path, help='Attach fixed terminal tuned optimizer × gallery comparison')
    parser.add_argument('--compact-run', type=Path, help='Attach shared-min uniform comparison on the existing Fashion models')
    parser.add_argument('--cifar-run', type=Path, help='Attach a labelled one-seed CIFAR model/data feasibility pilot')
    parser.add_argument('--cifar-time-run',type=Path,help='Attach the three-seed 600s CIFAR matched-quality study')
    args = parser.parse_args()
    evidence = ROOT / 'report/results.json'
    data = json.loads(evidence.read_text())
    if args.time_to_quality_run:
        path = args.time_to_quality_run.resolve()
        data['time_to_quality'] = json.loads(path.read_text())
        data['provenance']['time_to_quality_raw_path'] = str(path.relative_to(ROOT))
        data['provenance']['time_to_quality_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.batch_run:
        path = args.batch_run.resolve()
        data['batch_timing'] = json.loads(path.read_text())
        data['provenance']['batch_raw_path'] = str(path.relative_to(ROOT))
        data['provenance']['batch_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.tuned512_run:
        path = args.tuned512_run.resolve()
        data['tuned_batch512'] = json.loads(path.read_text())
        data['provenance']['tuned512_raw_path'] = str(path.relative_to(ROOT))
        data['provenance']['tuned512_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.integrated_run:
        path = args.integrated_run.resolve()
        data['integrated'] = json.loads(path.read_text())
        data['provenance']['integrated_raw_path'] = str(path.relative_to(ROOT))
        data['provenance']['integrated_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for argument, key in ((args.compact_run, 'compact_uniform'), (args.cifar_run, 'cifar_pilot'),
                          (args.cifar_time_run,'cifar_time_to_quality')):
        if argument:
            path = argument.resolve()
            data[key] = json.loads(path.read_text())
            data['provenance'][key+'_raw_path'] = str(path.relative_to(ROOT))
            data['provenance'][key+'_raw_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = collect_rows(data)
    if json_hash(data['system']) != data['provenance']['system_raw_sha256']:
        raise ValueError('Curated systems data differs from its frozen raw report')
    for name, study in data['studies'].items():
        if json_hash(study) != data['provenance']['raw_run_sha256'][name]:
            raise ValueError('Curated study differs from the frozen raw run: ' + name)
        for path, expected in study['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured quality source changed: ' + path)
    for path, expected in data['system']['protocol']['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Measured system source changed: ' + path)
    if 'time_to_quality' in data:
        run = data['time_to_quality']
        if json_hash(run) != data['provenance']['time_to_quality_raw_sha256']:
            raise ValueError('Time-to-quality data differs from its frozen raw run')
        for path, expected in run['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured time-to-quality source changed: ' + path)
    if 'batch_timing' in data:
        run = data['batch_timing']
        if json_hash(run) != data['provenance']['batch_raw_sha256']:
            raise ValueError('Batch timing differs from its frozen raw run')
        for path, expected in run['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured batch source changed: ' + path)
    if 'tuned_batch512' in data:
        run = data['tuned_batch512']
        if json_hash(run) != data['provenance']['tuned512_raw_sha256']:
            raise ValueError('Tuned batch512 evidence differs from its frozen raw run')
        for path, expected in run['final']['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured tuned batch512 source changed: ' + path)
    if 'integrated' in data:
        run = data['integrated']
        if json_hash(run) != data['provenance']['integrated_raw_sha256']:
            raise ValueError('Integrated comparison differs from its frozen raw run')
        if run['protocol']['parent_tuned_run_sha256'] != json_hash(data['tuned_batch512']):
            raise ValueError('Integrated encoders differ from the curated tuned study')
        for path, expected in run['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured integrated source changed: ' + path)
    for key in ('compact_uniform', 'cifar_pilot','cifar_time_to_quality'):
        if key in data and json_hash(data[key]) != data['provenance'][key+'_raw_sha256']:
            raise ValueError('New gallery/pilot evidence differs from its sealed raw run: ' + key)
    for method in OPTIMIZERS:
        selection = data['locked_rate_selection']
        if select_configuration(selection['all_candidates'][method]) != selection['selected'][method]:
            raise ValueError('Locked learning-rate selection rule is inconsistent')
    if args.verify_history:
        verify_history(data)
    if args.verify_artifacts:
        proof = verify_artifacts(data)
        print('Local artifact proof:', json.dumps(proof, sort_keys=True))
        if not args.check:
            data['provenance']['artifact_validation'] = proof
    if not args.check and (args.verify_artifacts or args.time_to_quality_run or args.batch_run or args.tuned512_run or args.integrated_run or args.compact_run or args.cifar_run or args.cifar_time_run):
        evidence.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    readme = ROOT / 'README.md'
    current = readme.read_text()
    before, rest = current.split('<!-- RESULTS:START -->\n')
    _, after = rest.split('<!-- RESULTS:END -->')
    wanted = before + '<!-- RESULTS:START -->\n' + render_markdown(rows) + '\n<!-- RESULTS:END -->' + after
    csv_path, csv_text = ROOT / 'report/results.csv', render_csv(rows)
    if args.check:
        if current != wanted or not csv_path.exists() or csv_path.read_text() != csv_text:
            parser.exit(1, 'README/CSV are stale;run python tools/build_report.py\n')
    else:
        readme.write_text(wanted)
        csv_path.write_text(csv_text)
    print(f'Validated {len(rows)} fresh consolidated rows;README and CSV match the evidence.')


if __name__ == '__main__':
    main()
