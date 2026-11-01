"""Validation-only rate search followed by locked, paired batch512 retrieval curves."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time

from common import digest, distribution, save_json
from time_to_quality import TARGETS, THREADS, check_frozen, first_crossing, parameter_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]


def learning_rate_grid():
    return {method: [dict(id=f'{method}_{i}', lr=lr, muon_lr=muon if method == 'muon' else None)
                     for i, (lr, muon) in enumerate(zip((.001, .003, .01), (.007, .021, .07)))]
            for method in ('adamw', 'muon')}


def final_schedule():
    return [0, *range(54, 1081, 54), *range(1296, 10801, 216)]


def validation_auc(curve):
    if len(curve) < 2 or curve[0]['steps'] != 0 or curve[-1]['steps'] <= 0:
        raise ValueError('Expected a complete validation learning curve beginning at zero')
    if any(not math.isfinite(r['precision']) or not 0 <= r['precision'] <= 1 for r in curve):
        raise ValueError('Non-finite validation retrieval precision')
    if any(a['steps'] >= b['steps'] for a, b in zip(curve, curve[1:])):
        raise ValueError('Validation checkpoints must increase')
    return math.fsum((a['precision'] + b['precision']) * .5 * (b['steps'] - a['steps'])
                     for a, b in zip(curve, curve[1:])) / curve[-1]['steps']


def choose_rates(records):
    grid = learning_rate_grid()
    expected = {c['id']: (method, c) for method, configs in grid.items() for c in configs}
    indexed = {r['candidate']['id']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(expected):
        raise ValueError('Expected all six unique learning-rate candidates')
    selected = {}
    for method in grid:
        eligible = []
        for config in grid[method]:
            r = indexed[config['id']]
            if r['optimizer'] != method or r['candidate'] != config or r['status'] not in ('complete', 'diverged'):
                raise ValueError('Candidate identity or training status differs')
            if r['status'] == 'complete':
                eligible.append(dict(candidate=config, validation_auc=validation_auc(r['validation_curve']),
                                     terminal_validation_precision=r['validation_curve'][-1]['precision']))
        if not eligible:
            raise RuntimeError('All validation candidates diverged: ' + method)
        selected[method] = min(eligible, key=lambda r: (-r['validation_auc'],
                                                      -r['terminal_validation_precision'], r['candidate']['id']))
    return selected


def train_updates(step, save_snapshot, schedule, clock=time.perf_counter):
    if not schedule or schedule[0] != 0 or len(schedule) < 2 or any(a >= b for a, b in zip(schedule, schedule[1:])):
        raise ValueError('Expected increasing checkpoints beginning at zero')
    record = dict(status='complete', steps=0, samples_seen=0, training_seconds=0., optimizer_seconds=0.,
                  checkpoint_seconds=0., snapshots=[])
    wall_started = clock()

    def save():
        row = {k: record[k] for k in ('steps', 'samples_seen', 'training_seconds', 'optimizer_seconds')}
        before = clock()
        try:
            row.update(save_snapshot(row))
        finally:
            record['checkpoint_seconds'] += clock() - before
        row['acquisition_wall_seconds'] = clock() - wall_started
        record['snapshots'].append(row)

    save()
    for target in schedule[1:]:
        before = clock()
        try:
            while record['steps'] < target:
                update = step()
                record['steps'] += 1
                record['samples_seen'] += update['samples']
                record['optimizer_seconds'] += update['optimizer_seconds']
        except FloatingPointError as error:
            record['training_seconds'] += clock() - before
            record.update(status='diverged', failure=str(error), failed_attempt_step=record['steps'] + 1)
            break
        record['training_seconds'] += clock() - before
        try:
            save()
        except FloatingPointError as error:
            record.update(status='diverged', failure=str(error), failed_attempt_step=record['steps'])
            break
    record['acquisition_wall_seconds'] = clock() - wall_started
    return record


def summarize_runs(records, targets=TARGETS, seeds=(8, 9, 10)):
    indexed = {(r['optimizer'], r['seed']): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(m, s) for m in ('adamw', 'muon') for s in seeds}:
        raise ValueError('Expected complete unique paired selected-model records')
    for key in ('protocol_sha256', 'gallery_ids_sha256', 'query_ids_sha256'):
        if len({r[key] for r in records}) != 1:
            raise ValueError('Selected models used different protocols or retrieval images')
    for seed in seeds:
        a, m = [indexed[method, seed] for method in ('adamw', 'muon')]
        for key in ('initial_parameter_sha256', 'train_indices_sha256', 'validation_indices_sha256'):
            if a[key] != m[key]:
                raise ValueError('Selected-model pair differs: ' + key)
        count = min(len(a['epoch_order_sha256']), len(m['epoch_order_sha256']))
        if a['epoch_order_sha256'][:count] != m['epoch_order_sha256'][:count]:
            raise ValueError('Paired training-order prefixes differ')
    result = {}
    for name, consecutive in [('first_observed', 1), ('two_consecutive', 2)]:
        result[name] = []
        for target in targets:
            row = dict(target=target, consecutive=consecutive)
            for method in ('adamw', 'muon'):
                hits = [dict(seed=s, run_status=indexed[method, s]['status'],
                             **first_crossing(indexed[method, s]['curve'], target, consecutive)) for s in seeds]
                complete = all(h['reached'] for h in hits)
                row[method] = dict(reached_count=sum(h['reached'] for h in hits), total_count=len(hits),
                                   per_seed=hits, **{k: distribution([h[k] for h in hits]) if complete else None
                                                   for k in ('steps', 'training_seconds', 'precision')})
            for key, suffix in [('steps', 'steps'), ('training_seconds', 'time')]:
                a, m = [row[method][key] for method in ('adamw', 'muon')]
                row['paired_muon_over_adamw_' + suffix] = (
                    distribution([y / x for x, y in zip(a['values'], m['values'])])
                    if a and m and all(x > 0 for x in a['values']) else None)
            result[name].append(row)
    return result


def check_record(record, protocol, protocol_sha, output):
    planned = protocol['snapshot_steps']
    actual = [r['steps'] for r in record['snapshots']]
    if (record['protocol_sha256'] != protocol_sha or actual != planned[:len(actual)]
            or not actual or record['status'] not in ('complete', 'diverged')
            or record['steps'] > protocol['max_steps']
            or (record['status'] == 'complete' and (actual != planned or record['steps'] != planned[-1]))):
        raise ValueError('Incomplete or mismatched selected-model training record')
    for row in record['snapshots']:
        if digest(output / row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Archived checkpoint changed: ' + row['checkpoint'])


def selected_config(protocol, method, candidate_id):
    if protocol['phase'] == 'final':
        config = protocol['selected'][method]['candidate']
        if config['id'] != candidate_id:
            raise ValueError('Final configuration differs from the locked selection')
        return config
    return next(c for c in protocol['candidate_grid'][method] if c['id'] == candidate_id)


def train_model(args, protocol, protocol_sha):
    import numpy as np
    import needle as ndl
    from needle import nn, optim
    from encoder_checkpoint import save_encoder
    from models import SmallEncoder
    from train_baseline import select_training_indices

    np.random.seed(args.seed)
    rng, device = np.random.default_rng(args.seed), ndl.cpu()
    data = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                               str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    train_ids, val_ids, _ = select_training_indices(len(data), len(data), protocol['train_rows'],
                                                   protocol['validation_rows'], rng,
                                                   protocol['validation_rows'], 17)
    x_train, y_train = data.X[train_ids], data.y[train_ids]
    config = selected_config(protocol, args.optimizer, args.candidate)
    model = SmallEncoder(hidden_dim=protocol['architecture'][1], feature_dim=protocol['architecture'][-2],
                         extra_hidden_layers=2, device=device)
    params = model.parameters()
    matrices = model.muon_parameters() if args.optimizer == 'muon' else []
    matrix_ids = {id(p) for p in matrices}
    optimizer = (optim.MuonWithAuxAdamW(matrices, [p for p in params if id(p) not in matrix_ids],
                                      lr=config['muon_lr'], aux_lr=config['lr'], weight_decay=.01)
                 if matrices else optim.AdamW(params, lr=config['lr'], weight_decay=.01))
    record = dict(optimizer=args.optimizer, seed=args.seed, candidate=config, protocol_sha256=protocol_sha,
                  initial_parameter_sha256=parameter_hash(model),
                  train_indices_sha256=hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
                  validation_indices_sha256=hashlib.sha256(val_ids.astype('int64').tobytes()).hexdigest(),
                  total_parameter_count=sum(math.prod(p.shape) for p in params),
                  muon_parameter_count=sum(math.prod(p.shape) for p in matrices))
    cursor, order, epoch_hashes, order_hash = len(x_train), None, [], hashlib.sha256()
    criterion = nn.SoftmaxLoss()

    def step():
        nonlocal cursor, order
        if cursor == len(x_train):
            order = rng.permutation(len(x_train))
            values = order.astype('int64').tobytes()
            epoch_hashes.append(hashlib.sha256(values).hexdigest())
            order_hash.update(values)
            cursor = 0
        ids = order[cursor:cursor + 512]
        cursor += len(ids)
        x = ndl.Tensor(x_train[ids], device=device, requires_grad=False)
        y = ndl.Tensor(y_train[ids], device=device, requires_grad=False)
        optimizer.reset_grad()
        loss = criterion(model(x), y)
        if not math.isfinite(float(loss.numpy().reshape(-1)[0])):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        before = time.perf_counter()
        optimizer.step()
        return dict(samples=len(ids), optimizer_seconds=time.perf_counter() - before)

    name = f'{args.optimizer}_seed{args.seed}'

    def snapshot(row):
        if any(not np.isfinite(p.numpy()).all() for p in params):
            raise FloatingPointError('Non-finite checkpoint parameters')
        relative = f'{name}/step{row["steps"]:06d}.npz'
        path = args.output_dir / relative
        save_encoder(model, path)
        result = dict(checkpoint=relative, checkpoint_sha256=digest(path), parameter_sha256=parameter_hash(model),
                      training_order_prefix_sha256=order_hash.hexdigest())
        print(json.dumps(dict(optimizer=args.optimizer, seed=args.seed, candidate=config['id'],
                              steps=row['steps'], training_seconds=row['training_seconds'])), flush=True)
        return result

    model.train()
    record.update(train_updates(step, snapshot, protocol['snapshot_steps']))
    record.update(epoch_order_sha256=epoch_hashes, training_order_sha256=order_hash.hexdigest(),
                  optimizer_numeric_state_bytes=optimizer.state_nbytes)
    save_json(args.output_dir / f'{name}_train.json', record)


def evaluate_model(args, protocol, protocol_sha):
    import gc
    import numpy as np
    import needle as ndl
    from benchmark_quantization import unit_rows
    from encoder_checkpoint import load_encoder
    from export_features import retrieval_indices
    from needle.quantization import FloatVectorStore

    name = f'{args.optimizer}_seed{args.seed}'
    training_path = args.output_dir / f'{name}_train.json'
    record = json.loads(training_path.read_text())
    check_record(record, protocol, protocol_sha, args.output_dir)
    data = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                               str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    split = np.random.default_rng(17).permutation(len(data))
    val_ids, pool = split[:protocol['validation_rows']], split[protocol['validation_rows']:]
    if protocol['phase'] == 'tuning':
        query_data, query_pool = data, val_ids
    else:
        query_data = ndl.data.MNISTDataset(str(args.data_dir / 't10k-images-idx3-ubyte.gz'),
                                         str(args.data_dir / 't10k-labels-idx1-ubyte.gz'))
        query_pool = np.arange(len(query_data))
    gallery_pos, query_pos = retrieval_indices(len(pool), len(query_pool), min(4096, protocol['gallery_rows']),
                                               min(256, protocol['query_rows']), protocol['gallery_rows'],
                                               protocol['query_rows'], 7)
    gallery_ids, query_ids = pool[gallery_pos], query_pool[query_pos]
    record.update(gallery_ids_sha256=hashlib.sha256(gallery_ids.astype('int64').tobytes()).hexdigest(),
                  query_ids_sha256=hashlib.sha256(query_ids.astype('int64').tobytes()).hexdigest(),
                  query_source=protocol['evaluation_split'], training_record_sha256=digest(training_path))
    key = 'validation_curve' if protocol['phase'] == 'tuning' else 'curve'
    path, curve = args.output_dir / f'{name}_curve.json', []
    if args.resume and path.exists():
        previous = json.loads(path.read_text())
        if (previous['protocol_sha256'] != protocol_sha
                or previous['training_record_sha256'] != record['training_record_sha256']):
            raise ValueError('Resumed evaluation refers to a different training run')
        curve = previous[key]
        if len(curve) > len(record['snapshots']) or any(
                point['checkpoint_sha256'] != snap['checkpoint_sha256'] or point['steps'] != snap['steps']
                for point, snap in zip(curve, record['snapshots'])):
            raise ValueError('Resumed checkpoint observations differ')
    device = ndl.cpu()
    for snapshot in record['snapshots'][len(curve):]:
        before = time.perf_counter()
        model = load_encoder(args.output_dir / snapshot['checkpoint'], device)

        def extract(dataset, ids):
            parts = []
            for start in range(0, len(ids), 512):
                x = ndl.Tensor(dataset.X[ids[start:start + 512]], device=device, requires_grad=False)
                parts.append(model.features(x).numpy())
            return unit_rows(np.concatenate(parts))

        gallery, queries = extract(data, gallery_ids), extract(query_data, query_ids)
        store = FloatVectorStore(gallery, device=device)
        found = store.search(queries, k=10, block_size=256, metric='cosine')
        counts = (data.y[gallery_ids][found.indices] == query_data.y[query_ids, None]).sum(axis=1)
        curve.append(dict(snapshot, precision=float(counts.sum() / (len(query_ids) * 10)),
                          same_category_matches=int(counts.sum()),
                          per_query_match_histogram=np.bincount(counts, minlength=11).tolist(),
                          neighbor_indices_sha256=hashlib.sha256(found.indices.astype('int64').tobytes()).hexdigest(),
                          evaluation_seconds=time.perf_counter() - before))
        save_json(path, dict(record, **{key: curve}))
        print(json.dumps(dict(optimizer=args.optimizer, seed=args.seed, candidate=args.candidate,
                              steps=snapshot['steps'], precision=curve[-1]['precision'])), flush=True)
        del model, gallery, queries, store, found
        gc.collect()


def phase_protocol(args, phase, selection=None):
    smoke = args.smoke
    points = ([0, 1, 2] if smoke else [0, 108, 216, 324, 540, 864, 1080, 1620]) if phase == 'tuning' else (
        [0, 1, 2, 3] if smoke else final_schedule())
    paths = [*sorted((ROOT / 'python').rglob('*.py')), *sorted((ROOT / 'src').glob('*.h')),
             ROOT / 'src/ndarray_backend_cpu.cc', *sorted((ROOT / 'apps').glob('*.py')),
             Path(__file__), ROOT / 'experiments/common.py', ROOT / 'experiments/time_to_quality.py']
    p = dict(dataset='Fashion-MNIST', study='tuned_batch512', phase=phase, smoke=smoke,
             batch_size=512, split_seed=17, train_rows=512 if smoke else 55000,
             validation_rows=128 if smoke else 5000, gallery_rows=512 if smoke else 55000,
             query_rows=16 if smoke else 1000, training_seeds=[7] if phase == 'tuning' else ([8] if smoke else [8, 9, 10]),
             architecture=[784, 16, 16, 16, 8, 10] if smoke else [784, 128, 128, 128, 64, 10],
             snapshot_steps=points, max_steps=points[-1], targets=list(TARGETS), weight_decay=.01,
             candidate_grid=learning_rate_grid(), offline_quality_workers=3, thread_limits=THREADS,
             selection_rule='max_normalized_trapezoidal_validation_P_at10_over_updates;tie_terminal_precision_then_id',
             evaluation_split='training_holdout' if phase == 'tuning' else 'official_test_exploratory',
             timing_scope='complete_training_loop;excludes_setup_snapshot_IO_offline_evaluation;includes_all_optimizer_work',
             gallery_format='fp32', search_metric='cosine', k=10, block_size=256,
             execution='sequential_training;parallel_offline_quality_after_phase_training',
             no_metric_based_stopping=True, python=sys.version, platform=platform.platform(),
             source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
             data_files_sha256={p.name: digest(p) for p in sorted(args.data_dir.glob('*.gz'))},
             backend=dict(name='native_cpu', dtype='float32', cpu_matmul='strided', tiled_ns=True,
                          extension_sha256=digest(ROOT / 'python/needle/backend_ndarray' /
                              ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX')))))
    if phase == 'final':
        p.update(selected=selection['selected'], selection_sha256=digest(args.output_dir / 'selection.json'),
                 tuning_run_sha256=selection['tuning_run_sha256'])
    return p


def freeze(path, value, resume):
    if path.exists():
        if not resume or json.loads(path.read_text()) != value:
            raise ValueError('Existing phase requires --resume with an identical frozen protocol')
    else:
        save_json(path, value)


def phase_jobs(phase_dir, protocol):
    if protocol['phase'] == 'tuning':
        jobs = []
        for i in range(3):
            for method in (('adamw', 'muon') if i % 2 == 0 else ('muon', 'adamw')):
                config = protocol['candidate_grid'][method][i]
                jobs.append(dict(optimizer=method, seed=7, candidate=config['id'],
                                 output=phase_dir / config['id']))
        return jobs
    return [dict(optimizer=method, seed=seed, candidate=protocol['selected'][method]['candidate']['id'], output=phase_dir)
            for i, seed in enumerate(protocol['training_seeds'])
            for method in (('adamw', 'muon') if i % 2 == 0 else ('muon', 'adamw'))]


def run_phase(args, protocol):
    folder = args.output_dir / protocol['phase']
    frozen = folder / 'protocol.json'
    freeze(frozen, protocol, args.resume)
    stamp = digest(frozen)
    jobs = phase_jobs(folder, protocol)

    def execute(kind, job):
        check_frozen(protocol, args.data_dir)
        job['output'].mkdir(parents=True, exist_ok=True)
        name = f"{job['optimizer']}_seed{job['seed']}"
        record_path = job['output'] / f'{name}_train.json'
        if kind == 'train' and args.resume and record_path.exists():
            record = json.loads(record_path.read_text())
            check_record(record, protocol, stamp, job['output'])
            if record['candidate'] != selected_config(protocol, job['optimizer'], job['candidate']):
                raise ValueError('Resumed training has a different learning rate')
            return
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', kind,
                   '--phase-dir', str(folder), '--optimizer', job['optimizer'], '--seed', str(job['seed']),
                   '--candidate', job['candidate'], '--data-dir', str(args.data_dir),
                   '--output-dir', str(job['output'])]
        if args.resume:
            command.append('--resume')
        with (job['output'] / f'{name}_{kind}.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, **THREADS),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    for job in jobs:
        execute('train', job)
        print('DONE train ' + protocol['phase'] + ' ' + job['candidate'] + f" seed{job['seed']}", flush=True)
    # These workers produce quality only; no retained training timing overlaps them.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, 'evaluate', job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            future.result()
            print('DONE evaluate ' + protocol['phase'] + ' ' + job['candidate'] + f" seed{job['seed']}", flush=True)
    records = []
    for job in jobs:
        name = f"{job['optimizer']}_seed{job['seed']}"
        record = json.loads((job['output'] / f'{name}_curve.json').read_text())
        check_record(record, protocol, stamp, job['output'])
        key = 'validation_curve' if protocol['phase'] == 'tuning' else 'curve'
        if len(record[key]) != len(record['snapshots']):
            raise ValueError('Incomplete offline retrieval curve')
        records.append(record)
    if protocol['phase'] == 'tuning':
        result = dict(protocol=protocol, candidates=records, selected=choose_rates(records))
    else:
        result = dict(protocol=protocol, trajectories=records,
                      summary=summarize_runs(records, protocol['targets'], protocol['training_seeds']))
    save_json(folder / 'run.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=['all', 'tuning', 'final'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/tuned_batch512_20261101')
    parser.add_argument('--worker', choices=['train', 'evaluate'], help=argparse.SUPPRESS)
    parser.add_argument('--phase-dir', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--candidate', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if args.worker:
        frozen = args.phase_dir / 'protocol.json'
        protocol = json.loads(frozen.read_text())
        check_frozen(protocol, args.data_dir)
        selected_config(protocol, args.optimizer, args.candidate)
        if protocol['phase'] == 'final' and digest(args.phase_dir.parent / 'selection.json') != protocol['selection_sha256']:
            raise ValueError('Locked learning-rate selection changed')
        (train_model if args.worker == 'train' else evaluate_model)(args, protocol, digest(frozen))
        return
    tuning_path = args.output_dir / 'tuning/run.json'
    if args.stage != 'final':
        tuning = run_phase(args, phase_protocol(args, 'tuning'))
    else:
        tuning = json.loads(tuning_path.read_text())
        if tuning['protocol'] != phase_protocol(args, 'tuning'):
            raise ValueError('Tuning source or protocol changed before the final study')
    chosen = choose_rates(tuning['candidates'])
    selection_path = args.output_dir / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if (selection['selected'] != chosen or selection['tuning_run_sha256'] != digest(tuning_path)
                or tuning['selected'] != chosen):
            raise ValueError('Locked learning-rate selection differs from validation evidence')
    else:
        selection = dict(selected=chosen, tuning_run_sha256=digest(tuning_path),
                         locked_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(selection_path, selection)
    print('LOCKED ' + json.dumps(selection['selected']), flush=True)
    if args.stage == 'tuning':
        return
    final = run_phase(args, phase_protocol(args, 'final', selection))
    save_json(args.output_dir / 'run.json', dict(tuning=tuning, selection=selection, final=final,
                                               selection_sha256=digest(selection_path),
                                               tuning_run_sha256=digest(tuning_path),
                                               final_run_sha256=digest(args.output_dir / 'final/run.json')))
    print('Saved ' + str(args.output_dir / 'run.json'), flush=True)


if __name__ == '__main__':
    main()
