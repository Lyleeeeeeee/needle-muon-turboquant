"""Measure native FP32 retrieval precision versus actual training steps and seconds.

Train fixed trajectories first, then evaluate every saved snapshot offline. All
thresholds and observation points are frozen before training; no early stopping
or retrieval-based hyperparameter selection is used. Missing crossings remain
censored rather than disappearing from an average.
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
TARGETS = (.80, .85, .87, .88)
THREADS = dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NEEDLE_BACKEND='nd', NEEDLE_CPU_MATMUL='strided')


def checkpoint_schedule(max_steps, early_until=8600, early_interval=430, late_interval=860):
    if any(type(n) is not int or n <= 0
           for n in (max_steps, early_until, early_interval, late_interval)):
        raise ValueError('Schedule counts must be positive integers')
    boundary = min(early_until, max_steps)
    return sorted({0, boundary, max_steps,
                   *range(early_interval, boundary + 1, early_interval),
                   *range(boundary + late_interval, max_steps + 1, late_interval)})


def run_fixed_updates(step, save_snapshot, schedule, clock=time.perf_counter):
    if (not schedule or schedule[0] != 0 or len(schedule) < 2
            or any(type(n) is not int for n in schedule)
            or any(a >= b for a, b in zip(schedule, schedule[1:]))):
        raise ValueError('Expected a strictly increasing schedule beginning at zero')
    record = dict(training_seconds=0., optimizer_seconds=0., checkpoint_seconds=0.,
                  samples_seen=0, steps=0, snapshots=[])
    started = clock()
    for target in schedule:
        if target:
            before = clock()
            while record['steps'] < target:
                update = step()
                record['steps'] += 1
                record['samples_seen'] += update['samples']
                record['optimizer_seconds'] += update['optimizer_seconds']
            record['training_seconds'] += clock() - before
        row = {key: record[key] for key in ('steps', 'samples_seen', 'training_seconds', 'optimizer_seconds')}
        before = clock()
        row.update(save_snapshot(row))
        record['checkpoint_seconds'] += clock() - before
        row['acquisition_wall_seconds'] = clock() - started
        record['snapshots'].append(row)
    record['acquisition_wall_seconds'] = clock() - started
    return record


def first_crossing(curve, target, consecutive=1):
    if not curve or not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('Expected observations and a precision target in [0, 1]')
    if type(consecutive) is not int or consecutive <= 0:
        raise ValueError('Consecutive count must be positive')
    for i, row in enumerate(curve):
        if (type(row['steps']) is not int or row['steps'] < 0
                or not math.isfinite(row['training_seconds']) or row['training_seconds'] < 0
                or not math.isfinite(row['precision']) or not 0 <= row['precision'] <= 1
                or (i and (row['steps'] <= curve[i - 1]['steps']
                           or row['training_seconds'] <= curve[i - 1]['training_seconds']))):
            raise ValueError('Invalid or unordered retrieval observations')
    count = 0
    for i, row in enumerate(curve):
        count = count + 1 if row['precision'] >= target else 0
        if count >= consecutive:
            first = i - consecutive + 1
            previous = curve[max(0, first - 1)]
            return dict(reached=True, steps=row['steps'], training_seconds=row['training_seconds'],
                        precision=row['precision'], first_qualifying_steps=curve[first]['steps'],
                        previous_steps=previous['steps'],
                        previous_training_seconds=previous['training_seconds'],
                        observation_index=i)
    return dict(reached=False, observed_until_steps=curve[-1]['steps'],
                observed_until_training_seconds=curve[-1]['training_seconds'])


def summarize_trajectories(records, targets=TARGETS, seeds=(8, 9, 10)):
    indexed = {(r['optimizer'], r['seed']): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(m, s) for m in ('adamw', 'muon') for s in seeds}:
        raise ValueError('Expected complete unique paired trajectories')
    shared = ('protocol_sha256', 'gallery_ids_sha256', 'query_ids_sha256')
    if any(len({r[key] for r in records}) != 1 for key in shared):
        raise ValueError('Trajectory protocols or retrieval images differ')
    for seed in seeds:
        a, m = [indexed[method, seed] for method in ('adamw', 'muon')]
        for key in ('initial_parameter_sha256', 'train_indices_sha256', 'training_order_sha256'):
            if a[key] != m[key]:
                raise ValueError('Paired trajectory mismatch: ' + key)
        if [r['steps'] for r in a['curve']] != [r['steps'] for r in m['curve']]:
            raise ValueError('Paired observation schedules differ')
    result = {}
    for name, consecutive in [('first_observed', 1), ('two_consecutive', 2)]:
        rows = []
        for target in targets:
            row = dict(target=target, consecutive=consecutive)
            for method in ('adamw', 'muon'):
                hits = [dict(seed=seed, **first_crossing(indexed[method, seed]['curve'], target, consecutive))
                        for seed in seeds]
                complete = all(h['reached'] for h in hits)
                row[method] = dict(reached_count=sum(h['reached'] for h in hits), total_count=len(hits),
                                   per_seed=hits, **{key: distribution([h[key] for h in hits]) if complete else None
                                                   for key in ('steps', 'training_seconds', 'precision')})
            for key, suffix in [('training_seconds', 'time'), ('steps', 'steps')]:
                a, m = [row[method][key] for method in ('adamw', 'muon')]
                row['paired_muon_over_adamw_' + suffix] = (
                    distribution([y / x for x, y in zip(a['values'], m['values'])])
                    if a and m and all(v > 0 for v in a['values']) else None)
            rows.append(row)
        result[name] = rows
    return result


def check_frozen(protocol, data_dir=None):
    if any(digest(ROOT / name) != sha for name, sha in protocol['source_sha256'].items()):
        raise ValueError('Measured source changed after freezing the protocol')
    extension = ROOT / 'python/needle/backend_ndarray' / (
        'ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    if digest(extension) != protocol['backend']['extension_sha256']:
        raise ValueError('Native extension changed after freezing the protocol')
    if data_dir is not None and any(digest(data_dir / name) != sha
                                    for name, sha in protocol['data_files_sha256'].items()):
        raise ValueError('Dataset changed after freezing the protocol')


def check_training(record, protocol, protocol_sha, output):
    if (record['protocol_sha256'] != protocol_sha
            or [s['steps'] for s in record['snapshots']] != protocol['snapshot_steps']
            or record['steps'] != protocol['max_steps']):
        raise ValueError('Incomplete or mismatched training trajectory')
    for row in record['snapshots']:
        if digest(output / row['checkpoint']) != row['checkpoint_sha256']:
            raise ValueError('Archived checkpoint changed: ' + row['checkpoint'])


def parameter_hash(model):
    result = hashlib.sha256()
    for p in model.parameters():
        result.update(str(p.shape).encode())
        result.update(p.numpy().tobytes())
    return result.hexdigest()


def train(args, protocol, protocol_sha):
    import numpy as np
    import needle as ndl
    from needle import nn, optim
    from encoder_checkpoint import save_encoder
    from models import SmallEncoder
    from train_baseline import select_training_indices

    np.random.seed(args.worker_seed)
    rng, device = np.random.default_rng(args.worker_seed), ndl.cpu()
    dataset = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    train_ids, val_ids, _ = select_training_indices(
        len(dataset), len(dataset), protocol['train_rows'], protocol['validation_rows'], rng,
        protocol['validation_rows'], protocol['split_seed'])
    x_train, y_train = dataset.X[train_ids], dataset.y[train_ids]
    model = SmallEncoder(hidden_dim=protocol['architecture'][1],
                         feature_dim=protocol['architecture'][-2], extra_hidden_layers=2, device=device)
    parameters = model.parameters()
    matrices = model.muon_parameters() if args.optimizer == 'muon' else []
    matrix_ids = {id(p) for p in matrices}
    optimizer = (optim.MuonWithAuxAdamW(matrices, [p for p in parameters if id(p) not in matrix_ids],
                                      lr=.007, aux_lr=.001, weight_decay=.01)
                 if matrices else optim.AdamW(parameters, lr=.001, weight_decay=.01))
    criterion = nn.SoftmaxLoss()
    order_hash, epoch_hashes = hashlib.sha256(), []
    cursor, order = len(x_train), None
    name = f'{args.optimizer}_seed{args.worker_seed}'
    record = dict(optimizer=args.optimizer, seed=args.worker_seed, protocol_sha256=protocol_sha,
                  initial_parameter_sha256=parameter_hash(model),
                  train_indices_sha256=hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
                  validation_indices_sha256=hashlib.sha256(val_ids.astype('int64').tobytes()).hexdigest(),
                  total_parameter_count=sum(math.prod(p.shape) for p in parameters),
                  muon_parameter_count=sum(math.prod(p.shape) for p in matrices))

    def step():
        nonlocal cursor, order
        if cursor == len(x_train):
            order = rng.permutation(len(x_train))
            order_hash.update(order.astype('int64').tobytes())
            epoch_hashes.append(hashlib.sha256(order.astype('int64').tobytes()).hexdigest())
            cursor = 0
        ids = order[cursor:cursor + protocol['batch_size']]
        cursor += len(ids)
        x = ndl.Tensor(x_train[ids], device=device, requires_grad=False)
        y = ndl.Tensor(y_train[ids], device=device, requires_grad=False)
        optimizer.reset_grad()
        loss = criterion(model(x), y)
        if not math.isfinite(float(loss.numpy().reshape(-1)[0])):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        before = time.perf_counter()
        optimizer.step()
        return dict(samples=len(ids), optimizer_seconds=time.perf_counter() - before)

    def snapshot(row):
        relative = f'{name}/step{row["steps"]:06d}.npz'
        path = args.output_dir / relative
        save_encoder(model, path)
        extra = dict(checkpoint=relative, checkpoint_sha256=digest(path),
                     parameter_sha256=parameter_hash(model),
                     training_order_prefix_sha256=order_hash.hexdigest())
        reference = protocol.get('reference_checkpoints', {}).get(name)
        if row['steps'] == 8600 and reference:
            previous = Path(reference['path'])
            if digest(previous) != reference['sha256']:
                raise ValueError('Reference checkpoint changed')
            with np.load(previous, allow_pickle=False) as a, np.load(path, allow_pickle=False) as b:
                if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
                    raise ValueError('Trajectory differs from previous 8600-step terminal checkpoint')
            extra['reference_terminal_arrays_equal'] = True
        print(json.dumps(dict(optimizer=args.optimizer, seed=args.worker_seed,
                              steps=row['steps'], training_seconds=row['training_seconds'])), flush=True)
        return extra

    model.train()
    record.update(run_fixed_updates(step, snapshot, protocol['snapshot_steps']))
    record.update(training_order_sha256=order_hash.hexdigest(), epoch_order_sha256=epoch_hashes,
                  optimizer_numeric_state_bytes=optimizer.state_nbytes)
    save_json(args.output_dir / f'{name}_train.json', record)


def evaluate(args, protocol, protocol_sha):
    import gc
    import numpy as np
    import needle as ndl
    from benchmark_quantization import unit_rows
    from encoder_checkpoint import load_encoder
    from export_features import retrieval_indices
    from needle.quantization import FloatVectorStore

    name = f'{args.optimizer}_seed{args.worker_seed}'
    record = json.loads((args.output_dir / f'{name}_train.json').read_text())
    check_training(record, protocol, protocol_sha, args.output_dir)
    dataset = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(args.data_dir / 't10k-images-idx3-ubyte.gz'),
                                 str(args.data_dir / 't10k-labels-idx1-ubyte.gz'))
    pool = np.random.default_rng(protocol['split_seed']).permutation(len(dataset))[protocol['validation_rows']:]
    gallery_positions, query_ids = retrieval_indices(
        len(pool), len(test), min(4096, protocol['gallery_rows']), min(256, protocol['query_rows']),
        protocol['gallery_rows'], protocol['query_rows'], 7)
    gallery_ids = pool[gallery_positions]
    record.update(gallery_ids_sha256=hashlib.sha256(gallery_ids.astype('int64').tobytes()).hexdigest(),
                  query_ids_sha256=hashlib.sha256(query_ids.astype('int64').tobytes()).hexdigest())
    curve = []
    curve_path = args.output_dir / f'{name}_curve.json'
    if args.resume and curve_path.exists():
        previous = json.loads(curve_path.read_text())
        if (previous['protocol_sha256'] != protocol_sha
                or previous['training_record_sha256'] != digest(args.output_dir / f'{name}_train.json')):
            raise ValueError('Curve resume provenance differs')
        curve = previous['curve']
        for point, snapshot in zip(curve, record['snapshots']):
            if point['checkpoint_sha256'] != snapshot['checkpoint_sha256'] or point['steps'] != snapshot['steps']:
                raise ValueError('Curve resume checkpoint differs')
        if len(curve) > len(record['snapshots']):
            raise ValueError('Too many saved curve observations')
    record['training_record_sha256'] = digest(args.output_dir / f'{name}_train.json')
    device = ndl.cpu()
    for snapshot in record['snapshots'][len(curve):]:
        before = time.perf_counter()
        model = load_encoder(args.output_dir / snapshot['checkpoint'], device)

        def extract(data, ids):
            batches = []
            for start in range(0, len(ids), protocol['batch_size']):
                x = ndl.Tensor(data.X[ids[start:start + protocol['batch_size']]],
                               device=device, requires_grad=False)
                batches.append(model.features(x).numpy())
            return unit_rows(np.concatenate(batches))

        gallery, queries = extract(dataset, gallery_ids), extract(test, query_ids)
        store = FloatVectorStore(gallery, device=device)
        found = store.search(queries, k=10, block_size=256, metric='cosine')
        counts = (dataset.y[gallery_ids][found.indices] == test.y[query_ids, None]).sum(axis=1)
        point = dict(snapshot, same_category_matches=int(counts.sum()),
                     precision=float(counts.sum() / (len(query_ids) * 10)),
                     per_query_match_histogram=np.bincount(counts, minlength=11).tolist(),
                     neighbor_indices_sha256=hashlib.sha256(found.indices.astype('int64').tobytes()).hexdigest(),
                     evaluation_seconds=time.perf_counter() - before)
        curve.append(point)
        save_json(curve_path, dict(record, curve=curve))
        print(json.dumps(dict(optimizer=args.optimizer, seed=args.worker_seed, steps=point['steps'],
                              training_seconds=point['training_seconds'], precision=point['precision'])), flush=True)
        del model, gallery, queries, store, found
        gc.collect()


def make_protocol(args):
    smoke = args.smoke
    steps = checkpoint_schedule(4, early_until=4, early_interval=2, late_interval=2) if smoke else checkpoint_schedule(43000)
    paths = [*sorted((ROOT / 'python').rglob('*.py')), *sorted((ROOT / 'src').glob('*.h')),
             ROOT / 'src/ndarray_backend_cpu.cc', *sorted((ROOT / 'apps').glob('*.py')),
             Path(__file__), ROOT / 'experiments/common.py']
    reference = {}
    if args.reference_study:
        for seed in (8, 9, 10):
            for method in ('adamw', 'muon'):
                path = args.reference_study / f'{method}_seed{seed}_terminal.npz'
                reference[f'{method}_seed{seed}'] = dict(path=str(path.resolve()), sha256=digest(path))
    return dict(dataset='Fashion-MNIST', study='time_to_retrieval_quality', smoke=smoke,
                training_seeds=[8] if smoke else [8, 9, 10],
                targets=list(TARGETS), snapshot_steps=steps, max_steps=steps[-1],
                train_rows=64 if smoke else 55000, validation_rows=16 if smoke else 5000,
                gallery_rows=64 if smoke else 55000, query_rows=16 if smoke else 1000,
                architecture=[784, 16, 16, 16, 8, 10] if smoke else [784, 128, 128, 128, 64, 10],
                batch_size=64, split_seed=17, rates=dict(adamw_auxiliary=.001, muon=.007), weight_decay=.01,
                gallery_format='fp32', search_metric='cosine', k=10, block_size=256,
                checkpoint_rule='first_observed_P_at10_ge_target;two_consecutive_sensitivity_uses_confirmation_snapshot',
                timing_scope='training_loop_including_data_and_forward_backward_update;excludes_snapshot_io_and_offline_evaluation',
                evaluation_scope='same_previously_explored_official_test_queries;exploratory_not_untouched_test',
                no_metric_based_stopping=True, reference_checkpoints=reference,
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                data_files_sha256={p.name: digest(p) for p in sorted(args.data_dir.glob('*.gz'))},
                backend=dict(name='native_cpu', dtype='float32', cpu_matmul='strided', tiled_ns=True,
                             extension_sha256=digest(ROOT / 'python/needle/backend_ndarray' /
                                 ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX')))),
                python=sys.version, platform=platform.platform())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate'], default='all')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/time_to_quality_20261101')
    parser.add_argument('--reference-study', type=Path, help='Verify 8600-step parameters against a prior epoch run')
    parser.add_argument('--worker', choices=['train', 'evaluate'], help=argparse.SUPPRESS)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], help=argparse.SUPPRESS)
    parser.add_argument('--worker-seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    frozen = args.output_dir / 'protocol.json'
    if args.worker:
        protocol = json.loads(frozen.read_text())
        check_frozen(protocol, args.data_dir)
        if args.worker == 'train': train(args, protocol, digest(frozen))
        else: evaluate(args, protocol, digest(frozen))
        return
    protocol = make_protocol(args)
    if frozen.exists():
        if not args.resume:
            parser.error('Run already exists; use a new output directory or --resume')
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError('Resume protocol differs from the frozen experiment')
    else:
        save_json(frozen, protocol)
    stages = ['train', 'evaluate'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        for i, seed in enumerate(protocol['training_seeds']):
            for method in (('adamw', 'muon') if i % 2 == 0 else ('muon', 'adamw')):
                check_frozen(protocol, args.data_dir)
                name = f'{method}_seed{seed}'
                training_path = args.output_dir / f'{name}_train.json'
                if stage == 'train' and args.resume and training_path.exists():
                    check_training(json.loads(training_path.read_text()), protocol, digest(frozen), args.output_dir)
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', stage,
                           '--optimizer', method, '--worker-seed', str(seed), '--data-dir', str(args.data_dir),
                           '--output-dir', str(args.output_dir)]
                if args.resume: command.append('--resume')
                with (args.output_dir / f'{name}_{stage}.log').open('a') as log:
                    subprocess.run(command, cwd=ROOT, env=dict(os.environ, **THREADS),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                print('DONE ' + stage + ' ' + name, flush=True)
    if args.stage != 'train':
        records = [json.loads((args.output_dir / f'{method}_seed{seed}_curve.json').read_text())
                   for seed in protocol['training_seeds'] for method in ('adamw', 'muon')]
        for record in records:
            check_training(record, protocol, digest(frozen), args.output_dir)
            if len(record['curve']) != len(protocol['snapshot_steps']):
                raise ValueError('Incomplete retrieval trajectory')
        summary = summarize_trajectories(records, protocol['targets'], protocol['training_seeds'])
        save_json(args.output_dir / 'run.json', dict(protocol=protocol, trajectories=records, summary=summary))
        print('Saved ' + str(args.output_dir / 'run.json'), flush=True)


if __name__ == '__main__':
    main()
