"""Sequential fixed-image AdamW/Muon timing across batch sizes, not quality."""
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
from time_to_quality import THREADS, check_frozen, parameter_hash

ROOT = Path(__file__).resolve().parents[1]
BATCHES = (64, 256, 512)


def time_updates(update, batch_size, measured_samples, warmup_samples,
                 clock=time.perf_counter, on_warmup=None):
    if (any(type(n) is not int or n <= 0 for n in (batch_size, measured_samples, warmup_samples))
            or measured_samples % batch_size or warmup_samples % batch_size):
        raise ValueError('Full batches must divide the positive warmup and measured image counts')
    for start in range(0, warmup_samples, batch_size):
        update(start, batch_size)
    if on_warmup is not None:
        on_warmup()
    optimizer_seconds = 0.
    before = clock()
    for start in range(warmup_samples, warmup_samples + measured_samples, batch_size):
        optimizer_seconds += update(start, batch_size)
    seconds = clock() - before
    return dict(batch_size=batch_size, measured_samples=measured_samples, warmup_samples=warmup_samples,
                updates=measured_samples // batch_size, warmup_updates=warmup_samples // batch_size,
                training_seconds=seconds, optimizer_seconds=optimizer_seconds,
                training_us_per_image=1e6 * seconds / measured_samples)


def summarize_batches(rows, blocks=5, batches=BATCHES):
    keys = {(r['block'], r['optimizer'], r['batch_size']): r for r in rows}
    expected = {(block, method, batch) for block in range(blocks)
                for method in ('adamw', 'muon') for batch in batches}
    if len(keys) != len(rows) or set(keys) != expected:
        raise ValueError('Expected complete unique paired batch measurements')
    for key in ('initial_parameter_sha256', 'train_indices_sha256', 'warmup_image_ids_sha256',
                'measured_image_ids_sha256', 'native_extension_sha256', 'protocol_sha256',
                'measured_samples', 'warmup_samples'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError('Batch measurements differ in shared input: ' + key)
    for r in rows:
        if (r['updates'] * r['batch_size'] != r['measured_samples']
                or r['warmup_updates'] * r['batch_size'] != r['warmup_samples']
                or not math.isfinite(r['training_seconds']) or r['training_seconds'] <= 0
                or not math.isfinite(r['optimizer_seconds'])
                or not 0 <= r['optimizer_seconds'] <= r['training_seconds']
                or not math.isclose(r['training_us_per_image'],
                                    1e6 * r['training_seconds'] / r['measured_samples'], rel_tol=1e-12)):
            raise ValueError('Invalid timing, denominator or measured work')
    summary = {}
    for batch in batches:
        result = {}
        for method in ('adamw', 'muon'):
            group = [keys[block, method, batch] for block in range(blocks)]
            for field in ('final_parameter_sha256', 'training_loss_sha256', 'post_warm_parameter_sha256'):
                if len({r[field] for r in group}) != 1:
                    raise ValueError('Timing repetitions changed arithmetic output: ' + field)
            result[method] = {
                'training_us_per_image': distribution([r['training_us_per_image'] for r in group]),
                'training_ms_per_update': distribution([1e3 * r['training_seconds'] / r['updates'] for r in group]),
                'optimizer_us_per_image': distribution([1e6 * r['optimizer_seconds'] / r['measured_samples'] for r in group]),
                'optimizer_ms_per_update': distribution([1e3 * r['optimizer_seconds'] / r['updates'] for r in group]),
                'other_us_per_image': distribution([1e6 * (r['training_seconds'] - r['optimizer_seconds'])
                                                    / r['measured_samples'] for r in group]),
                'updates': group[0]['updates'], 'measured_samples': group[0]['measured_samples']}
        a, m = [result[method]['training_us_per_image']['values'] for method in ('adamw', 'muon')]
        result['paired_muon_over_adamw'] = distribution([y / x for x, y in zip(a, m)])
        result['paired_extra_us_per_image'] = distribution([y - x for x, y in zip(a, m)])
        summary[str(batch)] = result
    return summary


def worker(args, protocol, protocol_sha):
    import numpy as np
    import needle as ndl
    from needle import nn, optim
    from models import SmallEncoder
    from train_baseline import select_training_indices

    np.random.seed(8)
    rng, device = np.random.default_rng(8), ndl.cpu()
    dataset = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    train_ids, _, _ = select_training_indices(len(dataset), len(dataset), protocol['train_rows'],
                                             protocol['validation_rows'], rng,
                                             protocol['validation_rows'], 17)
    x_train, y_train = dataset.X[train_ids], dataset.y[train_ids]
    model = SmallEncoder(hidden_dim=protocol['architecture'][1],
                         feature_dim=protocol['architecture'][-2], extra_hidden_layers=2, device=device)
    parameters = model.parameters()
    matrices = model.muon_parameters() if args.optimizer == 'muon' else []
    ids = {id(p) for p in matrices}
    optimizer = (optim.MuonWithAuxAdamW(matrices, [p for p in parameters if id(p) not in ids],
                                      lr=.007, aux_lr=.001, weight_decay=.01)
                 if matrices else optim.AdamW(parameters, lr=.001, weight_decay=.01))
    initial = parameter_hash(model)
    order = rng.permutation(len(x_train))
    warmup, samples = protocol['warmup_samples'], protocol['measured_samples']
    if warmup + samples > len(order):
        raise ValueError('The fixed image window does not fit the training pool')
    criterion, losses, warm_hash = nn.SoftmaxLoss(), [], []

    def update(start, count):
        selected = order[start:start + count]
        x = ndl.Tensor(x_train[selected], device=device, requires_grad=False)
        y = ndl.Tensor(y_train[selected], device=device, requires_grad=False)
        optimizer.reset_grad()
        loss = criterion(model(x), y)
        value = float(loss.numpy().reshape(-1)[0])
        if not math.isfinite(value):
            raise RuntimeError('Non-finite training loss')
        losses.append(value)
        loss.backward()
        before = time.perf_counter()
        optimizer.step()
        return time.perf_counter() - before

    model.train()
    measured = time_updates(update, args.batch_size, samples, warmup,
                            on_warmup=lambda: warm_hash.append(parameter_hash(model)))
    record = dict(measured, block=args.block, optimizer=args.optimizer, protocol_sha256=protocol_sha,
                  native_extension_sha256=digest(device.mod.__file__), initial_parameter_sha256=initial,
                  post_warm_parameter_sha256=warm_hash[0], final_parameter_sha256=parameter_hash(model),
                  training_loss_sha256=hashlib.sha256(np.asarray(
                      losses[measured['warmup_updates']:], dtype='float32').tobytes()).hexdigest(),
                  train_indices_sha256=hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
                  warmup_image_ids_sha256=hashlib.sha256(train_ids[order[:warmup]].astype('int64').tobytes()).hexdigest(),
                  measured_image_ids_sha256=hashlib.sha256(
                      train_ids[order[warmup:warmup + samples]].astype('int64').tobytes()).hexdigest(),
                  total_parameter_count=sum(math.prod(p.shape) for p in parameters),
                  muon_parameter_count=sum(math.prod(p.shape) for p in matrices),
                  optimizer_numeric_state_bytes=optimizer.state_nbytes)
    save_json(args.output_dir / f'block{args.block}_{args.optimizer}_batch{args.batch_size}.json', record)


def make_protocol(args):
    paths = [*sorted((ROOT / 'python').rglob('*.py')), *sorted((ROOT / 'src').glob('*.h')),
             ROOT / 'src/ndarray_backend_cpu.cc', *sorted((ROOT / 'apps').glob('*.py')),
             Path(__file__), ROOT / 'experiments/common.py', ROOT / 'experiments/time_to_quality.py']
    smoke = args.smoke
    return dict(dataset='Fashion-MNIST', study='batch_training_cost', smoke=smoke,
                blocks=2 if smoke else 5, batch_sizes=list(BATCHES), seed=8, split_seed=17,
                train_rows=2048 if smoke else 55000, validation_rows=128 if smoke else 5000,
                measured_samples=1024 if smoke else 32768, warmup_samples=512 if smoke else 2048,
                architecture=[784, 16, 16, 16, 8, 10] if smoke else [784, 128, 128, 128, 64, 10],
                rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                timing_scope='complete_training_loop_including_data_Tensor_loss_forward_backward_optimizer;excludes_setup_warmup_hashes',
                scope='timing_repetitions_at_fixed_seed;not_convergence_or_retrieval_quality',
                execution='sequential_isolated_workers;rotated_reversed_batches;alternating_optimizer_order',
                thread_limits=THREADS, python=sys.version, platform=platform.platform(),
                backend=dict(name='native_cpu', dtype='float32', cpu_matmul='strided', tiled_ns=True,
                             extension_sha256=digest(ROOT / 'python/needle/backend_ndarray' /
                                 ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX')))),
                source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
                data_files_sha256={name: digest(args.data_dir / name) for name in
                                   ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/batch_cost_20261101')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--block', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--batch-size', type=int, choices=BATCHES, help=argparse.SUPPRESS)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    frozen = args.output_dir / 'protocol.json'
    if args.worker:
        protocol = json.loads(frozen.read_text())
        check_frozen(protocol, args.data_dir)
        worker(args, protocol, digest(frozen))
        return
    protocol = make_protocol(args)
    if frozen.exists():
        if not args.resume or json.loads(frozen.read_text()) != protocol:
            parser.error('Existing run requires --resume and identical frozen protocol')
    else:
        save_json(frozen, protocol)
    rows = []
    for block in range(protocol['blocks']):
        batches = list(BATCHES[block % 3:] + BATCHES[:block % 3])
        if block % 2: batches.reverse()
        for position, batch in enumerate(batches):
            for method in (('adamw', 'muon') if (block + position) % 2 == 0 else ('muon', 'adamw')):
                check_frozen(protocol, args.data_dir)
                name = f'block{block}_{method}_batch{batch}'
                path = args.output_dir / f'{name}.json'
                if not (args.resume and path.exists()):
                    with (args.output_dir / f'{name}.log').open('w') as log:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker',
                                        '--optimizer', method, '--batch-size', str(batch), '--block', str(block),
                                        '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir)],
                                       cwd=ROOT, env=dict(os.environ, **THREADS), stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                row = json.loads(path.read_text())
                if (row['protocol_sha256'] != digest(frozen) or row['optimizer'] != method
                        or row['batch_size'] != batch or row['block'] != block):
                    raise ValueError('Worker resume provenance differs')
                rows.append(row)
                print('DONE ' + name, flush=True)
    save_json(args.output_dir / 'run.json', dict(protocol=protocol, runs=rows,
                                               summary=summarize_batches(rows, protocol['blocks'])))
    print('Saved ' + str(args.output_dir / 'run.json'), flush=True)


if __name__ == '__main__':
    main()
