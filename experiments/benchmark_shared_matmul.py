"""Same-binary ablation of public CPU Matmul packing, with native Muon NS enabled."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmark_compact_optimization import digest, save_json, summarize_pairs
from benchmark_muon_tiled import parameter_hash

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/shared_matmul'
ARTIFACTS = ROOT / 'results/shared_matmul'
BASELINE = 'ac6dca9b34b60d968558e0978667b69a33b0949e'
SOURCES = ['src/ndarray_backend_cpu.cc', 'src/matmul_cpu.h', 'src/muon_cpu.h',
           'src/quantization_cpu.h', 'python/needle/backend_ndarray/ndarray.py',
           'python/needle/autograd.py', 'python/needle/optim.py',
           'python/needle/ops/ops_mathematic.py', 'apps/models.py', 'apps/train_baseline.py',
           'experiments/profile_optimizer.py', 'experiments/benchmark_shared_matmul.py',
           'experiments/benchmark_muon_tiled.py', 'experiments/benchmark_compact_optimization.py',
           'tools/build_cpu.py']


def audit_step(step, device):
    """One untimed training update: distinguish generic copies and internal packing.

    Copy writes/passes are derived from operator dimensions, not DRAM counters.
    Matmul output zeroing/arithmetic and NS-internal layout writes are excluded.
    """
    counts = {'compact_calls': 0, 'compact_written_elements': 0,
              'strided_calls': 0, 'strided_layout_passes': 0,
              'strided_layout_written_elements': 0, 'tiled_calls': 0,
              'naive_calls': 0, 'native_ns_calls': 0}
    with ExitStack() as restore:
        for name in ('compact', 'matmul_strided', 'matmul_tiled', 'matmul', 'muon_ns_tiled'):
            original = getattr(device.mod, name)

            def record(*args, name=name, original=original):
                if name == 'compact':
                    counts['compact_calls'] += 1
                    counts['compact_written_elements'] += args[1].size
                elif name == 'matmul_strided':
                    m, n, p = args[3:6]
                    counts['strided_calls'] += 1
                    counts['strided_layout_passes'] += 3
                    counts['strided_layout_written_elements'] += m * n + n * p + m * p
                elif name == 'matmul_tiled': counts['tiled_calls'] += 1
                elif name == 'matmul': counts['naive_calls'] += 1
                else: counts['native_ns_calls'] += 1
                return original(*args)

            setattr(device.mod, name, record)
            restore.callback(setattr, device.mod, name, original)
        step()
    counts['public_products'] = counts['strided_calls'] + counts['tiled_calls'] + counts['naive_calls']
    counts['layout_passes_excluding_native_ns'] = counts['compact_calls'] + counts['strided_layout_passes']
    counts['layout_written_bytes_excluding_native_ns'] = 4 * (
        counts['compact_written_elements'] + counts['strided_layout_written_elements'])
    return counts


def worker(args):
    os.environ['NEEDLE_CPU_MATMUL'] = 'strided' if args.worker == 'new' else 'reference'
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn, optim
    from needle.backend_ndarray import NDArray
    from models import SmallEncoder
    from train_baseline import select_training_indices
    from profile_optimizer import make_fixture

    device = ndl.cpu()
    metrics, diagnostics = {}, {}
    cases = [('forward_input', (64, 784), (784, 128), False, False),
             ('backward_weight', (64, 784), (64, 128), True, False),
             ('backward_activation', (64, 128), (784, 128), False, True),
             ('hidden_square', (128, 128), (128, 128), False, False),
             ('head_fallback', (64, 64), (64, 10), False, False)]
    for name, a_shape, b_shape, transpose_a, transpose_b in cases:
        rng = np.random.default_rng(23)
        av, bv = rng.normal(size=a_shape).astype('float32'), rng.normal(size=b_shape).astype('float32')
        a, b = NDArray(av, device=device), NDArray(bv, device=device)
        if transpose_a: a = a.permute((1, 0))
        if transpose_b: b = b.permute((1, 0))
        for _ in range(10): output = a @ b
        started = time.perf_counter()
        for _ in range(200): output = a @ b
        elapsed = time.perf_counter() - started
        metrics['matmul_' + name] = {
            'seconds': elapsed / 200, 'calls': 200, 'a_shape': a.shape, 'b_shape': b.shape,
            'a_strides': a.strides, 'b_strides': b.strides,
            'input_sha256': hashlib.sha256(av.tobytes() + bv.tobytes()).hexdigest(),
            'output_sha256': hashlib.sha256(output.numpy().tobytes()).hexdigest()}
    # These optimizer-only controls should be unchanged: neither uses public Matmul.
    for name in ('adamw', 'muon'):
        fixture = make_fixture(name, use_tiled_ns=True)
        for _ in range(10): fixture['optimizer'].step()
        started = time.perf_counter()
        for _ in range(80): fixture['optimizer'].step()
        elapsed = time.perf_counter() - started
        metrics['optimizer_' + name] = {
            'seconds': elapsed / 80, 'calls': 80, 'use_tiled_ns': True,
            'gradient_sha256': fixture['gradient_sha256'],
            'initial_parameter_sha256': fixture['initial_parameter_sha256'],
            'output_sha256': parameter_hash(fixture['parameters'])}
    data_dir = ROOT / 'data/fashion-mnist'
    dataset = ndl.data.MNISTDataset(str(data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(data_dir / 'train-labels-idx1-ubyte.gz'))
    for name in ('adamw', 'muon'):
        np.random.seed(8)
        rng = np.random.default_rng(8)
        train_ids, val_ids, _ = select_training_indices(len(dataset), len(dataset), 55000, 128,
                                                       rng, 5000, 17)
        x_train, y_train = dataset.X[train_ids], dataset.y[train_ids]
        model = SmallEncoder(extra_hidden_layers=2, device=device)
        parameters = model.parameters()
        initial_hash = parameter_hash(parameters)
        if name == 'muon':
            matrices = model.muon_parameters()
            ids = {id(p) for p in matrices}
            optimizer = optim.MuonWithAuxAdamW(matrices, [p for p in parameters if id(p) not in ids],
                                              lr=.007, aux_lr=.001, weight_decay=.01,
                                              use_tiled_ns=True)
        else:
            optimizer = optim.AdamW(parameters, lr=.001, weight_decay=.01)
        criterion, losses = nn.SoftmaxLoss(), []
        order = rng.permutation(len(x_train))

        def update(step):
            batch = order[step * 64:(step + 1) * 64]
            x = ndl.Tensor(x_train[batch], device=device, requires_grad=False)
            y = ndl.Tensor(y_train[batch], device=device, dtype='float32', requires_grad=False)
            optimizer.reset_grad()
            loss = criterion(model(x), y)
            losses.append(float(loss.numpy().reshape(-1)[0]))
            loss.backward()
            optimizer.step()

        started = time.perf_counter()
        for step in range(128): update(step)
        elapsed = time.perf_counter() - started
        loss_bytes = np.asarray(losses, dtype='float32').tobytes()
        features = model.features(ndl.Tensor(dataset.X[val_ids], device=device, requires_grad=False)).numpy()
        param_hash = parameter_hash(parameters)
        fingerprint = hashlib.sha256(bytes.fromhex(param_hash) + loss_bytes + features.tobytes()).hexdigest()
        metrics['training_' + name] = {
            'seconds': elapsed / 128, 'calls': 128, 'seed': 8, 'use_tiled_ns': True,
            'initial_parameter_sha256': initial_hash, 'parameter_sha256': param_hash,
            'train_indices_sha256': hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
            'training_loss_sha256': hashlib.sha256(loss_bytes).hexdigest(),
            'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
            'output_sha256': fingerprint}
        diagnostics[name] = audit_step(lambda: update(128), device)
        diagnostics[name]['post_parameters_sha256'] = parameter_hash(parameters)
    save_json(args.output, {'variant': args.worker, 'block': args.block,
                           'matmul_mode': os.environ['NEEDLE_CPU_MATMUL'], 'use_tiled_ns': True,
                           'native_extension_sha256': digest(device.mod.__file__),
                           'metrics': metrics, 'diagnostics': diagnostics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['old', 'new'])
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--blocks', type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.blocks < 2: parser.error('At least two paired timing blocks required')
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/build_cpu.py')], cwd=ROOT, check=True)
    data_dir = ROOT / 'data/fashion-mnist'
    protocol = {'date': '2026-11-01', 'baseline_commit': BASELINE, 'blocks': args.blocks,
                'comparison': 'same final binary;NEEDLE_CPU_MATMUL=reference versus strided;native NS on in both',
                'reference': 'old', 'strided': 'new',
                'order': 'old,new on even blocks;new,old on odd blocks;sequential processes',
                'source_sha256': {p: digest(ROOT / p) for p in SOURCES},
                'data_files_sha256': {name: digest(data_dir / name) for name in
                                      ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')},
                'platform': platform.platform(), 'python': sys.version,
                'matmul_warmup_calls': 10, 'matmul_calls': 200,
                'optimizer_warmup_calls': 10, 'optimizer_calls': 80,
                'training_updates': 128, 'training_seed': 8, 'batch_size': 64, 'split_seed': 17,
                'architecture': [784, 128, 128, 128, 64, 10],
                'rates': {'adamw_auxiliary': .001, 'muon': .007},
                'timed_training': 'full loop incl batch/Tensor preparation,loss read,forward,backward,update;excl setup/export/audit',
                'scope': 'execution speed and exact output parity;not new model-quality evidence'}
    save_json(REPORTS / 'protocol.json', protocol)
    rows = []
    for block in range(args.blocks):
        for variant in (('old', 'new') if block % 2 == 0 else ('new', 'old')):
            assert {p: digest(ROOT / p) for p in SOURCES} == protocol['source_sha256']
            name = f'block{block}_{variant}'
            output = REPORTS / f'{name}.json'
            with (ARTIFACTS / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '--worker', variant,
                                '--block', str(block), '--output', str(output)], cwd=ROOT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                        VECLIB_MAXIMUM_THREADS='1'), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            rows.append(json.loads(output.read_text()))
            print('DONE ' + name, flush=True)
    assert len({r['native_extension_sha256'] for r in rows}) == 1
    summary = summarize_pairs(rows)
    for name in ('adamw', 'muon'):
        for block in range(args.blocks):
            pair = [r for r in rows if r['block'] == block]
            assert pair[0]['diagnostics'][name]['post_parameters_sha256'] == pair[1]['diagnostics'][name]['post_parameters_sha256']
    save_json(REPORTS / 'summary.json', {'protocol': protocol, 'summary': summary,
                                       'raw_report_sha256': {f'block{r["block"]}_{r["variant"]}.json':
                                                            digest(REPORTS / f'block{r["block"]}_{r["variant"]}.json')
                                                            for r in rows}})
    for key, values in summary.items():
        print(f"{key}: {1000*values['old_seconds']['median']:.4f} -> "
              f"{1000*values['new_seconds']['median']:.4f} ms; "
              f"paired speedup={values['speedup']['median']:.3f}x", flush=True)


if __name__ == '__main__':
    main()
