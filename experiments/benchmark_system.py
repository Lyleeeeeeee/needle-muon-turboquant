"""Paired original-versus-final CPU execution, with unchanged training arithmetic."""
import argparse
import importlib.util
import hashlib
import time
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from common import digest, save_json, summarize_pairs, parameter_hash

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'efe063d3b67153ced96c8435a381866ee0077605'
ARTIFACTS = ROOT / 'results/system'
SOURCES = ['src/ndarray_backend_cpu.cc', 'src/matmul_cpu.h', 'src/muon_cpu.h',
           'src/quantization_cpu.h', 'python/needle/backend_ndarray/ndarray.py',
           'python/needle/autograd.py', 'python/needle/optim.py',
           'python/needle/ops/ops_mathematic.py', 'apps/models.py', 'apps/train_baseline.py',
           'experiments/common.py', 'experiments/profile_optimizer.py',
           'experiments/benchmark_system.py', 'tools/build_cpu.py']


def validate_measurements(rows):
    summary = summarize_pairs(rows)
    binary_sets = {v: {r['native_extension_sha256'] for r in rows if r['variant'] == v}
                   for v in ('old', 'new')}
    if any(len(s) != 1 for s in binary_sets.values()) or binary_sets['old'] == binary_sets['new']:
        raise ValueError('Expected one distinct native binary per version')
    for record in rows:
        expected = record['variant'] == 'new'
        if record['capabilities'] != dict(muon_ns_tiled=expected, matmul_strided=expected):
            raise ValueError('Native dispatch capabilities differ from the requested version')
    for metric in summary:
        keys = [k for k in rows[0]['metrics'][metric] if k.endswith('sha256')]
        for key in keys:
            if len({r['metrics'][metric][key] for r in rows}) != 1:
                raise ValueError('Paired input/output mismatch: ' + key)
    return summary


def prepare_baseline():
    """Compile the original kernel from its pinned Git revision; share Python code."""
    target = ARTIFACTS / 'baseline'
    for relative in ('src/ndarray_backend_cpu.cc', 'src/quantization_cpu.h'):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(subprocess.check_output(['git', 'show', f'{BASELINE}:{relative}'], cwd=ROOT))
    assert digest(target / 'src/quantization_cpu.h') == digest(ROOT / 'src/quantization_cpu.h')
    shutil.copytree(ROOT / 'python', target / 'python', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.dylib'))
    spec = importlib.util.spec_from_file_location('system_build_helper', ROOT / 'tools/build_cpu.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.ROOT = target
    builder.main()
    subprocess.run([sys.executable, str(ROOT / 'tools/build_cpu.py')], cwd=ROOT, check=True)
    return target


def worker(args):
    os.environ['NEEDLE_CPU_MATMUL'] = 'strided' if args.worker == 'new' else 'reference'
    import numpy as np
    sys.path[:0] = [str(args.python_root), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn, optim
    from needle.backend_ndarray import NDArray
    from models import SmallEncoder
    from train_baseline import select_training_indices
    from profile_optimizer import make_fixture

    device = ndl.cpu()
    assert Path(device.mod.__file__).resolve().is_relative_to(args.python_root.resolve())
    metrics = {}
    # Fixed-gradient updates isolate optimizer execution from forward/backward.
    for name in ('adamw', 'muon'):
        fixture = make_fixture(name, use_tiled_ns=True)
        for _ in range(10): fixture['optimizer'].step()
        started = time.perf_counter()
        for _ in range(80): fixture['optimizer'].step()
        elapsed = time.perf_counter() - started
        metrics['optimizer_' + name] = {
            'seconds': elapsed / 80, 'calls': 80, 'request_tiled_ns': True,
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
            'seconds': elapsed / 128, 'calls': 128, 'seed': 8, 'request_tiled_ns': True,
            'initial_parameter_sha256': initial_hash, 'parameter_sha256': param_hash,
            'train_indices_sha256': hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
            'training_loss_sha256': hashlib.sha256(loss_bytes).hexdigest(),
            'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
            'output_sha256': fingerprint}
    save_json(args.output, {'variant': args.worker, 'block': args.block,
                           'matmul_mode': os.environ['NEEDLE_CPU_MATMUL'], 'request_tiled_ns': True,
                           'native_extension_sha256': digest(device.mod.__file__),
                           'metrics': metrics, 'capabilities': {name: hasattr(device, name) for name in ('muon_ns_tiled', 'matmul_strided')}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['old', 'new'])
    parser.add_argument('--python-root', type=Path, default=ROOT / 'python')
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/system.json')
    parser.add_argument('--blocks', type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.blocks < 2:
        parser.error('At least two paired timing blocks required')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    baseline = prepare_baseline()
    data_dir = ROOT / 'data/fashion-mnist'
    protocol = {'date': '2026-11-01', 'baseline_commit': BASELINE, 'blocks': args.blocks,
                'comparison': 'original Compact/course Matmul/reference NS versus all three final optimizations',
                'python': 'same current Python code; native capabilities route old extension to reference paths',
                'order': 'old,new on even blocks;new,old on odd blocks;sequential processes',
                'source_sha256': {p: digest(ROOT / p) for p in SOURCES},
                'baseline_source_sha256': {p: digest(baseline / p) for p in
                                            ('src/ndarray_backend_cpu.cc', 'src/quantization_cpu.h')},
                'data_files_sha256': {p: digest(data_dir / p) for p in
                                      ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')},
                'platform': platform.platform(), 'python_version': sys.version,
                'compiler': subprocess.check_output([shutil.which('clang++') or shutil.which('g++'), '--version'], text=True),
                'compiler_flags': '-std=c++17 -O2 -shared -fPIC; identical build helper for both',
                'optimizer_calls': 80, 'optimizer_warmup_calls': 10,
                'training_updates': 128, 'training_seed': 8, 'batch_size': 64, 'split_seed': 17,
                'architecture': [784, 128, 128, 128, 64, 10],
                'rates': {'adamw_auxiliary': .001, 'muon': .007},
                'thread_limits': 'OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1',
                'timed_training': 'batch/Tensor preparation,loss reads,forward,backward,optimizer;excl setup/export',
                'scope': 'execution speed and exact output parity;not new model-quality evidence'}
    rows = []
    for block in range(args.blocks):
        for variant in (('old', 'new') if block % 2 == 0 else ('new', 'old')):
            assert {p: digest(ROOT / p) for p in SOURCES} == protocol['source_sha256']
            name = f'block{block}_{variant}'
            output = ARTIFACTS / f'{name}.json'
            with (ARTIFACTS / f'{name}.log').open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '--worker', variant,
                                '--python-root', str(baseline / 'python' if variant == 'old' else ROOT / 'python'),
                                '--block', str(block), '--output', str(output)], cwd=ROOT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                        VECLIB_MAXIMUM_THREADS='1'), stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            rows.append(json.loads(output.read_text()))
            print('DONE ' + name, flush=True)
    summary = validate_measurements(rows)
    save_json(args.output, {'protocol': protocol, 'runs': rows, 'summary': summary})
    for key, values in summary.items():
        print(f"{key}: {1000*values['old_seconds']['median']:.3f} -> "
              f"{1000*values['new_seconds']['median']:.3f} ms; "
              f"paired speedup {values['speedup']['median']:.3f}x", flush=True)


if __name__ == '__main__':
    main()
