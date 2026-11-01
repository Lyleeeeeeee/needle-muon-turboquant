"""Paired native-backend measurements isolating the coordinate-carry change."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics as stats
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'efe063d3b67153ced96c8435a381866ee0077605'
REPORTS = ROOT / 'reports/compact_optimization'
ARTIFACTS = ROOT / 'results/compact_optimization'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(values):
    return {'values': values, 'mean': stats.mean(values), 'median': stats.median(values),
            'sample_std': stats.stdev(values) if len(values) > 1 else 0.}


def summarize_pairs(rows):
    pairs = {}
    for row in rows:
        key = row['block'], row['variant']
        if key in pairs or row['variant'] not in ('old', 'new'):
            raise ValueError('Duplicate or invalid pair variant')
        pairs[key] = row
    blocks = sorted({key[0] for key in pairs})
    if not blocks or any((b, v) not in pairs for b in blocks for v in ('old', 'new')):
        raise ValueError('Incomplete before/after pair')
    metrics = set(pairs[blocks[0], 'old']['metrics'])
    if any(set(r['metrics']) != metrics for r in pairs.values()):
        raise ValueError('Paired metric coverage differs')
    result = {}
    for metric in sorted(metrics):
        before, after = [], []
        for block in blocks:
            old, new = [pairs[block, v]['metrics'][metric] for v in ('old', 'new')]
            if old['output_sha256'] != new['output_sha256']:
                raise ValueError('Paired output changed: ' + metric)
            if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in (old, new)):
                raise ValueError('Invalid measured time')
            before.append(old['seconds'])
            after.append(new['seconds'])
        result[metric] = {'old_seconds': distribution(before), 'new_seconds': distribution(after),
                          'speedup': distribution([a / b for a, b in zip(before, after)]),
                          'outputs_bit_identical': True}
    return result


def prepare_baseline():
    """Use current shared Python code and a separately compiled old native source."""
    target = ARTIFACTS / 'baseline'
    for relative in ('src/ndarray_backend_cpu.cc', 'src/quantization_cpu.h'):
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        content = subprocess.check_output(['git', 'show', f'{BASELINE}:{relative}'], cwd=ROOT)
        output.write_bytes(content)
    assert digest(target / 'src/quantization_cpu.h') == digest(ROOT / 'src/quantization_cpu.h')
    python_root = target / 'python'
    shutil.copytree(ROOT / 'python', python_root, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.dylib'))
    spec = importlib.util.spec_from_file_location('compact_build_helper', ROOT / 'tools/build_cpu.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.ROOT = target
    builder.main()
    subprocess.run([sys.executable, str(ROOT / 'tools/build_cpu.py')], cwd=ROOT, check=True)
    return target


def worker(args):
    import numpy as np

    # Import Needle before helpers that insert the main Python path. Its entire
    # package, including the selected native extension, stays cached thereafter.
    sys.path[:0] = [str(args.python_root), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn, optim
    from needle.backend_ndarray import NDArray
    from models import SmallEncoder
    from profile_optimizer import make_fixture, instrumented_steps
    from train_baseline import select_training_indices

    device = ndl.cpu()
    assert Path(device.mod.__file__).resolve().is_relative_to(args.python_root.resolve())
    metrics = {}
    cases = {
        'transpose128': ((128, 128), (1, 128), 0),
        'transpose64x128': ((64, 128), (1, 64), 0),
        'tile128': ((16, 16, 8, 8), (1024, 8, 128, 1), 0),
        'untile128': ((16, 8, 16, 8), (1024, 8, 64, 1), 0),
        'tile784x128': ((98, 16, 8, 8), (1024, 8, 128, 1), 0),
        'slice_offset': ((64, 64), (256, 2), 257),
        'broadcast': ((32, 128), (0, 1), 0),
    }
    for name, (shape, strides, offset) in cases.items():
        maximum = offset + sum((n - 1) * s for n, s in zip(shape, strides))
        source = np.arange(maximum + 1, dtype='float32')
        native = NDArray(source, device=device)
        output = NDArray.make(shape, device=device)
        for _ in range(10):
            device.compact(native._handle, output._handle, shape, strides, offset)
        started = time.perf_counter()
        for _ in range(1000):
            device.compact(native._handle, output._handle, shape, strides, offset)
        elapsed = time.perf_counter() - started
        expected = np.ndarray(shape, dtype='float32', buffer=source, offset=offset * 4,
                              strides=tuple(s * 4 for s in strides))
        actual = output.numpy().copy()
        np.testing.assert_array_equal(actual, expected)
        metrics['compact_' + name] = {'seconds': elapsed / 1000, 'calls': 1000,
                                     'shape': list(shape), 'strides': list(strides), 'offset': offset,
                                     'output_sha256': hashlib.sha256(actual.tobytes()).hexdigest()}
    diagnostics = {}
    if args.mode == 'all':
        for method in ('adamw', 'muon'):
            fixture = make_fixture(method)
            for _ in range(10):
                fixture['optimizer'].step()
            started = time.perf_counter()
            for _ in range(100):
                fixture['optimizer'].step()
            elapsed = time.perf_counter() - started
            fingerprint = hashlib.sha256()
            for p in fixture['parameters']:
                fingerprint.update(p.numpy().tobytes())
            metrics['optimizer_' + method] = {'seconds': elapsed / 100, 'calls': 100,
                                              'gradient_sha256': fixture['gradient_sha256'],
                                              'output_sha256': fingerprint.hexdigest()}
            if method == 'muon':
                diagnostics['muon'] = instrumented_steps(fixture, 20)
                assert diagnostics['muon']['ndarray_matmul']['calls'] == 45 * 20
        data_dir = ROOT / 'data/fashion-mnist'
        dataset = ndl.data.MNISTDataset(str(data_dir / 'train-images-idx3-ubyte.gz'),
                                        str(data_dir / 'train-labels-idx1-ubyte.gz'))
        for method in ('adamw', 'muon'):
            np.random.seed(8)
            rng = np.random.default_rng(8)
            train_ids, val_ids, _ = select_training_indices(len(dataset), len(dataset), 55000,
                                                           128, rng, 5000, 17)
            x_train, y_train = dataset.X[train_ids], dataset.y[train_ids]
            model = SmallEncoder(extra_hidden_layers=2, device=device)
            params = model.parameters()
            if method == 'muon':
                matrices = model.muon_parameters()
                ids = {id(p) for p in matrices}
                optimizer = optim.MuonWithAuxAdamW(matrices, [p for p in params if id(p) not in ids],
                                                  lr=.007, aux_lr=.001, weight_decay=.01)
            else:
                optimizer = optim.AdamW(params, lr=.001, weight_decay=.01)
            criterion = nn.SoftmaxLoss()
            losses = []
            started = time.perf_counter()
            order = rng.permutation(len(x_train))
            for step in range(128):
                batch = order[step * 64:(step + 1) * 64]
                x = ndl.Tensor(x_train[batch], device=device, requires_grad=False)
                y = ndl.Tensor(y_train[batch], device=device, dtype='float32', requires_grad=False)
                optimizer.reset_grad()
                loss = criterion(model(x), y)
                losses.append(float(loss.numpy().reshape(-1)[0]))
                loss.backward()
                optimizer.step()
            elapsed = time.perf_counter() - started
            fingerprint = hashlib.sha256()
            for p in params:
                fingerprint.update(p.numpy().tobytes())
            loss_hash = hashlib.sha256(np.asarray(losses, dtype='float32').tobytes()).hexdigest()
            fingerprint.update(np.asarray(losses, dtype='float32').tobytes())
            features = model.features(ndl.Tensor(dataset.X[val_ids], device=device, requires_grad=False)).numpy()
            fingerprint.update(features.tobytes())
            metrics['training_' + method] = {
                'seconds': elapsed / 128, 'calls': 128, 'seed': 8,
                'train_indices_sha256': hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
                'training_loss_sha256': loss_hash,
                'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
                'output_sha256': fingerprint.hexdigest()}
    save_json(args.output, {'variant': args.worker, 'block': args.block, 'mode': args.mode,
                           'native_extension_sha256': digest(device.mod.__file__),
                           'metrics': metrics, 'diagnostics': diagnostics})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['old', 'new'])
    parser.add_argument('--python-root', type=Path)
    parser.add_argument('--block', type=int, default=0)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['compact', 'all'], default='all')
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--report-dir', type=Path, default=REPORTS)
    parser.add_argument('--require-speedup', type=float, default=0.)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.blocks < 2:
        parser.error('At least two paired blocks required')
    baseline = prepare_baseline()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    sources = ['src/ndarray_backend_cpu.cc', 'src/quantization_cpu.h', 'tools/build_cpu.py',
               'python/needle/backend_ndarray/ndarray.py', 'python/needle/optim.py', 'apps/models.py',
               'experiments/profile_optimizer.py', 'experiments/benchmark_compact_optimization.py']
    protocol = {'date': '2026-11-01', 'baseline_commit': BASELINE, 'mode': args.mode, 'blocks': args.blocks,
                'order': 'old,new on even blocks;new,old on odd blocks;sequential processes',
                'source_sha256': {p: digest(ROOT / p) for p in sources},
                'old_cpu_source_sha256': digest(baseline / 'src/ndarray_backend_cpu.cc'),
                'platform': platform.platform(), 'python': sys.version,
                'compact_calls': 1000, 'optimizer_calls': 100, 'training_updates': 128,
                'training': 'Fashion-MNIST;deep model;seed8;55000/5000 split17;batch64;LR.001/Muon.007',
                'scope': 'execution speed and exact outputs;not equal-time model quality or retrieval'}
    save_json(args.report_dir / 'protocol.json', protocol)
    rows = []
    for block in range(args.blocks):
        for variant in (('old', 'new') if block % 2 == 0 else ('new', 'old')):
            assert {p: digest(ROOT / p) for p in sources} == protocol['source_sha256']
            name = f'block{block}_{variant}'
            output = args.report_dir / f'{name}.json'
            python_root = baseline / 'python' if variant == 'old' else ROOT / 'python'
            with (ARTIFACTS / (args.report_dir.name + '_' + name + '.log')).open('w') as log:
                subprocess.run([sys.executable, str(Path(__file__)), '--worker', variant,
                                '--python-root', str(python_root), '--block', str(block),
                                '--mode', args.mode, '--output', str(output)], cwd=ROOT,
                               env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                        VECLIB_MAXIMUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            rows.append(json.loads(output.read_text()))
            print('DONE ' + name, flush=True)
    summary = summarize_pairs(rows)
    save_json(args.report_dir / 'summary.json', {'protocol': protocol, 'summary': summary,
                                               'raw_report_sha256': {f'block{r["block"]}_{r["variant"]}.json':
                                                                    digest(args.report_dir / f'block{r["block"]}_{r["variant"]}.json')
                                                                    for r in rows}})
    for key, values in summary.items():
        print(f"{key}: {values['old_seconds']['median']*1000:.4f} -> "
              f"{values['new_seconds']['median']*1000:.4f} ms; "
              f"paired median speedup={values['speedup']['median']:.3f}x", flush=True)
    if summary['compact_tile128']['speedup']['median'] < args.require_speedup:
        raise SystemExit('Performance target not met: representative tiled Compact below required speedup')


if __name__ == '__main__':
    main()
