"""Compare reference and tiled-intermediate NS using the same CPU binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from benchmark_compact_optimization import digest, save_json, summarize_pairs

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/muon_tiled'
ARTIFACTS = ROOT / 'results/muon_tiled'
BASELINE = 'e94a928cbecc78db6aaaaa047363b2f7140a007a'
SOURCES = ['src/ndarray_backend_cpu.cc', 'src/muon_cpu.h', 'src/quantization_cpu.h',
           'python/needle/optim.py', 'python/needle/backend_ndarray/ndarray.py',
           'python/needle/autograd.py', 'apps/models.py', 'apps/train_baseline.py',
           'experiments/profile_optimizer.py', 'experiments/benchmark_muon_tiled.py',
           'experiments/benchmark_compact_optimization.py', 'tools/build_cpu.py']


def parameter_hash(parameters):
    result = hashlib.sha256()
    for p in parameters:
        result.update(p.numpy().tobytes())
    return result.hexdigest()


def worker(args):
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn, optim
    from models import SmallEncoder
    from train_baseline import select_training_indices
    from profile_optimizer import make_fixture, instrumented_steps

    enabled = args.worker == 'new'
    device = ndl.cpu()
    metrics = {}
    for shape in ((128, 128), (128, 64), (64, 128)):
        values = np.random.default_rng(23).normal(size=shape).astype('float32')
        parameter = nn.Parameter(np.zeros(shape, dtype='float32'), device=device)
        method = optim.Muon([parameter], use_tiled_ns=enabled)
        update = ndl.Tensor(values, device=device, requires_grad=False)
        for _ in range(5):
            method._transform(update)
        started = time.perf_counter()
        for _ in range(40):
            output = method._transform(update)
        elapsed = time.perf_counter() - started
        metrics[f'transform_{shape[0]}x{shape[1]}'] = {
            'seconds': elapsed / 40, 'calls': 40,
            'input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
            'output_sha256': hashlib.sha256(output.numpy().tobytes()).hexdigest()}
    diagnostics = {}
    for name in ('adamw', 'muon'):
        fixture = make_fixture(name, use_tiled_ns=enabled)
        for _ in range(10):
            fixture['optimizer'].step()
        started = time.perf_counter()
        for _ in range(100):
            fixture['optimizer'].step()
        elapsed = time.perf_counter() - started
        metrics['optimizer_' + name] = {
            'seconds': elapsed / 100, 'calls': 100,
            'gradient_sha256': fixture['gradient_sha256'],
            'initial_parameter_sha256': fixture['initial_parameter_sha256'],
            'state_bytes': fixture['optimizer'].state_nbytes,
            'output_sha256': parameter_hash(fixture['parameters'])}
        if name == 'muon':
            diagnostics = instrumented_steps(fixture, 20)
            diagnostics['post_parameters_sha256'] = parameter_hash(fixture['parameters'])
            assert diagnostics['logical_ns_matmul_calls'] == 45 * 20
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
                                              use_tiled_ns=enabled)
        else:
            optimizer = optim.AdamW(parameters, lr=.001, weight_decay=.01)
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
        for p in parameters:
            fingerprint.update(p.numpy().tobytes())
        loss_bytes = np.asarray(losses, dtype='float32').tobytes()
        fingerprint.update(loss_bytes)
        features = model.features(ndl.Tensor(dataset.X[val_ids], device=device, requires_grad=False)).numpy()
        fingerprint.update(features.tobytes())
        metrics['training_' + name] = {
            'seconds': elapsed / 128, 'calls': 128, 'seed': 8,
            'initial_parameter_sha256': initial_hash,
            'train_indices_sha256': hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
            'training_loss_sha256': hashlib.sha256(loss_bytes).hexdigest(),
            'feature_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
            'output_sha256': fingerprint.hexdigest()}
    save_json(args.output, {'variant': args.worker, 'use_tiled_ns': enabled, 'block': args.block,
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
    if args.blocks < 2:
        parser.error('At least two paired timing blocks required')
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / 'tools/build_cpu.py')], cwd=ROOT, check=True)
    data_dir = ROOT / 'data/fashion-mnist'
    protocol = {'date': '2026-11-01', 'baseline_commit': BASELINE, 'blocks': args.blocks,
                'comparison': 'same final binary/Python/Compact;use_tiled_ns False versus True',
                'reference': 'old', 'packed': 'new',
                'order': 'old,new on even blocks;new,old on odd blocks;sequential processes',
                'source_sha256': {p: digest(ROOT / p) for p in SOURCES},
                'data_files_sha256': {name: digest(data_dir / name) for name in
                                      ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')},
                'platform': platform.platform(), 'python': sys.version,
                'transform_calls': 40, 'optimizer_calls': 100, 'training_updates': 128,
                'training_seed': 8, 'batch_size': 64, 'split_seed': 17,
                'architecture': [784, 128, 128, 128, 64, 10],
                'rates': {'adamw_auxiliary': .001, 'muon': .007},
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
