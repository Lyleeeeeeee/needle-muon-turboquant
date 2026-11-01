"""Profile the current native AdamW and Muon updates using fixed gradients."""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
import numpy as np
import needle as ndl
from needle import optim
from needle.backend_ndarray.ndarray import NDArray
from models import SmallEncoder
from common import digest, save_json


def make_fixture(method, use_tiled_ns=True):
    if method not in ('adamw', 'muon'):
        raise ValueError('Unknown optimizer')
    np.random.seed(23)
    rng = np.random.default_rng(23)
    device = ndl.cpu()
    model = SmallEncoder(extra_hidden_layers=2, device=device)
    params = model.parameters()
    gradient_hash = hashlib.sha256()
    parameter_hash = hashlib.sha256()
    for param in params:
        values = rng.normal(0, .01, param.shape).astype('float32')
        param.grad = ndl.Tensor(values, device=device, requires_grad=False)
        gradient_hash.update(values.tobytes())
        parameter_hash.update(param.numpy().tobytes())
    if method == 'muon':
        matrices = model.muon_parameters()
        ids = {id(p) for p in matrices}
        optimizer = optim.MuonWithAuxAdamW(matrices, [p for p in params if id(p) not in ids],
                                          lr=.007, aux_lr=.001, weight_decay=.01,
                                          use_tiled_ns=use_tiled_ns)
    else:
        optimizer = optim.AdamW(params, lr=.001, weight_decay=.01)
    return {'optimizer': optimizer, 'parameters': params, 'device': device,
            'gradient_sha256': gradient_hash.hexdigest(),
            'initial_parameter_sha256': parameter_hash.hexdigest()}


def instrumented_steps(fixture, steps):
    if type(steps) is not int or steps < 1:
        raise ValueError('Expected positive profiling steps')
    record = {'steps': steps, 'phase_seconds': {}, 'phase_calls': {},
              'ndarray_matmul': {'calls': 0, 'seconds': 0.},
              'ndarray_compact': {'calls': 0, 'seconds': 0.},
              'native_kernels': {}, 'matmul_shapes': {}, 'logical_ns_matmul_calls': 0,
              'packed_layout': {'passes': 0, 'written_elements': 0}}
    optimizer = fixture['optimizer']
    with ExitStack() as restore:
        def wrap(owner, name, counter):
            original = getattr(owner, name)

            def timed(*args, **kwargs):
                if owner is fixture['device'].mod and name == 'compact':
                    counter['written_elements'] = counter.get('written_elements', 0) + args[1].size
                if owner is fixture['device'].mod and name == 'muon_ns_tiled':
                    rows, columns, iterations = args[2:]
                    record['logical_ns_matmul_calls'] += 3 * iterations
                    record['packed_layout']['passes'] += iterations + 2
                    record['packed_layout']['written_elements'] += (iterations + 2) * rows * columns
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    counter['seconds'] += time.perf_counter() - started
                    counter['calls'] += 1

            setattr(owner, name, timed)
            restore.callback(setattr, owner, name, original)

        for name in dir(fixture['device'].mod):
            if (name in ('compact', 'matmul', 'matmul_tiled', 'matmul_strided', 'fill', 'muon_ns_tiled')
                    or name.startswith(('ewise_', 'scalar_', 'reduce_'))):
                counter = record['native_kernels'].setdefault(name, {'calls': 0, 'seconds': 0.})
                wrap(fixture['device'].mod, name, counter)
        wrap(NDArray, 'compact', record['ndarray_compact'])
        original_matmul = NDArray.__matmul__

        def matmul(a, b):
            shape = f'{a.shape} @ {b.shape}'
            record['matmul_shapes'][shape] = record['matmul_shapes'].get(shape, 0) + 1
            started = time.perf_counter()
            try:
                return original_matmul(a, b)
            finally:
                record['ndarray_matmul']['calls'] += 1
                record['ndarray_matmul']['seconds'] += time.perf_counter() - started

        NDArray.__matmul__ = matmul
        restore.callback(setattr, NDArray, '__matmul__', original_matmul)
        phases = ([('muon', optimizer.muon, 'step'), ('auxiliary_adamw', optimizer.adamw, 'step'),
                   ('ns_transform', optimizer.muon, '_transform')]
                  if isinstance(optimizer, optim.MuonWithAuxAdamW) else [('adamw', optimizer, 'step')])
        counters = {}
        for label, owner, name in phases:
            counters[label] = {'calls': 0, 'seconds': 0.}
            wrap(owner, name, counters[label])
        started = time.perf_counter()
        for _ in range(steps):
            optimizer.step()
        record['total_seconds'] = time.perf_counter() - started
        record['logical_ns_matmul_calls'] += record['ndarray_matmul']['calls']
        record['phase_seconds'] = {label: v['seconds'] for label, v in counters.items()}
        record['phase_calls'] = {label: v['calls'] for label, v in counters.items()}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/profile.json')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('Positive profiling steps required')
    records = {}
    for name in ('adamw', 'muon'):
        fixture = make_fixture(name)
        for _ in range(10):
            fixture['optimizer'].step()
        records[name] = instrumented_steps(fixture, args.steps)
    save_json(args.output, {'purpose': 'fixed-gradient cost audit, not convergence',
                           'native_extension_sha256': digest(ndl.cpu().mod.__file__),
                           'records': records})
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
