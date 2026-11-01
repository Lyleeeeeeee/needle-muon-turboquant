"""Separately profile original Needle optimizers with fixed synthetic gradients."""
import argparse
from contextlib import ExitStack
import cProfile
import hashlib
import json
import os
from pathlib import Path
import platform
import pstats
import statistics as stats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]

import numpy as np
import needle as ndl
from needle import optim
from needle.backend_ndarray.ndarray import NDArray
from models import SmallEncoder
from run_joint_retrieval import digest, save_json

REPORTS = ROOT / 'reports/time_budget_study'
ARTIFACTS = ROOT / 'results/time_budget_study'
SOURCES = ['experiments/profile_optimizer.py', 'apps/models.py', 'python/needle/optim.py',
           'python/needle/backend_ndarray/ndarray.py', 'python/needle/autograd.py',
           'python/needle/ops/ops_mathematic.py', 'src/ndarray_backend_cpu.cc', 'src/muon_cpu.h']


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
            if (name in ('compact', 'matmul', 'matmul_tiled', 'fill', 'muon_ns_tiled')
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
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--blocks', type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1 or args.blocks < 2:
        parser.error('Positive steps and at least two blocks required')
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    fixtures = {method: make_fixture(method) for method in ('adamw', 'muon')}
    assert fixtures['adamw']['gradient_sha256'] == fixtures['muon']['gradient_sha256']
    assert fixtures['adamw']['initial_parameter_sha256'] == fixtures['muon']['initial_parameter_sha256']
    for fixture in fixtures.values():
        for _ in range(10):
            fixture['optimizer'].step()
    plain = {method: [] for method in fixtures}
    for block in range(args.blocks):
        for method in (('adamw', 'muon') if block % 2 == 0 else ('muon', 'adamw')):
            started = time.perf_counter()
            for _ in range(args.steps):
                fixtures[method]['optimizer'].step()
            plain[method].append((time.perf_counter() - started) / args.steps)
    instrumented = {method: instrumented_steps(fixture, args.steps)
                    for method, fixture in fixtures.items()}
    assert instrumented['muon']['logical_ns_matmul_calls'] == 45 * args.steps
    assert instrumented['adamw']['ndarray_matmul']['calls'] == 0
    audit = make_fixture('muon')
    for _ in range(10):
        audit['optimizer'].step()
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(30):
        audit['optimizer'].step()
    profile.disable()
    profile.dump_stats(str(ARTIFACTS / 'muon.cprofile'))
    stats_rows = pstats.Stats(profile).stats
    calls = [{'file': str(Path(filename).relative_to(ROOT)) if str(filename).startswith(str(ROOT)) else filename,
              'line': line, 'function': function, 'primitive_calls': primitive, 'calls': total,
              'self_seconds': self_seconds, 'cumulative_seconds': cumulative}
             for (filename, line, function), (primitive, total, self_seconds, cumulative, callers)
             in stats_rows.items()]
    record = {'date': '2026-11-01', 'purpose': 'fixed-gradient cost audit, not convergence evidence',
              'source_sha256': {p: digest(ROOT / p) for p in SOURCES},
              'cpu_extension_sha256': digest(fixtures['muon']['device'].mod.__file__),
              'platform': platform.platform(), 'machine': platform.machine(),
              'thread_limits': {k: os.environ.get(k) for k in
                                ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
              'steps_per_block': args.steps, 'blocks': args.blocks, 'warmup_steps': 10,
              'gradient_sha256': fixtures['muon']['gradient_sha256'],
              'plain': {method: {'seconds_per_step': values, 'median_seconds': stats.median(values),
                                 'mean_seconds': stats.mean(values), 'sample_std_seconds': stats.stdev(values)}
                        for method, values in plain.items()},
              'instrumented': instrumented,
              'cprofile_top_self': sorted(calls, key=lambda r: r['self_seconds'], reverse=True)[:30],
              'cprofile_top_cumulative': sorted(calls, key=lambda r: r['cumulative_seconds'], reverse=True)[:30]}
    save_json(REPORTS / 'optimizer_profile.json', record)
    lines = ['# Optimizer cost audit', '',
             'Fixed deterministic gradients, current deep parameter shapes, ten warmup updates, '
             f'{args.blocks} alternating blocks of {args.steps} steps. No forward/backward computation is included. '
             'These are separate diagnostic runs after all retained training/search timings.', '',
             '| Method | Uninstrumented median, ms/update | Instrumented, ms/update | Instrumentation change, % |',
             '|---|---:|---:|---:|']
    for method in fixtures:
        baseline = record['plain'][method]['median_seconds']
        measured = instrumented[method]['total_seconds'] / args.steps
        lines.append(f'| {method} | {1000*baseline:.3f} | {1000*measured:.3f} | {100*(measured/baseline-1):+.2f} |')
    measured = instrumented['muon']
    lines += ['', '### Hybrid update phases', '', '| Phase | ms/update |', '|---|---:|']
    for label, seconds in measured['phase_seconds'].items():
        lines.append(f'| {label} | {1000*seconds/args.steps:.3f} |')
    lines += ['', 'The NS transform is inside the Muon phase; do not add these nested times.', '',
              '### Matrix products and native layout copies', '',
              '| Operation | Calls/update | ms/update |', '|---|---:|---:|']
    values = [('NDArray matmul, including packing and dispatch', measured['ndarray_matmul'])]
    values += [(f'Native {name}', row) for name, row in measured['native_kernels'].items()
               if row['calls'] and name in ('matmul', 'matmul_tiled', 'compact', 'muon_ns_tiled')]
    for label, row in values:
        lines.append(f"| {label} | {row['calls']/args.steps:g} | {1000*row['seconds']/args.steps:.3f} |")
    native_total = sum(r['seconds'] for r in measured['native_kernels'].values())
    lines += ['', f"All native kernels total {1000*native_total/args.steps:.3f} ms/update; "
              f"remaining Python dispatch/allocation/timer overhead is approximately "
              f"{1000*(measured['total_seconds']-native_total)/args.steps:.3f} ms/update. "
              'The remaining term is not an isolated allocator measurement. NDArray matmul is '
              'inclusive of native matrix kernels and their layout copies; do not sum nested rows.', '',
              'Three hidden matrices × five NS iterations × three products =45 extra matrix '
              'products per update. The reference path repacks both inputs and unpacks its '
              'output for every product. The packed NS path instead retains tiled intermediates '
              'and performs one input pack, five tiled transposes and one output unpack per '
              'hidden matrix. Generic Compact calls and derived internal layout passes are '
              'reported separately in JSON. Native NS time includes its internal matrix products '
              'and layout work; do not add it to nested logical operation counts.', '',
              'The NS matrix products alone add approximately138.94 million multiply/add FLOPs '
              'per update. The five dense layers\' forward and two backward products at batch64 '
              'are approximately54.51 million FLOPs; this estimate excludes elementwise/loss '
              'work and all AdamW operations. Muon overhead is therefore material for this small '
              'batch. The author\'s [runtime analysis](https://kellerjordan.github.io/posts/muon/#runtime-analysis) '
              'also makes overhead depend on matrix size, NS iterations and batch size; its '
              'large-token-batch GPU examples do not predict this float32 CPU implementation.', '',
              'The profiler preserves original update outputs bit-for-bit and restores every '
              'wrapped method/native function. Independent optimizer numerical-reference tests '
              'remain in the full suite. No formulas, NS counts or kernels were changed. The '
              'cProfile tables in JSON are a separate call-cost audit with additional overhead.', '']
    (REPORTS / 'optimizer_profile.md').write_text('\n'.join(lines))
    print(json.dumps({method: {'plain_ms': 1000*row['median_seconds'],
                              'instrumented_ms': 1000*instrumented[method]['total_seconds']/args.steps}
                      for method, row in record['plain'].items()}, indent=2))


if __name__ == '__main__':
    main()
