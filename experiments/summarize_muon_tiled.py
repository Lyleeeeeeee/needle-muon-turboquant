"""Validate paired tiled-NS measurements and retain report-ready evidence."""
import json
import os
from pathlib import Path
import shutil
import statistics as stats
import subprocess

from benchmark_muon_tiled import ROOT, REPORTS, ARTIFACTS, digest, save_json, summarize_pairs


def layout_work(record):
    d = record['diagnostics']
    count = d['steps']
    generic, packed = d['native_kernels']['compact'], d['packed_layout']
    return {'generic_compact_calls': generic['calls'] / count,
            'internal_passes': packed['passes'] / count,
            'total_passes': (generic['calls'] + packed['passes']) / count,
            'layout_bytes_written': 4 * (generic['written_elements'] + packed['written_elements']) / count,
            'logical_matrix_products': d['logical_ns_matmul_calls'] / count}


def draw_plot(summary, layouts):
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [(axes[0], ['transform_128x128', 'transform_128x64', 'transform_64x128'],
               ['128x128', '128x64', '64x128'], 'Whole Muon transform, including normalization'),
              (axes[1], ['optimizer_muon', 'training_muon', 'training_adamw'],
               ['Muon optimizer', 'Muon full step', 'AdamW full step'], 'Update cost and unchanged AdamW control')]
    for ax, keys, labels, title in panels:
        for index, variant in enumerate(('old', 'new')):
            values = [summary[k][variant + '_seconds'] for k in keys]
            ax.bar(np.arange(len(keys)) + (index - .5) * .32,
                   [1000 * v['mean'] for v in values], width=.32,
                   yerr=[1000 * v['sample_std'] for v in values], capsize=3,
                   color=['#2869a6', '#dd7924'][index],
                   label=['Reference NS', 'Packed NS'][index])
        ax.set(title=title, ylabel='Milliseconds', xticks=range(len(keys)), xticklabels=labels)
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
    generic = [layouts[v]['generic_compact_calls'] for v in ('old', 'new')]
    internal = [layouts[v]['internal_passes'] for v in ('old', 'new')]
    axes[2].bar([0, 1], generic, color='#2869a6', label='Measured generic Compact calls')
    axes[2].bar([0, 1], internal, bottom=generic, color='#dd7924', label='Derived internal packed layout passes')
    for index in range(2):
        axes[2].text(index, generic[index] + internal[index] + 3,
                     str(int(generic[index] + internal[index])), ha='center')
    axes[2].set(title='Layout passes per hybrid update, including internal work',
                ylabel='Logical full-matrix rearrangements', xticks=[0, 1],
                xticklabels=['Reference NS', 'Packed NS'], ylim=(0, 180))
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Tiled intermediate reuse: five alternating timing pairs; error bars = sample SD')
    fig.savefig(REPORTS / 'comparison.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'comparison.png', dpi=150)
    plt.close(fig)


def main():
    report = json.loads((REPORTS / 'summary.json').read_text())
    protocol = report['protocol']
    assert {p: digest(ROOT / p) for p in protocol['source_sha256']} == protocol['source_sha256']
    assert {p: digest(ROOT / 'data/fashion-mnist' / p) for p in protocol['data_files_sha256']} == protocol['data_files_sha256']
    rows = []
    for path, expected in report['raw_report_sha256'].items():
        assert digest(REPORTS / path) == expected
        rows.append(json.loads((REPORTS / path).read_text()))
    assert len(rows) == 10 and len({r['native_extension_sha256'] for r in rows}) == 1
    extensions = list((ROOT / 'python/needle/backend_ndarray').glob('ndarray_backend_cpu*.so'))
    assert any(digest(p) == rows[0]['native_extension_sha256'] for p in extensions)
    summary = summarize_pairs(rows)
    assert summary == report['summary']
    first_stage = json.loads((ROOT / 'reports/compact_optimization/block0_new.json').read_text())
    for key in summary:
        assert len({r['metrics'][key]['output_sha256'] for r in rows}) == 1
        if key in first_stage['metrics']:
            assert rows[0]['metrics'][key]['output_sha256'] == first_stage['metrics'][key]['output_sha256']
    layouts = {}
    for variant in ('old', 'new'):
        values = [layout_work(r) for r in rows if r['variant'] == variant]
        assert all(v == values[0] for v in values)
        layouts[variant] = values[0]
        assert layouts[variant]['logical_matrix_products'] == 45
    assert layouts['old']['total_passes'] == 157 and layouts['new']['total_passes'] == 28
    assert layouts['new']['internal_passes'] == 21
    assert len({r['diagnostics']['post_parameters_sha256'] for r in rows}) == 1
    environment = {'compiler': subprocess.check_output([shutil.which('clang++') or shutil.which('g++'), '--version'], text=True),
                   'thread_limits': 'OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1',
                   'report_generator_sha256': digest(Path(__file__)), 'layout_work_per_update': layouts}
    save_json(REPORTS / 'environment.json', environment)
    lines = ['# Retaining tiled intermediates in Muon', '',
             'The new native CPU NS path retains tiled intermediate matrices across all five '
             'iterations. Every measured transform, optimizer result and short training trace '
             'matches the reference path bit-for-bit. The reference and packed paths use the '
             'same binary and the previous Compact indexing optimization.', '',
             '## Paired execution time', '',
             'Five alternating sequential process pairs. Old/new columns are medians of the '
             'five per-process average times; speedup is the median of within-pair ratios. '
             'These are timing repetitions, not independent model-quality seeds.', '',
             '| Workload | Reference, ms | Packed, ms | Paired median speedup |',
             '|---|---:|---:|---:|']
    for key in ['transform_128x128', 'transform_128x64', 'transform_64x128',
                'optimizer_muon', 'optimizer_adamw', 'training_muon', 'training_adamw']:
        r = summary[key]
        lines.append(f"| {key} | {1000*r['old_seconds']['median']:.3f} | "
                     f"{1000*r['new_seconds']['median']:.3f} | {r['speedup']['median']:.3f}x |")
    lines += ['', 'Transforms include the unchanged Frobenius normalization and final aspect '
              'scaling, with five warmups and40 measured calls. Optimizer timing includes '
              'Muon and auxiliary AdamW, with ten warmups and100 measured updates. Full training '
              'uses128 updates of the deep Fashion-MNIST model, seed8/batch64, with the same '
              'split/order/rates. It includes tensor preparation, forward/backward and optimizer '
              'updates, excluding setup, data loading and final feature evaluation.', '',
              'All final parameters, all128 losses and128 validation embeddings agree exactly '
              'across paths and reproduce the earlier Compact-stage fixed-step outputs. '
              'AdamW is an unchanged control. No new convergence or retrieval-quality result '
              'is inferred from these short execution-time measurements.', '',
              '![Execution time and layout work](comparison.svg)', '',
              '## What was reduced', '',
              'The reference route packs both inputs and unpacks the result for each product. '
              'The new route packs X once and unpacks once per hidden matrix. X changes every '
              'iteration, so it still refreshes a tiled transpose five times. For three hidden '
              'matrices, this is21 internal layout passes. Seven other generic compactions '
              '(normalization/broadcast/final orientation) remain.', '',
              '| Work per hybrid optimizer update | Reference | Packed |',
              '|---|---:|---:|']
    for label, key in [('Measured generic Compact calls', 'generic_compact_calls'),
                       ('Derived internal packed layout passes', 'internal_passes'),
                       ('Total logical layout passes', 'total_passes'),
                       ('Logical NS matrix products', 'logical_matrix_products')]:
        lines.append(f"| {label} | {layouts['old'][key]:g} | {layouts['new'][key]:g} |")
    lines.append(f"| Layout writes, decimal MB | {layouts['old']['layout_bytes_written']/1e6:.6f} | "
                 f"{layouts['new']['layout_bytes_written']/1e6:.6f} |")
    lines += ['', 'Generic Compact calls and their written-element counts are observed by the '
              'profiler. Internal passes/written elements are derived from native loop control '
              'and actual matrix dimensions, not hardware traffic counters. These transfers '
              'have different sizes; the byte column counts float32 elements written during '
              'layout conversion, not total process memory or actual DRAM traffic. Native NS '
              'time includes all its internal transposes, products and allocation work.', '',
              'This local native implementation also removes per-operation Python dispatch and '
              'reuses scratch buffers. The experiment does not isolate the individual speed '
              'contributions of layout reuse, dispatch reduction and allocation reuse.', '',
              '## Algorithm and validation', '',
              '- MatmulTiled/AlignedDot arithmetic and accumulation order are reused. Scalar '
              'multiplication/addition stages remain separate to preserve float32 rounding.',
              '- Momentum, coefficients, NS5, Frobenius normalization, aspect scaling, '
              'weight decay and auxiliary AdamW are unchanged.',
              '- `use_tiled_ns=True` selects supported aligned CPU matrices automatically. '
              '`use_tiled_ns=False` retains the reference. Unsupported sizes/backends fall back.',
              '- The rebuilt backend passes126 tests, including direct native bit parity, '
              'independent float64 SVD-polynomial oracles, multi-update optimizer states, '
              'zero/missing gradients, shape errors and routing.',
              '- Five within-call scratch arrays hold3RC+2R² floats in normalized orientation. '
              'They are released after each call; no persistent cache/state was added. '
              'This is not a claim about whole-process peak memory.',
              '- Source, binary and data hashes and all raw pairs are retained. Historical '
              'experiments remain tied to their recorded revisions.', '',
              '## Reproduction', '', '```bash', 'python tools/build_cpu.py',
              'python -m pytest -q', 'python experiments/benchmark_muon_tiled.py',
              'python experiments/summarize_muon_tiled.py', '```', '',
              'See [the design](../../docs/muon_tiled_design.md) and '
              '[durable implementation log](../../docs/project_change_log.md) for report context.', '']
    text = '\n'.join(lines)
    for before, after in [('and40', 'and 40'), ('and100', 'and 100'), ('uses128', 'uses 128'),
                          ('seed8/batch64', 'seed 8/batch 64'), ('all128', 'all 128'),
                          ('and128', 'and 128'), ('is21', 'is 21'), ('NS5', 'NS 5'),
                          ('passes126', 'passes 126'), ('hold3RC', 'hold 3RC')]:
        text = text.replace(before, after)
    (REPORTS / 'summary.md').write_text(text)
    draw_plot(summary, layouts)
    print('Validated ten paired reports, baseline-output reproduction, and logical work accounting.')


if __name__ == '__main__':
    main()
