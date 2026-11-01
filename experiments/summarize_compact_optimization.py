"""Verify and summarize the kernel-only optimization, including the first trial."""
import json
import os
from pathlib import Path
import shutil
import statistics as stats
import subprocess

from benchmark_compact_optimization import BASELINE, ROOT, REPORTS, ARTIFACTS, digest, save_json, summarize_pairs


def validate_run(directory, source_override=None):
    report = json.loads((directory / 'summary.json').read_text())
    protocol = report['protocol']
    for relative, expected in protocol['source_sha256'].items():
        if relative == 'src/ndarray_backend_cpu.cc' and source_override is not None:
            assert digest(source_override) == expected
        elif relative == 'src/ndarray_backend_cpu.cc' and directory.name == 'old_control':
            import hashlib
            original = subprocess.check_output(['git', 'show', f'{BASELINE}:{relative}'], cwd=ROOT)
            assert hashlib.sha256(original).hexdigest() == expected
        else:
            assert digest(ROOT / relative) == expected
    rows = []
    for name, expected in report['raw_report_sha256'].items():
        assert digest(directory / name) == expected
        rows.append(json.loads((directory / name).read_text()))
    assert len(rows) == 2 * protocol['blocks']
    assert summarize_pairs(rows) == report['summary']
    for variant in ('old', 'new'):
        assert len({r['native_extension_sha256'] for r in rows if r['variant'] == variant}) == 1
    for key in report['summary']:
        assert len({r['metrics'][key]['output_sha256'] for r in rows}) == 1
    return report, rows


def draw_plot(summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    keys = ['transpose128', 'transpose64x128', 'tile128', 'untile128', 'tile784x128', 'slice_offset', 'broadcast']
    labels = ['Transpose128', 'Transpose64x128', 'Tile128', 'Untile128', 'Tile784x128', 'Slice+offset', 'Broadcast']
    values = [summary['compact_' + k]['speedup'] for k in keys]
    axes[0].bar(range(len(keys)), [v['mean'] for v in values],
                 yerr=[v['sample_std'] for v in values], color='#4b8f7c', capsize=3)
    axes[0].axhline(1, color='gray', linewidth=1)
    axes[0].set(title='Direct Compact calls: paired speedup', ylabel='Old time / new time',
                xticks=range(len(keys)), xticklabels=labels)
    axes[0].tick_params(axis='x', labelrotation=45, labelsize=8)
    for ax, prefix, title in [(axes[1], 'optimizer', 'Fixed-gradient optimizer update'),
                              (axes[2], 'training', 'Full training update (128-step run)')]:
        for index, variant in enumerate(('old', 'new')):
            values = [summary[f'{prefix}_{method}'][variant + '_seconds'] for method in ('adamw', 'muon')]
            ax.bar(np.arange(2) + (index - .5) * .32, [1000 * v['mean'] for v in values],
                   width=.32, yerr=[1000 * v['sample_std'] for v in values], capsize=3,
                   color=['#2869a6', '#dd7924'][index], label=['Old HW3 Compact', 'Optimized Compact'][index])
        ax.set(title=title, ylabel='Milliseconds per update', xticks=[0, 1],
               xticklabels=['AdamW', 'Muon + aux AdamW'])
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Compact indexing only: five alternating timing pairs; error bars = sample SD')
    fig.savefig(REPORTS / 'comparison.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'comparison.png', dpi=150)
    plt.close(fig)


def main():
    final, rows = validate_run(REPORTS)
    generic, _ = validate_run(REPORTS / 'generic_carry', REPORTS / 'generic_carry/native_source.cc')
    control, _ = validate_run(REPORTS / 'old_control')
    summary = final['summary']
    assert control['summary']['compact_tile128']['speedup']['median'] < 1.2
    for key in ('compact_transpose128', 'compact_transpose64x128', 'compact_broadcast', 'compact_slice_offset'):
        assert summary[key]['speedup']['median'] >= .95, 'Unresolved measured low-rank regression: ' + key
    assert summary['compact_tile128']['speedup']['median'] >= 1.2
    diagnostics = {}
    for variant in ('old', 'new'):
        trials = [r['diagnostics']['muon'] for r in rows if r['variant'] == variant]
        for r in trials:
            assert r['ndarray_matmul']['calls'] / r['steps'] == 45
            assert r['native_kernels']['compact']['calls'] / r['steps'] == 157
        diagnostics[variant] = {key: stats.median(r['native_kernels'][key]['seconds'] / r['steps'] for r in trials)
                                for key in ('compact', 'matmul_tiled')}
    environment = {'compiler': subprocess.check_output([shutil.which('clang++') or shutil.which('g++'), '--version'], text=True),
                   'flags': '-std=c++17 -O2 -shared -fPIC; macOS SDK and Python/pybind11 include paths from tools/build_cpu.py',
                   'thread_limits': 'OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1',
                   'report_generator_sha256': digest(Path(__file__))}
    save_json(REPORTS / 'environment.json', environment)
    lines = ['# Compact coordinate-carry optimization', '',
             'The final backend uses direct nested loops for rank two and coordinate carry for '
             'other ranks. Only Compact indexing and standard headers changed; both optimizers '
             'use the same improved backend. All paired output hashes are identical.', '',
             '## Direct native copy measurements', '',
             'Five sequential alternating old/new process pairs. Each cell is the median time '
             'over five repetitions; each repetition averages1,000 preallocated native calls. '
             'Python-to-native dispatch is included; output allocation is excluded. Speedup is '
             'the median of within-pair old/new ratios, which need not equal the ratio of medians.', '',
             '| Layout | Old, microseconds/call | Final, microseconds/call | Paired median speedup |',
             '|---|---:|---:|---:|']
    for key in ['transpose128', 'transpose64x128', 'tile128', 'untile128', 'tile784x128', 'slice_offset', 'broadcast']:
        value = summary['compact_' + key]
        lines.append(f"| {key} | {1e6*value['old_seconds']['median']:.3f} | "
                     f"{1e6*value['new_seconds']['median']:.3f} | {value['speedup']['median']:.3f}x |")
    lines += ['', '## Optimizer and short training measurements', '',
              '| Workload | Old, ms/update | Final, ms/update | Paired median speedup |',
              '|---|---:|---:|---:|']
    for key in ('optimizer_adamw', 'optimizer_muon', 'training_adamw', 'training_muon'):
        value = summary[key]
        lines.append(f"| {key} | {1000*value['old_seconds']['median']:.3f} | "
                     f"{1000*value['new_seconds']['median']:.3f} | {value['speedup']['median']:.3f}x |")
    lines += ['', 'Fixed-gradient updates have ten warmup steps followed by100 measured steps. '
              'Training uses the deep Fashion-MNIST encoder, seed8, batch64 and128 updates. '
              'The full training loop includes batch preparation, forward/backward and updates; '
              'it excludes setup, data loading and final feature evaluation. All final parameters, '
              'all128 training losses, and features for128 validation images match bit-for-bit '
              'between versions. Five blocks are timing repetitions of the same seed, not five '
              'independent model-quality trials.', '',
              '![Paired kernel and training measurements](comparison.svg)', '',
              '## Measured refinement: generic carry alone is not universally faster', '',
              'The initial all-ranks carry implementation regressed two-dimensional copies. '
              'The full raw trial and measured native source are retained in generic_carry/. '
              'This motivated direct row/column loops for rank two. The final 2D non-regression '
              'checks all pass. The initial and final trials are separate measurement rounds; '
              'their normalized speedups are each paired against the old backend.', '',
              '| Layout | Initial generic carry vs old | Final implementation vs old |',
              '|---|---:|---:|']
    for key in ('compact_transpose128', 'compact_broadcast', 'compact_slice_offset', 'compact_tile128'):
        lines.append(f"| {key} | {generic['summary'][key]['speedup']['median']:.3f}x | "
                     f"{summary[key]['speedup']['median']:.3f}x |")
    lines += ['', '## Why the optimizer became faster', '',
              'Separate instrumented fixed-gradient diagnostics retain45 matrix products '
              'and157 native compactions per Muon hybrid update in both versions. '
              'The change makes copies cheaper; it does not remove them.', '',
              '| Native operation within Muon | Old, ms/update | Final, ms/update |',
              '|---|---:|---:|']
    for key in ('compact', 'matmul_tiled'):
        lines.append(f"| {key} | {1000*diagnostics['old'][key]:.3f} | {1000*diagnostics['new'][key]:.3f} |")
    lines += ['', 'Instrumented kernel times are descriptive and include timing overhead; primary '
              'speedups use uninstrumented runs. Native matrix algorithms were unchanged. '
              'The generic path adds one O(rank) coordinate vector per call; the two-dimensional '
              'path avoids it. All these costs are included.', '',
              '## Validation, provenance and limits', '',
              '- The rebuilt final backend passes105 project tests, including independent '
              'bit-preservation/source-preservation checks over109 valid layouts, scalar/empty '
              'arrays, signed/zero strides, offsets and six-dimensional cases.',
              f"- Before the edit, old-vs-old control yielded {control['summary']['compact_tile128']['speedup']['median']:.3f}x "
              'on the representative tiled case, correctly failing the predefined1.2x target.',
              '- Compiler/flags are identical across variants; source and binary hashes, '
              'all raw pairs, initial regression results and compiler details are retained.',
              '- Muon/AdamW formulas, learning rates, parameter groups, matmul kernels, '
              'Python tile/untile conversion and quantization are unchanged.',
              '- This establishes execution speed at fixed work. It does not demonstrate '
              'that Muon beats AdamW at equal time, nor provide new retrieval-quality results.',
              f'- Historical training/search reports refer to baseline commit `{BASELINE}`. '
              'Their source-hash validators should be run at their recorded revision.', '',
              '## Reproduction', '', '```bash',
              'python experiments/benchmark_compact_optimization.py --require-speedup 1.2',
              'python experiments/summarize_compact_optimization.py', '```', '',
              'The baseline is compiled from the recorded Git C++ source, while both variants '
              'share the same Python/model/optimizer code. Temporary backend snapshots and '
              'binary/model/data artifacts stay under ignored results/ paths. See '
              '[the durable implementation log](../../docs/project_change_log.md) for the '
              'report-writing context.', '']
    text = '\n'.join(lines)
    for old, new in [('averages1,000', 'averages 1,000'), ('by100', 'by 100'), ('followed by100', 'followed by 100'),
                     ('seed8', 'seed 8'), ('batch64', 'batch 64'), ('and128', 'and 128'), ('all128', 'all 128'),
                     ('for128', 'for 128'), ('retain45', 'retain 45'), ('and157', 'and 157'),
                     ('passes105', 'passes 105'), ('over109', 'over 109'), ('predefined1.2x', 'predefined 1.2x')]:
        text = text.replace(old, new)
    (REPORTS / 'summary.md').write_text(text)
    draw_plot(summary)
    print('Validated final, generic-carry and old-control reports; all output hashes agree.')


if __name__ == '__main__':
    main()
