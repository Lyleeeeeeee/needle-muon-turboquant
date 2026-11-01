"""Validate shared Matmul ablation and produce English report artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from benchmark_shared_matmul import ROOT, REPORTS, ARTIFACTS, digest, save_json, summarize_pairs


def draw_plot(summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = [(axes[0], ['matmul_forward_input', 'matmul_backward_weight',
                         'matmul_backward_activation', 'matmul_hidden_square'],
               ['Input fwd', 'Weight grad', 'Activation grad', '128 square'], 'Shared public matrix products'),
              (axes[1], ['training_adamw', 'training_muon'],
               ['AdamW', 'Muon + aux AdamW'], 'Complete 128-update training loop'),
              (axes[2], ['optimizer_adamw', 'optimizer_muon'],
               ['AdamW', 'Muon + aux AdamW'], 'Optimizer-only controls')]
    for ax, keys, labels, title in panels:
        for index, variant in enumerate(('old', 'new')):
            values = [summary[k][variant + '_seconds'] for k in keys]
            ax.bar(np.arange(len(keys)) + (index - .5) * .34,
                   [1000 * v['mean'] for v in values], width=.34,
                   yerr=[1000 * v['sample_std'] for v in values], capsize=3,
                   color=['#2869a6', '#dd7924'][index],
                   label=['Reference wrapper', 'Direct view packing'][index])
        ax.set(title=title, ylabel='Milliseconds per call / update',
               xticks=range(len(keys)), xticklabels=labels)
        ax.tick_params(axis='x', labelrotation=12)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle('Shared CPU Matmul optimization | native Muon NS enabled in both arms\n'
                 'Five sequential paired blocks, one training seed; bars = mean, error bars = sample SD')
    path = REPORTS / 'comparison.svg'
    fig.savefig(path, metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'comparison.png', dpi=150)
    plt.close(fig)
    # Strip generator whitespace without changing XML tags, attributes or text tokens.
    before = path.read_text()
    after = '\n'.join(line.rstrip() for line in before.splitlines()) + '\n'
    def geometry(text):
        return [(n.tag, {k: v.split() for k, v in n.attrib.items()},
                 (n.text or '').split(), (n.tail or '').split())
                for n in ET.fromstring(text).iter()]
    assert geometry(before) == geometry(after)
    path.write_text(after)


def main():
    report = json.loads((REPORTS / 'summary.json').read_text())
    protocol = report['protocol']
    assert {p: digest(ROOT / p) for p in protocol['source_sha256']} == protocol['source_sha256']
    assert {p: digest(ROOT / 'data/fashion-mnist' / p) for p in protocol['data_files_sha256']} == protocol['data_files_sha256']
    rows = []
    for path, expected in report['raw_report_sha256'].items():
        assert digest(REPORTS / path) == expected
        rows.append(json.loads((REPORTS / path).read_text()))
    assert len(rows) == 2 * protocol['blocks'] == 10
    assert len({r['native_extension_sha256'] for r in rows}) == 1
    assert any(digest(p) == rows[0]['native_extension_sha256'] for p in
               (ROOT / 'python/needle/backend_ndarray').glob('ndarray_backend_cpu*.so'))
    summary = summarize_pairs(rows)
    assert summary == report['summary']
    for key in summary:
        assert len({r['metrics'][key]['output_sha256'] for r in rows}) == 1
        for field in rows[0]['metrics'][key]:
            if field.endswith('sha256'):
                assert len({r['metrics'][key][field] for r in rows}) == 1
    previous = json.loads((ROOT / 'reports/muon_tiled/block0_new.json').read_text())
    for key in ('training_adamw', 'training_muon'):
        for field in ('initial_parameter_sha256', 'train_indices_sha256',
                      'training_loss_sha256', 'feature_sha256'):
            assert rows[0]['metrics'][key][field] == previous['metrics'][key][field]
    layouts = {}
    for name in ('adamw', 'muon'):
        layouts[name] = {}
        for variant in ('old', 'new'):
            values = [r['diagnostics'][name] for r in rows if r['variant'] == variant]
            assert all(v == values[0] for v in values)
            layouts[name][variant] = values[0]
            assert values[0]['public_products'] == 15 and values[0]['naive_calls'] == 3
            assert values[0]['native_ns_calls'] == (3 if name == 'muon' else 0)
            assert values[0]['strided_calls'] == (12 if variant == 'new' else 0)
            assert values[0]['tiled_calls'] == (12 if variant == 'old' else 0)
        assert layouts[name]['old']['post_parameters_sha256'] == layouts[name]['new']['post_parameters_sha256']
        assert (layouts[name]['old']['layout_written_bytes_excluding_native_ns'] -
                layouts[name]['new']['layout_written_bytes_excluding_native_ns']) == 864256
        assert (layouts[name]['old']['layout_passes_excluding_native_ns'] -
                layouts[name]['new']['layout_passes_excluding_native_ns']) == 8
    environment = {
        'compiler': subprocess.check_output([shutil.which('clang++') or shutil.which('g++'), '--version'], text=True),
        'thread_limits': 'OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1',
        'report_generator_sha256': digest(Path(__file__)), 'layout_work_per_update': layouts,
        'unchanged_muon_internal_layout': {
            'passes': 21, 'written_bytes': 1146880,
            'basis': 'three normalized matrices128x128,128x128,64x128;pack+5 transposes+unpack each'},
        'validation': 'ten raw/source/data/binary hashes;exact outputs;prior-stage loss/feature reproduction'}
    save_json(REPORTS / 'environment.json', environment)
    lines = ['# Shared stride-aware CPU Matmul', '',
             'Direct native packing from ordinary or transposed views speeds up the shared '
             'forward/backward path used by both optimizers. The existing Compact optimization '
             'and native tiled Muon NS path are enabled in both arms. MatmulTiled/AlignedDot '
             'arithmetic and optimizer formulas are unchanged.', '',
             '## Paired timing', '',
             'Five alternating sequential process pairs use the same final CPU binary. '
             'Columns are medians of five per-process average times; speedup is the median '
             'within-pair ratio. Full distributions, means and sample SD are in summary.json. '
             'Five blocks are timing repetitions, not independent model-quality seeds.', '',
             '| Workload | Reference, ms | Direct packing, ms | Paired median speedup |',
             '|---|---:|---:|---:|']
    for key, r in summary.items():
        lines.append(f"| {key} | {1000*r['old_seconds']['median']:.3f} | "
                     f"{1000*r['new_seconds']['median']:.3f} | {r['speedup']['median']:.3f}x |")
    lines += ['', '![Shared product and training timing](comparison.svg)', '',
              'Microbenchmarks perform10 warmup and200 measured calls. Optimizer controls '
              'perform10 warmup and80 measured updates; neither optimizer-only path uses '
              'public Matmul with native NS enabled. Training uses128 updates, batch64, '
              'seed8, the same55k/5k split (seed17), architecture784-128-128-128-64-10, '
              'AdamW/auxiliary LR.001 and Muon LR.007. Timing includes batch/Tensor '
              'preparation, loss reads, forward/backward and updates; excludes loading, '
              'model setup, final embedding export and a separate instrumented audit update.', '',
              'Both paths produce bit-identical final parameters, all128 training losses '
              'and128 validation embeddings. Initial state, split, loss and feature hashes '
              'also reproduce the prior tiled-NS experiment. The separate audit update '
              'preserves exact parameter parity. This is execution-speed evidence; the '
              '220-second retrieval-quality comparison has not been rerun.', '',
              '## Data layout work', '',
              'The reference first compacts noncontiguous operands, packs each input through '
              'a generic four-dimensional Compact, multiplies, then unpacks through another '
              'generic Compact. The new operator packs directly from each source view, '
              'including transpose/slicing/offset, uses the same tiled multiplication, '
              'and directly unpacks an ordinary output. Already contiguous operands still '
              'need packing: their gain comes from specialized copy loops and fewer Python '
              'objects/dispatches. This experiment measures these effects together.', '',
              'An untimed training update after the128 timed updates observes15 public '
              'products in both arms:12 aligned products use the new path;3 classifier '
              'products retain the naive fallback. Muon additionally retains3 native NS '
              'calls (45 logical internal products). No matrix products are removed.', '',
              '| Work per complete training update | AdamW reference | AdamW direct | Muon reference | Muon direct |',
              '|---|---:|---:|---:|---:|']
    for label, key in [('Generic Compact calls', 'compact_calls'),
                       ('Internal public-Matmul layout passes', 'strided_layout_passes'),
                       ('Total layout passes excluding native NS', 'layout_passes_excluding_native_ns')]:
        values = [layouts[n][v][key] for n in ('adamw', 'muon') for v in ('old', 'new')]
        lines.append('| ' + label + ' | ' + ' | '.join(str(v) for v in values) + ' |')
    values = [layouts[n][v]['layout_written_bytes_excluding_native_ns'] / 1e6
              for n in ('adamw', 'muon') for v in ('old', 'new')]
    lines.append('| Layout writes excluding native NS, decimal MB | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', 'Generic Compact counts/elements are instrumented. Internal public-Matmul '
              'passes/elements are derived from dimensions (pack A, pack B, unpack C). '
              'The apparent44-call reduction in generic Compact includes36 copies moved '
              'inside C++; only8 full-matrix layout passes disappear. Each optimizer saves '
              '864,256 bytes of layout writes per training update. These counts are not '
              'physical DRAM traffic or whole-process peak memory.', '',
              'The unchanged Muon NS path adds21 internal layout passes and1,146,880 bytes '
              'to both Muon columns. Including these, complete Muon layout passes are91→83 '
              'and layout writes5.509632→4.645376 MB. These writes have different sizes.', '',
              'Three temporary packed arrays hold MN+NP+MP float32 elements per aligned '
              'public product and are freed after the call. No persistent packed cache '
              'or new optimizer state is added.', '',
              '## Scope and reproduction', '',
              '- Default `NEEDLE_CPU_MATMUL=strided` supports positive tile-aligned native '
              'CPU products. `NEEDLE_CPU_MATMUL=reference` uses the original course wrapper. '
              'Set the environment before importing Needle. Older native extensions, other '
              'devices and unaligned sizes fall back automatically.',
              '- The reference switch does not disable the separate Muon NS optimization. '
              'Earlier-stage benchmark workers explicitly select the reference shared path '
              'so those ablations remain isolated.',
              '- The full suite passes148 checks, including independent float64 product/gradient '
              'oracles, exact old-path parity, source preservation, signed/zero strides, '
              'offsets, invalid buffers, fallback and profiling accounting.',
              '- These host-specific small-MLP CPU results do not establish a GPU speedup '
              'or a Muon equal-time retrieval advantage. Optimizer-only controls and the '
              'unaligned classifier are retained to expose unrelated timing variation.', '',
              '```bash', 'python tools/build_cpu.py', 'PYTHONPATH=python:apps python -m pytest tests -q',
              'python experiments/benchmark_shared_matmul.py --blocks 5',
              'python experiments/summarize_shared_matmul.py', '```', '',
              'Source/binary/data hashes, ten raw reports and all timing distributions are '
              'retained beside this report. Historical experiments remain tied to their '
              'recorded revisions. See [design](../../docs/shared_matmul_design.md) and '
              '[change log](../../docs/project_change_log.md).', '']
    text = '\n'.join(lines)
    for before, after in [('perform10', 'perform 10'), ('and200', 'and 200'), ('and80', 'and 80'),
                          ('uses128', 'uses 128'), ('batch64', 'batch 64'), ('seed8', 'seed 8'),
                          ('same55k/5k', 'same 55k/5k'), ('seed17', 'seed 17'), ('architecture784', 'architecture 784'),
                          ('LR.001', 'LR .001'), ('LR.007', 'LR .007'), ('all128', 'all 128'), ('and128', 'and 128'),
                          ('the128', 'the 128'), ('observes15', 'observes 15'), (':12', ': 12'), (';3', '; 3'),
                          ('retains3', 'retains 3'), ('apparent44', 'apparent 44'), ('includes36', 'includes 36'),
                          ('only8', 'only 8'), ('adds21', 'adds 21'), ('and1,146', 'and 1,146'),
                          ('are91', 'are 91'), ('writes5.', 'writes 5.'), ('passes148', 'passes 148')]:
        text = text.replace(before, after)
    (REPORTS / 'summary.md').write_text(text)
    draw_plot(summary)
    print('Validated ten paired reports, exact trajectory parity, prior-stage reproduction and layout accounting.')


if __name__ == '__main__':
    main()
