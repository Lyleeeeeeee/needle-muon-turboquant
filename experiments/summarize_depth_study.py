"""Compare predefined shallow/deep encoders without selecting on retrieval."""
import json
import os
from pathlib import Path
import statistics as stats

from run_depth_study import REPORTS, ARTIFACTS, SOURCES, SEEDS
from run_joint_retrieval import digest, save_json
from run_lr_dataset_study import select_configuration, validate_pair
from summarize_lr_dataset_study import summarize_runs
from summarize_joint_retrieval import distribution, validate_and_summarize, draw_plot

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((REPORTS / 'protocol.json').read_text())
    assert {p: digest(ROOT / p) for p in SOURCES} == protocol['source_sha256']
    selection = json.loads((REPORTS / 'selection.json').read_text())
    for method in ('adamw', 'muon'):
        assert select_configuration(selection['all_candidates'][method]) == selection['selected'][method]
        for candidate in selection['all_candidates'][method]:
            assert digest(ROOT / candidate['record']) == candidate['record_sha256']
    records = {m: [] for m in ('adamw', 'muon')}
    for seed in SEEDS:
        pair = {}
        for method in records:
            candidate = selection['selected'][method]['candidate']
            record = json.loads((REPORTS / f'repeat_{method}_{candidate}_seed{seed}.json').read_text())
            assert record['study']['source_sha256'] == protocol['source_sha256']
            assert record['extra_hidden_layers'] == 2 and record['total_parameter_count'] == 142410
            assert record['muon_parameter_count'] == (40960 if method == 'muon' else 0)
            records[method].append(record)
            pair[method] = record
        validate_pair(pair['adamw'], pair['muon'])
    checked = json.loads((REPORTS / 'model_validation.json').read_text())
    assert len(checked) == 6
    for row in checked:
        record = records[row['encoder']][list(SEEDS).index(row['training_seed'])]
        assert row['checkpoint_sha256'] == record['checkpoint_sha256']
        assert abs(row['validation_reproduced']['loss'] - record['best_validation_loss']) < 1e-7
        assert row['validation_reproduced']['accuracy'] == record['best_validation_accuracy']
        assert row['official_test']['samples'] == 10000

    deep = validate_and_summarize(REPORTS / 'retrieval')
    shallow = json.loads((ROOT / 'reports/joint_retrieval/summary.json').read_text())
    for item in deep['feature_metadata']:
        match = next(r for r in shallow['feature_metadata'] if r['encoder'] == item['encoder']
                     and r['training_seed'] == item['training_seed'])
        for key in ('gallery_ids_sha256', 'query_ids_sha256', 'data_files_sha256'):
            assert item[key] == match[key]
    save_json(REPORTS / 'retrieval/summary.json', deep)
    draw_plot(deep, REPORTS / 'retrieval', ARTIFACTS / 'retrieval')
    training = {m: summarize_runs(rows) for m, rows in records.items()}
    test = {m: {'accuracy': distribution([r['official_test']['accuracy'] for r in checked if r['encoder'] == m]),
                'loss': distribution([r['official_test']['loss'] for r in checked if r['encoder'] == m])}
            for m in records}
    effects = {}
    for method in deep['protocol']['methods']:
        effects[method] = {}
        for encoder in records:
            old = shallow['encoders'][encoder][method]['label_precision_at10']['values']
            new = deep['encoders'][encoder][method]['label_precision_at10']['values']
            effects[method][encoder] = distribution([n - o for o, n in zip(old, new)])
        effects[method]['optimizer_by_depth'] = distribution([
            m - a for a, m in zip(effects[method]['adamw']['values'], effects[method]['muon']['values'])])
    summary = {'date': '2026-11-01', 'protocol': protocol, 'selection': selection,
               'training': training, 'official_classification_test': test,
               'all_validation_reloads_reproduced': True, 'common_shallow_deep_image_ids': True,
               'shallow_reference_summary_sha256': digest(ROOT / 'reports/joint_retrieval/summary.json'),
               'shallow_retrieval': shallow['encoders'], 'deep_retrieval': deep['encoders'],
               'deep_compression_interaction': deep['interaction'], 'depth_effects': effects,
               'paired_deep_muon_minus_adamw': deep['paired_muon_minus_adamw']}
    save_json(REPORTS / 'summary.json', summary)
    draw_depth_plot(summary)
    draw_training(records)
    write_report(summary)
    print('All twelve trainings, six validation reloads and54 deep retrieval workers verified.', flush=True)
    for encoder in records:
        print(encoder, {m: round(v['label_precision_at10']['mean'], 6)
                        for m, v in deep['encoders'][encoder].items()}, flush=True)


def configure_plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    return plt


def draw_depth_plot(summary):
    import numpy as np
    plt = configure_plotting()
    methods = ['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4']
    labels = ['FP32', 'Uniform2', 'Uniform4', 'Turbo2', 'Turbo4']
    x = np.arange(5)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    for i, encoder in enumerate(('adamw', 'muon')):
        color = '#2869a6' if encoder == 'adamw' else '#dd7924'
        for j, (depth, marker) in enumerate([('shallow', 'o'), ('deep', 's')]):
            values = [summary[depth + '_retrieval'][encoder][m]['label_precision_at10'] for m in methods]
            axes[0].errorbar(x + (2*i+j-1.5)*.09, [100*v['mean'] for v in values],
                             yerr=[100*v['sample_std'] for v in values], fmt=marker,
                             capsize=3, color=color, alpha=.6 if j == 0 else 1,
                             label=f'{encoder}: {depth}')
        values = [summary['depth_effects'][m][encoder] for m in methods]
        axes[1].errorbar(x + (i-.5)*.12, [100*v['mean'] for v in values],
                         yerr=[100*v['sample_std'] for v in values], fmt='o', capsize=4,
                         color=color, label=encoder)
    axes[0].set(ylabel='Same-category precision@10 (%)', title='Matched images; fixed shallow and deep model families')
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(ylabel='Deep minus shallow (percentage points)', xticks=x, xticklabels=labels)
    axes[1].axhline(0, color='gray', linewidth=1)
    for ax in axes:
        ax.grid(axis='y', alpha=.25)
    fig.suptitle('Depth comparison: rotation means within paired training seeds 8/9/10; error bars = sample SD')
    fig.savefig(REPORTS / 'depth_comparison.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'depth_comparison.png', dpi=150)
    plt.close(fig)


def draw_training(records):
    import numpy as np
    plt = configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method, runs in records.items():
        for ax, key, scale in [(axes[0], 'validation_loss', 1), (axes[1], 'validation_accuracy', 100)]:
            values = scale * np.array([[r[key] for r in run['epochs']] for run in runs])
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(range(1, 11), mean, marker='o', markersize=3, label=method)
            ax.fill_between(range(1, 11), mean-sd, mean+sd, alpha=.15)
            ax.set_xlabel('Epoch')
            ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation cross-entropy')
    axes[1].set_ylabel('Validation classification accuracy (%)')
    axes[0].legend()
    fig.suptitle('Predefined deeper model: three repeat seeds, selected learning rates')
    fig.savefig(REPORTS / 'training_curves.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'training_curves.png', dpi=150)
    plt.close(fig)


def write_report(summary):
    def percentage(value):
        return f"{100*value['mean']:.2f}% +/- {100*value['sample_std']:.2f}"

    methods = ['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4']
    shallow, deep = summary['shallow_retrieval'], summary['deep_retrieval']
    paired = summary['paired_deep_muon_minus_adamw']['turbo_2']
    seconds_a = summary['training']['adamw']['training_seconds']['mean']
    seconds_m = summary['training']['muon']['training_seconds']['mean']
    finding = (f"Deep Muon+TurboQuant2-bit retrieval averages {100*deep['muon']['turbo_2']['label_precision_at10']['mean']:.2f}%, "
               f"versus {100*shallow['muon']['turbo_2']['label_precision_at10']['mean']:.2f}% for shallow Muon "
               f"and {100*deep['adamw']['turbo_2']['label_precision_at10']['mean']:.2f}% for deep AdamW. "
               f"The paired deep-Muon advantage is {100*paired['mean']:+.3f} percentage points, "
               f"with wins in {sum(x>0 for x in paired['values'])}/3 training seeds. "
               f"Ten-epoch CPU loops take {seconds_m:.2f}s for Muon versus {seconds_a:.2f}s for AdamW "
               f"({seconds_m/seconds_a:.2f}x). These small-sample retrieval gains do not establish "
               'better classification, faster training or a general optimizer ranking.')
    lines = ['# Depth-only Muon and quantized retrieval study', '',
             f"The predefined deeper model has 142,410 parameters, with 40,960 (28.76%) assigned to Muon "
             'versus 7.49% in the shallow model. Uncompressed retrieval category precision@10 is '
             f"{100*deep['adamw']['fp32']['label_precision_at10']['mean']:.2f}% for deep AdamW and "
             f"{100*deep['muon']['fp32']['label_precision_at10']['mean']:.2f}% for deep Muon plus auxiliary AdamW. "
             'The tables retain all seeds, stores and both model depths; no retrieval result selects the setup.', '', finding, '',
             '## Architecture and fixed procedure', '',
             'Old: 784-128-64-10. New: 784-128-128-128-64-10. Two added 128-to128 Linear+ReLU '
             'layers increase depth while width 128 and embedding 64 remain fixed. Muon handles '
             'both added matrices and the 128x64 embedding weight. Input/classifier weights and '
             'all biases use AdamW. This is a plain MLP with the original initialization/loss; '
             'no new normalization, residual connections, dropout or quantizer changes.', '',
             'The [predeclared design](../../docs/depth_study_design.md) fixes Fashion-MNIST: '
             '55,000 training/5,000 validation images (split 17), batch 64, ten epochs, decay 0.01. '
             'Search seed 7 selects minimum validation loss from three configurations per optimizer; '
             'lock choices before repeat seeds 8/9/10. Repeat checkpoints also use minimum validation '
             'loss. Deeper models are retuned with the same candidate budget rather than inheriting '
             'the shallow winner. Both rates in a Muon candidate move together; no independent 2D '
             'search or schedule/decay tuning is performed.', '',
             'The retrieval phase reuses the exact shallow 55,000 gallery and 1,000 query image IDs, '
             'confirmed by data/ID hashes. Each encoder generates its own gallery/query embeddings; '
             'only gallery vectors are compressed. Native cosine top10, blocks of 256, FP32/uniform2/4 '
             'and TurboQuant-MSE2/4; Turbo rotations 7/8/9. Three sequential query timings follow '
             'warmup. Average rotations within training seed, then mean/sample SD over three seeds.', '',
             '## All learning-rate candidates (seed7)', '',
             '| Method | Candidate | AdamW/auxiliary rate | Muon matrix rate | Minimum validation loss | Accuracy at selected checkpoint |',
             '|---|---|---:|---:|---:|---:|']
    for method, candidates in summary['selection']['all_candidates'].items():
        for row in candidates:
            config = summary['protocol']['candidates'][row['candidate']]
            matrix = config['muon_lr'] if method == 'muon' else '-'
            lines.append(f"| {method} | {row['candidate']} | {config['adamw_lr']} | {matrix} | "
                         f"{row['best_validation_loss']:.6f} | {100*row['best_validation_accuracy']:.2f}% |")
    lines += ['', '## Selected deep training and classification results', '',
              '| Method | Candidate | Validation accuracy | Official test accuracy | Ten-epoch CPU training seconds | Numeric state bytes | Selected epochs |',
              '|---|---|---:|---:|---:|---:|---|']
    for encoder in ('adamw', 'muon'):
        train = summary['training'][encoder]
        test = summary['official_classification_test'][encoder]
        seconds = train['training_seconds']
        lines.append(f"| {encoder} | {summary['selection']['selected'][encoder]['candidate']} | "
                     f"{percentage(train['validation_accuracy'])} | {percentage(test['accuracy'])} | "
                     f"{seconds['mean']:.2f} +/- {seconds['sample_std']:.2f} | "
                     f"{train['numeric_state_bytes'][0]:,} | {train['selected_epochs']} |")
    lines += ['', 'Classification is separate from retrieval. Training-loop time excludes loading, '
              'validation and checkpoint I/O. Numeric state excludes parameters, gradients, Python '
              'objects and transient Newton-Schulz workspace; it is not peak memory. Three seeds '
              'are not confidence intervals. A grid endpoint or late selected epoch signals remaining '
              'tuning/budget uncertainty; no extra candidates/epochs were added after outcomes.', '',
              '![Deep validation curves](training_curves.svg)', '',
              '## Primary retrieval result: fraction of top10 sharing query category', '',
              '| Store | Shallow AdamW | Shallow Muon | Deep AdamW | Deep Muon |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        values = [shallow['adamw'][method], shallow['muon'][method], deep['adamw'][method], deep['muon'][method]]
        lines.append('| ' + method + ' | ' + ' | '.join(percentage(v['label_precision_at10']) for v in values) + ' |')
    lines += ['', 'This category-level metric counts retrieved true labels; it is not classification '
              'accuracy or exact-neighbor recall. Models use a classification objective rather than '
              'metric-learning training. Parameter count/compute and later-layer initialization differ '
              'between depths, so this is a model-family comparison, not isolated proof of a Muon '
              'coverage mechanism.', '', '![Paired depth effects](depth_comparison.svg)', '',
              '## Paired effects of added depth', '',
              'Each difference uses matching training seed and image IDs. Positive optimizer-by-depth '
              'means the deeper architecture changes Muon\'s relative category-precision gap versus '
              'AdamW favorably. It does not prove why, or establish general superiority.', '',
              '| Store | Deep-shallow AdamW (pp) | Deep-shallow Muon (pp) | Optimizer-by-depth (pp) |',
              '|---|---:|---:|---:|']
    for method in methods:
        effect = summary['depth_effects'][method]
        values = [effect['adamw'], effect['muon'], effect['optimizer_by_depth']]
        lines.append('| ' + method + ' | ' + ' | '.join(f"{100*v['mean']:+.3f} +/- {100*v['sample_std']:.3f}" for v in values) + ' |')
    lines += ['', '## Deep compression-induced changes and neighbor fidelity', '',
              'Compression changes are relative to each deep encoder\'s own FP32 category precision. '
              'Positive changes are retained. Neighbor recall instead measures overlap with that '
              'same encoder\'s exact FP32 top10. A lower compression loss can coexist with a lower '
              'absolute category precision.', '',
              '| Store | AdamW category change (pp) | Muon category change (pp) | AdamW own-FP32 recall | Muon own-FP32 recall |',
              '|---|---:|---:|---:|---:|']
    for method in methods:
        a, m = deep['adamw'][method], deep['muon'][method]
        lines.append(f"| {method} | {100*a['precision_change_vs_own_fp32']['mean']:+.3f} | "
                     f"{100*m['precision_change_vs_own_fp32']['mean']:+.3f} | "
                     f"{percentage(a['neighbor_recall_at10'])} | {percentage(m['neighbor_recall_at10'])} |")
    lines += ['', '![Deep-only retrieval diagnostics](retrieval/comparison.svg)', '',
              '## Storage and query time', '',
              '| Store | Gallery MB (all depths/encoders) | Deep AdamW 1,000-query seconds | Deep Muon 1,000-query seconds |',
              '|---|---:|---:|---:|']
    for method in methods:
        a, m = deep['adamw'][method], deep['muon'][method]
        lines.append(f"| {method} | {a['stored_bytes']['mean']/1e6:.4f} | {a['query_batch_seconds']['mean']:.3f} | "
                     f"{m['query_batch_seconds']['mean']:.3f} |")
    lines += ['', 'MB uses 1,000,000 bytes. Index size is fixed by 55,000x64 vectors and format, '
              'including persistent quantizer metadata, not model weights/images/queries/labels. '
              'Adding model layers cannot by itself shrink this fixed-format gallery. Query time '
              'includes normalization/rotation/block decode/native products/top-k, excluding '
              'embedding generation and construction. Raw rows retain all three timings, build '
              'cost, MSE and whole-worker peak RSS. Historical shallow timings were measured '
              'earlier on the same CPU, not interleaved with this study; do not read small temporal '
              'differences as depth-dependent search acceleration.', '',
              '## Validation, limitations and reproduction', '',
              'All six deep checkpoints reload to reproduce full 5,000-image validation scores. '
              'All 54 method rows verify checkpoint/archive/source provenance; common image IDs '
              'match the shallow study exactly. Shallow defaults, training/feature arrays and '
              'version1 checkpoint arrays reproduce exactly after adding the opt-in depth flag. '
              'Version2 stores/restores every extra layer. Original homework archives and '
              'optimizer/quantizer formulas remain unchanged.', '',
              'Fashion official test images were already viewed in the earlier studies. This is '
              'a predefined exploratory depth extension; no untouched-test claim or post-result '
              'best-depth selection. More classes/different data is a separate experiment. '
              'The current data is clothing 10 categories, not handwritten digits. Native CPU '
              'float32 findings do not establish large-model/GPU speedups. Retained optimizer '
              'state reductions do not establish lower peak process memory.', '',
              '```bash', 'python -m pytest -q', 'python experiments/run_depth_study.py',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_depth_study.py',
              '```', '', 'A fresh full reproduction requires the earlier data and shallow-model '
              'records/artifacts. Archive published depth output before regenerating; source hashes '
              'and model checkpoints must match to resume. See historical commits for the original '
              'shallow protocol/source versions. All data/feature/checkpoint binaries stay local '
              'and ignored. [Summary JSON](summary.json), [training protocol](protocol.json) and '
              '[retrieval summary/per-worker rows](retrieval/summary.json) retain all measurements.', '']
    (REPORTS / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
