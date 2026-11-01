"""Validate and report the fixed-wall-clock comparison and separate cost audit."""
import json
import os
from pathlib import Path
import statistics as stats

from run_joint_retrieval import digest, save_json
from run_time_budget_study import validate_timed_pair
from summarize_joint_retrieval import distribution, validate_and_summarize, draw_plot

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/time_budget_study'
ARTIFACTS = ROOT / 'results/time_budget_study'
SEEDS = (8, 9, 10)


def training_summary():
    protocol = json.loads((REPORTS / 'protocol.json').read_text())
    assert {p: digest(ROOT / p) for p in protocol['source_sha256']} == protocol['source_sha256']
    records = {(method, seed): json.loads((REPORTS / f'{method}_seed{seed}.json').read_text())
               for method in ('adamw', 'muon') for seed in SEEDS}
    validations = json.loads((REPORTS / 'model_validation.json').read_text())
    assert len(validations) == 6
    for seed in SEEDS:
        validate_timed_pair(records['adamw', seed], records['muon', seed])
    for (method, seed), row in records.items():
        assert row['source_sha256'] == protocol['source_sha256']
        assert digest(row['best_checkpoint_export']) == row['checkpoint_sha256']
        assert digest(row['checkpoint_export']) == row['terminal_checkpoint_sha256']
        assert row['time_budget_seconds'] == 220 and len(row['checkpoints']) == 4
        assert [r['target_seconds'] for r in row['checkpoints']] == [55, 110, 165, 220]
        assert 0 <= row['budget_overrun_seconds'] <= row['last_step_seconds'] + .001
        chosen = min(enumerate(row['checkpoints']), key=lambda item: (item[1]['validation_loss'], item[0]))
        assert chosen[0] == row['best_checkpoint_index']
        assert chosen[1]['validation_loss'] == row['best_validation_loss']
        reproduced = next(v for v in validations if v['encoder'] == method and v['training_seed'] == seed)
        assert abs(reproduced['validation_reproduced']['loss'] - row['best_validation_loss']) < 1e-7
        assert reproduced['validation_reproduced']['accuracy'] == row['best_validation_accuracy']
    result = {}
    for method in ('adamw', 'muon'):
        rows = [records[method, seed] for seed in SEEDS]
        metrics = {key: distribution([r[key] for r in rows]) for key in
                   ('timed_seconds', 'optimizer_steps', 'samples_seen', 'training_step_seconds',
                    'optimizer_step_seconds', 'best_validation_loss', 'best_validation_accuracy',
                    'optimizer_numeric_state_bytes', 'budget_overrun_seconds')}
        metrics.update(
            effective_epochs=distribution([r['samples_seen'] / r['train_samples'] for r in rows]),
            update_ms=distribution([1000 * r['optimizer_step_seconds'] / r['optimizer_steps'] for r in rows]),
            other_training_ms=distribution([1000 * (r['training_step_seconds'] - r['optimizer_step_seconds'])
                                           / r['optimizer_steps'] for r in rows]),
            selected_snapshot_seconds=distribution([r['checkpoints'][r['best_checkpoint_index']]['snapshot_seconds']
                                                    for r in rows]))
        result[method] = metrics
    return {'protocol': protocol, 'metrics': result, 'records': list(records.values())}


def draw_training(training):
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for index, method in enumerate(('adamw', 'muon')):
        color = ['#2869a6', '#dd7924'][index]
        label = 'AdamW' if method == 'adamw' else 'Muon + auxiliary AdamW'
        rows = [r for r in training['records'] if r['optimizer'] == method]
        for ax, key, scale, title, ylabel in [
            (axes[0, 0], 'validation_loss', 1, 'Validation loss at common time checkpoints', 'Cross-entropy'),
            (axes[0, 1], 'validation_accuracy', 100, 'Validation classification accuracy', 'Percent'),
            (axes[1, 0], 'optimizer_steps', 1, 'Updates completed within the same time', 'Updates')]:
            values = [[scale * r['checkpoints'][i][key] for r in rows] for i in range(4)]
            ax.errorbar([55, 110, 165, 220], [stats.mean(v) for v in values],
                        yerr=[stats.stdev(v) for v in values], color=color, marker='o',
                        capsize=3, label=label)
            ax.set(title=title, xlabel='Elapsed training wall-clock seconds', ylabel=ylabel)
            ax.grid(alpha=.2)
        m = training['metrics'][method]
        rest, update = m['other_training_ms']['mean'], m['update_ms']['mean']
        axes[1, 1].bar(index, rest, color='#a6b9c9', label='Other training work' if index == 0 else None)
        axes[1, 1].bar(index, update, bottom=rest, color='#bd7d4b',
                       label='Optimizer update' if index == 0 else None)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set(title='Uninstrumented retained runs: cost per update', ylabel='Milliseconds',
                    xticks=[0, 1], xticklabels=['AdamW', 'Muon + AdamW'])
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('220 seconds per run; three paired training seeds; error bars = sample SD')
    fig.savefig(REPORTS / 'training.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'training.png', dpi=150)
    plt.close(fig)


def draw_selection(summary):
    import matplotlib.pyplot as plt
    import numpy as np

    chosen, terminal = summary['retrieval'], summary['terminal_fp32']
    labels = ['Validation-loss selected', 'Terminal at 220s']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, method in enumerate(('adamw', 'muon')):
        values = [part['encoders'][method]['fp32']['label_precision_at10']
                  for part in (chosen, terminal)]
        x = np.arange(2) + (index - .5) * .18
        axes[0].errorbar(x, [100*v['mean'] for v in values],
                         yerr=[100*v['sample_std'] for v in values], fmt='o-', capsize=4,
                         color=['#2869a6', '#dd7924'][index],
                         label='AdamW' if method == 'adamw' else 'Muon + auxiliary AdamW')
    axes[0].set(title='FP32 same-category precision@10', ylabel='Percent',
                xticks=[0, 1], xticklabels=labels)
    axes[0].legend(fontsize=8)
    paired = [part['paired_muon_minus_adamw']['fp32'] for part in (chosen, terminal)]
    for index, seed in enumerate(SEEDS):
        axes[1].plot([0, 1], [100*v['values'][index] for v in paired], 'o-',
                     linewidth=1, alpha=.5, label=f'Training seed {seed}')
    axes[1].errorbar([0, 1], [100*v['mean'] for v in paired],
                     yerr=[100*v['sample_std'] for v in paired], color='#684297',
                     fmt='s', capsize=5, label='Mean ± sample SD')
    axes[1].axhline(0, color='gray', linewidth=1)
    axes[1].set(title='Paired Muon minus AdamW', ylabel='Percentage points',
                xticks=[0, 1], xticklabels=labels)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Equal 220s training: checkpoint rule changes the optimizer comparison')
    fig.text(.5, .015, 'Terminal check added after the first selected-model pair; all three seeds, no retrieval-based selection.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .93))
    fig.savefig(REPORTS / 'selection_sensitivity.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'selection_sensitivity.png', dpi=150)
    plt.close(fig)


def write_report(summary):
    training, quality = summary['training'], summary['retrieval']
    chosen_fp32 = quality['paired_muon_minus_adamw']['fp32']
    chosen_turbo2 = quality['paired_muon_minus_adamw']['turbo_2']
    terminal_fp32 = summary['terminal_fp32']['paired_muon_minus_adamw']['fp32']
    lines = ['# Equal-time training and retrieval', '',
             'The current Muon implementation has not demonstrated a robust equal-time retrieval '
             'advantage. Under the predefined classification-loss checkpoint rule, the paired '
             f"Muon minus AdamW difference is {100*chosen_fp32['mean']:+.2f} percentage points "
             f"in FP32 and {100*chosen_turbo2['mean']:+.2f} points with Turbo2. "
             'The secondary terminal-model check reverses FP32: the paired difference becomes '
             f"{100*terminal_fp32['mean']:+.2f} points, with Muon winning "
             f"{sum(v > 0 for v in terminal_fp32['values'])}/3 seeds. "
             'Report both comparisons; classification loss is not a '
             'direct retrieval-selection metric.', '',
             'Both optimizers receive 220 seconds on the native Needle CPU backend. '
             'The model, Fashion-MNIST split, batch 64 and previously tuned learning rates are fixed. '
             'This compares existing configurations, not fully retuned equal-time optima.', '',
             'Intermediate validation/checkpoint costs are inside the budget. Setup and the terminal '
             'evaluation/serialization are outside it for both methods. Updates are indivisible; '
             'actual overruns are recorded. Four common-time candidates at 55/110/165/220 seconds '
             'are selected by minimum validation loss, without retrieval-based selection.', '',
             '| Method | Actual time, s | Updates | Dataset passes | Optimizer, ms/update | Other training, ms/update |',
             '|---|---:|---:|---:|---:|---:|']
    for method in ('adamw', 'muon'):
        m = training['metrics'][method]
        values = [m[k] for k in ('timed_seconds', 'optimizer_steps', 'effective_epochs', 'update_ms', 'other_training_ms')]
        lines.append('| ' + method + ' | ' + ' | '.join(f"{v['mean']:.3f} ± {v['sample_std']:.3f}" for v in values) + ' |')
    lines += ['', 'Means ± sample SD over training seeds 8/9/10. Paired initialization, split and '
              'epoch-order prefixes are verified. Throughput varies across runs on this shared Mac; '
              'method order alternates, and exact completed-step counts remain in the raw records.', '',
              '## Same-category retrieval precision@10', '',
              'Each selected model generates its own gallery and query embeddings. The gallery has '
              '55,000 training-pool images, excluding the 5,000 validation images; queries are the same '
              '1,000 previously explored official-test images. Values below are percentages: the '
              'fraction of ten retrieved pictures sharing the query category.', '',
              '| Gallery store | AdamW | Muon + auxiliary AdamW | Paired Muon − AdamW, pp | Muon wins |',
              '|---|---:|---:|---:|---:|']
    for method in quality['protocol']['methods']:
        a, m = [quality['encoders'][e][method]['label_precision_at10'] for e in ('adamw', 'muon')]
        delta = quality['paired_muon_minus_adamw'][method]
        lines.append(f"| {method} | {100*a['mean']:.3f} ± {100*a['sample_std']:.3f} | "
                     f"{100*m['mean']:.3f} ± {100*m['sample_std']:.3f} | "
                     f"{100*delta['mean']:+.3f} ± {100*delta['sample_std']:.3f} | "
                     f"{sum(v > 0 for v in delta['values'])}/3 |")
    lines += ['', 'Turbo rotations 7/8/9 are averaged within training seed first. They are not nine '
              'independent trained models. Full raw summaries also retain recall of each model\'s '
              'own exact FP32 neighbors, persistent store bytes and query timings. Only gallery '
              'vectors are compressed; queries remain FP32. No query speedup is assumed.', '',
              '## Terminal-model sensitivity check (FP32 only)', '',
              'This secondary diagnostic was added after the first selected-model pair. It '
              'evaluates every terminal 220s model, without choosing models using retrieval. '
              'It checks whether classification-loss checkpoint selection changes the comparison.', '',
              '| Optimizer | Validation-selected FP32, % | Terminal 220s FP32, % |',
              '|---|---:|---:|']
    terminal = summary['terminal_fp32']
    for encoder in ('adamw', 'muon'):
        chosen = quality['encoders'][encoder]['fp32']['label_precision_at10']
        last = terminal['encoders'][encoder]['fp32']['label_precision_at10']
        lines.append(f"| {encoder} | {100*chosen['mean']:.3f} ± {100*chosen['sample_std']:.3f} | "
                     f"{100*last['mean']:.3f} ± {100*last['sample_std']:.3f} |")
    difference = terminal['paired_muon_minus_adamw']['fp32']
    lines += ['', f"Terminal paired Muon − AdamW: {100*difference['mean']:+.3f} ± "
              f"{100*difference['sample_std']:.3f} percentage points; "
              f"Muon wins {sum(v > 0 for v in difference['values'])}/3 seeds.", '',
              '![Checkpoint-selection sensitivity](selection_sensitivity.svg)', '',
              '## Validation-based model selection', '',
              '| Method | Seed | Selected time, s | Validation loss | Classification accuracy, % |',
              '|---|---:|---:|---:|---:|']
    for r in training['records']:
        point = r['checkpoints'][r['best_checkpoint_index']]
        lines.append(f"| {r['optimizer']} | {r['seed']} | {point['snapshot_seconds']:.3f} | "
                     f"{r['best_validation_loss']:.6f} | {100*r['best_validation_accuracy']:.2f} |")
    lines += ['', 'All six selected checkpoints reproduce full-validation loss and accuracy after '
              'reloading. Classification accuracy is separate from retrieval precision. Historical '
              'ten-epoch studies had epoch-boundary checkpoint opportunities; this new experiment '
              'uses a common-time schedule, so differences also reflect checkpoint sampling.', '',
              '## Why Muon costs more per update', '', summary['profile_report'], '',
              '## Reproduction', '', '```bash',
              'python experiments/run_time_budget_study.py --phase all',
              'python experiments/check_terminal_models.py',
              'python experiments/profile_optimizer.py',
              'python experiments/summarize_time_budget_study.py', '```', '',
              'The runner executes retained timings sequentially with CPU math thread limits set '
              'to one. Model/data/feature artifacts remain local and ignored; raw JSON, source '
              'hashes and plots are committed. The findings describe this float32 CPU implementation '
              'and these settings, not optimized bfloat16 GPU training.', '']
    (REPORTS / 'summary.md').write_text('\n'.join(lines))


def main():
    training = training_summary()
    retrieval = validate_and_summarize(REPORTS / 'retrieval')
    save_json(REPORTS / 'retrieval/summary.json', retrieval)
    profile = json.loads((REPORTS / 'optimizer_profile.json').read_text())
    assert {p: digest(ROOT / p) for p in profile['source_sha256']} == profile['source_sha256']
    terminal = json.loads((REPORTS / 'terminal/summary.json').read_text())
    assert digest(ROOT / 'experiments/check_terminal_models.py') == terminal['analysis_driver_sha256']
    assert {p: digest(ROOT / p) for p in terminal['raw_report_sha256']} == terminal['raw_report_sha256']
    summary = {'date': '2026-11-01', 'training': training, 'retrieval': retrieval,
               'terminal_fp32': terminal, 'profile': profile,
               'profile_report': (REPORTS / 'optimizer_profile.md').read_text().partition('\n')[2]}
    save_json(REPORTS / 'summary.json', summary)
    draw_training(training)
    draw_selection(summary)
    draw_plot(retrieval, REPORTS / 'retrieval', ARTIFACTS / 'retrieval')
    write_report(summary)
    print(json.dumps({e: {m: row['label_precision_at10']['mean'] for m, row in data.items()}
                      for e, data in retrieval['encoders'].items()}, indent=2))


if __name__ == '__main__':
    main()
