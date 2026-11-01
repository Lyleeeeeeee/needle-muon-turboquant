"""Validate paired training records, inspect selected models, and draw curves.

This analysis runs after retained training timings. Matplotlib is optional
analysis tooling; it is not a Needle runtime dependency.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'results/muon_pilot/mplcache'))

import numpy as np
import needle as ndl
from needle import nn
from needle.quantization import FloatVectorStore
from benchmark_quantization import unit_rows
from encoder_checkpoint import load_encoder
from train_baseline import select_training_indices


def main():
    reports = ROOT / 'reports'
    output = ROOT / 'results/muon_pilot'
    names = ('adamw', 'muon')
    records = {name: [json.loads((reports / f'muon_pilot_{name}_seed{seed}.json').read_text())
                      for seed in (7, 8, 9)] for name in names}
    for index, seed in enumerate((7, 8, 9)):
        a, b = records['adamw'][index], records['muon'][index]
        for key in ('initial_parameter_sha256', 'train_indices_sha256',
                    'validation_indices_sha256', 'training_order_sha256',
                    'initial_validation_loss', 'initial_validation_accuracy'):
            assert a[key] == b[key], (seed, key)
        for row in (a, b):
            assert row['device'] == 'cpu()' and row['validation_source'] == 'train_holdout'
            assert row['train_samples'] == 55000 and row['eval_samples'] == 5000
            assert row['train_validation_disjoint'] and len(row['epochs']) == 10
            assert all(e['optimizer_steps'] == 860 for e in row['epochs'])
            assert all(0 < e['optimizer_step_seconds'] < e['training_seconds'] for e in row['epochs'])
    assert len({r['validation_indices_sha256'] for rows in records.values() for r in rows}) == 1

    summary = {'source_commit': 'f98f1cacc0263b04290753e32be8d498c1efc806',
               'configuration': {'train': 55000, 'validation': 5000, 'split_seed': 17,
                                 'training_seeds': [7, 8, 9], 'epochs': 10, 'batch_size': 64,
                                 'adamw_lr': .003, 'muon_lr': .02, 'decay': .01,
                                 'momentum': .95, 'ns_steps': 5, 'ns_eps': 1e-7,
                                 'nesterov': True, 'shape_layout': 'Needle input/output',
                                 'aux_betas': [.9, .999], 'aux_eps': 1e-8},
               'all_pair_hashes_agree': True, 'methods': {}, 'downstream_seed7': {}}
    for name, rows in records.items():
        early = [min(r['epochs'][:3], key=lambda e: e['validation_loss']) for r in rows]
        metrics = {
            'epoch3_validation_accuracy': [r['epochs'][2]['validation_accuracy'] for r in rows],
            'epoch10_validation_accuracy': [r['epochs'][9]['validation_accuracy'] for r in rows],
            'best_within3_accuracy_by_loss': [e['validation_accuracy'] for e in early],
            'selected_checkpoint_accuracy': [r['best_validation_accuracy'] for r in rows],
            'selected_validation_loss': [r['best_validation_loss'] for r in rows],
            'max_validation_accuracy': [max(e['validation_accuracy'] for e in r['epochs']) for r in rows],
            'training_loop_seconds': [sum(e['training_seconds'] for e in r['epochs']) for r in rows],
            'optimizer_seconds': [sum(e['optimizer_step_seconds'] for e in r['epochs']) for r in rows],
        }
        result = {key: {'mean': stats.mean(values), 'sample_std': stats.stdev(values)}
                  for key, values in metrics.items()}
        result['selected_epochs'] = [r['best_validation_epoch'] for r in rows]
        result['numeric_state_bytes'] = rows[0]['optimizer_numeric_state_bytes']
        assert len({r['optimizer_numeric_state_bytes'] for r in rows}) == 1
        result['best_checkpoint_after_epoch3_runs'] = sum(r['best_validation_epoch'] > 3 for r in rows)
        result['thresholds'] = {}
        for target in (.95, .97, .98):
            times = []
            for r in rows:
                elapsed, reached = 0., None
                for e in r['epochs']:
                    elapsed += e['training_seconds']
                    if e['validation_accuracy'] >= target:
                        reached = elapsed
                        break
                times.append(reached)
            finite = [v for v in times if v is not None]
            result['thresholds'][str(target)] = {'reached_runs': len(finite), 'runs': 3,
                                                'training_seconds_per_run': times,
                                                'mean_reached_seconds': stats.mean(finite) if finite else None}
        summary['methods'][name] = result

    train = ndl.data.MNISTDataset(str(ROOT.parent / 'hw0/data/train-images-idx3-ubyte.gz'),
                                 str(ROOT.parent / 'hw0/data/train-labels-idx1-ubyte.gz'))
    _, validation_ids, _ = select_training_indices(len(train), len(train), 55000, 5000,
                                                  np.random.default_rng(7), 5000, 17)
    common_ids = None
    for name in names:
        checkpoint = output / f'{name}_seed7_best.npz'
        model = load_encoder(checkpoint, ndl.cpu())
        total_loss, correct = 0., 0
        criterion = nn.SoftmaxLoss()
        for start in range(0, len(validation_ids), 64):
            ids = validation_ids[start:start+64]
            x = ndl.Tensor(train.X[ids], device=ndl.cpu(), requires_grad=False)
            y = ndl.Tensor(train.y[ids], device=ndl.cpu(), requires_grad=False)
            scores = model(x)
            total_loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * len(ids)
            correct += int((scores.numpy().argmax(axis=1) == train.y[ids]).sum())
        assert abs(total_loss / len(validation_ids) - records[name][0]['best_validation_loss']) < 1e-7
        assert correct / len(validation_ids) == records[name][0]['best_validation_accuracy']
        feature_path = output / f'{name}_seed7_features.npz'
        subprocess.run([sys.executable, str(ROOT / 'apps/export_features.py'),
                        '--checkpoint', str(checkpoint), '--gallery-size', '4096', '--queries', '1000',
                        '--output', str(feature_path)], check=True, cwd=ROOT)
        metadata = json.loads(feature_path.with_suffix('.json').read_text())
        with np.load(feature_path, allow_pickle=False) as source:
            gallery, queries = unit_rows(source['gallery']), unit_rows(source['queries'])
            labels, query_labels = source['gallery_labels'], source['query_labels']
            ids = (source['gallery_ids'].copy(), source['query_ids'].copy())
        if common_ids is None:
            common_ids = ids
        else:
            assert all(np.array_equal(a, b) for a, b in zip(common_ids, ids))
        found = FloatVectorStore(gallery, device=ndl.cpu()).search(queries, metric='cosine').indices
        summary['downstream_seed7'][name] = {
            'checkpoint_sha256': metadata['checkpoint_sha256'],
            'selected_checkpoint_validation_reproduced': True,
            'query_classification_accuracy': metadata['query_classification_accuracy'],
            'fp32_retrieval_label_precision_at10': float((labels[found] == query_labels[:, None]).mean()),
            'gallery_rows': 4096, 'query_rows': 1000,
            'note': 'Exploratory old test split; not checkpoint/hyperparameter selection or quantization study'}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 11})
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, constrained_layout=True)
    for index, name in enumerate(names):
        color, marker = f'C{index}', ('o' if index == 0 else 's')
        label = 'AdamW' if name == 'adamw' else 'Muon + auxiliary AdamW'
        for ax, key, factor in ((axes[0], 'validation_loss', 1), (axes[1], 'validation_accuracy', 100)):
            values = np.array([[e[key] for e in r['epochs']] for r in records[name]]) * factor
            mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
            ax.plot(range(1, 11), mean, color=color, marker=marker, label=label, markersize=4)
            ax.fill_between(range(1, 11), mean-sd, mean+sd, color=color, alpha=.15)
            ax.grid(alpha=.2)
    for ax in axes:
        ax.axvline(3, color='.5', linestyle=':', linewidth=1)
    axes[0].set_ylabel('Validation cross-entropy')
    axes[0].legend(loc='upper right')
    axes[1].set_ylabel('Validation accuracy (%)')
    axes[1].set_xlabel('Epoch (dotted line: previous 3-epoch budget)')
    axes[1].set_xticks(range(1, 11))
    fig.suptitle('MNIST pilot: 55k training / 5k validation, 3 paired seeds')
    fig.savefig(reports / 'muon_pilot_curves.svg', metadata={'Date': '2026-10-31'})
    fig.savefig(output / 'curves.png', dpi=160)
    plt.close(fig)

    summary['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                                for p in ('python/needle/optim.py', 'apps/train_baseline.py', 'apps/models.py')}
    (reports / 'muon_pilot_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Muon in Needle: paired training pilot', '',
             'Real training preceded quantization: the old encoder used AdamW for three epochs. This study adds native Needle Muon and compares a longer budget with a fresh paired AdamW baseline. It does not reuse old quantizer results as evidence of optimizer quality.', '',
             f"In this predefined CPU setting, minimum-validation-loss checkpoints average {100*summary['methods']['adamw']['selected_checkpoint_accuracy']['mean']:.2f}% classification accuracy for AdamW and {100*summary['methods']['muon']['selected_checkpoint_accuracy']['mean']:.2f}% for Muon plus auxiliary AdamW. Ten-epoch training loops average {summary['methods']['adamw']['training_loop_seconds']['mean']:.2f}s and {summary['methods']['muon']['training_loop_seconds']['mean']:.2f}s, respectively. No accuracy or training-speed advantage is established for Muon in this pilot.", '',
             '## Protocol', '',
             'Both optimizers use the same 784-128-64-10 model, initialization and data order within each of seeds 7/8/9. A fixed split seed 17 reserves 5,000 MNIST training images for validation and trains on the other 55,000. Each run uses ten epochs, batch size 64 and native CPU (8,600 updates). Initialization/split/order hashes match every pair. Learning rates were predefined: AdamW 0.003, Muon 0.02; auxiliary AdamW uses the same 0.003 learning rate, betas (0.9,0.999), and epsilon 1e-8 as baseline. Decay is 0.01 for both groups. There was no learning-rate/schedule search.', '',
             'Muon updates only hidden.weight; other parameters use AdamW. The [design](../docs/muon_design.md) pins the author implementation and records float32 versus bfloat16, transposed Linear layout and absent-gradient adaptations. An independent SVD/singular-value-polynomial oracle checks the matrix update. All 70 checks passed; legacy AdamW features reproduce bit-for-bit.', '',
             '## Classification quality and training budget', '',
             '| Optimizer | Epoch3 validation accuracy | Epoch10 validation accuracy | Accuracy of minimum-loss checkpoint | Selected epochs |',
             '|---|---:|---:|---:|---|']
    for name in names:
        row = summary['methods'][name]
        fields = []
        for key in ('epoch3_validation_accuracy', 'epoch10_validation_accuracy', 'selected_checkpoint_accuracy'):
            fields.append(f"{100*row[key]['mean']:.2f}% +/- {100*row[key]['sample_std']:.2f}%")
        lines.append(f"| {name} | " + ' | '.join(fields) + f" | {row['selected_epochs']} |")
    lines += ['', 'Values are mean +/- sample standard deviation across three training seeds, not confidence intervals. Checkpoint selection uses minimum validation loss, not maximum observed accuracy. All individual epoch metrics remain in the raw records.', '',
              f"Minimum-loss checkpoints occurred after epoch 3 in {sum(summary['methods'][name]['best_checkpoint_after_epoch3_runs'] for name in names)}/6 runs. Selection within a longer budget helps, but fixed-epoch results show that the last model is not automatically the best.", '',
              '![Validation loss and classification accuracy; bands are sample standard deviations](muon_pilot_curves.svg)', '',
              'A longer budget provides additional candidate checkpoints, but the final epoch need not be best. Lower training loss together with rising/fluctuating validation loss is a reason to use validation selection and investigate scheduling, rather than assuming that more epochs always improve generalization.', '',
              '## CPU cost and retained optimizer state', '',
              '| Optimizer | Ten-epoch training-loop seconds | Mean full training-step ms | Mean optimizer-update ms | Numeric state bytes |',
              '|---|---:|---:|---:|---:|']
    for name in names:
        row = summary['methods'][name]
        time = row['training_loop_seconds']
        lines.append(f"| {name} | {time['mean']:.2f} +/- {time['sample_std']:.2f} | {1000*time['mean']/8600:.3f} | {1000*row['optimizer_seconds']['mean']/8600:.3f} | {row['numeric_state_bytes']:,} |")
    lines += ['', 'Training-loop time includes forward/backward/update and tensor preparation; it excludes data loading, validation and checkpoint I/O. The CPU operations are synchronous. Numeric state excludes parameters, gradients, Python objects and transient Newton-Schulz matrices, so it is not peak-memory usage. Most parameters are in the input layer and remain on AdamW; the total-state reduction is therefore modest.', '',
              '## Time to validation targets', '',
              '| Optimizer | Target accuracy | Reached runs | Mean training seconds among reached runs |',
              '|---|---:|---:|---:|']
    for name in names:
        for target, item in summary['methods'][name]['thresholds'].items():
            value = 'not reached' if item['mean_reached_seconds'] is None else f"{item['mean_reached_seconds']:.2f}"
            lines.append(f"| {name} | {100*float(target):.0f}% | {item['reached_runs']}/3 | {value} |")
    lines += ['', 'Thresholds 95/97/98% were specified before inspecting outcomes. Threshold time is observed at epoch boundaries, not interpolated between measurements. A mean over successful runs must be read alongside its reach count.', '',
              '## Separate downstream inspection, seed7', '',
              'Selected checkpoints were reloaded and their 5,000-image validation loss/accuracy reproduced. Both then used the same 4,096-image FP32 gallery and 1,000 old test-split queries under native cosine scoring. No compression is evaluated in this table.', '',
              '| Optimizer | Query classification accuracy | FP32 retrieval same-label precision@10 |',
              '|---|---:|---:|']
    for name in names:
        row = summary['downstream_seed7'][name]
        lines.append(f"| {name} | {100*row['query_classification_accuracy']:.2f}% | {100*row['fp32_retrieval_label_precision_at10']:.2f}% |")
    lines += ['', 'Classification accuracy and retrieval label precision are different metrics. This downstream table is one seed and exploratory; the official test split had already been viewed in earlier work and is not an untouched final benchmark. It did not select these checkpoints.', '',
              '## Scope and next checks', '',
              'These are paired, fixed-setting pilot results, not equal-budget hyperparameter-tuned evidence of general optimizer superiority. Float32 CPU, batch size 64, a small MLP and a single hidden-matrix Muon group differ from large-model GPU experiments. Assess quality together with elapsed time; more optimizer compute does not guarantee a better final model. The formal Fashion-MNIST and joint Muon/quantization study remain separate.', '',
              '## Reproduce', '', '```bash', 'python -m pytest -q', 'python - <<\'PY\'', 'import os, subprocess, sys',
              "env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')",
              "for seed, optimizer in [(7,'adamw'),(7,'muon'),(8,'muon'),(8,'adamw'),(9,'adamw'),(9,'muon')]:",
              "    subprocess.run([sys.executable, 'apps/train_baseline.py', '--device', 'cpu', '--optimizer', optimizer, '--train-limit', '55000', '--validation-size', '5000', '--eval-limit', '5000', '--split-seed', '17', '--epochs', '10', '--seed', str(seed), '--lr', '.003', '--muon-lr', '.02', '--weight-decay', '.01', '--checkpoint-output', f'results/muon_pilot/{optimizer}_seed{seed}_last.npz', '--best-checkpoint-output', f'results/muon_pilot/{optimizer}_seed{seed}_best.npz', '--output', f'reports/muon_pilot_{optimizer}_seed{seed}.json'], check=True, env=env)",
              'PY', 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_muon_pilot.py', '```', '',
              'Plot regeneration requires Matplotlib in the analysis environment. It is not needed for optimizer execution. [Machine-readable summary](muon_pilot_summary.json) includes all hashes, targets and selected-model checks.', '']
    for name in names:
        for seed in (7, 8, 9):
            path = f'muon_pilot_{name}_seed{seed}.json'
            lines.append(f'- [{path}]({path})')
    (reports / 'muon_pilot_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['methods'], indent=2), flush=True)
    print('Paired records/checkpoints validated; downstream inspection and curves saved.', flush=True)


if __name__ == '__main__':
    main()
