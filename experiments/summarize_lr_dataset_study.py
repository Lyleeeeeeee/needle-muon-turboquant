"""Validate completed training records and evaluate frozen Fashion checkpoints.

Run after all retained timings. NumPy/statistics/Matplotlib are analysis tools;
forward inference and loss evaluation use the native Needle CPU backend.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics as stats
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
from run_lr_dataset_study import (ARTIFACTS, CANDIDATES, DATASETS, REPORTS,
                                  digest, save_json, select_configuration, validate_pair)


def distribution(values):
    return {'values': values, 'mean': stats.mean(values), 'sample_std': stats.stdev(values)}


def summarize_runs(records):
    result = {
        'validation_accuracy': distribution([r['best_validation_accuracy'] for r in records]),
        'validation_loss': distribution([r['best_validation_loss'] for r in records]),
        'training_seconds': distribution([sum(e['training_seconds'] for e in r['epochs']) for r in records]),
        'optimizer_seconds': distribution([sum(e['optimizer_step_seconds'] for e in r['epochs']) for r in records]),
        'mean_update_ms': distribution([
            1000 * sum(e['optimizer_step_seconds'] for e in r['epochs']) /
            sum(e['optimizer_steps'] for e in r['epochs']) for r in records]),
        'selected_epochs': [r['best_validation_epoch'] for r in records],
        'numeric_state_bytes': [r['optimizer_numeric_state_bytes'] for r in records],
    }
    return result


def evaluate(model, data, ids):
    import numpy as np
    import needle as ndl
    from needle import nn

    model.eval()
    criterion = nn.SoftmaxLoss()
    total_loss, correct = 0., 0
    for start in range(0, len(ids), 64):
        batch = ids[start:start + 64]
        x = ndl.Tensor(data.X[batch], device=ndl.cpu(), requires_grad=False)
        y = ndl.Tensor(data.y[batch], device=ndl.cpu(), dtype='float32', requires_grad=False)
        scores = model(x)
        total_loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * len(batch)
        correct += int((scores.numpy().argmax(axis=1) == data.y[batch]).sum())
    return {'loss': total_loss / len(ids), 'accuracy': correct / len(ids), 'samples': len(ids)}


def draw_curves(records):
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {'adamw': '#2869a6', 'muon': '#dd7924'}
    for row, dataset in enumerate(DATASETS):
        for method, runs in records[dataset].items():
            loss = np.array([[e['validation_loss'] for e in r['epochs']] for r in runs])
            accuracy = 100 * np.array([[e['validation_accuracy'] for e in r['epochs']] for r in runs])
            times = np.cumsum([[e['training_seconds'] for e in r['epochs']] for r in runs], axis=1).mean(axis=0)
            label = 'AdamW' if method == 'adamw' else 'Muon + auxiliary AdamW'
            for ax, x, y in [(axes[row, 0], np.arange(1, 11), loss), (axes[row, 1], times, accuracy)]:
                mean, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
                ax.plot(x, mean, color=colors[method], marker='o', markersize=3, label=label)
                ax.fill_between(x, mean - sd, mean + sd, color=colors[method], alpha=.14)
                ax.grid(alpha=.2)
        axes[row, 0].set(xlabel='Epoch', ylabel='Validation cross-entropy', title=dataset)
        axes[row, 0].set_xticks(range(1, 11))
        axes[row, 1].set(xlabel='Cumulative training-loop seconds', ylabel='Validation accuracy (%)', title=dataset)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Selected rates: fresh seeds 8/9/10; shading = sample standard deviation')
    fig.savefig(REPORTS / 'curves.svg', metadata={'Date': '2026-10-31'})
    fig.savefig(ARTIFACTS / 'curves.png', dpi=150)
    plt.close(fig)


def main():
    import numpy as np
    import needle as ndl
    from encoder_checkpoint import load_encoder
    from train_baseline import select_training_indices

    protocol = json.loads((REPORTS / 'protocol.json').read_text())
    selections, repeats, raw_records = {}, {}, {}
    summary = {'date': '2026-10-31', 'protocol': protocol,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                               'platform': platform.platform(), 'machine': platform.machine(),
                               'backend': ndl.BACKEND, 'device': str(ndl.cpu()),
                               'blas_threads': 1}, 'datasets': {},
               'all_paired_controls_verified': True, 'validation_checkpoint_reloads': [],
               'classification_only': True, 'quantization_rerun': False}
    for dataset, (slug, _) in DATASETS.items():
        selection = json.loads((REPORTS / f'selection_{slug}.json').read_text())
        selections[dataset] = selection
        for candidate in CANDIDATES:
            paired = {}
            for method in ('adamw', 'muon'):
                path = REPORTS / f'{slug}_tune_{method}_{candidate}_seed7.json'
                record = json.loads(path.read_text())
                assert record['study']['source_sha256'] == protocol['source_sha256']
                row = next(r for r in selection['all_candidates'][method] if r['candidate'] == candidate)
                assert row['record_sha256'] == digest(path)
                assert row['best_validation_loss'] == record['best_validation_loss']
                raw_records[str(path.relative_to(ROOT))] = digest(path)
                paired[method] = record
            validate_pair(paired['adamw'], paired['muon'])
        for method in ('adamw', 'muon'):
            assert select_configuration(selection['all_candidates'][method]) == selection['selected'][method]
        runs = {method: [] for method in ('adamw', 'muon')}
        for seed in protocol['repeat_seeds']:
            pair = {}
            for method in runs:
                candidate = selection['selected'][method]['candidate']
                path = REPORTS / f'{slug}_repeat_{method}_{candidate}_seed{seed}.json'
                record = json.loads(path.read_text())
                assert record['seed'] == seed and record['dataset'] == dataset
                assert record['study']['source_sha256'] == protocol['source_sha256']
                assert record['study']['candidate'] == candidate
                assert record['learning_rate'] == CANDIDATES[candidate]['adamw_lr']
                assert record['muon_learning_rate'] == (CANDIDATES[candidate]['muon_lr'] if method == 'muon' else None)
                assert record['train_validation_disjoint'] and record['validation_source'] == 'train_holdout'
                assert len(record['epochs']) == 10 and all(e['optimizer_steps'] == 860 for e in record['epochs'])
                assert min(record['epochs'], key=lambda e: e['validation_loss'])['epoch'] == record['best_validation_epoch']
                assert digest(Path(record['best_checkpoint_export'])) == record['checkpoint_sha256']
                raw_records[str(path.relative_to(ROOT))] = digest(path)
                runs[method].append(record)
                pair[method] = record
            validate_pair(pair['adamw'], pair['muon'])
        repeats[dataset] = runs
        paired_differences = [b['best_validation_accuracy'] - a['best_validation_accuracy']
                              for a, b in zip(runs['adamw'], runs['muon'])]
        summary['datasets'][dataset] = {
            'selected_candidates': selection['selected'],
            'methods': {method: summarize_runs(rows) for method, rows in runs.items()},
            'paired_validation_accuracy_muon_minus_adamw': distribution(paired_differences)}
    assert len(raw_records) == 24

    # The old central configurations provide an exact numerical regression check.
    central_matches = {}
    for method in ('adamw', 'muon'):
        old = json.loads((ROOT / f'reports/muon_pilot_{method}_seed7.json').read_text())
        new = json.loads((REPORTS / f'mnist_tune_{method}_medium_seed7.json').read_text())
        keys = ('training_loss', 'validation_loss', 'validation_accuracy')
        central_matches[method] = all(a[k] == b[k] for a, b in zip(old['epochs'], new['epochs']) for k in keys)
    assert all(central_matches.values())
    summary['old_medium_seed7_epoch_metrics_reproduced_exactly'] = central_matches

    # All choices and raw records are hashed before any official Fashion test access.
    lock = {'raw_record_sha256': raw_records,
            'selection_sha256': {dataset: digest(REPORTS / f'selection_{slug}.json') for dataset, (slug, _) in DATASETS.items()},
            'test_selection': False}
    lock_path = REPORTS / 'evaluation_lock.json'
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock
    else:
        save_json(lock_path, lock)
    summary['evaluation_lock_sha256'] = digest(lock_path)

    fashion_dir = DATASETS['Fashion-MNIST'][1]
    for name, expected_md5 in [('t10k-images-idx3-ubyte.gz', 'bef4ecab320f06d8554ea6380940ec79'),
                               ('t10k-labels-idx1-ubyte.gz', 'bb300cfdad3c16e7a12a480ee83cd310')]:
        path = fashion_dir / name
        if not path.is_file():
            raise SystemExit('Choices locked. Download/check official Fashion test files, then rerun this analysis.')
        assert hashlib.md5(path.read_bytes()).hexdigest() == expected_md5

    test = ndl.data.MNISTDataset(str(fashion_dir / 't10k-images-idx3-ubyte.gz'),
                                str(fashion_dir / 't10k-labels-idx1-ubyte.gz'))
    assert len(test) == 10000
    fashion_test = {method: [] for method in ('adamw', 'muon')}
    for dataset, (_, data_dir) in DATASETS.items():
        data = ndl.data.MNISTDataset(str(data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(data_dir / 'train-labels-idx1-ubyte.gz'))
        _, validation_ids, _ = select_training_indices(len(data), len(data), 55000, 5000,
                                                       np.random.default_rng(7), 5000, 17)
        for method, rows in repeats[dataset].items():
            for record in rows:
                model = load_encoder(record['best_checkpoint_export'], ndl.cpu())
                measured = evaluate(model, data, validation_ids)
                assert abs(measured['loss'] - record['best_validation_loss']) < 1e-7
                assert measured['accuracy'] == record['best_validation_accuracy']
                summary['validation_checkpoint_reloads'].append({
                    'dataset': dataset, 'method': method, 'seed': record['seed'], 'matched': True})
                if dataset == 'Fashion-MNIST':
                    result = evaluate(model, test, np.arange(len(test)))
                    result.update(seed=record['seed'], checkpoint_sha256=record['checkpoint_sha256'])
                    fashion_test[method].append(result)
                    print('FASHION TEST', method, record['seed'], result['accuracy'], flush=True)
    summary['fashion_official_test'] = {
        'samples': 10000, 'selection_used_test': False,
        'data_sha256': {name: digest(fashion_dir / name) for name in
                        ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')},
        'methods': {method: {'runs': rows, 'accuracy': distribution([r['accuracy'] for r in rows]),
                             'loss': distribution([r['loss'] for r in rows])}
                    for method, rows in fashion_test.items()},
        'paired_accuracy_muon_minus_adamw': distribution([
            b['accuracy'] - a['accuracy'] for a, b in zip(fashion_test['adamw'], fashion_test['muon'])])}
    save_json(REPORTS / 'summary.json', summary)
    draw_curves(repeats)
    write_report(summary, selections)
    print('All24 records,12 validation reloads,6 final Fashion test evaluations verified.', flush=True)


def write_report(summary, selections):
    def fmt(value, percent=False):
        scale = 100 if percent else 1
        return f"{scale * value['mean']:.2f}{'%' if percent else ''} +/- {scale * value['sample_std']:.2f}"

    mnist = summary['datasets']['MNIST']['methods']
    fashion = summary['fashion_official_test']['methods']
    lead = (f"After the fixed three-candidate search, fresh-seed MNIST validation accuracy is "
            f"{100*mnist['adamw']['validation_accuracy']['mean']:.2f}% for AdamW versus "
            f"{100*mnist['muon']['validation_accuracy']['mean']:.2f}% for Muon plus auxiliary AdamW. "
            f"Final Fashion-MNIST test accuracy is {100*fashion['adamw']['accuracy']['mean']:.2f}% "
            f"versus {100*fashion['muon']['accuracy']['mean']:.2f}%, respectively. "
            'These are classification metrics, not retrieval precision. Validation and test '
            'results have different selection histories; compare methods within each dataset/split.')
    cpu_cost = {dataset: 100 * (result['methods']['muon']['training_seconds']['mean'] /
                               result['methods']['adamw']['training_seconds']['mean'] - 1)
                for dataset, result in summary['datasets'].items()}
    gap = 100 * summary['fashion_official_test']['paired_accuracy_muon_minus_adamw']['mean']
    cost_lead = (f"Muon ten-epoch CPU training loops take {cpu_cost['MNIST']:.1f}% longer on MNIST "
                 f"and {cpu_cost['Fashion-MNIST']:.1f}% longer on Fashion-MNIST. "
                 f"The mean final Fashion accuracy gap (Muon minus AdamW) is {gap:+.2f} percentage points. "
                 'Read the paired seed differences and training cost alongside the small mean gap; '
                 'this setting does not establish general optimizer superiority.')
    lines = ['# AdamW and Muon: learning-rate tuning and Fashion-MNIST', '', lead, '', cost_lead, '',
             'The study follows the [predeclared design](../../docs/lr_dataset_study_design.md). '
             'It gives both methods three configurations per dataset, locks choices using seed 7 '
             'validation loss, then runs fresh seeds 8/9/10. All 24 ten-epoch trainings use the same '
             '784-128-64-10 Needle float32 CPU encoder, 55,000 training/5,000 validation images '
             '(split17), batch 64, and weight decay0.01. No test data selects a rate or checkpoint.', '',
             '## All search candidates (seed7)', '',
             '| Dataset | Method | Candidate | AdamW/auxiliary rate | Muon matrix rate | Minimum validation loss | Accuracy at that checkpoint |',
             '|---|---|---|---:|---:|---:|---:|']
    for dataset, selection in selections.items():
        for method, rows in selection['all_candidates'].items():
            for row in rows:
                config = CANDIDATES[row['candidate']]
                lines.append(f"| {dataset} | {method} | {row['candidate']} | {config['adamw_lr']:g} | "
                             f"{config['muon_lr'] if method == 'muon' else '-'} | "
                             f"{row['best_validation_loss']:.6f} | {100*row['best_validation_accuracy']:.2f}% |")
    lines += ['', 'Each Muon configuration pairs a matrix rate with an auxiliary AdamW rate. '
              'This is three paired candidates, not an independent 2D search. Momentum 0.95, '
              'five Newton-Schulz steps and weight decay are fixed. Search seed 7 is excluded '
              'from the repeat averages. Rates/checkpoints are selected by loss, not accuracy.', '',
              '## Fresh repeat results', '',
              '| Dataset | Method | Selected candidate | Validation accuracy | Validation loss | Ten-epoch training seconds | Selected epochs |',
              '|---|---|---|---:|---:|---:|---|']
    for dataset, result in summary['datasets'].items():
        for method, metrics in result['methods'].items():
            lines.append(f"| {dataset} | {method} | {result['selected_candidates'][method]['candidate']} | "
                         f"{fmt(metrics['validation_accuracy'], True)} | {metrics['validation_loss']['mean']:.6f} | "
                         f"{fmt(metrics['training_seconds'])} | {metrics['selected_epochs']} |")
    lines += ['', 'Mean +/- sample standard deviation over seeds 8/9/10; these are not confidence intervals. '
              'Each repeat uses its minimum-validation-loss checkpoint. Validation data is reused for '
              'rate and checkpoint selection, so validation accuracy is selection-biased. '
              'Fresh means excluded from this study\'s seed 7 search: seeds 8/9 were already used in '
              'the earlier fixed-rate MNIST pilot. These are newly executed locked-configuration '
              'trials, not additional independent samples to pool with the old ones.', '',
              '![Validation curves of the selected configurations](curves.svg)', '',
              '## Final Fashion-MNIST test (10,000 images)', '',
              'All rates, checkpoint hashes and training-record hashes were frozen in '
              '[evaluation_lock.json](evaluation_lock.json) before downloading/reading Fashion test files. '
              'Each of the six fixed repeat checkpoints is evaluated once; test results cause no retuning.', '',
              '| Method | Test accuracy | Test cross-entropy |', '|---|---:|---:|']
    for method, values in summary['fashion_official_test']['methods'].items():
        lines.append(f"| {method} | {fmt(values['accuracy'], True)} | {values['loss']['mean']:.6f} |")
    difference = summary['fashion_official_test']['paired_accuracy_muon_minus_adamw']
    lines += ['', f"Paired Muon-minus-AdamW test-accuracy differences (percentage points): "
              f"{[round(100*v, 3) for v in difference['values']]}; mean {100*difference['mean']:+.3f}. "
              'Three seeds do not establish general optimizer superiority.', '',
              '## Cost and interpretation', '',
              '| Dataset | Method | Total three-candidate search training seconds | Mean optimizer update ms | Numeric state bytes |',
              '|---|---|---:|---:|---:|']
    for dataset, (slug, _) in DATASETS.items():
        for method in ('adamw', 'muon'):
            cost = sum(sum(e['training_seconds'] for e in json.loads(
                (REPORTS / f'{slug}_tune_{method}_{candidate}_seed7.json').read_text())['epochs'])
                       for candidate in CANDIDATES)
            metrics = summary['datasets'][dataset]['methods'][method]
            lines.append(f"| {dataset} | {method} | {cost:.2f} | {metrics['mean_update_ms']['mean']:.3f} | "
                         f"{metrics['numeric_state_bytes'][0]:,} |")
    lines += ['', 'Training-loop timing includes forward/backward/update, excluding loading, validation '
              'and checkpoint I/O. Search budgets match by configurations and updates; elapsed cost '
              'can differ. Numeric state excludes parameters, gradients, Python overhead and transient '
              'Newton-Schulz workspace; it is not peak memory. CPU float32 results do not predict GPU speed.', '',
              'Muon still updates only the 8,192-parameter hidden matrix out of 109,386 total parameters '
              '(about 7.5%). This small MLP, fixed decay, constant learning rates, short training budget '
              'and narrow paired grid limit conclusions. A selected endpoint does not establish an '
              'optimal rate. Changing dataset alone cannot establish why an optimizer differs; model '
              'and optimizer coverage remain unchanged. No additional candidates were added after outcomes.', '',
              'Classification accuracy is not retrieval same-label precision. No compression was rerun; '
              'the older quantization numbers belong to their original fixed encoder. The old MNIST test '
              'has already been inspected in prior work and is not reported as an untouched final test here.', '',
              '## Verification and reproduction', '',
              'All paired initialization/data/split/order hashes agree. All 12 repeat checkpoint reloads '
              'reproduce their full 5,000-image validation metrics. Both medium seed 7 MNIST trajectories '
              'match the earlier pilot exactly. Dataset identity/export provenance, loss-only rate '
              'selection, pair rejection and sample statistics have regression checks.', '',
              'Download the four official IDX gzip files into `data/fashion-mnist/`, checking the '
              '[published MD5 checksums](https://github.com/zalandoresearch/fashion-mnist#Get-the-Data). '
              'For the original staged protocol, acquire only training files first, run training, '
              'run the summarizer to create the evaluation lock, then acquire test files and rerun it.', '',
              'Published records are paired with local ignored checkpoints. For full reproduction '
              'from a fresh clone, first move the published `reports/lr_dataset_study` directory '
              'to an archive outside that output path (for example, under `results/reproduction_archive`). '
              'Then run the commands below to produce new records and checkpoints. The runner '
              'deliberately refuses to treat an old record with a missing checkpoint as a complete run.', '',
              '```bash', 'python -m pytest -q', 'python experiments/run_lr_dataset_study.py',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_lr_dataset_study.py',
              '```', '', 'The runner resumes only complete, source-compatible recorded trials. '
              'Data/checkpoints/logs remain ignored; raw JSON, protocol, selections and hash lock are '
              'published here. [Full machine-readable results](summary.json) include every final test '
              'evaluation and repeat metric. Matplotlib is analysis-only.', '']
    (REPORTS / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
