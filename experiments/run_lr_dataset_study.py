"""Run the predeclared paired learning-rate study without test-set selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/lr_dataset_study'
ARTIFACTS = ROOT / 'results/lr_dataset_study'
CANDIDATES = {
    'low': {'adamw_lr': .001, 'muon_lr': .007},
    'medium': {'adamw_lr': .003, 'muon_lr': .02},
    'high': {'adamw_lr': .01, 'muon_lr': .06},
}
DATASETS = {'MNIST': ('mnist', ROOT.parent / 'hw0/data'),
            'Fashion-MNIST': ('fashion', ROOT / 'data/fashion-mnist')}
SOURCE_PATHS = ['apps/train_baseline.py', 'apps/models.py', 'apps/encoder_checkpoint.py',
                'python/needle/optim.py', 'python/needle/data/datasets/mnist_dataset.py',
                'experiments/run_lr_dataset_study.py']
PAIR_KEYS = ('initial_parameter_sha256', 'train_indices_sha256',
             'validation_indices_sha256', 'training_order_sha256', 'data_files_sha256',
             'dataset', 'seed', 'train_samples', 'eval_samples', 'hidden_dim', 'feature_dim',
             'batch_size', 'validation_source', 'initial_validation_loss',
             'initial_validation_accuracy')


def select_configuration(records):
    if not records or len({r['candidate'] for r in records}) != len(records):
        raise ValueError('Expected unique, nonempty candidate records')
    if any(not math.isfinite(r['best_validation_loss']) for r in records):
        raise ValueError('Non-finite candidate validation loss')
    return min(records, key=lambda r: (r['best_validation_loss'], r['candidate']))


def validate_pair(a, b):
    for key in PAIR_KEYS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'Paired run mismatch: {key}')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen study file differs: {path}')
    else:
        save_json(path, value)


def run_trial(dataset, method, candidate, seed, phase, protocol):
    if {p: digest(ROOT / p) for p in SOURCE_PATHS} != protocol['source_sha256']:
        raise ValueError('Training source changed after the protocol was frozen')
    slug, data = DATASETS[dataset]
    trial_id = f'{slug}_{phase}_{method}_{candidate}_seed{seed}'
    output = REPORTS / f'{trial_id}.json'
    best = ARTIFACTS / f'{trial_id}_best.npz'
    config = CANDIDATES[candidate]
    study = {'phase': phase, 'candidate': candidate, 'method': method,
             'source_sha256': protocol['source_sha256']}
    if output.exists():
        record = json.loads(output.read_text())
        if record.get('study') != study or not best.exists() or digest(best) != record['checkpoint_sha256']:
            raise ValueError(f'Existing trial is incomplete or incompatible: {trial_id}')
    else:
        cmd = [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--device', 'cpu',
               '--dataset', dataset, '--data-dir', str(data), '--optimizer', method,
               '--lr', str(config['adamw_lr']), '--muon-lr', str(config['muon_lr']),
               '--weight-decay', '.01', '--epochs', '10', '--batch-size', '64',
               '--train-limit', '55000', '--validation-size', '5000', '--eval-limit', '5000',
               '--split-seed', '17', '--seed', str(seed), '--hidden-dim', '128',
               '--feature-dim', '64', '--best-checkpoint-output', str(best), '--output', str(output)]
        print(f'START {trial_id}', flush=True)
        started = time.perf_counter()
        with (ARTIFACTS / f'{trial_id}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(output.read_text())
        record['study'] = study
        record['checkpoint_sha256'] = digest(best)
        record['process_elapsed_seconds'] = time.perf_counter() - started
        save_json(output, record)
    assert record['dataset'] == dataset and record['seed'] == seed
    assert record['validation_source'] == 'train_holdout' and record['train_validation_disjoint']
    assert record['train_samples'] == 55000 and record['eval_samples'] == 5000
    assert len(record['epochs']) == 10 and all(e['optimizer_steps'] == 860 for e in record['epochs'])
    assert record['learning_rate'] == config['adamw_lr']
    assert record['muon_learning_rate'] == (config['muon_lr'] if method == 'muon' else None)
    selected = min(record['epochs'], key=lambda e: e['validation_loss'])
    assert selected['epoch'] == record['best_validation_epoch']
    assert selected['validation_loss'] == record['best_validation_loss']
    print(f"DONE {trial_id}: val_loss={record['best_validation_loss']:.6f} "
          f"val_accuracy={record['best_validation_accuracy']:.4f} "
          f"epoch={record['best_validation_epoch']}", flush=True)
    return record, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', *DATASETS], default='all')
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    protocol = {'date': '2026-10-31', 'selection': 'minimum_validation_loss_then_candidate_name',
                'candidates': CANDIDATES, 'tuning_seed': 7, 'repeat_seeds': [8, 9, 10],
                'epochs': 10, 'batch_size': 64, 'train_samples': 55000, 'validation_samples': 5000,
                'split_seed': 17, 'weight_decay': .01, 'model': [784, 128, 64, 10],
                'source_sha256': {p: digest(ROOT / p) for p in SOURCE_PATHS}}
    freeze_json(REPORTS / 'protocol.json', protocol)
    for dataset in DATASETS if args.dataset == 'all' else [args.dataset]:
        slug, _ = DATASETS[dataset]
        tuning = {name: [] for name in ('adamw', 'muon')}
        for index, candidate in enumerate(CANDIDATES):
            paired = {}
            for method in (('adamw', 'muon') if index % 2 == 0 else ('muon', 'adamw')):
                record, output = run_trial(dataset, method, candidate, 7, 'tune', protocol)
                tuning[method].append({'candidate': candidate, 'record': str(output.relative_to(ROOT)),
                                       'record_sha256': digest(output),
                                       'best_validation_loss': record['best_validation_loss'],
                                       'best_validation_accuracy': record['best_validation_accuracy']})
                paired[method] = record
            validate_pair(paired['adamw'], paired['muon'])
        selection = {'dataset': dataset, 'tuning_seed': 7, 'selection': protocol['selection'],
                     'test_set_used': False, 'all_candidates': tuning,
                     'selected': {name: select_configuration(rows) for name, rows in tuning.items()}}
        freeze_json(REPORTS / f'selection_{slug}.json', selection)
        print('LOCKED ' + json.dumps(selection['selected']), flush=True)
        for index, seed in enumerate(protocol['repeat_seeds']):
            paired = {}
            for method in (('muon', 'adamw') if index % 2 == 0 else ('adamw', 'muon')):
                candidate = selection['selected'][method]['candidate']
                paired[method], _ = run_trial(dataset, method, candidate, seed, 'repeat', protocol)
            validate_pair(paired['adamw'], paired['muon'])
        print(f'COMPLETE {dataset}', flush=True)


if __name__ == '__main__':
    main()
