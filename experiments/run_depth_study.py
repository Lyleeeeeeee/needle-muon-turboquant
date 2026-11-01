"""Run the predefined depth-only Fashion training and retrieval comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_lr_dataset_study import CANDIDATES, select_configuration, validate_pair
from run_joint_retrieval import digest, save_json, freeze_json, run_retrieval_study

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/depth_study'
ARTIFACTS = ROOT / 'results/depth_study'
SEEDS = (8, 9, 10)
SOURCES = ['apps/models.py', 'apps/encoder_checkpoint.py', 'apps/train_baseline.py',
           'python/needle/optim.py', 'experiments/run_depth_study.py',
           'experiments/run_joint_retrieval.py', 'experiments/run_lr_dataset_study.py',
           'experiments/summarize_lr_dataset_study.py']


def source_hashes():
    return {p: digest(ROOT / p) for p in SOURCES}


def trial(method, candidate, seed, phase, protocol):
    assert source_hashes() == protocol['source_sha256']
    config = CANDIDATES[candidate]
    run_id = f'{phase}_{method}_{candidate}_seed{seed}'
    output = REPORTS / f'{run_id}.json'
    best = ARTIFACTS / f'{run_id}_best.npz'
    study = {'phase': phase, 'candidate': candidate, 'method': method,
             'source_sha256': protocol['source_sha256']}
    if output.exists():
        record = json.loads(output.read_text())
        assert record['study'] == study and digest(best) == record['checkpoint_sha256']
    else:
        print('TRAIN ' + run_id, flush=True)
        cmd = [sys.executable, str(ROOT / 'apps/train_baseline.py'), '--device', 'cpu',
               '--dataset', 'Fashion-MNIST', '--optimizer', method, '--extra-hidden-layers', '2',
               '--hidden-dim', '128', '--feature-dim', '64', '--epochs', '10', '--batch-size', '64',
               '--train-limit', '55000', '--validation-size', '5000', '--eval-limit', '5000',
               '--split-seed', '17', '--seed', str(seed), '--lr', str(config['adamw_lr']),
               '--muon-lr', str(config['muon_lr']), '--weight-decay', '.01',
               '--best-checkpoint-output', str(best), '--output', str(output)]
        with (ARTIFACTS / f'{run_id}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1'),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        record = json.loads(output.read_text())
        record['study'] = study
        record['checkpoint_sha256'] = digest(best)
        save_json(output, record)
    assert record['extra_hidden_layers'] == 2 and record['total_parameter_count'] == 142410
    assert record['seed'] == seed and record['dataset'] == 'Fashion-MNIST'
    assert record['train_samples'] == 55000 and record['eval_samples'] == 5000
    assert record['validation_source'] == 'train_holdout' and record['train_validation_disjoint']
    assert len(record['epochs']) == 10 and all(e['optimizer_steps'] == 860 for e in record['epochs'])
    assert record['muon_parameter_count'] == (40960 if method == 'muon' else 0)
    assert record['learning_rate'] == config['adamw_lr']
    assert record['muon_learning_rate'] == (config['muon_lr'] if method == 'muon' else None)
    best_row = min(record['epochs'], key=lambda e: e['validation_loss'])
    assert best_row['epoch'] == record['best_validation_epoch']
    print(f"DONE {run_id} loss={record['best_validation_loss']:.6f} "
          f"accuracy={record['best_validation_accuracy']:.4f} epoch={record['best_validation_epoch']}", flush=True)
    return record, output


def train(protocol):
    tuning = {name: [] for name in ('adamw', 'muon')}
    for index, candidate in enumerate(CANDIDATES):
        pair = {}
        for method in (('adamw', 'muon') if index % 2 == 0 else ('muon', 'adamw')):
            record, path = trial(method, candidate, 7, 'tune', protocol)
            tuning[method].append({'candidate': candidate, 'record': str(path.relative_to(ROOT)),
                                   'record_sha256': digest(path),
                                   'best_validation_loss': record['best_validation_loss'],
                                   'best_validation_accuracy': record['best_validation_accuracy']})
            pair[method] = record
        validate_pair(pair['adamw'], pair['muon'])
    selection = {'selection': 'minimum_validation_loss_then_candidate_name', 'tuning_seed': 7,
                 'test_set_used': False, 'all_candidates': tuning,
                 'selected': {m: select_configuration(rows) for m, rows in tuning.items()}}
    freeze_json(REPORTS / 'selection.json', selection)
    print('LOCKED ' + json.dumps(selection['selected']), flush=True)
    catalog = []
    for index, seed in enumerate(SEEDS):
        pair = {}
        for method in (('muon', 'adamw') if index % 2 == 0 else ('adamw', 'muon')):
            candidate = selection['selected'][method]['candidate']
            record, path = trial(method, candidate, seed, 'repeat', protocol)
            pair[method] = record
            catalog.append({'encoder': method, 'training_seed': seed, 'candidate': candidate,
                            'training_record': str(path.relative_to(ROOT)),
                            'training_record_sha256': digest(path),
                            'checkpoint': str(Path(record['best_checkpoint_export']).relative_to(ROOT)),
                            'checkpoint_sha256': record['checkpoint_sha256'],
                            'extra_hidden_layers': 2, 'total_parameter_count': 142410,
                            'muon_parameter_count': record['muon_parameter_count'],
                            'adamw_or_auxiliary_learning_rate': record['learning_rate'],
                            'muon_matrix_learning_rate': record['muon_learning_rate']})
        validate_pair(pair['adamw'], pair['muon'])
        for key in ('train_indices_sha256', 'validation_indices_sha256', 'training_order_sha256'):
            shallow = json.loads((ROOT / f'reports/lr_dataset_study/fashion_repeat_adamw_low_seed{seed}.json').read_text())
            assert pair['adamw'][key] == shallow[key]
    freeze_json(REPORTS / 'models.json', catalog)
    return catalog


def verify_models(catalog):
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from encoder_checkpoint import load_encoder
    from summarize_lr_dataset_study import evaluate

    data_dir = ROOT / 'data/fashion-mnist'
    training = ndl.data.MNISTDataset(str(data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(data_dir / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(data_dir / 't10k-images-idx3-ubyte.gz'),
                                str(data_dir / 't10k-labels-idx1-ubyte.gz'))
    validation_ids = np.random.default_rng(17).permutation(len(training))[:5000]
    results = []
    for item in catalog:
        record = json.loads((ROOT / item['training_record']).read_text())
        assert digest(ROOT / item['checkpoint']) == item['checkpoint_sha256']
        model = load_encoder(ROOT / item['checkpoint'], ndl.cpu())
        assert model.extra_hidden_layers == 2
        validation = evaluate(model, training, validation_ids)
        assert abs(validation['loss'] - record['best_validation_loss']) < 1e-7
        assert validation['accuracy'] == record['best_validation_accuracy']
        result = {'encoder': item['encoder'], 'training_seed': item['training_seed'],
                  'checkpoint_sha256': item['checkpoint_sha256'], 'validation_reproduced': validation,
                  'official_test': evaluate(model, test, np.arange(len(test)))}
        print('VERIFY ' + json.dumps(result), flush=True)
        results.append(result)
    save_json(REPORTS / 'model_validation.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'train', 'retrieval'], default='all')
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    protocol = {'date': '2026-11-01', 'architecture': [784, 128, 128, 128, 64, 10],
                'extra_hidden_layers': 2, 'total_parameters': 142410, 'muon_parameters': 40960,
                'muon_parameter_fraction': 40960 / 142410, 'candidates': CANDIDATES,
                'tuning_seed': 7, 'repeat_seeds': list(SEEDS), 'epochs': 10, 'batch_size': 64,
                'train_rows': 55000, 'validation_rows': 5000, 'split_seed': 17,
                'weight_decay': .01, 'source_sha256': source_hashes(),
                'selection': 'minimum_validation_loss_then_candidate_name',
                'shallow_reference_commit': '74c161a67ddae08847478d04f8fb4e6c988999b5'}
    freeze_json(REPORTS / 'protocol.json', protocol)
    if args.phase in ('all', 'train'):
        catalog = train(protocol)
        verify_models(catalog)
    else:
        catalog = json.loads((REPORTS / 'models.json').read_text())
    if args.phase in ('all', 'retrieval'):
        run_retrieval_study(catalog, REPORTS / 'retrieval', ARTIFACTS / 'retrieval',
                            training_source_commit='see_depth_training_protocol_source_hashes',
                            extra_sources=SOURCES, architecture=protocol['architecture'])


if __name__ == '__main__':
    main()
