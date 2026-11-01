"""Run the frozen220s optimizer comparison, then reuse native joint retrieval."""
import argparse
import json
from pathlib import Path
import sys

from run_joint_retrieval import digest, freeze_json, save_json, run_logged, run_retrieval_study

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/time_budget_study'
ARTIFACTS = ROOT / 'results/time_budget_study'
SEEDS = (8, 9, 10)
SOURCES = ['apps/train_timed.py', 'apps/train_baseline.py', 'apps/models.py',
           'apps/encoder_checkpoint.py', 'python/needle/optim.py',
           'experiments/run_time_budget_study.py', 'experiments/run_joint_retrieval.py']


def validate_timed_pair(a, b):
    for key in ('initial_parameter_sha256', 'train_indices_sha256',
                'validation_indices_sha256', 'data_files_sha256', 'seed',
                'batch_size', 'time_budget_seconds', 'architecture'):
        if a[key] != b[key]:
            raise ValueError('Paired mismatch: ' + key)
    count = min(len(a['epoch_order_sha256']), len(b['epoch_order_sha256']))
    if count == 0 or a['epoch_order_sha256'][:count] != b['epoch_order_sha256'][:count]:
        raise ValueError('Paired epoch order prefixes differ')


def train(protocol):
    catalog = []
    for index, seed in enumerate(SEEDS):
        pair = {}
        for method in (('muon', 'adamw') if index % 2 == 0 else ('adamw', 'muon')):
            assert {p: digest(ROOT / p) for p in SOURCES} == protocol['source_sha256']
            run_id = f'{method}_seed{seed}'
            output, best = REPORTS / f'{run_id}.json', ARTIFACTS / f'{run_id}_best.npz'
            terminal = ARTIFACTS / f'{run_id}_last.npz'
            if not output.exists():
                print('TRAIN ' + run_id, flush=True)
                run_logged([sys.executable, str(ROOT / 'apps/train_timed.py'),
                            '--optimizer', method, '--budget-seconds', '220', '--seed', str(seed),
                            '--output', str(output), '--best-checkpoint-output', str(best),
                            '--checkpoint-output', str(terminal)], ARTIFACTS / f'{run_id}.log')
                record = json.loads(output.read_text())
                record.update(source_sha256=protocol['source_sha256'], checkpoint_sha256=digest(best),
                              terminal_checkpoint_sha256=digest(terminal))
                save_json(output, record)
            record = json.loads(output.read_text())
            assert record['source_sha256'] == protocol['source_sha256']
            assert digest(best) == record['checkpoint_sha256']
            assert digest(terminal) == record['terminal_checkpoint_sha256']
            assert record['time_budget_seconds'] == 220. and len(record['checkpoints']) == 4
            assert 0 <= record['budget_overrun_seconds'] <= record['last_step_seconds'] + .001
            assert record['architecture'] == protocol['architecture']
            pair[method] = record
            catalog.append({'encoder': method, 'training_seed': seed, 'candidate': 'prior_low_220s',
                            'training_record': str(output.relative_to(ROOT)),
                            'training_record_sha256': digest(output),
                            'checkpoint': str(best.relative_to(ROOT)), 'checkpoint_sha256': digest(best),
                            'adamw_or_auxiliary_learning_rate': .001,
                            'muon_matrix_learning_rate': .007 if method == 'muon' else None})
            print(f"DONE {run_id}: steps={record['optimizer_steps']} "
                  f"time={record['timed_seconds']:.3f}s val={record['best_validation_accuracy']:.4f}", flush=True)
        validate_timed_pair(pair['adamw'], pair['muon'])
    freeze_json(REPORTS / 'models.json', catalog)
    return catalog


def verify_models(catalog):
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from encoder_checkpoint import load_encoder
    from summarize_lr_dataset_study import evaluate

    data = ROOT / 'data/fashion-mnist'
    dataset = ndl.data.MNISTDataset(str(data / 'train-images-idx3-ubyte.gz'),
                                    str(data / 'train-labels-idx1-ubyte.gz'))
    ids = np.random.default_rng(17).permutation(len(dataset))[:5000]
    rows = []
    for item in catalog:
        model = load_encoder(ROOT / item['checkpoint'], ndl.cpu())
        result = evaluate(model, dataset, ids)
        record = json.loads((ROOT / item['training_record']).read_text())
        assert abs(result['loss'] - record['best_validation_loss']) < 1e-7
        assert result['accuracy'] == record['best_validation_accuracy']
        rows.append(dict(item, validation_reproduced=result))
    save_json(REPORTS / 'model_validation.json', rows)
    print('ALL SIX SELECTED VALIDATION CHECKPOINTS REPRODUCED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'train', 'retrieval'], default='all')
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    protocol = {'date': '2026-11-01', 'budget_seconds': 220.,
                'architecture': [784, 128, 128, 128, 64, 10], 'batch_size': 64,
                'training_seeds': list(SEEDS), 'train_rows': 55000, 'validation_rows': 5000,
                'split_seed': 17, 'checkpoint_targets_seconds': [55, 110, 165, 220],
                'selection': 'minimum_validation_loss_among_four_common_time_candidates',
                'adamw_or_auxiliary_lr': .001, 'muon_lr': .007, 'weight_decay': .01,
                'learning_rates_from': 'reports/depth_study/selection.json',
                'budget_policy': 'includes_intermediate_validation_and_save;excludes_setup_and_final_eval',
                'source_sha256': {p: digest(ROOT / p) for p in SOURCES}}
    freeze_json(REPORTS / 'protocol.json', protocol)
    if args.phase in ('all', 'train'):
        catalog = train(protocol)
        verify_models(catalog)
    else:
        catalog = json.loads((REPORTS / 'models.json').read_text())
    if args.phase in ('all', 'retrieval'):
        run_retrieval_study(catalog, REPORTS / 'retrieval', ARTIFACTS / 'retrieval',
                            training_source_commit='see_time_budget_training_protocol_source_hashes',
                            extra_sources=SOURCES, architecture=protocol['architecture'])


if __name__ == '__main__':
    main()
