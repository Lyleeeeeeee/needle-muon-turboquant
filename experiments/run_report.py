"""Train paired encoders, export their own embeddings and compare all gallery stores."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

from common import (aggregate_quality, digest, save_json, validate_pair,
                    validate_timed_pair)

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4']


def run(command, log):
    with log.open('w') as out:
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                VECLIB_MAXIMUM_THREADS='1', NEEDLE_BACKEND='nd',
                                NEEDLE_CPU_MATMUL='strided'), stdout=out,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['epochs', 'timed'], default='epochs')
    parser.add_argument('--checkpoint', choices=['best', 'terminal'], default='best')
    parser.add_argument('--seeds', type=int, nargs='+', default=[8, 9, 10])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Reuse only stages matching the frozen protocol and hashes')
    parser.add_argument('--checkpoint-check', action='store_true',
                        help='Evaluate the other checkpoint selection with FP32 as a sensitivity check')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('Training seeds must be unique')
    output = (args.output_dir or ROOT / f'results/report_{args.study}_{"smoke" if args.smoke else args.checkpoint}').resolve()
    if (output / 'protocol.json').exists() and not args.resume:
        parser.error('Completed run exists; use a new --output-dir to preserve the previous run')
    output.mkdir(parents=True, exist_ok=True)
    args.data_dir = args.data_dir.resolve()
    seeds = [8] if args.smoke else args.seeds
    epochs, train_rows, validation_rows, gallery_rows, queries = ((1, 512, 128, 128, 16)
                                                               if args.smoke else (10, 55000, 5000, 55000, 1000))
    paths = [*sorted((ROOT / 'python').rglob('*.py')), *sorted((ROOT / 'src').glob('*.h')),
             ROOT / 'src/ndarray_backend_cpu.cc', *sorted((ROOT / 'apps').glob('*.py')),
             Path(__file__), ROOT / 'experiments/common.py']
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    protocol = {'dataset': 'Fashion-MNIST', 'study': args.study, 'smoke': args.smoke,
                'training_seeds': seeds, 'checkpoint': args.checkpoint,
                'checkpoint_check': args.checkpoint_check,
                'epochs': epochs, 'budget_seconds': .3 if args.smoke else 220.,
                'train_rows': train_rows, 'validation_rows': validation_rows,
                'gallery_rows': gallery_rows, 'query_rows': queries, 'split_seed': 17,
                'architecture': [784, 128, 128, 128, 64, 10], 'batch_size': 64,
                'rates': {'adamw_auxiliary': .001, 'muon': .007}, 'weight_decay': .01,
                'rotation_seeds': [7, 8, 9], 'methods': METHODS, 'metric': 'cosine',
                'source_sha256': source_hashes,
                'python': sys.version, 'platform': platform.platform(),
                'backend': {'name': 'native_cpu', 'dtype': 'float32', 'cpu_matmul': 'strided',
                            'tiled_ns': True, 'extension_sha256': digest(
                                ROOT / 'python/needle/backend_ndarray' /
                                ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX')))},
                'data_files_sha256': {p.name: digest(p) for p in args.data_dir.glob('*.gz')},
                'note': 'new current-backend run; does not overwrite published historical evidence'}
    frozen = output / 'protocol.json'
    if frozen.exists():
        if json.loads(frozen.read_text()) != protocol:
            raise ValueError('Resume protocol differs; preserve this run and use a new output directory')
    else:
        save_json(frozen, protocol)
    protocol_sha = digest(frozen)
    provenance = {'protocol_sha256': protocol_sha}

    def exported(name, checkpoint):
        features = output / f'{name}_features.npz'
        path = features.with_suffix('.json')
        if not (args.resume and path.exists()):
            run([ROOT / 'apps/export_features.py', '--checkpoint', checkpoint, '--dataset', 'Fashion-MNIST',
                 '--data-dir', args.data_dir, '--validation-size', validation_rows, '--split-seed', 17,
                 '--gallery-size', gallery_rows, '--queries', queries, '--reference-train', min(4096, gallery_rows),
                 '--reference-queries', min(256, queries), '--seed', 7, '--output', features],
                output / f'{name}_export.log')
            feature = json.loads(path.read_text())
            feature['runner_provenance'] = provenance
            save_json(path, feature)
        feature = json.loads(path.read_text())
        if (feature['checkpoint_sha256'] != digest(checkpoint)
                or feature['feature_archive_sha256'] != digest(features)
                or feature['gallery_shape'] != [gallery_rows, 64]
                or feature['query_shape'] != [queries, 64]
                or feature.get('runner_provenance') != provenance):
            raise ValueError('Own-model feature provenance or dimensions differ')
        return features, feature

    def benchmark(name, features, feature, rotation, methods):
        path = output / f'{name}_rotation{rotation}.json'
        expected = dict(provenance, feature_archive_sha256=feature['feature_archive_sha256'])
        if not (args.resume and path.exists()):
            run([ROOT / 'apps/benchmark_quantization.py', '--features', features, '--metric', 'cosine',
                 '--device', 'cpu', '--seed', rotation, '--k', 10, '--block-size', 256,
                 '--repeats', 3, '--methods', *methods, '--output', path],
                output / f'{name}_rotation{rotation}.log')
            report = json.loads(path.read_text())
            report['runner_provenance'] = expected
            save_json(path, report)
        report = json.loads(path.read_text())
        if ([r['method'] for r in report['results']] != methods
                or report.get('runner_provenance') != expected):
            raise ValueError('Missing gallery baseline/compression method or mismatched resume provenance')
        return report['results']

    records, retrieval, metadata = [], [], []
    checkpoint_check, check_metadata = [], []
    common_ids = None
    for index, seed in enumerate(seeds):
        pair = {}
        for optimizer in (('adamw', 'muon') if index % 2 == 0 else ('muon', 'adamw')):
            if {p: digest(ROOT / p) for p in source_hashes} != source_hashes:
                raise ValueError('Sources changed during the run')
            name = f'{optimizer}_seed{seed}'
            training = output / f'{name}_train.json'
            best, terminal = output / f'{name}_best.npz', output / f'{name}_terminal.npz'
            shared = ['--optimizer', optimizer, '--seed', seed, '--data-dir', args.data_dir,
                      '--train-limit', train_rows, '--validation-size', validation_rows,
                      '--batch-size', 64, '--extra-hidden-layers', 2, '--output', training,
                      '--best-checkpoint-output', best, '--checkpoint-output', terminal]
            if args.study == 'epochs':
                command = [ROOT / 'apps/train_baseline.py', '--dataset', 'Fashion-MNIST',
                           '--device', 'cpu', '--epochs', epochs, '--eval-limit', validation_rows,
                           '--lr', .001, '--muon-lr', .007, '--weight-decay', .01, *shared]
            else:
                command = [ROOT / 'apps/train_timed.py', '--budget-seconds', protocol['budget_seconds'], *shared]
            if not (args.resume and training.exists()):
                run(command, output / f'{name}_train.log')
                record = json.loads(training.read_text())
                record.update(runner_provenance=provenance, best_checkpoint_sha256=digest(best),
                              terminal_checkpoint_sha256=digest(terminal))
                save_json(training, record)
            record = json.loads(training.read_text())
            if (record.get('runner_provenance') != provenance
                    or record['best_checkpoint_sha256'] != digest(best)
                    or record['terminal_checkpoint_sha256'] != digest(terminal)):
                raise ValueError('Training/checkpoint hashes differ from frozen protocol')
            pair[optimizer] = record
            records.append(record)
            checkpoint = best if args.checkpoint == 'best' else terminal
            features, feature = exported(name, checkpoint)
            ids = feature['gallery_ids_sha256'], feature['query_ids_sha256']
            if common_ids is None:
                common_ids = ids
            if ids != common_ids:
                raise ValueError('Encoders used different retrieval images')
            metadata.append(dict(feature, encoder=optimizer, training_seed=seed))
            for rotation in ((9, 8, 7) if seed == 9 else (7, 8, 9)):
                methods = METHODS if rotation == 7 else ['turbo_2', 'turbo_4']
                retrieval.extend(dict(r, encoder=optimizer, training_seed=seed, rotation_seed=rotation)
                                 for r in benchmark(name, features, feature, rotation, methods))
            if args.checkpoint_check:
                other = 'terminal' if args.checkpoint == 'best' else 'best'
                alternative = terminal if other == 'terminal' else best
                alternate_features, alternate_metadata = exported(name + '_' + other, alternative)
                if (alternate_metadata['gallery_ids_sha256'], alternate_metadata['query_ids_sha256']) != common_ids:
                    raise ValueError('Checkpoint sensitivity used different retrieval images')
                check_metadata.append(dict(alternate_metadata, encoder=optimizer, training_seed=seed))
                checkpoint_check.extend(dict(r, encoder=optimizer, training_seed=seed, rotation_seed=7)
                                        for r in benchmark(name + '_' + other, alternate_features,
                                                           alternate_metadata, 7, ['fp32']))
            print('DONE ' + name, flush=True)
        (validate_pair if args.study == 'epochs' else validate_timed_pair)(pair['adamw'], pair['muon'])
    result = {'protocol': protocol, 'training_records': records, 'retrieval_rows': retrieval,
              'feature_metadata': metadata, 'common_image_ids_verified': True,
              'checkpoint_check_rows': checkpoint_check, 'checkpoint_check_metadata': check_metadata}
    if set(seeds) == {8, 9, 10}:
        result['quality_summary'] = aggregate_quality(retrieval)
        if checkpoint_check:
            result['checkpoint_check_summary'] = aggregate_quality(checkpoint_check)
    save_json(output / 'run.json', result)
    print('Saved:', output / 'run.json')


if __name__ == '__main__':
    main()
