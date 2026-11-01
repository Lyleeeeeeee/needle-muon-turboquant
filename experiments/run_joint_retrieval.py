"""Connect fixed tuned Fashion encoders to native packed retrieval benchmarks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/joint_retrieval'
ARTIFACTS = ROOT / 'results/joint_retrieval'
SEEDS = (8, 9, 10)
METHODS = ['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4']
SOURCES = ['apps/export_features.py', 'apps/benchmark_quantization.py', 'apps/models.py',
           'apps/encoder_checkpoint.py', 'python/needle/quantization.py',
           'src/ndarray_backend_cpu.cc', 'src/quantization_cpu.h',
           'experiments/run_joint_retrieval.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen study file differs: {path}')
    else:
        save_json(path, value)


def source_hashes():
    return {p: digest(ROOT / p) for p in SOURCES}


def model_catalog():
    selection_path = ROOT / 'reports/lr_dataset_study/selection_fashion.json'
    selection = json.loads(selection_path.read_text())
    catalog = []
    for seed in SEEDS:
        for encoder in ('adamw', 'muon'):
            candidate = selection['selected'][encoder]['candidate']
            path = ROOT / f'reports/lr_dataset_study/fashion_repeat_{encoder}_{candidate}_seed{seed}.json'
            record = json.loads(path.read_text())
            checkpoint = Path(record['best_checkpoint_export'])
            if digest(checkpoint) != record['checkpoint_sha256']:
                raise ValueError('Model checkpoint differs from its fixed training record')
            assert record['dataset'] == 'Fashion-MNIST' and record['seed'] == seed
            assert record['validation_source'] == 'train_holdout' and record['split_seed'] == 17
            assert record['train_samples'] == 55000 and record['eval_samples'] == 5000
            catalog.append({'encoder': encoder, 'training_seed': seed, 'candidate': candidate,
                            'training_record': str(path.relative_to(ROOT)),
                            'training_record_sha256': digest(path),
                            'checkpoint': str(checkpoint.relative_to(ROOT)),
                            'checkpoint_sha256': record['checkpoint_sha256'],
                            'adamw_or_auxiliary_learning_rate': record['learning_rate'],
                            'muon_matrix_learning_rate': record['muon_learning_rate']})
    return catalog


def run_logged(command, path):
    with path.open('w') as log:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, OPENBLAS_NUM_THREADS='1',
                       OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1'),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def export_and_validate(catalog, protocol):
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from encoder_checkpoint import load_encoder

    data_dir = ROOT / 'data/fashion-mnist'
    train = ndl.data.MNISTDataset(str(data_dir / 'train-images-idx3-ubyte.gz'),
                                 str(data_dir / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(data_dir / 't10k-images-idx3-ubyte.gz'),
                                str(data_dir / 't10k-labels-idx1-ubyte.gz'))
    validation_ids = set(np.random.default_rng(17).permutation(len(train))[:5000])
    common = None
    exports = {}
    for item in catalog:
        encoder, seed = item['encoder'], item['training_seed']
        output = ARTIFACTS / f'{encoder}_seed{seed}_features.npz'
        report_path = REPORTS / f'features_{encoder}_seed{seed}.json'
        if not report_path.exists():
            print(f'EXPORT {encoder} seed{seed}', flush=True)
            run_logged([sys.executable, str(ROOT / 'apps/export_features.py'),
                        '--checkpoint', str(ROOT / item['checkpoint']), '--dataset', 'Fashion-MNIST',
                        '--validation-size', '5000', '--split-seed', '17', '--gallery-size', '55000',
                        '--queries', '1000', '--seed', '7', '--output', str(output)],
                       ARTIFACTS / f'export_{encoder}_seed{seed}.log')
            metadata = json.loads(output.with_suffix('.json').read_text())
        else:
            metadata = json.loads(report_path.read_text())
        assert metadata['dataset'] == 'Fashion-MNIST'
        assert metadata['checkpoint_sha256'] == item['checkpoint_sha256']
        assert metadata['feature_archive_sha256'] == digest(output)
        assert metadata['gallery_shape'] == [55000, 64] and metadata['query_shape'] == [1000, 64]
        assert metadata['validation_size_excluded'] == 5000 and metadata['split_seed'] == 17
        assert metadata['data_files_sha256'] == protocol['data_files_sha256']
        with np.load(output, allow_pickle=False) as data:
            ids = {k: data[k].copy() for k in ('gallery_ids', 'query_ids', 'gallery_labels', 'query_labels')}
            assert len(set(ids['gallery_ids'])) == 55000 and not set(ids['gallery_ids']) & validation_ids
            if common is None:
                common = ids
            else:
                for key in common:
                    np.testing.assert_array_equal(ids[key], common[key])
            model = load_encoder(ROOT / item['checkpoint'], ndl.cpu())
            for kind, dataset in [('gallery', train), ('queries', test)]:
                image_ids = ids['gallery_ids' if kind == 'gallery' else 'query_ids'][:64]
                proof = model.features(ndl.Tensor(dataset.X[image_ids], device=ndl.cpu(), requires_grad=False)).numpy()
                np.testing.assert_array_equal(proof, data[kind][:64])
            gallery = data['gallery']
            norms = np.linalg.norm(gallery.astype('float64'), axis=1)
            metadata['feature_geometry'] = {'zero_fraction': float(np.mean(gallery == 0)),
                                            'zero_vector_rows': int(np.count_nonzero(norms == 0)),
                                            'raw_l2_norm_mean': float(norms.mean()),
                                            'raw_l2_norm_std': float(norms.std())}
        metadata['encoder'] = encoder
        metadata['training_seed'] = seed
        metadata['own_checkpoint_gallery_query_samples_reproduced'] = True
        metadata['common_image_ids_and_labels_verified'] = True
        metadata['source_sha256'] = protocol['source_sha256']
        save_json(report_path, metadata)
        exports[encoder, seed] = metadata
    return exports


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    catalog = model_catalog()
    train_manifest = json.loads((ROOT / 'reports/lr_dataset_study/fashion_training_data.json').read_text())
    test_manifest = json.loads((ROOT / 'reports/lr_dataset_study/fashion_test_data.json').read_text())['files']
    data_hashes = {name: row['sha256'] for name, row in dict(train_manifest, **test_manifest).items()}
    for name, expected in data_hashes.items():
        assert digest(ROOT / 'data/fashion-mnist' / name) == expected
    protocol = {'date': '2026-11-01', 'training_source_commit': 'b397f002098f57e21aa8ad3fb5909546744eeadb',
                'dataset': 'Fashion-MNIST', 'models': catalog, 'training_seeds': list(SEEDS),
                'gallery_rows': 55000, 'query_rows': 1000, 'dimensions': 64, 'k': 10,
                'validation_size_excluded': 5000, 'split_seed': 17, 'image_selection_seed': 7,
                'rotation_seeds': [7, 8, 9], 'metric': 'cosine', 'block_size': 256,
                'warmups': 1, 'query_timing_repeats': 3, 'methods': METHODS,
                'primary_metric': 'same_label_precision_at10', 'source_sha256': source_hashes(),
                'data_files_sha256': data_hashes,
                'selection_rule': 'use_all_previously_fixed_models_no_new_selection'}
    freeze_json(REPORTS / 'protocol.json', protocol)
    exports = export_and_validate(catalog, protocol)
    print('ALL SIX OWN-MODEL EMBEDDINGS AND COMMON IMAGE IDS VERIFIED', flush=True)
    catalog_by_key = {(r['encoder'], r['training_seed']): r for r in catalog}
    for index, seed in enumerate(SEEDS):
        for encoder in (('adamw', 'muon') if index % 2 == 0 else ('muon', 'adamw')):
            metadata = exports[encoder, seed]
            for rotation in ((9, 8, 7) if seed == 9 else (7, 8, 9)):
                if source_hashes() != protocol['source_sha256']:
                    raise ValueError('Measured source changed during the study')
                methods = METHODS if rotation == 7 else ['turbo_2', 'turbo_4']
                run_id = f'{encoder}_seed{seed}_rotation{rotation}'
                output = REPORTS / f'{run_id}.json'
                provenance = {'encoder': encoder, 'training_seed': seed, 'rotation_seed': rotation,
                              'checkpoint_sha256': catalog_by_key[encoder, seed]['checkpoint_sha256'],
                              'feature_archive_sha256': metadata['feature_archive_sha256'],
                              'gallery_ids_sha256': metadata['gallery_ids_sha256'],
                              'query_ids_sha256': metadata['query_ids_sha256'],
                              'source_sha256': protocol['source_sha256']}
                if output.exists():
                    report = json.loads(output.read_text())
                    assert report['provenance'] == provenance
                else:
                    print(f'START {run_id} {methods}', flush=True)
                    raw = ARTIFACTS / f'{run_id}.json'
                    run_logged([sys.executable, str(ROOT / 'apps/benchmark_quantization.py'),
                                '--features', metadata['feature_archive'], '--metric', 'cosine',
                                '--device', 'cpu', '--seed', str(rotation), '--k', '10',
                                '--block-size', '256', '--repeats', '3', '--methods', *methods,
                                '--output', str(raw)], ARTIFACTS / f'{run_id}.log')
                    report = json.loads(raw.read_text())
                    report['provenance'] = provenance
                    save_json(output, report)
                assert report['source'] == 'Fashion-MNIST_frozen_encoder_' + provenance['checkpoint_sha256'][:12]
                assert report['gallery_shape'] == [55000, 64] and report['query_shape'] == [1000, 64]
                assert report['seed'] == rotation and report['search_metric'] == 'cosine'
                assert [r['method'] for r in report['results']] == methods
                for row in report['results']:
                    assert len(row['query_seconds']) == 3
                    assert 0 <= row['recall_at_k'] <= 1 and 0 <= row['label_precision_at_k'] <= 1
                    assert row['stored_bytes'] == sum(row['storage_breakdown'].values())
                    if row['method'] == 'fp32':
                        assert row['recall_at_k'] == 1 and row['stored_bytes'] == 55000 * 64 * 4
                print('DONE ' + run_id + ' ' + json.dumps({r['method']: r['label_precision_at_k']
                                                         for r in report['results']}), flush=True)
    print('COMPLETE:18 comparisons,54 isolated method workers; no retraining.', flush=True)


if __name__ == '__main__':
    main()
