"""Validate the fresh study and build one consolidated README/CSV table."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as stats
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from common import (SEEDS, aggregate_quality, distribution, select_configuration,
                    validate_pair, validate_timed_pair)
from benchmark_system import validate_measurements

STORES = {'fp32': 'FP32', 'uniform_2': 'Uniform 2-bit', 'uniform_4': 'Uniform 4-bit',
          'turbo_2': 'TurboQuant 2-bit', 'turbo_4': 'TurboQuant 4-bit'}
OPTIMIZERS = {'adamw': 'AdamW', 'muon': 'Muon + aux AdamW'}
METHOD_KEYS = {'adamw': 'adamw', 'muon': 'muon', 'AdamW': 'adamw', 'Muon+AdamW': 'muon'}
HEADERS = ['Study', 'Optimizer', 'Gallery format', 'P@10 (%)', 'Δ vs FP32 (pp)',
           'Own-FP32 R@10 (%)', 'Gallery MB', 'Compression', 'Training cost',
           'Updates', 'Optimizer state MB', 'Query s / 1k']


def validate_retrieval(rows):
    for row in rows:
        if row['device'] != 'cpu()' or row['search_metric'] != 'cosine' or not row['normalization_timed']:
            raise ValueError('Expected native cosine retrieval with normalization timed')
        if any(not math.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in ('label_precision_at_k', 'recall_at_k')):
            raise ValueError('Invalid retrieval quality')
        times = row['query_seconds']
        if len(times) != 3 or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError('Expected three positive query timings')
        if stats.median(times) != row['query_seconds_median']:
            raise ValueError('Query median does not match raw timings')
        if row['stored_bytes'] != sum(row['storage_breakdown'].values()):
            raise ValueError('Storage components do not match')


def training_by_seed(records, timed=False):
    indexed = {(METHOD_KEYS[r['optimizer']], r['seed']): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(m, s) for m in OPTIMIZERS for s in SEEDS}:
        raise ValueError('Expected all six unique paired training records')
    for seed in SEEDS:
        (validate_timed_pair if timed else validate_pair)(indexed['adamw', seed], indexed['muon', seed])
    return indexed


def quality_rows(study, quality, costs):
    result = []
    for optimizer in OPTIMIZERS:
        for method in STORES:
            if method not in quality['encoders'][optimizer]:
                continue
            metric = quality['encoders'][optimizer][method]
            if len(set(metric['stored_bytes']['values'])) != 1:
                raise ValueError('Persistent format bytes changed across training seeds')
            byte_count = metric['stored_bytes']['mean']
            result.append(dict(study=study, optimizer=OPTIMIZERS[optimizer], store=STORES[method],
                               precision_mean=metric['label_precision_at10']['mean'],
                               precision_sd=metric['label_precision_at10']['sample_std'],
                               recall_mean=metric['neighbor_recall_at10']['mean'],
                               recall_sd=metric['neighbor_recall_at10']['sample_std'],
                               gallery_bytes=byte_count, compression=14080000 / byte_count,
                               training_cost=costs[optimizer]['mean'],
                               training_sd=costs[optimizer]['sample_std'],
                               training_unit='s', speedup=None,
                               query_mean=metric['query_batch_seconds']['mean'],
                               query_sd=metric['query_batch_seconds']['sample_std']))
    return result


def json_hash(protocol):
    return hashlib.sha256((json.dumps(protocol, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def validate_fresh_study(run, name):
    """Validate complete paired runs without requiring ignored model artifacts."""
    p = run['protocol']
    expected = dict(dataset='Fashion-MNIST', study=name, smoke=False, checkpoint='best',
                    checkpoint_check=True, epochs=10, budget_seconds=220., train_rows=55000,
                    validation_rows=5000, gallery_rows=55000, query_rows=1000, split_seed=17,
                    architecture=[784, 128, 128, 128, 64, 10], batch_size=64,
                    rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                    rotation_seeds=[7, 8, 9], training_seeds=list(SEEDS), methods=list(STORES))
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Fresh study protocol differs from the locked comparison')
    if {k: p['backend'].get(k) for k in ('name', 'dtype', 'cpu_matmul', 'tiled_ns')} != dict(
            name='native_cpu', dtype='float32', cpu_matmul='strided', tiled_ns=True):
        raise ValueError('Fresh runs require the same final native CPU backend')
    indexed = training_by_seed(run['training_records'], timed=name == 'timed')
    stamp = {'protocol_sha256': json_hash(p)}
    for (method, seed), r in indexed.items():
        if (r.get('runner_provenance') != stamp or r['train_samples'] != 55000
                or r['eval_samples'] != 5000 or r['extra_hidden_layers'] != 2
                or r['device'] != 'cpu()' or r['backend'] != 'nd'
                or r['total_parameter_count'] != 142410 or r['learning_rate'] != .001
                or r['muon_parameter_count'] != (40960 if method == 'muon' else 0)
                or r['muon_learning_rate'] != (.007 if method == 'muon' else None)
                or r['weight_decay'] != .01
                or r['optimizer_numeric_state_bytes'] != (1139280 if method == 'adamw' else 975440)):
            raise ValueError('Fresh training provenance, shape or numeric state differs')
        if name == 'epochs':
            epochs = r['epochs']
            if (len(epochs) != 10 or any(e['optimizer_steps'] != 860 for e in epochs)
                    or any(not math.isfinite(e['training_seconds']) or e['training_seconds'] <= 0 for e in epochs)
                    or r['best_validation_epoch'] != 1 + min(range(10), key=lambda i: epochs[i]['validation_loss'])):
                raise ValueError('Incomplete epoch budget or incorrect checkpoint rule')
        elif (r['time_budget_seconds'] != 220. or not 220 <= r['timed_seconds'] <= 220 + r['last_step_seconds'] + .001
              or r['optimizer_steps'] <= 0
              or r['best_checkpoint_index'] != min(range(len(r['checkpoints'])),
                                                  key=lambda i: r['checkpoints'][i]['validation_loss'])):
            raise ValueError('Invalid wall-clock budget or checkpoint selection')
    expected_models = {(m, s) for m in OPTIMIZERS for s in SEEDS}
    common_ids = None
    for key, checkpoint_field in (('feature_metadata', 'best_checkpoint_sha256'),
                                  ('checkpoint_check_metadata', 'terminal_checkpoint_sha256')):
        metadata = run[key]
        if len(metadata) != 6 or {(r['encoder'], r['training_seed']) for r in metadata} != expected_models:
            raise ValueError('Incomplete own-model feature provenance')
        for r in metadata:
            if (r['checkpoint_sha256'] != indexed[r['encoder'], r['training_seed']][checkpoint_field]
                    or r.get('runner_provenance') != stamp or r['gallery_shape'] != [55000, 64]
                    or r['query_shape'] != [1000, 64] or r['validation_size_excluded'] != 5000
                    or r['split_seed'] != 17 or r['data_files_sha256'] != p['data_files_sha256']):
                raise ValueError('Own-model checkpoint, data or feature dimensions differ')
            ids = r['gallery_ids_sha256'], r['query_ids_sha256']
            if common_ids is None:
                common_ids = ids
            if ids != common_ids:
                raise ValueError('Fresh models used different retrieval images')
    for key, sensitivity in (('retrieval_rows', False), ('checkpoint_check_rows', True)):
        rows = run[key]
        expected_keys = {(m, s, q, rotation) for m in OPTIMIZERS for s in SEEDS
                         for q in (['fp32'] if sensitivity else STORES)
                         for rotation in ((7, 8, 9) if q.startswith('turbo') else (7,))}
        keys = {(r['encoder'], r['training_seed'], r['method'], r['rotation_seed']) for r in rows}
        if len(rows) != len(expected_keys) or keys != expected_keys:
            raise ValueError('Missing or duplicate optimizer/store/rotation trial')
        validate_retrieval(rows)
    quality, check = aggregate_quality(run['retrieval_rows']), aggregate_quality(run['checkpoint_check_rows'])
    if quality != run['quality_summary'] or check != run['checkpoint_check_summary']:
        raise ValueError('Fresh aggregates differ from raw trials')
    return quality, check, indexed, common_ids


def collect_fresh_rows(data):
    """One table: all current-backend methods, terminal checks, and total systems cost."""
    if set(data['studies']) != {'epochs', 'timed'}:
        raise ValueError('Expected both equal-epoch and equal-time studies')
    rows, checks, backends, images, sources, datasets = [], [], [], [], [], []
    for name in ('epochs', 'timed'):
        run = data['studies'][name]
        quality, terminal, indexed, common_ids = validate_fresh_study(run, name)
        p = run['protocol']
        backends.append(p['backend'])
        images.append(common_ids)
        sources.append(p['source_sha256'])
        datasets.append(p['data_files_sha256'])
        costs, updates, states = {}, {}, {}
        for m in OPTIMIZERS:
            records = [indexed[m, seed] for seed in SEEDS]
            costs[m] = distribution([sum(e['training_seconds'] for e in r['epochs'])
                                     if name == 'epochs' else r['timed_seconds'] for r in records])
            updates[m] = distribution([sum(e['optimizer_steps'] for e in r['epochs'])
                                       if name == 'epochs' else r['optimizer_steps'] for r in records])
            states[m] = records[0]['optimizer_numeric_state_bytes']
        title = '10 epochs' if name == 'epochs' else '220s'
        for destination, summary, label in ((rows, quality, title + ' / final CPU'),
                                            (checks, terminal, title + ' / terminal check')):
            for row in quality_rows(label, summary, costs):
                m = next(m for m, label in OPTIMIZERS.items() if label == row['optimizer'])
                q = next(q for q, label in STORES.items() if label == row['store'])
                delta = summary['encoders'][m][q]['precision_change_vs_own_fp32']
                row.update(precision_change_mean=delta['mean'], precision_change_sd=delta['sample_std'],
                           updates_mean=updates[m]['mean'], updates_sd=updates[m]['sample_std'],
                           optimizer_state_bytes=states[m])
                destination.append(row)
    if any(v != backends[0] for v in backends) or any(v != sources[0] for v in sources):
        raise ValueError('Fresh studies used different native binaries or source code')
    if any(v != images[0] for v in images) or any(v != datasets[0] for v in datasets):
        raise ValueError('Fresh studies used different images or data files')
    by_budget = [training_by_seed(data['studies'][name]['training_records'], timed=name == 'timed')
                 for name in ('epochs', 'timed')]
    for key in by_budget[0]:
        for field in ('initial_parameter_sha256', 'train_indices_sha256', 'validation_indices_sha256'):
            if by_budget[0][key][field] != by_budget[1][key][field]:
                raise ValueError('Training initialization or image selections differ across budgets')
    system = data['system']
    summary = validate_measurements(system['runs'])
    if summary != system['summary'] or len(system['runs']) != 2 * system['protocol']['blocks']:
        raise ValueError('System paired timing differs from raw measurements')
    new_binary = next(r['native_extension_sha256'] for r in system['runs'] if r['variant'] == 'new')
    if new_binary != backends[0]['extension_sha256']:
        raise ValueError('Systems and quality runs used different final binaries')
    for path in set(system['protocol']['source_sha256']) & set(sources[0]):
        if system['protocol']['source_sha256'][path] != sources[0][path]:
            raise ValueError('Systems and quality runs used different shared source code')
    for variant in ('old', 'new'):
        for m in OPTIMIZERS:
            metric = summary['training_' + m]
            checks.append(dict(study='128 updates / ' + ('original CPU' if variant == 'old' else 'final CPU'),
                               optimizer=OPTIMIZERS[m], store='—', precision_mean=None, precision_sd=None,
                               recall_mean=None, recall_sd=None, gallery_bytes=None, compression=None,
                               training_cost=1000 * metric[variant + '_seconds']['median'], training_sd=None,
                               training_unit='ms/step', speedup=metric['speedup']['median'] if variant == 'new' else None,
                               query_mean=None, query_sd=None, precision_change_mean=None, precision_change_sd=None,
                               updates_mean=128, updates_sd=0, optimizer_state_bytes=None))
    return rows + checks


def collect_rows(data):
    if data['schema_version'] != 2:
        raise ValueError('Expected the fresh study evidence schema')
    return collect_fresh_rows(data)


def display_rows(rows):
    def pm(mean, sd, scale=1, digits=2):
        return '—' if mean is None else f'{scale*mean:.{digits}f} ± {scale*sd:.{digits}f}'
    for r in rows:
        cost = (f"{r['training_cost']:.3f} ms/step" if r['training_sd'] is None else
                pm(r['training_cost'], r['training_sd'], digits=3) + ' s')
        if r['speedup'] is not None:
            cost += f" ({r['speedup']:.3f}×)"
        delta = ('—' if r['precision_change_mean'] is None else
                 f"{100*r['precision_change_mean']:+.2f} ± {100*r['precision_change_sd']:.2f}")
        updates = f"{r['updates_mean']:,.0f}"
        if r['updates_sd']:
            updates += f" ± {r['updates_sd']:,.0f}"
        yield [r['study'], r['optimizer'], r['store'], pm(r['precision_mean'], r['precision_sd'], 100),
               delta, pm(r['recall_mean'], r['recall_sd'], 100),
               '—' if r['gallery_bytes'] is None else f"{r['gallery_bytes']/1e6:.4f}",
               '—' if r['compression'] is None else f"{r['compression']:.2f}×", cost, updates,
               '—' if r['optimizer_state_bytes'] is None else f"{r['optimizer_state_bytes']/1e6:.4f}",
               pm(r['query_mean'], r['query_sd'], digits=3)]


def render_markdown(rows):
    lines = ['| ' + ' | '.join(HEADERS) + ' |', '|' + '|'.join(['---'] * len(HEADERS)) + '|']
    return '\n'.join(lines + ['| ' + ' | '.join(row) + ' |' for row in display_rows(rows)])


def render_csv(rows):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(HEADERS)
    writer.writerows(display_rows(rows))
    return output.getvalue()


def verify_history(data):
    revision = data['provenance']['previous_report_commit']
    previous = json.loads(subprocess.check_output(['git', 'show', f'{revision}:report/results.json'], cwd=ROOT))
    if (data['needle_origin'] != previous['needle_origin']
            or data['locked_rate_selection'] != previous['fixed_epochs']['selection']):
        raise ValueError('Origin or locked hyperparameter selection differs from history')
    baseline = data['system']['protocol']
    for path, expected in baseline['baseline_source_sha256'].items():
        raw = subprocess.check_output(['git', 'show', f'{baseline["baseline_commit"]}:{path}'], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original-backend source differs from pinned history')
    return


def verify_artifacts(data):
    """Independently reload all selected/terminal models and their own feature samples.

    Run after retained timings: no training, search timing or checkpoint selection
    occurs here. Local ignored artifacts are required for this optional audit.
    """
    if data['schema_version'] != 2:
        raise ValueError('Artifact audit requires the fresh study dataset')
    import sysconfig
    import numpy as np
    sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
    import needle as ndl
    from needle import nn
    from encoder_checkpoint import load_encoder
    from train_baseline import select_training_indices

    def check_hash(path, expected):
        path = Path(path)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Local artifact hash mismatch: ' + str(path))

    extension = ROOT / 'python/needle/backend_ndarray' / ('ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    check_hash(extension, data['studies']['epochs']['protocol']['backend']['extension_sha256'])
    directory = ROOT / 'data/fashion-mnist'
    train = ndl.data.MNISTDataset(str(directory / 'train-images-idx3-ubyte.gz'),
                                 str(directory / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(directory / 't10k-images-idx3-ubyte.gz'),
                                str(directory / 't10k-labels-idx1-ubyte.gz'))
    proof = dict(trained_models=0, checkpoint_validation_reloads=0,
                 feature_archives=0, feature_sample_rows=0, raw_query_reports=0)
    query_environment = None
    criterion = nn.SoftmaxLoss()
    for name, run in data['studies'].items():
        p = run['protocol']
        stamp = {'protocol_sha256': json_hash(p)}
        for path, expected in p['data_files_sha256'].items():
            check_hash(directory / path, expected)
        check_hash(ROOT / data['provenance']['raw_run_paths'][name], data['provenance']['raw_run_sha256'][name])
        indexed = training_by_seed(run['training_records'], timed=name == 'timed')
        for (method, seed), record in indexed.items():
            root = Path(record['best_checkpoint_export']).parent
            disk_record = json.loads((root / f'{method}_seed{seed}_train.json').read_text())
            if disk_record != record or json.loads((root / 'protocol.json').read_text()) != p:
                raise ValueError('Training record or frozen protocol differs from local originals')
            check_hash(record['best_checkpoint_export'], record['best_checkpoint_sha256'])
            check_hash(record['checkpoint_export'], record['terminal_checkpoint_sha256'])
            train_ids, val_ids, _ = select_training_indices(len(train), len(train), 55000, 5000,
                                                           np.random.default_rng(seed), 5000, 17)
            for ids, field in ((train_ids, 'train_indices_sha256'), (val_ids, 'validation_indices_sha256')):
                if hashlib.sha256(ids.astype('int64').tobytes()).hexdigest() != record[field]:
                    raise ValueError('Training/validation selection cannot be reproduced')
            proof['trained_models'] += 1
        for metadata_key, rows_key, terminal in (('feature_metadata', 'retrieval_rows', False),
                                                  ('checkpoint_check_metadata', 'checkpoint_check_rows', True)):
            for metadata in run[metadata_key]:
                method, seed = metadata['encoder'], metadata['training_seed']
                r = indexed[method, seed]
                checkpoint = Path(r['checkpoint_export'] if terminal else r['best_checkpoint_export'])
                features = Path(metadata['feature_archive'])
                check_hash(checkpoint, metadata['checkpoint_sha256'])
                check_hash(features, metadata['feature_archive_sha256'])
                if json.loads(features.with_suffix('.json').read_text()) != {
                        k: v for k, v in metadata.items() if k not in ('encoder', 'training_seed')}:
                    raise ValueError('Curated feature metadata differs from original')
                model = load_encoder(checkpoint, ndl.cpu())
                model.eval()
                val_ids = np.random.default_rng(17).permutation(len(train))[:5000]
                total_loss, correct = 0., 0
                for start in range(0, len(val_ids), 64):
                    ids = val_ids[start:start + 64]
                    x = ndl.Tensor(train.X[ids], device=ndl.cpu(), requires_grad=False)
                    y = ndl.Tensor(train.y[ids], device=ndl.cpu(), requires_grad=False)
                    scores = model(x)
                    total_loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * len(ids)
                    correct += int((scores.numpy().argmax(1) == train.y[ids]).sum())
                expected = (r['epochs'][-1] if name == 'epochs' else r['checkpoints'][-1]) if terminal else {
                    'validation_loss': r['best_validation_loss'], 'validation_accuracy': r['best_validation_accuracy']}
                if (abs(total_loss / 5000 - expected['validation_loss']) > 1e-7
                        or correct / 5000 != expected['validation_accuracy']):
                    raise ValueError('Reloaded checkpoint does not reproduce full validation')
                proof['checkpoint_validation_reloads'] += 1
                with np.load(features, allow_pickle=False) as arrays:
                    gallery_ids, query_ids = arrays['gallery_ids'], arrays['query_ids']
                    if (len(np.unique(gallery_ids)) != 55000 or np.intersect1d(gallery_ids, val_ids).size
                            or len(np.unique(query_ids)) != 1000):
                        raise ValueError('Retrieval IDs overlap validation or contain duplicates')
                    for ids, field in ((gallery_ids, 'gallery_ids_sha256'), (query_ids, 'query_ids_sha256')):
                        if hashlib.sha256(ids.astype('int64').tobytes()).hexdigest() != metadata[field]:
                            raise ValueError('Retrieval image IDs differ from frozen hashes')
                    np.testing.assert_array_equal(arrays['gallery_labels'], train.y[gallery_ids])
                    np.testing.assert_array_equal(arrays['query_labels'], test.y[query_ids])
                    for key, dataset, ids in (('gallery', train, gallery_ids), ('queries', test, query_ids)):
                        values = arrays[key]
                        if not np.isfinite(values).all():
                            raise ValueError('Feature archive contains nonfinite values')
                        sample = model.features(ndl.Tensor(dataset.X[ids[:64]], device=ndl.cpu(), requires_grad=False)).numpy()
                        np.testing.assert_array_equal(sample, values[:64])
                        proof['feature_sample_rows'] += 64
                proof['feature_archives'] += 1
                prefix = f'{method}_seed{seed}' + ('_terminal' if terminal else '')
                for rotation in ((7,) if terminal else (7, 8, 9)):
                    report = json.loads((features.parent / f'{prefix}_rotation{rotation}.json').read_text())
                    if query_environment is None:
                        query_environment = report['environment']
                    if report['environment'] != query_environment:
                        raise ValueError('Query workers used different software environments')
                    if report['runner_provenance'] != dict(stamp, feature_archive_sha256=metadata['feature_archive_sha256']):
                        raise ValueError('Query report refers to a different feature archive or protocol')
                    original = [dict(row, encoder=method, training_seed=seed, rotation_seed=rotation)
                                for row in report['results']]
                    retained = [row for row in run[rows_key] if row['encoder'] == method
                                and row['training_seed'] == seed and row['rotation_seed'] == rotation]
                    if original != retained:
                        raise ValueError('Curated query rows differ from original worker output')
                    proof['raw_query_reports'] += 1
    proof['query_environment'] = query_environment
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    parser.add_argument('--verify-artifacts', action='store_true', help='Audit original local models/features/query reports after retained timings')
    args = parser.parse_args()
    evidence = ROOT / 'report/results.json'
    data = json.loads(evidence.read_text())
    rows = collect_rows(data)
    if json_hash(data['system']) != data['provenance']['system_raw_sha256']:
        raise ValueError('Curated systems data differs from its frozen raw report')
    for name, study in data['studies'].items():
        if json_hash(study) != data['provenance']['raw_run_sha256'][name]:
            raise ValueError('Curated study differs from the frozen raw run: ' + name)
        for path, expected in study['protocol']['source_sha256'].items():
            if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                raise ValueError('Measured quality source changed: ' + path)
    for path, expected in data['system']['protocol']['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Measured system source changed: ' + path)
    for method in OPTIMIZERS:
        selection = data['locked_rate_selection']
        if select_configuration(selection['all_candidates'][method]) != selection['selected'][method]:
            raise ValueError('Locked learning-rate selection rule is inconsistent')
    if args.verify_history:
        verify_history(data)
    if args.verify_artifacts:
        proof = verify_artifacts(data)
        print('Local artifact proof:', json.dumps(proof, sort_keys=True))
        if not args.check:
            data['provenance']['artifact_validation'] = proof
            evidence.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    readme = ROOT / 'README.md'
    current = readme.read_text()
    before, rest = current.split('<!-- RESULTS:START -->\n')
    _, after = rest.split('<!-- RESULTS:END -->')
    wanted = before + '<!-- RESULTS:START -->\n' + render_markdown(rows) + '\n<!-- RESULTS:END -->' + after
    csv_path, csv_text = ROOT / 'report/results.csv', render_csv(rows)
    if args.check:
        if current != wanted or not csv_path.exists() or csv_path.read_text() != csv_text:
            parser.exit(1, 'README/CSV are stale;run python tools/build_report.py\n')
    else:
        readme.write_text(wanted)
        csv_path.write_text(csv_text)
    print(f'Validated {len(rows)} fresh consolidated rows;README and CSV match the evidence.')


if __name__ == '__main__':
    main()
