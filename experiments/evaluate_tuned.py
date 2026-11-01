"""Evaluate fixed terminal tuned encoders with every gallery format; never train."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from common import aggregate_quality, digest, save_json
from run_report import METHODS, run
from time_to_quality import THREADS, check_frozen
from tune_batch512 import choose_rates, freeze

ROOT = Path(__file__).resolve().parents[1]


def json_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + '\n').encode()).hexdigest()


def select_terminal_snapshots(parent):
    """Choose the allocated endpoint by step count, independent of retrieval quality."""
    tuning, final, selection = parent['tuning'], parent['final'], parent['selection']
    p = final['protocol']
    if (parent['tuning_run_sha256'] != json_digest(tuning)
            or parent['final_run_sha256'] != json_digest(final)
            or parent['selection_sha256'] != json_digest(selection)
            or selection['tuning_run_sha256'] != parent['tuning_run_sha256']
            or p['selection_sha256'] != parent['selection_sha256']
            or p['tuning_run_sha256'] != parent['tuning_run_sha256']):
        raise ValueError('Parent trajectory or locked selection seal differs')
    chosen = choose_rates(tuning['candidates'])
    if chosen != selection['selected'] or chosen != p['selected'] or chosen != tuning['selected']:
        raise ValueError('Parent final rates differ from validation selection')
    records = final['trajectories']
    expected = {(m, s) for s in p['training_seeds'] for m in ('adamw', 'muon')}
    indexed = {(r['optimizer'], r['seed']): r for r in records}
    if len(records) != len(expected) or set(indexed) != expected:
        raise ValueError('Expected every unique paired terminal model')
    selected = []
    for seed in p['training_seeds']:
        a, b = indexed['adamw', seed], indexed['muon', seed]
        for field in ('initial_parameter_sha256', 'train_indices_sha256',
                      'validation_indices_sha256', 'training_order_sha256', 'epoch_order_sha256',
                      'gallery_ids_sha256', 'query_ids_sha256'):
            if a[field] != b[field]:
                raise ValueError('Paired model inputs differ: ' + field)
        for method in ('adamw', 'muon'):
            record = indexed[method, seed]
            last = record['snapshots'][-1]
            raw = {k: v for k, v in record.items() if k not in (
                'curve', 'gallery_ids_sha256', 'query_ids_sha256', 'query_source', 'training_record_sha256')}
            if (record['status'] != 'complete' or record['steps'] != p['max_steps']
                    or last['steps'] != p['max_steps'] or record['candidate'] != chosen[method]['candidate']
                    or record['protocol_sha256'] != json_digest(p)
                    or record['training_record_sha256'] != json_digest(raw)
                    or record['training_seconds'] != last['training_seconds']
                    or any(record['curve'][-1].get(k) != v for k, v in last.items())):
                raise ValueError('Terminal model budget, rates or recorded provenance differs')
            selected.append(dict(
                optimizer=method, seed=seed, candidate=record['candidate'],
                **{k: last[k] for k in ('steps', 'samples_seen', 'training_seconds', 'optimizer_seconds',
                                       'checkpoint', 'checkpoint_sha256', 'parameter_sha256')},
                optimizer_numeric_state_bytes=record['optimizer_numeric_state_bytes'],
                training_record_sha256=record['training_record_sha256'],
                gallery_ids_sha256=record['gallery_ids_sha256'], query_ids_sha256=record['query_ids_sha256'],
                expected_fp32_precision=record['curve'][-1]['precision']))
    return selected


def make_protocol(parent, parent_path, selected):
    original = parent['final']['protocol']
    inherited = ('dataset', 'smoke', 'batch_size', 'split_seed', 'train_rows', 'validation_rows',
                 'gallery_rows', 'query_rows', 'training_seeds', 'architecture', 'weight_decay',
                 'backend', 'data_files_sha256', 'selected', 'evaluation_split', 'python', 'platform')
    # Synthetic unit fixtures need not contain interpreter metadata.
    p = {k: original[k] for k in inherited if k in original}
    p.update(study='tuned_terminal_quant', checkpoint_rule='fixed_final_training_step',
             parent_run=str(parent_path), parent_tuned_run_sha256=json_digest(parent),
             parent_final_protocol_sha256=json_digest(original), max_steps=original['max_steps'],
             selected_snapshots=selected, methods=METHODS, rotation_seeds=[7, 8, 9],
             search_metric='cosine', k=10, block_size=256, repeats=3, export_batch_size=512,
             feature_selection_seed=7, thread_limits=THREADS, execution='sequential_isolated_query_workers',
             training_timing_scope=original.get('timing_scope', 'complete_training_loop'),
             query_timing_scope='query preparation/rotation, decoded-block normalization, scoring and top-k; excludes feature extraction and build',
             source_sha256=dict(original['source_sha256'], **{
                 'experiments/evaluate_tuned.py': digest(Path(__file__)),
                 'experiments/run_report.py': digest(ROOT / 'experiments/run_report.py')}))
    return p


def checked_checkpoint(phase_dir, selected):
    path = (phase_dir / selected['checkpoint']).resolve()
    if not path.is_relative_to(phase_dir.resolve()) or digest(path) != selected['checkpoint_sha256']:
        raise ValueError('Source checkpoint path or hash differs')
    training = phase_dir / f"{selected['optimizer']}_seed{selected['seed']}_train.json"
    if digest(training) != selected['training_record_sha256']:
        raise ValueError('Source training record changed')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tuned-run', type=Path, default=ROOT / 'results/tuned_batch512_20261101/run.json')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/tuned_integration_20261101')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.tuned_run, args.data_dir, output = args.tuned_run.resolve(), args.data_dir.resolve(), args.output_dir.resolve()
    parent = json.loads(args.tuned_run.read_text())
    selected = select_terminal_snapshots(parent)
    protocol = make_protocol(parent, args.tuned_run, selected)
    if digest(args.tuned_run) != protocol['parent_tuned_run_sha256']:
        raise ValueError('Parent file serialization differs from its sealed evidence')
    output.mkdir(parents=True, exist_ok=True)
    freeze(output / 'protocol.json', protocol, args.resume)
    stamp = dict(protocol_sha256=digest(output / 'protocol.json'))
    rows, metadata = [], []
    common_ids = None
    started = time.perf_counter()
    for source in selected:
        check_frozen(protocol, args.data_dir)
        if digest(args.tuned_run) != protocol['parent_tuned_run_sha256']:
            raise ValueError('Parent tuned evidence changed during evaluation')
        checkpoint = checked_checkpoint(args.tuned_run.parent / 'final', source)
        name = f"{source['optimizer']}_seed{source['seed']}"
        archive = output / f'{name}_features.npz'
        feature_path = archive.with_suffix('.json')
        if not (args.resume and feature_path.exists()):
            before = time.perf_counter()
            run([ROOT / 'apps/export_features.py', '--checkpoint', checkpoint, '--dataset', 'Fashion-MNIST',
                 '--data-dir', args.data_dir, '--validation-size', protocol['validation_rows'], '--split-seed', 17,
                 '--gallery-size', protocol['gallery_rows'], '--queries', protocol['query_rows'],
                 '--reference-train', min(4096, protocol['gallery_rows']),
                 '--reference-queries', min(256, protocol['query_rows']), '--seed', 7,
                 '--batch-size', 512, '--output', archive], output / f'{name}_export.log')
            feature = json.loads(feature_path.read_text())
            feature.update(runner_provenance=stamp, export_wall_seconds=time.perf_counter() - before)
            save_json(feature_path, feature)
        feature = json.loads(feature_path.read_text())
        ids = feature['gallery_ids_sha256'], feature['query_ids_sha256']
        if (feature['checkpoint_sha256'] != source['checkpoint_sha256']
                or feature['feature_archive_sha256'] != digest(archive)
                or feature.get('runner_provenance') != stamp
                or ids != (source['gallery_ids_sha256'], source['query_ids_sha256'])
                or feature['gallery_shape'] != [protocol['gallery_rows'], protocol['architecture'][-2]]
                or feature['query_shape'] != [protocol['query_rows'], protocol['architecture'][-2]]):
            raise ValueError('Own-model feature provenance or dimensions differ')
        if common_ids is not None and ids != common_ids:
            raise ValueError('Different models used different retrieval images')
        common_ids = ids
        metadata.append(dict(feature, encoder=source['optimizer'], training_seed=source['seed']))
        rotations = [9, 8, 7] if source['seed'] == 9 else [7, 8, 9]
        for rotation in rotations:
            check_frozen(protocol, args.data_dir)
            methods = METHODS if rotation == 7 else ['turbo_2', 'turbo_4']
            path = output / f'{name}_rotation{rotation}.json'
            expected = dict(stamp, feature_archive_sha256=feature['feature_archive_sha256'])
            if not (args.resume and path.exists()):
                before = time.perf_counter()
                run([ROOT / 'apps/benchmark_quantization.py', '--features', archive, '--metric', 'cosine',
                     '--device', 'cpu', '--seed', rotation, '--k', 10, '--block-size', 256,
                     '--repeats', 3, '--methods', *methods, '--output', path],
                    output / f'{name}_rotation{rotation}.log')
                report = json.loads(path.read_text())
                report.update(runner_provenance=expected, acquisition_wall_seconds=time.perf_counter() - before)
                save_json(path, report)
            report = json.loads(path.read_text())
            if ([r['method'] for r in report['results']] != methods
                    or report.get('runner_provenance') != expected
                    or report['gallery_shape'] != feature['gallery_shape']
                    or report['query_shape'] != feature['query_shape']
                    or report['search_metric'] != 'cosine'):
                raise ValueError('Gallery methods or resumed benchmark provenance differ')
            for row in report['results']:
                if row['method'] == 'fp32' and row['label_precision_at_k'] != source['expected_fp32_precision']:
                    raise ValueError('FP32 precision differs from the original terminal trajectory')
                rows.append(dict(row, encoder=source['optimizer'], training_seed=source['seed'],
                                 rotation_seed=rotation, runner_provenance=expected))
            print(f'DONE {name} rotation{rotation}', flush=True)
    result = dict(protocol=protocol, selected_snapshots=selected, feature_metadata=metadata,
                  retrieval_rows=rows, common_image_ids_verified=True,
                  acquisition_wall_seconds=time.perf_counter() - started)
    if protocol['training_seeds'] == [8, 9, 10]:
        result['quality_summary'] = aggregate_quality(rows)
    save_json(output / 'run.json', result)
    print('Saved:', output / 'run.json', flush=True)


if __name__ == '__main__':
    main()
