"""Generate exactly two tables from the six-model CIFAR evidence."""
import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from common import distribution, first_crossing

SEEDS = (9, 10, 11)
METHODS = ('fp32', 'uniform_2', 'uniform_4', 'compact_2', 'compact_4',
           'symmetric_2', 'symmetric_4', 'turbo_2', 'turbo_4')
DISPLAY = ('fp32', 'uniform_2', 'symmetric_2', 'turbo_2',
           'uniform_4', 'symmetric_4', 'turbo_4')
LABELS = {'fp32': 'FP32', 'uniform_2': 'Affine uniform 2-bit',
          'uniform_4': 'Affine uniform 4-bit', 'symmetric_2': 'Symmetric uniform 2-bit',
          'symmetric_4': 'Symmetric uniform 4-bit', 'turbo_2': 'TurboQuant-MSE 2-bit',
          'turbo_4': 'TurboQuant-MSE 4-bit'}


def training_rows(records, targets):
    indexed = {(r['optimizer'], r['seed']): r for r in records}
    if len(indexed) != len(records) or set(indexed) != {(m, s) for m in ('adamw', 'muon') for s in SEEDS}:
        raise ValueError('Expected all six unique training trajectories')
    rows = []
    for target in targets:
        row = dict(target=target)
        for method in ('adamw', 'muon'):
            hits = [dict(seed=s, **first_crossing(indexed[method, s]['curve'], target)) for s in SEEDS]
            complete = all(h['reached'] for h in hits)
            row[method] = dict(reached_count=sum(h['reached'] for h in hits), total_count=3, per_seed=hits,
                               **{k: distribution([h[k] for h in hits]) if complete else None
                                  for k in ('steps', 'training_seconds')})
        rows.append(row)
    return rows


def validate_sources(data):
    for path, expected in data['provenance']['packaged_source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Archived algorithm source changed: ' + path)
    for name, info in data['provenance']['preserved_functions'].items():
        source = (ROOT / info['packaged_path']).read_text()
        node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
        if hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() != info['sha256']:
            raise ValueError('Preserved measurement function changed: ' + name)


def build_tables(data):
    records = data['training']['trajectories']
    rows = training_rows(records, data['training']['display_targets'])
    if [r['target'] for r in rows] != [.08, .10, .12, .14]:
        raise ValueError('The agreed supplementary targets changed')
    indexed = {(r['optimizer'], r['seed']): r for r in records}
    for record in records:
        if (record['status'] != 'complete' or not 600 <= record['training_seconds'] < 601
                or record['curve'][-1]['training_seconds'] != record['training_seconds']
                or record['curve'][-1]['steps'] != record['steps']):
            raise ValueError('Training endpoint or full budget differs')
        for point in record['curve']:
            if abs(point['precision'] - point['same_category_matches'] / 10000) > 1e-12:
                raise ValueError('Precision does not match raw category counts')
    for seed in SEEDS:
        a, m = indexed['adamw', seed], indexed['muon', seed]
        if a['initial_parameter_sha256'] != m['initial_parameter_sha256']:
            raise ValueError('Paired initial model differs')
        for field in ('train_ids_sha256', 'validation_ids_sha256', 'gallery_ids_sha256', 'query_ids_sha256'):
            if a[field] != m[field]:
                raise ValueError('Paired data split differs')
    trials = data['quantization']['trials']
    keyed = {(r['encoder'], r['training_seed'], r['method'], r['rotation_seed']): r for r in trials}
    expected = {(m, s, q, rotation) for m in ('adamw', 'muon') for s in SEEDS for q in METHODS
                for rotation in ([7, 8, 9] if q.startswith('turbo') else [7])}
    if len(trials) != len(expected) or set(keyed) != expected:
        raise ValueError('Incomplete or duplicated encoder/seed/format/rotation trials')
    for trial in trials:
        record = indexed[trial['encoder'], trial['training_seed']]
        if trial['feature_archive_sha256'] != record['feature_archive_sha256']:
            raise ValueError('Gallery comes from a different model feature archive')
        if trial['terminal_parameter_sha256'] != record['terminal_parameter_sha256']:
            raise ValueError('Gallery does not use the actual training endpoint')
        if (trial['stored_bytes'] != sum(trial['storage_breakdown'].values())
                or abs(trial['label_precision_at_k'] - trial['same_category_matches'] / 10000) > 1e-12
                or len(trial['query_seconds']) != 3 or min(trial['query_seconds']) <= 0):
            raise ValueError('Gallery byte count, precision or timing differs')
        if trial['method'] == 'fp32' and (
                trial['stored_bytes'] != 2560000
                or abs(trial['label_precision_at_k'] - record['curve'][-1]['precision']) > 1e-12):
            raise ValueError('FP32 gallery does not reproduce the training endpoint')
    for method in ('adamw', 'muon'):
        for seed in SEEDS:
            for bits in (2, 4):
                a, b = [keyed[method, seed, f'{name}_{bits}', 7] for name in ('uniform', 'compact')]
                if any(a[k] != b[k] for k in ('neighbor_indices_sha256', 'label_precision_at_k', 'recall_at_k')):
                    raise ValueError('Compact affine changes retrieval values')
    quantization = []
    for method in DISPLAY:
        selected = [r for r in trials if r['method'] == method]
        sizes = {r['stored_bytes'] for r in selected}
        if len(sizes) != 1:
            raise ValueError('Model or rotation changed the gallery format size')
        size = sizes.pop()
        row = dict(method=method, stored_bytes=size, compression=2560000 / size)
        for encoder in ('adamw', 'muon'):
            metrics = {}
            for source, field in [('label_precision_at_k', 'precision'), ('recall_at_k', 'recall'),
                                  ('query_seconds_median', 'query_seconds')]:
                values = [statistics.mean(r[source] for r in selected
                                          if (r['encoder'], r['training_seed']) == (encoder, seed)) for seed in SEEDS]
                metrics[field] = distribution(values)
            row[encoder] = metrics
        quantization.append(row)
    return rows, quantization


def estimate(value, scale=1, digits=2):
    return f"{scale * value['mean']:.{digits}f} ± {scale * value['sample_std']:.{digits}f}"


def render_training(rows):
    lines = ['| Target P@10 | AdamW updates | AdamW training s | Muon updates | Muon training s |',
             '|---|---|---|---|---|']
    for row in rows:
        cells = [f"{100 * row['target']:.0f}%"]
        for method in ('adamw', 'muon'):
            result = row[method]
            if result['reached_count'] == 3:
                cells += [estimate(result['steps'], digits=0), estimate(result['training_seconds'])]
            else:
                cells += [f"{result['reached_count']}/3 reached", 'Not observed within 600 s']
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def render_quantization(rows):
    lines = ['| Gallery format | AdamW P@10 (%) | Muon P@10 (%) | Gallery MB | Compression |',
             '|---|---|---|---|---|']
    for row in rows:
        cells = [LABELS[row['method']], estimate(row['adamw']['precision'], 100),
                 estimate(row['muon']['precision'], 100), f"{row['stored_bytes'] / 1e6:.4f}",
                 f"{row['compression']:.2f}×"]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def csv_text(rows, kind):
    buffer = io.StringIO(newline='')
    if kind == 'training':
        columns = ['target', 'adamw_reached', 'adamw_updates', 'adamw_seconds',
                   'muon_reached', 'muon_updates', 'muon_seconds']
        values = []
        for row in rows:
            value = dict(target=row['target'])
            for method in ('adamw', 'muon'):
                result = row[method]
                value[method + '_reached'] = result['reached_count']
                for key, suffix in [('steps', 'updates'), ('training_seconds', 'seconds')]:
                    value[method + '_' + suffix] = result[key]['mean'] if result[key] else ''
            values.append(value)
    else:
        columns = ['method', 'adamw_precision', 'adamw_precision_sd', 'muon_precision',
                   'muon_precision_sd', 'stored_bytes', 'compression']
        values = [dict(method=r['method'], stored_bytes=r['stored_bytes'], compression=r['compression'],
                       **{m + suffix: r[m]['precision'][field] for m in ('adamw', 'muon')
                          for suffix, field in [('_precision', 'mean'), ('_precision_sd', 'sample_std')]}) for r in rows]
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n')
    writer.writeheader(); writer.writerows(values)
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = json.loads((ROOT / 'report/evidence.json').read_text())
    validate_sources(data)
    training, quantization = build_tables(data)
    readme = ROOT / 'README.md'; current = readme.read_text(); wanted = current
    for name, text in [('TRAINING', render_training(training)), ('QUANTIZATION', render_quantization(quantization))]:
        before, rest = wanted.split(f'<!-- {name}:START -->\n')
        _, after = rest.split(f'<!-- {name}:END -->')
        wanted = before + f'<!-- {name}:START -->\n' + text + f'\n<!-- {name}:END -->' + after
    outputs = {readme: wanted, ROOT / 'report/training.csv': csv_text(training, 'training'),
               ROOT / 'report/quantization.csv': csv_text(quantization, 'quantization')}
    for path, text in outputs.items():
        if args.check:
            if not path.exists() or path.read_text() != text: parser.exit(1, 'Generated table is stale: ' + str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    print('Validated two tables: 6 training trajectories and 78 gallery trials; all paired endpoints match.')


if __name__ == '__main__': main()
