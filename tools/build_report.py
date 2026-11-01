"""Validate report evidence and generate the single README table and CSV export."""
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
HEADERS = ['Study', 'Optimizer', 'Gallery format', 'P@10 (%)', 'Own-FP32 R@10 (%)',
           'Gallery MB', 'Compression', 'Training cost', 'Query s / 1k']


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


def collect_rows(data):
    if data['schema_version'] != 1:
        raise ValueError('Unsupported evidence version')
    fixed = data['fixed_epochs']
    training = training_by_seed(fixed['training_records'])
    for optimizer in OPTIMIZERS:
        choice = select_configuration(fixed['selection']['all_candidates'][optimizer])
        if choice != fixed['selection']['selected'][optimizer]:
            raise ValueError('Learning-rate choice differs from the stated rule')
    validate_retrieval(fixed['retrieval_rows'])
    quality = aggregate_quality(fixed['retrieval_rows'])
    if any(set(quality['encoders'][m]) != set(STORES) for m in OPTIMIZERS):
        raise ValueError('Missing primary optimizer/store configuration')
    costs = {m: distribution([sum(e['training_seconds'] for e in training[m, s]['epochs'])
                              for s in SEEDS]) for m in OPTIMIZERS}
    result = quality_rows('10 epochs / original CPU', quality, costs)
    timed = data['equal_time']
    timed_training = training_by_seed(timed['training_records'], timed=True)
    validate_retrieval(timed['terminal_rows'])
    validate_retrieval(timed['selected_fp32_rows'])
    terminal = aggregate_quality(timed['terminal_rows'])
    aggregate_quality(timed['selected_fp32_rows'])
    if any(set(terminal['encoders'][m]) != {'fp32'} for m in OPTIMIZERS):
        raise ValueError('Expected FP32-only terminal sensitivity check')
    costs = {m: distribution([timed_training[m, s]['timed_seconds'] for s in SEEDS]) for m in OPTIMIZERS}
    result += quality_rows('220s terminal / original CPU (posthoc)', terminal, costs)
    system = data['system']
    summary = validate_measurements(system['runs'])
    if len(system['runs']) != 2 * system['protocol']['blocks'] or summary != system['summary']:
        raise ValueError('System aggregation differs from raw paired measurements')
    for variant in ('old', 'new'):
        for method in OPTIMIZERS:
            metric = summary['training_' + method]
            result.append(dict(study='128 updates / ' + ('original CPU' if variant == 'old' else 'final CPU'),
                               optimizer=OPTIMIZERS[method], store='—', precision_mean=None,
                               precision_sd=None, recall_mean=None, recall_sd=None,
                               gallery_bytes=None, compression=None,
                               training_cost=1000 * metric[variant + '_seconds']['median'],
                               training_sd=None, training_unit='ms/step',
                               speedup=metric['speedup']['median'] if variant == 'new' else None,
                               query_mean=None, query_sd=None))
    return result


def display_rows(rows):
    def pm(mean, sd, scale=1, digits=2):
        return '—' if mean is None else f'{scale*mean:.{digits}f} ± {scale*sd:.{digits}f}'
    for row in rows:
        training = (f"{row['training_cost']:.3f} ms/step" if row['training_sd'] is None else
                    pm(row['training_cost'], row['training_sd'], digits=3) + ' s')
        if row['speedup'] is not None:
            training += f" ({row['speedup']:.3f}×)"
        yield [row['study'], row['optimizer'], row['store'],
               pm(row['precision_mean'], row['precision_sd'], 100),
               pm(row['recall_mean'], row['recall_sd'], 100),
               '—' if row['gallery_bytes'] is None else f"{row['gallery_bytes']/1e6:.4f}",
               '—' if row['compression'] is None else f"{row['compression']:.2f}×",
               training, pm(row['query_mean'], row['query_sd'], digits=3)]


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
    provenance = data['provenance']
    snapshot = provenance['archive_snapshot_commit']
    originals = {}
    for path, expected in provenance['historical_files_sha256'].items():
        raw = subprocess.check_output(['git', 'show', f'{snapshot}:{path}'], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Historical evidence hash mismatch: ' + path)
        originals[path] = json.loads(raw)
    if data['needle_origin'] != originals['sources/manifest.json']:
        raise ValueError('Needle origin differs from archived provenance')
    fixed, timed = data['fixed_epochs'], data['equal_time']
    for actual, expected in (
            (fixed['protocol'], originals['reports/depth_study/protocol.json']),
            (fixed['selection'], originals['reports/depth_study/selection.json']),
            (fixed['retrieval_protocol'], originals['reports/depth_study/retrieval/summary.json']['protocol']),
            (timed['protocol'], originals['reports/time_budget_study/protocol.json']),
            (timed['terminal_protocol'], originals['reports/time_budget_study/terminal/summary.json']['protocol']),
            (timed['selected_protocol'], originals['reports/time_budget_study/retrieval/summary.json']['protocol'])):
        if actual != expected:
            raise ValueError('Curated protocol or selection differs from archive')
    for section, pattern in ((fixed, 'reports/depth_study/repeat_{m}_low_seed{s}.json'),
                             (timed, 'reports/time_budget_study/{m}_seed{s}.json')):
        for record in section['training_records']:
            if record != originals[pattern.format(m=METHOD_KEYS[record['optimizer']], s=record['seed'])]:
                raise ValueError('Curated training record differs from archive')
    for section, key, path in ((fixed, 'retrieval_rows', 'reports/depth_study/retrieval/summary.json'),
                               (timed, 'terminal_rows', 'reports/time_budget_study/terminal/summary.json'),
                               (timed, 'selected_fp32_rows', 'reports/time_budget_study/retrieval/summary.json')):
        archived = originals[path]['raw_method_rows']
        expected = [{k: row[k] for k in section[key][0]} for row in archived
                    if key != 'selected_fp32_rows' or row['method'] == 'fp32']
        if section[key] != expected:
            raise ValueError('Curated retrieval rows differ from archive')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify-history', action='store_true')
    args = parser.parse_args()
    data = json.loads((ROOT / 'report/results.json').read_text())
    rows = collect_rows(data)
    if args.verify_history:
        verify_history(data)
    for path, expected in data['system']['protocol']['source_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError('Measured system source changed: ' + path)
    readme = ROOT / 'README.md'
    current = readme.read_text()
    before, rest = current.split('<!-- RESULTS:START -->\n')
    _, after = rest.split('<!-- RESULTS:END -->')
    wanted = before + '<!-- RESULTS:START -->\n' + render_markdown(rows) + '\n<!-- RESULTS:END -->' + after
    csv_path, csv_text = ROOT / 'report/results.csv', render_csv(rows)
    if args.check:
        if current != wanted or not csv_path.exists() or csv_path.read_text() != csv_text:
            parser.exit(1, 'README/CSV are stale; run python tools/build_report.py\n')
    else:
        readme.write_text(wanted)
        csv_path.write_text(csv_text)
    print(f'Validated {len(rows)} consolidated rows; README and CSV match the evidence.')


if __name__ == '__main__':
    main()
