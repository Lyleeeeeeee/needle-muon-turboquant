"""Posthoc FP32 check of all saved220s terminal models, with no new selection."""
import json
from pathlib import Path
import sys

from run_joint_retrieval import digest, save_json, run_logged, export_and_validate
from summarize_joint_retrieval import aggregate_quality

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/time_budget_study/terminal'
ARTIFACTS = ROOT / 'results/time_budget_study/terminal'


def main():
    protocol = json.loads((REPORTS / 'protocol.json').read_text())
    assert protocol['methods'] == ['fp32'] and protocol['rotation_seeds'] == [7]
    assert {p: digest(ROOT / p) for p in protocol['source_sha256']} == protocol['source_sha256']
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    exports = export_and_validate(protocol['models'], protocol, REPORTS, ARTIFACTS)
    rows = []
    for item in protocol['models']:
        encoder, seed = item['encoder'], item['training_seed']
        metadata = exports[encoder, seed]
        selected_features = json.loads((REPORTS.parent / 'retrieval' /
                                        f'features_{encoder}_seed{seed}.json').read_text())
        for key in ('gallery_ids_sha256', 'query_ids_sha256', 'data_files_sha256'):
            assert metadata[key] == selected_features[key]
        name = f'{encoder}_seed{seed}_rotation7'
        output = REPORTS / f'{name}.json'
        provenance = {'encoder': encoder, 'training_seed': seed, 'rotation_seed': 7,
                      'checkpoint_sha256': item['checkpoint_sha256'],
                      'feature_archive_sha256': metadata['feature_archive_sha256'],
                      'source_sha256': protocol['source_sha256']}
        if not output.exists():
            print('TERMINAL FP32 ' + name, flush=True)
            raw = ARTIFACTS / f'{name}.json'
            run_logged([sys.executable, str(ROOT / 'apps/benchmark_quantization.py'),
                        '--features', metadata['feature_archive'], '--metric', 'cosine',
                        '--device', 'cpu', '--seed', '7', '--k', '10', '--block-size', '256',
                        '--repeats', '3', '--methods', 'fp32', '--output', str(raw)],
                       ARTIFACTS / f'{name}.log')
            report = json.loads(raw.read_text())
            report['provenance'] = provenance
            save_json(output, report)
        report = json.loads(output.read_text())
        assert report['provenance'] == provenance
        assert [r['method'] for r in report['results']] == ['fp32']
        assert report['gallery_shape'] == [55000, 64] and report['query_shape'] == [1000, 64]
        assert report['search_metric'] == 'cosine' and report['k'] == 10
        row = report['results'][0]
        assert row['recall_at_k'] == 1 and row['stored_bytes'] == 14080000
        selected = json.loads((REPORTS.parent / 'retrieval' / f'{name}.json').read_text())
        if selected['provenance']['checkpoint_sha256'] == item['checkpoint_sha256']:
            assert row['label_precision_at_k'] == selected['results'][0]['label_precision_at_k']
        rows.append(dict(row, encoder=encoder, training_seed=seed, rotation_seed=7))
        print('DONE TERMINAL ' + name + f" P@10={row['label_precision_at_k']:.4f}", flush=True)
    summary = aggregate_quality(rows)
    summary.update(protocol=protocol, raw_method_rows=rows,
                   analysis_driver_sha256=digest(Path(__file__)),
                   raw_report_sha256={str(p.relative_to(ROOT)): digest(p)
                                      for p in REPORTS.glob('*rotation7.json')},
                   all_own_checkpoint_samples_and_common_ids_verified=True)
    save_json(REPORTS / 'summary.json', summary)


if __name__ == '__main__':
    main()
