"""Aggregate rotation trials within each independently trained encoder first."""
import json
import math
import os
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/joint_retrieval'
ARTIFACTS = ROOT / 'results/joint_retrieval'
SEEDS = (8, 9, 10)


def distribution(values):
    return {'values': values, 'mean': stats.mean(values), 'sample_std': stats.stdev(values)}


def aggregate_quality(rows):
    methods = sorted({row['method'] for row in rows}, key=lambda m: (m != 'fp32', m))
    groups = {}
    for row in rows:
        groups.setdefault((row['encoder'], row['training_seed'], row['method']), []).append(row)
    result = {'encoders': {}, 'paired_muon_minus_adamw': {}, 'interaction': {}}
    for encoder in ('adamw', 'muon'):
        per_seed = {}
        for seed in SEEDS:
            exact = groups[encoder, seed, 'fp32']
            if len(exact) != 1 or exact[0]['rotation_seed'] != 7:
                raise ValueError('Expected one deterministic FP32 baseline per encoder')
            baseline = exact[0]['label_precision_at_k']
            for method in methods:
                trials = groups[encoder, seed, method]
                expected_rotations = {7, 8, 9} if method.startswith('turbo') else {7}
                if len(trials) != len(expected_rotations) or {r['rotation_seed'] for r in trials} != expected_rotations:
                    raise ValueError('Expected complete unique rotation trials')
                precision = stats.mean(r['label_precision_at_k'] for r in trials)
                if len({r['stored_bytes'] for r in trials}) != 1:
                    raise ValueError('Rotation changed persistent storage size')
                per_seed[seed, method] = {
                    'label_precision_at10': precision,
                    'neighbor_recall_at10': stats.mean(r['recall_at_k'] for r in trials),
                    'precision_change_vs_own_fp32': precision - baseline,
                    'query_batch_seconds': stats.mean(r['query_seconds_median'] for r in trials),
                    'stored_bytes': trials[0]['stored_bytes'],
                }
        result['encoders'][encoder] = {
            method: {key: distribution([per_seed[seed, method][key] for seed in SEEDS])
                     for key in ('label_precision_at10', 'neighbor_recall_at10',
                                 'precision_change_vs_own_fp32', 'query_batch_seconds', 'stored_bytes')}
            for method in methods}
    for method in methods:
        a = result['encoders']['adamw'][method]
        m = result['encoders']['muon'][method]
        result['paired_muon_minus_adamw'][method] = distribution([
            y - x for x, y in zip(a['label_precision_at10']['values'], m['label_precision_at10']['values'])])
        if method != 'fp32':
            result['interaction'][method] = distribution([
                y - x for x, y in zip(a['precision_change_vs_own_fp32']['values'],
                                      m['precision_change_vs_own_fp32']['values'])])
    return result


def main():
    from run_joint_retrieval import digest, save_json, source_hashes

    protocol_path = REPORTS / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    assert source_hashes() == protocol['source_sha256']
    metadata = {}
    common_ids = None
    for model in protocol['models']:
        encoder, seed = model['encoder'], model['training_seed']
        assert digest(ROOT / model['checkpoint']) == model['checkpoint_sha256']
        assert digest(ROOT / model['training_record']) == model['training_record_sha256']
        path = REPORTS / f'features_{encoder}_seed{seed}.json'
        record = json.loads(path.read_text())
        assert record['checkpoint_sha256'] == model['checkpoint_sha256']
        assert record['feature_archive_sha256'] == digest(record['feature_archive'])
        assert record['source_sha256'] == protocol['source_sha256']
        assert record['data_files_sha256'] == protocol['data_files_sha256']
        assert record['own_checkpoint_gallery_query_samples_reproduced']
        assert record['common_image_ids_and_labels_verified']
        ids = (record['gallery_ids_sha256'], record['query_ids_sha256'])
        if common_ids is None:
            common_ids = ids
        assert ids == common_ids
        metadata[encoder, seed] = record

    rows, raw_hashes = [], {}
    for seed in SEEDS:
        for encoder in ('adamw', 'muon'):
            for rotation in (7, 8, 9):
                path = REPORTS / f'{encoder}_seed{seed}_rotation{rotation}.json'
                raw = json.loads(path.read_text())
                provenance = raw['provenance']
                features = metadata[encoder, seed]
                assert provenance['encoder'] == encoder and provenance['training_seed'] == seed
                assert provenance['rotation_seed'] == rotation and raw['seed'] == rotation
                assert provenance['checkpoint_sha256'] == features['checkpoint_sha256']
                assert provenance['feature_archive_sha256'] == features['feature_archive_sha256']
                assert provenance['source_sha256'] == protocol['source_sha256']
                assert (provenance['gallery_ids_sha256'], provenance['query_ids_sha256']) == common_ids
                assert raw['gallery_shape'] == [55000, 64] and raw['query_shape'] == [1000, 64]
                assert raw['search_metric'] == raw['ground_truth_metric'] == 'cosine'
                assert raw['k'] == 10 and raw['block_size'] == 256 and raw['repeats'] == 3
                expected = protocol['methods'] if rotation == 7 else ['turbo_2', 'turbo_4']
                assert [r['method'] for r in raw['results']] == expected
                for row in raw['results']:
                    assert row['device'] == 'cpu()' and row['search_metric'] == 'cosine'
                    assert row['normalization_timed']
                    assert len(row['query_seconds']) == 3 and all(t > 0 for t in row['query_seconds'])
                    assert row['stored_bytes'] == sum(row['storage_breakdown'].values())
                    assert all(math.isfinite(row[k]) for k in
                               ('recall_at_k', 'label_precision_at_k', 'relative_reconstruction_mse'))
                    assert 0 <= row['recall_at_k'] <= 1 and 0 <= row['label_precision_at_k'] <= 1
                    if row['method'] == 'fp32':
                        assert row['recall_at_k'] == 1 and row['relative_reconstruction_mse'] == 0
                    rows.append(dict(row, encoder=encoder, training_seed=seed, rotation_seed=rotation))
                raw_hashes[str(path.relative_to(ROOT))] = digest(path)
    assert len(rows) == 54 and len(raw_hashes) == 18
    summary = aggregate_quality(rows)
    summary.update(date='2026-11-01', protocol=protocol, protocol_sha256=digest(protocol_path),
                   all_common_image_ids_verified=True, all_own_checkpoint_samples_reproduced=True,
                   source_sha256=protocol['source_sha256'], raw_report_sha256=raw_hashes,
                   feature_metadata=[metadata[key] for key in sorted(metadata)], raw_method_rows=rows,
                   aggregation='mean rotation trials within training seed; mean/sample SD over seeds 8/9/10',
                   timing='mean of per-rotation three-sample query medians; whole1000-query batch; excludes feature generation/build')
    for method in protocol['methods']:
        sizes = {r['stored_bytes'] for r in rows if r['method'] == method}
        assert len(sizes) == 1
    save_json(REPORTS / 'summary.json', summary)
    draw_plot(summary)
    write_report(summary)
    print('Verified six own-model archives,18 comparisons and54 method workers.', flush=True)
    for encoder in ('adamw', 'muon'):
        print(encoder, {m: round(v['label_precision_at10']['mean'], 6)
                        for m, v in summary['encoders'][encoder].items()}, flush=True)


def draw_plot(summary):
    import numpy as np
    os.environ.setdefault('MPLCONFIGDIR', str(ARTIFACTS / 'mplcache'))
    os.environ.setdefault('XDG_CACHE_HOME', str(ARTIFACTS / 'cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 10})
    methods = summary['protocol']['methods']
    labels = ['FP32', 'Uniform2', 'Uniform4', 'Turbo2', 'Turbo4']
    x = np.arange(len(methods))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = [(axes[0, 0], 'label_precision_at10', 100, 'Same-category precision@10', 'Percent'),
              (axes[0, 1], 'precision_change_vs_own_fp32', 100, 'Change from own FP32 precision', 'Percentage points'),
              (axes[1, 0], 'neighbor_recall_at10', 100, 'Recall of own exact FP32 top10', 'Percent'),
              (axes[1, 1], 'query_batch_seconds', 1, 'Whole 1,000-query search time', 'Seconds')]
    for index, encoder in enumerate(('adamw', 'muon')):
        color = '#2869a6' if encoder == 'adamw' else '#dd7924'
        name = 'AdamW encoder' if encoder == 'adamw' else 'Muon + auxiliary AdamW encoder'
        for ax, metric, scale, title, ylabel in panels:
            values = [summary['encoders'][encoder][m][metric] for m in methods]
            ax.errorbar(x + (index - .5) * .12, [scale * v['mean'] for v in values],
                        yerr=[scale * v['sample_std'] for v in values], fmt='o',
                        capsize=4, color=color, label=name)
            ax.set(title=title, ylabel=ylabel, xticks=x, xticklabels=labels)
            ax.grid(axis='y', alpha=.25)
    axes[0, 1].axhline(0, color='gray', linewidth=1)
    axes[1, 0].set_ylim(0, 105)
    axes[1, 1].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.suptitle('Fashion-MNIST: paired training seeds 8/9/10; rotation means; error bars = sample SD')
    fig.savefig(REPORTS / 'comparison.svg', metadata={'Date': '2026-11-01'})
    fig.savefig(ARTIFACTS / 'comparison.png', dpi=150)
    plt.close(fig)


def write_report(summary):
    def percent(value):
        return f"{100*value['mean']:.2f}% +/- {100*value['sample_std']:.2f}"

    a, m = summary['encoders']['adamw'], summary['encoders']['muon']
    methods = summary['protocol']['methods']
    size = a['turbo_2']['stored_bytes']['mean']
    ratio = a['fp32']['stored_bytes']['mean'] / size
    finding = (f"TurboQuant2-bit category precision is {100*a['turbo_2']['label_precision_at10']['mean']:.2f}% "
               f"with AdamW and {100*m['turbo_2']['label_precision_at10']['mean']:.2f}% with Muon. "
               f"Its gallery store uses {size/1e6:.4f}MB ({ratio:.2f}x compression). "
               'Muon has a smaller mean Turbo compression-induced category-precision loss, '
               'but its lower uncompressed retrieval baseline means the combined method does '
               'not exceed the AdamW counterpart in mean absolute category precision. '
               'Three paired training seeds and these CPU timings do not establish useful general synergy.')
    lines = ['# Joint training-to-embedding-to-quantization retrieval study', '',
             f"Uncompressed same-category precision@10 is {100*a['fp32']['label_precision_at10']['mean']:.2f}% "
             f"for the tuned AdamW encoders and {100*m['fp32']['label_precision_at10']['mean']:.2f}% "
             'for the tuned Muon plus auxiliary AdamW encoders. This measures retrieval, not the '
             'classification accuracy from the earlier study. Every quantizer below now consumes '
             'embeddings from the corresponding tuned model.', '', finding, '',
             '## Connected pipeline and fixed inputs', '',
             'Fashion images -> fixed trained encoder -> 64D embeddings -> FP32/uniform/TurboQuant-MSE '
             'store -> native cosine top10 search. Each encoder supplies both its own gallery and '
             'query vectors; feature spaces from different models are never mixed. Only gallery '
             'vectors are compressed; queries remain float32. Store bytes exclude model weights, '
             'original images, evaluation labels and query vectors.', '',
             'Use all three paired training seeds 8/9/10 from the '
             '[learning-rate study](../lr_dataset_study/summary.md). No retraining or best-seed '
             'selection. AdamW/auxiliary rate 0.001, Muon matrix rate 0.007; each checkpoint was '
             'previously selected by minimum validation loss. The gallery contains the same 55,000 '
             'training images in every archive, excluding the fixed 5,000-image split 17 validation '
             'holdout. Queries are the same 1,000 official-test image IDs. All use 64 features, '
             'cosine scoring, top10, blocks of 256 and native float32 CPU.', '',
             'The [predeclared design](../../docs/joint_retrieval_design.md), '
             '[frozen protocol and model hashes](protocol.json), six feature metadata records and '
             '18 raw comparison records make this chain auditable. Each feature archive was '
             'verified against its own checkpoint and common image IDs/labels before timing.', '',
             '## Primary task quality: same-category precision@10', '',
             'For each query, retrieve ten images and count the fraction sharing its clothing '
             'category; average over 1,000 queries. This is category-level retrieval, not an '
             'instance-level relevance judgment. Values are mean +/- sample SD over three training '
             'seeds after first averaging Turbo rotations 7/8/9 within each seed.', '',
             '| Store | AdamW encoder | Muon encoder | Paired Muon minus AdamW (pp) |',
             '|---|---:|---:|---:|']
    for method in methods:
        diff = summary['paired_muon_minus_adamw'][method]
        lines.append(f"| {method} | {percent(a[method]['label_precision_at10'])} | "
                     f"{percent(m[method]['label_precision_at10'])} | {100*diff['mean']:+.3f} +/- {100*diff['sample_std']:.3f} |")
    lines += ['', 'FP32 is a separate baseline for each trained model. Higher absolute precision '
              'means better category retrieval in this evaluation; it does not establish which '
              'optimizer is generally better. Both encoders were trained with a classification '
              'objective, not a dedicated metric-learning objective.', '',
              '![Quality, compression change, exact-neighbor retention and query time](comparison.svg)', '',
              '## Compression-induced task change and interaction', '',
              'Changes below are compressed precision minus that encoder\'s own FP32 precision. '
              'Negative means a drop; positive means compression happened to improve category '
              'precision. Do not clamp gains to zero. Interaction is Muon\'s change minus AdamW\'s '
              'change: positive means Muon loses less or gains more on this metric. A favorable '
              'interaction can coexist with worse absolute precision, so read both tables.', '',
              '| Store | AdamW change (pp) | Muon change (pp) | Paired interaction (pp) |',
              '|---|---:|---:|---:|']
    for method in methods[1:]:
        change_a, change_m = a[method]['precision_change_vs_own_fp32'], m[method]['precision_change_vs_own_fp32']
        effect = summary['interaction'][method]
        lines.append(f"| {method} | {100*change_a['mean']:+.3f} | {100*change_m['mean']:+.3f} | "
                     f"{100*effect['mean']:+.3f} +/- {100*effect['sample_std']:.3f} |")
    lines += ['', '## Exact-neighbor fidelity (different from category precision)', '',
              'Recall@10 measures overlap with that same model\'s own uncompressed cosine top10 '
              'image IDs. It measures how faithfully compression preserves the original search. '
              'The two encoders have different exact neighbors; they are not compared against a '
              'shared encoder\'s neighbor list.', '',
              '| Store | AdamW own-FP32 recall@10 | Muon own-FP32 recall@10 |', '|---|---:|---:|']
    for method in methods:
        lines.append(f"| {method} | {percent(a[method]['neighbor_recall_at10'])} | {percent(m[method]['neighbor_recall_at10'])} |")
    lines += ['', '## Storage and query cost', '',
              '| Store | Persistent MB (both encoders) | Compression vs FP32 | AdamW query batch seconds | Muon query batch seconds |',
              '|---|---:|---:|---:|---:|']
    base_bytes = a['fp32']['stored_bytes']['mean']
    for method in methods:
        size = a[method]['stored_bytes']['mean']
        ta, tm = a[method]['query_batch_seconds'], m[method]['query_batch_seconds']
        lines.append(f"| {method} | {size/1e6:.4f} | {base_bytes/size:.3f}x | "
                     f"{ta['mean']:.3f} +/- {ta['sample_std']:.3f} | {tm['mean']:.3f} +/- {tm['sample_std']:.3f} |")
    lines += ['', 'MB uses 1,000,000 bytes. Stores include packed codes and all persistent numerical '
              'metadata, including Turbo rotation/codebooks and per-row norms. Equal bit width '
              'does not imply equal total bytes. Storage is identical across encoders at fixed '
              'shape/format. This is not peak process memory.', '',
              'Query times cover the whole 1,000-query batch, including input normalization, '
              'rotation, block decode, decoded-vector normalization, native matrix products and '
              'top-k merging. They exclude model feature generation and index construction. '
              'Each worker warms up once and retains three timings. Aggregate each rotation\'s '
              'median within training seed, then average the three training seeds. All retained '
              'workers run sequentially. Per-worker construction time, reconstruction MSE, storage '
              'breakdown and whole-worker peak RSS remain in the raw JSON; RSS is not isolated '
              'query-only memory.', '',
              '## Feature geometry and limits', '',
              '| Encoder | Mean feature zero fraction | Query classification accuracy (diagnostic only) |',
              '|---|---:|---:|']
    for encoder in ('adamw', 'muon'):
        records = [r for r in summary['feature_metadata'] if r['encoder'] == encoder]
        zeros = stats.mean(r['feature_geometry']['zero_fraction'] for r in records)
        accuracy = stats.mean(r['query_classification_accuracy'] for r in records)
        lines.append(f'| {encoder} | {100*zeros:.2f}% | {100*accuracy:.2f}% |')
    lines += ['', 'Sparsity differences are observations, not a causal explanation for compression '
              'behavior. The native quantizer/scorer formulas are unchanged. TurboQuant is the '
              'MSE variant with 2/4-bit coordinate codes, not QJL. Rotation trials are not nine '
              'independent trained models. Error bars/sample SD describe only three training seeds, '
              'not confidence intervals. Previous classification test results already exposed '
              'Fashion test data; this is a fixed-model retrieval extension, not an untouched '
              'final test. No retrieval result selects a checkpoint, dataset or configuration.', '',
              'Training cost remains separate: the tuned Muon encoders took longer per ten epochs '
              'on this CPU in the earlier study. A compression-quality advantage alone would not '
              'establish an end-to-end training or query speedup. Old MNIST quantization records '
              'remain historical comparisons on their original encoder.', '',
              '## Reproduction and validation', '',
              'All six original checkpoint hashes and dataset hashes match. Common gallery/query '
              'IDs and labels match across all models, and the validation holdout is excluded. '
              'First 64 gallery and query embeddings for each model reproduce exactly from its '
              'own checkpoint. All 18 raw records/54 method workers pass provenance, shape, metric, '
              'storage and completeness checks. FP32 recall is 100% by construction for each '
              'model. The old default MNIST export reproduces every saved array/source exactly.', '',
              '```bash', 'python -m pytest -q', 'python experiments/run_joint_retrieval.py',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/summarize_joint_retrieval.py',
              '```', '', 'Data, checkpoints and feature NPZ files remain ignored. Full reproduction '
              'requires regenerating the local models/records following the learning-rate study. '
              'Archive the published joint report directory outside this output path before '
              'regenerating it; existing records resume only with matching sources and artifacts. '
              '[Full summary and per-worker rows](summary.json) contain all values and hashes. '
              'Matplotlib is optional analysis tooling, not a Needle runtime dependency.', '']
    (REPORTS / 'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
