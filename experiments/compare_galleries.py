"""Isolated native gallery comparison with affine, compact and symmetric baselines."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
from common import (digest, distribution, save_json, run, THREADS, check_frozen,
                    runtime_manifest, benchmark_backend)
os.environ.update(THREADS)

METHODS = ['fp32', 'uniform_2', 'uniform_4', 'compact_2', 'compact_4',
           'symmetric_2', 'symmetric_4', 'turbo_2', 'turbo_4']


def worker(args):
    import numpy as np
    import needle as ndl
    from common import retrieval_quality
    from needle.quantization import FloatVectorStore, TurboQuantMSE, UniformQuantizer
    from uniform_compact import CompactUniformQuantizer
    with np.load(args.features, allow_pickle=False) as f:
        gallery, queries, truth = f['gallery'], f['queries'], f['ground_truth']
        labels = f['gallery_labels'], f['query_labels']
    before = time.perf_counter()
    method = args.worker
    if method == 'fp32':
        store = FloatVectorStore(gallery, device=ndl.cpu())
    else:
        bits = int(method.rsplit('_', 1)[1])
        if method.startswith('turbo'):
            quantizer = TurboQuantMSE(gallery.shape[1], bits, seed=args.rotation, device=ndl.cpu())
        elif method.startswith(('compact', 'symmetric')):
            quantizer = CompactUniformQuantizer(gallery.shape[1], bits, device=ndl.cpu(), symmetric=method.startswith('symmetric'))
        else:
            quantizer = UniformQuantizer(gallery.shape[1], bits, device=ndl.cpu())
        store = quantizer.encode(gallery)
    build_seconds = time.perf_counter() - before
    squared = 0.
    for start in range(0, len(gallery), 256):
        difference = gallery[start:start+256].astype('float64') - store.decode_block(start, min(start+256, len(gallery))).numpy()
        squared += float(np.einsum('ij,ij->', difference, difference))
    energy = float(np.einsum('ij,ij->', gallery, gallery, dtype='float64'))
    archive = args.output.with_suffix('.npz')
    if method.startswith(('turbo', 'compact', 'symmetric')):
        store.save(archive)
    elif method == 'fp32':
        np.savez(archive, vectors=store.values.numpy())
    else:
        np.savez(archive, packed=store.packed, minimum=store.minimum, scale=store.scale)
    del gallery, difference
    gc.collect()
    store.search(queries, k=args.k, block_size=256, metric='cosine')
    times = []
    for _ in range(3):
        before = time.perf_counter()
        found = store.search(queries, k=args.k, block_size=256, metric='cosine')
        times.append(time.perf_counter()-before)
    quality = retrieval_quality(found.indices, truth, labels, args.k, len(queries))
    neighbors = args.output.with_name(args.output.stem + '_neighbors.npz')
    np.savez(neighbors, indices=found.indices)
    save_json(args.output, dict(method=method, rotation_seed=args.rotation, device='cpu()',
                               search_metric='cosine', normalization_timed=True,
                               stored_bytes=store.nbytes, storage_breakdown=store.storage_bytes,
                               serialized_path=str(archive), serialized_sha256=digest(archive),
                               serialized_bytes=archive.stat().st_size, build_seconds=build_seconds,
                               relative_reconstruction_mse=squared/energy if energy else 0.,
                               neighbor_indices_sha256=hashlib.sha256(found.indices.astype('int64').tobytes()).hexdigest(),
                               neighbor_archive=str(neighbors), neighbor_archive_sha256=digest(neighbors),
                               same_category_matches=int((labels[0][found.indices] == labels[1][:, None]).sum()),
                               query_seconds=times, query_seconds_median=float(np.median(times)),
                               recall_at_k=quality['recall_at_k'], label_precision_at_k=quality['label_precision_at_k']))


def summarize_galleries(rows, seeds):
    """Average rotations within training seed before any between-seed statistics."""
    methods = [m for m in METHODS if any(r['method'] == m for r in rows)]
    by_key = {(r['encoder'], r['training_seed'], r['method'], r['rotation_seed']): r for r in rows}
    expected = {(m, s, q, r) for m in ('adamw', 'muon') for s in seeds for q in methods
                for r in ([7, 8, 9] if q.startswith('turbo') else [7])}
    if len(rows) != len(expected) or set(by_key) != expected or 'fp32' not in methods:
        raise ValueError('Incomplete unique encoder/seed/format/rotation comparison')
    summary = dict(encoders={}, paired_muon_minus_adamw={}, interaction={})
    for encoder in ('adamw', 'muon'):
        result = {}
        for method in methods:
            metrics = {k: [] for k in ('label_precision_at10', 'neighbor_recall_at10',
                                       'precision_change_vs_own_fp32', 'query_batch_seconds', 'stored_bytes')}
            for seed in seeds:
                baseline = by_key[encoder, seed, 'fp32', 7]['label_precision_at_k']
                trials = [by_key[encoder, seed, method, r] for r in ([7, 8, 9] if method.startswith('turbo') else [7])]
                precision = sum(t['label_precision_at_k'] for t in trials) / len(trials)
                if len({t['stored_bytes'] for t in trials}) != 1:
                    raise ValueError('Rotation changed persistent gallery bytes')
                values = dict(label_precision_at10=precision,
                              neighbor_recall_at10=sum(t['recall_at_k'] for t in trials)/len(trials),
                              precision_change_vs_own_fp32=precision-baseline,
                              query_batch_seconds=sum(t['query_seconds_median'] for t in trials)/len(trials),
                              stored_bytes=trials[0]['stored_bytes'])
                for key, value in values.items(): metrics[key].append(value)
            result[method] = {k: distribution(v) for k, v in metrics.items()}
        summary['encoders'][encoder] = result
    for method in methods:
        a, m = [summary['encoders'][e][method] for e in ('adamw', 'muon')]
        summary['paired_muon_minus_adamw'][method] = distribution([
            y-x for x,y in zip(a['label_precision_at10']['values'],m['label_precision_at10']['values'])])
        if method != 'fp32':
            summary['interaction'][method] = distribution([
                y-x for x,y in zip(a['precision_change_vs_own_fp32']['values'],m['precision_change_vs_own_fp32']['values'])])
    return summary


def main():
    import numpy as np
    import needle as ndl
    from common import unit_rows
    from needle.quantization import FloatVectorStore
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--worker', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--rotation', type=int, default=7, help=argparse.SUPPRESS)
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if len(set(args.methods)) != len(args.methods): parser.error('Methods must be unique')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    folder = args.output.parent / args.output.stem
    folder.mkdir(exist_ok=True)
    with np.load(args.features, allow_pickle=False) as f:
        gallery, queries = unit_rows(f['gallery']), unit_rows(f['queries'])
        labels = {k: f[k] for k in ('gallery_labels', 'query_labels')}
    if not 0 < args.k <= len(gallery): parser.error('Invalid neighbor count')
    baseline = dict(source_sha256=runtime_manifest(), backend=benchmark_backend())
    sources = dict(baseline['source_sha256'])
    for relative in ('experiments/compare_galleries.py', 'experiments/uniform_compact.py'):
        sources[relative] = digest(ROOT / relative)
    p = dict(feature_archive_sha256=digest(args.features), gallery_shape=list(gallery.shape),
             query_shape=list(queries.shape), methods=args.methods, k=args.k, block_size=256,
             repeats=3, rotations=[7,8,9], backend=baseline['backend'], source_sha256=sources,
             thread_limits=THREADS, metric='cosine', execution='sequential_isolated_method_workers',
             query_timing_scope='query preparation, metadata derivation, decode, decoded normalization, native scoring and top-k; excludes feature generation and build')
    save_json(folder / 'protocol.json', p)
    check_frozen(p)
    truth = FloatVectorStore(gallery, device=ndl.cpu()).search(queries, k=args.k, block_size=256, metric='cosine').indices
    normalized = folder / 'normalized_features.npz'
    np.savez(normalized, gallery=gallery, queries=queries, ground_truth=truth, **labels)
    rows = []
    for rotation in [7, 8, 9]:
        for method in args.methods:
            if rotation != 7 and not method.startswith('turbo'): continue
            check_frozen(p)
            path = folder / f'{method}_rotation{rotation}.json'
            run([Path(__file__), '--worker', method, '--features', normalized,
                 '--rotation', rotation, '--k', args.k, '--output', path], path.with_suffix('.log'))
            row = json.loads(path.read_text())
            row['runner_provenance'] = dict(protocol_sha256=digest(folder / 'protocol.json'),
                                            feature_archive_sha256=p['feature_archive_sha256'])
            row['compression_ratio_vs_fp32'] = gallery.nbytes / row['stored_bytes']
            rows.append(row)
            print('DONE '+method+' rotation'+str(rotation), flush=True)
    save_json(args.output, dict(protocol=p, results=rows))


if __name__ == '__main__':
    main()
