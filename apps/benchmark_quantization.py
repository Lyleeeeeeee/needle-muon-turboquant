"""Compare packed quantizers using identical vectors and isolated CPU workers."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

import numpy as np
import needle as ndl
from needle.quantization import TurboQuantMSE, UniformQuantizer, FloatVectorStore


def retrieval_quality(indices, truth, labels, k, query_count):
    selected = indices[:query_count]
    recall = np.mean([len(set(row) & set(exact)) / k
                      for row, exact in zip(selected, truth[:query_count])])
    precision = float((labels[0][selected] == labels[1][:query_count, None]).mean()) if labels else None
    return {'query_count': query_count, 'recall_at_k': float(recall), 'label_precision_at_k': precision}


def worker(args):
    with np.load(args.features, allow_pickle=False) as source:
        gallery, queries, truth = source['gallery'], source['queries'], source['ground_truth']
        labels = ((source['gallery_labels'], source['query_labels'])
                  if 'gallery_labels' in source and 'query_labels' in source else None)
    device = ndl.cpu() if args.device == 'cpu' else ndl.cpu_numpy()
    method = args.worker_method
    started = time.perf_counter()
    quantizer = None
    if method == 'fp32':
        setup_seconds = 0.
        store = FloatVectorStore(gallery, device=device)
    else:
        bits = int(method.rsplit('_', 1)[1])
        constructor = TurboQuantMSE if method.startswith('turbo') else UniformQuantizer
        extra = {'seed': args.seed} if method.startswith('turbo') else {}
        quantizer = constructor(gallery.shape[1], bits, device=device, **extra)
        setup_seconds = time.perf_counter() - started
        store = quantizer.encode(gallery)
    build_seconds = time.perf_counter() - started
    squared_error = 0.
    for start in range(0, len(gallery), args.block_size):
        stop = min(start + args.block_size, len(gallery))
        reconstructed = store.decode_block(start, stop).numpy()
        difference = gallery[start:stop].astype('float64') - reconstructed
        squared_error += float(np.einsum('ij,ij->', difference, difference))
    energy = float(np.einsum('ij,ij->', gallery, gallery, dtype=np.float64))
    del gallery, reconstructed, difference
    gc.collect()
    archive = args.worker_output.with_suffix('.npz')
    if method.startswith('turbo'):
        store.save(archive)
    elif method.startswith('uniform'):
        np.savez(archive, bits=np.int32(store.bits), packed=store.packed,
                 minimum=store.minimum, scale=store.scale)
    else:
        np.savez(archive, vectors=store.values.numpy())
    # Warm up allocation and dispatch; report all subsequent samples.
    options = dict(k=args.k, block_size=args.block_size, metric=args.metric)
    store.search(queries, **options)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        result = store.search(queries, **options)
        samples.append(time.perf_counter() - started)
    quality = retrieval_quality(result.indices, truth, labels, args.k, len(queries))
    prefix_quality = [retrieval_quality(result.indices, truth, labels, args.k, count)
                      for count in sorted({min(256, len(queries)), len(queries)})]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record = {'method': method, 'device': str(device), 'search_metric': args.metric,
              'normalization_timed': args.metric == 'cosine', 'stored_bytes': store.nbytes,
              'storage_breakdown': store.storage_bytes, 'serialized_bytes': archive.stat().st_size,
              'setup_seconds': setup_seconds, 'build_seconds': build_seconds,
              'query_seconds': samples, 'query_seconds_median': float(np.median(samples)),
              'relative_reconstruction_mse': squared_error / energy if energy else 0.,
              'recall_at_k': quality['recall_at_k'], 'label_precision_at_k': quality['label_precision_at_k'],
              'query_prefix_quality': prefix_quality,
              'max_decoded_block_bytes': result.max_decoded_block_bytes,
              'max_score_block_bytes': result.max_score_block_bytes,
              'process_peak_rss_bytes': int(rss if platform.system() == 'Darwin' else rss * 1024)}
    if method.startswith('turbo'):
        record['codebook_setup_seconds'] = quantizer.codebook_setup_seconds
        record['rotation_setup_seconds'] = quantizer.rotation_setup_seconds
    args.worker_output.write_text(json.dumps(record, indent=2) + '\n')


def unit_rows(values):
    values = np.asarray(values, dtype='float32')
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite feature matrices')
    norms = np.linalg.norm(values.astype('float64'), axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, help='NPZ containing gallery/queries and optional labels')
    parser.add_argument('--gallery-size', type=int, default=2048)
    parser.add_argument('--queries', type=int, default=64)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--device', choices=['cpu', 'numpy'], default='cpu')
    parser.add_argument('--metric', choices=['inner_product', 'cosine'], default='inner_product',
                        help='Raw dot products or Needle-normalized query/decoded-block cosine')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/quantization.json')
    parser.add_argument('--worker-method', choices=['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4'], help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.gallery_size, args.queries, args.dim, args.k, args.block_size, args.repeats) <= 0:
        parser.error('Counts, dimensions, k, block size and repeats must be positive')
    if args.device == 'cpu' and not ndl.cpu().enabled():
        parser.error('Build the CPU backend first: python tools/build_cpu.py')
    if args.worker_method:
        worker(args)
        return
    if args.features:
        with np.load(args.features, allow_pickle=False) as data:
            gallery, queries = unit_rows(data['gallery']), unit_rows(data['queries'])
            source_name = str(data['source']) if 'source' in data else args.features.name
            labels = {key: data[key] for key in ['gallery_labels', 'query_labels'] if key in data}
    else:
        rng = np.random.default_rng(args.seed)
        gallery = unit_rows(rng.normal(size=(args.gallery_size, args.dim)))
        queries = unit_rows(rng.normal(size=(args.queries, args.dim)))
        source_name, labels = 'synthetic_sphere', {}
    if gallery.shape[1] != queries.shape[1] or gallery.shape[1] < 2 or args.k > len(gallery):
        parser.error('Feature dimensions must agree and k must not exceed gallery size')
    if labels and (set(labels) != {'gallery_labels', 'query_labels'}
                   or labels['gallery_labels'].shape != (len(gallery),)
                   or labels['query_labels'].shape != (len(queries),)):
        parser.error('Both label arrays must match their feature matrix row counts')
    device = ndl.cpu() if args.device == 'cpu' else ndl.cpu_numpy()
    truth = FloatVectorStore(gallery, device=device).search(
        queries, k=args.k, block_size=args.block_size, metric=args.metric).indices
    rows = []
    with tempfile.TemporaryDirectory(prefix='needle-quant-benchmark-') as folder:
        folder = Path(folder)
        features = folder / 'features.npz'
        np.savez(features, gallery=gallery, queries=queries, ground_truth=truth, **labels)
        for method in ['fp32', 'uniform_2', 'uniform_4', 'turbo_2', 'turbo_4']:
            output = folder / (method + '.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--worker-method', method,
                       '--features', str(features), '--worker-output', str(output), '--device', args.device,
                       '--seed', str(args.seed), '--k', str(args.k), '--block-size', str(args.block_size),
                       '--repeats', str(args.repeats), '--metric', args.metric]
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120)
            if result.returncode:
                raise RuntimeError(method + ': ' + result.stdout + result.stderr)
            row = json.loads(output.read_text())
            row['compression_ratio_vs_fp32'] = gallery.nbytes / row['stored_bytes']
            rows.append(row)
            print(method, 'bytes=', row['stored_bytes'], 'recall=', round(row['recall_at_k'], 4), flush=True)
    report = {'source': source_name, 'gallery_shape': list(gallery.shape), 'query_shape': list(queries.shape),
              'similarity': ('inner_product_on_l2_normalized_inputs' if args.metric == 'inner_product'
                             else 'cosine_on_decoded_vectors'),
              'search_metric': args.metric, 'ground_truth_metric': args.metric, 'seed': args.seed, 'k': args.k,
              'block_size': args.block_size, 'repeats': args.repeats,
              'query_timing_scope': 'query preparation/rotation, block decode, Needle matmul and top-k; cosine includes Needle query/block normalization; excludes feature extraction and build',
              'rss_scope': 'entire isolated worker including input loading, build, quality checks, serialization and query',
              'buffer_scope': 'persistent numerical buffers; excludes Python objects and allocator overhead',
              'block_scope': 'decoded and score arrays only; excludes normalization, matmul packing/workspace and top-k merge arrays',
              'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              **{name: importlib.metadata.version(name) for name in ['numpy', 'scipy']}},
              'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    markdown = ['# Preliminary quantization comparison', '',
                f'Source: {source_name}. Gallery: {gallery.shape}; queries: {queries.shape}; seed: {args.seed}; k: {args.k}; metric: {args.metric}.', '',
                '| Method | Buffer bytes | Compression | Relative MSE | Recall@k | Median query ms |',
                '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        markdown.append(f"| {row['method']} | {row['stored_bytes']} | {row['compression_ratio_vs_fp32']:.2f}x | {row['relative_reconstruction_mse']:.6f} | {row['recall_at_k']:.4f} | {1000 * row['query_seconds_median']:.3f} |")
    markdown += ['', 'Single-seed component check, not a full paper reproduction or optimizer comparison.',
                 'Storage includes shared rotation/codebook and per-row metadata. Query timing includes normalization for cosine; excludes feature generation and build.',
                 'Process RSS is a whole-worker high-water mark, not exact query-only peak memory. Decoded-block counters exclude normalization/matmul/top-k workspace.', '']
    args.output.with_suffix('.md').write_text('\n'.join(markdown))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
