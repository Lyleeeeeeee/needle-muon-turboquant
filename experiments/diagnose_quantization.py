"""Independent quality diagnostics; not a Needle runtime or speed benchmark.

Existing quantizers and search defaults remain unchanged. NumPy float64 is an
independent scoring reference, and numerical quadrature audits the codebooks.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

import numpy as np
from scipy.integrate import quad
import needle as ndl
from needle.quantization import TurboQuantMSE, UniformQuantizer, sphere_codebook


def rank_reconstructions(queries, decoded, k, metric, batch_size=64):
    """Float64 reference top-k, with exact row-ID tie handling at the cutoff."""
    q, d = np.asarray(queries, dtype='float64'), np.asarray(decoded, dtype='float64')
    if metric not in ('inner_product', 'cosine', 'squared_l2'):
        raise ValueError('Unknown diagnostic score')
    norms2 = np.einsum('ij,ij->i', d, d)
    if metric == 'cosine':
        # Query norms are positive constants within each ranking and can be
        # omitted. A zero query produces equal scores and deterministic ties.
        d = np.divide(d, np.sqrt(norms2[:, None]), out=np.zeros_like(d), where=norms2[:, None] != 0)
    result = np.empty((len(q), k), dtype='int64')
    for start in range(0, len(q), batch_size):
        scores = q[start:start + batch_size] @ d.T
        if metric == 'squared_l2':
            scores -= .5 * norms2[None, :]
        cutoff = np.partition(scores, len(d) - k, axis=1)[:, -k]
        for row, values in enumerate(scores):
            candidates = np.flatnonzero(values >= cutoff[row])
            order = np.lexsort((candidates, -values[candidates]))[:k]
            result[start + row] = candidates[order]
    return result


@lru_cache(maxsize=8)
def quadrature_codebook(dim, bits):
    """Solve Lloyd updates independently, without Beta CDF/analytic moments."""
    power = (dim - 3) / 2
    density = lambda x: max(0., 1 - x*x) ** power
    radius = min(.9, 3 / np.sqrt(dim))
    centers = np.linspace(-radius, radius, 1 << bits)
    for _ in range(4000):
        edges = np.r_[-1., (centers[:-1] + centers[1:]) / 2, 1.]
        updated = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mass = quad(density, lo, hi, epsabs=1e-14, epsrel=1e-10)[0]
            moment = quad(lambda x: x * density(x), lo, hi, epsabs=1e-14, epsrel=1e-10)[0]
            updated.append(moment / mass)
        updated = np.array(updated)
        if np.max(abs(updated - centers)) < 1e-11:
            centers = updated
            break
        centers = updated
    else:
        raise RuntimeError('Independent Lloyd quadrature did not converge')
    edges = np.r_[-1., (centers[:-1] + centers[1:]) / 2, 1.]
    error = sum(quad(lambda x: (x-c)**2 * density(x), lo, hi, epsabs=1e-14)[0]
                for c, lo, hi in zip(centers, edges[:-1], edges[1:]))
    normalizer = quad(density, -1., 1., epsabs=1e-14)[0]
    return centers, dim * error / normalizer


def unit_rows(values):
    values = np.asarray(values, dtype='float32')
    norms = np.linalg.norm(values.astype('float64'), axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


def quality(found, truth, labels):
    recall = np.mean([len(set(a) & set(b)) / len(b) for a, b in zip(found, truth)])
    hit1 = np.mean([b[0] in a for a, b in zip(found, truth)])
    precision = float((labels[0][found] == labels[1][:, None]).mean()) if labels else None
    return {'recall_at_10': float(recall), 'hit1_at_10': float(hit1), 'label_precision_at_10': precision}


def audit_native(values):
    rows = []
    dim = values.shape[1]
    for bits in (2, 4):
        independent_book, expected = quadrature_codebook(dim, bits)
        quantizer = TurboQuantMSE(dim, bits, seed=7, device=ndl.cpu())
        book = sphere_codebook(dim, bits)
        store = quantizer.encode(values)
        rotation = quantizer.rotation.numpy().astype('float64')
        norms = np.linalg.norm(values.astype('float64'), axis=1).astype('float32')
        unit = np.divide(values, norms[:, None], out=np.zeros_like(values), where=norms[:, None] != 0)
        rotated = (ndl.Tensor(unit, device=ndl.cpu(), requires_grad=False) @ quantizer.rotation).numpy()
        # Reverse-centroid argmin gives the same upper-index tie rule without
        # reusing the production midpoint/searchsorted assignment.
        nearest = len(book) - 1 - np.abs(rotated[:, :, None] - book[::-1]).argmin(axis=2)
        columns = np.arange(dim)
        slots = 8 // bits
        unpacked = (store.packed[:, columns // slots] >> ((columns % slots) * bits)) & ((1 << bits) - 1)
        original_space = (book[unpacked].astype('float64') * norms[:, None]) @ rotation.T
        native = store.decode_block(0, len(values)).numpy().astype('float64')
        math_rotated = unit.astype('float64') @ rotation
        math_codes = np.abs(math_rotated[:, :, None] - book.astype('float64')).argmin(axis=2)
        math_decoded = (book[math_codes].astype('float64') * norms[:, None]) @ rotation.T
        record = {'bits': bits,
                  'max_centroid_difference_from_quadrature': float(abs(book - independent_book).max()),
                  'theoretical_unit_vector_mse': float(expected),
                  'rotation_orthogonality_max_error': float(abs(rotation.T @ rotation - np.eye(dim)).max()),
                  'packed_code_mismatches_vs_direct_nearest': int((nearest != unpacked).sum()),
                  'max_native_decode_error_same_codes': float(abs(native - original_space).max()),
                  'code_changes_vs_float64_rotation': int((math_codes != unpacked).sum()),
                  'native_mse': float(np.square(native-values).sum(axis=1).mean()),
                  'independent_float64_mse': float(np.square(math_decoded-values).sum(axis=1).mean())}
        rows.append(record)
        print('AUDIT', json.dumps(record), flush=True)
    return rows


def feature_stats(values):
    mean = values.astype('float64').mean(axis=0)
    return {'zero_fraction': float((values == 0).mean()),
            'negative_fraction': float((values < 0).mean()),
            'mean_row_maximum': float(values.max(axis=1).mean()),
            'mean_row_minimum': float(values.min(axis=1).mean()),
            'common_mean_squared_norm': float(mean @ mean)}


def run_case(name, gallery, queries, labels, frames):
    truth = rank_reconstructions(queries, gallery, 10, 'inner_product')
    original_sample = queries[:32].astype('float64') @ gallery[:256].astype('float64').T
    dim = gallery.shape[1]
    rng = np.random.default_rng(20261031)
    signs = rng.choice([-1., 1.], size=dim)
    basis, triangular = np.linalg.qr(rng.normal(size=(dim, dim)))
    basis *= np.where(np.diag(triangular) >= 0, 1., -1.)
    outputs = []
    for frame in frames:
        if frame == 'identity':
            x, q = gallery, queries
        elif frame == 'sign_flip':
            x, q = (gallery * signs).astype('float32'), (queries * signs).astype('float32')
        elif frame == 'orthogonal':
            x, q = (gallery @ basis).astype('float32'), (queries @ basis).astype('float32')
        else:
            raise ValueError('Unknown coordinate frame')
        sample = q[:32].astype('float64') @ x[:256].astype('float64').T
        transformed_truth = rank_reconstructions(q, x, 10, 'inner_product')
        frame_record = {'case': name, 'frame': frame, 'gallery_shape': list(x.shape),
                        'query_shape': list(q.shape), 'features': feature_stats(x),
                        'max_sample_inner_product_invariance_error': float(abs(sample-original_sample).max()),
                        'fp32_truth_overlap_after_transform': quality(transformed_truth, truth, None)['recall_at_10'],
                        'results': []}
        for bits in (2, 4):
            for method, seed in [('uniform', 7), ('turbo', 7), ('turbo', 8), ('turbo', 9)]:
                quantizer = (UniformQuantizer(dim, bits, device=ndl.cpu()) if method == 'uniform'
                             else TurboQuantMSE(dim, bits, seed=seed, device=ndl.cpu()))
                store = quantizer.encode(x)
                decoded = store.decode_block(0, len(x)).numpy().astype('float64')
                norms = np.linalg.norm(decoded, axis=1)
                zero_mask = x == 0
                row = {'method': method, 'bits': bits, 'seed': seed,
                       'relative_reconstruction_mse': float(np.square(decoded-x).sum() / np.square(x.astype('float64')).sum()),
                       'decoded_norm_mean': float(norms.mean()), 'decoded_norm_std': float(norms.std()),
                       'mean_abs_error_at_original_zeros': float(abs(decoded[zero_mask]).mean()) if zero_mask.any() else None,
                       'stored_bytes': store.nbytes, 'scores': {}}
                for metric in ('inner_product', 'cosine', 'squared_l2'):
                    found = rank_reconstructions(q, decoded, 10, metric)
                    row['scores'][metric] = quality(found, truth, labels)
                    if metric == 'inner_product' and frame == 'identity':
                        native_ids = store.search(q[:16], k=10).indices
                        row['native_search_vs_float64_reference_overlap'] = quality(found[:16], native_ids, None)['recall_at_10']
                frame_record['results'].append(row)
                print('RESULT', name, frame, method, bits, seed,
                      {m: round(v['recall_at_10'], 5) for m, v in row['scores'].items()}, flush=True)
        outputs.append(frame_record)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path,
                        default=ROOT / 'results/scale/features_train60000_gallery60000.npz')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/quantization_diagnosis.json')
    args = parser.parse_args()
    with np.load(args.features, allow_pickle=False) as source:
        gallery, queries = unit_rows(source['gallery']), unit_rows(source['queries'])
        labels = (source['gallery_labels'], source['query_labels'])
    report = {'scope': 'Independent numerical and quality diagnostics; no timing or new runtime implementation',
              'source': str(args.features.relative_to(ROOT)) if args.features.is_relative_to(ROOT) else str(args.features),
              'encoder_training_samples': 60000, 'quantizers_unchanged': True,
              'reference_arithmetic': 'NumPy float64; native CPU quantization/decode',
              'paper': 'https://arxiv.org/html/2504.19874v1',
              'audit': audit_native(gallery[:4096]), 'cases': []}
    report['cases'] += run_case('learned_small', gallery[:4096], queries[:256],
                                (labels[0][:4096], labels[1][:256]),
                                ['identity', 'sign_flip', 'orthogonal'])
    report['cases'] += run_case('learned_full', gallery, queries, labels, ['identity'])
    rng = np.random.default_rng(20261031)
    sphere_gallery = unit_rows(rng.normal(size=(4096, 64)))
    sphere_queries = unit_rows(rng.normal(size=(256, 64)))
    report['cases'] += run_case('sphere_control', sphere_gallery, sphere_queries, None, ['identity'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
