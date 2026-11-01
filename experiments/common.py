"""Shared provenance, pairing and aggregation helpers for report experiments."""
import hashlib
import json
import math
from pathlib import Path
import statistics as stats

SEEDS = (8, 9, 10)
PAIR_KEYS = (
    "initial_parameter_sha256", "train_indices_sha256", "validation_indices_sha256",
    "training_order_sha256", "data_files_sha256", "dataset", "seed", "train_samples",
    "eval_samples", "hidden_dim", "feature_dim", "batch_size", "validation_source",
    "initial_validation_loss", "initial_validation_accuracy",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def distribution(values):
    return {'values': values, 'mean': stats.mean(values), 'median': stats.median(values),
            'sample_std': stats.stdev(values) if len(values) > 1 else 0.}


def summarize_pairs(rows):
    pairs = {}
    for row in rows:
        key = row['block'], row['variant']
        if key in pairs or row['variant'] not in ('old', 'new'):
            raise ValueError('Duplicate or invalid pair variant')
        pairs[key] = row
    blocks = sorted({key[0] for key in pairs})
    if not blocks or any((b, v) not in pairs for b in blocks for v in ('old', 'new')):
        raise ValueError('Incomplete before/after pair')
    metrics = set(pairs[blocks[0], 'old']['metrics'])
    if any(set(r['metrics']) != metrics for r in pairs.values()):
        raise ValueError('Paired metric coverage differs')
    result = {}
    for metric in sorted(metrics):
        before, after = [], []
        for block in blocks:
            old, new = [pairs[block, v]['metrics'][metric] for v in ('old', 'new')]
            if old['output_sha256'] != new['output_sha256']:
                raise ValueError('Paired output changed: ' + metric)
            if any(not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in (old, new)):
                raise ValueError('Invalid measured time')
            before.append(old['seconds'])
            after.append(new['seconds'])
        result[metric] = {'old_seconds': distribution(before), 'new_seconds': distribution(after),
                          'speedup': distribution([a / b for a, b in zip(before, after)]),
                          'outputs_bit_identical': True}
    return result


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


def select_configuration(records):
    if not records or len({r['candidate'] for r in records}) != len(records):
        raise ValueError('Expected unique, nonempty candidate records')
    if any(not math.isfinite(r['best_validation_loss']) for r in records):
        raise ValueError('Non-finite candidate validation loss')
    return min(records, key=lambda r: (r['best_validation_loss'], r['candidate']))


def validate_pair(a, b):
    for key in PAIR_KEYS:
        if key not in a or key not in b or a[key] != b[key]:
            raise ValueError(f'Paired run mismatch: {key}')


def summarize_runs(records):
    result = {
        'validation_accuracy': distribution([r['best_validation_accuracy'] for r in records]),
        'validation_loss': distribution([r['best_validation_loss'] for r in records]),
        'training_seconds': distribution([sum(e['training_seconds'] for e in r['epochs']) for r in records]),
        'optimizer_seconds': distribution([sum(e['optimizer_step_seconds'] for e in r['epochs']) for r in records]),
        'mean_update_ms': distribution([
            1000 * sum(e['optimizer_step_seconds'] for e in r['epochs']) /
            sum(e['optimizer_steps'] for e in r['epochs']) for r in records]),
        'selected_epochs': [r['best_validation_epoch'] for r in records],
        'numeric_state_bytes': [r['optimizer_numeric_state_bytes'] for r in records],
    }
    return result


def validate_timed_pair(a, b):
    for key in ('initial_parameter_sha256', 'train_indices_sha256',
                'validation_indices_sha256', 'data_files_sha256', 'seed',
                'batch_size', 'time_budget_seconds', 'architecture'):
        if a[key] != b[key]:
            raise ValueError('Paired mismatch: ' + key)
    count = min(len(a['epoch_order_sha256']), len(b['epoch_order_sha256']))
    if count == 0 or a['epoch_order_sha256'][:count] != b['epoch_order_sha256'][:count]:
        raise ValueError('Paired epoch order prefixes differ')


def parameter_hash(parameters):
    result = hashlib.sha256()
    for p in parameters:
        result.update(p.numpy().tobytes())
    return result.hexdigest()
