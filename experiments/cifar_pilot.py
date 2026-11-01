"""Bounded CIFAR-100 feasibility study with validation-only tuning in native Needle."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'apps')]
from common import digest, save_json
from run_report import run
from time_to_quality import THREADS, check_frozen, parameter_hash
from tune_batch512 import freeze, train_updates, validation_auc
os.environ.update(THREADS)
from cifar_data import patches, read_binary, stratified_split


def rate_grid():
    return {m: [dict(id=f'{m}_{i}', lr=a, muon_lr=b if m == 'muon' else None)
                 for i, (a, b) in enumerate(zip((.0003, .001, .003), (.0021, .007, .021)))]
            for m in ('adamw', 'muon')}


def choose_candidate(records):
    eligible = [dict(candidate=r['candidate'], validation_auc=validation_auc(r['curve']),
                     terminal_precision=r['curve'][-1]['precision']) for r in records if r['status'] == 'complete']
    if not eligible:
        raise ValueError('No complete finite candidate for this optimizer')
    return min(eligible, key=lambda r: (-r['validation_auc'], -r['terminal_precision'], r['candidate']['id']))


def final_epochs(step_seconds):
    return 20 if step_seconds <= .65 else 10 if step_seconds <= 1.3 else 5


def array_hash(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def model_config(smoke=False):
    return dict(patch_dim=192, tokens=16, width=8 if smoke else 64, depth=1 if smoke else 2,
                token_hidden=8 if smoke else 32, channel_hidden=16 if smoke else 128,
                classes=3 if smoke else 100)


def optimizer_for(model, method, config):
    from needle import optim
    if method == 'adamw':
        return optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=.01)
    matrices = model.muon_parameters()
    ids = {id(p) for p in matrices}
    return optim.MuonWithAuxAdamW(matrices, [p for p in model.parameters() if id(p) not in ids],
                                 lr=config['muon_lr'], aux_lr=config['lr'], weight_decay=.01)


def probe(smoke):
    import numpy as np
    import needle as ndl
    from needle import nn
    from cifar_model import PatchMixer
    samples = {}
    for method in ('adamw', 'muon'):
        np.random.seed(7)
        model = PatchMixer(**model_config(smoke), device=ndl.cpu())
        optimizer = optimizer_for(model, method, rate_grid()[method][1])
        values = np.random.default_rng(7).normal(size=(4 if smoke else 256, 16, 192)).astype('float32')
        labels = np.arange(len(values)) % model.config['classes']
        times = []
        for i in range(6):
            before = time.perf_counter()
            optimizer.reset_grad()
            loss = nn.SoftmaxLoss()(model(ndl.Tensor(values, device=ndl.cpu(), requires_grad=False)),
                                    ndl.Tensor(labels, dtype='float32', device=ndl.cpu(), requires_grad=False))
            loss.backward()
            optimizer.step()
            if i >= 3: times.append(time.perf_counter() - before)
        samples[method] = dict(step_seconds=times, median_seconds=statistics.median(times))
    return samples


def make_protocol(args, cost):
    import numpy as np
    original = json.loads((ROOT / 'report/results.json').read_text())['integrated']['protocol']
    x, labels = read_binary(args.train_file)
    smoke = args.smoke
    if not smoke and len(x) != 50000:
        raise ValueError('The real pilot requires the official 50,000-image training file')
    train, val = stratified_split(labels, 4 if smoke else 100, 2 if smoke else 20,
                                  classes=3 if smoke else 100, seed=17)
    pixels = x[train].astype('float32') / 255
    mean = pixels.mean(axis=(0, 2, 3), dtype='float64').astype('float32')
    std = np.maximum(pixels.std(axis=(0, 2, 3), dtype='float64'), 1e-6).astype('float32')
    source = dict(original['source_sha256'])
    for relative in ('experiments/cifar_pilot.py', 'experiments/cifar_data.py', 'apps/cifar_model.py'):
        source[relative] = digest(ROOT / relative)
    return dict(dataset='CIFAR-100', study='bounded_cifar_pilot', smoke=smoke,
                model=model_config(smoke), batch_size=4 if smoke else 256,
                train_file=str(args.train_file), data_sha256=digest(args.train_file),
                official_test_used=False, training_seed=7, final_seeds=[8], split_seed=17,
                train_per_class=4 if smoke else 100, validation_per_class=2 if smoke else 20,
                query_per_class=2 if smoke else 10, train_rows=len(train), validation_rows=len(val),
                train_ids_sha256=array_hash(train), validation_ids_sha256=array_hash(val),
                mean=mean.tolist(), std=std.tolist(), augmentation='horizontal_flip_probability_0.5',
                rate_grid=rate_grid(), tuning_epochs=1 if smoke else 5,
                final_epochs=2 if smoke else final_epochs(max(r['median_seconds'] for r in cost.values())),
                selection_rule='validation_Pat10_AUC_over_updates;tie_terminal_then_id',
                weight_decay=.01, feature_dim=model_config(smoke)['width'],
                gallery_format='fp32', k=10, metric='cosine', source_sha256=source,
                backend=original['backend'], thread_limits=THREADS,
                timing_scope='complete_training_updates;excludes_setup_checkpoint_IO_offline_evaluation',
                execution='sequential_training_then_sequential_offline_quality')


def job_name(args):
    return args.candidate if args.phase == 'tuning' else args.optimizer


def data_for(p):
    import numpy as np
    images, labels = read_binary(p['train_file'])
    train, val = stratified_split(labels, p['train_per_class'], p['validation_per_class'],
                                  classes=p['model']['classes'], seed=p['split_seed'])
    if array_hash(train) != p['train_ids_sha256'] or array_hash(val) != p['validation_ids_sha256']:
        raise ValueError('Pilot image selections changed')
    return images, labels, train, val


def configuration(args, p):
    config = next(c for c in p['rate_grid'][args.optimizer] if c['id'] == args.candidate)
    if args.phase == 'final':
        selected = json.loads((args.output_dir / 'selection.json').read_text())['selected'][args.optimizer]['candidate']
        if config != selected:
            raise ValueError('Final pilot rates differ from the immutable selection')
    return config


def train_worker(args, p):
    import numpy as np
    import needle as ndl
    from needle import nn
    from cifar_model import PatchMixer, save_checkpoint
    seed = 7 if args.phase == 'tuning' else 8
    np.random.seed(seed)
    images, labels, train, val = data_for(p)
    prepared = patches(images[train], p['mean'], p['std'])
    mirrored = patches(images[train, :, :, ::-1], p['mean'], p['std'])
    model = PatchMixer(**p['model'], device=ndl.cpu())
    config = configuration(args, p)
    optimizer = optimizer_for(model, args.optimizer, config)
    initial = parameter_hash(model)
    rng = np.random.default_rng(seed)
    steps_epoch = math.ceil(len(train) / p['batch_size'])
    epochs = p['tuning_epochs'] if args.phase == 'tuning' else p['final_epochs']
    points = sorted({0, *[e * steps_epoch for e in ([1, epochs] if p['smoke'] else
                    ([1, 3, 5] if args.phase == 'tuning' else [1, 5, 10, 15, 20])) if e <= epochs]})
    order_hash, flip_hash = hashlib.sha256(), hashlib.sha256()
    updates, order, losses = [0], None, []

    def step():
        nonlocal order
        offset = updates[0] % steps_epoch
        if offset == 0:
            order = rng.permutation(len(train))
            order_hash.update(order.astype('int64').tobytes())
        ids = order[offset*p['batch_size']:(offset+1)*p['batch_size']]
        flips = rng.random(len(ids)) < .5
        flip_hash.update(flips.tobytes())
        batch = prepared[ids]
        batch[flips] = mirrored[ids[flips]]
        optimizer.reset_grad()
        scores = model(ndl.Tensor(batch, device=ndl.cpu(), requires_grad=False))
        loss = nn.SoftmaxLoss()(scores, ndl.Tensor(labels[train[ids]], device=ndl.cpu(), dtype='float32', requires_grad=False))
        value = float(loss.numpy().item())
        if not math.isfinite(value):
            raise FloatingPointError('Non-finite CIFAR training loss')
        loss.backward()
        before = time.perf_counter()
        optimizer.step()
        seconds = time.perf_counter() - before
        updates[0] += 1
        losses.append(value)
        return dict(samples=len(ids), optimizer_seconds=seconds)

    def snapshot(point):
        if any(not np.isfinite(param.numpy()).all() for param in model.parameters()):
            raise FloatingPointError('Non-finite CIFAR parameters')
        path = args.output_dir / args.phase / f'{job_name(args)}_step{point["steps"]:06d}.npz'
        save_checkpoint(model, path)
        return dict(checkpoint=str(path.relative_to(args.output_dir)), checkpoint_sha256=digest(path),
                    parameter_sha256=parameter_hash(model), training_order_sha256=order_hash.hexdigest(),
                    augmentation_sha256=flip_hash.hexdigest())

    record = train_updates(step, snapshot, points)
    record.update(optimizer=args.optimizer, seed=seed, candidate=config,
                  initial_parameter_sha256=initial, training_order_sha256=order_hash.hexdigest(),
                  augmentation_sha256=flip_hash.hexdigest(), train_ids_sha256=p['train_ids_sha256'],
                  validation_ids_sha256=p['validation_ids_sha256'], protocol_sha256=digest(args.output_dir / 'protocol.json'),
                  total_parameter_count=sum(param.numpy().size for param in model.parameters()),
                  muon_eligible_parameter_count=sum(param.numpy().size for param in model.muon_parameters()),
                  muon_parameter_count=(sum(param.numpy().size for param in model.muon_parameters())
                                        if args.optimizer == 'muon' else 0),
                  optimizer_numeric_state_bytes=optimizer.state_nbytes, training_losses=losses)
    save_json(args.output_dir / args.phase / f'{job_name(args)}_train.json', record)


def evaluate_worker(args, p):
    import gc
    import numpy as np
    import needle as ndl
    from needle import nn
    from benchmark_quantization import unit_rows
    from cifar_model import load_checkpoint
    from needle.quantization import FloatVectorStore
    path = args.output_dir / args.phase / f'{job_name(args)}_train.json'
    record = json.loads(path.read_text())
    record['training_record_sha256'] = digest(path)
    images, labels, train, val = data_for(p)
    gallery_x = patches(images[train], p['mean'], p['std'])
    valid_x = patches(images[val], p['mean'], p['std'])
    query_positions = np.concatenate([np.arange(c*p['validation_per_class'], c*p['validation_per_class']+p['query_per_class'])
                                      for c in range(p['model']['classes'])])
    curve = []
    for snap in record['snapshots']:
        checkpoint = args.output_dir / snap['checkpoint']
        if digest(checkpoint) != snap['checkpoint_sha256']:
            raise ValueError('CIFAR checkpoint changed')
        model = load_checkpoint(checkpoint, ndl.cpu())
        if parameter_hash(model) != snap['parameter_sha256']:
            raise ValueError('Restored CIFAR parameters differ')
        def extract(values, classify=False):
            features, total_loss, correct = [], 0., 0
            for start in range(0, len(values), p['batch_size']):
                stop = min(start+p['batch_size'], len(values))
                f = model.features(ndl.Tensor(values[start:stop], device=ndl.cpu(), requires_grad=False))
                features.append(f.numpy())
                if classify:
                    scores = model.classifier(f)
                    y = ndl.Tensor(labels[val[start:stop]], device=ndl.cpu(), dtype='float32', requires_grad=False)
                    total_loss += float(nn.SoftmaxLoss()(scores, y).numpy().item()) * (stop-start)
                    correct += int((scores.numpy().argmax(axis=1) == labels[val[start:stop]]).sum())
            return np.concatenate(features), total_loss / len(values), correct / len(values)
        before = time.perf_counter()
        gallery, _, _ = extract(gallery_x)
        validation, loss, accuracy = extract(valid_x, True)
        queries = validation[query_positions]
        found = FloatVectorStore(unit_rows(gallery), device=ndl.cpu()).search(
            unit_rows(queries), k=min(10, len(gallery)), block_size=256, metric='cosine')
        counts = (labels[train][found.indices] == labels[val[query_positions], None]).sum(axis=1)
        curve.append(dict(snap, precision=float(counts.sum() / counts.size / found.indices.shape[1]),
                          validation_loss=loss, validation_accuracy=accuracy,
                          same_category_matches=int(counts.sum()), neighbor_indices_sha256=array_hash(found.indices.astype('int64')),
                          evaluation_seconds=time.perf_counter()-before))
        if snap == record['snapshots'][-1]:
            archive = args.output_dir / args.phase / f'{job_name(args)}_features.npz'
            np.savez(archive, gallery=gallery, queries=queries, gallery_labels=labels[train],
                     query_labels=labels[val[query_positions]], gallery_ids=train, query_ids=val[query_positions])
            record.update(feature_archive=str(archive.relative_to(args.output_dir)), feature_archive_sha256=digest(archive),
                          gallery_ids_sha256=array_hash(train), query_ids_sha256=array_hash(val[query_positions]))
        del model
        gc.collect()
    record['curve'] = curve
    save_json(args.output_dir / args.phase / f'{job_name(args)}_curve.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--probe-only', action='store_true')
    parser.add_argument('--train-file', type=Path, default=ROOT / 'data/cifar-100/train.bin')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/cifar_pilot_20261101')
    parser.add_argument('--worker', choices=['train', 'evaluate'], help=argparse.SUPPRESS)
    parser.add_argument('--phase', choices=['tuning', 'final'], help=argparse.SUPPRESS)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], help=argparse.SUPPRESS)
    parser.add_argument('--candidate', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.train_file, args.output_dir = args.train_file.resolve(), args.output_dir.resolve()
    if args.worker:
        p = json.loads((args.output_dir / 'protocol.json').read_text())
        check_frozen(p)
        if digest(args.train_file) != p['data_sha256']:
            raise ValueError('CIFAR input changed')
        (train_worker if args.worker == 'train' else evaluate_worker)(args, p)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cost_path = args.output_dir / 'probe.json'
    if cost_path.exists():
        cost = json.loads(cost_path.read_text())
    else:
        os.environ.update(THREADS)
        cost = probe(args.smoke)
        save_json(cost_path, cost)
    if args.probe_only:
        print(json.dumps(cost), flush=True)
        return
    p = make_protocol(args, cost)
    freeze(args.output_dir / 'protocol.json', p, args.resume)
    output, selected = {}, None
    for phase in ('tuning', 'final'):
        jobs = [(m, c) for i in range(3) for m in (('adamw', 'muon') if i % 2 == 0 else ('muon', 'adamw'))
                for c in [p['rate_grid'][m][i]]] if phase == 'tuning' else [
                    (m, selected[m]['candidate']) for m in ('adamw', 'muon')]
        for worker in ('train', 'evaluate'):
            for method, config in jobs:
                name = config['id'] if phase == 'tuning' else method
                path = args.output_dir / phase / f'{name}_{"train" if worker == "train" else "curve"}.json'
                if args.resume and path.exists():
                    old = json.loads(path.read_text())
                    if old['protocol_sha256'] != digest(args.output_dir / 'protocol.json'):
                        raise ValueError('Resumed CIFAR provenance differs')
                    for snap in old['snapshots']:
                        if digest(args.output_dir / snap['checkpoint']) != snap['checkpoint_sha256']:
                            raise ValueError('Resumed CIFAR checkpoint differs')
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    run([Path(__file__), '--worker', worker, '--phase', phase, '--optimizer', method,
                         '--candidate', config['id'], '--train-file', args.train_file, '--output-dir', args.output_dir],
                        args.output_dir / phase / f'{name}_{worker}.log')
                print(f'DONE {phase} {name} {worker}', flush=True)
        output[phase] = [json.loads((args.output_dir / phase /
                         f'{config["id"] if phase == "tuning" else method}_curve.json').read_text()) for method, config in jobs]
        if phase == 'tuning':
            selected = {m: choose_candidate([r for r in output[phase] if r['optimizer'] == m]) for m in ('adamw', 'muon')}
            locked = dict(selected=selected, tuning_sha256=hashlib.sha256(json.dumps(output[phase], sort_keys=True).encode()).hexdigest())
            freeze(args.output_dir / 'selection.json', locked, args.resume)
            print('LOCKED '+json.dumps(selected), flush=True)
    save_json(args.output_dir / 'run.json', dict(protocol=p, probe=cost, selection=locked,
                                               selection_sha256=digest(args.output_dir / 'selection.json'), **output))
    print('Saved CIFAR pilot', flush=True)


if __name__ == '__main__':
    main()
