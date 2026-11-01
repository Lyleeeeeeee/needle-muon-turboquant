"""Train a native Needle encoder under an explicit wall-clock budget."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

import numpy as np
import needle as ndl
from needle import nn, optim
from encoder_checkpoint import save_encoder
from models import SmallEncoder
from train_baseline import select_training_indices


def run_timed_updates(step, evaluate, save_best, budget, checkpoint_count=4,
                      clock=time.perf_counter, on_checkpoint=None):
    """Charge intermediate evaluation to budget; evaluate the final snapshot after it.

    Updates are synchronous and indivisible. A deadline crossed during an update
    finishes that update, so overrun is bounded by the last step. Checkpoint
    thresholds crossed by one unusually long step produce one candidate.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Time budget must be finite and positive')
    if type(checkpoint_count) is not int or checkpoint_count < 1:
        raise ValueError('Checkpoint count must be a positive integer')
    record = {'time_budget_seconds': budget, 'optimizer_steps': 0, 'samples_seen': 0,
              'optimizer_step_seconds': 0., 'training_step_seconds': 0.,
              'loss_sum': 0., 'checkpoints': []}
    started = clock()
    threshold = 1
    best = float('inf')

    def checkpoint(snapshot_seconds, target_seconds):
        nonlocal best
        validation_started = clock()
        loss, accuracy = evaluate()
        if not math.isfinite(loss) or not 0 <= accuracy <= 1:
            raise RuntimeError('Non-finite validation result')
        row = {'snapshot_seconds': snapshot_seconds, 'target_seconds': target_seconds,
               'optimizer_steps': record['optimizer_steps'], 'samples_seen': record['samples_seen'],
               'validation_loss': loss, 'validation_accuracy': accuracy,
               'evaluation_seconds': clock() - validation_started}
        if loss < best:
            best = loss
            save_best()
            record.update(best_validation_loss=loss, best_validation_accuracy=accuracy,
                          best_checkpoint_index=len(record['checkpoints']))
        record['checkpoints'].append(row)
        if on_checkpoint is not None:
            on_checkpoint(row)

    while clock() - started < budget:
        before = clock()
        result = step()
        after = clock()
        duration = after - before
        record['last_step_seconds'] = duration
        record['training_step_seconds'] += duration
        record['optimizer_steps'] += 1
        for key in ('samples_seen', 'loss_sum', 'optimizer_step_seconds'):
            source = {'samples_seen': 'samples', 'optimizer_step_seconds': 'optimizer_seconds'}.get(key, key)
            record[key] += result[source]
        elapsed = after - started
        if elapsed < budget and threshold < checkpoint_count and elapsed >= budget * threshold / checkpoint_count:
            checkpoint(elapsed, budget * threshold / checkpoint_count)
            threshold = int((clock() - started) * checkpoint_count / budget) + 1
    record['timed_seconds'] = clock() - started
    record['budget_overrun_seconds'] = max(0., record['timed_seconds'] - budget)
    final_started = clock()
    checkpoint(record['timed_seconds'], budget)
    record['finalization_seconds'] = clock() - final_started
    record['training_loss'] = record.pop('loss_sum') / record['samples_seen']
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], required=True)
    parser.add_argument('--budget-seconds', type=float, default=220.)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/fashion-mnist')
    parser.add_argument('--train-limit', type=int, default=55000)
    parser.add_argument('--validation-size', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--feature-dim', type=int, default=64)
    parser.add_argument('--extra-hidden-layers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--best-checkpoint-output', type=Path, required=True)
    parser.add_argument('--checkpoint-output', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
        parser.error('Time budget must be finite and positive')
    if min(args.train_limit, args.validation_size, args.batch_size,
           args.hidden_dim, args.feature_dim) <= 0 or args.extra_hidden_layers < 0:
        parser.error('Counts and dimensions must be positive; depth must be nonnegative')
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = ndl.cpu()
    dataset = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                                    str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    train_ids, val_ids, _ = select_training_indices(
        len(dataset), len(dataset), args.train_limit, args.validation_size, rng,
        args.validation_size, 17)
    x_train, y_train = dataset.X[train_ids], dataset.y[train_ids]
    x_val, y_val = dataset.X[val_ids], dataset.y[val_ids]
    model = SmallEncoder(hidden_dim=args.hidden_dim, feature_dim=args.feature_dim,
                         extra_hidden_layers=args.extra_hidden_layers, device=device)
    params = model.parameters()
    matrices = model.muon_parameters() if args.optimizer == 'muon' else []
    matrix_ids = {id(p) for p in matrices}
    optimizer = (optim.MuonWithAuxAdamW(matrices, [p for p in params if id(p) not in matrix_ids],
                                      lr=.007, aux_lr=.001, weight_decay=.01)
                 if matrices else optim.AdamW(params, lr=.001, weight_decay=.01))
    initial = hashlib.sha256()
    for param in params:
        initial.update(str(param.shape).encode())
        initial.update(param.numpy().tobytes())
    criterion = nn.SoftmaxLoss()

    def tensors(x, y):
        return (ndl.Tensor(x, device=device, requires_grad=False),
                ndl.Tensor(y, device=device, requires_grad=False))

    def evaluate():
        model.eval()
        loss, correct = 0., 0
        for start in range(0, len(x_val), args.batch_size):
            x, y = tensors(x_val[start:start + args.batch_size], y_val[start:start + args.batch_size])
            scores = model(x)
            count = min(args.batch_size, len(x_val) - start)
            loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * count
            correct += int((scores.numpy().argmax(1) == y.numpy()).sum())
        model.train()
        return loss / len(x_val), correct / len(x_val)

    initial_loss, initial_accuracy = evaluate()
    record = {'dataset': 'Fashion-MNIST', 'optimizer': args.optimizer, 'seed': args.seed,
              'backend': ndl.BACKEND, 'device': str(device), 'batch_size': args.batch_size,
              'architecture': [784, args.hidden_dim] + [args.hidden_dim] * args.extra_hidden_layers
                              + [args.feature_dim, 10],
              'extra_hidden_layers': args.extra_hidden_layers, 'learning_rate': .001,
              'muon_learning_rate': .007 if matrices else None, 'weight_decay': .01,
              'validation_size': args.validation_size, 'split_seed': 17,
              'train_samples': len(train_ids), 'eval_samples': len(val_ids),
              'initial_parameter_sha256': initial.hexdigest(),
              'initial_validation_loss': initial_loss, 'initial_validation_accuracy': initial_accuracy,
              'train_indices_sha256': hashlib.sha256(train_ids.astype('int64').tobytes()).hexdigest(),
              'validation_indices_sha256': hashlib.sha256(val_ids.astype('int64').tobytes()).hexdigest(),
              'data_files_sha256': {name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
                                    for name in ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')},
              'total_parameter_count': sum(math.prod(p.shape) for p in params),
              'muon_parameter_count': sum(math.prod(p.shape) for p in matrices),
              'muon_parameter_shapes': [list(p.shape) for p in matrices],
              'epoch_order_sha256': [], 'epochs_completed': 0,
              'clock_policy': 'wall_clock_includes_intermediate_validation_and_save;final_evaluation_excluded'}
    cursor, order = len(x_train), None

    def step():
        nonlocal cursor, order
        if cursor == len(x_train):
            order = rng.permutation(len(x_train))
            record['epoch_order_sha256'].append(hashlib.sha256(order.astype('int64').tobytes()).hexdigest())
            cursor = 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += len(indices)
        x, y = tensors(x_train[indices], y_train[indices])
        optimizer.reset_grad()
        loss = criterion(model(x), y)
        value = float(loss.numpy().reshape(-1)[0])
        if not math.isfinite(value):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        before = time.perf_counter()
        optimizer.step()
        seconds = time.perf_counter() - before
        if cursor == len(x_train):
            record['epochs_completed'] += 1
        return {'samples': len(indices), 'loss_sum': value * len(indices),
                'optimizer_seconds': seconds}

    record.update(run_timed_updates(
        step, evaluate, lambda: save_encoder(model, args.best_checkpoint_output), args.budget_seconds,
        on_checkpoint=lambda row: print(json.dumps(row), flush=True)))
    save_encoder(model, args.checkpoint_output)
    record.update(best_checkpoint_export=str(args.best_checkpoint_output),
                  checkpoint_export=str(args.checkpoint_output),
                  optimizer_numeric_state_bytes=optimizer.state_nbytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print('SAVED ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
