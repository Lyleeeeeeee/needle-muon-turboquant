"""Verify the integrated Needle on local MNIST before project experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import needle as ndl
from needle import nn, optim
from models import SmallEncoder
from encoder_checkpoint import save_encoder


def select_training_indices(train_count, test_count, train_limit, eval_limit, rng,
                            validation_size=0, split_seed=17):
    if not 0 <= validation_size < train_count:
        raise ValueError('Validation holdout must be smaller than the training set')
    if validation_size:
        split = np.random.default_rng(split_seed).permutation(train_count)
        validation_pool, training_pool = split[:validation_size], split[validation_size:]
        training = rng.choice(training_pool, min(train_limit, len(training_pool)), replace=False)
        return training, validation_pool[:min(eval_limit, validation_size)], 'train_holdout'
    training = rng.choice(train_count, min(train_limit, train_count), replace=False)
    validation = rng.choice(test_count, min(eval_limit, test_count), replace=False)
    return training, validation, 'official_test_exploratory'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "hw0/data")
    parser.add_argument("--device", choices=["numpy", "cpu", "cuda"], default="numpy")
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--optimizer", choices=['adamw', 'muon'], default='adamw')
    parser.add_argument("--muon-lr", type=float, default=.02)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--validation-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline.json")
    parser.add_argument("--feature-output", type=Path,
                        help="Export gallery/query features and labels for retrieval checks")
    parser.add_argument("--checkpoint-output", type=Path,
                        help="Save frozen encoder weights for independent feature export")
    parser.add_argument("--best-checkpoint-output", type=Path,
                        help="Save the minimum-validation-loss checkpoint")
    args = parser.parse_args()
    if min(args.train_limit, args.eval_limit, args.epochs, args.batch_size,
           args.hidden_dim, args.feature_dim) <= 0:
        parser.error("Sample counts, epochs, batch size and dimensions must be positive.")
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device == "numpy":
        device = ndl.cpu_numpy() if hasattr(ndl, "cpu_numpy") else ndl.cpu()
    else:
        if not hasattr(ndl, args.device):
            parser.error("Select NEEDLE_BACKEND=nd to use native CPU or CUDA.")
        device = getattr(ndl, args.device)()
    if not device.enabled():
        parser.error("Requested backend is unavailable; run python tools/build_cpu.py for CPU.")

    train = ndl.data.MNISTDataset(str(args.data_dir / "train-images-idx3-ubyte.gz"),
                                 str(args.data_dir / "train-labels-idx1-ubyte.gz"))
    valid = (train if args.validation_size else
             ndl.data.MNISTDataset(str(args.data_dir / "t10k-images-idx3-ubyte.gz"),
                                  str(args.data_dir / "t10k-labels-idx1-ubyte.gz")))
    try:
        train_indices, valid_indices, validation_source = select_training_indices(
            len(train), len(valid), args.train_limit, args.eval_limit, rng,
            args.validation_size, args.split_seed)
    except ValueError as error:
        parser.error(str(error))
    x_train, y_train = train.X[train_indices], train.y[train_indices]
    x_valid, y_valid = valid.X[valid_indices], valid.y[valid_indices]
    model = SmallEncoder(hidden_dim=args.hidden_dim, feature_dim=args.feature_dim, device=device)
    criterion = nn.SoftmaxLoss()
    params = model.parameters()
    if args.optimizer == 'muon':
        matrices = [model.hidden.weight]
        auxiliary = [p for p in params if p is not model.hidden.weight]
        optimizer = optim.MuonWithAuxAdamW(matrices, auxiliary, lr=args.muon_lr,
                                         aux_lr=args.lr, weight_decay=args.weight_decay)
    else:
        matrices = []
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    initial_hash = hashlib.sha256()
    for param in params:
        initial_hash.update(str(param.shape).encode())
        initial_hash.update(param.numpy().tobytes())

    def tensors(x, y):
        return (ndl.Tensor(x, device=device, dtype="float32", requires_grad=False),
                ndl.Tensor(y, device=device, dtype="float32", requires_grad=False))

    def evaluate():
        model.eval()
        total_loss, correct = 0., 0
        for start in range(0, len(x_valid), args.batch_size):
            x, y = tensors(x_valid[start:start + args.batch_size], y_valid[start:start + args.batch_size])
            scores = model(x)
            total_loss += float(criterion(scores, y).numpy().reshape(-1)[0]) * len(x.numpy())
            correct += int((scores.numpy().argmax(axis=1) == y.numpy()).sum())
        return total_loss / len(x_valid), correct / len(x_valid)

    initial_loss, initial_accuracy = evaluate()
    record = {"dataset": "MNIST", "purpose": "framework integration smoke run",
              "backend": ndl.BACKEND, "device": str(device),
              "optimizer": 'Muon+AdamW' if args.optimizer == 'muon' else 'AdamW',
              "validation_source": validation_source,
              "split_seed": args.split_seed if args.validation_size else None,
              "train_validation_disjoint": bool(not np.intersect1d(train_indices, valid_indices).size) if args.validation_size else True,
              "train_indices_sha256": hashlib.sha256(train_indices.astype('int64').tobytes()).hexdigest(),
              "validation_indices_sha256": hashlib.sha256(valid_indices.astype('int64').tobytes()).hexdigest(),
              "initial_parameter_sha256": initial_hash.hexdigest(),
              "muon_parameter_shapes": [list(p.shape) for p in matrices],
              "muon_learning_rate": args.muon_lr if matrices else None,
              "weight_decay": args.weight_decay,
              "seed": args.seed, "train_samples": len(x_train), "eval_samples": len(x_valid),
              "hidden_dim": args.hidden_dim, "feature_dim": args.feature_dim,
              "learning_rate": args.lr, "batch_size": args.batch_size,
              "initial_validation_loss": initial_loss, "initial_validation_accuracy": initial_accuracy,
              "epochs": []}
    order_hash = hashlib.sha256()
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(x_train))
        order_hash.update(order.astype('int64').tobytes())
        total_loss = 0.
        optimizer_seconds = 0.
        started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x, y = tensors(x_train[indices], y_train[indices])
            optimizer.reset_grad()
            loss = criterion(model(x), y)
            total_loss += float(loss.numpy().reshape(-1)[0]) * len(indices)
            loss.backward()
            update_started = time.perf_counter()
            optimizer.step()
            optimizer_seconds += time.perf_counter() - update_started
        seconds = time.perf_counter() - started
        validation_loss, accuracy = evaluate()
        row = {"epoch": epoch + 1, "training_loss": total_loss / len(order),
               "validation_loss": validation_loss, "validation_accuracy": accuracy,
               "training_seconds": seconds, "optimizer_step_seconds": optimizer_seconds,
               "optimizer_steps": (len(order) + args.batch_size - 1) // args.batch_size}
        if not np.isfinite([row["training_loss"], validation_loss, seconds]).all():
            raise RuntimeError("Training produced non-finite values.")
        record["epochs"].append(row)
        if validation_loss < best_loss:
            best_loss = validation_loss
            record['best_validation_epoch'] = epoch + 1
            record['best_validation_loss'] = validation_loss
            record['best_validation_accuracy'] = accuracy
            if args.best_checkpoint_output is not None:
                save_encoder(model, args.best_checkpoint_output)
                record['best_checkpoint_export'] = str(args.best_checkpoint_output)
        print(json.dumps(row), flush=True)
    record['training_order_sha256'] = order_hash.hexdigest()
    record['optimizer_numeric_state_bytes'] = optimizer.state_nbytes
    if args.checkpoint_output is not None:
        save_encoder(model, args.checkpoint_output)
        record["checkpoint_export"] = str(args.checkpoint_output)
    feature_batches = []
    for start in range(0, len(x_valid), args.batch_size):
        x, _ = tensors(x_valid[start:start + args.batch_size], y_valid[start:start + args.batch_size])
        feature_batches.append(model.features(x).numpy())
    features = np.concatenate(feature_batches)
    if not np.isfinite(features).all():
        raise RuntimeError("Feature extraction produced non-finite values.")
    record["feature_shape"] = list(features.shape)
    if args.feature_output is not None:
        gallery_batches = []
        for start in range(0, len(x_train), args.batch_size):
            x, _ = tensors(x_train[start:start + args.batch_size], y_train[start:start + args.batch_size])
            gallery_batches.append(model.features(x).numpy())
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.feature_output, gallery=np.concatenate(gallery_batches), queries=features,
                 gallery_labels=y_train, query_labels=y_valid, source=np.array("MNIST_encoder"))
        record["feature_export"] = str(args.feature_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
