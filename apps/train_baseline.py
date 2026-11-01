"""Verify the integrated Needle on local MNIST before project experiments."""
import argparse
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
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=ROOT / "results/baseline.json")
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
    valid = ndl.data.MNISTDataset(str(args.data_dir / "t10k-images-idx3-ubyte.gz"),
                                 str(args.data_dir / "t10k-labels-idx1-ubyte.gz"))
    train_indices = rng.choice(len(train), min(args.train_limit, len(train)), replace=False)
    valid_indices = rng.choice(len(valid), min(args.eval_limit, len(valid)), replace=False)
    x_train, y_train = train.X[train_indices], train.y[train_indices]
    x_valid, y_valid = valid.X[valid_indices], valid.y[valid_indices]
    model = SmallEncoder(hidden_dim=args.hidden_dim, feature_dim=args.feature_dim, device=device)
    criterion = nn.SoftmaxLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

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
              "backend": ndl.BACKEND, "device": str(device), "optimizer": "AdamW",
              "seed": args.seed, "train_samples": len(x_train), "eval_samples": len(x_valid),
              "hidden_dim": args.hidden_dim, "feature_dim": args.feature_dim,
              "learning_rate": args.lr, "batch_size": args.batch_size,
              "initial_validation_loss": initial_loss, "initial_validation_accuracy": initial_accuracy,
              "epochs": []}
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(x_train))
        total_loss = 0.
        started = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            x, y = tensors(x_train[indices], y_train[indices])
            optimizer.reset_grad()
            loss = criterion(model(x), y)
            total_loss += float(loss.numpy().reshape(-1)[0]) * len(indices)
            loss.backward()
            optimizer.step()
        seconds = time.perf_counter() - started
        validation_loss, accuracy = evaluate()
        row = {"epoch": epoch + 1, "training_loss": total_loss / len(order),
               "validation_loss": validation_loss, "validation_accuracy": accuracy,
               "training_seconds": seconds}
        if not np.isfinite([row["training_loss"], validation_loss, seconds]).all():
            raise RuntimeError("Training produced non-finite values.")
        record["epochs"].append(row)
        print(json.dumps(row), flush=True)
    feature_batches = []
    for start in range(0, len(x_valid), args.batch_size):
        x, _ = tensors(x_valid[start:start + args.batch_size], y_valid[start:start + args.batch_size])
        feature_batches.append(model.features(x).numpy())
    features = np.concatenate(feature_batches)
    if not np.isfinite(features).all():
        raise RuntimeError("Feature extraction produced non-finite values.")
    record["feature_shape"] = list(features.shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
