"""Export matched MNIST-compatible image selections from a frozen Needle encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

import numpy as np
import needle as ndl

from encoder_checkpoint import load_encoder


def retrieval_indices(train_count, query_count, reference_train, reference_queries,
                      gallery_size, query_size, seed):
    if not (0 < reference_train <= gallery_size <= train_count
            and 0 < reference_queries <= query_size <= query_count):
        raise ValueError('Selections must include the reference and fit the dataset')
    rng = np.random.default_rng(seed)
    gallery = rng.choice(train_count, reference_train, replace=False)
    queries = rng.choice(query_count, reference_queries, replace=False)
    remaining_gallery = np.setdiff1d(np.arange(train_count), gallery)
    remaining_queries = np.setdiff1d(np.arange(query_count), queries)
    extras = np.random.default_rng(seed + 1)
    # Preserve the original selections exactly; append unused IDs deterministically.
    gallery = np.r_[gallery, extras.permutation(remaining_gallery)][:gallery_size]
    queries = np.r_[queries, extras.permutation(remaining_queries)][:query_size]
    return gallery, queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=['MNIST', 'Fashion-MNIST'], default='MNIST')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--validation-size', type=int, default=0,
                        help='Exclude the fixed training holdout from the gallery pool')
    parser.add_argument('--split-seed', type=int, default=17)
    parser.add_argument('--reference-train', type=int, default=4096)
    parser.add_argument('--reference-queries', type=int, default=256)
    parser.add_argument('--gallery-size', type=int, default=60000)
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = (ROOT.parent / 'hw0/data' if args.dataset == 'MNIST'
                         else ROOT / 'data/fashion-mnist')
    if args.batch_size <= 0:
        parser.error('Batch size must be positive')
    device = ndl.cpu()
    if not device.enabled():
        parser.error('Build the native CPU backend first')
    model = load_encoder(args.checkpoint, device)
    train = ndl.data.MNISTDataset(str(args.data_dir / 'train-images-idx3-ubyte.gz'),
                                 str(args.data_dir / 'train-labels-idx1-ubyte.gz'))
    test = ndl.data.MNISTDataset(str(args.data_dir / 't10k-images-idx3-ubyte.gz'),
                                str(args.data_dir / 't10k-labels-idx1-ubyte.gz'))
    if not 0 <= args.validation_size < len(train):
        parser.error('Validation holdout must be smaller than the training set')
    pool = (np.random.default_rng(args.split_seed).permutation(len(train))[args.validation_size:]
            if args.validation_size else np.arange(len(train)))
    try:
        gallery_positions, query_ids = retrieval_indices(
            len(pool), len(test), args.reference_train, args.reference_queries,
            args.gallery_size, args.queries, args.seed)
    except ValueError as error:
        parser.error(str(error))
    gallery_ids = pool[gallery_positions]

    def extract(dataset, ids, classify=False):
        batches, correct = [], 0
        for start in range(0, len(ids), args.batch_size):
            selected = ids[start:start + args.batch_size]
            x = ndl.Tensor(dataset.X[selected], device=device, requires_grad=False)
            features = model.features(x)
            batches.append(features.numpy())
            if classify:
                scores = model.classifier(features).numpy()
                correct += int((scores.argmax(axis=1) == dataset.y[selected]).sum())
        values = np.concatenate(batches)
        if not np.isfinite(values).all():
            raise RuntimeError('Feature extraction produced non-finite values')
        return values, correct / len(ids)

    gallery, _ = extract(train, gallery_ids)
    queries, accuracy = extract(test, query_ids, classify=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    np.savez(args.output, gallery=gallery, queries=queries,
             gallery_labels=train.y[gallery_ids], query_labels=test.y[query_ids],
             gallery_ids=gallery_ids, query_ids=query_ids,
             source=np.array(args.dataset + '_frozen_encoder_' + checkpoint_hash[:12]))
    record = {'dataset': args.dataset,
              'checkpoint': str(args.checkpoint), 'checkpoint_sha256': checkpoint_hash,
              'data_files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [args.data_dir / name for name in
                                    ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                                     't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')]},
              'validation_size_excluded': args.validation_size,
              'split_seed': args.split_seed if args.validation_size else None,
              'gallery_ids_sha256': hashlib.sha256(gallery_ids.astype('int64').tobytes()).hexdigest(),
              'query_ids_sha256': hashlib.sha256(query_ids.astype('int64').tobytes()).hexdigest(),
              'device': str(device), 'selection_seed': args.seed,
              'reference_train': args.reference_train, 'reference_queries': args.reference_queries,
              'gallery_shape': list(gallery.shape), 'query_shape': list(queries.shape),
              'query_classification_accuracy': accuracy,
              'query_split': args.dataset + ' test split, exploratory evaluation',
              'feature_archive': str(args.output),
              'feature_archive_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
