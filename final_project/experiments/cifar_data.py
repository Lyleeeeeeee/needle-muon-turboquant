"""Official CIFAR-100 binary input and deterministic training-only pilot split."""
from pathlib import Path
import numpy as np


def read_binary(path):
    raw = np.frombuffer(Path(path).read_bytes(), dtype='uint8')
    if not len(raw) or len(raw) % 3074:
        raise ValueError('CIFAR-100 binary rows require two labels and 3072 pixels')
    rows = raw.reshape(-1, 3074)
    if (rows[:, 0] >= 20).any() or (rows[:, 1] >= 100).any():
        raise ValueError('CIFAR-100 labels are out of range')
    return rows[:, 2:].reshape(-1, 3, 32, 32), rows[:, 1].copy()


def stratified_split(labels, train_per_class, validation_per_class, classes=100, seed=17):
    if min(train_per_class, validation_per_class, classes) <= 0:
        raise ValueError('Class and split sizes must be positive')
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in range(classes):
        ids = rng.permutation(np.flatnonzero(labels == label))
        if len(ids) < train_per_class + validation_per_class:
            raise ValueError('Insufficient examples for a balanced disjoint split')
        validation.extend(ids[:validation_per_class])
        training.extend(ids[validation_per_class:validation_per_class + train_per_class])
    return np.asarray(training, dtype='int64'), np.asarray(validation, dtype='int64')


def patches(images, mean, std, patch_size=8):
    if images.ndim != 4 or images.shape[1:] != (3, 32, 32) or 32 % patch_size:
        raise ValueError('Expected NCHW CIFAR images and a patch size dividing 32')
    if np.shape(mean) != (3,) or np.shape(std) != (3,) or np.any(np.asarray(std) <= 0):
        raise ValueError('Expected three finite channel normalization scales')
    values = images.astype('float32') / np.float32(255)
    values = (values - np.asarray(mean, dtype='float32')[None, :, None, None]) / np.asarray(std, dtype='float32')[None, :, None, None]
    side = 32 // patch_size
    return values.transpose(0, 2, 3, 1).reshape(-1, side, patch_size, side, patch_size, 3).transpose(
        0, 1, 3, 2, 4, 5).reshape(-1, side * side, patch_size * patch_size * 3).copy()
