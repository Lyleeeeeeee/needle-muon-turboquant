"""Shared utilities for the standalone CIFAR experiment; no archived-run dependencies."""
import hashlib,json,math,os,statistics as stats,subprocess,sys,sysconfig,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
THREADS=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
             NEEDLE_BACKEND='nd',NEEDLE_CPU_MATMUL='strided')
os.environ.update(THREADS)
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def distribution(values):
    return {'values': values, 'mean': stats.mean(values), 'median': stats.median(values),
            'sample_std': stats.stdev(values) if len(values) > 1 else 0.}

def first_crossing(curve, target, consecutive=1):
    if not curve or not math.isfinite(target) or not 0 <= target <= 1:
        raise ValueError('Expected observations and a precision target in [0, 1]')
    if type(consecutive) is not int or consecutive <= 0:
        raise ValueError('Consecutive count must be positive')
    for i, row in enumerate(curve):
        if (type(row['steps']) is not int or row['steps'] < 0
                or not math.isfinite(row['training_seconds']) or row['training_seconds'] < 0
                or not math.isfinite(row['precision']) or not 0 <= row['precision'] <= 1
                or (i and (row['steps'] <= curve[i - 1]['steps']
                           or row['training_seconds'] <= curve[i - 1]['training_seconds']))):
            raise ValueError('Invalid or unordered retrieval observations')
    count = 0
    for i, row in enumerate(curve):
        count = count + 1 if row['precision'] >= target else 0
        if count >= consecutive:
            first = i - consecutive + 1
            previous = curve[max(0, first - 1)]
            return dict(reached=True, steps=row['steps'], training_seconds=row['training_seconds'],
                        precision=row['precision'], first_qualifying_steps=curve[first]['steps'],
                        previous_steps=previous['steps'],
                        previous_training_seconds=previous['training_seconds'],
                        observation_index=i)
    return dict(reached=False, observed_until_steps=curve[-1]['steps'],
                observed_until_training_seconds=curve[-1]['training_seconds'])

def parameter_hash(model):
    result = hashlib.sha256()
    for p in model.parameters():
        result.update(str(p.shape).encode())
        result.update(p.numpy().tobytes())
    return result.hexdigest()

def check_frozen(protocol, data_dir=None):
    if any(digest(ROOT / name) != sha for name, sha in protocol['source_sha256'].items()):
        raise ValueError('Measured source changed after freezing the protocol')
    extension = ROOT / 'python/needle/backend_ndarray' / (
        'ndarray_backend_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    if digest(extension) != protocol['backend']['extension_sha256']:
        raise ValueError('Native extension changed after freezing the protocol')
    if data_dir is not None and any(digest(data_dir / name) != sha
                                    for name, sha in protocol['data_files_sha256'].items()):
        raise ValueError('Dataset changed after freezing the protocol')

def run(command, log):
    with log.open('w') as out:
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                                VECLIB_MAXIMUM_THREADS='1', NEEDLE_BACKEND='nd',
                                NEEDLE_CPU_MATMUL='strided'), stdout=out,
                       stderr=subprocess.STDOUT, check=True)

def freeze(path, value, resume):
    if path.exists():
        if not resume or json.loads(path.read_text()) != value:
            raise ValueError('Existing phase requires --resume with an identical frozen protocol')
    else:
        save_json(path, value)

def unit_rows(values):
    values = np.asarray(values, dtype='float32')
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite feature matrices')
    norms = np.linalg.norm(values.astype('float64'), axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)

def retrieval_quality(indices, truth, labels, k, query_count):
    selected = indices[:query_count]
    recall = np.mean([len(set(row) & set(exact)) / k
                      for row, exact in zip(selected, truth[:query_count])])
    precision = float((labels[0][selected] == labels[1][:query_count, None]).mean()) if labels else None
    return {'query_count': query_count, 'recall_at_k': float(recall), 'label_precision_at_k': precision}

def runtime_manifest():
    paths=[*ROOT.glob('python/needle/**/*.py'),*ROOT.glob('src/*.h'),
           ROOT/'src/ndarray_backend_cpu.cc',*ROOT.glob('apps/*.py'),
           *ROOT.glob('experiments/*.py'),ROOT/'config.json']
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)}

def benchmark_backend():
    extension=ROOT/'python/needle/backend_ndarray'/('ndarray_backend_cpu'+sysconfig.get_config_var('EXT_SUFFIX'))
    return dict(name='native_cpu',dtype='float32',cpu_matmul='strided',tiled_ns=True,
                extension_sha256=digest(extension))
