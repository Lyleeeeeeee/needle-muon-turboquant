"""Fixed image exposure and complete paired timings for the batch diagnostic."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'tools')]


def test_training_timing_excludes_warmup_and_divides_by_real_images():
    from benchmark_batch import time_updates

    now, seen = [0.], []

    def update(start, count):
        seen.extend(range(start, start + count))
        now[0] += 3.
        return 1.

    result = time_updates(update, 4, 16, 8, clock=lambda: now[0])
    assert seen == list(range(24))
    assert result['training_seconds'] == 12.
    assert result['optimizer_seconds'] == 4.
    assert result['updates'] == 4 and result['measured_samples'] == 16
    assert result['warmup_updates'] == 2
    assert result['training_us_per_image'] == 750000.


def sample_rows(blocks=5):
    sha = lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    return [dict(block=block, optimizer=method, batch_size=batch,
                 measured_samples=32768, warmup_samples=2048,
                 updates=32768 // batch, warmup_updates=2048 // batch,
                 training_seconds=.032768 * ratio, optimizer_seconds=.008192 * ratio,
                 training_us_per_image=ratio, protocol_sha256=sha('protocol'),
                 native_extension_sha256=sha('binary'), initial_parameter_sha256=sha('initial'),
                 train_indices_sha256=sha('train'), warmup_image_ids_sha256=sha('warmup'),
                 measured_image_ids_sha256=sha('measured'),
                 final_parameter_sha256=sha((method, batch)), training_loss_sha256=sha((method, batch, 'loss')),
                 post_warm_parameter_sha256=sha((method, batch, 'warm')),
                 total_parameter_count=142410, muon_parameter_count=40960 if method == 'muon' else 0,
                 optimizer_numeric_state_bytes=1139280 if method == 'adamw' else 975440)
            for block in range(blocks) for batch in (64, 256, 512)
            for method, ratio in [('adamw', 1.), ('muon', 2.)]]


def test_batch_summary_preserves_same_image_exposure_and_paired_ratios():
    from benchmark_batch import summarize_batches

    summary = summarize_batches(sample_rows())
    for batch in ('64', '256', '512'):
        assert summary[batch]['adamw']['training_us_per_image']['mean'] == 1.
        assert summary[batch]['muon']['training_us_per_image']['mean'] == 2.
        assert summary[batch]['paired_muon_over_adamw']['mean'] == 2.
        assert summary[batch]['paired_extra_us_per_image']['mean'] == 1.
    # Batch changes the optimizer trajectory, so outputs may differ across batches.
    assert summary['64']['adamw']['updates'] == 512
    assert summary['512']['adamw']['updates'] == 64


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'images', 'initial', 'output', 'time', 'samples', 'binary'])
def test_batch_summary_rejects_unfair_or_invalid_measurements(mutation):
    from benchmark_batch import summarize_batches

    rows = copy.deepcopy(sample_rows())
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows.append(rows[0])
    if mutation == 'images': rows[-1]['measured_image_ids_sha256'] = 'other'
    if mutation == 'initial': rows[-1]['initial_parameter_sha256'] = 'other'
    if mutation == 'output': rows[-1]['final_parameter_sha256'] = 'other'
    if mutation == 'time': rows[-1]['training_seconds'] = 0.
    if mutation == 'samples': rows[-1]['measured_samples'] -= 1
    if mutation == 'binary': rows[-1]['native_extension_sha256'] = 'other'
    with pytest.raises(ValueError):
        summarize_batches(rows)


def test_native_batch_pipeline_records_all_configs_and_supports_checked_resume(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    count = 4096
    images = np.random.default_rng(4).integers(0, 256, (count, 28, 28), dtype='uint8')
    labels = np.arange(count, dtype='uint8') % 10
    (data / 'train-images-idx3-ubyte.gz').write_bytes(
        gzip.compress(struct.pack('>IIII', 2051, count, 28, 28) + images.tobytes()))
    (data / 'train-labels-idx1-ubyte.gz').write_bytes(
        gzip.compress(struct.pack('>II', 2049, count) + labels.tobytes()))
    output = tmp_path / 'run'
    command = [sys.executable, str(ROOT / 'experiments/benchmark_batch.py'), '--smoke',
               '--data-dir', str(data), '--output-dir', str(output)]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((output / 'run.json').read_text())
    assert len(report['runs']) == 12
    assert set(report['summary']) == {'64', '256', '512'}
    assert report['protocol']['smoke'] is True
    before = (output / 'block0_adamw_batch64.json').read_bytes()
    resumed = subprocess.run(command + ['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / 'block0_adamw_batch64.json').read_bytes() == before


def full_batch_fixture():
    from benchmark_batch import summarize_batches
    from build_report import json_hash
    from common import digest
    from time_to_quality import THREADS

    baseline = json.loads((ROOT / 'report/results.json').read_text())['studies']['epochs']
    before = baseline['protocol']
    protocol = dict(dataset='Fashion-MNIST', study='batch_training_cost', smoke=False,
                    blocks=5, batch_sizes=[64, 256, 512], seed=8, split_seed=17,
                    train_rows=55000, validation_rows=5000, measured_samples=32768, warmup_samples=2048,
                    architecture=[784, 128, 128, 128, 64, 10],
                    rates={'adamw_auxiliary': .001, 'muon': .007}, weight_decay=.01,
                    thread_limits=THREADS,
                    backend=copy.deepcopy(before['backend']),
                    data_files_sha256={name: sha for name, sha in before['data_files_sha256'].items()
                                       if name.startswith('train-')},
                    source_sha256={**before['source_sha256'],
                                   'experiments/benchmark_batch.py': digest(ROOT / 'experiments/benchmark_batch.py'),
                                   'experiments/time_to_quality.py': digest(ROOT / 'experiments/time_to_quality.py')})
    rows = sample_rows()
    ref = next(r for r in baseline['training_records'] if r['optimizer'] == 'AdamW' and r['seed'] == 8)
    for row in rows:
        row.update(protocol_sha256=json_hash(protocol),
                   native_extension_sha256=protocol['backend']['extension_sha256'],
                   initial_parameter_sha256=ref['initial_parameter_sha256'],
                   train_indices_sha256=ref['train_indices_sha256'])
    return dict(protocol=protocol, runs=rows, summary=summarize_batches(rows)), baseline


def test_batch_report_uses_per_image_units_and_does_not_invent_quality():
    from build_report import collect_batch_rows, render_markdown

    run, baseline = full_batch_fixture()
    rows = collect_batch_rows(run, baseline)
    assert len(rows) == 6
    assert [r['updates_mean'] for r in rows] == [512, 512, 128, 128, 64, 64]
    assert rows[0]['training_cost'] == 1. and rows[1]['training_cost'] == 2.
    assert all(r['precision_mean'] is None and r['training_unit'] == 'µs/image' for r in rows)
    assert '1.000 ± 0.000 µs/image' in render_markdown(rows)


@pytest.mark.parametrize('mutation', ['missing', 'binary', 'samples', 'initial', 'summary'])
def test_batch_report_rejects_mixed_or_incomplete_studies(mutation):
    from build_report import collect_batch_rows

    run, baseline = full_batch_fixture()
    if mutation == 'missing': run['runs'].pop()
    if mutation == 'binary': run['protocol']['backend']['extension_sha256'] = '0' * 64
    if mutation == 'samples': run['protocol']['measured_samples'] = 16384
    if mutation == 'initial':
        for row in run['runs']:
            row['initial_parameter_sha256'] = '0' * 64
    if mutation == 'summary': run['summary']['64']['adamw']['training_us_per_image']['mean'] += 1
    with pytest.raises(ValueError):
        collect_batch_rows(run, baseline)
