from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


def test_instrumentation_counts_ns_products_and_restores_backend():
    from profile_optimizer import make_fixture, instrumented_steps
    from needle.backend_ndarray.ndarray import NDArray

    fixture = make_fixture('muon')
    original_matmul = NDArray.__matmul__
    original_compact = fixture['device'].mod.compact
    before = [p.numpy().copy() for p in fixture['parameters']]
    result = instrumented_steps(fixture, 2)
    assert result['steps'] == 2
    assert result['logical_ns_matmul_calls'] == 90
    assert result['ndarray_matmul']['calls'] == 0
    assert result['native_kernels']['muon_ns_tiled']['calls'] == 6
    assert result['packed_layout']['passes'] == 42
    assert result['native_kernels']['compact']['calls'] > 0
    assert result['phase_seconds']['muon'] > 0
    assert result['phase_seconds']['auxiliary_adamw'] > 0
    assert NDArray.__matmul__ is original_matmul
    assert fixture['device'].mod.compact is original_compact
    assert all(np.isfinite(p.numpy()).all() for p in fixture['parameters'])
    assert any(not np.array_equal(a, p.numpy()) for a, p in zip(before, fixture['parameters']))


def test_reference_profile_keeps_original_calls_for_comparison():
    from profile_optimizer import make_fixture, instrumented_steps

    result = instrumented_steps(make_fixture('muon', use_tiled_ns=False), 1)
    assert result['logical_ns_matmul_calls'] == 45
    assert result['ndarray_matmul']['calls'] == 45
    assert result['native_kernels']['compact']['calls'] == 157
    assert result['packed_layout']['passes'] == 0


def test_instrumented_and_plain_updates_produce_identical_parameters():
    from profile_optimizer import make_fixture, instrumented_steps

    plain, measured = make_fixture('muon'), make_fixture('muon')
    for _ in range(2):
        plain['optimizer'].step()
    instrumented_steps(measured, 2)
    for a, b in zip(plain['parameters'], measured['parameters']):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
