"""Shared CPU products must preserve views, arithmetic and autodiff semantics."""
import importlib

import numpy as np
import pytest

import needle as ndl
from needle.backend_ndarray import NDArray

array_module = importlib.import_module('needle.backend_ndarray.ndarray')


def product(a, b, mode, monkeypatch):
    monkeypatch.setattr(array_module, '_CPU_MATMUL_MODE', mode, raising=False)
    return a @ b


def native_product(a, b):
    output = NDArray.make((a.shape[0], b.shape[1]), device=ndl.cpu())
    ndl.cpu().matmul_strided(a._handle, b._handle, output._handle,
                            a.shape[0], a.shape[1], b.shape[1],
                            a.strides, b.strides, a._offset, b._offset)
    return output


@pytest.mark.parametrize('layout', ['normal', 'transpose_a', 'transpose_b', 'transpose_both',
                                   'slice', 'offset', 'broadcast', 'reverse', 'reverse_both'])
def test_native_views_match_old_bits_and_numpy_and_preserve_sources(layout, monkeypatch):
    rng = np.random.default_rng(33)
    a_values = rng.normal(size=(16, 24)).astype('float32')
    b_values = rng.normal(size=(24, 32)).astype('float32')
    if layout in ('transpose_a', 'transpose_both'):
        a_values = a_values.T.copy()
    if layout in ('transpose_b', 'transpose_both'):
        b_values = b_values.T.copy()
    if layout == 'slice':
        a_values = rng.normal(size=(32, 48)).astype('float32')
        b_values = rng.normal(size=(48, 64)).astype('float32')
    if layout == 'offset':
        a_values = rng.normal(size=(18, 26)).astype('float32')
        b_values = rng.normal(size=(26, 34)).astype('float32')
    if layout == 'broadcast':
        a_values = a_values[:1].copy()
    a_base, b_base = NDArray(a_values, device=ndl.cpu()), NDArray(b_values, device=ndl.cpu())
    a, b, expected_a, expected_b = a_base, b_base, a_values, b_values
    if layout in ('transpose_a', 'transpose_both'):
        a, expected_a = a.permute((1, 0)), a_values.T
    if layout in ('transpose_b', 'transpose_both'):
        b, expected_b = b.permute((1, 0)), b_values.T
    if layout == 'slice':
        a, b, expected_a, expected_b = a[::2, ::2], b[::2, ::2], a_values[::2, ::2], b_values[::2, ::2]
    if layout == 'offset':
        a, b = a[1:17, 1:25], b[1:25, 1:33]
        expected_a, expected_b = a_values[1:17, 1:25], b_values[1:25, 1:33]
    if layout == 'broadcast':
        a, expected_a = a.broadcast_to((16, 24)), np.broadcast_to(a_values, (16, 24))
    if layout in ('reverse', 'reverse_both'):
        a = NDArray.make((16, 24), strides=(-24, -1), offset=383,
                         handle=a_base._handle, device=ndl.cpu())
        expected_a = a_values[::-1, ::-1]
    if layout == 'reverse_both':
        b = NDArray.make((24, 32), strides=(-32, 1), offset=23 * 32,
                         handle=b_base._handle, device=ndl.cpu())
        expected_b = b_values[::-1]
    # Never call numpy() on the negative-stride source view: HW3's export uses unsigned strides.
    output = native_product(a, b).numpy().copy()
    reference = product(a, b, 'reference', monkeypatch).numpy().copy()
    np.testing.assert_array_equal(output.view('uint32'), reference.view('uint32'))
    np.testing.assert_allclose(output, expected_a.astype('float64') @ expected_b.astype('float64'),
                               rtol=3e-5, atol=1e-5)
    np.testing.assert_array_equal(a_base.numpy().view('uint32'), a_values.view('uint32'))
    np.testing.assert_array_equal(b_base.numpy().view('uint32'), b_values.view('uint32'))


@pytest.mark.parametrize('bad', ['zero', 'unaligned', 'output_size', 'stride_rank',
                                'negative_offset', 'bounds_high', 'bounds_low', 'overflow', 'alias'])
def test_native_rejects_invalid_buffers_and_views(bad):
    device = ndl.cpu()
    a = NDArray(np.ones((8, 8), dtype='float32'), device=device)
    b = NDArray(np.ones((8, 8), dtype='float32'), device=device)
    output = device.empty((8, 8))
    dims, astrides, bstrides, aoffset, boffset = [8, 8, 8], (8, 1), (8, 1), 0, 0
    if bad == 'zero': dims[0] = 0
    if bad == 'unaligned': dims[1] = 7
    if bad == 'output_size': output = device.empty((8, 16))
    if bad == 'stride_rank': astrides = (1,)
    if bad == 'negative_offset': aoffset = -1
    if bad == 'bounds_high': boffset = 1
    if bad == 'bounds_low': bstrides = (-8, 1)
    if bad == 'overflow': astrides = (2**63 - 1, 1)
    if bad == 'alias': output = a
    with pytest.raises(ValueError):
        device.matmul_strided(a._handle, b._handle, output._handle, *dims,
                              astrides, bstrides, aoffset, boffset)


def test_shared_dispatch_reference_switch_and_missing_extension(monkeypatch):
    device = ndl.cpu()
    a = NDArray(np.ones((8, 16), dtype='float32'), device=device)
    b = NDArray(np.ones((16, 24), dtype='float32'), device=device)
    original = device.mod.matmul_strided
    calls = []

    def recorded(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(device.mod, 'matmul_strided', recorded)
    np.testing.assert_array_equal(product(a, b, 'strided', monkeypatch).numpy(), np.full((8, 24), 16))
    assert len(calls) == 1
    product(a, b, 'reference', monkeypatch)
    assert len(calls) == 1
    product(a, b[:, :10], 'strided', monkeypatch)
    assert len(calls) == 1
    monkeypatch.delattr(device.mod, 'matmul_strided')
    np.testing.assert_array_equal(product(a, b, 'strided', monkeypatch).numpy(), np.full((8, 24), 16))


def test_numpy_backend_keeps_its_own_path(monkeypatch):
    a = NDArray(np.ones((8, 16), dtype='float32'), device=ndl.cpu_numpy())
    b = NDArray(np.ones((16, 8), dtype='float32'), device=ndl.cpu_numpy())
    def unexpected(*args):
        pytest.fail('Native CPU strided path used by cpu_numpy')
    monkeypatch.setattr(a.device.mod, 'matmul_strided', unexpected, raising=False)
    np.testing.assert_array_equal(product(a, b, 'strided', monkeypatch).numpy(), np.full((8, 8), 16))


def test_tensor_forward_backward_match_reference_and_independent_gradients(monkeypatch):
    rng = np.random.default_rng(39)
    av = rng.normal(size=(16, 24)).astype('float32')
    bv = rng.normal(size=(24, 32)).astype('float32')
    weights = rng.normal(size=(16, 32)).astype('float32')
    results = []
    for mode in ('reference', 'strided'):
        monkeypatch.setattr(array_module, '_CPU_MATMUL_MODE', mode, raising=False)
        a, b = ndl.Tensor(av, device=ndl.cpu()), ndl.Tensor(bv, device=ndl.cpu())
        value = ((a @ b) * ndl.Tensor(weights, device=ndl.cpu(), requires_grad=False)).sum()
        value.backward()
        results.append([value.numpy().copy(), a.grad.numpy().copy(), b.grad.numpy().copy()])
    for before, after in zip(*results):
        np.testing.assert_array_equal(before.view('uint32'), after.view('uint32'))
    np.testing.assert_allclose(results[1][1], weights.astype('float64') @ bv.astype('float64').T,
                               rtol=3e-5, atol=1e-5)
    np.testing.assert_allclose(results[1][2], av.astype('float64').T @ weights.astype('float64'),
                               rtol=3e-5, atol=1e-5)
