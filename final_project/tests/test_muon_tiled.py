import numpy as np
import pytest

import needle as ndl
from needle import nn, optim
from needle.backend_ndarray import NDArray
from test_muon import reference_transform


def reference_core(values, steps):
    x = NDArray(values, device=ndl.cpu())
    for _ in range(steps):
        gram = x @ x.permute((1, 0))
        polynomial = gram * -4.7750 + (gram @ gram) * 2.0315
        x = x * 3.4445 + polynomial @ x
    return x.numpy().copy()


@pytest.mark.parametrize('shape', [(8, 8), (8, 16), (16, 32), (64, 128), (128, 128)])
@pytest.mark.parametrize('steps', [1, 5])
def test_native_tiled_ns_matches_reference_bits_and_preserves_input(shape, steps):
    rng = np.random.default_rng(31)
    values = rng.normal(size=shape).astype('float32')
    values /= np.linalg.norm(values.astype('float64'))
    device = ndl.cpu()
    source = NDArray(values, device=device)
    output = NDArray.make(shape, device=device)
    assert hasattr(device, 'muon_ns_tiled'), 'Native tiled NS entry point is missing'
    device.muon_ns_tiled(source._handle, output._handle, *shape, steps)
    expected = reference_core(values, steps)
    np.testing.assert_array_equal(output.numpy().view('uint32'), expected.view('uint32'))
    np.testing.assert_array_equal(source.numpy().view('uint32'), values.view('uint32'))


def test_native_tiled_ns_zero_and_invalid_dimensions():
    device = ndl.cpu()
    zero = NDArray(np.zeros((8, 16), dtype='float32'), device=device)
    output = NDArray.make((8, 16), device=device)
    assert hasattr(device, 'muon_ns_tiled'), 'Native tiled NS entry point is missing'
    device.muon_ns_tiled(zero._handle, output._handle, 8, 16, 5)
    np.testing.assert_array_equal(output.numpy(), np.zeros((8, 16), dtype='float32'))
    for rows, columns, steps in [(0, 16, 5), (7, 16, 5), (16, 8, 5), (8, 16, 0), (8, 24, 5)]:
        with pytest.raises(ValueError):
            device.muon_ns_tiled(zero._handle, output._handle, rows, columns, steps)
    wrong_output = NDArray.make((8, 8), device=device)
    with pytest.raises(ValueError):
        device.muon_ns_tiled(zero._handle, wrong_output._handle, 8, 16, 5)


@pytest.mark.parametrize('shape', [(8, 8), (8, 16), (16, 8), (16, 24)])
@pytest.mark.parametrize('nesterov', [False, True])
def test_tiled_optimizer_matches_reference_bits_and_independent_oracle(shape, nesterov):
    rng = np.random.default_rng(19)
    initial = rng.normal(size=shape).astype('float32')
    parameters = [nn.Parameter(initial.copy(), device=ndl.cpu()) for _ in range(2)]
    methods = [optim.Muon([p], lr=.02, momentum=.8, weight_decay=.1,
                          nesterov=nesterov, use_tiled_ns=packed)
               for p, packed in zip(parameters, (False, True))]
    expected, velocity = initial.astype('float64'), np.zeros(shape)
    for _ in range(3):
        gradient = rng.normal(size=shape).astype('float32')
        velocity = .8 * velocity + .2 * gradient
        update = .2 * gradient + .8 * velocity if nesterov else velocity
        expected = expected * (1 - .02 * .1) - .02 * reference_transform(update)
        for p, method in zip(parameters, methods):
            p.grad = ndl.Tensor(gradient, device=p.device, requires_grad=False)
            method.step()
        np.testing.assert_array_equal(parameters[0].numpy().view('uint32'), parameters[1].numpy().view('uint32'))
        np.testing.assert_allclose(parameters[1].numpy(), expected, atol=3e-5, rtol=2e-5)
        np.testing.assert_array_equal(methods[0].buffers[parameters[0]].numpy(),
                                      methods[1].buffers[parameters[1]].numpy())
    before = parameters[1].numpy().copy()
    methods[1].reset_grad()
    methods[1].step()
    np.testing.assert_array_equal(parameters[1].numpy(), before)


def test_tiled_dispatch_is_cpu_aligned_and_can_be_disabled(monkeypatch):
    device = ndl.cpu()
    assert hasattr(device, 'muon_ns_tiled'), 'Native tiled NS entry point is missing'
    original, calls = device.mod.muon_ns_tiled, []

    def tracked(*args):
        calls.append(args[2:])
        return original(*args)

    monkeypatch.setattr(device.mod, 'muon_ns_tiled', tracked)
    for shape, selected_device, enabled in [((8, 16), device, True), ((8, 16), device, False),
                                            ((7, 9), device, True), ((8, 16), ndl.cpu_numpy(), True)]:
        p = nn.Parameter(np.ones(shape, dtype='float32'), device=selected_device)
        optimizer = optim.Muon([p], use_tiled_ns=enabled)
        p.grad = ndl.Tensor(np.zeros(shape, dtype='float32'), device=selected_device, requires_grad=False)
        optimizer.step()
        assert np.isfinite(p.numpy()).all()
    assert calls == [(8, 16, 5)]
