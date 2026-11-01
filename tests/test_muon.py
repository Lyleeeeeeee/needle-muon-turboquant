import numpy as np
import pytest

import needle as ndl
from needle import nn, optim


def reference_transform(update, steps=5, eps=1e-7):
    # Logical PyTorch layout is output x input, opposite Needle Linear layout.
    logical = update.astype('float64').T
    left, values, right = np.linalg.svd(logical, full_matrices=False)
    values /= np.linalg.norm(values) + eps
    for _ in range(steps):
        values = 3.4445 * values - 4.7750 * values**3 + 2.0315 * values**5
    result = (left * values) @ right
    result *= np.sqrt(max(1, logical.shape[0] / logical.shape[1]))
    return result.T


@pytest.mark.parametrize('shape', [(3, 5), (5, 3), (4, 4)])
@pytest.mark.parametrize('nesterov', [False, True])
@pytest.mark.parametrize('device_name', ['cpu', 'cpu_numpy'])
def test_muon_matches_independent_singular_value_oracle(shape, nesterov, device_name):
    rng = np.random.default_rng(19)
    initial = rng.normal(size=shape).astype('float32')
    param = nn.Parameter(initial, device=getattr(ndl, device_name)())
    optimizer = optim.Muon([param], lr=.02, momentum=.8, weight_decay=.1, nesterov=nesterov)
    expected, velocity = initial.astype('float64'), np.zeros(shape)
    for _ in range(4):
        gradient = rng.normal(size=shape).astype('float32')
        param.grad = ndl.Tensor(gradient, device=param.device, requires_grad=False)
        velocity = .8 * velocity + .2 * gradient
        update = .2 * gradient + .8 * velocity if nesterov else velocity
        expected = expected * (1 - .02 * .1) - .02 * reference_transform(update)
        optimizer.step()
        np.testing.assert_allclose(param.numpy(), expected, atol=3e-5, rtol=2e-5)
        np.testing.assert_allclose(optimizer.buffers[param].numpy(), velocity, atol=1e-6)
        assert optimizer.buffers[param].is_leaf()
    assert optimizer.state_nbytes == initial.size * 4


def test_muon_distinguishes_zero_from_missing_gradients():
    p = nn.Parameter(np.ones((3, 5), dtype='float32'), device=ndl.cpu())
    q = nn.Parameter(np.ones((5, 3), dtype='float32'), device=ndl.cpu())
    optimizer = optim.Muon([p, q], lr=.01, weight_decay=.2)
    p.grad = ndl.Tensor(np.zeros(p.shape, dtype='float32'), device=p.device, requires_grad=False)
    optimizer.step()
    np.testing.assert_allclose(p.numpy(), np.full(p.shape, .998), atol=1e-7)
    np.testing.assert_array_equal(q.numpy(), np.ones(q.shape))
    assert q not in optimizer.buffers
    assert optimizer.state_nbytes == 3 * 5 * 4
    before = p.numpy().copy()
    optimizer.reset_grad()
    q.grad = ndl.Tensor(np.ones(q.shape, dtype='float32'), device=q.device, requires_grad=False)
    optimizer.step()
    np.testing.assert_array_equal(p.numpy(), before)
    assert optimizer.state_nbytes == 2 * 3 * 5 * 4


def test_hybrid_groups_and_state_accounting():
    matrix = nn.Parameter(np.ones((2, 3), dtype='float32'), device=ndl.cpu())
    bias = nn.Parameter(np.ones(3, dtype='float32'), device=ndl.cpu())
    optimizer = optim.MuonWithAuxAdamW([matrix], [bias])
    matrix.grad = ndl.Tensor(np.ones(matrix.shape, dtype='float32'), device=matrix.device, requires_grad=False)
    bias.grad = ndl.Tensor(np.ones(bias.shape, dtype='float32'), device=bias.device, requires_grad=False)
    optimizer.step()
    assert optimizer.state_nbytes == 6 * 4 + 3 * 2 * 4
    assert optimizer.muon.params == [matrix]
    assert optimizer.adamw.params == [bias]
    optimizer.reset_grad()
    assert matrix.grad is None and bias.grad is None
    with pytest.raises(ValueError, match='disjoint'):
        optim.MuonWithAuxAdamW([matrix], [matrix, bias])


def test_muon_rejects_invalid_parameters():
    matrix = nn.Parameter(np.ones((2, 3), dtype='float32'))
    for options in ({'lr': -.1}, {'momentum': 1}, {'ns_steps': 0}, {'eps': 0}):
        with pytest.raises(ValueError):
            optim.Muon([matrix], **options)
    with pytest.raises(ValueError):
        optim.Muon([nn.Parameter(np.ones(3, dtype='float32'))])
    with pytest.raises(ValueError):
        optim.Muon([matrix, matrix])


def test_hybrid_skips_auxiliary_parameter_without_a_grad_attribute():
    matrix = nn.Parameter(np.ones((2, 3), dtype='float32'), device=ndl.cpu())
    bias = nn.Parameter(np.ones(3, dtype='float32'), device=ndl.cpu())
    optimizer = optim.MuonWithAuxAdamW([matrix], [bias])
    matrix.grad = ndl.Tensor(np.zeros(matrix.shape, dtype='float32'), device=matrix.device, requires_grad=False)
    optimizer.step()
    np.testing.assert_array_equal(bias.numpy(), np.ones(3))
    assert bias not in optimizer.adamw.m
