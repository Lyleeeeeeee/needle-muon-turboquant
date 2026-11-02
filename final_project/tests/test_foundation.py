"""Cross-backend checks for the restored homework integration."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["np", "nd"])
def backend(request):
    return request.param


def run_check(backend, code):
    env = dict(os.environ, NEEDLE_BACKEND=backend, PYTHONPATH=str(ROOT / "python"))
    result = subprocess.run(
        [sys.executable, "-c", "import numpy as np\nimport needle as ndl\nfrom needle import nn, optim\n" + textwrap.dedent(code)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_device_and_initializers(backend):
    run_check(backend, """
        x = ndl.Tensor(np.ones((2, 3), dtype='float32'))
        assert x.device == ndl.default_device()
        np.testing.assert_array_equal(ndl.init.ones(2, 3).numpy(), np.ones((2, 3)))
        np.testing.assert_array_equal(ndl.init.zeros(3).numpy(), np.zeros(3))
        w = ndl.init.kaiming_uniform(3, 4)
        assert w.shape == (3, 4)
        assert np.isfinite(w.numpy()).all()
    """)


def test_branched_matmul_gradients(backend):
    run_check(backend, """
        a0 = np.array([[.2, -.7, .5], [.9, .3, -.4]], dtype='float32')
        w0 = np.array([[.2, .8], [-.3, .4], [.7, -.1]], dtype='float32')
        a, w = ndl.Tensor(a0), ndl.Tensor(w0)
        b = ndl.Tensor(np.array([.3, -.2], dtype='float32'))
        z = a @ w + b.reshape((1, 2)).broadcast_to((2, 2))
        loss = (z * z + z).sum()
        loss.backward()
        dz = 2 * (a0 @ w0 + b.numpy()) + 1
        np.testing.assert_allclose(a.grad.numpy(), dz @ w0.T, atol=1e-5)
        np.testing.assert_allclose(w.grad.numpy(), a0.T @ dz, atol=1e-5)
        np.testing.assert_allclose(b.grad.numpy(), dz.sum(axis=0), atol=1e-5)
    """)


def test_multi_axis_reduction_and_identity_broadcast(backend):
    run_check(backend, """
        values = np.arange(24, dtype='float32').reshape(2, 3, 4)
        x = ndl.Tensor(values)
        y = x.broadcast_to(x.shape).sum(axes=(0, -1))
        np.testing.assert_allclose(y.numpy(), values.sum(axis=(0, -1)))
        y.sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones_like(values))
        transposed = x.transpose((0, 2)).reshape((4, 6))
        np.testing.assert_array_equal(transposed.numpy(), values.swapaxes(0, 2).reshape(4, 6))
    """)


def test_stable_loss_and_logsoftmax_gradients(backend):
    run_check(backend, """
        values = np.array([[1000, 1001, 999], [-1000, -999, -998]], dtype='float32')
        x = ndl.Tensor(values)
        labels = ndl.Tensor(np.array([1, 0], dtype='float32'), requires_grad=False)
        shifted = values - values.max(axis=1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        expected = -np.log(probs[np.arange(2), [1, 0]]).mean()
        loss = nn.SoftmaxLoss()(x, labels)
        np.testing.assert_allclose(loss.numpy().reshape(-1)[0], expected, atol=1e-4)
        loss.backward()
        np.testing.assert_allclose(x.grad.numpy(), (probs - np.eye(3)[[1, 0]]) / 2, atol=3e-5)
        z = ndl.Tensor(values)
        out = ndl.ops.logsoftmax(z)
        np.testing.assert_allclose(out.numpy(), np.log(probs), atol=1e-4)
        out.sum().backward()
        np.testing.assert_allclose(z.grad.numpy(), 1 - 3 * probs, atol=1e-4)
    """)


def test_normalization_and_dropout_modes(backend):
    run_check(backend, """
        values = np.array([[1, 3, -2], [4, -1, 2], [2, 5, 0]], dtype='float32')
        x = ndl.Tensor(values)
        bn, ln = nn.BatchNorm1d(3), nn.LayerNorm1d(3)
        expected_bn = (values - values.mean(axis=0)) / np.sqrt(values.var(axis=0) + bn.eps)
        np.testing.assert_allclose(bn(x).numpy(), expected_bn, atol=1e-5)
        expected_ln = (values - values.mean(axis=1, keepdims=True)) / np.sqrt(values.var(axis=1, keepdims=True) + ln.eps)
        normalized = ln(x)
        np.testing.assert_allclose(normalized.numpy(), expected_ln, atol=1e-5)
        (normalized * normalized).sum().backward()
        assert np.isfinite(x.grad.numpy()).all()
        drop = nn.Dropout(.5)
        dropped = drop(ndl.init.ones(16, 16)).numpy()
        assert set(np.unique(dropped)) == {0., 2.}
        drop.eval()
        np.testing.assert_array_equal(drop(x).numpy(), values)
        bn.eval()
        before = bn.running_mean.numpy().copy()
        bn(x)
        np.testing.assert_array_equal(bn.running_mean.numpy(), before)
    """)


def test_restored_adam_matches_numpy(backend):
    run_check(backend, """
        p = nn.Parameter(np.array([1., -2.], dtype='float32'))
        opt = optim.Adam([p], lr=.01, weight_decay=.1)
        expected, m, v = p.numpy().copy(), np.zeros(2), np.zeros(2)
        for step, gradient in enumerate(([.2, -.3], [-.1, .4]), start=1):
            grad = np.array(gradient, dtype='float32')
            p.grad = ndl.Tensor(grad, requires_grad=False)
            adjusted = grad + .1 * expected
            m, v = .9 * m + .1 * adjusted, .999 * v + .001 * adjusted ** 2
            expected -= .01 * (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
            opt.step()
            np.testing.assert_allclose(p.numpy(), expected, atol=1e-6)
        opt.reset_grad()
        assert p.grad is None
    """)


def test_mlp_can_learn_small_dataset(backend):
    run_check(backend, """
        np.random.seed(7)
        centers = np.array([[-2, -1], [2, -1], [0, 2]], dtype='float32')
        features = np.repeat(centers, 16, axis=0) + .15 * np.random.randn(48, 2).astype('float32')
        labels = np.repeat(np.arange(3), 16).astype('float32')
        x, y = ndl.Tensor(features, requires_grad=False), ndl.Tensor(labels, requires_grad=False)
        model = nn.Sequential(nn.Linear(2, 12), nn.ReLU(), nn.Linear(12, 3))
        criterion = nn.SoftmaxLoss()
        optimizer = optim.Adam(model.parameters(), lr=.03)
        first = criterion(model(x), y).numpy().reshape(-1)[0]
        for _ in range(35):
            optimizer.reset_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        logits = model(x).numpy()
        last = criterion(model(x), y).numpy().reshape(-1)[0]
        assert last < first * .2, (first, last)
        assert (logits.argmax(axis=1) == labels).mean() > .95
    """)


def test_adamw_decouples_decay_and_skips_missing_gradients(backend):
    run_check(backend, """
        p = nn.Parameter(np.array([1., -2.], dtype='float32'))
        q = nn.Parameter(np.array([.5, 1.5], dtype='float32'))
        opt = optim.AdamW([p, q], lr=.01, weight_decay=.2)
        expected = {p: p.numpy().copy(), q: q.numpy().copy()}
        states = {p: [np.zeros(2), np.zeros(2), 0], q: [np.zeros(2), np.zeros(2), 0]}
        for step in range(3):
            opt.reset_grad()
            gradients = {p: np.array([.2, -.3], dtype='float32')}
            if step != 1:
                gradients[q] = np.array([.1, .4], dtype='float32')
            for param, grad in gradients.items():
                param.grad = ndl.Tensor(grad, requires_grad=False)
                m, v, t = states[param]
                m, v, t = .9 * m + .1 * grad, .999 * v + .001 * grad ** 2, t + 1
                expected[param] = expected[param] * (1 - .01 * .2) - .01 * (m / (1 - .9**t)) / (np.sqrt(v / (1 - .999**t)) + 1e-8)
                states[param] = [m, v, t]
            opt.step()
            for param in [p, q]:
                np.testing.assert_allclose(param.numpy(), expected[param], atol=1e-6)
                assert opt.m[param].is_leaf() and opt.v[param].is_leaf()
    """)


def test_native_cpu_backend_training():
    run_check("nd", """
        device = ndl.cpu()
        assert device.enabled(), 'Build the native CPU backend first with tools/build_cpu.py'
        np.random.seed(3)
        values = np.array([[-2, -1], [2, 1], [-1, -2], [1, 2]], dtype='float32')
        x = ndl.Tensor(values, device=device, requires_grad=False)
        y = ndl.Tensor(np.array([0, 1, 0, 1], dtype='float32'), device=device, requires_grad=False)
        model = nn.Sequential(nn.Linear(2, 8, device=device), nn.ReLU(), nn.Linear(8, 2, device=device))
        optimizer = optim.Adam(model.parameters(), lr=.05)
        loss_fn = nn.SoftmaxLoss()
        first = float(loss_fn(model(x), y).numpy().reshape(-1)[0])
        for _ in range(10):
            optimizer.reset_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            assert all(p.grad.device == device for p in model.parameters())
            optimizer.step()
        assert float(loss_fn(model(x), y).numpy().reshape(-1)[0]) < first * .3
    """)
