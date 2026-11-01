"""Optimization module"""
import needle as ndl
import numpy as np
import math


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            if param.grad is None:
                continue

            grad = (
                param.grad.data
                + self.weight_decay * param.data
            )

            velocity = self.u.get(param, 0)
            velocity = (
                self.momentum * velocity
                + (1 - self.momentum) * grad
            )
            self.u[param] = velocity

            param.data = (
                param.data - self.lr * velocity
            )
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for param in self.params:
            if param.grad is None:
                continue

            grad = (
                param.grad.data
                + self.weight_decay * param.data
            )

            first_moment = self.m.get(param, 0)
            second_moment = self.v.get(param, 0)

            first_moment = (
                self.beta1 * first_moment
                + (1 - self.beta1) * grad
            )
            second_moment = (
                self.beta2 * second_moment
                + (1 - self.beta2) * grad ** 2
            )

            self.m[param] = first_moment
            self.v[param] = second_moment

            first_unbiased = first_moment / (
                1 - self.beta1 ** self.t
            )
            second_unbiased = second_moment / (
                1 - self.beta2 ** self.t
            )

            param.data = param.data - (
                self.lr
                * first_unbiased
                / (second_unbiased ** 0.5 + self.eps)
            )
        ### END YOUR SOLUTION


class AdamW(Adam):
    """Adam with decoupled weight decay; the homework Adam remains unchanged."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, beta1, beta2, eps, weight_decay)
        self.steps = {}

    @property
    def state_nbytes(self):
        return sum(math.prod(value.shape) * np.dtype(value.dtype).itemsize
                   for state in (self.m, self.v) for value in state.values())

    def step(self):
        for param in self.params:
            if getattr(param, 'grad', None) is None:
                continue
            grad = param.grad.data
            t = self.steps.get(param, 0) + 1
            self.steps[param] = t
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad ** 2
            m_hat = self.m[param] / (1 - self.beta1 ** t)
            v_hat = self.v[param] / (1 - self.beta2 ** t)
            param.data = param.data * (1 - self.lr * self.weight_decay) - self.lr * m_hat / (v_hat ** 0.5 + self.eps)


class Muon(Optimizer):
    """Float32 Muon for hidden Needle Linear weights stored as (input, output).

    Formula reference: KellerJordan/Muon at f98f1cacc0263b04290753e32be8d498c1efc806.
    Unlike the author's bfloat16 implementation, this uses Needle float32.
    Missing gradients are skipped. Numeric state excludes transient NS matrices.
    """
    def __init__(self, params, lr=.02, momentum=.95, weight_decay=0.,
                 ns_steps=5, nesterov=True, eps=1e-7):
        params = list(params)
        if (not params or len({id(p) for p in params}) != len(params)
                or any(len(p.shape) != 2 or min(p.shape) <= 0 for p in params)):
            raise ValueError('Muon requires unique nonempty 2D Linear matrices')
        if (not all(math.isfinite(v) for v in (lr, momentum, weight_decay, eps))
                or lr < 0 or not 0 <= momentum < 1 or weight_decay < 0 or eps <= 0
                or not isinstance(ns_steps, int) or ns_steps <= 0):
            raise ValueError('Invalid Muon hyperparameters')
        super().__init__(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self.ns_steps, self.nesterov, self.eps = ns_steps, nesterov, eps
        self.buffers = {}

    @property
    def state_nbytes(self):
        return sum(math.prod(v.shape) * np.dtype(v.dtype).itemsize for v in self.buffers.values())

    def _transform(self, gradient):
        rows, columns = gradient.shape
        x = gradient.transpose() if rows > columns else gradient
        denominator = ((x * x).sum() ** .5 + self.eps).reshape((1, 1)).broadcast_to(x.shape)
        x = x / denominator
        for _ in range(self.ns_steps):
            gram = x @ x.transpose()
            polynomial = gram * -4.7750 + (gram @ gram) * 2.0315
            x = x * 3.4445 + polynomial @ x
        if rows > columns:
            x = x.transpose()
        # Author scaling is output/input; Needle Linear stores input/output.
        return (x * math.sqrt(max(1., columns / rows))).detach()

    def step(self):
        for param in self.params:
            gradient = getattr(param, 'grad', None)
            if gradient is None:
                continue
            if gradient.shape != param.shape:
                raise ValueError('Muon gradient shape must match its parameter')
            gradient = gradient.data
            velocity = (self.momentum * self.buffers.get(param, 0)
                        + (1 - self.momentum) * gradient).detach()
            self.buffers[param] = velocity
            update = ((1 - self.momentum) * gradient + self.momentum * velocity
                      if self.nesterov else velocity)
            transformed = self._transform(update)
            param.data = param.data * (1 - self.lr * self.weight_decay) - self.lr * transformed


class MuonWithAuxAdamW(Optimizer):
    """Disjoint hidden-matrix Muon and auxiliary AdamW parameter groups."""
    def __init__(self, muon_params, adamw_params, lr=.02, aux_lr=.003,
                 momentum=.95, weight_decay=.01, ns_steps=5):
        muon_params, adamw_params = list(muon_params), list(adamw_params)
        params = muon_params + adamw_params
        if len({id(p) for p in params}) != len(params):
            raise ValueError('Muon and AdamW groups must be disjoint without duplicates')
        super().__init__(params)
        self.muon = Muon(muon_params, lr=lr, momentum=momentum,
                         weight_decay=weight_decay, ns_steps=ns_steps)
        self.adamw = AdamW(adamw_params, lr=aux_lr, weight_decay=weight_decay)

    @property
    def state_nbytes(self):
        return self.muon.state_nbytes + self.adamw.state_nbytes

    def step(self):
        self.muon.step()
        self.adamw.step()
