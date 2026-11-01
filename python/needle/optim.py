"""Optimization module"""
import needle as ndl
import numpy as np


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

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.data
            t = self.steps.get(param, 0) + 1
            self.steps[param] = t
            self.m[param] = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * grad ** 2
            m_hat = self.m[param] / (1 - self.beta1 ** t)
            v_hat = self.v[param] / (1 - self.beta2 ** t)
            param.data = param.data * (1 - self.lr * self.weight_decay) - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
