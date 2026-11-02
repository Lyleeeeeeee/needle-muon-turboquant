"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                in_features, out_features, device=device, dtype=dtype
            )
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features, 1, device=device, dtype=dtype
                ).reshape((out_features,))
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
            out = out + self.bias.reshape(
                (1, self.out_features)
            ).broadcast_to(
                (X.shape[0], self.out_features)
            )
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        flattened_dim = 1
        for dim in X.shape[1:]:
            flattened_dim *= dim
        return X.reshape(
            (X.shape[0], flattened_dim)
        )
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape

        y_one_hot = init.one_hot(
            num_classes,
            y,
            device=logits.device,
            dtype=logits.dtype,
        )

        correct_logits = ops.summation(
            logits * y_one_hot,
            axes=(1,),
        )
        losses = (
            ops.logsumexp(logits, axes=(1,))
            - correct_logits
        )
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )
        self.running_mean = init.zeros(
            dim, device=device, dtype=dtype
        )
        self.running_var = init.ones(
            dim, device=device, dtype=dtype
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]

        if self.training:
            mean = ops.summation(
                x, axes=(0,)
            ) / batch_size

            mean_broadcast = mean.reshape(
                (1, self.dim)
            ).broadcast_to(x.shape)

            centered = x - mean_broadcast

            variance = ops.summation(
                centered ** 2, axes=(0,)
            ) / batch_size

            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.data
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * variance.data
            )
        else:
            mean = self.running_mean
            variance = self.running_var
            mean_broadcast = mean.reshape(
                (1, self.dim)
            ).broadcast_to(x.shape)

            centered = x - mean_broadcast


        variance = variance.reshape(
            (1, self.dim)
        ).broadcast_to(x.shape)

        normalized = centered / (
            variance + self.eps
        ) ** 0.5

        weight = self.weight.reshape(
            (1, self.dim)
        ).broadcast_to(x.shape)
        bias = self.bias.reshape(
            (1, self.dim)
        ).broadcast_to(x.shape)

        return normalized * weight + bias
        ### END YOUR SOLUTION




class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]

        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = mean.reshape(
            (batch_size, 1)
        ).broadcast_to(x.shape)

        centered = x - mean

        variance = ops.summation(
            centered ** 2, axes=(1,)
        ) / self.dim
        variance = variance.reshape(
            (batch_size, 1)
        ).broadcast_to(x.shape)

        normalized = centered / (
            variance + self.eps
        ) ** 0.5

        weight = self.weight.reshape(
            (1, self.dim)
        ).broadcast_to(x.shape)
        bias = self.bias.reshape(
            (1, self.dim)
        ).broadcast_to(x.shape)

        return normalized * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x

        mask = init.randb(
            *x.shape,
            p=1 - self.p,
            device=x.device,
            dtype=x.dtype,
        )
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
