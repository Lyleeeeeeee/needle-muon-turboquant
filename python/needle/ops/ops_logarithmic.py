from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import BACKEND, array_api


def _max_keepdims(value, axes):
    if BACKEND == "np":
        return array_api.max(value, axis=axes, keepdims=True)
    axes = tuple(range(len(value.shape))) if axes is None else axes
    axes = (axes,) if isinstance(axes, int) else axes
    for axis in axes:
        value = value.max(axis=axis % len(value.shape), keepdims=True)
    return value

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        shape = Z.shape[:-1] + (1,)
        normalizer = LogSumExp(axes=(-1,)).compute(Z).reshape(shape)
        return Z - array_api.broadcast_to(normalizer, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        summed_grad = summation(out_grad, axes=(-1,))

        reshape_shape = list(out_grad.shape)
        reshape_shape[-1] = 1

        summed_grad = reshape(
            summed_grad, tuple(reshape_shape)
        ).broadcast_to(out_grad.shape)

        return out_grad - exp(node) * summed_grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = _max_keepdims(Z, self.axes)
        shifted = Z - array_api.broadcast_to(max_z, Z.shape)
        sum_exp = Summation(self.axes).compute(array_api.exp(shifted))
        return array_api.log(sum_exp) + max_z.reshape(sum_exp.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        axes = self.axes
        if axes is None:
            axes = tuple(range(len(Z.shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        reshape_shape = list(Z.shape)
        for axis in axes:
            reshape_shape[axis] = 1
        reshape_shape = tuple(reshape_shape)

        logsumexp_z = reshape(node, reshape_shape).broadcast_to(
            Z.shape
        )
        out_grad = reshape(out_grad, reshape_shape).broadcast_to(
            Z.shape
        )
        return out_grad * exp(Z - logsumexp_z)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
