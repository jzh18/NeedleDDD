"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy
import math

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # #print((f'EWiseAdd: {out_grad}')
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        # #print((f'multiply grad: {out_grad * rhs},{out_grad * lhs}')
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # BEGIN YOUR SOLUTION
        return a**self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return (node.inputs[0]**(self.scalar-1)) * out_grad*(self.scalar)
        # END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        left, right = node.inputs

        # tmp1 = 1 / right
        # tmp2 = -1*left/(right*right)
        # print(f'tmp1 shape: {tmp1.shape}')
        # print(f'tmp2 shape: {tmp2.shape}')

        return out_grad/right, -1*out_grad*left/(right*right)
        # END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a/self.scalar
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        res = out_grad*1/self.scalar
        # #print((f'div scalar: {res}')
        return res
        # END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is not None:
            axes[self.axes[0]] = self.axes[1]
            axes[self.axes[1]] = self.axes[0]
        else:
            tmp = axes[-1]
            axes[-1] = axes[-2]
            axes[-2] = tmp

        return array_api.transpose(a, axes=axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        # END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        # END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        res = array_api.broadcast_to(a, self.shape)
        return res

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # print((f'broadcast_to outgrad: {out_grad.shape}')

        broadcast_shape = list(self.shape)

        broadcast_shape.reverse()
        # print((f'broadcast shape reversed: {broadcast_shape}')
        input_shape = list(node.inputs[0].shape)
        input_shape.reverse()
        # print((f'input shape reversed: {input_shape}')
        if len(input_shape) == 1:
            shape = input_shape[0]
            for i, v in enumerate(broadcast_shape):
                if v != shape:
                    input_shape.insert(0, 1)
                else:
                    break

        broad_axes = []

        final_index = len(broadcast_shape)-1
        for i, v in enumerate(broadcast_shape):
            # print((f'i:{i}, v:{v}')
            if i < len(input_shape):
                # print((f'input_shape[i]: {input_shape[i]}')
                if input_shape[i] == 1 and v > 1:
                    broad_axes.append(final_index-i)
            else:
                broad_axes.append(final_index-i)
        # print((f'sum axes: {broad_axes}')
        grad = out_grad.sum(axes=tuple(broad_axes)).reshape(
            node.inputs[0].shape)
        # #print((f'broadcast_to grad: {grad}')

        return grad

        # END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))
        if isinstance(self.axes, int):
            self.axes = (self.axes,)
        new_axes = list(self.axes)
        new_axes.sort()
        new_axes.reverse()
        for i in new_axes:
            a = array_api.summation(a, axis=i)
        return a

        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # print((f'out grad stride: {out_grad.realize_cached_data().strides}')
        # print((f'out grad shape: {out_grad.realize_cached_data().shape}')
        # print((f'out grad: {out_grad}')
        if self.axes is None:
            grad = out_grad.broadcast_to(node.inputs[0].shape)
            return grad

        new_shape = list(node.inputs[0].shape)
        if self.axes is not None:
            if type(self.axes) is not tuple:
                new_shape[self.axes] = 1
            else:
                for i in self.axes:
                    new_shape[i] = 1
        grad = out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        # #print((f'summation grad: {grad}')
        # BEGIN YOUR SOLUTION
        return grad
        # END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        left, right = node.inputs
        right_shape_len = len(right.shape)
        left_shape_len = len(left.shape)

        right_index = list(range(right_shape_len))
        left_index = list(range(left_shape_len))
        right_transposed = right.transpose(right_index[-2:])
        left_transposed = left.transpose(left_index[-2:])

        grad_left = out_grad.matmul(right_transposed)
        grad_right = left_transposed.matmul(out_grad)

        left_extend_len = len(grad_left.shape)-left_shape_len
        right_extend_len = len(grad_right.shape)-right_shape_len

        if left_extend_len > 0:
            axes = list(range(left_extend_len))
            grad_left = grad_left.sum(tuple(axes))

        if right_extend_len > 0:
            axes = list(range(right_extend_len))
            grad_right = grad_right.sum(tuple(axes))
        # #print((f'matmul grad: {grad_left},{grad_right}')
        return grad_left, grad_right
        # END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return -1*a
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad*-1
        # END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.log(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad*1/node.inputs[0]
        # END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return array_api.exp(a)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        # END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        data = (a > 0) * a
        return data
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()
        return out_grad*Tensor(data > 0, device=out_grad.device, dtype=out_grad.dtype)


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if type(axes) is int:
            axes = (axes,)
        self.axes = axes

    def compute(self, Z):
        # BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes)
        # array_api.max(Z, axis=self.axes)

        if self.axes is not None:
            reshape_size = [1]*len(Z.shape)
            for i in range(len(Z.shape)):
                if i not in self.axes:
                    reshape_size[i] = Z.shape[i]
            resize_max_z = max_z.compact().reshape(reshape_size)
            new_z = Z-array_api.broadcast_to(resize_max_z, Z.shape)
        else:
            new_z = Z-array_api.broadcast_to(max_z, Z.shape)

        res = array_api.log(array_api.summation(
            array_api.exp(new_z), axis=self.axes))+max_z

        return res
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        data = node.inputs[0].realize_cached_data()

        max_z = data.max(axis=self.axes)

        if self.axes is not None:
            reshape_size = [1]*len(data.shape)
            for i in range(len(data.shape)):
                if i not in self.axes:
                    reshape_size[i] = data.shape[i]
            resize_max_z = max_z.reshape(reshape_size)
            max_z = array_api.broadcast_to(resize_max_z, data.shape)

        exps = array_api.exp(data-max_z)

        exps_down = array_api.summation(exps, axis=self.axes)
        if self.axes is not None:
            reshape_size = [1]*len(data.shape)
            for i in range(len(data.shape)):
                if i not in self.axes:
                    reshape_size[i] = data.shape[i]
            resize_exps_down = exps_down.reshape(reshape_size)
            exps_down = array_api.broadcast_to(resize_exps_down, data.shape)
            resize_out_grad = out_grad.reshape(reshape_size)
            out_grad = resize_out_grad.broadcast_to(data.shape)

        res = Tensor(
            out_grad.realize_cached_data() * exps/exps_down, device=out_grad.device, dtype=out_grad.dtype)

        # #print((f'logsumexp grad: {res}')
        return res
        # END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        # BEGIN YOUR SOLUTION
        up = array_api.exp(a)-array_api.exp(-a)
        down = array_api.exp(a)+array_api.exp(-a)
        return up/down
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # bug in needle: 1-Tensor(0.2)=-0.8
        #
        a = node.inputs[0]
        up = exp(a)-exp(-a)
        down = exp(a)+exp(-a)
        data = up/down
        return out_grad*(1-data*data)
        # END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        # BEGIN YOUR SOLUTION
        res_shape = list(args[0].shape)
        res_shape.insert(self.axis, len(args))
        res = array_api.full(res_shape, 0, device=args[0].device)

        idxs = []
        for i in res_shape:
            idxs.append(slice(i))
        for i in range(len(args)):
            idxs[self.axis] = i
            res[tuple(idxs)] = args[i]

        return res
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        # END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        # BEGIN YOUR SOLUTION

        res_shape = list(A.shape)
        res_shape.pop(self.axis)
        idxs = []
        for i in A.shape:
            idxs.append(slice(i))
        res = []
        for i in range(A.shape[self.axis]):
            idxs[self.axis] = i
            ele = A[tuple(idxs)].compact().reshape(tuple(res_shape))
            res.append(ele)
        return tuple(res)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        # END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return out_grad.realize_cached_data().flip(self.axes)
        # END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION

        new_shape = list(a.shape)
        for i in self.axes:
            if i < len(a.shape):
                new_shape[i] = new_shape[i]*self.dilation+new_shape[i]
        idxs = [slice(0, s, 1) for s in new_shape]
        for i in self.axes:
            if i < len(a.shape):
                old_idx = idxs[i]
                idxs[i] = slice(old_idx.start, old_idx.stop,
                                self.dilation+old_idx.step)
        # #print((f'dia: {self.dilation}')
        # #print((f'axes: {self.axes}')
        # #print((f'old shape: {a.shape}')
        # #print((f'new_shape: {new_shape}')
        arr = array_api.full(tuple(new_shape), 0,
                             dtype=a.dtype, device=a.device)
        # #print((f'idxs: {idxs}')
        arr[tuple(idxs)] = a
        return arr
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        # END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        # BEGIN YOUR SOLUTION

        idxs = [slice(0, s, 1) for s in a.shape]
        for i in self.axes:
            if i < len(a.shape):
                old_idx = idxs[i]
                idxs[i] = slice(old_idx.start, old_idx.stop,
                                self.dilation+old_idx.step)

        return a[tuple(idxs)]
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        raise NotImplementedError()
        # END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        # #print((f'stride: {self.stride}')
        # #print((f'padding: {self.padding}')
        # #print((f'A: {A[0,:,:,0]}')
        # BEGIN YOUR SOLUTION
        padded_A = A.pad(((0, 0), (self.padding, self.padding),
                         (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = padded_A.shape
        # print(f'C_in in conv: {C_in}')

        K, _, _, C_out = B.shape
        Ns, Hs, Ws, Cs = padded_A.strides

        inner_dim = K*K*C_in
        # h_out = int((H+2*self.padding-K)/self.stride+1)
        # w_out = int((W+2*self.padding-K)/self.stride+1)
        # h_out = math.floor((H-K)/self.stride)+1
        # w_out = math.floor((W-K)/self.stride)+1
        h_out = H-K+1
        w_out = W-K+1
        # #print((f'h_out: {h_out}, w_out: {w_out}')
        new_shape = (N, h_out, w_out, K, K, C_in)
        new_strides = (Ns, Hs, Ws, Hs, Ws, Cs)

        # must need to be compacted
        A = NDArray.make(new_shape, strides=new_strides,
                         device=padded_A._device, handle=padded_A._handle, offset=padded_A._offset)

        hs_strided_idxs = slice(0, h_out, self.stride)
        ws_strided_idxs = slice(0, h_out, self.stride)
        # #print((f'before: {A.shape}')
        # #print((f'A ele: {A[0,0,2,:,:,0]}')

        # TODO: must need to be compacted
        # looks like if you need underlayer (C++/C) operation, you need to compact the array
        A = A[:, hs_strided_idxs, ws_strided_idxs, :, :, :].compact()
        # #print((f'after: {A.shape}')
        # #print((f'A ele: {A[0,0,1,:,:,0]}')

        final_h_out = math.floor((H-K)/self.stride)+1
        final_w_out = math.floor((W-K)/self.stride)+1

        A = A.reshape((N*final_h_out*final_w_out, inner_dim))

        # TODO: must need to be compacted
        # looks like if you need underlayer (C++/C) operation, you need to compact the array
        B = B.compact()
        # print(f'B shape in conv: {B.shape}')
        # print(f'B new shape in conv: {(K*K*C_in, C_out)}')
        # print(f'K,K,C_in in conv: {K},{K},{C_in}')
        out = A@B.reshape((K*K*C_in, C_out))
        out = out.compact()
        out = out.reshape((N, final_h_out, final_w_out, C_out))

        return out
        # END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # BEGIN YOUR SOLUTION
        # out_grad: N,H-K+1,W-K+1,C_out
        # padding: N,H-K+2*pad+1,W-K+2*pad+1,C_out
        # stride: N,(H-K+2*pad)/s+1,(W-K+2*pad)/s+1,C_out
        # print((f'out_grad shape: {out_grad.shape}')
        # print((f'stride: {self.stride}')
        _out_grad = dilate(out_grad, (1, 2), self.stride-1)
        # print((f'_out_grad shape: {_out_grad.shape}')
        X, W = node.inputs
        # print((f'W shape: {W.shape}')
        _W = flip(W, (0, 1))
        _W = transpose(_W, (2, 3))  # K,K,C_out,C_in

        # print((f'_W shape: {_W.shape}')
        # N,H,W,C_in
        # what if padding is very large?
        X_grad = conv(_out_grad, _W, padding=_W.shape[0]-1-self.padding)
        # print((f'X_grad shape: {X_grad.shape}')

        # print((f'X_shape: {X.shape}')
        _X = transpose(X, (0, 3))  # C_in,H,W,N
        # print((f'_X shape: {_X.shape}')
        # looks like we assume H=W, so that we can think _out_grad as a kernel
        # H-K+1,W-K+1,N,C_out; padding:N,H-K+1+2*pad,W-K+1+2*pad,C_out; stride:  N,(H-K+2*pad)/s+1,(W-K+2*pad)/s+1,C_out
        #_out_grad = transpose(_out_grad, (0, 2))
        _out_grad = transpose(_out_grad, (0, 1))
        _out_grad = transpose(_out_grad, (1, 2))
        # print((f'_out_grad shape: {_out_grad.shape}')

        W_grad = conv(_X, _out_grad, padding=self.padding)  # C_in,K,K,C_out
        # print((f'W_grad shape: {W_grad.shape}')
        W_grad = transpose(W_grad, (0, 1))
        # K,K,C_in,C_out, https://forum.dlsyscourse.org/t/q3-convolution-backward/2824
        W_grad = transpose(W_grad, (1, 2))

        # print((f'W_grad shape: {W_grad.shape}')

        return X_grad, W_grad
        # END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
