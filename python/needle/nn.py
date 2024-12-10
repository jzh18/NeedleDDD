"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
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


def _child_modules(value: object) -> List["Module"]:
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
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in=in_features, fan_out=out_features, device=device, dtype=dtype, requires_grad=True))
        if bias == True:
            self.bias = Parameter(init.kaiming_uniform(
                fan_in=out_features, fan_out=1, device=device, dtype=dtype, requires_grad=True).reshape((1, out_features)))
        # END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        x_shape = X.shape  # (batch_size, in_features)
        # BEGIN YOUR SOLUTION
        a = ops.matmul(X, self.weight)  # (batch_size,out_features)
        if self.bias == None:
            return a

        broadcast_size = [i for i in x_shape]
        broadcast_size[-1] = self.out_features

        b = ops.broadcast_to(self.bias, shape=tuple(broadcast_size))
        return a+b
        # END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        # BEGIN YOUR SOLUTION
        shape = X.shape
        batch_size = shape[0]
        flatten_len = 1
        for i in shape[1:]:
            flatten_len *= i
        return X.reshape((batch_size, flatten_len))
        # END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.relu(x)
        # END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return ops.tanh(x)
        # END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return (ops.exp(-1*x)+1)**(-1)
        # END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        for i, m in enumerate(self.modules):
            x = m(x)
            # print((f'x{i}: {x}')
        return x
        # END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # BEGIN YOUR SOLUTION

        num_classes = logits.shape[1]
        onehot = init.one_hot(num_classes, y, device=y.device, dtype=y.dtype)

        z_y = ops.multiply(onehot, logits)
        exps_up = ops.summation(z_y, axes=(1))

        exps_down = ops.logsumexp(logits, axes=(1,))
        res = ops.summation(
            exps_down-exps_up) / logits.shape[0]

        return res
        # END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype,)
        self.running_var = init.ones(dim, device=device, dtype=dtype,)
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        batch_size, in_features = x.shape

        sum = ops.summation(x, axes=(0,))
        new_running_mean = sum / batch_size
        # print(f'sum: {sum}')
        # print(f'batch_size: {batch_size}')
        # print(f'running mean: {new_running_mean}')
        if self.training:
            self.running_mean.data = ((1-self.momentum)*self.running_mean +
                                      self.momentum*new_running_mean).data
        mean = new_running_mean.reshape((1, in_features))
        if not self.training:
            mean = self.running_mean
        broadcast_mean = ops.broadcast_to(mean, (batch_size, in_features))

        new_running_var = (ops.summation(ops.power_scalar(
            x-broadcast_mean, 2), axes=(0,))/batch_size)
        if self.training:
            self.running_var.data = ((1-self.momentum)*self.running_var +
                                     self.momentum*new_running_var).data
        var = new_running_var.reshape((1, in_features))
        if not self.training:
            var = self.running_var
        broadcast_var = ops.broadcast_to(var, (batch_size, in_features))

        new_x = ops.divide((x-broadcast_mean),
                           ops.power_scalar((broadcast_var+self.eps), 0.5))  # (batch_size, in_features)
        broadcast_weight = ops.broadcast_to(
            self.weight, (batch_size, in_features))
        broadcast_bias = ops.broadcast_to(self.bias, (batch_size, in_features))

        return ops.multiply(new_x, broadcast_weight)+broadcast_bias

        # END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose(
            (2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2, 3)).transpose((1, 2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(
            dim, device=device, dtype=dtype, requires_grad=True))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        batch_size, in_features = x.shape  # (batch_size, in_features)
        mean = (ops.summation(x, axes=(1,)) /
                in_features).reshape((batch_size, 1))
        broadcast_mean = ops.broadcast_to(mean, (batch_size, in_features))
        var = (ops.summation(ops.power_scalar(
            x-broadcast_mean, 2), axes=(1,))/in_features).reshape((batch_size, 1))
        broadcast_var = ops.broadcast_to(var, (batch_size, in_features))
        new_x = ops.divide((x-broadcast_mean),
                           ops.power_scalar((broadcast_var+self.eps), 0.5))  # (batch_size, in_features)

        broadcast_weight = ops.broadcast_to(
            self.weight, (batch_size, in_features))
        broadcast_bias = ops.broadcast_to(self.bias, (batch_size, in_features))
        return ops.multiply(new_x, broadcast_weight)+broadcast_bias

        # END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        shape = x.shape
        len = 1
        for i in shape:
            len *= i
        if self.training:
            zeros = init.randb(len, p=(1-self.p)).reshape(shape)
            return ops.multiply(zeros, x)/(1-self.p)
        else:
            return x
        # END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        return self.fn(x)+x
        # END YOUR SOLUTION


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = None
        # print((f'out channel: {out_channels}')
        # BEGIN YOUR SOLUTION
        fan_in = self.kernel_size*self.kernel_size*self.in_channels
        fan_out = self.kernel_size*self.kernel_size*self.out_channels
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out, shape=(
            kernel_size, kernel_size, in_channels, out_channels), device=device, dtype=dtype, requires_grad=True))
        if bias == True:
            # ± 1.0/(in_channels * kernel_size**2)**0.5
            bound = 1.0/((in_channels*kernel_size**2)**0.5)
            self.bias = Parameter(init.rand(
                self.out_channels, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        # BEGIN YOUR SOLUTION
        N, C, H, W = x.shape
        # print(f'original shape: {x.shape}')
        # print(f'kernel shape: {self.weight.shape}')
        # print(f'stride: {self.stride}')

        padding = (self.kernel_size-1)//2
        # print(f'pad: {padding}')
        x = ops.transpose(x, (1, 2))  # NHCW
        x = ops.transpose(x, (2, 3))  # NHWC
        # print(f'_x: {x.shape}')
        output = ops.conv(x, self.weight, padding=padding,
                          stride=self.stride)  # NHWC
        output = ops.transpose(output, (2, 3))  # NHCW
        output = ops.transpose(output, (1, 2))  # NCHW
        # print(f'output shape: {output.shape}')
        if self.bias == None:
            return output

        # print(f'b shape: {self.bias.shape}')
        # print(f'b broadcast shape: {output.shape}')
        b = ops.broadcast_to(self.bias.reshape(
            (self.out_channels, 1, 1)), shape=output.shape)
        output = output+b
        return output

        # END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.nonlinear = Tanh()
        elif nonlinearity == 'relu':
            self.nonlinear = ReLU()

        bound = (1.0/hidden_size)**0.5
        self.W_ih = Parameter(init.rand(
            input_size*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True).reshape((input_size, hidden_size)))
        self.W_hh = Parameter(init.rand(
            hidden_size*hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True).reshape((hidden_size, hidden_size)))
        if self.bias:
            self.bias_ih = Parameter(init.rand(
                hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(
                hidden_size, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))

        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        # ℎ′=tanh(𝑥𝑊𝑖ℎ+𝑏𝑖ℎ+ℎ𝑊ℎℎ+𝑏ℎℎ)
        # BEGIN YOUR SOLUTION
        bs, _ = X.shape
        # print(f'X shape: {X.shape}')
        # print(f'W_ih shape: {self.W_ih.shape}')
        x_W_ih = X@self.W_ih  # (bs, hidden_size)
        if h is None:
            h = init.zeros(bs*self.hidden_size, device=X.device,
                           dtype=X.dtype).reshape((bs, self.hidden_size))
        h_W_hh = h@self.W_hh  # (bs, hidden_size)
        _h = x_W_ih+h_W_hh
        if self.bias:
            broadcast_bias_ih = ops.broadcast_to(
                self.bias_ih, shape=(bs, self.hidden_size))
            broadcast_bias_hh = ops.broadcast_to(
                self.bias_hh, shape=(bs, self.hidden_size))
            _h = _h+broadcast_bias_ih+broadcast_bias_hh
        _h = self.nonlinear(_h)
        return _h
        # END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device
        self.dtype = dtype
        self.rnn_cells = []
        for i in range(self.num_layers):
            cell = RNNCell(input_size, hidden_size, bias,
                           nonlinearity, device, dtype)
            input_size = hidden_size
            self.rnn_cells.append(cell)
        # END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        # ℎ𝑡=tanh(𝑥𝑡𝑊𝑖ℎ+𝑏𝑖ℎ+ℎ(𝑡−1)𝑊ℎℎ+𝑏ℎℎ)
        # BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers*bs*self.hidden_size, device=X.device,
                            dtype=X.dtype).reshape((self.num_layers, bs, self.hidden_size))
        X = ops.split(X, axis=0)
        h0 = ops.split(h0, axis=0)
        h = h0
        output = []
        for x in X:  # x: (bs, input_size), iteration for seq_len times
            new_h = []
            for i, c in enumerate(self.rnn_cells):
                # h[i]: (bs,hidden_size), h_it: (bs, hidden_size)
                #print(f'x shape: {x.shape}')
                #print(f'h[i] shape: {h[i].shape}')
                h_it = c(x, h[i])  # (bs, hidden_size)
                #print(f'h_it shape: {h_it.shape}')
                new_h.append(h_it)
                x = h_it
            h = new_h  # num_layers,bs, hidden_size
            output.append(x)
        output = ops.stack(output, axis=0)  # (seq_len,bs, hidden_size)
        h = ops.stack(h, axis=0)
        return output, h
        # END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype
        bound = (1.0/hidden_size)**0.5
        self.W_ih = Parameter(init.rand(
            input_size*hidden_size*4, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True).reshape((input_size, 4*hidden_size)))
        self.W_hh = Parameter(init.rand(
            hidden_size*hidden_size*4, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True).reshape((hidden_size, 4*hidden_size)))
        if self.bias:
            self.bias_ih = Parameter(init.rand(
                hidden_size*4, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(
                hidden_size*4, low=-bound, high=bound, device=device, dtype=dtype, requires_grad=True))
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        # BEGIN YOUR SOLUTION
        bs, input_size = X.shape
        # input_size, 4*hidden_size
        W_ih = ops.split(self.W_ih.reshape(
            (input_size, 4, self.hidden_size)), axis=1)
        # input_size, hidden_size

        W_ii = W_ih[0]
        W_if = W_ih[1]  # ditto
        W_ig = W_ih[2]  # ditto
        W_io = W_ih[3]  # ditto

        W_hh = ops.split(self.W_hh.reshape(
            (self.hidden_size, 4, self.hidden_size)), axis=1)  # hidden_size, 4*hidden_size
        # hidden_size, hidden_size
        W_hi = W_hh[0]
        W_hf = W_hh[1]  # ditto
        W_hg = W_hh[2]
        W_ho = W_hh[3]
        if h is None:
            h = init.zeros(bs*self.hidden_size, device=X.device,
                           dtype=X.dtype).reshape((bs, self.hidden_size))
            c = init.zeros(bs*self.hidden_size, device=X.device,
                           dtype=X.dtype).reshape((bs, self.hidden_size))
        else:
            c = h[1]
            h = h[0]

        # print(f'c device: {c.device}')
        # print(f'h device: {h.device}')
        if self.bias:
            b_ih = self.bias_ih.reshape((4, self.hidden_size))
            b_ih = ops.split(b_ih, axis=0)
            b_ii = ops.broadcast_to(
                b_ih[0], shape=(bs, self.hidden_size))
            b_if = ops.broadcast_to(
                b_ih[1], shape=(bs, self.hidden_size))
            b_ig = ops.broadcast_to(
                b_ih[2], shape=(bs, self.hidden_size))
            b_io = ops.broadcast_to(
                b_ih[3], shape=(bs, self.hidden_size))

            b_hh = self.bias_hh.reshape((4, self.hidden_size))
            b_hh = ops.split(b_hh, axis=0)
            b_hi = ops.broadcast_to(
                b_hh[0], shape=(bs, self.hidden_size))
            b_hf = ops.broadcast_to(
                b_hh[1], shape=(bs, self.hidden_size))
            b_hg = ops.broadcast_to(
                b_hh[2], shape=(bs, self.hidden_size))
            b_ho = ops.broadcast_to(
                b_hh[3], shape=(bs, self.hidden_size))

        i = X@W_ii+h@W_hi  # bs, hidden_size
        if self.bias:
            i = i+b_ii+b_hi
        i = self.sigmoid(i)

        f = X@W_if+h@W_hf
        if self.bias:
            f = f+b_if+b_hf
        f = self.sigmoid(f)

        g = X@W_ig+h@W_hg
        if self.bias:
            g = g+b_ig+b_hg
        g = self.tanh(g)

        o = X@W_io+h@W_ho
        if self.bias:
            o = o+b_io+b_ho
        o = self.sigmoid(o)

        _c = f*c+i*g  # bs,hidden_size
        _h = o*self.tanh(_c)  # bs,hidden_size
        return _h, _c

        # END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        # BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.lstm_cells = []
        for i in range(self.num_layers):
            cell = LSTMCell(input_size, hidden_size, bias, device, dtype)
            input_size = hidden_size
            self.lstm_cells.append(cell)
        # END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        # BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h is None:
            h0 = init.zeros(self.num_layers*bs*self.hidden_size, device=X.device,
                            dtype=X.dtype).reshape((self.num_layers, bs, self.hidden_size))
            c0 = init.zeros(self.num_layers*bs*self.hidden_size, device=X.device,
                            dtype=X.dtype).reshape((self.num_layers, bs, self.hidden_size))
        else:
            h0 = h[0]
            c0 = h[1]
        X = ops.split(X, axis=0)
        h0 = ops.split(h0, axis=0)
        c0 = ops.split(c0, axis=0)
        h = h0
        c = c0
        output = []
        for x in X:  # x: (bs, input_size), iteration for seq_len times
            new_h = []
            new_c = []
            for i, cell in enumerate(self.lstm_cells):
                # h[i]: (bs,hidden_size), h_it: (bs, hidden_size)
                h_it, c_it = cell(x, (h[i], c[i]))  # (bs, hidden_size)
                #print(f'h_it shape: {h_it.shape}')
                new_h.append(h_it)
                new_c.append(c_it)
                x = h_it
            h = new_h  # num_layers,bs, hidden_size
            c = new_c
            output.append(x)
        output = ops.stack(output, axis=0)  # (seq_len,bs, hidden_size)
        h = ops.stack(h, axis=0)
        c = ops.stack(c, axis=0)
        return output, (h, c)
        # END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        # BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(
            num_embeddings*embedding_dim, device=device, dtype=dtype, requires_grad=True).reshape((num_embeddings, embedding_dim)))
        # END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        # BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        # (seq_len, bs, num_embeddings)
        onehot = init.one_hot(self.num_embeddings, x,
                              device=x.device, dtype=x.dtype)
        onehot = onehot.reshape((seq_len*bs, self.num_embeddings))
        x = onehot@self.weight  # seq_len*bs, embedding)dim
        x = x.reshape((seq_len, bs, self.embedding_dim))
        return x

        # END YOUR SOLUTION
