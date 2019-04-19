import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

import os
import pathlib


cuda_include_path = pathlib.Path(os.environ['CUDA_PATH']) / 'include'
dfxp_backend = load(name='dfxp_backend', sources=['dfxp.cpp'],
    extra_cflags=['-std=c++11', '-O3', '-I%s' % str(cuda_include_path), '-v'],
    verbose=True, build_directory='.')


class Quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, qmin, qmax, step):
        return dfxp_backend.dfxp_quantize_forward(X, qmin, qmax, step)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


class GradQuantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, qmin, qmax, step):
        ctx.save_for_backward(qmin, qmax, step)
        return X

    @staticmethod
    def backward(ctx, grad):
        qmin, qmax, step = ctx.saved_tensors
        grad = dfxp_backend.dfxp_grad_quantize_backward(grad, qmin, qmax, step)
        return grad, None, None, None


class ForwardQuantizer(nn.Module):

    def __init__(self, bits, step=2.0 ** -5):
        super().__init__()

        if bits == 32:
            self.forward = self.identity
        else:
            self.forward = self.forward_q

        self.register_buffer('qmin', torch.tensor(-(2.0 ** (bits - 1))))
        self.register_buffer('qmax', torch.tensor(2.0 ** (bits - 1) - 1))
        self.register_buffer('step', torch.tensor(step))

    def forward_q(self, X):
        return Quantize.apply(X, self.qmin, self.qmax, self.step)

    def identity(self, X):
        return X


class BackwardQuantizer(nn.Module):

    def __init__(self, bits, step=2.0 ** -5):
        super().__init__()

        if bits == 32:
            self.forward = self.identity
        else:
            self.forward = self.forward_q

        self.register_buffer('qmin', torch.tensor(-(2.0 ** (bits - 1))))
        self.register_buffer('qmax', torch.tensor(2.0 ** (bits - 1) - 1))
        self.register_buffer('step', torch.tensor(step))

    def forward_q(self, X):
        return GradQuantize.apply(X, self.qmin, self.qmax, self.step)

    def identity(self, X):
        return X


class Conv2d_q(nn.Module):

    def __init__(self, bits, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        def pair(x):
            try:
                for _ in x:
                    pass
                return x
            except TypeError:
                return [x, x]

        self.bits = bits
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels,
            self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        if self.bias is not None:
            self.bias_q = ForwardQuantizer(bits)
        else:
            self.bias_q = lambda x: x
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = (6 / fan_in) ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = F.conv2d(X, weight, bias, self.stride, self.padding)
        out = self.grad_q(out)
        return out

    def extra_repr(self):
        s = ('{bits}bits, {in_channels}, {out_channels}, '
             'kernel_size={kernel_size}, stride={stride}')
        return s.format(**self.__dict__)


class Linear_q(nn.Module):

    def __init__(self, bits, in_features, out_features, bias=True):
        super().__init__()

        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        if self.bias is not None:
            self.bias_q = ForwardQuantizer(bits)
        else:
            self.bias_q = lambda x: x
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        bound = (6 / (self.in_features + self.out_features)) ** 0.5
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = F.linear(X, weight, bias)
        out = self.grad_q(out)
        return out

    def extra_repr(self):
        s = '{bits}bits, {in_features}, {out_features}'
        return s.format(**self.__dict__)


class Normalize2d_q(nn.Module):

    def __init__(self, bits, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        self.bit = bits
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # pylint: disable=not-callable,no-member
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        # pylint: enable=not-callable,no-member
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, X):
        X = self.input_q(X)
        out = F.batch_norm(X, self.running_mean, self.running_var,
            weight=None, bias=None, training=self.training,
            momentum=self.momentum, eps=self.eps)
        out = self.grad_q(out)
        return out


class Rescale2d_q(nn.Module):

    def __init__(self, bits, num_features):
        super().__init__()

        self.bits = bits
        self.num_features = num_features

        self.weight = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(num_features, 1, 1))
        self.reset_parameters()

        self.input_q = ForwardQuantizer(bits)
        self.weight_q = ForwardQuantizer(bits)
        self.bias_q = ForwardQuantizer(bits)
        self.grad_q = BackwardQuantizer(bits)

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X):
        X = self.input_q(X)
        weight = self.weight_q(self.weight)
        bias = self.bias_q(self.bias)
        out = X * weight + bias
        out = self.grad_q(out)
        return out


class BatchNorm2d_q(nn.Module):

    def __init__(self, bits, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        self.bits = bits
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.normalize = Normalize2d_q(bits, num_features, eps, momentum)
        if affine:
            self.rescale = Rescale2d_q(bits, num_features)
            self.weight = self.rescale.weight
            self.bias = self.rescale.bias
        else:
            self.rescale = lambda x: x

    def forward(self, X):
        out = self.normalize(X)
        out = self.rescale(out)
        return out

    def extra_repr(self):
        s = '{bits}bits, {num_features}, eps={eps}, momentum={momentum}, affine={affine}'
        return s.format(**self.__dict__)
