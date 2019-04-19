import torch
import torch.nn as nn
import torch.nn.functional as F

import dfxp

from IPython import get_ipython
ipython = get_ipython()


def test_int8():
    x = torch.randint(-128, 128, (32, 224, 224, 4)).to(torch.int8).cuda()
    w = torch.randint(-128, 128, (64, 7, 7, 4)).to(torch.int8).cuda()
    dfxp.dfxp_backend.dfxp_8bit_convolution_forward(x, w,
        [3, 3], [1, 1], [1, 1], 1) # first run
    ipython.magic(r'''timeit dfxp.dfxp_backend.dfxp_8bit_convolution_forward(x, w,
        [3, 3], [1, 1], [1, 1], 1)''')


def test_float():
    x = torch.randn(32, 4, 224, 224).cuda()
    w = torch.randn(64, 4, 7, 7).cuda()
    print('F.conv2d')
    y1 = F.conv2d(x, w, padding=3)
    ipython.magic(r'''timeit F.conv2d(x, w, padding=3)''')

    print('dfxp_32bit_convolution_forward')
    y2 = dfxp.dfxp_backend.dfxp_32bit_convolution_forward(x, w,
        [3, 3], [1, 1], [1, 1], 1) # first run
    ipython.magic(r'''timeit dfxp.dfxp_backend.dfxp_32bit_convolution_forward(x, w,
        [3, 3], [1, 1], [1, 1], 1)''')

    print('Mean element error:', (y1-y2).abs().mean().item())


def test_foo():
    dfxp.dfxp_backend.foo()


if __name__ == '__main__':
    # print('Testing foo..')
    # test_foo()
    print('Testing int8..')
    test_int8()
    print('Testing float..')
    test_float()