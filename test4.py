import mxnet as mx
from mxnet import np, npx, gluon
from mxnet.gluon import nn
import sys
from mxnet import autograd
npx.set_np()

mx.context._current.set(mx.gpu(0))

def test_slice_pooling2d():
    class Net(gluon.HybridBlock):
        def __init__(self,
                        slice,
                        **kwargs):
            super(Net, self).__init__(**kwargs)
            self.slice = slice
            self.pool0 = nn.GlobalMaxPool2D(layout='NCHW')

        def forward(self, x):
            x_slice = mx.npx.slice(x, begin=self.slice[0], end=self.slice[1])
            out = self.pool0(x_slice)
            return out

    xshape = (16, 128, 256, 256)
    slice_shape = (4, 16, 32, 64)
    x = mx.np.random.uniform(size=xshape)
    slice = [(0, 0, 0, 0), slice_shape]
    net = Net(slice)
    net.initialize()
    with mx.autograd.record():
        out1 = net(x)
    out1.backward()

if __name__ == '__main__':
    test_slice_pooling2d()
