import mxnet as mx
from mxnet import npx, gluon
npx.set_np()

def test():
    class Net(gluon.HybridBlock):
        def __init__(self,
                     **kwargs):
            super(Net, self).__init__(**kwargs)

        def forward(self, x):
            out = npx.activation(x)
            return out

    xshape = (2, 8, 16, 16)
    x = mx.np.random.uniform(size=xshape)
    for _ in range(50):
        net = Net()
        net.initialize()
        net.hybridize()
        out1 = net(x)

if __name__ == "__main__":
    test()
