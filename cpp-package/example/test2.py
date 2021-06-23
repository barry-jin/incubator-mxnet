import mxnet as mx
from mxnet import np, npx, gluon
from mxnet.gluon import nn
npx.set_np()

# mx.context._current.set(mx.gpu(0))

def check_layer_forward_withinput(net, x):
    # x_hybrid = x.copy()
    # x.attach_grad()
    # x_hybrid.attach_grad()
    net.initialize()
    net.hybridize()
    # with mx.autograd.record():
    out2 = net(x)
    
    # out2.backward()
    # mx.npx.waitall()
    # a, b = mx.context.gpu_memory_info(0)
    # print("Used memory {} GB, Total memory {} GB.".format((b - a) / (1024 * 1024 * 1024), b / (1024 * 1024 * 1024)))

def test():
    layer1 = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
    layer2 = nn.AvgPool2D(strides=(2, 2), padding=(1, 1))
    layer3 = nn.GlobalMaxPool2D()
    layer4 = nn.GlobalAvgPool2D()
    class Net(gluon.HybridBlock):
        def __init__(self,
                     **kwargs):
            super(Net, self).__init__(**kwargs)
            self.layer1 = nn.MaxPool2D(strides=(2, 3), padding=(1, 1))
            self.layer2 = nn.GlobalAvgPool2D()

        def forward(self, x):
            y = self.layer1(x)
            out = self.layer2(y)
            return out

    xshape = (4, 32, 64, 64)
    x = mx.np.random.uniform(size=xshape)
    for _ in range(7):
        net = Net()
        check_layer_forward_withinput(net, x)

if __name__ == "__main__":
    test()



