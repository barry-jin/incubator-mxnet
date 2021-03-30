# import mxnet as mx
# from mxnet.gluon import HybridBlock, nn
# from mxnet import np, npx
# npx.set_np()
# class Model(HybridBlock):
#     def __init__(self, **kwargs):
#         super(Model, self).__init__(**kwargs)
#         self.dense0 = nn.Dense(2)
#         self.dense1 = nn.Dense(2)

#     def forward(self, x):
#         x = self.relu(self.dense0(x))
#         return self.relu(self.dense1(x))

#     def relu(self, X):
#         return np.maximum(X, 0)

# model = Model()
# model.initialize(ctx=mx.cpu(0))
# model.hybridize()
# with mx.autograd.record():
#     res = model(mx.np.zeros((2, 2), ctx=mx.cpu(0)))
# res.backward()

import timeit

setup = """
import mxnet as mx
from mxnet.gluon import HybridBlock, nn
from mxnet import np, npx
npx.set_np()
class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.dense0 = nn.Dense(2)
        self.dense1 = nn.Dense(2)

    def forward(self, x):
        # return self.relu(self.dense0(x))
        x = self.relu(self.dense0(x))
        return self.relu(self.dense1(x))

    def relu(self, X):
        return np.maximum(X, 0)

model = Model()
model.initialize(ctx=mx.cpu(0))
model.hybridize()
input = np.zeros((2, 2), ctx=mx.cpu(0))
res = model(input)
"""
timer = timeit.Timer(setup=setup,
                     stmt='res = model(input)')
num_repeat = 1000
print(min(timer.repeat(repeat=10, number=num_repeat)) / num_repeat)
