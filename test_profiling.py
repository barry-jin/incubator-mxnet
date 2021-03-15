from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')

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
        x = self.relu(self.dense0(x))
        return self.relu(self.dense1(x))

    def relu(self, X):
        return np.maximum(X, 0)

model = Model()
model.initialize(ctx=mx.cpu(0))
model.hybridize()

profiler.set_state('run')

res = model(mx.np.zeros((2, 2), ctx=mx.cpu(0)))

profiler.set_state('stop')
print(profiler.dumps())
